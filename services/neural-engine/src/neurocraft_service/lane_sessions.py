"""Authenticated neural-session handovers inside one continuing Minecraft world.

Controllers are constructed and verified by the caller, never by an incoming Hello.
Only one session owns them at a time. Each accepted attempt consumes a global epoch,
including attempts that fail while retiring, resetting or warming a controller.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

from neurocraft_protocol import (
    MAX_FRAME_BYTES,
    WEBSOCKET_SUBPROTOCOL,
    ActuationReceipt,
    AuthenticationError,
    Command,
    FaultCode,
    Hello,
    IdentityMismatchError,
    PeerRole,
    ProgramReceipt,
    ProtocolError,
    RunIdentity,
    RunMode,
    SessionValidator,
    parse_message,
)
from websockets.asyncio.server import ServerConnection
from websockets.asyncio.server import serve as websocket_serve
from websockets.exceptions import ConnectionClosed
from websockets.typing import Subprotocol

from .controller import ControlHandler, Controller
from .server import ActionTelemetry, NeuralService, ServiceConfig

LOGGER = logging.getLogger("neurocraft_service.lanes")
LaneName = Literal["original", "shuffled_weight", "no_input"]
_MODES = {
    "original": RunMode.FAITHFUL,
    "shuffled_weight": RunMode.SHUFFLED_WEIGHT,
    "no_input": RunMode.NO_INPUT,
}


class SessionController(Controller, Protocol):
    @property
    def identity(self) -> RunIdentity: ...

    @property
    def run_mode(self) -> RunMode: ...

    def reset_session(self, *, session_epoch: int) -> None: ...

    def warm_compute_path(self, *, passes: int = 3) -> float: ...


@dataclass(frozen=True, slots=True)
class LaneSpec:
    """One pristine preconstructed controller and its immutable initial identity."""

    name: LaneName
    controller: SessionController
    identity_template: RunIdentity = field(init=False)

    def __post_init__(self) -> None:
        if self.name not in _MODES or self.controller.run_mode != _MODES[self.name]:
            raise ValueError("lane name must match the controller's actual run mode")
        if not isinstance(self.controller.identity, RunIdentity):
            raise ValueError("lane requires a validated full RunIdentity")
        object.__setattr__(self, "identity_template", self.controller.identity)


@dataclass(frozen=True, slots=True)
class SessionHooks:
    """Fresh per-epoch sinks. close runs once after all session worker jobs finish."""

    receipt_sink: Callable[[ActuationReceipt | ProgramReceipt], None] | None = None
    issued_action_sink: Callable[[Command], None] | None = None
    action_telemetry_sink: Callable[[ActionTelemetry], None] | None = None
    completed_step_observer: Callable[[Command], None] | None = None
    control_handler: ControlHandler | None = None
    close: Callable[[], None] | None = None


@dataclass(frozen=True, slots=True)
class _ActiveSession:
    lane: LaneSpec
    identity: RunIdentity
    connection: ServerConnection
    task: asyncio.Task[None]
    diagnostic: Callable[[str], None]


async def _join(task: asyncio.Task[Any]) -> Any:
    """Retain ownership through repeated cancellation, then propagate cancellation."""
    interrupted = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            interrupted = True
        except Exception:
            break
    if interrupted:
        # Consume a possible worker failure before propagating the caller's cancellation.
        with suppress(Exception, asyncio.CancelledError):
            task.result()
        raise asyncio.CancelledError
    return task.result()


class LaneSessionDispatcher:
    """One listener, three allowed identities, one active session, monotonic epochs.

    Supply newly constructed controllers at config.identity.session_epoch. No neural
    state is inherited across epochs. Initial templates may use large integer epochs;
    no conversion to floating point occurs. The factory must clean up its own partial
    allocations if it raises before returning SessionHooks.
    """

    def __init__(
        self,
        config: ServiceConfig,
        lanes: Sequence[LaneSpec],
        *,
        hooks_factory: Callable[[LaneSpec, RunIdentity], SessionHooks] | None = None,
    ) -> None:
        self.config = config
        self.lanes = tuple(lanes)
        if len(self.lanes) != 3 or {lane.name for lane in self.lanes} != set(_MODES):
            raise ValueError("exactly original, shuffled_weight and no_input lanes are required")
        if len({id(lane.controller) for lane in self.lanes}) != 3:
            raise ValueError("each lane must own a distinct controller")
        originals = [lane for lane in self.lanes if lane.name == "original"]
        if originals[0].identity_template != config.identity:
            raise ValueError("initial service identity must exactly match the original controller")
        # S2 controls change dynamics only; the stimulus/decoder/world boundary stays matched.
        for lane in self.lanes:
            template = lane.identity_template
            if replace(template, dynamics_hash=config.identity.dynamics_hash) != config.identity:
                raise ValueError("lane templates must share epoch and every non-dynamics identity")
        if len({lane.identity_template.dynamics_hash for lane in self.lanes}) != 3:
            raise ValueError("each lane must have a distinct executed dynamics identity")
        self._hooks_factory = hooks_factory or (lambda _lane, _identity: SessionHooks())
        self._handover_lock = asyncio.Lock()
        self._active: _ActiveSession | None = None
        self._last_claimed_epoch = config.identity.session_epoch - 1
        self._closed = False

    @property
    def last_claimed_epoch(self) -> int:
        return self._last_claimed_epoch

    @property
    def active_identity(self) -> RunIdentity | None:
        return None if self._active is None or self._active.task.done() else self._active.identity

    @property
    def active_lane(self) -> LaneName | None:
        return None if self._active is None or self._active.task.done() else self._active.lane.name

    def _select(self, hello: Hello) -> LaneSpec:
        if not hello.verify(self.config.token):
            raise AuthenticationError("hello authentication proof rejected")
        if hello.role is not PeerRole.MINECRAFT_SERVER:
            raise AuthenticationError("only the Minecraft server may establish a lane")
        for lane in self.lanes:
            expected = replace(lane.identity_template, session_epoch=hello.identity.session_epoch)
            if expected == hello.identity:
                # Keep the role/full-identity protocol gate in addition to the allowlist.
                SessionValidator(expected).accept_hello(
                    hello, expected_role=PeerRole.MINECRAFT_SERVER
                )
                return lane
        raise IdentityMismatchError("hello does not match a preconstructed lane identity")

    async def _retire_active(self) -> None:
        active = self._active
        if active is None:
            return
        try:
            await self._retire(active)
        finally:
            if active.task.done() and self._active is active:
                self._active = None

    @staticmethod
    async def _retire(active: _ActiveSession) -> None:
        async def close_and_join() -> None:
            try:
                active.diagnostic("old_socket_close_begin")
                await active.connection.close(code=1000, reason="neural session retired")
                active.diagnostic("old_socket_close_complete")
            finally:
                active.diagnostic("old_session_join_begin")
                await _join(active.task)
                active.diagnostic("old_session_join_complete")

        # Cancellation must not release the handover lock while old workers still run.
        await _join(asyncio.create_task(close_and_join()))

    async def _run_session(
        self,
        service: NeuralService,
        connection: ServerConnection,
        hello: Hello,
        hooks: SessionHooks,
        diagnostic: Callable[[str], None],
    ) -> None:
        try:
            await service.handle(connection, prefetched_hello=hello)
        finally:
            diagnostic("service_finished_after_worker_drain")
            if hooks.close is not None:
                diagnostic("hooks_close_begin")
                await _join(asyncio.create_task(asyncio.to_thread(hooks.close)))
                diagnostic("hooks_close_complete")

    async def handle(self, connection: ServerConnection) -> None:
        """Authenticate before touching the current owner; run the session outside the lock."""
        owned: _ActiveSession | None = None
        fault_identity = self.config.identity
        started_ns = time.monotonic_ns()
        authenticated = False

        def diagnostic(
            stage: str, error: BaseException | None = None, close_code: int | None = None
        ) -> None:
            # Fixed stage labels only; no Hello, HMAC, token, exception message or peer text.
            # Elapsed time is local to this handler, not neural or action latency.
            with suppress(Exception):
                LOGGER.warning(
                    "lane_lifecycle epoch=%d stage=%s handler_elapsed_ms=%.3f "
                    "error_class=%s close_code=%s",
                    fault_identity.session_epoch,
                    stage,
                    (time.monotonic_ns() - started_ns) / 1_000_000,
                    "none" if error is None else type(error).__name__[:80],
                    close_code,
                )

        try:
            if connection.subprotocol != WEBSOCKET_SUBPROTOCOL:
                raise ProtocolError("required WebSocket subprotocol was not negotiated")
            raw = await asyncio.wait_for(connection.recv(), self.config.handshake_timeout_seconds)
            hello = parse_message(raw)
            if not isinstance(hello, Hello):
                raise AuthenticationError("first frame must authenticate with hello")
            lane = self._select(hello)
            fault_identity = hello.identity
            authenticated = True
            diagnostic("hello_authenticated")
            diagnostic("handover_lock_wait")
            async with self._handover_lock:
                diagnostic("handover_lock_acquired")
                if self._closed or hello.identity.session_epoch <= self._last_claimed_epoch:
                    raise IdentityMismatchError(
                        "lane epoch was already consumed or dispatcher closed"
                    )
                # No await between check and claim. A valid failed attempt stays consumed.
                self._last_claimed_epoch = hello.identity.session_epoch
                diagnostic("previous_owner_retire_begin")
                await self._retire_active()
                diagnostic("previous_owner_retire_complete")

                hooks: SessionHooks | None = None
                try:

                    def prepare() -> None:
                        nonlocal hooks
                        if hello.identity.session_epoch > lane.controller.identity.session_epoch:
                            diagnostic("reset_begin")
                            lane.controller.reset_session(
                                session_epoch=hello.identity.session_epoch
                            )
                            diagnostic("reset_complete")
                        else:
                            diagnostic("reset_not_required")
                        if lane.controller.identity != hello.identity:
                            raise IdentityMismatchError("controller did not bind requested epoch")
                        diagnostic("warm_begin")
                        lane.controller.warm_compute_path()
                        diagnostic("warm_complete")
                        diagnostic("hooks_create_begin")
                        hooks = self._hooks_factory(lane, hello.identity)
                        diagnostic("hooks_create_complete")

                    await _join(asyncio.create_task(asyncio.to_thread(prepare)))
                    assert hooks is not None
                    if self._closed:
                        raise IdentityMismatchError("dispatcher closed during lane preparation")
                    service = NeuralService(
                        replace(self.config, identity=hello.identity),
                        lane.controller,
                        receipt_sink=hooks.receipt_sink,
                        issued_action_sink=hooks.issued_action_sink,
                        action_telemetry_sink=hooks.action_telemetry_sink,
                        completed_step_observer=hooks.completed_step_observer,
                        control_handler=hooks.control_handler,
                    )
                    diagnostic("service_task_start")
                    task = asyncio.create_task(
                        self._run_session(service, connection, hello, hooks, diagnostic)
                    )
                    owned = _ActiveSession(lane, hello.identity, connection, task, diagnostic)
                    self._active = owned
                finally:
                    if owned is None and hooks is not None and hooks.close is not None:
                        diagnostic("unstarted_hooks_close_begin")
                        await _join(asyncio.create_task(asyncio.to_thread(hooks.close)))
                        diagnostic("unstarted_hooks_close_complete")
                LOGGER.info(
                    "neural lane %s epoch=%d dynamics=%s",
                    lane.name,
                    hello.identity.session_epoch,
                    hello.identity.dynamics_hash,
                )
            await asyncio.shield(owned.task)
        except ConnectionClosed as exc:
            if authenticated:
                diagnostic("connection_closed", exc, None if exc.rcvd is None else exc.rcvd.code)
        except (TimeoutError, ProtocolError) as exc:
            if authenticated:
                diagnostic("refused", exc)
            code = (
                FaultCode.UNAUTHORIZED
                if isinstance(exc, AuthenticationError | TimeoutError)
                else FaultCode.IDENTITY_MISMATCH
            )
            await self._reject(connection, fault_identity, code)
        except Exception as exc:
            if authenticated:
                diagnostic("failed", exc)
            LOGGER.exception("neural lane session or handover failed; connection retired")
            await self._reject(connection, fault_identity, FaultCode.INTERNAL)
        finally:
            if owned is not None:
                try:
                    if not owned.task.done():
                        await self._retire(owned)
                finally:
                    if owned.task.done() and self._active is owned:
                        self._active = None

    async def _reject(
        self, connection: ServerConnection, identity: RunIdentity, code: FaultCode
    ) -> None:
        service = NeuralService(replace(self.config, identity=identity))
        await service._send_fault(connection, code=code, game_tick=0, observation_seq=None)
        with suppress(ConnectionClosed):
            await connection.close(code=1008, reason=f"lane refused: {code.value}")

    async def close(self) -> None:
        """Stop accepting handovers and join the current session and its workers."""

        async def stop() -> None:
            self._closed = True
            async with self._handover_lock:
                await self._retire_active()

        # Ownership begins before acquiring the lock: startup may hold it while
        # preparing a controller or its sinks in a still-running worker.
        await _join(asyncio.create_task(stop()))


def create_lane_server(dispatcher: LaneSessionDispatcher) -> Any:
    """The same loopback/subprotocol/frame bounds as the single-session server."""
    config = dispatcher.config
    return websocket_serve(
        dispatcher.handle,
        config.host,
        config.port,
        subprotocols=[Subprotocol(WEBSOCKET_SUBPROTOCOL)],
        compression=None,
        origins=[None],
        server_header=None,
        open_timeout=config.handshake_timeout_seconds,
        ping_interval=config.ping_interval_seconds,
        ping_timeout=config.ping_timeout_seconds,
        close_timeout=2,
        max_size=MAX_FRAME_BYTES,
        max_queue=1,
        write_limit=MAX_FRAME_BYTES,
    )
