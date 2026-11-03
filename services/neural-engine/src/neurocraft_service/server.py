"""Loopback-only WebSocket service with authenticated fail-closed sessions."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import logging
import secrets
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ParamSpec, TypeVar

from neurocraft_protocol import (
    ACTION_TARGET_OFFSET_TICKS,
    MAX_FRAME_BYTES,
    WEBSOCKET_SUBPROTOCOL,
    Action,
    ActuationReceipt,
    AuthenticationError,
    Command,
    ControlAck,
    ControlCommand,
    ControlOperation,
    ControlState,
    Fault,
    FaultCode,
    Hello,
    HelloAck,
    IdentityMismatchError,
    MotorVector,
    Observation,
    PeerRole,
    ProgramAction,
    ProgramReceipt,
    ProtocolError,
    RunIdentity,
    SequenceError,
    SessionValidator,
    canonical_json_bytes,
    encode_message,
    parse_message,
    require_sensor_contract,
    validate_run_token,
)
from websockets.asyncio.server import ServerConnection
from websockets.asyncio.server import serve as websocket_serve
from websockets.exceptions import ConnectionClosed
from websockets.typing import Subprotocol

from .controller import (
    ControlHandler,
    Controller,
    DeterministicNeutralController,
    ResettableController,
)
from .queues import BoundedCommandQueue, LatestValueQueue
from .session_telemetry import (
    CommandKind,
    PausedWindow,
    SessionSummary,
    SessionTelemetry,
    paused_window_message,
)

LOGGER = logging.getLogger("neurocraft_service")
_P = ParamSpec("_P")
_T = TypeVar("_T")

_SAFE_MESSAGES = {
    "MALFORMED": "malformed protocol frame",
    "OVERSIZE": "protocol frame exceeds the byte limit",
    "UNKNOWN_FIELD": "protocol frame contains an unknown field",
    "UNAUTHORIZED": "authentication failed",
    "IDENTITY_MISMATCH": "session or artifact identity mismatch",
    "SEQUENCE": "sequence or tick order rejected",
    "STALE": "stale frame rejected",
    "FUTURE": "future frame rejected",
    "QUEUE_OVERFLOW": "bounded command queue overflow",
    "TRACE_FAILURE": "mandatory trace failed",
    "INTERNAL": "internal service fault",
    "DISCONNECTED": "peer disconnected",
    "CONTROL_UNAVAILABLE": "requested control capability is unavailable",
    "INVALID_STATE": "control operation is invalid in the current state",
    "RECEIPT_MISMATCH": "actuation receipt does not bind the pending action",
}


def create_run_token() -> str:
    """Return a high-entropy URL-safe token. Callers must deliver it out of band."""

    return secrets.token_urlsafe(32)


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    identity: RunIdentity
    token: str = field(repr=False)
    host: str = "127.0.0.1"
    port: int = 8765
    handshake_timeout_seconds: float = 3.0
    ping_interval_seconds: float = 10.0
    ping_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RunIdentity):
            raise ProtocolError("service identity must be RunIdentity")
        try:
            address = ipaddress.ip_address(self.host)
        except ValueError as exc:
            raise ProtocolError("service host must be an IP literal") from exc
        if not address.is_loopback:
            raise ProtocolError("service host must be a loopback address")
        if (
            isinstance(self.port, bool)
            or not isinstance(self.port, int)
            or not 0 <= self.port <= 65535
        ):
            raise ProtocolError("service port must be between 0 and 65535")
        if not 0.1 <= self.handshake_timeout_seconds <= 60:
            raise ProtocolError("handshake timeout must be between 0.1 and 60 seconds")
        if not 0.1 <= self.ping_interval_seconds <= 300:
            raise ProtocolError("ping interval must be between 0.1 and 300 seconds")
        if not 0.1 <= self.ping_timeout_seconds <= 300:
            raise ProtocolError("ping timeout must be between 0.1 and 300 seconds")
        # Reuse the exact protocol token contract without retaining another copy. The
        # token is only ever an HMAC key: it never reaches the wire in either direction.
        validate_run_token(self.token)


@dataclass(frozen=True, slots=True)
class ActionTelemetry:
    """Measured service-side telemetry bound to one exact serialized command frame.

    The frame is an ``action`` or a ``program_action``; the hash is over that frame's canonical
    bytes either way, so the telemetry names exactly one issued command.
    """

    command_seq: int
    action_sha256: str
    queue_coalesced: int
    queue_overruns: int
    compute_ns: int
    serialization_ns: int
    write_ns: int

    def __post_init__(self) -> None:
        for name in (
            "command_seq",
            "queue_coalesced",
            "queue_overruns",
            "compute_ns",
            "serialization_ns",
            "write_ns",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ProtocolError(f"action telemetry {name} must be a non-negative integer")
        if len(self.action_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.action_sha256
        ):
            raise ProtocolError("action telemetry hash must be lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class _IssuedActionMetrics:
    action_sha256: str
    queue_coalesced: int
    queue_overruns: int
    compute_ns: int


@dataclass(frozen=True, slots=True)
class _PendingControlEffect:
    """Accepted control awaiting its first causally governed Action."""

    command: ControlCommand
    state: ControlState


_ACTION_EFFECT_OPERATIONS = {
    ControlOperation.PAUSE,
    ControlOperation.RESUME,
    ControlOperation.SINGLE_STEP,
    ControlOperation.RESET,
    ControlOperation.REPLAY_START,
    ControlOperation.REPLAY_STOP,
    ControlOperation.LESION_APPLY,
    ControlOperation.LESION_CLEAR,
    ControlOperation.STIMULATION_APPLY,
    ControlOperation.STIMULATION_CLEAR,
}


def _fault_code(exc: BaseException) -> FaultCode:
    raw = getattr(exc, "code", FaultCode.INTERNAL.value)
    try:
        return FaultCode(raw)
    except ValueError:
        return FaultCode.INTERNAL


class _FrameLane(StrEnum):
    """Which driving command kind this session has latched (design section 2.3)."""

    UNDECIDED = "UNDECIDED"
    MOTOR = "MOTOR"
    PROGRAM = "PROGRAM"


def _latched_lane(action: Command, lane: _FrameLane) -> _FrameLane:
    """One driving frame kind per session; refuse a controller that changes its mind.

    A session is driven by raw ``action`` frames or by ``program_action`` frames, never both:
    the two carry different receipt semantics and different body authority, and a mixed session
    would leave no honest answer to "what was driving the fly at tick T". The service side is
    structural — a controller profile returns one kind for the whole session — and this is the
    check that says so out loud rather than trusting it, mirroring the authority's own latch.

    **Neutral ``action`` frames are lane-agnostic**, exactly as the authority treats them: the
    paused, replaying, and faulted service emits them on both lanes, and they command nothing.
    So they neither decide the lane nor violate it.
    """

    if isinstance(action, ProgramAction):
        kind = _FrameLane.PROGRAM
    elif action.neutral:
        return lane
    else:
        kind = _FrameLane.MOTOR
    if lane is not _FrameLane.UNDECIDED and lane is not kind:
        raise _coded_error(
            FaultCode.MALFORMED,
            f"controller changed driving frame kind: {kind.value} frame in a {lane.value} session",
        )
    return kind


def _validate_generated_action(
    action: Command, observation: Observation, command_seq: int, *, lane: _FrameLane
) -> _FrameLane:
    """Refuse any controller command that does not bind the observation it was computed from.

    Every rule is envelope-only, so it holds identically for an ``action`` and a
    ``program_action``. Returns the session's latched driving lane, which the caller carries
    forward; see :func:`_latched_lane`.
    """

    if action.identity != observation.identity:
        raise IdentityMismatchError("controller returned the wrong identity")
    if action.command_seq != command_seq:
        raise SequenceError("controller returned the wrong command sequence")
    if (
        action.observation_seq != observation.observation_seq
        or action.observation_tick != observation.game_tick
    ):
        raise SequenceError("controller action does not bind the processed observation")
    if action.target_tick != observation.game_tick + ACTION_TARGET_OFFSET_TICKS:
        raise SequenceError(
            "controller action must target the observation tick plus the declared action budget"
        )
    return _latched_lane(action, lane)


def _neutral_action(
    observation: Observation,
    *,
    command_seq: int,
    neural_cursor: int,
    reason: str,
) -> Action:
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "kind": "service_neutral_v1",
                "reason": reason,
                "observation_seq": observation.observation_seq,
                "game_tick": observation.game_tick,
                "neural_cursor": neural_cursor,
            }
        )
    ).hexdigest()
    return Action(
        identity=observation.identity,
        command_seq=command_seq,
        observation_seq=observation.observation_seq,
        observation_tick=observation.game_tick,
        target_tick=observation.game_tick + ACTION_TARGET_OFFSET_TICKS,
        valid_through_tick=observation.game_tick + ACTION_TARGET_OFFSET_TICKS,
        neural_step_start=neural_cursor,
        neural_step_end=neural_cursor,
        state_digest=digest,
        motors=MotorVector.neutral(),
        neutral=True,
    )


def _coded_error(code: FaultCode, message: str) -> ProtocolError:
    error = ProtocolError(message)
    error.code = code.value
    return error


def _log_paused_windows(windows: tuple[PausedWindow, ...]) -> None:
    for window in windows:
        LOGGER.info(paused_window_message(window))


def _summary_caveats(summary: SessionSummary) -> str:
    """The suffix that says when the two attributed figures do not cover the validator's count."""

    caveats: list[str] = []
    if summary.unattributed_retirements:
        caveats.append(
            f"{summary.unattributed_retirements} retirement(s) unattributed "
            "(neither lateness nor paused-window)"
        )
    if summary.ledger_desyncs:
        caveats.append(
            f"{summary.ledger_desyncs} ledger desync(s): attribution above is unreliable"
        )
    return "".join(f"; {caveat}" for caveat in caveats)


class NeuralService:
    """Connection handler suitable for `websockets.asyncio.server.serve`."""

    def __init__(
        self,
        config: ServiceConfig,
        controller: Controller | None = None,
        *,
        receipt_sink: Callable[[ActuationReceipt | ProgramReceipt], None] | None = None,
        issued_action_sink: Callable[[Command], None] | None = None,
        action_telemetry_sink: Callable[[ActionTelemetry], None] | None = None,
        completed_step_observer: Callable[[Command], None] | None = None,
        control_handler: ControlHandler | None = None,
    ) -> None:
        self.config = config
        self.controller = controller or DeterministicNeutralController()
        self.receipt_sink = receipt_sink
        self.issued_action_sink = issued_action_sink
        self.action_telemetry_sink = action_telemetry_sink
        self.completed_step_observer = completed_step_observer
        self.display_observer_failures = 0
        self.control_handler = control_handler
        self.control_state = ControlState.RUNNING
        self._return_state = ControlState.RUNNING
        self._session_epoch_claimed = False
        self._worker_jobs: set[asyncio.Task[Any]] = set()
        #: Telemetry of the last authenticated session to end; ``None`` until one has.
        self.last_session_summary: SessionSummary | None = None

    async def _run_worker(
        self, function: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs
    ) -> _T:
        """Cancellation retires the awaiter, never the still-running thread's ownership."""
        job = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        self._worker_jobs.add(job)

        def finished(task: asyncio.Task[_T]) -> None:
            self._worker_jobs.discard(task)
            if not task.cancelled():
                # Retrieve failures even when the original awaiter was cancelled.
                task.exception()

        job.add_done_callback(finished)
        return await asyncio.shield(job)

    async def drain_workers(self) -> None:
        """Join all owned thread jobs before another session may reset a controller."""
        interrupted = False
        while self._worker_jobs:
            joined = asyncio.gather(*tuple(self._worker_jobs), return_exceptions=True)
            while not joined.done():
                try:
                    await asyncio.shield(joined)
                except asyncio.CancelledError:
                    interrupted = True
        if interrupted:
            raise asyncio.CancelledError

    def _step_with_display(self, observation: Observation, *, command_seq: int) -> Command:
        """Capture optional display data in the same serialized worker as the step.

        The caller holds execution_fence. Unlike mandatory receipt telemetry, a display
        failure disables only the observer and returns the original command unchanged.
        Display capture time remains included in the measured service compute time.
        """
        action = self.controller.step(observation, command_seq=command_seq)
        if self.completed_step_observer is not None:
            try:
                self.completed_step_observer(action)
            except Exception:
                self.display_observer_failures += 1
                self.completed_step_observer = None
                LOGGER.warning("optional brain display observer failed and was disabled")
        return action

    def _control_ack(
        self,
        command: ControlCommand,
        *,
        accepted: bool,
        fault_code: FaultCode | None,
        effective_tick: int | None = None,
        state: ControlState | None = None,
    ) -> ControlAck:
        resolved_effective_tick = effective_tick
        if resolved_effective_tick is None:
            resolved_effective_tick = command.issued_tick
            # Accepted RESET acknowledgements are materialized against a later Action.  The
            # provisional internal value must already satisfy the wire model even though this
            # object isn't queued; process_latest replaces it with the governed target tick.
            if accepted and command.operation is ControlOperation.RESET:
                resolved_effective_tick += ACTION_TARGET_OFFSET_TICKS
        return ControlAck(
            identity=command.identity,
            control_seq=command.control_seq,
            issued_tick=command.issued_tick,
            operation=command.operation,
            content_id=command.content_id,
            content_hash=command.content_hash,
            accepted=accepted,
            state=self.control_state if state is None else state,
            effective_tick=resolved_effective_tick,
            fault_code=fault_code,
        )

    async def _apply_control(self, command: ControlCommand) -> ControlAck:
        operation = command.operation
        execution_operations = {
            ControlOperation.PAUSE,
            ControlOperation.RESUME,
            ControlOperation.SINGLE_STEP,
            ControlOperation.RESET,
            ControlOperation.REPLAY_START,
            ControlOperation.REPLAY_STOP,
        }
        if self.control_state is ControlState.RECORDING and operation in execution_operations:
            return self._control_ack(command, accepted=False, fault_code=FaultCode.INVALID_STATE)
        if operation is ControlOperation.PAUSE:
            if self.control_state is not ControlState.RUNNING:
                return self._control_ack(
                    command, accepted=False, fault_code=FaultCode.INVALID_STATE
                )
            self.control_state = ControlState.PAUSED
            return self._control_ack(command, accepted=True, fault_code=None)
        if operation is ControlOperation.RESUME:
            if self.control_state is not ControlState.PAUSED:
                self.control_state = ControlState.PAUSED
                return self._control_ack(
                    command, accepted=False, fault_code=FaultCode.INVALID_STATE
                )
            self.control_state = ControlState.RUNNING
            return self._control_ack(command, accepted=True, fault_code=None)
        if operation is ControlOperation.SINGLE_STEP:
            if self.control_state is not ControlState.PAUSED:
                self.control_state = ControlState.PAUSED
                return self._control_ack(
                    command, accepted=False, fault_code=FaultCode.INVALID_STATE
                )
            self.control_state = ControlState.STEP_ARMED
            return self._control_ack(command, accepted=True, fault_code=None)
        if operation is ControlOperation.RESET:
            if self.control_state not in {ControlState.RUNNING, ControlState.PAUSED}:
                return self._control_ack(
                    command, accepted=False, fault_code=FaultCode.INVALID_STATE
                )
            self.control_state = ControlState.PAUSED
            if not isinstance(self.controller, ResettableController):
                return self._control_ack(
                    command, accepted=False, fault_code=FaultCode.CONTROL_UNAVAILABLE
                )
            self.controller.reset()
            return self._control_ack(command, accepted=True, fault_code=None)

        intervention_operations = {
            ControlOperation.LESION_APPLY,
            ControlOperation.LESION_CLEAR,
            ControlOperation.STIMULATION_APPLY,
            ControlOperation.STIMULATION_CLEAR,
        }
        if operation in intervention_operations:
            unchanged_state = self.control_state
            if unchanged_state not in {
                ControlState.RUNNING,
                ControlState.PAUSED,
                ControlState.RECORDING,
            }:
                return self._control_ack(
                    command, accepted=False, fault_code=FaultCode.INVALID_STATE
                )
            if self.control_handler is None:
                return self._control_ack(
                    command, accepted=False, fault_code=FaultCode.CONTROL_UNAVAILABLE
                )
            try:
                await self._run_worker(self.control_handler.handle_control, command)
            except Exception as exc:
                code = _fault_code(exc)
                if code in {FaultCode.CONTROL_UNAVAILABLE, FaultCode.INVALID_STATE}:
                    self.control_state = unchanged_state
                    return self._control_ack(command, accepted=False, fault_code=code)
                self.control_state = ControlState.FAULTED
                return self._control_ack(command, accepted=False, fault_code=FaultCode.INTERNAL)
            self.control_state = unchanged_state
            return self._control_ack(command, accepted=True, fault_code=None)

        if self.control_handler is None:
            self.control_state = ControlState.PAUSED
            return self._control_ack(
                command, accepted=False, fault_code=FaultCode.CONTROL_UNAVAILABLE
            )

        allowed = False
        next_state = self.control_state
        if operation is ControlOperation.RECORD_START:
            allowed = self.control_state is ControlState.RUNNING
            self._return_state = self.control_state
            next_state = ControlState.RECORDING
        elif operation is ControlOperation.RECORD_STOP:
            allowed = self.control_state is ControlState.RECORDING
            next_state = self._return_state
        elif operation is ControlOperation.REPLAY_START:
            allowed = self.control_state is ControlState.PAUSED
            self._return_state = ControlState.PAUSED
            next_state = ControlState.REPLAYING
        elif operation is ControlOperation.REPLAY_STOP:
            allowed = self.control_state is ControlState.REPLAYING
            next_state = ControlState.PAUSED
        if not allowed:
            self.control_state = ControlState.PAUSED
            return self._control_ack(command, accepted=False, fault_code=FaultCode.INVALID_STATE)
        try:
            await self._run_worker(self.control_handler.handle_control, command)
        except Exception as exc:
            code = _fault_code(exc)
            if code in {FaultCode.CONTROL_UNAVAILABLE, FaultCode.INVALID_STATE}:
                self.control_state = ControlState.PAUSED
                return self._control_ack(command, accepted=False, fault_code=code)
            self.control_state = ControlState.FAULTED
            return self._control_ack(command, accepted=False, fault_code=FaultCode.INTERNAL)
        self.control_state = next_state
        return self._control_ack(command, accepted=True, fault_code=None)

    async def _send_fault(
        self,
        connection: ServerConnection,
        *,
        code: FaultCode,
        game_tick: int,
        observation_seq: int | None,
    ) -> None:
        fault = Fault(
            identity=self.config.identity,
            code=code,
            message=_SAFE_MESSAGES[code.value],
            game_tick=max(0, game_tick),
            related_observation_seq=observation_seq,
            neutral_applied=True,
        )
        with suppress(ConnectionClosed):
            await connection.send(encode_message(fault))

    async def _authenticated_session(
        self,
        connection: ServerConnection,
        validator: SessionValidator,
    ) -> None:
        latest: LatestValueQueue[Observation] = LatestValueQueue()
        outbound: BoundedCommandQueue[Command | ControlAck] = BoundedCommandQueue()
        execution_fence = asyncio.Lock()
        command_counter = 0
        neural_cursor = 0
        coalesced_cursor = 0
        bridge_coalesced_cursor = 0
        # Newest observation tick the validator has accepted; process_latest
        # compares freshly computed actions against it so a frame the world
        # outran during compute is logged and counted as late, not silently
        # indistinguishable from an on-time one. It is sent either way.
        newest_game_tick = -1
        # Which commands are in the validator's window and why, so a retirement can be
        # attributed: lateness, or a paused-window neutral action consumed by a body program
        # (R-29). Read-only with respect to the wire; see session_telemetry.py.
        telemetry = SessionTelemetry()
        # One driving frame kind per session, latched on the first non-neutral command.
        driving_lane = _FrameLane.UNDECIDED
        pending_action_metrics: dict[int, _IssuedActionMetrics] = {}
        pending_control_effects: list[_PendingControlEffect] = []

        def queue_outbound(message: Command | ControlAck) -> None:
            if not outbound.put(message):
                raise _coded_error(
                    FaultCode.QUEUE_OVERFLOW, "bounded outbound command queue overflow"
                )

        async def receive_frames() -> None:
            nonlocal newest_game_tick
            async for raw in connection:
                message = parse_message(raw)
                if isinstance(message, Observation):
                    require_sensor_contract(
                        message.sensors, self.config.identity.observation_schema_hash
                    )
                    retired_before = validator.retired_action_count
                    validator.accept_observation(message, current_tick=message.game_tick)
                    newest_game_tick = max(newest_game_tick, message.game_tick)
                    if validator.retired_action_count != retired_before:
                        release = telemetry.observation_released(
                            validator.retired_action_count - retired_before
                        )
                        if release.late:
                            # The authority never receipted these, so it refused them.
                            # Recovering silently would hide a peer missing its deadline.
                            LOGGER.warning(
                                "retired %d validated command(s) the authority can no longer "
                                "apply at game tick %d; %d retired this session",
                                release.late,
                                message.game_tick,
                                telemetry.lateness_retired,
                            )
                        # A neutral action issued while PAUSED that a body program consumed
                        # unapplied is not lateness (R-29): counted, and summarised once per
                        # paused window when its last neutral action has left the window.
                        _log_paused_windows(release.completed_windows)
                    latest.put_latest(message)
                elif isinstance(message, ActuationReceipt | ProgramReceipt):
                    retired_before_receipt = validator.retired_action_count
                    validator.accept_receipt(message)
                    release = telemetry.receipt_released(
                        message.command_seq,
                        superseded=validator.retired_action_count - retired_before_receipt,
                    )
                    if release.late:
                        LOGGER.warning(
                            "receipt for command %d superseded %d unreceipted "
                            "command(s) the authority consumed unapplied; "
                            "%d retired this session",
                            message.command_seq,
                            release.late,
                            telemetry.lateness_retired,
                        )
                    _log_paused_windows(release.completed_windows)
                    if self.receipt_sink is not None:
                        try:
                            await self._run_worker(self.receipt_sink, message)
                        except Exception as exc:
                            LOGGER.exception(
                                "actuation receipt sink failed after protocol validation"
                            )
                            raise _coded_error(
                                FaultCode.TRACE_FAILURE, "actuation receipt sink failed"
                            ) from exc
                elif isinstance(message, ControlCommand):
                    if message.operation in _ACTION_EFFECT_OPERATIONS and pending_control_effects:
                        raise SequenceError(
                            "only one action-effect control may await causal materialization"
                        )
                    validator.accept_control(message, current_tick=message.issued_tick)
                    # Controller mutations and their acknowledgements share one
                    # ordered fence with Action generation. If an Action began
                    # first, it is queued before the acknowledgement; otherwise
                    # the control effect is visible to every later Action.
                    async with execution_fence:
                        acknowledgement = await self._apply_control(message)
                        if (
                            acknowledgement.accepted
                            and message.operation is ControlOperation.RESUME
                        ):
                            # The paused window closes here. Its one summary line is written
                            # now if every neutral action it issued has already left the
                            # validated window, otherwise by the release that empties it, so
                            # the count it reports is exact rather than short by the tail.
                            completed = telemetry.resume_accepted()
                            if completed is not None:
                                _log_paused_windows((completed,))
                        if (
                            acknowledgement.accepted
                            and message.operation in _ACTION_EFFECT_OPERATIONS
                        ):
                            pending_control_effects.append(
                                _PendingControlEffect(message, acknowledgement.state)
                            )
                        else:
                            validator.accept_control_ack(acknowledgement)
                            queue_outbound(acknowledgement)
                else:
                    raise ProtocolError(
                        "authenticated peer may send observation, receipt, or control frames"
                    )

        async def process_latest() -> None:
            nonlocal bridge_coalesced_cursor, coalesced_cursor, command_counter
            nonlocal driving_lane, neural_cursor
            while True:
                observation = await latest.get()
                try:
                    coalesced_now = latest.coalesced_count
                    if observation.bridge_observations_coalesced < bridge_coalesced_cursor:
                        raise SequenceError("bridge observation coalescence counter moved backward")
                    bridge_coalesced = (
                        observation.bridge_observations_coalesced - bridge_coalesced_cursor
                    )
                    bridge_coalesced_cursor = observation.bridge_observations_coalesced
                    queue_coalesced = coalesced_now - coalesced_cursor + bridge_coalesced
                    coalesced_cursor = coalesced_now
                    async with execution_fence:
                        compute_started = time.perf_counter_ns()
                        # Declared over the union so a program-lane controller's
                        # ProgramAction and the neutral Action below share one
                        # variable; the neutral branch is unchanged.
                        action: Command
                        kind = CommandKind.DRIVING
                        if self.control_state in {
                            ControlState.PAUSED,
                            ControlState.REPLAYING,
                            ControlState.FAULTED,
                        }:
                            kind = (
                                CommandKind.PAUSED_NEUTRAL
                                if self.control_state is ControlState.PAUSED
                                else CommandKind.OTHER_NEUTRAL
                            )
                            action = _neutral_action(
                                observation,
                                command_seq=command_counter,
                                neural_cursor=neural_cursor,
                                reason=self.control_state.value.lower(),
                            )
                        else:
                            # De-blinded compute: a controller step (25-300 ms for
                            # the full profile under load) runs in a worker thread
                            # so the event loop stays live -- receive_frames keeps
                            # accepting observations (whose acceptance drives the
                            # validator's window retirement) and send_outbound
                            # keeps flushing previously queued frames. This is
                            # race-free because the validator is only ever mutated
                            # from this event loop, and the controller is only
                            # ever entered under execution_fence (held here), so
                            # no control mutation can interleave with a step. The
                            # only concurrent controller touch is the full
                            # profile's pre-first-step idle re-warm
                            # (scripts/run_connectome_service.py,
                            # _serve_with_idle_rewarm), which reads only frozen
                            # view buffers plus its own locally allocated scratch
                            # and stops once stepping begins.
                            # A signal (Ctrl-C) reaches only the main thread;
                            # shutdown retains ownership of in-flight work until
                            # it finishes, before a new session can reset the controller.
                            action = await self._run_worker(
                                self._step_with_display, observation, command_seq=command_counter
                            )
                            if self.control_state is ControlState.STEP_ARMED:
                                self.control_state = ControlState.PAUSED
                        driving_lane = _validate_generated_action(
                            action, observation, command_counter, lane=driving_lane
                        )
                        if newest_game_tick > action.target_tick:
                            # The world outran this action while it was computed. It is
                            # sent anyway, on exactly the path an on-time frame takes:
                            # the authority applies only on target_tick, so it retires
                            # this frame fail-neutral (ADR 0011, "the authority can no
                            # longer apply") instead of applying it, and this service's
                            # observation-driven retirement frees its validated-window
                            # slot on the next observation.
                            #
                            # ADR 0018's pre-send drop is superseded here. It withheld
                            # the frame and deliberately left command_counter
                            # un-advanced "so the next computed action reuses this
                            # sequence number" — but the service has no authority to
                            # reuse one. Every controller records the sequence INSIDE
                            # step(), before the service sees the action at all
                            # (ConnectomeController.controller.py:360,
                            # FullBrainController.full_brain_controller.py:2063), so the
                            # next step under the reused counter raises
                            # ControllerIntegrityError("command sequence is stale or
                            # duplicated") and the session fails closed on INTERNAL.
                            # That killed two live program-lane sessions on 2026-09-03,
                            # whose action latencies were 87-99 ms against a 50 ms tick.
                            # The drop was unsound a second way even where it did not
                            # raise: the dropped frame's neural interval had already
                            # been stepped, so the next frame's neural_step_start opened
                            # past the last neural_step_end the mod had been shown.
                            #
                            # Sending cannot wedge the validated-command window, and
                            # the ordering that guarantees it is exact. The window holds
                            # ACTION_TARGET_OFFSET_TICKS + 1 commands, which is exactly
                            # how many the authority can still apply: at observation
                            # tick T those are the commands targeting T (computed from
                            # T - 2) and T + 1 (from T - 1), plus the one being accepted
                            # now. receive_frames retires and queues an observation in
                            # one event-loop slice with no await between, and the
                            # latest-value queue means observation N can only be queued
                            # after observation N-1 was dequeued — which is after action
                            # N-2 was accepted. So the retirement pass at observation
                            # N's tick always runs after action N-3 entered the window,
                            # and always removes it (target N-3 = tick N-3 + 2 =
                            # tick N-1 < tick N, since observation ticks strictly
                            # increase). At most actions N-2 and N-1 still hold slots
                            # when action N is accepted.
                            #
                            # The residual bound is elsewhere and is declared, not
                            # discovered: SessionValidator keeps a
                            # bounded eight-entry observation ledger, matching the
                            # authority's existing sent-observation history, and
                            # eviction from it is what ends a run — accept_action is
                            # called here with current_tick = observation.game_tick, so
                            # the command's age is always 0 and only the binding lookup
                            # can refuse it, fail-CLOSED on StaleMessageError. The
                            # retained late frame is still sent and retires fail-neutral
                            # at the authority under its unchanged target/age rules.
                            # A step outrun by eight observations loses its binding and
                            # the session ends. The authority's separate starvation hold
                            # is unchanged and may end the session first.
                            LOGGER.info(
                                "sending late action seq %d computed from game tick %d; "
                                "world reached tick %d before it could be sent; "
                                "%d sent late this session",
                                action.command_seq,
                                action.observation_tick,
                                newest_game_tick,
                                telemetry.record_late_send(),
                            )
                        try:
                            validator.accept_action(action, current_tick=observation.game_tick)
                        except ProtocolError:
                            # Completed-but-refused commands never reach issued/sent telemetry.
                            # Preserve bounded timing/sequence context, without payloads or tokens.
                            LOGGER.warning(
                                "completed command refused: command_seq=%d observation_seq=%d "
                                "observation_tick=%d target_tick=%d newest_observation_seq=%d "
                                "newest_game_tick=%d newer_observations=%d compute_elapsed_ns=%d "
                                "session_epoch=%d",
                                action.command_seq,
                                action.observation_seq,
                                action.observation_tick,
                                action.target_tick,
                                validator.last_observation_seq,
                                newest_game_tick,
                                validator.last_observation_seq - action.observation_seq,
                                time.perf_counter_ns() - compute_started,
                                action.identity.session_epoch,
                            )
                            raise
                        telemetry.record_issued(action, kind=kind)
                        compute_ns = time.perf_counter_ns() - compute_started
                        if self.issued_action_sink is not None:
                            try:
                                await self._run_worker(self.issued_action_sink, action)
                            except Exception as exc:
                                LOGGER.exception(
                                    "issued action sink failed after protocol validation"
                                )
                                raise _coded_error(
                                    FaultCode.TRACE_FAILURE, "issued action sink failed"
                                ) from exc
                        acknowledgements: list[ControlAck] = []
                        for effect in pending_control_effects:
                            reset_effect = effect.command.operation is ControlOperation.RESET
                            effective_tick = (
                                action.target_tick if reset_effect else observation.game_tick
                            )
                            # A RESET is stamped with the governed action's target_tick, so
                            # its effect lands one whole action budget after the observation
                            # the effect was materialized against, and that observation is
                            # the issuing tick or the one after it. PAUSE/RESUME are stamped
                            # with the observation tick itself (D-079) and keep their bounds.
                            minimum_tick = effect.command.issued_tick + (
                                ACTION_TARGET_OFFSET_TICKS if reset_effect else 0
                            )
                            maximum_tick = effect.command.issued_tick + (
                                ACTION_TARGET_OFFSET_TICKS + 1 if reset_effect else 1
                            )
                            if not minimum_tick <= effective_tick <= maximum_tick:
                                raise SequenceError(
                                    "accepted control missed its first-action effect deadline"
                                )
                            acknowledgement = self._control_ack(
                                effect.command,
                                accepted=True,
                                fault_code=None,
                                effective_tick=effective_tick,
                                state=effect.state,
                            )
                            validator.accept_control_ack(acknowledgement)
                            acknowledgements.append(acknowledgement)
                        pending_control_effects.clear()
                        for acknowledgement in acknowledgements:
                            queue_outbound(acknowledgement)
                        queue_outbound(action)
                        pending_action_metrics[action.command_seq] = _IssuedActionMetrics(
                            action_sha256=hashlib.sha256(
                                canonical_json_bytes(action.to_dict())
                            ).hexdigest(),
                            queue_coalesced=queue_coalesced,
                            queue_overruns=outbound.overrun_count,
                            compute_ns=compute_ns,
                        )
                        neural_cursor = action.neural_step_end
                        command_counter += 1
                finally:
                    latest.task_done()

        async def send_outbound() -> None:
            while True:
                message = await outbound.get()
                try:
                    serialization_started = time.perf_counter_ns()
                    encoded = encode_message(message)
                    serialization_ns = time.perf_counter_ns() - serialization_started
                    write_started = time.perf_counter_ns()
                    await connection.send(encoded)
                    write_ns = time.perf_counter_ns() - write_started
                    if isinstance(message, Action | ProgramAction):
                        metrics = pending_action_metrics.pop(message.command_seq, None)
                        if metrics is None:
                            raise _coded_error(
                                FaultCode.TRACE_FAILURE,
                                "serialized action lacks bound operational telemetry",
                            )
                        if self.action_telemetry_sink is not None:
                            action_telemetry = ActionTelemetry(
                                command_seq=message.command_seq,
                                action_sha256=metrics.action_sha256,
                                queue_coalesced=metrics.queue_coalesced,
                                queue_overruns=metrics.queue_overruns,
                                compute_ns=metrics.compute_ns,
                                serialization_ns=serialization_ns,
                                write_ns=write_ns,
                            )
                            try:
                                await self._run_worker(self.action_telemetry_sink, action_telemetry)
                            except Exception as exc:
                                LOGGER.exception("action telemetry sink failed after command write")
                                raise _coded_error(
                                    FaultCode.TRACE_FAILURE,
                                    "action telemetry sink failed",
                                ) from exc
                finally:
                    outbound.task_done()

        tasks = {
            asyncio.create_task(receive_frames(), name="neurocraft-receive"),
            asyncio.create_task(process_latest(), name="neurocraft-process"),
            asyncio.create_task(send_outbound(), name="neurocraft-send"),
        }
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            for task in pending:
                task.cancel()
        finally:
            try:
                for task in tasks:
                    task.cancel()
                for task in tasks:
                    with suppress(asyncio.CancelledError, ConnectionClosed):
                        await task
            finally:
                # Awaiting a task that failed re-raises its error, so the summary has a finally
                # of its own: a session that fails closed still reports its counts.
                self._close_session_telemetry(telemetry, validator)

    def _close_session_telemetry(
        self, telemetry: SessionTelemetry, validator: SessionValidator
    ) -> None:
        """One summary line per session, with lateness and paused-window counts kept apart."""

        for window in telemetry.unfinished_windows():
            LOGGER.info(paused_window_message(window, session_ended=True))
        summary = telemetry.summary(window_retirements_total=validator.retired_action_count)
        self.last_session_summary = summary
        LOGGER.log(
            (
                logging.WARNING
                if summary.unattributed_retirements or summary.ledger_desyncs
                else logging.INFO
            ),
            "session summary: %d command(s) issued, %d sent late, %d retired as lateness, "
            "%d paused-window neutral action(s) unapplied across %d paused window(s) "
            "(not lateness), %d window retirement(s) in total%s",
            summary.commands_issued,
            summary.late_sends,
            summary.lateness_retired,
            summary.paused_window_unapplied,
            summary.paused_windows,
            summary.window_retirements_total,
            _summary_caveats(summary),
        )

    async def handle(
        self, connection: ServerConnection, *, prefetched_hello: Hello | None = None
    ) -> None:
        game_tick = 0
        observation_seq: int | None = None
        validator = SessionValidator(self.config.identity)
        started_ns = time.monotonic_ns()

        def diagnostic(
            stage: str, error: BaseException | None = None, close_code: int | None = None
        ) -> None:
            if prefetched_hello is not None:
                with suppress(Exception):
                    LOGGER.warning(
                        "lane_session epoch=%d stage=%s session_elapsed_ms=%.3f "
                        "error_class=%s close_code=%s",
                        self.config.identity.session_epoch,
                        stage,
                        (time.monotonic_ns() - started_ns) / 1_000_000,
                        "none" if error is None else type(error).__name__[:80],
                        close_code,
                    )

        try:
            if connection.subprotocol != WEBSOCKET_SUBPROTOCOL:
                raise ProtocolError("required WebSocket subprotocol wasn't negotiated")
            if prefetched_hello is None:
                raw = await asyncio.wait_for(
                    connection.recv(), timeout=self.config.handshake_timeout_seconds
                )
                message = parse_message(raw)
            else:
                # Internal dispatch may read once to select a lane. Authentication,
                # role, full identity and one-shot consumption are still checked here.
                message = prefetched_hello
            if not isinstance(message, Hello):
                raise AuthenticationError("first frame must authenticate with hello")
            # The peer proves possession of the run token over its own fresh nonce; the
            # token itself is never transmitted, so nothing secret is disclosed even to a
            # peer that fails this check.
            if not message.verify(self.config.token):
                raise AuthenticationError("hello authentication proof rejected")
            validator.accept_hello(message, expected_role=PeerRole.MINECRAFT_SERVER)
            if self._session_epoch_claimed:
                raise IdentityMismatchError("session epoch was already consumed")
            # No await occurs between the check and assignment, so this is atomic
            # within the asyncio event loop even under simultaneous handshakes.
            self._session_epoch_claimed = True
            diagnostic("ack_write_begin")
            await connection.send(
                encode_message(
                    HelloAck.create(
                        identity=self.config.identity,
                        role=PeerRole.NEURAL_SERVICE,
                        token=self.config.token,
                        client_nonce=message.client_nonce,
                    )
                )
            )
            diagnostic("ack_write_complete")
            await self._authenticated_session(connection, validator)
        except ConnectionClosed as exc:
            diagnostic("connection_closed", exc, None if exc.rcvd is None else exc.rcvd.code)
            LOGGER.info("connection closed; controller state is neutral")
        except (TimeoutError, ProtocolError) as exc:
            diagnostic("session_refused", exc)
            if isinstance(exc, asyncio.TimeoutError):
                code = FaultCode.UNAUTHORIZED
            else:
                code = _fault_code(exc)
            game_tick = max(0, validator.last_observation_tick)
            observation_seq = (
                None if validator.last_observation_seq < 0 else validator.last_observation_seq
            )
            LOGGER.warning("connection entered fail-closed state: %s (%s)", code.value, exc)
            await self._send_fault(
                connection,
                code=code,
                game_tick=game_tick,
                observation_seq=observation_seq,
            )
            with suppress(ConnectionClosed):
                await connection.close(code=1008, reason=f"fail closed: {code.value}")
        except Exception as exc:
            diagnostic("session_failed", exc)
            LOGGER.exception("connection entered fail-closed internal state")
            await self._send_fault(
                connection,
                code=FaultCode.INTERNAL,
                game_tick=game_tick,
                observation_seq=observation_seq,
            )
            with suppress(ConnectionClosed):
                await connection.close(code=1011, reason="fail closed: INTERNAL")
        finally:
            diagnostic("worker_drain_begin")
            await self.drain_workers()
            diagnostic("worker_drain_complete")


def create_server(
    config: ServiceConfig,
    controller: Controller | None = None,
    *,
    receipt_sink: Callable[[ActuationReceipt | ProgramReceipt], None] | None = None,
    issued_action_sink: Callable[[Command], None] | None = None,
    action_telemetry_sink: Callable[[ActionTelemetry], None] | None = None,
    completed_step_observer: Callable[[Command], None] | None = None,
    control_handler: ControlHandler | None = None,
) -> Any:
    """Create the configured server context manager without binding yet."""

    service = NeuralService(
        config,
        controller,
        receipt_sink=receipt_sink,
        issued_action_sink=issued_action_sink,
        action_telemetry_sink=action_telemetry_sink,
        completed_step_observer=completed_step_observer,
        control_handler=control_handler,
    )
    return websocket_serve(
        service.handle,
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


async def serve(
    config: ServiceConfig,
    controller: Controller | None = None,
    *,
    receipt_sink: Callable[[ActuationReceipt | ProgramReceipt], None] | None = None,
    issued_action_sink: Callable[[Command], None] | None = None,
    action_telemetry_sink: Callable[[ActionTelemetry], None] | None = None,
    completed_step_observer: Callable[[Command], None] | None = None,
    control_handler: ControlHandler | None = None,
) -> None:
    """Bind the loopback endpoint and run until cancelled."""

    async with create_server(
        config,
        controller,
        receipt_sink=receipt_sink,
        issued_action_sink=issued_action_sink,
        action_telemetry_sink=action_telemetry_sink,
        completed_step_observer=completed_step_observer,
        control_handler=control_handler,
    ) as server:
        LOGGER.info(
            "neural service listening on loopback port %d for run %s",
            server.sockets[0].getsockname()[1],
            config.identity.run_id,
        )
        await server.serve_forever()
