"""Authenticated lane changes over real loopback sockets and bounded synthetic workers."""

from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
from neurocraft_protocol import (
    WEBSOCKET_SUBPROTOCOL,
    Action,
    Command,
    ControlCommand,
    ControlOperation,
    Fault,
    Hello,
    HelloAck,
    Observation,
    PeerRole,
    ProgramAction,
    RunIdentity,
    RunMode,
    encode_message,
    parse_message,
)
from neurocraft_service.lane_sessions import (
    LaneName,
    LaneSessionDispatcher,
    LaneSpec,
    SessionHooks,
    create_lane_server,
)
from neurocraft_service.server import NeuralService, ServiceConfig
from websockets.asyncio.client import ClientConnection, connect
from websockets.typing import Subprotocol

from tests.protocol.helpers import action, identity, observation, receipt

TOKEN = "T" * 43
INITIAL_EPOCH = 1_783_000_000_000_000_007


class ProbeController:
    """A transport test double; no biological activity or behavior is being simulated."""

    def __init__(self, run_identity: RunIdentity, mode: RunMode) -> None:
        self.identity = run_identity
        self.run_mode = mode
        self.steps = 0
        self.resets: list[int] = []
        self.warms = 0
        self.step_hook: Callable[[], None] | None = None
        self.fail_reset = False

    def step(self, observed: Observation, *, command_seq: int) -> Action:
        assert observed.identity == self.identity
        if self.step_hook is not None:
            self.step_hook()
        result = action(
            command_sequence=command_seq,
            observation_sequence=observed.observation_seq,
            observation_tick=observed.game_tick,
            neural_start=self.steps * 5,
            run_identity=self.identity,
        )
        self.steps += 1
        return result

    def reset_session(self, *, session_epoch: int) -> None:
        assert session_epoch > self.identity.session_epoch
        self.resets.append(session_epoch)
        if self.fail_reset:
            raise RuntimeError("synthetic reset failure")
        self.identity = replace(self.identity, session_epoch=session_epoch)
        self.steps = 0

    def warm_compute_path(self, *, passes: int = 3) -> float:
        self.warms += 1
        return 0.0


def _lanes() -> tuple[list[LaneSpec], dict[LaneName, ProbeController]]:
    controllers: dict[LaneName, ProbeController] = {}
    for name, mode, dynamics in (
        ("original", RunMode.FAITHFUL, "3"),
        ("shuffled_weight", RunMode.SHUFFLED_WEIGHT, "7"),
        ("no_input", RunMode.NO_INPUT, "8"),
    ):
        controllers[name] = ProbeController(  # type: ignore[index]
            replace(identity(session_epoch=INITIAL_EPOCH), dynamics_hash=dynamics * 64), mode
        )
    return [LaneSpec(name, controller) for name, controller in controllers.items()], controllers


def _dispatcher(
    *,
    hooks_factory: Callable[[LaneSpec, RunIdentity], SessionHooks] | None = None,
) -> tuple[LaneSessionDispatcher, dict[LaneName, ProbeController]]:
    lanes, controllers = _lanes()
    config = ServiceConfig(identity=controllers["original"].identity, token=TOKEN, port=0)
    return LaneSessionDispatcher(config, lanes, hooks_factory=hooks_factory), controllers


async def _connect(port: int) -> ClientConnection:
    return await connect(
        f"ws://127.0.0.1:{port}",
        subprotocols=[Subprotocol(WEBSOCKET_SUBPROTOCOL)],
        compression=None,
    )


async def _hello(
    socket: ClientConnection, run_identity: RunIdentity, *, token: str = TOKEN
) -> Hello:
    hello = Hello.create(identity=run_identity, role=PeerRole.MINECRAFT_SERVER, token=token)
    await socket.send(encode_message(hello))
    return hello


async def _ack(socket: ClientConnection, hello: Hello) -> HelloAck:
    message = parse_message(await asyncio.wait_for(socket.recv(), 3))
    assert isinstance(message, HelloAck)
    assert message.identity == hello.identity
    assert message.verify(TOKEN, client_nonce=hello.client_nonce)
    return message


async def _step(socket: ClientConnection, run_identity: RunIdentity, sequence: int = 0) -> Action:
    await socket.send(
        encode_message(
            observation(run_identity=run_identity, sequence=sequence, tick=100 + sequence)
        )
    )
    message = parse_message(await asyncio.wait_for(socket.recv(), 3))
    assert isinstance(message, Action)
    assert message.identity == run_identity
    assert message.command_seq == sequence
    await socket.send(encode_message(receipt(message)))
    return message


def test_original_shuffle_no_input_original_uses_fresh_full_epochs_and_session_hooks() -> None:
    async def scenario() -> None:
        opened: list[tuple[str, RunIdentity]] = []
        closed: list[int] = []
        commands: dict[int, list[Command]] = {}

        def hooks(lane: LaneSpec, current: RunIdentity) -> SessionHooks:
            opened.append((lane.name, current))
            commands[current.session_epoch] = []
            return SessionHooks(
                issued_action_sink=commands[current.session_epoch].append,
                close=lambda: closed.append(current.session_epoch),
            )

        dispatcher, controllers = _dispatcher(hooks_factory=hooks)
        sockets: list[ClientConnection] = []
        try:
            async with create_lane_server(dispatcher) as server:
                port = server.sockets[0].getsockname()[1]
                for offset, lane in enumerate(
                    ("original", "shuffled_weight", "no_input", "original")
                ):
                    template = next(
                        spec.identity_template for spec in dispatcher.lanes if spec.name == lane
                    )
                    current = replace(template, session_epoch=INITIAL_EPOCH + offset)
                    socket = await _connect(port)
                    sockets.append(socket)
                    hello = await _hello(socket, current)
                    await _ack(socket, hello)
                    assert dispatcher.active_identity == current
                    assert dispatcher.active_lane == lane
                    first = await _step(socket, current)
                    assert first.neural_step_start == 0
                    second = await _step(socket, current, 1)
                    assert second.neural_step_start == 5
                    if offset:
                        assert sockets[-2].close_code == 1000
                        assert INITIAL_EPOCH + offset - 1 in closed
                assert dispatcher.last_claimed_epoch == INITIAL_EPOCH + 3
                assert controllers["original"].resets == [INITIAL_EPOCH + 3]
                assert controllers["shuffled_weight"].resets == [INITIAL_EPOCH + 1]
                assert controllers["no_input"].resets == [INITIAL_EPOCH + 2]
        finally:
            await dispatcher.close()
            for socket in sockets:
                await socket.close()
        assert [name for name, _ in opened] == [
            "original",
            "shuffled_weight",
            "no_input",
            "original",
        ]
        assert closed == list(range(INITIAL_EPOCH, INITIAL_EPOCH + 4))
        assert all(
            [command.command_seq for command in values] == [0, 1] for values in commands.values()
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["auth", "stale", "identity", "role"])
def test_invalid_request_cannot_evict_or_consume_epoch(fault: str) -> None:
    async def scenario() -> None:
        dispatcher, controllers = _dispatcher()
        current = controllers["original"].identity
        try:
            async with create_lane_server(dispatcher) as server:
                port = server.sockets[0].getsockname()[1]
                async with await _connect(port) as original:
                    await _ack(original, await _hello(original, current))
                    await _step(original, current)
                    candidate = replace(
                        controllers["shuffled_weight"].identity, session_epoch=INITIAL_EPOCH + 1
                    )
                    if fault == "stale":
                        candidate = current
                    elif fault == "identity":
                        candidate = replace(candidate, decoder_hash="f" * 64)
                    async with await _connect(port) as invalid:
                        if fault == "role":
                            hello = Hello.create(
                                identity=candidate, role=PeerRole.NEURAL_SERVICE, token=TOKEN
                            )
                            await invalid.send(encode_message(hello))
                        else:
                            await _hello(
                                invalid, candidate, token="X" * 43 if fault == "auth" else TOKEN
                            )
                        assert isinstance(
                            parse_message(await asyncio.wait_for(invalid.recv(), 3)), Fault
                        )
                    assert dispatcher.active_identity == current
                    assert dispatcher.last_claimed_epoch == INITIAL_EPOCH
                    assert not controllers["shuffled_weight"].resets
                    await _step(original, current, 1)
        finally:
            await dispatcher.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("job", ["step", "receipt", "issued", "telemetry", "control"])
def test_handover_joins_every_old_worker_before_reset_or_ack(
    job: str, caplog: pytest.LogCaptureFixture
) -> None:
    async def scenario() -> None:
        entered, release = threading.Event(), threading.Event()
        events: list[str] = []

        def block(*args: object) -> None:
            events.append("worker_started")
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test failed to release worker")
            events.append("worker_finished")

        class Handler:
            def handle_control(self, command: ControlCommand) -> None:
                block()

        def hooks(lane: LaneSpec, current: RunIdentity) -> SessionHooks:
            if lane.name != "original":
                events.append("new_hooks")
                return SessionHooks()
            return SessionHooks(
                receipt_sink=block if job == "receipt" else None,
                issued_action_sink=block if job == "issued" else None,
                action_telemetry_sink=block if job == "telemetry" else None,
                control_handler=Handler() if job == "control" else None,
                close=lambda: events.append("old_hooks_closed"),
            )

        dispatcher, controllers = _dispatcher(hooks_factory=hooks)
        if job == "step":
            controllers["original"].step_hook = block
        try:
            async with create_lane_server(dispatcher) as server:
                port = server.sockets[0].getsockname()[1]
                async with await _connect(port) as original, await _connect(port) as replacement:
                    current = controllers["original"].identity
                    await _ack(original, await _hello(original, current))
                    if job == "control":
                        command = ControlCommand(
                            current,
                            0,
                            100,
                            ControlOperation.LESION_APPLY,
                            content_id="synthetic",
                            content_hash="a" * 64,
                        )
                        await original.send(encode_message(command))
                    else:
                        await original.send(encode_message(observation(run_identity=current)))
                        if job in {"receipt", "telemetry"}:
                            emitted = parse_message(await asyncio.wait_for(original.recv(), 3))
                            assert isinstance(emitted, Action)
                            if job == "receipt":
                                await original.send(encode_message(receipt(emitted)))
                    assert await asyncio.to_thread(entered.wait, 2)
                    target = replace(
                        controllers["shuffled_weight"].identity, session_epoch=INITIAL_EPOCH + 1
                    )
                    hello = await _hello(replacement, target)
                    pending_ack = asyncio.create_task(_ack(replacement, hello))
                    await asyncio.wait_for(original.wait_closed(), 2)
                    await asyncio.sleep(0.03)
                    assert dispatcher.last_claimed_epoch == INITIAL_EPOCH + 1
                    assert not pending_ack.done()
                    assert controllers["shuffled_weight"].resets == []
                    assert controllers["shuffled_weight"].warms == 0
                    old_stages = _stages(caplog, INITIAL_EPOCH)
                    new_stages = _stages(caplog, INITIAL_EPOCH + 1)
                    # Receipt/control workers can still be awaited inside the session loop;
                    # its final drain begins only after those callbacks return.
                    assert "old_session_join_begin" in old_stages
                    assert "worker_drain_complete" not in old_stages
                    assert new_stages[-1] == "previous_owner_retire_begin"
                    assert "ack_write_begin" not in new_stages
                    release.set()
                    await pending_ack
                    assert events.index("worker_finished") < events.index("old_hooks_closed")
                    assert events.index("old_hooks_closed") < events.index("new_hooks")
                    old_stages = _stages(caplog, INITIAL_EPOCH)
                    new_stages = _stages(caplog, INITIAL_EPOCH + 1)
                    assert old_stages.index("worker_drain_complete") < old_stages.index(
                        "hooks_close_complete"
                    )
                    assert new_stages.index("previous_owner_retire_complete") < new_stages.index(
                        "reset_begin"
                    )
                    assert new_stages.index("warm_complete") < new_stages.index(
                        "hooks_create_begin"
                    )
                    assert new_stages.index("hooks_create_complete") < new_stages.index(
                        "ack_write_begin"
                    )
                    assert new_stages.index("ack_write_begin") < new_stages.index(
                        "ack_write_complete"
                    )
                    assert controllers["shuffled_weight"].resets == [INITIAL_EPOCH + 1]
                    await _step(replacement, target)
        finally:
            release.set()
            await dispatcher.close()

    asyncio.run(scenario())


def test_valid_failed_reset_consumes_epoch_and_never_acknowledges_it() -> None:
    async def scenario() -> None:
        dispatcher, controllers = _dispatcher()
        controllers["shuffled_weight"].fail_reset = True
        try:
            async with create_lane_server(dispatcher) as server:
                port = server.sockets[0].getsockname()[1]
                target = replace(
                    controllers["shuffled_weight"].identity, session_epoch=INITIAL_EPOCH + 1
                )
                async with await _connect(port) as original:
                    await _ack(original, await _hello(original, controllers["original"].identity))
                    async with await _connect(port) as replacement:
                        await _hello(replacement, target)
                        result = parse_message(await asyncio.wait_for(replacement.recv(), 3))
                        assert isinstance(result, Fault)
                    assert dispatcher.active_identity is None
                    assert dispatcher.last_claimed_epoch == target.session_epoch
                    controllers["shuffled_weight"].fail_reset = False
                    async with await _connect(port) as replay:
                        await _hello(replay, target)
                        assert isinstance(
                            parse_message(await asyncio.wait_for(replay.recv(), 3)), Fault
                        )
                    target = replace(target, session_epoch=INITIAL_EPOCH + 2)
                    async with await _connect(port) as retry:
                        await _ack(retry, await _hello(retry, target))
                        await _step(retry, target)
        finally:
            await dispatcher.close()

    asyncio.run(scenario())


def test_prefetched_hello_cannot_bypass_service_authentication() -> None:
    async def scenario() -> None:
        run_identity = identity()
        invalid = Hello.create(
            identity=run_identity, role=PeerRole.MINECRAFT_SERVER, token="X" * 43
        )

        class Connection:
            subprotocol = WEBSOCKET_SUBPROTOCOL

            def __init__(self) -> None:
                self.sent: list[Any] = []

            async def recv(self) -> str:
                raise AssertionError("prefetched hello must not read a second frame")

            async def send(self, value: str) -> None:
                self.sent.append(parse_message(value))

            async def close(self, **kwargs: object) -> None:
                pass

        connection = Connection()
        service = NeuralService(ServiceConfig(identity=run_identity, token=TOKEN))
        await service.handle(connection, prefetched_hello=invalid)  # type: ignore[arg-type]
        assert len(connection.sent) == 1 and isinstance(connection.sent[0], Fault)
        assert not service._session_epoch_claimed

    asyncio.run(scenario())


def test_simultaneous_valid_requests_can_claim_one_epoch_only_once() -> None:
    async def scenario() -> None:
        dispatcher, controllers = _dispatcher()
        try:
            async with create_lane_server(dispatcher) as server:
                port = server.sockets[0].getsockname()[1]
                async with await _connect(port) as original:
                    await _ack(original, await _hello(original, controllers["original"].identity))
                    async with await _connect(port) as first, await _connect(port) as second:
                        identities = [
                            replace(controllers[name].identity, session_epoch=INITIAL_EPOCH + 1)
                            for name in ("shuffled_weight", "no_input")
                        ]
                        hellos = await asyncio.gather(
                            _hello(first, identities[0]), _hello(second, identities[1])
                        )
                        results = [
                            parse_message(raw)
                            for raw in await asyncio.wait_for(
                                asyncio.gather(first.recv(), second.recv()), 3
                            )
                        ]
                        assert sum(isinstance(result, HelloAck) for result in results) == 1
                        assert sum(isinstance(result, Fault) for result in results) == 1
                        winner = next(
                            i for i, result in enumerate(results) if isinstance(result, HelloAck)
                        )
                        acknowledgement = results[winner]
                        assert isinstance(acknowledgement, HelloAck)
                        assert acknowledgement.verify(
                            TOKEN, client_nonce=hellos[winner].client_nonce
                        )
                        assert dispatcher.active_identity == identities[winner]
                        assert (
                            sum(len(controller.resets) for controller in controllers.values()) == 1
                        )
                        await _step((first, second)[winner], identities[winner])
        finally:
            await dispatcher.close()

    asyncio.run(scenario())


def test_worker_drain_survives_repeated_cancellation() -> None:
    async def scenario() -> None:
        entered, release = threading.Event(), threading.Event()
        service = NeuralService(ServiceConfig(identity=identity(), token=TOKEN))

        def block() -> None:
            entered.set()
            release.wait(3)

        worker = asyncio.create_task(service._run_worker(block))
        assert await asyncio.to_thread(entered.wait, 2)
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker
        draining = asyncio.create_task(service.drain_workers())
        try:
            await asyncio.sleep(0)
            draining.cancel()
            await asyncio.sleep(0)
            draining.cancel()
            await asyncio.sleep(0.02)
            assert not draining.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await draining
            assert not service._worker_jobs
        finally:
            release.set()
            await service.drain_workers()

    asyncio.run(scenario())


def test_cancelled_dispatcher_close_joins_active_worker_before_returning() -> None:
    async def scenario() -> None:
        entered, release = threading.Event(), threading.Event()
        dispatcher, controllers = _dispatcher()

        def block() -> None:
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test did not release worker")

        controllers["original"].step_hook = block
        try:
            async with (
                create_lane_server(dispatcher) as server,
                await _connect(server.sockets[0].getsockname()[1]) as original,
            ):
                current = controllers["original"].identity
                await _ack(original, await _hello(original, current))
                await original.send(encode_message(observation(run_identity=current)))
                assert await asyncio.to_thread(entered.wait, 2)
                closing = asyncio.create_task(dispatcher.close())
                await asyncio.wait_for(original.wait_closed(), 2)
                closing.cancel()
                await asyncio.sleep(0)
                closing.cancel()
                await asyncio.sleep(0.02)
                assert not closing.done()
                assert dispatcher.active_identity == current
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(closing, 2)
                assert dispatcher.active_identity is None
        finally:
            release.set()
            await dispatcher.close()

    asyncio.run(scenario())


def test_cancelled_hook_startup_joins_factory_and_closes_returned_resources() -> None:
    async def scenario() -> None:
        entered, release = threading.Event(), threading.Event()
        events: list[str] = []

        def factory(lane: LaneSpec, current: RunIdentity) -> SessionHooks:
            events.append("factory_started")
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test did not release factory")
            events.append("factory_finished")
            return SessionHooks(close=lambda: events.append("resources_closed"))

        dispatcher, controllers = _dispatcher(hooks_factory=factory)
        hello = Hello.create(
            identity=controllers["original"].identity,
            role=PeerRole.MINECRAFT_SERVER,
            token=TOKEN,
        )

        class Connection:
            subprotocol = WEBSOCKET_SUBPROTOCOL

            def __init__(self) -> None:
                self.sent: list[str] = []

            async def recv(self) -> str:
                return encode_message(hello)

            async def send(self, value: str) -> None:
                self.sent.append(value)

            async def close(self, **kwargs: object) -> None:
                pass

        connection = Connection()
        handling = asyncio.create_task(dispatcher.handle(connection))  # type: ignore[arg-type]
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            handling.cancel()
            await asyncio.sleep(0.02)
            assert not handling.done()
            assert connection.sent == []
            assert dispatcher.last_claimed_epoch == INITIAL_EPOCH
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(handling, 2)
            assert events == ["factory_started", "factory_finished", "resources_closed"]
            assert dispatcher.active_identity is None
            assert connection.sent == []
        finally:
            release.set()
            await dispatcher.close()

    asyncio.run(scenario())


def test_cancelled_close_waits_for_startup_lock_and_prevents_late_ack() -> None:
    async def scenario() -> None:
        entered, release = threading.Event(), threading.Event()
        events: list[str] = []

        def factory(lane: LaneSpec, current: RunIdentity) -> SessionHooks:
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test did not release factory")
            events.append("factory_finished")
            return SessionHooks(close=lambda: events.append("resources_closed"))

        dispatcher, controllers = _dispatcher(hooks_factory=factory)
        try:
            async with (
                create_lane_server(dispatcher) as server,
                await _connect(server.sockets[0].getsockname()[1]) as original,
            ):
                await _hello(original, controllers["original"].identity)
                assert await asyncio.to_thread(entered.wait, 2)
                closing = asyncio.create_task(dispatcher.close())
                await asyncio.sleep(0)
                closing.cancel()
                await asyncio.sleep(0)
                closing.cancel()
                await asyncio.sleep(0.02)
                assert not closing.done()
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(closing, 2)
                assert events == ["factory_finished", "resources_closed"]
                assert dispatcher.active_identity is None
                assert isinstance(parse_message(await asyncio.wait_for(original.recv(), 2)), Fault)
                async with await _connect(server.sockets[0].getsockname()[1]) as retry:
                    target = replace(
                        controllers["original"].identity, session_epoch=INITIAL_EPOCH + 1
                    )
                    await _hello(retry, target)
                    assert isinstance(parse_message(await asyncio.wait_for(retry.recv(), 2)), Fault)
                assert controllers["original"].resets == []
        finally:
            release.set()
            await dispatcher.close()

    asyncio.run(scenario())


def test_real_small_program_controllers_restore_original_state_at_new_world_ticks() -> None:
    from neurocraft_controller.program_policy import ProgramSelectingController

    from tests.controller.test_brain_display import TinySpace
    from tests.controller.test_embodied_senses import _observation, _sensors
    from tests.controller.test_full_controller_controls import clone
    from tests.protocol.helpers import program_receipt

    async def scenario() -> None:
        base = TinySpace().controller()
        controllers = {
            "original": ProgramSelectingController(clone(base, "faithful")),
            "shuffled_weight": ProgramSelectingController(clone(base, "shuffled_weight")),
            "no_input": ProgramSelectingController(clone(base, "no_input")),
        }
        specs = [LaneSpec(name, controller) for name, controller in controllers.items()]  # type: ignore[arg-type]
        dispatcher = LaneSessionDispatcher(
            ServiceConfig(identity=controllers["original"].identity, token=TOKEN, port=0), specs
        )
        streams: list[list[ProgramAction]] = []
        sockets: list[ClientConnection] = []
        try:
            async with create_lane_server(dispatcher) as server:
                for epoch, name in enumerate(
                    ("original", "shuffled_weight", "no_input", "original")
                ):
                    controller = controllers[name]
                    current = replace(controller.identity, session_epoch=epoch)
                    socket = await _connect(server.sockets[0].getsockname()[1])
                    sockets.append(socket)
                    await _ack(socket, await _hello(socket, current))
                    stream = []
                    for sequence in range(3):
                        observed = _observation(
                            controller.controller,
                            sequence,
                            _sensors(forward=6.0 if sequence == 0 else 0.2),
                        )
                        observed = replace(observed, game_tick=100 + epoch * 10 + sequence)
                        await socket.send(encode_message(observed))
                        issued = parse_message(await asyncio.wait_for(socket.recv(), 3))
                        assert isinstance(issued, ProgramAction)
                        assert issued.identity == current
                        assert issued.command_seq == sequence
                        assert issued.neural_step_start == sequence * 5
                        stream.append(issued)
                        await socket.send(encode_message(program_receipt(issued)))
                    streams.append(stream)
        finally:
            await dispatcher.close()
            for socket in sockets:
                await socket.close()
        for first, restored in zip(streams[0], streams[-1], strict=True):
            assert first.state_digest == restored.state_digest
            assert first.readout == restored.readout
            assert first.program == restored.program
            assert first.modulation == restored.modulation
            assert first.identity.session_epoch == 0 and restored.identity.session_epoch == 3
            assert restored.observation_tick > first.observation_tick

    asyncio.run(scenario())


def _stages(caplog: pytest.LogCaptureFixture, epoch: int) -> list[str]:
    result = []
    for record in caplog.records:
        if record.name not in {"neurocraft_service.lanes", "neurocraft_service"}:
            continue
        match = re.search(r"epoch=(\d+) stage=([a-z_]+)", record.getMessage())
        if match is not None and int(match.group(1)) == epoch:
            result.append(match.group(2))
    return result


def test_sixty_one_synthetic_epochs_keep_single_ownership_and_bounded_private_diagnostics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Transport stress only: neither a full connectome nor matched Minecraft trials."""

    async def scenario() -> None:
        closed: list[int] = []
        dispatcher, _ = _dispatcher(
            hooks_factory=lambda lane, current: SessionHooks(
                close=lambda: closed.append(current.session_epoch)
            )
        )
        sockets: list[ClientConnection] = []
        secrets = [TOKEN]
        try:
            async with create_lane_server(dispatcher) as server:
                port = server.sockets[0].getsockname()[1]
                for offset in range(61):
                    # Alternate peer-initiated close with dispatcher retirement of an open peer.
                    if offset and offset % 2:
                        await sockets[-1].close()
                    spec = dispatcher.lanes[offset % 3]
                    current = replace(spec.identity_template, session_epoch=INITIAL_EPOCH + offset)
                    socket = await _connect(port)
                    sockets.append(socket)
                    hello = await _hello(socket, current)
                    secrets.extend((hello.client_nonce, hello.auth))
                    await _ack(socket, hello)
                    first = await _step(socket, current)
                    assert first.neural_step_start == 0
                    assert dispatcher.active_identity == current
                    assert dispatcher.last_claimed_epoch == current.session_epoch
                    if offset:
                        assert INITIAL_EPOCH + offset - 1 in closed
                assert dispatcher.active_lane == "original"
        finally:
            await dispatcher.close()
            for socket in sockets:
                await socket.close()
        assert closed == list(range(INITIAL_EPOCH, INITIAL_EPOCH + 61))
        for epoch in closed:
            stages = _stages(caplog, epoch)
            assert stages.count("hello_authenticated") == 1
            assert stages.count("ack_write_begin") == stages.count("ack_write_complete") == 1
            assert (
                stages.count("worker_drain_complete") == stages.count("hooks_close_complete") == 1
            )
            assert len(stages) <= 27  # Fixed transition budget; independent of observation count.
        diagnostic_text = "\n".join(record.getMessage() for record in caplog.records)
        assert all(secret not in diagnostic_text for secret in secrets)
        assert str(INITIAL_EPOCH + 60) in diagnostic_text  # Full integer, never float-rounded.

    asyncio.run(scenario())


def test_diagnostic_sink_failure_does_not_change_authenticated_handover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neurocraft_service import lane_sessions
    from neurocraft_service import server as service_module

    def broken_logger(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic diagnostic sink failure")

    monkeypatch.setattr(lane_sessions.LOGGER, "warning", broken_logger)
    monkeypatch.setattr(service_module.LOGGER, "warning", broken_logger)
    # This existing test checks commands, fresh neural counters, exact identities and hook close.
    test_original_shuffle_no_input_original_uses_fresh_full_epochs_and_session_hooks()
