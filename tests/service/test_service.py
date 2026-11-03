from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import replace
from pathlib import Path

import pytest
from neurocraft_protocol import (
    ACTION_TARGET_OFFSET_TICKS,
    WEBSOCKET_SUBPROTOCOL,
    Action,
    ControlAck,
    ControlCommand,
    ControlOperation,
    ControlState,
    Fault,
    FaultCode,
    Hello,
    HelloAck,
    MotorVector,
    Observation,
    PeerRole,
    ProgramAction,
    ProgramId,
    ProgramValidation,
    ProtocolError,
    encode_message,
    generate_client_nonce,
    handshake_auth,
    parse_message,
)
from neurocraft_service import (
    ActionTelemetry,
    BoundedCommandQueue,
    DeterministicNeutralController,
    LatestValueQueue,
    ServiceConfig,
    create_run_token,
    create_server,
)
from websockets.asyncio.client import connect

from tests.protocol.helpers import action as make_action
from tests.protocol.helpers import identity, observation, receipt
from tests.protocol.helpers import program_action as make_program_action
from tests.protocol.helpers import program_receipt as make_program_receipt

TOKEN = "T" * 43


class CountingController:
    def __init__(self) -> None:
        self.steps = 0
        self.resets = 0

    def step(self, observed, *, command_seq: int) -> Action:
        result = make_action(
            command_sequence=command_seq,
            observation_sequence=observed.observation_seq,
            observation_tick=observed.game_tick,
            neural_start=self.steps * 5,
            run_identity=observed.identity,
            motors=MotorVector(forward_thrust=0.4),
        )
        self.steps += 1
        return result

    def reset(self) -> None:
        self.resets += 1


class CountingProgramController:
    """Program-lane test double: it issues ``program_action`` frames and counts its steps.

    A transport double, never a Faithful Organism controller and never a selection policy: the
    program it names is a constant, not a readout mapping. It exists to prove the wire, the
    window, and the receipt binding carry a program frame, and that a paused service still emits
    a neutral ``action`` on the same session.
    """

    def __init__(self) -> None:
        self.steps = 0

    def step(self, observed, *, command_seq: int) -> ProgramAction:
        result = make_program_action(
            command_sequence=command_seq,
            observation_sequence=observed.observation_seq,
            observation_tick=observed.game_tick,
            neural_start=self.steps * 5,
            run_identity=observed.identity,
        )
        self.steps += 1
        return result

    def reset(self) -> None:
        """Explicit reset capability; this stateless double has no mutable model state."""


class OffBudgetController:
    """A defective controller that targets the wrong tick.

    No profile in this repository does this — every producer computes its target from the one
    declared ``ACTION_TARGET_OFFSET_TICKS`` — which is exactly why the service-side check has
    to be tested with a double. ``offset`` is the offset this double actually emits.
    """

    def __init__(self, offset: int) -> None:
        self.offset = offset
        self.steps = 0

    def step(self, observed, *, command_seq: int) -> Action:
        target = observed.game_tick + self.offset
        result = make_action(
            command_sequence=command_seq,
            observation_sequence=observed.observation_seq,
            observation_tick=observed.game_tick,
            target_tick=target,
            valid_through_tick=max(target, observed.game_tick + ACTION_TARGET_OFFSET_TICKS),
            neural_start=self.steps * 5,
            run_identity=observed.identity,
            motors=MotorVector(forward_thrust=0.4),
        )
        self.steps += 1
        return result

    def reset(self) -> None:
        """Explicit reset capability; this stateless double has no mutable model state."""


class LaneSwitchingController:
    """A defective controller that changes driving frame kind mid-session.

    No profile in this repository does this — the frame kind is a property of the controller,
    not a per-tick choice — which is exactly why the latch has to be tested with a double.
    """

    def __init__(self) -> None:
        self.steps = 0

    def step(self, observed, *, command_seq: int) -> Action | ProgramAction:
        common = {
            "command_sequence": command_seq,
            "observation_sequence": observed.observation_seq,
            "observation_tick": observed.game_tick,
            "neural_start": self.steps * 5,
            "run_identity": observed.identity,
        }
        result: Action | ProgramAction = (
            make_action(**common, motors=MotorVector(forward_thrust=0.4))
            if self.steps == 0
            else make_program_action(**common)
        )
        self.steps += 1
        return result


class RecordingControlHandler:
    def __init__(self) -> None:
        self.commands: list[ControlCommand] = []

    def handle_control(self, command: ControlCommand) -> None:
        self.commands.append(command)


class FailingControlHandler:
    def handle_control(self, command: ControlCommand) -> None:
        raise RuntimeError("deliberate control handler failure")


class RejectingControlHandler:
    def __init__(self, code: FaultCode) -> None:
        self.code = code.value

    def handle_control(self, command: ControlCommand) -> None:
        raise ExpectedControlRejection(self.code)


class ExpectedControlRejection(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__("deliberate expected control rejection")
        self.code = code


def config(*, token: str = TOKEN) -> ServiceConfig:
    return ServiceConfig(identity=identity(), token=token, port=0)


def test_service_config_requires_loopback_ip_literal_and_hides_token() -> None:
    assert TOKEN not in repr(config())
    for host in ("0.0.0.0", "192.168.1.4", "localhost"):
        with pytest.raises(ProtocolError, match=r"loopback|IP literal"):
            replace(config(), host=host)
    assert ServiceConfig(identity=identity(), token=TOKEN, host="::1", port=0).host == "::1"


def test_generated_tokens_are_protocol_valid_and_not_constant() -> None:
    first = create_run_token()
    second = create_run_token()
    assert first != second
    assert len(first) >= 32
    ServiceConfig(identity=identity(), token=first, port=0)


def test_latest_queue_coalesces_to_one_slot() -> None:
    async def scenario() -> None:
        queue: LatestValueQueue[int] = LatestValueQueue()
        queue.put_latest(1)
        queue.put_latest(2)
        queue.put_latest(3)
        assert queue.size == 1
        assert queue.coalesced_count == 2
        assert await queue.get() == 3
        queue.task_done()

    asyncio.run(scenario())


def test_command_queue_refuses_third_pending_value() -> None:
    queue: BoundedCommandQueue[int] = BoundedCommandQueue()
    assert queue.put(1)
    assert queue.put(2)
    assert not queue.put(3)
    assert queue.size == 2
    assert queue.overrun_count == 1


def test_default_controller_is_deterministic_explicitly_neutral() -> None:
    controller = DeterministicNeutralController()
    first = controller.step(observation(), command_seq=0)
    second = controller.step(observation(), command_seq=0)
    assert first == second
    assert first.neutral
    assert first.motors == MotorVector.neutral()
    assert first.target_tick == observation().game_tick + ACTION_TARGET_OFFSET_TICKS


async def authenticated_connection(
    service_config: ServiceConfig,
    *,
    controller=None,
    receipt_sink=None,
    issued_action_sink=None,
    action_telemetry_sink=None,
    control_handler=None,
):
    server_context = create_server(
        service_config,
        controller,
        receipt_sink=receipt_sink,
        issued_action_sink=issued_action_sink,
        action_telemetry_sink=action_telemetry_sink,
        control_handler=control_handler,
    )
    server = await server_context.__aenter__()
    port = server.sockets[0].getsockname()[1]
    websocket = await connect(
        f"ws://127.0.0.1:{port}",
        subprotocols=[WEBSOCKET_SUBPROTOCOL],
        compression=None,
        max_size=65_536,
    )
    return server_context, websocket


async def authenticate(websocket) -> HelloAck:
    hello = Hello.create(identity=identity(), role=PeerRole.MINECRAFT_SERVER, token=TOKEN)
    await websocket.send(encode_message(hello))
    acknowledgement = parse_message(await websocket.recv())
    assert isinstance(acknowledgement, HelloAck)
    assert acknowledgement.verify(TOKEN, client_nonce=hello.client_nonce)
    return acknowledgement


async def materialize_effectful_control(
    websocket,
    command: ControlCommand,
    observed: Observation,
) -> tuple[ControlAck, Action]:
    """Send a control and the first observation governed by its accepted effect."""
    await websocket.send(encode_message(command))
    await websocket.send(encode_message(observed))
    acknowledgement = parse_message(await websocket.recv())
    action = parse_message(await websocket.recv())
    assert isinstance(acknowledgement, ControlAck)
    assert isinstance(action, Action)
    expected_tick = (
        action.target_tick
        if command.operation is ControlOperation.RESET
        else action.observation_tick
    )
    assert acknowledgement.effective_tick == expected_tick
    return acknowledgement, action


def test_authenticated_websocket_round_trip_is_neutral_and_identity_bound() -> None:
    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(config())
        try:
            await websocket.send(
                encode_message(
                    Hello.create(
                        identity=identity(),
                        role=PeerRole.MINECRAFT_SERVER,
                        token=TOKEN,
                    )
                )
            )
            ack = parse_message(await websocket.recv())
            assert isinstance(ack, HelloAck)
            assert ack.identity == identity()

            await websocket.send(encode_message(observation()))
            action = parse_message(await websocket.recv())
            assert isinstance(action, Action)
            assert action.identity == identity()
            assert action.observation_seq == 0
            # observation 100 plus the declared two-tick budget; the ceiling and the
            # target coincide, so validity is the target tick itself.
            assert action.target_tick == 102
            assert action.valid_through_tick == 102
            assert action.neutral and action.motors.is_neutral
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_server_faults_unauthorized_on_bad_client_auth() -> None:
    """A hello whose proof was not derived from the run token never reaches a session."""

    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(config())
        try:
            honest = Hello.create(identity=identity(), role=PeerRole.MINECRAFT_SERVER, token=TOKEN)
            forged = replace(
                honest,
                auth=handshake_auth(
                    "F" * 43,
                    label=Hello.TYPE,
                    identity=honest.identity,
                    role=honest.role,
                    client_nonce=honest.client_nonce,
                ),
            )
            await websocket.send(encode_message(forged))
            message = parse_message(await websocket.recv())
            assert isinstance(message, Fault)
            assert message.code is FaultCode.UNAUTHORIZED
            assert message.neutral_applied
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_server_proves_possession_bound_to_fresh_nonce() -> None:
    """The service's acknowledgement authenticates only against the nonce it was sent."""

    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(config())
        try:
            hello = Hello.create(identity=identity(), role=PeerRole.MINECRAFT_SERVER, token=TOKEN)
            await websocket.send(encode_message(hello))
            raw = await websocket.recv()
            acknowledgement = parse_message(raw)
            assert isinstance(acknowledgement, HelloAck)

            assert acknowledgement.verify(TOKEN, client_nonce=hello.client_nonce)
            # Bound to this nonce, this token, and the service's own role.
            assert not acknowledgement.verify(TOKEN, client_nonce=generate_client_nonce())
            assert not acknowledgement.verify("Z" * 43, client_nonce=hello.client_nonce)
            assert acknowledgement.server_auth != hello.auth
            assert TOKEN not in raw
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_wrong_token_fails_closed_without_logging_token(caplog: pytest.LogCaptureFixture) -> None:
    wrong_token = "W" * 43

    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(config())
        try:
            await websocket.send(
                encode_message(
                    Hello.create(
                        identity=identity(),
                        role=PeerRole.MINECRAFT_SERVER,
                        token=wrong_token,
                    )
                )
            )
            message = parse_message(await websocket.recv())
            assert isinstance(message, Fault)
            assert message.code is FaultCode.UNAUTHORIZED
            assert message.neutral_applied
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    caplog.set_level(logging.DEBUG)
    asyncio.run(scenario())
    assert TOKEN not in caplog.text
    assert wrong_token not in caplog.text


def test_duplicate_observation_sequence_enters_neutral_fault() -> None:
    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(config())
        try:
            await websocket.send(
                encode_message(
                    Hello.create(identity=identity(), role=PeerRole.MINECRAFT_SERVER, token=TOKEN)
                )
            )
            assert isinstance(parse_message(await websocket.recv()), HelloAck)
            await websocket.send(encode_message(observation()))
            assert isinstance(parse_message(await websocket.recv()), Action)
            await websocket.send(encode_message(observation()))
            fault = parse_message(await websocket.recv())
            assert isinstance(fault, Fault)
            assert fault.code is FaultCode.SEQUENCE
            assert fault.related_observation_seq == 0
            assert fault.neutral_applied
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_reconnect_cannot_reuse_consumed_session_epoch() -> None:
    async def scenario() -> None:
        server_context = create_server(config())
        server = await server_context.__aenter__()
        port = server.sockets[0].getsockname()[1]
        try:
            first = await connect(
                f"ws://127.0.0.1:{port}",
                subprotocols=[WEBSOCKET_SUBPROTOCOL],
                compression=None,
            )
            await first.send(
                encode_message(
                    Hello.create(identity=identity(), role=PeerRole.MINECRAFT_SERVER, token=TOKEN)
                )
            )
            assert isinstance(parse_message(await first.recv()), HelloAck)
            await first.close()

            second = await connect(
                f"ws://127.0.0.1:{port}",
                subprotocols=[WEBSOCKET_SUBPROTOCOL],
                compression=None,
            )
            await second.send(
                encode_message(
                    Hello.create(identity=identity(), role=PeerRole.MINECRAFT_SERVER, token=TOKEN)
                )
            )
            fault = parse_message(await second.recv())
            assert isinstance(fault, Fault)
            assert fault.code is FaultCode.IDENTITY_MISMATCH
            await second.close()
        finally:
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_wrong_artifact_identity_is_rejected_after_handshake() -> None:
    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(config())
        try:
            await websocket.send(
                encode_message(
                    Hello.create(identity=identity(), role=PeerRole.MINECRAFT_SERVER, token=TOKEN)
                )
            )
            assert isinstance(parse_message(await websocket.recv()), HelloAck)
            wrong_identity = identity(graph_hash="d" * 64)
            await websocket.send(encode_message(observation(run_identity=wrong_identity)))
            fault = parse_message(await websocket.recv())
            assert isinstance(fault, Fault)
            assert fault.code is FaultCode.IDENTITY_MISMATCH
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_malformed_unknown_field_is_rejected_without_echo() -> None:
    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(config())
        try:
            await websocket.send(
                encode_message(
                    Hello.create(identity=identity(), role=PeerRole.MINECRAFT_SERVER, token=TOKEN)
                )
            )
            assert isinstance(parse_message(await websocket.recv()), HelloAck)
            raw = observation().to_dict()
            raw["hidden_policy"] = "do-not-echo-this"
            import json

            await websocket.send(json.dumps(raw))
            raw_fault = await websocket.recv()
            assert "do-not-echo-this" not in raw_fault
            fault = parse_message(raw_fault)
            assert isinstance(fault, Fault)
            assert fault.code is FaultCode.UNKNOWN_FIELD
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_pause_single_step_and_reset_state_machine_is_fail_neutral() -> None:
    controller = CountingController()

    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(config(), controller=controller)
        try:
            await authenticate(websocket)
            pause = ControlCommand(
                identity=identity(),
                control_seq=0,
                issued_tick=99,
                operation=ControlOperation.PAUSE,
            )
            pause_ack, paused_action = await materialize_effectful_control(
                websocket, pause, observation(sequence=0, tick=100)
            )
            assert pause_ack.accepted and pause_ack.state is ControlState.PAUSED
            assert paused_action.neutral and paused_action.motors.is_neutral
            assert controller.steps == 0
            await websocket.send(encode_message(receipt(paused_action)))

            step = ControlCommand(
                identity=identity(),
                control_seq=1,
                issued_tick=100,
                operation=ControlOperation.SINGLE_STEP,
            )
            step_ack, stepped_action = await materialize_effectful_control(
                websocket, step, observation(sequence=1, tick=101)
            )
            assert step_ack.accepted and step_ack.state is ControlState.STEP_ARMED
            assert not stepped_action.neutral
            assert stepped_action.motors.forward_thrust == 0.4
            assert controller.steps == 1
            await websocket.send(encode_message(receipt(stepped_action)))

            await websocket.send(encode_message(observation(sequence=2, tick=102)))
            paused_again = parse_message(await websocket.recv())
            assert isinstance(paused_again, Action)
            assert paused_again.neutral
            assert controller.steps == 1
            await websocket.send(encode_message(receipt(paused_again)))

            reset = ControlCommand(
                identity=identity(),
                control_seq=2,
                issued_tick=103,
                operation=ControlOperation.RESET,
            )
            reset_ack, reset_action = await materialize_effectful_control(
                websocket, reset, observation(sequence=3, tick=104)
            )
            assert reset_ack.accepted and reset_ack.state is ControlState.PAUSED
            assert controller.resets == 1
            assert reset_action.neutral
            await websocket.send(encode_message(receipt(reset_action)))
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_record_and_replay_reject_when_control_handler_is_unavailable() -> None:
    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(config())
        try:
            await authenticate(websocket)
            record = ControlCommand(
                identity=identity(),
                control_seq=0,
                issued_tick=100,
                operation=ControlOperation.RECORD_START,
                content_id="recording-1",
            )
            await websocket.send(encode_message(record))
            record_ack = parse_message(await websocket.recv())
            assert isinstance(record_ack, ControlAck)
            assert not record_ack.accepted
            assert record_ack.fault_code is FaultCode.CONTROL_UNAVAILABLE
            assert record_ack.state is ControlState.PAUSED

            replay = ControlCommand(
                identity=identity(),
                control_seq=1,
                issued_tick=101,
                operation=ControlOperation.REPLAY_START,
                content_id="recording-1",
                content_hash="a" * 64,
            )
            await websocket.send(encode_message(replay))
            replay_ack = parse_message(await websocket.recv())
            assert isinstance(replay_ack, ControlAck)
            assert not replay_ack.accepted
            assert replay_ack.fault_code is FaultCode.CONTROL_UNAVAILABLE
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_record_and_replay_delegate_to_explicit_handler() -> None:
    handler = RecordingControlHandler()

    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(
            config(), control_handler=handler
        )
        try:
            await authenticate(websocket)
            commands = [
                ControlCommand(
                    identity=identity(),
                    control_seq=0,
                    issued_tick=100,
                    operation=ControlOperation.RECORD_START,
                    content_id="recording-1",
                ),
                ControlCommand(
                    identity=identity(),
                    control_seq=1,
                    issued_tick=101,
                    operation=ControlOperation.RECORD_STOP,
                    content_id="recording-1",
                ),
                ControlCommand(
                    identity=identity(),
                    control_seq=2,
                    issued_tick=102,
                    operation=ControlOperation.PAUSE,
                ),
                ControlCommand(
                    identity=identity(),
                    control_seq=3,
                    issued_tick=103,
                    operation=ControlOperation.REPLAY_START,
                    content_id="recording-1",
                    content_hash="a" * 64,
                ),
                ControlCommand(
                    identity=identity(),
                    control_seq=4,
                    issued_tick=104,
                    operation=ControlOperation.REPLAY_STOP,
                    content_id="recording-1",
                    content_hash="a" * 64,
                ),
            ]
            for command, expected_state in zip(
                commands[:2],
                (ControlState.RECORDING, ControlState.RUNNING),
                strict=True,
            ):
                await websocket.send(encode_message(command))
                ack = parse_message(await websocket.recv())
                assert isinstance(ack, ControlAck)
                assert ack.accepted and ack.state is expected_state
            for observation_sequence, (command, expected_state) in enumerate(
                zip(
                    commands[2:],
                    (ControlState.PAUSED, ControlState.REPLAYING, ControlState.PAUSED),
                    strict=True,
                )
            ):
                ack, action = await materialize_effectful_control(
                    websocket,
                    command,
                    observation(sequence=observation_sequence, tick=command.issued_tick),
                )
                assert ack.accepted and ack.state is expected_state
                await websocket.send(encode_message(receipt(action)))
            assert [item.operation for item in handler.commands] == [
                ControlOperation.RECORD_START,
                ControlOperation.RECORD_STOP,
                ControlOperation.REPLAY_START,
                ControlOperation.REPLAY_STOP,
            ]
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_intervention_control_delegates_and_keeps_service_state_unchanged() -> None:
    handler = RecordingControlHandler()

    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(
            config(), control_handler=handler
        )
        try:
            await authenticate(websocket)
            command = ControlCommand(
                identity=identity(),
                control_seq=0,
                issued_tick=100,
                operation=ControlOperation.STIMULATION_APPLY,
                content_id="declared-stimulation",
                content_hash="b" * 64,
            )
            ack, action = await materialize_effectful_control(
                websocket, command, observation(sequence=0, tick=100)
            )
            assert ack.accepted
            assert ack.state is ControlState.RUNNING
            assert handler.commands == [command]
            await websocket.send(encode_message(receipt(action)))
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_control_ack_is_an_effect_fence_after_in_flight_action_issuance() -> None:
    action_sink_entered = threading.Event()
    release_action_sink = threading.Event()
    handler = RecordingControlHandler()

    def blocking_action_sink(_action: Action) -> None:
        action_sink_entered.set()
        if not release_action_sink.wait(timeout=2.0):
            raise RuntimeError("test did not release the action sink")

    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(
            config(),
            controller=CountingController(),
            issued_action_sink=blocking_action_sink,
            control_handler=handler,
        )
        try:
            await authenticate(websocket)
            await websocket.send(encode_message(observation(sequence=0, tick=100)))
            assert await asyncio.to_thread(action_sink_entered.wait, 1.0)

            command = ControlCommand(
                identity=identity(),
                control_seq=0,
                issued_tick=100,
                operation=ControlOperation.STIMULATION_APPLY,
                content_id="declared-stimulation",
                content_hash="b" * 64,
            )
            await websocket.send(encode_message(command))
            await asyncio.sleep(0.02)
            assert handler.commands == []

            release_action_sink.set()
            issued = parse_message(await websocket.recv())
            assert isinstance(issued, Action)
            await websocket.send(encode_message(receipt(issued)))
            await websocket.send(encode_message(observation(sequence=1, tick=101)))
            acknowledgement = parse_message(await websocket.recv())
            governed = parse_message(await websocket.recv())
            assert isinstance(acknowledgement, ControlAck)
            assert isinstance(governed, Action)
            assert acknowledgement.accepted
            assert acknowledgement.effective_tick == governed.observation_tick == 101
            assert handler.commands == [command]
            await websocket.send(encode_message(receipt(governed)))
        finally:
            release_action_sink.set()
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_pause_issued_after_an_observation_is_stamped_with_the_next_observation_tick() -> None:
    """The wire relation the mod's PAUSE transaction is built on (ADR 0020 lane A).

    Live, ``/neurocraftfly program <id>`` runs on the server thread after the entity tick, so
    its PAUSE is issued at tick T with observation T already sent -- and, because
    ``FlyEntity.tick()`` polls control acknowledgements *before* it sends the observation, no
    acknowledgement derived from observation X can reach the entity before tick X + 1. This
    pins what the service stamps under that ordering: the acknowledgement materializes against
    the next observation answered paused and carries ``effective_tick == T + 1``, the
    observation tick, while the neutral action issued beside it targets T + 3 under the
    two-tick action budget. The mod first polls the acknowledgement at T + 2 and engages at
    ``effective_tick + ACTION_TARGET_OFFSET_TICKS`` -- T + 3, the tick that first paused
    command governs -- because ADR 0020 section 3.3 lets a scripted program own a tick only
    when the admitted command is neutral, and engaging earlier hands the body to a program on
    a tick an in-flight driving command still governs.
    ``mod/src/test/.../StagedProgramAuthorityTest`` holds the Java half of this relation, and
    the opt-in ``ExternalBridgeProgramAuthorityGameTest`` drives it for real.
    """

    controller = CountingController()

    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(config(), controller=controller)
        try:
            await authenticate(websocket)
            await websocket.send(encode_message(observation(sequence=0, tick=100)))
            running = parse_message(await websocket.recv())
            assert isinstance(running, Action)
            assert not running.neutral and running.neural_step_end > running.neural_step_start
            await websocket.send(encode_message(receipt(running)))

            pause = ControlCommand(
                identity=identity(),
                control_seq=0,
                issued_tick=100,
                operation=ControlOperation.PAUSE,
            )
            await websocket.send(encode_message(pause))
            await websocket.send(encode_message(observation(sequence=1, tick=101)))
            acknowledgement = parse_message(await websocket.recv())
            paused = parse_message(await websocket.recv())
            assert isinstance(acknowledgement, ControlAck)
            assert isinstance(paused, Action)
            assert acknowledgement.accepted and acknowledgement.state is ControlState.PAUSED
            assert acknowledgement.issued_tick == 100
            # Stamped with the observation tick, not the tick the effect governs in the world.
            assert acknowledgement.effective_tick == paused.observation_tick == 101
            # The first command governed by the pause applies one action budget later.
            assert paused.target_tick == acknowledgement.effective_tick + 2 == 103
            assert paused.neutral
            assert paused.neural_step_start == paused.neural_step_end == running.neural_step_end
            assert controller.steps == 1
            await websocket.send(encode_message(receipt(paused)))
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "handler,expected_fault",
    [
        (None, FaultCode.CONTROL_UNAVAILABLE),
        (RejectingControlHandler(FaultCode.CONTROL_UNAVAILABLE), FaultCode.CONTROL_UNAVAILABLE),
        (RejectingControlHandler(FaultCode.INVALID_STATE), FaultCode.INVALID_STATE),
    ],
)
def test_intervention_rejection_does_not_mutate_service_state(
    handler, expected_fault: FaultCode
) -> None:
    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(
            config(), control_handler=handler
        )
        try:
            await authenticate(websocket)
            command = ControlCommand(
                identity=identity(),
                control_seq=0,
                issued_tick=100,
                operation=ControlOperation.LESION_APPLY,
                content_id="declared-lesion",
                content_hash="c" * 64,
            )
            await websocket.send(encode_message(command))
            ack = parse_message(await websocket.recv())
            assert isinstance(ack, ControlAck)
            assert not ack.accepted
            assert ack.fault_code is expected_fault
            assert ack.state is ControlState.RUNNING
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_receipt_sink_observes_only_valid_bound_receipt() -> None:
    observed_receipts = []

    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(
            config(), receipt_sink=observed_receipts.append
        )
        try:
            await authenticate(websocket)
            await websocket.send(encode_message(observation()))
            issued = parse_message(await websocket.recv())
            assert isinstance(issued, Action)
            applied = receipt(issued)
            await websocket.send(encode_message(applied))
            pause = ControlCommand(
                identity=identity(),
                control_seq=0,
                issued_tick=101,
                operation=ControlOperation.PAUSE,
            )
            pause_ack, paused = await materialize_effectful_control(
                websocket, pause, observation(sequence=1, tick=101)
            )
            assert pause_ack.accepted and paused.neutral
            await websocket.send(encode_message(receipt(paused)))
            assert observed_receipts == [applied]
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_issued_action_sink_observes_controller_and_paused_neutral_actions() -> None:
    issued_actions: list[Action] = []
    telemetry: list[ActionTelemetry] = []
    controller = CountingController()

    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(
            config(),
            controller=controller,
            issued_action_sink=issued_actions.append,
            action_telemetry_sink=telemetry.append,
        )
        try:
            await authenticate(websocket)
            await websocket.send(encode_message(observation(sequence=0, tick=100)))
            running = parse_message(await websocket.recv())
            assert isinstance(running, Action)
            await websocket.send(encode_message(receipt(running)))

            pause = ControlCommand(
                identity=identity(),
                control_seq=0,
                issued_tick=101,
                operation=ControlOperation.PAUSE,
            )
            pause_ack, paused = await materialize_effectful_control(
                websocket, pause, observation(sequence=1, tick=101)
            )
            assert pause_ack.accepted
            assert paused.neutral
            for _ in range(50):
                if len(telemetry) == 2:
                    break
                await asyncio.sleep(0.001)
            assert issued_actions == [running, paused]
            assert [item.command_seq for item in telemetry] == [0, 1]
            assert all(item.compute_ns > 0 for item in telemetry)
            assert all(item.serialization_ns > 0 for item in telemetry)
            assert all(item.write_ns > 0 for item in telemetry)
            assert all(item.queue_coalesced == 0 for item in telemetry)
            assert all(item.queue_overruns == 0 for item in telemetry)
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_issued_action_sink_observes_replay_and_fault_neutral_actions() -> None:
    async def replay_scenario() -> None:
        issued_actions: list[Action] = []
        server_context, websocket = await authenticated_connection(
            config(),
            issued_action_sink=issued_actions.append,
            control_handler=RecordingControlHandler(),
        )
        try:
            await authenticate(websocket)
            pause = ControlCommand(
                identity=identity(),
                control_seq=0,
                issued_tick=100,
                operation=ControlOperation.PAUSE,
            )
            replay = ControlCommand(
                identity=identity(),
                control_seq=1,
                issued_tick=101,
                operation=ControlOperation.REPLAY_START,
                content_id="trace-1",
                content_hash="a" * 64,
            )
            pause_ack, paused = await materialize_effectful_control(
                websocket, pause, observation(sequence=0, tick=100)
            )
            assert pause_ack.accepted and paused.neutral
            await websocket.send(encode_message(receipt(paused)))
            replay_ack, replay_neutral = await materialize_effectful_control(
                websocket, replay, observation(sequence=1, tick=101)
            )
            assert replay_ack.accepted
            assert replay_neutral.neutral
            assert issued_actions == [paused, replay_neutral]
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    async def fault_scenario() -> None:
        issued_actions: list[Action] = []
        server_context, websocket = await authenticated_connection(
            config(),
            issued_action_sink=issued_actions.append,
            control_handler=FailingControlHandler(),
        )
        try:
            await authenticate(websocket)
            record = ControlCommand(
                identity=identity(),
                control_seq=0,
                issued_tick=100,
                operation=ControlOperation.RECORD_START,
                content_id="trace-1",
            )
            await websocket.send(encode_message(record))
            ack = parse_message(await websocket.recv())
            assert isinstance(ack, ControlAck)
            assert ack.state is ControlState.FAULTED
            await websocket.send(encode_message(observation(sequence=0, tick=101)))
            fault_neutral = parse_message(await websocket.recv())
            assert isinstance(fault_neutral, Action)
            assert fault_neutral.neutral
            assert issued_actions == [fault_neutral]
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(replay_scenario())
    asyncio.run(fault_scenario())


def test_issued_action_sink_failure_fails_closed_as_trace_failure() -> None:
    def failing_sink(_action: Action) -> None:
        raise RuntimeError("deliberate issued action sink failure")

    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(
            config(), issued_action_sink=failing_sink
        )
        try:
            await authenticate(websocket)
            await websocket.send(encode_message(observation()))
            fault = parse_message(await websocket.recv())
            assert isinstance(fault, Fault)
            assert fault.code is FaultCode.TRACE_FAILURE
            assert fault.neutral_applied
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_forged_actuation_receipt_fails_closed() -> None:
    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(config())
        try:
            await authenticate(websocket)
            await websocket.send(encode_message(observation()))
            issued = parse_message(await websocket.recv())
            assert isinstance(issued, Action)
            forged = replace(
                receipt(issued),
                requested_motors=MotorVector(forward_thrust=0.5),
                applied_motors=MotorVector(forward_thrust=0.5),
            )
            await websocket.send(encode_message(forged))
            fault = parse_message(await websocket.recv())
            assert isinstance(fault, Fault)
            assert fault.code is FaultCode.RECEIPT_MISMATCH
            assert fault.neutral_applied
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_receipt_sink_failure_fails_closed_as_trace_failure() -> None:
    def failing_sink(_receipt) -> None:
        raise RuntimeError("deliberate sink failure")

    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(
            config(), receipt_sink=failing_sink
        )
        try:
            await authenticate(websocket)
            await websocket.send(encode_message(observation()))
            issued = parse_message(await websocket.recv())
            assert isinstance(issued, Action)
            await websocket.send(encode_message(receipt(issued)))
            fault = parse_message(await websocket.recv())
            assert isinstance(fault, Fault)
            assert fault.code is FaultCode.TRACE_FAILURE
            assert fault.neutral_applied
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_control_handler_failure_enters_faulted_neutral_state() -> None:
    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(
            config(), control_handler=FailingControlHandler()
        )
        try:
            await authenticate(websocket)
            command = ControlCommand(
                identity=identity(),
                control_seq=0,
                issued_tick=100,
                operation=ControlOperation.RECORD_START,
                content_id="recording-1",
            )
            await websocket.send(encode_message(command))
            ack = parse_message(await websocket.recv())
            assert isinstance(ack, ControlAck)
            assert not ack.accepted
            assert ack.state is ControlState.FAULTED
            assert ack.fault_code is FaultCode.INTERNAL
            await websocket.send(encode_message(observation()))
            neutral = parse_message(await websocket.recv())
            assert isinstance(neutral, Action)
            assert neutral.neutral and neutral.motors.is_neutral
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_duplicate_control_sequence_fails_closed() -> None:
    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(config())
        try:
            await authenticate(websocket)
            pause = ControlCommand(
                identity=identity(),
                control_seq=0,
                issued_tick=100,
                operation=ControlOperation.PAUSE,
            )
            pause_ack, paused = await materialize_effectful_control(
                websocket, pause, observation(sequence=0, tick=100)
            )
            assert pause_ack.accepted and paused.neutral
            await websocket.send(encode_message(pause))
            fault = parse_message(await websocket.recv())
            assert isinstance(fault, Fault)
            assert fault.code is FaultCode.SEQUENCE
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_second_unmaterialized_effect_control_fails_before_outbound_overflow() -> None:
    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(
            config(), controller=CountingController()
        )
        try:
            await authenticate(websocket)
            pause = ControlCommand(
                identity=identity(),
                control_seq=0,
                issued_tick=100,
                operation=ControlOperation.PAUSE,
            )
            reset = ControlCommand(
                identity=identity(),
                control_seq=1,
                issued_tick=101,
                operation=ControlOperation.RESET,
            )
            await websocket.send(encode_message(pause))
            await websocket.send(encode_message(reset))
            fault = parse_message(await websocket.recv())
            assert isinstance(fault, Fault)
            assert fault.code is FaultCode.SEQUENCE
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_recording_and_execution_state_interleavings_are_rejected_but_stoppable() -> None:
    handler = RecordingControlHandler()

    async def recording_scenario() -> None:
        server_context, websocket = await authenticated_connection(
            config(), controller=CountingController(), control_handler=handler
        )
        try:
            await authenticate(websocket)
            start = ControlCommand(
                identity=identity(),
                control_seq=0,
                issued_tick=100,
                operation=ControlOperation.RECORD_START,
                content_id="recording-1",
            )
            await websocket.send(encode_message(start))
            started = parse_message(await websocket.recv())
            assert isinstance(started, ControlAck)
            assert started.accepted and started.state is ControlState.RECORDING

            pause = ControlCommand(
                identity=identity(),
                control_seq=1,
                issued_tick=101,
                operation=ControlOperation.PAUSE,
            )
            await websocket.send(encode_message(pause))
            rejected = parse_message(await websocket.recv())
            assert isinstance(rejected, ControlAck)
            assert not rejected.accepted
            assert rejected.fault_code is FaultCode.INVALID_STATE
            assert rejected.state is ControlState.RECORDING

            stop = ControlCommand(
                identity=identity(),
                control_seq=2,
                issued_tick=102,
                operation=ControlOperation.RECORD_STOP,
                content_id="recording-1",
            )
            await websocket.send(encode_message(stop))
            stopped = parse_message(await websocket.recv())
            assert isinstance(stopped, ControlAck)
            assert stopped.accepted and stopped.state is ControlState.RUNNING
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    async def paused_scenario() -> None:
        server_context, websocket = await authenticated_connection(
            config(), controller=CountingController(), control_handler=RecordingControlHandler()
        )
        try:
            await authenticate(websocket)
            pause = ControlCommand(
                identity=identity(),
                control_seq=0,
                issued_tick=100,
                operation=ControlOperation.PAUSE,
            )
            pause_ack, paused = await materialize_effectful_control(
                websocket, pause, observation(sequence=0, tick=100)
            )
            assert pause_ack.accepted and paused.neutral
            await websocket.send(encode_message(receipt(paused)))

            start = ControlCommand(
                identity=identity(),
                control_seq=1,
                issued_tick=101,
                operation=ControlOperation.RECORD_START,
                content_id="recording-2",
            )
            await websocket.send(encode_message(start))
            rejected = parse_message(await websocket.recv())
            assert isinstance(rejected, ControlAck)
            assert not rejected.accepted
            assert rejected.fault_code is FaultCode.INVALID_STATE
            assert rejected.state is ControlState.PAUSED
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(recording_scenario())
    asyncio.run(paused_scenario())


def test_service_dependency_is_exactly_pinned() -> None:
    root = Path(__file__).resolve().parents[2]
    pyproject = (root / "services/neural-engine/pyproject.toml").read_text()
    assert '"websockets==17.0.1"' in pyproject
    assert '"neurocraft-protocol==0.1.0"' in pyproject
    assert '"src/neurocraft_service/py.typed"' in pyproject
    assert (root / "services/neural-engine/src/neurocraft_service/py.typed").is_file()
    protocol_project = (root / "packages/protocol/pyproject.toml").read_text()
    assert '"src/neurocraft_protocol/py.typed"' in protocol_project
    assert (root / "packages/protocol/src/neurocraft_protocol/py.typed").is_file()


def test_program_lane_controller_emits_program_actions_and_neutral_actions_while_paused() -> None:
    """One driving frame kind per session, and PAUSE is still a neutral ``action``.

    The paused, replaying, and faulted branch is untouched by the program lane: the connectome is
    not stepping, so nothing can be selecting, and the frame the service emits says so with an
    empty neural interval and the ``service_neutral_v1`` digest a reader can recompute.
    """

    controller = CountingProgramController()

    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(config(), controller=controller)
        try:
            await authenticate(websocket)
            await websocket.send(encode_message(observation(sequence=0, tick=100)))
            issued = parse_message(await websocket.recv())
            assert isinstance(issued, ProgramAction)
            assert issued.program is ProgramId.CRUISE
            assert issued.command_seq == 0
            assert issued.observation_seq == 0
            assert issued.observation_tick == 100
            assert issued.target_tick == 102
            assert issued.neural_step_end > issued.neural_step_start
            assert issued.readout.wing_common == 0.61
            assert len(issued.policy_hash) == 64
            assert controller.steps == 1
            await websocket.send(encode_message(make_program_receipt(issued)))

            pause = ControlCommand(
                identity=identity(),
                control_seq=0,
                issued_tick=101,
                operation=ControlOperation.PAUSE,
            )
            pause_ack, paused = await materialize_effectful_control(
                websocket, pause, observation(sequence=1, tick=101)
            )
            assert pause_ack.accepted and pause_ack.state is ControlState.PAUSED
            # A neutral Action, not a ProgramAction: the paused branch is byte-untouched.
            assert isinstance(paused, Action)
            assert not isinstance(paused, ProgramAction)
            assert paused.neutral and paused.motors.is_neutral
            assert paused.neural_step_end == paused.neural_step_start
            assert controller.steps == 1
            # The pending frame is an Action, so the actuation receipt is what binds it.
            await websocket.send(encode_message(receipt(paused)))

            resume = ControlCommand(
                identity=identity(),
                control_seq=1,
                issued_tick=102,
                operation=ControlOperation.RESUME,
            )
            await websocket.send(encode_message(resume))
            await websocket.send(encode_message(observation(sequence=2, tick=102)))
            resume_ack = parse_message(await websocket.recv())
            resumed = parse_message(await websocket.recv())
            assert isinstance(resume_ack, ControlAck) and resume_ack.accepted
            assert isinstance(resumed, ProgramAction)
            assert controller.steps == 2
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_program_receipt_sink_observes_only_bound_receipts() -> None:
    """An unbound program receipt faults the session closed and never reaches the sink."""

    observed_receipts = []
    controller = CountingProgramController()

    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(
            config(), controller=controller, receipt_sink=observed_receipts.append
        )
        try:
            await authenticate(websocket)
            await websocket.send(encode_message(observation(sequence=0, tick=100)))
            issued = parse_message(await websocket.recv())
            assert isinstance(issued, ProgramAction)
            applied = make_program_receipt(issued)
            await websocket.send(encode_message(applied))

            await websocket.send(encode_message(observation(sequence=1, tick=101)))
            second = parse_message(await websocket.recv())
            assert isinstance(second, ProgramAction)
            # An actuation receipt cannot retire a pending program frame, even with a
            # matching sequence, tick, and identity.
            impostor = receipt(
                make_action(
                    command_sequence=second.command_seq,
                    observation_sequence=second.observation_seq,
                    observation_tick=second.observation_tick,
                )
            )
            await websocket.send(encode_message(impostor))
            fault = parse_message(await websocket.recv())
            assert isinstance(fault, Fault)
            assert fault.code is FaultCode.RECEIPT_MISMATCH
            assert observed_receipts == [applied]
            assert observed_receipts[0].validation is ProgramValidation.PROGRAM_APPLIED
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_a_controller_that_changes_driving_frame_kind_is_refused() -> None:
    """One driving frame kind per session: a mixed session has no honest answer to
    "what was driving the fly at tick T", so the second kind fails the session closed."""

    controller = LaneSwitchingController()

    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(config(), controller=controller)
        try:
            await authenticate(websocket)
            await websocket.send(encode_message(observation(sequence=0, tick=100)))
            first = parse_message(await websocket.recv())
            assert isinstance(first, Action) and not first.neutral
            await websocket.send(encode_message(receipt(first)))

            await websocket.send(encode_message(observation(sequence=1, tick=101)))
            fault = parse_message(await websocket.recv())
            assert isinstance(fault, Fault)
            assert fault.code is FaultCode.MALFORMED
            assert controller.steps == 2
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("offset", "expected_code"),
    [(1, FaultCode.SEQUENCE), (3, FaultCode.MALFORMED)],
    ids=["one_tick_early_is_refused", "one_tick_late_cannot_be_expressed"],
)
def test_a_controller_that_misses_the_declared_action_budget_is_refused(
    offset: int, expected_code: FaultCode
) -> None:
    """The producer-side half of the budget contract, on both sides of it.

    Every command frame must target its observation tick plus ACTION_TARGET_OFFSET_TICKS. A
    controller that targets one tick early would be applied a tick before the world expects
    it, and ``_validate_generated_action`` refuses it with SEQUENCE. A controller that targets
    a tick late cannot even build the frame: validity may not exceed the budget ceiling and
    may not precede the target, so the frame model refuses it and the session fails closed on
    MALFORMED. Either way nothing reaches the wire, which is why the authority's own admission
    rules stay bounds rather than a second copy of this equality.
    """

    controller = OffBudgetController(offset)

    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(config(), controller=controller)
        try:
            await authenticate(websocket)
            await websocket.send(encode_message(observation(sequence=0, tick=100)))
            fault = parse_message(await websocket.recv())
            assert isinstance(fault, Fault)
            assert fault.code is expected_code
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_a_controller_on_the_declared_action_budget_is_accepted() -> None:
    """The same double, on the budget: the refusal above is about the offset and nothing else."""

    controller = OffBudgetController(ACTION_TARGET_OFFSET_TICKS)

    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(config(), controller=controller)
        try:
            await authenticate(websocket)
            await websocket.send(encode_message(observation(sequence=0, tick=100)))
            issued = parse_message(await websocket.recv())
            assert isinstance(issued, Action)
            assert issued.target_tick == 102 and issued.valid_through_tick == 102
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_paused_neutral_actions_do_not_decide_or_violate_the_program_lane() -> None:
    """Neutral actions are lane-agnostic, exactly as the authority's own latch treats them.

    A paused service emits them on both lanes and they command nothing, so a paused window
    inside a program session must neither latch the motor lane nor trip the latch when the
    program lane resumes.
    """

    controller = CountingProgramController()

    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(config(), controller=controller)
        try:
            await authenticate(websocket)
            pause = ControlCommand(
                identity=identity(),
                control_seq=0,
                issued_tick=99,
                operation=ControlOperation.PAUSE,
            )
            # The lane is still UNDECIDED here: the first frame of the session is neutral.
            pause_ack, paused = await materialize_effectful_control(
                websocket, pause, observation(sequence=0, tick=100)
            )
            assert pause_ack.accepted and paused.neutral
            assert controller.steps == 0
            await websocket.send(encode_message(receipt(paused)))

            resume = ControlCommand(
                identity=identity(),
                control_seq=1,
                issued_tick=100,
                operation=ControlOperation.RESUME,
            )
            await websocket.send(encode_message(resume))
            await websocket.send(encode_message(observation(sequence=1, tick=101)))
            resume_ack = parse_message(await websocket.recv())
            issued = parse_message(await websocket.recv())
            assert isinstance(resume_ack, ControlAck) and resume_ack.accepted
            assert isinstance(issued, ProgramAction)
            await websocket.send(encode_message(make_program_receipt(issued)))

            # A second paused window, now that the lane is latched PROGRAM.
            pause_again = ControlCommand(
                identity=identity(),
                control_seq=2,
                issued_tick=101,
                operation=ControlOperation.PAUSE,
            )
            second_ack, paused_again = await materialize_effectful_control(
                websocket, pause_again, observation(sequence=2, tick=102)
            )
            assert second_ack.accepted
            assert isinstance(paused_again, Action) and paused_again.neutral
            await websocket.send(encode_message(receipt(paused_again)))
            assert controller.steps == 1
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())
