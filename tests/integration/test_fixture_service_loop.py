from __future__ import annotations

import asyncio
import threading
from functools import partial
from pathlib import Path

import pytest
from neurocraft_connectome import load_malecns_fixture
from neurocraft_controller import (
    ConnectomeController,
    ConnectomeRuntimeControlHandler,
    DeclaredTransitionForces,
    MinecraftReceiptTraceRecorder,
)
from neurocraft_protocol import (
    WEBSOCKET_SUBPROTOCOL,
    Action,
    ActuationReceipt,
    ActuationValidation,
    ArtifactIdentities,
    ClaimScope,
    ControlAck,
    ControlCommand,
    ControlOperation,
    ControlState,
    DataTier,
    DeterminismLevel,
    FloatPrecision,
    Hello,
    HelloAck,
    Observation,
    PeerRole,
    SensorFrame,
    TraceManifest,
    TraceWriter,
    WorldFrame,
    encode_message,
    parse_message,
    verify_trace,
)
from neurocraft_service import ActionTelemetry, ServiceConfig, create_server
from websockets.asyncio.client import connect
from websockets.typing import Subprotocol

ROOT = Path(__file__).resolve().parents[2]
TOKEN = "fixture-integration-token-000000000001"


def _record_telemetry(recorder: MinecraftReceiptTraceRecorder, telemetry: ActionTelemetry) -> None:
    recorder.accept_action_telemetry(
        command_seq=telemetry.command_seq,
        action_sha256=telemetry.action_sha256,
        queue_coalesced=telemetry.queue_coalesced,
        queue_overruns=telemetry.queue_overruns,
        compute_ns=telemetry.compute_ns,
        serialization_ns=telemetry.serialization_ns,
        write_ns=telemetry.write_ns,
    )


def _receipt(action: Action) -> ActuationReceipt:
    return ActuationReceipt(
        identity=action.identity,
        command_seq=action.command_seq,
        observation_seq=action.observation_seq,
        target_tick=action.target_tick,
        applied_tick=action.target_tick,
        world_before=WorldFrame(
            position_m=(0.0, 64.0, 0.0),
            velocity_m_s=(0.0, 0.0, 0.0),
            yaw_rad=0.0,
            pitch_rad=0.0,
            on_ground=False,
            horizontal_collision=False,
            vertical_collision=False,
            in_fluid=False,
        ),
        requested_motors=action.motors,
        applied_motors=action.motors,
        validation=ActuationValidation.APPLIED,
    )


def _forces() -> DeclaredTransitionForces:
    return DeclaredTransitionForces.create(
        passive_forces=(0.0, -0.04, 0.0),
        external_forces=(0.0, 0.0, 0.0),
        provenance="integration_test_declared_gravity_v1",
        arena_material_hash="c" * 64,
    )


def _manifest(controller: ConnectomeController, forces: DeclaredTransitionForces) -> TraceManifest:
    graph = controller.graph
    return TraceManifest(
        run_id=controller.identity.run_id,
        initial_session_epoch=controller.identity.session_epoch,
        mode=controller.run_mode,
        data_tier=DataTier.MALECNS_FIXTURE,
        claim_scope=ClaimScope.G3,
        artifacts=ArtifactIdentities(
            source_hash=graph.manifest.source_manifest_sha256,
            canonical_graph_hash=graph.manifest.canonical_graph_sha256,
            runtime_graph_hash=graph.manifest.runtime_identity_sha256,
            fixture_hash=graph.manifest.fixture_file_sha256,
            code_hash="a" * 64,
            dependency_lock_hash="b" * 64,
            adapter_hash=controller.identity.adapter_hash,
            dynamics_hash=controller.identity.dynamics_hash,
            sign_hash=controller.sign_overlay.identity_sha256,
            decoder_hash=controller.identity.decoder_hash,
            arena_hash=forces.identity_sha256,
            configuration_hash=controller.configuration.identity_sha256,
            observation_schema_hash=controller.identity.observation_schema_hash,
        ),
        game_tick_hz=20.0,
        neural_step_ms=controller.dynamics.neural_step_ms,
        precision=FloatPrecision.FLOAT64,
        backend=controller.dynamics.backend,
        thread_count=1,
        seed=0,
        determinism_level=DeterminismLevel.SAME_BUILD_EXACT,
        machine="test-machine",
        os="test-os",
        python="3.13",
        java="21.0.2",
        minecraft="1.21.5",
        fabric="0.16.10",
        mod="neurocraft-fly-integration-test",
        privacy_policy="local_trace_no_secrets_v1",
        expected_chunks=1,
        checkpoint_interval_ticks=100,
    )


def _observation(
    controller: ConnectomeController,
    sequence: int,
    *,
    open_world: bool,
) -> Observation:
    ray = 6.0 if open_world else 0.0
    return Observation(
        identity=controller.identity,
        observation_seq=sequence,
        game_tick=sequence,
        sent_monotonic_ns=sequence * 50_000_000,
        sensors=SensorFrame(
            position_m=(0.0, 64.0, 0.0),
            velocity_m_s=(0.0, 0.0, 0.0),
            yaw_rad=0.0,
            pitch_rad=0.0,
            on_ground=False,
            horizontal_collision=False,
            vertical_collision=False,
            in_fluid=False,
            local_light=15 if open_world else 0,
            ray_max_distance_m=6.0,
            ray_distances_m=(ray, 6.0, ray, ray, 6.0, 6.0),
        ),
    )


def test_authenticated_service_executes_connectome_controller_not_neutral_fake() -> None:
    async def scenario() -> None:
        graph = load_malecns_fixture(ROOT / "data" / "fixtures" / "mcns-v1.0-flight-power-v1.json")
        controller = ConnectomeController.create(
            graph,
            run_id="fixture-service-loop",
            entity_id="integration-fly",
        )
        config = ServiceConfig(identity=controller.identity, token=TOKEN, port=0)
        context = create_server(config, controller)
        server = await context.__aenter__()
        port = server.sockets[0].getsockname()[1]
        websocket = await connect(
            f"ws://127.0.0.1:{port}",
            subprotocols=[Subprotocol(WEBSOCKET_SUBPROTOCOL)],
            compression=None,
            max_size=65_536,
        )
        try:
            await websocket.send(
                encode_message(
                    Hello.create(
                        identity=controller.identity,
                        role=PeerRole.MINECRAFT_SERVER,
                        token=TOKEN,
                    )
                )
            )
            assert isinstance(parse_message(await websocket.recv()), HelloAck)

            actions: list[Action] = []
            for sequence in range(4):
                await websocket.send(
                    encode_message(_observation(controller, sequence, open_world=True))
                )
                response = parse_message(await websocket.recv())
                assert isinstance(response, Action)
                actions.append(response)
                await websocket.send(encode_message(_receipt(response)))

            assert all(not action.neutral for action in actions)
            assert not actions[-1].motors.is_neutral
            assert actions[-1].state_digest == controller.engine.current_state().state_sha256
            assert controller.last_evidence is not None
            assert controller.last_evidence.action_sha256
        finally:
            await websocket.close()
            await context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_zero_physical_input_stays_zero_through_real_service() -> None:
    async def scenario() -> None:
        graph = load_malecns_fixture(ROOT / "data" / "fixtures" / "mcns-v1.0-flight-power-v1.json")
        controller = ConnectomeController.create(
            graph,
            run_id="fixture-service-zero",
            entity_id="integration-fly",
        )
        config = ServiceConfig(identity=controller.identity, token=TOKEN, port=0)
        context = create_server(config, controller)
        server = await context.__aenter__()
        port = server.sockets[0].getsockname()[1]
        websocket = await connect(
            f"ws://127.0.0.1:{port}",
            subprotocols=[Subprotocol(WEBSOCKET_SUBPROTOCOL)],
            compression=None,
        )
        try:
            await websocket.send(
                encode_message(
                    Hello.create(
                        identity=controller.identity,
                        role=PeerRole.MINECRAFT_SERVER,
                        token=TOKEN,
                    )
                )
            )
            assert isinstance(parse_message(await websocket.recv()), HelloAck)
            for sequence in range(4):
                await websocket.send(
                    encode_message(_observation(controller, sequence, open_world=False))
                )
                response = parse_message(await websocket.recv())
                assert isinstance(response, Action)
                assert response.motors.is_neutral
                assert not response.neutral  # computed zero, not the neutral fake/fault path
                await websocket.send(encode_message(_receipt(response)))
        finally:
            await websocket.close()
            await context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_record_start_detaches_action_issued_before_control_and_traces_after_floor(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        graph = load_malecns_fixture(ROOT / "data/fixtures/mcns-v1.0-flight-power-v1.json")
        controller = ConnectomeController.create(
            graph,
            run_id="fixture-service-start-floor",
            entity_id="integration-fly",
        )
        forces = _forces()
        output = tmp_path / "start-floor.ndjson"
        recorder = MinecraftReceiptTraceRecorder(
            controller,
            TraceWriter(output, _manifest(controller, forces)),
            forces=forces,
        )
        context = create_server(
            ServiceConfig(identity=controller.identity, token=TOKEN, port=0),
            controller,
            issued_action_sink=recorder.accept_issued_action,
            action_telemetry_sink=partial(_record_telemetry, recorder),
            receipt_sink=recorder.accept_receipt,
            control_handler=ConnectomeRuntimeControlHandler(recorder=recorder),
        )
        server = await context.__aenter__()
        port = server.sockets[0].getsockname()[1]
        websocket = await connect(
            f"ws://127.0.0.1:{port}",
            subprotocols=[Subprotocol(WEBSOCKET_SUBPROTOCOL)],
            compression=None,
        )
        try:
            await websocket.send(
                encode_message(
                    Hello.create(
                        identity=controller.identity,
                        role=PeerRole.MINECRAFT_SERVER,
                        token=TOKEN,
                    )
                )
            )
            assert isinstance(parse_message(await websocket.recv()), HelloAck)

            await websocket.send(encode_message(_observation(controller, 0, open_world=True)))
            before_start = parse_message(await websocket.recv())
            assert isinstance(before_start, Action)

            start = ControlCommand(
                identity=controller.identity,
                control_seq=0,
                issued_tick=0,
                operation=ControlOperation.RECORD_START,
                content_id="integration-recording",
            )
            await websocket.send(encode_message(start))
            start_ack = parse_message(await websocket.recv())
            assert isinstance(start_ack, ControlAck)
            assert start_ack.accepted and start_ack.state is ControlState.RECORDING

            await websocket.send(encode_message(_receipt(before_start)))
            await websocket.send(encode_message(_observation(controller, 1, open_world=True)))
            after_start = parse_message(await websocket.recv())
            assert isinstance(after_start, Action)
            assert after_start.command_seq == 1
            await websocket.send(encode_message(_receipt(after_start)))

            stop = ControlCommand(
                identity=controller.identity,
                control_seq=1,
                # One action budget after the last observation inside the recording: the
                # boundary keeps the actions targeting 2 and 3.
                issued_tick=3,
                operation=ControlOperation.RECORD_STOP,
                content_id="integration-recording",
            )
            await websocket.send(encode_message(stop))
            stop_ack = parse_message(await websocket.recv())
            assert isinstance(stop_ack, ControlAck)
            assert stop_ack.accepted and stop_ack.state is ControlState.RUNNING

            for sequence in (2, 3, 4):
                await websocket.send(
                    encode_message(_observation(controller, sequence, open_world=True))
                )
                after_stop = parse_message(await websocket.recv())
                assert isinstance(after_stop, Action)
                await websocket.send(encode_message(_receipt(after_stop)))

            assert recorder.state.value == "CLOSED"
            assert recorder.record_count == 1
            assert [binding.command_seq for binding in recorder.bindings] == [1]
        finally:
            await websocket.close()
            await context.__aexit__(None, None, None)

        verification = verify_trace(output)
        assert verification.record_count == 1
        assert verification.first_tick == 1
        assert verification.last_tick == 1

    asyncio.run(scenario())


def test_record_stop_flushes_last_receipt_on_next_observation_without_caching_new_action(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        graph = load_malecns_fixture(ROOT / "data/fixtures/mcns-v1.0-flight-power-v1.json")
        controller = ConnectomeController.create(
            graph,
            run_id="fixture-service-recording",
            entity_id="integration-fly",
        )
        forces = _forces()
        output = tmp_path / "receipt-bound.ndjson"
        recorder = MinecraftReceiptTraceRecorder(
            controller,
            TraceWriter(output, _manifest(controller, forces)),
            forces=forces,
        )
        control_handler = ConnectomeRuntimeControlHandler(recorder=recorder)
        context = create_server(
            ServiceConfig(identity=controller.identity, token=TOKEN, port=0),
            controller,
            issued_action_sink=recorder.accept_issued_action,
            action_telemetry_sink=partial(_record_telemetry, recorder),
            receipt_sink=recorder.accept_receipt,
            control_handler=control_handler,
        )
        server = await context.__aenter__()
        port = server.sockets[0].getsockname()[1]
        websocket = await connect(
            f"ws://127.0.0.1:{port}",
            subprotocols=[Subprotocol(WEBSOCKET_SUBPROTOCOL)],
            compression=None,
        )
        try:
            await websocket.send(
                encode_message(
                    Hello.create(
                        identity=controller.identity,
                        role=PeerRole.MINECRAFT_SERVER,
                        token=TOKEN,
                    )
                )
            )
            assert isinstance(parse_message(await websocket.recv()), HelloAck)
            start = ControlCommand(
                identity=controller.identity,
                control_seq=0,
                issued_tick=0,
                operation=ControlOperation.RECORD_START,
                content_id="integration-recording",
            )
            await websocket.send(encode_message(start))
            start_ack = parse_message(await websocket.recv())
            assert isinstance(start_ack, ControlAck)
            assert start_ack.accepted and start_ack.state is ControlState.RECORDING

            for sequence in (0, 1):
                await websocket.send(
                    encode_message(_observation(controller, sequence, open_world=True))
                )
                action = parse_message(await websocket.recv())
                assert isinstance(action, Action)
                await websocket.send(encode_message(_receipt(action)))

            stop = ControlCommand(
                identity=controller.identity,
                control_seq=1,
                # One action budget after the last observation inside the recording: the
                # boundary keeps the actions targeting 2 and 3.
                issued_tick=3,
                operation=ControlOperation.RECORD_STOP,
                content_id="integration-recording",
            )
            await websocket.send(encode_message(stop))
            stop_ack = parse_message(await websocket.recv())
            assert isinstance(stop_ack, ControlAck)
            assert stop_ack.accepted and stop_ack.state is ControlState.RUNNING

            for sequence in (2, 3, 4):
                await websocket.send(
                    encode_message(_observation(controller, sequence, open_world=True))
                )
                outside_action = parse_message(await websocket.recv())
                assert isinstance(outside_action, Action)
                await websocket.send(encode_message(_receipt(outside_action)))
            assert recorder.state.value == "CLOSED"
            assert recorder.record_count == 2
        finally:
            await websocket.close()
            await context.__aexit__(None, None, None)

        verification = verify_trace(output)
        assert verification.record_count == 2
        assert verification.first_tick == 0
        assert verification.last_tick == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("delay_outside_telemetry", [False, True])
def test_service_record_stop_drains_observation_first_receipt_pipeline(
    tmp_path: Path,
    delay_outside_telemetry: bool,
) -> None:
    async def scenario() -> None:
        graph = load_malecns_fixture(ROOT / "data/fixtures/mcns-v1.0-flight-power-v1.json")
        controller = ConnectomeController.create(
            graph,
            run_id="fixture-service-stop-pipeline",
            entity_id="integration-fly",
        )
        forces = _forces()
        output = tmp_path / "stop-pipeline.ndjson"
        recorder = MinecraftReceiptTraceRecorder(
            controller,
            TraceWriter(output, _manifest(controller, forces)),
            forces=forces,
        )
        outside_receipt_accepted = threading.Event()

        def record_telemetry(telemetry: ActionTelemetry) -> None:
            if delay_outside_telemetry and telemetry.command_seq == 2:
                # Force the socket peer's receipt to win the callback race. The
                # two in-bound actions still await their world-after observations.
                assert outside_receipt_accepted.wait(timeout=5)
                assert recorder.state.value == "STOPPING"
            _record_telemetry(recorder, telemetry)

        def record_receipt(receipt: ActuationReceipt) -> None:
            recorder.accept_receipt(receipt)
            if receipt.command_seq == 2:
                outside_receipt_accepted.set()

        context = create_server(
            ServiceConfig(identity=controller.identity, token=TOKEN, port=0),
            controller,
            issued_action_sink=recorder.accept_issued_action,
            action_telemetry_sink=record_telemetry,
            receipt_sink=record_receipt,
            control_handler=ConnectomeRuntimeControlHandler(recorder=recorder),
        )
        server = await context.__aenter__()
        port = server.sockets[0].getsockname()[1]
        websocket = await connect(
            f"ws://127.0.0.1:{port}",
            subprotocols=[Subprotocol(WEBSOCKET_SUBPROTOCOL)],
            compression=None,
        )
        try:
            await websocket.send(
                encode_message(
                    Hello.create(
                        identity=controller.identity,
                        role=PeerRole.MINECRAFT_SERVER,
                        token=TOKEN,
                    )
                )
            )
            assert isinstance(parse_message(await websocket.recv()), HelloAck)
            start = ControlCommand(
                identity=controller.identity,
                control_seq=0,
                issued_tick=0,
                operation=ControlOperation.RECORD_START,
                content_id="integration-recording",
            )
            await websocket.send(encode_message(start))
            start_ack = parse_message(await websocket.recv())
            assert isinstance(start_ack, ControlAck)
            assert start_ack.accepted and start_ack.state is ControlState.RECORDING

            # Mirror the Fabric bridge: send the next observation before the
            # preceding action's receipt. Two actions are therefore pending.
            await websocket.send(encode_message(_observation(controller, 0, open_world=True)))
            action_0 = parse_message(await websocket.recv())
            assert isinstance(action_0, Action)
            await websocket.send(encode_message(_observation(controller, 1, open_world=True)))
            action_1 = parse_message(await websocket.recv())
            assert isinstance(action_1, Action)
            await websocket.send(encode_message(_receipt(action_0)))

            stop = ControlCommand(
                identity=controller.identity,
                control_seq=1,
                # One action budget after the last observation inside the recording: the
                # boundary keeps the actions targeting 2 and 3.
                issued_tick=3,
                operation=ControlOperation.RECORD_STOP,
                content_id="integration-recording",
            )
            await websocket.send(encode_message(stop))
            stop_ack = parse_message(await websocket.recv())
            assert isinstance(stop_ack, ControlAck)
            assert stop_ack.accepted and stop_ack.state is ControlState.RUNNING
            assert recorder.state.value == "STOPPING"

            # The in-bound receipt may legitimately arrive after RECORD_STOP.
            await websocket.send(encode_message(_receipt(action_1)))
            for sequence in (2, 3, 4):
                await websocket.send(
                    encode_message(_observation(controller, sequence, open_world=True))
                )
                outside = parse_message(await websocket.recv())
                assert isinstance(outside, Action)
                await websocket.send(encode_message(_receipt(outside)))

            assert recorder.state.value == "CLOSED"
            assert [binding.command_seq for binding in recorder.bindings] == [0, 1]
        finally:
            await websocket.close()
            await context.__aexit__(None, None, None)

        verification = verify_trace(output)
        assert verification.record_count == 2
        assert verification.first_tick == 0
        assert verification.last_tick == 1

    asyncio.run(scenario())


def test_connectome_service_pause_step_reset_resume_preserves_transport_sequences() -> None:
    async def scenario() -> None:
        graph = load_malecns_fixture(ROOT / "data/fixtures/mcns-v1.0-flight-power-v1.json")
        controller = ConnectomeController.create(
            graph,
            run_id="fixture-service-controls",
            entity_id="integration-fly",
        )
        context = create_server(
            ServiceConfig(identity=controller.identity, token=TOKEN, port=0),
            controller,
        )
        server = await context.__aenter__()
        port = server.sockets[0].getsockname()[1]
        websocket = await connect(
            f"ws://127.0.0.1:{port}",
            subprotocols=[Subprotocol(WEBSOCKET_SUBPROTOCOL)],
            compression=None,
        )

        async def control(
            sequence: int,
            tick: int,
            operation: ControlOperation,
            observation_sequence: int,
        ) -> tuple[ControlAck, Action]:
            await websocket.send(
                encode_message(
                    ControlCommand(
                        identity=controller.identity,
                        control_seq=sequence,
                        issued_tick=tick,
                        operation=operation,
                    )
                )
            )
            await websocket.send(
                encode_message(_observation(controller, observation_sequence, open_world=True))
            )
            response = parse_message(await websocket.recv())
            action = parse_message(await websocket.recv())
            assert isinstance(response, ControlAck)
            assert isinstance(action, Action)
            expected_tick = (
                action.target_tick
                if operation is ControlOperation.RESET
                else action.observation_tick
            )
            assert response.effective_tick == expected_tick
            await websocket.send(encode_message(_receipt(action)))
            return response, action

        async def observe(sequence: int) -> Action:
            await websocket.send(
                encode_message(_observation(controller, sequence, open_world=True))
            )
            response = parse_message(await websocket.recv())
            assert isinstance(response, Action)
            await websocket.send(encode_message(_receipt(response)))
            return response

        try:
            await websocket.send(
                encode_message(
                    Hello.create(
                        identity=controller.identity,
                        role=PeerRole.MINECRAFT_SERVER,
                        token=TOKEN,
                    )
                )
            )
            assert isinstance(parse_message(await websocket.recv()), HelloAck)
            pause, paused = await control(0, 0, ControlOperation.PAUSE, 0)
            assert pause.accepted and pause.state is ControlState.PAUSED
            assert paused.command_seq == 0 and paused.neutral

            step, stepped = await control(1, 1, ControlOperation.SINGLE_STEP, 1)
            assert step.accepted and step.state is ControlState.STEP_ARMED
            assert stepped.command_seq == 1 and not stepped.neutral
            paused_again = await observe(2)
            assert paused_again.command_seq == 2 and paused_again.neutral

            reset, reset_action = await control(2, 3, ControlOperation.RESET, 3)
            assert reset.accepted and reset.state is ControlState.PAUSED
            # A RESET is stamped with the governed action's target tick, which is the
            # observation it materialized against (3) plus the two-tick action budget.
            assert reset.effective_tick == reset_action.target_tick == 5
            assert reset_action.command_seq == 3 and reset_action.neutral
            assert controller.engine.step_index == 0
            resume, after_reset = await control(3, 4, ControlOperation.RESUME, 4)
            assert resume.accepted and resume.state is ControlState.RUNNING
            assert after_reset.command_seq == 4
            assert after_reset.neural_step_start == 5
            assert after_reset.neural_step_end == 10
        finally:
            await websocket.close()
            await context.__aexit__(None, None, None)

    asyncio.run(scenario())
