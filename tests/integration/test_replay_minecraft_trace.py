from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from neurocraft_connectome import load_malecns_fixture
from neurocraft_controller import (
    ConnectomeController,
    DeclaredTransitionForces,
    MinecraftReceiptTraceRecorder,
    causal_record_from_applied_action,
    world_state_from_observation,
)
from neurocraft_protocol import (
    ACTION_TARGET_OFFSET_TICKS,
    Action,
    ActuationReceipt,
    ActuationValidation,
    ClaimScope,
    ControlCommand,
    ControlOperation,
    Observation,
    TraceWriter,
    WorldFrame,
    canonical_json_bytes,
    verify_trace,
)

from scripts.fixture_harness import HarnessWorld, build_manifest
from scripts.prepare_gametest_demo import _code_bundle, _dependency_bundle
from scripts.replay_minecraft_trace import iter_records, replay_minecraft_trace, write_receipt

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "data" / "fixtures" / "mcns-v1.0-flight-power-v1.json"
STEERING_FIXTURE = ROOT / "data" / "fixtures" / "mcns-v1.0-flight-steering-v1.json"
WING_CONTROLLER_PROFILE = "wing_motor_soma_side_v1"
TRM_CONTROLLER_PROFILE = "trm_cloned_readout_v1"
TRM_READOUT_ARTIFACT = ROOT / "data" / "readouts" / "trm-cloned-readout-v2.json"


def _minecraft_labelled_trace(path: Path, *, intervention: str | None = None) -> None:
    graph = load_malecns_fixture(FIXTURE)
    controller = ConnectomeController.create(
        graph,
        run_id="minecraft-replay-test",
        entity_id="00000000-0000-0000-0000-000000000001",
    )
    manifest = replace(
        build_manifest(controller, graph),
        claim_scope=ClaimScope.G3,
        artifacts=replace(
            build_manifest(controller, graph).artifacts,
            code_hash=_code_bundle()[0],
            dependency_lock_hash=_dependency_bundle()[0],
        ),
        java="21",
        minecraft="1.21.5",
        fabric="loader-0.16.10-api-0.128.2+1.21.5",
        mod="neurocraft-fly-0.1.0-test",
    )
    world = HarnessWorld()
    # Under the two-tick action budget the action answering observation T applies at
    # T + ACTION_TARGET_OFFSET_TICKS, so the world it acts on is observation T + 2 and the
    # world it produced is observation T + 3: one more observation than the one-tick lane.
    observations = tuple(world.observe(controller, tick) for tick in range(7))
    with TraceWriter(path, manifest) as writer:
        for tick in range(4):
            observation = observations[tick]
            action = controller.step(observation, command_seq=tick)
            applied = tick + ACTION_TARGET_OFFSET_TICKS
            record = causal_record_from_applied_action(
                controller,
                record_seq=tick,
                observation=observation,
                world_after=world_state_from_observation(observations[applied + 1]),
                world_after_observation_seq=observations[applied + 1].observation_seq,
                world_after_game_tick=observations[applied + 1].game_tick,
                observation_age_ticks=ACTION_TARGET_OFFSET_TICKS,
            )
            record = replace(
                record,
                actuation_receipt=_receipt(action, observations[applied]),
                world_before=world_state_from_observation(observations[applied]),
                compute_ns=100,
                serialization_ns=20,
                write_ns=30,
            )
            if intervention is not None and tick == 1:
                record = replace(record, interventions=(intervention,))
            writer.append(record)


def _wing_labelled_trace(path: Path) -> None:
    """Mirror of ``_minecraft_labelled_trace`` under the opt-in wing profile:
    the pinned 128-node steering fixture driven by the flight controller."""

    from neurocraft_connectome.flight_pin import (
        MALECNS_FLIGHT_FIXTURE_V1_FILE_BYTES,
        MALECNS_FLIGHT_FIXTURE_V1_PIN,
    )
    from neurocraft_controller import FlightConnectomeController

    graph = load_malecns_fixture(
        STEERING_FIXTURE,
        expected_pin=MALECNS_FLIGHT_FIXTURE_V1_PIN,
        expected_file_bytes=MALECNS_FLIGHT_FIXTURE_V1_FILE_BYTES,
    )
    controller = FlightConnectomeController(
        graph,
        run_id="minecraft-wing-replay-test",
        entity_id="00000000-0000-0000-0000-000000000003",
    )
    manifest = replace(
        build_manifest(controller, graph),
        claim_scope=ClaimScope.G3,
        artifacts=replace(
            build_manifest(controller, graph).artifacts,
            code_hash=_code_bundle()[0],
            dependency_lock_hash=_dependency_bundle()[0],
        ),
        java="21",
        minecraft="1.21.5",
        fabric="loader-0.16.10-api-0.128.2+1.21.5",
        mod="neurocraft-fly-0.1.0-test",
    )
    world = HarnessWorld()
    # Under the two-tick action budget the action answering observation T applies at
    # T + ACTION_TARGET_OFFSET_TICKS, so the world it acts on is observation T + 2 and the
    # world it produced is observation T + 3: one more observation than the one-tick lane.
    observations = tuple(world.observe(controller, tick) for tick in range(7))
    with TraceWriter(path, manifest) as writer:
        for tick in range(4):
            observation = observations[tick]
            action = controller.step(observation, command_seq=tick)
            applied = tick + ACTION_TARGET_OFFSET_TICKS
            record = causal_record_from_applied_action(
                controller,
                record_seq=tick,
                observation=observation,
                world_after=world_state_from_observation(observations[applied + 1]),
                world_after_observation_seq=observations[applied + 1].observation_seq,
                world_after_game_tick=observations[applied + 1].game_tick,
                observation_age_ticks=ACTION_TARGET_OFFSET_TICKS,
            )
            record = replace(
                record,
                actuation_receipt=_receipt(action, observations[applied]),
                world_before=world_state_from_observation(observations[applied]),
                compute_ns=100,
                serialization_ns=20,
                write_ns=30,
            )
            writer.append(record)


def _trm_labelled_trace(path: Path) -> None:
    """Mirror of ``_wing_labelled_trace`` under the opt-in trained-readout
    profile: the pinned steering fixture driven by the TRM demo controller
    bound to the committed cloned-readout artifact."""

    from neurocraft_connectome.flight_pin import (
        MALECNS_FLIGHT_FIXTURE_V1_FILE_BYTES,
        MALECNS_FLIGHT_FIXTURE_V1_PIN,
    )
    from neurocraft_controller.trained_readout_controller import (
        TrainedReadoutController,
        TrmDemoConfiguration,
    )

    graph = load_malecns_fixture(
        STEERING_FIXTURE,
        expected_pin=MALECNS_FLIGHT_FIXTURE_V1_PIN,
        expected_file_bytes=MALECNS_FLIGHT_FIXTURE_V1_FILE_BYTES,
    )
    artifact_bytes = TRM_READOUT_ARTIFACT.read_bytes()
    controller = TrainedReadoutController(
        graph,
        run_id="minecraft-trm-replay-test",
        entity_id="00000000-0000-0000-0000-000000000004",
        artifact_bytes=artifact_bytes,
        artifact_document=json.loads(artifact_bytes),
        configuration=TrmDemoConfiguration(
            readout_artifact_sha256=hashlib.sha256(artifact_bytes).hexdigest()
        ),
    )
    manifest = replace(
        build_manifest(controller, graph),
        claim_scope=ClaimScope.G3,
        artifacts=replace(
            build_manifest(controller, graph).artifacts,
            code_hash=_code_bundle()[0],
            dependency_lock_hash=_dependency_bundle()[0],
        ),
        java="21",
        minecraft="1.21.5",
        fabric="loader-0.16.10-api-0.128.2+1.21.5",
        mod="neurocraft-fly-0.1.0-test",
    )
    world = HarnessWorld()
    # Under the two-tick action budget the action answering observation T applies at
    # T + ACTION_TARGET_OFFSET_TICKS, so the world it acts on is observation T + 2 and the
    # world it produced is observation T + 3: one more observation than the one-tick lane.
    observations = tuple(world.observe(controller, tick) for tick in range(7))
    with TraceWriter(path, manifest) as writer:
        for tick in range(4):
            observation = observations[tick]
            action = controller.step(observation, command_seq=tick)
            applied = tick + ACTION_TARGET_OFFSET_TICKS
            record = causal_record_from_applied_action(
                controller,
                record_seq=tick,
                observation=observation,
                world_after=world_state_from_observation(observations[applied + 1]),
                world_after_observation_seq=observations[applied + 1].observation_seq,
                world_after_game_tick=observations[applied + 1].game_tick,
                observation_age_ticks=ACTION_TARGET_OFFSET_TICKS,
            )
            record = replace(
                record,
                actuation_receipt=_receipt(action, observations[applied]),
                world_before=world_state_from_observation(observations[applied]),
                compute_ns=100,
                serialization_ns=20,
                write_ns=30,
            )
            writer.append(record)


def _receipt(action: Action, world_before: Observation) -> ActuationReceipt:
    sensors = world_before.sensors
    return ActuationReceipt(
        identity=action.identity,
        command_seq=action.command_seq,
        observation_seq=action.observation_seq,
        target_tick=action.target_tick,
        applied_tick=action.target_tick,
        world_before=WorldFrame(
            position_m=sensors.position_m,
            velocity_m_s=sensors.velocity_m_s,
            yaw_rad=sensors.yaw_rad,
            pitch_rad=sensors.pitch_rad,
            on_ground=sensors.on_ground,
            horizontal_collision=sensors.horizontal_collision,
            vertical_collision=sensors.vertical_collision,
            in_fluid=sensors.in_fluid,
        ),
        requested_motors=action.motors,
        applied_motors=action.motors,
        validation=ActuationValidation.APPLIED,
    )


def _accept_with_test_telemetry(recorder: MinecraftReceiptTraceRecorder, action: Action) -> None:
    recorder.accept_issued_action(action)
    recorder.accept_action_telemetry(
        command_seq=action.command_seq,
        action_sha256=hashlib.sha256(canonical_json_bytes(action.to_dict())).hexdigest(),
        queue_coalesced=0,
        queue_overruns=0,
        compute_ns=100,
        serialization_ns=20,
        write_ns=30,
    )


def _record_control(
    controller: ConnectomeController,
    operation: ControlOperation,
    *,
    sequence: int,
    issued_tick: int,
) -> ControlCommand:
    return ControlCommand(
        identity=controller.identity,
        control_seq=sequence,
        issued_tick=issued_tick,
        operation=operation,
        content_id="minecraft-replay-pipeline",
    )


def _pipelined_receipt_trace(path: Path) -> None:
    graph = load_malecns_fixture(FIXTURE)
    controller = ConnectomeController.create(
        graph,
        run_id="minecraft-replay-pipeline",
        entity_id="00000000-0000-0000-0000-000000000002",
    )
    forces = DeclaredTransitionForces.create(
        passive_forces=(0.0, -HarnessWorld.GRAVITY, 0.0),
        external_forces=(0.0, 0.0, 0.0),
        provenance="integration_test_declared_gravity_v1",
        arena_material_hash="e" * 64,
    )
    base_manifest = build_manifest(controller, graph)
    manifest = replace(
        base_manifest,
        claim_scope=ClaimScope.G3,
        artifacts=replace(
            base_manifest.artifacts,
            arena_hash=forces.identity_sha256,
            code_hash=_code_bundle()[0],
            dependency_lock_hash=_dependency_bundle()[0],
        ),
        java="21",
        minecraft="1.21.5",
        fabric="loader-0.16.10-api-0.128.2+1.21.5",
        mod="neurocraft-fly-0.1.0-test",
    )
    recorder = MinecraftReceiptTraceRecorder(
        controller,
        TraceWriter(path, manifest),
        forces=forces,
    )
    recorder.handle_control(
        _record_control(
            controller,
            ControlOperation.RECORD_START,
            sequence=0,
            issued_tick=0,
        )
    )

    world = HarnessWorld()
    actions: list[Action] = []
    # The bridge samples tick T before applying the action targeting tick T, and under the
    # two-tick budget that is the action computed from observation T - 2.
    for tick in range(7):
        observation = world.observe(controller, tick)
        action = controller.step(observation, command_seq=tick)
        _accept_with_test_telemetry(recorder, action)
        actions.append(action)
        applied = tick - ACTION_TARGET_OFFSET_TICKS
        if applied >= 0:
            world.apply(actions[applied].motors)
            recorder.accept_receipt(_receipt(actions[applied], observation))

    recorder.handle_control(
        _record_control(
            controller,
            ControlOperation.RECORD_STOP,
            sequence=1,
            issued_tick=5,
        )
    )
    terminal_observation = world.observe(controller, 7)
    terminal_action = controller.step(terminal_observation, command_seq=7)
    _accept_with_test_telemetry(recorder, terminal_action)
    assert recorder.state.value == "CLOSED"


def _rewrite_with_tampered_world_after(source: Path, destination: Path) -> None:
    verification = verify_trace(source)
    records = tuple(iter_records(source))
    with TraceWriter(destination, verification.manifest) as writer:
        for record in records:
            if record.record_seq == 0:
                x, y, z = record.world_after.position_m
                record = replace(
                    record,
                    world_after=replace(record.world_after, position_m=(x + 1.0, y, z)),
                )
            writer.append(record)


def _rewrite_with_nonzero_initial_state(source: Path, destination: Path) -> None:
    verification = verify_trace(source)
    records = tuple(iter_records(source))
    with TraceWriter(destination, verification.manifest) as writer:
        for record in records:
            if record.record_seq == 0:
                activity = list(record.state_before_activity or ())
                activity[0] = 0.125
                record = replace(record, state_before_activity=tuple(activity))
            writer.append(record)


def test_minecraft_labelled_trace_replays_neural_state_exactly(tmp_path: Path) -> None:
    trace = tmp_path / "minecraft.ndjson"
    _minecraft_labelled_trace(trace)

    receipt = replay_minecraft_trace(fixture=FIXTURE, trace=trace, stimulations={})

    assert receipt["status"] == "exact_neural_match"
    # Four records rather than three: an action applies one action budget after the
    # observation it answers, so a fourth record is what keeps two world-before links and
    # one world-after link inside the trace, exactly as three did under the one-tick budget.
    assert receipt["record_count"] == 4
    assert receipt["stimulated_record_count"] == 0
    assert receipt["zero_recurrence_counterfactual"]["state_divergent_record_count"] >= 1
    assert receipt["zero_recurrence_counterfactual"]["action_divergent_record_count"] >= 1
    assert receipt["world_before_cross_linked"] == 2
    assert receipt["terminal_world_before_receipt_only"] == 2
    assert receipt["world_transitions_cross_linked"] == 1
    assert receipt["terminal_world_transitions_not_cross_linked"] == 3
    assert "Minecraft physics is not re-simulated" in receipt["evidence_boundary"]


def test_pipelined_receipt_trace_cross_links_exact_later_observations(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "minecraft-pipelined.ndjson"
    _pipelined_receipt_trace(trace)
    records = tuple(iter_records(trace))

    # Record j answers observation j, is applied at j + ACTION_TARGET_OFFSET_TICKS, and its
    # world-after is the observation one tick later still: the cross-links reach further down
    # the trace than they did under the one-tick budget.
    assert len(records) == 4
    assert records[0].world_after_observation_seq == records[3].observation.observation_seq
    assert records[0].world_after_game_tick == records[3].observation.game_tick
    assert records[0].world_before == world_state_from_observation(records[2].observation)
    assert records[0].world_after == world_state_from_observation(records[3].observation)
    assert records[0].world_after == records[1].world_before
    assert records[1].world_before == world_state_from_observation(records[3].observation)

    receipt = replay_minecraft_trace(fixture=FIXTURE, trace=trace, stimulations={})

    assert receipt["status"] == "exact_neural_match"
    assert receipt["world_before_cross_linked"] == 2
    assert receipt["terminal_world_before_receipt_only"] == 2
    assert receipt["world_transitions_cross_linked"] == 1
    assert receipt["terminal_world_transitions_not_cross_linked"] == 3


def test_wing_profile_replays_a_steering_trace_exactly(tmp_path: Path) -> None:
    """The opt-in profile must rebuild the flight controller (steering pin,
    wing-motor pools, flight zero-recurrence configuration) and reproduce the
    trace record-for-record, mirroring the identity pins in
    ``tests/controller/test_flight_controller.py`` at the replay boundary."""

    trace = tmp_path / "minecraft-wing.ndjson"
    _wing_labelled_trace(trace)

    receipt = replay_minecraft_trace(
        fixture=STEERING_FIXTURE,
        trace=trace,
        stimulations={},
        controller_profile=WING_CONTROLLER_PROFILE,
    )

    assert receipt["status"] == "exact_neural_match"
    assert receipt["record_count"] == 4
    assert receipt["stimulated_record_count"] == 0
    assert receipt["zero_recurrence_counterfactual"]["state_divergent_record_count"] >= 1
    assert receipt["zero_recurrence_counterfactual"]["action_divergent_record_count"] >= 1


def test_trm_profile_replays_a_trm_trace_exactly(tmp_path: Path) -> None:
    """The opt-in trained-readout profile must rebuild the TRM demo controller
    (steering pin, cloned readout bound to the committed artifact digest) and
    reproduce the trace record-for-record, including a diverging
    zero-recurrence counterfactual."""

    trace = tmp_path / "minecraft-trm.ndjson"
    _trm_labelled_trace(trace)

    receipt = replay_minecraft_trace(
        fixture=STEERING_FIXTURE,
        trace=trace,
        stimulations={},
        controller_profile=TRM_CONTROLLER_PROFILE,
    )

    assert receipt["status"] == "exact_neural_match"
    assert receipt["record_count"] == 4
    assert receipt["stimulated_record_count"] == 0
    assert receipt["zero_recurrence_counterfactual"]["state_divergent_record_count"] >= 1
    assert receipt["zero_recurrence_counterfactual"]["action_divergent_record_count"] >= 1


def test_wing_profile_refuses_a_trm_trace(tmp_path: Path) -> None:
    """A TRM trace under the wing profile must fail the identity gate instead
    of silently replaying through the wrong decoder."""

    trace = tmp_path / "minecraft-trm.ndjson"
    _trm_labelled_trace(trace)

    with pytest.raises(ValueError, match="identities do not match"):
        replay_minecraft_trace(
            fixture=STEERING_FIXTURE,
            trace=trace,
            stimulations={},
            controller_profile=WING_CONTROLLER_PROFILE,
        )


def test_default_profile_refuses_a_wing_trace(tmp_path: Path) -> None:
    """A wing trace under the default pair profile must fail the identity gate
    instead of silently replaying through the wrong controller."""

    trace = tmp_path / "minecraft-wing.ndjson"
    _wing_labelled_trace(trace)

    with pytest.raises(ValueError, match="identities do not match"):
        replay_minecraft_trace(fixture=FIXTURE, trace=trace, stimulations={})


def test_replay_rejects_hash_authentic_tampered_cross_link(tmp_path: Path) -> None:
    source = tmp_path / "minecraft-pipelined.ndjson"
    tampered = tmp_path / "minecraft-pipelined-tampered.ndjson"
    _pipelined_receipt_trace(source)
    _rewrite_with_tampered_world_after(source, tampered)

    with pytest.raises(ValueError, match="world-after lineage mismatch at record 0"):
        replay_minecraft_trace(fixture=FIXTURE, trace=tampered, stimulations={})


def test_replay_rejects_nonzero_pre_record_controller_state(tmp_path: Path) -> None:
    source = tmp_path / "minecraft.ndjson"
    tampered = tmp_path / "minecraft-nonzero-start.ndjson"
    _minecraft_labelled_trace(source)
    _rewrite_with_nonzero_initial_state(source, tampered)

    with pytest.raises(ValueError, match="acknowledged pre-spawn recording boundary"):
        replay_minecraft_trace(fixture=FIXTURE, trace=tampered, stimulations={})


def test_replay_rejects_undeclared_intervention(tmp_path: Path) -> None:
    trace = tmp_path / "undeclared.ndjson"
    _minecraft_labelled_trace(trace, intervention="unknown-stimulation")

    with pytest.raises(ValueError, match="undeclared intervention"):
        replay_minecraft_trace(fixture=FIXTURE, trace=trace, stimulations={})


def test_replay_receipt_is_no_clobber(tmp_path: Path) -> None:
    receipt = tmp_path / "replay.json"
    value = {"schema": "test"}
    write_receipt(receipt, value)
    with pytest.raises(FileExistsError):
        write_receipt(receipt, value)
