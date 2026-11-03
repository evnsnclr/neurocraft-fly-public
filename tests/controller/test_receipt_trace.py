from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from neurocraft_connectome import RuntimeGraph
from neurocraft_controller import (
    ConnectomeController,
    ControllerIntegrityError,
    DeclaredTransitionForces,
    MinecraftReceiptTraceRecorder,
)
from neurocraft_controller.config import canonical_sha256
from neurocraft_protocol import (
    Action,
    ActuationReceipt,
    ActuationValidation,
    ArtifactIdentities,
    ClaimScope,
    ControlCommand,
    ControlOperation,
    DataTier,
    DeterminismLevel,
    FloatPrecision,
    MotorVector,
    TraceManifest,
    TraceWriter,
    WorldFrame,
    verify_trace,
)

from .conftest import make_controller, make_observation


def _forces() -> DeclaredTransitionForces:
    return DeclaredTransitionForces.create(
        passive_forces=(0.0, -0.04, 0.0),
        external_forces=(0.0, 0.0, 0.0),
        provenance="test_arena_declared_gravity_v1",
        arena_material_hash="c" * 64,
    )


def _manifest(
    controller: ConnectomeController,
    forces: DeclaredTransitionForces,
) -> TraceManifest:
    graph = controller.graph
    return TraceManifest(
        run_id=controller.identity.run_id,
        initial_session_epoch=controller.identity.session_epoch,
        mode=controller.run_mode,
        data_tier=DataTier.MALECNS_FIXTURE,
        claim_scope=ClaimScope.G0,
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
        mod="neurocraft-fly-test",
        privacy_policy="local_trace_no_secrets_v1",
        expected_chunks=1,
        checkpoint_interval_ticks=100,
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


def _record_control(
    controller: ConnectomeController,
    operation: ControlOperation,
    sequence: int,
    *,
    issued_tick: int | None = None,
) -> ControlCommand:
    return ControlCommand(
        identity=controller.identity,
        control_seq=sequence,
        issued_tick=sequence if issued_tick is None else issued_tick,
        operation=operation,
        content_id="minecraft-trace-1",
    )


def _telemetry(recorder: MinecraftReceiptTraceRecorder, action: Action, **changes) -> None:
    values = {
        "command_seq": action.command_seq,
        "action_sha256": canonical_sha256(action.to_dict()),
        "queue_coalesced": 0,
        "queue_overruns": 0,
        "compute_ns": 10,
        "serialization_ns": 20,
        "write_ns": 30,
    }
    recorder.accept_action_telemetry(**(values | changes))


@pytest.mark.parametrize("outside_before_stop", [False, True])
@pytest.mark.parametrize("receipt_first", [False, True])
def test_record_stop_detached_callbacks_accept_both_orders_without_losing_binding(
    controller_graph: RuntimeGraph,
    tmp_path: Path,
    outside_before_stop: bool,
    receipt_first: bool,
) -> None:
    controller = make_controller(controller_graph, run_id="stop-delayed-telemetry")
    forces = _forces()
    output = tmp_path / "delayed-telemetry.ndjson"
    recorder = MinecraftReceiptTraceRecorder(
        controller,
        TraceWriter(output, replace(_manifest(controller, forces), claim_scope=ClaimScope.G3)),
        forces=forces,
    )
    recorder.handle_control(_record_control(controller, ControlOperation.RECORD_START, 0))
    included = controller.step(make_observation(controller, 0), command_seq=0)
    recorder.accept_issued_action(included)
    recorder.accept_receipt(_receipt(included))
    _telemetry(recorder, included)
    outside = controller.step(make_observation(controller, 1), command_seq=1)
    if outside_before_stop:
        recorder.accept_issued_action(outside)
    recorder.handle_control(
        _record_control(controller, ControlOperation.RECORD_STOP, 1, issued_tick=2)
    )
    if not outside_before_stop:
        recorder.accept_issued_action(outside)
    assert recorder.state.value == "STOPPING"

    if receipt_first:
        recorder.accept_receipt(_receipt(outside))
        _telemetry(recorder, outside)
    else:
        _telemetry(recorder, outside)
        recorder.accept_receipt(_receipt(outside))
    assert recorder.state.value == "STOPPING"  # included action still needs world-after
    assert not recorder._detached_after_stop

    later = controller.step(make_observation(controller, 2), command_seq=2)
    recorder.accept_issued_action(later)
    recorder.accept_receipt(_receipt(later))
    _telemetry(recorder, later)
    latest = controller.step(make_observation(controller, 3), command_seq=3)
    recorder.accept_issued_action(latest)
    assert recorder.state.value == "CLOSED"
    assert [binding.command_seq for binding in recorder.bindings] == [0]
    assert verify_trace(output).record_count == 1


@pytest.mark.parametrize("before_stop", ["receipt", "telemetry", "both"])
def test_detaching_preserves_callbacks_already_accepted_before_stop(
    controller_graph: RuntimeGraph, tmp_path: Path, before_stop: str
) -> None:
    controller = make_controller(controller_graph, run_id="stop-preserves-callbacks")
    forces = _forces()
    recorder = MinecraftReceiptTraceRecorder(
        controller,
        TraceWriter(
            tmp_path / "before-stop.ndjson",
            replace(_manifest(controller, forces), claim_scope=ClaimScope.G3),
        ),
        forces=forces,
    )
    recorder.handle_control(_record_control(controller, ControlOperation.RECORD_START, 0))
    included = controller.step(make_observation(controller, 0), command_seq=0)
    recorder.accept_issued_action(included)
    recorder.accept_receipt(_receipt(included))
    outside = controller.step(make_observation(controller, 1), command_seq=1)
    recorder.accept_issued_action(outside)
    if before_stop in {"receipt", "both"}:
        recorder.accept_receipt(_receipt(outside))
    if before_stop in {"telemetry", "both"}:
        _telemetry(recorder, outside)
    recorder.handle_control(
        _record_control(controller, ControlOperation.RECORD_STOP, 1, issued_tick=2)
    )
    if before_stop == "receipt":
        _telemetry(recorder, outside)
    if before_stop == "telemetry":
        recorder.accept_receipt(_receipt(outside))
    assert not recorder._detached_after_stop
    assert recorder.pending_state.actions_without_operational_telemetry == (0,)
    recorder.abort()


def test_detached_and_in_bound_telemetry_keep_hash_unknown_and_duplicate_validation(
    controller_graph: RuntimeGraph, tmp_path: Path
) -> None:
    controller = make_controller(controller_graph, run_id="stop-telemetry-validation")
    forces = _forces()
    recorder = MinecraftReceiptTraceRecorder(
        controller,
        TraceWriter(
            tmp_path / "validation.ndjson",
            replace(_manifest(controller, forces), claim_scope=ClaimScope.G3),
        ),
        forces=forces,
    )
    recorder.handle_control(_record_control(controller, ControlOperation.RECORD_START, 0))
    included = controller.step(make_observation(controller, 0), command_seq=0)
    recorder.accept_issued_action(included)
    recorder.handle_control(
        _record_control(controller, ControlOperation.RECORD_STOP, 1, issued_tick=2)
    )
    outside = controller.step(make_observation(controller, 1), command_seq=1)
    recorder.accept_issued_action(outside)
    for action in (included, outside):
        with pytest.raises(ControllerIntegrityError, match="does not bind"):
            _telemetry(recorder, action, action_sha256="0" * 64)
        with pytest.raises(ControllerIntegrityError, match="measured and positive"):
            _telemetry(recorder, action, write_ns=0)
        _telemetry(recorder, action)
        with pytest.raises(ControllerIntegrityError, match="already recorded"):
            _telemetry(recorder, action)
    with pytest.raises(ControllerIntegrityError, match="no pending issued action"):
        _telemetry(recorder, outside, command_seq=99)
    forged = replace(
        _receipt(outside),
        requested_motors=MotorVector(forward_thrust=0.5),
        applied_motors=MotorVector(forward_thrust=0.5),
    )
    with pytest.raises(ControllerIntegrityError, match="does not bind issued action"):
        recorder.accept_receipt(forged)
    recorder.accept_receipt(_receipt(outside))
    with pytest.raises(ControllerIntegrityError):
        recorder.accept_receipt(_receipt(outside))
    with pytest.raises(ControllerIntegrityError, match="no pending issued action"):
        _telemetry(recorder, outside)

    later = controller.step(make_observation(controller, 2), command_seq=2)
    recorder.accept_issued_action(later)
    recorder.accept_receipt(_receipt(later))
    with pytest.raises(ControllerIntegrityError, match="already recorded"):
        recorder.accept_receipt(_receipt(later))
    with pytest.raises(ControllerIntegrityError, match="does not bind"):
        _telemetry(recorder, later, action_sha256="0" * 64)
    _telemetry(recorder, later)
    assert not recorder._detached_after_stop
    assert recorder.pending_state.issued_without_receipt == (0,)
    recorder.abort()


@pytest.mark.parametrize("missing_callback", ["receipt", "telemetry"])
def test_detached_callback_backlog_is_bounded(
    controller_graph: RuntimeGraph, tmp_path: Path, missing_callback: str
) -> None:
    controller = make_controller(controller_graph, run_id="stop-bounded-callbacks")
    forces = _forces()
    recorder = MinecraftReceiptTraceRecorder(
        controller,
        TraceWriter(
            tmp_path / "bounded.ndjson",
            replace(_manifest(controller, forces), claim_scope=ClaimScope.G3),
        ),
        forces=forces,
    )
    recorder.handle_control(_record_control(controller, ControlOperation.RECORD_START, 0))
    included = controller.step(make_observation(controller, 0), command_seq=0)
    recorder.accept_issued_action(included)
    recorder.handle_control(
        _record_control(controller, ControlOperation.RECORD_STOP, 1, issued_tick=2)
    )
    for sequence in range(1, recorder.max_pending + 1):
        outside = controller.step(make_observation(controller, sequence), command_seq=sequence)
        recorder.accept_issued_action(outside)
        if missing_callback == "receipt":
            _telemetry(recorder, outside)
        else:
            recorder.accept_receipt(_receipt(outside))
    assert len(recorder._detached_after_stop) == recorder.max_pending
    sequence = recorder.max_pending + 1
    overflow = controller.step(make_observation(controller, sequence), command_seq=sequence)
    with pytest.raises(ControllerIntegrityError, match="detached window is full"):
        recorder.accept_issued_action(overflow)
    assert len(recorder._detached_after_stop) == recorder.max_pending
    recorder.abort()


def test_receipt_and_next_observation_are_both_required_before_trace_append(
    controller_graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    controller = make_controller(controller_graph, run_id="minecraft-receipt-trace")
    output = tmp_path / "minecraft.ndjson"
    forces = _forces()
    recorder = MinecraftReceiptTraceRecorder(
        controller,
        TraceWriter(output, _manifest(controller, forces)),
        forces=forces,
    )
    recorder.handle_control(_record_control(controller, ControlOperation.RECORD_START, 0))

    observation_0 = make_observation(controller, 0)
    action_0 = controller.step(observation_0, command_seq=0)
    expected_hash = controller.last_evidence.action_sha256 if controller.last_evidence else ""
    recorder.accept_issued_action(action_0)
    assert recorder.record_count == 0
    recorder.accept_receipt(_receipt(action_0))
    assert recorder.record_count == 0
    assert recorder.pending_state.receipts_waiting_for_world_after == (0,)

    observation_1 = make_observation(controller, 1)
    action_1 = controller.step(observation_1, command_seq=1)
    recorder.accept_issued_action(action_1)
    assert recorder.record_count == 0  # tick 1 is sampled before action 0 is applied
    recorder.accept_receipt(_receipt(action_1))

    # Under the two-tick action budget action 0 answers observation 0 and applies at
    # tick 2, so the world observation that proves the transition is observation 3, and
    # the record-stop boundary that keeps actions 0 and 1 (targets 2 and 3) is tick 3.
    recorder.handle_control(
        _record_control(controller, ControlOperation.RECORD_STOP, 1, issued_tick=3)
    )
    observation_2 = make_observation(controller, 2)
    action_2 = controller.step(observation_2, command_seq=2)
    recorder.accept_issued_action(action_2)
    assert recorder.record_count == 0  # tick 2 is the tick action 0 applies ON, not after
    recorder.accept_receipt(_receipt(action_2))
    observation_3 = make_observation(controller, 3)
    action_3 = controller.step(observation_3, command_seq=3)
    recorder.accept_issued_action(action_3)
    assert recorder.record_count == 1
    assert recorder.bindings[0].action_sha256 == expected_hash
    recorder.accept_receipt(_receipt(action_3))  # detached: outside the recorded interval
    observation_4 = make_observation(controller, 4)
    action_4 = controller.step(observation_4, command_seq=4)
    recorder.accept_issued_action(action_4)
    assert recorder.state.value == "CLOSED"
    recorder.accept_receipt(_receipt(action_4))  # detached: outside the recorded interval
    verification = verify_trace(output)
    assert verification.record_count == 2
    assert verification.first_tick == 0
    assert verification.last_tick == 1


def test_record_start_detaches_prestart_in_flight_receipt_and_traces_after_floor(
    controller_graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    controller = make_controller(controller_graph, run_id="minecraft-receipt-start-floor")
    output = tmp_path / "start-floor.ndjson"
    forces = _forces()
    recorder = MinecraftReceiptTraceRecorder(
        controller,
        TraceWriter(output, _manifest(controller, forces)),
        forces=forces,
    )

    before_start = controller.step(make_observation(controller, 0), command_seq=0)
    recorder.accept_issued_action(before_start)
    recorder.handle_control(_record_control(controller, ControlOperation.RECORD_START, 0))
    recorder.accept_receipt(_receipt(before_start))
    assert recorder.pending_state.is_clear
    assert recorder.record_count == 0

    after_start = controller.step(make_observation(controller, 1), command_seq=1)
    recorder.accept_issued_action(after_start)
    recorder.accept_receipt(_receipt(after_start))
    # The boundary keeps the action targeting tick 3 (observation 1 plus the budget).
    recorder.handle_control(
        _record_control(controller, ControlOperation.RECORD_STOP, 1, issued_tick=3)
    )

    after_stop = controller.step(make_observation(controller, 2), command_seq=2)
    recorder.accept_issued_action(after_stop)
    recorder.accept_receipt(_receipt(after_stop))
    later = controller.step(make_observation(controller, 3), command_seq=3)
    recorder.accept_issued_action(later)
    recorder.accept_receipt(_receipt(later))
    latest = controller.step(make_observation(controller, 4), command_seq=4)
    recorder.accept_issued_action(latest)

    assert recorder.state.value == "CLOSED"
    assert recorder.record_count == 1
    assert [binding.command_seq for binding in recorder.bindings] == [1]
    verification = verify_trace(output)
    assert verification.record_count == 1
    assert verification.first_tick == 1
    assert verification.last_tick == 1


def test_post_start_receipt_without_issued_action_is_not_detached(
    controller_graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    controller = make_controller(controller_graph, run_id="minecraft-receipt-post-start-missing")
    forces = _forces()
    recorder = MinecraftReceiptTraceRecorder(
        controller,
        TraceWriter(tmp_path / "post-start-missing.ndjson", _manifest(controller, forces)),
        forces=forces,
    )

    before_start = controller.step(make_observation(controller, 0), command_seq=0)
    recorder.accept_issued_action(before_start)
    recorder.handle_control(_record_control(controller, ControlOperation.RECORD_START, 0))
    after_start = controller.step(make_observation(controller, 1), command_seq=1)

    with pytest.raises(ControllerIntegrityError, match="no pending issued action"):
        recorder.accept_receipt(_receipt(after_start))
    assert recorder.record_count == 0
    recorder.abort()


def test_forged_and_out_of_order_receipts_never_append(
    controller_graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    controller = make_controller(controller_graph, run_id="minecraft-receipt-forgery")
    forces = _forces()
    recorder = MinecraftReceiptTraceRecorder(
        controller,
        TraceWriter(tmp_path / "forgery.ndjson", _manifest(controller, forces)),
        forces=forces,
    )
    recorder.handle_control(_record_control(controller, ControlOperation.RECORD_START, 0))
    action_0 = controller.step(make_observation(controller, 0), command_seq=0)
    recorder.accept_issued_action(action_0)
    action_1 = controller.step(make_observation(controller, 1), command_seq=1)
    recorder.accept_issued_action(action_1)

    with pytest.raises(ControllerIntegrityError, match="oldest"):
        recorder.accept_receipt(_receipt(action_1))
    forged = replace(
        _receipt(action_0),
        requested_motors=MotorVector(forward_thrust=0.5),
        applied_motors=MotorVector(forward_thrust=0.5),
    )
    with pytest.raises(ControllerIntegrityError, match="oldest"):
        recorder.accept_receipt(forged)
    assert recorder.record_count == 0
    assert recorder.pending_state.issued_without_receipt == (0, 1)
    recorder.abort()


def test_receipt_binds_oldest_unreceipted_action_while_older_waits_for_world_after(
    controller_graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    controller = make_controller(controller_graph, run_id="minecraft-receipt-pipeline")
    output = tmp_path / "receipt-pipeline.ndjson"
    forces = _forces()
    recorder = MinecraftReceiptTraceRecorder(
        controller,
        TraceWriter(output, _manifest(controller, forces)),
        forces=forces,
    )
    recorder.handle_control(_record_control(controller, ControlOperation.RECORD_START, 0))

    action_0 = controller.step(make_observation(controller, 0), command_seq=0)
    recorder.accept_issued_action(action_0)
    action_1 = controller.step(make_observation(controller, 1), command_seq=1)
    recorder.accept_issued_action(action_1)
    recorder.accept_receipt(_receipt(action_0))
    assert recorder.pending_state.receipts_waiting_for_world_after == (0,)
    assert recorder.pending_state.issued_without_receipt == (1,)

    # This is the live observation-first pipeline: action 0 is still the oldest
    # pending entry, but it already has its receipt. Receipt 1 must bind action 1.
    recorder.accept_receipt(_receipt(action_1))
    after_second_receipt = recorder.pending_state
    assert not after_second_receipt.issued_without_receipt
    recorder.handle_control(
        _record_control(controller, ControlOperation.RECORD_STOP, 1, issued_tick=3)
    )

    # One more tick than the one-tick budget needed: action 1 applies at tick 3, so the
    # observation that proves its transition is observation 4.
    for sequence in (2, 3, 4):
        outside = controller.step(make_observation(controller, sequence), command_seq=sequence)
        recorder.accept_issued_action(outside)
        recorder.accept_receipt(_receipt(outside))

    assert recorder.state.value == "CLOSED"
    assert [binding.command_seq for binding in recorder.bindings] == [0, 1]
    assert verify_trace(output).record_count == 2


def test_record_stop_detaches_only_actions_beyond_tick_boundary_and_checks_receipts(
    controller_graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    controller = make_controller(controller_graph, run_id="minecraft-receipt-stop-boundary")
    output = tmp_path / "stop-boundary.ndjson"
    forces = _forces()
    recorder = MinecraftReceiptTraceRecorder(
        controller,
        TraceWriter(output, _manifest(controller, forces)),
        forces=forces,
    )
    recorder.handle_control(_record_control(controller, ControlOperation.RECORD_START, 0))

    included = controller.step(make_observation(controller, 0), command_seq=0)
    recorder.accept_issued_action(included)
    recorder.accept_receipt(_receipt(included))
    outside = controller.step(make_observation(controller, 1), command_seq=1)
    recorder.accept_issued_action(outside)
    # Tick 2 is the boundary that includes command 0 (observation 0 plus the budget) and
    # excludes command 1, which targets 3.
    recorder.handle_control(
        _record_control(controller, ControlOperation.RECORD_STOP, 1, issued_tick=2)
    )
    assert recorder.state.value == "STOPPING"

    forged = replace(
        _receipt(outside),
        requested_motors=MotorVector(forward_thrust=0.5),
        applied_motors=MotorVector(forward_thrust=0.5),
    )
    with pytest.raises(ControllerIntegrityError, match="issued action"):
        recorder.accept_receipt(forged)
    recorder.accept_receipt(_receipt(outside))

    later = controller.step(make_observation(controller, 2), command_seq=2)
    recorder.accept_issued_action(later)
    recorder.accept_receipt(_receipt(later))
    latest = controller.step(make_observation(controller, 3), command_seq=3)
    recorder.accept_issued_action(latest)
    assert recorder.state.value == "CLOSED"
    recorder.accept_receipt(_receipt(latest))
    assert [binding.command_seq for binding in recorder.bindings] == [0]
    assert verify_trace(output).record_count == 1


def test_record_stop_waits_for_late_in_bound_receipt_and_later_world_observation(
    controller_graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    controller = make_controller(controller_graph, run_id="minecraft-receipt-stop-wait")
    output = tmp_path / "stop-wait.ndjson"
    forces = _forces()
    recorder = MinecraftReceiptTraceRecorder(
        controller,
        TraceWriter(output, _manifest(controller, forces)),
        forces=forces,
    )
    recorder.handle_control(_record_control(controller, ControlOperation.RECORD_START, 0))
    included = controller.step(make_observation(controller, 0), command_seq=0)
    recorder.accept_issued_action(included)
    recorder.handle_control(
        _record_control(controller, ControlOperation.RECORD_STOP, 1, issued_tick=2)
    )

    for sequence in (1, 2, 3):
        outside = controller.step(make_observation(controller, sequence), command_seq=sequence)
        recorder.accept_issued_action(outside)
    assert recorder.state.value == "STOPPING"
    assert recorder.pending_state.issued_without_receipt == (0,)

    recorder.accept_receipt(_receipt(included))
    assert recorder.state.value == "CLOSED"
    assert [binding.command_seq for binding in recorder.bindings] == [0]
    assert verify_trace(output).record_count == 1


def test_missing_receipt_or_terminal_world_state_blocks_trace_publication(
    controller_graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    controller = make_controller(controller_graph, run_id="minecraft-receipt-missing")
    output = tmp_path / "missing.ndjson"
    forces = _forces()
    recorder = MinecraftReceiptTraceRecorder(
        controller,
        TraceWriter(output, _manifest(controller, forces)),
        forces=forces,
    )
    recorder.handle_control(_record_control(controller, ControlOperation.RECORD_START, 0))
    action = controller.step(make_observation(controller, 0), command_seq=0)
    recorder.accept_issued_action(action)
    with pytest.raises(ControllerIntegrityError, match="only after RECORD_STOP"):
        recorder.close()
    assert not output.exists()

    recorder.handle_control(
        _record_control(controller, ControlOperation.RECORD_STOP, 1, issued_tick=2)
    )
    terminal_action = controller.step(make_observation(controller, 1), command_seq=1)
    recorder.accept_issued_action(terminal_action)
    assert recorder.state.value == "STOPPING"
    with pytest.raises(ControllerIntegrityError, match="missing receipt"):
        recorder.close()
    assert not output.exists()
    recorder.abort()
    assert recorder.state.value == "ABORTED"


def test_recorder_cannot_default_unknown_minecraft_forces(
    controller_graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    controller = make_controller(controller_graph, run_id="minecraft-force-required")
    forces = _forces()
    writer = TraceWriter(tmp_path / "force-required.ndjson", _manifest(controller, forces))
    with pytest.raises(TypeError, match="forces"):
        MinecraftReceiptTraceRecorder(controller, writer)  # type: ignore[call-arg]
    writer.abort()
