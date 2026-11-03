"""Bounded synthetic controller controls; no full artifact or outcome claims."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from neurocraft_controller.controller import ControllerIntegrityError
from neurocraft_controller.full_brain_controller import (
    FullBrainConfiguration,
    FullBrainController,
    FullBrainLoomV2Configuration,
    FullBrainProfileError,
)
from neurocraft_controller.loom_senses_v2 import bind_embodied_senses_loom_v2
from neurocraft_controller.program_policy import PolicyState, ProgramSelectingController
from neurocraft_neural.full_controls import FullControlKind, FullControlView, build_control_view
from neurocraft_neural.full_engine import ArrayNativeEngine
from neurocraft_neural.full_view import FullRuntimeView
from neurocraft_neural.spmv_kernel import NumpyReduceatKernel
from neurocraft_protocol import (
    ProgramId,
    RunMode,
    SensorFrameV2,
    TraceManifest,
    canonical_json_bytes,
)

from tests.controller.test_full_brain_controller import (
    GRAPH_SHA,
    OVERLAY_SHA,
    TEST_SCALE,
    _calibration_bytes,
    _controller,
    _observation,
    _sensors,
    _synthetic_view,
)
from tests.controller.test_steering_readout import _loom_controllers
from tests.protocol.helpers import manifest


def control_view(view: FullRuntimeView, mode: str, manifest: str) -> FullControlView:
    # Both imported synthetic-view helpers declare alternating counts 3/5
    # and signs +/-1. The builder verifies those against the supplied view.
    return build_control_view(
        view,
        kind=FullControlKind(mode),
        seed=7,
        artifact_manifest_sha256=manifest,
        raw_counts=np.tile([3, 5], view.node_count),
        edge_signs=np.tile([1, -1], view.node_count),
    )


def clone(base: FullBrainController, mode: str) -> FullBrainController:
    control = (
        None
        if mode == "faithful"
        else control_view(base.engine.view, mode, base.identity.graph_hash)
    )
    return FullBrainController(
        run_id=base.identity.run_id,
        entity_id=base.identity.entity_id,
        configuration=replace(base.configuration, mode=mode),
        engine=ArrayNativeEngine(base.engine.view if control is None else control.view),
        sensory_binding=base._sensory_binding,
        motor_binding=base._motor_binding,
        embodied=base.embodied_senses,
        escape=base.escape_binding,
        steering=base.steering_binding,
        control=control,
    )


def loom_v2() -> FullBrainController:
    base, _ = _loom_controllers()
    assert base.embodied_senses is not None and base.escape_binding is not None
    senses = bind_embodied_senses_loom_v2(base.embodied_senses.populations)
    return FullBrainController(
        run_id=base.identity.run_id,
        entity_id=base.identity.entity_id,
        configuration=FullBrainLoomV2Configuration(
            encoder_identity_sha256=senses.encoder.identity_sha256
        ),
        engine=base.engine,
        sensory_binding=base._sensory_binding,
        motor_binding=base._motor_binding,
        embodied=senses,
        escape=replace(base.escape_binding, encoder_identity_sha256=senses.encoder.identity_sha256),
        steering=base.steering_binding,
    )


def typed_observation(controller: FullBrainController, tick: int):
    sensors = SensorFrameV2.from_sensor_frame(
        _sensors(forward=max(0.5, 6.0 - tick * 0.4)), (2,) * 6
    )
    return _observation(controller, tick, sensors=sensors)


def test_original_identity_and_twenty_tick_stream_match_prechange_measurement() -> None:
    controller = _controller()
    assert (
        controller.configuration.identity_sha256
        == "71f5eb37fc286811f9dd413a4577857ad7a44d58d17e9e264c1ddf678a662a54"
    )
    assert controller.control_identity is None and controller.control_kind == "faithful"
    assert controller.identity.dynamics_hash == controller.engine.dynamics_identity_sha256
    frames = [
        controller.step(
            _observation(controller, tick, sensors=_sensors(forward=6.0 - tick / 5)),
            command_seq=tick,
        ).to_dict()
        for tick in range(20)
    ]
    # Measured on the unmodified parent 0440d0e, same C-kernel identity.
    if (
        controller.engine.dynamics_identity_sha256
        == "5da651e72477a186c82a6091afa0f415b7e4a182fcd8f014870922143d526d8e"
    ):
        assert (
            sha256(canonical_json_bytes(frames)).hexdigest()
            == "7f69b7430a729a1f6fd03ced2bd72719c7204275798594d0fa8c3ff054dbb968"
        )
    assert controller.last_evidence is not None
    assert controller.last_evidence.encoder_stimulation_before_control is None


@pytest.mark.parametrize("mode", ["shuffled_weight", "no_input"])
def test_only_control_dynamics_hash_changes_and_policy_bindings_stay_fixed(mode: str) -> None:
    base = loom_v2()
    controlled = clone(base, mode)
    assert controlled.identity.dynamics_hash != base.identity.dynamics_hash
    assert replace(controlled.identity, dynamics_hash=base.identity.dynamics_hash) == base.identity
    assert controlled.engine.dynamics_identity_sha256 == base.engine.dynamics_identity_sha256
    assert controlled.run_mode == RunMode(mode)
    assert controlled.control_identity is not None
    assert controlled.control_identity.faithful_view_sha256 == base.engine.view.view_sha256
    assert controlled._motor_binding == base._motor_binding
    assert controlled.escape_binding == base.escape_binding
    assert (
        ProgramSelectingController(controlled).policy_hash
        == ProgramSelectingController(base).policy_hash
    )
    if mode == "no_input":
        assert controlled.engine.view is base.engine.view
        assert controlled.engine.current_state() == base.engine.current_state()
    else:
        assert controlled.engine.view.view_sha256 != base.engine.view.view_sha256
        assert controlled.engine.current_state() != base.engine.current_state()


def test_no_input_zeroes_final_dense_vector_and_records_the_real_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = clone(loom_v2(), "no_input")
    original_step = controller.engine.step
    observed = []

    def record(**kwargs):
        observed.append(kwargs["sensory"].copy())
        return original_step(**kwargs)

    monkeypatch.setattr(controller.engine, "step", record)
    digests = []
    for tick in range(4):
        controller.step(typed_observation(controller, tick), command_seq=tick)
        evidence = controller.last_evidence
        assert evidence is not None
        before = evidence.encoder_stimulation_before_control
        assert before is not None and before["eem:light"] == 1.0
        if tick > 0:
            assert before["eem:threat_front"] > 0.0
        assert set(evidence.encoded_stimulation.values()) == {0.0}
        assert "eem_raw:threat_front" not in evidence.encoded_stimulation
        assert evidence.executed_stimulation_sha256 is not None
        digests.append(evidence.executed_stimulation_sha256)
    assert len(observed) == 20 and all(np.count_nonzero(vector) == 0 for vector in observed)
    assert len(set(digests)) == 1
    assert np.count_nonzero(controller.engine.activity()) == 0


def test_control_pairing_refuses_missing_foreign_and_mislabeled_views() -> None:
    base = _controller()
    control = control_view(base.engine.view, "shuffled_weight", GRAPH_SHA)

    def build(mode, selected, view):
        return FullBrainController(
            run_id="refusal",
            entity_id="fly",
            configuration=FullBrainConfiguration(mode=mode),
            engine=ArrayNativeEngine(view),
            sensory_binding=base._sensory_binding,
            motor_binding=base._motor_binding,
            control=selected,
        )

    for mode, selected, view in (
        ("shuffled_weight", None, base.engine.view),
        ("faithful", None, control.view),
        ("faithful", control, control.view),
        ("shuffled_weight", control, base.engine.view),
        (
            "shuffled_weight",
            replace(
                control, identity=replace(control.identity, artifact_manifest_sha256="ff" * 32)
            ),
            control.view,
        ),
        (
            "shuffled_weight",
            replace(control, identity=replace(control.identity, seed=8)),
            control.view,
        ),
    ):
        with pytest.raises(FullBrainProfileError):
            build(mode, selected, view)


def test_session_reset_clears_inner_clocks_encoder_and_policy_but_normal_reset_does_not() -> None:
    wrapped = ProgramSelectingController(loom_v2())
    inner = wrapped.controller
    for tick in range(4):
        wrapped.step(typed_observation(inner, tick), command_seq=tick)
    wrapped.reset()
    with pytest.raises(ControllerIntegrityError, match="sequence"):
        wrapped.step(typed_observation(inner, 0), command_seq=0)
    wrapped._state = replace(
        PolicyState(), program=ProgramId.ESCAPE_JUMP, hold_remaining=3, armed=False
    )
    old_identity = wrapped.identity
    wrapped.reset_session(session_epoch=1)
    assert wrapped.policy_state == PolicyState() and wrapped.last_evidence is None
    assert inner.engine.step_index == 0 and inner._neural_step_offset == 0
    assert inner._eem_contrast_means is None
    assert replace(wrapped.identity, session_epoch=0) == old_identity
    with pytest.raises(ControllerIntegrityError, match="identity"):
        wrapped.step(replace(typed_observation(inner, 0), identity=old_identity), command_seq=0)
    fresh = ProgramSelectingController(loom_v2())
    fresh.reset_session(session_epoch=1)
    for tick in range(4):
        assert wrapped.step(typed_observation(inner, tick), command_seq=tick) == fresh.step(
            typed_observation(fresh.controller, tick), command_seq=tick
        )
    snapshot = inner.engine.snapshot()
    for epoch in (0, 1, True, 1.5):
        with pytest.raises(ControllerIntegrityError, match="strictly increase"):
            wrapped.reset_session(session_epoch=epoch)
        assert inner.engine.snapshot() == snapshot


def test_factory_keeps_calibration_on_original_boundary_for_all_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import neurocraft_connectome.full_runtime as runtime
    import neurocraft_controller.full_brain_controller as module
    import neurocraft_neural.full_signs as signs
    import neurocraft_neural.full_view as views
    from neurocraft_controller.flight_pools import (
        LEFT_WING_MOTOR_BODY_IDS,
        RIGHT_WING_MOTOR_BODY_IDS,
    )

    nodes = 80
    view = _synthetic_view(nodes)
    mapping = {805722: 0, 77185: 1}
    mapping.update(
        {body: i + 2 for i, body in enumerate(LEFT_WING_MOTOR_BODY_IDS + RIGHT_WING_MOTOR_BODY_IDS)}
    )
    closed = []
    graph = SimpleNamespace(
        manifest=SimpleNamespace(manifest_file_sha256=GRAPH_SHA, node_count=nodes),
        index_of=mapping.__getitem__,
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(runtime, "load_full_runtime_artifact", lambda *a, **kw: graph)
    monkeypatch.setattr(views, "view_from_full_runtime_graph", lambda *a, **kw: view)
    monkeypatch.setattr(
        signs,
        "compute_full_sign_overlay",
        lambda *a: SimpleNamespace(identity_sha256=OVERLAY_SHA, edge_signs=np.tile([1, -1], nodes)),
    )
    monkeypatch.setattr(module, "_verified_graph_array", lambda *a: np.tile([3, 5], nodes))
    monkeypatch.setattr(module, "load_kernel", lambda **kw: NumpyReduceatKernel())
    (tmp_path / "manifest.json").write_text("{}")
    annotations = tmp_path / "annotations.feather"
    annotations.write_bytes(b"")
    calibration = tmp_path / "calibration.json"
    calibration.write_bytes(_calibration_bytes())
    controllers = [
        FullBrainController.from_artifact(
            run_id="factory",
            entity_id="fly",
            artifact_directory=tmp_path,
            annotations_path=annotations,
            calibration_path=calibration,
            configuration=FullBrainConfiguration(mode=mode),
            control_permutation_directory=tmp_path / "controls",
        )
        for mode in ("faithful", "shuffled_weight", "no_input")
    ]
    assert len(closed) == 3
    assert len({c.identity.decoder_hash for c in controllers}) == 1
    assert len({c.identity.dynamics_hash for c in controllers}) == 3
    for controller in controllers:
        assert controller.readout_scale == TEST_SCALE
        assert (
            controller.configuration.readout_calibration_sha256
            == sha256(calibration.read_bytes()).hexdigest()
        )
    assert len(list((tmp_path / "controls").glob("*.u4"))) == 1


@pytest.mark.parametrize("seed", [True, -1, 1 << 64, 1.5])
def test_invalid_control_seed_is_refused_before_artifact_loading(seed) -> None:
    with pytest.raises(FullBrainProfileError, match="control_seed"):
        FullBrainController.from_artifact(
            run_id="test",
            entity_id="fly",
            artifact_directory=Path("/absent"),
            annotations_path=Path("/absent"),
            configuration=FullBrainConfiguration(mode="no_input"),
            control_seed=seed,
        )


def test_no_input_trace_vocabulary_is_additive() -> None:
    assert {mode.value for mode in RunMode} == {
        "faithful",
        "shuffled_weight",
        "degree_preserving_random",
        "lesion",
        "scripted_baseline",
        "synthetic",
        "no_input",
    }
    old = manifest()
    new = replace(old, mode=RunMode.NO_INPUT)
    assert TraceManifest.from_dict(new.to_dict()) == new
    assert TraceManifest.from_dict(old.to_dict()) == old


def test_control_probe_reports_observed_results_and_checks_restoration(
    tmp_path, monkeypatch
) -> None:
    import json

    from scripts import probe_program_controls as probe

    base = loom_v2()
    monkeypatch.setattr(
        probe, "build", lambda mode, args: ProgramSelectingController(clone(base, mode))
    )
    output = tmp_path / "probe.json"
    assert probe.main(["--trials", "1", "--json", str(output)]) == 0
    result = json.loads(output.read_text())
    assert all(result["checks"].values())
    assert result["response_difference_is_measured_not_required"]
    assert set(result["lanes"]) == {"original", "shuffled_weight", "no_input", "original_restored"}
    for lane in result["lanes"].values():
        for condition in lane["conditions"].values():
            for trial in condition["trials"]:
                assert len(bytes.fromhex(trial["program_action_stream_sha256"])) == 32


def test_control_probe_refuses_restoration_with_changed_state_digest(tmp_path, monkeypatch) -> None:
    import json

    from scripts import probe_program_controls as probe

    base = loom_v2()
    faithful_builds = 0

    def build(mode, args):
        nonlocal faithful_builds
        controller = ProgramSelectingController(clone(base, mode))
        if mode == "faithful":
            faithful_builds += 1
            if faithful_builds == 2:
                original_step = controller.step

                def changed_state(*args, **kwargs):
                    action = original_step(*args, **kwargs)
                    # Change only a field summarize() omits, keeping a valid hash.
                    replacement = "00" * 32 if action.state_digest != "00" * 32 else "11" * 32
                    return replace(action, state_digest=replacement)

                monkeypatch.setattr(controller, "step", changed_state)
        return controller

    monkeypatch.setattr(probe, "build", build)
    output = tmp_path / "changed-state-probe.json"
    assert probe.main(["--trials", "1", "--json", str(output)]) == 1
    result = json.loads(output.read_text())
    assert result["checks"] == {
        "fixed_calibration_readouts_policy": True,
        "original_restored_exactly": False,
        "distinct_control_dynamics": True,
    }
    original = result["lanes"]["original"]["conditions"]
    restored = result["lanes"]["original_restored"]["conditions"]
    for name in original:
        original_trial = original[name]["trials"][0]
        restored_trial = restored[name]["trials"][0]
        assert original_trial.pop("program_action_stream_sha256") != restored_trial.pop(
            "program_action_stream_sha256"
        )
        assert original_trial == restored_trial
