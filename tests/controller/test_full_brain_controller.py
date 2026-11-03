"""Tests for the opt-in full-MaleCNS wing-motor controller profile.

Two tiers:

* CI-safe: a small synthetic ``FullRuntimeView`` with a handful of nodes
  standing in for the sensory entries and wing-motor pools. No artifact, no
  annotations table, no compiled kernel required.
* Artifact-gated: skipped unless the local-only full artifact and the
  neurotransmitters table are present.
"""

from __future__ import annotations

import math
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from neurocraft_controller import ConnectomeController, ControllerConfiguration
from neurocraft_controller.controller import ControllerIntegrityError
from neurocraft_controller.full_brain_controller import (
    FULL_MCNS_ARTIFACT_MANIFEST_SHA256,
    FULL_READOUT_CALIBRATION_SCHEMA,
    MAX_READOUT_SCALE,
    BoundMotorPool,
    BoundPoolMember,
    FullBrainConfiguration,
    FullBrainController,
    FullBrainProfileError,
    FullMotorBinding,
    FullSensoryBinding,
    bind_sensor_channels,
    load_readout_calibration,
    projection_identity_sha256,
)
from neurocraft_neural.full_engine import ArrayNativeEngine
from neurocraft_neural.full_view import FullRuntimeView, ViewProvenance, build_view
from neurocraft_protocol import Observation, RunMode, SensorFrame, canonical_json_bytes

REPO_ROOT = Path(__file__).resolve().parents[2]
FULL_ARTIFACT_DIRECTORY = REPO_ROOT / "data" / "processed" / "male-cns-v1.0-superclass-v1"
ANNOTATIONS_PATH = (
    REPO_ROOT / "data" / "cache" / "male-cns-v1.0"
) / "body-neurotransmitters-male-cns-v1.0.feather"

CALIBRATION_PATH = REPO_ROOT / "data" / "calibration" / "full-readout-calibration-v1.json"

artifact_gated = pytest.mark.skipif(
    not (FULL_ARTIFACT_DIRECTORY / "manifest.json").is_file() or not ANNOTATIONS_PATH.is_file(),
    reason="full MaleCNS artifact and annotations table are local-only",
)

NODE_COUNT = 40
GRAPH_SHA = "ab" * 32
OVERLAY_SHA = "cd" * 32
ENTRY_INDEX = {805_722: 0, 77_185: 1}
LEFT_TEST_POOL: tuple[tuple[int, int], ...] = ((9001, 2), (9002, 3))
RIGHT_TEST_POOL: tuple[tuple[int, int], ...] = ((9101, 4), (9102, 5))


def _synthetic_view(node_count: int = NODE_COUNT) -> FullRuntimeView:
    """Deterministic two-in-degree CSR ring with mixed signs."""

    degrees = np.full(node_count, 2, dtype=np.int64)
    indptr = np.zeros(node_count + 1, dtype=np.int64)
    np.cumsum(degrees, out=indptr[1:])
    columns = np.empty(2 * node_count, dtype=np.uint32)
    counts = np.empty(2 * node_count, dtype=np.int64)
    signs = np.empty(2 * node_count, dtype=np.int64)
    for row in range(node_count):
        columns[2 * row] = (row + 1) % node_count
        columns[2 * row + 1] = (row + 7) % node_count
        counts[2 * row] = 3
        counts[2 * row + 1] = 5
        signs[2 * row] = 1
        signs[2 * row + 1] = -1
    return build_view(
        indptr=indptr,
        pre_indices=columns,
        raw_counts=counts,
        signs=signs,
        provenance=ViewProvenance(
            origin="synthetic_full_brain_test",
            graph_identity_sha256="11" * 32,
        ),
    )


def _sensory_binding(node_count: int = NODE_COUNT) -> FullSensoryBinding:
    return bind_sensor_channels(
        channels=ConnectomeController._sensor_channels(ControllerConfiguration()),
        index_of=ENTRY_INDEX.__getitem__,
        node_count=node_count,
        graph_manifest_sha256=GRAPH_SHA,
    )


def _motor_binding(
    *,
    gain: float = 8.0,
    dead_zone: float = 0.0,
    overlay_sha256: str = OVERLAY_SHA,
    right_pool: tuple[tuple[int, int], ...] = RIGHT_TEST_POOL,
    readout_scale: float = 1.0,
    readout_calibration_sha256: str = "",
) -> FullMotorBinding:
    return FullMotorBinding(
        graph_manifest_sha256=GRAPH_SHA,
        sign_overlay_identity_sha256=overlay_sha256,
        pools=(
            BoundMotorPool(
                "left_motor",
                gain,
                dead_zone,
                tuple(BoundPoolMember(body, index, 1.0) for body, index in LEFT_TEST_POOL),
            ),
            BoundMotorPool(
                "right_motor",
                gain,
                dead_zone,
                tuple(BoundPoolMember(body, index, 1.0) for body, index in right_pool),
            ),
        ),
        projection_sha256=projection_identity_sha256(),
        readout_scale=readout_scale,
        readout_calibration_sha256=readout_calibration_sha256,
    )


def _controller(
    *,
    run_id: str = "full-brain-test",
    motor_binding: FullMotorBinding | None = None,
    configuration: FullBrainConfiguration | None = None,
) -> FullBrainController:
    return FullBrainController(
        run_id=run_id,
        entity_id="test-fly",
        configuration=configuration,
        engine=ArrayNativeEngine(_synthetic_view()),
        sensory_binding=_sensory_binding(),
        motor_binding=motor_binding or _motor_binding(),
    )


CALIBRATION_SHA = "9a" * 32
#: A scale small enough that the synthetic view's tiny pool means stay well
#: inside the +-1 clip, so the tests can assert exact multiplication.
TEST_SCALE = 500.0


def _calibrated_controller(
    *,
    scale: float = TEST_SCALE,
    calibration_sha256: str = CALIBRATION_SHA,
    run_id: str = "full-brain-test",
) -> FullBrainController:
    return _controller(
        run_id=run_id,
        configuration=FullBrainConfiguration(
            readout_scale=scale, readout_calibration_sha256=calibration_sha256
        ),
        motor_binding=_motor_binding(
            readout_scale=scale, readout_calibration_sha256=calibration_sha256
        ),
    )


def _calibration_document(
    *,
    schema: str = FULL_READOUT_CALIBRATION_SCHEMA,
    manifest_sha256: str = GRAPH_SHA,
    overlay_sha256: str = OVERLAY_SHA,
    scale: float = TEST_SCALE,
    scale_hex: str | None = None,
) -> dict[str, Any]:
    """A minimal document with exactly the fields the loader reads."""

    return {
        "schema": schema,
        "stream": {"stream_id": "unit-test-stream-v1"},
        "artifact": {"manifest_file_sha256": manifest_sha256},
        "sign_overlay": {"identity_sha256": overlay_sha256},
        "calibration": {
            "readout_scale_float_hex": float(scale).hex() if scale_hex is None else scale_hex,
            "readout_scale": scale,
        },
    }


def _calibration_bytes(**kwargs: Any) -> bytes:
    return canonical_json_bytes(_calibration_document(**kwargs))


def _sensors(
    *, forward: float = 6.0, left: float = 6.0, right: float = 6.0, light: int = 15
) -> SensorFrame:
    return SensorFrame(
        position_m=(0.0, 64.0, 0.0),
        velocity_m_s=(0.0, 0.0, 0.0),
        yaw_rad=0.0,
        pitch_rad=0.0,
        on_ground=False,
        horizontal_collision=False,
        vertical_collision=False,
        in_fluid=False,
        local_light=light,
        ray_max_distance_m=6.0,
        ray_distances_m=(forward, 6.0, left, right, 6.0, 6.0),
    )


def _observation(
    controller: FullBrainController,
    sequence: int,
    *,
    sensors: SensorFrame | None = None,
    missing_fields: tuple[str, ...] = (),
) -> Observation:
    return Observation(
        identity=controller.identity,
        observation_seq=sequence,
        game_tick=sequence,
        sent_monotonic_ns=sequence * 50_000_000,
        sensors=sensors or _sensors(),
        missing_fields=missing_fields,
    )


def _varied_sensors(step: int) -> SensorFrame:
    return _sensors(
        forward=6.0 - 0.5 * (step % 8),
        left=1.0 + 0.4 * (step % 9),
        right=6.0 - 0.3 * (step % 11),
        light=(step * 5) % 16,
    )


# ---------------------------------------------------------------- configuration


def test_configuration_refuses_lesion_mode() -> None:
    with pytest.raises(ValueError, match="faithful mode only"):
        FullBrainConfiguration(mode="lesion")
    with pytest.raises(ValueError, match="must be faithful"):
        FullBrainConfiguration(mode="synthetic")


def test_configuration_identity_excludes_kernel_threads() -> None:
    assert (
        FullBrainConfiguration(kernel_threads=2).identity_sha256
        == FullBrainConfiguration().identity_sha256
    )
    assert (
        FullBrainConfiguration(motor_pool_gain=4.0).identity_sha256
        != FullBrainConfiguration().identity_sha256
    )


def test_configuration_refuses_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="five neural steps"):
        FullBrainConfiguration(neural_steps_per_game_tick=4)
    with pytest.raises(ValueError, match="kernel_threads"):
        FullBrainConfiguration(kernel_threads=0)
    with pytest.raises(ValueError, match="sign overlay"):
        FullBrainConfiguration(sign_policy="unknown_zero")


# ------------------------------------------------------- readout calibration


def test_configuration_pairs_the_scale_with_its_calibration_digest() -> None:
    """Both directions are refused: a scale with no document, a document with
    no scale. A calibrated readout must always name the bytes it came from."""

    with pytest.raises(ValueError, match="requires the lowercase SHA-256"):
        FullBrainConfiguration(readout_scale=TEST_SCALE)
    with pytest.raises(ValueError, match="requires the lowercase SHA-256"):
        FullBrainConfiguration(readout_scale=TEST_SCALE, readout_calibration_sha256="not-a-digest")
    with pytest.raises(ValueError, match="must carry no"):
        FullBrainConfiguration(readout_calibration_sha256=CALIBRATION_SHA)
    calibrated = FullBrainConfiguration(
        readout_scale=TEST_SCALE, readout_calibration_sha256=CALIBRATION_SHA
    )
    assert calibrated.readout_scale == TEST_SCALE


@pytest.mark.parametrize(
    "scale",
    [0.0, -1.0, float("nan"), float("inf"), MAX_READOUT_SCALE * 2.0],
)
def test_configuration_refuses_an_out_of_range_readout_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="readout_scale must be finite"):
        FullBrainConfiguration(readout_scale=scale, readout_calibration_sha256=CALIBRATION_SHA)


def test_configuration_identity_rotates_with_the_calibration() -> None:
    baseline = FullBrainConfiguration()
    calibrated = FullBrainConfiguration(
        readout_scale=TEST_SCALE, readout_calibration_sha256=CALIBRATION_SHA
    )
    other_document = FullBrainConfiguration(
        readout_scale=TEST_SCALE, readout_calibration_sha256="1c" * 32
    )
    other_scale = FullBrainConfiguration(
        readout_scale=TEST_SCALE + 1.0, readout_calibration_sha256=CALIBRATION_SHA
    )
    identities = {
        baseline.identity_sha256,
        calibrated.identity_sha256,
        other_document.identity_sha256,
        other_scale.identity_sha256,
    }
    assert len(identities) == 4
    assert (
        FullBrainConfiguration(
            readout_scale=TEST_SCALE, readout_calibration_sha256=CALIBRATION_SHA
        ).identity_sha256
        == calibrated.identity_sha256
    )


def test_decoder_identity_rotates_with_the_calibration() -> None:
    baseline = _controller()
    calibrated = _calibrated_controller()
    assert calibrated.identity.decoder_hash != baseline.identity.decoder_hash
    assert calibrated.identity.adapter_hash == baseline.identity.adapter_hash
    assert calibrated.identity.dynamics_hash == baseline.identity.dynamics_hash
    assert calibrated.identity.graph_hash == baseline.identity.graph_hash


def test_constructor_refuses_a_calibration_the_configuration_does_not_declare() -> None:
    with pytest.raises(FullBrainProfileError, match="readout calibration does not match"):
        _controller(
            motor_binding=_motor_binding(
                readout_scale=TEST_SCALE, readout_calibration_sha256=CALIBRATION_SHA
            )
        )
    with pytest.raises(FullBrainProfileError, match="readout calibration does not match"):
        _controller(
            configuration=FullBrainConfiguration(
                readout_scale=TEST_SCALE, readout_calibration_sha256=CALIBRATION_SHA
            )
        )


def test_readout_scale_multiplies_the_pool_value_before_the_clip() -> None:
    """The calibrated value is exactly the uncalibrated one times the scale,
    and exactly an independent recomputation from the committed activity."""

    baseline = _controller()
    calibrated = _calibrated_controller()
    assert calibrated.readout_scale == TEST_SCALE
    assert baseline.readout_scale == 1.0

    unclipped_nonzero = 0
    for step in range(6):
        baseline.step(_observation(baseline, step, sensors=_varied_sensors(step)), command_seq=step)
        calibrated.step(
            _observation(calibrated, step, sensors=_varied_sensors(step)), command_seq=step
        )
        plain_evidence = baseline.last_evidence
        scaled_evidence = calibrated.last_evidence
        assert plain_evidence is not None and scaled_evidence is not None
        plain = dict(plain_evidence.motor_frame.values)
        scaled = dict(scaled_evidence.motor_frame.values)
        activity = dict(
            zip(
                scaled_evidence.activity_subset_body_ids,
                scaled_evidence.state_after_activity,
                strict=True,
            )
        )
        for name, members in (
            ("left_motor", LEFT_TEST_POOL),
            ("right_motor", RIGHT_TEST_POOL),
        ):
            expected = 8.0 * math.fsum(activity[body] for body, _ in members) / len(members)
            expected = min(1.0, max(-1.0, expected * TEST_SCALE))
            assert scaled[name] == expected
            if abs(scaled[name]) < 1.0 and scaled[name] != 0.0:
                # Unclipped: the calibrated value is exactly the uncalibrated
                # one times the declared scale, with nothing else applied.
                assert scaled[name] == plain[name] * TEST_SCALE
                unclipped_nonzero += 1
    assert unclipped_nonzero, "the synthetic stream must exercise an unclipped nonzero readout"


def test_pool_readout_means_expose_the_pre_gain_quantity() -> None:
    controller = _controller()
    controller.step(_observation(controller, 0, sensors=_varied_sensors(0)), command_seq=0)
    controller.step(_observation(controller, 1, sensors=_varied_sensors(1)), command_seq=1)
    evidence = controller.last_evidence
    assert evidence is not None
    activity = dict(
        zip(evidence.activity_subset_body_ids, evidence.state_after_activity, strict=True)
    )
    means = dict(controller.pool_readout_means())
    assert tuple(means) == ("left_motor", "right_motor")
    for name, members in (("left_motor", LEFT_TEST_POOL), ("right_motor", RIGHT_TEST_POOL)):
        assert means[name] == math.fsum(activity[body] for body, _ in members) / len(members)


# ------------------------------------------------ calibration document gates


def test_calibration_document_round_trips() -> None:
    raw = _calibration_bytes()
    calibration = load_readout_calibration(raw)
    assert calibration.document_sha256 == sha256(raw).hexdigest()
    assert calibration.readout_scale == TEST_SCALE
    assert calibration.stream_id == "unit-test-stream-v1"
    calibration.assert_binds(
        artifact_manifest_sha256=GRAPH_SHA, sign_overlay_identity_sha256=OVERLAY_SHA
    )


def test_calibration_document_refuses_a_different_boundary() -> None:
    calibration = load_readout_calibration(_calibration_bytes())
    with pytest.raises(FullBrainProfileError, match="different artifact manifest"):
        calibration.assert_binds(
            artifact_manifest_sha256="1f" * 32, sign_overlay_identity_sha256=OVERLAY_SHA
        )
    with pytest.raises(FullBrainProfileError, match="different sign overlay"):
        calibration.assert_binds(
            artifact_manifest_sha256=GRAPH_SHA, sign_overlay_identity_sha256="1f" * 32
        )


def test_calibration_document_refuses_malformed_content() -> None:
    with pytest.raises(FullBrainProfileError, match="not a valid document"):
        load_readout_calibration(b"{not json")
    with pytest.raises(FullBrainProfileError, match="unsupported readout calibration schema"):
        load_readout_calibration(_calibration_bytes(schema="neurocraft.something.else.v1"))
    with pytest.raises(FullBrainProfileError, match="artifact manifest digest"):
        load_readout_calibration(_calibration_bytes(manifest_sha256="short"))
    with pytest.raises(FullBrainProfileError, match="sign overlay identity"):
        load_readout_calibration(_calibration_bytes(overlay_sha256="AB" * 32))
    with pytest.raises(FullBrainProfileError, match="scale hex is malformed"):
        load_readout_calibration(_calibration_bytes(scale_hex="not-a-float"))
    for scale_hex in (float("inf").hex(), float("nan").hex(), (0.0).hex(), (-1.0).hex()):
        with pytest.raises(FullBrainProfileError, match="scale must be finite"):
            load_readout_calibration(_calibration_bytes(scale_hex=scale_hex))
    with pytest.raises(FullBrainProfileError, match="decimal scale disagrees"):
        load_readout_calibration(_calibration_bytes(scale_hex=(TEST_SCALE * 2.0).hex()))

    for section in ("stream", "artifact", "sign_overlay", "calibration"):
        document = _calibration_document()
        del document[section]
        with pytest.raises(FullBrainProfileError, match=f"missing the '{section}' section"):
            load_readout_calibration(canonical_json_bytes(document))


# ------------------------------------------------------------------ determinism


def test_step_determinism_two_controllers_identical_digests() -> None:
    first = _controller()
    second = _controller()
    for step in range(5):
        action_a = first.step(
            _observation(first, step, sensors=_varied_sensors(step)), command_seq=step
        )
        action_b = second.step(
            _observation(second, step, sensors=_varied_sensors(step)), command_seq=step
        )
        evidence_a = first.last_evidence
        evidence_b = second.last_evidence
        assert evidence_a is not None and evidence_b is not None
        assert evidence_a.action_sha256 == evidence_b.action_sha256
        assert action_a.state_digest == action_b.state_digest
        assert evidence_a.state_after_activity == evidence_b.state_after_activity


# ------------------------------------------------------------ action semantics


def test_action_field_semantics_and_bounds() -> None:
    controller = _controller()
    action = controller.step(_observation(controller, 3), command_seq=0)
    assert action.identity == controller.identity
    assert action.command_seq == 0
    assert action.observation_seq == 3
    assert action.observation_tick == 3
    # observation tick 3 plus the declared two-tick action budget; the budget ceiling and
    # the target coincide, so the validity window is the target tick itself.
    assert action.target_tick == 5
    assert action.valid_through_tick == 5
    assert action.neural_step_start == 0
    assert action.neural_step_end == 5
    assert action.neutral is False
    assert action.state_digest == controller.engine.current_state().state_sha256
    for name in ("forward_thrust", "vertical_thrust", "yaw_torque"):
        assert -1.0 <= getattr(action.motors, name) <= 1.0
    assert action.motors.lateral_thrust == 0.0
    assert action.motors.pitch_torque == 0.0

    follow_up = controller.step(_observation(controller, 4), command_seq=1)
    assert follow_up.neural_step_start == 5
    assert follow_up.neural_step_end == 10


def test_reset_preserves_the_neural_step_offset() -> None:
    controller = _controller()
    assert not controller.session_has_stepped
    controller.step(_observation(controller, 0), command_seq=0)
    controller.step(_observation(controller, 1), command_seq=1)
    controller.reset()
    assert controller.engine.step_index == 0
    assert controller.session_has_stepped
    assert controller.last_evidence is None
    action = controller.step(_observation(controller, 2), command_seq=2)
    assert action.neural_step_start == 10
    assert action.neural_step_end == 15
    controller.reset_session(session_epoch=controller.identity.session_epoch + 1)
    assert not controller.session_has_stepped


def test_run_mode_is_faithful_with_taxonomy_docstring() -> None:
    controller = _controller()
    assert controller.run_mode is RunMode.FAITHFUL
    documentation = type(controller).run_mode.fget.__doc__  # type: ignore[union-attr]
    assert documentation is not None and "not a claim" in documentation


# --------------------------------------------------------------------- identity


def test_identity_derivation_is_stable_and_sensitive() -> None:
    baseline = _controller()
    same = _controller()
    assert baseline.identity == same.identity
    assert baseline.identity.graph_hash == GRAPH_SHA
    assert baseline.identity.observation_schema_hash == baseline.features.identity_sha256
    assert baseline.identity.dynamics_hash == baseline.engine.dynamics_identity_sha256

    different_overlay = _controller(motor_binding=_motor_binding(overlay_sha256="ef" * 32))
    assert different_overlay.identity.decoder_hash != baseline.identity.decoder_hash
    assert different_overlay.identity.adapter_hash == baseline.identity.adapter_hash

    different_pool = _controller(
        motor_binding=_motor_binding(right_pool=((9101, 4), (9102, 5), (9103, 6)))
    )
    assert different_pool.identity.decoder_hash != baseline.identity.decoder_hash


def test_constructor_refuses_mismatched_components() -> None:
    with pytest.raises(FullBrainProfileError, match="gain/dead-zone"):
        _controller(motor_binding=_motor_binding(gain=4.0))
    with pytest.raises(FullBrainProfileError, match="FullBrainConfiguration"):
        FullBrainController(
            run_id="full-brain-test",
            entity_id="test-fly",
            configuration=ControllerConfiguration(),  # type: ignore[arg-type]
            engine=ArrayNativeEngine(_synthetic_view()),
            sensory_binding=_sensory_binding(),
            motor_binding=_motor_binding(),
        )
    with pytest.raises(FullBrainProfileError, match="out of range"):
        _controller(motor_binding=_motor_binding(right_pool=((9101, 4), (9999, NODE_COUNT))))


def test_bindings_refuse_missing_entry_bodies() -> None:
    with pytest.raises(FullBrainProfileError, match="not present"):
        bind_sensor_channels(
            channels=ConnectomeController._sensor_channels(ControllerConfiguration()),
            index_of={805_722: 0}.__getitem__,
            node_count=NODE_COUNT,
            graph_manifest_sha256=GRAPH_SHA,
        )


# --------------------------------------------------------------------- evidence


def test_evidence_subset_shape_and_semantics() -> None:
    controller = _controller()
    controller.step(_observation(controller, 0, sensors=_varied_sensors(0)), command_seq=0)
    evidence = controller.last_evidence
    assert evidence is not None
    assert evidence.activity_subset_body_ids == (805_722, 77_185, 9001, 9002, 9101, 9102)
    assert len(evidence.state_after_activity) == len(evidence.activity_subset_body_ids)
    assert all(-1.0 <= value <= 1.0 for value in evidence.state_after_activity)
    assert set(evidence.encoded_stimulation) == {
        "forward_clearance_fraction:805722",
        "forward_clearance_fraction:77185",
        "left_clearance_fraction:805722",
        "right_clearance_fraction:77185",
        "local_light_fraction:805722",
        "local_light_fraction:77185",
    }
    assert tuple(name for name, _ in evidence.motor_frame.values) == (
        "left_motor",
        "right_motor",
    )
    assert evidence.command_seq == 0
    assert evidence.observation.game_tick == 0
    assert len(evidence.encoded_stimulation_sha256) == 64
    assert evidence.state_after_sha256 != evidence.state_before_sha256


# ------------------------------------------------------------------- validation


def test_observation_validation_mirrors_the_pinned_checks() -> None:
    controller = _controller()
    controller.step(_observation(controller, 1), command_seq=1)

    with pytest.raises(ControllerIntegrityError, match="stale"):
        controller.step(_observation(controller, 1), command_seq=2)

    other = _controller(run_id="other-run")
    with pytest.raises(ControllerIntegrityError, match="identity"):
        controller.step(_observation(other, 2), command_seq=2)

    with pytest.raises(ControllerIntegrityError, match="missing"):
        controller.step(_observation(controller, 2, missing_fields=("local_light",)), command_seq=2)

    with pytest.raises(ControllerIntegrityError, match="command sequence"):
        controller.step(_observation(controller, 2), command_seq=1)

    with pytest.raises(ControllerIntegrityError, match="integer"):
        controller.step(_observation(controller, 2), command_seq=True)


# ------------------------------------------------------------- artifact-gated


@artifact_gated
def test_from_artifact_constructs_and_steps_non_neutral() -> None:
    controller = FullBrainController.from_artifact(
        run_id="full-brain-artifact-test",
        entity_id="test-fly",
        artifact_directory=FULL_ARTIFACT_DIRECTORY,
        annotations_path=ANNOTATIONS_PATH,
    )
    identity = controller.identity
    assert identity.graph_hash == FULL_MCNS_ARTIFACT_MANIFEST_SHA256
    print(
        "full_mcns_wing_v1 identity:"
        f" graph_hash={identity.graph_hash}"
        f" adapter_hash={identity.adapter_hash}"
        f" dynamics_hash={identity.dynamics_hash}"
        f" decoder_hash={identity.decoder_hash}"
        f" observation_schema_hash={identity.observation_schema_hash}"
    )
    assert controller.engine.view.node_count == 166_700
    assert controller.engine.view.edge_count == 25_582_938
    assert controller.engine.view.provenance.origin.startswith(
        "full_runtime_artifact:coarse_presynaptic_consensus_v1:"
    )
    for step in range(3):
        action = controller.step(
            _observation(controller, step, sensors=_varied_sensors(step)), command_seq=step
        )
        assert action.neutral is False
        for name in ("forward_thrust", "vertical_thrust", "yaw_torque"):
            value = getattr(action.motors, name)
            assert -1.0 <= value <= 1.0 and value == value  # finite by protocol bounds
    evidence = controller.last_evidence
    assert evidence is not None
    assert len(evidence.state_after_activity) == 54
    controller.engine.end_run()
    assert controller.engine.run_is_covered


@artifact_gated
def test_from_artifact_same_backend_exact_replay() -> None:
    def build() -> FullBrainController:
        return FullBrainController.from_artifact(
            run_id="full-brain-replay-test",
            entity_id="test-fly",
            artifact_directory=FULL_ARTIFACT_DIRECTORY,
            annotations_path=ANNOTATIONS_PATH,
        )

    first = build()
    second = build()
    assert first.identity == second.identity
    for step in range(3):
        first.step(_observation(first, step, sensors=_varied_sensors(step)), command_seq=step)
        second.step(_observation(second, step, sensors=_varied_sensors(step)), command_seq=step)
        evidence_a = first.last_evidence
        evidence_b = second.last_evidence
        assert evidence_a is not None and evidence_b is not None
        assert evidence_a.action_sha256 == evidence_b.action_sha256
        assert evidence_a.state_after_sha256 == evidence_b.state_after_sha256
    assert first.engine.trace_sha256 == second.engine.trace_sha256


@artifact_gated
@pytest.mark.skipif(
    not CALIBRATION_PATH.is_file(), reason="the committed readout calibration is absent"
)
def test_from_artifact_loads_the_committed_readout_calibration() -> None:
    """The committed calibration binds to this artifact and rotates the decoder."""

    raw = CALIBRATION_PATH.read_bytes()
    expected = load_readout_calibration(raw)

    calibrated = FullBrainController.from_artifact(
        run_id="full-brain-calibrated-test",
        entity_id="test-fly",
        artifact_directory=FULL_ARTIFACT_DIRECTORY,
        annotations_path=ANNOTATIONS_PATH,
        calibration_path=CALIBRATION_PATH,
    )
    uncalibrated = FullBrainController.from_artifact(
        run_id="full-brain-calibrated-test",
        entity_id="test-fly",
        artifact_directory=FULL_ARTIFACT_DIRECTORY,
        annotations_path=ANNOTATIONS_PATH,
    )
    assert expected.artifact_manifest_sha256 == FULL_MCNS_ARTIFACT_MANIFEST_SHA256
    assert calibrated.readout_scale == expected.readout_scale
    assert calibrated.configuration.readout_calibration_sha256 == sha256(raw).hexdigest()
    assert uncalibrated.readout_scale == 1.0
    assert uncalibrated.configuration.readout_calibration_sha256 == ""
    assert calibrated.identity.decoder_hash != uncalibrated.identity.decoder_hash
    assert calibrated.configuration.identity_sha256 != uncalibrated.configuration.identity_sha256
    # Everything outside the readout is unchanged by the calibration.
    assert calibrated.identity.graph_hash == uncalibrated.identity.graph_hash
    assert calibrated.identity.adapter_hash == uncalibrated.identity.adapter_hash
    assert calibrated.identity.dynamics_hash == uncalibrated.identity.dynamics_hash
    print(
        "full_mcns_wing_v1 calibrated identity:"
        f" readout_scale={calibrated.readout_scale!r}"
        f" calibration_sha256={calibrated.configuration.readout_calibration_sha256}"
        f" decoder_hash={calibrated.identity.decoder_hash}"
        f" configuration_identity_sha256={calibrated.configuration.identity_sha256}"
        f" (uncalibrated decoder_hash={uncalibrated.identity.decoder_hash})"
    )


@artifact_gated
def test_from_artifact_refuses_a_calibration_from_another_boundary(tmp_path: Path) -> None:
    foreign = tmp_path / "foreign-calibration.json"
    foreign.write_bytes(_calibration_bytes(manifest_sha256="1f" * 32))
    with pytest.raises(FullBrainProfileError, match="different artifact manifest"):
        FullBrainController.from_artifact(
            run_id="full-brain-calibration-refusal",
            entity_id="test-fly",
            artifact_directory=FULL_ARTIFACT_DIRECTORY,
            annotations_path=ANNOTATIONS_PATH,
            calibration_path=foreign,
        )


def test_from_artifact_refuses_a_missing_or_doubled_calibration(tmp_path: Path) -> None:
    """Both refusals fire before any artifact byte is read, so they need no
    local artifact: the artifact directory below cannot exist."""

    absent = tmp_path / "absent-calibration.json"
    with pytest.raises(FullBrainProfileError, match="could not read the readout calibration"):
        FullBrainController.from_artifact(
            run_id="full-brain-calibration-refusal",
            entity_id="test-fly",
            artifact_directory=tmp_path / "no-artifact",
            annotations_path=tmp_path / "no-annotations",
            calibration_path=absent,
        )
    present = tmp_path / "calibration.json"
    present.write_bytes(_calibration_bytes())
    with pytest.raises(FullBrainProfileError, match="supply exactly one"):
        FullBrainController.from_artifact(
            run_id="full-brain-calibration-refusal",
            entity_id="test-fly",
            configuration=FullBrainConfiguration(
                readout_scale=TEST_SCALE, readout_calibration_sha256=CALIBRATION_SHA
            ),
            artifact_directory=tmp_path / "no-artifact",
            annotations_path=tmp_path / "no-annotations",
            calibration_path=present,
        )


@artifact_gated
def test_from_artifact_service_entrypoint_smoke(tmp_path: Path) -> None:
    """The service CLI derives and writes the full-profile identity."""

    from scripts.run_connectome_service import build_controller, build_parser, write_identity

    args = build_parser().parse_args(
        ["--identity-only", "--controller-profile", "full_mcns_wing_v1"]
    )
    controller = build_controller(args)
    assert isinstance(controller, FullBrainController)
    assert controller.identity.graph_hash == FULL_MCNS_ARTIFACT_MANIFEST_SHA256
    # The service default is the calibrated readout.
    assert controller.readout_scale != 1.0
    assert (
        controller.configuration.readout_calibration_sha256
        == sha256(CALIBRATION_PATH.read_bytes()).hexdigest()
    )
    target = tmp_path / "identity.json"
    write_identity(target, controller)
    assert target.is_file()

    uncalibrated_args = build_parser().parse_args(
        ["--identity-only", "--controller-profile", "full_mcns_wing_v1", "--full-uncalibrated"]
    )
    uncalibrated = build_controller(uncalibrated_args)
    assert isinstance(uncalibrated, FullBrainController)
    assert uncalibrated.readout_scale == 1.0
    assert uncalibrated.identity.decoder_hash != controller.identity.decoder_hash


def test_calibration_document_refuses_the_identity_scale() -> None:
    """A scale of exactly 1.0 is unrepresentable under the pairing rule; the
    loader must refuse it with the profile error type, not a bare ValueError."""

    raw = canonical_json_bytes(_calibration_document(scale=1.0))
    with pytest.raises(FullBrainProfileError, match="identity scale"):
        load_readout_calibration(raw)


def test_from_artifact_refuses_a_caller_built_calibrated_configuration() -> None:
    """A calibrated configuration may only enter from_artifact through
    calibration_path, where the document is verified against the loaded
    boundary; a caller-built scale/digest pair is refused before any load."""

    configuration = FullBrainConfiguration(
        readout_scale=TEST_SCALE,
        readout_calibration_sha256="a" * 64,
    )
    with pytest.raises(FullBrainProfileError, match="requires calibration_path"):
        FullBrainController.from_artifact(
            run_id="refusal-test",
            entity_id="refusal-fly",
            artifact_directory="/nonexistent/artifact",
            annotations_path="/nonexistent/annotations.feather",
            configuration=configuration,
        )


def test_warm_compute_path_never_touches_neural_state() -> None:
    """Warming must be invisible to the causal chain: a warmed controller and
    a cold one produce bit-identical actions, and the step index is unmoved."""

    warmed = _controller()
    cold = _controller()
    seconds = warmed.warm_compute_path(passes=2)
    assert seconds >= 0.0
    assert warmed.engine.step_index == 0
    for step in range(3):
        action_warm = warmed.step(
            _observation(warmed, step, sensors=_varied_sensors(step)), command_seq=step
        )
        action_cold = cold.step(
            _observation(cold, step, sensors=_varied_sensors(step)), command_seq=step
        )
        assert action_warm.state_digest == action_cold.state_digest
    with pytest.raises(FullBrainProfileError, match="at least one pass"):
        warmed.warm_compute_path(passes=0)


def test_warm_skips_instead_of_racing_a_concurrent_step() -> None:
    """The service runs steps on a worker thread while the idle re-warm fires
    from the event loop; the numpy fallback kernel caches a mutable scratch
    buffer, so warm must skip — never run — while a step holds the lock."""

    controller = _controller()
    assert controller._compute_lock.acquire(blocking=False)
    try:
        assert controller.warm_compute_path(passes=1) == 0.0
    finally:
        controller._compute_lock.release()
    assert controller.warm_compute_path(passes=1) > 0.0
    assert controller.engine.step_index == 0
