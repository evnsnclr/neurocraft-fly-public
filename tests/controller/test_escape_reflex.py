"""Tests for the escape-reflex Engineered-Encoder profile (``full_mcns_escape_v1``).

All CI-safe by design (the phase-1 constraint of the escape lane): synthetic
annotation frames and synthetic views only -- no artifact, no feather tables,
no committed escape calibration (that document is a phase-2 deliverable).
Every ``from_artifact``/service-shaped refusal test fakes ALL presence gates
with tmp paths (the 3a64c72 lesson), so the intended gate is reached on hosts
without local data. The v1 helpers are imported from
``test_embodied_senses`` so the lanes share one synthetic node space; the
wing/eem-v1/eem-v2 identity pins asserted there and in
``test_embodied_senses_v2.py`` remain the regression gate that this additive
lane moved nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from neurocraft_controller import ConnectomeController, ControllerConfiguration
from neurocraft_controller.embodied_senses import (
    bind_embodied_senses,
    bind_embodied_senses_v2,
)
from neurocraft_controller.escape_reflex import (
    ESCAPE_JUMP_FORMULA,
    ESCAPE_TTMN_POOL_SCHEMA,
    EXPECTED_TTMN_NODES,
    EscapeReflexError,
    TtmnPoolBinding,
    select_ttmn_pool,
)
from neurocraft_controller.flight_pools import (
    LEFT_WING_MOTOR_BODY_IDS,
    RIGHT_WING_MOTOR_BODY_IDS,
)
from neurocraft_controller.full_brain_controller import (
    ESCAPE_JUMP_POOL_NAME,
    FULL_BRAIN_ESCAPE_CONFIG_SCHEMA,
    FULL_READOUT_CALIBRATION_ESCAPE_SCHEMA,
    EscapeMotorBinding,
    FullBrainController,
    FullBrainEemConfiguration,
    FullBrainEscapeConfiguration,
    FullBrainProfileError,
    bind_escape_jump_pool,
    bind_sensor_channels,
    bind_wing_motor_pools,
    load_escape_readout_calibration,
    load_readout_calibration,
)
from neurocraft_neural.full_engine import ArrayNativeEngine
from neurocraft_protocol import canonical_json_bytes

from tests.controller.test_embodied_senses import (
    ANNOTATIONS_SHA,
    EEM_ENCODER_SHA,
    GRAPH_SHA,
    OVERLAY_SHA,
    FakeFrame,
    _calibration_document,
    _controllers,
    _observation,
    _sensors,
    _synthetic_view,
)
from tests.controller.test_embodied_senses_v2 import EEM_V2_ENCODER_SHA

#: The census TTMn bodies (D-050-era anatomy recorded in D-060): 800146 has
#: somaSide R and 804642 somaSide L. Reused verbatim in the synthetic frame.
TTMN_RIGHT_BODY = 800_146
TTMN_LEFT_BODY = 804_642

#: Escape identity literal, derivable WITHOUT the artifact: the escape
#: configuration hashes only declared material plus the pinned v1 encoder
#: identity, so this pin holds on CI as pure hashing (mirrors the v2 pin).
ESCAPE_UNCALIBRATED_CONFIG_SHA = "54814d90b296949c2e92e712ab4c5228d4fd430d4a18e6fe03874f5af8b987da"


def _ttmn_frame() -> FakeFrame:
    """The v1 synthetic frame extended with the two census TTMn rows."""

    frame = FakeFrame()
    for body, side in ((TTMN_RIGHT_BODY, "R"), (TTMN_LEFT_BODY, "L")):
        frame.types.append("TTMn")
        frame.classes.append(None)
        frame.subclasses.append(None)
        frame.sides.append(side)
        frame.body_ids.append(body)
        frame.rows += 1
    return frame


def _select_ttmn(frame: FakeFrame, *, node_count: int | None = None) -> TtmnPoolBinding:
    count = node_count or frame.rows
    populations = frame.select(node_count=count)
    return select_ttmn_pool(
        type_column=frame.types,
        soma_side_column=frame.sides,
        body_ids=frame.body_ids,
        index_of=frame.index.__getitem__,
        node_count=count,
        graph_manifest_sha256=GRAPH_SHA,
        annotations_sha256=ANNOTATIONS_SHA,
        populations=populations,
    )


def _escape_controllers() -> tuple[FullBrainController, FullBrainController]:
    """A matched (escape, eem-v1) controller pair over ONE synthetic node space.

    Both controllers share the identical view, sensory bindings, and wing
    motor binding, so their neural trajectories are byte-equal and the ONLY
    declared difference is the escape decoder stage.
    """

    frame = _ttmn_frame()
    index = frame.index
    cursor = frame.rows
    for body in LEFT_WING_MOTOR_BODY_IDS + RIGHT_WING_MOTOR_BODY_IDS:
        index[body] = cursor
        cursor += 1
    node_count = cursor
    populations = frame.select(node_count=node_count)
    senses = bind_embodied_senses(populations)
    ttmn = select_ttmn_pool(
        type_column=frame.types,
        soma_side_column=frame.sides,
        body_ids=frame.body_ids,
        index_of=index.__getitem__,
        node_count=node_count,
        graph_manifest_sha256=GRAPH_SHA,
        annotations_sha256=ANNOTATIONS_SHA,
        populations=populations,
    )
    sensory = bind_sensor_channels(
        channels=ConnectomeController._sensor_channels(ControllerConfiguration()),
        index_of=index.__getitem__,
        node_count=node_count,
        graph_manifest_sha256=GRAPH_SHA,
    )
    motor = bind_wing_motor_pools(
        index_of=index.__getitem__,
        gain=8.0,
        dead_zone=0.0,
        graph_manifest_sha256=GRAPH_SHA,
        sign_overlay_identity_sha256=OVERLAY_SHA,
    )
    escape_binding = EscapeMotorBinding(
        wing=motor,
        jump_pool=bind_escape_jump_pool(ttmn, gain=8.0, dead_zone=0.0),
        jump_scale=1.0,
        ttmn_pool_identity_sha256=ttmn.identity_sha256,
        encoder_identity_sha256=senses.encoder.identity_sha256,
    )
    escape = FullBrainController(
        run_id="escape-test",
        entity_id="test-fly",
        configuration=FullBrainEscapeConfiguration(
            encoder_identity_sha256=senses.encoder.identity_sha256
        ),
        engine=ArrayNativeEngine(_synthetic_view(node_count)),
        sensory_binding=sensory,
        motor_binding=motor,
        embodied=senses,
        escape=escape_binding,
    )
    v1 = FullBrainController(
        run_id="escape-test",
        entity_id="test-fly",
        configuration=FullBrainEemConfiguration(
            encoder_identity_sha256=senses.encoder.identity_sha256
        ),
        engine=ArrayNativeEngine(_synthetic_view(node_count)),
        sensory_binding=sensory,
        motor_binding=motor,
        embodied=senses,
    )
    return escape, v1


# ------------------------------------------------------------ TTMn selection


def test_ttmn_selection_matches_census_membership_and_schema() -> None:
    binding = _select_ttmn(_ttmn_frame())
    assert len(binding.members) == EXPECTED_TTMN_NODES == 2
    assert tuple(member.body_id for member in binding.members) == (
        TTMN_RIGHT_BODY,
        TTMN_LEFT_BODY,
    )
    assert {member.soma_side for member in binding.members} == {"L", "R"}
    material = binding.to_identity_material()
    assert material["schema"] == ESCAPE_TTMN_POOL_SCHEMA
    assert material["selector"] == "type == 'TTMn'"
    assert len(binding.identity_sha256) == 64


def test_ttmn_selection_refuses_count_drift() -> None:
    missing = _ttmn_frame()
    missing.types[missing.types.index("TTMn")] = "TTMn_like"
    with pytest.raises(EscapeReflexError, match="TTMn count drifted"):
        _select_ttmn(missing)
    extra = _ttmn_frame()
    visual_row = extra.classes.index("visual")
    extra.classes[visual_row] = None
    extra.types[visual_row] = "TTMn"
    extra.sides[visual_row] = "L"
    # The population selector now sees 6090 visual rows; hold the populations
    # binding to a clean frame so the TTMn count gate is the one that fires.
    clean = _ttmn_frame()
    with pytest.raises(EscapeReflexError, match="TTMn count drifted"):
        select_ttmn_pool(
            type_column=extra.types,
            soma_side_column=extra.sides,
            body_ids=extra.body_ids,
            index_of=extra.index.__getitem__,
            node_count=extra.rows,
            graph_manifest_sha256=GRAPH_SHA,
            annotations_sha256=ANNOTATIONS_SHA,
            populations=clean.select(node_count=clean.rows),
        )


def test_ttmn_selection_refuses_side_drift_and_unknown_sides() -> None:
    both_right = _ttmn_frame()
    both_right.sides[both_right.body_ids.index(TTMN_LEFT_BODY)] = "R"
    with pytest.raises(EscapeReflexError, match="side split drifted"):
        _select_ttmn(both_right)
    unknown = _ttmn_frame()
    unknown.sides[unknown.body_ids.index(TTMN_LEFT_BODY)] = "M"
    with pytest.raises(EscapeReflexError, match="outside the declared L/R split"):
        _select_ttmn(unknown)
    absent = _ttmn_frame()
    absent.sides[absent.body_ids.index(TTMN_RIGHT_BODY)] = None
    with pytest.raises(EscapeReflexError, match="outside the declared L/R split"):
        _select_ttmn(absent)


def test_ttmn_selection_refuses_wing_overlap() -> None:
    frame = _ttmn_frame()
    clean = _ttmn_frame()
    populations = clean.select(node_count=clean.rows)
    wing_body = LEFT_WING_MOTOR_BODY_IDS[0]
    frame.body_ids[frame.body_ids.index(TTMN_RIGHT_BODY)] = wing_body
    with pytest.raises(EscapeReflexError, match="wing-motor readout"):
        select_ttmn_pool(
            type_column=frame.types,
            soma_side_column=frame.sides,
            body_ids=frame.body_ids,
            index_of=frame.index.__getitem__,
            node_count=frame.rows,
            graph_manifest_sha256=GRAPH_SHA,
            annotations_sha256=ANNOTATIONS_SHA,
            populations=populations,
        )


def test_ttmn_selection_refuses_population_and_faithful_overlap() -> None:
    clean = _ttmn_frame()
    populations = clean.select(node_count=clean.rows)
    captured = _ttmn_frame()
    visual_body = populations.population("visual").body_ids[0]
    captured.body_ids[captured.body_ids.index(TTMN_RIGHT_BODY)] = visual_body
    with pytest.raises(EscapeReflexError, match="sensory population 'visual'"):
        select_ttmn_pool(
            type_column=captured.types,
            soma_side_column=captured.sides,
            body_ids=captured.body_ids,
            index_of=captured.index.__getitem__,
            node_count=captured.rows,
            graph_manifest_sha256=GRAPH_SHA,
            annotations_sha256=ANNOTATIONS_SHA,
            populations=populations,
        )
    faithful = _ttmn_frame()
    faithful.body_ids[faithful.body_ids.index(TTMN_RIGHT_BODY)] = 805_722
    with pytest.raises(EscapeReflexError, match="faithful sensory entry"):
        select_ttmn_pool(
            type_column=faithful.types,
            soma_side_column=faithful.sides,
            body_ids=faithful.body_ids,
            index_of=faithful.index.__getitem__,
            node_count=faithful.rows,
            graph_manifest_sha256=GRAPH_SHA,
            annotations_sha256=ANNOTATIONS_SHA,
            populations=populations,
        )


def test_ttmn_selection_refuses_mismatched_population_binding() -> None:
    frame = _ttmn_frame()
    populations = frame.select(node_count=frame.rows)
    with pytest.raises(EscapeReflexError, match="different artifact"):
        select_ttmn_pool(
            type_column=frame.types,
            soma_side_column=frame.sides,
            body_ids=frame.body_ids,
            index_of=frame.index.__getitem__,
            node_count=frame.rows,
            graph_manifest_sha256="99" * 32,
            annotations_sha256=ANNOTATIONS_SHA,
            populations=populations,
        )
    with pytest.raises(EscapeReflexError, match="different annotations table"):
        select_ttmn_pool(
            type_column=frame.types,
            soma_side_column=frame.sides,
            body_ids=frame.body_ids,
            index_of=frame.index.__getitem__,
            node_count=frame.rows,
            graph_manifest_sha256=GRAPH_SHA,
            annotations_sha256="99" * 32,
            populations=populations,
        )


def test_ttmn_selection_refuses_dense_order_mismatch() -> None:
    frame = _ttmn_frame()
    populations = frame.select(node_count=frame.rows)
    scrambled = frame.index
    scrambled[TTMN_RIGHT_BODY], scrambled[TTMN_LEFT_BODY] = (
        scrambled[TTMN_LEFT_BODY],
        scrambled[TTMN_RIGHT_BODY],
    )
    with pytest.raises(EscapeReflexError, match="not in dense body order"):
        select_ttmn_pool(
            type_column=frame.types,
            soma_side_column=frame.sides,
            body_ids=frame.body_ids,
            index_of=scrambled.__getitem__,
            node_count=frame.rows,
            graph_manifest_sha256=GRAPH_SHA,
            annotations_sha256=ANNOTATIONS_SHA,
            populations=populations,
        )


# ----------------------------------------------------------------- jump math


def test_jump_is_one_additive_clipped_term_on_vertical_only() -> None:
    """Given equal inputs, every non-vertical axis is byte-equal to eem-v1."""

    escape, v1 = _escape_controllers()
    saw_nonzero_jump = False
    # Six ticks: the synthetic ring's shortest driven path to the TTMn nodes
    # is ~21 neural sub-steps, so their activity turns nonzero at tick 4.
    for step in range(6):
        sensors = _sensors(
            left=6.0 - 0.5 * step, light=(step * 5) % 16, horizontal_collision=step == 2
        )
        escape_action = escape.step(_observation(escape, step, sensors), command_seq=step)
        v1_action = v1.step(_observation(v1, step, sensors), command_seq=step)
        # Identical dense vectors over the identical view: the trajectories match.
        assert escape_action.state_digest == v1_action.state_digest
        escape_evidence = escape.last_evidence
        v1_evidence = v1.last_evidence
        assert escape_evidence is not None and v1_evidence is not None
        assert escape_evidence.encoded_stimulation_sha256 == v1_evidence.encoded_stimulation_sha256
        # The wing MotorFrame itself is untouched by the escape stage.
        assert escape_evidence.motor_frame.values == v1_evidence.motor_frame.values
        jump = escape_evidence.jump_value
        assert jump is not None and v1_evidence.jump_value is None
        # Expected jump from the recorded TTMn activity subset (the two
        # members appended after the wing pools): mean * gain * scale, clipped.
        ttmn_activity = escape_evidence.state_after_activity[-2:]
        expected_raw = 8.0 * (ttmn_activity[0] + ttmn_activity[1]) / 2.0 * 1.0
        expected_jump = min(1.0, max(-1.0, expected_raw))
        assert jump == expected_jump
        if jump != 0.0:
            saw_nonzero_jump = True
        assert escape_action.motors.forward_thrust == v1_action.motors.forward_thrust
        assert escape_action.motors.lateral_thrust == v1_action.motors.lateral_thrust
        assert escape_action.motors.yaw_torque == v1_action.motors.yaw_torque
        assert escape_action.motors.pitch_torque == v1_action.motors.pitch_torque
        assert escape_action.motors.vertical_thrust == min(
            1.0, max(-1.0, v1_action.motors.vertical_thrust + jump)
        )
    assert saw_nonzero_jump  # the synthetic ring drives the TTMn nodes


def test_escape_evidence_subset_appends_exactly_the_ttmn_members() -> None:
    escape, v1 = _escape_controllers()
    assert escape._evidence_body_ids[: len(v1._evidence_body_ids)] == v1._evidence_body_ids
    assert escape._evidence_body_ids[-2:] == (TTMN_RIGHT_BODY, TTMN_LEFT_BODY)
    assert len(escape._evidence_body_ids) == len(v1._evidence_body_ids) + 2


def test_jump_readout_mean_matches_the_decode_arithmetic() -> None:
    escape, v1 = _escape_controllers()
    escape.step(_observation(escape, 0, _sensors(left=1.5)), command_seq=0)
    evidence = escape.last_evidence
    assert evidence is not None
    ttmn_activity = evidence.state_after_activity[-2:]
    assert escape.jump_readout_mean() == (ttmn_activity[0] + ttmn_activity[1]) / 2.0
    with pytest.raises(FullBrainProfileError, match="escape profile only"):
        v1.jump_readout_mean()


def test_two_escape_controllers_replay_identically() -> None:
    first, _ = _escape_controllers()
    second, _ = _escape_controllers()
    assert first.identity == second.identity
    for step in range(3):
        sensors = _sensors(left=6.0 - step, light=(step * 7) % 16)
        action_a = first.step(_observation(first, step, sensors), command_seq=step)
        action_b = second.step(_observation(second, step, sensors), command_seq=step)
        assert action_a.state_digest == action_b.state_digest
        evidence_a = first.last_evidence
        evidence_b = second.last_evidence
        assert evidence_a is not None and evidence_b is not None
        assert evidence_a.action_sha256 == evidence_b.action_sha256
        assert evidence_a.jump_value == evidence_b.jump_value


# ------------------------------------------------------------ identity gates


def test_escape_identity_rotates_the_decoder_and_keeps_the_sensory_boundary() -> None:
    escape, v1 = _escape_controllers()
    _, wing = _controllers()
    # The declared design: the sensory boundary IS eem v1's, so the adapter
    # identity is byte-identical by construction; the rotation lives in the
    # decoder (escape schema, TTMn membership, jump formula) and the
    # configuration (escape schema, jump_scale).
    assert escape.identity.adapter_hash == v1.identity.adapter_hash
    assert escape.identity.decoder_hash != v1.identity.decoder_hash
    assert escape.identity.decoder_hash != wing.identity.decoder_hash
    assert escape.identity.dynamics_hash == v1.identity.dynamics_hash
    assert escape.identity.graph_hash == v1.identity.graph_hash
    assert escape.identity.observation_schema_hash == v1.identity.observation_schema_hash
    assert escape.configuration.identity_sha256 != v1.configuration.identity_sha256
    binding = escape.escape_binding
    assert binding is not None and v1.escape_binding is None
    assert escape.identity.decoder_hash == binding.identity_sha256
    # The escape decoder material chains the untouched wing decoder document.
    material = binding.to_identity_material()
    assert material["wing"] == binding.wing.to_identity_material()
    assert material["jump_formula"] == ESCAPE_JUMP_FORMULA


def test_escape_uncalibrated_config_identity_is_derivable_without_the_artifact() -> None:
    config = FullBrainEscapeConfiguration(encoder_identity_sha256=EEM_ENCODER_SHA)
    assert config.identity_sha256 == ESCAPE_UNCALIBRATED_CONFIG_SHA
    material = config.to_identity_material()
    assert material["schema"] == FULL_BRAIN_ESCAPE_CONFIG_SCHEMA
    assert material["mode_taxonomy"] == "engineered_encoder"
    assert material["jump_scale"] == (1.0).hex()


def test_escape_configuration_schema_separation_and_jump_pairing() -> None:
    v1_config = FullBrainEemConfiguration(encoder_identity_sha256=EEM_ENCODER_SHA)
    escape_config = FullBrainEscapeConfiguration(encoder_identity_sha256=EEM_ENCODER_SHA)
    assert escape_config.schema == FULL_BRAIN_ESCAPE_CONFIG_SCHEMA
    assert escape_config.identity_sha256 != v1_config.identity_sha256
    with pytest.raises(ValueError, match="unsupported escape-reflex"):
        FullBrainEscapeConfiguration(
            schema=v1_config.schema, encoder_identity_sha256=EEM_ENCODER_SHA
        )
    with pytest.raises(ValueError, match="encoder document identity"):
        FullBrainEscapeConfiguration()
    with pytest.raises(ValueError, match="mode_taxonomy"):
        FullBrainEscapeConfiguration(
            encoder_identity_sha256=EEM_ENCODER_SHA, mode_taxonomy="faithful"
        )
    # A declared jump scale requires the calibration digest, and vice versa.
    with pytest.raises(ValueError, match="requires the SHA-256"):
        FullBrainEscapeConfiguration(encoder_identity_sha256=EEM_ENCODER_SHA, jump_scale=500.0)
    with pytest.raises(ValueError, match="must carry the measured"):
        FullBrainEscapeConfiguration(
            encoder_identity_sha256=EEM_ENCODER_SHA,
            readout_scale=76.0,
            readout_calibration_sha256="6f" * 32,
            jump_scale=1.0,
        )
    with pytest.raises(ValueError, match="jump_scale must be finite"):
        FullBrainEscapeConfiguration(
            encoder_identity_sha256=EEM_ENCODER_SHA,
            readout_scale=76.0,
            readout_calibration_sha256="6f" * 32,
            jump_scale=float("inf"),
        )
    calibrated = FullBrainEscapeConfiguration(
        encoder_identity_sha256=EEM_ENCODER_SHA,
        readout_scale=76.0,
        readout_calibration_sha256="6f" * 32,
        jump_scale=500.0,
    )
    assert calibrated.identity_sha256 != escape_config.identity_sha256


def test_controller_refuses_mismatched_escape_pairings() -> None:
    escape, v1 = _escape_controllers()
    binding = escape.escape_binding
    senses = escape.embodied_senses
    assert binding is not None and senses is not None
    with pytest.raises(FullBrainProfileError, match="requires the escape motor binding"):
        FullBrainController(
            run_id="pairing-test",
            entity_id="test-fly",
            configuration=escape.configuration,
            engine=escape.engine,
            sensory_binding=escape._sensory_binding,
            motor_binding=escape._motor_binding,
            embodied=senses,
        )
    with pytest.raises(FullBrainProfileError, match="requires a FullBrainEscapeConfiguration"):
        FullBrainController(
            run_id="pairing-test",
            entity_id="test-fly",
            configuration=v1.configuration,
            engine=escape.engine,
            sensory_binding=escape._sensory_binding,
            motor_binding=escape._motor_binding,
            embodied=senses,
            escape=binding,
        )
    mismatched_jump = EscapeMotorBinding(
        wing=binding.wing,
        jump_pool=binding.jump_pool,
        jump_scale=500.0,
        ttmn_pool_identity_sha256=binding.ttmn_pool_identity_sha256,
        encoder_identity_sha256=binding.encoder_identity_sha256,
    )
    with pytest.raises(FullBrainProfileError, match="jump_scale does not match"):
        FullBrainController(
            run_id="pairing-test",
            entity_id="test-fly",
            configuration=escape.configuration,
            engine=escape.engine,
            sensory_binding=escape._sensory_binding,
            motor_binding=escape._motor_binding,
            embodied=senses,
            escape=mismatched_jump,
        )
    with pytest.raises(FullBrainProfileError, match="v1/v2 mix is refused"):
        FullBrainController(
            run_id="pairing-test",
            entity_id="test-fly",
            configuration=escape.configuration,
            engine=escape.engine,
            sensory_binding=escape._sensory_binding,
            motor_binding=escape._motor_binding,
            embodied=bind_embodied_senses_v2(senses.populations),
            escape=binding,
        )


def test_escape_binding_refuses_formula_and_pool_drift() -> None:
    escape, _ = _escape_controllers()
    binding = escape.escape_binding
    assert binding is not None
    with pytest.raises(FullBrainProfileError, match="declared jump mapping formula"):
        EscapeMotorBinding(
            wing=binding.wing,
            jump_pool=binding.jump_pool,
            jump_scale=binding.jump_scale,
            ttmn_pool_identity_sha256=binding.ttmn_pool_identity_sha256,
            encoder_identity_sha256=binding.encoder_identity_sha256,
            jump_formula="jump_raw = something else",
        )
    from neurocraft_controller.full_brain_controller import BoundMotorPool, BoundPoolMember

    misnamed = BoundMotorPool(
        name="left_motor",
        gain=8.0,
        dead_zone=0.0,
        members=binding.jump_pool.members,
    )
    with pytest.raises(FullBrainProfileError, match="must be named 'jump_motor'"):
        EscapeMotorBinding(
            wing=binding.wing,
            jump_pool=misnamed,
            jump_scale=binding.jump_scale,
            ttmn_pool_identity_sha256=binding.ttmn_pool_identity_sha256,
            encoder_identity_sha256=binding.encoder_identity_sha256,
        )
    weighted = BoundMotorPool(
        name=ESCAPE_JUMP_POOL_NAME,
        gain=8.0,
        dead_zone=0.0,
        members=(
            BoundPoolMember(
                body_id=binding.jump_pool.members[0].body_id,
                dense_index=binding.jump_pool.members[0].dense_index,
                weight=2.0,
            ),
            binding.jump_pool.members[1],
        ),
    )
    with pytest.raises(FullBrainProfileError, match="unit weights only"):
        EscapeMotorBinding(
            wing=binding.wing,
            jump_pool=weighted,
            jump_scale=binding.jump_scale,
            ttmn_pool_identity_sha256=binding.ttmn_pool_identity_sha256,
            encoder_identity_sha256=binding.encoder_identity_sha256,
        )
    single = BoundMotorPool(
        name=ESCAPE_JUMP_POOL_NAME,
        gain=8.0,
        dead_zone=0.0,
        members=binding.jump_pool.members[:1],
    )
    with pytest.raises(FullBrainProfileError, match="exactly the two TTMn members"):
        EscapeMotorBinding(
            wing=binding.wing,
            jump_pool=single,
            jump_scale=binding.jump_scale,
            ttmn_pool_identity_sha256=binding.ttmn_pool_identity_sha256,
            encoder_identity_sha256=binding.encoder_identity_sha256,
        )


# ----------------------------------------------------- calibration cross-binds


def _escape_calibration_document(
    *,
    encoder_sha256: str = EEM_ENCODER_SHA,
    ttmn_sha256: str = "7a" * 32,
    wing_scale: float = 76.0,
    jump_scale: float = 500.0,
) -> dict[str, Any]:
    return {
        "schema": FULL_READOUT_CALIBRATION_ESCAPE_SCHEMA,
        "stream": {"stream_id": "unit-test-escape-stream-v1"},
        "artifact": {"manifest_file_sha256": GRAPH_SHA},
        "sign_overlay": {"identity_sha256": OVERLAY_SHA},
        "calibration": {
            "wing_scale_float_hex": float(wing_scale).hex(),
            "wing_scale": float(wing_scale),
            "jump_scale_float_hex": float(jump_scale).hex(),
            "jump_scale": float(jump_scale),
        },
        "encoder": {"identity_sha256": encoder_sha256},
        "jump_readout": {"ttmn_pool_binding_sha256": ttmn_sha256},
    }


def test_escape_calibration_loads_and_binds_its_own_boundary() -> None:
    calibration = load_escape_readout_calibration(
        canonical_json_bytes(_escape_calibration_document())
    )
    assert calibration.wing_scale == 76.0
    assert calibration.jump_scale == 500.0
    assert calibration.encoder_identity_sha256 == EEM_ENCODER_SHA
    assert calibration.ttmn_pool_identity_sha256 == "7a" * 32
    calibration.assert_binds(
        artifact_manifest_sha256=GRAPH_SHA,
        sign_overlay_identity_sha256=OVERLAY_SHA,
        encoder_identity_sha256=EEM_ENCODER_SHA,
        ttmn_pool_identity_sha256="7a" * 32,
    )
    with pytest.raises(FullBrainProfileError, match="different encoder document"):
        calibration.assert_binds(
            artifact_manifest_sha256=GRAPH_SHA,
            sign_overlay_identity_sha256=OVERLAY_SHA,
            encoder_identity_sha256=EEM_V2_ENCODER_SHA,
            ttmn_pool_identity_sha256="7a" * 32,
        )
    with pytest.raises(FullBrainProfileError, match="different TTMn pool binding"):
        calibration.assert_binds(
            artifact_manifest_sha256=GRAPH_SHA,
            sign_overlay_identity_sha256=OVERLAY_SHA,
            encoder_identity_sha256=EEM_ENCODER_SHA,
            ttmn_pool_identity_sha256="8b" * 32,
        )
    with pytest.raises(FullBrainProfileError, match="different artifact manifest"):
        calibration.assert_binds(
            artifact_manifest_sha256="99" * 32,
            sign_overlay_identity_sha256=OVERLAY_SHA,
            encoder_identity_sha256=EEM_ENCODER_SHA,
            ttmn_pool_identity_sha256="7a" * 32,
        )
    with pytest.raises(FullBrainProfileError, match="different sign overlay"):
        calibration.assert_binds(
            artifact_manifest_sha256=GRAPH_SHA,
            sign_overlay_identity_sha256="99" * 32,
            encoder_identity_sha256=EEM_ENCODER_SHA,
            ttmn_pool_identity_sha256="7a" * 32,
        )


def test_calibration_schemas_refuse_every_cross_profile_pair() -> None:
    """The escape and v1 calibration formats refuse each other, all pairs.

    The wing, eem-v1, and eem-v2 boundaries all load documents through
    ``load_readout_calibration``, which refuses the escape schema; the escape
    boundary loads through ``load_escape_readout_calibration``, which refuses
    every v1-schema document by name. Together with the encoder-identity
    refusals pinned in ``test_embodied_senses_v2.py``, every directed pair of
    the four profiles is refused.
    """

    v1_documents = {
        "wing": _calibration_document(encoder_sha256=None),
        "eem_v1": _calibration_document(encoder_sha256=EEM_ENCODER_SHA),
        "eem_v2": _calibration_document(encoder_sha256=EEM_V2_ENCODER_SHA),
    }
    for document in v1_documents.values():
        with pytest.raises(FullBrainProfileError, match="escape boundary refuses"):
            load_escape_readout_calibration(canonical_json_bytes(document))
    escape_bytes = canonical_json_bytes(_escape_calibration_document())
    with pytest.raises(FullBrainProfileError, match="unsupported readout calibration schema"):
        load_readout_calibration(escape_bytes)


def test_escape_calibration_scale_validation() -> None:
    identity_scale = _escape_calibration_document(jump_scale=1.0)
    with pytest.raises(FullBrainProfileError, match=r"identity jump_scale 1\.0"):
        load_escape_readout_calibration(canonical_json_bytes(identity_scale))
    disagreeing = _escape_calibration_document()
    disagreeing["calibration"]["wing_scale"] = 77.0
    with pytest.raises(FullBrainProfileError, match="disagrees with its float hex"):
        load_escape_readout_calibration(canonical_json_bytes(disagreeing))
    missing_jump = _escape_calibration_document()
    del missing_jump["jump_readout"]
    with pytest.raises(FullBrainProfileError, match="jump_readout"):
        load_escape_readout_calibration(canonical_json_bytes(missing_jump))
    bad_ttmn = _escape_calibration_document(ttmn_sha256="not-a-digest")
    with pytest.raises(FullBrainProfileError, match="TTMn pool binding identity"):
        load_escape_readout_calibration(canonical_json_bytes(bad_ttmn))


# -------------------------------------------------- from_artifact argument gates


def test_from_artifact_refuses_escape_argument_misuse(tmp_path: Path) -> None:
    """Every refusal fires before any artifact byte is read; all presence
    gates are faked with tmp paths so the intended gate is reached on hosts
    without local data."""

    with pytest.raises(FullBrainProfileError, match="requires embodied_senses=True"):
        FullBrainController.from_artifact(
            run_id="refusal",
            entity_id="fly",
            artifact_directory=tmp_path / "absent",
            annotations_path=tmp_path / "absent.feather",
            escape_reflex=True,
        )
    with pytest.raises(FullBrainProfileError, match=r"eem v1 \(absolute\) encoder only"):
        FullBrainController.from_artifact(
            run_id="refusal",
            entity_id="fly",
            artifact_directory=tmp_path / "absent",
            annotations_path=tmp_path / "absent.feather",
            embodied_senses=True,
            embodied_encoder_version=2,
            body_annotations_path=tmp_path / "absent-annotations.feather",
            escape_reflex=True,
        )
    with pytest.raises(FullBrainProfileError, match="full_mcns_escape_v1 is a local-only"):
        FullBrainController.from_artifact(
            run_id="refusal",
            entity_id="fly",
            artifact_directory=tmp_path / "absent",
            annotations_path=tmp_path / "absent.feather",
            embodied_senses=True,
            body_annotations_path=tmp_path / "absent-annotations.feather",
            escape_reflex=True,
        )
