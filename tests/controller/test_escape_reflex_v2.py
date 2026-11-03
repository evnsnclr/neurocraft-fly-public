"""Tests for the escape-reflex v2 profile (``full_mcns_escape_v2``).

The lane's two declared changes against escape v1 (identity recorded in
``experiments/preregistration/eem-escape-v2.md``): the ``touch`` channel
alone moves under the frozen D-059 temporal-contrast formula, and the jump
mapping is rectified to the non-negative half of its declared codomain.

All CI-safe by design (the freeze-phase constraint of this lane): synthetic
annotation frames and synthetic views only -- no artifact, no feather tables,
no committed escape-v2 calibration (that document is a phase-2 deliverable,
and NO measurement has been run under this identity). Every
``from_artifact``/service-shaped refusal test fakes ALL preceding presence
gates with tmp paths (the 3a64c72 lesson), so the intended gate is the one
reached on hosts without local data. The v1/v2 helpers are imported from the
sibling suites so every lane shares one synthetic node space, and the
wing/eem-v1/eem-v2/escape-v1 identity pins asserted there and re-asserted
here remain the regression gate that this additive lane moved nothing.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from neurocraft_controller import ConnectomeController, ControllerConfiguration
from neurocraft_controller.embodied_senses import (
    EEM_CHANNELS,
    EEM_CONTRAST_ALPHA,
    EEM_ENCODER_ESCAPE_V2_SCHEMA,
    EEM_V2_ABSOLUTE_CHANNEL_NAMES,
    EEM_V2_CONTRAST_CHANNEL_NAMES,
    ESCAPE_V2_ABSOLUTE_CHANNEL_NAMES,
    ESCAPE_V2_CHANNELS,
    ESCAPE_V2_CONTRAST_CHANNEL_NAMES,
    EemEncoderDocument,
    EemEncoderDocumentEscapeV2,
    EemEncoderDocumentV2,
    EmbodiedSensesError,
    apply_temporal_contrast,
    bind_embodied_senses,
    bind_embodied_senses_escape_v2,
    bind_embodied_senses_v2,
    encode_channel_values,
)
from neurocraft_controller.escape_reflex import (
    ESCAPE_JUMP_FORMULA,
    ESCAPE_JUMP_FORMULA_V2,
    select_ttmn_pool,
)
from neurocraft_controller.flight_pools import (
    LEFT_WING_MOTOR_BODY_IDS,
    RIGHT_WING_MOTOR_BODY_IDS,
)
from neurocraft_controller.full_brain_controller import (
    FULL_BRAIN_EEM_ESCAPE_V2_ENCODED_STIMULATION_SCHEMA,
    FULL_BRAIN_EEM_V2_ENCODED_STIMULATION_SCHEMA,
    FULL_BRAIN_ESCAPE_CONFIG_SCHEMA,
    FULL_BRAIN_ESCAPE_V2_CONFIG_SCHEMA,
    FULL_BRAIN_ESCAPE_V2_DECODER_SCHEMA,
    FULL_READOUT_CALIBRATION_ESCAPE_V2_SCHEMA,
    EscapeMotorBinding,
    EscapeMotorBindingV2,
    FullBrainController,
    FullBrainEemConfiguration,
    FullBrainEemV2Configuration,
    FullBrainEscapeConfiguration,
    FullBrainEscapeV2Configuration,
    FullBrainProfileError,
    bind_escape_jump_pool,
    bind_sensor_channels,
    bind_wing_motor_pools,
    load_escape_readout_calibration,
    load_escape_v2_readout_calibration,
    load_readout_calibration,
)
from neurocraft_neural.full_engine import ArrayNativeEngine
from neurocraft_protocol import canonical_json_bytes

from tests.controller.test_embodied_senses import (
    ANNOTATIONS_SHA,
    EEM_ENCODER_SHA,
    EEM_POPULATION_BINDING_SHA,
    GRAPH_SHA,
    OVERLAY_SHA,
    _calibration_document,
    _observation,
    _sensors,
    _synthetic_view,
)
from tests.controller.test_embodied_senses_v2 import (
    EEM_V1_UNCALIBRATED_CONFIG_SHA,
    EEM_V2_ENCODER_SHA,
    EEM_V2_UNCALIBRATED_CONFIG_SHA,
)
from tests.controller.test_escape_reflex import (
    ESCAPE_UNCALIBRATED_CONFIG_SHA,
    TTMN_LEFT_BODY,
    TTMN_RIGHT_BODY,
    _escape_calibration_document,
    _ttmn_frame,
)

#: The escape-v2 freeze pins, frozen in
#: ``experiments/preregistration/eem-escape-v2.md`` and derivable WITHOUT the
#: artifact: the encoder document hashes only declared material plus the
#: pinned population-binding identity, and the configuration hashes only
#: declared material plus that encoder identity (the escape-v1 and eem
#: derivations verbatim).
ESCAPE_V2_ENCODER_SHA = "1cb591a222806ea5e1b687a1ccdb092809a9a7223f5ebc1d523556478d82ea9c"
ESCAPE_V2_UNCALIBRATED_CONFIG_SHA = (
    "32624431aefac74248edf51b58824dc97a37116abb26e9e90633bb2dcc0142c2"
)


def _escape_v2_controllers(
    *, dead_zone: float = 0.0
) -> tuple[FullBrainController, FullBrainController]:
    """A matched (escape-v2, escape-v1) pair over ONE synthetic node space.

    Both controllers share the identical view, sensory binding, wing motor
    binding, and TTMn pool, so the ONLY declared differences are the encoder
    document (transient touch) and the jump mapping (rectified).
    """

    frame = _ttmn_frame()
    index = frame.index
    cursor = frame.rows
    for body in LEFT_WING_MOTOR_BODY_IDS + RIGHT_WING_MOTOR_BODY_IDS:
        index[body] = cursor
        cursor += 1
    node_count = cursor
    populations = frame.select(node_count=node_count)
    senses_v1 = bind_embodied_senses(populations)
    senses_v2 = bind_embodied_senses_escape_v2(populations)
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
        dead_zone=dead_zone,
        graph_manifest_sha256=GRAPH_SHA,
        sign_overlay_identity_sha256=OVERLAY_SHA,
    )
    jump_pool = bind_escape_jump_pool(ttmn, gain=8.0, dead_zone=dead_zone)
    escape_v2 = FullBrainController(
        run_id="escape-v2-test",
        entity_id="test-fly",
        configuration=FullBrainEscapeV2Configuration(
            motor_dead_zone=dead_zone,
            encoder_identity_sha256=senses_v2.encoder.identity_sha256,
        ),
        engine=ArrayNativeEngine(_synthetic_view(node_count)),
        sensory_binding=sensory,
        motor_binding=motor,
        embodied=senses_v2,
        escape=EscapeMotorBindingV2(
            wing=motor,
            jump_pool=jump_pool,
            jump_scale=1.0,
            ttmn_pool_identity_sha256=ttmn.identity_sha256,
            encoder_identity_sha256=senses_v2.encoder.identity_sha256,
        ),
    )
    escape_v1 = FullBrainController(
        run_id="escape-v2-test",
        entity_id="test-fly",
        configuration=FullBrainEscapeConfiguration(
            motor_dead_zone=dead_zone,
            encoder_identity_sha256=senses_v1.encoder.identity_sha256,
        ),
        engine=ArrayNativeEngine(_synthetic_view(node_count)),
        sensory_binding=sensory,
        motor_binding=motor,
        embodied=senses_v1,
        escape=EscapeMotorBinding(
            wing=motor,
            jump_pool=jump_pool,
            jump_scale=1.0,
            ttmn_pool_identity_sha256=ttmn.identity_sha256,
            encoder_identity_sha256=senses_v1.encoder.identity_sha256,
        ),
    )
    return escape_v2, escape_v1


def _jump_activity(controller: FullBrainController, value: float) -> np.ndarray[Any, Any]:
    """A dense activity vector with both TTMn members held at ``value``."""

    escape = controller.escape_binding
    assert escape is not None
    activity = np.zeros(controller.engine.view.node_count, dtype=np.float64)
    for member in escape.jump_pool.members:
        activity[member.dense_index] = value
    return activity


# ---------------------------------------------------------- the v2 encoder document


def test_escape_v2_encoder_applies_contrast_to_touch_only() -> None:
    encoder = EemEncoderDocumentEscapeV2(population_binding_sha256=EEM_POPULATION_BINDING_SHA)
    assert encoder.contrast_channels == ESCAPE_V2_CONTRAST_CHANNEL_NAMES == ("touch",)
    assert (
        encoder.absolute_channels
        == ESCAPE_V2_ABSOLUTE_CHANNEL_NAMES
        == (
            "threat_left",
            "threat_right",
            "threat_front",
            "light",
        )
    )
    # The eem-v2 partition is exactly the complement, byte-unchanged.
    assert set(EEM_V2_CONTRAST_CHANNEL_NAMES) == set(ESCAPE_V2_ABSOLUTE_CHANNEL_NAMES)
    assert EEM_V2_ABSOLUTE_CHANNEL_NAMES == ESCAPE_V2_CONTRAST_CHANNEL_NAMES
    assert encoder.alpha == EEM_CONTRAST_ALPHA == 0.125
    material = encoder.to_identity_material()
    assert material["schema"] == EEM_ENCODER_ESCAPE_V2_SCHEMA
    assert material["schema"] == "neurocraft.eem.encoder.escape.v2"
    encodings = {entry["name"]: entry["encoding"] for entry in material["channels"]}
    assert encodings == {
        "threat_left": "absolute",
        "threat_right": "absolute",
        "threat_front": "absolute",
        "light": "absolute",
        "touch": "temporal_contrast",
    }
    # The four absolute channels are the v1 channel objects verbatim; only
    # touch's declared formula is wrapped in the frozen contrast declaration.
    assert ESCAPE_V2_CHANNELS[:4] == EEM_CHANNELS[:4]
    assert ESCAPE_V2_CHANNELS[4].name == EEM_CHANNELS[4].name == "touch"
    assert ESCAPE_V2_CHANNELS[4].populations == EEM_CHANNELS[4].populations
    assert ESCAPE_V2_CHANNELS[4].gain == EEM_CHANNELS[4].gain == 1.0
    assert EEM_CHANNELS[4].formula in ESCAPE_V2_CHANNELS[4].formula
    assert ESCAPE_V2_CHANNELS[4].formula.startswith("temporal contrast of the v1 raw value")


def test_escape_v2_encoder_refuses_a_different_contrast_set() -> None:
    with pytest.raises(EmbodiedSensesError, match="touch channel only"):
        EemEncoderDocumentEscapeV2(
            population_binding_sha256=EEM_POPULATION_BINDING_SHA,
            contrast_channels=("light",),
            absolute_channels=("threat_left", "threat_right", "threat_front", "touch"),
        )
    with pytest.raises(EmbodiedSensesError, match="alpha = 1/8"):
        EemEncoderDocumentEscapeV2(population_binding_sha256=EEM_POPULATION_BINDING_SHA, alpha=0.25)


def test_escape_v2_identity_pins_derive_without_the_artifact() -> None:
    """Pure-hashing freeze pins, plus every neighbouring lane's regression pin."""

    encoder = EemEncoderDocumentEscapeV2(population_binding_sha256=EEM_POPULATION_BINDING_SHA)
    assert encoder.identity_sha256 == ESCAPE_V2_ENCODER_SHA
    assert (
        FullBrainEscapeV2Configuration(
            encoder_identity_sha256=ESCAPE_V2_ENCODER_SHA
        ).identity_sha256
        == ESCAPE_V2_UNCALIBRATED_CONFIG_SHA
    )
    # Byte-unchanged: the wing/eem-v1/eem-v2/escape-v1 identities this lane
    # must not move a bit.
    assert (
        EemEncoderDocument(population_binding_sha256=EEM_POPULATION_BINDING_SHA).identity_sha256
        == EEM_ENCODER_SHA
    )
    assert (
        EemEncoderDocumentV2(population_binding_sha256=EEM_POPULATION_BINDING_SHA).identity_sha256
        == EEM_V2_ENCODER_SHA
    )
    assert (
        FullBrainEemConfiguration(encoder_identity_sha256=EEM_ENCODER_SHA).identity_sha256
        == EEM_V1_UNCALIBRATED_CONFIG_SHA
    )
    assert (
        FullBrainEemV2Configuration(encoder_identity_sha256=EEM_V2_ENCODER_SHA).identity_sha256
        == EEM_V2_UNCALIBRATED_CONFIG_SHA
    )
    assert (
        FullBrainEscapeConfiguration(encoder_identity_sha256=EEM_ENCODER_SHA).identity_sha256
        == ESCAPE_UNCALIBRATED_CONFIG_SHA
    )
    # Every escape-v2 identity is distinct from every other lane's.
    assert (
        len(
            {
                ESCAPE_V2_ENCODER_SHA,
                ESCAPE_V2_UNCALIBRATED_CONFIG_SHA,
                EEM_ENCODER_SHA,
                EEM_V2_ENCODER_SHA,
                EEM_V1_UNCALIBRATED_CONFIG_SHA,
                EEM_V2_UNCALIBRATED_CONFIG_SHA,
                ESCAPE_UNCALIBRATED_CONFIG_SHA,
            }
        )
        == 7
    )


# ------------------------------------------------------- touch contrast trajectory


def _touch_trajectory(raw_flags: list[bool]) -> list[tuple[float, float, float]]:
    """Drive the declared escape-v2 contrast over a raw collision sequence.

    Returns ``(raw, baseline, drive)`` per tick, using the exact machinery the
    controller executes (``apply_temporal_contrast`` with the encoder
    document's own alpha and contrast set).
    """

    encoder = EemEncoderDocumentEscapeV2(population_binding_sha256=EEM_POPULATION_BINDING_SHA)
    means: dict[str, float] | None = None
    trajectory: list[tuple[float, float, float]] = []
    for flag in raw_flags:
        raw_values = encode_channel_values(_sensors(horizontal_collision=flag))
        encoded, means_used, means = apply_temporal_contrast(
            raw_values,
            means,
            alpha=encoder.alpha,
            contrast_channels=encoder.contrast_channels,
        )
        raw = dict(raw_values)["touch"]
        trajectory.append((raw, dict(means_used)["touch"], dict(encoded)["touch"]))
    return trajectory


def test_touch_contrast_first_tick_drive_is_zero() -> None:
    """The declared first tick: baseline := raw_0, so drive_0 = 0 either way."""

    assert _touch_trajectory([False])[0] == (0.0, 0.0, 0.0)
    assert _touch_trajectory([True])[0] == (1.0, 1.0, 0.0)


def test_touch_contrast_onset_then_geometric_decay_at_alpha_one_eighth() -> None:
    # Quiet, then a contact onset held for the rest of the run.
    trajectory = _touch_trajectory([False, False, True, True, True, True, True])
    assert trajectory[0][2] == 0.0  # declared first tick
    assert trajectory[1][2] == 0.0  # still quiet: raw == baseline == 0
    onset = trajectory[2]
    assert onset[0] == 1.0 and onset[1] == 0.0
    assert onset[2] == 1.0  # onset drives the bristles at full amplitude
    # Held contact decays exactly as (1 - alpha)^n against the pre-update mean:
    # every value is dyadic, so these are bitwise equalities, not tolerances.
    held = [entry[2] for entry in trajectory[2:]]
    assert held[:4] == [1.0, 0.875, 0.765625, 0.669921875]
    for entry in trajectory[2:]:
        # drive + pre-update baseline == raw, exactly, while contact is held.
        assert entry[2] + entry[1] == entry[0] == 1.0
    assert all(later < earlier for earlier, later in pairwise(held))


def test_touch_contrast_sustained_contact_adapts_toward_zero() -> None:
    trajectory = _touch_trajectory([False, *([True] * 60)])
    drives = [entry[2] for entry in trajectory[1:]]
    assert drives[0] == 1.0
    assert all(later < earlier for earlier, later in pairwise(drives))
    assert drives[-1] < 1e-3  # sustained resting contact vanishes from the drive


def test_touch_contrast_release_produces_an_off_transient_of_the_same_sign() -> None:
    # Onset, five held ticks, then release.
    trajectory = _touch_trajectory([False, True, True, True, True, True, False])
    baseline_at_release = trajectory[-1][1]
    release_drive = trajectory[-1][2]
    assert trajectory[-1][0] == 0.0
    assert release_drive > 0.0  # OFF transient, same sign as the onset
    assert release_drive == baseline_at_release  # |0 - m| == m, exactly


def test_absolute_channels_are_bit_identical_to_v1_under_identical_inputs() -> None:
    """The four threat/light channels pass through the v2 encoder verbatim."""

    encoder = EemEncoderDocumentEscapeV2(population_binding_sha256=EEM_POPULATION_BINDING_SHA)
    means: dict[str, float] | None = None
    for step in range(8):
        sensors = _sensors(
            forward=6.0 - 0.5 * step,
            left=6.0 - 0.25 * step,
            right=1.0 + 0.5 * step,
            light=(step * 3) % 16,
            horizontal_collision=step % 2 == 0,
        )
        raw_values = encode_channel_values(sensors)
        encoded, _means_used, means = apply_temporal_contrast(
            raw_values,
            means,
            alpha=encoder.alpha,
            contrast_channels=encoder.contrast_channels,
        )
        raw = dict(raw_values)
        got = dict(encoded)
        for name in ESCAPE_V2_ABSOLUTE_CHANNEL_NAMES:
            assert got[name] == raw[name]  # bitwise: the v1 absolute value


def test_live_controller_touch_evidence_matches_the_declared_trajectory() -> None:
    """The executed controller path reproduces the pure-function trajectory."""

    escape_v2, escape_v1 = _escape_v2_controllers()
    flags = [False, True, True, True, False]
    expected = _touch_trajectory(flags)
    for step, flag in enumerate(flags):
        sensors = _sensors(horizontal_collision=flag)
        escape_v2.step(_observation(escape_v2, step, sensors), command_seq=step)
        escape_v1.step(_observation(escape_v1, step, sensors), command_seq=step)
        evidence = escape_v2.last_evidence
        v1_evidence = escape_v1.last_evidence
        assert evidence is not None and v1_evidence is not None
        encoded = evidence.encoded_stimulation
        assert encoded["eem_raw:touch"] == expected[step][0]
        assert encoded["eem_mean:touch"] == expected[step][1]
        assert encoded["eem:touch"] == expected[step][2]
        assert encoded["eem_drive:bristle"] == expected[step][2]
        # The four absolute channels are byte-identical to escape-v1's.
        for name in ESCAPE_V2_ABSOLUTE_CHANNEL_NAMES:
            assert encoded[f"eem:{name}"] == v1_evidence.encoded_stimulation[f"eem:{name}"]


def test_reset_clears_the_touch_running_mean() -> None:
    escape_v2, _ = _escape_v2_controllers()
    for step, flag in enumerate([False, True, True]):
        escape_v2.step(
            _observation(escape_v2, step, _sensors(horizontal_collision=flag)), command_seq=step
        )
    adapted = escape_v2.last_evidence
    assert adapted is not None and 0.0 < adapted.encoded_stimulation["eem:touch"] < 1.0
    escape_v2.reset()
    assert escape_v2._eem_contrast_means is None
    # The session clocks are deliberately not rewound by reset(), so the next
    # observation continues the sequence; the ENCODER state is what resets.
    escape_v2.step(_observation(escape_v2, 3, _sensors(horizontal_collision=True)), command_seq=3)
    fresh = escape_v2.last_evidence
    assert fresh is not None
    assert fresh.encoded_stimulation["eem:touch"] == 0.0  # declared first tick again


def test_escape_v2_carries_its_own_stimulation_provenance_schema() -> None:
    """The per-step provenance schema separates escape-v2 from eem-v2.

    The declared value is asserted directly, and the contrast-only evidence
    fields prove the escape-v2 encoder rides the contrast path (the v1
    absolute path emits neither ``eem_raw:`` nor ``eem_mean:``).
    """

    assert FULL_BRAIN_EEM_ESCAPE_V2_ENCODED_STIMULATION_SCHEMA == (
        "neurocraft.full_brain.encoded_stimulation.eem.escape.v2"
    )
    assert (
        FULL_BRAIN_EEM_ESCAPE_V2_ENCODED_STIMULATION_SCHEMA
        != FULL_BRAIN_EEM_V2_ENCODED_STIMULATION_SCHEMA
    )
    escape_v2, escape_v1 = _escape_v2_controllers()
    sensors = _sensors(horizontal_collision=True)
    escape_v2.step(_observation(escape_v2, 0, sensors), command_seq=0)
    escape_v1.step(_observation(escape_v1, 0, sensors), command_seq=0)
    v2_evidence = escape_v2.last_evidence
    v1_evidence = escape_v1.last_evidence
    assert v2_evidence is not None and v1_evidence is not None
    assert "eem_mean:touch" in v2_evidence.encoded_stimulation
    assert "eem_raw:touch" in v2_evidence.encoded_stimulation
    assert not any(key.startswith("eem_mean:") for key in v1_evidence.encoded_stimulation)
    assert v2_evidence.encoded_stimulation_sha256 != v1_evidence.encoded_stimulation_sha256


# ------------------------------------------------------------- jump rectification


def test_jump_rectification_maps_negative_pool_activity_to_zero() -> None:
    escape_v2, escape_v1 = _escape_v2_controllers()
    for value in (-1.0, -0.5, -1e-9):
        activity = _jump_activity(escape_v2, value)
        assert escape_v2._decode_jump(activity) == 0.0
        # The v1 binding still admits the signed codomain (the D-062 rail).
        assert escape_v1._decode_jump(activity) == min(1.0, max(-1.0, 8.0 * value))


def test_jump_rectification_passes_positive_activity_and_clips_at_one() -> None:
    escape_v2, escape_v1 = _escape_v2_controllers()
    small = _jump_activity(escape_v2, 0.01)
    assert escape_v2._decode_jump(small) == pytest.approx(0.08, abs=0.0)
    assert escape_v2._decode_jump(small) == escape_v1._decode_jump(small)
    large = _jump_activity(escape_v2, 0.5)
    assert escape_v2._decode_jump(large) == 1.0  # clipped at the top of the codomain
    assert escape_v2._decode_jump(_jump_activity(escape_v2, 0.0)) == 0.0


def test_jump_rectification_applies_the_declared_dead_zone() -> None:
    escape_v2, _ = _escape_v2_controllers(dead_zone=0.25)
    assert escape_v2._decode_jump(_jump_activity(escape_v2, 0.02)) == 0.0  # 0.16 < 0.25
    assert escape_v2._decode_jump(_jump_activity(escape_v2, 0.05)) == pytest.approx(0.4, abs=0.0)


def test_jump_is_still_one_additive_term_on_the_vertical_axis_only() -> None:
    escape_v2, _ = _escape_v2_controllers()
    for step in range(6):
        sensors = _sensors(left=6.0 - 0.5 * step, light=(step * 5) % 16)
        action = escape_v2.step(_observation(escape_v2, step, sensors), command_seq=step)
        evidence = escape_v2.last_evidence
        assert evidence is not None
        jump = evidence.jump_value
        assert jump is not None and jump >= 0.0  # rectified: never an anti-jump
        ttmn_activity = evidence.state_after_activity[-2:]
        expected_raw = 8.0 * (ttmn_activity[0] + ttmn_activity[1]) / 2.0
        assert jump == min(1.0, max(0.0, expected_raw))
        # The pinned primitive projection of the untouched wing MotorFrame is
        # passed through bit-for-bit on every axis but the vertical.
        projected = ConnectomeController._project_motor_frame(evidence.motor_frame)
        assert action.motors.forward_thrust == projected.forward_thrust
        assert action.motors.lateral_thrust == projected.lateral_thrust
        assert action.motors.yaw_torque == projected.yaw_torque
        assert action.motors.pitch_torque == projected.pitch_torque
        assert action.motors.vertical_thrust == min(
            1.0, max(-1.0, projected.vertical_thrust + jump)
        )


# ------------------------------------------------------------------ decoder document


def test_escape_v2_decoder_rotates_under_its_own_schema_and_formula() -> None:
    escape_v2, escape_v1 = _escape_v2_controllers()
    binding = escape_v2.escape_binding
    v1_binding = escape_v1.escape_binding
    assert isinstance(binding, EscapeMotorBindingV2)
    assert v1_binding is not None and not isinstance(v1_binding, EscapeMotorBindingV2)
    material = binding.to_identity_material()
    assert material["schema"] == FULL_BRAIN_ESCAPE_V2_DECODER_SCHEMA
    assert material["schema"] == "neurocraft.full_brain.decoder.escape.v2"
    assert material["jump_formula"] == ESCAPE_JUMP_FORMULA_V2
    # The untouched wing readout document is chained verbatim.
    assert material["wing"] == binding.wing.to_identity_material()
    assert material["ttmn_pool_binding_sha256"] == v1_binding.ttmn_pool_identity_sha256
    assert binding.identity_sha256 != v1_binding.identity_sha256
    # The adapter rotates too (escape-v1's was byte-identical to eem-v1's).
    assert escape_v2.identity.adapter_hash != escape_v1.identity.adapter_hash
    assert escape_v2.identity.decoder_hash != escape_v1.identity.decoder_hash
    assert escape_v2.identity.dynamics_hash == escape_v1.identity.dynamics_hash
    assert escape_v2.identity.graph_hash == escape_v1.identity.graph_hash
    assert escape_v2.identity.observation_schema_hash == escape_v1.identity.observation_schema_hash
    assert escape_v2.configuration.identity_sha256 != escape_v1.configuration.identity_sha256


def test_escape_decoder_bindings_refuse_each_other_s_formula() -> None:
    escape_v2, escape_v1 = _escape_v2_controllers()
    binding = escape_v2.escape_binding
    v1_binding = escape_v1.escape_binding
    assert binding is not None and v1_binding is not None
    with pytest.raises(FullBrainProfileError, match="declared rectified jump"):
        EscapeMotorBindingV2(
            wing=binding.wing,
            jump_pool=binding.jump_pool,
            jump_scale=binding.jump_scale,
            ttmn_pool_identity_sha256=binding.ttmn_pool_identity_sha256,
            encoder_identity_sha256=binding.encoder_identity_sha256,
            jump_formula=ESCAPE_JUMP_FORMULA,
        )
    with pytest.raises(FullBrainProfileError, match="declared jump mapping formula"):
        EscapeMotorBinding(
            wing=v1_binding.wing,
            jump_pool=v1_binding.jump_pool,
            jump_scale=v1_binding.jump_scale,
            ttmn_pool_identity_sha256=v1_binding.ttmn_pool_identity_sha256,
            encoder_identity_sha256=v1_binding.encoder_identity_sha256,
            jump_formula=ESCAPE_JUMP_FORMULA_V2,
        )


# -------------------------------------------------------------- configuration gates


def test_escape_v2_configuration_has_its_own_schema_and_refusals() -> None:
    config = FullBrainEscapeV2Configuration(encoder_identity_sha256=ESCAPE_V2_ENCODER_SHA)
    assert config.schema == FULL_BRAIN_ESCAPE_V2_CONFIG_SCHEMA
    assert config.schema == "neurocraft.controller.full_brain_escape_config.v2"
    material = config.to_identity_material()
    assert material["mode_taxonomy"] == "engineered_encoder"
    assert material["jump_scale"] == (1.0).hex()
    assert material["encoder_identity_sha256"] == ESCAPE_V2_ENCODER_SHA
    # Same encoder identity, distinct schema: never the same configuration.
    assert (
        config.identity_sha256
        != FullBrainEscapeConfiguration(
            encoder_identity_sha256=ESCAPE_V2_ENCODER_SHA
        ).identity_sha256
    )
    with pytest.raises(ValueError, match="unsupported escape-reflex v2"):
        FullBrainEscapeV2Configuration(
            schema=FULL_BRAIN_ESCAPE_CONFIG_SCHEMA,
            encoder_identity_sha256=ESCAPE_V2_ENCODER_SHA,
        )
    with pytest.raises(ValueError, match="encoder document identity"):
        FullBrainEscapeV2Configuration()
    with pytest.raises(ValueError, match="mode_taxonomy"):
        FullBrainEscapeV2Configuration(
            encoder_identity_sha256=ESCAPE_V2_ENCODER_SHA, mode_taxonomy="faithful"
        )
    with pytest.raises(ValueError, match="requires the SHA-256"):
        FullBrainEscapeV2Configuration(
            encoder_identity_sha256=ESCAPE_V2_ENCODER_SHA, jump_scale=500.0
        )
    with pytest.raises(ValueError, match="must carry the measured"):
        FullBrainEscapeV2Configuration(
            encoder_identity_sha256=ESCAPE_V2_ENCODER_SHA,
            readout_scale=76.0,
            readout_calibration_sha256="6f" * 32,
            jump_scale=1.0,
        )
    with pytest.raises(ValueError, match="jump_scale must be finite"):
        FullBrainEscapeV2Configuration(
            encoder_identity_sha256=ESCAPE_V2_ENCODER_SHA,
            readout_scale=76.0,
            readout_calibration_sha256="6f" * 32,
            jump_scale=float("inf"),
        )
    with pytest.raises(ValueError, match="five neural steps"):
        FullBrainEscapeV2Configuration(
            encoder_identity_sha256=ESCAPE_V2_ENCODER_SHA, neural_steps_per_game_tick=4
        )


# ------------------------------------------------- configuration/document pairings


def _rebuild(
    controller: FullBrainController,
    *,
    configuration: Any,
    embodied: Any,
    escape: Any,
) -> None:
    FullBrainController(
        run_id="pairing-test",
        entity_id="test-fly",
        configuration=configuration,
        engine=controller.engine,
        sensory_binding=controller._sensory_binding,
        motor_binding=controller._motor_binding,
        embodied=embodied,
        escape=escape,
    )


def test_controller_refuses_every_escape_v2_encoder_mispairing() -> None:
    """Both directions, against the v1 absolute and the eem-v2 documents."""

    escape_v2, escape_v1 = _escape_v2_controllers()
    senses_v2 = escape_v2.embodied_senses
    senses_v1 = escape_v1.embodied_senses
    assert senses_v2 is not None and senses_v1 is not None
    senses_eem_v2 = bind_embodied_senses_v2(senses_v2.populations)
    binding_v2 = escape_v2.escape_binding
    binding_v1 = escape_v1.escape_binding
    assert binding_v2 is not None and binding_v1 is not None

    # An escape-v2 configuration over the v1 absolute encoder document.
    with pytest.raises(FullBrainProfileError, match=r"escape-v2 .*must be paired"):
        _rebuild(
            escape_v2,
            configuration=FullBrainEscapeV2Configuration(
                encoder_identity_sha256=senses_v1.encoder.identity_sha256
            ),
            embodied=senses_v1,
            escape=binding_v2,
        )
    # An escape-v2 configuration over the eem-v2 all-channel-contrast document.
    with pytest.raises(FullBrainProfileError, match=r"escape-v2 .*must be paired"):
        _rebuild(
            escape_v2,
            configuration=FullBrainEscapeV2Configuration(
                encoder_identity_sha256=senses_eem_v2.encoder.identity_sha256
            ),
            embodied=senses_eem_v2,
            escape=binding_v2,
        )
    # A v1 escape configuration over the escape-v2 encoder document.
    with pytest.raises(FullBrainProfileError, match=r"escape-v2 .*must be paired"):
        _rebuild(
            escape_v2,
            configuration=FullBrainEscapeConfiguration(
                encoder_identity_sha256=senses_v2.encoder.identity_sha256
            ),
            embodied=senses_v2,
            escape=binding_v1,
        )
    # An eem-v2 (non-escape) configuration over the escape-v2 encoder document.
    with pytest.raises(FullBrainProfileError, match=r"escape-v2 .*must be paired"):
        _rebuild(
            escape_v2,
            configuration=FullBrainEemV2Configuration(
                encoder_identity_sha256=senses_v2.encoder.identity_sha256
            ),
            embodied=senses_v2,
            escape=None,
        )
    # An eem-v1 (non-escape) configuration over the escape-v2 encoder document.
    with pytest.raises(FullBrainProfileError, match=r"escape-v2 .*must be paired"):
        _rebuild(
            escape_v2,
            configuration=FullBrainEemConfiguration(
                encoder_identity_sha256=senses_v2.encoder.identity_sha256
            ),
            embodied=senses_v2,
            escape=None,
        )
    # And the eem-v2 lane is untouched: its own pairing still constructs.
    _rebuild(
        escape_v2,
        configuration=FullBrainEemV2Configuration(
            encoder_identity_sha256=senses_eem_v2.encoder.identity_sha256
        ),
        embodied=senses_eem_v2,
        escape=None,
    )


def test_controller_refuses_a_v1_v2_escape_decoder_mix_in_both_directions() -> None:
    escape_v2, escape_v1 = _escape_v2_controllers()
    senses_v2 = escape_v2.embodied_senses
    senses_v1 = escape_v1.embodied_senses
    binding_v2 = escape_v2.escape_binding
    binding_v1 = escape_v1.escape_binding
    assert senses_v2 is not None and senses_v1 is not None
    assert binding_v2 is not None and binding_v1 is not None
    with pytest.raises(FullBrainProfileError, match="v1/v2 escape mix is refused"):
        _rebuild(
            escape_v2,
            configuration=escape_v2.configuration,
            embodied=senses_v2,
            escape=binding_v1,
        )
    with pytest.raises(FullBrainProfileError, match="v1/v2 escape mix is refused"):
        _rebuild(
            escape_v1,
            configuration=escape_v1.configuration,
            embodied=senses_v1,
            escape=binding_v2,
        )


# ----------------------------------------------------- calibration cross-refusals


def _escape_v2_calibration_document(
    *,
    encoder_sha256: str = ESCAPE_V2_ENCODER_SHA,
    ttmn_sha256: str = "7a" * 32,
    wing_scale: float = 76.0,
    jump_scale: float = 500.0,
) -> dict[str, Any]:
    document = _escape_calibration_document(
        encoder_sha256=encoder_sha256,
        ttmn_sha256=ttmn_sha256,
        wing_scale=wing_scale,
        jump_scale=jump_scale,
    )
    document["schema"] = FULL_READOUT_CALIBRATION_ESCAPE_V2_SCHEMA
    document["stream"] = {"stream_id": "unit-test-escape-stream-v2"}
    return document


def test_escape_v2_calibration_loads_and_binds_its_own_boundary() -> None:
    calibration = load_escape_v2_readout_calibration(
        canonical_json_bytes(_escape_v2_calibration_document())
    )
    assert calibration.wing_scale == 76.0
    assert calibration.jump_scale == 500.0
    assert calibration.encoder_identity_sha256 == ESCAPE_V2_ENCODER_SHA
    calibration.assert_binds(
        artifact_manifest_sha256=GRAPH_SHA,
        sign_overlay_identity_sha256=OVERLAY_SHA,
        encoder_identity_sha256=ESCAPE_V2_ENCODER_SHA,
        ttmn_pool_identity_sha256="7a" * 32,
    )
    with pytest.raises(FullBrainProfileError, match="different encoder document"):
        calibration.assert_binds(
            artifact_manifest_sha256=GRAPH_SHA,
            sign_overlay_identity_sha256=OVERLAY_SHA,
            encoder_identity_sha256=EEM_ENCODER_SHA,
            ttmn_pool_identity_sha256="7a" * 32,
        )
    with pytest.raises(FullBrainProfileError, match="different TTMn pool binding"):
        calibration.assert_binds(
            artifact_manifest_sha256=GRAPH_SHA,
            sign_overlay_identity_sha256=OVERLAY_SHA,
            encoder_identity_sha256=ESCAPE_V2_ENCODER_SHA,
            ttmn_pool_identity_sha256="8b" * 32,
        )


def test_every_calibration_boundary_refuses_escape_v2_and_vice_versa() -> None:
    """All four directed pairs against the escape-v2 calibration schema."""

    escape_v2_bytes = canonical_json_bytes(_escape_v2_calibration_document())
    escape_v1_bytes = canonical_json_bytes(_escape_calibration_document())
    v1_documents = {
        "wing": _calibration_document(encoder_sha256=None),
        "eem_v1": _calibration_document(encoder_sha256=EEM_ENCODER_SHA),
        "eem_v2": _calibration_document(encoder_sha256=EEM_V2_ENCODER_SHA),
    }
    # The escape-v2 loader refuses every other boundary by name.
    for document in v1_documents.values():
        with pytest.raises(FullBrainProfileError, match="escape-v2 boundary refuses"):
            load_escape_v2_readout_calibration(canonical_json_bytes(document))
    with pytest.raises(FullBrainProfileError, match="escape-v1 calibration"):
        load_escape_v2_readout_calibration(escape_v1_bytes)
    # And every other boundary refuses the escape-v2 document.
    with pytest.raises(FullBrainProfileError, match="escape-v2 calibration"):
        load_escape_readout_calibration(escape_v2_bytes)
    with pytest.raises(FullBrainProfileError, match="unsupported readout calibration schema"):
        load_readout_calibration(escape_v2_bytes)
    # An unknown schema at the escape-v2 loader is refused too.
    stray = _escape_v2_calibration_document()
    stray["schema"] = "neurocraft.full_readout_calibration.made.up"
    with pytest.raises(
        FullBrainProfileError, match="unsupported escape-v2 readout calibration schema"
    ):
        load_escape_v2_readout_calibration(canonical_json_bytes(stray))


def test_escape_v2_calibration_scale_validation_names_its_own_boundary() -> None:
    identity_scale = _escape_v2_calibration_document(jump_scale=1.0)
    with pytest.raises(FullBrainProfileError, match=r"escape-v2 calibration.*jump_scale 1\.0"):
        load_escape_v2_readout_calibration(canonical_json_bytes(identity_scale))
    disagreeing = _escape_v2_calibration_document()
    disagreeing["calibration"]["wing_scale"] = 77.0
    with pytest.raises(FullBrainProfileError, match="disagrees with its float hex"):
        load_escape_v2_readout_calibration(canonical_json_bytes(disagreeing))


# ------------------------------------------------ from_artifact argument gates


def test_from_artifact_refuses_escape_version_misuse(tmp_path: Path) -> None:
    """Every refusal fires before any artifact byte is read; all presence
    gates are faked with tmp paths so the intended gate is the one reached."""

    with pytest.raises(FullBrainProfileError, match="escape_version must be 1 or 2"):
        FullBrainController.from_artifact(
            run_id="refusal",
            entity_id="fly",
            artifact_directory=tmp_path / "absent",
            annotations_path=tmp_path / "absent.feather",
            embodied_senses=True,
            body_annotations_path=tmp_path / "absent-annotations.feather",
            escape_reflex=True,
            escape_version=3,
        )
    with pytest.raises(FullBrainProfileError, match="escape-reflex profile only"):
        FullBrainController.from_artifact(
            run_id="refusal",
            entity_id="fly",
            artifact_directory=tmp_path / "absent",
            annotations_path=tmp_path / "absent.feather",
            embodied_senses=True,
            body_annotations_path=tmp_path / "absent-annotations.feather",
            escape_version=2,
        )
    with pytest.raises(FullBrainProfileError, match="full_mcns_escape_v2 is a local-only"):
        FullBrainController.from_artifact(
            run_id="refusal",
            entity_id="fly",
            artifact_directory=tmp_path / "absent",
            annotations_path=tmp_path / "absent.feather",
            embodied_senses=True,
            body_annotations_path=tmp_path / "absent-annotations.feather",
            escape_reflex=True,
            escape_version=2,
        )


# ------------------------------------------------------- calibration script wiring


def test_calibration_script_registers_the_escape_v2_profile() -> None:
    """Argument surface only -- NO calibration is run (phase-2 deliverable)."""

    import scripts.calibrate_full_readout as calibrate

    assert calibrate.ESCAPE_V2_PROFILE == "escape-v2"
    assert calibrate.ESCAPE_V2_STREAM_ID == "full-readout-calibration-escape-v2"
    assert calibrate.ESCAPE_PROFILES == (calibrate.ESCAPE_PROFILE, calibrate.ESCAPE_V2_PROFILE)
    assert calibrate.DEFAULT_ESCAPE_V2_OUTPUT.name == "full-readout-calibration-escape-v2.json"
    # Every profile writes its own document under its own stream id.
    outputs = {
        calibrate.DEFAULT_OUTPUT,
        calibrate.DEFAULT_EEM_OUTPUT,
        calibrate.DEFAULT_EEM_V2_OUTPUT,
        calibrate.DEFAULT_ESCAPE_OUTPUT,
        calibrate.DEFAULT_ESCAPE_V2_OUTPUT,
    }
    assert len(outputs) == 5
    with pytest.raises(calibrate.ReadoutCalibrationError, match="unknown escape calibration"):
        calibrate.build_escape_document(None, {}, profile="wing")  # type: ignore[arg-type]


def test_calibration_script_leaves_the_escape_v1_document_bytes_unchanged() -> None:
    """The escape-v1 builder strings still match the committed document.

    ``write_or_verify`` refuses any document whose bytes differ from the
    committed one, so this lane's parameterization of ``build_escape_document``
    must reproduce every escape-v1 literal exactly. Every field the builder
    emits from a module constant or a profile branch -- i.e. every field whose
    value does not come from the controller or from the measured stream -- is
    pinned here against the committed bytes.
    """

    import json

    import scripts.calibrate_full_readout as calibrate

    committed = json.loads(calibrate.DEFAULT_ESCAPE_OUTPUT.read_bytes())
    assert committed["schema"] == calibrate.FULL_READOUT_CALIBRATION_ESCAPE_SCHEMA
    assert committed["profile"] == "full_mcns_escape_v1"
    assert committed["stream"]["stream_id"] == calibrate.ESCAPE_STREAM_ID
    assert committed["stream"]["warmup_ticks"] == calibrate.WARMUP_TICKS
    assert committed["stream"]["measured_ticks"] == calibrate.MEASURED_TICKS
    assert committed["stream"]["observation_derivation"] == (
        f"rays (bytes 0:12) and light (byte 12, uniform 0..15) from "
        f"sha256('{calibrate.ESCAPE_STREAM_ID}:<tick>') bytes; horizontal/vertical "
        "collisions from bytes 13:15 / 15:17 as declared "
        f"Bernoulli({calibrate.COLLISION_THRESHOLD}/{calibrate.COLLISION_DENOMINATOR}); "
        "deterministic, no PRNG"
    )
    assert committed["encoder"]["note"] == (
        "binds this calibration to the exact engineered-encoder document "
        "(the eem v1 absolute encoder) and population membership it was "
        "measured under; every v1-schema calibration is refused by the "
        "escape loader and vice versa"
    )
    assert committed["jump_readout"]["formula"] == ESCAPE_JUMP_FORMULA
    assert committed["calibration"]["derivation"] == (
        f"wing_scale = {calibrate.TARGET_P95_ACTUATION} / (wing p95 * motor_pool_gain); "
        f"jump_scale = {calibrate.TARGET_P95_ACTUATION} / (jump p95 * motor_pool_gain)"
    )
    assert committed["claim_scope"] == (
        "declared_boundary_unit_calibration_only_no_behavior_performance_or_biological_claim"
    )
    collision = committed["stream"]["collision_derivation"]
    assert collision["threshold"] == calibrate.COLLISION_THRESHOLD
    assert collision["denominator"] == calibrate.COLLISION_DENOMINATOR
    assert collision["probability"] == calibrate._float_pair(
        calibrate.COLLISION_THRESHOLD / calibrate.COLLISION_DENOMINATOR
    )
    assert collision["note"] == (
        "the closest two-byte approximation of the declared "
        "Bernoulli p = 0.1; the exact executed probability is "
        "6554/65536"
    )
    assert committed["stream"]["ray_max_distance_m"] == calibrate._float_pair(
        calibrate.RAY_MAX_DISTANCE_M
    )
    assert committed["measured_under"]["note"] == (
        "the pre-calibration (readout_scale 1.0, jump_scale 1.0) controller "
        "identity the distributions below were measured with; the calibrated "
        "controller necessarily carries a different configuration and decoder "
        "identity"
    )
    assert committed["distributions"]["wing"]["quantity"] == (
        "abs(pre-gain, pre-scale pool mean) = "
        "abs(fsum(weight * activity) / summed_abs_weight), pooled over both "
        "wing-motor pools and every measured tick"
    )
    assert committed["distributions"]["jump"]["quantity"] == (
        "abs(pre-gain, pre-scale TTMn jump pool mean) = "
        "abs(fsum(weight * activity) / summed_abs_weight) over the two "
        "TTMn members, every measured tick"
    )
    for group in ("wing", "jump"):
        assert committed["distributions"][group]["percentile_method"] == calibrate.PERCENTILE_METHOD
    assert committed["calibration"]["target_p95_actuation"] == calibrate._float_pair(
        calibrate.TARGET_P95_ACTUATION
    )
    assert committed["calibration"]["disclosure"]["note"] == (
        "the scales that would put each group's measured MAXIMUM pool "
        "magnitude exactly at the +-1 clip; recorded so the headroom this "
        "calibration leaves is auditable, not applied"
    )


def test_escape_v2_evidence_subset_still_appends_exactly_the_ttmn_members() -> None:
    escape_v2, escape_v1 = _escape_v2_controllers()
    assert escape_v2._evidence_body_ids == escape_v1._evidence_body_ids
    assert escape_v2._evidence_body_ids[-2:] == (TTMN_RIGHT_BODY, TTMN_LEFT_BODY)


def test_two_escape_v2_controllers_replay_identically() -> None:
    first, _ = _escape_v2_controllers()
    second, _ = _escape_v2_controllers()
    assert first.identity == second.identity
    for step in range(4):
        sensors = _sensors(left=6.0 - step, horizontal_collision=step >= 1)
        action_a = first.step(_observation(first, step, sensors), command_seq=step)
        action_b = second.step(_observation(second, step, sensors), command_seq=step)
        assert action_a.state_digest == action_b.state_digest
        evidence_a = first.last_evidence
        evidence_b = second.last_evidence
        assert evidence_a is not None and evidence_b is not None
        assert evidence_a.action_sha256 == evidence_b.action_sha256
        assert evidence_a.jump_value == evidence_b.jump_value
