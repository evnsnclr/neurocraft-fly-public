"""Tests for the approach-gated looming profile (``full_mcns_loom_v1``).

The lane's one declared change against escape-v2: the three threat channels
move under a RECTIFIED approach contrast (``clip(raw_t - m_(t-1), 0, 1)``)
instead of staying absolute, so an approaching object fires LC4 and a standing
or receding one does not. ``touch`` keeps the escape-v2 transient encoding
byte-identically, ``light`` stays absolute, and the decoder (the rectified
TTMn jump term plus the wing common term) is the escape-v2 one unchanged.

All CI-safe: synthetic annotation frames and synthetic views only -- no
artifact and no feather tables. The full-artifact tests live beside the other
lanes' and follow the same skip-if-absent pattern. The v1/eem-v2/escape-v1/
escape-v2 identity pins asserted in the sibling suites are re-asserted here,
because this additive lane must not move a bit of any of them.
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
    ESCAPE_V2_CHANNELS,
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
from neurocraft_controller.escape_reflex import select_ttmn_pool
from neurocraft_controller.flight_pools import (
    LEFT_WING_MOTOR_BODY_IDS,
    RIGHT_WING_MOTOR_BODY_IDS,
)
from neurocraft_controller.full_brain_controller import (
    FULL_BRAIN_EEM_ESCAPE_V2_ENCODED_STIMULATION_SCHEMA,
    FULL_BRAIN_EEM_LOOM_V1_ENCODED_STIMULATION_SCHEMA,
    FULL_BRAIN_EEM_V2_ENCODED_STIMULATION_SCHEMA,
    FULL_BRAIN_ESCAPE_V2_CONFIG_SCHEMA,
    FULL_BRAIN_ESCAPE_V2_DECODER_SCHEMA,
    FULL_BRAIN_LOOM_V1_CONFIG_SCHEMA,
    FULL_READOUT_CALIBRATION_LOOM_V1_SCHEMA,
    EscapeMotorBinding,
    EscapeMotorBindingV2,
    FullBrainController,
    FullBrainEemConfiguration,
    FullBrainEemV2Configuration,
    FullBrainEscapeConfiguration,
    FullBrainEscapeV2Configuration,
    FullBrainLoomV1Configuration,
    FullBrainProfileError,
    bind_escape_jump_pool,
    bind_sensor_channels,
    bind_wing_motor_pools,
    load_escape_readout_calibration,
    load_escape_v2_readout_calibration,
    load_loom_v1_readout_calibration,
    load_readout_calibration,
)
from neurocraft_controller.loom_senses import (
    LOOM_V1_ABSOLUTE_CHANNEL_NAMES,
    LOOM_V1_CHANNELS,
    LOOM_V1_CONTRAST_CHANNEL_NAMES,
    LOOM_V1_ENCODER_SCHEMA,
    LOOM_V1_RECTIFIED_CHANNEL_NAMES,
    EemEncoderDocumentLoomV1,
    bind_embodied_senses_loom_v1,
)
from neurocraft_neural.full_engine import ArrayNativeEngine
from neurocraft_protocol import canonical_json_bytes

from tests.controller.test_embodied_senses import (
    ANNOTATIONS_SHA,
    CURRENT_EEM_POPULATION_BINDING_SHA,
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
from tests.controller.test_escape_reflex_v2 import (
    ESCAPE_V2_ENCODER_SHA,
    ESCAPE_V2_UNCALIBRATED_CONFIG_SHA,
    _escape_v2_calibration_document,
)

#: The loom-v1 identity pins. Both derive WITHOUT the artifact: the encoder
#: document hashes only declared material plus the pinned population-binding
#: identity, and the configuration hashes only declared material plus that
#: encoder identity (the escape-v1/escape-v2/eem derivations verbatim).
LOOM_V1_ENCODER_SHA = "9f4b8876a2d3a50380aaf0903d16b157e33cfdb776338d53ac5097432289b126"
LOOM_V1_UNCALIBRATED_CONFIG_SHA = "7240492264f71b1af276e5c9ec763eecdc8c73e9aebace2da6e4c2417b9e193c"

#: The committed loom-v1 calibration is checked on CI without the local full artifact.
REPO_ROOT = Path(__file__).resolve().parents[2]
LOOM_V1_CALIBRATION_PATH = (
    REPO_ROOT / "data" / "calibration" / "full-readout-calibration-loom-v1.json"
)

#: The committed document's SHA-256 and the two scales it declares. Measured
#: by ``scripts/calibrate_full_readout.py --profile loom-v1 --threads 8`` on
#: 2026-10-24 after the portable artifact/kernel rebuild. Historical synthetic
#: identity constants remain separate from these current measured document pins.
LOOM_V1_CALIBRATION_SHA = "d1dd2ef9c77f8aa6a17b06609e28a65e9fa4bbcfb5e74fb6602a9eb58dd61ba0"
CURRENT_LOOM_V1_ENCODER_SHA = "c9c9437c90874c4166d4c294c9c33d948fdcb1fbab1894e2c7094db90497f6b5"
LOOM_V1_WING_SCALE_HEX = "0x1.a0fd0307362bep+6"  # 104.24708186405311
LOOM_V1_JUMP_SCALE_HEX = "0x1.b4945f22255b6p+2"  # 6.821555884690886
#: The uncalibrated controller identities the document says it was measured
#: under. ``dynamics_hash`` is the C-kernel identity every committed
#: calibration in this repository carries, so a numpy-fallback measurement
#: could never reproduce this document.
LOOM_V1_MEASURED_UNDER_CONFIG_SHA = (
    "b4712e83a6eaa1a64a138164afc5398988de9ae1b599ed770f931b31d948e5aa"
)
LOOM_V1_MEASURED_UNDER_DYNAMICS_SHA = (
    "24257e5d1cc3f632c66a4d4bdd9e3944c9ee5066561359e3b1b259b1da71ddbe"
)


def _loom_controllers(*, dead_zone: float = 0.0) -> tuple[FullBrainController, FullBrainController]:
    """A matched (loom-v1, escape-v2) pair over ONE synthetic node space.

    Both controllers share the identical view, sensory binding, wing motor
    binding, TTMn pool, and decoder generation, so the ONLY declared
    difference is the encoder document.
    """

    frame = _ttmn_frame()
    index = frame.index
    cursor = frame.rows
    for body in LEFT_WING_MOTOR_BODY_IDS + RIGHT_WING_MOTOR_BODY_IDS:
        index[body] = cursor
        cursor += 1
    node_count = cursor
    populations = frame.select(node_count=node_count)
    senses_loom = bind_embodied_senses_loom_v1(populations)
    senses_escape_v2 = bind_embodied_senses_escape_v2(populations)
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

    def build(senses: Any, configuration: Any) -> FullBrainController:
        return FullBrainController(
            run_id="loom-v1-test",
            entity_id="test-fly",
            configuration=configuration,
            engine=ArrayNativeEngine(_synthetic_view(node_count)),
            sensory_binding=sensory,
            motor_binding=motor,
            embodied=senses,
            escape=EscapeMotorBindingV2(
                wing=motor,
                jump_pool=jump_pool,
                jump_scale=1.0,
                ttmn_pool_identity_sha256=ttmn.identity_sha256,
                encoder_identity_sha256=senses.encoder.identity_sha256,
            ),
        )

    loom = build(
        senses_loom,
        FullBrainLoomV1Configuration(
            motor_dead_zone=dead_zone,
            encoder_identity_sha256=senses_loom.encoder.identity_sha256,
        ),
    )
    escape_v2 = build(
        senses_escape_v2,
        FullBrainEscapeV2Configuration(
            motor_dead_zone=dead_zone,
            encoder_identity_sha256=senses_escape_v2.encoder.identity_sha256,
        ),
    )
    return loom, escape_v2


# ---------------------------------------------------------- the encoder document


def test_loom_encoder_rectifies_exactly_the_three_threat_channels() -> None:
    encoder = EemEncoderDocumentLoomV1(population_binding_sha256=EEM_POPULATION_BINDING_SHA)
    assert (
        encoder.rectified_channels
        == LOOM_V1_RECTIFIED_CHANNEL_NAMES
        == ("threat_left", "threat_right", "threat_front")
    )
    assert (
        encoder.contrast_channels
        == LOOM_V1_CONTRAST_CHANNEL_NAMES
        == ("threat_left", "threat_right", "threat_front", "touch")
    )
    assert encoder.absolute_channels == LOOM_V1_ABSOLUTE_CHANNEL_NAMES == ("light",)
    assert set(encoder.rectified_channels) <= set(encoder.contrast_channels)
    assert encoder.alpha == EEM_CONTRAST_ALPHA == 0.125
    material = encoder.to_identity_material()
    assert material["schema"] == LOOM_V1_ENCODER_SCHEMA == "neurocraft.eem.encoder.loom.v1"
    encodings = {entry["name"]: entry["encoding"] for entry in material["channels"]}
    assert encodings == {
        "threat_left": "rectified_approach_contrast",
        "threat_right": "rectified_approach_contrast",
        "threat_front": "rectified_approach_contrast",
        "light": "absolute",
        "touch": "temporal_contrast",
    }
    assert material["contrast"]["rectified_channels"] == list(LOOM_V1_RECTIFIED_CHANNEL_NAMES)
    assert material["contrast"]["update"] == (
        "drive_t = clip(raw_t - m_(t-1), 0, 1); m_t = (1 - alpha) * m_(t-1) + "
        "alpha * raw_t; one update per game tick, never per neural sub-step"
    )


def test_loom_channels_keep_v1_names_units_populations_and_gains() -> None:
    """Only the declared encoding wrapper changes; nothing else moves."""

    assert len(LOOM_V1_CHANNELS) == len(EEM_CHANNELS) == 5
    for loom, v1 in zip(LOOM_V1_CHANNELS, EEM_CHANNELS, strict=True):
        assert loom.name == v1.name
        assert loom.unit == v1.unit
        assert loom.populations == v1.populations
        assert loom.gain == v1.gain == 1.0
        # The v1 raw formula is carried inside the declared wrapper verbatim.
        assert v1.formula in loom.formula
    threat = {channel.name: channel for channel in LOOM_V1_CHANNELS}
    for name in LOOM_V1_RECTIFIED_CHANNEL_NAMES:
        assert threat[name].formula.startswith("rectified approach contrast of the v1 raw value")
    # light is the v1 channel object verbatim; touch is the escape-v2 object verbatim.
    assert threat["light"] == EEM_CHANNELS[3]
    assert threat["touch"] == ESCAPE_V2_CHANNELS[4]


def test_loom_encoder_refuses_an_undeclared_rectified_or_alpha_choice() -> None:
    with pytest.raises(EmbodiedSensesError, match="three threat channels"):
        EemEncoderDocumentLoomV1(
            population_binding_sha256=EEM_POPULATION_BINDING_SHA,
            rectified_channels=("threat_left",),
        )
    with pytest.raises(EmbodiedSensesError, match="alpha = 1/8"):
        EemEncoderDocumentLoomV1(population_binding_sha256=EEM_POPULATION_BINDING_SHA, alpha=0.25)
    for alpha in (0.0625, 0.25, 0.1, 1.0):
        if alpha == EEM_CONTRAST_ALPHA:  # pragma: no cover - guard, never taken
            continue
        with pytest.raises(EmbodiedSensesError, match="alpha = 1/8"):
            EemEncoderDocumentLoomV1(
                population_binding_sha256=EEM_POPULATION_BINDING_SHA, alpha=alpha
            )
    # A rectified channel that is not a contrast channel has no baseline.
    # (Reached only via the partition gate below, which fires first; both
    # messages are asserted so neither gate can quietly stop firing.)
    with pytest.raises(EmbodiedSensesError, match="different partition"):
        EemEncoderDocumentLoomV1(
            population_binding_sha256=EEM_POPULATION_BINDING_SHA,
            rectified_channels=("threat_left", "threat_right", "threat_front"),
            contrast_channels=("threat_left", "threat_right", "touch"),
            absolute_channels=("threat_front", "light"),
        )
    with pytest.raises(EmbodiedSensesError, match="must also be a contrast channel"):
        apply_temporal_contrast(
            encode_channel_values(_sensors()),
            None,
            contrast_channels=("threat_left", "threat_right", "touch"),
            rectified_channels=("threat_front",),
        )


def test_loom_encoder_refuses_an_undeclared_contrast_absolute_partition() -> None:
    """The v2 gate only requires a partition; this identity pins WHICH one.

    Both documents below satisfy alpha = 1/8, the declared rectified set, and
    the v2 partition rule, and both constructed before this gate existed.
    """

    for label, contrast, absolute in (
        (
            "light moved into contrast",
            ("threat_left", "threat_right", "threat_front", "touch", "light"),
            (),
        ),
        (
            "touch moved into absolute",
            ("threat_left", "threat_right", "threat_front"),
            ("light", "touch"),
        ),
    ):
        assert label
        with pytest.raises(EmbodiedSensesError, match="different partition"):
            EemEncoderDocumentLoomV1(
                population_binding_sha256=EEM_POPULATION_BINDING_SHA,
                contrast_channels=contrast,
                absolute_channels=absolute,
            )
    # The declared partition still constructs, at the pinned identity.
    assert (
        EemEncoderDocumentLoomV1(
            population_binding_sha256=EEM_POPULATION_BINDING_SHA,
            contrast_channels=LOOM_V1_CONTRAST_CHANNEL_NAMES,
            absolute_channels=LOOM_V1_ABSOLUTE_CHANNEL_NAMES,
        ).identity_sha256
        == LOOM_V1_ENCODER_SHA
    )


def test_loom_identity_pins_derive_without_the_artifact() -> None:
    """Pure-hashing pins, plus every neighbouring lane's regression pin."""

    encoder = EemEncoderDocumentLoomV1(population_binding_sha256=EEM_POPULATION_BINDING_SHA)
    assert encoder.identity_sha256 == LOOM_V1_ENCODER_SHA
    assert (
        FullBrainLoomV1Configuration(encoder_identity_sha256=LOOM_V1_ENCODER_SHA).identity_sha256
        == LOOM_V1_UNCALIBRATED_CONFIG_SHA
    )
    # Byte-unchanged: the identities this additive lane must not move.
    assert (
        EemEncoderDocument(population_binding_sha256=EEM_POPULATION_BINDING_SHA).identity_sha256
        == EEM_ENCODER_SHA
    )
    assert (
        EemEncoderDocumentV2(population_binding_sha256=EEM_POPULATION_BINDING_SHA).identity_sha256
        == EEM_V2_ENCODER_SHA
    )
    assert (
        EemEncoderDocumentEscapeV2(
            population_binding_sha256=EEM_POPULATION_BINDING_SHA
        ).identity_sha256
        == ESCAPE_V2_ENCODER_SHA
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
    assert (
        FullBrainEscapeV2Configuration(
            encoder_identity_sha256=ESCAPE_V2_ENCODER_SHA
        ).identity_sha256
        == ESCAPE_V2_UNCALIBRATED_CONFIG_SHA
    )
    # Every loom identity is distinct from every other lane's.
    assert (
        len(
            {
                LOOM_V1_ENCODER_SHA,
                LOOM_V1_UNCALIBRATED_CONFIG_SHA,
                ESCAPE_V2_ENCODER_SHA,
                ESCAPE_V2_UNCALIBRATED_CONFIG_SHA,
                EEM_ENCODER_SHA,
                EEM_V2_ENCODER_SHA,
                EEM_V1_UNCALIBRATED_CONFIG_SHA,
                EEM_V2_UNCALIBRATED_CONFIG_SHA,
                ESCAPE_UNCALIBRATED_CONFIG_SHA,
            }
        )
        == 9
    )
    # Same encoder identity, distinct schema: never the same configuration.
    assert (
        FullBrainEscapeV2Configuration(encoder_identity_sha256=LOOM_V1_ENCODER_SHA).identity_sha256
        != LOOM_V1_UNCALIBRATED_CONFIG_SHA
    )


# ------------------------------------------------------ the rectified trajectory

#: The blocked frontal ray of the ``ESCAPE-ARC-01`` reference stimulus
#: (``scripts/measure_escape_arc.BLOCKED_RAY_M``), and the raw front-channel
#: value it produces through the byte-identical v1 formula:
#: clip(1 - 0.5/6, 0, 1) / 2.
BLOCKED_RAY_M = 0.5
BLOCKED_THREAT_FRONT_RAW = (1.0 - BLOCKED_RAY_M / 6.0) / 2.0


def _threat_trajectory(
    front_rays: list[float], *, channel: str = "threat_front"
) -> list[tuple[float, float, float]]:
    """Drive the declared loom-v1 rectified contrast over a ray sequence.

    Returns ``(raw, baseline, drive)`` per tick, using the exact machinery the
    controller executes (``apply_temporal_contrast`` with the encoder
    document's own alpha, contrast set, and rectified set).
    """

    encoder = EemEncoderDocumentLoomV1(population_binding_sha256=EEM_POPULATION_BINDING_SHA)
    means: dict[str, float] | None = None
    trajectory: list[tuple[float, float, float]] = []
    for distance in front_rays:
        raw_values = encode_channel_values(_sensors(forward=distance))
        encoded, means_used, means = apply_temporal_contrast(
            raw_values,
            means,
            alpha=encoder.alpha,
            contrast_channels=encoder.contrast_channels,
            rectified_channels=encoder.rectified_channels,
        )
        trajectory.append(
            (dict(raw_values)[channel], dict(means_used)[channel], dict(encoded)[channel])
        )
    return trajectory


def test_first_tick_threat_drive_is_zero_whatever_the_ray() -> None:
    """The declared first tick: baseline := raw_0, so drive_0 = 0 exactly."""

    for distance in (6.0, 3.0, BLOCKED_RAY_M, 0.0):
        first = _threat_trajectory([distance])[0]
        assert first[1] == first[0]
        assert first[2] == 0.0


def test_approach_fires_and_a_standing_obstacle_decays_to_exactly_zero() -> None:
    # One open tick to set the baseline, then the blocked ray held.
    trajectory = _threat_trajectory([6.0, *([BLOCKED_RAY_M] * 80)])
    assert trajectory[0][2] == 0.0  # declared first tick
    onset = trajectory[1]
    assert onset[0] == BLOCKED_THREAT_FRONT_RAW and onset[1] == 0.0
    assert onset[2] == BLOCKED_THREAT_FRONT_RAW  # the rise IS the drive
    held = [entry[2] for entry in trajectory[1:]]
    assert all(later < earlier for earlier, later in pairwise(held))
    # While the ray is held the drive is exactly ``raw - baseline`` (no clip
    # engages, so the executed subtraction IS the declared formula) and the
    # baseline follows the declared leaky-mean recurrence.
    for entry in trajectory[1:]:
        assert entry[0] == BLOCKED_THREAT_FRONT_RAW
        assert entry[2] == entry[0] - entry[1]
    # Geometric at (1 - alpha) per tick, to within one rounding per step.
    for earlier, later in pairwise(held[:20]):
        assert later == pytest.approx(0.875 * earlier, rel=1e-12)
    assert held[-1] < 1e-3
    # Underflow to EXACTLY zero, not merely to something small.
    long_hold = _threat_trajectory([6.0, *([BLOCKED_RAY_M] * 700)])
    assert long_hold[-1][2] == 0.0


def test_receding_gives_zero_drive_on_every_tick() -> None:
    ramp_down = [BLOCKED_RAY_M + (6.0 - BLOCKED_RAY_M) * step / 20.0 for step in range(21)]
    trajectory = _threat_trajectory(ramp_down)
    assert trajectory[0][0] == BLOCKED_THREAT_FRONT_RAW  # starts at the blocked value
    assert all(entry[2] == 0.0 for entry in trajectory)
    # And the two-sided (escape-v2 touch) encoding would NOT have been zero:
    # the rectification is what drops the recede half.
    assert any(abs(entry[0] - entry[1]) > 0.0 for entry in trajectory[1:])


def test_approach_ramp_fires_on_every_rising_tick() -> None:
    ramp_up = [6.0 - (6.0 - BLOCKED_RAY_M) * step / 20.0 for step in range(21)]
    trajectory = _threat_trajectory([*ramp_up, *([BLOCKED_RAY_M] * 20)])
    assert trajectory[0][2] == 0.0  # declared first tick
    assert all(entry[2] > 0.0 for entry in trajectory[1:21])
    # After the ramp the ray is standing: the drive decays, it does not hold.
    standing = [entry[2] for entry in trajectory[21:]]
    assert all(later < earlier for earlier, later in pairwise(standing))


def test_light_stays_absolute_and_bitwise_equal_to_v1() -> None:
    encoder = EemEncoderDocumentLoomV1(population_binding_sha256=EEM_POPULATION_BINDING_SHA)
    means: dict[str, float] | None = None
    for step in range(8):
        raw_values = encode_channel_values(_sensors(light=(step * 3) % 16))
        encoded, _used, means = apply_temporal_contrast(
            raw_values,
            means,
            alpha=encoder.alpha,
            contrast_channels=encoder.contrast_channels,
            rectified_channels=encoder.rectified_channels,
        )
        assert dict(encoded)["light"] == dict(raw_values)["light"]


def test_touch_channel_is_bit_identical_to_escape_v2() -> None:
    """The one channel this lane does not touch, proved against escape-v2."""

    loom = EemEncoderDocumentLoomV1(population_binding_sha256=EEM_POPULATION_BINDING_SHA)
    escape_v2 = EemEncoderDocumentEscapeV2(population_binding_sha256=EEM_POPULATION_BINDING_SHA)
    loom_means: dict[str, float] | None = None
    escape_means: dict[str, float] | None = None
    for flag in [False, True, True, True, True, False, False, True]:
        raw_values = encode_channel_values(_sensors(horizontal_collision=flag))
        loom_encoded, loom_used, loom_means = apply_temporal_contrast(
            raw_values,
            loom_means,
            alpha=loom.alpha,
            contrast_channels=loom.contrast_channels,
            rectified_channels=loom.rectified_channels,
        )
        escape_encoded, escape_used, escape_means = apply_temporal_contrast(
            raw_values,
            escape_means,
            alpha=escape_v2.alpha,
            contrast_channels=escape_v2.contrast_channels,
        )
        assert dict(loom_encoded)["touch"] == dict(escape_encoded)["touch"]
        assert dict(loom_used)["touch"] == dict(escape_used)["touch"]
        assert loom_means["touch"] == escape_means["touch"]


def test_rectified_channels_default_to_empty_so_existing_callers_are_unchanged() -> None:
    """The v2/escape-v2 arithmetic is byte-unchanged without the new keyword."""

    escape_v2 = EemEncoderDocumentEscapeV2(population_binding_sha256=EEM_POPULATION_BINDING_SHA)
    means_default: dict[str, float] | None = None
    means_explicit: dict[str, float] | None = None
    for step in range(6):
        raw_values = encode_channel_values(
            _sensors(forward=6.0 - 0.5 * step, horizontal_collision=step % 2 == 0)
        )
        default, _used_a, means_default = apply_temporal_contrast(
            raw_values, means_default, contrast_channels=escape_v2.contrast_channels
        )
        explicit, _used_b, means_explicit = apply_temporal_contrast(
            raw_values,
            means_explicit,
            contrast_channels=escape_v2.contrast_channels,
            rectified_channels=(),
        )
        assert default == explicit
    with pytest.raises(EmbodiedSensesError, match="must also be a contrast channel"):
        apply_temporal_contrast(
            encode_channel_values(_sensors()),
            None,
            contrast_channels=("touch",),
            rectified_channels=("threat_front",),
        )


# ------------------------------------------------------------- live controller


def test_live_controller_reproduces_the_declared_rectified_trajectory() -> None:
    loom, escape_v2 = _loom_controllers()
    rays = [6.0, 3.0, BLOCKED_RAY_M, BLOCKED_RAY_M, BLOCKED_RAY_M, 6.0]
    expected = _threat_trajectory(rays)
    for step, distance in enumerate(rays):
        sensors = _sensors(forward=distance)
        loom.step(_observation(loom, step, sensors), command_seq=step)
        escape_v2.step(_observation(escape_v2, step, sensors), command_seq=step)
        evidence = loom.last_evidence
        v2_evidence = escape_v2.last_evidence
        assert evidence is not None and v2_evidence is not None
        encoded = evidence.encoded_stimulation
        assert encoded["eem_raw:threat_front"] == expected[step][0]
        assert encoded["eem_mean:threat_front"] == expected[step][1]
        assert encoded["eem:threat_front"] == expected[step][2]
        # escape-v2 drives the same population with the ABSOLUTE level.
        assert v2_evidence.encoded_stimulation["eem:threat_front"] == expected[step][0]


def test_live_controller_decays_the_lc4_drive_under_a_standing_wall() -> None:
    """The recorded live defect, as a unit assertion.

    Under escape-v2 a held frontal wall is a held per-capita LC4 drive; under
    loom-v1 the same wall has decayed below 1e-3 by tick 79. The drive reaches
    EXACTLY zero later, which
    :func:`test_approach_fires_and_a_standing_obstacle_decays_to_exactly_zero`
    proves over a 700-tick hold; this test asserts only the 80-tick threshold
    its loop actually runs.
    """

    loom, escape_v2 = _loom_controllers()
    for step in range(80):
        distance = 6.0 if step == 0 else BLOCKED_RAY_M
        sensors = _sensors(forward=distance)
        loom.step(_observation(loom, step, sensors), command_seq=step)
        escape_v2.step(_observation(escape_v2, step, sensors), command_seq=step)
    loom_evidence = loom.last_evidence
    v2_evidence = escape_v2.last_evidence
    assert loom_evidence is not None and v2_evidence is not None
    for population in ("lc4_left", "lc4_right"):
        assert loom_evidence.encoded_stimulation[f"eem_drive:{population}"] < 1e-3
        assert v2_evidence.encoded_stimulation[f"eem_drive:{population}"] > 0.4


def test_reset_clears_the_loom_running_means() -> None:
    loom, _ = _loom_controllers()
    for step, distance in enumerate([6.0, 3.0, 1.0]):
        loom.step(_observation(loom, step, _sensors(forward=distance)), command_seq=step)
    adapted = loom.last_evidence
    assert adapted is not None
    assert 0.0 < adapted.encoded_stimulation["eem:threat_front"] < 1.0
    loom.reset()
    assert loom._eem_contrast_means is None
    loom.step(_observation(loom, 3, _sensors(forward=BLOCKED_RAY_M)), command_seq=3)
    fresh = loom.last_evidence
    assert fresh is not None
    assert fresh.encoded_stimulation["eem:threat_front"] == 0.0  # first tick again


def test_loom_carries_its_own_stimulation_provenance_schema() -> None:
    assert FULL_BRAIN_EEM_LOOM_V1_ENCODED_STIMULATION_SCHEMA == (
        "neurocraft.full_brain.encoded_stimulation.eem.loom.v1"
    )
    assert (
        len(
            {
                FULL_BRAIN_EEM_LOOM_V1_ENCODED_STIMULATION_SCHEMA,
                FULL_BRAIN_EEM_ESCAPE_V2_ENCODED_STIMULATION_SCHEMA,
                FULL_BRAIN_EEM_V2_ENCODED_STIMULATION_SCHEMA,
            }
        )
        == 3
    )
    loom, escape_v2 = _loom_controllers()
    sensors = _sensors(forward=BLOCKED_RAY_M, horizontal_collision=True)
    loom.step(_observation(loom, 0, sensors), command_seq=0)
    escape_v2.step(_observation(escape_v2, 0, sensors), command_seq=0)
    loom_evidence = loom.last_evidence
    v2_evidence = escape_v2.last_evidence
    assert loom_evidence is not None and v2_evidence is not None
    assert "eem_mean:threat_front" in loom_evidence.encoded_stimulation
    assert "eem_mean:threat_front" not in v2_evidence.encoded_stimulation
    assert loom_evidence.encoded_stimulation_sha256 != v2_evidence.encoded_stimulation_sha256


def test_loom_decoder_is_the_escape_v2_document_with_a_rotated_encoder() -> None:
    """The decoder is deliberately unchanged; its identity still separates."""

    loom, escape_v2 = _loom_controllers()
    binding = loom.escape_binding
    v2_binding = escape_v2.escape_binding
    assert isinstance(binding, EscapeMotorBindingV2)
    assert isinstance(v2_binding, EscapeMotorBindingV2)
    material = binding.to_identity_material()
    assert material["schema"] == FULL_BRAIN_ESCAPE_V2_DECODER_SCHEMA
    assert material["jump_formula"] == v2_binding.to_identity_material()["jump_formula"]
    assert material["wing"] == v2_binding.to_identity_material()["wing"]
    assert material["ttmn_pool_binding_sha256"] == v2_binding.ttmn_pool_identity_sha256
    # The chained encoder identity is what rotates the decoder identity. (The
    # synthetic frame's population binding is not the pinned real one, so this
    # compares against the bound document rather than LOOM_V1_ENCODER_SHA.)
    senses = loom.embodied_senses
    assert senses is not None
    assert material["encoder_document_sha256"] == senses.encoder.identity_sha256
    assert isinstance(senses.encoder, EemEncoderDocumentLoomV1)
    assert binding.identity_sha256 != v2_binding.identity_sha256
    assert loom.identity.adapter_hash != escape_v2.identity.adapter_hash
    assert loom.identity.decoder_hash != escape_v2.identity.decoder_hash
    assert loom.identity.dynamics_hash == escape_v2.identity.dynamics_hash
    assert loom.identity.graph_hash == escape_v2.identity.graph_hash
    assert loom.identity.observation_schema_hash == escape_v2.identity.observation_schema_hash
    assert loom.configuration.identity_sha256 != escape_v2.configuration.identity_sha256


def test_jump_term_is_still_the_rectified_escape_v2_mapping() -> None:
    loom, _ = _loom_controllers()
    escape = loom.escape_binding
    assert escape is not None
    activity = np.zeros(loom.engine.view.node_count, dtype=np.float64)
    for member in escape.jump_pool.members:
        activity[member.dense_index] = -0.5
    assert loom._decode_jump(activity) == 0.0  # rectified: no anti-jump
    for member in escape.jump_pool.members:
        activity[member.dense_index] = 0.01
    assert loom._decode_jump(activity) == pytest.approx(0.08, abs=0.0)


def test_two_loom_controllers_replay_identically() -> None:
    first, _ = _loom_controllers()
    second, _ = _loom_controllers()
    assert first.identity == second.identity
    for step in range(5):
        sensors = _sensors(forward=6.0 - step, horizontal_collision=step >= 2)
        action_a = first.step(_observation(first, step, sensors), command_seq=step)
        action_b = second.step(_observation(second, step, sensors), command_seq=step)
        assert action_a.state_digest == action_b.state_digest
        evidence_a = first.last_evidence
        evidence_b = second.last_evidence
        assert evidence_a is not None and evidence_b is not None
        assert evidence_a.action_sha256 == evidence_b.action_sha256
        assert evidence_a.jump_value == evidence_b.jump_value


def test_loom_evidence_subset_still_appends_exactly_the_ttmn_members() -> None:
    loom, escape_v2 = _loom_controllers()
    assert loom._evidence_body_ids == escape_v2._evidence_body_ids
    assert loom._evidence_body_ids[-2:] == (TTMN_RIGHT_BODY, TTMN_LEFT_BODY)


# -------------------------------------------------------------- configuration gates


def test_loom_configuration_has_its_own_schema_and_refusals() -> None:
    config = FullBrainLoomV1Configuration(encoder_identity_sha256=LOOM_V1_ENCODER_SHA)
    assert config.schema == FULL_BRAIN_LOOM_V1_CONFIG_SCHEMA
    assert config.schema == "neurocraft.controller.full_brain_loom_config.v1"
    material = config.to_identity_material()
    assert material["mode_taxonomy"] == "engineered_encoder"
    assert material["jump_scale"] == (1.0).hex()
    assert material["encoder_identity_sha256"] == LOOM_V1_ENCODER_SHA
    with pytest.raises(ValueError, match="unsupported approach-gated loom"):
        FullBrainLoomV1Configuration(
            schema=FULL_BRAIN_ESCAPE_V2_CONFIG_SCHEMA,
            encoder_identity_sha256=LOOM_V1_ENCODER_SHA,
        )
    with pytest.raises(ValueError, match="encoder document identity"):
        FullBrainLoomV1Configuration()
    with pytest.raises(ValueError, match="mode_taxonomy"):
        FullBrainLoomV1Configuration(
            encoder_identity_sha256=LOOM_V1_ENCODER_SHA, mode_taxonomy="faithful"
        )
    with pytest.raises(ValueError, match="requires the SHA-256"):
        FullBrainLoomV1Configuration(encoder_identity_sha256=LOOM_V1_ENCODER_SHA, jump_scale=500.0)
    with pytest.raises(ValueError, match="must carry the measured"):
        FullBrainLoomV1Configuration(
            encoder_identity_sha256=LOOM_V1_ENCODER_SHA,
            readout_scale=76.0,
            readout_calibration_sha256="6f" * 32,
            jump_scale=1.0,
        )
    with pytest.raises(ValueError, match="jump_scale must be finite"):
        FullBrainLoomV1Configuration(
            encoder_identity_sha256=LOOM_V1_ENCODER_SHA,
            readout_scale=76.0,
            readout_calibration_sha256="6f" * 32,
            jump_scale=float("inf"),
        )
    with pytest.raises(ValueError, match="five neural steps"):
        FullBrainLoomV1Configuration(
            encoder_identity_sha256=LOOM_V1_ENCODER_SHA, neural_steps_per_game_tick=4
        )


# ------------------------------------------------ configuration/document pairings


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


def test_controller_refuses_every_loom_encoder_mispairing() -> None:
    """Both directions, against the v1, eem-v2, and escape-v2 documents."""

    loom, escape_v2 = _loom_controllers()
    senses_loom = loom.embodied_senses
    senses_escape_v2 = escape_v2.embodied_senses
    assert senses_loom is not None and senses_escape_v2 is not None
    senses_v1 = bind_embodied_senses(senses_loom.populations)
    senses_eem_v2 = bind_embodied_senses_v2(senses_loom.populations)
    binding_loom = loom.escape_binding
    binding_v2 = escape_v2.escape_binding
    assert binding_loom is not None and binding_v2 is not None

    # An escape-v2 configuration over the loom encoder document.
    with pytest.raises(FullBrainProfileError, match=r"loom-v1 .*must be paired"):
        _rebuild(
            loom,
            configuration=FullBrainEscapeV2Configuration(
                encoder_identity_sha256=senses_loom.encoder.identity_sha256
            ),
            embodied=senses_loom,
            escape=binding_loom,
        )
    # A loom configuration over the escape-v2 transient-touch document.
    with pytest.raises(FullBrainProfileError, match=r"loom-v1 .*must be paired"):
        _rebuild(
            loom,
            configuration=FullBrainLoomV1Configuration(
                encoder_identity_sha256=senses_escape_v2.encoder.identity_sha256
            ),
            embodied=senses_escape_v2,
            escape=binding_v2,
        )
    # A loom configuration over the v1 absolute document.
    with pytest.raises(FullBrainProfileError, match=r"loom-v1 .*must be paired"):
        _rebuild(
            loom,
            configuration=FullBrainLoomV1Configuration(
                encoder_identity_sha256=senses_v1.encoder.identity_sha256
            ),
            embodied=senses_v1,
            escape=binding_v2,
        )
    # A loom configuration over the eem-v2 all-channel-contrast document.
    with pytest.raises(FullBrainProfileError, match=r"loom-v1 .*must be paired"):
        _rebuild(
            loom,
            configuration=FullBrainLoomV1Configuration(
                encoder_identity_sha256=senses_eem_v2.encoder.identity_sha256
            ),
            embodied=senses_eem_v2,
            escape=binding_v2,
        )
    # An eem-v1 / eem-v2 (non-escape) configuration over the loom document.
    for configuration in (
        FullBrainEemConfiguration(encoder_identity_sha256=senses_loom.encoder.identity_sha256),
        FullBrainEemV2Configuration(encoder_identity_sha256=senses_loom.encoder.identity_sha256),
    ):
        with pytest.raises(FullBrainProfileError, match=r"loom-v1 .*must be paired"):
            _rebuild(loom, configuration=configuration, embodied=senses_loom, escape=None)
    # An escape-v1 configuration over the loom document.
    with pytest.raises(FullBrainProfileError, match=r"loom-v1 .*must be paired"):
        _rebuild(
            loom,
            configuration=FullBrainEscapeConfiguration(
                encoder_identity_sha256=senses_loom.encoder.identity_sha256
            ),
            embodied=senses_loom,
            escape=binding_loom,
        )
    # And the neighbouring lanes still construct: nothing was widened away.
    _rebuild(
        loom,
        configuration=escape_v2.configuration,
        embodied=senses_escape_v2,
        escape=binding_v2,
    )
    _rebuild(
        loom,
        configuration=FullBrainEemV2Configuration(
            encoder_identity_sha256=senses_eem_v2.encoder.identity_sha256
        ),
        embodied=senses_eem_v2,
        escape=None,
    )
    _rebuild(
        loom,
        configuration=FullBrainEemConfiguration(
            encoder_identity_sha256=senses_v1.encoder.identity_sha256
        ),
        embodied=senses_v1,
        escape=None,
    )


def test_loom_configuration_refuses_the_unrectified_escape_v1_decoder() -> None:
    loom, _ = _loom_controllers()
    senses_loom = loom.embodied_senses
    binding = loom.escape_binding
    assert senses_loom is not None and binding is not None
    with pytest.raises(FullBrainProfileError, match="v1/v2 escape mix is refused"):
        _rebuild(
            loom,
            configuration=loom.configuration,
            embodied=senses_loom,
            escape=EscapeMotorBinding(
                wing=binding.wing,
                jump_pool=binding.jump_pool,
                jump_scale=binding.jump_scale,
                ttmn_pool_identity_sha256=binding.ttmn_pool_identity_sha256,
                encoder_identity_sha256=binding.encoder_identity_sha256,
            ),
        )


# ---------------------------------------------------- calibration cross-refusals


def _loom_calibration_document(
    *,
    encoder_sha256: str = LOOM_V1_ENCODER_SHA,
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
    document["schema"] = FULL_READOUT_CALIBRATION_LOOM_V1_SCHEMA
    document["stream"] = {"stream_id": "unit-test-loom-stream-v1"}
    return document


def test_loom_calibration_loads_and_binds_its_own_boundary() -> None:
    calibration = load_loom_v1_readout_calibration(
        canonical_json_bytes(_loom_calibration_document())
    )
    assert calibration.wing_scale == 76.0
    assert calibration.jump_scale == 500.0
    assert calibration.encoder_identity_sha256 == LOOM_V1_ENCODER_SHA
    calibration.assert_binds(
        artifact_manifest_sha256=GRAPH_SHA,
        sign_overlay_identity_sha256=OVERLAY_SHA,
        encoder_identity_sha256=LOOM_V1_ENCODER_SHA,
        ttmn_pool_identity_sha256="7a" * 32,
    )
    with pytest.raises(FullBrainProfileError, match="different encoder document"):
        calibration.assert_binds(
            artifact_manifest_sha256=GRAPH_SHA,
            sign_overlay_identity_sha256=OVERLAY_SHA,
            encoder_identity_sha256=ESCAPE_V2_ENCODER_SHA,
            ttmn_pool_identity_sha256="7a" * 32,
        )
    with pytest.raises(FullBrainProfileError, match="different TTMn pool binding"):
        calibration.assert_binds(
            artifact_manifest_sha256=GRAPH_SHA,
            sign_overlay_identity_sha256=OVERLAY_SHA,
            encoder_identity_sha256=LOOM_V1_ENCODER_SHA,
            ttmn_pool_identity_sha256="8b" * 32,
        )


def test_every_calibration_boundary_refuses_loom_v1_and_vice_versa() -> None:
    """All directed pairs against the loom-v1 calibration schema."""

    loom_bytes = canonical_json_bytes(_loom_calibration_document())
    # The loom loader refuses every other boundary by name.
    for label, document in (
        ("wing", _calibration_document(encoder_sha256=None)),
        ("eem_v1", _calibration_document(encoder_sha256=EEM_ENCODER_SHA)),
        ("eem_v2", _calibration_document(encoder_sha256=EEM_V2_ENCODER_SHA)),
    ):
        assert label
        with pytest.raises(FullBrainProfileError, match="loom-v1 boundary refuses"):
            load_loom_v1_readout_calibration(canonical_json_bytes(document))
    with pytest.raises(FullBrainProfileError, match="escape-v1 calibration"):
        load_loom_v1_readout_calibration(canonical_json_bytes(_escape_calibration_document()))
    with pytest.raises(FullBrainProfileError, match="escape-v2 calibration"):
        load_loom_v1_readout_calibration(canonical_json_bytes(_escape_v2_calibration_document()))
    # And every other boundary refuses the loom-v1 document.
    with pytest.raises(FullBrainProfileError, match="loom-v1 calibration"):
        load_escape_readout_calibration(loom_bytes)
    with pytest.raises(FullBrainProfileError, match="loom-v1 calibration"):
        load_escape_v2_readout_calibration(loom_bytes)
    with pytest.raises(FullBrainProfileError, match="unsupported readout calibration schema"):
        load_readout_calibration(loom_bytes)
    # An unknown schema at the loom loader is refused too.
    stray = _loom_calibration_document()
    stray["schema"] = "neurocraft.full_readout_calibration.made.up"
    with pytest.raises(
        FullBrainProfileError, match="unsupported loom-v1 readout calibration schema"
    ):
        load_loom_v1_readout_calibration(canonical_json_bytes(stray))


def test_loom_calibration_scale_validation_names_its_own_boundary() -> None:
    identity_scale = _loom_calibration_document(jump_scale=1.0)
    with pytest.raises(FullBrainProfileError, match=r"loom-v1 calibration.*jump_scale 1\.0"):
        load_loom_v1_readout_calibration(canonical_json_bytes(identity_scale))
    disagreeing = _loom_calibration_document()
    disagreeing["calibration"]["wing_scale"] = 77.0
    with pytest.raises(FullBrainProfileError, match="disagrees with its float hex"):
        load_loom_v1_readout_calibration(canonical_json_bytes(disagreeing))


# ------------------------------------------------ from_artifact argument gates


def test_from_artifact_refuses_loom_encoder_misuse(tmp_path: Path) -> None:
    """Every refusal fires before any artifact byte is read; all presence
    gates are faked with tmp paths so the intended gate is the one reached."""

    def build(**overrides: Any) -> None:
        FullBrainController.from_artifact(
            run_id="refusal",
            entity_id="fly",
            artifact_directory=tmp_path / "absent",
            annotations_path=tmp_path / "absent.feather",
            embodied_senses=True,
            body_annotations_path=tmp_path / "absent-annotations.feather",
            loom_encoder=True,
            **overrides,
        )

    # loom_encoder without the escape reflex at all.
    with pytest.raises(FullBrainProfileError, match="requires escape_reflex=True"):
        build()
    # loom_encoder on the unrectified escape-v1 decoder.
    with pytest.raises(FullBrainProfileError, match="with escape_version=2"):
        build(escape_reflex=True, escape_version=1)
    # With both satisfied, the next gate is the local-only presence gate, and
    # it names the loom profile.
    with pytest.raises(FullBrainProfileError, match="full_mcns_loom_v1 is a local-only"):
        build(escape_reflex=True, escape_version=2)


# ------------------------------------------------------- calibration script wiring


def test_calibration_script_registers_the_loom_v1_profile() -> None:
    import scripts.calibrate_full_readout as calibrate

    assert calibrate.LOOM_V1_PROFILE == "loom-v1"
    assert calibrate.LOOM_V1_STREAM_ID == "full-readout-calibration-loom-v1"
    assert calibrate.DEFAULT_LOOM_V1_OUTPUT.name == "full-readout-calibration-loom-v1.json"
    # The escape tuple is byte-unchanged; the loom profile is its own row.
    assert calibrate.ESCAPE_PROFILES == (calibrate.ESCAPE_PROFILE, calibrate.ESCAPE_V2_PROFILE)
    assert calibrate.JUMP_READOUT_PROFILES == (
        calibrate.ESCAPE_PROFILE,
        calibrate.ESCAPE_V2_PROFILE,
        calibrate.LOOM_V1_PROFILE,
        calibrate.LOOM_V2_PROFILE,
    )
    outputs = {
        calibrate.DEFAULT_OUTPUT,
        calibrate.DEFAULT_EEM_OUTPUT,
        calibrate.DEFAULT_EEM_V2_OUTPUT,
        calibrate.DEFAULT_ESCAPE_OUTPUT,
        calibrate.DEFAULT_ESCAPE_V2_OUTPUT,
        calibrate.DEFAULT_LOOM_V1_OUTPUT,
    }
    assert len(outputs) == 6
    with pytest.raises(calibrate.ReadoutCalibrationError, match="unknown escape calibration"):
        calibrate.build_escape_document(None, {}, profile="wing")  # type: ignore[arg-type]


# ---------------------------------------------------------------- service wiring


def test_service_and_launcher_register_the_loom_profile() -> None:
    import scripts.run_connectome_service as service
    import scripts.run_interactive_demo as launcher

    assert service.FULL_MCNS_LOOM_V1_PROFILE == "full_mcns_loom_v1"
    assert service.FULL_MCNS_LOOM_V1_PROFILE in service.CONTROLLER_PROFILES
    assert service.FULL_MCNS_LOOM_V1_PROFILE in service.FULL_MCNS_PROFILES
    assert service.FULL_MCNS_LOOM_V1_PROFILE in service.FULL_MCNS_BODY_ANNOTATION_PROFILES
    assert service.FULL_MCNS_LOOM_V1_PROFILE in service.FULL_MCNS_JUMP_READOUT_PROFILES
    # The escape tuples stay exactly the two escape generations.
    assert service.FULL_MCNS_ESCAPE_PROFILES == (
        service.FULL_MCNS_ESCAPE_PROFILE,
        service.FULL_MCNS_ESCAPE_V2_PROFILE,
    )
    assert service.FULL_MCNS_LOOM_V1_PROFILE not in service.FULL_MCNS_ESCAPE_PROFILES
    assert service.FULL_MCNS_LOOM_V1_PROFILE not in service.FULL_MCNS_EEM_PROFILES
    path, hint = service._full_profile_calibration(service.FULL_MCNS_LOOM_V1_PROFILE)
    assert path.name == "full-readout-calibration-loom-v1.json"
    assert hint == "scripts/calibrate_full_readout.py --profile loom-v1"
    # The launcher's profile names are the service's, exactly.
    assert set(launcher.CONTROLLER_PROFILES) == set(service.CONTROLLER_PROFILES)


# ------------------------------------------------------- committed calibration


@pytest.mark.skipif(
    not LOOM_V1_CALIBRATION_PATH.is_file(),
    reason="the committed loom-v1 calibration is absent",
)
def test_committed_loom_calibration_binds_the_declared_identities() -> None:
    """Document-only: no artifact is loaded, only the committed bytes read.

    Deliberately NOT gated on the local-only artifact: the calibration
    document is tracked, so these pins (its digest, both scale hexes, and the
    identities it says it was measured under) run on CI, where the artifact is
    absent. Nothing else in this suite needs the artifact either -- it is
    synthetic frames and synthetic views throughout.
    """

    import json
    from hashlib import sha256

    raw = LOOM_V1_CALIBRATION_PATH.read_bytes()
    assert sha256(raw).hexdigest() == LOOM_V1_CALIBRATION_SHA
    calibration = load_loom_v1_readout_calibration(raw)
    assert calibration.document_sha256 == LOOM_V1_CALIBRATION_SHA
    assert calibration.encoder_identity_sha256 == CURRENT_LOOM_V1_ENCODER_SHA
    assert calibration.stream_id == "full-readout-calibration-loom-v1"
    assert float(calibration.wing_scale).hex() == LOOM_V1_WING_SCALE_HEX
    assert float(calibration.jump_scale).hex() == LOOM_V1_JUMP_SCALE_HEX
    document = json.loads(raw)
    assert document["schema"] == FULL_READOUT_CALIBRATION_LOOM_V1_SCHEMA
    assert document["profile"] == "full_mcns_loom_v1"
    assert document["encoder"]["identity_sha256"] == CURRENT_LOOM_V1_ENCODER_SHA
    assert document["encoder"]["population_binding_sha256"] == CURRENT_EEM_POPULATION_BINDING_SHA
    measured_under = document["measured_under"]
    assert measured_under["configuration_identity_sha256"] == LOOM_V1_MEASURED_UNDER_CONFIG_SHA
    assert measured_under["dynamics_hash"] == LOOM_V1_MEASURED_UNDER_DYNAMICS_SHA
    assert measured_under["readout_scale_float_hex"] == (1.0).hex()
    assert measured_under["jump_scale_float_hex"] == (1.0).hex()
    assert document["coverage"]["run_is_covered"] is True
    # Every other boundary's loader refuses the committed document.
    with pytest.raises(FullBrainProfileError, match="loom-v1 calibration"):
        load_escape_v2_readout_calibration(raw)
    with pytest.raises(FullBrainProfileError, match="loom-v1 calibration"):
        load_escape_readout_calibration(raw)
