#!/usr/bin/env python3
"""Measure the full-brain wing readout once and derive its declared scale.

Stage 4 of the full-brain-live lane. The a-priori pool gain 8.0 was declared
for the *fixture-scale* boundary; at full scale the same 52-member wing pools
read a mean over 166,700 row-normalized neurons whose inputs are overwhelmingly
quiet, so the pre-gain pool means sit near 1e-5 and the actuation derived from
them is invisible against the +-1 protocol range. This script measures what
those pool means actually are under one named deterministic observation stream
and derives **one** scalar from the measurement:

    readout_scale = 0.5 / (p95 * motor_pool_gain)

so that a p95-magnitude pool value lands at 0.5 actuation, leaving headroom to
the +-1 clip. The number is written once into a committed calibration document
whose SHA-256 the controller binds into both its configuration identity and its
decoder identity.

What this is NOT
----------------
* Not tuning. The scale is measured once under a declared stream and frozen;
  nothing is searched, fitted, or re-run against an outcome.
* Not shaping. One factor, both pools, applied after the declared gain and
  before the pinned clip and dead zone. There is no baseline subtraction and no
  per-axis term, so a persistent readout bias stays visible in the calibrated
  output instead of being removed.
* Not a behavior claim. Rescaling the readout changes the units of what the
  brain already produced. Whether the calibrated profile behaves better is an
  observation that has not been made here.

The observation stream is SHA-256-derived and carries no PRNG, so the neural
trajectory -- and therefore the measured distribution and the derived scale --
reproduce run to run. The document contains no timestamps and no environment
data: a rerun must reproduce it byte-identically, and this script refuses to
overwrite a differing one (the write-or-verify discipline
``scripts/measure_full_sign_overlay.py`` uses).
"""

from __future__ import annotations

import argparse
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from neurocraft_controller.escape_reflex import ESCAPE_JUMP_FORMULA, ESCAPE_JUMP_FORMULA_V2
from neurocraft_controller.full_brain_controller import (
    FULL_MCNS_ARTIFACT_MANIFEST_SHA256,
    FULL_READOUT_CALIBRATION_ESCAPE_SCHEMA,
    FULL_READOUT_CALIBRATION_ESCAPE_V2_SCHEMA,
    FULL_READOUT_CALIBRATION_LOOM_V1_SCHEMA,
    FULL_READOUT_CALIBRATION_LOOM_V2_SCHEMA,
    FULL_READOUT_CALIBRATION_SCHEMA,
    MAX_READOUT_SCALE,
    FullBrainConfiguration,
    FullBrainController,
)
from neurocraft_protocol import Observation, SensorFrame, SensorFrameV2, canonical_json_bytes
from neurocraft_protocol.sensor_contract import OBSERVATION_SCHEMA_V2_SHA256

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIRECTORY = REPO_ROOT / "data" / "processed" / "male-cns-v1.0-superclass-v1"
DEFAULT_ANNOTATIONS = (
    REPO_ROOT / "data" / "cache" / "male-cns-v1.0"
) / "body-neurotransmitters-male-cns-v1.0.feather"
DEFAULT_BODY_ANNOTATIONS = (
    REPO_ROOT / "data" / "cache" / "male-cns-v1.0"
) / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "calibration" / "full-readout-calibration-v1.json"
DEFAULT_EEM_OUTPUT = REPO_ROOT / "data" / "calibration" / "full-readout-calibration-eem-v1.json"
DEFAULT_EEM_V2_OUTPUT = REPO_ROOT / "data" / "calibration" / "full-readout-calibration-eem-v2.json"
DEFAULT_ESCAPE_OUTPUT = (
    REPO_ROOT / "data" / "calibration" / "full-readout-calibration-escape-v1.json"
)
DEFAULT_ESCAPE_V2_OUTPUT = (
    REPO_ROOT / "data" / "calibration" / "full-readout-calibration-escape-v2.json"
)
DEFAULT_LOOM_V1_OUTPUT = (
    REPO_ROOT / "data" / "calibration" / "full-readout-calibration-loom-v1.json"
)

#: The declared stream identity. It is both the document's ``stream_id`` and
#: the SHA-256 seed prefix, so the name and the executed bytes cannot drift.
STREAM_ID = "full-readout-calibration-v1"
#: The Engineered-Encoder profile's own stream: same ray/light byte layout as
#: the wing stream under a distinct seed prefix, plus SHA-256-derived
#: collision flags, because the EEM boundary is driven by rays AND light AND
#: collisions and must be calibrated under all three.
EEM_STREAM_ID = "full-readout-calibration-eem-v1"
#: The temporal-contrast (eem v2) profile's stream: byte layout and declared
#: distributions IDENTICAL to the eem v1 stream (rays, uniform light,
#: Bernoulli collisions) under its own seed prefix, so the contrast dynamics
#: are calibrated against the same realistic variation rather than a new
#: distribution choice.
EEM_V2_STREAM_ID = "full-readout-calibration-eem-v2"
#: The escape profile's stream: byte layout and declared distributions
#: IDENTICAL to the eem v1 stream (rays, uniform light, Bernoulli
#: collisions) under its own seed prefix -- the escape sensory boundary IS
#: the eem v1 encoder, so its calibration sees the same realistic variation.
#: One run of this stream measures BOTH pool groups (wing and TTMn jump) so
#: one document binds the whole boundary.
ESCAPE_STREAM_ID = "full-readout-calibration-escape-v1"
#: The escape-v2 profile's stream (identity recorded in
#: experiments/preregistration/eem-escape-v2.md): byte layout and declared
#: distributions IDENTICAL to the escape-v1 stream under its own seed prefix.
#: The stream drives RAW channel values; the declared touch temporal contrast
#: transforms them inside the encoder exactly as it does live, so the measured
#: distributions are the escape-v2 boundary's own.
ESCAPE_V2_STREAM_ID = "full-readout-calibration-escape-v2"
#: The loom-v1 profile's stream (declared in ``docs/encoders/eem-loom-v1.md``):
#: byte layout and declared distributions IDENTICAL to the escape-v2 stream
#: under its own seed prefix. The stream drives RAW channel values; the
#: declared rectified approach contrast transforms them inside the encoder
#: exactly as it does live, so the measured distributions are the loom-v1
#: boundary's own. The tick-to-tick ray variation this stream already has is
#: what a rise-sensitive encoder needs to see; no distribution was changed for
#: it.
LOOM_V1_STREAM_ID = "full-readout-calibration-loom-v1"
WARMUP_TICKS = 50
MEASURED_TICKS = 600
RAY_MAX_DISTANCE_M = 6.0
GAME_TICK_NS = 50_000_000
PERCENTILE_METHOD = "nearest_rank_rounded_index"

WING_PROFILE = "wing"
EEM_PROFILE = "eem"
EEM_V2_PROFILE = "eem-v2"
ESCAPE_PROFILE = "escape"
ESCAPE_V2_PROFILE = "escape-v2"
LOOM_V1_PROFILE = "loom-v1"
LOOM_V2_PROFILE = "loom-v2"
DEFAULT_LOOM_V2_OUTPUT = (
    REPO_ROOT / "data" / "calibration" / "full-readout-calibration-loom-v2.json"
)
EEM_PROFILES = (EEM_PROFILE, EEM_V2_PROFILE)
ESCAPE_PROFILES = (ESCAPE_PROFILE, ESCAPE_V2_PROFILE)
#: Every profile whose calibration document carries the two named scales
#: {wing_scale, jump_scale} and the ``jump_readout`` section: the two escape
#: generations plus loom-v1, which reuses the escape-v2 decoder unchanged.
JUMP_READOUT_PROFILES = (*ESCAPE_PROFILES, LOOM_V1_PROFILE, LOOM_V2_PROFILE)

#: Declared collision distribution for the EEM stream: each flag reads two
#: digest bytes as a big-endian integer and fires below this threshold, so the
#: exact probability is 6554/65536 ~= 0.1000061 -- the closest two-byte
#: approximation of the declared Bernoulli p = 0.1, disclosed as such.
COLLISION_THRESHOLD = 6554
COLLISION_DENOMINATOR = 65536
HORIZONTAL_COLLISION_BYTES = slice(13, 15)
VERTICAL_COLLISION_BYTES = slice(15, 17)

#: Target actuation magnitude for the stream's p95 pool value. Declared here,
#: once, with the headroom rationale: 0.5 of the +-1 range leaves the top half
#: of the range for excursions above p95 before the pinned clip engages.
TARGET_P95_ACTUATION = 0.5


class ReadoutCalibrationError(RuntimeError):
    """The calibration could not be measured or written exactly as declared."""


def synthetic_observation(
    controller: FullBrainController,
    tick: int,
    *,
    stream_id: str = STREAM_ID,
    collisions: bool = False,
) -> Observation:
    """One deterministic SHA-256-derived observation with varying rays.

    Byte layout is deliberately identical to
    ``scripts/benchmark_full_controller.py``: each ray distance reads two bytes
    of ``sha256(f"{stream_id}:{tick}")`` and the light level reads byte 12
    (uniform integers 0..15, since 256 divides evenly). Only the seed prefix
    differs, so each stream is a distinct, named stream rather than an
    unlabeled reuse of the benchmark's. With ``collisions=True`` (the EEM
    stream) the two collision flags additionally read digest bytes 13:15 and
    15:17 as declared Bernoulli(6554/65536) draws.
    """

    digest = sha256(f"{stream_id}:{tick}".encode()).digest()

    def ray(slot: int) -> float:
        raw = int.from_bytes(digest[2 * slot : 2 * slot + 2], "big")
        return RAY_MAX_DISTANCE_M * raw / 65535.0

    def collision(byte_range: slice) -> bool:
        return collisions and (int.from_bytes(digest[byte_range], "big") < COLLISION_THRESHOLD)

    sensors = SensorFrame(
        position_m=(0.0, 66.0, 0.0),
        velocity_m_s=(0.0, 0.0, 0.0),
        yaw_rad=0.0,
        pitch_rad=0.0,
        on_ground=False,
        horizontal_collision=collision(HORIZONTAL_COLLISION_BYTES),
        vertical_collision=collision(VERTICAL_COLLISION_BYTES),
        in_fluid=False,
        local_light=digest[12] % 16,
        ray_max_distance_m=RAY_MAX_DISTANCE_M,
        ray_distances_m=(ray(0), ray(1), ray(2), ray(3), ray(4), ray(5)),
    )
    if controller.identity.observation_schema_hash == OBSERVATION_SCHEMA_V2_SHA256:
        sensors = SensorFrameV2.from_sensor_frame(sensors, (2, 2, 2, 2, 2, 2))
    return Observation(
        identity=controller.identity,
        observation_seq=tick,
        game_tick=tick,
        sent_monotonic_ns=tick * GAME_TICK_NS,
        sensors=sensors,
    )


def _percentile(ordered: list[float], fraction: float) -> float:
    """Nearest-rank percentile over an already-sorted sample, rounded index."""

    if not ordered:
        raise ReadoutCalibrationError("cannot take a percentile of an empty sample")
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _float_pair(value: float) -> dict[str, Any]:
    """A measured float as both its exact hex and its decimal form."""

    return {"float_hex": float(value).hex(), "value": float(value)}


def measure(
    controller: FullBrainController,
    *,
    stream_id: str = STREAM_ID,
    collisions: bool = False,
) -> dict[str, Any]:
    """Drive the declared stream and return the pooled magnitude distribution."""

    def observation(tick: int) -> Observation:
        return synthetic_observation(controller, tick, stream_id=stream_id, collisions=collisions)

    tick = 0
    for _ in range(WARMUP_TICKS):
        controller.step(observation(tick), command_seq=tick)
        tick += 1

    magnitudes: list[float] = []
    pool_names: tuple[str, ...] = ()
    for _ in range(MEASURED_TICKS):
        controller.step(observation(tick), command_seq=tick)
        means = controller.pool_readout_means()
        if pool_names and tuple(name for name, _ in means) != pool_names:
            raise ReadoutCalibrationError("pool ordering changed mid-measurement")
        pool_names = tuple(name for name, _ in means)
        for _name, value in means:
            if not np.isfinite(value):
                raise ReadoutCalibrationError("measured a non-finite pool mean")
            magnitudes.append(abs(float(value)))
        tick += 1

    # Declared run_end trigger: without it the engine exits mid-coverage-pass
    # and the measurement must not be reported as covered.
    controller.engine.end_run()

    ordered = sorted(magnitudes)
    return {
        "pool_names": list(pool_names),
        "ordered": ordered,
        "sample_sha256": sha256(
            b"".join(float(value).hex().encode() + b"\n" for value in magnitudes)
        ).hexdigest(),
    }


def build_document(
    controller: FullBrainController,
    measurement: dict[str, Any],
    *,
    profile: str = WING_PROFILE,
) -> dict[str, Any]:
    """Assemble the canonical, environment-free calibration document.

    The wing document is byte-for-byte the committed v1 layout; the EEM
    documents add the collision-derivation disclosure and an ``encoder``
    section binding the encoder-document and population-binding identities,
    so no profile's calibration can ever be loaded for another: the wing
    document carries no encoder section, and the eem v1 and eem v2 documents
    bind different encoder identities, so
    ``FullReadoutCalibration.assert_binds`` refuses every cross pairing.
    """

    ordered: list[float] = measurement["ordered"]
    gain = controller.configuration.motor_pool_gain
    p50 = _percentile(ordered, 0.50)
    p95 = _percentile(ordered, 0.95)
    p99 = _percentile(ordered, 0.99)
    largest = ordered[-1]
    if p95 <= 0.0 or not np.isfinite(p95):
        raise SystemExit(
            "refusing to derive a calibration: the measured p95 pool magnitude is "
            f"{p95!r}; a zero or non-finite p95 means the stream drove no readout"
        )
    scale = TARGET_P95_ACTUATION / (p95 * gain)
    if not np.isfinite(scale) or not 0.0 < scale <= MAX_READOUT_SCALE:
        raise SystemExit(
            f"refusing to derive a calibration: derived scale {scale!r} is not finite "
            f"and inside (0, {MAX_READOUT_SCALE}]"
        )
    saturating_scale = 1.0 / (largest * gain) if largest > 0.0 else float("inf")
    if not np.isfinite(saturating_scale):
        raise SystemExit(
            "refusing to derive a calibration: the measured maximum pool magnitude is "
            "zero, so the saturating scale is undefined"
        )
    engine = controller.engine
    if profile == WING_PROFILE:
        stream: dict[str, Any] = {
            "stream_id": STREAM_ID,
            "warmup_ticks": WARMUP_TICKS,
            "measured_ticks": MEASURED_TICKS,
            "neural_steps_per_game_tick": (controller.configuration.neural_steps_per_game_tick),
            "observation_derivation": (
                f"rays and light from sha256('{STREAM_ID}:<tick>') bytes; deterministic, no PRNG"
            ),
            "ray_max_distance_m": _float_pair(RAY_MAX_DISTANCE_M),
        }
        profile_name = "full_mcns_wing_v1"
    else:
        # eem and eem-v2 share the byte layout and declared distributions;
        # only the seed prefix (the stream id) and the profile name differ,
        # so the v1 document stays byte-identical to its committed form.
        eem_stream_id = EEM_STREAM_ID if profile == EEM_PROFILE else EEM_V2_STREAM_ID
        stream = {
            "stream_id": eem_stream_id,
            "warmup_ticks": WARMUP_TICKS,
            "measured_ticks": MEASURED_TICKS,
            "neural_steps_per_game_tick": (controller.configuration.neural_steps_per_game_tick),
            "observation_derivation": (
                f"rays (bytes 0:12) and light (byte 12, uniform 0..15) from "
                f"sha256('{eem_stream_id}:<tick>') bytes; horizontal/vertical "
                "collisions from bytes 13:15 / 15:17 as declared "
                f"Bernoulli({COLLISION_THRESHOLD}/{COLLISION_DENOMINATOR}); "
                "deterministic, no PRNG"
            ),
            "collision_derivation": {
                "threshold": COLLISION_THRESHOLD,
                "denominator": COLLISION_DENOMINATOR,
                "probability": _float_pair(COLLISION_THRESHOLD / COLLISION_DENOMINATOR),
                "note": (
                    "the closest two-byte approximation of the declared "
                    "Bernoulli p = 0.1; the exact executed probability is "
                    "6554/65536"
                ),
            },
            "ray_max_distance_m": _float_pair(RAY_MAX_DISTANCE_M),
        }
        profile_name = "full_mcns_eem_v1" if profile == EEM_PROFILE else "full_mcns_eem_v2"
    document = {
        "schema": FULL_READOUT_CALIBRATION_SCHEMA,
        "claim_scope": (
            "declared_boundary_unit_calibration_only_no_behavior_performance_or_biological_claim"
        ),
        "profile": profile_name,
        "stream": stream,
        "artifact": {
            "manifest_file_sha256": controller.identity.graph_hash,
            "node_count": engine.view.node_count,
            "edge_count": engine.view.edge_count,
            "runtime_view_sha256": engine.view.view_sha256,
        },
        "sign_overlay": {
            "policy": controller.configuration.sign_policy,
            "identity_sha256": controller.sign_overlay_identity_sha256,
        },
        "measured_under": {
            "note": (
                "the pre-calibration (readout_scale 1.0) controller identity the "
                "distribution below was measured with; the calibrated controller "
                "necessarily carries a different configuration and decoder identity"
            ),
            "configuration_identity_sha256": controller.configuration.identity_sha256,
            "readout_scale_float_hex": float(controller.readout_scale).hex(),
            "motor_pool_gain": _float_pair(gain),
            "motor_dead_zone": _float_pair(controller.configuration.motor_dead_zone),
            "adapter_hash": controller.identity.adapter_hash,
            "decoder_hash": controller.identity.decoder_hash,
            "dynamics_hash": controller.identity.dynamics_hash,
            "observation_schema_hash": controller.identity.observation_schema_hash,
        },
        "distribution": {
            "quantity": (
                "abs(pre-gain, pre-scale pool mean) = "
                "abs(fsum(weight * activity) / summed_abs_weight), pooled over both "
                "wing-motor pools and every measured tick"
            ),
            "pools": measurement["pool_names"],
            "count": len(ordered),
            "percentile_method": PERCENTILE_METHOD,
            "min": _float_pair(ordered[0]),
            "p50": _float_pair(p50),
            "p95": _float_pair(p95),
            "p99": _float_pair(p99),
            "max": _float_pair(largest),
            "sample_sha256": measurement["sample_sha256"],
        },
        "calibration": {
            "derivation": (f"readout_scale = {TARGET_P95_ACTUATION} / (p95 * motor_pool_gain)"),
            "target_p95_actuation": _float_pair(TARGET_P95_ACTUATION),
            "readout_scale_float_hex": float(scale).hex(),
            "readout_scale": float(scale),
            "disclosure": {
                "note": (
                    "the scale that would put the measured MAXIMUM pool magnitude "
                    "exactly at the +-1 clip; recorded so the headroom this "
                    "calibration leaves is auditable, not applied"
                ),
                "saturating_scale_float_hex": float(saturating_scale).hex(),
                "saturating_scale": float(saturating_scale),
            },
        },
        "coverage": {
            "final_trace_sha256": engine.trace_sha256,
            "verified_through_step": engine.verified_through_step,
            "uncovered_steps_at_exit": engine.uncovered_steps,
            "run_is_covered": engine.run_is_covered,
        },
    }
    if profile in EEM_PROFILES:
        embodied = controller.embodied_senses
        if embodied is None:
            raise ReadoutCalibrationError(
                "the eem profiles require a controller with the embodied senses binding"
            )
        document["encoder"] = {
            "identity_sha256": embodied.encoder.identity_sha256,
            "population_binding_sha256": embodied.populations.identity_sha256,
            "note": (
                "binds this calibration to the exact engineered-encoder document "
                "and population membership it was measured under; the wing "
                "calibration carries no encoder section and the two are mutually "
                "refused at load time"
            ),
        }
    return document


def measure_escape(controller: FullBrainController, *, stream_id: str) -> dict[str, Any]:
    """Drive the escape stream once, measuring BOTH pool groups per tick.

    One warmup + measurement pass (the standard 50 + 600 ticks) reads the
    wing pool means through :meth:`pool_readout_means` and the TTMn jump pool
    mean through :meth:`jump_readout_mean` on every measured tick, so the two
    scales in the escape calibration document come from a single run of a
    single named stream.
    """

    def observation(tick: int) -> Observation:
        return synthetic_observation(controller, tick, stream_id=stream_id, collisions=True)

    tick = 0
    for _ in range(WARMUP_TICKS):
        controller.step(observation(tick), command_seq=tick)
        tick += 1

    wing_magnitudes: list[float] = []
    jump_magnitudes: list[float] = []
    pool_names: tuple[str, ...] = ()
    for _ in range(MEASURED_TICKS):
        controller.step(observation(tick), command_seq=tick)
        means = controller.pool_readout_means()
        if pool_names and tuple(name for name, _ in means) != pool_names:
            raise ReadoutCalibrationError("pool ordering changed mid-measurement")
        pool_names = tuple(name for name, _ in means)
        for _name, value in means:
            if not np.isfinite(value):
                raise ReadoutCalibrationError("measured a non-finite pool mean")
            wing_magnitudes.append(abs(float(value)))
        jump_mean = controller.jump_readout_mean()
        if not np.isfinite(jump_mean):
            raise ReadoutCalibrationError("measured a non-finite jump pool mean")
        jump_magnitudes.append(abs(float(jump_mean)))
        tick += 1

    controller.engine.end_run()

    def group(magnitudes: list[float]) -> dict[str, Any]:
        return {
            "ordered": sorted(magnitudes),
            "sample_sha256": sha256(
                b"".join(float(value).hex().encode() + b"\n" for value in magnitudes)
            ).hexdigest(),
        }

    return {
        "pool_names": list(pool_names),
        "wing": group(wing_magnitudes),
        "jump": group(jump_magnitudes),
    }


def _escape_scale(ordered: list[float], *, gain: float, label: str) -> tuple[float, float]:
    """One group's (scale, saturating_scale) by the standard p95 derivation."""

    p95 = _percentile(ordered, 0.95)
    largest = ordered[-1]
    if p95 <= 0.0 or not np.isfinite(p95):
        raise SystemExit(
            f"refusing to derive a calibration: the measured {label} p95 pool magnitude "
            f"is {p95!r}; a zero or non-finite p95 means the stream drove no readout"
        )
    scale = TARGET_P95_ACTUATION / (p95 * gain)
    if not np.isfinite(scale) or not 0.0 < scale <= MAX_READOUT_SCALE:
        raise SystemExit(
            f"refusing to derive a calibration: derived {label} scale {scale!r} is not "
            f"finite and inside (0, {MAX_READOUT_SCALE}]"
        )
    saturating_scale = 1.0 / (largest * gain) if largest > 0.0 else float("inf")
    if not np.isfinite(saturating_scale):
        raise SystemExit(
            f"refusing to derive a calibration: the measured maximum {label} pool "
            "magnitude is zero, so the saturating scale is undefined"
        )
    return scale, saturating_scale


def _escape_distribution(
    ordered: list[float], *, pools: list[str], sample_sha256: str, quantity: str
) -> dict[str, Any]:
    return {
        "quantity": quantity,
        "pools": pools,
        "count": len(ordered),
        "percentile_method": PERCENTILE_METHOD,
        "min": _float_pair(ordered[0]),
        "p50": _float_pair(_percentile(ordered, 0.50)),
        "p95": _float_pair(_percentile(ordered, 0.95)),
        "p99": _float_pair(_percentile(ordered, 0.99)),
        "max": _float_pair(ordered[-1]),
        "sample_sha256": sample_sha256,
    }


def build_escape_document(
    controller: FullBrainController, measurement: dict[str, Any], *, profile: str = ESCAPE_PROFILE
) -> dict[str, Any]:
    """Assemble the canonical escape calibration document (two named scales).

    The layout extends the v1 document: ``distributions`` carries one group
    per pool family (wing / jump), ``calibration`` carries the named scales
    {wing_scale, jump_scale} each derived by the standard
    ``0.5 / (p95 * motor_pool_gain)``, and the ``encoder`` and
    ``jump_readout`` sections bind the exact encoder identity and the TTMn
    pool identity the boundary was measured under, so every other profile's
    loader refuses this document and this loader refuses theirs.

    ``profile`` selects the generation: ``escape`` writes the frozen escape-v1
    document, ``escape-v2`` writes the byte-identical layout under the
    escape-v2 schema, stream id, profile label, and rectified jump formula
    (identity recorded in ``experiments/preregistration/eem-escape-v2.md``),
    and ``loom-v1`` writes the same layout under the loom-v1 schema and
    stream id, keeping the escape-v2 rectified jump formula verbatim because
    the loom lane changes the sensory boundary only (declared in
    ``docs/encoders/eem-loom-v1.md``).
    """

    if profile not in JUMP_READOUT_PROFILES:
        raise ReadoutCalibrationError(f"unknown escape calibration profile {profile!r}")
    loom_v2 = profile == LOOM_V2_PROFILE
    loom_v1 = profile in (LOOM_V1_PROFILE, LOOM_V2_PROFILE)
    escape_v2 = profile == ESCAPE_V2_PROFILE
    # The loom lane executes the escape-v2 decoder verbatim, so it carries the
    # rectified formula; only the schema, stream id, label, and encoder note
    # distinguish its document.
    rectified = escape_v2 or loom_v1
    if loom_v1:
        schema = (
            FULL_READOUT_CALIBRATION_LOOM_V2_SCHEMA
            if loom_v2
            else FULL_READOUT_CALIBRATION_LOOM_V1_SCHEMA
        )
    elif escape_v2:
        schema = FULL_READOUT_CALIBRATION_ESCAPE_V2_SCHEMA
    else:
        schema = FULL_READOUT_CALIBRATION_ESCAPE_SCHEMA
    if loom_v1:
        profile_label = "full_mcns_loom_v2" if loom_v2 else "full_mcns_loom_v1"
        stream_id = LOOM_V1_STREAM_ID
    elif escape_v2:
        profile_label = "full_mcns_escape_v2"
        stream_id = ESCAPE_V2_STREAM_ID
    else:
        profile_label = "full_mcns_escape_v1"
        stream_id = ESCAPE_STREAM_ID
    jump_formula = ESCAPE_JUMP_FORMULA_V2 if rectified else ESCAPE_JUMP_FORMULA
    # The escape-v1 and escape-v2 notes are reproduced byte-for-byte: those
    # committed documents are immutable and write-or-verify re-derives them
    # exactly.
    if loom_v2:
        encoder_note = (
            "loom-v2 entity-gated encoder; same loom-v1 seed and numeric sensor stream, "
            "with all six ray_hit_kinds explicitly 2 (entity) on every tick. "
            "The loom-v2 loader refuses all other calibration boundaries."
        )
    elif loom_v1:
        encoder_note = (
            "binds this calibration to the exact engineered-encoder document "
            "(the loom-v1 approach-gated encoder) and population membership "
            "it was measured under; every other boundary's calibration is "
            "refused by the loom-v1 loader and vice versa"
        )
    elif escape_v2:
        encoder_note = (
            "binds this calibration to the exact engineered-encoder document "
            "(the escape-v2 transient-touch encoder) and population membership "
            "it was measured under; every other boundary's calibration is "
            "refused by the escape-v2 loader and vice versa"
        )
    else:
        encoder_note = (
            "binds this calibration to the exact engineered-encoder document "
            "(the eem v1 absolute encoder) and population membership it was "
            "measured under; every v1-schema calibration is refused by the "
            "escape loader and vice versa"
        )
    embodied = controller.embodied_senses
    escape = controller.escape_binding
    if embodied is None or escape is None:
        raise ReadoutCalibrationError(
            "the escape profile requires a controller with the escape binding"
        )
    gain = controller.configuration.motor_pool_gain
    wing_ordered: list[float] = measurement["wing"]["ordered"]
    jump_ordered: list[float] = measurement["jump"]["ordered"]
    wing_scale, wing_saturating = _escape_scale(wing_ordered, gain=gain, label="wing")
    jump_scale, jump_saturating = _escape_scale(jump_ordered, gain=gain, label="jump")
    engine = controller.engine
    return {
        "schema": schema,
        "claim_scope": (
            "declared_boundary_unit_calibration_only_no_behavior_performance_or_biological_claim"
        ),
        "profile": profile_label,
        "stream": {
            "stream_id": stream_id,
            "warmup_ticks": WARMUP_TICKS,
            "measured_ticks": MEASURED_TICKS,
            "neural_steps_per_game_tick": (controller.configuration.neural_steps_per_game_tick),
            "observation_derivation": (
                f"rays (bytes 0:12) and light (byte 12, uniform 0..15) from "
                f"sha256('{stream_id}:<tick>') bytes; horizontal/vertical "
                "collisions from bytes 13:15 / 15:17 as declared "
                f"Bernoulli({COLLISION_THRESHOLD}/{COLLISION_DENOMINATOR}); "
                "deterministic, no PRNG"
            ),
            "collision_derivation": {
                "threshold": COLLISION_THRESHOLD,
                "denominator": COLLISION_DENOMINATOR,
                "probability": _float_pair(COLLISION_THRESHOLD / COLLISION_DENOMINATOR),
                "note": (
                    "the closest two-byte approximation of the declared "
                    "Bernoulli p = 0.1; the exact executed probability is "
                    "6554/65536"
                ),
            },
            "ray_max_distance_m": _float_pair(RAY_MAX_DISTANCE_M),
        },
        "artifact": {
            "manifest_file_sha256": controller.identity.graph_hash,
            "node_count": engine.view.node_count,
            "edge_count": engine.view.edge_count,
            "runtime_view_sha256": engine.view.view_sha256,
        },
        "sign_overlay": {
            "policy": controller.configuration.sign_policy,
            "identity_sha256": controller.sign_overlay_identity_sha256,
        },
        "measured_under": {
            "note": (
                "the pre-calibration (readout_scale 1.0, jump_scale 1.0) controller "
                "identity the distributions below were measured with; the calibrated "
                "controller necessarily carries a different configuration and decoder "
                "identity"
            ),
            "configuration_identity_sha256": controller.configuration.identity_sha256,
            "readout_scale_float_hex": float(controller.readout_scale).hex(),
            "jump_scale_float_hex": float(escape.jump_scale).hex(),
            "motor_pool_gain": _float_pair(gain),
            "motor_dead_zone": _float_pair(controller.configuration.motor_dead_zone),
            "adapter_hash": controller.identity.adapter_hash,
            "decoder_hash": controller.identity.decoder_hash,
            "dynamics_hash": controller.identity.dynamics_hash,
            "observation_schema_hash": controller.identity.observation_schema_hash,
        },
        "distributions": {
            "wing": _escape_distribution(
                wing_ordered,
                pools=measurement["pool_names"],
                sample_sha256=measurement["wing"]["sample_sha256"],
                quantity=(
                    "abs(pre-gain, pre-scale pool mean) = "
                    "abs(fsum(weight * activity) / summed_abs_weight), pooled over both "
                    "wing-motor pools and every measured tick"
                ),
            ),
            "jump": _escape_distribution(
                jump_ordered,
                pools=[escape.jump_pool.name],
                sample_sha256=measurement["jump"]["sample_sha256"],
                quantity=(
                    "abs(pre-gain, pre-scale TTMn jump pool mean) = "
                    "abs(fsum(weight * activity) / summed_abs_weight) over the two "
                    "TTMn members, every measured tick"
                ),
            ),
        },
        "calibration": {
            "derivation": (
                f"wing_scale = {TARGET_P95_ACTUATION} / (wing p95 * motor_pool_gain); "
                f"jump_scale = {TARGET_P95_ACTUATION} / (jump p95 * motor_pool_gain)"
            ),
            "target_p95_actuation": _float_pair(TARGET_P95_ACTUATION),
            "wing_scale_float_hex": float(wing_scale).hex(),
            "wing_scale": float(wing_scale),
            "jump_scale_float_hex": float(jump_scale).hex(),
            "jump_scale": float(jump_scale),
            "disclosure": {
                "note": (
                    "the scales that would put each group's measured MAXIMUM pool "
                    "magnitude exactly at the +-1 clip; recorded so the headroom this "
                    "calibration leaves is auditable, not applied"
                ),
                "wing_saturating_scale_float_hex": float(wing_saturating).hex(),
                "wing_saturating_scale": float(wing_saturating),
                "jump_saturating_scale_float_hex": float(jump_saturating).hex(),
                "jump_saturating_scale": float(jump_saturating),
            },
        },
        "encoder": {
            "identity_sha256": embodied.encoder.identity_sha256,
            "population_binding_sha256": embodied.populations.identity_sha256,
            "note": encoder_note,
        },
        "jump_readout": {
            "ttmn_pool_binding_sha256": escape.ttmn_pool_identity_sha256,
            "formula": jump_formula,
            "note": (
                "binds this calibration to the exact TTMn pool membership and the "
                "declared jump mapping it was measured under"
            ),
        },
        "coverage": {
            "final_trace_sha256": engine.trace_sha256,
            "verified_through_step": engine.verified_through_step,
            "uncovered_steps_at_exit": engine.uncovered_steps,
            "run_is_covered": engine.run_is_covered,
        },
    }


def write_or_verify(output: Path, document: dict[str, Any]) -> tuple[str, str]:
    """Write the calibration once; afterwards only byte-identical output passes."""

    payload = canonical_json_bytes(document)
    digest = sha256(payload).hexdigest()
    if output.exists():
        if output.read_bytes() != payload:
            raise ReadoutCalibrationError(
                f"existing calibration at {output} differs from this measurement; "
                "a committed calibration is immutable -- investigate, or declare a "
                "new stream id, instead of overwriting"
            )
        return digest, "verified"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return digest, "wrote"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-directory", type=Path, default=DEFAULT_ARTIFACT_DIRECTORY)
    parser.add_argument(
        "--artifact-manifest-sha256",
        default=FULL_MCNS_ARTIFACT_MANIFEST_SHA256,
        help=(
            "external pin for the artifact manifest; defaults to the reviewed module "
            "constant for the current candidate (historical receipts retain their own pins)"
        ),
    )
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument(
        "--body-annotations",
        type=Path,
        default=DEFAULT_BODY_ANNOTATIONS,
        help="pinned body-annotations table for the eem profiles' population selectors",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="kernel thread count; default defers to NEUROCRAFT_SPMV_THREADS or the CPU cap",
    )
    parser.add_argument(
        "--profile",
        choices=(
            WING_PROFILE,
            EEM_PROFILE,
            EEM_V2_PROFILE,
            ESCAPE_PROFILE,
            ESCAPE_V2_PROFILE,
            LOOM_V1_PROFILE,
            LOOM_V2_PROFILE,
        ),
        default=WING_PROFILE,
        help=(
            "wing measures the committed full_mcns_wing_v1 stream byte-identically; "
            "eem measures the engineered-encoder boundary under its own stream "
            "(rays AND light AND collisions) and writes the eem calibration; "
            "eem-v2 measures the temporal-contrast boundary under the same "
            "distributions with its own seed prefix and writes the eem-v2 calibration; "
            "escape measures the escape-reflex boundary (eem v1 encoder plus the TTMn "
            "jump pool) under the same distributions with its own seed prefix, "
            "measuring BOTH pool groups in one run, and writes the escape calibration "
            "with named scales {wing_scale, jump_scale}; escape-v2 measures the "
            "escape-v2 boundary (transient-touch encoder plus the same TTMn pool) "
            "the same way under its own seed prefix and writes the escape-v2 "
            "calibration; loom-v1 measures the approach-gated boundary (the "
            "loom-v1 encoder over the same populations and the same escape-v2 "
            "rectified jump decoder) the same way under its own seed prefix "
            "and writes the loom-v1 calibration"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="calibration document path; defaults to the profile's committed location",
    )
    arguments = parser.parse_args(argv)
    eem = arguments.profile in EEM_PROFILES
    eem_v2 = arguments.profile == EEM_V2_PROFILE
    escape = arguments.profile in JUMP_READOUT_PROFILES
    loom_v1 = arguments.profile in (LOOM_V1_PROFILE, LOOM_V2_PROFILE)
    escape_version = (
        2 if arguments.profile in (ESCAPE_V2_PROFILE, LOOM_V1_PROFILE, LOOM_V2_PROFILE) else 1
    )
    default_output = {
        WING_PROFILE: DEFAULT_OUTPUT,
        EEM_PROFILE: DEFAULT_EEM_OUTPUT,
        EEM_V2_PROFILE: DEFAULT_EEM_V2_OUTPUT,
        ESCAPE_PROFILE: DEFAULT_ESCAPE_OUTPUT,
        ESCAPE_V2_PROFILE: DEFAULT_ESCAPE_V2_OUTPUT,
        LOOM_V1_PROFILE: DEFAULT_LOOM_V1_OUTPUT,
        LOOM_V2_PROFILE: DEFAULT_LOOM_V2_OUTPUT,
    }[arguments.profile]
    output: Path = arguments.output or default_output
    stream_id = {
        WING_PROFILE: STREAM_ID,
        EEM_PROFILE: EEM_STREAM_ID,
        EEM_V2_PROFILE: EEM_V2_STREAM_ID,
        ESCAPE_PROFILE: ESCAPE_STREAM_ID,
        ESCAPE_V2_PROFILE: ESCAPE_V2_STREAM_ID,
        LOOM_V1_PROFILE: LOOM_V1_STREAM_ID,
        LOOM_V2_PROFILE: LOOM_V1_STREAM_ID,
    }[arguments.profile]

    started = time.perf_counter()
    # Measured with the UNCALIBRATED readout by construction: no calibration
    # path is passed, so the configuration carries scale 1.0 and the pool means
    # this script reads are the raw boundary quantity.
    controller = FullBrainController.from_artifact(
        run_id="full-readout-calibration",
        entity_id="calibration-fly",
        configuration=FullBrainConfiguration(kernel_threads=arguments.threads),
        artifact_directory=arguments.artifact_directory,
        annotations_path=arguments.annotations,
        expected_manifest_sha256=arguments.artifact_manifest_sha256,
        embodied_senses=eem or escape,
        embodied_encoder_version=2 if eem_v2 else 1,
        body_annotations_path=arguments.body_annotations if eem or escape else None,
        escape_reflex=escape,
        escape_version=escape_version,
        loom_encoder=loom_v1,
        loom_version=2 if arguments.profile == LOOM_V2_PROFILE else 1,
    )
    if controller.readout_scale != 1.0:
        raise SystemExit("the calibration must be measured with an uncalibrated readout")
    escape_binding = controller.escape_binding
    if escape and (escape_binding is None or escape_binding.jump_scale != 1.0):
        raise SystemExit("the escape calibration must be measured with jump_scale 1.0")
    constructed = time.perf_counter()
    if escape:
        measurement = measure_escape(controller, stream_id=stream_id)
        measured = time.perf_counter()
        document = build_escape_document(controller, measurement, profile=arguments.profile)
    else:
        measurement = measure(controller, stream_id=stream_id, collisions=eem)
        measured = time.perf_counter()
        document = build_document(controller, measurement, profile=arguments.profile)
    digest, action = write_or_verify(output, document)

    calibration = document["calibration"]
    if escape:
        for group_name in ("wing", "jump"):
            group = document["distributions"][group_name]
            print(
                f"{action} {stream_id} [{group_name}]: "
                f"n={group['count']} "
                f"p50={group['p50']['value']:.6e} "
                f"p95={group['p95']['value']:.6e} "
                f"p99={group['p99']['value']:.6e} "
                f"max={group['max']['value']:.6e} | "
                f"{group_name}_scale={calibration[f'{group_name}_scale']:.6f} "
                f"(saturating "
                f"{calibration['disclosure'][f'{group_name}_saturating_scale']:.6f})"
            )
        print(f"covered {document['coverage']['run_is_covered']}")
    else:
        distribution = document["distribution"]
        print(
            f"{action} {stream_id}: "
            f"n={distribution['count']} "
            f"p50={distribution['p50']['value']:.6e} "
            f"p95={distribution['p95']['value']:.6e} "
            f"p99={distribution['p99']['value']:.6e} "
            f"max={distribution['max']['value']:.6e} | "
            f"readout_scale={calibration['readout_scale']:.6f} "
            f"(saturating {calibration['disclosure']['saturating_scale']:.6f}) | "
            f"covered {document['coverage']['run_is_covered']}"
        )
    print(f"{action} {output}")
    print(f"document sha256: {digest}")
    if eem or escape:
        encoder = document["encoder"]
        print(
            f"encoder identity sha256: {encoder['identity_sha256']} "
            f"(population binding {encoder['population_binding_sha256']})"
        )
    if escape:
        jump_readout = document["jump_readout"]
        print(f"ttmn pool binding sha256: {jump_readout['ttmn_pool_binding_sha256']}")
    print(
        "wall time (environment-scoped, not part of the canonical document): "
        f"construction {constructed - started:.3f}s, "
        f"measurement {measured - constructed:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
