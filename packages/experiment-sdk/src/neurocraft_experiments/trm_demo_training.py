"""Behavioral-cloning training for the TRM live demo (v1 and v2 rounds).

Implements exactly the declared designs in
``experiments/preregistration/trm-live-demo-v1.md`` and
``experiments/preregistration/trm-live-demo-v2.md``: synthetic,
SHA-256-derived sensor trajectories pass through the *pinned* faithful sensory
path — the ``PhysicalFeatureEncoder``, the controller's own sensor channels,
and five engine steps per tick with the encoded stimulation held — while the
committed ``scripted_baseline`` labels every frame. A ridge readout with
intercept then maps the post-step hidden state to the expert's three ray-only
action components. New module; no pilot-pinned byte is touched.

The two rounds are explicit, never a silent default change: every ``*_v2``
name implements the v2 identity (hold segments, zero washout, pure-constant
scenarios — the recorded fix for the saturation `EXTERNAL-BRIDGE-06`
measured), while the unsuffixed v1 functions keep their exact v1 behavior so
the committed v1 artifact stays reproducible as history.

Open-loop behavioral cloning, disclosed as such. The artifacts this module
writes are the single source of the live demo's trained weights, and the
current round's SHA-256 is what the demo configuration binds.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from neurocraft_connectome import RuntimeGraph
from neurocraft_connectome.model import canonical_json_bytes, sha256_json
from neurocraft_controller import ConnectomeController, ControllerConfiguration
from neurocraft_controller.features import PhysicalFeatureEncoder
from neurocraft_neural import NeuralEngine
from neurocraft_neural.adapters import SensoryAdapter
from neurocraft_neural.model import SignOverlay
from neurocraft_protocol import SensorFrame

from .baseline import RawPhysicalObservation, scripted_baseline
from .reservoir import ridge_readout
from .trm_tasks import unit_stream

DEMO_ID = "trm-live-demo-v1"
ARTIFACT_SCHEMA = "neurocraft.trm.cloned_readout.v1"

SCENARIO_COUNT = 60
TRAIN_SCENARIOS = range(0, 40)
VALIDATION_SCENARIOS = range(40, 50)
TEST_SCENARIOS = range(50, 60)
TICKS_PER_SCENARIO = 200
WASHOUT_TICKS = 20
SEGMENT_TICKS = 50
NEURAL_STEPS_PER_TICK = 5

RAY_MIN_M = 0.3
RAY_MAX_M = 6.0
WALK_STEP_M = 0.6
APPROACH_DECAY = 0.85
OPEN_DRIFT_M = 0.2
LOCAL_LIGHT = 8

CLONED_CHANNELS = ("forward", "lateral", "yaw")
L2_GRID = (1e-6, 1e-4, 1e-2, 1.0)

#: Segment behaviors, chosen per segment from the scenario's mode stream. Every
#: expert branch (blocked forward, left-blocked, right-blocked, open) appears.
_SEGMENT_MODES = ("approach_forward", "approach_left", "approach_right", "open")
_RAY_CHANNELS = {"approach_forward": 0, "approach_left": 1, "approach_right": 2}

#: Yaw directional-sanity threshold: frames whose forward ray is under this are
#: "blocked"; agreement is counted only where the expert's yaw is non-trivial.
BLOCKED_FORWARD_M = 1.0
YAW_EPSILON = 0.05

# --- v2 round (trm-live-demo-v2): the recorded fix for the saturation that ---
# --- EXTERNAL-BRIDGE-06 measured. Everything unsuffixed above stays v1.    ---

DEMO_ID_V2 = "trm-live-demo-v2"

#: v2 keeps every tick: the live loop has no washout and starts a fresh
#: engine, so cold-start transient states are in-distribution by construction.
WASHOUT_TICKS_V2 = 0

WALK_SCENARIO_COUNT_V2 = 60
CONSTANT_SCENARIO_COUNT_V2 = 8
SCENARIO_COUNT_V2 = WALK_SCENARIO_COUNT_V2 + CONSTANT_SCENARIO_COUNT_V2

#: Declared v2 split, extending the v1 40/10/10 proportionally: each split
#: keeps its v1 walk indices and adds declared constant scenarios.
TRAIN_SCENARIOS_V2 = (*range(0, 40), *range(60, 64))
VALIDATION_SCENARIOS_V2 = (*range(40, 50), *range(64, 66))
TEST_SCENARIOS_V2 = (*range(50, 60), *range(66, 68))

#: v2 segment behaviors: the four v1 modes plus ``hold`` listed twice, so a
#: hold segment — rays frozen for the whole segment — is drawn with
#: probability 1/3 and quasi-static input becomes common in training.
_SEGMENT_MODES_V2 = (
    "approach_forward",
    "approach_left",
    "approach_right",
    "open",
    "hold",
    "hold",
)

#: The eight declared pure-constant scenarios (ray0 forward, ray2 left,
#: ray3 right, metres), fixed for all 200 ticks: open, blocked-forward,
#: left-blocked, and right-blocked cases, one milder and one sharper variant
#: each, exactly as recorded in trm-live-demo-v2.md.
CONSTANT_SCENARIO_RAYS_V2: tuple[tuple[float, float, float], ...] = (
    (6.0, 6.0, 6.0),
    (0.5, 6.0, 6.0),
    (2.0, 1.0, 2.4),
    (2.0, 2.4, 1.0),
    (4.0, 5.0, 5.0),
    (0.8, 3.0, 3.0),
    (0.9, 0.6, 3.0),
    (0.9, 3.0, 0.6),
)


class TrmDemoTrainingError(ValueError):
    """Raised when trajectory generation, collection, or writing is invalid."""


def _clip_ray(value: float) -> float:
    return min(RAY_MAX_M, max(RAY_MIN_M, value))


def ray_trajectory(scenario: int) -> tuple[tuple[float, float, float], ...]:
    """One scenario's (ray0, ray2, ray3) per tick, from declared streams."""

    if isinstance(scenario, bool) or not isinstance(scenario, int) or scenario < 0:
        raise TrmDemoTrainingError("scenario index must be a non-negative integer")
    walk = [
        unit_stream(DEMO_ID, "walk", seed_index=scenario, channel=c, length=TICKS_PER_SCENARIO)
        for c in range(3)
    ]
    segments = TICKS_PER_SCENARIO // SEGMENT_TICKS
    mode_stream = unit_stream(DEMO_ID, "mode", seed_index=scenario, channel=0, length=segments)
    rays = [RAY_MAX_M, RAY_MAX_M, RAY_MAX_M]
    out: list[tuple[float, float, float]] = []
    for tick in range(TICKS_PER_SCENARIO):
        mode = _SEGMENT_MODES[int(mode_stream[tick // SEGMENT_TICKS] * len(_SEGMENT_MODES))]
        for channel in range(3):
            step = (walk[channel][tick] - 0.5) * WALK_STEP_M
            rays[channel] = _clip_ray(rays[channel] + step)
        if mode == "open":
            for channel in range(3):
                rays[channel] = _clip_ray(rays[channel] + OPEN_DRIFT_M)
        else:
            target = _RAY_CHANNELS[mode]
            rays[target] = _clip_ray(rays[target] * APPROACH_DECAY)
        out.append((rays[0], rays[1], rays[2]))
    return tuple(out)


def _walk_trajectory_v2(
    scenario: int,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[bool, ...]]:
    """One v2 walk scenario's rays plus its per-tick quasi-static flags."""

    walk = [
        unit_stream(DEMO_ID_V2, "walk", seed_index=scenario, channel=c, length=TICKS_PER_SCENARIO)
        for c in range(3)
    ]
    segments = TICKS_PER_SCENARIO // SEGMENT_TICKS
    mode_stream = unit_stream(DEMO_ID_V2, "mode", seed_index=scenario, channel=0, length=segments)
    rays = [RAY_MAX_M, RAY_MAX_M, RAY_MAX_M]
    out: list[tuple[float, float, float]] = []
    quasi: list[bool] = []
    for tick in range(TICKS_PER_SCENARIO):
        mode = _SEGMENT_MODES_V2[int(mode_stream[tick // SEGMENT_TICKS] * len(_SEGMENT_MODES_V2))]
        if mode == "hold":
            # Rays frozen for the whole segment at the value they carried
            # entering it: no walk step, no drift, no decay.
            quasi.append(True)
            out.append((rays[0], rays[1], rays[2]))
            continue
        quasi.append(False)
        for channel in range(3):
            step = (walk[channel][tick] - 0.5) * WALK_STEP_M
            rays[channel] = _clip_ray(rays[channel] + step)
        if mode == "open":
            for channel in range(3):
                rays[channel] = _clip_ray(rays[channel] + OPEN_DRIFT_M)
        else:
            target = _RAY_CHANNELS[mode]
            rays[target] = _clip_ray(rays[target] * APPROACH_DECAY)
        out.append((rays[0], rays[1], rays[2]))
    return tuple(out), tuple(quasi)


def _trajectory_and_mask_v2(
    scenario: int,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[bool, ...]]:
    if isinstance(scenario, bool) or not isinstance(scenario, int) or scenario < 0:
        raise TrmDemoTrainingError("scenario index must be a non-negative integer")
    if scenario >= SCENARIO_COUNT_V2:
        raise TrmDemoTrainingError(f"v2 scenario index must be below {SCENARIO_COUNT_V2}")
    if scenario < WALK_SCENARIO_COUNT_V2:
        return _walk_trajectory_v2(scenario)
    fixed = CONSTANT_SCENARIO_RAYS_V2[scenario - WALK_SCENARIO_COUNT_V2]
    return (fixed,) * TICKS_PER_SCENARIO, (True,) * TICKS_PER_SCENARIO


def ray_trajectory_v2(scenario: int) -> tuple[tuple[float, float, float], ...]:
    """One v2 scenario's (ray0, ray2, ray3) per tick, from declared streams."""

    return _trajectory_and_mask_v2(scenario)[0]


def quasi_static_mask_v2(scenario: int) -> tuple[bool, ...]:
    """Per-tick flags: True where the v2 input is quasi-static (hold segments
    of walk scenarios; every tick of a pure-constant scenario)."""

    return _trajectory_and_mask_v2(scenario)[1]


def sensor_frame(ray0: float, ray2: float, ray3: float) -> SensorFrame:
    """A benign-constant frame carrying only the declared varying rays."""

    return SensorFrame(
        position_m=(0.0, 80.0, 0.0),
        velocity_m_s=(0.0, 0.0, 0.0),
        yaw_rad=0.0,
        pitch_rad=0.0,
        on_ground=False,
        horizontal_collision=False,
        vertical_collision=False,
        in_fluid=False,
        local_light=LOCAL_LIGHT,
        ray_max_distance_m=RAY_MAX_M,
        ray_distances_m=(ray0, RAY_MAX_M, ray2, ray3, RAY_MAX_M, RAY_MAX_M),
    )


def expert_targets(ray0: float, ray2: float, ray3: float) -> tuple[float, float, float]:
    """The committed expert's ray-only components on one frame.

    ``on_ground=False`` and zero vertical velocity make the expert's vertical
    component exactly zero, so nothing vertical is silently discarded here —
    the exclusion happens by declared information availability.
    """

    primitive = scripted_baseline(
        RawPhysicalObservation(
            forward_clearance=ray0,
            left_clearance=ray2,
            right_clearance=ray3,
            vertical_velocity=0.0,
            on_ground=False,
        )
    )
    return (primitive.forward_thrust, primitive.lateral_thrust, primitive.yaw_torque)


def collect_scenarios(
    graph: RuntimeGraph, scenarios: Sequence[int]
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[tuple[float, float, float], ...],
    tuple[tuple[float, float, float], ...],
]:
    """Roll the declared trajectories through the pinned faithful sensory path.

    Returns ``(hidden_states, targets, rays)`` with washout ticks discarded; a
    fresh engine per scenario. The encoder/adapter/step count reproduce the
    live faithful controller exactly.
    """

    # The pinned faithful sensory path, built directly: the default controller
    # cannot construct on the steering graph (its default motor ids are v1
    # bodies), so the encoder and the controller's own channel builder are
    # used piecewise — identical identities, no readout involved.
    features = PhysicalFeatureEncoder()
    adapter = SensoryAdapter(
        graph, ConnectomeController._sensor_channels(ControllerConfiguration())
    )
    consensus = SignOverlay.coarse_presynaptic_consensus(graph)
    states: list[tuple[float, ...]] = []
    targets: list[tuple[float, float, float]] = []
    rays_kept: list[tuple[float, float, float]] = []
    for scenario in scenarios:
        engine = NeuralEngine(
            graph, sign_overlay=consensus, sensory_adapter_sha256=adapter.identity_sha256
        )
        for tick, (ray0, ray2, ray3) in enumerate(ray_trajectory(scenario)):
            frame = sensor_frame(ray0, ray2, ray3)
            encoded = adapter.encode(features.encode(frame).values)
            for _ in range(NEURAL_STEPS_PER_TICK):
                engine.step(encoded)
            if tick < WASHOUT_TICKS:
                continue
            states.append(engine.current_state().activity)
            targets.append(expert_targets(ray0, ray2, ray3))
            rays_kept.append((ray0, ray2, ray3))
    return tuple(states), tuple(targets), tuple(rays_kept)


def collect_scenarios_v2(
    graph: RuntimeGraph, scenarios: Sequence[int]
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[tuple[float, float, float], ...],
    tuple[tuple[float, float, float], ...],
    tuple[bool, ...],
]:
    """Roll the declared v2 trajectories through the pinned faithful sensory
    path.

    Returns ``(hidden_states, targets, rays, quasi_static_flags)``; a fresh
    engine per scenario, exactly the v1 collection except that
    ``WASHOUT_TICKS_V2 = 0`` keeps every cold-start tick and each frame
    carries its declared quasi-static flag. The encoder/adapter/step count
    reproduce the live faithful controller exactly. Kept parallel to the v1
    ``collect_scenarios`` on purpose: the v1 code path stays byte-stable while
    the committed v1 artifact remains reproducible history.
    """

    features = PhysicalFeatureEncoder()
    adapter = SensoryAdapter(
        graph, ConnectomeController._sensor_channels(ControllerConfiguration())
    )
    consensus = SignOverlay.coarse_presynaptic_consensus(graph)
    states: list[tuple[float, ...]] = []
    targets: list[tuple[float, float, float]] = []
    rays_kept: list[tuple[float, float, float]] = []
    quasi_kept: list[bool] = []
    for scenario in scenarios:
        engine = NeuralEngine(
            graph, sign_overlay=consensus, sensory_adapter_sha256=adapter.identity_sha256
        )
        trajectory, mask = _trajectory_and_mask_v2(scenario)
        for (ray0, ray2, ray3), quasi_static in zip(trajectory, mask, strict=True):
            frame = sensor_frame(ray0, ray2, ray3)
            encoded = adapter.encode(features.encode(frame).values)
            for _ in range(NEURAL_STEPS_PER_TICK):
                engine.step(encoded)
            states.append(engine.current_state().activity)
            targets.append(expert_targets(ray0, ray2, ray3))
            rays_kept.append((ray0, ray2, ray3))
            quasi_kept.append(quasi_static)
    return tuple(states), tuple(targets), tuple(rays_kept), tuple(quasi_kept)


def _augmented(rows: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    return tuple((*row, 1.0) for row in rows)


def _predict(
    row: Sequence[float], weights: Sequence[Sequence[float]]
) -> tuple[float, float, float]:
    forward, lateral, yaw = (
        math.fsum(row[feature] * weights[feature][out] for feature in range(len(row)))
        for out in range(3)
    )
    return (forward, lateral, yaw)


def _mse_per_channel(
    rows: Sequence[Sequence[float]],
    targets: Sequence[tuple[float, float, float]],
    weights: Sequence[Sequence[float]],
) -> tuple[float, float, float]:
    totals = [0.0, 0.0, 0.0]
    for row, target in zip(rows, targets, strict=True):
        prediction = _predict(row, weights)
        for out in range(3):
            totals[out] += (prediction[out] - target[out]) ** 2
    count = len(rows)
    if count == 0:
        raise TrmDemoTrainingError("metrics require at least one row")
    result = tuple(total / count for total in totals)
    if any(not math.isfinite(v) for v in result):
        raise TrmDemoTrainingError("non-finite mse in cloned-readout metrics")
    return (result[0], result[1], result[2])


def _r2_per_channel(
    rows: Sequence[Sequence[float]],
    targets: Sequence[tuple[float, float, float]],
    weights: Sequence[Sequence[float]],
) -> tuple[float, float, float]:
    mse = _mse_per_channel(rows, targets, weights)
    out: list[float] = []
    count = len(targets)
    for channel in range(3):
        mean = math.fsum(target[channel] for target in targets) / count
        variance = math.fsum((target[channel] - mean) ** 2 for target in targets) / count
        out.append(1.0 - mse[channel] / variance if variance > 0 else 0.0)
    return (out[0], out[1], out[2])


def train_cloned_readout(graph: RuntimeGraph) -> dict[str, Any]:
    """The full declared training run; returns the artifact document."""

    train_states, train_targets, _ = collect_scenarios(graph, list(TRAIN_SCENARIOS))
    val_states, val_targets, _ = collect_scenarios(graph, list(VALIDATION_SCENARIOS))
    test_states, test_targets, test_rays = collect_scenarios(graph, list(TEST_SCENARIOS))

    x_train = _augmented(train_states)
    x_val = _augmented(val_states)
    x_test = _augmented(test_states)
    best: tuple[float, float, tuple[tuple[float, ...], ...]] | None = None
    for l2 in L2_GRID:
        weights = ridge_readout(x_train, train_targets, l2=l2)
        val_mse = sum(_mse_per_channel(x_val, val_targets, weights)) / 3.0
        if best is None or val_mse < best[1]:
            best = (l2, val_mse, weights)
    assert best is not None
    chosen_l2, validation_mean_mse, weights = best

    test_mse = _mse_per_channel(x_test, test_targets, weights)
    test_r2 = _r2_per_channel(x_test, test_targets, weights)

    agreements = 0
    considered = 0
    for row, target, rays in zip(x_test, test_targets, test_rays, strict=True):
        if rays[0] >= BLOCKED_FORWARD_M or abs(target[2]) <= YAW_EPSILON:
            continue
        considered += 1
        if math.copysign(1.0, _predict(row, weights)[2]) == math.copysign(1.0, target[2]):
            agreements += 1

    return {
        "schema": ARTIFACT_SCHEMA,
        "demo_id": DEMO_ID,
        "graph_runtime_identity_sha256": graph.manifest.runtime_identity_sha256,
        "channels": list(CLONED_CHANNELS),
        "chosen_l2": chosen_l2,
        "feature_width": len(x_train[0]),
        "generator": {
            "scenario_count": SCENARIO_COUNT,
            "ticks_per_scenario": TICKS_PER_SCENARIO,
            "washout_ticks": WASHOUT_TICKS,
            "segment_ticks": SEGMENT_TICKS,
            "ray_bounds_m": [RAY_MIN_M, RAY_MAX_M],
            "walk_step_m": WALK_STEP_M,
            "approach_decay": APPROACH_DECAY,
            "open_drift_m": OPEN_DRIFT_M,
            "local_light": LOCAL_LIGHT,
            "neural_steps_per_tick": NEURAL_STEPS_PER_TICK,
        },
        "metrics": {
            "validation_mean_mse": validation_mean_mse,
            "test_mse": {c: test_mse[i] for i, c in enumerate(CLONED_CHANNELS)},
            "test_r2": {c: test_r2[i] for i, c in enumerate(CLONED_CHANNELS)},
            "yaw_directional_agreement": {
                "considered_frames": considered,
                "agreeing_frames": agreements,
                "rate": (agreements / considered) if considered else None,
            },
        },
        "weights_float_hex": [[float(v).hex() for v in row] for row in weights],
        "weights_identity_sha256": sha256_json([[float(v).hex() for v in row] for row in weights]),
    }


def _select_l2(
    x_train: Sequence[Sequence[float]],
    train_targets: Sequence[tuple[float, float, float]],
    x_val: Sequence[Sequence[float]],
    val_targets: Sequence[tuple[float, float, float]],
) -> tuple[float, float, tuple[tuple[float, ...], ...]]:
    """The declared grid search: mean validation MSE selects from L2_GRID."""

    best: tuple[float, float, tuple[tuple[float, ...], ...]] | None = None
    for l2 in L2_GRID:
        weights = ridge_readout(x_train, train_targets, l2=l2)
        # _mse_per_channel refuses non-finite values, so val_mse is always
        # finite here and a blown-up candidate can never poison the running
        # best via NaN comparison — it fails loudly at its own grid point.
        val_mse = sum(_mse_per_channel(x_val, val_targets, weights)) / 3.0
        if best is None or val_mse < best[1]:
            best = (l2, val_mse, weights)
    assert best is not None
    return best


def _yaw_directional_agreement(
    rows: Sequence[Sequence[float]],
    targets: Sequence[tuple[float, float, float]],
    rays: Sequence[tuple[float, float, float]],
    weights: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """The v1 disclosure pattern: blocked-forward frames, non-trivial expert
    yaw only, sign agreement reported with its thresholds — never gated."""

    agreements = 0
    considered = 0
    for row, target, frame_rays in zip(rows, targets, rays, strict=True):
        if frame_rays[0] >= BLOCKED_FORWARD_M or abs(target[2]) <= YAW_EPSILON:
            continue
        considered += 1
        if math.copysign(1.0, _predict(row, weights)[2]) == math.copysign(1.0, target[2]):
            agreements += 1
    return {
        "blocked_forward_below_m": BLOCKED_FORWARD_M,
        "considered_frames": considered,
        "agreeing_frames": agreements,
        "rate": (agreements / considered) if considered else None,
        "yaw_epsilon": YAW_EPSILON,
    }


def _saturation_bucket(
    rows: Sequence[Sequence[float]],
    weights: Sequence[Sequence[float]],
) -> dict[str, Any]:
    frame_count = len(rows)
    exceeded = [0, 0, 0]
    any_exceeded = 0
    for row in rows:
        prediction = _predict(row, weights)
        frame_hit = False
        for out in range(3):
            if abs(prediction[out]) > 1.0:
                exceeded[out] += 1
                frame_hit = True
        if frame_hit:
            any_exceeded += 1

    def fraction(count: int) -> float | None:
        return count / frame_count if frame_count else None

    return {
        "frame_count": frame_count,
        "per_channel": {c: fraction(exceeded[i]) for i, c in enumerate(CLONED_CHANNELS)},
        "any_channel": fraction(any_exceeded),
    }


def _pre_clip_saturation(
    rows: Sequence[Sequence[float]],
    quasi_flags: Sequence[bool],
    weights: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """The saturation-relevant v2 metric: the fraction of held-out frames
    whose raw readout value exceeds 1.0 in magnitude before the clip,
    reported separately for v1-like varying frames and quasi-static frames."""

    varying = [row for row, quasi in zip(rows, quasi_flags, strict=True) if not quasi]
    quasi_static = [row for row, quasi in zip(rows, quasi_flags, strict=True) if quasi]
    return {
        "definition": (
            "fraction of held-out test frames whose raw readout value exceeds "
            "1.0 in magnitude before the [-1, 1] clip"
        ),
        "varying": _saturation_bucket(varying, weights),
        "quasi_static": _saturation_bucket(quasi_static, weights),
    }


def train_cloned_readout_v2(graph: RuntimeGraph) -> dict[str, Any]:
    """The full declared v2 training run; returns the artifact document."""

    train_states, train_targets, _, _ = collect_scenarios_v2(graph, TRAIN_SCENARIOS_V2)
    val_states, val_targets, _, _ = collect_scenarios_v2(graph, VALIDATION_SCENARIOS_V2)
    test_states, test_targets, test_rays, test_quasi = collect_scenarios_v2(
        graph, TEST_SCENARIOS_V2
    )

    x_train = _augmented(train_states)
    x_val = _augmented(val_states)
    x_test = _augmented(test_states)
    chosen_l2, validation_mean_mse, weights = _select_l2(x_train, train_targets, x_val, val_targets)

    test_mse = _mse_per_channel(x_test, test_targets, weights)
    test_r2 = _r2_per_channel(x_test, test_targets, weights)

    return {
        "schema": ARTIFACT_SCHEMA,
        "demo_id": DEMO_ID_V2,
        "graph_runtime_identity_sha256": graph.manifest.runtime_identity_sha256,
        "channels": list(CLONED_CHANNELS),
        "chosen_l2": chosen_l2,
        "feature_width": len(x_train[0]),
        "generator": {
            "scenario_count": SCENARIO_COUNT_V2,
            "walk_scenario_count": WALK_SCENARIO_COUNT_V2,
            "constant_scenario_count": CONSTANT_SCENARIO_COUNT_V2,
            "constant_scenario_rays_m": [list(rays) for rays in CONSTANT_SCENARIO_RAYS_V2],
            "segment_modes": list(_SEGMENT_MODES_V2),
            "ticks_per_scenario": TICKS_PER_SCENARIO,
            "washout_ticks": WASHOUT_TICKS_V2,
            "segment_ticks": SEGMENT_TICKS,
            "ray_bounds_m": [RAY_MIN_M, RAY_MAX_M],
            "walk_step_m": WALK_STEP_M,
            "approach_decay": APPROACH_DECAY,
            "open_drift_m": OPEN_DRIFT_M,
            "local_light": LOCAL_LIGHT,
            "neural_steps_per_tick": NEURAL_STEPS_PER_TICK,
            "scenario_split": {
                "train": list(TRAIN_SCENARIOS_V2),
                "validation": list(VALIDATION_SCENARIOS_V2),
                "test": list(TEST_SCENARIOS_V2),
            },
        },
        "metrics": {
            "validation_mean_mse": validation_mean_mse,
            "test_mse": {c: test_mse[i] for i, c in enumerate(CLONED_CHANNELS)},
            "test_r2": {c: test_r2[i] for i, c in enumerate(CLONED_CHANNELS)},
            "yaw_directional_agreement": _yaw_directional_agreement(
                x_test, test_targets, test_rays, weights
            ),
            "pre_clip_saturation": _pre_clip_saturation(x_test, test_quasi, weights),
        },
        "weights_float_hex": [[float(v).hex() for v in row] for row in weights],
        "weights_identity_sha256": sha256_json([[float(v).hex() for v in row] for row in weights]),
    }


def write_artifact(path: Path, document: dict[str, Any]) -> str:
    """Write the artifact exactly once and return its file SHA-256."""

    if path.exists():
        raise TrmDemoTrainingError(
            f"readout artifact already exists at {path}; artifacts are immutable"
        )
    payload = canonical_json_bytes(document) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256(payload).hexdigest()
