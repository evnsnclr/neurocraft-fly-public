#!/usr/bin/env python3
"""Probe the approach-gated loom-v1 boundary against the escape-v2 boundary.

Five declared sensor sequences driven through the CALIBRATED
``full_mcns_loom_v1`` controller on the real full MaleCNS artifact, with the
CALIBRATED ``full_mcns_escape_v2`` controller driven through the identical
sequences as the paired comparison. The applied forward/vertical/yaw per tick
and the TTMn jump term are reported for both.

The five conditions (declared before any measurement, in
``docs/encoders/eem-loom-v1.md``):

1. ``standing_front`` -- one open tick, then the ``ESCAPE-ARC-01`` blocked
   frontal ray (0.5 m) held for 80 ticks. Reports the first tick from which
   ``|output - baseline| < 1e-3`` holds on every axis for the rest of the run,
   where the baseline is the paired all-open control run at the same tick
   index.
2. ``approach_ramp`` -- front-ray proximity rising linearly from 0 to the
   blocked value over 20 ticks, then held for 60. Reports the peak jump and
   vertical and the tick each occurred on.
3. ``recede`` -- front-ray proximity falling from the blocked value to 0 over
   20 ticks. Reports the maximum ``threat_front`` drive over the run.
4. ``touch`` -- quiet, a collision onset, four held collision ticks, then
   release. Compared tick-by-tick against escape-v2 for bitwise equality of
   the touch channel, the bristle drive, and the applied motors.
5. ``escape_arc`` -- the ``ESCAPE-ARC-01`` reference stimulus (blocked frontal
   ray) from a fresh reset, 30-tick hold. Reports the DNp01 pair mean and the
   TTMn pool mean on the FIRST tick after onset and over the last 10 ticks,
   each against the paired all-open control, so the record shows what the new
   encoding does to the arc's EVENT response versus its STANDING response.

Nothing here is fitted. The encoder identity, alpha, the rectification set,
and the calibration were all frozen before this script ran; whatever it
measures is reported, including anything disappointing.

The DNp01 read is offline and diagnostic-only, exactly as in
``scripts/measure_escape_arc.py``: the live decoder never reads DNp01, and
this measurement does not weaken that claim boundary.

Offline dynamics measurement over declared boundaries only -- no behavior,
performance, or biological claim, and no claim that this is how LC4 computes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from neurocraft_controller.embodied_senses import (
    SIDE_COLUMN,
    TYPE_COLUMN,
    _string_column,
    _verified_annotation_table,
    _verified_graph_array,
)
from neurocraft_controller.full_brain_controller import (
    FULL_MCNS_ARTIFACT_MANIFEST_SHA256,
    FullBrainConfiguration,
    FullBrainController,
)
from neurocraft_protocol import Observation, SensorFrame

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIRECTORY = REPO_ROOT / "data" / "processed" / "male-cns-v1.0-superclass-v1"
DEFAULT_ANNOTATIONS = (
    REPO_ROOT / "data" / "cache" / "male-cns-v1.0"
) / "body-neurotransmitters-male-cns-v1.0.feather"
DEFAULT_BODY_ANNOTATIONS = (
    REPO_ROOT / "data" / "cache" / "male-cns-v1.0"
) / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
DEFAULT_LOOM_CALIBRATION = (
    REPO_ROOT / "data" / "calibration" / "full-readout-calibration-loom-v1.json"
)
DEFAULT_ESCAPE_V2_CALIBRATION = (
    REPO_ROOT / "data" / "calibration" / "full-readout-calibration-escape-v2.json"
)
#: The committed ESCAPE-ARC-01 evidence. Condition 5 cross-checks the
#: escape-v2 lane against it bitwise, which is what licenses saying this probe
#: measures the receipt's series rather than a lookalike.
DEFAULT_ESCAPE_ARC_RECEIPT = REPO_ROOT / "docs" / "receipts" / "ESCAPE-ARC-01.json"

#: The ``ESCAPE-ARC-01`` stimulus geometry, taken verbatim from
#: ``scripts/measure_escape_arc.py`` so the two records are comparable.
RAY_MAX_DISTANCE_M = 6.0
BLOCKED_RAY_M = 0.5
GAME_TICK_NS = 50_000_000
ARC_HOLD_TICKS = 30
ARC_SUMMARY_WINDOW_TICKS = 10
STANDING_HOLD_TICKS = 80
RAMP_TICKS = 20
RAMP_HOLD_TICKS = 60
SETTLE_THRESHOLD = 1.0e-3

#: Proximity of the blocked frontal ray, so the ramps are declared in the
#: encoder's own units rather than in metres.
BLOCKED_PROXIMITY = 1.0 - BLOCKED_RAY_M / RAY_MAX_DISTANCE_M

#: The two DNp01 giant descending neurons, selected the way
#: ``scripts/measure_escape_arc.py`` selects them (offline, diagnostic-only).
DNP01_TYPE = "DNp01"
EXPECTED_DNP01_NODES = 2

MOTOR_AXES = ("forward", "vertical", "yaw", "lateral")


class LoomProbeError(RuntimeError):
    """The loom probe battery could not be executed exactly as declared."""


def sensors(
    *,
    front: float = RAY_MAX_DISTANCE_M,
    light: int = 15,
    horizontal_collision: bool = False,
) -> SensorFrame:
    """One sensor frame with every unnamed ray open and the light at maximum."""

    return SensorFrame(
        position_m=(0.0, 66.0, 0.0),
        velocity_m_s=(0.0, 0.0, 0.0),
        yaw_rad=0.0,
        pitch_rad=0.0,
        on_ground=False,
        horizontal_collision=horizontal_collision,
        vertical_collision=False,
        in_fluid=False,
        local_light=light,
        ray_max_distance_m=RAY_MAX_DISTANCE_M,
        ray_distances_m=(
            front,
            RAY_MAX_DISTANCE_M,
            RAY_MAX_DISTANCE_M,
            RAY_MAX_DISTANCE_M,
            RAY_MAX_DISTANCE_M,
            RAY_MAX_DISTANCE_M,
        ),
    )


def _distance_for(proximity: float) -> float:
    """Invert the declared proximity formula ``clip(1 - d / ray_max, 0, 1)``."""

    return RAY_MAX_DISTANCE_M * (1.0 - proximity)


# ------------------------------------------------------------------- sequences


def sequence_standing_front(hold_ticks: int = STANDING_HOLD_TICKS) -> list[SensorFrame]:
    """One open tick, then the blocked frontal ray held."""

    return [sensors(), *[sensors(front=BLOCKED_RAY_M)] * hold_ticks]


def sequence_all_open(length: int) -> list[SensorFrame]:
    """The paired control: every ray open for the same number of ticks."""

    return [sensors()] * length


def sequence_approach_ramp() -> list[SensorFrame]:
    """Front proximity rising linearly 0 -> blocked over 20 ticks, then held."""

    ramp = [
        sensors(front=_distance_for(BLOCKED_PROXIMITY * step / RAMP_TICKS))
        for step in range(RAMP_TICKS + 1)
    ]
    return [*ramp, *[sensors(front=BLOCKED_RAY_M)] * RAMP_HOLD_TICKS]


def sequence_recede() -> list[SensorFrame]:
    """Front proximity falling linearly blocked -> 0 over 20 ticks."""

    return [
        sensors(front=_distance_for(BLOCKED_PROXIMITY * (1.0 - step / RAMP_TICKS)))
        for step in range(RAMP_TICKS + 1)
    ]


def sequence_touch() -> list[SensorFrame]:
    """Quiet, a collision onset, four held ticks, then release."""

    flags = (False, True, True, True, True, True, False, False)
    return [sensors(horizontal_collision=flag) for flag in flags]


def sequence_escape_arc() -> list[SensorFrame]:
    """The ESCAPE-ARC-01 reference stimulus: blocked frontal ray, 30-tick hold."""

    return [sensors(front=BLOCKED_RAY_M)] * ARC_HOLD_TICKS


# --------------------------------------------------------------- measurement


def dnp01_dense_indices(artifact_directory: Path, body_annotations: Path) -> tuple[int, ...]:
    """The two DNp01 dense indices, selected offline for the diagnostic read.

    The same digest-verified join discipline
    ``scripts/measure_escape_arc.select_dnp01_pair`` uses: the annotation table
    is opened only after its declared digest matches, and the pointer rows are
    taken through the artifact's own ``annotation_source_rows``.
    """

    from neurocraft_connectome.full_runtime import load_full_runtime_artifact

    graph = load_full_runtime_artifact(
        artifact_directory, expected_manifest_sha256=FULL_MCNS_ARTIFACT_MANIFEST_SHA256
    )
    try:
        table, _annotations_sha256 = _verified_annotation_table(graph, body_annotations)
        rows = _verified_graph_array(graph, "annotation_source_rows", "<u8").astype(np.int64)
        taken = table.take(rows)
        types = _string_column(taken, TYPE_COLUMN)
        sides = _string_column(taken, SIDE_COLUMN)
        indices = tuple(index for index, value in enumerate(types) if value == DNP01_TYPE)
        if len(indices) != EXPECTED_DNP01_NODES:
            raise LoomProbeError(
                f"expected exactly {EXPECTED_DNP01_NODES} {DNP01_TYPE} nodes, "
                f"selected {len(indices)}"
            )
        if {sides[index] for index in indices} != {"L", "R"}:
            raise LoomProbeError(f"the {DNP01_TYPE} pair is not one L and one R")
        return indices
    finally:
        graph.close()


def build_controller(
    *,
    artifact_directory: Path,
    annotations: Path,
    body_annotations: Path,
    calibration: Path,
    loom: bool,
    threads: int | None,
    loom_version: int = 1,
) -> FullBrainController:
    """One calibrated controller: the selected loom generation or escape-v2."""

    return FullBrainController.from_artifact(
        run_id="loom-probe",
        entity_id="probe-fly",
        configuration=FullBrainConfiguration(kernel_threads=threads),
        artifact_directory=artifact_directory,
        annotations_path=annotations,
        calibration_path=calibration,
        embodied_senses=True,
        body_annotations_path=body_annotations,
        escape_reflex=True,
        escape_version=2,
        loom_encoder=loom,
        loom_version=loom_version,
    )


@dataclass(slots=True)
class Driver:
    """One controller plus its monotonic session clock.

    ``reset()`` deliberately does not rewind the live session clocks, so each
    condition continues the sequence numbers while starting from fresh neural
    state and a fresh (declared-first-tick) encoder baseline.
    """

    controller: FullBrainController
    dnp01_indices: tuple[int, ...]
    _tick: int = 0

    def run(self, frames: Sequence[SensorFrame]) -> list[dict[str, float]]:
        controller = self.controller
        controller.reset()
        records: list[dict[str, float]] = []
        for offset, frame in enumerate(frames):
            tick = self._tick
            self._tick += 1
            action = controller.step(
                Observation(
                    identity=controller.identity,
                    observation_seq=tick,
                    game_tick=tick,
                    sent_monotonic_ns=tick * GAME_TICK_NS,
                    sensors=frame,
                ),
                command_seq=tick,
            )
            evidence = controller.last_evidence
            if evidence is None:  # pragma: no cover - a completed step records evidence
                raise LoomProbeError("the controller recorded no evidence for a completed step")
            stimulation = evidence.encoded_stimulation
            jump = evidence.jump_value
            record = {
                "tick": float(offset),
                "forward": action.motors.forward_thrust,
                "vertical": action.motors.vertical_thrust,
                "yaw": action.motors.yaw_torque,
                "lateral": action.motors.lateral_thrust,
                "jump": float("nan") if jump is None else jump,
                "ttmn_pool_mean": controller.jump_readout_mean(),
                "threat_front_drive": stimulation.get("eem:threat_front", float("nan")),
                "lc4_left_drive": stimulation.get("eem_drive:lc4_left", float("nan")),
                "lc4_right_drive": stimulation.get("eem_drive:lc4_right", float("nan")),
                "touch_drive": stimulation.get("eem:touch", float("nan")),
                "bristle_drive": stimulation.get("eem_drive:bristle", float("nan")),
            }
            if self.dnp01_indices:
                activity = controller.engine.activity()
                record["dnp01_pair_mean"] = math.fsum(
                    float(activity[index]) for index in self.dnp01_indices
                ) / len(self.dnp01_indices)
            records.append(record)
        return records


Records = Sequence[dict[str, float]]


def settle_tick(
    stimulus: Records, control: Records, *, axes: tuple[str, ...] = MOTOR_AXES
) -> int | None:
    """First tick from which every named axis stays within SETTLE_THRESHOLD.

    Compared against the PAIRED control run at the same tick index, so the
    baseline is the controller's own all-open trajectory rather than a
    remembered constant.
    """

    for start in range(len(stimulus)):
        if all(
            abs(stimulus[tick][axis] - control[tick][axis]) < SETTLE_THRESHOLD
            for tick in range(start, len(stimulus))
            for axis in axes
        ):
            return start
    return None


def peak(records: Records, key: str) -> tuple[int, float]:
    """The ``(tick, value)`` of the maximum ``key`` over the run."""

    best_tick, best_value = int(records[0]["tick"]), records[0][key]
    for record in records:
        if record[key] > best_value:
            best_tick, best_value = int(record["tick"]), record[key]
    return best_tick, best_value


def window_mean(records: Records, key: str, ticks: int) -> float:
    tail = [record[key] for record in records[-ticks:]]
    return math.fsum(tail) / len(tail)


def series(records: Records, key: str) -> list[float]:
    return [record[key] for record in records]


def ratio(numerator: float, denominator: float) -> float:
    return float("inf") if denominator == 0.0 else numerator / denominator


# ------------------------------------------------------------------- reporting


def row(label: str, loom: str = "", escape_v2: str = "") -> str:
    return f"{label:<34}{loom:>24}{escape_v2:>24}"


def fmt(value: float) -> str:
    if value != value:  # NaN
        return "n/a"
    if value == 0.0:
        return "0"
    if abs(value) < 1.0e-4:
        return f"{value:.3e}"
    return f"{value:.6f}"


def heading(name: str) -> None:
    print(f"\n--- {name} " + "-" * max(0, 78 - len(name)))
    print(row("", "loom-v1", "escape-v2"))


def _report_standing(
    loom: Driver, escape_v2: Driver, payload: dict[str, Any], hold_ticks: int
) -> None:
    frames = sequence_standing_front(hold_ticks)
    control_frames = sequence_all_open(len(frames))
    loom_run = loom.run(frames)
    loom_control = loom.run(control_frames)
    v2_run = escape_v2.run(frames)
    v2_control = escape_v2.run(control_frames)
    payload["standing_front"] = {
        "loom": loom_run,
        "loom_control": loom_control,
        "escape_v2": v2_run,
        "escape_v2_control": v2_control,
    }
    loom_settle = settle_tick(loom_run, loom_control)
    v2_settle = settle_tick(v2_run, v2_control)
    heading(f"1. standing blocked front ray ({hold_ticks} ticks held)")
    print(
        row(
            "threat_front drive @ onset",
            fmt(loom_run[1]["threat_front_drive"]),
            fmt(v2_run[1]["threat_front_drive"]),
        )
    )
    print(
        row(
            "threat_front drive @ last tick",
            fmt(loom_run[-1]["threat_front_drive"]),
            fmt(v2_run[-1]["threat_front_drive"]),
        )
    )
    print(
        row(
            "lc4_left drive @ last tick",
            fmt(loom_run[-1]["lc4_left_drive"]),
            fmt(v2_run[-1]["lc4_left_drive"]),
        )
    )
    for axis in MOTOR_AXES:
        print(
            row(
                f"{axis} @ last / control",
                f"{loom_run[-1][axis]:+.6f} / {loom_control[-1][axis]:+.6f}",
                f"{v2_run[-1][axis]:+.6f} / {v2_control[-1][axis]:+.6f}",
            )
        )
    print(row("jump @ last / control", fmt(loom_run[-1]["jump"]), fmt(v2_run[-1]["jump"])))
    print(
        row(
            "settle tick (|d| < 1e-3, all axes)",
            "never" if loom_settle is None else str(loom_settle),
            "never" if v2_settle is None else str(v2_settle),
        )
    )
    # Per axis, because the axes settle at very different rates and a single
    # "never" would hide which one is responsible.
    for axis in MOTOR_AXES:
        loom_axis = settle_tick(loom_run, loom_control, axes=(axis,))
        v2_axis = settle_tick(v2_run, v2_control, axes=(axis,))
        print(
            row(
                f"settle tick, {axis} only",
                "never" if loom_axis is None else str(loom_axis),
                "never" if v2_axis is None else str(v2_axis),
            )
        )


def _report_ramp(loom: Driver, escape_v2: Driver, payload: dict[str, Any]) -> None:
    frames = sequence_approach_ramp()
    loom_run = loom.run(frames)
    v2_run = escape_v2.run(frames)
    payload["approach_ramp"] = {"loom": loom_run, "escape_v2": v2_run}
    ramp_drives = series(loom_run, "threat_front_drive")[1 : RAMP_TICKS + 1]
    hold_drives = series(loom_run, "threat_front_drive")[RAMP_TICKS + 1 :]
    heading(f"2. approach ramp (0 -> blocked over {RAMP_TICKS} ticks, held {RAMP_HOLD_TICKS})")
    for key in ("jump", "vertical", "forward"):
        loom_peak = peak(loom_run, key)
        v2_peak = peak(v2_run, key)
        print(
            row(
                f"peak {key} (tick)",
                f"{fmt(loom_peak[1])} ({loom_peak[0]})",
                f"{fmt(v2_peak[1])} ({v2_peak[0]})",
            )
        )
    print(
        row(
            "ramp ticks with drive > 0",
            f"{sum(1 for value in ramp_drives if value > 0.0)}/{len(ramp_drives)}",
            "n/a (absolute)",
        )
    )
    print(
        row(
            "drive after ramp: first / last",
            f"{fmt(hold_drives[0])} / {fmt(hold_drives[-1])}",
            fmt(v2_run[-1]["threat_front_drive"]),
        )
    )


def _report_recede(loom: Driver, escape_v2: Driver, payload: dict[str, Any]) -> None:
    frames = sequence_recede()
    loom_run = loom.run(frames)
    v2_run = escape_v2.run(frames)
    payload["recede"] = {"loom": loom_run, "escape_v2": v2_run}
    loom_drives = series(loom_run, "threat_front_drive")
    v2_drives = series(v2_run, "threat_front_drive")
    heading(f"3. recede (blocked -> 0 over {RAMP_TICKS} ticks)")
    print(row("max threat_front drive", fmt(max(loom_drives)), fmt(max(v2_drives))))
    print(
        row(
            "ticks with drive > 0",
            f"{sum(1 for value in loom_drives if value > 0.0)}/{len(loom_drives)}",
            f"{sum(1 for value in v2_drives if value > 0.0)}/{len(v2_drives)}",
        )
    )
    print(row("peak jump", fmt(peak(loom_run, "jump")[1]), fmt(peak(v2_run, "jump")[1])))
    print(
        row(
            "peak vertical",
            fmt(peak(loom_run, "vertical")[1]),
            fmt(peak(v2_run, "vertical")[1]),
        )
    )


def _report_touch(loom: Driver, escape_v2: Driver, payload: dict[str, Any]) -> None:
    frames = sequence_touch()
    loom_run = loom.run(frames)
    v2_run = escape_v2.run(frames)
    payload["touch"] = {"loom": loom_run, "escape_v2": v2_run}
    heading("4. touch onset, four held ticks, release")
    for left, right in zip(loom_run, v2_run, strict=True):
        print(
            row(
                f"tick {int(left['tick'])} touch / bristle drive",
                f"{fmt(left['touch_drive'])} / {fmt(left['bristle_drive'])}",
                f"{fmt(right['touch_drive'])} / {fmt(right['bristle_drive'])}",
            )
        )
    loom_peak = peak(loom_run, "vertical")
    v2_peak = peak(v2_run, "vertical")
    print(
        row(
            "peak vertical (tick)",
            f"{fmt(loom_peak[1])} ({loom_peak[0]})",
            f"{fmt(v2_peak[1])} ({v2_peak[0]})",
        )
    )
    touch_equal = all(
        left["touch_drive"] == right["touch_drive"]
        and left["bristle_drive"] == right["bristle_drive"]
        for left, right in zip(loom_run, v2_run, strict=True)
    )
    motors_equal = all(
        all(left[axis] == right[axis] for axis in MOTOR_AXES) and left["jump"] == right["jump"]
        for left, right in zip(loom_run, v2_run, strict=True)
    )
    print(row("touch channel bitwise equal", str(touch_equal)))
    print(row("applied motors bitwise equal", str(motors_equal)))


def _arc_values(stimulus: Records, control: Records) -> dict[str, float]:
    window = ARC_SUMMARY_WINDOW_TICKS
    return {
        "dnp01_first": abs(stimulus[0]["dnp01_pair_mean"]),
        "dnp01_first_control": abs(control[0]["dnp01_pair_mean"]),
        "ttmn_first": abs(stimulus[0]["ttmn_pool_mean"]),
        "ttmn_first_control": abs(control[0]["ttmn_pool_mean"]),
        "dnp01_window": abs(window_mean(stimulus, "dnp01_pair_mean", window)),
        "dnp01_window_control": abs(window_mean(control, "dnp01_pair_mean", window)),
        "ttmn_window": abs(window_mean(stimulus, "ttmn_pool_mean", window)),
        "ttmn_window_control": abs(window_mean(control, "ttmn_pool_mean", window)),
    }


def _cell(values: dict[str, float], key: str) -> str:
    return f"{fmt(values[key])} ({fmt(ratio(values[key], values[f'{key}_control']))}x)"


#: Which probe series is compared against which committed receipt condition.
#: The receipt's escape-arc hold used the escape-v1 CALIBRATION, but both
#: quantities below are PRE-gain and PRE-scale pool means and this condition
#: has no collisions, so the escape-v1 and escape-v2 sensory drives -- and
#: therefore the neural trajectories -- are identical. A bitwise match is the
#: expectation, not a coincidence.
_RECEIPT_SERIES_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("escape_v2", "blocked_front", "ttmn_pool_mean"),
    ("escape_v2", "blocked_front", "dnp01_pair_mean"),
    ("escape_v2_control", "baseline_all_open", "ttmn_pool_mean"),
    ("escape_v2_control", "baseline_all_open", "dnp01_pair_mean"),
)


def compare_against_receipt(
    payload_arc: dict[str, Any], receipt_path: Path
) -> list[tuple[str, str, bool, float]]:
    """Compare the probe's escape-v2 arc series to the committed receipt's.

    Returns ``(condition, series, bitwise_equal, max_abs_diff)`` per pair. The
    receipt stores exact float hexes, so the comparison is over the same
    values the receipt hashed, not over re-rounded decimals.
    """

    receipt = json.loads(receipt_path.read_text())
    series_by_condition = receipt["conditions"]["series"]
    results: list[tuple[str, str, bool, float]] = []
    for lane, condition, name in _RECEIPT_SERIES_PAIRS:
        reference = [
            float.fromhex(value) for value in series_by_condition[condition][name]["float_hex"]
        ]
        measured = [record[name] for record in payload_arc[lane]]
        if len(reference) != len(measured):
            raise LoomProbeError(
                f"receipt {condition}/{name} has {len(reference)} ticks, the probe "
                f"measured {len(measured)}; the two holds are not comparable"
            )
        results.append(
            (
                condition,
                name,
                all(left == right for left, right in zip(reference, measured, strict=True)),
                max(abs(left - right) for left, right in zip(reference, measured, strict=True)),
            )
        )
    return results


def _report_arc(
    loom: Driver, escape_v2: Driver, payload: dict[str, Any], receipt_path: Path | None
) -> None:
    frames = sequence_escape_arc()
    control_frames = sequence_all_open(len(frames))
    loom_run = loom.run(frames)
    loom_control = loom.run(control_frames)
    v2_run = escape_v2.run(frames)
    v2_control = escape_v2.run(control_frames)
    payload["escape_arc"] = {
        "loom": loom_run,
        "loom_control": loom_control,
        "escape_v2": v2_run,
        "escape_v2_control": v2_control,
    }
    loom_values = _arc_values(loom_run, loom_control)
    v2_values = _arc_values(v2_run, v2_control)
    heading(f"5. ESCAPE-ARC-01 blocked frontal ray (fresh reset, {ARC_HOLD_TICKS}-tick hold)")
    for label, key in (
        ("DNp01 |mean| tick 0 (x control)", "dnp01_first"),
        ("TTMn  |mean| tick 0 (x control)", "ttmn_first"),
        ("DNp01 |mean| last-10 (x control)", "dnp01_window"),
        ("TTMn  |mean| last-10 (x control)", "ttmn_window"),
    ):
        print(row(label, _cell(loom_values, key), _cell(v2_values, key)))
    for axis in ("vertical", "forward"):
        print(
            row(
                f"{axis} tick 0 / last",
                f"{loom_run[0][axis]:+.6f} / {loom_run[-1][axis]:+.6f}",
                f"{v2_run[0][axis]:+.6f} / {v2_run[-1][axis]:+.6f}",
            )
        )
    print(
        row(
            "jump tick 0 / last",
            f"{fmt(loom_run[0]['jump'])} / {fmt(loom_run[-1]['jump'])}",
            f"{fmt(v2_run[0]['jump'])} / {fmt(v2_run[-1]['jump'])}",
        )
    )
    if receipt_path is None or not receipt_path.is_file():
        print(row("receipt cross-check", "skipped (receipt absent)"))
        return
    # The receipt's TTMn event contrast divides the PEAK of the per-tick
    # |pool mean| by the all-open summary-window baseline; the table above
    # divides the last-10 MEAN by the same baseline. Different numerators, so
    # the two ratios are not the same quantity -- what makes this probe the
    # receipt's measurement is that the underlying series match bitwise.
    print(row("", "series", "bitwise / max|diff|"))
    for condition, name, exact, largest in compare_against_receipt(
        payload["escape_arc"], receipt_path
    ):
        print(row(f"  vs receipt {condition}", name, f"{exact} / {largest!r}"))
    peak = max(abs(record["ttmn_pool_mean"]) for record in v2_run)
    baseline = abs(window_mean(v2_control, "ttmn_pool_mean", ARC_SUMMARY_WINDOW_TICKS))
    print(
        row(
            "escape-v2 TTMn contrast, peak / last-10",
            f"{peak / baseline:.6f}",
            f"{v2_values['ttmn_window'] / baseline:.6f}",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loom-version", type=int, choices=(1, 2), default=1)
    parser.add_argument("--artifact-directory", type=Path, default=DEFAULT_ARTIFACT_DIRECTORY)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--body-annotations", type=Path, default=DEFAULT_BODY_ANNOTATIONS)
    parser.add_argument("--loom-calibration", type=Path, default=DEFAULT_LOOM_CALIBRATION)
    parser.add_argument("--escape-v2-calibration", type=Path, default=DEFAULT_ESCAPE_V2_CALIBRATION)
    parser.add_argument(
        "--escape-arc-receipt",
        type=Path,
        default=DEFAULT_ESCAPE_ARC_RECEIPT,
        help=(
            "committed ESCAPE-ARC-01 evidence whose blocked_front and "
            "baseline_all_open series condition 5 cross-checks bitwise"
        ),
    )
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--standing-hold-ticks",
        type=int,
        default=STANDING_HOLD_TICKS,
        help=(
            "ticks the blocked frontal ray is held in condition 1 (default "
            f"{STANDING_HOLD_TICKS}, the declared probe length); a longer hold "
            "characterizes how the recurrent state relaxes after the drive has "
            "already decayed"
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="optional path for the full per-tick series (an artifact, never tracked)",
    )
    arguments = parser.parse_args(argv)
    if arguments.loom_version == 2:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.probe_entity_loom import run

        return run(arguments)

    dnp01 = dnp01_dense_indices(arguments.artifact_directory, arguments.body_annotations)
    loom = Driver(
        build_controller(
            artifact_directory=arguments.artifact_directory,
            annotations=arguments.annotations,
            body_annotations=arguments.body_annotations,
            calibration=arguments.loom_calibration,
            loom=True,
            threads=arguments.threads,
        ),
        dnp01,
    )
    escape_v2 = Driver(
        build_controller(
            artifact_directory=arguments.artifact_directory,
            annotations=arguments.annotations,
            body_annotations=arguments.body_annotations,
            calibration=arguments.escape_v2_calibration,
            loom=False,
            threads=arguments.threads,
        ),
        dnp01,
    )
    print(f"loom-v1   adapter  {loom.controller.identity.adapter_hash}")
    print(f"loom-v1   decoder  {loom.controller.identity.decoder_hash}")
    print(f"loom-v1   config   {loom.controller.configuration.identity_sha256}")
    print(f"loom-v1   dynamics {loom.controller.identity.dynamics_hash}")
    print(f"escape-v2 adapter  {escape_v2.controller.identity.adapter_hash}")
    print(f"escape-v2 decoder  {escape_v2.controller.identity.decoder_hash}")
    print(f"dnp01 dense indices (diagnostic-only) {dnp01}")
    print(f"blocked-front threat_front raw {BLOCKED_PROXIMITY / 2.0:.12f}")

    payload: dict[str, Any] = {}
    if arguments.standing_hold_ticks < 1:
        raise LoomProbeError("--standing-hold-ticks must be at least 1")
    _report_standing(loom, escape_v2, payload, arguments.standing_hold_ticks)
    _report_ramp(loom, escape_v2, payload)
    _report_recede(loom, escape_v2, payload)
    _report_touch(loom, escape_v2, payload)
    _report_arc(loom, escape_v2, payload, arguments.escape_arc_receipt)

    if arguments.json_out is not None:
        arguments.json_out.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_out.write_text(json.dumps(payload, indent=1, sort_keys=True))
        print(f"\nper-tick series: {arguments.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
