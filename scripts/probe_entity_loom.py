"""Paired loom-v1/v2 probes with explicitly declared per-ray hit kinds.

Called by ``probe_loom_encoder.py --loom-version 2``. Offline measurements
only; the per-tick output is an artifact, never a historical receipt edit.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from neurocraft_controller.program_policy import ESCAPE_OFF, ESCAPE_ON, ESCAPE_URGENCY_SPAN
from neurocraft_protocol import SensorFrame, SensorFrameV2

from scripts.probe_loom_encoder import (
    Driver,
    build_controller,
    dnp01_dense_indices,
    sensors,
    sequence_approach_ramp,
    sequence_escape_arc,
    sequence_recede,
    sequence_standing_front,
    sequence_touch,
)


def typed(frames: list[SensorFrame], kind: int) -> list[SensorFrame]:
    """Assign the stated kind to every ray, including maximum-range hits."""

    return [SensorFrameV2.from_sensor_frame(frame, (kind,) * 6) for frame in frames]


def run(arguments: argparse.Namespace) -> int:
    calibration_v2 = Path(arguments.loom_calibration).with_name(
        "full-readout-calibration-loom-v2.json"
    )
    indices = dnp01_dense_indices(arguments.artifact_directory, arguments.body_annotations)
    drivers = [
        Driver(
            build_controller(
                artifact_directory=arguments.artifact_directory,
                annotations=arguments.annotations,
                body_annotations=arguments.body_annotations,
                calibration=arguments.loom_calibration if version == 1 else calibration_v2,
                loom=True,
                loom_version=version,
                threads=arguments.threads,
            ),
            indices,
        )
        for version in (1, 2)
    ]
    wall_frames = [sensors(front=max(1.5, 6.0 - tick * 0.088)) for tick in range(80)]
    crossing = [sensors()] + [sensors(front=4.0)] * 24
    conditions = {
        "standing_front_entity": (sequence_standing_front(), 2),
        "approach_ramp_entity": (sequence_approach_ramp(), 2),
        "recede_entity": (sequence_recede(), 2),
        "touch": (sequence_touch(), 0),
        "escape_arc_entity": (sequence_escape_arc(), 2),
        "wall_approach_0088": (wall_frames, 1),
        "two_ray_corner_0088": (
            [
                replace(
                    f,
                    ray_distances_m=(
                        f.ray_distances_m[0],
                        6.0,
                        f.ray_distances_m[0],
                        6.0,
                        6.0,
                        6.0,
                    ),
                )
                for f in wall_frames
            ],
            1,
        ),
        "three_ray_corner_0088": (
            [
                replace(
                    f,
                    ray_distances_m=(
                        f.ray_distances_m[0],
                        6.0,
                        f.ray_distances_m[0],
                        f.ray_distances_m[0],
                        6.0,
                        6.0,
                    ),
                )
                for f in wall_frames
            ],
            1,
        ),
        "crossing_entity_at_4m": (crossing, 2),
    }
    output: dict[str, Any] = {
        "schema": "neurocraft.entity_loom_probe.v1",
        "diagnostic_only": True,
        "identities": {
            f"loom_v{v}": d.controller.identity.to_dict()
            for v, d in zip((1, 2), drivers, strict=True)
        },
        "conditions": {},
    }
    try:
        for name, (frames, kind) in conditions.items():
            v1 = drivers[0].run(frames)
            v2 = drivers[1].run(typed(frames, kind))
            values_equal = [[value.hex() for value in row.values()] for row in v1] == [
                [value.hex() for value in row.values()] for row in v2
            ]
            row: dict[str, Any] = {
                "ray_hit_kinds": [kind] * 6,
                "loom_v1": v1,
                "loom_v2": v2,
                "all_reported_values_bitwise_equal": values_equal,
                "v1_peak_jump": max(item["jump"] for item in v1),
                "v2_peak_jump": max(item["jump"] for item in v2),
                "v2_max_threat_drive": max(item["threat_front_drive"] for item in v2),
            }
            output["conditions"][name] = row
            print(
                f"{name}: v1 peak jump={row['v1_peak_jump']:.17g}; "
                f"v2={row['v2_peak_jump']:.17g}; "
                f"v2 threat={row['v2_max_threat_drive']:.17g}; equal={values_equal}"
            )
            if name == "approach_ramp_entity":
                peak = row["v2_peak_jump"]
                derived = {
                    "escape_on": peak * 0.5,
                    "escape_off": peak * 0.25,
                    "escape_urgency_span": peak - peak * 0.5,
                }
                output["derived_policy_constants"] = derived
                output["policy_constants_reproduce"] = (
                    derived["escape_on"] == ESCAPE_ON
                    and derived["escape_off"] == ESCAPE_OFF
                    and derived["escape_urgency_span"] == ESCAPE_URGENCY_SPAN
                )
        # A standing entity steps aside to expose a slightly more distant wall.
        frames = [sensors(front=2.0)] * 8 + [sensors(front=2.1)] * 12
        v2_frames = typed(frames[:8], 2) + typed(frames[8:], 1)
        output["entity_steps_aside"] = {
            "ray_hit_kinds_per_tick": [[2] * 6] * 8 + [[1] * 6] * 12,
            "loom_v2": drivers[1].run(v2_frames),
        }
    finally:
        for driver in drivers:
            driver.controller.engine.end_run()
    walls = [row for row in output["conditions"].values() if row["ray_hit_kinds"][0] == 1]
    output["brief_hypotheses"] = {
        "single_front_v1_reproduces_escape_threshold": output["conditions"]["wall_approach_0088"][
            "v1_peak_jump"
        ]
        > ESCAPE_ON,
        "single_front_v2_jump_exactly_zero": output["conditions"]["wall_approach_0088"][
            "v2_peak_jump"
        ]
        == 0.0,
        "correction": (
            "zero gated threat does not imply zero neural state/readout; "
            "the original single-ray reproduction failed"
        ),
    }
    checks = {
        "entity_series_match": all(
            row["all_reported_values_bitwise_equal"]
            for row in output["conditions"].values()
            if row["ray_hit_kinds"][0] != 1
        ),
        "v2_wall_drive_exactly_zero": all(row["v2_max_threat_drive"] == 0.0 for row in walls),
        "v2_wall_below_escape_threshold": all(row["v2_peak_jump"] < ESCAPE_ON for row in walls),
        "policy_constants_reproduce": output["policy_constants_reproduce"],
    }
    output["checks"] = checks
    print(json.dumps(checks, sort_keys=True))
    if arguments.json_out is not None:
        arguments.json_out.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return 0 if all(checks.values()) else 1
