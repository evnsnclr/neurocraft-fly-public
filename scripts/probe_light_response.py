#!/usr/bin/env python3
"""Measure light transitions through the existing full event controller, offline.

All sensor histories are synthetic. No new encoder, policy, threshold, body program
or Minecraft execution is involved. Pre/post scores could be computed online;
paired differences are diagnostic counterfactuals and must not be used as live data.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

from neurocraft_controller.full_brain_controller import FullBrainConfiguration, FullBrainController
from neurocraft_protocol import EventCounter, EventSample, Observation, SensorFrameV3

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_gametest_demo import execution_source_origin
from scripts.probe_event_contexts import CALIBRATION, Schedule, background_frame
from scripts.probe_event_populations import ANNOTATIONS, ARTIFACT, ROOT

BASELINE_TICKS = 20
RESPONSE_TICKS = 16
SCORE_WINDOW_TICKS = 8
BASELINE_SAMPLES = 4
CASES = {
    "dark_constant": (0, 0, False),
    "bright_constant": (15, 15, False),
    "brighten": (0, 15, False),
    "dim": (15, 0, False),
    "ground_dark_constant": (0, 0, True),
    "ground_bright_constant": (15, 15, True),
    "ground_brighten": (0, 15, True),
    "ground_dim": (15, 0, True),
}


def sensor_frame(case: str, tick: int) -> SensorFrameV3:
    if case not in CASES or type(tick) is not int or not 0 <= tick < 36:
        raise ValueError("unknown light case or tick outside fixed schedule")
    before, after, ground = CASES[case]
    physical = background_frame("open_air_bright", tick, Schedule(recovery_ticks=40))
    physical = replace(
        physical,
        local_light=before if tick < BASELINE_TICKS else after,
        on_ground=ground,
        vertical_collision=ground,
        ray_distances_m=(6.0, 6.0, 6.0, 6.0, 6.0, 0.11 if ground else 6.0),
        ray_hit_kinds=(0, 0, 0, 0, 0, 1 if ground else 0),
    )
    zero = EventCounter(0, None)
    return SensorFrameV3.from_typed_frame(physical, EventSample(False, False, zero, zero, zero))


def reference_case(case: str) -> str:
    before, _, ground = CASES[case]
    return ("ground_" if ground else "") + ("dark_constant" if before == 0 else "bright_constant")


def summarize(values: list[float], reference: list[float]) -> dict[str, float]:
    total = BASELINE_TICKS + RESPONSE_TICKS
    if len(values) != total or len(reference) != total:
        raise ValueError("complete declared light timelines required")
    if any(not math.isfinite(v) for v in (*values, *reference)):
        raise ValueError("finite raw means required")
    baseline = (
        math.fsum(values[BASELINE_TICKS - BASELINE_SAMPLES : BASELINE_TICKS]) / BASELINE_SAMPLES
    )
    end = BASELINE_TICKS + SCORE_WINDOW_TICKS
    changes = [v - baseline for v in values[BASELINE_TICKS:end]]
    paired = [values[i] - reference[i] for i in range(BASELINE_TICKS, end)]
    return {
        "pre_event_four_observation_mean": baseline,
        "max_positive_pre_post_change": max(0.0, max(changes)),
        "max_negative_pre_post_change_magnitude": max(0.0, -min(changes)),
        "max_absolute_pre_post_change": max(abs(v) for v in changes),
        "max_absolute_paired_change_offline_only": max(abs(v) for v in paired),
        "signed_final_mean": values[-1],
    }


def code_sources() -> dict[str, dict[str, str]]:
    from scripts.probe_event_contexts import code_sources as background_sources

    result = background_sources()
    for name in (
        __name__,
        "scripts.prepare_gametest_demo",
        "neurocraft_controller.events_senses",
        "neurocraft_controller.steering_readout",
        "neurocraft_controller.flight_pools",
        "neurocraft_protocol.messages",
        "neurocraft_protocol.event_sensors",
        "neurocraft_protocol.sensor_contract",
    ):
        filename = importlib.import_module(name).__file__
        if filename is None:
            raise RuntimeError(f"missing source for {name}")
        path = Path(filename).resolve()
        result[name] = {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}
    return result


def measure_case(core: FullBrainController, case: str) -> dict[str, Any]:
    core.reset_session(session_epoch=core.identity.session_epoch + 1)
    rows = []
    for tick in range(BASELINE_TICKS + RESPONSE_TICKS):
        sensors = sensor_frame(case, tick)
        observation = Observation(core.identity, tick, tick, tick * 50_000_000, sensors)
        core.step(observation, command_seq=tick)
        event = core.last_event_readout
        step = core.last_evidence
        if event is None or step is None:
            raise RuntimeError("missing completed controller readout")
        left, right = core.steering_readout_means()
        means = {
            **dict(core.pool_readout_means()),
            "ttmn": core.jump_readout_mean(),
            "dna01_left": left,
            "dna01_right": right,
            "mn9": event.mn9_raw,
            "antenna": event.antenna_raw,
        }
        if core.control_kind == "no_input" and any(value != 0 for value in means.values()):
            raise RuntimeError("no-input raw activity is not zero after reset")
        rows.append(
            {
                "observation": observation.to_dict(),
                "raw_signed_means": means,
                "executed_stimulation_sha256": step.executed_stimulation_sha256,
                "encoded_stimulation_sha256": step.encoded_stimulation_sha256,
                "encoded_stimulation": dict(step.encoded_stimulation),
                "state_after_sha256": step.state_after_sha256,
                "neural_step_start": step.neural_step_start,
                "neural_step_end": step.neural_step_end,
            }
        )
    return {"case": case, "rows": rows, "reference_case": reference_case(case)}


def run_probe(threads: int) -> dict[str, Any]:
    sources = code_sources()
    origin = execution_source_origin(ROOT, sources)
    lanes = []
    for mode in ("faithful", "shuffled_weight", "no_input"):
        print(f"Measuring existing light input: {mode}", flush=True)
        core = FullBrainController.from_artifact(
            run_id="light-transition-diagnostic",
            entity_id="synthetic-light-input",
            configuration=FullBrainConfiguration(mode=mode, kernel_threads=threads),
            artifact_directory=ARTIFACT,
            annotations_path=ANNOTATIONS / "body-neurotransmitters-male-cns-v1.0.feather",
            body_annotations_path=ANNOTATIONS
            / "body-annotations-male-cns-v1.0-minconf-0.5.feather",
            calibration_path=CALIBRATION,
            embodied_senses=True,
            escape_reflex=True,
            escape_version=2,
            loom_encoder=True,
            loom_version=2,
            steering_readout=True,
            event_inputs=True,
            control_seed=7,
        )
        try:
            cases = {case: measure_case(core, case) for case in CASES}
            for result in cases.values():
                reference = cases[result["reference_case"]]
                result["responses"] = {
                    name: summarize(
                        [row["raw_signed_means"][name] for row in result["rows"]],
                        [row["raw_signed_means"][name] for row in reference["rows"]],
                    )
                    for name in result["rows"][0]["raw_signed_means"]
                }
            binding = core.event_encoder
            steering = core.steering_binding
            if binding is None or steering is None:
                raise RuntimeError("missing event or steering binding")
            lanes.append(
                {
                    "lane": mode,
                    "identity": core.identity.to_dict(),
                    "configuration": core.configuration.to_identity_material(),
                    "engine_configuration": asdict(core.engine.config),
                    "control_identity": None
                    if core.control_identity is None
                    else core.control_identity.material,
                    "event_population_binding": binding.populations.to_identity_material(),
                    "steering_binding": steering.to_identity_material(),
                    "steering_binding_sha256": steering.identity_sha256,
                    "cases": cases,
                }
            )
        finally:
            core.engine.end_run()
        del core
    if code_sources() != sources:
        raise RuntimeError("imported source changed during measurement")
    final_origin = execution_source_origin(ROOT, sources)
    if final_origin != origin:
        raise RuntimeError("execution source origin changed during measurement")
    return {
        "schema": "neurocraft.light_transition_probe.v1",
        "diagnostic_only": True,
        "scope": "synthetic light transitions through existing event controller; no new policy, "
        "threshold fitting, body execution, live behavior or biological claim",
        "git_head": final_origin["git_head"],
        "source_origin_at_start": origin,
        "source_origin_at_end": final_origin,
        "source_files": sources,
        "schedule": {
            "baseline_observations": BASELINE_TICKS,
            "response_observations": RESPONSE_TICKS,
            "pre_event_samples": BASELINE_SAMPLES,
            "score_window_ticks": SCORE_WINDOW_TICKS,
            "neural_updates_per_observation": 5,
            "reset_each_case": True,
            "controls_receive_identical_prescribed_sensor_histories": True,
        },
        "lanes": lanes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.threads <= 64 or args.output.exists():
        parser.error("threads must be in [1,64] and output must not exist")
    result = run_probe(args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {"output": str(args.output), "sha256": sha256(args.output.read_bytes()).hexdigest()}
        )
    )


if __name__ == "__main__":
    main()
