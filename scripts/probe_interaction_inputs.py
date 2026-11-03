#!/usr/bin/env python3
"""Full-network V4 input diagnostic, without a program policy or Minecraft body.

Actor categories are supplied engine metadata, not inferred neural categories.
The separate attack mapping is a declared tactile proxy, not injury or nociception.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

from neurocraft_controller.full_brain_controller import FullBrainConfiguration, FullBrainController
from neurocraft_protocol import EventCounter, EventSample, Observation, SensorFrameV4

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_gametest_demo import execution_source_origin
from scripts.probe_event_contexts import CALIBRATION
from scripts.probe_event_populations import ANNOTATIONS, ARTIFACT, ROOT
from scripts.probe_light_response import (
    BASELINE_TICKS,
    RESPONSE_TICKS,
    summarize,
)
from scripts.probe_light_response import (
    code_sources as light_sources,
)
from scripts.probe_light_response import (
    sensor_frame as light_frame,
)

CASES = (
    "bright_sham",
    "brush",
    "attack",
    "collision",
    "food",
    "player_approach",
    "mob_approach",
    "light_off",
    "dark_sham",
    "light_on",
)


def sensor_frame(case: str, tick: int) -> SensorFrameV4:
    if case not in CASES:
        raise ValueError("unknown interaction case")
    light_case = {"light_off": "dim", "dark_sham": "dark_constant", "light_on": "brighten"}.get(
        case, "bright_constant"
    )
    base = light_frame(light_case, tick)
    active = tick >= BASELINE_TICKS
    count = int(active)
    event = EventCounter(count, BASELINE_TICKS if active else None)
    zero = EventCounter(0, None)
    events = EventSample(
        case == "food" and active,
        case == "food" and active,
        event if case == "food" else zero,
        event if case == "brush" else zero,
        event if case == "attack" else zero,
    )
    actors = (0, 0, 0, 0, 0, 0)
    if case in {"collision", "food"} and active:
        base = replace(
            base,
            on_ground=True,
            vertical_collision=True,
            ray_distances_m=(6.0, 6.0, 6.0, 6.0, 6.0, 0.11),
            ray_hit_kinds=(0, 0, 0, 0, 0, 1),
        )
    if case in {"player_approach", "mob_approach"} and active:
        front = max(1.2, 6.0 - 0.8 * (tick - BASELINE_TICKS + 1))
        base = replace(
            base, ray_distances_m=(front, 6.0, 6.0, 6.0, 6.0, 6.0), ray_hit_kinds=(2, 0, 0, 0, 0, 0)
        )
        actors = (1 if case == "player_approach" else 2, 0, 0, 0, 0, 0)
    return SensorFrameV4.from_event_frame(replace(base, events=events), actors)


def sources() -> dict[str, dict[str, str]]:
    result = light_sources()
    for name in (__name__, "neurocraft_controller.interaction_senses"):
        filename = importlib.import_module(name).__file__
        if filename is None:
            raise RuntimeError("missing executed source")
        path = Path(filename).resolve()
        result[name] = {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}
    # A broad project source inventory, including dependencies omitted from the
    # earlier selected-module lists. Third-party runtime bytes are not inventoried.
    for path in sorted((ROOT / "packages").glob("*/src/**/*.py")):
        result[str(path.relative_to(ROOT))] = {
            "path": str(path),
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
    for path in (ROOT / "pyproject.toml", ROOT / "uv.lock"):
        result[str(path.relative_to(ROOT))] = {
            "path": str(path),
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
    return result


def measure_case(core: FullBrainController, case: str) -> dict[str, Any]:
    core.reset_session(session_epoch=core.identity.session_epoch + 1)
    rows = []
    zero_vector_sha256 = sha256(bytes(core.engine.view.node_count * 8)).hexdigest()
    for tick in range(BASELINE_TICKS + RESPONSE_TICKS):
        observation = Observation(
            core.identity, tick, tick, tick * 50_000_000, sensor_frame(case, tick)
        )
        core.step(observation, command_seq=tick)
        completed, step = core.last_event_readout, core.last_evidence
        if completed is None or step is None:
            raise RuntimeError("missing completed model evidence")
        left, right = core.steering_readout_means()
        means = {
            **dict(core.pool_readout_means()),
            "ttmn": core.jump_readout_mean(),
            "dna01_left": left,
            "dna01_right": right,
            "mn9": completed.mn9_raw,
            "antenna": completed.antenna_raw,
        }
        activity_sha256 = core.engine.current_state().activity_sha256
        if core.control_kind == "no_input" and (
            step.executed_stimulation_sha256 != zero_vector_sha256
            or activity_sha256 != zero_vector_sha256
            or any(value != 0 for value in means.values())
        ):
            raise RuntimeError("no-input delivered input or full activity is nonzero")
        rows.append(
            {
                "observation": observation.to_dict(),
                "raw_signed_means": means,
                "completed_pulse": completed.pulse.to_dict(),
                "encoded_stimulation": dict(step.encoded_stimulation),
                "requested_before_control": None
                if step.encoder_stimulation_before_control is None
                else dict(step.encoder_stimulation_before_control),
                "encoded_stimulation_sha256": step.encoded_stimulation_sha256,
                # The existing core exposes this field for controlled lanes.
                "executed_stimulation_sha256": step.executed_stimulation_sha256,
                "full_activity_sha256": activity_sha256,
                "state_after_sha256": step.state_after_sha256,
                "neural_step_start": step.neural_step_start,
                "neural_step_end": step.neural_step_end,
            }
        )
    return {"case": case, "rows": rows}


def run_probe(threads: int) -> dict[str, Any]:
    code = sources()
    origin = execution_source_origin(ROOT, code)
    lanes = []
    for mode in ("faithful", "shuffled_weight", "no_input"):
        print(f"Measuring V4 interaction inputs: {mode}", flush=True)
        core = FullBrainController.from_artifact(
            run_id="interaction-input-diagnostic",
            entity_id="synthetic-fly",
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
            event_inputs_version=2,
            control_seed=7,
        )
        try:
            cases = {case: measure_case(core, case) for case in CASES}
            for name, case in cases.items():
                reference = (
                    "collision"
                    if name == "food"
                    else "dark_sham"
                    if name in {"dark_sham", "light_on"}
                    else "bright_sham"
                )
                case["reference_case"] = reference
                case["responses"] = {
                    pool: summarize(
                        [r["raw_signed_means"][pool] for r in case["rows"]],
                        [r["raw_signed_means"][pool] for r in cases[reference]["rows"]],
                    )
                    for pool in case["rows"][0]["raw_signed_means"]
                }
            actor_states_equal = [
                r["state_after_sha256"] for r in cases["player_approach"]["rows"]
            ] == [r["state_after_sha256"] for r in cases["mob_approach"]["rows"]]
            if not actor_states_equal:
                raise RuntimeError("actor metadata unexpectedly changed neural input dynamics")
            actor_delivered_equal = [
                r["executed_stimulation_sha256"] for r in cases["player_approach"]["rows"]
            ] == [r["executed_stimulation_sha256"] for r in cases["mob_approach"]["rows"]]
            if not actor_delivered_equal:
                raise RuntimeError("actor metadata changed delivered controlled-lane inputs")
            encoder, steering = core.event_encoder, core.steering_binding
            if encoder is None or steering is None:
                raise RuntimeError("missing declared bindings")
            lanes.append(
                {
                    "lane": mode,
                    "identity": core.identity.to_dict(),
                    "configuration": core.configuration.to_identity_material(),
                    "engine_configuration": asdict(core.engine.config),
                    "kernel_identity": core.engine.kernel_identity.material,
                    "control_identity": None
                    if core.control_identity is None
                    else core.control_identity.material,
                    "event_encoder": encoder.to_identity_material(),
                    "event_population_binding": encoder.populations.to_identity_material(),
                    "steering_binding": steering.to_identity_material(),
                    "steering_binding_sha256": steering.identity_sha256,
                    "actor_class_does_not_change_neural_state": actor_states_equal,
                    "actor_controlled_lane_delivered_inputs_equal": (
                        actor_delivered_equal if mode != "faithful" else None
                    ),
                    "zero_float64_vector_sha256": sha256(
                        bytes(core.engine.view.node_count * 8)
                    ).hexdigest(),
                    "cases": cases,
                }
            )
        finally:
            core.engine.end_run()
        del core
    if sources() != code:
        raise RuntimeError("executed sources changed during measurement")
    final_origin = execution_source_origin(ROOT, code)
    if final_origin != origin:
        raise RuntimeError("execution source origin changed during measurement")
    return {
        "schema": "neurocraft.interaction_input_probe.v2",
        "diagnostic_only": True,
        "scope": (
            "prescribed observations through actual V4 core; "
            "no program policy, Minecraft behavior or biological claim"
        ),
        "git_head": final_origin["git_head"],
        "source_origin_at_start": origin,
        "source_origin_at_end": final_origin,
        "source_files": code,
        "source_inventory_scope": (
            "all package Python source files, selected executed diagnostic scripts, "
            "pyproject and lockfile; concrete kernel build material is per lane; "
            "third-party runtime bytes are not inventoried"
        ),
        "schedule": {
            "baseline_observations": BASELINE_TICKS,
            "response_observations": RESPONSE_TICKS,
            "neural_updates_per_observation": 5,
            "onset_tick": BASELINE_TICKS,
            "controls": "same prescribed sensor histories; one shuffle seed7",
            "scores": (
                "first8 post-onset observations minus previous4 mean; paired deltas offline only"
            ),
        },
        "lanes": lanes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.threads <= 64:
        parser.error("new output and threads in [1,64] required")
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
