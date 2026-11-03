"""Explicit offline candidate diagnostic; never run by pytest or on a live world.

Run manually with --repository and --output. Uses actual FullBrainController.step
and default recurrent memory/history, with five real neural updates per supplied
observation. World events and contact transitions are synthetic declared inputs,
not observations of Minecraft or executions of its body programs. No throughput
claim is made, and thresholds are neither optimized nor changed by this probe.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import subprocess
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from neurocraft_controller.event_program_policy import EventProgramSelectingController
from neurocraft_controller.full_brain_controller import FullBrainConfiguration, FullBrainController
from neurocraft_protocol import EventCounter, EventSample, Observation, SensorFrameV3

from scripts.probe_event_contexts import Schedule, background_frame

CASES = (
    "contact_only",
    "food_contact",
    "near_land_contact_feed",
    "brush",
    "attack_attempt",
    "simultaneous",
    "near_loss",
    "near_fluid",
    "near_timeout",
    "near_escape",
)
SOURCE_MODULES = (
    __name__,
    "neurocraft_controller.full_brain_controller",
    "neurocraft_controller.events_senses",
    "neurocraft_controller.event_program_policy",
    "neurocraft_controller.named_event_evidence",
    "neurocraft_controller.program_policy",
    "neurocraft_controller.loom_senses_v2",
    "neurocraft_controller.loom_senses",
    "neurocraft_controller.embodied_senses",
    "neurocraft_neural.full_engine",
    "neurocraft_neural.full_controls",
    "neurocraft_neural.full_signs",
    "neurocraft_neural.spmv_kernel",
    "neurocraft_protocol.event_sensors",
    "neurocraft_protocol.sensor_contract",
    "scripts.probe_event_contexts",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files() -> dict[str, dict[str, str]]:
    result = {}
    for name in SOURCE_MODULES:
        filename = inspect.getsourcefile(importlib.import_module(name))
        if filename is None:
            raise RuntimeError(f"source is unavailable for {name}")
        path = Path(filename).resolve()
        result[name] = {"path": str(path), "sha256": sha(path)}
    return result


def stimulus(case: str, tick: int) -> SensorFrameV3:
    if case not in CASES:
        raise ValueError("unknown candidate diagnostic case")
    food_case = case in {
        "food_contact",
        "near_land_contact_feed",
        "simultaneous",
        "near_loss",
        "near_fluid",
        "near_timeout",
        "near_escape",
    }
    food_counter = int(food_case and tick >= 4)
    near = bool(food_counter) and not (case == "near_loss" and tick >= 10)
    contact = (case == "food_contact" and tick >= 4) or (
        case == "near_land_contact_feed" and tick >= 10
    )
    grounded = contact or (case == "contact_only" and tick >= 4)
    physical = background_frame("open_air_bright", tick, Schedule(recovery_ticks=210))
    if case == "near_escape":
        physical = background_frame(
            "entity_approach", max(0, tick - 4), Schedule(recovery_ticks=210)
        )
    physical = replace(
        physical,
        on_ground=grounded,
        vertical_collision=grounded,
        in_fluid=case == "near_fluid" and tick >= 10,
    )
    brush = int(case in {"brush", "simultaneous"} and tick >= 4)
    attack = int(case == "attack_attempt" and tick >= 4)
    events = EventSample(
        contact,
        near,
        EventCounter(food_counter, 4 if food_counter else None),
        EventCounter(brush, 4 if brush else None),
        EventCounter(attack, 4 if attack else None),
    )
    return SensorFrameV3.from_typed_frame(physical, events)


def run_case(controller: EventProgramSelectingController, case: str, epoch: int) -> dict[str, Any]:
    controller.reset_session(session_epoch=epoch)
    length = 207 if case == "near_timeout" else 37 if case == "near_escape" else 25
    rows = []
    for tick in range(length):
        observation = Observation(
            controller.identity, tick, tick, tick * 50_000_000, stimulus(case, tick)
        )
        command = controller.step(observation, command_seq=tick)
        evidence = controller.last_event_evidence
        inner = controller.controller.last_evidence
        assert evidence is not None and inner is not None
        rows.append(
            {
                "observation": observation.to_dict(),
                "command": command.to_dict(),
                "named_evidence": evidence.to_dict(),
                "encoded_stimulation": dict(inner.encoded_stimulation),
                "encoded_stimulation_sha256": inner.encoded_stimulation_sha256,
                "requested_before_control": (
                    None
                    if inner.encoder_stimulation_before_control is None
                    else dict(inner.encoder_stimulation_before_control)
                ),
            }
        )
    transitions = []
    previous = None
    for row in rows:
        command = row["command"]
        if command["program"] != previous:
            transitions.append(
                {
                    "tick": command["observation_tick"],
                    "program": command["program"],
                    "reason": row["named_evidence"]["reason"],
                }
            )
            previous = command["program"]
    return {
        "case": case,
        "identity": controller.identity.to_dict(),
        "observations_executed": length,
        "neural_updates_executed": length * 5,
        "issued_program_counts": dict(Counter(row["command"]["program"] for row in rows)),
        "issued_transitions": transitions,
        "peak_positive_scores": {
            name: max(row["named_evidence"][name]["score"] for row in rows)
            for name in ("mn9", "antenna")
        },
        "raw_mean_extrema": {
            name: {
                "min": min(row["named_evidence"][name]["raw"] for row in rows),
                "max": max(row["named_evidence"][name]["raw"] for row in rows),
            }
            for name in ("mn9", "antenna")
        },
        "rows": rows,
    }


def without_epoch(value: Any) -> Any:
    """Normalize the new epoch and its derived command hashes; full commands remain compared."""
    if isinstance(value, dict):
        return {
            key: without_epoch(item)
            for key, item in value.items()
            if key not in {"session_epoch", "command_sha256"}
        }
    if isinstance(value, list):
        return [without_epoch(item) for item in value]
    return value


def run_probe(repository: Path, threads: int) -> dict[str, Any]:
    sources = source_files()
    common = dict(
        run_id="event-candidate-diagnostic",
        entity_id="synthetic-world-input",
        artifact_directory=repository / "data/processed/male-cns-v1.0-superclass-v1",
        annotations_path=repository
        / "data/cache/male-cns-v1.0/body-neurotransmitters-male-cns-v1.0.feather",
        body_annotations_path=repository
        / "data/cache/male-cns-v1.0/body-annotations-male-cns-v1.0-minconf-0.5.feather",
        calibration_path=repository / "data/calibration/full-readout-calibration-loom-v2.json",
        embodied_senses=True,
        escape_reflex=True,
        escape_version=2,
        loom_encoder=True,
        loom_version=2,
        steering_readout=True,
        event_inputs=True,
        control_seed=7,
    )
    lanes = []
    original = None
    epoch = 0
    for mode in ("faithful", "shuffled_weight", "no_input"):
        core = FullBrainController.from_artifact(
            configuration=FullBrainConfiguration(mode=mode, kernel_threads=threads), **common
        )
        controller = EventProgramSelectingController(core)
        if original is None:
            original = controller
        cases = []
        for case in CASES:
            epoch += 1
            cases.append(run_case(controller, case, epoch))
        assert core.event_encoder is not None
        lanes.append(
            {
                "control_kind": core.control_kind,
                "control_identity": None
                if core.control_identity is None
                else core.control_identity.material,
                "configuration": core.configuration.to_identity_material(),
                "engine_configuration": asdict(core.engine.config),
                "event_encoder": core.event_encoder.to_identity_material(),
                "event_encoder_sha256": core.event_encoder.identity_sha256,
                "population_binding": core.event_encoder.populations.to_identity_material(),
                "population_binding_sha256": core.event_encoder.populations.identity_sha256,
                "policy": controller.policy.to_identity_material(),
                "policy_sha256": controller.policy_hash,
                "cases": cases,
            }
        )
    assert original is not None
    restored = run_case(original, "near_land_contact_feed", epoch + 1)
    before = lanes[0]["cases"][CASES.index("near_land_contact_feed")]
    exact = without_epoch(before["rows"]) == without_epoch(restored["rows"])
    if source_files() != sources:
        raise RuntimeError("imported sources changed during diagnostic")
    return {
        "schema": "neurocraft.event_program_candidate_probe.v1",
        "diagnostic_only": True,
        "claim": "synthetic observations through actual candidate controller; issued selections "
        "only; no Minecraft/body execution, no throughput or biological claim",
        "source": {
            "files": sources,
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
            ).strip(),
        },
        "schedule": {
            "baseline_observations": 4,
            "onset_tick": 4,
            "neural_updates_per_observation": 5,
            "histories": "real default recurrent and encoder state, reset each case",
            "inputs": "near/contact/fluid/entity timelines explicitly synthetic in rows",
        },
        "lanes": lanes,
        "original_restoration": {
            "full_rows_equal_excluding_epoch_and_derived_command_hashes": exact,
            "case": restored,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.threads <= 64:
        parser.error("threads must be in [1,64]")
    if args.output.exists():
        parser.error("output is immutable; choose a new path")
    result = run_probe(args.repository, args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha(args.output),
                "original_restored": result["original_restoration"][
                    "full_rows_equal_excluding_epoch_and_derived_command_hashes"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
