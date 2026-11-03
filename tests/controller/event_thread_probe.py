"""Bounded execution-thread comparison on the actual candidate; no Minecraft claim.

Each repetition builds a fresh controller and runs the same 180 supplied observations.
Timing excludes construction, warmup and JSON serialization. Complete command digests
must match across repetitions. Thread counts are execution settings, not neural changes.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from hashlib import sha256
from pathlib import Path

from neurocraft_controller.event_program_policy import EventProgramSelectingController
from neurocraft_controller.full_brain_controller import FullBrainConfiguration, FullBrainController
from neurocraft_protocol import Observation, canonical_json_bytes

from tests.controller.event_candidate_probe import source_files, stimulus

OBSERVATIONS = 180


def distribution(values):
    ordered = sorted(values)

    def q(fraction):
        index = (len(ordered) - 1) * fraction
        lower = int(index)
        return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (
            index - lower
        )

    return {
        "samples": len(values),
        "p50": q(0.5),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": max(values),
        "unit": "ms",
        "method": "linear (n-1) quantile",
    }


def run(repository: Path, counts: list[int]):
    sources = source_files()
    sources["thread_probe"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    results = []
    for threads in counts:
        start = time.perf_counter_ns()
        core = FullBrainController.from_artifact(
            run_id="candidate-thread-diagnostic",
            entity_id="synthetic-input",
            configuration=FullBrainConfiguration(kernel_threads=threads),
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
        )
        controller = EventProgramSelectingController(core)
        construction_ms = (time.perf_counter_ns() - start) / 1e6
        controller.warm_compute_path()
        timings = []
        digest = sha256()
        for tick in range(OBSERVATIONS):
            observation = Observation(
                core.identity,
                tick,
                tick,
                tick * 50_000_000,
                stimulus("near_land_contact_feed", tick),
            )
            started = time.perf_counter_ns()
            command = controller.step(observation, command_seq=tick)
            timings.append((time.perf_counter_ns() - started) / 1e6)
            digest.update(canonical_json_bytes(command.to_dict()))
        result = {
            "requested_threads": threads,
            "actual_threads": core.engine.kernel_identity.thread_count,
            "construction_ms": construction_ms,
            "step_ms": timings,
            "distribution": distribution(timings),
            "identity": core.identity.to_dict(),
            "policy_hash": controller.policy_hash,
            "commands_sha256": digest.hexdigest(),
        }
        results.append(result)
        print(
            json.dumps(
                {k: result[k] for k in ("actual_threads", "distribution", "commands_sha256")}
            ),
            flush=True,
        )
        del controller, core
        gc.collect()
    if len({row["commands_sha256"] for row in results}) != 1:
        raise RuntimeError("complete commands changed across execution thread counts")
    return {
        "schema": "neurocraft.event_thread_probe.v1",
        "diagnostic_only": True,
        "scope": (
            "sequential synthetic full-candidate compute; busy host; no live recording or TPS claim"
        ),
        "platform": platform.platform(),
        "stimulus_case": "near_land_contact_feed",
        "observations_per_repetition": OBSERVATIONS,
        "neural_updates_per_repetition": OBSERVATIONS * 5,
        "warmup_passes": 3,
        "source": sources,
        "results": results,
        "complete_command_digests_match": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", default="8,4,2,8")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    counts = [int(value) for value in args.threads.split(",")]
    if not counts or len(counts) > 6 or any(count not in (1, 2, 4, 8) for count in counts):
        parser.error("one to six repetitions of 1,2,4,8 threads required")
    document = run(args.repository.resolve(), counts)
    with args.output.open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
