#!/usr/bin/env python3
"""Benchmark the end-to-end full-brain controller step at real artifact scale.

Stage 2 of the full-brain-live lane: measures what one game tick costs through
``FullBrainController.step`` -- observation validation, the pinned feature
encoding, the dense sensory placement, five array-native engine steps over the
166,700-neuron / 25,582,938-edge signed view, the wing-pool decode, and the
protocol ``Action`` build -- plus the one-time ``from_artifact`` construction
(verified load, sign overlay, view build, engine construction).

Every number written here is measured in this process; nothing is extrapolated.
The observation sequence is SHA-256-derived and fully deterministic (no
``random``), so the neural trajectory is reproducible run to run even though
the wall-clock percentiles are not. Wall-clock percentiles are measurements and
belong in the payload of a benchmark record; the payload is serialized
canonically so an identical payload is byte-identical.

Output is JSON under ``artifacts/benchmark/`` (gitignored). It is a measurement
record, not a committed receipt; the orchestrator drafts any receipt.

Claim boundary: measured controller latency on this named machine only. Not a
G3-R qualification, not a Minecraft closed-loop measurement, and not evidence
that the full graph has driven Minecraft (that run has not happened).
"""

from __future__ import annotations

import argparse
import gc
import os
import platform
import resource
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from neurocraft_controller.full_brain_controller import (
    FULL_MCNS_ARTIFACT_MANIFEST_SHA256,
    FullBrainConfiguration,
    FullBrainController,
)
from neurocraft_protocol import Observation, SensorFrame, canonical_json_bytes

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIRECTORY = REPO_ROOT / "data" / "processed" / "male-cns-v1.0-superclass-v1"
DEFAULT_ANNOTATIONS = (
    REPO_ROOT / "data" / "cache" / "male-cns-v1.0"
) / "body-neurotransmitters-male-cns-v1.0.feather"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "benchmark" / "full-controller.json"

OBSERVATION_SEED_PREFIX = "neurocraft-full-controller-benchmark:v1"
RAY_MAX_DISTANCE_M = 6.0
GAME_TICK_NS = 50_000_000


def _percentile(samples: list[int], fraction: float) -> int:
    if not samples:
        return 0
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _distribution(samples: list[int]) -> dict[str, Any]:
    return {
        "percentile_method": "nearest_rank_rounded_index",
        "count": len(samples),
        "min_ns": min(samples) if samples else 0,
        "p50_ns": _percentile(samples, 0.50),
        "p95_ns": _percentile(samples, 0.95),
        "p99_ns": _percentile(samples, 0.99),
        "p99_9_ns": _percentile(samples, 0.999),
        "max_ns": max(samples) if samples else 0,
        "mean_ns": int(sum(samples) / len(samples)) if samples else 0,
    }


def _samples_digest(samples: list[int]) -> str:
    return sha256(np.asarray(samples, dtype=np.int64).astype(">i8").tobytes()).hexdigest()


def synthetic_observation(controller: FullBrainController, tick: int) -> Observation:
    """One deterministic SHA-256-derived observation with varying rays.

    Each ray distance and the light level are read from independent bytes of
    ``sha256(f"{OBSERVATION_SEED_PREFIX}:{tick}")``, so consecutive ticks vary
    but any two runs derive the identical sequence.
    """

    digest = sha256(f"{OBSERVATION_SEED_PREFIX}:{tick}".encode()).digest()

    def ray(slot: int) -> float:
        raw = int.from_bytes(digest[2 * slot : 2 * slot + 2], "big")
        return RAY_MAX_DISTANCE_M * raw / 65535.0

    sensors = SensorFrame(
        position_m=(0.0, 66.0, 0.0),
        velocity_m_s=(0.0, 0.0, 0.0),
        yaw_rad=0.0,
        pitch_rad=0.0,
        on_ground=False,
        horizontal_collision=False,
        vertical_collision=False,
        in_fluid=False,
        local_light=digest[12] % 16,
        ray_max_distance_m=RAY_MAX_DISTANCE_M,
        ray_distances_m=(ray(0), ray(1), ray(2), ray(3), ray(4), ray(5)),
    )
    return Observation(
        identity=controller.identity,
        observation_seq=tick,
        game_tick=tick,
        sent_monotonic_ns=tick * GAME_TICK_NS,
        sensors=sensors,
    )


def run_benchmark(arguments: argparse.Namespace) -> dict[str, Any]:
    construction_start = time.perf_counter_ns()
    controller = FullBrainController.from_artifact(
        run_id="full-controller-benchmark",
        entity_id="benchmark-fly",
        configuration=FullBrainConfiguration(kernel_threads=arguments.threads),
        artifact_directory=arguments.artifact_directory,
        annotations_path=arguments.annotations,
        expected_manifest_sha256=arguments.artifact_manifest_sha256,
    )
    construction_ns = time.perf_counter_ns() - construction_start
    engine = controller.engine
    kernel_identity = engine.kernel_identity

    tick = 0
    for _ in range(arguments.warmup_ticks):
        controller.step(synthetic_observation(controller, tick), command_seq=tick)
        tick += 1

    non_neutral_actions = 0
    tick_samples: list[int] = []
    gc.collect()
    gc.freeze()
    for _ in range(arguments.ticks):
        observation = synthetic_observation(controller, tick)
        tick_start = time.perf_counter_ns()
        action = controller.step(observation, command_seq=tick)
        tick_samples.append(time.perf_counter_ns() - tick_start)
        if not action.motors.is_neutral:
            non_neutral_actions += 1
        tick += 1
    gc.unfreeze()

    # Declared run_end trigger: without it the engine exits mid-coverage-pass
    # and the run must not be reported as covered.
    engine.end_run()

    return {
        "schema": "neurocraft.benchmark.full-brain-controller.v1",
        "machine": {
            "architecture": platform.machine(),
            "os": platform.platform(),
            "python": sys.version,
            "logical_cpu_count": os.cpu_count() or 0,
            "peak_rss_bytes": _peak_rss_bytes(),
        },
        "implementation": {
            "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "uv_lock_sha256": sha256((REPO_ROOT / "uv.lock").read_bytes()).hexdigest(),
            "numpy_version": np.__version__,
            "argv": sys.argv[1:],
        },
        "clock": "time.perf_counter_ns",
        "artifact": {
            "directory": str(arguments.artifact_directory),
            "manifest_sha256_pin": arguments.artifact_manifest_sha256,
            "node_count": engine.view.node_count,
            "edge_count": engine.view.edge_count,
            "runtime_view_sha256": engine.view.view_sha256,
            "view_provenance": engine.view.provenance.origin,
            "view_provenance_identity_sha256": engine.view.provenance.identity_sha256,
        },
        "controller": {
            "profile": "full_mcns_wing_v1",
            "configuration_identity_sha256": controller.configuration.identity_sha256,
            "graph_hash": controller.identity.graph_hash,
            "adapter_hash": controller.identity.adapter_hash,
            "dynamics_hash": controller.identity.dynamics_hash,
            "decoder_hash": controller.identity.decoder_hash,
            "observation_schema_hash": controller.identity.observation_schema_hash,
        },
        "kernel": {
            "name": kernel_identity.kernel,
            "thread_count": engine.kernel.thread_count,
            "identity_sha256": kernel_identity.identity_sha256,
            "load_note": kernel_identity.load_note,
        },
        "protocol": {
            "warmup_ticks": arguments.warmup_ticks,
            "timed_ticks": arguments.ticks,
            "neural_steps_per_game_tick": controller.configuration.neural_steps_per_game_tick,
            "observation_derivation": (
                f"rays and light from sha256('{OBSERVATION_SEED_PREFIX}:<tick>') bytes; "
                "deterministic, no PRNG"
            ),
        },
        "construction": {
            "ns": construction_ns,
            "seconds": construction_ns / 1e9,
            "scope": (
                "verified artifact load + coarse-consensus sign overlay + view build "
                "+ engine construction + boundary binding"
            ),
        },
        "tick": _distribution(tick_samples),
        "tick_sample_sha256": _samples_digest(tick_samples),
        "tick_samples_ns": tick_samples,
        "actions": {
            "timed_actions": arguments.ticks,
            "non_neutral": non_neutral_actions,
        },
        "coverage": {
            "final_trace_sha256": engine.trace_sha256,
            "verified_through_step": engine.verified_through_step,
            "uncovered_steps_at_exit": engine.uncovered_steps,
            "run_is_covered": engine.run_is_covered,
        },
        "evidence_boundary": (
            "Measured end-to-end FullBrainController.step latency and construction time "
            "on this named machine, driven by a deterministic synthetic observation "
            "sequence. Not a G3-R qualification, not a Minecraft closed-loop "
            "measurement, and not evidence that the full graph has driven Minecraft. "
            "The sign overlay is the disclosed coarse presynaptic-consensus policy; the "
            "backend carries its own dynamics identity."
        ),
    }


def _peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage) if sys.platform == "darwin" else int(usage) * 1024


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-directory", type=Path, default=DEFAULT_ARTIFACT_DIRECTORY)
    parser.add_argument(
        "--artifact-manifest-sha256",
        default=FULL_MCNS_ARTIFACT_MANIFEST_SHA256,
        help=(
            "external pin for the artifact manifest; defaults to the reviewed module "
            "constant recorded by RUNTIME-MCNS-01"
        ),
    )
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--warmup-ticks", type=int, default=50)
    parser.add_argument("--ticks", type=int, default=600)
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="kernel thread count; default defers to NEUROCRAFT_SPMV_THREADS or the CPU cap",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    if arguments.ticks < 600:
        raise SystemExit("--ticks must be at least 600 for a reportable distribution")

    document = run_benchmark(arguments)
    payload = canonical_json_bytes(document)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(payload)

    tick = document["tick"]
    kernel = document["kernel"]
    print(
        f"construction {document['construction']['seconds']:.3f}s | "
        f"tick p50 {tick['p50_ns'] / 1e6:.3f} ms  p95 {tick['p95_ns'] / 1e6:.3f} ms  "
        f"p99 {tick['p99_ns'] / 1e6:.3f} ms  max {tick['max_ns'] / 1e6:.3f} ms | "
        f"kernel {kernel['name']} t={kernel['thread_count']} | "
        f"non-neutral {document['actions']['non_neutral']}/{document['actions']['timed_actions']} "
        f"| covered {document['coverage']['run_is_covered']}"
    )
    print(f"wrote {arguments.output}")
    print(f"document sha256: {sha256(payload).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
