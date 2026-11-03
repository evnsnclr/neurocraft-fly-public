#!/usr/bin/env python3
"""Measure the deterministic fixture controller without promoting scientific claims."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import resource
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

from neurocraft_connectome import load_malecns_fixture
from neurocraft_controller import ConnectomeController
from neurocraft_protocol import Observation, SensorFrame, canonical_json_bytes

ROOT = Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(sorted_values: list[int], percentile: float) -> int:
    if not sorted_values:
        raise ValueError("cannot summarize an empty sample")
    index = max(0, min(len(sorted_values) - 1, int(percentile * len(sorted_values) + 0.999) - 1))
    return sorted_values[index]


def _peak_rss_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return observed if sys.platform == "darwin" else observed * 1024


def _observation(controller: ConnectomeController, sequence: int) -> Observation:
    open_world = sequence % 20 < 10
    ray = 6.0 if open_world else 0.5
    return Observation(
        identity=controller.identity,
        observation_seq=sequence,
        game_tick=sequence,
        sent_monotonic_ns=sequence * 50_000_000,
        sensors=SensorFrame(
            position_m=(0.0, 64.0, 0.0),
            velocity_m_s=(0.0, 0.0, 0.0),
            yaw_rad=0.0,
            pitch_rad=0.0,
            on_ground=False,
            horizontal_collision=False,
            vertical_collision=False,
            in_fluid=False,
            local_light=15 if open_world else 2,
            ray_max_distance_m=6.0,
            ray_distances_m=(ray, 6.0, ray, ray, 6.0, 6.0),
        ),
    )


def _receipt(fixture: Path, warmup: int, iterations: int) -> dict[str, Any]:
    load_start = time.perf_counter_ns()
    graph = load_malecns_fixture(fixture)
    controller = ConnectomeController.create(
        graph,
        run_id="fixture-benchmark",
        entity_id="benchmark-fly",
    )
    load_ns = time.perf_counter_ns() - load_start

    for sequence in range(warmup):
        controller.step(_observation(controller, sequence), command_seq=sequence)

    durations: list[int] = []
    for offset in range(iterations):
        sequence = warmup + offset
        start = time.perf_counter_ns()
        controller.step(_observation(controller, sequence), command_seq=sequence)
        durations.append(time.perf_counter_ns() - start)
    ordered = sorted(durations)
    last = controller.last_evidence
    assert last is not None
    return {
        "artifacts": {
            "adapter_sha256": controller.identity.adapter_hash,
            "configuration_sha256": controller.configuration.identity_sha256,
            "control_overlay_sha256": controller.engine.control_identity_sha256,
            "decoder_sha256": controller.identity.decoder_hash,
            "delay_sha256": controller.delay_overlay.identity_sha256,
            "dynamics_sha256": controller.identity.dynamics_hash,
            "graph_sha256": controller.identity.graph_hash,
            "observation_schema_sha256": controller.identity.observation_schema_hash,
            "sign_sha256": controller.sign_overlay.identity_sha256,
        },
        "benchmark": {
            "clock": "time.perf_counter_ns",
            "fixture_load_ns": load_ns,
            "iterations": iterations,
            "mean_tick_ns": round(fmean(durations)),
            "neural_steps_per_tick": controller.configuration.neural_steps_per_game_tick,
            "p50_tick_ns": _percentile(ordered, 0.50),
            "p95_tick_ns": _percentile(ordered, 0.95),
            "p99_9_tick_ns": _percentile(ordered, 0.999),
            "p99_tick_ns": _percentile(ordered, 0.99),
            "sample_count": len(durations),
            "sample_sha256": hashlib.sha256(canonical_json_bytes(durations)).hexdigest(),
            "samples_ns": durations,
            "warmup_ticks": warmup,
        },
        "evidence_boundary": (
            "Measured fixture/controller latency on this named machine; not full-graph, "
            "Minecraft, biological-behavior, or cross-process performance."
        ),
        "final": {
            "action_sha256": last.action_sha256,
            "feature_values_sha256": last.feature_values_sha256,
            "neural_trace_sha256": controller.engine.trace_sha256,
            "state_sha256": last.state_after_sha256,
        },
        "graph": {
            "edge_count": graph.manifest.edge_count,
            "node_count": graph.manifest.node_count,
            "raw_count_sum": graph.manifest.raw_count_sum,
        },
        "implementation": {
            "argv_parameters": {
                "fixture": fixture.as_posix(),
                "iterations": iterations,
                "warmup": warmup,
            },
            "script_sha256": _sha256_file(Path(__file__)),
            "uv_lock_sha256": _sha256_file(ROOT / "uv.lock"),
        },
        "machine": {
            "architecture": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "peak_rss_bytes": _peak_rss_bytes(),
            "os": platform.platform(),
            "python": sys.version,
        },
        "schema": "neurocraft.benchmark.fixture-controller.v1",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("data/fixtures/mcns-v1.0-flight-power-v1.json"),
    )
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=2_000)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.warmup < 0 or args.iterations < 1:
        raise SystemExit("warmup must be non-negative and iterations must be positive")
    receipt = _receipt(args.fixture, args.warmup, args.iterations)
    encoded = canonical_json_bytes(receipt) + b"\n"
    if args.output is None:
        sys.stdout.buffer.write(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
