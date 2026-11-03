#!/usr/bin/env python3
"""Benchmark the array-native full-graph backend across the Milestone 5 matrix.

Cases: ``DENSE-256``, the exact 14-node MaleCNS fixture, synthetic graphs at
1M/3M/6M edges under three degree profiles, and the real 25,582,938-edge
MaleCNS graph. One *interval* is one 50 ms game tick, which the v1 clock
contract fixes at five neural steps.

Every number this script writes is measured in this process. Nothing is
extrapolated, and the protocol actually used (including any scaled-down
interval count for the largest cases) is recorded in the receipt next to the
measurement so a reader can see what was traded.

Output is JSON under ``artifacts/benchmark/`` (gitignored). It is a
measurement record, not a committed receipt.

Examples::

    make spmv-build
    UV_CACHE_DIR=.uv-cache uv run --frozen python scripts/benchmark_full_engine.py \\
        --cases dense256,fixture,syn1m
    UV_CACHE_DIR=.uv-cache uv run --frozen python scripts/benchmark_full_engine.py \\
        --cases full --artifact-manifest-sha256 <sha256 of manifest.json>
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _package in ("connectome-data", "neural-runtime"):
    sys.path.insert(0, str(REPO_ROOT / "packages" / _package / "src"))

from neurocraft_connectome import load_malecns_fixture  # noqa: E402
from neurocraft_connectome.full_runtime import load_full_runtime_artifact  # noqa: E402
from neurocraft_neural.full_engine import ArrayNativeEngine  # noqa: E402
from neurocraft_neural.full_view import (  # noqa: E402
    FullRuntimeView,
    ViewProvenance,
    build_view,
    view_construction_sha256,
    view_from_full_runtime_graph,
    view_from_runtime_graph,
)
from neurocraft_neural.model import SignOverlay  # noqa: E402
from neurocraft_neural.spmv_kernel import (  # noqa: E402
    NumpyReduceatKernel,
    SpmvKernel,
    SpmvKernelError,
    kernel_source_sha256,
    load_c_kernel,
)
from neurocraft_neural.view_verification import VerificationPolicy  # noqa: E402

NEURAL_STEPS_PER_TICK = 5
INTERVAL_BUDGET_P99_NS = 35_000_000
INTERVAL_BUDGET_P99_9_NS = 45_000_000
FULL_ARTIFACT_DIRECTORY = REPO_ROOT / "data/processed/male-cns-v1.0-superclass-v1"
FIXTURE_PATH = REPO_ROOT / "data/fixtures/mcns-v1.0-flight-power-v1.json"
NODE_COUNT_FULL = 166_700


@dataclass(frozen=True, slots=True)
class CaseSpec:
    case_id: str
    node_count: int
    edge_count: int
    degree_profile: str
    warmup_intervals: int
    timed_intervals: int
    repetitions: int


def _percentile(samples: list[int], fraction: float) -> int:
    if not samples:
        return 0
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _distribution(samples: list[int]) -> dict[str, Any]:
    return {
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
    return hashlib.sha256(np.asarray(samples, dtype=np.int64).astype(">i8").tobytes()).hexdigest()


def _synthetic_view(
    *,
    node_count: int,
    edge_count: int,
    profile: str,
    seed: int,
) -> FullRuntimeView:
    """Synthetic CSR shaped like the artifact: sorted columns, empty rows, a max row."""

    rng = np.random.default_rng(seed)
    raw: Any
    if profile == "uniform":
        raw = np.full(node_count, edge_count / node_count)
    elif profile == "powerlaw":
        raw = rng.pareto(1.5, size=node_count) + 1.0
    else:
        raw = rng.lognormal(mean=np.log(98.0), sigma=1.0, size=node_count)
        tail = rng.random(node_count) < 0.005
        raw[tail] *= rng.uniform(3.0, 12.0, size=int(tail.sum()))
    raw = np.clip(raw, 0.0, 11_526.0)
    degrees = np.maximum(np.rint(raw * (edge_count / raw.sum())), 0.0).astype(np.int64)
    degrees[:370] = 0  # match the artifact's 370 in-degree-zero rows
    degrees[node_count // 2] = min(11_526, int(degrees.max()) + 11_526)
    total = int(degrees.sum())
    indptr = np.zeros(node_count + 1, dtype=np.int64)
    np.cumsum(degrees, out=indptr[1:])
    columns = rng.integers(0, node_count, size=total, dtype=np.int64)
    rows = np.repeat(np.arange(node_count, dtype=np.int64), degrees)
    columns = columns[np.lexsort((columns, rows))]
    counts = rng.integers(1, 2592, size=total, dtype=np.int64)
    signs = rng.choice(np.array([-1, 0, 1], dtype=np.int64), size=total)
    return build_view(
        indptr=indptr,
        pre_indices=columns.astype(np.uint32),
        raw_counts=counts,
        signs=signs,
        provenance=ViewProvenance(
            origin=f"synthetic_csr:{profile}:seed={seed}",
            graph_identity_sha256=hashlib.sha256(
                f"synthetic:{node_count}:{edge_count}:{profile}:{seed}".encode()
            ).hexdigest(),
        ),
    )


def _dense_view(node_count: int) -> FullRuntimeView:
    """Complete directed graph including self edges: the dense limit."""

    indptr = (np.arange(node_count + 1, dtype=np.int64)) * node_count
    columns = np.tile(np.arange(node_count, dtype=np.uint32), node_count)
    rng = np.random.default_rng(1234)
    counts = rng.integers(1, 2592, size=node_count * node_count, dtype=np.int64)
    signs = rng.choice(np.array([-1, 0, 1], dtype=np.int64), size=node_count * node_count)
    return build_view(
        indptr=indptr,
        pre_indices=columns,
        raw_counts=counts,
        signs=signs,
        provenance=ViewProvenance(
            origin=f"dense_complete_{node_count}",
            graph_identity_sha256=hashlib.sha256(f"dense:{node_count}".encode()).hexdigest(),
        ),
    )


def _fixture_view() -> FullRuntimeView:
    graph = load_malecns_fixture(FIXTURE_PATH)
    return view_from_runtime_graph(
        graph, sign_overlay=SignOverlay.coarse_presynaptic_consensus(graph)
    )


def _full_view(manifest_sha256: str, directory: Path) -> tuple[FullRuntimeView, dict[str, Any]]:
    graph = load_full_runtime_artifact(directory, expected_manifest_sha256=manifest_sha256)
    try:
        view = view_from_full_runtime_graph(graph)
        metadata = {
            "artifact_id": graph.manifest.artifact_id,
            "manifest_file_sha256": graph.manifest.manifest_file_sha256,
            "source_manifest_sha256": graph.manifest.source_manifest_sha256,
            "structural_graph_sha256": graph.manifest.structural_graph_sha256,
            "provenance_graph_sha256": graph.manifest.provenance_graph_sha256,
            "array_sha256": {spec.name: spec.sha256 for spec in graph.manifest.arrays},
            "sign_overlay": (
                "PLACEHOLDER +1 for every edge; not a transmitter-derived sign overlay "
                "and not Faithful Organism Mode dynamics"
            ),
        }
    finally:
        graph.close()
    return view, metadata


def _timed_case(
    view: FullRuntimeView,
    kernel: SpmvKernel,
    spec: CaseSpec,
    *,
    policy: VerificationPolicy,
) -> dict[str, Any]:
    engine = ArrayNativeEngine(view, kernel=kernel, verification_policy=policy)
    rng = np.random.default_rng(99)
    direct = rng.uniform(-0.2, 0.2, size=view.node_count)
    interval_samples: list[int] = []
    step_samples: list[int] = []

    first_touch_start = time.perf_counter_ns()
    engine.step(direct=direct)
    first_touch_ns = time.perf_counter_ns() - first_touch_start

    for _ in range(spec.warmup_intervals):
        for _ in range(NEURAL_STEPS_PER_TICK):
            engine.step(direct=direct)

    # Allocation guard runs in its own short pass: tracemalloc instruments every
    # allocation and would dominate the timings if left on during them.
    tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]
    for _ in range(50):
        engine.step(direct=direct)
    allocation_delta = tracemalloc.get_traced_memory()[0] - baseline
    tracemalloc.stop()

    gc.collect()
    gc.freeze()
    for _ in range(spec.timed_intervals):
        interval_start = time.perf_counter_ns()
        for _ in range(NEURAL_STEPS_PER_TICK):
            step_start = time.perf_counter_ns()
            engine.step(direct=direct)
            step_samples.append(time.perf_counter_ns() - step_start)
        interval_samples.append(time.perf_counter_ns() - interval_start)
    gc.unfreeze()

    # Declared run_end trigger. Without it the run stops mid-coverage-pass and
    # exits with uncovered records, which is not a passed run.
    engine.end_run()

    intervals = _distribution(interval_samples)
    return {
        "thread_count": kernel.thread_count,
        "kernel": kernel.name,
        "kernel_identity_sha256": kernel.identity.identity_sha256,
        "kernel_library_sha256": kernel.identity.material.get("library_sha256"),
        "kernel_load_note": kernel.identity.load_note,
        "dynamics_identity_sha256": engine.dynamics_identity_sha256,
        "first_touch_ns": first_touch_ns,
        "step": _distribution(step_samples),
        "interval": intervals,
        "interval_sample_sha256": _samples_digest(interval_samples),
        "interval_samples_ns": interval_samples,
        "post_warmup_alloc_delta_bytes": int(allocation_delta),
        "meets_p99_budget": intervals["p99_ns"] <= INTERVAL_BUDGET_P99_NS,
        "meets_p99_9_budget": intervals["p99_9_ns"] <= INTERVAL_BUDGET_P99_9_NS,
        "final_trace_sha256": engine.trace_sha256,
        "verified_through_step": engine.verified_through_step,
        "uncovered_steps_at_exit": engine.uncovered_steps,
        "run_is_covered": engine.run_is_covered,
        "coverage_slice_count": engine.coverage_slice_count,
        "verification_triggers_invoked": [
            {"trigger": trigger, "step": step} for trigger, step in engine.verification_events
        ],
    }


def _component_profile(
    view: FullRuntimeView,
    kernel: SpmvKernel,
    *,
    repeats: int = 20,
) -> dict[str, int]:
    """Per-stage budget breakdown, each stage timed on its own."""

    node_count = view.node_count
    rng = np.random.default_rng(7)
    x = rng.uniform(-1.0, 1.0, size=node_count)
    out = np.empty(node_count, dtype=np.float64)
    scratch = np.empty(node_count, dtype=np.float64)
    leak = 0.8187307530779818

    def _median(function: Any) -> int:
        timings: list[int] = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            function()
            timings.append(time.perf_counter_ns() - start)
        return sorted(timings)[len(timings) // 2]

    def _matvec() -> None:
        kernel.matvec(
            indptr=view.indptr,
            indices=view.indices,
            values=view.weights,
            x=x,
            out=out,
            empty_rows=view.empty_rows,
        )

    def _pointwise() -> None:
        np.multiply(out, 0.9, out=scratch)
        np.tanh(scratch, out=scratch)
        np.multiply(scratch, 1.0 - leak, out=scratch)
        np.multiply(x, leak, out=out)
        np.add(scratch, out, out=scratch)

    def _bounds() -> None:
        assert bool(np.isfinite(scratch).all())
        assert not bool((np.abs(scratch) > 1e9).any())

    def _digest() -> None:
        hashlib.sha256(scratch.astype(">f8").tobytes()).hexdigest()

    policy = VerificationPolicy()
    budget = max(1, policy.bytes_per_step(view.buffer_bytes))

    def _checkpoint_slice() -> None:
        with memoryview(view.weights).cast("B") as raw, raw[:budget] as block:
            hashlib.sha256(block).hexdigest()

    _matvec()
    return {
        "spmv_ns": _median(_matvec),
        "pointwise_update_ns": _median(_pointwise),
        "bounds_checks_ns": _median(_bounds),
        "activity_digest_ns": _median(_digest),
        "amortised_checkpoint_slice_ns": _median(_checkpoint_slice),
        "checkpoint_bytes_per_step": budget,
    }


def _thread_determinism(view: FullRuntimeView, threads: tuple[int, ...]) -> dict[str, Any]:
    """Run the same trace at several thread counts; the digests must be equal."""

    try:
        base = load_c_kernel(thread_count=1)
    except SpmvKernelError as exc:
        return {"status": "skipped", "reason": str(exc)}
    rng = np.random.default_rng(4321)
    direct = rng.uniform(-0.2, 0.2, size=view.node_count)
    traces: dict[str, str] = {}
    for count in threads:
        engine = ArrayNativeEngine(
            view,
            kernel=base.with_thread_count(count),
            verification_policy=VerificationPolicy(mode="load_only"),
        )
        for _ in range(50):
            engine.step(direct=direct)
        engine.end_run()
        traces[str(count)] = engine.trace_sha256
    return {
        "status": "measured",
        "trace_sha256_by_thread_count": traces,
        "bit_identical": len(set(traces.values())) == 1,
    }


def _fidelity_against_reference() -> dict[str, Any]:
    """Reference-engine comparison on the exact fixture, 400 steps."""

    graph = load_malecns_fixture(FIXTURE_PATH)
    signs = SignOverlay.coarse_presynaptic_consensus(graph)
    from neurocraft_neural.engine import NeuralEngine

    reference = NeuralEngine(graph, sign_overlay=signs)
    view = view_from_runtime_graph(graph, sign_overlay=signs)
    try:
        kernel: SpmvKernel = load_c_kernel(thread_count=4)
    except SpmvKernelError:
        kernel = NumpyReduceatKernel()
    fast = ArrayNativeEngine(view, kernel=kernel)
    from neurocraft_neural.model import Stimulation

    body_ids = [node.body_id for node in graph.nodes]
    stimulation = Stimulation.from_body_values(
        graph, "benchmark-fidelity", {body_ids[0]: 0.9, body_ids[1]: -0.4}
    )
    dense = np.zeros(len(body_ids), dtype=np.float64)
    for index, value in stimulation.assignments:
        dense[index] += value
    trajectory: list[dict[str, Any]] = []
    worst = 0.0
    for step in range(400):
        reference.step(stimulation=stimulation)
        fast.step(direct=dense, direct_provenance_sha256=stimulation.identity_sha256)
        divergence = float(
            np.max(np.abs(np.asarray(reference.current_state().activity) - fast.activity()))
        )
        worst = max(worst, divergence)
        if step + 1 in (1, 5, 25, 50, 100, 200, 300, 400):
            trajectory.append({"step": step + 1, "linf": divergence, "running_max_linf": worst})
    fast.end_run()
    return {
        "reference_backend": "python_single_thread_reference",
        "kernel": kernel.name,
        "weights_bit_identical": tuple(view.weights.tolist()) == reference.effective_weights,
        "max_state_divergence_linf": worst,
        "tolerance_linf": 1e-13,
        "trajectory": trajectory,
        "reference_trace_sha256": reference.trace_sha256,
        "array_trace_sha256": fast.trace_sha256,
        "array_run_is_covered": fast.run_is_covered,
    }


def _build_cases(selected: set[str], arguments: argparse.Namespace) -> list[tuple[CaseSpec, Any]]:
    cases: list[tuple[CaseSpec, Any]] = []
    warmup = arguments.warmup_intervals
    timed = arguments.timed_intervals
    if "dense256" in selected:
        cases.append(
            (
                CaseSpec("DENSE-256", 256, 65_536, "complete", warmup, timed, 1),
                lambda: _dense_view(256),
            )
        )
    if "fixture" in selected:
        cases.append((CaseSpec("FIXTURE-MCNS", 14, 22, "actual", warmup, timed, 1), _fixture_view))
    for label, edges in (("syn1m", 1_000_000), ("syn3m", 3_000_000), ("syn6m", 6_000_000)):
        if label not in selected:
            continue
        for profile in arguments.profiles.split(","):
            cases.append(
                (
                    CaseSpec(
                        f"SYN-{edges // 1_000_000}M-{profile}",
                        NODE_COUNT_FULL,
                        edges,
                        profile,
                        arguments.large_warmup_intervals,
                        arguments.large_timed_intervals,
                        1,
                    ),
                    (
                        lambda edges=edges, profile=profile: _synthetic_view(
                            node_count=NODE_COUNT_FULL,
                            edge_count=edges,
                            profile=profile,
                            seed=2026,
                        )
                    ),
                )
            )
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="dense256,fixture,syn1m")
    parser.add_argument("--profiles", default="measured")
    parser.add_argument("--threads", default="1,2,4,8,12")
    parser.add_argument("--warmup-intervals", type=int, default=1_000)
    parser.add_argument("--timed-intervals", type=int, default=10_000)
    parser.add_argument("--large-warmup-intervals", type=int, default=200)
    parser.add_argument("--large-timed-intervals", type=int, default=1_000)
    parser.add_argument("--full-warmup-intervals", type=int, default=100)
    parser.add_argument("--full-timed-intervals", type=int, default=400)
    parser.add_argument("--artifact-directory", type=Path, default=FULL_ARTIFACT_DIRECTORY)
    parser.add_argument("--artifact-manifest-sha256", default=None)
    parser.add_argument("--include-numpy-kernel", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts/benchmark/full-engine.json",
    )
    arguments = parser.parse_args(argv)
    selected = {item.strip() for item in arguments.cases.split(",") if item.strip()}
    thread_counts = tuple(int(item) for item in arguments.threads.split(",") if item.strip())

    results: list[dict[str, Any]] = []
    cases = _build_cases(selected, arguments)
    if "full" in selected:
        if arguments.artifact_manifest_sha256 is None:
            raise SystemExit("--artifact-manifest-sha256 is mandatory for the FULL-MCNS case")
        cases.append(
            (
                CaseSpec(
                    "FULL-MCNS",
                    NODE_COUNT_FULL,
                    25_582_938,
                    "actual",
                    arguments.full_warmup_intervals,
                    arguments.full_timed_intervals,
                    1,
                ),
                lambda: _full_view(
                    arguments.artifact_manifest_sha256, arguments.artifact_directory
                ),
            )
        )

    for spec, factory in cases:
        print(f"== {spec.case_id} ==", flush=True)
        build_start = time.perf_counter_ns()
        built = factory()
        view, artifact_metadata = built if isinstance(built, tuple) else (built, None)
        build_ns = time.perf_counter_ns() - build_start
        kernels: list[SpmvKernel] = []
        for count in thread_counts:
            try:
                kernels.append(load_c_kernel(thread_count=count))
            except SpmvKernelError as exc:
                print(f"   C kernel unavailable at {count} threads: {exc}", flush=True)
        if arguments.include_numpy_kernel or not kernels:
            kernels.append(NumpyReduceatKernel())
        measurements = []
        for kernel in kernels:
            measured = _timed_case(view, kernel, spec, policy=VerificationPolicy())
            interval = measured["interval"]
            print(
                f"   {kernel.name} t={kernel.thread_count:>2}  "
                f"step p50 {measured['step']['p50_ns'] / 1e6:8.3f} ms  "
                f"interval p50 {interval['p50_ns'] / 1e6:8.3f} ms  "
                f"p99 {interval['p99_ns'] / 1e6:8.3f} ms  "
                f"budget {'PASS' if measured['meets_p99_budget'] else 'FAIL'}",
                flush=True,
            )
            measurements.append(measured)
        best = min(kernels, key=lambda item: -item.thread_count)
        results.append(
            {
                "case": {
                    "id": spec.case_id,
                    "node_count": view.node_count,
                    "edge_count": view.edge_count,
                    "degree_profile": spec.degree_profile,
                    "empty_row_count": int(view.empty_rows.shape[0]),
                    "max_in_degree": view.max_in_degree,
                    "max_row_l1": view.max_row_l1,
                    "max_row_l1_excess_over_one": max(0.0, view.max_row_l1 - 1.0),
                    "warmup_intervals": spec.warmup_intervals,
                    "timed_intervals": spec.timed_intervals,
                    "scaled_down_protocol": spec.timed_intervals < 10_000,
                },
                "view": {
                    "runtime_view_sha256": view.view_sha256,
                    "view_bytes": view.buffer_bytes,
                    "indptr_dtype": str(view.indptr.dtype),
                    "indices_dtype": str(view.indices.dtype),
                    "weights_dtype": str(view.weights.dtype),
                    "provenance": view.provenance.origin,
                    "provenance_identity_sha256": view.provenance.identity_sha256,
                    "validation_sha256": view.validation.identity_sha256,
                    "max_observed_index": view.validation.max_observed_index,
                    "build_ns": build_ns,
                    "artifact": artifact_metadata,
                },
                "components": _component_profile(view, best),
                "measurements": measurements,
                "thread_determinism": _thread_determinism(view, thread_counts),
            }
        )
        del view
        gc.collect()

    document = {
        "schema": "neurocraft.benchmark.full-graph-backend.v1",
        "machine": {
            "architecture": platform.machine(),
            "os": platform.platform(),
            "python": sys.version,
            "logical_cpu_count": os.cpu_count() or 0,
            "peak_rss_bytes": _peak_rss_bytes(),
        },
        "implementation": {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "kernel_source_sha256": kernel_source_sha256(),
            "view_construction_sha256": view_construction_sha256(),
            "numpy_version": np.__version__,
            "uv_lock_sha256": hashlib.sha256((REPO_ROOT / "uv.lock").read_bytes()).hexdigest(),
            "argv": sys.argv[1:],
        },
        "clock": "time.perf_counter_ns",
        "neural_steps_per_tick": NEURAL_STEPS_PER_TICK,
        "interval_budget_p99_ns": INTERVAL_BUDGET_P99_NS,
        "interval_budget_p99_9_ns": INTERVAL_BUDGET_P99_9_NS,
        "verification_policy": {
            "mode": VerificationPolicy().mode,
            "full_coverage_period_steps": VerificationPolicy().full_coverage_period_steps,
            "identity_sha256": VerificationPolicy().identity_sha256,
            "unconditional_triggers": list(VerificationPolicy().unconditional_triggers),
            "detection_semantics": (
                "Each amortised slice is compared against its recorded digest on the step "
                "that reads it, so a persistent in-place mutation is detected within "
                "full_coverage_period_steps. A mutation reverted before its slice is next "
                "read is not detected at all."
            ),
        },
        "fidelity": _fidelity_against_reference(),
        "results": results,
        "evidence_boundary": (
            "Measured array-native backend latency and fidelity on this named machine under "
            "the declared verification policy. Not a G3-R real-time qualification, not a "
            "Minecraft closed-loop measurement, and not a G4 behavioural result. Synthetic "
            "cases carry no biological evidence label. Every measured step derives the "
            "stimulation digest from the executed vector and range-checks every CSR offset "
            "and index it dereferences; both costs are inside these numbers."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {arguments.output}")
    print(f"document sha256: {hashlib.sha256(arguments.output.read_bytes()).hexdigest()}")
    return 0


def _peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage) if sys.platform == "darwin" else int(usage) * 1024


if __name__ == "__main__":
    raise SystemExit(main())
