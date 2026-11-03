#!/usr/bin/env python3
"""Build the full-scale control views and smoke-test them against the faithful one.

Local-only: it needs the gitignored ``male-cns-v1.0-superclass-v1`` artifact and
the cached MaleCNS annotation tables. It builds the faithful view exactly as
``FullBrainController.from_artifact`` does, then builds the two control views
beside it with a declared seed, and reports:

* wall time and peak-RSS movement per build;
* the exact buffer accounting per view (which arrays are shared by reference
  with the faithful view and which are new), so the memory figure is derived,
  not estimated;
* the duplicate-pair and self-edge counts the presynaptic permutation produced,
  against the artifact's own ``raw_duplicate_directed_pairs`` and
  ``excluded_self_edge_rows`` manifest counts;
* the control identities;
* an ENGINE-LEVEL smoke: each view stepped from a fresh ``reset()`` for N ticks
  under (a) zero input and (b) a fixed synthetic drive held on the 126 LC4
  dense indices, reporting the L2 norm of the activity vector per tick and the
  TTMn pair mean.

What this is and is not
-----------------------
This is a **smoke test of the plumbing**: it shows that a control view builds,
that its buffers are what the identity says they are, and that an engine steps
it without faulting. The numbers below are dimensionless model activities under
an arbitrary declared drive on a hand-chosen population. They are **not**
evidence about a fly, not a comparison of behaviour, and not the proof moment;
the controlled comparison with the real encoder, the real bindings and the
calibrated decoder is lane S4's job. Nothing here is fitted and nothing here is
interpreted: whatever it measures is printed, including anything disappointing.

The LC4 drive is an engineered stimulus, not a looming stimulus: it is a
constant applied to the neurons annotated ``LC4``. Per-capita drive is not
retinotopy, and a constant is not an approach.

Usage::

    UV_CACHE_DIR=.uv-cache uv run --frozen python scripts/probe_full_controls.py \
        --seed 7 --ticks 30
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from neurocraft_neural.full_controls import (
    FullControlKind,
    FullControlSignPolicy,
    FullControlView,
    build_control_view,
)
from neurocraft_neural.full_engine import ArrayNativeEngine
from neurocraft_neural.full_signs import (
    _verified_graph_array,
    compute_full_sign_overlay,
)
from neurocraft_neural.full_view import FullRuntimeView, view_from_full_runtime_graph
from neurocraft_neural.spmv_kernel import load_kernel
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIRECTORY = REPO_ROOT / "data" / "processed" / "male-cns-v1.0-superclass-v1"
DEFAULT_ANNOTATIONS = (
    REPO_ROOT / "data" / "cache" / "male-cns-v1.0"
) / "body-neurotransmitters-male-cns-v1.0.feather"
DEFAULT_BODY_ANNOTATIONS = (
    REPO_ROOT / "data" / "cache" / "male-cns-v1.0"
) / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
DEFAULT_PERMUTATION_DIRECTORY = REPO_ROOT / "artifacts" / "controls"

#: The declared synthetic drive held on every LC4 dense index for the "driven"
#: smoke condition. One number, declared here, applied identically to every
#: view; it is an engineering stimulus with no biological interpretation.
LC4_DRIVE = 1.0

#: ``ViewProvenance.identity_sha256`` of the faithful view that
#: ``FullBrainController.from_artifact`` builds for
#: ``FULL_MCNS_ARTIFACT_MANIFEST_SHA256``. Checked below so this probe cannot
#: silently report a view the controller would never execute; the check is
#: skipped for any other artifact.
CONTROLLER_FAITHFUL_PROVENANCE_SHA256 = (
    "300cf2db2ee6d5fdb718c5ae2614f363cb6b24627cf61398295dd5ef3b9253ef"
)

#: ``ru_maxrss`` is bytes on Darwin and kibibytes on Linux.
_RSS_SCALE = 1.0 if platform.system() == "Darwin" else 1024.0


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _RSS_SCALE / (1024.0 * 1024.0)


@dataclass(frozen=True, slots=True)
class Timing:
    label: str
    seconds: float
    peak_rss_before_mb: float
    peak_rss_after_mb: float

    @property
    def peak_rss_delta_mb(self) -> float:
        return self.peak_rss_after_mb - self.peak_rss_before_mb

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "seconds": round(self.seconds, 3),
            "peak_rss_before_mb": round(self.peak_rss_before_mb, 1),
            "peak_rss_after_mb": round(self.peak_rss_after_mb, 1),
            "peak_rss_delta_mb": round(self.peak_rss_delta_mb, 1),
        }


@contextmanager
def timed(label: str, sink: list[Timing]) -> Iterator[None]:
    before = _peak_rss_mb()
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    sink.append(Timing(label, elapsed, before, _peak_rss_mb()))


def _buffer_accounting(view: FullRuntimeView, faithful: FullRuntimeView) -> dict[str, Any]:
    """Exactly which buffers are new bytes and which are the faithful view's."""

    faithful_buffers = dict(faithful.named_buffers())
    rows: dict[str, Any] = {}
    new_bytes = 0
    for name, array in view.named_buffers():
        shared = array is faithful_buffers[name]
        rows[name] = {
            "shared_by_reference": shared,
            "bytes": int(array.nbytes),
            "dtype": str(array.dtype),
            "writeable": bool(array.flags.writeable),
        }
        if not shared:
            new_bytes += int(array.nbytes)
    rows["new_bytes_over_the_faithful_view"] = new_bytes
    rows["new_mib_over_the_faithful_view"] = round(new_bytes / (1024.0 * 1024.0), 1)
    return rows


@dataclass(frozen=True, slots=True)
class Lane:
    """One view under test, with the label that must appear beside its numbers."""

    label: str
    view: FullRuntimeView
    control: FullControlView | None
    zero_input_only: bool = False


def _smoke(
    lane: Lane,
    *,
    drive_indices: NDArray[np.int64],
    ttmn_indices: NDArray[np.int64],
    ticks: int,
    drive: float,
    kernel_threads: int | None,
) -> dict[str, Any]:
    """Step one view for ``ticks`` under zero input and under the LC4 drive."""

    engine = ArrayNativeEngine(lane.view, kernel=load_kernel(thread_count=kernel_threads))
    node_count = lane.view.node_count
    results: dict[str, Any] = {
        "control_identity_sha256": (
            None if lane.control is None else lane.control.identity.identity_sha256
        ),
        "view_sha256": lane.view.view_sha256,
        "dynamics_identity_sha256": engine.dynamics_identity_sha256,
    }
    for condition in ("zero_input", "lc4_drive"):
        engine.reset()
        sensory: NDArray[np.float64] | None = None
        if condition == "lc4_drive" and not lane.zero_input_only:
            sensory = np.zeros(node_count, dtype=np.float64)
            sensory[drive_indices] = drive
        l2: list[float] = []
        ttmn: list[float] = []
        for _ in range(ticks):
            engine.step(sensory=sensory)
            activity = engine.activity()
            l2.append(float(np.linalg.norm(activity)))
            ttmn.append(float(activity[ttmn_indices].mean()))
        results[condition] = {
            "l2_first": l2[0],
            "l2_mid": l2[ticks // 2],
            "l2_final": l2[-1],
            "l2_max": max(l2),
            "ttmn_pair_mean_first": ttmn[0],
            "ttmn_pair_mean_final": ttmn[-1],
            "ttmn_pair_mean_max": max(ttmn, key=abs),
            "l2_trajectory": [round(value, 9) for value in l2],
            "step_index": engine.step_index,
            "state_sha256": engine.current_state().state_sha256,
        }
    return results


def _build_spread(
    faithful: FullRuntimeView,
    *,
    manifest_sha256: str,
    seed: int,
    raw_counts: NDArray[np.int64],
    edge_signs: NDArray[np.int64],
    node_signs: NDArray[np.int8],
    repeats: int,
) -> dict[str, list[float]]:
    """Time each control build ``repeats`` times so one sample is never quoted.

    A single timing on a laptop is not a measurement; the spread is what is
    reportable. The rewire's duplicate/self-edge census is included, because it
    is part of the build.
    """

    spread: dict[str, list[float]] = {}
    for kind in (FullControlKind.SHUFFLED_WEIGHT, FullControlKind.DEGREE_PRESERVING_RANDOM):
        samples: list[float] = []
        for _ in range(max(1, repeats)):
            start = time.perf_counter()
            build_control_view(
                faithful,
                kind=kind,
                seed=seed,
                artifact_manifest_sha256=manifest_sha256,
                raw_counts=raw_counts,
                edge_signs=edge_signs,
                node_signs=node_signs,
            )
            samples.append(round(time.perf_counter() - start, 3))
        spread[kind.value] = samples
    return spread


def _format_float(value: float) -> str:
    return f"{value: .6e}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_DIRECTORY)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--body-annotations", type=Path, default=DEFAULT_BODY_ANNOTATIONS)
    parser.add_argument("--seed", type=int, default=7, help="control seed (declared, default 7)")
    parser.add_argument("--ticks", type=int, default=30)
    parser.add_argument("--drive", type=float, default=LC4_DRIVE)
    parser.add_argument("--kernel-threads", type=int, default=None)
    parser.add_argument(
        "--build-repeats",
        type=int,
        default=3,
        help="rebuild each control this many times and report the spread (default 3)",
    )
    parser.add_argument(
        "--permutation-directory",
        type=Path,
        default=DEFAULT_PERMUTATION_DIRECTORY,
        help="where the permutations are persisted (gitignored)",
    )
    parser.add_argument("--no-persist", action="store_true")
    parser.add_argument("--json", type=Path, default=None, help="also write the record here")
    arguments = parser.parse_args()

    if not (arguments.artifact / "manifest.json").is_file():
        parser.error(
            f"the full artifact is not present at {arguments.artifact}; this probe is local-only"
        )
    for path in (arguments.annotations, arguments.body_annotations):
        if not path.is_file():
            parser.error(f"annotation table {path} is not present; this probe is local-only")

    import hashlib

    from neurocraft_connectome.full_runtime import load_full_runtime_artifact
    from neurocraft_controller.embodied_senses import load_embodied_population_binding
    from neurocraft_controller.escape_reflex import load_escape_ttmn_binding
    from neurocraft_controller.full_brain_controller import (
        FULL_MCNS_ARTIFACT_MANIFEST_SHA256,
        SIGN_OVERLAY_POLICY_PREFIX,
    )

    manifest_sha256 = hashlib.sha256(
        (arguments.artifact / "manifest.json").read_bytes()
    ).hexdigest()
    timings: list[Timing] = []

    with timed("load artifact + sign overlay + faithful view", timings):
        graph = load_full_runtime_artifact(
            arguments.artifact, expected_manifest_sha256=manifest_sha256
        )
        try:
            overlay = compute_full_sign_overlay(graph, arguments.annotations)
            faithful = view_from_full_runtime_graph(
                graph,
                signs=overlay.edge_signs,
                sign_provenance=f"{SIGN_OVERLAY_POLICY_PREFIX}:{overlay.identity_sha256}",
            )
            raw_counts = _verified_graph_array(graph, "edge_raw_counts", "<u8").astype(np.int64)
            populations = load_embodied_population_binding(graph, arguments.body_annotations)
            ttmn_pool = load_escape_ttmn_binding(graph, arguments.body_annotations, populations)
            manifest_counts = dict(graph.manifest.counts)
        finally:
            graph.close()

    # M3: the faithful view this probe builds must be the controller's, not a
    # lookalike. The sign-provenance string is the only free parameter of
    # `view_from_full_runtime_graph`, and it enters `ViewProvenance` (and so
    # every state digest), so it is imported from the controller rather than
    # retyped. The check is skipped on any other artifact, where the pinned
    # provenance identity would not apply.
    provenance_identity = faithful.provenance.identity_sha256
    provenance_checked = manifest_sha256 == FULL_MCNS_ARTIFACT_MANIFEST_SHA256
    if provenance_checked and provenance_identity != CONTROLLER_FAITHFUL_PROVENANCE_SHA256:
        raise SystemExit(
            "the faithful view's provenance identity is not the controller's "
            f"({provenance_identity}, expected {CONTROLLER_FAITHFUL_PROVENANCE_SHA256}); "
            "this probe would be reporting a lookalike view"
        )

    drive_indices = np.array(
        sorted(
            populations.population("lc4_left").dense_indices
            + populations.population("lc4_right").dense_indices
        ),
        dtype=np.int64,
    )
    ttmn_indices = np.array(
        sorted(member.dense_index for member in ttmn_pool.members), dtype=np.int64
    )
    ttmn_bodies = sorted(member.body_id for member in ttmn_pool.members)
    edge_signs = overlay.edge_signs.astype(np.int64)
    node_signs = overlay.node_signs

    persist = None if arguments.no_persist else arguments.permutation_directory

    with timed("build shuffled_weight control view", timings):
        shuffled = build_control_view(
            faithful,
            kind=FullControlKind.SHUFFLED_WEIGHT,
            seed=arguments.seed,
            artifact_manifest_sha256=manifest_sha256,
            raw_counts=raw_counts,
            edge_signs=edge_signs,
            persist_directory=persist,
        )

    with timed("build degree_preserving_random control view", timings):
        rewired = build_control_view(
            faithful,
            kind=FullControlKind.DEGREE_PRESERVING_RANDOM,
            seed=arguments.seed,
            artifact_manifest_sha256=manifest_sha256,
            raw_counts=raw_counts,
            node_signs=node_signs,
            sign_policy=FullControlSignPolicy.REDERIVED_FROM_PRESYNAPTIC,
            persist_directory=persist,
        )

    with timed("build no_input control identity", timings):
        no_input = build_control_view(
            faithful,
            kind=FullControlKind.NO_INPUT,
            seed=arguments.seed,
            artifact_manifest_sha256=manifest_sha256,
        )

    engine_probe = ArrayNativeEngine(
        faithful, kernel=load_kernel(thread_count=arguments.kernel_threads)
    )
    faithful_dynamics_hash = engine_probe.dynamics_identity_sha256
    del engine_probe

    lanes = [
        Lane("faithful (measured wiring)", faithful, None),
        Lane("shuffled_weight", shuffled.view, shuffled),
        Lane("degree_preserving_random", rewired.view, rewired),
        Lane("no_input (faithful wiring, input zeroed)", no_input.view, no_input, True),
    ]

    smoke: dict[str, Any] = {}
    for lane in lanes:
        with timed(f"engine smoke {lane.label}", timings):
            smoke[lane.label] = _smoke(
                lane,
                drive_indices=drive_indices,
                ttmn_indices=ttmn_indices,
                ticks=arguments.ticks,
                drive=arguments.drive,
                kernel_threads=arguments.kernel_threads,
            )

    build_spread = _build_spread(
        faithful,
        manifest_sha256=manifest_sha256,
        seed=arguments.seed,
        raw_counts=raw_counts,
        edge_signs=edge_signs,
        node_signs=node_signs,
        repeats=arguments.build_repeats,
    )

    statistics = rewired.rewire_statistics
    assert statistics is not None
    record: dict[str, Any] = {
        "schema": "neurocraft.probe.full_controls.v1",
        "artifact_manifest_sha256": manifest_sha256,
        "seed": arguments.seed,
        "ticks": arguments.ticks,
        "lc4_drive": arguments.drive,
        "node_count": faithful.node_count,
        "edge_count": faithful.edge_count,
        "faithful_view_sha256": faithful.view_sha256,
        "faithful_provenance_identity_sha256": provenance_identity,
        "faithful_provenance_matches_the_controller": provenance_checked,
        "faithful_dynamics_identity_sha256": faithful_dynamics_hash,
        "sign_overlay_identity_sha256": overlay.identity_sha256,
        "lc4_dense_index_count": int(drive_indices.size),
        "lc4_index_source": "load_embodied_population_binding lc4_left + lc4_right",
        "ttmn_body_ids": ttmn_bodies,
        "controls": {},
        "rewire_statistics": {
            "edge_count": statistics.edge_count,
            "distinct_pair_count": statistics.distinct_pair_count,
            "duplicate_edge_count": statistics.duplicate_edge_count,
            "duplicate_fraction": statistics.duplicate_edge_count / statistics.edge_count,
            "self_edge_count": statistics.self_edge_count,
            "source_self_edge_count": statistics.source_self_edge_count,
            "artifact_retained_duplicate_directed_pairs": manifest_counts.get(
                "retained_duplicate_directed_pairs"
            ),
            "artifact_retained_self_edge_rows": manifest_counts.get("retained_self_edge_rows"),
        },
        "timings": [timing.as_dict() for timing in timings],
        "build_spread_seconds": build_spread,
        "smoke": smoke,
    }
    for control in (shuffled, rewired, no_input):
        record["controls"][control.kind.value] = {
            "identity_sha256": control.identity.identity_sha256,
            "label": control.identity.label,
            "material": control.identity.material,
            "view_sha256": control.view.view_sha256,
            "provenance_identity_sha256": control.view.provenance.identity_sha256,
            "permutation_path": (
                str(control.permutation_path) if control.permutation_path is not None else None
            ),
            "buffers": _buffer_accounting(control.view, faithful),
            "zeroes_sensory_input": control.zeroes_sensory_input,
        }

    _print_report(record)
    if arguments.json is not None:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"\nrecord written to {arguments.json}")
    return 0


def _print_report(record: dict[str, Any]) -> None:
    print("=" * 78)
    print("FULL-SCALE CONTROL VIEWS -- BUILD AND ENGINE SMOKE (lane S1)")
    print("=" * 78)
    print(
        "Control views are ENGINEERING constructs. Nothing below is evidence about a fly:\n"
        "it reports what changed in the wiring and that an engine steps it. The LC4 drive\n"
        "is a declared constant on the neurons annotated LC4 -- not a looming stimulus,\n"
        "and per-capita drive is not retinotopy. The controlled comparison is lane S4."
    )
    print()
    print(f"artifact manifest sha256   {record['artifact_manifest_sha256']}")
    print(f"nodes / edges              {record['node_count']:,} / {record['edge_count']:,}")
    print(f"faithful view_sha256       {record['faithful_view_sha256']}")
    print(f"faithful dynamics_hash     {record['faithful_dynamics_identity_sha256']}")
    checked = record["faithful_provenance_matches_the_controller"]
    note = "== FullBrainController.from_artifact" if checked else "not checked: other artifact"
    print(f"faithful provenance id     {record['faithful_provenance_identity_sha256']}  ({note})")
    print(f"control seed               {record['seed']}")
    print(
        f"LC4 drive indices          {record['lc4_dense_index_count']} "
        f"({record['lc4_index_source']})"
    )
    print(f"TTMn bodies                {record['ttmn_body_ids']}")

    print("\n-- control identities " + "-" * 56)
    for kind, control in record["controls"].items():
        print(f"{kind}")
        print(f"    control_identity_sha256  {control['identity_sha256']}")
        print(f"    view_sha256              {control['view_sha256']}")
        print(f"    provenance_identity      {control['provenance_identity_sha256']}")
        print(f"    permutation file         {control['permutation_path']}")
        shared = [
            name
            for name, row in control["buffers"].items()
            if isinstance(row, dict) and row["shared_by_reference"]
        ]
        print(
            f"    new buffer bytes         "
            f"{control['buffers']['new_mib_over_the_faithful_view']} MiB "
            f"(shared by reference: {', '.join(shared) or 'none'})"
        )

    statistics = record["rewire_statistics"]
    print("\n-- degree_preserving_random: what the permutation produced " + "-" * 19)
    print(f"    edges                        {statistics['edge_count']:,}")
    print(f"    distinct (post, pre) pairs   {statistics['distinct_pair_count']:,}")
    print(
        f"    duplicate edge slots         {statistics['duplicate_edge_count']:,} "
        f"({statistics['duplicate_fraction'] * 100:.4f}% of edges)"
    )
    print(f"    self-edges after             {statistics['self_edge_count']:,}")
    print(f"    self-edges before            {statistics['source_self_edge_count']:,}")
    print(
        f"    artifact manifest says       retained_duplicate_directed_pairs="
        f"{statistics['artifact_retained_duplicate_directed_pairs']}, "
        f"retained_self_edge_rows={statistics['artifact_retained_self_edge_rows']}"
    )
    print(
        "    Duplicates and self-edges are ALLOWED under this control and counted here;\n"
        "    the fixture-scale rewire refused both. That is the declared difference."
    )

    print("\n-- build cost " + "-" * 64)
    print(
        "    peak RSS is ru_maxrss, a process high-water mark: it never falls, so the\n"
        "    delta column bounds what a step added and is not a steady-state figure.\n"
        "    The exact per-view buffer bytes are in the identities block above."
    )
    print(f"{'step':<52}{'seconds':>10}{'peak RSS MiB':>15}{'delta':>9}")
    for timing in record["timings"]:
        print(
            f"{timing['label']:<52}{timing['seconds']:>10.3f}"
            f"{timing['peak_rss_after_mb']:>15.1f}{timing['peak_rss_delta_mb']:>9.1f}"
        )

    print("\n-- build spread (repeated builds, one loaded artifact) " + "-" * 23)
    for kind, samples in record["build_spread_seconds"].items():
        ordered = sorted(samples)
        print(
            f"    {kind:<28} n={len(samples)}  "
            f"min {ordered[0]:.3f}  median {ordered[len(ordered) // 2]:.3f}  "
            f"max {ordered[-1]:.3f}  (seconds)"
        )
    print("    Each includes the mandatory faithful-view rebuild that verifies the inputs.")

    print("\n-- engine smoke: 30-tick activity, side by side " + "-" * 30)
    print(
        "    Units are dimensionless model activity. L2 is over all 166,700 neurons.\n"
        "    TTMn is the mean of the two jump motoneurons -- a raw pre-gain read, not\n"
        "    the decoded jump term and not a behaviour."
    )
    header = f"{'lane':<42}{'cond':<11}{'L2 t1':>13}{'L2 tN':>13}{'TTMn tN':>14}"
    print(header)
    print("-" * len(header))
    for lane, conditions in record["smoke"].items():
        for condition, values in conditions.items():
            if not isinstance(values, dict) or "l2_first" not in values:
                continue
            print(
                f"{lane:<42}{condition:<11}"
                f"{_format_float(values['l2_first']):>13}"
                f"{_format_float(values['l2_final']):>13}"
                f"{_format_float(values['ttmn_pair_mean_final']):>14}"
            )
    print(
        "\n    no_input steps the FAITHFUL wiring with the sensory vector zeroed, so its\n"
        "    two rows are identical by construction and equal the faithful zero-input row."
    )


if __name__ == "__main__":
    raise SystemExit(main())
