#!/usr/bin/env python3
"""Phase 3 readout census: what the shipped motor readout is, and what it could be.

The bounded question. ``docs/receipts/REACHABILITY-01.md`` closed one question
and opened another. It showed the unbounded 99.17% path-support figure is
degenerate — the path-support set is the graph's giant strongly connected
component — and that the surviving quantitative statement is a graph-hop
bound: 27.36% of retained nodes can lie on an injection-to-readout path of at
most five edges. This is not five neural updates: fresh input first qualifies
after 1 + 2*h updates for h edges under the baseline delay convention. That
historical figure was measured for one
specific readout: two named motor bodies. This analysis asks what those two
bodies are in the pinned annotations, what other readout populations the artifact
offers, and reports both the historical graph-hop bound and separate fresh-input
one-tick topological eligibility for each of them.

Method, and why it was chosen:

* Identity. Every array comes from ``load_full_runtime_artifact``, so the
  manifest, payload, structural, provenance and per-array SHA-256 checks all run
  before a number is computed. NumPy views are taken over the exact mappings the
  loader hashed rather than by reopening a path; the views are copied into owned
  arrays and the artifact is closed before the analysis runs.

* Integrity first, exactly as ``docs/receipts/OPTIC-CENSUS-01.md`` did. Retained
  annotation values are obtained by cross-referencing ``annotation_source_rows``
  into the pinned source table, never by re-deriving a retention rule, and five
  gates must pass before any figure exists: the pinned table still hashes to the
  manifest's declared source digest; the taken ``bodyId`` column is bitwise
  identical to ``body_ids.bin``; the per-superclass counter reconciles with
  ``retained_by_superclass`` across every key with zero mismatches; the source
  rows are unique and in range; and the retained edge weights sum to the
  manifest's ``retained_weight_sum``. A failure raises and no file is written.

* Reproduction, not re-implementation. The latency table reuses
  :mod:`scripts.reachability_graph` unchanged, so the shipped-configuration row
  must reproduce REACHABILITY-01's L=5 figure exactly. A deviation is treated as
  a blocking error rather than as a new result.

* Layers. :mod:`scripts.readout_annotations` owns the verified annotation join
  and the gates, :mod:`scripts.readout_census` turns annotation columns into
  report fragments, :mod:`scripts.readout_connectivity` owns everything needing
  the edge support, and this file binds them to the verified manifest and writes
  the document.

Claim discipline. This is a topology and annotation census and nothing else. The
word "causal" is reserved by ``docs/SCIENTIFIC_CLAIMS.md`` for the G3 gate and is
not used here. Reachability **bounds participation and never demonstrates it**: a
path may carry vanishing, cancelling or sign-inverted influence under the
assigned leaky-rate dynamics, which this analysis does not execute. No behaviour
is assigned to any neuron. Where a cell type has a function reported in the
literature, that reading is labelled literature-inferred and is carried in an
unhashed sidecar block, outside the hashed measurement body.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

# ``python scripts/analyze_readout.py`` places ``scripts/`` rather than the
# repository root on sys.path. Add only this script's resolved repository root so
# the sibling analysis modules have the same import identity under direct CLI and
# test use; this is the convention analyze_reachability.py already follows.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurocraft_connectome import load_full_runtime_artifact
from neurocraft_connectome.full_runtime import FullRuntimeManifest
from neurocraft_connectome.model import canonical_json_bytes
from neurocraft_controller.config import ControllerConfiguration

from scripts.reachability_graph import IntArray, graph_from_full_runtime, mapped_numpy
from scripts.readout_annotations import (
    IntegrityGate,
    RetainedAnnotations,
    annotation_gates,
    gates_report,
    require_all_gates,
    take_retained_annotations,
)
from scripts.readout_census import (
    boundary_identity_report,
    descending_census,
    fixture_named_descending,
    readout_population_report,
    vnc_motor_census,
)
from scripts.readout_connectivity import (
    descending_to_motor_report,
    direct_contacts,
    latency_comparison,
    readout_sets,
    weight_sum_gate,
)

SCHEMA = "neurocraft.analysis.readout.v1"
DEFAULT_ARTIFACT_DIR = Path("data/processed/male-cns-v1.0-superclass-v1")
DEFAULT_ANNOTATIONS = Path(
    "data/cache/male-cns-v1.0/body-annotations-male-cns-v1.0-minconf-0.5.feather"
)
TRACKED_MANIFEST_SHA256 = "b33bdda8b1a9abfc6b23f7346197442546f7a2aec0a03fc7c07fbdfe4f67c98c"
DEFAULT_OUTPUT = Path("artifacts/analysis/readout-census.json")
NEURAL_STEPS_PER_GAME_TICK = ControllerConfiguration().neural_steps_per_game_tick

#: The figures ``docs/receipts/REACHABILITY-01.md`` published for the shipped
#: two-body readout. The shipped row of this analysis's latency table must
#: reproduce them exactly or the run is refused.
REACHABILITY_01: dict[str, Any] = {
    "receipt": "docs/receipts/REACHABILITY-01.md",
    "one_game_tick_path_support_nodes": 45610,
    "unbounded_path_support_nodes": 165314,
    "shortest_injection_to_readout_path": 2,
    "node_count": 166700,
}

#: Function readings taken from the cell-biology literature rather than measured
#: here. Kept OUT of the hashed analysis body on purpose: a hashed body must
#: contain only what this run measured from the pinned bytes.
LITERATURE_INFERRED_NOTES: dict[str, Any] = {
    "status": "literature-inferred, not measured by this analysis",
    "scope": (
        "these readings are context for the annotation strings the hashed body"
        " records. They are not evidence, they were not produced by this run, and no"
        " figure in the hashed body depends on them. This analysis assigns no"
        " behaviour to any neuron"
    ),
    "cell_types": {
        "hg1 MN": "reported in the literature as a wing steering muscle motor neuron",
        "SNpp10": "reported as a campaniform sensillum afferent (mechanoreceptive)",
        "JO-EV3": (
            "reported as a Johnston's organ afferent subtype associated with wind and"
            " gravity mechanosensation"
        ),
        "TTMn": "reported as the tergotrochanteral muscle motor neuron",
        "DNp01": "reported as the giant fibre descending neuron",
    },
    "subclass_codes": {
        "wm": "wing motor",
        "nm": "neck motor",
        "hm": "haltere motor",
        "fl": "front leg motor",
        "ml": "middle leg motor",
        "hl": "hind leg motor",
        "ad": "abdominal motor",
        "xm": "other motor",
    },
}


class ReadoutReproductionError(RuntimeError):
    """Raised when the shipped readout row fails to reproduce REACHABILITY-01."""


# ------------------------------------------------------------------- artifact


def artifact_identity(manifest: FullRuntimeManifest) -> dict[str, Any]:
    """Manifest-derived identity, matching the Phase 0 analysis field for field."""

    return {
        "artifact_id": manifest.artifact_id,
        "dataset_id": manifest.dataset_id,
        "manifest_file_sha256": manifest.manifest_file_sha256,
        "payload_sha256": manifest.payload_sha256,
        "structural_graph_sha256": manifest.structural_graph_sha256,
        "provenance_graph_sha256": manifest.provenance_graph_sha256,
        "source_manifest_sha256": manifest.source_manifest_sha256,
        "builder_implementation_sha256": manifest.builder_implementation_sha256,
        "builder_configuration_sha256": manifest.builder_configuration_sha256,
        "boundary_role_overlay_sha256": manifest.boundary_role_overlay.overlay_sha256,
        "node_count": manifest.node_count,
        "edge_count": manifest.edge_count,
        "raw_count_sum": manifest.raw_count_sum,
        "retained_weight_sum": manifest.counts["retained_weight_sum"],
        "annotation_source_rows": manifest.counts["annotation_source_rows"],
    }


def boundary_sites(manifest: FullRuntimeManifest) -> dict[str, list[dict[str, Any]]]:
    """Overlay sites keyed by role, ordered by body ID."""

    return {
        role: sorted(
            (
                {
                    "body_id": node.body_id,
                    "dense_index": node.dense_index,
                    "annotation_source_row": node.annotation_source_row,
                    "superclass": node.superclass,
                }
                for node in nodes
            ),
            key=lambda entry: int(entry["body_id"]),
        )
        for role, nodes in sorted(manifest.boundary_role_overlay.roles.items())
    }


def declared_annotations_sha256(manifest: FullRuntimeManifest) -> str:
    """The source digest the manifest declares for the pinned annotations table."""

    for source in manifest.source_artifacts:
        if source.source_id == "annotations":
            return source.sha256
    raise KeyError("the artifact manifest declares no 'annotations' source")


# ------------------------------------------------------------------- analysis


def reproduction_check(comparison: dict[str, Any], node_count: int) -> dict[str, Any]:
    """Compare the shipped-readout row against the published Phase 0 figures."""

    shipped = next(
        row for row in comparison["rows"] if row["readout_set"] == "shipped_two_body_readout"
    )
    checks = {
        "one_game_tick_path_support_nodes": (
            int(shipped["one_game_tick_path_support_nodes"]),
            int(REACHABILITY_01["one_game_tick_path_support_nodes"]),
        ),
        "unbounded_path_support_nodes": (
            int(shipped["unbounded_path_support_nodes"]),
            int(REACHABILITY_01["unbounded_path_support_nodes"]),
        ),
        "shortest_injection_to_readout_path": (
            int(shipped["shortest_injection_to_readout_path"]),
            int(REACHABILITY_01["shortest_injection_to_readout_path"]),
        ),
        "node_count": (int(node_count), int(REACHABILITY_01["node_count"])),
    }
    rows: list[dict[str, Any]] = [
        {
            "quantity": name,
            "observed": observed,
            "published": published,
            "matches": observed == published,
        }
        for name, (observed, published) in sorted(checks.items())
    ]
    return {
        "receipt": REACHABILITY_01["receipt"],
        "rule": (
            "this analysis reuses the Phase 0 traversal code unchanged, so the shipped"
            " two-body readout row must reproduce the published figures exactly. A"
            " deviation is a blocking error, never a new result"
        ),
        "reproduced": all(row["matches"] for row in rows),
        "checks": rows,
    }


def analyse(
    annotations: RetainedAnnotations,
    taken: Any,
    graph: Any,
    edges: tuple[IntArray, IntArray, IntArray],
    *,
    identity: dict[str, Any],
    sites: dict[str, list[dict[str, Any]]],
    gates: Sequence[IntegrityGate],
    steps_per_game_tick: int = NEURAL_STEPS_PER_GAME_TICK,
) -> dict[str, Any]:
    """Produce the whole deterministic analysis body."""

    require_all_gates(gates)
    pre, post, weights = edges
    injection: IntArray = np.asarray(
        sorted(int(site["dense_index"]) for site in sites["sensory"]), dtype=np.int64
    )
    shipped = [int(site["dense_index"]) for site in sites["motor"]]
    comparison = latency_comparison(
        graph,
        injection,
        readout_sets(annotations, shipped),
        steps_per_game_tick=steps_per_game_tick,
    )
    contacts = direct_contacts(annotations, pre, post, weights)
    return {
        "schema": SCHEMA,
        "artifact": identity,
        "integrity_gates": gates_report(gates),
        "boundary_role_overlay": sites,
        "readout_eligible_population": readout_population_report(annotations),
        "vnc_motor_census": vnc_motor_census(annotations),
        "current_readout_identity": boundary_identity_report(annotations, taken, sites),
        "descending_neuron_census": descending_census(annotations),
        "fixture_named_descending_neurons": fixture_named_descending(annotations, taken),
        "latency_bounded_readout_comparison": comparison,
        "reachability_01_reproduction": reproduction_check(comparison, graph.node_count),
        "descending_to_motor_connectivity": descending_to_motor_report(annotations, contacts),
        "interpretation_bounds": {
            "topology_and_annotation_only": (
                "every figure is either a count of source-observed annotation strings or"
                " a topological measurement over the retained edge support. Nothing here"
                " executes dynamics, and nothing here is evidence in the sense"
                " docs/SCIENTIFIC_CLAIMS.md reserves for G3"
            ),
            "reachability_is_an_upper_bound": (
                "reachability bounds participation and never demonstrates it. A node"
                " inside a path-support set is only topologically able to lie on a path,"
                " at an unmodelled and possibly vanishing, cancelling or sign-inverted"
                " gain"
            ),
            "widening_is_not_a_recommendation": (
                "the readout ladder measures what each candidate population would make"
                " topologically eligible under a stated graph-hop budget. Legacy"
                " one_game_tick fields are loose historical bounds; fresh-input tick"
                " eligibility is reported separately. It does not show that a wider"
                " readout produces better, more biological or more faithful behaviour,"
                " and it is not an adapter design decision"
            ),
            "annotation_labels_are_dataset_labels": (
                "superclass, subclass, type and somaSide are pinned-table annotations."
                " They are not measured functions, and this analysis assigns no"
                " behaviour to any neuron"
            ),
            "weights_are_raw_counts": (
                "edge weight is the retained raw synapse count as shipped: no"
                " normalisation, no sign policy and no efficacy claim is applied"
            ),
            "no_gate_movement": (
                "this analysis promotes no gate, invalidates no existing receipt, and"
                " changes no runtime behaviour"
            ),
        },
    }


# -------------------------------------------------------------------- output


def _peak_rss_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return observed if sys.platform == "darwin" else observed * 1024


def _write_report(path: Path, document: dict[str, Any]) -> None:
    encoded = json.dumps(document, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _print_gates(analysis: dict[str, Any]) -> None:
    gates = analysis["integrity_gates"]
    print(f"integrity gates: {len(gates['gates'])} evaluated, all_passed={gates['all_passed']}")
    for gate in gates["gates"]:
        print(f"  [{'PASS' if gate['passed'] else 'FAIL'}] {gate['name']}")


def _print_population(analysis: dict[str, Any]) -> None:
    print("superclass                 retained  fraction")
    for row in analysis["readout_eligible_population"]["rows"]:
        print(
            f"{row['superclass']:<24}  {row['retained_nodes']:>8,}"
            f"  {row['fraction_of_retained_nodes']:>8.4%}"
        )
    naive = analysis["readout_eligible_population"]["naive_descending_filter"]
    print(
        "superclass literally 'descending': "
        f"{naive['nodes_matching_superclass_exactly_descending']}"
        f"  (substring matches: {', '.join(naive['substring_matching_superclasses'])})"
    )
    motor = analysis["vnc_motor_census"]
    print(f"vnc_motor subclasses: {motor['subclass_counts']}")
    wing = motor["wing_motor_subclass"]
    print(
        f"wing motor (wm): {wing['cells']} cells, {wing['distinct_types']} types,"
        f" {wing['untyped_cells']} untyped; neuromeres {wing['soma_neuromere_counts']}"
    )


def _print_identity(analysis: dict[str, Any]) -> None:
    summary = analysis["current_readout_identity"]["current_motor_readout_summary"]
    print(
        f"current motor readout {summary['bodies']} -> type(s)"
        f" {summary['distinct_types']} subclass {summary['subclasses']}"
        f" sides {summary['soma_sides']} neuromere {summary['soma_neuromeres']}"
        f" exit nerve {summary['exit_nerves']}"
    )
    for role in sorted(analysis["current_readout_identity"]["roles"]):
        for entry in analysis["current_readout_identity"]["roles"][role]:
            row = entry["annotation_row"]
            print(
                f"  {role:<8} {entry['body_id']:>8}  type={row['type']!s:<12}"
                f" subclass={row['subclass']!s:<24} class={row['class']!s}"
            )


def _print_descending(analysis: dict[str, Any]) -> None:
    census = analysis["descending_neuron_census"]
    print(
        f"descending_neuron: {census['cells']} cells, {census['typed_cells']} typed,"
        f" {census['untyped_cells']} untyped, {census['distinct_types']} distinct types"
    )
    print(f"  prefixes: {census['type_prefix_counts']}")
    print(f"  bilaterality: {census['bilaterality']}")
    print(f"  somaSide: {census['soma_side_counts']}")
    for name, block in sorted(analysis["fixture_named_descending_neurons"]["types"].items()):
        print(
            f"  {name}: {block['cells_used_by_ci_fixture']} of"
            f" {block['cells_in_full_artifact']} cells used by the CI fixture"
            f"  sides {block['soma_side_counts']}"
        )


def _print_latency(analysis: dict[str, Any]) -> None:
    comparison = analysis["latency_bounded_readout_comparison"]
    tick = comparison["neural_steps_per_game_tick"]
    fresh_hops = comparison["fresh_input_timing"]["max_graph_hops"]
    print(f"legacy graph-hop budget L={tick}: loose upper bound, not fresh-input timing")
    print(f"fresh input: {tick} updates permit at most {fresh_hops} graph hops (eligibility only)")
    print("readout set                              sites   legacy nodes  fresh nodes  unbounded")
    for row in comparison["rows"]:
        print(
            f"{row['readout_set']:<38}  {row['readout_sites']:>5}"
            f"  {row['one_game_tick_path_support_nodes']:>9,}"
            f"  {row['fresh_input_one_game_tick_path_support_nodes']:>9,}"
            f"  {row['unbounded_path_support_fraction']:>8.4%}"
        )
    check = analysis["reachability_01_reproduction"]
    print(f"REACHABILITY-01 reproduction: {check['reproduced']}")
    for row in check["checks"]:
        print(
            f"  {row['quantity']:<40} observed {row['observed']:>8}"
            f"  published {row['published']:>8}"
        )


def _print_connectivity(analysis: dict[str, Any]) -> None:
    block = analysis["descending_to_motor_connectivity"]
    print(
        f"descending -> vnc_motor: {block['edges']:,} edges, {block['weight']:,} weight ="
        f" {block['share_of_all_incoming_vnc_motor_weight']:.4%} of all incoming vnc_motor"
        f" weight ({block['total_incoming_vnc_motor_weight']:,})"
    )
    print(
        f"  descending cells with a direct motor target:"
        f" {block['descending_cells_with_a_direct_motor_target']:,} of"
        f" {block['descending_cells']:,} ="
        f" {block['descending_cells_with_a_direct_motor_target_fraction']:.4%}"
    )
    print("type       cells  edges   weight     wing      leg   laterality")
    for row in block["named_types"]:
        wing = row["wing_weight_fraction"]
        leg = row["leg_weight_fraction"]
        index = row["laterality"]["laterality_index"]
        print(
            f"{row['type']:<10} {row['cells']:>5}  {row['edges']:>5}"
            f"  {row['motor_weight']:>7,}"
            f"  {0.0 if wing is None else wing:>6.1%}"
            f"  {0.0 if leg is None else leg:>6.1%}"
            f"   {'n/a' if index is None else f'{index:+.4f}'}"
        )


def _print_summary(analysis: dict[str, Any], digest: str) -> None:
    print(f"artifact {analysis['artifact']['artifact_id']}")
    print(f"manifest {analysis['artifact']['manifest_file_sha256'][:12]}")
    _print_gates(analysis)
    _print_population(analysis)
    _print_identity(analysis)
    _print_descending(analysis)
    _print_latency(analysis)
    _print_connectivity(analysis)
    print(f"analysis_sha256 {digest}")


# ----------------------------------------------------------------------- CLI


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--manifest-sha256", default=TRACKED_MANIFEST_SHA256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-reproduction-drift",
        action="store_true",
        help=(
            "record a REACHABILITY-01 mismatch in the report instead of refusing;"
            " for diagnosing a deviation, never for producing evidence"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.perf_counter()
    with load_full_runtime_artifact(
        args.artifact, expected_manifest_sha256=args.manifest_sha256
    ) as artifact:
        manifest = artifact.manifest
        identity = artifact_identity(manifest)
        sites = boundary_sites(manifest)
        declared_sha256 = declared_annotations_sha256(manifest)
        retained_by_superclass = dict(manifest.retained_by_superclass)
        declared_rows = manifest.counts["annotation_source_rows"]
        retained_weight_sum = manifest.counts["retained_weight_sum"]
        source_rows = np.array(mapped_numpy(artifact.annotation_source_rows, "<u8"))
        body_ids = np.array(mapped_numpy(artifact.body_ids, "<u8"))
        pre = np.array(mapped_numpy(artifact.edge_pre_indices, "<u4"))
        post = np.array(mapped_numpy(artifact.edge_post_indices, "<u4"))
        weights = np.array(mapped_numpy(artifact.edge_raw_counts, "<u8")).astype(np.int64)
        graph = graph_from_full_runtime(artifact)

    annotations, taken = take_retained_annotations(
        args.annotations, source_rows=source_rows, body_ids=body_ids
    )
    gates = [
        *annotation_gates(
            annotations,
            declared_annotations_sha256=declared_sha256,
            declared_source_rows=declared_rows,
            retained_by_superclass=retained_by_superclass,
        ),
        weight_sum_gate(weights, retained_weight_sum),
    ]
    analysis = analyse(
        annotations,
        taken,
        graph,
        (pre, post, weights),
        identity=identity,
        sites=sites,
        gates=gates,
    )
    if not analysis["reachability_01_reproduction"]["reproduced"] and not (
        args.allow_reproduction_drift
    ):
        raise ReadoutReproductionError(
            "the shipped two-body readout row did not reproduce"
            f" {REACHABILITY_01['receipt']}; refusing to write a report"
        )
    digest = hashlib.sha256(canonical_json_bytes(analysis)).hexdigest()
    document = {
        "analysis": analysis,
        "analysis_sha256": digest,
        "literature_inferred_notes": LITERATURE_INFERRED_NOTES,
        "execution": {
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "peak_rss_bytes": _peak_rss_bytes(),
            "numpy": np.__version__,
            "python": sys.version,
            "os": platform.platform(),
        },
    }
    _write_report(args.output, document)
    _print_summary(analysis, digest)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
