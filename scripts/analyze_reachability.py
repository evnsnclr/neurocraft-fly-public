#!/usr/bin/env python3
"""Phase 0 topological reachability analysis of the full MaleCNS runtime graph.

The bounded question: with the current engineered adapters injecting into two
named sensory boundary neurons and reading two named motor boundary neurons,
how many of the 166,700 retained nodes can lie on any directed path from an
injected node to a read node? Everything below is graph topology only.

Method, and why it was chosen:

* Identity. Every array comes from ``load_full_runtime_artifact``, so the
  manifest, payload, structural, provenance and per-array SHA-256 checks all run
  before a number is computed. NumPy views are taken over the exact mappings the
  loader hashed rather than by reopening a path, so no verify-then-reopen window
  is introduced; the views are copied into owned arrays and the artifact is
  closed before the analysis runs.

* Layers. :mod:`scripts.reachability_graph` owns the CSR edge support and the two
  traversal directions and documents the transpose choice;
  :mod:`scripts.reachability_measures` turns traversals into report fragments;
  this file binds those fragments to the verified manifest, adds the
  manifest-derived context a reader needs to size the result, and writes the
  document.

Claim discipline. This analysis deliberately avoids the word "causal": that term
is reserved by ``docs/SCIENTIFIC_CLAIMS.md`` for the G3 gate, and nothing here
touches dynamics. The measured set is called *path support*: the nodes lying on
at least one directed path from an injection site to a readout site through the
retained edge support. Path support is an upper bound on participation. A node
outside it provably cannot reach the motor readout through this edge support at
any number of steps. A node inside it is only *topologically able* to, at an
unmodelled and possibly vanishing gain; this analysis says nothing about
dynamics, signs, weights or behaviour, and its latency table bounds hop counts,
not milliseconds.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ``python scripts/analyze_reachability.py`` places ``scripts/`` rather than the
# repository root on sys.path. Add only this script's resolved repository root so
# the two sibling analysis modules have the same import identity under direct CLI
# and test use; this is the convention run_external_gametest.py already follows.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurocraft_connectome import load_full_runtime_artifact
from neurocraft_connectome.full_runtime import FullRuntimeManifest
from neurocraft_connectome.model import canonical_json_bytes
from neurocraft_controller.config import ControllerConfiguration

from scripts.reachability_graph import (
    BoolArray,
    DirectedGraph,
    WeightedEdges,
    graph_from_full_runtime,
    weighted_edges_from_full_runtime,
)
from scripts.reachability_measures import (
    Sites,
    baseline_report,
    ceiling_report,
    core_report,
    degeneracy_report,
    latency_report,
    scaling_points,
    scaling_report,
    structure_report,
    weight_sensitivity_report,
)

SCHEMA = "neurocraft.analysis.reachability.v2"
DEFAULT_ARTIFACT_DIR = Path("data/processed/male-cns-v1.0-superclass-v1")
TRACKED_MANIFEST_SHA256 = "b33bdda8b1a9abfc6b23f7346197442546f7a2aec0a03fc7c07fbdfe4f67c98c"
DEFAULT_OUTPUT = Path("artifacts/analysis/reachability.json")
DEFAULT_SITE_COUNTS = (1, 2, 4, 8, 16, 64, 256, 1024, 4096)
DEFAULT_WEIGHT_THRESHOLDS = (1, 2, 3, 5, 10)
DEFAULT_REPEATS = 5
DEFAULT_BASE_SEED = 20260818
NEURAL_STEPS_PER_GAME_TICK = ControllerConfiguration().neural_steps_per_game_tick


@dataclass(frozen=True, slots=True)
class ArtifactFacts:
    """Manifest-derived facts the report needs beyond the edge support itself."""

    identity: dict[str, Any]
    sites: Sites
    counts: dict[str, int]
    retained_by_superclass: dict[str, int]


def _artifact_identity(manifest: FullRuntimeManifest) -> dict[str, Any]:
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
        "retained_self_edge_rows": manifest.counts["retained_self_edge_rows"],
    }


def _boundary_sites(manifest: FullRuntimeManifest) -> Sites:
    return {
        role: [
            {
                "body_id": node.body_id,
                "dense_index": node.dense_index,
                "superclass": node.superclass,
            }
            for node in nodes
        ]
        for role, nodes in sorted(manifest.boundary_role_overlay.roles.items())
    }


def artifact_facts(manifest: FullRuntimeManifest) -> ArtifactFacts:
    """Collect everything the report needs from the verified manifest."""

    return ArtifactFacts(
        identity=_artifact_identity(manifest),
        sites=_boundary_sites(manifest),
        counts=dict(manifest.counts),
        retained_by_superclass=dict(manifest.retained_by_superclass),
    )


def adapter_coverage(sites: Sites, retained_by_superclass: dict[str, int]) -> dict[str, Any]:
    """How much of each named boundary population the two-site adapters touch.

    Phase 0 asks what the current adapters can reach, and the answer is bounded
    as much by how few boundary neurons they use as by the topology, so the ratio
    belongs in the same report as the reachability figure.
    """

    report: dict[str, Any] = {
        "denominator_rule": (
            "the population of a role is the retained node count of exactly the"
            " superclasses its own overlay sites carry. Other boundary-like superclasses"
            " (ol_sensory, sensory_ascending, cb_motor and the *_tbc variants) are"
            " excluded, so each reported fraction is an upper bound on adapter coverage"
            " of everything the artifact labels with that role"
        )
    }
    for role in sorted(sites):
        superclasses = sorted({str(entry["superclass"]) for entry in sites[role]})
        population = sum(retained_by_superclass[name] for name in superclasses)
        used = len(sites[role])
        report[role] = {
            "sites": used,
            "superclasses": superclasses,
            "population_by_superclass": {
                name: retained_by_superclass[name] for name in superclasses
            },
            "retained_population": population,
            "site_fraction_of_population": used / population if population else None,
        }
    return report


def attrition_report(counts: dict[str, int]) -> dict[str, Any]:
    """Record which way source-row filtering moves the measured reachability."""

    pre_only = counts["excluded_pre_only_rows"]
    post_only = counts["excluded_post_only_rows"]
    retained = counts["retained_edge_rows"]
    leaving = retained + pre_only
    arriving = retained + post_only
    raw_weight = counts["raw_weight_sum"]
    return {
        "excluded_pre_only_rows": pre_only,
        "excluded_post_only_rows": post_only,
        "excluded_neither_endpoint_rows": counts["excluded_neither_endpoint_rows"],
        "pre_only_over_post_only": pre_only / post_only if post_only else None,
        "row_semantics": (
            "the builder counts excluded_pre_only_rows as rows whose presynaptic body is"
            " retained and whose postsynaptic body was dropped, and"
            " excluded_post_only_rows as the mirror case. The attrition is therefore"
            " overwhelmingly postsynaptic: a retained node loses far more outgoing rows"
            " to dropped targets than incoming rows from dropped sources"
        ),
        "rows_leaving_retained_presynaptic_nodes": leaving,
        "retained_fraction_of_rows_leaving_retained_presynaptic_nodes": retained / leaving,
        "rows_arriving_at_retained_postsynaptic_nodes": arriving,
        "retained_fraction_of_rows_arriving_at_retained_postsynaptic_nodes": retained / arriving,
        "raw_weight_sum": raw_weight,
        "retained_weight_sum": counts["retained_weight_sum"],
        "retained_weight_fraction_of_raw": (
            counts["retained_weight_sum"] / raw_weight if raw_weight else None
        ),
        "direction": (
            "deleting an edge can remove a directed path but can never create one, so"
            " every reachability figure in this report is a LOWER bound on the value the"
            " unfiltered source rows would give. The filtering cuts in this analysis's"
            " favour; a reader who assumes it inflated the numbers has the sign backwards"
        ),
    }


def analyse(
    graph: DirectedGraph,
    edges: WeightedEdges,
    facts: ArtifactFacts,
    *,
    site_counts: Sequence[int],
    repeats: int,
    base_seed: int,
    weight_thresholds: Sequence[int] = DEFAULT_WEIGHT_THRESHOLDS,
    steps_per_game_tick: int = NEURAL_STEPS_PER_GAME_TICK,
) -> dict[str, Any]:
    """Produce the whole deterministic analysis body."""

    sites = facts.sites
    baseline, forward, backward = baseline_report(graph, sites)
    forward_mask: BoolArray = forward >= 0
    backward_mask: BoolArray = backward >= 0
    support: BoolArray = forward_mask & backward_mask
    points = scaling_points(
        graph,
        forward_mask,
        backward_mask,
        site_counts=site_counts,
        repeats=repeats,
        base_seed=base_seed,
    )
    degeneracy = degeneracy_report(graph, sites, support, backward_mask, points)
    return {
        "schema": SCHEMA,
        "artifact": facts.identity,
        "boundary_role_overlay": sites,
        "adapter_coverage": adapter_coverage(sites, facts.retained_by_superclass),
        "structure": structure_report(graph),
        "current_configuration": baseline,
        "latency_bounds": latency_report(
            forward, backward, graph.node_count, steps_per_game_tick=steps_per_game_tick
        ),
        "weight_threshold_sensitivity": weight_sensitivity_report(edges, sites, weight_thresholds),
        "scaling": scaling_report(
            points, graph.node_count, degeneracy, repeats=repeats, base_seed=base_seed
        ),
        "ceilings": ceiling_report(graph, forward_mask, backward_mask),
        "connected_core": core_report(graph, sites, support),
        "interpretation_bounds": {
            "topology_only": (
                "topological reachability only; an upper bound on which nodes can"
                " participate, never evidence that they do participate, and never a"
                " statement about dynamics, signs, weights or behaviour. Nothing here is"
                " causal evidence in the sense docs/SCIENTIFIC_CLAIMS.md reserves for G3"
            ),
            "latency_scope": (
                "the table bounds graph hops, not neural updates or milliseconds. Legacy"
                " one_game_tick keys retain a loose historical L=update-count hop budget."
                " Fresh input first qualifies at update 1+2*h under baseline delay 1;"
                " separate fresh-input tick fields remain topology-only eligibility,"
                " never a real-time (G3-R) measurement"
            ),
            "edge_attrition_direction": attrition_report(facts.counts),
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


def _print_headline(analysis: dict[str, Any], digest: str) -> None:
    node_count = analysis["structure"]["node_count"]
    current = analysis["current_configuration"]
    tick = analysis["latency_bounds"]["one_game_tick"]
    print(f"artifact {analysis['artifact']['artifact_id']} manifest {digest[:12]}")
    print(f"nodes {node_count}  edges {analysis['structure']['edge_count']}")
    print(
        "current 2-in/2-out: forward "
        f"{current['forward_reachable']} ({current['forward_reachable_fraction']:.4%})"
        f"  backward {current['backward_reachable']}"
        f" ({current['backward_reachable_fraction']:.4%})"
    )
    print(
        "PATH SUPPORT (unbounded): "
        f"{current['path_support_nodes']} nodes ="
        f" {current['path_support_fraction']:.4%} of {node_count}"
    )
    if tick is not None:
        print(
            f"PATH SUPPORT (legacy <= {tick['max_path_length']} graph hops; loose bound): "
            f"{tick['path_support_nodes']} nodes = {tick['path_support_fraction']:.4%}"
        )
    fresh = analysis["latency_bounds"]["fresh_input_one_game_tick"]
    print(
        f"FRESH-INPUT TICK ELIGIBILITY (<= {fresh['max_graph_hops']} graph hops): "
        f"{fresh['path_support_nodes']} nodes = {fresh['path_support_fraction']:.4%}"
        " (topology only)"
    )
    print(f"forward hop profile: {current['forward_hops']['hops_to_cover_reached_fraction']}")
    print(f"backward hop profile: {current['backward_hops']['hops_to_cover_reached_fraction']}")
    print(f"sensory -> motor hops: {current['hops_from_sensory_to_motor_body']}")


def _print_tables(analysis: dict[str, Any]) -> None:
    print("max_graph_hops  path_support  fraction")
    for row in analysis["latency_bounds"]["table"]:
        print(
            f"{row['max_path_length']:>15}  {row['path_support_nodes']:>12,}"
            f"  {row['path_support_fraction']:>9.4%}"
        )
    print("min_edge_weight  retained_edges  path_support  fraction")
    for row in analysis["weight_threshold_sensitivity"]["thresholds"]:
        print(
            f"{row['minimum_edge_weight']:>15}  {row['retained_edges']:>14,}"
            f"  {row['path_support_nodes']:>12,}  {row['path_support_fraction']:>9.4%}"
        )
    print("DEGENERATE widening curve: measures sampling of non-component nodes only")
    print("site_count  widen-injection  widen-readout  widen-both  (median path support)")
    curves = analysis["scaling"]["curves"]
    for injection, readout, both in zip(
        curves["widened_injection_actual_readout"],
        curves["widened_readout_actual_injection"],
        curves["widened_both"],
        strict=True,
    ):
        print(
            f"{injection['site_count']:>10}  {injection['median']:>15,.0f}"
            f"  {readout['median']:>13,.0f}  {both['median']:>10,.0f}"
        )
    peripheral = analysis["scaling"]["degeneracy"]["peripheral_injection"]
    rows = ", ".join(
        f"K={row['peripheral_sites']}->{row['path_support_nodes']:,}" for row in peripheral["rows"]
    )
    print(
        f"peripheral injection ({peripheral['available_peripheral_nodes']} nodes"
        f" outside the component): {rows}"
    )


def _print_summary(analysis: dict[str, Any], digest: str) -> None:
    _print_headline(analysis, digest)
    _print_tables(analysis)
    core = analysis["connected_core"]
    print(
        "strongly connected component of the first injection site: "
        f"{core['sensory']['strongly_connected_component']}"
        f"  of the first readout site: {core['motor']['strongly_connected_component']}"
        f"  set-equal to path support: {core['sensory']['equals_path_support_set']}"
        f"/{core['motor']['equals_path_support_set']}"
    )
    ceilings = analysis["ceilings"]
    print(
        "ceilings: unbounded injection -> "
        f"{ceilings['unbounded_injection_actual_readout']}"
        f"  unbounded readout -> {ceilings['unbounded_readout_actual_injection']}"
        f"  zero-in to zero-out -> {ceilings['zero_in_to_zero_out_path_nodes']}"
        f"  nontrivial any-to-any -> {ceilings['nontrivial_any_to_any']}"
    )
    coverage = analysis["adapter_coverage"]
    print(
        "adapter coverage: "
        f"{coverage['sensory']['sites']}/{coverage['sensory']['retained_population']} sensory"
        f"  {coverage['motor']['sites']}/{coverage['motor']['retained_population']} motor"
    )
    attrition = analysis["interpretation_bounds"]["edge_attrition_direction"]
    print(
        "edge attrition: retained rows are "
        f"{attrition['retained_fraction_of_rows_leaving_retained_presynaptic_nodes']:.2%}"
        " of rows leaving retained presynaptic nodes and "
        f"{attrition['retained_weight_fraction_of_raw']:.2%} of raw weight;"
        " every figure above is therefore a LOWER bound"
    )


# ----------------------------------------------------------------------- CLI


def _int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in raw.split(",") if item.strip())
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--manifest-sha256", default=TRACKED_MANIFEST_SHA256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--site-counts",
        type=_int_list,
        default=DEFAULT_SITE_COUNTS,
        help="comma-separated injection/readout widths for the degenerate scaling curves",
    )
    parser.add_argument(
        "--weight-thresholds",
        type=_int_list,
        default=DEFAULT_WEIGHT_THRESHOLDS,
        help="comma-separated minimum retained synapse counts for the sensitivity table",
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.repeats < 1:
        raise SystemExit("repeats must be positive")
    started = time.perf_counter()
    with load_full_runtime_artifact(
        args.artifact, expected_manifest_sha256=args.manifest_sha256
    ) as artifact:
        facts = artifact_facts(artifact.manifest)
        graph = graph_from_full_runtime(artifact)
        edges = weighted_edges_from_full_runtime(artifact)
    analysis = analyse(
        graph,
        edges,
        facts,
        site_counts=tuple(args.site_counts),
        repeats=args.repeats,
        base_seed=args.base_seed,
        weight_thresholds=tuple(args.weight_thresholds),
    )
    digest = hashlib.sha256(canonical_json_bytes(analysis)).hexdigest()
    document = {
        "analysis": analysis,
        "analysis_sha256": digest,
        "execution": {
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "peak_rss_bytes": _peak_rss_bytes(),
            "numpy": np.__version__,
            "python": sys.version,
            "os": platform.platform(),
        },
    }
    _write_report(args.output, document)
    _print_summary(analysis, facts.identity["manifest_file_sha256"])
    print(f"analysis_sha256 {digest}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
