"""Topological measurements over the retained edge support.

The middle layer of the Phase 0 reachability analysis:
:mod:`scripts.reachability_graph` owns the edge support and the traversals, this
module turns traversals into report fragments, and
``scripts/analyze_reachability.py`` binds those fragments to the verified artifact
and writes the document. Split three ways so that no file exceeds the
repository's 800-line ceiling; the import path is the same
scripts-are-importable convention the rest of ``tests/`` already uses.

Claim discipline. The measured quantity is *path support*: the nodes lying on at
least one directed path from an injection site to a readout site through the
retained edge support. "Causal" is reserved by ``docs/SCIENTIFIC_CLAIMS.md`` for
the G3 gate and is deliberately not used for anything here, which is pure
topology.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from neurocraft_neural import DynamicsConfig

from scripts.reachability_graph import (
    BoolArray,
    DirectedGraph,
    IntArray,
    WeightedEdges,
    backward_distances,
    forward_distances,
)

REPORTED_FRACTIONS = (0.5, 0.9, 0.99, 1.0)
INJECTION_STREAM = 1
READOUT_STREAM = 2
SELECTION_RULE = (
    "numpy.random.default_rng([base_seed, repeat, stream]).permutation(node_count);"
    " sites(K) is the first K entries of that one permutation, so sites(K) is always"
    " a subset of sites(2K) within a repeat and any increase is attributable to the"
    " added sites rather than to an independent redraw. Stream 1 selects injection"
    " sites and stream 2 selects readout sites"
)

CEILING_DEFINITIONS = {
    "unbounded_injection_actual_readout": (
        "nodes that reach the two named motor boundary neurons through the retained edge"
        " support over any number of hops; the K -> node_count limit of the injection curve"
    ),
    "unbounded_readout_actual_injection": (
        "nodes reached from the two named sensory boundary neurons through the retained"
        " edge support over any number of hops; the K -> node_count limit of the readout"
        " curve"
    ),
    "zero_in_to_zero_out_path_nodes": (
        "nodes on at least one directed path from some zero-in-degree node to some"
        " zero-out-degree node; bounds a system whose input is restricted to topological"
        " entry points and whose output is read at topological exit points, and excludes"
        " nodes that only sit on cycles"
    ),
    "nontrivial_any_to_any": (
        "nodes with in-degree >= 1 and out-degree >= 1, i.e. nodes on some path of"
        " length >= 2; the ceiling when injection and readout sites are both"
        " unrestricted and a zero-hop self-path does not count"
    ),
    "trivial_all_to_all": (
        "every node reaches itself in zero hops, so injecting into and reading from all"
        " nodes trivially yields the whole graph; reported to make that vacuity explicit"
        " rather than quotable"
    ),
}

Sites = dict[str, list[dict[str, Any]]]


def dense_indices(sites: Sites, role: str) -> IntArray:
    """Dense indices of every overlay site holding ``role``."""

    indices: IntArray = np.asarray([entry["dense_index"] for entry in sites[role]], dtype=np.int64)
    return indices


def boundary_hops(distances: IntArray, sites: list[dict[str, Any]]) -> dict[str, int]:
    """Hop distance to each named boundary body, keyed by source body ID."""

    return {str(entry["body_id"]): int(distances[entry["dense_index"]]) for entry in sites}


# ------------------------------------------------------------------ summaries


@dataclass(frozen=True, slots=True)
class PathSupportCounts:
    """Node counts for one (injection set, readout set) pair."""

    forward: int
    backward: int
    path_support: int
    forward_only: int
    backward_only: int
    untouched: int


def path_support_counts(
    forward: BoolArray, backward: BoolArray, node_count: int
) -> PathSupportCounts:
    """Intersect a downstream mask with an upstream mask."""

    reached = int(np.count_nonzero(forward))
    upstream = int(np.count_nonzero(backward))
    both = int(np.count_nonzero(forward & backward))
    return PathSupportCounts(
        forward=reached,
        backward=upstream,
        path_support=both,
        forward_only=reached - both,
        backward_only=upstream - both,
        untouched=node_count - reached - upstream + both,
    )


def _fractions(counts: PathSupportCounts, node_count: int) -> dict[str, Any]:
    return {
        "forward_reachable": counts.forward,
        "forward_reachable_fraction": counts.forward / node_count,
        "backward_reachable": counts.backward,
        "backward_reachable_fraction": counts.backward / node_count,
        "path_support_nodes": counts.path_support,
        "path_support_fraction": counts.path_support / node_count,
        "forward_only": counts.forward_only,
        "backward_only": counts.backward_only,
        "untouched": counts.untouched,
    }


def hop_profile(distances: IntArray) -> dict[str, Any]:
    """Hop histogram plus the hop at which each coverage fraction is reached."""

    reached = int(np.count_nonzero(distances >= 0))
    max_hops = int(distances.max())
    hop_counts = [int(np.count_nonzero(distances == hop)) for hop in range(max_hops + 1)]
    cumulative = np.cumsum(np.asarray(hop_counts, dtype=np.int64))
    coverage = {
        f"p{int(fraction * 100)}": int(
            np.searchsorted(cumulative, int(np.ceil(fraction * reached)), side="left")
        )
        for fraction in REPORTED_FRACTIONS
    }
    return {
        "reached": reached,
        "max_hops": max_hops,
        "nodes_per_hop": hop_counts,
        "hops_to_cover_reached_fraction": coverage,
    }


def degree_summary(degrees: IntArray) -> dict[str, Any]:
    """Min/median/mean/max plus tail percentiles for one degree distribution."""

    return {
        "min": int(degrees.min()),
        "median": float(np.median(degrees)),
        "mean": float(degrees.mean()),
        "p90": float(np.percentile(degrees, 90)),
        "p99": float(np.percentile(degrees, 99)),
        "max": int(degrees.max()),
        "zero_count": int(np.count_nonzero(degrees == 0)),
    }


def structure_report(graph: DirectedGraph) -> dict[str, Any]:
    """Degree distributions and the source/sink/isolated node partition."""

    in_degrees = graph.in_degrees
    out_degrees = graph.out_degrees
    sources = in_degrees == 0
    sinks = out_degrees == 0
    return {
        "node_count": graph.node_count,
        "edge_count": graph.edge_count,
        "self_edges": graph.self_edges,
        "in_degree": degree_summary(in_degrees),
        "out_degree": degree_summary(out_degrees),
        "source_nodes_zero_in_degree": int(np.count_nonzero(sources)),
        "sink_nodes_zero_out_degree": int(np.count_nonzero(sinks)),
        "isolated_nodes": int(np.count_nonzero(sources & sinks)),
        "interior_nodes": int(np.count_nonzero(~sources & ~sinks)),
        "degree_note": (
            "degrees count retained directed rows including preserved self-edges;"
            " a self-edge alone keeps a node out of the source and sink partitions"
        ),
    }


def baseline_report(
    graph: DirectedGraph, sites: Sites
) -> tuple[dict[str, Any], IntArray, IntArray]:
    """Analyse the shipped two-in / two-out configuration."""

    forward = forward_distances(graph, dense_indices(sites, "sensory"))
    backward = backward_distances(graph, dense_indices(sites, "motor"))
    counts = path_support_counts(forward >= 0, backward >= 0, graph.node_count)
    report = {
        "injection_sites": sites["sensory"],
        "readout_sites": sites["motor"],
        **_fractions(counts, graph.node_count),
        "forward_hops": hop_profile(forward),
        "backward_hops": hop_profile(backward),
        "hops_from_sensory_to_motor_body": boundary_hops(forward, sites["motor"]),
        "hops_from_motor_to_sensory_body": boundary_hops(backward, sites["sensory"]),
    }
    return report, forward, backward


# ------------------------------------------------------------- latency bounds


def fresh_input_timing(steps_per_game_tick: int) -> dict[str, Any]:
    """Structural eligibility for a fresh input, not measured signal propagation.

    Sensory drive first enters the newly committed activity at update 1. The
    recurrence reads the state from ``default_delay_steps + 1`` updates before
    its output: at delay 1, an edge first carries that new input at update 3,
    and a two-edge route first qualifies at update 5.
    """

    if type(steps_per_game_tick) is not int or steps_per_game_tick < 1:
        raise ValueError("steps_per_game_tick must be a positive integer")
    delay = DynamicsConfig().default_delay_steps
    stride = delay + 1
    return {
        "default_delay_steps": delay,
        "neural_updates_per_graph_hop": stride,
        "first_sensory_update": 1,
        "minimum_neural_updates_formula": "1 + (default_delay_steps + 1) * graph_hops",
        "neural_updates_available": steps_per_game_tick,
        "max_graph_hops": (steps_per_game_tick - 1) // stride,
        "scope": (
            "fresh sensory input with identical prior neural history; baseline delay only."
            " This is topological eligibility, not executed or nonzero influence. Larger"
            " delays can only reduce this bound; existing recurrent activity may already"
            " be present but is not a response to the fresh observation"
        ),
    }


def latency_report(
    forward: IntArray, backward: IntArray, node_count: int, *, steps_per_game_tick: int
) -> dict[str, Any]:
    """Path support restricted to injection -> readout paths of at most L hops.

    A node ``v`` lies on some injection -> readout path of length at most ``L``
    exactly when ``forward_hops(v) + hops_to_readout(v) <= L``, because the
    shortest walk through ``v`` is the sum of the two shortest one-sided walks.
    Both distance vectors are the ones already computed for the unbounded figure,
    so the whole table costs one vector add.

    Legacy ``one_game_tick`` keys retain their original L = update-count graph
    budget for historical reproduction. They are loose bounds, not timing.
    ``fresh_input_one_game_tick`` applies the executed delay-index convention.
    """

    timing = fresh_input_timing(steps_per_game_tick)
    reachable = (forward >= 0) & (backward >= 0)
    lengths = np.where(reachable, forward.astype(np.int64) + backward.astype(np.int64), -1)
    unbounded = int(np.count_nonzero(reachable))
    longest = int(lengths.max()) if unbounded else 0
    rows: list[dict[str, Any]] = []
    for bound in range(1, max(longest, steps_per_game_tick) + 1):
        nodes = int(np.count_nonzero(reachable & (lengths <= bound)))
        rows.append(
            {
                "max_path_length": bound,
                "max_graph_hops": bound,
                "fresh_input_minimum_neural_updates_for_hop_budget": (
                    1 + timing["neural_updates_per_graph_hop"] * bound
                ),
                "path_support_nodes": nodes,
                "path_support_fraction": nodes / node_count,
            }
        )
    saturated = [row for row in rows if row["path_support_nodes"] == unbounded]
    one_tick = [row for row in rows if row["max_path_length"] == steps_per_game_tick]
    fresh_hops = timing["max_graph_hops"]
    fresh_nodes = int(np.count_nonzero(reachable & (lengths <= fresh_hops)))
    return {
        "definition": (
            "a node lies on some injection -> readout directed path of length at most L"
            " exactly when its forward hop distance plus its hop distance to the readout"
            " is at most L"
        ),
        "step_model": (
            "graph hops are not neural updates. Fresh sensory drive first enters activity"
            " at update 1; a route of h edges is first eligible at update 1 + 2*h under"
            " the baseline default_delay_steps == 1. No dynamics are executed here"
        ),
        "tick_scope": (
            "legacy one_game_tick selects L = neural_steps_per_game_tick graph hops;"
            " it is a loose historical upper bound, not fresh-input reach within one"
            " tick. fresh_input_one_game_tick uses floor((updates-1)/2) graph hops."
            " Recurrent state persists across ticks, so these bounds do not limit"
            " participation from earlier observations"
        ),
        "fresh_input_timing": timing,
        "fresh_input_one_game_tick": {
            "max_graph_hops": fresh_hops,
            "path_support_nodes": fresh_nodes,
            "path_support_fraction": fresh_nodes / node_count,
            "interpretation": "topological eligibility only; not measured influence",
        },
        "neural_steps_per_game_tick": steps_per_game_tick,
        "shortest_injection_to_readout_path": int(lengths[reachable].min()) if unbounded else None,
        "longest_shortest_path_through_a_supported_node": longest,
        "saturation_path_length": saturated[0]["max_path_length"] if saturated else None,
        "one_game_tick": one_tick[0] if one_tick else None,
        "unbounded_path_support_nodes": unbounded,
        "unbounded_path_support_fraction": unbounded / node_count,
        "table": rows,
    }


# ------------------------------------------------ weight-threshold sensitivity


def weight_sensitivity_report(
    edges: WeightedEdges, sites: Sites, thresholds: Sequence[int]
) -> dict[str, Any]:
    """Recompute path support over edges of at least each minimum synapse count."""

    injection = dense_indices(sites, "sensory")
    readout = dense_indices(sites, "motor")
    rows: list[dict[str, Any]] = []
    for threshold in sorted({int(value) for value in thresholds}):
        subgraph = edges.subgraph(threshold)
        support = (forward_distances(subgraph, injection) >= 0) & (
            backward_distances(subgraph, readout) >= 0
        )
        nodes = int(np.count_nonzero(support))
        rows.append(
            {
                "minimum_edge_weight": threshold,
                "retained_edges": subgraph.edge_count,
                "retained_edge_fraction": subgraph.edge_count / edges.edge_count,
                "path_support_nodes": nodes,
                "path_support_fraction": nodes / edges.node_count,
            }
        )
        del subgraph
    return {
        "weight_semantics": (
            "raw retained synapse counts as shipped; no normalisation, no sign and no"
            " incoming-weight scaling is applied here"
        ),
        "direction_note": (
            "raising the threshold deletes edges, which can only remove directed paths,"
            " so this curve is monotone non-increasing by construction; it is reported"
            " because a steep fall would mean the unbounded figure rests on the weakest"
            " contacts"
        ),
        "edge_weight_median": float(np.median(edges.weights)),
        "edge_weight_mean": float(edges.weights.mean()),
        "edge_weight_max": int(edges.weights.max()),
        "thresholds": rows,
    }


# -------------------------------------------------------------- scaling curve


def site_permutation(node_count: int, key: tuple[int, ...]) -> IntArray:
    """One seeded permutation per (repeat, stream); every width is a prefix of it."""

    permutation: IntArray = np.random.default_rng(list(key)).permutation(node_count)
    return permutation


def select_sites(permutation: IntArray, site_count: int) -> IntArray:
    """Take the first ``site_count`` entries, sorted for contiguous CSR gathers."""

    if site_count < 1 or site_count > permutation.size:
        raise ValueError("site count must be between 1 and the node count")
    selection: IntArray = np.sort(permutation[:site_count])
    return selection


@dataclass(frozen=True, slots=True)
class ScalingPoint:
    """Per-repeat path-support counts at one site count, in repeat order."""

    site_count: int
    widened_injection: tuple[int, ...]
    widened_readout: tuple[int, ...]
    widened_both: tuple[int, ...]


def scaling_points(
    graph: DirectedGraph,
    baseline_forward: BoolArray,
    baseline_backward: BoolArray,
    *,
    site_counts: Sequence[int],
    repeats: int,
    base_seed: int,
) -> tuple[ScalingPoint, ...]:
    """Sweep nested injection/readout widths, reusing each sample for all curves."""

    node_count = graph.node_count
    widths = tuple(sorted({int(value) for value in site_counts}))
    injection: dict[int, list[int]] = {width: [] for width in widths}
    readout: dict[int, list[int]] = {width: [] for width in widths}
    both: dict[int, list[int]] = {width: [] for width in widths}
    for repeat in range(repeats):
        injection_order = site_permutation(node_count, (base_seed, repeat, INJECTION_STREAM))
        readout_order = site_permutation(node_count, (base_seed, repeat, READOUT_STREAM))
        for width in widths:
            wide_forward = forward_distances(graph, select_sites(injection_order, width)) >= 0
            wide_backward = backward_distances(graph, select_sites(readout_order, width)) >= 0
            injection[width].append(int(np.count_nonzero(wide_forward & baseline_backward)))
            readout[width].append(int(np.count_nonzero(baseline_forward & wide_backward)))
            both[width].append(int(np.count_nonzero(wide_forward & wide_backward)))
    return tuple(
        ScalingPoint(
            site_count=width,
            widened_injection=tuple(injection[width]),
            widened_readout=tuple(readout[width]),
            widened_both=tuple(both[width]),
        )
        for width in widths
    )


def _sample_spread(values: tuple[int, ...], node_count: int) -> dict[str, Any]:
    ordered = sorted(values)
    median = float(np.median(np.asarray(ordered, dtype=np.int64)))
    return {
        "median": median,
        "median_fraction": median / node_count,
        "min": ordered[0],
        "min_fraction": ordered[0] / node_count,
        "max": ordered[-1],
        "max_fraction": ordered[-1] / node_count,
        "samples": [
            {"repeat": repeat, "path_support_nodes": value} for repeat, value in enumerate(values)
        ],
    }


def peripheral_injection_points(
    graph: DirectedGraph, sites: Sites, path_support: BoolArray, baseline_backward: BoolArray
) -> dict[str, Any]:
    """Widen injection using only nodes outside the path support, in nested prefixes.

    This is the fair version of the widening question. Adding a site that is
    already inside the mutual component cannot change the answer, so the only
    informative injection sites are the outside ones that still have an outgoing
    edge. If this saturates too, widening is measuring nothing.
    """

    peripheral = np.flatnonzero((~path_support) & (graph.out_degrees > 0))
    actual = dense_indices(sites, "sensory")
    widths: list[int] = []
    width = 1
    while width < peripheral.size:
        widths.append(width)
        width *= 2
    if peripheral.size:
        widths.append(int(peripheral.size))
    rows: list[dict[str, Any]] = []
    for count in widths:
        seeds = np.concatenate([actual, peripheral[:count].astype(np.int64)])
        widened = forward_distances(graph, seeds) >= 0
        rows.append(
            {
                "peripheral_sites": count,
                "path_support_nodes": int(np.count_nonzero(widened & baseline_backward)),
            }
        )
    return {
        "selection_rule": (
            "all nodes outside the path support that still have out-degree >= 1, in"
            " ascending dense-index order; sites(K) is a prefix of sites(2K)"
        ),
        "available_peripheral_nodes": int(peripheral.size),
        "rows": rows,
    }


def degeneracy_report(
    graph: DirectedGraph,
    sites: Sites,
    path_support: BoolArray,
    baseline_backward: BoolArray,
    points: tuple[ScalingPoint, ...],
) -> dict[str, Any]:
    """Record that the widening curves are a measured tautology, with the derivation."""

    supported = int(np.count_nonzero(path_support))
    fraction = supported / graph.node_count
    curves = ("widened_injection", "widened_readout", "widened_both")

    def median_at(point: ScalingPoint, field: str) -> float:
        return float(np.median(np.asarray(getattr(point, field), dtype=np.int64)))

    smallest = {field: median_at(points[0], field) for field in curves} if points else {}
    largest = {field: median_at(points[-1], field) for field in curves} if points else {}
    return {
        "degenerate": True,
        "do_not_quote_as_a_scaling_result": (
            "the path-support set is exactly the graph's giant strongly connected"
            " component, so for any injection site drawn from inside that component the"
            " widened result is fully determined before the traversal runs. The curves"
            " therefore measure how often a uniform random draw lands outside the"
            " component and nothing else. They are retained as evidence that this metric"
            " is degenerate, never as a scaling result"
        ),
        "path_support_fraction_of_nodes": fraction,
        "probability_uniform_node_outside_path_support": 1.0 - fraction,
        "expected_outside_nodes_per_sample": [
            {"site_count": point.site_count, "expected": point.site_count * (1.0 - fraction)}
            for point in points
        ],
        "expected_versus_observed_note": (
            "compare expected_outside_nodes_per_sample at the largest site count with"
            " observed_median_increment_over_the_swept_range.widened_both: agreement is"
            " the whole result, because it shows the curve is reproduced by counting"
            " uniform draws that land outside the component"
        ),
        "median_at_smallest_site_count": smallest,
        "median_at_largest_site_count": largest,
        "observed_median_increment_over_the_swept_range": {
            field: largest[field] - smallest[field] for field in curves
        }
        if points
        else {},
        "peripheral_injection": peripheral_injection_points(
            graph, sites, path_support, baseline_backward
        ),
    }


def scaling_report(
    points: tuple[ScalingPoint, ...],
    node_count: int,
    degeneracy: dict[str, Any],
    *,
    repeats: int,
    base_seed: int,
) -> dict[str, Any]:
    """Serialize the three widening curves behind their degeneracy marker."""

    def curve(field: str) -> list[dict[str, Any]]:
        return [
            {
                "site_count": point.site_count,
                **_sample_spread(getattr(point, field), node_count),
            }
            for point in points
        ]

    return {
        "degenerate": True,
        "degeneracy": degeneracy,
        "base_seed": base_seed,
        "repeats": repeats,
        "selection_rule": SELECTION_RULE,
        "reuse_note": (
            "the combined curve intersects the same two sampled sets used by the"
            " injection-only and readout-only curves at that site count and repeat"
        ),
        "site_counts": [point.site_count for point in points],
        "curves": {
            "widened_injection_actual_readout": curve("widened_injection"),
            "widened_readout_actual_injection": curve("widened_readout"),
            "widened_both": curve("widened_both"),
        },
    }


# ------------------------------------------------------------------ ceilings


def ceiling_report(
    graph: DirectedGraph, baseline_forward: BoolArray, baseline_backward: BoolArray
) -> dict[str, Any]:
    """Purely topological best-case bounds; no cell-type strings exist here."""

    node_count = graph.node_count
    in_degrees = graph.in_degrees
    out_degrees = graph.out_degrees
    zero_in = np.flatnonzero(in_degrees == 0)
    zero_out = np.flatnonzero(out_degrees == 0)
    from_zero_in = forward_distances(graph, zero_in) >= 0
    to_zero_out = backward_distances(graph, zero_out) >= 0
    interior = int(np.count_nonzero((in_degrees > 0) & (out_degrees > 0)))
    source_to_sink = int(np.count_nonzero(from_zero_in & to_zero_out))
    unbounded_injection = int(np.count_nonzero(baseline_backward))
    unbounded_readout = int(np.count_nonzero(baseline_forward))
    return {
        "unbounded_injection_actual_readout": unbounded_injection,
        "unbounded_injection_actual_readout_fraction": unbounded_injection / node_count,
        "unbounded_readout_actual_injection": unbounded_readout,
        "unbounded_readout_actual_injection_fraction": unbounded_readout / node_count,
        "forward_reachable_from_all_zero_in_degree": int(np.count_nonzero(from_zero_in)),
        "backward_reachable_from_all_zero_out_degree": int(np.count_nonzero(to_zero_out)),
        "zero_in_to_zero_out_path_nodes": source_to_sink,
        "zero_in_to_zero_out_path_fraction": source_to_sink / node_count,
        "nontrivial_any_to_any": interior,
        "nontrivial_any_to_any_fraction": interior / node_count,
        "trivial_all_to_all": node_count,
        "definitions": CEILING_DEFINITIONS,
    }


def core_report(graph: DirectedGraph, sites: Sites, path_support: BoolArray) -> dict[str, Any]:
    """Size the strongly connected component around each named boundary probe.

    A large mutual component is the structural reason a two-site configuration
    can already saturate reachability: inside such a component every node reaches
    every other node in both directions. Equality with the path support is
    recorded as a set comparison, not inferred from two matching cardinalities.
    """

    report: dict[str, Any] = {
        "note": (
            "component of one probe node computed as the intersection of its"
            " downstream and upstream closures; no full SCC decomposition was run."
            " equals_path_support_set is an elementwise mask comparison, so a match"
            " is genuine set equality rather than equal cardinality"
        )
    }
    for role in sorted(sites):
        probe = sites[role][0]
        index: IntArray = np.asarray([probe["dense_index"]], dtype=np.int64)
        mutual = (forward_distances(graph, index) >= 0) & (backward_distances(graph, index) >= 0)
        size = int(np.count_nonzero(mutual))
        report[role] = {
            "probe_body_id": probe["body_id"],
            "probe_dense_index": probe["dense_index"],
            "strongly_connected_component": size,
            "strongly_connected_component_fraction": size / graph.node_count,
            "equals_path_support_set": bool(np.array_equal(mutual, path_support)),
            "symmetric_difference_with_path_support": int(np.count_nonzero(mutual != path_support)),
        }
    return report
