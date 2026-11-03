"""Reachability analysis must be exact on graphs whose answers are hand-checkable.

Every structural test builds a small synthetic graph, so the suite never needs
the ignored full artifact. The two tests that touch the real artifact are
skipped when it is absent, which is the normal state on CI.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from neurocraft_connectome import load_full_runtime_artifact
from neurocraft_connectome.model import canonical_json_bytes

from scripts.analyze_reachability import (
    DEFAULT_ARTIFACT_DIR,
    NEURAL_STEPS_PER_GAME_TICK,
    TRACKED_MANIFEST_SHA256,
    ArtifactFacts,
    adapter_coverage,
    analyse,
    attrition_report,
)
from scripts.reachability_graph import (
    DirectedGraph,
    WeightedEdges,
    assert_post_major_incoming,
    backward_distances,
    build_csr,
    forward_distances,
    graph_from_edges,
    graph_from_full_runtime,
    mapped_numpy,
    traverse,
)
from scripts.reachability_measures import (
    baseline_report,
    ceiling_report,
    core_report,
    degeneracy_report,
    degree_summary,
    fresh_input_timing,
    hop_profile,
    latency_report,
    path_support_counts,
    scaling_points,
    select_sites,
    site_permutation,
    structure_report,
    weight_sensitivity_report,
)

ROOT = Path(__file__).resolve().parents[2]

# These are reviewed external pins, not identities authorized by the local
# artifact itself. The portable builder changes implementation/manifest identity
# while preserving every structural and lineage array from the historical graph.
HISTORICAL_ARTIFACT_MANIFEST_SHA256 = (
    "b33bdda8b1a9abfc6b23f7346197442546f7a2aec0a03fc7c07fbdfe4f67c98c"
)
PORTABLE_ARTIFACT_MANIFEST_SHA256 = (
    "0485f7fe5aeede0a23e71d8191825d5b8129abc46a38c10563c94c604a840546"
)
REVIEWED_ARTIFACT_MANIFEST_SHA256 = frozenset(
    (HISTORICAL_ARTIFACT_MANIFEST_SHA256, PORTABLE_ARTIFACT_MANIFEST_SHA256)
)


def reviewed_full_artifact_manifest_sha256(artifact: Path) -> str:
    """Choose between explicit reviewed pins; refuse every unknown local graph."""
    observed = hashlib.sha256((artifact / "manifest.json").read_bytes()).hexdigest()
    if observed not in REVIEWED_ARTIFACT_MANIFEST_SHA256:
        raise AssertionError(
            f"full artifact manifest is not an explicitly reviewed pin: {observed}"
        )
    return observed


def test_historical_analysis_defaults_retain_the_recorded_artifact_pin() -> None:
    from scripts import analyze_readout, analyze_vision_feasibility

    assert TRACKED_MANIFEST_SHA256 == HISTORICAL_ARTIFACT_MANIFEST_SHA256
    assert analyze_readout.TRACKED_MANIFEST_SHA256 == HISTORICAL_ARTIFACT_MANIFEST_SHA256
    assert analyze_vision_feasibility.TRACKED_MANIFEST_SHA256 == HISTORICAL_ARTIFACT_MANIFEST_SHA256


def test_artifact_pin_selection_refuses_an_unknown_local_manifest(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_bytes(b"{}\n")
    with pytest.raises(AssertionError, match="not an explicitly reviewed pin"):
        reviewed_full_artifact_manifest_sha256(tmp_path)


# 0 -> 1 -> 2 -> 3, 4 -> 2, 5 isolated, 6 -> 6 self, 7 <-> 8 pure cycle.
CHAIN_PRE = np.array([0, 1, 2, 4, 6, 7, 8], dtype=np.uint32)
CHAIN_POST = np.array([1, 2, 3, 2, 6, 8, 7], dtype=np.uint32)
CHAIN_WEIGHTS = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.uint32)
CHAIN_NODES = 9
SITES: dict[str, list[dict[str, Any]]] = {
    "motor": [{"body_id": 200, "dense_index": 3, "superclass": "vnc_motor"}],
    "sensory": [{"body_id": 100, "dense_index": 0, "superclass": "cb_sensory"}],
}
# Synthetic manifest counts with the same internal relations as the real artifact.
CHAIN_COUNTS = {
    "retained_edge_rows": 7,
    "excluded_pre_only_rows": 30,
    "excluded_post_only_rows": 3,
    "excluded_neither_endpoint_rows": 1,
    "raw_weight_sum": 100,
    "retained_weight_sum": 28,
}
CHAIN_SUPERCLASSES = {"cb_sensory": 4, "vnc_motor": 2}
CHAIN_FACTS = ArtifactFacts(
    identity={"artifact_id": "synthetic", "node_count": CHAIN_NODES},
    sites=SITES,
    counts=CHAIN_COUNTS,
    retained_by_superclass=CHAIN_SUPERCLASSES,
)


@pytest.fixture
def chain() -> DirectedGraph:
    return graph_from_edges(CHAIN_NODES, CHAIN_PRE, CHAIN_POST)


@pytest.fixture
def chain_edges() -> WeightedEdges:
    return WeightedEdges(
        node_count=CHAIN_NODES, pre=CHAIN_PRE, post=CHAIN_POST, weights=CHAIN_WEIGHTS
    )


def _brute_force_reachable(
    node_count: int, pre: np.ndarray[Any, np.dtype[Any]], post: np.ndarray[Any, np.dtype[Any]]
) -> dict[int, set[int]]:
    """Independent transitive closure so the CSR BFS cannot mark its own homework."""

    closure: dict[int, set[int]] = {}
    for start in range(node_count):
        seen = {start}
        pending = [start]
        while pending:
            node = pending.pop()
            for index in range(pre.size):
                if int(pre[index]) == node and int(post[index]) not in seen:
                    seen.add(int(post[index]))
                    pending.append(int(post[index]))
        closure[start] = seen
    return closure


def test_build_csr_groups_payload_and_rejects_bad_input() -> None:
    offsets, grouped = build_csr(CHAIN_PRE, CHAIN_POST, CHAIN_NODES)
    assert offsets.tolist() == [0, 1, 2, 3, 3, 4, 4, 5, 6, 7]
    assert grouped.tolist() == [1, 2, 3, 2, 6, 8, 7]
    assert int(offsets[-1]) == int(CHAIN_PRE.size)
    with pytest.raises(ValueError, match="equal length"):
        build_csr(CHAIN_PRE, CHAIN_POST[:-1], CHAIN_NODES)
    with pytest.raises(ValueError, match="out of range"):
        build_csr(CHAIN_PRE, CHAIN_POST, 8)


def test_incoming_csr_matches_the_artifact_post_major_convention(chain: DirectedGraph) -> None:
    """The artifact ships this CSR; rebuilding it must reproduce it exactly."""

    order = np.lexsort((CHAIN_PRE, CHAIN_POST))
    incoming = [0] * CHAIN_NODES
    for target in CHAIN_POST.tolist():
        incoming[target] += 1
    expected_offsets = [0]
    for degree in incoming:
        expected_offsets.append(expected_offsets[-1] + degree)
    assert chain.in_offsets.tolist() == expected_offsets
    assert chain.in_sources.tolist() == CHAIN_PRE[order].tolist()
    assert chain.self_edges == 1


def test_post_major_assumption_is_asserted_not_assumed() -> None:
    """Reusing the shipped incoming CSR unpermuted requires both invariants."""

    post = np.array([0, 0, 1, 3, 3, 3], dtype=np.uint32)
    offsets = np.array([0, 2, 3, 3, 6], dtype=np.int64)
    assert assert_post_major_incoming(post, offsets, 4) is None
    descending = np.array([1, 0], dtype=np.uint32)
    # The unsigned view is exactly why the check cannot be written with np.diff:
    # a genuine decrease wraps to a large positive difference and slips through.
    assert bool(np.all(np.diff(descending) >= 0))
    with pytest.raises(ValueError, match="non-decreasing"):
        assert_post_major_incoming(descending, np.array([0, 1, 2], dtype=np.int64), 2)
    with pytest.raises(ValueError, match="degree histogram"):
        assert_post_major_incoming(post, np.array([0, 1, 2, 3, 6], dtype=np.int64), 4)
    with pytest.raises(ValueError, match="row pointers"):
        assert_post_major_incoming(post, offsets[:-1], 4)


def test_forward_and_backward_distances_are_exact(chain: DirectedGraph) -> None:
    forward = forward_distances(chain, np.array([0]))
    assert forward.tolist() == [0, 1, 2, 3, -1, -1, -1, -1, -1]
    backward = backward_distances(chain, np.array([3]))
    assert backward.tolist() == [3, 2, 1, 0, 2, -1, -1, -1, -1]
    cycle = forward_distances(chain, np.array([7]))
    assert cycle.tolist() == [-1, -1, -1, -1, -1, -1, -1, 0, 1]
    isolated = forward_distances(chain, np.array([5]))
    assert int(np.count_nonzero(isolated >= 0)) == 1
    self_loop = forward_distances(chain, np.array([6]))
    assert self_loop.tolist()[6] == 0


def test_traversal_matches_a_brute_force_closure_on_a_random_graph() -> None:
    generator = np.random.default_rng(11)
    node_count = 40
    pre = generator.integers(0, node_count, size=90, dtype=np.uint32)
    post = generator.integers(0, node_count, size=90, dtype=np.uint32)
    graph = graph_from_edges(node_count, pre, post)
    closure = _brute_force_reachable(node_count, pre, post)
    for start in range(node_count):
        downstream = forward_distances(graph, np.array([start]))
        assert set(np.flatnonzero(downstream >= 0).tolist()) == closure[start]
    for target in range(node_count):
        upstream = set(np.flatnonzero(backward_distances(graph, np.array([target])) >= 0).tolist())
        assert upstream == {node for node, seen in closure.items() if target in seen}


def test_gather_budget_does_not_change_the_answer() -> None:
    generator = np.random.default_rng(3)
    node_count = 200
    pre = generator.integers(0, node_count, size=1200, dtype=np.uint32)
    post = generator.integers(0, node_count, size=1200, dtype=np.uint32)
    graph = graph_from_edges(node_count, pre, post)
    seeds = np.array([0, 7, 199])
    reference = traverse(graph.out_offsets, graph.out_targets, seeds, node_count)
    for budget in (1, 2, 17, 10_000):
        chunked = traverse(graph.out_offsets, graph.out_targets, seeds, node_count, budget=budget)
        assert chunked.tolist() == reference.tolist()


def test_seeds_are_validated(chain: DirectedGraph) -> None:
    with pytest.raises(ValueError, match="out of range"):
        forward_distances(chain, np.array([CHAIN_NODES]))
    with pytest.raises(ValueError, match="out of range"):
        forward_distances(chain, np.array([-1]))
    assert forward_distances(chain, np.array([], dtype=np.int64)).tolist() == [-1] * CHAIN_NODES


def test_path_support_counts_partition_the_graph(chain: DirectedGraph) -> None:
    forward = forward_distances(chain, np.array([0])) >= 0
    backward = backward_distances(chain, np.array([3])) >= 0
    counts = path_support_counts(forward, backward, CHAIN_NODES)
    assert (counts.forward, counts.backward, counts.path_support) == (4, 5, 4)
    assert counts.forward_only == 0
    assert counts.backward_only == 1
    assert counts.untouched == 4
    assert counts.path_support + counts.forward_only + counts.backward_only + counts.untouched == (
        CHAIN_NODES
    )


def test_hop_profile_reports_coverage_hops(chain: DirectedGraph) -> None:
    profile = hop_profile(forward_distances(chain, np.array([0])))
    assert profile["reached"] == 4
    assert profile["max_hops"] == 3
    assert profile["nodes_per_hop"] == [1, 1, 1, 1]
    assert profile["hops_to_cover_reached_fraction"] == {"p50": 1, "p90": 3, "p99": 3, "p100": 3}


def test_degree_and_structure_summaries(chain: DirectedGraph) -> None:
    in_degrees = degree_summary(chain.in_degrees)
    assert (in_degrees["min"], in_degrees["max"], in_degrees["zero_count"]) == (0, 2, 3)
    structure = structure_report(chain)
    assert structure["edge_count"] == int(CHAIN_PRE.size)
    assert structure["self_edges"] == 1
    assert structure["source_nodes_zero_in_degree"] == 3
    assert structure["sink_nodes_zero_out_degree"] == 2
    assert structure["isolated_nodes"] == 1
    assert structure["interior_nodes"] == 5


def test_latency_bounds_count_only_short_enough_paths(chain: DirectedGraph) -> None:
    """Every chain node sits on the single length-3 path, so nothing appears earlier."""

    forward = forward_distances(chain, np.array([0]))
    backward = backward_distances(chain, np.array([3]))
    report = latency_report(forward, backward, CHAIN_NODES, steps_per_game_tick=5)
    lengths = [row["max_path_length"] for row in report["table"]]
    counts = [row["path_support_nodes"] for row in report["table"]]
    assert lengths == [1, 2, 3, 4, 5]
    assert counts == [0, 0, 4, 4, 4]
    assert report["table"][2]["path_support_fraction"] == 4 / CHAIN_NODES
    assert report["shortest_injection_to_readout_path"] == 3
    assert report["saturation_path_length"] == 3
    assert report["longest_shortest_path_through_a_supported_node"] == 3
    assert report["unbounded_path_support_nodes"] == 4
    assert report["one_game_tick"] == {
        "max_path_length": 5,
        "max_graph_hops": 5,
        "fresh_input_minimum_neural_updates_for_hop_budget": 11,
        "path_support_nodes": 4,
        "path_support_fraction": 4 / CHAIN_NODES,
    }
    assert report["neural_steps_per_game_tick"] == 5
    assert report["fresh_input_timing"]["max_graph_hops"] == 2
    assert report["fresh_input_one_game_tick"]["path_support_nodes"] == 0
    assert report["table"][2]["fresh_input_minimum_neural_updates_for_hop_budget"] == 7
    assert "loose historical upper bound" in report["tick_scope"]
    # The bound is the sum of the two shortest one-sided distances, node by node.
    for node in range(CHAIN_NODES):
        if forward[node] >= 0 and backward[node] >= 0:
            assert int(forward[node]) + int(backward[node]) == 3


def test_latency_table_reaches_the_tick_bound_on_a_shallow_graph() -> None:
    graph = graph_from_edges(2, np.array([0], dtype=np.uint32), np.array([1], dtype=np.uint32))
    forward = forward_distances(graph, np.array([0]))
    backward = backward_distances(graph, np.array([1]))
    report = latency_report(forward, backward, 2, steps_per_game_tick=NEURAL_STEPS_PER_GAME_TICK)
    assert report["one_game_tick"] is not None
    assert report["one_game_tick"]["max_path_length"] == NEURAL_STEPS_PER_GAME_TICK
    assert report["one_game_tick"]["path_support_nodes"] == 2
    assert report["fresh_input_one_game_tick"]["path_support_nodes"] == 2


def test_fresh_input_budget_keeps_zero_hop_paths_and_refuses_invalid_update_counts() -> None:
    report = latency_report(np.array([0]), np.array([0]), 1, steps_per_game_tick=1)
    assert report["fresh_input_one_game_tick"]["max_graph_hops"] == 0
    assert report["fresh_input_one_game_tick"]["path_support_nodes"] == 1
    budgets = [fresh_input_timing(updates)["max_graph_hops"] for updates in range(1, 8)]
    assert budgets == [0, 0, 1, 1, 2, 2, 3]
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            fresh_input_timing(invalid)


def test_reported_fresh_input_hop_budget_matches_actual_array_engine_onsets() -> None:
    from neurocraft_connectome import SyntheticEdge, SyntheticNode, build_synthetic_graph
    from neurocraft_neural import SignOverlay
    from neurocraft_neural.full_engine import ArrayNativeEngine
    from neurocraft_neural.full_view import view_from_runtime_graph
    from neurocraft_neural.spmv_kernel import NumpyReduceatKernel

    graph = build_synthetic_graph(
        "synthetic:fresh-input-timing",
        [SyntheticNode(1), SyntheticNode(2), SyntheticNode(3)],
        [SyntheticEdge(1, 2, 1, 0), SyntheticEdge(2, 3, 1, 1)],
    )
    signs = SignOverlay.explicit(graph, [1, 1], ["synthetic", "synthetic"])
    engine = ArrayNativeEngine(
        view_from_runtime_graph(graph, sign_overlay=signs), kernel=NumpyReduceatKernel()
    )
    first_nonzero = {}
    for update in range(1, 6):
        engine.step(sensory=np.array([1.0, 0.0, 0.0]))
        for hops, value in enumerate(engine.activity()):
            if value != 0.0:
                first_nonzero.setdefault(hops, update)
                assert hops <= fresh_input_timing(update)["max_graph_hops"]
    assert first_nonzero == {0: 1, 1: 3, 2: 5}


def test_weight_thresholds_remove_support_when_they_remove_edges(
    chain_edges: WeightedEdges,
) -> None:
    report = weight_sensitivity_report(chain_edges, SITES, (3, 1, 2, 2))
    rows = report["thresholds"]
    assert [row["minimum_edge_weight"] for row in rows] == [1, 2, 3]
    assert [row["retained_edges"] for row in rows] == [7, 6, 5]
    # Dropping the weight-1 edge 0 -> 1 severs the only injection -> readout path.
    assert [row["path_support_nodes"] for row in rows] == [4, 0, 0]
    assert rows[0]["path_support_fraction"] == 4 / CHAIN_NODES
    assert rows[0]["retained_edge_fraction"] == 1.0
    assert report["edge_weight_median"] == 4.0
    assert report["edge_weight_max"] == 7


def test_ceilings_exclude_cycle_only_nodes(chain: DirectedGraph) -> None:
    forward = forward_distances(chain, np.array([0])) >= 0
    backward = backward_distances(chain, np.array([3])) >= 0
    ceilings = ceiling_report(chain, forward, backward)
    assert ceilings["unbounded_injection_actual_readout"] == 5
    assert ceilings["unbounded_readout_actual_injection"] == 4
    assert ceilings["zero_in_to_zero_out_path_nodes"] == 6
    assert ceilings["nontrivial_any_to_any"] == 5
    assert ceilings["trivial_all_to_all"] == CHAIN_NODES
    assert "influence" not in canonical_json_bytes(ceilings["definitions"]).decode()


def test_core_report_compares_sets_not_cardinalities(chain: DirectedGraph) -> None:
    forward = forward_distances(chain, np.array([0])) >= 0
    backward = backward_distances(chain, np.array([3])) >= 0
    report = core_report(chain, SITES, forward & backward)
    assert report["sensory"]["strongly_connected_component"] == 1
    assert report["motor"]["strongly_connected_component"] == 1
    assert report["sensory"]["probe_body_id"] == 100
    # Equal cardinality, unequal sets: {0} and {3} both have size one.
    assert report["sensory"]["equals_path_support_set"] is False
    assert report["sensory"]["symmetric_difference_with_path_support"] == 3
    assert report["motor"]["symmetric_difference_with_path_support"] == 3
    cyclic: dict[str, list[dict[str, Any]]] = {
        "motor": [{"body_id": 8, "dense_index": 8, "superclass": "vnc_motor"}],
        "sensory": [{"body_id": 7, "dense_index": 7, "superclass": "cb_sensory"}],
    }
    cycle_support = (forward_distances(chain, np.array([7])) >= 0) & (
        backward_distances(chain, np.array([8])) >= 0
    )
    cyclic_report = core_report(chain, cyclic, cycle_support)
    assert cyclic_report["sensory"]["strongly_connected_component"] == 2
    assert cyclic_report["motor"]["strongly_connected_component_fraction"] == 2 / CHAIN_NODES
    assert cyclic_report["sensory"]["equals_path_support_set"] is True
    assert cyclic_report["motor"]["symmetric_difference_with_path_support"] == 0


def test_site_selection_is_nested_within_one_seeded_permutation() -> None:
    order = site_permutation(500, (1, 0, 1))
    assert order.tolist() == site_permutation(500, (1, 0, 1)).tolist()
    assert order.tolist() != site_permutation(500, (1, 1, 1)).tolist()
    assert order.tolist() != site_permutation(500, (1, 0, 2)).tolist()
    small = select_sites(order, 8)
    large = select_sites(order, 16)
    assert small.size == 8
    assert len(set(small.tolist())) == 8
    assert small.tolist() == sorted(small.tolist())
    assert set(small.tolist()) < set(large.tolist())
    assert select_sites(order, 500).tolist() == list(range(500))
    for bad in (0, 501):
        with pytest.raises(ValueError, match="site count"):
            select_sites(order, bad)


def test_scaling_is_monotone_inside_every_repeat_not_only_in_median() -> None:
    """Nested sampling makes each repeat individually monotone, not just the medians."""

    generator = np.random.default_rng(5)
    node_count = 300
    pre = generator.integers(0, node_count, size=900, dtype=np.uint32)
    post = generator.integers(0, node_count, size=900, dtype=np.uint32)
    graph = graph_from_edges(node_count, pre, post)
    forward = forward_distances(graph, np.array([0])) >= 0
    backward = backward_distances(graph, np.array([1])) >= 0
    points = scaling_points(
        graph,
        forward,
        backward,
        site_counts=(16, 1, 4, node_count, 4),
        repeats=3,
        base_seed=99,
    )
    assert [point.site_count for point in points] == [1, 4, 16, node_count]
    for repeat in range(3):
        for field in ("widened_injection", "widened_readout", "widened_both"):
            series = [getattr(point, field)[repeat] for point in points]
            assert series == sorted(series)
    ceiling = ceiling_report(graph, forward, backward)
    assert max(points[-1].widened_injection) <= ceiling["unbounded_injection_actual_readout"]
    assert points[-1].widened_injection[0] == ceiling["unbounded_injection_actual_readout"]
    assert points[-1].widened_readout[0] == ceiling["unbounded_readout_actual_injection"]


def test_degeneracy_marker_carries_the_derivation(chain: DirectedGraph) -> None:
    forward = forward_distances(chain, np.array([0])) >= 0
    backward = backward_distances(chain, np.array([3])) >= 0
    support = forward & backward
    points = scaling_points(chain, forward, backward, site_counts=(1, 2), repeats=2, base_seed=7)
    report = degeneracy_report(chain, SITES, support, backward, points)
    assert report["degenerate"] is True
    assert report["path_support_fraction_of_nodes"] == 4 / CHAIN_NODES
    assert report["probability_uniform_node_outside_path_support"] == 1 - 4 / CHAIN_NODES
    assert report["expected_outside_nodes_per_sample"] == [
        {"site_count": 1, "expected": 1 - 4 / CHAIN_NODES},
        {"site_count": 2, "expected": 2 * (1 - 4 / CHAIN_NODES)},
    ]
    assert set(report["observed_median_increment_over_the_swept_range"]) == {
        "widened_injection",
        "widened_readout",
        "widened_both",
    }
    assert report["median_at_smallest_site_count"]["widened_injection"] == float(
        np.median(np.asarray(points[0].widened_injection))
    )
    assert report["observed_median_increment_over_the_swept_range"]["widened_both"] == (
        report["median_at_largest_site_count"]["widened_both"]
        - report["median_at_smallest_site_count"]["widened_both"]
    )
    peripheral = report["peripheral_injection"]
    assert peripheral["available_peripheral_nodes"] == 4
    assert [row["peripheral_sites"] for row in peripheral["rows"]] == [1, 2, 4]
    # Widening with every usable outside node adds one node and then saturates.
    assert [row["path_support_nodes"] for row in peripheral["rows"]] == [5, 5, 5]


def test_adapter_coverage_reports_the_site_to_population_ratio() -> None:
    coverage = adapter_coverage(SITES, CHAIN_SUPERCLASSES)
    assert coverage["sensory"]["sites"] == 1
    assert coverage["sensory"]["superclasses"] == ["cb_sensory"]
    assert coverage["sensory"]["retained_population"] == 4
    assert coverage["sensory"]["site_fraction_of_population"] == 0.25
    assert coverage["motor"]["retained_population"] == 2
    assert coverage["motor"]["site_fraction_of_population"] == 0.5
    assert coverage["motor"]["population_by_superclass"] == {"vnc_motor": 2}


def test_attrition_direction_is_recorded_as_a_lower_bound() -> None:
    report = attrition_report(CHAIN_COUNTS)
    assert report["rows_leaving_retained_presynaptic_nodes"] == 37
    assert report["retained_fraction_of_rows_leaving_retained_presynaptic_nodes"] == 7 / 37
    assert report["rows_arriving_at_retained_postsynaptic_nodes"] == 10
    assert report["retained_fraction_of_rows_arriving_at_retained_postsynaptic_nodes"] == 0.7
    assert report["pre_only_over_post_only"] == 10.0
    assert report["retained_weight_fraction_of_raw"] == 0.28
    assert "LOWER bound" in report["direction"]


def test_analysis_body_is_deterministic_and_matches_the_baseline(
    chain: DirectedGraph, chain_edges: WeightedEdges
) -> None:
    first = analyse(chain, chain_edges, CHAIN_FACTS, site_counts=(1, 2), repeats=2, base_seed=42)
    second = analyse(chain, chain_edges, CHAIN_FACTS, site_counts=(1, 2), repeats=2, base_seed=42)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    baseline, _, _ = baseline_report(chain, SITES)
    assert first["current_configuration"]["path_support_nodes"] == baseline["path_support_nodes"]
    assert baseline["path_support_nodes"] == 4
    assert baseline["path_support_fraction"] == 4 / CHAIN_NODES
    assert baseline["hops_from_sensory_to_motor_body"] == {"200": 3}
    assert baseline["hops_from_motor_to_sensory_body"] == {"100": 3}
    assert first["schema"] == "neurocraft.analysis.reachability.v2"
    assert first["scaling"]["base_seed"] == 42
    assert first["scaling"]["degenerate"] is True
    assert first["scaling"]["site_counts"] == [1, 2]
    samples = first["scaling"]["curves"]["widened_both"][0]["samples"]
    assert [entry["repeat"] for entry in samples] == [0, 1]
    assert all(isinstance(entry["path_support_nodes"], int) for entry in samples)
    assert first["latency_bounds"]["unbounded_path_support_nodes"] == 4
    assert first["connected_core"]["sensory"]["strongly_connected_component"] == 1
    assert first["adapter_coverage"]["motor"]["site_fraction_of_population"] == 0.5
    assert (
        "LOWER bound" in (first["interpretation_bounds"]["edge_attrition_direction"]["direction"])
    )
    other = analyse(chain, chain_edges, CHAIN_FACTS, site_counts=(1, 2), repeats=2, base_seed=43)
    assert canonical_json_bytes(other) != canonical_json_bytes(first)


def test_no_reserved_causal_vocabulary_survives_in_the_report_body(
    chain: DirectedGraph, chain_edges: WeightedEdges
) -> None:
    """`causal` is reserved for the G3 gate; this analysis is pure topology."""

    body = canonical_json_bytes(
        analyse(chain, chain_edges, CHAIN_FACTS, site_counts=(1,), repeats=1, base_seed=1)
    ).decode()
    assert "causal_path" not in body
    assert body.count("causal") == 1
    assert "causal evidence in the sense docs/SCIENTIFIC_CLAIMS.md reserves for G3" in body


@pytest.mark.skipif(
    not (ROOT / DEFAULT_ARTIFACT_DIR / "manifest.json").is_file(),
    reason="the full MaleCNS runtime artifact is ignored and absent",
)
def test_full_artifact_binds_the_analysis_to_verified_bytes() -> None:
    with load_full_runtime_artifact(
        ROOT / DEFAULT_ARTIFACT_DIR,
        expected_manifest_sha256=reviewed_full_artifact_manifest_sha256(
            ROOT / DEFAULT_ARTIFACT_DIR
        ),
    ) as artifact:
        graph = graph_from_full_runtime(artifact)
        node_count = artifact.manifest.node_count
        edge_count = artifact.manifest.edge_count
        self_edge_rows = artifact.manifest.counts["retained_self_edge_rows"]
    assert (graph.node_count, graph.edge_count) == (node_count, edge_count)
    assert graph.self_edges == self_edge_rows
    assert int(graph.out_offsets[-1]) == edge_count
    assert int(graph.in_offsets[-1]) == edge_count
    assert int(graph.in_degrees.sum()) == edge_count
    assert int(graph.out_degrees.sum()) == edge_count


@pytest.mark.skipif(
    not (ROOT / DEFAULT_ARTIFACT_DIR / "manifest.json").is_file(),
    reason="the full MaleCNS runtime artifact is ignored and absent",
)
def test_full_artifact_production_transpose_equals_an_independent_build() -> None:
    """The shipped-CSR shortcut must reproduce a from-scratch build array for array.

    Degree sums and row-pointer spans are invariant under any permutation of
    ``in_sources``, so they cannot detect a wrong transpose. This compares the
    production path against ``graph_from_edges``, which sorts both directions
    itself and shares no code with the shortcut.
    """

    with load_full_runtime_artifact(
        ROOT / DEFAULT_ARTIFACT_DIR,
        expected_manifest_sha256=reviewed_full_artifact_manifest_sha256(
            ROOT / DEFAULT_ARTIFACT_DIR
        ),
    ) as artifact:
        production = graph_from_full_runtime(artifact)
        node_count = artifact.manifest.node_count
        pre = np.array(mapped_numpy(artifact.edge_pre_indices, "<u4"))
        post = np.array(mapped_numpy(artifact.edge_post_indices, "<u4"))
    independent = graph_from_edges(node_count, pre, post)
    assert np.array_equal(production.out_offsets, independent.out_offsets)
    assert np.array_equal(production.out_targets, independent.out_targets)
    assert np.array_equal(production.in_offsets, independent.in_offsets)
    assert np.array_equal(production.in_sources, independent.in_sources)
    assert production.self_edges == independent.self_edges
