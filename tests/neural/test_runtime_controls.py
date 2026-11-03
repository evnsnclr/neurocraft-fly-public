from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest
from neurocraft_neural import (
    NeuralConfigurationError,
    NeuralEngine,
    NeuralIntegrityError,
    SignOverlay,
    Stimulation,
    TopologyOverlay,
    WeightOverlay,
)


def _degrees(indices: tuple[int, ...], node_count: int) -> tuple[int, ...]:
    counts = [0] * node_count
    for index in indices:
        counts[index] += 1
    return tuple(counts)


def test_shuffled_weight_control_is_deterministic_and_source_traceable(
    male_fixture_graph,
) -> None:
    provenance = "control:fixture-global-weight-permutation"
    first = WeightOverlay.shuffled(male_fixture_graph, seed=20260817, provenance=provenance)
    second = WeightOverlay.shuffled(male_fixture_graph, seed=20260817, provenance=provenance)
    other = WeightOverlay.shuffled(male_fixture_graph, seed=20260818, provenance=provenance)
    source_counts = tuple(edge.raw_count for edge in male_fixture_graph.edges)

    assert first == second
    assert first.identity_sha256 != other.identity_sha256
    assert first.identity_sha256 != "0" * 64
    assert first.provenance == provenance
    assert sorted(first.values) == sorted(source_counts)
    assert sorted(first.source_edge_indices) == list(range(len(source_counts)))
    assert all(
        first.values[target_index] == male_fixture_graph.edges[source_index].raw_count
        for target_index, source_index in enumerate(first.source_edge_indices)
    )
    first.assert_compatible(male_fixture_graph)
    male_fixture_graph.assert_integrity()


def test_engine_uses_and_binds_weight_control_without_mutating_graph(
    male_fixture_graph,
) -> None:
    source_raw_counts_sha256 = male_fixture_graph.raw_counts_sha256
    weights = WeightOverlay.shuffled(
        male_fixture_graph,
        seed=7,
        provenance="control:fixture-global-weight-permutation",
    )
    signs = SignOverlay.explicit(
        male_fixture_graph,
        [1] * len(male_fixture_graph.edges),
        ["test-positive"] * len(male_fixture_graph.edges),
    )
    engine = NeuralEngine(
        male_fixture_graph,
        sign_overlay=signs,
        weight_overlay=weights,
    )
    expected_denominators = [0] * len(male_fixture_graph.nodes)
    for edge, weight in zip(male_fixture_graph.edges, weights.values, strict=True):
        expected_denominators[edge.post_index] += weight

    assert engine.incoming_denominators == tuple(expected_denominators)
    state = engine.current_state()
    assert state.weight_overlay_sha256 == weights.identity_sha256
    assert state.topology_overlay_sha256 == "0" * 64
    snapshot = engine.snapshot()
    assert snapshot.weight_overlay_sha256 == weights.identity_sha256
    assert male_fixture_graph.raw_counts_sha256 == source_raw_counts_sha256

    canonical = NeuralEngine(male_fixture_graph, sign_overlay=signs)
    with pytest.raises(NeuralIntegrityError, match="snapshot identity mismatch: weight"):
        canonical.restore(snapshot)


def test_degree_preserving_topology_control_is_deterministic_and_exact(
    male_fixture_graph,
) -> None:
    provenance = "control:fixture-directed-degree-preserving-random"
    first = TopologyOverlay.degree_preserving_random(
        male_fixture_graph, seed=20260817, provenance=provenance
    )
    second = TopologyOverlay.degree_preserving_random(
        male_fixture_graph, seed=20260817, provenance=provenance
    )
    source_pre = tuple(edge.pre_index for edge in male_fixture_graph.edges)
    source_post = tuple(edge.post_index for edge in male_fixture_graph.edges)
    source_pairs = tuple(zip(source_pre, source_post, strict=True))
    control_pairs = tuple(zip(first.pre_indices, first.post_indices, strict=True))

    assert first == second
    assert first.identity_sha256 != "0" * 64
    assert first.provenance == provenance
    assert first.accepted_swaps == first.requested_swaps == len(male_fixture_graph.edges)
    assert first.attempted_swaps >= first.accepted_swaps
    assert control_pairs != source_pairs
    assert len(set(control_pairs)) == len(control_pairs)
    assert _degrees(first.pre_indices, len(male_fixture_graph.nodes)) == _degrees(
        source_pre, len(male_fixture_graph.nodes)
    )
    assert _degrees(first.post_indices, len(male_fixture_graph.nodes)) == _degrees(
        source_post, len(male_fixture_graph.nodes)
    )
    assert sum(pre == post for pre, post in control_pairs) == sum(
        pre == post for pre, post in source_pairs
    )
    first.assert_compatible(male_fixture_graph)
    male_fixture_graph.assert_integrity()


def test_engine_uses_and_binds_topology_control(male_fixture_graph) -> None:
    topology = TopologyOverlay.degree_preserving_random(
        male_fixture_graph,
        seed=11,
        provenance="control:fixture-directed-degree-preserving-random",
    )
    signs = SignOverlay.explicit(
        male_fixture_graph,
        [1] * len(male_fixture_graph.edges),
        ["test-positive"] * len(male_fixture_graph.edges),
    )
    engine = NeuralEngine(
        male_fixture_graph,
        sign_overlay=signs,
        topology_overlay=topology,
    )
    expected_denominators = [0] * len(male_fixture_graph.nodes)
    for post_index, edge in zip(topology.post_indices, male_fixture_graph.edges, strict=True):
        expected_denominators[post_index] += edge.raw_count

    assert engine.incoming_denominators == tuple(expected_denominators)
    assert engine.current_state().topology_overlay_sha256 == topology.identity_sha256
    assert engine.snapshot().topology_overlay_sha256 == topology.identity_sha256


def test_combined_controls_have_replayable_distinct_trace_identity(
    male_fixture_graph,
) -> None:
    weights = WeightOverlay.shuffled(
        male_fixture_graph,
        seed=19,
        provenance="control:fixture-global-weight-permutation",
    )
    topology = TopologyOverlay.degree_preserving_random(
        male_fixture_graph,
        seed=23,
        provenance="control:fixture-directed-degree-preserving-random",
    )
    signs = SignOverlay.explicit(
        male_fixture_graph,
        [1] * len(male_fixture_graph.edges),
        ["test-positive"] * len(male_fixture_graph.edges),
    )
    pulse = Stimulation.from_body_values(male_fixture_graph, "control-probe", {77185: 0.75})

    def execute(*, controlled: bool) -> tuple[str, tuple[object, ...]]:
        engine = NeuralEngine(
            male_fixture_graph,
            sign_overlay=signs,
            weight_overlay=weights if controlled else None,
            topology_overlay=topology if controlled else None,
        )
        states = tuple(
            engine.step(stimulation=pulse if step == 0 else None).state for step in range(5)
        )
        return engine.trace_sha256, states

    first = execute(controlled=True)
    second = execute(controlled=True)
    canonical = execute(controlled=False)
    assert first == second
    assert first != canonical


def test_control_tampering_invalid_parameters_and_immutability_are_refused(
    male_fixture_graph, two_node_graph
) -> None:
    weights = WeightOverlay.shuffled(
        male_fixture_graph,
        seed=3,
        provenance="control:fixture-global-weight-permutation",
    )
    topology = TopologyOverlay.degree_preserving_random(
        male_fixture_graph,
        seed=5,
        provenance="control:fixture-directed-degree-preserving-random",
    )
    with pytest.raises(NeuralIntegrityError, match="weight overlay identity mismatch"):
        replace(weights, values=tuple(reversed(weights.values))).assert_compatible(
            male_fixture_graph
        )
    with pytest.raises(NeuralIntegrityError, match="topology overlay identity mismatch"):
        replace(topology, identity_sha256="f" * 64).assert_compatible(male_fixture_graph)
    with pytest.raises(NeuralConfigurationError, match="seed must be a uint64"):
        WeightOverlay.shuffled(male_fixture_graph, seed=-1, provenance="invalid")
    with pytest.raises(NeuralConfigurationError, match="named provenance"):
        WeightOverlay.shuffled(male_fixture_graph, seed=1, provenance="")
    with pytest.raises(NeuralConfigurationError, match="at least two source edges"):
        TopologyOverlay.degree_preserving_random(
            two_node_graph, seed=1, provenance="not-enough-edges"
        )
    with pytest.raises(FrozenInstanceError):
        weights.seed = 99  # type: ignore[misc]
