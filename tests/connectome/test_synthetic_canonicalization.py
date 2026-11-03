from __future__ import annotations

from dataclasses import replace

import pytest
from neurocraft_connectome import (
    FixtureSchemaError,
    FixtureTier,
    SyntheticEdge,
    SyntheticNode,
    build_synthetic_graph,
)


def _records() -> tuple[list[SyntheticNode], list[SyntheticEdge]]:
    nodes = [
        SyntheticNode(30, ("motor",), side="R", type_name="motor"),
        SyntheticNode(10, ("sensory",), side="L", type_name="sensor"),
        SyntheticNode(20, ("intermediate",), type_name="relay"),
    ]
    edges = [
        SyntheticEdge(20, 30, 9, 102),
        SyntheticEdge(10, 20, 7, 101),
        SyntheticEdge(10, 30, 2, 103),
    ]
    return nodes, edges


def test_input_row_permutation_is_byte_identity_invariant() -> None:
    nodes, edges = _records()
    forward = build_synthetic_graph("synthetic:permutation", nodes, edges)
    reverse = build_synthetic_graph(
        "synthetic:permutation", list(reversed(nodes)), list(reversed(edges))
    )

    assert forward == reverse
    assert forward.manifest.canonical_graph_sha256 == reverse.manifest.canonical_graph_sha256
    assert forward.manifest.runtime_identity_sha256 == reverse.manifest.runtime_identity_sha256
    assert forward.manifest.fixture_tier is FixtureTier.SYNTHETIC_ALGEBRA
    assert forward.body_ids == (10, 20, 30)


def test_synthetic_identity_cannot_claim_malecns() -> None:
    nodes, edges = _records()

    with pytest.raises(FixtureSchemaError, match="must start with 'synthetic:'"):
        build_synthetic_graph("male-cns:v1.0", nodes, edges)


def test_duplicate_ids_pairs_rows_and_overflow_are_rejected() -> None:
    nodes, edges = _records()

    with pytest.raises(FixtureSchemaError, match="duplicate synthetic body ID"):
        build_synthetic_graph("synthetic:duplicate-node", [nodes[0], nodes[0]], [])
    with pytest.raises(FixtureSchemaError, match="duplicate directed pair"):
        build_synthetic_graph(
            "synthetic:duplicate-edge", nodes, [edges[0], replace(edges[0], source_row=999)]
        )
    with pytest.raises(FixtureSchemaError, match="duplicate synthetic source row"):
        build_synthetic_graph(
            "synthetic:duplicate-row", nodes, [edges[0], replace(edges[1], source_row=102)]
        )
    with pytest.raises(FixtureSchemaError, match=r"outside|inclusive range"):
        build_synthetic_graph("synthetic:overflow", [SyntheticNode(1 << 64)], [])


def test_direction_and_raw_counts_are_not_aggregated_or_reversed() -> None:
    nodes, edges = _records()
    graph = build_synthetic_graph("synthetic:direction", nodes, edges)

    assert {
        (edge.body_pre, edge.body_post, edge.raw_count, edge.source_row) for edge in graph.edges
    } == {
        (10, 20, 7, 101),
        (20, 30, 9, 102),
        (10, 30, 2, 103),
    }
    assert graph.manifest.raw_count_sum == 18
    assert graph.manifest.edge_count == 3
