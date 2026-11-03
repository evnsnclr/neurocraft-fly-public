from __future__ import annotations

from collections import Counter

import pytest
from neurocraft_experiments.controls import (
    Edge,
    degree_preserving_rewire,
    lesion_mask,
    shuffled_weights,
)

EDGES = (
    Edge(1, 2, 10, 100),
    Edge(1, 3, 20, 101),
    Edge(2, 4, 30, 102),
    Edge(3, 4, 40, 103),
    Edge(4, 1, 50, 104),
)


def degrees(edges: tuple[Edge, ...]) -> tuple[Counter[int], Counter[int]]:
    return Counter(edge.pre for edge in edges), Counter(edge.post for edge in edges)


def test_weight_shuffle_preserves_support_rows_and_multiset() -> None:
    shuffled = shuffled_weights(EDGES, seed=19)
    assert [(edge.pre, edge.post, edge.source_row) for edge in shuffled] == [
        (edge.pre, edge.post, edge.source_row) for edge in EDGES
    ]
    assert sorted(edge.weight for edge in shuffled) == sorted(edge.weight for edge in EDGES)
    assert shuffled == shuffled_weights(EDGES, seed=19)


def test_degree_preserving_rewire_is_deterministic_and_drops_source_lineage() -> None:
    rewired = degree_preserving_rewire(EDGES, seed=3, swaps=2)
    assert degrees(rewired) == degrees(EDGES)
    assert rewired == degree_preserving_rewire(EDGES, seed=3, swaps=2)
    assert all(edge.source_row is None for edge in rewired if edge not in EDGES)


def test_lesion_mask_fails_on_unknown_body() -> None:
    assert lesion_mask((10, 20, 30), (20,)) == (True, False, True)
    with pytest.raises(ValueError, match="unknown lesion"):
        lesion_mask((10, 20), (99,))


def test_duplicate_edges_fail() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        shuffled_weights((*EDGES, EDGES[0]), seed=1)
