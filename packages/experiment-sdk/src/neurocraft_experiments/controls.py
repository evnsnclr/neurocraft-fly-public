"""Deterministic graph controls that cannot masquerade as faithful topology."""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, order=True, slots=True)
class Edge:
    pre: int
    post: int
    weight: int
    source_row: int | None

    def __post_init__(self) -> None:
        if self.pre < 0 or self.post < 0 or self.pre == self.post:
            raise ValueError("control edges require distinct non-negative endpoints")
        if self.weight <= 0:
            raise ValueError("edge weights must be positive")
        if self.source_row is not None and self.source_row < 0:
            raise ValueError("source rows must be non-negative")


def _validated(edges: Iterable[Edge]) -> tuple[Edge, ...]:
    ordered = tuple(sorted(edges, key=lambda edge: (edge.pre, edge.post)))
    pairs = [(edge.pre, edge.post) for edge in ordered]
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate directed edge")
    return ordered


def shuffled_weights(edges: Iterable[Edge], *, seed: int) -> tuple[Edge, ...]:
    """Reassign weights across fixed directed support without editing source rows."""

    ordered = _validated(edges)
    weights = [edge.weight for edge in ordered]
    random.Random(seed).shuffle(weights)
    return tuple(
        Edge(edge.pre, edge.post, weight, edge.source_row)
        for edge, weight in zip(ordered, weights, strict=True)
    )


def degree_preserving_rewire(
    edges: Iterable[Edge], *, seed: int, swaps: int, max_attempt_factor: int = 50
) -> tuple[Edge, ...]:
    """Directed double-edge swaps preserving every node's in/out degree and weights."""

    if swaps < 0:
        raise ValueError("swaps must be non-negative")
    current = list(_validated(edges))
    if swaps and len(current) < 2:
        raise ValueError("at least two edges are required for rewiring")
    rng = random.Random(seed)
    completed = 0
    attempts = 0
    max_attempts = max(1, swaps * max_attempt_factor)
    while completed < swaps and attempts < max_attempts:
        attempts += 1
        first_index, second_index = rng.sample(range(len(current)), 2)
        first = current[first_index]
        second = current[second_index]
        candidate_a = (first.pre, second.post)
        candidate_b = (second.pre, first.post)
        if candidate_a[0] == candidate_a[1] or candidate_b[0] == candidate_b[1]:
            continue
        untouched = {
            (edge.pre, edge.post)
            for index, edge in enumerate(current)
            if index not in (first_index, second_index)
        }
        if candidate_a == candidate_b or candidate_a in untouched or candidate_b in untouched:
            continue
        current[first_index] = Edge(first.pre, second.post, first.weight, None)
        current[second_index] = Edge(second.pre, first.post, second.weight, None)
        completed += 1
    if completed != swaps:
        raise ValueError(f"only {completed} of {swaps} requested degree-preserving swaps succeeded")
    return _validated(current)


def lesion_mask(body_ids: Iterable[int], lesioned_body_ids: Iterable[int]) -> tuple[bool, ...]:
    """Return a stable active mask; unknown lesion identifiers fail closed."""

    ordered = tuple(body_ids)
    if len(set(ordered)) != len(ordered):
        raise ValueError("body IDs must be unique")
    lesions = set(lesioned_body_ids)
    missing = lesions.difference(ordered)
    if missing:
        raise ValueError(f"unknown lesion body IDs: {sorted(missing)}")
    return tuple(body_id not in lesions for body_id in ordered)
