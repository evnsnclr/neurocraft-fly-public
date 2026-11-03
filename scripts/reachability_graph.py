"""Owned CSR edge support and BFS traversal for the Phase 0 reachability analysis.

Split out of ``scripts/analyze_reachability.py`` so that the analysis script stays
inside the repository's 800-line file ceiling. Nothing in this module knows about
the report: it owns the directed edge support, the two traversal directions and the
weighted edge list used for threshold sensitivity, and it is imported by
``scripts.analyze_reachability`` under the same scripts-are-importable convention
that ``tests/ab11`` and ``tests/integration`` already rely on.

Direction. The artifact stores one CSR grouped by POST index (``incoming_offsets``
plus ``edge_pre_indices``), which is already the backward adjacency and is reused
as-is once :func:`assert_post_major_incoming` has proved the two invariants that
reuse depends on. The forward adjacency is obtained by transposing that structure
exactly once with a stable ``argsort`` over ``edge_pre_indices`` (radix, O(E), one
transient 25,582,938-element index array of about 205 MB) and permuting
``edge_post_indices`` into a 102 MB owned array. The rejected alternatives: a dense
adjacency matrix is 166,700 x 166,700 and impossible, and re-scanning the whole
25.6M-edge list once per BFS hop costs O(E) per hop instead of O(E) per whole
traversal, which matters because the analysis runs on the order of a hundred
traversals.

Traversal. BFS is frontier-based over CSR with int32 hop distances and a boolean
discovery mask; ragged neighbour gathers are chunked to a fixed element budget so
peak memory does not track the widest frontier.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import numpy.typing as npt
from neurocraft_connectome.full_runtime import FullRuntimeGraph, MappedUnsignedArray

IntArray = npt.NDArray[Any]
BoolArray = npt.NDArray[np.bool_]

GATHER_BUDGET = 8_000_000


# --------------------------------------------------------------- graph model


@dataclass(frozen=True, slots=True)
class DirectedGraph:
    """Owned forward and backward CSR views over one dense-index edge support."""

    node_count: int
    edge_count: int
    self_edges: int
    out_offsets: IntArray
    out_targets: IntArray
    in_offsets: IntArray
    in_sources: IntArray

    @property
    def out_degrees(self) -> IntArray:
        degrees: IntArray = self.out_offsets[1:] - self.out_offsets[:-1]
        return degrees

    @property
    def in_degrees(self) -> IntArray:
        degrees: IntArray = self.in_offsets[1:] - self.in_offsets[:-1]
        return degrees


@dataclass(frozen=True, slots=True)
class WeightedEdges:
    """Owned raw directed edge list plus its retained synapse counts.

    Kept alongside :class:`DirectedGraph` rather than inside it because only the
    weight-threshold sensitivity needs the unsorted endpoints and the weights; the
    traversals never look at a weight.
    """

    node_count: int
    pre: IntArray
    post: IntArray
    weights: IntArray

    @property
    def edge_count(self) -> int:
        return int(self.pre.size)

    def subgraph(self, minimum_weight: int) -> DirectedGraph:
        """Rebuild both CSR directions over edges of at least ``minimum_weight``."""

        keep = self.weights >= minimum_weight
        return graph_from_edges(self.node_count, self.pre[keep], self.post[keep])


def build_csr(groups: IntArray, payload: IntArray, node_count: int) -> tuple[IntArray, IntArray]:
    """Group ``payload`` by ``groups`` into CSR offsets and a permuted payload."""

    if groups.shape != payload.shape:
        raise ValueError("edge group and payload arrays must have equal length")
    if groups.size and int(groups.max()) >= node_count:
        raise ValueError("edge endpoint dense index is out of range")
    offsets = np.zeros(node_count + 1, dtype=np.int64)
    np.cumsum(np.bincount(groups, minlength=node_count), out=offsets[1:])
    order = np.argsort(groups, kind="stable")
    grouped: IntArray = payload[order]
    return offsets, grouped


def graph_from_edges(node_count: int, pre: IntArray, post: IntArray) -> DirectedGraph:
    """Build both CSR directions from a raw directed edge list."""

    out_offsets, out_targets = build_csr(pre, post, node_count)
    in_offsets, in_sources = build_csr(post, pre, node_count)
    return DirectedGraph(
        node_count=node_count,
        edge_count=int(pre.size),
        self_edges=int(np.count_nonzero(pre == post)),
        out_offsets=out_offsets,
        out_targets=out_targets,
        in_offsets=in_offsets,
        in_sources=in_sources,
    )


def mapped_numpy(array: MappedUnsignedArray, dtype: str) -> IntArray:
    """View the exact mapping the loader hashed; the loader stays sole verifier.

    Reading the private mapping is deliberate: reopening the path here would
    reintroduce the verify-then-reopen window that ``MappedUnsignedArray``
    exists to close.
    """

    values: IntArray = np.frombuffer(array._mmap, dtype=dtype, count=len(array))
    return values


def assert_post_major_incoming(post: IntArray, incoming_offsets: IntArray, node_count: int) -> None:
    """Prove the two invariants that let the shipped incoming CSR be reused as-is.

    The manifest declares ``edge_order = [post_dense_index, pre_dense_index]`` and
    the loader hashes the bytes, but nothing in the loader proves the ordering
    claim itself. Reusing ``edge_pre_indices`` unpermuted as the backward CSR
    payload is only correct if POST is already non-decreasing *and* the shipped row
    pointers are exactly the cumulative POST degree histogram; a row-pointer span
    check alone cannot tell a correct CSR from an arbitrarily permuted one. Both
    checks below are vectorised over the whole 25.6M-row edge list.

    The monotonicity test compares adjacent slices rather than using
    ``numpy.diff``: ``post`` is an unsigned view, so a genuine decrease would wrap
    to a large positive difference and ``diff(post) >= 0`` would silently pass.
    """

    if incoming_offsets.size != node_count + 1:
        raise ValueError(
            "invariant violated: incoming_offsets must hold node_count + 1 row pointers"
        )
    if post.size and not bool(np.all(post[1:] >= post[:-1])):
        raise ValueError(
            "invariant violated: edge_post_indices is not non-decreasing, so"
            " edge_pre_indices is not already grouped by POST and must not be"
            " reused as the backward CSR payload"
        )
    expected = np.zeros(node_count + 1, dtype=np.int64)
    np.cumsum(np.bincount(post, minlength=node_count), out=expected[1:])
    if expected.size != incoming_offsets.size or not bool(
        np.array_equal(expected, incoming_offsets)
    ):
        raise ValueError(
            "invariant violated: the shipped incoming_offsets are not the cumulative"
            " POST degree histogram of edge_post_indices"
        )


def graph_from_full_runtime(graph: FullRuntimeGraph) -> DirectedGraph:
    """Own copies of the verified arrays so the artifact mapping can be closed."""

    node_count = graph.manifest.node_count
    edge_count = graph.manifest.edge_count
    pre = mapped_numpy(graph.edge_pre_indices, "<u4")
    post = mapped_numpy(graph.edge_post_indices, "<u4")
    if pre.size != edge_count or post.size != edge_count:
        raise ValueError("edge endpoint arrays do not match the manifest edge count")
    in_offsets = mapped_numpy(graph.incoming_offsets, "<u8").astype(np.int64)
    assert_post_major_incoming(post, in_offsets, node_count)
    if int(in_offsets[-1]) != edge_count:
        raise ValueError("incoming CSR row pointers do not span the retained edge count")
    out_offsets, out_targets = build_csr(pre, post, node_count)
    in_sources = np.array(pre, dtype=np.uint32)
    return DirectedGraph(
        node_count=node_count,
        edge_count=edge_count,
        self_edges=int(np.count_nonzero(pre == post)),
        out_offsets=out_offsets,
        out_targets=out_targets,
        in_offsets=in_offsets,
        in_sources=in_sources,
    )


def weighted_edges_from_full_runtime(graph: FullRuntimeGraph) -> WeightedEdges:
    """Own the unsorted endpoints and weights the threshold sensitivity needs."""

    manifest = graph.manifest
    weights = mapped_numpy(graph.edge_raw_counts, "<u8").astype(np.uint32)
    if int(weights.sum(dtype=np.uint64)) != manifest.raw_count_sum:
        raise ValueError("retained raw counts do not sum to the manifest raw_count_sum")
    return WeightedEdges(
        node_count=manifest.node_count,
        pre=np.array(mapped_numpy(graph.edge_pre_indices, "<u4")),
        post=np.array(mapped_numpy(graph.edge_post_indices, "<u4")),
        weights=weights,
    )


# ----------------------------------------------------------------- traversal


def _chunk_boundaries(counts: IntArray, budget: int) -> tuple[int, ...]:
    """Split a frontier so no single ragged gather exceeds ``budget`` elements."""

    if counts.size == 0:
        return (0,)
    cumulative = np.cumsum(counts)
    boundaries = [0]
    while boundaries[-1] < counts.size:
        start = boundaries[-1]
        consumed = int(cumulative[start - 1]) if start else 0
        stop = int(np.searchsorted(cumulative, consumed + budget, side="right"))
        boundaries.append(min(max(stop, start + 1), int(counts.size)))
    return tuple(boundaries)


def _gather_neighbours(
    offsets: IntArray, neighbours: IntArray, frontier: IntArray, budget: int
) -> Iterator[IntArray]:
    """Yield the neighbour blocks of ``frontier`` within a fixed element budget."""

    starts = offsets[frontier].astype(np.int64)
    counts = offsets[frontier + 1].astype(np.int64) - starts
    for begin, end in pairwise(_chunk_boundaries(counts, budget)):
        block = counts[begin:end]
        total = int(block.sum())
        if total == 0:
            continue
        prefix = np.cumsum(block) - block
        index = np.repeat(starts[begin:end] - prefix, block) + np.arange(total, dtype=np.int64)
        gathered: IntArray = neighbours[index]
        yield gathered


def traverse(
    offsets: IntArray,
    neighbours: IntArray,
    seeds: IntArray,
    node_count: int,
    *,
    budget: int = GATHER_BUDGET,
) -> IntArray:
    """Return int32 BFS hop distances from ``seeds``; -1 marks an unreached node."""

    distances = np.full(node_count, -1, dtype=np.int32)
    frontier: IntArray = np.unique(np.asarray(seeds, dtype=np.int64))
    if frontier.size and (int(frontier[0]) < 0 or int(frontier[-1]) >= node_count):
        raise ValueError("seed dense index is out of range")
    distances[frontier] = 0
    discovered = np.zeros(node_count, dtype=bool)
    hop = 0
    while frontier.size:
        hop += 1
        discovered[:] = False
        found = False
        for block in _gather_neighbours(offsets, neighbours, frontier, budget):
            candidates = block[distances[block] < 0]
            if candidates.size:
                discovered[candidates] = True
                found = True
        if not found:
            break
        frontier = np.flatnonzero(discovered)
        distances[frontier] = hop
    return distances


def forward_distances(graph: DirectedGraph, seeds: IntArray) -> IntArray:
    """Hop distances downstream of ``seeds`` over the retained edge support."""

    return traverse(graph.out_offsets, graph.out_targets, seeds, graph.node_count)


def backward_distances(graph: DirectedGraph, seeds: IntArray) -> IntArray:
    """Hop distances upstream of ``seeds`` over the retained edge support."""

    return traverse(graph.in_offsets, graph.in_sources, seeds, graph.node_count)
