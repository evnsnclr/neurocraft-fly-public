"""Circuit resolution for the MaleCNS v1.0 flight-steering fixture.

This module owns the frozen selection policy: the exact descending types, the
pinned sensory entries, the direct wing-motor readout rule, the maximum-bottleneck
path-support objective, and the weight-thresholded closure. It reads only
validated arrays produced by :mod:`scripts.flight_fixture_sources` and emits
plain immutable values; JSON shaping and provenance live in
``scripts/extract_flight_fixture.py``.

Every threshold here is a declared constant, and the whole policy is hashed into
the fixture through ``selection_policy()`` in the extractor.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from scripts.flight_fixture_sources import (
    AnnotationRecord,
    CircuitResolutionError,
    IntArray,
    RetainedWeights,
)

#: Exact descending types that define the steering drive onto the wing motors.
DESCENDING_TYPES = (
    "DNg02_a",
    "DNg02_b",
    "DNg02_c",
    "DNg02_d",
    "DNg02_e",
    "DNg02_f",
    "DNg02_g",
    "DNp03",
    "DNp31",
)
#: The two pinned sensory entries, shared with the flight-power fixture.
SENSORY_BODY_IDS = (77185, 805722)
MOTOR_SUPERCLASS = "vnc_motor"
MOTOR_SUBCLASS = "wm"
#: Inclusive minimum weight of a direct descending-to-wing-motor contact.
READOUT_MIN_WEIGHT = 5
#: Inclusive minimum weight of every edge on a sensory-to-descending path.
PATH_MIN_WEIGHT = 10
#: Inclusive minimum weight of an emitted closure edge.
INDUCED_MIN_WEIGHT = 5
#: Maximum number of source edges on one selected sensory-to-descending path.
MAX_PATH_HOPS = 3
#: Exclusive emitted-edge ceiling; the controller re-verifies O(E) work per step.
MAX_EMITTED_EDGES = 2000

INSTANCE_SIDE_RE = re.compile(r"_([LR])\Z")
ROLE_ORDER = ("sensory", "descending", "motor", "intermediate")


@dataclass(frozen=True, slots=True)
class Edge:
    """One directed source edge with its zero-based weight-table row."""

    body_pre: int
    body_post: int
    weight: int
    source_row: int


@dataclass(frozen=True, slots=True)
class SelectedPath:
    """The chosen sensory-entry-to-descending-body path for one ordered pair."""

    source_body_id: int
    target_body_id: int
    nodes: tuple[int, ...]
    edges: tuple[Edge, ...]
    bottleneck: int


class PathGraph:
    """Deterministic adjacency over the annotated path-eligible subgraph."""

    def __init__(self, retained: RetainedWeights, annotated: set[int]) -> None:
        bodies = np.asarray(sorted(annotated), dtype=np.int64)
        keep = (
            (retained.weight >= PATH_MIN_WEIGHT)
            & np.isin(retained.body_pre, bodies)
            & np.isin(retained.body_post, bodies)
        )
        self.pre: IntArray = retained.body_pre[keep]
        self.post: IntArray = retained.body_post[keep]
        self.weight: IntArray = retained.weight[keep]
        self.source_row: IntArray = retained.source_row[keep]
        self.edge_count = int(self.pre.size)

        self._out_order: IntArray = np.lexsort((self.post, self.pre))
        self._out_keys: IntArray = self.pre[self._out_order]
        self._in_order: IntArray = np.lexsort((self.pre, self.post))
        self._in_keys: IntArray = self.post[self._in_order]
        pair_keys = (self.pre.astype(np.uint64) << np.uint64(32)) | self.post.astype(np.uint64)
        self._pair_order: IntArray = np.argsort(pair_keys, kind="stable")
        self._pair_keys = pair_keys[self._pair_order]

    @staticmethod
    def _slice(keys: IntArray, order: IntArray, body_id: int) -> IntArray:
        left = int(np.searchsorted(keys, body_id, side="left"))
        right = int(np.searchsorted(keys, body_id, side="right"))
        return order[left:right]

    def successors(self, body_id: int) -> tuple[IntArray, IntArray]:
        """Successor bodies in ascending order with their raw source weights."""

        indices = self._slice(self._out_keys, self._out_order, body_id)
        return self.post[indices], self.weight[indices]

    def predecessors(self, body_id: int) -> tuple[IntArray, IntArray]:
        """Predecessor bodies in ascending order with their raw source weights."""

        indices = self._slice(self._in_keys, self._in_order, body_id)
        return self.pre[indices], self.weight[indices]

    def edge_index(self, body_pre: int, body_post: int) -> int | None:
        key = np.uint64((body_pre << 32) | body_post)
        position = int(np.searchsorted(self._pair_keys, key, side="left"))
        if position >= self._pair_keys.size or self._pair_keys[position] != key:
            return None
        return int(self._pair_order[position])

    def edge_for(self, body_pre: int, body_post: int) -> Edge:
        index = self.edge_index(body_pre, body_post)
        if index is None:
            raise CircuitResolutionError(
                f"selected path edge {body_pre}->{body_post} is absent from the path subgraph"
            )
        return Edge(
            body_pre=int(self.pre[index]),
            body_post=int(self.post[index]),
            weight=int(self.weight[index]),
            source_row=int(self.source_row[index]),
        )

    def rows_from(self, bodies: IntArray) -> IntArray:
        """Row indices of every path edge leaving one of ``bodies``."""

        return np.flatnonzero(np.isin(self.pre, bodies)).astype(np.int64, copy=False)


def derive_side(record: Mapping[str, Any], body_id: int) -> tuple[str | None, dict[str, str]]:
    """Resolve an exact L/R side, refusing any disagreement between evidence."""

    evidence: dict[str, str] = {}
    for field in ("rootSide", "somaSide"):
        value = record.get(field)
        if value in {"L", "R"}:
            evidence[field] = str(value)
    instance = record.get("instance")
    if isinstance(instance, str):
        match = INSTANCE_SIDE_RE.search(instance)
        if match is not None:
            evidence["instance_suffix"] = match.group(1)
    observed = set(evidence.values())
    if len(observed) > 1:
        raise CircuitResolutionError(f"side mismatch for body {body_id}: {evidence}")
    return (next(iter(observed)) if observed else None), evidence


def _is_sensory_entry(record: Mapping[str, Any]) -> bool:
    neuron_type = record.get("type")
    if isinstance(neuron_type, str) and neuron_type.startswith("JO-E"):
        return True
    return record.get("subclass") == "campaniform sensilla" and record.get("entryNerve") in {
        "ADMN",
        "DMetaN",
    }


def resolve_sensory(annotations: Mapping[int, AnnotationRecord]) -> tuple[int, ...]:
    """Verify each pinned sensory entry against the exact annotation rule."""

    for body_id in SENSORY_BODY_IDS:
        annotation = annotations.get(body_id)
        if annotation is None:
            raise CircuitResolutionError(f"pinned sensory entry {body_id} has no annotation row")
        if not _is_sensory_entry(annotation.record):
            raise CircuitResolutionError(
                f"pinned sensory entry {body_id} does not satisfy the exact sensory rule"
            )
        derive_side(annotation.record, body_id)
    return tuple(sorted(SENSORY_BODY_IDS))


def resolve_descending(annotations: Mapping[int, AnnotationRecord]) -> dict[str, tuple[int, ...]]:
    """Group the exact descending types, refusing any absent declared type."""

    grouped: dict[str, list[int]] = {name: [] for name in DESCENDING_TYPES}
    for body_id, annotation in annotations.items():
        neuron_type = annotation.record.get("type")
        if isinstance(neuron_type, str) and neuron_type in grouped:
            derive_side(annotation.record, body_id)
            grouped[neuron_type].append(body_id)
    missing = sorted(name for name, bodies in grouped.items() if not bodies)
    if missing:
        raise CircuitResolutionError(f"missing exact descending types: {missing!r}")
    return {name: tuple(sorted(bodies)) for name, bodies in grouped.items()}


def resolve_readout(
    annotations: Mapping[int, AnnotationRecord],
    descending: Sequence[int],
    retained: RetainedWeights,
) -> tuple[int, tuple[int, ...], int]:
    """Wing-motor bodies contacted directly by the declared descending bodies."""

    candidates = sorted(
        body_id
        for body_id, annotation in annotations.items()
        if annotation.record.get("superclass") == MOTOR_SUPERCLASS
        and annotation.record.get("subclass") == MOTOR_SUBCLASS
    )
    if not candidates:
        raise CircuitResolutionError("no exact vnc_motor/wm candidate exists in the source")
    mask = (
        (retained.weight >= READOUT_MIN_WEIGHT)
        & np.isin(retained.body_pre, np.asarray(sorted(descending), dtype=np.int64))
        & np.isin(retained.body_post, np.asarray(candidates, dtype=np.int64))
    )
    contact_rows = int(np.count_nonzero(mask))
    readout = tuple(int(body) for body in np.unique(retained.body_post[mask]))
    if not readout:
        raise CircuitResolutionError(
            "no exact vnc_motor/wm body receives a direct declared descending contact at "
            f"weight >= {READOUT_MIN_WEIGHT}"
        )
    for body_id in readout:
        derive_side(annotations[body_id].record, body_id)
    return len(candidates), readout, contact_rows


def _best_two_edge(
    source: int,
    target: int,
    successors: tuple[IntArray, IntArray],
    predecessors: tuple[IntArray, IntArray],
) -> tuple[int, tuple[int, ...]] | None:
    succ_bodies, succ_weights = successors
    pred_bodies, pred_weights = predecessors
    common, succ_index, pred_index = np.intersect1d(
        succ_bodies, pred_bodies, assume_unique=True, return_indices=True
    )
    if common.size == 0:
        return None
    keep = (common != source) & (common != target)
    if not bool(np.any(keep)):
        return None
    bottlenecks = np.minimum(succ_weights[succ_index], pred_weights[pred_index])[keep]
    best = int(bottlenecks.max())
    middle = int(common[keep][bottlenecks == best].min())
    return best, (source, middle, target)


def _best_three_edge(
    source: int,
    target: int,
    successors: tuple[IntArray, IntArray],
    predecessors: tuple[IntArray, IntArray],
    graph: PathGraph,
    outgoing_rows: IntArray,
) -> tuple[int, tuple[int, ...]] | None:
    succ_bodies, succ_weights = successors
    pred_bodies, pred_weights = predecessors
    if outgoing_rows.size == 0 or pred_bodies.size == 0:
        return None
    inner = outgoing_rows[np.isin(graph.post[outgoing_rows], pred_bodies)]
    if inner.size == 0:
        return None
    first = graph.pre[inner]
    second = graph.post[inner]
    valid = (
        (first != source)
        & (first != target)
        & (second != source)
        & (second != target)
        & (first != second)
    )
    if not bool(np.any(valid)):
        return None
    first = first[valid]
    second = second[valid]
    middle_weight = graph.weight[inner][valid]
    entry_weight = succ_weights[np.searchsorted(succ_bodies, first)]
    exit_weight = pred_weights[np.searchsorted(pred_bodies, second)]
    bottlenecks = np.minimum(np.minimum(entry_weight, middle_weight), exit_weight)
    best = int(bottlenecks.max())
    tied = bottlenecks == best
    first_body = int(first[tied].min())
    second_body = int(second[tied][first[tied] == first_body].min())
    return best, (source, first_body, second_body, target)


def select_path(
    graph: PathGraph, source: int, target: int, outgoing_rows: IntArray
) -> SelectedPath | None:
    """The maximum-bottleneck 1..3 edge path, tie-broken lexicographically.

    This applies the flight-power extractor's frozen objective to one ordered
    (sensory entry, descending body) pair: maximize the minimum raw source weight
    over the path, then take the lexicographically smallest ordered source
    body-ID tuple among the paths that achieve it.
    """

    successors = graph.successors(source)
    predecessors = graph.predecessors(target)
    candidates: list[tuple[int, tuple[int, ...]]] = []
    direct = graph.edge_index(source, target)
    if direct is not None:
        candidates.append((int(graph.weight[direct]), (source, target)))
    two = _best_two_edge(source, target, successors, predecessors)
    if two is not None:
        candidates.append(two)
    three = _best_three_edge(source, target, successors, predecessors, graph, outgoing_rows)
    if three is not None:
        candidates.append(three)
    if not candidates:
        return None

    bottleneck = max(value for value, _ in candidates)
    nodes = min(item for value, item in candidates if value == bottleneck)
    if len(set(nodes)) != len(nodes):
        raise CircuitResolutionError(f"selected path {nodes} repeats a body")
    if len(nodes) - 1 > MAX_PATH_HOPS:
        raise CircuitResolutionError(f"selected path {nodes} exceeds {MAX_PATH_HOPS} edges")
    edges = tuple(graph.edge_for(nodes[index], nodes[index + 1]) for index in range(len(nodes) - 1))
    observed = min(edge.weight for edge in edges)
    if observed != bottleneck:
        raise CircuitResolutionError(
            f"selected path {nodes} bottleneck {observed} does not equal {bottleneck}"
        )
    return SelectedPath(
        source_body_id=source,
        target_body_id=target,
        nodes=nodes,
        edges=edges,
        bottleneck=bottleneck,
    )


def select_paths(
    graph: PathGraph, sensory: Sequence[int], descending: Sequence[int]
) -> tuple[tuple[SelectedPath, ...], tuple[tuple[int, int], ...]]:
    """One selected path per ordered (sensory entry, descending body) pair."""

    selected: list[SelectedPath] = []
    unresolved: list[tuple[int, int]] = []
    for source in sorted(sensory):
        successor_bodies, _ = graph.successors(source)
        outgoing_rows = graph.rows_from(successor_bodies)
        for target in sorted(descending):
            path = select_path(graph, source, target, outgoing_rows)
            if path is None:
                unresolved.append((source, target))
            else:
                selected.append(path)
    if not selected:
        raise CircuitResolutionError(
            "no sensory entry reaches any declared descending body within "
            f"{MAX_PATH_HOPS} edges of weight >= {PATH_MIN_WEIGHT}"
        )
    return tuple(selected), tuple(unresolved)


def induced_edges(selected: set[int], retained: RetainedWeights) -> list[Edge]:
    """Every retained source edge whose endpoints are both selected bodies."""

    bodies = np.asarray(sorted(selected), dtype=np.int64)
    mask = np.isin(retained.body_pre, bodies) & np.isin(retained.body_post, bodies)
    edges = [
        Edge(
            body_pre=int(retained.body_pre[index]),
            body_post=int(retained.body_post[index]),
            weight=int(retained.weight[index]),
            source_row=int(retained.source_row[index]),
        )
        for index in np.flatnonzero(mask)
    ]
    return sorted(edges, key=lambda edge: (edge.body_pre, edge.body_post))


@dataclass(frozen=True, slots=True)
class ResolvedCircuit:
    """Everything the document layer needs, already checked and ordered."""

    sensory: tuple[int, ...]
    descending_by_type: Mapping[str, tuple[int, ...]]
    descending: tuple[int, ...]
    motor_candidate_count: int
    readout: tuple[int, ...]
    descending_to_motor_contact_rows: int
    paths: tuple[SelectedPath, ...]
    unresolved_pairs: tuple[tuple[int, int], ...]
    path_support: tuple[int, ...]
    intermediates: tuple[int, ...]
    selected: tuple[int, ...]
    edges: tuple[Edge, ...]
    path_subgraph_edge_rows: int


def resolve_circuit(
    annotations: Mapping[int, AnnotationRecord], retained: RetainedWeights
) -> ResolvedCircuit:
    """Apply the frozen policy end to end, refusing every ambiguity."""

    sensory = resolve_sensory(annotations)
    descending_by_type = resolve_descending(annotations)
    descending = tuple(sorted({body for group in descending_by_type.values() for body in group}))
    candidate_count, readout, contact_rows = resolve_readout(annotations, descending, retained)
    graph = PathGraph(retained, set(annotations))
    paths, unresolved = select_paths(graph, sensory, descending)
    support = tuple(sorted({body for path in paths for body in path.nodes}))

    selected = set(sensory) | set(descending) | set(readout) | set(support)
    missing = sorted(selected - set(annotations))
    if missing:
        raise CircuitResolutionError(f"selected bodies without annotation rows: {missing!r}")

    edges = induced_edges(selected, retained)
    if len(edges) >= MAX_EMITTED_EDGES:
        raise CircuitResolutionError(
            f"emitted closure has {len(edges)} edges, at or above the declared "
            f"{MAX_EMITTED_EDGES} ceiling"
        )

    return ResolvedCircuit(
        sensory=sensory,
        descending_by_type=descending_by_type,
        descending=descending,
        motor_candidate_count=candidate_count,
        readout=readout,
        descending_to_motor_contact_rows=contact_rows,
        paths=paths,
        unresolved_pairs=unresolved,
        path_support=support,
        intermediates=tuple(sorted(selected - set(sensory) - set(descending) - set(readout))),
        selected=tuple(sorted(selected)),
        edges=tuple(edges),
        path_subgraph_edge_rows=graph.edge_count,
    )
