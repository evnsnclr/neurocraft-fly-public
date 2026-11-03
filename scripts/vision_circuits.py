"""Edge-support measurements for the Phase 2 visual feasibility assessment.

Everything that needs the retained directed edge support lives here:

* **Cross-connectivity.** How many retained edges run between a retinotopic
  patch and the committed 128-node flight-steering fixture. This is measured
  directly on the edge list in both directions, never by subtracting one
  aggregate from another, because a zero has to be a zero that was looked for.

* **Latency-bounded descending support.** What fraction of a patch's cells can
  reach any descending neuron in at most two hops over a weight-thresholded
  subgraph.

* **Circuit closure.** How large a fixture the flight-steering fixture's own
  selection rule would produce if its sensory entries were replaced by a
  retinotopic patch, measured at three nested constructions so the answer does
  not depend on one choice of intermediate pool.

* **The lobula-plate tangential alternative.** A census of the tangential cells
  and the size of the circuits they close with the wing-motor pool, plus the
  composition of their retained input weight.

Claim discipline, carried verbatim in spirit from
``docs/receipts/REACHABILITY-01.md``: **reachability bounds participation and
never demonstrates it.** A node inside a path-support set is only topologically
able to lie on a path, at an unmodelled and possibly vanishing, cancelling or
sign-inverted gain; no dynamics are executed here. An edge weight is a retained
raw synapse count, not an efficacy. The word "causal" is reserved by
``docs/SCIENTIFIC_CLAIMS.md`` for the G3 gate and is not used for any result in
this module, and no behaviour is assigned to any neuron.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from scripts.reachability_graph import (
    DirectedGraph,
    backward_distances,
    forward_distances,
    graph_from_edges,
)
from scripts.reachability_measures import fresh_input_timing
from scripts.readout_annotations import (
    BoolArray,
    IntArray,
    ObjectArray,
    RetainedAnnotations,
    equals,
    key_of,
)
from scripts.readout_census import WING_MOTOR_SUBCLASS, indices, superclass_mask

#: Retained edge-weight thresholds every circuit size is reported at.
WEIGHT_THRESHOLDS: tuple[int, ...] = (1, 5, 10, 20)

#: The weight floor the flight-steering fixture's path rule uses when it selects
#: a 1..3 edge path between a sensory entry and a descending body.
PATH_EDGE_MINIMUM_WEIGHT = 10

#: The longest path the flight-steering fixture's selection rule admits.
MAXIMUM_PATH_EDGES = 3

#: The exclusive emitted-edge ceiling the flight-steering fixture's selection
#: policy declares, against which every closure below is sized.
FLIGHT_FIXTURE_EDGE_CEILING = 2000

#: Type prefixes of the lobula-plate tangential cells. Prefix matching makes
#: some entries redundant (``VS`` already covers ``VST1``, ``VST2`` and
#: ``VSm``); they are listed anyway so the declared set is the set a reader can
#: check, and every matched type string is recorded in the report.
LPTC_TYPE_PREFIXES: tuple[str, ...] = (
    "VS",
    "VST1",
    "VST2",
    "VSm",
    "H2",
    "HSE",
    "HSN",
    "HSS",
    "HST",
)

#: Type prefixes of the columnar motion-pathway cells whose share of tangential
#: input weight is measured.
MOTION_PATHWAY_PREFIXES: tuple[str, ...] = ("T4", "T5")

#: How many presynaptic types of the tangential pool are ranked in the report.
TOP_INPUT_TYPE_ROWS = 15


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _node_mask(node_count: int, nodes: IntArray) -> BoolArray:
    mask: BoolArray = np.zeros(node_count, dtype=bool)
    mask[np.asarray(nodes, dtype=np.int64)] = True
    return mask


# --------------------------------------------------------------- edge support


@dataclass(frozen=True, slots=True)
class EdgeSupport:
    """The retained directed edge list and its raw synapse counts."""

    node_count: int
    pre: IntArray
    post: IntArray
    weights: IntArray

    def induced_edges(
        self, nodes: IntArray, *, thresholds: Sequence[int] = WEIGHT_THRESHOLDS
    ) -> dict[str, int]:
        """Retained edges with both endpoints inside ``nodes``, per threshold."""

        mask = _node_mask(self.node_count, nodes)
        inside: BoolArray = mask[self.pre] & mask[self.post]
        return {
            str(threshold): int(np.count_nonzero(inside & (self.weights >= threshold)))
            for threshold in thresholds
        }

    def induced_weight(self, nodes: IntArray, *, threshold: int = 1) -> int:
        mask = _node_mask(self.node_count, nodes)
        inside: BoolArray = mask[self.pre] & mask[self.post] & (self.weights >= threshold)
        return int(self.weights[inside].sum(dtype=np.int64))

    def subgraph(self, minimum_weight: int) -> DirectedGraph:
        keep: BoolArray = self.weights >= minimum_weight
        return graph_from_edges(
            self.node_count,
            self.pre[keep].astype(np.int64),
            self.post[keep].astype(np.int64),
        )


# ------------------------------------------------------- cross-connectivity


def cross_connectivity(
    support: EdgeSupport,
    *,
    left: IntArray,
    right: IntArray,
    left_name: str,
    right_name: str,
    types: ObjectArray,
    body_ids: IntArray,
    threshold: int = 1,
    listing_limit: int = 16,
) -> dict[str, Any]:
    """Count retained edges running between two node sets, in both directions.

    The count is taken directly off the edge list rather than by differencing
    two larger aggregates, so a reported zero is the result of looking for an
    edge and not finding one.
    """

    left_mask = _node_mask(support.node_count, left)
    right_mask = _node_mask(support.node_count, right)
    keep: BoolArray = support.weights >= threshold
    forward: BoolArray = left_mask[support.pre] & right_mask[support.post] & keep
    backward: BoolArray = right_mask[support.pre] & left_mask[support.post] & keep
    both: BoolArray = forward | backward
    listed = [
        {
            "pre_body_id": int(body_ids[support.pre[position]]),
            "pre_type": key_of(types[support.pre[position]]),
            "post_body_id": int(body_ids[support.post[position]]),
            "post_type": key_of(types[support.post[position]]),
            "weight": int(support.weights[position]),
        }
        for position in np.flatnonzero(both).tolist()[:listing_limit]
    ]
    overlap = int(np.count_nonzero(left_mask & right_mask))
    return {
        "definition": (
            "retained directed edges of at least the stated weight with one endpoint in"
            " each set, counted directly on the edge list in both directions. No figure"
            " here is obtained by subtracting one aggregate from another"
        ),
        "left_set": left_name,
        "right_set": right_name,
        "left_nodes": int(np.asarray(left).size),
        "right_nodes": int(np.asarray(right).size),
        "shared_nodes": overlap,
        "minimum_weight": threshold,
        "left_to_right_edges": int(np.count_nonzero(forward)),
        "right_to_left_edges": int(np.count_nonzero(backward)),
        "total_edges": int(np.count_nonzero(both)),
        "total_weight": int(support.weights[both].sum(dtype=np.int64)),
        "edges": listed,
        "edges_listed_is_complete": int(np.count_nonzero(both)) <= listing_limit,
    }


# ------------------------------------------------- descending path support


def descending_support(
    graph: DirectedGraph,
    *,
    patch: IntArray,
    descending: IntArray,
    minimum_weight: int,
    maximum_hops: int = 2,
) -> dict[str, Any]:
    """Fraction of patch cells within ``maximum_hops`` of any descending neuron."""

    distances = backward_distances(graph, descending)
    observed: IntArray = distances[patch]
    within = int(np.count_nonzero((observed >= 0) & (observed <= maximum_hops)))
    counter = Counter(int(value) for value in observed.tolist())
    return {
        "definition": (
            "hop distance from each patch cell to the nearest descending neuron over the"
            " weight-thresholded subgraph, and the share of patch cells whose distance is"
            " at most the stated graph-hop bound. Graph hops are not neural updates"
        ),
        "evidence_boundary": (
            "for fresh sensory drive, a route of h edges first becomes eligible at"
            " update 1 + 2*h under baseline delay 1. This is only topological eligibility,"
            " not evidence that a signal reaches the descending layer; no dynamics run"
        ),
        "minimum_edge_weight": minimum_weight,
        "subgraph_edges": graph.edge_count,
        "maximum_hops": maximum_hops,
        "max_graph_hops": maximum_hops,
        "fresh_input_minimum_neural_updates_for_hop_budget": (
            1 + fresh_input_timing(1)["neural_updates_per_graph_hop"] * maximum_hops
        ),
        "patch_cells": int(np.asarray(patch).size),
        "descending_sites": int(np.asarray(descending).size),
        "cells_within_maximum_hops": within,
        "fraction_within_maximum_hops": _fraction(within, int(np.asarray(patch).size)),
        "hop_distance_histogram": {
            ("unreachable" if hop < 0 else str(hop)): counter[hop] for hop in sorted(counter)
        },
    }


# ---------------------------------------------------------- circuit closure


@dataclass(frozen=True, slots=True)
class ClosureTier:
    """One nested construction of the circuit a retinotopic patch would close."""

    name: str
    rule: str
    intermediate_pool: IntArray
    nodes: IntArray


def closure_tiers(
    graph: DirectedGraph,
    *,
    patch: IntArray,
    descending: IntArray,
    motor: IntArray,
    maximum_edges: int = MAXIMUM_PATH_EDGES,
) -> tuple[tuple[ClosureTier, ...], IntArray]:
    """Build the three nested closures and return them with the reached readout.

    The flight-steering fixture's rule is: for every ordered (sensory entry,
    descending body) pair, select a simple directed path of one to three edges
    over the weight-thresholded subgraph, and every body on every selected path
    joins the fixture. Three constructions bracket what that rule costs when the
    sensory entries are a retinotopic patch:

    * ``mandatory_core`` adds no intermediate at all. It is contained in every
      admissible closure, so its induced edge count is a floor that no choice of
      intermediates can go below.
    * ``path_interior_closure`` adds exactly the bodies that lie on at least one
      qualifying path — forward distance from the patch plus hop distance to a
      reached descending neuron at most the path bound. Any minimal intermediate
      pool is a subset of this one, so its induced edge count brackets the
      minimal closure from above.
    * ``two_hop_candidate_closure`` adds every body within two hops of the patch,
      which is the complete candidate pool before the path condition is applied.
      It contains bodies lying on no qualifying path and is reported as the
      upper construction, not as the cost of the rule.
    """

    forward = forward_distances(graph, patch)
    reached: IntArray = descending[
        (forward[descending] >= 1) & (forward[descending] <= maximum_edges)
    ]
    backward = backward_distances(graph, reached)
    interior: IntArray = np.flatnonzero(
        (forward >= 1)
        & (backward >= 1)
        & ((forward.astype(np.int64) + backward.astype(np.int64)) <= maximum_edges)
    ).astype(np.int64)
    candidate: IntArray = np.flatnonzero((forward >= 1) & (forward <= maximum_edges - 1)).astype(
        np.int64
    )
    core: IntArray = np.unique(np.concatenate([patch, reached, motor]))
    tiers = (
        ClosureTier(
            name="mandatory_core",
            rule=(
                "the patch, every descending neuron it reaches within the path bound,"
                " and the wing-motor readout pool, with no intermediate body at all."
                " Every admissible closure contains this set, so its induced edge count"
                " is a floor"
            ),
            intermediate_pool=np.asarray([], dtype=np.int64),
            nodes=core,
        ),
        ClosureTier(
            name="path_interior_closure",
            rule=(
                "the core plus every body lying on at least one qualifying path: forward"
                " hop distance from the patch at least one, hop distance to a reached"
                " descending neuron at least one, and the two summing to at most the path"
                " bound. Any minimal intermediate pool is a subset of this one"
            ),
            intermediate_pool=interior,
            nodes=np.unique(np.concatenate([core, interior])),
        ),
        ClosureTier(
            name="two_hop_candidate_closure",
            rule=(
                "the core plus every body within two hops of the patch — the complete"
                " candidate interior pool before the path condition is applied. It"
                " contains bodies that lie on no qualifying path"
            ),
            intermediate_pool=candidate,
            nodes=np.unique(np.concatenate([core, candidate])),
        ),
    )
    return tiers, reached


def closure_report(
    support: EdgeSupport,
    graph: DirectedGraph,
    *,
    patch: IntArray,
    descending: IntArray,
    motor: IntArray,
    ceiling: int = FLIGHT_FIXTURE_EDGE_CEILING,
    thresholds: Sequence[int] = WEIGHT_THRESHOLDS,
) -> dict[str, Any]:
    """Size every closure tier against the fixture's declared edge ceiling."""

    tiers, reached = closure_tiers(graph, patch=patch, descending=descending, motor=motor)
    rows: list[dict[str, Any]] = []
    for tier in tiers:
        counts = support.induced_edges(tier.nodes, thresholds=thresholds)
        at_fixture_floor = counts[str(_fixture_floor(thresholds))]
        rows.append(
            {
                "closure": tier.name,
                "construction_rule": tier.rule,
                "intermediate_pool_bodies": int(tier.intermediate_pool.size),
                "closure_nodes": int(tier.nodes.size),
                "induced_edges": counts,
                "induced_edges_at_the_fixture_weight_floor": at_fixture_floor,
                "multiple_of_the_fixture_edge_ceiling": at_fixture_floor / ceiling,
                "exceeds_the_fixture_edge_ceiling": at_fixture_floor >= ceiling,
            }
        )
    return {
        "definition": (
            "the circuit the flight-steering fixture's own selection rule would close if"
            " its two sensory entries were replaced by a retinotopic patch: the patch,"
            " the descending neurons it reaches over one to three edges of the"
            " weight-thresholded subgraph, the bodies on those paths, and the wing-motor"
            " readout pool. Induced edges are every retained edge with both endpoints"
            " inside the closure"
        ),
        "evidence_boundary": (
            "closure size is a topological count over the retained edge support. It says"
            " nothing about dynamics, about whether any of these bodies would carry"
            " signal, or about what the resulting circuit would do"
        ),
        "path_edge_minimum_weight": PATH_EDGE_MINIMUM_WEIGHT,
        "maximum_path_edges": MAXIMUM_PATH_EDGES,
        "fixture_edge_ceiling_exclusive": ceiling,
        "patch_cells": int(np.asarray(patch).size),
        "descending_sites_offered": int(np.asarray(descending).size),
        "descending_neurons_reached": int(reached.size),
        "descending_neurons_reached_fraction": _fraction(
            int(reached.size), int(np.asarray(descending).size)
        ),
        "wing_motor_readout_bodies": int(np.asarray(motor).size),
        "tiers": rows,
        "ordering_note": (
            "the three tiers are nested by construction, so their induced edge counts are"
            " non-decreasing. The floor tier is what no intermediate choice can avoid;"
            " the middle tier is what the path rule itself costs; the upper tier is what"
            " an unfiltered two-hop candidate pool would cost"
        ),
    }


def _fixture_floor(thresholds: Sequence[int]) -> int:
    """The declared weight floor the fixture emits at, if it is measured here."""

    return 5 if 5 in tuple(thresholds) else int(min(thresholds))


# ----------------------------------------------- lobula plate tangential cells


def type_prefix_members(types: ObjectArray, prefixes: Sequence[str]) -> IntArray:
    """Dense indices whose type string starts with any declared prefix."""

    wanted = tuple(prefixes)
    selected: IntArray = np.asarray(
        sorted(
            index
            for index, value in enumerate(types.tolist())
            if value is not None and str(value).startswith(wanted)
        ),
        dtype=np.int64,
    )
    return selected


def tangential_census(
    annotations: RetainedAnnotations, members: IntArray, *, prefixes: Sequence[str]
) -> dict[str, Any]:
    """Type, side and superclass census of the tangential-cell population."""

    types = annotations.column("type")
    sides = annotations.column("somaSide")
    superclasses = annotations.column("superclass")
    listed = members.tolist()
    type_counts = Counter(key_of(types[index]) for index in listed)
    side_counts = Counter(key_of(sides[index]) for index in listed)
    superclass_counts = Counter(key_of(superclasses[index]) for index in listed)
    return {
        "definition": (
            "retained nodes whose pinned-table type string starts with one of the declared"
            " prefixes. Prefix membership is a label rule over dataset strings and carries"
            " no functional claim"
        ),
        "declared_prefixes": list(prefixes),
        "cells": int(members.size),
        "type_counts": {key: type_counts[key] for key in sorted(type_counts)},
        "soma_side_counts": {key: side_counts[key] for key in sorted(side_counts)},
        "superclass_counts": {key: superclass_counts[key] for key in sorted(superclass_counts)},
        "body_ids": sorted(int(annotations.body_ids[index]) for index in listed),
    }


def tangential_circuits(
    support: EdgeSupport,
    annotations: RetainedAnnotations,
    *,
    tangential: IntArray,
    motor: IntArray,
    descending_threshold: int = 20,
    thresholds: Sequence[int] = WEIGHT_THRESHOLDS,
) -> dict[str, Any]:
    """Size the tangential-cell circuits, alone and closed onto the motor pool."""

    descending = indices(superclass_mask(annotations, "descending_neuron"))
    tangential_mask = _node_mask(support.node_count, tangential)
    descending_mask = _node_mask(support.node_count, descending)
    direct: BoolArray = (
        tangential_mask[support.pre]
        & descending_mask[support.post]
        & (support.weights >= descending_threshold)
    )
    reached: IntArray = np.unique(support.post[direct]).astype(np.int64)
    circuits = (
        ("tangential_only", tangential, "the tangential cells alone"),
        (
            "tangential_and_wing_motor",
            np.unique(np.concatenate([tangential, motor])),
            "the tangential cells plus the whole wing-motor readout pool",
        ),
        (
            "tangential_descending_and_wing_motor",
            np.unique(np.concatenate([tangential, reached, motor])),
            (
                "the tangential cells, every descending neuron they contact directly at"
                f" weight at least {descending_threshold}, and the wing-motor readout pool"
            ),
        ),
    )
    rows = [
        {
            "circuit": name,
            "membership_rule": rule,
            "nodes": int(nodes.size),
            "induced_edges": support.induced_edges(nodes, thresholds=thresholds),
        }
        for name, nodes, rule in circuits
    ]
    return {
        "definition": (
            "node and induced-edge counts of the circuits the tangential cells close with"
            " the wing-motor readout pool over the retained edge support"
        ),
        "descending_contact_minimum_weight": descending_threshold,
        "descending_neurons_contacted_directly": int(reached.size),
        "wing_motor_readout_bodies": int(np.asarray(motor).size),
        "rows": rows,
        "fixture_edge_ceiling_exclusive": FLIGHT_FIXTURE_EDGE_CEILING,
    }


def _weight_by_presynaptic_type(
    support: EdgeSupport, incoming: BoolArray, types: ObjectArray
) -> Counter[str]:
    labels = [key_of(value) for value in types.tolist()]
    codes = {label: code for code, label in enumerate(sorted(set(labels)))}
    encoded: IntArray = np.asarray([codes[label] for label in labels], dtype=np.int64)
    totals: IntArray = np.zeros(len(codes), dtype=np.int64)
    np.add.at(totals, encoded[support.pre[incoming]], support.weights[incoming])
    counter: Counter[str] = Counter()
    for label, code in codes.items():
        value = int(totals[code])
        if value:
            counter[label] = value
    return counter


def tangential_input_composition(
    support: EdgeSupport,
    annotations: RetainedAnnotations,
    *,
    tangential: IntArray,
    prefixes: Sequence[str] = MOTION_PATHWAY_PREFIXES,
    thresholds: Sequence[int] = WEIGHT_THRESHOLDS,
    ranking_rows: int = TOP_INPUT_TYPE_ROWS,
) -> dict[str, Any]:
    """Share of retained tangential input weight arriving from the motion pathway."""

    types = annotations.column("type")
    tangential_mask = _node_mask(support.node_count, tangential)
    rows: list[dict[str, Any]] = []
    ranked: list[dict[str, Any]] = []
    breakdown: dict[str, dict[str, int]] = {}
    for threshold in thresholds:
        incoming: BoolArray = tangential_mask[support.post] & (support.weights >= threshold)
        counter = _weight_by_presynaptic_type(support, incoming, types)
        total = sum(counter.values())
        pathway = {
            label: weight for label, weight in counter.items() if label.startswith(tuple(prefixes))
        }
        pathway_total = sum(pathway.values())
        rows.append(
            {
                "minimum_weight": threshold,
                "total_incoming_weight": total,
                "motion_pathway_weight": pathway_total,
                "motion_pathway_weight_fraction": _fraction(pathway_total, total),
                "incoming_edges": int(np.count_nonzero(incoming)),
            }
        )
        breakdown[str(threshold)] = {key: pathway[key] for key in sorted(pathway)}
        if threshold == thresholds[0]:
            ranked = [
                {"presynaptic_type": label, "weight": weight}
                for label, weight in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[
                    :ranking_rows
                ]
            ]
    return {
        "definition": (
            "retained incoming weight of the tangential pool grouped by the presynaptic"
            " node's type string, and the share of it whose presynaptic type starts with"
            " one of the declared motion-pathway prefixes. Weight is the retained raw"
            " synapse count as shipped: no normalisation, no sign, no efficacy claim"
        ),
        "motion_pathway_prefixes": list(prefixes),
        "threshold_note": (
            "the share depends on the weight floor, so every threshold is reported. A"
            " single quoted percentage without its floor is not reproducible"
        ),
        "rows": rows,
        "motion_pathway_weight_by_type": breakdown,
        "top_presynaptic_types": ranked,
    }


def motion_pathway_addressing(
    annotations: RetainedAnnotations,
    addressed: BoolArray,
    *,
    prefixes: Sequence[str] = MOTION_PATHWAY_PREFIXES,
) -> dict[str, Any]:
    """Whether the motion-pathway cells carry a column address of their own."""

    members = type_prefix_members(annotations.column("type"), prefixes)
    types = annotations.column("type")
    counter = Counter(key_of(types[index]) for index in members.tolist())
    with_address = int(np.count_nonzero(addressed[members])) if members.size else 0
    return {
        "definition": (
            "how many cells of the declared motion-pathway prefixes carry both column"
            " address values in the pinned table"
        ),
        "declared_prefixes": list(prefixes),
        "cells": int(members.size),
        "cells_with_a_column_address": with_address,
        "cells_with_a_column_address_fraction": _fraction(with_address, int(members.size)),
        "type_counts": {key: counter[key] for key in sorted(counter)},
    }


# ------------------------------------------------------------- readout pools


def wing_motor_pool(annotations: RetainedAnnotations) -> IntArray:
    """The retained `vnc_motor` cells the pinned table labels wing motor."""

    motor = superclass_mask(annotations, "vnc_motor")
    subclass = annotations.column("subclass")
    return indices(motor & equals(subclass, WING_MOTOR_SUBCLASS))


def descending_pool(annotations: RetainedAnnotations) -> IntArray:
    """The retained cells the pinned table labels `descending_neuron`."""

    return indices(superclass_mask(annotations, "descending_neuron"))


def dense_indices_for(body_ids: IntArray, wanted: Sequence[int]) -> IntArray:
    """Resolve exact source body IDs to dense indices, refusing a near match."""

    order: IntArray = np.argsort(body_ids)
    target: IntArray = np.asarray(sorted(wanted), dtype=np.uint64)
    position: IntArray = np.searchsorted(body_ids[order], target)
    if position.size and int(position.max()) >= body_ids.size:
        raise KeyError("a requested body ID is beyond the artifact body ID range")
    resolved: IntArray = order[position].astype(np.int64)
    if not bool(np.array_equal(body_ids[resolved], target)):
        raise KeyError("a requested body ID is not present in the artifact")
    return resolved


def pool_summary(annotations: RetainedAnnotations, nodes: IntArray, name: str) -> Mapping[str, Any]:
    """A short identity block for one named pool, for the report header."""

    sides = annotations.column("somaSide")
    counter = Counter(key_of(sides[index]) for index in nodes.tolist())
    return {
        "pool": name,
        "cells": int(nodes.size),
        "soma_side_counts": {key: counter[key] for key in sorted(counter)},
    }
