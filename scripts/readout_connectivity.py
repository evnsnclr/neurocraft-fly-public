"""Edge-support measurements for the Phase 3 readout analysis.

Two questions live here, both over the retained directed edge support of the
pinned G1 artifact:

* **The graph-hop-bounded readout comparison.** Holding the injection sites fixed
  at the two named sensory boundary bodies, how much of the graph can lie on an
  injection-to-readout path of at most L hops when the readout is widened from
  the shipped two motor neurons to progressively larger annotated populations?
  This directly extends the open question ``docs/receipts/REACHABILITY-01.md``
  states: the historical L=5 table is a graph-path-length claim, not five
  neural updates. Fresh input can traverse at most two edges in five updates
  under the baseline delay convention; both bounds are reported separately.
  Traversal is the same code the Phase 0 receipt used —
  :mod:`scripts.reachability_graph` — so the shipped-configuration row is a
  reproduction, not a re-implementation.

* **Direct descending-to-motor connectivity.** How much retained weight reaches
  `vnc_motor` neurons straight from `descending_neuron` neurons, how many
  descending neurons have any such contact, and how that weight distributes over
  motor subclasses and soma sides for a named set of descending types.

Claim discipline, taken from REACHABILITY-01 verbatim in spirit: **reachability
bounds participation and never demonstrates it.** A node inside a path-support
set is only topologically able to lie on a path, at an unmodelled and possibly
vanishing, cancelling or sign-inverted gain; no dynamics are executed here.
Likewise an edge weight is a retained raw synapse count, not an efficacy. The
word "causal" is reserved by ``docs/SCIENTIFIC_CLAIMS.md`` for the G3 gate and
is not used for any result in this module.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from scripts.reachability_graph import DirectedGraph, backward_distances, forward_distances
from scripts.reachability_measures import fresh_input_timing
from scripts.readout_annotations import (
    NULL_KEY,
    BoolArray,
    IntArray,
    IntegrityGate,
    ObjectArray,
    RetainedAnnotations,
    equals,
    is_in,
    key_of,
)
from scripts.readout_census import (
    LEG_MOTOR_SUBCLASSES,
    WING_MOTOR_SUBCLASS,
    indices,
    superclass_mask,
)

#: Additional `vnc_motor` subclasses grouped with the wing motor pool in the
#: middle rung of the readout ladder. Grouping is by annotation code only.
WING_ADJACENT_SUBCLASSES: tuple[str, ...] = (WING_MOTOR_SUBCLASS, "nm", "hm")

#: Descending types reported in full in the connectivity table.
NAMED_DESCENDING_TYPES: tuple[str, ...] = (
    "DNa01",
    "DNa02",
    "DNa04",
    "DNa08",
    "DNa15",
    "DNg02_a",
    "DNg07",
    "DNg110",
    "DNg13",
    "DNg99",
    "DNp01",
    "DNp03",
    "DNp09",
    "DNp18",
    "DNp31",
    "MDN",
)

#: How many rows of the whole-population ranking are serialized.
TOP_TYPE_ROWS = 40


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


# --------------------------------------------------------------- readout sets


@dataclass(frozen=True, slots=True)
class ReadoutSet:
    """One candidate readout population, named by its annotation membership rule."""

    name: str
    membership_rule: str
    dense_indices: IntArray


def readout_sets(
    annotations: RetainedAnnotations, shipped_motor_indices: Sequence[int]
) -> tuple[ReadoutSet, ...]:
    """The readout ladder, from the two shipped neurons to every descending neuron."""

    subclass: ObjectArray = annotations.column("subclass")
    motor = superclass_mask(annotations, "vnc_motor")
    return (
        ReadoutSet(
            name="shipped_two_body_readout",
            membership_rule=(
                "exactly the two motor bodies the manifest boundary overlay names;"
                " the configuration REACHABILITY-01 measured"
            ),
            dense_indices=np.asarray(sorted(shipped_motor_indices), dtype=np.int64),
        ),
        ReadoutSet(
            name="vnc_motor_wing_subclass",
            membership_rule="superclass == vnc_motor and subclass == wm",
            dense_indices=indices(motor & equals(subclass, WING_MOTOR_SUBCLASS)),
        ),
        ReadoutSet(
            name="vnc_motor_wing_neck_haltere_subclasses",
            membership_rule="superclass == vnc_motor and subclass in {wm, nm, hm}",
            dense_indices=indices(motor & is_in(subclass, WING_ADJACENT_SUBCLASSES)),
        ),
        ReadoutSet(
            name="vnc_motor_all",
            membership_rule="superclass == vnc_motor",
            dense_indices=indices(motor),
        ),
        ReadoutSet(
            name="descending_neuron_all",
            membership_rule="superclass == descending_neuron",
            dense_indices=indices(superclass_mask(annotations, "descending_neuron")),
        ),
    )


def _hop_histogram(distances: IntArray, targets: IntArray) -> dict[str, int]:
    counter = Counter(int(value) for value in distances[targets].tolist())
    return {("unreached" if hop < 0 else str(hop)): counter[hop] for hop in sorted(counter)}


def _latency_rows(
    lengths: IntArray, reachable: BoolArray, node_count: int, bound: int
) -> list[dict[str, Any]]:
    timing = fresh_input_timing(1)
    rows: list[dict[str, Any]] = []
    for limit in range(1, bound + 1):
        nodes = int(np.count_nonzero(reachable & (lengths <= limit)))
        rows.append(
            {
                "max_path_length": limit,
                "max_graph_hops": limit,
                "fresh_input_minimum_neural_updates_for_hop_budget": (
                    1 + timing["neural_updates_per_graph_hop"] * limit
                ),
                "path_support_nodes": nodes,
                "path_support_fraction": nodes / node_count,
            }
        )
    return rows


def latency_comparison(
    graph: DirectedGraph,
    injection: IntArray,
    sets: Sequence[ReadoutSet],
    *,
    steps_per_game_tick: int,
) -> dict[str, Any]:
    """Legacy L=update-count topology and fresh-input eligibility, separately."""

    timing = fresh_input_timing(steps_per_game_tick)
    forward = forward_distances(graph, injection)
    node_count = graph.node_count
    rows: list[dict[str, Any]] = []
    for candidate in sets:
        backward = backward_distances(graph, candidate.dense_indices)
        reachable: BoolArray = (forward >= 0) & (backward >= 0)
        lengths: IntArray = np.where(
            reachable, forward.astype(np.int64) + backward.astype(np.int64), -1
        )
        unbounded = int(np.count_nonzero(reachable))
        bounded = int(np.count_nonzero(reachable & (lengths <= steps_per_game_tick)))
        fresh_bounded = int(np.count_nonzero(reachable & (lengths <= timing["max_graph_hops"])))
        longest = int(lengths.max()) if unbounded else 0
        rows.append(
            {
                "readout_set": candidate.name,
                "membership_rule": candidate.membership_rule,
                "readout_sites": int(candidate.dense_indices.size),
                "one_game_tick_max_path_length": steps_per_game_tick,
                "one_game_tick_path_support_nodes": bounded,
                "one_game_tick_path_support_fraction": bounded / node_count,
                "legacy_one_game_tick_scope": (
                    "L = neural update count in graph hops; loose historical upper bound,"
                    " not fresh-input timing"
                ),
                "fresh_input_one_game_tick_max_graph_hops": timing["max_graph_hops"],
                "fresh_input_one_game_tick_path_support_nodes": fresh_bounded,
                "fresh_input_one_game_tick_path_support_fraction": fresh_bounded / node_count,
                "unbounded_path_support_nodes": unbounded,
                "unbounded_path_support_fraction": unbounded / node_count,
                "shortest_injection_to_readout_path": (
                    int(lengths[reachable].min()) if unbounded else None
                ),
                "longest_shortest_path_through_a_supported_node": longest,
                "forward_hops_from_injection_to_readout_population": _hop_histogram(
                    forward, candidate.dense_indices
                ),
                "table": _latency_rows(
                    lengths, reachable, node_count, max(longest, steps_per_game_tick)
                ),
            }
        )
    return {
        "definition": (
            "a node lies on some injection -> readout directed path of length at most L"
            " exactly when its forward hop distance plus its hop distance to the readout"
            " set is at most L graph hops. Hops are not neural updates: fresh input first"
            " qualifies at update 1 + 2*h for h edges under baseline delay 1"
        ),
        "evidence_boundary": (
            "reachability is an UPPER BOUND on participation and never evidence of"
            " participation: a path may carry vanishing, cancelling or sign-inverted"
            " influence under the assigned leaky-rate dynamics, which this analysis does"
            " not execute. Widening the readout may raise the topological bound; it does not"
            " show that any additional neuron affects anything"
        ),
        "injection_sites": [int(value) for value in injection.tolist()],
        "neural_steps_per_game_tick": steps_per_game_tick,
        "fresh_input_timing": timing,
        "node_count": node_count,
        "unbounded_degeneracy_note": (
            "every row's unbounded figure sits near 99.17% because the path-support set"
            " is essentially the graph's giant strongly connected component, which"
            " REACHABILITY-01 established and marked as degenerate. Only the"
            " graph-hop-bounded columns distinguish these readout sets"
        ),
        "rows": rows,
    }


# --------------------------------------------- descending -> motor connectivity


@dataclass(frozen=True, slots=True)
class DirectContacts:
    """The retained descending -> motor edge slice and its context totals."""

    pre: IntArray
    post: IntArray
    weights: IntArray
    total_incoming_motor_weight: int
    descending_cells: int
    motor_cells: int


def weight_sum_gate(weights: IntArray, retained_weight_sum: int) -> IntegrityGate:
    """Edge weights must sum to the manifest's retained weight total."""

    observed = int(weights.sum(dtype=np.int64))
    return IntegrityGate(
        name="edge_raw_counts_sum_matches_manifest",
        requirement=(
            "the retained edge_raw_counts must sum to the manifest retained_weight_sum,"
            " so every weight figure below is over the same edge support the manifest"
            " declares"
        ),
        passed=observed == retained_weight_sum,
        observed=observed,
        expected=retained_weight_sum,
    )


def direct_contacts(
    annotations: RetainedAnnotations, pre: IntArray, post: IntArray, weights: IntArray
) -> DirectContacts:
    """Slice the retained edge list down to descending -> vnc_motor rows."""

    descending = superclass_mask(annotations, "descending_neuron")
    motor = superclass_mask(annotations, "vnc_motor")
    into_motor: BoolArray = motor[post]
    selected: BoolArray = descending[pre] & into_motor
    return DirectContacts(
        pre=pre[selected].astype(np.int64),
        post=post[selected].astype(np.int64),
        weights=weights[selected].astype(np.int64),
        total_incoming_motor_weight=int(weights[into_motor].sum(dtype=np.int64)),
        descending_cells=int(np.count_nonzero(descending)),
        motor_cells=int(np.count_nonzero(motor)),
    )


def _laterality(contacts: DirectContacts, side: ObjectArray, mask: BoolArray) -> dict[str, Any]:
    """Ipsilateral-minus-contralateral weight index over annotated soma sides."""

    ipsilateral = 0
    contralateral = 0
    unassigned = 0
    for position in np.flatnonzero(mask).tolist():
        source = side[int(contacts.pre[position])]
        target = side[int(contacts.post[position])]
        weight = int(contacts.weights[position])
        if source in {"L", "R"} and target in {"L", "R"}:
            if source == target:
                ipsilateral += weight
            else:
                contralateral += weight
        else:
            unassigned += weight
    assigned = ipsilateral + contralateral
    return {
        "ipsilateral_weight": ipsilateral,
        "contralateral_weight": contralateral,
        "unassigned_side_weight": unassigned,
        "laterality_index": ((ipsilateral - contralateral) / assigned if assigned else None),
    }


def _subclass_weights(
    contacts: DirectContacts, subclass: ObjectArray, mask: BoolArray
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for position in np.flatnonzero(mask).tolist():
        counter[key_of(subclass[int(contacts.post[position])])] += int(contacts.weights[position])
    return {key: counter[key] for key in sorted(counter)}


def _target_types(
    contacts: DirectContacts, types: ObjectArray, mask: BoolArray
) -> list[tuple[str, int]]:
    """Every postsynaptic type this selection reaches, heaviest first."""

    counter: Counter[str] = Counter()
    for position in np.flatnonzero(mask).tolist():
        counter[key_of(types[int(contacts.post[position])])] += int(contacts.weights[position])
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))


def _type_row(
    contacts: DirectContacts,
    annotations: RetainedAnnotations,
    name: str,
    members: IntArray,
    *,
    target_type_limit: int,
) -> dict[str, Any]:
    subclass = annotations.column("subclass")
    types = annotations.column("type")
    side = annotations.column("somaSide")
    mask: BoolArray = np.isin(contacts.pre, members)
    total = int(contacts.weights[mask].sum(dtype=np.int64))
    by_subclass = _subclass_weights(contacts, subclass, mask)
    wing = by_subclass.get(WING_MOTOR_SUBCLASS, 0)
    leg = sum(by_subclass.get(code, 0) for code in LEG_MOTOR_SUBCLASSES)
    ranked = _target_types(contacts, types, mask)
    return {
        "type": name,
        "cells": int(members.size),
        "cells_with_a_direct_motor_target": int(np.unique(contacts.pre[mask]).size),
        "edges": int(np.count_nonzero(mask)),
        "motor_weight": total,
        "motor_weight_fraction_of_all_descending_to_motor": _fraction(
            total, int(contacts.weights.sum(dtype=np.int64))
        ),
        "distinct_motor_targets": int(np.unique(contacts.post[mask]).size),
        "weight_by_motor_subclass": by_subclass,
        "wing_weight": wing,
        "wing_weight_fraction": _fraction(wing, total),
        "leg_weight": leg,
        "leg_weight_fraction": _fraction(leg, total),
        "other_subclass_weight": total - wing - leg,
        "laterality": _laterality(contacts, side, mask),
        "distinct_target_types": len(ranked),
        "dominant_target_type": ranked[0][0] if ranked else None,
        "dominant_target_type_weight": ranked[0][1] if ranked else None,
        "dominant_target_type_weight_fraction": (
            _fraction(ranked[0][1], total) if ranked else None
        ),
        "top_target_types": [
            {"target_type": label, "weight": weight} for label, weight in ranked[:target_type_limit]
        ],
    }


def descending_to_motor_report(
    annotations: RetainedAnnotations,
    contacts: DirectContacts,
    *,
    named_types: Sequence[str] = NAMED_DESCENDING_TYPES,
    target_type_limit: int = 12,
    ranking_rows: int = TOP_TYPE_ROWS,
) -> dict[str, Any]:
    """Totals, coverage, and a per-descending-type weight and laterality table."""

    types: ObjectArray = annotations.column("type")
    descending = superclass_mask(annotations, "descending_neuron")
    total_weight = int(contacts.weights.sum(dtype=np.int64))
    sources = np.unique(contacts.pre)
    targets = np.unique(contacts.post)

    members: dict[str, list[int]] = {}
    for dense_index in indices(descending).tolist():
        label = types[int(dense_index)]
        if label is not None:
            members.setdefault(str(label), []).append(int(dense_index))

    named_rows = [
        _type_row(
            contacts,
            annotations,
            name,
            np.asarray(sorted(members.get(name, [])), dtype=np.int64),
            target_type_limit=target_type_limit,
        )
        for name in sorted(named_types)
    ]
    totals: list[tuple[str, int, int]] = sorted(
        (
            (
                name,
                len(dense_indices),
                int(
                    contacts.weights[
                        np.isin(contacts.pre, np.asarray(dense_indices, dtype=np.int64))
                    ].sum(dtype=np.int64)
                ),
            )
            for name, dense_indices in members.items()
        ),
        key=lambda entry: (-entry[2], entry[0]),
    )
    ranking: list[dict[str, Any]] = [
        {"type": name, "cells": cells, "motor_weight": weight} for name, cells, weight in totals
    ]
    return {
        "definition": (
            "retained directed rows whose presynaptic node carries superclass"
            " descending_neuron and whose postsynaptic node carries superclass"
            " vnc_motor. Weight is the retained raw synapse count as shipped: no"
            " normalisation, no sign, and no efficacy claim"
        ),
        "evidence_boundary": (
            "a direct contact bounds how few steps a descending neuron needs to reach a"
            " motor neuron; it is not evidence that it does, and this analysis executes"
            " no dynamics and assigns no behaviour to any neuron"
        ),
        "edges": int(contacts.weights.size),
        "weight": total_weight,
        "total_incoming_vnc_motor_weight": contacts.total_incoming_motor_weight,
        "share_of_all_incoming_vnc_motor_weight": _fraction(
            total_weight, contacts.total_incoming_motor_weight
        ),
        "descending_cells": contacts.descending_cells,
        "descending_cells_with_a_direct_motor_target": int(sources.size),
        "descending_cells_with_a_direct_motor_target_fraction": _fraction(
            int(sources.size), contacts.descending_cells
        ),
        "vnc_motor_cells": contacts.motor_cells,
        "vnc_motor_cells_receiving_a_direct_descending_input": int(targets.size),
        "vnc_motor_cells_receiving_a_direct_descending_input_fraction": _fraction(
            int(targets.size), contacts.motor_cells
        ),
        "laterality_definition": (
            "(ipsilateral - contralateral) / (ipsilateral + contralateral) over retained"
            " weight, with sides taken from the somaSide annotation of both endpoints."
            " Rows whose either endpoint lacks an L/R somaSide are excluded from the"
            " index and reported separately as unassigned_side_weight"
        ),
        "subclass_grouping": {
            "wing": [WING_MOTOR_SUBCLASS],
            "leg": list(LEG_MOTOR_SUBCLASSES),
            "note": (
                "wing and leg here are the pinned table's own vnc_motor subclass codes;"
                " weight outside both groups is reported as other_subclass_weight and"
                " itemised in weight_by_motor_subclass"
            ),
            "null_key": NULL_KEY,
        },
        "named_types": named_rows,
        "ranking_note": (
            f"the {ranking_rows} descending types carrying the most direct motor weight,"
            " so the named table above can be located against the whole population"
        ),
        "top_types_by_motor_weight": ranking[:ranking_rows],
        "distinct_descending_types_with_motor_weight": sum(
            1 for _, _, weight in totals if weight > 0
        ),
    }


def contact_summary(report: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Pull one named-type row back out of a serialized report, for printing."""

    for row in report["named_types"]:
        if row["type"] == name:
            return dict(row)
    raise KeyError(name)
