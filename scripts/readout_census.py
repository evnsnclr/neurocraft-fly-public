"""Annotation census fragments for the Phase 3 readout analysis.

The middle layer, mirroring the three-way split of the Phase 0 reachability
analysis: :mod:`scripts.readout_annotations` owns the verified join and the
integrity gates, this module turns annotation columns into report fragments,
:mod:`scripts.readout_connectivity` owns everything that needs the edge support,
and ``scripts/analyze_readout.py`` binds the fragments to the verified manifest
and writes the document.

Claim discipline. Every figure here is a count of source-observed annotation
strings over the retained nodes of one pinned artifact. A superclass, subclass or
type label is a dataset annotation, not a measured function; this module assigns
no behaviour to any neuron, and the word "causal" — reserved by
``docs/SCIENTIFIC_CLAIMS.md`` for the G3 gate — does not appear in anything it
produces.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from scripts.readout_annotations import (
    NULL_KEY,
    BoolArray,
    IntArray,
    ObjectArray,
    RetainedAnnotations,
    equals,
    full_row,
    histogram,
    is_in,
    key_of,
)

#: Superclasses whose members could serve as a motor-side readout population.
#: Ordered as they are reported; the ordering carries no precedence.
READOUT_ELIGIBLE_SUPERCLASSES: tuple[str, ...] = (
    "descending_neuron",
    "descending_neuron_tbc",
    "efferent_descending",
    "sensory_descending",
    "vnc_motor",
    "cb_motor",
    "vnc_efferent",
)

#: `vnc_motor` subclass codes as the pinned table spells them. These are source
#: labels; the target-organ reading of each code is literature-inferred and is
#: kept out of the hashed measurement body.
WING_MOTOR_SUBCLASS = "wm"
LEG_MOTOR_SUBCLASSES: tuple[str, ...] = ("fl", "ml", "hl")

#: Descending-neuron types the CI fixture names, located in the full artifact.
FIXTURE_NAMED_TYPES: tuple[str, ...] = ("DNg07", "DNp18")

#: Body IDs the CI fixture actually instantiates for those named types.
FIXTURE_MEMBER_BODY_IDS: Mapping[str, tuple[int, ...]] = {
    "DNg07": (27143, 28949),
    "DNp18": (10026,),
}

_ALPHA_HEAD = re.compile(r"^([A-Za-z]+)")


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def superclass_mask(annotations: RetainedAnnotations, name: str) -> BoolArray:
    """Dense-index mask of every retained node carrying ``name`` as superclass."""

    return equals(annotations.column("superclass"), name)


def indices(mask: BoolArray) -> IntArray:
    """Dense indices of a mask, ascending."""

    selected: IntArray = np.flatnonzero(mask).astype(np.int64)
    return selected


# --------------------------------------------------- readout-eligible census


def readout_population_report(annotations: RetainedAnnotations) -> dict[str, Any]:
    """Counts of every superclass a motor-side readout could be drawn from."""

    present = histogram(annotations.column("superclass").tolist())
    counts = {name: present.get(name, 0) for name in READOUT_ELIGIBLE_SUPERCLASSES}
    rows: list[dict[str, Any]] = [
        {
            "superclass": name,
            "retained_nodes": count,
            "fraction_of_retained_nodes": _fraction(count, annotations.node_count),
        }
        for name, count in counts.items()
    ]
    total = sum(counts.values())
    literal = present.get("descending", 0)
    return {
        "definition": (
            "retained-node counts of the superclasses whose members could serve as a"
            " motor-side readout population. Membership is the pinned table's own"
            " superclass string; nothing here measures what any of these neurons does"
        ),
        "rows": rows,
        "total_readout_eligible_nodes": total,
        "total_fraction_of_retained_nodes": _fraction(total, annotations.node_count),
        "naive_descending_filter": {
            "nodes_matching_superclass_exactly_descending": literal,
            "note": (
                "no superclass string is literally 'descending'. A naive equality filter"
                " on that word matches nothing, and a substring filter would silently"
                " sweep in descending_neuron_tbc, efferent_descending and"
                " sensory_descending, which are three different populations"
            ),
            "substring_matching_superclasses": sorted(
                name for name in present if "descending" in name
            ),
        },
        "all_retained_superclasses": present,
    }


# ------------------------------------------------------ vnc_motor functional


def vnc_motor_census(annotations: RetainedAnnotations) -> dict[str, Any]:
    """Subclass census of `vnc_motor`, plus the wing-motor type inventory."""

    motor = superclass_mask(annotations, "vnc_motor")
    subclass = annotations.column("subclass")
    types = annotations.column("type")
    wing = motor & equals(subclass, WING_MOTOR_SUBCLASS)
    body_ids = annotations.body_ids

    wing_types = histogram(types[wing].tolist())
    wing_rows = [
        {
            "type": name,
            "cells": count,
            "body_ids": sorted(
                int(body_ids[index]) for index in indices(wing) if key_of(types[index]) == name
            ),
        }
        for name, count in sorted(wing_types.items())
    ]
    return {
        "definition": (
            "counts of the pinned table's subclass string over retained vnc_motor"
            " nodes, and the type inventory of the wing-motor subclass. Subclass and"
            " type are source-observed labels"
        ),
        "retained_vnc_motor_nodes": int(np.count_nonzero(motor)),
        "subclass_counts": histogram(subclass[motor].tolist()),
        "soma_side_counts": histogram(annotations.column("somaSide")[motor].tolist()),
        "soma_neuromere_counts": histogram(annotations.column("somaNeuromere")[motor].tolist()),
        "exit_nerve_counts": histogram(annotations.column("exitNerve")[motor].tolist()),
        "wing_motor_subclass": {
            "subclass": WING_MOTOR_SUBCLASS,
            "cells": int(np.count_nonzero(wing)),
            "distinct_types": len([name for name in wing_types if name != NULL_KEY]),
            "untyped_cells": wing_types.get(NULL_KEY, 0),
            "soma_neuromere_counts": histogram(annotations.column("somaNeuromere")[wing].tolist()),
            "exit_nerve_counts": histogram(annotations.column("exitNerve")[wing].tolist()),
            "types": wing_rows,
        },
        "leg_motor_subclasses": {
            "subclasses": list(LEG_MOTOR_SUBCLASSES),
            "cells": int(np.count_nonzero(motor & is_in(subclass, LEG_MOTOR_SUBCLASSES))),
        },
    }


# ------------------------------------------------------ current readout identity


def boundary_identity_report(
    annotations: RetainedAnnotations,
    taken: Any,
    sites: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Full annotation rows for the four neurons the shipped adapters bind to.

    The boundary is four named bodies resolved from the manifest overlay. What
    they *are* in the pinned table is a fact about the current interface that no
    other report in this repository records.
    """

    report: dict[str, Any] = {
        "definition": (
            "the complete pinned-table annotation row of every neuron the shipped"
            " boundary overlay names, resolved by dense index from the manifest and"
            " cross-checked against body_ids.bin"
        ),
        "roles": {},
    }
    for role in sorted(sites):
        entries: list[dict[str, Any]] = []
        for site in sorted(sites[role], key=lambda entry: int(entry["body_id"])):
            dense_index = int(site["dense_index"])
            body_id = int(site["body_id"])
            if int(annotations.body_ids[dense_index]) != body_id:
                raise ValueError(
                    f"overlay site {body_id} does not match body_ids.bin at dense index"
                    f" {dense_index}"
                )
            entries.append(
                {
                    "body_id": body_id,
                    "dense_index": dense_index,
                    "annotation_source_row": int(annotations.source_rows[dense_index]),
                    "annotation_row": full_row(taken, dense_index),
                }
            )
        report["roles"][role] = entries
    motor: list[dict[str, Any]] = report["roles"].get("motor", [])
    motor_types = sorted({str(entry["annotation_row"]["type"]) for entry in motor})
    report["current_motor_readout_summary"] = {
        "bodies": [int(entry["body_id"]) for entry in motor],
        "distinct_types": motor_types,
        "is_a_single_bilateral_type": len(motor_types) == 1 and len(motor) == 2,
        "soma_sides": sorted(key_of(entry["annotation_row"]["somaSide"]) for entry in motor),
        "subclasses": sorted({key_of(entry["annotation_row"]["subclass"]) for entry in motor}),
        "soma_neuromeres": sorted(
            {key_of(entry["annotation_row"]["somaNeuromere"]) for entry in motor}
        ),
        "exit_nerves": sorted({key_of(entry["annotation_row"]["exitNerve"]) for entry in motor}),
        "note": (
            "the two motor bodies carry one shared type string and opposite somaSide"
            " values, so the shipped readout is a bilateral pair of one annotated type"
            " rather than two unrelated neurons. This is a statement about the"
            " annotation, not about what either neuron does"
        ),
    }
    return report


# --------------------------------------------------------- descending census


def _alpha_prefix(name: str) -> str:
    match = _ALPHA_HEAD.match(name)
    return match.group(1) if match else NULL_KEY


def descending_census(annotations: RetainedAnnotations) -> dict[str, Any]:
    """Type, prefix, bilaterality, side and subclass census of `descending_neuron`."""

    mask = superclass_mask(annotations, "descending_neuron")
    types = annotations.column("type")[mask]
    sides = annotations.column("somaSide")[mask]
    typed = [str(name) for name in types.tolist() if name is not None]
    per_type = Counter(typed)

    size_histogram = Counter(per_type.values())
    bilateral_pairs = 0
    for name, count in per_type.items():
        if count != 2:
            continue
        member_sides = sorted(
            key_of(side)
            for side, value in zip(sides.tolist(), types.tolist(), strict=True)
            if value == name
        )
        if member_sides == ["L", "R"]:
            bilateral_pairs += 1

    prefixes = Counter(_alpha_prefix(name) for name in typed)
    return {
        "definition": (
            "census of the pinned table's type, somaSide and subclass strings over the"
            " retained descending_neuron superclass. A type is a dataset label; two"
            " cells sharing one are annotated as the same type, which this analysis"
            " does not independently verify"
        ),
        "cells": int(np.count_nonzero(mask)),
        "typed_cells": len(typed),
        "untyped_cells": int(np.count_nonzero(mask)) - len(typed),
        "distinct_types": len(per_type),
        "type_prefix_rule": (
            "the maximal leading run of ASCII letters in the type string, so DNge001 ->"
            " DNge and DNg02_a -> DNg. Prefix classes are therefore disjoint and sum to"
            " typed_cells; DNg and DNge are separate rows, not nested ones"
        ),
        "type_prefix_counts": {key: prefixes[key] for key in sorted(prefixes)},
        "type_prefix_sums_to_typed_cells": sum(prefixes.values()) == len(typed),
        "type_size_histogram": {str(size): size_histogram[size] for size in sorted(size_histogram)},
        "bilaterality": {
            "types_with_exactly_two_cells": sum(1 for count in per_type.values() if count == 2),
            "types_with_exactly_two_cells_one_per_side": bilateral_pairs,
            "types_with_one_cell": sum(1 for count in per_type.values() if count == 1),
            "types_with_more_than_two_cells": sum(1 for count in per_type.values() if count > 2),
            "largest_types": [
                {"type": name, "cells": count}
                for name, count in sorted(per_type.items(), key=lambda item: (-item[1], item[0]))[
                    :10
                ]
            ],
        },
        "soma_side_counts": histogram(sides.tolist()),
        "subclass_counts": histogram(annotations.column("subclass")[mask].tolist()),
        "subclass_note": (
            "for descending neurons the pinned table's subclass column carries the VNC"
            " target neuropil code rather than a motor-organ code; the mapping from code"
            " to neuropil is a dataset convention, recorded here only as a histogram"
        ),
    }


# ----------------------------------------------------- named fixture neurons


def _member_rows(
    annotations: RetainedAnnotations, taken: Any, mask: BoolArray, name: str
) -> list[dict[str, Any]]:
    fixture_members = set(FIXTURE_MEMBER_BODY_IDS.get(name, ()))
    rows = []
    for dense_index in indices(mask).tolist():
        body_id = int(annotations.body_ids[dense_index])
        rows.append(
            {
                "body_id": body_id,
                "dense_index": int(dense_index),
                "annotation_source_row": int(annotations.source_rows[dense_index]),
                "soma_side": key_of(annotations.column("somaSide")[dense_index]),
                "instance": key_of(annotations.column("instance")[dense_index]),
                "used_by_ci_fixture": body_id in fixture_members,
                "annotation_row": full_row(taken, int(dense_index)),
            }
        )
    return sorted(rows, key=lambda row: int(row["body_id"]))


def fixture_named_descending(
    annotations: RetainedAnnotations, taken: Any, names: Sequence[str] = FIXTURE_NAMED_TYPES
) -> dict[str, Any]:
    """Locate every cell of each fixture-named descending type in the full artifact."""

    types: ObjectArray = annotations.column("type")
    report: dict[str, Any] = {
        "definition": (
            "every retained cell whose type string equals a type the CI fixture names,"
            " with the subset the fixture actually instantiates marked. This sizes the"
            " fixture against the full artifact and makes no claim about either"
        ),
        "types": {},
    }
    for name in names:
        mask = equals(types, name)
        rows = _member_rows(annotations, taken, mask, name)
        used = [row for row in rows if row["used_by_ci_fixture"]]
        report["types"][name] = {
            "cells_in_full_artifact": len(rows),
            "cells_used_by_ci_fixture": len(used),
            "fixture_coverage_fraction": _fraction(len(used), len(rows)),
            "soma_side_counts": histogram(row["soma_side"] for row in rows),
            "fixture_member_body_ids": sorted(int(row["body_id"]) for row in used),
            "cells": rows,
        }
        expected = sorted(FIXTURE_MEMBER_BODY_IDS.get(name, ()))
        if sorted(int(row["body_id"]) for row in used) != expected:
            raise ValueError(
                f"fixture member body IDs for {name} were not all found in the artifact"
            )
    return report
