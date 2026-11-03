"""Retinotopic column addressing for the Phase 2 visual feasibility assessment.

The optic-lobe half of the analysis lives here: which retained nodes carry a
source-observed hex column address, which columns are complete over a declared
type set, how a bounded retinotopic patch is selected from them, and how much
that selection depends on a hex-axis convention this repository has never
frozen.

Method, taken from ``docs/receipts/OPTIC-CENSUS-01.md`` and
``docs/receipts/READOUT-CENSUS-01.md``: column coordinates are read only
through the verified annotation join that :mod:`scripts.readout_annotations`
owns, never by reopening the source table, and the census reproduces the
published optic figures as a built-in guard so a drifting join is a blocking
error rather than a new result.

Claim discipline. Everything here is annotation transcription and lattice
arithmetic. ``assignedOlHex1``/``assignedOlHex2`` are dataset labels, a type
string is a dataset label, and no figure in this module assigns a behaviour, a
receptive field, or a visual function to any neuron. The word "causal" is
reserved by ``docs/SCIENTIFIC_CLAIMS.md`` for the G3 gate and does not appear
in anything this module produces.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from scripts.readout_annotations import BoolArray, IntArray, ObjectArray, key_of

FloatArray = npt.NDArray[np.float64]

#: Column-address column names as the pinned annotations table spells them.
HEX_COLUMNS: tuple[str, str] = ("assignedOlHex1", "assignedOlHex2")

#: The eight columnar types the retinotopic patch is built from. This set is a
#: choice made by this analysis, not a frozen repository rule; it is the largest
#: subset of the fifteen column-addressed types that is complete in the most
#: right-eye columns while staying inside the lamina-to-medulla stack.
PATCH_TYPES: tuple[str, ...] = ("L1", "L2", "L3", "L5", "Mi1", "Tm1", "Tm2", "Tm9")

#: Every type ``OPTIC-CENSUS-01`` measured as carrying a column address.
COLUMNAR_TYPES: tuple[str, ...] = (
    "C2",
    "C3",
    "L1",
    "L2",
    "L3",
    "L5",
    "Mi1",
    "Mi4",
    "Mi9",
    "T1",
    "Tm1",
    "Tm2",
    "Tm20",
    "Tm4",
    "Tm9",
)

#: The eye the patch is drawn from. ``OPTIC-CENSUS-01`` measured that no
#: left-eye column is complete over the fifteen column-addressed types, so a
#: column-addressed patch is defensible for the right eye only.
PATCH_SOMA_SIDE = "R"

#: Figures ``docs/receipts/OPTIC-CENSUS-01.md`` published for the same pinned
#: artifact and annotations table. The census below must reproduce them exactly
#: or the run is refused; a deviation is a blocking error, never a new result.
OPTIC_CENSUS_01: dict[str, Any] = {
    "receipt": "docs/receipts/OPTIC-CENSUS-01.md",
    "column_addressed_nodes": 23720,
    "column_addressed_types": list(COLUMNAR_TYPES),
    "right_eye_columns": 892,
    "left_eye_columns": 879,
    "right_eye_columns_complete_over_all_columnar_types": 797,
    "left_eye_columns_complete_over_all_columnar_types": 0,
}


# ------------------------------------------------------------------ addresses


@dataclass(frozen=True, slots=True)
class HexAddresses:
    """The two column-address columns of the retained rows, plus their mask."""

    hex1: FloatArray
    hex2: FloatArray
    addressed: BoolArray

    @property
    def node_count(self) -> int:
        return int(self.addressed.size)

    def key(self, dense_index: int) -> tuple[int, int]:
        return (int(self.hex1[dense_index]), int(self.hex2[dense_index]))


def numeric_column(taken: Any, name: str) -> FloatArray:
    """Read one numeric column of the taken annotation rows as float64.

    Nulls arrive as NaN, which is the only reason a value in this module is
    ever non-finite; nothing derived from a NaN reaches the report.
    """

    values: FloatArray = np.asarray(
        taken.column(name).to_numpy(zero_copy_only=False), dtype=np.float64
    )
    return values


def hex_addresses(taken: Any) -> HexAddresses:
    """Take both column-address columns and mark the fully addressed nodes."""

    hex1 = numeric_column(taken, HEX_COLUMNS[0])
    hex2 = numeric_column(taken, HEX_COLUMNS[1])
    addressed: BoolArray = np.asarray(~np.isnan(hex1) & ~np.isnan(hex2), dtype=bool)
    return HexAddresses(hex1=hex1, hex2=hex2, addressed=addressed)


def address_census(
    addresses: HexAddresses, types: ObjectArray, sides: ObjectArray
) -> dict[str, Any]:
    """Count the column-addressed nodes and the types and sides carrying them."""

    selected = np.flatnonzero(addresses.addressed).tolist()
    integral = bool(
        np.all(addresses.hex1[addresses.addressed] == np.floor(addresses.hex1[addresses.addressed]))
        and np.all(
            addresses.hex2[addresses.addressed] == np.floor(addresses.hex2[addresses.addressed])
        )
    )
    partial = int(
        np.count_nonzero(np.isnan(addresses.hex1) ^ np.isnan(addresses.hex2)),
    )
    type_counts = Counter(key_of(types[index]) for index in selected)
    return {
        "definition": (
            "a node is column-addressed when both assignedOlHex1 and assignedOlHex2"
            " carry a value in the pinned annotations table. Both are dataset labels;"
            " this analysis does not verify or reconstruct them"
        ),
        "nodes_examined": addresses.node_count,
        "column_addressed_nodes": len(selected),
        "partially_addressed_nodes": partial,
        "addresses_are_integral": integral,
        "hex1_range": _range(addresses.hex1[addresses.addressed]),
        "hex2_range": _range(addresses.hex2[addresses.addressed]),
        "types_carrying_an_address": {key: type_counts[key] for key in sorted(type_counts)},
        "soma_side_counts": _histogram(sides, selected),
    }


def _range(values: FloatArray) -> dict[str, int] | None:
    if values.size == 0:
        return None
    return {"minimum": int(values.min()), "maximum": int(values.max())}


def _histogram(column: ObjectArray, indices: Sequence[int]) -> dict[str, int]:
    counter = Counter(key_of(column[index]) for index in indices)
    return {key: counter[key] for key in sorted(counter)}


# -------------------------------------------------------------- type families


@dataclass(frozen=True, slots=True)
class TypeFamily:
    """One named lamina, medulla or photoreceptor type family and its rule."""

    name: str
    exact: tuple[str, ...] = ()
    prefixes: tuple[str, ...] = ()

    @property
    def rule(self) -> str:
        parts: list[str] = []
        if self.exact:
            parts.append("type equals " + " or ".join(repr(value) for value in self.exact))
        if self.prefixes:
            parts.append("type starts with " + " or ".join(repr(value) for value in self.prefixes))
        return "; ".join(parts)

    def matches(self, value: object) -> bool:
        if value is None:
            return False
        label = str(value)
        return label in self.exact or label.startswith(self.prefixes)


#: The visual type families a retinotopic injection would have to reach. Each
#: rule is stated explicitly so a reader can re-derive the count without
#: guessing what a wildcard meant.
VISUAL_TYPE_FAMILIES: tuple[TypeFamily, ...] = (
    TypeFamily(name="L1-L5", exact=("L1", "L2", "L3", "L4", "L5")),
    TypeFamily(name="Mi", prefixes=("Mi",)),
    TypeFamily(name="Tm", prefixes=("Tm",)),
    TypeFamily(name="R1-R6", exact=("R1-R6",)),
    TypeFamily(name="R7", prefixes=("R7",)),
    TypeFamily(name="R8", prefixes=("R8",)),
    TypeFamily(name="T4", prefixes=("T4",)),
    TypeFamily(name="T5", prefixes=("T5",)),
    TypeFamily(name="C2", exact=("C2",)),
    TypeFamily(name="C3", exact=("C3",)),
)


def type_family_report(
    types: ObjectArray,
    *,
    families: Sequence[TypeFamily] = VISUAL_TYPE_FAMILIES,
    scope: str,
) -> dict[str, Any]:
    """Count every declared visual type family over one population of nodes."""

    labels = types.tolist()
    rows: list[dict[str, Any]] = []
    for family in families:
        members = [index for index, value in enumerate(labels) if family.matches(value)]
        counter = Counter(key_of(labels[index]) for index in members)
        rows.append(
            {
                "family": family.name,
                "membership_rule": family.rule,
                "cells": len(members),
                "matched_types": {key: counter[key] for key in sorted(counter)},
            }
        )
    return {
        "scope": scope,
        "definition": (
            "counts of the pinned table's type string over the named lamina, medulla"
            " and photoreceptor families. A type is a dataset label, not a measured"
            " function, and this analysis assigns no behaviour to any neuron"
        ),
        "families": rows,
        "total_cells_in_any_family": sum(int(row["cells"]) for row in rows),
        "families_with_no_member": sorted(
            str(row["family"]) for row in rows if int(row["cells"]) == 0
        ),
    }


# ------------------------------------------------------------------- columns


@dataclass(frozen=True, slots=True)
class ColumnIndex:
    """Column-addressed cells of one eye, grouped by column then by type."""

    soma_side: str
    members: Mapping[tuple[int, int], Mapping[str, tuple[int, ...]]]

    def cells(self, columns: Sequence[tuple[int, int]], types: Sequence[str]) -> IntArray:
        """Dense indices of every cell of ``types`` inside ``columns``, ascending."""

        collected: list[int] = []
        for column in columns:
            per_type = self.members[column]
            for name in types:
                collected.extend(per_type.get(name, ()))
        selected: IntArray = np.asarray(sorted(collected), dtype=np.int64)
        return selected

    def complete(self, types: Sequence[str]) -> tuple[tuple[int, int], ...]:
        """Columns holding at least one cell of every declared type, ascending."""

        return tuple(
            sorted(
                column
                for column, per_type in self.members.items()
                if all(per_type.get(name) for name in types)
            )
        )


def build_column_index(
    addresses: HexAddresses,
    types: ObjectArray,
    sides: ObjectArray,
    *,
    soma_side: str = PATCH_SOMA_SIDE,
) -> ColumnIndex:
    """Group one eye's column-addressed cells by (hex1, hex2) and by type."""

    members: dict[tuple[int, int], dict[str, list[int]]] = {}
    for index in np.flatnonzero(addresses.addressed).tolist():
        if key_of(sides[index]) != soma_side:
            continue
        column = addresses.key(index)
        members.setdefault(column, {}).setdefault(key_of(types[index]), []).append(index)
    frozen: dict[tuple[int, int], Mapping[str, tuple[int, ...]]] = {
        column: {name: tuple(sorted(values)) for name, values in sorted(per_type.items())}
        for column, per_type in members.items()
    }
    return ColumnIndex(soma_side=soma_side, members=frozen)


def column_census(
    addresses: HexAddresses,
    types: ObjectArray,
    sides: ObjectArray,
    *,
    patch_types: Sequence[str] = PATCH_TYPES,
    columnar_types: Sequence[str] = COLUMNAR_TYPES,
) -> dict[str, Any]:
    """Per-eye column counts and completeness under both declared type sets."""

    rows: list[dict[str, Any]] = []
    observed_sides = sorted(
        {key_of(sides[index]) for index in np.flatnonzero(addresses.addressed).tolist()}
    )
    for side in observed_sides:
        index = build_column_index(addresses, types, sides, soma_side=side)
        complete_all = index.complete(columnar_types)
        complete_patch = index.complete(patch_types)
        rows.append(
            {
                "soma_side": side,
                "columns": len(index.members),
                "cells": sum(
                    len(values)
                    for per_type in index.members.values()
                    for values in per_type.values()
                ),
                "columns_complete_over_all_columnar_types": len(complete_all),
                "columns_complete_over_the_patch_types": len(complete_patch),
                "cells_of_patch_types_in_complete_columns": int(
                    index.cells(complete_patch, patch_types).size
                ),
                "cells_of_all_types_in_complete_columns": int(
                    index.cells(complete_patch, columnar_types).size
                ),
            }
        )
    return {
        "definition": (
            "a column is the pair (assignedOlHex1, assignedOlHex2) within one somaSide."
            " A column is complete over a type set when it holds at least one cell of"
            " every type in that set"
        ),
        "patch_types": list(patch_types),
        "patch_type_selection_note": (
            "the eight patch types are a choice made by this analysis and are not a"
            " frozen repository rule. Any other subset of the column-addressed types"
            " would give different patch sizes; the selection is recorded so the"
            " measurement can be repeated under a different one"
        ),
        "columnar_types": list(columnar_types),
        "rows": rows,
    }


def reproduction_check(census: Mapping[str, Any], addressed_nodes: int) -> dict[str, Any]:
    """Compare the column census against the published optic-lobe figures."""

    by_side = {str(row["soma_side"]): row for row in census["rows"]}
    right = by_side.get(PATCH_SOMA_SIDE, {})
    left = by_side.get("L", {})
    checks: dict[str, tuple[Any, Any]] = {
        "column_addressed_nodes": (
            addressed_nodes,
            OPTIC_CENSUS_01["column_addressed_nodes"],
        ),
        "right_eye_columns": (
            int(right.get("columns", -1)),
            OPTIC_CENSUS_01["right_eye_columns"],
        ),
        "left_eye_columns": (
            int(left.get("columns", -1)),
            OPTIC_CENSUS_01["left_eye_columns"],
        ),
        "right_eye_columns_complete_over_all_columnar_types": (
            int(right.get("columns_complete_over_all_columnar_types", -1)),
            OPTIC_CENSUS_01["right_eye_columns_complete_over_all_columnar_types"],
        ),
        "left_eye_columns_complete_over_all_columnar_types": (
            int(left.get("columns_complete_over_all_columnar_types", -1)),
            OPTIC_CENSUS_01["left_eye_columns_complete_over_all_columnar_types"],
        ),
    }
    rows = [
        {
            "quantity": name,
            "observed": observed,
            "published": published,
            "matches": observed == published,
        }
        for name, (observed, published) in sorted(checks.items())
    ]
    return {
        "receipt": OPTIC_CENSUS_01["receipt"],
        "rule": (
            "this analysis reads the same pinned artifact and annotations table through"
            " the same verified join, so the column census must reproduce the published"
            " optic-lobe figures exactly. A deviation is a blocking error, never a new"
            " result"
        ),
        "reproduced": all(bool(row["matches"]) for row in rows),
        "checks": rows,
    }


# -------------------------------------------------------------- hex geometry


#: Candidate lattice metrics over the (hex1, hex2) address pair. The repository
#: has never frozen which of the two axes are 60 degrees apart and which are
#: 120, and the pinned table documents no convention, so the patch is measured
#: under every candidate and the spread is reported rather than hidden.
HEX_METRICS: tuple[str, ...] = ("axial_60", "axial_120", "chebyshev", "euclidean")

#: The metric whose patches are reported as the headline figures.
DEFAULT_HEX_METRIC = "axial_60"


def hex_distance(first: tuple[float, float], second: tuple[float, float], metric: str) -> float:
    """Lattice distance between two column addresses under a named convention."""

    delta_one = first[0] - second[0]
    delta_two = first[1] - second[1]
    if metric == "axial_60":
        return (abs(delta_one) + abs(delta_two) + abs(delta_one + delta_two)) / 2.0
    if metric == "axial_120":
        return (abs(delta_one) + abs(delta_two) + abs(delta_one - delta_two)) / 2.0
    if metric == "chebyshev":
        return max(abs(delta_one), abs(delta_two))
    if metric == "euclidean":
        return float(np.hypot(delta_one, delta_two))
    raise ValueError(f"unknown hex metric {metric!r}")


def hex_neighbours(column: tuple[int, int], metric: str) -> tuple[tuple[int, int], ...]:
    """The six lattice neighbours of one column under a named convention."""

    offsets: tuple[tuple[int, int], ...]
    if metric == "axial_60":
        offsets = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))
    elif metric == "axial_120":
        offsets = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1))
    else:
        offsets = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1))
    return tuple((column[0] + one, column[1] + two) for one, two in offsets)


@dataclass(frozen=True, slots=True)
class Disc:
    """One selected patch of complete columns and the rule that produced it."""

    metric: str
    centroid: tuple[float, float]
    centre: tuple[int, int]
    columns: tuple[tuple[int, int], ...]

    @property
    def radius(self) -> float:
        return max(
            (hex_distance(column, self.centre, self.metric) for column in self.columns),
            default=0.0,
        )


def select_disc(
    complete: Sequence[tuple[int, int]], count: int, *, metric: str = DEFAULT_HEX_METRIC
) -> Disc:
    """Take the ``count`` complete columns nearest the centroid of all of them.

    The rule, stated once so it can be disagreed with: the centroid is the
    arithmetic mean of the complete columns' addresses; the centre is the
    complete column closest to that centroid; the patch is the ``count`` complete
    columns nearest the centre, ordered by distance then by address so the
    selection is total and deterministic. This is this analysis's own choice of
    patch geometry, not a repository rule, and nothing downstream depends on it
    being the only defensible one.
    """

    if count > len(complete):
        raise ValueError("cannot select more columns than are complete")
    centroid = (
        sum(column[0] for column in complete) / len(complete),
        sum(column[1] for column in complete) / len(complete),
    )
    centre = min(complete, key=lambda column: (hex_distance(column, centroid, metric), column))
    ordered = sorted(complete, key=lambda column: (hex_distance(column, centre, metric), column))
    return Disc(metric=metric, centroid=centroid, centre=centre, columns=tuple(ordered[:count]))


def is_contiguous(columns: Sequence[tuple[int, int]], metric: str) -> bool:
    """Whether the selected columns form one lattice-connected component."""

    remaining = set(columns)
    if not remaining:
        return True
    frontier = [next(iter(sorted(remaining)))]
    remaining.discard(frontier[0])
    while frontier:
        column = frontier.pop()
        for neighbour in hex_neighbours(column, metric):
            if neighbour in remaining:
                remaining.discard(neighbour)
                frontier.append(neighbour)
    return not remaining
