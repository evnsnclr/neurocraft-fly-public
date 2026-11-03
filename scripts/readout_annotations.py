"""Verified retained-node annotation access for the Phase 3 readout census.

The census needs annotation strings (`superclass`, `subclass`, `type`,
`somaSide`, ...) that the G1 artifact deliberately does not carry: the artifact
stores only `annotation_source_rows`, 166,700 uint64 pointers into the pinned
source annotations table. This module owns that one join and the integrity gates
that make it trustworthy, so no other module in this analysis touches Arrow.

Method, taken verbatim in spirit from ``docs/receipts/OPTIC-CENSUS-01.md``:
retained-node values are obtained by *cross-referencing*, never by reading the
source table directly and re-deriving a retention rule. The pointer array is used
to take rows of the pinned table, and four gates must pass before any figure is
computed:

1. the pinned annotations file still hashes to the SHA-256 the artifact manifest
   declares for it;
2. the ``bodyId`` column taken through the pointers is **bitwise identical** to
   ``body_ids.bin``;
3. the per-superclass counter reconciles with the manifest's
   ``retained_by_superclass`` across all 27 keys with zero mismatches;
4. the pointers are unique and in range against the declared source row count.

A fifth gate over the edge weights lives in :mod:`scripts.readout_connectivity`
because that is where the weights are owned. The caller refuses to emit a report
unless every gate passes.

Claim discipline. Everything here is annotation transcription. A superclass or
subclass string is a *source-observed label*, not a measured function, and this
module never infers one from the other.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow.feather as feather  # type: ignore[import-untyped]

ObjectArray = npt.NDArray[np.object_]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[Any]

#: Histogram key standing in for a null annotation cell. Chosen so it can never
#: collide with a real annotation string, which never contains angle brackets.
NULL_KEY = "<unannotated>"

#: Columns materialised as object arrays for vectorised census work. Everything
#: else is read per-row, only for the handful of neurons reported in full.
CENSUS_COLUMNS: tuple[str, ...] = (
    "bodyId",
    "superclass",
    "subclass",
    "type",
    "somaSide",
    "somaNeuromere",
    "exitNerve",
    "entryNerve",
    "class",
    "instance",
    "rootSide",
)

_HASH_BLOCK_BYTES = 8 * 1024 * 1024


class ReadoutIntegrityError(RuntimeError):
    """Raised when an integrity gate fails and no report may be emitted."""


@dataclass(frozen=True, slots=True)
class IntegrityGate:
    """One pass/fail precondition, recorded in the report whether or not it holds."""

    name: str
    requirement: str
    passed: bool
    observed: Any
    expected: Any

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "requirement": self.requirement,
            "passed": self.passed,
            "observed": self.observed,
            "expected": self.expected,
        }


@dataclass(frozen=True, slots=True)
class RetainedAnnotations:
    """Annotation columns for the 166,700 retained nodes, in dense-index order."""

    body_ids: IntArray
    source_rows: IntArray
    columns: Mapping[str, ObjectArray]
    source_row_count: int
    annotations_sha256: str

    @property
    def node_count(self) -> int:
        return int(self.body_ids.size)

    def column(self, name: str) -> ObjectArray:
        try:
            return self.columns[name]
        except KeyError as exc:  # pragma: no cover - guarded by CENSUS_COLUMNS
            raise KeyError(f"annotation column {name!r} was not materialised") from exc


# ------------------------------------------------------------------ utilities


def equals(column: ObjectArray, value: str) -> BoolArray:
    """Elementwise ``column == value`` with a concrete boolean dtype.

    NumPy's object-array comparison is typed as ``Any``; forcing the dtype here
    keeps every mask in this analysis statically a boolean array.
    """

    mask: BoolArray = np.asarray(column == value, dtype=bool)
    return mask


def is_in(column: ObjectArray, values: Sequence[str]) -> BoolArray:
    """Elementwise membership over an object column, null-safe."""

    wanted = set(values)
    mask: BoolArray = np.asarray([item in wanted for item in column.tolist()], dtype=bool)
    return mask


def key_of(value: object) -> str:
    """Histogram key for one annotation cell, mapping null to :data:`NULL_KEY`."""

    return NULL_KEY if value is None else str(value)


def histogram(values: Iterable[object]) -> dict[str, int]:
    """Deterministic count-by-value, key-sorted so the JSON body is stable."""

    counter = Counter(key_of(value) for value in values)
    return {key: counter[key] for key in sorted(counter)}


def json_safe(value: object) -> Any:
    """Convert one Arrow cell to a JSON-encodable value without losing identity.

    Non-finite doubles become ``None`` rather than raising inside the canonical
    encoder, which rejects NaN by design; that substitution is recorded here so a
    reader knows a ``null`` may mean either "absent" or "non-finite".
    """

    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    return str(value)


# ------------------------------------------------------------------- loading


def file_sha256(path: Path) -> str:
    """Block-wise SHA-256 of a file on disk."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(_HASH_BLOCK_BYTES)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _object_column(table: Any, name: str) -> ObjectArray:
    values = [None if item is None else str(item) for item in table.column(name).to_pylist()]
    column: ObjectArray = np.asarray(values, dtype=object)
    return column


def take_retained_annotations(
    annotations_path: Path,
    *,
    source_rows: IntArray,
    body_ids: IntArray,
    columns: Sequence[str] = CENSUS_COLUMNS,
) -> tuple[RetainedAnnotations, Any]:
    """Take the retained rows of the pinned annotations table.

    Returns the materialised census columns and the taken Arrow table itself, so
    a caller can still read a full row for the handful of neurons it reports in
    full without materialising 36 columns for all 166,700 nodes.
    """

    observed_sha256 = file_sha256(annotations_path)
    table = feather.read_table(annotations_path, memory_map=True)
    source_row_count = int(table.num_rows)
    taken = table.take(source_rows)
    materialised = {name: _object_column(taken, name) for name in columns}
    annotations = RetainedAnnotations(
        body_ids=body_ids,
        source_rows=source_rows,
        columns=materialised,
        source_row_count=source_row_count,
        annotations_sha256=observed_sha256,
    )
    return annotations, taken


def full_row(taken: Any, dense_index: int) -> dict[str, Any]:
    """The complete annotation row of one retained node, JSON-safe and key-sorted."""

    record = taken.slice(dense_index, 1).to_pylist()[0]
    return {name: json_safe(record[name]) for name in sorted(record)}


# --------------------------------------------------------------------- gates


def annotation_gates(
    annotations: RetainedAnnotations,
    *,
    declared_annotations_sha256: str,
    declared_source_rows: int,
    retained_by_superclass: Mapping[str, int],
) -> list[IntegrityGate]:
    """The four annotation-side gates, evaluated in a fixed order."""

    taken_body_ids = np.asarray(annotations.column("bodyId").tolist(), dtype=np.uint64)
    bitwise = bool(np.array_equal(taken_body_ids, annotations.body_ids.astype(np.uint64)))

    observed = histogram(annotations.column("superclass").tolist())
    keys = sorted(set(observed) | set(retained_by_superclass))
    mismatches = {
        key: {"observed": observed.get(key, 0), "manifest": retained_by_superclass.get(key, 0)}
        for key in keys
        if observed.get(key, 0) != retained_by_superclass.get(key, 0)
    }

    rows = annotations.source_rows
    unique = int(np.unique(rows).size) == int(rows.size)
    in_range = bool(rows.size == 0 or int(rows.max()) < annotations.source_row_count)

    return [
        IntegrityGate(
            name="pinned_annotations_file_sha256",
            requirement=(
                "the pinned annotations table still hashes to the SHA-256 the artifact"
                " manifest declares for it, so the join target is the same bytes the"
                " builder read"
            ),
            passed=annotations.annotations_sha256 == declared_annotations_sha256,
            observed=annotations.annotations_sha256,
            expected=declared_annotations_sha256,
        ),
        IntegrityGate(
            name="taken_body_id_column_bitwise_identical",
            requirement=(
                "the bodyId column taken through annotation_source_rows.bin must be"
                " bitwise identical to body_ids.bin; a single misaligned pointer would"
                " silently relabel every count in this report"
            ),
            passed=bitwise,
            observed={"bitwise_identical": bitwise, "compared_values": int(taken_body_ids.size)},
            expected={"bitwise_identical": True, "compared_values": int(annotations.node_count)},
        ),
        IntegrityGate(
            name="superclass_counts_reconcile_with_manifest",
            requirement=(
                "the per-superclass counter over the taken rows must reconcile with the"
                " manifest retained_by_superclass across every key with zero mismatches"
            ),
            passed=not mismatches,
            observed={"keys_compared": len(keys), "mismatches": mismatches},
            expected={"keys_compared": len(retained_by_superclass), "mismatches": {}},
        ),
        IntegrityGate(
            name="annotation_source_rows_unique_and_in_range",
            requirement=(
                "the annotation source rows must be unique and strictly inside the"
                " declared source table, so no retained node borrows another node's row"
            ),
            passed=bool(
                unique and in_range and annotations.source_row_count == declared_source_rows
            ),
            observed={
                "unique": unique,
                "maximum_row": int(rows.max()) if rows.size else None,
                "source_table_rows": annotations.source_row_count,
            },
            expected={
                "unique": True,
                "maximum_row_below": declared_source_rows,
                "source_table_rows": declared_source_rows,
            },
        ),
    ]


def require_all_gates(gates: Sequence[IntegrityGate]) -> None:
    """Refuse to continue unless every gate passed."""

    failed = [gate.name for gate in gates if not gate.passed]
    if failed:
        raise ReadoutIntegrityError(
            "refusing to emit a readout census report; failed integrity gates: " + ", ".join(failed)
        )


def gates_report(gates: Sequence[IntegrityGate]) -> dict[str, Any]:
    """Serialize the gates, including the refusal rule itself."""

    return {
        "refusal_rule": (
            "every gate below must pass before any figure in this report is computed;"
            " a failure raises and no report file is written, so a published report is"
            " evidence that all of them held at the moment of measurement"
        ),
        "all_passed": all(gate.passed for gate in gates),
        "gates": [gate.as_json() for gate in gates],
    }
