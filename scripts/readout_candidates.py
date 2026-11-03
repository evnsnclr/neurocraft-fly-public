"""Candidate readout populations for the stage-4 body-program census.

`PLAN.md` stage 4 needs, for each behavior it wants to wire, a **recorded
population selection**: an exact selector, a member count, a sidedness split, and
an overlap refusal against everything the runtime already reads. This module
enumerates the candidate motor and descending populations from the pinned
annotation columns and produces exactly that record.

Layering, continuing the split the Phase 3 readout analysis established:
:mod:`scripts.readout_annotations` owns the verified join and its integrity
gates, this module owns candidate selection and the overlap arithmetic against
the runtime populations, and ``scripts/analyze_readout_candidates.py`` binds them
to a verified artifact and writes the document. Nothing here reopens Arrow,
re-derives a retention rule, or touches the runtime.

Method
------
Selection copies the fail-closed shape of
:func:`neurocraft_controller.escape_reflex.select_ttmn_pool`: an exact selector
over the source's own annotation strings, a recorded count, an explicit soma-side
split, and a refusal on any overlap with a population the runtime already binds.
The difference is that a census *discovers* counts rather than pinning them, so
the pinned expectations here are on the **runtime** populations only (52 wing
bodies, the TTMn pair, LC4 L/R, the visual class, the bristles) — those are the
proof that this census reads the same artifact the controller binds, and a drift
in any of them refuses the whole report.

Claim discipline
----------------
Every figure this module produces is a count of source-observed annotation
strings over the retained nodes of one pinned artifact. A ``type``, ``class``,
``subclass`` or ``somaSide`` value is a dataset label, not a measured function.
This module executes no dynamics, assigns no behavior to any neuron, and selects
nothing for the runtime: a candidate appearing here is a population that exists
with a stated shape, and nothing more.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

import numpy as np

from scripts.readout_annotations import (
    BoolArray,
    IntegrityGate,
    ObjectArray,
    RetainedAnnotations,
    equals,
    key_of,
)

#: Annotation columns this census reads, spelled exactly as the pinned table
#: spells them and as ``neurocraft_controller.embodied_senses`` imports them.
TYPE_COLUMN: Final = "type"
CLASS_COLUMN: Final = "class"
SUBCLASS_COLUMN: Final = "subclass"
SUPERCLASS_COLUMN: Final = "superclass"
SIDE_COLUMN: Final = "somaSide"

#: The two soma sides the runtime's strict selectors accept. A member outside
#: this set does not fail this census — it is reported, because a strict
#: ``select_ttmn_pool``-shaped gate would refuse the population outright.
STRICT_SIDES: Final = ("L", "R")

#: Overlap policies. ``must_be_disjoint`` refuses the report if the candidate
#: shares a body with any runtime population; ``records_overlap`` keeps the
#: candidate in the census with its overlap named and marks it ineligible.
MUST_BE_DISJOINT: Final = "must_be_disjoint"
RECORDS_OVERLAP: Final = "records_overlap"

_ALPHA_HEAD: Final = re.compile(r"^([A-Za-z]+)")


class CandidateCensusError(RuntimeError):
    """A candidate census could not be produced exactly as declared."""


class CandidateOverlapError(CandidateCensusError):
    """A candidate declared disjoint shares a body with a runtime population."""


# ------------------------------------------------------------------ selectors


@dataclass(frozen=True, slots=True)
class CandidateSelector:
    """One declared candidate population, as annotation-column predicates.

    The predicate fields and the human-readable :attr:`expression` are produced
    from the same data, so a selector string in the report can never drift from
    the mask that produced the members beside it.
    """

    name: str
    superclass: str | None = None
    subclass: str | None = None
    type_name: str | None = None
    type_alpha_prefix: str | None = None
    type_startswith: str | None = None
    excludes: tuple[str, ...] = ()
    overlap_policy: str = MUST_BE_DISJOINT
    note: str = ""

    def __post_init__(self) -> None:
        if self.overlap_policy not in {MUST_BE_DISJOINT, RECORDS_OVERLAP}:
            raise CandidateCensusError(
                f"candidate {self.name!r} declares unknown overlap policy {self.overlap_policy!r}"
            )
        if self.type_alpha_prefix is not None and self.type_startswith is not None:
            raise CandidateCensusError(
                f"candidate {self.name!r} declares both an alpha-prefix and a"
                " startswith rule over the same column; the two disagree by"
                " construction and a report may not leave which one ran ambiguous"
            )
        if not any(
            (
                self.superclass,
                self.subclass,
                self.type_name,
                self.type_alpha_prefix,
                self.type_startswith,
            )
        ):
            raise CandidateCensusError(
                f"candidate {self.name!r} declares no annotation predicate; a selector"
                " that matches every retained node is never a readout candidate"
            )

    @property
    def expression(self) -> str:
        """The selector exactly as it is evaluated, in one readable string."""

        terms: list[str] = []
        if self.superclass is not None:
            terms.append(f"{SUPERCLASS_COLUMN} == '{self.superclass}'")
        if self.subclass is not None:
            terms.append(f"{SUBCLASS_COLUMN} == '{self.subclass}'")
        if self.type_name is not None:
            terms.append(f"{TYPE_COLUMN} == '{self.type_name}'")
        if self.type_alpha_prefix is not None:
            terms.append(f"alpha_prefix({TYPE_COLUMN}) == '{self.type_alpha_prefix}'")
        if self.type_startswith is not None:
            terms.append(f"{TYPE_COLUMN}.startswith('{self.type_startswith}')")
        expression = " and ".join(terms)
        for name in self.excludes:
            expression += f", minus {name}"
        return expression


def alpha_prefix(value: object) -> str | None:
    """The maximal leading run of ASCII letters in a type string, or ``None``.

    The rule ``scripts.readout_census`` already uses: ``DNp09 -> DNp`` and
    ``DNpe001 -> DNpe``, so prefix classes are disjoint rather than nested and a
    ``DNp`` selector can never silently sweep in ``DNpe``. A naive
    ``startswith`` filter does exactly that; the two rules are measured side by
    side in the report so the difference is a number rather than a footnote.
    """

    if value is None:
        return None
    match = _ALPHA_HEAD.match(str(value))
    return match.group(1) if match else None


def _prefix_mask(column: ObjectArray, prefix: str) -> BoolArray:
    mask: BoolArray = np.asarray(
        [alpha_prefix(item) == prefix for item in column.tolist()], dtype=bool
    )
    return mask


def _startswith_mask(column: ObjectArray, prefix: str) -> BoolArray:
    mask: BoolArray = np.asarray(
        [item is not None and str(item).startswith(prefix) for item in column.tolist()],
        dtype=bool,
    )
    return mask


def _body_mask(body_ids: Any, excluded: Iterable[int]) -> BoolArray:
    wanted = set(excluded)
    mask: BoolArray = np.asarray([int(body) in wanted for body in body_ids.tolist()], dtype=bool)
    return mask


# ------------------------------------------------------- runtime populations


@dataclass(frozen=True, slots=True)
class RuntimePopulation:
    """One population the shipped runtime already reads or writes.

    ``origin`` records where the membership came from, because the two kinds
    behave differently under artifact drift: a literal tuple in the source is
    fixed, while an annotation selector is re-derived from the pinned table on
    every run and is what the count gate below actually guards.
    """

    name: str
    selector: str
    origin: str
    body_ids: frozenset[int]
    soma_side_counts: tuple[tuple[str, int], ...] = ()

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "selector": self.selector,
            "origin": self.origin,
            "members": len(self.body_ids),
            "soma_side_counts": dict(self.soma_side_counts),
            "body_ids": sorted(self.body_ids),
        }


def _selected_body_ids(annotations: RetainedAnnotations, mask: BoolArray) -> frozenset[int]:
    return frozenset(int(body) for body in annotations.body_ids[mask].tolist())


def _soma_side_counts(
    annotations: RetainedAnnotations, body_ids: frozenset[int]
) -> tuple[tuple[str, int], ...]:
    """The pinned table's ``somaSide`` histogram over an arbitrary body set.

    Recorded for the runtime populations too, not only the candidates: the
    declared wing readout is a literal list of bodies, and how it happens to
    split by side is a measured property of that list rather than something its
    two pool names guarantee.
    """

    sides = annotations.column(SIDE_COLUMN)
    counts: dict[str, int] = {}
    for position, body in enumerate(annotations.body_ids.tolist()):
        if int(body) in body_ids:
            side = key_of(sides[position])
            counts[side] = counts.get(side, 0) + 1
    return tuple((key, counts[key]) for key in sorted(counts))


def runtime_populations(
    annotations: RetainedAnnotations,
    *,
    wing_body_ids: Sequence[int],
    faithful_sensory_body_ids: Sequence[int],
    faithful_motor_body_ids: Sequence[int],
) -> tuple[RuntimePopulation, ...]:
    """Every population the runtime already binds, re-derived where it is a selector.

    The wing readout and the four faithful boundary bodies are literals in the
    controller source and are passed in; LC4, the visual class, the bristles and
    the TTMn pair are annotation selectors and are re-derived here from the same
    columns the controller binds them from.
    """

    types = annotations.column(TYPE_COLUMN)
    classes = annotations.column(CLASS_COLUMN)
    subclasses = annotations.column(SUBCLASS_COLUMN)
    sides = annotations.column(SIDE_COLUMN)
    lc4 = equals(types, "LC4")
    declared = (
        RuntimePopulation(
            name="wing_motor_readout_52",
            selector="declared 52-body wing-motor readout (left + right pools)",
            origin="neurocraft_controller.flight_pools literal",
            body_ids=frozenset(int(body) for body in wing_body_ids),
        ),
        RuntimePopulation(
            name="ttmn_jump_pool",
            selector=f"{TYPE_COLUMN} == 'TTMn'",
            origin="neurocraft_controller.escape_reflex selector, re-derived here",
            body_ids=_selected_body_ids(annotations, equals(types, "TTMn")),
        ),
        RuntimePopulation(
            name="lc4_left",
            selector=f"{TYPE_COLUMN} == 'LC4' and {SIDE_COLUMN} == 'L'",
            origin="neurocraft_controller.embodied_senses selector, re-derived here",
            body_ids=_selected_body_ids(annotations, lc4 & equals(sides, "L")),
        ),
        RuntimePopulation(
            name="lc4_right",
            selector=f"{TYPE_COLUMN} == 'LC4' and {SIDE_COLUMN} == 'R'",
            origin="neurocraft_controller.embodied_senses selector, re-derived here",
            body_ids=_selected_body_ids(annotations, lc4 & equals(sides, "R")),
        ),
        RuntimePopulation(
            name="visual_class",
            selector=f"{CLASS_COLUMN} == 'visual'",
            origin="neurocraft_controller.embodied_senses selector, re-derived here",
            body_ids=_selected_body_ids(annotations, equals(classes, "visual")),
        ),
        RuntimePopulation(
            name="mechanosensory_bristle",
            selector=f"{SUBCLASS_COLUMN} == 'mechanosensory bristle'",
            origin="neurocraft_controller.embodied_senses selector, re-derived here",
            body_ids=_selected_body_ids(annotations, equals(subclasses, "mechanosensory bristle")),
        ),
        RuntimePopulation(
            name="faithful_sensory_entries",
            selector="ControllerConfiguration left_sensor_body_id / right_sensor_body_id",
            origin="neurocraft_controller.config literal",
            body_ids=frozenset(int(body) for body in faithful_sensory_body_ids),
        ),
        RuntimePopulation(
            name="faithful_motor_entries",
            selector="ControllerConfiguration left_motor_body_id / right_motor_body_id",
            origin="neurocraft_controller.config literal",
            body_ids=frozenset(int(body) for body in faithful_motor_body_ids),
        ),
    )
    return tuple(
        replace(
            population,
            soma_side_counts=_soma_side_counts(annotations, population.body_ids),
        )
        for population in declared
    )


def runtime_population_gates(
    populations: Sequence[RuntimePopulation],
    expected: Mapping[str, int],
) -> list[IntegrityGate]:
    """One gate per runtime population, pinning the count the runtime declares.

    These are the census's proof of identity with the controller. A selector that
    no longer yields the count ``embodied_senses`` / ``escape_reflex`` /
    ``flight_pools`` declare means this census is reading a different artifact
    than the one the runtime binds, and every overlap figure below would be about
    the wrong bodies.
    """

    by_name = {population.name: population for population in populations}
    missing = sorted(set(expected) - set(by_name))
    gates: list[IntegrityGate] = [
        IntegrityGate(
            name="runtime_populations_all_present",
            requirement=(
                "every runtime population this census measures overlap against must"
                " have been derived; a missing one would silently drop an overlap check"
            ),
            passed=not missing,
            observed={"derived": sorted(by_name), "missing": missing},
            expected={"derived": sorted(expected), "missing": []},
        )
    ]
    for name in sorted(expected):
        observed = len(by_name[name].body_ids) if name in by_name else None
        gates.append(
            IntegrityGate(
                name=f"runtime_population_count_{name}",
                requirement=(
                    f"the {name!r} population must carry exactly the member count the"
                    " shipped runtime declares for it, so this census is demonstrably"
                    " reading the same artifact the controller binds"
                ),
                passed=observed == expected[name],
                observed=observed,
                expected=expected[name],
            )
        )
    return gates


# --------------------------------------------------------------- the census


@dataclass(frozen=True, slots=True)
class CandidateCensus:
    """One candidate population as measured: members, sides, and overlaps."""

    selector: CandidateSelector
    body_ids: tuple[int, ...]
    dense_indices: tuple[int, ...]
    soma_sides: tuple[str, ...]
    overlaps: tuple[tuple[str, tuple[int, ...]], ...]

    def __post_init__(self) -> None:
        if len(self.body_ids) != len(self.dense_indices) or len(self.body_ids) != len(
            self.soma_sides
        ):
            raise CandidateCensusError(f"candidate {self.selector.name!r} members are misaligned")
        pairs = zip(self.body_ids, self.body_ids[1:], strict=False)
        if any(later <= earlier for earlier, later in pairs):
            raise CandidateCensusError(
                f"candidate {self.selector.name!r} body IDs must ascend strictly"
            )

    @property
    def members(self) -> int:
        return len(self.body_ids)

    @property
    def side_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for side in self.soma_sides:
            counts[side] = counts.get(side, 0) + 1
        return {key: counts[key] for key in sorted(counts)}

    @property
    def non_strict_side_members(self) -> tuple[tuple[int, str], ...]:
        """Members whose ``somaSide`` is neither ``L`` nor ``R``, with their value."""

        return tuple(
            (body, side)
            for body, side in zip(self.body_ids, self.soma_sides, strict=True)
            if side not in STRICT_SIDES
        )

    @property
    def strict_side_gate_passes(self) -> bool:
        """Whether a ``select_ttmn_pool``-shaped strict L/R gate would accept this pool."""

        return self.members > 0 and not self.non_strict_side_members

    @property
    def balanced(self) -> bool:
        counts = self.side_counts
        return self.strict_side_gate_passes and counts.get("L", 0) == counts.get("R", 0)

    @property
    def overlapping_populations(self) -> tuple[str, ...]:
        return tuple(name for name, bodies in self.overlaps if bodies)

    @property
    def disjoint(self) -> bool:
        return not self.overlapping_populations

    @property
    def eligible(self) -> bool:
        """Whether this population could be bound by a strict, disjoint selector.

        Eligibility is an *anatomical* verdict and nothing else: the population
        exists, splits cleanly into left and right, and shares no body with
        anything the runtime already reads. It says nothing about whether the
        population carries any signal, and nothing about whether it should be
        wired.
        """

        return self.disjoint and self.strict_side_gate_passes

    def as_json(self) -> dict[str, Any]:
        counts = self.side_counts
        return {
            "name": self.selector.name,
            "selector": self.selector.expression,
            "overlap_policy": self.selector.overlap_policy,
            "note": self.selector.note,
            "members": self.members,
            "soma_side_counts": counts,
            "left": counts.get("L", 0),
            "right": counts.get("R", 0),
            "non_strict_side_members": [
                {"body_id": body, "soma_side": side} for body, side in self.non_strict_side_members
            ],
            "strict_side_gate_passes": self.strict_side_gate_passes,
            "balanced_left_right": self.balanced,
            "disjoint_from_runtime_populations": self.disjoint,
            "anatomically_eligible": self.eligible,
            "overlaps": {name: list(bodies) for name, bodies in self.overlaps if bodies},
            "body_ids": list(self.body_ids),
            "dense_indices": list(self.dense_indices),
            "member_soma_sides": list(self.soma_sides),
        }


def candidate_mask(
    annotations: RetainedAnnotations,
    selector: CandidateSelector,
    populations: Sequence[RuntimePopulation],
) -> BoolArray:
    """The dense-index mask of one candidate selector, exclusions applied."""

    mask: BoolArray = np.ones(annotations.node_count, dtype=bool)
    if selector.superclass is not None:
        mask &= equals(annotations.column(SUPERCLASS_COLUMN), selector.superclass)
    if selector.subclass is not None:
        mask &= equals(annotations.column(SUBCLASS_COLUMN), selector.subclass)
    if selector.type_name is not None:
        mask &= equals(annotations.column(TYPE_COLUMN), selector.type_name)
    if selector.type_alpha_prefix is not None:
        mask &= _prefix_mask(annotations.column(TYPE_COLUMN), selector.type_alpha_prefix)
    if selector.type_startswith is not None:
        mask &= _startswith_mask(annotations.column(TYPE_COLUMN), selector.type_startswith)
    if selector.excludes:
        by_name = {population.name: population for population in populations}
        for name in selector.excludes:
            if name not in by_name:
                raise CandidateCensusError(
                    f"candidate {selector.name!r} excludes unknown population {name!r}"
                )
            mask &= ~_body_mask(annotations.body_ids, by_name[name].body_ids)
    return mask


def census_candidate(
    annotations: RetainedAnnotations,
    selector: CandidateSelector,
    populations: Sequence[RuntimePopulation],
) -> CandidateCensus:
    """Measure one candidate: its members, their sides, and every overlap."""

    mask = candidate_mask(annotations, selector, populations)
    dense = np.flatnonzero(mask).astype(np.int64).tolist()
    sides = annotations.column(SIDE_COLUMN)
    body_ids = [int(annotations.body_ids[index]) for index in dense]
    order = sorted(range(len(dense)), key=lambda position: body_ids[position])
    ordered_bodies = tuple(body_ids[position] for position in order)
    member_set = set(ordered_bodies)
    return CandidateCensus(
        selector=selector,
        body_ids=ordered_bodies,
        dense_indices=tuple(int(dense[position]) for position in order),
        soma_sides=tuple(key_of(sides[dense[position]]) for position in order),
        overlaps=tuple(
            (population.name, tuple(sorted(member_set & population.body_ids)))
            for population in populations
        ),
    )


def census_candidates(
    annotations: RetainedAnnotations,
    selectors: Sequence[CandidateSelector],
    populations: Sequence[RuntimePopulation],
) -> tuple[CandidateCensus, ...]:
    """Measure every candidate, in declaration order."""

    return tuple(census_candidate(annotations, selector, populations) for selector in selectors)


def require_declared_overlaps(censuses: Sequence[CandidateCensus]) -> None:
    """Refuse the report unless every ``must_be_disjoint`` candidate is disjoint.

    This is the census-side counterpart of
    :func:`neurocraft_controller.escape_reflex._refuse_ttmn_overlaps`. A candidate
    that captures a body the runtime already reads is not a new readout, it is a
    silent second reader of an existing one, and the difference is invisible once
    the population is pooled.
    """

    offenders = [
        f"{census.selector.name} overlaps "
        + ", ".join(f"{name} on {list(bodies)}" for name, bodies in census.overlaps if bodies)
        for census in censuses
        if census.selector.overlap_policy == MUST_BE_DISJOINT and not census.disjoint
    ]
    if offenders:
        raise CandidateOverlapError(
            "refusing to emit a candidate census; declared-disjoint candidates capture"
            " bodies the runtime already reads: " + "; ".join(offenders)
        )


def candidate_cross_overlaps(
    censuses: Sequence[CandidateCensus],
) -> tuple[dict[str, Any], ...]:
    """Every pair of candidates that share at least one body.

    Candidates are allowed to nest — ``MN9`` sits inside the proboscis pool and
    ``DNa01`` inside the ``DNa`` prefix aggregate — but two nested candidates are
    not two independent readouts, and pooling both would read the same neurons
    twice under two names. Recorded so that is visible when a pair is chosen.
    """

    rows: list[dict[str, Any]] = []
    for position, left in enumerate(censuses):
        left_bodies = set(left.body_ids)
        for right in censuses[position + 1 :]:
            shared = sorted(left_bodies & set(right.body_ids))
            if not shared:
                continue
            rows.append(
                {
                    "candidates": [left.selector.name, right.selector.name],
                    "shared_members": len(shared),
                    "relation": (
                        "identical"
                        if len(shared) == left.members == right.members
                        else "first_contains_second"
                        if len(shared) == right.members
                        else "second_contains_first"
                        if len(shared) == left.members
                        else "partial"
                    ),
                }
            )
    return tuple(rows)


def require_nonempty(censuses: Sequence[CandidateCensus]) -> None:
    """Refuse the report if any declared candidate selected nothing.

    An empty candidate is never a negative result about the fly — it means the
    selector does not match the pinned table's own spelling, and every count
    beside it was measured against a different vocabulary than intended.
    """

    empty = sorted(census.selector.name for census in censuses if census.members == 0)
    if empty:
        raise CandidateCensusError(
            "refusing to emit a candidate census; these selectors matched no retained"
            " node, which means the selector is wrong rather than the population"
            f" absent: {', '.join(empty)}"
        )
