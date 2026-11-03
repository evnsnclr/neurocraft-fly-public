"""The stage-4 candidate census must be exact on data whose answers are hand-checkable.

Every test here builds a small synthetic annotation frame in memory, so the suite
never needs the ignored full artifact and runs identically on CI, where no local
data exists at all. Nothing in this module opens a path, reads Arrow, or
constructs a controller: :class:`RetainedAnnotations` is a plain frozen record of
NumPy object columns, which is exactly the seam the census was layered on.

The refusal tests follow the discipline ``tests/controller/test_escape_reflex.py``
established: each one arranges for every *preceding* gate to pass, so the gate it
is actually testing is the one that fires. A refusal test that trips an earlier
gate proves nothing about the later one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.analyze_readout_candidates import (
    ACTIVITY_CANDIDATES,
    ACTIVITY_CONDITIONS,
    CANDIDATE_SELECTORS,
    RUNTIME_POPULATION_EXPECTED,
    ReadoutCandidateError,
    activity_pools,
    build_census,
    document_bytes,
    render_markdown,
    write_documents,
)
from scripts.readout_annotations import (
    IntegrityGate,
    ReadoutIntegrityError,
    RetainedAnnotations,
    annotation_gates,
)
from scripts.readout_candidates import (
    MUST_BE_DISJOINT,
    RECORDS_OVERLAP,
    CandidateCensusError,
    CandidateOverlapError,
    CandidateSelector,
    alpha_prefix,
    candidate_cross_overlaps,
    census_candidate,
    census_candidates,
    require_declared_overlaps,
    require_nonempty,
    runtime_population_gates,
    runtime_populations,
)

#: The two faithful motor entries the shipped controller declares. 800241 is the
#: one the 52-body wing readout does NOT contain, which is why an "everything in
#: the wing subclass we do not already read" selector captures it.
LEFT_MOTOR_BODY = 800_241
RIGHT_MOTOR_BODY = 800_917
LEFT_SENSOR_BODY = 805_722
RIGHT_SENSOR_BODY = 77_185

SYNTHETIC_SHA = "a" * 64

# A synthetic census written row-major so the whole frame is readable at once.
# Dense index is row order. The rows are chosen to exercise every predicate: two
# spellings of a DN prefix, a midline soma side, both faithful motor entries, a
# TTMn pair, LC4 on both sides, a visual cell, and a bristle.
COLUMNS = ("bodyId", "superclass", "subclass", "type", "class", "somaSide")
ROWS: tuple[tuple[Any, ...], ...] = (
    (10_001, "descending_neuron", "xn", "DNp01", None, "L"),
    (10_010, "descending_neuron", "xn", "DNp01", None, "R"),
    (10_442, "descending_neuron", "ut", "DNa01", None, "L"),
    (10_760, "descending_neuron", "ut", "DNa01", None, "R"),
    (12_885, "descending_neuron", "xn", "DNpe001", None, "M"),
    (13_000, "descending_neuron", "xn", "DNpe002", None, "L"),
    (20_000, "cb_motor", "nm", "CvN4", None, "L"),
    (20_001, "cb_motor", "nm", "CvN4", None, "R"),
    (30_000, "vnc_motor", "hm", "hi1 MN", None, "L"),
    (30_001, "vnc_motor", "hm", "hi1 MN", None, "R"),
    (800_146, "vnc_motor", "wm", "TTMn", None, "R"),
    (804_642, "vnc_motor", "wm", "TTMn", None, "L"),
    (LEFT_MOTOR_BODY, "vnc_motor", "wm", "hg1 MN", None, "L"),
    (RIGHT_MOTOR_BODY, "vnc_motor", "wm", "hg1 MN", None, "R"),
    (800_999, "vnc_motor", "wm", "b1 MN", None, "L"),
    (60_000, "visual_projection", None, "LC4", None, "L"),
    (60_001, "visual_projection", None, "LC4", None, "R"),
    (70_000, "cb_intrinsic", None, "R1-R6", "visual", None),
    (LEFT_SENSOR_BODY, "vnc_sensory", "campaniform sensilla", "SNpp10", None, None),
    (RIGHT_SENSOR_BODY, "cb_sensory", "mechanosensory bristle", "JO-EV3", None, None),
)

#: The declared 52-body wing readout, standing in at synthetic scale: the right
#: faithful motor body is inside it and the left one is not, which is the shape
#: the real pool has.
SYNTHETIC_WING_POOL = (RIGHT_MOTOR_BODY,)

SYNTHETIC_EXPECTED = {
    "wing_motor_readout_52": len(SYNTHETIC_WING_POOL),
    "ttmn_jump_pool": 2,
    "lc4_left": 1,
    "lc4_right": 1,
    "visual_class": 1,
    "mechanosensory_bristle": 1,
    "faithful_sensory_entries": 2,
    "faithful_motor_entries": 2,
}


def _annotations(rows: tuple[tuple[Any, ...], ...] = ROWS) -> RetainedAnnotations:
    """A synthetic retained-annotation frame in dense row order."""

    columns = {
        name: np.asarray([row[position] for row in rows], dtype=object)
        for position, name in enumerate(COLUMNS)
    }
    body_ids = np.asarray([row[0] for row in rows], dtype=np.uint64)
    return RetainedAnnotations(
        body_ids=body_ids,
        source_rows=np.arange(len(rows), dtype=np.uint64),
        columns=columns,
        source_row_count=len(rows),
        annotations_sha256=SYNTHETIC_SHA,
    )


def _populations(annotations: RetainedAnnotations | None = None) -> Any:
    frame = annotations if annotations is not None else _annotations()
    return runtime_populations(
        frame,
        wing_body_ids=SYNTHETIC_WING_POOL,
        faithful_sensory_body_ids=(LEFT_SENSOR_BODY, RIGHT_SENSOR_BODY),
        faithful_motor_body_ids=(LEFT_MOTOR_BODY, RIGHT_MOTOR_BODY),
    )


def _join_gates(annotations: RetainedAnnotations) -> list[IntegrityGate]:
    """The four real annotation-join gates, arranged to pass on the synthetic frame.

    Built rather than faked so every refusal test below trips the gate it names
    instead of an earlier one.
    """

    superclasses: dict[str, int] = {}
    for value in annotations.column("superclass").tolist():
        key = str(value)
        superclasses[key] = superclasses.get(key, 0) + 1
    return annotation_gates(
        annotations,
        declared_annotations_sha256=SYNTHETIC_SHA,
        declared_source_rows=annotations.source_row_count,
        retained_by_superclass=superclasses,
    )


#: The synthetic selector set the whole-document tests use. It carries the two
#: ordinary candidates plus the four family aggregates, because the rendered
#: cautions are computed from those four rows rather than typed as prose and the
#: renderer refuses a census that cannot supply them.
DEFAULT_SYNTHETIC_SELECTORS = (
    CandidateSelector(name="dna01", type_name="DNa01"),
    CandidateSelector(name="neck", superclass="cb_motor", subclass="nm"),
    CandidateSelector(
        name="dna_prefix_aggregate", superclass="descending_neuron", type_alpha_prefix="DNa"
    ),
    CandidateSelector(
        name="dnp_prefix_aggregate", superclass="descending_neuron", type_alpha_prefix="DNp"
    ),
    CandidateSelector(
        name="dna_startswith_aggregate", superclass="descending_neuron", type_startswith="DNa"
    ),
    CandidateSelector(
        name="dnp_startswith_aggregate", superclass="descending_neuron", type_startswith="DNp"
    ),
)


def _census(**overrides: Any) -> dict[str, Any]:
    annotations = _annotations()
    kwargs: dict[str, Any] = {
        "identity": {
            "artifact_id": "synthetic",
            "manifest_file_sha256": SYNTHETIC_SHA,
            "node_count": len(ROWS),
            "edge_count": 0,
        },
        "join_gates": _join_gates(annotations),
        "selectors": DEFAULT_SYNTHETIC_SELECTORS,
        "expected": SYNTHETIC_EXPECTED,
        "wing_body_ids": SYNTHETIC_WING_POOL,
        "faithful_sensory_body_ids": (LEFT_SENSOR_BODY, RIGHT_SENSOR_BODY),
        "faithful_motor_body_ids": (LEFT_MOTOR_BODY, RIGHT_MOTOR_BODY),
    }
    kwargs.update(overrides)
    return build_census(annotations, **kwargs)


# ----------------------------------------------------------------- selectors


def test_selector_expression_is_built_from_the_predicates_that_run() -> None:
    selector = CandidateSelector(
        name="off_pool",
        superclass="vnc_motor",
        subclass="wm",
        excludes=("wing_motor_readout_52", "ttmn_jump_pool"),
    )
    assert selector.expression == (
        "superclass == 'vnc_motor' and subclass == 'wm',"
        " minus wing_motor_readout_52, minus ttmn_jump_pool"
    )


def test_selector_refuses_a_predicate_free_selector() -> None:
    with pytest.raises(CandidateCensusError, match="declares no annotation predicate"):
        CandidateSelector(name="everything")


def test_selector_refuses_two_disagreeing_prefix_rules() -> None:
    with pytest.raises(CandidateCensusError, match="alpha-prefix and a startswith"):
        CandidateSelector(name="both", type_alpha_prefix="DNp", type_startswith="DNp")


def test_selector_refuses_an_unknown_overlap_policy() -> None:
    with pytest.raises(CandidateCensusError, match="unknown overlap policy"):
        CandidateSelector(name="odd", type_name="DNa01", overlap_policy="whatever")


def test_alpha_prefix_separates_dnp_from_dnpe() -> None:
    assert alpha_prefix("DNp01") == "DNp"
    assert alpha_prefix("DNpe001") == "DNpe"
    assert alpha_prefix("DNg02_a") == "DNg"
    assert alpha_prefix(None) is None


# ------------------------------------------------------------------ selection


def test_subclass_selector_takes_exactly_its_members_in_dense_order() -> None:
    annotations = _annotations()
    census = census_candidate(
        annotations,
        CandidateSelector(name="neck", superclass="cb_motor", subclass="nm"),
        _populations(annotations),
    )
    assert census.members == 2
    assert census.body_ids == (20_000, 20_001)
    assert census.dense_indices == (6, 7)
    assert census.side_counts == {"L": 1, "R": 1}
    assert census.strict_side_gate_passes
    assert census.balanced
    assert census.eligible


def test_startswith_sweeps_in_what_the_alpha_prefix_rule_keeps_out() -> None:
    annotations = _annotations()
    populations = _populations(annotations)
    prefixed = census_candidate(
        annotations,
        CandidateSelector(
            name="dnp_prefix", superclass="descending_neuron", type_alpha_prefix="DNp"
        ),
        populations,
    )
    naive = census_candidate(
        annotations,
        CandidateSelector(
            name="dnp_startswith", superclass="descending_neuron", type_startswith="DNp"
        ),
        populations,
    )
    assert prefixed.body_ids == (10_001, 10_010)
    assert naive.body_ids == (10_001, 10_010, 12_885, 13_000)
    assert set(prefixed.body_ids) < set(naive.body_ids)


def test_a_midline_member_fails_the_strict_side_gate_without_failing_the_census() -> None:
    annotations = _annotations()
    census = census_candidate(
        annotations,
        CandidateSelector(
            name="dnp_startswith", superclass="descending_neuron", type_startswith="DNp"
        ),
        _populations(annotations),
    )
    assert census.side_counts == {"L": 2, "M": 1, "R": 1}
    assert census.non_strict_side_members == ((12_885, "M"),)
    assert not census.strict_side_gate_passes
    assert census.disjoint
    assert not census.eligible


def test_exclusion_subtracts_the_named_runtime_populations() -> None:
    annotations = _annotations()
    populations = _populations(annotations)
    whole = census_candidate(
        annotations,
        CandidateSelector(
            name="wm_all", superclass="vnc_motor", subclass="wm", overlap_policy=RECORDS_OVERLAP
        ),
        populations,
    )
    trimmed = census_candidate(
        annotations,
        CandidateSelector(
            name="wm_off_pool",
            superclass="vnc_motor",
            subclass="wm",
            excludes=("wing_motor_readout_52", "ttmn_jump_pool"),
            overlap_policy=RECORDS_OVERLAP,
        ),
        populations,
    )
    assert whole.body_ids == (800_146, LEFT_MOTOR_BODY, RIGHT_MOTOR_BODY, 800_999, 804_642)
    assert trimmed.body_ids == (LEFT_MOTOR_BODY, 800_999)


def test_excluding_an_unknown_population_refuses() -> None:
    annotations = _annotations()
    with pytest.raises(CandidateCensusError, match="excludes unknown population"):
        census_candidate(
            annotations,
            CandidateSelector(name="oops", superclass="vnc_motor", excludes=("no_such_pool",)),
            _populations(annotations),
        )


# ------------------------------------------------------------- overlap gates


def test_overlap_refusal_catches_the_faithful_motor_entry_the_wing_pool_omits() -> None:
    """The 800241 case, which the shipped escape-reflex gate would let through.

    Subtracting the declared wing readout and the TTMn pair leaves a wing-motor
    remainder that still contains the left faithful motor entry, because the
    52-body pool contains only the right one.
    """

    annotations = _annotations()
    censuses = census_candidates(
        annotations,
        (
            CandidateSelector(
                name="wm_off_pool",
                superclass="vnc_motor",
                subclass="wm",
                excludes=("wing_motor_readout_52", "ttmn_jump_pool"),
                overlap_policy=MUST_BE_DISJOINT,
            ),
        ),
        _populations(annotations),
    )
    assert LEFT_MOTOR_BODY in censuses[0].body_ids
    with pytest.raises(CandidateOverlapError) as raised:
        require_declared_overlaps(censuses)
    assert "faithful_motor_entries" in str(raised.value)
    assert str(LEFT_MOTOR_BODY) in str(raised.value)


def test_excluding_the_faithful_motor_entries_as_well_is_accepted() -> None:
    annotations = _annotations()
    censuses = census_candidates(
        annotations,
        (
            CandidateSelector(
                name="wm_off_pool_disjoint",
                superclass="vnc_motor",
                subclass="wm",
                excludes=(
                    "wing_motor_readout_52",
                    "ttmn_jump_pool",
                    "faithful_motor_entries",
                ),
            ),
        ),
        _populations(annotations),
    )
    require_declared_overlaps(censuses)
    assert censuses[0].body_ids == (800_999,)
    assert censuses[0].eligible


def test_a_records_overlap_candidate_survives_but_is_never_eligible() -> None:
    annotations = _annotations()
    censuses = census_candidates(
        annotations,
        (
            CandidateSelector(
                name="wm_all",
                superclass="vnc_motor",
                subclass="wm",
                overlap_policy=RECORDS_OVERLAP,
            ),
        ),
        _populations(annotations),
    )
    require_declared_overlaps(censuses)
    row = censuses[0].as_json()
    assert row["overlaps"]["ttmn_jump_pool"] == [800_146, 804_642]
    assert row["overlaps"]["faithful_motor_entries"] == [LEFT_MOTOR_BODY, RIGHT_MOTOR_BODY]
    assert not row["anatomically_eligible"]


def test_an_empty_selector_refuses_rather_than_reporting_a_negative_result() -> None:
    annotations = _annotations()
    censuses = census_candidates(
        annotations,
        (CandidateSelector(name="typo", type_name="DNa01_typo"),),
        _populations(annotations),
    )
    assert censuses[0].members == 0
    with pytest.raises(CandidateCensusError, match="matched no retained node"):
        require_nonempty(censuses)


def test_cross_overlaps_name_the_containment_direction() -> None:
    annotations = _annotations()
    populations = _populations(annotations)
    censuses = census_candidates(
        annotations,
        (
            CandidateSelector(name="dna01", type_name="DNa01"),
            CandidateSelector(
                name="dna_all", superclass="descending_neuron", type_alpha_prefix="DNa"
            ),
            CandidateSelector(name="neck", superclass="cb_motor", subclass="nm"),
        ),
        populations,
    )
    rows = candidate_cross_overlaps(censuses)
    assert rows == (
        {
            "candidates": ["dna01", "dna_all"],
            "shared_members": 2,
            "relation": "identical",
        },
    )


# --------------------------------------------------------- count/side gates


def test_runtime_population_counts_are_measured_and_recorded() -> None:
    gates = runtime_population_gates(_populations(), SYNTHETIC_EXPECTED)
    assert all(gate.passed for gate in gates)
    names = {gate.name for gate in gates}
    assert "runtime_populations_all_present" in names
    assert "runtime_population_count_ttmn_jump_pool" in names


def test_a_drifted_runtime_population_count_refuses_the_whole_report() -> None:
    drifted = dict(SYNTHETIC_EXPECTED)
    drifted["ttmn_jump_pool"] = 3
    with pytest.raises(ReadoutIntegrityError, match="runtime_population_count_ttmn_jump_pool"):
        _census(expected=drifted)


def test_a_missing_runtime_population_refuses_the_whole_report() -> None:
    unknown = dict(SYNTHETIC_EXPECTED)
    unknown["a_population_that_does_not_exist"] = 1
    with pytest.raises(ReadoutIntegrityError, match="runtime_populations_all_present"):
        _census(expected=unknown)


def test_build_census_refuses_an_overlapping_declared_disjoint_candidate() -> None:
    """The overlap refusal must fire through ``build_census``, not only in isolation.

    Every preceding gate is arranged to pass — the join gates are the real ones
    over the synthetic frame, and the runtime counts match — so the failure this
    asserts is the overlap refusal itself.
    """

    with pytest.raises(CandidateOverlapError, match="faithful_motor_entries"):
        _census(
            selectors=(
                CandidateSelector(
                    name="wm_off_pool",
                    superclass="vnc_motor",
                    subclass="wm",
                    excludes=("wing_motor_readout_52", "ttmn_jump_pool"),
                ),
            )
        )


def test_build_census_refuses_a_selector_that_matched_nothing() -> None:
    """The non-empty refusal must fire through ``build_census`` too.

    An empty candidate means the selector disagrees with the pinned table's own
    spelling; reporting it as a count of zero would read as a fact about the fly.
    """

    with pytest.raises(CandidateCensusError, match="matched no retained node"):
        _census(selectors=(CandidateSelector(name="typo", type_name="DNa01_typo"),))


def test_a_failed_join_gate_refuses_before_any_candidate_is_measured() -> None:
    annotations = _annotations()
    gates = annotation_gates(
        annotations,
        declared_annotations_sha256="b" * 64,
        declared_source_rows=annotations.source_row_count,
        retained_by_superclass={},
    )
    with pytest.raises(ReadoutIntegrityError, match="pinned_annotations_file_sha256"):
        _census(join_gates=gates)


def test_runtime_populations_record_their_own_soma_side_split() -> None:
    wing = next(
        population for population in _populations() if population.name == "wing_motor_readout_52"
    )
    assert dict(wing.soma_side_counts) == {"R": 1}


# ------------------------------------------------------- the declared inputs


def test_every_declared_candidate_name_is_unique() -> None:
    names = [selector.name for selector in CANDIDATE_SELECTORS]
    assert len(names) == len(set(names))


def test_the_expected_counts_cover_exactly_the_runtime_populations() -> None:
    derived = {population.name for population in _populations()}
    assert derived == set(RUNTIME_POPULATION_EXPECTED)


def test_the_activity_pass_names_candidates_the_census_declares() -> None:
    declared = {selector.name for selector in CANDIDATE_SELECTORS}
    assert set(ACTIVITY_CANDIDATES) <= declared


def test_the_six_held_conditions_are_declared_once_each() -> None:
    names = [name for name, _slot, _light, _collision in ACTIVITY_CONDITIONS]
    assert len(names) == len(set(names)) == 6


# ------------------------------------------------------------ census + output


def test_the_whole_census_reports_gates_populations_and_candidates() -> None:
    census = _census()
    assert census["integrity_gates"]["all_passed"] is True
    assert len(census["integrity_gates"]["gates"]) == 4 + 1 + len(SYNTHETIC_EXPECTED)
    rows = {row["name"]: row for row in census["candidates"]["rows"]}
    assert rows["dna01"]["members"] == 2
    assert rows["dna01"]["body_ids"] == [10_442, 10_760]
    assert rows["dna01"]["member_soma_sides"] == ["L", "R"]
    assert rows["neck"]["anatomically_eligible"] is True


#: The synthetic stand-ins for the two candidates the activity pass names, so the
#: pool builder is exercised over the real names it will look for.
ACTIVITY_SELECTORS = (
    CandidateSelector(name="dna01_steering_pair", type_name="DNa01"),
    CandidateSelector(name="cb_neck_motor_pool", superclass="cb_motor", subclass="nm"),
)


def test_activity_pools_partition_a_pool_by_soma_side() -> None:
    annotations = _annotations()
    census = _census(selectors=ACTIVITY_SELECTORS)
    pools = activity_pools(census, annotations)
    assert pools["dna01_steering_pair"].members == (2, 3)
    assert pools["dna01_steering_pair"].left == (2,)
    assert pools["dna01_steering_pair"].right == (3,)
    assert pools["cb_neck_motor_pool"].members == (6, 7)
    #: The reference pools come from the runtime side of the census, resolved
    #: back to dense indices through the body-ID map rather than re-selected.
    assert pools["ttmn_jump_pool"].members == (10, 11)
    assert pools["wing_motor_readout_52"].members == (13,)


def test_activity_pools_refuse_a_pool_the_census_never_produced() -> None:
    annotations = _annotations()
    census = _census(selectors=ACTIVITY_SELECTORS)
    census["candidates"]["rows"] = []
    with pytest.raises(ReadoutCandidateError, match="the census did not produce"):
        activity_pools(census, annotations)


def _document() -> dict[str, Any]:
    census = _census()
    return {"census": census, "census_sha256": SYNTHETIC_SHA, "activity": None}


def test_the_document_writer_is_byte_idempotent(tmp_path: Path) -> None:
    document = _document()
    output = tmp_path / "census.json"
    write_documents(output, document)
    first_json = output.read_bytes()
    first_markdown = output.with_suffix(".md").read_bytes()
    write_documents(output, document)
    assert output.read_bytes() == first_json
    assert output.with_suffix(".md").read_bytes() == first_markdown


def test_a_second_census_over_the_same_frame_is_byte_identical() -> None:
    assert document_bytes(_document()) == document_bytes(_document())
    assert render_markdown(_document()) == render_markdown(_document())


def test_the_json_document_round_trips_and_names_no_host_detail() -> None:
    payload = document_bytes(_document())
    restored = json.loads(payload)
    assert restored["census"]["schema"] == "neurocraft.analysis.readout_candidates.v1"
    assert "execution" not in restored
    assert "elapsed_seconds" not in payload.decode("utf-8")


def test_the_markdown_states_what_the_census_is_not() -> None:
    rendered = render_markdown(_document())
    assert "**No activity, no signal, no behavior.**" in rendered
    assert "annotation census" in rendered
    assert "## First activity measurement" in rendered
    assert "Not run for this document" in rendered


def test_the_markdown_reports_the_wing_pool_side_split_from_the_measurement() -> None:
    rendered = render_markdown(_document())
    assert "not side-balanced" in rendered
    assert f"contains [{RIGHT_MOTOR_BODY}] and not [{LEFT_MOTOR_BODY}]" in rendered


def test_the_markdown_reports_the_prefix_rule_delta_from_the_measurement() -> None:
    """The naive-selector caution must carry measured counts, not typed ones."""

    rendered = render_markdown(_document())
    assert "4 bodies against 2, so the naive rule adds 2 bodies" in rendered
    assert "M (body 12885)" in rendered


def test_the_prefix_caution_refuses_a_census_missing_an_aggregate_row() -> None:
    document = _document()
    document["census"]["candidates"]["rows"] = [
        row
        for row in document["census"]["candidates"]["rows"]
        if row["name"] != "dnp_startswith_aggregate"
    ]
    with pytest.raises(ReadoutCandidateError, match="no candidate"):
        render_markdown(document)
