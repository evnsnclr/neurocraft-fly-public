"""The readout census must be exact on data whose answers are hand-checkable.

Every structural test builds a small synthetic annotation table and a small
synthetic graph, so the suite never needs the ignored full artifact; the two
tests that touch the real artifact are skipped when it is absent, the normal
state on CI, and assert the published figures rather than recomputing them
loosely. Two properties are checked by an independent route so the production
code cannot mark its own homework: latency-bounded path support against a
pure-Python BFS, and the weight table against a per-edge Python accumulation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from neurocraft_connectome import load_full_runtime_artifact
from neurocraft_connectome.model import canonical_json_bytes

from scripts.analyze_readout import (
    DEFAULT_ANNOTATIONS,
    DEFAULT_ARTIFACT_DIR,
    NEURAL_STEPS_PER_GAME_TICK,
    REACHABILITY_01,
    SCHEMA,
    analyse,
    artifact_identity,
    boundary_sites,
    declared_annotations_sha256,
    reproduction_check,
)
from scripts.reachability_graph import graph_from_edges, graph_from_full_runtime, mapped_numpy
from scripts.readout_annotations import (
    NULL_KEY,
    IntegrityGate,
    ReadoutIntegrityError,
    RetainedAnnotations,
    annotation_gates,
    equals,
    file_sha256,
    full_row,
    gates_report,
    histogram,
    is_in,
    json_safe,
    key_of,
    require_all_gates,
    take_retained_annotations,
)
from scripts.readout_census import (
    boundary_identity_report,
    descending_census,
    fixture_named_descending,
    readout_population_report,
    vnc_motor_census,
)
from scripts.readout_connectivity import (
    descending_to_motor_report,
    direct_contacts,
    latency_comparison,
    readout_sets,
    weight_sum_gate,
)
from tests.analysis.test_analyze_reachability import reviewed_full_artifact_manifest_sha256

ROOT = Path(__file__).resolve().parents[2]

# A thirteen-node synthetic census, written row-major so the whole table is
# readable at once. Dense index is row order; the last three rows carry the real
# body IDs the CI fixture instantiates, so the whole-report path exercises the
# fixture-membership assertion.
COLUMN_NAMES = (
    "bodyId", "superclass", "subclass", "type", "somaSide", "somaNeuromere",
    "exitNerve", "entryNerve", "class", "instance", "rootSide",
)  # fmt: skip
_S = "cb_sensory"
_V = "vnc_sensory"
_M = "vnc_motor"
_D = "descending_neuron"
_MECH = "mechanosensory"
ROWS: tuple[tuple[Any, ...], ...] = (
    # body sc  subclass  type  side neuromere exit entry class instance root
    (10, _S, "wind_gravity", "JO-EV3", None, None, None, "AN", _MECH, "JO-EV3_R", "R"),
    (
        11,
        _V,
        "campaniform sensilla",
        "SNpp10",
        None,
        None,
        None,
        "ADMN",
        f"{_MECH}_proprioceptive",
        None,
        "L",
    ),
    (12, "cb_intrinsic", None, "WED203", "R", "MD", None, None, None, "WED203_R", None),
    (20, _M, "wm", "hg1 MN", "L", "T2", "ADMN", None, None, "hg1 MN_L", None),
    (21, _M, "wm", "hg1 MN", "R", "T2", "ADMN", None, None, "hg1 MN_R", None),
    (22, _M, "fl", "Ti flexor MN", "L", "T1", "ProLN", None, None, "Ti flexor MN_L", None),
    (23, _M, "nm", "MNnm08", "R", "T1", "ADMN", None, None, "MNnm08_R", None),
    (30, _D, "ut", "DNa02", "L", "MD", None, None, None, "DNa02_L", None),
    (31, _D, "ut", "DNa02", "R", "MD", None, None, None, "DNa02_R", None),
    (32, _D, None, None, "M", None, None, None, None, None, None),
    (10026, _D, "xn", "DNp18", "R", None, None, None, None, "DNp18_R", None),
    (27143, _D, "ut", "DNg07", "L", "LB", None, None, None, "DNg07_L", None),
    (28949, _D, "ut", "DNg07", "R", "LB", None, None, None, "DNg07_R", None),
)
SYNTHETIC: dict[str, list[Any]] = {
    name: [row[position] for row in ROWS] for position, name in enumerate(COLUMN_NAMES)
}
SYNTHETIC_SUPERCLASS_COUNTS = {
    "cb_intrinsic": 1,
    "cb_sensory": 1,
    "descending_neuron": 6,
    "vnc_motor": 4,
    "vnc_sensory": 1,
}
SYNTHETIC_SOURCE_ROWS = 100

# Synthetic edge list. Sensory 0 and 1 feed interneuron 2 and the descending
# neurons; the descending neurons feed the motor neurons directly.
EDGE_PRE = np.array([0, 1, 2, 2, 7, 7, 7, 8, 8, 9, 5], dtype=np.uint32)
EDGE_POST = np.array([2, 2, 7, 8, 3, 5, 6, 4, 5, 3, 3], dtype=np.uint32)
EDGE_WEIGHT = np.array([5, 7, 3, 4, 11, 13, 2, 17, 19, 23, 29], dtype=np.int64)
NODE_COUNT = 13

SYNTHETIC_SITES: dict[str, list[dict[str, Any]]] = {
    "motor": [
        {"body_id": 20, "dense_index": 3, "annotation_source_row": 3, "superclass": "vnc_motor"},
        {"body_id": 21, "dense_index": 4, "annotation_source_row": 4, "superclass": "vnc_motor"},
    ],
    "sensory": [
        {"body_id": 10, "dense_index": 0, "annotation_source_row": 0, "superclass": "cb_sensory"},
        {"body_id": 11, "dense_index": 1, "annotation_source_row": 1, "superclass": "vnc_sensory"},
    ],
}


def _table() -> Any:
    return pa.table({name: pa.array(values) for name, values in SYNTHETIC.items()})


def _annotations(**overrides: Any) -> RetainedAnnotations:
    columns = {
        name: np.asarray(values, dtype=object)
        for name, values in SYNTHETIC.items()
        if name != "bodyId"
    }
    columns["bodyId"] = np.asarray(SYNTHETIC["bodyId"], dtype=object)
    defaults: dict[str, Any] = {
        "body_ids": np.asarray(SYNTHETIC["bodyId"], dtype=np.uint64),
        "source_rows": np.arange(NODE_COUNT, dtype=np.uint64),
        "columns": columns,
        "source_row_count": SYNTHETIC_SOURCE_ROWS,
        "annotations_sha256": "a" * 64,
    }
    defaults.update(overrides)
    return RetainedAnnotations(**defaults)


def _gates(annotations: RetainedAnnotations) -> list[IntegrityGate]:
    return [
        *annotation_gates(
            annotations,
            declared_annotations_sha256="a" * 64,
            declared_source_rows=SYNTHETIC_SOURCE_ROWS,
            retained_by_superclass=SYNTHETIC_SUPERCLASS_COUNTS,
        ),
        weight_sum_gate(EDGE_WEIGHT, int(EDGE_WEIGHT.sum())),
    ]


@pytest.fixture
def annotations() -> RetainedAnnotations:
    return _annotations()


@pytest.fixture
def taken() -> Any:
    return _table()


@pytest.fixture
def graph() -> Any:
    return graph_from_edges(NODE_COUNT, EDGE_PRE, EDGE_POST)


# ------------------------------------------------------------------ utilities


def test_key_and_histogram_map_nulls_to_one_reserved_key() -> None:
    assert (key_of(None), key_of("wm")) == (NULL_KEY, "wm")
    assert histogram(["b", None, "a", "b"]) == {NULL_KEY: 1, "a": 1, "b": 2}
    assert list(histogram(["b", "a"])) == ["a", "b"]


def test_equals_and_is_in_are_null_safe_boolean_masks(annotations: RetainedAnnotations) -> None:
    mask = equals(annotations.column("superclass"), "vnc_motor")
    assert mask.dtype == np.bool_
    assert np.flatnonzero(mask).tolist() == [3, 4, 5, 6]
    assert mask.size == NODE_COUNT
    membership = is_in(annotations.column("subclass"), ("wm", "nm"))
    assert np.flatnonzero(membership).tolist() == [3, 4, 6]


def test_json_safe_replaces_non_finite_doubles_with_null() -> None:
    assert json_safe(float("nan")) is None and json_safe(float("inf")) is None
    assert json_safe(1.5) == 1.5 and json_safe(None) is None
    assert json_safe([1, None, "x"]) == [1, None, "x"]


def test_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    payload = b"neurocraft" * 1024
    path.write_bytes(payload)
    assert file_sha256(path) == hashlib.sha256(payload).hexdigest()


def test_all_integrity_gates_pass_on_a_consistent_table(
    annotations: RetainedAnnotations,
) -> None:
    gates = _gates(annotations)
    assert [gate.name for gate in gates] == [
        "pinned_annotations_file_sha256",
        "taken_body_id_column_bitwise_identical",
        "superclass_counts_reconcile_with_manifest",
        "annotation_source_rows_unique_and_in_range",
        "edge_raw_counts_sum_matches_manifest",
    ]
    report = gates_report(gates)
    assert all(gate.passed for gate in gates) and report["all_passed"] is True
    assert len(report["gates"]) == 5
    require_all_gates(gates)


def test_source_digest_gate_fails_when_the_pinned_table_changed(
    annotations: RetainedAnnotations,
) -> None:
    gates = annotation_gates(
        annotations,
        declared_annotations_sha256="b" * 64,
        declared_source_rows=SYNTHETIC_SOURCE_ROWS,
        retained_by_superclass=SYNTHETIC_SUPERCLASS_COUNTS,
    )
    assert [gate.name for gate in gates if not gate.passed] == ["pinned_annotations_file_sha256"]


def test_body_id_gate_fails_on_a_single_misaligned_pointer() -> None:
    shifted = _annotations(body_ids=np.asarray([9, *SYNTHETIC["bodyId"][1:]], dtype=np.uint64))
    gates = _gates(shifted)
    failures = [gate.name for gate in gates if not gate.passed]
    assert failures == ["taken_body_id_column_bitwise_identical"]
    with pytest.raises(ReadoutIntegrityError, match="taken_body_id_column_bitwise_identical"):
        require_all_gates(gates)


def test_superclass_gate_reports_every_mismatching_key(
    annotations: RetainedAnnotations,
) -> None:
    wrong = dict(SYNTHETIC_SUPERCLASS_COUNTS)
    wrong["vnc_motor"] = 5
    wrong["ENS"] = 3
    gates = annotation_gates(
        annotations,
        declared_annotations_sha256="a" * 64,
        declared_source_rows=SYNTHETIC_SOURCE_ROWS,
        retained_by_superclass=wrong,
    )
    gate = next(gate for gate in gates if gate.name == "superclass_counts_reconcile_with_manifest")
    assert gate.passed is False
    assert gate.observed["mismatches"] == {
        "ENS": {"observed": 0, "manifest": 3},
        "vnc_motor": {"observed": 4, "manifest": 5},
    }


def test_source_row_gate_rejects_duplicates_and_out_of_range_pointers() -> None:
    duplicated = _annotations(source_rows=np.asarray([0, *range(NODE_COUNT - 1)], dtype=np.uint64))
    gate = next(
        gate
        for gate in _gates(duplicated)
        if gate.name == "annotation_source_rows_unique_and_in_range"
    )
    assert gate.passed is False and gate.observed["unique"] is False

    beyond = _annotations(source_rows=np.arange(NODE_COUNT, dtype=np.uint64) + 1_000)
    gate = next(
        gate for gate in _gates(beyond) if gate.name == "annotation_source_rows_unique_and_in_range"
    )
    assert gate.passed is False and gate.observed["maximum_row"] == 999 + NODE_COUNT


def test_weight_sum_gate_binds_the_edge_support_to_the_manifest() -> None:
    assert weight_sum_gate(EDGE_WEIGHT, int(EDGE_WEIGHT.sum())).passed is True
    failing = weight_sum_gate(EDGE_WEIGHT, int(EDGE_WEIGHT.sum()) + 1)
    assert failing.passed is False
    assert failing.observed == int(EDGE_WEIGHT.sum())


def test_readout_population_records_that_no_superclass_is_literally_descending(
    annotations: RetainedAnnotations,
) -> None:
    report = readout_population_report(annotations)
    counts = {row["superclass"]: row["retained_nodes"] for row in report["rows"]}
    assert counts["descending_neuron"] == 6
    assert counts["vnc_motor"] == 4
    assert counts["cb_motor"] == 0
    naive = report["naive_descending_filter"]
    assert naive["nodes_matching_superclass_exactly_descending"] == 0
    assert naive["substring_matching_superclasses"] == ["descending_neuron"]
    assert report["total_readout_eligible_nodes"] == 10


def test_vnc_motor_census_counts_subclasses_and_inventories_wing_types(
    annotations: RetainedAnnotations,
) -> None:
    report = vnc_motor_census(annotations)
    assert report["retained_vnc_motor_nodes"] == 4
    assert report["subclass_counts"] == {"fl": 1, "nm": 1, "wm": 2}
    wing = report["wing_motor_subclass"]
    assert (wing["cells"], wing["distinct_types"], wing["untyped_cells"]) == (2, 1, 0)
    assert wing["types"] == [{"type": "hg1 MN", "cells": 2, "body_ids": [20, 21]}]
    assert wing["exit_nerve_counts"] == {"ADMN": 2}
    assert report["leg_motor_subclasses"]["cells"] == 1


def test_boundary_identity_reports_full_rows_and_names_the_bilateral_pair(
    annotations: RetainedAnnotations, taken: Any
) -> None:
    report = boundary_identity_report(annotations, taken, SYNTHETIC_SITES)
    motor = report["roles"]["motor"]
    assert [entry["body_id"] for entry in motor] == [20, 21]
    assert motor[0]["annotation_row"]["type"] == "hg1 MN"
    assert motor[0]["annotation_row"]["exitNerve"] == "ADMN"
    assert sorted(motor[0]["annotation_row"]) == sorted(SYNTHETIC)
    summary = report["current_motor_readout_summary"]
    assert summary["distinct_types"] == ["hg1 MN"]
    assert summary["is_a_single_bilateral_type"] is True
    assert summary["soma_sides"] == ["L", "R"]
    assert summary["subclasses"] == ["wm"]
    sensory = report["roles"]["sensory"]
    assert [entry["annotation_row"]["type"] for entry in sensory] == ["JO-EV3", "SNpp10"]


def test_boundary_identity_refuses_an_overlay_that_disagrees_with_body_ids(
    annotations: RetainedAnnotations, taken: Any
) -> None:
    corrupted = {
        "motor": [
            {"body_id": 999, "dense_index": 3, "superclass": "vnc_motor"},
        ]
    }
    with pytest.raises(ValueError, match=r"does not match body_ids\.bin"):
        boundary_identity_report(annotations, taken, corrupted)


def test_descending_census_uses_disjoint_maximal_alpha_prefixes() -> None:
    census = descending_census(
        RetainedAnnotations(
            body_ids=np.arange(6, dtype=np.uint64),
            source_rows=np.arange(6, dtype=np.uint64),
            columns={
                "superclass": np.asarray(["descending_neuron"] * 6, dtype=object),
                "type": np.asarray(
                    ["DNge001", "DNge002", "DNg02_a", "DNp01", "MDN", None], dtype=object
                ),
                "subclass": np.asarray(["ut", "ut", "xn", "xn", None, None], dtype=object),
                "somaSide": np.asarray(["L", "R", "L", "R", "M", None], dtype=object),
                "bodyId": np.asarray(list(range(6)), dtype=object),
            },
            source_row_count=6,
            annotations_sha256="a" * 64,
        )
    )
    assert (census["cells"], census["typed_cells"], census["untyped_cells"]) == (6, 5, 1)
    assert census["distinct_types"] == 5
    assert census["type_prefix_counts"] == {"DNg": 1, "DNge": 2, "DNp": 1, "MDN": 1}
    assert census["type_prefix_sums_to_typed_cells"] is True
    assert census["soma_side_counts"] == {NULL_KEY: 1, "L": 2, "M": 1, "R": 2}
    assert census["subclass_counts"] == {NULL_KEY: 2, "ut": 2, "xn": 2}


def test_descending_census_separates_bilateral_pairs_from_larger_types(
    annotations: RetainedAnnotations,
) -> None:
    census = descending_census(annotations)
    bilaterality = census["bilaterality"]
    assert bilaterality["types_with_exactly_two_cells"] == 2
    assert bilaterality["types_with_exactly_two_cells_one_per_side"] == 2
    assert (
        bilaterality["types_with_one_cell"],
        bilaterality["types_with_more_than_two_cells"],
    ) == (1, 0)
    assert census["type_size_histogram"] == {"1": 1, "2": 2}


def test_fixture_named_types_report_coverage_and_refuse_a_missing_member(
    annotations: RetainedAnnotations, taken: Any
) -> None:
    report = fixture_named_descending(annotations, taken)
    dng07 = report["types"]["DNg07"]
    assert (dng07["cells_in_full_artifact"], dng07["cells_used_by_ci_fixture"]) == (2, 2)
    assert dng07["fixture_coverage_fraction"] == 1.0
    assert dng07["soma_side_counts"] == {"L": 1, "R": 1}
    assert [cell["body_id"] for cell in dng07["cells"]] == [27143, 28949]
    assert dng07["fixture_member_body_ids"] == [27143, 28949]
    dnp18 = report["types"]["DNp18"]
    assert (dnp18["cells_in_full_artifact"], dnp18["cells_used_by_ci_fixture"]) == (1, 1)

    absent = _annotations()
    absent.columns["type"][11] = "DNg07_renamed"
    with pytest.raises(ValueError, match="were not all found"):
        fixture_named_descending(absent, taken)


# ------------------------------------------------------- latency comparison


def _brute_force_distances(
    node_count: int, pre: Any, post: Any, seeds: list[int], *, reverse: bool
) -> dict[int, int]:
    """Independent pure-Python BFS so the CSR traversal cannot mark its own homework."""

    source, target = (post, pre) if reverse else (pre, post)
    distances = dict.fromkeys(seeds, 0)
    frontier = list(seeds)
    hop = 0
    while frontier:
        hop += 1
        nxt = []
        for node in frontier:
            for index in range(len(pre)):
                if int(source[index]) == node and int(target[index]) not in distances:
                    distances[int(target[index])] = hop
                    nxt.append(int(target[index]))
        frontier = nxt
    assert node_count >= len(distances)
    return distances


def test_readout_sets_are_defined_by_their_stated_membership_rule(
    annotations: RetainedAnnotations,
) -> None:
    sets = {entry.name: entry for entry in readout_sets(annotations, [3, 4])}
    assert sets["shipped_two_body_readout"].dense_indices.tolist() == [3, 4]
    assert sets["vnc_motor_wing_subclass"].dense_indices.tolist() == [3, 4]
    assert sets["vnc_motor_wing_neck_haltere_subclasses"].dense_indices.tolist() == [3, 4, 6]
    assert sets["vnc_motor_all"].dense_indices.tolist() == [3, 4, 5, 6]
    assert sets["descending_neuron_all"].dense_indices.tolist() == [7, 8, 9, 10, 11, 12]


def test_latency_comparison_matches_an_independent_python_traversal(
    annotations: RetainedAnnotations, graph: Any
) -> None:
    injection = np.asarray([0, 1], dtype=np.int64)
    report = latency_comparison(
        graph, injection, readout_sets(annotations, [3, 4]), steps_per_game_tick=3
    )
    forward = _brute_force_distances(NODE_COUNT, EDGE_PRE, EDGE_POST, [0, 1], reverse=False)
    for row in report["rows"]:
        sites = next(
            entry.dense_indices.tolist()
            for entry in readout_sets(annotations, [3, 4])
            if entry.name == row["readout_set"]
        )
        backward = _brute_force_distances(NODE_COUNT, EDGE_PRE, EDGE_POST, sites, reverse=True)
        both = set(forward) & set(backward)
        assert row["unbounded_path_support_nodes"] == len(both)
        expected = sum(1 for node in both if forward[node] + backward[node] <= 3)
        assert row["one_game_tick_path_support_nodes"] == expected
        assert row["one_game_tick_path_support_fraction"] == expected / NODE_COUNT
        fresh_expected = sum(1 for node in both if forward[node] + backward[node] <= 1)
        assert row["fresh_input_one_game_tick_max_graph_hops"] == 1
        assert row["fresh_input_one_game_tick_path_support_nodes"] == fresh_expected
        assert row["fresh_input_one_game_tick_path_support_fraction"] == fresh_expected / NODE_COUNT
        assert "loose historical upper bound" in row["legacy_one_game_tick_scope"]
        if both:
            assert row["shortest_injection_to_readout_path"] == min(
                forward[node] + backward[node] for node in both
            )


def test_latency_table_is_monotone_and_records_the_hop_histogram(
    annotations: RetainedAnnotations, graph: Any
) -> None:
    report = latency_comparison(
        graph,
        np.asarray([0, 1], dtype=np.int64),
        readout_sets(annotations, [3, 4]),
        steps_per_game_tick=3,
    )
    for row in report["rows"]:
        counts = [entry["path_support_nodes"] for entry in row["table"]]
        assert counts == sorted(counts)
        assert max(counts) <= row["unbounded_path_support_nodes"]
        assert (
            sum(row["forward_hops_from_injection_to_readout_population"].values())
            == (row["readout_sites"])
        )
    shipped = report["rows"][0]
    assert shipped["forward_hops_from_injection_to_readout_population"] == {"3": 2}
    assert "UPPER BOUND" in report["evidence_boundary"]
    assert shipped["table"][2]["max_graph_hops"] == 3
    assert shipped["table"][2]["fresh_input_minimum_neural_updates_for_hop_budget"] == 7


def test_direct_contacts_and_weight_table_match_a_per_edge_accumulation(
    annotations: RetainedAnnotations,
) -> None:
    contacts = direct_contacts(annotations, EDGE_PRE, EDGE_POST, EDGE_WEIGHT)
    superclass = SYNTHETIC["superclass"]
    selected = [
        index
        for index in range(len(EDGE_PRE))
        if superclass[int(EDGE_PRE[index])] == "descending_neuron"
        and superclass[int(EDGE_POST[index])] == "vnc_motor"
    ]
    into_motor = [
        index for index in range(len(EDGE_PRE)) if superclass[int(EDGE_POST[index])] == "vnc_motor"
    ]
    assert contacts.weights.size == len(selected)
    assert int(contacts.weights.sum()) == sum(int(EDGE_WEIGHT[i]) for i in selected)
    assert contacts.total_incoming_motor_weight == sum(int(EDGE_WEIGHT[i]) for i in into_motor)
    assert (contacts.descending_cells, contacts.motor_cells) == (6, 4)

    report = descending_to_motor_report(annotations, contacts, named_types=("DNa02",))
    assert (report["edges"], report["weight"]) == (6, 11 + 13 + 2 + 17 + 19 + 23)
    assert report["descending_cells_with_a_direct_motor_target"] == 3
    assert report["descending_cells_with_a_direct_motor_target_fraction"] == 0.5
    assert report["vnc_motor_cells_receiving_a_direct_descending_input"] == 4
    assert (
        report["share_of_all_incoming_vnc_motor_weight"]
        == report["weight"] / (report["total_incoming_vnc_motor_weight"])
    )


def test_named_type_row_splits_wing_from_leg_and_indexes_laterality(
    annotations: RetainedAnnotations,
) -> None:
    contacts = direct_contacts(annotations, EDGE_PRE, EDGE_POST, EDGE_WEIGHT)
    report = descending_to_motor_report(annotations, contacts, named_types=("DNa02",))
    row = report["named_types"][0]
    assert (row["type"], row["cells"], row["edges"]) == ("DNa02", 2, 5)
    assert row["motor_weight"] == 11 + 13 + 2 + 17 + 19
    # DNa02_L (index 7) -> wm 3 (11); DNa02_R (index 8) -> wm 4 (17).
    assert row["weight_by_motor_subclass"] == {"fl": 32, "nm": 2, "wm": 28}
    assert (row["wing_weight"], row["leg_weight"], row["other_subclass_weight"]) == (28, 32, 2)
    assert (row["wing_weight_fraction"], row["leg_weight_fraction"]) == (28 / 62, 32 / 62)
    # Ipsilateral: 7->3 (11, L/L), 7->5 (13, L/L), 8->4 (17, R/R).
    # Contralateral: 7->6 (2, L/R), 8->5 (19, R/L).
    laterality = row["laterality"]
    assert (laterality["ipsilateral_weight"], laterality["contralateral_weight"]) == (41, 21)
    assert laterality["unassigned_side_weight"] == 0
    assert laterality["laterality_index"] == (41 - 21) / 62
    assert (row["dominant_target_type"], row["dominant_target_type_weight"]) == (
        "Ti flexor MN",
        32,
    )
    assert row["distinct_target_types"] == 3


def test_laterality_excludes_endpoints_without_an_l_or_r_soma_side(
    annotations: RetainedAnnotations,
) -> None:
    contacts = direct_contacts(annotations, EDGE_PRE, EDGE_POST, EDGE_WEIGHT)
    report = descending_to_motor_report(annotations, contacts, named_types=("DNa02",))
    ranking = {row["type"]: row for row in report["top_types_by_motor_weight"]}
    assert ranking["DNa02"]["motor_weight"] == 62
    # Node 9 is somaSide "M", so its single 23-unit contact carries no index.
    untyped_weight = report["weight"] - ranking["DNa02"]["motor_weight"]
    assert untyped_weight == 23
    assert report["distinct_descending_types_with_motor_weight"] == 1
    assert len(report["top_types_by_motor_weight"]) == 3


def _analysis(annotations: RetainedAnnotations, taken: Any, graph: Any) -> dict[str, Any]:
    return analyse(
        annotations,
        taken,
        graph,
        (EDGE_PRE, EDGE_POST, EDGE_WEIGHT),
        identity={"artifact_id": "synthetic", "manifest_file_sha256": "0" * 64},
        sites=SYNTHETIC_SITES,
        gates=_gates(annotations),
        steps_per_game_tick=3,
    )


def test_analysis_body_is_deterministic_and_carries_every_section(
    annotations: RetainedAnnotations, taken: Any, graph: Any
) -> None:
    first = _analysis(annotations, taken, graph)
    second = _analysis(_annotations(), _table(), graph_from_edges(NODE_COUNT, EDGE_PRE, EDGE_POST))
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first["schema"] == SCHEMA
    assert set(first) == {
        "schema", "artifact", "integrity_gates", "boundary_role_overlay",
        "readout_eligible_population", "vnc_motor_census", "current_readout_identity",
        "descending_neuron_census", "fixture_named_descending_neurons",
        "latency_bounded_readout_comparison", "reachability_01_reproduction",
        "descending_to_motor_connectivity", "interpretation_bounds",
    }  # fmt: skip
    assert first["integrity_gates"]["all_passed"] is True


def test_analysis_refuses_to_produce_a_body_when_a_gate_fails(taken: Any, graph: Any) -> None:
    broken = _annotations(body_ids=np.asarray([9, *SYNTHETIC["bodyId"][1:]], dtype=np.uint64))
    with pytest.raises(ReadoutIntegrityError):
        analyse(
            broken,
            taken,
            graph,
            (EDGE_PRE, EDGE_POST, EDGE_WEIGHT),
            identity={},
            sites=SYNTHETIC_SITES,
            gates=_gates(broken),
            steps_per_game_tick=3,
        )


def test_reproduction_check_flags_a_deviation_from_the_published_figures() -> None:
    published = {
        name: REACHABILITY_01[name]
        for name in (
            "one_game_tick_path_support_nodes",
            "unbounded_path_support_nodes",
            "shortest_injection_to_readout_path",
        )
    }
    row = {"readout_set": "shipped_two_body_readout", **published}
    nodes = REACHABILITY_01["node_count"]
    assert reproduction_check({"rows": [row]}, nodes)["reproduced"] is True
    checked = reproduction_check(
        {"rows": [{**row, "one_game_tick_path_support_nodes": 45_611}]}, nodes
    )
    assert checked["reproduced"] is False
    assert [entry["quantity"] for entry in checked["checks"] if not entry["matches"]] == [
        "one_game_tick_path_support_nodes"
    ]


def test_no_reserved_causal_vocabulary_survives_in_the_report_body(
    annotations: RetainedAnnotations, taken: Any, graph: Any
) -> None:
    """`causal` is reserved for the G3 gate; this analysis is census and topology."""

    analysis = _analysis(annotations, taken, graph)
    body = canonical_json_bytes(analysis).decode()
    assert "causal" not in body.replace(
        "causal evidence in the sense docs/SCIENTIFIC_CLAIMS.md reserves for G3", ""
    )
    bounds = analysis["interpretation_bounds"]
    assert (
        "bounds participation and never demonstrates it"
        in (bounds["reachability_is_an_upper_bound"])
    )
    assert "invalidates no existing receipt" in bounds["no_gate_movement"]


# ------------------------------------------------------- full-artifact tests


_ARTIFACT_PRESENT = (ROOT / DEFAULT_ARTIFACT_DIR / "manifest.json").is_file() and (
    ROOT / DEFAULT_ANNOTATIONS
).is_file()


def _full_artifact() -> dict[str, Any]:
    """Load the pinned artifact once and run the production gate sequence."""

    with load_full_runtime_artifact(
        ROOT / DEFAULT_ARTIFACT_DIR,
        expected_manifest_sha256=reviewed_full_artifact_manifest_sha256(
            ROOT / DEFAULT_ARTIFACT_DIR
        ),
    ) as artifact:
        manifest = artifact.manifest
        loaded = {
            "identity": artifact_identity(manifest),
            "sites": boundary_sites(manifest),
            "declared": declared_annotations_sha256(manifest),
            "retained_by_superclass": dict(manifest.retained_by_superclass),
            "declared_rows": manifest.counts["annotation_source_rows"],
            "retained_weight_sum": manifest.counts["retained_weight_sum"],
            "source_rows": np.array(mapped_numpy(artifact.annotation_source_rows, "<u8")),
            "body_ids": np.array(mapped_numpy(artifact.body_ids, "<u8")),
            "pre": np.array(mapped_numpy(artifact.edge_pre_indices, "<u4")),
            "post": np.array(mapped_numpy(artifact.edge_post_indices, "<u4")),
            "weights": np.array(mapped_numpy(artifact.edge_raw_counts, "<u8")).astype(np.int64),
            "graph": graph_from_full_runtime(artifact),
        }
    annotations, taken = take_retained_annotations(
        ROOT / DEFAULT_ANNOTATIONS,
        source_rows=loaded["source_rows"],
        body_ids=loaded["body_ids"],
    )
    loaded["annotations"] = annotations
    loaded["taken"] = taken
    loaded["gates"] = [
        *annotation_gates(
            annotations,
            declared_annotations_sha256=loaded["declared"],
            declared_source_rows=loaded["declared_rows"],
            retained_by_superclass=loaded["retained_by_superclass"],
        ),
        weight_sum_gate(loaded["weights"], loaded["retained_weight_sum"]),
    ]
    return loaded


@pytest.mark.skipif(
    not _ARTIFACT_PRESENT,
    reason="the full MaleCNS runtime artifact and pinned annotations are ignored and absent",
)
def test_full_artifact_gates_pass_and_reproduce_the_published_figures() -> None:
    loaded = _full_artifact()
    gates = loaded["gates"]
    assert all(gate.passed for gate in gates), [g.name for g in gates if not g.passed]
    analysis = analyse(
        loaded["annotations"],
        loaded["taken"],
        loaded["graph"],
        (loaded["pre"], loaded["post"], loaded["weights"]),
        identity=loaded["identity"],
        sites=loaded["sites"],
        gates=gates,
    )
    assert analysis["reachability_01_reproduction"]["reproduced"] is True
    rows = {
        row["readout_set"]: row for row in analysis["latency_bounded_readout_comparison"]["rows"]
    }
    assert rows["shipped_two_body_readout"]["one_game_tick_path_support_nodes"] == 45_610
    assert rows["vnc_motor_wing_subclass"]["one_game_tick_path_support_nodes"] == 58_699
    assert (
        rows["vnc_motor_wing_neck_haltere_subclasses"]["one_game_tick_path_support_nodes"] == 60_118
    )
    assert rows["vnc_motor_all"]["one_game_tick_path_support_nodes"] == 64_195
    assert rows["descending_neuron_all"]["one_game_tick_path_support_nodes"] == 152_983
    assert analysis["latency_bounded_readout_comparison"]["neural_steps_per_game_tick"] == (
        NEURAL_STEPS_PER_GAME_TICK
    )


@pytest.mark.skipif(
    not _ARTIFACT_PRESENT,
    reason="the full MaleCNS runtime artifact and pinned annotations are ignored and absent",
)
def test_full_artifact_census_matches_the_recorded_population_counts() -> None:
    loaded = _full_artifact()
    annotations = loaded["annotations"]
    taken = loaded["taken"]
    population = readout_population_report(annotations)
    counts = {row["superclass"]: row["retained_nodes"] for row in population["rows"]}
    assert counts == {
        "descending_neuron": 1_314, "descending_neuron_tbc": 2, "efferent_descending": 4,
        "sensory_descending": 12, "vnc_motor": 708, "cb_motor": 107, "vnc_efferent": 94,
    }  # fmt: skip
    assert (
        population["naive_descending_filter"]["nodes_matching_superclass_exactly_descending"] == 0
    )

    motor = vnc_motor_census(annotations)
    assert motor["subclass_counts"] == {
        "ad": 214, "fl": 135, "hl": 130, "hm": 16,
        "ml": 116, "nm": 24, "wm": 67, "xm": 6,
    }  # fmt: skip
    assert motor["wing_motor_subclass"]["cells"] == 67

    census = descending_census(annotations)
    assert (census["distinct_types"], census["typed_cells"], census["untyped_cells"]) == (
        480,
        1_310,
        4,
    )
    prefixes = census["type_prefix_counts"]
    assert (prefixes["DNge"], prefixes["DNg"], prefixes["DNp"]) == (429, 427, 162)
    assert census["bilaterality"]["types_with_exactly_two_cells"] == 386

    summary = boundary_identity_report(annotations, taken, loaded["sites"])[
        "current_motor_readout_summary"
    ]
    assert summary["distinct_types"] == ["hg1 MN"]
    assert summary["is_a_single_bilateral_type"] is True

    fixture = fixture_named_descending(annotations, taken)["types"]
    assert (
        fixture["DNg07"]["cells_in_full_artifact"],
        fixture["DNg07"]["cells_used_by_ci_fixture"],
    ) == (16, 2)
    assert (
        fixture["DNp18"]["cells_in_full_artifact"],
        fixture["DNp18"]["cells_used_by_ci_fixture"],
    ) == (2, 1)


def test_full_row_reads_one_complete_annotation_record(taken: Any) -> None:
    record = full_row(taken, 3)
    assert (record["bodyId"], record["type"], record["somaSide"]) == (20, "hg1 MN", "L")
    assert list(record) == sorted(SYNTHETIC)
