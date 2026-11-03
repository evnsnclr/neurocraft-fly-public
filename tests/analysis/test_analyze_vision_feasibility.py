"""The Phase 2 vision feasibility assessment must be exact on hand-checkable data.

Every structural test builds a small synthetic annotation table with its own hex
lattice and a small synthetic edge list, so the suite never needs the ignored
full artifact. Three tests touch the tracked flight-steering fixture, which is
committed and therefore always present. The tests that need the ignored artifact
and the ignored annotations table are skipped when either is absent, the normal
state on CI, and assert the published figures rather than recomputing them
loosely.

Five properties are checked by an independent route so the production code
cannot mark its own homework: induced edge counts against a per-edge Python
accumulation, descending support against a pure-Python breadth-first traversal,
the closure interior pool against a brute-force enumeration of every path of at
most three edges, the cross-connectivity count against a per-edge Python scan,
and the tangential input composition against a per-edge Python accumulation.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from neurocraft_connectome import build_synthetic_graph, load_malecns_fixture
from neurocraft_connectome.flight_pin import (
    MALECNS_FLIGHT_FIXTURE_V1_EDGE_COUNT,
    MALECNS_FLIGHT_FIXTURE_V1_FILE_BYTES,
    MALECNS_FLIGHT_FIXTURE_V1_NODE_COUNT,
    MALECNS_FLIGHT_FIXTURE_V1_PATH,
    MALECNS_FLIGHT_FIXTURE_V1_PIN,
)
from neurocraft_connectome.model import (
    RuntimeGraph,
    SyntheticEdge,
    SyntheticNode,
    canonical_json_bytes,
)

from scripts.analyze_vision_feasibility import (
    DEFAULT_ANNOTATIONS,
    DEFAULT_ARTIFACT_DIR,
    SCHEMA,
    analyse,
    fixture_gates,
    main,
)
from scripts.readout_annotations import (
    IntegrityGate,
    ReadoutIntegrityError,
    RetainedAnnotations,
    annotation_gates,
)
from scripts.readout_connectivity import weight_sum_gate
from scripts.vision_circuits import (
    FLIGHT_FIXTURE_EDGE_CEILING,
    LPTC_TYPE_PREFIXES,
    EdgeSupport,
    closure_report,
    closure_tiers,
    cross_connectivity,
    dense_indices_for,
    descending_pool,
    descending_support,
    motion_pathway_addressing,
    tangential_census,
    tangential_circuits,
    tangential_input_composition,
    type_prefix_members,
    wing_motor_pool,
)
from scripts.vision_columns import (
    OPTIC_CENSUS_01,
    TypeFamily,
    address_census,
    build_column_index,
    column_census,
    hex_addresses,
    hex_distance,
    hex_neighbours,
    is_contiguous,
    reproduction_check,
    select_disc,
    type_family_report,
)
from scripts.vision_report import (
    cross_connectivity_report,
    fixture_visual_substrate,
    metric_ambiguity,
    patch_sizing,
)
from tests.analysis.test_analyze_reachability import reviewed_full_artifact_manifest_sha256

ROOT = Path(__file__).resolve().parents[2]

# A twenty-node synthetic census. Dense index is row order. The first nine rows
# lay out a small right-eye hex lattice with three columns complete over the
# synthetic patch types, one incomplete column, one left-eye column, and one row
# carrying only half a column address.
COLUMN_NAMES = (
    "bodyId", "superclass", "subclass", "type", "somaSide",
    "assignedOlHex1", "assignedOlHex2", "instance",
)  # fmt: skip
_OL = "ol_intrinsic"
_DN = "descending_neuron"
_VM = "vnc_motor"
_VP = "visual_projection"
_CB = "cb_intrinsic"
ROWS: tuple[tuple[Any, ...], ...] = (
    # body superclass subclass type    side hex1  hex2  instance
    (100, _OL, None, "L1", "R", 5.0, 5.0, "L1_R"),
    (101, _OL, None, "L2", "R", 5.0, 5.0, "L2_R"),
    (102, _OL, None, "L1", "R", 6.0, 5.0, "L1_R"),
    (103, _OL, None, "L2", "R", 6.0, 5.0, "L2_R"),
    (104, _OL, None, "L1", "R", 5.0, 6.0, "L1_R"),
    (105, _OL, None, "L2", "R", 5.0, 6.0, "L2_R"),
    (106, _OL, None, "L1", "R", 9.0, 9.0, "L1_R"),
    (107, _OL, None, "L1", "L", 5.0, 5.0, "L1_L"),
    (108, _OL, None, "L2", "L", 5.0, 5.0, "L2_L"),
    (109, _OL, None, "T4a", "R", None, None, "T4a_R"),
    (110, _DN, "ut", "DNa02", "R", None, None, "DNa02_R"),
    (111, _DN, "ut", "DNa02", "L", None, None, "DNa02_L"),
    (112, _VM, "wm", "hg1 MN", "L", None, None, "hg1 MN_L"),
    (113, _VM, "wm", "hg1 MN", "R", None, None, "hg1 MN_R"),
    (114, _VP, None, "VS", "R", None, None, "VS_R"),
    (115, _VP, None, "HSE", "L", None, None, "HSE_L"),
    (116, _CB, None, "CB0285", "R", None, None, "CB0285_R"),
    (117, _OL, None, "Mi1", "R", 5.0, 5.0, "Mi1_R"),
    (118, _OL, None, "L1", "R", 7.0, None, "L1_R"),
    (119, _CB, None, "CB0540", "R", None, None, "CB0540_R"),
)
SYNTHETIC: dict[str, list[Any]] = {
    name: [row[position] for row in ROWS] for position, name in enumerate(COLUMN_NAMES)
}
SYNTHETIC_SUPERCLASS_COUNTS = {
    _CB: 2,
    _DN: 2,
    _OL: 12,
    _VM: 2,
    _VP: 2,
}
SYNTHETIC_SOURCE_ROWS = 500
NODE_COUNT = len(ROWS)

# Synthetic patch and columnar type sets, so the lattice above is small enough
# to check by hand while exercising exactly the production code paths.
PATCH_TYPES: tuple[str, ...] = ("L1", "L2")
COLUMNAR_TYPES: tuple[str, ...] = ("L1", "L2", "Mi1")

# Synthetic edge list. Patch cells wire among themselves and into one
# intermediate; the intermediate reaches the right descending neuron, which
# reaches both motor neurons. One tangential cell contacts the descending
# neuron heavily and one motor neuron directly; one motion-pathway cell feeds
# both tangential cells. Node 119 is a dead end two hops from the patch, so the
# candidate pool and the path-interior pool differ. Node 106 is the only
# column-addressed cell with an edge into the synthetic fixture, and it lies in
# an incomplete column, so every patch selection is disconnected from it.
EDGE_PRE = np.array(
    [0, 2, 0, 4, 0, 4, 16, 10, 10, 14, 15, 9, 9, 16, 6, 11, 14, 16], dtype=np.uint32
)
EDGE_POST = np.array(
    [1, 3, 2, 5, 16, 16, 10, 12, 13, 10, 10, 14, 15, 14, 12, 12, 13, 19], dtype=np.uint32
)
EDGE_WEIGHT = np.array(
    [7, 12, 4, 25, 30, 15, 20, 40, 35, 25, 5, 30, 10, 6, 9, 2, 11, 15], dtype=np.int64
)

SYNTHETIC_IDENTITY: dict[str, Any] = {
    "artifact_id": "synthetic-vision-census",
    "node_count": NODE_COUNT,
    "edge_count": int(EDGE_PRE.size),
    "manifest_file_sha256": "b" * 64,
}

#: Body IDs the synthetic stand-in fixture instantiates.
FIXTURE_BODY_IDS: tuple[int, ...] = (110, 112, 113)


# ------------------------------------------------------------------ fixtures


def _table() -> Any:
    return pa.table({name: pa.array(values) for name, values in SYNTHETIC.items()})


def _annotations(**overrides: Any) -> RetainedAnnotations:
    columns = {name: np.asarray(values, dtype=object) for name, values in SYNTHETIC.items()}
    defaults: dict[str, Any] = {
        "body_ids": np.asarray(SYNTHETIC["bodyId"], dtype=np.uint64),
        "source_rows": np.arange(NODE_COUNT, dtype=np.uint64),
        "columns": columns,
        "source_row_count": SYNTHETIC_SOURCE_ROWS,
        "annotations_sha256": "a" * 64,
    }
    defaults.update(overrides)
    return RetainedAnnotations(**defaults)


def _support() -> EdgeSupport:
    return EdgeSupport(node_count=NODE_COUNT, pre=EDGE_PRE, post=EDGE_POST, weights=EDGE_WEIGHT)


def _fixture_rows() -> list[int]:
    return [SYNTHETIC["bodyId"].index(body) for body in FIXTURE_BODY_IDS]


def _fixture_annotations() -> RetainedAnnotations:
    rows = _fixture_rows()
    columns = {
        name: np.asarray([values[row] for row in rows], dtype=object)
        for name, values in SYNTHETIC.items()
    }
    return RetainedAnnotations(
        body_ids=np.asarray(FIXTURE_BODY_IDS, dtype=np.uint64),
        source_rows=np.asarray(rows, dtype=np.uint64),
        columns=columns,
        source_row_count=SYNTHETIC_SOURCE_ROWS,
        annotations_sha256="a" * 64,
    )


def _fixture_taken() -> Any:
    return _table().take(np.asarray(_fixture_rows(), dtype=np.int64))


def _fixture_graph() -> RuntimeGraph:
    nodes = [
        SyntheticNode(
            body_id=body,
            roles=("motor",) if body != 110 else ("descending",),
            side="R",
            type_name=str(SYNTHETIC["type"][SYNTHETIC["bodyId"].index(body)]),
            superclass=str(SYNTHETIC["superclass"][SYNTHETIC["bodyId"].index(body)]),
        )
        for body in FIXTURE_BODY_IDS
    ]
    edges = [
        SyntheticEdge(body_pre=110, body_post=112, raw_count=40, source_row=1),
        SyntheticEdge(body_pre=110, body_post=113, raw_count=35, source_row=2),
    ]
    return build_synthetic_graph("synthetic:vision-fixture", nodes, edges)


def _gates(annotations: RetainedAnnotations | None = None) -> list[IntegrityGate]:
    resolved = _annotations() if annotations is None else annotations
    return [
        *annotation_gates(
            resolved,
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
def support() -> EdgeSupport:
    return _support()


@pytest.fixture
def index(annotations: RetainedAnnotations, taken: Any) -> Any:
    return build_column_index(
        hex_addresses(taken), annotations.column("type"), annotations.column("somaSide")
    )


# ------------------------------------------------------------ hex addressing


def test_hex_addresses_require_both_coordinates(taken: Any) -> None:
    addresses = hex_addresses(taken)
    assert int(np.count_nonzero(addresses.addressed)) == 10
    # Row 18 carries only assignedOlHex1 and must not count as addressed.
    assert bool(addresses.addressed[18]) is False
    assert addresses.key(0) == (5, 5)


def test_address_census_reports_partials_integrality_and_ranges(
    annotations: RetainedAnnotations, taken: Any
) -> None:
    census = address_census(
        hex_addresses(taken), annotations.column("type"), annotations.column("somaSide")
    )
    assert census["column_addressed_nodes"] == 10
    assert census["partially_addressed_nodes"] == 1
    assert census["addresses_are_integral"] is True
    assert census["hex1_range"] == {"minimum": 5, "maximum": 9}
    assert census["types_carrying_an_address"] == {"L1": 5, "L2": 4, "Mi1": 1}
    assert census["soma_side_counts"] == {"L": 2, "R": 8}


def test_type_family_rules_are_exact_or_prefix_and_null_safe() -> None:
    exact = TypeFamily(name="C2", exact=("C2",))
    prefix = TypeFamily(name="Tm", prefixes=("Tm",))
    assert exact.matches("C2") and not exact.matches("C20")
    assert prefix.matches("Tm9") and prefix.matches("Tm") and not prefix.matches("T5a")
    assert not exact.matches(None) and not prefix.matches(None)
    assert "type equals 'C2'" in exact.rule
    assert "type starts with 'Tm'" in prefix.rule


def test_type_family_report_counts_every_declared_family(
    annotations: RetainedAnnotations,
) -> None:
    report = type_family_report(annotations.column("type"), scope="synthetic")
    counts = {row["family"]: row["cells"] for row in report["families"]}
    assert counts["L1-L5"] == 10
    assert counts["Mi"] == 1
    assert counts["T4"] == 1
    assert counts["Tm"] == 0 and counts["R1-R6"] == 0 and counts["C2"] == 0
    assert "Tm" in report["families_with_no_member"]
    matched = {row["family"]: row["matched_types"] for row in report["families"]}
    assert matched["L1-L5"] == {"L1": 6, "L2": 4}


# ------------------------------------------------------------------ columns


def test_column_index_groups_one_eye_by_address_and_type(index: Any) -> None:
    assert sorted(index.members) == [(5, 5), (5, 6), (6, 5), (9, 9)]
    assert index.members[(5, 5)] == {"L1": (0,), "L2": (1,), "Mi1": (17,)}
    assert index.complete(PATCH_TYPES) == ((5, 5), (5, 6), (6, 5))
    assert index.complete(COLUMNAR_TYPES) == ((5, 5),)
    assert index.cells([(5, 5), (6, 5)], PATCH_TYPES).tolist() == [0, 1, 2, 3]


def test_column_census_counts_both_eyes_separately(
    annotations: RetainedAnnotations, taken: Any
) -> None:
    census = column_census(
        hex_addresses(taken),
        annotations.column("type"),
        annotations.column("somaSide"),
        patch_types=PATCH_TYPES,
        columnar_types=COLUMNAR_TYPES,
    )
    rows = {row["soma_side"]: row for row in census["rows"]}
    assert rows["R"]["columns"] == 4
    assert rows["R"]["columns_complete_over_the_patch_types"] == 3
    assert rows["R"]["columns_complete_over_all_columnar_types"] == 1
    assert rows["L"]["columns"] == 1
    assert rows["L"]["columns_complete_over_all_columnar_types"] == 0
    assert rows["R"]["cells_of_patch_types_in_complete_columns"] == 6
    assert rows["R"]["cells_of_all_types_in_complete_columns"] == 7
    assert "not a frozen repository rule" in census["patch_type_selection_note"]


def test_reproduction_check_flags_a_deviation_from_the_published_optic_figures() -> None:
    census = {
        "rows": [
            {
                "soma_side": "R",
                "columns": 892,
                "columns_complete_over_all_columnar_types": 797,
            },
            {
                "soma_side": "L",
                "columns": 879,
                "columns_complete_over_all_columnar_types": 0,
            },
        ]
    }
    assert reproduction_check(census, OPTIC_CENSUS_01["column_addressed_nodes"])["reproduced"]
    drifted = reproduction_check(census, 23_719)
    assert drifted["reproduced"] is False
    failing = [row for row in drifted["checks"] if not row["matches"]]
    assert [row["quantity"] for row in failing] == ["column_addressed_nodes"]


# ------------------------------------------------------------ hex geometry


def test_hex_distance_is_zero_on_identity_symmetric_and_metric_specific() -> None:
    assert hex_distance((3, 4), (3, 4), "axial_60") == 0.0
    assert hex_distance((0, 0), (1, -1), "axial_60") == 1.0
    assert hex_distance((0, 0), (1, -1), "axial_120") == 2.0
    assert hex_distance((0, 0), (1, 1), "axial_60") == 2.0
    assert hex_distance((0, 0), (1, 1), "axial_120") == 1.0
    assert hex_distance((0, 0), (3, 4), "chebyshev") == 4.0
    assert hex_distance((0, 0), (3, 4), "euclidean") == pytest.approx(5.0)
    assert hex_distance((1, 2), (4, 9), "axial_60") == hex_distance((4, 9), (1, 2), "axial_60")
    with pytest.raises(ValueError, match="unknown hex metric"):
        hex_distance((0, 0), (1, 1), "spiral")


def test_hex_neighbours_are_all_at_lattice_distance_one() -> None:
    for metric in ("axial_60", "axial_120"):
        neighbours = hex_neighbours((4, 7), metric)
        assert len(set(neighbours)) == 6
        assert all(hex_distance((4, 7), item, metric) == 1.0 for item in neighbours)


def test_select_disc_is_deterministic_and_centred_on_the_complete_columns(
    index: Any,
) -> None:
    complete = index.complete(PATCH_TYPES)
    disc = select_disc(complete, 2)
    assert disc.centre == (5, 5)
    assert disc.columns == ((5, 5), (5, 6))
    assert disc.centroid == pytest.approx((16 / 3, 16 / 3))
    assert select_disc(complete, 3).columns == ((5, 5), (5, 6), (6, 5))
    assert select_disc(complete, 3).radius == 1.0
    assert select_disc(complete, 2) == select_disc(complete, 2)
    with pytest.raises(ValueError, match="more columns than are complete"):
        select_disc(complete, 4)


def test_is_contiguous_separates_a_connected_patch_from_a_split_one() -> None:
    assert is_contiguous([(5, 5), (5, 6), (6, 5)], "axial_60") is True
    assert is_contiguous([(5, 5), (9, 9)], "axial_60") is False
    assert is_contiguous([], "axial_60") is True


# -------------------------------------------------------------- edge support


def _python_induced(nodes: list[int], threshold: int) -> int:
    """Independent per-edge accumulation of induced edges."""

    members = set(nodes)
    return sum(
        1
        for position in range(EDGE_PRE.size)
        if int(EDGE_PRE[position]) in members
        and int(EDGE_POST[position]) in members
        and int(EDGE_WEIGHT[position]) >= threshold
    )


def test_induced_edges_match_an_independent_per_edge_accumulation(
    support: EdgeSupport,
) -> None:
    patch = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    counts = support.induced_edges(patch)
    assert counts == {"1": 4, "5": 3, "10": 2, "20": 1}
    for threshold in (1, 5, 10, 20):
        assert counts[str(threshold)] == _python_induced(patch.tolist(), threshold)
    assert support.induced_weight(patch) == 7 + 12 + 4 + 25


def test_patch_sizing_records_geometry_and_edges_at_every_threshold(
    support: EdgeSupport, index: Any, annotations: RetainedAnnotations
) -> None:
    complete = index.complete(PATCH_TYPES)
    report = patch_sizing(
        support, index, annotations, complete, sizes=(2, 3), patch_types=PATCH_TYPES
    )
    rows = {row["columns"]: row for row in report["rows"]}
    assert rows[2]["nodes"] == 4 and rows[3]["nodes"] == 6
    assert rows[3]["induced_edges"] == {"1": 4, "5": 3, "10": 2, "20": 1}
    assert rows[3]["contiguous"] is True
    assert rows[3]["nodes_per_type"] == {"L1": 3, "L2": 3}
    assert rows[3]["body_ids"] == [100, 101, 102, 103, 104, 105]
    assert "NOT a frozen repository rule" in report["selection_rule"]


def test_metric_ambiguity_reports_overlap_and_the_worst_deviation(
    support: EdgeSupport, index: Any
) -> None:
    complete = index.complete(PATCH_TYPES)
    report = metric_ambiguity(support, index, complete, sizes=(2,), patch_types=PATCH_TYPES)
    reference = [row for row in report["rows"] if row["hex_metric"] == "axial_60"]
    assert reference[0]["columns_shared_with_reference"] == 2
    assert reference[0]["largest_relative_edge_deviation_from_reference"] == 0.0
    assert report["largest_relative_edge_deviation_across_metrics"] >= 0.0
    assert len(report["rows"]) == 4


# ------------------------------------------------------- cross-connectivity


def _python_cross(left: list[int], right: list[int]) -> tuple[int, int]:
    """Independent per-edge scan of edges running between two node sets."""

    first, second = set(left), set(right)
    forward = sum(
        1
        for position in range(EDGE_PRE.size)
        if int(EDGE_PRE[position]) in first and int(EDGE_POST[position]) in second
    )
    backward = sum(
        1
        for position in range(EDGE_PRE.size)
        if int(EDGE_PRE[position]) in second and int(EDGE_POST[position]) in first
    )
    return forward, backward


def test_cross_connectivity_counts_both_directions_directly(
    support: EdgeSupport, annotations: RetainedAnnotations
) -> None:
    patch = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    fixture = np.asarray([10, 12, 13], dtype=np.int64)
    report = cross_connectivity(
        support,
        left=patch,
        right=fixture,
        left_name="patch",
        right_name="fixture",
        types=annotations.column("type"),
        body_ids=annotations.body_ids,
    )
    assert (report["left_to_right_edges"], report["right_to_left_edges"]) == (0, 0)
    assert report["total_edges"] == 0 and report["total_weight"] == 0
    assert report["edges"] == [] and report["edges_listed_is_complete"] is True
    assert (report["left_to_right_edges"], report["right_to_left_edges"]) == _python_cross(
        patch.tolist(), fixture.tolist()
    )

    addressed = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 17], dtype=np.int64)
    wider = cross_connectivity(
        support,
        left=addressed,
        right=fixture,
        left_name="all_addressed",
        right_name="fixture",
        types=annotations.column("type"),
        body_ids=annotations.body_ids,
    )
    assert (wider["left_to_right_edges"], wider["right_to_left_edges"]) == (1, 0)
    assert wider["edges"] == [
        {
            "pre_body_id": 106,
            "pre_type": "L1",
            "post_body_id": 112,
            "post_type": "hg1 MN",
            "weight": 9,
        }
    ]


def test_cross_connectivity_report_names_the_populations_with_no_edge(
    support: EdgeSupport, annotations: RetainedAnnotations, taken: Any, index: Any
) -> None:
    addresses = hex_addresses(taken)
    complete = index.complete(PATCH_TYPES)
    fixture = dense_indices_for(annotations.body_ids, FIXTURE_BODY_IDS)
    report = cross_connectivity_report(
        support,
        index,
        annotations,
        addresses,
        complete,
        fixture,
        sizes=(2, 3),
        patch_types=PATCH_TYPES,
        columnar_types=COLUMNAR_TYPES,
    )
    rows = {row["left_set"]: row for row in report["rows"]}
    assert rows["patch_of_2_complete_columns"]["total_edges"] == 0
    assert rows["patch_of_3_complete_columns"]["total_edges"] == 0
    assert rows["all_patch_type_cells_in_every_complete_column"]["total_edges"] == 0
    assert rows["all_column_addressed_nodes"]["total_edges"] == 1
    assert "all_column_addressed_nodes" not in report["populations_with_no_edge_to_the_fixture"]
    assert "patch_of_3_complete_columns" in report["populations_with_no_edge_to_the_fixture"]


def test_dense_indices_refuse_a_body_the_artifact_does_not_hold(
    annotations: RetainedAnnotations,
) -> None:
    assert dense_indices_for(annotations.body_ids, FIXTURE_BODY_IDS).tolist() == [10, 12, 13]
    with pytest.raises(KeyError):
        dense_indices_for(annotations.body_ids, [100, 999_999])


# ----------------------------------------------------- descending support


def _python_hops_to(targets: list[int], threshold: int) -> dict[int, int]:
    """Independent breadth-first traversal against the reversed edge list."""

    incoming: dict[int, list[int]] = {}
    for position in range(EDGE_PRE.size):
        if int(EDGE_WEIGHT[position]) >= threshold:
            incoming.setdefault(int(EDGE_POST[position]), []).append(int(EDGE_PRE[position]))
    distances = {node: 0 for node in targets}
    queue = deque(targets)
    while queue:
        node = queue.popleft()
        for source in incoming.get(node, []):
            if source not in distances:
                distances[source] = distances[node] + 1
                queue.append(source)
    return distances


def test_descending_support_matches_an_independent_python_traversal(
    support: EdgeSupport, annotations: RetainedAnnotations
) -> None:
    descending = descending_pool(annotations)
    assert descending.tolist() == [10, 11]
    patch = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    report = descending_support(
        support.subgraph(10), patch=patch, descending=descending, minimum_weight=10
    )
    assert report["cells_within_maximum_hops"] == 2
    assert report["fraction_within_maximum_hops"] == pytest.approx(2 / 6)
    assert report["hop_distance_histogram"] == {"unreachable": 4, "2": 2}
    assert report["maximum_hops"] == report["max_graph_hops"] == 2
    assert report["fresh_input_minimum_neural_updates_for_hop_budget"] == 5

    expected = _python_hops_to(descending.tolist(), 10)
    assert sum(1 for node in patch.tolist() if expected.get(node, 99) <= 2) == 2
    tighter = descending_support(
        support.subgraph(20), patch=patch, descending=descending, minimum_weight=20
    )
    assert tighter["cells_within_maximum_hops"] == 1
    assert tighter["subgraph_edges"] < report["subgraph_edges"]


# ------------------------------------------------------------- closure tiers


def _python_path_interiors(patch: list[int], targets: list[int], threshold: int) -> set[int]:
    """Brute-force every simple path of at most three edges, collecting interiors."""

    outgoing: dict[int, list[int]] = {}
    for position in range(EDGE_PRE.size):
        if int(EDGE_WEIGHT[position]) >= threshold:
            outgoing.setdefault(int(EDGE_PRE[position]), []).append(int(EDGE_POST[position]))
    interiors: set[int] = set()
    goals = set(targets)
    for start in patch:
        stack: list[list[int]] = [[start]]
        while stack:
            path = stack.pop()
            if len(path) > 4:
                continue
            head = path[-1]
            if len(path) > 1 and head in goals:
                interiors.update(path[1:-1])
                continue
            if len(path) == 4:
                continue
            for target in outgoing.get(head, []):
                if target not in path:
                    stack.append([*path, target])
    return interiors


def test_closure_tiers_are_nested_and_the_interior_pool_is_exact(
    support: EdgeSupport, annotations: RetainedAnnotations
) -> None:
    patch = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    descending = descending_pool(annotations)
    motor = wing_motor_pool(annotations)
    assert motor.tolist() == [12, 13]
    tiers, reached = closure_tiers(
        support.subgraph(10), patch=patch, descending=descending, motor=motor
    )
    assert reached.tolist() == [10]
    names = [tier.name for tier in tiers]
    assert names == [
        "mandatory_core",
        "path_interior_closure",
        "two_hop_candidate_closure",
    ]
    assert tiers[0].intermediate_pool.size == 0
    assert tiers[1].intermediate_pool.tolist() == [16]
    assert set(tiers[1].intermediate_pool.tolist()) == _python_path_interiors(
        patch.tolist(), reached.tolist(), 10
    )
    assert 19 in tiers[2].intermediate_pool.tolist()
    assert set(tiers[0].nodes.tolist()) < set(tiers[1].nodes.tolist())
    assert set(tiers[1].nodes.tolist()) < set(tiers[2].nodes.tolist())


def test_closure_report_sizes_every_tier_against_the_fixture_ceiling(
    support: EdgeSupport, annotations: RetainedAnnotations
) -> None:
    patch = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    report = closure_report(
        support,
        support.subgraph(10),
        patch=patch,
        descending=descending_pool(annotations),
        motor=wing_motor_pool(annotations),
    )
    assert report["descending_neurons_reached"] == 1
    assert report["fixture_edge_ceiling_exclusive"] == FLIGHT_FIXTURE_EDGE_CEILING
    rows = {row["closure"]: row for row in report["tiers"]}
    assert [int(row["induced_edges"]["1"]) for row in report["tiers"]] == [6, 9, 10]
    tiers, _ = closure_tiers(
        support.subgraph(10),
        patch=patch,
        descending=descending_pool(annotations),
        motor=wing_motor_pool(annotations),
    )
    for tier in tiers:
        row = rows[tier.name]
        assert row["closure_nodes"] == int(tier.nodes.size)
        assert row["induced_edges"]["1"] == _python_induced(tier.nodes.tolist(), 1)
        assert row["exceeds_the_fixture_edge_ceiling"] is False
        assert row["multiple_of_the_fixture_edge_ceiling"] < 1.0


# ------------------------------------------------ lobula plate tangential


def test_tangential_census_reports_types_and_sides(
    annotations: RetainedAnnotations,
) -> None:
    members = type_prefix_members(annotations.column("type"), LPTC_TYPE_PREFIXES)
    assert members.tolist() == [14, 15]
    census = tangential_census(annotations, members, prefixes=LPTC_TYPE_PREFIXES)
    assert census["cells"] == 2
    assert census["type_counts"] == {"HSE": 1, "VS": 1}
    assert census["soma_side_counts"] == {"L": 1, "R": 1}
    assert census["superclass_counts"] == {"visual_projection": 2}
    assert census["body_ids"] == [114, 115]


def test_tangential_circuits_grow_with_the_declared_membership(
    support: EdgeSupport, annotations: RetainedAnnotations
) -> None:
    members = type_prefix_members(annotations.column("type"), LPTC_TYPE_PREFIXES)
    report = tangential_circuits(
        support, annotations, tangential=members, motor=wing_motor_pool(annotations)
    )
    rows = {row["circuit"]: row for row in report["rows"]}
    assert report["descending_neurons_contacted_directly"] == 1
    assert rows["tangential_only"]["nodes"] == 2
    assert rows["tangential_only"]["induced_edges"] == {"1": 0, "5": 0, "10": 0, "20": 0}
    assert rows["tangential_and_wing_motor"]["nodes"] == 4
    assert rows["tangential_and_wing_motor"]["induced_edges"] == {
        "1": 1,
        "5": 1,
        "10": 1,
        "20": 0,
    }
    assert rows["tangential_descending_and_wing_motor"]["nodes"] == 5
    assert rows["tangential_descending_and_wing_motor"]["induced_edges"] == {
        "1": 5,
        "5": 5,
        "10": 4,
        "20": 3,
    }


def _python_input_weight(members: list[int], threshold: int) -> tuple[int, int]:
    """Independent per-edge accumulation of incoming weight and its T4/T5 share."""

    inside = set(members)
    total = 0
    pathway = 0
    for position in range(EDGE_PRE.size):
        weight = int(EDGE_WEIGHT[position])
        if int(EDGE_POST[position]) not in inside or weight < threshold:
            continue
        total += weight
        label = SYNTHETIC["type"][int(EDGE_PRE[position])]
        if label is not None and str(label).startswith(("T4", "T5")):
            pathway += weight
    return pathway, total


def test_tangential_input_composition_depends_on_the_declared_weight_floor(
    support: EdgeSupport, annotations: RetainedAnnotations
) -> None:
    members = type_prefix_members(annotations.column("type"), LPTC_TYPE_PREFIXES)
    report = tangential_input_composition(support, annotations, tangential=members)
    rows = {row["minimum_weight"]: row for row in report["rows"]}
    assert rows[1]["total_incoming_weight"] == 46
    assert rows[1]["motion_pathway_weight"] == 40
    assert rows[1]["motion_pathway_weight_fraction"] == pytest.approx(40 / 46)
    assert rows[10]["motion_pathway_weight_fraction"] == pytest.approx(1.0)
    for threshold, row in rows.items():
        pathway, total = _python_input_weight(members.tolist(), threshold)
        assert (row["motion_pathway_weight"], row["total_incoming_weight"]) == (pathway, total)
    assert report["motion_pathway_weight_by_type"]["1"] == {"T4a": 40}
    assert report["top_presynaptic_types"][0] == {"presynaptic_type": "T4a", "weight": 40}


def test_motion_pathway_cells_carry_no_column_address(
    annotations: RetainedAnnotations, taken: Any
) -> None:
    report = motion_pathway_addressing(annotations, hex_addresses(taken).addressed)
    assert report["cells"] == 1
    assert report["cells_with_a_column_address"] == 0
    assert report["cells_with_a_column_address_fraction"] == 0.0
    assert report["type_counts"] == {"T4a": 1}


# ------------------------------------------------------------ whole report


def _analysis(**overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "identity": SYNTHETIC_IDENTITY,
        "gates": _gates(),
        "sizes": (2, 3),
        "closure_sizes": (3,),
        "support_thresholds": (10, 20),
        "patch_types": PATCH_TYPES,
        "columnar_types": COLUMNAR_TYPES,
    }
    defaults.update(overrides)
    return analyse(
        _annotations(),
        _table(),
        _support(),
        _fixture_graph(),
        _fixture_annotations(),
        _fixture_taken(),
        **defaults,
    )


def test_analysis_body_is_deterministic_and_carries_every_section() -> None:
    first = _analysis()
    second = _analysis()
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first["schema"] == SCHEMA
    assert set(first) == {
        "schema",
        "artifact",
        "integrity_gates",
        "question",
        "flight_fixture_visual_substrate",
        "column_address_census",
        "optic_census_01_reproduction",
        "retinotopic_patch_sizing",
        "hex_metric_ambiguity",
        "patch_to_flight_fixture_cross_connectivity",
        "patch_to_descending_support",
        "patch_circuit_closure",
        "lobula_plate_tangential_alternative",
        "interpretation_bounds",
    }
    assert first["integrity_gates"]["all_passed"] is True
    assert len(first["patch_to_descending_support"]["rows"]) == 4
    assert len(first["patch_circuit_closure"]["rows"]) == 1


def test_analysis_refuses_to_produce_a_body_when_a_gate_fails() -> None:
    failing = _gates(_annotations(annotations_sha256="c" * 64))
    with pytest.raises(ReadoutIntegrityError, match="pinned_annotations_file_sha256"):
        _analysis(gates=failing)


def test_no_reserved_causal_vocabulary_survives_in_the_report_body() -> None:
    body = canonical_json_bytes(_analysis()).decode()
    assert "causal" not in body
    bounds = _analysis()["interpretation_bounds"]
    assert (
        "bounds participation and never demonstrates it"
        in (bounds["reachability_is_an_upper_bound"])
    )
    assert "invalidates no existing receipt" in bounds["no_gate_movement"]
    assert "not repository rules" in bounds["patch_geometry_is_this_analysis_own_choice"]


def test_interpretation_bounds_state_that_a_zero_is_topological() -> None:
    bounds = _analysis()["interpretation_bounds"]
    assert "not a claim about the animal" in bounds["a_zero_edge_count_is_a_topological_fact"]
    assert "no gate" in bounds["no_gate_movement"]


# -------------------------------------------------- tracked flight fixture


def _tracked_fixture() -> RuntimeGraph:
    return load_malecns_fixture(
        ROOT / MALECNS_FLIGHT_FIXTURE_V1_PATH,
        expected_pin=MALECNS_FLIGHT_FIXTURE_V1_PIN,
        expected_file_bytes=MALECNS_FLIGHT_FIXTURE_V1_FILE_BYTES,
    )


def test_tracked_flight_fixture_holds_no_visual_type_family_member() -> None:
    """The committed fixture is the reason a retinotopic injection needs a third one."""

    graph = _tracked_fixture()
    graph.assert_integrity()
    assert graph.manifest.node_count == MALECNS_FLIGHT_FIXTURE_V1_NODE_COUNT
    assert graph.manifest.edge_count == MALECNS_FLIGHT_FIXTURE_V1_EDGE_COUNT
    types = np.asarray([node.type_name for node in graph.nodes], dtype=object)
    report = type_family_report(types, scope="tracked flight fixture")
    assert report["total_cells_in_any_family"] == 0
    assert len(report["families_with_no_member"]) == len(report["families"])


def test_tracked_flight_fixture_has_exactly_one_visual_projection_node() -> None:
    graph = _tracked_fixture()
    visual = [node for node in graph.nodes if node.superclass == "visual_projection"]
    assert len(visual) == 1
    only = visual[0]
    assert (only.body_id, only.type_name, only.side, only.roles) == (
        16_366,
        "LoVP50",
        "R",
        ("intermediate",),
    )


def test_fixture_gates_bind_the_tracked_fixture_to_the_pinned_annotations() -> None:
    graph = _tracked_fixture()
    declared = next(
        source.sha256
        for source in graph.manifest.source_artifacts
        if source.source_id == "annotations"
    )
    gates = fixture_gates(graph, declared, 211_577)
    assert all(gate.passed for gate in gates), [g.name for g in gates if not g.passed]
    drifted = fixture_gates(graph, "d" * 64, 211_577)
    assert [gate.name for gate in drifted if not gate.passed] == [
        "fixture_and_artifact_share_one_annotations_source"
    ]
    narrow = fixture_gates(graph, declared, 10)
    assert not narrow[2].passed


def test_fixture_visual_substrate_reports_a_synthetic_fixture_faithfully() -> None:
    block = fixture_visual_substrate(_fixture_graph(), _fixture_annotations(), _fixture_taken())
    assert block["fixture_nodes"] == 3
    assert block["column_address_census"]["column_addressed_nodes"] == 0
    assert block["visual_type_families"]["total_cells_in_any_family"] == 0
    assert block["visual_superclass_nodes"] == []
    assert block["superclass_composition"] == {"descending_neuron": 1, "vnc_motor": 2}
    assert "would require a third fixture" in block["finding"]


# ------------------------------------------------------- full-artifact tests


_ARTIFACT_PRESENT = (ROOT / DEFAULT_ARTIFACT_DIR / "manifest.json").is_file() and (
    ROOT / DEFAULT_ANNOTATIONS
).is_file()

_SKIP = pytest.mark.skipif(
    not _ARTIFACT_PRESENT,
    reason="the full MaleCNS runtime artifact and pinned annotations are ignored and absent",
)


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    """Run the production entry point once and read the report it wrote."""

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "vision-feasibility.json"
        assert (
            main(
                [
                    "--artifact",
                    str(ROOT / DEFAULT_ARTIFACT_DIR),
                    "--manifest-sha256",
                    reviewed_full_artifact_manifest_sha256(ROOT / DEFAULT_ARTIFACT_DIR),
                    "--annotations",
                    str(ROOT / DEFAULT_ANNOTATIONS),
                    "--fixture",
                    str(ROOT / MALECNS_FLIGHT_FIXTURE_V1_PATH),
                    "--output",
                    str(output),
                ]
            )
            == 0
        )
        document: dict[str, Any] = json.loads(output.read_text())
    return document


@_SKIP
def test_full_artifact_gates_pass_and_reproduce_the_optic_census(
    report: dict[str, Any],
) -> None:
    analysis = report["analysis"]
    assert analysis["integrity_gates"]["all_passed"] is True
    assert len(analysis["integrity_gates"]["gates"]) == 12
    assert analysis["optic_census_01_reproduction"]["reproduced"] is True
    rows = {row["soma_side"]: row for row in analysis["column_address_census"]["rows"]}
    assert rows["R"]["columns"] == 892
    assert rows["L"]["columns"] == 879
    assert rows["R"]["columns_complete_over_all_columnar_types"] == 797
    assert rows["R"]["columns_complete_over_the_patch_types"] == 871
    assert rows["R"]["cells_of_patch_types_in_complete_columns"] == 6_980
    assert analysis["column_address_census"]["rows"][0]["soma_side"] == "L"


@_SKIP
def test_full_artifact_flight_fixture_has_no_visual_substrate(
    report: dict[str, Any],
) -> None:
    block = report["analysis"]["flight_fixture_visual_substrate"]
    assert block["fixture_nodes"] == 128
    assert block["column_address_census"]["column_addressed_nodes"] == 0
    assert block["visual_type_families"]["total_cells_in_any_family"] == 0
    assert [row["cells"] for row in block["visual_type_families"]["families"]] == [0] * 10
    assert len(block["visual_superclass_nodes"]) == 1
    only = block["visual_superclass_nodes"][0]
    assert (only["body_id"], only["type"], only["soma_side"], only["roles"]) == (
        16_366,
        "LoVP50",
        "R",
        ["intermediate"],
    )
    assert only["column_address"] is None
    assert block["superclass_composition"]["visual_projection"] == 1


@_SKIP
def test_full_artifact_patch_sizes_match_the_recorded_figures(
    report: dict[str, Any],
) -> None:
    rows = {row["columns"]: row for row in report["analysis"]["retinotopic_patch_sizing"]["rows"]}
    expected = {
        4: (32, {"1": 99, "5": 59, "10": 51, "20": 43}),
        8: (64, {"1": 231, "5": 118, "10": 103, "20": 89}),
        16: (128, {"1": 507, "5": 240, "10": 207, "20": 175}),
        32: (256, {"1": 1075, "5": 482, "10": 408, "20": 356}),
    }
    for columns, (nodes, edges) in expected.items():
        assert rows[columns]["nodes"] == nodes
        assert rows[columns]["induced_edges"] == edges
        assert rows[columns]["contiguous"] is True
        assert rows[columns]["centre_column"] == [19, 20]


@_SKIP
def test_full_artifact_patch_has_no_edge_to_the_committed_fixture(
    report: dict[str, Any],
) -> None:
    block = report["analysis"]["patch_to_flight_fixture_cross_connectivity"]
    rows = {row["left_set"]: row for row in block["rows"]}
    for name in (
        "patch_of_4_complete_columns",
        "patch_of_8_complete_columns",
        "patch_of_16_complete_columns",
        "patch_of_32_complete_columns",
        "all_patch_type_cells_in_every_complete_column",
    ):
        assert rows[name]["total_edges"] == 0, name
        assert rows[name]["shared_nodes"] == 0
    assert rows["all_patch_type_cells_in_every_complete_column"]["left_nodes"] == 6_980
    assert rows["all_column_addressed_nodes"]["left_nodes"] == 23_720
    assert rows["all_column_addressed_nodes"]["left_to_right_edges"] == 4
    assert rows["all_column_addressed_nodes"]["right_to_left_edges"] == 0
    assert {edge["post_type"] for edge in rows["all_column_addressed_nodes"]["edges"]} == {"LoVP50"}
    assert {edge["pre_type"] for edge in rows["all_column_addressed_nodes"]["edges"]} == {"Tm4"}


@_SKIP
def test_full_artifact_descending_support_is_sparse_and_vanishes_at_weight_twenty(
    report: dict[str, Any],
) -> None:
    rows = {
        (row["minimum_edge_weight"], row["columns"]): row
        for row in report["analysis"]["patch_to_descending_support"]["rows"]
    }
    assert rows[(10, 4)]["cells_within_maximum_hops"] == 1
    assert rows[(10, 4)]["fraction_within_maximum_hops"] == pytest.approx(1 / 32)
    assert rows[(10, 32)]["cells_within_maximum_hops"] == 9
    assert rows[(10, 32)]["fraction_within_maximum_hops"] == pytest.approx(9 / 256)
    for columns in (4, 8, 16, 32):
        assert rows[(20, columns)]["cells_within_maximum_hops"] == 0
    assert report["analysis"]["patch_to_descending_support"]["descending_pool"]["cells"] == 1_314


@_SKIP
def test_full_artifact_closure_exceeds_the_fixture_ceiling_at_every_tier(
    report: dict[str, Any],
) -> None:
    blocks = {row["columns"]: row for row in report["analysis"]["patch_circuit_closure"]["rows"]}
    expected = {
        4: {
            "mandatory_core": (0, 305, 6_035, 2_375),
            "path_interior_closure": (309, 614, 16_772, 6_744),
            "two_hop_candidate_closure": (9_298, 9_598, 408_095, 133_126),
        },
        16: {
            "mandatory_core": (0, 532, 12_686, 4_679),
            "path_interior_closure": (732, 1_264, 47_096, 17_540),
            "two_hop_candidate_closure": (11_154, 11_679, 635_939, 203_931),
        },
        32: {
            "mandatory_core": (0, 684, 14_428, 5_266),
            "path_interior_closure": (1_166, 1_850, 73_899, 26_408),
            "two_hop_candidate_closure": (12_435, 13_112, 843_264, 261_448),
        },
    }
    for columns, tiers in expected.items():
        rows = {row["closure"]: row for row in blocks[columns]["tiers"]}
        for name, (pool, nodes, at_one, at_five) in tiers.items():
            assert rows[name]["intermediate_pool_bodies"] == pool, (columns, name)
            assert rows[name]["closure_nodes"] == nodes, (columns, name)
            assert rows[name]["induced_edges"]["1"] == at_one, (columns, name)
            assert rows[name]["induced_edges"]["5"] == at_five, (columns, name)
            assert rows[name]["exceeds_the_fixture_edge_ceiling"] is True, (columns, name)
    assert blocks[4]["descending_neurons_reached"] == 206
    assert blocks[32]["descending_neurons_reached"] == 361
    assert blocks[32]["tiers"][2]["multiple_of_the_fixture_edge_ceiling"] == pytest.approx(
        261_448 / FLIGHT_FIXTURE_EDGE_CEILING
    )


@_SKIP
def test_full_artifact_tangential_alternative_fits_inside_the_ceiling(
    report: dict[str, Any],
) -> None:
    block = report["analysis"]["lobula_plate_tangential_alternative"]
    census = block["census"]
    assert census["cells"] == 44
    assert census["soma_side_counts"] == {"L": 21, "R": 23}
    assert census["type_counts"] == {
        "H2": 2, "HSE": 2, "HSN": 2, "HSS": 2, "HST": 2,
        "VS": 18, "VST1": 5, "VST2": 7, "VSm": 4,
    }  # fmt: skip
    rows = {row["circuit"]: row for row in block["circuits"]["rows"]}
    assert rows["tangential_only"]["nodes"] == 44
    assert rows["tangential_only"]["induced_edges"] == {"1": 260, "5": 45, "10": 14, "20": 2}
    assert rows["tangential_and_wing_motor"]["nodes"] == 111
    assert rows["tangential_and_wing_motor"]["induced_edges"] == {
        "1": 563,
        "5": 96,
        "10": 28,
        "20": 5,
    }
    assert rows["tangential_descending_and_wing_motor"]["nodes"] == 159
    assert rows["tangential_descending_and_wing_motor"]["induced_edges"] == {
        "1": 1_448,
        "5": 584,
        "10": 393,
        "20": 247,
    }
    assert block["circuits"]["descending_neurons_contacted_directly"] == 48


@_SKIP
def test_full_artifact_tangential_input_is_dominated_by_the_motion_pathway(
    report: dict[str, Any],
) -> None:
    block = report["analysis"]["lobula_plate_tangential_alternative"]
    rows = {row["minimum_weight"]: row for row in block["input_composition"]["rows"]}
    assert rows[1]["total_incoming_weight"] == 423_346
    assert rows[1]["motion_pathway_weight"] == 348_515
    assert rows[1]["motion_pathway_weight_fraction"] == pytest.approx(0.823239, abs=5e-6)
    assert rows[5]["motion_pathway_weight_fraction"] == pytest.approx(0.843036, abs=5e-6)
    assert block["input_composition"]["motion_pathway_weight_by_type"]["1"] == {
        "T4a": 59_285, "T4b": 37_767, "T4c": 147, "T4d": 67_871,
        "T5a": 69_211, "T5b": 42_604, "T5c": 115, "T5d": 71_515,
    }  # fmt: skip
    addressing = block["motion_pathway_column_addressing"]
    assert addressing["cells"] == 13_585
    assert addressing["cells_with_a_column_address"] == 0


@_SKIP
def test_full_artifact_report_is_deterministic_and_keeps_literature_unhashed(
    report: dict[str, Any],
) -> None:
    digest = hashlib.sha256(canonical_json_bytes(report["analysis"])).hexdigest()
    assert digest == report["analysis_sha256"]
    assert "causal" not in canonical_json_bytes(report["analysis"]).decode()
    assert report["literature_inferred_notes"]["status"].startswith("literature-inferred")
    assert "literature_inferred_notes" not in report["analysis"]
    assert set(report["execution"]) == {
        "elapsed_seconds",
        "peak_rss_bytes",
        "numpy",
        "python",
        "os",
    }


@_SKIP
def test_full_artifact_hex_metric_ambiguity_is_recorded_not_hidden(
    report: dict[str, Any],
) -> None:
    block = report["analysis"]["hex_metric_ambiguity"]
    rows = {(row["hex_metric"], row["columns"]): row for row in block["rows"]}
    assert rows[("axial_60", 4)]["induced_edges"]["1"] == 99
    assert rows[("axial_120", 4)]["induced_edges"]["1"] == 110
    assert rows[("chebyshev", 4)]["induced_edges"]["1"] == 100
    assert block["largest_relative_edge_deviation_across_metrics"] == pytest.approx(
        11 / 99, abs=1e-9
    )
    assert rows[("chebyshev", 32)]["columns_shared_with_reference"] == 26
    assert rows[("axial_120", 32)]["columns_shared_with_reference"] == 23
