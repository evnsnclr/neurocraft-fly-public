"""Policy-equality and integrity tests for the full-graph sign overlay.

``neurocraft_neural.full_signs`` duplicates the transmitter sets of the pinned
``SignOverlay.coarse_presynaptic_consensus`` because that implementation is
identity-bound and must not be edited. These tests are the pin on the
duplication: the CI-safe suite runs both implementations over one synthetic
graph covering every transmitter class and requires edge-for-edge equality,
and the artifact-gated suite requires the full-graph signs to agree exactly
with the pinned overlay on every edge of the committed steering fixture.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from neurocraft_connectome import (
    RuntimeGraph,
    SyntheticEdge,
    SyntheticNode,
    build_synthetic_graph,
    load_malecns_fixture,
)
from neurocraft_connectome.flight_pin import MALECNS_FLIGHT_FIXTURE_V1_PIN
from neurocraft_neural.full_signs import (
    NEGATIVE_TRANSMITTERS,
    POSITIVE_TRANSMITTERS,
    SIGN_POLICY_ID,
    FullSignOverlayError,
    edge_signs_from_node_signs,
    node_signs_from_transmitters,
    normalized_transmitter,
    transmitter_sign,
)
from neurocraft_neural.model import SignOverlay

REPO_ROOT = Path(__file__).resolve().parents[2]
FULL_ARTIFACT_DIRECTORY = REPO_ROOT / "data" / "processed" / "male-cns-v1.0-superclass-v1"
ANNOTATIONS_PATH = (
    REPO_ROOT / "data" / "cache" / "male-cns-v1.0" / "body-neurotransmitters-male-cns-v1.0.feather"
)
STEERING_FIXTURE = REPO_ROOT / "data" / "fixtures" / "mcns-v1.0-flight-steering-v1.json"
STEERING_FIXTURE_BYTES = 284_949

#: Every transmitter class of the pinned policy, plus the fall-through cases:
#: absent (None), empty string, mixed case, and unmapped labels.
_COVERAGE_TRANSMITTERS: tuple[str | None, ...] = (
    "acetylcholine",
    "dopamine",
    "octopamine",
    "serotonin",
    "tyramine",
    "gaba",
    "glutamate",
    "histamine",
    "GABA",
    "Acetylcholine",
    "GlUtAmAtE",
    "unclear",
    "unknown",
    "",
    None,
)


def _coverage_graph() -> RuntimeGraph:
    """Synthetic graph with one presynaptic node per coverage transmitter."""

    nodes = [
        SyntheticNode(
            body_id=index + 1,
            roles=("intermediate",),
            type_name=f"synthetic-{index + 1}",
            consensus_nt=transmitter,
        )
        for index, transmitter in enumerate(_COVERAGE_TRANSMITTERS)
    ]
    sink_body_id = len(_COVERAGE_TRANSMITTERS) + 1
    nodes.append(SyntheticNode(body_id=sink_body_id, roles=("motor",), type_name="sink"))
    edges = [
        SyntheticEdge(
            body_pre=index + 1,
            body_post=sink_body_id,
            raw_count=index + 1,
            source_row=index,
        )
        for index in range(len(_COVERAGE_TRANSMITTERS))
    ]
    return build_synthetic_graph("synthetic:sign-policy-coverage", nodes, edges)


def test_duplicated_transmitter_sets_match_the_pinned_literals() -> None:
    assert {
        "acetylcholine",
        "dopamine",
        "octopamine",
        "serotonin",
        "tyramine",
    } == POSITIVE_TRANSMITTERS
    assert {"gaba", "glutamate", "histamine"} == NEGATIVE_TRANSMITTERS
    assert not POSITIVE_TRANSMITTERS & NEGATIVE_TRANSMITTERS


def test_normalization_matches_the_pinned_fallback() -> None:
    assert normalized_transmitter(None) == "unknown"
    assert normalized_transmitter("") == "unknown"
    assert normalized_transmitter("GABA") == "gaba"
    assert transmitter_sign("acetylcholine") == 1
    assert transmitter_sign("gaba") == -1
    assert transmitter_sign("unclear") == 0
    assert transmitter_sign("unknown") == 0


def test_policy_equality_against_the_pinned_overlay_on_synthetic_graph() -> None:
    """The array implementation must equal the pinned policy edge for edge."""

    graph = _coverage_graph()
    pinned = SignOverlay.coarse_presynaptic_consensus(graph)
    node_signs = node_signs_from_transmitters([node.consensus_nt for node in graph.nodes])
    edge_pre = np.asarray([edge.pre_index for edge in graph.edges], dtype=np.uint32)
    edge_signs = edge_signs_from_node_signs(node_signs, edge_pre)
    assert edge_signs.tolist() == list(pinned.values)
    # The coverage set really exercised every branch: both polarities and zero.
    assert set(edge_signs.tolist()) == {-1, 0, 1}
    # Provenance cross-check: the pinned policy's normalized transmitter must
    # produce the same sign this module assigns for that transmitter.
    for entry, value in zip(pinned.provenance, pinned.values, strict=True):
        normalized = entry.removeprefix("inferred_from_presynaptic_consensus_nt:")
        assert transmitter_sign(normalized) == value


def test_node_signs_are_int8_read_only() -> None:
    signs = node_signs_from_transmitters(["gaba", None, "serotonin"])
    assert signs.dtype == np.int8
    assert signs.tolist() == [-1, 0, 1]
    assert not signs.flags.writeable


def test_edge_gather_refuses_bad_sign_values() -> None:
    bad = np.asarray([2, 0, -1], dtype=np.int8)
    with pytest.raises(FullSignOverlayError, match="exactly -1, 0, or \\+1"):
        edge_signs_from_node_signs(bad, np.asarray([0, 1], dtype=np.uint32))


def test_edge_gather_refuses_wrong_dtype_and_shape() -> None:
    with pytest.raises(FullSignOverlayError, match="int8"):
        edge_signs_from_node_signs(
            np.asarray([1, 0], dtype=np.int64),
            np.asarray([0], dtype=np.uint32),
        )
    with pytest.raises(FullSignOverlayError, match="one-dimensional"):
        edge_signs_from_node_signs(
            np.zeros((2, 2), dtype=np.int8), np.asarray([0], dtype=np.uint32)
        )
    with pytest.raises(FullSignOverlayError, match="not be empty"):
        edge_signs_from_node_signs(np.asarray([], dtype=np.int8), np.asarray([], dtype=np.uint32))
    with pytest.raises(FullSignOverlayError, match="integer array"):
        edge_signs_from_node_signs(
            np.asarray([1, 0], dtype=np.int8), np.asarray([0.5], dtype=np.float64)
        )


def test_edge_gather_refuses_out_of_range_indices() -> None:
    signs = np.asarray([1, -1], dtype=np.int8)
    with pytest.raises(FullSignOverlayError, match="outside the node sign array"):
        edge_signs_from_node_signs(signs, np.asarray([0, 2], dtype=np.uint32))
    with pytest.raises(FullSignOverlayError, match="outside the node sign array"):
        edge_signs_from_node_signs(signs, np.asarray([-1, 0], dtype=np.int64))


@pytest.mark.skipif(
    not (FULL_ARTIFACT_DIRECTORY / "manifest.json").is_file() or not ANNOTATIONS_PATH.is_file(),
    reason="full MaleCNS runtime artifact or source table absent (bulk data is not tracked)",
)
def test_full_graph_signs_agree_with_pinned_overlay_on_steering_fixture() -> None:
    """Full-graph signs restricted to the committed fixture must agree exactly.

    Edges are matched by ``(pre_body_id, post_body_id)``. The full artifact's
    edges are target-major and pre-sorted within each target, so each fixture
    pair is located with one binary search inside its target's incoming slice.
    """

    from neurocraft_connectome.full_runtime import load_full_runtime_artifact
    from neurocraft_neural.full_signs import compute_full_sign_overlay

    fixture_graph = load_malecns_fixture(
        STEERING_FIXTURE,
        expected_pin=MALECNS_FLIGHT_FIXTURE_V1_PIN,
        expected_file_bytes=STEERING_FIXTURE_BYTES,
    )
    pinned = SignOverlay.coarse_presynaptic_consensus(fixture_graph)

    manifest_sha256 = hashlib.sha256(
        (FULL_ARTIFACT_DIRECTORY / "manifest.json").read_bytes()
    ).hexdigest()
    graph = load_full_runtime_artifact(
        FULL_ARTIFACT_DIRECTORY, expected_manifest_sha256=manifest_sha256
    )
    try:
        result = compute_full_sign_overlay(graph, ANNOTATIONS_PATH)
        assert result.policy_id == SIGN_POLICY_ID
        assert result.edge_signs.size == graph.manifest.edge_count
        # Copies, not aliases: graph.close() refuses while mapping views are live.
        edge_pre = np.frombuffer(graph.edge_pre_indices._mmap, dtype="<u4").copy()
        offsets = np.frombuffer(graph.incoming_offsets._mmap, dtype="<u8").astype(np.int64)
        compared = 0
        disagreements: list[tuple[int, int, int, int]] = []
        for edge, pinned_value in zip(fixture_graph.edges, pinned.values, strict=True):
            pre_dense = graph.index_of(edge.body_pre)
            post_dense = graph.index_of(edge.body_post)
            start, stop = int(offsets[post_dense]), int(offsets[post_dense + 1])
            row = start + int(np.searchsorted(edge_pre[start:stop], np.uint32(pre_dense)))
            assert row < stop and int(edge_pre[row]) == pre_dense, (
                f"fixture edge {edge.body_pre}->{edge.body_post} absent from full graph"
            )
            compared += 1
            full_value = int(result.edge_signs[row])
            if full_value != pinned_value:
                disagreements.append((edge.body_pre, edge.body_post, pinned_value, full_value))
        assert compared == len(fixture_graph.edges)
        assert not disagreements, disagreements
    finally:
        graph.close()
