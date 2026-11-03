"""View construction: bit-exact weights, frozen buffers, refused inputs."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from neurocraft_connectome import (
    RuntimeGraph,
    SyntheticEdge,
    SyntheticNode,
    build_synthetic_graph,
    load_malecns_fixture,
)
from neurocraft_neural.engine import NeuralEngine
from neurocraft_neural.full_view import (
    BUFFER_ORDER,
    EXACT_INTEGER_LIMIT,
    ROW_L1_ROUNDING_TOLERANCE,
    ViewProvenance,
    ViewValidation,
    build_view,
    view_construction_sha256,
    view_from_runtime_graph,
)
from neurocraft_neural.model import (
    DelayOverlay,
    NeuralConfigurationError,
    SignOverlay,
    TopologyOverlay,
)

from tests.full_backend.helpers import FIXTURE_PATH, effective_weights, synthetic_csr


def _provenance() -> ViewProvenance:
    return ViewProvenance(origin="synthetic_csr", graph_identity_sha256="a" * 64)


def _view_from(csr) -> object:
    return build_view(
        indptr=csr.indptr,
        pre_indices=csr.indices,
        raw_counts=csr.counts,
        signs=csr.signs,
        provenance=_provenance(),
    )


@pytest.fixture
def fixture_graph() -> RuntimeGraph:
    return load_malecns_fixture(FIXTURE_PATH)


def test_effective_weights_are_bit_identical_to_the_reference(fixture_graph: RuntimeGraph) -> None:
    """Not ``allclose`` -- ``==``. Any difference here is a bug, not rounding."""

    signs = SignOverlay.coarse_presynaptic_consensus(fixture_graph)
    reference = NeuralEngine(fixture_graph, sign_overlay=signs)
    view = view_from_runtime_graph(fixture_graph, sign_overlay=signs)
    assert tuple(view.weights.tolist()) == reference.effective_weights
    assert view.max_row_l1 <= 1.0 + ROW_L1_ROUNDING_TOLERANCE


def test_effective_weights_match_a_pure_python_derivation() -> None:
    csr = synthetic_csr(
        node_count=256,
        edge_count=9_000,
        seed=17,
        empty_rows=(0, 40, 41, 42, 255),
    )
    view = _view_from(csr)
    assert view.weights.tolist() == effective_weights(csr)
    assert view.empty_rows.tolist() == np.flatnonzero(np.diff(csr.indptr) == 0).tolist()


def test_single_incoming_edge_rows_are_exact() -> None:
    """An in-degree-1 row's sum is one product, so it must be bit-exact."""

    indptr = np.array([0, 1, 2, 3], dtype=np.int64)
    view = build_view(
        indptr=indptr,
        pre_indices=np.array([0, 1, 2], dtype=np.uint32),
        raw_counts=np.array([7, 13, 2591], dtype=np.int64),
        signs=np.array([1, -1, 1], dtype=np.int64),
        provenance=_provenance(),
    )
    assert view.weights.tolist() == [7 / 7, -13 / 13, 2591 / 2591]


def test_buffers_are_frozen_and_digest_covers_them() -> None:
    csr = synthetic_csr(node_count=64, edge_count=500, seed=3)
    view = _view_from(csr)
    for _, array in view.named_buffers():
        assert array.flags.writeable is False
    twin = _view_from(csr)
    assert twin.view_sha256 == view.view_sha256
    assert len(view.view_sha256) == 64


def test_b3_empty_rows_is_a_hashed_pinned_buffer() -> None:
    """B3: ``empty_rows`` is dereferenced by the numpy kernel, so it is hashed.

    The reviewer set ``empty_rows[0]`` to a *non-empty* row, which zeroed that
    row's entire recurrent drive (max state divergence 2.18e-01, three orders
    above the adopted 1e-13 tolerance) with no fault, because the buffer was in
    neither the view digest nor the pin set. Both now cover it.
    """

    csr = synthetic_csr(node_count=64, edge_count=500, seed=3, empty_rows=(0, 5, 6, 63))
    view = _view_from(csr)
    assert [name for name, _ in view.named_buffers()] == list(BUFFER_ORDER)
    assert BUFFER_ORDER[-1] == "empty_rows"
    assert view.buffer_bytes == sum(array.nbytes for _, array in view.named_buffers())
    assert view.empty_rows.nbytes > 0
    assert view.empty_rows.flags.writeable is False

    three = hashlib.sha256()
    for _, array in view.named_buffers()[:3]:
        with memoryview(array).cast("B") as block:
            three.update(block)
    assert view.view_sha256 != three.hexdigest(), "empty_rows is not inside the digest"


def test_b4_view_records_the_validated_index_bound() -> None:
    """B4: the ``max(indices) < node_count`` check becomes hashed evidence."""

    csr = synthetic_csr(node_count=64, edge_count=500, seed=3, empty_rows=(0, 63))
    view = _view_from(csr)
    assert view.validation == ViewValidation.derive(
        indptr=view.indptr, indices=view.indices, empty_rows=view.empty_rows
    )
    assert view.validation.max_observed_index == int(view.indices.max())
    assert view.validation.max_observed_index < view.validation.index_upper_bound_exclusive
    assert view.validation.empty_row_count == int(view.empty_rows.shape[0])
    assert len(view.validation.identity_sha256) == 64

    empty = build_view(
        indptr=np.zeros(3, dtype=np.int64),
        pre_indices=np.zeros(0, dtype=np.uint32),
        raw_counts=np.zeros(0, dtype=np.int64),
        signs=np.zeros(0, dtype=np.int64),
        provenance=_provenance(),
    )
    assert empty.validation.max_observed_index == -1


def test_n11_row_l1_is_one_up_to_rounding_not_bounded_at_one() -> None:
    """N11: a correctly derived row can have L1 strictly greater than 1.0.

    Each weight is one correctly rounded division and the row sum is one more
    rounding, so the realised L1 can exceed one by ulps. The real artifact
    measures 1.0000000000000004; this seven-edge row reproduces the same effect
    at fixture scale, which is what makes the corrected wording testable rather
    than a matter of opinion.
    """

    counts = [803, 2894, 2641, 559, 1529, 2819, 2541]
    view = build_view(
        indptr=np.array([0, len(counts)], dtype=np.int64),
        pre_indices=np.zeros(len(counts), dtype=np.uint32),
        raw_counts=np.array(counts, dtype=np.int64),
        signs=np.ones(len(counts), dtype=np.int64),
        provenance=_provenance(),
    )
    assert view.max_row_l1 > 1.0, "this row no longer reproduces the rounding excess"
    assert view.max_row_l1 <= 1.0 + ROW_L1_ROUNDING_TOLERANCE
    assert sum(counts) == 13_786


def test_view_construction_digest_is_stable_within_a_process() -> None:
    assert view_construction_sha256() == view_construction_sha256()
    assert len(view_construction_sha256()) == 64


def test_leading_trailing_and_consecutive_empty_rows_are_all_detected() -> None:
    degrees = [0, 3, 0, 0, 0, 2, 0]
    indptr = np.zeros(len(degrees) + 1, dtype=np.int64)
    np.cumsum(np.asarray(degrees, dtype=np.int64), out=indptr[1:])
    edges = int(indptr[-1])
    view = build_view(
        indptr=indptr,
        pre_indices=np.arange(edges, dtype=np.uint32) % len(degrees),
        raw_counts=np.full(edges, 5, dtype=np.int64),
        signs=np.ones(edges, dtype=np.int64),
        provenance=_provenance(),
    )
    assert view.empty_rows.tolist() == [0, 2, 3, 4, 6]
    # The trailing empty row must not have stolen the previous row's last edge.
    assert view.weights.tolist() == [5 / 15, 5 / 15, 5 / 15, 5 / 10, 5 / 10]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("indptr_endpoint", "indptr endpoints"),
        ("indptr_decreasing", "non-decreasing"),
        ("index_out_of_range", "out of range"),
        ("bad_sign", "exactly -1, 0, or"),
        ("zero_count", "positive integers"),
        ("length_mismatch", "one entry per edge"),
    ],
)
def test_invalid_inputs_are_refused(mutation: str, message: str) -> None:
    indptr = np.array([0, 2, 4], dtype=np.int64)
    indices = np.array([0, 1, 0, 1], dtype=np.uint32)
    counts = np.array([1, 2, 3, 4], dtype=np.int64)
    signs = np.array([1, -1, 1, -1], dtype=np.int64)
    if mutation == "indptr_endpoint":
        indptr = np.array([0, 2, 3], dtype=np.int64)
    elif mutation == "indptr_decreasing":
        indptr = np.array([0, 4, 2, 4], dtype=np.int64)
    elif mutation == "index_out_of_range":
        indices = np.array([0, 1, 0, 9], dtype=np.uint32)
    elif mutation == "bad_sign":
        signs = np.array([1, -1, 2, -1], dtype=np.int64)
    elif mutation == "zero_count":
        counts = np.array([1, 0, 3, 4], dtype=np.int64)
    elif mutation == "length_mismatch":
        counts = np.array([1, 2, 3], dtype=np.int64)
    with pytest.raises(NeuralConfigurationError, match=message):
        build_view(
            indptr=indptr,
            pre_indices=indices,
            raw_counts=counts,
            signs=signs,
            provenance=_provenance(),
        )


def test_count_total_beyond_exact_float64_range_is_refused() -> None:
    with pytest.raises(NeuralConfigurationError, match="exact float64 integer range"):
        build_view(
            indptr=np.array([0, 2], dtype=np.int64),
            pre_indices=np.array([0, 0], dtype=np.uint32),
            raw_counts=np.array([EXACT_INTEGER_LIMIT - 1, 2], dtype=np.int64),
            signs=np.array([1, 1], dtype=np.int64),
            provenance=_provenance(),
        )


def _chain_graph() -> RuntimeGraph:
    return build_synthetic_graph(
        "synthetic:view-chain",
        [SyntheticNode(index, ("interneuron",)) for index in range(1, 5)],
        [SyntheticEdge(1, 2, 5, 0), SyntheticEdge(2, 3, 7, 1), SyntheticEdge(3, 4, 11, 2)],
    )


def test_non_uniform_delay_is_refused() -> None:
    graph = _chain_graph()
    signs = SignOverlay.coarse_presynaptic_consensus(graph)
    delays = DelayOverlay.explicit(graph, [1, 2, 1], provenance="mixed")
    with pytest.raises(NeuralConfigurationError, match="uniform delay 1 only"):
        view_from_runtime_graph(graph, sign_overlay=signs, delay_overlay=delays)


def test_non_csr_topology_is_refused() -> None:
    graph = _chain_graph()
    signs = SignOverlay.coarse_presynaptic_consensus(graph)
    canonical = TopologyOverlay.canonical(graph)
    scrambled = TopologyOverlay(
        graph_support_sha256=canonical.graph_support_sha256,
        graph_raw_counts_sha256=canonical.graph_raw_counts_sha256,
        kind=canonical.kind,
        pre_indices=canonical.pre_indices,
        post_indices=tuple(reversed(canonical.post_indices)),
        seed=canonical.seed,
        requested_swaps=canonical.requested_swaps,
        attempted_swaps=canonical.attempted_swaps,
        accepted_swaps=canonical.accepted_swaps,
        provenance=canonical.provenance,
        identity_sha256=canonical.identity_sha256,
    )
    with pytest.raises(Exception, match=r"target-major|overlay"):
        view_from_runtime_graph(graph, sign_overlay=signs, topology_overlay=scrambled)
