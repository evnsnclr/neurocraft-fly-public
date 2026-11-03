"""Real full-graph artifact checks.

The artifact is gitignored bulk data and absent on CI, so every test here
skips when it is not present. Nothing in this module fabricates artifact
properties: if the bytes are missing, the test does not run.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pytest
from neurocraft_connectome.full_runtime import load_full_runtime_artifact
from neurocraft_neural.full_engine import ArrayNativeEngine
from neurocraft_neural.full_view import (
    ROW_L1_ROUNDING_TOLERANCE,
    view_from_full_runtime_graph,
)
from neurocraft_neural.spmv_kernel import NumpyReduceatKernel, SpmvKernelError, load_c_kernel
from neurocraft_neural.view_verification import VerificationPolicy

from tests.full_backend.helpers import FULL_ARTIFACT_DIRECTORY

pytestmark = pytest.mark.skipif(
    not (FULL_ARTIFACT_DIRECTORY / "manifest.json").is_file(),
    reason="full MaleCNS runtime artifact is not present (bulk data is not tracked)",
)

EXPECTED_NODE_COUNT = 166_700
EXPECTED_EDGE_COUNT = 25_582_938


def _manifest_sha256() -> str:
    return hashlib.sha256((FULL_ARTIFACT_DIRECTORY / "manifest.json").read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def full_view() -> Any:
    graph = load_full_runtime_artifact(
        FULL_ARTIFACT_DIRECTORY, expected_manifest_sha256=_manifest_sha256()
    )
    try:
        return view_from_full_runtime_graph(graph)
    finally:
        graph.close()


def test_view_matches_the_artifact_shape(full_view: Any) -> None:
    assert full_view.node_count == EXPECTED_NODE_COUNT
    assert full_view.edge_count == EXPECTED_EDGE_COUNT
    assert full_view.indptr.dtype == np.int64
    assert full_view.indices.dtype == np.uint32
    assert full_view.weights.dtype == np.float64
    empty_row_count = int(full_view.empty_rows.shape[0])
    assert full_view.buffer_bytes == (
        (EXPECTED_NODE_COUNT + 1) * 8
        + EXPECTED_EDGE_COUNT * 4
        + EXPECTED_EDGE_COUNT * 8
        + empty_row_count * 8
    )
    assert [name for name, _ in full_view.named_buffers()] == [
        "indptr",
        "indices",
        "weights",
        "empty_rows",
    ]
    for _, array in full_view.named_buffers():
        assert array.flags.writeable is False


def test_empty_rows_and_degree_landmarks(full_view: Any) -> None:
    degrees = np.diff(full_view.indptr)
    assert int(degrees.min()) == 0
    assert int(full_view.empty_rows.shape[0]) == int((degrees == 0).sum())
    assert full_view.max_in_degree == int(degrees.max())


def test_row_l1_normalisation_bounds_the_contraction(full_view: Any) -> None:
    """Row L1 is 1.0 up to float64 rounding; that bounds the contraction.

    N11: the measured maximum on this artifact is ``1.0000000000000004``, two
    ulps *above* one, so the module must not claim a bound at exactly 1.0. The
    assertion below is the honest one and the reported value is printed so a
    reader can see the margin.
    """

    measured = full_view.max_row_l1
    assert measured > 1.0 - ROW_L1_ROUNDING_TOLERANCE
    assert measured <= 1.0 + ROW_L1_ROUNDING_TOLERANCE, measured


def test_empty_rows_receive_exactly_zero_recurrent_drive(full_view: Any) -> None:
    try:
        kernel = load_c_kernel(thread_count=4)
    except SpmvKernelError:
        kernel = NumpyReduceatKernel()
    rng = np.random.default_rng(2026)
    x = rng.uniform(-1.0, 1.0, size=full_view.node_count)
    out = np.empty(full_view.node_count, dtype=np.float64)
    kernel.matvec(
        indptr=full_view.indptr,
        indices=full_view.indices,
        values=full_view.weights,
        x=x,
        out=out,
        empty_rows=full_view.empty_rows,
    )
    assert (out[full_view.empty_rows] == 0.0).all()
    assert bool(np.isfinite(out).all())
    assert float(np.abs(out).max()) <= 1.0


def test_engine_steps_and_stays_bounded(full_view: Any) -> None:
    engine = ArrayNativeEngine(
        full_view,
        kernel=NumpyReduceatKernel(),
        verification_policy=VerificationPolicy(mode="load_only"),
    )
    rng = np.random.default_rng(11)
    direct = rng.uniform(-0.2, 0.2, size=full_view.node_count)
    for _ in range(3):
        result = engine.step(direct=direct)
        assert result.record.view_digest_sha256 == full_view.view_sha256
        assert result.record.view_validation_sha256 == full_view.validation.identity_sha256
    engine.end_run()
    assert engine.run_is_covered
    activity = engine.activity()
    assert bool(np.isfinite(activity).all())
    assert float(np.abs(activity).max()) <= 1.0


def test_thread_counts_produce_one_trace(full_view: Any) -> None:
    try:
        base = load_c_kernel(thread_count=1)
    except SpmvKernelError:
        pytest.skip("compiled spmv kernel unavailable")
    rng = np.random.default_rng(13)
    direct = rng.uniform(-0.2, 0.2, size=full_view.node_count)
    traces = set()
    for threads in (1, 4, 12):
        engine = ArrayNativeEngine(
            full_view,
            kernel=base.with_thread_count(threads),
            verification_policy=VerificationPolicy(mode="load_only"),
        )
        for _ in range(5):
            engine.step(direct=direct)
        engine.end_run()
        traces.add(engine.trace_sha256)
    assert len(traces) == 1
