"""Kernel-level tests: the empty-row trap, thread independence, fail-closed loading."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from neurocraft_connectome.model import sha256_json
from neurocraft_neural.spmv_kernel import (
    BUILD_SIDECAR_KEYS,
    BUILD_SIDECAR_NAME,
    LIBRARY_NAME,
    LIBRARY_PATH_ENVIRONMENT_VARIABLE,
    THREAD_COUNT_ENVIRONMENT_VARIABLE,
    CRowBlockKernel,
    NumpyReduceatKernel,
    SpmvKernel,
    SpmvKernelError,
    c_kernel_identity_material,
    default_thread_count,
    kernel_source_sha256,
    library_search_paths,
    library_sha256,
    load_c_kernel,
    load_kernel,
)

from tests.full_backend.helpers import synthetic_csr

THREAD_SWEEP = (1, 2, 3, 4, 6, 8, 12, 13)


def test_native_library_name_matches_platform() -> None:
    assert ("_spmv.dll" if sys.platform == "win32" else "_spmv.so") == LIBRARY_NAME


def _c_kernel_or_skip(thread_count: int = 4) -> CRowBlockKernel:
    try:
        return load_c_kernel(thread_count=thread_count)
    except SpmvKernelError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"compiled spmv kernel unavailable: {exc}")


def _run(kernel: SpmvKernel, indptr, indices, values, x, empty_rows) -> np.ndarray:
    out = np.empty(indptr.shape[0] - 1, dtype=np.float64)
    kernel.matvec(
        indptr=indptr,
        indices=indices,
        values=values,
        x=x,
        out=out,
        empty_rows=empty_rows,
    )
    return out


def _empty_row_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Row 0 empty, rows 3-5 empty, row 7 (last) empty; every other row loaded.

    Those are the three shapes ``np.add.reduceat`` gets wrong differently:
    a leading zero-length segment, a run of consecutive zero-length segments,
    and a trailing segment whose start offset equals the edge count.
    """

    degrees = [0, 2, 3, 0, 0, 0, 4, 0]
    indptr = np.zeros(len(degrees) + 1, dtype=np.int64)
    np.cumsum(np.asarray(degrees, dtype=np.int64), out=indptr[1:])
    edge_count = int(indptr[-1])
    indices = np.array([1, 2, 0, 1, 6, 0, 2, 4, 6], dtype=np.uint32)
    assert indices.shape[0] == edge_count
    values = np.array([0.5, -0.25, 1.0, -0.75, 0.125, 2.0, -1.5, 0.25, -0.5], dtype=np.float64)
    x = np.array([1.0, -1.0, 0.5, 0.25, -0.75, 0.125, 0.875, -0.375], dtype=np.float64)
    empty_rows = np.flatnonzero(np.diff(indptr) == 0).astype(np.int64)
    return indptr, indices, values, x, empty_rows


def test_in_degree_zero_rows_yield_exactly_zero_recurrent_drive() -> None:
    """MANDATORY: empty rows must be exactly 0.0 in every kernel.

    ``np.add.reduceat`` returns ``operand[start]`` for a zero-length segment,
    which is another edge's product. The real MaleCNS graph has 370 such rows,
    the injected value stays inside ``[-1, 1]``, and no bounds or finiteness
    check would ever fire. The last assertion runs the deliberately broken
    variant to prove this test has the power to catch that regression.
    """

    indptr, indices, values, x, empty_rows = _empty_row_case()
    assert empty_rows.tolist() == [0, 3, 4, 5, 7]

    kernels: list[SpmvKernel] = [NumpyReduceatKernel()]
    with contextlib.suppress(SpmvKernelError):  # the C kernel is optional
        kernels.append(load_c_kernel(thread_count=4))
    for kernel in kernels:
        out = _run(kernel, indptr, indices, values, x, empty_rows)
        for row in empty_rows.tolist():
            assert out[row] == 0.0, f"{kernel.name} row {row} was {out[row]!r}"
            assert np.signbit(out[row]) == np.signbit(0.0)

    # The broken variant: reduceat without the explicit empty-row zeroing.
    products = values * x[indices]
    starts = np.minimum(indptr[:-1], products.shape[0] - 1)
    broken = np.add.reduceat(products, starts)
    corrupted = [row for row in empty_rows.tolist() if broken[row] != 0.0]
    assert corrupted, "the empty-row trap did not reproduce; this test would have no power"


def test_kernels_agree_on_a_graph_with_empty_and_dense_rows() -> None:
    csr = synthetic_csr(
        node_count=512,
        edge_count=20_000,
        seed=11,
        empty_rows=(0, 200, 201, 202, 511),
    )
    weights = np.asarray(
        [
            float(sign * count)
            for sign, count in zip(csr.signs.tolist(), csr.counts.tolist(), strict=True)
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(3)
    x = rng.uniform(-1.0, 1.0, size=csr.node_count)
    empty_rows = np.flatnonzero(np.diff(csr.indptr) == 0).astype(np.int64)
    numpy_out = _run(NumpyReduceatKernel(), csr.indptr, csr.indices, weights, x, empty_rows)
    c_out = _run(_c_kernel_or_skip(), csr.indptr, csr.indices, weights, x, empty_rows)
    assert np.allclose(numpy_out, c_out, rtol=0.0, atol=1e-9 * float(np.abs(weights).max()))
    assert (numpy_out[empty_rows] == 0.0).all()
    assert (c_out[empty_rows] == 0.0).all()


def test_row_block_partition_is_bit_identical_across_thread_counts() -> None:
    """Row blocks never split a row, so thread count cannot change any bit."""

    base = _c_kernel_or_skip(thread_count=1)
    csr = synthetic_csr(
        node_count=4096,
        edge_count=200_000,
        seed=7,
        empty_rows=(0, 100, 101, 102, 4095),
        max_row_override=(2048, 6000),
    )
    weights = (csr.signs * csr.counts).astype(np.float64) / 2591.0
    rng = np.random.default_rng(5)
    x = rng.uniform(-1.0, 1.0, size=csr.node_count)
    empty_rows = np.flatnonzero(np.diff(csr.indptr) == 0).astype(np.int64)
    reference = _run(base, csr.indptr, csr.indices, weights, x, empty_rows)
    for threads in THREAD_SWEEP:
        observed = _run(
            base.with_thread_count(threads), csr.indptr, csr.indices, weights, x, empty_rows
        )
        assert observed.tobytes() == reference.tobytes(), f"thread count {threads} changed bits"


def test_scalar_product_and_sum_are_not_fused() -> None:
    """An FMA would return -2**-54 here; separate operations return +0.0."""
    indptr = np.array([0, 2, 4], dtype=np.int64)
    indices = np.array([0, 1, 0, 1], dtype=np.uint32)
    values = np.array([-1.0, 1.0 + 2.0**-27] * 2, dtype=np.float64)
    x = np.array([1.0, 1.0 - 2.0**-27], dtype=np.float64)
    for count in (1, 2, 8):
        result = _run(
            _c_kernel_or_skip(count), indptr, indices, values, x, np.array([], dtype=np.int64)
        )
        assert result.tobytes() == np.zeros(2, dtype=np.float64).tobytes()


def test_thread_count_is_absent_from_the_kernel_identity() -> None:
    base = _c_kernel_or_skip(thread_count=1)
    twelve = base.with_thread_count(12)
    assert base.identity.identity_sha256 == twelve.identity.identity_sha256
    assert base.identity.material["thread_count_affects_results"] is False


def test_c_and_numpy_kernels_have_distinct_identities() -> None:
    numpy_identity = NumpyReduceatKernel().identity
    assert numpy_identity.numeric_reproducibility_class == "numpy_simd_dispatch_dependent"
    c_kernel = _c_kernel_or_skip()
    assert c_kernel.identity.identity_sha256 != numpy_identity.identity_sha256
    assert c_kernel.identity.material["fp_contract"] == "off"
    assert c_kernel.identity.material["fast_math"] is False
    assert c_kernel.identity.material["kernel_source_sha256"] == kernel_source_sha256()


def test_stale_build_sidecar_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Editing the C source without rebuilding disables the C path, not silently runs it."""

    real = _c_kernel_or_skip()
    staged = tmp_path / real.library_path.name
    staged.write_bytes(real.library_path.read_bytes())
    document = real.build_document
    document["source_sha256"] = "0" * 64
    (tmp_path / BUILD_SIDECAR_NAME).write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr("neurocraft_neural.spmv_kernel.library_search_paths", lambda: (staged,))
    with pytest.raises(SpmvKernelError, match=r"different _spmv\.c"):
        load_c_kernel()


def test_missing_library_degrades_to_numpy_and_records_the_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(LIBRARY_PATH_ENVIRONMENT_VARIABLE, str(tmp_path / "absent.so"))
    monkeypatch.setattr(
        "neurocraft_neural.spmv_kernel.library_search_paths",
        lambda: (tmp_path / "absent.so",),
    )
    kernel = load_kernel()
    assert kernel.name == "numpy_add_reduceat"
    assert "C kernel unavailable" in kernel.identity.load_note
    with pytest.raises(SpmvKernelError):
        load_kernel(require_c=True)


def test_fallback_note_does_not_change_the_hashed_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "neurocraft_neural.spmv_kernel.library_search_paths",
        lambda: (tmp_path / "absent.so",),
    )
    noted = load_kernel()
    assert noted.identity.identity_sha256 == NumpyReduceatKernel().identity.identity_sha256


def test_thread_count_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(THREAD_COUNT_ENVIRONMENT_VARIABLE, "3")
    assert default_thread_count() == 3
    monkeypatch.setenv(THREAD_COUNT_ENVIRONMENT_VARIABLE, "0")
    with pytest.raises(SpmvKernelError):
        default_thread_count()
    monkeypatch.setenv(THREAD_COUNT_ENVIRONMENT_VARIABLE, "not-a-number")
    with pytest.raises(SpmvKernelError):
        default_thread_count()


def test_kernel_rejects_an_invalid_thread_count() -> None:
    real = _c_kernel_or_skip()
    with pytest.raises(SpmvKernelError):
        real.with_thread_count(0)
    with pytest.raises(SpmvKernelError):
        real.with_thread_count(65)


# ---------------------------------------------------------------------------
# B1 -- the executed binary is hashed, and the identity is bound to those bytes
# ---------------------------------------------------------------------------


def test_b1_kernel_identity_binds_the_executed_library_bytes() -> None:
    """B1: ``library_sha256`` is of the file that was ``dlopen``-ed."""

    real = _c_kernel_or_skip()
    document = real.build_document
    assert document["schema"] == "neurocraft.spmv.build.v2"
    assert set(document) == set(BUILD_SIDECAR_KEYS)
    assert document["library_sha256"] == library_sha256(real.library_path)
    assert real.identity.material["library_sha256"] == document["library_sha256"]
    assert document["library_sha256"][:16] in real.identity.load_note


def test_b1_a_library_whose_bytes_do_not_match_the_sidecar_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B1: the reviewer's attack -- honest sidecar, different binary.

    A reversed-summation build and an ``-ffp-contract=fast`` build were each
    placed beside the repository's own unmodified sidecar. Both loaded, both
    produced different trace digests, and both published the exact
    ``kernel_identity`` and ``dynamics_identity`` printed in the receipt,
    because nothing hashed the ``.so``. The refusal below happens before
    ``dlopen``, so the foreign code never runs.
    """

    real = _c_kernel_or_skip()
    honest = real.build_document
    staged = tmp_path / real.library_path.name
    staged.write_bytes(real.library_path.read_bytes() + b"\x00foreign build\x00")
    (tmp_path / BUILD_SIDECAR_NAME).write_text(json.dumps(honest), encoding="utf-8")
    monkeypatch.setattr("neurocraft_neural.spmv_kernel.library_search_paths", lambda: (staged,))

    # The pre-B1 checks all pass on this exact pair: that is the whole point.
    assert honest["source_sha256"] == kernel_source_sha256()
    assert honest["library_filename"] == staged.name
    assert honest["abi_version"] == 2

    with pytest.raises(SpmvKernelError, match="does not describe this binary"):
        load_c_kernel()
    fallback = load_kernel()
    assert fallback.name == "numpy_add_reduceat"
    assert "does not describe this binary" in fallback.identity.load_note


def test_b1_a_different_library_cannot_reuse_the_published_kernel_identity() -> None:
    """B1: a self-consistent hostile sidecar still yields a different identity.

    Recomputing ``library_sha256`` for a foreign binary makes it loadable, and
    that is intended -- a developer must be able to build their own. What must
    be impossible is *attesting* it as this one. The identity material carries
    the library digest, so the hostile build necessarily publishes a different
    ``kernel_identity_sha256`` and therefore a different dynamics identity.
    """

    real = _c_kernel_or_skip()
    honest = real.build_document
    hostile = dict(honest)
    hostile["library_sha256"] = "f" * 64
    honest_material = c_kernel_identity_material(honest)
    hostile_material = c_kernel_identity_material(hostile)
    assert honest_material["library_sha256"] != hostile_material["library_sha256"]
    assert sha256_json(honest_material) != sha256_json(hostile_material)
    assert sha256_json(honest_material) == real.identity.identity_sha256

    # Same for a build that lies about its flags but keeps the honest binary:
    # the flag list is provenance, the digests are the binding.
    flag_lie = dict(honest)
    flag_lie["flags"] = [*honest["flags"], "-ffast-math"]
    assert sha256_json(c_kernel_identity_material(flag_lie)) != real.identity.identity_sha256


def test_b1_a_sidecar_without_the_library_digest_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-B1 (v1) sidecar is not canonical and cannot be used."""

    real = _c_kernel_or_skip()
    staged = tmp_path / real.library_path.name
    staged.write_bytes(real.library_path.read_bytes())
    legacy = real.build_document
    del legacy["library_sha256"]
    legacy["schema"] = "neurocraft.spmv.build.v1"
    legacy["abi_version"] = 1
    (tmp_path / BUILD_SIDECAR_NAME).write_text(json.dumps(legacy), encoding="utf-8")
    monkeypatch.setattr("neurocraft_neural.spmv_kernel.library_search_paths", lambda: (staged,))
    with pytest.raises(SpmvKernelError, match="fields are not canonical"):
        load_c_kernel()


def test_b1_a_sidecar_naming_another_library_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = _c_kernel_or_skip()
    staged = tmp_path / "renamed.so"
    staged.write_bytes(real.library_path.read_bytes())
    document = real.build_document
    document["library_sha256"] = library_sha256(staged)
    (tmp_path / BUILD_SIDECAR_NAME).write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr("neurocraft_neural.spmv_kernel.library_search_paths", lambda: (staged,))
    with pytest.raises(SpmvKernelError, match="names library"):
        load_c_kernel()


def test_b1_environment_override_is_still_subject_to_both_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``NEUROCRAFT_SPMV_LIBRARY`` accepts any path, but not any bytes."""

    real = _c_kernel_or_skip()
    staged = tmp_path / real.library_path.name
    staged.write_bytes(real.library_path.read_bytes() + b"\x01")
    (tmp_path / BUILD_SIDECAR_NAME).write_text(json.dumps(real.build_document), encoding="utf-8")
    monkeypatch.setenv(LIBRARY_PATH_ENVIRONMENT_VARIABLE, str(staged))
    assert library_search_paths()[0] == staged, "the override is not highest precedence"
    # The override is refused and the loader falls through to the tracked
    # library rather than executing the tampered one.
    assert load_c_kernel().library_path != staged
    monkeypatch.setattr("neurocraft_neural.spmv_kernel.library_search_paths", lambda: (staged,))
    with pytest.raises(SpmvKernelError, match="does not describe this binary"):
        load_c_kernel()


# ---------------------------------------------------------------------------
# B4 -- index range violations are contained, not undefined behaviour
# ---------------------------------------------------------------------------


def test_b4_both_kernels_refuse_an_out_of_range_column_index() -> None:
    indptr, indices, values, x, empty_rows = _empty_row_case()
    corrupted = indices.copy()
    corrupted[0] = 0xFFFFFFF0
    kernels: list[SpmvKernel] = [NumpyReduceatKernel()]
    with contextlib.suppress(SpmvKernelError):
        kernels.append(load_c_kernel(thread_count=4))
    for kernel in kernels:
        with pytest.raises(SpmvKernelError, match="out of range"):
            _run(kernel, indptr, corrupted, values, x, empty_rows)


def test_b4_the_c_kernel_refuses_corrupted_csr_offsets() -> None:
    kernel = _c_kernel_or_skip()
    indptr, indices, values, x, empty_rows = _empty_row_case()
    broken = indptr.copy()
    broken[2] = broken[-1] + 1_000
    with pytest.raises(SpmvKernelError, match=r"status -3|out of range"):
        _run(kernel, broken, indices, values, x, empty_rows)


def test_b4_bounds_checking_does_not_change_any_bit() -> None:
    """The range tests are integer comparisons; the reduction is untouched."""

    base = _c_kernel_or_skip(thread_count=1)
    csr = synthetic_csr(node_count=1024, edge_count=40_000, seed=41, empty_rows=(0, 1023))
    weights = (csr.signs * csr.counts).astype(np.float64) / 2591.0
    rng = np.random.default_rng(19)
    x = rng.uniform(-1.0, 1.0, size=csr.node_count)
    empty_rows = np.flatnonzero(np.diff(csr.indptr) == 0).astype(np.int64)
    reference = _run(NumpyReduceatKernel(), csr.indptr, csr.indices, weights, x, empty_rows)
    observed = _run(base, csr.indptr, csr.indices, weights, x, empty_rows)
    assert np.allclose(observed, reference, rtol=0.0, atol=1e-9 * float(np.abs(weights).max()))
    for threads in (2, 4, 8):
        assert (
            _run(
                base.with_thread_count(threads), csr.indptr, csr.indices, weights, x, empty_rows
            ).tobytes()
            == observed.tobytes()
        )
