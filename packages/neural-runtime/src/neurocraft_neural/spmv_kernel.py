"""Sparse CSR matrix/vector kernels for the array-native full-graph backend.

Two interchangeable implementations of one operation are provided:

``CRowBlockKernel``
    A fused C99 CSR kernel (:mod:`neurocraft_neural._spmv`, compiled to a
    gitignored shared library by ``make spmv-build``) loaded through
    :mod:`ctypes` from the standard library. No third-party binding layer is
    involved. Rows are partitioned into contiguous nonzero-balanced blocks and
    each row is summed left to right by exactly one thread, so results are
    bit-identical across thread counts.

``NumpyReduceatKernel``
    A pure-numpy three-pass fallback used when the shared library is absent or
    refuses to load. It is *numerically distinct* from the C kernel -- see
    ``NUMERIC_REPRODUCIBILITY`` below -- and therefore carries a different
    kernel identity, which propagates into a different dynamics identity.

Degradation is explicit, never silent. :func:`load_kernel` records which path
was taken and why in :attr:`SpmvKernel.identity`, and the reason string is
carried into every receipt. Passing ``require_c=True`` turns a failed C load
into an exception instead of a fallback.

Fail-closed build binding
-------------------------
``make spmv-build`` writes ``_spmv.build.json`` beside the shared library,
recording the compiler identity, the exact normalised flag list, the target
triple, the SHA-256 of ``_spmv.c`` **and the SHA-256 of the emitted shared
library itself**. The loader refuses the shared library unless

1. the sidecar exists and is canonical;
2. its ``source_sha256`` equals the SHA-256 of the tracked ``_spmv.c``
   currently on disk (so the binary is bound to reviewed source);
3. its ``library_filename`` equals the name of the library being loaded;
4. its ``library_sha256`` equals the SHA-256 of *the bytes about to be
   ``dlopen``-ed*; and
5. the library's own ``neurocraft_spmv_abi_version()`` matches
   :data:`SPMV_ABI_VERSION`.

Check 4 is what makes the recorded identity an identity of the executed
machine code rather than of a text file that merely sits nearby. Without it,
an honest sidecar could be copied beside a differently-compiled library --
reversed row summation, ``-ffp-contract=fast``, anything -- and that library
would run while publishing the honest ``kernel_identity_sha256``. The library
digest is therefore also hashed into :class:`KernelIdentity`, so any other
binary necessarily carries a *different* kernel identity and a different
dynamics identity; it can be executed, but it can never be attested as this
one. Editing the C source, or swapping the binary, without rebuilding disables
the C path rather than silently running foreign object code.

What this does **not** claim: the sidecar's ``flags``, ``compiler_id`` and
``target_triple`` are self-reported build metadata and are not re-derived from
the binary. They are provenance. The load-bearing bindings are the two
digests.

Range preconditions are checked, not assumed
--------------------------------------------
Both kernels range-check every CSR offset and every column index they
dereference and refuse the call rather than reading out of bounds. This is not
belt-and-braces: the view's construction-time validation happens once, while
the amortised integrity checkpoint that is supposed to catch later corruption
runs *after* the kernel has already read the buffers for that step. A single
out-of-range index written after construction therefore reached the gather
first, which was undefined behaviour and was observed as SIGSEGV. The numpy
kernel gets this from numpy's own bounds checking, normalised to
:class:`SpmvKernelError`; the C kernel does it explicitly per row and per
element. The checks are integer comparisons on values that never enter the
floating-point reduction, so no stored bit can change.

Numeric reproducibility
-----------------------
The C kernel sums each row with a single scalar ``double`` accumulator in
stored column order, compiled with ``-ffp-contract=off`` and without
fast-math, so its output is fixed by the IEEE-754 definitions of ``*`` and
``+``. ``numpy.add.reduceat`` uses its own SIMD-blocked accumulation whose
kernel is selected at runtime from CPU features; it is neither strictly
sequential nor numpy's pairwise ``sum``, and its cross-architecture bit
stability is unverified. The two kernels therefore agree only to a few ulps,
never bit for bit.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from neurocraft_connectome.model import sha256_json
from numpy.typing import NDArray

#: Bumped to 2 when the kernel gained per-element index bounds checking, so a
#: stale pre-check library can never satisfy a current sidecar.
SPMV_ABI_VERSION: Final = 2
KERNEL_SOURCE_NAME: Final = "_spmv.c"
LIBRARY_NAME: Final = "_spmv.dll" if sys.platform == "win32" else "_spmv.so"
BUILD_SIDECAR_NAME: Final = "_spmv.build.json"
BUILD_SIDECAR_SCHEMA: Final = "neurocraft.spmv.build.v2"

#: Exactly the sidecar fields the loader accepts. ``library_sha256`` is the
#: field that binds the document to the executed machine code.
BUILD_SIDECAR_KEYS: Final[frozenset[str]] = frozenset(
    {
        "abi_version",
        "compiler_id",
        "compiler_version",
        "flags",
        "library_filename",
        "library_sha256",
        "schema",
        "source_filename",
        "source_sha256",
        "target_triple",
    }
)

#: Status returned by the C kernel when an index fails its range precondition.
SPMV_STATUS_EDOMAIN: Final = -3
LIBRARY_PATH_ENVIRONMENT_VARIABLE: Final = "NEUROCRAFT_SPMV_LIBRARY"
THREAD_COUNT_ENVIRONMENT_VARIABLE: Final = "NEUROCRAFT_SPMV_THREADS"
MAX_THREADS: Final = 64
DEFAULT_THREAD_CAP: Final = 8

#: Canonical compiler flags. ``-ffp-contract=off`` forbids the FMA
#: contraction clang enables by default for C99 and later; ``-O3`` and
#: ``-march=native`` are performance-only and cannot reorder a floating-point
#: reduction in the absence of fast-math.
BUILD_FLAGS: Final[tuple[str, ...]] = (
    "-O3",
    "-std=c99",
    "-march=native",
    "-ffp-contract=off",
    "-fPIC",
    "-shared",
    "-pthread",
    "-Wall",
    "-Wextra",
)

#: Native Windows builds use OS threads and the cdecl DLL ABI. The MSVC
#: drivers require their own floating-point spelling; /fp:strict forbids FMA
#: contraction and reassociation. No /arch:AVX* flag is used for release builds.
WINDOWS_CLANG_BUILD_FLAGS: Final[tuple[str, ...]] = (
    "-O3",
    "-std=c99",
    "-ffp-contract=off",
    "-shared",
    "-Wall",
    "-Wextra",
)
WINDOWS_MSVC_BUILD_FLAGS: Final[tuple[str, ...]] = (
    "/nologo",
    "/O2",
    "/std:c11",
    "/fp:strict",
    "/LD",
    "/W4",
)

_MODULE_DIRECTORY: Final = Path(__file__).resolve().parent


class SpmvKernelError(RuntimeError):
    """A sparse kernel could not be loaded or refused an argument."""


@dataclass(frozen=True, slots=True)
class KernelIdentity:
    """Hashable description of exactly which arithmetic produced a result."""

    kernel: str
    thread_count: int
    numeric_reproducibility_class: str
    load_path: str
    load_note: str
    material: dict[str, Any]

    @property
    def identity_sha256(self) -> str:
        return sha256_json(self.material)


def kernel_source_path() -> Path:
    return _MODULE_DIRECTORY / KERNEL_SOURCE_NAME


def kernel_source_sha256() -> str:
    """SHA-256 of the *tracked* C source, byte-bound into the backend identity.

    Deliberately the repository's ``_spmv.c`` and never a source file sitting
    beside an arbitrary library: the point is to bind the loaded binary to
    reviewed source. The binary itself is bound separately by
    :func:`library_sha256`.
    """

    return hashlib.sha256(kernel_source_path().read_bytes()).hexdigest()


def library_sha256(library_path: Path) -> str:
    """SHA-256 of the shared-library bytes that are about to be executed."""

    return hashlib.sha256(library_path.read_bytes()).hexdigest()


def c_kernel_identity_material(document: dict[str, Any]) -> dict[str, Any]:
    """Hashed material describing exactly which compiled kernel ran.

    Exposed as a free function so a test can compare the material of two build
    documents without dlopening either library.
    """

    return {
        "compiler_id": document["compiler_id"],
        "compiler_version": document["compiler_version"],
        "empty_row_policy": "fused_accumulator_starts_at_zero",
        "fast_math": False,
        "fp_contract": "off",
        "index_bounds_checked": True,
        "kernel": CRowBlockKernel.name,
        "kernel_build_sha256": sha256_json(document),
        "kernel_source_sha256": document["source_sha256"],
        "library_sha256": document["library_sha256"],
        "numeric_reproducibility_class": "ieee754_binary64_scalar_sequential",
        "row_partition": "contiguous_row_blocks_balanced_by_nnz",
        "summation_order": "csr_row_sequential_stored_column_order",
        "target_triple": document["target_triple"],
        "thread_count_affects_results": False,
    }


def default_thread_count() -> int:
    """Thread count from the environment, else a conservative CPU-derived cap."""

    raw = os.environ.get(THREAD_COUNT_ENVIRONMENT_VARIABLE)
    if raw is not None:
        try:
            requested = int(raw)
        except ValueError as exc:
            raise SpmvKernelError(
                f"{THREAD_COUNT_ENVIRONMENT_VARIABLE} must be an integer, got {raw!r}"
            ) from exc
        if not 1 <= requested <= MAX_THREADS:
            raise SpmvKernelError(f"thread count must lie in [1, {MAX_THREADS}]")
        return requested
    return max(1, min(DEFAULT_THREAD_CAP, os.cpu_count() or 1))


class SpmvKernel:
    """Common interface: ``out[r] = sum_k values[k] * x[indices[k]]`` per row."""

    name: str = "abstract"

    @property
    def thread_count(self) -> int:  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def identity(self) -> KernelIdentity:  # pragma: no cover - overridden
        raise NotImplementedError

    def matvec(
        self,
        *,
        indptr: NDArray[np.int64],
        indices: NDArray[np.uint32],
        values: NDArray[np.float64],
        x: NDArray[np.float64],
        out: NDArray[np.float64],
        empty_rows: NDArray[np.int64],
    ) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


class NumpyReduceatKernel(SpmvKernel):
    """Three-pass gather/multiply/segmented-reduce fallback.

    Two distinct traps live here. ``np.add.reduceat`` returns
    ``operand[start]`` -- a foreign edge's product, not zero -- for a
    zero-length segment; the real MaleCNS graph has 370 in-degree-zero rows,
    and such a corruption keeps the state inside ``[-1, 1]``, so no bounds
    check would ever notice. Zeroing ``empty_rows`` afterwards is therefore
    mandatory. Separately, a *trailing* empty row makes the last start offset
    equal the edge count, which ``reduceat`` rejects outright; the product
    buffer is allocated one element longer and that element is held at 0.0, so
    every start index is in range and the extra term is additively neutral.
    Clipping the start offsets instead would silently truncate the last
    non-empty row's segment.
    """

    name = "numpy_add_reduceat"

    def __init__(self) -> None:
        self._products: NDArray[np.float64] | None = None

    @property
    def thread_count(self) -> int:
        return 1

    @property
    def identity(self) -> KernelIdentity:
        material: dict[str, Any] = {
            "empty_row_policy": "explicit_zero_after_reduceat",
            "index_bounds_checked": True,
            "kernel": self.name,
            "numeric_reproducibility_class": "numpy_simd_dispatch_dependent",
            "numpy_version": np.__version__,
            "summation_order": "numpy_add_reduceat_internal_blocking",
            "thread_count_affects_results": False,
        }
        return KernelIdentity(
            kernel=self.name,
            thread_count=1,
            numeric_reproducibility_class="numpy_simd_dispatch_dependent",
            load_path="numpy",
            load_note="pure-numpy fallback kernel",
            material=material,
        )

    def matvec(
        self,
        *,
        indptr: NDArray[np.int64],
        indices: NDArray[np.uint32],
        values: NDArray[np.float64],
        x: NDArray[np.float64],
        out: NDArray[np.float64],
        empty_rows: NDArray[np.int64],
    ) -> None:
        edge_count = int(values.shape[0])
        if edge_count == 0:
            out[:] = 0.0
            return
        products = self._products
        if products is None or products.shape[0] != edge_count + 1:
            products = np.zeros(edge_count + 1, dtype=np.float64)
            self._products = products
        head = products[:edge_count]
        # Every indexing operation below is bounds-checked by numpy
        # (``np.take`` defaults to mode="raise"), so a buffer corrupted after
        # view construction is contained here rather than read out of bounds.
        # The exception type is normalised to ``SpmvKernelError`` so both
        # kernels present the same contract; the C kernel range-checks the same
        # three preconditions per row and per element.
        try:
            np.take(x, indices, out=head)
            np.multiply(head, values, out=head)
            products[edge_count] = 0.0
            np.add.reduceat(products, indptr[:-1], out=out)
            out[empty_rows] = 0.0
        except IndexError as exc:
            raise SpmvKernelError(
                f"CSR offset, presynaptic index or empty-row index out of range: {exc}"
            ) from exc


class CRowBlockKernel(SpmvKernel):
    """Fused CSR kernel loaded via ctypes, with POSIX or native Win32 threads."""

    name = "c99_rowblock_sequential"

    def __init__(
        self,
        library_path: Path,
        build_document: dict[str, Any],
        *,
        thread_count: int,
    ) -> None:
        if not 1 <= thread_count <= MAX_THREADS:
            raise SpmvKernelError(f"thread count must lie in [1, {MAX_THREADS}]")
        self._library_path = library_path
        self._build_document = build_document
        self._thread_count = thread_count
        library = ctypes.CDLL(str(library_path))
        abi = library.neurocraft_spmv_abi_version
        abi.restype = ctypes.c_int32
        abi.argtypes = []
        observed_abi = int(abi())
        if observed_abi != SPMV_ABI_VERSION:
            raise SpmvKernelError(
                f"spmv library ABI {observed_abi} does not match expected {SPMV_ABI_VERSION}"
            )
        entry = library.neurocraft_spmv_csr_f64
        entry.restype = ctypes.c_int32
        entry.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int32,
        ]
        self._library = library
        self._entry = entry

    @property
    def library_path(self) -> Path:
        return self._library_path

    @property
    def build_document(self) -> dict[str, Any]:
        return dict(self._build_document)

    @property
    def thread_count(self) -> int:
        return self._thread_count

    def with_thread_count(self, thread_count: int) -> CRowBlockKernel:
        """Return an equivalent kernel bound to a different thread count."""

        return CRowBlockKernel(
            self._library_path,
            self._build_document,
            thread_count=thread_count,
        )

    @property
    def identity(self) -> KernelIdentity:
        material = c_kernel_identity_material(self._build_document)
        return KernelIdentity(
            kernel=self.name,
            thread_count=self._thread_count,
            numeric_reproducibility_class="ieee754_binary64_scalar_sequential",
            load_path="ctypes",
            load_note=(
                f"loaded {self._library_path.name} with a sidecar-verified "
                f"library digest {self._build_document['library_sha256'][:16]}"
            ),
            material=material,
        )

    def matvec(
        self,
        *,
        indptr: NDArray[np.int64],
        indices: NDArray[np.uint32],
        values: NDArray[np.float64],
        x: NDArray[np.float64],
        out: NDArray[np.float64],
        empty_rows: NDArray[np.int64],
    ) -> None:
        del empty_rows  # the fused accumulator yields exactly 0.0 for empty rows
        status = int(
            self._entry(
                indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int64(int(out.shape[0])),
                ctypes.c_int64(int(x.shape[0])),
                ctypes.c_int64(int(values.shape[0])),
                ctypes.c_int32(self._thread_count),
            )
        )
        if status == SPMV_STATUS_EDOMAIN:
            raise SpmvKernelError(
                "spmv kernel refused the call: a presynaptic index is out of range "
                "(the CSR index buffer was corrupted after view construction)"
            )
        if status != 0:
            raise SpmvKernelError(f"spmv kernel refused the call with status {status}")


def library_search_paths() -> tuple[Path, ...]:
    """Candidate shared-library locations, highest precedence first."""

    override = os.environ.get(LIBRARY_PATH_ENVIRONMENT_VARIABLE)
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    candidates.append(_MODULE_DIRECTORY / LIBRARY_NAME)
    return tuple(candidates)


def _read_build_document(library_path: Path) -> dict[str, Any]:
    sidecar = library_path.with_name(BUILD_SIDECAR_NAME)
    if not sidecar.is_file():
        raise SpmvKernelError(f"missing build sidecar {sidecar}")
    try:
        document = json.loads(sidecar.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpmvKernelError(f"build sidecar {sidecar} is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise SpmvKernelError(f"build sidecar {sidecar} must be a JSON object")
    if set(document) != BUILD_SIDECAR_KEYS:
        raise SpmvKernelError(f"build sidecar {sidecar} fields are not canonical")
    if document["schema"] != BUILD_SIDECAR_SCHEMA:
        raise SpmvKernelError(f"build sidecar {sidecar} has an unsupported schema")
    if document["abi_version"] != SPMV_ABI_VERSION:
        raise SpmvKernelError(f"build sidecar {sidecar} records a different ABI version")
    observed_source = kernel_source_sha256()
    if document["source_sha256"] != observed_source:
        raise SpmvKernelError(
            f"build sidecar {sidecar} was produced from a different {KERNEL_SOURCE_NAME}; "
            "rebuild with `make spmv-build`"
        )
    if document["library_filename"] != library_path.name:
        raise SpmvKernelError(
            f"build sidecar {sidecar} names library {document['library_filename']!r}, "
            f"not {library_path.name!r}"
        )
    observed_library = library_sha256(library_path)
    if document["library_sha256"] != observed_library:
        raise SpmvKernelError(
            f"build sidecar {sidecar} records library digest "
            f"{document['library_sha256']} but {library_path} hashes to "
            f"{observed_library}; the sidecar does not describe this binary. "
            "Rebuild with `make spmv-build`"
        )
    typed: dict[str, Any] = document
    return typed


def load_c_kernel(*, thread_count: int | None = None) -> CRowBlockKernel:
    """Load the compiled kernel or raise with the exact reason it was refused."""

    resolved_threads = default_thread_count() if thread_count is None else thread_count
    reasons: list[str] = []
    for candidate in library_search_paths():
        if not candidate.is_file():
            reasons.append(f"{candidate}: not present")
            continue
        try:
            document = _read_build_document(candidate)
            return CRowBlockKernel(candidate, document, thread_count=resolved_threads)
        except (SpmvKernelError, OSError) as exc:
            reasons.append(f"{candidate}: {exc}")
    raise SpmvKernelError("no usable spmv shared library; " + "; ".join(reasons))


def load_kernel(
    *,
    prefer_c: bool = True,
    require_c: bool = False,
    thread_count: int | None = None,
) -> SpmvKernel:
    """Return the best available kernel, recording which path was taken.

    With ``require_c`` set, a missing or refused shared library raises
    :class:`SpmvKernelError`. Otherwise the numpy fallback is returned and the
    refusal reason is preserved in the kernel identity's ``load_note``.
    """

    if not prefer_c and require_c:
        raise SpmvKernelError("require_c and prefer_c=False are contradictory")
    if not prefer_c:
        return NumpyReduceatKernel()
    try:
        return load_c_kernel(thread_count=thread_count)
    except SpmvKernelError as exc:
        if require_c:
            raise
        fallback = NumpyReduceatKernel()
        fallback_identity = fallback.identity
        return _NotedNumpyKernel(f"C kernel unavailable ({exc}); {fallback_identity.load_note}")


class _NotedNumpyKernel(NumpyReduceatKernel):
    """Numpy fallback that remembers why the C kernel was not used.

    The note is deliberately outside the hashed identity material: *why* the
    fallback ran is provenance for the receipt, while *what arithmetic ran* is
    identical to any other numpy fallback and must hash the same.
    """

    def __init__(self, note: str) -> None:
        super().__init__()
        self._note = note

    @property
    def identity(self) -> KernelIdentity:
        base = super().identity
        return KernelIdentity(
            kernel=base.kernel,
            thread_count=base.thread_count,
            numeric_reproducibility_class=base.numeric_reproducibility_class,
            load_path=base.load_path,
            load_note=self._note,
            material=base.material,
        )
