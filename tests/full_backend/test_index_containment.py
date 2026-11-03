"""B4: post-verification CSR corruption must be contained, not undefined.

The reviewer set ``indices[0] = 0xFFFFFFF0`` after view construction and the
process died with exit 139 (SIGSEGV): the kernel dereferences the index buffer
at the top of the step, *before* the amortised checkpoint that was supposed to
mitigate exactly this. The reference engine, by contrast, raised a contained
``IndexError``.

Containment cannot be asserted from inside the interpreter that would die, so
each case runs in a child process and the parent asserts on the exit status.
A regression here shows up as ``returncode == -11`` rather than as a normal
test failure, which is precisely the signal that matters.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

_PROGRAM = textwrap.dedent(
    """
    import numpy as np
    from neurocraft_neural.full_engine import ArrayNativeEngine
    from neurocraft_neural.full_view import ViewProvenance, build_view
    from neurocraft_neural.spmv_kernel import (
        NumpyReduceatKernel,
        SpmvKernelError,
        load_c_kernel,
    )
    from neurocraft_neural.view_verification import VerificationPolicy

    KIND = {kind!r}
    FIELD = {field!r}

    if KIND == "c":
        try:
            kernel = load_c_kernel(thread_count=4)
        except SpmvKernelError as exc:
            print("SKIP", exc)
            raise SystemExit(0)
    else:
        kernel = NumpyReduceatKernel()

    node_count = 64
    degrees = np.full(node_count, 4, dtype=np.int64)
    degrees[0] = 0
    degrees[-1] = 0
    indptr = np.zeros(node_count + 1, dtype=np.int64)
    np.cumsum(degrees, out=indptr[1:])
    total = int(indptr[-1])
    view = build_view(
        indptr=indptr,
        pre_indices=(np.arange(total, dtype=np.int64) % node_count).astype(np.uint32),
        raw_counts=np.full(total, 3, dtype=np.int64),
        signs=np.ones(total, dtype=np.int64),
        provenance=ViewProvenance(origin="containment", graph_identity_sha256="e" * 64),
    )
    engine = ArrayNativeEngine(
        view,
        kernel=kernel,
        verification_policy=VerificationPolicy(full_coverage_period_steps=1000),
    )
    engine.step()

    target = getattr(view, FIELD)
    target.flags.writeable = True
    if FIELD == "indices":
        target[0] = 0xFFFFFFF0
    else:
        target[2] = total + 4096
    target.flags.writeable = False

    try:
        engine.step()
    except SpmvKernelError as exc:
        print("CONTAINED", type(exc).__name__, exc)
        raise SystemExit(0)
    print("NOT_CONTAINED: the corrupted buffer was read without complaint")
    raise SystemExit(1)
    """
)


def _run(kind: str, field: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _PROGRAM.format(kind=kind, field=field)],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )


@pytest.mark.parametrize("kind", ["numpy", "c"])
@pytest.mark.parametrize("field", ["indices", "indptr"])
def test_b4_post_verification_csr_corruption_is_contained(kind: str, field: str) -> None:
    result = _run(kind, field)
    assert result.returncode == 0, (
        f"kind={kind} field={field} exited {result.returncode}; "
        f"stdout={result.stdout!r} stderr={result.stderr[-2000:]!r}"
    )
    assert result.returncode not in {-11, 139}, "the corrupted buffer was dereferenced"
    if result.stdout.startswith("SKIP"):  # pragma: no cover - environment dependent
        pytest.skip(result.stdout.strip())
    assert "CONTAINED" in result.stdout, result.stdout
    assert "SpmvKernelError" in result.stdout
