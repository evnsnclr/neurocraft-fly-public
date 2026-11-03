#!/usr/bin/env python3
"""Compile the fused CSR SpMV kernel and write its build-identity sidecar.

The shared library and its sidecar are build outputs, never tracked. The
sidecar is what lets the runtime refuse a stale *or foreign* binary. It records
two digests:

``source_sha256``
    the SHA-256 of ``_spmv.c`` as compiled, so a library built from edited
    source is refused;
``library_sha256``
    the SHA-256 of the shared library this run actually emitted, so a sidecar
    can never describe a binary other than the one beside it.

:mod:`neurocraft_neural.spmv_kernel` checks both before ``dlopen`` and hashes
``library_sha256`` into the kernel identity, so a differently-compiled library
carries a different published identity rather than borrowing this one.

Usage::

    make spmv-build
    UV_CACHE_DIR=.uv-cache uv run --frozen python scripts/build_spmv_kernel.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages" / "neural-runtime" / "src"))

from neurocraft_neural.spmv_kernel import (  # noqa: E402
    BUILD_FLAGS,
    BUILD_SIDECAR_NAME,
    BUILD_SIDECAR_SCHEMA,
    LIBRARY_NAME,
    SPMV_ABI_VERSION,
    WINDOWS_CLANG_BUILD_FLAGS,
    WINDOWS_MSVC_BUILD_FLAGS,
    kernel_source_path,
    library_sha256,
)

PORTABLE_REPLACEMENTS: dict[str, str | None] = {"-march=native": None}


def _resolve_compiler(requested: str | None) -> str:
    # An explicit compiler is a requirement, not a hint: quietly falling back
    # would make the release provenance disagree with the requested toolchain.
    explicit = requested or os.environ.get("CC")
    candidates = (
        [explicit]
        if explicit
        else (["clang", "clang-cl", "cl"] if sys.platform == "win32" else ["clang", "cc", "gcc"])
    )
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise SystemExit("no C compiler found; set CC or pass --compiler")


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def _driver_kind(compiler: str) -> str:
    name = Path(compiler).name.lower().removesuffix(".exe")
    return "msvc" if name in {"cl", "clang-cl"} else "gnu"


def _compiler_version(compiler: str) -> tuple[str, str]:
    if Path(compiler).name.lower() in {"cl", "cl.exe"}:
        # cl has no --version switch. Its banner is written to stderr even
        # when /Bv exits nonzero because no translation unit was supplied.
        result = _run([compiler, "/Bv"])
        lines = (result.stdout + "\n" + result.stderr).splitlines()
        banner = next((line.strip() for line in lines if "Compiler Version" in line), None)
        if not banner:
            raise SystemExit(f"cannot identify MSVC compiler: {compiler}")
        return "msvc", banner
    result = _run([compiler, "--version"])
    if result.returncode != 0:
        raise SystemExit(f"{compiler} --version failed: {result.stderr.strip()}")
    first_line = result.stdout.strip().splitlines()[0].strip()
    identifier = "clang" if "clang" in first_line.lower() else first_line.split()[0]
    return identifier, first_line


def _target_triple(compiler: str) -> str:
    if _driver_kind(compiler) == "msvc":
        if Path(compiler).name.lower() in {"clang-cl", "clang-cl.exe"}:
            result = _run([compiler, "--version"])
            target = re.search(r"^Target:\s*(\S+)", result.stdout, re.MULTILINE)
            if result.returncode == 0 and target:
                return target[1]
        else:
            _, version = _compiler_version(compiler)
            for label, architecture in (("x64", "x86_64"), ("ARM64", "aarch64"), ("x86", "i686")):
                if version.endswith(f"for {label}"):
                    return f"{architecture}-pc-windows-msvc"
        raise SystemExit(f"cannot determine Windows compiler target: {compiler}")
    result = _run([compiler, "-dumpmachine"])
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _resolve_flags(*, portable: bool, compiler: str = "clang") -> tuple[str, ...]:
    if sys.platform == "win32":
        if _driver_kind(compiler) == "msvc":
            return WINDOWS_MSVC_BUILD_FLAGS
        return WINDOWS_CLANG_BUILD_FLAGS
    if not portable:
        return BUILD_FLAGS
    resolved: list[str] = []
    for flag in BUILD_FLAGS:
        if flag in PORTABLE_REPLACEMENTS:
            replacement = PORTABLE_REPLACEMENTS[flag]
            if replacement is not None:
                resolved.append(replacement)
            continue
        resolved.append(flag)
    if sys.platform == "darwin":
        # Mach-O otherwise embeds the absolute output path as LC_ID_DYLIB,
        # leaking the developer's build directory and rotating binary identity
        # whenever a portable release is rebuilt at a different location.
        resolved.append("-Wl,-install_name,@rpath/_spmv.so")
    return tuple(resolved)


def build(
    *,
    compiler: str | None,
    output: Path,
    portable: bool,
    verbose: bool,
) -> int:
    resolved_compiler = _resolve_compiler(compiler)
    compiler_id, compiler_version = _compiler_version(resolved_compiler)
    source = kernel_source_path()
    if not source.is_file():
        raise SystemExit(f"kernel source not found: {source}")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    flags = _resolve_flags(portable=portable, compiler=resolved_compiler)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep MSVC's intermediate .obj/.lib/.exp files outside the identity root.
    # Replace the library only after the compiler exits successfully.
    with tempfile.TemporaryDirectory(prefix="neurocraft-spmv-") as scratch:
        scratch_path = Path(scratch)
        staged_output = scratch_path / output.name
        if _driver_kind(resolved_compiler) == "msvc":
            argv = [
                resolved_compiler,
                *flags,
                str(source),
                f"/Fo{scratch_path / '_spmv.obj'}",
                f"/Fe{staged_output}",
                "/link",
                f"/IMPLIB:{scratch_path / '_spmv.lib'}",
            ]
        else:
            argv = [resolved_compiler, *flags, "-o", str(staged_output), str(source)]
        print("compile:", " ".join(argv))
        result = _run(argv)
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        if result.returncode != 0:
            return result.returncode
        if not staged_output.is_file():
            raise SystemExit(f"compiler did not emit the expected library: {staged_output}")
        shutil.copyfile(staged_output, output)

    document = {
        "abi_version": SPMV_ABI_VERSION,
        "compiler_id": compiler_id,
        "compiler_version": compiler_version,
        "flags": list(flags),
        "library_filename": output.name,
        # Hashed after the compiler has finished writing, so the digest is of
        # the bytes the loader will map. Any later edit invalidates the sidecar.
        "library_sha256": library_sha256(output),
        "schema": BUILD_SIDECAR_SCHEMA,
        "source_filename": source.name,
        "source_sha256": source_sha256,
        "target_triple": _target_triple(resolved_compiler),
    }
    sidecar = output.with_name(BUILD_SIDECAR_NAME)
    sidecar.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"library: {output} ({output.stat().st_size} bytes)")
    print(f"sidecar: {sidecar}")
    if verbose:
        print(json.dumps(document, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compiler", default=None, help="C compiler (default: $CC, clang, clang-cl/cl on Windows)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=kernel_source_path().with_name(LIBRARY_NAME),
        help="shared library output path",
    )
    parser.add_argument(
        "--portable",
        action="store_true",
        help=(
            "drop -march=native for a portable POSIX build; Windows builds always use baseline ISA"
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="print the build sidecar")
    arguments = parser.parse_args(argv)
    return build(
        compiler=arguments.compiler,
        output=arguments.output,
        portable=arguments.portable,
        verbose=arguments.verbose,
    )


if __name__ == "__main__":
    raise SystemExit(main())
