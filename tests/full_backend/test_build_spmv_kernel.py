"""Native build-driver contracts, including Windows command and digest binding."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from neurocraft_neural.spmv_kernel import BUILD_SIDECAR_NAME, kernel_source_path

from scripts import build_spmv_kernel as builder


@pytest.mark.parametrize("compiler", ["cl.exe", "clang-cl.exe", "C:/LLVM/bin/clang-cl.exe"])
def test_windows_msvc_flags_forbid_fma_and_do_not_require_pthreads(
    monkeypatch: pytest.MonkeyPatch, compiler: str
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    flags = builder._resolve_flags(portable=True, compiler=compiler)
    assert "/fp:strict" in flags
    assert "/LD" in flags
    assert not any("pthread" in flag or "fast" in flag or "AVX" in flag for flag in flags)


def test_windows_clang_uses_native_threads_and_unfused_arithmetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    flags = builder._resolve_flags(portable=False, compiler="clang.exe")
    assert "-ffp-contract=off" in flags
    assert "-shared" in flags
    assert "-pthread" not in flags and "-fPIC" not in flags and "-march=native" not in flags


def test_portable_posix_build_retains_numeric_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    flags = builder._resolve_flags(portable=True)
    assert "-ffp-contract=off" in flags and "-pthread" in flags
    assert "-march=native" not in flags
    assert "-march=native" in builder._resolve_flags(portable=False)


def test_portable_macos_build_uses_relative_install_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    flags = builder._resolve_flags(portable=True)
    assert "-Wl,-install_name,@rpath/_spmv.so" in flags
    assert "-march=native" not in flags
    assert "-ffp-contract=off" in flags


def test_explicit_missing_compiler_does_not_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda candidate: None)
    with pytest.raises(SystemExit, match="no C compiler"):
        builder._resolve_compiler("required-clang")


def test_msvc_version_and_target_use_the_compiler_banner(monkeypatch: pytest.MonkeyPatch) -> None:
    banner = "Microsoft (R) C/C++ Optimizing Compiler Version 19.44.35207.1 for x64\n"
    monkeypatch.setattr(
        builder, "_run", lambda argv: subprocess.CompletedProcess(argv, 2, "", banner)
    )
    assert builder._compiler_version("cl.exe") == ("msvc", banner.strip())
    assert builder._target_triple("cl.exe") == "x86_64-pc-windows-msvc"


def test_clang_cl_target_comes_from_its_version_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        builder,
        "_run",
        lambda argv: subprocess.CompletedProcess(
            argv, 0, "clang version 20.1.0\nTarget: x86_64-pc-windows-msvc\n", ""
        ),
    )
    assert builder._target_triple("clang-cl.exe") == "x86_64-pc-windows-msvc"


def test_msvc_build_is_path_safe_and_binds_actual_dll_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(builder, "_resolve_compiler", lambda requested: "cl.exe")
    monkeypatch.setattr(builder, "_compiler_version", lambda compiler: ("msvc", "test compiler"))
    monkeypatch.setattr(builder, "_target_triple", lambda compiler: "x86_64-pc-windows-msvc")
    calls: list[list[str]] = []

    def compile_stub(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        target = next(argument[3:] for argument in argv if argument.startswith("/Fe"))
        Path(target).write_bytes(b"fixture DLL bytes")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(builder, "_run", compile_stub)
    output = tmp_path / "path with spaces" / "_spmv.dll"
    assert builder.build(compiler="cl.exe", output=output, portable=True, verbose=False) == 0
    document = json.loads(output.with_name(BUILD_SIDECAR_NAME).read_bytes())
    assert output.read_bytes() == b"fixture DLL bytes"
    assert document["library_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert (
        document["source_sha256"] == hashlib.sha256(kernel_source_path().read_bytes()).hexdigest()
    )
    assert document["library_filename"] == "_spmv.dll"
    assert "/fp:strict" in document["flags"]
    assert calls[0][0] == "cl.exe" and "/link" in calls[0]
    assert not list(output.parent.glob("*.obj")) and not list(output.parent.glob("*.lib"))


def test_failed_compile_never_replaces_existing_library_or_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builder, "_resolve_compiler", lambda requested: "clang")
    monkeypatch.setattr(builder, "_compiler_version", lambda compiler: ("clang", "test compiler"))
    monkeypatch.setattr(
        builder, "_run", lambda argv: subprocess.CompletedProcess(argv, 1, "", "compile failed")
    )
    output = tmp_path / "_spmv.so"
    sidecar = output.with_name(BUILD_SIDECAR_NAME)
    output.write_bytes(b"prior library")
    sidecar.write_bytes(b"prior sidecar")
    assert builder.build(compiler="clang", output=output, portable=True, verbose=False) == 1
    assert output.read_bytes() == b"prior library" and sidecar.read_bytes() == b"prior sidecar"
