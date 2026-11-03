"""Private companion packaging refuses contamination and tests relocation."""

from __future__ import annotations

import json
import os
import platform
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

from scripts import build_companion as builder
from scripts.build_public_release import MANIFEST_NAME, SourceFile, canonical_json, make_manifest
from scripts.installed_mod import FABRIC_API, LOADER, MINECRAFT

ROOT = Path(__file__).resolve().parents[2]


def _source(root: Path) -> Path:
    source = root / "curated source"
    source.mkdir()
    content = {name: b"fixture\n" for name in builder.REQUIRED_SOURCE}
    content["LICENSE"] = (ROOT / "LICENSE").read_bytes()
    content["release/companion-runtime-pins.json"] = (
        ROOT / "release/companion-runtime-pins.json"
    ).read_bytes()
    content["mod/gradle.properties"] = b"archives_base_name=neurocraft-fly\nmod_version=0.1.0\n"
    content["packages/neural-runtime/src/neurocraft_neural/__init__.py"] = b""
    content["services/connectome-service/src/neurocraft_service/__init__.py"] = b""
    for path in (ROOT / "release/runtime-legal").rglob("*"):
        if path.is_file():
            content[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    records = []
    for name, data in content.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        records.append(SourceFile(name, name, "a" * 40, 0o100644, data))
    manifest = make_manifest("b" * 40, "selection.json", b"{}", {"exclusions": []}, records)
    (source / MANIFEST_NAME).write_bytes(canonical_json(manifest))
    return source


def _mod(root: Path) -> Path:
    jar = root / "neurocraft-fly.jar"
    with zipfile.ZipFile(jar, "w") as archive:
        archive.writestr("LICENSE", "MIT")
        archive.writestr("NOTICE", "fixture notice")
        archive.writestr("dev/neurocraft/fly/NeuroCraftFly.class", b"fixture class bytes")
        archive.writestr(
            "fabric.mod.json",
            json.dumps(
                {
                    "id": "neurocraft_fly",
                    "version": "0.1.0",
                    "depends": {
                        "minecraft": "=" + MINECRAFT,
                        "fabricloader": "=" + LOADER,
                        "fabric-api": "=" + FABRIC_API,
                    },
                }
            ),
        )
    return jar


def _kernel(root: Path, source: Path) -> Path:
    path = root / ("_spmv.dll" if sys.platform == "win32" else "_spmv.so")
    path.write_bytes(b"fixture native library")
    document: dict[str, Any] = {
        "schema": "neurocraft.spmv.build.v2",
        "abi_version": 2,
        "source_filename": "_spmv.c",
        "source_sha256": builder.digest(source / builder.KERNEL_DIRECTORY / "_spmv.c"),
        "library_filename": path.name,
        "library_sha256": builder.digest(path),
        "flags": ["-ffp-contract=off", "-O3"],
        "compiler_id": "fixture",
        "compiler_version": "fixture compiler",
        "target_triple": "fixture-target",
    }
    if sys.platform == "darwin":
        document["flags"].append("-Wl,-install_name,@rpath/_spmv.so")
    path.with_name(builder.KERNEL_SIDECAR).write_bytes(canonical_json(document))
    return path


def _runtime(root: Path) -> Path:
    runtime = root / "runtime input"
    interpreter = runtime / ("python.exe" if sys.platform == "win32" else "bin/python3")
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"fixture interpreter")
    (runtime / "LICENSE.txt").write_text("CPython license fixture\n")
    return runtime


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, Path]:
    source = _source(tmp_path)
    return {
        "source": source,
        "python_runtime": _runtime(tmp_path),
        "mod_jar": _mod(tmp_path),
        "kernel_library": _kernel(tmp_path, source),
        "output": tmp_path / "candidate with spaces",
        "uv": Path("fixture-uv"),
    }


def _stub_runtime_and_wheels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        builder,
        "_runtime_info",
        lambda prefix: (
            {
                "version": builder.PYTHON_VERSION,
                "origin": "supplied_local_prefix",
                "upstream_authenticity_verified": False,
                "input_prefix_inventory_sha256": "c" * 64,
            },
            Path("Lib/site-packages")
            if sys.platform == "win32"
            else Path("lib/python3.13/site-packages"),
        ),
    )

    def install(uv: Path, app: Path, runtime: Path, purelib: Path, requirements: Path) -> None:
        requirements.write_text("numpy==2.2.5 --hash=sha256:" + "a" * 64 + "\n")
        (purelib / "installed-wheel-license.txt").write_text("preserved license")

    monkeypatch.setattr(builder, "_install_dependencies", install)


def test_build_copies_only_curated_source_and_verifies_relocated_payload(
    inputs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_runtime_and_wheels(monkeypatch)
    private = inputs["source"] / "artifacts/private-world/level.dat"
    private.parent.mkdir(parents=True)
    private.write_bytes(b"must never ship")
    (inputs["python_runtime"] / "module.pyc").write_bytes(b"generated cache")
    tested: list[Path] = []

    def self_test(bundle: Path) -> dict[str, Any]:
        tested.append(bundle)
        assert bundle.name == "relocated candidate with spaces"
        pth = next((bundle / "runtime").rglob("neurocraft-projects.pth"))
        for line in pth.read_text().splitlines():
            assert not Path(line).is_absolute()
            assert (pth.parent / line).resolve().is_relative_to(bundle / "app")
        assert not (bundle / "app/artifacts").exists()
        assert not list(bundle.rglob("*.pyc"))
        assert (bundle / "app/mod/build/libs/neurocraft-fly-0.1.0.jar").is_file()
        return {"status": "passed", "scope": "fixture runner; no native execution"}

    monkeypatch.setattr(builder, "_self_test", self_test)
    manifest = builder.build(**inputs)
    assert len(tested) == 1
    assert manifest["source_commit"] == "b" * 40
    assert manifest["distribution_ready"] is False and manifest["publication_performed"] is False
    assert manifest["python_runtime"]["upstream_authenticity_verified"] is False
    result = builder.verify_bundle(inputs["output"])
    assert result["files"] == len(manifest["files"])
    saved = (inputs["output"] / builder.MANIFEST_NAME).read_text()
    assert str(inputs["source"]) not in saved and str(inputs["python_runtime"]) not in saved
    payload = inputs["output"] / "app/LICENSE"
    payload.write_bytes(b"changed")
    with pytest.raises(builder.CompanionBuildError, match="immutable companion file changed"):
        builder.verify_bundle(inputs["output"])


def test_existing_output_is_never_modified(inputs: dict[str, Path]) -> None:
    inputs["output"].mkdir()
    sentinel = inputs["output"] / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(builder.CompanionBuildError, match="new path"):
        builder.build(**inputs)
    assert sentinel.read_text() == "keep"


def test_missing_component_license_is_refused(inputs: dict[str, Path]) -> None:
    source = inputs["source"]
    document = json.loads((source / MANIFEST_NAME).read_bytes())
    key = builder.platform_key()
    path = f"release/runtime-legal/{key}/licenses/LICENSE.openssl-3.txt"
    document["files"] = [entry for entry in document["files"] if entry["path"] != path]
    with pytest.raises(builder.CompanionBuildError, match="notice is missing"):
        builder._runtime_legal(source, document)


def test_modified_component_license_is_refused(inputs: dict[str, Path]) -> None:
    source = inputs["source"]
    document = json.loads((source / MANIFEST_NAME).read_bytes())
    key = builder.platform_key()
    (source / f"release/runtime-legal/{key}/licenses/LICENSE.openssl-3.txt").write_bytes(b"MIT")
    with pytest.raises(builder.CompanionBuildError, match="notice hash mismatch"):
        builder._runtime_legal(source, document)


def test_component_notices_from_another_python_build_are_refused(inputs: dict[str, Path]) -> None:
    source = inputs["source"]
    path = source / "release/runtime-legal/manifest.json"
    legal = json.loads(path.read_bytes())
    legal["build"] = "20250612"
    path.write_bytes(canonical_json(legal))
    document = json.loads((source / MANIFEST_NAME).read_bytes())
    with pytest.raises(builder.CompanionBuildError, match="unsupported runtime"):
        builder._runtime_legal(source, document)


def test_old_cpython_runtime_is_refused(
    inputs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        builder,
        "_run",
        lambda argv, cwd: json.dumps({"version": "3.13.5", "implementation": "cpython"}),
    )
    with pytest.raises(builder.CompanionBuildError, match="runtime must be CPython"):
        builder._runtime_info(inputs["python_runtime"])


def test_output_inside_runtime_is_refused(inputs: dict[str, Path]) -> None:
    inputs["output"] = inputs["python_runtime"] / "recursive copy"
    with pytest.raises(builder.CompanionBuildError, match="outside the source"):
        builder.build(**inputs)
    assert not inputs["output"].exists()


def test_changed_selected_source_is_refused_before_build(inputs: dict[str, Path]) -> None:
    (inputs["source"] / "uv.lock").write_bytes(b"foreign lock bytes")
    with pytest.raises(ValueError, match=r"size changed|hash mismatch"):
        builder.build(**inputs)
    assert not inputs["output"].exists()


def test_venv_cannot_be_used_as_distributable_runtime(inputs: dict[str, Path]) -> None:
    (inputs["python_runtime"] / "pyvenv.cfg").write_text("home = elsewhere")
    with pytest.raises(builder.CompanionBuildError, match="not a venv"):
        builder._runtime_info(inputs["python_runtime"])


def test_runtime_cannot_link_to_external_interpreter(inputs: dict[str, Path]) -> None:
    interpreter = builder._python_executable(inputs["python_runtime"])
    interpreter.unlink()
    try:
        interpreter.symlink_to(sys.executable)
    except OSError:
        pytest.skip("host does not permit creating test symlinks")
    with pytest.raises(builder.CompanionBuildError, match="symlink"):
        builder._runtime_info(inputs["python_runtime"])


def test_runtime_rejects_preinstalled_packages(
    inputs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = inputs["python_runtime"]
    site = prefix / "lib/python3.13/site-packages"
    site.mkdir(parents=True)
    (site / "sitecustomize.py").write_text("raise RuntimeError('must not execute')")
    monkeypatch.setattr(
        builder,
        "_run",
        lambda argv, cwd: json.dumps(
            {
                "version": builder.PYTHON_VERSION,
                "implementation": "cpython",
                "prefix": str(prefix),
                "base_prefix": str(prefix),
                "purelib": str(site),
                "system": platform.system(),
                "machine": platform.machine(),
            }
        ),
    )
    with pytest.raises(builder.CompanionBuildError, match="preinstalled packages"):
        builder._runtime_info(prefix)


@pytest.mark.parametrize("field", ["source_sha256", "library_sha256", "abi_version"])
def test_foreign_native_kernel_is_refused(inputs: dict[str, Path], field: str) -> None:
    path = inputs["kernel_library"].with_name(builder.KERNEL_SIDECAR)
    document = json.loads(path.read_bytes())
    document[field] = "foreign"
    path.write_bytes(canonical_json(document))
    with pytest.raises(builder.CompanionBuildError, match="does not match"):
        builder._kernel_document(inputs["kernel_library"], inputs["source"])


def test_cpu_specific_native_build_is_refused(inputs: dict[str, Path]) -> None:
    path = inputs["kernel_library"].with_name(builder.KERNEL_SIDECAR)
    document = json.loads(path.read_bytes())
    document["flags"].append("-march=native")
    path.write_bytes(canonical_json(document))
    with pytest.raises(builder.CompanionBuildError, match="portable flags"):
        builder._kernel_document(inputs["kernel_library"], inputs["source"])


def test_macos_kernel_without_relative_install_name_is_refused(
    inputs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    path = inputs["kernel_library"].with_name(builder.KERNEL_SIDECAR)
    document = json.loads(path.read_bytes())
    document["flags"] = [flag for flag in document["flags"] if "install_name" not in flag]
    path.write_bytes(canonical_json(document))
    # Windows fixture filenames are independently platform-rejected, so use
    # the native Unix filename for this platform-specific metadata test.
    library = inputs["kernel_library"].with_name("_spmv.so")
    if inputs["kernel_library"] != library:
        library.write_bytes(inputs["kernel_library"].read_bytes())
        document["library_filename"] = library.name
        path.write_bytes(canonical_json(document))
    with pytest.raises(builder.CompanionBuildError, match="relative Mach-O"):
        builder._kernel_document(library, inputs["source"])


def test_failed_relocation_check_leaves_no_partial_candidate(
    inputs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_runtime_and_wheels(monkeypatch)

    def refuse(bundle: Path) -> dict[str, Any]:
        raise builder.CompanionBuildError("native test failed")

    monkeypatch.setattr(builder, "_self_test", refuse)
    with pytest.raises(builder.CompanionBuildError, match="native test failed"):
        builder.build(**inputs)
    assert not inputs["output"].exists()
    assert not list(inputs["output"].parent.glob(".neurocraft-companion-*"))


def test_wheel_install_uses_only_locked_hashed_production_dependencies(
    inputs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def run(argv: list[str], *, cwd: Path) -> str:
        calls.append(argv)
        return f"uv {builder.UV_VERSION} (fixture)\n" if "--version" in argv else ""

    monkeypatch.setattr(builder, "_run", run)
    site = inputs["python_runtime"] / "lib/python3.13/site-packages"
    site.mkdir(parents=True)
    builder._install_dependencies(
        inputs["uv"],
        inputs["source"],
        inputs["python_runtime"],
        site,
        inputs["output"].with_suffix(".txt"),
    )
    export, install = calls[1:]
    assert all(flag in export for flag in ("--frozen", "--no-dev", "--no-emit-workspace"))
    assert all(
        flag in install
        for flag in (
            "--only-binary",
            ":all:",
            "--require-hashes",
            "--no-deps",
            "--target",
            "--link-mode",
            "copy",
        )
    )
    assert "--editable" not in install and "--system" not in install


def test_launchers_quote_paths_and_use_isolated_bundled_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    builder._write_launchers(tmp_path, Path("runtime/bin/python3"))
    command = (tmp_path / "Start NeuroCraft Fly.command").read_text()
    assert not (tmp_path / "Start NeuroCraft Fly.cmd").exists()
    monkeypatch.setattr(sys, "platform", "win32")
    builder._write_launchers(tmp_path, Path("runtime/bin/python3"))
    windows = (tmp_path / "Start NeuroCraft Fly.cmd").read_text()
    assert '"./runtime/bin/python3" -I -B' in command
    assert '"%~dp0runtime\\bin\\python3" -I -B' in windows
    assert "companion_app.py" in command and "companion_app.py" in windows
    assert "DisableDelayedExpansion" in windows
    assert os.access(tmp_path / "Start NeuroCraft Fly.command", os.X_OK)


def test_relocated_self_test_checks_the_isolated_builder_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interpreter = tmp_path / "runtime/python.exe"
    monkeypatch.setattr(builder, "_python_executable", lambda prefix: interpreter)
    calls: list[list[str]] = []

    def run(argv: list[str], *, cwd: Path) -> str:
        calls.append(argv)
        return "usage: fixture" if argv[-1] == "--help" else '{"status": "passed"}'

    monkeypatch.setattr(builder, "_run", run)
    result = builder._self_test(tmp_path)
    assert result["isolated_graph_builder_help"] == "passed"
    assert calls[0][1:3] == ["-I", "-B"]
    assert calls[0][-3:] == [
        str(tmp_path / "app/scripts/companion_bootstrap.py"),
        "build_malecns_runtime.py",
        "--help",
    ]
