#!/usr/bin/env python3
"""Build a private, relocatable companion testing bundle for the current host.

Inputs are an extracted curated source candidate, a self-contained CPython
prefix matching the pinned version, the remapped mod, and the native kernel plus build
sidecar. No Minecraft runtime, data download, publication or signing occurs.
The supplied runtime's bytes are inventoried, not authenticated as upstream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_public_release import (  # noqa: E402
    MANIFEST_NAME as SOURCE_MANIFEST_NAME,
)
from scripts.build_public_release import (  # noqa: E402
    canonical_json,
    relative_path,
    verify_source_snapshot,
)
from scripts.build_sqlite_runtime import verify_runtime as verify_sqlite_runtime  # noqa: E402
from scripts.companion_downloads import platform_key  # noqa: E402
from scripts.installed_mod import validate_mod  # noqa: E402
from scripts.public_alpha import PYTHON_VERSION, UV_VERSION, source_mod  # noqa: E402

MANIFEST_NAME = "COMPANION_MANIFEST.json"
SCHEMA = "neurocraft.companion-candidate.v1"
KERNEL_DIRECTORY = Path("packages/neural-runtime/src/neurocraft_neural")
KERNEL_SIDECAR = "_spmv.build.json"
PROJECT_IMPORTS = (
    "neurocraft_connectome",
    "neurocraft_controller",
    "neurocraft_embodiment",
    "neurocraft_experiments",
    "neurocraft_neural",
    "neurocraft_protocol",
    "neurocraft_service",
    "scripts.companion_app",
)
REQUIRED_SOURCE = (
    "pyproject.toml",
    "uv.lock",
    "LICENSE",
    "NOTICE",
    "mod/gradle.properties",
    "scripts/companion_app.py",
    "scripts/companion_bootstrap.py",
    "scripts/companion_downloads.py",
    "scripts/build_malecns_runtime.py",
    "release/companion-runtime-pins.json",
    "release/sqlite-runtime.json",
    "scripts/build_sqlite_runtime.py",
    "release/runtime-legal/manifest.json",
    "apps/companion/index.html",
    "apps/companion/app.js",
    "apps/companion/style.css",
    (KERNEL_DIRECTORY / "_spmv.c").as_posix(),
)
_PROBE = """import json, platform, sys, sysconfig
print(json.dumps({"version": platform.python_version(), "implementation": sys.implementation.name,
"prefix": sys.prefix, "base_prefix": sys.base_prefix, "purelib": sysconfig.get_path("purelib"),
"system": platform.system(), "machine": platform.machine()}))
"""
_SELF_TEST = """import importlib, importlib.metadata, json, pathlib, platform, sys
import pyexpat, sqlite3, ssl, zlib
root = pathlib.Path(sys.argv[1]).resolve()
runtime = root / "runtime"
assert pathlib.Path(sys.prefix).resolve() == runtime, "interpreter escaped bundled runtime"
assert platform.python_version() == sys.argv[3], "wrong bundled Python version"
policy = json.loads((root / "app/release/sqlite-runtime.json").read_text())
assert sqlite3.sqlite_version == policy["version"], "unpatched bundled SQLite"
with sqlite3.connect(':memory:') as database:
    assert database.execute('select sqlite_source_id()').fetchone()[0] == policy["source_id"]
sys.path.insert(0, str(root / "app"))
modules = {}
for name in json.loads(sys.argv[2]):
    module = importlib.import_module(name)
    path = pathlib.Path(module.__file__).resolve()
    assert path.is_relative_to(root), "import escaped bundle: " + name
    modules[name] = path.relative_to(root).as_posix()
import numpy as np
from neurocraft_neural.spmv_kernel import load_c_kernel
indptr = np.array([0, 2, 2, 4], dtype=np.int64)
indices = np.array([0, 1, 0, 1], dtype=np.uint32)
values = np.array([-1.0, 1.0 + 2.0**-27] * 2, dtype=np.float64)
x = np.array([1.0, 1.0 - 2.0**-27], dtype=np.float64)
for threads in (1, 2, 8):
    kernel = load_c_kernel(thread_count=threads)
    assert kernel.library_path.resolve().is_relative_to(root), "kernel escaped bundle"
    out = np.empty(3, dtype=np.float64)
    kernel.matvec(indptr=indptr, indices=indices, values=values, x=x, out=out,
                  empty_rows=np.array([1], dtype=np.int64))
    assert out.tobytes() == np.zeros(3, dtype=np.float64).tobytes(), "kernel arithmetic changed"
for name in ("numpy", "pyarrow", "websockets"):
    module = importlib.import_module(name)
    assert pathlib.Path(module.__file__).resolve().is_relative_to(runtime), name
print(json.dumps({"status": "passed", "python_version": platform.python_version(),
"imports": modules,
"runtime_components": {"openssl": ssl.OPENSSL_VERSION, "sqlite": sqlite3.sqlite_version,
                       "expat": pyexpat.EXPAT_VERSION, "zlib": zlib.ZLIB_RUNTIME_VERSION},
"external_versions": {name: importlib.metadata.version(name)
                      for name in ("numpy", "pyarrow", "websockets")},
"kernel_identity_sha256": kernel.identity.identity_sha256, "thread_counts": [1, 2, 8],
"scope": "relocated local-host imports and native arithmetic; no Minecraft execution"}))
"""


class CompanionBuildError(ValueError):
    """A testing candidate failed an input or relocation check."""


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _environment() -> dict[str, str]:
    # Explicit lock/hash inputs must not be overridden by local pip/uv settings.
    # A cache location changes neither locked requirements nor accepted hashes.
    return {
        key: value
        for key, value in os.environ.items()
        if (not key.startswith(("UV_", "PIP_", "PYTHON")) or key == "UV_CACHE_DIR")
        and key not in {"VIRTUAL_ENV", "CONDA_PREFIX", "NEUROCRAFT_SPMV_LIBRARY"}
    }


def _run(argv: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        argv,
        cwd=cwd,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
        stdin=subprocess.DEVNULL,
    )
    if result.returncode:
        raise CompanionBuildError(f"{Path(argv[0]).name} failed: {result.stderr.strip()[-2000:]}")
    return result.stdout


def _python_executable(prefix: Path) -> Path:
    names = (
        ("python.exe", "bin/python.exe")
        if sys.platform == "win32"
        else ("bin/python3", "bin/python")
    )
    for name in names:
        path = prefix / name
        if path.is_file() and path.resolve().is_relative_to(prefix.resolve()):
            return path
    raise CompanionBuildError("self-contained runtime has no native Python executable")


def _inventory(root: Path, *, allow_internal_symlinks: bool = False) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(root.rglob("*")):
        name = relative_path(path.relative_to(root).as_posix())
        if path.is_symlink():
            if (
                not allow_internal_symlinks
                or not path.is_file()
                or not path.resolve().is_relative_to(root.resolve())
            ):
                raise CompanionBuildError(f"external, directory or unsupported symlink: {name}")
        elif not path.is_file():
            if not path.is_dir():
                raise CompanionBuildError(f"unsupported file kind: {name}")
            continue
        if name.casefold() in seen:
            raise CompanionBuildError(f"nonportable duplicate payload path: {name}")
        seen.add(name.casefold())
        record: dict[str, Any] = {
            "path": name,
            "bytes": path.stat().st_size,
            "sha256": digest(path),
            "mode": stat.S_IMODE(path.stat().st_mode),
        }
        if path.is_symlink():
            record["symlink_target"] = os.readlink(path)
        files.append(record)
    return files


def _runtime_info(prefix: Path) -> tuple[dict[str, Any], Path]:
    if (prefix / "pyvenv.cfg").exists():
        raise CompanionBuildError("Python runtime must be a self-contained prefix, not a venv")
    # Validate links before executing even Python's isolated, no-site probe.
    inventory = _inventory(prefix, allow_internal_symlinks=True)
    if any(".git" in PurePosixPath(record["path"]).parts for record in inventory):
        raise CompanionBuildError("runtime must not contain private Git history")
    interpreter = _python_executable(prefix)
    info = json.loads(_run([str(interpreter), "-I", "-B", "-S", "-c", _PROBE], cwd=prefix))
    if info.get("version") != PYTHON_VERSION or info.get("implementation") != "cpython":
        raise CompanionBuildError("runtime must be CPython " + PYTHON_VERSION)
    if any(Path(info[key]).resolve() != prefix for key in ("prefix", "base_prefix")):
        raise CompanionBuildError("runtime Python resolves outside the supplied prefix")
    if info.get("system") != platform.system() or info.get("machine") != platform.machine():
        raise CompanionBuildError("build the companion on its own native OS and architecture")
    purelib = Path(info["purelib"]).resolve()
    if not purelib.is_relative_to(prefix):
        raise CompanionBuildError("runtime site-packages resolves outside the prefix")
    if purelib.exists():
        for path in purelib.iterdir():
            if path.name not in {"pip", "README.txt", "__pycache__"} and not re.fullmatch(
                r"pip-[0-9][A-Za-z0-9.+-]*\.dist-info", path.name
            ):
                raise CompanionBuildError("runtime contains preinstalled packages: " + path.name)
    licenses = (
        prefix / "LICENSE",
        prefix / "LICENSE.txt",
        prefix / "Lib/LICENSE.txt",
        prefix / "lib/python3.13/LICENSE.txt",
    )
    if not any(path.is_file() for path in licenses):
        raise CompanionBuildError("runtime is missing the CPython license")
    try:
        sqlite_build = verify_sqlite_runtime(prefix)
    except (OSError, ValueError) as error:
        raise CompanionBuildError("patched SQLite runtime is required: " + str(error)) from error
    public_info = {key: info[key] for key in ("version", "implementation", "system", "machine")}
    public_info["sqlite_build"] = sqlite_build
    public_info["input_prefix_inventory_sha256"] = hashlib.sha256(
        canonical_json(inventory)
    ).hexdigest()
    public_info["origin"] = "supplied_local_prefix"
    public_info["upstream_authenticity_verified"] = False
    return public_info, purelib.relative_to(prefix)


def _source_manifest(source: Path) -> dict[str, Any]:
    verify_source_snapshot(source)
    document: dict[str, Any] = json.loads((source / SOURCE_MANIFEST_NAME).read_bytes())
    if re.fullmatch(r"[0-9a-f]{40}", document.get("source", {}).get("commit", "")) is None:
        raise CompanionBuildError("source candidate needs a committed source identity")
    selected = {record["path"] for record in document["files"]}
    if missing := set(REQUIRED_SOURCE) - selected:
        raise CompanionBuildError(
            "source candidate is missing companion files: " + ", ".join(sorted(missing))
        )
    return document


def _runtime_legal(source: Path, document: dict[str, Any]) -> dict[str, Any]:
    """Require the matching full-distribution metadata and component notices."""
    base = Path("release/runtime-legal")
    index = source / base / "manifest.json"
    legal = json.loads(index.read_bytes())
    key = platform_key()
    pin = json.loads((source / "release/companion-runtime-pins.json").read_bytes())["platforms"][
        key
    ]["python"]
    if (
        legal.get("schema") != "neurocraft.runtime-legal.v1"
        or legal.get("python_version") != PYTHON_VERSION
        or pin.get("version") != PYTHON_VERSION
        or legal.get("build") != pin.get("build")
    ):
        raise CompanionBuildError("unsupported runtime component notice inventory")
    if key not in legal.get("platforms", {}):
        raise CompanionBuildError("runtime component notices are missing for this platform")
    record = legal["platforms"][key]
    selected = {entry["path"] for entry in document["files"]}
    paths = set()
    for entry in record["files"]:
        relative = base / key / relative_path(entry["path"])
        path = source / relative
        if relative.as_posix() not in selected or not path.is_file():
            raise CompanionBuildError("runtime component notice is missing from curated source")
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise CompanionBuildError("runtime component notice hash mismatch")
        paths.add(entry["path"])
    if "PYTHON.json" not in paths:
        raise CompanionBuildError("runtime component build metadata is missing")
    metadata = json.loads((source / base / key / "PYTHON.json").read_bytes())
    if (
        metadata.get("python_version") != PYTHON_VERSION
        or metadata.get("target_triple") != record["target_triple"]
    ):
        raise CompanionBuildError("runtime component build metadata does not match")
    references: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for name, item in value.items():
                if name == "license_path":
                    references.add(item)
                elif name == "license_paths":
                    references.update(item)
                else:
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(metadata)
    if not references or not references <= paths:
        raise CompanionBuildError("runtime component license collection is incomplete")
    return {"platform": key, "inventory_sha256": digest(index), "files": len(paths)}


def _kernel_document(library: Path, source: Path) -> dict[str, Any]:
    sidecar = library.with_name(KERNEL_SIDECAR)
    if library.is_symlink() or sidecar.is_symlink() or not library.is_file():
        raise CompanionBuildError("kernel library and sidecar must be regular files")
    document: dict[str, Any] = json.loads(sidecar.read_bytes())
    if (
        document.get("schema") != "neurocraft.spmv.build.v2"
        or document.get("abi_version") != 2
        or document.get("source_filename") != "_spmv.c"
        or document.get("source_sha256") != digest(source / KERNEL_DIRECTORY / "_spmv.c")
        or document.get("library_filename") != library.name
        or document.get("library_sha256") != digest(library)
    ):
        raise CompanionBuildError("kernel source/binary/ABI does not match its build sidecar")
    expected = "_spmv.dll" if sys.platform == "win32" else "_spmv.so"
    if library.name != expected:
        raise CompanionBuildError("kernel library filename does not match the build platform")
    flags = document.get("flags")
    if (
        not isinstance(flags, list)
        or any(not isinstance(flag, str) for flag in flags)
        or any(
            "native" in flag or "fast-math" in flag or flag.lower().startswith("/arch:")
            for flag in flags
        )
        or not ({"-ffp-contract=off", "/fp:strict"} & set(flags))
    ):
        raise CompanionBuildError(
            "kernel must use portable flags with unfused floating-point arithmetic"
        )
    if sys.platform == "darwin" and "-Wl,-install_name,@rpath/_spmv.so" not in flags:
        raise CompanionBuildError("portable macOS kernel must use a relative Mach-O install name")
    return document


def _copy_source(source: Path, app: Path, document: dict[str, Any]) -> None:
    app.mkdir()
    # Only manifest-selected files: verified snapshots may contain generated
    # Minecraft worlds/caches which the source verifier deliberately ignores.
    for record in document["files"]:
        destination = app / record["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / record["path"], destination)
    shutil.copy2(source / SOURCE_MANIFEST_NAME, app / SOURCE_MANIFEST_NAME)
    verify_source_snapshot(app)


def _project_paths(app: Path, purelib: Path) -> None:
    paths = sorted([*(app / "packages").glob("*/src"), *(app / "services").glob("*/src")])
    if not paths:
        raise CompanionBuildError("curated source has no project modules")
    entries = [os.path.relpath(path, purelib).replace(os.sep, "/") for path in paths]
    (purelib / "neurocraft-projects.pth").write_text("\n".join(entries) + "\n", encoding="utf-8")


def _remove_caches(root: Path) -> None:
    for path in sorted(root.rglob("__pycache__"), reverse=True):
        if path.is_dir():
            shutil.rmtree(path)
    for pattern in ("*.pyc", "*.pyo"):
        for path in root.rglob(pattern):
            path.unlink()


def _install_dependencies(
    uv: Path, app: Path, runtime: Path, purelib: Path, requirements: Path
) -> None:
    version = _run([str(uv), "--version"], cwd=app).strip()
    if not version.startswith("uv " + UV_VERSION + " ") and version != "uv " + UV_VERSION:
        raise CompanionBuildError("builder requires uv " + UV_VERSION)
    _run(
        [
            str(uv),
            "--no-config",
            "export",
            "--frozen",
            "--no-dev",
            "--no-emit-workspace",
            "--no-header",
            "--no-annotate",
            "--project",
            str(app),
            "--output-file",
            str(requirements),
        ],
        cwd=app,
    )
    _run(
        [
            str(uv),
            "--no-config",
            "pip",
            "install",
            "--python",
            str(_python_executable(runtime)),
            "--no-python-downloads",
            "--only-binary",
            ":all:",
            "--require-hashes",
            "--no-deps",
            "--link-mode",
            "copy",
            "--target",
            str(purelib),
            "--requirements",
            str(requirements),
        ],
        cwd=app,
    )
    # Wheel command-line entry points can embed the temporary build path.
    # They are unnecessary for the companion; package modules and licenses stay.
    for name in ("bin", "Scripts"):
        scripts = purelib / name
        if scripts.is_dir():
            shutil.rmtree(scripts)


def _write_launchers(bundle: Path, interpreter_relative: Path) -> None:
    if sys.platform == "win32":
        windows_python = interpreter_relative.as_posix().replace("/", "\\")
        (bundle / "Start NeuroCraft Fly.cmd").write_bytes(
            (
                '@echo off\r\nsetlocal DisableDelayedExpansion\r\ncd /d "%~dp0"\r\n'
                f'"%~dp0{windows_python}" -I -B "%~dp0app\\scripts\\companion_app.py"\r\n'
                "if errorlevel 1 pause\r\n"
            ).encode()
        )
        return
    command = bundle / "Start NeuroCraft Fly.command"
    command.write_text(
        '#!/bin/sh\nset -eu\ncd -- "$(dirname -- "$0")"\n'
        f'exec "./{interpreter_relative.as_posix()}" -I -B ./app/scripts/companion_app.py\n',
        encoding="utf-8",
    )
    command.chmod(0o755)


def _self_test(bundle: Path) -> dict[str, Any]:
    interpreter = _python_executable(bundle / "runtime")
    _run(
        [
            str(interpreter),
            "-I",
            "-B",
            str(bundle / "app/scripts/companion_bootstrap.py"),
            "build_malecns_runtime.py",
            "--help",
        ],
        cwd=bundle.parent,
    )
    result: dict[str, Any] = json.loads(
        _run(
            [
                str(interpreter),
                "-I",
                "-B",
                "-c",
                _SELF_TEST,
                str(bundle),
                json.dumps(PROJECT_IMPORTS),
                PYTHON_VERSION,
            ],
            cwd=bundle.parent,
        )
    )
    if result.get("status") != "passed":
        raise CompanionBuildError("relocated companion self-test did not pass")
    result["isolated_graph_builder_help"] = "passed"
    return result


def build(
    *,
    source: Path,
    python_runtime: Path,
    mod_jar: Path,
    kernel_library: Path,
    output: Path,
    uv: Path | None = None,
) -> dict[str, Any]:
    """Build locally, then move and self-test before emitting any candidate."""
    if output.exists() or output.is_symlink():
        raise CompanionBuildError("output must be a new path")
    source, python_runtime = source.resolve(), python_runtime.resolve()
    output = output.absolute()
    if output.is_relative_to(source) or output.is_relative_to(python_runtime):
        raise CompanionBuildError("output must be outside the source and Python runtime inputs")
    document = _source_manifest(source)
    legal = _runtime_legal(source, document)
    runtime_info, purelib_relative = _runtime_info(python_runtime)
    runtime_info["component_notices"] = legal
    kernel = _kernel_document(kernel_library, source)
    mod_version = validate_mod(mod_jar)
    declared_jar, declared_version = source_mod(source)
    if declared_version != mod_version:
        raise CompanionBuildError("mod JAR version differs from the curated source version")
    executable = uv or Path(shutil.which("uv") or "uv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".neurocraft-companion-", dir=output.parent) as scratch:
        stage = Path(scratch) / "original"
        stage.mkdir()
        app, runtime = stage / "app", stage / "runtime"
        _copy_source(source, app, document)
        shutil.copytree(python_runtime, runtime, symlinks=False)
        purelib = runtime / purelib_relative
        purelib.mkdir(parents=True, exist_ok=True)
        jar_target = app / declared_jar.relative_to(source)
        jar_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(mod_jar, jar_target)
        kernel_target = app / KERNEL_DIRECTORY / kernel_library.name
        shutil.copyfile(kernel_library, kernel_target)
        shutil.copyfile(
            kernel_library.with_name(KERNEL_SIDECAR), kernel_target.with_name(KERNEL_SIDECAR)
        )
        requirements = stage / "production-requirements.txt"
        _install_dependencies(executable, app, runtime, purelib, requirements)
        _project_paths(app, purelib)
        _remove_caches(stage)
        _write_launchers(stage, _python_executable(runtime).relative_to(stage))
        relocated = Path(scratch) / "relocated candidate with spaces"
        stage.rename(relocated)
        result = _self_test(relocated)
        _remove_caches(relocated)
        (relocated / "self-test.json").write_bytes(canonical_json(result))
        manifest = {
            "schema": SCHEMA,
            "status": "private-testing-candidate",
            "distribution_ready": False,
            "publication_performed": False,
            "source_commit": document["source"]["commit"],
            "source_manifest_sha256": digest(source / SOURCE_MANIFEST_NAME),
            "platform": {"system": platform.system(), "machine": platform.machine()},
            "python_runtime": runtime_info,
            "uv_version": UV_VERSION,
            "mod": {"version": mod_version, "sha256": digest(mod_jar)},
            "kernel": {
                "library_sha256": kernel["library_sha256"],
                "source_sha256": kernel["source_sha256"],
                "build_sidecar_sha256": digest(kernel_library.with_name(KERNEL_SIDECAR)),
            },
            "self_test": result,
            "files": _inventory(relocated),
            "limitations": [
                "Unsigned private testing candidate; publication requires owner approval.",
                "Self-test covers relocated imports and native arithmetic on the build host.",
                "Independent installation and sustained native gameplay require separate checks.",
                "No Minecraft runtime, account data or bulk connectome arrays are bundled.",
            ],
        }
        (relocated / MANIFEST_NAME).write_bytes(canonical_json(manifest))
        if output.exists() or output.is_symlink():
            raise CompanionBuildError("output appeared while building; refusing to overwrite")
        relocated.rename(output)
    return manifest


def verify_bundle(root: Path) -> dict[str, Any]:
    """Verify inventoried immutable payload bytes without executing the bundle."""
    manifest: dict[str, Any] = json.loads((root / MANIFEST_NAME).read_bytes())
    if manifest.get("schema") != SCHEMA or not isinstance(manifest.get("files"), list):
        raise CompanionBuildError("unsupported companion manifest")
    for record in manifest["files"]:
        path = root
        for part in PurePosixPath(relative_path(record["path"])).parts:
            path /= part
            if path.is_symlink():
                raise CompanionBuildError("symlink replaced an immutable payload file")
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or digest(path) != record["sha256"]
        ):
            raise CompanionBuildError("immutable companion file changed: " + record["path"])
    return {
        "schema": SCHEMA,
        "status": "verified-against-accompanying-manifest",
        "files": len(manifest["files"]),
        "distribution_ready": False,
        "publisher_authenticity_verified": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--python-runtime", type=Path)
    parser.add_argument("--mod-jar", type=Path)
    parser.add_argument("--kernel-library", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--uv", type=Path, help=f"pinned uv {UV_VERSION} executable")
    parser.add_argument(
        "--verify-bundle", type=Path, help="only inspect immutable candidate hashes"
    )
    args = parser.parse_args(argv)
    try:
        if args.verify_bundle is not None:
            if any(
                getattr(args, name) is not None
                for name in (
                    "source",
                    "python_runtime",
                    "mod_jar",
                    "kernel_library",
                    "output",
                    "uv",
                )
            ):
                parser.error("--verify-bundle cannot be combined with build arguments")
            print(json.dumps(verify_bundle(args.verify_bundle), sort_keys=True))
            return 0
        if any(
            getattr(args, name) is None
            for name in ("source", "python_runtime", "mod_jar", "kernel_library", "output")
        ):
            parser.error("source, Python runtime, mod JAR, kernel library and output are required")
        result = build(
            source=args.source,
            python_runtime=args.python_runtime,
            mod_jar=args.mod_jar,
            kernel_library=args.kernel_library,
            output=args.output,
            uv=args.uv,
        )
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "schema",
                        "status",
                        "source_commit",
                        "platform",
                        "distribution_ready",
                    )
                },
                sort_keys=True,
            )
        )
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print("companion build refused: " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
