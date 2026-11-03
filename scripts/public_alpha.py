#!/usr/bin/env python3
"""Inspect, prepare or run the source-based local alpha; never a standalone mod installer.

``doctor`` is read-only and does not download, build, load the full graph or start
Minecraft. ``setup`` prints explicit steps only. ``prepare`` creates a fresh
launcher plan after doctor succeeds; ``run`` explicitly starts the real six-source
application. Both use the existing launcher, identities and numerical code.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "full_mcns_interactions_program_v2"
PYTHON_VERSION = "3.13.18"
UV_VERSION = "0.13.6"
MAX_METADATA_BYTES = 65_536
PROBE_TIMEOUT = 10
CALIBRATION = Path("data/calibration/full-readout-calibration-loom-v2.json")
ARTIFACT = Path("data/processed/male-cns-v1.0-superclass-v1")
SOURCES = Path("data/manifests/male-cns-v1.0.sources.json")
DEPENDENCIES = {"numpy": "2.2.5", "pyarrow": "21.0.0"}
FORBIDDEN_ENV = ("PYTHONPATH", "PYTHONHOME", "JAVA_TOOL_OPTIONS", "JDK_JAVA_OPTIONS", "GRADLE_OPTS")

# This child only queries installed metadata and the existing kernel loader. It
# performs no matrix operation and -B prevents import-cache writes. A separate
# process isolates a refused native load from the doctor process.
RUNTIME_PROBE = """
import importlib.util, json, platform
from importlib.metadata import version
result = {"python": platform.python_version(), "versions": {}, "origins": {}}
for name in ("numpy", "pyarrow"):
    try:
        result["versions"][name] = version(name)
    except Exception:
        result["versions"][name] = None
for name in ("neurocraft_neural", "neurocraft_controller", "neurocraft_service"):
    spec = importlib.util.find_spec(name)
    result["origins"][name] = None if spec is None else spec.origin
try:
    from neurocraft_neural.spmv_kernel import load_c_kernel
    kernel = load_c_kernel(thread_count=1)
    result["kernel"] = {"name": kernel.name, "identity_sha256": kernel.identity.identity_sha256}
except Exception as exc:
    result["kernel"] = None
    result["kernel_error"] = type(exc).__name__
print(json.dumps(result, sort_keys=True))
"""


@dataclass(frozen=True)
class Check:
    name: str
    status: Literal["pass", "warn", "fail"]
    message: str
    next_steps: tuple[str, ...] = ()


def command(argv: Sequence[str]) -> str:
    return subprocess.list2cmdline(argv) if platform.system() == "Windows" else shlex.join(argv)


def runtime_python(root: Path) -> Path:
    return root / (
        ".venv/Scripts/python.exe" if platform.system() == "Windows" else ".venv/bin/python3"
    )


def setup_steps(root: Path) -> list[dict[str, str]]:
    python = str(runtime_python(root))
    return [
        {
            "name": "tools",
            "instruction": f"Install uv {UV_VERSION}, Java 21 and a C compiler (clang/cc/gcc). "
            "Use the pinned Gradle wrapper; setup does not install tools.",
        },
        {
            "name": "python",
            "instruction": "From the source root, sync the pinned production runtime.",
            "command": f"uv sync --frozen --all-packages --no-dev --python {PYTHON_VERSION}",
        },
        {
            "name": "data",
            "instruction": "Explicit download: about 1.1 GB plus metadata, "
            "verified against the committed source manifest.",
            "command": command([python, "-B", "scripts/acquire_malecns.py", "--include-metadata"]),
        },
        {
            "name": "graph",
            "instruction": "Only if the processed artifact is absent. "
            "Historical builder peak RSS was about 6.39 GB, not a gameplay memory requirement. "
            "Choose a new receipt filename; the builder can replace an existing receipt.",
            "command": command(
                [
                    python,
                    "-B",
                    "scripts/build_malecns_runtime.py",
                    "--receipt",
                    "artifacts/data-builds/NEW-public-alpha-build.json",
                ]
            ),
        },
        {
            "name": "kernel",
            "instruction": "Build the local C kernel and its identity sidecar.",
            "command": command([python, "-B", "scripts/build_spmv_kernel.py"]),
        },
        {
            "name": "eula",
            "instruction": "Read https://www.minecraft.net/eula. Only if you accept, "
            "create mod/run/eula.txt containing eula=true. No acceptance is created by this tool.",
        },
        {
            "name": "doctor",
            "instruction": "Recheck, then prepare or explicitly run the local alpha.",
            "command": command([python, "-I", "scripts/public_alpha.py", "doctor"]),
        },
    ]


def _probe(argv: Sequence[str], *, root: Path) -> tuple[int, str]:
    """Local tool probes only; bounded duration, no shell or inherited injection values."""
    environment = {key: value for key, value in os.environ.items() if key not in FORBIDDEN_ENV}
    environment.pop("NEUROCRAFT_RUN_TOKEN", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        result = subprocess.run(
            list(argv),
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 1, "local probe failed or timed out"
    output = result.stdout + result.stderr
    if len(output.encode()) > MAX_METADATA_BYTES:
        return 1, "local probe output exceeded 64 KiB"
    return result.returncode, output.strip()


def _json_object(raw: bytes) -> dict[str, Any]:
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError("metadata exceeds 64 KiB")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate metadata key")
            result[key] = value
        return result

    result = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(result, dict):
        raise ValueError("metadata must be an object")
    return result


def _read_metadata(path: Path) -> bytes:
    with path.open("rb") as stream:
        raw = stream.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError("metadata exceeds 64 KiB")
    return raw


def _sized_file(base: Path, entry: Mapping[str, Any]) -> None:
    filename, size = entry.get("filename"), entry.get("bytes")
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or filename in (".", "..")
        or type(size) is not int
        or size <= 0
    ):
        raise ValueError("invalid file declaration")
    path = base / filename
    if not path.is_file() or path.stat().st_size != size:
        raise ValueError(f"missing or wrong-size file: {filename}")


def _data_checks(root: Path) -> list[Check]:
    steps = setup_steps(root)
    next_steps = (steps[2]["command"], steps[3]["command"])
    try:
        calibration = _json_object(_read_metadata(root / CALIBRATION))
        if calibration.get("schema") != "neurocraft.full_readout_calibration.loom.v2":
            raise ValueError("wrong loom-v2 decoder calibration schema")
        raw = _read_metadata(root / ARTIFACT / "manifest.json")
        manifest = _json_object(raw)
        expected = calibration["artifact"]["manifest_file_sha256"]
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("artifact manifest does not match the committed decoder component")
        arrays = manifest.get("arrays")
        if not isinstance(arrays, dict) or len(arrays) != 8:
            raise ValueError("expected eight declared graph arrays")
        for entry in arrays.values():
            if not isinstance(entry, dict):
                raise ValueError("invalid array declaration")
            _sized_file(root / ARTIFACT, entry)
        source = _json_object(_read_metadata(root / SOURCES))
        found: set[str] = set()
        for entry in source["sources"]:
            if entry["id"] in ("annotations", "neurotransmitters"):
                _sized_file(root / "data/cache/male-cns-v1.0", entry)
                found.add(entry["id"])
        if found != {"annotations", "neurotransmitters"}:
            raise ValueError("required source annotations absent from manifest")
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as exc:
        return [Check("data", "fail", f"Data preparation incomplete: {exc}", next_steps)]
    return [
        Check(
            "data",
            "pass",
            "Decoder/manifest binding and file sizes present. "
            "Bulk hashes and graph semantics are checked by the real launcher/service, "
            "not this lightweight doctor.",
        )
    ]


def _runtime_checks(root: Path) -> list[Check]:
    sync = setup_steps(root)[1]["command"]
    build = setup_steps(root)[4]["command"]
    python = runtime_python(root)
    if not python.is_file():
        return [Check("runtime", "fail", "The source workspace .venv is missing.", (sync,))]
    code, output = _probe([str(python), "-I", "-B", "-c", RUNTIME_PROBE], root=root)
    try:
        if code != 0:
            raise ValueError("runtime probe failed; resync the source workspace")
        value = _json_object(output.encode())
        origins = value.get("origins")
        if not isinstance(origins, dict) or set(origins) != {
            "neurocraft_neural",
            "neurocraft_controller",
            "neurocraft_service",
        }:
            raise ValueError("runtime source origins unavailable")
        expected_roots = {
            "neurocraft_neural": root / "packages/neural-runtime/src",
            "neurocraft_controller": root / "packages/controller-runtime/src",
            "neurocraft_service": root / "services/neural-engine/src",
        }
        for name, location in origins.items():
            if not isinstance(location, str) or not Path(location).resolve().is_relative_to(
                expected_roots[name].resolve()
            ):
                raise ValueError("runtime packages point outside this source checkout")
        if value.get("python") != PYTHON_VERSION or value.get("versions") != DEPENDENCIES:
            raise ValueError(
                f"runtime requires Python {PYTHON_VERSION}, NumPy 2.2.5 and PyArrow 21.0.0"
            )
        kernel = value.get("kernel")
        checks = [
            Check("runtime", "pass", "Pinned Python, NumPy/PyArrow and local source imports.")
        ]
        if (
            not isinstance(kernel, dict)
            or kernel.get("name") != "c99_rowblock_sequential"
            or not isinstance(kernel.get("identity_sha256"), str)
            or re.fullmatch("[0-9a-f]{64}", kernel["identity_sha256"]) is None
        ):
            checks.append(
                Check(
                    "kernel",
                    "fail",
                    "Verified C kernel unavailable; "
                    "the alpha will not silently rely on the NumPy fallback.",
                    (build,),
                )
            )
        else:
            checks.append(
                Check(
                    "kernel",
                    "pass",
                    "Existing C kernel source/binary/ABI validated; "
                    "no graph computation performed. Identity " + kernel["identity_sha256"],
                )
            )
        return checks
    except (ValueError, TypeError, OSError, RecursionError) as exc:
        return [Check("runtime", "fail", str(exc), (sync,))]


def source_mod(root: Path) -> tuple[Path, str]:
    """Read the production JAR path and declared version without executing Gradle."""
    values = {}
    for line in _read_metadata(root / "mod/gradle.properties").decode().splitlines():
        name, separator, value = line.strip().partition("=")
        if separator and name in ("archives_base_name", "mod_version"):
            if name in values or re.fullmatch(r"[A-Za-z0-9.+-]{1,64}", value) is None:
                raise ValueError("invalid or duplicate source mod version/name")
            values[name] = value
    if set(values) != {"archives_base_name", "mod_version"}:
        raise ValueError("source mod version/name is missing")
    path = root / "mod/build/libs" / f"{values['archives_base_name']}-{values['mod_version']}.jar"
    return path, values["mod_version"]


def _installed_checks(
    root: Path, installed_runtime: Path, client_mods_dir: Path | None
) -> list[Check]:
    try:
        spec = importlib.util.spec_from_file_location(
            "neurocraft_installed_check", root / "scripts/installed_mod.py"
        )
        if spec is None or spec.loader is None:
            raise ValueError("installed runtime checker is unavailable")
        module = importlib.util.module_from_spec(spec)
        # Importing this source helper must not create __pycache__ in a read-only doctor.
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous
        mod_jar, expected_version = source_mod(root)
        files = module.validate_candidate(
            installed_runtime, mod_jar, client_mods_dir, expected_version=expected_version
        )
    except (OSError, ValueError, TypeError, RecursionError, zipfile.BadZipFile) as exc:
        return [
            Check(
                "installed_runtime",
                "fail",
                f"Installed candidate is not ready: {exc}",
                (
                    "Build the current source's remapped mod, prepare a new runtime directory, "
                    "and copy its two mods JARs to the isolated client. See INSTALL.md.",
                ),
            )
        ]
    return [
        Check(
            "installed_runtime",
            "pass",
            "Inventoried runtime matches the current source-build JAR bytes: "
            + files["mods/neurocraft-fly.jar"]
            + ". Matching bytes alone do not establish source-build provenance.",
        ),
        Check(
            "client_mods",
            "pass" if client_mods_dir is not None else "warn",
            "Both client JARs match the installed server; client was not started."
            if client_mods_dir is not None
            else "Client mods were not inspected; pass --client-mods-dir to check them.",
        ),
    ]


def doctor(
    root: Path = ROOT,
    *,
    installed_runtime: Path | None = None,
    client_mods_dir: Path | None = None,
) -> dict[str, Any]:
    if client_mods_dir is not None and installed_runtime is None:
        raise ValueError("client mods directory requires an installed runtime")
    checks: list[Check] = []
    system, machine = platform.system(), platform.machine()
    if system not in ("Darwin", "Linux", "Windows"):
        checks.append(
            Check(
                "platform",
                "fail",
                "This alpha targets macOS, Linux and experimental Windows 10/11.",
            )
        )
    else:
        measured = system == "Darwin" and machine == "arm64"
        checks.append(
            Check(
                "platform",
                "pass" if measured else "warn",
                "macOS arm64 has measured development-session evidence; clean-machine "
                "installation remains unverified."
                if measured
                else (
                    "Experimental Windows 10/11 support: Job Object cleanup and native kernel "
                    "need the Windows candidate checks; full-network gameplay and independent "
                    "installation remain unverified."
                    if system == "Windows"
                    else "Experimental platform: Linux has fixture CI; full-network gameplay "
                    "and clean-machine installation are unverified here."
                ),
            )
        )
    required = (
        "uv.lock",
        "pyproject.toml",
        "scripts/run_interactive_demo.py",
        "scripts/run_connectome_service.py",
        "mod/gradlew.bat" if system == "Windows" else "mod/gradlew",
        "mod/gradle/wrapper/gradle-wrapper.jar",
    )
    missing = [name for name in required if not (root / name).is_file()]
    checks.append(
        Check(
            "checkout",
            "fail" if missing else "pass",
            "Missing source files: " + ", ".join(missing)
            if missing
            else "Source checkout and Gradle wrapper present; wrapper not executed.",
        )
    )
    uv = shutil.which("uv")
    uv_result = _probe([uv, "--version"], root=root) if uv else (1, "")
    valid_uv = (
        uv_result[0] == 0
        and re.match(rf"^uv {re.escape(UV_VERSION)}(?:\s|$)", uv_result[1]) is not None
    )
    checks.append(
        Check(
            "uv",
            "pass" if valid_uv else "fail",
            f"uv {UV_VERSION} available." if valid_uv else f"Install uv {UV_VERSION}.",
        )
    )
    java_home = os.environ.get("JAVA_HOME")
    java_name = "java.exe" if system == "Windows" else "java"
    java = str(Path(java_home) / "bin" / java_name) if java_home else shutil.which(java_name)
    java_result = _probe([java, "-version"], root=root) if java else (1, "")
    valid_java = (
        java_result[0] == 0 and re.search(r'version "21(?:[.\"])', java_result[1]) is not None
    )
    checks.append(
        Check(
            "java",
            "pass" if valid_java else "fail",
            "Effective Java 21 available (JAVA_HOME takes precedence)."
            if valid_java
            else "Install Java 21 and point JAVA_HOME/PATH at it.",
        )
    )
    compiler = next(
        (
            path
            for name in (os.environ.get("CC"), "clang", "cc", "gcc", "cl", "clang-cl")
            if name and (path := shutil.which(name))
        ),
        None,
    )
    checks.append(
        Check(
            "compiler",
            "pass" if compiler else "warn",
            "C compiler available for local rebuilds."
            if compiler
            else "No C compiler found. Install clang/cc/gcc before rebuilding the kernel.",
        )
    )
    blocked = [key for key in FORBIDDEN_ENV if os.environ.get(key)]
    checks.append(
        Check(
            "environment",
            "fail" if blocked else "pass",
            "Unset launcher-incompatible variable names: " + ", ".join(blocked)
            if blocked
            else "No launcher-incompatible injection variables.",
        )
    )
    try:
        lines = _read_metadata(root / "mod/run/eula.txt").decode().splitlines()
        accepted = "eula=true" in {
            line.strip().lower()
            for line in lines
            if line.strip() and not line.strip().startswith("#")
        }
    except (OSError, ValueError):
        accepted = False
    checks.append(
        Check(
            "eula",
            "pass" if accepted else "fail",
            "Existing local EULA acceptance found; unchanged."
            if accepted
            else setup_steps(root)[5]["instruction"],
        )
    )
    checks.extend(_data_checks(root))
    checks.extend(_runtime_checks(root))
    if installed_runtime is not None:
        checks.extend(_installed_checks(root, installed_runtime, client_mods_dir))
    return {
        "schema": "neurocraft.public_alpha.doctor.v1",
        "scope": "source-based local development alpha; not standalone wheels or consumer install",
        "platform": {"system": system, "machine": machine},
        "ready_for_prepare": not any(check.status == "fail" for check in checks),
        "checks": [asdict(check) for check in checks],
        "limits": [
            "No download, build, full-graph computation, port binding or Minecraft launch.",
            "Presence/size checks do not certify bulk data integrity; the service validates it.",
            "No achieved TPS/FPS, hardware minimum or clean-machine support claim.",
        ],
    }


def launcher_command(
    root: Path,
    *,
    prepare_only: bool,
    kernel_threads: int = 8,
    circuit_playground: bool = False,
    installed_runtime: Path | None = None,
    manual_client_name: str | None = None,
) -> list[str]:
    if type(kernel_threads) is not int or kernel_threads not in (1, 2, 4, 8):
        raise ValueError("kernel threads must be 1, 2, 4 or 8")
    if (installed_runtime is None) != (manual_client_name is None):
        raise ValueError("installed runtime and manual client name must be supplied together")
    if (
        manual_client_name is not None
        and re.fullmatch("[A-Za-z0-9_]{3,16}", manual_client_name) is None
    ):
        raise ValueError("manual client name must be a Java player name (3..16 letters/digits/_)")
    return [
        str(runtime_python(root)),
        "-I",
        str(root / "scripts/run_interactive_demo.py"),
        "--controller-profile",
        PROFILE,
        "--showcase-world",
        *(["--circuit-playground"] if circuit_playground else []),
        "--comparison-lanes",
        "--control-seed",
        "7",
        "--action-latency",
        "--kernel-threads",
        str(kernel_threads),
        "--game-tick-rate",
        "20",
        "--client-width",
        "1280",
        "--client-height",
        "720",
        "--client-max-fps",
        "60",
        *(
            [
                "--installed-runtime",
                str(installed_runtime.absolute()),
                "--manual-client-name",
                manual_client_name,
            ]
            if installed_runtime is not None and manual_client_name is not None
            else []
        ),
        *(["--prepare-only"] if prepare_only else []),
    ]


def print_report(report: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return
    print(report["scope"])
    for item in report["checks"]:
        print(f"[{item['status'].upper()}] {item['name']}: {item['message']}")
        for step in item["next_steps"]:
            print(f"  Next: {step}")
    print("Ready to prepare: " + ("yes" if report["ready_for_prepare"] else "no"))
    for limit in report["limits"]:
        print(limit)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    subparsers.add_parser("setup").add_argument("--json", action="store_true")
    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--json", action="store_true")
    doctor_parser.add_argument("--installed-runtime", type=Path)
    doctor_parser.add_argument("--client-mods-dir", type=Path)
    for name in ("prepare", "run"):
        launch_parser = subparsers.add_parser(name)
        launch_parser.add_argument("--kernel-threads", type=int, choices=(1, 2, 4, 8), default=8)
        launch_parser.add_argument(
            "--circuit-playground",
            action="store_true",
            help="use the close-up playground scene in the fresh showcase world",
        )
        launch_parser.add_argument("--installed-runtime", type=Path)
        launch_parser.add_argument("--manual-client-name")
        launch_parser.add_argument("--client-mods-dir", type=Path)
    args = parser.parse_args(argv)
    if args.operation == "setup":
        steps = setup_steps(ROOT)
        if args.json:
            print(
                json.dumps(
                    {
                        "schema": "neurocraft.public_alpha.setup.v1",
                        "executes": False,
                        "working_directory": str(ROOT),
                        "steps": steps,
                    },
                    indent=2,
                )
            )
        else:
            print(f"Setup plan only; no commands executed. Run these from {ROOT}.")
            for step in steps:
                print(f"\n{step['name']}: {step['instruction']}")
                if "command" in step:
                    print(step["command"])
        return 0
    if args.client_mods_dir is not None and args.installed_runtime is None:
        parser.error("--client-mods-dir requires --installed-runtime")
    selected: list[str] = []
    if args.operation in ("prepare", "run"):
        try:
            selected = launcher_command(
                ROOT,
                prepare_only=args.operation == "prepare",
                kernel_threads=args.kernel_threads,
                circuit_playground=args.circuit_playground,
                installed_runtime=args.installed_runtime,
                manual_client_name=args.manual_client_name,
            )
        except ValueError as exc:
            parser.error(str(exc))
    report = doctor(
        ROOT,
        installed_runtime=args.installed_runtime,
        client_mods_dir=args.client_mods_dir,
    )
    print_report(report, as_json=args.json if args.operation == "doctor" else False)
    if not report["ready_for_prepare"]:
        return 1
    if args.operation == "doctor":
        return 0
    print(
        "Preparing a fresh run only."
        if args.operation == "prepare"
        else "Starting the six-interaction alpha: 20 TPS target, comparisons, no PNG capture.",
        flush=True,
    )
    # The existing launcher owns child lifecycle, identity, EULA handling and
    # fail-closed operation. We do not create a second implementation here.
    try:
        if args.operation == "run":
            # Replace this wrapper so signals/lifecycle belong directly to the
            # existing launcher rather than leaving a second supervisor alive.
            os.execv(selected[0], selected)
        return subprocess.run(selected, cwd=ROOT, check=False).returncode
    except OSError as exc:
        print(f"Cannot execute the local launcher: {type(exc).__name__}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
