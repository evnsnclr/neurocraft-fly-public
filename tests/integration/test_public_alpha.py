from __future__ import annotations

import hashlib
import json
import subprocess
import tomllib
import zipfile
from pathlib import Path
from typing import Any

import pytest

from scripts import java_runtime
from scripts import public_alpha as alpha
from scripts import run_interactive_demo as launcher


def _write(path: Path, data: bytes = b"source") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _runtime(root: Path) -> dict[str, Any]:
    return {
        "python": alpha.PYTHON_VERSION,
        "versions": {"numpy": "2.2.5", "pyarrow": "21.0.0"},
        "origins": {
            "neurocraft_neural": str(
                root / "packages/neural-runtime/src/neurocraft_neural/__init__.py"
            ),
            "neurocraft_controller": str(
                root / "packages/controller-runtime/src/neurocraft_controller/__init__.py"
            ),
            "neurocraft_service": str(
                root / "services/neural-engine/src/neurocraft_service/__init__.py"
            ),
        },
        "kernel": {"name": "c99_rowblock_sequential", "identity_sha256": "a" * 64},
    }


@pytest.fixture
def prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "source with spaces"
    for name in (
        "uv.lock",
        "pyproject.toml",
        "scripts/run_interactive_demo.py",
        "scripts/run_connectome_service.py",
        "mod/gradlew",
        "mod/gradlew.bat",
        "mod/gradle/wrapper/gradle-wrapper.jar",
        ".venv/bin/python3",
        ".venv/Scripts/python.exe",
    ):
        _write(root / name)
    _write(root / "mod/run/eula.txt", b"# user acceptance\neula=true\n")
    manifest: dict[str, Any] = {"arrays": {}}
    for index in range(8):
        filename = f"array-{index}.bin"
        _write(root / alpha.ARTIFACT / filename, b"1234")
        manifest["arrays"][str(index)] = {"filename": filename, "bytes": 4}
    raw = json.dumps(manifest).encode()
    _write(root / alpha.ARTIFACT / "manifest.json", raw)
    _write(
        root / alpha.CALIBRATION,
        json.dumps(
            {
                "schema": "neurocraft.full_readout_calibration.loom.v2",
                "artifact": {"manifest_file_sha256": hashlib.sha256(raw).hexdigest()},
            }
        ).encode(),
    )
    sources = []
    for name in ("annotations", "neurotransmitters"):
        _write(root / f"data/cache/male-cns-v1.0/{name}.feather", b"1234")
        sources.append({"id": name, "filename": f"{name}.feather", "bytes": 4})
    _write(root / alpha.SOURCES, json.dumps({"sources": sources}).encode())
    monkeypatch.setattr(alpha, "ROOT", root)
    monkeypatch.setattr(alpha.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(alpha.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(alpha.shutil, "which", lambda name: "/tools/" + name)
    for key in (*alpha.FORBIDDEN_ENV, "JAVA_HOME"):
        monkeypatch.delenv(key, raising=False)

    def probe(argv: list[str], *, root: Path) -> tuple[int, str]:
        if argv[-1] == "--version":
            return 0, f"uv {alpha.UV_VERSION} (test)"
        if argv[-1] == "-version":
            return 0, 'openjdk version "21.0.8" 2025-07-15'
        assert argv[-1] == alpha.RUNTIME_PROBE
        return 0, json.dumps(_runtime(root))

    monkeypatch.setattr(alpha, "_probe", probe)
    return root


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


def test_doctor_is_repeatable_read_only_and_scopes_its_checks(prepared: Path) -> None:
    before = _files(prepared)
    first = alpha.doctor(prepared)
    assert first == alpha.doctor(prepared)
    assert first["ready_for_prepare"] is True
    assert _files(prepared) == before
    assert "No achieved TPS/FPS" in " ".join(first["limits"])
    assert "not this lightweight doctor" in next(
        item["message"] for item in first["checks"] if item["name"] == "data"
    )


def test_missing_setup_reports_failures_without_creating_files(
    prepared: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (prepared / "mod/run/eula.txt").unlink()
    (prepared / ".venv/bin/python3").unlink()
    (prepared / alpha.ARTIFACT / "manifest.json").unlink()
    before = _files(prepared)
    monkeypatch.setattr(alpha.subprocess, "run", lambda *a, **k: pytest.fail("must not launch"))
    assert alpha.main(["prepare"]) == 1
    assert "Ready to prepare: no" in capsys.readouterr().out
    assert _files(prepared) == before


@pytest.mark.parametrize(
    "system,machine,status",
    [
        ("Darwin", "arm64", "pass"),
        ("Linux", "x86_64", "warn"),
        ("Windows", "AMD64", "warn"),
        ("Darwin", "x86_64", "warn"),
    ],
)
def test_platform_scope(
    prepared: Path, monkeypatch: pytest.MonkeyPatch, system: str, machine: str, status: str
) -> None:
    monkeypatch.setattr(alpha.platform, "system", lambda: system)
    monkeypatch.setattr(alpha.platform, "machine", lambda: machine)
    report = alpha.doctor(prepared)
    assert report["checks"][0]["status"] == status
    assert report["ready_for_prepare"] is (status != "fail")


@pytest.mark.parametrize(
    "mutation",
    ["manifest", "array_missing", "array_short", "annotations_missing", "calibration_schema"],
)
def test_bad_data_cannot_prepare(prepared: Path, mutation: str) -> None:
    if mutation == "manifest":
        with (prepared / alpha.ARTIFACT / "manifest.json").open("ab") as stream:
            stream.write(b" ")
    elif mutation == "array_missing":
        (prepared / alpha.ARTIFACT / "array-0.bin").unlink()
    elif mutation == "array_short":
        (prepared / alpha.ARTIFACT / "array-0.bin").write_bytes(b"1")
    elif mutation == "annotations_missing":
        (prepared / "data/cache/male-cns-v1.0/annotations.feather").unlink()
    else:
        value = json.loads((prepared / alpha.CALIBRATION).read_bytes())
        value["schema"] = "wrong"
        (prepared / alpha.CALIBRATION).write_text(json.dumps(value))
    report = alpha.doctor(prepared)
    assert report["ready_for_prepare"] is False
    assert next(item for item in report["checks"] if item["name"] == "data")["status"] == "fail"


@pytest.mark.parametrize(
    "entry",
    [
        {"filename": "../outside", "bytes": 4},
        {"filename": "/outside", "bytes": 4},
        {"filename": "array-0.bin", "bytes": True},
        {"filename": "array-0.bin", "bytes": 4.0},
    ],
)
def test_data_paths_and_lengths_are_strict(prepared: Path, entry: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="invalid file declaration"):
        alpha._sized_file(prepared / alpha.ARTIFACT, entry)


def test_metadata_duplicate_and_oversize_refuse() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        alpha._json_object(b'{"sources":[],"sources":[]}')
    with pytest.raises(ValueError, match="64 KiB"):
        alpha._json_object(b" " * (alpha.MAX_METADATA_BYTES + 1))


@pytest.mark.parametrize("mutation", ["python", "numpy", "kernel", "foreign"])
def test_wrong_runtime_or_missing_c_kernel_blocks(
    prepared: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    runtime = _runtime(prepared)
    if mutation == "python":
        runtime["python"] = "3.14.0"
    elif mutation == "numpy":
        runtime["versions"]["numpy"] = None
    elif mutation == "kernel":
        runtime["kernel"] = None
    else:
        runtime["origins"]["neurocraft_neural"] = "/other-checkout/neurocraft_neural/__init__.py"
    monkeypatch.setattr(alpha, "_probe", lambda *a, **k: (0, json.dumps(runtime)))
    assert any(check.status == "fail" for check in alpha._runtime_checks(prepared))


def test_reports_never_expose_environment_values(
    prepared: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "private-value-must-not-appear"
    monkeypatch.setenv("JAVA_TOOL_OPTIONS", secret)
    monkeypatch.setenv("NEUROCRAFT_RUN_TOKEN", secret)
    report = alpha.doctor(prepared)
    assert report["ready_for_prepare"] is False
    assert "JAVA_TOOL_OPTIONS" in json.dumps(report)
    assert secret not in json.dumps(report)


def test_setup_only_prints_plan(
    prepared: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(alpha, "doctor", lambda *a: pytest.fail("setup must not probe"))
    monkeypatch.setattr(alpha.subprocess, "run", lambda *a, **k: pytest.fail("no download"))
    before = _files(prepared)
    assert alpha.main(["setup", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["executes"] is False
    assert "--no-dev" in result["steps"][1]["command"]
    assert "scripts/run_connectome_service.py" not in result["steps"][1]["command"]
    assert _files(prepared) == before


def test_prepare_routes_to_real_launcher_defaults_and_preserves_exit(
    prepared: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        assert argv[:3] == [
            str(alpha.runtime_python(prepared)),
            "-I",
            str(prepared / "scripts/run_interactive_demo.py"),
        ]
        args = launcher.build_parser().parse_args(argv[3:])
        assert args.prepare_only and args.comparison_lanes and args.showcase_world
        assert args.controller_profile == "full_mcns_interactions_program_v2"
        assert args.game_tick_rate == 20 and args.control_seed == 7
        assert args.client_width == 1280 and args.client_height == 720
        assert args.client_max_fps == 60 and args.kernel_threads == 4
        assert not args.diagnostic_tools
        assert not args.circuit_playground
        assert kwargs == {"cwd": prepared, "check": False}
        return subprocess.CompletedProcess(argv, 17)

    monkeypatch.setattr(alpha.subprocess, "run", run)
    assert alpha.main(["prepare", "--kernel-threads", "4"]) == 17
    assert len(calls) == 1


@pytest.mark.parametrize("operation", ["prepare", "run"])
def test_playground_option_reaches_real_launcher(
    prepared: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    calls: list[list[str]] = []

    class ReplacedProcess(BaseException):
        pass

    def record(argv: list[str]) -> None:
        calls.append(argv)
        args = launcher.build_parser().parse_args(argv[3:])
        assert args.circuit_playground and args.showcase_world and args.comparison_lanes
        assert args.prepare_only is (operation == "prepare")
        assert args.controller_profile == alpha.PROFILE and args.game_tick_rate == 20
        assert not args.diagnostic_tools

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        record(argv)
        return subprocess.CompletedProcess(argv, 0)

    def replace(executable: str, argv: list[str]) -> None:
        record(argv)
        raise ReplacedProcess

    monkeypatch.setattr(alpha.subprocess, "run", run)
    monkeypatch.setattr(alpha.os, "execv", replace)
    if operation == "run":
        with pytest.raises(ReplacedProcess):
            alpha.main([operation, "--circuit-playground"])
    else:
        assert alpha.main([operation, "--circuit-playground"]) == 0
    assert len(calls) == 1


def test_run_replaces_wrapper_with_real_launcher_and_forwards_installed_flags(
    prepared: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    class ReplacedProcess(BaseException):
        pass

    def replace(executable: str, argv: list[str]) -> None:
        calls.append((executable, argv))
        raise ReplacedProcess

    monkeypatch.setattr(alpha.os, "execv", replace)
    monkeypatch.setattr(alpha, "_installed_checks", lambda *a: [])
    with pytest.raises(ReplacedProcess):
        alpha.main(
            [
                "run",
                "--installed-runtime",
                "runtime with spaces",
                "--manual-client-name",
                "ActualPlayer",
            ]
        )
    assert len(calls) == 1
    executable, args = calls[0]
    assert executable == str(alpha.runtime_python(prepared))
    assert "--prepare-only" not in args and "--diagnostic-tools" not in args
    assert args[args.index("--installed-runtime") + 1] == str(
        Path("runtime with spaces").absolute()
    )
    assert args[args.index("--manual-client-name") + 1] == "ActualPlayer"
    parsed = launcher.build_parser().parse_args(args[3:])
    assert parsed.installed_runtime == Path("runtime with spaces").absolute()
    assert parsed.manual_client_name == "ActualPlayer"


@pytest.mark.parametrize(
    "args",
    [
        ["run", "--installed-runtime", "somewhere"],
        ["prepare", "--manual-client-name", "ActualPlayer"],
        ["run", "--installed-runtime", "somewhere", "--manual-client-name", "Bad Name"],
        ["doctor", "--client-mods-dir", "mods"],
        ["prepare", "--client-mods-dir", "mods"],
    ],
)
def test_bad_installed_arguments_refuse_before_any_probe(
    prepared: Path, monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    monkeypatch.setattr(alpha, "doctor", lambda *a: pytest.fail("must validate arguments first"))
    with pytest.raises(SystemExit) as exc:
        alpha.main(args)
    assert exc.value.code == 2


def test_installed_doctor_mismatch_blocks_launch_without_output(
    prepared: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = []

    def checks(root: Path, runtime: Path, client: Path | None) -> list[alpha.Check]:
        seen.append((root, runtime, client))
        return [alpha.Check("installed_runtime", "fail", "Mismatched old candidate")]

    monkeypatch.setattr(alpha, "_installed_checks", checks)
    monkeypatch.setattr(alpha.os, "execv", lambda *a: pytest.fail("must refuse before launch"))
    before = _files(prepared)
    assert (
        alpha.main(
            [
                "run",
                "--installed-runtime",
                "runtime",
                "--manual-client-name",
                "ActualPlayer",
                "--client-mods-dir",
                "client mods",
            ]
        )
        == 1
    )
    assert seen == [(prepared, Path("runtime"), Path("client mods"))]
    assert _files(prepared) == before


def test_matching_old_jars_refuse_before_launch_when_source_version_changed(
    prepared: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime, client = tmp_path / "installed", tmp_path / "client mods"
    api = b"synthetic pinned API fixture"
    _write(runtime / "server/mods/fabric-api.jar", api)
    _write(runtime / "server/game/server.jar", b"synthetic game")
    _write(runtime / "server/java-launch.json", b"{}")
    # Java-policy validation has its own tampering tests; this fixture exercises
    # the real inventory and mod/source version checks with synthetic JAR bytes.
    monkeypatch.setattr(java_runtime, "validate_server", lambda server: {})
    old_mod = runtime / "server/mods/neurocraft-fly.jar"
    with zipfile.ZipFile(old_mod, "w") as jar:
        jar.writestr(
            "fabric.mod.json",
            json.dumps(
                {
                    "id": "neurocraft_fly",
                    "version": "0.1.0-alpha.1",
                    "depends": {
                        "minecraft": "=1.21.5",
                        "fabricloader": "=0.16.10",
                        "fabric-api": "=0.128.2+1.21.5",
                    },
                }
            ),
        )
        jar.writestr("dev/neurocraft/fly/NeuroCraftFly.class", b"synthetic remapped marker")
        jar.writestr("LICENSE", b"fixture license")
        jar.writestr("NOTICE", b"fixture notice")
    _write(
        runtime / "runtime-manifest.json",
        json.dumps(
            {
                "schema": "neurocraft.installed_runtime.v2",
                "minecraft": "1.21.5",
                "loader": "0.16.10",
                "fabric_api": "0.128.2+1.21.5",
                "mod_version": "0.1.0-alpha.1",
                "files": {
                    name: hashlib.sha256(data).hexdigest()
                    for name, data in _files(runtime / "server").items()
                },
            }
        ).encode(),
    )
    helper = Path(alpha.__file__).with_name("installed_mod.py").read_text()
    # Only the API pin changes to permit the declared synthetic fixture.
    helper = helper.replace(
        "a82fd00827206e911936ed1e0ceaec6eb55d061ca5d3c5d63c7f0031426d29ae",
        hashlib.sha256(api).hexdigest(),
    )
    _write(prepared / "scripts/installed_mod.py", helper.encode())
    _write(
        prepared / "mod/gradle.properties",
        b"archives_base_name=neurocraft-fly\nmod_version=0.1.0-alpha.2\n",
    )
    source_jar, _ = alpha.source_mod(prepared)
    _write(source_jar, old_mod.read_bytes())
    _write(client / "renamed-mod.jar", old_mod.read_bytes())
    _write(client / "renamed-api.jar", api)
    monkeypatch.setattr(alpha.os, "execv", lambda *a: pytest.fail("must refuse before launch"))
    before = _files(tmp_path)
    assert (
        alpha.main(
            [
                "run",
                "--installed-runtime",
                str(runtime),
                "--manual-client-name",
                "ActualPlayer",
                "--client-mods-dir",
                str(client),
            ]
        )
        == 1
    )
    assert "built mod version differs from the current source version" in capsys.readouterr().out
    assert _files(tmp_path) == before


def test_runtime_path_remains_relative_to_callers_directory_during_prepare(
    prepared: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    caller = tmp_path / "different caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    selected = alpha.launcher_command(
        prepared,
        prepare_only=True,
        installed_runtime=Path("my runtime"),
        manual_client_name="ActualPlayer",
    )
    assert selected[selected.index("--installed-runtime") + 1] == str(caller / "my runtime")


@pytest.mark.parametrize("properties", ["", "mod_version=../bad", "mod_version=1\nmod_version=2"])
def test_source_build_name_rejects_bad_or_missing_metadata(prepared: Path, properties: str) -> None:
    _write(prepared / "mod/gradle.properties", properties.encode())
    with pytest.raises(ValueError, match="version/name"):
        alpha.source_mod(prepared)


def test_tool_probe_timeout_and_output_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*args: Any, **kwargs: Any) -> None:
        assert kwargs["timeout"] == alpha.PROBE_TIMEOUT
        assert "NEUROCRAFT_RUN_TOKEN" not in kwargs["env"]
        assert "JAVA_TOOL_OPTIONS" not in kwargs["env"]
        raise subprocess.TimeoutExpired("probe", 10)

    monkeypatch.setenv("NEUROCRAFT_RUN_TOKEN", "secret")
    monkeypatch.setenv("JAVA_TOOL_OPTIONS", "secret")
    monkeypatch.setattr(alpha.subprocess, "run", timeout)
    assert alpha._probe(["java", "-version"], root=tmp_path)[0] == 1
    monkeypatch.setattr(
        alpha.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            [], 0, "x" * (alpha.MAX_METADATA_BYTES + 1), ""
        ),
    )
    assert alpha._probe(["java", "-version"], root=tmp_path)[0] == 1


def test_full_runtime_pins_are_production_dependencies() -> None:
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    neural = tomllib.loads((root / "packages/neural-runtime/pyproject.toml").read_text())
    lock = tomllib.loads((root / "uv.lock").read_text())
    assert {"numpy==2.2.5", "pyarrow==21.0.0"} <= set(neural["project"]["dependencies"])
    assert not {"numpy==2.2.5", "pyarrow==21.0.0"} & set(project["dependency-groups"]["dev"])
    locked = {entry["name"]: entry for entry in lock["package"]}
    assert {"numpy", "pyarrow"} <= {
        entry["name"] for entry in locked["neurocraft-neural"]["dependencies"]
    }
    assert locked["numpy"]["version"] == "2.2.5" and locked["pyarrow"]["version"] == "21.0.0"


def test_windows_source_tool_paths_and_experimental_status(
    prepared: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(alpha.platform, "system", lambda: "Windows")
    monkeypatch.setattr(alpha.platform, "machine", lambda: "AMD64")
    _write(prepared / ".venv/Scripts/python.exe")
    _write(prepared / "mod/gradlew.bat")
    assert alpha.runtime_python(prepared) == prepared / ".venv/Scripts/python.exe"
    report = alpha.doctor(prepared)
    assert report["ready_for_prepare"]
    assert report["checks"][0]["status"] == "warn"
    assert "unverified" in report["checks"][0]["message"]
    assert alpha.command(["C:/folder with spaces/python.exe", "-I", "script.py"]).startswith('"C:/')
