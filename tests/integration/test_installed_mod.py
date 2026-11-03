"""Small synthetic runtime checks; no downloads, Java, game, or account access."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from scripts import installed_mod as installer
from scripts import public_alpha as alpha
from scripts import run_interactive_demo as launcher

ENTRY = "dev/neurocraft/fly/NeuroCraftFly.class"
VERSION = "0.1.0-alpha.1"
API_BYTES = b"synthetic API fixture, not executable"


@pytest.fixture(autouse=True)
def isolate_java_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    # These tests own inventory/mod/session behavior. Actual Java extraction,
    # pinned-library rejection and launcher profiles are covered in test_java_runtime.
    monkeypatch.setattr(installer.java_runtime, "validate_server", lambda root: {})
    monkeypatch.setattr(installer.java_runtime, "validate_client", lambda root: None)
    monkeypatch.setattr(
        installer.java_runtime,
        "prepare_server",
        lambda root: (root / installer.java_runtime.PLAN).write_text("{}"),
    )
    monkeypatch.setattr(
        installer.java_runtime,
        "server_arguments",
        lambda root: [
            "-Dfabric.gameJarPath.server=" + str(root / "game/server.jar"),
            "-cp",
            "libraries/patched.jar",
            "net.fabricmc.loader.impl.launch.knot.KnotServer",
            "nogui",
        ],
    )


def _metadata() -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "id": "neurocraft_fly",
        "version": VERSION,
        "depends": {
            "minecraft": "=1.21.5",
            "fabricloader": "=0.16.10",
            "fabric-api": "=0.128.2+1.21.5",
        },
    }


def _mod(path: Path, *, mutation: str | None = None) -> Path:
    metadata: Any = _metadata()
    entries = {
        ENTRY: b"synthetic class marker net/minecraft/class_2960",
        "LICENSE": b"fixture rights notice",
        "NOTICE": b"fixture attribution",
    }
    if mutation == "dev":
        entries[ENTRY] = b"synthetic net/minecraft/resources/ResourceLocation"
    elif mutation in ("LICENSE", "NOTICE", ENTRY):
        del entries[mutation]
    elif mutation == "minecraft_entry":
        entries["net/minecraft/Game.class"] = b"unwanted"
    elif mutation == "id":
        metadata["id"] = "another_mod"
    elif mutation == "tuple":
        metadata["depends"]["minecraft"] = "*"
    elif mutation == "version":
        metadata["version"] = "../../bad"
    elif mutation == "metadata_list":
        metadata = []
    elif mutation == "depends_list":
        metadata["depends"] = []
    if mutation != "missing_metadata":
        entries["fabric.mod.json"] = (
            b"{" if mutation == "bad_json" else json.dumps(metadata).encode()
        )
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
        if mutation == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate"):
                archive.writestr("NOTICE", b"ambiguous")
    return path


def _write_manifest(root: Path) -> dict[str, Any]:
    (root / "server" / installer.java_runtime.PLAN).write_text("{}")
    manifest: dict[str, Any] = {
        "schema": "neurocraft.installed_runtime.v2",
        "minecraft": installer.MINECRAFT,
        "loader": installer.LOADER,
        "fabric_api": installer.FABRIC_API,
        "installer": installer.INSTALLER,
        "mod_version": VERSION,
        "scope": "synthetic local test fixture",
        "files": installer._files(root / "server"),
    }
    (root / installer.MANIFEST).write_text(json.dumps(manifest))
    return manifest


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "runtime"
    server = root / "server"
    (server / "mods").mkdir(parents=True)
    (server / "libraries").mkdir()
    (server / "fabric-server-launch.jar").write_bytes(b"synthetic launch")
    (server / "server.jar").write_bytes(b"synthetic game placeholder")
    (server / "libraries" / "dependency.jar").write_bytes(b"synthetic dependency")
    (server / "mods" / "fabric-api.jar").write_bytes(API_BYTES)
    _mod(server / "mods" / "neurocraft-fly.jar")
    monkeypatch.setitem(
        installer.DOWNLOADS,
        "fabric-api.jar",
        (installer.DOWNLOADS["fabric-api.jar"][0], sha256(API_BYTES).hexdigest()),
    )
    _write_manifest(root)
    return root


def test_synthetic_mod_is_accepted_without_executing_it(tmp_path: Path) -> None:
    assert installer.validate_mod(_mod(tmp_path / "mod.jar")) == VERSION


@pytest.mark.parametrize(
    "mutation",
    [
        "dev",
        "LICENSE",
        "NOTICE",
        ENTRY,
        "minecraft_entry",
        "id",
        "tuple",
        "version",
        "metadata_list",
        "depends_list",
        "missing_metadata",
        "bad_json",
        "duplicate",
    ],
)
def test_mod_refuses_bad_metadata_development_and_missing_notices(
    tmp_path: Path, mutation: str
) -> None:
    with pytest.raises(ValueError):
        installer.validate_mod(_mod(tmp_path / "mod.jar", mutation=mutation))


def test_mod_symlink_is_refused(tmp_path: Path) -> None:
    target = _mod(tmp_path / "mod.jar")
    link = tmp_path / "linked.jar"
    link.symlink_to(target)
    with pytest.raises(ValueError):
        installer.validate_mod(link)


@pytest.mark.parametrize("entry", ["fabric.mod.json", ENTRY])
def test_mod_bounds_uncompressed_metadata_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    mod = _mod(tmp_path / "mod.jar")
    with zipfile.ZipFile(mod) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    contents[entry] = b"x" * 1_000_001
    with zipfile.ZipFile(mod, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in contents.items():
            archive.writestr(name, content)
    assert mod.stat().st_size < 16_000_000
    original_read = zipfile.ZipFile.read

    def checked_read(self: zipfile.ZipFile, name: Any, *args: Any, **kwargs: Any) -> bytes:
        assert name != entry, "must check declared size before decompression"
        return original_read(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", checked_read)
    with pytest.raises(ValueError):
        installer.validate_mod(mod)


def test_main_does_not_resolve_away_mod_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _mod(tmp_path / "mod.jar")
    link = tmp_path / "linked.jar"
    link.symlink_to(target)
    output = tmp_path / "new-output"
    monkeypatch.setattr(
        sys, "argv", ["installed_mod", "--output", str(output), "--mod-jar", str(link)]
    )
    monkeypatch.setattr(installer, "_download", lambda *a: pytest.fail("download after bad input"))
    with pytest.raises(SystemExit) as exc:
        installer.main()
    assert exc.value.code == 1
    assert not output.exists()


def test_install_is_no_clobber_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _mod(tmp_path / "mod.jar")
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("unchanged")
    monkeypatch.setattr(
        installer, "_download", lambda *a: pytest.fail("must refuse before download")
    )
    with pytest.raises(FileExistsError):
        installer.install(output, mod)
    assert marker.read_text() == "unchanged"
    assert list(output.iterdir()) == [marker]


@pytest.mark.parametrize("java_home", [False, True])
def test_install_fake_subprocess_inventory_and_no_eula(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, java_home: bool
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []
    # These paths are deliberately different; neither executable is started.
    monkeypatch.setenv("PATH", str(tmp_path / "different-path-java" / "bin"))
    if java_home:
        monkeypatch.setenv("JAVA_HOME", str(tmp_path / "java-21-home"))
        expected_java = str(
            tmp_path
            / "java-21-home"
            / "bin"
            / ("java.exe" if installer.sys.platform == "win32" else "java")
        )
    else:
        monkeypatch.delenv("JAVA_HOME", raising=False)
        expected_java = "java.exe" if installer.sys.platform == "win32" else "java"

    def fake_download(name: str, destination: Path) -> None:
        with destination.open("xb") as stream:
            stream.write(API_BYTES if name == "fabric-api.jar" else b"synthetic installer")

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        server = Path(command[command.index("-dir") + 1])
        (server / "fabric-server-launch.jar").write_bytes(b"synthetic launch")
        (server / "server.jar").write_bytes(b"synthetic game")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(installer, "_download", fake_download)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setitem(
        installer.DOWNLOADS,
        "fabric-api.jar",
        ("https://fixture.invalid/api", sha256(API_BYTES).hexdigest()),
    )
    output = tmp_path / "new-runtime"
    manifest = installer.install(output, _mod(tmp_path / "mod.jar"))
    assert manifest == output / installer.MANIFEST
    files = installer.validate_runtime(output)
    assert len(files) == 5
    command, options = calls[0]
    assert len(calls) == 1
    assert command[:2] == [expected_java, "-jar"]
    assert command[command.index("-mcversion") + 1] == "1.21.5"
    assert command[command.index("-loader") + 1] == "0.16.10"
    assert options == {"check": True, "timeout": 300, "stdin": subprocess.DEVNULL}
    assert not list(output.rglob("eula.txt"))
    assert not list(output.rglob("launcher_profiles.json"))
    assert not list(output.rglob("world"))


@pytest.mark.parametrize("change", ["bytes", "extra", "missing"])
def test_runtime_inventory_detects_changed_files(runtime: Path, change: str) -> None:
    target = runtime / "server" / "server.jar"
    if change == "bytes":
        target.write_bytes(b"altered")
    elif change == "extra":
        (runtime / "server" / "extra.jar").write_bytes(b"not inventoried")
    else:
        target.unlink()
    with pytest.raises(ValueError):
        installer.validate_runtime(runtime)


@pytest.mark.parametrize("bad_path", ["../outside", "/absolute", "a/../b", "a//b", "a\\b", "./a"])
def test_runtime_rejects_inventory_path_tampering(runtime: Path, bad_path: str) -> None:
    manifest = json.loads((runtime / installer.MANIFEST).read_text())
    manifest["files"][bad_path] = "a" * 64
    (runtime / installer.MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        installer.validate_runtime(runtime)


@pytest.mark.parametrize("mutation", ["hash", "tuple", "version", "list", "too_many", "too_few"])
def test_runtime_manifest_semantics_are_checked(runtime: Path, mutation: str) -> None:
    manifest = json.loads((runtime / installer.MANIFEST).read_text())
    if mutation == "hash":
        manifest["files"]["server.jar"] = "G" * 64
    elif mutation == "tuple":
        manifest["loader"] = "0.16.11"
    elif mutation == "version":
        manifest["mod_version"] = "another"
    elif mutation == "list":
        manifest = []
    elif mutation == "too_many":
        manifest["files"] = {f"file{i}": "a" * 64 for i in range(2001)}
    else:
        manifest["files"] = {}
    (runtime / installer.MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        installer.validate_runtime(runtime)


@pytest.mark.parametrize("position", ["root", "manifest", "file", "directory"])
def test_runtime_rejects_symlinks(runtime: Path, tmp_path: Path, position: str) -> None:
    tested = runtime
    if position == "root":
        tested = tmp_path / "linked-runtime"
        tested.symlink_to(runtime, target_is_directory=True)
    else:
        target = {
            "manifest": runtime / installer.MANIFEST,
            "file": runtime / "server" / "server.jar",
            "directory": runtime / "server" / "libraries",
        }[position]
        moved = tmp_path / "moved"
        target.rename(moved)
        target.symlink_to(moved, target_is_directory=position == "directory")
    with pytest.raises(ValueError):
        installer.validate_runtime(tested)


def test_runtime_cannot_rehash_modified_api_to_accept_it(runtime: Path) -> None:
    (runtime / "server" / "mods" / "fabric-api.jar").write_bytes(b"changed API")
    _write_manifest(runtime)
    with pytest.raises(ValueError, match="pinned"):
        installer.validate_runtime(runtime)


def test_candidate_matching_accepts_renamed_jars_and_does_not_change_files(
    runtime: Path, tmp_path: Path
) -> None:
    client = tmp_path / "client mods"
    client.mkdir()
    mod = runtime / "server/mods/neurocraft-fly.jar"
    shutil.copyfile(mod, client / "arbitrary-mod-name.jar")
    shutil.copyfile(runtime / "server/mods/fabric-api.jar", client / "arbitrary-api-name.jar")
    before = installer._files(tmp_path)
    files = installer.validate_candidate(runtime, mod, client, expected_version=VERSION)
    assert files["mods/neurocraft-fly.jar"] == installer.digest(mod)
    assert installer._files(tmp_path) == before


@pytest.mark.parametrize("mutation", ["old_source", "old_client", "extra", "missing", "symlink"])
def test_candidate_matching_rejects_mixed_source_or_client(
    runtime: Path, tmp_path: Path, mutation: str
) -> None:
    mod = tmp_path / "built-mod.jar"
    shutil.copyfile(runtime / "server/mods/neurocraft-fly.jar", mod)
    client = tmp_path / "client mods"
    client.mkdir()
    shutil.copyfile(mod, client / "mod.jar")
    shutil.copyfile(runtime / "server/mods/fabric-api.jar", client / "api.jar")
    if mutation == "old_source":
        with zipfile.ZipFile(mod, "a") as archive:
            archive.writestr("changed.txt", "another same-version build")
    elif mutation == "old_client":
        (client / "mod.jar").write_bytes(b"another same-version build")
    elif mutation == "extra":
        (client / "another.jar").write_bytes(b"conflicting mod")
    elif mutation == "missing":
        (client / "api.jar").unlink()
    else:
        (client / "mod.jar").unlink()
        (client / "mod.jar").symlink_to(mod)
    with pytest.raises(ValueError):
        installer.validate_candidate(runtime, mod, client, expected_version=VERSION)


@pytest.mark.parametrize("source_version", [VERSION, "0.1.0-alpha.2"])
def test_installed_doctor_checks_actual_helper_without_writing_import_cache(
    runtime: Path, tmp_path: Path, source_version: str
) -> None:
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    helper = Path(installer.__file__).read_text()
    # Replace only the pinned API hash with these deliberately synthetic fixture bytes.
    helper = helper.replace(
        "a82fd00827206e911936ed1e0ceaec6eb55d061ca5d3c5d63c7f0031426d29ae",
        sha256(API_BYTES).hexdigest(),
    )
    (source / "scripts/installed_mod.py").write_text(helper)
    (source / "mod/build/libs").mkdir(parents=True)
    (source / "mod/gradle.properties").write_text(
        f"mod_version={source_version}\narchives_base_name=neurocraft-fly\n"
    )
    mod_jar, expected_version = alpha.source_mod(source)
    assert expected_version == source_version
    shutil.copyfile(runtime / "server/mods/neurocraft-fly.jar", mod_jar)
    before = installer._files(tmp_path)
    checks = alpha._installed_checks(source, runtime, None)
    if source_version == VERSION:
        assert [check.status for check in checks] == ["pass", "warn"]
    else:
        assert [check.status for check in checks] == ["fail"]
        assert "version" in checks[0].message
    assert installer._files(tmp_path) == before
    mod_jar.unlink()
    assert alpha._installed_checks(source, runtime, None)[0].status == "fail"


def test_copy_preserves_existing_session_configuration(runtime: Path, tmp_path: Path) -> None:
    destination = tmp_path / "fresh-server"
    destination.mkdir()
    (destination / "eula.txt").write_text("local acceptance")
    (destination / "server.properties").write_text("fresh identity")
    installer.copy_runtime(runtime, destination)
    assert (destination / "eula.txt").read_text() == "local acceptance"
    assert (destination / "server.properties").read_text() == "fresh identity"
    for name, digest in installer.validate_runtime(runtime).items():
        assert installer.digest(destination / name) == digest


@pytest.mark.parametrize(
    "name", ["eula.txt", "server.properties", "world/level.dat", "logs/latest.log"]
)
def test_copy_rejects_installed_session_state_before_any_copy(
    runtime: Path, tmp_path: Path, name: str
) -> None:
    state = runtime / "server" / name
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_bytes(b"old session state")
    _write_manifest(runtime)
    destination = tmp_path / "fresh-server"
    destination.mkdir()
    with pytest.raises(ValueError):
        installer.copy_runtime(runtime, destination)
    assert list(destination.iterdir()) == []


def test_copy_refuses_overwrite_before_any_copy(runtime: Path, tmp_path: Path) -> None:
    destination = tmp_path / "fresh-server"
    destination.mkdir()
    original = destination / "server.jar"
    original.write_bytes(b"keep me")
    with pytest.raises(ValueError):
        installer.copy_runtime(runtime, destination)
    assert original.read_bytes() == b"keep me"
    assert list(destination.iterdir()) == [original]


@pytest.mark.parametrize("position", ["root", "parent"])
def test_copy_refuses_destination_symlink_escape(
    runtime: Path, tmp_path: Path, position: str
) -> None:
    destination = tmp_path / "fresh-server"
    outside = tmp_path / "outside"
    outside.mkdir()
    if position == "root":
        destination.symlink_to(outside, target_is_directory=True)
    else:
        destination.mkdir()
        (destination / "libraries").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        installer.copy_runtime(runtime, destination)
    assert list(outside.iterdir()) == []


class _Response(io.BytesIO):
    def __init__(self, content: bytes, url: str) -> None:
        super().__init__(content)
        self.url = url


@pytest.mark.parametrize("change", ["digest", "redirect"])
def test_pinned_download_mismatch_is_refused_without_real_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    url = installer.DOWNLOADS["fabric-installer.jar"][0]
    calls: list[tuple[str, int]] = []

    def fake_open(requested: str, *, timeout: int) -> _Response:
        calls.append((requested, timeout))
        return _Response(
            b"not pinned bytes", url if change == "digest" else "https://elsewhere.invalid/"
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_open)
    with pytest.raises(ValueError, match=r"digest mismatch|redirect"):
        installer._download("fabric-installer.jar", tmp_path / "download.jar")
    assert calls == [(url, 30)]


def test_download_checks_exact_digest_and_does_not_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = "https://fixture.invalid/exact.jar"
    content = b"verified synthetic bytes"
    monkeypatch.setitem(
        installer.DOWNLOADS, "fabric-installer.jar", (url, sha256(content).hexdigest())
    )
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Response(content, url))
    destination = tmp_path / "pinned.jar"
    installer._download("fabric-installer.jar", destination)
    assert destination.read_bytes() == content
    destination.write_bytes(b"existing local file")
    with pytest.raises(FileExistsError):
        installer._download("fabric-installer.jar", destination)
    assert destination.read_bytes() == b"existing local file"


@pytest.mark.parametrize(
    "options",
    [
        {"installed_runtime": Path("unused")},
        {"manual_client_name": "ValidName"},
        {"installed_runtime": Path("unused"), "manual_client_name": "bad name"},
        {"installed_runtime": Path("unused"), "manual_client_name": "ab"},
        {"installed_runtime": Path("unused"), "manual_client_name": "a" * 17},
        {
            "installed_runtime": Path("unused"),
            "manual_client_name": "ValidName",
            "diagnostic_tools": True,
        },
        {
            "installed_runtime": Path("unused"),
            "manual_client_name": "ValidName",
            "capture_png_compression": 3,
        },
    ],
)
def test_manual_route_bad_flags_refuse_before_output(
    tmp_path: Path, options: dict[str, Any]
) -> None:
    output = tmp_path / "runs"
    with pytest.raises(launcher.LauncherError):
        launcher.prepare_run(run_root=output, **options)
    assert not output.exists()


def _plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> launcher.RunPlan:
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    eula = tmp_path / "accepted-eula.txt"
    eula.write_text("eula=true\n")
    return launcher.prepare_run(run_root=tmp_path / "runs", accepted_eula=eula)


def test_manual_prepare_copies_verified_runtime_and_records_unmanaged_client(
    runtime: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _plan(tmp_path, monkeypatch)
    monkeypatch.setattr(launcher, "_installed_mod_module", lambda: installer)
    plan = launcher.prepare_run(
        run_root=tmp_path / "manual-runs",
        accepted_eula=tmp_path / "accepted-eula.txt",
        installed_runtime=runtime,
        manual_client_name="Local_Player",
    )
    metadata = json.loads(plan.session_path.read_text())
    assert metadata["launch_mode"] == "installed_mod"
    assert metadata["client_settings_managed"] is False
    assert metadata["state"] == "prepared"
    assert plan.manual_client_name == "Local_Player"
    assert plan.console_path is plan.client_actions_path is None
    assert (plan.server_dir / "mods" / "neurocraft-fly.jar").read_bytes() == (
        runtime / "server" / "mods" / "neurocraft-fly.jar"
    ).read_bytes()
    assert plan.entity_id != base.entity_id
    assert "server-ip=127.0.0.1" in (plan.server_dir / "server.properties").read_text()


def test_manual_prepare_does_not_bypass_runtime_symlink_refusal(
    runtime: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(tmp_path, monkeypatch)
    monkeypatch.setattr(launcher, "_installed_mod_module", lambda: installer)
    link = tmp_path / "linked-runtime"
    link.symlink_to(runtime, target_is_directory=True)
    output = tmp_path / "manual-runs"
    with pytest.raises(launcher.LauncherError, match="runtime"):
        launcher.prepare_run(
            run_root=output,
            accepted_eula=tmp_path / "accepted-eula.txt",
            installed_runtime=link,
            manual_client_name="Local_Player",
        )
    assert not output.exists()


@pytest.mark.parametrize("comparison", [False, True])
@pytest.mark.parametrize("java_home", [False, True])
def test_installed_server_keeps_all_identity_properties_and_environment_only_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, comparison: bool, java_home: bool
) -> None:
    base = replace(
        _plan(tmp_path, monkeypatch),
        comparison_lanes=comparison,
        controller_profile=launcher.FULL_MCNS_INTERACTIONS_PROGRAM_V2_PROFILE,
    )
    installed = replace(
        base, installed_runtime=tmp_path / "runtime", manual_client_name="Local_Player"
    )
    monkeypatch.setattr(launcher, "_installed_mod_module", lambda: installer)
    monkeypatch.setenv("PATH", str(tmp_path / "different-path-java" / "bin"))
    if java_home:
        monkeypatch.setenv("JAVA_HOME", str(tmp_path / "java-21-home"))
        expected_java = str(
            tmp_path
            / "java-21-home"
            / "bin"
            / ("java.exe" if installer.sys.platform == "win32" else "java")
        )
    else:
        monkeypatch.delenv("JAVA_HOME", raising=False)
        expected_java = "java.exe" if installer.sys.platform == "win32" else "java"
    identity: dict[str, object] = {
        name: str(i) * 64 for i, name in enumerate(launcher.HASH_KEYS, 1)
    }
    digest = "a" * 64 if comparison else None
    development = launcher._server_command(base, identity, catalog_sha256=digest)
    production = launcher._server_command(installed, identity, catalog_sha256=digest)
    assert production[:2] == [expected_java, "-Xmx2G"]
    assert production[-2:] == ["net.fabricmc.loader.impl.launch.knot.KnotServer", "nogui"]
    assert "-cp" in production and "-jar" not in production
    assert {
        arg for arg in production if arg.startswith("-Dneurocraft") or arg.startswith("-Djava.")
    } == {arg for arg in development if arg.startswith("-D")}
    assert f"-Dneurocraft.fly.runDirectory={base.server_dir}" in production
    assert "runServer" not in production and "./gradlew" not in production
    if comparison:
        assert f"-Dneurocraft.fly.laneCatalogSha256={digest}" in production
        with pytest.raises(launcher.LauncherError, match="catalog digest"):
            launcher._server_command(installed, identity)
    for key in ("PYTHONPATH", "PYTHONHOME", "JAVA_TOOL_OPTIONS", "JDK_JAVA_OPTIONS", "GRADLE_OPTS"):
        monkeypatch.delenv(key, raising=False)
    token = "synthetic-secret-never-command-line"
    environment = launcher._child_environment(token)
    assert environment[launcher.TOKEN_NAME] == token
    assert not any(token in arg or launcher.TOKEN_NAME in arg for arg in production)


def test_windows_java_home_selects_executable_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(installer.sys, "platform", "win32")
    monkeypatch.setenv("JAVA_HOME", str(tmp_path / "java with spaces"))
    assert installer.java_executable() == str(tmp_path / "java with spaces/bin/java.exe")
    monkeypatch.delenv("JAVA_HOME")
    assert installer.java_executable() == "java.exe"


@pytest.mark.parametrize(
    "name", ["C:/outside", "NUL.txt", "mods/AUX", "mods/file.", "mods/file ", "mods/a:b"]
)
def test_windows_ambiguous_inventory_names_are_refused(runtime: Path, name: str) -> None:
    path = runtime / installer.MANIFEST
    manifest = json.loads(path.read_text())
    manifest["files"][name] = "a" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inventory entry"):
        installer.validate_runtime(runtime)
