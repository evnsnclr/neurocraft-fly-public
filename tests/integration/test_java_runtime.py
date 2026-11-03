"""Synthetic extraction/profile attacks plus native Java checks in the Gradle suite."""

from __future__ import annotations

import io
import json
import zipfile
from hashlib import sha1, sha256
from pathlib import Path
from typing import Any

import pytest

from scripts import java_runtime as java


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "server with spaces"
    root.mkdir()
    real = java.policy()
    pins: dict[str, Any] = {"libraries": [], "loader_libraries": []}
    payloads = {}
    for kind in ("libraries", "loader_libraries"):
        for old in real[kind]:
            data = (old["name"] + " synthetic JAR").encode()
            pin = {
                **old,
                "sha256": sha256(data).hexdigest(),
                "sha1": sha1(data).hexdigest(),
                "bytes": len(data),
            }
            pins[kind].append(pin)
            payloads[pin["name"]] = data
    game = b"synthetic extracted game"
    baseline = b"synthetic unchanged library"
    with zipfile.ZipFile(root / "server.jar", "w") as archive:
        archive.writestr(
            "META-INF/versions.list", sha256(game).hexdigest() + "\t1.21.5\t1.21.5/server.jar\n"
        )
        archive.writestr("META-INF/versions/1.21.5/server.jar", game)
        archive.writestr(
            "META-INF/libraries.list",
            sha256(baseline).hexdigest()
            + "\tfixture:baseline:1\tfixture/baseline/1/baseline-1.jar\n"
            + "0" * 64
            + "\tcom.google.guava:guava:old\told-guava.jar\n"
            + "0" * 64
            + "\torg.ow2.asm:asm:9.6\told-asm.jar\n",
        )
        archive.writestr("META-INF/libraries/fixture/baseline/1/baseline-1.jar", baseline)
        archive.writestr("META-INF/libraries/old-guava.jar", b"never selected")
    (root / "fabric-server-launch.jar").write_bytes(b"synthetic bundler entrypoint")
    pins["minecraft_server"] = {"sha256": java.digest(root / "server.jar")}
    monkeypatch.setattr(java, "policy", lambda: pins)

    def download(pin: dict[str, Any], target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payloads[pin["name"]])

    monkeypatch.setattr(java, "download", download)
    return root


def test_extraction_discards_affected_embedded_jars_and_uses_explicit_classpath(
    server: Path,
) -> None:
    java.prepare_server(server)
    plan = java.validate_server(server)
    assert not (server / "server.jar").exists()
    assert not (server / "fabric-server-launch.jar").exists()
    assert not list(server.rglob("old-guava.jar"))
    assert not list(server.rglob("old-asm.jar"))
    arguments = java.server_arguments(server)
    assert "-jar" not in arguments and "*" not in " ".join(arguments)
    assert arguments[-2:] == ["net.fabricmc.loader.impl.launch.knot.KnotServer", "nogui"]
    assert plan["game_jar"] in plan["classpath"]
    assert len(plan["classpath"]) == 15


@pytest.mark.parametrize("mutation", ["pin", "missing", "shadow", "stale", "bundler", "escape"])
def test_rebased_launch_plan_cannot_bypass_patched_library_policy(
    server: Path, mutation: str
) -> None:
    java.prepare_server(server)
    plan = json.loads((server / java.PLAN).read_text())
    pin = java.policy()["libraries"][0]
    name = "libraries/" + pin["path"]
    if mutation == "pin":
        (server / name).write_bytes(b"old or modified library")
    elif mutation == "missing":
        plan["classpath"].remove(name)
    elif mutation == "shadow":
        old = str(Path(name).parent.parent / "old" / "old.jar")
        (server / old).parent.mkdir()
        (server / old).write_bytes(b"old library")
        plan["classpath"].insert(0, old)
    elif mutation == "stale":
        plan["policy_sha256"] = "0" * 64
    elif mutation == "bundler":
        (server / "server.jar").write_bytes(b"old library bundler")
    else:
        plan["classpath"][0] = "libraries/../../outside.jar"
    (server / java.PLAN).write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        java.validate_server(server)


def test_modified_minecraft_input_is_rejected_before_extraction(server: Path) -> None:
    with (server / "server.jar").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="upstream bytes"):
        java.prepare_server(server)
    assert not (server / "game").exists()


@pytest.mark.parametrize(
    "name",
    [
        "../escape.jar",
        "/absolute.jar",
        "libraries/../../escape.jar",
        "libraries/NUL.jar",
        "libraries/file. ",
        "libraries/a:b.jar",
    ],
)
def test_paths_cannot_escape_or_alias_on_windows(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError):
        java.safe_path(tmp_path, name)


def test_profile_creation_keeps_accounts_and_existing_profiles_unchanged(
    server: Path, tmp_path: Path
) -> None:
    java.prepare_server(server)
    launcher = tmp_path / "normal launcher"
    launcher.mkdir()
    accounts = launcher / "launcher_accounts.json"
    accounts.write_bytes(b"private account fixture")
    profiles = launcher / "launcher_profiles.json"
    profiles.write_bytes(b"player profile fixture")
    mods = tmp_path / "isolated game" / "mods"
    mods.mkdir(parents=True)
    name = java.install_client(server, launcher, mods)
    assert name.startswith("neurocraft-fly-1.21.5-")
    java.validate_client(mods)
    assert java.install_client(server, launcher, mods) == name
    assert accounts.read_bytes() == b"private account fixture"
    assert profiles.read_bytes() == b"player profile fixture"
    profile = json.loads((launcher / f"versions/{name}/{name}.json").read_text())
    assert profile["inheritsFrom"] == "1.21.5"
    assert len(profile["libraries"]) == 13


@pytest.mark.parametrize("mutation", ["profile", "library", "marker"])
def test_client_rejects_drift_before_gameplay(server: Path, tmp_path: Path, mutation: str) -> None:
    java.prepare_server(server)
    launcher = tmp_path / "launcher"
    launcher.mkdir()
    mods = tmp_path / "game/mods"
    mods.mkdir(parents=True)
    name = java.install_client(server, launcher, mods)
    if mutation == "profile":
        profile = launcher / f"versions/{name}/{name}.json"
        value = json.loads(profile.read_text())
        value["libraries"].append({"name": "com.google.guava:guava:old"})
        profile.write_text(json.dumps(value))
    elif mutation == "library":
        (launcher / "libraries" / java.policy()["libraries"][0]["path"]).write_bytes(b"old")
    else:
        (mods.parent / java.CLIENT_MARKER).write_text('{"version":"old"}')
    with pytest.raises(ValueError):
        java.validate_client(mods)


def test_profile_does_not_overwrite_existing_changed_version(server: Path, tmp_path: Path) -> None:
    java.prepare_server(server)
    launcher = tmp_path / "launcher"
    launcher.mkdir()
    mods = tmp_path / "game/mods"
    mods.mkdir(parents=True)
    name = java.client_profile()["id"]
    target = launcher / f"versions/{name}/{name}.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"user changed this")
    with pytest.raises(ValueError, match="different contents"):
        java.install_client(server, launcher, mods)
    assert target.read_bytes() == b"user changed this"
    assert not (launcher / "libraries").exists()


def test_failed_client_copy_removes_only_files_created_by_the_attempt(
    server: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    java.prepare_server(server)
    launcher = tmp_path / "launcher"
    launcher.mkdir()
    owned = launcher / "launcher_accounts.json"
    owned.write_bytes(b"player-owned")
    mods = tmp_path / "game/mods"
    mods.mkdir(parents=True)

    def fail_copy(source: Any, target: Any) -> None:
        target.write(b"partial")
        raise OSError("simulated interrupted disk write")

    monkeypatch.setattr(java.shutil, "copyfileobj", fail_copy)
    with pytest.raises(OSError, match="interrupted"):
        java.install_client(server, launcher, mods)
    assert owned.read_bytes() == b"player-owned"
    assert not list(launcher.rglob("*.jar"))
    assert not (mods.parent / java.CLIENT_MARKER).exists()


@pytest.mark.parametrize("mutation", ["digest", "length", "redirect"])
def test_download_rejects_unpinned_bytes_and_cleans_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    class Response(io.BytesIO):
        url = (
            "https://fixture.invalid/artifact"
            if mutation != "redirect"
            else "https://other.invalid"
        )

    data = b"expected"
    pin = {
        "url": "https://fixture.invalid/artifact",
        "bytes": len(data),
        "sha256": sha256(data).hexdigest(),
    }
    returned = (
        b"modified" if mutation == "digest" else data + b"extra" if mutation == "length" else data
    )
    monkeypatch.setattr(java.urllib.request, "urlopen", lambda *args, **kwargs: Response(returned))
    target = tmp_path / "library.jar"
    with pytest.raises(ValueError):
        java.download(pin, target)
    assert not target.exists()


def test_download_preserves_file_created_by_another_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Response(io.BytesIO):
        url = "https://fixture.invalid/artifact"

    target = tmp_path / "library.jar"
    data = b"expected"
    pin = {"url": Response.url, "bytes": len(data), "sha256": sha256(data).hexdigest()}

    def response(*args: Any, **kwargs: Any) -> Response:
        target.write_bytes(b"belongs to another writer")
        return Response(data)

    monkeypatch.setattr(java.urllib.request, "urlopen", response)
    with pytest.raises(FileExistsError):
        java.download(pin, target)
    assert target.read_bytes() == b"belongs to another writer"
