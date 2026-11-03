"""Pinned Java libraries for development, the local server and an isolated client version."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PLAN = "java-launch.json"
CLIENT_MARKER = "NEUROCRAFT_CLIENT.json"


def policy() -> dict[str, Any]:
    return json.loads((ROOT / "release/java-runtime-pins.json").read_bytes())  # type: ignore[no-any-return]


def policy_hash() -> str:
    return hashlib.sha256((ROOT / "release/java-runtime-pins.json").read_bytes()).hexdigest()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_path(root: Path, name: str) -> Path:
    value = PurePosixPath(name)
    if (
        not name
        or value.is_absolute()
        or str(value) != name
        or any(
            p in (".", "..")
            or p.endswith((" ", "."))
            or p.split(".", 1)[0].upper() in {"CON", "PRN", "AUX", "NUL", "COM1", "LPT1"}
            or any(ord(c) < 32 or c in '\\:<>"|?*' for c in p)
            for p in value.parts
        )
    ):
        raise ValueError("Invalid Java library path")
    result = root / name
    for path in (root, *result.relative_to(root).parents):
        candidate = path if path == root else root / path
        if candidate.is_symlink() or (
            hasattr(candidate, "is_junction") and candidate.is_junction()
        ):
            raise ValueError("Java library path is redirected")
    if result.is_symlink():
        raise ValueError("Java library path is redirected")
    return result


def download(pin: dict[str, Any], target: Path) -> None:
    if target.exists():
        if target.stat().st_size == pin["bytes"] and digest(target) == pin["sha256"]:
            return
        raise ValueError("Existing Java library differs from pinned bytes")
    target.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with (
            urllib.request.urlopen(pin["url"], timeout=30) as response,
            target.open("xb") as stream,
        ):
            created = True
            if response.url != pin["url"]:
                raise ValueError("Pinned Java library redirected unexpectedly")
            total = 0
            while block := response.read(1024 * 1024):
                total += len(block)
                if total > pin["bytes"]:
                    raise ValueError("Java library download exceeds pinned size")
                stream.write(block)
        if total != pin["bytes"] or digest(target) != pin["sha256"]:
            raise ValueError("Java library download does not match pinned bytes")
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise


def _extract(archive: zipfile.ZipFile, member: str, target: Path, expected: str) -> None:
    info = archive.getinfo(member)
    if info.file_size > 100_000_000 or info.is_dir():
        raise ValueError("Invalid bundled Java library")
    target.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(info) as source, target.open("xb") as output:
        shutil.copyfileobj(source, output)
    if digest(target) != expected:
        raise ValueError("Bundled Java library digest mismatch")


def prepare_server(server: Path) -> None:
    """Unpack the verified game; never put Mojang's old-library bundler on the classpath."""
    pins = policy()
    if digest(server / "server.jar") != pins["minecraft_server"]["sha256"]:
        raise ValueError("Minecraft bundle differs from the pinned upstream bytes")
    replacements = {
        ":".join(pin["name"].split(":")[:2]) for pin in pins["loader_libraries"] + pins["libraries"]
    }
    jars: list[str] = []
    with zipfile.ZipFile(server / "server.jar") as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("Duplicate entries in Minecraft server bundle")
        versions = archive.read("META-INF/versions.list").decode().strip().splitlines()
        if len(versions) != 1:
            raise ValueError("Expected exactly one bundled Minecraft version")
        expected, version, path = versions[0].split("\t")
        if version != "1.21.5":
            raise ValueError("Unexpected bundled Minecraft version")
        game = "game/" + path
        _extract(archive, "META-INF/versions/" + path, safe_path(server, game), expected)
        for row in archive.read("META-INF/libraries.list").decode().strip().splitlines():
            expected, name, path = row.split("\t")
            if ":".join(name.split(":")[:2]) in replacements:
                continue
            relative = "libraries/" + path
            _extract(archive, "META-INF/libraries/" + path, safe_path(server, relative), expected)
            jars.append(relative)
    for pin in pins["loader_libraries"] + pins["libraries"]:
        relative = "libraries/" + pin["path"]
        download(pin, safe_path(server, relative))
        jars.append(relative)
    jars.append(game)
    plan = {
        "schema": "neurocraft.java-launch.v1",
        "policy_sha256": policy_hash(),
        "game_jar": game,
        "classpath": jars,
    }
    (server / PLAN).write_text(json.dumps(plan, indent=2) + "\n")
    # The original bundler embeds affected libraries and can restore them on launch.
    # It is an acquisition input only; the extracted, inventoried game is launched.
    (server / "server.jar").unlink()
    (server / "fabric-server-launch.jar").unlink()
    (server / "fabric-server-launcher.properties").unlink(missing_ok=True)
    validate_server(server)


def validate_server(server: Path) -> dict[str, Any]:
    plan = json.loads(safe_path(server, PLAN).read_bytes())
    if not isinstance(plan, dict):
        raise ValueError("Java launch plan must be an object")
    if (
        plan.get("schema") != "neurocraft.java-launch.v1"
        or plan.get("policy_sha256") != policy_hash()
    ):
        raise ValueError("Installed Java policy is stale; prepare a new runtime")
    jars = plan.get("classpath")
    if (
        not isinstance(jars, list)
        or not 13 <= len(jars) <= 200
        or not all(isinstance(name, str) for name in jars)
        or len(jars) != len(set(jars))
    ):
        raise ValueError("Invalid Java classpath")
    game = plan.get("game_jar")
    if not isinstance(game, str) or not game.startswith("game/") or game not in jars:
        raise ValueError("Missing extracted Minecraft game JAR")
    for name in jars:
        if (
            not isinstance(name, str)
            or not name.endswith(".jar")
            or not name.startswith(("libraries/", "game/"))
        ):
            raise ValueError("Unexpected Java classpath entry")
        if not safe_path(server, name).is_file():
            raise ValueError("Missing Java classpath file")
    pins = policy()
    for pin in pins["loader_libraries"] + pins["libraries"]:
        name = "libraries/" + pin["path"]
        if name not in jars or digest(safe_path(server, name)) != pin["sha256"]:
            raise ValueError("Runtime does not contain the patched Java library bytes")
    for pin in pins["loader_libraries"] + pins["libraries"]:
        group, artifact, _ = pin["name"].split(":")
        prefix = "libraries/" + group.replace(".", "/") + "/" + artifact + "/"
        if [name for name in jars if name.startswith(prefix)] != ["libraries/" + pin["path"]]:
            raise ValueError("Runtime contains a shadowing Java library")
    if (server / "server.jar").exists() or (server / "fabric-server-launch.jar").exists():
        raise ValueError("Old Minecraft library bundler is present")
    return plan


def server_arguments(server: Path) -> list[str]:
    plan = validate_server(server)
    return [
        "-Dfabric.gameJarPath.server=" + str((server / plan["game_jar"]).absolute()),
        "-cp",
        os.pathsep.join(str((server / name).absolute()) for name in plan["classpath"]),
        "net.fabricmc.loader.impl.launch.knot.KnotServer",
        "nogui",
    ]


def client_profile() -> dict[str, Any]:
    pins = policy()
    libraries = []
    for pin in pins["loader_libraries"] + pins["libraries"]:
        libraries.append(
            {
                "name": pin["name"],
                "downloads": {
                    "artifact": {
                        "path": pin["path"],
                        "url": pin["url"],
                        "sha1": pin["sha1"],
                        "size": pin["bytes"],
                    }
                },
            }
        )
    return {
        "id": "neurocraft-fly-1.21.5-" + policy_hash()[:12],
        "inheritsFrom": "1.21.5",
        "type": "release",
        "mainClass": "net.fabricmc.loader.impl.launch.knot.KnotClient",
        "time": "2026-10-30T00:00:00Z",
        "releaseTime": "2026-10-30T00:00:00Z",
        "arguments": {"game": [], "jvm": ["-DFabricMcEmu= net.minecraft.client.main.Main "]},
        "libraries": libraries,
    }


def install_client(server: Path, launcher: Path, mods: Path) -> str:
    """Create a new launcher version only; never edit accounts or launcher profiles."""
    validate_server(server)
    if not launcher.is_dir() or launcher.is_symlink():
        raise ValueError("Choose the existing Minecraft launcher folder")
    profile = client_profile()
    name = profile["id"]
    destination = safe_path(launcher, f"versions/{name}/{name}.json")
    encoded = json.dumps(profile, indent=2) + "\n"
    if destination.exists() and destination.read_text() != encoded:
        raise ValueError("Generated launcher version already exists with different contents")
    marker = {"launcher": str(launcher.absolute()), "version": name, "policy_sha256": policy_hash()}
    marker_path = safe_path(mods.parent, CLIENT_MARKER)
    marker_bytes = json.dumps(marker, indent=2) + "\n"
    if marker_path.exists() and marker_path.read_text() != marker_bytes:
        raise ValueError("This game folder already belongs to a different launcher policy")
    for pin in policy()["loader_libraries"] + policy()["libraries"]:
        target = safe_path(launcher, "libraries/" + pin["path"])
        if target.exists() and digest(target) != pin["sha256"]:
            raise ValueError("Existing launcher library differs from pinned bytes")
    created: list[Path] = []
    try:
        for pin in policy()["loader_libraries"] + policy()["libraries"]:
            target = safe_path(launcher, "libraries/" + pin["path"])
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output:
                    created.append(target)
                    with (server / "libraries" / pin["path"]).open("rb") as source:
                        shutil.copyfileobj(source, output)
        for target, content in ((destination, encoded), (marker_path, marker_bytes)):
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("x") as output:
                    created.append(target)
                    output.write(content)
        validate_client(mods)
    except BaseException:
        for target in reversed(created):
            target.unlink(missing_ok=True)
        raise
    return str(name)


def validate_client(mods: Path) -> None:
    marker = json.loads(safe_path(mods.parent, CLIENT_MARKER).read_bytes())
    if not isinstance(marker, dict) or not isinstance(marker.get("launcher"), str):
        raise ValueError("Invalid client launcher marker")
    expected = client_profile()
    if marker.get("version") != expected["id"] or marker.get("policy_sha256") != policy_hash():
        raise ValueError("Client launcher policy is stale; install the matching client files")
    launcher = Path(marker["launcher"])
    if not launcher.is_absolute():
        raise ValueError("Client launcher folder must be absolute")
    name = expected["id"]
    observed = json.loads(safe_path(launcher, f"versions/{name}/{name}.json").read_bytes())
    if observed != expected:
        raise ValueError("Client launcher version differs from the patched policy")
    for pin in policy()["loader_libraries"] + policy()["libraries"]:
        if digest(safe_path(launcher, "libraries/" + pin["path"])) != pin["sha256"]:
            raise ValueError("Client launcher library differs from the patched policy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed-runtime", type=Path, required=True)
    parser.add_argument("--launcher-root", type=Path, required=True)
    parser.add_argument("--mods", type=Path, required=True)
    args = parser.parse_args()
    if not args.mods.is_dir():
        parser.error("mods must be an existing isolated game folder's mods directory")
    print(
        "Select this version in your normal launcher: "
        + install_client(
            args.installed_runtime.absolute() / "server",
            args.launcher_root.absolute(),
            args.mods.absolute(),
        )
    )


if __name__ == "__main__":
    main()
