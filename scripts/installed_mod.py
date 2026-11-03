#!/usr/bin/env python3
"""Prepare a private, isolated Fabric runtime; never package Minecraft for release.

The pinned official installer resolves Minecraft's own distribution. Its resulting
files are inventoried locally and rechecked before each copy into a fresh run.
The inventory detects later changes; it is not a signature or an upstream audit.
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
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import java_runtime as java_runtime  # noqa: E402

MINECRAFT = "1.21.5"
LOADER = "0.16.10"
FABRIC_API = "0.128.2+1.21.5"
INSTALLER = "1.1.2"
DOWNLOADS = {
    "fabric-installer.jar": (
        "https://maven.fabricmc.net/net/fabricmc/fabric-installer/1.1.2/fabric-installer-1.1.2.jar",
        "61e035bf7bf70153e127440ce34de47c9036f0a2d0c65d1529454bd35ceefe4f",
    ),
    "fabric-api.jar": (
        "https://maven.fabricmc.net/net/fabricmc/fabric-api/fabric-api/"
        "0.128.2+1.21.5/fabric-api-0.128.2+1.21.5.jar",
        "a82fd00827206e911936ed1e0ceaec6eb55d061ca5d3c5d63c7f0031426d29ae",
    ),
}
MANIFEST = "runtime-manifest.json"
_WINDOWS_DEVICES = frozenset(
    (
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "CONIN$",
        "CONOUT$",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    )
)


def _redirected(path: Path) -> bool:
    """Windows junctions redirect directory copies just as symlinks do."""
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _portable_name(name: str) -> bool:
    """Inventory paths must mean the same in ordinary Windows and POSIX copies."""
    return all(
        part not in ("", ".", "..")
        and not part.endswith((" ", "."))
        and part.split(".", 1)[0].upper() not in _WINDOWS_DEVICES
        and not any(ord(character) < 32 or character in '<>:"|?*' for character in part)
        for part in PurePosixPath(name).parts
    )


def java_executable() -> str:
    """Match the companion doctor and Gradle's JAVA_HOME precedence."""
    configured = os.environ.get("JAVA_HOME")
    name = "java.exe" if sys.platform == "win32" else "java"
    return str(Path(configured) / "bin" / name) if configured else name


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_mod(path: Path) -> str:
    """Reject source/dev/game JARs and a different dependency tuple."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16_000_000:
        raise ValueError("mod must be a regular project JAR smaller than 16 MB")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names) or any(n.startswith("net/minecraft/") for n in names):
            raise ValueError("mod JAR contains duplicate or Minecraft entries")
        if "dev/neurocraft/fly/NeuroCraftFly.class" not in names:
            raise ValueError("expected the compiled NeuroCraft Fly mod")
        if "LICENSE" not in names or "NOTICE" not in names:
            raise ValueError("mod JAR must include LICENSE and NOTICE")
        for required in ("fabric.mod.json", "dev/neurocraft/fly/NeuroCraftFly.class"):
            if required not in names or archive.getinfo(required).file_size > 1_000_000:
                raise ValueError("mod metadata or entrypoint is missing or excessive")
        metadata = json.loads(archive.read("fabric.mod.json"))
        if not isinstance(metadata, dict) or not isinstance(metadata.get("depends"), dict):
            raise ValueError("mod metadata and dependencies must be objects")
        depends = metadata["depends"]
        if metadata.get("id") != "neurocraft_fly" or any(
            depends.get(key) != value
            for key, value in {
                "minecraft": f"={MINECRAFT}",
                "fabricloader": f"={LOADER}",
                "fabric-api": f"={FABRIC_API}",
            }.items()
        ):
            raise ValueError("mod does not declare the pinned Minecraft/Fabric tuple")
        # Loom's ordinary dev JAR retains Mojang names; only its remapped output is installable.
        class_bytes = archive.read("dev/neurocraft/fly/NeuroCraftFly.class")
        if b"net/minecraft/resources/ResourceLocation" in class_bytes:
            raise ValueError("development JAR detected; use Gradle's remapped build/libs JAR")
        version = metadata.get("version")
        if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9.+-]{1,64}", version):
            raise ValueError("mod version is invalid")
        return version


def _download(name: str, destination: Path) -> None:
    url, expected = DOWNLOADS[name]
    with urllib.request.urlopen(url, timeout=30) as response, destination.open("xb") as stream:
        if response.url != url:
            raise ValueError("pinned download redirected unexpectedly")
        total = 0
        while block := response.read(1024 * 1024):
            total += len(block)
            if total > 16_000_000:
                raise ValueError("pinned download exceeds size limit")
            stream.write(block)
    if digest(destination) != expected:
        raise ValueError(f"pinned {name} digest mismatch")


def _files(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if _redirected(path):
            raise ValueError("runtime must not contain symlinks")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = digest(path)
    return result


def install(output: Path, mod_jar: Path) -> Path:
    """Install into a new private directory, without accepting the EULA or editing profiles."""
    version = validate_mod(mod_jar)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    installer = output / "fabric-installer.jar"
    _download("fabric-installer.jar", installer)
    runtime = output / "server"
    runtime.mkdir()
    subprocess.run(
        [
            java_executable(),
            "-jar",
            str(installer.resolve()),
            "server",
            "-dir",
            str(runtime.resolve()),
            "-mcversion",
            MINECRAFT,
            "-loader",
            LOADER,
            "-downloadMinecraft",
        ],
        check=True,
        timeout=300,
        stdin=subprocess.DEVNULL,
    )
    mods = runtime / "mods"
    mods.mkdir()
    _download("fabric-api.jar", mods / "fabric-api.jar")
    shutil.copyfile(mod_jar, mods / "neurocraft-fly.jar")
    java_runtime.prepare_server(runtime)
    manifest = {
        "schema": "neurocraft.installed_runtime.v2",
        "minecraft": MINECRAFT,
        "loader": LOADER,
        "fabric_api": FABRIC_API,
        "installer": INSTALLER,
        "mod_version": version,
        "scope": "private local runtime; contains Minecraft; do not redistribute",
        "files": _files(runtime),
    }
    (output / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    validate_runtime(output)
    return output / MANIFEST


def validate_runtime(root: Path) -> dict[str, str]:
    """Validate every stored byte, a bounded inventory, and the installable mod."""
    manifest_path = root / MANIFEST
    if _redirected(root) or _redirected(manifest_path) or manifest_path.stat().st_size > 1_000_000:
        raise ValueError("invalid runtime manifest")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("runtime manifest must be an object")
    if (
        manifest.get("schema") != "neurocraft.installed_runtime.v2"
        or manifest.get("minecraft") != MINECRAFT
        or manifest.get("loader") != LOADER
        or manifest.get("fabric_api") != FABRIC_API
    ):
        raise ValueError("runtime manifest has a different pinned tuple")
    files = manifest.get("files")
    if not isinstance(files, dict) or not 4 <= len(files) <= 2000:
        raise ValueError("runtime inventory is missing or excessive")
    for name, expected in files.items():
        if not isinstance(name, str):
            raise ValueError("invalid runtime inventory entry")
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or str(path) != name
            or not _portable_name(name)
            or not isinstance(expected, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
        ):
            raise ValueError("invalid runtime inventory entry")
    runtime = root / "server"
    if _redirected(runtime) or _files(runtime) != files:
        raise ValueError("runtime files changed; prepare a new installation")
    for required in (
        java_runtime.PLAN,
        "mods/fabric-api.jar",
        "mods/neurocraft-fly.jar",
    ):
        if required not in files:
            raise ValueError(f"runtime missing {required}")
    if files["mods/fabric-api.jar"] != DOWNLOADS["fabric-api.jar"][1]:
        raise ValueError("runtime Fabric API differs from the pinned upstream bytes")
    if validate_mod(runtime / "mods" / "neurocraft-fly.jar") != manifest.get("mod_version"):
        raise ValueError("runtime mod version differs from inventory")
    java_runtime.validate_server(runtime)
    return files


def validate_candidate(
    root: Path,
    mod_jar: Path,
    client_mods: Path | None = None,
    *,
    expected_version: str,
) -> dict[str, str]:
    """Match the source version, installed server and optional isolated client.

    This is byte matching, not proof that the selected JAR was built from a
    particular source revision. The release build must establish that separately.
    Client file names may differ; their contents and two-JAR count must match.
    """
    files = validate_runtime(root)
    if validate_mod(mod_jar) != expected_version:
        raise ValueError("built mod version differs from the current source version")
    if digest(mod_jar) != files["mods/neurocraft-fly.jar"]:
        raise ValueError("installed mod differs from the current source-build JAR")
    if client_mods is not None:
        validate_client_mods(files, client_mods)
        java_runtime.validate_client(client_mods)
    return files


def validate_client_mods(files: dict[str, str], client_mods: Path) -> None:
    """Check the mod pair before installing or validating the separate launcher policy."""
    if _redirected(client_mods) or not client_mods.is_dir():
        raise ValueError("client mods path must be a regular isolated mods directory")
    jars = [path for path in client_mods.iterdir() if path.suffix.lower() == ".jar"]
    if len(jars) != 2 or any(path.is_symlink() or not path.is_file() for path in jars):
        raise ValueError("isolated client needs exactly two regular JARs: the mod and API")
    expected = {files["mods/neurocraft-fly.jar"], files["mods/fabric-api.jar"]}
    if {digest(path) for path in jars} != expected:
        raise ValueError("client mod/API bytes differ from the installed server")


def copy_runtime(root: Path, server_dir: Path) -> None:
    """Copy verified installation files only; never carry a world or identity between runs."""
    files = validate_runtime(root)
    if _redirected(server_dir) or not server_dir.is_dir():
        raise ValueError("session server directory must be a regular directory")
    for name in files:
        for ancestor in (server_dir / name).parents:
            if ancestor == server_dir:
                break
            if _redirected(ancestor):
                raise ValueError("session destination contains a symlink")
        if name in ("eula.txt", "server.properties") or (server_dir / name).exists():
            raise ValueError("installed runtime would overwrite session configuration")
        if name.startswith(("world/", "logs/")):
            raise ValueError("installed runtime contains session state")
    for name, expected in files.items():
        destination = server_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with (root / "server" / name).open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target)
        if digest(destination) != expected:
            raise ValueError("runtime changed during copy")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new private runtime directory")
    parser.add_argument("--mod-jar", type=Path, required=True, help="remapped build/libs mod JAR")
    args = parser.parse_args()
    try:
        result = install(args.output.absolute(), args.mod_jar.absolute())
    except (OSError, ValueError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Runtime preparation failed: {exc}\nUse a new output directory to retry.\n")
    print(f"Private runtime prepared: {result}")
    print("Use Companion to install the matching client files and patched launcher version.")
    print("Do not upload this runtime: it contains Minecraft. No EULA was accepted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
