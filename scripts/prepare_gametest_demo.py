#!/usr/bin/env python3
"""Prepare, but do not execute, the fixed external-bridge Fabric GameTest run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from neurocraft_connectome import FixturePin, RuntimeGraph, load_malecns_fixture
from neurocraft_connectome.flight_pin import (
    MALECNS_FLIGHT_FIXTURE_V1_FILE_BYTES,
    MALECNS_FLIGHT_FIXTURE_V1_PIN,
)
from neurocraft_controller import (
    WING_MOTOR_POOL_PROFILE,
    ConnectomeControlHandler,
    ConnectomeController,
    ControllerConfiguration,
    DeclaredTransitionForces,
    FlightConnectomeController,
    FlightControllerConfiguration,
    NamedStimulation,
)
from neurocraft_controller.trained_readout_controller import (
    TrainedReadoutController,
    TrmDemoConfiguration,
)
from neurocraft_protocol import (
    ArtifactIdentities,
    ClaimScope,
    DataTier,
    DeterminismLevel,
    FloatPrecision,
    TraceManifest,
    canonical_json_bytes,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "data" / "fixtures" / "mcns-v1.0-flight-power-v1.json"
STEERING_FIXTURE_PATH = ROOT / "data" / "fixtures" / "mcns-v1.0-flight-steering-v1.json"
ARENA_PATH = ROOT / "experiments" / "arenas" / "external-bridge-gametest-v1.json"

RUN_ID = "external-bridge-gametest-v1"
ENTITY_ID = "c1e5707a-45b8-59b2-b792-4c9b9a546e93"
STEERING_RUN_ID = "external-bridge-steering-v1"
# str(uuid.uuid5(uuid.NAMESPACE_URL, "neurocraft:external-bridge-steering-v1")),
# computed once and pinned as a literal so the steering run keeps one stable,
# deterministic entity identity distinct from the v1 GameTest entity.
STEERING_ENTITY_ID = "056b9dda-7e14-564e-bdc0-1fc40785afa3"
TRM_RUN_ID = "external-bridge-trm-demo-v1"
# str(uuid.uuid5(uuid.NAMESPACE_URL, "neurocraft:external-bridge-trm-demo-v1")),
# computed once and pinned as a literal so the TRM demo run keeps one stable,
# deterministic entity identity distinct from the v1 and steering entities.
TRM_ENTITY_ID = "3ae16116-3d9a-55ab-a1b8-f1d46d274e8f"
TRM_READOUT_ARTIFACT_PATH = ROOT / "data" / "readouts" / "trm-cloned-readout-v2.json"
SESSION_EPOCH = 0
STIMULATION_ID = "sensor-pulse-v1"
STIMULATION_BODY_ID = 77_185
STIMULATION_VALUE = 1.0
BRIDGE_URI = "ws://127.0.0.1:8765/control"
TRACE_NAME = "trace.ndjson"

#: Untracked build products ``make spmv-build`` writes beside the tracked
#: ``_spmv.c``, which lives inside a hashed identity root. They are
#: machine-specific and are never hashed, so a machine that has built the native
#: kernel binds a commit exactly as a clean checkout does; they are named here so
#: the untracked-intruder guard in :func:`_collect_files` lets them through
#: instead of refusing to prepare on a developer machine.
BUNDLE_BUILD_PRODUCT_NAMES = frozenset(
    {
        "_spmv.build.json",
        "_spmv.dylib",
        "_spmv.so",
        "_spmv.dll",
        "_spmv.obj",
        "_spmv.lib",
        "_spmv.exp",
    }
)

LAUNCH_SCHEMA_V1 = "neurocraft.external_bridge_launch.v1"
LAUNCH_SCHEMA_V2 = "neurocraft.external_bridge_launch.v2"
LAUNCH_SCHEMA_V3 = "neurocraft.external_bridge_launch.v3"
PAIR_CONTROLLER_PROFILE = "pair_v1"
TRM_CLONED_READOUT_PROFILE = "trm_cloned_readout_v1"

CIRCUIT_FLIGHT_POWER_V1 = "flight-power-v1"
CIRCUIT_FLIGHT_STEERING_V1 = "flight-steering-v1"
CIRCUIT_TRM_DEMO_V1 = "trm-demo-v1"


@dataclass(frozen=True, slots=True)
class CircuitProfile:
    """Frozen description of one preparable circuit.

    The shipped ``flight-power-v1`` row reproduces today's behavior byte for
    byte: loader defaults, the v1 pair controller, the v1 launch schema, and no
    extra service arguments. The ``flight-steering-v1`` row is the explicit
    opt-in: pinned 128-node fixture, wing-motor flight controller, v2 launch
    schema carrying the profile, and the matching service flag. The
    ``trm-demo-v1`` row is the second opt-in: the same pinned steering fixture
    driven by the trained-readout demo controller, under the v3 launch schema
    that additionally carries the committed readout artifact's digest.
    """

    circuit: str
    fixture_path: Path
    fixture_pin: FixturePin | None
    fixture_file_bytes: int | None
    run_id: str
    entity_id: str
    controller_profile: str
    launch_schema: str
    service_argv_extras: tuple[str, ...]
    readout_artifact_path: Path | None = None

    @property
    def fixture_reference(self) -> str:
        return self.fixture_path.relative_to(ROOT).as_posix()

    def readout_material(self) -> tuple[bytes, dict[str, Any], str]:
        """Read, parse, and hash the committed readout artifact this row pins.

        The digest is always derived from the file at call time; nothing in
        this module caches or pins the artifact hash.
        """

        if self.readout_artifact_path is None:
            raise PreparationError(f"circuit {self.circuit!r} declares no readout artifact")
        try:
            raw = self.readout_artifact_path.read_bytes()
        except OSError as exc:
            raise PreparationError(
                f"could not read the committed readout artifact: {exc.strerror}"
            ) from exc
        try:
            parsed: object = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PreparationError("committed readout artifact is not valid JSON") from exc
        if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
            raise PreparationError("committed readout artifact must be a JSON object")
        return raw, cast(dict[str, Any], parsed), hashlib.sha256(raw).hexdigest()

    def load_graph(self) -> RuntimeGraph:
        if self.fixture_pin is None or self.fixture_file_bytes is None:
            return load_malecns_fixture(self.fixture_path)
        return load_malecns_fixture(
            self.fixture_path,
            expected_pin=self.fixture_pin,
            expected_file_bytes=self.fixture_file_bytes,
        )

    def create_controller(
        self,
        graph: RuntimeGraph,
        *,
        run_id: str,
        entity_id: str,
        session_epoch: int,
    ) -> ConnectomeController:
        if self.controller_profile == PAIR_CONTROLLER_PROFILE:
            return ConnectomeController.create(
                graph,
                run_id=run_id,
                entity_id=entity_id,
                session_epoch=session_epoch,
                configuration=ControllerConfiguration(mode="faithful"),
            )
        if self.controller_profile == TRM_CLONED_READOUT_PROFILE:
            artifact_bytes, artifact_document, artifact_sha256 = self.readout_material()
            return TrainedReadoutController(
                graph,
                run_id=run_id,
                entity_id=entity_id,
                session_epoch=session_epoch,
                artifact_bytes=artifact_bytes,
                artifact_document=artifact_document,
                configuration=TrmDemoConfiguration(readout_artifact_sha256=artifact_sha256),
            )
        return FlightConnectomeController(
            graph,
            run_id=run_id,
            entity_id=entity_id,
            session_epoch=session_epoch,
            configuration=FlightControllerConfiguration(),
        )


CIRCUIT_PROFILES: Mapping[str, CircuitProfile] = {
    CIRCUIT_FLIGHT_POWER_V1: CircuitProfile(
        circuit=CIRCUIT_FLIGHT_POWER_V1,
        fixture_path=FIXTURE_PATH,
        fixture_pin=None,
        fixture_file_bytes=None,
        run_id=RUN_ID,
        entity_id=ENTITY_ID,
        controller_profile=PAIR_CONTROLLER_PROFILE,
        launch_schema=LAUNCH_SCHEMA_V1,
        service_argv_extras=(),
    ),
    CIRCUIT_FLIGHT_STEERING_V1: CircuitProfile(
        circuit=CIRCUIT_FLIGHT_STEERING_V1,
        fixture_path=STEERING_FIXTURE_PATH,
        fixture_pin=MALECNS_FLIGHT_FIXTURE_V1_PIN,
        fixture_file_bytes=MALECNS_FLIGHT_FIXTURE_V1_FILE_BYTES,
        run_id=STEERING_RUN_ID,
        entity_id=STEERING_ENTITY_ID,
        controller_profile=WING_MOTOR_POOL_PROFILE,
        launch_schema=LAUNCH_SCHEMA_V2,
        service_argv_extras=("--controller-profile", WING_MOTOR_POOL_PROFILE),
    ),
    CIRCUIT_TRM_DEMO_V1: CircuitProfile(
        circuit=CIRCUIT_TRM_DEMO_V1,
        fixture_path=STEERING_FIXTURE_PATH,
        fixture_pin=MALECNS_FLIGHT_FIXTURE_V1_PIN,
        fixture_file_bytes=MALECNS_FLIGHT_FIXTURE_V1_FILE_BYTES,
        run_id=TRM_RUN_ID,
        entity_id=TRM_ENTITY_ID,
        controller_profile=TRM_CLONED_READOUT_PROFILE,
        launch_schema=LAUNCH_SCHEMA_V3,
        service_argv_extras=("--controller-profile", TRM_CLONED_READOUT_PROFILE),
        readout_artifact_path=TRM_READOUT_ARTIFACT_PATH,
    ),
}


_OUTPUT_NAMES = (
    "identity.json",
    "trace-manifest.json",
    "transition-forces.json",
    "control-catalog.json",
    "launch.json",
)


class PreparationError(RuntimeError):
    """A deterministic run input is absent, invalid, or inconsistent."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_document(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreparationError(f"could not parse canonical JSON input {path.name}: {exc}") from exc
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise PreparationError(f"canonical JSON input {path.name} must be an object")
    return raw, canonical_json_bytes(raw)


def _arena_material() -> tuple[dict[str, Any], str, str]:
    document, canonical = _canonical_document(ARENA_PATH)
    expected = {
        "arena_id",
        "schema",
        "version",
        "coordinate_frame",
        "minecraft",
        "initial_state",
        "perturbation",
        "declared_forces",
        "control_sequence",
    }
    if set(document) != expected:
        raise PreparationError("tracked external GameTest arena has missing or unknown fields")
    if document["arena_id"] != "external-bridge-gametest-v1":
        raise PreparationError("tracked external GameTest arena ID changed")
    if document["schema"] != "neurocraft.minecraft_gametest_arena.v1":
        raise PreparationError("tracked external GameTest arena schema changed")
    return document, _sha256_bytes(canonical), _sha256_file(ARENA_PATH)


def _is_bundle_noise(path: Path) -> bool:
    """Interpreter and desktop droppings that were never part of any bundle."""

    return (
        "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"} or path.name == ".DS_Store"
    )


def _refuse_source_symlink(path: Path, *, root: Path | None = None) -> None:
    root = ROOT if root is None else root
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            raise PreparationError(f"symlink in identity source: {path.relative_to(root)}")


def _archive_source_files(root: Path | None = None) -> frozenset[str] | None:
    """Verified archive membership, or None for a checkout with its own Git index.

    The accompanying manifest detects changes, not publisher authenticity. Its
    source commit is provenance only; it does not recreate Git history or make
    an archive inherit an earlier execution receipt. Do not cache this check.
    """

    root = ROOT if root is None else root
    git_marker = root / ".git"
    if git_marker.exists() or git_marker.is_symlink():
        return None
    manifest_path = root / "RELEASE_MANIFEST.json"
    _refuse_source_symlink(manifest_path, root=root)
    max_manifest_bytes = 2 * 1024 * 1024
    try:
        with manifest_path.open("rb") as stream:
            encoded = stream.read(max_manifest_bytes + 1)
        if len(encoded) > max_manifest_bytes:
            raise PreparationError("source release manifest exceeds 2 MiB")
        manifest = json.loads(encoded)
    except (OSError, ValueError, UnicodeError) as exc:
        raise PreparationError(
            "identity sources require this checkout's .git or a valid RELEASE_MANIFEST.json"
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != (
        "neurocraft.public-source-candidate.v1"
    ):
        raise PreparationError("unsupported source release manifest schema")
    source = manifest.get("source")
    if (
        not isinstance(source, dict)
        or not isinstance(source.get("commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", source["commit"]) is None
    ):
        raise PreparationError("source release manifest needs a declared source commit")
    records = manifest.get("files")
    if not isinstance(records, list) or not 1 <= len(records) <= 10_000:
        raise PreparationError("source release manifest needs 1-10000 file entries")
    names: set[str] = set()
    folded_names: set[str] = set()
    total_bytes = 0
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "source_path",
            "source_blob",
            "mode",
            "bytes",
            "sha256",
        }:
            raise PreparationError("source release file entry has missing or unknown fields")
        name = record["path"]
        if (
            not isinstance(name, str)
            or not name
            or any(ord(char) < 32 for char in name)
            or "\\" in name
            or ":" in name
            or PurePosixPath(name).is_absolute()
            or PurePosixPath(name).as_posix() != name
            or any(part in {".", "..", ".git"} for part in PurePosixPath(name).parts)
            or name == "RELEASE_MANIFEST.json"
        ):
            raise PreparationError("source release file path must be normalized and relative")
        if name.casefold() in folded_names:
            raise PreparationError("duplicate path in source release manifest")
        size = record["bytes"]
        digest = record["sha256"]
        if (
            type(size) is not int
            or not 0 <= size <= 16 * 1024 * 1024
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or not isinstance(record["mode"], str)
            or record["mode"] not in {"100644", "100755"}
        ):
            raise PreparationError("source release file needs bounded size, SHA-256 and file mode")
        total_bytes += size
        if total_bytes > 96 * 1024 * 1024:
            raise PreparationError("source release file bytes exceed 96 MiB")
        path = root / name
        _refuse_source_symlink(path, root=root)
        if not path.is_file() or path.stat().st_size != size:
            raise PreparationError(f"source release file missing or size changed: {name}")
        if _sha256_file(path) != digest:
            raise PreparationError(f"source release file hash mismatch: {name}")
        names.add(name)
        folded_names.add(name.casefold())
    return frozenset(names)


def execution_source_origin(
    root: Path, source_files: Mapping[str, Mapping[str, str]]
) -> dict[str, Any]:
    """Record actual checkout HEAD or verified archive provenance before compute.

    Call again after compute and compare results as well as the source inventory.
    Archive commits are declarations, never an observed checkout or authentication.
    """
    root = root.resolve()
    marker = root / ".git"
    if marker.exists() or marker.is_symlink():
        try:
            head = subprocess.check_output(
                ["git", "rev-parse", "--verify", "HEAD"], cwd=root, text=True
            ).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise PreparationError("could not read this checkout's Git HEAD") from exc
        if re.fullmatch(r"[0-9a-f]{40}", head) is None:
            raise PreparationError("invalid checkout Git HEAD")
        return {
            "kind": "git_checkout",
            "git_head": head,
            "source_commit": head,
            "release_manifest_sha256": None,
        }

    path = root / "RELEASE_MANIFEST.json"
    _refuse_source_symlink(path, root=root)
    try:
        with path.open("rb") as stream:
            before = stream.read(2 * 1024 * 1024 + 1)
    except OSError as exc:
        raise PreparationError("source archive needs RELEASE_MANIFEST.json before compute") from exc
    if len(before) > 2 * 1024 * 1024:
        raise PreparationError("source release manifest exceeds 2 MiB")
    names = _archive_source_files(root)
    if names is None or _sha256_file(path) != _sha256_bytes(before):
        raise PreparationError("source origin changed while verifying archive")
    for item in source_files.values():
        if set(item) != {"path", "sha256"}:
            raise PreparationError("executed source inventory needs path and SHA-256")
        source_path = Path(item["path"])
        try:
            name = source_path.relative_to(root).as_posix()
        except ValueError as exc:
            raise PreparationError("executed source is outside the verified archive") from exc
        if name not in names or _sha256_file(source_path) != item["sha256"]:
            raise PreparationError(f"executed source is not bound by this release: {name}")
    manifest = json.loads(before)
    return {
        "kind": "verified_source_archive",
        "git_head": None,
        "source_commit": manifest["source"]["commit"],
        "release_manifest_sha256": _sha256_bytes(before),
        "publisher_authenticity_verified": False,
    }


def _tracked_under(roots: Sequence[Path]) -> frozenset[str]:
    """Repository-relative POSIX paths Git tracks under ``roots``.

    Enumerating the index rather than the working tree is what makes a bundle a
    function of the git tree: an untracked build product sitting inside a hashed
    root cannot change the identity a commit hashes to, so two parties reading
    the same source agree whether or not either has run a build.
    """

    if not roots:
        return frozenset()
    command = [
        "git",
        "ls-files",
        "-z",
        "--",
        *(root.relative_to(ROOT).as_posix() for root in roots),
    ]
    try:
        completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
    except (OSError, subprocess.SubprocessError) as error:
        raise PreparationError(f"could not enumerate tracked identity sources: {error}") from error
    return frozenset(entry.decode("utf-8") for entry in completed.stdout.split(b"\0") if entry)


def _collect_files(roots: Iterable[Path], explicit: Iterable[Path] = ()) -> tuple[Path, ...]:
    """Identity sources from the Git index or verified release destination list.

    Git or the archive manifest decides membership. The source tree is
    still walked, for two reasons a pure ``git ls-files`` enumeration would miss.
    An untracked source file inside a hashed root is refused rather than silently
    dropped, because it can be imported at run time and would otherwise leave the
    recorded identity describing a code set that is not the one executing; and a
    tracked file deleted from the working tree is refused rather than hashed as a
    read error.
    """

    root_paths = tuple(roots)
    archive_files = _archive_source_files()
    tracked = (
        _tracked_under(root_paths)
        if archive_files is None
        else frozenset(
            name
            for name in archive_files
            if any((ROOT / name).is_relative_to(root) for root in root_paths)
        )
    )
    resolved: set[Path] = set()
    for root in root_paths:
        _refuse_source_symlink(root)
        if not root.is_dir():
            relative = root.relative_to(ROOT)
            raise PreparationError(f"required source directory is missing: {relative}")
        for path in root.rglob("*"):
            _refuse_source_symlink(path)
            if not path.is_file() or _is_bundle_noise(path):
                continue
            reference = path.relative_to(ROOT).as_posix()
            if reference in tracked:
                resolved.add(path)
            elif path.name not in BUNDLE_BUILD_PRODUCT_NAMES:
                raise PreparationError(
                    f"untracked file inside a hashed identity root: {reference}; "
                    "it must be tracked in Git or listed in the verified release manifest "
                    "unless it is an explicitly permitted build product"
                )
    for reference in sorted(tracked):
        if not (ROOT / reference).is_file():
            raise PreparationError(f"tracked identity source is missing: {reference}")
    for path in explicit:
        _refuse_source_symlink(path)
        if not path.is_file():
            raise PreparationError(f"required identity input is missing: {path.relative_to(ROOT)}")
        if archive_files is not None and path.relative_to(ROOT).as_posix() not in archive_files:
            raise PreparationError(
                f"identity input is not listed in source release: {path.relative_to(ROOT)}"
            )
        resolved.add(path)
    return tuple(sorted(resolved, key=lambda path: path.relative_to(ROOT).as_posix()))


def _file_bundle(paths: Sequence[Path], *, schema: str) -> tuple[str, list[dict[str, str]]]:
    files = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in paths
    ]
    material = {"files": files, "schema": schema}
    return _sha256_bytes(canonical_json_bytes(material)), files


def _code_bundle() -> tuple[str, list[dict[str, str]]]:
    """Identity of the declared execution source set in a checkout or archive.

    Membership comes from Git or a verified release manifest, so permitted build
    products inside a source root do not move the hash. Historical bundles taken
    by walking the whole tree can differ between built and clean machines.
    """

    forbidden_build_logic = (
        ROOT / "mod" / "buildSrc",
        ROOT / "mod" / "build-logic",
    )
    present = [path.relative_to(ROOT).as_posix() for path in forbidden_build_logic if path.exists()]
    settings_text = (ROOT / "mod" / "settings.gradle").read_text(encoding="utf-8")
    if present or re.search(r"\bincludeBuild\s*\(", settings_text):
        raise PreparationError(
            "unbound Gradle build logic is forbidden for evidence preparation: "
            + ", ".join(present or ["mod/settings.gradle includeBuild"])
        )
    roots = (
        ROOT / "packages" / "connectome-data" / "src",
        ROOT / "packages" / "controller-runtime" / "src",
        ROOT / "packages" / "neural-runtime" / "src",
        ROOT / "packages" / "protocol" / "src",
        ROOT / "services" / "neural-engine" / "src",
        ROOT / "mod" / "src",
    )
    explicit = (
        ROOT / "scripts" / "__init__.py",
        ROOT / "scripts" / "prepare_gametest_demo.py",
        ROOT / "scripts" / "run_connectome_service.py",
        ROOT / "scripts" / "run_external_gametest.py",
        ROOT / "scripts" / "public_alpha.py",
        ROOT / "scripts" / "process_lifecycle.py",
        ROOT / "scripts" / "replay_minecraft_trace.py",
        ROOT / "mod" / "build.gradle",
    )
    return _file_bundle(
        _collect_files(roots, explicit),
        schema="neurocraft.gametest_code_bundle.v1",
    )


def _dependency_bundle() -> tuple[str, list[dict[str, str]]]:
    paths = (
        ROOT / ".python-version",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "packages" / "connectome-data" / "pyproject.toml",
        ROOT / "packages" / "controller-runtime" / "pyproject.toml",
        ROOT / "packages" / "experiment-sdk" / "pyproject.toml",
        ROOT / "packages" / "neural-runtime" / "pyproject.toml",
        ROOT / "packages" / "protocol" / "pyproject.toml",
        ROOT / "services" / "neural-engine" / "pyproject.toml",
        ROOT / "mod" / "build.gradle",
        ROOT / "mod" / "gradlew",
        ROOT / "mod" / "settings.gradle",
        ROOT / "mod" / "gradle.properties",
        ROOT / "mod" / "gradle.lockfile",
        ROOT / "mod" / "gradle" / "verification-metadata.xml",
        ROOT / "mod" / "gradle" / "wrapper" / "gradle-wrapper.jar",
        ROOT / "mod" / "gradle" / "wrapper" / "gradle-wrapper.properties",
        ROOT / "release" / "java-runtime-pins.json",
    )
    return _file_bundle(
        _collect_files((), paths),
        schema="neurocraft.gametest_dependency_bundle.v1",
    )


def _gradle_properties() -> dict[str, str]:
    result: dict[str, str] = {}
    path = ROOT / "mod" / "gradle.properties"
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        result[key.strip()] = value.strip()
    expected = {
        "minecraft_version": "1.21.5",
        "loader_version": "0.16.10",
        "fabric_api_version": "0.128.2+1.21.5",
        "loom_version": "1.10.5",
    }
    mismatches = {
        key: (expected_value, result.get(key))
        for key, expected_value in expected.items()
        if result.get(key) != expected_value
    }
    if mismatches:
        raise PreparationError(f"Fabric toolchain differs from the preparation pins: {mismatches}")
    # These versions may be prepared; neither inherits another build's receipt.
    # The manifest records the actual version and the dependency bundle binds
    # gradle.properties bytes, so execution requires a fresh matching bundle.
    preparable_mod_versions = (
        "0.1.0",
        "0.1.0-alpha.1",
        "0.2.0-rc.1",
        "0.2.0-rc.2",
        "0.2.0-rc.3",
        "0.2.0-rc.4",
        "0.2.0-rc.5",
    )
    if result.get("mod_version") not in preparable_mod_versions:
        raise PreparationError(
            f"mod_version must be explicitly pinned for preparation: {preparable_mod_versions}; "
            f"got {result.get('mod_version')!r}"
        )
    return result


def _bundle_reference(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise PreparationError("bundle reference must be a normalized repository-relative path")
    return path.as_posix()


def _default_bundle_reference(output: Path) -> str:
    try:
        relative = output.resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise PreparationError(
            "--bundle-reference is required when --output is outside the repository"
        ) from exc
    return _bundle_reference(relative.as_posix())


def _controller(profile: CircuitProfile) -> ConnectomeController:
    graph = profile.load_graph()
    return profile.create_controller(
        graph,
        run_id=profile.run_id,
        entity_id=profile.entity_id,
        session_epoch=SESSION_EPOCH,
    )


def _trace_manifest(
    controller: ConnectomeController,
    *,
    forces: DeclaredTransitionForces,
    execution_code_bundle_hash: str,
    dependency_lock_hash: str,
    machine_label: str,
    os_label: str,
    versions: Mapping[str, str],
) -> TraceManifest:
    graph: RuntimeGraph = controller.graph
    python_version = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    if not python_version:
        raise PreparationError(".python-version is empty")
    return TraceManifest(
        run_id=controller.identity.run_id,
        initial_session_epoch=controller.identity.session_epoch,
        mode=controller.run_mode,
        data_tier=DataTier.MALECNS_FIXTURE,
        claim_scope=ClaimScope.G3,
        artifacts=ArtifactIdentities(
            source_hash=graph.manifest.source_manifest_sha256,
            canonical_graph_hash=graph.manifest.canonical_graph_sha256,
            runtime_graph_hash=graph.manifest.runtime_identity_sha256,
            fixture_hash=graph.manifest.fixture_file_sha256,
            code_hash=execution_code_bundle_hash,
            dependency_lock_hash=dependency_lock_hash,
            adapter_hash=controller.identity.adapter_hash,
            dynamics_hash=controller.identity.dynamics_hash,
            sign_hash=controller.sign_overlay.identity_sha256,
            decoder_hash=controller.identity.decoder_hash,
            arena_hash=forces.identity_sha256,
            configuration_hash=controller.configuration.identity_sha256,
            observation_schema_hash=controller.identity.observation_schema_hash,
        ),
        game_tick_hz=20.0,
        neural_step_ms=controller.dynamics.neural_step_ms,
        precision=FloatPrecision.FLOAT64,
        backend=controller.dynamics.backend,
        thread_count=1,
        seed=0,
        determinism_level=DeterminismLevel.SAME_BUILD_EXACT,
        machine=machine_label,
        os=os_label,
        python=python_version,
        java="21",
        minecraft=versions["minecraft_version"],
        fabric=(
            f"Loader {versions['loader_version']}; "
            f"API {versions['fabric_api_version']}; Loom {versions['loom_version']}"
        ),
        mod=f"neurocraft-fly {versions['mod_version']}",
        privacy_policy="local_trace_no_secrets_v1",
        expected_chunks=1,
        checkpoint_interval_ticks=100,
    )


def _encoded(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _launch_document(
    *,
    profile: CircuitProfile,
    bundle_reference: str,
    controller: ConnectomeController,
    stimulation: NamedStimulation,
    arena_material_hash: str,
    arena_file_hash: str,
    forces: DeclaredTransitionForces,
    execution_code_bundle_hash: str,
    code_files: list[dict[str, str]],
    dependency_hash: str,
    dependency_files: list[dict[str, str]],
    prepared_hashes: Mapping[str, str],
) -> dict[str, Any]:
    identity = controller.identity
    bundle = PurePosixPath(bundle_reference)
    trace_path = (bundle / TRACE_NAME).as_posix()
    manifest_path = (bundle / "trace-manifest.json").as_posix()
    forces_path = (bundle / "transition-forces.json").as_posix()
    service_command = [
        "uv",
        "--no-config",
        "run",
        "--isolated",
        "--exact",
        "--no-dev",
        "--project",
        ".",
        "--frozen",
        "--no-python-downloads",
        "--python",
        ".venv/bin/python3",
        "python",
        "-I",
        "scripts/run_connectome_service.py",
        "--fixture",
        profile.fixture_reference,
        "--run-id",
        identity.run_id,
        "--entity-id",
        identity.entity_id,
        "--session-epoch",
        str(SESSION_EPOCH),
        "--mode",
        "faithful",
        *profile.service_argv_extras,
        "--declare-stimulation",
        f"{STIMULATION_ID}={STIMULATION_BODY_ID}={STIMULATION_VALUE:.1f}",
        "--host",
        "127.0.0.1",
        "--port",
        "8765",
        "--trace-out",
        trace_path,
        "--trace-manifest",
        manifest_path,
        "--transition-forces",
        forces_path,
    ]
    gradle_command = [
        "./gradlew",
        "--no-daemon",
        "--dependency-verification",
        "strict",
        "-Dneurocraft.fly.externalBridgeTest=true",
        f"-Dneurocraft.fly.bridgeUri={BRIDGE_URI}",
        f"-Dneurocraft.fly.runId={identity.run_id}",
        f"-Dneurocraft.fly.sessionEpoch={identity.session_epoch}",
        f"-Dneurocraft.fly.entityId={identity.entity_id}",
        f"-Dneurocraft.fly.graphHash={identity.graph_hash}",
        f"-Dneurocraft.fly.adapterHash={identity.adapter_hash}",
        f"-Dneurocraft.fly.dynamicsHash={identity.dynamics_hash}",
        f"-Dneurocraft.fly.decoderHash={identity.decoder_hash}",
        f"-Dneurocraft.fly.observationSchemaHash={identity.observation_schema_hash}",
        f"-Dneurocraft.fly.arenaMaterialHash={arena_material_hash}",
        f"-Dneurocraft.fly.stimulationId={stimulation.content_id}",
        f"-Dneurocraft.fly.stimulationHash={stimulation.content_hash}",
        "clean",
        "runExternalBridgeGameTest",
    ]
    document: dict[str, Any] = {
        "schema": profile.launch_schema,
        "state": "prepared_not_executed",
        "claim_scope": {
            "manifest_target": "G3",
            "status": "not_passed_by_preparation",
        },
        "identity": identity.to_dict(),
        "bundle_reference": bundle_reference,
        "credential_environment_variable": "NEUROCRAFT_RUN_TOKEN",
        "credential_embedded": False,
        "bridge_uri": BRIDGE_URI,
        "stimulation": {
            "content_id": stimulation.content_id,
            "content_hash": stimulation.content_hash,
            "body_id": STIMULATION_BODY_ID,
            "value": STIMULATION_VALUE,
        },
        "arena": {
            "tracked_path": ARENA_PATH.relative_to(ROOT).as_posix(),
            "canonical_material_sha256": arena_material_hash,
            "file_sha256": arena_file_hash,
            "declared_force_identity_sha256": forces.identity_sha256,
        },
        "identity_bundles": {
            "code": {
                "schema": "neurocraft.gametest_code_bundle.v1",
                "sha256": execution_code_bundle_hash,
                "files": code_files,
            },
            "dependencies": {
                "schema": "neurocraft.gametest_dependency_bundle.v1",
                "sha256": dependency_hash,
                "files": dependency_files,
            },
        },
        "prepared_file_sha256": dict(sorted(prepared_hashes.items())),
        "commands": {
            "service": {
                "working_directory": "repository_root",
                "argv": service_command,
                "requires_out_of_band_environment": ["NEUROCRAFT_RUN_TOKEN"],
            },
            "gametest": {
                "working_directory": "mod",
                "argv": gradle_command,
                "requires_out_of_band_environment": ["NEUROCRAFT_RUN_TOKEN"],
            },
        },
        "expected_generated_outputs": [
            trace_path,
            "mod/build/receipts/external-bridge-gametest.json",
        ],
    }
    if profile.launch_schema == LAUNCH_SCHEMA_V2:
        # The one v2 addition: the controller profile becomes explicit. The v1
        # schema stays byte-identical and never carries the key.
        document["controller_profile"] = profile.controller_profile
    if profile.launch_schema == LAUNCH_SCHEMA_V3:
        # The one v3 addition over v2: the committed readout artifact's digest,
        # derived from the file at preparation time so the orchestrator can
        # recompute it and refuse a swapped weights file. v1 and v2 launches
        # never carry either key beyond their own exact sets.
        document["controller_profile"] = profile.controller_profile
        document["readout_artifact_sha256"] = profile.readout_material()[2]
    return document


def prepare_bundle(
    output: Path,
    *,
    bundle_reference: str,
    machine_label: str = "local-macos-arm64",
    os_label: str = "macOS",
    circuit: str = CIRCUIT_FLIGHT_POWER_V1,
) -> dict[str, str]:
    """Create all preparation documents in a new directory and return file hashes."""

    profile = CIRCUIT_PROFILES.get(circuit)
    if profile is None:
        raise PreparationError(f"unknown circuit: {circuit!r}")
    bundle_reference = _bundle_reference(bundle_reference)
    controller = _controller(profile)
    stimulation = NamedStimulation.create(
        controller,
        STIMULATION_ID,
        {STIMULATION_BODY_ID: STIMULATION_VALUE},
    )
    handler = ConnectomeControlHandler(controller, (stimulation,))
    arena_document, arena_material_hash, arena_file_hash = _arena_material()
    declared_forces = arena_document["declared_forces"]
    if not isinstance(declared_forces, dict):
        raise PreparationError("arena declared_forces must be an object")
    passive = declared_forces.get("passive_force_vector")
    external = declared_forces.get("external_force_vector")
    provenance = declared_forces.get("provenance")
    if not isinstance(passive, list) or not isinstance(external, list):
        raise PreparationError("arena force vectors must be arrays")
    if not isinstance(provenance, str):
        raise PreparationError("arena force provenance must be text")
    forces = DeclaredTransitionForces.create(
        passive_forces=tuple(passive),
        external_forces=tuple(external),
        provenance=provenance,
        arena_material_hash=arena_material_hash,
    )
    execution_code_bundle_hash, code_files = _code_bundle()
    dependency_hash, dependency_files = _dependency_bundle()
    versions = _gradle_properties()
    manifest = _trace_manifest(
        controller,
        forces=forces,
        execution_code_bundle_hash=execution_code_bundle_hash,
        dependency_lock_hash=dependency_hash,
        machine_label=machine_label,
        os_label=os_label,
        versions=versions,
    )

    transition_document: dict[str, Any] = {
        "schema": "neurocraft.declared_transition_forces.v1",
        "passive_forces": list(forces.passive_forces),
        "external_forces": list(forces.external_forces),
        "provenance": forces.provenance,
        "arena_material_hash": arena_material_hash,
    }
    catalog = {
        "schema": "neurocraft.control_catalog.v1",
        "interventions": [
            {"content_id": content_id, "content_hash": content_hash}
            for content_id, content_hash in sorted(handler.content_hashes.items())
        ],
    }
    initial: dict[str, bytes] = {
        "identity.json": _encoded(controller.identity.to_dict()),
        "trace-manifest.json": _encoded(manifest.to_dict()),
        "transition-forces.json": _encoded(transition_document),
        "control-catalog.json": _encoded(catalog),
    }
    prepared_hashes = {name: _sha256_bytes(payload) for name, payload in initial.items()}
    launch = _launch_document(
        profile=profile,
        bundle_reference=bundle_reference,
        controller=controller,
        stimulation=stimulation,
        arena_material_hash=arena_material_hash,
        arena_file_hash=arena_file_hash,
        forces=forces,
        execution_code_bundle_hash=execution_code_bundle_hash,
        code_files=code_files,
        dependency_hash=dependency_hash,
        dependency_files=dependency_files,
        prepared_hashes=prepared_hashes,
    )
    documents = {**initial, "launch.json": _encoded(launch)}

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir(mode=0o755)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to clobber existing output directory: {output}") from exc
    try:
        for name in _OUTPUT_NAMES:
            _write_exclusive(output / name, documents[name])
    except BaseException:
        shutil.rmtree(output)
        raise
    return {name: _sha256_bytes(documents[name]) for name in _OUTPUT_NAMES}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--circuit",
        choices=tuple(CIRCUIT_PROFILES),
        default=CIRCUIT_FLIGHT_POWER_V1,
        help=(
            "flight-power-v1 keeps the shipped v1 preparation byte for byte; "
            "flight-steering-v1 opts in to the wing-motor flight controller; "
            "trm-demo-v1 opts in to the trained-readout demo controller"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="defaults to artifacts/gametest/<run-id> for the selected circuit",
    )
    parser.add_argument(
        "--bundle-reference",
        help="repository-relative path written into launch commands; inferred for in-repo outputs",
    )
    parser.add_argument("--machine-label", default="local-macos-arm64")
    parser.add_argument("--os-label", default="macOS")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = CIRCUIT_PROFILES[args.circuit]
    output: Path | None = args.output
    if output is None:
        output = ROOT / "artifacts" / "gametest" / profile.run_id
    try:
        reference = (
            _default_bundle_reference(output)
            if args.bundle_reference is None
            else _bundle_reference(args.bundle_reference)
        )
        hashes = prepare_bundle(
            output,
            bundle_reference=reference,
            machine_label=args.machine_label,
            os_label=args.os_label,
            circuit=args.circuit,
        )
    except (FileExistsError, PreparationError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    summary = {
        "schema": "neurocraft.gametest_preparation_summary.v1",
        "state": "prepared_not_executed",
        "output": output.as_posix(),
        "file_sha256": hashes,
    }
    sys.stdout.buffer.write(_encoded(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
