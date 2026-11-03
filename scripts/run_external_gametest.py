#!/usr/bin/env python3
"""Execute and verify one prepared external Fabric bridge GameTest bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pwd
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast
from urllib.parse import urlsplit

# ``python scripts/run_external_gametest.py`` places ``scripts/`` rather than the
# repository root on sys.path. Add only this script's resolved repository root so
# the replay module has the same import identity under direct CLI and test use.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurocraft_controller import (
    WING_MOTOR_POOL_PROFILE,
    ConnectomeController,
    ControllerIntegrityError,
    DeclaredTransitionForces,
    FlightConnectomeController,
    NamedStimulation,
)
from neurocraft_controller.trained_readout_controller import (
    TrainedReadoutController,
    TrmDemoConfiguration,
)
from neurocraft_protocol import (
    PROTOCOL_VERSION,
    CausalRecord,
    ClaimScope,
    DataTier,
    RunIdentity,
    TraceManifest,
    TraceVerification,
    canonical_json_bytes,
    verify_trace,
)

from scripts.prepare_gametest_demo import (
    CIRCUIT_PROFILES,
    LAUNCH_SCHEMA_V1,
    LAUNCH_SCHEMA_V2,
    LAUNCH_SCHEMA_V3,
    PAIR_CONTROLLER_PROFILE,
    TRM_CLONED_READOUT_PROFILE,
    CircuitProfile,
    PreparationError,
)
from scripts.prepare_gametest_demo import (
    _code_bundle as _current_code_bundle,
)
from scripts.prepare_gametest_demo import (
    _dependency_bundle as _current_dependency_bundle,
)
from scripts.process_lifecycle import terminate_posix
from scripts.public_alpha import UV_VERSION
from scripts.replay_minecraft_trace import (
    iter_records,
    replay_minecraft_trace,
)

ROOT = Path(__file__).resolve().parents[1]
MOD_ROOT = ROOT / "mod"
DEFAULT_BUNDLE = ROOT / "artifacts" / "gametest" / "external-bridge-gametest-v1"
GAME_RECEIPT = ROOT / "mod" / "build" / "receipts" / "external-bridge-gametest.json"
REPLAY_RECEIPT_NAME = "replay-receipt.json"
COMBINED_RECEIPT_NAME = "external-bridge-run-receipt.json"
EXPECTED_PACING_MODE = "external_gametest_probe_monotonic_20hz"
EXPECTED_PACING_PERIOD_NS = 50_000_000

#: Each controller profile appears in exactly one prepared circuit, so a launch
#: document's profile resolves to the same frozen row the preparer used.
_PROFILE_BY_CONTROLLER: dict[str, CircuitProfile] = {
    profile.controller_profile: profile for profile in CIRCUIT_PROFILES.values()
}

_HASH = re.compile(r"[0-9a-f]{64}")
_TOKEN = re.compile(r"[A-Za-z0-9_-]{32,256}")
_MAX_JSON_BYTES = 32 * 1024 * 1024
_CHILD_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
_FORBIDDEN_ENV_EXACT = {
    "BASH_ENV",
    "CDPATH",
    "CLASSPATH",
    "ENV",
    "GROOVY_OPTS",
    "UV_BUILD_CONSTRAINT",
    "UV_CONFIG_FILE",
    "UV_CONSTRAINT",
    "UV_DEFAULT_INDEX",
    "UV_ENV_FILE",
    "UV_EXTRA_INDEX_URL",
    "UV_FIND_LINKS",
    "UV_INDEX",
    "UV_OVERRIDE",
    "UV_PROJECT_ENVIRONMENT",
    "UV_PYTHON",
    "_JAVA_OPTIONS",
}
_FORBIDDEN_ENV_PREFIXES = (
    "DYLD_",
    "GRADLE_",
    "JAVA_",
    "JDK_",
    "LD_",
    "PYTHON",
)


class OrchestrationError(RuntimeError):
    """A prepared command or generated receipt failed a closed validation gate."""


@dataclass(frozen=True, slots=True)
class StimulationDeclaration:
    content_id: str
    content_hash: str
    body_id: int
    value: float


@dataclass(frozen=True, slots=True)
class ValidatedBundle:
    bundle: Path
    bundle_reference: str
    launch_path: Path
    launch_sha256: str
    prepared_file_sha256: Mapping[str, str]
    identity: RunIdentity
    manifest: TraceManifest
    forces: DeclaredTransitionForces
    stimulation: StimulationDeclaration
    controller_profile: str
    service_argv: tuple[str, ...]
    gradle_argv: tuple[str, ...]
    host: str
    port: int
    fixture: Path
    trace: Path
    gametest_receipt: Path
    replay_receipt: Path
    combined_receipt: Path
    code_bundle_sha256: str
    dependency_bundle_sha256: str


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise OrchestrationError(f"{name} must be a JSON object")
    return cast(dict[str, object], value)


def _array(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise OrchestrationError(f"{name} must be a JSON array")
    return cast(list[object], value)


def _exact(value: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise OrchestrationError(f"{name} has missing or unknown fields")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise OrchestrationError(f"{name} must be non-empty text")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise OrchestrationError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise OrchestrationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise OrchestrationError(f"{name} must be a finite number")
    return result


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise OrchestrationError(f"{name} must be boolean")
    return value


def _hash(value: object, name: str) -> str:
    result = _text(value, name)
    if _HASH.fullmatch(result) is None:
        raise OrchestrationError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _string_array(value: object, name: str) -> tuple[str, ...]:
    raw = _array(value, name)
    if not all(isinstance(item, str) and item for item in raw):
        raise OrchestrationError(f"{name} must contain only non-empty strings")
    return tuple(cast(list[str], raw))


def _read_json(path: Path, *, canonical: bool) -> tuple[dict[str, object], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise OrchestrationError(f"could not read {path.name}: {exc.strerror}") from exc
    if not raw or len(raw) > _MAX_JSON_BYTES:
        raise OrchestrationError(f"{path.name} is empty or exceeds the JSON size bound")
    try:
        parsed: object = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OrchestrationError(f"could not parse {path.name} as JSON") from exc
    document = _object(parsed, path.name)
    if canonical and raw != canonical_json_bytes(document) + b"\n":
        raise OrchestrationError(f"{path.name} is not newline-terminated canonical JSON")
    return document, raw


def _repo_path(reference: object, name: str, *, must_exist: bool = False) -> Path:
    text = _text(reference, name)
    if "\\" in text:
        raise OrchestrationError(f"{name} must use normalized POSIX separators")
    relative = PurePosixPath(text)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise OrchestrationError(f"{name} must be a normalized repository-relative path")
    result = (ROOT / Path(*relative.parts)).resolve()
    try:
        result.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise OrchestrationError(f"{name} resolves outside the repository") from exc
    if must_exist and not result.exists():
        raise OrchestrationError(f"{name} does not exist")
    return result


def _verify_file_bundle(
    value: object,
    *,
    name: str,
    schema: str,
) -> tuple[str, dict[str, str]]:
    document = _object(value, name)
    expected = {"schema", "sha256", "files"}
    _exact(document, expected, name)
    if document["schema"] != schema:
        raise OrchestrationError(f"{name}.schema changed")
    expected_bundle_hash = _hash(document["sha256"], f"{name}.sha256")
    files: list[dict[str, str]] = []
    observed: dict[str, str] = {}
    for index, raw_entry in enumerate(_array(document["files"], f"{name}.files")):
        entry = _object(raw_entry, f"{name}.files[{index}]")
        _exact(entry, {"path", "sha256"}, f"{name}.files[{index}]")
        path_text = _text(entry["path"], f"{name}.files[{index}].path")
        if path_text in observed:
            raise OrchestrationError(f"{name}.files contains a duplicate path")
        expected_hash = _hash(entry["sha256"], f"{name}.files[{index}].sha256")
        path = _repo_path(path_text, f"{name}.files[{index}].path", must_exist=True)
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise OrchestrationError(f"current file does not match {name}: {path_text}")
        observed[path_text] = expected_hash
        files.append({"path": path_text, "sha256": expected_hash})
    paths = list(observed)
    if not paths or paths != sorted(paths):
        raise OrchestrationError(f"{name}.files must be non-empty and sorted")
    material = {"files": files, "schema": schema}
    if _sha256_bytes(canonical_json_bytes(material)) != expected_bundle_hash:
        raise OrchestrationError(f"{name} aggregate hash is invalid")
    return expected_bundle_hash, observed


def _entries_by_path(entries: Sequence[Mapping[str, str]]) -> dict[str, str]:
    return {entry["path"]: entry["sha256"] for entry in entries}


def _vector(value: object, name: str) -> tuple[float, float, float]:
    items = _array(value, name)
    if len(items) != 3:
        raise OrchestrationError(f"{name} must contain exactly three values")
    values = tuple(_number(item, f"{name}[{index}]") for index, item in enumerate(items))
    return cast(tuple[float, float, float], values)


def validate_bundle(
    bundle: Path,
    *,
    enforce_bundle_location: bool = True,
) -> ValidatedBundle:
    """Validate every executable input in a prepared bundle before process launch."""

    bundle = bundle.resolve()
    if not bundle.is_dir():
        raise OrchestrationError("prepared bundle directory does not exist")
    launch_path = bundle / "launch.json"
    launch, launch_raw = _read_json(launch_path, canonical=True)
    base_launch_keys = {
        "schema",
        "state",
        "claim_scope",
        "identity",
        "bundle_reference",
        "credential_environment_variable",
        "credential_embedded",
        "bridge_uri",
        "stimulation",
        "arena",
        "identity_bundles",
        "prepared_file_sha256",
        "commands",
        "expected_generated_outputs",
    }
    schema = launch.get("schema")
    if schema == LAUNCH_SCHEMA_V1:
        # v1 has no profile key; it always means the shipped pair readout.
        _exact(launch, base_launch_keys, "launch.json")
        controller_profile = PAIR_CONTROLLER_PROFILE
    elif schema == LAUNCH_SCHEMA_V2:
        # v2 adds exactly one key: the explicit controller profile. The pair
        # profile is never prepared under v2, so only the wing profile is
        # allow-listed here.
        _exact(launch, base_launch_keys | {"controller_profile"}, "launch.json")
        controller_profile = _text(launch["controller_profile"], "launch.controller_profile")
        if controller_profile != WING_MOTOR_POOL_PROFILE:
            raise OrchestrationError(
                "launch.controller_profile is not an allow-listed v2 controller profile"
            )
    elif schema == LAUNCH_SCHEMA_V3:
        # v3 adds exactly one key over v2: the committed readout artifact's
        # digest. Only the trained-readout profile is ever prepared under v3,
        # so only it is allow-listed here.
        _exact(
            launch,
            base_launch_keys | {"controller_profile", "readout_artifact_sha256"},
            "launch.json",
        )
        controller_profile = _text(launch["controller_profile"], "launch.controller_profile")
        if controller_profile != TRM_CLONED_READOUT_PROFILE:
            raise OrchestrationError(
                "launch.controller_profile is not an allow-listed v3 controller profile"
            )
    else:
        raise OrchestrationError("unsupported launch schema")
    profile = _PROFILE_BY_CONTROLLER[controller_profile]
    trm_material: tuple[bytes, dict[str, object], str] | None = None
    if controller_profile == TRM_CLONED_READOUT_PROFILE:
        # Recompute the artifact digest from the committed file and compare it
        # with the launch's declared digest, so a swapped weights file and a
        # tampered launch both fail closed before any process starts.
        try:
            trm_material = profile.readout_material()
        except PreparationError as exc:
            raise OrchestrationError(str(exc)) from exc
        if trm_material[2] != _hash(
            launch["readout_artifact_sha256"], "launch.readout_artifact_sha256"
        ):
            raise OrchestrationError(
                "committed readout artifact does not match launch.readout_artifact_sha256"
            )
    if launch["state"] != "prepared_not_executed":
        raise OrchestrationError("launch state is not prepared_not_executed")
    claim = _object(launch["claim_scope"], "launch.claim_scope")
    _exact(claim, {"manifest_target", "status"}, "launch.claim_scope")
    if claim != {"manifest_target": "G3", "status": "not_passed_by_preparation"}:
        raise OrchestrationError("launch claim boundary changed")
    if launch["credential_environment_variable"] != "NEUROCRAFT_RUN_TOKEN":
        raise OrchestrationError("launch credential environment variable changed")
    if _boolean(launch["credential_embedded"], "launch.credential_embedded"):
        raise OrchestrationError("launch claims to embed a credential")

    bundle_reference = _text(launch["bundle_reference"], "launch.bundle_reference")
    referenced_bundle = _repo_path(bundle_reference, "launch.bundle_reference")
    if enforce_bundle_location and bundle != referenced_bundle:
        raise OrchestrationError("bundle path does not match launch.bundle_reference")
    fixture_reference = profile.fixture_reference
    fixture = _repo_path(fixture_reference, "service fixture", must_exist=True)

    prepared = _object(launch["prepared_file_sha256"], "launch.prepared_file_sha256")
    expected_prepared_names = {
        "identity.json",
        "trace-manifest.json",
        "transition-forces.json",
        "control-catalog.json",
    }
    _exact(prepared, expected_prepared_names, "launch.prepared_file_sha256")
    prepared_hashes: dict[str, str] = {}
    documents: dict[str, dict[str, object]] = {}
    for name in sorted(expected_prepared_names):
        expected_hash = _hash(prepared[name], f"launch.prepared_file_sha256.{name}")
        document, raw = _read_json(bundle / name, canonical=True)
        if _sha256_bytes(raw) != expected_hash:
            raise OrchestrationError(f"prepared file hash mismatch: {name}")
        prepared_hashes[name] = expected_hash
        documents[name] = document

    identity = RunIdentity.from_dict(documents["identity.json"])
    launch_identity = RunIdentity.from_dict(_object(launch["identity"], "launch.identity"))
    if launch_identity != identity:
        raise OrchestrationError("launch and identity.json identities differ")
    # The profile row pins its run/entity identifiers; a bundle whose identity
    # borrows another profile's labels must not validate even when every
    # cryptographic hash inside it is truthful.
    if identity.run_id != profile.run_id or identity.entity_id != profile.entity_id:
        raise OrchestrationError(
            "identity run_id/entity_id do not match the launch controller profile"
        )
    if identity.session_epoch != 0:
        raise OrchestrationError("prepared bundles declare session epoch 0")
    manifest = TraceManifest.from_dict(documents["trace-manifest.json"])
    if (
        manifest.run_id != identity.run_id
        or manifest.initial_session_epoch != identity.session_epoch
        or manifest.claim_scope is not ClaimScope.G3
        or manifest.data_tier is not DataTier.MALECNS_FIXTURE
    ):
        raise OrchestrationError("trace manifest does not bind the prepared G3 fixture identity")
    artifacts = manifest.artifacts
    if (
        artifacts.runtime_graph_hash != identity.graph_hash
        or artifacts.adapter_hash != identity.adapter_hash
        or artifacts.dynamics_hash != identity.dynamics_hash
        or artifacts.decoder_hash != identity.decoder_hash
        or artifacts.observation_schema_hash != identity.observation_schema_hash
    ):
        raise OrchestrationError("trace manifest artifact identities differ from RunIdentity")

    stimulation_raw = _object(launch["stimulation"], "launch.stimulation")
    _exact(
        stimulation_raw,
        {"content_id", "content_hash", "body_id", "value"},
        "launch.stimulation",
    )
    stimulation = StimulationDeclaration(
        content_id=_text(stimulation_raw["content_id"], "launch.stimulation.content_id"),
        content_hash=_hash(stimulation_raw["content_hash"], "launch.stimulation.content_hash"),
        body_id=_integer(stimulation_raw["body_id"], "launch.stimulation.body_id", minimum=1),
        value=_number(stimulation_raw["value"], "launch.stimulation.value"),
    )
    catalog = documents["control-catalog.json"]
    _exact(catalog, {"schema", "interventions"}, "control-catalog.json")
    if catalog["schema"] != "neurocraft.control_catalog.v1":
        raise OrchestrationError("unsupported control catalog schema")
    catalog_items = _array(catalog["interventions"], "control-catalog.interventions")
    if len(catalog_items) != 1:
        raise OrchestrationError("external GameTest requires exactly one named intervention")
    catalog_item = _object(catalog_items[0], "control-catalog.interventions[0]")
    _exact(catalog_item, {"content_id", "content_hash"}, "control-catalog intervention")
    if catalog_item != {
        "content_id": stimulation.content_id,
        "content_hash": stimulation.content_hash,
    }:
        raise OrchestrationError("control catalog does not bind the declared stimulation")
    graph = profile.load_graph()
    controller: ConnectomeController
    if controller_profile == PAIR_CONTROLLER_PROFILE:
        controller = ConnectomeController.create(
            graph,
            run_id=identity.run_id,
            entity_id=identity.entity_id,
            session_epoch=identity.session_epoch,
        )
    elif controller_profile == TRM_CLONED_READOUT_PROFILE:
        if trm_material is None:  # pragma: no cover - profile dispatch guarantees this
            raise OrchestrationError("trained-readout material was not resolved")
        artifact_bytes, artifact_document, artifact_sha256 = trm_material
        try:
            controller = TrainedReadoutController(
                graph,
                run_id=identity.run_id,
                entity_id=identity.entity_id,
                session_epoch=identity.session_epoch,
                artifact_bytes=artifact_bytes,
                artifact_document=artifact_document,
                configuration=TrmDemoConfiguration(readout_artifact_sha256=artifact_sha256),
            )
        except ControllerIntegrityError as exc:
            raise OrchestrationError(
                f"committed readout artifact failed controller integrity: {exc}"
            ) from exc
    else:
        controller = FlightConnectomeController(
            graph,
            run_id=identity.run_id,
            entity_id=identity.entity_id,
            session_epoch=identity.session_epoch,
        )
    named_stimulation = NamedStimulation.create(
        controller,
        stimulation.content_id,
        {stimulation.body_id: stimulation.value},
    )
    if (
        controller.identity != identity
        or named_stimulation.content_hash != stimulation.content_hash
    ):
        raise OrchestrationError("current controller does not reproduce the prepared identities")
    if (
        artifacts.source_hash != graph.manifest.source_manifest_sha256
        or artifacts.canonical_graph_hash != graph.manifest.canonical_graph_sha256
        or artifacts.runtime_graph_hash != graph.manifest.runtime_identity_sha256
        or artifacts.fixture_hash != _sha256_file(fixture)
        or artifacts.sign_hash != controller.sign_overlay.identity_sha256
        or artifacts.configuration_hash != controller.configuration.identity_sha256
    ):
        raise OrchestrationError("current fixture/controller does not reproduce the manifest")

    force_document = documents["transition-forces.json"]
    _exact(
        force_document,
        {"schema", "passive_forces", "external_forces", "provenance", "arena_material_hash"},
        "transition-forces.json",
    )
    if force_document["schema"] != "neurocraft.declared_transition_forces.v1":
        raise OrchestrationError("unsupported transition-force schema")
    forces = DeclaredTransitionForces.create(
        passive_forces=_vector(force_document["passive_forces"], "forces.passive_forces"),
        external_forces=_vector(force_document["external_forces"], "forces.external_forces"),
        provenance=_text(force_document["provenance"], "forces.provenance"),
        arena_material_hash=_hash(
            force_document["arena_material_hash"], "forces.arena_material_hash"
        ),
    )

    arena = _object(launch["arena"], "launch.arena")
    _exact(
        arena,
        {
            "tracked_path",
            "canonical_material_sha256",
            "file_sha256",
            "declared_force_identity_sha256",
        },
        "launch.arena",
    )
    arena_path = _repo_path(arena["tracked_path"], "launch.arena.tracked_path", must_exist=True)
    arena_document, arena_raw = _read_json(arena_path, canonical=False)
    arena_material_hash = _sha256_bytes(canonical_json_bytes(arena_document))
    if (
        arena_material_hash
        != _hash(arena["canonical_material_sha256"], "launch.arena.canonical_material_sha256")
        or _sha256_bytes(arena_raw) != _hash(arena["file_sha256"], "launch.arena.file_sha256")
        or forces.arena_material_hash != arena_material_hash
        or forces.identity_sha256
        != _hash(
            arena["declared_force_identity_sha256"],
            "launch.arena.declared_force_identity_sha256",
        )
        or artifacts.arena_hash != forces.identity_sha256
    ):
        raise OrchestrationError("arena, force, and manifest identities are inconsistent")

    bundles = _object(launch["identity_bundles"], "launch.identity_bundles")
    _exact(bundles, {"code", "dependencies"}, "launch.identity_bundles")
    code_hash, code_files = _verify_file_bundle(
        bundles["code"],
        name="launch.identity_bundles.code",
        schema="neurocraft.gametest_code_bundle.v1",
    )
    dependency_hash, dependency_files = _verify_file_bundle(
        bundles["dependencies"],
        name="launch.identity_bundles.dependencies",
        schema="neurocraft.gametest_dependency_bundle.v1",
    )
    required_dependency_files = {
        "uv.lock",
        "mod/gradle/verification-metadata.xml",
        "mod/gradle/wrapper/gradle-wrapper.jar",
        "mod/gradle/wrapper/gradle-wrapper.properties",
    }
    if not required_dependency_files.issubset(dependency_files):
        raise OrchestrationError("dependency bundle omits a required strict-build input")
    if artifacts.dependency_lock_hash != dependency_hash:
        raise OrchestrationError("manifest dependency identity does not bind the full build tuple")
    required_code_files = {
        "scripts/__init__.py",
        "scripts/prepare_gametest_demo.py",
        "scripts/run_connectome_service.py",
        "scripts/run_external_gametest.py",
        "scripts/replay_minecraft_trace.py",
    }
    missing_code_files = required_code_files.difference(code_files)
    if missing_code_files:
        raise OrchestrationError(
            f"code bundle omits required execution or verification code: "
            f"{sorted(missing_code_files)}"
        )
    if code_hash != artifacts.code_hash:
        raise OrchestrationError("manifest code hash differs from the prepared execution bundle")
    # Both bundles are now enumerated from the Git index (D-072), so they refuse
    # an untracked file inside a hashed root. That is a closed validation gate
    # like any other here and must reach the operator as one, not as a traceback.
    try:
        current_code_hash, current_code_entries = _current_code_bundle()
        current_dependency_hash, current_dependency_entries = _current_dependency_bundle()
    except PreparationError as exc:
        raise OrchestrationError(str(exc)) from exc
    if current_code_hash != code_hash or _entries_by_path(current_code_entries) != code_files:
        raise OrchestrationError(
            "code bundle is not the exact current execution source set; re-prepare the run"
        )
    if (
        current_dependency_hash != dependency_hash
        or _entries_by_path(current_dependency_entries) != dependency_files
    ):
        raise OrchestrationError(
            "dependency bundle is not the exact current build input set; re-prepare the run"
        )

    bridge_uri = _text(launch["bridge_uri"], "launch.bridge_uri")
    uri = urlsplit(bridge_uri)
    if (
        uri.scheme != "ws"
        or uri.hostname != "127.0.0.1"
        or uri.path != "/control"
        or uri.username is not None
        or uri.password is not None
        or uri.query
        or uri.fragment
        or uri.port is None
    ):
        raise OrchestrationError("bridge URI must be an unauthenticated IPv4 loopback /control URI")
    host = "127.0.0.1"
    port = uri.port
    trace_reference = (PurePosixPath(bundle_reference) / "trace.ndjson").as_posix()
    expected_service = (
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
        fixture_reference,
        "--run-id",
        identity.run_id,
        "--entity-id",
        identity.entity_id,
        "--session-epoch",
        str(identity.session_epoch),
        "--mode",
        "faithful",
        *profile.service_argv_extras,
        "--declare-stimulation",
        f"{stimulation.content_id}={stimulation.body_id}={stimulation.value:.1f}",
        "--host",
        host,
        "--port",
        str(port),
        "--trace-out",
        trace_reference,
        "--trace-manifest",
        (PurePosixPath(bundle_reference) / "trace-manifest.json").as_posix(),
        "--transition-forces",
        (PurePosixPath(bundle_reference) / "transition-forces.json").as_posix(),
    )
    expected_gradle = (
        "./gradlew",
        "--no-daemon",
        "--dependency-verification",
        "strict",
        "-Dneurocraft.fly.externalBridgeTest=true",
        f"-Dneurocraft.fly.bridgeUri={bridge_uri}",
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
    )
    commands = _object(launch["commands"], "launch.commands")
    _exact(commands, {"service", "gametest"}, "launch.commands")
    for command_name, cwd, expected_argv in (
        ("service", "repository_root", expected_service),
        ("gametest", "mod", expected_gradle),
    ):
        command = _object(commands[command_name], f"launch.commands.{command_name}")
        _exact(
            command,
            {"working_directory", "argv", "requires_out_of_band_environment"},
            f"launch.commands.{command_name}",
        )
        if command["working_directory"] != cwd:
            raise OrchestrationError(f"launch.commands.{command_name} working directory changed")
        if _string_array(command["argv"], f"launch.commands.{command_name}.argv") != expected_argv:
            raise OrchestrationError(f"launch.commands.{command_name}.argv is not allow-listed")
        if _string_array(
            command["requires_out_of_band_environment"],
            f"launch.commands.{command_name}.requires_out_of_band_environment",
        ) != ("NEUROCRAFT_RUN_TOKEN",):
            raise OrchestrationError(f"launch.commands.{command_name} environment contract changed")

    expected_outputs = (trace_reference, "mod/build/receipts/external-bridge-gametest.json")
    if _string_array(launch["expected_generated_outputs"], "launch.expected_generated_outputs") != (
        expected_outputs
    ):
        raise OrchestrationError("launch expected output list changed")
    trace = _repo_path(trace_reference, "launch trace output")
    if enforce_bundle_location and trace != bundle / "trace.ndjson":
        raise OrchestrationError("trace output is not inside the prepared bundle")

    return ValidatedBundle(
        bundle=bundle,
        bundle_reference=bundle_reference,
        launch_path=launch_path,
        launch_sha256=_sha256_bytes(launch_raw),
        prepared_file_sha256=prepared_hashes,
        identity=identity,
        manifest=manifest,
        forces=forces,
        stimulation=stimulation,
        controller_profile=controller_profile,
        service_argv=expected_service,
        gradle_argv=expected_gradle,
        host=host,
        port=port,
        fixture=fixture,
        trace=trace,
        gametest_receipt=GAME_RECEIPT,
        replay_receipt=bundle / REPLAY_RECEIPT_NAME,
        combined_receipt=bundle / COMBINED_RECEIPT_NAME,
        code_bundle_sha256=code_hash,
        dependency_bundle_sha256=dependency_hash,
    )


def _require_run_token(environment: Mapping[str, str]) -> str:
    token = environment.get("NEUROCRAFT_RUN_TOKEN")
    if token is None or _TOKEN.fullmatch(token) is None:
        raise OrchestrationError("NEUROCRAFT_RUN_TOKEN must contain 32-256 URL-safe characters")
    return token


def _require_outputs_absent(bundle: ValidatedBundle) -> None:
    for path in (
        bundle.trace,
        bundle.gametest_receipt,
        bundle.replay_receipt,
        bundle.combined_receipt,
    ):
        if path.exists():
            raise FileExistsError(f"refusing to clobber existing run output: {path.name}")


def _command_output(
    argv: Sequence[str],
    *,
    environment: Mapping[str, str],
) -> str:
    try:
        completed = subprocess.run(
            list(argv),
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=10.0,
            shell=False,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise OrchestrationError(f"could not verify execution tool: {argv[0]}") from exc
    return completed.stdout.decode("utf-8", errors="strict").strip()


def _execution_environment(
    source: Mapping[str, str],
    *,
    temporary_directory: Path,
) -> tuple[dict[str, str], dict[str, object], str]:
    """Construct and describe the minimal environment used by both child processes."""

    injected = sorted(
        key
        for key in source
        if key in _FORBIDDEN_ENV_EXACT
        or any(key.startswith(prefix) for prefix in _FORBIDDEN_ENV_PREFIXES)
    )
    if injected:
        raise OrchestrationError(
            f"refusing execution with environment-injection variables set: {injected}"
        )
    _require_private_temporary_directory(temporary_directory)
    token = _require_run_token(source)
    uv_path_text = shutil.which("uv", path=source.get("PATH"))
    if uv_path_text is None:
        raise OrchestrationError("could not resolve the pinned uv executable")
    uv_path = Path(uv_path_text).resolve()
    python_path = (ROOT / ".venv" / "bin" / "python3").resolve()
    if not uv_path.is_file() or not python_path.is_file():
        raise OrchestrationError("pinned uv or project Python executable is missing")

    login_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    gradle_home = login_home / ".gradle"
    forbidden_gradle_files = [
        gradle_home / "init.gradle",
        gradle_home / "init.gradle.kts",
        gradle_home / "gradle.properties",
    ]
    init_directory = gradle_home / "init.d"
    if init_directory.is_dir():
        forbidden_gradle_files.extend(path for path in init_directory.rglob("*") if path.is_file())
    present_gradle_files = [path for path in forbidden_gradle_files if path.is_file()]
    if present_gradle_files:
        raise OrchestrationError(
            "Gradle user-home init/config files are forbidden for evidence runs"
        )

    probe_environment = {
        "HOME": str(login_home),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": _CHILD_PATH,
        "TMPDIR": str(temporary_directory),
    }
    uv_version = _command_output((str(uv_path), "--version"), environment=probe_environment)
    if not uv_version.startswith(f"uv {UV_VERSION} "):
        raise OrchestrationError(f"uv executable does not match required version {UV_VERSION}")
    python_version = _command_output(
        (str(python_path), "-c", "import platform; print(platform.python_version())"),
        environment=probe_environment,
    )
    required_python = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    if python_version != required_python:
        raise OrchestrationError("project Python executable does not match .python-version")

    java_home_tool = Path("/usr/libexec/java_home")
    if java_home_tool.is_file():
        java_home = Path(
            _command_output((str(java_home_tool), "-v", "21"), environment=probe_environment)
        ).resolve()
    else:
        java_candidate = shutil.which("java", path=_CHILD_PATH)
        if java_candidate is None:
            raise OrchestrationError("could not resolve Java 21")
        java_home = Path(java_candidate).resolve().parents[1]
    java_path = java_home / "bin" / "java"
    if not java_path.is_file():
        raise OrchestrationError("resolved Java home has no bin/java")
    java_version = _command_output(
        (str(java_path), "-version"),
        environment=probe_environment,
    )
    if not re.search(r'version "21(?:\.|\")', java_version):
        raise OrchestrationError("resolved Java executable is not Java 21")

    child = {
        "GRADLE_USER_HOME": str(gradle_home),
        "HOME": str(login_home),
        "JAVA_HOME": str(java_home),
        "LANG": "C",
        "LC_ALL": "C",
        "NEUROCRAFT_RUN_TOKEN": token,
        "PATH": _CHILD_PATH,
        "PYTHONHASHSEED": "0",
        "TMPDIR": str(temporary_directory),
        "UV_CACHE_DIR": str(ROOT / ".uv-cache"),
        "UV_NO_CONFIG": "1",
        # This option is produced here after rejecting all caller-supplied JVM
        # injection variables. The wrapper, Gradle daemon and GameTest JVM all
        # inherit it; TMPDIR alone does not establish Java's effective property.
        "JAVA_TOOL_OPTIONS": f'-Djava.io.tmpdir="{temporary_directory}"',
    }
    _verify_java_temporary_directory(java_path, child, temporary_directory)
    receipt: dict[str, object] = {
        "schema": "neurocraft.execution_environment.v2",
        "policy": "minimal_allow_list_no_user_init",
        "forwarded_secret": "NEUROCRAFT_RUN_TOKEN (value omitted)",
        "child_keys": sorted(child),
        "path": _CHILD_PATH,
        "uv_cache": "repository/.uv-cache",
        "gradle_user_home": "login-home/.gradle; init and user properties absent",
        "temporary_directory": {
            "path": "<private-run-directory>",
            "mode": "0700",
            "owner": "execution-user",
            "unique_per_run": True,
            "java_option": "-Djava.io.tmpdir=<private-run-directory>",
            "actual_jvm_property_verified": True,
            "lifecycle": "removed after owned groups stop; retained if shutdown is unconfirmed",
        },
        "toolchain": {
            "uv": {"version": uv_version, "sha256": _sha256_file(uv_path)},
            "python": {"version": python_version, "sha256": _sha256_file(python_path)},
            "java": {"version": java_version.splitlines()[0], "sha256": _sha256_file(java_path)},
            "gradle_wrapper": {
                "sha256": _sha256_file(ROOT / "mod" / "gradlew"),
                "wrapper_jar_sha256": _sha256_file(
                    ROOT / "mod" / "gradle" / "wrapper" / "gradle-wrapper.jar"
                ),
            },
        },
    }
    return child, receipt, str(uv_path)


def _require_private_temporary_directory(directory: Path) -> None:
    """Reject a replaced, shared or non-owner temporary directory before JNI use."""

    metadata = directory.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or directory != directory.resolve()
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or any(character in str(directory) for character in ('"', "'", "\r", "\n"))
    ):
        raise OrchestrationError("JVM temporary directory must be owner-only mode 0700")
    for parent in directory.parents:
        parent_mode = parent.stat().st_mode
        if parent_mode & 0o022 and not parent_mode & stat.S_ISVTX:
            raise OrchestrationError("JVM temporary directory has an unsafe writable ancestor")


def _verify_java_temporary_directory(
    java_executable: Path,
    environment: Mapping[str, str],
    directory: Path,
) -> None:
    """Observe the effective property in a real, otherwise harmless child JVM."""

    _require_private_temporary_directory(directory)
    settings = _command_output(
        (str(java_executable), "-XshowSettings:properties", "-version"),
        environment=environment,
    )
    values = re.findall(r"(?m)^\s*java\.io\.tmpdir = (.+)$", settings)
    if len(values) != 1 or Path(values[0].strip()).resolve() != directory:
        raise OrchestrationError("child JVM did not use the private java.io.tmpdir")


def _tcp_ready(host: str, port: int, *, timeout: float = 0.15) -> bool:
    """Probe TCP readiness without transmitting WebSocket or protocol bytes."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        stream.settimeout(timeout)
        return stream.connect_ex((host, port)) == 0


def _wait_for_service(
    process: subprocess.Popen[bytes],
    *,
    host: str,
    port: int,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise OrchestrationError(
                f"neural service exited before readiness with status {return_code}"
            )
        if _tcp_ready(host, port):
            return
        time.sleep(0.05)
    raise OrchestrationError("neural service did not become ready before the timeout")


class ProcessCleanupError(OrchestrationError):
    """Owned children could not be confirmed stopped; retain their private scratch."""


def _terminate_child(process: subprocess.Popen[bytes], *, timeout: float) -> None:
    terminate_posix(process, timeout=timeout)


def _run_processes(
    bundle: ValidatedBundle,
    *,
    environment: Mapping[str, str],
    readiness_timeout: float,
    gametest_timeout: float,
    termination_timeout: float,
) -> dict[str, object]:
    if _tcp_ready(bundle.host, bundle.port):
        raise OrchestrationError("refusing to use a bridge port that already has a listener")
    private_directory = Path(tempfile.mkdtemp(prefix="neurocraft-external-")).resolve()
    cleanup_confirmed = True
    try:
        return _run_processes_in_private_directory(
            bundle,
            environment=environment,
            temporary_directory=private_directory,
            readiness_timeout=readiness_timeout,
            gametest_timeout=gametest_timeout,
            termination_timeout=termination_timeout,
        )
    except ProcessCleanupError:
        cleanup_confirmed = False
        raise
    finally:
        if cleanup_confirmed:
            shutil.rmtree(private_directory)


def _run_processes_in_private_directory(
    bundle: ValidatedBundle,
    *,
    environment: Mapping[str, str],
    temporary_directory: Path,
    readiness_timeout: float,
    gametest_timeout: float,
    termination_timeout: float,
) -> dict[str, object]:
    child_environment, execution_receipt, uv_executable = _execution_environment(
        environment, temporary_directory=temporary_directory
    )
    service_argv = [uv_executable, *bundle.service_argv[1:]]
    service = subprocess.Popen(
        service_argv,
        cwd=ROOT,
        env=child_environment,
        stdin=subprocess.DEVNULL,
        # Keep sanitized service diagnostics visible to the invoking terminal so
        # fail-closed protocol errors are actionable. Service configuration hides
        # the token in reprs and never logs its value.
        stdout=None,
        stderr=None,
        start_new_session=True,
        shell=False,
    )
    gradle: subprocess.Popen[bytes] | None = None
    try:
        _wait_for_service(
            service,
            host=bundle.host,
            port=bundle.port,
            timeout=readiness_timeout,
        )
        gradle = subprocess.Popen(
            list(bundle.gradle_argv),
            cwd=MOD_ROOT,
            env=child_environment,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            shell=False,
        )
        try:
            returncode = gradle.wait(timeout=gametest_timeout)
        except subprocess.TimeoutExpired as exc:
            raise OrchestrationError("external Fabric GameTest exceeded its timeout") from exc
        if returncode != 0:
            raise OrchestrationError(f"external Fabric GameTest failed with status {returncode}")
        if service.poll() is not None:
            raise OrchestrationError("neural service exited before GameTest completion")
    finally:
        cleanup_errors = []
        for child in (gradle, service):
            if child is not None:
                try:
                    _terminate_child(child, timeout=termination_timeout)
                except BaseException as exc:
                    cleanup_errors.append(exc)
        if cleanup_errors:
            raise ProcessCleanupError(
                "owned child shutdown was not confirmed; private scratch retained"
            ) from cleanup_errors[0]
    return execution_receipt


_GAME_KEYS = {
    "schema",
    "protocol",
    "entity_id",
    "arena_material_hash",
    "bridge_state",
    "pacing_mode",
    "pacing_period_ns",
    "open_forward_ray_blocks",
    "blocked_forward_ray_blocks",
    "obstacle_block",
    "open_command_seq",
    "blocked_command_seq",
    "blocked_observation_tick",
    "non_neutral_command_seq",
    "neural_step_start",
    "neural_step_end",
    "state_digest",
    "applied_motors",
    "record_start",
    "stimulation_apply",
    "stimulation_clear",
    "record_stop",
    "trace_record_expectation",
    "post_stop_observations",
    "initial_position_blocks",
    "final_position_blocks",
    "world_transition_observed",
    "actuation_receipts_sent",
    "last_receipt_command_seq",
    "completed_game_tick",
}


def _verify_gametest_pacing(receipt: Mapping[str, object]) -> None:
    if _text(receipt.get("pacing_mode"), "receipt.pacing_mode") != EXPECTED_PACING_MODE:
        raise OrchestrationError("GameTest receipt has the wrong pacing mode")
    if (
        _integer(receipt.get("pacing_period_ns"), "receipt.pacing_period_ns", minimum=1)
        != EXPECTED_PACING_PERIOD_NS
    ):
        raise OrchestrationError("GameTest receipt has the wrong pacing period")


def _receipt_ack(
    value: object,
    name: str,
    *,
    control_seq: int,
    state: str | None,
    stimulation: StimulationDeclaration | None,
) -> dict[str, object]:
    document = _object(value, name)
    expected = {"control_seq", "issued_tick", "effective_tick"}
    if state is not None:
        expected.add("state")
    if stimulation is not None:
        expected.update({"content_id", "content_hash"})
    _exact(document, expected, name)
    if _integer(document["control_seq"], f"{name}.control_seq") != control_seq:
        raise OrchestrationError(f"{name} has the wrong control sequence")
    issued = _integer(document["issued_tick"], f"{name}.issued_tick")
    effective = _integer(document["effective_tick"], f"{name}.effective_tick")
    if effective < issued:
        raise OrchestrationError(f"{name} effective tick precedes its issue tick")
    if state is not None and document["state"] != state:
        raise OrchestrationError(f"{name} has the wrong control state")
    if stimulation is not None and (
        document["content_id"] != stimulation.content_id
        or document["content_hash"] != stimulation.content_hash
    ):
        raise OrchestrationError(f"{name} does not bind the declared stimulation")
    return document


def _verify_gametest_receipt(
    bundle: ValidatedBundle,
    verification: TraceVerification,
) -> tuple[dict[str, object], str]:
    receipt, raw = _read_json(bundle.gametest_receipt, canonical=False)
    _exact(receipt, _GAME_KEYS, "GameTest receipt")
    if (
        receipt["schema"] != "neurocraft.external_bridge_gametest.v1"
        or receipt["protocol"] != PROTOCOL_VERSION
        or receipt["entity_id"] != bundle.identity.entity_id
        or receipt["arena_material_hash"] != bundle.forces.arena_material_hash
        or receipt["bridge_state"] != "ACTIVE"
    ):
        raise OrchestrationError("GameTest receipt does not bind the prepared active bridge")
    _verify_gametest_pacing(receipt)
    open_ray = _number(receipt["open_forward_ray_blocks"], "receipt.open_forward_ray_blocks")
    blocked_ray = _number(
        receipt["blocked_forward_ray_blocks"], "receipt.blocked_forward_ray_blocks"
    )
    if not 0.0 < blocked_ray < open_ray:
        raise OrchestrationError("GameTest obstacle did not shorten a positive forward ray")
    obstacle = _array(receipt["obstacle_block"], "receipt.obstacle_block")
    if len(obstacle) != 3 or any(
        isinstance(value, bool) or not isinstance(value, int) for value in obstacle
    ):
        raise OrchestrationError("GameTest obstacle block is invalid")

    open_sequence = _integer(receipt["open_command_seq"], "receipt.open_command_seq")
    blocked_sequence = _integer(receipt["blocked_command_seq"], "receipt.blocked_command_seq")
    blocked_observation_tick = _integer(
        receipt["blocked_observation_tick"],
        "receipt.blocked_observation_tick",
    )
    non_neutral_sequence = _integer(
        receipt["non_neutral_command_seq"], "receipt.non_neutral_command_seq"
    )
    if not open_sequence < blocked_sequence or non_neutral_sequence < open_sequence:
        raise OrchestrationError("GameTest command evidence is not causally ordered")
    neural_start = _integer(receipt["neural_step_start"], "receipt.neural_step_start")
    neural_end = _integer(receipt["neural_step_end"], "receipt.neural_step_end")
    if neural_end <= neural_start:
        raise OrchestrationError("GameTest neural interval did not advance")
    state_digest = _hash(receipt["state_digest"], "receipt.state_digest")

    motor_names = {
        "forward_thrust",
        "lateral_thrust",
        "vertical_thrust",
        "yaw_torque",
        "pitch_torque",
    }
    motors_raw = _object(receipt["applied_motors"], "receipt.applied_motors")
    _exact(motors_raw, motor_names, "receipt.applied_motors")
    motors = {
        name: _number(motors_raw[name], f"receipt.applied_motors.{name}")
        for name in sorted(motor_names)
    }
    if any(abs(value) > 1.0 for value in motors.values()) or not any(
        value != 0.0 for value in motors.values()
    ):
        raise OrchestrationError("GameTest motor evidence is neutral or out of bounds")

    _receipt_ack(
        receipt["record_start"],
        "receipt.record_start",
        control_seq=0,
        state="RECORDING",
        stimulation=None,
    )
    _receipt_ack(
        receipt["stimulation_apply"],
        "receipt.stimulation_apply",
        control_seq=1,
        state=None,
        stimulation=bundle.stimulation,
    )
    _receipt_ack(
        receipt["stimulation_clear"],
        "receipt.stimulation_clear",
        control_seq=2,
        state=None,
        stimulation=bundle.stimulation,
    )
    _receipt_ack(
        receipt["record_stop"],
        "receipt.record_stop",
        control_seq=3,
        state="RUNNING",
        stimulation=None,
    )

    expectation = _object(receipt["trace_record_expectation"], "receipt.trace_record_expectation")
    _exact(
        expectation,
        {
            "content_id",
            "first_command_seq",
            "last_command_seq",
            "minimum_action_records",
            "contains_stimulation_round_trip",
        },
        "receipt.trace_record_expectation",
    )
    if expectation["content_id"] != "external-bridge-gametest" or not _boolean(
        expectation["contains_stimulation_round_trip"],
        "receipt.trace_record_expectation.contains_stimulation_round_trip",
    ):
        raise OrchestrationError("GameTest did not declare the required recorded intervention")
    first_expected = _integer(
        expectation["first_command_seq"], "receipt.trace_record_expectation.first_command_seq"
    )
    last_expected = _integer(
        expectation["last_command_seq"], "receipt.trace_record_expectation.last_command_seq"
    )
    minimum_records = _integer(
        expectation["minimum_action_records"],
        "receipt.trace_record_expectation.minimum_action_records",
        minimum=1,
    )
    if first_expected != open_sequence or last_expected < blocked_sequence:
        raise OrchestrationError("GameTest trace expectation does not cover its causal actions")
    if minimum_records != last_expected - first_expected + 1:
        raise OrchestrationError("GameTest trace expectation has an invalid action count")

    records = tuple(iter_records(bundle.trace))
    command_sequences = tuple(record.command_seq for record in records)
    if (
        len(records) != verification.record_count
        or len(records) < minimum_records
        or not command_sequences
        or command_sequences[0] != first_expected
        or command_sequences[-1] != last_expected
        or any(sequence is None for sequence in command_sequences)
    ):
        raise OrchestrationError("verified trace does not satisfy the GameTest record expectation")
    open_records = [record for record in records if record.command_seq == open_sequence]
    blocked_records = [record for record in records if record.command_seq == blocked_sequence]
    if len(open_records) != 1 or len(blocked_records) != 1:
        raise OrchestrationError("GameTest ray evidence is not uniquely bound to traced commands")
    open_record = open_records[0]
    blocked_record = blocked_records[0]
    if (
        blocked_record.observation.game_tick != blocked_observation_tick
        or not math.isclose(
            open_record.observation.sensors.ray_distances_m[0],
            open_ray,
            rel_tol=0.0,
            abs_tol=5.1e-10,
        )
        or not math.isclose(
            blocked_record.observation.sensors.ray_distances_m[0],
            blocked_ray,
            rel_tol=0.0,
            abs_tol=5.1e-10,
        )
    ):
        raise OrchestrationError("GameTest ray evidence does not bind its traced observations")
    matching = [record for record in records if record.command_seq == non_neutral_sequence]
    if len(matching) != 1:
        raise OrchestrationError("GameTest non-neutral command is not unique in the trace")
    recorded = matching[0]
    if (
        recorded.state_after_digest != state_digest
        or recorded.neural_step_start != neural_start
        or recorded.neural_step_end != neural_end
        or any(
            not math.isclose(
                recorded.applied_action.to_dict()[name],
                value,
                rel_tol=0.0,
                abs_tol=5.1e-10,
            )
            for name, value in motors.items()
        )
    ):
        raise OrchestrationError("GameTest neural/action evidence differs from the trace")
    if not any(bundle.stimulation.content_id in record.interventions for record in records):
        raise OrchestrationError("trace contains no record under the declared stimulation")

    initial_position = tuple(
        _number(value, f"receipt.initial_position_blocks[{index}]")
        for index, value in enumerate(
            _array(receipt["initial_position_blocks"], "receipt.initial_position_blocks")
        )
    )
    final_position = tuple(
        _number(value, f"receipt.final_position_blocks[{index}]")
        for index, value in enumerate(
            _array(receipt["final_position_blocks"], "receipt.final_position_blocks")
        )
    )
    if (
        len(initial_position) != 3
        or len(final_position) != 3
        or initial_position == final_position
        or not _boolean(receipt["world_transition_observed"], "receipt.world_transition_observed")
    ):
        raise OrchestrationError("GameTest receipt lacks an authoritative world transition")
    post_stop = _integer(receipt["post_stop_observations"], "receipt.post_stop_observations")
    receipts_sent = _integer(receipt["actuation_receipts_sent"], "receipt.actuation_receipts_sent")
    last_receipt = _integer(receipt["last_receipt_command_seq"], "receipt.last_receipt_command_seq")
    _integer(receipt["completed_game_tick"], "receipt.completed_game_tick", minimum=1)
    if post_stop < 5 or receipts_sent < verification.record_count or last_receipt < last_expected:
        raise OrchestrationError("GameTest receipt coverage is incomplete")
    return receipt, _sha256_bytes(raw)


def _atomic_write_canonical(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to clobber existing run output: {path.name}")
    payload = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    return _sha256_bytes(payload)


def _summarize_operational_telemetry(
    records: Sequence[CausalRecord],
) -> dict[str, object]:
    """Validate and summarize measured per-action service telemetry."""
    if not records:
        raise OrchestrationError("trace has no operational telemetry records")
    for index, record in enumerate(records):
        if record.compute_ns <= 0 or record.serialization_ns <= 0 or record.write_ns <= 0:
            raise OrchestrationError(
                f"trace record {index} lacks measured operational timing telemetry"
            )

    def timing(name: str) -> dict[str, int]:
        values = [cast(int, getattr(record, name)) for record in records]
        return {"min": min(values), "max": max(values), "total": sum(values)}

    return {
        "record_count": len(records),
        "queue_coalesced_total": sum(record.queue_coalesced for record in records),
        "queue_overruns_total": sum(record.queue_overruns for record in records),
        "compute_ns": timing("compute_ns"),
        "serialization_ns": timing("serialization_ns"),
        "write_ns": timing("write_ns"),
    }


def _verify_outputs(
    bundle: ValidatedBundle,
    *,
    token: str,
    execution_environment: Mapping[str, object],
) -> dict[str, Any]:
    verification = verify_trace(bundle.trace)
    if verification.record_count < 1 or verification.manifest != bundle.manifest:
        raise OrchestrationError("trace is empty or does not use the prepared manifest")
    gametest, gametest_hash = _verify_gametest_receipt(bundle, verification)
    records = tuple(iter_records(bundle.trace))
    operational_telemetry = _summarize_operational_telemetry(records)
    replay = replay_minecraft_trace(
        fixture=bundle.fixture,
        trace=bundle.trace,
        stimulations={
            bundle.stimulation.content_id: {
                bundle.stimulation.body_id: bundle.stimulation.value,
            }
        },
        controller_profile=bundle.controller_profile,
    )
    counterfactual = replay.get("zero_recurrence_counterfactual")
    if (
        replay.get("schema") != "neurocraft.minecraft_trace_replay.v1"
        or replay.get("status") != "exact_neural_match"
        or replay.get("replay_implementation_sha256")
        != _sha256_file(ROOT / "scripts" / "replay_minecraft_trace.py")
        or replay.get("trace_file_sha256") != verification.file_sha256
        or replay.get("record_count") != verification.record_count
        or replay.get("stimulated_record_count", 0) < 1
        or not isinstance(counterfactual, dict)
        or counterfactual.get("record_count") != verification.record_count
        or not isinstance(counterfactual.get("state_divergent_record_count"), int)
        or counterfactual["state_divergent_record_count"] < 1
        or not isinstance(counterfactual.get("action_divergent_record_count"), int)
        or counterfactual["action_divergent_record_count"] < 1
    ):
        raise OrchestrationError("neural replay receipt is incomplete or inconsistent")
    replay_hash = _atomic_write_canonical(bundle.replay_receipt, replay)
    combined: dict[str, Any] = {
        "schema": "neurocraft.external_bridge_run_receipt.v1",
        "status": "integration_tested_and_exactly_replayed",
        "evidence_boundary": (
            "One opt-in Fabric GameTest exercised the prepared MaleCNS-fixture controller, "
            "custom fly, named stimulation round trip, receipt-gated causal trace, and "
            "same-build exact neural replay. This receipt does not independently promote G3, "
            "establish real-time qualification, or constitute a launcher observation."
        ),
        "bundle_reference": bundle.bundle_reference,
        "identity": bundle.identity.to_dict(),
        "launch_sha256": bundle.launch_sha256,
        "prepared_file_sha256": dict(sorted(bundle.prepared_file_sha256.items())),
        "identity_bundles": {
            "code_sha256": bundle.code_bundle_sha256,
            "dependencies_sha256": bundle.dependency_bundle_sha256,
            "orchestrator_code_sha256": _sha256_file(Path(__file__)),
            "replay_verifier_sha256": _sha256_file(ROOT / "scripts" / "replay_minecraft_trace.py"),
        },
        "commands": {
            "service_argv": list(bundle.service_argv),
            "gametest_argv": list(bundle.gradle_argv),
            "dependency_verification": "strict",
            "credential_source": "NEUROCRAFT_RUN_TOKEN environment (value omitted)",
        },
        "execution_environment": dict(execution_environment),
        "gametest": {"file_sha256": gametest_hash, "receipt": gametest},
        "trace": {
            "file_sha256": verification.file_sha256,
            "manifest_hash": verification.manifest_hash,
            "final_hash": verification.final_hash,
            "record_count": verification.record_count,
            "first_tick": verification.first_tick,
            "last_tick": verification.last_tick,
        },
        "operational_telemetry": operational_telemetry,
        "replay": {"file_sha256": replay_hash, "receipt": replay},
    }
    encoded = canonical_json_bytes(combined) + b"\n"
    if token.encode("ascii") in encoded:
        raise OrchestrationError("refusing to write a receipt containing the run credential")
    combined_hash = _atomic_write_canonical(bundle.combined_receipt, combined)
    return {
        "schema": "neurocraft.external_bridge_orchestrator_summary.v1",
        "status": "verified",
        "receipt": (PurePosixPath(bundle.bundle_reference) / COMBINED_RECEIPT_NAME).as_posix(),
        "receipt_sha256": combined_hash,
        "trace_sha256": verification.file_sha256,
        "record_count": verification.record_count,
    }


def run_bundle(
    bundle_path: Path,
    *,
    readiness_timeout: float,
    gametest_timeout: float,
    termination_timeout: float,
) -> dict[str, Any]:
    bundle = validate_bundle(bundle_path)
    token = _require_run_token(os.environ)
    _require_outputs_absent(bundle)
    execution_environment = _run_processes(
        bundle,
        environment=os.environ,
        readiness_timeout=readiness_timeout,
        gametest_timeout=gametest_timeout,
        termination_timeout=termination_timeout,
    )
    return _verify_outputs(
        bundle,
        token=token,
        execution_environment=execution_environment,
    )


def _positive_timeout(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timeout must be numeric") from exc
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("timeout must be finite and positive")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--readiness-timeout", type=_positive_timeout, default=30.0)
    parser.add_argument("--gametest-timeout", type=_positive_timeout, default=900.0)
    parser.add_argument("--termination-timeout", type=_positive_timeout, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_bundle(
            args.bundle,
            readiness_timeout=args.readiness_timeout,
            gametest_timeout=args.gametest_timeout,
            termination_timeout=args.termination_timeout,
        )
    except (FileExistsError, OSError, OrchestrationError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    sys.stdout.buffer.write(canonical_json_bytes(summary) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
