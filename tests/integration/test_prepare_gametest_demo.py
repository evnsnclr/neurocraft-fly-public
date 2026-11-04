from __future__ import annotations

import json
from pathlib import Path

import pytest
from neurocraft_connectome import load_malecns_fixture
from neurocraft_controller import (
    ConnectomeController,
    DeclaredTransitionForces,
    NamedStimulation,
)
from neurocraft_protocol import ClaimScope, RunIdentity, TraceManifest, canonical_json_bytes

from scripts import prepare_gametest_demo as preparer
from scripts.prepare_gametest_demo import (
    ARENA_PATH,
    BUNDLE_BUILD_PRODUCT_NAMES,
    CIRCUIT_FLIGHT_STEERING_V1,
    CIRCUIT_TRM_DEMO_V1,
    ENTITY_ID,
    FIXTURE_PATH,
    RUN_ID,
    SESSION_EPOCH,
    STEERING_ENTITY_ID,
    STEERING_FIXTURE_PATH,
    STEERING_RUN_ID,
    STIMULATION_BODY_ID,
    STIMULATION_ID,
    STIMULATION_VALUE,
    TRM_ENTITY_ID,
    TRM_READOUT_ARTIFACT_PATH,
    TRM_RUN_ID,
    PreparationError,
    _code_bundle,
    _dependency_bundle,
    prepare_bundle,
)
from scripts.run_connectome_service import build_trace_recorder

ROOT = Path(__file__).resolve().parents[2]

#: The steering fixture's runtime graph identity, pinned independently so a
#: silent re-derivation of the 128-node circuit cannot go unnoticed here.
STEERING_GRAPH_HASH = "ebc50c0db92f7ec0e0eeb737af3d6d446cd569a488564a5682a0f255af401c26"

#: A hashed code-bundle root that a build genuinely writes into: ``make
#: spmv-build`` drops the compiled kernel and its sidecar beside the tracked
#: ``_spmv.c``. The two tests below use the real directory rather than a
#: contrived one because the root set is what the defect lived in.
SPMV_SOURCE_DIRECTORY = ROOT / "packages" / "neural-runtime" / "src" / "neurocraft_neural"


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _controller() -> ConnectomeController:
    return ConnectomeController.create(
        load_malecns_fixture(FIXTURE_PATH),
        run_id=RUN_ID,
        entity_id=ENTITY_ID,
        session_epoch=SESSION_EPOCH,
    )


@pytest.mark.parametrize(
    "mod_version",
    [
        "0.1.0",
        "0.1.0-alpha.1",
        "0.2.0-rc.1",
        "0.2.0-rc.2",
        "0.2.0-rc.3",
        "0.2.0-rc.4",
        "0.2.0-rc.5",
        "0.2.0-rc.6",
    ],
)
def test_preparation_accepts_only_declared_build_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mod_version: str
) -> None:
    properties = (ROOT / "mod" / "gradle.properties").read_text(encoding="utf-8")
    current = preparer._gradle_properties()["mod_version"]
    path = tmp_path / "mod" / "gradle.properties"
    path.parent.mkdir()
    path.write_text(properties.replace(f"mod_version={current}", f"mod_version={mod_version}"))
    monkeypatch.setattr(preparer, "ROOT", tmp_path)

    assert preparer._gradle_properties()["mod_version"] == mod_version


@pytest.mark.parametrize(
    ("key", "replacement"),
    [
        ("minecraft_version", "1.21.6"),
        ("loader_version", "0.16.11"),
        ("fabric_api_version", "0.128.3+1.21.5"),
        ("loom_version", "1.10.6"),
        ("mod_version", "0.1.0-alpha.2"),
        ("mod_version", "0.2.0-rc.7"),
        ("mod_version", ""),
    ],
)
def test_preparation_still_refuses_unpinned_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, replacement: str
) -> None:
    versions = preparer._gradle_properties()
    versions[key] = replacement
    path = tmp_path / "mod" / "gradle.properties"
    path.parent.mkdir()
    path.write_text("\n".join(f"{name}={value}" for name, value in versions.items()))
    monkeypatch.setattr(preparer, "ROOT", tmp_path)

    with pytest.raises(PreparationError, match=key):
        preparer._gradle_properties()


def test_prepared_bundle_binds_controller_arena_forces_and_named_stimulation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bundle"
    hashes = prepare_bundle(
        output,
        bundle_reference="artifacts/test/external-bridge",
        machine_label="test-machine",
        os_label="test-os",
    )

    identity = RunIdentity.from_dict(_read(output / "identity.json"))
    manifest = TraceManifest.from_dict(_read(output / "trace-manifest.json"))
    transition = _read(output / "transition-forces.json")
    catalog = _read(output / "control-catalog.json")
    launch = _read(output / "launch.json")
    controller = _controller()
    stimulation = NamedStimulation.create(
        controller,
        STIMULATION_ID,
        {STIMULATION_BODY_ID: STIMULATION_VALUE},
    )

    arena_document = json.loads(ARENA_PATH.read_text(encoding="utf-8"))
    arena_hash = __import__("hashlib").sha256(canonical_json_bytes(arena_document)).hexdigest()
    forces = DeclaredTransitionForces.create(
        passive_forces=tuple(transition["passive_forces"]),  # type: ignore[arg-type]
        external_forces=tuple(transition["external_forces"]),  # type: ignore[arg-type]
        provenance=str(transition["provenance"]),
        arena_material_hash=str(transition["arena_material_hash"]),
    )

    assert identity == controller.identity
    assert manifest.mod == "neurocraft-fly 0.2.0-rc.6"
    assert manifest.java == "21"
    assert manifest.minecraft == "1.21.5"
    assert manifest.fabric == "Loader 0.16.10; API 0.128.2+1.21.5; Loom 1.10.5"
    assert manifest.claim_scope is ClaimScope.G3
    assert manifest.artifacts.runtime_graph_hash == identity.graph_hash
    assert manifest.artifacts.code_hash == _code_bundle()[0]
    assert manifest.artifacts.dependency_lock_hash == _dependency_bundle()[0]
    assert manifest.artifacts.adapter_hash == identity.adapter_hash
    assert manifest.artifacts.dynamics_hash == identity.dynamics_hash
    assert manifest.artifacts.decoder_hash == identity.decoder_hash
    assert manifest.artifacts.observation_schema_hash == identity.observation_schema_hash
    assert manifest.artifacts.sign_hash == controller.sign_overlay.identity_sha256
    assert manifest.artifacts.configuration_hash == controller.configuration.identity_sha256
    assert transition["arena_material_hash"] == arena_hash
    assert manifest.artifacts.arena_hash == forces.identity_sha256
    assert catalog == {
        "schema": "neurocraft.control_catalog.v1",
        "interventions": [{"content_hash": stimulation.content_hash, "content_id": STIMULATION_ID}],
    }
    assert launch["state"] == "prepared_not_executed"
    assert launch["claim_scope"] == {
        "manifest_target": "G3",
        "status": "not_passed_by_preparation",
    }
    # The default circuit keeps the v1 launch schema with no profile key at all;
    # only the opt-in steering circuit is allowed to move to v2.
    assert launch["schema"] == "neurocraft.external_bridge_launch.v1"
    assert "controller_profile" not in launch
    assert launch["credential_embedded"] is False
    dependency_files = launch["identity_bundles"]["dependencies"]["files"]  # type: ignore[index]
    assert next(item for item in dependency_files if item["path"] == "mod/gradle.properties") == {
        "path": "mod/gradle.properties",
        "sha256": preparer._sha256_file(ROOT / "mod" / "gradle.properties"),
    }
    assert next(
        item for item in dependency_files if item["path"] == "release/java-runtime-pins.json"
    ) == {
        "path": "release/java-runtime-pins.json",
        "sha256": preparer._sha256_file(ROOT / "release/java-runtime-pins.json"),
    }
    code_files = launch["identity_bundles"]["code"]["files"]  # type: ignore[index]
    assert all("__pycache__" not in item["path"] for item in code_files)
    assert all(not item["path"].endswith((".pyc", ".pyo")) for item in code_files)
    assert {
        "scripts/__init__.py",
        "scripts/prepare_gametest_demo.py",
        "scripts/run_connectome_service.py",
        "scripts/run_external_gametest.py",
        "scripts/replay_minecraft_trace.py",
    }.issubset({item["path"] for item in code_files})
    assert launch["stimulation"] == {
        "body_id": STIMULATION_BODY_ID,
        "content_hash": stimulation.content_hash,
        "content_id": STIMULATION_ID,
        "value": STIMULATION_VALUE,
    }
    assert set(hashes) == {
        "identity.json",
        "trace-manifest.json",
        "transition-forces.json",
        "control-catalog.json",
        "launch.json",
    }


def test_preparation_is_byte_deterministic_and_refuses_clobber(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    options = {
        "bundle_reference": "artifacts/test/external-bridge",
        "machine_label": "test-machine",
        "os_label": "test-os",
    }
    first_hashes = prepare_bundle(first, **options)
    second_hashes = prepare_bundle(second, **options)

    assert first_hashes == second_hashes
    for name in first_hashes:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    before = {name: (first / name).read_bytes() for name in first_hashes}
    with pytest.raises(FileExistsError, match="refusing to clobber"):
        prepare_bundle(first, **options)
    assert before == {name: (first / name).read_bytes() for name in first_hashes}


def test_prepared_manifest_and_forces_are_accepted_by_service_recorder(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    prepare_bundle(
        output,
        bundle_reference="artifacts/test/external-bridge",
        machine_label="test-machine",
        os_label="test-os",
    )
    args = type(
        "Args",
        (),
        {
            "trace_out": tmp_path / "unexecuted.ndjson",
            "trace_manifest": output / "trace-manifest.json",
            "transition_forces": output / "transition-forces.json",
        },
    )()
    recorder = build_trace_recorder(args, _controller())
    assert recorder is not None
    recorder.abort()
    assert not (tmp_path / "unexecuted.ndjson").exists()


def test_steering_circuit_bundle_binds_the_wing_motor_controller(tmp_path: Path) -> None:
    """Opt-in steering preparation must carry the v2 launch schema, the explicit
    controller profile, and identities only the wing-motor flight controller over
    the pinned 128-node steering fixture can reproduce."""

    from neurocraft_connectome.flight_pin import (
        MALECNS_FLIGHT_FIXTURE_V1_FILE_BYTES,
        MALECNS_FLIGHT_FIXTURE_V1_PIN,
    )
    from neurocraft_controller import FlightConnectomeController
    from neurocraft_controller.flight_controller import DEFAULT_V1_CONFIGURATION_SHA256

    output = tmp_path / "bundle"
    hashes = prepare_bundle(
        output,
        bundle_reference="artifacts/test/external-bridge-steering",
        machine_label="test-machine",
        os_label="test-os",
        circuit=CIRCUIT_FLIGHT_STEERING_V1,
    )

    identity = RunIdentity.from_dict(_read(output / "identity.json"))
    manifest = TraceManifest.from_dict(_read(output / "trace-manifest.json"))
    launch = _read(output / "launch.json")

    assert launch["schema"] == "neurocraft.external_bridge_launch.v2"
    assert launch["controller_profile"] == "wing_motor_soma_side_v1"
    assert identity.run_id == STEERING_RUN_ID == "external-bridge-steering-v1"
    assert identity.entity_id == STEERING_ENTITY_ID
    assert identity.entity_id != ENTITY_ID
    assert identity.graph_hash == STEERING_GRAPH_HASH
    assert manifest.artifacts.runtime_graph_hash == STEERING_GRAPH_HASH

    graph = load_malecns_fixture(
        STEERING_FIXTURE_PATH,
        expected_pin=MALECNS_FLIGHT_FIXTURE_V1_PIN,
        expected_file_bytes=MALECNS_FLIGHT_FIXTURE_V1_FILE_BYTES,
    )
    controller = FlightConnectomeController(
        graph,
        run_id=STEERING_RUN_ID,
        entity_id=STEERING_ENTITY_ID,
        session_epoch=SESSION_EPOCH,
    )
    assert identity == controller.identity
    assert manifest.artifacts.configuration_hash == controller.configuration.identity_sha256
    assert manifest.artifacts.configuration_hash != DEFAULT_V1_CONFIGURATION_SHA256
    assert manifest.artifacts.decoder_hash == controller.identity.decoder_hash

    argv = launch["commands"]["service"]["argv"]  # type: ignore[index]
    assert isinstance(argv, list)
    profile_index = argv.index("--controller-profile")
    assert argv[profile_index + 1] == "wing_motor_soma_side_v1"
    fixture_index = argv.index("--fixture")
    assert argv[fixture_index + 1] == "data/fixtures/mcns-v1.0-flight-steering-v1.json"
    # The stimulation body exists in both fixtures and stays unchanged.
    assert launch["stimulation"]["body_id"] == STIMULATION_BODY_ID  # type: ignore[index]
    assert set(hashes) == {
        "identity.json",
        "trace-manifest.json",
        "transition-forces.json",
        "control-catalog.json",
        "launch.json",
    }


def test_trm_demo_bundle_binds_the_trained_readout_controller(tmp_path: Path) -> None:
    """Opt-in TRM preparation must carry the v3 launch schema, the explicit
    trained-readout profile, the artifact digest derived from the committed
    readout file, and identities only the trained-readout controller over the
    pinned 128-node steering fixture can reproduce."""

    import hashlib
    import uuid

    from neurocraft_connectome.flight_pin import (
        MALECNS_FLIGHT_FIXTURE_V1_FILE_BYTES,
        MALECNS_FLIGHT_FIXTURE_V1_PIN,
    )
    from neurocraft_controller.trained_readout_controller import (
        TrainedReadoutController,
        TrmDemoConfiguration,
    )

    output = tmp_path / "bundle"
    hashes = prepare_bundle(
        output,
        bundle_reference="artifacts/test/external-bridge-trm",
        machine_label="test-machine",
        os_label="test-os",
        circuit=CIRCUIT_TRM_DEMO_V1,
    )

    identity = RunIdentity.from_dict(_read(output / "identity.json"))
    manifest = TraceManifest.from_dict(_read(output / "trace-manifest.json"))
    launch = _read(output / "launch.json")

    artifact_bytes = TRM_READOUT_ARTIFACT_PATH.read_bytes()
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    assert launch["schema"] == "neurocraft.external_bridge_launch.v3"
    assert launch["controller_profile"] == "trm_cloned_readout_v1"
    assert launch["readout_artifact_sha256"] == artifact_sha256
    # The committed artifact this profile was written against; swapping the
    # committed weights file must be a visible, deliberate re-pin here.
    # Re-pinned to the trm-live-demo-v2 artifact (the recorded v2 retraining);
    # the v1 artifact sha 527eb4f7... remains pinned as history in
    # tests/controller/test_trained_readout_controller.py.
    assert artifact_sha256 == ("23938cd78cc7783de63af02343f7a0d89b40e707e2951c4203e2a90c6b6d8db0")
    assert identity.run_id == TRM_RUN_ID == "external-bridge-trm-demo-v1"
    assert identity.entity_id == TRM_ENTITY_ID
    # Recompute the pinned uuid5 literal so a silent edit cannot drift it.
    assert identity.entity_id == str(
        uuid.uuid5(uuid.NAMESPACE_URL, "neurocraft:external-bridge-trm-demo-v1")
    )
    assert identity.entity_id not in {ENTITY_ID, STEERING_ENTITY_ID}
    assert identity.graph_hash == STEERING_GRAPH_HASH
    assert manifest.artifacts.runtime_graph_hash == STEERING_GRAPH_HASH

    graph = load_malecns_fixture(
        STEERING_FIXTURE_PATH,
        expected_pin=MALECNS_FLIGHT_FIXTURE_V1_PIN,
        expected_file_bytes=MALECNS_FLIGHT_FIXTURE_V1_FILE_BYTES,
    )
    controller = TrainedReadoutController(
        graph,
        run_id=TRM_RUN_ID,
        entity_id=TRM_ENTITY_ID,
        session_epoch=SESSION_EPOCH,
        artifact_bytes=artifact_bytes,
        artifact_document=json.loads(artifact_bytes),
        configuration=TrmDemoConfiguration(readout_artifact_sha256=artifact_sha256),
    )
    assert identity == controller.identity
    assert manifest.artifacts.configuration_hash == controller.configuration.identity_sha256
    assert manifest.artifacts.decoder_hash == controller.identity.decoder_hash

    argv = launch["commands"]["service"]["argv"]  # type: ignore[index]
    assert isinstance(argv, list)
    profile_index = argv.index("--controller-profile")
    assert argv[profile_index + 1] == "trm_cloned_readout_v1"
    fixture_index = argv.index("--fixture")
    assert argv[fixture_index + 1] == "data/fixtures/mcns-v1.0-flight-steering-v1.json"
    # The stimulation body exists in the steering fixture and stays unchanged.
    assert launch["stimulation"]["body_id"] == STIMULATION_BODY_ID  # type: ignore[index]
    assert set(hashes) == {
        "identity.json",
        "trace-manifest.json",
        "transition-forces.json",
        "control-catalog.json",
        "launch.json",
    }


def test_launch_documents_contain_no_embedded_credential_value(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    prepare_bundle(
        output,
        bundle_reference="artifacts/test/external-bridge",
        machine_label="test-machine",
        os_label="test-os",
    )
    launch = _read(output / "launch.json")
    serialized = json.dumps(launch, sort_keys=True)
    assert "NEUROCRAFT_RUN_TOKEN" in serialized
    assert '"credential_embedded": false' in serialized
    assert "fixture-integration-token" not in serialized
    assert "example-token" not in serialized


def test_code_bundle_is_unmoved_by_a_build_product_inside_a_hashed_root() -> None:
    """A built machine and a clean checkout must bind the same commit identically.

    Before the roots were enumerated from Git they were walked with ``rglob``, so
    the gitignored outputs of ``make spmv-build`` — which land inside a hashed
    root — entered the bundle and moved ``code_hash``. Two parties reading
    identical source could then disagree about its identity.
    """

    baseline_hash, baseline_files = _code_bundle()
    created: list[Path] = []
    try:
        for name in sorted(BUNDLE_BUILD_PRODUCT_NAMES):
            path = SPMV_SOURCE_DIRECTORY / name
            if path.exists():
                continue
            path.write_bytes(b"machine-specific build product")
            created.append(path)
        assert created, "every build-product name already exists; the guard was not exercised"
        observed_hash, observed_files = _code_bundle()
    finally:
        for path in created:
            path.unlink()

    assert observed_hash == baseline_hash
    assert observed_files == baseline_files
    assert not any(
        Path(entry["path"]).name in BUNDLE_BUILD_PRODUCT_NAMES for entry in observed_files
    )


def test_code_bundle_refuses_an_untracked_source_file_inside_a_hashed_root() -> None:
    """Git decides membership, so an untracked source file must fail loudly.

    Dropping it silently would be the same defect in the other direction: the
    recorded identity would describe a code set that is not the one an import can
    reach.
    """

    intruder = SPMV_SOURCE_DIRECTORY / "_untracked_identity_probe.py"
    assert not intruder.exists()
    intruder.write_text("VALUE = 1\n", encoding="utf-8")
    try:
        with pytest.raises(PreparationError, match="untracked file inside a hashed identity root"):
            _code_bundle()
    finally:
        intruder.unlink()


def _write_release_fixture(root: Path, files: dict[str, bytes]) -> None:
    """Use the actual exporter schema, with no Git repository or mocked reader."""
    from scripts.build_public_release import SourceFile, canonical_json, make_manifest

    records = []
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        records.append(SourceFile(name, name, "a" * 40, 0o100644, data))
    manifest = make_manifest(
        "b" * 40, "release/public-files.json", b"{}", {"exclusions": []}, records
    )
    (root / "RELEASE_MANIFEST.json").write_bytes(canonical_json(manifest))


@pytest.fixture
def release_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "archive"
    _write_release_fixture(
        root,
        {
            "LICENSE": b"Example license\n",
            "packages/demo/src/demo.py": b"VALUE = 1\n",
            "scripts/entry.py": b"pass\n",
            "mod/gradle.properties": b"mod_version=0.1.0-alpha.1\n",
        },
    )
    monkeypatch.setattr(preparer, "ROOT", root)
    return root


def test_release_source_uses_manifest_even_inside_unrelated_parent_git(
    release_source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=release_source.parent, check=True)

    def no_git(*args: object, **kwargs: object) -> None:
        pytest.fail("archive must not use the unrelated parent Git index")

    monkeypatch.setattr(subprocess, "run", no_git)
    source = release_source / "packages/demo/src"
    explicit = release_source / "scripts/entry.py"
    assert preparer._collect_files((source,), (explicit,)) == (source / "demo.py", explicit)
    assert preparer._collect_files((), (explicit,)) == (explicit,)


def test_release_code_and_dependency_bundles_match_checkout_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = _code_bundle()
    dependencies = _dependency_bundle()
    names = {entry["path"] for _, entries in (code, dependencies) for entry in entries}
    root = tmp_path / "complete-source"
    files = {name: (ROOT / name).read_bytes() for name in names}
    files["LICENSE"] = (ROOT / "LICENSE").read_bytes()
    _write_release_fixture(root, files)
    monkeypatch.setattr(preparer, "ROOT", root)

    assert _code_bundle() == code
    assert _dependency_bundle() == dependencies


@pytest.mark.parametrize("change", ["same_size", "different_size", "missing"])
def test_release_source_detects_changed_listed_bytes(release_source: Path, change: str) -> None:
    # Even listed files outside the selected root must match this source release.
    preparer._collect_files((release_source / "packages/demo/src",))
    path = release_source / "scripts/entry.py"
    if change == "missing":
        path.unlink()
    else:
        path.write_bytes(b"xxxx\n" if change == "same_size" else b"changed\n")
    with pytest.raises(PreparationError, match=r"source release file .*scripts/entry\.py"):
        preparer._collect_files((release_source / "packages/demo/src",))


def test_release_source_refuses_unlisted_source_and_explicit_input(release_source: Path) -> None:
    source = release_source / "packages/demo/src"
    intruder = source / "intruder.py"
    intruder.write_bytes(b"pass\n")
    with pytest.raises(PreparationError, match="untracked file inside a hashed identity root"):
        preparer._collect_files((source,))
    intruder.unlink()
    explicit = release_source / "scripts/unlisted.py"
    explicit.write_bytes(b"pass\n")
    with pytest.raises(PreparationError, match="not listed in source release"):
        preparer._collect_files((), (explicit,))


def test_release_source_preserves_build_product_exclusions(release_source: Path) -> None:
    source = release_source / "packages/demo/src"
    baseline = preparer._collect_files((source,))
    for name in BUNDLE_BUILD_PRODUCT_NAMES:
        (source / name).write_bytes(b"compiled locally")
    (source / "__pycache__").mkdir()
    (source / "__pycache__/demo.pyc").write_bytes(b"interpreter cache")
    assert preparer._collect_files((source,)) == baseline


@pytest.mark.parametrize("kind", ["listed_file", "listed_parent", "unlisted", "cache", "manifest"])
def test_release_source_refuses_symlinks(release_source: Path, kind: str) -> None:
    source = release_source / "packages/demo/src"
    if kind == "listed_file":
        path = source / "demo.py"
        destination = source / "copy.py"
        path.rename(destination)
    elif kind == "listed_parent":
        path = source
        destination = source.with_name("moved")
        path.rename(destination)
    elif kind == "manifest":
        path = release_source / "RELEASE_MANIFEST.json"
        destination = path.with_suffix(".copy")
        path.rename(destination)
    else:
        path = source / ("__pycache__" if kind == "cache" else "unknown.py")
        destination = source / "nonexistent"
    path.symlink_to(destination)
    with pytest.raises(PreparationError, match="symlink"):
        preparer._collect_files((source,))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path", "../outside.py"),
        ("path", "/absolute.py"),
        ("path", "packages//demo.py"),
        ("path", "packages\\demo.py"),
        ("bytes", True),
        ("bytes", 1.5),
        ("bytes", 16 * 1024 * 1024 + 1),
        ("sha256", "not a hash"),
        ("mode", "120000"),
        ("mode", []),
    ],
)
def test_release_source_refuses_malformed_entries(
    release_source: Path, field: str, value: object
) -> None:
    manifest_path = release_source / "RELEASE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"][0][field] = value
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(PreparationError, match="source release"):
        preparer._collect_files((release_source / "packages/demo/src",))


@pytest.mark.parametrize("change", ["missing", "schema", "duplicate", "empty", "oversize"])
def test_release_source_requires_bounded_valid_manifest(release_source: Path, change: str) -> None:
    path = release_source / "RELEASE_MANIFEST.json"
    manifest = json.loads(path.read_bytes())
    if change == "missing":
        path.unlink()
    elif change == "oversize":
        path.write_bytes(b" " * (2 * 1024 * 1024 + 1))
    else:
        if change == "schema":
            manifest["schema"] = "unknown"
        elif change == "duplicate":
            manifest["files"].append(manifest["files"][0])
        else:
            manifest["files"] = []
        path.write_text(json.dumps(manifest))
    with pytest.raises(PreparationError, match=r"(?i)manifest|MiB"):
        preparer._collect_files((release_source / "packages/demo/src",))


def test_checkout_keeps_git_semantics_with_invalid_manifest(
    release_source: Path,
) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=release_source, check=True)
    subprocess.run(["git", "add", "."], cwd=release_source, check=True)
    (release_source / "RELEASE_MANIFEST.json").write_text("not an archive manifest")
    source = release_source / "packages/demo/src"
    # Local source edits remain permitted and are hashed as in an ordinary checkout.
    (source / "demo.py").write_bytes(b"VALUE = 2\n")
    assert preparer._collect_files((source,)) == (source / "demo.py",)


def test_execution_origin_verifies_archive_without_borrowing_parent_git(
    release_source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=release_source.parent, check=True)
    source = release_source / "scripts/entry.py"
    inventory = {"entry": {"path": str(source), "sha256": preparer._sha256_file(source)}}

    def no_git(*args: object, **kwargs: object) -> None:
        pytest.fail("archive provenance must not read a parent Git HEAD")

    monkeypatch.setattr(subprocess, "check_output", no_git)
    origin = preparer.execution_source_origin(release_source, inventory)
    assert origin["kind"] == "verified_source_archive"
    assert origin["git_head"] is None
    assert origin["source_commit"] == "b" * 40
    assert origin["publisher_authenticity_verified"] is False


@pytest.mark.parametrize("change", ["unlisted", "outside", "digest", "missing_path"])
def test_execution_origin_refuses_unbound_inventory(release_source: Path, change: str) -> None:
    source = release_source / "scripts/entry.py"
    if change == "unlisted":
        source = release_source / "scripts/unknown.py"
        source.write_bytes(b"pass\n")
    elif change == "outside":
        source = release_source.parent / "external.py"
        source.write_bytes(b"pass\n")
    inventory = {"entry": {"path": str(source), "sha256": preparer._sha256_file(source)}}
    if change == "digest":
        inventory["entry"]["sha256"] = "0" * 64
    elif change == "missing_path":
        del inventory["entry"]["path"]
    with pytest.raises(PreparationError, match="executed source"):
        preparer.execution_source_origin(release_source, inventory)


def test_execution_origin_retains_actual_checkout_head(release_source: Path) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=release_source, check=True)
    subprocess.run(["git", "add", "."], cwd=release_source, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "source fixture",
        ],
        cwd=release_source,
        check=True,
    )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=release_source, text=True
    ).strip()
    (release_source / "RELEASE_MANIFEST.json").write_text("unused in checkout")
    assert preparer.execution_source_origin(release_source, {}) == {
        "kind": "git_checkout",
        "git_head": head,
        "source_commit": head,
        "release_manifest_sha256": None,
    }
