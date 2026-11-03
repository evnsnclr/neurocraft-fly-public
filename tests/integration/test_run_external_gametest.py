from __future__ import annotations

import json
import os
import shutil
import signal
import stat
import subprocess
import sys
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from neurocraft_protocol import canonical_json_bytes

from scripts import run_external_gametest as orchestrator
from scripts.prepare_gametest_demo import (
    CIRCUIT_FLIGHT_STEERING_V1,
    CIRCUIT_TRM_DEMO_V1,
    _gradle_properties,
    prepare_bundle,
)
from tests.protocol.helpers import causal_record

STEERING_GRAPH_HASH = "ebc50c0db92f7ec0e0eeb737af3d6d446cd569a488564a5682a0f255af401c26"


def _prepared(tmp_path: Path) -> orchestrator.ValidatedBundle:
    output = tmp_path / "bundle"
    prepare_bundle(
        output,
        bundle_reference="artifacts/test/external-bridge",
        machine_label="test-machine",
        os_label="test-os",
    )
    return orchestrator.validate_bundle(output, enforce_bundle_location=False)


def _prepared_steering(tmp_path: Path) -> Path:
    output = tmp_path / "steering-bundle"
    prepare_bundle(
        output,
        bundle_reference="artifacts/test/external-bridge-steering",
        machine_label="test-machine",
        os_label="test-os",
        circuit=CIRCUIT_FLIGHT_STEERING_V1,
    )
    return output


def _prepared_trm(tmp_path: Path) -> Path:
    output = tmp_path / "trm-bundle"
    prepare_bundle(
        output,
        bundle_reference="artifacts/test/external-bridge-trm",
        machine_label="test-machine",
        os_label="test-os",
        circuit=CIRCUIT_TRM_DEMO_V1,
    )
    return output


def test_validator_accepts_the_prepared_allow_list_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    validated = _prepared(tmp_path)

    assert validated.service_argv[:15] == (
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
    )
    assert validated.gradle_argv[:4] == (
        "./gradlew",
        "--no-daemon",
        "--dependency-verification",
        "strict",
    )
    assert validated.gradle_argv[-2:] == ("clean", "runExternalBridgeGameTest")
    assert validated.host == "127.0.0.1"
    assert validated.port == 8765

    catalog_path = validated.bundle / "control-catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["interventions"][0]["content_hash"] = "f" * 64
    catalog_path.write_bytes(canonical_json_bytes(catalog) + b"\n")
    with pytest.raises(orchestrator.OrchestrationError, match="prepared file hash mismatch"):
        orchestrator.validate_bundle(validated.bundle, enforce_bundle_location=False)


def test_validator_accepts_the_default_bundle_with_the_pair_profile(tmp_path: Path) -> None:
    validated = _prepared(tmp_path)

    assert validated.controller_profile == "pair_v1"
    assert validated.fixture.name == "mcns-v1.0-flight-power-v1.json"
    assert "--controller-profile" not in validated.service_argv


def test_validator_refuses_prior_mod_build_even_with_consistent_bundle_hashes(
    tmp_path: Path,
) -> None:
    """An allowed historical version cannot lend its bundle to the alpha build."""
    validated = _prepared(tmp_path)
    properties = (orchestrator.ROOT / "mod" / "gradle.properties").read_bytes()
    current = _gradle_properties()["mod_version"]
    old_properties = properties.replace(f"mod_version={current}".encode(), b"mod_version=0.1.0")
    assert old_properties != properties
    launch_path = validated.bundle / "launch.json"
    launch = json.loads(launch_path.read_bytes())
    dependencies = launch["identity_bundles"]["dependencies"]
    for entry in dependencies["files"]:
        if entry["path"] == "mod/gradle.properties":
            entry["sha256"] = orchestrator._sha256_bytes(old_properties)
    dependencies["sha256"] = orchestrator._sha256_bytes(
        canonical_json_bytes({"schema": dependencies["schema"], "files": dependencies["files"]})
    )
    manifest_path = validated.bundle / "trace-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["mod"] = "neurocraft-fly 0.1.0"
    manifest["artifacts"]["dependency_lock_hash"] = dependencies["sha256"]
    encoded = canonical_json_bytes(manifest) + b"\n"
    manifest_path.write_bytes(encoded)
    launch["prepared_file_sha256"]["trace-manifest.json"] = orchestrator._sha256_bytes(encoded)
    launch_path.write_bytes(canonical_json_bytes(launch) + b"\n")

    with pytest.raises(
        orchestrator.OrchestrationError,
        match=r"current file does not match .*dependencies: mod/gradle.properties",
    ):
        orchestrator.validate_bundle(validated.bundle, enforce_bundle_location=False)


def test_validator_round_trips_the_steering_circuit_bundle(tmp_path: Path) -> None:
    """A v2 steering launch must be re-validated with the wing-motor flight
    controller over the pinned steering fixture, and its allow-listed service
    argv must carry the explicit controller profile."""

    output = _prepared_steering(tmp_path)

    validated = orchestrator.validate_bundle(output, enforce_bundle_location=False)

    assert validated.controller_profile == "wing_motor_soma_side_v1"
    assert validated.fixture.name == "mcns-v1.0-flight-steering-v1.json"
    assert validated.identity.run_id == "external-bridge-steering-v1"
    assert validated.identity.graph_hash == STEERING_GRAPH_HASH
    profile_index = validated.service_argv.index("--controller-profile")
    assert validated.service_argv[profile_index + 1] == "wing_motor_soma_side_v1"
    fixture_index = validated.service_argv.index("--fixture")
    assert (
        validated.service_argv[fixture_index + 1]
        == "data/fixtures/mcns-v1.0-flight-steering-v1.json"
    )


def test_validator_rejects_a_v2_launch_that_claims_the_pair_profile(tmp_path: Path) -> None:
    """The preparer never emits a v2 launch for the pair readout; the validator
    must refuse one instead of guessing a fixture for it."""

    output = _prepared_steering(tmp_path)
    launch_path = output / "launch.json"
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    launch["controller_profile"] = "pair_v1"
    launch_path.write_bytes(canonical_json_bytes(launch) + b"\n")

    with pytest.raises(orchestrator.OrchestrationError, match="controller profile"):
        orchestrator.validate_bundle(output, enforce_bundle_location=False)


def test_validator_round_trips_the_trm_demo_bundle(tmp_path: Path) -> None:
    """A v3 TRM launch must be re-validated with the trained-readout controller
    over the pinned steering fixture, and its allow-listed service argv must
    carry the explicit trained-readout profile."""

    output = _prepared_trm(tmp_path)

    validated = orchestrator.validate_bundle(output, enforce_bundle_location=False)

    assert validated.controller_profile == "trm_cloned_readout_v1"
    assert validated.fixture.name == "mcns-v1.0-flight-steering-v1.json"
    assert validated.identity.run_id == "external-bridge-trm-demo-v1"
    assert validated.identity.graph_hash == STEERING_GRAPH_HASH
    profile_index = validated.service_argv.index("--controller-profile")
    assert validated.service_argv[profile_index + 1] == "trm_cloned_readout_v1"
    fixture_index = validated.service_argv.index("--fixture")
    assert (
        validated.service_argv[fixture_index + 1]
        == "data/fixtures/mcns-v1.0-flight-steering-v1.json"
    )


def test_validator_rejects_a_v3_launch_that_claims_another_profile(tmp_path: Path) -> None:
    """v3 launches carry the trained-readout profile only; a v3 launch naming
    the wing or pair profile must be refused instead of resolved."""

    output = _prepared_trm(tmp_path)
    launch_path = output / "launch.json"
    original = launch_path.read_bytes()
    for borrowed in ("wing_motor_soma_side_v1", "pair_v1"):
        launch = json.loads(original)
        launch["controller_profile"] = borrowed
        launch_path.write_bytes(canonical_json_bytes(launch) + b"\n")
        with pytest.raises(orchestrator.OrchestrationError, match="v3 controller profile"):
            orchestrator.validate_bundle(output, enforce_bundle_location=False)


def test_validator_rejects_a_v2_launch_that_claims_the_trm_profile(tmp_path: Path) -> None:
    """The trained-readout profile is prepared only under v3; a v2 launch
    naming it must be refused rather than resolved without its artifact
    digest."""

    output = _prepared_steering(tmp_path)
    launch_path = output / "launch.json"
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    launch["controller_profile"] = "trm_cloned_readout_v1"
    launch_path.write_bytes(canonical_json_bytes(launch) + b"\n")

    with pytest.raises(orchestrator.OrchestrationError, match="v2 controller profile"):
        orchestrator.validate_bundle(output, enforce_bundle_location=False)


def test_validator_rejects_v3_key_set_violations(tmp_path: Path) -> None:
    """Exact-key discipline per schema version: v3 requires both added keys,
    tolerates no extras, and v2 must never carry the artifact digest."""

    trm_output = _prepared_trm(tmp_path)
    launch_path = trm_output / "launch.json"
    original = launch_path.read_bytes()

    launch = json.loads(original)
    del launch["readout_artifact_sha256"]
    launch_path.write_bytes(canonical_json_bytes(launch) + b"\n")
    with pytest.raises(orchestrator.OrchestrationError, match="missing or unknown fields"):
        orchestrator.validate_bundle(trm_output, enforce_bundle_location=False)

    launch = json.loads(original)
    launch["readout_artifact_path"] = "data/readouts/trm-cloned-readout-v1.json"
    launch_path.write_bytes(canonical_json_bytes(launch) + b"\n")
    with pytest.raises(orchestrator.OrchestrationError, match="missing or unknown fields"):
        orchestrator.validate_bundle(trm_output, enforce_bundle_location=False)

    wing_output = _prepared_steering(tmp_path)
    wing_launch_path = wing_output / "launch.json"
    wing_launch = json.loads(wing_launch_path.read_text(encoding="utf-8"))
    wing_launch["readout_artifact_sha256"] = "f" * 64
    wing_launch_path.write_bytes(canonical_json_bytes(wing_launch) + b"\n")
    with pytest.raises(orchestrator.OrchestrationError, match="missing or unknown fields"):
        orchestrator.validate_bundle(wing_output, enforce_bundle_location=False)


def test_validator_rejects_a_readout_artifact_digest_mismatch(tmp_path: Path) -> None:
    """The launch's artifact digest must equal the digest recomputed from the
    committed readout file; a well-formed but different digest must fail."""

    output = _prepared_trm(tmp_path)
    launch_path = output / "launch.json"
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    launch["readout_artifact_sha256"] = "f" * 64
    launch_path.write_bytes(canonical_json_bytes(launch) + b"\n")

    with pytest.raises(orchestrator.OrchestrationError, match="readout artifact"):
        orchestrator.validate_bundle(output, enforce_bundle_location=False)


def test_validator_pins_trm_identity_labels_to_the_profile_row(tmp_path: Path) -> None:
    """A TRM bundle whose identity borrows the steering run/entity labels must
    fail the profile-row pin exactly as the wing bundle does (finding F1)."""

    from hashlib import sha256

    output = _prepared_trm(tmp_path)
    identity = json.loads((output / "identity.json").read_text())
    identity["run_id"] = "external-bridge-steering-v1"
    identity["entity_id"] = "056b9dda-7e14-564e-bdc0-1fc40785afa3"
    encoded = canonical_json_bytes(identity) + b"\n"
    (output / "identity.json").write_bytes(encoded)
    launch = json.loads((output / "launch.json").read_text())
    launch["identity"] = identity
    launch["prepared_file_sha256"]["identity.json"] = sha256(encoded).hexdigest()
    (output / "launch.json").write_bytes(canonical_json_bytes(launch) + b"\n")
    with pytest.raises(
        orchestrator.OrchestrationError,
        match="do not match the launch controller profile",
    ):
        orchestrator.validate_bundle(output, enforce_bundle_location=False)


def test_validator_rejects_a_v1_launch_carrying_a_profile_key(tmp_path: Path) -> None:
    """Exact-key discipline per schema version: the profile key is v2-only."""

    validated = _prepared(tmp_path)
    launch_path = validated.bundle / "launch.json"
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    launch["controller_profile"] = "pair_v1"
    launch_path.write_bytes(canonical_json_bytes(launch) + b"\n")

    with pytest.raises(orchestrator.OrchestrationError, match="missing or unknown fields"):
        orchestrator.validate_bundle(validated.bundle, enforce_bundle_location=False)


def test_validator_rejects_bundle_that_omits_replay_verifier(tmp_path: Path) -> None:
    validated = _prepared(tmp_path)
    launch_path = validated.bundle / "launch.json"
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    code = launch["identity_bundles"]["code"]
    code["files"] = [
        entry for entry in code["files"] if entry["path"] != "scripts/replay_minecraft_trace.py"
    ]
    code["sha256"] = orchestrator._sha256_bytes(
        canonical_json_bytes(
            {
                "files": code["files"],
                "schema": "neurocraft.gametest_code_bundle.v1",
            }
        )
    )
    launch_path.write_bytes(canonical_json_bytes(launch) + b"\n")

    with pytest.raises(orchestrator.OrchestrationError, match="omits required"):
        orchestrator.validate_bundle(validated.bundle, enforce_bundle_location=False)


def test_validator_rejects_unexpected_current_execution_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = _prepared(tmp_path)
    launch = json.loads((validated.bundle / "launch.json").read_text(encoding="utf-8"))
    entries = list(launch["identity_bundles"]["code"]["files"])
    entries.append({"path": "mod/src/main/resources/data/injected.json", "sha256": "f" * 64})
    entries.sort(key=lambda entry: entry["path"])
    monkeypatch.setattr(
        orchestrator,
        "_current_code_bundle",
        lambda: (
            orchestrator._sha256_bytes(
                canonical_json_bytes(
                    {"files": entries, "schema": "neurocraft.gametest_code_bundle.v1"}
                )
            ),
            entries,
        ),
    )
    with pytest.raises(orchestrator.OrchestrationError, match="exact current execution source"):
        orchestrator.validate_bundle(validated.bundle, enforce_bundle_location=False)


def test_execution_environment_rejects_injection_variables() -> None:
    with pytest.raises(orchestrator.OrchestrationError, match="environment-injection"):
        orchestrator._execution_environment(
            {
                "NEUROCRAFT_RUN_TOKEN": "A" * 43,
                "PATH": "/usr/bin:/bin",
                "PYTHONPATH": "/tmp/injected",
            },
            temporary_directory=Path("/unused-for-rejected-environment"),
        )


@pytest.mark.parametrize("injected", ["JAVA_TOOL_OPTIONS", "JDK_JAVA_OPTIONS", "_JAVA_OPTIONS"])
def test_private_temp_does_not_allow_caller_jvm_options(injected: str) -> None:
    with pytest.raises(orchestrator.OrchestrationError, match="environment-injection"):
        orchestrator._execution_environment(
            {"NEUROCRAFT_RUN_TOKEN": "A" * 43, injected: "-Djava.io.tmpdir=/tmp"},
            temporary_directory=Path("/unused-for-rejected-environment"),
        )


def test_private_temporary_directory_rejects_shared_and_symlinked_paths(tmp_path: Path) -> None:
    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    orchestrator._require_private_temporary_directory(directory.resolve())
    directory.chmod(0o770)
    with pytest.raises(orchestrator.OrchestrationError, match="owner-only"):
        orchestrator._require_private_temporary_directory(directory.resolve())
    directory.chmod(0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)
    with pytest.raises(orchestrator.OrchestrationError, match="owner-only"):
        orchestrator._require_private_temporary_directory(alias)


def test_private_temp_rejects_replaceable_ancestor_and_option_quote(tmp_path: Path) -> None:
    parent = tmp_path / "shared"
    parent.mkdir(mode=0o700)
    directory = parent / "private"
    directory.mkdir(mode=0o700)
    parent.chmod(0o777)
    with pytest.raises(orchestrator.OrchestrationError, match="unsafe writable ancestor"):
        orchestrator._require_private_temporary_directory(directory.resolve())
    parent.chmod(0o700)
    quoted = tmp_path / 'private" -Dinjected=true'
    quoted.mkdir(mode=0o700)
    with pytest.raises(orchestrator.OrchestrationError, match="owner-only"):
        orchestrator._require_private_temporary_directory(quoted.resolve())


def test_real_child_jvm_observes_private_temp_and_rejects_wrong_property(tmp_path: Path) -> None:
    java = shutil.which("java")
    if java is None:
        pytest.skip("JVM property probe requires Java")
    private_directory = (tmp_path / "private directory with spaces").resolve()
    private_directory.mkdir(mode=0o700)
    environment = {
        "PATH": orchestrator._CHILD_PATH,
        "JAVA_TOOL_OPTIONS": f'-Djava.io.tmpdir="{private_directory}"',
    }
    orchestrator._verify_java_temporary_directory(Path(java), environment, private_directory)
    environment["JAVA_TOOL_OPTIONS"] = '-Djava.io.tmpdir="/tmp"'
    with pytest.raises(orchestrator.OrchestrationError, match="did not use the private"):
        orchestrator._verify_java_temporary_directory(Path(java), environment, private_directory)


def test_token_and_all_generated_outputs_are_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = _prepared(tmp_path)
    isolated = replace(
        validated,
        trace=tmp_path / "trace.ndjson",
        gametest_receipt=tmp_path / "gametest.json",
        replay_receipt=tmp_path / "replay.json",
        combined_receipt=tmp_path / "combined.json",
    )

    with pytest.raises(orchestrator.OrchestrationError, match="32-256"):
        orchestrator._require_run_token({})
    token = "A" * 43
    assert orchestrator._require_run_token({"NEUROCRAFT_RUN_TOKEN": token}) == token

    isolated.trace.write_text("already present", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to clobber"):
        orchestrator._require_outputs_absent(isolated)
    isolated.trace.unlink()
    monkeypatch.setattr(orchestrator, "_tcp_ready", lambda *_args, **_kwargs: True)
    with pytest.raises(orchestrator.OrchestrationError, match="already has a listener"):
        orchestrator._run_processes(
            isolated,
            environment={"NEUROCRAFT_RUN_TOKEN": token},
            readiness_timeout=1.0,
            gametest_timeout=1.0,
            termination_timeout=1.0,
        )


def test_gametest_pacing_receipt_is_exact() -> None:
    receipt: dict[str, object] = {
        "pacing_mode": orchestrator.EXPECTED_PACING_MODE,
        "pacing_period_ns": orchestrator.EXPECTED_PACING_PERIOD_NS,
    }

    orchestrator._verify_gametest_pacing(receipt)

    receipt["pacing_mode"] = "accelerated"
    with pytest.raises(orchestrator.OrchestrationError, match="wrong pacing mode"):
        orchestrator._verify_gametest_pacing(receipt)
    receipt["pacing_mode"] = orchestrator.EXPECTED_PACING_MODE
    receipt["pacing_period_ns"] = orchestrator.EXPECTED_PACING_PERIOD_NS - 1
    with pytest.raises(orchestrator.OrchestrationError, match="wrong pacing period"):
        orchestrator._verify_gametest_pacing(receipt)


def test_operational_telemetry_requires_measured_timings() -> None:
    first = causal_record(sequence=0)
    second = replace(
        causal_record(sequence=1),
        queue_coalesced=2,
        queue_overruns=1,
        compute_ns=2_000_000,
    )

    assert orchestrator._summarize_operational_telemetry((first, second)) == {
        "record_count": 2,
        "queue_coalesced_total": 2,
        "queue_overruns_total": 1,
        "compute_ns": {"min": 1_000_000, "max": 2_000_000, "total": 3_000_000},
        "serialization_ns": {"min": 20_000, "max": 20_000, "total": 40_000},
        "write_ns": {"min": 30_000, "max": 30_000, "total": 60_000},
    }

    with pytest.raises(orchestrator.OrchestrationError, match="lacks measured"):
        orchestrator._summarize_operational_telemetry((replace(first, compute_ns=0),))


def test_process_runner_uses_exact_argv_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = _prepared(tmp_path)
    calls: dict[str, Any] = {"terminated": []}

    class FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def poll(self) -> None:
            return None

        def wait(self, *, timeout: float) -> int:
            assert timeout == 2.0
            return 0

    service, gradle = FakeProcess(12001), FakeProcess(12002)

    def fake_popen(argv: list[str], **kwargs: object) -> FakeProcess:
        name = "service" if argv[0] == "/pinned/uv" else "gradle"
        calls[name + "_argv"] = argv
        calls[name + "_kwargs"] = kwargs
        return service if name == "service" else gradle

    monkeypatch.setattr(orchestrator, "_tcp_ready", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(orchestrator.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(orchestrator, "_wait_for_service", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        orchestrator,
        "_terminate_child",
        lambda observed, **_kwargs: calls["terminated"].append(observed),
    )
    environment = {"NEUROCRAFT_RUN_TOKEN": "B" * 43, "PATH": "/test/bin"}
    child_environment = {
        "NEUROCRAFT_RUN_TOKEN": environment["NEUROCRAFT_RUN_TOKEN"],
        "PATH": orchestrator._CHILD_PATH,
    }

    def fake_environment(
        _environment: object, *, temporary_directory: Path
    ) -> tuple[dict[str, str], dict[str, str], str]:
        metadata = temporary_directory.stat()
        assert metadata.st_uid == os.getuid()
        assert stat.S_IMODE(metadata.st_mode) == 0o700
        calls["temporary_directory"] = temporary_directory
        return (
            child_environment,
            {"schema": "neurocraft.execution_environment.v2"},
            "/pinned/uv",
        )

    monkeypatch.setattr(orchestrator, "_execution_environment", fake_environment)
    receipt = orchestrator._run_processes(
        validated,
        environment=environment,
        readiness_timeout=1.0,
        gametest_timeout=2.0,
        termination_timeout=1.0,
    )
    assert calls["service_argv"] == ["/pinned/uv", *validated.service_argv[1:]]
    assert calls["gradle_argv"] == list(validated.gradle_argv)
    for name, cwd in (("service", orchestrator.ROOT), ("gradle", orchestrator.MOD_ROOT)):
        assert calls[name + "_kwargs"]["cwd"] == cwd
        assert calls[name + "_kwargs"]["env"] == child_environment
        assert calls[name + "_kwargs"]["start_new_session"] is True
    assert calls["service_kwargs"]["stdout"] is None
    assert calls["service_kwargs"]["stderr"] is None
    assert calls["terminated"] == [gradle, service]
    assert not calls["temporary_directory"].exists()
    assert receipt == {"schema": "neurocraft.execution_environment.v2"}
    assert environment["NEUROCRAFT_RUN_TOKEN"] not in " ".join(validated.service_argv)
    assert environment["NEUROCRAFT_RUN_TOKEN"] not in " ".join(validated.gradle_argv)


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_process_timeout_cleans_both_groups_and_retains_scratch_if_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_failure: bool,
) -> None:
    validated = _prepared(tmp_path)
    terminated: list[object] = []
    private_directories: list[Path] = []

    class FakeProcess:
        pid = 12003

        def poll(self) -> None:
            return None

        def wait(self, *, timeout: float) -> int:
            raise subprocess.TimeoutExpired("gradle", timeout=timeout)

    service, gradle = FakeProcess(), FakeProcess()
    children = iter((service, gradle))

    def fake_environment(
        environment: dict[str, str], *, temporary_directory: Path
    ) -> tuple[dict[str, str], dict[str, str], str]:
        private_directories.append(temporary_directory)
        return dict(environment), {"schema": "neurocraft.execution_environment.v2"}, "/pinned/uv"

    def cleanup(observed: object, **_kwargs: object) -> None:
        assert private_directories[0].is_dir()
        terminated.append(observed)
        if cleanup_failure and observed is gradle:
            raise OSError("group still exists")

    monkeypatch.setattr(orchestrator, "_tcp_ready", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(orchestrator.subprocess, "Popen", lambda *_args, **_kwargs: next(children))
    monkeypatch.setattr(orchestrator, "_execution_environment", fake_environment)
    monkeypatch.setattr(orchestrator, "_wait_for_service", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(orchestrator, "_terminate_child", cleanup)
    expected = "private scratch retained" if cleanup_failure else "exceeded its timeout"
    try:
        with pytest.raises(orchestrator.OrchestrationError, match=expected) as caught:
            orchestrator._run_processes(
                validated,
                environment={"NEUROCRAFT_RUN_TOKEN": "C" * 43},
                readiness_timeout=1.0,
                gametest_timeout=2.0,
                termination_timeout=1.0,
            )
        assert isinstance(caught.value, orchestrator.ProcessCleanupError) is cleanup_failure
        assert terminated == [gradle, service]
        assert len(private_directories) == 1
        assert private_directories[0].exists() is cleanup_failure
    finally:
        if private_directories and private_directories[0].exists():
            shutil.rmtree(private_directories[0])


@pytest.mark.skipif(os.name == "nt", reason="external evidence runner uses POSIX session groups")
@pytest.mark.parametrize("wrapper_exits", [False, True])
def test_real_gradle_descendant_stops_before_scratch_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wrapper_exits: bool
) -> None:
    validated = _prepared(tmp_path)
    child_pid = tmp_path / "descendant.pid"
    stopped = tmp_path / "scratch-exists-on-stop.txt"
    child_code = (
        "import os,signal,time; from pathlib import Path; "
        f"marker=Path({str(stopped)!r}); "
        "signal.signal(signal.SIGTERM,lambda *_: "
        "(marker.write_text(str(Path(os.environ['TMPDIR']).is_dir())),exit(0))); "
        f"Path({str(child_pid)!r}).write_text(str(os.getpid())); "
        "time.sleep(30)"
    )
    wrapper_code = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
        f"marker=Path({str(child_pid)!r}); "
        "deadline=time.monotonic()+5; "
        "exec('while not marker.exists() and time.monotonic()<deadline: time.sleep(0.01)'); "
        f"time.sleep({0 if wrapper_exits else 30})"
    )
    validated = replace(
        validated,
        service_argv=(sys.executable, "-c", "import time; time.sleep(30)"),
        gradle_argv=(sys.executable, "-c", wrapper_code),
    )
    private_directories: list[Path] = []
    children: list[subprocess.Popen[bytes]] = []
    real_popen = subprocess.Popen
    unrelated = real_popen([sys.executable, "-c", "import time; time.sleep(30)"])

    def popen(argv: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        process = real_popen(argv, **kwargs)
        children.append(process)
        return process

    def environment(
        _environment: object, *, temporary_directory: Path
    ) -> tuple[dict[str, str], dict[str, str], str]:
        private_directories.append(temporary_directory)
        return {**os.environ, "TMPDIR": str(temporary_directory)}, {"scope": "test"}, sys.executable

    monkeypatch.setattr(orchestrator.subprocess, "Popen", popen)
    monkeypatch.setattr(orchestrator, "_execution_environment", environment)
    monkeypatch.setattr(orchestrator, "_tcp_ready", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(orchestrator, "_wait_for_service", lambda *_args, **_kwargs: None)
    try:
        if wrapper_exits:
            assert orchestrator._run_processes(
                validated,
                environment={},
                readiness_timeout=1,
                gametest_timeout=2,
                termination_timeout=2,
            ) == {"scope": "test"}
        else:
            with pytest.raises(orchestrator.OrchestrationError, match="exceeded its timeout"):
                orchestrator._run_processes(
                    validated,
                    environment={},
                    readiness_timeout=1,
                    gametest_timeout=1,
                    termination_timeout=2,
                )
        assert child_pid.exists(), "stand-in must actually start a descendant"
        assert stopped.read_text() == "True"
        assert not private_directories[0].exists()
        assert unrelated.poll() is None
        with pytest.raises(ProcessLookupError):
            os.kill(int(child_pid.read_text()), 0)
    finally:
        for process in children:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)
        unrelated.terminate()
        unrelated.wait(timeout=3)
        if child_pid.exists():
            with suppress(ProcessLookupError):
                os.kill(int(child_pid.read_text()), signal.SIGKILL)


def test_canonical_receipt_writer_is_atomic_and_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    receipt = {"schema": "test.v1", "status": "ok"}

    digest = orchestrator._atomic_write_canonical(path, receipt)

    assert path.read_bytes() == canonical_json_bytes(receipt) + b"\n"
    assert len(digest) == 64
    with pytest.raises(FileExistsError, match="refusing to clobber"):
        orchestrator._atomic_write_canonical(path, receipt)


def test_validator_pins_identity_labels_to_the_profile_row(tmp_path: Path) -> None:
    """A wing bundle whose identity borrows the v1 run/entity labels must fail.

    Every cryptographic hash inside such a bundle can remain truthful — the
    labels are the only lie — so the profile row's pinned identifiers are the
    check that catches it (adversarial-review finding F1).
    """

    from hashlib import sha256

    output = _prepared_steering(tmp_path)
    identity = json.loads((output / "identity.json").read_text())
    identity["run_id"] = "external-bridge-gametest-v1"
    identity["entity_id"] = "c1e5707a-45b8-59b2-b792-4c9b9a546e93"
    encoded = canonical_json_bytes(identity) + b"\n"
    (output / "identity.json").write_bytes(encoded)
    launch = json.loads((output / "launch.json").read_text())
    launch["identity"] = identity
    launch["prepared_file_sha256"]["identity.json"] = sha256(encoded).hexdigest()
    (output / "launch.json").write_bytes(canonical_json_bytes(launch) + b"\n")
    with pytest.raises(
        orchestrator.OrchestrationError,
        match="do not match the launch controller profile",
    ):
        orchestrator.validate_bundle(output, enforce_bundle_location=False)
