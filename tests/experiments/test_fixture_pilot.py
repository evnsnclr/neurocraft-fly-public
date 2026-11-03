from __future__ import annotations

import inspect
import json
from collections import Counter
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any

import neurocraft_experiments.pilot as pilot_module
import pytest
from neurocraft_connectome import load_malecns_fixture
from neurocraft_controller import ConnectomeController, ControllerConfiguration
from neurocraft_experiments import (
    RunMode,
    build_pilot_result,
    canonical_sha256,
    load_arena,
    load_evaluation_protocol,
    load_pilot_protocol,
    policy_sensor_fields,
    result_bytes,
    run_pilot_trial,
    select_policy_action,
    verify_reexecution,
    verify_result_envelope,
    write_result_no_clobber,
)
from neurocraft_protocol import SensorFrame

ROOT = Path(__file__).resolve().parents[2]
PILOT_PROTOCOL = ROOT / "experiments/preregistration/fixture-physics-pilot-v3.json"
MINECRAFT_PROTOCOL = ROOT / "experiments/preregistration/control-matrix-v2.json"
FIXTURE = ROOT / "data/fixtures/mcns-v1.0-flight-power-v1.json"
STORED_RESULT = ROOT / "experiments/results/fixture-physics-pilot-v3.json"
HISTORICAL_PROTOCOL = ROOT / "experiments/preregistration/fixture-physics-pilot-v1.json"
HISTORICAL_RESULT = ROOT / "experiments/results/fixture-physics-pilot-v1.json"
EXECUTED_V2_PROTOCOL = ROOT / "experiments/preregistration/fixture-physics-pilot-v2.json"
EXECUTED_V2_RESULT = ROOT / "experiments/results/fixture-physics-pilot-v2.json"


def _mutated_protocol_path(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    *,
    name: str = "mutated-pilot-protocol.json",
) -> Path:
    document = json.loads(PILOT_PROTOCOL.read_text(encoding="utf-8"))
    mutate(document)
    target = tmp_path / name
    target.write_text(json.dumps(document), encoding="utf-8")
    return target


def test_historical_v1_bytes_are_preserved_but_not_current_code_evidence() -> None:
    assert sha256(HISTORICAL_PROTOCOL.read_bytes()).hexdigest() == (
        "c3bb48fbb844e8b4d731a3e85709169ac108c50caf06ca56e5a73c392f56629e"
    )
    assert sha256(HISTORICAL_RESULT.read_bytes()).hexdigest() == (
        "7a9b8ae83dcb364b9e8b0960a661583f94efe7da47c67165780225a45a308081"
    )
    with pytest.raises(ValueError, match=r"fixture_physics_pilot\.v3"):
        load_pilot_protocol(HISTORICAL_PROTOCOL, repository_root=ROOT)


def test_historical_v2_bytes_are_preserved_as_immutable_evidence() -> None:
    assert sha256(EXECUTED_V2_PROTOCOL.read_bytes()).hexdigest() == (
        pilot_module.EXECUTED_V2_PROTOCOL_FILE_SHA256
    )
    assert sha256(EXECUTED_V2_RESULT.read_bytes()).hexdigest() == (
        pilot_module.EXECUTED_V2_RESULT_FILE_SHA256
    )
    v2_payload = json.loads(EXECUTED_V2_PROTOCOL.read_text(encoding="utf-8"))
    assert canonical_sha256(v2_payload) == pilot_module.EXECUTED_V2_PROTOCOL_PAYLOAD_SHA256


def test_v3_bytes_are_preserved_and_its_implementation_pin_is_historical() -> None:
    """v3 joins v1 and v2 as frozen history; its implementation pin is recorded, not verified.

    The pin included ``pilot.py``'s own bytes, so it was self-sealing, and a preregistration is
    append-only, so it could not be refreshed when a pinned file legitimately changed. The bytes
    stay frozen and pinned; what a load no longer does is compare them to the working tree.
    """

    assert sha256(PILOT_PROTOCOL.read_bytes()).hexdigest() == (
        pilot_module.EXECUTED_V3_PROTOCOL_FILE_SHA256
    )
    assert sha256(STORED_RESULT.read_bytes()).hexdigest() == (
        pilot_module.EXECUTED_V3_RESULT_FILE_SHA256
    )
    assert len(pilot_module.IMPLEMENTATION_HASH_TREE_COMMIT) == 40

    protocol = load_pilot_protocol(PILOT_PROTOCOL, repository_root=ROOT)
    document = json.loads(PILOT_PROTOCOL.read_text(encoding="utf-8"))
    assert dict(protocol.implementation_hashes) == document["implementation_hashes"]
    # The result envelope binds the canonical protocol payload, so the recorded hashes still
    # cannot be edited without breaking scripts/verify_fixture_pilot_snapshot.py.
    envelope = json.loads(STORED_RESULT.read_text(encoding="utf-8"))
    assert envelope["result"]["protocol_sha256"] == protocol.sha256


def test_recorded_implementation_hashes_are_not_read_from_the_working_tree(
    tmp_path: Path,
) -> None:
    """A load succeeds in a root that contains no pinned implementation file at all."""

    document = json.loads(PILOT_PROTOCOL.read_text(encoding="utf-8"))
    root = tmp_path / "repo"
    for relative in [document["fixture"]["path"], *[e["path"] for e in document["arenas"]]]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    for relative in document["implementation_hashes"]:
        assert not (root / relative).exists()

    protocol_file = tmp_path / "recorded-protocol.json"
    protocol_file.write_text(json.dumps(document), encoding="utf-8")
    loaded = load_pilot_protocol(protocol_file, repository_root=root)
    assert dict(loaded.implementation_hashes) == document["implementation_hashes"]


def test_recorded_implementation_hashes_still_fail_closed_on_shape_and_path_escape(
    tmp_path: Path,
) -> None:
    """Dropping the byte comparison did not drop the shape or the containment check."""

    mutated = _mutated_protocol_path(
        tmp_path,
        lambda document: document["implementation_hashes"].clear(),
        name="empty-hashes.json",
    )
    with pytest.raises(ValueError, match="implementation_hashes cannot be empty"):
        load_pilot_protocol(mutated, repository_root=ROOT)

    def bad_hash(document: dict[str, Any]) -> None:
        first = next(iter(document["implementation_hashes"]))
        document["implementation_hashes"][first] = "NOTAHASH"

    mutated = _mutated_protocol_path(tmp_path, bad_hash, name="bad-hash.json")
    with pytest.raises(ValueError, match="implementation hash must be lowercase SHA-256"):
        load_pilot_protocol(mutated, repository_root=ROOT)

    def traversal(document: dict[str, Any]) -> None:
        document["implementation_hashes"]["../outside.py"] = "0" * 64

    mutated = _mutated_protocol_path(tmp_path, traversal, name="traversal-hash.json")
    with pytest.raises(ValueError, match="implementation_hashes path must be repository-relative"):
        load_pilot_protocol(mutated, repository_root=ROOT)


def test_pilot_v2_document_refused_by_v3_loader() -> None:
    with pytest.raises(ValueError, match="frozen historical evidence"):
        load_pilot_protocol(EXECUTED_V2_PROTOCOL, repository_root=ROOT)


def test_executed_v2_lineage_hashes_recorded_in_v3_amendment() -> None:
    document = json.loads(PILOT_PROTOCOL.read_text(encoding="utf-8"))
    amendment = document["amendment"]
    assert amendment["supersedes"] == "fixture-physics-pilot-v2"
    assert amendment["timing"] == "after_v2_execution_before_v3_execution"
    assert amendment["prior_protocol_file_sha256"] == (
        sha256(EXECUTED_V2_PROTOCOL.read_bytes()).hexdigest()
    )
    assert amendment["prior_protocol_payload_sha256"] == (
        canonical_sha256(json.loads(EXECUTED_V2_PROTOCOL.read_text(encoding="utf-8")))
    )
    assert amendment["prior_result_file_sha256"] == (
        sha256(EXECUTED_V2_RESULT.read_bytes()).hexdigest()
    )
    assert amendment["prior_protocol_file_sha256"] == (
        pilot_module.EXECUTED_V2_PROTOCOL_FILE_SHA256
    )
    assert amendment["prior_protocol_payload_sha256"] == (
        pilot_module.EXECUTED_V2_PROTOCOL_PAYLOAD_SHA256
    )
    assert amendment["prior_result_file_sha256"] == pilot_module.EXECUTED_V2_RESULT_FILE_SHA256
    assert "not a design change" in amendment["reason"]


def test_pilot_protocol_is_separate_balanced_and_explicitly_non_minecraft() -> None:
    pilot = load_pilot_protocol(PILOT_PROTOCOL, repository_root=ROOT)
    minecraft = load_evaluation_protocol(MINECRAFT_PROTOCOL)

    assert pilot.planned_trial_count == 60
    assert len(pilot.arena_paths) == 2
    assert len(pilot.seeds) == 5
    assert pilot.replicates == 1
    assert {condition.mode for condition in pilot.conditions} == {
        RunMode.FAITHFUL,
        RunMode.SHUFFLED_WEIGHT,
        RunMode.DEGREE_PRESERVING_RANDOM,
        RunMode.LESION,
        RunMode.SCRIPTED_BASELINE,
        RunMode.NO_INPUT,
    }
    assert "not Minecraft" in pilot.claim_boundary
    assert minecraft.status == "preregistered_no_trials_run"
    assert minecraft.planned_trial_count == 640


def test_single_trial_is_deterministic_and_binds_all_frozen_identities() -> None:
    protocol = load_pilot_protocol(PILOT_PROTOCOL, repository_root=ROOT)
    graph = load_malecns_fixture(FIXTURE)
    arena = load_arena(ROOT / protocol.arena_paths[0])
    condition = next(item for item in protocol.conditions if item.mode is RunMode.FAITHFUL)

    first = run_pilot_trial(graph, protocol, arena, condition, seed=101, replicate=0)
    second = run_pilot_trial(graph, protocol, arena, condition, seed=101, replicate=0)

    assert result_bytes(first) == result_bytes(second)
    assert first["status"] == "completed_fixture_physics_trial"
    assert first["condition_mode"] == "faithful"
    assert first["controller_execution"] == "faithful"
    assert first["arena_sha256"] == arena.sha256
    assert first["controller_hashes"]["controller_run_identity_sha256"]
    assert first["experiment_identity_sha256"] != first["trial_identity_sha256"]


def test_every_control_is_separately_labelled_and_changes_identity_and_trace() -> None:
    protocol = load_pilot_protocol(PILOT_PROTOCOL, repository_root=ROOT)
    graph = load_malecns_fixture(FIXTURE)
    arena = load_arena(ROOT / protocol.arena_paths[0])
    trials = {
        condition.condition_id: run_pilot_trial(
            graph,
            protocol,
            arena,
            condition,
            seed=307,
            replicate=0,
        )
        for condition in protocol.conditions
    }
    faithful = trials["faithful"]

    assert len({trial["experiment_identity_sha256"] for trial in trials.values()}) == 6
    for control_id in (
        "shuffled-weight",
        "degree-preserving",
        "dng07-lesion",
        "scripted",
        "no-input",
    ):
        assert trials[control_id]["condition_mode"] != "faithful"
        assert (
            trials[control_id]["hashes"]["action_stream_sha256"]
            != faithful["hashes"]["action_stream_sha256"]
        )
    assert trials["scripted"]["controller_execution"] == "not_executed_scripted_baseline"
    assert (
        trials["no-input"]["controller_execution"]
        == "canonical_recurrence_with_zeroed_adapter_inputs"
    )
    assert trials["shuffled-weight"]["controller_hashes"]["weight_overlay_sha256"] != "0" * 64
    assert trials["degree-preserving"]["controller_hashes"]["topology_overlay_sha256"] != "0" * 64


def test_policy_boundary_has_no_target_or_exit_coordinate_access() -> None:
    parameters = set(inspect.signature(select_policy_action).parameters)
    assert parameters == {"condition", "controller", "sensors", "sequence"}
    sensors = SensorFrame(
        position_m=(123.0, 4.0, -55.0),
        velocity_m_s=(0.0, -0.1, 0.0),
        yaw_rad=0.0,
        pitch_rad=0.0,
        on_ground=False,
        horizontal_collision=False,
        vertical_collision=False,
        in_fluid=False,
        local_light=7,
        ray_max_distance_m=6.0,
        ray_distances_m=(2.0, 3.0, 1.0, 4.0, 5.0, 0.5),
    )
    boundary = policy_sensor_fields(sensors)
    assert set(boundary) == {
        "forward_clearance",
        "left_clearance",
        "right_clearance",
        "vertical_velocity",
        "on_ground",
        "local_light",
    }
    assert not any("target" in name or "exit" in name or "position" in name for name in boundary)

    protocol = load_pilot_protocol(PILOT_PROTOCOL, repository_root=ROOT)
    scripted = next(item for item in protocol.conditions if item.mode is RunMode.SCRIPTED_BASELINE)
    first = select_policy_action(scripted, None, sensors, sequence=0)
    moved_pose = SensorFrame(
        position_m=(-999.0, 4.0, 999.0),
        velocity_m_s=sensors.velocity_m_s,
        yaw_rad=sensors.yaw_rad,
        pitch_rad=sensors.pitch_rad,
        on_ground=sensors.on_ground,
        horizontal_collision=sensors.horizontal_collision,
        vertical_collision=sensors.vertical_collision,
        in_fluid=sensors.in_fluid,
        local_light=sensors.local_light,
        ray_max_distance_m=sensors.ray_max_distance_m,
        ray_distances_m=sensors.ray_distances_m,
    )
    assert first.motors == select_policy_action(scripted, None, moved_pose, sequence=0).motors


def test_connectome_policy_feature_evidence_excludes_pose_and_target() -> None:
    protocol = load_pilot_protocol(PILOT_PROTOCOL, repository_root=ROOT)
    faithful = next(item for item in protocol.conditions if item.mode is RunMode.FAITHFUL)
    graph = load_malecns_fixture(FIXTURE)
    controller = ConnectomeController.create(
        graph,
        run_id="target-access-audit",
        entity_id="fixture-audit-fly",
        configuration=ControllerConfiguration(),
    )
    sensors = SensorFrame(
        position_m=(1.0, 2.0, 3.0),
        velocity_m_s=(0.0, 0.0, 0.0),
        yaw_rad=0.0,
        pitch_rad=0.0,
        on_ground=False,
        horizontal_collision=False,
        vertical_collision=False,
        in_fluid=False,
        local_light=10,
        ray_max_distance_m=6.0,
        ray_distances_m=(5.0, 5.0, 4.0, 3.0, 2.0, 1.0),
    )
    step = select_policy_action(faithful, controller, sensors, sequence=0)
    assert set(step.feature_values) == {
        "forward_clearance_fraction",
        "left_clearance_fraction",
        "right_clearance_fraction",
        "local_light_fraction",
    }


def test_result_envelope_and_no_clobber_writer(tmp_path: Path) -> None:
    core = {
        "evidence_scope": "NON_MINECRAFT_LOCAL_FIXTURE_PHYSICS_ONLY",
        "trials": [],
    }
    from neurocraft_experiments.identity import canonical_sha256

    envelope = {
        "execution_environment": {
            "platform": "test-platform",
            "python_implementation": "CPython",
            "python_version": "3.13.5",
        },
        "result": core,
        "result_sha256": canonical_sha256(core),
    }
    assert verify_result_envelope(envelope) == envelope
    output = tmp_path / "pilot.json"
    digest = write_result_no_clobber(output, envelope)
    assert len(digest) == 64
    assert json.loads(output.read_text(encoding="utf-8")) == envelope
    with pytest.raises(FileExistsError):
        write_result_no_clobber(output, envelope)


def test_result_writer_cleans_temporary_file_when_publication_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = {"evidence_scope": "NON_MINECRAFT_LOCAL_FIXTURE_PHYSICS_ONLY", "trials": []}
    from neurocraft_experiments.identity import canonical_sha256

    envelope = {
        "execution_environment": {
            "platform": "test-platform",
            "python_implementation": "CPython",
            "python_version": "3.13.5",
        },
        "result": core,
        "result_sha256": canonical_sha256(core),
    }
    output = tmp_path / "pilot.json"

    def fail_link(_source: Path, _target: Path) -> None:
        raise OSError("injected publication failure")

    monkeypatch.setattr(pilot_module.os, "link", fail_link)
    with pytest.raises(OSError, match="injected"):
        write_result_no_clobber(output, envelope)
    assert not output.exists()
    assert list(tmp_path.glob(".pilot.json.*.tmp")) == []


def test_stored_result_has_every_balanced_cell_and_no_silent_failure() -> None:
    envelope = verify_result_envelope(json.loads(STORED_RESULT.read_text(encoding="utf-8")))
    result = envelope["result"]
    trials = result["trials"]
    counts = Counter((trial["arena_id"], trial["condition_id"]) for trial in trials)

    assert len(trials) == result["planned_trial_count"] == 60
    assert set(counts.values()) == {5}
    assert all(trial["status"] == "completed_fixture_physics_trial" for trial in trials)
    assert result["summary"]["failed_trial_count"] == 0
    assert all(
        not comparison["missing_or_failed_cells"]
        for comparison in result["summary"]["paired_primary_comparisons"]
    )


def test_reexecution_is_exact_for_a_result_built_in_the_current_environment(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "pilot.json"
    protocol = load_pilot_protocol(PILOT_PROTOCOL, repository_root=ROOT)
    local_result = build_pilot_result(protocol, repository_root=ROOT)
    write_result_no_clobber(copied, local_result)

    assert (
        verify_reexecution(copied, protocol, repository_root=ROOT)
        == sha256(copied.read_bytes()).hexdigest()
    )


def _set_tick_limit(document: dict[str, Any], value: int) -> None:
    document["physics"]["tick_limit"] = value


def _set_seed_count(document: dict[str, Any], value: int) -> None:
    document["seeds"] = list(range(value))


def _set_replicates(document: dict[str, Any], value: int) -> None:
    document["replicates"] = value


def _set_bootstrap_samples(document: dict[str, Any], value: int) -> None:
    document["analysis"]["bootstrap_samples"] = value


def _set_lesion_count(document: dict[str, Any], value: int) -> None:
    for condition in document["conditions"]:
        if condition["condition_id"] == "dng07-lesion":
            condition["lesion_body_ids"] = list(range(1, value + 1))


def _set_condition_count(document: dict[str, Any], value: int) -> None:
    faithful = next(
        condition for condition in document["conditions"] if condition["condition_id"] == "faithful"
    )
    extras = [
        {**faithful, "condition_id": f"faithful-extra-{index}"}
        for index in range(value - len(document["conditions"]))
    ]
    document["conditions"] = document["conditions"] + extras


@pytest.mark.parametrize(
    ("mutator", "limit", "message"),
    [
        (_set_tick_limit, 10_000, r"physics\.tick_limit must be an integer <= 10000"),
        (_set_seed_count, 64, r"seeds must contain at most 64 entries"),
        (_set_replicates, 16, r"replicates must be an integer <= 16"),
        (
            _set_bootstrap_samples,
            100_000,
            r"analysis\.bootstrap_samples must be an integer <= 100000",
        ),
        (
            _set_lesion_count,
            4_096,
            r"condition\.lesion_body_ids must contain at most 4096 entries",
        ),
        (_set_condition_count, 16, r"conditions must contain at most 16 entries"),
    ],
    ids=[
        "tick_limit",
        "seeds",
        "replicates",
        "bootstrap_samples",
        "lesion_body_ids",
        "conditions",
    ],
)
def test_pilot_work_bounds(
    tmp_path: Path,
    mutator: Callable[[dict[str, Any], int], None],
    limit: int,
    message: str,
) -> None:
    at_limit = _mutated_protocol_path(
        tmp_path, lambda document: mutator(document, limit), name="at-limit.json"
    )
    protocol = load_pilot_protocol(at_limit, repository_root=ROOT)
    assert protocol.protocol_id == "fixture-physics-pilot-v3"

    over_limit = _mutated_protocol_path(
        tmp_path, lambda document: mutator(document, limit + 1), name="over-limit.json"
    )
    with pytest.raises(ValueError, match=message):
        load_pilot_protocol(over_limit, repository_root=ROOT)


def test_planned_trial_count_product_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    assert pilot_module._bounded_planned_trial_count(2, 125, 10, 40) == 100_000
    with pytest.raises(ValueError, match=r"planned_trial_count 102500 exceeds the 100000-trial"):
        pilot_module._bounded_planned_trial_count(2, 125, 10, 41)

    def exploding_gate(*_args: int) -> int:
        raise ValueError("planned_trial_count gate invoked")

    monkeypatch.setattr(pilot_module, "_bounded_planned_trial_count", exploding_gate)
    with pytest.raises(ValueError, match="planned_trial_count gate invoked"):
        load_pilot_protocol(PILOT_PROTOCOL, repository_root=ROOT)


def test_pilot_absolute_fixture_path_refused(tmp_path: Path) -> None:
    def mutate(document: dict[str, Any]) -> None:
        document["fixture"]["path"] = str(FIXTURE)

    mutated = _mutated_protocol_path(tmp_path, mutate)
    with pytest.raises(ValueError, match=r"fixture\.path must be repository-relative"):
        load_pilot_protocol(mutated, repository_root=ROOT)


def test_pilot_parent_traversal_arena_refused(tmp_path: Path) -> None:
    def mutate(document: dict[str, Any]) -> None:
        document["arenas"][0]["path"] = "experiments/arenas/../arenas/light-corridor-v1.json"

    mutated = _mutated_protocol_path(tmp_path, mutate)
    with pytest.raises(ValueError, match=r"arenas\[\]\.path must be repository-relative"):
        load_pilot_protocol(mutated, repository_root=ROOT)


def _materialized_repository(tmp_path: Path) -> tuple[Path, Path]:
    """Copy every protocol-referenced file into an isolated repository root."""

    document = json.loads(PILOT_PROTOCOL.read_text(encoding="utf-8"))
    root = tmp_path / "repo"
    referenced = [
        *document["implementation_hashes"],
        document["fixture"]["path"],
        *[entry["path"] for entry in document["arenas"]],
    ]
    for relative in referenced:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    protocol_file = tmp_path / "materialized-protocol.json"
    protocol_file.write_text(json.dumps(document), encoding="utf-8")
    return root, protocol_file


def test_pilot_symlinked_arena_escape_refused(tmp_path: Path) -> None:
    root, protocol_file = _materialized_repository(tmp_path)
    loaded = load_pilot_protocol(protocol_file, repository_root=root)
    assert loaded.protocol_id == "fixture-physics-pilot-v3"

    arena_relative = loaded.arena_paths[0]
    escaped = tmp_path / "outside-arena.json"
    escaped.write_bytes((ROOT / arena_relative).read_bytes())
    arena_file = root / arena_relative
    arena_file.unlink()
    arena_file.symlink_to(escaped)

    with pytest.raises(ValueError, match=r"arenas\[\]\.path escapes the repository root"):
        load_pilot_protocol(protocol_file, repository_root=root)


def test_pilot_v3_arena_hash_mismatch_refused(tmp_path: Path) -> None:
    def mutate(document: dict[str, Any]) -> None:
        document["arenas"][0]["file_sha256"] = "1" * 64

    mutated = _mutated_protocol_path(tmp_path, mutate)
    with pytest.raises(ValueError, match="arena file hash mismatch"):
        load_pilot_protocol(mutated, repository_root=ROOT)


def test_pilot_v3_arena_entry_requires_hash_field(tmp_path: Path) -> None:
    def mutate(document: dict[str, Any]) -> None:
        document["arenas"][0] = {"path": document["arenas"][0]["path"]}

    mutated = _mutated_protocol_path(tmp_path, mutate)
    with pytest.raises(ValueError, match=r"arenas\[\] fields must be exactly"):
        load_pilot_protocol(mutated, repository_root=ROOT)


def test_pilot_protocol_oversize_refused(tmp_path: Path) -> None:
    oversized = tmp_path / "oversized-protocol.json"
    oversized.write_bytes(b"{" + b" " * pilot_module.MAX_PROTOCOL_BYTES)
    with pytest.raises(ValueError, match="pilot protocol exceeds"):
        load_pilot_protocol(oversized, repository_root=ROOT)


def test_result_oversize_refused(tmp_path: Path) -> None:
    protocol = load_pilot_protocol(PILOT_PROTOCOL, repository_root=ROOT)
    oversized = tmp_path / "oversized-result.json"
    oversized.write_bytes(b"{" + b" " * pilot_module.MAX_RESULT_BYTES)
    with pytest.raises(ValueError, match="pilot result exceeds"):
        verify_reexecution(oversized, protocol, repository_root=ROOT)
