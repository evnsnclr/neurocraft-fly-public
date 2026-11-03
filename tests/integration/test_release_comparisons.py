"""Synthetic release-protocol evidence checks; no live game, downloads or neural work."""

from __future__ import annotations

import copy
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import run_release_comparisons as release

POLICY = "8" * 64


@pytest.fixture
def plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Isolate runner identities from concurrent source edits during implementation.
    root = tmp_path / "protocol source"
    for name in release.FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic identity for {name}\n")
    original = release.declaration
    monkeypatch.setattr(release, "declaration", lambda: original(root))
    output = tmp_path / "declared comparison"
    release.make_plan(output)
    return output


def _row(case: release.trials.Case, *, response: bool = True) -> dict[str, Any]:
    return {
        **asdict(case),
        "status": "observed",
        "fixture_verified": True,
        "source_verified": True,
        "issued_response": response,
        "applied_response": response,
        "issued_without_applied_receipt": [],
        "complete_applied_hold": response,
        "start_tick": 10,
        "end_tick": 109,
        "response_window_issued": 100,
        "response_window_missing_receipts": [],
        "response_window_unapplied": [],
        "input_gate_verified": True,
        "policy_hash": POLICY,
        "policy_verified": True,
        "waypoints_verified": True,
        "outcome": "response_applied" if response else "no_applied_response",
        "epoch_identity": _identity(case),
    }


def _identity(case: release.trials.Case) -> dict[str, Any]:
    return release.trials.RunIdentity(
        run_id=f"run-{case.repetition}",
        session_epoch=case.index + 1,
        entity_id=f"fly-{case.repetition}",
        graph_hash=("2" if case.lane == "shuffled_weight" else "1") * 64,
        adapter_hash=("4" if case.lane == "no_input" else "3") * 64,
        dynamics_hash="5" * 64,
        decoder_hash="6" * 64,
        observation_schema_hash="7" * 64,
    ).to_dict()


def _models() -> dict[str, dict[str, Any]]:
    return {
        lane: {
            key: value
            for key, value in _identity(release.trials.Case(0, lane, 0, "brush")).items()
            if key.endswith("_hash")
        }
        for lane in release.trials.LANES
    }


def _block(plan: Path, block: int) -> tuple[Path, dict[str, Any]]:
    if not (plan / "models.json").exists():
        release.write_new(plan / "models.json", {"schema": release.SCHEMA, "models": _models()})
    if not (plan / "policy.json").exists():
        release.write_new(plan / "policy.json", {"schema": release.SCHEMA, "policy_hash": POLICY})
    folder = plan / f"block-{block}"
    folder.mkdir()
    result = {
        "schema": release.trials.SCHEMA,
        "cases": [
            _row(case, response=case.lane != "no_input") for case in release.block_cases(block)
        ],
        "original_restored": True,
        "all_cases_observed": True,
        "failure": None,
    }
    release.write_new(folder / "summary.json", result)
    release.write_new(
        folder / "block.json",
        {
            "schema": release.SCHEMA,
            "block": block,
            "plan_sha256": release.sha(plan / "plan.json"),
            "session_sha256": str(block + 1) * 64,
            "lane_catalog_sha256": "a" * 64,
            "run_id": f"run-{block}",
            "model_identities": _models(),
        },
    )
    return folder, result


def _replace(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value))


def test_schedule_has_54_balanced_core_cases_and_three_separate_restorations() -> None:
    all_core = []
    for block in range(3):
        cases = release.block_cases(block)
        assert len(cases) == 19
        assert [case.index for case in cases] == list(range(19))
        assert all(case.repetition == block for case in cases)
        expected_order = release.trials.LANES[block:] + release.trials.LANES[:block]
        for index, source in enumerate(release.trials.SOURCES):
            group = cases[index * 3 : index * 3 + 3]
            assert [case.source for case in group] == [source] * 3
            assert tuple(case.lane for case in group) == expected_order
        assert cases[18] == release.trials.Case(18, "original", block, "brush")
        all_core.extend(cases[:18])
    assert len(all_core) == 54
    assert Counter((case.source, case.lane) for case in all_core) == {
        (source, lane): 3 for source in release.trials.SOURCES for lane in release.trials.LANES
    }
    assert release.ReleaseRunner.reset_each_case is True
    assert release.trials.Runner.reset_each_case is False
    assert release.trials.Runner.native_targeting == "strict"
    assert release.trials.Runner.player_locomotion == "scripted"


@pytest.mark.parametrize("block", [-1, 3, True, 0.0, "0"])
def test_schedule_rejects_invalid_block(block: Any) -> None:
    with pytest.raises(release.trials.TrialError):
        release.block_cases(block)


def test_plan_is_verifiable_and_cannot_be_overwritten(plan: Path) -> None:
    before = {path.name: path.read_bytes() for path in plan.iterdir()}
    assert release.verify_plan(plan)["core_cases"] == 54
    with pytest.raises(FileExistsError):
        release.make_plan(plan)
    assert {path.name: path.read_bytes() for path in plan.iterdir()} == before


@pytest.mark.parametrize("rehash", [False, True])
def test_plan_mutation_refuses_even_with_recomputed_checksum(plan: Path, rehash: bool) -> None:
    value = release.read_json(plan / "plan.json")
    value["blocks"][0][0]["source"] = "brush"
    _replace(plan / "plan.json", value)
    if rehash:
        (plan / "plan.sha256").write_text(release.sha(plan / "plan.json") + "\n")
    with pytest.raises(release.trials.TrialError, match="plan"):
        release.verify_plan(plan)


def test_changed_runner_source_invalidates_frozen_plan(plan: Path) -> None:
    root = plan.parent / "protocol source"
    (root / release.FILES[0]).write_text("changed protocol implementation\n")
    with pytest.raises(release.trials.TrialError, match="runner bytes"):
        release.verify_plan(plan)


def test_missing_blocks_retain_all_scheduled_cases_as_unqualified(plan: Path) -> None:
    result = release.aggregate(plan)
    assert result["scheduled"] == len(result["cases"]) == 54
    assert result["qualified"] == 0
    assert result["comparison_qualified"] is False
    assert all(row["status"] == "unattempted" for row in result["cases"])
    assert len(result["groups"]) == 18
    assert all(group["scheduled"] == 3 and group["qualified"] == 0 for group in result["groups"])
    assert len(result["restorations"]) == 3
    assert all(not row["handover_confirmed"] for row in result["restorations"])


def test_partial_blocks_never_shrink_denominator(plan: Path) -> None:
    _block(plan, 0)
    result = release.aggregate(plan)
    assert result["scheduled"] == 54
    assert len(result["cases"]) == 54
    assert result["qualified"] == 18
    assert result["comparison_qualified"] is False
    assert sum(row["status"] == "unattempted" for row in result["cases"]) == 36


def test_complete_valid_blocks_can_qualify_including_no_input_nulls(plan: Path) -> None:
    for block in range(3):
        _block(plan, block)
    result = release.aggregate(plan)
    assert result["qualified"] == 54
    assert result["comparison_qualified"] is True
    assert len(result["block_summary_sha256"]) == 3
    assert result["model_identities"] == _models()
    assert result["policy_hash"] == POLICY
    assert all(group["qualified"] == 3 for group in result["groups"])
    assert all(
        group["target_applied"] == (0 if group["lane"] == "no_input" else 3)
        for group in result["groups"]
    )


@pytest.mark.parametrize("mutation", ["handover_failed", "restore_brush_null", "block_failed"])
def test_all_core_cases_cannot_hide_failed_restoration_or_block_failure(
    plan: Path, mutation: str
) -> None:
    for block in range(3):
        folder, result = _block(plan, block)
    if mutation == "handover_failed":
        result["original_restored"] = False
    elif mutation == "restore_brush_null":
        result["cases"][18] = _row(release.block_cases(2)[18], response=False)
    else:
        result["failure"] = "retained post-case failure"
    _replace(folder / "summary.json", result)
    aggregated = release.aggregate(plan)
    assert aggregated["qualified"] == 54
    assert aggregated["comparison_qualified"] is False
    if mutation == "block_failed":
        assert aggregated["block_failures"]["2"] == "retained post-case failure"


@pytest.mark.parametrize(
    "mutation", ["omitted", "duplicate", "reordered", "wrong_plan", "wrong_block"]
)
def test_block_schedule_or_provenance_mutation_refuses(plan: Path, mutation: str) -> None:
    folder, result = _block(plan, 0)
    if mutation == "omitted":
        result["cases"].pop(3)
    elif mutation == "duplicate":
        result["cases"][3] = copy.deepcopy(result["cases"][4])
    elif mutation == "reordered":
        result["cases"][3], result["cases"][4] = result["cases"][4], result["cases"][3]
    else:
        provenance = release.read_json(folder / "block.json")
        provenance["plan_sha256" if mutation == "wrong_plan" else "block"] = (
            "0" * 64 if mutation == "wrong_plan" else 2
        )
        _replace(folder / "block.json", provenance)
    _replace(folder / "summary.json", result)
    with pytest.raises(release.trials.TrialError):
        release.aggregate(plan)


def test_wrong_case_session_cannot_qualify_under_another_block(plan: Path) -> None:
    folder, result = _block(plan, 0)
    result["cases"][4]["epoch_identity"]["run_id"] = "unrelated-session"
    _replace(folder / "summary.json", result)
    with pytest.raises(release.trials.TrialError, match=r"session|run"):
        release.aggregate(plan)


def test_same_session_cannot_supply_multiple_completed_blocks(plan: Path) -> None:
    _block(plan, 0)
    folder, result = _block(plan, 1)
    for row in result["cases"]:
        row["epoch_identity"]["run_id"] = "run-0"
    _replace(folder / "summary.json", result)
    provenance = release.read_json(folder / "block.json")
    provenance["run_id"] = "run-0"
    _replace(folder / "block.json", provenance)
    with pytest.raises(release.trials.TrialError, match=r"session|run"):
        release.aggregate(plan)


@pytest.mark.parametrize(
    "field",
    ["graph_hash", "adapter_hash", "dynamics_hash", "decoder_hash", "observation_schema_hash"],
)
@pytest.mark.parametrize("location", ["same_block", "later_block", "restoration"])
def test_all_cases_and_restorations_must_use_bound_model_identity(
    plan: Path, field: str, location: str
) -> None:
    folder, result = _block(plan, 0)
    if location == "later_block":
        folder, result = _block(plan, 1)
    index = 18 if location == "restoration" else 4
    result["cases"][index]["epoch_identity"][field] = "f" * 64
    _replace(folder / "summary.json", result)
    with pytest.raises(release.trials.TrialError, match="model identity"):
        release.aggregate(plan)


@pytest.mark.parametrize(
    "mutation", ["missing_hash", "invalid_hash", "missing_entity", "missing_epoch", "extra_key"]
)
def test_partial_or_invalid_run_identity_cannot_support_observed_case(
    plan: Path, mutation: str
) -> None:
    folder, result = _block(plan, 0)
    identity = result["cases"][4]["epoch_identity"]
    if mutation == "missing_hash":
        del identity["dynamics_hash"]
    elif mutation == "invalid_hash":
        identity["decoder_hash"] = "not-a-hash"
    elif mutation == "missing_entity":
        del identity["entity_id"]
    elif mutation == "missing_epoch":
        del identity["session_epoch"]
    else:
        identity["unrecognized"] = "value"
    _replace(folder / "summary.json", result)
    with pytest.raises((release.trials.TrialError, ValueError)):
        release.aggregate(plan)


def test_model_binding_cannot_change_after_first_block(plan: Path) -> None:
    _block(plan, 0)
    value = release.read_json(plan / "models.json")
    value["models"]["original"]["graph_hash"] = "f" * 64
    _replace(plan / "models.json", value)
    with pytest.raises(release.trials.TrialError, match="bound models"):
        release.aggregate(plan)


@pytest.mark.parametrize(
    "mutation", ["changed_case", "changed_binding", "missing_binding", "missing_case_policy"]
)
def test_policy_binding_cannot_be_omitted_or_changed(plan: Path, mutation: str) -> None:
    folder, result = _block(plan, 0)
    if mutation == "changed_case":
        result["cases"][4]["policy_hash"] = "f" * 64
    elif mutation == "missing_case_policy":
        del result["cases"][4]["policy_hash"]
    elif mutation == "changed_binding":
        _replace(plan / "policy.json", {"schema": release.SCHEMA, "policy_hash": "f" * 64})
    else:
        (plan / "policy.json").unlink()
    _replace(folder / "summary.json", result)
    with pytest.raises(release.trials.TrialError, match="policy"):
        release.aggregate(plan)


@pytest.mark.parametrize("policy", [None, "invalid", "f" * 64])
def test_initial_baseline_refuses_missing_or_changed_policy_before_scene_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str | None
) -> None:
    output = tmp_path / "block-0"
    output.mkdir()
    original = {"schema": release.SCHEMA, "policy_hash": POLICY}
    release.write_new(tmp_path / "policy.json", original)
    runner = object.__new__(release.ReleaseRunner)
    runner.output = output
    calls = []
    runner.epoch = SimpleNamespace(latest=lambda: {"policy_hash": policy})
    monkeypatch.setattr(
        release.trials.Runner, "observe_current_epoch", lambda self: calls.append("baseline")
    )
    with pytest.raises(release.trials.TrialError, match="policy"):
        runner.observe_current_epoch()
    assert calls == ["baseline"]
    assert release.read_json(tmp_path / "policy.json") == original
    assert not hasattr(runner, "policy_hash")


def test_initial_policy_is_bound_once_and_every_handover_rechecks_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = object.__new__(release.ReleaseRunner)
    runner.output = tmp_path / "block-0"
    runner.output.mkdir()
    row = {"policy_hash": POLICY}
    runner.epoch = SimpleNamespace(latest=lambda: row)
    monkeypatch.setattr(release.trials.Runner, "observe_current_epoch", lambda self: None)
    monkeypatch.setattr(release.trials.Runner, "handover", lambda self, lane: None)
    runner.observe_current_epoch()
    assert runner.policy_hash == POLICY
    runner.handover("original")
    assert release.read_json(tmp_path / "policy.json")["policy_hash"] == POLICY
    row["policy_hash"] = "f" * 64
    with pytest.raises(release.trials.TrialError, match="policy"):
        runner.handover("shuffled_weight")


def test_already_attempted_block_refuses_without_touching_failed_output(
    plan: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = plan / "block-0"
    folder.mkdir()
    marker = folder / "preflight_failure.json"
    marker.write_text('{"failure":"retained"}')
    monkeypatch.setattr(
        release, "ReleaseRunner", lambda *a, **k: pytest.fail("must refuse before opening inboxes")
    )
    with pytest.raises(release.trials.TrialError, match="already attempted"):
        release.run_block(plan, plan.parent / "absent-run", 0)
    assert marker.read_text() == '{"failure":"retained"}'


def test_same_session_refuses_before_constructing_second_runner(
    plan: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _block(plan, 0)
    run = plan.parent / "existing session"
    run.mkdir()
    (run / "session.json").write_text(
        json.dumps({"run_id": "run-0", "requested_capture_seconds": 720})
    )
    monkeypatch.setattr(
        release, "ReleaseRunner", lambda *a, **k: pytest.fail("must refuse before opening inboxes")
    )
    with pytest.raises(release.trials.TrialError, match=r"session|run"):
        release.run_block(plan, run, 1)
    assert not (plan / "block-1").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        {"source_verified": False},
        {"fixture_verified": False},
        {"status": "stimulus_failed"},
        {"issued_without_applied_receipt": [7]},
        {"response_window_missing_receipts": [8]},
        {"response_window_unapplied": [8]},
        {"response_window_issued": 99},
        {"input_gate_verified": False},
        {"policy_verified": False},
        {"start_tick": True},
        {"end_tick": 110},
        {"issued_response": True, "applied_response": False},
        {"issued_response": True, "applied_response": True, "complete_applied_hold": False},
    ],
)
def test_missing_source_or_receipt_cannot_be_a_clean_null(mutation: dict[str, Any]) -> None:
    row = _row(release.trials.Case(2, "no_input", 0, "brush"), response=False)
    row.update(mutation)
    assert release.qualify_case(row) is False


def test_valid_no_input_null_and_valid_applied_response_both_qualify() -> None:
    assert (
        release.qualify_case(_row(release.trials.Case(2, "no_input", 0, "brush"), response=False))
        is True
    )
    assert release.qualify_case(_row(release.trials.Case(0, "original", 0, "brush"))) is True


@pytest.mark.parametrize("source", ["player_approach", "mob_approach"])
def test_approach_commands_use_one_fixed_anchor_for_all_five_points(source: str) -> None:
    commands = release.approach_commands(source)
    assert len(commands) == len(set(commands)) == 5
    assert all(f"execute at {release.ANCHOR} run tp" in command for command in commands)
    assert all(release.trials.FLY not in command for command in commands)
    target = release.trials.PLAYER if source == "player_approach" else release.trials.COW
    for command, distance in zip(commands, release.DISTANCES, strict=True):
        assert f"run tp {target} " in command
        assert command.endswith(f"^{distance} ~180 0")


@pytest.mark.parametrize("source", ["player_approach", "mob_approach"])
def test_approach_requires_verified_waypoints_even_when_source_and_response_exist(
    source: str,
) -> None:
    row = _row(release.trials.Case(0, "original", 0, source))
    assert release.qualify_case(row) is True
    row["waypoints_verified"] = False
    assert release.qualify_case(row) is False
    del row["waypoints_verified"]
    assert release.qualify_case(row) is False


@pytest.mark.parametrize("source", ["player_approach", "mob_approach"])
def test_fixed_approach_delivers_all_points_without_reading_readouts_and_waits_after_confirmation(
    source: str,
) -> None:
    commands: list[str] = []
    waits: list[int] = []
    queries: list[tuple[str, str]] = []
    clock = [500]
    position_index = [0]
    runner = object.__new__(release.ReleaseRunner)
    runner.results = [{}]
    runner.send = commands.append  # type: ignore[method-assign]
    runner.record = lambda *a, **k: None  # type: ignore[method-assign]
    # No policy/readout key is available: adaptation would raise instead of silently passing.
    runner.epoch = SimpleNamespace(latest=lambda: {"observation_tick": clock[0]})

    def vector(target: str, field: str, deadline: float) -> list[float]:
        queries.append((target, field))
        if target == release.ANCHOR:
            return [2.0, 66.5, 3.0] if field == "Pos" else [90.0, 0.0]
        distance = release.DISTANCES[position_index[0]]
        position_index[0] += 1
        # A deliberately large delay versus case start catches stale start_tick waits.
        clock[0] += 11
        return [2.0 - distance, 65.0 if source == "player_approach" else 66.5, 3.0]

    def observed(tick: int, deadline: float) -> dict[str, Any]:
        assert tick == clock[0] + 6
        waits.append(tick)
        clock[0] = tick
        return {"observation_tick": tick}

    def query(command: str, pattern: str, deadline: float) -> str:
        assert command == "time query gametime"
        assert pattern == r"The time is (\d+)"
        clock[0] += 13
        return str(clock[0])

    runner.server_vector = vector  # type: ignore[method-assign]
    runner.server_query = query  # type: ignore[method-assign]
    runner.observed_after = observed  # type: ignore[method-assign]
    runner.perform_approach(release.trials.Case(0, "original", 0, source), 10, 9999.0)
    points = runner.results[0]["approach_waypoints"]
    assert len(points) == len(waits) == position_index[0] == 5
    assert commands[-1] == release.approach_commands(source)[-1]
    assert [command for command in commands if " run tp " in command and "^0" in command] == list(
        release.approach_commands(source)
    )
    assert sum(target == release.ANCHOR and field == "Pos" for target, field in queries) == 1
    assert all(point["position_confirmed"] is True for point in points)
    assert all(point["hold_end_tick"] - point["confirmed_tick"] == 6 for point in points)


def _input(lane: str = "original") -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    request = {
        "command_seq": 4,
        "observation_seq": 4,
        "observation_tick": 20,
        "state_digest": "a" * 64,
        "policy_hash": POLICY,
    }
    identity = _identity(release.trials.Case(0, lane, 0, "brush"))
    sample = {
        "schema": "neurocraft.program_input_diagnostic.v1",
        "bound": True,
        "identity": identity.copy(),
        **request,
        "encoded_stimulation": {"input_a": 0.0 if lane == "no_input" else 1.0},
        "before_control": {"input_a": 1.0} if lane == "no_input" else None,
    }
    return sample, request, identity


@pytest.mark.parametrize("lane", ["original", "shuffled_weight", "no_input"])
def test_input_sidecar_matches_bound_request_and_actual_gate(lane: str) -> None:
    sample, request, identity = _input(lane)
    assert release.input_matches(sample, request, identity, lane) is True


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "bound",
        "identity",
        "command_seq",
        "observation_seq",
        "observation_tick",
        "state_digest",
        "nan",
        "infinite",
        "boolean",
        "empty",
        "unexpected_before_control",
    ],
)
def test_input_sidecar_refuses_missing_or_wrong_bound_evidence(mutation: str) -> None:
    sample, request, identity = _input()
    if mutation in ("nan", "infinite", "boolean", "empty"):
        sample["encoded_stimulation"] = (
            {}
            if mutation == "empty"
            else {"input_a": {"nan": math.nan, "infinite": math.inf, "boolean": True}[mutation]}
        )
    elif mutation == "unexpected_before_control":
        sample["before_control"] = {"input_a": 1.0}
    else:
        sample[mutation] = "wrong"
    assert release.input_matches(sample, request, identity, "original") is False


@pytest.mark.parametrize(
    "mutation", ["nonzero_after_gate", "no_before_control", "wrong_keys", "nan", "boolean"]
)
def test_no_input_needs_explicit_before_gate_and_zero_actual_drive(mutation: str) -> None:
    sample, request, identity = _input("no_input")
    if mutation == "nonzero_after_gate":
        sample["encoded_stimulation"]["input_a"] = 1.0
    elif mutation == "no_before_control":
        sample["before_control"] = None
    elif mutation == "wrong_keys":
        sample["before_control"] = {"another_input": 1.0}
    else:
        sample["before_control"]["input_a"] = math.nan if mutation == "nan" else True
    assert release.input_matches(sample, request, identity, "no_input") is False


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "missing_point",
        "short_hold",
        "outside_window",
        "wrong_position",
        "changed_window_policy",
        "missing_window_policy",
        "partial_window",
    ],
)
def test_run_case_qualification_requires_all_five_confirmed_held_waypoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    case = release.trials.Case(0, "original", 0, "player_approach")
    runner = object.__new__(release.ReleaseRunner)
    row = _row(case)
    runner.policy_hash = POLICY
    row["approach_waypoints"] = [
        {"position_confirmed": True, "confirmed_tick": 20 + i * 10, "hold_end_tick": 26 + i * 10}
        for i in range(5)
    ]
    if mutation == "missing_point":
        row["approach_waypoints"].pop()
    elif mutation == "short_hold":
        row["approach_waypoints"][2]["hold_end_tick"] -= 1
    elif mutation == "outside_window":
        row["approach_waypoints"][4]["hold_end_tick"] = row["end_tick"] + 1
    elif mutation == "wrong_position":
        row["approach_waypoints"][2]["position_confirmed"] = False
    runner.results = [row]
    requests = {}
    samples = []
    for seq, tick in enumerate(range(row["start_tick"], row["end_tick"] + 1)):
        sample, request, identity = _input()
        request.update(command_seq=seq, observation_seq=seq, observation_tick=tick)
        sample.update(request)
        requests[seq] = request
        samples.append(sample)
    if mutation == "changed_window_policy":
        requests[50]["policy_hash"] = "f" * 64
    elif mutation == "missing_window_policy":
        del requests[50]["policy_hash"]
    elif mutation == "partial_window":
        del requests[50]
    (tmp_path / "program-decisions.inputs.ndjson").write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples)
    )
    runner.epoch = SimpleNamespace(path=tmp_path, identity=identity, issued=requests)
    runner.record = lambda *a, **k: None  # type: ignore[method-assign]
    monkeypatch.setattr(release.trials.Runner, "run_case", lambda *a: None)
    runner.run_case(case)
    assert row["input_gate_verified"] is (mutation != "partial_window")
    assert row["input_samples_matched"] == (99 if mutation == "partial_window" else 100)
    assert row["waypoints_verified"] is (
        mutation in ("none", "changed_window_policy", "missing_window_policy", "partial_window")
    )
    assert row["policy_verified"] is (
        mutation not in ("changed_window_policy", "missing_window_policy")
    )
    assert release.qualify_case(row) is (mutation == "none")


@pytest.mark.parametrize("metadata_failure", [False, True])
def test_block_cleanup_preserves_failure_and_does_not_double_close_success(
    plan: Path, monkeypatch: pytest.MonkeyPatch, metadata_failure: bool
) -> None:
    run = plan.parent / "fresh session"
    run.mkdir()
    (run / "session.json").write_text(
        json.dumps({"run_id": "fresh-run", "requested_capture_seconds": 720})
    )
    (run / "lane-catalog.json").write_text("{}")
    resources = []

    class Resource:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            if self.closed:
                raise OSError(9, "Bad file descriptor")
            self.closed = True

    class FakeRunner:
        def __init__(self, run: Path, output: Path, cases: Any, **kwargs: Any) -> None:
            output.mkdir()
            self.session = {"run_id": "fresh-run"}
            self.catalog = {
                lane: {"identity": _identity(release.trials.Case(0, lane, 0, "brush"))}
                for lane in release.trials.LANES
            }
            self.resources = [Resource(), Resource()]
            resources.extend(self.resources)

        def execute(self) -> dict[str, Any]:
            assert not metadata_failure, "must not execute after metadata failure"
            # The real legacy Runner owns normal successful-path resource cleanup.
            for resource in self.resources:
                resource.close()
            return {"all_cases_observed": True, "original_restored": True}

    monkeypatch.setattr(release, "ReleaseRunner", FakeRunner)
    original_write = release.write_new

    def write(path: Path, data: dict[str, Any]) -> None:
        if metadata_failure and path.name == "block.json":
            raise OSError("synthetic metadata write failure")
        original_write(path, data)

    monkeypatch.setattr(release, "write_new", write)
    if metadata_failure:
        with pytest.raises(OSError, match="metadata write failure"):
            release.run_block(plan, run, 0)
        failure = release.read_json(plan / "block-0/block_failure.json")
        assert "metadata write failure" in failure["failure"]
    else:
        assert release.run_block(plan, run, 0) == {
            "all_cases_observed": True,
            "original_restored": True,
        }
    assert len(resources) == 2 and all(resource.closed for resource in resources)


def test_short_capture_refuses_before_opening_runner(
    plan: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = plan.parent / "short capture"
    run.mkdir()
    (run / "session.json").write_text(
        json.dumps({"run_id": "fresh-run", "requested_capture_seconds": 120})
    )
    monkeypatch.setattr(
        release, "ReleaseRunner", lambda *a, **k: pytest.fail("must refuse before opening inboxes")
    )
    with pytest.raises(release.trials.TrialError, match="capture-seconds 720"):
        release.run_block(plan, run, 0)
    assert not (plan / "block-0").exists()
