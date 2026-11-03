"""Tamper and interruption checks for local matched-trial diagnostic attribution."""

from __future__ import annotations

import copy
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from scripts.summarize_interactive_session import DiagnosticError
from scripts.summarize_matched_trials import JAVA_IDENTITY, summarize

from .test_comparison_summary import _fixture as comparison_fixture

Json = dict[str, Any]
GOLDEN = Path(__file__).resolve().parents[2] / "packages/protocol/golden"


def _java(identity: Json) -> Json:
    return {key: identity[value] for key, value in JAVA_IDENTITY.items()}


def _row(kind: str, identity: Json | None = None, trial: Json | None = None, **fields: Any) -> Json:
    return {
        "schema": "neurocraft.matched_trial_ledger.v1",
        "diagnostic_only": True,
        "kind": kind,
        "identity": None if identity is None else _java(identity),
        "trial": trial,
        **fields,
    }


def _write(path: Path, rows: list[Json]) -> None:
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def _rehash(header: Json) -> None:
    header["plan_json"] = json.dumps(header["plan"], separators=(",", ":"))
    header["plan_sha256"] = sha256(header["plan_json"].encode()).hexdigest()


def _fixed_cells() -> list[Json]:
    # Independent expression of the committed Java plan, including its execution order.
    rows: list[Json] = []
    for pair, seed in enumerate((0, 7919, 15838, 23757, 31676, 39595, 47514, 55433, 63352, 71271)):
        for stimulus in ("ENTITY", "BLOCK"):
            for lane in ("original", "shuffled_weight", "no_input"):
                rows.append(
                    {
                        "index": len(rows),
                        "pair": pair,
                        "bodySeed": seed,
                        "stimulus": stimulus,
                        "lane": lane,
                    }
                )
    return rows


def _world(tick: int, entity: str) -> Json:
    return {
        "gameTick": tick,
        "entityId": entity,
        "xBlocks": 0.5,
        "yBlocks": 67.0,
        "zBlocks": 0.5,
        "velocityXBlocksPerTick": 0.0,
        "velocityYBlocksPerTick": 0.0,
        "velocityZBlocksPerTick": 0.0,
        "yawDegrees": 0.0,
        "pitchDegrees": 0.0,
        "horizontalCollision": False,
        "verticalCollision": False,
        "onGround": False,
        "inWater": False,
        "inLava": False,
        "blockLight": 0,
        "skyLight": 15,
        "rayMaxDistanceBlocks": 16.0,
        "rayForwardBlocks": 16.0,
        "rayBackBlocks": 16.0,
        "rayLeftBlocks": 16.0,
        "rayRightBlocks": 16.0,
        "rayUpBlocks": 16.0,
        "rayDownBlocks": 2.0,
        "rayHitKinds": [0, 0, 0, 0, 0, 1],
        "events": None,
    }


def _receipt(identity: Json, tick: int, *, seq: int = 0, escape: bool = False) -> Json:
    value: Json = json.loads((GOLDEN / "program_receipt.json").read_text())
    value.update(
        identity=identity, command_seq=seq, observation_seq=seq, target_tick=tick, applied_tick=tick
    )
    value["world_before"].update(
        position_m=[0.5, 67.0, 0.5], velocity_m_s=[0, 0, 0], yaw_rad=0, pitch_rad=0
    )
    if escape:
        value["requested_program"] = value["executed_program"] = "ESCAPE_JUMP"
    return value


def _build(directory: Path, *, console: bool = False, count: int = 2) -> tuple[Path, list[Json]]:
    comparison_fixture(directory)
    catalog = json.loads((directory / "lane-catalog.json").read_text())
    lanes = {entry["name"]: entry for entry in catalog["lanes"]}
    base = catalog["initial_identity"]["session_epoch"]
    cells = _fixed_cells() if console else _fixed_cells()[:count]
    plan: Json = {
        "schema": "neurocraft.matched_trial_plan.v1",
        "diagnostic_only": True,
        "baseline_ticks": 20 if console else 1,
        "approach_ticks": 60 if console else 1,
        "recovery_ticks": 80 if console else 1,
        "anchor_wait_ticks": 200,
        "anchor": "applied program command 0, observation 0, neural steps 0..5",
        "front_planes": "z=6,5,4,3,2,1; ten ticks each; x centre=.5; block bottom y=67",
        "initial_pose": (
            "feet=(.5,67,.5); yaw=0; pitch=0; velocity=0; dry and airborne; "
            "full sturdy floor exactly 2 blocks below feet"
        ),
        "trials": [
            {
                "index": c["index"],
                "pair": c["pair"],
                "body_seed": c["bodySeed"],
                "stimulus": c["stimulus"],
                "lane": c["lane"],
            }
            for c in _fixed_cells()
        ],
    }
    header = _row(
        "plan",
        plan=plan,
        cell_count=len(cells),
        executed_cells=cells,
        world_origin={"x": 0.0, "y": 0.0, "z": 0.0},
    )
    _rehash(header)
    rows = [header]
    total = plan["baseline_ticks"] + plan["approach_ticks"] + plan["recovery_ticks"]

    def sidecar(lane: str, epoch: int) -> Json:
        identity = {**lanes[lane]["identity"], "session_epoch": epoch}
        target = directory / "epochs" / str(epoch)
        target.mkdir(exist_ok=True)
        (target / "epoch.json").write_text(
            json.dumps(
                {
                    **lanes[lane],
                    "identity": identity,
                    "schema": "neurocraft.lane_epoch.v1",
                    "diagnostic_only": True,
                    "controller_profile": "program",
                }
            )
        )
        return identity

    for cell in cells:
        identity = sidecar(cell["lane"], base + cell["index"] + 1)
        reset_tick = 100 + cell["index"] * (total + 10)
        anchor = reset_tick + 2
        rows.append(_row("trial_start", identity, cell, reset_game_tick=reset_tick))
        for tick in range(reset_tick, anchor + total):
            fields: Json = {
                "game_tick": tick,
                "anchor_offset": None if tick < anchor else tick - anchor,
                "world_before": _world(tick, identity["entity_id"]),
                "stimulus_forward_ray_matches_geometry": tick == anchor + 1,
                "command": None,
                "command_admitted": False,
            }
            if tick == anchor:
                fields.update(
                    command={
                        "identity": _java(identity),
                        "commandSequence": 0,
                        "observationSequence": 0,
                        "observationTick": reset_tick,
                        "targetTick": anchor,
                        "validThroughTick": anchor,
                        "neuralStepStart": 0,
                        "neuralStepEnd": 5,
                        "program": "CRUISE",
                    },
                    command_admitted=True,
                    initial_observation_zero=_world(reset_tick, identity["entity_id"]),
                    program_receipt_offered=_receipt(identity, tick),
                )
            elif tick == anchor + 1:
                fields["program_receipt_offered"] = _receipt(identity, tick, seq=1, escape=True)
            rows.append(_row("tick", identity, cell, **fields))
        rows.append(
            _row(
                "trial_end",
                identity,
                cell,
                game_tick=anchor + total,
                anchor_game_tick=anchor,
                outcome="complete",
                reason="fixture schedule ended",
            )
        )
    restored = sidecar("original", base + len(cells) + 1)
    rows.append(_row("original_restoration_attempt", restored, trial_seed_cleared=True))
    rows.append(
        _row("original_restored", restored, authenticated_active=True, trial_seed_cleared=True)
    )
    rows.append(
        _row(
            "run_end",
            restored,
            reason="complete; original restored",
            observed_current_identity=_java(restored),
            game_tick=anchor + total + 2,
            trial_seed_latched=False,
            cell_in_progress=False,
        )
    )
    ledger = directory / "matched.ndjson"
    _write(ledger, rows)
    return ledger, rows


def _first(rows: list[Json], kind: str) -> Json:
    return next(row for row in rows if row["kind"] == kind)


def _anchor(rows: list[Json]) -> Json:
    return next(row for row in rows if row.get("anchor_offset") == 0)


def test_completed_fixture_retains_hashes_and_does_not_claim_console_target(tmp_path: Path) -> None:
    ledger, rows = _build(tmp_path)
    result = summarize(tmp_path, ledger)
    assert result["all_declared_cells_complete"] is True
    assert result["fixed_console_60_target_complete"] is False
    assert result["schedule_scope"] == "custom_fixture"
    assert result["declared_cells"] == 2
    assert result["ledger_sha256"] == sha256(ledger.read_bytes()).hexdigest()
    assert result["plan_sha256"] == rows[0]["plan_sha256"]
    assert result["restoration_epoch_metadata_sha256"] is not None
    assert result["trials"][0]["identity"]["session_epoch"] > 2**53
    assert result["trials"][0]["applied_program_receipt_offers"] == {"CRUISE": 1, "ESCAPE_JUMP": 1}
    assert result["trials"][1]["control_seed"] == "18446744073709551615"
    assert result["groups"]["ENTITY/original"]["trials_with_applied_escape"] == 1
    assert "transport-delivery" in result["claim_boundary"]


def test_full_console_schedule_requires_all_sixty_cells_and_restoration(tmp_path: Path) -> None:
    ledger, rows = _build(tmp_path, console=True)
    result = summarize(tmp_path, ledger)
    assert result["fixed_console_60_target_complete"] is True
    assert len(result["groups"]) == 6
    assert all(group["planned"] == 10 for group in result["groups"].values())
    # Identical 60-cell completions at a translated test origin remain a fixture.
    rows[0]["world_origin"]["y"] = 20
    _write(ledger, rows)
    assert summarize(tmp_path, ledger)["fixed_console_60_target_complete"] is False


def test_invalid_incomplete_and_unattempted_remain_in_denominators(tmp_path: Path) -> None:
    ledger, rows = _build(tmp_path, count=3)
    first_end = next(i for i, row in enumerate(rows) if row["kind"] == "trial_end")
    # End the first cell before its anchor, start the next, then lose diagnostics.
    first_start = rows[1]
    invalid_end = {
        **rows[first_end],
        "anchor_game_tick": None,
        "outcome": "invalid_initial_state",
        "game_tick": first_start["reset_game_tick"],
    }
    second_start = rows[first_end + 1]
    rows = [rows[0], first_start, invalid_end, second_start]
    epoch = second_start["identity"]["sessionEpoch"]
    (tmp_path / "epochs" / str(epoch) / "epoch.json").unlink()
    _write(ledger, rows)
    result = summarize(tmp_path, ledger)
    assert [row["outcome"] for row in result["trials"]] == [
        "invalid_initial_state",
        "incomplete",
        "unattempted",
    ]
    assert sum(group["planned"] for group in result["groups"].values()) == 3
    assert result["trials"][1]["epoch_metadata_sha256"] is None
    assert result["trials"][1]["epoch_attribution"] == "unavailable: epoch.json missing"
    assert result["trials"][2]["identity"] is None
    assert result["all_declared_cells_complete"] is False


@pytest.mark.parametrize(
    "key", ["baseline_ticks", "approach_ticks", "recovery_ticks", "anchor_wait_ticks"]
)
@pytest.mark.parametrize("value", [0, -1, True, 1.0, "1", None])
def test_phase_counts_are_strict_positive_integers(tmp_path: Path, key: str, value: Any) -> None:
    ledger, rows = _build(tmp_path)
    rows[0]["plan"][key] = value
    _rehash(rows[0])
    _write(ledger, rows)
    with pytest.raises(DiagnosticError, match="must be an integer"):
        summarize(tmp_path, ledger)


@pytest.mark.parametrize("field", ["game_tick", "anchor_offset"])
@pytest.mark.parametrize("value", [True, 1.0, -1])
def test_timeline_integer_types(tmp_path: Path, field: str, value: Any) -> None:
    ledger, rows = _build(tmp_path)
    _anchor(rows)[field] = value
    _write(ledger, rows)
    with pytest.raises(DiagnosticError):
        summarize(tmp_path, ledger)


@pytest.mark.parametrize(
    "tamper",
    [
        "after_end",
        "gap",
        "duplicate",
        "lost_anchor",
        "wrong_end_anchor",
        "short_schedule",
        "missing_reset_tick",
        "end_before_schedule",
        "missing_initial",
        "initial_changed",
        "command_zero",
        "observation_zero",
        "neural_start",
        "neural_end",
        "command_unadmitted",
        "missing_anchor_receipt",
        "receipt_foreign",
        "receipt_duplicate",
        "receipt_mistimed",
    ],
)
def test_timeline_and_anchor_tampering_is_rejected(tmp_path: Path, tamper: str) -> None:
    ledger, rows = _build(tmp_path)
    anchor = _anchor(rows)
    first_tick = _first(rows, "tick")
    end = _first(rows, "trial_end")
    if tamper == "after_end":
        rows.insert(rows.index(end) + 1, copy.deepcopy(first_tick))
    elif tamper == "gap":
        rows.remove(rows[rows.index(anchor) + 1])
    elif tamper == "duplicate":
        rows.insert(rows.index(anchor), copy.deepcopy(anchor))
    elif tamper == "lost_anchor":
        rows[rows.index(anchor) + 1]["anchor_offset"] = None
    elif tamper == "wrong_end_anchor":
        end["anchor_game_tick"] += 1
    elif tamper == "short_schedule":
        rows.remove(rows[rows.index(end) - 1])
    elif tamper == "missing_reset_tick":
        rows.remove(first_tick)
    elif tamper == "end_before_schedule":
        end["game_tick"] -= 1
    elif tamper == "missing_initial":
        del anchor["initial_observation_zero"]
    elif tamper == "initial_changed":
        anchor["initial_observation_zero"]["xBlocks"] = 8
    elif tamper in {"command_zero", "observation_zero", "neural_start", "neural_end"}:
        key = {
            "command_zero": "commandSequence",
            "observation_zero": "observationSequence",
            "neural_start": "neuralStepStart",
            "neural_end": "neuralStepEnd",
        }[tamper]
        anchor["command"][key] += 1
    elif tamper == "command_unadmitted":
        anchor["command_admitted"] = False
    elif tamper == "missing_anchor_receipt":
        del anchor["program_receipt_offered"]
    elif tamper == "receipt_foreign":
        anchor["program_receipt_offered"]["identity"]["dynamics_hash"] = "f" * 64
    elif tamper == "receipt_duplicate":
        rows[rows.index(anchor) + 1]["program_receipt_offered"]["command_seq"] = 0
    else:
        anchor["program_receipt_offered"]["applied_tick"] += 1
    _write(ledger, rows)
    with pytest.raises(DiagnosticError):
        summarize(tmp_path, ledger)


@pytest.mark.parametrize(
    "tamper",
    [
        "missing_attempt",
        "duplicate_attempt",
        "reused_epoch",
        "catalog_changed",
        "missing_sidecar",
        "sidecar_control_changed",
        "sidecar_profile_changed",
        "run_end_identity",
        "run_end_observed_identity",
        "run_end_reason",
        "run_end_seed",
        "still_in_progress",
    ],
)
def test_original_restoration_must_bind_later_epoch_and_run_end(
    tmp_path: Path, tamper: str
) -> None:
    ledger, rows = _build(tmp_path)
    attempt, restored, end = rows[-3:]
    if tamper == "missing_attempt":
        rows.remove(attempt)
    elif tamper == "duplicate_attempt":
        rows.insert(-2, copy.deepcopy(attempt))
    elif tamper == "reused_epoch":
        for row in (attempt, restored, end):
            row["identity"]["sessionEpoch"] = rows[1]["identity"]["sessionEpoch"]
    elif tamper == "catalog_changed":
        restored["identity"]["dynamicsHash"] = "f" * 64
    elif tamper.startswith("sidecar") or tamper == "missing_sidecar":
        path = tmp_path / "epochs" / str(restored["identity"]["sessionEpoch"]) / "epoch.json"
        if tamper == "missing_sidecar":
            path.unlink()
        else:
            metadata = json.loads(path.read_text())
            metadata[
                "control_kind" if tamper == "sidecar_control_changed" else "controller_profile"
            ] = "wrong"
            path.write_text(json.dumps(metadata))
    elif tamper == "run_end_identity":
        end["identity"]["sessionEpoch"] += 1
    elif tamper == "run_end_observed_identity":
        end["observed_current_identity"]["dynamicsHash"] = "f" * 64
    elif tamper == "run_end_reason":
        end["reason"] = "original restoration timed out"
    elif tamper == "run_end_seed":
        end["trial_seed_latched"] = True
    else:
        end["cell_in_progress"] = True
    _write(ledger, rows)
    with pytest.raises(DiagnosticError):
        summarize(tmp_path, ledger)


def test_complete_trials_require_sidecars_but_missing_run_end_is_incomplete(tmp_path: Path) -> None:
    ledger, rows = _build(tmp_path)
    _write(ledger, rows[:-1])
    assert summarize(tmp_path, ledger)["all_declared_cells_complete"] is False
    sidecar = tmp_path / "epochs" / str(rows[1]["identity"]["sessionEpoch"]) / "epoch.json"
    sidecar.unlink()
    with pytest.raises(DiagnosticError, match="completed trial lacks sidecar"):
        summarize(tmp_path, ledger)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_json_is_refused_in_unused_nested_fields(tmp_path: Path, token: str) -> None:
    ledger, rows = _build(tmp_path)
    raw = json.dumps(rows[0])[:-1] + ', "unused": {"nested": [' + token + "]}}\n"
    ledger.write_text(raw)
    with pytest.raises(DiagnosticError, match="non-finite"):
        summarize(tmp_path, ledger)


@pytest.mark.parametrize(
    "tamper",
    [
        "truncated",
        "duplicate_key",
        "bad_hash",
        "plan_differs",
        "after_run_end",
        "bad_utf8",
        "oversize",
        "symlink",
        "reused_trial_epoch",
        "out_of_order",
    ],
)
def test_ledger_boundary_tampering(tmp_path: Path, tamper: str) -> None:
    ledger, rows = _build(tmp_path)
    if tamper == "truncated":
        ledger.write_bytes(ledger.read_bytes().rstrip(b"\n"))
    elif tamper == "duplicate_key":
        ledger.write_text(json.dumps(rows[0])[:-1] + ', "kind": "plan"}\n')
    elif tamper == "bad_utf8":
        ledger.write_bytes(b"\xff\n")
    elif tamper == "oversize":
        ledger.write_bytes(b" " * 131_073 + b"\n")
    elif tamper == "symlink":
        link = tmp_path / "link.ndjson"
        link.symlink_to(ledger)
        ledger = link
    else:
        if tamper == "bad_hash":
            rows[0]["plan_sha256"] = "a" * 64
        elif tamper == "plan_differs":
            rows[0]["plan"]["baseline_ticks"] += 1
        elif tamper == "after_run_end":
            rows.append(copy.deepcopy(rows[-1]))
        elif tamper == "reused_trial_epoch":
            starts = [row for row in rows if row["kind"] == "trial_start"]
            starts[1]["identity"]["sessionEpoch"] = starts[0]["identity"]["sessionEpoch"]
        else:
            rows[1], rows[8] = rows[8], rows[1]
        _write(ledger, rows)
    with pytest.raises(DiagnosticError):
        summarize(tmp_path, ledger)
