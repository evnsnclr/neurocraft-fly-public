#!/usr/bin/env python3
"""Reduce authoritative trial diagnostics, retaining invalid and unattempted cells.

Local receipt offers describe body application, not transport delivery. This is
structural attribution to a catalog and epoch sidecars, not authenticated replay.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurocraft_protocol import ProgramReceipt, ProtocolError, parse_message

from scripts.summarize_comparison_session import ENTRY_FIELDS, _bounded, _catalog, _identity
from scripts.summarize_interactive_session import DiagnosticError

Json = dict[str, Any]
JAVA_IDENTITY = {
    "runId": "run_id",
    "entityId": "entity_id",
    "sessionEpoch": "session_epoch",
    "graphHash": "graph_hash",
    "adapterHash": "adapter_hash",
    "dynamicsHash": "dynamics_hash",
    "decoderHash": "decoder_hash",
    "observationSchemaHash": "observation_schema_hash",
}
PHASES = ("baseline_ticks", "approach_ticks", "recovery_ticks")
COMPLETE = {"complete", "complete_stimulus_ray_miss"}
OUTCOMES = COMPLETE | {"invalid_initial_state", "missing_command_zero", "invalid_anchor"}


def _integer(value: object, name: str, minimum: int = 0, maximum: int = 2**63 - 1) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise DiagnosticError(f"{name} must be an integer in {minimum}..{maximum}")
    return value


def _java_identity(value: object) -> Json:
    if not isinstance(value, dict) or set(value) != set(JAVA_IDENTITY):
        raise DiagnosticError("unexpected Java identity fields")
    return _identity({JAVA_IDENTITY[key]: item for key, item in value.items()})


def _finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise DiagnosticError("non-finite JSON number")
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)


def _unique(pairs: list[tuple[str, Any]]) -> Json:
    value: Json = {}
    for key, item in pairs:
        if key in value:
            raise DiagnosticError("duplicate ledger key")
        value[key] = item
    return value


def _json(data: str | bytes) -> Any:
    try:
        value = json.loads(data, object_pairs_hook=_unique)
        _finite(value)  # Includes overflow such as 1e999, not only NaN/Infinity tokens.
        return value
    except (ValueError, RecursionError) as exc:
        raise DiagnosticError(f"invalid trial JSON: {exc}") from exc


def _load(path: Path) -> tuple[list[Json], str]:
    if path.is_symlink():
        raise DiagnosticError("trial ledger must not be a symlink")
    rows: list[Json] = []
    digest = sha256()
    try:
        with path.open("rb") as stream:
            while line := stream.readline(131_073):
                if len(line) > 131_072 or not line.endswith(b"\n") or len(rows) >= 25_000:
                    raise DiagnosticError("truncated or oversized trial ledger")
                digest.update(line)
                row = _json(line)
                if (
                    not isinstance(row, dict)
                    or row.get("schema") != "neurocraft.matched_trial_ledger.v1"
                    or row.get("diagnostic_only") is not True
                ):
                    raise DiagnosticError("unsupported trial ledger schema or scope")
                rows.append(row)
    except OSError as exc:
        raise DiagnosticError(f"trial ledger unavailable: {exc}") from exc
    return rows, digest.hexdigest()


def _sidecar(directory: Path, session: Json, lane: Json, identity: Json) -> str | None:
    path = directory / "epochs" / str(identity["session_epoch"]) / "epoch.json"
    if path.is_symlink() or path.parent.is_symlink():
        raise DiagnosticError("epoch metadata must not be a symlink")
    if not path.exists():
        return None  # A refused or interrupted startup may never publish an epoch sidecar.
    metadata, digest = _bounded(path)
    _finite(metadata)
    expected = {**lane, "identity": identity}
    if (
        metadata.get("schema") != "neurocraft.lane_epoch.v1"
        or metadata.get("diagnostic_only") is not True
        or metadata.get("controller_profile") != session.get("controller_profile")
        or any(metadata.get(key) != expected[key] for key in ENTRY_FIELDS)
        or _identity(metadata.get("identity")) != identity
    ):
        raise DiagnosticError("trial epoch metadata differs from catalog or session")
    return digest


def _fixed_cells() -> list[Json]:
    return [
        {
            "index": pair * 6 + stimulus_index * 3 + lane_index,
            "pair": pair,
            "bodySeed": pair * 7919,
            "stimulus": stimulus,
            "lane": lane,
        }
        for pair in range(10)
        for stimulus_index, stimulus in enumerate(("ENTITY", "BLOCK"))
        for lane_index, lane in enumerate(("original", "shuffled_weight", "no_input"))
    ]


def _plan(header: Json, lanes: dict[str, Json]) -> tuple[Json, list[Json], bool]:
    serialized = header.get("plan_json")
    if not isinstance(serialized, str) or sha256(serialized.encode()).hexdigest() != header.get(
        "plan_sha256"
    ):
        raise DiagnosticError("trial plan byte digest differs")
    plan = _json(serialized)
    if (
        not isinstance(plan, dict)
        or plan != header.get("plan")
        or plan.get("schema") != "neurocraft.matched_trial_plan.v1"
        or plan.get("diagnostic_only") is not True
    ):
        raise DiagnosticError("trial plan content differs")
    for key in (*PHASES, "anchor_wait_ticks"):
        _integer(plan.get(key), key, 1, 25_000)
    cells = header.get("executed_cells")
    if not isinstance(cells, list) or len(cells) != _integer(
        header.get("cell_count"), "cell_count", 1, 60
    ):
        raise DiagnosticError("invalid executed cell count")
    seen: set[int] = set()
    for cell in cells:
        if not isinstance(cell, dict) or set(cell) != {
            "index",
            "pair",
            "bodySeed",
            "stimulus",
            "lane",
        }:
            raise DiagnosticError("invalid trial cell fields")
        index = _integer(cell["index"], "cell index")
        _integer(cell["pair"], "cell pair")
        _integer(cell["bodySeed"], "body seed")
        if (
            index in seen
            or not isinstance(cell["lane"], str)
            or not isinstance(cell["stimulus"], str)
            or cell["lane"] not in lanes
            or cell["stimulus"] not in {"ENTITY", "BLOCK"}
        ):
            raise DiagnosticError("invalid trial cell")
        seen.add(index)
    origin = header.get("world_origin")
    if (
        not isinstance(origin, dict)
        or set(origin) != {"x", "y", "z"}
        or any(type(value) not in (int, float) for value in origin.values())
    ):
        raise DiagnosticError("invalid world origin")
    # The translated/subset GameTest constructor carries the full plan document too.
    # Only the actual cells, origin and declared fixed schedule identify the console target.
    fixed = _fixed_cells()
    fixed_document_cells = [
        {
            **{key: value for key, value in cell.items() if key != "bodySeed"},
            "body_seed": cell["bodySeed"],
        }
        for cell in fixed
    ]
    console = (
        cells == fixed
        and plan.get("trials") == fixed_document_cells
        and origin == {"x": 0, "y": 0, "z": 0}
        and [plan[key] for key in PHASES] == [20, 60, 80]
        and plan["anchor_wait_ticks"] == 200
        and plan.get("anchor") == "applied program command 0, observation 0, neural steps 0..5"
        and plan.get("front_planes")
        == "z=6,5,4,3,2,1; ten ticks each; x centre=.5; block bottom y=67"
        and plan.get("initial_pose")
        == (
            "feet=(.5,67,.5); yaw=0; pitch=0; velocity=0; dry and airborne; "
            "full sturdy floor exactly 2 blocks below feet"
        )
    )
    return plan, cells, console


def _timeline(result: Json, ticks: list[Json], plan: Json) -> None:
    expected_ticks = sum(plan[key] for key in PHASES)
    complete = result["outcome"] in COMPLETE
    anchor: int | None = None
    previous: int | None = None
    anchored_count = 0
    applied: Counter[str] = Counter()
    commands: set[int] = set()
    first_escape = first_stimulus = None
    for row in ticks:
        tick = _integer(row.get("game_tick"), "game_tick")
        if previous is None:
            if tick != result["reset_game_tick"]:
                raise DiagnosticError("trial timeline omits its first reset tick")
        elif tick != previous + 1:
            raise DiagnosticError("trial tick ledger contains gaps or duplicates")
        previous = tick
        before = row.get("world_before")
        if not isinstance(before, dict) or (
            _integer(before.get("gameTick"), "world_before.gameTick") != tick
            or before.get("entityId") != result["identity"]["entity_id"]
        ):
            raise DiagnosticError("tick world sample differs from trial")
        offset = row.get("anchor_offset")
        if offset is not None:
            _integer(offset, "anchor_offset")
            if anchor is None:
                anchor = tick
                if tick - result["reset_game_tick"] >= plan["anchor_wait_ticks"]:
                    raise DiagnosticError("anchor is outside the declared wait budget")
                command = row.get("command")
                initial = row.get("initial_observation_zero")
                if (
                    not isinstance(command, dict)
                    or not isinstance(initial, dict)
                    or (
                        _java_identity(command.get("identity")) != result["identity"]
                        or any(
                            _integer(command.get(key), key) != value
                            for key, value in (
                                ("commandSequence", 0),
                                ("observationSequence", 0),
                                ("neuralStepStart", 0),
                                ("neuralStepEnd", 5),
                            )
                        )
                        or row.get("command_admitted") is not True
                        or not any(
                            prior.get("world_before") == initial
                            for prior in ticks
                            if tick - 2 <= prior["game_tick"] <= tick
                        )
                        or command.get("observationTick") != initial.get("gameTick")
                    )
                ):
                    raise DiagnosticError("anchor lacks its bound actual command/observation zero")
            if offset != tick - anchor or offset != anchored_count or offset >= expected_ticks:
                raise DiagnosticError("trial anchor offsets are inconsistent or out of schedule")
            anchored_count += 1
        elif anchor is not None:
            raise DiagnosticError("trial lost its anchor inside the timeline")
        hit = row.get("stimulus_forward_ray_matches_geometry")
        if type(hit) is not bool:
            raise DiagnosticError("stimulus ray attribution must be boolean")
        if hit and first_stimulus is None:
            first_stimulus = tick
        offered = row.get("program_receipt_offered")
        receipt: ProgramReceipt | None = None
        if offered is not None:
            try:
                parsed = parse_message(json.dumps(offered, allow_nan=False))
            except ProtocolError as exc:
                raise DiagnosticError(f"invalid offered receipt: {exc}") from exc
            if (
                not isinstance(parsed, ProgramReceipt)
                or parsed.identity.to_dict() != result["identity"]
            ):
                raise DiagnosticError("offered receipt identity differs from trial")
            receipt = parsed
            if receipt.command_seq in commands or receipt.applied_tick != tick:
                raise DiagnosticError("duplicate or mistimed receipt offer")
            commands.add(receipt.command_seq)
            if offered["validation"] == "PROGRAM_APPLIED":
                applied[offered["executed_program"]] += 1
                if offered["executed_program"] == "ESCAPE_JUMP" and first_escape is None:
                    first_escape = tick
        if offset == 0 and (
            receipt is None
            or (
                receipt.command_seq != 0
                or receipt.observation_seq != 0
                or receipt.validation.value != "PROGRAM_APPLIED"
                or receipt.target_tick != row["command"].get("targetTick")
                or not receipt.target_tick
                <= tick
                <= _integer(row["command"].get("validThroughTick"), "validThroughTick")
                or receipt.requested_program.value != row["command"].get("program")
            )
        ):
            raise DiagnosticError("anchor lacks an applied command-zero receipt offer")
    if "end_game_tick" in result:
        if anchor != result["anchor_game_tick"]:
            raise DiagnosticError("trial end anchor differs from its timeline")
        if previous is not None and result["end_game_tick"] < previous:
            raise DiagnosticError("trial end precedes its last tick")
    if complete and (
        result["epoch_metadata_sha256"] is None
        or anchor is None
        or anchored_count != expected_ticks
        or result["end_game_tick"] != anchor + expected_ticks
    ):
        raise DiagnosticError("completed trial lacks sidecar or full anchored timeline")
    if complete and (result["outcome"] == "complete") != (first_stimulus is not None):
        raise DiagnosticError("completed outcome disagrees with recorded stimulus ray attribution")
    result.update(
        recorded_ticks=len(ticks),
        complete_schedule=complete,
        applied_program_receipt_offers=dict(applied),
        first_applied_escape_tick=first_escape,
        first_stimulus_forward_ray_tick=first_stimulus,
    )


def summarize(directory: Path, ledger: Path) -> Json:
    session, session_hash = _bounded(directory / "session.json")
    _finite(session)
    if session.get("comparison_lanes") is not True:
        raise DiagnosticError("matched ledger requires a comparison session")
    lanes, catalog_hash = _catalog(directory, session)
    rows, ledger_hash = _load(ledger)
    if not rows or rows[0].get("kind") != "plan":
        raise DiagnosticError("trial plan header is missing")
    header = rows[0]
    plan, cells, console = _plan(header, lanes)
    results: dict[int, Json] = {
        cell["index"]: {
            "trial": cell,
            "started": False,
            "outcome": "unattempted",
            "identity": None,
            "epoch_metadata_sha256": None,
            "epoch_attribution": "unavailable: trial not started",
        }
        for cell in cells
    }
    tick_rows: dict[int, list[Json]] = {index: [] for index in results}
    restored = attempt = end = None
    restoration_hash = None
    last_epoch = session["session_epoch"]
    next_cell = 0
    active: Json | None = None
    last_game_tick = 0
    for row in rows[1:]:
        if end is not None:
            raise DiagnosticError("ledger rows follow run_end")
        kind = row.get("kind")
        if kind == "run_end":
            end = row
            if _integer(row.get("game_tick"), "run_end game_tick") < last_game_tick:
                raise DiagnosticError("run_end precedes the recorded timeline")
            if not isinstance(row.get("reason"), str) or not row["reason"]:
                raise DiagnosticError("run_end reason is missing")
            if (restored is not None or row["reason"] == "complete; original restored") and (
                restored is None
                or _java_identity(row.get("identity")) != restored
                or _java_identity(row.get("observed_current_identity")) != restored
                or row["reason"] != "complete; original restored"
                or row.get("trial_seed_latched") is not False
                or row.get("cell_in_progress") is not False
                or row.get("trial") is not None
            ):
                raise DiagnosticError("run_end does not confirm the restored original identity")
            continue
        if kind in {"original_restoration_attempt", "original_restored"}:
            identity = _java_identity(row.get("identity"))
            if row.get("trial") is not None or identity != {
                **lanes["original"]["identity"],
                "session_epoch": identity["session_epoch"],
            }:
                raise DiagnosticError("restoration identity differs from original catalog")
            if kind == "original_restoration_attempt":
                if attempt is not None or next_cell != len(cells) or active is not None:
                    raise DiagnosticError("restoration precedes trial ends or repeats")
                if row.get("trial_seed_cleared") is not True:
                    raise DiagnosticError("restoration attempt retained the trial seed")
                if identity["session_epoch"] <= last_epoch:
                    raise DiagnosticError("restoration requires a later unique epoch")
                last_epoch = identity["session_epoch"]
                attempt = identity
            elif (
                attempt != identity
                or restored is not None
                or row.get("authenticated_active") is not True
                or row.get("trial_seed_cleared") is not True
            ):
                raise DiagnosticError("ambiguous original restoration")
            else:
                restoration_hash = _sidecar(directory, session, lanes["original"], identity)
                if restoration_hash is None:
                    raise DiagnosticError("confirmed restoration requires an epoch sidecar")
                restored = identity
            continue
        if attempt is not None:
            raise DiagnosticError("trial rows follow original restoration")
        cell = row.get("trial")
        if (
            not isinstance(cell, dict)
            or type(cell.get("index")) is not int
            or cell["index"] not in results
        ):
            raise DiagnosticError("ledger row names an unknown cell")
        result = results[cell["index"]]
        if cell != result["trial"]:
            raise DiagnosticError("trial definition changed within the ledger")
        identity = _java_identity(row.get("identity"))
        if identity != {
            **lanes[cell["lane"]]["identity"],
            "session_epoch": identity["session_epoch"],
        }:
            raise DiagnosticError("trial identity differs from its catalog lane")
        if kind == "trial_start":
            if active is not None or result["started"] or cells[next_cell] != cell:
                raise DiagnosticError("duplicate or out-of-order trial start")
            if identity["session_epoch"] <= last_epoch:
                raise DiagnosticError("trial requires a later unique epoch")
            last_epoch = identity["session_epoch"]
            next_cell += 1
            active = result
            reset_tick = _integer(row.get("reset_game_tick"), "reset_game_tick")
            if reset_tick < last_game_tick:
                raise DiagnosticError("trial reset precedes the recorded timeline")
            last_game_tick = reset_tick
            metadata_hash = _sidecar(directory, session, lanes[cell["lane"]], identity)
            result.update(
                started=True,
                outcome="incomplete",
                identity=identity,
                reset_game_tick=reset_tick,
                epoch_metadata_sha256=metadata_hash,
                epoch_attribution="catalog_and_sidecar"
                if metadata_hash
                else "unavailable: epoch.json missing",
                control_seed=lanes[cell["lane"]]["seed"],
                control_sha256=lanes[cell["lane"]]["control_sha256"],
            )
        elif active is not result or identity != result["identity"]:
            raise DiagnosticError(
                "trial row precedes its start, follows its end, or changes identity"
            )
        elif kind == "tick":
            last_game_tick = _integer(row.get("game_tick"), "game_tick")
            tick_rows[cell["index"]].append(row)
        elif kind == "trial_end":
            if (
                not isinstance(row.get("outcome"), str)
                or row.get("outcome") not in OUTCOMES
                or not isinstance(row.get("reason"), str)
                or not row["reason"]
            ):
                raise DiagnosticError("invalid trial outcome or reason")
            anchor = row.get("anchor_game_tick")
            if anchor is not None:
                _integer(anchor, "anchor_game_tick")
            end_tick = _integer(row.get("game_tick"), "trial_end game_tick")
            if end_tick < last_game_tick:
                raise DiagnosticError("trial end precedes the recorded timeline")
            last_game_tick = end_tick
            result.update(
                outcome=row["outcome"],
                reason=row["reason"],
                anchor_game_tick=anchor,
                end_game_tick=end_tick,
            )
            active = None
        else:
            raise DiagnosticError("unknown trial row kind")
    for index, result in results.items():
        _timeline(result, tick_rows[index], plan)
    groups: dict[str, Json] = {}
    for result in results.values():
        key = result["trial"]["stimulus"] + "/" + result["trial"]["lane"]
        group = groups.setdefault(
            key,
            {
                "planned": 0,
                "started": 0,
                "complete_schedule": 0,
                "trials_with_applied_escape": 0,
                "outcomes": Counter(),
            },
        )
        group["planned"] += 1
        group["started"] += int(result["started"])
        group["complete_schedule"] += int(result["complete_schedule"])
        group["trials_with_applied_escape"] += int(result["first_applied_escape_tick"] is not None)
        group["outcomes"][result["outcome"]] += 1
    complete = (
        end is not None
        and restored is not None
        and all(result["complete_schedule"] for result in results.values())
    )
    return {
        "schema": "neurocraft.matched_trial_summary.v1",
        "diagnostic_only": True,
        "claim_boundary": (
            "Catalog/sidecar and local body receipt-offer attribution, not authenticated replay "
            "or transport-delivery proof. Invalid and unattempted cells remain in denominators; "
            "escape counts do not alone establish a stimulus effect. Schedule completion does "
            "not establish equal realized sensory histories, normal-speed timing, "
            "or a causal effect."
        ),
        "run_id": session["run_id"],
        "session_sha256": session_hash,
        "catalog_sha256": catalog_hash,
        "ledger_sha256": ledger_hash,
        "plan_sha256": header["plan_sha256"],
        "world_origin": header["world_origin"],
        "declared_phase_ticks": {key: plan[key] for key in PHASES},
        "declared_cells": len(cells),
        "schedule_scope": "fixed_console_60" if console else "custom_fixture",
        "fixed_console_60_target_complete": console and complete,
        "run_end": end,
        "restoration_attempt_identity": attempt,
        "restored_original_identity": restored,
        "restoration_epoch_metadata_sha256": restoration_hash,
        "all_declared_cells_complete": complete,
        "groups": groups,
        "trials": list(results.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_directory", type=Path)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.session_directory, args.ledger)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
