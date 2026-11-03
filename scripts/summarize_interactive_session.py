#!/usr/bin/env python3
"""Summarize recorded interactive diagnostics, without replay or live-performance claims.

Usage: python scripts/summarize_interactive_session.py artifacts/live-interactive/<run>
Missing sources remain unavailable. A torn final NDJSON record is reported and skipped;
malformed complete/interior records fail rather than silently changing the counts.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

Json = dict[str, Any]
FILES = ("session.json", "program-decisions.ndjson", "action-latency.ndjson", "server-gradle.log")
FAULT_STATES = frozenset(
    {
        "MISSING_CONFIGURATION",
        "DISCONNECTED",
        "PROTOCOL_ERROR",
        "IDENTITY_MISMATCH",
        "STALE_COMMAND",
        "FUTURE_COMMAND",
        "DEADLINE_MISSED",
        "SEQUENCE_ERROR",
        "OVERSIZE_MESSAGE",
        "QUEUE_OVERFLOW",
        "INTERNAL_ERROR",
    }
)
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
POSITION = re.compile(rf"\bposition=\(({NUMBER}),\s*({NUMBER}),\s*({NUMBER})\)")


class DiagnosticError(ValueError):
    """The recorded input cannot support an unambiguous summary."""


class MalformedJsonError(DiagnosticError):
    """A record is not complete syntactically valid JSON."""


def _object(raw: bytes, label: str) -> Json:
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise MalformedJsonError(f"{label}: malformed JSON ({exc})") from exc
    if not isinstance(value, dict):
        raise DiagnosticError(f"{label}: expected a JSON object")
    return value


def _records(path: Path, warnings: list[str]) -> list[Json] | None:
    if not path.exists():
        return None
    lines = path.read_bytes().splitlines(keepends=True)
    result = []
    for index, line in enumerate(lines, 1):
        try:
            record = _object(line, f"{path.name}:{index}")
        except MalformedJsonError:
            if index == len(lines) and not line.endswith((b"\n", b"\r")):
                warnings.append(f"{path.name}:{index}: skipped incomplete final JSON record")
                break
            raise
        result.append(record)
    return result


def _integer(record: Json, field: str, label: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise DiagnosticError(f"{label}: {field} must be a nonnegative integer")
    return value


def _range(values: list[int]) -> Json | None:
    return {"min": min(values), "max": max(values)} if values else None


def _check_identity(records: list[Json], session: Json, filename: str) -> None:
    for field in ("run_id", "controller_profile"):
        observed = [record.get(field) for record in records]
        if any(not isinstance(value, str) or not value for value in observed):
            raise DiagnosticError(f"{filename}: missing or invalid {field}")
        values = set(observed)
        expected = session.get(field)
        if len(values) > 1 or (expected is not None and values and values != {expected}):
            raise DiagnosticError(f"{filename}: conflicting {field}; refusing to combine runs")


def _programs(records: list[Json] | None, warnings: list[str]) -> Json | None:
    if records is None:
        return None
    issued: dict[int, Json] = {}
    receipts: dict[int, Json] = {}
    for number, record in enumerate(records, 1):
        label = f"program-decisions.ndjson:{number}"
        if record.get("schema") != "neurocraft.program_decision.v1":
            raise DiagnosticError(f"{label}: unsupported schema")
        kind = record.get("kind")
        if kind not in {"issued", "receipt"}:
            raise DiagnosticError(f"{label}: unknown record kind")
        sequence = _integer(record, "command_seq", label)
        for field in ("observation_seq", "target_tick"):
            _integer(record, field, label)
        destination = issued if kind == "issued" else receipts
        if sequence in destination:
            raise DiagnosticError(f"{label}: duplicate {kind} command_seq {sequence}")
        program_field = "program" if kind == "issued" else "requested_program"
        if not isinstance(record.get(program_field), str):
            raise DiagnosticError(f"{label}: missing {program_field}")
        if kind == "issued":
            for field in ("observation_tick", "neural_step_start", "neural_step_end"):
                _integer(record, field, label)
        else:
            _integer(record, "applied_tick", label)
            if record.get("validation") not in {"PROGRAM_APPLIED", "PROGRAM_REFUSED", "REJECTED"}:
                raise DiagnosticError(f"{label}: unknown receipt validation")
            executed = record.get("executed_program")
            if (record["validation"] == "REJECTED" and executed is not None) or (
                record["validation"] != "REJECTED" and not isinstance(executed, str)
            ):
                raise DiagnosticError(f"{label}: executed_program disagrees with validation")
        destination[sequence] = record

    matched = set()
    mismatched = []
    for sequence in issued.keys() & receipts.keys():
        command, receipt = issued[sequence], receipts[sequence]
        if (
            all(command[key] == receipt[key] for key in ("observation_seq", "target_tick"))
            and command["program"] == receipt["requested_program"]
        ):
            matched.add(sequence)
        else:
            mismatched.append(sequence)
    if mismatched:
        warnings.append("program-decisions.ndjson: receipt binding mismatches remain unmatched")

    escape_onsets, escape_holds, escape_unknown = [], [], []
    for sequence, record in issued.items():
        if record["program"] != "ESCAPE_JUMP":
            continue
        before, after = record.get("policy_state_before"), record.get("policy_state_after")
        if (
            record.get("policy_state_bound") is not True
            or not isinstance(before, dict)
            or not isinstance(after, dict)
            or after.get("program") != "ESCAPE_JUMP"
        ):
            escape_unknown.append(sequence)
        elif before.get("program") == "CRUISE":
            escape_onsets.append(sequence)
        elif before.get("program") == "ESCAPE_JUMP":
            escape_holds.append(sequence)
        else:
            escape_unknown.append(sequence)
    if escape_unknown:
        warnings.append("program-decisions.ndjson: escape frames lack bound onset/hold evidence")

    validations = Counter(record["validation"] for record in receipts.values())
    executed = {
        validation: dict(
            sorted(
                Counter(
                    record["executed_program"]
                    for record in receipts.values()
                    if record["validation"] == validation
                    and record.get("executed_program") is not None
                ).items()
            )
        )
        for validation in sorted(validations)
    }
    return {
        "issued_frames": len(issued),
        "issued_by_program": dict(sorted(Counter(r["program"] for r in issued.values()).items())),
        "receipt_frames": len(receipts),
        "receipts_by_validation": dict(sorted(validations.items())),
        "reported_executed_programs_by_validation": executed,
        "applied_receipts": validations["PROGRAM_APPLIED"],
        "refused_receipts": validations["PROGRAM_REFUSED"],
        "rejected_receipts": validations["REJECTED"],
        "matched_pairs": len(matched),
        "unmatched_issued_count": len(issued.keys() - matched),
        "unmatched_receipt_count": len(receipts.keys() - matched),
        "issued_without_matching_receipt": sorted(issued.keys() - matched),
        "receipts_without_matching_issue": sorted(receipts.keys() - matched),
        "binding_mismatch_command_sequences": sorted(mismatched),
        "escape_jump": {
            "explicit_policy_onsets": len(escape_onsets),
            "onset_command_sequences": sorted(escape_onsets),
            "policy_hold_frames": len(escape_holds),
            "unclassified_frames": len(escape_unknown),
            "basis": "bound recorded policy state; an issued onset does not prove execution",
        },
        "issued_command_sequence_range": _range(list(issued)),
        "receipt_command_sequence_range": _range(list(receipts)),
        "issued_neural_step_range": _range(
            [
                record[field]
                for record in issued.values()
                for field in ("neural_step_start", "neural_step_end")
            ]
        ),
    }


def _latency(records: list[Json] | None) -> Json | None:
    if records is None:
        return None
    values, sequences = [], []
    for number, record in enumerate(records, 1):
        label = f"action-latency.ndjson:{number}"
        if record.get("schema") != "neurocraft.action_latency.v1":
            raise DiagnosticError(f"{label}: unsupported schema")
        values.append(_integer(record, "compute_ns", label) / 1_000_000)
        sequences.append(_integer(record, "command_seq", label))
    if len(set(sequences)) != len(sequences):
        raise DiagnosticError("action-latency.ndjson: duplicate command sequence")
    values.sort()

    def percentile(fraction: float) -> float | None:
        if not values:
            return None
        position = (len(values) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        return values[low] + (values[high] - values[low]) * (position - low)

    return {
        "scope": "service compute only; all recorded frames, including paused neutral frames",
        "unit": "ms",
        "sample_count": len(values),
        "percentile_method": "linear interpolation at (n - 1) * quantile",
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": max(values) if values else None,
        "command_sequence_range": _range(sequences),
    }


def _server(path: Path, warnings: list[str]) -> tuple[Json | None, list[int]]:
    if not path.exists():
        return None, []
    statuses: list[str] = []
    positions: list[list[float]] = []
    commands: list[int] = []
    steps: list[int] = []
    retirements: list[int] = []
    ticks: list[int] = []
    faults: dict[str, Json] = {}
    transitions = 0
    for number, line in enumerate(path.read_text().splitlines(), 1):
        # Only explicitly observed transition ticks, never a status's stale since_tick.
        transition = re.search(r"BODY PROGRAM \w+ -> (\w+) tick=(\d+)", line)
        if transition:
            ticks.append(int(transition[2]))
            transitions += transition[1] == "ESCAPE_JUMP"
        state = re.search(r"NeuroCraft bridge=(\w+)", line)
        if state is None:
            continue
        statuses.append(state[1])
        if state[1] in FAULT_STATES:
            event = faults.setdefault(
                state[1],
                {"state": state[1], "first_line": number, "last_line": number, "status_samples": 0},
            )
            event["last_line"] = number
            event["status_samples"] += 1
        elif state[1] not in {"ACTIVE", "STARTING"}:
            warnings.append(f"server-gradle.log:{number}: unknown bridge state {state[1]}")
        position = POSITION.search(line)
        if position:
            point = [float(position[index]) for index in (1, 2, 3)]
            if not all(math.isfinite(value) for value in point):
                raise DiagnosticError(f"server-gradle.log:{number}: non-finite position")
            positions.append(point)
        command = re.search(r"\bcommand_seq=(\d+)", line)
        if command:
            commands.append(int(command[1]))
        neural = re.search(r"\bneural_steps=(\d+)\.\.(\d+)", line)
        if neural:
            steps.extend((int(neural[1]), int(neural[2])))
        retirement = re.search(r"\blateness_retired=(\d+)", line)
        if retirement:
            retirements.append(int(retirement[1]))
    if not statuses:
        warnings.append("server-gradle.log: no bridge status samples available")
    return {
        "status_sample_count": len(statuses),
        "final_observed_bridge_state": statuses[-1] if statuses else None,
        "terminal_bridge_faults_observed": list(faults.values()) if statuses else None,
        "position_samples": {
            "sample_count": len(positions),
            "y_min": min(point[1] for point in positions) if positions else None,
            "y_max": max(point[1] for point in positions) if positions else None,
            "final_xyz": positions[-1] if positions else None,
            "scope": "logged status samples only, not continuous trajectory extrema",
        },
        "command_sequence_range": _range(commands),
        "neural_step_range": _range(steps),
        "lateness_retired_max": max(retirements) if retirements else None,
        "escape_jump_transition_lines": transitions,
    }, ticks


def summarize_session(directory: Path) -> Json:
    """Return a deterministic summary of available records, or reject malformed input."""
    if not directory.is_dir():
        raise DiagnosticError(f"session directory does not exist: {directory}")
    warnings = [
        f"{name}: unavailable (missing file)" for name in FILES if not (directory / name).exists()
    ]
    session_path = directory / "session.json"
    session = _object(session_path.read_bytes(), "session.json") if session_path.exists() else {}
    decisions = _records(directory / "program-decisions.ndjson", warnings)
    latency = _records(directory / "action-latency.ndjson", warnings)
    # Also compare both streams with each other when session.json is unavailable.
    _check_identity((decisions or []) + (latency or []), session, "diagnostic streams")
    programs = _programs(decisions, warnings)
    compute = _latency(latency)
    server, ticks = _server(directory / "server-gradle.log", warnings)
    for record in decisions or []:
        ticks.append(
            record["observation_tick"] if record["kind"] == "issued" else record["applied_tick"]
        )
    tick_range = _range(ticks)
    return {
        "schema": "neurocraft.interactive_diagnostic_summary.v1",
        "diagnostic_only": True,
        "claim_boundary": (
            "recorded diagnostics; no replay, live validation, or biological inference"
        ),
        "files": {
            name: "available" if (directory / name).exists() else "unavailable" for name in FILES
        },
        "session": {
            "run_id": session.get("run_id"),
            "controller_profile": session.get("controller_profile"),
            "identity": session.get("identity"),
            "recorded_final_state": session.get("state"),
            "recorded_exit_reason": session.get("exit_reason"),
            "requested_game_ticks_per_wall_second": session.get(
                "requested_game_ticks_per_wall_second"
            ),
            "playback_speed": session.get("playback_speed"),
        },
        "programs": programs,
        "compute_latency": compute,
        "server_status": server,
        "observed_simulated_time": {
            "tick_range": tick_range,
            "tick_span": max(ticks) - min(ticks) if ticks else None,
            "seconds_at_20_ticks_per_second": (max(ticks) - min(ticks)) / 20 if ticks else None,
            "basis": "observed observation, receipt application, and body transition ticks; "
            "excludes unobserved session edges; not wall-clock duration",
        },
        "warnings": warnings,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_directory", type=Path)
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    args = parser.parse_args(argv)
    try:
        summary = summarize_session(args.session_directory)
        rendered = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is None:
            sys.stdout.write(rendered)
        else:
            if args.output.resolve() in {
                (args.session_directory / name).resolve() for name in FILES
            }:
                raise DiagnosticError("--output must not overwrite a source diagnostic file")
            args.output.write_text(rendered)
    except (DiagnosticError, OSError) as exc:
        print(f"session summary failed: {exc}", file=sys.stderr)
        return 2
    for warning in summary["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
