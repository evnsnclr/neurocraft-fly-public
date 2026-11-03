from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.summarize_interactive_session import DiagnosticError, main, summarize_session


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _issued(sequence: int, program: str = "CRUISE", before: str = "CRUISE") -> dict[str, Any]:
    return {
        "schema": "neurocraft.program_decision.v1",
        "run_id": "test-run",
        "controller_profile": "test-program",
        "kind": "issued",
        "command_seq": sequence,
        "observation_seq": sequence,
        "observation_tick": 100 + sequence,
        "target_tick": 102 + sequence,
        "program": program,
        "neural_step_start": sequence * 5,
        "neural_step_end": (sequence + 1) * 5,
        "policy_state_bound": True,
        "policy_state_before": {"program": before},
        "policy_state_after": {"program": program},
    }


def _receipt(
    sequence: int,
    requested: str = "CRUISE",
    validation: str = "PROGRAM_APPLIED",
    executed: str | None = "CRUISE",
) -> dict[str, Any]:
    return {
        "schema": "neurocraft.program_decision.v1",
        "run_id": "test-run",
        "controller_profile": "test-program",
        "kind": "receipt",
        "command_seq": sequence,
        "observation_seq": sequence,
        "target_tick": 102 + sequence,
        "applied_tick": 102 + sequence,
        "requested_program": requested,
        "validation": validation,
        "executed_program": executed,
    }


def test_receipt_joins_separate_execution_from_requests_and_escape_holds(tmp_path: Path) -> None:
    records = [
        _receipt(1, "ESCAPE_JUMP", "PROGRAM_REFUSED", "HOVER"),  # deliberately out of order
        _issued(0),
        _issued(1, "ESCAPE_JUMP"),
        _issued(2, "ESCAPE_JUMP", "ESCAPE_JUMP"),
        _issued(3),
        _receipt(0),
        _receipt(3, validation="REJECTED", executed=None),
        _receipt(7),  # receipt whose issued record is missing
    ]
    unbound = _issued(4, "ESCAPE_JUMP")
    unbound["policy_state_bound"] = False
    records.append(unbound)
    _write_records(tmp_path / "program-decisions.ndjson", records)

    summary = summarize_session(tmp_path)
    programs = summary["programs"]
    assert programs["issued_frames"] == 5
    assert programs["receipt_frames"] == 4
    assert programs["matched_pairs"] == 3
    assert programs["applied_receipts"] == 2  # includes the explicitly unmatched receipt
    assert programs["refused_receipts"] == 1
    assert programs["rejected_receipts"] == 1
    assert programs["issued_without_matching_receipt"] == [2, 4]
    assert programs["receipts_without_matching_issue"] == [7]
    assert programs["reported_executed_programs_by_validation"] == {
        "PROGRAM_APPLIED": {"CRUISE": 2},
        "PROGRAM_REFUSED": {"HOVER": 1},
        "REJECTED": {},
    }
    assert programs["escape_jump"] == {
        "explicit_policy_onsets": 1,
        "onset_command_sequences": [1],
        "policy_hold_frames": 1,
        "unclassified_frames": 1,
        "basis": "bound recorded policy state; an issued onset does not prove execution",
    }
    assert programs["issued_neural_step_range"] == {"min": 0, "max": 25}
    assert summary["observed_simulated_time"]["tick_span"] == 9
    assert summary["observed_simulated_time"]["seconds_at_20_ticks_per_second"] == 0.45


def test_equal_sequence_with_wrong_binding_is_unmatched_on_both_sides(tmp_path: Path) -> None:
    receipt = _receipt(0)
    receipt["target_tick"] = 999
    _write_records(tmp_path / "program-decisions.ndjson", [_issued(0), receipt])
    summary = summarize_session(tmp_path)
    assert summary["programs"]["matched_pairs"] == 0
    assert summary["programs"]["binding_mismatch_command_sequences"] == [0]
    assert summary["programs"]["unmatched_issued_count"] == 1
    assert summary["programs"]["unmatched_receipt_count"] == 1
    assert any("binding mismatches" in warning for warning in summary["warnings"])


def test_missing_sources_are_unavailable_not_zero(tmp_path: Path) -> None:
    summary = summarize_session(tmp_path)
    assert summary["programs"] is None
    assert summary["compute_latency"] is None
    assert summary["server_status"] is None
    assert summary["session"]["recorded_final_state"] is None
    assert summary["observed_simulated_time"]["tick_span"] is None
    assert len(summary["warnings"]) == 4
    assert set(summary["files"].values()) == {"unavailable"}


def test_torn_final_record_warns_but_keeps_complete_records(tmp_path: Path) -> None:
    path = tmp_path / "program-decisions.ndjson"
    path.write_bytes(json.dumps(_issued(0)).encode() + b'\n{"kind": "receipt", "command_seq":')
    summary = summarize_session(tmp_path)
    assert summary["programs"]["issued_frames"] == 1
    assert summary["programs"]["receipt_frames"] == 0
    assert "program-decisions.ndjson:2: skipped incomplete final JSON record" in summary["warnings"]


@pytest.mark.parametrize("broken", [b"oops\n", b"{broken}\n", b"[]", b"\n"])
def test_malformed_interior_or_complete_record_is_never_discarded(
    tmp_path: Path,
    broken: bytes,
) -> None:
    path = tmp_path / "program-decisions.ndjson"
    path.write_bytes(json.dumps(_issued(0)).encode() + b"\n" + broken)
    with pytest.raises(DiagnosticError, match=r"program-decisions\.ndjson:2"):
        summarize_session(tmp_path)


def test_malformed_unterminated_record_interior_fails(tmp_path: Path) -> None:
    path = tmp_path / "program-decisions.ndjson"
    path.write_bytes(b'{"kind":\n' + json.dumps(_issued(0)).encode())
    with pytest.raises(DiagnosticError, match=r"program-decisions\.ndjson:1"):
        summarize_session(tmp_path)


def test_duplicate_or_foreign_records_do_not_silently_overwrite_joins(tmp_path: Path) -> None:
    path = tmp_path / "program-decisions.ndjson"
    _write_records(path, [_issued(0), _issued(0)])
    with pytest.raises(DiagnosticError, match="duplicate issued"):
        summarize_session(tmp_path)
    foreign = _receipt(0)
    foreign["run_id"] = "another-run"
    _write_records(path, [_issued(0), foreign])
    with pytest.raises(DiagnosticError, match="conflicting run_id"):
        summarize_session(tmp_path)


def test_compute_percentiles_status_extrema_and_cumulative_retirement_max(tmp_path: Path) -> None:
    session = {
        "run_id": "test-run",
        "controller_profile": "test-program",
        "identity": {"graph_hash": "recorded-hash"},
        "state": "stopped",
        "exit_reason": "user interrupted launcher",
        "requested_game_ticks_per_wall_second": 10,
        "playback_speed": "half-speed diagnostic",
    }
    (tmp_path / "session.json").write_text(json.dumps(session))
    _write_records(
        tmp_path / "action-latency.ndjson",
        [
            {
                "schema": "neurocraft.action_latency.v1",
                "run_id": "test-run",
                "controller_profile": "test-program",
                "command_seq": index,
                "compute_ns": milliseconds * 1_000_000,
            }
            for index, milliseconds in enumerate((10, 20, 30, 40, 50))
        ],
    )
    (tmp_path / "server-gradle.log").write_text(
        "[23:59:59] NeuroCraft bridge=ACTIVE position=(1,67,3) command_seq=0 "
        "neural_steps=0..5 lateness_retired=2 since_tick=1\n"
        "[00:00:01] BODY PROGRAM CRUISE -> ESCAPE_JUMP tick=100 source=READOUT\n"
        "[00:00:02] NeuroCraft bridge=ACTIVE position=(2,70,4) command_seq=5 "
        "neural_steps=25..30 lateness_retired=4 since_tick=100\n"
        "[00:00:03] BODY PROGRAM ESCAPE_JUMP -> CRUISE tick=120 source=READOUT\n"
        "[00:00:04] NeuroCraft bridge=DEADLINE_MISSED position=(2,68,4) command_seq=6 "
        "neural_steps=30..35 lateness_retired=4 since_tick=120\n"
    )
    summary = summarize_session(tmp_path)
    assert summary["session"]["identity"] == {"graph_hash": "recorded-hash"}
    assert summary["session"]["recorded_final_state"] == "stopped"
    assert summary["session"]["requested_game_ticks_per_wall_second"] == 10
    assert summary["session"]["playback_speed"] == "half-speed diagnostic"
    compute = summary["compute_latency"]
    assert compute["sample_count"] == 5
    assert compute["p50"] == 30
    assert compute["p95"] == 48
    assert compute["p99"] == pytest.approx(49.6)
    assert compute["max"] == 50
    assert "compute only" in compute["scope"]
    status = summary["server_status"]
    assert status["lateness_retired_max"] == 4  # 2 + 4 + 4 is wrong
    assert status["position_samples"]["y_min"] == 67
    assert status["position_samples"]["y_max"] == 70
    assert status["position_samples"]["final_xyz"] == [2, 68, 4]
    assert status["command_sequence_range"] == {"min": 0, "max": 6}
    assert status["neural_step_range"] == {"min": 0, "max": 35}
    assert status["terminal_bridge_faults_observed"][0]["state"] == "DEADLINE_MISSED"
    assert summary["observed_simulated_time"]["seconds_at_20_ticks_per_second"] == 1


def test_cli_stdout_output_file_and_visible_incomplete_warning(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "program-decisions.ndjson").write_bytes(b'{"kind":')
    assert main([str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["diagnostic_only"] is True
    assert "skipped incomplete final JSON record" in captured.err
    output = tmp_path / "summary.json"
    assert main([str(tmp_path), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["programs"]["issued_frames"] == 0
    original = (tmp_path / "program-decisions.ndjson").read_bytes()
    assert main([str(tmp_path), "--output", str(tmp_path / "program-decisions.ndjson")]) == 2
    assert (tmp_path / "program-decisions.ndjson").read_bytes() == original
