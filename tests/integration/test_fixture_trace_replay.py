from __future__ import annotations

from pathlib import Path

import pytest

from scripts import fixture_harness
from scripts.fixture_harness import replay_trace, run_harness

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "data" / "fixtures" / "mcns-v1.0-flight-power-v1.json"


def test_fixture_harness_trace_replays_exactly(tmp_path: Path) -> None:
    trace = tmp_path / "harness.ndjson"
    run_receipt = run_harness(fixture=FIXTURE, output=trace, ticks=24)
    replay_receipt = replay_trace(fixture=FIXTURE, trace=trace)

    assert run_receipt["record_count"] == 24
    assert "not a Minecraft run" in run_receipt["evidence_boundary"]
    assert replay_receipt["status"] == "exact_match"
    assert replay_receipt["claim_scope"] == "G2"
    assert replay_receipt["replay_implementation_sha256"] == (
        fixture_harness.replay_implementation_sha256()
    )
    assert replay_receipt["record_count"] == 24
    assert replay_receipt["world_transitions_cross_linked"] == 23
    assert replay_receipt["terminal_world_transitions_not_cross_linked"] == 1
    assert replay_receipt["file_sha256"] == run_receipt["file_sha256"]


def test_fixture_harness_refuses_trace_overwrite(tmp_path: Path) -> None:
    trace = tmp_path / "harness.ndjson"
    run_harness(fixture=FIXTURE, output=trace, ticks=1)
    with pytest.raises(FileExistsError):
        run_harness(fixture=FIXTURE, output=trace, ticks=1)


def test_fixture_replay_rejects_current_implementation_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = tmp_path / "harness.ndjson"
    run_harness(fixture=FIXTURE, output=trace, ticks=2)
    monkeypatch.setattr(fixture_harness, "replay_implementation_sha256", lambda: "f" * 64)
    with pytest.raises(ValueError, match="current replay environment"):
        replay_trace(fixture=FIXTURE, trace=trace)
