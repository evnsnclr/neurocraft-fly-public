from __future__ import annotations

import json
from pathlib import Path

import pytest
from neurocraft_experiments.evaluation import (
    MAX_EVALUATION_PROTOCOL_BYTES,
    load_evaluation_protocol,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "experiments" / "preregistration" / "control-matrix-v2.json"
HISTORICAL_PROTOCOL = ROOT / "experiments" / "preregistration" / "control-matrix-v1.json"


def test_control_matrix_is_prerun_complete_and_content_addressed() -> None:
    protocol = load_evaluation_protocol(PROTOCOL)
    assert protocol.status == "preregistered_no_trials_run"
    assert protocol.planned_trial_count == 640
    assert protocol.development_seeds == (101, 211, 307, 401)
    assert protocol.held_out_seeds == (503, 601, 701, 809, 907, 1009)
    assert protocol.raw["primary_endpoint"]["metric"] == "completion_rate"
    assert "not independent statistical units" in protocol.raw["analysis"]["replicate_role"]
    assert len(protocol.sha256) == 64
    assert any(not condition.receives_sensors for condition in protocol.conditions)
    assert protocol.raw["amendment"]["supersedes"] == "neurocraft-control-matrix-v1"


def test_historical_v1_is_not_silently_reinterpreted_as_v2() -> None:
    with pytest.raises(ValueError, match="control-matrix-v2"):
        load_evaluation_protocol(HISTORICAL_PROTOCOL)


def test_evaluation_protocol_oversize_refused(tmp_path: Path) -> None:
    """An oversized protocol is refused by the descriptor gate, never parsed."""

    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["hypothesis"] = "x" * (MAX_EVALUATION_PROTOCOL_BYTES + 1)
    oversize = tmp_path / "oversize-protocol.json"
    oversize.write_text(json.dumps(payload), encoding="utf-8")
    assert oversize.stat().st_size > MAX_EVALUATION_PROTOCOL_BYTES

    with pytest.raises(ValueError, match=r"evaluation protocol exceeds the \d+") as refusal:
        load_evaluation_protocol(oversize)
    assert "protocol fields" not in str(refusal.value)


def test_result_embedding_and_missing_control_fail_closed(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["results"] = {"completion_rate": 1.0}
    result_path = tmp_path / "bad-results.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        load_evaluation_protocol(result_path)

    payload.pop("results")
    payload["conditions"] = [
        condition for condition in payload["conditions"] if condition["mode"] != "scripted_baseline"
    ]
    payload["analysis"]["trial_count"] -= 40
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="mandatory"):
        load_evaluation_protocol(result_path)
