from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from scripts.summarize_comparison_session import HASH_FIELDS, summarize_comparison
from scripts.summarize_interactive_session import DiagnosticError


def _fixture(directory: Path) -> list[Path]:
    initial: dict[str, Any] = {
        "run_id": "run",
        "entity_id": "entity",
        "session_epoch": 9007199254740993,
        **{key: "1" * 64 for key in HASH_FIELDS},
    }
    lanes = []
    paths = []
    for i, name in enumerate(("original", "shuffled_weight", "no_input")):
        entry: dict[str, Any] = {
            "name": name,
            "identity": {**initial, "dynamics_hash": str(i + 1) * 64},
            "control_kind": "faithful" if i == 0 else name,
            "control_sha256": None if i == 0 else "a" * 64,
            "seed": None if i == 0 else "18446744073709551615",
        }
        lanes.append(entry)
        epoch = initial["session_epoch"] + i
        path = directory / "epochs" / str(epoch)
        path.mkdir(parents=True)
        metadata = {
            **entry,
            "identity": {**entry["identity"], "session_epoch": epoch},
            "schema": "neurocraft.lane_epoch.v1",
            "diagnostic_only": True,
            "controller_profile": "program",
        }
        (path / "epoch.json").write_text(json.dumps(metadata))
        paths.append(path)
    catalog = json.dumps(
        {
            "schema": "neurocraft.lane_catalog.v1",
            "initial_identity": initial,
            "lanes": lanes,
        }
    )
    (directory / "lane-catalog.json").write_text(catalog)
    (directory / "session.json").write_text(
        json.dumps(
            {
                "comparison_lanes": True,
                "identity": initial,
                "controller_profile": "program",
                "run_id": "run",
                "entity_id": initial["entity_id"],
                "session_epoch": initial["session_epoch"],
                "lane_catalog_sha256": sha256(catalog.encode()).hexdigest(),
            }
        )
    )
    return paths


def _records(path: Path, *, receipt: bool) -> None:
    row: dict[str, Any] = {
        "schema": "neurocraft.program_decision.v1",
        "run_id": "run",
        "controller_profile": "program",
        "command_seq": 0,
        "observation_seq": 0,
        "target_tick": 102,
    }
    if receipt:
        row.update(
            kind="receipt",
            requested_program="CRUISE",
            applied_tick=102,
            validation="PROGRAM_APPLIED",
            executed_program="CRUISE",
        )
    else:
        row.update(
            kind="issued",
            program="CRUISE",
            observation_tick=100,
            neural_step_start=0,
            neural_step_end=5,
        )
    (path / "program-decisions.ndjson").write_text(json.dumps(row) + "\n")


def test_equal_sequences_never_join_across_epochs_and_prepared_is_not_active(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    _records(paths[0], receipt=False)
    _records(paths[1], receipt=True)
    result = summarize_comparison(tmp_path)
    epochs = result["epochs"]
    assert epochs[0]["programs"]["matched_pairs"] == 0
    assert epochs[0]["programs"]["issued_without_matching_receipt"] == [0]
    assert epochs[1]["programs"]["receipts_without_matching_issue"] == [0]
    assert epochs[1]["recorded_applied_program"] is True
    assert epochs[2]["programs"] is None
    assert epochs[2]["recorded_applied_program"] is False
    assert epochs[2]["epoch_metadata"]["seed"] == "18446744073709551615"
    assert epochs[0]["epoch_metadata"]["identity"]["session_epoch"] == 9007199254740993


@pytest.mark.parametrize("mutation", ["name", "epoch", "hash", "seed", "profile", "float"])
def test_epoch_sidecar_cannot_relabel_another_lane(tmp_path: Path, mutation: str) -> None:
    paths = _fixture(tmp_path)
    path = paths[0] / "epoch.json"
    metadata = json.loads(path.read_text())
    if mutation == "name":
        metadata["name"] = "no_input"
    elif mutation == "epoch":
        metadata["identity"]["session_epoch"] += 1
    elif mutation == "hash":
        metadata["identity"]["dynamics_hash"] = "f" * 64
    elif mutation == "seed":
        metadata["seed"] = "7"
    elif mutation == "profile":
        metadata["controller_profile"] = "other"
    else:
        metadata["identity"]["session_epoch"] = float(metadata["identity"]["session_epoch"])
    path.write_text(json.dumps(metadata))
    with pytest.raises(DiagnosticError):
        summarize_comparison(tmp_path)


def test_changed_catalog_and_foreign_records_are_refused(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _records(paths[0], receipt=False)
    records = paths[0] / "program-decisions.ndjson"
    records.write_text(records.read_text().replace('"run"', '"foreign"'))
    with pytest.raises(DiagnosticError, match="conflicting run_id"):
        summarize_comparison(tmp_path)
    catalog = tmp_path / "lane-catalog.json"
    catalog.write_text(catalog.read_text() + " ")
    with pytest.raises(DiagnosticError, match="catalog digest"):
        summarize_comparison(tmp_path)


@pytest.mark.parametrize("field", ["run_id", "entity_id", "session_epoch"])
def test_session_headline_cannot_disagree_with_bound_identity(tmp_path: Path, field: str) -> None:
    _fixture(tmp_path)
    path = tmp_path / "session.json"
    session = json.loads(path.read_text())
    session[field] = 99 if field == "session_epoch" else "different"
    path.write_text(json.dumps(session))
    with pytest.raises(DiagnosticError, match="session envelope"):
        summarize_comparison(tmp_path)
