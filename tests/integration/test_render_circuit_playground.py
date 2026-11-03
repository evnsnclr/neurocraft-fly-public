from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from scripts import render_circuit_playground as report


def _save(path: Path, value: Any) -> str:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path: Path) -> Path:
    """Explicitly synthetic renderer data; this never runs or validates a neural model."""
    directory = tmp_path / "synthetic-evidence"
    directory.mkdir()
    plan = {
        "schema": report.SCHEMA,
        "scope": "SYNTHETIC TEST DATA",
        "origin": {"source_commit": "f" * 40},
        "sources": {"<script>source</script>": {"sha256": "a" * 64}},
        "matching_scope": "size/side/sign matched, activity/strength ranked",
        "ratio_floor": 1e-12,
        "onset": "synthetic test rule",
        "timing_resolution": "five neural steps per observation",
    }
    response = {
        "ttmn_signed_evoked_mean": [0, 0.2, -0.1],
        "ttmn_signed_evoked_lr": [[0, 0], [0.1, 0.3], [-0.2, 0]],
        "peak_abs_mean": 0.2,
        "integral_abs_mean_tick": 0.3,
        "peak_difference_from_original": 0.0,
        "peak_ratio_to_original": 1.0,
        "onset_tick": None,
        "issued_program_counts": {"CRUISE": 2, "GROOM": 1},
        "selected_sources": ["brush"],
    }
    masks = {
        "pairs": [
            {
                "type": "<img src=x onerror=alert(1)>",
                "body_ids": [101, 102],
                "pair_rms_feature_distance": 4.5,
                "within_original_activity_calipers": False,
                "matching_rank": 1,
                "seed": 20261024,
            }
        ]
    }
    plan_hash = _save(directory / "plan.json", plan)
    masks_hash = _save(directory / "frozen-comparison-masks.json", masks)
    summary = {
        "status": "complete",
        "claim_scope": "SYNTHETIC TEST DATA",
        "plan_sha256": report.canonical_digest(plan),
        "fingerprints": {
            name: {case: response for case, _ in report.EVENTS}
            for name in ("original", "shuffle_7", "shuffle_11", "shuffle_23", "no_input")
        },
        "pilot": {
            name: {"player_approach": response}
            for name in ("original", "sham", "dnp01_silenced", "comparison_00", "restored")
        },
        "restoration_exact": {"sham": True, "restored": True},
        "files": {"plan.json": plan_hash, "frozen-comparison-masks.json": masks_hash},
    }
    _save(directory / "summary.json", summary)
    return directory


def test_standalone_report_uses_recorded_counts_plots_scope_and_escaped_text(
    tmp_path: Path,
) -> None:
    directory = _bundle(tmp_path)
    output = report.write_report(directory, tmp_path / "reports" / "synthetic.html")
    document = output.read_text()
    assert document.startswith("<!doctype html>")
    assert "SYNTHETIC TEST DATA" in document
    assert "No Minecraft applications" in document
    assert '<span class="program"><b>2</b> CRUISE</span>' in document
    assert '<span class="program"><b>1</b> GROOM</span>' in document
    assert "3 issued observations" in document
    assert "Selected sources" in document and "brush" in document
    assert "seed 7" in document and "seed 11" in document and "seed 23" in document
    assert document.count("<svg ") == 3
    assert "Signed evoked TTMn left" in document and "Signed evoked TTMn right" in document
    assert "Outside descriptive calipers" in document
    assert "descriptive comparison sample" in document
    assert "&lt;script&gt;source&lt;/script&gt;" in document
    assert "&lt;img src=x onerror=alert(1)&gt;" in document
    assert "<script" not in document and "<img " not in document
    assert sha256((directory / "summary.json").read_bytes()).hexdigest() in document
    assert "f" * 40 in document and "a" * 64 in document


def test_output_cannot_replace_evidence_or_existing_report(tmp_path: Path) -> None:
    directory = _bundle(tmp_path)
    with pytest.raises(ValueError, match="outside the frozen"):
        report.write_report(directory, directory / "report.html")
    output = report.write_report(directory / "summary.json", tmp_path / "view.html")
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        report.write_report(directory, output)
    assert output.read_bytes() == original
    assert not (directory / "report.html").exists()


@pytest.mark.parametrize("filename", ["plan.json", "frozen-comparison-masks.json"])
def test_changed_consumed_evidence_is_refused(tmp_path: Path, filename: str) -> None:
    directory = _bundle(tmp_path)
    with (directory / filename).open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match=r"differs|differ"):
        report.write_report(directory, tmp_path / "view.html")


def test_null_ratio_and_missing_metrics_are_not_invented_as_zero(tmp_path: Path) -> None:
    directory = _bundle(tmp_path)
    path = directory / "summary.json"
    summary = json.loads(path.read_text())
    row = summary["pilot"]["dnp01_silenced"]["player_approach"]
    row["peak_ratio_to_original"] = None
    del row["peak_difference_from_original"]
    del row["integral_abs_mean_tick"]
    _save(path, summary)
    document = report.write_report(directory, tmp_path / "view.html").read_text()
    assert "Undefined" in document and "Unavailable" in document
    assert "No crossing / undefined" in document


def test_nonfinite_input_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"value":NaN}')
    with pytest.raises(ValueError, match="non-finite"):
        report.read_json(path)
