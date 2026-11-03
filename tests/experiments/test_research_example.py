"""Source-derived teaching experiments must reproduce, and fail visibly on drift."""

from __future__ import annotations

import csv
import io
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from scripts import research_example as example


@pytest.fixture(scope="module")
def result() -> dict[str, Any]:
    return example.simulate()


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "example"
    example.generate(path)
    return path


def _replace_and_rehash(dataset: Path, filename: str, payload: bytes) -> None:
    (dataset / filename).write_bytes(payload)
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    manifest["files"][filename] = {"bytes": len(payload), "sha256": sha256(payload).hexdigest()}
    (dataset / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))


def _alter_activity(dataset: Path, delta: float, *, identifier: bool = False) -> None:
    rows = list(csv.DictReader(io.StringIO((dataset / "activity.csv").read_text())))
    target = next(
        row for row in rows if row["condition"] == "original" and float(row["activity"]) > 0
    )
    if identifier:
        target["body_id"] = "999999999"
    else:
        target["activity"] = format(float(target["activity"]) + delta, ".17g")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _replace_and_rehash(dataset, "activity.csv", output.getvalue().encode())


def test_fixture_ids_counts_and_overlay_donor_lineage(result: dict[str, Any]) -> None:
    graph = example.load_graph()
    assert len(graph.nodes) == 14
    assert len(graph.edges) == 22
    assert graph.manifest.fixture_file_sha256 == (
        "3076c22fd0c87041c601b865c9fc8e913466200491f80775060e28412a087ff8"
    )
    assert {row["body_id"] for row in result["tables"]["nodes"]} == set(graph.body_ids)
    original = [row for row in result["tables"]["edges"] if row["condition"] == "original"]
    shuffled = [row for row in result["tables"]["edges"] if row["condition"] == "shuffled_weights"]
    assert sorted(row["overlay_count"] for row in shuffled) == sorted(
        e.raw_count for e in graph.edges
    )
    assert any(
        a["overlay_count"] != b["overlay_count"] for a, b in zip(original, shuffled, strict=True)
    )
    for row in shuffled:
        source = graph.edges[row["edge_index"]]
        assert (row["body_pre"], row["body_post"], row["raw_count"], row["source_row"]) == (
            source.body_pre,
            source.body_post,
            source.raw_count,
            source.source_row,
        )
        assert row["overlay_count"] == graph.edges[row["overlay_source_edge_index"]].raw_count
        assert row["canonical_sign_status"] == "unknown"


def test_requested_inputs_match_but_no_input_is_zero(result: dict[str, Any]) -> None:
    schedules = {}
    for condition in example.CONDITIONS:
        rows = [row for row in result["tables"]["inputs"] if row["condition"] == condition]
        schedules[condition] = [
            (r["step_before"], r["body_id"], r["requested_drive"]) for r in rows
        ]
        assert len(rows) == 200
        if condition == "no_input":
            assert all(row["applied_drive"] == 0 for row in rows)
        else:
            assert sum(row["applied_drive"] != 0 for row in rows) == 40
            assert all(row["requested_drive"] == row["applied_drive"] for row in rows)
    assert schedules["original"] == schedules["shuffled_weights"] == schedules["no_input"]
    for row in result["tables"]["activity"]:
        if row["condition"] == "no_input" or row["step"] <= 10:
            assert row["activity"] == 0
    assert any(row["activity"] < 0 for row in result["tables"]["activity"])


def test_analytic_direct_pulse_and_declared_reference_delay(result: dict[str, Any]) -> None:
    import math

    activities = {
        (r["step"], r["body_id"]): r["activity"]
        for r in result["tables"]["activity"]
        if r["condition"] == "original"
    }
    expected = (1 - math.exp(-10 / 50)) * math.tanh(0.8)
    assert activities[11, 805722] == pytest.approx(expected, rel=1e-14)
    # This source node projects to 10180; a[n-1] enters a[n+1].
    assert activities[12, 10180] == 0
    assert activities[13, 10180] > 0


def test_inspection_pools_equal_source_activity_and_shuffle_retains_response(
    result: dict[str, Any],
) -> None:
    lookup = {
        (row["condition"], row["step"], row["body_id"]): row["activity"]
        for row in result["tables"]["activity"]
    }
    for row in result["tables"]["readouts"]:
        assert (
            row["mean_activity"]
            == lookup[row["condition"], row["step"], example.POOLS[row["pool"]][0]]
        )
    for pool, values in result["summary"]["comparison"].items():
        assert values["shuffled_response_retained"]
        if pool.startswith("sensory_"):
            assert values["original_shuffled_max_abs_difference"] == 0
        else:
            assert values["original_shuffled_max_abs_difference"] > 0.01


def test_zero_drive_and_zero_recurrence_remove_downstream_response() -> None:
    zero_drive = example.simulate(pulse_amplitude=0)
    assert all(row["activity"] == 0 for row in zero_drive["tables"]["activity"])
    no_recurrence = example.simulate(recurrent_gain=0)
    assert any(row["activity"] != 0 for row in no_recurrence["tables"]["activity"])
    assert all(
        row["activity"] == 0
        for row in no_recurrence["tables"]["activity"]
        if row["body_id"] not in example.TARGETS
    )


def test_different_seed_changes_only_shuffled_condition(result: dict[str, Any]) -> None:
    changed = example.simulate(shuffle_seed=8)
    for condition in example.CONDITIONS:
        left = [r for r in result["tables"]["activity"] if r["condition"] == condition]
        right = [r for r in changed["tables"]["activity"] if r["condition"] == condition]
        assert (left != right) == (condition == "shuffled_weights")
    assert result["metadata"]["source_graph"] == changed["metadata"]["source_graph"]


def test_generation_is_byte_reproducible_and_no_clobber(dataset: Path, tmp_path: Path) -> None:
    other = tmp_path / "other"
    example.generate(other)
    for path in dataset.iterdir():
        assert path.read_bytes() == (other / path.name).read_bytes()
    assert example.verify(dataset)["mode"] == "byte_exact_replay"
    with pytest.raises(example.ResearchExampleError, match="already exists"):
        example.generate(dataset)
    assert example.verify(dataset)["verified"]


def test_checked_in_example_replays_and_declares_separate_rights() -> None:
    # Checked-in floats were generated on another host. Keep the same-host
    # generation test byte-exact and use the documented portable replay here.
    report = example.verify(tolerant=True)
    assert report["verified"] and report["mode"] == "numerical_replay"
    # Numerical replay already succeeded above; do not repeat an exact replay.
    loaded = example.load_dataset(replay=False)
    rights = loaded["manifest"]["example_data_rights"]
    assert rights["copied_source_structure_and_annotations"] == "CC-BY-4.0"
    assert rights["project_generated_simulation_outputs"] == "MIT"
    assert len(loaded["tables"]["activity"]) == 4242


@pytest.mark.parametrize("fault", ["missing", "extra", "hash", "symlink", "malformed_manifest"])
def test_refuses_incomplete_or_tampered_storage(dataset: Path, fault: str) -> None:
    if fault == "missing":
        (dataset / "inputs.csv").unlink()
    elif fault == "extra":
        (dataset / "unexpected.txt").write_text("extra")
    elif fault == "hash":
        with (dataset / "activity.csv").open("a") as handle:
            handle.write("tampered\n")
    elif fault == "symlink":
        (dataset / "inputs.csv").unlink()
        (dataset / "inputs.csv").symlink_to(dataset / "readouts.csv")
    else:
        (dataset / "manifest.json").write_text("[]")
    with pytest.raises(example.ResearchExampleError):
        example.verify(dataset, tolerant=True)


@pytest.mark.parametrize("tolerant", [False, True])
def test_rehashed_material_changes_fail_replay(dataset: Path, tolerant: bool) -> None:
    _alter_activity(dataset, 0.01)
    with pytest.raises(example.ResearchExampleError, match="replay mismatch"):
        example.verify(dataset, tolerant=tolerant)


def test_tolerance_is_explicit_and_does_not_relax_identifiers(dataset: Path) -> None:
    _alter_activity(dataset, 1e-14)
    with pytest.raises(example.ResearchExampleError, match="byte-exact replay"):
        example.verify(dataset)
    assert example.verify(dataset, tolerant=True)["mode"] == "numerical_replay"
    _alter_activity(dataset, 0, identifier=True)
    with pytest.raises(example.ResearchExampleError, match="CSV identity mismatch"):
        example.verify(dataset, tolerant=True)


@pytest.mark.parametrize("fault", ["code", "source", "dynamics", "numeric_type", "duplicate_key"])
def test_manifest_semantics_are_exact_even_under_tolerance(dataset: Path, fault: str) -> None:
    path = dataset / "manifest.json"
    manifest = json.loads(path.read_text())
    if fault == "code":
        manifest["software"]["source_sha256"]["scripts/research_example.py"] = "0" * 64
    elif fault == "source":
        manifest["source_graph"]["runtime_identity_sha256"] = "0" * 64
    elif fault == "dynamics":
        manifest["dynamics"]["recurrent_gain"] = 0.5
    elif fault == "numeric_type":
        manifest["input_schedule"]["steps"] = 100.0
    path.write_text(json.dumps(manifest, sort_keys=True))
    if fault == "duplicate_key":
        path.write_text(
            path.read_text().replace(
                '"schema_version": 1', '"schema_version": 1, "schema_version": 1', 1
            )
        )
    with pytest.raises(example.ResearchExampleError):
        example.verify(dataset, tolerant=True)
