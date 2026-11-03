from __future__ import annotations

import json
from pathlib import Path

import pytest
from neurocraft_experiments import (
    ExperimentPlan,
    RawPhysicalObservation,
    RunMode,
    load_arena,
    scripted_baseline,
)
from neurocraft_experiments.arena import MAX_ARENA_BYTES

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ["light-corridor-v1.json", "crosswind-gap-v1.json"])
def test_arena_specs_are_strict_and_content_addressed(name: str) -> None:
    arena = load_arena(ROOT / "experiments" / "arenas" / name)
    assert len(arena.sha256) == 64
    assert arena.required_sensors
    assert arena.metrics


def test_arena_oversize_refused(tmp_path: Path) -> None:
    """The byte cap is applied on the descriptor, before the document is parsed."""

    payload = json.loads((ROOT / "experiments" / "arenas" / "light-corridor-v1.json").read_text())
    payload["description"] = "x" * (MAX_ARENA_BYTES + 1)
    oversize = tmp_path / "oversize-arena.json"
    oversize.write_text(json.dumps(payload), encoding="utf-8")
    assert oversize.stat().st_size > MAX_ARENA_BYTES

    with pytest.raises(ValueError, match=r"arena exceeds the \d+-byte limit") as refusal:
        load_arena(oversize)
    assert "unknown arena fields" not in str(refusal.value)

    payload["description"] = "within the cap"
    bounded = tmp_path / "bounded-arena.json"
    bounded.write_text(json.dumps(payload), encoding="utf-8")
    assert load_arena(bounded).description == "within the cap"


def test_scripted_control_is_bounded_and_explicitly_non_neural() -> None:
    action = scripted_baseline(RawPhysicalObservation(0.2, 2.0, 0.1, -0.5, False))
    assert all(
        -1.0 <= value <= 1.0
        for value in (
            action.forward_thrust,
            action.lateral_thrust,
            action.vertical_thrust,
            action.yaw_torque,
        )
    )
    assert action.yaw_torque < 0


def test_matched_plan_has_equal_cells_per_mode() -> None:
    plan = ExperimentPlan.matched(
        plan_id="control-matrix-v1",
        hypothesis="Topology contributes to corridor traversal.",
        primary_metric="completion_rate",
        arenas=("light-corridor-v1",),
        modes=(RunMode.FAITHFUL, RunMode.SHUFFLED_WEIGHT, RunMode.SCRIPTED_BASELINE),
        seeds=(7, 11),
        replicates=2,
    )
    counts = {mode: sum(cell.mode is mode for cell in plan.cells) for mode in RunMode}
    assert counts[RunMode.FAITHFUL] == counts[RunMode.SHUFFLED_WEIGHT] == 4
    assert counts[RunMode.FAITHFUL] == counts[RunMode.SCRIPTED_BASELINE]
    assert len(plan.sha256) == 64


def test_lesion_cell_requires_named_intervention() -> None:
    from neurocraft_experiments.preregister import TrialCell

    with pytest.raises(ValueError, match="intervention"):
        TrialCell("arena", RunMode.LESION, 1, 0)
