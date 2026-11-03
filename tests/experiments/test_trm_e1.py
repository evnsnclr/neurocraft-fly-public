"""Frozen-runner tests: determinism, protocol shape, refusal paths, analysis."""

from __future__ import annotations

from pathlib import Path

import pytest
from neurocraft_connectome import RuntimeGraph
from neurocraft_experiments.trm_e1 import (
    DETERMINISTIC_CONDITIONS,
    DRIVE_SEED_COUNT,
    NULL_SAMPLE_COUNT,
    STOCHASTIC_CONDITIONS,
    TrmProtocolError,
    condition_seed,
    enumerate_cells,
    execute_protocol,
    load_steering_graph,
    run_cell,
    summarize,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def graph() -> RuntimeGraph:
    return load_steering_graph(REPO_ROOT)


def test_cell_enumeration_matches_the_declared_power() -> None:
    cells = enumerate_cells()
    assert len(cells) == (
        len(DETERMINISTIC_CONDITIONS) * DRIVE_SEED_COUNT
        + len(STOCHASTIC_CONDITIONS) * NULL_SAMPLE_COUNT * DRIVE_SEED_COUNT
    )
    # 2x10 deterministic + 3x199x10 stochastic; N=199 puts the minimum
    # achievable empirical p at 1/200 = 0.005, strictly below the first Holm
    # threshold 0.05/6 (the N=100 draft could never reject anything).
    assert len(cells) == 5_990
    assert len(set(cells)) == len(cells)


def test_condition_seeds_never_collide_across_conditions() -> None:
    seeds = {
        condition_seed(condition, index)
        for condition in STOCHASTIC_CONDITIONS
        for index in range(NULL_SAMPLE_COUNT)
    }
    assert len(seeds) == len(STOCHASTIC_CONDITIONS) * NULL_SAMPLE_COUNT


def test_run_cell_is_deterministic_and_shaped(graph: RuntimeGraph) -> None:
    first = run_cell(graph, "connectome", 0, 0)
    second = run_cell(graph, "connectome", 0, 0)
    assert first == second
    assert first["run_identity_sha256"] and len(first["run_identity_sha256"]) == 64
    for task in ("delayed_reconstruction", "narma10"):
        fit = first["tasks"][task]
        assert fit["test_mse"] > 0.0 and fit["validation_mse"] > 0.0
        assert fit["chosen_l2"] in (1e-6, 1e-4, 1e-2, 1.0)


def test_run_cell_distinguishes_conditions_and_reports_swaps(
    graph: RuntimeGraph,
) -> None:
    connectome = run_cell(graph, "connectome", 0, 1)
    rewired = run_cell(graph, "degree_preserving_rewire", 3, 1)
    assert connectome["run_identity_sha256"] != rewired["run_identity_sha256"]
    assert rewired["accepted_swaps"] > 0
    with pytest.raises(TrmProtocolError, match="deterministic"):
        run_cell(graph, "connectome", 1, 0)


def test_execute_protocol_smoke_writes_once_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    output = tmp_path / "trm-e1-smoke.json"
    document = execute_protocol(REPO_ROOT, output, limit=2)
    assert document["cell_count"] == 2
    assert document["summary"] == {"partial": True}
    assert output.exists()
    with pytest.raises(TrmProtocolError, match="immutable"):
        execute_protocol(REPO_ROOT, output, limit=2)


def test_summarize_computes_paired_empirical_p() -> None:
    rows = []
    for condition, instances, base in (
        ("connectome", 1, 0.10),
        ("sign_ablated_zero", 1, 0.50),
        ("sign_permuted", 4, 0.30),
        ("weight_shuffled", 4, 0.05),
        ("degree_preserving_rewire", 4, 0.30),
    ):
        for null_index in range(instances):
            for seed in range(DRIVE_SEED_COUNT):
                rows.append(
                    {
                        "condition": condition,
                        "null_index": null_index,
                        "drive_seed_index": seed,
                        "tasks": {
                            "delayed_reconstruction": {"test_mse": base + null_index * 0.01},
                            "narma10": {"test_mse": base + null_index * 0.01},
                        },
                    }
                )
    summary = summarize(rows)
    p_values = summary["empirical_p_lower_mse_is_better"]
    # All 4 sign_permuted instances have higher MSE than connectome: p = 1/5.
    assert p_values["narma10:connectome_vs_sign_permuted"] == pytest.approx(1 / 5)
    # All 4 weight_shuffled instances beat connectome: p = 5/5.
    assert p_values["narma10:connectome_vs_weight_shuffled"] == pytest.approx(1.0)
    assert set(summary["holm_bonferroni_rejected_at_0_05"]) == set(p_values)
