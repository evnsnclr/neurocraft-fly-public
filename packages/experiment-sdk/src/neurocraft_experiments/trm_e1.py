"""The frozen TRM E1 runner: Structure vs. Learning v1 (ADR 0008, D-045).

Everything the preregistration binds is a module-level constant here, and the
freeze pins this file's exact bytes: changing any experimental choice after
results exist requires a new, openly recorded protocol version, never an edit.

Run-identity hardening (closing the recorded review findings): every cell
builds a fresh engine and computes its content-addressed run identity in one
atomic helper before a single step executes, with the condition label taken
from the same call that built the engine — the label cannot be reassigned, a
pre-stepped engine cannot be hashed, and the drive is validated by
construction before it is hashed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from neurocraft_connectome import RuntimeGraph, load_malecns_fixture
from neurocraft_connectome.flight_pin import MALECNS_FLIGHT_FIXTURE_V1_PIN
from neurocraft_connectome.model import canonical_json_bytes, sha256_json
from neurocraft_neural import NeuralEngine

from .reservoir import (
    build_condition_engine,
    drive_and_collect,
    reservoir_run_identity,
    ridge_readout,
)
from .trm_tasks import (
    build_drive,
    delayed_reconstruction_targets,
    narma10_targets,
)

PROTOCOL_ID = "trm-structure-vs-learning-v1"
RESULT_SCHEMA_TRM = "neurocraft.trm.e1_result.v1"

SENSORY_BODY_IDS = (77_185, 805_722)
WASHOUT_STEPS = 100
TRAIN_STEPS = 400
VALIDATION_STEPS = 100
TEST_STEPS = 200
TOTAL_STEPS = WASHOUT_STEPS + TRAIN_STEPS + VALIDATION_STEPS + TEST_STEPS
DRIVE_AMPLITUDE = 0.5
RECONSTRUCTION_DELAYS = (1, 2, 4, 8, 16)
L2_GRID = (1e-6, 1e-4, 1e-2, 1.0)
DRIVE_SEED_COUNT = 10
NULL_SAMPLE_COUNT = 199
REWIRE_REQUESTED_SWAPS = 887
DETERMINISTIC_CONDITIONS = ("connectome", "sign_ablated_zero")
STOCHASTIC_CONDITIONS = ("sign_permuted", "weight_shuffled", "degree_preserving_rewire")
TASKS = ("delayed_reconstruction", "narma10")

STEERING_FIXTURE_RELATIVE = "data/fixtures/mcns-v1.0-flight-steering-v1.json"
STEERING_FIXTURE_BYTES = 284_949


class TrmProtocolError(ValueError):
    """Raised when a cell request or result write violates the frozen protocol."""


def load_steering_graph(repository_root: Path) -> RuntimeGraph:
    return load_malecns_fixture(
        repository_root / STEERING_FIXTURE_RELATIVE,
        expected_pin=MALECNS_FLIGHT_FIXTURE_V1_PIN,
        expected_file_bytes=STEERING_FIXTURE_BYTES,
    )


def condition_seed(condition: str, null_index: int) -> int:
    """Per-condition seed derivation, closing the shared-permutation finding."""

    material = sha256_json(
        {
            "condition": condition,
            "null_index": null_index,
            "protocol_id": PROTOCOL_ID,
            "role": "null",
        }
    )
    return int(material[:16], 16)


def built_run(
    graph: RuntimeGraph,
    condition: str,
    null_index: int,
    drive: Sequence[Mapping[int, float]],
) -> tuple[NeuralEngine, str]:
    """Fresh engine plus its run identity, atomically, before any step."""

    engine = build_condition_engine(
        graph,
        condition,
        seed=condition_seed(condition, null_index),
        protocol_id=PROTOCOL_ID,
        requested_swaps=REWIRE_REQUESTED_SWAPS,
    )
    if engine.step_index != 0:
        raise TrmProtocolError("run identities are defined over fresh engines only")
    identity = reservoir_run_identity(
        engine,
        condition=condition,
        seed=condition_seed(condition, null_index),
        protocol_id=PROTOCOL_ID,
        drive=drive,
    )
    return engine, identity


def _augmented(rows: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    """The declared intercept: one constant 1.0 column appended to each row."""

    return tuple((*row, 1.0) for row in rows)


def _mse(
    states: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    weights: Sequence[Sequence[float]],
) -> float:
    total = 0.0
    count = 0
    for row, target in zip(states, targets, strict=True):
        for out_index, expected in enumerate(target):
            predicted = math.fsum(
                row[feature] * weights[feature][out_index] for feature in range(len(row))
            )
            total += (predicted - expected) ** 2
            count += 1
    if count == 0:
        raise TrmProtocolError("mse requires at least one prediction")
    value = total / count
    if not math.isfinite(value):
        raise TrmProtocolError("non-finite mse: the protocol refuses divergent fits")
    return value


def _fit_task(
    states: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
) -> dict[str, float]:
    """Train on the train slice, choose l2 on validation, report test MSE."""

    x = _augmented(states)
    train = slice(WASHOUT_STEPS, WASHOUT_STEPS + TRAIN_STEPS)
    val = slice(train.stop, train.stop + VALIDATION_STEPS)
    test = slice(val.stop, val.stop + TEST_STEPS)
    best_l2 = None
    best_val = None
    best_weights = None
    for l2 in L2_GRID:
        weights = ridge_readout(x[train], targets[train], l2=l2)
        val_mse = _mse(x[val], targets[val], weights)
        if best_val is None or val_mse < best_val:
            best_l2, best_val, best_weights = l2, val_mse, weights
    assert best_l2 is not None and best_val is not None and best_weights is not None
    return {
        "chosen_l2": best_l2,
        "validation_mse": best_val,
        "test_mse": _mse(x[test], targets[test], best_weights),
    }


def run_cell(
    graph: RuntimeGraph,
    condition: str,
    null_index: int,
    drive_seed_index: int,
) -> dict[str, Any]:
    """One (condition, null instance, drive seed) cell, fully deterministic."""

    if condition in DETERMINISTIC_CONDITIONS and null_index != 0:
        raise TrmProtocolError(f"{condition} is deterministic: null_index must be 0")
    drive, streams = build_drive(
        PROTOCOL_ID,
        seed_index=drive_seed_index,
        steps=TOTAL_STEPS,
        sensory_body_ids=SENSORY_BODY_IDS,
        amplitude=DRIVE_AMPLITUDE,
    )
    engine, run_identity = built_run(graph, condition, null_index, drive)
    states = drive_and_collect(engine, drive, protocol_id=PROTOCOL_ID)

    reconstruction = delayed_reconstruction_targets(streams, delays=RECONSTRUCTION_DELAYS)
    narma_input = tuple(value * DRIVE_AMPLITUDE for value in streams[min(SENSORY_BODY_IDS)])
    narma = tuple((value,) for value in narma10_targets(narma_input))

    row: dict[str, Any] = {
        "condition": condition,
        "drive_seed_index": drive_seed_index,
        "null_index": null_index,
        "run_identity_sha256": run_identity,
        "tasks": {
            "delayed_reconstruction": _fit_task(states, reconstruction),
            "narma10": _fit_task(states, narma),
        },
    }
    if condition == "degree_preserving_rewire":
        row["accepted_swaps"] = engine.topology_overlay.accepted_swaps
    return row


def enumerate_cells() -> tuple[tuple[str, int, int], ...]:
    cells: list[tuple[str, int, int]] = []
    for condition in DETERMINISTIC_CONDITIONS:
        for drive_seed in range(DRIVE_SEED_COUNT):
            cells.append((condition, 0, drive_seed))
    for condition in STOCHASTIC_CONDITIONS:
        for null_index in range(NULL_SAMPLE_COUNT):
            for drive_seed in range(DRIVE_SEED_COUNT):
                cells.append((condition, null_index, drive_seed))
    return tuple(cells)


_WORKER_GRAPH: RuntimeGraph | None = None


def _worker_initialize(repository_root: str) -> None:
    global _WORKER_GRAPH
    _WORKER_GRAPH = load_steering_graph(Path(repository_root))


def _worker_cell(cell: tuple[str, int, int]) -> dict[str, Any]:
    if _WORKER_GRAPH is None:
        raise TrmProtocolError("worker graph was not initialized")
    condition, null_index, drive_seed = cell
    return run_cell(_WORKER_GRAPH, condition, null_index, drive_seed)


def _holm_bonferroni(p_values: Mapping[str, float], alpha: float = 0.05) -> dict[str, bool]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    rejected: dict[str, bool] = {}
    still_rejecting = True
    for rank, (name, p_value) in enumerate(ordered):
        threshold = alpha / (len(ordered) - rank)
        still_rejecting = still_rejecting and p_value <= threshold
        rejected[name] = still_rejecting
    return rejected


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Paired comparison: per null instance, mean test MSE across drive seeds."""

    def instance_means(condition: str, task: str) -> list[float]:
        by_instance: dict[int, list[float]] = {}
        for row in rows:
            if row["condition"] == condition:
                by_instance.setdefault(int(row["null_index"]), []).append(
                    float(row["tasks"][task]["test_mse"])
                )
        means = []
        for null_index in sorted(by_instance):
            values = by_instance[null_index]
            if len(values) != DRIVE_SEED_COUNT:
                raise TrmProtocolError(
                    f"{condition} null {null_index} has {len(values)} of "
                    f"{DRIVE_SEED_COUNT} drive seeds"
                )
            means.append(sum(values) / len(values))
        return means

    summary: dict[str, Any] = {"comparisons": {}, "condition_means": {}}
    for task in TASKS:
        for condition in DETERMINISTIC_CONDITIONS + STOCHASTIC_CONDITIONS:
            means = instance_means(condition, task)
            summary["condition_means"][f"{task}:{condition}"] = {
                "instances": len(means),
                "mean_of_means": sum(means) / len(means),
                "min": min(means),
                "max": max(means),
            }
    p_values: dict[str, float] = {}
    for task in TASKS:
        connectome = instance_means("connectome", task)
        connectome_mean = sum(connectome) / len(connectome)
        for condition in STOCHASTIC_CONDITIONS:
            nulls = instance_means(condition, task)
            better_or_equal = sum(1 for value in nulls if value <= connectome_mean)
            p_values[f"{task}:connectome_vs_{condition}"] = (1 + better_or_equal) / (len(nulls) + 1)
    summary["empirical_p_lower_mse_is_better"] = p_values
    summary["holm_bonferroni_rejected_at_0_05"] = _holm_bonferroni(p_values)
    return summary


def execute_protocol(
    repository_root: Path,
    output: Path,
    *,
    workers: int = 1,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run every cell, write the result exactly once, and return the document."""

    if output.exists():
        raise TrmProtocolError(
            f"result already exists at {output}; results are immutable — a rerun "
            "needs a new declared output, never an overwrite"
        )
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise TrmProtocolError("workers must be a positive integer")
    cells = enumerate_cells()
    if limit is not None:
        cells = cells[:limit]
    graph = load_steering_graph(repository_root)
    rows: list[dict[str, Any]]
    if workers == 1:
        rows = [run_cell(graph, *cell) for cell in cells]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_initialize,
            initargs=(str(repository_root),),
        ) as pool:
            rows = list(pool.map(_worker_cell, cells, chunksize=8))
    document: dict[str, Any] = {
        "schema": RESULT_SCHEMA_TRM,
        "protocol_id": PROTOCOL_ID,
        "evidence_scope": "NON_MINECRAFT_OFFLINE_TRM_RESERVOIR_ONLY",
        "fixture": {
            "path": STEERING_FIXTURE_RELATIVE,
            "runtime_identity_sha256": graph.manifest.runtime_identity_sha256,
        },
        "parameters": {
            "delays": list(RECONSTRUCTION_DELAYS),
            "drive_amplitude": DRIVE_AMPLITUDE,
            "drive_seed_count": DRIVE_SEED_COUNT,
            "l2_grid": list(L2_GRID),
            "null_sample_count": NULL_SAMPLE_COUNT,
            "rewire_requested_swaps": REWIRE_REQUESTED_SWAPS,
            "steps": {
                "test": TEST_STEPS,
                "train": TRAIN_STEPS,
                "validation": VALIDATION_STEPS,
                "washout": WASHOUT_STEPS,
            },
        },
        "cell_count": len(rows),
        "rows": rows,
        "summary": summarize(rows) if limit is None else {"partial": True},
    }
    payload = canonical_json_bytes(document) + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return document
