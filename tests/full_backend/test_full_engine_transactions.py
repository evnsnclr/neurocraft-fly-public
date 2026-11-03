"""Rejected state changes must leave live arrays and their attestation untouched."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from neurocraft_neural.full_engine import ArrayNativeEngine, ArraySnapshot
from neurocraft_neural.full_records import _activity_digest, array_snapshot_digest
from neurocraft_neural.full_view import ViewProvenance, build_view
from neurocraft_neural.model import NeuralConfigurationError, NeuralIntegrityError
from neurocraft_neural.spmv_kernel import NumpyReduceatKernel


@pytest.fixture
def engine() -> ArrayNativeEngine:
    view = build_view(
        indptr=[0, 1, 2],
        pre_indices=[1, 0],
        raw_counts=[1, 1],
        signs=[1, -1],
        provenance=ViewProvenance(
            origin="synthetic_transaction_checks", graph_identity_sha256="a" * 64
        ),
    )
    result = ArrayNativeEngine(view, kernel=NumpyReduceatKernel())
    result.reset([0.5, -0.25])
    result.step(direct=np.array([0.1, 0.2]))
    result.step(direct=np.array([0.1, 0.2]))
    return result


def assert_rejection_is_atomic(
    engine: ArrayNativeEngine, operation: Any, error: type[Exception]
) -> None:
    before = engine.snapshot()
    before_state = engine.current_state()
    before_events = engine.verification_events
    before_coverage = (engine.verified_through_step, engine.run_is_covered)
    with pytest.raises(error):
        operation()
    assert engine.snapshot() == before
    assert engine.current_state() == before_state
    assert engine.verification_events == before_events
    assert (engine.verified_through_step, engine.run_is_covered) == before_coverage
    assert _activity_digest(engine.activity()) == engine.current_state().activity_sha256


@pytest.mark.parametrize(
    "indices",
    [[0, 2], [0, -1], [0, 0.9], [0, True], [0, np.bool_(True)], [0, "1"]],
)
def test_rejected_lesion_does_not_partially_apply(engine: ArrayNativeEngine, indices: Any) -> None:
    assert_rejection_is_atomic(
        engine, lambda: engine.set_ablated(indices), NeuralConfigurationError
    )


@pytest.mark.parametrize("ablated", [1, None, "true", np.bool_(True)])
def test_lesion_flag_requires_boolean(engine: ArrayNativeEngine, ablated: Any) -> None:
    assert_rejection_is_atomic(
        engine, lambda: engine.set_ablated([0], ablated=ablated), NeuralConfigurationError
    )


def test_exact_numpy_integer_lesions_clear_current_and_delayed_state(
    engine: ArrayNativeEngine,
) -> None:
    indices: Any = [np.int64(0)]
    engine.set_ablated(indices)
    lesioned = engine.snapshot()
    assert lesioned.activity[0] == lesioned.previous[0] == 0.0
    assert lesioned.ablation_mask == (True, False)
    assert lesioned.snapshot_sha256 == array_snapshot_digest(lesioned)
    engine.step(direct=np.array([0.8, 0.2]))
    assert engine.activity()[0] == 0.0
    engine.set_ablated([0], ablated=False)
    engine.step(direct=np.array([0.8, 0.2]))
    assert engine.activity()[0] > 0.0


@pytest.mark.parametrize(
    "activity",
    [
        0.5,
        [0.5],
        [[0.5, 0.25]],
        [0.5, float("nan")],
        [0.5, float("inf")],
        [0.5, 1.01],
        [0.5, -1.01],
        ["0.5", "0.25"],
        [True, False],
        [0.5, 0.25j],
    ],
)
def test_rejected_reset_preserves_state_and_trace(engine: ArrayNativeEngine, activity: Any) -> None:
    assert_rejection_is_atomic(engine, lambda: engine.reset(activity), NeuralConfigurationError)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("step_index", -1),
        ("step_index", True),
        ("step_index", 1.5),
        ("trace_sha256", "not_a_digest"),
        ("trace_sha256", "A" * 64),
        ("trace_sha256", None),
        ("activity", (0.2, float("nan"))),
        ("activity", (0.2, 1.01)),
        ("activity", ((0.2, 0.3),)),
        ("activity", (0.2,)),
        ("previous", (0.2, float("inf"))),
        ("previous", (0.2, -1.01)),
        ("previous", (0.2,)),
        ("ablation_mask", (0, 1)),
        ("ablation_mask", (False,)),
        ("ablation_mask", ((False, False),)),
        ("ablation_mask", (True, False)),
    ],
)
def test_legacy_opt_in_still_validates_before_commit(
    engine: ArrayNativeEngine, field: str, value: Any
) -> None:
    malformed = replace(engine.snapshot(), snapshot_sha256="", **{field: value})
    assert_rejection_is_atomic(
        engine,
        lambda: engine.restore(malformed, allow_legacy_snapshot=True),
        NeuralIntegrityError,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("activity", (0.1, 0.2)),
        ("previous", (0.1, 0.2)),
        ("step_index", 12),
        ("trace_sha256", "b" * 64),
        ("snapshot_sha256", "b" * 64),
    ],
)
def test_snapshot_digest_rejects_changed_valid_content(
    engine: ArrayNativeEngine, field: str, value: Any
) -> None:
    tampered = replace(engine.snapshot(), **{field: value})
    assert_rejection_is_atomic(engine, lambda: engine.restore(tampered), NeuralIntegrityError)


def test_legacy_snapshot_requires_explicit_opt_in(engine: ArrayNativeEngine) -> None:
    current = engine.snapshot()
    # The new optional field preserves old constructor arguments.
    legacy = ArraySnapshot(
        current.dynamics_identity_sha256,
        current.view_sha256,
        current.step_index,
        current.activity,
        current.previous,
        current.ablation_mask,
        current.trace_sha256,
    )
    assert_rejection_is_atomic(engine, lambda: engine.restore(legacy), NeuralIntegrityError)
    engine.step(direct=np.array([0.1, 0.2]))
    restored = engine.restore(legacy, allow_legacy_snapshot=True)
    assert restored.step_index == current.step_index
    assert engine.snapshot() == current


def test_mask_history_and_replay_survive_snapshot_round_trip(
    engine: ArrayNativeEngine,
) -> None:
    engine.set_ablated([0])
    saved = engine.snapshot()
    first = engine.step(direct=np.array([0.6, 0.2]))
    first_snapshot = engine.snapshot()
    engine.restore(saved)
    repeated = engine.step(direct=np.array([0.6, 0.2]))
    assert repeated == first
    assert engine.snapshot() == first_snapshot


def test_reset_and_restore_reject_other_types_without_mutation(
    engine: ArrayNativeEngine,
) -> None:
    invalid_snapshot: Any = None
    invalid_flag: Any = 1
    assert_rejection_is_atomic(
        engine, lambda: engine.restore(invalid_snapshot), NeuralIntegrityError
    )
    assert_rejection_is_atomic(
        engine,
        lambda: engine.restore(engine.snapshot(), allow_legacy_snapshot=invalid_flag),
        NeuralIntegrityError,
    )
