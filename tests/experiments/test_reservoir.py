"""Trained-Readout Mode harness foundation: conditions, drive, ridge, identity.

Machinery tests only. No experiment is run here and no result is claimed: per
ADR 0008 the Trained-Readout Mode preregistration must be recorded before any
result is generated, and this suite exercises determinism and refusal behavior
on synthetic drives, not tasks.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from neurocraft_connectome import RuntimeGraph, load_malecns_fixture
from neurocraft_connectome.flight_pin import MALECNS_FLIGHT_FIXTURE_V1_PIN
from neurocraft_experiments.reservoir import (
    SUPPORTED_CONDITIONS,
    ReservoirConditionError,
    build_condition_engine,
    drive_and_collect,
    reservoir_run_identity,
    ridge_readout,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
STEERING_FIXTURE = REPO_ROOT / "data" / "fixtures" / "mcns-v1.0-flight-steering-v1.json"
STEERING_FIXTURE_BYTES = 284_949

#: The steering fixture's mechanosensory entries, driven in every synthetic test.
SENSORY_BODY_IDS = (77_185, 805_722)


@pytest.fixture(scope="module")
def graph() -> RuntimeGraph:
    return load_malecns_fixture(
        STEERING_FIXTURE,
        expected_pin=MALECNS_FLIGHT_FIXTURE_V1_PIN,
        expected_file_bytes=STEERING_FIXTURE_BYTES,
    )


def _pulse_drive(steps: int) -> tuple[dict[int, float] | None, ...]:
    """A declared deterministic stimulus: alternate driving the two entries."""

    drive: list[dict[int, float] | None] = []
    for step in range(steps):
        if step % 3 == 2:
            drive.append(None)
        else:
            drive.append({SENSORY_BODY_IDS[step % 2]: 0.5})
    return tuple(drive)


def test_every_supported_condition_builds_and_unknown_is_refused(
    graph: RuntimeGraph,
) -> None:
    identities = set()
    for condition in SUPPORTED_CONDITIONS:
        engine = build_condition_engine(graph, condition, seed=3, protocol_id="test-prereg")
        key = (
            engine.sign_overlay.identity_sha256,
            engine.weight_overlay.identity_sha256,
            engine.topology_overlay.identity_sha256,
        )
        identities.add(key)
    # Every condition must be distinguishable from every other by overlay identity.
    assert len(identities) == len(SUPPORTED_CONDITIONS)
    with pytest.raises(ReservoirConditionError, match="unknown reservoir condition"):
        build_condition_engine(graph, "cpg", seed=3, protocol_id="test-prereg")


def test_drive_and_collect_is_deterministic_and_full_width(graph: RuntimeGraph) -> None:
    drive = _pulse_drive(12)
    first = drive_and_collect(
        build_condition_engine(graph, "connectome", seed=3, protocol_id="p"),
        drive,
        protocol_id="p",
    )
    second = drive_and_collect(
        build_condition_engine(graph, "connectome", seed=3, protocol_id="p"),
        drive,
        protocol_id="p",
    )
    assert first == second
    assert len(first) == len(drive)
    assert all(len(row) == len(graph.nodes) for row in first)


def test_sign_ablation_with_no_drive_stays_silent(graph: RuntimeGraph) -> None:
    """All-zero signs kill recurrence; with no input and zero baseline the
    hidden state must remain exactly zero — the zero-recurrence control."""

    states = drive_and_collect(
        build_condition_engine(graph, "sign_ablated_zero", seed=3, protocol_id="p"),
        (None,) * 6,
        protocol_id="p",
    )
    assert all(value == 0.0 for row in states for value in row)


def test_conditions_diverge_under_identical_drive(graph: RuntimeGraph) -> None:
    drive = _pulse_drive(15)
    trajectories = {}
    for condition in ("connectome", "sign_permuted", "sign_ablated_zero"):
        trajectories[condition] = drive_and_collect(
            build_condition_engine(graph, condition, seed=3, protocol_id="p"),
            drive,
            protocol_id="p",
        )
    assert trajectories["connectome"] != trajectories["sign_permuted"]
    assert trajectories["connectome"] != trajectories["sign_ablated_zero"]


def test_ridge_readout_recovers_a_planted_linear_map() -> None:
    # Deterministic synthetic states: distinct, well-conditioned rows.
    states = tuple(tuple(((i * 7 + j * 3) % 11 - 5) / 5.0 for j in range(4)) for i in range(40))
    planted = ((0.5, -1.0), (0.25, 0.75), (-0.5, 0.0), (1.0, 0.5))
    targets = tuple(
        tuple(sum(state[k] * planted[k][out] for k in range(4)) for out in range(2))
        for state in states
    )
    weights = ridge_readout(states, targets, l2=1e-9)
    assert len(weights) == 4 and all(len(row) == 2 for row in weights)
    for k in range(4):
        for out in range(2):
            assert weights[k][out] == pytest.approx(planted[k][out], abs=1e-6)


def test_ridge_readout_refuses_mismatched_shapes() -> None:
    with pytest.raises(ReservoirConditionError, match="shape"):
        ridge_readout(((1.0, 2.0),), ((1.0,), (2.0,)), l2=1.0)
    with pytest.raises(ReservoirConditionError, match="l2"):
        ridge_readout(((1.0, 2.0),), ((1.0,),), l2=-1.0)


def test_run_identity_binds_condition_seed_and_drive(graph: RuntimeGraph) -> None:
    drive = _pulse_drive(6)
    base = reservoir_run_identity(
        build_condition_engine(graph, "connectome", seed=3, protocol_id="p"),
        condition="connectome",
        seed=3,
        protocol_id="p",
        drive=drive,
    )
    assert len(base) == 64
    other_seed = reservoir_run_identity(
        build_condition_engine(graph, "connectome", seed=4, protocol_id="p"),
        condition="connectome",
        seed=4,
        protocol_id="p",
        drive=drive,
    )
    other_drive = reservoir_run_identity(
        build_condition_engine(graph, "connectome", seed=3, protocol_id="p"),
        condition="connectome",
        seed=3,
        protocol_id="p",
        drive=_pulse_drive(7),
    )
    assert base != other_seed
    assert base != other_drive


def test_refusal_paths_are_pinned(graph: RuntimeGraph) -> None:
    """Every fail-closed branch stays fail-closed (adversarial-review gap)."""

    engine = build_condition_engine(graph, "connectome", seed=3, protocol_id="p")
    with pytest.raises(ReservoirConditionError, match="at least one step"):
        drive_and_collect(engine, (), protocol_id="p")
    with pytest.raises(ReservoirConditionError, match="bounded"):
        drive_and_collect(engine, (None,) * 100_001, protocol_id="p")
    with pytest.raises(ReservoirConditionError, match="protocol_id"):
        drive_and_collect(engine, (None,), protocol_id="")
    with pytest.raises(ReservoirConditionError, match="protocol_id"):
        build_condition_engine(graph, "connectome", seed=3, protocol_id="")
    with pytest.raises(ReservoirConditionError, match="unknown reservoir condition"):
        reservoir_run_identity(engine, condition="cpg", seed=3, protocol_id="p", drive=(None,))
    with pytest.raises(ReservoirConditionError, match="l2"):
        ridge_readout(((1.0, 2.0),), ((1.0,),), l2=0.0)
