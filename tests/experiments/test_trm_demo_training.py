"""TRM demo training: declared generator, pinned sensory path, expert labels."""

from __future__ import annotations

from pathlib import Path

import pytest
from neurocraft_connectome import RuntimeGraph, load_malecns_fixture
from neurocraft_connectome.flight_pin import MALECNS_FLIGHT_FIXTURE_V1_PIN
from neurocraft_experiments.trm_demo_training import (
    CONSTANT_SCENARIO_RAYS_V2,
    RAY_MAX_M,
    RAY_MIN_M,
    SCENARIO_COUNT_V2,
    TEST_SCENARIOS_V2,
    TICKS_PER_SCENARIO,
    TRAIN_SCENARIOS_V2,
    VALIDATION_SCENARIOS_V2,
    WALK_SCENARIO_COUNT_V2,
    WASHOUT_TICKS,
    WASHOUT_TICKS_V2,
    TrmDemoTrainingError,
    collect_scenarios,
    collect_scenarios_v2,
    expert_targets,
    quasi_static_mask_v2,
    ray_trajectory,
    ray_trajectory_v2,
    sensor_frame,
    write_artifact,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
STEERING_FIXTURE = REPO_ROOT / "data" / "fixtures" / "mcns-v1.0-flight-steering-v1.json"


@pytest.fixture(scope="module")
def graph() -> RuntimeGraph:
    return load_malecns_fixture(
        STEERING_FIXTURE,
        expected_pin=MALECNS_FLIGHT_FIXTURE_V1_PIN,
        expected_file_bytes=284_949,
    )


def test_trajectories_are_deterministic_bounded_and_branch_covering() -> None:
    first = ray_trajectory(3)
    assert first == ray_trajectory(3)
    assert first != ray_trajectory(4)
    assert len(first) == TICKS_PER_SCENARIO
    for rays in first:
        for value in rays:
            assert RAY_MIN_M <= value <= RAY_MAX_M
    # Across a handful of scenarios, every expert branch must be exercised:
    # blocked forward, left-blocked (yaw right), right-blocked (yaw left), open.
    saw_blocked_forward = saw_left_turn = saw_right_turn = saw_open = False
    for scenario in range(8):
        for ray0, ray2, ray3 in ray_trajectory(scenario):
            forward, _, yaw = expert_targets(ray0, ray2, ray3)
            if ray0 < 1.0:
                saw_blocked_forward = True
            if yaw > 0.1:
                saw_left_turn = True
            if yaw < -0.1:
                saw_right_turn = True
            if forward > 0.9:
                saw_open = True
    assert saw_blocked_forward and saw_left_turn and saw_right_turn and saw_open
    with pytest.raises(TrmDemoTrainingError, match="scenario"):
        ray_trajectory(-1)


def test_sensor_frames_are_valid_and_expert_vertical_is_exactly_zero() -> None:
    frame = sensor_frame(1.5, 4.0, 0.6)
    assert frame.ray_distances_m == (1.5, RAY_MAX_M, 4.0, 0.6, RAY_MAX_M, RAY_MAX_M)
    from neurocraft_experiments.baseline import RawPhysicalObservation, scripted_baseline

    primitive = scripted_baseline(
        RawPhysicalObservation(
            forward_clearance=1.5,
            left_clearance=4.0,
            right_clearance=0.6,
            vertical_velocity=0.0,
            on_ground=False,
        )
    )
    assert primitive.vertical_thrust == 0.0
    assert expert_targets(1.5, 4.0, 0.6) == (
        primitive.forward_thrust,
        primitive.lateral_thrust,
        primitive.yaw_torque,
    )


def test_collection_is_deterministic_and_shaped(graph: RuntimeGraph) -> None:
    states, targets, rays = collect_scenarios(graph, [0])
    again = collect_scenarios(graph, [0])
    assert (states, targets, rays) == again
    assert len(states) == TICKS_PER_SCENARIO - WASHOUT_TICKS
    assert all(len(row) == len(graph.nodes) for row in states)
    assert len(targets) == len(states) == len(rays)


def test_v2_walk_trajectories_are_deterministic_bounded_and_hold_freezing() -> None:
    first = ray_trajectory_v2(3)
    assert first == ray_trajectory_v2(3)
    assert first != ray_trajectory_v2(4)
    assert first != ray_trajectory(3)  # v2 streams derive from the v2 demo id
    assert len(first) == TICKS_PER_SCENARIO
    for rays in first:
        for value in rays:
            assert RAY_MIN_M <= value <= RAY_MAX_M
    # Hold segments freeze the rays: wherever two consecutive ticks are both
    # flagged quasi-static in a walk scenario, the rays must be identical.
    saw_hold = False
    for scenario in range(8):
        trajectory = ray_trajectory_v2(scenario)
        mask = quasi_static_mask_v2(scenario)
        assert len(mask) == TICKS_PER_SCENARIO
        for tick in range(1, TICKS_PER_SCENARIO):
            if mask[tick] and mask[tick - 1]:
                saw_hold = True
                assert trajectory[tick] == trajectory[tick - 1]
    assert saw_hold
    # Every expert branch must still be exercised across a handful of
    # v2 walk scenarios: blocked forward, left turn, right turn, open.
    saw_blocked_forward = saw_left_turn = saw_right_turn = saw_open = False
    for scenario in range(8):
        for ray0, ray2, ray3 in ray_trajectory_v2(scenario):
            forward, _, yaw = expert_targets(ray0, ray2, ray3)
            if ray0 < 1.0:
                saw_blocked_forward = True
            if yaw > 0.1:
                saw_left_turn = True
            if yaw < -0.1:
                saw_right_turn = True
            if forward > 0.9:
                saw_open = True
    assert saw_blocked_forward and saw_left_turn and saw_right_turn and saw_open


def test_v2_constant_scenarios_are_fixed_at_the_declared_rays() -> None:
    assert len(CONSTANT_SCENARIO_RAYS_V2) == SCENARIO_COUNT_V2 - WALK_SCENARIO_COUNT_V2
    for offset, declared in enumerate(CONSTANT_SCENARIO_RAYS_V2):
        scenario = WALK_SCENARIO_COUNT_V2 + offset
        trajectory = ray_trajectory_v2(scenario)
        assert len(trajectory) == TICKS_PER_SCENARIO
        assert all(rays == declared for rays in trajectory)
        assert all(quasi_static_mask_v2(scenario))
        for value in declared:
            assert RAY_MIN_M <= value <= RAY_MAX_M
    with pytest.raises(TrmDemoTrainingError, match="scenario"):
        ray_trajectory_v2(-1)
    with pytest.raises(TrmDemoTrainingError, match="scenario"):
        ray_trajectory_v2(SCENARIO_COUNT_V2)
    with pytest.raises(TrmDemoTrainingError, match="scenario"):
        quasi_static_mask_v2(SCENARIO_COUNT_V2)


def test_v2_split_is_declared_disjoint_and_complete() -> None:
    train = set(TRAIN_SCENARIOS_V2)
    validation = set(VALIDATION_SCENARIOS_V2)
    test = set(TEST_SCENARIOS_V2)
    assert len(TRAIN_SCENARIOS_V2) == 44
    assert len(VALIDATION_SCENARIOS_V2) == 12
    assert len(TEST_SCENARIOS_V2) == 12
    assert train | validation | test == set(range(SCENARIO_COUNT_V2))
    assert not (train & validation or train & test or validation & test)
    # Every split carries constant-scenario coverage at the declared indices.
    assert train & set(range(60, 68)) == {60, 61, 62, 63}
    assert validation & set(range(60, 68)) == {64, 65}
    assert test & set(range(60, 68)) == {66, 67}


def test_v2_collection_keeps_every_tick_and_flags_quasi_static(graph: RuntimeGraph) -> None:
    assert WASHOUT_TICKS_V2 == 0
    states, targets, rays, quasi = collect_scenarios_v2(graph, [0])
    again = collect_scenarios_v2(graph, [0])
    assert (states, targets, rays, quasi) == again
    assert len(states) == TICKS_PER_SCENARIO  # cold-start ticks are kept
    assert all(len(row) == len(graph.nodes) for row in states)
    assert len(targets) == len(states) == len(rays) == len(quasi)
    assert quasi == quasi_static_mask_v2(0)
    constant_states, _, constant_rays, constant_quasi = collect_scenarios_v2(
        graph, [WALK_SCENARIO_COUNT_V2]
    )
    assert len(constant_states) == TICKS_PER_SCENARIO
    assert all(constant_quasi)
    assert set(constant_rays) == {CONSTANT_SCENARIO_RAYS_V2[0]}


def test_artifact_write_is_once_only(tmp_path: Path) -> None:
    target = tmp_path / "readout.json"
    digest = write_artifact(target, {"schema": "test", "x": 1})
    assert len(digest) == 64 and target.exists()
    with pytest.raises(TrmDemoTrainingError, match="immutable"):
        write_artifact(target, {"schema": "test", "x": 2})
