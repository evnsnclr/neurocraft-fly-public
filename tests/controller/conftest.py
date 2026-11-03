from __future__ import annotations

from pathlib import Path

import pytest
from neurocraft_connectome import RuntimeGraph, load_malecns_fixture
from neurocraft_controller import ConnectomeController, ControllerConfiguration
from neurocraft_protocol import Observation, SensorFrame

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def controller_graph() -> RuntimeGraph:
    return load_malecns_fixture(REPO_ROOT / "data/fixtures/mcns-v1.0-flight-power-v1.json")


def make_controller(
    graph: RuntimeGraph,
    *,
    mode: str = "faithful",
    run_id: str = "controller-test",
) -> ConnectomeController:
    return ConnectomeController.create(
        graph,
        run_id=run_id,
        entity_id="test-fly",
        configuration=ControllerConfiguration(mode=mode),
    )


def make_sensors(
    *,
    forward: float = 6.0,
    left: float = 6.0,
    right: float = 6.0,
    light: int = 15,
) -> SensorFrame:
    return SensorFrame(
        position_m=(0.0, 64.0, 0.0),
        velocity_m_s=(0.0, 0.0, 0.0),
        yaw_rad=0.0,
        pitch_rad=0.0,
        on_ground=False,
        horizontal_collision=False,
        vertical_collision=False,
        in_fluid=False,
        local_light=light,
        ray_max_distance_m=6.0,
        ray_distances_m=(forward, 6.0, left, right, 6.0, 6.0),
    )


def make_observation(
    controller: ConnectomeController,
    sequence: int,
    *,
    sensors: SensorFrame | None = None,
    missing_fields: tuple[str, ...] = (),
) -> Observation:
    return Observation(
        identity=controller.identity,
        observation_seq=sequence,
        game_tick=sequence,
        sent_monotonic_ns=sequence * 50_000_000,
        sensors=sensors or make_sensors(),
        missing_fields=missing_fields,
    )
