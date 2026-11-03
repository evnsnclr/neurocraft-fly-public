"""R-27 regression: the pilot world's lateral rays follow ADR 0015's handedness.

Minecraft's basis is +X east, +Y up, +Z south, and a body's right is a 90-degree
clockwise turn from its heading viewed from above, which is the engine's own
``Direction.getClockWise()`` table: facing east the right is south, facing south
it is west, facing west it is north, facing north it is east. The pilot's actuation
already used that frame; until 2026-11-03 its sensing frame reported the right-hand
clearance in the ``left`` slot, so the scripted comparator steered toward the
obstructed side and the neural conditions' left/right channels received the
opposite sides.

The tracked light corridor (x 0..32, y 0..8, z 0..8, no force zone) supplies the
walls. The pilot's heading convention is ``forward = (cos yaw, sin yaw)``, so yaw 0
faces east and yaw pi/2 faces south.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from neurocraft_experiments import load_arena, pilot
from neurocraft_experiments.arena import ArenaSpec
from neurocraft_experiments.baseline import RawPhysicalObservation, scripted_baseline
from neurocraft_protocol import MotorVector

ROOT = Path(__file__).resolve().parents[2]
RAY_MAX = 16.0
CENTER = (16.0, 2.0, 4.0)

#: yaw -> (facing, the body's right in the arena plane), straight from the clockwise table.
HEADINGS: dict[float, tuple[str, tuple[float, float]]] = {
    0.0: ("east", (0.0, 1.0)),
    math.pi / 2: ("south", (-1.0, 0.0)),
    math.pi: ("west", (0.0, -1.0)),
    -math.pi / 2: ("north", (1.0, 0.0)),
}


def _arena() -> ArenaSpec:
    arena = load_arena(ROOT / "experiments/arenas/light-corridor-v1.json")
    assert arena.min_corner == (0.0, 0.0, 0.0) and arena.max_corner == (32.0, 8.0, 8.0)
    return arena


def _physics() -> pilot.PhysicsConfiguration:
    return pilot.PhysicsConfiguration(
        tick_seconds=0.05,
        tick_limit=10,
        ray_max_distance_m=RAY_MAX,
        horizontal_acceleration_m_s2=10.0,
        vertical_acceleration_m_s2=0.0,
        gravity_m_s2=0.0,
        velocity_retention=1.0,
        yaw_radians_per_tick=0.1,
        body_radius_m=0.25,
    )


def _world(x: float, z: float, yaw: float) -> pilot._World:
    return pilot._World(x=x, y=CENTER[1], z=z, vx=0.0, vy=0.0, vz=0.0, yaw=yaw)


def _near_right_wall(yaw: float) -> pilot._World:
    """One metre from the wall on the body's right; far from the wall on its left."""
    right_x, right_z = HEADINGS[yaw][1]
    return _world(CENTER[0] + 15.0 * right_x, CENTER[2] + 3.0 * right_z, yaw)


def _rays(world: pilot._World) -> tuple[float, ...]:
    return pilot._sensor_frame(world, _arena(), _physics()).ray_distances_m


def _expected_distance(world: pilot._World, direction: tuple[float, float]) -> float:
    """Independent oracle: distance along ``direction`` to the corridor's vertical walls."""
    candidates: list[float] = []
    for axis, position in ((0, world.x), (2, world.z)):
        component = direction[0] if axis == 0 else direction[1]
        if abs(component) < 1e-12:
            continue
        bound = _arena().max_corner[axis] if component > 0 else _arena().min_corner[axis]
        candidates.append((bound - position) / component)
    return min(RAY_MAX, min(distance for distance in candidates if distance >= 0.0))


@pytest.mark.parametrize("yaw", list(HEADINGS))
def test_right_ray_is_the_clockwise_wall_at_every_cardinal_heading(yaw: float) -> None:
    facing, _ = HEADINGS[yaw]
    _, _, left, right = _rays(_near_right_wall(yaw))[:4]
    assert right == pytest.approx(1.0), f"facing {facing}: slot 3 must be the body's right"
    assert left > right, f"facing {facing}: slot 2 must be the body's left"
    assert left in (pytest.approx(7.0), pytest.approx(RAY_MAX))


@pytest.mark.parametrize("yaw", list(HEADINGS))
def test_lateral_thrust_slides_toward_the_right_ray_on_the_same_world(yaw: float) -> None:
    facing, (right_x, right_z) = HEADINGS[yaw]
    world = _near_right_wall(yaw)
    before = _rays(world)[3]
    pilot._step_physics(world, MotorVector(lateral_thrust=1.0), _arena(), _physics())
    speed = math.hypot(world.vx, world.vz)
    assert speed > 0.0
    along_right = (world.vx * right_x + world.vz * right_z) / speed
    assert along_right == pytest.approx(1.0), f"facing {facing}: lateral thrust is not rightward"
    assert _rays(world)[3] < before, f"facing {facing}: sliding right must shorten the right ray"


@pytest.mark.parametrize("yaw", [0.7, -2.1, 2.9])
def test_oblique_headings_sense_and_actuate_in_one_frame(yaw: float) -> None:
    # The clockwise quarter turn of (x, z) viewed from +Y is (x, z) -> (-z, x).
    forward = (math.cos(yaw), math.sin(yaw))
    expected_right = (-forward[1], forward[0])
    world = _world(CENTER[0], CENTER[2], yaw)
    rays = _rays(world)
    assert rays[0] == pytest.approx(_expected_distance(world, forward))
    assert rays[3] == pytest.approx(_expected_distance(world, expected_right))
    assert rays[2] == pytest.approx(
        _expected_distance(world, (-expected_right[0], -expected_right[1]))
    )
    pilot._step_physics(world, MotorVector(lateral_thrust=1.0), _arena(), _physics())
    speed = math.hypot(world.vx, world.vz)
    assert (world.vx * expected_right[0] + world.vz * expected_right[1]) / speed == pytest.approx(
        1.0
    )


def test_scripted_baseline_turns_away_from_the_nearer_right_wall() -> None:
    forward, _, left, right = _rays(_near_right_wall(0.0))[:4]
    action = scripted_baseline(
        RawPhysicalObservation(
            forward_clearance=forward,
            left_clearance=left,
            right_clearance=right,
            vertical_velocity=0.0,
            on_ground=False,
        )
    )
    assert action.yaw_torque < 0.0 and action.lateral_thrust < 0.0
    world = _near_right_wall(0.0)
    pilot._step_physics(world, MotorVector(yaw_torque=action.yaw_torque), _arena(), _physics())
    assert world.yaw < 0.0, "negative torque turns east toward north, away from the south wall"
