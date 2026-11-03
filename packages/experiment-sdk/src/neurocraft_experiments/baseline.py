"""Explicit direct scripted control used only as a labelled comparison."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _clip(value: float) -> float:
    return max(-1.0, min(1.0, value))


@dataclass(frozen=True, slots=True)
class RawPhysicalObservation:
    forward_clearance: float
    left_clearance: float
    right_clearance: float
    vertical_velocity: float
    on_ground: bool

    def __post_init__(self) -> None:
        for field_name in (
            "forward_clearance",
            "left_clearance",
            "right_clearance",
            "vertical_velocity",
        ):
            _finite(getattr(self, field_name), field_name)
        if min(self.forward_clearance, self.left_clearance, self.right_clearance) < 0:
            raise ValueError("clearances cannot be negative")


@dataclass(frozen=True, slots=True)
class PrimitiveAction:
    forward_thrust: float
    lateral_thrust: float
    vertical_thrust: float
    yaw_torque: float
    pitch_torque: float = 0.0

    def __post_init__(self) -> None:
        for field_name in (
            "forward_thrust",
            "lateral_thrust",
            "vertical_thrust",
            "yaw_torque",
            "pitch_torque",
        ):
            value = _finite(getattr(self, field_name), field_name)
            if value < -1.0 or value > 1.0:
                raise ValueError(f"{field_name} is outside [-1, 1]")


def scripted_baseline(observation: RawPhysicalObservation) -> PrimitiveAction:
    """Low-capacity non-neural comparator; never callable in faithful mode."""

    clearance_scale = 2.0
    forward = _clip(observation.forward_clearance / clearance_scale)
    asymmetry = observation.right_clearance - observation.left_clearance
    yaw = _clip(asymmetry / clearance_scale)
    lateral = _clip(asymmetry / (clearance_scale * 2.0))
    vertical = _clip((0.35 if observation.on_ground else 0.0) - 0.2 * observation.vertical_velocity)
    return PrimitiveAction(forward, lateral, vertical, yaw)
