"""Versioned, deliberately low-capacity controller assumptions."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from neurocraft_protocol import canonical_json_bytes


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class ControllerConfiguration:
    """All manually selected boundary assumptions used by the reference demo.

    The selected sensory cells are not a homologous bilateral pair. The mapping is
    an engineering adapter for the bounded G0 fixture, not a biological claim.
    Likewise, the motor projection is a fixed two-channel linear readout rather
    than an inferred Drosophila motor program.
    """

    schema: str = "neurocraft.controller.config.v1"
    mode: str = "faithful"
    sign_policy: str = "coarse_presynaptic_consensus"
    neural_steps_per_game_tick: int = 5
    left_sensor_body_id: int = 805_722
    right_sensor_body_id: int = 77_185
    left_motor_body_id: int = 800_241
    right_motor_body_id: int = 800_917
    motor_pool_gain: float = 8.0
    motor_dead_zone: float = 0.0

    def __post_init__(self) -> None:
        if self.schema != "neurocraft.controller.config.v1":
            raise ValueError("unsupported controller configuration schema")
        if self.mode not in {"faithful", "lesion", "synthetic"}:
            raise ValueError("controller mode must be faithful, lesion, or synthetic")
        if self.sign_policy not in {"coarse_presynaptic_consensus", "unknown_zero"}:
            raise ValueError("unsupported sign policy")
        if self.neural_steps_per_game_tick != 5:
            raise ValueError("the v1 clock contract requires exactly five neural steps per tick")
        for name in (
            "left_sensor_body_id",
            "right_sensor_body_id",
            "left_motor_body_id",
            "right_motor_body_id",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive source body ID")
        if (
            len(
                {
                    self.left_sensor_body_id,
                    self.right_sensor_body_id,
                    self.left_motor_body_id,
                    self.right_motor_body_id,
                }
            )
            != 4
        ):
            raise ValueError("controller boundary body IDs must be distinct")
        if not math.isfinite(self.motor_pool_gain) or not 0.0 <= self.motor_pool_gain <= 8.0:
            raise ValueError("motor_pool_gain must be finite and in [0, 8]")
        if not math.isfinite(self.motor_dead_zone) or not 0.0 <= self.motor_dead_zone < 1.0:
            raise ValueError("motor_dead_zone must be finite and in [0, 1)")

    def to_identity_material(self) -> dict[str, Any]:
        return {
            "boundary": {
                "left_motor_body_id": self.left_motor_body_id,
                "left_sensor_body_id": self.left_sensor_body_id,
                "right_motor_body_id": self.right_motor_body_id,
                "right_sensor_body_id": self.right_sensor_body_id,
            },
            "mode": self.mode,
            "motor_dead_zone": float(self.motor_dead_zone).hex(),
            "motor_pool_gain": float(self.motor_pool_gain).hex(),
            "neural_steps_per_game_tick": self.neural_steps_per_game_tick,
            "schema": self.schema,
            "sign_policy": self.sign_policy,
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.to_identity_material())


DEFAULT_FIXTURE_CONFIGURATION = ControllerConfiguration()
