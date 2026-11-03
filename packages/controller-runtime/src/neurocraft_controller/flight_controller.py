"""Opt-in wing-motor flight controller for the 128-node steering circuit.

This module exists so the readout `flight_pools` derived (D-035-compliant,
somaSide-split, 25+27 unit-weight members) can drive a controller **without
touching a single pilot-pinned byte**: fixture-physics pilot v3 pins the exact
bytes of `config.py` and `controller.py`, and the shipped default
`ControllerConfiguration` identity (`974085d1…24d5`) is pinned inside the
EXTERNAL-BRIDGE-02/-03/-04 trace manifests. Everything here is therefore a new
module, a new configuration schema, and an explicit opt-in — the same
new-module pattern `flight_pin.py` uses for the loader and D-043 used for the
FlyGym lane. Constructing this controller changes no default hash anywhere.

A-priori gain policy (AB-10): the pool gain and dead zone default to the
values the v1 boundary has pinned since before the steering fixture existed —
``gain = 8.0`` (also the hard adapter ceiling) and ``dead_zone = 0.0``. They
are declared here, ahead of any steering-circuit bridge execution, and must
not be re-chosen after observing steering behavior; a weak readout is a
reported result, not a tuning target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from neurocraft_connectome import RuntimeGraph
from neurocraft_neural.adapters import MotorPool
from neurocraft_neural.model import TopologyOverlay, WeightOverlay

from .config import ControllerConfiguration
from .controller import ConnectomeController, ControllerIntegrityError
from .flight_pools import (
    LEFT_WING_MOTOR_BODY_IDS,
    RIGHT_WING_MOTOR_BODY_IDS,
    WING_MOTOR_POOL_PROFILE,
    WING_MOTOR_POOL_RULE,
    derive_wing_motor_pools,
)

FLIGHT_CONFIG_SCHEMA = "neurocraft.controller.flight_config.v1"

#: The v1 default configuration identity that the EXTERNAL-BRIDGE-02/-03/-04
#: trace manifests pin. A regression test asserts the default stays exactly
#: this; the flight configuration below must never collide with it.
DEFAULT_V1_CONFIGURATION_SHA256 = "974085d1ab3acf7fab3bd31ae349f597cae29d1c11af634a8942679bcfa224d5"


@dataclass(frozen=True, slots=True)
class FlightControllerConfiguration(ControllerConfiguration):
    """The wing-motor boundary configuration, under its own schema.

    The inherited ``left_motor_body_id``/``right_motor_body_id`` fields are
    **inert** in this profile — the readout is the derived 52-member somaSide
    split, not any single pair — so they are frozen at the v1 defaults, refused
    if customized, and deliberately excluded from the identity material so the
    configuration hash cannot imply a boundary this controller does not read.
    """

    schema: str = FLIGHT_CONFIG_SCHEMA
    motor_pool_profile: str = WING_MOTOR_POOL_PROFILE

    def __post_init__(self) -> None:
        if self.schema != FLIGHT_CONFIG_SCHEMA:
            raise ValueError("unsupported flight controller configuration schema")
        if self.motor_pool_profile != WING_MOTOR_POOL_PROFILE:
            raise ValueError(
                f"flight configuration supports only the {WING_MOTOR_POOL_PROFILE!r} profile"
            )
        if self.mode not in {"faithful", "lesion"}:
            raise ValueError("flight controller mode must be faithful or lesion")
        if self.sign_policy not in {"coarse_presynaptic_consensus", "unknown_zero"}:
            raise ValueError("unsupported sign policy")
        if self.neural_steps_per_game_tick != 5:
            raise ValueError("the v1 clock contract requires exactly five neural steps per tick")
        wing_members = set(LEFT_WING_MOTOR_BODY_IDS) | set(RIGHT_WING_MOTOR_BODY_IDS)
        for name in ("left_sensor_body_id", "right_sensor_body_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive source body ID")
            # The parent's sensor-vs-motor disjointness rule, restated against
            # the real readout: a sensor cell may not simultaneously be a
            # wing-motor pool member.
            if value in wing_members:
                raise ValueError(f"{name} is a wing-motor pool member and cannot be a sensor")
        if self.left_sensor_body_id == self.right_sensor_body_id:
            raise ValueError("sensor body IDs must be distinct")
        defaults = ControllerConfiguration()
        inert_ok = all(
            not isinstance(value, bool) and isinstance(value, int) and value == expected
            for value, expected in (
                (self.left_motor_body_id, defaults.left_motor_body_id),
                (self.right_motor_body_id, defaults.right_motor_body_id),
            )
        )
        if not inert_ok:
            raise ValueError(
                "the single motor body IDs are inert under the wing-motor profile; "
                "customizing them would imply a boundary this controller does not read"
            )
        if not math.isfinite(self.motor_pool_gain) or not 0.0 <= self.motor_pool_gain <= 8.0:
            raise ValueError("motor_pool_gain must be finite and in [0, 8]")
        if not math.isfinite(self.motor_dead_zone) or not 0.0 <= self.motor_dead_zone < 1.0:
            raise ValueError("motor_dead_zone must be finite and in [0, 1)")

    def to_identity_material(self) -> dict[str, Any]:
        return {
            "boundary": {
                "left_sensor_body_id": self.left_sensor_body_id,
                "right_sensor_body_id": self.right_sensor_body_id,
            },
            "mode": self.mode,
            "motor_dead_zone": float(self.motor_dead_zone).hex(),
            "motor_pool_gain": float(self.motor_pool_gain).hex(),
            "motor_pool_profile": self.motor_pool_profile,
            "motor_pool_rule": WING_MOTOR_POOL_RULE,
            "neural_steps_per_game_tick": self.neural_steps_per_game_tick,
            "schema": self.schema,
            "sign_policy": self.sign_policy,
        }


class FlightConnectomeController(ConnectomeController):
    """The unchanged controller with the derived wing-motor pools as readout.

    Only the motor pools differ from the parent: sensory adapter, dynamics,
    sign policy, projection, clocking, and every validation rule are inherited
    byte-for-byte. Construction refuses a plain v1 configuration so a trace can
    never carry the v1 configuration identity over the wing readout.
    """

    def __init__(
        self,
        graph: RuntimeGraph,
        *,
        run_id: str,
        entity_id: str,
        session_epoch: int = 0,
        configuration: ControllerConfiguration | None = None,
        weight_overlay: WeightOverlay | None = None,
        topology_overlay: TopologyOverlay | None = None,
    ) -> None:
        config = configuration or FlightControllerConfiguration()
        if not isinstance(config, FlightControllerConfiguration):
            raise ControllerIntegrityError(
                "the flight controller requires a FlightControllerConfiguration; "
                "a v1 configuration would misdescribe the wing-motor readout"
            )
        super().__init__(
            graph,
            run_id=run_id,
            entity_id=entity_id,
            session_epoch=session_epoch,
            configuration=config,
            weight_overlay=weight_overlay,
            topology_overlay=topology_overlay,
        )

    # The parent declares _motor_pools as a staticmethod over the configuration
    # alone; this profile's pools are re-derived from the graph, which the
    # parent has already bound to ``self.graph`` before the decoder is built,
    # so the override is an instance method by necessity.
    def _motor_pools(  # type: ignore[override]
        self,
        configuration: ControllerConfiguration,
    ) -> tuple[MotorPool, ...]:
        return derive_wing_motor_pools(
            self.graph,
            gain=configuration.motor_pool_gain,
            dead_zone=configuration.motor_dead_zone,
        )
