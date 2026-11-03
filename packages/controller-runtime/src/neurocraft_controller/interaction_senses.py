"""Candidate V4 inputs: brushing and attack attempts stimulate different populations.

Attack is an engineered tactile proxy on the existing bristle population, including
its unknown_sensory members. It is not nociception. The full controller builder and
the opt-in interactions-v2 service profile select it. Live acceptance is separate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from neurocraft_protocol import EventSample, SensorFrame, SensorFrameV4
from neurocraft_protocol.sensor_contract import OBSERVATION_SCHEMA_V4_SHA256
from numpy.typing import NDArray

from .config import canonical_sha256
from .embodied_senses import EmbodiedPopulationBinding
from .events_senses import (
    MAX_EVENT_AGE_TICKS,
    PULSE_AMPLITUDE,
    EventEncoderBinding,
    EventInputState,
    EventPopulationBinding,
    EventPulse,
    EventSensesError,
    _integer,
)
from .features import EncodedFeatures, EntityRayFeatureEncoder


@dataclass(frozen=True, slots=True)
class InteractionEncoderBinding(EventEncoderBinding):
    embodied_populations: EmbodiedPopulationBinding
    # Fixture callers explicitly declare their smaller synthetic membership.
    expected_bristle_count: int = 2206
    _identity_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        EventEncoderBinding.__post_init__(self)
        base = self.populations
        embodied = self.embodied_populations
        if not isinstance(base, EventPopulationBinding) or not isinstance(
            embodied, EmbodiedPopulationBinding
        ):
            raise EventSensesError("interaction requires immutable verified population bindings")
        if (
            base.graph_hash != embodied.graph_manifest_sha256
            or base.annotations_hash != embodied.annotations_sha256
            or base.node_count != embodied.node_count
        ):
            raise EventSensesError(
                "interaction populations must share the verified graph and annotations"
            )
        _integer(self.expected_bristle_count, minimum=1, maximum=base.node_count)
        bristles = embodied.population("bristle")
        if len(bristles.body_ids) != self.expected_bristle_count:
            raise EventSensesError("attack proxy bristle membership count drifted")
        if len(set(bristles.dense_indices)) != len(bristles.dense_indices):
            raise EventSensesError("attack proxy indices must be unique")
        for index in bristles.dense_indices:
            _integer(index, maximum=base.node_count - 1)
        if any(
            set(bristles.dense_indices).intersection(pop.indices)
            or set(bristles.body_ids).intersection(row[0] for row in pop.members)
            for pop in base.populations
        ):
            raise EventSensesError(
                "attack proxy must be disjoint from food/groom inputs and event readouts"
            )

        # Parent event memberships and the embodied binding have already copied
        # caller containers into immutable tuples and passed source checks.
        object.__setattr__(self, "_identity_sha256", canonical_sha256(self.to_identity_material()))

    @property
    def identity_sha256(self) -> str:
        return self._identity_sha256

    def to_identity_material(self) -> dict[str, Any]:
        bristles = self.embodied_populations.population("bristle")
        return {
            **EventEncoderBinding.to_identity_material(self),
            "schema": "neurocraft.interactions.encoder.v2",
            "observation_schema_sha256": OBSERVATION_SCHEMA_V4_SHA256,
            "embodied_population_binding_sha256": self.embodied_populations.identity_sha256,
            "attack_population": {
                "name": bristles.name,
                "selector": bristles.selector,
                "expected_count": self.expected_bristle_count,
                "members_body_dense": list(
                    zip(bristles.body_ids, bristles.dense_indices, strict=True)
                ),
            },
            "groom": "one pulse on fresh brush only; attack no longer drives grooming afferents",
            "attack": "one pulse on fresh player attack-attempt to each selected bristle member",
            "proxy": (
                "attack-attempt is an engineered bristle tactile proxy, not damage or nociception"
            ),
            "attack_overlap": (
                "same source population as existing collision/touch input; additions may sum"
            ),
            "simultaneous_events": (
                "one bounded pulse per active source population; counter jumps never amplify"
            ),
            "actor_metadata": (
                "declared Minecraft class gates for later readout selection; "
                "no neural recognition claim"
            ),
            "light": (
                "unchanged absolute loom-v2 light input; "
                "no additional pulse or temporal transformation"
            ),
        }


class InteractionFeatureEncoder(EntityRayFeatureEncoder):
    def __init__(self) -> None:
        super().__init__()
        self._identity = OBSERVATION_SCHEMA_V4_SHA256

    def encode(self, sensors: SensorFrame) -> EncodedFeatures:
        if not isinstance(sensors, SensorFrameV4):
            raise EventSensesError("interaction candidate requires SensorFrameV4")
        return super().encode(sensors)


@dataclass(frozen=True, slots=True)
class InteractionPulse(EventPulse):
    attack: bool
    attack_age: int | None


class InteractionInputState(EventInputState):
    def __init__(self, binding: InteractionEncoderBinding) -> None:
        super().__init__(binding)
        self.interaction_binding = binding

    def prepare(self, events: EventSample, game_tick: int) -> InteractionPulse:
        # The parent validates every monotonic counter before changing its baseline.
        old = super().prepare(events, game_tick)
        brush_age = (
            None if events.brush.last_game_tick is None else game_tick - events.brush.last_game_tick
        )
        attack_age = (
            None
            if events.attack_attempt.last_game_tick is None
            else game_tick - events.attack_attempt.last_game_tick
        )
        brush = old.brush_delta > 0 and brush_age is not None and brush_age <= MAX_EVENT_AGE_TICKS
        attack = (
            old.attack_delta > 0 and attack_age is not None and attack_age <= MAX_EVENT_AGE_TICKS
        )
        return InteractionPulse(
            old.food,
            brush,
            old.food_age,
            brush_age if brush else None,
            old.food_delta,
            old.brush_delta,
            old.attack_delta,
            old.baseline_absorbed,
            attack,
            attack_age if attack else None,
        )

    def apply(self, vector: NDArray[np.float64], pulse: EventPulse) -> None:
        if not isinstance(pulse, InteractionPulse):
            raise EventSensesError("interaction input requires its own separate attack pulse")
        super().apply(vector, pulse)
        if pulse.attack:
            indices = self.interaction_binding.embodied_populations.population(
                "bristle"
            ).dense_indices
            vector[list(indices)] += PULSE_AMPLITUDE
