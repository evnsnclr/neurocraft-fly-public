"""Entity-gated approach contrast, eem-loom-v2.

Only typed entity hits contribute raw threat proximity. The existing v1
contrast, initialization, touch, light and population placement stay intact.
This is a demo threat filter, not an LC4 model. A close entity crossing a ray
causes a proximity step; motion toward a standing entity also counts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Final

from neurocraft_protocol import SensorFrame, SensorFrameV2
from neurocraft_protocol.sensor_contract import OBSERVATION_SCHEMA_V2_SHA256, SENSOR_SCHEMA_V2

from .embodied_senses import (
    EemChannel,
    EmbodiedPopulationBinding,
    EmbodiedSensesBinding,
    EmbodiedSensesError,
    encode_channel_values,
)
from .loom_senses import LOOM_V1_CHANNELS, EemEncoderDocumentLoomV1

LOOM_V2_ENCODER_SCHEMA: Final = "neurocraft.eem.encoder.loom.v2"
THREAT_RAY_INDICES: Final = {"threat_left": 2, "threat_right": 3, "threat_front": 0}
LOOM_V2_CHANNELS: Final[tuple[EemChannel, ...]] = tuple(
    replace(
        channel,
        formula=(
            f"before the v1 contrast, raw_t is exactly 0.0 unless "
            f"sensors.ray_hit_kinds[{THREAT_RAY_INDICES[channel.name]}] == 2; "
            f"entity hits retain the following formula unchanged: {channel.formula}"
        ),
    )
    if channel.name in THREAT_RAY_INDICES
    else channel
    for channel in LOOM_V1_CHANNELS
)


@dataclass(frozen=True, slots=True)
class EemEncoderDocumentLoomV2(EemEncoderDocumentLoomV1):
    channels: tuple[EemChannel, ...] = LOOM_V2_CHANNELS

    def __post_init__(self) -> None:
        EemEncoderDocumentLoomV1.__post_init__(self)
        if self.channels != LOOM_V2_CHANNELS:
            raise EmbodiedSensesError("loom-v2 requires its declared entity-gated channels")

    def to_identity_material(self) -> dict[str, Any]:
        material = EemEncoderDocumentLoomV1.to_identity_material(self)
        material.update(
            schema=LOOM_V2_ENCODER_SCHEMA,
            sensor_schema=SENSOR_SCHEMA_V2,
            observation_schema_sha256=OBSERVATION_SCHEMA_V2_SHA256,
            threat_rationale=(
                "only the nearest hit's entity kind admits raw proximity to the inherited "
                "rectified contrast; block and none are exactly zero. A close entity crossing "
                "a ray causes a step and decay; motion toward a standing entity counts. "
                "Real LC4 responds to looming regardless of what looms; this filter is "
                "engineering about the demo's threat sense, not biology."
            ),
        )
        return material


def encode_entity_channel_values(sensors: SensorFrame) -> tuple[tuple[str, float], ...]:
    if not isinstance(sensors, SensorFrameV2):
        raise EmbodiedSensesError("loom-v2 requires neurocraft.sensors.v2 with typed rays")
    return tuple(
        (
            name,
            0.0
            if name in THREAT_RAY_INDICES and sensors.ray_hit_kinds[THREAT_RAY_INDICES[name]] != 2
            else value,
        )
        for name, value in encode_channel_values(sensors)
    )


def bind_embodied_senses_loom_v2(
    populations: EmbodiedPopulationBinding,
) -> EmbodiedSensesBinding:
    return EmbodiedSensesBinding(
        populations=populations,
        encoder=EemEncoderDocumentLoomV2(population_binding_sha256=populations.identity_sha256),
    )
