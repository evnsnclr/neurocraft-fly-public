"""Physical-only Minecraft observation boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar

from neurocraft_protocol import SensorFrame
from neurocraft_protocol.sensor_contract import OBSERVATION_SCHEMA_V2_SHA256

from .config import canonical_sha256


@dataclass(frozen=True, slots=True)
class EncodedFeatures:
    values: Mapping[str, float]
    identity_sha256: str


class PhysicalFeatureEncoder:
    """Normalize four raw scalar observations without semantic policy logic.

    Ray order in protocol v1 is forward, back, left, right, up, down. The
    encoder uses forward/left/right distances and local light. It performs only
    division by the declared physical range; it does not turn, seek, avoid, or
    inspect targets/categories.
    """

    SCHEMA_MATERIAL: ClassVar[dict[str, Any]] = {
        "channels": [
            {
                "name": "forward_clearance_fraction",
                "source": "sensors.ray_distances_m[0]",
                "transform": "x / sensors.ray_max_distance_m",
                "unit": "fraction",
            },
            {
                "name": "left_clearance_fraction",
                "source": "sensors.ray_distances_m[2]",
                "transform": "x / sensors.ray_max_distance_m",
                "unit": "fraction",
            },
            {
                "name": "right_clearance_fraction",
                "source": "sensors.ray_distances_m[3]",
                "transform": "x / sensors.ray_max_distance_m",
                "unit": "fraction",
            },
            {
                "name": "local_light_fraction",
                "source": "sensors.local_light",
                "transform": "x / 15",
                "unit": "fraction",
            },
        ],
        "ray_order": ["forward", "back", "left", "right", "up", "down"],
        "schema": "neurocraft.observation.features.v1",
    }

    def __init__(self) -> None:
        self._identity = canonical_sha256(self.SCHEMA_MATERIAL)

    @property
    def identity_sha256(self) -> str:
        return self._identity

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(str(item["name"]) for item in self.SCHEMA_MATERIAL["channels"])

    def encode(self, sensors: SensorFrame) -> EncodedFeatures:
        if not isinstance(sensors, SensorFrame):
            raise TypeError("feature encoder accepts only a protocol SensorFrame")
        ray_max = sensors.ray_max_distance_m
        values = {
            "forward_clearance_fraction": sensors.ray_distances_m[0] / ray_max,
            "left_clearance_fraction": sensors.ray_distances_m[2] / ray_max,
            "right_clearance_fraction": sensors.ray_distances_m[3] / ray_max,
            "local_light_fraction": sensors.local_light / 15.0,
        }
        if set(values) != set(self.channel_names):
            raise RuntimeError("feature implementation/schema mismatch")
        if any(not 0.0 <= value <= 1.0 for value in values.values()):
            raise RuntimeError("normalized physical feature is outside [0, 1]")
        identity = canonical_sha256(
            {
                "schema_sha256": self._identity,
                "values": {name: float(value).hex() for name, value in sorted(values.items())},
            }
        )
        return EncodedFeatures(MappingProxyType(values), identity)


class EntityRayFeatureEncoder(PhysicalFeatureEncoder):
    """Same four feature values, bound to the additional typed-ray contract.

    The EEM encoder consumes the kinds separately. Keeping this subclass
    opt-in preserves every historical feature, adapter and observation hash.
    """

    def __init__(self) -> None:
        self._identity = OBSERVATION_SCHEMA_V2_SHA256
