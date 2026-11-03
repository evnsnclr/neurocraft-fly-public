"""Sensor transport revisions within protocol v2.

The old observation hash names the frozen four-feature boundary, not the
whole JSON object. Existing hashes retain the legacy transport. Only the
new, explicitly declared contract selects typed rays; a peer cannot upgrade
or downgrade a session by changing its frame's discriminator.
"""

from __future__ import annotations

import hashlib
from typing import Any, Final

from .canonical import canonical_json_bytes

SENSOR_SCHEMA_V1: Final = "neurocraft.sensors.v1"
SENSOR_SCHEMA_V2: Final = "neurocraft.sensors.v2"
SENSOR_SCHEMA_V3: Final = "neurocraft.sensors.v3"
SENSOR_SCHEMA_V4: Final = "neurocraft.sensors.v4"
LEGACY_OBSERVATION_SCHEMA_SHA256: Final = (
    "fbfdd38a650f1496dbf946eea7ab686f039f271e5250139259a3931f95335356"
)
SENSOR_V2_DECLARATION: Final[dict[str, Any]] = {
    "schema": "neurocraft.observation.contract.v2",
    "sensor_schema": SENSOR_SCHEMA_V2,
    "base_feature_schema_sha256": LEGACY_OBSERVATION_SCHEMA_SHA256,
    "ray_order": ["forward", "back", "left", "right", "up", "down"],
    "ray_hit_kinds": {
        "length": 6,
        "type": "integer",
        "values": {"0": "none", "1": "block", "2": "entity"},
        "semantics": "kind of the nearest hit producing the corresponding ray distance",
        "tie": "entity wins an exact block/entity distance tie",
        "no_hit": "none only when both sweeps miss; a hit at maximum range retains its kind",
        "predicate": "ADR 0017 obstructsSensorRay, zero entity hitbox inflation",
    },
}
OBSERVATION_SCHEMA_V2_SHA256: Final = hashlib.sha256(
    canonical_json_bytes(SENSOR_V2_DECLARATION)
).hexdigest()

SENSOR_V3_DECLARATION: Final[dict[str, Any]] = {
    "schema": "neurocraft.observation.contract.v3",
    "sensor_schema": SENSOR_SCHEMA_V3,
    "base_observation_schema_sha256": OBSERVATION_SCHEMA_V2_SHA256,
    "events": {
        "keys": ["food_contact", "food_near", "food", "brush", "attack_attempt"],
        "food_contact": "boolean; exact current scripted FeedProgram.canEnter predicate",
        "food_near": (
            "boolean; nearest visible whitelisted dropped item within 3.0 blocks inclusive, "
            "then absolute body bearing <=60 degrees; food_contact implies food_near"
        ),
        "counter": {
            "keys": ["count", "last_game_tick"],
            "count": "integer 0..9223372036854775807; cumulative per entity",
            "last_game_tick": "null iff count=0; otherwise integer 0..observation.game_tick",
            "overflow": "refuse further increments without wrapping or crashing the server",
            "neural_reset": "counters persist; encoder initializes from first observed counters",
            "coalescing": "latest counter and latest event tick retained; not an event queue",
        },
        "food": {
            "event": "false-to-true food_near sampled once per authoritative game tick",
            "contact_support": (
                "complete sturdy five-point footprint and current ground contact; no fluid"
            ),
            "items": [
                "minecraft:apple",
                "minecraft:glow_berries",
                "minecraft:sugar",
                "minecraft:sweet_berries",
            ],
            "distance": (
                "bounding-box centre distance; near radius 3.0 blocks, "
                "contact radius 0.75, inclusive"
            ),
            "visibility": (
                "collider line of sight, fluid ignored; UUID ordering on exact distance ties"
            ),
            "bearing": "nearest visible item first, then absolute body bearing <=60 degrees",
            "world_effect": "no consumption, digestion, reward, health or energy change",
        },
        "brush": (
            "live non-spectator server player uses minecraft:brush on pickable fly "
            "outside water/lava"
        ),
        "interaction_predicate": (
            "player and unremoved fly in the same server level; brush uses invoked hand, "
            "attack origin is DamageSource.getEntity including player-owned projectiles"
        ),
        "interaction_effects": (
            "counters only; brush returns SUCCESS_SERVER when incremented, "
            "no item consumption/durability or health/velocity change"
        ),
        "attack_attempt": (
            "finite positive live non-spectator server player-origin hurt request; "
            "hurtServer remains false"
        ),
        "scope": "engineered world interactions; no fly health, damage or nociception model",
    },
}
OBSERVATION_SCHEMA_V3_SHA256: Final = hashlib.sha256(
    canonical_json_bytes(SENSOR_V3_DECLARATION)
).hexdigest()

SENSOR_V4_DECLARATION: Final[dict[str, Any]] = {
    "schema": "neurocraft.observation.contract.v4",
    "sensor_schema": SENSOR_SCHEMA_V4,
    "base_observation_schema_sha256": OBSERVATION_SCHEMA_V3_SHA256,
    "ray_actor_kinds": {
        "length": 6,
        "type": "integer",
        "values": {"0": "no entity hit", "1": "Player", "2": "Mob", "3": "other pickable entity"},
        "ray_order": ["forward", "back", "left", "right", "up", "down"],
        "semantics": (
            "classification of the same winning hit producing ray_hit_kinds and ray_distances_m"
        ),
        "consistency": "0 iff corresponding ray_hit_kind is not 2; entity hits require 1, 2 or 3",
        "classification": (
            "Player first, then Mob, then other entity admitted by obstructsSensorRay"
        ),
        "entity_tie": (
            "smallest UUID by Java UUID natural ordering wins exact entity/entity distance ties"
        ),
        "block_entity_tie": "entity wins an exact block/entity distance tie",
        "scope": (
            "engineered Minecraft class metadata; no identity, intention or biological recognition"
        ),
    },
}
OBSERVATION_SCHEMA_V4_SHA256: Final = hashlib.sha256(
    canonical_json_bytes(SENSOR_V4_DECLARATION)
).hexdigest()


def sensor_schema_for_identity(observation_schema_hash: str) -> str:
    """Select from a trusted identity; undeclared historical identities stay v1.

    Each declared hash opts into exactly one revision. A frame discriminator
    cannot upgrade or downgrade the authenticated session's transport.
    """

    if observation_schema_hash == OBSERVATION_SCHEMA_V4_SHA256:
        return SENSOR_SCHEMA_V4
    if observation_schema_hash == OBSERVATION_SCHEMA_V3_SHA256:
        return SENSOR_SCHEMA_V3
    return (
        SENSOR_SCHEMA_V2
        if observation_schema_hash == OBSERVATION_SCHEMA_V2_SHA256
        else SENSOR_SCHEMA_V1
    )
