from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from neurocraft_protocol import (
    EventCounter,
    EventSample,
    Observation,
    ProtocolError,
    SensorFrameV2,
    SensorFrameV3,
    canonical_json_bytes,
    encode_message,
    parse_message,
    require_sensor_contract,
)
from neurocraft_protocol.sensor_contract import (
    LEGACY_OBSERVATION_SCHEMA_SHA256,
    OBSERVATION_SCHEMA_V2_SHA256,
    OBSERVATION_SCHEMA_V3_SHA256,
)

from tests.protocol.helpers import observation, sensors


def _frame() -> Observation:
    base = replace(observation(), game_tick=420)
    sample = SensorFrameV3.from_typed_frame(
        SensorFrameV2.from_sensor_frame(sensors(), (0, 1, 2, 0, 1, 2)),
        EventSample(True, True, EventCounter(3, 419), EventCounter(1, 400), EventCounter(0, None)),
    )
    return replace(
        base,
        identity=replace(base.identity, observation_schema_hash=OBSERVATION_SCHEMA_V3_SHA256),
        sensors=sample,
    )


def test_event_observation_round_trip_preserves_typed_rays_and_exact_counters() -> None:
    value = _frame()
    assert parse_message(encode_message(value)) == value
    assert canonical_json_bytes(Observation.from_dict(value.to_dict())) == canonical_json_bytes(
        value
    )
    assert isinstance(value.sensors, SensorFrameV3)
    assert value.sensors.ray_hit_kinds == (0, 1, 2, 0, 1, 2)
    with pytest.raises(FrozenInstanceError):
        value.sensors.events.food.count = 4  # type: ignore[misc]


@pytest.mark.parametrize(
    ("count", "tick"),
    [
        (True, 1),
        (1.0, 1),
        (-1, 1),
        (2**63, 1),
        (0, 0),
        (1, None),
        (1, -1),
        (1, True),
        (1, 1.0),
        (1, 2**63),
    ],
)
def test_event_counter_refuses_ambiguous_or_unrepresentable_values(
    count: object, tick: object
) -> None:
    with pytest.raises(ProtocolError):
        EventCounter.from_dict({"count": count, "last_game_tick": tick})


@pytest.mark.parametrize("counter_name", EventSample.counter_names())
def test_any_event_from_after_the_observation_is_refused(counter_name: str) -> None:
    value = _frame().to_dict()
    value["sensors"]["events"][counter_name] = {"count": 1, "last_game_tick": 421}
    with pytest.raises(ProtocolError, match="after the observation"):
        parse_message(json.dumps(value))


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_events",
        "missing_food",
        "unknown_event",
        "counter_extra",
        "counter_missing",
        "numeric_contact",
        "unknown_schema",
        "v2_discriminator",
        "missing_discriminator",
        "no_kinds",
    ],
)
def test_event_shape_cannot_be_guessed_or_downgraded(mutation: str) -> None:
    raw = copy.deepcopy(_frame().to_dict())
    sample = raw["sensors"]
    if mutation == "missing_events":
        del sample["events"]
    elif mutation == "missing_food":
        del sample["events"]["food"]
    elif mutation == "unknown_event":
        sample["events"]["reward"] = 1
    elif mutation == "counter_extra":
        sample["events"]["food"]["reward"] = 1
    elif mutation == "counter_missing":
        del sample["events"]["brush"]["last_game_tick"]
    elif mutation == "numeric_contact":
        sample["events"]["food_contact"] = 1
    elif mutation == "unknown_schema":
        sample["sensor_schema"] = "neurocraft.sensors.v4"
    elif mutation == "v2_discriminator":
        sample["sensor_schema"] = "neurocraft.sensors.v2"
    elif mutation == "missing_discriminator":
        del sample["sensor_schema"]
    else:
        del sample["ray_hit_kinds"]
    with pytest.raises(ProtocolError):
        parse_message(json.dumps(raw))


def test_each_identity_requires_its_own_sensor_shape() -> None:
    shapes = [sensors(), SensorFrameV2.from_sensor_frame(sensors(), (2,) * 6), _frame().sensors]
    identities = [
        LEGACY_OBSERVATION_SCHEMA_SHA256,
        OBSERVATION_SCHEMA_V2_SHA256,
        OBSERVATION_SCHEMA_V3_SHA256,
    ]
    assert len(set(identities)) == 3
    for index, shape in enumerate(shapes):
        for other_index, identity in enumerate(identities):
            if index == other_index:
                require_sensor_contract(shape, identity)
            else:
                with pytest.raises(ProtocolError, match="session requires"):
                    require_sensor_contract(shape, identity)


def test_large_counters_are_integers_and_coalesced_history_is_not_fabricated() -> None:
    base = _frame()
    assert isinstance(base.sensors, SensorFrameV3)
    events = replace(base.sensors.events, brush=EventCounter(2**63 - 1, base.game_tick))
    frame = replace(
        base, sensors=replace(base.sensors, events=events), bridge_observations_coalesced=9
    )
    assert parse_message(encode_message(frame)) == frame
    assert frame.sensors.events.brush.count == 2**63 - 1
    # Transport carries only a cumulative count and latest tick, never invented intermediate events.
    assert set(frame.to_dict()["sensors"]["events"]["brush"]) == {"count", "last_game_tick"}


def test_old_observation_goldens_stay_byte_exact() -> None:
    root = Path(__file__).resolve().parents[2]
    for filename in ("observation.json", "observation_v2.json"):
        raw = json.loads((root / "packages/protocol/golden" / filename).read_text())
        assert canonical_json_bytes(Observation.from_dict(raw)) == canonical_json_bytes(raw)
        assert "events" not in raw["sensors"]


def test_published_json_schema_accepts_all_sensor_goldens_and_the_full_v3_flag_set() -> None:
    root = Path(__file__).resolve().parents[2]
    schema = json.loads((root / "packages/protocol/schema/protocol-v2.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    for filename in ("observation.json", "observation_v2.json", "observation_v3.json"):
        raw = json.loads((root / "packages/protocol/golden" / filename).read_text())
        validator.validate(raw)
        assert canonical_json_bytes(Observation.from_dict(raw)) == canonical_json_bytes(raw)
    frame = _frame()
    flagged = replace(frame, clamped_fields=tuple(sorted(frame.sensors.FIELDS - {"sensor_schema"})))
    assert len(flagged.clamped_fields) == 13
    validator.validate(flagged.to_dict())


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_events",
        "unknown_event",
        "unknown_counter",
        "zero_with_tick",
        "no_tick",
        "negative_count",
        "counter_overflow",
        "numeric_contact",
        "schema_downgrade",
    ],
)
def test_published_event_json_schema_rejects_bad_shapes_and_count_tick_pairs(mutation: str) -> None:
    root = Path(__file__).resolve().parents[2]
    schema = json.loads((root / "packages/protocol/schema/protocol-v2.schema.json").read_text())
    raw = _frame().to_dict()
    event = raw["sensors"]["events"]
    if mutation == "missing_events":
        del raw["sensors"]["events"]
    elif mutation == "unknown_event":
        event["goal"] = "food"
    elif mutation == "unknown_counter":
        event["food"]["goal"] = "food"
    elif mutation == "zero_with_tick":
        event["food"] = {"count": 0, "last_game_tick": 1}
    elif mutation == "no_tick":
        event["food"] = {"count": 1, "last_game_tick": None}
    elif mutation == "negative_count":
        event["food"]["count"] = -1
    elif mutation == "counter_overflow":
        event["food"]["count"] = 2**63
    elif mutation == "numeric_contact":
        event["food_contact"] = 1
    else:
        raw["sensors"]["sensor_schema"] = "neurocraft.sensors.v2"
    assert not Draft202012Validator(schema).is_valid(raw)


@pytest.mark.parametrize("near", [False, 1, None])
def test_contact_requires_a_boolean_near_signal_in_both_parser_and_schema(near: object) -> None:
    root = Path(__file__).resolve().parents[2]
    schema = json.loads((root / "packages/protocol/schema/protocol-v2.schema.json").read_text())
    raw = _frame().to_dict()
    raw["sensors"]["events"]["food_near"] = near
    with pytest.raises(ProtocolError):
        parse_message(json.dumps(raw))
    assert not Draft202012Validator(schema).is_valid(raw)


def test_near_food_does_not_imply_ground_contact() -> None:
    frame = _frame()
    assert isinstance(frame.sensors, SensorFrameV3)
    airborne_offer = replace(frame.sensors.events, food_contact=False)
    changed = replace(frame, sensors=replace(frame.sensors, events=airborne_offer))
    assert parse_message(encode_message(changed)) == changed
    assert changed.sensors.events.food_near
    assert not changed.sensors.events.food_contact
