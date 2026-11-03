from __future__ import annotations

import hashlib
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
    SensorFrameV4,
    canonical_json_bytes,
    encode_message,
    parse_message,
    require_sensor_contract,
)
from neurocraft_protocol.sensor_contract import (
    LEGACY_OBSERVATION_SCHEMA_SHA256,
    OBSERVATION_SCHEMA_V2_SHA256,
    OBSERVATION_SCHEMA_V3_SHA256,
    OBSERVATION_SCHEMA_V4_SHA256,
    SENSOR_SCHEMA_V4,
    SENSOR_V4_DECLARATION,
    sensor_schema_for_identity,
)

from tests.protocol.helpers import observation, sensors

ROOT = Path(__file__).resolve().parents[2]


def _event_frame() -> SensorFrameV3:
    return SensorFrameV3.from_typed_frame(
        SensorFrameV2.from_sensor_frame(sensors(), (0, 1, 2, 2, 2, 0)),
        EventSample(True, True, EventCounter(3, 419), EventCounter(1, 400), EventCounter(0, None)),
    )


def _frame() -> Observation:
    base = replace(observation(), game_tick=420)
    return replace(
        base,
        identity=replace(base.identity, observation_schema_hash=OBSERVATION_SCHEMA_V4_SHA256),
        sensors=SensorFrameV4.from_event_frame(_event_frame(), (0, 0, 1, 2, 3, 0)),
    )


def _validator() -> Draft202012Validator:
    schema = json.loads((ROOT / "packages/protocol/schema/protocol-v2.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_actor_observation_round_trip_preserves_events_light_and_winning_hit_metadata() -> None:
    frame = _frame()
    assert parse_message(encode_message(frame)) == frame
    assert canonical_json_bytes(Observation.from_dict(frame.to_dict())) == canonical_json_bytes(
        frame
    )
    assert isinstance(frame.sensors, SensorFrameV4)
    assert frame.sensors.ray_actor_kinds == (0, 0, 1, 2, 3, 0)
    original = _event_frame().to_dict()
    actual = frame.sensors.to_dict()
    for key, value in original.items():
        if key != "sensor_schema":
            assert actual[key] == value
    assert actual["sensor_schema"] == SENSOR_SCHEMA_V4
    _validator().validate(frame.to_dict())
    with pytest.raises(FrozenInstanceError):
        frame.sensors.ray_actor_kinds = (0,) * 6  # type: ignore[misc]


@pytest.mark.parametrize("bad", [None, "000000", {}, (), (0,) * 5, (0,) * 7])
def test_actor_vector_requires_exactly_six_values(bad: object) -> None:
    with pytest.raises(ProtocolError, match="exactly six integers"):
        SensorFrameV4.from_event_frame(_event_frame(), bad)
    raw = _frame().to_dict()
    raw["sensors"]["ray_actor_kinds"] = bad
    with pytest.raises(ProtocolError):
        parse_message(json.dumps(raw))
    assert not _validator().is_valid(raw)


@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize("bad", [True, False, -1, 4, None, "1", 0.5])
def test_each_actor_value_refuses_bad_type_or_range(index: int, bad: object) -> None:
    raw = _frame().to_dict()
    raw["sensors"]["ray_actor_kinds"][index] = bad
    with pytest.raises(ProtocolError):
        parse_message(json.dumps(raw))
    assert not _validator().is_valid(raw)


@pytest.mark.parametrize("index", range(6))
def test_parser_refuses_integral_float_actor_values(index: int) -> None:
    # JSON Schema's integer type admits mathematically integral JSON numbers;
    # the wire parser is deliberately stricter and refuses a float spelling.
    raw = _frame().to_dict()
    raw["sensors"]["ray_actor_kinds"][index] = float(raw["sensors"]["ray_actor_kinds"][index])
    with pytest.raises(ProtocolError, match="must be an integer"):
        parse_message(json.dumps(raw))


@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize(("hit", "actor"), [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 0)])
def test_every_ray_requires_actor_metadata_consistent_with_its_hit(
    index: int, hit: int, actor: int
) -> None:
    raw = _frame().to_dict()
    raw["sensors"]["ray_hit_kinds"][index] = hit
    raw["sensors"]["ray_actor_kinds"][index] = actor
    with pytest.raises(ProtocolError, match="ray_actor_kinds"):
        parse_message(json.dumps(raw))
    assert not _validator().is_valid(raw)


@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize(("hit", "actor"), [(0, 0), (1, 0), (2, 1), (2, 2), (2, 3)])
def test_every_valid_hit_actor_pair_is_admitted_at_maximum_range(
    index: int, hit: int, actor: int
) -> None:
    raw = _frame().to_dict()
    raw["sensors"]["ray_hit_kinds"][index] = hit
    raw["sensors"]["ray_actor_kinds"][index] = actor
    raw["sensors"]["ray_distances_m"][index] = raw["sensors"]["ray_max_distance_m"]
    assert Observation.from_dict(raw).to_dict() == raw
    _validator().validate(raw)


@pytest.mark.parametrize(
    "bad", [None, {}, sensors(), SensorFrameV2.from_sensor_frame(sensors(), (0,) * 6)]
)
def test_actor_constructor_requires_an_explicit_event_frame(bad: object) -> None:
    with pytest.raises(ProtocolError, match="SensorFrameV3"):
        SensorFrameV4.from_event_frame(bad, (0,) * 6)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_actor",
        "missing_events",
        "missing_schema",
        "missing_kinds",
        "extra",
        "unknown_schema",
        "v3_schema",
    ],
)
def test_actor_shape_cannot_be_guessed_or_downgraded(mutation: str) -> None:
    raw = _frame().to_dict()
    sample = raw["sensors"]
    removed = {
        "missing_actor": "ray_actor_kinds",
        "missing_events": "events",
        "missing_schema": "sensor_schema",
        "missing_kinds": "ray_hit_kinds",
    }
    if mutation in removed:
        del sample[removed[mutation]]
    elif mutation == "extra":
        sample["entity_uuid"] = "invented"
    elif mutation == "unknown_schema":
        sample["sensor_schema"] = "neurocraft.sensors.v5"
    else:
        sample["sensor_schema"] = "neurocraft.sensors.v3"
    with pytest.raises(ProtocolError):
        parse_message(json.dumps(raw))
    assert not _validator().is_valid(raw)


def test_v4_retains_event_observation_tick_validation() -> None:
    raw = _frame().to_dict()
    raw["sensors"]["events"]["brush"] = {"count": 2, "last_game_tick": 421}
    with pytest.raises(ProtocolError, match="after the observation"):
        parse_message(json.dumps(raw))


def test_all_four_identities_refuse_every_other_sensor_revision() -> None:
    shapes = [
        sensors(),
        SensorFrameV2.from_sensor_frame(sensors(), (2,) * 6),
        _event_frame(),
        _frame().sensors,
    ]
    identities = [
        LEGACY_OBSERVATION_SCHEMA_SHA256,
        OBSERVATION_SCHEMA_V2_SHA256,
        OBSERVATION_SCHEMA_V3_SHA256,
        OBSERVATION_SCHEMA_V4_SHA256,
    ]
    assert len(set(identities)) == 4
    for index, shape in enumerate(shapes):
        for other_index, identity in enumerate(identities):
            if index == other_index:
                require_sensor_contract(shape, identity)
            else:
                with pytest.raises(ProtocolError, match="session requires"):
                    require_sensor_contract(shape, identity)
    assert sensor_schema_for_identity("0" * 64) == "neurocraft.sensors.v1"


def test_contract_hashes_and_historical_golden_file_bytes_are_frozen() -> None:
    assert (
        LEGACY_OBSERVATION_SCHEMA_SHA256
        == "fbfdd38a650f1496dbf946eea7ab686f039f271e5250139259a3931f95335356"
    )
    assert (
        OBSERVATION_SCHEMA_V2_SHA256
        == "032b6a74d57b37f9fc1883b4865c89ec2eeccc5c9084831e29cd986738559308"
    )
    assert (
        OBSERVATION_SCHEMA_V3_SHA256
        == "d7b135a1ce79d034454394d8ee73eb5b9f5fe46bb79ee924da465b8cec62ba80"
    )
    assert (
        OBSERVATION_SCHEMA_V4_SHA256
        == "e89419ad6b9c174e243e12180d9ffe0f131256a2e0cc2ec14693df0367d601f1"
    )
    assert (
        hashlib.sha256(canonical_json_bytes(SENSOR_V4_DECLARATION)).hexdigest()
        == OBSERVATION_SCHEMA_V4_SHA256
    )
    expected = {
        "observation.json": "7266fe610822d253fbfdc3f9fbedd4ad58ed9fa4de779fa684760abf4f2af1e9",
        "observation_v2.json": "514c96bc01d7272552a844b20c038099a7e6046fe2227e22089dbb459c5ba679",
        "observation_v3.json": "aa9173498a8c39e88ddfd53f84e87d54bf3d90864c8014e9d926c1796f3d9359",
    }
    for name, digest in expected.items():
        content = (ROOT / "packages/protocol/golden" / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == digest
        parsed = parse_message(content)
        assert canonical_json_bytes(parsed) == canonical_json_bytes(json.loads(content))
        assert "ray_actor_kinds" not in json.loads(content)["sensors"]


def test_published_schema_and_new_golden_cover_v4_and_all_fourteen_sensor_flags() -> None:
    content = (ROOT / "packages/protocol/golden/observation_v4.json").read_bytes()
    frame = parse_message(content)
    assert isinstance(frame, Observation)
    assert isinstance(frame.sensors, SensorFrameV4)
    assert canonical_json_bytes(frame) == canonical_json_bytes(json.loads(content))
    _validator().validate(frame.to_dict())
    flags = tuple(sorted(frame.sensors.FIELDS - {"sensor_schema"}))
    assert len(flags) == 14
    flagged = replace(frame, missing_fields=flags, clamped_fields=flags)
    assert parse_message(encode_message(flagged)) == flagged
    _validator().validate(flagged.to_dict())
