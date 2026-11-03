from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from neurocraft_protocol import (
    Observation,
    ProtocolError,
    SensorFrame,
    SensorFrameV2,
    canonical_json_bytes,
    encode_message,
    parse_message,
    require_sensor_contract,
)
from neurocraft_protocol.sensor_contract import (
    LEGACY_OBSERVATION_SCHEMA_SHA256,
    OBSERVATION_SCHEMA_V2_SHA256,
)

from tests.protocol.helpers import observation, sensors

ROOT = Path(__file__).resolve().parents[2]


def test_both_wire_shapes_round_trip_without_inventing_kinds() -> None:
    old = observation()
    new = replace(old, sensors=SensorFrameV2.from_sensor_frame(old.sensors, (0, 1, 2, 0, 1, 2)))
    for frame in (old, new):
        assert parse_message(encode_message(frame)) == frame
    assert type(Observation.from_dict(old.to_dict()).sensors) is SensorFrame
    assert "ray_hit_kinds" not in old.to_dict()["sensors"]
    for name in ("observation", "observation_v2"):
        raw = json.loads((ROOT / f"packages/protocol/golden/{name}.json").read_text())
        assert canonical_json_bytes(Observation.from_dict(raw)) == canonical_json_bytes(raw)


@pytest.mark.parametrize("bad", [(0,) * 5, (0,) * 7, (True,) * 6, (1.0,) * 6, (-1,) * 6, (3,) * 6])
def test_v2_refuses_non_integer_or_out_of_range_kinds(bad: object) -> None:
    with pytest.raises(ProtocolError):
        SensorFrameV2.from_sensor_frame(sensors(), bad)


@pytest.mark.parametrize("mutation", ["drop_kinds", "drop_schema", "unknown_schema", "extra"])
def test_closed_v2_shape_refuses_partial_or_unknown_contract(mutation: str) -> None:
    raw = observation().to_dict()
    raw["sensors"] = SensorFrameV2.from_sensor_frame(sensors(), (2,) * 6).to_dict()
    if mutation == "drop_kinds":
        del raw["sensors"]["ray_hit_kinds"]
    elif mutation == "drop_schema":
        del raw["sensors"]["sensor_schema"]
    elif mutation == "unknown_schema":
        raw["sensors"]["sensor_schema"] = "neurocraft.sensors.v3"
    else:
        raw["sensors"]["target"] = "player"
    with pytest.raises(ProtocolError):
        parse_message(json.dumps(raw))


def test_session_contract_refuses_upgrade_and_downgrade() -> None:
    old = sensors()
    new = SensorFrameV2.from_sensor_frame(old, (2,) * 6)
    require_sensor_contract(old, LEGACY_OBSERVATION_SCHEMA_SHA256)
    require_sensor_contract(new, OBSERVATION_SCHEMA_V2_SHA256)
    with pytest.raises(ProtocolError, match=r"requires neurocraft\.sensors\.v2"):
        require_sensor_contract(old, OBSERVATION_SCHEMA_V2_SHA256)
    with pytest.raises(ProtocolError, match=r"requires neurocraft\.sensors\.v1"):
        require_sensor_contract(new, LEGACY_OBSERVATION_SCHEMA_SHA256)


def test_all_seven_historical_sensor_payloads_round_trip_exactly() -> None:
    # EXTERNAL-BRIDGE-01 predates trace/protocol v2 and already cannot be parsed
    # by the current full-trace reader. Do not silently migrate its envelope;
    # the compatibility promise here is their unchanged physical sensor block.
    paths = sorted((ROOT / "docs/receipts").glob("*-trace.ndjson"))
    assert len(paths) == 7
    for path in paths:
        for line in path.read_text().splitlines():
            envelope = json.loads(line)
            if envelope["kind"] != "record":
                continue
            raw = envelope["payload"]["observation"]["sensors"]
            assert canonical_json_bytes(SensorFrame.from_dict(raw)) == canonical_json_bytes(raw)
