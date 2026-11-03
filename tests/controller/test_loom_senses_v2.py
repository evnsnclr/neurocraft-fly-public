from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from neurocraft_controller.embodied_senses import (
    EmbodiedSensesError,
    apply_temporal_contrast,
    encode_channel_values,
)
from neurocraft_controller.features import EntityRayFeatureEncoder, PhysicalFeatureEncoder
from neurocraft_controller.full_brain_controller import (
    FullBrainController,
    FullBrainLoomV2Configuration,
    FullBrainProfileError,
    load_loom_v1_readout_calibration,
    load_loom_v2_readout_calibration,
)
from neurocraft_controller.loom_senses import (
    LOOM_V1_CONTRAST_CHANNEL_NAMES,
    LOOM_V1_RECTIFIED_CHANNEL_NAMES,
)
from neurocraft_controller.loom_senses_v2 import (
    EemEncoderDocumentLoomV2,
    bind_embodied_senses_loom_v2,
    encode_entity_channel_values,
)
from neurocraft_protocol import ProtocolError, SensorFrame, SensorFrameV2
from neurocraft_protocol.sensor_contract import (
    LEGACY_OBSERVATION_SCHEMA_SHA256,
    OBSERVATION_SCHEMA_V2_SHA256,
)

from scripts.probe_loom_encoder import sensors as probe_sensors
from tests.controller.test_embodied_senses import _observation
from tests.controller.test_loom_senses import _loom_controllers


def drives(frames: list[SensorFrame], *, entity_gate: bool) -> list[dict[str, float]]:
    means = None
    result = []
    for frame in frames:
        raw = encode_entity_channel_values(frame) if entity_gate else encode_channel_values(frame)
        values, _, means = apply_temporal_contrast(
            raw,
            means,
            alpha=0.125,
            contrast_channels=LOOM_V1_CONTRAST_CHANNEL_NAMES,
            rectified_channels=LOOM_V1_RECTIFIED_CHANNEL_NAMES,
        )
        result.append(dict(values))
    return result


@pytest.mark.parametrize(
    "slot,name", [(0, "threat_front"), (2, "threat_left"), (3, "threat_right")]
)
@pytest.mark.parametrize("non_entity_kind", [0, 1])
def test_block_approach_is_zero_and_entity_series_matches_v1_bitwise(
    slot: int, name: str, non_entity_kind: int
) -> None:
    frames = []
    for tick in range(52):
        rays = [6.0] * 6
        rays[slot] = 6.0 - tick * 0.088
        frames.append(replace(probe_sensors(), ray_distances_m=tuple(rays)))
    legacy = drives(frames, entity_gate=False)
    block = drives(
        [SensorFrameV2.from_sensor_frame(f, (non_entity_kind,) * 6) for f in frames],
        entity_gate=True,
    )
    entity = drives(
        [SensorFrameV2.from_sensor_frame(f, (2,) * 6) for f in frames], entity_gate=True
    )
    assert max(row[name] for row in legacy) > 0.0
    assert all(row[name] == 0.0 for row in block)
    assert [[v.hex() for v in row.values()] for row in entity] == [
        [v.hex() for v in row.values()] for row in legacy
    ]


def test_entity_steps_aside_and_crossing_step_are_explicit() -> None:
    entity = SensorFrameV2.from_sensor_frame(probe_sensors(front=2.0), (2,) * 6)
    wall = SensorFrameV2.from_sensor_frame(probe_sensors(front=2.1), (1,) * 6)
    assert all(row["threat_front"] == 0.0 for row in drives([entity, wall, wall], entity_gate=True))
    open_frame = SensorFrameV2.from_sensor_frame(probe_sensors(), (0,) * 6)
    crossing = SensorFrameV2.from_sensor_frame(probe_sensors(front=4.0), (2,) * 6)
    rows = drives([open_frame, crossing, crossing, crossing], entity_gate=True)
    assert rows[1]["threat_front"] == (1.0 - 4.0 / 6.0) / 2.0
    assert rows[1]["threat_front"] > rows[2]["threat_front"] > rows[3]["threat_front"] > 0.0


def test_legacy_input_is_never_classified_as_entity() -> None:
    with pytest.raises(EmbodiedSensesError, match="typed rays"):
        encode_entity_channel_values(probe_sensors(front=0.5))


def test_frozen_feature_identity_and_values_stay_intact() -> None:
    old, new = PhysicalFeatureEncoder(), EntityRayFeatureEncoder()
    assert old.identity_sha256 == LEGACY_OBSERVATION_SCHEMA_SHA256
    assert new.identity_sha256 == OBSERVATION_SCHEMA_V2_SHA256
    frame = probe_sensors(front=1.0)
    assert old.encode(frame).values == new.encode(frame).values


def typed_controller() -> FullBrainController:
    old, _ = _loom_controllers()
    assert old.embodied_senses is not None and old.escape_binding is not None
    binding = bind_embodied_senses_loom_v2(old.embodied_senses.populations)
    return FullBrainController(
        run_id="loom-v2-test",
        entity_id="test-fly",
        engine=old.engine,
        configuration=FullBrainLoomV2Configuration(
            encoder_identity_sha256=binding.encoder.identity_sha256
        ),
        sensory_binding=old._sensory_binding,
        motor_binding=old.escape_binding.wing,
        embodied=binding,
        escape=replace(old.escape_binding, encoder_identity_sha256=binding.encoder.identity_sha256),
    )


def test_direct_controller_also_refuses_downgrade_before_stepping() -> None:
    controller = typed_controller()
    frame = _observation(controller, 0, probe_sensors(front=0.5))
    with pytest.raises(ProtocolError, match=r"requires neurocraft\.sensors\.v2"):
        controller.step(frame, command_seq=0)
    assert controller.engine.step_index == 0
    typed = replace(frame, sensors=SensorFrameV2.from_sensor_frame(frame.sensors, (2,) * 6))
    controller.step(typed, command_seq=0)
    assert controller.engine.step_index == 5


def test_committed_calibration_preserves_measurements_and_refuses_cross_pairings() -> None:
    import json

    root = Path(__file__).resolve().parents[2] / "data/calibration"
    old_raw = (root / "full-readout-calibration-loom-v1.json").read_bytes()
    new_raw = (root / "full-readout-calibration-loom-v2.json").read_bytes()
    old, new = load_loom_v1_readout_calibration(old_raw), load_loom_v2_readout_calibration(new_raw)
    assert new.document_sha256 == "a09379bdec6e81091e6e4ff4ff82e29ae9c21347e14d6dddb3479d80c211ea96"
    assert float(new.wing_scale).hex() == float(old.wing_scale).hex() == "0x1.a0fd0307362bep+6"
    assert float(new.jump_scale).hex() == float(old.jump_scale).hex() == "0x1.b4945f22255b6p+2"
    old_doc, new_doc = json.loads(old_raw), json.loads(new_raw)
    assert new_doc["distributions"] == old_doc["distributions"]
    assert new_doc["artifact"] == old_doc["artifact"]
    assert new_doc["measured_under"]["dynamics_hash"] == old_doc["measured_under"]["dynamics_hash"]
    assert new_doc["measured_under"]["observation_schema_hash"] == OBSERVATION_SCHEMA_V2_SHA256
    declared = EemEncoderDocumentLoomV2(
        population_binding_sha256=new_doc["encoder"]["population_binding_sha256"]
    )
    assert declared.identity_sha256 == new.encoder_identity_sha256
    assert (
        declared.identity_sha256
        == "722b43bd0e697e9921b66e196b578813150c6c49e30bd981dadcefc16c2b408b"
    )
    for loader, raw in (
        (load_loom_v1_readout_calibration, new_raw),
        (load_loom_v2_readout_calibration, old_raw),
    ):
        with pytest.raises(FullBrainProfileError, match="calibration"):
            loader(raw)
