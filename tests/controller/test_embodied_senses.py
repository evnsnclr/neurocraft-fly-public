"""Tests for the Engineered-Encoder Mode sensory profile (``full_mcns_eem_v1``).

Two tiers, mirroring ``test_full_brain_controller.py``:

* CI-safe: a synthetic annotation frame carrying exactly the declared
  population counts plus a synthetic ``FullRuntimeView``. No artifact, no
  feather tables, no compiled kernel required.
* Artifact-gated: skipped unless the local-only full artifact and both pinned
  annotation tables are present. These pin the real counts, the derived EEM
  identities, and -- as a byte-regression gate -- the wing profile's
  source/configuration identities as literals from the original tree. Runtime
  dynamics bind the actual compiled kernel; historical dynamics apply only to
  their recorded kernel identity.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from neurocraft_controller import ConnectomeController, ControllerConfiguration
from neurocraft_controller.embodied_senses import (
    EEM_CHANNELS,
    EEM_ENCODER_SCHEMA,
    EEM_POPULATION_BINDING_SCHEMA,
    EemEncoderDocument,
    EmbodiedPopulationBinding,
    EmbodiedSensesBinding,
    EmbodiedSensesError,
    bind_embodied_senses,
    encode_channel_values,
    population_drives,
    select_embodied_populations,
)
from neurocraft_controller.flight_pools import (
    LEFT_WING_MOTOR_BODY_IDS,
    RIGHT_WING_MOTOR_BODY_IDS,
)
from neurocraft_controller.full_brain_controller import (
    FULL_BRAIN_CONFIG_SCHEMA,
    FULL_BRAIN_EEM_CONFIG_SCHEMA,
    FULL_MCNS_ARTIFACT_MANIFEST_SHA256,
    FullBrainConfiguration,
    FullBrainController,
    FullBrainEemConfiguration,
    FullBrainProfileError,
    bind_sensor_channels,
    bind_wing_motor_pools,
    load_readout_calibration,
)
from neurocraft_neural.full_engine import ArrayNativeEngine
from neurocraft_neural.full_view import FullRuntimeView, ViewProvenance, build_view
from neurocraft_protocol import Observation, SensorFrame, canonical_json_bytes

REPO_ROOT = Path(__file__).resolve().parents[2]
FULL_ARTIFACT_DIRECTORY = REPO_ROOT / "data" / "processed" / "male-cns-v1.0-superclass-v1"
NEUROTRANSMITTERS_PATH = (
    REPO_ROOT / "data" / "cache" / "male-cns-v1.0"
) / "body-neurotransmitters-male-cns-v1.0.feather"
BODY_ANNOTATIONS_PATH = (
    REPO_ROOT / "data" / "cache" / "male-cns-v1.0"
) / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
WING_CALIBRATION_PATH = REPO_ROOT / "data" / "calibration" / "full-readout-calibration-v1.json"
EEM_CALIBRATION_PATH = REPO_ROOT / "data" / "calibration" / "full-readout-calibration-eem-v1.json"

artifact_gated = pytest.mark.skipif(
    not (FULL_ARTIFACT_DIRECTORY / "manifest.json").is_file()
    or not NEUROTRANSMITTERS_PATH.is_file()
    or not BODY_ANNOTATIONS_PATH.is_file(),
    reason="full MaleCNS artifact and annotation tables are local-only",
)

GRAPH_SHA = "ab" * 32
OVERLAY_SHA = "cd" * 32
ANNOTATIONS_SHA = "ef" * 32

#: Current wing-profile pins measured with the portable artifact and regenerated
#: calibration on 2026-10-24. Historical dynamics below still name the recorded
#: FULL-ENGINE-02 binary; a fresh binary must carry its actual identity.
WING_ADAPTER_SHA = "7f1980c280bd1ac47e03071a55f2a950464c9e43cbd55460b7470a530859e6eb"
WING_UNCALIBRATED_DECODER_SHA = "41f6f78d4e8507abb03e9a8d43a36a559ce42da21343ea938902bb606d409f98"
WING_CALIBRATED_DECODER_SHA = "f2bfc064a75051882a9bea552cb692e9ba0d50fe64f49ef16af8f87cbc96af9b"
WING_UNCALIBRATED_CONFIG_SHA = "71f5eb37fc286811f9dd413a4577857ad7a44d58d17e9e264c1ddf678a662a54"
WING_CALIBRATED_CONFIG_SHA = "828c113af53090308aae35dec21962f7e0d218326f927da85125b02978d11902"
WING_DYNAMICS_SHA = "5da651e72477a186c82a6091afa0f415b7e4a182fcd8f014870922143d526d8e"
WING_HISTORICAL_KERNEL_SHA = "ac916a89fe49ca91a2d6a533e4c72a518e209f3e74eb56b93bf58f0a78a75939"
WING_OBSERVATION_SCHEMA_SHA = "fbfdd38a650f1496dbf946eea7ab686f039f271e5250139259a3931f95335356"

#: EEM identity literals derived once from the pinned annotations table and
#: the declared v1 encoder document (measured 2026-08-30).
EEM_POPULATION_BINDING_SHA = "bf45222dd478ecb5dcb35df8ed34e3dd2193449f0d8fbd5f931a24319d023bae"
EEM_ENCODER_SHA = "e1bb7a1fa9838e890e543e1f83b0e605c04657bd65d9051cb5ddba208e25d901"
#: The synthetic tests retain the historical identities above. Actual full-artifact
#: construction below binds the 2026-10-24 portable candidate measured identities.
CURRENT_EEM_POPULATION_BINDING_SHA = (
    "c619c99cb3711541f10a11247ad529d13824f9348abacd4cf96ccbc108be0f1a"
)
CURRENT_EEM_ENCODER_SHA = "c9cfd375a0e14d82d21d474aaf553e5d9864871d8bd14b3c8de9f14d7834ff68"

FAITHFUL_LEFT_BODY = 805_722
FAITHFUL_RIGHT_BODY = 77_185


# ------------------------------------------------------------- synthetic frame


class FakeFrame:
    """A mutable synthetic annotation frame with the exact declared counts."""

    def __init__(self) -> None:
        counts = (("lc4_left", 71), ("lc4_right", 55), ("visual", 6091), ("bristle", 2206))
        rows = sum(count for _, count in counts) + 2
        self.types: list[str | None] = [None] * rows
        self.classes: list[str | None] = [None] * rows
        self.subclasses: list[str | None] = [None] * rows
        self.sides: list[str | None] = [None] * rows
        cursor = 0
        for name, count in counts:
            for _ in range(count):
                if name == "lc4_left":
                    self.types[cursor] = "LC4"
                    self.sides[cursor] = "L"
                elif name == "lc4_right":
                    self.types[cursor] = "LC4"
                    self.sides[cursor] = "R"
                elif name == "visual":
                    self.classes[cursor] = "visual"
                else:
                    self.subclasses[cursor] = "mechanosensory bristle"
                cursor += 1
        self.body_ids: list[int] = [1000 + index for index in range(rows - 2)] + [
            FAITHFUL_LEFT_BODY,
            FAITHFUL_RIGHT_BODY,
        ]
        self.rows = rows

    @property
    def index(self) -> dict[int, int]:
        return {body: position for position, body in enumerate(self.body_ids)}

    def select(
        self,
        *,
        node_count: int | None = None,
        index_of: Callable[[int], int] | None = None,
    ) -> EmbodiedPopulationBinding:
        return select_embodied_populations(
            type_column=self.types,
            class_column=self.classes,
            subclass_column=self.subclasses,
            soma_side_column=self.sides,
            body_ids=self.body_ids,
            index_of=index_of or self.index.__getitem__,
            node_count=node_count or self.rows,
            graph_manifest_sha256=GRAPH_SHA,
            annotations_sha256=ANNOTATIONS_SHA,
        )


def _synthetic_view(node_count: int) -> FullRuntimeView:
    """Deterministic two-in-degree CSR ring with mixed signs (test_full_brain's)."""

    degrees = np.full(node_count, 2, dtype=np.int64)
    indptr = np.zeros(node_count + 1, dtype=np.int64)
    np.cumsum(degrees, out=indptr[1:])
    columns = np.empty(2 * node_count, dtype=np.uint32)
    counts = np.empty(2 * node_count, dtype=np.int64)
    signs = np.empty(2 * node_count, dtype=np.int64)
    rows = np.arange(node_count, dtype=np.int64)
    columns[0::2] = (rows + 1) % node_count
    columns[1::2] = (rows + 7) % node_count
    counts[0::2] = 3
    counts[1::2] = 5
    signs[0::2] = 1
    signs[1::2] = -1
    return build_view(
        indptr=indptr,
        pre_indices=columns,
        raw_counts=counts,
        signs=signs,
        provenance=ViewProvenance(
            origin="synthetic_embodied_senses_test",
            graph_identity_sha256="11" * 32,
        ),
    )


def _controllers(
    frame: FakeFrame | None = None,
) -> tuple[FullBrainController, FullBrainController]:
    """A matched (eem, wing) controller pair over one synthetic node space."""

    frame = frame or FakeFrame()
    index = frame.index
    cursor = frame.rows
    for body in LEFT_WING_MOTOR_BODY_IDS + RIGHT_WING_MOTOR_BODY_IDS:
        index[body] = cursor
        cursor += 1
    node_count = cursor
    senses = bind_embodied_senses(frame.select(node_count=node_count))
    sensory = bind_sensor_channels(
        channels=ConnectomeController._sensor_channels(ControllerConfiguration()),
        index_of=index.__getitem__,
        node_count=node_count,
        graph_manifest_sha256=GRAPH_SHA,
    )
    motor = bind_wing_motor_pools(
        index_of=index.__getitem__,
        gain=8.0,
        dead_zone=0.0,
        graph_manifest_sha256=GRAPH_SHA,
        sign_overlay_identity_sha256=OVERLAY_SHA,
    )
    eem = FullBrainController(
        run_id="embodied-test",
        entity_id="test-fly",
        configuration=FullBrainEemConfiguration(
            encoder_identity_sha256=senses.encoder.identity_sha256
        ),
        engine=ArrayNativeEngine(_synthetic_view(node_count)),
        sensory_binding=sensory,
        motor_binding=motor,
        embodied=senses,
    )
    wing = FullBrainController(
        run_id="embodied-test",
        entity_id="test-fly",
        engine=ArrayNativeEngine(_synthetic_view(node_count)),
        sensory_binding=sensory,
        motor_binding=motor,
    )
    return eem, wing


def _sensors(
    *,
    forward: float = 6.0,
    left: float = 6.0,
    right: float = 6.0,
    light: int = 15,
    horizontal_collision: bool = False,
    vertical_collision: bool = False,
) -> SensorFrame:
    return SensorFrame(
        position_m=(0.0, 64.0, 0.0),
        velocity_m_s=(0.0, 0.0, 0.0),
        yaw_rad=0.0,
        pitch_rad=0.0,
        on_ground=False,
        horizontal_collision=horizontal_collision,
        vertical_collision=vertical_collision,
        in_fluid=False,
        local_light=light,
        ray_max_distance_m=6.0,
        ray_distances_m=(forward, 6.0, left, right, 6.0, 6.0),
    )


def _observation(
    controller: FullBrainController, sequence: int, sensors: SensorFrame
) -> Observation:
    return Observation(
        identity=controller.identity,
        observation_seq=sequence,
        game_tick=sequence,
        sent_monotonic_ns=sequence * 50_000_000,
        sensors=sensors,
    )


# --------------------------------------------------------- selector count gates


def test_selection_matches_declared_counts_and_sorting() -> None:
    frame = FakeFrame()
    binding = frame.select()
    names = tuple(population.name for population in binding.populations)
    assert names == ("lc4_left", "lc4_right", "visual", "bristle")
    sizes = {population.name: len(population.body_ids) for population in binding.populations}
    assert sizes == {"lc4_left": 71, "lc4_right": 55, "visual": 6091, "bristle": 2206}
    for population in binding.populations:
        assert population.body_ids == tuple(sorted(population.body_ids))
        assert len(set(population.dense_indices)) == len(population.dense_indices)
    material = binding.to_identity_material()
    assert material["schema"] == EEM_POPULATION_BINDING_SCHEMA
    assert len(binding.identity_sha256) == 64


def test_selection_refuses_lc4_total_drift() -> None:
    frame = FakeFrame()
    lc4_row = frame.types.index("LC4")
    frame.types[lc4_row] = "LC6"
    frame.sides[lc4_row] = None
    with pytest.raises(EmbodiedSensesError, match="'lc4' count drifted"):
        frame.select()
    extra = FakeFrame()
    visual_row = extra.classes.index("visual")
    extra.types[visual_row] = "LC4"
    extra.sides[visual_row] = "L"
    with pytest.raises(EmbodiedSensesError, match="'lc4' count drifted"):
        extra.select()


def test_selection_refuses_side_split_drift_and_unknown_sides() -> None:
    frame = FakeFrame()
    left_row = frame.sides.index("L")
    frame.sides[left_row] = "R"
    with pytest.raises(EmbodiedSensesError, match="'lc4_left' count drifted"):
        frame.select()
    unknown = FakeFrame()
    unknown.sides[unknown.sides.index("L")] = "M"
    with pytest.raises(EmbodiedSensesError, match="outside the declared L/R split"):
        unknown.select()
    absent = FakeFrame()
    absent.sides[absent.sides.index("R")] = None
    with pytest.raises(EmbodiedSensesError, match="outside the declared L/R split"):
        absent.select()


def test_selection_refuses_visual_and_bristle_drift() -> None:
    frame = FakeFrame()
    frame.classes[frame.classes.index("visual")] = "visual_projection"
    with pytest.raises(EmbodiedSensesError, match="'visual' count drifted"):
        frame.select()
    other = FakeFrame()
    other.subclasses[other.subclasses.index("mechanosensory bristle")] = "campaniform sensilla"
    with pytest.raises(EmbodiedSensesError, match="'bristle' count drifted"):
        other.select()


def test_selection_refuses_population_overlap_without_count_drift() -> None:
    """One row in both visual and bristle, counts held at 6091/2206 exactly."""

    frame = FakeFrame()
    visual_row = frame.classes.index("visual")
    bristle_row = frame.subclasses.index("mechanosensory bristle")
    frame.classes[visual_row] = None
    frame.subclasses[bristle_row] = None
    assert frame.classes[bristle_row] is None  # now fully unlabeled
    frame.classes[bristle_row] = "visual"
    frame.subclasses[bristle_row] = "mechanosensory bristle"
    with pytest.raises(EmbodiedSensesError, match="overlap on 1 bodies"):
        frame.select()


def test_selection_refuses_faithful_entry_capture() -> None:
    frame = FakeFrame()
    visual_row = frame.classes.index("visual")
    frame.classes[visual_row] = None
    frame.classes[frame.rows - 2] = "visual"  # the 805722 row
    with pytest.raises(EmbodiedSensesError, match="faithful sensory entry"):
        frame.select()


def test_selection_refuses_dense_order_mismatch() -> None:
    frame = FakeFrame()
    scrambled = frame.index
    first, second = frame.body_ids[0], frame.body_ids[1]
    scrambled[first], scrambled[second] = scrambled[second], scrambled[first]
    with pytest.raises(EmbodiedSensesError, match="not in dense body order"):
        frame.select(index_of=scrambled.__getitem__)


# ------------------------------------------------------------- encoder formulas


def test_channel_formulas_and_declared_ray_side_mapping() -> None:
    """Index 2 is the left ray and index 3 the right ray, exactly as declared."""

    left_blocked = encode_channel_values(_sensors(left=1.5))
    values = dict(left_blocked)
    assert values["threat_left"] == 0.75
    assert values["threat_right"] == 0.0
    assert values["threat_front"] == 0.0
    right_blocked = dict(encode_channel_values(_sensors(right=1.5)))
    assert right_blocked["threat_right"] == 0.75
    assert right_blocked["threat_left"] == 0.0
    front_blocked = dict(encode_channel_values(_sensors(forward=0.6)))
    assert front_blocked["threat_front"] == (1.0 - 0.6 / 6.0) / 2.0
    assert front_blocked["threat_left"] == 0.0 and front_blocked["threat_right"] == 0.0


def test_channel_formulas_light_touch_and_bounds() -> None:
    assert dict(encode_channel_values(_sensors(light=7)))["light"] == 7.0 / 15.0
    assert dict(encode_channel_values(_sensors(light=0)))["light"] == 0.0
    assert dict(encode_channel_values(_sensors(light=15)))["light"] == 1.0
    assert dict(encode_channel_values(_sensors()))["touch"] == 0.0
    assert dict(encode_channel_values(_sensors(horizontal_collision=True)))["touch"] == 1.0
    assert dict(encode_channel_values(_sensors(vertical_collision=True)))["touch"] == 1.0
    everything = encode_channel_values(
        _sensors(forward=0.0, left=0.0, right=0.0, light=15, horizontal_collision=True)
    )
    assert all(0.0 <= value <= 1.0 for _, value in everything)


def test_population_drives_are_additive_and_clipped() -> None:
    values = {
        "threat_left": 0.8,
        "threat_right": 0.1,
        "threat_front": 0.45,
        "light": 0.5,
        "touch": 1.0,
    }
    drives = dict(population_drives(values, EEM_CHANNELS))
    assert drives["lc4_left"] == 1.0  # 0.8 + 0.45 clipped
    assert drives["lc4_right"] == 0.1 + 0.45
    assert drives["visual"] == 0.5
    assert drives["bristle"] == 1.0


def test_all_declared_gains_default_to_one() -> None:
    assert tuple(channel.gain for channel in EEM_CHANNELS) == (1.0,) * 5
    names = tuple(channel.name for channel in EEM_CHANNELS)
    assert names == ("threat_left", "threat_right", "threat_front", "light", "touch")


def test_encoder_document_binds_its_population_binding() -> None:
    binding = FakeFrame().select()
    senses = bind_embodied_senses(binding)
    assert senses.encoder.population_binding_sha256 == binding.identity_sha256
    assert senses.encoder.to_identity_material()["schema"] == EEM_ENCODER_SCHEMA
    with pytest.raises(EmbodiedSensesError, match="does not bind"):
        EmbodiedSensesBinding(
            populations=binding,
            encoder=EemEncoderDocument(population_binding_sha256="1f" * 32),
        )


# ------------------------------------------------------------- identity gates


def test_eem_configuration_has_its_own_schema_and_rotates() -> None:
    config = FullBrainEemConfiguration(encoder_identity_sha256=EEM_ENCODER_SHA)
    wing = FullBrainConfiguration()
    assert config.schema == FULL_BRAIN_EEM_CONFIG_SCHEMA
    assert wing.schema == FULL_BRAIN_CONFIG_SCHEMA
    assert config.identity_sha256 != wing.identity_sha256
    material = config.to_identity_material()
    assert material["mode_taxonomy"] == "engineered_encoder"
    assert material["encoder_identity_sha256"] == EEM_ENCODER_SHA
    other = FullBrainEemConfiguration(encoder_identity_sha256="2b" * 32)
    assert other.identity_sha256 != config.identity_sha256


def test_eem_configuration_refusals() -> None:
    with pytest.raises(ValueError, match="encoder document identity"):
        FullBrainEemConfiguration()
    with pytest.raises(ValueError, match="mode_taxonomy"):
        FullBrainEemConfiguration(encoder_identity_sha256=EEM_ENCODER_SHA, mode_taxonomy="faithful")
    with pytest.raises(ValueError, match="unsupported engineered-encoder"):
        FullBrainEemConfiguration(
            schema=FULL_BRAIN_CONFIG_SCHEMA, encoder_identity_sha256=EEM_ENCODER_SHA
        )
    with pytest.raises(ValueError, match="five neural steps"):
        FullBrainEemConfiguration(
            encoder_identity_sha256=EEM_ENCODER_SHA, neural_steps_per_game_tick=4
        )


def test_controller_refuses_mismatched_embodied_pairings() -> None:
    eem, wing = _controllers()
    senses = eem.embodied_senses
    assert senses is not None and wing.embodied_senses is None
    with pytest.raises(FullBrainProfileError, match="requires the embodied senses binding"):
        FullBrainController(
            run_id="pairing-test",
            entity_id="test-fly",
            configuration=FullBrainEemConfiguration(
                encoder_identity_sha256=senses.encoder.identity_sha256
            ),
            engine=eem.engine,
            sensory_binding=eem._sensory_binding,
            motor_binding=eem._motor_binding,
        )
    with pytest.raises(FullBrainProfileError, match="requires a FullBrainEemConfiguration"):
        FullBrainController(
            run_id="pairing-test",
            entity_id="test-fly",
            engine=eem.engine,
            sensory_binding=eem._sensory_binding,
            motor_binding=eem._motor_binding,
            embodied=senses,
        )
    with pytest.raises(FullBrainProfileError, match="declared encoder identity"):
        FullBrainController(
            run_id="pairing-test",
            entity_id="test-fly",
            configuration=FullBrainEemConfiguration(encoder_identity_sha256="3c" * 32),
            engine=eem.engine,
            sensory_binding=eem._sensory_binding,
            motor_binding=eem._motor_binding,
            embodied=senses,
        )


def test_adapter_identity_rotates_and_wing_boundary_is_untouched() -> None:
    eem, wing = _controllers()
    assert eem.identity.adapter_hash != wing.identity.adapter_hash
    assert eem.identity.decoder_hash == wing.identity.decoder_hash
    assert eem.identity.dynamics_hash == wing.identity.dynamics_hash
    assert eem.identity.graph_hash == wing.identity.graph_hash
    assert eem.identity.observation_schema_hash == wing.identity.observation_schema_hash
    assert eem.configuration.identity_sha256 != wing.configuration.identity_sha256


# --------------------------------------------------------------- step behavior


def test_step_places_faithful_channels_unchanged_plus_eem_drives() -> None:
    eem, wing = _controllers()
    sensors = _sensors(left=1.5, light=7, horizontal_collision=True)
    eem.step(_observation(eem, 0, sensors), command_seq=0)
    wing.step(_observation(wing, 0, sensors), command_seq=0)
    eem_evidence = eem.last_evidence
    wing_evidence = wing.last_evidence
    assert eem_evidence is not None and wing_evidence is not None
    faithful_keys = set(wing_evidence.encoded_stimulation)
    for key in faithful_keys:
        assert eem_evidence.encoded_stimulation[key] == wing_evidence.encoded_stimulation[key]
    extras = {
        key: value
        for key, value in eem_evidence.encoded_stimulation.items()
        if key not in faithful_keys
    }
    assert extras == {
        "eem:threat_left": 0.75,
        "eem:threat_right": 0.0,
        "eem:threat_front": 0.0,
        "eem:light": 7.0 / 15.0,
        "eem:touch": 1.0,
        "eem_drive:lc4_left": 0.75,
        "eem_drive:lc4_right": 0.0,
        "eem_drive:visual": 7.0 / 15.0,
        "eem_drive:bristle": 1.0,
    }
    assert eem_evidence.state_after_sha256 != wing_evidence.state_after_sha256
    assert eem_evidence.encoded_stimulation_sha256 != wing_evidence.encoded_stimulation_sha256
    # The declared evidence subset is unchanged by the EEM lane.
    assert eem_evidence.activity_subset_body_ids == wing_evidence.activity_subset_body_ids


def test_step_determinism_two_eem_controllers_identical_digests() -> None:
    first, _ = _controllers()
    second, _ = _controllers()
    assert first.identity == second.identity
    for step in range(3):
        sensors = _sensors(left=6.0 - step, light=(step * 5) % 16)
        action_a = first.step(_observation(first, step, sensors), command_seq=step)
        action_b = second.step(_observation(second, step, sensors), command_seq=step)
        assert action_a.state_digest == action_b.state_digest
        evidence_a = first.last_evidence
        evidence_b = second.last_evidence
        assert evidence_a is not None and evidence_b is not None
        assert evidence_a.action_sha256 == evidence_b.action_sha256


# ----------------------------------------------------- calibration cross-binds


def _calibration_document(*, encoder_sha256: str | None) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema": "neurocraft.full_readout_calibration.v1",
        "stream": {"stream_id": "unit-test-stream-v1"},
        "artifact": {"manifest_file_sha256": GRAPH_SHA},
        "sign_overlay": {"identity_sha256": OVERLAY_SHA},
        "calibration": {"readout_scale_float_hex": (500.0).hex(), "readout_scale": 500.0},
    }
    if encoder_sha256 is not None:
        document["encoder"] = {"identity_sha256": encoder_sha256}
    return document


def test_calibration_cross_binding_refuses_both_directions() -> None:
    wing_calibration = load_readout_calibration(
        canonical_json_bytes(_calibration_document(encoder_sha256=None))
    )
    eem_calibration = load_readout_calibration(
        canonical_json_bytes(_calibration_document(encoder_sha256=EEM_ENCODER_SHA))
    )
    assert wing_calibration.encoder_identity_sha256 == ""
    assert eem_calibration.encoder_identity_sha256 == EEM_ENCODER_SHA
    # Matched boundaries pass.
    wing_calibration.assert_binds(
        artifact_manifest_sha256=GRAPH_SHA, sign_overlay_identity_sha256=OVERLAY_SHA
    )
    eem_calibration.assert_binds(
        artifact_manifest_sha256=GRAPH_SHA,
        sign_overlay_identity_sha256=OVERLAY_SHA,
        encoder_identity_sha256=EEM_ENCODER_SHA,
    )
    # A wing document on the EEM boundary is refused.
    with pytest.raises(FullBrainProfileError, match="without an engineered-encoder"):
        wing_calibration.assert_binds(
            artifact_manifest_sha256=GRAPH_SHA,
            sign_overlay_identity_sha256=OVERLAY_SHA,
            encoder_identity_sha256=EEM_ENCODER_SHA,
        )
    # An EEM document on the wing boundary is refused.
    with pytest.raises(FullBrainProfileError, match="wing boundary refuses"):
        eem_calibration.assert_binds(
            artifact_manifest_sha256=GRAPH_SHA, sign_overlay_identity_sha256=OVERLAY_SHA
        )
    # A different encoder document is refused.
    with pytest.raises(FullBrainProfileError, match="different encoder document"):
        eem_calibration.assert_binds(
            artifact_manifest_sha256=GRAPH_SHA,
            sign_overlay_identity_sha256=OVERLAY_SHA,
            encoder_identity_sha256="4d" * 32,
        )


def test_calibration_encoder_section_must_be_well_formed() -> None:
    malformed = _calibration_document(encoder_sha256=None)
    malformed["encoder"] = {"identity_sha256": "not-a-digest"}
    with pytest.raises(FullBrainProfileError, match="encoder section must name"):
        load_readout_calibration(canonical_json_bytes(malformed))
    not_object = _calibration_document(encoder_sha256=None)
    not_object["encoder"] = "wrong"
    with pytest.raises(FullBrainProfileError, match="encoder section must be an object"):
        load_readout_calibration(canonical_json_bytes(not_object))


def test_from_artifact_refuses_embodied_argument_misuse(tmp_path: Path) -> None:
    """Argument-pairing refusals fire before any artifact byte is read."""

    with pytest.raises(FullBrainProfileError, match="requires body_annotations_path"):
        FullBrainController.from_artifact(
            run_id="refusal",
            entity_id="fly",
            artifact_directory=tmp_path / "absent",
            annotations_path=tmp_path / "absent.feather",
            embodied_senses=True,
        )
    with pytest.raises(FullBrainProfileError, match="embodied-senses profile only"):
        FullBrainController.from_artifact(
            run_id="refusal",
            entity_id="fly",
            artifact_directory=tmp_path / "absent",
            annotations_path=tmp_path / "absent.feather",
            body_annotations_path=tmp_path / "absent-annotations.feather",
        )
    with pytest.raises(FullBrainProfileError, match="plain FullBrainConfiguration base"):
        FullBrainController.from_artifact(
            run_id="refusal",
            entity_id="fly",
            configuration=FullBrainEemConfiguration(encoder_identity_sha256="5e" * 32),
            artifact_directory=tmp_path / "absent",
            annotations_path=tmp_path / "absent.feather",
            embodied_senses=True,
            body_annotations_path=tmp_path / "absent-annotations.feather",
        )
    with pytest.raises(FullBrainProfileError, match="body-annotations"):
        FullBrainController.from_artifact(
            run_id="refusal",
            entity_id="fly",
            artifact_directory=tmp_path / "absent",
            annotations_path=tmp_path / "absent.feather",
            embodied_senses=True,
            body_annotations_path=tmp_path / "absent-annotations.feather",
        )


# ------------------------------------------------------------- artifact-gated


@artifact_gated
def test_artifact_population_counts_and_derived_identities() -> None:
    controller = FullBrainController.from_artifact(
        run_id="eem-artifact-test",
        entity_id="test-fly",
        artifact_directory=FULL_ARTIFACT_DIRECTORY,
        annotations_path=NEUROTRANSMITTERS_PATH,
        embodied_senses=True,
        body_annotations_path=BODY_ANNOTATIONS_PATH,
    )
    senses = controller.embodied_senses
    assert senses is not None
    sizes = {
        population.name: len(population.body_ids) for population in senses.populations.populations
    }
    assert sizes == {"lc4_left": 71, "lc4_right": 55, "visual": 6091, "bristle": 2206}
    assert sum(sizes.values()) - sizes["visual"] - sizes["bristle"] == 126  # LC4 total
    assert senses.populations.identity_sha256 == CURRENT_EEM_POPULATION_BINDING_SHA
    assert senses.encoder.identity_sha256 == CURRENT_EEM_ENCODER_SHA
    assert controller.identity.graph_hash == FULL_MCNS_ARTIFACT_MANIFEST_SHA256
    assert isinstance(controller.configuration, FullBrainEemConfiguration)
    print(
        "full_mcns_eem_v1 identity:"
        f" adapter_hash={controller.identity.adapter_hash}"
        f" decoder_hash={controller.identity.decoder_hash}"
        f" configuration_identity_sha256={controller.configuration.identity_sha256}"
        f" population_binding={senses.populations.identity_sha256}"
        f" encoder={senses.encoder.identity_sha256}"
    )


@artifact_gated
def test_artifact_eem_exact_same_backend_replay() -> None:
    def build() -> FullBrainController:
        return FullBrainController.from_artifact(
            run_id="eem-replay-test",
            entity_id="test-fly",
            artifact_directory=FULL_ARTIFACT_DIRECTORY,
            annotations_path=NEUROTRANSMITTERS_PATH,
            embodied_senses=True,
            body_annotations_path=BODY_ANNOTATIONS_PATH,
        )

    first = build()
    second = build()
    assert first.identity == second.identity
    for step in range(3):
        sensors = _sensors(left=1.0 + step, light=(step * 7) % 16, horizontal_collision=step == 1)
        first.step(_observation(first, step, sensors), command_seq=step)
        second.step(_observation(second, step, sensors), command_seq=step)
        evidence_a = first.last_evidence
        evidence_b = second.last_evidence
        assert evidence_a is not None and evidence_b is not None
        assert evidence_a.action_sha256 == evidence_b.action_sha256
        assert evidence_a.state_after_sha256 == evidence_b.state_after_sha256
    assert first.engine.trace_sha256 == second.engine.trace_sha256


@artifact_gated
@pytest.mark.skipif(
    not WING_CALIBRATION_PATH.is_file(), reason="the committed wing calibration is absent"
)
def test_artifact_wing_identities_are_byte_unchanged() -> None:
    """The EEM change preserves wing pins under the same actual kernel build."""

    calibrated = FullBrainController.from_artifact(
        run_id="wing-regression-pin",
        entity_id="test-fly",
        artifact_directory=FULL_ARTIFACT_DIRECTORY,
        annotations_path=NEUROTRANSMITTERS_PATH,
        calibration_path=WING_CALIBRATION_PATH,
    )
    uncalibrated = FullBrainController.from_artifact(
        run_id="wing-regression-pin",
        entity_id="test-fly",
        artifact_directory=FULL_ARTIFACT_DIRECTORY,
        annotations_path=NEUROTRANSMITTERS_PATH,
    )
    assert calibrated.identity.adapter_hash == WING_ADAPTER_SHA
    assert uncalibrated.identity.adapter_hash == WING_ADAPTER_SHA
    assert calibrated.identity.decoder_hash == WING_CALIBRATED_DECODER_SHA
    assert uncalibrated.identity.decoder_hash == WING_UNCALIBRATED_DECODER_SHA
    assert calibrated.configuration.identity_sha256 == WING_CALIBRATED_CONFIG_SHA
    assert uncalibrated.configuration.identity_sha256 == WING_UNCALIBRATED_CONFIG_SHA
    current_dynamics = calibrated.engine.dynamics_identity_sha256
    assert calibrated.identity.dynamics_hash == uncalibrated.identity.dynamics_hash
    assert calibrated.identity.dynamics_hash == current_dynamics
    assert uncalibrated.engine.dynamics_identity_sha256 == current_dynamics
    if calibrated.engine.kernel.identity.identity_sha256 == WING_HISTORICAL_KERNEL_SHA:
        assert current_dynamics == WING_DYNAMICS_SHA
    else:
        assert current_dynamics != WING_DYNAMICS_SHA
    # Historical identity math and current-library binding/rotation are independently
    # checked in tests/full_backend/test_full_controls.py, without requiring a new
    # build to impersonate the historical binary.
    assert calibrated.identity.observation_schema_hash == WING_OBSERVATION_SCHEMA_SHA
    assert type(calibrated.configuration) is FullBrainConfiguration


@artifact_gated
@pytest.mark.skipif(
    not WING_CALIBRATION_PATH.is_file() or not EEM_CALIBRATION_PATH.is_file(),
    reason="both committed calibrations are required",
)
def test_artifact_calibrations_refuse_the_other_boundary() -> None:
    with pytest.raises(FullBrainProfileError, match="without an engineered-encoder"):
        FullBrainController.from_artifact(
            run_id="cross-binding-refusal",
            entity_id="test-fly",
            artifact_directory=FULL_ARTIFACT_DIRECTORY,
            annotations_path=NEUROTRANSMITTERS_PATH,
            calibration_path=WING_CALIBRATION_PATH,
            embodied_senses=True,
            body_annotations_path=BODY_ANNOTATIONS_PATH,
        )
    with pytest.raises(FullBrainProfileError, match="wing boundary refuses"):
        FullBrainController.from_artifact(
            run_id="cross-binding-refusal",
            entity_id="test-fly",
            artifact_directory=FULL_ARTIFACT_DIRECTORY,
            annotations_path=NEUROTRANSMITTERS_PATH,
            calibration_path=EEM_CALIBRATION_PATH,
        )


@artifact_gated
@pytest.mark.skipif(
    not EEM_CALIBRATION_PATH.is_file(), reason="the committed EEM calibration is absent"
)
def test_artifact_eem_calibration_binds_and_rotates_the_decoder() -> None:
    from hashlib import sha256

    raw = EEM_CALIBRATION_PATH.read_bytes()
    expected = load_readout_calibration(raw)
    assert expected.encoder_identity_sha256 == CURRENT_EEM_ENCODER_SHA
    calibrated = FullBrainController.from_artifact(
        run_id="eem-calibrated-test",
        entity_id="test-fly",
        artifact_directory=FULL_ARTIFACT_DIRECTORY,
        annotations_path=NEUROTRANSMITTERS_PATH,
        calibration_path=EEM_CALIBRATION_PATH,
        embodied_senses=True,
        body_annotations_path=BODY_ANNOTATIONS_PATH,
    )
    assert calibrated.readout_scale == expected.readout_scale
    assert calibrated.configuration.readout_calibration_sha256 == sha256(raw).hexdigest()
    assert calibrated.identity.decoder_hash != WING_CALIBRATED_DECODER_SHA
    assert calibrated.identity.decoder_hash != WING_UNCALIBRATED_DECODER_SHA
    print(
        "full_mcns_eem_v1 calibrated identity:"
        f" readout_scale={calibrated.readout_scale!r}"
        f" calibration_sha256={calibrated.configuration.readout_calibration_sha256}"
        f" decoder_hash={calibrated.identity.decoder_hash}"
        f" configuration_identity_sha256={calibrated.configuration.identity_sha256}"
    )
