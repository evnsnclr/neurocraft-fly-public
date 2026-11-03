"""Tests for the temporal-contrast Engineered-Encoder profile (``full_mcns_eem_v2``).

All CI-safe by design (a phase-1 constraint of the eem-v2 lane): synthetic
annotation frames and synthetic views only -- no artifact, no feather tables,
no committed eem-v2 calibration (that document is a phase-2 deliverable). The
v1 helpers are imported from ``test_embodied_senses`` so the two lanes share
one synthetic node space; the v1 and wing identity pins asserted there remain
the regression gate that this additive lane moved nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from neurocraft_controller.embodied_senses import (
    EEM_CHANNELS,
    EEM_CONTRAST_ALPHA,
    EEM_ENCODER_V2_SCHEMA,
    EEM_V2_ABSOLUTE_CHANNEL_NAMES,
    EEM_V2_CHANNELS,
    EEM_V2_CONTRAST_CHANNEL_NAMES,
    EemEncoderDocument,
    EemEncoderDocumentV2,
    EmbodiedSensesError,
    apply_temporal_contrast,
    bind_embodied_senses,
    bind_embodied_senses_v2,
)
from neurocraft_controller.full_brain_controller import (
    FULL_BRAIN_EEM_CONFIG_SCHEMA,
    FULL_BRAIN_EEM_V2_CONFIG_SCHEMA,
    FullBrainController,
    FullBrainEemConfiguration,
    FullBrainEemV2Configuration,
    FullBrainProfileError,
    load_readout_calibration,
)
from neurocraft_neural.full_engine import ArrayNativeEngine
from neurocraft_protocol import canonical_json_bytes

from tests.controller.test_embodied_senses import (
    EEM_ENCODER_SHA,
    EEM_POPULATION_BINDING_SHA,
    GRAPH_SHA,
    OVERLAY_SHA,
    FakeFrame,
    _calibration_document,
    _controllers,
    _observation,
    _sensors,
    _synthetic_view,
)

#: v2 identity literals, derivable WITHOUT the artifact: the encoder document
#: hashes only declared material plus the pinned population-binding identity
#: (measured 2026-08-30 for D-058 and asserted artifact-gated in
#: ``test_embodied_senses.py``), so these pins hold on CI as pure hashing.
EEM_V2_ENCODER_SHA = "31c4cd68cbbe3d7afb32d53d963a237f3fe27da8023e2bfc4dc818aaf11c9cd6"
EEM_V2_UNCALIBRATED_CONFIG_SHA = "63cfde2cb15627dec2907df250504b3b652407f7be2af2e83b1110ab0604ec94"
#: CI-safe v1 regression pins re-derived the same way: the v1 encoder document
#: and uncalibrated v1 EEM configuration identities must not move a bit.
EEM_V1_UNCALIBRATED_CONFIG_SHA = "90da8e7056a92f62737cf31b9f983b1612d133704a4d3833186178df2137dd56"


def _v2_controllers() -> tuple[FullBrainController, FullBrainController]:
    """A matched (v2, v1) EEM controller pair over one synthetic node space.

    Mirrors ``test_embodied_senses._controllers`` exactly, swapping only the
    encoder document and configuration type for the first controller.
    """

    from neurocraft_controller import ConnectomeController, ControllerConfiguration
    from neurocraft_controller.flight_pools import (
        LEFT_WING_MOTOR_BODY_IDS,
        RIGHT_WING_MOTOR_BODY_IDS,
    )
    from neurocraft_controller.full_brain_controller import (
        bind_sensor_channels,
        bind_wing_motor_pools,
    )

    frame = FakeFrame()
    index = frame.index
    cursor = frame.rows
    for body in LEFT_WING_MOTOR_BODY_IDS + RIGHT_WING_MOTOR_BODY_IDS:
        index[body] = cursor
        cursor += 1
    node_count = cursor
    binding = frame.select(node_count=node_count)
    senses_v2 = bind_embodied_senses_v2(binding)
    senses_v1 = bind_embodied_senses(binding)
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
    v2 = FullBrainController(
        run_id="embodied-v2-test",
        entity_id="test-fly",
        configuration=FullBrainEemV2Configuration(
            encoder_identity_sha256=senses_v2.encoder.identity_sha256
        ),
        engine=ArrayNativeEngine(_synthetic_view(node_count)),
        sensory_binding=sensory,
        motor_binding=motor,
        embodied=senses_v2,
    )
    v1 = FullBrainController(
        run_id="embodied-v2-test",
        entity_id="test-fly",
        configuration=FullBrainEemConfiguration(
            encoder_identity_sha256=senses_v1.encoder.identity_sha256
        ),
        engine=ArrayNativeEngine(_synthetic_view(node_count)),
        sensory_binding=sensory,
        motor_binding=motor,
        embodied=senses_v1,
    )
    return v2, v1


# ------------------------------------------------------------- contrast math


def test_first_tick_baseline_is_the_raw_value_and_drive_is_zero() -> None:
    raw = (("threat_left", 0.5), ("threat_right", 0.0), ("light", 1.0), ("touch", 1.0))
    encoded, means_used, new_means = apply_temporal_contrast(raw, None)
    assert dict(encoded) == {
        "threat_left": 0.0,
        "threat_right": 0.0,
        "light": 0.0,
        "touch": 1.0,
    }
    assert dict(means_used) == {"threat_left": 0.5, "threat_right": 0.0, "light": 1.0}
    assert new_means == {"threat_left": 0.5, "threat_right": 0.0, "light": 1.0}


def test_running_mean_update_uses_alpha_one_eighth_and_prior_baseline() -> None:
    assert EEM_CONTRAST_ALPHA == 0.125 == 1.0 / 8.0
    _, _, means = apply_temporal_contrast((("light", 1.0), ("touch", 0.0)), None)
    encoded, means_used, means = apply_temporal_contrast((("light", 0.2), ("touch", 0.0)), means)
    # The drive is scored against m_(t-1) = 1.0, BEFORE the update.
    assert dict(encoded)["light"] == 0.8
    assert dict(means_used)["light"] == 1.0
    # m_1 = (1 - 1/8) * 1.0 + (1/8) * 0.2 -- exact, all inputs dyadic-friendly.
    assert means["light"] == 0.875 * 1.0 + 0.125 * 0.2
    encoded, means_used, means = apply_temporal_contrast((("light", 0.2), ("touch", 0.0)), means)
    # A held signal decays toward baseline: |0.2 - 0.9| = 0.7 this tick.
    assert dict(encoded)["light"] == pytest.approx(0.7)
    assert means["light"] == pytest.approx(0.875 * 0.9 + 0.125 * 0.2)


def test_touch_stays_absolute_and_never_adapts() -> None:
    means: dict[str, float] | None = None
    for _ in range(5):
        encoded, means_used, means = apply_temporal_contrast(
            (("light", 0.5), ("touch", 1.0)), means
        )
        assert dict(encoded)["touch"] == 1.0  # a held collision never fades
        assert "touch" not in dict(means_used)
        assert "touch" not in means


def test_contrast_refuses_bad_inputs() -> None:
    with pytest.raises(EmbodiedSensesError, match="outside \\[0, 1\\]"):
        apply_temporal_contrast((("light", 1.5), ("touch", 0.0)), None)
    with pytest.raises(EmbodiedSensesError, match="running mean"):
        apply_temporal_contrast((("light", 0.5), ("touch", 0.0)), {"light": 2.0})
    with pytest.raises(EmbodiedSensesError, match="exactly the contrast channels"):
        apply_temporal_contrast((("light", 0.5), ("touch", 0.0)), {"wrong": 0.5})
    with pytest.raises(EmbodiedSensesError, match="exactly the contrast channels"):
        apply_temporal_contrast((("light", 0.5), ("touch", 0.0)), {"light": 0.5, "extra": 0.5})
    with pytest.raises(EmbodiedSensesError, match="alpha"):
        apply_temporal_contrast((("light", 0.5),), None, alpha=0.0)


# --------------------------------------------------------------- declaration


def test_v2_channels_keep_v1_targets_units_and_gains() -> None:
    by_name_v1 = {channel.name: channel for channel in EEM_CHANNELS}
    assert tuple(channel.name for channel in EEM_V2_CHANNELS) == tuple(
        channel.name for channel in EEM_CHANNELS
    )
    for channel in EEM_V2_CHANNELS:
        v1 = by_name_v1[channel.name]
        assert channel.populations == v1.populations
        assert channel.unit == v1.unit
        assert channel.gain == v1.gain == 1.0
        if channel.name in EEM_V2_ABSOLUTE_CHANNEL_NAMES:
            assert channel.formula == v1.formula  # touch carried over verbatim
        else:
            assert channel.formula != v1.formula
            assert "temporal contrast" in channel.formula
            assert v1.formula in channel.formula  # the v1 raw formula is embedded


def test_v2_contrast_and_absolute_sets_partition_the_channels() -> None:
    names = {channel.name for channel in EEM_V2_CHANNELS}
    contrast = set(EEM_V2_CONTRAST_CHANNEL_NAMES)
    absolute = set(EEM_V2_ABSOLUTE_CHANNEL_NAMES)
    assert contrast | absolute == names
    assert not contrast & absolute
    assert absolute == {"touch"}


def test_v2_encoder_document_schema_alpha_and_rotation() -> None:
    binding = FakeFrame().select()
    senses = bind_embodied_senses_v2(binding)
    encoder = senses.encoder
    assert isinstance(encoder, EemEncoderDocumentV2)
    material = encoder.to_identity_material()
    assert material["schema"] == EEM_ENCODER_V2_SCHEMA
    assert material["contrast"]["alpha_float_hex"] == (0.125).hex()
    assert "touch_exception" in material
    encodings = {entry["name"]: entry["encoding"] for entry in material["channels"]}
    assert encodings == {
        "threat_left": "temporal_contrast",
        "threat_right": "temporal_contrast",
        "threat_front": "temporal_contrast",
        "light": "temporal_contrast",
        "touch": "absolute",
    }
    v1 = bind_embodied_senses(binding).encoder
    assert encoder.identity_sha256 != v1.identity_sha256
    assert encoder.population_binding_sha256 == binding.identity_sha256
    with pytest.raises(EmbodiedSensesError, match="alpha = 1/8"):
        EemEncoderDocumentV2(population_binding_sha256=binding.identity_sha256, alpha=0.25)
    with pytest.raises(EmbodiedSensesError, match="partition"):
        EemEncoderDocumentV2(
            population_binding_sha256=binding.identity_sha256,
            contrast_channels=("threat_left",),
        )


def test_identities_derive_from_the_pinned_population_binding_without_the_artifact() -> None:
    """Pure-hashing pins: the v2 identities and the v1 regression identities."""

    v1 = EemEncoderDocument(population_binding_sha256=EEM_POPULATION_BINDING_SHA)
    v2 = EemEncoderDocumentV2(population_binding_sha256=EEM_POPULATION_BINDING_SHA)
    assert v1.identity_sha256 == EEM_ENCODER_SHA
    assert v2.identity_sha256 == EEM_V2_ENCODER_SHA
    assert (
        FullBrainEemConfiguration(encoder_identity_sha256=EEM_ENCODER_SHA).identity_sha256
        == EEM_V1_UNCALIBRATED_CONFIG_SHA
    )
    assert (
        FullBrainEemV2Configuration(encoder_identity_sha256=EEM_V2_ENCODER_SHA).identity_sha256
        == EEM_V2_UNCALIBRATED_CONFIG_SHA
    )


# ------------------------------------------------------------- configuration


def test_v2_configuration_has_its_own_schema_and_rotates() -> None:
    v2 = FullBrainEemV2Configuration(encoder_identity_sha256=EEM_V2_ENCODER_SHA)
    v1 = FullBrainEemConfiguration(encoder_identity_sha256=EEM_V2_ENCODER_SHA)
    assert v2.schema == FULL_BRAIN_EEM_V2_CONFIG_SCHEMA
    assert v2.identity_sha256 != v1.identity_sha256  # same encoder sha, distinct schema
    material = v2.to_identity_material()
    assert material["schema"] == FULL_BRAIN_EEM_V2_CONFIG_SCHEMA
    assert material["mode_taxonomy"] == "engineered_encoder"
    assert material["encoder_identity_sha256"] == EEM_V2_ENCODER_SHA


def test_v2_configuration_refusals() -> None:
    with pytest.raises(ValueError, match="encoder document identity"):
        FullBrainEemV2Configuration()
    with pytest.raises(ValueError, match="mode_taxonomy"):
        FullBrainEemV2Configuration(
            encoder_identity_sha256=EEM_V2_ENCODER_SHA, mode_taxonomy="faithful"
        )
    with pytest.raises(ValueError, match="unsupported temporal-contrast"):
        FullBrainEemV2Configuration(
            schema=FULL_BRAIN_EEM_CONFIG_SCHEMA, encoder_identity_sha256=EEM_V2_ENCODER_SHA
        )
    with pytest.raises(ValueError, match="five neural steps"):
        FullBrainEemV2Configuration(
            encoder_identity_sha256=EEM_V2_ENCODER_SHA, neural_steps_per_game_tick=4
        )


def test_controller_refuses_v1_v2_config_document_mixes() -> None:
    v2, v1 = _v2_controllers()
    senses_v2 = v2.embodied_senses
    senses_v1 = v1.embodied_senses
    assert senses_v2 is not None and senses_v1 is not None
    with pytest.raises(FullBrainProfileError, match="must be paired"):
        FullBrainController(
            run_id="mix-test",
            entity_id="test-fly",
            configuration=FullBrainEemV2Configuration(
                encoder_identity_sha256=senses_v1.encoder.identity_sha256
            ),
            engine=v2.engine,
            sensory_binding=v2._sensory_binding,
            motor_binding=v2._motor_binding,
            embodied=senses_v1,
        )
    with pytest.raises(FullBrainProfileError, match="must be paired"):
        FullBrainController(
            run_id="mix-test",
            entity_id="test-fly",
            configuration=FullBrainEemConfiguration(
                encoder_identity_sha256=senses_v2.encoder.identity_sha256
            ),
            engine=v2.engine,
            sensory_binding=v2._sensory_binding,
            motor_binding=v2._motor_binding,
            embodied=senses_v2,
        )


# --------------------------------------------------------------- step behavior


def test_first_tick_contrast_drives_are_zero_and_touch_is_absolute() -> None:
    v2, _ = _v2_controllers()
    sensors = _sensors(left=1.5, light=7, horizontal_collision=True)
    v2.step(_observation(v2, 0, sensors), command_seq=0)
    evidence = v2.last_evidence
    assert evidence is not None
    encoded = dict(evidence.encoded_stimulation)
    assert encoded["eem:threat_left"] == 0.0  # raw 0.75, but the baseline is raw_0
    assert encoded["eem:light"] == 0.0
    assert encoded["eem:touch"] == 1.0
    assert encoded["eem_raw:threat_left"] == 0.75
    assert encoded["eem_raw:light"] == 7.0 / 15.0
    assert encoded["eem_mean:threat_left"] == 0.75
    assert encoded["eem_mean:light"] == 7.0 / 15.0
    assert "eem_mean:touch" not in encoded
    assert encoded["eem_drive:lc4_left"] == 0.0
    assert encoded["eem_drive:lc4_right"] == 0.0
    assert encoded["eem_drive:visual"] == 0.0
    assert encoded["eem_drive:bristle"] == 1.0


def test_later_ticks_encode_deviation_from_the_leaky_mean() -> None:
    v2, _ = _v2_controllers()
    v2.step(_observation(v2, 0, _sensors(light=15)), command_seq=0)  # baselines: 0.0 / 1.0
    v2.step(_observation(v2, 1, _sensors(left=1.5, light=3)), command_seq=1)
    evidence = v2.last_evidence
    assert evidence is not None
    encoded = dict(evidence.encoded_stimulation)
    # threat_left: raw 0.75 against baseline 0.0; light: raw 0.2 against 1.0.
    assert encoded["eem:threat_left"] == 0.75
    assert encoded["eem_mean:threat_left"] == 0.0
    assert encoded["eem:light"] == 0.8
    assert encoded["eem_mean:light"] == 1.0
    assert encoded["eem_drive:lc4_left"] == 0.75
    assert encoded["eem_drive:visual"] == 0.8
    # Held condition adapts: baseline moved by alpha toward the held value.
    v2.step(_observation(v2, 2, _sensors(left=1.5, light=3)), command_seq=2)
    evidence = v2.last_evidence
    assert evidence is not None
    encoded = dict(evidence.encoded_stimulation)
    assert encoded["eem_mean:threat_left"] == 0.125 * 0.75
    assert encoded["eem:threat_left"] == 0.75 - 0.125 * 0.75


def test_two_v2_controllers_replay_identically() -> None:
    first, _ = _v2_controllers()
    second, _ = _v2_controllers()
    assert first.identity == second.identity
    for step in range(4):
        sensors = _sensors(left=6.0 - step, light=(step * 5) % 16, horizontal_collision=step == 2)
        action_a = first.step(_observation(first, step, sensors), command_seq=step)
        action_b = second.step(_observation(second, step, sensors), command_seq=step)
        assert action_a.state_digest == action_b.state_digest
        evidence_a = first.last_evidence
        evidence_b = second.last_evidence
        assert evidence_a is not None and evidence_b is not None
        assert evidence_a.action_sha256 == evidence_b.action_sha256
        assert evidence_a.encoded_stimulation_sha256 == evidence_b.encoded_stimulation_sha256


def test_reset_clears_the_running_means() -> None:
    v2, _ = _v2_controllers()
    v2.step(_observation(v2, 0, _sensors(light=15)), command_seq=0)
    v2.step(_observation(v2, 1, _sensors(light=0)), command_seq=1)
    evidence = v2.last_evidence
    assert evidence is not None
    assert dict(evidence.encoded_stimulation)["eem:light"] == 1.0  # 0.0 against baseline 1.0
    v2.reset()
    # Session clocks keep advancing after reset; only neural + contrast state clears.
    v2.step(_observation(v2, 2, _sensors(light=0)), command_seq=2)
    evidence = v2.last_evidence
    assert evidence is not None
    encoded = dict(evidence.encoded_stimulation)
    assert encoded["eem:light"] == 0.0  # first-tick semantics again
    assert encoded["eem_mean:light"] == 0.0


def test_v2_identity_rotates_against_v1_and_the_wing_boundary_is_untouched() -> None:
    v2, v1 = _v2_controllers()
    _, wing = _controllers()
    assert v2.identity.adapter_hash != v1.identity.adapter_hash
    assert v2.identity.adapter_hash != wing.identity.adapter_hash
    assert v2.identity.decoder_hash == v1.identity.decoder_hash == wing.identity.decoder_hash
    assert v2.identity.dynamics_hash == v1.identity.dynamics_hash
    assert v2.identity.graph_hash == v1.identity.graph_hash
    assert v2.identity.observation_schema_hash == v1.identity.observation_schema_hash
    assert v2.configuration.identity_sha256 != v1.configuration.identity_sha256
    assert v2.configuration.identity_sha256 != wing.configuration.identity_sha256


def test_v2_stimulation_provenance_differs_from_v1_but_faithful_channels_match() -> None:
    v2, v1 = _v2_controllers()
    sensors = _sensors(left=1.5, light=7)
    v2.step(_observation(v2, 0, sensors), command_seq=0)
    v1.step(_observation(v1, 0, sensors), command_seq=0)
    evidence_v2 = v2.last_evidence
    evidence_v1 = v1.last_evidence
    assert evidence_v2 is not None and evidence_v1 is not None
    assert evidence_v2.encoded_stimulation_sha256 != evidence_v1.encoded_stimulation_sha256
    for key, value in evidence_v1.encoded_stimulation.items():
        if not key.startswith("eem"):
            assert evidence_v2.encoded_stimulation[key] == value
    # v1 drives the raw levels on tick 0; v2 drives zero contrast -- the
    # executed dense vectors differ, so the committed states must differ.
    assert evidence_v2.state_after_sha256 != evidence_v1.state_after_sha256


# ----------------------------------------------------- calibration cross-binds


def test_calibration_cross_binding_refuses_every_profile_pair() -> None:
    """Wing, eem-v1, and eem-v2 calibrations each bind only their own boundary."""

    documents = {
        "wing": load_readout_calibration(
            canonical_json_bytes(_calibration_document(encoder_sha256=None))
        ),
        "eem_v1": load_readout_calibration(
            canonical_json_bytes(_calibration_document(encoder_sha256=EEM_ENCODER_SHA))
        ),
        "eem_v2": load_readout_calibration(
            canonical_json_bytes(_calibration_document(encoder_sha256=EEM_V2_ENCODER_SHA))
        ),
    }
    boundaries = {"wing": "", "eem_v1": EEM_ENCODER_SHA, "eem_v2": EEM_V2_ENCODER_SHA}
    expected_refusals = {
        ("wing", "eem_v1"): "without an engineered-encoder",
        ("wing", "eem_v2"): "without an engineered-encoder",
        ("eem_v1", "wing"): "wing boundary refuses",
        ("eem_v2", "wing"): "wing boundary refuses",
        ("eem_v1", "eem_v2"): "different encoder document",
        ("eem_v2", "eem_v1"): "different encoder document",
    }
    for document_profile, calibration in documents.items():
        for boundary_profile, encoder_identity in boundaries.items():
            if document_profile == boundary_profile:
                calibration.assert_binds(
                    artifact_manifest_sha256=GRAPH_SHA,
                    sign_overlay_identity_sha256=OVERLAY_SHA,
                    encoder_identity_sha256=encoder_identity,
                )
                continue
            with pytest.raises(
                FullBrainProfileError,
                match=expected_refusals[(document_profile, boundary_profile)],
            ):
                calibration.assert_binds(
                    artifact_manifest_sha256=GRAPH_SHA,
                    sign_overlay_identity_sha256=OVERLAY_SHA,
                    encoder_identity_sha256=encoder_identity,
                )


# -------------------------------------------------- from_artifact argument gates


def test_from_artifact_refuses_bad_encoder_versions(tmp_path: Path) -> None:
    """Version gates fire before any artifact byte is read."""

    with pytest.raises(FullBrainProfileError, match="must be 1 or 2"):
        FullBrainController.from_artifact(
            run_id="refusal",
            entity_id="fly",
            artifact_directory=tmp_path / "absent",
            annotations_path=tmp_path / "absent.feather",
            embodied_senses=True,
            embodied_encoder_version=3,
            body_annotations_path=tmp_path / "absent-annotations.feather",
        )
    with pytest.raises(FullBrainProfileError, match="embodied-senses profile only"):
        FullBrainController.from_artifact(
            run_id="refusal",
            entity_id="fly",
            artifact_directory=tmp_path / "absent",
            annotations_path=tmp_path / "absent.feather",
            embodied_encoder_version=2,
        )
    with pytest.raises(FullBrainProfileError, match="full_mcns_eem_v2 is a local-only"):
        FullBrainController.from_artifact(
            run_id="refusal",
            entity_id="fly",
            artifact_directory=tmp_path / "absent",
            annotations_path=tmp_path / "absent.feather",
            embodied_senses=True,
            embodied_encoder_version=2,
            body_annotations_path=tmp_path / "absent-annotations.feather",
        )
