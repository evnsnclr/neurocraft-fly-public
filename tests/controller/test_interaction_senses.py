"""Separate candidate sensory drives; these tests do not select or apply body programs."""

from dataclasses import fields, replace

import numpy as np
import pytest
from neurocraft_controller.controller import ControllerIntegrityError
from neurocraft_controller.events_senses import EventInputState, EventSensesError
from neurocraft_controller.full_brain_controller import (
    FullBrainController,
    FullBrainInteractionsConfiguration,
    FullBrainProfileError,
)
from neurocraft_controller.interaction_senses import (
    InteractionEncoderBinding,
    InteractionInputState,
)
from neurocraft_neural.full_engine import ArrayNativeEngine
from neurocraft_neural.spmv_kernel import NumpyReduceatKernel
from neurocraft_protocol import Observation, SensorFrameV4

from tests.controller.test_events_senses import controllers, frame, sample


def binding(core=None):
    if core is None:
        core, _ = controllers()
    base = core.event_encoder
    embodied = core.embodied_senses.populations
    return InteractionEncoderBinding(
        base.populations,
        base.background_encoder_sha256,
        base.component_decoder_sha256,
        base.component_calibration_sha256,
        embodied,
        expected_bristle_count=len(embodied.population("bristle").body_ids),
    )


def paired_controllers(mode="faithful"):
    old, _ = controllers(mode)
    bound = binding(old)
    config = FullBrainInteractionsConfiguration(
        **{
            f.name: getattr(old.configuration, f.name)
            for f in fields(old.configuration)
            if f.name not in {"schema", "event_encoder_sha256"}
        },
        event_encoder_sha256=bound.identity_sha256,
    )
    arguments = dict(
        run_id=old.identity.run_id,
        entity_id=old.identity.entity_id,
        sensory_binding=old._sensory_binding,
        motor_binding=old._motor_binding,
        embodied=old.embodied_senses,
        escape=old.escape_binding,
        steering=old.steering_binding,
        control=old._control,
    )
    candidate = FullBrainController(
        configuration=config,
        event_encoder=bound,
        engine=ArrayNativeEngine(old.engine.view, kernel=NumpyReduceatKernel()),
        **arguments,
    )
    return candidate, old, arguments


def test_brush_attack_and_simultaneous_events_have_distinct_declared_vectors():
    bound = binding()
    results = {}
    for name, events in (
        ("brush", sample(brush=1, tick=4)),
        ("attack", sample(attack=1, tick=4)),
        ("both", sample(brush=100, attack=200, tick=4)),
    ):
        state = InteractionInputState(bound)
        state.prepare(sample(), 0)
        pulse = state.prepare(events, 4)
        vector = np.zeros(bound.populations.node_count)
        state.apply(vector, pulse)
        results[name] = vector
        # Counter jumps carry one pulse; repeated observations do not replay it.
        repeated = state.prepare(events, 5)
        assert not repeated.groom and not repeated.attack
        assert vector.max() == 1
    groom = bound.populations.population("groom").indices
    bristles = bound.embodied_populations.population("bristle").dense_indices
    assert np.count_nonzero(results["brush"]) == len(groom)
    assert np.count_nonzero(results["attack"]) == len(bristles)
    assert np.all(results["attack"][list(groom)] == 0)
    assert np.all(results["brush"][list(bristles)] == 0)
    np.testing.assert_array_equal(results["both"], results["brush"] + results["attack"])


def test_old_encoder_still_merges_and_new_binding_rotates_only_its_declaration():
    core, _ = controllers()
    bound = binding()
    assert bound.identity_sha256 != core.event_encoder.identity_sha256
    assert bound.populations == core.event_encoder.populations
    old = EventInputState(core.event_encoder)
    old.prepare(sample(), 0)
    assert old.prepare(sample(attack=1, tick=1), 1).groom
    new = InteractionInputState(bound)
    new.prepare(sample(), 0)
    assert not new.prepare(sample(attack=1, tick=1), 1).groom


def test_stale_and_simultaneous_ages_are_independent_and_reset_absorbs_history():
    state = InteractionInputState(binding())
    state.prepare(sample(), 0)
    state.prepare(sample(brush=1, tick=2), 2)
    events = replace(
        sample(brush=1, tick=2), attack_attempt=sample(attack=1, tick=9).attack_attempt
    )
    pulse = state.prepare(events, 10)
    assert pulse.attack and pulse.attack_age == 1
    assert not pulse.groom and pulse.groom_age is None
    stale = state.prepare(
        replace(events, attack_attempt=sample(attack=2, tick=9).attack_attempt), 18
    )
    assert not stale.attack
    state.reset()
    absorbed = state.prepare(events, 19)
    assert absorbed.baseline_absorbed and not absorbed.attack and not absorbed.groom


def test_binding_refuses_different_source_identity_count_and_event_overlap():
    bound = binding()
    with pytest.raises(EventSensesError, match="share"):
        replace(
            bound,
            embodied_populations=replace(bound.embodied_populations, annotations_sha256="f" * 64),
        )
    with pytest.raises(EventSensesError, match="count"):
        replace(bound, expected_bristle_count=bound.expected_bristle_count + 1)
    original = bound.embodied_populations
    bristles = original.population("bristle")
    groom = bound.populations.population("groom").members[0]
    altered = replace(bristles, dense_indices=(groom[1], *bristles.dense_indices[1:]))
    with pytest.raises(EventSensesError, match="disjoint"):
        replace(
            bound,
            embodied_populations=replace(
                original,
                populations=tuple(
                    altered if pop.name == "bristle" else pop for pop in original.populations
                ),
            ),
        )


@pytest.mark.parametrize("kind", ["brush", "attack", "both"])
def test_actual_engine_uses_declared_separate_pulses_and_five_updates(kind, monkeypatch):
    candidate, old, _ = paired_controllers()
    vectors = [[], []]
    for core, captured in zip((candidate, old), vectors, strict=True):
        real_step = core.engine.step

        def capture(*, sensory, sensory_provenance_sha256, real_step=real_step, captured=captured):
            captured.append(sensory.copy())
            return real_step(sensory=sensory, sensory_provenance_sha256=sensory_provenance_sha256)

        monkeypatch.setattr(core.engine, "step", capture)
    for tick in range(7):
        events = sample(
            brush=int(tick >= 4 and kind in {"brush", "both"}),
            attack=int(tick >= 4 and kind in {"attack", "both"}),
            tick=4,
        )
        legacy = frame(events, horizontal_collision=tick >= 4)
        sensors = SensorFrameV4.from_event_frame(legacy, (0,) * 6)
        for core, sensor in ((candidate, sensors), (old, legacy)):
            core.step(
                Observation(core.identity, tick, tick, tick * 50_000_000, sensor), command_seq=tick
            )
    increment = np.zeros(candidate.engine.view.node_count)
    if kind in {"attack", "both"}:
        increment[
            list(candidate.embodied_senses.populations.population("bristle").dense_indices)
        ] += 1
    if kind == "attack":
        increment[list(candidate.event_encoder.populations.population("groom").indices)] -= 1
    for index, (new, original) in enumerate(zip(*vectors, strict=True)):
        np.testing.assert_allclose(
            new, original + (increment if 20 <= index < 25 else 0), rtol=0, atol=1e-15
        )
    assert candidate.engine.step_index == old.engine.step_index == 35
    assert candidate.identity.graph_hash == old.identity.graph_hash
    assert candidate.identity.dynamics_hash == old.identity.dynamics_hash
    assert candidate.identity.decoder_hash == old.identity.decoder_hash
    assert candidate.identity.adapter_hash != old.identity.adapter_hash
    assert candidate.identity.observation_schema_hash != old.identity.observation_schema_hash


def test_no_input_gate_and_reset_cover_new_attack_input_and_actor_quality():
    core, _, _ = paired_controllers("no_input")
    histories = []
    for epoch in (1, 2):
        core.reset_session(session_epoch=epoch)
        hashes = []
        for tick in range(7):
            sensors = SensorFrameV4.from_event_frame(
                frame(sample(attack=int(tick >= 4), tick=4)), (0,) * 6
            )
            obs = Observation(core.identity, tick, tick, tick * 50_000_000, sensors)
            if tick == 0:
                with pytest.raises(ControllerIntegrityError, match="actor metadata"):
                    core.step(replace(obs, clamped_fields=("ray_actor_kinds",)), command_seq=tick)
                assert core.engine.step_index == 0
            core.step(obs, command_seq=tick)
            np.testing.assert_array_equal(core.engine.activity(), 0)
            hashes.append(core.last_evidence.state_after_sha256)
            if tick == 4:
                assert core.last_event_readout.pulse.attack
                assert core.last_evidence.encoder_stimulation_before_control["event:attack"] == 1
                assert core.last_evidence.encoded_stimulation["event:attack"] == 0
        histories.append(hashes)
    assert histories[0] == histories[1]


def test_configuration_version_cannot_mislabel_existing_event_encoder():
    core, old, arguments = paired_controllers()
    with pytest.raises(FullBrainProfileError, match="must be paired"):
        FullBrainController(
            configuration=core.configuration,
            event_encoder=old.event_encoder,
            engine=old.engine,
            **arguments,
        )
    with pytest.raises(FullBrainProfileError, match="must be paired"):
        FullBrainController(
            configuration=old.configuration,
            event_encoder=core.event_encoder,
            engine=core.engine,
            **arguments,
        )
