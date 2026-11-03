"""Cached immutable declarations must preserve actual stimulation, state and commands."""

from dataclasses import fields, replace
from hashlib import sha256

import numpy as np
import pytest
from neurocraft_controller import embodied_senses, interaction_senses
from neurocraft_controller.config import canonical_sha256
from neurocraft_controller.embodied_senses import EmbodiedPopulation, EmbodiedPopulationBinding
from neurocraft_controller.events_senses import (
    EventPopulation,
    EventPopulationBinding,
    EventSensesError,
)
from neurocraft_controller.full_brain_controller import FullBrainController, FullBrainProfileError
from neurocraft_controller.interaction_program_controller import (
    InteractionProgramSelectingController,
)
from neurocraft_controller.interaction_senses import InteractionEncoderBinding
from neurocraft_neural.full_engine import ArrayNativeEngine
from neurocraft_neural.spmv_kernel import NumpyReduceatKernel
from neurocraft_protocol import Observation

from scripts.probe_interaction_inputs import sensor_frame
from tests.controller.test_interaction_senses import binding, paired_controllers


class ReferencePopulations(EmbodiedPopulationBinding):
    """The pre-cache property, evaluated from the same declared source document."""

    __slots__ = ()

    @property
    def identity_sha256(self):
        return canonical_sha256(self.to_identity_material())


class ReferenceInteraction(InteractionEncoderBinding):
    __slots__ = ()

    @property
    def identity_sha256(self):
        return canonical_sha256(self.to_identity_material())


def constructor_fields(value):
    return {f.name: getattr(value, f.name) for f in fields(value) if f.init}


def controllers(mode):
    cached, _, arguments = paired_controllers(mode)
    populations = ReferencePopulations(**constructor_fields(cached.embodied_senses.populations))
    embodied = replace(cached.embodied_senses, populations=populations)
    encoder = ReferenceInteraction(
        **{**constructor_fields(cached.event_encoder), "embodied_populations": populations}
    )
    direct = FullBrainController(
        engine=ArrayNativeEngine(cached.engine.view, kernel=NumpyReduceatKernel()),
        configuration=cached.configuration,
        event_encoder=encoder,
        **{**arguments, "embodied": embodied},
    )
    return InteractionProgramSelectingController(cached), InteractionProgramSelectingController(
        direct
    )


def test_population_membership_and_outer_lists_are_copied_before_identity_cache():
    bodies, indices = [10, 20], [0, 1]
    population = EmbodiedPopulation("fixture", "fixture selector", bodies, indices)
    groups = [population]
    bound = EmbodiedPopulationBinding("a" * 64, "b" * 64, 2, groups)
    before, identity = bound.to_identity_material(), bound.identity_sha256
    bodies[0], indices[0] = 99, 99
    groups.clear()
    assert bound.populations == (population,)
    assert population.body_ids == (10, 20) and population.dense_indices == (0, 1)
    assert bound.to_identity_material() == before
    assert bound.identity_sha256 == identity == canonical_sha256(before)
    # Returned material is also independent; callers cannot mutate cached content.
    material = bound.to_identity_material()
    material["populations"][0]["members"][0][0] = 999
    assert bound.to_identity_material() == before


def test_interaction_cache_cannot_alias_nested_event_membership_lists():
    original = binding()
    rows = [
        [list(row) for row in population.members] for population in original.populations.populations
    ]
    groups = [
        EventPopulation(pop.name, members)
        for pop, members in zip(original.populations.populations, rows, strict=True)
    ]
    events = EventPopulationBinding(
        original.populations.graph_hash,
        original.populations.annotations_hash,
        original.populations.node_count,
        groups,
    )
    bound = replace(original, populations=events)
    before, identity = bound.to_identity_material(), bound.identity_sha256
    rows[0][0][0] += 500
    rows[1].clear()
    groups.clear()
    assert bound.to_identity_material() == before
    assert identity == bound.identity_sha256 == canonical_sha256(before)
    assert identity == original.identity_sha256


def test_static_hashes_execute_once_per_instance_and_replacement_recomputes(monkeypatch):
    original = binding()
    population_calls, encoder_calls = [], []

    def population_hash(material):
        population_calls.append(material["schema"])
        return canonical_sha256(material)

    def encoder_hash(material):
        encoder_calls.append(material["schema"])
        return canonical_sha256(material)

    monkeypatch.setattr(embodied_senses, "canonical_sha256", population_hash)
    monkeypatch.setattr(interaction_senses, "canonical_sha256", encoder_hash)
    populations = replace(original.embodied_populations)
    encoder = replace(original, embodied_populations=populations)
    assert len(population_calls) == len(encoder_calls) == 1
    for _ in range(10):
        assert populations.identity_sha256 == canonical_sha256(populations.to_identity_material())
        assert encoder.identity_sha256 == canonical_sha256(encoder.to_identity_material())
    assert len(population_calls) == len(encoder_calls) == 1
    changed_populations = replace(populations, graph_manifest_sha256="f" * 64)
    assert changed_populations.identity_sha256 != populations.identity_sha256
    assert len(population_calls) == 2
    changed_encoder = replace(encoder, component_decoder_sha256="f" * 64)
    assert changed_encoder.identity_sha256 != encoder.identity_sha256
    assert len(encoder_calls) == 2
    assert encoder.to_identity_material() == original.to_identity_material()


@pytest.mark.parametrize("mode", ["faithful", "shuffled_weight", "no_input"])
def test_cached_and_reference_actual_commands_states_and_stimulation_across_events_resets(mode):
    cached, reference = controllers(mode)
    assert cached.identity == reference.identity
    assert cached.policy_hash == reference.policy_hash
    for epoch, case in enumerate(
        (
            "bright_sham",
            "brush",
            "attack",
            "food",
            "player_approach",
            "mob_approach",
            "light_off",
            "light_on",
        ),
        start=1,
    ):
        for controller in (cached, reference):
            controller.reset_session(session_epoch=epoch)
        for tick in range(30):
            observation = Observation(
                cached.identity, tick, tick, tick * 50_000_000, sensor_frame(case, tick)
            )
            actual = cached.step(observation, command_seq=tick)
            expected = reference.step(observation, command_seq=tick)
            assert actual == expected
            a, b = cached.controller.last_evidence, reference.controller.last_evidence
            assert a.encoded_stimulation_sha256 == b.encoded_stimulation_sha256
            assert a.executed_stimulation_sha256 == b.executed_stimulation_sha256
            assert a.state_before_sha256 == b.state_before_sha256
            assert a.state_after_sha256 == b.state_after_sha256
            assert cached.last_event_evidence.to_dict() == reference.last_event_evidence.to_dict()
            assert actual.neural_step_start == tick * 5 and actual.neural_step_end == (tick + 1) * 5
            np.testing.assert_array_equal(
                cached.controller.engine.activity(), reference.controller.engine.activity()
            )
            if mode == "no_input":
                zero = sha256(bytes(cached.controller.engine.view.node_count * 8)).hexdigest()
                assert a.executed_stimulation_sha256 == zero
                assert cached.controller.engine.current_state().activity_sha256 == zero
        # In-world reset clears neural/policy history while transport clocks continue.
        for controller in (cached, reference):
            controller.reset()
        observation = Observation(cached.identity, 30, 30, 30 * 50_000_000, sensor_frame(case, 29))
        assert cached.step(observation, command_seq=30) == reference.step(
            observation, command_seq=30
        )
        assert cached.controller.engine.step_index == reference.controller.engine.step_index == 5


def test_hot_steps_do_not_rebuild_cached_documents_but_dynamic_hashes_still_change(monkeypatch):
    wrapper, _ = controllers("faithful")
    original_population_hash = embodied_senses.canonical_sha256
    original_encoder_hash = interaction_senses.canonical_sha256
    population_calls, encoder_calls = [], []

    def population_hash(material):
        if material["schema"] == embodied_senses.EEM_POPULATION_BINDING_SCHEMA:
            population_calls.append(material)
        return original_population_hash(material)

    def encoder_hash(material):
        encoder_calls.append(material)
        return original_encoder_hash(material)

    monkeypatch.setattr(embodied_senses, "canonical_sha256", population_hash)
    monkeypatch.setattr(interaction_senses, "canonical_sha256", encoder_hash)
    stimulation, states = set(), set()
    for tick in range(26):
        obs = Observation(
            wrapper.identity, tick, tick, tick * 50_000_000, sensor_frame("attack", tick)
        )
        wrapper.step(obs, command_seq=tick)
        evidence = wrapper.controller.last_evidence
        stimulation.add(evidence.encoded_stimulation_sha256)
        states.add(evidence.state_after_sha256)
    assert not population_calls and not encoder_calls
    assert len(stimulation) > 1 and len(states) == 26
    assert wrapper.controller.engine.step_index == 130


def test_cached_bindings_keep_constructor_identity_and_source_mismatch_refusals():
    core, _, arguments = paired_controllers()
    with pytest.raises(EventSensesError, match="share"):
        replace(
            core.event_encoder,
            embodied_populations=replace(
                core.embodied_senses.populations, annotations_sha256="f" * 64
            ),
        )
    wrong = replace(core.event_encoder, component_decoder_sha256="f" * 64)
    with pytest.raises(FullBrainProfileError, match="identity"):
        FullBrainController(
            configuration=core.configuration, engine=core.engine, event_encoder=wrong, **arguments
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"graph_manifest_sha256": []},
        {"annotations_sha256": {}},
        {"node_count": [2]},
    ],
)
def test_cache_refuses_mutable_scalar_metadata(changes):
    from neurocraft_controller.embodied_senses import EmbodiedSensesError

    population = EmbodiedPopulation("fixture", "fixture selector", (10,), (0,))
    with pytest.raises(EmbodiedSensesError, match="immutable"):
        EmbodiedPopulationBinding(
            **{
                "graph_manifest_sha256": "a" * 64,
                "annotations_sha256": "b" * 64,
                "node_count": 2,
                "populations": (population,),
                **changes,
            }
        )


def test_cache_refuses_mutable_membership_values_and_nonbinding_objects():
    from types import SimpleNamespace

    from neurocraft_controller.embodied_senses import EmbodiedSensesError

    with pytest.raises(EmbodiedSensesError, match="immutable integers"):
        EmbodiedPopulation("fixture", "fixture selector", [[10]], [[0]])
    original = binding()
    with pytest.raises(EventSensesError, match="immutable verified"):
        replace(
            original,
            embodied_populations=SimpleNamespace(
                **constructor_fields(original.embodied_populations)
            ),
        )
