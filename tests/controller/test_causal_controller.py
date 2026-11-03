from __future__ import annotations

from dataclasses import replace

import pytest
from neurocraft_connectome import RuntimeGraph, SyntheticEdge, SyntheticNode, build_synthetic_graph
from neurocraft_controller import (
    ConnectomeController,
    ControllerConfiguration,
    ControllerIntegrityError,
)
from neurocraft_neural import NeuralConfigurationError, Stimulation
from neurocraft_protocol import Action, RunIdentity

from .conftest import make_controller, make_observation, make_sensors


def _advance(
    controller: ConnectomeController,
    *,
    count: int,
    start: int = 0,
    zero_input: bool = False,
) -> list[Action]:
    sensors = (
        make_sensors(forward=0.0, left=0.0, right=0.0, light=0) if zero_input else make_sensors()
    )
    return [
        controller.step(make_observation(controller, index, sensors=sensors), command_seq=index)
        for index in range(start, start + count)
    ]


def test_real_input_reaches_neural_state_and_primitive_motor(
    controller_graph: RuntimeGraph,
) -> None:
    driven = make_controller(controller_graph)
    zero = make_controller(controller_graph)

    driven_actions = _advance(driven, count=4)
    zero_actions = _advance(zero, count=4, zero_input=True)
    driven_evidence = driven.last_evidence
    zero_evidence = zero.last_evidence

    assert driven_evidence is not None
    assert zero_evidence is not None
    assert driven_evidence.feature_values_sha256 != zero_evidence.feature_values_sha256
    assert driven_evidence.encoded_stimulation_sha256 != zero_evidence.encoded_stimulation_sha256
    assert driven_evidence.state_after_sha256 != zero_evidence.state_after_sha256
    assert driven_evidence.motor_frame.neural_state_sha256 == driven_evidence.state_after_sha256
    assert driven_actions[-1].state_digest == driven_evidence.state_after_sha256
    assert not driven_actions[-1].motors.is_neutral
    assert zero_actions[-1].motors.is_neutral
    assert driven_actions[-1].neural_step_end - driven_actions[-1].neural_step_start == 5
    assert len(driven_evidence.neural_step_records) == 5


def test_fixture_sign_ablation_removes_recurrent_sensor_to_motor_effect(
    controller_graph: RuntimeGraph,
) -> None:
    """AB-02/AB-11: the faithful output must depend on the signed recurrence."""

    prefix = make_sensors(forward=6.0, left=1.5, right=6.0, light=15)
    blocked = make_sensors(
        forward=1.499980129301548,
        left=1.5,
        right=6.0,
        light=15,
    )
    faithful = make_controller(controller_graph, run_id="mutation-faithful")
    zero_sign = ConnectomeController.create(
        controller_graph,
        run_id="mutation-zero-sign",
        entity_id="test-fly",
        configuration=ControllerConfiguration(sign_policy="unknown_zero"),
    )

    for sequence, sensors in enumerate((prefix, prefix, blocked)):
        faithful_action = faithful.step(
            make_observation(faithful, sequence, sensors=sensors),
            command_seq=sequence,
        )
        zero_action = zero_sign.step(
            make_observation(zero_sign, sequence, sensors=sensors),
            command_seq=sequence,
        )

    assert not faithful_action.motors.is_neutral
    assert zero_action.motors.is_neutral
    assert faithful_action.state_digest != zero_action.state_digest


def test_fixture_forward_ray_perturbation_propagates_from_identical_state(
    controller_graph: RuntimeGraph,
) -> None:
    """AB-06/AB-07: one physical sensor change must alter state and motors."""

    prefix = make_sensors(forward=6.0, left=1.5, right=6.0, light=15)
    blocked = make_sensors(
        forward=1.499980129301548,
        left=1.5,
        right=6.0,
        light=15,
    )
    actual = make_controller(controller_graph, run_id="mutation-ray-actual")
    open_counterfactual = make_controller(
        controller_graph,
        run_id="mutation-ray-counterfactual",
    )
    for sequence in range(2):
        actual.step(
            make_observation(actual, sequence, sensors=prefix),
            command_seq=sequence,
        )
        open_counterfactual.step(
            make_observation(open_counterfactual, sequence, sensors=prefix),
            command_seq=sequence,
        )

    assert actual.engine.current_state() == open_counterfactual.engine.current_state()
    actual_action = actual.step(
        make_observation(actual, 2, sensors=blocked),
        command_seq=2,
    )
    open_action = open_counterfactual.step(
        make_observation(open_counterfactual, 2, sensors=prefix),
        command_seq=2,
    )

    assert actual.last_evidence is not None
    assert open_counterfactual.last_evidence is not None
    assert (
        actual.last_evidence.state_before_sha256
        == open_counterfactual.last_evidence.state_before_sha256
    )
    assert actual_action.state_digest != open_action.state_digest
    assert actual_action.motors != open_action.motors


def test_fixed_inputs_reproduce_actions_and_hash_chain(controller_graph: RuntimeGraph) -> None:
    first = make_controller(controller_graph)
    second = make_controller(controller_graph)

    first_actions = _advance(first, count=10)
    second_actions = _advance(second, count=10)

    assert first_actions == second_actions
    assert first.last_evidence == second.last_evidence
    first.reset()
    replayed = _advance(first, count=10, start=10)
    assert [(item.state_digest, item.motors) for item in replayed] == [
        (item.state_digest, item.motors) for item in first_actions
    ]
    assert replayed[0].command_seq == 10
    assert replayed[0].neural_step_start == 50


def test_lesion_changes_state_and_output(controller_graph: RuntimeGraph) -> None:
    intact = make_controller(controller_graph, mode="lesion")
    lesioned = make_controller(controller_graph, mode="lesion")
    lesioned.set_lesion(
        (
            lesioned.configuration.left_sensor_body_id,
            lesioned.configuration.right_sensor_body_id,
        )
    )

    intact_actions = _advance(intact, count=4)
    lesioned_actions = _advance(lesioned, count=4)

    assert not intact_actions[-1].motors.is_neutral
    assert lesioned_actions[-1].motors.is_neutral
    assert intact_actions[-1].state_digest != lesioned_actions[-1].state_digest


def test_named_stimulation_changes_neural_output(controller_graph: RuntimeGraph) -> None:
    control = make_controller(controller_graph)
    stimulated = make_controller(controller_graph)
    stimulated.set_stimulation(
        "test-motor-stimulation",
        {stimulated.configuration.left_motor_body_id: 1.0},
    )

    control_action = _advance(control, count=1, zero_input=True)[0]
    stimulated_action = _advance(stimulated, count=1, zero_input=True)[0]

    assert control_action.motors.is_neutral
    assert not stimulated_action.motors.is_neutral
    assert stimulated.last_evidence is not None
    assert stimulated.last_evidence.interventions == ("test-motor-stimulation",)
    body_id = stimulated.configuration.left_motor_body_id
    expected = Stimulation.from_body_values(
        controller_graph,
        "test-motor-stimulation",
        {body_id: 1.0},
    )
    assert dict(stimulated.last_evidence.direct_stimulation) == {str(body_id): 1.0}
    assert stimulated.last_evidence.direct_stimulation_sha256 == expected.identity_sha256
    assert all(
        record.direct_stimulation_sha256 == expected.identity_sha256
        for record in stimulated.last_evidence.neural_step_records
    )


def test_identity_missing_input_and_sequence_fail_closed(controller_graph: RuntimeGraph) -> None:
    controller = make_controller(controller_graph)
    wrong_identity = replace(controller.identity, graph_hash="f" * 64)
    wrong = replace(make_observation(controller, 0), identity=wrong_identity)
    with pytest.raises(ControllerIntegrityError, match="identity"):
        controller.step(wrong, command_seq=0)

    missing = make_observation(controller, 0, missing_fields=("local_light",))
    with pytest.raises(ControllerIntegrityError, match="missing"):
        controller.step(missing, command_seq=0)

    controller.step(make_observation(controller, 0), command_seq=0)
    with pytest.raises(ControllerIntegrityError, match="stale"):
        controller.step(make_observation(controller, 0), command_seq=1)
    with pytest.raises(ControllerIntegrityError, match="command sequence"):
        controller.step(make_observation(controller, 1), command_seq=0)


def test_synthetic_graph_cannot_silently_replace_faithful_graph() -> None:
    graph = build_synthetic_graph(
        "synthetic:anti-fallback",
        [
            SyntheticNode(1, ("sensory",)),
            SyntheticNode(2, ("motor",)),
        ],
        [SyntheticEdge(1, 2, 1, 0)],
    )
    with pytest.raises(ControllerIntegrityError, match="cannot replace"):
        ConnectomeController.create(graph, run_id="bad", entity_id="fly")


def test_decoder_and_engine_reject_raw_observation_bypass(
    controller_graph: RuntimeGraph,
) -> None:
    controller = make_controller(controller_graph)
    observation = make_observation(controller, 0)
    with pytest.raises(NeuralConfigurationError, match="NeuralState"):
        controller.decoder.decode(observation)  # type: ignore[arg-type]
    with pytest.raises(NeuralConfigurationError, match="EncodedStimulation"):
        controller.engine.step(observation)  # type: ignore[arg-type]


def test_run_identity_binds_every_controller_boundary(controller_graph: RuntimeGraph) -> None:
    controller = make_controller(controller_graph)
    identity = controller.identity

    assert identity.graph_hash == controller_graph.manifest.runtime_identity_sha256
    assert identity.adapter_hash == controller.adapter_identity_sha256
    assert identity.dynamics_hash == controller.dynamics_identity_sha256
    assert identity.decoder_hash == controller.decoder_identity_sha256
    assert identity.observation_schema_hash == controller.features.identity_sha256
    assert controller.projection_identity_sha256 != controller.decoder.identity_sha256


def test_constructor_does_not_accept_caller_supplied_identity(
    controller_graph: RuntimeGraph,
) -> None:
    fake = RunIdentity(
        run_id="fake",
        session_epoch=0,
        entity_id="fly",
        graph_hash="0" * 64,
        adapter_hash="0" * 64,
        dynamics_hash="0" * 64,
        decoder_hash="0" * 64,
        observation_schema_hash="0" * 64,
    )
    with pytest.raises(TypeError):
        ConnectomeController(  # type: ignore[call-arg]
            controller_graph,
            run_id="real",
            entity_id="fly",
            identity=fake,
        )


def test_configuration_rejects_scripted_faithful_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        ControllerConfiguration(mode="scripted_baseline")
