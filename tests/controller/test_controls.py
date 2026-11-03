from __future__ import annotations

import pytest
from neurocraft_connectome import RuntimeGraph
from neurocraft_controller import (
    ConnectomeControlHandler,
    ConnectomeController,
    ControllerIntegrityError,
    InterventionControlError,
    NamedLesion,
    NamedStimulation,
)
from neurocraft_neural import TopologyOverlay, WeightOverlay
from neurocraft_protocol import ControlCommand, ControlOperation, FaultCode, RunMode

from .conftest import make_controller, make_observation


def _command(
    controller: ConnectomeController,
    operation: ControlOperation,
    content_id: str,
    content_hash: str | None,
) -> ControlCommand:
    return ControlCommand(
        identity=controller.identity,
        control_seq=0,
        issued_tick=0,
        operation=operation,
        content_id=content_id,
        content_hash=content_hash,
    )


def test_named_lesion_registry_applies_and_clears_exact_hash(
    controller_graph: RuntimeGraph,
) -> None:
    controller = make_controller(controller_graph, mode="lesion")
    lesion = NamedLesion.create(
        controller,
        "left-sensory-lesion-v1",
        (controller.configuration.left_sensor_body_id,),
    )
    handler = ConnectomeControlHandler(controller, (lesion,))
    handler.handle_control(
        _command(
            controller,
            ControlOperation.LESION_APPLY,
            lesion.content_id,
            lesion.content_hash,
        )
    )
    assert controller.active_named_lesions == (lesion.content_id,)

    controller.step(make_observation(controller, 0), command_seq=0)
    assert controller.last_evidence is not None
    assert controller.last_evidence.interventions == (lesion.content_id,)

    handler.handle_control(
        _command(controller, ControlOperation.LESION_CLEAR, lesion.content_id, None)
    )
    assert not controller.active_named_lesions


def test_unknown_or_mismatched_intervention_rejects_without_mutation(
    controller_graph: RuntimeGraph,
) -> None:
    controller = make_controller(controller_graph, mode="lesion")
    lesion = NamedLesion.create(
        controller,
        "declared-lesion",
        (controller.configuration.left_sensor_body_id,),
    )
    handler = ConnectomeControlHandler(controller, (lesion,))
    with pytest.raises(InterventionControlError) as mismatch:
        handler.handle_control(
            _command(
                controller,
                ControlOperation.LESION_APPLY,
                lesion.content_id,
                "f" * 64,
            )
        )
    assert mismatch.value.code == FaultCode.CONTROL_UNAVAILABLE.value
    assert not controller.active_named_lesions

    with pytest.raises(InterventionControlError) as unknown:
        handler.handle_control(
            _command(
                controller,
                ControlOperation.LESION_APPLY,
                "unknown-lesion",
                "e" * 64,
            )
        )
    assert unknown.value.code == FaultCode.CONTROL_UNAVAILABLE.value
    assert not controller.active_named_lesions


def test_named_stimulation_lifecycle_and_reset_preserve_transport_monotonicity(
    controller_graph: RuntimeGraph,
) -> None:
    controller = make_controller(controller_graph)
    stimulation = NamedStimulation.create(
        controller,
        "left-motor-stimulation-v1",
        {controller.configuration.left_motor_body_id: 0.75},
    )
    handler = ConnectomeControlHandler(controller, (stimulation,))
    handler.handle_control(
        _command(
            controller,
            ControlOperation.STIMULATION_APPLY,
            stimulation.content_id,
            stimulation.content_hash,
        )
    )
    active_before = controller.active_stimulation_id
    assert active_before == stimulation.content_id
    controller.step(make_observation(controller, 0), command_seq=0)

    controller.reset()
    active_after = controller.active_stimulation_id
    assert active_after is None
    post_reset = controller.step(make_observation(controller, 1), command_seq=1)
    assert post_reset.command_seq == 1
    assert post_reset.observation_seq == 1
    assert post_reset.neural_step_start == 5
    assert post_reset.neural_step_end == 10
    assert controller.last_evidence is not None
    assert not controller.last_evidence.interventions


def test_weight_and_topology_controls_are_bound_into_controller_identity(
    controller_graph: RuntimeGraph,
) -> None:
    baseline = make_controller(controller_graph, run_id="overlay-control")
    weight = WeightOverlay.shuffled(
        controller_graph,
        seed=7,
        provenance="controller-test-shuffled-weight",
    )
    topology = TopologyOverlay.degree_preserving_random(
        controller_graph,
        seed=11,
        requested_swaps=1,
        provenance="controller-test-degree-preserving",
    )
    weight_controlled = baseline.create(
        controller_graph,
        run_id="overlay-control",
        entity_id="test-fly",
        weight_overlay=weight,
    )
    topology_controlled = baseline.create(
        controller_graph,
        run_id="overlay-control",
        entity_id="test-fly",
        topology_overlay=topology,
    )
    assert weight_controlled.engine.weight_overlay == weight
    assert topology_controlled.engine.topology_overlay == topology
    assert weight_controlled.identity.dynamics_hash != baseline.identity.dynamics_hash
    assert topology_controlled.identity.dynamics_hash != baseline.identity.dynamics_hash
    assert weight_controlled.run_mode is RunMode.SHUFFLED_WEIGHT
    assert topology_controlled.run_mode is RunMode.DEGREE_PRESERVING_RANDOM

    with pytest.raises(ControllerIntegrityError, match="must run separately"):
        baseline.create(
            controller_graph,
            run_id="combined-control",
            entity_id="test-fly",
            weight_overlay=weight,
            topology_overlay=topology,
        )
