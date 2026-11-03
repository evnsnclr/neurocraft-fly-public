from __future__ import annotations

import pytest
from neurocraft_connectome import RuntimeGraph
from neurocraft_controller import (
    ControllerIntegrityError,
    causal_record_from_applied_action,
    world_state_from_observation,
)
from neurocraft_protocol import ValidationResult, WorldState

from .conftest import make_controller, make_observation


def test_trace_factory_refuses_decorative_evidence(controller_graph: RuntimeGraph) -> None:
    controller = make_controller(controller_graph)
    observation = make_observation(controller, 0)
    with pytest.raises(ControllerIntegrityError, match="no controller step"):
        causal_record_from_applied_action(
            controller,
            record_seq=0,
            observation=observation,
            world_after=world_state_from_observation(observation),
            world_after_observation_seq=observation.observation_seq + 1,
            world_after_game_tick=observation.game_tick + 1,
        )


def test_trace_factory_binds_full_controller_chain(controller_graph: RuntimeGraph) -> None:
    controller = make_controller(controller_graph)
    observation = make_observation(controller, 0)
    action = controller.step(observation, command_seq=0)
    after = WorldState(
        position_m=(0.0, 64.0, 0.01),
        velocity_m_s=(0.0, 0.0, 0.01),
        yaw_rad=0.0,
        pitch_rad=0.0,
        on_ground=False,
        horizontal_collision=False,
        vertical_collision=False,
        in_fluid=False,
    )

    record = causal_record_from_applied_action(
        controller,
        record_seq=0,
        observation=observation,
        world_after=after,
        world_after_observation_seq=observation.observation_seq + 1,
        world_after_game_tick=observation.game_tick + 1,
        compute_ns=123,
    )
    evidence = controller.last_evidence
    assert evidence is not None

    assert record.identity == controller.identity
    assert record.observation == observation
    assert record.command_seq == action.command_seq
    assert record.neural_step_end - record.neural_step_start == 5
    assert record.state_after_digest == action.state_digest
    assert record.motor_pool_values == dict(evidence.motor_frame.values)
    assert record.decoder_output == action.motors
    assert record.applied_action == action.motors
    assert record.validation is ValidationResult.APPLIED
    assert record.world_after == after
    assert record.world_after_observation_seq == observation.observation_seq + 1
    assert record.world_after_game_tick == observation.game_tick + 1
    assert record.compute_ns == 123


def test_trace_factory_rejects_substituted_observation(controller_graph: RuntimeGraph) -> None:
    controller = make_controller(controller_graph)
    executed = make_observation(controller, 0)
    controller.step(executed, command_seq=0)
    substituted = make_observation(controller, 1)
    with pytest.raises(ControllerIntegrityError, match="does not match executed"):
        causal_record_from_applied_action(
            controller,
            record_seq=0,
            observation=substituted,
            world_after=world_state_from_observation(substituted),
            world_after_observation_seq=substituted.observation_seq + 1,
            world_after_game_tick=substituted.game_tick + 1,
        )
