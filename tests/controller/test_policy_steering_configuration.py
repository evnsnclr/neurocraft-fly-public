"""Declared steering parameters must change commands in every policy wrapper."""

from dataclasses import replace

import pytest
from neurocraft_controller.event_program_policy import EventProgramSelectingController
from neurocraft_controller.interaction_program_controller import (
    InteractionProgramSelectingController,
)
from neurocraft_controller.program_policy import ProgramPolicyError, ProgramSelectingController
from neurocraft_protocol import Observation

from tests.controller.test_events_senses import controllers, frame
from tests.controller.test_interaction_program_controller import observation
from tests.controller.test_interaction_senses import paired_controllers
from tests.controller.test_program_policy import _policy, _sensors, _synthetic_program_controller


@pytest.mark.parametrize("kind", ["program", "event", "interaction"])
@pytest.mark.parametrize(
    "parameters,expected",
    [
        ({"dna01_floor": 1.0, "turn_deadband": 0.0}, 0.001 / 1.001),
        ({"turn_deadband": 0.95}, 0.0),
    ],
)
def test_custom_steering_configuration_reaches_actual_issued_commands(
    kind, parameters, expected, monkeypatch
):
    if kind == "program":
        default = _synthetic_program_controller()
        core = default.controller
        constructor = ProgramSelectingController
        obs = Observation(core.identity, 0, 0, 0, _sensors())
    elif kind == "event":
        core, _ = controllers()
        default = EventProgramSelectingController(core)
        constructor = EventProgramSelectingController
        obs = Observation(core.identity, 0, 0, 0, frame())
    else:
        core, _, _ = paired_controllers()
        default = InteractionProgramSelectingController(core)
        constructor = InteractionProgramSelectingController
        obs = observation(core, "player_approach", 0)
    # Actual small engines still execute; replace only the two measured scalars
    # so this regression isolates the policy boundary from neural dynamics.
    monkeypatch.setattr(core, "steering_readout_means", lambda: (0.0, 0.001))
    varied = constructor(core, policy=replace(default.policy, **parameters))
    command = varied.step(obs, command_seq=0)
    assert varied.policy_hash != default.policy_hash
    assert command.readout.dna01_turn == pytest.approx(expected)
    assert command.parameters.turn_bias == pytest.approx(expected)


@pytest.mark.parametrize(
    "parameters",
    [
        {"dna01_floor": float("nan")},
        {"dna01_floor": float("inf")},
        {"dna01_floor": 0.0},
        {"dna01_floor": True},
        {"turn_deadband": float("nan")},
        {"turn_deadband": float("inf")},
        {"turn_deadband": -0.1},
        {"turn_deadband": 1.1},
        {"turn_deadband": True},
    ],
)
def test_invalid_steering_configuration_is_rejected_before_stepping(parameters):
    with pytest.raises(ProgramPolicyError):
        replace(_policy(), **parameters)
