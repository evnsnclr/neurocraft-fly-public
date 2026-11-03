"""V2 wrapper on real small engines; application and full-network results are separate."""

from __future__ import annotations

from dataclasses import replace

import pytest
from neurocraft_controller.events_senses import EventSensesError
from neurocraft_controller.interaction_program_controller import (
    InteractionProgramSelectingController,
)
from neurocraft_protocol import Observation, ProgramId

from scripts.probe_interaction_inputs import sensor_frame
from tests.controller.test_interaction_senses import paired_controllers


def observation(core, case, tick):
    return Observation(core.identity, tick, tick, tick * 50_000_000, sensor_frame(case, tick))


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("light_off", ProgramId.HOVER),
        ("player_approach", ProgramId.ESCAPE_JUMP),
        ("mob_approach", ProgramId.BANK),
    ],
)
def test_real_steps_preserve_envelopes_raw_projection_and_source_readouts(case, expected):
    core, _, _ = paired_controllers()
    direct, _, _ = paired_controllers()
    wrapper = InteractionProgramSelectingController(core)
    selected = []
    for tick in range(30):
        obs = observation(core, case, tick)
        raw = direct.step(obs, command_seq=tick)
        command = wrapper.step(obs, command_seq=tick)
        for field in (
            "identity",
            "command_seq",
            "observation_seq",
            "observation_tick",
            "target_tick",
            "valid_through_tick",
            "neural_step_start",
            "neural_step_end",
            "state_digest",
        ):
            assert getattr(command, field) == getattr(raw, field)
        assert wrapper.last_evidence.raw_projection == raw.motors
        assert command.target_tick == tick + 2
        ev = wrapper.last_event_evidence
        doc = ev.to_dict()
        assert doc["interaction"]["raw_ttmn"] == direct.jump_readout_mean()
        assert doc["interaction"]["dna01_left"] == direct.steering_readout_means()[0]
        assert doc["requested_pulse"] == direct.last_event_readout.pulse.to_dict()
        assert doc["readout"] == command.readout.to_dict()
        assert len(ev.json_bytes()) < 16 * 1024
        if ev.decision.source is not None:
            selected.append(command.program)
            assert ev.trigger.source == ev.decision.source
            assert ev.trigger.command.command_seq <= command.command_seq
    assert selected and set(selected) == {expected}


@pytest.mark.parametrize(
    "case", ["attack", "brush", "food", "light_off", "player_approach", "mob_approach"]
)
def test_zero_input_cannot_select_any_event_program_despite_prescribed_source(case):
    core, _, _ = paired_controllers("no_input")
    wrapper = InteractionProgramSelectingController(core)
    for tick in range(30):
        command = wrapper.step(observation(core, case, tick), command_seq=tick)
        ev = wrapper.last_event_evidence
        assert command.program is ProgramId.CRUISE
        assert ev.decision.source is None and ev.trigger is None
        assert all(score.score == 0 for score in ev.decision.scores.values())
        assert ev.completed.mn9_raw == ev.completed.antenna_raw == ev.ttmn_raw == 0
    assert core.engine.step_index == 150


@pytest.mark.parametrize("reset_method", ["reset", "reset_session"])
def test_reset_clears_held_source_trigger_and_history_and_reproduces_actual_sequence(reset_method):
    core, _, _ = paired_controllers()
    wrapper = InteractionProgramSelectingController(core)
    before = []
    for tick in range(30):
        command = wrapper.step(observation(core, "light_off", tick), command_seq=tick)
        before.append((command.program, command.state_digest, wrapper.policy_state.to_dict()))
    assert wrapper.last_event_evidence.trigger is not None
    if reset_method == "reset":
        wrapper.reset()
    else:
        wrapper.reset_session(session_epoch=wrapper.identity.session_epoch + 1)
    assert wrapper.last_event_evidence is None and wrapper.last_evidence is None
    for tick in range(30):
        # In-world reset preserves transport/game clocks; a new session restarts
        # sequences. Replay the same sensory history with the appropriate clock.
        offset = 30 if reset_method == "reset" else 0
        obs = replace(
            observation(core, "light_off", tick),
            observation_seq=tick + offset,
            game_tick=tick + offset,
            sent_monotonic_ns=(tick + offset) * 50_000_000,
        )
        command = wrapper.step(obs, command_seq=tick + offset)
        assert (command.program, command.state_digest) == before[tick][:2]
        state = wrapper.policy_state
        assert state.history == before[tick][2]["history"]
        assert state.source == before[tick][2]["source"]


def test_wrapper_refuses_old_input_and_different_actual_population_bindings():
    core, old, _ = paired_controllers()
    with pytest.raises(EventSensesError, match="separate V4"):
        InteractionProgramSelectingController(old)
    wrapper = InteractionProgramSelectingController(core)
    with pytest.raises(EventSensesError, match="differs"):
        InteractionProgramSelectingController(
            core, replace(wrapper.policy, event_binding_sha256="a" * 64)
        )
