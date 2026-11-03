"""New profile construction and optional evidence through production hooks."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from neurocraft_controller.interaction_program_controller import (
    InteractionProgramSelectingController,
)

from scripts import run_connectome_service as cli
from scripts import run_interactive_demo as launcher
from tests.controller.test_interaction_program_controller import observation
from tests.controller.test_interaction_senses import paired_controllers

PROFILE = "full_mcns_interactions_program_v2"


def test_profile_builds_v4_core_and_wrapper_with_explicit_component_calibration(monkeypatch):
    from neurocraft_controller.full_brain_controller import FullBrainController

    core, _, _ = paired_controllers()
    captured = []

    def build(**kwargs):
        captured.append(kwargs)
        return core

    monkeypatch.setattr(FullBrainController, "from_artifact", build)
    args = cli.build_parser().parse_args(["--controller-profile", PROFILE, "--kernel-threads", "8"])
    monkeypatch.setattr(cli, "_refuse_full_profile_conflicts", lambda args: None)
    result = cli.build_controller(args)
    assert isinstance(result, InteractionProgramSelectingController)
    call = captured[0]
    assert call["event_inputs"] and call["event_inputs_version"] == 2
    assert call["loom_version"] == call["escape_version"] == 2
    assert call["steering_readout"] and call["embodied_senses"]
    assert call["configuration"].kernel_threads == 8
    path, hint = cli._full_profile_calibration(PROFILE)
    assert path == cli.DEFAULT_FULL_LOOM_V2_CALIBRATION and "component only" in hint
    for module in (cli, launcher):
        assert PROFILE in module.CONTROLLER_PROFILES
        assert PROFILE in module.FULL_MCNS_PROGRAM_PROFILES
        assert PROFILE in module.FULL_MCNS_COMPARISON_PROFILES
        assert PROFILE not in module.FULL_MCNS_LOOM_V2_PROFILES


@pytest.mark.parametrize("mode", ["faithful", "shuffled_weight", "no_input"])
def test_optional_v2_display_writes_matching_command_without_changing_decision(tmp_path, mode):
    core, _, _ = paired_controllers(mode)
    twin, _, _ = paired_controllers(mode)
    wrapper = InteractionProgramSelectingController(core)
    direct = InteractionProgramSelectingController(twin)
    path = tmp_path / "event.json"
    observe, close = cli._with_event_display(wrapper, path, None, None)
    assert observe is not None
    for tick in range(30):
        obs = observation(core, "light_off", tick)
        command = wrapper.step(obs, command_seq=tick)
        assert command == direct.step(obs, command_seq=tick)
        if tick == 0:
            observe(replace(command, command_seq=1))
            assert not path.exists()
        observe(command)
    close()
    document = json.loads(path.read_bytes())
    assert document == wrapper.last_event_evidence.to_dict()
    assert document["schema"] == "neurocraft.interaction_selection_display.v2"
    assert document["requested_pulse"]["attack"] is False


def test_v2_display_failure_does_not_change_commands_or_primary_observer(tmp_path, monkeypatch):
    from neurocraft_controller.brain_display import AtomicBrainDisplayPublisher

    core, _, _ = paired_controllers()
    twin, _, _ = paired_controllers()
    wrapper = InteractionProgramSelectingController(core)
    direct = InteractionProgramSelectingController(twin)
    seen = []

    def fail(self, evidence):
        raise OSError("deliberate optional display failure")

    monkeypatch.setattr(AtomicBrainDisplayPublisher, "offer", fail)
    observe, close = cli._with_event_display(wrapper, tmp_path / "event.json", seen.append, None)
    for tick in range(4):
        obs = observation(core, "attack", tick)
        command = wrapper.step(obs, command_seq=tick)
        observe(command)
        assert command == direct.step(obs, command_seq=tick)
        assert seen[-1] == command
    close()
    assert not (tmp_path / "event.json").exists()
