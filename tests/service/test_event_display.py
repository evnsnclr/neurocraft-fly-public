"""Real small-network event snapshots through the production service hooks."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from neurocraft_controller.event_program_policy import EventProgramSelectingController
from neurocraft_service.lane_sessions import LaneSpec

from scripts import run_connectome_service as cli
from tests.controller.test_embodied_senses import _observation
from tests.controller.test_events_senses import controllers, frame, sample


@pytest.mark.parametrize("mode", ["faithful", "shuffled_weight", "no_input"])
def test_epoch_hook_writes_only_matching_issued_evidence(tmp_path: Path, mode: str) -> None:
    core, _ = controllers(mode)
    controller = EventProgramSelectingController(core)
    args = cli.build_parser().parse_args(
        [
            "--controller-profile",
            cli.FULL_MCNS_EVENTS_PROGRAM_V1_PROFILE,
            "--epoch-diagnostics-dir",
            str(tmp_path),
            "--brain-display-out",
            str(tmp_path / "unused-base.json"),
            "--program-decisions-out",
            str(tmp_path / "unused-decisions.ndjson"),
        ]
    )
    name = "original" if mode == "faithful" else mode
    hooks = cli._comparison_hooks_factory(args, None)(
        LaneSpec(name, controller),
        controller.identity,  # type: ignore[arg-type]
    )
    directory = tmp_path / str(controller.identity.session_epoch)
    path = directory / "event-selection-display.json"
    assert hooks.completed_step_observer is not None and hooks.close is not None
    command = controller.step(_observation(core, 0, frame()), command_seq=0)
    hooks.completed_step_observer(replace(command, command_seq=1))
    assert not path.exists()
    for tick in range(1, 8):
        command = controller.step(
            _observation(core, tick, frame(sample(food=int(tick >= 5), tick=5, near=tick >= 5))),
            command_seq=tick,
        )
        assert hooks.issued_action_sink is not None
        hooks.issued_action_sink(command)
        hooks.completed_step_observer(command)
    expected = controller.last_event_evidence.to_dict()
    hooks.close()
    document = json.loads(path.read_bytes())
    assert document == expected
    assert document["identity"] == controller.identity.to_dict()
    assert document["selection_status"] == "issued; requires authoritative applied-command match"
    assert document["command_seq"] == 7
    assert not (tmp_path / "event-selection-display.json").exists()
    if mode == "no_input":
        assert document["mn9"]["raw"] == document["antenna"]["raw"] == 0


def test_candidate_build_selects_event_wrapper_and_explicit_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core, _ = controllers()
    calls = []

    def build(args):
        calls.append(args.controller_profile)
        return core

    monkeypatch.setattr(cli, "_build_full_brain_controller", build)
    # The construction branch is tested with a real small controller. Bulk-data
    # presence/calibration gates have separate tests and are unavailable in CI.
    monkeypatch.setattr(cli, "_refuse_full_profile_conflicts", lambda args: None)
    args = cli.build_parser().parse_args(
        [
            "--controller-profile",
            cli.FULL_MCNS_EVENTS_PROGRAM_V1_PROFILE,
        ]
    )
    result = cli.build_controller(args)
    assert isinstance(result, EventProgramSelectingController)
    assert calls == [cli.FULL_MCNS_EVENTS_PROGRAM_V1_PROFILE]
    path, hint = cli._full_profile_calibration(args.controller_profile)
    assert path == cli.DEFAULT_FULL_LOOM_V2_CALIBRATION
    assert "component only" in hint
    assert args.controller_profile not in cli.FULL_MCNS_LOOM_V2_PROFILES


def test_kernel_thread_request_reaches_full_configuration_only(monkeypatch) -> None:
    from neurocraft_controller.full_brain_controller import FullBrainController

    core, _ = controllers()
    captured = []

    def construct(**kwargs):
        captured.append(kwargs)
        return core

    monkeypatch.setattr(FullBrainController, "from_artifact", construct)
    args = cli.build_parser().parse_args(
        [
            "--controller-profile",
            cli.FULL_MCNS_EVENTS_PROGRAM_V1_PROFILE,
            "--kernel-threads",
            "4",
        ]
    )
    assert cli._build_full_brain_controller(args) is core
    assert captured[0]["configuration"].kernel_threads == 4
    assert captured[0]["event_inputs"] is True
    with pytest.raises(SystemExit, match="only to full"):
        cli.build_controller(cli.build_parser().parse_args(["--kernel-threads", "4"]))


def test_legacy_display_failure_does_not_suppress_event_snapshot(tmp_path: Path) -> None:
    core, _ = controllers()
    controller = EventProgramSelectingController(core)

    def broken(command):
        raise OSError("optional legacy display failed")

    observe, close = cli._with_event_display(controller, tmp_path / "event.json", broken, None)
    command = controller.step(_observation(core, 0, frame()), command_seq=0)
    assert observe is not None
    observe(command)
    close()
    assert json.loads((tmp_path / "event.json").read_bytes())["command_seq"] == 0


def test_event_display_failure_keeps_legacy_callback_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neurocraft_controller.brain_display import AtomicBrainDisplayPublisher

    core, _ = controllers()
    controller = EventProgramSelectingController(core)
    captured = []

    def broken_offer(self, evidence):
        raise OSError("optional event offer failed")

    monkeypatch.setattr(AtomicBrainDisplayPublisher, "offer", broken_offer)
    observe, close = cli._with_event_display(
        controller,
        tmp_path / "event.json",
        captured.append,
        None,
    )
    assert observe is not None
    for tick in range(2):
        command = controller.step(_observation(core, tick, frame()), command_seq=tick)
        observe(command)  # No exception reaches the service's disable-on-error boundary.
        assert captured[-1] == command
    close()
    assert len(captured) == 2
    assert not (tmp_path / "event.json").exists()
