"""The paused-window verifier is the checkable half of the ADR 0020 honesty claim.

A scripted body program may own a tick only while the neural service is paused. These
tests exercise the recipe the service actually stamps onto a paused action, so a claim
made over footage ("the connectome was not driving") can be checked against a committed
trace rather than believed.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest
from neurocraft_protocol import (
    ACTION_TARGET_OFFSET_TICKS,
    Action,
    MotorVector,
    RunIdentity,
)

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "verify_paused_window.py"


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_paused_window", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # The module defines dataclasses, so it must be importable by name while it executes.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = load_module()


def identity() -> RunIdentity:
    return RunIdentity(
        run_id="paused-window-test",
        session_epoch=1,
        entity_id="fly-01",
        graph_hash="1" * 64,
        adapter_hash="2" * 64,
        dynamics_hash="3" * 64,
        decoder_hash="4" * 64,
        observation_schema_hash="5" * 64,
    )


def neutral_action(*, command_seq: int, game_tick: int, cursor: int, reason: str) -> Action:
    """Build exactly the action the service emits for a non-running control state."""

    return Action(
        identity=identity(),
        command_seq=command_seq,
        observation_seq=command_seq,
        observation_tick=game_tick,
        target_tick=game_tick + ACTION_TARGET_OFFSET_TICKS,
        valid_through_tick=game_tick + ACTION_TARGET_OFFSET_TICKS,
        neural_step_start=cursor,
        neural_step_end=cursor,
        state_digest=VERIFIER.neutral_digest(
            reason,
            observation_seq=command_seq,
            game_tick=game_tick,
            neural_cursor=cursor,
        ),
        motors=MotorVector.neutral(),
        neutral=True,
    )


def driving_action(*, command_seq: int, game_tick: int, cursor: int) -> Action:
    return Action(
        identity=identity(),
        command_seq=command_seq,
        observation_seq=command_seq,
        observation_tick=game_tick,
        target_tick=game_tick + ACTION_TARGET_OFFSET_TICKS,
        valid_through_tick=game_tick + ACTION_TARGET_OFFSET_TICKS,
        neural_step_start=cursor,
        neural_step_end=cursor + 4,
        state_digest="a" * 64,
        motors=MotorVector(forward_thrust=0.25),
        neutral=False,
    )


def test_a_paused_window_is_proven_from_the_actions_alone() -> None:
    actions = [
        neutral_action(command_seq=index, game_tick=100 + index, cursor=77, reason="paused")
        for index in range(8)
    ]

    outcome = VERIFIER.verify_actions(actions)

    assert outcome.fully_paused
    assert outcome.paused_ticks == 8
    assert outcome.driving_ticks == 0
    assert outcome.unexplained_ticks == 0
    # The window is keyed on the ticks the actions target: observations 100..107 plus the
    # declared two-tick action budget.
    assert outcome.longest_paused_window == (102, 109)
    assert all(not tick.neural_cursor_advanced for tick in outcome.ticks)


def test_a_stepping_service_is_never_reported_as_paused() -> None:
    actions = [
        driving_action(command_seq=index, game_tick=200 + index, cursor=index * 4)
        for index in range(4)
    ]

    outcome = VERIFIER.verify_actions(actions)

    assert not outcome.fully_paused
    assert outcome.paused_ticks == 0
    assert outcome.driving_ticks == 4
    assert outcome.longest_paused_window is None


def test_a_neutral_action_stamped_with_another_reason_is_not_a_paused_window() -> None:
    """REPLAYING and FAULTED also emit neutral actions; only PAUSED backs the claim."""

    actions = [
        neutral_action(command_seq=0, game_tick=10, cursor=5, reason="paused"),
        neutral_action(command_seq=1, game_tick=11, cursor=5, reason="replaying"),
        neutral_action(command_seq=2, game_tick=12, cursor=5, reason="faulted"),
    ]

    outcome = VERIFIER.verify_actions(actions)

    assert not outcome.fully_paused
    assert outcome.paused_ticks == 1
    assert [tick.classification for tick in outcome.ticks] == ["paused", "replaying", "faulted"]


def test_a_forged_neutral_digest_is_refused_rather_than_believed() -> None:
    """The digest is the whole check: a neutral flag alone proves nothing."""

    forged = replace(
        neutral_action(command_seq=0, game_tick=10, cursor=5, reason="paused"),
        state_digest="f" * 64,
    )

    outcome = VERIFIER.verify_actions([forged])

    assert not outcome.fully_paused
    assert outcome.unexplained_ticks == 1
    assert outcome.ticks[0].classification == VERIFIER.UNEXPLAINED


def test_an_advancing_neural_cursor_breaks_the_window_even_when_the_digest_matches() -> None:
    """The cursor is the second, independent signature; both must hold."""

    paused = neutral_action(command_seq=0, game_tick=10, cursor=5, reason="paused")
    advanced = replace(paused, command_seq=1, observation_seq=1, neural_step_end=9)
    stamped = replace(
        advanced,
        state_digest=VERIFIER.neutral_digest(
            "paused", observation_seq=1, game_tick=10, neural_cursor=5
        ),
    )

    outcome = VERIFIER.verify_actions([paused, stamped])

    assert outcome.paused_ticks == 2
    assert outcome.longest_paused_window == (12, 12)


def test_an_empty_trace_is_not_a_paused_window() -> None:
    outcome = VERIFIER.verify_actions([])

    assert not outcome.fully_paused
    assert outcome.longest_paused_window is None


def test_the_command_line_refuses_a_missing_trace() -> None:
    with pytest.raises(OSError):
        VERIFIER.main(["--trace", str(ROOT / "artifacts" / "no-such-trace.ndjson")])
