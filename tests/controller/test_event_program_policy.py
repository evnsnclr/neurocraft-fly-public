"""Candidate policy boundaries plus actual-engine selections and immutable evidence."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest
from neurocraft_controller.config import canonical_sha256
from neurocraft_controller.event_program_policy import (
    EventPolicyState,
    EventProgramSelectingController,
)
from neurocraft_controller.events_senses import CompletedEventReadout, EventPulse
from neurocraft_protocol import ProgramId, ReadoutValues

from tests.controller.test_embodied_senses import _observation
from tests.controller.test_events_senses import controllers, frame, sample

QUIET = ReadoutValues(0.0, 0.0, 0.0)


def completed(mn9=0.0, antenna=0.0, *, food=False, groom=False, age=0):
    return CompletedEventReadout(
        EventPulse(
            food,
            groom,
            age if food else None,
            age if groom else None,
            int(food),
            int(groom),
            0,
            False,
        ),
        mn9,
        antenna,
        mn9,
        mn9,
        antenna,
        antenna,
    )


def policy_and_ready():
    core, _ = controllers()
    policy = EventProgramSelectingController(core).policy
    state = EventPolicyState()
    for tick in range(4):
        result = policy.select_events(QUIET, completed(), frame(), state, tick)
        state = result.program.state_after
    return policy, state


def select(
    policy,
    state,
    tick,
    *,
    mn9=0,
    antenna=0,
    food=False,
    groom=False,
    near=False,
    contact=False,
    fluid=False,
    escape=False,
    age=0,
):
    return policy.select_events(
        ReadoutValues(0.1 if escape else 0.0, 0.0, 0.0),
        completed(mn9, antenna, food=food, groom=groom, age=age),
        frame(sample(near=near, contact=contact), in_fluid=fluid),
        state,
        tick,
    )


def test_provisional_thresholds_and_named_bindings_are_in_the_declaration():
    policy, _ = policy_and_ready()
    material = policy.to_identity_material()
    assert material["thresholds"] == {"mn9": (1e-4).hex(), "antenna": (3e-4).hex()}
    assert "not a paired counterfactual" in material["score"]
    assert "not behavior-validated" in material["status"]
    assert material["landing_deadline_game_ticks"] == 200
    for key, value in (
        ("mn9_threshold", 2e-4),
        ("antenna_threshold", 4e-4),
        ("event_binding_sha256", "2" * 64),
        ("response_window_ticks", 7),
        ("event_cooldown_ticks", 121),
    ):
        assert replace(policy, **{key: value}).identity_sha256 != policy.identity_sha256


def test_four_completed_observations_baseline_frozen_before_event_not_counterfactual():
    policy, _ = policy_and_ready()
    state = EventPolicyState()
    # The fourth observation contains an event; there are only three PRE-event
    # samples so it must not become eligible on the following quiet observation.
    for tick in range(4):
        decision = select(policy, state, tick, mn9=0.001, food=tick == 3, near=True, contact=True)
        state = decision.program.state_after
        assert decision.program.program is ProgramId.CRUISE
    assert state.food is None
    # Pre-event background is high, and the event has no increment despite raw
    # activity already exceeding threshold: still no selection.
    flat = select(policy, state, 4, mn9=0.001, food=True, contact=True)
    assert flat.mn9.baseline == pytest.approx(0.001)
    assert flat.mn9.score == 0 and flat.program.program is ProgramId.CRUISE
    delayed = select(policy, flat.program.state_after, 5, mn9=0.0012, contact=True)
    assert delayed.mn9.baseline == flat.mn9.baseline
    assert delayed.mn9.score == pytest.approx(0.0002)
    assert delayed.program.program is ProgramId.FEED


def test_food_requires_event_and_near_even_with_contact_confounded_large_mn9():
    policy, state = policy_and_ready()
    assert select(policy, state, 4, mn9=0.01, contact=True).program.program is ProgramId.CRUISE
    assert select(policy, state, 4, mn9=0.01, food=True).program.program is ProgramId.CRUISE
    assert (
        select(policy, state, 4, mn9=0.01, food=True, near=True).program.program is ProgramId.LAND
    )
    assert (
        select(policy, state, 4, mn9=0.01, food=True, contact=True).program.program
        is ProgramId.FEED
    )
    assert (
        select(policy, state, 4, mn9=0.01, food=True, near=True, fluid=True).program.program
        is ProgramId.CRUISE
    )


def test_simultaneous_inputs_choose_food_sequence_and_escape_discards_both():
    policy, state = policy_and_ready()
    both = dict(mn9=0.01, antenna=0.01, food=True, groom=True, near=True)
    decision = select(policy, state, 4, **both)
    assert decision.program.program is ProgramId.LAND
    assert decision.program.state_after.food is decision.program.state_after.groom is None
    escaped = select(policy, state, 4, **both, escape=True)
    assert escaped.program.program is ProgramId.ESCAPE_JUMP
    assert escaped.program.state_after.food is escaped.program.state_after.groom is None
    # Returning to quiet does not recover a queued food/groom request.
    state = escaped.program.state_after
    for tick in range(5, 80):
        result = select(policy, state, tick, mn9=0.01, antenna=0.01, contact=True)
        state = result.program.state_after
        assert result.program.program not in (ProgramId.LAND, ProgramId.FEED, ProgramId.GROOM)


def test_land_latches_score_until_contact_then_sixty_ticks_and_120_tick_cooldown():
    policy, state = policy_and_ready()
    decision = select(policy, state, 4, mn9=0.001, food=True, near=True)
    state = decision.program.state_after
    trigger = state.held_event
    # Actual LAND descent is multiplied by speed; zero would strand the fly.
    assert decision.program.modulation.speed == 1.0
    # Landing advances without a new event or a fresh positive neural score.
    for tick in range(5, 11):
        result = select(policy, state, tick, near=True, contact=tick == 10)
        state = result.program.state_after
        assert state.held_event == trigger
        assert result.program.program is (ProgramId.FEED if tick == 10 else ProgramId.LAND)
    assert result.reason == "landed_feed"
    assert state.active_until == 70 and state.cooldown_until == 190
    for tick in range(11, 190):
        result = select(
            policy,
            state,
            tick,
            mn9=0.9,
            antenna=0.9,
            food=tick == 100,
            groom=tick == 100,
            contact=True,
        )
        state = result.program.state_after
        assert result.program.program is (ProgramId.FEED if tick < 70 else ProgramId.CRUISE)
        assert result.program.modulation.urgency == 0
    # Cooldown expiring doesn't replay the ignored event at tick100.
    result = select(policy, state, 190, mn9=0.9, contact=True)
    assert result.program.program is ProgramId.CRUISE
    assert result.program.state_after.food is None


@pytest.mark.parametrize("cause", ["near_lost", "fluid", "escape", "timeout"])
def test_land_cancellation_does_not_renew_or_advance_feed(cause):
    policy, state = policy_and_ready()
    state = select(policy, state, 4, mn9=0.001, food=True, near=True).program.state_after
    tick = 204 if cause == "timeout" else 5
    decision = select(
        policy,
        state,
        tick,
        near=cause != "near_lost",
        fluid=cause == "fluid",
        escape=cause == "escape",
        food=True,
        mn9=0.001,
    )
    assert decision.program.program is (
        ProgramId.ESCAPE_JUMP if cause == "escape" else ProgramId.CRUISE
    )
    assert decision.program.state_after.held_event is None
    assert decision.program.state_after.food is None
    assert decision.program.state_after.cooldown_until == tick + 120


def test_pending_event_expires_after_eight_world_ticks_including_skipped_observations():
    policy, state = policy_and_ready()
    # A pulse arriving at age8 can still be used at that observation.
    assert (
        select(policy, state, 12, food=True, age=8, mn9=0.001, contact=True).program.program
        is ProgramId.FEED
    )
    quiet = select(policy, state, 4, food=True, near=True)
    late = select(policy, quiet.program.state_after, 13, mn9=0.01, contact=True)
    assert late.program.program is ProgramId.CRUISE and late.program.state_after.food is None


def test_groom_requires_a_fresh_neural_response_and_exits_on_fluid():
    policy, state = policy_and_ready()
    assert select(policy, state, 4, antenna=0.01).program.program is ProgramId.CRUISE
    assert (
        select(policy, state, 4, antenna=0.01, groom=True, fluid=True).program.program
        is ProgramId.CRUISE
    )
    chosen = select(policy, state, 4, antenna=0.01, groom=True)
    assert chosen.program.program is ProgramId.GROOM
    stopped = select(policy, chosen.program.state_after, 5, fluid=True)
    assert stopped.program.program is ProgramId.CRUISE and stopped.reason == "gate_lost"
    assert stopped.program.state_after.cooldown_until == 125


@pytest.mark.parametrize("mode", ["faithful", "no_input", "shuffled_weight"])
def test_actual_engine_issues_named_evidence_without_relabeling_motor_readouts(mode):
    core, _ = controllers(mode)
    controller = EventProgramSelectingController(core)
    issued = []
    for tick in range(12):
        obs = _observation(
            core,
            tick,
            frame(sample(food=int(tick >= 4), tick=4, near=tick >= 4, contact=tick >= 8)),
        )
        action = controller.step(obs, command_seq=tick)
        issued.append(action)
        assert action.readout == controller.last_evidence.readout
        assert controller.last_evidence.raw_projection == core.last_evidence.motors
        assert action.identity == core.identity
        evidence = controller.last_event_evidence
        assert evidence.command is action
        encoded = json.loads(evidence.json_bytes())
        assert encoded["identity"] == action.identity.to_dict()
        assert encoded["state_digest"] == action.state_digest
        assert encoded["policy_hash"] == action.policy_hash == controller.policy.identity_sha256
        assert encoded["command_sha256"] == canonical_sha256(action.to_dict())
        assert encoded["command_seq"] == action.command_seq
        assert encoded["observation_seq"] == action.observation_seq
        assert encoded["target_tick"] == action.target_tick
        assert encoded["valid_through_tick"] == action.valid_through_tick
        assert encoded["modulation"] == action.modulation.to_dict()
        assert "before final control input gate" in encoded["pulse_scope"]
        assert encoded["presentation_only"] is True
        assert "requires authoritative" in encoded["selection_status"]
        with pytest.raises(FrozenInstanceError):
            evidence.reason = "invented"
        if mode == "no_input":
            assert encoded["mn9"]["raw"] == encoded["antenna"]["raw"] == 0
            assert action.program is ProgramId.CRUISE
    if mode == "faithful":
        assert any(action.program is ProgramId.LAND for action in issued)
        assert issued[-1].program is ProgramId.FEED
        origin = controller.last_event_evidence.trigger
        assert origin.command.program is ProgramId.LAND
        assert origin.command.command_seq < issued[-1].command_seq
        assert origin.score.score >= controller.policy.mn9_threshold
        assert origin.command.state_digest != issued[-1].state_digest
    controller.reset_session(session_epoch=2**60 + 1)
    assert controller.last_event_evidence is None
    assert controller.policy_state.history == ()
    assert controller.policy_state.held_event is None
    first = controller.step(
        _observation(core, 0, frame(sample(food=999, tick=0, contact=True))), command_seq=0
    )
    assert first.program is ProgramId.CRUISE
    assert controller.last_event_evidence.completed.pulse.baseline_absorbed


def test_probe_runs_real_candidate_on_all_declared_synthetic_streams():
    from tests.controller.event_candidate_probe import CASES, run_case, without_epoch

    core, _ = controllers()
    controller = EventProgramSelectingController(core)
    cases = [run_case(controller, case, index + 1) for index, case in enumerate(CASES)]
    assert [case["case"] for case in cases] == list(CASES)
    for case in cases:
        assert case["neural_updates_executed"] == 5 * len(case["rows"])
        assert sum(case["issued_program_counts"].values()) == len(case["rows"])
    original = cases[CASES.index("near_land_contact_feed")]
    restored = run_case(controller, "near_land_contact_feed", len(CASES) + 1)
    assert without_epoch(original["rows"]) == without_epoch(restored["rows"])
    assert (
        "LAND" in original["issued_program_counts"] and "FEED" in original["issued_program_counts"]
    )
