"""V2 source gates and exact game-tick boundaries; applied body behavior is separate."""

from dataclasses import FrozenInstanceError, replace

import pytest
from neurocraft_controller.events_senses import CompletedEventReadout, EventSensesError
from neurocraft_controller.interaction_program_policy import (
    SOURCES,
    InteractionPolicyState,
    InteractionSelectionPolicy,
)
from neurocraft_controller.interaction_senses import InteractionInputState
from neurocraft_controller.program_policy import PolicyReadoutBinding
from neurocraft_protocol import EventCounter, ProgramId, ReadoutValues, SensorFrameV4

from tests.controller.test_events_senses import frame, sample

QUIET = ReadoutValues(0.0, 0.0, 0.0)


def policy():
    return InteractionSelectionPolicy(
        PolicyReadoutBinding(*("1" * 64,) * 4),
        event_encoder_sha256="2" * 64,
        event_binding_sha256="3" * 64,
    )


def sensors(events=None, *, actor=0, distance=6.0, light=15, fluid=False, ray=0):
    kinds, actors, distances = [0] * 6, [0] * 6, [6.0] * 6
    kinds[ray], actors[ray], distances[ray] = (2 if actor else 0), actor, distance
    return SensorFrameV4.from_event_frame(
        replace(
            frame(events, local_light=light, in_fluid=fluid, ray_distances_m=tuple(distances)),
            ray_hit_kinds=tuple(kinds),
        ),
        tuple(actors),
    )


def step(
    p,
    state,
    tick,
    *,
    sensory=None,
    ttmn=0.0,
    dna=(0.0, 0.0),
    mn9=0.0,
    antenna=0.0,
    jump=0.0,
    readout=None,
):
    sensory = sensory or sensors(state.previous_sensors.events if state.previous_sensors else None)
    # Execute the actual input-state pulse coalescing; its binding is not consulted
    # by prepare(), and no synthetic neural engine is claimed by these policy tests.
    input_state = InteractionInputState.__new__(InteractionInputState)
    input_state._previous = (
        None if state.previous_sensors is None else state.previous_sensors.events
    )
    pulse = input_state.prepare(sensory.events, tick)
    completed = CompletedEventReadout(pulse, mn9, antenna, mn9, mn9, antenna, antenna)
    return p.select_interactions(
        readout or ReadoutValues(jump, 0.0, 0.0),
        completed,
        sensory,
        state,
        tick,
        ttmn_raw=ttmn,
        dna01_means=dna,
    )


def ready(*, actor=0, distance=6.0, light=15, ttmn=0, mn9=0, antenna=0, dna=(0, 0), ray=0):
    p, state = policy(), InteractionPolicyState()
    for tick in range(4):
        state = step(
            p,
            state,
            tick,
            sensory=sensors(actor=actor, distance=distance, light=light, ray=ray),
            ttmn=ttmn,
            mn9=mn9,
            antenna=antenna,
            dna=dna,
        ).program.state_after
    return p, state


def stimulus(source, tick=4, *, strength=True, prior_events=None):
    events = prior_events or sample()
    kw = {}
    if source == "attack":
        events = replace(events, attack_attempt=EventCounter(events.attack_attempt.count + 1, tick))
        kw["ttmn"] = -0.004 if strength else -0.001
    elif source in ("player_approach", "mob_approach"):
        kw["jump"] = 0.1 if strength else 0.0
        return dict(
            sensory=sensors(events, actor=1 if source == "player_approach" else 2, distance=5.0),
            **kw,
        )
    elif source == "food":
        events = replace(
            events,
            food=EventCounter(events.food.count + 1, tick),
            food_near=True,
            food_contact=True,
        )
        kw["mn9"] = 0.001 if strength else 0.00001
    elif source == "brush":
        events = replace(events, brush=EventCounter(events.brush.count + 1, tick))
        kw["antenna"] = 0.001 if strength else 0.00001
    elif source == "light":
        return dict(sensory=sensors(events, light=0), dna=(1e-5, 1e-5) if strength else (0.0, 0.0))
    return dict(sensory=sensors(events), **kw)


@pytest.mark.parametrize(
    "source,expected",
    tuple(
        zip(
            SOURCES,
            (
                ProgramId.FLEE,
                ProgramId.ESCAPE_JUMP,
                ProgramId.BANK,
                ProgramId.FEED,
                ProgramId.GROOM,
                ProgramId.HOVER,
            ),
            strict=True,
        )
    ),
)
def test_six_distinct_sources_require_actual_neural_gate_and_retain_raw(source, expected):
    actor = 1 if source == "player_approach" else 2 if source == "mob_approach" else 0
    p, state = ready(actor=actor)
    low = step(p, state, 4, **stimulus(source, strength=False))
    high = step(p, state, 4, **stimulus(source))
    assert low.program.program is ProgramId.CRUISE
    assert high.program.program is expected
    assert high.source == source and high.reason == "fresh_eligible_response"
    assert set(high.scores) == set(SOURCES)
    assert high.scores[source].baseline == 0.0
    assert high.scores[source].event_age == 0
    assert high.scores[source].eligible
    assert high.program.readout is not None
    if source == "attack":
        assert high.scores[source].raw == -0.004 and high.scores[source].score == 0.004
        assert high.program.readout.ttmn_jump == 0.0
    if source == "mob_approach":
        assert high.program.modulation.turn_rate > 0 and high.program.parameters.turn_bias == 1
    if source in ("attack", "mob_approach", "light"):
        assert high.program.modulation.urgency == 0
        assert high.program.state_after.active_until == 44


@pytest.mark.parametrize("actor,ray", [(0, 0), (3, 0), (1, 1), (2, 1), (1, 4), (2, 5)])
def test_wrong_actor_or_nonfrontal_ray_cannot_trigger_an_approach(actor, ray):
    p, state = ready(actor=actor, ray=ray)
    result = step(p, state, 4, sensory=sensors(actor=actor, distance=1, ray=ray), jump=1)
    assert result.program.program is ProgramId.CRUISE


def test_presence_without_decreasing_distance_or_category_switch_is_not_approach():
    p, state = ready(actor=1, distance=4)
    for sensory in (
        sensors(actor=1, distance=4),
        sensors(actor=1, distance=5),
        sensors(actor=2, distance=2),
    ):
        assert step(p, state, 4, sensory=sensory, jump=1).program.program is ProgramId.CRUISE
    p, state = ready()
    assert step(p, state, 4, sensory=sensors(actor=1, distance=1), jump=1).source is None


def test_approach_holds_original_actor_after_metadata_changes_and_expires_exactly():
    p, state = ready(actor=1)
    entered = step(p, state, 4, **stimulus("player_approach"))
    state = entered.program.state_after
    original = state.held_event
    switched = step(p, state, 5, sensory=sensors(actor=2, distance=1))
    assert switched.source == "player_approach"
    assert switched.program.program is ProgramId.ESCAPE_JUMP
    assert switched.program.state_after.held_event == original
    assert switched.program.modulation == entered.program.modulation
    last = step(p, switched.program.state_after, 27)
    assert last.program.state_after.hold_remaining == 1
    expired = step(p, last.program.state_after, 28)
    assert expired.source is None and expired.reason == "hold_expired"
    assert expired.program.state_after.cooldown_until == 148


def test_source_priority_and_fresh_gated_preemption_discard_blocked_events():
    p, state = ready(actor=1)
    all_events = sample(food=1, brush=1, attack=1, tick=4, contact=True)
    result = step(
        p,
        state,
        4,
        sensory=sensors(all_events, actor=1, distance=3, light=0),
        jump=1,
        ttmn=-0.004,
        mn9=0.01,
        antenna=0.01,
        dna=(1e-5, 1e-5),
    )
    assert result.source == "attack"
    assert all(event is None for event in result.program.state_after.pending)
    p, state = ready()
    light = step(p, state, 4, **stimulus("light"))
    # A weak fresh attack is discarded during HOVER rather than queued for later gain.
    weak = step(p, light.program.state_after, 5, **stimulus("attack", 5, strength=False))
    assert weak.source == "light"
    late_gain = step(p, weak.program.state_after, 6, ttmn=-0.01)
    assert late_gain.source == "light"
    strong = step(
        p,
        late_gain.program.state_after,
        7,
        **{
            **stimulus(
                "attack", 7, prior_events=late_gain.program.state_after.previous_sensors.events
            ),
            "ttmn": -0.02,
        },
    )
    assert strong.source == "attack" and strong.reason == "fresh_eligible_response"
    assert strong.program.state_after.held_event.event_tick == 7


def test_delayed_score_uses_frozen_four_sample_baseline_and_inclusive_eight_tick_window():
    p, state = ready(antenna=0.01)
    first = step(p, state, 4, sensory=sensors(sample(brush=10, tick=4)), antenna=0.01)
    assert first.source is None
    frozen = first.program.state_after.pending[4]
    assert frozen.baseline == 0.01
    boundary = step(p, first.program.state_after, 12, antenna=0.011)
    assert boundary.source == "brush" and boundary.scores["brush"].event_age == 8
    assert boundary.scores["brush"].baseline == 0.01
    expired = step(p, first.program.state_after, 13, antenna=0.1)
    assert expired.source is None and expired.program.state_after.pending[4] is None
    # A counter observed eight ticks late coalesces once, using prior observations.
    coalesced = step(p, state, 12, sensory=sensors(sample(brush=500, tick=4)), antenna=0.011)
    assert coalesced.source == "brush" and coalesced.scores["brush"].event_age == 8
    stale = step(p, state, 13, sensory=sensors(sample(brush=500, tick=4)), antenna=0.1)
    assert stale.source is None


def test_first_observation_and_incomplete_baseline_absorb_events_without_replay():
    p, state = policy(), InteractionPolicyState()
    sensory = sensors(
        sample(food=5, brush=7, attack=9, tick=0, contact=True), actor=1, distance=1, light=0
    )
    for tick in range(4):
        result = step(
            p, state, tick, sensory=sensory, ttmn=-1, jump=1, mn9=1, antenna=1, dna=(1, 1)
        )
        state = result.program.state_after
        assert result.source is None and all(event is None for event in state.pending)
    reset = step(
        p,
        InteractionPolicyState(),
        80,
        sensory=sensory,
        ttmn=-1,
        jump=1,
        mn9=1,
        antenna=1,
        dna=(1, 1),
    )
    assert reset.source is None and len(reset.program.state_after.history) == 1


def test_food_land_to_feed_preserves_event_and_sixty_tick_hold_then_cooldown():
    p, state = ready()
    events = sample(food=1, tick=4, near=True)
    land = step(p, state, 4, sensory=sensors(events), mn9=0.001)
    assert land.program.program is ProgramId.LAND and land.program.state_after.active_until == 204
    trigger = land.program.state_after.held_event
    feed = step(
        p, land.program.state_after, 10, sensory=sensors(replace(events, food_contact=True))
    )
    assert feed.program.program is ProgramId.FEED and feed.reason == "land_to_feed"
    assert feed.program.state_after.active_until == 70
    assert feed.program.state_after.held_event == trigger
    assert feed.scores["food"].baseline == 0 and feed.scores["food"].event_age == 6
    last = step(p, feed.program.state_after, 69)
    assert last.program.program is ProgramId.FEED and last.program.state_after.hold_remaining == 1
    expired = step(p, last.program.state_after, 70)
    assert expired.source is None and expired.program.state_after.cooldown_until == 190
    # Counter during cooldown is consumed, including the immediately preceding tick.
    blocked = step(
        p, expired.program.state_after, 189, **stimulus("food", 189, prior_events=events)
    )
    assert blocked.source is None
    quiet = step(p, blocked.program.state_after, 190, mn9=1)
    assert quiet.source is None
    repeated = step(
        p,
        quiet.program.state_after,
        191,
        **stimulus("food", 191, prior_events=quiet.program.state_after.previous_sensors.events),
    )
    # The mean includes the previous high MN9 sample, so another high background is not sufficient.
    assert repeated.source is None
    fresh = step(
        p,
        blocked.program.state_after,
        190,
        sensory=sensors(
            replace(blocked.program.state_after.previous_sensors.events, food=EventCounter(3, 190))
        ),
        mn9=1,
    )
    assert fresh.source == "food"


@pytest.mark.parametrize("loss", ["near", "fluid", "deadline"])
def test_land_cancels_at_declared_gate_or_exclusive_deadline(loss):
    p, state = ready()
    events = sample(food=1, tick=4, near=True)
    land = step(p, state, 4, sensory=sensors(events), mn9=1)
    tick = 204 if loss == "deadline" else 5
    result = step(
        p,
        land.program.state_after,
        tick,
        sensory=sensors(replace(events, food_near=loss != "near"), fluid=loss == "fluid"),
    )
    assert result.source is None
    assert result.program.state_after.cooldown_until == tick + 120


def test_light_bilateral_signed_averaging_and_background_cruise_readout_preserved():
    p, state = ready(dna=(0.1, 0.2))
    unchanged_mean = step(p, state, 4, sensory=sensors(light=0), dna=(0.2, 0.1))
    assert unchanged_mean.source is None and unchanged_mean.scores["light"].score == 0
    changed = step(p, state, 4, sensory=sensors(light=0), dna=(0.1, 0.2001))
    assert changed.source == "light"
    assert changed.scores["light"].score == pytest.approx(0.00005)
    readout = ReadoutValues(1, -0.3, 0.2)
    no_source = step(p, state, 4, readout=readout)
    assert no_source.program.program is ProgramId.CRUISE
    assert no_source.program.readout is readout
    assert no_source.program.modulation.speed == p.cruise_speed(0.2)
    assert no_source.program.parameters.turn_bias == -0.3


@pytest.mark.parametrize("source", SOURCES)
def test_zero_neural_signals_never_select_an_interaction(source):
    actor = 1 if source == "player_approach" else 2 if source == "mob_approach" else 0
    p, state = ready(actor=actor)
    kwargs = stimulus(source)
    kwargs.update(ttmn=0, dna=(0, 0), mn9=0, antenna=0, jump=0)
    assert step(p, state, 4, **kwargs).source is None


@pytest.mark.parametrize("source", SOURCES)
def test_flee_presentation_speed_keeps_neural_gate_and_other_outputs(source):
    actor = 1 if source == "player_approach" else 2 if source == "mob_approach" else 0
    p, state = ready(actor=actor)
    before = step(replace(p, flee_speed=1.0), state, 4, **stimulus(source))
    after = step(p, state, 4, **stimulus(source))
    assert before.scores == after.scores
    assert before.source == after.source == source
    assert before.program.program is after.program.program
    assert before.program.state_after.active_until == after.program.state_after.active_until
    assert before.program.parameters == after.program.parameters
    if source == "attack":
        assert before.program.modulation.speed == 1.0
        assert after.program.modulation.speed == 0.65
        held = step(p, after.program.state_after, 43)
        assert held.program.program is ProgramId.FLEE
        assert held.program.modulation == after.program.modulation
        assert held.program.state_after.active_until == 44
    else:
        assert before.program.modulation == after.program.modulation


def test_declaration_and_state_are_immutable_and_hashes_track_constants():
    p, state = ready()
    assert p.identity_sha256 == policy().identity_sha256
    for name, value in (
        ("attack_threshold", 0.003),
        ("light_threshold", 2e-6),
        ("event_encoder_sha256", "4" * 64),
        ("bank_turn_bias", -1.0),
        ("motion_hold_ticks", 39),
        ("event_cooldown_ticks", 121),
        ("flee_speed", 0.7),
    ):
        assert replace(p, **{name: value}).identity_sha256 != p.identity_sha256
    result = step(p, state, 4, **stimulus("brush"))
    with pytest.raises(TypeError):
        result.scores["brush"] = result.scores["food"]
    with pytest.raises(FrozenInstanceError):
        result.program.state_after.source = "attack"
    assert state.last_game_tick == 3 and state.source is None


@pytest.mark.parametrize(
    "changes",
    [
        {"attack_threshold": float("nan")},
        {"light_threshold": 0},
        {"escape_on": 0, "escape_off": 0},
        {"motion_hold_ticks": True},
        {"baseline_observations": 3},
        {"event_hold_ticks": 59},
        {"event_speed": 0},
        {"flee_speed": 0},
        {"flee_speed": float("nan")},
        {"flee_speed": 1.1},
        {"flee_speed": True},
        {"motion_turn_rate": 0},
        {"event_urgency": 1},
        {"speed_span": float("inf")},
    ],
)
def test_malformed_policy_constants_refused(changes):
    with pytest.raises(ValueError):
        replace(policy(), **changes)


def test_bad_ticks_readouts_state_and_forged_pulse_are_refused_without_mutation():
    p, state = ready()
    for tick in (3, True, -1, 4.0):
        with pytest.raises(ValueError):
            step(p, state, tick)
    for changes in ({"ttmn": float("nan")}, {"dna": (0, float("inf"))}, {"mn9": 1.1}):
        with pytest.raises(ValueError):
            step(p, state, 4, **changes)
    with pytest.raises(ValueError):
        replace(state, history=((float("nan"), 0, 0, 0),))
    with pytest.raises(ValueError):
        replace(state, program=ProgramId.FLEE)
    helper = InteractionInputState.__new__(InteractionInputState)
    helper._previous = state.previous_sensors.events
    forged = replace(helper.prepare(sample(), 4), attack=True, attack_age=0, attack_delta=1)
    completed = CompletedEventReadout(forged, 0, 0, 0, 0, 0, 0)
    with pytest.raises(EventSensesError, match="pulse disagrees"):
        p.select_interactions(
            QUIET, completed, sensors(), state, 4, ttmn_raw=-1, dna01_means=(0, 0)
        )
    assert state.last_game_tick == 3 and state.source is None


def test_motion_expiry_uses_world_ticks_and_coalesced_cooldown_events_do_not_replay():
    p, state = ready()
    entered = step(p, state, 4, **stimulus("attack"))
    active = step(p, entered.program.state_after, 43)
    assert active.source == "attack" and active.program.state_after.hold_remaining == 1
    # Jump across expiry: the cooldown starts at 44, not the next observed tick.
    late = step(p, active.program.state_after, 160)
    assert late.source is None and late.program.state_after.cooldown_until == 164
    events = late.program.state_after.previous_sensors.events
    missed_during_cooldown = step(
        p,
        late.program.state_after,
        164,
        sensory=sensors(replace(events, attack_attempt=EventCounter(2, 163))),
        ttmn=-0.1,
    )
    assert missed_during_cooldown.source is None
    fresh_on_boundary = step(
        p,
        late.program.state_after,
        164,
        sensory=sensors(replace(events, attack_attempt=EventCounter(2, 164))),
        ttmn=-0.1,
    )
    assert fresh_on_boundary.source == "attack"


def test_continuous_approach_is_one_episode_until_observed_nonapproach():
    p, state = ready(actor=1)
    result = step(p, state, 4, **stimulus("player_approach"))
    state = result.program.state_after
    # Observations may skip ticks, but their distances keep decreasing throughout
    # the hold and cooldown. The same approach cannot automatically retrigger.
    for tick, distance in ((28, 4), (147, 3), (148, 2), (149, 1)):
        result = step(p, state, tick, sensory=sensors(actor=1, distance=distance), jump=1)
        state = result.program.state_after
        assert result.source is None
    stopped = step(p, state, 150, sensory=sensors(actor=1, distance=1), jump=1)
    resumed = step(
        p, stopped.program.state_after, 151, sensory=sensors(actor=1, distance=0.5), jump=1
    )
    assert resumed.source == "player_approach"


@pytest.mark.parametrize("source", ["brush", "food"])
def test_feed_and_groom_early_gate_loss_discards_competing_pulses(source):
    p, state = ready()
    entered = step(p, state, 4, **stimulus(source))
    events = entered.program.state_after.previous_sensors.events
    stopped = step(
        p,
        entered.program.state_after,
        5,
        sensory=sensors(replace(events, attack_attempt=EventCounter(1, 5)), fluid=True),
        ttmn=-1,
    )
    assert stopped.source is None and stopped.reason == "gate_lost"
    assert stopped.program.state_after.cooldown_until == 125
    assert stopped.program.state_after.history[-1][0] == -1
    assert all(event is None for event in stopped.program.state_after.pending)


@pytest.mark.parametrize(
    "actor,source,program",
    [
        (1, "player_approach", ProgramId.ESCAPE_JUMP),
        (2, "mob_approach", ProgramId.BANK),
    ],
)
@pytest.mark.parametrize("ray", [0, 2, 3])
def test_forward_left_right_approach_mirrors_both_motion_and_current_actor_gate(
    actor, source, program, ray
):
    p, state = ready(actor=actor, ray=ray)
    # Freeze an eligible source before its neural gate opens, then prove the
    # current actor gate also retains the right-side pending source.
    onset = step(p, state, 4, sensory=sensors(actor=actor, distance=5, ray=ray), jump=0)
    assert onset.source is None
    assert onset.scores[source].eligible and onset.scores[source].event_age == 0
    delayed = step(
        p, onset.program.state_after, 5, sensory=sensors(actor=actor, distance=5, ray=ray), jump=1
    )
    assert delayed.source == source and delayed.program.program is program
    assert delayed.scores[source].event_age == 1
    assert p.to_identity_material()["approach_ray_indices_forward_left_right"] == [0, 2, 3]
