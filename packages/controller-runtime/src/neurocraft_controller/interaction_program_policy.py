"""Declared V2 interaction gates over actual readouts and scripted programs.

These post-probe development thresholds are engineering choices, not natural
behavior. Actor class comes from Minecraft, not neural actor recognition. Attack
uses either direction of signed TTMn change, not positive escape activation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from neurocraft_protocol import (
    ProgramId,
    ProgramModulation,
    ProgramParameters,
    ReadoutValues,
    SensorFrameV4,
)

from .event_program_policy import EventScore
from .events_senses import MAX_EVENT_AGE_TICKS, CompletedEventReadout, EventSensesError, _hash
from .interaction_senses import InteractionPulse
from .program_policy import PolicyState, ProgramDecision, ProgramPolicyError, ProgramSelectionPolicy

SOURCES = ("attack", "player_approach", "mob_approach", "food", "brush", "light")
PROGRAMS = (
    ProgramId.FLEE,
    ProgramId.ESCAPE_JUMP,
    ProgramId.BANK,
    ProgramId.LAND,
    ProgramId.GROOM,
    ProgramId.HOVER,
)
READOUTS = ("ttmn_absolute_change", "ttmn", "ttmn", "mn9", "antenna", "dna01_absolute_change")
MAX_INTEGER = 2**63 - 1
# Contract order is forward, back, left, right, up, down.
APPROACH_RAY_INDICES = (0, 2, 3)


def _int(value: object, minimum: int = 0, maximum: int = MAX_INTEGER) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise EventSensesError("interaction integers must be bounded and cannot be booleans")


def _mean(value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or not -1.0 <= value <= 1.0
    ):
        raise EventSensesError("interaction activity must be a finite signed model mean")


@dataclass(frozen=True, slots=True)
class InteractionEvent:
    event_tick: int
    baseline: float

    def __post_init__(self) -> None:
        _int(self.event_tick)
        _mean(self.baseline)

    def to_dict(self) -> dict[str, int | float]:
        return {"event_tick": self.event_tick, "baseline": self.baseline}


@dataclass(frozen=True, slots=True)
class InteractionPolicyState(PolicyState):
    """Complete immutable policy memory; construct fresh to reset and absorb history."""

    history: tuple[tuple[float, float, float, float], ...] = ()
    previous_sensors: SensorFrameV4 | None = None
    approaching: tuple[bool, bool] = (False, False)
    pending: tuple[InteractionEvent | None, ...] = (None,) * 6
    source: str | None = None
    held_event: InteractionEvent | None = None
    active_until: int = 0
    cooldown_until: int = 0
    last_game_tick: int = -1
    latched_modulation: ProgramModulation = field(
        default_factory=lambda: ProgramModulation(0.0, 0.0, 1.0, 0.0)
    )
    latched_parameters: ProgramParameters = field(default_factory=lambda: ProgramParameters(0.0))

    def __post_init__(self) -> None:
        for value in (
            self.hold_remaining,
            self.refractory_remaining,
            self.active_until,
            self.cooldown_until,
        ):
            _int(value)
        _int(self.last_game_tick, -1)
        if type(self.armed) is not bool:
            raise EventSensesError("interaction armed must be boolean")
        if self.program not in (*PROGRAMS, ProgramId.CRUISE, ProgramId.FEED):
            raise EventSensesError("program outside the V2 interaction vocabulary")
        active = self.program is not ProgramId.CRUISE
        if active != (self.source is not None) or active != (self.held_event is not None):
            raise EventSensesError("active interaction must retain its source and event")
        if active != (self.hold_remaining > 0):
            raise EventSensesError("active interaction requires a positive hold")
        if self.source is not None:
            if self.source not in SOURCES:
                raise EventSensesError("unknown interaction source")
            expected = PROGRAMS[SOURCES.index(self.source)]
            if self.program != expected and not (
                self.source == "food" and self.program is ProgramId.FEED
            ):
                raise EventSensesError("interaction source cannot be relabeled during a hold")
            if self.hold_remaining != self.active_until - self.last_game_tick:
                raise EventSensesError("interaction hold must use the exclusive game-tick boundary")
        if not isinstance(self.history, tuple) or len(self.history) > 4:
            raise EventSensesError("history holds at most four immutable completed observations")
        for row in self.history:
            if not isinstance(row, tuple) or len(row) != 4:
                raise EventSensesError("history rows require four named raw means")
            for activity in row:
                _mean(activity)
        if (
            not isinstance(self.pending, tuple)
            or len(self.pending) != 6
            or any(
                event is not None and not isinstance(event, InteractionEvent)
                for event in self.pending
            )
        ):
            raise EventSensesError("pending interactions must be six immutable optional events")
        for event in (*self.pending, self.held_event):
            if event is not None and event.event_tick > self.last_game_tick:
                raise EventSensesError("state event cannot occur after its observation")
        if (
            not isinstance(self.approaching, tuple)
            or len(self.approaching) != 2
            or any(type(value) is not bool for value in self.approaching)
        ):
            raise EventSensesError("approach episode latches must be two booleans")
        if (self.previous_sensors is None) != (self.last_game_tick == -1):
            raise EventSensesError("previous sensors and completed observation must agree")
        if self.previous_sensors is not None:
            if not isinstance(self.previous_sensors, SensorFrameV4):
                raise EventSensesError("previous interaction sensors require V4 actor metadata")
            self.previous_sensors.events.require_observation_tick(self.last_game_tick)
            if not self.history:
                raise EventSensesError("a completed observation requires nonempty baseline history")
        elif self.history or any(self.pending) or self.source or any(self.approaching):
            raise EventSensesError("fresh interaction state cannot contain history or events")
        if not isinstance(self.latched_modulation, ProgramModulation) or not isinstance(
            self.latched_parameters, ProgramParameters
        ):
            raise EventSensesError("held outputs require validated immutable program scalars")

    def to_dict(self) -> dict[str, Any]:
        return {
            **PolicyState.to_dict(self),
            "history": self.history,
            "previous_sensors": None
            if self.previous_sensors is None
            else self.previous_sensors.to_dict(),
            "approaching": self.approaching,
            "pending": {
                name: None if event is None else event.to_dict()
                for name, event in zip(SOURCES, self.pending, strict=True)
            },
            "source": self.source,
            "held_event": None if self.held_event is None else self.held_event.to_dict(),
            "active_until": self.active_until,
            "cooldown_until": self.cooldown_until,
            "last_game_tick": self.last_game_tick,
            "latched_modulation": self.latched_modulation.to_dict(),
            "latched_parameters": self.latched_parameters.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class InteractionDecision:
    program: ProgramDecision
    scores: Mapping[str, EventScore]
    source: str | None
    reason: str

    def __post_init__(self) -> None:
        if set(self.scores) != set(SOURCES):
            raise EventSensesError("interaction decision requires all six source scores")
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))


@dataclass(frozen=True, slots=True)
class InteractionSelectionPolicy(ProgramSelectionPolicy):
    event_encoder_sha256: str = ""
    event_binding_sha256: str = ""
    baseline_observations: int = 4
    response_window_ticks: int = 8
    attack_threshold: float = 0.002
    mn9_threshold: float = 1e-4
    antenna_threshold: float = 3e-4
    light_threshold: float = 1e-6
    event_hold_ticks: int = 60
    motion_hold_ticks: int = 40
    landing_deadline_ticks: int = 200
    event_cooldown_ticks: int = 120
    event_speed: float = 1.0
    flee_speed: float = 0.65
    motion_turn_rate: float = 1.0
    event_amplitude: float = 1.0
    event_urgency: float = 0.0
    bank_turn_bias: float = 1.0

    def __post_init__(self) -> None:
        try:
            ProgramSelectionPolicy.__post_init__(self)
        except ProgramPolicyError as error:
            raise EventSensesError(str(error)) from error
        for digest in (
            self.event_encoder_sha256,
            self.event_binding_sha256,
            *self.readouts.to_identity_material().values(),
        ):
            _hash(digest)
        for name in ProgramSelectionPolicy.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or (isinstance(value, float) and not math.isfinite(value)):
                raise EventSensesError("background policy constants must be finite numeric values")
        for value in (
            self.response_window_ticks,
            self.motion_hold_ticks,
            self.landing_deadline_ticks,
            self.event_cooldown_ticks,
            self.escape_hold_ticks,
        ):
            _int(value, 1, 1000)
        _int(self.escape_refractory_ticks, 0, 1000)
        if type(self.baseline_observations) is not int or self.baseline_observations != 4:
            raise EventSensesError("V2 baseline uses exactly four completed observations")
        if type(self.event_hold_ticks) is not int or self.event_hold_ticks != 60:
            raise EventSensesError("FEED and GROOM have the declared 60-tick body hold")
        if self.escape_on <= 0:
            raise EventSensesError("zero neural activity must not pass an approach gate")
        for value in (
            self.attack_threshold,
            self.mn9_threshold,
            self.antenna_threshold,
            self.light_threshold,
        ):
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 2:
                raise EventSensesError("interaction thresholds must be finite and positive")
        ProgramModulation(
            self.event_speed, self.motion_turn_rate, self.event_amplitude, self.event_urgency
        )
        ProgramModulation(self.flee_speed, 0.0, 0.0, 0.0)
        ProgramParameters(self.bank_turn_bias)
        if self.event_speed <= 0 or self.motion_turn_rate <= 0 or self.bank_turn_bias == 0:
            raise EventSensesError("LAND, FLEE and BANK require nonzero declared motion")
        if self.flee_speed <= 0:
            raise EventSensesError("FLEE requires nonzero declared speed")
        if self.event_urgency != 0:
            raise EventSensesError("V2 event programs use urgency zero")

    def to_identity_material(self) -> dict[str, Any]:
        return {
            "schema": "neurocraft.interactions.program_policy.v2",
            "profile": "full_mcns_interactions_program_v2",
            "status": "candidate; post-probe development thresholds; not behavior-validated",
            "background_component": ProgramSelectionPolicy.to_identity_material(self),
            "background_scope": "CRUISE formulas and approach escape outputs only; "
            "old ungated escape state machine is never executed",
            "event_encoder_sha256": self.event_encoder_sha256,
            "event_binding_sha256": self.event_binding_sha256,
            "baseline_observations": self.baseline_observations,
            "response_window_game_ticks": self.response_window_ticks,
            "thresholds": {
                "attack_absolute_ttmn": float(self.attack_threshold).hex(),
                "mn9": float(self.mn9_threshold).hex(),
                "antenna": float(self.antenna_threshold).hex(),
                "light_absolute_bilateral_dna01": float(self.light_threshold).hex(),
                "approach_rectified_ttmn": float(self.escape_on).hex(),
            },
            "priority": list(SOURCES),
            "mapping": dict(zip(SOURCES, [p.value for p in PROGRAMS], strict=True)),
            "score": "attack/light abs(current raw - frozen pre-event mean); food/brush "
            "max(0,current raw - frozen pre-event mean); approaches current rectified ttmn_jump; "
            "light averages actual left/right DNa01 first; not a paired counterfactual",
            "readout_sampling": "last actual state of each five-update observation; "
            "history advances on every valid observation, including blocked events",
            "counter_events": "validate InteractionPulse against monotone observed counters; "
            "first observation absorbs history; jumps coalesce once; ages refer to world ticks",
            "approach_ray_indices_forward_left_right": list(APPROACH_RAY_INDICES),
            "approach_source": "rising edge of any decreasing front/left/right ray with "
            "the same actor category in consecutive observations; actor 1 player, 2 Mob; "
            "newly visible actor alone is insufficient; class metadata is an engineered gate, "
            "not neural actor recognition; category 3 never qualifies",
            "light_source": "observed integer local_light change after first observation; "
            "Minecraft transmits max(block light, sky light); no synthetic light pulse",
            "window": "source age <= response_window_game_ticks, expires afterwards; "
            "baseline freezes on first observed pulse, requires four prior completed samples",
            "pending": "fresh unblocked source replaces its same-kind pending event; "
            "food-near/fluid loss cancels food, fluid cancels brush; missing actor category "
            "cancels pending approach; selection consumes all pending events",
            "preemption": "during a held episode only a fresh higher-priority source already "
            "passing its neural gate on this observation can preempt; other blocked pulses "
            "are discarded, never queued; no preemption during cooldown",
            "hold": "entry modulation and original source/event are latched; no neural "
            "retrigger during hold; exclusive entry tick plus duration; approach actor loss "
            "does not relabel a held program; no receipt feedback",
            "food_sequence": "MN9 plus food-near and not fluid selects FEED on current "
            "contact, else LAND; LAND keeps original event until contact then FEED for 60; "
            "LAND cancels on near loss, fluid or exclusive deadline; FEED cancels on contact "
            "loss or fluid; GROOM cancels in fluid",
            "landing_deadline_game_ticks": self.landing_deadline_ticks,
            "event_hold_game_ticks": self.event_hold_ticks,
            "motion_hold_game_ticks": self.motion_hold_ticks,
            "escape_hold_game_ticks": self.escape_hold_ticks,
            "cooldown_game_ticks": self.event_cooldown_ticks,
            "cooldown": "global exclusive episode end plus cooldown ticks; early gate loss "
            "uses current tick; coalesced expiry uses scheduled active_until; discard events "
            "through cooldown, admit only a fresh event whose world timestamp is at or after "
            "its exclusive end (including coalesced counters)",
            "modulation": {
                "speed": float(self.event_speed).hex(),
                "flee_speed": float(self.flee_speed).hex(),
                "motion_turn_rate": float(self.motion_turn_rate).hex(),
                "amplitude": float(self.event_amplitude).hex(),
                "urgency": float(self.event_urgency).hex(),
                "bank_turn_bias": float(self.bank_turn_bias).hex(),
            },
            "output_scope": "FLEE uses its separate declared presentation speed; "
            "FLEE/BANK use motion turn rate; BANK uses declared signed bias; "
            "LAND/FEED/GROOM/HOVER turn rate and bias zero; ESCAPE_JUMP uses latched "
            "background escape outputs and entry urgency; CRUISE retains actual readouts",
            "application": "issued selections are not applied behavior; FLEE requires a "
            "visible eligible player in Java and may end early; controls share this declaration",
        }

    def select_interactions(
        self,
        readout: ReadoutValues,
        completed: CompletedEventReadout,
        sensors: SensorFrameV4,
        state: InteractionPolicyState,
        game_tick: int,
        *,
        ttmn_raw: float,
        dna01_means: tuple[float, float],
    ) -> InteractionDecision:
        _int(game_tick, 0, MAX_INTEGER - 2000)
        if not isinstance(state, InteractionPolicyState) or game_tick <= state.last_game_tick:
            raise EventSensesError("interaction policy requires increasing game ticks")
        if not isinstance(sensors, SensorFrameV4):
            raise EventSensesError("interaction policy requires the V4 actor contract")
        # Revalidate the public immutable inputs at this pure boundary.
        SensorFrameV4.from_dict(sensors.to_dict())
        ReadoutValues.from_dict(readout.to_dict())
        if not isinstance(completed, CompletedEventReadout):
            raise EventSensesError("interaction policy requires completed event means")
        for value in (
            ttmn_raw,
            completed.mn9_raw,
            completed.antenna_raw,
            completed.mn9_left,
            completed.mn9_right,
            completed.antenna_left,
            completed.antenna_right,
        ):
            _mean(value)
        if not isinstance(dna01_means, tuple) or len(dna01_means) != 2:
            raise EventSensesError("light readout requires two raw DNa01 means")
        for value in dna01_means:
            _mean(value)
        pulse = _checked_pulse(completed, sensors, state.previous_sensors, game_tick)
        raw = (ttmn_raw, completed.mn9_raw, completed.antenna_raw, math.fsum(dna01_means) / 2)
        ready = len(state.history) == self.baseline_observations
        prior = state.previous_sensors
        approaching = tuple(
            prior is not None
            and any(
                sensors.ray_actor_kinds[i] == prior.ray_actor_kinds[i] == actor
                and sensors.ray_distances_m[i] < prior.ray_distances_m[i]
                for i in APPROACH_RAY_INDICES
            )
            for actor in (1, 2)
        )
        fresh = (
            pulse.attack,
            approaching[0] and not state.approaching[0],
            approaching[1] and not state.approaching[1],
            pulse.food,
            pulse.groom,
            prior is not None and sensors.local_light != prior.local_light,
        )
        ages = (pulse.attack_age, 0, 0, pulse.food_age, pulse.groom_age, 0)
        gates = (
            True,
            any(sensors.ray_actor_kinds[i] == 1 for i in APPROACH_RAY_INDICES),
            any(sensors.ray_actor_kinds[i] == 2 for i in APPROACH_RAY_INDICES),
            sensors.events.food_near and not sensors.in_fluid,
            not sensors.in_fluid,
            True,
        )
        raw_scores = (
            ttmn_raw,
            readout.ttmn_jump,
            readout.ttmn_jump,
            completed.mn9_raw,
            completed.antenna_raw,
            raw[3],
        )
        baselines = (0, None, None, 1, 2, 3)
        thresholds = (
            self.attack_threshold,
            self.escape_on,
            self.escape_on,
            self.mn9_threshold,
            self.antenna_threshold,
            self.light_threshold,
        )
        active = state.source is not None and game_tick < state.active_until
        cooldown = state.cooldown_until
        program = state.program if active else ProgramId.CRUISE
        source, held = (state.source, state.held_event) if active else (None, None)
        active_until = state.active_until
        reason = "held" if active else "no_eligible_response"
        if state.source is not None and not active:
            cooldown = state.active_until + self.event_cooldown_ticks
            reason = "hold_expired"
        if active:
            gate = (
                gates[3]
                if program is ProgramId.LAND
                else sensors.events.food_contact and not sensors.in_fluid
                if program is ProgramId.FEED
                else gates[4]
                if program is ProgramId.GROOM
                else True
            )
            if not gate:
                active, program, source, held = False, ProgramId.CRUISE, None, None
                active_until, cooldown = game_tick, game_tick + self.event_cooldown_ticks
                reason = "gate_lost"
            elif program is ProgramId.LAND and sensors.events.food_contact:
                program, active_until, reason = (
                    ProgramId.FEED,
                    game_tick + self.event_hold_ticks,
                    "land_to_feed",
                )
        blocked = active or game_tick < cooldown
        pending = [None] * 6 if blocked else list(state.pending)
        candidates: list[InteractionEvent | None] = []
        for index in range(6):
            event = pending[index]
            if fresh[index] and ready and gates[index]:
                age = ages[index]
                assert age is not None
                column = baselines[index]
                baseline = (
                    0.0 if column is None else math.fsum(row[column] for row in state.history) / 4
                )
                new = InteractionEvent(game_tick - age, baseline)
                can_preempt = active and source is not None and index < SOURCES.index(source)
                if (not blocked or can_preempt) and new.event_tick >= cooldown:
                    event = new
            if event is not None and (
                game_tick - event.event_tick > self.response_window_ticks or not gates[index]
            ):
                event = None
            candidates.append(event)
        scores: dict[str, EventScore] = {}
        selected: int | None = None
        for index, name in enumerate(SOURCES):
            event = candidates[index]
            scored_event = held if active and source == name else event
            current = raw_scores[index]
            score = (
                0.0
                if scored_event is None
                else current
                if index in (1, 2)
                else abs(current - scored_event.baseline)
                if index in (0, 5)
                else max(0.0, current - scored_event.baseline)
            )
            scores[name] = EventScore(
                current,
                None if scored_event is None else scored_event.baseline,
                score,
                gates[index] and scored_event is not None,
                None if scored_event is None else game_tick - scored_event.event_tick,
                ready if scored_event is None else True,
            )
            if selected is None and event is not None and score >= thresholds[index]:
                selected = index
        modulation, parameters = state.latched_modulation, state.latched_parameters
        if selected is not None:
            source, held = SOURCES[selected], candidates[selected]
            program = PROGRAMS[selected]
            reason = "fresh_eligible_response"
            if program is ProgramId.LAND and sensors.events.food_contact:
                program = ProgramId.FEED
            duration = (
                self.landing_deadline_ticks
                if program is ProgramId.LAND
                else self.event_hold_ticks
                if program in (ProgramId.FEED, ProgramId.GROOM)
                else self.escape_hold_ticks
                if program is ProgramId.ESCAPE_JUMP
                else self.motion_hold_ticks
            )
            active_until = game_tick + duration
            if program is ProgramId.ESCAPE_JUMP:
                modulation = ProgramModulation(
                    self.escape_output_speed,
                    self.escape_output_turn_rate,
                    self.escape_output_amplitude,
                    self.escape_urgency(readout.ttmn_jump),
                )
                parameters = ProgramParameters(self.escape_output_turn_bias)
            else:
                modulation = ProgramModulation(
                    self.flee_speed if program is ProgramId.FLEE else self.event_speed,
                    self.motion_turn_rate if program in (ProgramId.FLEE, ProgramId.BANK) else 0.0,
                    self.event_amplitude,
                    self.event_urgency,
                )
                parameters = ProgramParameters(
                    self.bank_turn_bias if program is ProgramId.BANK else 0.0
                )
            candidates = [None] * 6
        elif blocked:
            candidates = [None] * 6
        if source is None:
            modulation = ProgramModulation(
                self.cruise_speed(readout.wing_common),
                self.cruise_output_turn_rate,
                self.cruise_output_amplitude,
                self.cruise_output_urgency,
            )
            parameters = ProgramParameters(readout.dna01_turn)
        after = InteractionPolicyState(
            program=program,
            hold_remaining=0 if source is None else active_until - game_tick,
            refractory_remaining=max(0, cooldown - game_tick),
            armed=source is None,
            history=(*state.history, raw)[-4:],
            previous_sensors=sensors,
            approaching=(approaching[0], approaching[1]),
            pending=tuple(candidates),
            source=source,
            held_event=held,
            active_until=active_until,
            cooldown_until=cooldown,
            last_game_tick=game_tick,
            latched_modulation=modulation,
            latched_parameters=parameters,
        )
        decision = ProgramDecision(
            program,
            modulation,
            parameters,
            readout,
            state,
            after,
            game_tick,
            self.cruise_driving_readout if source is None else READOUTS[SOURCES.index(source)],
        )
        return InteractionDecision(decision, scores, source, reason)


def _checked_pulse(
    completed: CompletedEventReadout,
    sensors: SensorFrameV4,
    previous: SensorFrameV4 | None,
    game_tick: int,
) -> InteractionPulse:
    pulse = completed.pulse
    if not isinstance(pulse, InteractionPulse):
        raise EventSensesError("V2 selection requires separate brush and attack pulses")
    sensors.events.require_observation_tick(game_tick)
    flags = (pulse.food, pulse.groom, pulse.attack)
    ages = (pulse.food_age, pulse.groom_age, pulse.attack_age)
    deltas = (pulse.food_delta, pulse.brush_delta, pulse.attack_delta)
    if type(pulse.baseline_absorbed) is not bool or pulse.baseline_absorbed != (previous is None):
        raise EventSensesError("pulse reset baseline must match policy observation history")
    for index, name in enumerate(sensors.events.counter_names()):
        counter = getattr(sensors.events, name)
        old = None if previous is None else getattr(previous.events, name)
        delta = 0 if old is None else counter.count - old.count
        if old is not None and (
            delta < 0
            or (delta == 0 and counter.last_game_tick != old.last_game_tick)
            or (
                old.last_game_tick is not None
                and counter.last_game_tick is not None
                and counter.last_game_tick < old.last_game_tick
            )
        ):
            raise EventSensesError("interaction event counters or timestamps moved inconsistently")
        age = None if counter.last_game_tick is None else game_tick - counter.last_game_tick
        expected = delta > 0 and age is not None and age <= MAX_EVENT_AGE_TICKS
        _int(deltas[index])
        if ages[index] is not None:
            _int(ages[index])
        if (
            type(flags[index]) is not bool
            or flags[index] != expected
            or deltas[index] != delta
            or ages[index] != (age if expected else None)
        ):
            raise EventSensesError("interaction pulse disagrees with actual observed counters")
    return pulse
