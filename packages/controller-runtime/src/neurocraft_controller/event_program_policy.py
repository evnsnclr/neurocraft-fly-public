"""Candidate five-program engineering policy; thresholds are provisional declarations.

MN9 does not discriminate food by itself: real food-near/contact gates are required.
Scores compare current signed model means to a frozen four-observation mean,
NOT to an unavailable live counterfactual. Applied behavior belongs to receipts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from neurocraft_protocol import (
    Observation,
    ProgramAction,
    ProgramId,
    ProgramModulation,
    ProgramParameters,
    ReadoutValues,
    SensorFrameV3,
)

from .events_senses import CompletedEventReadout, EventSensesError
from .full_brain_controller import FullBrainController
from .named_event_evidence import EventSelectionTrigger, NamedEventEvidence
from .program_policy import (
    PolicyState,
    ProgramDecision,
    ProgramSelectingController,
    ProgramSelectionPolicy,
    ProgramStepEvidence,
    policy_for_controller,
    readout_values_from_step,
)


@dataclass(frozen=True, slots=True)
class PendingEvent:
    event_tick: int
    baseline: float


@dataclass(frozen=True, slots=True)
class EventPolicyState(PolicyState):
    """Top-level selected state plus the unchanged escape-policy component state."""

    escape_state: PolicyState = field(default_factory=PolicyState)
    history: tuple[tuple[float, float], ...] = ()
    food: PendingEvent | None = None
    groom: PendingEvent | None = None
    held_event: PendingEvent | None = None
    active_until: int = 0
    cooldown_until: int = 0
    last_game_tick: int = -1
    latched_modulation: ProgramModulation = field(
        default_factory=lambda: ProgramModulation(0.0, 0.0, 1.0, 0.0)
    )

    def __post_init__(self) -> None:
        if self.program not in (
            ProgramId.CRUISE,
            ProgramId.ESCAPE_JUMP,
            ProgramId.LAND,
            ProgramId.FEED,
            ProgramId.GROOM,
        ):
            raise EventSensesError("program outside the declared candidate vocabulary")
        if self.hold_remaining < 0 or self.refractory_remaining < 0:
            raise EventSensesError("candidate hold/refractory counters must be nonnegative")
        if (self.program is not ProgramId.CRUISE) != (self.hold_remaining > 0):
            raise EventSensesError("held candidate program requires a positive hold counter")
        if any(
            not math.isfinite(value) or not -1 <= value <= 1
            for row in self.history
            for value in row
        ):
            raise EventSensesError("baseline history must contain finite signed model means")

    def to_dict(self) -> dict[str, Any]:
        return {
            **PolicyState.to_dict(self),
            "escape_component": self.escape_state.to_dict(),
            "history": self.history,
            "food": None
            if self.food is None
            else {"event_tick": self.food.event_tick, "baseline": self.food.baseline},
            "groom": None
            if self.groom is None
            else {"event_tick": self.groom.event_tick, "baseline": self.groom.baseline},
            "held_event": None
            if self.held_event is None
            else {"event_tick": self.held_event.event_tick, "baseline": self.held_event.baseline},
            "active_until": self.active_until,
            "cooldown_until": self.cooldown_until,
            "last_game_tick": self.last_game_tick,
            "latched_modulation": self.latched_modulation.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class EventScore:
    raw: float
    baseline: float | None
    score: float
    eligible: bool
    event_age: int | None
    baseline_ready: bool

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class EventDecision:
    program: ProgramDecision
    mn9: EventScore
    antenna: EventScore
    reason: str


@dataclass(frozen=True, slots=True)
class EventSelectionPolicy(ProgramSelectionPolicy):
    event_encoder_sha256: str = ""
    event_binding_sha256: str = ""
    baseline_observations: int = 4
    response_window_ticks: int = 8
    mn9_threshold: float = 1e-4
    antenna_threshold: float = 3e-4
    event_hold_ticks: int = 60
    landing_deadline_ticks: int = 200
    event_cooldown_ticks: int = 120
    # LAND multiplies descent by speed; FEED/GROOM ignore this scalar.
    event_speed: float = 1.0
    event_turn_rate: float = 0.0
    event_amplitude: float = 1.0
    event_urgency: float = 0.0

    def __post_init__(self) -> None:
        from .events_senses import _hash

        ProgramSelectionPolicy.__post_init__(self)
        _hash(self.event_encoder_sha256)
        _hash(self.event_binding_sha256)
        for value in (
            self.baseline_observations,
            self.response_window_ticks,
            self.event_hold_ticks,
            self.landing_deadline_ticks,
            self.event_cooldown_ticks,
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 1000:
                raise EventSensesError("event policy durations must be bounded positive integers")
        for threshold in (self.mn9_threshold, self.antenna_threshold):
            if (
                isinstance(threshold, bool)
                or not math.isfinite(threshold)
                or not 0 < threshold <= 2
            ):
                raise EventSensesError("event thresholds must be finite and positive")
        ProgramModulation(
            self.event_speed, self.event_turn_rate, self.event_amplitude, self.event_urgency
        )
        if self.event_speed <= 0.0:
            raise EventSensesError("candidate LAND requires nonzero descent speed")
        if self.event_urgency != 0.0 or self.event_hold_ticks != 60:
            raise EventSensesError(
                "candidate FEED/GROOM binding requires urgency zero and 60 ticks"
            )

    def to_identity_material(self) -> dict[str, Any]:
        return {
            "schema": "neurocraft.events.program_policy.v1",
            "profile": "full_mcns_events_program_v1",
            "status": "candidate; provisional thresholds; not behavior-validated",
            "background_policy": ProgramSelectionPolicy.to_identity_material(self),
            "event_encoder_sha256": self.event_encoder_sha256,
            "event_binding_sha256": self.event_binding_sha256,
            "baseline_observations": self.baseline_observations,
            "response_window_game_ticks": self.response_window_ticks,
            "thresholds": {
                "mn9": float(self.mn9_threshold).hex(),
                "antenna": float(self.antenna_threshold).hex(),
            },
            "score": "max(0, current raw population mean - mean of N completed pre-event means); "
            "baseline frozen on pulse; not a paired counterfactual",
            "readout_sampling": "last actual neural state of each five-update observation",
            "window": "world event age <= response_window_game_ticks; expires after boundary",
            "new_event": "fresh pulse replaces same-kind pending event only when not blocked",
            "priority": ["ESCAPE_JUMP", "food sequence LAND/FEED", "GROOM", "CRUISE"],
            "food_gate": "current sensors.events.food_near and not sensors.in_fluid",
            "feed_gate": "current sensors.events.food_contact",
            "food_sequence": "eligible MN9 score selects FEED if contact else LAND; "
            "LAND latches original scored event until contact; no seeking",
            "landing_deadline_game_ticks": self.landing_deadline_ticks,
            "landing_cancel": "food-near loss, fluid, timeout, ESCAPE, or reset; never renewed",
            "landing_timeout": "exclusive entry tick + deadline; then cooldown and fresh event",
            "groom_gate": "not sensors.in_fluid",
            "competing_events": "selection consumes both; escape discards pending events",
            "blocked_events": "pulses during escape, held event program, or cooldown are discarded",
            "hold_game_ticks": self.event_hold_ticks,
            "cooldown_game_ticks": self.event_cooldown_ticks,
            "hold": "latch entry modulation; stop on gate loss, escape, or exclusive active_until",
            "cooldown": (
                f"{self.event_cooldown_ticks} world ticks after normal/early end; "
                "fresh event required afterwards"
            ),
            "event_modulation": ProgramModulation(
                self.event_speed, self.event_turn_rate, self.event_amplitude, self.event_urgency
            ).to_dict(),
            "event_parameters": {"turn_bias": 0.0},
            "event_speed_scope": "LAND descent attenuation; FEED/GROOM ignore speed",
            "receipt_feedback": "none; issued selection is not proof of applied body behavior",
        }

    def select_events(
        self,
        readout: ReadoutValues,
        completed: CompletedEventReadout,
        sensors: SensorFrameV3,
        state: EventPolicyState,
        game_tick: int,
    ) -> EventDecision:
        if game_tick <= state.last_game_tick:
            raise EventSensesError("event policy game tick must increase")
        escape = ProgramSelectionPolicy.select(self, readout, state.escape_state, game_tick)
        ready = len(state.history) == self.baseline_observations
        held = state.program in (ProgramId.LAND, ProgramId.FEED, ProgramId.GROOM)
        active = held and game_tick < state.active_until
        blocked = (
            active or game_tick < state.cooldown_until or escape.program is ProgramId.ESCAPE_JUMP
        )
        food, groom = state.food, state.groom
        if blocked:
            food = groom = None
        elif ready:
            if completed.pulse.food:
                assert completed.pulse.food_age is not None
                food = PendingEvent(
                    game_tick - completed.pulse.food_age,
                    math.fsum(row[0] for row in state.history) / self.baseline_observations,
                )
            if completed.pulse.groom:
                assert completed.pulse.groom_age is not None
                groom = PendingEvent(
                    game_tick - completed.pulse.groom_age,
                    math.fsum(row[1] for row in state.history) / self.baseline_observations,
                )
        if food is not None and game_tick - food.event_tick > self.response_window_ticks:
            food = None
        if groom is not None and game_tick - groom.event_tick > self.response_window_ticks:
            groom = None
        food_gate = sensors.events.food_near and not sensors.in_fluid
        groom_gate = not sensors.in_fluid
        if not food_gate:
            food = None
        if not groom_gate:
            groom = None

        def score(raw: float, event: PendingEvent | None, gate: bool) -> EventScore:
            return EventScore(
                raw,
                None if event is None else event.baseline,
                0.0 if event is None else max(0.0, raw - event.baseline),
                gate,
                None if event is None else game_tick - event.event_tick,
                ready if event is None else True,
            )

        mn9 = score(
            completed.mn9_raw,
            state.held_event
            if active and state.program in (ProgramId.LAND, ProgramId.FEED)
            else food,
            food_gate,
        )
        antenna = score(
            completed.antenna_raw,
            state.held_event if active and state.program is ProgramId.GROOM else groom,
            groom_gate,
        )
        program, modulation, parameters = escape.program, escape.modulation, escape.parameters
        active_until, cooldown = state.active_until, state.cooldown_until
        latched = state.latched_modulation
        held_event = state.held_event if active else None
        reason = "escape" if program is ProgramId.ESCAPE_JUMP else "no_eligible_response"
        if program is ProgramId.ESCAPE_JUMP:
            food = groom = None
            if active:
                active_until, cooldown = game_tick, game_tick + self.event_cooldown_ticks
        elif active:
            gate = (
                food_gate
                if state.program is ProgramId.LAND
                else sensors.events.food_contact
                if state.program is ProgramId.FEED
                else groom_gate
            )
            if gate:
                program, modulation, parameters = state.program, latched, ProgramParameters(0.0)
                reason = "held"
                if state.program is ProgramId.LAND and sensors.events.food_contact:
                    program = ProgramId.FEED
                    active_until = game_tick + self.event_hold_ticks
                    cooldown = active_until + self.event_cooldown_ticks
                    reason = "landed_feed"
            else:
                active_until, cooldown = game_tick, game_tick + self.event_cooldown_ticks
                reason = "gate_lost"
        elif game_tick < cooldown:
            reason = "land_timeout" if state.program is ProgramId.LAND else "cooldown"
        else:
            selected = (
                (ProgramId.FEED if sensors.events.food_contact else ProgramId.LAND)
                if food is not None and food_gate and mn9.score >= self.mn9_threshold
                else ProgramId.GROOM
                if groom is not None and groom_gate and antenna.score >= self.antenna_threshold
                else None
            )
            if selected is not None:
                program = selected
                held_event = food if selected in (ProgramId.LAND, ProgramId.FEED) else groom
                latched = ProgramModulation(
                    self.event_speed, self.event_turn_rate, self.event_amplitude, self.event_urgency
                )
                modulation, parameters = latched, ProgramParameters(0.0)
                active_until = game_tick + (
                    self.landing_deadline_ticks
                    if selected is ProgramId.LAND
                    else self.event_hold_ticks
                )
                cooldown = active_until + self.event_cooldown_ticks
                food = groom = None
                reason = "fresh_eligible_response"
            elif not ready:
                reason = "baseline_not_ready"
        after = EventPolicyState(
            program=program,
            hold_remaining=(
                active_until - game_tick
                if program in (ProgramId.LAND, ProgramId.FEED, ProgramId.GROOM)
                else escape.state_after.hold_remaining
            ),
            refractory_remaining=escape.state_after.refractory_remaining,
            armed=escape.state_after.armed,
            escape_state=escape.state_after,
            history=(*state.history, (completed.mn9_raw, completed.antenna_raw))[
                -self.baseline_observations :
            ],
            food=food,
            groom=groom,
            held_event=held_event
            if program in (ProgramId.LAND, ProgramId.FEED, ProgramId.GROOM)
            else None,
            active_until=active_until,
            cooldown_until=cooldown,
            last_game_tick=game_tick,
            latched_modulation=latched,
        )
        decision = ProgramDecision(
            program,
            modulation,
            parameters,
            readout,
            state,
            after,
            game_tick,
            "mn9"
            if program in (ProgramId.LAND, ProgramId.FEED)
            else "antenna"
            if program is ProgramId.GROOM
            else escape.driving_readout,
        )
        return EventDecision(decision, mn9, antenna, reason)


class EventProgramSelectingController(ProgramSelectingController):
    """Existing command envelope, genuine old readouts, separately named event evidence."""

    def __init__(
        self, controller: FullBrainController, policy: EventSelectionPolicy | None = None
    ) -> None:
        encoder = controller.event_encoder
        if encoder is None:
            raise EventSensesError("event program controller requires the candidate event encoder")
        bound = EventSelectionPolicy(
            readouts=policy_for_controller(controller).readouts,
            event_encoder_sha256=encoder.identity_sha256,
            event_binding_sha256=encoder.populations.identity_sha256,
        )
        if policy is not None and (
            policy.event_encoder_sha256 != bound.event_encoder_sha256
            or policy.event_binding_sha256 != bound.event_binding_sha256
        ):
            raise EventSensesError("event policy binding differs from the actual controller")
        super().__init__(controller, policy=policy or bound)
        self._state = EventPolicyState()
        self._last_event_evidence: NamedEventEvidence | None = None
        self._event_trigger: EventSelectionTrigger | None = None

    @property
    def policy(self) -> EventSelectionPolicy:
        assert isinstance(self._policy, EventSelectionPolicy)
        return self._policy

    @property
    def policy_state(self) -> EventPolicyState:
        assert isinstance(self._state, EventPolicyState)
        return self._state

    @property
    def last_event_evidence(self) -> NamedEventEvidence | None:
        return self._last_event_evidence

    def reset(self) -> None:
        super().reset()
        self._state = EventPolicyState()
        self._last_event_evidence = None
        self._event_trigger = None

    def reset_session(self, *, session_epoch: int) -> None:
        super().reset_session(session_epoch=session_epoch)
        self._state = EventPolicyState()
        self._last_event_evidence = None
        self._event_trigger = None

    def step(self, observation: Observation, *, command_seq: int) -> ProgramAction:
        action = self.controller.step(observation, command_seq=command_seq)
        inner = self.controller.last_evidence
        completed = self.controller.last_event_readout
        assert inner is not None and completed is not None
        assert isinstance(observation.sensors, SensorFrameV3)
        readout = readout_values_from_step(
            inner,
            self.controller.steering_readout_means(),
            policy=self.policy,
        )
        decision = self.policy.select_events(
            readout, completed, observation.sensors, self.policy_state, observation.game_tick
        )
        self._state = decision.program.state_after
        frame = self._program_action(action, decision.program, readout)
        self._last_evidence = ProgramStepEvidence(
            inner, readout, decision.program, action.motors, self.policy_hash, frame
        )
        if decision.reason == "fresh_eligible_response":
            name, score = (
                ("antenna", decision.antenna)
                if frame.program is ProgramId.GROOM
                else ("mn9", decision.mn9)
            )
            self._event_trigger = EventSelectionTrigger(
                frame, name, score, observation.sensors.events
            )
        elif frame.program not in (ProgramId.LAND, ProgramId.FEED, ProgramId.GROOM):
            self._event_trigger = None
        self._last_event_evidence = NamedEventEvidence(
            frame,
            self.policy.event_encoder_sha256,
            self.policy.event_binding_sha256,
            completed,
            decision.mn9,
            decision.antenna,
            decision.reason,
            observation.sensors.events,
            self._event_trigger,
        )
        return frame
