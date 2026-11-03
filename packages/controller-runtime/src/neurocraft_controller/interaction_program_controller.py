"""V4 input/core wrapper for the declared six-interaction candidate policy."""

from __future__ import annotations

from neurocraft_protocol import Observation, ProgramAction, SensorFrameV4

from .events_senses import EventSensesError
from .full_brain_controller import FullBrainController
from .interaction_program_policy import InteractionPolicyState, InteractionSelectionPolicy
from .interaction_senses import InteractionEncoderBinding
from .named_interaction_evidence import (
    SOURCE_CHANNELS,
    InteractionSelectionTrigger,
    NamedInteractionEvidence,
)
from .program_policy import (
    ProgramSelectingController,
    ProgramStepEvidence,
    policy_for_controller,
    readout_values_from_step,
)


class InteractionProgramSelectingController(ProgramSelectingController):
    """Actual model readouts and unchanged command timing; actor context is engineered."""

    def __init__(
        self, controller: FullBrainController, policy: InteractionSelectionPolicy | None = None
    ) -> None:
        encoder = controller.event_encoder
        if not isinstance(encoder, InteractionEncoderBinding):
            raise EventSensesError("interaction policy requires the separate V4 input binding")
        bound = InteractionSelectionPolicy(
            readouts=policy_for_controller(controller).readouts,
            event_encoder_sha256=encoder.identity_sha256,
            event_binding_sha256=encoder.populations.identity_sha256,
        )
        if policy is not None and (
            policy.event_encoder_sha256 != bound.event_encoder_sha256
            or policy.event_binding_sha256 != bound.event_binding_sha256
        ):
            raise EventSensesError("interaction policy binding differs from actual controller")
        super().__init__(controller, policy=policy or bound)
        self._state = InteractionPolicyState()
        self._last_event_evidence: NamedInteractionEvidence | None = None
        self._interaction_trigger: InteractionSelectionTrigger | None = None

    @property
    def policy(self) -> InteractionSelectionPolicy:
        assert isinstance(self._policy, InteractionSelectionPolicy)
        return self._policy

    @property
    def policy_state(self) -> InteractionPolicyState:
        assert isinstance(self._state, InteractionPolicyState)
        return self._state

    @property
    def last_event_evidence(self) -> NamedInteractionEvidence | None:
        return self._last_event_evidence

    def reset(self) -> None:
        super().reset()
        self._state = InteractionPolicyState()
        self._last_event_evidence = None
        self._interaction_trigger = None

    def reset_session(self, *, session_epoch: int) -> None:
        super().reset_session(session_epoch=session_epoch)
        self._state = InteractionPolicyState()
        self._last_event_evidence = None
        self._interaction_trigger = None

    def step(self, observation: Observation, *, command_seq: int) -> ProgramAction:
        action = self.controller.step(observation, command_seq=command_seq)
        inner = self.controller.last_evidence
        completed = self.controller.last_event_readout
        assert inner is not None and completed is not None
        assert isinstance(observation.sensors, SensorFrameV4)
        steering = self.controller.steering_readout_means()
        raw_ttmn = self.controller.jump_readout_mean()
        readout = readout_values_from_step(
            inner,
            steering,
            policy=self.policy,
        )
        decision = self.policy.select_interactions(
            readout,
            completed,
            observation.sensors,
            self.policy_state,
            observation.game_tick,
            ttmn_raw=raw_ttmn,
            dna01_means=steering,
        )
        self._state = decision.program.state_after
        frame = self._program_action(action, decision.program, readout)
        self._last_evidence = ProgramStepEvidence(
            inner, readout, decision.program, action.motors, self.policy_hash, frame
        )
        if decision.reason == "fresh_eligible_response":
            if decision.source is None:
                raise EventSensesError("a new interaction selection must declare its source")
            formula, name = SOURCE_CHANNELS[decision.source]
            self._interaction_trigger = InteractionSelectionTrigger(
                frame,
                name,
                decision.scores[decision.source],
                observation.sensors.events,
                decision.source,
                formula,
            )
        elif decision.source is None:
            self._interaction_trigger = None
        if decision.source is not None and (
            self._interaction_trigger is None or self._interaction_trigger.source != decision.source
        ):
            raise EventSensesError("held interaction must retain its original issued trigger")
        self._last_event_evidence = NamedInteractionEvidence(
            frame,
            self.policy.event_encoder_sha256,
            self.policy.event_binding_sha256,
            completed,
            decision,
            observation.sensors,
            raw_ttmn,
            steering,
            self._interaction_trigger,
        )
        return frame
