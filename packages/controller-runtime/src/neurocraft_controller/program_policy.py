"""The declared readout-to-program selection policy, and the controller that issues it.

**Everything in this module is a hand-chosen engineering mapping.** It is not the fly
deciding, it is not anatomy, and no constant here is derived from any measurement of fly
behaviour. The policy is a pure function of three labeled readout scalars and an explicit
state; every constant carries the reason it was chosen, in a string beside it, so it can be
argued with; and the whole declaration is hashed into ``policy_hash``, which rides on every
``program_action`` frame precisely so that no surface can present the selection as the
connectome deciding anything. The programs it names are scripted presentation (ADR 0020);
the architecture -- labeled readouts selecting scripted body programs over event-gated
inputs -- is the one ADR 0019 describes, described as exactly that.

**``policy_hash`` is a DECLARATION hash, and the distinction matters.** Every numeric
constant and every per-state output scalar is a dataclass field that :meth:`select` reads
and :meth:`ProgramSelectionPolicy.to_identity_material` hashes, so for those the hash and
the executed behaviour cannot disagree. The *formulas*, however, enter the hash as prose
strings: nothing machine-checks that ``DNA01_TURN_FORMULA`` describes what
:func:`readout_values_from_step` computes. A sign flip inside that function changes the
behaviour and leaves ``policy_hash`` untouched -- which is exactly what the reverted NC4
control demonstrated. What guards the code-to-declaration agreement is the unit tests
(``test_dna01_turn_sign_and_deadband_on_the_census_rows`` and its siblings), not the hash.
Read ``policy_hash`` as "which declaration was in force", never as "the arithmetic was
this".

What this module does NOT establish
-----------------------------------
That the connectome selected a program. That the mapping is good, tuned, or biological.
That a fly reacts to a player. That ``dna01_turn``'s declared sign matches any organism.
That the readouts mean "turn", "escape", or "go". The policy reads three numbers the
existing decoder and two hashed pool bindings already produce, and applies arithmetic that
was written down before it was run.

Structure
---------
* :class:`ReadoutValues` (from ``neurocraft_protocol``) -- the three scalars, built here by
  :func:`readout_values_from_step`.
* :class:`PolicyState` -- the four-field anti-flapping state.
* :class:`ProgramDecision` -- one tick's output plus the state it produced.
* :class:`ProgramSelectionPolicy` -- the declared constants, the pure :meth:`select`, and
  :attr:`identity_sha256`.
* :class:`ProgramSelectingController` -- wraps a :class:`FullBrainController`, runs it
  unchanged, and issues a ``program_action`` instead of the raw ``action``. The raw motor
  projection the inner controller computed is recorded in ``last_evidence`` as
  ``raw_projection`` and never sent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Final

from neurocraft_protocol import (
    Action,
    MotorVector,
    Observation,
    ProgramAction,
    ProgramId,
    ProgramModulation,
    ProgramParameters,
    ReadoutValues,
    RunIdentity,
    RunMode,
)

from .config import canonical_sha256
from .full_brain_controller import (
    EscapeMotorBindingV2,
    FullBrainConfiguration,
    FullBrainController,
    FullBrainStepEvidence,
)

if TYPE_CHECKING:
    from neurocraft_neural.full_engine import ArrayNativeEngine

    from .steering_readout import SteeringPairBinding

PROGRAM_POLICY_SCHEMA: Final = "neurocraft.program_policy.v1"

#: The two programs this policy can select. ``HOVER`` is NOT in this vocabulary: it is the
#: authority's declared refusal fallback (ADR 0020 section 2), which the policy never asks
#: for and never sees itself ask for.
POLICY_VOCABULARY: Final = (ProgramId.CRUISE, ProgramId.ESCAPE_JUMP)
POLICY_FALLBACK: Final = ProgramId.HOVER


class ProgramPolicyError(RuntimeError):
    """The selection policy or its controller could not run exactly as declared."""


# --------------------------------------------------------------------- constants

# Each constant below is a DECLARED ENGINEERING CONSTANT. The rationale string beside it
# is hashed into the policy identity along with the value, so changing either rotates
# ``policy_hash`` and every frame that carries it.

#: Derived, not eyeballed. The rule was written down before the measurement:
#: ``ESCAPE_ON = 0.5 x (peak jump on the standard 20-tick approach ramp under
#: full_mcns_loom_v1)``. The measured peak is 0.02277816208187006 at tick 21 of the ramp
#: (front-ray proximity rising linearly 0 -> blocked over 20 ticks, then held), which
#: reproduces the value ``docs/encoders/eem-loom-v1.md`` records for the same stimulus
#: (0.022778, t=21). Half of a peak approach transient, so an approach crosses the
#: threshold about half way up its rise and a smaller transient does not. The design's
#: original 0.05 was declared against escape-v2 numbers and, under loom-v1, would never
#: have been crossed by an approach at all.
ESCAPE_ON: Final = float.fromhex("0x1.7532894d187b1p-7")  # 0.01138908104093503

#: Half of ``ESCAPE_ON``: hysteresis, so a term hovering at the threshold cannot re-arm
#: and retrigger on consecutive ticks.
ESCAPE_OFF: Final = float.fromhex("0x1.7532894d187b1p-8")  # 0.005694540520467515

#: Equal to the ESCAPE_JUMP body program's fixed duration
#: (``EscapeJumpProgram.ESCAPE_JUMP_TICKS``, 24 ticks for every urgency). The policy hold
#: and the program duration must agree exactly, or the program's own exit and the next
#: frame's re-selection produce a one-tick flap.
#: ``test_escape_hold_equals_the_java_program_duration`` reads the Java constant.
ESCAPE_HOLD_TICKS: Final = 24

#: 2 s at 20 Hz. Long enough that a jump into a wall cannot chain into a second jump
#: before CRUISE has had time to turn away.
ESCAPE_REFRACTORY_TICKS: Final = 40

#: ``urgency = clip((ttmn_jump - ESCAPE_ON) / ESCAPE_URGENCY_SPAN, 0, 1)``. Derived by its
#: own rule, declared before the measurement it uses:
#: ``ESCAPE_URGENCY_SPAN = (peak jump on the standard 20-tick approach ramp under
#: full_mcns_loom_v1) - ESCAPE_ON``. So the standard approach reaches urgency exactly 1.0
#: at its peak, and a transient sitting exactly at the threshold gives exactly 0.0.
#: The measured peak is 0.02277816208187006, so the span is 0.01138908104093503 -- which
#: is numerically equal to ``ESCAPE_ON``, and that is a consequence of the two rules
#: (``ESCAPE_ON`` is half the same peak) rather than a third choice.
#: The design's original 0.25 was declared against escape-v2 magnitudes; under loom-v1 it
#: capped the whole approach transient at urgency 0.046, which would have run the
#: ESCAPE_JUMP program at essentially its declared floor for every stimulus the probe can
#: produce.
ESCAPE_URGENCY_SPAN: Final = float.fromhex("0x1.7532894d187b1p-7")  # 0.01138908104093503

#: The census baseline DNa01 pooled magnitude is about 1.1e-4 and its baseline L/R
#: difference about 7.8e-6; the floor in the denominator keeps that baseline difference
#: from normalizing to a full-scale bias when both members sit near zero.
DNA01_FLOOR: Final = 1.0e-4

#: Zeroes noise-dominated small biases: below this the normalized difference is reported
#: as exactly 0.0 rather than as a small turn.
TURN_DEADBAND: Final = 0.10

#: CRUISE never drops below half its declared thrust. A floor of 0 would let the wing
#: readout's negative baseline (about -0.007 in the one observed live session) park the
#: fly, and the demo default must keep moving. The readout sets cruise speed inside a
#: declared band; it does not decide whether to fly.
SPEED_FLOOR: Final = 0.5

#: ``speed = SPEED_FLOOR + (1 - SPEED_FLOOR) * clip(max(0, wing_common) / SPEED_SPAN, 0, 1)``.
#: The one observed live session's in-flight common wing term was 0.13-0.25, so the band
#: is inside the range that was actually seen.
SPEED_SPAN: Final = 0.25

# The per-state output table (design section 3.3). Declared as constants and carried as
# policy fields so that ``policy_hash`` covers them: before this, a change to any of these
# scalars changed what the fly did and left the identity untouched, which made the claim
# "a one-constant change rotates it" false for exactly the values a viewer sees.

#: ESCAPE_JUMP runs at full authority: the program's own declared maxima are the bound, and
#: the policy does not attenuate a reflex it has already decided to select.
ESCAPE_OUTPUT_SPEED: Final = 1.0
ESCAPE_OUTPUT_TURN_RATE: Final = 1.0
ESCAPE_OUTPUT_AMPLITUDE: Final = 1.0
#: An escape does not steer: the turn bias is released to zero for the whole hold, so the
#: jump direction is the program's business and not the steering readout's.
ESCAPE_OUTPUT_TURN_BIAS: Final = 0.0
#: CRUISE turns and beats at full declared rate; only its speed and bias are modulated.
CRUISE_OUTPUT_TURN_RATE: Final = 1.0
CRUISE_OUTPUT_AMPLITUDE: Final = 1.0
#: CRUISE is never urgent by declaration; urgency is the escape channel alone.
CRUISE_OUTPUT_URGENCY: Final = 0.0

#: Presentation labels naming which labeled readout the current selection is a function of.
#: Hashed because they are what a HUD or log prints beside the program name.
ESCAPE_DRIVING_READOUT: Final = "ttmn"
CRUISE_DRIVING_READOUT: Final = "wing"

#: The declared formulas, hashed verbatim into the policy identity.
DNA01_TURN_FORMULA: Final = (
    "dna01_turn = (right - left) / (|left| + |right| + DNA01_FLOOR), where left and right "
    "are the pre-gain committed activities of the declared DNa01 L and R members; then "
    "0.0 when |dna01_turn| < TURN_DEADBAND, then clip to [-1, 1]. Sign convention "
    "(DECLARED, not measured): positive means turn toward the body's right (ADR 0015 "
    "positive yaw). Chosen so the escape-v2 census rows read as turning AWAY from a "
    "blocked side; whether that survives loom-v1 is what the offline probe measures"
)
CRUISE_SPEED_FORMULA: Final = (
    "wing_common = 0.5 * (left_motor + right_motor) from the pinned projection's decoded "
    "pool values; speed = SPEED_FLOOR + (1 - SPEED_FLOOR) * "
    "clip(max(0.0, wing_common) / SPEED_SPAN, 0, 1)"
)
ESCAPE_URGENCY_FORMULA: Final = (
    "urgency = clip((ttmn_jump - ESCAPE_ON) / ESCAPE_URGENCY_SPAN, 0, 1), where ttmn_jump "
    "is the rectified TTMn decode the escape-v2 decoder already produces. Both constants "
    "are derived from one measured quantity, the peak jump on the standard 20-tick "
    "approach ramp under full_mcns_loom_v1: ESCAPE_ON is half of it and the span is the "
    "remainder, so that ramp reaches urgency 1.0 exactly at its peak"
)
STATE_MACHINE_FORMULA: Final = (
    "CRUISE: if armed and refractory_remaining == 0 and ttmn_jump >= ESCAPE_ON then "
    "ESCAPE_JUMP with hold_remaining = ESCAPE_HOLD_TICKS and armed = False; otherwise "
    "stay CRUISE, refractory_remaining = max(0, refractory_remaining - 1), and "
    "armed = armed or ttmn_jump < ESCAPE_OFF. ESCAPE_JUMP: hold_remaining -= 1, and when "
    "it reaches 0 return to CRUISE with refractory_remaining = ESCAPE_REFRACTORY_TICKS. "
    "Minimum dwell on CRUISE is deliberately zero; dwell lives on the way back"
)


def _clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


# ------------------------------------------------------------------ readout inputs


def readout_values_from_step(
    evidence: FullBrainStepEvidence,
    steering_means: tuple[float, float],
    *,
    policy: ProgramSelectionPolicy | None = None,
) -> ReadoutValues:
    """Build the three declared readout scalars from one committed controller step.

    ``ttmn_jump`` is ``evidence.jump_value`` verbatim -- the rectified TTMn term the
    escape-v2 decoder already computed and already added to vertical thrust; nothing here
    recomputes or rescales it. ``wing_common`` is ``0.5 * (left_motor + right_motor)`` over
    the decoded pool values, arithmetically the same common term the pinned projection uses
    for forward and vertical thrust. ``dna01_turn`` is the declared normalized difference of
    the DNa01 pair's committed activities.
    """

    dna01_floor = DNA01_FLOOR if policy is None else policy.dna01_floor
    turn_deadband = TURN_DEADBAND if policy is None else policy.turn_deadband
    jump = evidence.jump_value
    if jump is None:
        raise ProgramPolicyError(
            "the selection policy requires the TTMn jump readout; the wrapped controller "
            "produced no jump value for this step"
        )
    if not 0.0 <= jump <= 1.0:
        raise ProgramPolicyError(
            f"ttmn_jump {jump!r} is outside the rectified decoder's declared [0, 1] "
            "codomain; refusing to present it as a readout scalar"
        )
    values = dict(evidence.motor_frame.values)
    try:
        left_motor = values["left_motor"]
        right_motor = values["right_motor"]
    except KeyError as exc:  # pragma: no cover - the pinned projection always has both
        raise ProgramPolicyError(
            "the wrapped controller's motor frame is missing a declared wing pool"
        ) from exc
    wing_common = 0.5 * (left_motor + right_motor)
    left, right = steering_means
    if not (math.isfinite(left) and math.isfinite(right)):
        raise ProgramPolicyError("the DNa01 steering readout produced a non-finite activity")
    turn = (right - left) / (abs(left) + abs(right) + dna01_floor)
    if abs(turn) < turn_deadband:
        turn = 0.0
    return ReadoutValues(
        ttmn_jump=jump,
        dna01_turn=_clip(turn, -1.0, 1.0),
        wing_common=_clip(wing_common, -1.0, 1.0),
    )


# -------------------------------------------------------------------- policy state


@dataclass(frozen=True, slots=True)
class PolicyState:
    """The policy's whole memory: one program, two counters, one arming latch.

    Immutable; :meth:`ProgramSelectionPolicy.select` returns the successor rather than
    mutating this one, so a verifier can recompute any tick from the recorded state before
    it and the recorded readout.
    """

    program: ProgramId = ProgramId.CRUISE
    hold_remaining: int = 0
    refractory_remaining: int = 0
    armed: bool = True

    def __post_init__(self) -> None:
        if self.program not in POLICY_VOCABULARY:
            raise ProgramPolicyError(
                f"{self.program.value} is not in the declared policy vocabulary "
                f"{[member.value for member in POLICY_VOCABULARY]}"
            )
        if self.hold_remaining < 0 or self.refractory_remaining < 0:
            raise ProgramPolicyError("policy counters are non-negative by declaration")
        if (self.program is ProgramId.ESCAPE_JUMP) != (self.hold_remaining > 0):
            raise ProgramPolicyError(
                "a held ESCAPE_JUMP and a positive hold counter are the same condition"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "armed": self.armed,
            "hold_remaining": self.hold_remaining,
            "program": self.program.value,
            "refractory_remaining": self.refractory_remaining,
        }


@dataclass(frozen=True, slots=True)
class ProgramDecision:
    """One tick's selection: what to run, how to modulate it, and the resulting state.

    ``driving_readout`` names which labeled readout the current selection is a function of
    (``ttmn`` while ESCAPE_JUMP is held, ``wing`` while CRUISE runs). It is a presentation
    label for logs and HUDs, not a causal claim: all three scalars are recorded on the
    frame either way.
    """

    program: ProgramId
    modulation: ProgramModulation
    parameters: ProgramParameters
    readout: ReadoutValues
    state_before: PolicyState
    state_after: PolicyState
    game_tick: int
    driving_readout: str


# --------------------------------------------------------------- readout bindings


@dataclass(frozen=True, slots=True)
class PolicyReadoutBinding:
    """The four hashed surfaces the policy reads, chained into ``policy_hash``.

    Binding these here rather than into ``RunIdentity`` is the containment decision of
    stage 4b: the DNa01 pair is read by this policy and by nothing else, so it must not
    rotate the decoder identity that every readout calibration cross-binds against.
    """

    ttmn_pool_binding_sha256: str
    dna01_pair_binding_sha256: str
    wing_readout_sha256: str
    decoder_hash: str

    def to_identity_material(self) -> dict[str, str]:
        return {
            "decoder_hash": self.decoder_hash,
            "dna01_pair_binding_sha256": self.dna01_pair_binding_sha256,
            "ttmn_pool_binding_sha256": self.ttmn_pool_binding_sha256,
            "wing_readout_sha256": self.wing_readout_sha256,
        }


# --------------------------------------------------------------------- the policy


@dataclass(frozen=True, slots=True)
class ProgramSelectionPolicy:
    """The declared mapping from three readout scalars to one scripted body program.

    Every constant is a field with the module-level declared value as its default, so a
    test can vary exactly one and observe the identity rotate. :meth:`select` is pure: it
    reads nothing but its arguments and writes nothing at all. It is **open loop with
    respect to receipts** -- it never reads a ``program_receipt``, so a tick's decision is a
    function of that tick's readouts and the prior state alone, and a refusal shows up in
    the receipt statistics rather than changing what the policy asks for next.
    """

    readouts: PolicyReadoutBinding
    escape_on: float = ESCAPE_ON
    escape_off: float = ESCAPE_OFF
    escape_hold_ticks: int = ESCAPE_HOLD_TICKS
    escape_refractory_ticks: int = ESCAPE_REFRACTORY_TICKS
    escape_urgency_span: float = ESCAPE_URGENCY_SPAN
    dna01_floor: float = DNA01_FLOOR
    turn_deadband: float = TURN_DEADBAND
    speed_floor: float = SPEED_FLOOR
    speed_span: float = SPEED_SPAN
    escape_output_speed: float = ESCAPE_OUTPUT_SPEED
    escape_output_turn_rate: float = ESCAPE_OUTPUT_TURN_RATE
    escape_output_amplitude: float = ESCAPE_OUTPUT_AMPLITUDE
    escape_output_turn_bias: float = ESCAPE_OUTPUT_TURN_BIAS
    cruise_output_turn_rate: float = CRUISE_OUTPUT_TURN_RATE
    cruise_output_amplitude: float = CRUISE_OUTPUT_AMPLITUDE
    cruise_output_urgency: float = CRUISE_OUTPUT_URGENCY
    escape_driving_readout: str = ESCAPE_DRIVING_READOUT
    cruise_driving_readout: str = CRUISE_DRIVING_READOUT

    def __post_init__(self) -> None:
        if not 0.0 <= self.escape_off <= self.escape_on <= 1.0:
            raise ProgramPolicyError(
                "the escape thresholds must satisfy 0 <= ESCAPE_OFF <= ESCAPE_ON <= 1"
            )
        if self.escape_hold_ticks < 1 or self.escape_refractory_ticks < 0:
            raise ProgramPolicyError(
                "the escape hold must be at least one tick and the refractory non-negative"
            )
        for name in ("escape_urgency_span", "dna01_floor", "speed_span"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
                raise ProgramPolicyError(f"{name} must be positive and finite")
        if (
            isinstance(self.turn_deadband, bool)
            or not math.isfinite(self.turn_deadband)
            or not 0.0 <= self.turn_deadband <= 1.0
        ):
            raise ProgramPolicyError("the turn deadband must lie in [0, 1]")
        if not 0.0 <= self.speed_floor <= 1.0:
            raise ProgramPolicyError("the speed floor must lie in [0, 1]")
        for name in (
            "escape_output_speed",
            "escape_output_turn_rate",
            "escape_output_amplitude",
            "cruise_output_turn_rate",
            "cruise_output_amplitude",
            "cruise_output_urgency",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ProgramPolicyError(f"{name} is an attenuator and must lie in [0, 1]")
        if not -1.0 <= self.escape_output_turn_bias <= 1.0:
            raise ProgramPolicyError("escape_output_turn_bias must lie in [-1, 1]")
        if not self.escape_driving_readout or not self.cruise_driving_readout:
            raise ProgramPolicyError("each state must name the readout its selection follows")

    # ------------------------------------------------------------------ identity

    def to_identity_material(self) -> dict[str, Any]:
        return {
            "constants": {
                "DNA01_FLOOR": float(self.dna01_floor).hex(),
                "ESCAPE_HOLD_TICKS": float(self.escape_hold_ticks).hex(),
                "ESCAPE_OFF": float(self.escape_off).hex(),
                "ESCAPE_ON": float(self.escape_on).hex(),
                "ESCAPE_REFRACTORY_TICKS": float(self.escape_refractory_ticks).hex(),
                "ESCAPE_URGENCY_SPAN": float(self.escape_urgency_span).hex(),
                "SPEED_FLOOR": float(self.speed_floor).hex(),
                "SPEED_SPAN": float(self.speed_span).hex(),
                "TURN_DEADBAND": float(self.turn_deadband).hex(),
            },
            "fallback": POLICY_FALLBACK.value,
            # The per-state output table, so a change to any value a viewer actually sees
            # rotates the identity. The two entries that are functions of a readout name
            # the function rather than a value; those names are the formula keys below.
            "outputs": {
                "CRUISE": {
                    "amplitude": float(self.cruise_output_amplitude).hex(),
                    "driving_readout": self.cruise_driving_readout,
                    "speed": "speed",
                    "turn_bias": "dna01_turn",
                    "turn_rate": float(self.cruise_output_turn_rate).hex(),
                    "urgency": float(self.cruise_output_urgency).hex(),
                },
                "ESCAPE_JUMP": {
                    "amplitude": float(self.escape_output_amplitude).hex(),
                    "driving_readout": self.escape_driving_readout,
                    "speed": float(self.escape_output_speed).hex(),
                    "turn_bias": float(self.escape_output_turn_bias).hex(),
                    "turn_rate": float(self.escape_output_turn_rate).hex(),
                    "urgency": "urgency",
                },
            },
            "formulas": {
                "dna01_turn": DNA01_TURN_FORMULA,
                "speed": CRUISE_SPEED_FORMULA,
                "state_machine": STATE_MACHINE_FORMULA,
                "urgency": ESCAPE_URGENCY_FORMULA,
            },
            "readouts": self.readouts.to_identity_material(),
            "schema": PROGRAM_POLICY_SCHEMA,
            "vocabulary": [member.value for member in POLICY_VOCABULARY],
        }

    @property
    def identity_sha256(self) -> str:
        """``policy_hash``: the whole declaration, one constant away from rotating."""

        return canonical_sha256(self.to_identity_material())

    # -------------------------------------------------------------------- select

    def select(self, readout: ReadoutValues, state: PolicyState, game_tick: int) -> ProgramDecision:
        """Apply the declared state machine to one tick's readouts. Pure.

        The whole anti-flapping story lives here: a fixed hold equal to the program's own
        duration, a refractory window after it, and an arming latch that only re-arms once
        the jump term has fallen back below ``ESCAPE_OFF``. Minimum dwell on CRUISE is
        deliberately zero -- a threshold crossing selects on the tick it happens.
        """

        if state.program is ProgramId.ESCAPE_JUMP:
            hold_remaining = state.hold_remaining - 1
            if hold_remaining > 0:
                after = replace(state, hold_remaining=hold_remaining)
            else:
                after = PolicyState(
                    program=ProgramId.CRUISE,
                    hold_remaining=0,
                    refractory_remaining=self.escape_refractory_ticks,
                    armed=state.armed,
                )
        elif (
            state.armed and state.refractory_remaining == 0 and readout.ttmn_jump >= self.escape_on
        ):
            after = PolicyState(
                program=ProgramId.ESCAPE_JUMP,
                hold_remaining=self.escape_hold_ticks,
                refractory_remaining=0,
                armed=False,
            )
        else:
            after = PolicyState(
                program=ProgramId.CRUISE,
                hold_remaining=0,
                refractory_remaining=max(0, state.refractory_remaining - 1),
                armed=state.armed or readout.ttmn_jump < self.escape_off,
            )
        # The program that RUNS this tick is the one the transition produced, so a
        # threshold crossing runs ESCAPE_JUMP on the crossing tick itself and the
        # hold counts that tick as its first.
        if after.program is ProgramId.ESCAPE_JUMP:
            modulation = ProgramModulation(
                speed=self.escape_output_speed,
                turn_rate=self.escape_output_turn_rate,
                amplitude=self.escape_output_amplitude,
                urgency=self.escape_urgency(readout.ttmn_jump),
            )
            parameters = ProgramParameters(turn_bias=self.escape_output_turn_bias)
            driving_readout = self.escape_driving_readout
        else:
            modulation = ProgramModulation(
                speed=self.cruise_speed(readout.wing_common),
                turn_rate=self.cruise_output_turn_rate,
                amplitude=self.cruise_output_amplitude,
                urgency=self.cruise_output_urgency,
            )
            parameters = ProgramParameters(turn_bias=readout.dna01_turn)
            driving_readout = self.cruise_driving_readout
        return ProgramDecision(
            program=after.program,
            modulation=modulation,
            parameters=parameters,
            readout=readout,
            state_before=state,
            state_after=after,
            game_tick=game_tick,
            driving_readout=driving_readout,
        )

    # ------------------------------------------------------------------ formulas

    def cruise_speed(self, wing_common: float) -> float:
        """``CRUISE_SPEED_FORMULA``, as one function so the probe and the tests share it."""

        return self.speed_floor + (1.0 - self.speed_floor) * _clip(
            max(0.0, wing_common) / self.speed_span, 0.0, 1.0
        )

    def escape_urgency(self, ttmn_jump: float) -> float:
        """``ESCAPE_URGENCY_FORMULA``, as one function."""

        return _clip((ttmn_jump - self.escape_on) / self.escape_urgency_span, 0.0, 1.0)


def policy_for_controller(controller: FullBrainController) -> ProgramSelectionPolicy:
    """Build the declared policy over one controller's four hashed readout surfaces."""

    escape = controller.escape_binding
    steering = controller.steering_binding
    if escape is None or steering is None:
        raise ProgramPolicyError(
            "the selection policy requires both the TTMn jump binding and the DNa01 "
            "steering binding; construct the controller with escape_reflex=True and "
            "steering_readout=True"
        )
    return ProgramSelectionPolicy(
        readouts=PolicyReadoutBinding(
            ttmn_pool_binding_sha256=escape.ttmn_pool_identity_sha256,
            dna01_pair_binding_sha256=steering.identity_sha256,
            wing_readout_sha256=escape.wing.identity_sha256,
            decoder_hash=controller.identity.decoder_hash,
        )
    )


# ------------------------------------------------------------------- the controller


@dataclass(frozen=True, slots=True)
class ProgramStepEvidence:
    """Per-step evidence for one issued ``program_action``.

    ``raw_projection`` is the ``MotorVector`` the inner controller computed and this
    controller did NOT send: a program frame requests no primitive motors. It is recorded
    so the relationship between the readouts and the motor projection stays visible, and it
    is never applied to anything.
    """

    inner: FullBrainStepEvidence
    readout: ReadoutValues
    decision: ProgramDecision
    raw_projection: MotorVector
    policy_hash: str
    program_action: ProgramAction

    @property
    def command_seq(self) -> int:
        return self.inner.command_seq


class ProgramSelectingController:
    """A ``FullBrainController`` whose issued command is a body-program selection.

    The wrapped controller runs completely unchanged -- same encoder, same dynamics, same
    decoder, same identity -- and this wrapper turns its three labeled readouts into one
    ``program_action`` through the declared policy. The connectome is not deciding
    anything: the deciding function is :class:`ProgramSelectionPolicy`, it is declared, and
    its identity is on every frame.

    The envelope is copied field for field from the inner ``action``, so the causal binding
    (observation, target tick, freshness bound, neural interval, post-step state digest) is
    the same one the raw lane produces. ``RunIdentity`` is the inner controller's, byte for
    byte: ``policy_hash`` rides on the frame, never in the identity.
    """

    def __init__(
        self,
        controller: FullBrainController,
        *,
        policy: ProgramSelectionPolicy | None = None,
    ) -> None:
        escape = controller.escape_binding
        if escape is None:
            raise ProgramPolicyError(
                "the program-selecting controller requires the TTMn jump readout; wrap a "
                "controller built with escape_reflex=True"
            )
        if not isinstance(escape, EscapeMotorBindingV2):
            raise ProgramPolicyError(
                "the program-selecting controller requires the rectified (escape-v2) jump "
                "decoder: ReadoutValues.ttmn_jump is declared non-negative, and the "
                "unrectified escape-v1 decoder can produce a negative jump term"
            )
        if controller.steering_binding is None:
            raise ProgramPolicyError(
                "the program-selecting controller requires the DNa01 steering readout; "
                "wrap a controller built with steering_readout=True"
            )
        self._controller = controller
        bound = policy_for_controller(controller)
        if policy is not None and policy.readouts != bound.readouts:
            # An injected policy whose readout binding is not this controller's would put a
            # policy_hash on every frame attesting a TTMn pool, a DNa01 pair, a wing readout
            # or a decoder that is not the one producing the numbers. That is the one thing
            # the hash exists to prevent, so it is a refusal rather than a warning.
            raise ProgramPolicyError(
                "the supplied policy attests readout bindings that are not this "
                "controller's; policy_hash must name the bindings actually in use"
            )
        self._policy = policy or bound
        self._policy_hash = self._policy.identity_sha256
        self._state = PolicyState()
        self._last_evidence: ProgramStepEvidence | None = None

    # ------------------------------------------------------------------ surface

    @property
    def controller(self) -> FullBrainController:
        """The wrapped controller, unchanged."""

        return self._controller

    @property
    def policy(self) -> ProgramSelectionPolicy:
        return self._policy

    @property
    def policy_hash(self) -> str:
        """The declared policy's identity, carried on every frame this controller issues."""

        return self._policy_hash

    @property
    def policy_state(self) -> PolicyState:
        return self._state

    @property
    def steering_binding(self) -> SteeringPairBinding | None:
        return self._controller.steering_binding

    @property
    def identity(self) -> RunIdentity:
        """The wrapped controller's ``RunIdentity``, unchanged. Nothing is added."""

        return self._controller.identity

    @property
    def configuration(self) -> FullBrainConfiguration:
        return self._controller.configuration

    @property
    def run_mode(self) -> RunMode:
        return self._controller.run_mode

    @property
    def engine(self) -> ArrayNativeEngine:
        """The wrapped controller's engine, so warm/idle-rewarm paths see one object."""

        return self._controller.engine

    @property
    def last_evidence(self) -> ProgramStepEvidence | None:
        return self._last_evidence

    def reset(self) -> None:
        """Reset neural state AND the policy state together.

        The policy's hold, refractory, and arming latch describe a window of ticks in the
        body's recent past; after a neural reset there is no such past, so carrying them
        across would let a pre-reset threshold crossing keep suppressing selection in a
        run whose neural trajectory no longer contains it.
        """

        self._controller.reset()
        self._state = PolicyState()
        self._last_evidence = None

    def reset_session(self, *, session_epoch: int) -> None:
        """Rebind the inner session and discard all policy state under the service fence."""

        self._controller.reset_session(session_epoch=session_epoch)
        self._state = PolicyState()
        self._last_evidence = None

    def warm_compute_path(self, *, passes: int = 3) -> float:
        return self._controller.warm_compute_path(passes=passes)

    @property
    def session_has_stepped(self) -> bool:
        return self._controller.session_has_stepped

    # --------------------------------------------------------------------- step

    def step(self, observation: Observation, *, command_seq: int) -> ProgramAction:
        action = self._controller.step(observation, command_seq=command_seq)
        evidence = self._controller.last_evidence
        if evidence is None:  # pragma: no cover - a completed step records evidence
            raise ProgramPolicyError("the wrapped controller recorded no evidence for its step")
        readout = readout_values_from_step(
            evidence,
            self._controller.steering_readout_means(),
            policy=self._policy,
        )
        state_before = self._state
        decision = self._policy.select(readout, state_before, observation.game_tick)
        self._state = decision.state_after
        frame = self._program_action(action, decision, readout)
        self._last_evidence = ProgramStepEvidence(
            inner=evidence,
            readout=readout,
            decision=decision,
            raw_projection=action.motors,
            policy_hash=self._policy_hash,
            program_action=frame,
        )
        return frame

    def _program_action(
        self, action: Action, decision: ProgramDecision, readout: ReadoutValues
    ) -> ProgramAction:
        """Copy the inner action's envelope verbatim and carry the selection instead.

        The declared action budget rides along with the envelope rather than being recomputed
        here: ``target_tick`` and ``valid_through_tick`` are the wrapped controller's, so a
        program frame is due on exactly the tick the raw frame it replaces would have been.
        """

        if action.neutral or action.neural_step_end <= action.neural_step_start:
            raise ProgramPolicyError(
                "a program frame is never neutral: the wrapped controller produced an "
                "empty neural interval, so no selection can be issued for this tick"
            )
        return ProgramAction(
            identity=action.identity,
            command_seq=action.command_seq,
            observation_seq=action.observation_seq,
            observation_tick=action.observation_tick,
            target_tick=action.target_tick,
            valid_through_tick=action.valid_through_tick,
            neural_step_start=action.neural_step_start,
            neural_step_end=action.neural_step_end,
            state_digest=action.state_digest,
            program=decision.program,
            modulation=decision.modulation,
            parameters=decision.parameters,
            readout=readout,
            policy_hash=self._policy_hash,
        )
