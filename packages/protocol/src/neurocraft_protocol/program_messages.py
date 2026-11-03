"""Body-program command and receipt frames inside ``neurocraft.protocol.v2``.

These are siblings of :class:`~neurocraft_protocol.messages.Action` and
:class:`~neurocraft_protocol.messages.ActuationReceipt`, not variants of them: a new message
``type`` inside the same protocol version, exactly as ADR 0020 section 7 decided. A version bump
would make every committed trace unverifiable and would change the length-prefixed handshake
transcript, so the version string is frozen and the vocabulary grows sideways.

**Nothing here is anatomy.** A ``program_action`` carries a scripted body-program identifier, four
declared presentation attenuators, one declared signed parameter, and the labeled readout scalars a
hand-chosen mapping consumed to pick them. The mapping is engineering (ADR 0019); the programs are
scripted presentation (ADR 0020). The frame carries ``policy_hash`` precisely so that no surface can
present the selection as the connectome deciding anything: the deciding function is declared,
identified, and recomputable.

The frames live in this module rather than in ``messages.py`` because ``messages.py`` is already
past the repository's file-size guidance; ``messages.py`` re-exports nothing, but its dispatch
(``_MESSAGE_TYPES``, ``parse_message``, ``ProtocolMessage``) knows these types, and the package
``__init__`` exports them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, TypeAlias, cast

from .budget import ACTION_TARGET_OFFSET_TICKS
from .errors import ProtocolError
from .messages import (
    PROTOCOL_VERSION,
    Action,
    FaultCode,
    MotorVector,
    RunIdentity,
    WorldFrame,
    _enum,
    _exact_keys,
    _finite_number,
    _hash,
    _mapping,
    _protocol_header,
    _strict_int,
)


class ProgramId(StrEnum):
    """The ten declared body programs, verbatim from ``BodyProgramId.java`` (ADR 0020).

    Scripted presentation and engineering. Nothing in this vocabulary is derived from the
    connectome, from fly anatomy, or from any measurement of fly behaviour, and a running body
    program is never evidence that the connectome did anything. Wire values are the Java enum
    constant names because ``_identifier`` forbids underscores, so ``ESCAPE_JUMP`` can only travel
    as an enum. ``tests/protocol/test_program_vocabulary.py`` pins set equality with the Java
    source.
    """

    HOVER = "HOVER"
    CRUISE = "CRUISE"
    BANK = "BANK"
    LAND = "LAND"
    WALK = "WALK"
    GROOM = "GROOM"
    FEED = "FEED"
    ESCAPE_JUMP = "ESCAPE_JUMP"
    FLEE = "FLEE"
    APPROACH = "APPROACH"


class ProgramValidation(StrEnum):
    """How the authority handled one issued ``program_action``.

    ``PROGRAM_APPLIED`` the declared program ran; ``PROGRAM_REFUSED`` the authority fell back to
    the declared HOVER fallback and says why; ``REJECTED`` the envelope failed validation and the
    authority applied neutral motors.
    """

    PROGRAM_APPLIED = "PROGRAM_APPLIED"
    PROGRAM_REFUSED = "PROGRAM_REFUSED"
    REJECTED = "REJECTED"


class ProgramRefusalReason(StrEnum):
    """The three declared reasons a program tick can fall back to HOVER (ADR 0020 section 2)."""

    UNREGISTERED = "unregistered"
    PRECONDITION = "precondition"
    OPERATOR = "operator"


@dataclass(frozen=True, slots=True)
class ProgramModulation:
    """Four declared presentation attenuators, each in ``[0, 1]``.

    Attenuators only: by declaration (ADR 0020 section 4) a modulation can never widen a program's
    own declared maxima. The wire refuses out-of-range values so the Java-side clamp never engages
    on the wire path.
    """

    speed: float
    turn_rate: float
    amplitude: float
    urgency: float

    FIELDS: ClassVar[set[str]] = {"speed", "turn_rate", "amplitude", "urgency"}

    def __post_init__(self) -> None:
        for name in self.FIELDS:
            _finite_number(getattr(self, name), f"modulation.{name}", minimum=0.0, maximum=1.0)

    @classmethod
    def from_dict(cls, raw: Any, *, name: str = "modulation") -> ProgramModulation:
        value = _mapping(raw, name)
        _exact_keys(value, required=cls.FIELDS, name=name)
        return cls(
            speed=_finite_number(value["speed"], f"{name}.speed", minimum=0.0, maximum=1.0),
            turn_rate=_finite_number(
                value["turn_rate"], f"{name}.turn_rate", minimum=0.0, maximum=1.0
            ),
            amplitude=_finite_number(
                value["amplitude"], f"{name}.amplitude", minimum=0.0, maximum=1.0
            ),
            urgency=_finite_number(value["urgency"], f"{name}.urgency", minimum=0.0, maximum=1.0),
        )

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in sorted(self.FIELDS)}


@dataclass(frozen=True, slots=True)
class ProgramParameters:
    """The signed program inputs of v1: one declared turn bias in ``[-1, 1]``.

    Signed values are a different kind of object from an attenuator, so they travel in their own
    exact-key object rather than as a fifth ``modulation`` field. Adding a parameter later is a new
    frame type, not an optional key: every object on this wire is closed.
    """

    turn_bias: float

    FIELDS: ClassVar[set[str]] = {"turn_bias"}

    def __post_init__(self) -> None:
        _finite_number(self.turn_bias, "parameters.turn_bias", minimum=-1.0, maximum=1.0)

    @classmethod
    def from_dict(cls, raw: Any, *, name: str = "parameters") -> ProgramParameters:
        value = _mapping(raw, name)
        _exact_keys(value, required=cls.FIELDS, name=name)
        return cls(
            turn_bias=_finite_number(
                value["turn_bias"], f"{name}.turn_bias", minimum=-1.0, maximum=1.0
            )
        )

    def to_dict(self) -> dict[str, float]:
        return {"turn_bias": self.turn_bias}


@dataclass(frozen=True, slots=True)
class ReadoutValues:
    """The exact labeled readout scalars the selection policy consumed on one tick.

    Carried on the frame so a HUD shows the number that crossed the wire with the command rather
    than one it looked up elsewhere, and so one trace record binds inputs to output. These are
    readout values, not decisions: ``ttmn_jump`` is a rectified motor-stage term, ``dna01_turn`` a
    normalized descending left/right difference, ``wing_common`` the projection's common wing term.
    """

    ttmn_jump: float
    dna01_turn: float
    wing_common: float

    FIELDS: ClassVar[set[str]] = {"ttmn_jump", "dna01_turn", "wing_common"}

    def __post_init__(self) -> None:
        _finite_number(self.ttmn_jump, "readout.ttmn_jump", minimum=0.0, maximum=1.0)
        _finite_number(self.dna01_turn, "readout.dna01_turn", minimum=-1.0, maximum=1.0)
        _finite_number(self.wing_common, "readout.wing_common", minimum=-1.0, maximum=1.0)

    @classmethod
    def from_dict(cls, raw: Any, *, name: str = "readout") -> ReadoutValues:
        value = _mapping(raw, name)
        _exact_keys(value, required=cls.FIELDS, name=name)
        return cls(
            ttmn_jump=_finite_number(
                value["ttmn_jump"], f"{name}.ttmn_jump", minimum=0.0, maximum=1.0
            ),
            dna01_turn=_finite_number(
                value["dna01_turn"], f"{name}.dna01_turn", minimum=-1.0, maximum=1.0
            ),
            wing_common=_finite_number(
                value["wing_common"], f"{name}.wing_common", minimum=-1.0, maximum=1.0
            ),
        )

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in sorted(self.FIELDS)}


@dataclass(frozen=True, slots=True)
class ProgramAction:
    """One tick's body-program selection, in the same causal envelope as an ``action``.

    The envelope is field-for-field identical to :class:`Action` — identity, sequences, the
    observation binding, the two-tick freshness bound, the neural interval, the post-step state
    digest — so the authority's envelope-only validator applies unchanged and the selection stays
    inside the causal envelope the connectome produced.

    Two differences from :class:`Action`. There are no ``motors`` and no ``neutral``: a program
    frame requests no primitive motors, and the paused, replaying, and faulted service keeps
    emitting neutral ``action`` frames. And the neural interval is **strictly** non-empty: a
    program frame is never neutral, so the connectome stepped.
    """

    identity: RunIdentity
    command_seq: int
    observation_seq: int
    observation_tick: int
    target_tick: int
    valid_through_tick: int
    neural_step_start: int
    neural_step_end: int
    state_digest: str
    program: ProgramId
    modulation: ProgramModulation
    parameters: ProgramParameters
    readout: ReadoutValues
    policy_hash: str

    TYPE: ClassVar[str] = "program_action"
    FIELDS: ClassVar[set[str]] = {
        "type",
        "protocol",
        "identity",
        "command_seq",
        "observation_seq",
        "observation_tick",
        "target_tick",
        "valid_through_tick",
        "neural_step_start",
        "neural_step_end",
        "state_digest",
        "program",
        "modulation",
        "parameters",
        "readout",
        "policy_hash",
    }

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RunIdentity):
            raise ProtocolError("program_action.identity must be a RunIdentity")
        for name in (
            "command_seq",
            "observation_seq",
            "observation_tick",
            "target_tick",
            "valid_through_tick",
            "neural_step_start",
            "neural_step_end",
        ):
            _strict_int(getattr(self, name), f"program_action.{name}")
        if self.target_tick < self.observation_tick:
            raise ProtocolError("program_action.target_tick precedes its observation")
        if self.valid_through_tick < self.target_tick:
            raise ProtocolError("program_action.valid_through_tick precedes target_tick")
        if self.valid_through_tick > self.observation_tick + ACTION_TARGET_OFFSET_TICKS:
            raise ProtocolError("program_action validity exceeds the declared action budget")
        if self.neural_step_end <= self.neural_step_start:
            raise ProtocolError("program_action neural interval must be non-empty")
        _hash(self.state_digest, "program_action.state_digest")
        if not isinstance(self.program, ProgramId):
            raise ProtocolError("program_action.program must be a ProgramId")
        if not isinstance(self.modulation, ProgramModulation):
            raise ProtocolError("program_action.modulation must be a ProgramModulation")
        if not isinstance(self.parameters, ProgramParameters):
            raise ProtocolError("program_action.parameters must be a ProgramParameters")
        if not isinstance(self.readout, ReadoutValues):
            raise ProtocolError("program_action.readout must be a ReadoutValues")
        _hash(self.policy_hash, "program_action.policy_hash")

    @classmethod
    def from_dict(cls, raw: Any) -> ProgramAction:
        value = _mapping(raw, "program_action")
        _exact_keys(value, required=cls.FIELDS, name="program_action")
        _protocol_header(value, cls.TYPE)
        return cls(
            identity=RunIdentity.from_dict(value["identity"]),
            command_seq=_strict_int(value["command_seq"], "program_action.command_seq"),
            observation_seq=_strict_int(value["observation_seq"], "program_action.observation_seq"),
            observation_tick=_strict_int(
                value["observation_tick"], "program_action.observation_tick"
            ),
            target_tick=_strict_int(value["target_tick"], "program_action.target_tick"),
            valid_through_tick=_strict_int(
                value["valid_through_tick"], "program_action.valid_through_tick"
            ),
            neural_step_start=_strict_int(
                value["neural_step_start"], "program_action.neural_step_start"
            ),
            neural_step_end=_strict_int(value["neural_step_end"], "program_action.neural_step_end"),
            state_digest=_hash(value["state_digest"], "program_action.state_digest"),
            program=cast(ProgramId, _enum(value["program"], ProgramId, "program_action.program")),
            modulation=ProgramModulation.from_dict(
                value["modulation"], name="program_action.modulation"
            ),
            parameters=ProgramParameters.from_dict(
                value["parameters"], name="program_action.parameters"
            ),
            readout=ReadoutValues.from_dict(value["readout"], name="program_action.readout"),
            policy_hash=_hash(value["policy_hash"], "program_action.policy_hash"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.TYPE,
            "protocol": PROTOCOL_VERSION,
            "identity": self.identity.to_dict(),
            "command_seq": self.command_seq,
            "observation_seq": self.observation_seq,
            "observation_tick": self.observation_tick,
            "target_tick": self.target_tick,
            "valid_through_tick": self.valid_through_tick,
            "neural_step_start": self.neural_step_start,
            "neural_step_end": self.neural_step_end,
            "state_digest": self.state_digest,
            "program": self.program.value,
            "modulation": self.modulation.to_dict(),
            "parameters": self.parameters.to_dict(),
            "readout": self.readout.to_dict(),
            "policy_hash": self.policy_hash,
        }


Command: TypeAlias = Action | ProgramAction
"""The two driving command kinds a controller may issue. One kind drives one session."""


@dataclass(frozen=True, slots=True)
class ProgramReceipt:
    """Authoritative Minecraft report of how one issued ``program_action`` was handled.

    There is no honest ``actuation_receipt`` for a program tick: the requested-versus-applied
    difference an ``actuation_receipt`` reports is a motor-vector difference, and a program frame
    requests no motors (ADR 0020 section 3). This receipt reports the program difference instead —
    what was requested, what actually ran, and the primitive motors the running program emitted on
    that tick.

    ``executed_*`` are null exactly when ``validation`` is ``REJECTED``: a rejected frame ran no
    program at all. Successful and refused receipts are confined to the frame's two-tick window;
    a rejected receipt keeps the actual validation tick, which can lie outside it.

    **The three refusal shapes.** ``PROGRAM_REFUSED`` covers three different things happening on
    the body, and the receipt must be able to say which truthfully:

    * ``unregistered`` — the frame named a declared program that no registry entry provides;
    * ``precondition`` — the program declined to enter;
    * ``operator`` — an operator hold owned the tick, so the frame never reached the body.

    The first two ran the declared HOVER fallback, so ``executed_program`` is ``HOVER``. The
    third did not: the operator's own program ran, and ``executed_program`` is that program,
    which may be any of the ten. Forcing HOVER there would make the receipt claim a program ran
    that did not, which is the one thing a receipt may never do.

    **Refusal echo convention (what the goldens show).** For ``unregistered`` and
    ``precondition`` the fallback runs under the *requested* modulation and parameters, because
    the runtime sets them before the registry and ``canEnter`` checks, so ``executed_modulation``
    and ``executed_parameters`` echo the request. For ``operator`` they are whatever the operator
    lane's program is actually running under, and are generally unrelated to the request.
    """

    identity: RunIdentity
    command_seq: int
    observation_seq: int
    target_tick: int
    applied_tick: int
    world_before: WorldFrame
    requested_program: ProgramId
    requested_modulation: ProgramModulation
    requested_parameters: ProgramParameters
    executed_program: ProgramId | None
    executed_modulation: ProgramModulation | None
    executed_parameters: ProgramParameters | None
    emitted_motors: MotorVector
    validation: ProgramValidation
    reason: ProgramRefusalReason | None = None
    fault_code: FaultCode | None = None

    TYPE: ClassVar[str] = "program_receipt"
    FIELDS: ClassVar[set[str]] = {
        "type",
        "protocol",
        "identity",
        "command_seq",
        "observation_seq",
        "target_tick",
        "applied_tick",
        "world_before",
        "requested_program",
        "requested_modulation",
        "requested_parameters",
        "executed_program",
        "executed_modulation",
        "executed_parameters",
        "emitted_motors",
        "validation",
        "reason",
        "fault_code",
    }

    def __post_init__(self) -> None:
        self._validate_shape()
        rejected = self.validation is ProgramValidation.REJECTED
        executed = (self.executed_program, self.executed_modulation, self.executed_parameters)
        if rejected and any(item is not None for item in executed):
            raise ProtocolError("REJECTED program_receipt executed nothing and must be null")
        if not rejected and any(item is None for item in executed):
            raise ProtocolError("applied or refused program_receipt requires executed values")
        if not rejected and (
            self.applied_tick < self.target_tick or self.applied_tick > self.target_tick + 2
        ):
            raise ProtocolError(
                "applied or refused program_receipt.applied_tick must be within two target ticks"
            )
        if self.validation is ProgramValidation.PROGRAM_APPLIED:
            self._validate_applied()
        elif self.validation is ProgramValidation.PROGRAM_REFUSED:
            self._validate_refused()
        else:
            self._validate_rejected()

    def _validate_shape(self) -> None:
        if not isinstance(self.identity, RunIdentity):
            raise ProtocolError("program_receipt.identity must be a RunIdentity")
        for name in ("command_seq", "observation_seq", "target_tick", "applied_tick"):
            _strict_int(getattr(self, name), f"program_receipt.{name}")
        if not isinstance(self.world_before, WorldFrame):
            raise ProtocolError("program_receipt.world_before must be a WorldFrame")
        if not isinstance(self.requested_program, ProgramId):
            raise ProtocolError("program_receipt.requested_program must be a ProgramId")
        if not isinstance(self.requested_modulation, ProgramModulation):
            raise ProtocolError("program_receipt.requested_modulation must be a ProgramModulation")
        if not isinstance(self.requested_parameters, ProgramParameters):
            raise ProtocolError("program_receipt.requested_parameters must be a ProgramParameters")
        if self.executed_program is not None and not isinstance(self.executed_program, ProgramId):
            raise ProtocolError("program_receipt.executed_program must be a ProgramId or null")
        if self.executed_modulation is not None and not isinstance(
            self.executed_modulation, ProgramModulation
        ):
            raise ProtocolError(
                "program_receipt.executed_modulation must be a ProgramModulation or null"
            )
        if self.executed_parameters is not None and not isinstance(
            self.executed_parameters, ProgramParameters
        ):
            raise ProtocolError(
                "program_receipt.executed_parameters must be a ProgramParameters or null"
            )
        if not isinstance(self.emitted_motors, MotorVector):
            raise ProtocolError("program_receipt.emitted_motors must be a MotorVector")
        if not isinstance(self.validation, ProgramValidation):
            raise ProtocolError("program_receipt.validation must be ProgramValidation")
        if self.reason is not None and not isinstance(self.reason, ProgramRefusalReason):
            raise ProtocolError("program_receipt.reason must be a ProgramRefusalReason or null")
        if self.fault_code is not None and not isinstance(self.fault_code, FaultCode):
            raise ProtocolError("program_receipt.fault_code must be FaultCode or null")

    def _validate_applied(self) -> None:
        if self.reason is not None or self.fault_code is not None:
            raise ProtocolError("PROGRAM_APPLIED program_receipt cannot carry a reason or a fault")
        if (
            self.executed_program is not self.requested_program
            or self.executed_modulation != self.requested_modulation
            or self.executed_parameters != self.requested_parameters
        ):
            raise ProtocolError("PROGRAM_APPLIED program_receipt must execute what was requested")

    def _validate_refused(self) -> None:
        """Refusals that ran the fallback must say HOVER; an operator hold must not.

        ``unregistered`` and ``precondition`` both mean the declared HOVER fallback ran, so the
        receipt is pinned to it. ``operator`` means the frame never reached the body at all
        because an operator hold owned the tick: the program that ran is the operator's, it can
        be any of the ten, and pinning HOVER here would make the receipt assert a program ran
        that did not.
        """

        if self.reason is None or self.fault_code is not None:
            raise ProtocolError("PROGRAM_REFUSED program_receipt requires a reason and no fault")
        if (
            self.reason is not ProgramRefusalReason.OPERATOR
            and self.executed_program is not ProgramId.HOVER
        ):
            raise ProtocolError(
                "unregistered or precondition program_receipt must fall back to HOVER"
            )

    def _validate_rejected(self) -> None:
        if self.reason is not None or self.fault_code is None:
            raise ProtocolError("REJECTED program_receipt requires a fault and no reason")
        if not self.emitted_motors.is_neutral:
            raise ProtocolError("REJECTED program_receipt requires neutral emitted motors")

    @classmethod
    def from_dict(cls, raw: Any) -> ProgramReceipt:
        value = _mapping(raw, "program_receipt")
        _exact_keys(value, required=cls.FIELDS, name="program_receipt")
        _protocol_header(value, cls.TYPE)
        executed_program = value["executed_program"]
        executed_modulation = value["executed_modulation"]
        executed_parameters = value["executed_parameters"]
        reason = value["reason"]
        fault_code = value["fault_code"]
        return cls(
            identity=RunIdentity.from_dict(value["identity"]),
            command_seq=_strict_int(value["command_seq"], "program_receipt.command_seq"),
            observation_seq=_strict_int(
                value["observation_seq"], "program_receipt.observation_seq"
            ),
            target_tick=_strict_int(value["target_tick"], "program_receipt.target_tick"),
            applied_tick=_strict_int(value["applied_tick"], "program_receipt.applied_tick"),
            world_before=WorldFrame.from_dict(
                value["world_before"], name="program_receipt.world_before"
            ),
            requested_program=cast(
                ProgramId,
                _enum(value["requested_program"], ProgramId, "program_receipt.requested_program"),
            ),
            requested_modulation=ProgramModulation.from_dict(
                value["requested_modulation"], name="program_receipt.requested_modulation"
            ),
            requested_parameters=ProgramParameters.from_dict(
                value["requested_parameters"], name="program_receipt.requested_parameters"
            ),
            executed_program=(
                None
                if executed_program is None
                else cast(
                    ProgramId,
                    _enum(executed_program, ProgramId, "program_receipt.executed_program"),
                )
            ),
            executed_modulation=(
                None
                if executed_modulation is None
                else ProgramModulation.from_dict(
                    executed_modulation, name="program_receipt.executed_modulation"
                )
            ),
            executed_parameters=(
                None
                if executed_parameters is None
                else ProgramParameters.from_dict(
                    executed_parameters, name="program_receipt.executed_parameters"
                )
            ),
            emitted_motors=MotorVector.from_dict(
                value["emitted_motors"], name="program_receipt.emitted_motors"
            ),
            validation=cast(
                ProgramValidation,
                _enum(value["validation"], ProgramValidation, "program_receipt.validation"),
            ),
            reason=(
                None
                if reason is None
                else cast(
                    ProgramRefusalReason,
                    _enum(reason, ProgramRefusalReason, "program_receipt.reason"),
                )
            ),
            fault_code=(
                None
                if fault_code is None
                else cast(FaultCode, _enum(fault_code, FaultCode, "program_receipt.fault_code"))
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.TYPE,
            "protocol": PROTOCOL_VERSION,
            "identity": self.identity.to_dict(),
            "command_seq": self.command_seq,
            "observation_seq": self.observation_seq,
            "target_tick": self.target_tick,
            "applied_tick": self.applied_tick,
            "world_before": self.world_before.to_dict(),
            "requested_program": self.requested_program.value,
            "requested_modulation": self.requested_modulation.to_dict(),
            "requested_parameters": self.requested_parameters.to_dict(),
            "executed_program": (
                None if self.executed_program is None else self.executed_program.value
            ),
            "executed_modulation": (
                None if self.executed_modulation is None else self.executed_modulation.to_dict()
            ),
            "executed_parameters": (
                None if self.executed_parameters is None else self.executed_parameters.to_dict()
            ),
            "emitted_motors": self.emitted_motors.to_dict(),
            "validation": self.validation.value,
            "reason": None if self.reason is None else self.reason.value,
            "fault_code": None if self.fault_code is None else self.fault_code.value,
        }
