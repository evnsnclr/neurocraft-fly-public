"""Stateful session, sequence, tick, freshness, and identity validation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TypeAlias

from .budget import ACTION_TARGET_OFFSET_TICKS
from .errors import (
    FutureMessageError,
    IdentityMismatchError,
    ProtocolError,
    ReceiptMismatchError,
    SequenceError,
    StaleMessageError,
)
from .messages import (
    Action,
    ActuationReceipt,
    ActuationValidation,
    ControlAck,
    ControlCommand,
    Hello,
    Observation,
    PeerRole,
    RunIdentity,
)
from .program_messages import Command, ProgramAction, ProgramReceipt, ProgramValidation

Receipt: TypeAlias = ActuationReceipt | ProgramReceipt
"""The two authoritative receipt kinds; each binds only a pending command of its own kind."""

OBSERVATION_HISTORY_CAPACITY = 8
"""Accepted bindings, matching Java SentObservationWindow's existing 2 + 5 + 1 entries.

This is history for verifying completed work, not an action-validity or starvation budget.
Four through seven newer observations must not make Python discard a binding the authority
can still verify and retire fail-neutral. Eight newer observations still evict it.
"""


def _receipt_label(receipt: Receipt) -> str:
    return "program receipt" if isinstance(receipt, ProgramReceipt) else "actuation receipt"


def _receipt_binds_command(receipt: Receipt, command: Command) -> bool:
    """Whether a receipt binds this pending command, kind included.

    A receipt binds only a pending command of its own kind: a ``program_receipt`` echoes the
    requested program, modulation, and parameters of a pending ``program_action``, and an
    ``actuation_receipt`` echoes the requested motors of a pending ``action``. A receipt of the
    wrong kind for the sequence it names is a contract violation, not a near miss, so it fails
    closed exactly as a mismatched echo does.
    """

    if isinstance(receipt, ProgramReceipt):
        return (
            isinstance(command, ProgramAction)
            and receipt.requested_program is command.program
            and receipt.requested_modulation == command.modulation
            and receipt.requested_parameters == command.parameters
        )
    return isinstance(command, Action) and receipt.requested_motors == command.motors


def _receipt_reports_application(receipt: Receipt) -> bool:
    """Whether a receipt claims the authority actually ran the command it names.

    These are the variants the protocol confines to the command's window; the faulted and
    rejected variants keep their real validation tick, which can lie outside it. A refused
    program tick ran the declared HOVER fallback on the target tick, so it is confined too.
    """

    if isinstance(receipt, ProgramReceipt):
        return receipt.validation in {
            ProgramValidation.PROGRAM_APPLIED,
            ProgramValidation.PROGRAM_REFUSED,
        }
    return receipt.validation in {ActuationValidation.APPLIED, ActuationValidation.CLAMPED}


@dataclass(slots=True)
class SessionValidator:
    """Mutable validator owned by one authenticated connection session.

    It intentionally doesn't use wall-clock time. Game ticks and protocol sequences
    are authoritative; the caller supplies the current server tick when validating
    frames that can become stale or future-dated.
    """

    expected_identity: RunIdentity
    max_action_age_ticks: int = 2
    max_command_window: int = ACTION_TARGET_OFFSET_TICKS + 1
    """Commands that may be validated but not yet applied.

    Derived, not chosen: at observation tick ``T`` the authority can still apply the command
    targeting ``T`` (computed from ``T - 2``) and the one targeting ``T + 1`` (from ``T - 1``),
    and the command computed from ``T`` itself is about to be reserved — three, which is
    ``ACTION_TARGET_OFFSET_TICKS + 1``. It was two under the one-tick budget for the same
    reason. Sizing it any smaller would re-open ADR 0018's defect B on the very first
    receipt-less command: retirement releases a slot only once the world is past a command's
    ``target_tick``, so a run of commands that produce no receipt (a program-owned tick, an
    ADR 0011 lateness retirement, an ADR 0006 reset-boundary retirement) would wedge the
    window and fail the session closed.
    """

    max_control_window: int = 2
    """Control commands awaiting acknowledgement; deliberately not the command window.

    A control acknowledgement is stamped with an observation tick (D-079), not with an action
    target, so the action budget does not reach it and this bound stays where ADR 0006 put it.
    """
    last_observation_seq: int = -1
    last_observation_tick: int = -1
    last_command_seq: int = -1
    last_neural_step_end: int = -1
    last_control_seq: int = -1
    last_control_tick: int = -1
    retired_action_count: int = 0
    # The observation ledger an arriving command's binding is looked up in, and the mechanism
    # that decides what a late step costs. Eviction here is NOT a fail-neutral retirement: a
    # command whose observation has left the ledger raises StaleMessageError("action references
    # an unavailable observation") from accept_action, and the SERVICE FAILS CLOSED on STALE.
    #
    # The service calls accept_action with current_tick = observation.game_tick -- the tick of
    # the observation the command answers -- so its age is always 0 there and neither the
    # freshness rule nor the target rules can fire on a late step. The ledger is the only thing
    # that does, which makes its size the whole boundary between D-080's "send the late frame and
    # let the authority retire it fail-neutral" and a dead session.
    #
    # Match the authority's eight-entry sent-observation history. A four-entry service ledger
    # killed live command 3217 (observation tick 3521) while Java was still inside its existing
    # fail-neutral hold. Retaining the same bounded history permits verification, not late
    # application: Java still checks the exact target and age and never extends its starvation
    # hold for a late frame. Eight outrunning observations still lose the binding and fail closed.
    _observations: deque[tuple[int, int]] = field(
        default_factory=lambda: deque(maxlen=OBSERVATION_HISTORY_CAPACITY)
    )
    _pending_actions: deque[Command] = field(default_factory=deque)
    _pending_controls: deque[ControlCommand] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if not isinstance(self.expected_identity, RunIdentity):
            raise ProtocolError("expected_identity must be a RunIdentity")
        if self.max_action_age_ticks < 0:
            raise ProtocolError("max_action_age_ticks must be non-negative")
        if self.max_command_window < 1 or self.max_command_window > ACTION_TARGET_OFFSET_TICKS + 1:
            raise ProtocolError(
                "max_command_window must be between one and the action budget plus one"
            )
        if self.max_control_window < 1 or self.max_control_window > 2:
            raise ProtocolError("max_control_window must be one or two")

    def validate_identity(self, actual: RunIdentity) -> None:
        if actual != self.expected_identity:
            raise IdentityMismatchError("run, session, entity, or artifact identity mismatch")

    def accept_hello(self, hello: Hello, *, expected_role: PeerRole) -> None:
        self.validate_identity(hello.identity)
        if hello.role is not expected_role:
            raise IdentityMismatchError(f"peer role must be {expected_role.value}")

    def accept_observation(
        self, observation: Observation, *, current_tick: int | None = None
    ) -> None:
        self.validate_identity(observation.identity)
        expected_seq = self.last_observation_seq + 1
        if observation.observation_seq != expected_seq:
            raise SequenceError(
                f"observation sequence must be {expected_seq}, got {observation.observation_seq}"
            )
        if observation.game_tick <= self.last_observation_tick:
            raise SequenceError("observation game tick must increase")
        if current_tick is not None:
            if observation.game_tick > current_tick:
                raise FutureMessageError("observation is from a future game tick")
            if current_tick - observation.game_tick > self.max_action_age_ticks:
                raise StaleMessageError("observation exceeds the tick freshness bound")
        self.last_observation_seq = observation.observation_seq
        self.last_observation_tick = observation.game_tick
        self._observations.append((observation.observation_seq, observation.game_tick))
        self.retire_expired_actions(current_tick=observation.game_tick)

    def retire_expired_actions(self, *, current_tick: int) -> tuple[Command, ...]:
        """Release window slots the authoritative side can no longer legally use.

        The authority applies a command on exactly one tick — its ``target_tick``,
        which ``CommandValidator`` enforces with ``targetTick() == currentGameTick``
        — and it sends a receipt only for commands it actually applied. Without this,
        every refusal — a stale command, a frame retired at an accepted RESET
        boundary, or an ADR 0011 lateness retirement — costs a window slot
        permanently: ``max_command_window`` refusals fill it, ``accept_action``
        raises ``SequenceError("validated command window is full")``, the service
        fails closed, and the authority sits at STALE_COMMAND forever. That is a
        mutual deadlock in which both sides are individually behaving correctly.

        **The boundary is ``target_tick``, not ``valid_through_tick`` (ADR 0018).**
        The protocol guarantees ``target_tick <= valid_through_tick``
        (``Action.__post_init__``), so this retires at the same tick or one tick
        earlier than the previous ``valid_through_tick`` rule — strictly tighter,
        never looser, and therefore incapable of weakening anything. It is also the
        correct question to ask: once the world has passed ``target_tick`` the
        authority can no longer apply the command at all, so the extra
        ``valid_through_tick`` slack was dead window occupancy. Holding it was what
        let two un-receipted commands wedge the window and kill a session that had
        already cleared its cold start.

        Retiring here does not weaken the freshness rule; it applies the same rule to
        this side's own bookkeeping. ``accept_receipt`` independently refuses a
        receipt claiming application after expiry.

        Correctness rests on the transport being ordered, which the protocol's single
        WebSocket connection guarantees: the authority sends a receipt at the tick it
        applies, which is exactly ``target_tick``, and it sends that tick's
        observation before it polls and applies, so any legitimate receipt has
        already been parsed before an observation carrying a later tick arrives.
        Retiring on observation therefore cannot race a valid receipt for
        the SAME action, and a receipt arriving after its action's retirement is a
        genuine violation that still fails closed rather than being absorbed here.
        The authority's ADR 0011 lateness retirement adds one more consumption
        path that produces no receipt at all; ``accept_receipt`` handles it by
        supersession (a later receipt proves the earlier pendings were consumed
        unapplied) rather than through this observation-driven path.

        Retirements are counted rather than silently discarded: a nonzero count means
        the authority is missing its deadline, which is a real signal about the run.
        """

        retired: list[Command] = []
        while self._pending_actions and self._pending_actions[0].target_tick < current_tick:
            retired.append(self._pending_actions.popleft())
        self.retired_action_count += len(retired)
        return tuple(retired)

    def accept_action(self, action: Command, *, current_tick: int) -> None:
        """Validate and reserve one command in the ``max_command_window`` window.

        Every rule here is envelope-only — sequence continuity, the observation binding, age and
        freshness, target ordering, neural-interval continuity, window size — and both command
        kinds carry the identical envelope, so a ``program_action`` is admitted under exactly the
        rules an ``action`` is, and ADR 0018's ``target_tick`` retirement releases its slot the
        same way.

        **The action budget is checked here, not only where frames are produced.** A command
        targets exactly ``observation_tick + ACTION_TARGET_OFFSET_TICKS``; the frame model
        permits anything from the observation tick to that ceiling so that frames recorded under
        the one-tick budget still parse, so this is the acceptance rule that says the producing
        controller obeyed the declared budget. With it, and with ``age >= 0``, the future bound
        below is the same relative statement it always was: no frame may claim a target more
        than one whole budget ahead of the authoritative clock.

        **The observation ledger, not the freshness rules, is what a late step runs into.** The
        service calls this with ``current_tick = observation.game_tick``, so ``age`` is always 0
        here and neither the freshness bound nor the target bounds can fire on a step the world
        outran — the binding lookup above is the rule that does, and it fails the session
        **closed** on ``StaleMessageError``, not fail-neutral. The ledger holds eight entries,
        matching the authority's existing history. A retained late frame can be sent for the
        authority to retire fail-neutral; eight outrunning observations still evict its binding.
        Supplying an actual current tick instead of the source tick still enforces the unchanged
        freshness bound below. Retention grants no extra time to apply a command.
        """

        self.validate_identity(action.identity)
        expected_seq = self.last_command_seq + 1
        if action.command_seq != expected_seq:
            raise SequenceError(
                f"command sequence must be {expected_seq}, got {action.command_seq}"
            )
        observed = dict(self._observations)
        if action.observation_seq not in observed:
            if action.observation_seq > self.last_observation_seq:
                raise FutureMessageError("action references a future observation")
            raise StaleMessageError("action references an unavailable observation")
        if observed[action.observation_seq] != action.observation_tick:
            raise IdentityMismatchError(
                "action observation tick does not match the accepted observation"
            )
        # The target bound and the age bound are the same clock inconsistency and raise the
        # same error; the target one is evaluated first so it stays reachable. An action may
        # declare validity at most ACTION_TARGET_OFFSET_TICKS past its own observation, so
        # once the age is known non-negative its target can no longer exceed the bound.
        if action.target_tick > current_tick + ACTION_TARGET_OFFSET_TICKS:
            raise FutureMessageError("action target is beyond the declared action budget")
        age = current_tick - action.observation_tick
        if age < 0:
            raise FutureMessageError("action observation is from a future game tick")
        if age > self.max_action_age_ticks or current_tick > action.valid_through_tick:
            raise StaleMessageError("action exceeds the two-tick freshness bound")
        if action.target_tick != action.observation_tick + ACTION_TARGET_OFFSET_TICKS:
            raise SequenceError(
                "action target tick does not match the declared action budget "
                f"(observation {action.observation_tick} + {ACTION_TARGET_OFFSET_TICKS})"
            )
        if action.target_tick < current_tick:
            raise StaleMessageError("action target tick has passed")
        if self.last_neural_step_end >= 0 and action.neural_step_start < self.last_neural_step_end:
            raise SequenceError("neural intervals overlap or move backward")
        if len(self._pending_actions) >= self.max_command_window:
            raise SequenceError("validated command window is full")
        if self._pending_actions and action.target_tick <= self._pending_actions[-1].target_tick:
            raise SequenceError("target ticks must increase within the command window")
        self.last_command_seq = action.command_seq
        self.last_neural_step_end = action.neural_step_end
        self._pending_actions.append(action)

    def accept_receipt(self, receipt: Receipt) -> Command:
        """Bind an authoritative application receipt to a pending action.

        The authority consumes commands in strict sequence (its validator
        refuses any gap), and its ADR 0011 lateness retirement consumes a
        verified-late frame without ever applying it — producing no receipt.
        A receipt for a later sequence therefore PROVES every earlier pending
        action was consumed unapplied, so those heads retire fail-neutral
        here (counted) instead of failing the session closed on the proof of
        their own supersession. A receipt whose sequence matches no pending
        action at all remains a genuine violation and fails closed with the
        window untouched. The superseded action's disposal is trusted, not
        verified: it rests entirely on the authority's strict gapless
        sequence-continuity invariant, and the window bounds it to at most
        ``max_command_window - 1`` unverified retirements per receipt.

        A receipt binds only a pending command of its own kind: a ``program_receipt`` cannot
        retire a pending ``action`` and an ``actuation_receipt`` cannot retire a pending
        ``program_action``. A kind mismatch is a ``ReceiptMismatchError`` with the window
        untouched, as any other unbound receipt is.
        """

        label = _receipt_label(receipt)
        self.validate_identity(receipt.identity)
        if not self._pending_actions:
            raise ReceiptMismatchError(f"{label} has no pending action")
        if all(pending.command_seq != receipt.command_seq for pending in self._pending_actions):
            raise ReceiptMismatchError(f"{label} does not bind the pending action")
        while self._pending_actions[0].command_seq < receipt.command_seq:
            self._pending_actions.popleft()
            self.retired_action_count += 1
        expected = self._pending_actions[0]
        if (
            receipt.command_seq != expected.command_seq
            or receipt.observation_seq != expected.observation_seq
            or receipt.target_tick != expected.target_tick
            or not _receipt_binds_command(receipt, expected)
        ):
            raise ReceiptMismatchError(f"{label} does not bind the pending action")
        if (
            _receipt_reports_application(receipt)
            and receipt.applied_tick > expected.valid_through_tick
        ):
            raise StaleMessageError(f"{label} reports application after command expiry")
        self._pending_actions.popleft()
        return expected

    def accept_control(self, command: ControlCommand, *, current_tick: int | None = None) -> None:
        self.validate_identity(command.identity)
        expected_seq = self.last_control_seq + 1
        if command.control_seq != expected_seq:
            raise SequenceError(
                f"control sequence must be {expected_seq}, got {command.control_seq}"
            )
        if command.issued_tick <= self.last_control_tick:
            raise SequenceError("control issued tick must increase")
        if current_tick is not None:
            if command.issued_tick > current_tick:
                raise FutureMessageError("control command is from a future game tick")
            if current_tick - command.issued_tick > self.max_action_age_ticks:
                raise StaleMessageError("control command exceeds the tick freshness bound")
        if len(self._pending_controls) >= self.max_control_window:
            raise SequenceError("control acknowledgement window is full")
        self.last_control_seq = command.control_seq
        self.last_control_tick = command.issued_tick
        self._pending_controls.append(command)

    def accept_control_ack(self, ack: ControlAck) -> ControlCommand:
        self.validate_identity(ack.identity)
        if not self._pending_controls:
            raise SequenceError("control acknowledgement has no pending command")
        expected = self._pending_controls[0]
        if (
            ack.control_seq != expected.control_seq
            or ack.issued_tick != expected.issued_tick
            or ack.operation is not expected.operation
            or ack.content_id != expected.content_id
            or ack.content_hash != expected.content_hash
        ):
            raise IdentityMismatchError("control acknowledgement does not bind the pending command")
        self._pending_controls.popleft()
        return expected

    def mark_applied(self, *, game_tick: int) -> None:
        if not self._pending_actions:
            raise SequenceError("no validated command is pending")
        target = self._pending_actions[0].target_tick
        if game_tick != target:
            raise SequenceError(f"next command targets tick {target}, not {game_tick}")
        self._pending_actions.popleft()

    def clear_pending(self) -> None:
        """Fail-closed transition: discard every command awaiting application."""

        self._pending_actions.clear()

    def clear_controls(self) -> None:
        self._pending_controls.clear()

    @property
    def pending_command_count(self) -> int:
        return len(self._pending_actions)

    @property
    def pending_control_count(self) -> int:
        return len(self._pending_controls)
