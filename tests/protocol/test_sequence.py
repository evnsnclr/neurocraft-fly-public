from __future__ import annotations

from dataclasses import replace

import pytest
from neurocraft_protocol import (
    ActuationReceipt,
    ActuationValidation,
    ControlOperation,
    ControlState,
    FaultCode,
    FutureMessageError,
    Hello,
    IdentityMismatchError,
    MotorVector,
    PeerRole,
    ProgramId,
    ProgramValidation,
    ReceiptMismatchError,
    SequenceError,
    SessionValidator,
    StaleMessageError,
)

from .helpers import (
    action,
    control_ack,
    control_command,
    identity,
    modulation,
    observation,
    parameters,
    program_action,
    program_receipt,
    receipt,
    world_frame,
)


def test_hello_requires_exact_identity_and_role() -> None:
    validator = SessionValidator(identity())
    validator.accept_hello(
        Hello.create(identity=identity(), role=PeerRole.MINECRAFT_SERVER, token="A" * 43),
        expected_role=PeerRole.MINECRAFT_SERVER,
    )
    with pytest.raises(IdentityMismatchError):
        validator.accept_hello(
            Hello.create(identity=identity(), role=PeerRole.NEURAL_SERVICE, token="A" * 43),
            expected_role=PeerRole.MINECRAFT_SERVER,
        )
    with pytest.raises(IdentityMismatchError):
        validator.accept_hello(
            Hello.create(
                identity=identity(session_epoch=2), role=PeerRole.MINECRAFT_SERVER, token="A" * 43
            ),
            expected_role=PeerRole.MINECRAFT_SERVER,
        )


def test_observations_require_exact_sequence_and_increasing_tick() -> None:
    validator = SessionValidator(identity())
    validator.accept_observation(observation(sequence=0, tick=100), current_tick=100)
    with pytest.raises(SequenceError):
        validator.accept_observation(observation(sequence=0, tick=101), current_tick=101)

    validator = SessionValidator(identity())
    with pytest.raises(SequenceError):
        validator.accept_observation(observation(sequence=1, tick=100), current_tick=100)

    validator = SessionValidator(identity())
    validator.accept_observation(observation(sequence=0, tick=100), current_tick=100)
    with pytest.raises(SequenceError):
        validator.accept_observation(observation(sequence=1, tick=100), current_tick=100)


def test_future_and_stale_observations_are_rejected() -> None:
    with pytest.raises(FutureMessageError):
        SessionValidator(identity()).accept_observation(
            observation(sequence=0, tick=101), current_tick=100
        )
    with pytest.raises(StaleMessageError):
        SessionValidator(identity()).accept_observation(
            observation(sequence=0, tick=97), current_tick=100
        )


def test_valid_action_binds_known_observation_and_can_be_applied() -> None:
    validator = SessionValidator(identity())
    validator.accept_observation(observation(sequence=0, tick=100), current_tick=100)
    validator.accept_action(action(), current_tick=100)
    assert validator.pending_command_count == 1
    validator.mark_applied(game_tick=102)
    assert validator.pending_command_count == 0


def test_action_identity_observation_and_command_sequence_are_strict() -> None:
    validator = SessionValidator(identity())
    validator.accept_observation(observation(), current_tick=100)
    with pytest.raises(IdentityMismatchError):
        validator.accept_action(
            action(run_identity=identity(graph_hash="d" * 64)), current_tick=100
        )
    with pytest.raises(IdentityMismatchError):
        validator.accept_action(
            action(observation_tick=99),
            current_tick=100,
        )
    with pytest.raises(FutureMessageError):
        validator.accept_action(action(observation_sequence=1), current_tick=100)
    with pytest.raises(SequenceError):
        validator.accept_action(action(command_sequence=1), current_tick=100)


def test_stale_and_future_action_ticks_are_rejected() -> None:
    validator = SessionValidator(identity())
    validator.accept_observation(observation(tick=100), current_tick=100)
    with pytest.raises(StaleMessageError):
        validator.accept_action(action(), current_tick=103)
    # An action whose own observation is ahead of the authoritative clock. The target-side
    # future bound is now subsumed by the frame model — an action may not declare validity
    # past observation + ACTION_TARGET_OFFSET_TICKS, so with a non-negative age its target
    # cannot exceed current + ACTION_TARGET_OFFSET_TICKS either — and this is the reachable
    # future path that remains.
    ahead = SessionValidator(identity())
    ahead.accept_observation(observation(tick=101), current_tick=101)
    with pytest.raises(FutureMessageError):
        ahead.accept_action(action(observation_tick=101), current_tick=100)


@pytest.mark.parametrize("make_command", [action, program_action])
@pytest.mark.parametrize("newer", range(4, 8))
def test_retained_completed_binding_does_not_extend_application_freshness(
    make_command, newer
) -> None:
    validator = SessionValidator(identity())
    for sequence in range(newer + 1):
        validator.accept_observation(observation(sequence=sequence, tick=100 + sequence))
    completed = make_command()
    # A real-current-tick caller still refuses a late frame despite retaining its identity.
    with pytest.raises(StaleMessageError, match="freshness"):
        validator.accept_action(completed, current_tick=100 + newer)
    assert validator.pending_command_count == 0
    # The service's source-tick validation permits sending only; retirement releases its slot.
    validator.accept_action(completed, current_tick=100)
    assert validator.retire_expired_actions(current_tick=100 + newer) == (completed,)
    assert validator.retired_action_count == 1
    assert validator.pending_command_count == 0
    assert validator.last_command_seq == 0 and validator.last_neural_step_end == 5


@pytest.mark.parametrize("make_command", [action, program_action])
def test_eight_entry_history_preserves_binding_checks_and_remains_bounded(make_command) -> None:
    validator = SessionValidator(identity())
    for sequence in range(8):
        validator.accept_observation(observation(sequence=sequence, tick=100 + sequence))
    with pytest.raises(IdentityMismatchError):
        validator.accept_action(
            make_command(run_identity=identity(session_epoch=2)), current_tick=100
        )
    with pytest.raises(IdentityMismatchError, match="observation tick"):
        validator.accept_action(make_command(observation_tick=99), current_tick=100)
    with pytest.raises(FutureMessageError, match="future observation"):
        validator.accept_action(make_command(observation_sequence=8), current_tick=100)
    assert validator.pending_command_count == 0 and validator.last_command_seq == -1

    validator.accept_observation(observation(sequence=8, tick=108))
    with pytest.raises(StaleMessageError, match="unavailable observation"):
        validator.accept_action(make_command(), current_tick=100)
    # The immediately newer actual binding remains available; eviction removes one entry.
    validator.accept_action(
        make_command(observation_sequence=1, observation_tick=101), current_tick=101
    )
    assert validator.last_command_seq == 0


def test_retained_history_does_not_enlarge_command_reservation_window() -> None:
    validator = SessionValidator(identity())
    for sequence in range(8):
        validator.accept_observation(observation(sequence=sequence, tick=100 + sequence))
    for sequence in range(3):
        validator.accept_action(
            action(
                command_sequence=sequence,
                observation_sequence=sequence,
                observation_tick=100 + sequence,
                neural_start=5 * sequence,
            ),
            current_tick=100 + sequence,
        )
    with pytest.raises(SequenceError, match="window is full"):
        validator.accept_action(
            action(
                command_sequence=3, observation_sequence=3, observation_tick=103, neural_start=15
            ),
            current_tick=103,
        )
    assert validator.pending_command_count == 3 and validator.last_command_seq == 2


def test_a_command_that_arrives_a_tick_early_is_reserved_and_waits() -> None:
    """The behaviour the two-tick budget exists for: a frame may arrive before its tick.

    A command computed from observation 100 targets 102. If it reaches the validator at 101
    -- the fast case the budget buys margin for -- it is admitted and held: it is neither
    applied early nor retired, and the tick it is applied on is still exactly its target.
    """

    validator = SessionValidator(identity())
    validator.accept_observation(observation(sequence=0, tick=100), current_tick=100)
    validator.accept_observation(observation(sequence=1, tick=101), current_tick=101)
    early = action(command_sequence=0, observation_sequence=0, observation_tick=100)
    validator.accept_action(early, current_tick=101)
    assert validator.pending_command_count == 1

    # Tick 101 is not its tick: nothing is applied and nothing is retired.
    with pytest.raises(SequenceError, match="targets tick 102"):
        validator.mark_applied(game_tick=101)
    assert validator.retire_expired_actions(current_tick=101) == ()
    assert validator.pending_command_count == 1

    validator.mark_applied(game_tick=102)
    assert validator.pending_command_count == 0
    assert validator.retired_action_count == 0


def test_an_action_that_misses_the_declared_budget_is_refused() -> None:
    """The offset is a contract, checked on admission and not only where frames are made.

    The frame model permits any target from the observation tick to the budget ceiling so
    that frames recorded under the one-tick budget still parse; this is the rule that says
    a live controller obeyed the budget it declared.
    """

    validator = SessionValidator(identity())
    validator.accept_observation(observation(tick=100), current_tick=100)
    for target in (100, 101):
        with pytest.raises(SequenceError, match="declared action budget"):
            validator.accept_action(
                action(target_tick=target, valid_through_tick=102), current_tick=100
            )
    assert validator.pending_command_count == 0
    validator.accept_action(action(), current_tick=100)
    assert validator.pending_command_count == 1


def test_command_window_holds_the_action_budget_plus_one() -> None:
    """Three commands may be validated and unapplied at once, and no more.

    At tick 102 the authority can still apply the command targeting 102 (computed from
    observation 100) and the one targeting 103 (from 101), and the command computed from
    102 itself is arriving — ``ACTION_TARGET_OFFSET_TICKS + 1``. It was two commands under
    the one-tick budget for the same reason.

    The history window does not enlarge the command reservation window. With a common current
    tick, freshness admits at most three observations. Pin the overflow guard on a deliberately
    narrowed reservation window independently of observation-history retention.
    """

    validator = SessionValidator(identity())
    for index in range(3):
        validator.accept_observation(
            observation(sequence=index, tick=100 + index), current_tick=100 + index
        )
    for index in range(3):
        validator.accept_action(
            action(
                command_sequence=index,
                observation_sequence=index,
                observation_tick=100 + index,
                neural_start=index * 5,
            ),
            current_tick=102,
        )
    assert validator.pending_command_count == 3

    narrowed = SessionValidator(identity(), max_command_window=2)
    for index in range(2):
        narrowed.accept_observation(
            observation(sequence=index, tick=100 + index), current_tick=100 + index
        )
        narrowed.accept_action(
            action(
                command_sequence=index,
                observation_sequence=index,
                observation_tick=100 + index,
                neural_start=index * 5,
            ),
            current_tick=101,
        )
    narrowed.accept_observation(observation(sequence=2, tick=102), current_tick=102)
    with pytest.raises(SequenceError, match="window is full"):
        narrowed.accept_action(
            action(
                command_sequence=2, observation_sequence=2, observation_tick=102, neural_start=10
            ),
            current_tick=102,
        )
    narrowed.clear_pending()
    assert narrowed.pending_command_count == 0


def test_neural_intervals_cannot_overlap() -> None:
    validator = SessionValidator(identity())
    validator.accept_observation(observation(sequence=0, tick=100), current_tick=100)
    validator.accept_observation(observation(sequence=1, tick=101), current_tick=101)
    validator.accept_action(action(), current_tick=100)
    with pytest.raises(SequenceError, match="overlap"):
        validator.accept_action(
            action(
                command_sequence=1, observation_sequence=1, observation_tick=101, neural_start=4
            ),
            current_tick=101,
        )


def test_receipt_must_bind_oldest_pending_action_exactly() -> None:
    validator = SessionValidator(identity())
    validator.accept_observation(observation(), current_tick=100)
    issued = action()
    validator.accept_action(issued, current_tick=100)
    forged = replace(
        receipt(issued),
        requested_motors=MotorVector.neutral(),
        applied_motors=MotorVector.neutral(),
    )
    with pytest.raises(ReceiptMismatchError):
        validator.accept_receipt(forged)
    assert validator.accept_receipt(receipt(issued)) == issued
    assert validator.pending_command_count == 0
    with pytest.raises(ReceiptMismatchError):
        validator.accept_receipt(receipt(issued))


def test_fault_receipt_must_still_bind_pending_action() -> None:
    validator = SessionValidator(identity())
    validator.accept_observation(observation(), current_tick=100)
    issued = action()
    validator.accept_action(issued, current_tick=100)
    rejected = ActuationReceipt(
        identity=identity(),
        command_seq=0,
        observation_seq=0,
        target_tick=102,
        applied_tick=102,
        world_before=world_frame(),
        requested_motors=issued.motors,
        applied_motors=MotorVector.neutral(),
        validation=ActuationValidation.NEUTRAL_FAULT,
        fault_code=FaultCode.STALE,
    )
    validator.accept_receipt(rejected)
    assert validator.pending_command_count == 0


def test_control_sequence_and_ack_binding_prevent_replay_and_forgery() -> None:
    validator = SessionValidator(identity())
    first = control_command()
    validator.accept_control(first, current_tick=100)
    with pytest.raises(SequenceError):
        validator.accept_control(first, current_tick=100)
    forged = replace(
        control_ack(first),
        state=ControlState.RUNNING,
        operation=ControlOperation.RESET,
        effective_tick=101,
    )
    with pytest.raises(IdentityMismatchError):
        validator.accept_control_ack(forged)
    validator.accept_control_ack(control_ack(first))
    assert validator.pending_control_count == 0


def test_control_ticks_are_bounded_and_monotonic() -> None:
    validator = SessionValidator(identity())
    with pytest.raises(FutureMessageError):
        validator.accept_control(control_command(tick=101), current_tick=100)
    with pytest.raises(StaleMessageError):
        validator.accept_control(control_command(tick=97), current_tick=100)
    first = control_command(tick=100)
    validator.accept_control(first, current_tick=100)
    validator.accept_control_ack(control_ack(first))
    with pytest.raises(SequenceError):
        validator.accept_control(
            control_command(sequence=1, tick=100, operation=ControlOperation.RESUME),
            current_tick=100,
        )


def test_intervention_control_ack_is_bound_to_named_content_hash() -> None:
    validator = SessionValidator(identity())
    command = control_command(
        operation=ControlOperation.LESION_APPLY,
        content_id="declared-lesion",
        content_hash="a" * 64,
    )
    validator.accept_control(command, current_tick=100)
    forged = replace(control_ack(command), content_hash="b" * 64)
    with pytest.raises(IdentityMismatchError):
        validator.accept_control_ack(forged)
    validator.accept_control_ack(control_ack(command, state=ControlState.RUNNING))
    assert validator.pending_control_count == 0


def _fill_window_without_receipts(validator: SessionValidator) -> None:
    """Drive two validated actions into the window and never receipt either.

    This is what the authoritative side does whenever it refuses a command:
    `FlyControllerRuntime.recordActuation` returns without offering a receipt when
    the validator produced no applied command, so the service is never told.
    """

    validator.accept_observation(observation(sequence=0, tick=100), current_tick=100)
    validator.accept_action(
        action(command_sequence=0, observation_sequence=0, observation_tick=100, neural_start=0),
        current_tick=100,
    )
    validator.accept_observation(observation(sequence=1, tick=101), current_tick=101)
    validator.accept_action(
        action(command_sequence=1, observation_sequence=1, observation_tick=101, neural_start=5),
        current_tick=101,
    )
    assert validator.pending_command_count == 2


def test_unreceipted_expired_actions_do_not_wedge_the_command_window() -> None:
    """A refused command must not cost a window slot forever.

    Reproduces the live deadlock: the authority refuses an action (stale, or retired
    at a RESET boundary), sends no receipt, and the window fills. Every later action
    then raises SequenceError, the service fails closed, and the authority sits at
    STALE_COMMAND with command_seq=none.
    """

    validator = SessionValidator(identity())
    _fill_window_without_receipts(validator)

    # ADR 0018 moved the boundary to target_tick; the two-tick budget moved the targets.
    # Command 0 (observation 100) targets 102 and command 1 (observation 101) targets 103,
    # so tick 103 is the first tick past command 0's target and 104 the first past
    # command 1's: by 104 the authority can no longer apply either and neither side
    # should still be holding them.
    validator.accept_observation(observation(sequence=2, tick=102), current_tick=102)
    validator.accept_observation(observation(sequence=3, tick=103), current_tick=103)
    validator.accept_observation(observation(sequence=4, tick=104), current_tick=104)

    # Before the fix this raised SequenceError("validated command window is full").
    validator.accept_action(
        action(
            command_sequence=2,
            observation_sequence=4,
            observation_tick=104,
            neural_start=10,
        ),
        current_tick=104,
    )
    # Both wedging commands retired and command 2 took a free slot.
    assert validator.retired_action_count == 2
    assert validator.pending_command_count == 1


def test_retirement_only_drops_what_the_authority_can_no_longer_apply() -> None:
    validator = SessionValidator(identity())
    _fill_window_without_receipts(validator)

    # ADR 0018 corrected what "can no longer apply" means here. The authority
    # applies a command on exactly its target_tick, so at 102 command 0 (target
    # 102 under the two-tick budget) is applicable and nothing retires; the old
    # valid_through_tick rule held it one tick longer than the authority could ever
    # have used it, and the budget moved the tick it is applicable on.
    assert validator.retire_expired_actions(current_tick=102) == ()
    assert validator.pending_command_count == 2
    assert validator.retired_action_count == 0

    # At 103 command 0 is past its target; command 1 (target 103) survives.
    validator.accept_observation(observation(sequence=2, tick=102), current_tick=102)
    validator.accept_observation(observation(sequence=3, tick=103), current_tick=103)
    assert validator.pending_command_count == 1
    assert validator.retired_action_count == 1

    # At 104 command 1 is past its target too and the window empties.
    validator.accept_observation(observation(sequence=4, tick=104), current_tick=104)
    assert validator.pending_command_count == 0
    assert validator.retired_action_count == 2


def test_a_long_run_of_unreceipted_commands_never_wedges_the_window() -> None:
    """ADR 0018: the defect that killed a session AFTER it had cleared its cold start.

    The authority's ADR 0011 lateness retirement consumes a frame without ever
    applying it, so it produces no receipt. Under the old ``valid_through_tick``
    boundary the window head then survived one tick longer than the authority could
    ever have used it, so a run of receipt-less commands backed the two-slot window
    up and the third raised SequenceError("validated command window is full") — the
    service fails closed and the run ends. This killed session
    ``interactive-20260831T164331Z-9295f2f3`` after 2,538 successfully applied
    commands, and one of the three cold-start sessions after it had recovered.

    With the ``target_tick`` boundary each observation frees the head the authority is
    already past, so an unbounded run of receipt-less commands stays inside the window.
    Under the two-tick budget the window is one slot wider and each command is applicable
    one tick longer, so the steady state is three in flight instead of two; the property
    the test pins — that the run never wedges — is unchanged.
    """

    validator = SessionValidator(identity())
    for index in range(12):
        tick = 100 + index
        validator.accept_observation(observation(sequence=index, tick=tick), current_tick=tick)
        validator.accept_action(
            action(
                command_sequence=index,
                observation_sequence=index,
                observation_tick=tick,
                neural_start=index * 5,
            ),
            current_tick=tick,
        )
        assert validator.pending_command_count <= 3

    # Nine of the twelve retired (the three most recent are still legally applicable),
    # and not one accept raised. The old rule raised on the third iteration.
    assert validator.retired_action_count == 9
    assert validator.pending_command_count == 3


def test_a_receipt_for_a_retired_action_still_fails_closed() -> None:
    """Retirement must not silently absorb a late receipt for an expired command."""

    validator = SessionValidator(identity())
    _fill_window_without_receipts(validator)
    expired = action(
        command_sequence=0, observation_sequence=0, observation_tick=100, neural_start=0
    )
    validator.accept_observation(observation(sequence=2, tick=102), current_tick=102)
    validator.accept_observation(observation(sequence=3, tick=103), current_tick=103)
    validator.accept_observation(observation(sequence=4, tick=104), current_tick=104)
    # Two under ADR 0018's target_tick boundary: both filled commands are past
    # their targets by 104 under the two-tick budget. The pin is that the receipt
    # still fails closed.
    assert validator.retired_action_count == 2

    with pytest.raises(ReceiptMismatchError):
        validator.accept_receipt(receipt(expired))


def test_retired_actions_are_reported_rather_than_silently_dropped() -> None:
    """A retirement is a real signal that the authority is missing its deadline."""

    validator = SessionValidator(identity())
    _fill_window_without_receipts(validator)
    retired = validator.retire_expired_actions(current_tick=104)
    assert tuple(item.command_seq for item in retired) == (0, 1)
    assert validator.retired_action_count == 2
    assert validator.pending_command_count == 0


def test_later_receipt_supersedes_a_retired_unreceipted_head() -> None:
    """The authority's ADR 0011 retirement consumes a frame without a receipt;
    the successor's receipt proves the head was consumed unapplied, so it
    retires here instead of failing the session closed."""

    validator = SessionValidator(identity())
    validator.accept_observation(observation(sequence=0, tick=100), current_tick=100)
    validator.accept_observation(observation(sequence=1, tick=101), current_tick=101)
    retired_head = action()
    validator.accept_action(retired_head, current_tick=100)
    successor = action(
        command_sequence=1, observation_sequence=1, observation_tick=101, neural_start=5
    )
    validator.accept_action(successor, current_tick=101)

    assert validator.retired_action_count == 0
    assert validator.accept_receipt(receipt(successor)) == successor
    assert validator.retired_action_count == 1
    assert validator.pending_command_count == 0


def test_receipt_matching_no_pending_sequence_fails_closed_untouched() -> None:
    validator = SessionValidator(identity())
    validator.accept_observation(observation(sequence=0, tick=100), current_tick=100)
    issued = action()
    validator.accept_action(issued, current_tick=100)

    phantom = action(command_sequence=7, observation_sequence=0, observation_tick=100)
    with pytest.raises(ReceiptMismatchError):
        validator.accept_receipt(receipt(phantom))
    assert validator.pending_command_count == 1
    assert validator.retired_action_count == 0
    assert validator.accept_receipt(receipt(issued)) == issued


def test_supersession_never_excuses_field_forgery_on_the_matching_sequence() -> None:
    validator = SessionValidator(identity())
    validator.accept_observation(observation(sequence=0, tick=100), current_tick=100)
    validator.accept_observation(observation(sequence=1, tick=101), current_tick=101)
    validator.accept_action(action(), current_tick=100)
    successor = action(
        command_sequence=1, observation_sequence=1, observation_tick=101, neural_start=5
    )
    validator.accept_action(successor, current_tick=101)

    forged = replace(
        receipt(successor),
        requested_motors=MotorVector.neutral(),
        applied_motors=MotorVector.neutral(),
    )
    with pytest.raises(ReceiptMismatchError):
        validator.accept_receipt(forged)


def test_program_actions_are_admitted_and_retired_exactly_like_actions() -> None:
    """The window rules are envelope-only, so ADR 0018 retirement applies to program frames."""

    validator = SessionValidator(identity())
    validator.accept_observation(observation(sequence=0, tick=100), current_tick=100)
    first = program_action()
    validator.accept_action(first, current_tick=100)
    assert validator.pending_command_count == 1

    validator.accept_observation(observation(sequence=1, tick=101), current_tick=101)
    assert validator.pending_command_count == 1
    second = program_action(
        command_sequence=1, observation_sequence=1, observation_tick=101, neural_start=5
    )
    validator.accept_action(second, current_tick=101)
    assert validator.pending_command_count == 2

    # The world passes the first frame's target tick (102) without a receipt: its slot is
    # released and counted, exactly as an unreceipted action's is. Tick 102 is still the
    # tick the authority applies it on, so the release is at 103.
    validator.accept_observation(observation(sequence=2, tick=102), current_tick=102)
    assert validator.retired_action_count == 0
    assert validator.pending_command_count == 2
    validator.accept_observation(observation(sequence=3, tick=103), current_tick=103)
    assert validator.retired_action_count == 1
    assert validator.pending_command_count == 1


def test_program_actions_obey_the_shared_sequence_and_freshness_rules() -> None:
    validator = SessionValidator(identity())
    validator.accept_observation(observation(sequence=0, tick=100), current_tick=100)
    with pytest.raises(SequenceError, match="command sequence must be 0"):
        validator.accept_action(program_action(command_sequence=3), current_tick=100)
    validator.accept_action(program_action(), current_tick=100)
    validator.accept_observation(observation(sequence=1, tick=101), current_tick=101)
    with pytest.raises(SequenceError, match="overlap"):
        validator.accept_action(
            program_action(
                command_sequence=1, observation_sequence=1, observation_tick=101, neural_start=4
            ),
            current_tick=101,
        )


def test_program_receipt_binds_only_a_pending_program_action() -> None:
    validator = SessionValidator(identity())
    validator.accept_observation(observation(), current_tick=100)
    issued = program_action()
    validator.accept_action(issued, current_tick=100)

    # Each forgery is internally valid — the receipt's own executed values match its own
    # requested values — and still fails closed, because it echoes a frame the service
    # never issued.
    for divergent in (
        program_action(program=ProgramId.HOVER),
        program_action(program_modulation=modulation(speed=0.1)),
        program_action(program_parameters=parameters(turn_bias=0.5)),
    ):
        forged = program_receipt(divergent)
        assert forged.command_seq == issued.command_seq
        assert forged.target_tick == issued.target_tick
        with pytest.raises(ReceiptMismatchError):
            validator.accept_receipt(forged)
    assert validator.pending_command_count == 1

    assert validator.accept_receipt(program_receipt(issued)) == issued
    assert validator.pending_command_count == 0


def test_a_receipt_never_binds_a_pending_command_of_the_other_kind() -> None:
    """A kind mismatch is a contract violation, not a near miss: it fails closed untouched."""

    validator = SessionValidator(identity())
    validator.accept_observation(observation(), current_tick=100)
    issued_program = program_action()
    validator.accept_action(issued_program, current_tick=100)
    impostor = receipt(action())
    assert impostor.command_seq == issued_program.command_seq
    assert impostor.target_tick == issued_program.target_tick
    with pytest.raises(ReceiptMismatchError):
        validator.accept_receipt(impostor)
    assert validator.pending_command_count == 1
    assert validator.retired_action_count == 0
    assert validator.accept_receipt(program_receipt(issued_program)) == issued_program

    validator = SessionValidator(identity())
    validator.accept_observation(observation(), current_tick=100)
    issued_action = action()
    validator.accept_action(issued_action, current_tick=100)
    with pytest.raises(ReceiptMismatchError):
        validator.accept_receipt(program_receipt(program_action()))
    assert validator.pending_command_count == 1
    assert validator.accept_receipt(receipt(issued_action)) == issued_action


def test_program_receipt_supersession_retires_unreceipted_program_frames() -> None:
    validator = SessionValidator(identity())
    validator.accept_observation(observation(sequence=0, tick=100), current_tick=100)
    validator.accept_observation(observation(sequence=1, tick=101), current_tick=101)
    validator.accept_action(program_action(), current_tick=100)
    successor = program_action(
        command_sequence=1, observation_sequence=1, observation_tick=101, neural_start=5
    )
    validator.accept_action(successor, current_tick=101)

    assert validator.retired_action_count == 0
    assert validator.accept_receipt(program_receipt(successor)) == successor
    assert validator.retired_action_count == 1
    assert validator.pending_command_count == 0


def test_refused_program_receipt_cannot_claim_application_after_expiry() -> None:
    """A refused tick ran the HOVER fallback, so it is window-confined like an applied one."""

    validator = SessionValidator(identity())
    validator.accept_observation(observation(), current_tick=100)
    issued = program_action()
    validator.accept_action(issued, current_tick=100)
    late = replace(
        program_receipt(issued, validation=ProgramValidation.PROGRAM_REFUSED),
        applied_tick=issued.valid_through_tick + 1,
    )
    with pytest.raises(StaleMessageError, match="after command expiry"):
        validator.accept_receipt(late)
    assert validator.pending_command_count == 1


def test_rejected_program_receipt_keeps_its_real_validation_tick() -> None:
    validator = SessionValidator(identity())
    validator.accept_observation(observation(), current_tick=100)
    issued = program_action()
    validator.accept_action(issued, current_tick=100)
    rejected = program_receipt(
        issued,
        validation=ProgramValidation.REJECTED,
        applied_tick=issued.valid_through_tick + 5,
    )
    assert validator.accept_receipt(rejected) == issued
    assert validator.pending_command_count == 0
