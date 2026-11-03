"""The service-side window ledger, exercised as the pure bookkeeping it is.

``SessionValidator`` counts retirements but does not name them, so the service mirrors the window
itself (``session_telemetry.py``). These tests hold the mirror to the validator's FIFO contract
and pin the paused-window lifecycle that decides what is lateness and what is not (R-29).
"""

from __future__ import annotations

from neurocraft_service.session_telemetry import (
    CommandKind,
    PausedWindow,
    SessionSummary,
    SessionTelemetry,
    paused_window_message,
)

from tests.protocol.helpers import action


def issue(telemetry: SessionTelemetry, sequence: int, tick: int, kind: CommandKind) -> None:
    telemetry.record_issued(
        action(command_sequence=sequence, observation_sequence=sequence, observation_tick=tick),
        kind=kind,
    )


def test_the_oldest_commands_are_the_ones_released_and_only_driving_ones_are_lateness() -> None:
    telemetry = SessionTelemetry()
    issue(telemetry, 0, 100, CommandKind.DRIVING)
    issue(telemetry, 1, 101, CommandKind.PAUSED_NEUTRAL)
    issue(telemetry, 2, 102, CommandKind.PAUSED_NEUTRAL)
    issue(telemetry, 3, 103, CommandKind.OTHER_NEUTRAL)

    first = telemetry.observation_released(2)

    assert (first.late, first.paused) == (1, 1)
    assert telemetry.lateness_retired == 1
    assert telemetry.paused_window_unapplied == 1

    # A replaying/faulted neutral action is not a paused-window action: it stays lateness.
    second = telemetry.observation_released(2)

    assert (second.late, second.paused) == (1, 1)
    assert telemetry.lateness_retired == 2
    assert telemetry.paused_window_unapplied == 2


def test_a_receipt_supersedes_the_heads_but_the_receipted_command_is_not_lateness() -> None:
    telemetry = SessionTelemetry()
    issue(telemetry, 0, 100, CommandKind.DRIVING)
    issue(telemetry, 1, 101, CommandKind.DRIVING)
    issue(telemetry, 2, 102, CommandKind.DRIVING)

    release = telemetry.receipt_released(1, superseded=1)

    assert (release.late, release.paused) == (1, 0)
    assert telemetry.lateness_retired == 1
    # Sequence 2 is the only command left; the next observation release names it.
    assert telemetry.observation_released(1).late == 1
    assert telemetry.summary(window_retirements_total=2).ledger_desyncs == 0


def test_a_paused_window_is_summarised_once_when_its_last_neutral_action_has_left() -> None:
    telemetry = SessionTelemetry()
    issue(telemetry, 0, 100, CommandKind.DRIVING)
    telemetry.receipt_released(0, superseded=0)
    for sequence in range(1, 9):
        issue(telemetry, sequence, 100 + sequence, CommandKind.PAUSED_NEUTRAL)
    assert telemetry.observation_released(5).paused == 5

    # RESUME lands while three neutral actions are still in the window: no line yet.
    assert telemetry.resume_accepted() is None
    issue(telemetry, 9, 109, CommandKind.DRIVING)
    assert telemetry.observation_released(1).completed_windows == ()
    assert telemetry.observation_released(1).completed_windows == ()

    release = telemetry.observation_released(1)

    assert release.completed_windows == (
        PausedWindow(
            index=1,
            first_tick=101,
            last_tick=108,
            issued=8,
            pending=0,
            unapplied=8,
            closed=True,
        ),
    )
    assert telemetry.unfinished_windows() == ()
    assert telemetry.summary(window_retirements_total=8) == SessionSummary(
        commands_issued=10,
        late_sends=0,
        lateness_retired=0,
        paused_window_unapplied=8,
        paused_windows=1,
        window_retirements_total=8,
        unattributed_retirements=0,
        ledger_desyncs=0,
    )
    assert paused_window_message(release.completed_windows[0]) == (
        "paused window ticks 101..108: 8 neutral action(s) issued while PAUSED were unreceipted "
        "(body ownership and cause unconfirmed; counted separately from driving lateness)"
    )


def test_a_resume_with_nothing_pending_returns_the_window_immediately() -> None:
    telemetry = SessionTelemetry()
    issue(telemetry, 0, 100, CommandKind.PAUSED_NEUTRAL)
    issue(telemetry, 1, 101, CommandKind.PAUSED_NEUTRAL)
    # The authority applied and receipted both: a plain pause, no body program in charge.
    telemetry.receipt_released(0, superseded=0)
    telemetry.receipt_released(1, superseded=0)

    window = telemetry.resume_accepted()

    assert window is not None
    assert (window.issued, window.pending, window.unapplied, window.applied) == (2, 0, 0, 2)
    assert telemetry.paused_window_unapplied == 0
    assert paused_window_message(window) == (
        "paused window ticks 100..101: all 2 neutral action(s) were applied and receipted by the "
        "authority (not lateness)"
    )


def test_a_resume_without_any_neutral_action_opens_and_closes_nothing() -> None:
    telemetry = SessionTelemetry()

    assert telemetry.resume_accepted() is None
    assert telemetry.summary(window_retirements_total=0).paused_windows == 0


def test_a_second_pause_can_open_while_the_first_window_is_still_draining() -> None:
    telemetry = SessionTelemetry()
    issue(telemetry, 0, 100, CommandKind.PAUSED_NEUTRAL)
    issue(telemetry, 1, 101, CommandKind.PAUSED_NEUTRAL)
    assert telemetry.resume_accepted() is None
    issue(telemetry, 2, 102, CommandKind.DRIVING)
    # PAUSE again before the first window's neutral actions have been retired.
    issue(telemetry, 3, 103, CommandKind.PAUSED_NEUTRAL)

    first_drain = telemetry.observation_released(2)
    assert [window.index for window in first_drain.completed_windows] == [1]
    assert first_drain.completed_windows[0].unapplied == 2

    assert telemetry.observation_released(1).late == 1
    assert telemetry.resume_accepted() is None
    second_drain = telemetry.observation_released(1)

    assert [window.index for window in second_drain.completed_windows] == [2]
    assert second_drain.completed_windows[0].first_tick == 103
    assert telemetry.summary(window_retirements_total=4) == SessionSummary(
        commands_issued=4,
        late_sends=0,
        lateness_retired=1,
        paused_window_unapplied=3,
        paused_windows=2,
        window_retirements_total=4,
        unattributed_retirements=0,
        ledger_desyncs=0,
    )


def test_a_window_the_session_ends_on_is_reported_as_unfinished() -> None:
    telemetry = SessionTelemetry()
    issue(telemetry, 0, 100, CommandKind.PAUSED_NEUTRAL)
    issue(telemetry, 1, 101, CommandKind.PAUSED_NEUTRAL)
    telemetry.observation_released(1)

    (still_open,) = telemetry.unfinished_windows()
    assert paused_window_message(still_open, session_ended=True) == (
        "paused window ticks 100..101: 1 neutral action(s) issued while PAUSED were unreceipted "
        "(body ownership and cause unconfirmed; counted separately from driving lateness); "
        "the session ended with the window still open "
        "and 1 neutral action(s) still in the validated window"
    )

    assert telemetry.resume_accepted() is None
    (closing,) = telemetry.unfinished_windows()
    assert paused_window_message(closing, session_ended=True) == (
        "paused window ticks 100..101: 1 neutral action(s) issued while PAUSED were unreceipted "
        "(body ownership and cause unconfirmed; counted separately from driving lateness); "
        "the session ended with 1 neutral action(s) "
        "still in the validated window"
    )


def test_a_mixed_window_names_both_dispositions() -> None:
    telemetry = SessionTelemetry()
    issue(telemetry, 0, 100, CommandKind.PAUSED_NEUTRAL)
    issue(telemetry, 1, 101, CommandKind.PAUSED_NEUTRAL)
    issue(telemetry, 2, 102, CommandKind.PAUSED_NEUTRAL)
    telemetry.receipt_released(0, superseded=0)
    telemetry.observation_released(2)

    window = telemetry.resume_accepted()

    assert window is not None
    assert paused_window_message(window) == (
        "paused window ticks 100..102: 2 neutral action(s) issued while PAUSED were unreceipted "
        "(body ownership and cause unconfirmed; counted separately from driving lateness); "
        "the other 1 were applied and receipted"
    )


def test_a_release_the_ledger_cannot_account_for_is_counted_not_raised() -> None:
    telemetry = SessionTelemetry()

    release = telemetry.observation_released(1)

    assert release.late == 0 and release.paused == 0
    summary = telemetry.summary(window_retirements_total=1)
    assert summary.ledger_desyncs == 1
    # A release the ledger could not attribute is, by the same token, unattributed.
    assert summary.unattributed_retirements == 1


def test_a_validator_retirement_the_ledger_never_saw_is_reported_as_unattributed() -> None:
    """``accept_receipt`` pops the superseded heads before it can still refuse the receipt, and
    the session then fails closed without that release reaching the ledger. The summary owns the
    gap rather than letting the two attributed figures pretend to cover the validator's count."""

    telemetry = SessionTelemetry()
    issue(telemetry, 0, 100, CommandKind.DRIVING)
    issue(telemetry, 1, 101, CommandKind.DRIVING)

    summary = telemetry.summary(window_retirements_total=1)

    assert summary.unattributed_retirements == 1
    assert summary.ledger_desyncs == 0
    assert (
        summary.lateness_retired
        + summary.paused_window_unapplied
        + summary.unattributed_retirements
        == summary.window_retirements_total
    )
