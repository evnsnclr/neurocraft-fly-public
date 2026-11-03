"""Per-session lateness and paused-window bookkeeping for the neural service.

``SessionValidator`` retires validated commands the authority can no longer apply and counts
them, but it does not say *which* commands they were, and its source is not edited here: it is
hashed into every run identity. Read from the count alone, a paused window looks exactly like a
peer missing its deadlines. That is risk R-29, and it fired live on 2026-09-04 (``STATUS.md``,
lane L): while the operator's HOVER owned the body for 1,786 ticks the service logged one
lateness WARNING per tick and ended the session at 1,790 "retired" against the four frames the
authority had actually retired as late.

This module tells the two apart. The service records what it puts into the validated window --
a controller-computed frame, or a neutral action it issued because it was PAUSED -- and, when
the validator releases ``k`` slots, resolves which ``k`` commands left. The mirror is exact
because the window is a FIFO of consecutive command sequences: ``accept_action`` admits only
``last_command_seq + 1`` and appends, ``retire_expired_actions`` pops from the head, and
``accept_receipt`` pops the superseded heads and then the receipted head. Every release
therefore advances the head by exactly one sequence, and "the oldest ``k``" is the whole answer.

A neutral action issued while PAUSED that leaves the window without a receipt was consumed
unapplied by the authority while a body program owned the body (ADR 0020 section 3: scripted
authority is gated on an accepted PAUSE and takes the receipt-free tick path). It is counted
under ``paused_window_unapplied``, never under ``lateness_retired``, and one INFO line per paused
window replaces the per-tick WARNING. Everything else -- a controller frame the world outran, a
single-stepped frame, a neutral action issued while REPLAYING or FAULTED -- keeps the lateness
WARNING and the lateness count exactly as before. Nothing here reaches the wire: no frame, no
validation and no state transition depends on this bookkeeping.

What the attribution cannot see: a neutral action the authority failed to apply during a pause
for some *other* reason (a dead or late peer) is indistinguishable, from this side, from one a
body program consumed by design. Only driving-lane frames carry lateness here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from neurocraft_protocol import Command


class CommandKind(StrEnum):
    """What the service knew about a command when it put it in the validated window."""

    DRIVING = "driving"
    """A controller-computed frame (``action`` or ``program_action``), a single step included."""

    PAUSED_NEUTRAL = "paused_neutral"
    """A neutral action issued because the service was PAUSED (ADR 0020 section 3)."""

    OTHER_NEUTRAL = "other_neutral"
    """A neutral action issued because the service was REPLAYING or FAULTED."""


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One command as the service recorded it on admission to the window."""

    command_seq: int
    observation_tick: int
    kind: CommandKind
    paused_window: int | None
    """Index of the paused window a ``PAUSED_NEUTRAL`` command belongs to; ``None`` otherwise."""


@dataclass(frozen=True, slots=True)
class PausedWindow:
    """One PAUSED span as the service saw it, keyed by the observation ticks it answered neutrally.

    ``first_tick``/``last_tick`` are observation ticks, not the PAUSE and RESUME issue ticks: the
    window is defined by what the service did while paused, which is answer observations with
    neutral actions. ``issued`` counts those; ``pending`` is how many are still in the validated
    window; ``unapplied`` is how many left it without a receipt.
    """

    index: int
    first_tick: int
    last_tick: int
    issued: int
    pending: int
    unapplied: int
    closed: bool
    """The RESUME that ended the window has been accepted."""

    @property
    def complete(self) -> bool:
        """Closed, and none of its neutral actions is still in the validated window."""

        return self.closed and self.pending == 0

    @property
    def applied(self) -> int:
        """Neutral actions the authority applied and receipted; zero under a body program."""

        return self.issued - self.pending - self.unapplied


@dataclass(frozen=True, slots=True)
class Release:
    """What one validator release contained, resolved against the ledger."""

    late: int
    """Commands consumed unapplied that are lateness: everything but paused-window neutrals."""

    paused: int
    """Paused-window neutral actions consumed unapplied."""

    completed_windows: tuple[PausedWindow, ...]
    """Paused windows this release emptied after their RESUME; each gets its one summary line."""


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """The session's telemetry, logged once when it ends and kept on the service.

    ``lateness_retired + paused_window_unapplied + unattributed_retirements ==
    window_retirements_total``, always: the first two partition the validator's own count between
    lateness and paused-window consumption, and the third is whatever the ledger never got to
    classify. The two named figures are never summed back into one "late" figure; that conflation
    is what R-29 records.
    """

    commands_issued: int
    late_sends: int
    lateness_retired: int
    paused_window_unapplied: int
    paused_windows: int
    window_retirements_total: int
    unattributed_retirements: int
    """Validator retirements the ledger did not classify: ``window_retirements_total`` less the two
    attributed figures. Zero in every session that ends cleanly. Nonzero when the validator retired
    heads and the session failed closed before the ledger saw the release -- ``accept_receipt``
    pops the superseded heads before it can still refuse the receipt itself -- or when the ledger
    missed entries (then also counted in ``ledger_desyncs``). Logged as a WARNING."""

    ledger_desyncs: int
    """Releases the ledger could not account for. Always zero unless the validator's window stops
    being a FIFO of consecutive sequences; a nonzero value means the attribution above is not
    trustworthy and is logged as a WARNING."""


def paused_window_message(window: PausedWindow, *, session_ended: bool = False) -> str:
    """The one INFO line a paused window gets, worded for the operator reading a live log."""

    head = f"paused window ticks {window.first_tick}..{window.last_tick}: "
    if window.unapplied == 0 and window.pending == 0:
        body = (
            f"all {window.issued} neutral action(s) were applied and receipted by the authority "
            "(not lateness)"
        )
    else:
        body = (
            f"{window.unapplied} neutral action(s) issued while PAUSED were unreceipted "
            "(body ownership and cause unconfirmed; counted separately from driving lateness)"
        )
        if window.applied:
            body += f"; the other {window.applied} were applied and receipted"
    if not session_ended:
        return head + body
    if not window.closed:
        return (
            f"{head}{body}; the session ended with the window still open and {window.pending} "
            "neutral action(s) still in the validated window"
        )
    return (
        f"{head}{body}; the session ended with {window.pending} neutral action(s) still in the "
        "validated window"
    )


class SessionTelemetry:
    """Mutable bookkeeping for one authenticated session; see the module docstring.

    Call order mirrors the validator's: :meth:`record_issued` after ``accept_action`` succeeds,
    :meth:`observation_released` with the count ``accept_observation`` retired,
    :meth:`receipt_released` after ``accept_receipt`` succeeds, :meth:`resume_accepted` when a
    RESUME is acknowledged as accepted, and :meth:`summary` once the session ends.
    """

    def __init__(self) -> None:
        self._entries: dict[int, LedgerEntry] = {}
        self._head_seq = 0
        self._windows: dict[int, PausedWindow] = {}
        self._open_window: int | None = None
        self._windows_opened = 0
        self._commands_issued = 0
        self._late_sends = 0
        self._lateness_retired = 0
        self._paused_window_unapplied = 0
        self._desyncs = 0

    @property
    def lateness_retired(self) -> int:
        """Driving-lane commands consumed unapplied so far: the "retired this session" figure."""

        return self._lateness_retired

    @property
    def paused_window_unapplied(self) -> int:
        """Paused-window neutral actions consumed unapplied so far."""

        return self._paused_window_unapplied

    @property
    def late_sends(self) -> int:
        return self._late_sends

    def record_issued(self, command: Command, *, kind: CommandKind) -> None:
        """Record a command the validator has just admitted to its window."""

        window: int | None = None
        if kind is CommandKind.PAUSED_NEUTRAL:
            window = self._open_or_extend_window(command.observation_tick)
        self._entries[command.command_seq] = LedgerEntry(
            command_seq=command.command_seq,
            observation_tick=command.observation_tick,
            kind=kind,
            paused_window=window,
        )
        self._commands_issued += 1

    def record_late_send(self) -> int:
        """Count a frame sent after the world passed its target tick; returns the running total."""

        self._late_sends += 1
        return self._late_sends

    def observation_released(self, count: int) -> Release:
        """Resolve ``count`` head retirements made by ``accept_observation``."""

        return self._consume(self._pop_oldest(count), receipted=False)

    def receipt_released(self, command_seq: int, *, superseded: int) -> Release:
        """Resolve a receipt: ``superseded`` heads consumed unapplied, then the receipted command.

        The receipted command left the window applied, so it is neither lateness nor unapplied;
        it can still complete a paused window whose last neutral action it was.
        """

        unapplied = self._consume(self._pop_oldest(superseded), receipted=False)
        receipted = self._pop_oldest(1)
        if receipted and receipted[0].command_seq != command_seq:
            self._desyncs += 1
        applied = self._consume(receipted, receipted=True)
        return Release(
            late=unapplied.late,
            paused=unapplied.paused,
            completed_windows=unapplied.completed_windows + applied.completed_windows,
        )

    def resume_accepted(self) -> PausedWindow | None:
        """Close the open paused window; return it only if nothing of it is still pending.

        A window with neutral actions still in the validated window is returned later, by the
        release that empties it, so its one summary line carries the exact count.
        """

        index = self._open_window
        self._open_window = None
        if index is None:
            return None
        window = replace(self._windows[index], closed=True)
        if window.complete:
            del self._windows[index]
            return window
        self._windows[index] = window
        return None

    def unfinished_windows(self) -> tuple[PausedWindow, ...]:
        """Windows still open, or still holding pending neutral actions, when the session ended."""

        return tuple(self._windows[index] for index in sorted(self._windows))

    def summary(self, *, window_retirements_total: int) -> SessionSummary:
        """The session's figures; ``window_retirements_total`` is the validator's own count."""

        return SessionSummary(
            commands_issued=self._commands_issued,
            late_sends=self._late_sends,
            lateness_retired=self._lateness_retired,
            paused_window_unapplied=self._paused_window_unapplied,
            paused_windows=self._windows_opened,
            window_retirements_total=window_retirements_total,
            unattributed_retirements=(
                window_retirements_total - self._lateness_retired - self._paused_window_unapplied
            ),
            ledger_desyncs=self._desyncs,
        )

    def _open_or_extend_window(self, observation_tick: int) -> int:
        if self._open_window is None:
            self._windows_opened += 1
            index = self._windows_opened
            self._windows[index] = PausedWindow(
                index=index,
                first_tick=observation_tick,
                last_tick=observation_tick,
                issued=1,
                pending=1,
                unapplied=0,
                closed=False,
            )
            self._open_window = index
            return index
        window = self._windows[self._open_window]
        self._windows[self._open_window] = replace(
            window,
            last_tick=observation_tick,
            issued=window.issued + 1,
            pending=window.pending + 1,
        )
        return self._open_window

    def _pop_oldest(self, count: int) -> tuple[LedgerEntry, ...]:
        popped: list[LedgerEntry] = []
        for _ in range(count):
            entry = self._entries.pop(self._head_seq, None)
            self._head_seq += 1
            if entry is None:
                self._desyncs += 1
            else:
                popped.append(entry)
        return tuple(popped)

    def _consume(self, entries: tuple[LedgerEntry, ...], *, receipted: bool) -> Release:
        late = 0
        paused = 0
        completed: list[PausedWindow] = []
        for entry in entries:
            if entry.paused_window is None:
                late += 0 if receipted else 1
                continue
            paused += 0 if receipted else 1
            window = self._windows[entry.paused_window]
            window = replace(
                window,
                pending=window.pending - 1,
                unapplied=window.unapplied + (0 if receipted else 1),
            )
            if window.complete:
                del self._windows[entry.paused_window]
                completed.append(window)
            else:
                self._windows[entry.paused_window] = window
        self._lateness_retired += late
        self._paused_window_unapplied += paused
        return Release(late=late, paused=paused, completed_windows=tuple(completed))
