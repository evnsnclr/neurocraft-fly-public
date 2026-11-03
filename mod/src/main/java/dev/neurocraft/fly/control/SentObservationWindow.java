package dev.neurocraft.fly.control;

import java.util.ArrayDeque;
import java.util.OptionalLong;

/**
 * Thread-safe ledger of observations this authority has sent, used to verify that an
 * arriving action frame really binds an observation this session produced.
 *
 * <p>It deliberately no longer matches the service-side validator's last-three observation
 * deque. ADR 0018's COLD_START_GRACE_TICKS lets a cold session survive seven frameless
 * ticks instead of two, which makes frames up to seven ticks older than the current tick
 * reachable at all for the first time. A frame whose receipt has aged out of this ledger is
 * unverifiable and latches STALE_COMMAND, so a three-entry ledger would have converted the
 * grace into a newly reachable hard fault. The capacity is therefore the graced cold-start
 * span plus the current tick — FIRST_ACTION_DEADLINE_TICKS + COLD_START_GRACE_TICKS + 1 —
 * so every frame the graced hold tolerates stays verifiable and, being past
 * MAX_COMMAND_AGE_TICKS, retires fail-neutral (ADR 0011) instead of latching.
 *
 * <p>Declared trade: a replayed or very late frame whose observation was sent three to
 * seven ticks ago (the bound is in ledger entries, so it spans more wall-clock ticks
 * whenever observations coalesce) is now verified and then refused fail-neutral by the unchanged
 * {@code age > MAX_COMMAND_AGE_TICKS} rule, rather than latching STALE_COMMAND. What is
 * lost is adversarial <em>signal</em>, not integrity — such a frame remains impossible to
 * apply, and sequence continuity still latches SEQUENCE_ERROR on an actual replay.
 */
final class SentObservationWindow {
    private static final int CAPACITY = (int)
            (CommandValidator.FIRST_ACTION_DEADLINE_TICKS
                    + CommandValidator.COLD_START_GRACE_TICKS
                    + 1);

    private final ArrayDeque<Receipt> receipts = new ArrayDeque<>(CAPACITY);

    synchronized void record(long sequence, long gameTick) {
        if (sequence < 0 || gameTick < 0) {
            throw new IllegalArgumentException("observation receipt values must be non-negative");
        }
        if (!receipts.isEmpty() && sequence <= receipts.getLast().sequence()) {
            throw new IllegalStateException("observation receipts must increase");
        }
        if (receipts.size() == CAPACITY) {
            receipts.removeFirst();
        }
        receipts.addLast(new Receipt(sequence, gameTick));
    }

    synchronized OptionalLong tickFor(long sequence) {
        for (Receipt receipt : receipts) {
            if (receipt.sequence() == sequence) {
                return OptionalLong.of(receipt.gameTick());
            }
        }
        return OptionalLong.empty();
    }

    synchronized void clear() {
        receipts.clear();
    }

    private record Receipt(long sequence, long gameTick) {}
}
