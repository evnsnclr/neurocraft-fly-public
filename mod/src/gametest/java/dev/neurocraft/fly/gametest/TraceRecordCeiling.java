package dev.neurocraft.fly.gametest;

import dev.neurocraft.fly.control.ActionFrame;
import java.util.Objects;

/**
 * Test-only witness for the atomic RECORD_STOP boundary. Actions targeting the acknowledgement's
 * effective tick are included; the first action targeting a later tick seals the trace ceiling.
 */
final class TraceRecordCeiling {
    private final long effectiveTick;
    private long lastObservedCommandSequence = -1;
    private long highestIncludedCommandSequence = -1;
    private long firstLaterCommandSequence = -1;

    TraceRecordCeiling(long effectiveTick) {
        if (effectiveTick < 0) {
            throw new IllegalArgumentException("record stop effective tick is out of range");
        }
        this.effectiveTick = effectiveTick;
    }

    boolean observe(ActionFrame action) {
        Objects.requireNonNull(action, "action");
        long commandSequence = action.commandSequence();
        if (commandSequence == lastObservedCommandSequence) {
            return false;
        }
        if (commandSequence < lastObservedCommandSequence) {
            throw new IllegalStateException("record stop actions are not sequence-monotonic");
        }
        if (firstLaterCommandSequence >= 0) {
            if (action.targetTick() <= effectiveTick) {
                throw new IllegalStateException(
                        "record stop target tick returned inside a sealed trace boundary");
            }
            lastObservedCommandSequence = commandSequence;
            return false;
        }
        if (action.targetTick() <= effectiveTick) {
            highestIncludedCommandSequence = commandSequence;
            lastObservedCommandSequence = commandSequence;
            return false;
        }
        if (highestIncludedCommandSequence < 0) {
            throw new IllegalStateException(
                    "record stop boundary has no observed action at or before its effective tick");
        }
        if (commandSequence != highestIncludedCommandSequence + 1) {
            throw new IllegalStateException("record stop boundary contains an action sequence gap");
        }
        firstLaterCommandSequence = commandSequence;
        lastObservedCommandSequence = commandSequence;
        return true;
    }

    boolean isSealed() {
        return firstLaterCommandSequence >= 0;
    }

    long commandSequence() {
        if (!isSealed()) {
            throw new IllegalStateException("record stop trace ceiling is not sealed");
        }
        return highestIncludedCommandSequence;
    }

    long firstLaterCommandSequence() {
        if (!isSealed()) {
            throw new IllegalStateException("record stop trace ceiling is not sealed");
        }
        return firstLaterCommandSequence;
    }
}
