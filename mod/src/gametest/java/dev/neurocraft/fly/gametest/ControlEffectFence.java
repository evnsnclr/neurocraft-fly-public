package dev.neurocraft.fly.gametest;

import dev.neurocraft.fly.control.ActionFrame;
import java.util.Objects;

/**
 * Test-only causal fence between an accepted control acknowledgement and the first action that can
 * reflect that control. The latched command never advances while its receipt is in flight.
 */
final class ControlEffectFence {
    private final long acknowledgementActionSequence;
    private final long effectiveTick;
    private long latchedCommandSequence = -1;

    ControlEffectFence(long acknowledgementActionSequence, long effectiveTick) {
        if (acknowledgementActionSequence < -1 || effectiveTick < 0) {
            throw new IllegalArgumentException("control effect fence values are out of range");
        }
        this.acknowledgementActionSequence = acknowledgementActionSequence;
        this.effectiveTick = effectiveTick;
    }

    boolean tryLatch(ActionFrame action) {
        Objects.requireNonNull(action, "action");
        if (latchedCommandSequence >= 0
                || action.commandSequence() <= acknowledgementActionSequence
                || action.observationTick() != effectiveTick) {
            return false;
        }
        latchedCommandSequence = action.commandSequence();
        return true;
    }

    long latchedCommandSequence() {
        if (latchedCommandSequence < 0) {
            throw new IllegalStateException("control effect action has not been latched");
        }
        return latchedCommandSequence;
    }

    boolean receiptSentThrough(long lastSentReceiptCommandSequence) {
        return latchedCommandSequence >= 0
                && lastSentReceiptCommandSequence >= latchedCommandSequence;
    }
}
