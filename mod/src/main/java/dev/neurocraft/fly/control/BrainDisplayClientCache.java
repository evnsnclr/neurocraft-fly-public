package dev.neurocraft.fly.control;

import java.util.Optional;

/** Receive-time freshness independent of when the HUD first becomes visible. */
public final class BrainDisplayClientCache {
    private BrainDisplayPresentation value = new BrainDisplayPresentation(
            BrainDisplayState.Status.UNAVAILABLE, 0, Optional.empty());
    private long receivedNanos;

    public void receive(BrainDisplayPresentation presentation, long nowNanos) {
        value = presentation;
        receivedNanos = nowNanos;
    }

    public BrainDisplayPresentation view(long nowNanos) {
        if (value.status() != BrainDisplayState.Status.MATCHED) {
            return value;
        }
        long elapsed = nowNanos - receivedNanos;
        long elapsedMillis = Math.max(0, elapsed / 1_000_000);
        long age = value.ageMillis() > Long.MAX_VALUE - elapsedMillis
                ? Long.MAX_VALUE : value.ageMillis() + elapsedMillis;
        if (elapsed < 0 || age > BrainDisplayState.MAX_AGE_NANOS / 1_000_000) {
            return new BrainDisplayPresentation(BrainDisplayState.Status.STALE, age, Optional.empty());
        }
        return new BrainDisplayPresentation(value.status(), age, value.frame());
    }
}
