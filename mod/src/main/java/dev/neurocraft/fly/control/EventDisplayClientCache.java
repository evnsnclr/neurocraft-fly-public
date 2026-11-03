package dev.neurocraft.fly.control;

import java.util.Optional;

/** Receive-time freshness independent of when the HUD first becomes visible. */
public final class EventDisplayClientCache {
    private EventDisplayPresentation value = EventDisplayPresentation.disabled();
    private long receivedNanos;

    public void receive(EventDisplayPresentation presentation, long nowNanos) {
        value = presentation;
        receivedNanos = nowNanos;
    }

    public EventDisplayPresentation view(long nowNanos) {
        if (value.status() != EventDisplayState.Status.MATCHED) {
            return value;
        }
        long elapsed = nowNanos - receivedNanos;
        long elapsedMillis = Math.max(0, elapsed / 1_000_000);
        long age = value.ageMillis() > Long.MAX_VALUE - elapsedMillis
                ? Long.MAX_VALUE : value.ageMillis() + elapsedMillis;
        if (elapsed < 0 || age > EventDisplayState.MAX_AGE_NANOS / 1_000_000) {
            return new EventDisplayPresentation(EventDisplayState.Status.STALE, age, Optional.empty());
        }
        return new EventDisplayPresentation(value.status(), age, value.frame());
    }
}
