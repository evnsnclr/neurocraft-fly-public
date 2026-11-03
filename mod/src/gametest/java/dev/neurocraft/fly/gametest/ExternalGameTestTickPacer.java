package dev.neurocraft.fly.gametest;

import java.util.Objects;
import java.util.concurrent.locks.LockSupport;
import java.util.function.LongConsumer;
import java.util.function.LongSupplier;

/**
 * Monotonic wall-clock pacing for the opt-in external GameTest process only.
 *
 * <p>Vanilla's {@code GameTestServer.waitUntilNextTick()} deliberately does not wait. This pacer
 * runs from the external probe callback after the entity tick and prevents the next test-server
 * tick from starting early. Overshoot resets the next absolute deadline from the actual release
 * time, so a slow tick is never followed by an unsafe catch-up burst.
 */
final class ExternalGameTestTickPacer {
    static final String MODE = "external_gametest_probe_monotonic_20hz";
    static final long PERIOD_NANOS = 50_000_000L;

    private final LongSupplier nanoTime;
    private final LongConsumer parkNanos;
    private long previousReleaseNanos;
    private long nextDeadlineNanos;
    private long pacedTickCount;
    private long minimumIntervalNanos = Long.MAX_VALUE;

    static ExternalGameTestTickPacer system() {
        return new ExternalGameTestTickPacer(System::nanoTime, LockSupport::parkNanos);
    }

    ExternalGameTestTickPacer(LongSupplier nanoTime, LongConsumer parkNanos) {
        this.nanoTime = Objects.requireNonNull(nanoTime, "nanoTime");
        this.parkNanos = Objects.requireNonNull(parkNanos, "parkNanos");
        previousReleaseNanos = nanoTime.getAsLong();
        nextDeadlineNanos = previousReleaseNanos + PERIOD_NANOS;
    }

    void pace() {
        while (true) {
            if (Thread.currentThread().isInterrupted()) {
                throw new IllegalStateException("external GameTest pacing thread was interrupted");
            }
            long remainingNanos = nextDeadlineNanos - nanoTime.getAsLong();
            if (remainingNanos <= 0) {
                break;
            }
            parkNanos.accept(remainingNanos);
        }
        long releasedNanos = nanoTime.getAsLong();
        long intervalNanos = releasedNanos - previousReleaseNanos;
        if (intervalNanos < minimumIntervalNanos) {
            minimumIntervalNanos = intervalNanos;
        }
        pacedTickCount++;
        previousReleaseNanos = releasedNanos;
        nextDeadlineNanos = releasedNanos + PERIOD_NANOS;
    }

    long pacedTickCount() {
        return pacedTickCount;
    }

    long minimumIntervalNanos() {
        return minimumIntervalNanos;
    }

    boolean hasVerifiedCadence() {
        return pacedTickCount >= 2 && minimumIntervalNanos >= PERIOD_NANOS;
    }
}
