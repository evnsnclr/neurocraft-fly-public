package dev.neurocraft.fly.control;

import java.util.ArrayDeque;
import java.util.Iterator;
import java.util.Optional;

/**
 * Bounded send-time ledger for the first observations of a session, used only to measure how long
 * an action takes to come back.
 *
 * <p>Neither side of the bridge can attribute first-action latency on its own. The service's
 * {@code compute_ns} starts inside its execution fence, so it excludes the time an observation
 * spends queued and every microsecond on the wire; the mod sees only that an action was late.
 * Pairing the instant an observation was offered with the arrival of the action bound to it splits
 * that total into transit plus service time.
 *
 * <p>The sample budget is spent, not refilled: only the first few actions of a session are
 * measured, so the logging this feeds cannot perturb steady-state operation. All timestamps are raw
 * {@link System#nanoTime()} readings; only differences are taken, so their sign does not matter.
 * Nothing here is evidence and nothing here reaches the wire.
 */
final class FirstActionLatencyWindow {
    /** Deep enough to outlive the service's three-observation validator window. */
    private static final int CAPACITY = 4;

    private final ArrayDeque<Offer> offers = new ArrayDeque<>(CAPACITY);
    private int remaining;

    FirstActionLatencyWindow(int samples) {
        if (samples < 0) {
            throw new IllegalArgumentException("latency sample budget must be non-negative");
        }
        remaining = samples;
    }

    /** Samples still to be taken; zero means this window has gone permanently quiet. */
    synchronized int remaining() {
        return remaining;
    }

    /** Records that {@code observationSequence} was handed to the socket. */
    synchronized void recordSend(
            long observationSequence, long offeredMonotonicNs, long sentMonotonicNs) {
        if (remaining == 0) {
            return;
        }
        if (offers.size() == CAPACITY) {
            offers.removeFirst();
        }
        offers.addLast(new Offer(observationSequence, offeredMonotonicNs, sentMonotonicNs));
    }

    /**
     * The measured sample for the action bound to {@code observationSequence}, or empty when the
     * budget is spent or that observation has already left the window. A returned sample consumes
     * one unit of the budget, so every observation is measured at most once.
     */
    synchronized Optional<Sample> completeArrival(
            long observationSequence, long arrivedMonotonicNs) {
        if (remaining == 0) {
            return Optional.empty();
        }
        Iterator<Offer> candidates = offers.iterator();
        while (candidates.hasNext()) {
            Offer offer = candidates.next();
            if (offer.observationSequence() != observationSequence) {
                continue;
            }
            candidates.remove();
            remaining--;
            return Optional.of(new Sample(
                    observationSequence,
                    arrivedMonotonicNs - offer.offeredMonotonicNs(),
                    arrivedMonotonicNs - offer.sentMonotonicNs()));
        }
        return Optional.empty();
    }

    /** Drops unmatched offers; the spent budget is not restored. */
    synchronized void clear() {
        offers.clear();
    }

    /**
     * One measured round trip: {@code offerToActionNs} spans the whole path from the server thread
     * publishing the snapshot, and {@code sendToActionNs} starts at the instant the wire's
     * {@code sent_monotonic_ns} records, so their difference is the bridge's own outbound queueing.
     */
    record Sample(long observationSequence, long offerToActionNs, long sendToActionNs) {}

    private record Offer(
            long observationSequence, long offeredMonotonicNs, long sentMonotonicNs) {}
}
