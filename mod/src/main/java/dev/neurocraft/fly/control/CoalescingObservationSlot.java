package dev.neurocraft.fly.control;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

/**
 * One-slot latest-observation handoff with an atomically bound, session-cumulative overwrite
 * count.
 */
final class CoalescingObservationSlot {
    private final AtomicReference<State> state =
            new AtomicReference<>(new State(null, 0));

    /**
     * Publishes the latest snapshot. Returns false only when another overwrite cannot be counted
     * without exceeding the protocol's signed-int64 bound.
     */
    boolean offer(PhysicalSensorSnapshot snapshot) {
        Objects.requireNonNull(snapshot, "snapshot");
        // Stamped here and carried through the CAS so the time belongs to the exact
        // snapshot that survives coalescing, not to whichever offer happened last.
        long offeredMonotonicNs = System.nanoTime();
        while (true) {
            State current = state.get();
            boolean overwritesPending = current.pending() != null;
            if (overwritesPending && current.coalescedTotal() == Long.MAX_VALUE) {
                return false;
            }
            long nextTotal = current.coalescedTotal() + (overwritesPending ? 1 : 0);
            State next = new State(
                    new PendingObservation(snapshot, nextTotal, offeredMonotonicNs), nextTotal);
            if (state.compareAndSet(current, next)) {
                return true;
            }
        }
    }

    PendingObservation pollLatest() {
        while (true) {
            State current = state.get();
            if (current.pending() == null) {
                return null;
            }
            State next = new State(null, current.coalescedTotal());
            if (state.compareAndSet(current, next)) {
                return current.pending();
            }
        }
    }

    boolean hasPending() {
        return state.get().pending() != null;
    }

    void clear() {
        state.updateAndGet(current -> current.pending() == null
                ? current
                : new State(null, current.coalescedTotal()));
    }

    /**
     * The published snapshot, its cumulative coalescence count, and the raw
     * {@link System#nanoTime()} reading taken when this snapshot was offered.
     * The timestamp is diagnostic only: it never reaches the wire, and it is raw
     * rather than sign-masked because only differences are ever taken from it.
     */
    record PendingObservation(
            PhysicalSensorSnapshot snapshot,
            long bridgeObservationsCoalesced,
            long offeredMonotonicNs) {
        PendingObservation {
            Objects.requireNonNull(snapshot, "snapshot");
            if (bridgeObservationsCoalesced < 0) {
                throw new IllegalArgumentException(
                        "bridge observation coalescence count must be non-negative");
            }
        }
    }

    private record State(PendingObservation pending, long coalescedTotal) {
        private State {
            if (coalescedTotal < 0) {
                throw new IllegalArgumentException(
                        "bridge observation coalescence total must be non-negative");
            }
        }
    }
}
