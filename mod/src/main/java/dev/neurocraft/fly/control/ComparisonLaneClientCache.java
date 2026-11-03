package dev.neurocraft.fly.control;

import java.util.Optional;

/** Authenticated lane identity fences optional display updates across session changes. */
public final class ComparisonLaneClientCache {
    private final EventDisplayClientCache events = new EventDisplayClientCache();
    private final BrainDisplayClientCache display = new BrainDisplayClientCache();
    private ComparisonLanePresentation lane = ComparisonLanePresentation.unavailable();
    private boolean comparisonSeen;
    private long latestEpoch = -1;

    public void receiveLane(ComparisonLanePresentation next, long nowNanos) {
        if (next.status() != ComparisonLanePresentation.Status.UNAVAILABLE) {
            if (next.epoch() < latestEpoch) {
                return;
            }
            if (next.epoch() == latestEpoch
                    && lane.status() == ComparisonLanePresentation.Status.FAILED
                    && next.status() != ComparisonLanePresentation.Status.FAILED) {
                return;
            }
            comparisonSeen = true;
            latestEpoch = next.epoch();
        }
        if (!lane.equals(next)) {
            display.receive(unavailable(), nowNanos);
            boolean candidate = next.activeIdentity().filter(id ->
                    ControlProtocol.isEventObservationSchema(id.observationSchemaHash())).isPresent();
            events.receive(candidate || events.view(nowNanos).status() != EventDisplayState.Status.DISABLED
                    ? EventDisplayPresentation.unavailable() : EventDisplayPresentation.disabled(), nowNanos);
        }
        lane = next;
    }

    public void receiveDisplay(BrainDisplayPresentation next, long nowNanos) {
        if (comparisonSeen && next.frame().isPresent()
                && (lane.activeIdentity().isEmpty()
                        || !lane.activeIdentity().orElseThrow().equals(next.frame().orElseThrow().identity()))) {
            return;
        }
        display.receive(next, nowNanos);
    }

    public void receiveEvents(EventDisplayPresentation next, long nowNanos) {
        if (comparisonSeen && next.frame().isPresent()
                && (lane.activeIdentity().isEmpty()
                        || !lane.activeIdentity().orElseThrow().equals(next.frame().orElseThrow().identity()))) {
            return;
        }
        events.receive(next, nowNanos);
    }

    public EventDisplayPresentation events(long nowNanos) { return events.view(nowNanos); }

    public ComparisonLanePresentation lane() { return lane; }
    public BrainDisplayPresentation display(long nowNanos) { return display.view(nowNanos); }

    private static BrainDisplayPresentation unavailable() {
        return new BrainDisplayPresentation(BrainDisplayState.Status.UNAVAILABLE, 0, Optional.empty());
    }
}
