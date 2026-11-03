package dev.neurocraft.fly.control;

import dev.neurocraft.fly.body.SelectionSource;
import java.util.ArrayDeque;
import java.util.List;

/** Bounded client-observed history of already matched display packets; never controls the model. */
public final class EventReplayHistory {
    public static final int MAX_SAMPLES = 256;
    public static final List<String> SOURCES = List.of(
            "attack", "player_approach", "mob_approach", "food", "brush", "light");

    public record Sample(EventDisplayFrame frame, String lane, String seed) {
        /** A source may be observed/eligible without passing the neural selection gate. */
        public String inspectedSource() {
            var interaction = frame.interaction();
            if (interaction.source() != null) { return interaction.source(); }
            return SOURCES.stream().filter(name -> interaction.scores().get(name).value().eligible())
                    .findFirst().orElse(null);
        }

        public boolean selectedSource() { return frame.interaction().source() != null; }
    }

    private final ArrayDeque<Sample> samples = new ArrayDeque<>();
    private EventDisplayFrame latest;

    /**
     * MATCHED means the server found an exact successfully applied program command. The
     * source payload is still optional diagnostic evidence: it contains requested pulses,
     * not an executed sensory vector, and receipt age is not reaction latency.
     */
    public boolean record(EventDisplayPresentation presentation, ComparisonLanePresentation lane,
            ControllerFault state, SelectionSource authority) {
        if (state != ControllerFault.ACTIVE || authority != SelectionSource.READOUT
                || presentation.status() != EventDisplayState.Status.MATCHED
                || presentation.ageMillis() > EventDisplayState.MAX_AGE_NANOS / 1_000_000
                || (lane.status() != ComparisonLanePresentation.Status.ACTIVE
                        && lane.status() != ComparisonLanePresentation.Status.UNAVAILABLE)) {
            return false;
        }
        var frame = presentation.frame().orElseThrow();
        if (frame.interaction() == null
                || (lane.status() == ComparisonLanePresentation.Status.ACTIVE
                        && !lane.activeIdentity().orElseThrow().equals(frame.identity()))
                || (latest != null && latest.identity().equals(frame.identity())
                        && frame.commandSequence() <= latest.commandSequence())) {
            return false;
        }
        samples.addLast(new Sample(frame,
                lane.status() == ComparisonLanePresentation.Status.ACTIVE ? lane.activeLane() : "unavailable",
                lane.seed()));
        latest = frame;
        while (samples.size() > MAX_SAMPLES) { samples.removeFirst(); }
        return true;
    }

    public List<Sample> snapshot() { return List.copyOf(samples); }

    public void clear() {
        samples.clear();
        latest = null;
    }

    public static String sourceLabel(String source) {
        if (source == null) { return "No selected or eligible source"; }
        return switch (source) {
            case "attack" -> "Player attack attempt (tactile proxy)";
            case "player_approach" -> "Player approach (Minecraft category)";
            case "mob_approach" -> "Mob approach (Minecraft category)";
            case "food" -> "Sweet food near/contact";
            case "brush" -> "Brush contact";
            case "light" -> "Local light change";
            default -> throw new IllegalArgumentException("unknown interaction source");
        };
    }
}
