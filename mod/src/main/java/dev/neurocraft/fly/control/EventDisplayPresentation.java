package dev.neurocraft.fly.control;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/**
 * One complete server-to-client synchronized display value. Age is milliseconds since first
 * file read on the server, not sample/transport latency. It is never a nanoTime timestamp
 * compared between different JVMs. Observation-tick freshness is checked separately by State.
 */
public record EventDisplayPresentation(
        EventDisplayState.Status status,
        long ageMillis,
        Optional<EventDisplayFrame> frame) {
    public static final String SCHEMA = "neurocraft.event_display_presentation.v1";
    public static final int MAX_BYTES = EventDisplayFrame.MAX_BYTES;

    public EventDisplayPresentation {
        Objects.requireNonNull(status, "status");
        Objects.requireNonNull(frame, "frame");
        if (ageMillis < 0 || (status == EventDisplayState.Status.MATCHED) != frame.isPresent()) {
            throw new IllegalArgumentException("only MATCHED presentations carry a frame; age >= 0");
        }
    }

    public static EventDisplayPresentation fromView(EventDisplayState.View view) {
        return new EventDisplayPresentation(view.status(), Math.max(0, view.ageNanos() / 1_000_000),
                view.frame());
    }

    public String encode() {
        Map<String, Object> values = new LinkedHashMap<>();
        values.put("schema", SCHEMA);
        values.put("status", status.name());
        values.put("age_ms", ageMillis);
        values.put("frame", frame.map(EventDisplayFrame::toMap).orElse(null));
        return StrictJson.encode(values, MAX_BYTES);
    }

    public static EventDisplayPresentation parse(String json) throws ProtocolException {
        Map<String, Object> root = StrictJson.parseObject(json, MAX_BYTES);
        BrainDisplayFrame.exact(root, Set.of("schema", "status", "age_ms", "frame"));
        if (!SCHEMA.equals(root.get("schema"))) {
            throw new ProtocolException("unsupported event display presentation schema");
        }
        try {
            EventDisplayState.Status status = EventDisplayState.Status.valueOf(
                    BrainDisplayFrame.text(root, "status"));
            long age = BrainDisplayFrame.integer(root, "age_ms");
            Optional<EventDisplayFrame> frame = root.get("frame") == null ? Optional.empty()
                    : Optional.of(EventDisplayFrame.parse(StrictJson.encode(
                            BrainDisplayFrame.object(root, "frame"), EventDisplayFrame.MAX_BYTES)));
            return new EventDisplayPresentation(status, age, frame);
        } catch (IllegalArgumentException exception) {
            throw new ProtocolException("invalid event display presentation", exception);
        }
    }

    /** A malformed synchronized display value cannot crash the HUD or affect motor control. */
    public static EventDisplayPresentation parseOrInvalid(String json) {
        if (json != null && json.isEmpty()) {
            return disabled();
        }
        try {
            return parse(json);
        } catch (ProtocolException | RuntimeException exception) {
            return new EventDisplayPresentation(EventDisplayState.Status.INVALID, 0, Optional.empty());
        }
    }
    public static EventDisplayPresentation disabled() {
        return new EventDisplayPresentation(EventDisplayState.Status.DISABLED, 0, Optional.empty());
    }
    public static EventDisplayPresentation unavailable() {
        return new EventDisplayPresentation(EventDisplayState.Status.UNAVAILABLE, 0, Optional.empty());
    }

}
