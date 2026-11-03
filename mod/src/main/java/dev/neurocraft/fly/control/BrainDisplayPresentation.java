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
public record BrainDisplayPresentation(
        BrainDisplayState.Status status,
        long ageMillis,
        Optional<BrainDisplayFrame> frame) {
    public static final String SCHEMA = "neurocraft.brain_display_presentation.v1";
    public static final int MAX_BYTES = BrainDisplayFrame.MAX_BYTES;

    public BrainDisplayPresentation {
        Objects.requireNonNull(status, "status");
        Objects.requireNonNull(frame, "frame");
        if (ageMillis < 0 || (status == BrainDisplayState.Status.MATCHED) != frame.isPresent()) {
            throw new IllegalArgumentException("only MATCHED presentations carry a frame; age >= 0");
        }
    }

    public static BrainDisplayPresentation fromView(BrainDisplayState.View view) {
        return new BrainDisplayPresentation(view.status(), Math.max(0, view.ageNanos() / 1_000_000),
                view.frame());
    }

    public String encode() {
        Map<String, Object> values = new LinkedHashMap<>();
        values.put("schema", SCHEMA);
        values.put("status", status.name());
        values.put("age_ms", ageMillis);
        values.put("frame", frame.map(BrainDisplayFrame::toMap).orElse(null));
        return StrictJson.encode(values, MAX_BYTES);
    }

    public static BrainDisplayPresentation parse(String json) throws ProtocolException {
        Map<String, Object> root = StrictJson.parseObject(json, MAX_BYTES);
        BrainDisplayFrame.exact(root, Set.of("schema", "status", "age_ms", "frame"));
        if (!SCHEMA.equals(root.get("schema"))) {
            throw new ProtocolException("unsupported brain display presentation schema");
        }
        try {
            BrainDisplayState.Status status = BrainDisplayState.Status.valueOf(
                    BrainDisplayFrame.text(root, "status"));
            long age = BrainDisplayFrame.integer(root, "age_ms");
            Optional<BrainDisplayFrame> frame = root.get("frame") == null ? Optional.empty()
                    : Optional.of(BrainDisplayFrame.parse(StrictJson.encode(
                            BrainDisplayFrame.object(root, "frame"), BrainDisplayFrame.MAX_BYTES)));
            return new BrainDisplayPresentation(status, age, frame);
        } catch (IllegalArgumentException exception) {
            throw new ProtocolException("invalid brain display presentation", exception);
        }
    }

    /** A malformed synchronized display value cannot crash the HUD or affect motor control. */
    public static BrainDisplayPresentation parseOrInvalid(String json) {
        try {
            return parse(json);
        } catch (ProtocolException | RuntimeException exception) {
            return new BrainDisplayPresentation(BrainDisplayState.Status.INVALID, 0, Optional.empty());
        }
    }
}
