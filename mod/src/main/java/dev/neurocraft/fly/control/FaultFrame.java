package dev.neurocraft.fly.control;

import java.util.Objects;

/** Strictly decoded fail-closed fault emitted by the neural service. */
public record FaultFrame(
        ControllerIdentity identity,
        String code,
        String message,
        long gameTick,
        Long relatedObservationSequence) implements InboundFrame {
    public FaultFrame {
        Objects.requireNonNull(identity, "identity");
        Objects.requireNonNull(code, "code");
        Objects.requireNonNull(message, "message");
        if (gameTick < 0 || (relatedObservationSequence != null && relatedObservationSequence < 0)) {
            throw new IllegalArgumentException("fault clocks and sequences must be non-negative");
        }
    }
}
