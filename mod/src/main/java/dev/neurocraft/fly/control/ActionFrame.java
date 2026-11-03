package dev.neurocraft.fly.control;

import java.util.Objects;

/**
 * Strictly decoded primitive-motor command envelope. Final validation still occurs on the
 * Minecraft server thread.
 *
 * <p>One of the two {@link CommandFrame} kinds; the other, {@link ProgramActionFrame}, requests a
 * scripted body program instead of motors. A session is driven by one kind or the other, never
 * both, and the validator latches which.
 */
public record ActionFrame(
        ControllerIdentity identity,
        long commandSequence,
        long observationSequence,
        long observationTick,
        long targetTick,
        long validThroughTick,
        long neuralStepStart,
        long neuralStepEnd,
        String stateDigest,
        RawPrimitiveAction rawAction,
        boolean neutral) implements CommandFrame, InboundFrame {
    public ActionFrame {
        Objects.requireNonNull(identity, "identity");
        Objects.requireNonNull(stateDigest, "stateDigest");
        Objects.requireNonNull(rawAction, "rawAction");
        if (observationSequence < 0
                || commandSequence < 0
                || observationTick < 0
                || targetTick < 0
                || validThroughTick < 0
                || neuralStepStart < 0
                || neuralStepEnd < 0) {
            throw new IllegalArgumentException("sequence and clock fields must be non-negative");
        }
        if (targetTick < observationTick) {
            throw new IllegalArgumentException("target tick precedes observation tick");
        }
        if (validThroughTick < targetTick
                || validThroughTick - observationTick
                        > CommandValidator.ACTION_TARGET_OFFSET_TICKS) {
            // The budget ceiling, and deliberately only the ceiling: a frame produced under
            // the one-tick budget (target = observation + 1) still decodes, so committed
            // goldens and historical frames parse. The exact offset is enforced where frames
            // arrive from the wire (NeuralBridge) rather than in the frame model.
            throw new IllegalArgumentException("action validity exceeds the declared budget");
        }
        if (neuralStepEnd < neuralStepStart || (!neutral && neuralStepEnd == neuralStepStart)) {
            throw new IllegalArgumentException(
                    "neural interval may be empty only for a neutral action");
        }
        if (!stateDigest.matches("[0-9a-f]{64}")) {
            throw new IllegalArgumentException("state digest must be lowercase SHA-256");
        }
        if (neutral && !rawAction.equals(RawPrimitiveAction.ZERO)) {
            throw new IllegalArgumentException("neutral action must have zero motors");
        }
    }
}
