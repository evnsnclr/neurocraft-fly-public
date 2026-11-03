package dev.neurocraft.fly.control;

import java.util.Objects;

/** Strictly decoded acknowledgement that causally binds one outbound operator control request. */
public record ControlAckFrame(
        ControllerIdentity identity,
        long controlSequence,
        long issuedTick,
        ControlOperation operation,
        String contentId,
        String contentHash,
        boolean accepted,
        ControlState state,
        long effectiveTick,
        String faultCode) implements InboundFrame {
    public ControlAckFrame {
        Objects.requireNonNull(identity, "identity");
        Objects.requireNonNull(operation, "operation");
        Objects.requireNonNull(state, "state");
        if (controlSequence < 0 || issuedTick < 0 || effectiveTick < 0) {
            throw new IllegalArgumentException("control sequence and ticks must be non-negative");
        }
        long tickDelta = effectiveTick - issuedTick;
        if (accepted && operation == ControlOperation.RESET) {
            // An accepted RESET is stamped with the governed action's target_tick, so its
            // delay is the action budget, or one more when the effect materialized against
            // the following observation. Permissive from 1 so acknowledgements recorded
            // under the one-tick budget still decode; StagedReset holds the exact pair.
            if (tickDelta < 1
                    || tickDelta > CommandValidator.ACTION_TARGET_OFFSET_TICKS + 1) {
                throw new IllegalArgumentException(
                        "accepted RESET acknowledgement is outside its tick window");
            }
        } else if (tickDelta < 0 || tickDelta > 1) {
            throw new IllegalArgumentException("control acknowledgement is outside its tick window");
        }
        if (accepted != (faultCode == null)) {
            throw new IllegalArgumentException("accepted state and fault code are inconsistent");
        }
    }

    public boolean binds(ControlCommandFrame command) {
        return identity.equals(command.identity())
                && controlSequence == command.controlSequence()
                && issuedTick == command.issuedTick()
                && operation == command.operation()
                && Objects.equals(contentId, command.contentId())
                && Objects.equals(contentHash, command.contentHash());
    }
}
