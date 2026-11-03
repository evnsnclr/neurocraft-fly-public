package dev.neurocraft.fly.entity;

import dev.neurocraft.fly.control.PrimitiveAction;
import dev.neurocraft.fly.control.RawPrimitiveAction;
import java.util.Objects;

/** Raw/applied primitive values and authoritative before/after kinematics for trace handoff. */
public record ActuationReceipt(
        RawPrimitiveAction rawAction,
        PrimitiveAction appliedAction,
        double velocityBeforeX,
        double velocityBeforeY,
        double velocityBeforeZ,
        double velocityAfterX,
        double velocityAfterY,
        double velocityAfterZ,
        float yawBeforeDegrees,
        float yawAfterDegrees,
        float pitchBeforeDegrees,
        float pitchAfterDegrees) {
    public ActuationReceipt {
        Objects.requireNonNull(rawAction, "rawAction");
        Objects.requireNonNull(appliedAction, "appliedAction");
    }
}
