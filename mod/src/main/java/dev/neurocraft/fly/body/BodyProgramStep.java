package dev.neurocraft.fly.body;

import dev.neurocraft.fly.control.PrimitiveAction;
import java.util.Objects;
import java.util.Optional;

/**
 * One tick of a body program: the motors it asks for, the state it leaves behind, the phase it
 * publishes for presentation, and the successor it exits to (ADR 0020 section 1).
 *
 * <p>{@code motors} has already passed through {@link PrimitiveAction#clamped}, so a buggy program
 * is bounded by exactly the same rails the neural lane is, and then by the actuator's own speed
 * cap, yaw rate, and pitch clamp.
 *
 * <p>{@code exitTo} is the one place a program influences what runs next, and it can only name a
 * successor its own declared table allows. A program never selects <em>itself</em>.
 */
public record BodyProgramStep(
        PrimitiveAction motors,
        BodyProgramState next,
        float presentationPhase,
        Optional<BodyProgramId> exitTo) {
    public BodyProgramStep {
        Objects.requireNonNull(motors, "motors");
        Objects.requireNonNull(next, "next");
        Objects.requireNonNull(exitTo, "exitTo");
        if (!Float.isFinite(presentationPhase)
                || presentationPhase < 0.0F
                || presentationPhase >= 1.0F) {
            throw new IllegalArgumentException("presentationPhase must be in [0, 1)");
        }
    }
}
