package dev.neurocraft.fly.body;

import dev.neurocraft.fly.control.PrimitiveAction;
import java.util.Objects;
import java.util.Optional;

/**
 * What actually ran on one readout-lane tick, and the motors it actually emitted.
 *
 * <p>This is the receipt's whole source of truth. {@code executedProgram} is never the requested
 * program by assumption: it is whatever the runtime resolved and stepped, which is the requested
 * program on the normal path, HOVER after an {@code unregistered} or {@code precondition} refusal,
 * and the operator lane's own program when an operator hold owned the tick. {@code emittedMotors}
 * are the clamped motors that program returned for this tick — the ones the actuator applied — so
 * the receipt can never claim motion the world did not see.
 *
 * @param executedProgram the program the runtime actually stepped
 * @param executedModulation the modulation it actually ran under
 * @param executedParameters the signed parameters it actually ran under
 * @param emittedMotors the clamped motors it emitted for this tick
 * @param refusalReason empty when the requested program ran as asked
 */
public record ProgramExecution(
        BodyProgramId executedProgram,
        Modulation executedModulation,
        ProgramParameters executedParameters,
        PrimitiveAction emittedMotors,
        Optional<ProgramRefusalReason> refusalReason) {
    public ProgramExecution {
        Objects.requireNonNull(executedProgram, "executedProgram");
        Objects.requireNonNull(executedModulation, "executedModulation");
        Objects.requireNonNull(executedParameters, "executedParameters");
        Objects.requireNonNull(emittedMotors, "emittedMotors");
        Objects.requireNonNull(refusalReason, "refusalReason");
    }

    /** Whether the requested program ran exactly as requested. */
    public boolean applied() {
        return refusalReason.isEmpty();
    }
}
