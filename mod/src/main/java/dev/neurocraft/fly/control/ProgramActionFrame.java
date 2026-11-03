package dev.neurocraft.fly.control;

import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ProgramParameters;
import dev.neurocraft.fly.body.ReadoutValues;
import java.util.Objects;

/**
 * Strictly decoded body-program selection, in the same causal envelope as an {@link ActionFrame}.
 *
 * <p><strong>Nothing on this frame is anatomy.</strong> {@code program} names a scripted body
 * program (ADR 0020), {@code modulation} four declared presentation attenuators, {@code parameters}
 * one declared signed input, {@code readout} the labeled readout scalars a hand-chosen mapping
 * consumed, and {@code policyHash} the identity of that mapping (ADR 0019). The frame carries the
 * policy identity precisely so no surface can present the selection as the connectome deciding
 * anything: the deciding function is declared, identified, and recomputable.
 *
 * <p>Two differences from {@link ActionFrame}. There are no motors and no neutral flag — a program
 * frame requests no primitive motors, and a paused, replaying, or faulted service keeps emitting
 * neutral {@code action} frames. And the neural interval is <strong>strictly</strong> non-empty: a
 * program frame is never neutral, so the connectome stepped. Every other envelope rule is the same
 * rule, written the same way.
 */
public record ProgramActionFrame(
        ControllerIdentity identity,
        long commandSequence,
        long observationSequence,
        long observationTick,
        long targetTick,
        long validThroughTick,
        long neuralStepStart,
        long neuralStepEnd,
        String stateDigest,
        BodyProgramId program,
        Modulation modulation,
        ProgramParameters parameters,
        ReadoutValues readout,
        String policyHash) implements CommandFrame, InboundFrame {
    public ProgramActionFrame {
        Objects.requireNonNull(identity, "identity");
        Objects.requireNonNull(stateDigest, "stateDigest");
        Objects.requireNonNull(program, "program");
        Objects.requireNonNull(modulation, "modulation");
        Objects.requireNonNull(parameters, "parameters");
        Objects.requireNonNull(readout, "readout");
        Objects.requireNonNull(policyHash, "policyHash");
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
        if (neuralStepEnd <= neuralStepStart) {
            throw new IllegalArgumentException(
                    "a program frame is never neutral, so its neural interval must be non-empty");
        }
        if (!stateDigest.matches("[0-9a-f]{64}")) {
            throw new IllegalArgumentException("state digest must be lowercase SHA-256");
        }
        if (!policyHash.matches("[0-9a-f]{64}")) {
            throw new IllegalArgumentException("policy hash must be lowercase SHA-256");
        }
    }
}
