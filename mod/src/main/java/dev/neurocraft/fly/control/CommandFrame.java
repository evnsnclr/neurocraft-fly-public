package dev.neurocraft.fly.control;

/**
 * The causal envelope every driving command carries, whatever it drives.
 *
 * <p>Two frame kinds implement it: {@link ActionFrame}, which requests primitive motors, and
 * {@link ProgramActionFrame}, which requests a scripted body program selected by a declared
 * hand-chosen readout mapping. The nine accessors below are exactly the fields
 * {@link CommandValidator} reads, so every admission rule — identity, sequence continuity,
 * observation receipt, freshness, neural-step continuity, target ordering, window size — is
 * written once and applies identically to both kinds.
 *
 * <p><strong>One driving kind per session.</strong> The validator latches the kind the first
 * non-neutral frame declares and refuses the other for the life of the session; see
 * {@code CommandValidator.FrameLane}. A mixed session would leave no honest answer to "what was
 * driving the fly at tick T", which is the whole reason this interface is sealed rather than open.
 */
public sealed interface CommandFrame permits ActionFrame, ProgramActionFrame {
    ControllerIdentity identity();

    long commandSequence();

    long observationSequence();

    long observationTick();

    long targetTick();

    long validThroughTick();

    long neuralStepStart();

    long neuralStepEnd();

    String stateDigest();
}
