package dev.neurocraft.fly.body;

/**
 * A body program is a pure function (ADR 0020 section 1).
 *
 * <p>Implementations hold no field, no clock, and no random source. {@link #step} is a function of
 * {@code (state, context, modulation, parameters)} alone, so determinism is structural rather than
 * tested in, and {@code BodyPackageBoundaryTest} pins the absence of every non-deterministic source
 * in this package.
 *
 * <p>{@link #canEnter} is a <em>precondition</em>, never a trigger: it can only refuse a selection
 * someone else made. A program that decided <em>when</em> to run would be the vanilla-mob-AI-shaped
 * thing {@code projectOwnedEntityHasNoMobOrNavigationInheritance} exists to forbid, wearing a
 * different name.
 *
 * <p><strong>Two selector inputs, not one, and they are different kinds of object.</strong>
 * {@link Modulation} is four attenuators in {@code [0, 1]} that can only make the body less
 * energetic than the program's declared envelope; {@link ProgramParameters} is one signed value in
 * {@code [-1, 1]} that selects a <em>direction</em> inside a maximum the program declares. A
 * program states which of either it reads, and a per-program test pins that varying an unread one
 * produces a bit-identical motor sequence.
 */
public interface BodyProgram {
    /** The identifier this program answers to; it must match its registry key. */
    BodyProgramId id();

    /** Whether this program may be entered in this world context. Never a trigger. */
    boolean canEnter(BodyProgramContext context);

    /** One deterministic tick. Must return clamped motors and a phase in {@code [0, 1)}. */
    BodyProgramStep step(
            BodyProgramState state,
            BodyProgramContext context,
            Modulation modulation,
            ProgramParameters parameters);
}
