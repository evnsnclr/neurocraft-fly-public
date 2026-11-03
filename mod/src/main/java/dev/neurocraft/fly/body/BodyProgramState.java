package dev.neurocraft.fly.body;

import java.util.Objects;

/**
 * Immutable per-tick state of the running body program (ADR 0020 section 1).
 *
 * <p>Replaced each tick, never mutated. {@code ticksInProgram} counts authoritative game ticks the
 * program has owned since it was entered; no wall clock and no random source reaches this package,
 * which is what makes a program's motor sequence a pure function of its inputs.
 */
public record BodyProgramState(
        BodyProgramId id, long enteredGameTick, int ticksInProgram, int phaseIndex, long wanderSeed) {
    public BodyProgramState {
        Objects.requireNonNull(id, "id");
        if (enteredGameTick < 0) {
            throw new IllegalArgumentException("enteredGameTick must be non-negative");
        }
        if (wanderSeed < 0) {
            throw new IllegalArgumentException("wanderSeed must be non-negative");
        }
        if (ticksInProgram < 0 || phaseIndex < 0) {
            throw new IllegalArgumentException("tick and phase counters must be non-negative");
        }
    }

    /** Ordinary programs retain the historical entry-tick phase seed exactly. */
    public BodyProgramState(
            BodyProgramId id, long enteredGameTick, int ticksInProgram, int phaseIndex) {
        this(id, enteredGameTick, ticksInProgram, phaseIndex, enteredGameTick);
    }

    /** First tick of a newly selected program. */
    public static BodyProgramState entering(BodyProgramId id, long gameTick) {
        return new BodyProgramState(id, gameTick, 0, 0);
    }

    /** Explicit trial seed, independent of the actual world tick at entry. */
    public static BodyProgramState entering(BodyProgramId id, long gameTick, long wanderSeed) {
        return new BodyProgramState(id, gameTick, 0, 0, wanderSeed);
    }

    /** The successor state after one owned tick, carrying the program's own phase index. */
    public BodyProgramState advanced(int nextPhaseIndex) {
        return new BodyProgramState(id, enteredGameTick, ticksInProgram + 1, nextPhaseIndex, wanderSeed);
    }
}
