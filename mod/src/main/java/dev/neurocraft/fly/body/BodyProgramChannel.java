package dev.neurocraft.fly.body;

import java.util.Objects;

/**
 * The single synchronized presentation word describing what the body is doing and who chose it
 * (ADR 0020 sections 6 and 8).
 *
 * <p>The entity publishes exactly two new synchronized fields beside the five motor ones: this
 * packed word and a normalized program phase. Packing the selection source, the program id, and
 * the modulation into one {@code int} keeps the entity's synchronized surface at two new
 * accessors rather than six.
 *
 * <p><strong>The published modulation is a presentation rendering, not the authoritative value.</strong>
 * Each scalar is quantized to {@link #PUBLISHED_MODULATION_STEPS} tenths for the wire, so the HUD's
 * {@code MODULATION:} line shows the modulation to one decimal place. The authoritative
 * {@link Modulation} the runtime applies is the exact value the selector supplied, and
 * {@code /neurocraftfly status} reports it exactly. This follows the ADR 0016 / D-074 discipline:
 * a presentation transform is declared where it is applied.
 *
 * <p>Bit layout, all fields non-negative so the packed word is always a positive {@code int}:
 *
 * <pre>
 *   bits  0-3   program ordinal        (0 .. 9)
 *   bits  4-5   selection source       (0 .. 2)
 *   bits  6-9   speed level            (0 .. 10)
 *   bits 10-13  turn-rate level        (0 .. 10)
 *   bits 14-17  amplitude level        (0 .. 10)
 *   bits 18-21  urgency level          (0 .. 10)
 *   bits 22-26  turn-bias level        (0 .. 20)
 * </pre>
 *
 * <p>The turn bias is signed, so its level is {@code round(turnBias * 10) + 10}: five bits, still a
 * non-negative field, and still one decimal on the HUD. It is a declared program input the readout
 * lane supplies, not something the body measured.
 */
public record BodyProgramChannel(
        SelectionSource source,
        BodyProgramId program,
        Modulation modulation,
        ProgramParameters parameters) {
    /** Quantization levels used on the wire: tenths, so a published value has one decimal. */
    public static final int PUBLISHED_MODULATION_STEPS = 10;

    /** What the entity publishes at spawn, after a fixture reset, and after any release. */
    public static final BodyProgramChannel DEFAULT = new BodyProgramChannel(
            SelectionSource.DEFAULT,
            BodyProgramId.HOVER,
            Modulation.NEUTRAL,
            ProgramParameters.NEUTRAL);

    private static final int PROGRAM_BITS = 4;
    private static final int SOURCE_BITS = 2;
    private static final int LEVEL_BITS = 4;
    private static final int PROGRAM_SHIFT = 0;
    private static final int SOURCE_SHIFT = PROGRAM_SHIFT + PROGRAM_BITS;
    private static final int SPEED_SHIFT = SOURCE_SHIFT + SOURCE_BITS;
    private static final int TURN_SHIFT = SPEED_SHIFT + LEVEL_BITS;
    private static final int AMPLITUDE_SHIFT = TURN_SHIFT + LEVEL_BITS;
    private static final int URGENCY_SHIFT = AMPLITUDE_SHIFT + LEVEL_BITS;
    private static final int BIAS_BITS = 5;
    private static final int BIAS_SHIFT = URGENCY_SHIFT + LEVEL_BITS;
    private static final int BIAS_MASK = (1 << BIAS_BITS) - 1;
    private static final int BIAS_OFFSET = PUBLISHED_MODULATION_STEPS;
    private static final int MAX_BIAS_LEVEL = 2 * PUBLISHED_MODULATION_STEPS;
    private static final int PROGRAM_MASK = (1 << PROGRAM_BITS) - 1;
    private static final int SOURCE_MASK = (1 << SOURCE_BITS) - 1;
    private static final int LEVEL_MASK = (1 << LEVEL_BITS) - 1;

    public BodyProgramChannel {
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(program, "program");
        Objects.requireNonNull(modulation, "modulation");
        Objects.requireNonNull(parameters, "parameters");
    }

    /** The synchronized word. Always positive; every field round-trips through {@link #decode}. */
    public int packed() {
        return (program.ordinal() << PROGRAM_SHIFT)
                | (source.ordinal() << SOURCE_SHIFT)
                | (level(modulation.speed()) << SPEED_SHIFT)
                | (level(modulation.turnRate()) << TURN_SHIFT)
                | (level(modulation.amplitude()) << AMPLITUDE_SHIFT)
                | (level(modulation.urgency()) << URGENCY_SHIFT)
                | (biasLevel(parameters.turnBias()) << BIAS_SHIFT);
    }

    /**
     * Reads a synchronized word. An unrecognized program or source ordinal — the only way a
     * malformed word can arrive is a desynchronized or hostile client packet — degrades to the
     * default rather than throwing inside the render loop.
     */
    public static BodyProgramChannel decode(int packed) {
        BodyProgramId[] programs = BodyProgramId.values();
        SelectionSource[] sources = SelectionSource.values();
        int programOrdinal = (packed >>> PROGRAM_SHIFT) & PROGRAM_MASK;
        int sourceOrdinal = (packed >>> SOURCE_SHIFT) & SOURCE_MASK;
        if (packed < 0 || programOrdinal >= programs.length || sourceOrdinal >= sources.length) {
            return DEFAULT;
        }
        return new BodyProgramChannel(
                sources[sourceOrdinal],
                programs[programOrdinal],
                new Modulation(
                        value((packed >>> SPEED_SHIFT) & LEVEL_MASK),
                        value((packed >>> TURN_SHIFT) & LEVEL_MASK),
                        value((packed >>> AMPLITUDE_SHIFT) & LEVEL_MASK),
                        value((packed >>> URGENCY_SHIFT) & LEVEL_MASK)),
                new ProgramParameters(biasValue((packed >>> BIAS_SHIFT) & BIAS_MASK)));
    }

    private static int level(double scalar) {
        long rounded = Math.round(scalar * PUBLISHED_MODULATION_STEPS);
        return (int) Math.max(0L, Math.min(PUBLISHED_MODULATION_STEPS, rounded));
    }

    private static double value(int level) {
        return Math.min(PUBLISHED_MODULATION_STEPS, level) / (double) PUBLISHED_MODULATION_STEPS;
    }

    /**
     * Symmetric rounding, unlike {@link #level}'s. {@link Math#round} rounds a half toward positive
     * infinity, so a raw bias of -0.25 published as -0.2 while +0.25 published as +0.3 -- a signed
     * channel whose two directions did not agree. Rounding the magnitude and restoring the sign
     * makes the published bias the same size either way.
     */
    private static int biasLevel(double scalar) {
        long magnitude = Math.round(Math.abs(scalar) * PUBLISHED_MODULATION_STEPS);
        long rounded = (scalar < 0.0 ? -magnitude : magnitude) + BIAS_OFFSET;
        return (int) Math.max(0L, Math.min(MAX_BIAS_LEVEL, rounded));
    }

    private static double biasValue(int level) {
        return (Math.min(MAX_BIAS_LEVEL, level) - BIAS_OFFSET)
                / (double) PUBLISHED_MODULATION_STEPS;
    }
}
