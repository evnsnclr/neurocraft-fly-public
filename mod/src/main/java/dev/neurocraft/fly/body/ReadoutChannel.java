package dev.neurocraft.fly.body;

import java.util.Objects;

/**
 * The synchronized presentation word carrying the three labeled readout scalars that a declared
 * hand-chosen mapping consumed to pick the running body program.
 *
 * <p>One {@code int} rather than three floats, for the same reason {@link BodyProgramChannel} packs
 * its fields: the entity's synchronized surface grows by one accessor, not three.
 *
 * <p><strong>The published values are a presentation rendering, not the authoritative ones.</strong>
 * Each scalar is quantized to hundredths for the wire, so the HUD shows two decimals; the exact
 * values the frame carried are reported by {@code /neurocraftfly status} to six. This is the
 * ADR 0016 / D-074 discipline: a presentation transform is declared where it is applied.
 *
 * <p>Bit layout, all fields non-negative so the packed word is always a positive {@code int}. Each
 * level is {@code round(v * 100) + 100}, which is {@code [0, 200]} for a value in {@code [-1, 1]}
 * and {@code [100, 200]} for one in {@code [0, 1]}:
 *
 * <pre>
 *   bits  0-7   TTMn jump level        (100 .. 200)
 *   bits  8-15  DNa01 turn level       (0 .. 200)
 *   bits 16-23  wing common level      (0 .. 200)
 * </pre>
 */
public record ReadoutChannel(ReadoutValues readout) {
    /** Quantization levels per unit used on the wire: hundredths, so a published value has two. */
    public static final int PUBLISHED_READOUT_STEPS = 100;

    /** What the entity publishes whenever no readout is driving the body. */
    public static final ReadoutChannel DEFAULT = new ReadoutChannel(ReadoutValues.ZERO);

    private static final int LEVEL_BITS = 8;
    private static final int LEVEL_MASK = (1 << LEVEL_BITS) - 1;
    private static final int TTMN_SHIFT = 0;
    private static final int DNA01_SHIFT = TTMN_SHIFT + LEVEL_BITS;
    private static final int WING_SHIFT = DNA01_SHIFT + LEVEL_BITS;
    private static final int LEVEL_OFFSET = PUBLISHED_READOUT_STEPS;
    private static final int MAX_LEVEL = 2 * PUBLISHED_READOUT_STEPS;

    public ReadoutChannel {
        Objects.requireNonNull(readout, "readout");
    }

    /** The synchronized word. Always positive; every field round-trips through {@link #decode}. */
    public int packed() {
        return (level(readout.ttmnJump()) << TTMN_SHIFT)
                | (level(readout.dna01Turn()) << DNA01_SHIFT)
                | (level(readout.wingCommon()) << WING_SHIFT);
    }

    /**
     * Reads a synchronized word. A level outside the written range — the only way one can arrive is
     * a desynchronized or hostile client packet — is clamped rather than thrown, because this is
     * read inside the render loop.
     */
    public static ReadoutChannel decode(int packed) {
        if (packed < 0) {
            return DEFAULT;
        }
        return new ReadoutChannel(new ReadoutValues(
                value((packed >>> TTMN_SHIFT) & LEVEL_MASK),
                value((packed >>> DNA01_SHIFT) & LEVEL_MASK),
                value((packed >>> WING_SHIFT) & LEVEL_MASK)));
    }

    /**
     * Symmetric rounding: {@link Math#round} rounds a half toward positive infinity, which would
     * publish -0.005 as 0.00 and +0.005 as +0.01. Two signed readouts travel through here, so the
     * magnitude is rounded and the sign restored, and a left bias renders the same size as the
     * mirrored right one.
     */
    private static int level(double scalar) {
        long magnitude = Math.round(Math.abs(scalar) * PUBLISHED_READOUT_STEPS);
        long rounded = (scalar < 0.0 ? -magnitude : magnitude) + LEVEL_OFFSET;
        return (int) Math.max(0L, Math.min(MAX_LEVEL, rounded));
    }

    private static double value(int level) {
        return (Math.min(MAX_LEVEL, level) - LEVEL_OFFSET) / (double) PUBLISHED_READOUT_STEPS;
    }
}
