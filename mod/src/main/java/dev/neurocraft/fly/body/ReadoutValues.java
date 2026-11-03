package dev.neurocraft.fly.body;

/**
 * The exact labeled readout scalars a hand-chosen selection mapping consumed on one tick.
 *
 * <p><strong>These are readout values, not decisions.</strong> {@code ttmnJump} is a rectified
 * motor-stage term, {@code dna01Turn} a normalized descending left/right difference, and
 * {@code wingCommon} the projection's common wing term. The mapping that turns them into a program
 * is engineering (ADR 0019), it lives in the neural service, and its identity travels on the same
 * frame. Nothing in this record says the connectome chose anything.
 *
 * <p>The mod never computes these. They arrive on a {@code program_action} frame, are published for
 * the HUD, and are reported by {@code /neurocraftfly status}; the sensory vector the encoder
 * consumes is untouched, so {@code observation_schema_hash} cannot move because of them.
 */
public record ReadoutValues(double ttmnJump, double dna01Turn, double wingCommon) {
    /** What a body that no readout is driving reports. */
    public static final ReadoutValues ZERO = new ReadoutValues(0.0, 0.0, 0.0);

    public ReadoutValues {
        ttmnJump = bounded(ttmnJump, 0.0, "ttmnJump");
        dna01Turn = bounded(dna01Turn, -1.0, "dna01Turn");
        wingCommon = bounded(wingCommon, -1.0, "wingCommon");
    }

    private static double bounded(double value, double minimum, String name) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(name + " must be finite");
        }
        double clamped = Math.max(minimum, Math.min(1.0, value));
        return clamped == 0.0 ? 0.0 : clamped;
    }
}
