package dev.neurocraft.fly.body;

/**
 * The signed program inputs of v1: one declared turn bias in {@code [-1, 1]} (ADR 0020 section 4,
 * stage 4b design section 3.4).
 *
 * <p><strong>Why this is not a fifth {@link Modulation} field.</strong> A modulation scalar is an
 * attenuator by declaration — a multiplier in {@code [0, 1]} on a maximum the program itself
 * declares — and {@link Modulation#isNeutral()} is the switch that decides whether the HUD draws a
 * MODULATION line at all. A signed bias is a different kind of object: it selects a direction, not
 * an amount. Keeping it in its own record preserves the never-widen guarantee literally
 * ({@code |turnBias| <= 1} multiplies a maximum the program declares, inside the same actuator
 * rails) and keeps the modulation vocabulary honest.
 *
 * <p>{@link BodyProgram#step} takes it as a fourth argument, so a program can read it. Today
 * exactly one does: CRUISE adds {@code CRUISE_BIAS_TORQUE * turnBias} to its wander yaw term,
 * inside its own declared yaw maximum and behind the same avoidance gate the wander sits behind.
 * HOVER and ESCAPE_JUMP ignore it, and a bit-identity test pins that. <strong>What the HUD shows
 * is still what the mapping asked for, not a measurement of the body</strong>: whether the body
 * actually turned that way is what the world-space curvature pins measure.
 */
public record ProgramParameters(double turnBias) {
    /** No bias. What every non-readout lane runs under. */
    public static final ProgramParameters NEUTRAL = new ProgramParameters(0.0);

    public ProgramParameters {
        if (!Double.isFinite(turnBias)) {
            throw new IllegalArgumentException("turnBias must be finite");
        }
        // Negative zero is normalized here, not only at the encoder: a record's equals() uses
        // Double.compare, so an un-normalized -0.0 would make an otherwise neutral bias compare
        // unequal to NEUTRAL and publish a different channel word than the same value written 0.0.
        double bounded = Math.max(-1.0, Math.min(1.0, turnBias));
        turnBias = bounded == 0.0 ? 0.0 : bounded;
    }

    public boolean isNeutral() {
        return equals(NEUTRAL);
    }
}
