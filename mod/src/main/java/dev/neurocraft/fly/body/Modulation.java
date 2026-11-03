package dev.neurocraft.fly.body;

/**
 * Four attenuating scalars a selector may apply to a running body program (ADR 0020 section 4).
 *
 * <p><strong>A modulation scalar can never widen a bound.</strong> Every value is a multiplier in
 * {@code [0, 1]} on a maximum the program itself declares, and that maximum already sits inside
 * the {@code PrimitiveActuator} rails. Modulation can only make the body <em>less</em> energetic
 * than the program's declared envelope, so nothing a future readout emits can push the body past
 * what the program says it may do. The clamp in the compact constructor is the whole of that
 * guarantee.
 *
 * <p>The neutral default is {@code speed = 1, turnRate = 1, amplitude = 1, urgency = 0} — "the
 * program exactly as written". That is deliberate rather than a mid-range default: the operator
 * lane must be able to run a program at its written form so that a stage-4 readout's effect is
 * measurable as a departure from the operator's baseline.
 *
 * <p>A program declares which scalars it reads; varying an unread scalar produces a bit-identical
 * motor sequence, and a per-program test pins that.
 */
public record Modulation(double speed, double turnRate, double amplitude, double urgency) {
    /** The program exactly as written. */
    public static final Modulation NEUTRAL = new Modulation(1.0, 1.0, 1.0, 0.0);

    public Modulation {
        speed = clampUnitInterval(speed, "speed");
        turnRate = clampUnitInterval(turnRate, "turnRate");
        amplitude = clampUnitInterval(amplitude, "amplitude");
        urgency = clampUnitInterval(urgency, "urgency");
    }

    public boolean isNeutral() {
        return equals(NEUTRAL);
    }

    private static double clampUnitInterval(double value, String name) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(name + " must be finite");
        }
        return Math.max(0.0, Math.min(1.0, value));
    }
}
