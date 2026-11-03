package dev.neurocraft.fly.control;

/**
 * Bounded primitive control channels. Values are dimensionless fractions of the declared actuator
 * maxima; this type contains no semantic movement operations.
 */
public record PrimitiveAction(
        double forwardThrust,
        double lateralThrust,
        double verticalThrust,
        double yawTorque,
        double pitchTorque) {
    public static final PrimitiveAction NEUTRAL = new PrimitiveAction(0.0, 0.0, 0.0, 0.0, 0.0);

    public PrimitiveAction {
        requireFinite(forwardThrust, "forwardThrust");
        requireFinite(lateralThrust, "lateralThrust");
        requireFinite(verticalThrust, "verticalThrust");
        requireFinite(yawTorque, "yawTorque");
        requireFinite(pitchTorque, "pitchTorque");
        requireUnitRange(forwardThrust, "forwardThrust");
        requireUnitRange(lateralThrust, "lateralThrust");
        requireUnitRange(verticalThrust, "verticalThrust");
        requireUnitRange(yawTorque, "yawTorque");
        requireUnitRange(pitchTorque, "pitchTorque");
    }

    public boolean isNeutral() {
        return equals(NEUTRAL);
    }

    public static PrimitiveAction clamped(
            double forward, double lateral, double vertical, double yaw, double pitch) {
        requireFinite(forward, "forward");
        requireFinite(lateral, "lateral");
        requireFinite(vertical, "vertical");
        requireFinite(yaw, "yaw");
        requireFinite(pitch, "pitch");
        return new PrimitiveAction(
                clampUnit(forward),
                clampUnit(lateral),
                clampUnit(vertical),
                clampUnit(yaw),
                clampUnit(pitch));
    }

    private static double clampUnit(double value) {
        return Math.max(-1.0, Math.min(1.0, value));
    }

    private static void requireFinite(double value, String name) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(name + " must be finite");
        }
    }

    private static void requireUnitRange(double value, String name) {
        if (value < -1.0 || value > 1.0) {
            throw new IllegalArgumentException(name + " must be in [-1, 1]");
        }
    }
}
