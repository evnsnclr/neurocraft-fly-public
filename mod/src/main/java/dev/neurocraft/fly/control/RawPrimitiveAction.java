package dev.neurocraft.fly.control;

/** Finite canonical wire values retained separately from server-side actuator application. */
public record RawPrimitiveAction(
        double forwardThrust,
        double lateralThrust,
        double verticalThrust,
        double yawTorque,
        double pitchTorque) {
    public static final RawPrimitiveAction ZERO = new RawPrimitiveAction(0.0, 0.0, 0.0, 0.0, 0.0);

    public RawPrimitiveAction {
        requireFinite(forwardThrust, "forwardThrust");
        requireFinite(lateralThrust, "lateralThrust");
        requireFinite(verticalThrust, "verticalThrust");
        requireFinite(yawTorque, "yawTorque");
        requireFinite(pitchTorque, "pitchTorque");
    }

    public PrimitiveAction clamped() {
        return PrimitiveAction.clamped(
                forwardThrust, lateralThrust, verticalThrust, yawTorque, pitchTorque);
    }

    private static void requireFinite(double value, String name) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(name + " must be finite");
        }
    }
}
