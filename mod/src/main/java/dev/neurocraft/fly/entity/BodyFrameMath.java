package dev.neurocraft.fly.entity;

/** Pure coordinate conversion for primitive body-frame thrust. */
public final class BodyFrameMath {
    private BodyFrameMath() {}

    public static HorizontalVector toWorld(
            double yawDegrees, double forwardMagnitude, double rightMagnitude) {
        requireFinite(yawDegrees, "yawDegrees");
        requireFinite(forwardMagnitude, "forwardMagnitude");
        requireFinite(rightMagnitude, "rightMagnitude");
        // Body frame, ADR 0015. Matches FlySensors: forward = (-sin, cos) is south at yaw
        // 0, and the body's right is forward x up = (-cos, -sin), which is west at yaw 0
        // and north at yaw 90 -- the engine's own Direction.getClockWise() table. The
        // former (cos, sin) was east at yaw 0, i.e. the body's left.
        double radians = Math.toRadians(yawDegrees);
        double forwardX = -Math.sin(radians);
        double forwardZ = Math.cos(radians);
        double rightX = -Math.cos(radians);
        double rightZ = -Math.sin(radians);
        return new HorizontalVector(
                forwardX * forwardMagnitude + rightX * rightMagnitude,
                forwardZ * forwardMagnitude + rightZ * rightMagnitude);
    }

    /**
     * The exact inverse of {@link #toWorld}: resolves a world horizontal vector onto the same
     * ADR 0015 body basis. The basis is orthonormal, so the projection is a pair of dot products
     * and {@code toBody(yaw, toWorld(yaw, f, r))} returns {@code (f, r)}.
     *
     * <p>This exists so a body program can oppose a world-frame drift without restating the
     * handedness. ADR 0015 exists because that sign is easy to get wrong; there must be exactly
     * one statement of it.
     */
    public static BodyVector toBody(double yawDegrees, double worldX, double worldZ) {
        requireFinite(yawDegrees, "yawDegrees");
        requireFinite(worldX, "worldX");
        requireFinite(worldZ, "worldZ");
        double radians = Math.toRadians(yawDegrees);
        double forwardX = -Math.sin(radians);
        double forwardZ = Math.cos(radians);
        double rightX = -Math.cos(radians);
        double rightZ = -Math.sin(radians);
        return new BodyVector(
                worldX * forwardX + worldZ * forwardZ, worldX * rightX + worldZ * rightZ);
    }

    private static void requireFinite(double value, String name) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(name + " must be finite");
        }
    }

    public record HorizontalVector(double x, double z) {}

    /** A horizontal vector resolved onto the body's own forward and right axes (ADR 0015). */
    public record BodyVector(double forward, double right) {}
}
