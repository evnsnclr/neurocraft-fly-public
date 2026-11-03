package dev.neurocraft.fly.entity;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

final class BodyFrameMathTest {
    private static final double EPSILON = 1.0e-12;

    /**
     * ADR 0015. At yaw 0 the body faces south, {@code (0, 0, +1)}. Its right is a 90 degree
     * clockwise turn from there -- {@code Direction.SOUTH.getClockWise()} is {@code WEST} --
     * so unit lateral thrust must push west, {@code (-1, 0)}. This assertion previously read
     * {@code (+1, 0)} east, the body's left, and was the unit-test half of the swapped
     * lateral convention.
     */
    @Test
    void yawZeroMapsForwardToSouthAndRightToWest() {
        BodyFrameMath.HorizontalVector forward = BodyFrameMath.toWorld(0, 1, 0);
        BodyFrameMath.HorizontalVector right = BodyFrameMath.toWorld(0, 0, 1);

        assertEquals(0, forward.x(), EPSILON);
        assertEquals(1, forward.z(), EPSILON);
        assertEquals(-1, right.x(), EPSILON);
        assertEquals(0, right.z(), EPSILON);
    }

    /** At yaw 90 the body faces west; {@code WEST.getClockWise()} is {@code NORTH}. */
    @Test
    void yawNinetyMapsForwardToWestAndRightToNorth() {
        BodyFrameMath.HorizontalVector forward = BodyFrameMath.toWorld(90, 1, 0);
        BodyFrameMath.HorizontalVector right = BodyFrameMath.toWorld(90, 0, 1);

        assertEquals(-1, forward.x(), EPSILON);
        assertEquals(0, forward.z(), EPSILON);
        assertEquals(0, right.x(), EPSILON);
        assertEquals(-1, right.z(), EPSILON);
    }

    /**
     * The lateral axis is {@code forward x up} at every yaw, not only at the two pinned
     * ones. Restricted to the horizontal plane that cross product is exactly
     * {@code (-forward.z, forward.x)}, so a future edit cannot satisfy the two cases above
     * by hand while leaving the frame wrong in between -- and the sign here is the whole
     * defect, since {@code up x forward} is the negation.
     */
    @Test
    void lateralAxisIsForwardCrossUpAtEveryYaw() {
        for (int yaw = -180; yaw <= 180; yaw += 15) {
            BodyFrameMath.HorizontalVector forward = BodyFrameMath.toWorld(yaw, 1, 0);
            BodyFrameMath.HorizontalVector right = BodyFrameMath.toWorld(yaw, 0, 1);

            assertEquals(1, Math.hypot(right.x(), right.z()), EPSILON, "yaw " + yaw);
            assertEquals(-forward.z(), right.x(), EPSILON, "yaw " + yaw);
            assertEquals(forward.x(), right.z(), EPSILON, "yaw " + yaw);
        }
    }

    /**
     * ADR 0020 uses {@code toBody} to oppose a world-frame drift without restating the handedness,
     * so it has to be the exact inverse of {@code toWorld} at every yaw rather than merely close.
     * A sign error here would put a body program's station keeping the wrong way round while every
     * existing pin still passed.
     */
    @Test
    void toBodyInvertsToWorldAtEveryYaw() {
        for (int yaw = -180; yaw <= 180; yaw += 15) {
            for (double forward : new double[] {-0.7, 0.0, 0.3, 1.0}) {
                for (double right : new double[] {-1.0, 0.0, 0.45}) {
                    BodyFrameMath.HorizontalVector world =
                            BodyFrameMath.toWorld(yaw, forward, right);
                    BodyFrameMath.BodyVector body =
                            BodyFrameMath.toBody(yaw, world.x(), world.z());

                    assertEquals(forward, body.forward(), EPSILON, "yaw " + yaw);
                    assertEquals(right, body.right(), EPSILON, "yaw " + yaw);
                }
            }
        }
    }

    /** At yaw 0 the body faces south and its right is west, so world south resolves to forward. */
    @Test
    void toBodyResolvesWorldSouthToForwardAtYawZero() {
        BodyFrameMath.BodyVector south = BodyFrameMath.toBody(0, 0, 1);
        BodyFrameMath.BodyVector west = BodyFrameMath.toBody(0, -1, 0);

        assertEquals(1, south.forward(), EPSILON);
        assertEquals(0, south.right(), EPSILON);
        assertEquals(0, west.forward(), EPSILON);
        assertEquals(1, west.right(), EPSILON);
    }

    @Test
    void rejectsNonFiniteCoordinates() {
        assertThrows(
                IllegalArgumentException.class,
                () -> BodyFrameMath.toWorld(Double.NaN, 1, 0));
        assertThrows(
                IllegalArgumentException.class,
                () -> BodyFrameMath.toBody(0, Double.NaN, 0));
    }
}
