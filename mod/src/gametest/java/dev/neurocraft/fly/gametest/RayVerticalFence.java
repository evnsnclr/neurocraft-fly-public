package dev.neurocraft.fly.gametest;

import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import net.minecraft.core.BlockPos;
import net.minecraft.world.phys.Vec3;

/**
 * Test-only handedness reference for the two vertical sensor rays.
 *
 * <p><b>Why this class has to exist.</b> Before ADR 0015 the up and down rays were accidentally
 * correct. {@code FlySensors} built its lateral axis as {@code WORLD_UP.cross(forward)}, which is
 * the body's LEFT (see {@link RaySideFence}), and then built {@code bodyUp} as
 * {@code forward.cross(thatAxis)}. Because {@code forward x left == -(forward x right) == +up}, the
 * two sign errors cancelled and the vertical pair read correctly at every attitude. Correcting only
 * the lateral axis would therefore have silently inverted up and down, and nothing in the repository
 * would have noticed: no test read {@code rayUpBlocks} or {@code rayDownBlocks} at all. This class
 * and its two probes close that hole, so the vertical axis is now held by evidence rather than by a
 * cancellation nobody had written down.
 *
 * <p><b>Independent reference for body-up.</b> The expectation below must not be a second copy of
 * the cross products under test, or it would agree with any consistent error. It is instead taken
 * from the engine's own look-direction function: body-up is the direction the body would look if it
 * pitched 90 degrees further up, so
 *
 * <pre>
 *   bodyUp(pitch, yaw) = Vec3.directionFromRotation(pitch - 90, yaw)
 * </pre>
 *
 * <p>Worked from the shipped bytecode of {@code directionFromRotation(xRot, yRot)}, which returns
 * {@code (g*h, i, f*h)} for {@code f = cos(-yaw_rad - PI)}, {@code g = sin(-yaw_rad - PI)},
 * {@code h = -cos(-pitch_rad)}, {@code i = sin(-pitch_rad)}:
 *
 * <pre>
 *   pitch   0, yaw 0:  directionFromRotation(-90, 0)
 *                      i = sin(90) = +1, h = -cos(90) = 0  ->  ( 0, +1,  0)  world up
 *   pitch -40, yaw 0:  directionFromRotation(-130, 0)
 *                      i = sin(130) = +0.766, h = -cos(130) = +0.643, f = cos(-PI) = -1
 *                      -> ( 0, +0.766, -0.643)  up and NORTH, tilted 40 degrees off vertical
 * </pre>
 *
 * <p>The second case is the one that matters: a body with its nose pitched up 40 degrees has its
 * back tilted 40 degrees toward its tail, so its up ray leans north while it faces south. A ray that
 * tracked world up would read the ceiling at the plain vertical distance; an inverted one would look
 * at the floor. Both are separated from the correct answer by more than half a block in the probe
 * geometry, so the check does not depend on a tight tolerance to have teeth.
 */
final class RayVerticalFence {
    /**
     * Vertical readings at level pitch are exact block-face intersections; this only absorbs double
     * rounding, exactly as {@link RaySideFence#TOLERANCE_BLOCKS} does laterally.
     */
    static final double TOLERANCE_BLOCKS = 1.0e-6;

    /**
     * Pitched readings reconcile two different float paths to the same geometry -- the sensor's two
     * cross products and normalizations, against this class's single table lookup -- through
     * Minecraft's quantized {@code Mth.sin}/{@code Mth.cos}. Their disagreement is float noise on a
     * multi-block distance. The discriminating alternatives (world-locked up, or an inverted axis)
     * sit more than 0.5 blocks away, so this tolerance is still four orders of magnitude tighter
     * than the thing it has to separate.
     */
    static final double PITCHED_TOLERANCE_BLOCKS = 1.0e-4;

    private RayVerticalFence() {}

    /**
     * The body's up axis, from the engine's own look-direction table rather than from the cross
     * products under test. See the class notes for the derivation.
     */
    static Vec3 bodyUp(float pitchDegrees, float yawDegrees) {
        return Vec3.directionFromRotation(pitchDegrees - 90.0F, yawDegrees).normalize();
    }

    /** Which declared vertical slot moved off its open-arena baseline. */
    static VerticalSlot shortenedSlot(PhysicalSensorSnapshot open, PhysicalSensorSnapshot blocked) {
        boolean up = blocked.rayUpBlocks() < open.rayUpBlocks() - TOLERANCE_BLOCKS;
        boolean down = blocked.rayDownBlocks() < open.rayDownBlocks() - TOLERANCE_BLOCKS;
        if (up && down) {
            return VerticalSlot.BOTH;
        }
        if (up) {
            return VerticalSlot.UP;
        }
        return down ? VerticalSlot.DOWN : VerticalSlot.NEITHER;
    }

    /** Reading of the declared slot named by {@code slot}. */
    static double reading(PhysicalSensorSnapshot snapshot, VerticalSlot slot) {
        return switch (slot) {
            case UP -> snapshot.rayUpBlocks();
            case DOWN -> snapshot.rayDownBlocks();
            case NEITHER, BOTH ->
                throw new IllegalArgumentException("no single declared slot to read");
        };
    }

    /** True when every non-vertical ray is bit-identical across the two samples. */
    static boolean nonVerticalRaysUnchanged(
            PhysicalSensorSnapshot open, PhysicalSensorSnapshot blocked) {
        return open.rayForwardBlocks() == blocked.rayForwardBlocks()
                && open.rayBackBlocks() == blocked.rayBackBlocks()
                && open.rayLeftBlocks() == blocked.rayLeftBlocks()
                && open.rayRightBlocks() == blocked.rayRightBlocks();
    }

    /**
     * True when {@code column} is the block column the body occupies, so a block placed there really
     * is directly overhead or directly underfoot. Keeps the probe honest about placement the way
     * {@link RaySideFence#obstacleLiesOn} does laterally: a translated structure would break this
     * loudly instead of quietly measuring an oblique hit.
     */
    static boolean liesInColumnOf(Vec3 flyAbsolute, BlockPos columnAbsolute) {
        return flyAbsolute.x > columnAbsolute.getX()
                && flyAbsolute.x < columnAbsolute.getX() + 1.0
                && flyAbsolute.z > columnAbsolute.getZ()
                && flyAbsolute.z < columnAbsolute.getZ() + 1.0;
    }

    /** Distance from {@code originY} up to the bottom face of {@code blockAbsolute}. */
    static double distanceUpToBottomFace(double originY, BlockPos blockAbsolute) {
        return blockAbsolute.getY() - originY;
    }

    /** Distance from {@code originY} down to the top face of {@code blockAbsolute}. */
    static double distanceDownToTopFace(double originY, BlockPos blockAbsolute) {
        return originY - (blockAbsolute.getY() + 1.0);
    }

    /** The two body-relative vertical ray slots as the protocol declares them. */
    enum VerticalSlot {
        UP,
        DOWN,
        NEITHER,
        BOTH
    }
}
