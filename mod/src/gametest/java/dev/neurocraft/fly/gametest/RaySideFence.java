package dev.neurocraft.fly.gametest;

import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import dev.neurocraft.fly.entity.FlyEntity;
import java.util.Optional;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.world.phys.Vec3;

/**
 * Test-only handedness reference for the two lateral sensor rays, plus a causal gate that admits
 * only an observation sampled strictly after obstacle placement.
 *
 * <p>D-068 probe, ADR 0015. The expected answers below are derived from Minecraft 1.21.5's own
 * coordinate and yaw conventions and are anchored on the engine's {@code Direction.getClockWise()},
 * so this class never hard-codes a hand-picked "left"/"right" constant that could silently agree
 * with the production convention it is testing.
 *
 * <p><b>Conventions.</b> {@code +X} is east, {@code +Y} is up, {@code +Z} is south. Yaw 0 faces
 * {@code +Z} (south) and yaw increases clockwise viewed from above. Verified against the shipped
 * bytecode of {@code Vec3.directionFromRotation(xRot, yRot)}, which returns {@code (g*h, i, f*h)}
 * for {@code f = cos(-yaw_rad - PI)}, {@code g = sin(-yaw_rad - PI)}, {@code h = -cos(-pitch_rad)},
 * {@code i = sin(-pitch_rad)}:
 *
 * <pre>
 *   yaw   0, pitch 0:  f = cos(-PI)    = -1,  g = sin(-PI)    =  0,  h = -1
 *                      forward = (g*h, i, f*h) = ( 0, 0, +1)  = SOUTH
 *   yaw  90, pitch 0:  f = cos(-3PI/2) =  0,  g = sin(-3PI/2) = +1,  h = -1
 *                      forward = (g*h, i, f*h) = (-1, 0,  0)  = WEST
 * </pre>
 *
 * <p><b>Which way is the body's right?</b> A body's right is the heading it faces after turning 90
 * degrees clockwise as seen from above. That is exactly {@code Direction.getClockWise()}, whose
 * shipped table is NORTH to EAST, EAST to SOUTH, SOUTH to WEST, WEST to NORTH. Face north and your
 * right hand points east; face south and it points west. Therefore:
 *
 * <pre>
 *   yaw   0: facing SOUTH -> right = WEST  = (-1, 0,  0),  left = EAST  = (+1, 0,  0)
 *   yaw  90: facing WEST  -> right = NORTH = ( 0, 0, -1),  left = SOUTH = ( 0, 0, +1)
 * </pre>
 *
 * <p><b>As a cross product</b> (using {@code a x b = (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx)}
 * in this right-handed basis, since {@code east x up = south}), the body's right is
 * {@code forward x up}:
 *
 * <pre>
 *   yaw   0: forward x up = ( 0,0, 1) x (0,1,0) = (0*0 - 1*1, 1*0 - 0*0, 0*1 - 0*0) = (-1,0, 0) WEST
 *   yaw  90: forward x up = (-1,0, 0) x (0,1,0) = (0*0 - 0*1, 0*0 + 1*0, -1*1 - 0*0) = ( 0,0,-1) NORTH
 * </pre>
 *
 * <p><b>The defect this class was written to measure.</b> Before ADR 0015 {@code FlySensors} computed
 * its lateral axis as {@code WORLD_UP.cross(forward)}, which is the negation of the above:
 *
 * <pre>
 *   yaw   0: up x forward = (0,1,0) x ( 0,0, 1) = (1*1 - 0*0, 0*0 - 0*1, 0*0 - 1*0) = (+1,0, 0) EAST
 *   yaw  90: up x forward = (0,1,0) x (-1,0, 0) = (1*0 - 0*0, 0*-1 - 0*0, 0*0 + 1*1) = ( 0,0,+1) SOUTH
 * </pre>
 *
 * <p>So the variable named {@code right} in {@code FlySensors} was the body's LEFT at every yaw, and
 * the slot assignments that fed {@code rayLeftBlocks} and {@code rayRightBlocks} were transposed.
 * {@code BodyFrameMath} repeated the same convention with {@code rightX = cos(yaw)},
 * {@code rightZ = sin(yaw)}, which is {@code (+1, 0)} east at yaw 0 and {@code (0, +1)} south at yaw
 * 90 -- the body's left in both cases. Both sites now use {@code forward x up}; this class stays as
 * the pin that keeps them there.
 *
 * <p>Note that the up/down rays were unaffected: {@code bodyUp = forward.cross(right)} with a
 * left-handed {@code right} evaluates to {@code forward x left = +up}, so the two sign errors
 * cancelled on the vertical axis and only the lateral pair was mislabelled. See
 * {@link RayVerticalFence} for the coverage that now holds the vertical axis independently.
 */
final class RaySideFence {
    /** Face-to-origin distance for the obstacle geometry used by the lateral ray probes. */
    static final double EXPECTED_OBSTACLE_DISTANCE_BLOCKS = 1.5;

    /** Ray distances are exact block-face intersections; this only absorbs double rounding. */
    static final double TOLERANCE_BLOCKS = 1.0e-6;

    /**
     * Ticks scanned back from "now" when recovering a retained snapshot. {@code FlyEntity} keeps a
     * bounded three-entry history, so anything older than this is already evicted and a wider scan
     * would only invent false confidence.
     */
    private static final int SCAN_WINDOW_TICKS = 4;

    private final long obstaclePlacementTick;

    RaySideFence(long obstaclePlacementTick) {
        if (obstaclePlacementTick < 0) {
            throw new IllegalArgumentException("obstacle placement tick must be non-negative");
        }
        this.obstaclePlacementTick = obstaclePlacementTick;
    }

    /**
     * Newest retained authoritative snapshot sampled strictly after obstacle placement, or empty
     * when the entity has not yet published one inside the bounded history window.
     */
    Optional<PhysicalSensorSnapshot> latestPostPlacementSnapshot(FlyEntity fly, long nowTick) {
        long oldest = Math.max(obstaclePlacementTick + 1, nowTick - SCAN_WINDOW_TICKS + 1);
        for (long tick = nowTick; tick >= oldest; tick--) {
            Optional<PhysicalSensorSnapshot> retained = fly.sensorSnapshotAt(tick);
            if (retained.isPresent()) {
                return retained;
            }
        }
        return Optional.empty();
    }

    /** The heading a body at {@code yawDegrees} faces, per the engine's own yaw table. */
    static Direction facing(float yawDegrees) {
        return Direction.fromYRot(yawDegrees);
    }

    /** The body's right: a 90 degree clockwise turn from its heading, per the engine's own table. */
    static Direction bodyRight(float yawDegrees) {
        return facing(yawDegrees).getClockWise();
    }

    /** The body's left: a 90 degree counter-clockwise turn from its heading. */
    static Direction bodyLeft(float yawDegrees) {
        return facing(yawDegrees).getCounterClockWise();
    }

    /**
     * The declared slot that must shorten when an obstacle sits on the body's
     * {@code obstacleDirection} side, derived from the engine handedness rather than asserted.
     */
    static LateralSlot expectedSlot(float yawDegrees, Direction obstacleDirection) {
        if (obstacleDirection == bodyRight(yawDegrees)) {
            return LateralSlot.RIGHT;
        }
        if (obstacleDirection == bodyLeft(yawDegrees)) {
            return LateralSlot.LEFT;
        }
        throw new IllegalArgumentException("obstacle direction is not a lateral body axis");
    }

    /** Which declared lateral slot moved off its open-arena baseline. */
    static LateralSlot shortenedSlot(PhysicalSensorSnapshot open, PhysicalSensorSnapshot blocked) {
        boolean left = blocked.rayLeftBlocks() < open.rayLeftBlocks() - TOLERANCE_BLOCKS;
        boolean right = blocked.rayRightBlocks() < open.rayRightBlocks() - TOLERANCE_BLOCKS;
        if (left && right) {
            return LateralSlot.BOTH;
        }
        if (left) {
            return LateralSlot.LEFT;
        }
        return right ? LateralSlot.RIGHT : LateralSlot.NEITHER;
    }

    /** Reading of the declared slot named by {@code slot}. */
    static double reading(PhysicalSensorSnapshot snapshot, LateralSlot slot) {
        return switch (slot) {
            case LEFT -> snapshot.rayLeftBlocks();
            case RIGHT -> snapshot.rayRightBlocks();
            case NEITHER, BOTH ->
                throw new IllegalArgumentException("no single declared slot to read");
        };
    }

    /**
     * True when the obstacle really is on the {@code expected} world side of the body, checked in
     * absolute coordinates. This keeps the probe honest about structure placement: a rotated or
     * translated test structure would break the relative-to-absolute mapping loudly here instead of
     * silently measuring the wrong axis.
     */
    static boolean obstacleLiesOn(
            Direction expected, Vec3 flyAbsolute, BlockPos obstacleAbsolute) {
        Vec3 center = Vec3.atCenterOf(obstacleAbsolute);
        Vec3 horizontal = new Vec3(center.x - flyAbsolute.x, 0.0, center.z - flyAbsolute.z);
        if (horizontal.length() < 1.0e-9) {
            return false;
        }
        return horizontal.normalize().dot(expected.getUnitVec3()) > 0.999;
    }

    /** True when every non-lateral ray is bit-identical across the two samples. */
    static boolean nonLateralRaysUnchanged(
            PhysicalSensorSnapshot open, PhysicalSensorSnapshot blocked) {
        return open.rayForwardBlocks() == blocked.rayForwardBlocks()
                && open.rayBackBlocks() == blocked.rayBackBlocks()
                && open.rayUpBlocks() == blocked.rayUpBlocks()
                && open.rayDownBlocks() == blocked.rayDownBlocks();
    }

    /** The two body-relative lateral ray slots as the protocol declares them. */
    enum LateralSlot {
        LEFT,
        RIGHT,
        NEITHER,
        BOTH
    }
}
