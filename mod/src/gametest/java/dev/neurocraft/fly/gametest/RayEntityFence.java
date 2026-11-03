package dev.neurocraft.fly.gametest;

import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import net.minecraft.core.BlockPos;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.Vec3;

/**
 * Test-only geometric reference for the entity half of the sensor rays.
 *
 * <p><b>The defect this class was written to measure.</b> D-070, ADR 0017. Before that lane
 * {@code FlySensors.rayDistance} cast every ray with a single {@code level.clip(...)} under
 * {@code ClipContext.Block.COLLIDER}. That call consults <em>blocks only</em>: entities are not in
 * its search at all, at any {@code ClipContext.Block} value. A player standing in the fly's path,
 * walking at it, or swinging at it produced a bit-identical sensor vector to an empty arena -- the
 * ray passed through the body and reported whatever wall lay behind it. Every downstream proximity
 * channel therefore read the operator as vacuum. The rays now also sweep entities over the same
 * segment and keep the nearer of the two hits.
 *
 * <p><b>Independent expectations.</b> As in {@link RaySideFence} and {@link RayVerticalFence}, the
 * expected distances below must not be a second copy of the code under test, or they would agree
 * with any consistent error. For an axis-aligned ray the entry point is a single subtraction on the
 * near face of the target, so this class computes it by plain arithmetic on the target's own
 * geometry -- {@code entity.getBoundingBox()} straight from the engine, or the integer
 * {@link BlockPos} the test placed -- and never calls {@code AABB.clip}, which is the very routine
 * {@code ProjectileUtil.getEntityHitResult} uses internally.
 *
 * <pre>
 *   ray +Z (south):  distance = target.minZ - origin.z
 *   ray -Z (north):  distance = origin.z - target.maxZ
 * </pre>
 *
 * <p><b>Why an armor stand stands in for the operator.</b> The dedicated GameTest server has no
 * player to walk in front of the fly, so the probes use the simplest available body that takes the
 * same path through the declared predicate. {@code ArmorStand} extends {@code LivingEntity} and its
 * {@code isPickable()} disassembles to {@code super.isPickable() && !isMarker()}, where the super
 * call is {@code LivingEntity.isPickable() -> !isRemoved()}. A {@code Player} overrides neither
 * method, so it reaches the predicate through that identical {@code LivingEntity} override. The
 * clause under test is therefore the same one the operator's body would take, and a non-marker
 * armor stand is additionally motionless and AI-free, which keeps the geometry exact.
 */
final class RayEntityFence {
    /**
     * Axis-aligned ray readings are exact face intersections; this only absorbs double rounding,
     * exactly as {@link RaySideFence#TOLERANCE_BLOCKS} does laterally.
     */
    static final double TOLERANCE_BLOCKS = 1.0e-6;

    private RayEntityFence() {}

    /** Distance from {@code origin} along {@code +Z} to the near face of {@code box}. */
    static double distanceSouthToNearFace(Vec3 origin, AABB box) {
        return box.minZ - origin.z;
    }

    /** Distance from {@code origin} along {@code -Z} to the near face of {@code box}. */
    static double distanceNorthToNearFace(Vec3 origin, AABB box) {
        return origin.z - box.maxZ;
    }

    /** Distance from {@code origin} along {@code +Z} to the near face of a full block. */
    static double distanceSouthToNearFace(Vec3 origin, BlockPos blockAbsolute) {
        return blockAbsolute.getZ() - origin.z;
    }

    /** Distance from {@code origin} along {@code -Z} to the near face of a full block. */
    static double distanceNorthToNearFace(Vec3 origin, BlockPos blockAbsolute) {
        return origin.z - (blockAbsolute.getZ() + 1.0);
    }

    /**
     * True when a ray leaving {@code origin} along {@code +/-Z} actually enters {@code box}, i.e.
     * the box straddles the ray line in the two axes the ray does not travel along.
     *
     * <p>This keeps the probe honest about placement the way {@link RaySideFence#obstacleLiesOn} and
     * {@link RayVerticalFence#liesInColumnOf} do: a target nudged out of the ray line would make
     * "the reading did not change" trivially true, which is exactly the false pass this whole lane
     * exists to prevent. It is asserted on the negative probes too, so "an item entity does not
     * obstruct" cannot pass merely because the item drifted out of the way.
     */
    static boolean straddlesDepthRay(Vec3 origin, AABB box) {
        return origin.x > box.minX
                && origin.x < box.maxX
                && origin.y > box.minY
                && origin.y < box.maxY;
    }

    /** True when every ray except the forward one is bit-identical across the two samples. */
    static boolean nonForwardRaysUnchanged(
            PhysicalSensorSnapshot open, PhysicalSensorSnapshot blocked) {
        return open.rayBackBlocks() == blocked.rayBackBlocks()
                && open.rayLeftBlocks() == blocked.rayLeftBlocks()
                && open.rayRightBlocks() == blocked.rayRightBlocks()
                && open.rayUpBlocks() == blocked.rayUpBlocks()
                && open.rayDownBlocks() == blocked.rayDownBlocks();
    }

    /** True when all six declared rays are bit-identical across the two samples. */
    static boolean allRaysUnchanged(PhysicalSensorSnapshot open, PhysicalSensorSnapshot blocked) {
        return open.rayForwardBlocks() == blocked.rayForwardBlocks()
                && nonForwardRaysUnchanged(open, blocked);
    }

    /**
     * True when every declared ray reports at least {@code minimumBlocks}.
     *
     * <p>Deliberately not an equality against {@code RAY_LENGTH_BLOCKS}. The default 8x8x8 GameTest
     * structure is walled, and a body centred in it at {@code (4.5, *, 4.5)} reads its own shell at
     * 3.5 and 4.5 blocks rather than the 6.0 maximum -- measured, not assumed. Every other
     * assertion in this lane is therefore a delta against an open baseline taken in the same
     * structure. This one absolute check survives because it only has to separate "the body did not
     * hit itself" from "it did": the fly is 0.38 wide by 0.22 tall and the ray origin is its own
     * bounding-box centre, so a self-hit reports at most 0.19, an order of magnitude below any
     * threshold worth passing here.
     */
    static boolean allRaysAtLeast(PhysicalSensorSnapshot snapshot, double minimumBlocks) {
        return snapshot.rayForwardBlocks() >= minimumBlocks
                && snapshot.rayBackBlocks() >= minimumBlocks
                && snapshot.rayLeftBlocks() >= minimumBlocks
                && snapshot.rayRightBlocks() >= minimumBlocks
                && snapshot.rayUpBlocks() >= minimumBlocks
                && snapshot.rayDownBlocks() >= minimumBlocks;
    }

    /** True when {@code reading} matches {@code expected} to the exact-face tolerance. */
    static boolean reads(double reading, double expected) {
        return Math.abs(reading - expected) < TOLERANCE_BLOCKS;
    }
}
