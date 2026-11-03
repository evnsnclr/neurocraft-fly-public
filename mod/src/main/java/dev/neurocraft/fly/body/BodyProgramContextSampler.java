package dev.neurocraft.fly.body;

import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import dev.neurocraft.fly.entity.BodyFrameMath;
import dev.neurocraft.fly.entity.FlyEntity;
import java.util.Optional;
import java.util.Set;
import net.minecraft.core.Direction;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.entity.item.ItemEntity;
import net.minecraft.world.level.ClipContext;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.HitResult;
import net.minecraft.world.phys.Vec3;

/**
 * Bounded world context for scripted programs, never additional neural sensory input.
 * Five vertical block probes cover the centre and inset corners of the current footprint.
 * Surface distance is vertical clearance from the FEET, independent of the body's pitch.
 * Support ignores entity hits and fluids. Player targeting is a separate, explicit engineering
 * source: nearest visible alive non-spectator player by centre distance, UUID order on exact ties.
 * Food is an independent nearest-visible-item lookup over four declared vanilla item ids.
 * No context is added to the sensory snapshot, and no item is consumed or navigated toward.
 */
public final class BodyProgramContextSampler {
    /** Scripted support search ends six blocks below the feet; no unbounded ground search. */
    public static final double MAX_SURFACE_DISTANCE_BLOCKS = 6.0;
    public static final double FOOTPRINT_INSET_BLOCKS = 0.0001;
    public static final double SUPPORT_HEIGHT_TOLERANCE_BLOCKS = 0.0001;
    /** Chosen local interaction radius, inclusive, measured between bounding-box centres. */
    public static final double TARGET_RADIUS_BLOCKS = 12.0;

    /** Chosen close interaction radius; Euclidean bounding-box-centre distance, inclusive. */
    public static final double FOOD_RADIUS_BLOCKS = 0.75;
    /** Explicit presentation whitelist, not a biological diet or Minecraft consumption recipe. */
    public static final Set<String> FOOD_ITEM_IDS = Set.of(
            "minecraft:sugar", "minecraft:apple", "minecraft:sweet_berries", "minecraft:glow_berries");

    private BodyProgramContextSampler() {}

    public static BodyProgramContext sample(
            ServerLevel level, FlyEntity entity, PhysicalSensorSnapshot snapshot) {
        if (entity.level() != level || snapshot.gameTick() != level.getGameTime()
                || !snapshot.entityId().equals(entity.getStringUUID())) {
            throw new IllegalArgumentException("body context requires the same entity and tick");
        }
        Optional<BodyProgramContext.BodyTargetVector> target = target(level, entity);
        Optional<BodyProgramContext.ReachableFood> food = food(level, entity);
        var box = entity.getBoundingBox();
        Vec3 centre = box.getCenter();
        double insetX = Math.min(FOOTPRINT_INSET_BLOCKS, box.getXsize() / 4.0);
        double insetZ = Math.min(FOOTPRINT_INSET_BLOCKS, box.getZsize() / 4.0);
        double[][] points = {
            {centre.x, centre.z},
            {box.minX + insetX, box.minZ + insetZ},
            {box.minX + insetX, box.maxZ - insetZ},
            {box.maxX - insetX, box.minZ + insetZ},
            {box.maxX - insetX, box.maxZ - insetZ}
        };
        Optional<BodyProgramContext.SurfaceBelow> centreSurface =
                surface(level, entity, points[0], centre.y, box.minY);
        if (centreSurface.isEmpty()) {
            return new BodyProgramContext(snapshot, Optional.empty(), target, food);
        }
        double distance = centreSurface.orElseThrow().distanceBlocks();
        boolean sturdy = centreSurface.orElseThrow().sturdy();
        for (int index = 1; index < points.length; index++) {
            Optional<BodyProgramContext.SurfaceBelow> corner =
                    surface(level, entity, points[index], centre.y, box.minY);
            sturdy &= corner.isPresent() && corner.orElseThrow().sturdy()
                    && Math.abs(corner.orElseThrow().distanceBlocks() - distance)
                            <= SUPPORT_HEIGHT_TOLERANCE_BLOCKS;
        }
        return new BodyProgramContext(snapshot,
                Optional.of(new BodyProgramContext.SurfaceBelow(distance, sturdy)),
                target, food);
    }

    private static Optional<BodyProgramContext.BodyTargetVector> target(
            ServerLevel level, FlyEntity entity) {
        Vec3 from = entity.getBoundingBox().getCenter();
        net.minecraft.world.entity.player.Player nearest = null;
        Vec3 nearestDelta = null;
        double nearestSquared = TARGET_RADIUS_BLOCKS * TARGET_RADIUS_BLOCKS;
        for (var player : level.players()) {
            if (player.isSpectator() || !player.isAlive() || player.isRemoved()) {
                continue;
            }
            Vec3 delta = player.getBoundingBox().getCenter().subtract(from);
            double squared = delta.lengthSqr();
            if (squared > nearestSquared || (squared == nearestSquared && nearest != null
                    && player.getUUID().compareTo(nearest.getUUID()) >= 0)) {
                continue;
            }
            // Block collision shapes occlude; fluids and other entities do not. A glass wall
            // therefore blocks this scripted target even though the player is visually visible.
            BlockHitResult hit = level.clip(new ClipContext(from, from.add(delta),
                    ClipContext.Block.COLLIDER, ClipContext.Fluid.NONE, entity));
            if (hit.getType() != HitResult.Type.MISS) {
                continue;
            }
            nearest = player;
            nearestDelta = delta;
            nearestSquared = squared;
        }
        if (nearestDelta == null) {
            return Optional.empty();
        }
        BodyFrameMath.BodyVector horizontal = BodyFrameMath.toBody(
                entity.getYRot(), nearestDelta.x, nearestDelta.z);
        double bearing = Math.toDegrees(StrictMath.atan2(horizontal.right(), horizontal.forward()));
        // The equivalent +/-180 headings have one representation and one turn direction.
        if (bearing == -180.0) {
            bearing = 180.0;
        }
        return Optional.of(new BodyProgramContext.BodyTargetVector(
                bearing + 0.0,
                Math.toDegrees(StrictMath.atan2(nearestDelta.y, nearestDelta.horizontalDistance())),
                Math.sqrt(nearestSquared)));
    }

    private static Optional<BodyProgramContext.ReachableFood> food(ServerLevel level, FlyEntity entity) {
        Vec3 from = entity.getBoundingBox().getCenter();
        ItemEntity nearest = null;
        Vec3 nearestDelta = null;
        double nearestSquared = FOOD_RADIUS_BLOCKS * FOOD_RADIUS_BLOCKS;
        for (ItemEntity item : level.getEntitiesOfClass(ItemEntity.class,
                entity.getBoundingBox().inflate(FOOD_RADIUS_BLOCKS), candidate ->
                        candidate.isAlive() && !candidate.isRemoved() && !candidate.getItem().isEmpty()
                                && FOOD_ITEM_IDS.contains(BuiltInRegistries.ITEM
                                        .getKey(candidate.getItem().getItem()).toString()))) {
            Vec3 delta = item.getBoundingBox().getCenter().subtract(from);
            double squared = delta.lengthSqr();
            if (squared > nearestSquared || (squared == nearestSquared && nearest != null
                    && item.getUUID().compareTo(nearest.getUUID()) >= 0)) {
                continue;
            }
            if (level.clip(new ClipContext(from, from.add(delta), ClipContext.Block.COLLIDER,
                    ClipContext.Fluid.NONE, entity)).getType() != HitResult.Type.MISS) {
                continue;
            }
            nearest = item;
            nearestDelta = delta;
            nearestSquared = squared;
        }
        if (nearest == null) {
            return Optional.empty();
        }
        BodyFrameMath.BodyVector horizontal = BodyFrameMath.toBody(
                entity.getYRot(), nearestDelta.x, nearestDelta.z);
        double bearing = Math.toDegrees(StrictMath.atan2(horizontal.right(), horizontal.forward()));
        if (bearing == -180.0) {
            bearing = 180.0;
        }
        return Optional.of(new BodyProgramContext.ReachableFood(bearing + 0.0,
                Math.sqrt(nearestSquared), BuiltInRegistries.ITEM.getKey(nearest.getItem().getItem()).toString()));
    }

    private static Optional<BodyProgramContext.SurfaceBelow> surface(
            ServerLevel level, FlyEntity entity, double[] point, double centreY, double feetY) {
        Vec3 from = new Vec3(point[0], centreY, point[1]);
        Vec3 to = new Vec3(point[0], feetY - MAX_SURFACE_DISTANCE_BLOCKS, point[1]);
        BlockHitResult hit = level.clip(new ClipContext(
                from, to, ClipContext.Block.COLLIDER, ClipContext.Fluid.NONE, entity));
        if (hit.getType() == HitResult.Type.MISS) {
            return Optional.empty();
        }
        double distance = feetY - hit.getLocation().y;
        if (distance < -SUPPORT_HEIGHT_TOLERANCE_BLOCKS
                || distance > MAX_SURFACE_DISTANCE_BLOCKS) {
            return Optional.empty();
        }
        boolean sturdy = hit.getDirection() == Direction.UP
                && level.getBlockState(hit.getBlockPos())
                        .isFaceSturdy(level, hit.getBlockPos(), Direction.UP);
        return Optional.of(new BodyProgramContext.SurfaceBelow(Math.max(0.0, distance), sturdy));
    }
}
