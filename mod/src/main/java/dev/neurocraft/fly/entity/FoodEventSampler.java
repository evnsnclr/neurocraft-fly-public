package dev.neurocraft.fly.entity;

import dev.neurocraft.fly.body.BodyProgramContextSampler;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.entity.item.ItemEntity;
import net.minecraft.world.level.ClipContext;
import net.minecraft.world.phys.HitResult;
import net.minecraft.world.phys.Vec3;

/** Declared three-block food presence; no bearing, selected item, or target vector reaches the wire. */
public final class FoodEventSampler {
    public static final double RADIUS_BLOCKS = 3.0;
    public static final double FORWARD_CONE_DEGREES = 60.0;

    private FoodEventSampler() {}

    public static boolean foodNear(ServerLevel level, FlyEntity fly) {
        Vec3 from = fly.getBoundingBox().getCenter();
        ItemEntity nearest = null;
        Vec3 nearestDelta = null;
        double nearestSquared = RADIUS_BLOCKS * RADIUS_BLOCKS;
        for (ItemEntity item : level.getEntitiesOfClass(ItemEntity.class,
                fly.getBoundingBox().inflate(RADIUS_BLOCKS), candidate ->
                        candidate.isAlive() && !candidate.isRemoved() && !candidate.getItem().isEmpty()
                                && BodyProgramContextSampler.FOOD_ITEM_IDS.contains(BuiltInRegistries.ITEM
                                        .getKey(candidate.getItem().getItem()).toString()))) {
            Vec3 delta = item.getBoundingBox().getCenter().subtract(from);
            double squared = delta.lengthSqr();
            if (squared > nearestSquared || (squared == nearestSquared && nearest != null
                    && item.getUUID().compareTo(nearest.getUUID()) >= 0)) { continue; }
            if (level.clip(new ClipContext(from, from.add(delta), ClipContext.Block.COLLIDER,
                    ClipContext.Fluid.NONE, fly)).getType() != HitResult.Type.MISS) { continue; }
            nearest = item;
            nearestDelta = delta;
            nearestSquared = squared;
        }
        if (nearest == null) { return false; }
        var horizontal = BodyFrameMath.toBody(fly.getYRot(), nearestDelta.x, nearestDelta.z);
        double bearing = Math.toDegrees(StrictMath.atan2(horizontal.right(), horizontal.forward()));
        // Select the nearest visible item first. A nearer off-cone item masks farther food.
        return Math.abs(bearing) <= FORWARD_CONE_DEGREES;
    }
}
