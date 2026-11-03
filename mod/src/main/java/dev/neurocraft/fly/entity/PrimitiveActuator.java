package dev.neurocraft.fly.entity;

import dev.neurocraft.fly.control.PrimitiveAction;
import dev.neurocraft.fly.control.RawPrimitiveAction;
import net.minecraft.util.Mth;
import net.minecraft.world.phys.Vec3;

/** Applies bounded force/torque primitives only; it has no target, path, or semantic action API. */
public final class PrimitiveActuator {
    public static final double MAX_THRUST_DELTA_BLOCKS_PER_TICK = 0.04;
    public static final double MAX_SPEED_BLOCKS_PER_TICK = 0.6;
    public static final float MAX_YAW_DELTA_DEGREES_PER_TICK = 6.0F;
    public static final float MAX_PITCH_DELTA_DEGREES_PER_TICK = 4.0F;
    public static final float MAX_ABS_PITCH_DEGREES = 89.0F;

    private PrimitiveActuator() {}

    public static ActuationReceipt apply(
            FlyEntity entity, RawPrimitiveAction rawAction, PrimitiveAction action) {
        Vec3 before = entity.getDeltaMovement();
        float yawBefore = entity.getYRot();
        float pitchBefore = entity.getXRot();
        BodyFrameMath.HorizontalVector horizontal = BodyFrameMath.toWorld(
                yawBefore,
                action.forwardThrust() * MAX_THRUST_DELTA_BLOCKS_PER_TICK,
                action.lateralThrust() * MAX_THRUST_DELTA_BLOCKS_PER_TICK);
        Vec3 after = before.add(
                horizontal.x(),
                action.verticalThrust() * MAX_THRUST_DELTA_BLOCKS_PER_TICK,
                horizontal.z());
        if (after.lengthSqr() > MAX_SPEED_BLOCKS_PER_TICK * MAX_SPEED_BLOCKS_PER_TICK) {
            after = after.normalize().scale(MAX_SPEED_BLOCKS_PER_TICK);
        }
        float yawAfter = Mth.wrapDegrees(
                yawBefore + (float) action.yawTorque() * MAX_YAW_DELTA_DEGREES_PER_TICK);
        float pitchAfter = Mth.clamp(
                pitchBefore + (float) action.pitchTorque() * MAX_PITCH_DELTA_DEGREES_PER_TICK,
                -MAX_ABS_PITCH_DEGREES,
                MAX_ABS_PITCH_DEGREES);
        entity.setDeltaMovement(after);
        entity.setYRot(yawAfter);
        entity.setXRot(pitchAfter);
        return new ActuationReceipt(
                rawAction,
                action,
                before.x,
                before.y,
                before.z,
                after.x,
                after.y,
                after.z,
                yawBefore,
                yawAfter,
                pitchBefore,
                pitchAfter);
    }
}
