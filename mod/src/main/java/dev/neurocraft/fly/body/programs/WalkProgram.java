package dev.neurocraft.fly.body.programs;

import dev.neurocraft.fly.body.BodyProgram;
import dev.neurocraft.fly.body.BodyProgramContext;
import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.BodyProgramState;
import dev.neurocraft.fly.body.BodyProgramStep;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ProgramParameters;
import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import dev.neurocraft.fly.control.PrimitiveAction;
import dev.neurocraft.fly.entity.BodyFrameMath;
import java.util.Optional;

/**
 * WALK: operator-selected ground motion over the currently supported footprint, with no target.
 * Speed scales forward thrust; amplitude scales its 12-tick gait attenuation; turnRate scales
 * signed turnBias inside the declared yaw limit. The operator supplies zero turnBias and walks
 * straight. Urgency is ignored. These are scripted body terms, not neural or anatomical claims.
 * A small downward force maintains collision contact in the gravity-free body. Lost ground or
 * sturdy support exits HOVER; a frontal obstacle brakes without choosing an avoidance direction.
 */
public final class WalkProgram implements BodyProgram {
    public static final double FORWARD_THRUST = 0.12;
    public static final double YAW_TORQUE = 0.08;
    public static final double GAIT_ATTENUATION = 0.25;
    public static final int GAIT_PERIOD_TICKS = 12;
    public static final double STOP_CLEARANCE_BLOCKS = 0.35;

    @Override
    public BodyProgramId id() {
        return BodyProgramId.WALK;
    }

    @Override
    public boolean canEnter(BodyProgramContext context) {
        return LandProgram.hasGroundContact(context);
    }

    @Override
    public BodyProgramStep step(BodyProgramState state, BodyProgramContext context,
            Modulation modulation, ProgramParameters parameters) {
        PhysicalSensorSnapshot snapshot = context.snapshot();
        int phase = Math.floorMod(state.ticksInProgram(), GAIT_PERIOD_TICKS);
        if (!canEnter(context)) {
            return new BodyProgramStep(
                    LandProgram.brakeHorizontal(snapshot,
                            LandProgram.verticalThrust(snapshot.velocityYBlocksPerTick(), 0.0)),
                    state.advanced(phase), 0.0F, Optional.of(BodyProgramId.HOVER));
        }
        BodyFrameMath.BodyVector drift = BodyFrameMath.toBody(snapshot.yawDegrees(),
                snapshot.velocityXBlocksPerTick(), snapshot.velocityZBlocksPerTick());
        double gait = 1.0 - GAIT_ATTENUATION * modulation.amplitude()
                * (0.5 + 0.5 * Math.cos(2.0 * Math.PI * phase / GAIT_PERIOD_TICKS));
        boolean stopped = modulation.speed() == 0.0 || snapshot.horizontalCollision()
                || snapshot.rayForwardBlocks() <= STOP_CLEARANCE_BLOCKS;
        double forward = stopped ? LandProgram.brake(drift.forward())
                : FORWARD_THRUST * modulation.speed() * gait;
        return new BodyProgramStep(
                PrimitiveAction.clamped(forward, LandProgram.brake(drift.right()),
                        -LandProgram.GROUND_CONTACT_THRUST,
                        YAW_TORQUE * modulation.turnRate() * parameters.turnBias(), 0.0),
                state.advanced(Math.floorMod(phase + 1, GAIT_PERIOD_TICKS)),
                (float) (phase / (double) GAIT_PERIOD_TICKS), Optional.empty());
    }
}
