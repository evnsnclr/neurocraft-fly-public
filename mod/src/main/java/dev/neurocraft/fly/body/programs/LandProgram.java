package dev.neurocraft.fly.body.programs;

import dev.neurocraft.fly.body.BodyProgram;
import dev.neurocraft.fly.body.BodyProgramContext;
import dev.neurocraft.fly.body.BodyProgramContextSampler;
import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.BodyProgramState;
import dev.neurocraft.fly.body.BodyProgramStep;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ProgramParameters;
import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import dev.neurocraft.fly.control.PrimitiveAction;
import dev.neurocraft.fly.entity.BodyFrameMath;
import dev.neurocraft.fly.entity.PrimitiveActuator;
import java.util.Optional;

/**
 * LAND: operator-selected, vertical descent onto a complete sturdy footprint.
 * These are declared Minecraft body constants, not anatomy or a neural policy.
 * The descent phase has 200 owned ticks to touch down; lost support or timeout exits HOVER.
 * Touchdown enters a separate grounded hold, retained until the operator selects or releases.
 * It never selects WALK. Only speed modulates the descent target; other inputs are ignored.
 */
public final class LandProgram implements BodyProgram {
    public static final int MAX_DESCENT_TICKS = 200;
    public static final int DESCENDING = 0;
    public static final int GROUNDED_HOLD = 1;
    public static final double MAX_DESCENT_RATE_BLOCKS_PER_TICK = 0.06;
    public static final double MIN_CONTACT_RATE_BLOCKS_PER_TICK = 0.005;
    public static final double SLOWDOWN_DISTANCE_BLOCKS = 0.5;
    public static final double CONTACT_CLEARANCE_BLOCKS = 0.02;
    public static final double VERTICAL_THRUST_LIMIT = 0.35;
    public static final double HORIZONTAL_BRAKE_LIMIT = 0.25;
    /** No ambient gravity exists: this bounded force keeps collision contact alive at rest. */
    public static final double GROUND_CONTACT_THRUST = 0.02;
    /** Resting support may retain no more downward speed than one declared contact-force tick. */
    public static final double MAX_RESTING_DESCENT_BLOCKS_PER_TICK =
            GROUND_CONTACT_THRUST * PrimitiveActuator.MAX_THRUST_DELTA_BLOCKS_PER_TICK;

    @Override
    public BodyProgramId id() {
        return BodyProgramId.LAND;
    }

    @Override
    public boolean canEnter(BodyProgramContext context) {
        return hasLandableSurface(context);
    }

    public static boolean hasLandableSurface(BodyProgramContext context) {
        return !context.snapshot().inWater() && !context.snapshot().inLava()
                && context.surfaceBelow().filter(surface -> surface.sturdy()
                        && surface.distanceBlocks() >= 0.0
                        && surface.distanceBlocks()
                                <= BodyProgramContextSampler.MAX_SURFACE_DISTANCE_BLOCKS).isPresent();
    }

    /**
     * Scripted body support, not a replacement for Minecraft's movement-contact flags.
     * A neutral tick in the gravity-free body clears onGround even while feet remain exactly
     * on a sturdy floor. Retain that physical resting case only inside the sampler's precision
     * tolerance, with no upward motion and at most one contact-force tick of downward motion.
     * Every call rechecks the complete footprint; no remembered contact can bridge a lost edge.
     */
    public static boolean hasGroundContact(BodyProgramContext context) {
        double vertical = context.snapshot().velocityYBlocksPerTick();
        if (!hasLandableSurface(context) || vertical > 0.0) {
            return false;
        }
        double distance = context.surfaceBelow().orElseThrow().distanceBlocks();
        return (context.snapshot().onGround() && distance <= CONTACT_CLEARANCE_BLOCKS)
                || (distance <= BodyProgramContextSampler.SUPPORT_HEIGHT_TOLERANCE_BLOCKS
                        && vertical >= -MAX_RESTING_DESCENT_BLOCKS_PER_TICK);
    }

    @Override
    public BodyProgramStep step(BodyProgramState state, BodyProgramContext context,
            Modulation modulation, ProgramParameters parameters) {
        boolean grounded = hasGroundContact(context);
        boolean exit = !hasLandableSurface(context)
                || (state.phaseIndex() == GROUNDED_HOLD && !grounded)
                || (!grounded && state.ticksInProgram() >= MAX_DESCENT_TICKS - 1);
        double vertical;
        if (exit) {
            vertical = verticalThrust(context.snapshot().velocityYBlocksPerTick(), 0.0);
        } else if (grounded) {
            vertical = -GROUND_CONTACT_THRUST;
        } else {
            double distance = context.surfaceBelow().orElseThrow().distanceBlocks();
            double target = -modulation.speed() * Math.max(MIN_CONTACT_RATE_BLOCKS_PER_TICK,
                    MAX_DESCENT_RATE_BLOCKS_PER_TICK
                            * Math.min(1.0, distance / SLOWDOWN_DISTANCE_BLOCKS));
            vertical = verticalThrust(context.snapshot().velocityYBlocksPerTick(), target);
        }
        return new BodyProgramStep(brakeHorizontal(context.snapshot(), vertical),
                state.advanced(grounded ? GROUNDED_HOLD : DESCENDING),
                grounded ? 0.0F : (float) Math.min(0.99, state.ticksInProgram() / (double) MAX_DESCENT_TICKS),
                exit ? Optional.of(BodyProgramId.HOVER) : Optional.empty());
    }

    /** Rate feedback may brake in either direction, but remains within the declared thrust. */
    public static double verticalThrust(double currentRate, double targetRate) {
        return Math.max(-VERTICAL_THRUST_LIMIT, Math.min(VERTICAL_THRUST_LIMIT,
                (targetRate - currentRate) / PrimitiveActuator.MAX_THRUST_DELTA_BLOCKS_PER_TICK)) + 0.0;
    }

    public static PrimitiveAction brakeHorizontal(PhysicalSensorSnapshot snapshot, double vertical) {
        BodyFrameMath.BodyVector drift = BodyFrameMath.toBody(snapshot.yawDegrees(),
                snapshot.velocityXBlocksPerTick(), snapshot.velocityZBlocksPerTick());
        return PrimitiveAction.clamped(brake(drift.forward()), brake(drift.right()), vertical, 0.0, 0.0);
    }

    public static double brake(double speed) {
        return Math.max(-HORIZONTAL_BRAKE_LIMIT, Math.min(HORIZONTAL_BRAKE_LIMIT,
                -speed / PrimitiveActuator.MAX_THRUST_DELTA_BLOCKS_PER_TICK)) + 0.0;
    }
}
