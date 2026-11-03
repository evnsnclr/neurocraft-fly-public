package dev.neurocraft.fly.body.programs;

import dev.neurocraft.fly.body.BodyProgram;
import dev.neurocraft.fly.body.BodyProgramContext;
import dev.neurocraft.fly.body.BodyProgramContextSampler;
import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.BodyProgramState;
import dev.neurocraft.fly.body.BodyProgramStep;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ProgramParameters;
import dev.neurocraft.fly.control.PrimitiveAction;
import java.util.Optional;

/**
 * Scripted horizontal approach to the current visible player, with a chosen 2.5-block stand-off.
 * This is engineered player context, not a sensory or neural target. No vertical pursuit: the
 * vertical channel is CRUISE's floor-band feedback. No target or stand-off reached brakes and
 * exits to HOVER using the existing generic exit reason. Entry failure is PRECONDITION.
 * Reads speed and turnRate; amplitude, urgency and signed parameters are ignored.
 */
public final class ApproachProgram implements BodyProgram {
    public static final double MAX_FORWARD_THRUST = 0.35;
    public static final double MAX_YAW_TORQUE = 0.6;
    public static final double STAND_OFF_BLOCKS = 2.5;
    public static final double SLOW_DISTANCE_BLOCKS = 4.0;
    public static final double FORWARD_CONE_DEGREES = 45.0;
    /** Full steering at 45 degrees of error; positive/right breaks the exact 180-degree tie. */
    public static final double FULL_TURN_ERROR_DEGREES = 45.0;
    public static final int PHASE_PERIOD_TICKS = 20;

    @Override public BodyProgramId id() { return BodyProgramId.APPROACH; }
    @Override public boolean canEnter(BodyProgramContext context) { return hasTarget(context); }

    @Override
    public BodyProgramStep step(BodyProgramState state, BodyProgramContext context,
            Modulation modulation, ProgramParameters parameters) {
        if (!hasTarget(context)) {
            return BankProgram.stop(state, context);
        }
        var target = context.target().orElseThrow();
        double horizontalDistance = target.distanceBlocks()
                * StrictMath.cos(Math.toRadians(target.elevationDegrees()));
        if (horizontalDistance <= STAND_OFF_BLOCKS) {
            return BankProgram.stop(state, context);
        }
        var snapshot = context.snapshot();
        double bearing = wrapBearing(target.bearingDegrees());
        double yaw = BankProgram.steering(snapshot,
                MAX_YAW_TORQUE * bearing / FULL_TURN_ERROR_DEGREES,
                MAX_YAW_TORQUE, modulation.turnRate(), bearing < 0 ? -1.0 : 1.0);
        PrimitiveAction motors;
        if (BankProgram.avoiding(snapshot) || Math.abs(bearing) > FORWARD_CONE_DEGREES
                || modulation.speed() == 0.0) {
            motors = BankProgram.braking(snapshot, yaw);
        } else {
            double slow = Math.min(1.0,
                    (horizontalDistance - STAND_OFF_BLOCKS) / (SLOW_DISTANCE_BLOCKS - STAND_OFF_BLOCKS));
            motors = PrimitiveAction.clamped(MAX_FORWARD_THRUST * modulation.speed()
                    * slow * BankProgram.clearanceScale(snapshot), 0.0,
                    CruiseProgram.verticalThrust(snapshot), yaw, 0.0);
        }
        return new BodyProgramStep(motors, state.advanced(0),
                Math.floorMod(state.ticksInProgram(), PHASE_PERIOD_TICKS) / (float) PHASE_PERIOD_TICKS,
                Optional.empty());
    }

    static boolean hasTarget(BodyProgramContext context) {
        return context.target().filter(target -> target.distanceBlocks() >= 0.0
                && target.distanceBlocks() <= BodyProgramContextSampler.TARGET_RADIUS_BLOCKS
                && Math.abs(target.elevationDegrees()) <= 90.0
                && Math.abs(target.bearingDegrees()) <= 180.0).isPresent();
    }

    /** Canonical (-180,180], so exactly behind always requests a right turn. */
    static double wrapBearing(double degrees) {
        double wrapped = degrees % 360.0;
        if (wrapped <= -180.0) {
            wrapped += 360.0;
        } else if (wrapped > 180.0) {
            wrapped -= 360.0;
        }
        return wrapped + 0.0;
    }
}
