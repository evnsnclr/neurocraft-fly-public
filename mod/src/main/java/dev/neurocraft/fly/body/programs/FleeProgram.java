package dev.neurocraft.fly.body.programs;

import dev.neurocraft.fly.body.BodyProgram;
import dev.neurocraft.fly.body.BodyProgramContext;
import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.BodyProgramState;
import dev.neurocraft.fly.body.BodyProgramStep;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ProgramParameters;
import dev.neurocraft.fly.control.PrimitiveAction;
import java.util.Optional;

/**
 * Scripted turn away from the current visible player, then a bounded horizontal burst.
 * All limits are chosen engineering values; this is not the neural escape arc. The target is
 * resampled each tick, never remembered after removal or occlusion. No target brakes and exits
 * HOVER using the existing generic exit reason; entry failure is PRECONDITION.
 * Reads speed, turnRate and urgency (burst duration only). Amplitude and signed bias are ignored.
 * Horizontal braking (0.25 per axis) and CRUISE floor-band feedback are never widened by modulation.
 */
public final class FleeProgram implements BodyProgram {
    public static final int MAX_TURN_TICKS = 40;
    public static final int MAX_BURST_TICKS = 40;
    public static final int MIN_BURST_TICKS = 20;
    public static final double MAX_FORWARD_THRUST = 0.65;
    public static final double MAX_YAW_TORQUE = 0.9;
    public static final double FORWARD_CONE_DEGREES = 45.0;
    public static final double FULL_TURN_ERROR_DEGREES = 45.0;

    @Override public BodyProgramId id() { return BodyProgramId.FLEE; }
    @Override public boolean canEnter(BodyProgramContext context) { return ApproachProgram.hasTarget(context); }

    @Override
    public BodyProgramStep step(BodyProgramState state, BodyProgramContext context,
            Modulation modulation, ProgramParameters parameters) {
        int duration = burstTicks(modulation.urgency());
        if (!canEnter(context) || state.ticksInProgram() >= MAX_TURN_TICKS + duration
                || state.phaseIndex() >= duration) {
            return BankProgram.stop(state, context);
        }
        var snapshot = context.snapshot();
        double away = ApproachProgram.wrapBearing(context.target().orElseThrow().bearingDegrees() + 180.0);
        boolean aligned = Math.abs(away) <= FORWARD_CONE_DEGREES;
        boolean bursting = state.phaseIndex() > 0 || aligned;
        if (!bursting && state.ticksInProgram() >= MAX_TURN_TICKS) {
            return BankProgram.stop(state, context);
        }
        double yaw = BankProgram.steering(snapshot,
                MAX_YAW_TORQUE * away / FULL_TURN_ERROR_DEGREES,
                MAX_YAW_TORQUE, modulation.turnRate(), away < 0 ? -1.0 : 1.0);
        PrimitiveAction motors;
        if (!bursting || !aligned || BankProgram.avoiding(snapshot) || modulation.speed() == 0.0) {
            motors = BankProgram.braking(snapshot, yaw);
        } else {
            motors = PrimitiveAction.clamped(MAX_FORWARD_THRUST * modulation.speed()
                    * BankProgram.clearanceScale(snapshot), 0.0,
                    CruiseProgram.verticalThrust(snapshot), yaw, 0.0);
        }
        // Zero names the turn phase; positive values count elapsed burst ticks. The burst's
        // clock keeps advancing while an obstacle or moving target prevents forward thrust.
        int nextPhase = bursting ? state.phaseIndex() + 1 : 0;
        return new BodyProgramStep(motors, state.advanced(nextPhase),
                bursting ? state.phaseIndex() / (float) duration : 0.0F, Optional.empty());
    }

    public static int burstTicks(double urgency) {
        return MAX_BURST_TICKS - (int) Math.floor(
                (MAX_BURST_TICKS - MIN_BURST_TICKS) * Math.max(0.0, Math.min(1.0, urgency)));
    }
}
