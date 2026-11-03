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
import java.util.Optional;

/**
 * Scripted 40-tick coordinated arc, then HOVER. All limits are chosen engineering constants,
 * not anatomy. Operator-neutral bias deliberately banks RIGHT; nonzero signed bias attenuates
 * and chooses the other direction when negative. Only turnRate is read from modulation.
 * Speed, amplitude and urgency are ignored. Obstacle clearance can override the requested side.
 * The vertical channel is exactly CRUISE's existing floor-band feedback, without extra lift.
 */
public final class BankProgram implements BodyProgram {
    public static final int DURATION_TICKS = 40;
    public static final double MAX_FORWARD_THRUST = 0.35;
    public static final double MAX_LATERAL_THRUST = 0.12;
    public static final double MAX_YAW_TORQUE = 0.45;
    public static final double DEFAULT_TURN_SIGN = 1.0;

    /** Shared local flight clearance constants; no path search or remembered world target. */
    public static final double HARD_CLEARANCE_BLOCKS = 1.5;
    public static final double SLOW_CLEARANCE_BLOCKS = 4.0;
    public static final double AVOID_FORWARD_BLOCKS = 2.0;
    public static final double AVOID_SIDE_BLOCKS = 0.75;
    public static final double SIDE_TIE_BLOCKS = 0.25;

    @Override public BodyProgramId id() { return BodyProgramId.BANK; }
    @Override public boolean canEnter(BodyProgramContext context) { return true; }

    @Override
    public BodyProgramStep step(BodyProgramState state, BodyProgramContext context,
            Modulation modulation, ProgramParameters parameters) {
        if (state.ticksInProgram() >= DURATION_TICKS) {
            return stop(state, context);
        }
        var snapshot = context.snapshot();
        double bias = parameters.turnBias() == 0.0 ? DEFAULT_TURN_SIGN : parameters.turnBias();
        double yaw = steering(snapshot, MAX_YAW_TORQUE * bias,
                MAX_YAW_TORQUE, modulation.turnRate(), Math.signum(bias));
        PrimitiveAction motors;
        if (avoiding(snapshot)) {
            motors = braking(snapshot, yaw);
        } else {
            motors = PrimitiveAction.clamped(MAX_FORWARD_THRUST * clearanceScale(snapshot),
                    MAX_LATERAL_THRUST * bias * modulation.turnRate(),
                    CruiseProgram.verticalThrust(snapshot), yaw, 0.0);
        }
        return new BodyProgramStep(motors, state.advanced(0),
                state.ticksInProgram() / (float) DURATION_TICKS, Optional.empty());
    }

    static double clearanceScale(PhysicalSensorSnapshot snapshot) {
        return Math.max(0.0, Math.min(1.0, (snapshot.rayForwardBlocks() - HARD_CLEARANCE_BLOCKS)
                / (SLOW_CLEARANCE_BLOCKS - HARD_CLEARANCE_BLOCKS)));
    }

    static boolean avoiding(PhysicalSensorSnapshot snapshot) {
        return snapshot.horizontalCollision() || snapshot.rayForwardBlocks() < AVOID_FORWARD_BLOCKS
                || Math.min(snapshot.rayLeftBlocks(), snapshot.rayRightBlocks()) < AVOID_SIDE_BLOCKS;
    }

    /** Steering stays inside the program's own modulated yaw envelope, including avoidance. */
    static double steering(PhysicalSensorSnapshot snapshot, double wanted, double maximum,
            double turnRate, double tieSign) {
        if (avoiding(snapshot)) {
            double difference = snapshot.rayRightBlocks() - snapshot.rayLeftBlocks();
            wanted = maximum * (Math.abs(difference) > SIDE_TIE_BLOCKS
                    ? Math.signum(difference) : tieSign);
        }
        return Math.max(-maximum, Math.min(maximum, wanted)) * turnRate + 0.0;
    }

    /** Horizontal braking is capped at LAND's 0.25 per axis and is independent of modulation. */
    static PrimitiveAction braking(PhysicalSensorSnapshot snapshot, double yaw) {
        PrimitiveAction brake = LandProgram.brakeHorizontal(snapshot, CruiseProgram.verticalThrust(snapshot));
        return PrimitiveAction.clamped(brake.forwardThrust(), brake.lateralThrust(),
                brake.verticalThrust(), yaw, 0.0);
    }

    static BodyProgramStep stop(BodyProgramState state, BodyProgramContext context) {
        return new BodyProgramStep(braking(context.snapshot(), 0.0), state.advanced(0),
                0.0F, Optional.of(BodyProgramId.HOVER));
    }
}
