package dev.neurocraft.fly.gametest;

import dev.neurocraft.fly.body.BodyProgramContext;
import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.BodyProgramState;
import dev.neurocraft.fly.body.BodyProgramStep;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ProgramParameters;
import dev.neurocraft.fly.body.programs.EscapeJumpProgram;
import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import dev.neurocraft.fly.control.PrimitiveAction;
import dev.neurocraft.fly.control.RawPrimitiveAction;
import dev.neurocraft.fly.entity.FlyEntity;
import dev.neurocraft.fly.entity.FlySensors;
import dev.neurocraft.fly.entity.PrimitiveActuator;
import java.util.List;
import java.util.Locale;
import net.minecraft.gametest.framework.GameTestHelper;
import net.minecraft.network.chat.Component;
import net.minecraft.util.Mth;

/**
 * Test-only reference for the ESCAPE_JUMP profile (ADR 0020, programs table entry 8).
 *
 * <p>The two {@code declared*} methods restate the three-phase profile <em>independently of the
 * program</em>: they are written from the phase table, with the brake ramp expressed as
 * {@code (brake - 1 - b) / brake} rather than the program's own {@code 1 - (b + 1) / brake}, so a
 * profile test compares two statements of the same intent instead of the implementation with
 * itself. Only the declared constants are shared, which is the point of naming them.
 *
 * <p>It also owns the lane's declared bounds and the assertions written against them.
 * {@link #raySnapshot} builds a synthetic sample so the ray-dependent branches can be swept without
 * placing a block for every combination.
 *
 * <p><strong>Nothing here touches a level.</strong> World-anchored claims — where a body stands,
 * which side a ray is on, which way a heading actually turned — belong to
 * {@link EscapeArenaFence} and, under it, to {@link RaySideFence} and the engine's own tables.
 */
final class EscapeProfileFence {
    /** Motor channels recorded per tick by {@link #motorWords}. */
    static final int MOTOR_CHANNELS = 5;

    /** Ray distances are exact; this only absorbs double rounding in the brake ramp. */
    static final double TOLERANCE = 1.0e-12;

    /** Comparison bound for the published float phase channel; the motors use a double bound. */
    static final float PHASE_TOLERANCE = 1.0E-6F;

    private EscapeProfileFence() {}

    // ---- declared bounds ----------------------------------------------------------------

    /**
     * Modulations the envelope sweep runs, including two a caller asked for outside {@code [0, 1]}.
     * Those two are the point: a modulation scalar can never widen a bound (ADR 0020 section 4),
     * and the clamp in {@link Modulation}'s compact constructor is the whole of that guarantee.
     */
    static final List<Modulation> ENVELOPE_MODULATIONS = List.of(
            Modulation.NEUTRAL,
            new Modulation(0.0, 0.0, 0.0, 0.0),
            new Modulation(0.5, 0.5, 0.5, 0.5),
            new Modulation(1.0, 1.0, 1.0, 1.0),
            new Modulation(5.0, 5.0, 5.0, 5.0),
            new Modulation(-3.0, -3.0, -3.0, -3.0));

    /**
     * Ray-and-velocity configurations the envelope sweep runs: fully open at rest, contact on every
     * side at once, each horizontal side alone at the closest a ray can report outside the body's
     * own hull, a low ceiling, an asymmetric near threat, the exact threat-range boundary, and two
     * bodies arriving with vertical velocity in each direction so the brake branch is swept.
     */
    static final List<PhysicalSensorSnapshot> ENVELOPE_ARENAS = List.of(
            raySnapshot(6.0, 6.0, 6.0),
            raySnapshot(0.0, 0.0, 0.0),
            raySnapshot(0.19, 6.0, 6.0),
            raySnapshot(6.0, 0.19, 6.0),
            raySnapshot(6.0, 6.0, 0.19),
            arenaSnapshot(6.0, 6.0, 6.0, 0.5, 0.0),
            raySnapshot(1.5, 2.5, 2.5),
            raySnapshot(3.0, 3.0, 3.0),
            arenaSnapshot(6.0, 6.0, 6.0, 6.0, 0.6),
            arenaSnapshot(6.0, 6.0, 6.0, 6.0, -0.6));

    // ---- shared assertions ---------------------------------------------------------------

    /**
     * Sweeps {@code urgency} across its whole range in hundredths and holds the reshaping to three
     * properties: the total is <strong>always</strong> {@code ESCAPE_JUMP_TICKS}, no phase is ever
     * longer than at a lower urgency, and no phase is shorter than its own declared floor.
     * Hundredths rather than the two endpoints because {@code Math.round} is a step function — the
     * endpoints alone would pass for a reshaping that jumped around in between.
     */
    static void assertReshapingIsMonotoneAndFloored(GameTestHelper helper) {
        int previousCrouch = EscapeJumpProgram.ESCAPE_CROUCH_TICKS;
        int previousLaunch = EscapeJumpProgram.ESCAPE_LAUNCH_TICKS;
        for (int percent = 0; percent <= 100; percent++) {
            int crouch = EscapeJumpProgram.crouchTicks(percent);
            int launch = EscapeJumpProgram.launchTicks(percent);
            int brake = EscapeJumpProgram.brakeTicks(percent);
            String where = " at urgency " + percent + "%: " + crouch + "/" + launch + "/" + brake;
            helper.assertTrue(
                    crouch + launch + brake == EscapeJumpProgram.ESCAPE_JUMP_TICKS,
                    Component.literal("the reshaped phases do not sum to the fixed total" + where));
            helper.assertTrue(
                    crouch <= previousCrouch && launch <= previousLaunch,
                    Component.literal("the reshaping is not monotone" + where));
            helper.assertTrue(
                    crouch >= EscapeJumpProgram.ESCAPE_CROUCH_MIN_TICKS
                            && launch >= EscapeJumpProgram.ESCAPE_LAUNCH_MIN_TICKS,
                    Component.literal("the reshaping passed a declared floor" + where));
            previousCrouch = crouch;
            previousLaunch = launch;
        }
    }

    /** Every motor inside the rails, inside the program's own narrower declared bands, and typed. */
    static void assertInsideTheDeclaredEnvelope(
            GameTestHelper helper,
            BodyProgramStep step,
            PhysicalSensorSnapshot arena,
            boolean braking,
            String where) {
        PrimitiveAction motors = step.motors();
        assertUnitRange(helper, motors.forwardThrust(), "forward" + where);
        assertUnitRange(helper, motors.lateralThrust(), "lateral" + where);
        assertUnitRange(helper, motors.verticalThrust(), "vertical" + where);
        assertUnitRange(helper, motors.yawTorque(), "yaw" + where);
        assertUnitRange(helper, motors.pitchTorque(), "pitch" + where);
        helper.assertTrue(
                motors.lateralThrust() == 0.0 && motors.pitchTorque() == 0.0,
                Component.literal("ESCAPE_JUMP owns no lateral or pitch axis" + where));
        helper.assertTrue(
                motors.forwardThrust() >= 0.0
                        && motors.forwardThrust()
                                <= EscapeJumpProgram.ESCAPE_LAUNCH_FORWARD + TOLERANCE,
                Component.literal("forward left its declared band" + where));
        helper.assertTrue(
                Math.abs(motors.yawTorque())
                        <= EscapeJumpProgram.ESCAPE_TURN_TORQUE + TOLERANCE,
                Component.literal("yaw exceeded the declared away-turn bound" + where));
        // Sharper than a vertical band, and scoped to the phase that claims it: the BRAKE never
        // adds to the body's existing vertical speed, whatever that speed is. The crouch and the
        // launch make no such claim -- a launch is a deliberate acceleration, and asserting
        // otherwise would be asserting the program does not jump.
        if (braking) {
            helper.assertTrue(
                    motors.verticalThrust() * arena.velocityYBlocksPerTick() <= TOLERANCE,
                    Component.literal("the brake reinforced the body's vertical speed" + where));
        } else {
            helper.assertTrue(
                    motors.verticalThrust() >= EscapeJumpProgram.ESCAPE_CROUCH_VERTICAL - TOLERANCE
                            && motors.verticalThrust()
                                    <= EscapeJumpProgram.ESCAPE_LAUNCH_VERTICAL + TOLERANCE,
                    Component.literal("crouch or launch left its declared band" + where));
        }
        helper.assertTrue(
                step.presentationPhase() >= 0.0F && step.presentationPhase() < 1.0F,
                Component.literal("presentation phase left [0, 1)" + where));
    }

    /** Applies the step to a real body and holds the result to the actuator's own two rails. */
    static void assertInsideTheActuatorRails(
            GameTestHelper helper, FlyEntity fly, BodyProgramStep step, String where) {
        float yawBefore = fly.getYRot();
        PrimitiveActuator.apply(fly, RawPrimitiveAction.ZERO, step.motors());
        helper.assertTrue(
                fly.getDeltaMovement().length()
                        <= PrimitiveActuator.MAX_SPEED_BLOCKS_PER_TICK + 1.0e-9,
                Component.literal("the jump drove the body past the speed cap" + where));
        // Wrapped, not subtracted. The sweep turns the body through hundreds of degrees and the
        // actuator wraps yaw into (-180, 180]: a plain difference reads about 360 the moment it
        // crosses, which is a defect in the measurement rather than in the program.
        helper.assertTrue(
                Math.abs(Mth.wrapDegrees(fly.getYRot() - yawBefore))
                        <= PrimitiveActuator.MAX_YAW_DELTA_DEGREES_PER_TICK + 1.0e-6,
                Component.literal("the jump turned faster than the yaw rate" + where));
    }

    // ---- the independently declared profile ------------------------------------------------

    /**
     * The declared vertical thrust at one tick, for a body <strong>at rest</strong>.
     *
     * <p>The brake phase is a function of the body's vertical velocity rather than of the tick, so
     * a tick-indexed statement of it only exists for a resting body — where the brake correctly
     * asks for nothing. {@code escapeJumpBrakeOpposesVerticalSpeed} pins the moving case.
     */
    static double declaredVerticalAtRest(int tick, int crouchTicks, int launchTicks) {
        if (tick < crouchTicks) {
            return EscapeJumpProgram.ESCAPE_CROUCH_VERTICAL;
        }
        return tick < crouchTicks + launchTicks ? EscapeJumpProgram.ESCAPE_LAUNCH_VERTICAL : 0.0;
    }

    /** The declared forward thrust at one tick, for a launch peak of {@code peakForward}. */
    static double declaredForward(int tick, int crouchTicks, int launchTicks, double peakForward) {
        if (tick < crouchTicks) {
            return 0.0;
        }
        if (tick < crouchTicks + launchTicks) {
            return peakForward;
        }
        int brakeTicks = EscapeJumpProgram.ESCAPE_JUMP_TICKS - crouchTicks - launchTicks;
        int brakeTick = tick - crouchTicks - launchTicks;
        return peakForward * (brakeTicks - 1 - brakeTick) / (double) brakeTicks;
    }

    // ---- synthetic samples -----------------------------------------------------------------

    /** A sample carrying three horizontal rays, an open ceiling, and a body at rest. */
    static PhysicalSensorSnapshot raySnapshot(double front, double left, double right) {
        return arenaSnapshot(front, left, right, FlySensors.RAY_LENGTH_BLOCKS, 0.0);
    }

    /** A sample carrying three horizontal rays, an up ray, and a vertical velocity. */
    static PhysicalSensorSnapshot arenaSnapshot(
            double front, double left, double right, double up, double velocityY) {
        return new PhysicalSensorSnapshot(
                7L, "escape-jump-fence", 0.0, 0.0, 0.0,
                0.0, velocityY, 0.0, 0.0, 0.0,
                false, false, false, false, false, 0, 0,
                FlySensors.RAY_LENGTH_BLOCKS,
                front, FlySensors.RAY_LENGTH_BLOCKS, left, right, up,
                FlySensors.RAY_LENGTH_BLOCKS);
    }

    /** A fixed context, so a step sequence taken from it is a pure function of state alone. */
    static BodyProgramContext context(PhysicalSensorSnapshot snapshot) {
        return BodyProgramContext.sensorsOnly(snapshot);
    }

    /** Raw bits of the five motor channels over one whole profile from a fixed context. */
    static long[] motorWords(BodyProgramContext context, Modulation modulation) {
        EscapeJumpProgram program = new EscapeJumpProgram();
        BodyProgramState state = BodyProgramState.entering(BodyProgramId.ESCAPE_JUMP, 0L);
        long[] words = new long[EscapeJumpProgram.ESCAPE_JUMP_TICKS * MOTOR_CHANNELS];
        for (int tick = 0; tick < EscapeJumpProgram.ESCAPE_JUMP_TICKS; tick++) {
            BodyProgramStep step =
                    program.step(state, context, modulation, ProgramParameters.NEUTRAL);
            PrimitiveAction motors = step.motors();
            int base = tick * MOTOR_CHANNELS;
            words[base] = Double.doubleToRawLongBits(motors.forwardThrust());
            words[base + 1] = Double.doubleToRawLongBits(motors.lateralThrust());
            words[base + 2] = Double.doubleToRawLongBits(motors.verticalThrust());
            words[base + 3] = Double.doubleToRawLongBits(motors.yawTorque());
            words[base + 4] = Double.doubleToRawLongBits(motors.pitchTorque());
            state = step.next();
        }
        return words;
    }

    /**
     * The clearance scale the program actually latches for a ray: the declared scaling, quantized
     * to {@code SCALE_UNIT} exactly as {@code latch} quantizes it.
     *
     * <p>This exists so a test can compare an <em>emitted motor</em> against a value derived from
     * the <em>sensed ray</em>. The review found both scalings unpinned precisely because the only
     * assertion on them compared one restatement of the formula with another and never looked at
     * what the program put on the wire.
     */
    static double quantizedScale(double rayBlocks) {
        return Math.round(
                        EscapeJumpProgram.clearanceScale(rayBlocks) * EscapeJumpProgram.SCALE_UNIT)
                / (double) EscapeJumpProgram.SCALE_UNIT;
    }

    static void assertClose(
            GameTestHelper helper, double actual, double expected, String label) {
        helper.assertTrue(
                Math.abs(actual - expected) <= TOLERANCE,
                Component.literal(String.format(
                        Locale.ROOT, "%s was %.17f, declared %.17f", label, actual, expected)));
    }

    static void assertUnitRange(GameTestHelper helper, double value, String label) {
        helper.assertTrue(
                value >= -1.0 && value <= 1.0,
                Component.literal("motor left the rails: " + label + " = " + value));
    }
}
