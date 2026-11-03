package dev.neurocraft.fly.gametest;

import dev.neurocraft.fly.NeuroCraftFly;
import dev.neurocraft.fly.body.BodyProgramContext;
import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.BodyProgramRuntime;
import dev.neurocraft.fly.body.BodyProgramState;
import dev.neurocraft.fly.body.BodyProgramStep;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ProgramParameters;
import dev.neurocraft.fly.body.SelectionSource;
import dev.neurocraft.fly.body.programs.EscapeJumpProgram;
import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import dev.neurocraft.fly.control.PrimitiveAction;
import dev.neurocraft.fly.control.RawPrimitiveAction;
import dev.neurocraft.fly.entity.FlyEntity;
import dev.neurocraft.fly.entity.FlySensors;
import dev.neurocraft.fly.entity.PrimitiveActuator;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import net.fabricmc.fabric.api.gametest.v1.GameTest;
import net.minecraft.gametest.framework.GameTestHelper;
import net.minecraft.network.chat.Component;
import net.minecraft.world.phys.Vec3;

/**
 * ADR 0020 lane D pins for {@code ESCAPE_JUMP}: the scripted, bounded, self-arresting jump.
 *
 * <p><strong>Nothing here is evidence about the connectome.</strong> These tests measure a hand
 * written script against its own declared profile and against the world the mod actually simulates.
 * The escape readout that will select this program in stage 4 is a hand-chosen mapping (ADR 0019),
 * and a passing test in this file says only that the script does what it says.
 *
 * <p>Everything that depends on the world is driven one authoritative tick at a time through
 * {@link EscapeArenaFence}'s runners, so the physics between steps is real. The pure branches are
 * driven as the pure function
 * {@code program.step(state, context, modulation, parameters)} is.
 */
public final class EscapeJumpProgramGameTests {
    // ---- a. the declared three-phase profile ---------------------------------------------

    /**
     * The per-tick motor sequence at neutral modulation is exactly the declared three-phase
     * profile: crouch ticks 0-5 at {@code vertical = -0.30}, launch ticks 6-13 at
     * {@code vertical = +1.0} and {@code forward = +0.60}, brake ticks 14-23 easing forward
     * linearly to zero and asking for no vertical thrust from a body already at rest. The profile
     * is 24 ticks long and its last tick names CRUISE.
     */
    @GameTest(maxTicks = 20)
    public void escapeJumpProfileHasThreeDeclaredPhases(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, EscapeArenaFence.OPEN_PROBE_SPAWN);
        fly.setYRot(0.0F);
        fly.setXRot(0.0F);
        PhysicalSensorSnapshot open = EscapeArenaFence.assertOpenArena(helper, fly);
        BodyProgramContext context = EscapeProfileFence.context(open);

        int crouch = EscapeJumpProgram.ESCAPE_CROUCH_TICKS;
        int launch = EscapeJumpProgram.ESCAPE_LAUNCH_TICKS;
        helper.assertTrue(
                crouch + launch + EscapeJumpProgram.brakeTicks(0)
                        == EscapeJumpProgram.ESCAPE_JUMP_TICKS,
                Component.literal("the declared phases do not sum to the fixed profile length"));
        helper.assertTrue(
                EscapeJumpProgram.clearanceScale(open.rayForwardBlocks()) == 1.0
                        && EscapeJumpProgram.clearanceScale(open.rayUpBlocks()) == 1.0,
                Component.literal("an open arena must give unscaled launch thrust, got front "
                        + open.rayForwardBlocks() + " up " + open.rayUpBlocks()));

        EscapeJumpProgram program = new EscapeJumpProgram();
        BodyProgramState state =
                BodyProgramState.entering(BodyProgramId.ESCAPE_JUMP, open.gameTick());
        for (int tick = 0; tick < EscapeJumpProgram.ESCAPE_JUMP_TICKS; tick++) {
            BodyProgramStep step =
                    program.step(state, context, Modulation.NEUTRAL, ProgramParameters.NEUTRAL);
            PrimitiveAction motors = step.motors();
            String where = " at tick " + tick;
            EscapeProfileFence.assertClose(
                    helper,
                    motors.verticalThrust(),
                    EscapeProfileFence.declaredVerticalAtRest(tick, crouch, launch),
                    "vertical" + where);
            EscapeProfileFence.assertClose(
                    helper,
                    motors.forwardThrust(),
                    EscapeProfileFence.declaredForward(
                            tick, crouch, launch, EscapeJumpProgram.ESCAPE_LAUNCH_FORWARD),
                    "forward" + where);
            helper.assertTrue(
                    motors.lateralThrust() == 0.0
                            && motors.pitchTorque() == 0.0
                            && motors.yawTorque() == 0.0,
                    Component.literal("an open arena commanded a lateral, pitch or yaw" + where));
            helper.assertTrue(
                    Math.abs(step.presentationPhase()
                                    - tick / (float) EscapeJumpProgram.ESCAPE_JUMP_TICKS)
                            <= EscapeProfileFence.PHASE_TOLERANCE,
                    Component.literal("presentation phase was " + step.presentationPhase() + where));

            boolean last = tick == EscapeJumpProgram.ESCAPE_JUMP_TICKS - 1;
            helper.assertTrue(
                    step.exitTo().equals(
                            last ? Optional.of(BodyProgramId.CRUISE) : Optional.empty()),
                    Component.literal("the profile exit is wrong" + where + ": " + step.exitTo()));
            state = step.next();
        }
        helper.assertTrue(
                state.ticksInProgram() == EscapeJumpProgram.ESCAPE_JUMP_TICKS,
                Component.literal("the profile did not advance its own clock monotonically"));
        helper.succeed();
    }

    // ---- b. the jump leaves the ground and stops climbing --------------------------------

    /**
     * Driven against the real actuator and the real passive physics, the profile lifts a grounded
     * body by at least the declared minimum, hands over with its vertical axis essentially
     * stopped, and never comes within the declared clearance of the lid of the sealed box.
     *
     * <p>The exit-speed assertion is the review's H3 in one line. Before the brake phase the
     * profile handed over at 0.394 blocks/tick with commanded thrust already zero, which under drag
     * 0.98 and no ambient gravity coasts roughly twenty further blocks — a floor jump in the demo
     * arena ending in the ceiling, which is the live failure this program exists to cure.
     */
    @GameTest(maxTicks = 90)
    public void escapeJumpLiftsAGroundedFlyAndStopsClimbing(GameTestHelper helper) {
        EscapeArenaFence.layFloor(helper);
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, EscapeArenaFence.GROUNDED_SPAWN);
        fly.setYRot(0.0F);
        fly.setXRot(0.0F);
        EscapeArenaFence.JumpRun run =
                new EscapeArenaFence.JumpRun(helper, fly, Modulation.NEUTRAL, true);

        helper.onEachTick(() -> {
            if (!run.advance()) {
                return;
            }
            helper.assertTrue(
                    run.rise >= EscapeArenaFence.ESCAPE_JUMP_MIN_RISE_BLOCKS,
                    Component.literal(String.format(
                            Locale.ROOT,
                            "the jump rose %.6f blocks, below the declared minimum %.4f",
                            run.rise,
                            EscapeArenaFence.ESCAPE_JUMP_MIN_RISE_BLOCKS)));
            helper.assertFalse(
                    fly.onGround(), Component.literal("the jump never left the ground"));
            helper.assertTrue(
                    run.exitVerticalSpeed
                            <= EscapeArenaFence.ESCAPE_EXIT_VERTICAL_SPEED_BLOCKS_PER_TICK,
                    Component.literal(String.format(
                            Locale.ROOT,
                            "the jump handed over still moving vertically at %.6f blocks/tick,"
                                    + " above the declared %.4f",
                            run.exitVerticalSpeed,
                            EscapeArenaFence.ESCAPE_EXIT_VERTICAL_SPEED_BLOCKS_PER_TICK)));
            helper.assertTrue(
                    run.minUpRay < FlySensors.RAY_LENGTH_BLOCKS,
                    Component.literal("the up ray never saw the lid, so the clearance claim is"
                            + " empty; minimum was " + run.minUpRay));
            helper.assertTrue(
                    run.minUpRay > EscapeArenaFence.ESCAPE_CEILING_CLEARANCE_BLOCKS,
                    Component.literal(String.format(
                            Locale.ROOT,
                            "the jump closed to %.6f blocks of the lid, inside the declared %.4f",
                            run.minUpRay,
                            EscapeArenaFence.ESCAPE_CEILING_CLEARANCE_BLOCKS)));
            helper.succeed();
        });
    }

    // ---- c. the jump is directed away from the threat, and commits ------------------------

    /**
     * The away-turn points the body at the open side, in world space, and never dithers.
     *
     * <p>The verdict is never taken from the sign of a variable. Each case reads the actual ray
     * distances first, drives crouch and launch <strong>one authoritative tick at a time</strong>,
     * and then asks whether the body's world heading swung toward the open direction — where "open"
     * is the body's left or right as {@link RaySideFence} derives it from the engine's own
     * {@code Direction.getClockWise()} table.
     *
     * <p>The fourth case is the review's H1: an obstacle sitting just outside the right ray at
     * entry, which a right turn sweeps that ray onto. Before the side latch this produced nine to
     * eleven sign flips per jump and a swing of 0.063 — below the declared floor. The sign-flip
     * count is asserted at zero, so the pin fails on dithering rather than only on the outcome.
     */
    @GameTest(maxTicks = 120)
    public void escapeJumpTurnsAwayFromTheShortestRayWithoutDithering(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, EscapeArenaFence.OPEN_PROBE_SPAWN);
        EscapeArenaFence.TurnRun run = new EscapeArenaFence.TurnRun(helper, fly);
        int[] index = {0};
        boolean[] running = {false};

        helper.onEachTick(() -> {
            if (index[0] >= EscapeArenaFence.TURN_CASES.size()) {
                assertSideBranchesAndTies(helper);
                helper.succeed();
                return;
            }
            EscapeArenaFence.TurnCase probe = EscapeArenaFence.TURN_CASES.get(index[0]);
            if (!running[0]) {
                run.setUp(probe);
                running[0] = true;
                return;
            }
            if (!run.advance()) {
                return;
            }
            double swing = run.towardnessAfter - run.towardnessBefore;
            helper.assertTrue(
                    run.signFlips == 0,
                    Component.literal(probe.name() + ": the away-turn changed sign "
                            + run.signFlips + " times; a latched reflex commits"));
            if (probe.frontWallX() > 0) {
                EscapeArenaFence.assertLaunchThrustIsScaledByFrontClearance(
                        helper, run, probe.name());
                helper.assertTrue(
                        swing >= EscapeArenaFence.ESCAPE_MIN_HEADING_SWING,
                        Component.literal(String.format(
                                Locale.ROOT,
                                "%s: the heading swung %.6f toward the open %s, below the declared"
                                        + " %.4f (from %.6f to %.6f)",
                                probe.name(),
                                swing,
                                run.openSide,
                                EscapeArenaFence.ESCAPE_MIN_HEADING_SWING,
                                run.towardnessBefore,
                                run.towardnessAfter)));
            } else {
                helper.assertTrue(
                        swing == 0.0,
                        Component.literal("an open arena turned the body by " + swing));
            }
            index[0]++;
            running[0] = false;
        });
    }

    /** The side branch and the tie rule, on synthetic samples: cheap, exhaustive, world-free. */
    private static void assertSideBranchesAndTies(GameTestHelper helper) {
        helper.assertTrue(
                EscapeJumpProgram.threatSide(EscapeProfileFence.raySnapshot(6.0, 1.5, 6.0))
                                == EscapeJumpProgram.SIDE_RIGHT
                        && EscapeJumpProgram.threatSide(
                                        EscapeProfileFence.raySnapshot(6.0, 6.0, 1.5))
                                == EscapeJumpProgram.SIDE_LEFT,
                Component.literal("a short side ray did not commit away from that side"));
        helper.assertTrue(
                EscapeJumpProgram.threatSide(EscapeProfileFence.raySnapshot(1.5, 2.0, 2.0))
                                == EscapeJumpProgram.SIDE_RIGHT
                        && EscapeJumpProgram.threatSide(
                                        EscapeProfileFence.raySnapshot(2.0, 2.0, 2.0))
                                == EscapeJumpProgram.SIDE_RIGHT,
                Component.literal("a symmetric threat did not resolve to the declared right turn"));
        helper.assertTrue(
                EscapeJumpProgram.threatSide(EscapeProfileFence.raySnapshot(6.0, 6.0, 6.0))
                        == EscapeJumpProgram.SIDE_NONE,
                Component.literal("an open arena committed to a side"));
    }

    // ---- d. urgency reshapes the profile without changing its length ----------------------

    /**
     * {@code urgency} is the one scalar the program reads, and after the review it no longer
     * changes how long the jump lasts: the total is always {@code ESCAPE_JUMP_TICKS}, and urgency
     * only moves ticks out of the crouch and the launch and into the brake.
     *
     * <p>That is what keeps the program's duration equal to stage 4b's 24-tick hold. With a
     * shortening duration, a policy re-sending its urgency every frame could drive the program to
     * an early exit and be re-selected into a second jump inside one hold.
     */
    @GameTest(maxTicks = 20)
    public void escapeJumpUrgencyReshapesWithoutChangingTheDuration(GameTestHelper helper) {
        helper.assertTrue(
                EscapeJumpProgram.crouchTicks(0) == EscapeJumpProgram.ESCAPE_CROUCH_TICKS
                        && EscapeJumpProgram.launchTicks(0) == EscapeJumpProgram.ESCAPE_LAUNCH_TICKS
                        && EscapeJumpProgram.brakeTicks(0) == 10,
                Component.literal("zero urgency did not give the declared 6/8/10 profile"));
        helper.assertTrue(
                EscapeJumpProgram.crouchTicks(100) == EscapeJumpProgram.ESCAPE_CROUCH_MIN_TICKS
                        && EscapeJumpProgram.launchTicks(100)
                                == EscapeJumpProgram.ESCAPE_LAUNCH_MIN_TICKS
                        && EscapeJumpProgram.brakeTicks(100) == 15,
                Component.literal("full urgency did not give the declared 4/5/15 profile"));
        EscapeProfileFence.assertReshapingIsMonotoneAndFloored(helper);

        // The reshaped profile still runs the full 24 ticks and still exits exactly once.
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, EscapeArenaFence.OPEN_PROBE_SPAWN);
        fly.setYRot(0.0F);
        fly.setXRot(0.0F);
        BodyProgramContext context =
                EscapeProfileFence.context(EscapeArenaFence.assertOpenArena(helper, fly));
        Modulation urgent = new Modulation(1.0, 1.0, 1.0, 1.0);
        EscapeJumpProgram program = new EscapeJumpProgram();
        BodyProgramState state = BodyProgramState.entering(BodyProgramId.ESCAPE_JUMP, 0L);
        for (int tick = 0; tick < EscapeJumpProgram.ESCAPE_JUMP_TICKS; tick++) {
            BodyProgramStep step =
                    program.step(state, context, urgent, ProgramParameters.NEUTRAL);
            boolean last = tick == EscapeJumpProgram.ESCAPE_JUMP_TICKS - 1;
            helper.assertTrue(
                    step.exitTo().equals(
                            last ? Optional.of(BodyProgramId.CRUISE) : Optional.empty()),
                    Component.literal("the reshaped profile exits at the wrong tick " + tick));
            EscapeProfileFence.assertClose(
                    helper,
                    step.motors().verticalThrust(),
                    EscapeProfileFence.declaredVerticalAtRest(
                            tick,
                            EscapeJumpProgram.ESCAPE_CROUCH_MIN_TICKS,
                            EscapeJumpProgram.ESCAPE_LAUNCH_MIN_TICKS),
                    "reshaped vertical at tick " + tick);
            state = step.next();
        }

        // A policy that changes urgency mid-jump cannot move the boundaries under the body: the
        // entry urgency is latched in the state word, so a sequence that alternates 0 and 1 gives
        // the same motors as a constant 0.
        long[] steady = EscapeProfileFence.motorWords(context, Modulation.NEUTRAL);
        long[] alternating = new long[steady.length];
        BodyProgramState latched = BodyProgramState.entering(BodyProgramId.ESCAPE_JUMP, 0L);
        for (int tick = 0; tick < EscapeJumpProgram.ESCAPE_JUMP_TICKS; tick++) {
            BodyProgramStep step = program.step(
                    latched,
                    context,
                    tick % 2 == 0 ? Modulation.NEUTRAL : urgent,
                    ProgramParameters.NEUTRAL);
            PrimitiveAction motors = step.motors();
            int base = tick * EscapeProfileFence.MOTOR_CHANNELS;
            alternating[base] = Double.doubleToRawLongBits(motors.forwardThrust());
            alternating[base + 1] = Double.doubleToRawLongBits(motors.lateralThrust());
            alternating[base + 2] = Double.doubleToRawLongBits(motors.verticalThrust());
            alternating[base + 3] = Double.doubleToRawLongBits(motors.yawTorque());
            alternating[base + 4] = Double.doubleToRawLongBits(motors.pitchTorque());
            latched = step.next();
        }
        helper.assertTrue(
                Arrays.equals(steady, alternating),
                Component.literal("urgency arriving mid-jump reshaped a profile already running"));
        helper.succeed();
    }

    // ---- e. the three unread scalars are structurally ignored -----------------------------

    /**
     * Varying {@code speed}, {@code turnRate}, or {@code amplitude} produces a bit-identical motor
     * sequence, because the program does not read them; varying {@code urgency} at entry does not,
     * because it does.
     */
    @GameTest(maxTicks = 20)
    public void escapeJumpSpeedTurnRateAndAmplitudeAreStructurallyIgnored(GameTestHelper helper) {
        // A threat is present, so the ray-dependent branches are live and cannot mask a scalar
        // that only shows up in a turn or in the scaled launch thrust.
        BodyProgramContext context =
                EscapeProfileFence.context(EscapeProfileFence.raySnapshot(1.5, 6.0, 2.5));
        long[] neutral = EscapeProfileFence.motorWords(context, Modulation.NEUTRAL);
        helper.assertTrue(
                Arrays.equals(neutral, EscapeProfileFence.motorWords(context, Modulation.NEUTRAL)),
                Component.literal("ESCAPE_JUMP is not deterministic across identical runs"));

        List<Modulation> unread = List.of(
                new Modulation(0.0, 1.0, 1.0, 0.0),
                new Modulation(1.0, 0.0, 1.0, 0.0),
                new Modulation(1.0, 1.0, 0.0, 0.0),
                new Modulation(0.0, 0.25, 0.5, 0.0));
        for (Modulation modulation : unread) {
            helper.assertTrue(
                    Arrays.equals(neutral, EscapeProfileFence.motorWords(context, modulation)),
                    Component.literal(
                            "ESCAPE_JUMP read a modulation scalar it does not declare: "
                                    + modulation));
        }
        helper.assertFalse(
                Arrays.equals(
                        neutral,
                        EscapeProfileFence.motorWords(
                                context, new Modulation(1.0, 1.0, 1.0, 1.0))),
                Component.literal("ESCAPE_JUMP ignored the urgency scalar it does declare"));
        helper.succeed();
    }

    // ---- f. the actuation envelope --------------------------------------------------------

    /**
     * Every motor the program emits is inside the rails, for every modulation — including values a
     * caller asked for outside {@code [0, 1]} — and for every ray-and-velocity configuration,
     * including contact. Lateral and pitch are exactly zero, and the vertical channel never adds
     * to the body's existing vertical speed.
     */
    @GameTest(maxTicks = 20)
    public void escapeJumpStaysInsideTheActuationEnvelope(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, EscapeArenaFence.OPEN_PROBE_SPAWN);
        fly.setYRot(53.0F);
        fly.setXRot(0.0F);
        EscapeJumpProgram program = new EscapeJumpProgram();
        for (Modulation modulation : EscapeProfileFence.ENVELOPE_MODULATIONS) {
            int percent = EscapeJumpProgram.urgencyPercent(modulation);
            for (PhysicalSensorSnapshot arena : EscapeProfileFence.ENVELOPE_ARENAS) {
                BodyProgramContext context = EscapeProfileFence.context(arena);
                BodyProgramState state =
                        BodyProgramState.entering(BodyProgramId.ESCAPE_JUMP, 0L);
                fly.setDeltaMovement(Vec3.ZERO);
                // Two profiles' worth, so the clamped over-run past the exit tick is swept too.
                for (int tick = 0; tick < 2 * EscapeJumpProgram.ESCAPE_JUMP_TICKS; tick++) {
                    BodyProgramStep step = program.step(
                            state, context, modulation, ProgramParameters.NEUTRAL);
                    String where = " at tick " + tick + " under " + modulation
                            + " with front ray " + arena.rayForwardBlocks()
                            + " and vy " + arena.velocityYBlocksPerTick();
                    int clamped = Math.min(tick, EscapeJumpProgram.ESCAPE_JUMP_TICKS - 1);
                    boolean braking = clamped
                            >= EscapeJumpProgram.crouchTicks(percent)
                                    + EscapeJumpProgram.launchTicks(percent);
                    EscapeProfileFence.assertInsideTheDeclaredEnvelope(
                            helper, step, arena, braking, where);
                    EscapeProfileFence.assertInsideTheActuatorRails(helper, fly, step, where);
                    state = step.next();
                }
            }
        }
        helper.succeed();
    }

    // ---- g. the brake is a real deadbeat brake --------------------------------------------

    /**
     * The brake reference equals the actuator's own thrust rail, so the brake is deadbeat rather
     * than approximately so, and the brake branch opposes whatever vertical speed it finds.
     *
     * <p>The reference is restated inside the body package because
     * {@code BodyPackageBoundaryTest} does not admit {@code PrimitiveActuator} there. This is the
     * pin that stops the restatement drifting.
     */
    @GameTest(maxTicks = 20)
    public void escapeJumpBrakeReferenceMatchesTheRail(GameTestHelper helper) {
        helper.assertTrue(
                EscapeJumpProgram.ESCAPE_BRAKE_REFERENCE_SPEED_BLOCKS_PER_TICK
                        == PrimitiveActuator.MAX_THRUST_DELTA_BLOCKS_PER_TICK,
                Component.literal("the declared brake reference has drifted from the actuator rail:"
                        + " " + EscapeJumpProgram.ESCAPE_BRAKE_REFERENCE_SPEED_BLOCKS_PER_TICK
                        + " against " + PrimitiveActuator.MAX_THRUST_DELTA_BLOCKS_PER_TICK));
        for (double speed : new double[] {-0.6, -0.04, -0.01, 0.0, 0.01, 0.04, 0.6}) {
            double thrust = EscapeJumpProgram.brakeVertical(speed);
            helper.assertTrue(
                    thrust * speed <= 0.0,
                    Component.literal("the brake reinforced a vertical speed of " + speed));
            double applied = Math.max(-1.0, Math.min(1.0, thrust));
            helper.assertTrue(
                    Math.abs(speed) > PrimitiveActuator.MAX_THRUST_DELTA_BLOCKS_PER_TICK
                            || Math.abs(
                                            speed
                                                    + applied
                                                            * PrimitiveActuator
                                                                    .MAX_THRUST_DELTA_BLOCKS_PER_TICK)
                                    <= EscapeProfileFence.TOLERANCE,
                    Component.literal("the brake is not deadbeat inside the rail at " + speed));
        }
        helper.succeed();
    }

    /**
     * From a mid-height airborne start, the profile ends with the vertical axis stopped and the lid
     * unreached — the case the review's probe ran, where the unbraked profile ended tick 23 at
     * y = 7.66 under an 8.0 lid and still rising.
     */
    @GameTest(maxTicks = 60)
    public void escapeJumpFromMidHeightEndsWithTheVerticalAxisStopped(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, EscapeArenaFence.OPEN_PROBE_SPAWN);
        fly.setYRot(0.0F);
        fly.setXRot(0.0F);
        EscapeArenaFence.JumpRun run =
                new EscapeArenaFence.JumpRun(helper, fly, Modulation.NEUTRAL, false);

        helper.onEachTick(() -> {
            if (!run.advance()) {
                return;
            }
            helper.assertTrue(
                    run.exitVerticalSpeed
                            <= EscapeArenaFence.ESCAPE_EXIT_VERTICAL_SPEED_BLOCKS_PER_TICK,
                    Component.literal(String.format(
                            Locale.ROOT,
                            "a mid-height jump handed over at %.6f blocks/tick, above %.4f",
                            run.exitVerticalSpeed,
                            EscapeArenaFence.ESCAPE_EXIT_VERTICAL_SPEED_BLOCKS_PER_TICK)));
            helper.assertTrue(
                    run.minUpRay > EscapeArenaFence.ESCAPE_CEILING_CLEARANCE_BLOCKS
                            && run.minUpRay < FlySensors.RAY_LENGTH_BLOCKS,
                    Component.literal(String.format(
                            Locale.ROOT,
                            "a mid-height jump closed to %.6f blocks of the lid", run.minUpRay)));
            helper.succeed();
        });
    }

    /**
     * Under a lid inside the threat range, the launch's <em>emitted</em> vertical motor is the
     * declared up-clearance scaling of {@code ESCAPE_LAUNCH_VERTICAL}, and the jump still does not
     * reach the lid.
     *
     * <p>This is the pin the re-review found missing. Every other jump in this lane starts with
     * more than three blocks of headroom, where the scale is 1.0 and a program that ignored the up
     * ray entirely would emit exactly the same motors. The lid here is stone laid by the test, not
     * the structure's barrier: the barrier reads past the 6.0-block ray clamp from anywhere a jump
     * can start.
     */
    @GameTest(maxTicks = 60)
    public void escapeJumpUnderALowLidScalesTheLaunchByUpClearance(GameTestHelper helper) {
        EscapeArenaFence.layLowLid(helper);
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, EscapeArenaFence.LOW_LID_SPAWN);
        fly.setYRot(0.0F);
        fly.setXRot(0.0F);
        EscapeArenaFence.JumpRun run =
                new EscapeArenaFence.JumpRun(helper, fly, Modulation.NEUTRAL, false);

        helper.onEachTick(() -> {
            if (!run.advance()) {
                return;
            }
            EscapeArenaFence.assertLaunchThrustIsScaledByUpClearance(helper, run);
            helper.assertTrue(
                    run.minUpRay > EscapeArenaFence.ESCAPE_CEILING_CLEARANCE_BLOCKS,
                    Component.literal(String.format(
                            Locale.ROOT,
                            "the scaled jump still closed to %.6f blocks of the low lid",
                            run.minUpRay)));
            helper.assertTrue(
                    run.exitVerticalSpeed
                            <= EscapeArenaFence.ESCAPE_EXIT_VERTICAL_SPEED_BLOCKS_PER_TICK,
                    Component.literal(String.format(
                            Locale.ROOT,
                            "the scaled jump handed over at %.6f blocks/tick",
                            run.exitVerticalSpeed)));
            helper.succeed();
        });
    }

    // ---- h. the hand-over to CRUISE --------------------------------------------------------

    /**
     * The exit is a real transition, not a name in a record: driven through
     * {@link BodyProgramRuntime}, the last tick of the profile leaves the runtime running CRUISE,
     * and CRUISE then holds the body without reaching the lid.
     *
     * <p>CRUISE's vertical channel is a rate loop that brakes whatever vertical velocity it
     * inherits, so this is also the outer half of the review's H3: even the residual the brake
     * leaves is absorbed by the successor this program names.
     */
    @GameTest(maxTicks = 140)
    public void escapeJumpHandsOverToCruiseWhichHoldsTheBody(GameTestHelper helper) {
        EscapeArenaFence.layFloor(helper);
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, EscapeArenaFence.GROUNDED_SPAWN);
        fly.setYRot(0.0F);
        fly.setXRot(0.0F);
        BodyProgramRuntime runtime = fly.bodyPrograms();
        runtime.engageScripted(
                BodyProgramId.ESCAPE_JUMP,
                Modulation.NEUTRAL,
                SelectionSource.OPERATOR,
                helper.getLevel().getGameTime());
        int[] driven = {0};
        double[] minUpRay = {Double.MAX_VALUE};
        int cruiseTicks = 80;

        helper.onEachTick(() -> {
            PhysicalSensorSnapshot snapshot = FlySensors.sample(helper.getLevel(), fly);
            minUpRay[0] = Math.min(minUpRay[0], snapshot.rayUpBlocks());
            int tick = driven[0];
            if (tick == EscapeJumpProgram.ESCAPE_JUMP_TICKS) {
                helper.assertTrue(
                        runtime.state().id() == BodyProgramId.CRUISE,
                        Component.literal("the exit did not enter CRUISE; the runtime is running "
                                + runtime.state().id() + " with ledger "
                                + runtime.transitionLedger()));
                helper.assertTrue(
                        runtime.transitionLedger().getLast().contains("-> CRUISE")
                                && runtime.transitionLedger().getLast().contains("reason=exit"),
                        Component.literal("the hand-over was not logged as an exit to CRUISE: "
                                + runtime.transitionLedger()));
            }
            if (tick >= EscapeJumpProgram.ESCAPE_JUMP_TICKS + cruiseTicks) {
                helper.assertTrue(
                        Math.abs(snapshot.velocityYBlocksPerTick())
                                <= EscapeArenaFence.ESCAPE_EXIT_VERTICAL_SPEED_BLOCKS_PER_TICK,
                        Component.literal(String.format(
                                Locale.ROOT,
                                "CRUISE left the body climbing at %.6f blocks/tick",
                                snapshot.velocityYBlocksPerTick())));
                helper.assertTrue(
                        minUpRay[0] > EscapeArenaFence.ESCAPE_CEILING_CLEARANCE_BLOCKS
                                && minUpRay[0] < FlySensors.RAY_LENGTH_BLOCKS,
                        Component.literal(String.format(
                                Locale.ROOT,
                                "the jump and its hand-over closed to %.6f blocks of the lid",
                                minUpRay[0])));
                helper.succeed();
                return;
            }
            Optional<BodyProgramStep> step = runtime.advance(snapshot, true, true);
            helper.assertTrue(
                    step.isPresent(),
                    Component.literal("the engaged program did not own tick " + tick));
            PrimitiveActuator.apply(
                    fly, RawPrimitiveAction.ZERO, step.orElseThrow().motors());
            driven[0]++;
        });
    }
}
