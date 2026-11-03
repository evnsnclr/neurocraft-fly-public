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
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.gametest.framework.GameTestHelper;
import net.minecraft.network.chat.Component;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.phys.Vec3;

/**
 * Test-only world probe for the ESCAPE_JUMP lane: where the bodies stand, where the walls go, and
 * how a jump and a turn are actually driven and measured (ADR 0020 lane D).
 *
 * <p>Split from {@link EscapeProfileFence}, which is pure: nothing here computes what the profile
 * should be, and nothing there touches a level.
 *
 * <p><strong>Everything below is driven one authoritative tick at a time.</strong> The first draft
 * of this lane applied fourteen actuator steps inside a single callback with nothing integrating
 * between them, which measures a body rotating in place rather than a body escaping, and the review
 * caught it. {@link JumpRun} and {@link TurnRun} are stepped from {@code onEachTick}, so the world
 * moves between steps and the rays the program reads are the rays a real jump would see.
 *
 * <p>Directions come from {@link RaySideFence}, which derives them from the engine's own
 * {@code Direction.getClockWise()} table, and headings from {@code Vec3.directionFromRotation}. No
 * body side and no turn sense is hard-coded here.
 *
 * <p><strong>How the figures quoted in this lane were measured.</strong> Each was read by a
 * temporary {@code assertTrue(false, ...)} placed at <em>exactly</em> the point the passing
 * assertion reads, then removed. Never from a callback firing on a later tick than the assertion:
 * that is the mistake the review found in the first draft's numbers.
 */
final class EscapeArenaFence {
    /**
     * Probe body centred in the 8x8x8 barrier-encased test structure.
     *
     * <p>Measured readings from here at yaw 0, which is not symmetric and which the mirror cases
     * below depend on: <strong>forward (south) 3.500, back (north) 4.500, left (east) 3.500, right
     * (west) 4.500</strong>. So every horizontal ray is outside the program's 3.0-block threat
     * range, and the body's <em>right</em> is the more open side.
     */
    static final Vec3 OPEN_PROBE_SPAWN = new Vec3(4.5, 4.0, 4.5);

    /**
     * Mirror-case body, one block west of centre: left (east) reads 4.500 and right (west) 3.500,
     * so the body's <em>left</em> is the more open side and the same front wall must turn it the
     * other way.
     */
    static final Vec3 LEFT_OPEN_PROBE_SPAWN = new Vec3(3.5, 4.0, 4.5);

    /**
     * Vertical extent of every probe wall, in structure-local blocks.
     *
     * <p>A column, not a single block, and that is load-bearing: the crouch sinks the body about a
     * tenth of a block, and against a one-block wall the forward ray drops under its bottom face
     * partway through the crouch, the threat vanishes, and the turn stops. The review measured
     * exactly that. These columns span 3.0 to 6.0 blocks, which covers the whole vertical excursion
     * of crouch plus launch from {@link #OPEN_PROBE_SPAWN}.
     */
    private static final int WALL_MIN_Y = 3;

    private static final int WALL_MAX_Y = 5;

    /** Wall on the forward ray of {@link #OPEN_PROBE_SPAWN}; its near face is 1.5 blocks away. */
    private static final int FRONT_WALL_CENTRED_X = 4;

    /** Wall on the forward ray of {@link #LEFT_OPEN_PROBE_SPAWN}, at the same 1.5 blocks. */
    private static final int FRONT_WALL_WEST_X = 3;

    private static final int FRONT_WALL_Z = 6;

    /**
     * The dither trap: a column that is <em>off</em> the right ray at entry and sweeps onto it as
     * the body turns right.
     *
     * <p>At yaw 0 the right ray runs due west along {@code z = 4.5}, and this column occupies
     * {@code z in [3, 4]}, so it is invisible at tick 0. A few degrees of right turn swing the ray
     * onto it, which — before the side latch — made the front branch's "more open side" comparison
     * flip and the body oscillate instead of escaping. This placement is the pin for that.
     */
    private static final int LURKER_X = 1;

    private static final int LURKER_Z = 3;

    /** Floor patch for the grounded jump, low in the structure so the lid is as far as it gets. */
    private static final int FLOOR_Y = 0;

    /** Body resting on that floor patch. */
    static final Vec3 GROUNDED_SPAWN = new Vec3(4.5, 1.0, 4.5);

    /** Height of the stone lid the up-clearance probe flies under, in structure-local blocks. */
    private static final int LOW_LID_Y = 5;

    /**
     * Airborne body under that lid.
     *
     * <p>Chosen so the up ray is unambiguously inside the 3.0-block threat range and the scaling
     * is unambiguously below 1: the ray origin sits at {@code 2.5 + 0.11} and the lid's bottom face
     * at 5.0. Measured, the up ray reads 2.39 at entry and <strong>2.633766</strong> at launch
     * start, the crouch having sunk the body about a quarter of a block, so the latched scale is
     * 0.878 and the emitted launch vertical is <strong>0.878000000</strong> rather than the
     * unscaled 1.0. The structure's own barrier lid sits at 8.0 and reads past the 6.0-block ray
     * clamp from anywhere a jump can start, which is exactly why this probe brings its own.
     */
    static final Vec3 LOW_LID_SPAWN = new Vec3(4.5, 2.5, 4.5);

    /** Commanded descent used only to settle the body onto the floor before the reflex starts. */
    private static final PrimitiveAction GROUNDING_DESCENT = new PrimitiveAction(0, 0, -1, 0, 0);

    /**
     * Declared minimum rise of the jump, in blocks, over the fixed 24-tick profile from a grounded
     * start.
     *
     * <p>A floor, not a prediction, and much lower than the pre-review value because the profile now
     * <em>brakes</em>: the jump trades altitude for stopping where it meant to. Measured on the
     * declared constants at the observation tick this pin reads, the rise is
     * <strong>2.231804 blocks</strong> from a grounded start and 1.176737 from the airborne
     * mid-height start; the pre-brake profile rose 5.281624 and was still accelerating.
     */
    static final double ESCAPE_JUMP_MIN_RISE_BLOCKS = 1.5;

    /**
     * Vertical speed, in blocks/tick, the body may still carry when the profile hands over.
     *
     * <p>This is the review's H3 in one number. With drag 0.98 and no ambient gravity a body coasts
     * about {@code v / 0.02} further blocks, so 0.02 blocks/tick is a coast of about one block —
     * inside what CRUISE's own vertical-rate loop absorbs, and two orders below the 0.394
     * blocks/tick the unbraked profile handed over.
     *
     * <p>Measured at the observation tick this pin reads, the deadbeat brake hands over at
     * <strong>0.000000 blocks/tick to six decimals</strong> from the grounded, mid-height and
     * low-lid starts, and the body is still there eighty ticks into CRUISE. It is not always
     * bit-zero: about one sub-rail velocity in nine leaves a 1-ulp residual for a single tick
     * (3.4e-18 at full urgency, bit-zero the tick after), because the deadbeat command is a
     * division that does not always round back exactly. <strong>The claim is the bound, not the
     * literal zero</strong>, and the bound is deliberately looser than the measurement so a
     * physics change degrades it rather than breaking it.
     */
    static final double ESCAPE_EXIT_VERTICAL_SPEED_BLOCKS_PER_TICK = 0.02;

    /**
     * Headroom, in blocks, the up ray must keep for the whole profile and its hand-over.
     *
     * <p>The box is the Fabric default GameTest structure: 8x8x8, encased in barriers because
     * {@code GameTest.skyAccess} defaults to false. The lid sits 7.0 blocks above the floor patch,
     * so the usable interior is 7 blocks rather than the 10 a purpose-built structure would give.
     * The claim is made through the body's own up ray, and the tests additionally require that ray
     * to have <em>seen</em> the lid, so "did not touch the ceiling" cannot be satisfied by open sky.
     *
     * <p>Measured minima at the observation tick: <strong>4.658196</strong> blocks from the
     * grounded start, 2.713263 from the mid-height start, and 1.544467 from under the low lid
     * {@link #LOW_LID_SPAWN} flies beneath, against the pre-brake profile's 1.608376 and rising.
     */
    static final double ESCAPE_CEILING_CLEARANCE_BLOCKS = 0.5;

    /**
     * Minimum heading swing toward the open side over crouch plus launch, as the cosine component
     * of the final heading along the open direction.
     *
     * <p>Re-derived from the <em>dynamic</em> drive after the review: the static rotate-in-place
     * probe it replaced measured a body pivoting on the spot. Measured swings across the three
     * threat cases, in the order {@link #TURN_CASES} runs them: <strong>0.337612, 0.337702,
     * 0.360789</strong>, with zero sign flips in every case. The open control swings exactly
     * 0.000000. With the side latch removed as a negative control, the third case flips seven
     * times on this geometry; the review measured nine to eleven, and a swing of 0.063, on its own.
     */
    static final double ESCAPE_MIN_HEADING_SWING = 0.10;

    private EscapeArenaFence() {}

    // ---- driving a whole jump ---------------------------------------------------------------

    /** One tick-driven ESCAPE_JUMP, from a grounded or an airborne start. */
    static final class JumpRun {
        private final GameTestHelper helper;
        private final FlyEntity fly;
        private final EscapeJumpProgram program = new EscapeJumpProgram();
        private final Modulation modulation;
        private final boolean settleOnGround;
        private final int crouchTicks;
        private BodyProgramState state;
        private int driven;

        double startY = Double.NaN;
        double minUpRay = Double.MAX_VALUE;
        double minDownRay = Double.MAX_VALUE;
        double exitVerticalSpeed = Double.NaN;
        double rise;

        /** The rays the program saw, and the motors it emitted, on the first launch tick. */
        double upRayAtLaunchStart = Double.NaN;

        double frontRayAtLaunchStart = Double.NaN;

        double launchVerticalMotor = Double.NaN;

        double launchForwardMotor = Double.NaN;

        JumpRun(GameTestHelper helper, FlyEntity fly, Modulation modulation, boolean settleOnGround) {
            this.helper = helper;
            this.fly = fly;
            this.modulation = modulation;
            this.settleOnGround = settleOnGround;
            this.crouchTicks =
                    EscapeJumpProgram.crouchTicks(EscapeJumpProgram.urgencyPercent(modulation));
        }

        /** One authoritative tick. Returns true once the profile is done and observed. */
        boolean advance() {
            if (state == null) {
                if (settleOnGround && !fly.onGround()) {
                    PrimitiveActuator.apply(fly, RawPrimitiveAction.ZERO, GROUNDING_DESCENT);
                    return false;
                }
                startY = fly.getY();
                assertOpenArena(helper, fly);
                state = BodyProgramState.entering(
                        BodyProgramId.ESCAPE_JUMP, helper.getLevel().getGameTime());
                return false;
            }
            PhysicalSensorSnapshot snapshot = FlySensors.sample(helper.getLevel(), fly);
            minUpRay = Math.min(minUpRay, snapshot.rayUpBlocks());
            minDownRay = Math.min(minDownRay, snapshot.rayDownBlocks());
            if (driven >= EscapeJumpProgram.ESCAPE_JUMP_TICKS) {
                // One observation tick after the last commanded tick, so the final move has been
                // integrated before the rise and the residual speed are read.
                exitVerticalSpeed = Math.abs(snapshot.velocityYBlocksPerTick());
                rise = fly.getY() - startY;
                return true;
            }
            BodyProgramStep step = program.step(
                    state,
                    BodyProgramContext.sensorsOnly(snapshot),
                    modulation,
                    ProgramParameters.NEUTRAL);
            if (driven == crouchTicks) {
                upRayAtLaunchStart = snapshot.rayUpBlocks();
                frontRayAtLaunchStart = snapshot.rayForwardBlocks();
                launchVerticalMotor = step.motors().verticalThrust();
                launchForwardMotor = step.motors().forwardThrust();
            }
            PrimitiveActuator.apply(fly, RawPrimitiveAction.ZERO, step.motors());
            state = step.next();
            driven++;
            return false;
        }
    }

    // ---- driving a whole turn ----------------------------------------------------------------

    /** One probe case for the away-turn: where the body stands and what is around it. */
    record TurnCase(
            String name,
            Vec3 spawn,
            float yaw,
            int frontWallX,
            boolean lurker,
            boolean rightIsOpen) {}

    /**
     * The four cases. Three exercise the front-threat branch — no threat, right open, left open —
     * and the fourth repeats the right-open case with the dither trap in place.
     */
    static final List<TurnCase> TURN_CASES = List.of(
            // The control faces north, where the barrier is 4.5 blocks away rather than 3.5, so
            // that the launch -- which really does fly the body forward about half a block -- can
            // never bring the front ray inside the threat range and make the premise false. The
            // first draft faced south and the body committed to a real, tiny, correct turn at the
            // very end of the drive, which is the program working, not the control holding.
            new TurnCase("open arena", OPEN_PROBE_SPAWN, 180.0F, 0, false, true),
            new TurnCase(
                    "wall ahead, right open", OPEN_PROBE_SPAWN, 0.0F, FRONT_WALL_CENTRED_X, false, true),
            new TurnCase(
                    "wall ahead, left open",
                    LEFT_OPEN_PROBE_SPAWN, 0.0F, FRONT_WALL_WEST_X, false, false),
            new TurnCase(
                    "wall ahead, right open, lurker",
                    OPEN_PROBE_SPAWN, 0.0F, FRONT_WALL_CENTRED_X, true, true));

    /** One tick-driven crouch-plus-launch, counting the sign flips of the commanded away-turn. */
    static final class TurnRun {
        private final GameTestHelper helper;
        private final FlyEntity fly;
        private final EscapeJumpProgram program = new EscapeJumpProgram();
        private final int driveTicks =
                EscapeJumpProgram.ESCAPE_CROUCH_TICKS + EscapeJumpProgram.ESCAPE_LAUNCH_TICKS;
        private BodyProgramState state;
        private int driven;
        private int lastSign;
        private boolean openControl;

        Direction openSide;
        double towardnessBefore;
        double towardnessAfter;
        int signFlips;
        PhysicalSensorSnapshot entry;

        /** The forward ray the program saw, and the forward motor it emitted, at launch start. */
        double frontRayAtLaunchStart = Double.NaN;

        double launchForwardMotor = Double.NaN;

        TurnRun(GameTestHelper helper, FlyEntity fly) {
            this.helper = helper;
            this.fly = fly;
        }

        /** Places the case's blocks, parks the body, and proves the arena before driving it. */
        void setUp(TurnCase probe) {
            for (int y = WALL_MIN_Y; y <= WALL_MAX_Y; y++) {
                helper.setBlock(new BlockPos(FRONT_WALL_CENTRED_X, y, FRONT_WALL_Z), Blocks.AIR);
                helper.setBlock(new BlockPos(FRONT_WALL_WEST_X, y, FRONT_WALL_Z), Blocks.AIR);
                helper.setBlock(new BlockPos(LURKER_X, y, LURKER_Z), Blocks.AIR);
                if (probe.frontWallX() > 0) {
                    helper.setBlock(new BlockPos(probe.frontWallX(), y, FRONT_WALL_Z), Blocks.STONE);
                }
                if (probe.lurker()) {
                    helper.setBlock(new BlockPos(LURKER_X, y, LURKER_Z), Blocks.STONE);
                }
            }
            park(helper, fly, probe.spawn(), probe.yaw());
            entry = FlySensors.sample(helper.getLevel(), fly);
            openControl = probe.frontWallX() <= 0;
            if (!openControl) {
                assertFrontThreat(helper, entry, probe.rightIsOpen(), probe.name());
            } else {
                assertOpenArena(helper, fly);
            }
            openSide = probe.rightIsOpen()
                    ? RaySideFence.bodyRight(fly.getYRot())
                    : RaySideFence.bodyLeft(fly.getYRot());
            towardnessBefore = towardness(fly.getYRot(), openSide);
            state = BodyProgramState.entering(
                    BodyProgramId.ESCAPE_JUMP, helper.getLevel().getGameTime());
            driven = 0;
            lastSign = 0;
            signFlips = 0;
        }

        /** One authoritative tick. Returns true once crouch plus launch are done. */
        boolean advance() {
            if (driven >= driveTicks) {
                towardnessAfter = towardness(fly.getYRot(), openSide);
                return true;
            }
            PhysicalSensorSnapshot snapshot = FlySensors.sample(helper.getLevel(), fly);
            if (openControl) {
                // The control asserts its own premise every tick. A launch really does fly the
                // body forward, so "the arena stayed open" is a claim about the whole drive and
                // not only about tick 0 -- and if it ever stops holding, this says so rather than
                // reporting a turn the program was right to make.
                assertOpenSample(helper, snapshot, "the open control's arena closed");
            }
            BodyProgramStep step = program.step(
                    state,
                    BodyProgramContext.sensorsOnly(snapshot),
                    Modulation.NEUTRAL,
                    ProgramParameters.NEUTRAL);
            if (openControl) {
                helper.assertTrue(
                        step.motors().yawTorque() == 0.0,
                        Component.literal("an open arena commanded a turn of "
                                + step.motors().yawTorque() + " at tick " + driven));
            }
            int sign = (int) Math.signum(step.motors().yawTorque());
            if (sign != 0 && lastSign != 0 && sign != lastSign) {
                signFlips++;
            }
            if (sign != 0) {
                lastSign = sign;
            }
            if (driven == EscapeJumpProgram.ESCAPE_CROUCH_TICKS) {
                frontRayAtLaunchStart = snapshot.rayForwardBlocks();
                launchForwardMotor = step.motors().forwardThrust();
            }
            PrimitiveActuator.apply(fly, RawPrimitiveAction.ZERO, step.motors());
            state = step.next();
            driven++;
            return false;
        }
    }

    // ---- world assertions ---------------------------------------------------------------------

    /** Lays the floor patch the grounded jump starts from. */
    static void layFloor(GameTestHelper helper) {
        for (int x = 3; x <= 5; x++) {
            for (int z = 3; z <= 5; z++) {
                helper.setBlock(new BlockPos(x, FLOOR_Y, z), Blocks.STONE);
            }
        }
    }

    /** Lays a stone lid low enough that the launch's up-clearance scaling actually engages. */
    static void layLowLid(GameTestHelper helper) {
        for (int x = 3; x <= 5; x++) {
            for (int z = 3; z <= 5; z++) {
                helper.setBlock(new BlockPos(x, LOW_LID_Y, z), Blocks.STONE);
            }
        }
    }

    /** Returns the sample, having proved every horizontal ray is outside the declared threat range. */
    static PhysicalSensorSnapshot assertOpenArena(GameTestHelper helper, FlyEntity fly) {
        PhysicalSensorSnapshot snapshot = FlySensors.sample(helper.getLevel(), fly);
        assertOpenSample(helper, snapshot, "the probe arena is not open");
        return snapshot;
    }

    /** The same claim, about a sample the caller already holds. */
    static void assertOpenSample(
            GameTestHelper helper, PhysicalSensorSnapshot snapshot, String what) {
        helper.assertTrue(
                snapshot.rayForwardBlocks() >= EscapeJumpProgram.ESCAPE_THREAT_RANGE_BLOCKS
                        && snapshot.rayLeftBlocks() >= EscapeJumpProgram.ESCAPE_THREAT_RANGE_BLOCKS
                        && snapshot.rayRightBlocks() >= EscapeJumpProgram.ESCAPE_THREAT_RANGE_BLOCKS,
                Component.literal(String.format(
                        Locale.ROOT,
                        "%s: front %.4f left %.4f right %.4f",
                        what,
                        snapshot.rayForwardBlocks(),
                        snapshot.rayLeftBlocks(),
                        snapshot.rayRightBlocks())));
    }

    /**
     * Proves the front ray is the threat and that the named side really is the more open one,
     * before any verdict is taken from the turn it produces.
     */
    static void assertFrontThreat(
            GameTestHelper helper,
            PhysicalSensorSnapshot snapshot,
            boolean rightIsOpen,
            String where) {
        helper.assertTrue(
                snapshot.rayForwardBlocks() < EscapeJumpProgram.ESCAPE_THREAT_RANGE_BLOCKS
                        && snapshot.rayForwardBlocks() <= snapshot.rayLeftBlocks()
                        && snapshot.rayForwardBlocks() <= snapshot.rayRightBlocks(),
                Component.literal(String.format(
                        Locale.ROOT,
                        "%s: the front ray is not the threat: front %.4f left %.4f right %.4f",
                        where,
                        snapshot.rayForwardBlocks(),
                        snapshot.rayLeftBlocks(),
                        snapshot.rayRightBlocks())));
        helper.assertTrue(
                rightIsOpen
                        ? snapshot.rayRightBlocks() > snapshot.rayLeftBlocks()
                        : snapshot.rayLeftBlocks() > snapshot.rayRightBlocks(),
                Component.literal(String.format(
                        Locale.ROOT,
                        "%s: the side declared open is not the more open one: left %.4f right %.4f",
                        where,
                        snapshot.rayLeftBlocks(),
                        snapshot.rayRightBlocks())));
    }

    /**
     * The launch thrust a wall-facing body commands is strictly less than the open-arena one, and
     * is exactly the declared front-clearance scaling of it. Both halves matter: the first says the
     * jump reacts to the wall at all, the second says it reacts by the declared amount rather than
     * by some other monotone function that would also pass the first.
     */
    static void assertLaunchThrustIsScaledByFrontClearance(
            GameTestHelper helper, TurnRun run, String where) {
        double emitted = run.launchForwardMotor;
        double expected = EscapeJumpProgram.ESCAPE_LAUNCH_FORWARD
                * EscapeProfileFence.quantizedScale(run.frontRayAtLaunchStart);
        helper.assertTrue(
                run.frontRayAtLaunchStart < EscapeJumpProgram.ESCAPE_THREAT_RANGE_BLOCKS,
                Component.literal(String.format(
                        Locale.ROOT,
                        "%s: the wall had left the forward ray by launch start (%.4f), so there is"
                                + " nothing for the scaling to do",
                        where,
                        run.frontRayAtLaunchStart)));
        helper.assertTrue(
                emitted < EscapeJumpProgram.ESCAPE_LAUNCH_FORWARD - EscapeProfileFence.TOLERANCE,
                Component.literal(String.format(
                        Locale.ROOT,
                        "%s: facing a wall did not reduce the emitted launch thrust: %.9f against"
                                + " the unscaled %.9f",
                        where,
                        emitted,
                        EscapeJumpProgram.ESCAPE_LAUNCH_FORWARD)));
        EscapeProfileFence.assertClose(
                helper, emitted, expected, where + ": emitted launch forward motor");
    }

    /**
     * The launch's vertical motor is scaled by the up ray exactly as its forward is by the front
     * ray. Compared against the <em>emitted</em> motor, because that is the only comparison that
     * can fail if the program stops consulting the up ray at all.
     */
    static void assertLaunchThrustIsScaledByUpClearance(GameTestHelper helper, JumpRun run) {
        helper.assertTrue(
                run.upRayAtLaunchStart < EscapeJumpProgram.ESCAPE_THREAT_RANGE_BLOCKS,
                Component.literal(String.format(
                        Locale.ROOT,
                        "the lid was not inside the threat range at launch start (%.4f), so the"
                                + " probe cannot see the scaling",
                        run.upRayAtLaunchStart)));
        helper.assertTrue(
                run.launchVerticalMotor
                        < EscapeJumpProgram.ESCAPE_LAUNCH_VERTICAL - EscapeProfileFence.TOLERANCE,
                Component.literal(String.format(
                        Locale.ROOT,
                        "a low lid did not reduce the emitted launch vertical: %.9f against the"
                                + " unscaled %.9f",
                        run.launchVerticalMotor,
                        EscapeJumpProgram.ESCAPE_LAUNCH_VERTICAL)));
        EscapeProfileFence.assertClose(
                helper,
                run.launchVerticalMotor,
                EscapeJumpProgram.ESCAPE_LAUNCH_VERTICAL
                        * EscapeProfileFence.quantizedScale(run.upRayAtLaunchStart),
                "emitted launch vertical motor");
    }

    /** Parks the body, level and at rest, at a structure-relative position with yaw 0. */
    static void park(GameTestHelper helper, FlyEntity fly, Vec3 relative, float yaw) {
        Vec3 absolute = helper.absoluteVec(relative);
        fly.resetForFixture(absolute.x, absolute.y, absolute.z, yaw);
    }

    /** How much of the body's world heading points along {@code side}, in {@code [-1, 1]}. */
    static double towardness(float yawDegrees, Direction side) {
        Vec3 heading = Vec3.directionFromRotation(0.0F, yawDegrees);
        return heading.x * side.getStepX() + heading.z * side.getStepZ();
    }

}
