package dev.neurocraft.fly.gametest;

import dev.neurocraft.fly.NeuroCraftFly;
import dev.neurocraft.fly.body.BodyProgramContext;
import dev.neurocraft.fly.body.BodyProgramContextSampler;
import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.BodyProgramRuntime;
import dev.neurocraft.fly.body.BodyProgramState;
import dev.neurocraft.fly.body.BodyProgramStep;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ProgramParameters;
import dev.neurocraft.fly.body.SelectionSource;
import dev.neurocraft.fly.body.programs.CruiseProgram;
import dev.neurocraft.fly.body.programs.LandProgram;
import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import dev.neurocraft.fly.control.PrimitiveAction;
import dev.neurocraft.fly.control.RawPrimitiveAction;
import dev.neurocraft.fly.entity.FlyEntity;
import dev.neurocraft.fly.entity.FlySensors;
import dev.neurocraft.fly.entity.PrimitiveActuator;
import dev.neurocraft.fly.gametest.CruiseFlightFence.Flight;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.function.Consumer;
import net.fabricmc.fabric.api.gametest.v1.GameTest;
import net.minecraft.core.BlockPos;
import net.minecraft.gametest.framework.GameTestHelper;
import net.minecraft.network.chat.Component;
import net.minecraft.util.Mth;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.phys.Vec3;

/**
 * Per-program body-program pins (ADR 0020). Registered as a second {@code fabric-gametest}
 * entrypoint beside {@code FlyEntityGameTests}, which holds the global frame pins and is already
 * 2,286 lines; the external-bridge lane registers neither.
 *
 * <p>These pins drive {@link BodyProgramRuntime} directly, exactly as the frame pins do and for the
 * same reason: the default GameTest lane runs with no bridge configured, so a PAUSE
 * acknowledgement cannot be injected through the real WebSocket path.
 * {@code BodyPackageBoundaryTest.onlyTheStagedPauseTransactionCanEngageScriptedAuthority} names
 * this file in its allowlist for that. The arena measurement, the synthetic contexts, the declared
 * bounds and the flight metrics live in {@link CruiseFlightFence}.
 *
 * <p><strong>Nothing here is a claim about the connectome.</strong> A body program is scripted
 * presentation and engineering; these pins say what the script does to the body.
 */
public final class BodyProgramGameTests {
    private static final double EXACT = 1.0e-12;

    /** Real vertical block probes: feet clearance, footprint support, bounded range, and no target. */
    @GameTest(maxTicks = 20)
    public void groundContextUsesFeetAndRejectsPartialFootprintsAndStaleSamples(GameTestHelper helper) {
        helper.setBlock(new BlockPos(1, 1, 1), Blocks.STONE);
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 4.0, 1.5));
        fly.setXRot(80.0F);
        PhysicalSensorSnapshot snapshot = FlySensors.sample(helper.getLevel(), fly);
        BodyProgramContext sampled = BodyProgramContextSampler.sample(helper.getLevel(), fly, snapshot);
        helper.assertTrue(sampled.snapshot() == snapshot && sampled.target().isEmpty() && sampled.food().isEmpty(),
                Component.literal("support sampling changed sensors or introduced a target"));
        helper.assertTrue(sampled.surfaceBelow().orElseThrow().sturdy()
                        && Math.abs(sampled.surfaceBelow().orElseThrow().distanceBlocks() - 2.0) < EXACT,
                Component.literal("clearance must be two blocks from feet, independent of pitch"));

        FlyEntity edge = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.98, 4.0, 1.5));
        BodyProgramContext partial = groundContext(helper, edge);
        helper.assertTrue(partial.surfaceBelow().isPresent() && !partial.surfaceBelow().orElseThrow().sturdy(),
                Component.literal("a partly supported footprint was declared sturdy"));
        FlyEntity distant = helper.spawn(NeuroCraftFly.FLY, new Vec3(4.5, 7.0, 4.5));
        helper.assertTrue(groundContext(helper, distant).surfaceBelow().isEmpty(),
                Component.literal("a floor beyond the six-block search was supplied"));

        boolean wrongEntityRefused = false;
        try {
            BodyProgramContextSampler.sample(helper.getLevel(), edge, snapshot);
        } catch (IllegalArgumentException expected) {
            wrongEntityRefused = true;
        }
        helper.assertTrue(wrongEntityRefused, Component.literal("another entity's sample was accepted"));
        helper.runAfterDelay(1, () -> {
            boolean staleRefused = false;
            try {
                BodyProgramContextSampler.sample(helper.getLevel(), fly, snapshot);
            } catch (IllegalArgumentException expected) {
                staleRefused = true;
            }
            helper.assertTrue(staleRefused, Component.literal("a prior-tick sample was accepted"));
            helper.succeed();
        });
    }

    /** The complete-context overload retains admitted-neutral and preemption interlocks. */
    @GameTest(maxTicks = 20)
    public void groundProgramsRequireContextAndAdmittedScriptedAuthority(GameTestHelper helper) {
        helper.setBlock(new BlockPos(1, 1, 1), Blocks.STONE);
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 4.0, 1.5));
        BodyProgramContext context = groundContext(helper, fly);
        BodyProgramRuntime runtime = fly.bodyPrograms();
        helper.assertTrue(runtime.advance(context, true, true).isEmpty(),
                Component.literal("complete context obtained motor authority without PAUSE"));
        runtime.engageScripted(BodyProgramId.LAND, Modulation.NEUTRAL,
                SelectionSource.OPERATOR, context.snapshot().gameTick());
        helper.assertTrue(runtime.advance(context, false, true).isEmpty() && runtime.programOwnedTicks() == 0,
                Component.literal("a missing neutral command advanced LAND"));
        helper.assertTrue(runtime.advance(context, true, true).orElseThrow().next().id() == BodyProgramId.LAND,
                Component.literal("LAND did not receive complete support context"));
        helper.assertTrue(runtime.advance(context, true, false).isEmpty()
                        && runtime.selectionSource() == SelectionSource.DEFAULT,
                Component.literal("LAND displaced an admitted neural motor command"));
        runtime.engageScripted(BodyProgramId.LAND, Modulation.NEUTRAL,
                SelectionSource.OPERATOR, context.snapshot().gameTick());
        helper.assertTrue(runtime.advance(context.snapshot(), true, true).orElseThrow().next().id() == BodyProgramId.HOVER,
                Component.literal("legacy sensor-only context invented LAND support"));
        helper.succeed();
    }

    /** Actual actuator, Entity.move collision and drag; no setOnGround, velocity reset, or teleport. */
    @GameTest(maxTicks = 220)
    public void landDescendsToPhysicalFloorAndRetainsGroundedHold(GameTestHelper helper) {
        helper.setBlock(new BlockPos(2, 1, 2), Blocks.STONE);
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(2.5, 5.5, 2.5));
        BodyProgramRuntime runtime = fly.bodyPrograms();
        runtime.engageScripted(BodyProgramId.LAND, Modulation.NEUTRAL,
                SelectionSource.OPERATOR, helper.getLevel().getGameTime());
        double startY = fly.getY();
        int[] driven = {0};
        int[] groundedTicks = {0};
        helper.onEachTick(() -> {
            BodyProgramContext context = groundContext(helper, fly);
            BodyProgramStep step = runtime.advance(context, true, true).orElseThrow();
            helper.assertTrue(runtime.state().id() == BodyProgramId.LAND && step.exitTo().isEmpty(),
                    Component.literal("LAND timed out or selected a successor before stable touchdown"));
            PrimitiveActuator.apply(fly, RawPrimitiveAction.ZERO, step.motors());
            driven[0]++;
            if (context.snapshot().onGround()) {
                groundedTicks[0]++;
                helper.assertTrue(runtime.state().phaseIndex() == LandProgram.GROUNDED_HOLD
                                && Math.abs(fly.getY() - (startY - 3.5)) < 1.0e-8,
                        Component.literal("touchdown was not on the physical floor"));
                if (groundedTicks[0] == 25) {
                    helper.assertTrue(driven[0] < LandProgram.MAX_DESCENT_TICKS,
                            Component.literal("LAND failed its declared descent budget"));
                    helper.succeed();
                }
            } else {
                helper.assertTrue(groundedTicks[0] == 0,
                        Component.literal("LAND lost collision contact during grounded hold"));
            }
        });
    }

    /** WALK moves over a sturdy floor; removing it stops thrust on that same sampled tick. */
    @GameTest(maxTicks = 100)
    public void walkMaintainsGroundContactMovesAndExitsWhenSupportDisappears(GameTestHelper helper) {
        for (int x = 1; x <= 6; x++) {
            for (int z = 1; z <= 6; z++) {
                helper.setBlock(new BlockPos(x, 1, z), Blocks.STONE);
            }
        }
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(3.5, 2.0, 2.0));
        fly.setYRot(0);
        BodyProgramRuntime runtime = fly.bodyPrograms();
        double startZ = fly.getZ();
        double floorY = fly.getY();
        int[] walked = {0};
        boolean[] selected = {false};
        helper.onEachTick(() -> {
            if (!selected[0] && !fly.onGround()) {
                PrimitiveActuator.apply(fly, RawPrimitiveAction.ZERO,
                        PrimitiveAction.clamped(0, 0, -LandProgram.GROUND_CONTACT_THRUST, 0, 0));
                return;
            }
            BodyProgramContext context = groundContext(helper, fly);
            if (!selected[0]) {
                runtime.engageScripted(BodyProgramId.WALK, Modulation.NEUTRAL,
                        SelectionSource.OPERATOR, context.snapshot().gameTick());
                selected[0] = true;
            }
            helper.assertTrue(context.snapshot().onGround() && Math.abs(fly.getY() - floorY) < 1.0e-8,
                    Component.literal("WALK lost physical ground contact"));
            if (walked[0] == 60) {
                double distance = fly.getZ() - startZ;
                helper.assertTrue(distance > 1.5 && distance < 3.0,
                        Component.literal("WALK's 60-tick displacement is outside [1.5,3]: " + distance));
                for (int x = 1; x <= 6; x++) {
                    for (int z = 1; z <= 6; z++) {
                        helper.setBlock(new BlockPos(x, 1, z), Blocks.AIR);
                    }
                }
                BodyProgramStep stopped = runtime.advance(groundContext(helper, fly), true, true).orElseThrow();
                helper.assertTrue(stopped.exitTo().equals(Optional.of(BodyProgramId.HOVER))
                                && runtime.state().id() == BodyProgramId.HOVER
                                && stopped.motors().forwardThrust() < 0,
                        Component.literal("WALK failed to brake and exit immediately after floor removal"));
                helper.succeed();
                return;
            }
            BodyProgramStep step = runtime.advance(context, true, true).orElseThrow();
            helper.assertTrue(runtime.state().id() == BodyProgramId.WALK && step.exitTo().isEmpty(),
                    Component.literal("supported WALK was refused"));
            PrimitiveActuator.apply(fly, RawPrimitiveAction.ZERO, step.motors());
            walked[0]++;
        });
    }

    private static BodyProgramContext groundContext(GameTestHelper helper, FlyEntity fly) {
        return BodyProgramContextSampler.sample(helper.getLevel(), fly,
                FlySensors.sample(helper.getLevel(), fly));
    }

    // ---- CRUISE, in-world -------------------------------------------------------------------
    //
    // Six of the ten pins below spawn a body and drive it through PrimitiveActuator; four are
    // pure functions of CruiseProgram that happen to run inside the GameTest lane. Each of the
    // four says so in its own Javadoc, because "a server GameTest" and "a unit test that runs on
    // the server" are not the same evidence.

    /**
     * CRUISE is registered, enters unconditionally, and never names a successor. It is intended to
     * become the demo's default locomotion, so a refusal would need a fallback of its own.
     */
    @GameTest(maxTicks = 20)
    public void cruiseIsRegisteredEntersUnconditionallyAndNeverSelfExits(GameTestHelper helper) {
        helper.assertTrue(
                BodyProgramRuntime.registeredPrograms().contains(BodyProgramId.CRUISE),
                Component.literal("CRUISE is not in the registry"));

        helper.setBlock(new BlockPos(1, 1, 1), Blocks.STONE);
        FlyEntity grounded = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 2.0, 1.5));
        FlyEntity airborne = helper.spawn(NeuroCraftFly.FLY, new Vec3(4.5, 4.0, 4.5));
        CruiseProgram cruise = new CruiseProgram();
        for (FlyEntity fly : List.of(grounded, airborne)) {
            helper.assertTrue(
                    cruise.canEnter(bodyContext(helper, fly)),
                    Component.literal("CRUISE refused a context; it enters unconditionally"));
        }

        BodyProgramState state =
                BodyProgramState.entering(BodyProgramId.CRUISE, helper.getLevel().getGameTime());
        BodyProgramContext context = bodyContext(helper, airborne);
        for (int tick = 0; tick < 600; tick++) {
            BodyProgramStep step =
                    cruise.step(state, context, Modulation.NEUTRAL, ProgramParameters.NEUTRAL);
            helper.assertTrue(
                    step.exitTo().isEmpty(),
                    Component.literal("CRUISE named a successor at tick " + tick));
            helper.assertTrue(
                    step.next().id() == BodyProgramId.CRUISE,
                    Component.literal("CRUISE left itself at tick " + tick));
            state = step.next();
        }
        helper.assertTrue(
                state.ticksInProgram() == 600,
                Component.literal("CRUISE's tick counter did not advance monotonically"));
        helper.succeed();
    }

    /**
     * Altitude comes from the rays, not from an arena constant: a body spawned 0.49 blocks under
     * the ceiling leaves it and ends at least the declared minimum clearance from both surfaces,
     * and a body spawned 0.61 blocks over the floor climbs to the minimum and holds it.
     *
     * <p>Since the floor band (lane E2) the ceiling body does not stop 2.5 under the lid: its floor
     * ray reads the clamped 6.0 from there, so it keeps descending into the band and settles inside
     * it -- {@code CruiseFloorBandGameTests} measures where. This pin keeps asserting what it always
     * did, which both bodies still satisfy.
     *
     * <p>This is the direct pin on the live failure's first half -- the neural lane climbed into the
     * glass ceiling 62 s after spawn and stayed there. Both flies are driven in the same callback;
     * {@code FlySensors.obstructsSensorRay} excludes {@code FlyEntity}, so neither is in the other's
     * sensory world, and a plain {@code Entity} pushes nothing.
     */
    @GameTest(maxTicks = 400)
    public void cruiseHoldsClearanceFromCeilingAndFloor(GameTestHelper helper) {
        FlyEntity ceiling = helper.spawn(NeuroCraftFly.FLY, new Vec3(2.5, 7.4, 2.5));
        FlyEntity floor = helper.spawn(NeuroCraftFly.FLY, new Vec3(5.5, 0.5, 5.5));
        PhysicalSensorSnapshot ceilingEntry = FlySensors.sample(helper.getLevel(), ceiling);
        PhysicalSensorSnapshot floorEntry = FlySensors.sample(helper.getLevel(), floor);
        helper.assertTrue(
                ceilingEntry.rayUpBlocks() < CruiseProgram.CRUISE_MIN_VERTICAL_CLEARANCE_BLOCKS
                        && floorEntry.rayDownBlocks()
                                < CruiseProgram.CRUISE_MIN_FLOOR_CLEARANCE_BLOCKS,
                Component.literal(String.format(
                        Locale.ROOT,
                        "the scenario must start in violation, but up=%.3f down=%.3f",
                        ceilingEntry.rayUpBlocks(),
                        floorEntry.rayDownBlocks())));

        driveCruise(
                helper,
                List.of(ceiling, floor),
                Modulation.NEUTRAL,
                CruiseFlightFence.SETTLE_TICKS,
                flights -> {
                    assertSettledClearance(helper, flights.get(0), "ceiling");
                    assertSettledClearance(helper, flights.get(1), "floor");
                });
    }


    /**
     * A body released from the neural lane while descending is braked back to level rather than
     * left to coast. This is the case the free-band rate-error term was argued from, so it is
     * pinned in-world instead of argued from a scratch simulation.
     *
     * <p><strong>Historical limits of the old 2.5–4.0 band; this fixed scenario is rerun.</strong> Driven 300
     * ticks and entering at 0.4 blocks/tick downward, the body stays off the floor from 4.0 blocks
     * of clearance (lowest reading 0.657) and from 3.5 (0.158), but reaches the floor from 3.25,
     * 3.0 and 2.5; entering at the actuator's 0.6 blocks/tick speed cap it reaches the floor from
     * every clearance tried, including 6.0. Those readings are in-world, from a probe run of this
     * class. So this pin says the term works in the case it was justified by, and
     * {@code CruiseProgram.verticalThrust} states the cases where it does not.
     */
    @GameTest(maxTicks = 400)
    public void cruiseBrakesADescentEnteredFromTheNeuralLane(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(
                NeuroCraftFly.FLY,
                new Vec3(4.5, CruiseFlightFence.RELEASE_ENTRY_CLEARANCE_BLOCKS - 0.11, 4.5));
        fly.setDeltaMovement(0.0, CruiseFlightFence.RELEASE_DESCENT_BLOCKS_PER_TICK, 0.0);
        PhysicalSensorSnapshot entry = FlySensors.sample(helper.getLevel(), fly);
        helper.assertTrue(
                Math.abs(entry.rayDownBlocks() - CruiseFlightFence.RELEASE_ENTRY_CLEARANCE_BLOCKS)
                                < 0.05
                        && entry.rayDownBlocks()
                                > CruiseProgram.CRUISE_MAX_FLOOR_CLEARANCE_BLOCKS,
                Component.literal(String.format(
                        Locale.ROOT,
                        "the historical release must start above the new floor band at its unchanged clearance,"
                                + " but the down ray reads %.3f",
                        entry.rayDownBlocks())));

        driveCruise(
                helper,
                List.of(fly),
                Modulation.NEUTRAL,
                CruiseFlightFence.SETTLE_TICKS,
                flights -> {
                    List<Flight> flight = flights.getFirst();
                    double lowest = CruiseFlightFence.lowestDownClearance(flight);
                    helper.assertTrue(
                            lowest >= CruiseFlightFence.RELEASE_FLOOR_CLEARANCE_BLOCKS,
                            Component.literal(String.format(
                                    Locale.ROOT,
                                    "a body released at %.1f blocks/tick from %.1f blocks of"
                                            + " clearance fell to %.3f, past the declared %.2f",
                                    CruiseFlightFence.RELEASE_DESCENT_BLOCKS_PER_TICK,
                                    CruiseFlightFence.RELEASE_ENTRY_CLEARANCE_BLOCKS,
                                    lowest,
                                    CruiseFlightFence.RELEASE_FLOOR_CLEARANCE_BLOCKS)));
                    List<Flight> contacts = CruiseFlightFence.realVerticalContacts(flight);
                    helper.assertTrue(
                            contacts.isEmpty(),
                            Component.literal("the released body touched the floor on "
                                    + contacts.size() + " ticks"));
                    assertSettledClearance(helper, flight, "released");
                });
    }

    /**
     * Facing a wall, the body turns away instead of pushing into it.
     *
     * <p>The live failure pushed forward 0.25 with a yaw torque of -0.003 for 83 seconds. The spawn
     * puts a wall exactly four blocks off the nose -- {@link
     * CruiseProgram#CRUISE_SLOW_DISTANCE_BLOCKS}, so the body starts at the instant avoidance
     * begins and has to both slow and turn inside its own turn radius.
     *
     * <p><strong>The body does clip the wall, and the pin says so rather than pretending
     * otherwise.</strong> An in-world probe over entry ticks 0 through 23 counted 0 contact ticks
     * on 6 seeds, 1 on 17 and 2 on one, with heading changes of 248 to 274 degrees. The run below
     * is pinned to the worst of those seeds, and produces 2 contact ticks and a heading change of
     * 249.6 degrees. What separates this from the live failure is not the absence of contact but
     * its scale: the avoidance-sign negative control produces 26 to 28 contact ticks and turns 34
     * degrees instead of 250.
     */
    @GameTest(maxTicks = 180)
    public void cruiseTurnsAwayFromAWallInsteadOfPushingIntoIt(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(4.5, 4.0, 4.0));
        fly.setYRot(0.0F);
        PhysicalSensorSnapshot entry = FlySensors.sample(helper.getLevel(), fly);
        helper.assertTrue(
                Math.abs(entry.rayForwardBlocks() - CruiseProgram.CRUISE_SLOW_DISTANCE_BLOCKS)
                        < 0.05,
                Component.literal(
                        "the wall must sit at the declared slow distance, but the forward ray"
                                + " reads " + entry.rayForwardBlocks()));

        driveCruise(
                helper,
                List.of(fly),
                Modulation.NEUTRAL,
                100,
                flights -> assertTurnedAwayFromTheWall(helper, flights.getFirst(), "neutral"));
    }

    /**
     * The body leaves a corner it is facing into, and does not wedge in it.
     *
     * <p>This is the live failure reproduced as a scenario: spawned 0.85 blocks from the corner post
     * at yaw 135, which points the nose straight into it and makes the left and right rays read the
     * same 0.849 blocks -- the symmetric case where "turn toward the more open side" has no answer
     * and {@link CruiseProgram#CRUISE_SIDE_TIE_BLOCKS} has to supply one.
     *
     * <p>At the pinned entry tick the body clears 3.0 blocks on tick 43, reaches 10.82 blocks at
     * its furthest, and its longest run inside 1.5 blocks of the corner is 34 ticks. With the
     * avoidance sign flipped it never reaches 3.0 blocks and stays inside 1.5 for all 300.
     */
    @GameTest(maxTicks = 400)
    public void cruiseLeavesACorner(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(0.6, 4.0, 0.6));
        fly.setYRot(135.0F);
        PhysicalSensorSnapshot entry = FlySensors.sample(helper.getLevel(), fly);
        helper.assertTrue(
                Math.abs(entry.rayLeftBlocks() - entry.rayRightBlocks())
                        <= CruiseProgram.CRUISE_SIDE_TIE_BLOCKS,
                Component.literal(String.format(
                        Locale.ROOT,
                        "the corner must be inside the tie deadband, but left=%.3f right=%.3f",
                        entry.rayLeftBlocks(),
                        entry.rayRightBlocks())));

        driveCruise(
                helper,
                List.of(fly),
                Modulation.NEUTRAL,
                CruiseFlightFence.SETTLE_TICKS,
                flights -> assertLeftTheCorner(helper, flights.getFirst()));
    }

    /**
     * <strong>Pure-function pin</strong> (it runs in the GameTest lane but never touches the
     * world). In open air the wander is the only yaw source, it covers a real heading range, and it
     * is a pure deterministic function of {@code (enteredGameTick, ticksInProgram)}: two bodies that
     * entered CRUISE on the same game tick produce bit-identical motor sequences, and two that
     * entered on different ticks do not.
     *
     * <p><strong>This is a pure-function pin, and says so.</strong> The heading is integrated at the
     * actuator's own declared yaw rate rather than measured in the world, because this arena has no
     * open air -- see {@link CruiseFlightFence}. The context is the open-air sample the six rays
     * actually produce outdoors.
     */
    @GameTest(maxTicks = 20)
    public void cruiseWandersInOpenSpaceDeterministically(GameTestHelper helper) {
        CruiseProgram cruise = new CruiseProgram();
        long entered = 4242L;
        double heading = 0.0;
        double lowest = 0.0;
        double highest = 0.0;
        BodyProgramState state = BodyProgramState.entering(BodyProgramId.CRUISE, entered);
        for (int tick = 0; tick < CruiseFlightFence.WANDER_TICKS; tick++) {
            BodyProgramStep step = cruise.step(
                    state,
                    BodyProgramContext.sensorsOnly(CruiseFlightFence.openAir(tick, heading, 0.0)),
                    Modulation.NEUTRAL,
                    ProgramParameters.NEUTRAL);
            heading += step.motors().yawTorque()
                    * PrimitiveActuator.MAX_YAW_DELTA_DEGREES_PER_TICK;
            lowest = Math.min(lowest, heading);
            highest = Math.max(highest, heading);
            state = step.next();
        }
        helper.assertTrue(
                highest - lowest >= CruiseFlightFence.WANDER_RANGE_DEGREES,
                Component.literal(String.format(
                        Locale.ROOT,
                        "the open-air wander covered only %.1f degrees in %d ticks",
                        highest - lowest,
                        CruiseFlightFence.WANDER_TICKS)));

        int ticks = CruiseFlightFence.WANDER_TICKS;
        long[] first = cruiseMotorWords(entered, Modulation.NEUTRAL, ProgramParameters.NEUTRAL, ticks);
        helper.assertTrue(
                Arrays.equals(first, cruiseMotorWords(entered, Modulation.NEUTRAL, ProgramParameters.NEUTRAL, ticks)),
                Component.literal("CRUISE is not deterministic across identical entries"));
        helper.assertFalse(
                Arrays.equals(first, cruiseMotorWords(entered + 1L, Modulation.NEUTRAL, ProgramParameters.NEUTRAL, ticks)),
                Component.literal("CRUISE's wander is not seeded from the entry tick"));
        helper.succeed();
    }

    /**
     * <strong>Pure-function pin</strong> (it runs in the GameTest lane but never touches the world):
     * {@code speed} scales forward thrust and nothing else.
     */
    @GameTest(maxTicks = 20)
    public void cruiseSpeedScalesForwardThrust(GameTestHelper helper) {
        CruiseProgram cruise = new CruiseProgram();
        BodyProgramState state = BodyProgramState.entering(BodyProgramId.CRUISE, 11L);
        BodyProgramContext context =
                BodyProgramContext.sensorsOnly(CruiseFlightFence.openAir(0L, 0.0, 0.0));
        PrimitiveAction full = cruise.step(
                        state, context, Modulation.NEUTRAL, ProgramParameters.NEUTRAL)
                .motors();
        helper.assertTrue(
                Math.abs(full.forwardThrust() - CruiseProgram.CRUISE_THRUST) < EXACT,
                Component.literal("open-air forward thrust must be the declared maximum, got "
                        + full.forwardThrust()));

        for (double speed : new double[] {0.75, 0.5, 0.25, 0.0}) {
            PrimitiveAction scaled = cruise.step(
                            state,
                            context,
                            new Modulation(speed, 1.0, 1.0, 0.0),
                            ProgramParameters.NEUTRAL)
                    .motors();
            helper.assertTrue(
                    Math.abs(scaled.forwardThrust() - CruiseProgram.CRUISE_THRUST * speed) < EXACT,
                    Component.literal(String.format(
                            Locale.ROOT,
                            "speed %.2f gave forward %.9f, not %.9f",
                            speed,
                            scaled.forwardThrust(),
                            CruiseProgram.CRUISE_THRUST * speed)));
            helper.assertTrue(
                    scaled.yawTorque() == full.yawTorque()
                            && scaled.verticalThrust() == full.verticalThrust(),
                    Component.literal("speed moved a channel it does not scale"));
        }
        helper.succeed();
    }

    /**
     * {@code turnRate} scales the wander and nothing else. At {@code turnRate = 0} the open-air yaw
     * torque is exactly zero, and a body facing a wall still turns away from it inside the same
     * bounds as at full turn rate.
     *
     * <p>That second half is the point. Avoidance yaw is never attenuated by modulation, because a
     * fly a selector can steer into a wall is a defect, and in stage 4 the selector is a readout.
     */
    @GameTest(maxTicks = 180)
    public void cruiseTurnRateScalesWanderOnly(GameTestHelper helper) {
        CruiseProgram cruise = new CruiseProgram();
        BodyProgramContext openAir =
                BodyProgramContext.sensorsOnly(CruiseFlightFence.openAir(0L, 0.0, 0.0));
        Modulation noTurn = new Modulation(1.0, 0.0, 1.0, 0.0);
        BodyProgramState state = BodyProgramState.entering(BodyProgramId.CRUISE, 97L);
        for (int tick = 0; tick < 200; tick++) {
            BodyProgramStep still =
                    cruise.step(state, openAir, noTurn, ProgramParameters.NEUTRAL);
            helper.assertTrue(
                    still.motors().yawTorque() == 0.0,
                    Component.literal("turnRate 0 left an open-air yaw torque of "
                            + still.motors().yawTorque() + " at tick " + tick));
            BodyProgramStep half = cruise.step(
                    state, openAir, new Modulation(1.0, 0.5, 1.0, 0.0), ProgramParameters.NEUTRAL);
            BodyProgramStep whole =
                    cruise.step(state, openAir, Modulation.NEUTRAL, ProgramParameters.NEUTRAL);
            helper.assertTrue(
                    Math.abs(half.motors().yawTorque() * 2.0 - whole.motors().yawTorque()) < EXACT,
                    Component.literal("turnRate does not scale the wander linearly"));
            state = whole.next();
        }

        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(4.5, 4.0, 4.0));
        fly.setYRot(0.0F);
        driveCruise(
                helper,
                List.of(fly),
                noTurn,
                100,
                flights -> assertTurnedAwayFromTheWall(helper, flights.getFirst(), "turnRate 0"));
    }

    /**
     * <strong>Pure-function pin</strong> (it runs in the GameTest lane but never touches the
     * world). {@code amplitude} and {@code urgency} are structurally ignored: varying either, including past
     * {@code [0, 1]} where {@code Modulation}'s own clamp is the guarantee, produces a bit-identical
     * motor sequence. The two scalars CRUISE does declare are checked to move it, so the pin
     * discriminates in both directions.
     */
    @GameTest(maxTicks = 20)
    public void cruiseAmplitudeAndUrgencyAreStructurallyIgnored(GameTestHelper helper) {
        long[] declared = cruiseMotorWords(2026L, Modulation.NEUTRAL, ProgramParameters.NEUTRAL, 200);
        for (Modulation variant : List.of(
                new Modulation(1.0, 1.0, 0.0, 0.0),
                new Modulation(1.0, 1.0, 0.5, 1.0),
                new Modulation(1.0, 1.0, 0.0, 1.0),
                new Modulation(1.0, 1.0, 9.0, 9.0),
                new Modulation(1.0, 1.0, -4.0, -4.0))) {
            helper.assertTrue(
                    Arrays.equals(declared, cruiseMotorWords(2026L, variant, ProgramParameters.NEUTRAL, 200)),
                    Component.literal(
                            "CRUISE read a modulation scalar it does not declare: " + variant));
        }
        helper.assertFalse(
                Arrays.equals(
                        declared, cruiseMotorWords(
                                2026L,
                                new Modulation(0.5, 1.0, 1.0, 0.0),
                                ProgramParameters.NEUTRAL,
                                200)),
                Component.literal("CRUISE ignored the speed scalar it does declare"));
        helper.assertFalse(
                Arrays.equals(
                        declared, cruiseMotorWords(
                                2026L,
                                new Modulation(1.0, 0.5, 1.0, 0.0),
                                ProgramParameters.NEUTRAL,
                                200)),
                Component.literal("CRUISE ignored the turnRate scalar it does declare"));
        helper.succeed();
    }

    /**
     * Every motor CRUISE emits is inside the rails, inside its own declared per-channel envelope,
     * and exactly zero on the two channels it does not use -- over {@link
     * CruiseFlightFence#envelopeSweep} at five modulations including two the caller asked for
     * outside {@code [0, 1]}, crossed with five turn biases including the two saturating ends, and
     * then through the real actuator so the speed cap and the yaw rate are checked on the body
     * rather than on the command.
     *
     * <p><strong>The bias sweep is what keeps the yaw bound honest.</strong>
     * {@code assertDeclaredEnvelope} holds every tick to
     * {@code |yaw| <= CruiseProgram.CRUISE_AVOID_TORQUE}, and the bias is the one term that could
     * push past it: the sum {@code 0.9*avoid + (0.25 + CRUISE_BIAS_TORQUE)*(1 - avoid)} exceeds 0.9
     * as soon as {@code CRUISE_BIAS_TORQUE} exceeds 0.65. Sweeping the ends of the bias range
     * against the ends of the avoidance range is therefore a direct pin on that constant, not a
     * restatement of it.
     */
    @GameTest(maxTicks = 20)
    public void cruiseStaysInsideTheActuationEnvelope(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(4.5, 4.0, 4.5));
        CruiseProgram cruise = new CruiseProgram();
        List<Modulation> modulations = List.of(
                Modulation.NEUTRAL,
                new Modulation(0.0, 0.0, 0.0, 0.0),
                new Modulation(0.5, 0.5, 0.5, 0.5),
                new Modulation(7.0, 7.0, 7.0, 7.0),
                new Modulation(-2.0, -2.0, -2.0, -2.0));
        List<ProgramParameters> biases = List.of(
                ProgramParameters.NEUTRAL,
                new ProgramParameters(1.0),
                new ProgramParameters(-1.0),
                new ProgramParameters(0.5),
                // Past the declared range on purpose: ProgramParameters' own clamp is the
                // guarantee, and this is where it is exercised end to end.
                new ProgramParameters(-9.0));

        for (Modulation modulation : modulations) {
            for (ProgramParameters parameters : biases) {
                for (PhysicalSensorSnapshot sample : CruiseFlightFence.envelopeSweep()) {
                    BodyProgramContext context = BodyProgramContext.sensorsOnly(sample);
                    BodyProgramState state =
                            BodyProgramState.entering(BodyProgramId.CRUISE, sample.gameTick());
                    for (int tick = 0; tick < 40; tick++) {
                        BodyProgramStep step =
                                cruise.step(state, context, modulation, parameters);
                        String where = " at tick " + tick + " under " + modulation + " and "
                                + parameters + " with forward ray " + sample.rayForwardBlocks();
                        assertDeclaredEnvelope(helper, step.motors(), where);

                        float yawBefore = fly.getYRot();
                        PrimitiveActuator.apply(fly, RawPrimitiveAction.ZERO, step.motors());
                        helper.assertTrue(
                                fly.getDeltaMovement().length()
                                        <= PrimitiveActuator.MAX_SPEED_BLOCKS_PER_TICK + 1.0e-9,
                                Component.literal(
                                        "CRUISE drove the body past the speed cap" + where));
                        helper.assertTrue(
                                Math.abs(Mth.degreesDifference(yawBefore, fly.getYRot()))
                                        <= PrimitiveActuator.MAX_YAW_DELTA_DEGREES_PER_TICK
                                                + 1.0e-4,
                                Component.literal(
                                        "CRUISE turned faster than the yaw rate" + where));
                        state = step.next();
                    }
                    fly.setDeltaMovement(Vec3.ZERO);
                }
            }
        }
        helper.succeed();
    }

    /**
     * Level flight is a property of the floor band, and the band is where the body flies.
     *
     * <p><strong>Pure-function half</strong> (it runs in the GameTest lane but never touches the
     * world): inside the band -- every horizontal ray and the up ray at the clamp, the floor ray at
     * {@link CruiseFlightFence#IN_BAND_CLEARANCE_BLOCKS} -- the vertical channel is exactly zero at
     * zero vertical rate, and a body sinking there is brought back to level instead of coasting for
     * the 49 ticks the 0.98 vertical drag would otherwise give it. The same sample with the floor
     * ray at its 6.0 clamp -- open air -- commands a descent, which is the whole of what lane E2
     * changed: until it, this pin asserted zero in open air, and that zero was the R-31 defect.
     *
     * <p><strong>In-world half:</strong> a body entered into CRUISE at 1.25 blocks of floor
     * clearance and driven for {@link CruiseFlightFence#LEVEL_TICKS} ticks through the real
     * actuator and the entity's own physics keeps its floor ray inside the band, both edges
     * inclusive, on every tick. The drift it does produce is in the report line and the failure
     * message, so the bound can be read against the measurement rather than trusted. Measured:
     * with the old 2.5–4.0 band, none: the floor ray read 3.2000 on all 400 ticks. The new entry is 1.25.
     */
    @GameTest(maxTicks = 500)
    public void cruiseFliesLevelInsideTheBand(GameTestHelper helper) {
        CruiseProgram cruise = new CruiseProgram();
        BodyProgramState state = BodyProgramState.entering(BodyProgramId.CRUISE, 5L);
        for (int tick = 0; tick < 200; tick++) {
            BodyProgramStep level = cruise.step(
                    state,
                    BodyProgramContext.sensorsOnly(CruiseFlightFence.inBand(tick, 0.0, 0.0)),
                    Modulation.NEUTRAL,
                    ProgramParameters.NEUTRAL);
            helper.assertTrue(
                    level.motors().verticalThrust() == 0.0,
                    Component.literal("in-band level flight commanded a vertical thrust of "
                            + level.motors().verticalThrust() + " at tick " + tick));
            BodyProgramStep sinking = cruise.step(
                    state,
                    BodyProgramContext.sensorsOnly(CruiseFlightFence.inBand(tick, 0.0, -0.2)),
                    Modulation.NEUTRAL,
                    ProgramParameters.NEUTRAL);
            helper.assertTrue(
                    sinking.motors().verticalThrust() > 0.0,
                    Component.literal(
                            "a body sinking inside the band was not brought back to level"));
            BodyProgramStep openAir = cruise.step(
                    state,
                    BodyProgramContext.sensorsOnly(CruiseFlightFence.openAir(tick, 0.0, 0.0)),
                    Modulation.NEUTRAL,
                    ProgramParameters.NEUTRAL);
            helper.assertTrue(
                    openAir.motors().verticalThrust() < 0.0,
                    Component.literal("open air -- the floor ray at its clamp -- must command a"
                            + " descent, and commanded " + openAir.motors().verticalThrust()
                            + " at tick " + tick));
            state = level.next();
        }

        FlyEntity fly = helper.spawn(
                NeuroCraftFly.FLY,
                new Vec3(4.5, CruiseFlightFence.IN_BAND_CLEARANCE_BLOCKS - 0.11, 4.5));
        PhysicalSensorSnapshot entry = FlySensors.sample(helper.getLevel(), fly);
        helper.assertTrue(
                Math.abs(entry.rayDownBlocks() - CruiseFlightFence.IN_BAND_CLEARANCE_BLOCKS)
                        < 0.01,
                Component.literal(String.format(
                        Locale.ROOT,
                        "the body must enter at the declared in-band clearance, but the down ray"
                                + " reads %.4f",
                        entry.rayDownBlocks())));
        driveCruise(
                helper,
                List.of(fly),
                Modulation.NEUTRAL,
                CruiseFlightFence.LEVEL_TICKS,
                flights -> {
                    List<Flight> flight = flights.getFirst();
                    double lowest = CruiseFlightFence.lowestDownClearance(flight);
                    double highest = CruiseFlightFence.highestDownClearance(flight);
                    helper.assertTrue(
                            lowest >= CruiseProgram.CRUISE_MIN_FLOOR_CLEARANCE_BLOCKS
                                    && highest <= CruiseProgram.CRUISE_MAX_FLOOR_CLEARANCE_BLOCKS,
                            Component.literal(String.format(
                                    Locale.ROOT,
                                    "a body entered at %.2f blocks of floor clearance left the"
                                            + " %.1f-%.1f band inside %d ticks: the down ray ran"
                                            + " from %.4f to %.4f",
                                    CruiseFlightFence.IN_BAND_CLEARANCE_BLOCKS,
                                    CruiseProgram.CRUISE_MIN_FLOOR_CLEARANCE_BLOCKS,
                                    CruiseProgram.CRUISE_MAX_FLOOR_CLEARANCE_BLOCKS,
                                    flight.size(),
                                    lowest,
                                    highest)));
                    FloorBandFence.report(
                            "cruiseFliesLevelInsideTheBand",
                            String.format(
                                    Locale.ROOT,
                                    "entered at down %.4f; over %d ticks the down ray ran %.4f to"
                                            + " %.4f (drift %.4f)",
                                    entry.rayDownBlocks(),
                                    flight.size(),
                                    lowest,
                                    highest,
                                    Math.max(
                                            Math.abs(lowest - entry.rayDownBlocks()),
                                            Math.abs(highest - entry.rayDownBlocks()))));
                });
    }

    // ---- CRUISE, the signed turn bias ---------------------------------------------------------

    /**
     * <strong>Pure-function pin</strong> (it runs in the GameTest lane but never touches the
     * world). The signed turn bias moves yaw and only yaw, orders the yaw monotonically, is
     * antisymmetric about the unbiased command, survives {@code turnRate = 0}, and yields
     * <em>completely</em> at full avoidance authority.
     *
     * <p>None of those is a restatement of the yaw law. Monotonicity and antisymmetry hold for any
     * signed term added linearly with a positive coefficient; the four-channel bit-identity says
     * the bias reached no other motor; {@code turnRate = 0} pins the declared independence of the
     * two selector inputs; and the contact-sample identity is the whole "avoidance keeps priority"
     * claim at the authority where it is absolute.
     */
    @GameTest(maxTicks = 20)
    public void cruiseTurnBiasIsSignedMovesOnlyYawAndYieldsAtFullAvoidance(GameTestHelper helper) {
        CruiseProgram cruise = new CruiseProgram();
        List<Double> biases = List.of(-1.0, -0.5, 0.0, 0.5, 1.0);
        BodyProgramContext openAir =
                BodyProgramContext.sensorsOnly(CruiseFlightFence.openAir(0L, 0.0, 0.0));
        BodyProgramState state = BodyProgramState.entering(BodyProgramId.CRUISE, 4242L);

        for (int tick = 0; tick < 200; tick++) {
            PrimitiveAction unbiased = cruise.step(
                            state, openAir, Modulation.NEUTRAL, ProgramParameters.NEUTRAL)
                    .motors();
            double previousYaw = Double.NEGATIVE_INFINITY;
            for (double bias : biases) {
                PrimitiveAction biased = cruise.step(
                                state, openAir, Modulation.NEUTRAL, new ProgramParameters(bias))
                        .motors();
                helper.assertTrue(
                        biased.yawTorque() > previousYaw,
                        Component.literal(String.format(
                                Locale.ROOT,
                                "the turn bias is not monotone: %+.2f gave yaw %.9f, which does"
                                        + " not exceed %.9f, at tick %d",
                                bias,
                                biased.yawTorque(),
                                previousYaw,
                                tick)));
                previousYaw = biased.yawTorque();
                helper.assertTrue(
                        biased.forwardThrust() == unbiased.forwardThrust()
                                && biased.lateralThrust() == unbiased.lateralThrust()
                                && biased.verticalThrust() == unbiased.verticalThrust()
                                && biased.pitchTorque() == unbiased.pitchTorque(),
                        Component.literal(
                                "the turn bias moved a channel other than yaw at tick " + tick));
                PrimitiveAction mirrored = cruise.step(
                                state, openAir, Modulation.NEUTRAL, new ProgramParameters(-bias))
                        .motors();
                helper.assertTrue(
                        Math.abs(biased.yawTorque() + mirrored.yawTorque()
                                        - 2.0 * unbiased.yawTorque())
                                < EXACT,
                        Component.literal("the turn bias is not antisymmetric at tick " + tick));
            }
            state = cruise.step(state, openAir, Modulation.NEUTRAL, ProgramParameters.NEUTRAL)
                    .next();
        }

        // turnRate attenuates the wander and only the wander: at turnRate 0 the open-air yaw is
        // exactly the bias, so a body under a full bias still turns.
        Modulation noTurn = new Modulation(1.0, 0.0, 1.0, 0.0);
        BodyProgramState still = BodyProgramState.entering(BodyProgramId.CRUISE, 97L);
        PrimitiveAction unattenuated =
                cruise.step(still, openAir, noTurn, new ProgramParameters(1.0)).motors();
        helper.assertTrue(
                unattenuated.yawTorque() > 0.0,
                Component.literal("turnRate 0 attenuated the signed turn bias to "
                        + unattenuated.yawTorque()));
        helper.assertTrue(
                cruise.step(still, openAir, noTurn, ProgramParameters.NEUTRAL)
                                .motors()
                                .yawTorque()
                        == 0.0,
                Component.literal("turnRate 0 with no bias must command no open-air yaw"));

        // Full avoidance authority: a contact tick. The bias contributes exactly nothing, so the
        // whole motor word is bit-identical across the range including both saturating ends.
        BodyProgramContext contact = BodyProgramContext.sensorsOnly(
                CruiseFlightFence.synthetic(8L, 214.0, 0.0, true, 0.19, 6.0, 6.0, 6.0, 6.0));
        long[] declared = cruiseMotorWords(
                contact, 11L, Modulation.NEUTRAL, ProgramParameters.NEUTRAL, 40);
        for (double bias : biases) {
            helper.assertTrue(
                    Arrays.equals(
                            declared,
                            cruiseMotorWords(
                                    contact,
                                    11L,
                                    Modulation.NEUTRAL,
                                    new ProgramParameters(bias),
                                    40)),
                    Component.literal("a turn bias of " + bias
                            + " changed the motors on a contact tick, where avoidance holds full"
                            + " authority and the bias must contribute exactly nothing"));
        }
        helper.succeed();
    }

    /**
     * <strong>The sign chain, measured in the world.</strong> Three bodies, identical in every way
     * except the signed turn bias they are given, are driven for
     * {@link TurnBiasFence#CURVE_TICKS} ticks through the real actuator and the entity's own
     * passive physics. The body under a positive bias must end up displaced toward <em>its own
     * initial right</em>, the one under a negative bias toward its left, and the unbiased one
     * between them -- with "the body's right" taken from the engine's own
     * {@code Direction.getClockWise()} table by way of {@link RaySideFence}, never from the
     * production frame under test.
     *
     * <p><strong>The sensory context is the declared in-band free-flight sample, and that is the
     * honest substitute rather than a convenience.</strong> This arena is 8 blocks across and the
     * rays reach 6.0, so a wall is always inside sensing range and an in-world "bias in open space"
     * run would be measuring avoidance instead -- {@link CruiseFlightFence} says so, and
     * {@code cruiseWandersInOpenSpaceDeterministically} relies on the same open horizon. The floor
     * ray sits inside the floor band rather than at its clamp because, since lane E2, a clamped
     * floor ray commands a descent and this pin is about the lateral channels; inside the band the
     * vertical channel is exactly what it was before the band, so the paths measured here are the
     * ones this pin has always measured. Everything downstream of the program is real: the
     * world-space thrust direction, the yaw integration, the speed cap, the move, and the drag. The
     * bodies never touch anything, and the pin asserts that rather than assuming it.
     *
     * <p>The three bodies are spawned on the same block. {@code FlySensors.obstructsSensorRay}
     * excludes {@code FlyEntity} and a plain {@code Entity} neither pushes nor collides with
     * another, so co-locating them is what makes "identical except the bias" literally true.
     */
    @GameTest(maxTicks = 120)
    public void cruiseTurnBiasCurvesTheDeclaredWay(GameTestHelper helper) {
        List<ProgramParameters> biases = List.of(
                new ProgramParameters(1.0), ProgramParameters.NEUTRAL, new ProgramParameters(-1.0));
        List<FlyEntity> flies = new ArrayList<>();
        for (int index = 0; index < biases.size(); index++) {
            FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(4.0, 4.0, 4.0));
            fly.setYRot(0.0F);
            flies.add(fly);
        }

        driveCruiseProgram(
                helper,
                flies,
                biases,
                TurnBiasFence.CURVE_MODULATION,
                true,
                TurnBiasFence.CURVE_TICKS,
                flights -> {
                    double[] lateral = new double[3];
                    double[] turned = new double[3];
                    for (int index = 0; index < 3; index++) {
                        List<Flight> flight = flights.get(index);
                        lateral[index] =
                                TurnBiasFence.lateralOffsetTowardBodyRight(flight, 0.0F);
                        turned[index] = TurnBiasFence.turnedTowardBodyRight(flight, 0.0F);
                        helper.assertTrue(
                                CruiseFlightFence.wallContactTicks(flight) == 0,
                                Component.literal(
                                        "a curvature run touched a wall, so its path is not free"
                                                + " flight: " + describe(lateral, turned)));
                        helper.assertTrue(
                                TurnBiasFence.furthestFromStart(flight) < 4.0,
                                Component.literal("a curvature run left the arena's clear space"));
                    }
                    helper.assertTrue(
                            lateral[0] >= TurnBiasFence.CURVE_LATERAL_BLOCKS
                                    && turned[0] >= TurnBiasFence.CURVE_PATH_TURN_DEGREES,
                            Component.literal(
                                    "a positive turn bias must curve the path toward the body's"
                                            + " right, and did not: " + describe(lateral, turned)));
                    helper.assertTrue(
                            lateral[2] <= -TurnBiasFence.CURVE_LATERAL_BLOCKS
                                    && turned[2] <= -TurnBiasFence.CURVE_PATH_TURN_DEGREES,
                            Component.literal(
                                    "a negative turn bias must curve the path toward the body's"
                                            + " left, and did not: " + describe(lateral, turned)));
                    helper.assertTrue(
                            lateral[0] > lateral[1]
                                    && lateral[1] > lateral[2]
                                    && turned[0] > turned[1]
                                    && turned[1] > turned[2],
                            Component.literal(
                                    "the flown paths are not ordered by the bias they were given: "
                                            + describe(lateral, turned)));
                });
    }

    /**
     * <strong>A full bias pointed straight into a wall does not put the body into it.</strong> The
     * body is spawned with a wall exactly {@link CruiseProgram#CRUISE_SIDE_CLEARANCE_BLOCKS} off
     * its right side -- the distance at which that side ray begins defending -- and given
     * {@code turnBias = +1}, which is a request to turn into it. It must close to a standoff and
     * turn away, inside the same contact budget the unbiased wall pin uses.
     *
     * <p><strong>A matched unbiased body flies the same scenario, and the contact assertion is
     * relative to it.</strong> The arena is a closed box, so a glancing contact with some other
     * wall is CRUISE's own documented regime and says nothing about the bias; comparing against a
     * body that differs <em>only</em> in its bias is what makes the number mean "what the bias
     * cost". The absolute budget is kept beside it, so a run where both bodies scrape still fails.
     * See {@link TurnBiasFence#WALL_CONTACT_BIAS_BUDGET}.
     *
     * <p><strong>What this does and does not establish, because the composition is not "the bias
     * is ignored".</strong> Near the wall the avoidance term is the larger one and wins; further
     * out the bias is larger and reduces the turn, and a sustained bias into a wall therefore
     * settles into wall-following at about the crossover clearance rather than into contact.
     * {@code CruiseProgram.CRUISE_BIAS_TORQUE} writes out where the two cross over and what that
     * looks like. So the claim is the one that matters for a demo -- a selector cannot steer this
     * body into a wall -- and not the stronger claim that a bias has no effect near one, which is
     * false.
     */
    @GameTest(maxTicks = 160)
    public void cruiseTurnBiasNeverOverridesAvoidance(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(
                NeuroCraftFly.FLY,
                new Vec3(CruiseProgram.CRUISE_SIDE_CLEARANCE_BLOCKS, 4.0, 1.5));
        fly.setYRot(0.0F);
        PhysicalSensorSnapshot entry = FlySensors.sample(helper.getLevel(), fly);
        helper.assertTrue(
                Math.abs(entry.rayRightBlocks() - CruiseProgram.CRUISE_SIDE_CLEARANCE_BLOCKS)
                                < 0.05
                        && entry.rayLeftBlocks() == FlySensors.RAY_LENGTH_BLOCKS
                        && entry.rayForwardBlocks() == FlySensors.RAY_LENGTH_BLOCKS,
                Component.literal(String.format(
                        Locale.ROOT,
                        "the wall must sit at the declared side clearance on the body's right and"
                                + " nowhere else, but forward=%.3f left=%.3f right=%.3f",
                        entry.rayForwardBlocks(),
                        entry.rayLeftBlocks(),
                        entry.rayRightBlocks())));

        FlyEntity matched = helper.spawn(
                NeuroCraftFly.FLY,
                new Vec3(CruiseProgram.CRUISE_SIDE_CLEARANCE_BLOCKS, 4.0, 1.5));
        matched.setYRot(0.0F);

        driveCruiseProgram(
                helper,
                List.of(fly, matched),
                List.of(new ProgramParameters(1.0), ProgramParameters.NEUTRAL),
                Modulation.NEUTRAL,
                false,
                TurnBiasFence.WALL_TICKS,
                flights -> {
                    List<Flight> flight = flights.getFirst();
                    List<Flight> unbiasedFlight = flights.get(1);
                    double closest = TurnBiasFence.closestApproachToWestWall(flight);
                    long contacts = CruiseFlightFence.wallContactTicks(flight);
                    long unbiased = CruiseFlightFence.wallContactTicks(unbiasedFlight);
                    String measured = String.format(
                            Locale.ROOT,
                            "closest approach %.3f blocks, %d contact ticks of %d against %d for"
                                    + " a matched unbiased body",
                            closest,
                            contacts,
                            flight.size(),
                            unbiased);
                    helper.assertTrue(
                            closest >= TurnBiasFence.WALL_STANDOFF_BLOCKS,
                            Component.literal(
                                    "a full bias into the wall closed past the declared standoff"
                                            + " of " + TurnBiasFence.WALL_STANDOFF_BLOCKS + ": "
                                            + measured));
                    helper.assertTrue(
                            contacts <= unbiased + TurnBiasFence.WALL_CONTACT_BIAS_BUDGET,
                            Component.literal(
                                    "a full bias into the wall cost more contact ticks than the"
                                            + " declared budget of "
                                            + TurnBiasFence.WALL_CONTACT_BIAS_BUDGET + ": "
                                            + measured));
                    helper.assertTrue(
                            contacts <= CruiseFlightFence.WALL_CONTACT_TICK_BUDGET,
                            Component.literal(
                                    "a full bias into the wall pushed the body into it past"
                                            + " CRUISE's own budget of "
                                            + CruiseFlightFence.WALL_CONTACT_TICK_BUDGET + ": "
                                            + measured));
                });
    }

    // ---- driving and verdicts ----------------------------------------------------------------

    /**
     * Engages the scripted lane on each fly and drives one CRUISE step per game tick, then hands
     * the per-fly flight logs to {@code verdict} and succeeds.
     *
     * <p>The tick is owned exactly the way {@code FlyEntity.tick()} owns it under an accepted PAUSE
     * and an admitted neutral command: {@code advance(snapshot, true, true)} once per tick, then
     * {@code PrimitiveActuator.apply}. The entity's own tick still runs and still calls
     * {@code advance} with no admitted command, which retains authority and returns empty, so each
     * game tick carries exactly one program step; the move and the drag come from the entity's own
     * passive-physics integration.
     */
    private static void driveCruise(
            GameTestHelper helper,
            List<FlyEntity> flies,
            Modulation modulation,
            int ticks,
            Consumer<List<List<Flight>>> verdict) {
        List<BodyProgramRuntime> runtimes = new ArrayList<>();
        List<List<Flight>> logs = new ArrayList<>();
        for (FlyEntity fly : flies) {
            BodyProgramRuntime runtime = fly.bodyPrograms();
            runtime.engageScripted(
                    BodyProgramId.CRUISE,
                    modulation,
                    SelectionSource.OPERATOR,
                    CruiseFlightFence.PINNED_ENTRY_TICK);
            helper.assertTrue(
                    runtime.state().id() == BodyProgramId.CRUISE,
                    Component.literal("CRUISE was refused at selection"));
            runtimes.add(runtime);
            logs.add(new ArrayList<>());
        }
        Vec3 origin = helper.absoluteVec(Vec3.ZERO);
        int[] driven = {0};
        helper.onEachTick(() -> {
            if (driven[0] == ticks) {
                verdict.accept(logs);
                helper.succeed();
                return;
            }
            for (int index = 0; index < flies.size(); index++) {
                FlyEntity fly = flies.get(index);
                PhysicalSensorSnapshot snapshot = FlySensors.sample(helper.getLevel(), fly);
                Optional<BodyProgramStep> step = runtimes.get(index).advance(snapshot, true, true);
                helper.assertTrue(
                        step.isPresent(),
                        Component.literal("CRUISE did not own tick " + driven[0]));
                PrimitiveActuator.apply(fly, RawPrimitiveAction.ZERO, step.orElseThrow().motors());
                logs.get(index)
                        .add(new Flight(
                                fly.getX() - origin.x,
                                fly.getZ() - origin.z,
                                fly.getYRot(),
                                snapshot.rayUpBlocks(),
                                snapshot.rayDownBlocks(),
                                snapshot.horizontalCollision(),
                                snapshot.verticalCollision()));
            }
            driven[0]++;
        });
    }

    /**
     * Drives {@code CruiseProgram} directly, one step per game tick, and logs each body's
     * world-space path.
     *
     * <p><strong>Why this driver exists beside {@link #driveCruise}.</strong> A signed turn bias
     * only travels on the readout lane, and the readout lane cannot be held across game ticks in
     * the default GameTest lane: with no bridge configured {@code control()} reports a paused
     * controller every tick, and {@code FlyEntity.tick} releases READOUT authority on exactly that
     * (which is what {@code aLatchedFaultReleasesReadoutAuthority} pins). So a multi-tick biased
     * drive steps the program itself. Everything downstream is identical to {@link #driveCruise}:
     * the same {@code PrimitiveActuator.apply}, the same entity-owned move and drag. What this
     * driver does <em>not</em> exercise is the runtime plumbing, which
     * {@code ReadoutProgramGameTests} covers end to end through the real validator.
     *
     * <p>{@code syntheticFreeFlight} chooses the sensory context: the declared in-band free-flight
     * sample ({@link CruiseFlightFence#inBand}, carrying the body's real yaw and vertical rate) or
     * the authoritative sample the world actually produces. The logged rays and contact flags always
     * come from the real sample either way, so a run that claims to be free flight can be checked
     * against the world rather than against its own context.
     */
    private static void driveCruiseProgram(
            GameTestHelper helper,
            List<FlyEntity> flies,
            List<ProgramParameters> biases,
            Modulation modulation,
            boolean syntheticFreeFlight,
            int ticks,
            Consumer<List<List<Flight>>> verdict) {
        CruiseProgram cruise = new CruiseProgram();
        List<BodyProgramState> states = new ArrayList<>();
        List<List<Flight>> logs = new ArrayList<>();
        for (int index = 0; index < flies.size(); index++) {
            states.add(BodyProgramState.entering(
                    BodyProgramId.CRUISE, CruiseFlightFence.PINNED_ENTRY_TICK));
            logs.add(new ArrayList<>());
        }
        Vec3 origin = helper.absoluteVec(Vec3.ZERO);
        int[] driven = {0};
        helper.onEachTick(() -> {
            if (driven[0] == ticks) {
                verdict.accept(logs);
                helper.succeed();
                return;
            }
            for (int index = 0; index < flies.size(); index++) {
                FlyEntity fly = flies.get(index);
                PhysicalSensorSnapshot snapshot = FlySensors.sample(helper.getLevel(), fly);
                PhysicalSensorSnapshot seen = syntheticFreeFlight
                        ? CruiseFlightFence.inBand(
                                CruiseFlightFence.PINNED_ENTRY_TICK + driven[0],
                                fly.getYRot(),
                                fly.getDeltaMovement().y)
                        : snapshot;
                BodyProgramStep step = cruise.step(
                        states.get(index),
                        BodyProgramContext.sensorsOnly(seen),
                        modulation,
                        biases.get(index));
                PrimitiveActuator.apply(fly, RawPrimitiveAction.ZERO, step.motors());
                states.set(index, step.next());
                logs.get(index)
                        .add(new Flight(
                                fly.getX() - origin.x,
                                fly.getZ() - origin.z,
                                fly.getYRot(),
                                snapshot.rayUpBlocks(),
                                snapshot.rayDownBlocks(),
                                snapshot.horizontalCollision(),
                                snapshot.verticalCollision()));
            }
            driven[0]++;
        });
    }

    /** The three curvature runs, for a failure message. */
    private static String describe(double[] lateral, double[] turned) {
        return String.format(
                Locale.ROOT,
                "bias +1 lateral %+.4f turned %+.1f | bias 0 lateral %+.4f turned %+.1f |"
                        + " bias -1 lateral %+.4f turned %+.1f (blocks toward the body's initial"
                        + " right, and degrees of path turning toward it)",
                lateral[0],
                turned[0],
                lateral[1],
                turned[1],
                lateral[2],
                turned[2]);
    }

    private static void assertDeclaredEnvelope(
            GameTestHelper helper, PrimitiveAction motors, String where) {
        for (double channel : new double[] {
            motors.forwardThrust(),
            motors.lateralThrust(),
            motors.verticalThrust(),
            motors.yawTorque(),
            motors.pitchTorque()
        }) {
            helper.assertTrue(
                    channel >= -1.0 && channel <= 1.0,
                    Component.literal("a motor left the rails: " + channel + where));
        }
        helper.assertTrue(
                motors.lateralThrust() == 0.0 && motors.pitchTorque() == 0.0,
                Component.literal("CRUISE used a channel it declares unused" + where));
        helper.assertTrue(
                motors.forwardThrust() >= 0.0
                        && motors.forwardThrust() <= CruiseProgram.CRUISE_THRUST + EXACT,
                Component.literal("CRUISE left its declared thrust band" + where));
        helper.assertTrue(
                Math.abs(motors.verticalThrust()) <= CruiseProgram.CRUISE_ALTITUDE_THRUST + EXACT,
                Component.literal("CRUISE exceeded its declared altitude thrust" + where));
        helper.assertTrue(
                Math.abs(motors.yawTorque()) <= CruiseProgram.CRUISE_AVOID_TORQUE + EXACT,
                Component.literal("CRUISE exceeded its declared avoidance torque" + where));
    }

    private static void assertSettledClearance(
            GameTestHelper helper, List<Flight> flight, String what) {
        List<Flight> settled = CruiseFlightFence.settledWindow(flight);
        double lowestUp = CruiseFlightFence.lowestUpClearance(settled);
        double lowestDown = CruiseFlightFence.lowestDownClearance(settled);
        double ceilingBound = CruiseProgram.CRUISE_MIN_VERTICAL_CLEARANCE_BLOCKS
                - CruiseFlightFence.CLEARANCE_TOLERANCE_BLOCKS;
        double floorBound = CruiseProgram.CRUISE_MIN_FLOOR_CLEARANCE_BLOCKS
                - CruiseFlightFence.CLEARANCE_TOLERANCE_BLOCKS;
        helper.assertTrue(
                lowestUp >= ceilingBound && lowestDown >= floorBound,
                Component.literal(String.format(
                        Locale.ROOT,
                        "the %s body did not hold its declared clearance: over the last %d ticks"
                                + " the lowest up reading was %.3f and the lowest down reading"
                                + " %.3f, against ceiling %.3f and floor %.3f",
                        what,
                        CruiseFlightFence.SETTLE_WINDOW_TICKS,
                        lowestUp,
                        lowestDown,
                        ceilingBound,
                        floorBound)));

        List<Flight> contacts = CruiseFlightFence.realVerticalContacts(settled);
        helper.assertTrue(
                contacts.isEmpty(),
                Component.literal(String.format(
                        Locale.ROOT,
                        "the %s body was still touching a horizontal surface on %d of the last"
                                + " %d ticks: %s",
                        what,
                        contacts.size(),
                        CruiseFlightFence.SETTLE_WINDOW_TICKS,
                        contacts.stream()
                                .limit(4L)
                                .map(sample -> String.format(
                                        Locale.ROOT,
                                        "[up=%.4f down=%.4f]",
                                        sample.up(),
                                        sample.down()))
                                .toList())));
    }

    private static void assertTurnedAwayFromTheWall(
            GameTestHelper helper, List<Flight> flight, String what) {
        double turned = CruiseFlightFence.headingRange(flight);
        helper.assertTrue(
                turned >= CruiseFlightFence.WALL_TURN_DEGREES,
                Component.literal(String.format(
                        Locale.ROOT,
                        "at %s the body turned only %.1f degrees away from a wall four blocks"
                                + " ahead, against a bound of %.1f",
                        what,
                        turned,
                        CruiseFlightFence.WALL_TURN_DEGREES)));
        long contacts = CruiseFlightFence.wallContactTicks(flight);
        helper.assertTrue(
                contacts <= CruiseFlightFence.WALL_CONTACT_TICK_BUDGET,
                Component.literal(String.format(
                        Locale.ROOT,
                        "at %s the body pushed into the wall on %d of %d ticks, against a budget"
                                + " of %d",
                        what,
                        contacts,
                        flight.size(),
                        CruiseFlightFence.WALL_CONTACT_TICK_BUDGET)));
    }

    private static void assertLeftTheCorner(GameTestHelper helper, List<Flight> flight) {
        int escaped = CruiseFlightFence.cornerEscapeTick(flight);
        helper.assertTrue(
                escaped >= 0 && escaped <= CruiseFlightFence.CORNER_ESCAPE_DEADLINE_TICKS,
                Component.literal(String.format(
                        Locale.ROOT,
                        "the body never got %.1f blocks from the corner inside %d ticks; it"
                                + " reached %.2f",
                        CruiseFlightFence.CORNER_ESCAPE_BLOCKS,
                        CruiseFlightFence.CORNER_ESCAPE_DEADLINE_TICKS,
                        CruiseFlightFence.furthestFromCorner(flight))));
        int longestWedge = CruiseFlightFence.longestWedgeRun(flight);
        helper.assertTrue(
                longestWedge <= CruiseFlightFence.CORNER_WEDGE_TICK_BUDGET,
                Component.literal(String.format(
                        Locale.ROOT,
                        "the body wedged: %d consecutive ticks inside %.1f blocks of the corner,"
                                + " against a budget of %d",
                        longestWedge,
                        CruiseFlightFence.CORNER_NEAR_BLOCKS,
                        CruiseFlightFence.CORNER_WEDGE_TICK_BUDGET)));
    }

    /** Raw bits of CRUISE's five motor channels over {@code ticks} steps of one fixed context. */
    private static long[] cruiseMotorWords(
            BodyProgramContext context,
            long entered,
            Modulation modulation,
            ProgramParameters parameters,
            int ticks) {
        CruiseProgram cruise = new CruiseProgram();
        BodyProgramState state = BodyProgramState.entering(BodyProgramId.CRUISE, entered);
        long[] words = new long[ticks * 5];
        for (int tick = 0; tick < ticks; tick++) {
            BodyProgramStep step = cruise.step(state, context, modulation, parameters);
            PrimitiveAction motors = step.motors();
            words[tick * 5] = Double.doubleToRawLongBits(motors.forwardThrust());
            words[tick * 5 + 1] = Double.doubleToRawLongBits(motors.lateralThrust());
            words[tick * 5 + 2] = Double.doubleToRawLongBits(motors.verticalThrust());
            words[tick * 5 + 3] = Double.doubleToRawLongBits(motors.yawTorque());
            words[tick * 5 + 4] = Double.doubleToRawLongBits(motors.pitchTorque());
            state = step.next();
        }
        return words;
    }

    /** Raw bits of CRUISE's five motor channels over {@code ticks} open-air steps. */
    private static long[] cruiseMotorWords(
            long entered, Modulation modulation, ProgramParameters parameters, int ticks) {
        CruiseProgram cruise = new CruiseProgram();
        BodyProgramState state = BodyProgramState.entering(BodyProgramId.CRUISE, entered);
        long[] words = new long[ticks * 5];
        for (int tick = 0; tick < ticks; tick++) {
            BodyProgramStep step = cruise.step(
                    state,
                    BodyProgramContext.sensorsOnly(CruiseFlightFence.openAir(tick, 0.0, 0.0)),
                    modulation,
                    parameters);
            PrimitiveAction motors = step.motors();
            words[tick * 5] = Double.doubleToRawLongBits(motors.forwardThrust());
            words[tick * 5 + 1] = Double.doubleToRawLongBits(motors.lateralThrust());
            words[tick * 5 + 2] = Double.doubleToRawLongBits(motors.verticalThrust());
            words[tick * 5 + 3] = Double.doubleToRawLongBits(motors.yawTorque());
            words[tick * 5 + 4] = Double.doubleToRawLongBits(motors.pitchTorque());
            state = step.next();
        }
        return words;
    }

    private static BodyProgramContext bodyContext(GameTestHelper helper, FlyEntity fly) {
        return BodyProgramContext.sensorsOnly(FlySensors.sample(helper.getLevel(), fly));
    }
}
