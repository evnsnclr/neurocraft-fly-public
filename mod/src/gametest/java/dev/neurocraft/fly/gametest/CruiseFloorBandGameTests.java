package dev.neurocraft.fly.gametest;

import com.mojang.authlib.GameProfile;
import dev.neurocraft.fly.NeuroCraftFly;
import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.BodyProgramRuntime;
import dev.neurocraft.fly.body.BodyProgramStep;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.SelectionSource;
import dev.neurocraft.fly.body.programs.CruiseProgram;
import dev.neurocraft.fly.body.programs.EscapeJumpProgram;
import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import dev.neurocraft.fly.control.RawPrimitiveAction;
import dev.neurocraft.fly.entity.FlyEntity;
import dev.neurocraft.fly.entity.FlySensors;
import dev.neurocraft.fly.entity.PrimitiveActuator;
import dev.neurocraft.fly.gametest.FloorBandFence.Sample;
import io.netty.channel.embedded.EmbeddedChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.function.IntConsumer;
import net.fabricmc.fabric.api.gametest.v1.GameTest;
import net.minecraft.gametest.framework.GameTestHelper;
import net.minecraft.network.Connection;
import net.minecraft.network.chat.Component;
import net.minecraft.network.protocol.PacketFlow;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.server.network.CommonListenerCookie;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EntityType;
import net.minecraft.world.entity.MoverType;
import net.minecraft.world.entity.animal.Cow;
import net.minecraft.world.level.GameType;
import net.minecraft.world.phys.Vec3;

/**
 * Lane E2 pins for CRUISE's floor band (R-31): the body-side answer to the altitude ratchet the
 * program lane's first live session showed on 2026-09-04.
 *
 * <p>These pins drive {@link BodyProgramRuntime} directly, as {@code BodyProgramGameTests} does and
 * for the same reason: the default GameTest lane has no bridge, so scripted authority cannot be
 * engaged through the real PAUSE path. {@code BodyPackageBoundaryTest} names this file in its
 * {@code engageScripted} allowlist for that. The arena, the entry points, the declared bounds and
 * the metrics live in {@link FloorBandFence}.
 *
 * <p><strong>Nothing here is a claim about the connectome.</strong> The band is a declared
 * engineering constant on a scripted program. These pins measure what the script does to the body
 * in the launcher's own arena height, one authoritative tick at a time, and report what they
 * measured on the server log so a passing run is a measurement and not only a verdict.
 */
public final class CruiseFloorBandGameTests {
    /**
     * <strong>The R-31 scenario, from where it ended.</strong> A body entered into CRUISE 2.39
     * blocks under the lid of the launcher-height arena -- live y 85.5, where the ratcheted fly
     * spent its last three minutes -- brings its floor ray inside the band within the declared
     * budget, settles inside it, never reads below {@link FloorBandFence#LOWEST_CLEARANCE_BLOCKS},
     * and touches neither surface.
     *
     * <p>The band top must sit below the ray clamp for a clamped reading to mean "descend", and the
     * pin asserts that ordering before anything else; the executed negative control
     * ({@code CRUISE_DESCENT_RATE_BLOCKS_PER_TICK = 0}) fails this pin at the budget with the body
     * holding 2.5 under the lid exactly as it did live.
     */
    @GameTest(structure = FloorBandFence.TALL_ARENA, maxTicks = 700)
    public void cruiseDescendsFromTheCeilingIntoTheFloorBand(GameTestHelper helper) {
        helper.assertTrue(
                CruiseProgram.CRUISE_MIN_FLOOR_CLEARANCE_BLOCKS
                                < CruiseProgram.CRUISE_MAX_FLOOR_CLEARANCE_BLOCKS
                        && CruiseProgram.CRUISE_MAX_FLOOR_CLEARANCE_BLOCKS
                                < FlySensors.RAY_LENGTH_BLOCKS,
                Component.literal("the band must be ordered under the ray clamp, or a clamped"
                        + " reading could not mean descend: "
                        + CruiseProgram.CRUISE_MIN_FLOOR_CLEARANCE_BLOCKS + " < "
                        + CruiseProgram.CRUISE_MAX_FLOOR_CLEARANCE_BLOCKS + " < "
                        + FlySensors.RAY_LENGTH_BLOCKS));
        FlyEntity fly = FloorBandFence.spawnAtFloorClearance(
                helper, 4.5, 4.5, 0.0, FloorBandFence.CEILING_ENTRY_FEET_Y
                        + FloorBandFence.BODY_CENTRE_OFFSET_BLOCKS);
        PhysicalSensorSnapshot entry = FlySensors.sample(helper.getLevel(), fly);
        helper.assertTrue(
                entry.rayUpBlocks() < CruiseProgram.CRUISE_MIN_VERTICAL_CLEARANCE_BLOCKS
                        && entry.rayDownBlocks() == FlySensors.RAY_LENGTH_BLOCKS,
                Component.literal(String.format(
                        Locale.ROOT,
                        "the body must enter under the lid with its floor ray at the clamp, but"
                                + " up=%.4f down=%.4f",
                        entry.rayUpBlocks(),
                        entry.rayDownBlocks())));

        driveCruise(
                helper,
                List.of(fly),
                FloorBandFence.DESCENT_BUDGET_TICKS + CruiseFlightFence.SETTLE_WINDOW_TICKS,
                null,
                flights -> {
                    List<Sample> flight = flights.getFirst();
                    int entered = FloorBandFence.firstIndexInBand(flight);
                    helper.assertTrue(
                            entered >= 0 && entered <= FloorBandFence.DESCENT_BUDGET_TICKS,
                            Component.literal(String.format(
                                    Locale.ROOT,
                                    "the floor ray did not reach the band inside %d ticks: first"
                                            + " in-band tick %d, down ray ended at %.4f (live y"
                                            + " %.2f)",
                                    FloorBandFence.DESCENT_BUDGET_TICKS,
                                    entered,
                                    flight.getLast().down(),
                                    FloorBandFence.liveY(flight.getLast().y()))));
                    double lowest = FloorBandFence.lowestDown(flight);
                    helper.assertTrue(
                            lowest >= FloorBandFence.LOWEST_CLEARANCE_BLOCKS,
                            Component.literal(String.format(
                                    Locale.ROOT,
                                    "the descent overshot the band: lowest down ray %.4f against"
                                            + " a declared %.2f",
                                    lowest,
                                    FloorBandFence.LOWEST_CLEARANCE_BLOCKS)));
                    String settled = FloorBandFence.assertSettledInsideTheBand(
                            helper, flight, "the body entered under the lid");
                    FloorBandFence.report(
                            "cruiseDescendsFromTheCeilingIntoTheFloorBand",
                            String.format(
                                    Locale.ROOT,
                                    "entered up %.4f down %.4f (live y %.2f); first in-band tick"
                                            + " %d; lowest down %.4f; %s; final live y %.2f",
                                    entry.rayUpBlocks(),
                                    entry.rayDownBlocks(),
                                    FloorBandFence.liveY(FloorBandFence.CEILING_ENTRY_FEET_Y),
                                    entered,
                                    lowest,
                                    settled,
                                    FloorBandFence.liveY(flight.getLast().y())));
                });
    }

    /**
     * <strong>The minimum-clearance climb keeps its priority, and the descent stops where it
     * should.</strong> Two bodies in the launcher-height arena. The first cruises at 1.25 blocks of
     * floor clearance until, on {@link FloorBandFence#STEP_TICK}, a one-block stone step is laid
     * under the whole floor: its floor ray drops to 0.25, it climbs back to the declared minimum
     * inside {@link FloorBandFence#RECOVERY_BUDGET_TICKS}, and it then stays inside the band --
     * never overshooting its top into the descent, which is the one way the two rules could
     * alternate. The second enters in mid-air with both vertical rays at the clamp, so the descent
     * runs alone toward the floor, and its floor ray never reads under the declared minimum.
     *
     * <p>Structurally the climb and the descent cannot both be live -- the climb is under 1.0, the
     * descent over 1.5 -- and {@code CruiseFloorBandTest} pins that on the law. This pin is the
     * in-world half: a step under a cruising body climbs, as it did before the band.
     */
    @GameTest(structure = FloorBandFence.TALL_ARENA, maxTicks = 400)
    public void cruiseClimbStillWinsOverDescent(GameTestHelper helper) {
        FlyEntity stepped = FloorBandFence.spawnAtFloorClearance(
                helper, 2.5, 2.5, 0.0, FloorBandFence.STEP_ENTRY_CLEARANCE_BLOCKS);
        FlyEntity midAir = FloorBandFence.spawnAtFloorClearance(
                helper, 5.5, 5.5, 0.0, FloorBandFence.MID_AIR_ENTRY_FEET_Y
                        + FloorBandFence.BODY_CENTRE_OFFSET_BLOCKS);
        PhysicalSensorSnapshot steppedEntry = FloorBandFence.assertEntry(
                helper, stepped, FloorBandFence.STEP_ENTRY_CLEARANCE_BLOCKS, "the stepped body");
        PhysicalSensorSnapshot midAirEntry = FlySensors.sample(helper.getLevel(), midAir);
        helper.assertTrue(
                midAirEntry.rayDownBlocks() == FlySensors.RAY_LENGTH_BLOCKS
                        && midAirEntry.rayUpBlocks() == FlySensors.RAY_LENGTH_BLOCKS,
                Component.literal(String.format(
                        Locale.ROOT,
                        "the mid-air body must see neither surface, but up=%.4f down=%.4f",
                        midAirEntry.rayUpBlocks(),
                        midAirEntry.rayDownBlocks())));

        driveCruise(
                helper,
                List.of(stepped, midAir),
                CruiseFlightFence.SETTLE_TICKS,
                tick -> {
                    if (tick == FloorBandFence.STEP_TICK) {
                        FloorBandFence.layStoneLayer(helper, 0);
                    }
                },
                flights -> {
                    List<Sample> flight = flights.get(0);
                    List<Sample> afterStep = flight.subList(FloorBandFence.STEP_TICK, flight.size());
                    double steppedTo = afterStep.getFirst().down();
                    helper.assertTrue(
                            steppedTo < CruiseProgram.CRUISE_MIN_FLOOR_CLEARANCE_BLOCKS,
                            Component.literal(String.format(
                                    Locale.ROOT,
                                    "the step did not put the body under its minimum clearance:"
                                            + " down ray %.4f on tick %d",
                                    steppedTo,
                                    FloorBandFence.STEP_TICK)));
                    int recovered = FloorBandFence.firstIndexAtLeast(
                            flight,
                            FloorBandFence.STEP_TICK,
                            CruiseProgram.CRUISE_MIN_FLOOR_CLEARANCE_BLOCKS
                                    - CruiseFlightFence.CLEARANCE_TOLERANCE_BLOCKS);
                    helper.assertTrue(
                            recovered >= 0
                                    && recovered - FloorBandFence.STEP_TICK
                                            <= FloorBandFence.RECOVERY_BUDGET_TICKS,
                            Component.literal(String.format(
                                    Locale.ROOT,
                                    "the body did not climb back to its minimum clearance inside"
                                            + " %d ticks of the step (recovered on tick %d)",
                                    FloorBandFence.RECOVERY_BUDGET_TICKS,
                                    recovered)));
                    double highestAfterStep = FloorBandFence.highestDown(afterStep);
                    helper.assertTrue(
                            highestAfterStep <= CruiseProgram.CRUISE_MAX_FLOOR_CLEARANCE_BLOCKS,
                            Component.literal(String.format(
                                    Locale.ROOT,
                                    "the climb overshot the band top and handed the body to the"
                                            + " descent: highest down ray %.4f after the step",
                                    highestAfterStep)));
                    String steppedSettled = FloorBandFence.assertSettledInsideTheBand(
                            helper, flight, "the stepped body");

                    List<Sample> descent = flights.get(1);
                    double lowest = FloorBandFence.lowestDown(descent);
                    helper.assertTrue(
                            lowest >= CruiseProgram.CRUISE_MIN_FLOOR_CLEARANCE_BLOCKS,
                            Component.literal(String.format(
                                    Locale.ROOT,
                                    "a body descending from the clamp crossed the minimum"
                                            + " clearance: lowest down ray %.4f",
                                    lowest)));
                    int entered = FloorBandFence.firstIndexInBand(descent);
                    helper.assertTrue(
                            entered >= 0,
                            Component.literal("the mid-air body never reached the band; its down"
                                    + " ray ended at " + descent.getLast().down()));
                    String descentSettled = FloorBandFence.assertSettledInsideTheBand(
                            helper, descent, "the mid-air body");
                    FloorBandFence.report(
                            "cruiseClimbStillWinsOverDescent",
                            String.format(
                                    Locale.ROOT,
                                    "stepped body: entered down %.4f, step on tick %d read %.4f,"
                                            + " back to %.2f on tick %d (%d ticks), lowest %.4f,"
                                            + " highest after the step %.4f, %s | mid-air body:"
                                            + " entered down %.4f, first in-band tick %d, lowest"
                                            + " %.4f, %s",
                                    steppedEntry.rayDownBlocks(),
                                    FloorBandFence.STEP_TICK,
                                    steppedTo,
                                    CruiseProgram.CRUISE_MIN_FLOOR_CLEARANCE_BLOCKS
                                            - CruiseFlightFence.CLEARANCE_TOLERANCE_BLOCKS,
                                    recovered,
                                    recovered - FloorBandFence.STEP_TICK,
                                    FloorBandFence.lowestDown(afterStep),
                                    highestAfterStep,
                                    steppedSettled,
                                    midAirEntry.rayDownBlocks(),
                                    entered,
                                    lowest,
                                    descentSettled));
                });
    }

    /**
     * <strong>The pin that says R-31 is closed.</strong> Three ESCAPE_JUMPs in sequence, each
     * driven through {@link BodyProgramRuntime} from a real {@code engageScripted} and each exiting
     * to CRUISE on its own 24-tick clock as a real transition, with
     * {@link FloorBandFence#CRUISE_GAP_TICKS} of CRUISE between them. Every jump ends higher than
     * it started, the sequence takes the floor ray above the band top, and CRUISE then brings it
     * back inside the band within {@link FloorBandFence#GIVE_BACK_BUDGET_TICKS} of the last exit
     * and settles there, never touching either surface.
     *
     * <p><strong>What is deliberately not asserted, because the historical old-band run measured it.</strong>
     * The exits do not ratchet monotonically with one second of CRUISE between jumps: the first
     * two jumps banked 1.18 and 1.12 blocks, but the band gave 0.36 and then 0.82 back in the gaps,
     * and the third jump -- launched from a body already descending at the band's rate -- banked
     * only 0.37, ending below the second exit. That is the fix acting between jumps, not a defect,
     * and it is what the live session could not do; the per-jump gains and the give-back are in
     * the report line so a reviewer can see the dynamics rather than a verdict.
     *
     * <p>ESCAPE_JUMP is unchanged by lane E2. What the pin measures is that the successor it names
     * now gives back what the jump banked, which is the whole of the fix.
     */
    @GameTest(structure = FloorBandFence.TALL_ARENA, maxTicks = 800)
    public void escapeJumpAltitudeIsGivenBackByCruise(GameTestHelper helper) {
        FlyEntity fly = FloorBandFence.spawnAtFloorClearance(
                helper, 4.5, 4.5, 0.0, FloorBandFence.JUMP_ENTRY_CLEARANCE_BLOCKS);
        PhysicalSensorSnapshot entry = FloorBandFence.assertEntry(
                helper, fly, FloorBandFence.JUMP_ENTRY_CLEARANCE_BLOCKS, "the jumping body");
        helper.assertTrue(
                FloorBandFence.inBand(entry.rayDownBlocks())
                        && entry.rayUpBlocks() == FlySensors.RAY_LENGTH_BLOCKS,
                Component.literal(String.format(
                        Locale.ROOT,
                        "the body must start inside the band with open air above, so the launch"
                                + " is unscaled, but down=%.4f up=%.4f",
                        entry.rayDownBlocks(),
                        entry.rayUpBlocks())));
        BodyProgramRuntime runtime = fly.bodyPrograms();
        double[] entryY = new double[FloorBandFence.ESCAPE_JUMPS];
        double[] exitY = new double[FloorBandFence.ESCAPE_JUMPS];
        int[] exitIndex = new int[FloorBandFence.ESCAPE_JUMPS];
        int[] jumpsStarted = {0};
        int[] jumpsDone = {0};
        int[] jumpTick = {0};
        int[] sinceExit = {0};
        int[] driven = {0};
        boolean[] inJump = {false};
        List<Sample> flight = new ArrayList<>();
        double originY = helper.absoluteVec(Vec3.ZERO).y;
        int settleAfterLastExit =
                FloorBandFence.GIVE_BACK_BUDGET_TICKS + CruiseFlightFence.SETTLE_WINDOW_TICKS;

        helper.onEachTick(() -> {
            PhysicalSensorSnapshot snapshot = FlySensors.sample(helper.getLevel(), fly);
            if (inJump[0] && jumpTick[0] == EscapeJumpProgram.ESCAPE_JUMP_TICKS) {
                // One observation tick after the last commanded jump tick, so the final move has
                // been integrated before the exit altitude is read -- as EscapeArenaFence.JumpRun
                // reads its rise.
                helper.assertTrue(
                        runtime.state().id() == BodyProgramId.CRUISE
                                && runtime.transitionLedger().getLast().contains("-> CRUISE")
                                && runtime.transitionLedger().getLast().contains("reason=exit"),
                        Component.literal("jump " + (jumpsDone[0] + 1) + " did not exit into"
                                + " CRUISE as a real transition: " + runtime.transitionLedger()));
                exitY[jumpsDone[0]] = fly.getY() - originY;
                exitIndex[jumpsDone[0]] = flight.size();
                jumpsDone[0]++;
                inJump[0] = false;
                sinceExit[0] = 0;
            }
            if (!inJump[0]
                    && jumpsStarted[0] < FloorBandFence.ESCAPE_JUMPS
                    && (jumpsStarted[0] == 0 || sinceExit[0] == FloorBandFence.CRUISE_GAP_TICKS)) {
                entryY[jumpsStarted[0]] = fly.getY() - originY;
                runtime.engageScripted(
                        BodyProgramId.ESCAPE_JUMP,
                        Modulation.NEUTRAL,
                        SelectionSource.OPERATOR,
                        helper.getLevel().getGameTime());
                helper.assertTrue(
                        runtime.state().id() == BodyProgramId.ESCAPE_JUMP,
                        Component.literal("ESCAPE_JUMP was refused at selection "
                                + (jumpsStarted[0] + 1)));
                jumpsStarted[0]++;
                inJump[0] = true;
                jumpTick[0] = 0;
            }
            if (jumpsDone[0] == FloorBandFence.ESCAPE_JUMPS && sinceExit[0] == settleAfterLastExit) {
                assertAltitudeGivenBack(helper, flight, entryY, exitY, exitIndex);
                helper.succeed();
                return;
            }
            Optional<BodyProgramStep> step = runtime.advance(snapshot, true, true);
            helper.assertTrue(
                    step.isPresent(),
                    Component.literal("the engaged program did not own tick " + driven[0]));
            PrimitiveActuator.apply(fly, RawPrimitiveAction.ZERO, step.orElseThrow().motors());
            flight.add(new Sample(
                    driven[0],
                    fly.getY() - originY,
                    snapshot.rayUpBlocks(),
                    snapshot.rayDownBlocks(),
                    snapshot.verticalCollision(),
                    snapshot.onGround()));
            if (inJump[0]) {
                jumpTick[0]++;
            } else {
                sinceExit[0]++;
            }
            driven[0]++;
        });
    }

    private static void assertAltitudeGivenBack(
            GameTestHelper helper,
            List<Sample> flight,
            double[] entryY,
            double[] exitY,
            int[] exitIndex) {
        StringBuilder jumps = new StringBuilder();
        for (int jump = 0; jump < FloorBandFence.ESCAPE_JUMPS; jump++) {
            jumps.append(String.format(
                    Locale.ROOT,
                    "jump %d: y %.4f -> %.4f (+%.4f; live %.2f -> %.2f)%s",
                    jump + 1,
                    entryY[jump],
                    exitY[jump],
                    exitY[jump] - entryY[jump],
                    FloorBandFence.liveY(entryY[jump]),
                    FloorBandFence.liveY(exitY[jump]),
                    jump + 1 < FloorBandFence.ESCAPE_JUMPS ? "; " : ""));
        }
        for (int jump = 0; jump < FloorBandFence.ESCAPE_JUMPS; jump++) {
            helper.assertTrue(
                    exitY[jump] > entryY[jump],
                    Component.literal("jump " + (jump + 1) + " did not bank altitude, so there is"
                            + " nothing for the band to give back: " + jumps));
        }
        double highest = FloorBandFence.highestDown(flight);
        helper.assertTrue(
                highest > CruiseProgram.CRUISE_MAX_FLOOR_CLEARANCE_BLOCKS,
                Component.literal(String.format(
                        Locale.ROOT,
                        "the jumps never took the floor ray above the band top (highest %.4f), so"
                                + " the run does not exercise the give-back: %s",
                        highest,
                        jumps)));
        List<Sample> afterLastExit = flight.subList(exitIndex[exitIndex.length - 1], flight.size());
        int entered = FloorBandFence.firstIndexInBand(afterLastExit);
        helper.assertTrue(
                entered >= 0 && entered <= FloorBandFence.GIVE_BACK_BUDGET_TICKS,
                Component.literal(String.format(
                        Locale.ROOT,
                        "CRUISE did not bring the floor ray back inside the band within %d ticks"
                                + " of the last exit (first in-band tick %d; down ray ended at"
                                + " %.4f): %s",
                        FloorBandFence.GIVE_BACK_BUDGET_TICKS,
                        entered,
                        flight.getLast().down(),
                        jumps)));
        double lowest = FloorBandFence.lowestDown(flight);
        helper.assertTrue(
                lowest >= FloorBandFence.LOWEST_CLEARANCE_BLOCKS,
                Component.literal(String.format(
                        Locale.ROOT,
                        "the sequence brought the body to %.4f blocks of floor clearance, under"
                                + " the declared %.2f",
                        lowest,
                        FloorBandFence.LOWEST_CLEARANCE_BLOCKS)));
        String settled = FloorBandFence.assertSettledInsideTheBand(
                helper, flight, "the body after three jumps");
        FloorBandFence.report(
                "escapeJumpAltitudeIsGivenBackByCruise",
                String.format(
                        Locale.ROOT,
                        "%s; highest down ray %.4f; back in the band %d ticks after the last exit;"
                                + " lowest down %.4f; %s; final live y %.2f",
                        jumps,
                        highest,
                        entered,
                        lowest,
                        settled,
                        FloorBandFence.liveY(flight.getLast().y())));
    }

    /**
     * <strong>Both vertical rules live at once, and no hunting.</strong> The default 8-block box
     * with a stone layer laid at y 1 is a 6-block gap. At the high entry the descent and ceiling rule are both live;
     * lower down the floor-only descent continues until the new band. A body
     * entered at the top (down 5.0, up 1.0) and a body entered near the floor (down 0.5, up 5.5)
     * both settle inside the band with an excursion under
     * {@link FloorBandFence#SETTLED_AMPLITUDE_BLOCKS} over the last 100 ticks and touch neither
     * surface.
     *
     * <p>The two rules push the same way, so there is nothing to oscillate between; the pin exists
     * because the lane brief asked for the gap case to be measured rather than argued.
     */
    @GameTest(maxTicks = 400)
    public void cruiseHoldsInANarrowGap(GameTestHelper helper) {
        FloorBandFence.layStoneLayer(helper, FloorBandFence.GAP_FLOOR_LAYER_Y);
        double floorY = FloorBandFence.GAP_FLOOR_LAYER_Y + 1.0;
        FlyEntity high = FloorBandFence.spawnAtFloorClearance(
                helper, 2.5, 2.5, floorY, FloorBandFence.GAP_HIGH_ENTRY_CLEARANCE_BLOCKS);
        FlyEntity low = FloorBandFence.spawnAtFloorClearance(
                helper, 5.5, 5.5, floorY, FloorBandFence.GAP_LOW_ENTRY_CLEARANCE_BLOCKS);
        PhysicalSensorSnapshot highEntry = FloorBandFence.assertEntry(
                helper, high, FloorBandFence.GAP_HIGH_ENTRY_CLEARANCE_BLOCKS, "the high body");
        PhysicalSensorSnapshot lowEntry = FloorBandFence.assertEntry(
                helper, low, FloorBandFence.GAP_LOW_ENTRY_CLEARANCE_BLOCKS, "the low body");
        helper.assertTrue(
                highEntry.rayUpBlocks() < CruiseProgram.CRUISE_MIN_VERTICAL_CLEARANCE_BLOCKS
                        && highEntry.rayDownBlocks()
                                > CruiseProgram.CRUISE_MAX_FLOOR_CLEARANCE_BLOCKS
                        && Math.abs(highEntry.rayUpBlocks() + highEntry.rayDownBlocks() - 6.0)
                                < FloorBandFence.ENTRY_TOLERANCE_BLOCKS,
                Component.literal(String.format(
                        Locale.ROOT,
                        "the high body must have both rules live in a 6-block gap, but up=%.4f"
                                + " down=%.4f",
                        highEntry.rayUpBlocks(),
                        highEntry.rayDownBlocks())));

        driveCruise(
                helper,
                List.of(high, low),
                CruiseFlightFence.SETTLE_TICKS,
                null,
                flights -> {
                    String highSettled = FloorBandFence.assertSettledInsideTheBand(
                            helper, flights.get(0), "the high body");
                    String lowSettled = FloorBandFence.assertSettledInsideTheBand(
                            helper, flights.get(1), "the low body");
                    helper.assertTrue(
                            FloorBandFence.lowestUp(FloorBandFence.settledWindow(flights.get(0)))
                                    >= CruiseProgram.CRUISE_MIN_VERTICAL_CLEARANCE_BLOCKS
                                            - CruiseFlightFence.CLEARANCE_TOLERANCE_BLOCKS,
                            Component.literal("the high body settled inside the band but still"
                                    + " inside its ceiling clearance: " + highSettled));
                    FloorBandFence.report(
                            "cruiseHoldsInANarrowGap",
                            String.format(
                                    Locale.ROOT,
                                    "high body entered up %.4f down %.4f, %s; low body entered up"
                                            + " %.4f down %.4f, %s",
                                    highEntry.rayUpBlocks(),
                                    highEntry.rayDownBlocks(),
                                    highSettled,
                                    lowEntry.rayUpBlocks(),
                                    lowEntry.rayDownBlocks(),
                                    lowSettled));
                });
    }

    /**
     * Actual CRUISE descent and settling, followed by a geometry fixture of ordinary grounded
     * Player/Cow bounding boxes. Actors are placed once on the floor; Entity.move resolves their
     * downward preparation against the real floor. The fly's settled position and velocity are
     * untouched; only its yaw is aimed inward for the independent cardinal ray measurement.
     * This proves sensor reachability at the observed height, not walking or neural selection.
     */
    @GameTest(maxTicks = 400)
    public void settledCruiseRaysReachGroundedPlayerAndCow(GameTestHelper helper) {
        FlyEntity fly = FloorBandFence.spawnAtFloorClearance(helper, 4.5, 4.5, 0.0, 5.0);
        driveCruise(helper, List.of(fly), CruiseFlightFence.SETTLE_TICKS, null, flights -> {
            String settled = FloorBandFence.assertSettledInsideTheBand(
                    helper, flights.getFirst(), "the actor-reachability body");
            Vec3 origin = helper.absoluteVec(Vec3.ZERO);
            Vec3 centre = fly.getBoundingBox().getCenter();
            double forwardSign = centre.z - origin.z < 4.0 ? 1.0 : -1.0;
            double sideSign = centre.x - origin.x < 4.0 ? 1.0 : -1.0;
            fly.setYRot(forwardSign > 0.0 ? 0.0F : 180.0F);
            helper.assertTrue(fly.getXRot() == 0.0F, Component.literal("CRUISE pitched the sensor rays"));
            var profile = new GameProfile(UUID.randomUUID(), "cruise-height-test");
            var cookie = CommonListenerCookie.createInitial(profile, false);
            var level = helper.getLevel();
            ServerPlayer player = new ServerPlayer(level.getServer(), level, profile, cookie.clientInformation());
            Connection connection = new Connection(PacketFlow.SERVERBOUND);
            new EmbeddedChannel(connection);
            level.getServer().getPlayerList().placeNewPlayer(connection, player, cookie);
            Cow cow = helper.spawn(EntityType.COW, new Vec3(4.0, 0.0, 4.0));
            try {
                player.setGameMode(GameType.SURVIVAL);
                player.setPos(centre.x, origin.y, centre.z + 2.0 * forwardSign);
                cow.setNoAi(true);
                cow.setPos(centre.x + 2.0 * sideSign, origin.y, centre.z);
                // Preparation-only actual collision resolution, not a fabricated onGround flag.
                player.move(MoverType.SELF, new Vec3(0.0, -0.01, 0.0));
                cow.move(MoverType.SELF, new Vec3(0.0, -0.01, 0.0));
                helper.assertTrue(player.onGround() && cow.onGround()
                                && !player.getAbilities().flying && !player.isSpectator()
                                && player.isAlive() && cow.isAlive(),
                        Component.literal("ordinary actors did not stand on the real floor"));
                helper.assertTrue(centre.y > player.getBoundingBox().minY
                                && centre.y < player.getBoundingBox().maxY
                                && centre.y > cow.getBoundingBox().minY
                                && centre.y < cow.getBoundingBox().maxY,
                        Component.literal("settled ray origin is above a grounded actor's actual body"));
                PhysicalSensorSnapshot seen = FlySensors.sample(level, fly);
                int cowRay = sideSign == forwardSign ? 2 : 3;
                helper.assertTrue(seen.rayActorKinds().getFirst() == 1
                                && seen.rayActorKinds().get(cowRay) == 2,
                        Component.literal("settled CRUISE did not see Player/front and Cow/side: "
                                + seen.rayActorKinds()));
                double expectedPlayer = Math.abs(player.getZ() - centre.z)
                        - player.getBoundingBox().getZsize() / 2.0;
                double cowDistance = cowRay == 2 ? seen.rayLeftBlocks() : seen.rayRightBlocks();
                double expectedCow = Math.abs(cow.getX() - centre.x)
                        - cow.getBoundingBox().getXsize() / 2.0;
                helper.assertTrue(Math.abs(seen.rayForwardBlocks() - expectedPlayer) < 1.0e-6
                                && Math.abs(cowDistance - expectedCow) < 1.0e-6,
                        Component.literal("actor ray distance did not match the actual near face"));
                // A separate height-only fixture proves this pin discriminates the old band.
                FlyEntity oldHeight = helper.spawn(NeuroCraftFly.FLY,
                        new Vec3(centre.x - origin.x, 3.2 - FloorBandFence.BODY_CENTRE_OFFSET_BLOCKS,
                                centre.z - origin.z));
                oldHeight.setYRot(fly.getYRot());
                PhysicalSensorSnapshot oldSeen = FlySensors.sample(level, oldHeight);
                helper.assertTrue(oldSeen.rayActorKinds().getFirst() == 0
                                && oldSeen.rayActorKinds().get(cowRay) == 0,
                        Component.literal("old-height fixture unexpectedly saw grounded actors"));
                oldHeight.discard();
                FloorBandFence.report("settledCruiseRaysReachGroundedPlayerAndCow", String.format(
                        Locale.ROOT, "%s; ray height %.4f; Player top %.4f/ray %.4f; Cow top %.4f/ray %.4f;"
                                + " grounded Player/Cow; old-height 3.2 sees neither; no neural selection",
                        settled, centre.y - origin.y, player.getBoundingBox().maxY - origin.y,
                        seen.rayForwardBlocks(), cow.getBoundingBox().maxY - origin.y, cowDistance));
            } finally {
                cow.discard();
                level.removePlayerImmediately(player, Entity.RemovalReason.DISCARDED);
            }
        });
    }

    // ---- driving ---------------------------------------------------------------------------

    /**
     * Engages the scripted lane on each fly at {@link CruiseFlightFence#PINNED_ENTRY_TICK} and
     * drives one CRUISE step per game tick through the runtime and the real actuator, exactly as
     * {@code BodyProgramGameTests.driveCruise} does; {@code worldEvent}, if given, runs before the
     * sample on each tick so a pin can change the world under a cruising body.
     */
    private static void driveCruise(
            GameTestHelper helper,
            List<FlyEntity> flies,
            int ticks,
            IntConsumer worldEvent,
            Consumer<List<List<Sample>>> verdict) {
        List<BodyProgramRuntime> runtimes = new ArrayList<>();
        List<List<Sample>> logs = new ArrayList<>();
        for (FlyEntity fly : flies) {
            BodyProgramRuntime runtime = fly.bodyPrograms();
            runtime.engageScripted(
                    BodyProgramId.CRUISE,
                    Modulation.NEUTRAL,
                    SelectionSource.OPERATOR,
                    CruiseFlightFence.PINNED_ENTRY_TICK);
            helper.assertTrue(
                    runtime.state().id() == BodyProgramId.CRUISE,
                    Component.literal("CRUISE was refused at selection"));
            runtimes.add(runtime);
            logs.add(new ArrayList<>());
        }
        double originY = helper.absoluteVec(Vec3.ZERO).y;
        int[] driven = {0};
        helper.onEachTick(() -> {
            if (driven[0] == ticks) {
                verdict.accept(logs);
                helper.succeed();
                return;
            }
            if (worldEvent != null) {
                worldEvent.accept(driven[0]);
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
                        .add(new Sample(
                                driven[0],
                                fly.getY() - originY,
                                snapshot.rayUpBlocks(),
                                snapshot.rayDownBlocks(),
                                snapshot.verticalCollision(),
                                snapshot.onGround()));
            }
            driven[0]++;
        });
    }
}
