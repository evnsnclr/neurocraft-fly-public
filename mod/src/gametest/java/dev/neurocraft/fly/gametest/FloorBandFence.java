package dev.neurocraft.fly.gametest;

import dev.neurocraft.fly.NeuroCraftFly;
import dev.neurocraft.fly.body.programs.CruiseProgram;
import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import dev.neurocraft.fly.entity.FlyEntity;
import dev.neurocraft.fly.entity.FlySensors;
import java.util.List;
import java.util.Locale;
import net.minecraft.core.BlockPos;
import net.minecraft.gametest.framework.GameTestHelper;
import net.minecraft.network.chat.Component;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.phys.Vec3;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Test-only reference for the CRUISE floor-band pins (lane E2, R-31): the tall arena, where the
 * bodies enter, the declared budgets and bounds, and the metrics the assertions read.
 *
 * <p><b>Why this class exists.</b> Same reason as {@link CruiseFlightFence}: an expectation that
 * restates the code under test agrees with any consistent error in it. Nothing here computes what
 * {@code CruiseProgram}'s vertical law should emit; everything is a place in the world, a declared
 * bound with its own argument, or plain arithmetic over a flight log.
 *
 * <p><b>The arena is the launcher's, in height.</b> {@code scripts/run_interactive_demo.py} seals
 * the live demo in a box whose interior air runs from y 64 to y 87 inclusive
 * ({@code ARENA_INTERIOR_FLOOR_Y}, {@code ARENA_INTERIOR_CEILING_Y}): 24 blocks, floor surface at
 * 64.0, ceiling underside at 88.0 -- which is why the ratcheted fly of 2026-09-04 sat at y 85.4 to
 * 85.8, its declared 2.5 under the lid. The default GameTest structure is 8 blocks tall and cannot
 * hold a descent long enough to be the live one, so this lane ships {@code tall_arena.snbt}: 8 x
 * 24 x 8 of air in Fabric's own {@code empty.snbt} layout (1,536 entries, x then y then z,
 * DataVersion 4325 for 1.21.5), encased in barriers by the framework exactly as the default is. The
 * interior is therefore {@code [0, 8]} on x and z and {@code [0, 24]} on y in structure-relative
 * coordinates, and a relative y is the live y minus {@link #LIVE_FLOOR_Y}; the up and down rays of
 * any body inside sum to {@link #TALL_ARENA_INTERIOR_BLOCKS} before either clamps.
 *
 * <p><b>Historical measured values below describe the old 2.5–4.0 floor band.</b> They were
 * produced by a GameTest in this repository and remain context for the original bounds, not
 * measurements of the new 1.0–1.5 band. Each pin also logs what it measured on the server
 * log under the {@code [floor-band]} prefix ({@link #report}), so a passing run reports its numbers
 * instead of hiding them behind a green tick.
 *
 * <p><b>Nothing here is a claim about the connectome.</b> The band is a body-program constant; a
 * fly seen drifting back down after a jump is evidence of that constant and of nothing else.
 */
final class FloorBandFence {
    /** The 8 x 24 x 8 structure template this lane ships; see the class comment. */
    static final String TALL_ARENA = "neurocraft_fly_gametest:tall_arena";

    /** Interior height of {@link #TALL_ARENA}, in blocks: the launcher's 64..87 inclusive. */
    static final double TALL_ARENA_INTERIOR_BLOCKS = 24.0;

    /** The live arena's floor surface, for translating a relative y in a report. */
    static final int LIVE_FLOOR_Y = 64;

    /**
     * Offset, in blocks, from a body's feet to the bounding-box centre every ray is cast from.
     *
     * <p>Half the declared 0.22 height. Used only to <em>aim</em> a spawn at a clearance; every pin
     * then reads the real ray and asserts it landed within {@link #ENTRY_TOLERANCE_BLOCKS}, so the
     * offset is never itself the premise.
     */
    static final double BODY_CENTRE_OFFSET_BLOCKS = 0.11;

    static final double ENTRY_TOLERANCE_BLOCKS = 0.01;

    /**
     * Feet height, in structure-relative blocks, of the body entered under the ceiling: live y
     * 85.5, the middle of the 85.4-85.8 the ratcheted fly held for its last three minutes. The body
     * centre is then 2.39 under the lid, so the ceiling rule is (barely) live at entry and the
     * floor ray reads the clamp.
     */
    static final double CEILING_ENTRY_FEET_Y = 21.5;

    /**
     * Feet height of the body entered in mid-air: both vertical rays at the clamp, so the descent
     * runs alone with no ceiling rule helping it -- the pure "descending from 6.0 toward a floor"
     * case.
     */
    static final double MID_AIR_ENTRY_FEET_Y = 12.0;

    /**
     * Ticks a body may take to bring its floor ray inside the band from the ceiling entry.
     *
     * <p>Arithmetic on the declared constants: the loop settles at 0.880 of the 0.06 command,
     * 0.0528 blocks/tick, so the 20.1 blocks from the entry centre to the new band top take about 381
     * ticks plus a ten-tick ramp. 500 is the budget the lane brief declared (its own arithmetic
     * was "21 blocks at 0.06, about 350 ticks"), and it is kept rather than tightened to the
     * measurement so the pin fails on a real regression rather than on a physics nudge. Measured:
     * the floor ray is inside the band on tick 332.
     */
    static final int DESCENT_BUDGET_TICKS = 500;

    /**
     * Lowest floor clearance, in blocks, any descent pin may record over its whole drive: the
     * declared minimum less 0.2. The arithmetic above says a body stops about 0.32 under the band
     * top, so the expected lowest reading is about 1.18; the 0.8 bound is 0.2 below the new minimum;
     * it exists to catch a descent that overshoots the band, not to fit one that does not.
     * Measured: 3.7015 from the ceiling entry and 3.7268 from mid-air; 2.7291 over the jump
     * sequence, where the first crouch sinks a body at rest a quarter of a block from 3.0 before
     * the launch.
     */
    static final double LOWEST_CLEARANCE_BLOCKS = 0.8;

    /**
     * Peak-to-peak excursion of the floor ray, in blocks, permitted over a settled window.
     *
     * <p>The band has no term pushing up between 1.0 and 1.5, so a body that has stopped inside it
     * has nothing to oscillate against; a nonzero amplitude would be the rate loop hunting. 0.10 is
     * {@code CruiseFlightFence.CLEARANCE_TOLERANCE_BLOCKS}, the same slack the settled-clearance
     * pins allow.
     */
    static final double SETTLED_AMPLITUDE_BLOCKS = 0.10;

    /** Floor clearance, in blocks, the step pin's body cruises at before the floor rises. */
    static final double STEP_ENTRY_CLEARANCE_BLOCKS = 1.25;

    /** Tick on which the step is laid under the cruising body. */
    static final int STEP_TICK = 40;

    /**
     * Ticks after the step for the body to be back at the declared minimum clearance.
     *
     * <p>Historical arithmetic for the old 2.5-block floor ramp: it was critically damped with a natural frequency of 0.0849
     * rad/tick (see {@code CruiseProgram.CRUISE_ALTITUDE_THRUST}), so its settling time is a few
     * multiples of 12 ticks. 150 is generous, deliberately. Measured: the step on tick 40 takes
     * the floor ray to 2.0000 and the body is back at 2.40 on tick 75, 35 ticks later; it then
     * settles on 2.5000 and never reads above it.
     */
    static final int RECOVERY_BUDGET_TICKS = 150;

    /** Floor clearance, in blocks, the jump-sequence body starts from: inside the new band. */
    static final double JUMP_ENTRY_CLEARANCE_BLOCKS = 1.25;

    static final int ESCAPE_JUMPS = 3;

    /**
     * Ticks of CRUISE between one jump's exit and the next jump's entry: one second.
     *
     * <p>Chosen as a plausible burst cadence, and it turned out to be long enough for the band to
     * act inside the sequence. Measured: jump 1 banks 1.1767 blocks (feet 2.89 to 4.07), the gap
     * gives 0.36 back; jump 2 banks 1.1203, the gap gives 0.82 back; jump 3, launched from a body
     * already descending at the band's rate, banks 0.3675 and ends below the second exit. The
     * floor ray peaks at 4.9378. So the exits do not ratchet monotonically here -- the live
     * session's did because CRUISE never descended -- and the pin asserts the per-jump gain and the
     * return, not a monotone ratchet.
     */
    static final int CRUISE_GAP_TICKS = 20;

    /**
     * Ticks after the last exit within which the floor ray must be back inside the band: 20 s.
     * Measured: 14 ticks, the band having already given back most of the sequence in the gaps.
     */
    static final int GIVE_BACK_BUDGET_TICKS = 400;

    /**
     * Structure-relative y of the stone layer that turns the default 8-block box into a 6-block
     * gap: the layer occupies y 1, so the floor surface is 2.0 and the lid 8.0.
     */
    static final int GAP_FLOOR_LAYER_Y = 1;

    /**
     * Gap-pin entry with both the descent and the ceiling rule live: down 5.0, up 1.0. Measured:
     * settles at down 3.4227 / up 2.5773 with zero excursion over the last 100 ticks -- through the
     * ceiling rule's 2.5 and on into the band by the same stopping distance the open descents show.
     */
    static final double GAP_HIGH_ENTRY_CLEARANCE_BLOCKS = 5.0;

    /** Gap-pin entry with the new climb live: down 0.5, up 5.5. The old entry at 1.2 settled at 2.5000 / 3.5000. */
    static final double GAP_LOW_ENTRY_CLEARANCE_BLOCKS = 0.5;

    private static final Logger LOGGER = LoggerFactory.getLogger("neurocraft-fly-gametest");

    private FloorBandFence() {}

    /**
     * One driven tick: the body's structure-relative altitude (feet) after the program's motors
     * were applied, the vertical rays the program saw, and the contact flags of the sample it saw.
     */
    record Sample(int tick, double y, double up, double down, boolean vertical, boolean onGround) {}

    /** Spawns a body aimed at a floor clearance over a floor whose surface is at {@code floorY}. */
    static FlyEntity spawnAtFloorClearance(
            GameTestHelper helper, double x, double z, double floorY, double clearance) {
        FlyEntity fly = helper.spawn(
                NeuroCraftFly.FLY, new Vec3(x, floorY + clearance - BODY_CENTRE_OFFSET_BLOCKS, z));
        fly.setYRot(0.0F);
        fly.setXRot(0.0F);
        return fly;
    }

    /** Reads the real rays and proves the body entered where the pin says it did. */
    static PhysicalSensorSnapshot assertEntry(
            GameTestHelper helper, FlyEntity fly, double expectedDown, String what) {
        PhysicalSensorSnapshot entry = FlySensors.sample(helper.getLevel(), fly);
        helper.assertTrue(
                Math.abs(entry.rayDownBlocks() - expectedDown) < ENTRY_TOLERANCE_BLOCKS,
                Component.literal(String.format(
                        Locale.ROOT,
                        "%s must enter at %.2f blocks of floor clearance, but the down ray reads"
                                + " %.4f (up %.4f)",
                        what,
                        expectedDown,
                        entry.rayDownBlocks(),
                        entry.rayUpBlocks())));
        return entry;
    }

    /** Lays a full 8 x 8 stone layer at a structure-relative height. */
    static void layStoneLayer(GameTestHelper helper, int y) {
        for (int x = 0; x < 8; x++) {
            for (int z = 0; z < 8; z++) {
                helper.setBlock(new BlockPos(x, y, z), Blocks.STONE);
            }
        }
    }

    /** Inside the declared band, both edges inclusive. */
    static boolean inBand(double down) {
        return down >= CruiseProgram.CRUISE_MIN_FLOOR_CLEARANCE_BLOCKS
                && down <= CruiseProgram.CRUISE_MAX_FLOOR_CLEARANCE_BLOCKS;
    }

    /**
     * Inside the band as a settled body reads it: the top edge exact (inclusive, no slack), the
     * bottom edge with {@link CruiseFlightFence#CLEARANCE_TOLERANCE_BLOCKS} of slack under it.
     *
     * <p>The slack is on one side for a reason. The climb rule's fixed point <em>is</em> the band's
     * bottom edge. Historically the old critically damped floor loop approached it from below and
     * read 2.4999... for as long as it was driven (the step pin measured exactly that: over its
     * last 100 ticks the floor ray ran 2.4999822046 to 2.4999999716, strictly under 2.5 by less
     * than 1.8e-5 and printed as 2.5000 to four decimals). The top edge has no such fixed point --
     * a body above it is being pushed down -- so a settled body must be at or under it exactly.
     */
    static boolean settledInBand(double down) {
        return down >= CruiseProgram.CRUISE_MIN_FLOOR_CLEARANCE_BLOCKS
                        - CruiseFlightFence.CLEARANCE_TOLERANCE_BLOCKS
                && down <= CruiseProgram.CRUISE_MAX_FLOOR_CLEARANCE_BLOCKS;
    }

    /** Index of the first sample whose floor ray is inside the band, or -1 if none is. */
    static int firstIndexInBand(List<Sample> flight) {
        for (int index = 0; index < flight.size(); index++) {
            if (inBand(flight.get(index).down())) {
                return index;
            }
        }
        return -1;
    }

    /** First index at or after {@code from} whose floor ray is at least {@code clearance}. */
    static int firstIndexAtLeast(List<Sample> flight, int from, double clearance) {
        for (int index = from; index < flight.size(); index++) {
            if (flight.get(index).down() >= clearance) {
                return index;
            }
        }
        return -1;
    }

    static List<Sample> settledWindow(List<Sample> flight) {
        return flight.subList(
                flight.size() - CruiseFlightFence.SETTLE_WINDOW_TICKS, flight.size());
    }

    static double lowestDown(List<Sample> window) {
        return window.stream().mapToDouble(Sample::down).min().orElseThrow();
    }

    static double highestDown(List<Sample> window) {
        return window.stream().mapToDouble(Sample::down).max().orElseThrow();
    }

    static double lowestUp(List<Sample> window) {
        return window.stream().mapToDouble(Sample::up).min().orElseThrow();
    }

    static double peakToPeakDown(List<Sample> window) {
        return highestDown(window) - lowestDown(window);
    }

    /**
     * Ticks that report a vertical collision <em>and</em> have a vertical ray short enough for that
     * to be possible; the same predicate as {@link CruiseFlightFence#realVerticalContacts}, and
     * for the same reason ({@code Entity.move} flags a vanishing vertical rate against a wall).
     */
    static List<Sample> realVerticalContacts(List<Sample> flight) {
        return flight.stream()
                .filter(sample -> sample.vertical()
                        && Math.min(sample.up(), sample.down())
                                < CruiseFlightFence.REAL_VERTICAL_CONTACT_CLEARANCE_BLOCKS)
                .toList();
    }

    /**
     * Ticks that report {@code onGround} <em>and</em> have a floor ray short enough for that to be
     * possible.
     *
     * <p>The filter is needed for the same reason {@link #realVerticalContacts} filters:
     * {@code Entity.move} sets {@code onGround = verticalCollision && movement.y < 0}, and the
     * vanishing-rate snap against a wall sets {@code verticalCollision} with nothing under the
     * body. Observed in-world by the first run of this lane, and reproduced with the filter removed:
     * one {@code onGround} tick on the ceiling pin (tick 430, floor ray 3.7015, ceiling ray 6.0)
     * and one on the gap pin (tick 103, floor ray 3.4227, ceiling ray 2.5773), each at the body's
     * settled altitude with {@code verticalCollision} set on the same tick, neither with any
     * vertical ray under 1.0 block -- so the only collider the snap could have found was a wall.
     */
    static long realGroundedTicks(List<Sample> flight) {
        return flight.stream()
                .filter(sample -> sample.onGround()
                        && sample.down() < CruiseFlightFence.REAL_VERTICAL_CONTACT_CLEARANCE_BLOCKS)
                .count();
    }

    /**
     * Asserts a settled window sits inside the band with no hunting and no contact, and returns a
     * one-line description of it for the report.
     */
    static String assertSettledInsideTheBand(
            GameTestHelper helper, List<Sample> flight, String what) {
        List<Sample> settled = settledWindow(flight);
        double lowest = lowestDown(settled);
        double highest = highestDown(settled);
        helper.assertTrue(
                settledInBand(lowest) && settledInBand(highest),
                Component.literal(String.format(
                        Locale.ROOT,
                        "%s did not settle inside the %.1f-%.1f band (%.2f of slack under the"
                                + " bottom edge): over the last %d ticks the down ray ran %.4f to"
                                + " %.4f",
                        what,
                        CruiseProgram.CRUISE_MIN_FLOOR_CLEARANCE_BLOCKS,
                        CruiseProgram.CRUISE_MAX_FLOOR_CLEARANCE_BLOCKS,
                        CruiseFlightFence.CLEARANCE_TOLERANCE_BLOCKS,
                        settled.size(),
                        lowest,
                        highest)));
        helper.assertTrue(
                highest - lowest <= SETTLED_AMPLITUDE_BLOCKS,
                Component.literal(String.format(
                        Locale.ROOT,
                        "%s is still hunting inside the band: a %.4f-block excursion over the last"
                                + " %d ticks, against a declared %.2f",
                        what,
                        highest - lowest,
                        settled.size(),
                        SETTLED_AMPLITUDE_BLOCKS)));
        List<Sample> contacts = realVerticalContacts(flight);
        helper.assertTrue(
                contacts.isEmpty() && realGroundedTicks(flight) == 0,
                Component.literal(String.format(
                        Locale.ROOT,
                        "%s touched a horizontal surface: %d real vertical contacts, %d real"
                                + " grounded ticks, lowest down %.4f, lowest up %.4f",
                        what,
                        contacts.size(),
                        realGroundedTicks(flight),
                        lowestDown(flight),
                        lowestUp(flight))));
        return String.format(
                Locale.ROOT,
                "settled down %.4f-%.4f (up %.4f) over the last %d ticks",
                lowest,
                highest,
                lowestUp(settled),
                settled.size());
    }

    /** Live-arena y for a structure-relative altitude (floor surface 64.0), for reports only. */
    static double liveY(double relativeY) {
        return relativeY + LIVE_FLOOR_Y;
    }

    /** Reports a measurement on the server log, so a passing run shows its numbers. */
    static void report(String pin, String measured) {
        LOGGER.info("[floor-band] {}: {}", pin, measured);
    }
}
