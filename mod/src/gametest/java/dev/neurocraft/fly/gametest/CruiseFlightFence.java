package dev.neurocraft.fly.gametest;

import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import dev.neurocraft.fly.entity.FlySensors;
import java.util.List;
import net.minecraft.util.Mth;

/**
 * Test-only reference for the CRUISE flight pins: the arena as it was measured, the synthetic
 * sensory contexts, the declared bounds, and the metrics the assertions read (ADR 0020 lane B).
 *
 * <p><b>Why this class exists.</b> Same reason as {@link RaySideFence} and
 * {@link RayVerticalFence}: an expectation that restates the code under test agrees with any
 * consistent error in it. Everything here is either a measurement of the world the tests run in, a
 * declared bound with its own argument, or plain arithmetic over a flight log — never a second copy
 * of {@code CruiseProgram}'s law. It also keeps {@code BodyProgramGameTests} to its assertions.
 *
 * <p><b>Every number below that is called observed was produced by a GameTest in this repository
 * and can be reproduced by running the lane.</b> Where a bound was originally chosen against an
 * uncommitted scratch simulation, that is said so explicitly and the in-world value is given
 * beside it. The two disagreed once, materially, and the in-world value is the one that governs.
 *
 * <p><b>The arena, measured rather than assumed.</b> The default Fabric structure is an 8x8x8 empty
 * volume enclosed by barriers. A probe run of {@code BodyProgramGameTests} reported the interior as
 * exactly {@code [0, 8]} on every axis in structure-relative coordinates: a body at relative
 * {@code (4.5, 4.0, 4.5)} facing yaw 0 read forward 3.500, back 4.500, left 3.500, right 4.500, up
 * 3.890 and down 4.110. The last two are 8 minus and plus the 0.11-block offset from the fly's feet
 * to its bounding-box centre, which is where {@code FlySensors} puts every ray origin.
 *
 * <p><b>What this arena cannot provide, stated because one pin depends on it.</b> It has no open
 * air. The interior is 8 blocks across and the rays reach 6.0, so some wall is always inside
 * sensing range and an in-world "wander in open space" run would be measuring avoidance instead.
 * {@link #openAir} is therefore the honest substitute: the sample the six rays actually produce
 * outdoors, all of them at their clamp. Since the floor band (lane E2) that sample also commands
 * the declared descent on the vertical channel, so the free-flight drives that are about the
 * lateral channels use {@link #inBand} instead: the same open horizon with a floor inside the band,
 * where a level body still receives zero vertical thrust.
 */
final class CruiseFlightFence {
    /**
     * Entry game tick handed to {@code engageScripted} by every in-world pin, which becomes
     * {@code BodyProgramState.enteredGameTick} and therefore CRUISE's wander seed.
     *
     * <p><b>Pinning it is a correctness fix, not a convenience.</b> These pins originally engaged at
     * {@code helper.getLevel().getGameTime()}, which differs on every run, so each run sampled a
     * different wander and a bound with one tick of margin could pass and then fail for no reason
     * in the code. The runtime already takes the entry tick from its caller, so a literal makes the
     * whole in-world lane reproducible.
     *
     * <p><b>23 is the worst of the 24 sampled, not the best.</b> An in-world probe drove the
     * four-blocks-from-a-wall scenario at entry ticks 0 through 23 and counted wall-contact ticks
     * over 100 driven ticks: 6 seeds gave 0, 17 gave 1, and seed 23 gave 2. Pinning the seed that
     * produced the most contact means the pin runs the hardest case that probe found rather than a
     * lucky one.
     */
    static final long PINNED_ENTRY_TICK = 23L;

    /**
     * Clearance tolerance, in blocks, on the declared minimum vertical clearance.
     *
     * <p>The old 2.5-block single-surface clearance loop was critically damped (see
     * {@code CruiseProgram.CRUISE_ALTITUDE_THRUST}), and its historical settled reading sat on 2.500
     * in the floor case, so 0.10 is slack rather than a fitted bound. Before the floor band the
     * ceiling case settled on 2.500 as well; it now descends through that clearance into the band
     * and settles inside it, and {@link FloorBandFence} holds those measurements.
     */
    static final double CLEARANCE_TOLERANCE_BLOCKS = 0.10;

    /** Ticks driven in the in-band level-flight pin: 20 s, the demo's own time scale. */
    static final int LEVEL_TICKS = 400;

    /**
     * Floor clearance, in blocks, of the in-band free-flight sample and of the level-flight pin's
     * entry.
     *
     * <p>Strictly inside the declared band: 0.25 above {@code CRUISE_MIN_FLOOR_CLEARANCE_BLOCKS}
     * and 0.25 below {@code CRUISE_MAX_FLOOR_CLEARANCE_BLOCKS}, so neither edge is within reach of
     * any vertical drift a 400-tick drive has produced, and a pin that says "stayed in the band"
     * is saying something about the program rather than about where it started.
     */
    static final double IN_BAND_CLEARANCE_BLOCKS = 1.25;

    /** Ticks driven in the settling pins, and the window at the end that must be settled. */
    static final int SETTLE_TICKS = 300;

    static final int SETTLE_WINDOW_TICKS = 100;

    /**
     * Minimum heading change, in degrees, a body facing a wall four blocks away must produce in
     * 100 ticks.
     *
     * <p>Observed in-world across the 24-seed probe: 248 to 274 degrees. The bound is 90 because a
     * quarter turn is the least that can be called "turned away", and because the executed
     * avoidance-sign negative control produced 33.9 degrees at neutral modulation and 31.5 at
     * {@code turnRate = 0} — so the bound sits in a gap of more than 200 degrees.
     */
    static final double WALL_TURN_DEGREES = 90.0;

    /**
     * Contact ticks allowed while turning away from that wall, out of 100.
     *
     * <p><b>The observed value is a distribution, not a zero.</b> The 24-seed in-world probe gave 0
     * contact ticks on 6 seeds, 1 on 17, and 2 on one — range 0 to 2. An earlier single run of this
     * pin happened to land on a zero seed and was reported as "observed 0"; that was one sample of
     * a spread and the correction is recorded in the CHANGELOG. An independent review probe over
     * its own 24 seeds reported 1 to 2 on 23 of 24; the two probes agree on the range and on the
     * conclusion, and disagree on how many seeds reach zero.
     *
     * <p>5 rather than 2 so the pin is not one tick from flapping: the arena is 8 blocks wide
     * against a 2.36-block minimum turn radius, so a glancing contact is a geometry accident rather
     * than a defect. The avoidance-sign control produces 26 to 28, so a budget of 5 still
     * discriminates by a factor of five.
     */
    static final int WALL_CONTACT_TICK_BUDGET = 5;

    /** Distance from the corner post, in blocks, that counts as having left the corner. */
    static final double CORNER_ESCAPE_BLOCKS = 3.0;

    /**
     * Deadline, in ticks, for leaving it. Observed in-world at {@link #PINNED_ENTRY_TICK}: the body
     * clears 3.0 blocks on tick 43 and reaches 10.82 blocks at its furthest.
     */
    static final int CORNER_ESCAPE_DEADLINE_TICKS = 200;

    /** Distance from the corner post, in blocks, inside which the body counts as still in it. */
    static final double CORNER_NEAR_BLOCKS = 1.5;

    /**
     * Longest run of consecutive ticks the body may spend inside {@link #CORNER_NEAR_BLOCKS}.
     *
     * <p>This is the wedge signature, and it is what the live failure looked like: 83 seconds —
     * 1,660 ticks — in one top corner. <b>The 8-block arena is a closed loop, so a cruising body
     * re-passes every corner and "gets 3 blocks away and stays there" is not a property this arena
     * can express</b>; "leaves, and never wedges again" is. Observed in-world at
     * {@link #PINNED_ENTRY_TICK}: 34 ticks. With the avoidance sign flipped the executed negative
     * control gives 300 consecutive ticks of 300 and never reaches 3.0 blocks, so the budget
     * discriminates by an order of magnitude in both directions.
     */
    static final int CORNER_WEDGE_TICK_BUDGET = 100;

    /**
     * Minimum heading range, in degrees, the wander must cover in open air over
     * {@link #WANDER_TICKS} ticks. Chosen against a scratch-simulation sweep that put the range at
     * 85 to 100 degrees depending on entry tick; that sweep is not reproducible from this
     * repository, but {@code cruiseWandersInOpenSpaceDeterministically} computes the actual range
     * from the shipped code every run and reports it on failure.
     */
    static final double WANDER_RANGE_DEGREES = 45.0;

    static final int WANDER_TICKS = 300;

    /**
     * Floor clearance, in blocks, a body released from the neural lane may not drop below.
     *
     * <p>Observed in-world by {@code cruiseBrakesADescentEnteredFromTheNeuralLane}: entering at
     * {@link #RELEASE_DESCENT_BLOCKS_PER_TICK} with {@link #RELEASE_ENTRY_CLEARANCE_BLOCKS} of
     * clearance, the lowest reading is 0.657 blocks, no tick reports a real vertical contact, and
     * the old-band body recovered to a settled 2.500. That value is historical; this scenario is rerun for the lower band. 0.30 is below that lowest reading and well above the
     * 0.110 a body resting on the floor reads, so the pin separates "braked" from "landed".
     */
    static final double RELEASE_FLOOR_CLEARANCE_BLOCKS = 0.30;

    /**
     * Entry clearance, in blocks, for the release pin.
     *
     * <p><b>This value is load-bearing and the pin's Javadoc says why.</b> The braking term rescues
     * a 0.4 blocks/tick descent only from about 3.5 blocks of clearance or more; from 3.25 and
     * below, and from every clearance tried at the actuator's 0.6 blocks/tick speed cap, the body
     * reaches the floor. 4.0 was inside that old-band window. It remains the fixed entry scenario even though it is above the new band.
     */
    static final double RELEASE_ENTRY_CLEARANCE_BLOCKS = 4.0;

    /** Downward rate, in blocks/tick, the release pin enters with. */
    static final double RELEASE_DESCENT_BLOCKS_PER_TICK = -0.4;

    /**
     * Vertical clearance, in blocks, below which a reported vertical collision is taken at face
     * value.
     *
     * <p>Arithmetic: the body is 0.22 blocks tall, so its centre sits 0.11 from a surface it is
     * resting on, and the actuator's speed cap is 0.6 blocks/tick, so a body that genuinely touched
     * a floor or ceiling on the tick a flag was set reads at most 0.71 on that ray at the next
     * sample. 1.0 is a margin above that.
     */
    static final double REAL_VERTICAL_CONTACT_CLEARANCE_BLOCKS = 1.0;

    private CruiseFlightFence() {}

    /**
     * The open-air sensory sample: every ray at its 6.0-block clamp, no contact, and no rate on any
     * axis except the vertical one the caller names.
     *
     * <p>This is what the six rays report with nothing within sensing range, which is the context
     * CRUISE's lateral channels were written for and the one this arena cannot provide. Since the
     * floor band it is also a context in which CRUISE descends: a clamped floor ray means "no floor
     * within six blocks", and the band answers that with the declared descent rate.
     */
    static PhysicalSensorSnapshot openAir(long gameTick, double yawDegrees, double verticalRate) {
        double clear = FlySensors.RAY_LENGTH_BLOCKS;
        return synthetic(gameTick, yawDegrees, verticalRate, false, clear, clear, clear, clear, clear);
    }

    /**
     * The in-band free-flight sample: every horizontal ray and the up ray at the 6.0-block clamp,
     * the floor ray at {@link #IN_BAND_CLEARANCE_BLOCKS}, no contact, and no rate on any axis except
     * the vertical one the caller names.
     *
     * <p>This is what the rays report cruising over open ground inside the floor band. It is the
     * context the multi-tick free-flight drives use, because inside the band the vertical channel
     * holds zero target rate and the lateral channels are what those drives measure.
     */
    static PhysicalSensorSnapshot inBand(long gameTick, double yawDegrees, double verticalRate) {
        double clear = FlySensors.RAY_LENGTH_BLOCKS;
        return synthetic(
                gameTick,
                yawDegrees,
                verticalRate,
                false,
                clear,
                clear,
                clear,
                clear,
                IN_BAND_CLEARANCE_BLOCKS);
    }

    /**
     * The contexts the envelope pin sweeps. Each is a place the three channels reach a different
     * corner of their declared bounds, so the pin exercises the bounds rather than one operating
     * point.
     */
    static List<PhysicalSensorSnapshot> envelopeSweep() {
        double clear = FlySensors.RAY_LENGTH_BLOCKS;
        return List.of(
                openAir(0L, 0.0, 0.0),
                openAir(1L, 37.0, 0.31),
                openAir(2L, -118.0, -0.31),
                // Nose almost against a wall with both side rays short: full avoidance authority.
                synthetic(3L, 0.0, 0.0, false, 0.2, 0.3, 0.25, clear, clear),
                // A wall on one side only, the other side wide open.
                synthetic(4L, 90.0, 0.0, false, 2.0, 0.4, clear, clear, clear),
                // Hard against the ceiling, sinking.
                synthetic(5L, 0.0, -0.4, false, clear, clear, clear, 0.05, clear),
                // Hard against the floor, climbing.
                synthetic(6L, 0.0, 0.4, false, clear, clear, clear, clear, 0.05),
                // A gap narrower than twice the declared clearance: the two pushes oppose.
                synthetic(7L, 0.0, 0.0, false, clear, clear, clear, 1.0, 1.0),
                // A contact tick.
                synthetic(8L, 214.0, 0.0, true, 0.19, clear, clear, clear, clear));
    }

    /** A sensory sample with the six rays and the vertical rate set and everything else inert. */
    static PhysicalSensorSnapshot synthetic(
            long gameTick,
            double yawDegrees,
            double verticalRate,
            boolean horizontalCollision,
            double forward,
            double left,
            double right,
            double up,
            double down) {
        return new PhysicalSensorSnapshot(
                gameTick,
                "cruise-synthetic-context",
                0.0,
                0.0,
                0.0,
                0.0,
                verticalRate,
                0.0,
                yawDegrees,
                0.0,
                horizontalCollision,
                false,
                false,
                false,
                false,
                0,
                15,
                FlySensors.RAY_LENGTH_BLOCKS,
                forward,
                FlySensors.RAY_LENGTH_BLOCKS,
                left,
                right,
                up,
                down);
    }

    /** The last {@link #SETTLE_WINDOW_TICKS} ticks of a driven flight. */
    static List<Flight> settledWindow(List<Flight> flight) {
        return flight.subList(flight.size() - SETTLE_WINDOW_TICKS, flight.size());
    }

    /**
     * Ticks that report a vertical collision <em>and</em> have a vertical ray short enough for that
     * to be possible.
     *
     * <p><b>The filter is needed because {@code Entity.verticalCollision} can be set with nothing
     * above or below the body, and the mechanism is worth writing down.</b> {@code Entity.move}
     * assigns {@code verticalCollision = requested.y != collided.y} with an exact inequality, and
     * the collision resolver reaches that inequality through {@code Shapes.collide} /
     * {@code VoxelShape.collideX}, which return {@code 0.0} for any {@code |delta| < 1e-7} whenever
     * the candidate collider list is non-empty. So a body with a vanishing vertical rate has its
     * y-movement snapped to zero — and therefore flagged — as soon as its movement-expanded box
     * intersects any block, which is what pushing into a wall does. The mechanism was established
     * in review from the shipped 1.21.5 bytecode; this file did not disassemble it.
     *
     * <p>It was observed here once in 300 driven ticks, at relative {@code z = 0.190} (the body's
     * own half-width from the wall) with the vertical rays reading up 2.5000 and down 5.5000.
     *
     * <p><b>The filter is on the rays rather than on {@code !horizontalCollision}</b>, which was the
     * first attempt and is only a proxy: it would also have excluded a genuine floor contact that
     * happened to occur while scraping a wall. Asking whether a vertical ray is short enough for a
     * touch to be physically possible is the predicate the argument above actually supports.
     */
    static List<Flight> realVerticalContacts(List<Flight> window) {
        return window.stream()
                .filter(sample -> sample.vertical()
                        && Math.min(sample.up(), sample.down())
                                < REAL_VERTICAL_CONTACT_CLEARANCE_BLOCKS)
                .toList();
    }

    static double lowestUpClearance(List<Flight> window) {
        return window.stream().mapToDouble(Flight::up).min().orElseThrow();
    }

    static double lowestDownClearance(List<Flight> window) {
        return window.stream().mapToDouble(Flight::down).min().orElseThrow();
    }

    static double highestDownClearance(List<Flight> window) {
        return window.stream().mapToDouble(Flight::down).max().orElseThrow();
    }

    static long wallContactTicks(List<Flight> flight) {
        return flight.stream().filter(Flight::horizontal).count();
    }

    /** Total heading excursion, in degrees, unwrapped across the 180-degree seam. */
    static double headingRange(List<Flight> flight) {
        double heading = flight.getFirst().yaw();
        double lowest = heading;
        double highest = heading;
        for (int tick = 1; tick < flight.size(); tick++) {
            heading += Mth.degreesDifference(flight.get(tick - 1).yaw(), flight.get(tick).yaw());
            lowest = Math.min(lowest, heading);
            highest = Math.max(highest, heading);
        }
        return highest - lowest;
    }

    /** Horizontal distance from the structure's own {@code (0, 0)} corner post, in blocks. */
    static double cornerDistance(Flight sample) {
        return Math.hypot(sample.x(), sample.z());
    }

    /** First tick past {@link #CORNER_ESCAPE_BLOCKS}, or -1 if the body never got there. */
    static int cornerEscapeTick(List<Flight> flight) {
        for (int tick = 0; tick < flight.size(); tick++) {
            if (cornerDistance(flight.get(tick)) > CORNER_ESCAPE_BLOCKS) {
                return tick;
            }
        }
        return -1;
    }

    static double furthestFromCorner(List<Flight> flight) {
        return flight.stream().mapToDouble(CruiseFlightFence::cornerDistance).max().orElseThrow();
    }

    /** Longest run of consecutive ticks inside {@link #CORNER_NEAR_BLOCKS} of the corner post. */
    static int longestWedgeRun(List<Flight> flight) {
        int longest = 0;
        int current = 0;
        for (Flight sample : flight) {
            current = cornerDistance(sample) < CORNER_NEAR_BLOCKS ? current + 1 : 0;
            longest = Math.max(longest, current);
        }
        return longest;
    }

    /**
     * One driven tick: the body's structure-relative horizontal position and heading after the
     * program's motors were applied, and the sensory readings the program saw on that tick.
     */
    record Flight(
            double x,
            double z,
            float yaw,
            double up,
            double down,
            boolean horizontal,
            boolean vertical) {}
}
