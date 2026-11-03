package dev.neurocraft.fly.gametest;

import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.programs.CruiseProgram;
import dev.neurocraft.fly.entity.PrimitiveActuator;
import dev.neurocraft.fly.gametest.CruiseFlightFence.Flight;
import java.util.List;
import net.minecraft.util.Mth;
import net.minecraft.world.phys.Vec3;

/**
 * Test-only reference for the CRUISE turn-bias pins: the declared drive settings, the bounds, and
 * the world-space path metrics the assertions read.
 *
 * <p><b>Why this class exists.</b> Same reason as {@link CruiseFlightFence} and
 * {@link RaySideFence}: an expectation that restates the code under test agrees with any consistent
 * error in it. <b>Nothing here reads {@code CruiseProgram}'s yaw law.</b> The handedness comes from
 * the engine's own {@code Direction.getClockWise()} table by way of {@link RaySideFence}, and the
 * metrics are plain geometry over a list of world positions the body actually occupied.
 *
 * <p><b>The sign under test, stated once.</b> A positive {@code turnBias} is declared to mean
 * "toward the body's right" (stage-4b design §3.1, D-077; ADR 0015 for what the body's right is).
 * Every pin that uses this class measures where the body went, never the sign of a variable, and
 * "where" is resolved onto the body's own initial forward/right basis rather than onto a world axis
 * so the claim survives a differently oriented test structure.
 *
 * <p><b>Nothing here is a claim about the connectome.</b> A turn bias is a declared program input;
 * on the readout lane it is supplied by a hand-chosen mapping (ADR 0019). A curving fly is
 * evidence that the mapping's number reached the body, and of nothing else.
 */
final class TurnBiasFence {
    /**
     * Ticks driven in the curvature pins. At {@link #CURVE_MODULATION} the body turns
     * {@code CRUISE_BIAS_TORQUE * 6.0} = 2.1 degrees per tick at full bias, so 60 ticks is about
     * 126 degrees of commanded heading change: an arc, not a wobble, and short enough that the
     * whole path stays inside the arena (see {@link #CURVE_MODULATION}).
     */
    static final int CURVE_TICKS = 60;

    /**
     * Forward-thrust attenuation for the curvature pins, and the reason it is not neutral.
     *
     * <p>Arithmetic on declared constants: {@code CRUISE_THRUST * 0.1} = 0.055 settles at
     * {@code 0.055 * 0.04 * 0.91 / (1 - 0.91)} = 0.0222 blocks/tick, so 60 ticks is at most 1.34
     * blocks of travel and the full-bias turn circle has radius
     * {@code 0.0222 / radians(2.1)} = 0.61 blocks. The whole path therefore stays within about 1.2
     * blocks of the spawn, and a body spawned at the arena's centre has 4.0 blocks of clearance on
     * every side: <b>the measured path is free flight, never a slide along a wall.</b> At neutral
     * speed the same 60 ticks would carry the body 11 blocks, which an 8-block arena cannot hold.
     *
     * <p>{@code turnRate} is left at 1.0 on purpose: the wander is the program exactly as written,
     * and the pin has to hold with it running. It does, by arithmetic rather than by luck —
     * {@code CRUISE_BIAS_TORQUE} (0.35) exceeds {@code CRUISE_WANDER_TORQUE} (0.25), so at full
     * bias in open air the commanded yaw cannot change sign whatever the wander does, and the pin
     * does not depend on the entry seed.
     */
    static final Modulation CURVE_MODULATION = new Modulation(0.1, 1.0, 1.0, 0.0);

    /**
     * Minimum net sideways displacement, in blocks, a full bias must produce along the body's own
     * initial right axis over {@link #CURVE_TICKS}.
     *
     * <p>Observed in-world at {@link CruiseFlightFence#PINNED_ENTRY_TICK}: +0.8695 blocks at bias
     * +1, +0.1025 at bias 0, and -0.7237 at bias -1. The bound is 0.35 — under half the observed
     * magnitude, and more than three times the unbiased run's own drift, so it separates "the bias
     * steered the body" from "the wander happened to drift".
     */
    static final double CURVE_LATERAL_BLOCKS = 0.35;

    /**
     * Minimum signed turning of the world-space path, in degrees, a full bias must produce over
     * {@link #CURVE_TICKS}.
     *
     * <p>Observed in-world at {@link CruiseFlightFence#PINNED_ENTRY_TICK}: +97.7 degrees at bias
     * +1, -3.7 at bias 0, and -106.0 at bias -1. The bound is 45 degrees, a comfortable factor of
     * two below the observed arcs and more than an order of magnitude above the unbiased run's
     * own drift.
     */
    static final double CURVE_PATH_TURN_DEGREES = 45.0;

    /** Ticks driven in the end-to-end chain pin. */
    static final int CHAIN_TICKS = 60;

    /**
     * Drive settings for the end-to-end chain pin.
     *
     * <p>{@code speed} 0.05 keeps the body within about 0.5 blocks of the arena centre over
     * {@link #CHAIN_TICKS}, which is what keeps the <em>real</em> sensor rays out of the avoidance
     * ramp: the shortest forward clearance the run may reach is
     * {@link #CHAIN_MIN_FORWARD_CLEARANCE_BLOCKS}, and the pin asserts that in-world rather than
     * assuming it.
     *
     * <p>{@code turnRate} 0 removes the wander entirely, so the bias and the residual avoidance are
     * the only yaw sources and the measurement does not depend on the arena's own game clock (the
     * chain pin drives the real validator, so its entry tick is whatever the server is at and
     * cannot be pinned). <b>It is also the pin that the bias is not attenuated by
     * {@code turnRate}</b>: the body must still turn, at a turn rate of zero.
     */
    static final Modulation CHAIN_MODULATION = new Modulation(0.05, 0.0, 1.0, 0.0);

    /**
     * Shortest forward clearance, in blocks, the chain pin's run may reach.
     *
     * <p>This is a <b>premise assertion, not a result</b>. Avoidance begins at
     * {@code CRUISE_SLOW_DISTANCE_BLOCKS} = 4.0, so a run that stays above 3.5 has an avoidance
     * fraction of at most {@code (4.0 - 3.5) / 2.5} = 0.2 and an avoidance yaw of at most
     * {@code 0.9 * 0.2} = 0.18, against a bias yaw of at least {@code 0.35 * (1 - 0.2)} = 0.28 —
     * so the commanded yaw keeps the bias's sign on every tick of the run, and the pin says so from
     * a measured floor rather than from a hope. Observed in-world: the forward ray falls to 3.778
     * blocks and the nearest side ray to 3.703, on zero contact ticks.
     */
    static final double CHAIN_MIN_FORWARD_CLEARANCE_BLOCKS = 3.5;

    /**
     * Minimum net sideways displacement, in blocks, for the chain pin. Observed in-world: +0.4028
     * blocks at {@code turn_bias = +1} and -0.4028 at -1; the bound is 0.15.
     */
    static final double CHAIN_LATERAL_BLOCKS = 0.15;

    /**
     * Minimum signed path turning, in degrees, for the chain pin. Observed in-world: +105.5 degrees
     * at {@code turn_bias = +1} and -105.5 at -1; the bound is 45. The two runs are exact mirrors
     * because {@code turnRate = 0} leaves no seeded term in the yaw at all.
     */
    static final double CHAIN_PATH_TURN_DEGREES = 45.0;

    /** Ticks driven in the bias-against-avoidance pin. */
    static final int WALL_TICKS = 100;

    /**
     * Closest the body may come to the wall a full bias is pointed straight into, in blocks,
     * measured as the structure-relative coordinate of the wall's own plane.
     *
     * <p>Observed in-world: the body closes to 0.296 blocks and turns away without a single contact
     * tick over the {@link #WALL_TICKS}-tick drive.
     *
     * <p><b>0.190 is a floor, not a result, and the first version of this bound sat under it.</b>
     * {@code Entity.move} resolves against the body's own collision box, so the centre can never
     * get closer to a wall plane than the body's half-width, 0.190 blocks — a body pinned flat
     * against the wall reads exactly that. A bound below 0.190 is therefore unfalsifiable: every
     * run passes it, including one that spends the whole drive scraping. This value was 0.15 and
     * was exactly that mistake; the review caught it.
     *
     * <p>0.25 sits between the observed standoff (0.296) and that floor (0.190), so it fails the
     * executed negative control, which puts the body against the wall. The contact count beside it
     * is an independent discriminator — 0 ticks as written against 17 of 100 under the control —
     * and this bound adds the case the contact flag cannot see: a body that comes to rest flush
     * with a wall without {@code Entity.move} ever reporting a horizontal collision.
     */
    static final double WALL_STANDOFF_BLOCKS = 0.25;

    /**
     * Extra wall-contact ticks a full bias may cost against a matched unbiased body, over
     * {@link #WALL_TICKS}.
     *
     * <p><b>Why the pin is relative and not only absolute.</b> The arena is a closed 8-block box, so
     * a body that leaves one wall meets another; a glancing contact somewhere else in the room is
     * CRUISE's own documented regime (see {@code CruiseFlightFence.WALL_CONTACT_TICK_BUDGET}) and
     * has nothing to do with the bias. An absolute budget alone would let such a contact fail this
     * pin — or, over a longer window, would have to be loosened until it stopped meaning anything.
     * Driving an unbiased body from the same spawn on the same entry tick and comparing gives the
     * quantity the pin is actually about: what the bias cost.
     *
     * <p>Observed in-world over {@link #WALL_TICKS} ticks: <b>0 contact ticks under a full bias
     * into the wall, against 1 for the matched unbiased body</b> — the bias cost nothing here, and
     * on this run it cost less than nothing. An independent review probe extended the same scenario
     * to 300 ticks and reported 5 contact ticks at bias +1, 6 unbiased and 3 at bias -1, all after
     * tick 100 — <em>that measurement is the reviewer's, recorded here as reported rather than as
     * this lane's own</em>. The budget is 3, which every one of those numbers sits inside in both
     * directions, and which is deliberately not 0: the two bodies fly different routes around a
     * closed room, so demanding exact parity would be pinning the room.
     */
    static final int WALL_CONTACT_BIAS_BUDGET = 3;

    private TurnBiasFence() {}

    /**
     * The body's own right axis as a world unit vector, taken from the engine's
     * {@code Direction.getClockWise()} table rather than restated here.
     *
     * @throws IllegalArgumentException unless {@code yawDegrees} is a cardinal heading, because
     *     {@code Direction.fromYRot} silently snaps anything else and a snapped axis would be a
     *     quietly wrong expectation
     */
    static Vec3 bodyRightUnit(float yawDegrees) {
        return RaySideFence.bodyRight(requireCardinal(yawDegrees)).getUnitVec3();
    }

    /** The body's own forward axis as a world unit vector, from the same engine table. */
    static Vec3 bodyForwardUnit(float yawDegrees) {
        return RaySideFence.facing(requireCardinal(yawDegrees)).getUnitVec3();
    }

    /**
     * Net displacement of the flown path along the body's initial right axis, in blocks. Positive
     * means the body ended up to the right of where it was pointed when it started.
     */
    static double lateralOffsetTowardBodyRight(List<Flight> flight, float startYaw) {
        Vec3 right = bodyRightUnit(startYaw);
        Flight first = flight.getFirst();
        Flight last = flight.getLast();
        return (last.x() - first.x()) * right.x + (last.z() - first.z()) * right.z;
    }

    /**
     * Total signed turning of the flown path, in degrees, positive toward the body's right.
     *
     * <p>Computed from the direction of successive world-space displacements, resolved onto the
     * body's initial forward/right basis and unwrapped across the seam, so it is a property of
     * where the body went and not of any heading field the actuator wrote.
     */
    static double turnedTowardBodyRight(List<Flight> flight, float startYaw) {
        Vec3 right = bodyRightUnit(startYaw);
        Vec3 forward = bodyForwardUnit(startYaw);
        double previous = Double.NaN;
        double turned = 0.0;
        for (int index = 1; index < flight.size(); index++) {
            double dx = flight.get(index).x() - flight.get(index - 1).x();
            double dz = flight.get(index).z() - flight.get(index - 1).z();
            if (Math.hypot(dx, dz) < 1.0e-9) {
                continue;
            }
            double heading = Math.toDegrees(
                    Math.atan2(dx * right.x + dz * right.z, dx * forward.x + dz * forward.z));
            if (!Double.isNaN(previous)) {
                turned += Mth.degreesDifference((float) previous, (float) heading);
            }
            previous = heading;
        }
        return turned;
    }

    /** Furthest the flown path ever got from where it started, in blocks. */
    static double furthestFromStart(List<Flight> flight) {
        Flight first = flight.getFirst();
        return flight.stream()
                .mapToDouble(sample -> Math.hypot(sample.x() - first.x(), sample.z() - first.z()))
                .max()
                .orElseThrow();
    }

    /** Smallest structure-relative x the flown path reached: its closest approach to the x=0 wall. */
    static double closestApproachToWestWall(List<Flight> flight) {
        return flight.stream().mapToDouble(Flight::x).min().orElseThrow();
    }

    /**
     * The commanded yaw, in degrees per tick, at full bias in open air — arithmetic on two declared
     * constants, used only in failure messages so a reader can see what was expected of the run.
     */
    static double fullBiasDegreesPerTick() {
        return CruiseProgram.CRUISE_BIAS_TORQUE
                * PrimitiveActuator.MAX_YAW_DELTA_DEGREES_PER_TICK;
    }

    private static float requireCardinal(float yawDegrees) {
        float wrapped = Mth.wrapDegrees(yawDegrees);
        if (Math.abs(Math.IEEEremainder(wrapped, 90.0)) > 1.0e-4) {
            throw new IllegalArgumentException(
                    "the body frame anchor needs a cardinal start yaw, got " + yawDegrees);
        }
        return wrapped;
    }
}
