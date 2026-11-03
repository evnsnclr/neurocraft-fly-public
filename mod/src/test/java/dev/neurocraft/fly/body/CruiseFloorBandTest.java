package dev.neurocraft.fly.body;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.neurocraft.fly.body.programs.CruiseProgram;
import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import org.junit.jupiter.api.Test;

/**
 * Pure-law checks for the separately declared floor and ceiling clearances. The frozen
 * 4552fc9 reference remains historical: the new floor ramp deliberately differs, while the
 * ceiling contribution and rate-to-thrust conversion must still match it exactly.
 */
final class CruiseFloorBandTest {
    private static final double MIN = CruiseProgram.CRUISE_MIN_FLOOR_CLEARANCE_BLOCKS;
    private static final double MAX = CruiseProgram.CRUISE_MAX_FLOOR_CLEARANCE_BLOCKS;
    private static final double CLIMB = CruiseProgram.CRUISE_CLIMB_RATE_BLOCKS_PER_TICK;
    private static final double DESCENT = CruiseProgram.CRUISE_DESCENT_RATE_BLOCKS_PER_TICK;
    private static final double THRUST = CruiseProgram.CRUISE_ALTITUDE_THRUST;
    // Frozen reference values at 4552fc9, independent of future production tuning.
    private static final double BASELINE_MIN = 2.5;
    private static final double BASELINE_CLIMB = 0.12;
    private static final double BASELINE_THRUST = 0.45;

    /** The 6.0-block ray clamp, restated: loading {@code FlySensors} here would load Minecraft. */
    private static final double RAY_CLAMP = 6.0;

    private static final double[] UP_RAYS = {0.0, 0.05, 1.0, 2.49, 2.5, 3.0, RAY_CLAMP};
    private static final double[] VERTICAL_RATES = {-0.6, -0.4, -0.12, -0.04, 0.0, 0.01, 0.12, 0.6};

    @Test
    void theBandIsDeclaredInOrderAndTheDescentIsHalfTheClimb() {
        assertEquals(1.0, MIN);
        assertEquals(1.5, MAX);
        assertEquals(2.5, CruiseProgram.CRUISE_MIN_VERTICAL_CLEARANCE_BLOCKS);
        assertEquals(BASELINE_CLIMB, CLIMB);
        assertEquals(BASELINE_THRUST, THRUST);
        assertTrue(0.0 < MIN && MIN < MAX && MAX < RAY_CLAMP, "the band must sit under the clamp");
        assertEquals(CLIMB / 2.0, DESCENT, 0.0, "the descent is declared as half the climb rate");
    }

    @Test
    void theVerticalRateIsExactlyZeroInsideTheBandEdgesInclusive() {
        for (int step = 0; step <= 1500; step++) {
            double down = MIN + (MAX - MIN) * step / 1500.0;
            double rate = CruiseProgram.targetVerticalRate(down, RAY_CLAMP);
            assertEquals(0L, Double.doubleToRawLongBits(rate), "not positive zero at down " + down);
            assertEquals(
                    0L,
                    Double.doubleToRawLongBits(CruiseProgram.verticalThrust(sample(down, RAY_CLAMP, 0.0))),
                    "a level body inside the band was commanded a vertical thrust at down " + down);
        }
        for (double edge : new double[] {MIN, MAX, Math.nextDown(MAX), Math.nextUp(MIN)}) {
            assertEquals(0.0, CruiseProgram.targetVerticalRate(edge, RAY_CLAMP), 0.0, "edge " + edge);
        }
    }

    @Test
    void theDescentIsCommandedJustAboveTheBandAndAtTheClamp() {
        for (double down : new double[] {Math.nextUp(MAX), 4.01, 4.5, 5.0, RAY_CLAMP, 64.0}) {
            assertEquals(-DESCENT, CruiseProgram.targetVerticalRate(down, RAY_CLAMP), 0.0, "at " + down);
            double thrust = CruiseProgram.verticalThrust(sample(down, RAY_CLAMP, 0.0));
            assertEquals(-THRUST * DESCENT / CLIMB, thrust, 1.0e-15, "level body at " + down);
            assertTrue(thrust < 0.0 && thrust >= -THRUST, "descent thrust outside its envelope");
            // The loop is satisfied once the body is falling at the commanded rate.
            assertEquals(
                    0.0, CruiseProgram.verticalThrust(sample(down, RAY_CLAMP, -DESCENT)), 1.0e-15);
        }
    }

    @Test
    void theClimbIsCommandedUnderTheMinimumAndNeverMeetsTheDescent() {
        double previous = Double.POSITIVE_INFINITY;
        for (int step = 0; step < 250; step++) {
            double down = MIN * step / 250.0;
            double rate = CruiseProgram.targetVerticalRate(down, RAY_CLAMP);
            assertTrue(rate > 0.0 && rate <= CLIMB, "no bounded climb at down " + down + ": " + rate);
            assertTrue(rate <= previous, "the climb is not monotone at down " + down);
            assertEquals(
                    Double.doubleToRawLongBits(CLIMB * ((MIN - down) / MIN)),
                    Double.doubleToRawLongBits(rate),
                    "the proportional floor climb changed at down " + down);
            previous = rate;
        }
        for (int step = 0; step <= 640; step++) {
            double down = RAY_CLAMP * step / 640.0;
            double rate = CruiseProgram.targetVerticalRate(down, RAY_CLAMP);
            assertTrue(rate <= 0.0 || down < MIN, "a climb outside the climb range at " + down);
            assertTrue(rate >= 0.0 || down > MAX, "a descent outside the descent range at " + down);
        }
    }

    @Test
    void ceilingAndRateLoopAreBitIdenticalToTheHistoricalLawWithItsFloorTermInactive() {
        long compared = 0;
        for (int step = 0; step <= 1500; step++) {
            double down = MIN + (MAX - MIN) * step / 1500.0;
            for (double up : UP_RAYS) {
                for (double rate : VERTICAL_RATES) {
                    assertEquals(
                            Double.doubleToRawLongBits(preBandVerticalThrust(BASELINE_MIN, up, rate)),
                            Double.doubleToRawLongBits(
                                    CruiseProgram.verticalThrust(sample(down, up, rate))),
                            "in-band output moved at down " + down + " up " + up + " vy " + rate);
                    compared++;
                }
            }
        }
        assertTrue(compared > 80_000, "the sweep collapsed: " + compared);
        // The reference is not the code under test: above the band the two disagree.
        assertNotEquals(
                Double.doubleToRawLongBits(preBandVerticalThrust(RAY_CLAMP, RAY_CLAMP, 0.0)),
                Double.doubleToRawLongBits(
                        CruiseProgram.verticalThrust(sample(RAY_CLAMP, RAY_CLAMP, 0.0))),
                "the pre-band reference agrees with the banded law in open air, so it is not a"
                        + " reference");
    }

    @Test
    void aboveTheBandTheUnchangedCeilingAndRateLoopAddTheDeclaredDescent() {
        for (int step = 1; step <= 200; step++) {
            double down = MAX + (RAY_CLAMP - MAX) * step / 200.0;
            for (double up : UP_RAYS) {
                assertEquals(
                        preBandTargetRate(BASELINE_MIN, up) - DESCENT,
                        CruiseProgram.targetVerticalRate(down, up),
                        0.0,
                        "the descent is not one subtraction at down " + down + " up " + up);
                for (double rate : VERTICAL_RATES) {
                    assertEquals(
                            preBandVerticalThrust(BASELINE_MIN, up, rate + DESCENT),
                            CruiseProgram.verticalThrust(sample(down, up, rate)),
                            1.0e-12,
                            "above the band at down " + down + " up " + up + " vy " + rate);
                }
            }
        }
    }

    @Test
    void theCeilingRuleAddsToTheDescentAndNeverOpposesIt() {
        for (double up : new double[] {0.0, 0.5, 1.0, 2.0, 2.49}) {
            for (double down : new double[] {Math.nextUp(MAX), 4.5, 5.0, RAY_CLAMP}) {
                double rate = CruiseProgram.targetVerticalRate(down, up);
                assertTrue(rate < -DESCENT, "the ceiling did not add to the descent at up " + up);
                assertEquals(
                        preBandTargetRate(BASELINE_MIN, up) - DESCENT,
                        rate,
                        0.0,
                        "the ceiling term is not the pre-band term less the descent at up " + up);
            }
        }
    }

    @Test
    void theLowerFloorBandDeliberatelyDiffersFromTheHistoricalFloorRamp() {
        assertTrue(preBandTargetRate(1.0, RAY_CLAMP) > 0.0);
        assertEquals(0.0, CruiseProgram.targetVerticalRate(1.0, RAY_CLAMP));
        assertEquals(0.0, preBandTargetRate(3.0, RAY_CLAMP));
        assertEquals(-0.06, CruiseProgram.targetVerticalRate(3.0, RAY_CLAMP));
    }

    // ---- the pre-band law, copied from main at 4552fc9 ------------------------------------

    /**
     * {@code CruiseProgram.verticalThrust} as it stood before the floor band ({@code main} at
     * {@code 4552fc9}), with the three private helpers it used, copied verbatim as the regression
     * reference.
     */
    private static double preBandVerticalThrust(double rayDown, double rayUp, double velocityY) {
        double targetRate = preBandTargetRate(rayDown, rayUp);
        double rateError = targetRate - velocityY;
        return clampSymmetric(
                BASELINE_THRUST * rateError / BASELINE_CLIMB, BASELINE_THRUST);
    }

    private static double preBandTargetRate(double rayDown, double rayUp) {
        double floorPush = encroachment(rayDown, BASELINE_MIN);
        double ceilingPush = encroachment(rayUp, BASELINE_MIN);
        return BASELINE_CLIMB * (floorPush - ceilingPush);
    }

    private static double encroachment(double clearanceBlocks, double declaredBlocks) {
        return clampUnitInterval((declaredBlocks - clearanceBlocks) / declaredBlocks);
    }

    private static double clampUnitInterval(double value) {
        return Math.max(0.0, Math.min(1.0, value));
    }

    private static double clampSymmetric(double value, double bound) {
        return Math.max(-bound, Math.min(bound, value));
    }

    private static PhysicalSensorSnapshot sample(double down, double up, double velocityY) {
        return new PhysicalSensorSnapshot(
                23L,
                "cruise-floor-band-test",
                0.0,
                64.0,
                0.0,
                0.0,
                velocityY,
                0.0,
                0.0,
                0.0,
                false,
                false,
                false,
                false,
                false,
                0,
                15,
                RAY_CLAMP,
                RAY_CLAMP,
                RAY_CLAMP,
                RAY_CLAMP,
                RAY_CLAMP,
                up,
                down);
    }
}
