package dev.neurocraft.fly.body;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.neurocraft.fly.body.programs.FeedProgram;
import dev.neurocraft.fly.body.programs.GroomProgram;
import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Test;

/** Pure preconditions, clocks and brakes. Real contact and item sampling are server GameTests. */
final class GroomFeedProgramTest {
    @Test
    void everyDeclaredProgramExistsAndGenuinelyUnknownNamesStillRefuse() {
        assertEquals(Set.of(BodyProgramId.values()), BodyProgramRuntime.registeredPrograms());
        assertThrows(IllegalArgumentException.class, () -> BodyProgramId.valueOf("EAT_EVERYTHING"));
        assertEquals(Set.of("minecraft:sugar", "minecraft:apple", "minecraft:sweet_berries",
                "minecraft:glow_berries"), BodyProgramContextSampler.FOOD_ITEM_IDS);
    }

    @Test
    void feedRequiresGroundSupportAndCurrentWhitelistedFoodInsideReachAndCone() {
        var feed = new FeedProgram();
        var ground = sample(true, false, false, 0, 0, 0, 0);
        for (String item : BodyProgramContextSampler.FOOD_ITEM_IDS) {
            assertTrue(feed.canEnter(context(ground, 0.75, 60, item)));
        }
        for (var invalid : List.of(
                context(ground, 0.75001, 0, "minecraft:sugar"),
                context(ground, -0.001, 0, "minecraft:sugar"),
                context(ground, 0.5, 60.001, "minecraft:sugar"),
                context(ground, 0.5, -60.001, "minecraft:sugar"),
                context(ground, 0.5, 0, "minecraft:stone"),
                // Airborne means a real air gap, not merely a cleared movement-contact flag.
                new BodyProgramContext(sample(false, false, false, 0, 0, 0, 0),
                        Optional.of(new BodyProgramContext.SurfaceBelow(0.001, true)), Optional.empty(),
                        Optional.of(new BodyProgramContext.ReachableFood(0, 0.5, "minecraft:sugar"))),
                context(sample(true, true, false, 0, 0, 0, 0), 0.5, 0, "minecraft:sugar"),
                context(sample(true, false, true, 0, 0, 0, 0), 0.5, 0, "minecraft:sugar"),
                new BodyProgramContext(ground, Optional.of(new BodyProgramContext.SurfaceBelow(0, true)),
                        Optional.empty(), Optional.empty()),
                new BodyProgramContext(ground, Optional.of(new BodyProgramContext.SurfaceBelow(0, false)),
                        Optional.empty(), Optional.of(new BodyProgramContext.ReachableFood(0, 0.5, "minecraft:sugar"))))) {
            assertFalse(feed.canEnter(invalid));
            var exit = step(feed, invalid, 25, Modulation.NEUTRAL, ProgramParameters.NEUTRAL);
            assertEquals(Optional.of(BodyProgramId.HOVER), exit.exitTo());
            assertEquals(0.0F, exit.presentationPhase());
        }
    }

    @Test
    void groomingCanParkInAirOrOnGroundButNeverInWaterOrLava() {
        var groom = new GroomProgram();
        for (boolean ground : new boolean[] {false, true}) {
            assertTrue(groom.canEnter(context(sample(ground, false, false, 0, 0, 0, 0), 0.5, 0, "minecraft:sugar")));
        }
        for (var sample : List.of(sample(false, true, false, 0, 0, 0, 0),
                sample(false, false, true, 0, 0, 0, 0))) {
            assertFalse(groom.canEnter(BodyProgramContext.sensorsOnly(sample)));
            assertEquals(Optional.of(BodyProgramId.HOVER), step(groom,
                    BodyProgramContext.sensorsOnly(sample), 10, Modulation.NEUTRAL, ProgramParameters.NEUTRAL).exitTo());
        }
    }

    @Test
    void fourPresentationQuartersKeepTheirMeaningAtEveryUrgencyAndExpireAtTheirBound() {
        var context = context(sample(true, false, false, 0, 0, 0, 0), 0.5, 0, "minecraft:sugar");
        for (double urgency : new double[] {0, 0.5, 1}) {
            int groomStroke = 20 - (int) Math.floor(10 * urgency);
            int feedHold = 30 - (int) Math.floor(20 * urgency);
            verifyPhases(new GroomProgram(), context, urgency, new int[] {10, groomStroke, groomStroke, 10});
            verifyPhases(new FeedProgram(), context, urgency, new int[] {10, 10, feedHold, 10});
        }
    }

    @Test
    void unusedInputsNeverChangeMotorsPhaseOrExpiryAndAmplitudeCannotCreateAForce() {
        var context = context(sample(true, false, false, 0.12, 0, -0.05, 63), 0.5, 0, "minecraft:sugar");
        for (BodyProgram program : new BodyProgram[] {new GroomProgram(), new FeedProgram()}) {
            for (int tick = 0; tick <= 60; tick++) {
                assertEquals(step(program, context, tick, new Modulation(0, 0, 0, 0.5), new ProgramParameters(-1)),
                        step(program, context, tick, new Modulation(1, 1, 1, 0.5), new ProgramParameters(1)));
                var quiet = step(program, context, tick, new Modulation(1, 1, 1, 0), ProgramParameters.NEUTRAL);
                var urgent = step(program, context, tick, new Modulation(1, 1, 1, 1), ProgramParameters.NEUTRAL);
                assertEquals(quiet.motors(), urgent.motors());
            }
        }
    }

    @Test
    void parkingOpposesAllDriftInsideDeclaredBoundsAtEveryCardinalHeading() {
        for (boolean grounded : new boolean[] {false, true}) {
            for (double yaw : new double[] {0, 90, 180, -90}) {
                for (double velocity : new double[] {-0.6, -0.01, 0, 0.01, 0.6}) {
                    var ctx = context(sample(grounded, false, false, velocity, velocity, -velocity, yaw),
                            0.5, 0, "minecraft:sugar");
                    for (BodyProgram program : new BodyProgram[] {new GroomProgram(), new FeedProgram()}) {
                        var result = step(program, ctx, 0, Modulation.NEUTRAL, ProgramParameters.NEUTRAL).motors();
                        assertTrue(Math.abs(result.forwardThrust()) <= 0.25);
                        assertTrue(Math.abs(result.lateralThrust()) <= 0.25);
                        assertTrue(Math.abs(result.verticalThrust()) <= 0.35);
                        // Literal Minecraft cardinal table, independent of the shared body-frame helper.
                        double forwardVelocity = switch ((int) yaw) {
                            case 0, 90 -> -velocity;
                            default -> velocity;
                        };
                        double rightVelocity = switch ((int) yaw) {
                            case 0, -90 -> -velocity;
                            default -> velocity;
                        };
                        assertTrue(result.forwardThrust() * forwardVelocity <= 0);
                        assertTrue(result.lateralThrust() * rightVelocity <= 0);
                        assertEquals(0.0, result.yawTorque());
                        assertEquals(0.0, result.pitchTorque());
                        if (grounded && velocity <= 0) {
                            assertEquals(-0.02, result.verticalThrust());
                        } else {
                            assertTrue(result.verticalThrust() * velocity <= 0);
                        }
                    }
                }
            }
        }
    }

    private static void verifyPhases(BodyProgram program, BodyProgramContext context, double urgency, int[] lengths) {
        int tick = 0;
        for (int phase = 0; phase < 4; phase++) {
            for (int local = 0; local < lengths[phase]; local++, tick++) {
                var result = step(program, context, tick, new Modulation(1, 1, 1, urgency), ProgramParameters.NEUTRAL);
                float value = result.presentationPhase();
                assertTrue(value >= phase / 4.0 && value < (phase + 1) / 4.0);
                if (local == 0) { assertEquals(phase / 4.0F, value); }
                assertTrue(result.exitTo().isEmpty());
            }
        }
        assertTrue(tick >= 40 && tick <= 60);
        assertEquals(Optional.of(BodyProgramId.HOVER), step(program, context, tick,
                new Modulation(1, 1, 1, urgency), ProgramParameters.NEUTRAL).exitTo());
    }

    private static BodyProgramStep step(BodyProgram program, BodyProgramContext context, int tick,
            Modulation modulation, ProgramParameters parameters) {
        return program.step(new BodyProgramState(program.id(), 1, tick, 0), context, modulation, parameters);
    }

    private static BodyProgramContext context(PhysicalSensorSnapshot sample, double distance, double bearing, String item) {
        return new BodyProgramContext(sample, Optional.of(new BodyProgramContext.SurfaceBelow(0, true)),
                Optional.empty(), Optional.of(new BodyProgramContext.ReachableFood(bearing, distance, item)));
    }

    private static PhysicalSensorSnapshot sample(boolean ground, boolean water, boolean lava,
            double vx, double vy, double vz, double yaw) {
        return new PhysicalSensorSnapshot(100, "groom-feed-test", 1, 2, 1, vx, vy, vz, yaw, 0,
                false, ground, ground, water, lava, 0, 0, 6, 6, 6, 6, 6, 6, 3);
    }
}
