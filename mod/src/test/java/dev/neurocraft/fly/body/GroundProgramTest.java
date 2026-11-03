package dev.neurocraft.fly.body;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.neurocraft.fly.body.programs.LandProgram;
import dev.neurocraft.fly.body.programs.WalkProgram;
import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import dev.neurocraft.fly.control.PrimitiveAction;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Pure law and precondition pins; physical touchdown and support sampling are server GameTests. */
final class GroundProgramTest {
    private final LandProgram land = new LandProgram();
    private final WalkProgram walk = new WalkProgram();

    @Test
    void landingRequiresDeclaredSturdySurfaceWithinSixBlocks() {
        PhysicalSensorSnapshot sample = sample(false, 0, 0, 0, 0, 6, false, false);
        assertFalse(land.canEnter(BodyProgramContext.sensorsOnly(sample)));
        for (double distance : new double[] {-0.001, 6.001}) {
            assertFalse(land.canEnter(context(sample, distance, true)));
        }
        assertFalse(land.canEnter(context(sample, 1, false)));
        assertFalse(land.canEnter(context(sample(false, 0, 0, 0, 0, 6, true, false), 1, true)));
        for (double distance : new double[] {0, 0.001, 1, 6}) {
            assertTrue(land.canEnter(context(sample, distance, true)));
        }
    }

    @Test
    void descentSlowsNearTheFloorAndBrakesExcessFallSpeed() {
        var distant = context(sample(false, 0, 0, 0, 0, 6, false, false), 4, true);
        var near = context(distant.snapshot(), 0.01, true);
        double down = step(land, 0, distant, Modulation.NEUTRAL).motors().verticalThrust();
        double slow = step(land, 0, near, Modulation.NEUTRAL).motors().verticalThrust();
        assertTrue(down < slow && slow < 0, "near-floor descent must remain downward but slow");
        var falling = context(sample(false, 0, -0.6, 0, 0, 6, false, false), 4, true);
        assertTrue(step(land, 0, falling, Modulation.NEUTRAL).motors().verticalThrust() > 0);
        var stopped = new Modulation(0, 1, 1, 0);
        assertEquals(0.0, step(land, 0, distant, stopped).motors().verticalThrust());
        assertTrue(step(land, 0, falling, stopped).motors().verticalThrust() > 0);
    }

    @Test
    void landingOnlyReadsSpeedAndBrakesDriftInTheBodyFrame() {
        for (double yaw : new double[] {0, 90, -90, 173}) {
            var ctx = context(sample(false, 0.2, -0.03, -0.1, yaw, 6, false, false), 0.1, true);
            var baseline = step(land, 7, ctx, new Modulation(0.5, 0, 0, 0));
            var changed = land.step(new BodyProgramState(BodyProgramId.LAND, 10, 7, 0), ctx,
                    new Modulation(0.5, 1, 1, 1), new ProgramParameters(-1));
            assertEquals(baseline, changed);
            double angle = Math.toRadians(yaw);
            double forwardVelocity = -Math.sin(angle) * 0.2 + Math.cos(angle) * -0.1;
            double rightVelocity = -Math.cos(angle) * 0.2 - Math.sin(angle) * -0.1;
            assertTrue(baseline.motors().forwardThrust() * forwardVelocity <= 0);
            assertTrue(baseline.motors().lateralThrust() * rightVelocity <= 0);
        }
    }

    @Test
    void timeoutAndTouchdownAreDifferentStatesAndNeitherSelectsWalk() {
        var air = context(sample(false, 0, 0, 0, 0, 6, false, false), 2, true);
        assertTrue(step(land, 198, air, Modulation.NEUTRAL).exitTo().isEmpty());
        var timeout = step(land, 199, air, Modulation.NEUTRAL);
        assertEquals(Optional.of(BodyProgramId.HOVER), timeout.exitTo());
        assertEquals(0.0, timeout.motors().verticalThrust());

        var ground = context(sample(true, 0, 0, 0, 0, 6, false, false), 0, true);
        var touchdown = step(land, 199, ground, Modulation.NEUTRAL);
        assertEquals(LandProgram.GROUNDED_HOLD, touchdown.next().phaseIndex());
        assertTrue(touchdown.exitTo().isEmpty());
        BodyProgramState state = touchdown.next();
        for (int tick = 0; tick < 500; tick++) {
            var held = land.step(state, ground, Modulation.NEUTRAL, ProgramParameters.NEUTRAL);
            assertTrue(held.exitTo().isEmpty());
            assertEquals(BodyProgramId.LAND, held.next().id());
            assertEquals(-0.02, held.motors().verticalThrust());
            state = held.next();
        }
        assertEquals(Optional.of(BodyProgramId.HOVER),
                land.step(state, air, Modulation.NEUTRAL, ProgramParameters.NEUTRAL).exitTo());
    }

    @Test
    void walkingNeedsCollisionOrPreciselyRestingSupport() {
        var ground = sample(true, 0, 0, 0, 0, 6, false, false);
        assertTrue(walk.canEnter(context(ground, 0, true)));
        assertFalse(walk.canEnter(context(ground, 0.021, true)));
        assertFalse(walk.canEnter(context(ground, 0, false)));
        assertFalse(walk.canEnter(BodyProgramContext.sensorsOnly(ground)));
        assertTrue(walk.canEnter(context(sample(false, 0, 0, 0, 0, 6, false, false), 0, true)));
        assertFalse(walk.canEnter(context(sample(false, 0, 0, 0, 0, 6, false, false), 0.001, true)));
    }

    @Test
    void restingSupportUsesTightClearanceAndNonpositiveBoundedVelocityWithoutFlags() {
        for (double distance : new double[] {0, 0.0001}) {
            for (double vertical : new double[] {0, -0.0008}) {
                assertTrue(LandProgram.hasGroundContact(context(
                        sample(false, 0, vertical, 0, 0, 6, false, false), distance, true)));
            }
        }
        for (double distance : new double[] {-0.0001, Math.nextUp(0.0001), 0.001}) {
            assertFalse(LandProgram.hasGroundContact(context(
                    sample(false, 0, 0, 0, 0, 6, false, false), distance, true)));
        }
        for (double vertical : new double[] {Math.nextDown(-0.0008), -0.1, 0.000001, 0.1}) {
            assertFalse(LandProgram.hasGroundContact(context(
                    sample(false, 0, vertical, 0, 0, 6, false, false), 0, true)));
        }
        assertFalse(LandProgram.hasGroundContact(context(
                sample(false, 0, 0, 0, 0, 6, false, false), 0, false)));
        assertFalse(LandProgram.hasGroundContact(context(
                sample(false, 0, 0, 0, 0, 6, true, false), 0, true)));
        // An inherited flag cannot keep an upward-commanded body in a ground program.
        assertFalse(LandProgram.hasGroundContact(context(
                sample(true, 0, 0.000001, 0, 0, 6, false, false), 0, true)));
        assertTrue(LandProgram.hasGroundContact(context(
                sample(true, 0, -0.01, 0, 0, 6, false, false), 0, true)));
    }

    @Test
    void gaitAndTurningStayInsideDeclaredBoundsAndUrgencyDoesNotChangeMotors() {
        var ground = context(sample(true, 0, 0, 0, 0, 6, false, false), 0, true);
        boolean gaitVaried = false;
        double first = step(walk, 0, ground, Modulation.NEUTRAL).motors().forwardThrust();
        for (int tick = 0; tick < 48; tick++) {
            var state = new BodyProgramState(BodyProgramId.WALK, 10, tick, 0);
            var neutral = walk.step(state, ground, Modulation.NEUTRAL, new ProgramParameters(-1));
            var urgent = walk.step(state, ground, new Modulation(1, 1, 1, 1), new ProgramParameters(-1));
            assertEquals(neutral, urgent);
            assertTrue(neutral.motors().forwardThrust() >= 0.09
                    && neutral.motors().forwardThrust() <= 0.12);
            assertEquals(-0.08, neutral.motors().yawTorque());
            assertEquals(-0.02, neutral.motors().verticalThrust());
            var attenuated = walk.step(state, ground, new Modulation(0.5, 0.5, 0, 0),
                    new ProgramParameters(-1));
            assertEquals(0.06, attenuated.motors().forwardThrust());
            assertEquals(-0.04, attenuated.motors().yawTorque());
            assertEquals(0.0, attenuated.motors().pitchTorque());
            gaitVaried |= first != neutral.motors().forwardThrust();
        }
        assertTrue(gaitVaried, "the declared gait must have a measurable phase");
    }

    @Test
    void stoppedOrObstructedWalkBrakesInsteadOfSelectingAnAvoidanceDirection() {
        for (var ctx : new BodyProgramContext[] {
            context(sample(true, 0, 0, 0.05, 0, 0.1, false, false), 0, true),
            context(sample(true, 0, 0, 0.05, 0, 6, false, true), 0, true)
        }) {
            var stopped = step(walk, 0, ctx, Modulation.NEUTRAL);
            assertTrue(stopped.motors().forwardThrust() < 0);
            assertEquals(0.0, stopped.motors().yawTorque());
            assertTrue(stopped.exitTo().isEmpty());
        }
        var moving = context(sample(true, 0, 0, 0.05, 0, 6, false, false), 0, true);
        assertTrue(step(walk, 0, moving, new Modulation(0, 1, 1, 0)).motors().forwardThrust() < 0);
    }

    @Test
    void lostSupportStopsWalkingOnThatTickAndNamesOnlyHover() {
        var unsupported = context(sample(true, 0, 0, 0.05, 0, 6, false, false), 0, false);
        var result = step(walk, 8, unsupported, Modulation.NEUTRAL);
        assertEquals(Optional.of(BodyProgramId.HOVER), result.exitTo());
        assertTrue(result.motors().forwardThrust() < 0);
        assertEquals(0.0, result.motors().yawTorque());
    }

    @Test
    void allLawOutputsRemainBoundedAtThePhysicalSpeedExtremes() {
        for (double velocity : new double[] {-0.6, -0.01, 0, 0.01, 0.6}) {
            for (double yaw : new double[] {-180, -90, 0, 90, 180}) {
                for (BodyProgram program : new BodyProgram[] {land, walk}) {
                    var ctx = context(sample(true, velocity, velocity, velocity, yaw, 6, false, false), 0, true);
                    PrimitiveAction motors = step(program, 4, ctx, Modulation.NEUTRAL).motors();
                    assertTrue(Math.abs(motors.forwardThrust()) <= 0.25);
                    assertTrue(Math.abs(motors.lateralThrust()) <= 0.25);
                    assertTrue(Math.abs(motors.verticalThrust()) <= 0.35);
                    assertTrue(Math.abs(motors.yawTorque()) <= 0.08);
                    assertEquals(0.0, motors.pitchTorque());
                }
            }
        }
    }

    private static BodyProgramStep step(BodyProgram program, int tick,
            BodyProgramContext context, Modulation modulation) {
        return program.step(new BodyProgramState(program.id(), 10, tick, 0),
                context, modulation, ProgramParameters.NEUTRAL);
    }

    private static BodyProgramContext context(PhysicalSensorSnapshot snapshot, double distance, boolean sturdy) {
        return new BodyProgramContext(snapshot, Optional.of(new BodyProgramContext.SurfaceBelow(distance, sturdy)),
                Optional.empty(), Optional.empty());
    }

    private static PhysicalSensorSnapshot sample(boolean ground, double vx, double vy, double vz,
            double yaw, double front, boolean water, boolean collision) {
        return new PhysicalSensorSnapshot(10, "ground-test", 1, 2, 1, vx, vy, vz, yaw, 0,
                collision, ground, ground, water, false, 0, 0, 6, front, 6, 6, 6, 6, 6);
    }
}
