package dev.neurocraft.fly.body;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.neurocraft.fly.body.programs.ApproachProgram;
import dev.neurocraft.fly.body.programs.BankProgram;
import dev.neurocraft.fly.body.programs.CruiseProgram;
import dev.neurocraft.fly.body.programs.FleeProgram;
import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Pure limits and state-machine tests; cardinal movement and real player sampling are GameTests. */
final class TargetFlightProgramTest {
    @Test
    void allThreeAreActuallyRegisteredAndBankDefaultsRightWithSignedOverride() {
        assertTrue(BodyProgramRuntime.registeredPrograms().containsAll(
                java.util.Set.of(BodyProgramId.BANK, BodyProgramId.FLEE, BodyProgramId.APPROACH)));
        var program = new BankProgram();
        var neutral = step(program, 0, 0, context(0, 6, 6, 6, 6, 0), Modulation.NEUTRAL, 0);
        assertEquals(0.45, neutral.motors().yawTorque());
        assertEquals(0.12, neutral.motors().lateralThrust());
        var left = step(program, 0, 0, context(0, 6, 6, 6, 6, 0), Modulation.NEUTRAL, -0.5);
        assertEquals(-0.225, left.motors().yawTorque());
        assertEquals(-0.06, left.motors().lateralThrust());
        assertTrue(step(program, 39, 0, context(0, 6, 6, 6, 6, 0), Modulation.NEUTRAL, 0).exitTo().isEmpty());
        assertEquals(Optional.of(BodyProgramId.HOVER),
                step(program, 40, 0, context(0, 6, 6, 6, 6, 0), Modulation.NEUTRAL, 0).exitTo());
    }

    @Test
    void targetAbsenceOrInvalidGeometryRefusesAndStopsOnTheSameTick() {
        for (BodyProgram program : new BodyProgram[] {new FleeProgram(), new ApproachProgram()}) {
            for (BodyProgramContext context : new BodyProgramContext[] {
                BodyProgramContext.sensorsOnly(sample(6, 6, 6, 0)),
                context(0, 12.001, 6, 6, 6, 0), context(181, 6, 6, 6, 6, 0),
                context(0, -1, 6, 6, 6, 0)
            }) {
                assertFalse(program.canEnter(context));
                var result = step(program, 8, 1, context, Modulation.NEUTRAL, 0);
                assertEquals(Optional.of(BodyProgramId.HOVER), result.exitTo());
                assertEquals(0.0, result.motors().forwardThrust());
                assertEquals(0.0, result.motors().yawTorque());
            }
        }
    }

    @Test
    void behindHasOneRightTurnAndFleeDoesNotBurstTowardAPlayer() {
        var approach = new ApproachProgram();
        for (double bearing : new double[] {-180, 180}) {
            var result = step(approach, 0, 0, context(bearing, 6, 6, 6, 6, 0), Modulation.NEUTRAL, 0);
            assertEquals(0.6, result.motors().yawTorque());
            assertEquals(0.0, result.motors().forwardThrust());
        }
        var flee = new FleeProgram();
        var turning = step(flee, 0, 0, context(0, 6, 6, 6, 6, 0), Modulation.NEUTRAL, 0);
        assertEquals(0.9, turning.motors().yawTorque());
        assertEquals(0.0, turning.motors().forwardThrust());
        assertEquals(0, turning.next().phaseIndex());
        var burst = step(flee, 0, 0, context(180, 6, 6, 6, 6, 0), Modulation.NEUTRAL, 0);
        assertEquals(0.65, burst.motors().forwardThrust());
        assertEquals(1, burst.next().phaseIndex());
        var targetMoved = step(flee, 5, 5, context(0, 6, 6, 6, 6, 0), Modulation.NEUTRAL, 0);
        assertEquals(0.0, targetMoved.motors().forwardThrust());
        assertEquals(6, targetMoved.next().phaseIndex());
    }

    @Test
    void urgencyOnlyCompressesBurstAndEveryTurnBurstOrBlockedBurstExpires() {
        var flee = new FleeProgram();
        for (double urgency : new double[] {0, 0.3, 1}) {
            var modulation = new Modulation(1, 1, 1, urgency);
            int duration = FleeProgram.burstTicks(urgency);
            assertTrue(duration >= 20 && duration <= 40);
            assertTrue(step(flee, duration - 1, duration - 1,
                    context(180, 6, 6, 6, 6, 0), modulation, 0).exitTo().isEmpty());
            assertEquals(Optional.of(BodyProgramId.HOVER), step(flee, duration, duration,
                    context(180, 6, 6, 6, 6, 0), modulation, 0).exitTo());
            assertEquals(Optional.of(BodyProgramId.HOVER), step(flee, 40, 0,
                    context(0, 6, 6, 6, 6, 0), modulation, 0).exitTo());
            assertEquals(Optional.of(BodyProgramId.HOVER), step(flee, 40 + duration, 0,
                    context(180, 6, 6, 6, 6, 0), modulation, 0).exitTo());
        }
    }

    @Test
    void approachSlowsThenStopsAtHorizontalStandOffAndDoesNotClimbToTheTarget() {
        var approach = new ApproachProgram();
        assertEquals(0.35, step(approach, 0, 0, context(0, 4, 6, 6, 6, 0), Modulation.NEUTRAL, 0)
                .motors().forwardThrust());
        assertEquals(0.175, step(approach, 0, 0, context(0, 3.25, 6, 6, 6, 0), Modulation.NEUTRAL, 0)
                .motors().forwardThrust());
        assertEquals(Optional.of(BodyProgramId.HOVER), step(approach, 0, 0,
                context(0, 2.5, 6, 6, 6, 0), Modulation.NEUTRAL, 0).exitTo());
        var high = new BodyProgramContext(sample(6, 6, 6, 0), Optional.empty(),
                Optional.of(new BodyProgramContext.BodyTargetVector(0, 60, 4)), Optional.empty());
        var stopped = step(approach, 0, 0, high, Modulation.NEUTRAL, 0);
        assertEquals(Optional.of(BodyProgramId.HOVER), stopped.exitTo());
        // The floor ray is 3.0, above the new band: the unchanged target logic uses the
        // declared floor descent, independent of the target 60 degrees overhead.
        assertEquals(-0.225, stopped.motors().verticalThrust());
    }

    @Test
    void declaredUnusedInputsAreBitIdenticalAndUsedInputsOnlyAttenuate() {
        var clear = context(150, 6, 6, 6, 6, 0);
        var bank = new BankProgram();
        assertEquals(step(bank, 7, 0, clear, new Modulation(0, 0.5, 0, 0), -1),
                step(bank, 7, 0, clear, new Modulation(1, 0.5, 1, 1), -1));
        var approach = new ApproachProgram();
        assertEquals(step(approach, 7, 0, clear, new Modulation(0.5, 0.5, 0, 0), -1),
                step(approach, 7, 0, clear, new Modulation(0.5, 0.5, 1, 1), 1));
        var flee = new FleeProgram();
        assertEquals(step(flee, 7, 4, clear, new Modulation(0.5, 0.5, 0, 0), -1),
                step(flee, 7, 4, clear, new Modulation(0.5, 0.5, 1, 0), 1));
        for (BodyProgram program : new BodyProgram[] {bank, flee, approach}) {
            var ctx = context(program.id() == BodyProgramId.FLEE ? 150 : 30, 6, 6, 6, 6, 0);
            var full = step(program, 0, 0, ctx, Modulation.NEUTRAL, 0).motors();
            var half = step(program, 0, 0, ctx, new Modulation(0.5, 0.5, 1, 0), 0).motors();
            assertEquals(full.yawTorque() * 0.5, half.yawTorque());
            assertEquals(full.forwardThrust() * (program.id() == BodyProgramId.BANK ? 1.0 : 0.5),
                    half.forwardThrust());
        }
    }

    @Test
    void obstaclesNeverWidenYawAndBrakingOrVerticalFeedbackKeepTheirOwnBounds() {
        for (BodyProgram program : new BodyProgram[] {new BankProgram(), new FleeProgram(), new ApproachProgram()}) {
            double maximum = program.id() == BodyProgramId.BANK ? 0.45 : program.id() == BodyProgramId.FLEE ? 0.9 : 0.6;
            for (double rate : new double[] {0, 0.3, 1}) {
                var ctx = context(180, 6, 1, 0.5, 6, -0.6);
                var result = step(program, 0, 0, ctx, new Modulation(1, rate, 1, 0), -1).motors();
                assertTrue(Math.abs(result.yawTorque()) <= maximum * rate);
                assertTrue(result.yawTorque() >= 0, "turn into the clear right side");
                assertEquals(0.0, result.forwardThrust());
                assertEquals(CruiseProgram.verticalThrust(ctx.snapshot()), result.verticalThrust());
                assertTrue(Math.abs(result.verticalThrust()) <= 0.45);
                assertEquals(0.0, result.pitchTorque());
            }
        }
    }

    @Test
    void repeatedEntriesDoNotAddAnyVerticalImpulseBeyondTheExistingFloorLaw() {
        for (double vy : new double[] {-0.6, 0, 0.6}) {
            for (BodyProgram program : new BodyProgram[] {new BankProgram(), new FleeProgram(), new ApproachProgram()}) {
                var ctx = context(180, 6, 6, 6, 6, vy);
                for (int tick : new int[] {0, 1, 39, 40, 79, 80}) {
                    assertEquals(CruiseProgram.verticalThrust(ctx.snapshot()),
                            step(program, tick, 0, ctx, Modulation.NEUTRAL, 0).motors().verticalThrust());
                }
            }
        }
    }

    private static BodyProgramStep step(BodyProgram program, int tick, int phase,
            BodyProgramContext context, Modulation modulation, double bias) {
        return program.step(new BodyProgramState(program.id(), 10, tick, phase),
                context, modulation, new ProgramParameters(bias));
    }

    private static BodyProgramContext context(double bearing, double distance,
            double front, double left, double right, double vy) {
        return new BodyProgramContext(sample(front, left, right, vy), Optional.empty(),
                Optional.of(new BodyProgramContext.BodyTargetVector(bearing, 0, distance)), Optional.empty());
    }

    private static PhysicalSensorSnapshot sample(double front, double left, double right, double vy) {
        return new PhysicalSensorSnapshot(10, "target-test", 0, 3, 0, 0, vy, 0, 0, 0,
                false, false, false, false, false, 0, 0, 6, front, 6, left, right, 6, 3);
    }
}
