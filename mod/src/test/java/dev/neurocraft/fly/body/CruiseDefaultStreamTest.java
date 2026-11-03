package dev.neurocraft.fly.body;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.neurocraft.fly.body.programs.CruiseProgram;
import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import org.junit.jupiter.api.Test;

/**
 * Frozen default stream from fa9c131, before the separate trial seed was introduced. The lowered
 * floor band intentionally changes the vertical motor only. Substituting its frozen historical
 * formula preserves the original digest as an independent pin on every other motor/state field.
 */
final class CruiseDefaultStreamTest {
    @Test
    void nonVerticalMotorsAndOriginalStateFieldsMatchTheHistoricalStream() throws NoSuchAlgorithmException {
        assertEquals("3738a693d07923e15ec0bd43e418f58d9f1231fd272ca74e72cd6a840e124022", streamDigest());
    }

    /** Prints the composite historical-reference digest, not a current executed motor receipt. */
    public static void main(String[] args) throws NoSuchAlgorithmException {
        System.out.println(streamDigest());
    }

    private static String streamDigest() throws NoSuchAlgorithmException {
        var digest = MessageDigest.getInstance("SHA-256");
        var program = new CruiseProgram();
        int changedVerticalSamples = 0;
        for (long entered : new long[] {0, 1, 42, 71_999, 72_001, 9_999_999_999L}) {
            for (var modulation : new Modulation[] {Modulation.NEUTRAL,
                    new Modulation(0.7, 0.8, 0.6, 0.3), new Modulation(0, 0, 0, 1)}) {
                for (double bias : new double[] {-1, 0, 0.25}) {
                    var state = BodyProgramState.entering(BodyProgramId.CRUISE, entered);
                    for (int tick = 0; tick < 400; tick++) {
                        int shape = tick % 7;
                        var snapshot = new PhysicalSensorSnapshot(entered + tick, "default-stream-fixture",
                                1, 64, -2, 0, (tick % 9 - 4) / 10.0, 0, 0, 0,
                                shape == 6, false, false, false, false, 12, 10, 6,
                                shape == 1 ? 0 : shape == 2 ? 1.5 : 6, 6,
                                shape == 3 ? 0.2 : 6, shape == 4 ? 0.2 : 6,
                                shape == 5 ? 0.3 : 6, tick % 61 / 10.0);
                        var step = program.step(state, BodyProgramContext.sensorsOnly(snapshot),
                                modulation, new ProgramParameters(bias));
                        var motors = step.motors();
                        double historicalVertical = historicalVerticalThrust(snapshot);
                        if (Double.doubleToRawLongBits(historicalVertical)
                                != Double.doubleToRawLongBits(motors.verticalThrust())) {
                            changedVerticalSamples++;
                        }
                        var bytes = ByteBuffer.allocate(80);
                        bytes.putDouble(motors.forwardThrust()).putDouble(motors.lateralThrust())
                                .putDouble(historicalVertical).putDouble(motors.yawTorque())
                                .putDouble(motors.pitchTorque()).putFloat(step.presentationPhase())
                                .putInt(step.next().id().ordinal()).putLong(step.next().enteredGameTick())
                                .putInt(step.next().ticksInProgram()).putInt(step.next().phaseIndex())
                                .putInt(step.exitTo().map(Enum::ordinal).orElse(-1));
                        digest.update(bytes.array(), 0, bytes.position());
                        state = step.next();
                    }
                }
            }
        }
        assertTrue(changedVerticalSamples > 0, "the floor change did not affect any vertical sample");
        return HexFormat.of().formatHex(digest.digest());
    }
    /** Frozen 2.5–4.0 floor law from fa9c131, with constants independent of production. */
    private static double historicalVerticalThrust(PhysicalSensorSnapshot snapshot) {
        double floorPush = Math.max(0.0, Math.min(1.0, (2.5 - snapshot.rayDownBlocks()) / 2.5));
        double ceilingPush = Math.max(0.0, Math.min(1.0, (2.5 - snapshot.rayUpBlocks()) / 2.5));
        double targetRate = 0.12 * (floorPush - ceilingPush);
        if (snapshot.rayDownBlocks() > 4.0) { targetRate -= 0.06; }
        double rateError = targetRate - snapshot.velocityYBlocksPerTick();
        return Math.max(-0.45, Math.min(0.45, 0.45 * rateError / 0.12));
    }

}
