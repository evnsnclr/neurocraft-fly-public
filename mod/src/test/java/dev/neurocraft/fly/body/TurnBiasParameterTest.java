package dev.neurocraft.fly.body;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import dev.neurocraft.fly.body.programs.CruiseProgram;
import dev.neurocraft.fly.body.programs.EscapeJumpProgram;
import dev.neurocraft.fly.body.programs.HoverProgram;
import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import dev.neurocraft.fly.control.PrimitiveAction;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Which programs read the one signed program input, pinned as bit-identity.
 *
 * <p>A program declares which selector inputs it reads, and the declaration is worth nothing
 * unless varying an undeclared one provably changes nothing. These are pure functions over a
 * synthetic sensor sample, so they are JUnit tests rather than GameTests: no world is involved and
 * running a Minecraft server would only make the evidence harder to read.
 *
 * <p>Every value here is a literal. Nothing in this file is a claim about the connectome.
 */
final class TurnBiasParameterTest {
    /** Both saturating ends, both halves, and neutral. */
    private static final List<ProgramParameters> BIASES = List.of(
            new ProgramParameters(-1.0),
            new ProgramParameters(-0.5),
            ProgramParameters.NEUTRAL,
            new ProgramParameters(0.5),
            new ProgramParameters(1.0));

    private static final int MOTOR_CHANNELS = 5;

    @Test
    void hoverIgnoresTheTurnBias() {
        long[] declared = motorWords(new HoverProgram(), BodyProgramId.HOVER, BIASES.get(2), 200);
        for (ProgramParameters bias : BIASES) {
            assertArrayEquals(
                    declared,
                    motorWords(new HoverProgram(), BodyProgramId.HOVER, bias, 200),
                    "HOVER read a signed program input it does not declare: " + bias);
        }
    }

    @Test
    void escapeJumpIgnoresTheTurnBias() {
        // Two profiles' worth, so the clamped over-run past the exit tick is swept too.
        int ticks = 2 * EscapeJumpProgram.ESCAPE_JUMP_TICKS;
        long[] declared = motorWords(
                new EscapeJumpProgram(), BodyProgramId.ESCAPE_JUMP, BIASES.get(2), ticks);
        for (ProgramParameters bias : BIASES) {
            assertArrayEquals(
                    declared,
                    motorWords(new EscapeJumpProgram(), BodyProgramId.ESCAPE_JUMP, bias, ticks),
                    "ESCAPE_JUMP read a signed program input it does not declare: " + bias);
        }
    }

    /**
     * The discriminating half: CRUISE <em>does</em> read it, so the two identities above are
     * statements about those programs rather than about a parameter nothing can reach.
     */
    @Test
    void cruiseReadsTheTurnBias() {
        long[] declared = motorWords(new CruiseProgram(), BodyProgramId.CRUISE, BIASES.get(2), 200);
        for (ProgramParameters bias : List.of(BIASES.get(0), BIASES.get(4))) {
            assertFalse(
                    Arrays.equals(
                            declared, motorWords(new CruiseProgram(), BodyProgramId.CRUISE, bias, 200)),
                    "CRUISE ignored the signed program input it does declare: " + bias);
        }
    }

    private static long[] motorWords(
            BodyProgram program, BodyProgramId id, ProgramParameters parameters, int ticks) {
        BodyProgramContext context = BodyProgramContext.sensorsOnly(snapshot());
        BodyProgramState state = BodyProgramState.entering(id, 2026L);
        long[] words = new long[ticks * MOTOR_CHANNELS];
        for (int tick = 0; tick < ticks; tick++) {
            BodyProgramStep step = program.step(state, context, Modulation.NEUTRAL, parameters);
            PrimitiveAction motors = step.motors();
            int base = tick * MOTOR_CHANNELS;
            words[base] = Double.doubleToRawLongBits(motors.forwardThrust());
            words[base + 1] = Double.doubleToRawLongBits(motors.lateralThrust());
            words[base + 2] = Double.doubleToRawLongBits(motors.verticalThrust());
            words[base + 3] = Double.doubleToRawLongBits(motors.yawTorque());
            words[base + 4] = Double.doubleToRawLongBits(motors.pitchTorque());
            state = step.next();
        }
        return words;
    }

    /**
     * One asymmetric sensory sample: a drift to settle, an obstruction ahead and one to the side,
     * so HOVER's settle term, ESCAPE_JUMP's away-turn latch, and CRUISE's avoidance are all live
     * and a bias that leaked into any of them would show.
     */
    private static PhysicalSensorSnapshot snapshot() {
        return new PhysicalSensorSnapshot(
                100L,
                "00000000-0000-0000-0000-000000000001",
                1.0,
                64.0,
                -2.0,
                0.12,
                0.0,
                -0.07,
                37.0,
                0.0,
                false,
                false,
                false,
                false,
                false,
                12,
                10,
                6.0,
                3.2,
                6.0,
                5.0,
                1.1,
                4.0,
                3.5);
    }
}
