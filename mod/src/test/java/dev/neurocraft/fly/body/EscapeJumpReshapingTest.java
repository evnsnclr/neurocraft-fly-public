package dev.neurocraft.fly.body;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.neurocraft.fly.body.programs.EscapeJumpProgram;
import org.junit.jupiter.api.Test;

/**
 * ADR 0020 lane D: the urgency reshaping is a pure integer function, so it is pinned here rather
 * than inside a Minecraft server.
 *
 * <p>The property that matters is not any single phase length but the invariant across all of them:
 * <strong>the jump is always {@code ESCAPE_JUMP_TICKS} long</strong>. Stage 4b's policy holds
 * ESCAPE_JUMP for 24 ticks and re-sends its urgency on every frame; a total that varied with
 * urgency would let the program exit early and be re-selected into a second jump inside one hold.
 */
final class EscapeJumpReshapingTest {
    @Test
    void everyUrgencyGivesTheSameTotalDuration() {
        for (int percent = -50; percent <= 150; percent++) {
            int crouch = EscapeJumpProgram.crouchTicks(percent);
            int launch = EscapeJumpProgram.launchTicks(percent);
            int brake = EscapeJumpProgram.brakeTicks(percent);
            assertEquals(
                    EscapeJumpProgram.ESCAPE_JUMP_TICKS,
                    crouch + launch + brake,
                    "phases must sum to the fixed total at urgency " + percent + "%");
        }
    }

    @Test
    void reshapingIsMonotoneAndNeverPassesAPhaseFloor() {
        int previousCrouch = Integer.MAX_VALUE;
        int previousLaunch = Integer.MAX_VALUE;
        for (int percent = 0; percent <= 100; percent++) {
            int crouch = EscapeJumpProgram.crouchTicks(percent);
            int launch = EscapeJumpProgram.launchTicks(percent);
            assertTrue(
                    crouch <= previousCrouch && launch <= previousLaunch,
                    "reshaping is not monotone at " + percent + "%: " + crouch + "/" + launch);
            assertTrue(
                    crouch >= EscapeJumpProgram.ESCAPE_CROUCH_MIN_TICKS
                            && launch >= EscapeJumpProgram.ESCAPE_LAUNCH_MIN_TICKS,
                    "reshaping passed a floor at " + percent + "%: " + crouch + "/" + launch);
            previousCrouch = crouch;
            previousLaunch = launch;
        }
    }

    @Test
    void theTwoEndpointsAreTheDeclaredProfiles() {
        assertEquals(EscapeJumpProgram.ESCAPE_CROUCH_TICKS, EscapeJumpProgram.crouchTicks(0));
        assertEquals(EscapeJumpProgram.ESCAPE_LAUNCH_TICKS, EscapeJumpProgram.launchTicks(0));
        assertEquals(10, EscapeJumpProgram.brakeTicks(0));
        assertEquals(EscapeJumpProgram.ESCAPE_CROUCH_MIN_TICKS, EscapeJumpProgram.crouchTicks(100));
        assertEquals(EscapeJumpProgram.ESCAPE_LAUNCH_MIN_TICKS, EscapeJumpProgram.launchTicks(100));
        assertEquals(15, EscapeJumpProgram.brakeTicks(100));
    }

    /**
     * The brake at rest emits <strong>positive</strong> zero.
     *
     * <p>{@code -0.0 / x} is {@code -0.0} in IEEE-754, and the HUD renders the published vertical
     * channel with a sign, so a stopped body would read "V -0.00" on every brake tick of every
     * jump. Nothing else about the program changes; this pins the one bit that does.
     */
    @Test
    void brakeAtRestIsPositiveZero() {
        assertEquals(
                0L,
                Double.doubleToRawLongBits(EscapeJumpProgram.brakeVertical(0.0)),
                "the brake emitted negative zero for a body at rest");
        assertEquals(
                0L,
                Double.doubleToRawLongBits(EscapeJumpProgram.brakeVertical(-0.0)),
                "the brake emitted negative zero for a body at negative-zero speed");
        // And the sign is still correct for real speeds either side of rest.
        assertTrue(
                EscapeJumpProgram.brakeVertical(0.1) < 0.0
                        && EscapeJumpProgram.brakeVertical(-0.1) > 0.0,
                "normalizing the zero must not change any non-zero brake command");
    }

    @Test
    void urgencyIsQuantizedToPercentAndClampedBothWays() {
        assertEquals(0, EscapeJumpProgram.urgencyPercent(new Modulation(1, 1, 1, 0.0)));
        assertEquals(50, EscapeJumpProgram.urgencyPercent(new Modulation(1, 1, 1, 0.5)));
        assertEquals(100, EscapeJumpProgram.urgencyPercent(new Modulation(1, 1, 1, 1.0)));
        // Modulation clamps out-of-range requests before the program ever sees them.
        assertEquals(100, EscapeJumpProgram.urgencyPercent(new Modulation(1, 1, 1, 5.0)));
        assertEquals(0, EscapeJumpProgram.urgencyPercent(new Modulation(1, 1, 1, -3.0)));
        // And the reshaping itself refuses a percent outside its own range.
        assertEquals(
                EscapeJumpProgram.crouchTicks(0),
                EscapeJumpProgram.reshape(
                        EscapeJumpProgram.ESCAPE_CROUCH_TICKS,
                        EscapeJumpProgram.ESCAPE_CROUCH_MIN_TICKS,
                        -20));
        assertEquals(
                EscapeJumpProgram.crouchTicks(100),
                EscapeJumpProgram.reshape(
                        EscapeJumpProgram.ESCAPE_CROUCH_TICKS,
                        EscapeJumpProgram.ESCAPE_CROUCH_MIN_TICKS,
                        220));
    }
}
