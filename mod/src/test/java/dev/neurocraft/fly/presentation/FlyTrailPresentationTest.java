package dev.neurocraft.fly.presentation;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

final class FlyTrailPresentationTest {
    @Test
    void colorIsAPureDeterministicFunctionOfTheDecoderHash() {
        String hash = "4b".repeat(32);
        assertEquals(
                FlyTrailColor.fromDecoderHash(hash), FlyTrailColor.fromDecoderHash(hash));
        assertEquals(0x4C4C4C, FlyTrailColor.fromDecoderHash("1".repeat(64)));
        assertEquals(0x595959, FlyTrailColor.fromDecoderHash("2".repeat(64)));
    }

    @Test
    void distinctDecoderHashesYieldDistinctColors() {
        assertNotEquals(
                FlyTrailColor.fromDecoderHash("1".repeat(64)),
                FlyTrailColor.fromDecoderHash("2".repeat(64)));
        assertNotEquals(
                FlyTrailColor.fromDecoderHash("a1b2c3" + "0".repeat(58)),
                FlyTrailColor.fromDecoderHash("c3b2a1" + "0".repeat(58)));
    }

    @Test
    void lightnessFloorBoundsEveryChannelWithoutClippingTheTop() {
        assertEquals(0x404040, FlyTrailColor.fromDecoderHash("0".repeat(64)));
        assertEquals(0xFFFFFF, FlyTrailColor.fromDecoderHash("f".repeat(64)));
        assertEquals(0x40, FlyTrailColor.MIN_CHANNEL);
    }

    @Test
    void trailDefaultsAreDisabledWithIntervalThreeAndBoundedMutation() {
        assertFalse(FlyTrailPresentation.enabled());
        assertEquals(3, FlyTrailPresentation.intervalTicks());
        assertEquals(0, FlyTrailPresentation.emittedParticleCount());
        assertEquals(0xB4B4B4, FlyTrailColor.FALLBACK_RGB);
        assertThrows(
                IllegalArgumentException.class,
                () -> FlyTrailPresentation.setIntervalTicks(
                        FlyTrailPresentation.MIN_INTERVAL_TICKS - 1));
        assertThrows(
                IllegalArgumentException.class,
                () -> FlyTrailPresentation.setIntervalTicks(
                        FlyTrailPresentation.MAX_INTERVAL_TICKS + 1));
    }
}
