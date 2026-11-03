package dev.neurocraft.fly.body;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/**
 * The synchronized readout word is a lossless codec at its declared hundredths resolution, and it
 * is only ever a rendering: the exact values the frame carried are reported by
 * {@code /neurocraftfly status}, not read back out of here.
 */
final class ReadoutChannelTest {
    @Test
    void everyHundredthRoundTripsThroughThePackedWord() {
        int steps = ReadoutChannel.PUBLISHED_READOUT_STEPS;
        for (int level = -steps; level <= steps; level++) {
            double signed = level / (double) steps;
            double unsigned = Math.abs(signed);
            ReadoutChannel channel =
                    new ReadoutChannel(new ReadoutValues(unsigned, signed, -signed));
            int packed = channel.packed();
            assertTrue(packed >= 0, "packed readout word must stay positive");
            assertEquals(channel, ReadoutChannel.decode(packed));
        }
    }

    @Test
    void anExactReadoutIsPublishedAtHundredths() {
        ReadoutChannel channel =
                new ReadoutChannel(new ReadoutValues(0.4213, -0.0120, 0.6149));
        ReadoutValues published = ReadoutChannel.decode(channel.packed()).readout();

        assertEquals(0.42, published.ttmnJump(), 1.0e-12);
        assertEquals(-0.01, published.dna01Turn(), 1.0e-12);
        assertEquals(0.61, published.wingCommon(), 1.0e-12);
    }

    /**
     * Two of the three scalars are signed, so a half must round to the same magnitude either way.
     * With {@link Math#round}'s toward-positive-infinity halves, -0.005 published as 0.00 while
     * +0.005 published as +0.01.
     */
    @Test
    void mirroredReadoutsPublishAtTheSameMagnitude() {
        for (int thousandths = 0; thousandths <= 1000; thousandths += 5) {
            double value = thousandths / 1000.0;
            ReadoutValues positive = published(new ReadoutValues(0.0, value, value));
            ReadoutValues negative = published(new ReadoutValues(0.0, -value, -value));
            assertEquals(
                    positive.dna01Turn(), -negative.dna01Turn(), 1.0e-12, "dna01 " + value);
            assertEquals(
                    positive.wingCommon(), -negative.wingCommon(), 1.0e-12, "wing " + value);
        }
        assertEquals(0.01, published(new ReadoutValues(0.0, 0.005, 0.0)).dna01Turn(), 1.0e-12);
        assertEquals(-0.01, published(new ReadoutValues(0.0, -0.005, 0.0)).dna01Turn(), 1.0e-12);
    }

    private static ReadoutValues published(ReadoutValues exact) {
        return ReadoutChannel.decode(new ReadoutChannel(exact).packed()).readout();
    }

    @Test
    void theDefaultIsAllZeroAndSurvivesTheRoundTrip() {
        assertEquals(ReadoutValues.ZERO, ReadoutChannel.DEFAULT.readout());
        assertEquals(ReadoutChannel.DEFAULT, ReadoutChannel.decode(ReadoutChannel.DEFAULT.packed()));
    }

    /**
     * Eight bits hold levels 0-255 but only 0-200 are ever written, so a desynchronized or hostile
     * packet can carry 201-255. Decoding must stay inside the declared ranges rather than producing
     * a value the runtime's own bounds would have refused, and must never throw in a render loop.
     */
    @Test
    void aMalformedWordDegradesInsteadOfThrowing() {
        assertEquals(ReadoutChannel.DEFAULT, ReadoutChannel.decode(-1));

        ReadoutValues saturated = ReadoutChannel.decode(0x00FFFFFF).readout();
        assertEquals(1.0, saturated.ttmnJump());
        assertEquals(1.0, saturated.dna01Turn());
        assertEquals(1.0, saturated.wingCommon());

        ReadoutValues floored = ReadoutChannel.decode(0).readout();
        assertEquals(0.0, floored.ttmnJump(), "a rectified term never publishes as negative");
        assertEquals(-1.0, floored.dna01Turn());
        assertEquals(-1.0, floored.wingCommon());
    }
}
