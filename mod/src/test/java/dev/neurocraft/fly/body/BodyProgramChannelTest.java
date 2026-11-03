package dev.neurocraft.fly.body;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** The synchronized body-program word is a lossless codec for everything the HUD names. */
final class BodyProgramChannelTest {
    @Test
    void everySourceAndProgramRoundTripsThroughThePackedWord() {
        for (SelectionSource source : SelectionSource.values()) {
            for (BodyProgramId program : BodyProgramId.values()) {
                BodyProgramChannel channel = new BodyProgramChannel(
                        source, program, Modulation.NEUTRAL, ProgramParameters.NEUTRAL);
                int packed = channel.packed();
                assertTrue(packed >= 0, "packed channel word must stay positive");
                assertEquals(channel, BodyProgramChannel.decode(packed));
            }
        }
    }

    @Test
    void modulationRoundTripsExactlyAtThePublishedResolution() {
        int steps = BodyProgramChannel.PUBLISHED_MODULATION_STEPS;
        for (int level = 0; level <= steps; level++) {
            double value = level / (double) steps;
            double complement = (steps - level) / (double) steps;
            BodyProgramChannel channel = new BodyProgramChannel(
                    SelectionSource.OPERATOR,
                    BodyProgramId.HOVER,
                    new Modulation(value, complement, value, complement),
                    ProgramParameters.NEUTRAL);
            assertEquals(channel, BodyProgramChannel.decode(channel.packed()));
        }
    }

    @Test
    void anExactModulationIsPublishedAtTenthsAndSaysSo() {
        BodyProgramChannel channel = new BodyProgramChannel(
                SelectionSource.OPERATOR,
                BodyProgramId.HOVER,
                new Modulation(0.6, 1.0, 0.44, 0.0),
                ProgramParameters.NEUTRAL);
        Modulation published = BodyProgramChannel.decode(channel.packed()).modulation();
        assertEquals(0.6, published.speed(), 1.0e-12);
        assertEquals(1.0, published.turnRate(), 1.0e-12);
        assertEquals(0.4, published.amplitude(), 1.0e-12, "0.44 publishes at the nearest tenth");
        assertEquals(0.0, published.urgency(), 1.0e-12);
    }

    /**
     * The turn bias is the one signed field in the packed word, so it needs its own round trip: a
     * sign lost in the packing would show the HUD a bias pointing the other way.
     */
    @Test
    void turnBiasLevelRoundTrips() {
        int steps = BodyProgramChannel.PUBLISHED_MODULATION_STEPS;
        for (int level = -steps; level <= steps; level++) {
            double bias = level / (double) steps;
            BodyProgramChannel channel = new BodyProgramChannel(
                    SelectionSource.READOUT,
                    BodyProgramId.CRUISE,
                    new Modulation(0.7, 1.0, 1.0, 0.0),
                    new ProgramParameters(bias));
            int packed = channel.packed();
            assertTrue(packed >= 0, "packed channel word must stay positive with a signed bias");
            assertEquals(channel, BodyProgramChannel.decode(packed));
        }
    }

    /** The bias publishes at the same tenths the modulation does, and says so. */
    @Test
    void anExactTurnBiasIsPublishedAtTenths() {
        BodyProgramChannel channel = new BodyProgramChannel(
                SelectionSource.READOUT,
                BodyProgramId.CRUISE,
                Modulation.NEUTRAL,
                new ProgramParameters(-0.44));

        assertEquals(
                -0.4,
                BodyProgramChannel.decode(channel.packed()).parameters().turnBias(),
                1.0e-12,
                "-0.44 publishes at the nearest tenth");
    }

    /**
     * A left bias and its mirror publish at the same size. {@link Math#round} rounds a half toward
     * positive infinity, so the naive encoding published -0.25 as -0.2 and +0.25 as +0.3: a signed
     * channel whose two directions disagreed, and a HUD that made a leftward bias look weaker than
     * the identical rightward one.
     */
    @Test
    void mirroredTurnBiasesPublishAtTheSameMagnitude() {
        for (int hundredths = 0; hundredths <= 100; hundredths++) {
            double bias = hundredths / 100.0;
            double positive = decodedBias(bias);
            double negative = decodedBias(-bias);
            assertEquals(positive, -negative, 1.0e-12, "bias " + bias + " is not mirrored");
        }
        assertEquals(0.3, decodedBias(0.25), 1.0e-12);
        assertEquals(-0.3, decodedBias(-0.25), 1.0e-12);
    }

    private static double decodedBias(double turnBias) {
        return BodyProgramChannel.decode(new BodyProgramChannel(
                                SelectionSource.READOUT,
                                BodyProgramId.CRUISE,
                                Modulation.NEUTRAL,
                                new ProgramParameters(turnBias))
                        .packed())
                .parameters()
                .turnBias();
    }

    /** Every other field keeps its value when the bias moves; the fields do not share bits. */
    @Test
    void theBiasFieldDoesNotDisturbTheModulationFields() {
        Modulation modulation = new Modulation(0.3, 0.6, 0.9, 1.0);
        for (int level = -10; level <= 10; level++) {
            BodyProgramChannel decoded = BodyProgramChannel.decode(new BodyProgramChannel(
                            SelectionSource.READOUT,
                            BodyProgramId.ESCAPE_JUMP,
                            modulation,
                            new ProgramParameters(level / 10.0))
                    .packed());
            assertEquals(modulation, decoded.modulation());
            assertEquals(BodyProgramId.ESCAPE_JUMP, decoded.program());
            assertEquals(SelectionSource.READOUT, decoded.source());
        }
    }

    @Test
    void aMalformedWordDegradesToTheDefaultInsteadOfThrowing() {
        assertEquals(BodyProgramChannel.DEFAULT, BodyProgramChannel.decode(-1));
        assertEquals(BodyProgramChannel.DEFAULT, BodyProgramChannel.decode(0x0F));
    }

    /**
     * Four bits hold levels 0-15 but only 0-10 are ever written, so a desynchronized or hostile
     * packet can carry 11-15. Decoding must stay inside the unit interval rather than producing a
     * modulation the runtime's own clamp would have refused.
     */
    @Test
    void outOfRangeModulationLevelsDecodeToTheUnitInterval() {
        for (int level = BodyProgramChannel.PUBLISHED_MODULATION_STEPS + 1; level <= 15; level++) {
            int packed = BodyProgramId.HOVER.ordinal()
                    | (SelectionSource.OPERATOR.ordinal() << 4)
                    | (level << 6)
                    | (level << 10)
                    | (level << 14)
                    | (level << 18);
            Modulation decoded = BodyProgramChannel.decode(packed).modulation();
            assertEquals(1.0, decoded.speed(), 1.0e-12, "level " + level);
            assertEquals(1.0, decoded.turnRate(), 1.0e-12, "level " + level);
            assertEquals(1.0, decoded.amplitude(), 1.0e-12, "level " + level);
            assertEquals(1.0, decoded.urgency(), 1.0e-12, "level " + level);
        }
    }

    @Test
    void modulationClampsIntoTheUnitInterval() {
        Modulation clamped = new Modulation(5.0, -2.0, 0.5, 11.0);
        assertEquals(1.0, clamped.speed());
        assertEquals(0.0, clamped.turnRate());
        assertEquals(0.5, clamped.amplitude());
        assertEquals(1.0, clamped.urgency());
    }
}
