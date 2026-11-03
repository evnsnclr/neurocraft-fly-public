package dev.neurocraft.fly.control;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Path;
import java.nio.file.Files;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

final class ComparisonLaneClientCacheTest {
    @TempDir Path directory;

    @Test
    void activeMetadataRoundTripsExactEpochAndUnsignedSeedAndWaitingCannotClaimIt() throws Exception {
        LaneCatalog catalog = ComparisonFixtures.catalog(directory);
        var active = ComparisonLanePresentation.active(catalog.entry("no_input"),
                catalog.resolve("no_input", ComparisonFixtures.EPOCH + 1));
        assertEquals(active, ComparisonLanePresentation.parseOrUnavailable(active.encode()));
        assertEquals("18446744073709551615", active.seed());
        assertThrows(IllegalArgumentException.class, () -> new ComparisonLanePresentation(
                ComparisonLanePresentation.Status.SWITCHING, "no_input", active.epoch(),
                active.activeIdentity(), "no_input", "no_input", active.controlSha256(), active.seed()));
        assertEquals(ComparisonLanePresentation.Status.UNAVAILABLE,
                ComparisonLanePresentation.parseOrUnavailable("broken").status());
    }

    @Test
    void statusChangeClearsOldDisplayAndOnlyAuthenticatedCurrentIdentityIsVisible() throws Exception {
        LaneCatalog catalog = ComparisonFixtures.catalog(directory);
        var cache = new ComparisonLaneClientCache();
        cache.receiveLane(ComparisonLanePresentation.active(catalog.entry("original"),
                ComparisonFixtures.INITIAL), 10);
        cache.receiveDisplay(ComparisonFixtures.display(ComparisonFixtures.INITIAL), 10);
        assertTrue(cache.display(11).frame().isPresent());
        ControllerIdentity next = catalog.resolve("original", ComparisonFixtures.EPOCH + 1);
        cache.receiveLane(ComparisonLanePresentation.waiting(ComparisonLanePresentation.Status.SWITCHING,
                "original", next.sessionEpoch()), 20);
        cache.receiveDisplay(ComparisonFixtures.display(ComparisonFixtures.INITIAL), 21);
        cache.receiveDisplay(ComparisonFixtures.display(next), 21);
        assertTrue(cache.display(22).frame().isEmpty());
        var active = ComparisonLanePresentation.active(catalog.entry("original"), next);
        cache.receiveLane(active, 30);
        cache.receiveDisplay(ComparisonFixtures.display(next), 30);
        cache.receiveDisplay(ComparisonFixtures.display(ComparisonFixtures.INITIAL), 31);
        assertEquals(next, cache.display(32).frame().orElseThrow().identity());
        cache.receiveLane(ComparisonLanePresentation.active(catalog.entry("original"),
                ComparisonFixtures.INITIAL), 33);
        assertEquals(active, cache.lane());
        cache.receiveLane(ComparisonLanePresentation.waiting(ComparisonLanePresentation.Status.FAILED,
                "original", next.sessionEpoch()), 40);
        cache.receiveLane(active, 41);
        cache.receiveDisplay(ComparisonFixtures.display(next), 42);
        assertEquals(ComparisonLanePresentation.Status.FAILED, cache.lane().status());
        assertTrue(cache.display(43).frame().isEmpty());
    }

    @Test
    void ordinaryDisplayStillHasReceiveTimeFreshnessWithoutComparisonMetadata() {
        var cache = new ComparisonLaneClientCache();
        cache.receiveDisplay(ComparisonFixtures.display(ComparisonFixtures.INITIAL), 10);
        assertTrue(cache.display(11).frame().isPresent());
        assertEquals(BrainDisplayState.Status.STALE,
                cache.display(BrainDisplayState.MAX_AGE_NANOS + 2_000_000).status());
    }

    @Test
    void freshAuthenticatedLaneEnablesExactlyV3AndV4BeforeAnyEventFrameArrives() {
        for (String hash : List.of("5".repeat(64), ControlProtocol.OBSERVATION_SCHEMA_V2_SHA256,
                ControlProtocol.OBSERVATION_SCHEMA_V3_SHA256, ControlProtocol.OBSERVATION_SCHEMA_V4_SHA256,
                "f".repeat(64))) {
            var old = ComparisonFixtures.INITIAL;
            var identity = new ControllerIdentity(old.runId(), old.sessionEpoch(), old.entityId(),
                    old.graphHash(), old.adapterHash(), old.dynamicsHash(), old.decoderHash(), hash);
            var cache = new ComparisonLaneClientCache();
            cache.receiveLane(active(identity), 100);
            boolean supported = hash.equals(ControlProtocol.OBSERVATION_SCHEMA_V3_SHA256)
                    || hash.equals(ControlProtocol.OBSERVATION_SCHEMA_V4_SHA256);
            assertEquals(supported ? EventDisplayState.Status.UNAVAILABLE : EventDisplayState.Status.DISABLED,
                    cache.events(101).status(), hash);
            assertTrue(cache.events(101).frame().isEmpty());
        }
    }

    @Test
    void v4ActivationStillFencesEpochIdentityAndReceiveFreshness() throws Exception {
        var frame = EventDisplayFrame.parse(Files.readString(
                Path.of("..", "packages", "protocol", "golden", "interaction_display_v2.json")));
        var cache = new ComparisonLaneClientCache();
        cache.receiveLane(active(frame.identity()), 100);
        assertEquals(EventDisplayState.Status.UNAVAILABLE, cache.events(100).status());
        var matched = new EventDisplayPresentation(EventDisplayState.Status.MATCHED, 0, Optional.of(frame));
        cache.receiveEvents(matched, 100);
        assertEquals(matched, cache.events(101));
        long expired = EventDisplayState.MAX_AGE_NANOS + 2_000_000;
        assertEquals(EventDisplayState.Status.STALE, cache.events(expired).status());
        var old = frame.identity();
        var next = new ControllerIdentity(old.runId(), old.sessionEpoch() + 1, old.entityId(),
                old.graphHash(), old.adapterHash(), old.dynamicsHash(), old.decoderHash(), old.observationSchemaHash());
        cache.receiveLane(active(next), expired + 100);
        cache.receiveEvents(matched, expired + 101);
        cache.receiveLane(active(old), expired + 102);
        assertEquals(next, cache.lane().activeIdentity().orElseThrow());
        assertEquals(EventDisplayState.Status.UNAVAILABLE, cache.events(expired + 103).status());
    }

    private static ComparisonLanePresentation active(ControllerIdentity identity) {
        return new ComparisonLanePresentation(ComparisonLanePresentation.Status.ACTIVE, "original",
                identity.sessionEpoch(), Optional.of(identity), "original", "faithful", null, null);
    }
}
