package dev.neurocraft.fly.control;

import static org.junit.jupiter.api.Assertions.*;
import java.util.LinkedHashMap;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class BrainDisplayClientCacheTest {
    private static BrainDisplayPresentation matched(long ageMillis) {
        ControllerIdentity identity = new ControllerIdentity("display-test", 1, "fly",
                "a".repeat(64), "b".repeat(64), "c".repeat(64), "d".repeat(64), "e".repeat(64));
        var populations = new LinkedHashMap<String, BrainDisplayFrame.Population>();
        for (String name : BrainDisplayFrame.POPULATION_NAMES) {
            populations.put(name, new BrainDisplayFrame.Population(0.01, 1));
        }
        var frame = new BrainDisplayFrame(identity, 0, 100, 0, 5, "f".repeat(64),
                "synthetic-display-test", "1".repeat(64), populations);
        return new BrainDisplayPresentation(BrainDisplayState.Status.MATCHED, ageMillis,
                Optional.of(frame));
    }

    @Test void serverAgeAndLocalElapsedBothConsumeFreshness() {
        var cache = new BrainDisplayClientCache();
        cache.receive(matched(900), 1_000_000_000L);
        assertEquals(BrainDisplayState.Status.MATCHED, cache.view(1_100_000_000L).status());
        assertEquals(BrainDisplayState.Status.STALE, cache.view(1_101_000_000L).status());
        assertTrue(cache.view(1_101_000_000L).frame().isEmpty());
    }

    @Test void openingHudAfterStalledPacketDoesNotStartANewFreshnessWindow() {
        var cache = new BrainDisplayClientCache();
        cache.receive(matched(0), 1_000_000_000L);
        assertEquals(BrainDisplayState.Status.STALE, cache.view(2_100_000_000L).status());
        assertEquals(BrainDisplayState.Status.STALE, cache.view(2_200_000_000L).status());
        cache.receive(matched(50), 2_300_000_000L);
        assertEquals(BrainDisplayState.Status.MATCHED, cache.view(2_300_000_000L).status());
    }

    @Test void clockRollbackAndExtremeAgeCannotResurrectAFrame() {
        var cache = new BrainDisplayClientCache();
        cache.receive(matched(0), 100);
        assertEquals(BrainDisplayState.Status.STALE, cache.view(99).status());
        cache.receive(matched(Long.MAX_VALUE), 0);
        assertEquals(BrainDisplayState.Status.STALE, cache.view(10_000_000L).status());
    }
}
