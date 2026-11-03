package dev.neurocraft.fly.control;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.OptionalLong;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

final class ComparisonLaneSessionTest {
    @TempDir Path directory;

    @Test
    void fourLanesConsumeExactEpochsAndWaitForAuthenticationBeforeLabelingActive() throws Exception {
        LaneCatalog catalog = ComparisonFixtures.catalog(directory);
        ComparisonFixtures.Probe original = new ComparisonFixtures.Probe();
        original.state = ControllerFault.ACTIVE;
        List<ComparisonFixtures.Probe> built = new ArrayList<>();
        List<BridgeConfig> configurations = new ArrayList<>();
        AtomicLong resets = new AtomicLong();
        ComparisonLaneSession session = new ComparisonLaneSession(catalog, ComparisonFixtures.config(),
                original.bridge, config -> {
                    assertTrue(original.closed);
                    assertEquals(configurations.size() + 1, resets.get());
                    configurations.add(config);
                    var probe = new ComparisonFixtures.Probe();
                    built.add(probe);
                    return probe.bridge;
                }, System::nanoTime);
        assertEquals("original", session.presentation().activeLane());
        int offset = 0;
        for (String name : List.of("shuffled_weight", "no_input", "original")) {
            offset++;
            session.switchTo(name, resets::incrementAndGet);
            ComparisonLanePresentation waiting = session.presentation();
            assertEquals(ComparisonLanePresentation.Status.SWITCHING, waiting.status());
            assertEquals(name, waiting.attemptedLane());
            assertEquals(ComparisonFixtures.EPOCH + offset, waiting.epoch());
            assertTrue(waiting.activeIdentity().isEmpty());
            assertNull(waiting.activeLane());
            assertNull(waiting.controlSha256());
            assertThrows(IllegalStateException.class, () -> session.switchTo("original", () -> fail()));
            original.state = ControllerFault.ACTIVE; // Late old-connection updates cannot activate it.
            assertTrue(session.presentation().activeIdentity().isEmpty());
            built.getLast().state = ControllerFault.ACTIVE;
            assertEquals(name, session.presentation().activeLane());
            assertEquals(session.config().identity(), session.presentation().activeIdentity().orElseThrow());
        }
        assertEquals(3, configurations.size());
        assertEquals(ComparisonFixtures.INITIAL.dynamicsHash(), session.config().identity().dynamicsHash());
        assertEquals(ComparisonFixtures.EPOCH + 3, session.config().identity().sessionEpoch());
        assertTrue(built.get(0).closed && built.get(1).closed);
    }

    @Test
    void invalidLaneCannotEvictAndStartFailureConsumesEpochForNextAttempt() throws Exception {
        var original = new ComparisonFixtures.Probe();
        original.state = ControllerFault.ACTIVE;
        List<ComparisonFixtures.Probe> built = new ArrayList<>();
        ComparisonLaneSession session = new ComparisonLaneSession(ComparisonFixtures.catalog(directory),
                ComparisonFixtures.config(), original.bridge, config -> {
                    var next = new ComparisonFixtures.Probe();
                    next.failStart = built.isEmpty();
                    built.add(next);
                    return next.bridge;
                }, System::nanoTime);
        assertThrows(IllegalArgumentException.class, () -> session.switchTo("invented", () -> fail()));
        assertFalse(original.closed);
        assertEquals(ComparisonFixtures.EPOCH, session.config().identity().sessionEpoch());
        session.switchTo("no_input", () -> {});
        assertEquals(ComparisonLanePresentation.Status.FAILED, session.presentation().status());
        assertTrue(session.presentation().activeIdentity().isEmpty());
        assertTrue(original.closed && built.getFirst().closed);
        session.switchTo("original", () -> {});
        assertEquals(ComparisonFixtures.EPOCH + 2, session.config().identity().sessionEpoch());
    }

    @Test
    void unansweredHandshakeFailsNeutralWithoutReusingItsEpoch() throws Exception {
        try (var logs = new LifecycleLogCapture("neurocraft-fly-comparison")) {
            AtomicLong clock = new AtomicLong();
            var old = new ComparisonFixtures.Probe();
            old.state = ControllerFault.ACTIVE;
            var pending = new ComparisonFixtures.Probe();
            ComparisonLaneSession session = new ComparisonLaneSession(ComparisonFixtures.catalog(directory),
                    ComparisonFixtures.config(), old.bridge, config -> pending.bridge, clock::get);
            session.switchTo("shuffled_weight", () -> {});
            clock.set(BridgeConfig.CONNECT_TIMEOUT.toNanos());
            assertFalse(session.pending());
            assertEquals(ComparisonLanePresentation.Status.FAILED, session.presentation().status());
            assertTrue(pending.closed);
            pending.state = ControllerFault.ACTIVE;
            assertTrue(session.presentation().activeIdentity().isEmpty());
            assertEquals(ComparisonFixtures.EPOCH + 1, session.presentation().epoch());
            assertEquals(1, logs.stages().stream().filter("handshake_watchdog_expired"::equals).count());
            String terminal = logs.messages().getLast();
            assertTrue(terminal.contains("epoch=" + (ComparisonFixtures.EPOCH + 1) + " "));
            assertTrue(terminal.contains("handover_elapsed_ms=3000"));
            assertTrue(terminal.contains("fault=DISCONNECTED"));
            assertEquals(List.of("bridge_active_observed", "switch_begin", "previous_bridge_close_begin",
                    "previous_bridge_close_returned", "body_session_reset_begin", "body_session_reset_complete",
                    "new_bridge_start_begin", "new_bridge_start_returned", "handshake_watchdog_expired"), logs.stages());
        }
    }

    @Test
    void oldQueuedCommandIsRetiredAndCannotEnterFreshValidatorEvenAfterRestoringOriginal() throws Exception {
        var old = new ComparisonFixtures.Probe();
        old.state = ControllerFault.ACTIVE;
        ActionFrame stale = ComparisonFixtures.command(ComparisonFixtures.INITIAL);
        old.commands.add(stale);
        var fresh = new ComparisonFixtures.Probe();
        ComparisonLaneSession session = new ComparisonLaneSession(ComparisonFixtures.catalog(directory),
                ComparisonFixtures.config(), old.bridge, config -> fresh.bridge, System::nanoTime);
        session.switchTo("original", () -> {});
        assertTrue(old.commands.isEmpty());
        assertTrue(session.bridge().pollAction().isEmpty());
        ControllerIdentity next = session.config().identity();
        ValidationResult refused = new CommandValidator(next).validate(List.of(stale), 102,
                ControllerFault.ACTIVE, sequence -> OptionalLong.of(100));
        assertEquals(ControllerFault.IDENTITY_MISMATCH, refused.state());
        assertFalse(refused.accepted());
        ValidationResult accepted = new CommandValidator(next).validate(
                List.of(ComparisonFixtures.command(next)), 102, ControllerFault.ACTIVE,
                sequence -> OptionalLong.of(100));
        assertTrue(accepted.accepted());
    }
}
