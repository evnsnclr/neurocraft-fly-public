package dev.neurocraft.fly.control;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

final class EventDisplayTest {
    @TempDir Path directory;

    // Serialized by the production Python NamedEventEvidence API, with explicitly synthetic means.
    static String json() throws Exception {
        return Files.readString(Path.of("src/gametest/resources/neurocraft/event-display.json"));
    }
    static EventDisplayFrame frame() throws Exception { return EventDisplayFrame.parse(json()); }
    static Map<String, Object> values() throws Exception { return StrictJson.parseObject(json(), 16384); }
    static EventDisplayFrame parse(Map<String, Object> values) throws Exception {
        return EventDisplayFrame.parse(StrictJson.encode(values, 16384));
    }
    static EventDisplayReader.Snapshot snapshot(EventDisplayFrame frame, long time) {
        return new EventDisplayReader.Snapshot(EventDisplayReader.Status.AVAILABLE, Optional.of(frame), time);
    }
    static Map<String, Object> object(Map<String, Object> parent, String key) throws Exception {
        return BrainDisplayFrame.object(parent, key);
    }

    @Test void readsProductionPythonShapePreservingCompleteCommandAndIssuedOrigin() throws Exception {
        var frame = frame();
        assertEquals(1783000000000000007L, frame.identity().sessionEpoch());
        assertEquals(0.0008, frame.mn9().left());
        assertEquals(0.0002, frame.mn9().right());
        assertEquals("FEED", frame.command().program().name());
        assertEquals("LAND", frame.trigger().command().program().name());
        assertEquals(4, frame.trigger().command().commandSequence());
        assertEquals(5, frame.commandSequence());
        assertEquals(0.0005, frame.trigger().score().score());
        assertTrue(frame.matches(frame.command()));
        assertFalse(frame.matches(frame.trigger().command()));
        assertEquals(frame, parse(frame.toMap()));
        assertThrows(UnsupportedOperationException.class, () -> frame.toMap().clear());
        assertThrows(UnsupportedOperationException.class, () -> object(frame.toMap(), "events").clear());
    }

    @Test void refusesClosedShapeBadDeclarationAndNonFiniteOrInconsistentNumbers() throws Exception {
        for (String malformed : new String[] {
                json().replace("\"presentation_only\":true", "\"presentation_only\":false"),
                json().replace("\"schema\":", "\"extra\":0,\"schema\":"),
                json().replace("\"schema\":", "\"schema\":null,\"schema\":"),
                json().replace("neurocraft.event_selection_display.v1", "neurocraft.event_selection_display.v2"),
                json().replace("0.0008", "NaN"), json().replace("0.0008", "1e999"),
                json().replace("0.0008", "1.1"),
                json().replace("\"left_count\":6", "\"left_count\":7"),
                json().replace("\"baseline_ready\":true", "\"baseline_ready\":false"),
                json().replace("\"score\":0.0005", "\"score\":0.1"),
                json().replace("\"food_age\":null", "\"food_age\":9"),
                json().replace("\"food_near\":true", "\"food_near\":false"),
                json().replace("1783000000000000007", "9223372036854775808"),
                json() + " ".repeat(EventDisplayFrame.MAX_BYTES)}) {
            assertThrows(ProtocolException.class, () -> EventDisplayFrame.parse(malformed));
        }
        var values = values();
        object(object(values, "events"), "food").put("last_game_tick", 6);
        assertThrows(ProtocolException.class, () -> parse(values));
    }

    @Test void originalIssuedTriggerCannotSatisfyCurrentAppliedLedger() throws Exception {
        var sample = frame();
        var state = new EventDisplayState(sample.identity());
        state.recordApplied(sample.trigger().command());
        assertEquals(EventDisplayState.Status.WAITING_FOR_APPLIED_COMMAND,
                state.view(snapshot(sample, 100), 101, 7).status());
        state.recordApplied(sample.command());
        assertEquals(EventDisplayState.Status.MATCHED, state.view(snapshot(sample, 100), 101, 7).status());
        var changed = values();
        object(changed, "issued_trigger").put("target_tick", 5);
        assertThrows(ProtocolException.class, () -> parse(changed));
    }

    @Test void everyCommandFieldMustMatchActualApplication() throws Exception {
        var expected = frame();
        for (String key : new String[] {"observation_seq", "observation_tick", "target_tick",
                "neural_step_start", "neural_step_end", "state_digest", "policy_hash", "program",
                "modulation", "parameters", "readout"}) {
            var raw = values();
            raw.put("issued_trigger", null);
            switch (key) {
                case "state_digest", "policy_hash" -> raw.put(key, "9".repeat(64));
                case "program" -> raw.put(key, "GROOM");
                case "modulation" -> object(raw, key).put("speed", 0.5);
                case "parameters" -> object(raw, key).put("turn_bias", -0.5);
                case "readout" -> object(raw, key).put("dna01_turn", 0.5);
                case "target_tick" -> raw.put(key, 6);
                case "neural_step_start" -> raw.put(key, 24);
                case "neural_step_end" -> raw.put(key, 31);
                default -> raw.put(key, 6);
            }
            var altered = parse(raw);
            var state = new EventDisplayState(expected.identity());
            state.recordApplied(expected.command());
            assertEquals(EventDisplayState.Status.ENVELOPE_MISMATCH,
                    state.view(snapshot(altered, 100), 101, 7).status(), key);
        }
        var window = values();
        window.put("issued_trigger", null);
        window.put("target_tick", 6); window.put("valid_through_tick", 6);
        assertFalse(parse(window).matches(expected.command()));
    }

    @Test void countersAndTriggerIdentityCannotSmuggleOldEpochIntoCurrentSample() throws Exception {
        var values = values();
        object(object(values, "issued_trigger"), "identity").put("session_epoch", 1);
        assertThrows(ProtocolException.class, () -> parse(values));
        var sample = frame();
        var foreign = EventDisplayFrame.parse(json().replace("event-display-fixture", "foreign"));
        var state = new EventDisplayState(sample.identity());
        assertFalse(state.recordApplied(foreign.command()));
        assertEquals(EventDisplayState.Status.IDENTITY_MISMATCH,
                state.view(snapshot(foreign, 100), 101, 7).status());
    }

    @Test void acceptedProvenanceAndSameCommandContentsCannotChange() throws Exception {
        var sample = frame();
        var state = new EventDisplayState(sample.identity());
        state.recordApplied(sample.command());
        assertEquals(EventDisplayState.Status.MATCHED, state.view(snapshot(sample, 100), 101, 7).status());
        var changed = values(); changed.put("command_sha256", "9".repeat(64));
        assertEquals(EventDisplayState.Status.ENVELOPE_MISMATCH,
                state.view(snapshot(parse(changed), 200), 201, 7).status());
        var next = values(); next.put("issued_trigger", null); next.put("command_seq", 6);
        next.put("binding_hash", "9".repeat(64));
        var nextFrame = parse(next); state.recordApplied(nextFrame.command());
        assertEquals(EventDisplayState.Status.PROVENANCE_CHANGED,
                state.view(snapshot(nextFrame, 200), 201, 7).status());
    }

    @Test void filePollingIsBoundedAndOldFileCannotRefreshOrBlockNewIdentity() throws Exception {
        var path = directory.resolve("event-selection-display.json");
        var clock = new AtomicLong(100);
        try (var reader = new EventDisplayReader(path, frame().identity(), clock::get, false)) {
            reader.pollOnce(); assertEquals(EventDisplayReader.Status.UNAVAILABLE, reader.snapshot().status());
            Files.writeString(path, json().replace("event-display-fixture", "foreign"));
            reader.pollOnce(); assertEquals(EventDisplayReader.Status.INVALID, reader.snapshot().status());
            Files.writeString(path, json()); reader.pollOnce();
            assertEquals(100, reader.snapshot().firstSeenNanos());
            clock.set(2_000_000_000L); reader.pollOnce();
            assertEquals(100, reader.snapshot().firstSeenNanos());
            Files.writeString(path, "x".repeat(EventDisplayFrame.MAX_BYTES + 1)); reader.pollOnce();
            assertEquals(EventDisplayReader.Status.INVALID, reader.snapshot().status());
            Files.write(path, new byte[] {(byte) 0xc0, (byte) 0x80}); reader.pollOnce();
            assertEquals(EventDisplayReader.Status.UNAVAILABLE, reader.snapshot().status());
        }
    }

    @Test void readAndObservationAgeExpireAndResetDropsEveryAppliedReference() throws Exception {
        var sample = frame(); var state = new EventDisplayState(sample.identity());
        state.recordApplied(sample.command()); state.view(snapshot(sample, 100), 100, 7);
        assertEquals(EventDisplayState.Status.STALE,
                state.view(snapshot(sample, 2_000_000_000L), 2_000_000_000L, 7).status());
        assertEquals(EventDisplayState.Status.STALE, state.view(snapshot(sample, 100), 101, 26).status());
        state.reset(); assertEquals(0, state.appliedLedgerSize());
        assertEquals(EventDisplayState.Status.WAITING_FOR_APPLIED_COMMAND,
                state.view(snapshot(sample, 100), 101, 7).status());
        for (int i = 0; i < 150; i++) {
            var raw = values(); raw.put("issued_trigger", null); raw.put("command_seq", i);
            state.recordApplied(parse(raw).command());
        }
        assertEquals(128, state.appliedLedgerSize());
    }

    @Test void clientReceiptClockExpiresEvenWithoutRenderingAndComparisonClearsEpoch() throws Exception {
        var sample = frame();
        var presentation = new EventDisplayPresentation(EventDisplayState.Status.MATCHED, 500, Optional.of(sample));
        assertEquals(presentation, EventDisplayPresentation.parse(presentation.encode()));
        var cache = new ComparisonLaneClientCache();
        cache.receiveEvents(presentation, 100);
        assertEquals(EventDisplayState.Status.STALE, cache.events(600_000_101).status());
        cache.receiveEvents(presentation, 100);
        cache.receiveLane(ComparisonLanePresentation.waiting(ComparisonLanePresentation.Status.SWITCHING,
                "no_input", sample.identity().sessionEpoch() + 1), 101);
        cache.receiveEvents(presentation, 102);
        assertTrue(cache.events(103).frame().isEmpty());
        assertEquals(EventDisplayState.Status.INVALID, EventDisplayPresentation.parseOrInvalid("{}").status());
        assertEquals(EventDisplayState.Status.DISABLED, EventDisplayPresentation.parseOrInvalid("").status());
    }
}
