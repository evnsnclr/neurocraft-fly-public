package dev.neurocraft.fly.control;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

final class BrainDisplayTest {
    @TempDir Path directory;

    @Test
    void strictFrameCarriesAllSixActualMeansAndExactCommandEnvelope() throws Exception {
        BrainDisplayFrame frame = frame(0);
        assertEquals(6, frame.populations().size());
        assertEquals(-0.25, frame.populations().get("lc4_left").mean());
        assertEquals(71, frame.populations().get("lc4_left").count());
        assertTrue(frame.matches(command(0)));
        assertFalse(frame.matches(command(1)));
        assertThrows(UnsupportedOperationException.class,
                () -> frame.populations().clear());
    }

    @Test
    void rejectsUnknownMissingDuplicateAndNonPresentationFields() throws Exception {
        String valid = json(0);
        for (String malformed : new String[] {
            valid.replace("\"presentation_only\":true", "\"presentation_only\":false"),
            valid.replace("\"schema\":", "\"extra\":0,\"schema\":"),
            valid.replace("\"schema\":", "\"command_seq\":0,\"schema\":"),
            valid.replace("neurocraft.brain_display.v1", "neurocraft.brain_display.v2"),
            valid.replace("ttmn_right", "unlabeled"),
            valid.replace("\"neural_step_end\":5", "\"neural_step_end\":0")
        }) {
            assertThrows(ProtocolException.class, () -> BrainDisplayFrame.parse(malformed), malformed);
        }
    }

    @Test
    void rejectsNonFiniteMeansAndUnboundedCountsClocksAndBytes() throws Exception {
        String valid = json(0);
        for (String value : new String[] {"NaN", "1e999", "1.01", "-1.01", "true", "\"0.1\""}) {
            assertThrows(ProtocolException.class,
                    () -> BrainDisplayFrame.parse(valid.replace("-0.25", value)), value);
        }
        for (String value : new String[] {"0", "166701", "1.5", "9223372036854775808"}) {
            assertThrows(ProtocolException.class,
                    () -> BrainDisplayFrame.parse(valid.replace("\"count\":71", "\"count\":" + value)));
        }
        assertThrows(ProtocolException.class,
                () -> BrainDisplayFrame.parse(valid + " ".repeat(BrainDisplayFrame.MAX_BYTES)));
        assertThrows(IllegalArgumentException.class,
                () -> new BrainDisplayFrame.Population(Double.NaN, 1));
    }

    @Test
    void mustMatchAnActuallyRecordedApplicationBeforeShowingAnyPopulation() throws Exception {
        BrainDisplayState state = new BrainDisplayState(identity());
        BrainDisplayReader.Snapshot snapshot = snapshot(frame(0), 100);
        assertEquals(BrainDisplayState.Status.WAITING_FOR_APPLIED_COMMAND,
                state.view(snapshot, 101, 102).status());
        assertTrue(state.view(snapshot, 101, 102).frame().isEmpty());
        assertTrue(state.recordApplied(command(0)));
        assertEquals(BrainDisplayState.Status.MATCHED, state.view(snapshot, 102, 102).status());
        assertEquals(frame(0), state.view(snapshot, 102, 102).frame().orElseThrow());
    }

    @Test
    void envelopeMismatchesDoNotBecomeDisplayData() throws Exception {
        for (String key : new String[] {
            "observation_tick", "neural_step_start", "neural_step_end", "state_digest"
        }) {
            Map<String, Object> raw = values(0);
            raw.put(key, key.equals("state_digest") ? "f".repeat(64)
                    : key.equals("neural_step_start") ? 1
                    : key.equals("observation_tick") ? 101 : 999);
            BrainDisplayState state = new BrainDisplayState(identity());
            state.recordApplied(command(0));
            BrainDisplayFrame altered = BrainDisplayFrame.parse(
                    StrictJson.encode(raw, BrainDisplayFrame.MAX_BYTES));
            assertEquals(BrainDisplayState.Status.ENVELOPE_MISMATCH,
                    state.view(snapshot(altered, 100), 101, 103).status(), key);
        }
        BrainDisplayState state = new BrainDisplayState(identity());
        BrainDisplayFrame foreign = BrainDisplayFrame.parse(json(0).replace("display-run", "foreign"));
        assertEquals(BrainDisplayState.Status.IDENTITY_MISMATCH,
                state.view(snapshot(foreign, 100), 101, 102).status());
        assertEquals(0, state.appliedLedgerSize());
    }

    @Test
    void repeatedSnapshotsCannotFreshenAnAcceptedFrame() throws Exception {
        BrainDisplayState state = new BrainDisplayState(identity());
        state.recordApplied(command(0));
        state.view(snapshot(frame(0), 100), 101, 102);
        long later = 100 + BrainDisplayState.MAX_AGE_NANOS + 1;
        BrainDisplayState.View result = state.view(snapshot(frame(0), later), later, 102);
        assertEquals(BrainDisplayState.Status.STALE, result.status());
        assertEquals(BrainDisplayState.MAX_AGE_NANOS + 1, result.ageNanos());
        assertTrue(result.frame().isEmpty());
    }

    @Test
    void provenanceCannotDriftAndResetClearsAllOldApplicationReferences() throws Exception {
        BrainDisplayState state = new BrainDisplayState(identity());
        state.recordApplied(command(0));
        state.view(snapshot(frame(0), 100), 100, 102);
        state.recordApplied(command(1));
        Map<String, Object> changedFields = values(1);
        changedFields.put("encoder_schema", "binding.v2");
        changedFields.put("binding_hash", "f".repeat(64));
        BrainDisplayFrame changed = BrainDisplayFrame.parse(
                StrictJson.encode(changedFields, BrainDisplayFrame.MAX_BYTES));
        assertEquals(BrainDisplayState.Status.PROVENANCE_CHANGED,
                state.view(snapshot(changed, 200), 201, 103).status());
        state.reset();
        assertEquals(0, state.appliedLedgerSize());
        assertEquals(BrainDisplayState.Status.WAITING_FOR_APPLIED_COMMAND,
                state.view(snapshot(changed, 200), 201, 103).status());
        state.recordApplied(command(1));
        assertEquals(BrainDisplayState.Status.MATCHED,
                state.view(snapshot(changed, 200), 201, 103).status());
    }

    @Test
    void appliedLedgerIsBoundedAndOlderDisplaysDoNotReplaceNewerOnes() throws Exception {
        BrainDisplayState state = new BrainDisplayState(identity());
        for (int sequence = 0; sequence <= BrainDisplayState.APPLIED_LEDGER_CAPACITY; sequence++) {
            state.recordApplied(command(sequence));
        }
        assertEquals(BrainDisplayState.APPLIED_LEDGER_CAPACITY, state.appliedLedgerSize());
        assertEquals(BrainDisplayState.Status.WAITING_FOR_APPLIED_COMMAND,
                state.view(snapshot(frame(0), 100), 101, 102).status());
        assertEquals(BrainDisplayState.Status.MATCHED,
                state.view(snapshot(frame(127), 100), 101, 230).status());
        assertEquals(BrainDisplayState.Status.SUPERSEDED,
                state.view(snapshot(frame(126), 100), 101, 230).status());
    }

    @Test
    void anOldAppliedFrameFirstReadLateIsAlreadyStaleAtZeroReadAge() throws Exception {
        BrainDisplayState state = new BrainDisplayState(identity());
        state.recordApplied(command(0));
        BrainDisplayReader.Snapshot firstRead = snapshot(frame(0), 500);
        BrainDisplayState.View result = state.view(firstRead, 500, 121);
        assertEquals(BrainDisplayState.Status.STALE, result.status());
        assertEquals(0, result.ageNanos());
        assertTrue(result.frame().isEmpty());
        assertEquals(BrainDisplayState.Status.STALE, state.view(firstRead, 500, 99).status());
        assertEquals(BrainDisplayState.Status.MATCHED, state.view(firstRead, 500, 120).status());
    }

    @Test
    void readerUsesOneSlotAndFirstReadTimeAcrossRepeatsAndMissingFile() throws Exception {
        Path path = directory.resolve("brain-display.json");
        AtomicLong clock = new AtomicLong(100);
        try (BrainDisplayReader reader = new BrainDisplayReader(path, identity(), clock::get, false)) {
            reader.pollOnce();
            assertEquals(BrainDisplayReader.Status.UNAVAILABLE, reader.snapshot().status());
            Files.writeString(path, json(0));
            reader.pollOnce();
            assertEquals(100, reader.snapshot().firstSeenNanos());
            clock.set(500);
            reader.pollOnce();
            assertEquals(100, reader.snapshot().firstSeenNanos());
            Files.delete(path);
            reader.pollOnce();
            assertTrue(reader.snapshot().frame().isEmpty());
            Files.writeString(path, json(0));
            reader.pollOnce();
            assertEquals(100, reader.snapshot().firstSeenNanos());
            Files.writeString(path, json(1));
            assertEquals(0, reader.snapshot().frame().orElseThrow().commandSequence()); // no I/O
            reader.pollOnce();
            assertEquals(1, reader.snapshot().frame().orElseThrow().commandSequence());
            assertEquals(500, reader.snapshot().firstSeenNanos());
        }
    }

    @Test
    void invalidOversizeAndChangedSameCommandFilesOnlyHideTheDisplay() throws Exception {
        Path path = directory.resolve("brain-display.json");
        BrainDisplayReader reader = new BrainDisplayReader(path, identity(), () -> 100L, false);
        try {
            Files.writeString(path, json(0));
            reader.pollOnce();
            Files.writeString(path, json(0).replace("-0.25", "0.5"));
            reader.pollOnce();
            assertEquals(BrainDisplayReader.Status.INVALID, reader.snapshot().status());
            Files.writeString(path, "x".repeat(BrainDisplayFrame.MAX_BYTES + 1));
            reader.pollOnce();
            assertEquals(BrainDisplayReader.Status.INVALID, reader.snapshot().status());
            Files.write(path, new byte[] {(byte) 0xff});
            reader.pollOnce();
            assertTrue(reader.snapshot().frame().isEmpty());
        } finally {
            reader.close();
        }
        reader.pollOnce();
        assertEquals(BrainDisplayReader.Status.CLOSED, reader.snapshot().status());
    }

    @Test
    void regressedOrForeignFilesCannotRestartTheReadersFreshnessClock() throws Exception {
        Path path = directory.resolve("brain-display.json");
        AtomicLong clock = new AtomicLong(100);
        try (BrainDisplayReader reader = new BrainDisplayReader(path, identity(), clock::get, false)) {
            Files.writeString(path, json(1));
            reader.pollOnce();
            clock.set(500);
            Files.writeString(path, json(0));
            reader.pollOnce();
            assertEquals(BrainDisplayReader.Status.INVALID, reader.snapshot().status());
            Files.writeString(path, json(2).replace("display-run", "foreign"));
            reader.pollOnce();
            assertEquals(BrainDisplayReader.Status.INVALID, reader.snapshot().status());
            Files.writeString(path, json(1));
            reader.pollOnce();
            assertEquals(100, reader.snapshot().firstSeenNanos());
        }
    }

    @Test
    void anOldEpochFileAtStartupCannotPoisonTheConfiguredRunsReader() throws Exception {
        Path path = directory.resolve("reused-brain-display.json");
        AtomicLong clock = new AtomicLong(100);
        Map<String, Object> previousRun = values(100);
        Map<String, Object> oldIdentity = new LinkedHashMap<>(
                BrainDisplayFrame.object(previousRun, "identity"));
        oldIdentity.put("session_epoch", 0);
        previousRun.put("identity", oldIdentity);
        String oldJson = StrictJson.encode(previousRun, BrainDisplayFrame.MAX_BYTES);
        assertEquals(0, BrainDisplayFrame.parse(oldJson).identity().sessionEpoch());
        try (BrainDisplayReader reader = new BrainDisplayReader(path, identity(), clock::get, false)) {
            Files.writeString(path, oldJson);
            reader.pollOnce();
            assertEquals(BrainDisplayReader.Status.INVALID, reader.snapshot().status());
            assertTrue(reader.snapshot().frame().isEmpty());
            assertEquals(0, reader.snapshot().firstSeenNanos());

            // A lower sequence in the expected epoch must recover on the next poll.
            clock.set(500);
            Files.writeString(path, json(0));
            reader.pollOnce();
            assertEquals(BrainDisplayReader.Status.AVAILABLE, reader.snapshot().status());
            assertEquals(frame(0), reader.snapshot().frame().orElseThrow());
            assertEquals(500, reader.snapshot().firstSeenNanos());

            clock.set(900);
            Files.writeString(path, oldJson);
            reader.pollOnce();
            assertEquals(BrainDisplayReader.Status.INVALID, reader.snapshot().status());
            Files.writeString(path, json(0));
            reader.pollOnce();
            assertEquals(BrainDisplayReader.Status.AVAILABLE, reader.snapshot().status());
            assertEquals(500, reader.snapshot().firstSeenNanos());
        }
    }

    @Test
    void synchronizedPresentationIsOneCompleteValueWithServerElapsedAge() throws Exception {
        BrainDisplayPresentation display = BrainDisplayPresentation.fromView(
                new BrainDisplayState.View(BrainDisplayState.Status.MATCHED,
                        Optional.of(frame(0)), 123_456_789));
        assertEquals(123, display.ageMillis());
        assertEquals(display, BrainDisplayPresentation.parse(display.encode()));
        assertTrue(display.encode().contains("\"age_ms\":123"));
        assertFalse(display.encode().contains("nano"));
        BrainDisplayPresentation stale = BrainDisplayPresentation.fromView(
                new BrainDisplayState.View(BrainDisplayState.Status.STALE,
                        Optional.empty(), 2_000_000_000));
        assertEquals(stale, BrainDisplayPresentation.parse(stale.encode()));
        assertTrue(stale.frame().isEmpty());
    }

    @Test
    void malformedOrInconsistentSynchronizedDisplayIsHidden() throws Exception {
        BrainDisplayPresentation display = new BrainDisplayPresentation(
                BrainDisplayState.Status.MATCHED, 0, Optional.of(frame(0)));
        for (String invalid : new String[] {
            "{", "null", display.encode().replace("\"MATCHED\"", "\"STALE\""),
            display.encode().replace("\"age_ms\":0", "\"age_ms\":-1"),
            display.encode().replace("\"age_ms\":0", "\"age_ms\":0,\"age_ms\":0")
        }) {
            BrainDisplayPresentation result = BrainDisplayPresentation.parseOrInvalid(invalid);
            assertEquals(BrainDisplayState.Status.INVALID, result.status());
            assertTrue(result.frame().isEmpty());
        }
    }

    private static BrainDisplayReader.Snapshot snapshot(BrainDisplayFrame frame, long seen) {
        return new BrainDisplayReader.Snapshot(
                BrainDisplayReader.Status.AVAILABLE, Optional.of(frame), seen);
    }

    private static BrainDisplayFrame frame(long sequence) throws Exception {
        return BrainDisplayFrame.parse(json(sequence));
    }

    private static String json(long sequence) {
        return StrictJson.encode(values(sequence), BrainDisplayFrame.MAX_BYTES);
    }

    private static Map<String, Object> values(long sequence) {
        ControllerIdentity identity = identity();
        Map<String, Object> id = Map.of(
                "run_id", identity.runId(), "session_epoch", identity.sessionEpoch(),
                "entity_id", identity.entityId(), "graph_hash", identity.graphHash(),
                "adapter_hash", identity.adapterHash(), "dynamics_hash", identity.dynamicsHash(),
                "decoder_hash", identity.decoderHash(),
                "observation_schema_hash", identity.observationSchemaHash());
        Map<String, Object> populations = new LinkedHashMap<>();
        for (String name : BrainDisplayFrame.POPULATION_NAMES) {
            populations.put(name, Map.of("mean", -0.25, "count", 71));
        }
        Map<String, Object> values = new LinkedHashMap<>();
        values.put("schema", BrainDisplayFrame.SCHEMA);
        values.put("presentation_only", true);
        values.put("identity", id);
        values.put("command_seq", sequence);
        values.put("observation_tick", 100 + sequence);
        values.put("neural_step_start", 5 * sequence);
        values.put("neural_step_end", 5 * (sequence + 1));
        values.put("state_digest", "d".repeat(64));
        values.put("encoder_schema", "binding.v1");
        values.put("binding_hash", "e".repeat(64));
        values.put("populations", populations);
        return values;
    }

    private static ActionFrame command(long sequence) {
        return new ActionFrame(identity(), sequence, sequence, 100 + sequence, 102 + sequence,
                102 + sequence, 5 * sequence, 5 * (sequence + 1), "d".repeat(64),
                RawPrimitiveAction.ZERO, false);
    }

    private static ControllerIdentity identity() {
        return new ControllerIdentity("display-run", 1, "display-fly", "a".repeat(64),
                "b".repeat(64), "c".repeat(64), "d".repeat(64), "e".repeat(64));
    }
}
