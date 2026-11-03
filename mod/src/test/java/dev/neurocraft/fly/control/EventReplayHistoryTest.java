package dev.neurocraft.fly.control;

import static org.junit.jupiter.api.Assertions.*;

import dev.neurocraft.fly.body.SelectionSource;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class EventReplayHistoryTest {
    private static EventDisplayFrame frame(int offset) throws Exception {
        var values = StrictJson.parseObject(Files.readString(Path.of(
                "src/gametest/resources/neurocraft/interaction-display-v2.json")), EventDisplayFrame.MAX_BYTES);
        for (String key : new String[] {"command_seq", "observation_seq", "observation_tick", "target_tick", "valid_through_tick"}) {
            values.put(key, ((Number) values.get(key)).longValue() + offset);
        }
        for (String key : new String[] {"neural_step_start", "neural_step_end"}) {
            values.put(key, ((Number) values.get(key)).longValue() + offset * 5L);
        }
        return EventDisplayFrame.parse(StrictJson.encode(values, EventDisplayFrame.MAX_BYTES));
    }

    private static ComparisonLanePresentation lane(EventDisplayFrame frame) {
        return new ComparisonLanePresentation(ComparisonLanePresentation.Status.ACTIVE, "original",
                frame.identity().sessionEpoch(), Optional.of(frame.identity()), "original", "faithful", null, null);
    }

    private static EventDisplayPresentation matched(EventDisplayFrame frame) {
        return new EventDisplayPresentation(EventDisplayState.Status.MATCHED, 0, Optional.of(frame));
    }

    @Test void storesOnlyFreshMatchedReadoutSamplesAndDeduplicatesCommands() throws Exception {
        var sample = frame(0);
        var history = new EventReplayHistory();
        for (var status : EventDisplayState.Status.values()) {
            if (status == EventDisplayState.Status.MATCHED) { continue; }
            assertFalse(history.record(new EventDisplayPresentation(status, 0, Optional.empty()),
                    lane(sample), ControllerFault.ACTIVE, SelectionSource.READOUT));
        }
        for (var state : ControllerFault.values()) {
            if (state == ControllerFault.ACTIVE) { continue; }
            assertFalse(history.record(matched(sample), lane(sample), state, SelectionSource.READOUT));
        }
        assertFalse(history.record(matched(sample), lane(sample), ControllerFault.ACTIVE, SelectionSource.OPERATOR));
        assertFalse(history.record(new EventDisplayPresentation(EventDisplayState.Status.MATCHED, 1001, Optional.of(sample)),
                lane(sample), ControllerFault.ACTIVE, SelectionSource.READOUT));
        assertFalse(history.record(matched(sample), ComparisonLanePresentation.waiting(
                ComparisonLanePresentation.Status.SWITCHING, "no_input", 1), ControllerFault.ACTIVE, SelectionSource.READOUT));
        assertTrue(history.record(matched(sample), lane(sample), ControllerFault.ACTIVE, SelectionSource.READOUT));
        assertFalse(history.record(matched(sample), lane(sample), ControllerFault.ACTIVE, SelectionSource.READOUT));
        assertEquals(1, history.snapshot().size());
        assertEquals("light", history.snapshot().getFirst().inspectedSource());
        assertTrue(history.snapshot().getFirst().selectedSource());
        assertEquals(sample.command(), history.snapshot().getFirst().frame().command());
    }

    @Test void historyIsBoundedImmutableAndRetainsHistoricalIdentity() throws Exception {
        var history = new EventReplayHistory();
        for (int index = 0; index < EventReplayHistory.MAX_SAMPLES + 4; index++) {
            var sample = frame(index);
            assertTrue(history.record(matched(sample), lane(sample), ControllerFault.ACTIVE, SelectionSource.READOUT));
        }
        var snapshot = history.snapshot();
        assertEquals(EventReplayHistory.MAX_SAMPLES, snapshot.size());
        assertEquals(30, snapshot.getFirst().frame().commandSequence());
        assertEquals("original", snapshot.getFirst().lane());
        assertThrows(UnsupportedOperationException.class, snapshot::clear);
        assertFalse(history.record(matched(frame(0)), lane(frame(0)), ControllerFault.ACTIVE, SelectionSource.READOUT));
        history.clear();
        assertTrue(history.snapshot().isEmpty());
        assertEquals(EventReplayHistory.MAX_SAMPLES, snapshot.size());
    }

    @Test void eligibleContextWithoutSelectedSourceDoesNotBecomeAChosenProgram() throws Exception {
        var sample = frame(0);
        var details = sample.interaction();
        var unselected = new EventDisplayFrame(sample.command(), sample.encoderHash(), sample.bindingHash(),
                sample.mn9(), sample.antenna(), "no_eligible_response", null, sample.payload(),
                new EventDisplayFrame.InteractionDetails(null, details.scores(), details.rawTtmn(),
                        details.dna01Left(), details.dna01Right(), details.rayHitKinds(), details.rayActorKinds(), details.localLight()));
        var captured = new EventReplayHistory.Sample(unselected, "no_input", "7");
        assertEquals("light", captured.inspectedSource());
        assertFalse(captured.selectedSource());
        assertEquals("Brush contact", EventReplayHistory.sourceLabel("brush"));
        assertTrue(EventReplayHistory.sourceLabel("attack").contains("tactile proxy"));
    }

    @Test void activeLaneMustMatchTheRecordedIdentity() throws Exception {
        var sample = frame(0);
        var id = sample.identity();
        var other = new ControllerIdentity(id.runId(), id.sessionEpoch() + 1, id.entityId(), id.graphHash(),
                id.adapterHash(), id.dynamicsHash(), id.decoderHash(), id.observationSchemaHash());
        var lane = new ComparisonLanePresentation(ComparisonLanePresentation.Status.ACTIVE, "original",
                other.sessionEpoch(), Optional.of(other), "original", "faithful", null, null);
        assertFalse(new EventReplayHistory().record(matched(sample), lane, ControllerFault.ACTIVE, SelectionSource.READOUT));
    }
}
