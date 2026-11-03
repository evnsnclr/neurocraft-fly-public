package dev.neurocraft.fly.control;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

final class EventObservationTest {
    @Test
    void v3ObservationMatchesTheSharedPythonGolden() throws Exception {
        var sample = new PhysicalSensorSnapshot(100, "fly-01", 1, 64, -2, 0.005, 0, -0.01,
                Math.toDegrees(0.25), Math.nextUp(Math.toDegrees(-0.1)),
                false, false, false, false, false, 12, 10, 8, 8, 6, 5, 8, 2.5, 3,
                List.of(0, 1, 2, 0, 1, 2), new EventSample(false, false,
                        new EventSample.Counter(1, 90L), EventSample.Counter.ZERO, new EventSample.Counter(3, 99L)));
        var identity = new ControllerIdentity("demo-001", 1, "fly-01", "1".repeat(64),
                "2".repeat(64), "3".repeat(64), "4".repeat(64), ControlProtocol.OBSERVATION_SCHEMA_V3_SHA256);
        String actual = ControlProtocol.encodeObservation(identity, sample, 0, 123456789, 0);
        String expected = Files.readString(Path.of("..", "packages", "protocol", "golden", "observation_v3.json"));
        assertEquals(StrictJson.parseObject(expected, ControlProtocol.MAX_MESSAGE_BYTES),
                StrictJson.parseObject(actual, ControlProtocol.MAX_MESSAGE_BYTES));
    }

    @Test
    void v3RequiresBothMeasuredKindsAndEventsWithTheExactNestedShape() {
        var zero = snapshot(10).withEvents(EventSample.EMPTY);
        String frame = encode(ControlProtocol.OBSERVATION_SCHEMA_V3_SHA256, zero);
        assertTrue(frame.contains("\"sensor_schema\":\"neurocraft.sensors.v3\""));
        assertTrue(frame.contains("\"events\":{\"food_contact\":false,\"food_near\":false,\"food\":{\"count\":0,\"last_game_tick\":null},"
                + "\"brush\":{\"count\":0,\"last_game_tick\":null},\"attack_attempt\":{\"count\":0,\"last_game_tick\":null}}"));
        assertThrows(IllegalArgumentException.class,
                () -> encode(ControlProtocol.OBSERVATION_SCHEMA_V3_SHA256, snapshot(10)));
        var noKinds = new PhysicalSensorSnapshot(10, "fly", 0, 2, 0, 0, 0, 0, 0, 0,
                false, false, false, false, false, 15, 15, 6, 6, 6, 6, 6, 6, 6);
        assertThrows(IllegalArgumentException.class,
                () -> encode(ControlProtocol.OBSERVATION_SCHEMA_V3_SHA256, noKinds.withEvents(EventSample.EMPTY)));
    }

    @Test
    void addingEventsCannotChangeHistoricalV1OrV2ObservationBytes() {
        var original = snapshot(10);
        var augmented = original.withEvents(new EventSample(true, true,
                new EventSample.Counter(1, 10L), new EventSample.Counter(3, 8L), EventSample.Counter.ZERO));
        for (String hash : List.of("5".repeat(64), ControlProtocol.OBSERVATION_SCHEMA_V2_SHA256)) {
            assertEquals(encode(hash, original), encode(hash, augmented));
            assertFalse(encode(hash, augmented).contains("\"events\":"));
        }
    }

    @Test
    void int64CountersStayExactAndFutureEventTimestampsRefuseBeforeEncoding() {
        var sample = snapshot(10).withEvents(new EventSample(false, false,
                EventSample.Counter.ZERO, new EventSample.Counter(Long.MAX_VALUE, 10L), EventSample.Counter.ZERO));
        assertTrue(encode(ControlProtocol.OBSERVATION_SCHEMA_V3_SHA256, sample)
                .contains("\"count\":9223372036854775807,\"last_game_tick\":10"));
        assertThrows(IllegalArgumentException.class, () -> snapshot(10).withEvents(new EventSample(false, false,
                new EventSample.Counter(1, 11L), EventSample.Counter.ZERO, EventSample.Counter.ZERO)));
    }

    @Test
    void coalescingPreservesCumulativeEventsWhenTheirOriginalTickIsOverwritten() {
        var slot = new CoalescingObservationSlot();
        var event = new EventSample(false, false, EventSample.Counter.ZERO,
                new EventSample.Counter(1, 10L), EventSample.Counter.ZERO);
        assertTrue(slot.offer(snapshot(10).withEvents(event)));
        assertTrue(slot.offer(snapshot(11).withEvents(event)));
        assertTrue(slot.offer(snapshot(12).withEvents(event)));
        var latest = slot.pollLatest();
        assertEquals(12, latest.snapshot().gameTick());
        assertEquals(2, latest.bridgeObservationsCoalesced());
        assertEquals(new EventSample.Counter(1, 10L), latest.snapshot().events().brush());
        assertNull(slot.pollLatest());
    }

    private static String encode(String hash, PhysicalSensorSnapshot snapshot) {
        return ControlProtocol.encodeObservation(new ControllerIdentity("events", 0, "fly",
                "1".repeat(64), "2".repeat(64), "3".repeat(64), "4".repeat(64), hash), snapshot, 0, 0, 0);
    }

    private static PhysicalSensorSnapshot snapshot(long tick) {
        return new PhysicalSensorSnapshot(tick, "fly", 0, 2, 0, 0, 0, 0, 0, 0,
                false, false, false, false, false, 15, 15, 6, 6, 6, 6, 6, 6, 6,
                List.of(0, 0, 0, 0, 0, 0));
    }
}
