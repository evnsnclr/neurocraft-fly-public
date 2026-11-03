package dev.neurocraft.fly.control;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

final class ActorObservationTest {
    private static final List<Integer> HITS = List.of(0, 1, 2, 2, 2, 0);
    private static final List<Integer> ACTORS = List.of(0, 0, 1, 2, 3, 0);

    @Test
    void v4MatchesTheSharedPythonGolden() throws Exception {
        var sample = new PhysicalSensorSnapshot(100, "fly-01", 1, 64, -2, 0.005, 0, -0.01,
                Math.toDegrees(0.25), Math.nextUp(Math.toDegrees(-0.1)),
                false, false, false, false, false, 12, 10, 8, 8, 6, 5, 8, 2.5, 3,
                HITS, ACTORS, new EventSample(false, false,
                        new EventSample.Counter(1, 90L), EventSample.Counter.ZERO, new EventSample.Counter(3, 99L)));
        var identity = new ControllerIdentity("demo-001", 1, "fly-01", "1".repeat(64),
                "2".repeat(64), "3".repeat(64), "4".repeat(64), ControlProtocol.OBSERVATION_SCHEMA_V4_SHA256);
        String actual = ControlProtocol.encodeObservation(identity, sample, 0, 123456789, 0);
        String expected = Files.readString(Path.of("..", "packages", "protocol", "golden", "observation_v4.json"));
        assertEquals(StrictJson.parseObject(expected, ControlProtocol.MAX_MESSAGE_BYTES),
                StrictJson.parseObject(actual, ControlProtocol.MAX_MESSAGE_BYTES));
    }

    @Test
    void explicitActorMetadataCannotChangeHistoricalV1V2OrV3Bytes() {
        var generic = legacySnapshot();
        var classified = snapshot(HITS, ACTORS, EventSample.EMPTY);
        assertNull(generic.rayActorKinds());
        assertThrows(IllegalArgumentException.class,
                () -> encode(ControlProtocol.OBSERVATION_SCHEMA_V4_SHA256, generic));
        for (String hash : List.of("5".repeat(64), ControlProtocol.OBSERVATION_SCHEMA_V2_SHA256,
                ControlProtocol.OBSERVATION_SCHEMA_V3_SHA256)) {
            assertEquals(encode(hash, generic), encode(hash, classified));
            assertFalse(encode(hash, classified).contains("ray_actor_kinds"));
            assertFalse(encode(hash, classified).contains("sensors.v4"));
        }
        String v4 = encode(ControlProtocol.OBSERVATION_SCHEMA_V4_SHA256, classified);
        assertTrue(v4.contains("\"sensor_schema\":\"neurocraft.sensors.v4\""));
        assertTrue(v4.contains("\"ray_actor_kinds\":[0,0,1,2,3,0]"));
        assertTrue(v4.contains("\"local_light\":15"));
    }

    @Test
    void actorMetadataIsImmutableValidatedAndPreservedWithEvents() {
        var mutable = new ArrayList<>(ACTORS);
        var sample = snapshot(HITS, mutable, EventSample.EMPTY);
        mutable.set(2, 3);
        assertEquals(ACTORS, sample.rayActorKinds());
        assertEquals(ACTORS, sample.withEvents(EventSample.EMPTY).rayActorKinds());
        assertThrows(UnsupportedOperationException.class, () -> sample.rayActorKinds().set(2, 3));
        assertNull(snapshot(HITS, null, EventSample.EMPTY).rayActorKinds());
        assertThrows(IllegalArgumentException.class, () -> snapshot(null, ACTORS, EventSample.EMPTY));
        assertThrows(IllegalArgumentException.class, () -> snapshot(HITS, List.of(0, 0, 1), EventSample.EMPTY));
        for (List<Integer> invalid : List.of(List.of(1, 0, 1, 2, 3, 0), List.of(0, 1, 1, 2, 3, 0),
                List.of(0, 0, 0, 2, 3, 0), List.of(0, 0, 4, 2, 3, 0), List.of(0, 0, -1, 2, 3, 0))) {
            assertThrows(IllegalArgumentException.class, () -> snapshot(HITS, invalid, EventSample.EMPTY));
        }
    }

    @Test
    void v4RefusesMissingKindsOrEventsBeforeSerialization() {
        assertThrows(IllegalArgumentException.class,
                () -> encode(ControlProtocol.OBSERVATION_SCHEMA_V4_SHA256, snapshot(null, null, EventSample.EMPTY)));
        assertThrows(IllegalArgumentException.class,
                () -> encode(ControlProtocol.OBSERVATION_SCHEMA_V4_SHA256, snapshot(HITS, ACTORS, null)));
    }

    private static PhysicalSensorSnapshot snapshot(List<Integer> hits, List<Integer> actors, EventSample events) {
        return new PhysicalSensorSnapshot(10, "fly", 0, 2, 0, 0, 0, 0, 0, 0,
                false, false, false, false, false, 7, 15, 6, 6, 6, 6, 6, 6, 6, hits, actors, events);
    }

    private static PhysicalSensorSnapshot legacySnapshot() {
        return new PhysicalSensorSnapshot(10, "fly", 0, 2, 0, 0, 0, 0, 0, 0,
                false, false, false, false, false, 7, 15, 6, 6, 6, 6, 6, 6, 6, HITS, EventSample.EMPTY);
    }

    private static String encode(String hash, PhysicalSensorSnapshot sample) {
        return ControlProtocol.encodeObservation(new ControllerIdentity("actors", 0, "fly",
                "1".repeat(64), "2".repeat(64), "3".repeat(64), "4".repeat(64), hash), sample, 0, 0, 0);
    }
}
