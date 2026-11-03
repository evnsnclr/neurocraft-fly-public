package dev.neurocraft.fly.control;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

final class CoalescingObservationSlotTest {
    private static final ControllerIdentity IDENTITY = new ControllerIdentity(
            "run-1",
            7,
            "fly-01",
            "1".repeat(64),
            "2".repeat(64),
            "3".repeat(64),
            "4".repeat(64),
            "5".repeat(64));

    @Test
    void forcedOverwritesAreCumulativeAndBoundToTheSerializedLatestSnapshot()
            throws Exception {
        CoalescingObservationSlot slot = new CoalescingObservationSlot();

        assertTrue(slot.offer(snapshot(100)));
        assertTrue(slot.offer(snapshot(101)));
        assertTrue(slot.offer(snapshot(102)));

        CoalescingObservationSlot.PendingObservation pending = slot.pollLatest();
        assertNotNull(pending);
        assertEquals(102L, pending.snapshot().gameTick());
        assertEquals(2L, pending.bridgeObservationsCoalesced());
        assertFalse(slot.hasPending());

        String encoded = ControlProtocol.encodeObservation(
                IDENTITY,
                pending.snapshot(),
                0,
                123,
                pending.bridgeObservationsCoalesced());
        Map<String, Object> message =
                StrictJson.parseObject(encoded, ControlProtocol.MAX_MESSAGE_BYTES);
        assertEquals(102L, ((Number) message.get("game_tick")).longValue());
        assertEquals(
                2L,
                ((Number) message.get("bridge_observations_coalesced")).longValue());

        assertTrue(slot.offer(snapshot(103)));
        CoalescingObservationSlot.PendingObservation next = slot.pollLatest();
        assertNotNull(next);
        assertEquals(103L, next.snapshot().gameTick());
        assertEquals(2L, next.bridgeObservationsCoalesced());
        assertNull(slot.pollLatest());
    }

    private static PhysicalSensorSnapshot snapshot(long gameTick) {
        return new PhysicalSensorSnapshot(
                gameTick,
                "fly-01",
                1,
                2,
                3,
                0.1,
                -0.2,
                0.3,
                45,
                -10,
                true,
                false,
                false,
                false,
                false,
                6,
                15,
                6,
                4,
                3,
                2,
                1,
                5,
                0.5);
    }
}
