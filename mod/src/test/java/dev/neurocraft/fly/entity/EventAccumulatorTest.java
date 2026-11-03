package dev.neurocraft.fly.entity;

import static org.junit.jupiter.api.Assertions.*;

import dev.neurocraft.fly.control.EventSample;
import org.junit.jupiter.api.Test;

final class EventAccumulatorTest {
    @Test
    void nearbyFoodCountsBeforeContactAndLandingDoesNotCreateAnotherFoodEdge() {
        var events = new EventAccumulator();
        var near = events.sampleFood(true, false, 10);
        assertTrue(near.foodNear());
        assertFalse(near.foodContact());
        assertEquals(new EventSample.Counter(1, 10L), near.food());
        var contact = events.sampleFood(true, true, 11);
        assertTrue(contact.foodContact());
        assertEquals(near.food(), contact.food());
        assertThrows(IllegalArgumentException.class, () -> events.sampleFood(false, true, 12));
        assertThrows(IllegalArgumentException.class, () -> new EventSample(true, false,
                EventSample.Counter.ZERO, EventSample.Counter.ZERO, EventSample.Counter.ZERO));
    }

    @Test
    void foodCountsEdgesOncePerAuthoritativeTickAndSamplingDoesNotConsume() {
        var events = new EventAccumulator();
        assertEquals(EventSample.EMPTY, events.snapshot());
        var first = events.sampleFood(true, true, 10);
        assertEquals(new EventSample.Counter(1, 10L), first.food());
        assertSame(first, events.sampleFood(false, false, 10));
        assertSame(first, events.snapshot());
        events.sampleFood(true, true, 11);
        events.sampleFood(false, false, 12);
        var second = events.sampleFood(true, true, 13);
        assertEquals(new EventSample.Counter(2, 13L), second.food());
        assertEquals(new EventSample.Counter(1, 10L), first.food());
        assertThrows(IllegalArgumentException.class, () -> events.sampleFood(false, false, 12));
    }

    @Test
    void independentCountersIncludeSeveralInteractionsOnTheSameTick() {
        var events = new EventAccumulator();
        events.recordBrush(10);
        events.recordBrush(10);
        events.recordAttackAttempt(11);
        events.sampleFood(true, true, 11);
        var sample = events.snapshot();
        assertEquals(new EventSample.Counter(2, 10L), sample.brush());
        assertEquals(new EventSample.Counter(1, 11L), sample.attackAttempt());
        assertEquals(new EventSample.Counter(1, 11L), sample.food());
        assertFalse(events.recordBrush(9));
        assertFalse(events.recordAttackAttempt(-1));
        assertEquals(sample, events.snapshot());
    }

    @Test
    void everyOverflowRefusesOnlyTheIncrementWithoutWrappingOrChangingTimestamp() {
        var max = new EventSample.Counter(Long.MAX_VALUE, 4L);
        var events = new EventAccumulator(new EventSample(false, false, max, max, max));
        assertFalse(events.recordBrush(5));
        assertFalse(events.recordAttackAttempt(5));
        var sample = events.sampleFood(true, true, 5);
        assertTrue(sample.foodContact());
        assertEquals(max, sample.food());
        assertEquals(max, sample.brush());
        assertEquals(max, sample.attackAttempt());
    }

    @Test
    void counterAndObservationTimestampBoundariesAreStrict() {
        assertThrows(IllegalArgumentException.class, () -> new EventSample.Counter(-1, 0L));
        assertThrows(IllegalArgumentException.class, () -> new EventSample.Counter(0, 0L));
        assertThrows(IllegalArgumentException.class, () -> new EventSample.Counter(1, null));
        assertThrows(IllegalArgumentException.class, () -> new EventSample.Counter(1, -1L));
        var atZero = new EventSample(false, false, new EventSample.Counter(1, 0L),
                EventSample.Counter.ZERO, EventSample.Counter.ZERO);
        assertDoesNotThrow(() -> atZero.validateAt(0));
        var future = new EventSample(false, false, EventSample.Counter.ZERO,
                new EventSample.Counter(1, 10L), EventSample.Counter.ZERO);
        assertThrows(IllegalArgumentException.class, () -> future.validateAt(9));
    }
}
