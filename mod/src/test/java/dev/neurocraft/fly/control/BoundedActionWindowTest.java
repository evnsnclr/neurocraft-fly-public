package dev.neurocraft.fly.control;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

final class BoundedActionWindowTest {
    private static final ControllerIdentity IDENTITY = new ControllerIdentity(
            "run-1",
            7,
            "fly-01",
            "1".repeat(64),
            "2".repeat(64),
            "3".repeat(64),
            "4".repeat(64),
            "5".repeat(64));

    /**
     * The capacity is the action budget plus two -- every frame that can arrive while a staged
     * RESET blocks polling -- so the pin is written against the constant rather than against the
     * number it currently has.
     */
    @Test
    void neverAcceptsMoreThanItsCapacityAndPreservesOrder() {
        BoundedActionWindow window = new BoundedActionWindow();

        assertEquals(CommandValidator.ACTION_TARGET_OFFSET_TICKS + 2, BoundedActionWindow.CAPACITY);
        for (int index = 0; index < BoundedActionWindow.CAPACITY; index++) {
            assertTrue(window.offer(frame(index)));
        }
        assertFalse(window.offer(frame(BoundedActionWindow.CAPACITY)));
        assertEquals(BoundedActionWindow.CAPACITY, window.size());
        for (int index = 0; index < BoundedActionWindow.CAPACITY; index++) {
            assertEquals(frame(index), window.poll().orElseThrow());
        }
        assertTrue(window.poll().isEmpty());
    }

    /**
     * The staged-RESET arrival window, written out as the ticks it actually spans.
     *
     * <p>An operator RESET issued at tick 100 runs after that tick's entity tick, so the last
     * poll before the block is at 100 and the next is at the accepted effective tick -- at worst
     * 103, one action budget after the observation at 101 the effect materialized against.
     * Four frames can land in between: the answer to observation 99, still in flight at the poll
     * at 100 under the lane's measured 43-75 ms latencies, and the answers to observations 100,
     * 101 and 102. All four must be held, because an offer this window refuses latches
     * QUEUE_OVERFLOW and kills the session on an operator reset. The service cannot produce a
     * fifth: it answers at most one observation per tick.
     */
    @Test
    void holdsEveryFrameThatCanArriveWhileAStagedResetBlocksPolling() {
        BoundedActionWindow window = new BoundedActionWindow();

        // In flight at the last poll before the block (observation 99, answered late).
        assertTrue(window.offer(budgetFrame(0, 99)), "the in-flight frame was refused");
        // One answer per blocked tick: observations 100, 101 and 102.
        assertTrue(window.offer(budgetFrame(1, 100)), "the answer to observation 100 was refused");
        assertTrue(window.offer(budgetFrame(2, 101)), "the answer to observation 101 was refused");
        assertTrue(window.offer(budgetFrame(3, 102)), "the answer to observation 102 was refused");
        assertEquals(4, window.size());
        assertEquals(BoundedActionWindow.CAPACITY, window.size());

        // The reset resolves and the server thread polls them all in one pass.
        for (long observation = 99; observation <= 102; observation++) {
            CommandFrame polled = window.poll().orElseThrow();
            assertEquals(observation, polled.observationTick());
            assertEquals(
                    observation + CommandValidator.ACTION_TARGET_OFFSET_TICKS,
                    polled.targetTick());
        }
        assertTrue(window.poll().isEmpty());
    }

    private static ActionFrame frame(long sequence) {
        return new ActionFrame(
                IDENTITY,
                sequence,
                sequence,
                100 + sequence,
                101 + sequence,
                102 + sequence,
                sequence * 5,
                sequence * 5 + 5,
                "6".repeat(64),
                RawPrimitiveAction.ZERO,
                true);
    }

    /** A frame as the service emits it: target = observation + the declared action budget. */
    private static ActionFrame budgetFrame(long sequence, long observationTick) {
        long target = observationTick + CommandValidator.ACTION_TARGET_OFFSET_TICKS;
        return new ActionFrame(
                IDENTITY,
                sequence,
                sequence,
                observationTick,
                target,
                target,
                sequence * 5,
                sequence * 5 + 5,
                "6".repeat(64),
                RawPrimitiveAction.ZERO,
                true);
    }
}
