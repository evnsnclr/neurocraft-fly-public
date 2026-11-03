package dev.neurocraft.fly.entity;

import dev.neurocraft.fly.control.EventSample;

/** Server-thread-only, entity-lifetime counters. Neural epoch/reset operations never clear them. */
public final class EventAccumulator {
    private EventSample current;
    private long lastFoodSampleTick = -1;

    public EventAccumulator() { this(EventSample.EMPTY); }

    /** Package-local initial state also allows the signed-int64 boundary to be tested directly. */
    EventAccumulator(EventSample initial) { this(initial, -1); }

    EventAccumulator(EventSample initial, long lastFoodTick) {
        current = java.util.Objects.requireNonNull(initial);
        lastFoodSampleTick = lastFoodTick;
    }

    long lastFoodSampleTick() { return lastFoodSampleTick; }

    public EventSample snapshot() { return current; }

    /** The first authoritative sample on a tick wins; persistent food-near counts only its rising edge. */
    public EventSample sampleFood(boolean near, boolean contact, long gameTick) {
        if (contact && !near) { throw new IllegalArgumentException("food contact requires food near"); }
        if (gameTick < 0 || gameTick < lastFoodSampleTick) {
            throw new IllegalArgumentException("food sampling game ticks must be monotonic");
        }
        if (gameTick == lastFoodSampleTick) { return current; }
        lastFoodSampleTick = gameTick;
        EventSample.Counter food = current.food();
        if (near && !current.foodNear() && canIncrement(food, gameTick)) {
            food = increment(food, gameTick);
        }
        current = new EventSample(contact, near, food, current.brush(), current.attackAttempt());
        return current;
    }

    /** Overflow refuses this increment, without wrapping, mutating the timestamp, or throwing. */
    public boolean recordBrush(long gameTick) {
        if (!canIncrement(current.brush(), gameTick)) { return false; }
        current = new EventSample(current.foodContact(), current.foodNear(), current.food(),
                increment(current.brush(), gameTick), current.attackAttempt());
        return true;
    }

    public boolean recordAttackAttempt(long gameTick) {
        if (!canIncrement(current.attackAttempt(), gameTick)) { return false; }
        current = new EventSample(current.foodContact(), current.foodNear(), current.food(), current.brush(),
                increment(current.attackAttempt(), gameTick));
        return true;
    }

    private static boolean canIncrement(EventSample.Counter counter, long gameTick) {
        return gameTick >= 0 && counter.count() < Long.MAX_VALUE
                && (counter.lastGameTick() == null || gameTick >= counter.lastGameTick());
    }

    private static EventSample.Counter increment(EventSample.Counter counter, long gameTick) {
        return new EventSample.Counter(counter.count() + 1, gameTick);
    }
}
