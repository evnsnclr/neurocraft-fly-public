package dev.neurocraft.fly.control;

import java.util.Objects;

/** Immutable raw Minecraft interaction evidence; these counters are not damage or neural activity. */
public record EventSample(boolean foodContact, boolean foodNear, Counter food, Counter brush, Counter attackAttempt) {
    public static final EventSample EMPTY = new EventSample(false, false, Counter.ZERO, Counter.ZERO, Counter.ZERO);

    public EventSample {
        if (foodContact && !foodNear) {
            throw new IllegalArgumentException("food contact requires food near");
        }
        Objects.requireNonNull(food, "food");
        Objects.requireNonNull(brush, "brush");
        Objects.requireNonNull(attackAttempt, "attackAttempt");
    }

    public void validateAt(long gameTick) {
        for (Counter counter : new Counter[] {food, brush, attackAttempt}) {
            if (counter.lastGameTick() != null && counter.lastGameTick() > gameTick) {
                throw new IllegalArgumentException("event timestamp exceeds observation game tick");
            }
        }
    }

    public record Counter(long count, Long lastGameTick) {
        public static final Counter ZERO = new Counter(0, null);

        public Counter {
            if (count < 0 || (count == 0) != (lastGameTick == null)
                    || (lastGameTick != null && lastGameTick < 0)) {
                throw new IllegalArgumentException("event count and last game tick are inconsistent");
            }
        }
    }
}
