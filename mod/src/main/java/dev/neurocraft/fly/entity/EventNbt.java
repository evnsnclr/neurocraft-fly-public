package dev.neurocraft.fly.entity;

import dev.neurocraft.fly.control.EventSample;
import java.util.Set;
import net.minecraft.nbt.ByteTag;
import net.minecraft.nbt.CompoundTag;
import net.minecraft.nbt.LongTag;
import net.minecraft.nbt.StringTag;
import net.minecraft.nbt.Tag;

/** Closed, fixed-depth NBT payload. Unknown keys and numeric coercions are deliberately refused. */
final class EventNbt {
    static final String KEY = "neurocraft_events";
    private static final String SCHEMA = "neurocraft.events.v1";
    private static final Set<String> KEYS = Set.of("schema", "food_near", "food_contact", "last_food_sample_tick",
            "food", "brush", "attack_attempt");

    private EventNbt() {}

    static EventAccumulator read(Tag value, long gameTick) {
        if (!(value instanceof CompoundTag data) || !data.keySet().equals(KEYS)
                || !(data.get("schema") instanceof StringTag schema) || !SCHEMA.equals(schema.value())) {
            throw new IllegalArgumentException("event NBT schema or keys are invalid");
        }
        if (!(data.get("food_near") instanceof ByteTag near) || (near.value() != 0 && near.value() != 1)
                || !(data.get("food_contact") instanceof ByteTag contact)
                || (contact.value() != 0 && contact.value() != 1)) {
            throw new IllegalArgumentException("event NBT food_contact must be a boolean byte");
        }
        long lastFoodTick = exactLong(data, "last_food_sample_tick");
        if (lastFoodTick < -1 || lastFoodTick > gameTick) {
            throw new IllegalArgumentException("event NBT sample timestamp is invalid");
        }
        EventSample sample = new EventSample(contact.value() == 1, near.value() == 1, readCounter(data.get("food")),
                readCounter(data.get("brush")), readCounter(data.get("attack_attempt")));
        sample.validateAt(gameTick);
        if (((sample.foodNear() || sample.food().count() != 0) && lastFoodTick < 0)
                || (sample.food().lastGameTick() != null && sample.food().lastGameTick() > lastFoodTick)) {
            throw new IllegalArgumentException("event NBT food edge precedes its sample history");
        }
        return new EventAccumulator(sample, lastFoodTick);
    }

    static CompoundTag write(EventAccumulator events) {
        CompoundTag data = new CompoundTag();
        if (events == null) {
            // Persist unavailability too: dropping this marker would turn corruption into zeros.
            data.putString("schema", "neurocraft.events.unavailable.v1");
            return data;
        }
        EventSample sample = events.snapshot();
        data.putString("schema", SCHEMA);
        data.putBoolean("food_near", sample.foodNear());
        data.putBoolean("food_contact", sample.foodContact());
        data.putLong("last_food_sample_tick", events.lastFoodSampleTick());
        data.put("food", writeCounter(sample.food()));
        data.put("brush", writeCounter(sample.brush()));
        data.put("attack_attempt", writeCounter(sample.attackAttempt()));
        return data;
    }

    private static EventSample.Counter readCounter(Tag value) {
        if (!(value instanceof CompoundTag data)) {
            throw new IllegalArgumentException("event NBT counter must be a compound");
        }
        long count = exactLong(data, "count");
        if (!data.keySet().equals(count == 0 ? Set.of("count") : Set.of("count", "last_game_tick"))) {
            throw new IllegalArgumentException("event NBT counter keys are invalid");
        }
        return new EventSample.Counter(count, count == 0 ? null : exactLong(data, "last_game_tick"));
    }

    private static CompoundTag writeCounter(EventSample.Counter counter) {
        CompoundTag data = new CompoundTag();
        data.putLong("count", counter.count());
        if (counter.lastGameTick() != null) { data.putLong("last_game_tick", counter.lastGameTick()); }
        return data;
    }

    private static long exactLong(CompoundTag data, String key) {
        if (!(data.get(key) instanceof LongTag number)) {
            throw new IllegalArgumentException("event NBT " + key + " must be an exact long");
        }
        return number.value();
    }
}
