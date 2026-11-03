package dev.neurocraft.fly.entity;

import static org.junit.jupiter.api.Assertions.*;

import dev.neurocraft.fly.control.EventSample;
import java.util.ArrayList;
import net.minecraft.nbt.CompoundTag;
import net.minecraft.nbt.StringTag;
import org.junit.jupiter.api.Test;

final class EventNbtTest {
    @Test
    void roundTripPreservesCountersEdgeStateAndTheAlreadySampledTick() {
        var original = new EventAccumulator();
        original.recordBrush(2);
        original.recordAttackAttempt(3);
        original.sampleFood(true, true, 4);
        var stored = EventNbt.write(original);
        assertTrue(stored.sizeInBytes() < 2048);
        var loaded = EventNbt.read(stored, 4);
        assertEquals(original.snapshot(), loaded.snapshot());
        assertEquals(original.snapshot(), loaded.sampleFood(false, false, 4));
        assertEquals(1, loaded.sampleFood(true, true, 5).food().count());
        loaded.sampleFood(false, false, 6);
        assertEquals(2, loaded.sampleFood(true, true, 7).food().count());
    }

    @Test
    void numericCoercionsPartialUnknownAndFutureRecordsAreRefused() {
        var original = new EventAccumulator();
        original.sampleFood(true, true, 4);
        var good = EventNbt.write(original);
        var invalid = new ArrayList<CompoundTag>();
        var partial = good.copy(); partial.remove("brush"); invalid.add(partial);
        var extra = good.copy(); extra.putString("unexpected", "value"); invalid.add(extra);
        var numeric = good.copy(); numeric.getCompoundOrEmpty("food").putDouble("count", 1.0); invalid.add(numeric);
        var bool = good.copy(); bool.putByte("food_contact", (byte) 2); invalid.add(bool);
        var tick = good.copy(); tick.putLong("last_food_sample_tick", 5); invalid.add(tick);
        var eventTick = good.copy(); eventTick.getCompoundOrEmpty("food").putLong("last_game_tick", 5); invalid.add(eventTick);
        var absentTick = good.copy(); absentTick.getCompoundOrEmpty("food").remove("last_game_tick"); invalid.add(absentTick);
        var zeroWithTick = good.copy(); zeroWithTick.getCompoundOrEmpty("food").putLong("count", 0); invalid.add(zeroWithTick);
        var schema = good.copy(); schema.putString("schema", "unrecognized"); invalid.add(schema);
        for (var value : invalid) { assertThrows(IllegalArgumentException.class, () -> EventNbt.read(value, 4)); }
        assertThrows(IllegalArgumentException.class, () -> EventNbt.read(StringTag.valueOf("wrong"), 4));
    }

    @Test
    void unavailableStateSurvivesSavingInsteadOfBecomingLegacyZeroState() {
        var unavailable = EventNbt.write(null);
        assertFalse(unavailable.isEmpty());
        assertThrows(IllegalArgumentException.class, () -> EventNbt.read(unavailable, 10));
        assertEquals(EventSample.EMPTY, EventNbt.read(EventNbt.write(new EventAccumulator()), 0).snapshot());
    }
}
