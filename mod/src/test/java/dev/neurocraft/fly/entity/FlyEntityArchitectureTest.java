package dev.neurocraft.fly.entity;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.Mob;
import org.junit.jupiter.api.Test;

final class FlyEntityArchitectureTest {
    @Test
    void directEntityInheritanceMakesMobGoalsAndNavigationUnavailable() {
        assertEquals(Entity.class, FlyEntity.class.getSuperclass());
        assertFalse(Mob.class.isAssignableFrom(FlyEntity.class));
    }
}
