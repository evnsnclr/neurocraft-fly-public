package dev.neurocraft.fly.control;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

final class SensorRayHitTest {
    @Test
    void nearestKindAndExactTieAreDeclared() {
        assertEquals(new SensorRayHit(6.0, 0), SensorRayHit.nearest(6.0, null, null));
        assertEquals(new SensorRayHit(3.0, 1), SensorRayHit.nearest(6.0, 3.0, null));
        assertEquals(new SensorRayHit(2.0, 2), SensorRayHit.nearest(6.0, 3.0, 2.0));
        assertEquals(new SensorRayHit(2.0, 1), SensorRayHit.nearest(6.0, 2.0, 3.0));
        assertEquals(new SensorRayHit(2.0, 2), SensorRayHit.nearest(6.0, 2.0, 2.0));
        assertEquals(new SensorRayHit(6.0, 1), SensorRayHit.nearest(6.0, 6.0, null));
        assertEquals(new SensorRayHit(6.0, 2), SensorRayHit.nearest(6.0, null, 6.0));
    }

    @Test
    void actorClassFollowsTheWinnerIncludingExactTieAndMaximumRange() {
        for (int actor = 1; actor <= 3; actor++) {
            assertEquals(new SensorRayHit(2.0, 2, actor), SensorRayHit.nearest(6.0, 2.0, 2.0, actor));
            assertEquals(new SensorRayHit(6.0, 2, actor), SensorRayHit.nearest(6.0, null, 6.0, actor));
            assertEquals(new SensorRayHit(1.0, 1, 0), SensorRayHit.nearest(6.0, 1.0, 2.0, actor));
            assertEquals(new SensorRayHit(6.0, 0, 0), SensorRayHit.nearest(6.0, null, 6.1, actor));
        }
        assertNull(new SensorRayHit(2, 2).actorKind());
        assertThrows(IllegalArgumentException.class, () -> new SensorRayHit(2, 2, 0));
        assertThrows(IllegalArgumentException.class, () -> new SensorRayHit(2, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> SensorRayHit.nearest(6, null, 2.0, 0));
        assertThrows(IllegalArgumentException.class, () -> SensorRayHit.nearest(6, null, null, 2));
    }
}
