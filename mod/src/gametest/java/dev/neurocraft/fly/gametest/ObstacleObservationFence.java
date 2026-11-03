package dev.neurocraft.fly.gametest;

import dev.neurocraft.fly.control.ActionFrame;
import java.util.Objects;

/**
 * Test-only causal fence that accepts evidence only from an observation sampled after obstacle
 * placement. The first qualifying command is stable while its receipt is in flight.
 */
final class ObstacleObservationFence {
    private final long openCommandSequence;
    private final long obstaclePlacementTick;
    private long blockedCommandSequence = -1;
    private long blockedObservationTick = -1;

    ObstacleObservationFence(long openCommandSequence, long obstaclePlacementTick) {
        if (openCommandSequence < 0 || obstaclePlacementTick < 0) {
            throw new IllegalArgumentException("obstacle fence values must be non-negative");
        }
        this.openCommandSequence = openCommandSequence;
        this.obstaclePlacementTick = obstaclePlacementTick;
    }

    boolean tryLatch(ActionFrame action) {
        Objects.requireNonNull(action, "action");
        if (blockedCommandSequence >= 0
                || action.commandSequence() <= openCommandSequence
                || action.observationTick() <= obstaclePlacementTick) {
            return false;
        }
        blockedCommandSequence = action.commandSequence();
        blockedObservationTick = action.observationTick();
        return true;
    }

    long blockedCommandSequence() {
        if (blockedCommandSequence < 0) {
            throw new IllegalStateException("blocked action has not been latched");
        }
        return blockedCommandSequence;
    }

    long blockedObservationTick() {
        if (blockedObservationTick < 0) {
            throw new IllegalStateException("blocked action has not been latched");
        }
        return blockedObservationTick;
    }
}
