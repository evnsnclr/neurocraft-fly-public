package dev.neurocraft.fly;

import java.util.Objects;
import java.util.Optional;
import java.util.UUID;

/** One process-wide reset transaction; overlap is rejected rather than overwritten. */
final class StagedResetSlot {
    private StagedReset pending;

    boolean stage(StagedReset candidate) {
        Objects.requireNonNull(candidate, "candidate");
        if (pending != null) {
            return false;
        }
        pending = candidate;
        return true;
    }

    Optional<StagedReset> pending() {
        return Optional.ofNullable(pending);
    }

    /** Prevents an interleaved target-tick action from outrunning the tick-boundary reset. */
    boolean blocksActuation(UUID entityUuid) {
        Objects.requireNonNull(entityUuid, "entityUuid");
        return pending != null && pending.targets(entityUuid);
    }

    void clear(StagedReset expected) {
        if (pending != expected) {
            throw new IllegalStateException("reset slot changed unexpectedly");
        }
        pending = null;
    }

    void clear() {
        pending = null;
    }
}
