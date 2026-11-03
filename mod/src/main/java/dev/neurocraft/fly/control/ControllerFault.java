package dev.neurocraft.fly.control;

/** Fail-closed controller state exposed to the entity and read-only presentation. */
public enum ControllerFault {
    STARTING(false),
    ACTIVE(false),
    MISSING_CONFIGURATION(true),
    DISCONNECTED(true),
    PROTOCOL_ERROR(true),
    IDENTITY_MISMATCH(true),
    STALE_COMMAND(true),
    FUTURE_COMMAND(true),
    DEADLINE_MISSED(true),
    SEQUENCE_ERROR(true),
    OVERSIZE_MESSAGE(true),
    QUEUE_OVERFLOW(true),
    INTERNAL_ERROR(true);

    private final boolean paused;

    ControllerFault(boolean paused) {
        this.paused = paused;
    }

    public boolean isPaused() {
        return paused;
    }
}
