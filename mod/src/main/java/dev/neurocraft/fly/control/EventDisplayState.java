package dev.neurocraft.fly.control;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Game-thread display gate over commands that the caller has actually applied.
 * This class never admits, rejects, or modifies a motor command. No display data reaches control.
 */
public final class EventDisplayState {
    public static final int APPLIED_LEDGER_CAPACITY = 128;
    public static final long MAX_AGE_NANOS = 1_000_000_000L;
    public static final long MAX_OBSERVATION_AGE_TICKS = 20;
    public enum Status {
        MATCHED, DISABLED, UNAVAILABLE, INVALID, WAITING_FOR_APPLIED_COMMAND, ENVELOPE_MISMATCH,
        IDENTITY_MISMATCH, PROVENANCE_CHANGED, SUPERSEDED, STALE
    }

    /** ageNanos is elapsed since first file read, not sample production or transport latency. */
    public record View(Status status, Optional<EventDisplayFrame> frame, long ageNanos) {}

    private final ControllerIdentity identity;
    private final Map<Long, CommandFrame> applied = new LinkedHashMap<>();
    private EventDisplayFrame accepted;
    private long firstSeenNanos;
    private String encoderHash;
    private String bindingHash;

    public EventDisplayState(ControllerIdentity identity) {
        this.identity = Objects.requireNonNull(identity, "identity");
    }

    /** Call after actual application, never merely after parsing, admission, or receipt refusal. */
    public boolean recordApplied(CommandFrame command) {
        if (!(command instanceof ProgramActionFrame) || !identity.equals(command.identity())) {
            return false;
        }
        CommandFrame existing = applied.get(command.commandSequence());
        if (existing != null) {
            return existing.equals(command);
        }
        applied.put(command.commandSequence(), command);
        if (applied.size() > APPLIED_LEDGER_CAPACITY) {
            applied.remove(applied.keySet().iterator().next());
        }
        return true;
    }

    /**
     * Use the reader's monotonic clock and the actual authoritative game tick. Both age bounds
     * apply: first read <= 1 second ago, and the source observation <= 20 game ticks ago. Thus an
     * old file first discovered after a stalled producer cannot appear fresh merely on read.
     */
    public View view(EventDisplayReader.Snapshot snapshot, long nowNanos, long currentGameTick) {
        if (snapshot.status() != EventDisplayReader.Status.AVAILABLE) {
            return hidden(snapshot.status() == EventDisplayReader.Status.INVALID
                    ? Status.INVALID : Status.UNAVAILABLE, 0);
        }
        EventDisplayFrame frame = snapshot.frame().orElseThrow();
        if (!identity.equals(frame.identity())) {
            return hidden(Status.IDENTITY_MISMATCH, 0);
        }
        if (accepted != null && frame.commandSequence() < accepted.commandSequence()) {
            return hidden(Status.SUPERSEDED, 0);
        }
        boolean sameCommand = accepted != null
                && frame.commandSequence() == accepted.commandSequence();
        if (sameCommand && !accepted.equals(frame)) {
            return hidden(Status.ENVELOPE_MISMATCH, 0);
        }
        long seen = sameCommand ? firstSeenNanos : snapshot.firstSeenNanos();
        long age = nowNanos - seen;
        if (age < 0 || age > MAX_AGE_NANOS || currentGameTick < frame.observationTick()
                || currentGameTick - frame.observationTick() > MAX_OBSERVATION_AGE_TICKS) {
            return hidden(Status.STALE, age);
        }
        CommandFrame command = applied.get(frame.commandSequence());
        if (command == null) {
            return hidden(Status.WAITING_FOR_APPLIED_COMMAND, age);
        }
        if (!frame.matches(command)) {
            return hidden(Status.ENVELOPE_MISMATCH, age);
        }
        // The local file is diagnostic, not authenticated. Freeze its first matched provenance
        // for this epoch so later frames cannot silently swap the displayed population binding.
        if (bindingHash != null && (!bindingHash.equals(frame.bindingHash())
                || !encoderHash.equals(frame.encoderHash()))) {
            return hidden(Status.PROVENANCE_CHANGED, age);
        }
        if (!sameCommand) {
            accepted = frame;
            firstSeenNanos = seen;
            bindingHash = frame.bindingHash();
            encoderHash = frame.encoderHash();
        }
        return new View(Status.MATCHED, Optional.of(frame), age);
    }

    /** Clear on entity/session reset, including every old application and display reference. */
    public void reset() {
        applied.clear();
        accepted = null;
        firstSeenNanos = 0;
        encoderHash = null;
        bindingHash = null;
    }

    public int appliedLedgerSize() {
        return applied.size();
    }

    private static View hidden(Status status, long age) {
        return new View(status, Optional.empty(), age);
    }
}
