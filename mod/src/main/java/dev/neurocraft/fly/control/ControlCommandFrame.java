package dev.neurocraft.fly.control;

import java.util.Objects;
import java.util.regex.Pattern;

/** Immutable outbound operator request; movement remains exclusively Action motor primitives. */
public record ControlCommandFrame(
        ControllerIdentity identity,
        long controlSequence,
        long issuedTick,
        ControlOperation operation,
        String contentId,
        String contentHash) {
    private static final Pattern IDENTIFIER =
            Pattern.compile("[A-Za-z0-9][A-Za-z0-9._:-]{0,127}");
    private static final Pattern HASH = Pattern.compile("[0-9a-f]{64}");

    public ControlCommandFrame {
        Objects.requireNonNull(identity, "identity");
        Objects.requireNonNull(operation, "operation");
        if (controlSequence < 0 || issuedTick < 0) {
            throw new IllegalArgumentException("control sequence and tick must be non-negative");
        }
        if (contentId != null && !IDENTIFIER.matcher(contentId).matches()) {
            throw new IllegalArgumentException("contentId is not a protocol identifier");
        }
        if (contentHash != null && !HASH.matcher(contentHash).matches()) {
            throw new IllegalArgumentException("contentHash is not a lowercase SHA-256 digest");
        }
        switch (operation) {
            case PAUSE, RESUME, SINGLE_STEP, RESET -> {
                if (contentId != null || contentHash != null) {
                    throw new IllegalArgumentException(operation + " cannot carry content identity");
                }
            }
            case REPLAY_START, LESION_APPLY, STIMULATION_APPLY -> {
                if (contentId == null || contentHash == null) {
                    throw new IllegalArgumentException(
                            operation + " requires content identity and hash");
                }
            }
            case LESION_CLEAR, STIMULATION_CLEAR -> {
                if (contentId == null) {
                    throw new IllegalArgumentException(operation + " requires contentId");
                }
            }
            default -> {
                if (contentId == null) {
                    throw new IllegalArgumentException(operation + " requires contentId");
                }
            }
        }
    }
}
