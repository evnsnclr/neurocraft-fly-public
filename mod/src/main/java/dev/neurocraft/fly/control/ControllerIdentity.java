package dev.neurocraft.fly.control;

import java.util.regex.Pattern;

/** Immutable identities that bind every command to one declared experiment configuration. */
public record ControllerIdentity(
        String runId,
        long sessionEpoch,
        String entityId,
        String graphHash,
        String adapterHash,
        String dynamicsHash,
        String decoderHash,
        String observationSchemaHash) {
    private static final Pattern SAFE_ID =
            Pattern.compile("[A-Za-z0-9][A-Za-z0-9._:-]{0,127}");
    private static final Pattern SHA_256 = Pattern.compile("[0-9a-f]{64}");

    public ControllerIdentity {
        runId = requireIdentifier(runId, "runId");
        if (sessionEpoch < 0) {
            throw new IllegalArgumentException("sessionEpoch must be non-negative");
        }
        entityId = requireIdentifier(entityId, "entityId");
        graphHash = requireHash(graphHash, "graphHash");
        adapterHash = requireHash(adapterHash, "adapterHash");
        dynamicsHash = requireHash(dynamicsHash, "dynamicsHash");
        decoderHash = requireHash(decoderHash, "decoderHash");
        observationSchemaHash = requireHash(observationSchemaHash, "observationSchemaHash");
    }

    public boolean matches(CommandFrame frame) {
        return equals(frame.identity());
    }

    public boolean matches(HelloAck frame) {
        return equals(frame.identity());
    }

    private static String requireIdentifier(String value, String name) {
        if (!SAFE_ID.matcher(value).matches()) {
            throw new IllegalArgumentException(name + " is not a valid protocol identifier");
        }
        return value;
    }

    private static String requireHash(String value, String name) {
        if (!SHA_256.matcher(value).matches()) {
            throw new IllegalArgumentException(name + " must be a lowercase SHA-256 digest");
        }
        return value;
    }
}
