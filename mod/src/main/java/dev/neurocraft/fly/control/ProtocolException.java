package dev.neurocraft.fly.control;

/** Checked failure for malformed or contract-invalid protocol input. */
public final class ProtocolException extends Exception {
    private static final long serialVersionUID = 1L;

    public ProtocolException(String message) {
        super(message);
    }

    public ProtocolException(String message, Throwable cause) {
        super(message, cause);
    }
}
