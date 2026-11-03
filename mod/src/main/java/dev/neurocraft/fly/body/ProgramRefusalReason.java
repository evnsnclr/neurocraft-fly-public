package dev.neurocraft.fly.body;

/**
 * The three declared reasons a requested body program did not run as asked (ADR 0020 section 2).
 *
 * <p>A closed enum rather than a string because the wire value is a closed vocabulary
 * ({@code unregistered | precondition | operator}, lowercase) and the Python
 * {@code ProgramRefusalReason} enforces exactly these three. The wire values are lowercase, unlike
 * {@link BodyProgramId}, because they are protocol vocabulary rather than the frozen program names.
 *
 * <p>The first two mean the declared HOVER fallback ran. The third does not: an operator hold owned
 * the tick, the operator's own program ran, and the receipt reports <em>that</em> program. Forcing
 * HOVER there would make the receipt claim a program ran that did not.
 */
public enum ProgramRefusalReason {
    /** The frame named a declared program that no registry entry provides. */
    UNREGISTERED("unregistered"),

    /** The program declined to enter this world context. */
    PRECONDITION("precondition"),

    /** An operator hold owned the tick, so the requested program never reached the body. */
    OPERATOR("operator");

    private final String wireValue;

    ProgramRefusalReason(String wireValue) {
        this.wireValue = wireValue;
    }

    /** The lowercase protocol vocabulary word. */
    public String wireValue() {
        return wireValue;
    }
}
