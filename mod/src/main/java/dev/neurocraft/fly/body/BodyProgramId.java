package dev.neurocraft.fly.body;

/**
 * The ten declared body programs (ADR 0020).
 *
 * <p><strong>These are scripted presentation and engineering.</strong> Nothing in this enum is
 * derived from the connectome, from fly anatomy, or from any measurement of fly behaviour, and a
 * running body program is never evidence that the connectome did anything. In stage 3 the running
 * program is chosen by an operator command; nothing selects one from a neural readout yet.
 *
 * <p>Only the programs registered by {@link BodyProgramRuntime} can actually run. The remaining
 * identifiers are declared here so the published presentation channel, the operator surface, and
 * the transition log have a frozen vocabulary before the programs that fill them exist.
 */
public enum BodyProgramId {
    HOVER,
    CRUISE,
    BANK,
    LAND,
    WALK,
    GROOM,
    FEED,
    ESCAPE_JUMP,
    FLEE,
    APPROACH
}
