package dev.neurocraft.fly.body;

/**
 * Where the running body program's selection came from (ADR 0020 section 2).
 *
 * <p>A program never selects itself. The source is named in code, in the transition log, in
 * {@code /neurocraftfly status}, and on the HUD every frame, because the difference between an
 * operator-chosen program and a readout-chosen one is exactly the difference this project must
 * never blur.
 */
public enum SelectionSource {
    /** {@code HOVER}, entered at spawn, on any exit, on any refusal, and on release. */
    DEFAULT,

    /** Chosen by {@code /neurocraftfly program}. Scripted; the connectome is not driving. */
    OPERATOR,

    /** Reserved for stage 4: chosen by a labeled readout. Nothing emits this value yet. */
    READOUT
}
