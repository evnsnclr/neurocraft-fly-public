package dev.neurocraft.fly.body;

/**
 * Who owns the motor vector on a given tick (ADR 0020 section 3).
 *
 * <p>The three values are mutually exclusive by construction, and each is reached by exactly one
 * route. {@link #SCRIPTED} is reached only through an accepted {@code PAUSE} acknowledgement on the
 * operator control channel, and a scripted program may own a tick only while the admitted command
 * carries no motor request: an admitted <em>non-neutral {@code action}</em> drops the program the
 * instant it arrives. {@link #READOUT} is reached only by an admitted {@code program_action} frame,
 * which means the neural service is RUNNING and stepping.
 *
 * <p><strong>An admitted {@code program_action} does not preempt {@link #SCRIPTED}</strong>, which
 * narrows ADR 0020 section 3.3's "the connectome always wins" to the frame kind that argument was
 * written about. That rule exists because a scripted program displacing a requested motor vector
 * would leave {@code encodeActuationReceipt} with no honest word for the difference; a program
 * frame requests no motors and has its own honest receipt, which reports the operator's program and
 * refuses with reason {@code operator}. The operator keeps the body, and the receipt says so.
 *
 * <p><strong>{@link #READOUT} is not {@link #SCRIPTED} under another name.</strong> Their interlocks
 * are opposites: a scripted program is dropped by any non-neutral admitted command, whereas the
 * readout lane is <em>driven</em> by admitted command frames and is dropped by a <em>neutral</em>
 * one, which is what a paused service emits. Reusing SCRIPTED would also have silently changed what
 * {@code requestReset}'s refusal and {@code releaseAndResetBodyPrograms} key on.
 */
public enum MotorAuthority {
    /** The validated neural command drives the body directly, as primitive motors. */
    NEURAL,

    /** A scripted body program drives the body while the neural service is paused. */
    SCRIPTED,

    /**
     * A scripted body program selected each tick by the connectome's labeled readouts, through a
     * declared hand-chosen mapping (ADR 0019), while the neural service is RUNNING and
     * {@code program_action} frames are admitted. The mapping is engineering and the programs are
     * scripted; this value names who moved the body, not who decided anything.
     */
    READOUT
}
