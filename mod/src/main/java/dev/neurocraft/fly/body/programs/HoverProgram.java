package dev.neurocraft.fly.body.programs;

import dev.neurocraft.fly.body.BodyProgram;
import dev.neurocraft.fly.body.BodyProgramContext;
import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.BodyProgramState;
import dev.neurocraft.fly.body.BodyProgramStep;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ProgramParameters;
import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import dev.neurocraft.fly.control.PrimitiveAction;
import dev.neurocraft.fly.entity.BodyFrameMath;
import dev.neurocraft.fly.entity.FlyEntity;
import java.util.Optional;

/**
 * HOVER: the default body program (ADR 0020, programs table entry 1).
 *
 * <p><strong>Every constant below is a declared engineering constant.</strong> None is derived from
 * the connectome, from fly biology, or from any measurement of fly behaviour. They are named here
 * so they can be argued with.
 *
 * <p>HOVER is unconditional, unbounded, and never self-exits; it is entered at spawn, on every
 * exit, on every refusal, on release, and on preemption. It reads only {@code amplitude}, so
 * varying {@code speed}, {@code turnRate}, or {@code urgency} produces a bit-identical motor
 * sequence.
 *
 * <p><strong>The bob is actuation, not animation.</strong> Neutral motors hold altitude exactly
 * (pinned by {@code neutralMotorsHoldAltitudeWithoutAmbientSink}), so a zero-motor hover is a body
 * frozen in space with beating wings. The bob moves the authoritative body and enters the world
 * transition, which is why it must be zero-mean over one period and why that is pinned.
 */
public final class HoverProgram implements BodyProgram {
    /** Peak commanded vertical thrust of the bob, as a fraction of the vertical rail. */
    public static final double HOVER_BOB_AMPLITUDE = 0.12;

    /** Bob period in game ticks; 40 ticks is two seconds at 20 Hz. */
    public static final int HOVER_BOB_PERIOD_TICKS = 40;

    /** Body-frame horizontal speed, in blocks/tick, below which the fly is treated as parked. */
    public static final double HOVER_SETTLE_SPEED_BLOCKS_PER_TICK = 0.02;

    /** Maximum station-keeping thrust, as a fraction of the horizontal rails. */
    public static final double HOVER_SETTLE_THRUST = 0.25;

    /** Body-frame horizontal speed, in blocks/tick, at which station keeping reaches full thrust. */
    public static final double HOVER_SETTLE_FULL_SPEED_BLOCKS_PER_TICK = 0.10;

    private static final double BOB_ANGULAR_RATE = 2.0 * Math.PI / HOVER_BOB_PERIOD_TICKS;

    /**
     * Entry phase of the bob, in radians, <strong>derived rather than chosen</strong>.
     *
     * <p>The vertical axis is a first-order filter: the actuator adds thrust to the velocity, the
     * body moves, and then {@code AIR_VERTICAL_DRAG} scales the velocity. Driving that filter with
     * a sine that starts at zero leaves a velocity transient — the steady-state velocity at
     * {@code t = 0} is not zero — and the position integral of that transient is roughly
     * {@code 1/(1 - drag)} times its size. Measured on the declared constants, a plain
     * {@code sin(2*pi*t/40)} bob climbs about <strong>0.83 blocks over its first period</strong>:
     * a hovering fly that slowly ascends forever. The zero-mean pin caught it.
     *
     * <p>Entering at the phase whose steady-state velocity is zero removes the transient entirely.
     * That phase is the filter's own lag at the bob frequency, {@code arg(e^{i*w} - drag)}, which
     * is a closed form in two already-declared constants and moves correctly if either changes.
     */
    private static final double BOB_ENTRY_PHASE_RADIANS = Math.atan2(
            Math.sin(BOB_ANGULAR_RATE), Math.cos(BOB_ANGULAR_RATE) - FlyEntity.AIR_VERTICAL_DRAG);

    @Override
    public BodyProgramId id() {
        return BodyProgramId.HOVER;
    }

    @Override
    public boolean canEnter(BodyProgramContext context) {
        // Unconditional. HOVER is the fallback from every refusal, so it must never refuse.
        return true;
    }

    /**
     * HOVER reads {@code amplitude} alone. It ignores the signed {@link ProgramParameters}
     * entirely: a hover has no heading to bias, and {@code hoverIgnoresTheTurnBias} pins that
     * varying it produces a bit-identical motor sequence.
     */
    @Override
    public BodyProgramStep step(
            BodyProgramState state,
            BodyProgramContext context,
            Modulation modulation,
            ProgramParameters parameters) {
        int tick = state.ticksInProgram();
        double vertical = HOVER_BOB_AMPLITUDE
                * modulation.amplitude()
                * Math.sin(BOB_ANGULAR_RATE * tick + BOB_ENTRY_PHASE_RADIANS);

        PhysicalSensorSnapshot snapshot = context.snapshot();
        BodyFrameMath.BodyVector drift = BodyFrameMath.toBody(
                snapshot.yawDegrees(),
                snapshot.velocityXBlocksPerTick(),
                snapshot.velocityZBlocksPerTick());
        double driftSpeed = Math.hypot(drift.forward(), drift.right());
        double forward = 0.0;
        double lateral = 0.0;
        if (driftSpeed > HOVER_SETTLE_SPEED_BLOCKS_PER_TICK) {
            double thrust = HOVER_SETTLE_THRUST
                    * Math.min(1.0, driftSpeed / HOVER_SETTLE_FULL_SPEED_BLOCKS_PER_TICK);
            forward = -drift.forward() / driftSpeed * thrust;
            lateral = -drift.right() / driftSpeed * thrust;
        }

        int phaseIndex = Math.floorMod(tick + 1, HOVER_BOB_PERIOD_TICKS);
        float presentationPhase =
                (float) (Math.floorMod(tick, HOVER_BOB_PERIOD_TICKS) / (double) HOVER_BOB_PERIOD_TICKS);
        return new BodyProgramStep(
                PrimitiveAction.clamped(forward, lateral, vertical, 0.0, 0.0),
                state.advanced(phaseIndex),
                presentationPhase,
                Optional.empty());
    }
}
