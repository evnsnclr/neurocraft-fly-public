package dev.neurocraft.fly;

import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.BodyProgramRuntime;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.MotorAuthority;
import dev.neurocraft.fly.body.SelectionSource;
import dev.neurocraft.fly.control.CommandValidator;
import dev.neurocraft.fly.control.ControlAckFrame;
import dev.neurocraft.fly.control.ControlOperation;
import java.util.Objects;
import java.util.Optional;
import java.util.UUID;

/**
 * A motor-authority handover that cannot take effect before its bound control acknowledgement
 * (ADR 0020 section 3). Modelled directly on {@link StagedReset}, which stages a world mutation
 * behind an accepted RESET.
 *
 * <p>Entering a body program issues {@link ControlOperation#PAUSE} on the existing acknowledged
 * operator control channel and engages scripted authority <strong>only</strong> on the accepted
 * acknowledgement, on the first authoritative tick that acknowledgement governs. Releasing is the
 * mirror, on {@link ControlOperation#RESUME}. A rejected, invalid, or missed acknowledgement leaves
 * authority at {@code NEURAL} and moves nothing.
 *
 * <p><strong>Which tick the acknowledgement governs.</strong> The service stamps an accepted
 * PAUSE or RESUME acknowledgement with {@code effective_tick = observation.game_tick}: the tick of
 * the <em>observation</em> it first answered in the new control state
 * ({@code services/neural-engine/src/neurocraft_service/server.py}, the action-effect branch of
 * {@code process_latest}). That observation is sent from {@code FlyEntity.tick()} <em>after</em>
 * the same tick has already polled control acknowledgements ({@code processPendingProgramAuthority}
 * runs before {@code control()} offers the observation), so an acknowledgement stamped
 * {@code X} cannot be polled before tick {@code X + 1}, and the paused service's first neutral
 * action -- the first command the authority applies in the new state -- targets
 * {@code X + ACTION_TARGET_OFFSET_TICKS}. The handover engages on <strong>that</strong> tick:
 * {@link #engageTick()} {@code = effectiveTick + ACTION_TARGET_OFFSET_TICKS}, which is also
 * always at or after the first tick the acknowledgement can be polled.
 *
 * <p><strong>The coincidence is the mechanism, not a convenience.</strong> ADR 0020 section 3.3
 * lets a scripted program own a tick only when the admitted command is neutral, so engaging
 * before the paused service's first neutral command lands hands the body to a program on a tick
 * an in-flight <em>driving</em> command still governs, and the program is preempted on the tick
 * it engaged. This offset was one tick while the action budget was one tick; it moved with the
 * budget, and the relation it expresses -- engage on the tick the first paused command governs
 * -- is unchanged. An acknowledgement first seen after its engage tick is {@link Outcome#MISSED}.
 * {@link StagedReset} does not need this offset because the service stamps a RESET with the
 * governed action's {@code target_tick} instead; the two transactions share a shape, not a stamp.
 *
 * <p>This is what makes "the connectome was not driving during this footage" an enforced invariant
 * rather than a promise: a paused service returns a neutral action for every observation instead of
 * stepping the controller, so its trace carries a recomputable {@code service_neutral_v1} digest
 * over {@code reason = "paused"} and a neural cursor that visibly does not advance. See
 * {@code scripts/verify_paused_window.py}.
 */
public final class StagedProgramAuthority {
    /**
     * Ticks a PAUSE/RESUME acknowledgement may take before the transaction is abandoned. The
     * service answers the observation issued at {@code issuedTick + 1} at the latest, and that
     * acknowledgement is first visible at {@code issuedTick + 2}, so this is exactly the window in
     * which an acknowledgement can still engage.
     */
    public static final long ACK_DEADLINE_TICKS = 2;

    /**
     * Offset from an acknowledgement's observation-stamped effective tick to the tick it governs:
     * the action budget, because that is when the paused service's first neutral command applies.
     */
    static final long ENGAGE_OFFSET_TICKS = CommandValidator.ACTION_TARGET_OFFSET_TICKS;

    public enum Outcome {
        WAITING,
        READY,
        REJECTED,
        MISSED,
        INVALID,
        APPLIED
    }

    private final UUID entityUuid;
    private final long issuedTick;
    private final ControlOperation operation;
    private final BodyProgramId program;
    private final Modulation modulation;
    private long controlSequence = -1;
    private Outcome outcome = Outcome.WAITING;
    private long effectiveTick = -1;
    private long engageTick = -1;
    private String failureCode;

    private StagedProgramAuthority(
            UUID entityUuid,
            long issuedTick,
            ControlOperation operation,
            BodyProgramId program,
            Modulation modulation) {
        this.entityUuid = Objects.requireNonNull(entityUuid, "entityUuid");
        if (issuedTick < 0
                || issuedTick > Long.MAX_VALUE - ACK_DEADLINE_TICKS - ENGAGE_OFFSET_TICKS) {
            throw new IllegalArgumentException("staged program authority tick is out of range");
        }
        this.issuedTick = issuedTick;
        this.operation = Objects.requireNonNull(operation, "operation");
        this.program = program;
        this.modulation = modulation;
    }

    /** Stages the PAUSE that hands the motor vector to a scripted program. */
    public static StagedProgramAuthority engage(
            UUID entityUuid, long issuedTick, BodyProgramId program, Modulation modulation) {
        return new StagedProgramAuthority(
                entityUuid,
                issuedTick,
                ControlOperation.PAUSE,
                Objects.requireNonNull(program, "program"),
                Objects.requireNonNull(modulation, "modulation"));
    }

    /** Stages the RESUME that hands the motor vector back to the neural lane. */
    public static StagedProgramAuthority release(UUID entityUuid, long issuedTick) {
        return new StagedProgramAuthority(
                entityUuid, issuedTick, ControlOperation.RESUME, null, null);
    }

    public Outcome advance(long currentTick, Optional<ControlAckFrame> acknowledgement) {
        Objects.requireNonNull(acknowledgement, "acknowledgement");
        if (controlSequence < 0) {
            throw new IllegalStateException("staged authority is not bound to a control sequence");
        }
        if (currentTick < issuedTick) {
            throw new IllegalArgumentException("current tick precedes the authority request");
        }
        if (outcome != Outcome.WAITING) {
            return outcome;
        }
        if (acknowledgement.isPresent()) {
            ControlAckFrame ack = acknowledgement.orElseThrow();
            if (ack.controlSequence() != controlSequence
                    || ack.operation() != operation
                    || ack.issuedTick() != issuedTick) {
                return fail(Outcome.INVALID, "ACK_BINDING");
            }
            if (!ack.accepted()) {
                return fail(Outcome.REJECTED, ack.faultCode());
            }
            long effectiveDelay = ack.effectiveTick() - issuedTick;
            if (effectiveDelay < 0 || effectiveDelay > 1) {
                return fail(Outcome.INVALID, "EFFECTIVE_TICK");
            }
            effectiveTick = ack.effectiveTick();
            engageTick = effectiveTick + ENGAGE_OFFSET_TICKS;
            if (currentTick > engageTick) {
                return fail(Outcome.MISSED, "LATE_ACK");
            }
        }
        if (engageTick >= 0) {
            if (currentTick == engageTick) {
                outcome = Outcome.READY;
            } else if (currentTick > engageTick) {
                // The world passed the governed tick without this entity ticking on it
                // (chunk-unload dormancy). Resolve terminally rather than pin the operator
                // lane WAITING for the rest of the process, exactly as StagedReset does.
                return fail(Outcome.MISSED, "EFFECTIVE_TICK_PASSED");
            }
            return outcome;
        }
        if (currentTick - issuedTick > ACK_DEADLINE_TICKS) {
            return fail(Outcome.MISSED, "ACK_DEADLINE");
        }
        return outcome;
    }

    /** Applies the handover. Refuses outside {@link Outcome#READY} at its exact engage tick. */
    public void apply(BodyProgramRuntime runtime, long currentTick) {
        Objects.requireNonNull(runtime, "runtime");
        if (outcome != Outcome.READY || currentTick != engageTick) {
            throw new IllegalStateException("program authority is not ready for this tick");
        }
        if (operation == ControlOperation.PAUSE) {
            if (runtime.authority() == MotorAuthority.READOUT) {
                // The operator wins, and the handover is logged from both sides: the readout lane
                // loses the body with reason=operator, and the operator lane takes it with
                // reason=selected. One line that blended the two would be the exact defect this
                // project must never ship.
                runtime.releaseScripted(currentTick, "operator");
            }
            runtime.engageScripted(
                    Objects.requireNonNull(program),
                    Objects.requireNonNull(modulation),
                    SelectionSource.OPERATOR,
                    currentTick);
        } else {
            runtime.releaseScripted(currentTick, "released");
        }
        outcome = Outcome.APPLIED;
    }

    public void bindControlSequence(long sequence) {
        if (sequence < 0 || controlSequence >= 0) {
            throw new IllegalStateException("program authority control-sequence binding is invalid");
        }
        controlSequence = sequence;
    }

    public long controlSequence() {
        if (controlSequence < 0) {
            throw new IllegalStateException("program authority has no control sequence");
        }
        return controlSequence;
    }

    public ControlOperation operation() {
        return operation;
    }

    public long issuedTick() {
        return issuedTick;
    }

    /** The accepted acknowledgement's own stamp: the observation tick the service answered paused. */
    public long effectiveTick() {
        if (effectiveTick < 0) {
            throw new IllegalStateException("program authority has no accepted effective tick");
        }
        return effectiveTick;
    }

    /**
     * The authoritative tick the handover engages on: {@link #effectiveTick()} plus
     * {@link #ENGAGE_OFFSET_TICKS}, the action budget, which is the tick the paused service's
     * first neutral command governs.
     */
    public long engageTick() {
        if (engageTick < 0) {
            throw new IllegalStateException("program authority has no accepted engage tick");
        }
        return engageTick;
    }

    public String failureCode() {
        if (failureCode == null) {
            throw new IllegalStateException("program authority has no failure");
        }
        return failureCode;
    }

    public boolean targets(UUID candidate) {
        return entityUuid.equals(candidate);
    }

    private Outcome fail(Outcome failed, String code) {
        outcome = failed;
        failureCode = code == null ? failed.name() : code;
        return outcome;
    }
}
