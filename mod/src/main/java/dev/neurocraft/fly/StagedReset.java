package dev.neurocraft.fly;

import dev.neurocraft.fly.control.CommandValidator;
import dev.neurocraft.fly.control.ControlAckFrame;
import dev.neurocraft.fly.control.ControlOperation;
import dev.neurocraft.fly.entity.FlyEntity;
import java.util.Objects;
import java.util.Optional;
import java.util.UUID;

/** Server-owned absolute reset that cannot mutate the entity before its bound control ACK. */
public final class StagedReset {
    /** RESET alone permits an accepted delay of one action budget, or one tick more. */
    public enum Outcome {
        WAITING,
        READY,
        REJECTED,
        MISSED,
        INVALID,
        APPLIED
    }

    /**
     * Ticks an unacknowledged reset may wait before it is abandoned. See the calibration note
     * in {@link #advance}: this bounds the validate() gap the staged reset opens, so it is tied
     * to {@code CommandValidator.DORMANCY_GAP_TICKS} rather than to the action budget.
     */
    private static final long ACK_DEADLINE_TICKS = 2;

    private final UUID entityUuid;
    private final long issuedTick;
    private final double x;
    private final double y;
    private final double z;
    private final float yawDegrees;
    private long controlSequence = -1;
    private Outcome outcome = Outcome.WAITING;
    private long effectiveTick = -1;
    private String failureCode;

    public StagedReset(
            UUID entityUuid,
            long issuedTick,
            double x,
            double y,
            double z,
            float yawDegrees) {
        this.entityUuid = Objects.requireNonNull(entityUuid, "entityUuid");
        if (issuedTick < 0
                || issuedTick > Long.MAX_VALUE - CommandValidator.ACTION_TARGET_OFFSET_TICKS - 1
                || !Double.isFinite(x)
                || !Double.isFinite(y)
                || !Double.isFinite(z)
                || !Float.isFinite(yawDegrees)) {
            throw new IllegalArgumentException("staged reset values are out of range");
        }
        this.issuedTick = issuedTick;
        this.x = x;
        this.y = y;
        this.z = z;
        this.yawDegrees = yawDegrees;
    }

    public Outcome advance(long currentTick, Optional<ControlAckFrame> acknowledgement) {
        Objects.requireNonNull(acknowledgement, "acknowledgement");
        if (controlSequence < 0) {
            throw new IllegalStateException("staged reset is not bound to a control sequence");
        }
        if (currentTick < issuedTick) {
            throw new IllegalArgumentException("current tick precedes reset request");
        }
        if (outcome != Outcome.WAITING) {
            return outcome;
        }
        if (acknowledgement.isPresent()) {
            ControlAckFrame ack = acknowledgement.orElseThrow();
            if (ack.controlSequence() != controlSequence
                    || ack.operation() != ControlOperation.RESET
                    || ack.issuedTick() != issuedTick) {
                outcome = Outcome.INVALID;
                failureCode = "ACK_BINDING";
                return outcome;
            }
            if (!ack.accepted()) {
                outcome = Outcome.REJECTED;
                failureCode = ack.faultCode();
                return outcome;
            }
            long effectiveDelay = ack.effectiveTick() - issuedTick;
            // The service stamps an accepted RESET with the governed action's target_tick,
            // which is one action budget after the observation the effect was materialized
            // against -- the issuing tick or the one after it. So the accepted pair is
            // {budget, budget + 1}, and it moved with the budget: it was {1, 2}.
            if (effectiveDelay < CommandValidator.ACTION_TARGET_OFFSET_TICKS
                    || effectiveDelay > CommandValidator.ACTION_TARGET_OFFSET_TICKS + 1) {
                outcome = Outcome.INVALID;
                failureCode = "EFFECTIVE_TICK";
                return outcome;
            }
            effectiveTick = ack.effectiveTick();
            if (currentTick > effectiveTick) {
                outcome = Outcome.MISSED;
                failureCode = "LATE_ACK";
                return outcome;
            }
        }
        if (effectiveTick >= 0) {
            if (currentTick == effectiveTick) {
                outcome = Outcome.READY;
            } else if (currentTick > effectiveTick) {
                // The world passed the accepted boundary without this entity
                // ticking on it (chunk-unload dormancy). Left WAITING, the
                // slot would block actuation - and the ADR 0012 re-anchor -
                // for the rest of the process; resolve MISSED like every
                // other unapplicable reset instead.
                outcome = Outcome.MISSED;
                failureCode = "EFFECTIVE_TICK_PASSED";
            }
            return outcome;
        }
        if (currentTick - issuedTick > ACK_DEADLINE_TICKS) {
            // Deliberately NOT widened with the action budget, and the reason is a
            // calibration, not an oversight. While a reset is staged, blocksActuation
            // returns from FlyControllerRuntime.control() before validate() runs, so this
            // deadline bounds a gap in validate() calls that CommandValidator must not read
            // as chunk-unload dormancy: DORMANCY_GAP_TICKS is 3 and the last validate before
            // the reset is at the issuing tick, so an unacknowledged reset must resolve by
            // issued + 3 exactly as before. Widening the budget did not move this, because an
            // acknowledgement arriving on issued + 3 and naming issued + 3 is still READY on
            // that tick -- the effective-tick branch above runs first and returns.
            outcome = Outcome.MISSED;
            failureCode = "ACK_DEADLINE";
        }
        return outcome;
    }

    public void apply(FlyEntity entity, long currentTick) {
        Objects.requireNonNull(entity, "entity");
        if (outcome != Outcome.READY
                || currentTick != effectiveTick
                || !entityUuid.equals(entity.getUUID())) {
            throw new IllegalStateException("reset is not ready for this entity and tick");
        }
        entity.resetForFixture(x, y, z, yawDegrees);
        outcome = Outcome.APPLIED;
    }

    public long issuedTick() {
        return issuedTick;
    }

    public void bindControlSequence(long sequence) {
        if (sequence < 0 || controlSequence >= 0) {
            throw new IllegalStateException("reset control sequence binding is invalid");
        }
        controlSequence = sequence;
    }

    public long controlSequence() {
        if (controlSequence < 0) {
            throw new IllegalStateException("reset has no control sequence");
        }
        return controlSequence;
    }

    public long effectiveTick() {
        if (effectiveTick < 0) {
            throw new IllegalStateException("reset has no accepted effective tick");
        }
        return effectiveTick;
    }

    public String failureCode() {
        if (failureCode == null) {
            throw new IllegalStateException("reset has no failure");
        }
        return failureCode;
    }

    boolean targets(UUID candidate) {
        return entityUuid.equals(candidate);
    }
}
