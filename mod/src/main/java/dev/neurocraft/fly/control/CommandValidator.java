package dev.neurocraft.fly.control;

import java.util.ArrayDeque;
import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.LongFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Stateful final validator invoked only from the authoritative Minecraft server thread. */
public final class CommandValidator {
    /**
     * Which driving command kind this session has latched (stage 4b design section 2.3).
     *
     * <p>A session is driven by raw {@code action} frames or by {@code program_action} frames, never
     * both: the two carry different receipt semantics and different body authority, and a mixed
     * session would leave no honest answer to "what was driving the fly at tick T". The service
     * side is structural — a controller profile returns one kind for the whole session — and this
     * latch is the independent second check, mirroring {@code _FrameLane} in the service.
     */
    public enum FrameLane {
        UNDECIDED,
        MOTOR,
        PROGRAM
    }

    private static final Logger LOGGER = LoggerFactory.getLogger("neurocraft-fly-controller");

    public static final long MAX_COMMAND_AGE_TICKS = 2;
    public static final long FIRST_ACTION_DEADLINE_TICKS = MAX_COMMAND_AGE_TICKS;
    public static final int MAX_PENDING_COMMANDS = 2;

    /**
     * Ticks between the observation a command answers and the tick it is applied on: every
     * command frame declares {@code target_tick = observation_tick + ACTION_TARGET_OFFSET_TICKS}
     * and {@code valid_through_tick = target_tick}, and this validator applies a frame only on
     * its own target tick. The constant is the whole round-trip budget -- encode, send, compute,
     * return, decode, and be reserved here when that tick arrives -- and it is the Java half of
     * {@code neurocraft_protocol.ACTION_TARGET_OFFSET_TICKS}; the two must move together.
     *
     * <p><strong>Two, not one.</strong> Under the one-tick budget (ADR 0018) the whole round trip
     * had to fit in 50 ms while the controller step alone measures p50 38.9 / p99 56.7 / max
     * 62.9 ms offline on the operator's machine, so about half of all frames were late by
     * construction and the first live program-lane sessions died on DEADLINE_MISSED within two
     * seconds. The control lag grows by exactly one tick.
     *
     * <p><strong>What it changes here, and what it does not.</strong> A frame may now be
     * reserved one tick before it applies, so the steady state holds one frame waiting plus the
     * arriving one -- exactly {@link #MAX_PENDING_COMMANDS}. The future bound below moves with
     * it, staying the same relative statement (no frame may target more than one whole budget
     * ahead of the authoritative clock). {@code MAX_COMMAND_AGE_TICKS} is deliberately untouched:
     * a frame's age at application is now exactly 2, which the {@code age >} comparison admits,
     * and a frame arriving a tick later still retires fail-neutral on that same unchanged rule.
     */
    public static final long ACTION_TARGET_OFFSET_TICKS = 2;

    /**
     * Declared fail-neutral grace beyond the last applied frame's validity window before
     * an absent applicable action latches DEADLINE_MISSED (ADR 0013). The value is a
     * declared boundary constant, not a tuning knob: seven ticks (350 ms at 20 TPS) covers
     * a burst of roughly twice the measured p99.9 live full-brain compute spike plus
     * tick jitter, while still failing closed within four tenths of a second of genuine
     * starvation. It extends only the applied-anchor hold in actuationHoldThroughTick()
     * (the re-armed post-RESET hold shares that anchor and carries the same grace);
     * DORMANCY_GAP_TICKS is deliberately untouched. The cold-start path carries its own
     * separate COLD_START_GRACE_TICKS since ADR 0018 &mdash; this constant does not reach it.
     *
     * <p><strong>It was six under the one-tick budget, and the tolerated gap did not
     * change when it became seven.</strong> The anchor is the applied frame's
     * validThroughTick. Under the one-tick budget a frame applied at tick {@code t} was
     * computed from observation {@code t - 1} and declared {@code validThrough = t + 1},
     * so the hold ran through {@code t + 7}. Under ACTION_TARGET_OFFSET_TICKS = 2 the
     * budget ceiling and the target coincide ({@code validThrough == target == t}), so
     * the anchor sits exactly one tick earlier and the grace absorbs exactly that one
     * tick. The session still hovers fail-neutral for seven frameless ticks after an
     * application and still latches on the eighth &mdash; the same 8-tick (400 ms)
     * tolerance ADR 0018 declared as an invariant across the first application, which
     * {@code coldStartToleratesExactlyTheSameFramelessGapAsTheAppliedHold} pins from both
     * sides. What the constant expresses is the tolerated gap; the number moved so the
     * gap would not.
     */
    public static final long HOLD_GRACE_TICKS = 7;

    /**
     * Declared fail-neutral grace on the cold-start hold, before any action has applied
     * (ADR 0018). Like HOLD_GRACE_TICKS this is a declared boundary constant, not a
     * tuning knob, and it is derived rather than chosen: the applied-anchor hold latches
     * eight ticks after the last applied tick (under the two-tick budget validThrough == t,
     * plus HOLD_GRACE_TICKS = 7, latch at t+8; it was validThrough = t+1 plus a grace of 6),
     * so for the cold branch to latch eight ticks after activation the hold must run through
     * firstActive + FIRST_ACTION_DEADLINE_TICKS + 5. The declared invariant is that a session
     * tolerates the same 8-tick (400 ms) frameless gap whether or not an action has yet
     * applied.
     *
     * <p><strong>Disclosed: the gap is the same eight ticks, but a cold session now gets one
     * fewer application attempt.</strong> The hold still ends at firstActive + 7, and the
     * first frame a service can answer with now targets firstActive + ACTION_TARGET_OFFSET_TICKS
     * rather than firstActive + 1, so the applicable ticks are firstActive+2..firstActive+7 --
     * six, where the one-tick budget gave seven. Steady state keeps seven (the applied hold
     * moved with the anchor) and the post-RESET window gains one. This constant is
     * deliberately NOT raised to compensate: it is the term ADR 0018 derived to make the cold
     * and applied holds tolerate the same frameless gap, and raising it would break that
     * symmetry, which coldStartToleratesExactlyTheSameFramelessGapAsTheAppliedHold pins.
     *
     * <p>It is an additive term on the two cold-start hold evaluations only. It must never
     * be folded into FIRST_ACTION_DEADLINE_TICKS, which is aliased to
     * MAX_COMMAND_AGE_TICKS: raising that alias would widen the freshness bound (admitting
     * a genuinely stale frame for application) and shift DORMANCY_GAP_TICKS. Every
     * admission rule is byte-identical under this constant; graced ticks are
     * rejected-ACTIVE with neutral motors, exactly as the ungraced first two already were.
     */
    public static final long COLD_START_GRACE_TICKS = 5;

    /**
     * Largest gap between consecutive validate() calls the tick-continuous world can
     * produce. validate() runs on every entity tick while the fly is live EXCEPT for
     * two bounded windows that skip it without dormancy: a staged RESET blocking
     * actuation (bounded by StagedReset's 2-tick ACK deadline, resolving by tick
     * issued+3 worst case) and a pending terminal fault (latched synchronously the
     * same tick). This bound is therefore calibrated against StagedReset's ACK
     * window; changing either in isolation reopens a spurious-reanchor hole
     * (ADR 0012, adversarial-review note).
     */
    public static final long DORMANCY_GAP_TICKS = FIRST_ACTION_DEADLINE_TICKS + 1;

    /** Process-lifetime lateness retirements across validators; operator status surface only. */
    private static final AtomicLong TOTAL_LATENESS_RETIREMENTS = new AtomicLong();

    /** Process-lifetime dormancy re-anchors across validators; operator status surface only. */
    private static final AtomicLong TOTAL_DORMANCY_REANCHORS = new AtomicLong();

    private final ControllerIdentity identity;
    private final ArrayDeque<CommandFrame> pending = new ArrayDeque<>(MAX_PENDING_COMMANDS);
    private long lastCommandSequence = -1;
    private long lastNeuralStepEnd = -1;
    private long lastAppliedValidThroughTick = -1;
    private long firstActiveGameTick = -1;
    private long resetBoundaryTick = -1;
    private long lastValidatedGameTick = -1;
    private long latenessRetirements;
    private long dormancyReanchors;
    private FrameLane lane = FrameLane.UNDECIDED;
    private ControllerFault state = ControllerFault.STARTING;

    public CommandValidator(ControllerIdentity identity) {
        this.identity = Objects.requireNonNull(identity, "identity");
    }

    public ValidationResult validate(
            List<? extends CommandFrame> candidates,
            long currentGameTick,
            ControllerFault bridgeState,
            LongFunction<OptionalLong> sentObservationLookup) {
        Objects.requireNonNull(candidates, "candidates");
        Objects.requireNonNull(bridgeState, "bridgeState");
        Objects.requireNonNull(sentObservationLookup, "sentObservationLookup");
        if (currentGameTick < 0) {
            throw new IllegalArgumentException("current game tick must be non-negative");
        }
        try {
            return validateChecked(
                    candidates, currentGameTick, bridgeState, sentObservationLookup);
        } finally {
            lastValidatedGameTick = currentGameTick;
        }
    }

    private ValidationResult validateChecked(
            List<? extends CommandFrame> candidates,
            long currentGameTick,
            ControllerFault bridgeState,
            LongFunction<OptionalLong> sentObservationLookup) {
        if (state.isPaused()) {
            return ValidationResult.rejected(state);
        }
        if (bridgeState.isPaused()) {
            return reject(bridgeState);
        }
        if (bridgeState != ControllerFault.ACTIVE) {
            state = ControllerFault.STARTING;
            firstActiveGameTick = -1;
            return ValidationResult.rejected(ControllerFault.STARTING);
        }
        state = ControllerFault.ACTIVE;
        if (firstActiveGameTick < 0) {
            firstActiveGameTick = currentGameTick;
        }
        if (lastValidatedGameTick >= 0
                && currentGameTick - lastValidatedGameTick > DORMANCY_GAP_TICKS) {
            reanchorAfterDormancy(currentGameTick);
        }
        if (lastCommandSequence < 0 && currentGameTick > coldStartHoldThroughTick()) {
            // The second of the two cold-start hold evaluations, and the binding one
            // whenever no frame has arrived at all: it runs before the candidate loop, so
            // it decides the session before the hold rule below ever sees a frame. It
            // therefore carries the same COLD_START_GRACE_TICKS as the hold (ADR 0018),
            // through the same helper, so the two can never drift apart. Before ADR 0018
            // this compared against FIRST_ACTION_DEADLINE_TICKS directly; the redundant
            // `currentGameTick > firstActiveGameTick` clause is subsumed, because the
            // graced bound is strictly greater than firstActiveGameTick.
            return reject(ControllerFault.DEADLINE_MISSED);
        }

        for (CommandFrame candidate : candidates) {
            ControllerFault fault = acceptCandidate(
                    Objects.requireNonNull(candidate, "candidate"),
                    currentGameTick,
                    sentObservationLookup);
            if (fault != null) {
                return reject(fault, candidate);
            }
        }

        while (true) {
            CommandFrame next = pending.peekFirst();
            if (next == null) {
                if (currentGameTick <= actuationHoldThroughTick()) {
                    return ValidationResult.rejected(ControllerFault.ACTIVE);
                }
                return reject(ControllerFault.DEADLINE_MISSED);
            }
            if (next.targetTick() < currentGameTick) {
                // The frame was fully verified and reserved while valid, but its target
                // passed during a queued wait (a silent catch-up tick burst). Its only
                // defect is lateness, so retire it fail-neutral (ADR 0011): drop the
                // reservation without applying anything and examine the next reserved
                // frame in this same pass. The hold rule above still latches
                // DEADLINE_MISSED on persistent absence, so the deadline is unchanged.
                pending.removeFirst();
                countLatenessRetirement();
                continue;
            }
            if (next.targetTick() > currentGameTick) {
                if (currentGameTick > actuationHoldThroughTick()) {
                    return reject(ControllerFault.DEADLINE_MISSED, next);
                }
                return ValidationResult.rejected(ControllerFault.ACTIVE);
            }
            pending.removeFirst();
            lastAppliedValidThroughTick = next.validThroughTick();
            return ValidationResult.accepted(next);
        }
    }

    public ControllerFault state() {
        return state;
    }

    /** The driving command kind this session has latched; {@code UNDECIDED} until one arrives. */
    public FrameLane lane() {
        return lane;
    }

    public long lastCommandSequence() {
        return lastCommandSequence;
    }

    public int pendingCommandCount() {
        return pending.size();
    }

    /** Verified frames retired for pure lateness by this validator (ADR 0011). */
    public long latenessRetirements() {
        return latenessRetirements;
    }

    /** Process-lifetime lateness retirements across all validators, for operator status. */
    public static long totalLatenessRetirements() {
        return TOTAL_LATENESS_RETIREMENTS.get();
    }

    /** Dormancy re-anchors performed by this validator (ADR 0012). */
    public long dormancyReanchors() {
        return dormancyReanchors;
    }

    /** Process-lifetime dormancy re-anchors across all validators, for operator status. */
    public static long totalDormancyReanchors() {
        return TOTAL_DORMANCY_REANCHORS.get();
    }

    /**
     * Zeroes both process-global counters (lateness retirements and dormancy
     * re-anchors). This class stays Fabric-free, so the SERVER_STOPPING registration
     * lives in NeuroCraftFly.registerLifecycle, mirroring FlyTrailPresentation's
     * reset convention; that call site owns the single stop-time hook, so this method
     * keeps its original name and resets every validator counter.
     */
    public static void resetTotalLatenessRetirements() {
        TOTAL_LATENESS_RETIREMENTS.set(0);
        TOTAL_DORMANCY_REANCHORS.set(0);
    }

    /**
     * Clears only reserved actuation at an accepted RESET boundary; session sequences persist.
     *
     * <p>The latched frame lane is deliberately <strong>not</strong> cleared: a RESET replaces the
     * world, not the controller on the other end of the socket, so a session that was driving with
     * program frames is still that session afterwards.
     */
    public void resetActuationWindow(long effectiveTick) {
        if (effectiveTick < 0) {
            throw new IllegalArgumentException("reset effective tick must be non-negative");
        }
        pending.clear();
        resetBoundaryTick = effectiveTick;
        firstActiveGameTick = effectiveTick;
        lastAppliedValidThroughTick = effectiveTick > Long.MAX_VALUE - FIRST_ACTION_DEADLINE_TICKS
                ? Long.MAX_VALUE
                : effectiveTick + FIRST_ACTION_DEADLINE_TICKS;
        state = ControllerFault.ACTIVE;
    }

    private ControllerFault acceptCandidate(
            CommandFrame frame,
            long currentGameTick,
            LongFunction<OptionalLong> sentObservationLookup) {
        if (!identity.matches(frame)) {
            return ControllerFault.IDENTITY_MISMATCH;
        }
        if (!latchLane(frame)) {
            return ControllerFault.PROTOCOL_ERROR;
        }
        if (frame.commandSequence() != lastCommandSequence + 1) {
            return ControllerFault.SEQUENCE_ERROR;
        }
        if (retiresAtResetBoundary(frame, currentGameTick)) {
            // The action was produced for a world the accepted RESET has already replaced, so it
            // can never be applied. Retire it fail-neutral instead of latching: consume the frame,
            // carry the session sequence forward so the next fresh frame is not a SEQUENCE_ERROR,
            // and reserve nothing. Its observation receipt may also have aged out of the bounded
            // ledger during the ticks the pending reset blocked actuation.
            lastCommandSequence = frame.commandSequence();
            lastNeuralStepEnd = frame.neuralStepEnd();
            return null;
        }
        OptionalLong recordedTick = sentObservationLookup.apply(frame.observationSequence());
        if (recordedTick.isEmpty()) {
            // A missing receipt leaves the frame unverifiable, so it still latches. The
            // ledger (SentObservationWindow) holds
            // FIRST_ACTION_DEADLINE_TICKS + COLD_START_GRACE_TICKS + 1 entries, which is
            // every tick the graced cold-start hold tolerates (ADR 0018), so no frame a
            // live session can still be waiting on has aged its receipt out.
            return frame.observationTick() > currentGameTick
                    ? ControllerFault.FUTURE_COMMAND
                    : ControllerFault.STALE_COMMAND;
        }
        if (recordedTick.getAsLong() != frame.observationTick()) {
            return ControllerFault.IDENTITY_MISMATCH;
        }
        // Both of these are the same clock inconsistency (ADR 0011: "a future frame is a
        // clock inconsistency, not lateness"), and both latch the same fault. The target
        // bound is evaluated first so it stays reachable: a frame may declare validity at
        // most ACTION_TARGET_OFFSET_TICKS past its own observation, so once the age is known
        // non-negative the target bound can no longer be violated. Evaluating it first
        // changes no outcome -- every frame that reaches either check leaves with
        // FUTURE_COMMAND -- and keeps the rule pinned by a test rather than unreachable.
        if (frame.targetTick() > currentGameTick + ACTION_TARGET_OFFSET_TICKS) {
            return ControllerFault.FUTURE_COMMAND;
        }
        long age = currentGameTick - frame.observationTick();
        if (age < 0) {
            return ControllerFault.FUTURE_COMMAND;
        }
        if (lastNeuralStepEnd >= 0 && frame.neuralStepStart() < lastNeuralStepEnd) {
            return ControllerFault.SEQUENCE_ERROR;
        }
        if (!pending.isEmpty() && frame.targetTick() <= pending.getLast().targetTick()) {
            return ControllerFault.SEQUENCE_ERROR;
        }
        if (age > MAX_COMMAND_AGE_TICKS
                || currentGameTick > frame.validThroughTick()
                || frame.targetTick() < currentGameTick) {
            // Identity, sequence, observation receipt, recorded tick, neural-step
            // continuity, and target ordering are all verified: this is provably the next
            // honest frame of this session's stream and its only defect is lateness, so
            // it can never be applied. Extend the reset-boundary precedent above
            // (ADR 0006 -> ADR 0011): retire it fail-neutral by consuming the frame,
            // advancing session bookkeeping, reserving nothing, and latching nothing.
            lastCommandSequence = frame.commandSequence();
            lastNeuralStepEnd = frame.neuralStepEnd();
            countLatenessRetirement();
            return null;
        }
        if (pending.size() >= MAX_PENDING_COMMANDS) {
            return ControllerFault.SEQUENCE_ERROR;
        }
        pending.addLast(frame);
        lastCommandSequence = frame.commandSequence();
        lastNeuralStepEnd = frame.neuralStepEnd();
        return null;
    }

    /**
     * Latches this session's driving frame kind, or reports a mixed session.
     *
     * <p>A {@code program_action} decides the PROGRAM lane; a <strong>non-neutral</strong>
     * {@code action} decides the MOTOR lane; the other kind is refused thereafter. A
     * <strong>neutral</strong> {@code action} neither decides nor violates the lane: a paused,
     * replaying, or faulted service emits neutral actions on both lanes, and they command nothing.
     * That is exactly how the service's own {@code _latched_lane} treats them.
     *
     * <p>The refusal reuses {@code PROTOCOL_ERROR} rather than adding a {@link ControllerFault}
     * value, because the launcher's terminal-fault pattern is regression-pinned against this enum's
     * source and {@code MALFORMED} on the wire is an honest word for a contract violation.
     *
     * @return {@code false} when this frame's kind contradicts the latched lane
     */
    private boolean latchLane(CommandFrame frame) {
        FrameLane kind;
        if (frame instanceof ProgramActionFrame) {
            kind = FrameLane.PROGRAM;
        } else if (frame instanceof ActionFrame action && action.neutral()) {
            return true;
        } else {
            kind = FrameLane.MOTOR;
        }
        if (lane != FrameLane.UNDECIDED && lane != kind) {
            LOGGER.warn(
                    "lane mixing refused: {} in {} session (command_seq={})",
                    kind == FrameLane.PROGRAM ? "program frame" : "non-neutral action",
                    lane,
                    frame.commandSequence());
            return false;
        }
        lane = kind;
        return true;
    }

    /** Reports a candidate observed before an accepted RESET, inside its re-armed window. */
    private boolean retiresAtResetBoundary(CommandFrame frame, long currentGameTick) {
        return resetBoundaryTick >= 0
                && frame.observationTick() < resetBoundaryTick
                && currentGameTick - resetBoundaryTick <= FIRST_ACTION_DEADLINE_TICKS;
    }

    /**
     * Latest tick the session may pass without an applicable action before DEADLINE_MISSED.
     * Once an action has applied (or a RESET has re-armed the same anchor), the hold is
     * the applied validity window plus HOLD_GRACE_TICKS of fail-neutral grace (ADR 0013):
     * during grace ticks the fly stays rejected-ACTIVE with neutral motors, exactly as
     * within the window itself. Before any applied action this is the first-action
     * deadline plus COLD_START_GRACE_TICKS (ADR 0018), which makes the tolerated frameless
     * gap the same eight ticks on both sides of the first application; graced cold ticks
     * are rejected-ACTIVE with neutral motors exactly as the ungraced first two already
     * were. Lateness retirements still never extend or reset either deadline. A dormancy
     * re-anchor (ADR 0012) clears the applied anchor and therefore discards its grace, but
     * lands on the cold branch, which now carries a grace of its own — see ADR 0018's
     * dormancy-interaction clause, which supersedes ADR 0012's prose on that point.
     */
    private long actuationHoldThroughTick() {
        if (lastAppliedValidThroughTick >= 0) {
            return lastAppliedValidThroughTick > Long.MAX_VALUE - HOLD_GRACE_TICKS
                    ? Long.MAX_VALUE
                    : lastAppliedValidThroughTick + HOLD_GRACE_TICKS;
        }
        return coldStartHoldThroughTick();
    }

    /**
     * Single expression of the graced cold-start hold, shared by both evaluations that
     * enforce it (the pre-candidate first-action guard in validateChecked and the frameless
     * hold rule). Keeping one helper is the point: the two guards decide the same question
     * at different places in the pass, and before ADR 0018 they carried the same bound
     * written twice.
     */
    private long coldStartHoldThroughTick() {
        long span = FIRST_ACTION_DEADLINE_TICKS + COLD_START_GRACE_TICKS;
        return firstActiveGameTick > Long.MAX_VALUE - span
                ? Long.MAX_VALUE
                : firstActiveGameTick + span;
    }

    /**
     * Re-anchors the session after entity-tick dormancy instead of evaluating deadlines
     * against pre-gap state (ADR 0012). validate() runs on every entity tick while the
     * fly is live, so a gap above DORMANCY_GAP_TICKS can only mean the authority stopped
     * ticking the entity (chunk unload): nothing starved while it was actually live.
     * Every reserved frame is dropped fail-neutral (its target passed while no authority
     * could apply it) and the hold anchors reset exactly as a fresh activation would, so
     * the first-action deadline re-arms from this tick and genuine post-resume starvation
     * still fails closed — eight ticks after the re-anchor rather than three, because the
     * cold branch it lands on now carries COLD_START_GRACE_TICKS (ADR 0018). A re-anchor
     * therefore no longer necessarily shortens the tolerated gap relative to the grace it
     * discards; ADR 0018 records that consequence explicitly. Session sequence identity
     * persists: in-flight pre-gap frames
     * meet the unchanged ADR 0011 machinery in this same pass (verified-late retires,
     * unverifiable still latches STALE_COMMAND).
     */
    private void reanchorAfterDormancy(long currentGameTick) {
        pending.clear();
        firstActiveGameTick = currentGameTick;
        lastAppliedValidThroughTick = -1;
        dormancyReanchors++;
        TOTAL_DORMANCY_REANCHORS.incrementAndGet();
    }

    private void countLatenessRetirement() {
        latenessRetirements++;
        TOTAL_LATENESS_RETIREMENTS.incrementAndGet();
    }

    private ValidationResult reject(ControllerFault fault) {
        state = fault;
        pending.clear();
        return ValidationResult.rejected(fault);
    }

    private ValidationResult reject(ControllerFault fault, CommandFrame command) {
        state = fault;
        pending.clear();
        return ValidationResult.rejected(command, fault);
    }
}
