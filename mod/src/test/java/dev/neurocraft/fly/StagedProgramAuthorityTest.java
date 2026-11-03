package dev.neurocraft.fly;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.BodyProgramRuntime;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.MotorAuthority;
import dev.neurocraft.fly.body.SelectionSource;
import dev.neurocraft.fly.control.ControlAckFrame;
import dev.neurocraft.fly.control.ControlOperation;
import dev.neurocraft.fly.control.ControlState;
import dev.neurocraft.fly.control.ControllerIdentity;
import java.util.Optional;
import java.util.UUID;
import org.junit.jupiter.api.Test;

/**
 * The PAUSE/RESUME transaction against the acknowledgement the service actually produces.
 *
 * <p>The service stamps an accepted PAUSE or RESUME acknowledgement with the game tick of the
 * <em>observation</em> it first answered in the new state
 * ({@code services/neural-engine/src/neurocraft_service/server.py}, the action-effect branch of
 * {@code process_latest}: {@code effective_tick = observation.game_tick}). The mod sends that
 * observation from {@code FlyEntity.tick()} <em>after</em> it has already polled control
 * acknowledgements for the same tick, so no acknowledgement derived from observation {@code X}
 * can be polled before tick {@code X + 1}. These pins hold the transaction to that relation. The
 * pre-fix transaction demanded {@code currentTick == effectiveTick}, which the real timeline can
 * never satisfy, and failed every live PAUSE with {@code MISSED/LATE_ACK}; the GameTest pins did not
 * notice because they hand-built acknowledgements with {@code effectiveTick == currentTick}.
 */
final class StagedProgramAuthorityTest {
    private static final UUID ENTITY_UUID =
            UUID.fromString("00000000-0000-0000-0000-000000000001");
    private static final ControllerIdentity IDENTITY = new ControllerIdentity(
            "run-1",
            7,
            ENTITY_UUID.toString(),
            "1".repeat(64),
            "2".repeat(64),
            "3".repeat(64),
            "4".repeat(64),
            "5".repeat(64));

    /**
     * The common live timeline. The operator issues at tick 100, after observation 100 is already
     * on the wire; the service pauses and answers observation 101 neutral, stamping
     * {@code effective_tick = 101}; the mod first polls that acknowledgement at tick 102, and the
     * paused action it was issued beside targets 103 under the two-tick action budget. The
     * transaction must be READY on that tick -- the tick the neutral command governs -- because a
     * scripted program may only own a tick whose admitted command is neutral.
     */
    @Test
    void acknowledgementStampedWithTheNextObservationTickEngagesOneActionBudgetLater() {
        StagedProgramAuthority engage = engage(100);

        assertEquals(
                StagedProgramAuthority.Outcome.WAITING, engage.advance(100, Optional.empty()));
        assertEquals(
                StagedProgramAuthority.Outcome.WAITING, engage.advance(101, Optional.empty()));
        assertEquals(
                StagedProgramAuthority.Outcome.WAITING,
                engage.advance(102, Optional.of(pauseAck(100, 101, true))));
        assertEquals(101, engage.effectiveTick());
        assertEquals(103, engage.engageTick());
        assertEquals(StagedProgramAuthority.Outcome.READY, engage.advance(103, Optional.empty()));

        BodyProgramRuntime runtime = new BodyProgramRuntime();
        engage.apply(runtime, 103);
        assertEquals(MotorAuthority.SCRIPTED, runtime.authority());
        assertEquals(SelectionSource.OPERATOR, runtime.selectionSource());
        assertEquals(BodyProgramId.HOVER, runtime.state().id());
        assertEquals(103, runtime.state().enteredGameTick());
        assertEquals(StagedProgramAuthority.Outcome.APPLIED, engage.advance(104, Optional.empty()));
    }

    /**
     * The fast timeline. The control overtakes the still-queued observation 100 on the bridge
     * (controls drain ahead of the pending observation), so the service answers observation 100
     * itself neutral and stamps {@code effective_tick = 100}. The acknowledgement is still first
     * visible at tick 101, and the neutral action it rode beside targets 102, so the transaction
     * waits one more tick and is READY there, not MISSED.
     */
    @Test
    void acknowledgementStampedWithTheIssuingTickEngagesOneActionBudgetLater() {
        StagedProgramAuthority engage = engage(100);

        assertEquals(
                StagedProgramAuthority.Outcome.WAITING,
                engage.advance(101, Optional.of(pauseAck(100, 100, true))));
        assertEquals(100, engage.effectiveTick());
        assertEquals(102, engage.engageTick());
        assertEquals(StagedProgramAuthority.Outcome.READY, engage.advance(102, Optional.empty()));
    }

    /** An acknowledgement can never be applied on its own effective tick: it is not visible yet. */
    @Test
    void acknowledgementSeenOnItsEffectiveTickWaitsForTheEngageTick() {
        StagedProgramAuthority engage = engage(100);

        assertEquals(
                StagedProgramAuthority.Outcome.WAITING,
                engage.advance(100, Optional.of(pauseAck(100, 100, true))));
        assertEquals(102, engage.engageTick());
        assertThrows(
                IllegalStateException.class, () -> engage.apply(new BodyProgramRuntime(), 100));
        assertEquals(
                StagedProgramAuthority.Outcome.WAITING, engage.advance(101, Optional.empty()));
        assertEquals(StagedProgramAuthority.Outcome.READY, engage.advance(102, Optional.empty()));
    }

    /** The same budget an action has: a poll after the governed tick is terminal. */
    @Test
    void acknowledgementFirstSeenAfterItsEngageTickIsMissed() {
        StagedProgramAuthority engage = engage(100);

        assertEquals(
                StagedProgramAuthority.Outcome.MISSED,
                engage.advance(104, Optional.of(pauseAck(100, 101, true))));
        assertEquals("LATE_ACK", engage.failureCode());
        assertThrows(
                IllegalStateException.class, () -> engage.apply(new BodyProgramRuntime(), 104));
    }

    @Test
    void dormancyPastTheEngageTickResolvesMissedInsteadOfWaitingForever() {
        StagedProgramAuthority engage = engage(100);

        assertEquals(
                StagedProgramAuthority.Outcome.WAITING,
                engage.advance(100, Optional.of(pauseAck(100, 100, true))));
        assertEquals(StagedProgramAuthority.Outcome.MISSED, engage.advance(400, Optional.empty()));
        assertEquals("EFFECTIVE_TICK_PASSED", engage.failureCode());
    }

    @Test
    void noAcknowledgementWithinTheDeadlineIsMissed() {
        StagedProgramAuthority engage = engage(100);

        assertEquals(
                StagedProgramAuthority.Outcome.WAITING, engage.advance(102, Optional.empty()));
        assertEquals(
                StagedProgramAuthority.Outcome.MISSED, engage.advance(103, Optional.empty()));
        assertEquals("ACK_DEADLINE", engage.failureCode());
    }

    @Test
    void rejectedAndMisboundAcknowledgementsNeverEngage() {
        StagedProgramAuthority rejected = engage(100);
        assertEquals(
                StagedProgramAuthority.Outcome.REJECTED,
                rejected.advance(101, Optional.of(pauseAck(100, 100, false))));
        assertEquals("INVALID_STATE", rejected.failureCode());

        StagedProgramAuthority wrongSequence = engage(100);
        assertEquals(
                StagedProgramAuthority.Outcome.INVALID,
                wrongSequence.advance(
                        101, Optional.of(ack(1, 100, 100, ControlOperation.PAUSE, true))));
        assertEquals("ACK_BINDING", wrongSequence.failureCode());

        StagedProgramAuthority wrongOperation = engage(100);
        assertEquals(
                StagedProgramAuthority.Outcome.INVALID,
                wrongOperation.advance(
                        101, Optional.of(ack(0, 100, 100, ControlOperation.RESUME, true))));
        assertEquals("ACK_BINDING", wrongOperation.failureCode());
    }

    /**
     * RESUME is the mirror: the same relation, releasing instead of engaging. Authority is taken
     * through a PAUSE transaction first, because that transaction is the only production caller of
     * {@code engageScripted} and {@code BodyPackageBoundaryTest} pins the call-site set.
     */
    @Test
    void releaseFollowsTheSameRelationAndDropsAuthorityOnItsEngageTick() {
        BodyProgramRuntime runtime = new BodyProgramRuntime();
        StagedProgramAuthority engage = engage(90);
        assertEquals(
                StagedProgramAuthority.Outcome.WAITING,
                engage.advance(92, Optional.of(pauseAck(90, 91, true))));
        assertEquals(StagedProgramAuthority.Outcome.READY, engage.advance(93, Optional.empty()));
        engage.apply(runtime, 93);
        assertEquals(MotorAuthority.SCRIPTED, runtime.authority());

        StagedProgramAuthority release = StagedProgramAuthority.release(ENTITY_UUID, 100);
        release.bindControlSequence(1);
        assertEquals(
                StagedProgramAuthority.Outcome.WAITING, release.advance(101, Optional.empty()));
        assertEquals(
                StagedProgramAuthority.Outcome.WAITING,
                release.advance(
                        102, Optional.of(ack(1, 100, 101, ControlOperation.RESUME, true))));
        assertEquals(StagedProgramAuthority.Outcome.READY, release.advance(103, Optional.empty()));
        release.apply(runtime, 103);
        assertEquals(MotorAuthority.NEURAL, runtime.authority());
        assertEquals(SelectionSource.DEFAULT, runtime.selectionSource());
        assertEquals(
                "BODY PROGRAM HOVER -> HOVER tick=103 source=OPERATOR reason=released",
                runtime.transitionLedger().getLast());
    }

    private static StagedProgramAuthority engage(long issuedTick) {
        StagedProgramAuthority transaction = StagedProgramAuthority.engage(
                ENTITY_UUID, issuedTick, BodyProgramId.HOVER, Modulation.NEUTRAL);
        transaction.bindControlSequence(0);
        return transaction;
    }

    private static ControlAckFrame pauseAck(long issuedTick, long effectiveTick, boolean accepted) {
        return ack(0, issuedTick, effectiveTick, ControlOperation.PAUSE, accepted);
    }

    private static ControlAckFrame ack(
            long controlSequence,
            long issuedTick,
            long effectiveTick,
            ControlOperation operation,
            boolean accepted) {
        return new ControlAckFrame(
                IDENTITY,
                controlSequence,
                issuedTick,
                operation,
                null,
                null,
                accepted,
                accepted ? ControlState.PAUSED : ControlState.RUNNING,
                effectiveTick,
                accepted ? null : "INVALID_STATE");
    }
}
