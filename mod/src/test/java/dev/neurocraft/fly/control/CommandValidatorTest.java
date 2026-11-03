package dev.neurocraft.fly.control;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ProgramParameters;
import dev.neurocraft.fly.body.ReadoutValues;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.function.LongFunction;
import org.junit.jupiter.api.Test;

final class CommandValidatorTest {
    private static final ControllerIdentity IDENTITY = new ControllerIdentity(
            "run-1",
            7,
            "fly-01",
            "1".repeat(64),
            "2".repeat(64),
            "3".repeat(64),
            "4".repeat(64),
            "5".repeat(64));

    @Test
    void acceptsExactSequenceObservationReceiptAndCurrentTarget() {
        CommandValidator validator = new CommandValidator(IDENTITY);

        ValidationResult result = validator.validate(
                List.of(frame(0, 0, 100, 101, 102, 0.25)),
                101,
                ControllerFault.ACTIVE,
                lookup(Map.of(0L, 100L)));

        assertTrue(result.accepted());
        assertEquals(0.25, result.rawAction().forwardThrust());
        assertEquals(0.25, result.appliedAction().forwardThrust());
        assertEquals(0, validator.lastCommandSequence());
    }

    @Test
    void reservesNextTickWithoutApplyingEarly() {
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 100L));

        ValidationResult early = validator.validate(
                List.of(frame(0, 0, 100, 101, 102, 0.5)),
                100,
                ControllerFault.ACTIVE,
                receipts);
        ValidationResult due =
                validator.validate(List.of(), 101, ControllerFault.ACTIVE, receipts);

        assertFalse(early.accepted());
        assertEquals(ControllerFault.ACTIVE, early.state());
        assertTrue(due.accepted());
        assertEquals(0.5, due.appliedAction().forwardThrust());
    }

    @Test
    void twoCommandWindowPreservesIncreasingTargets() {
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 100L, 1L, 101L));

        ValidationResult first = validator.validate(
                List.of(
                        frame(0, 0, 100, 101, 102, 0.25),
                        frame(1, 1, 101, 102, 103, 0.75)),
                101,
                ControllerFault.ACTIVE,
                receipts);
        ValidationResult second =
                validator.validate(List.of(), 102, ControllerFault.ACTIVE, receipts);

        assertEquals(0.25, first.appliedAction().forwardThrust());
        assertEquals(0.75, second.appliedAction().forwardThrust());
        assertEquals(0, validator.pendingCommandCount());
    }

    @Test
    void waitingForHandshakeOrFirstActionIsNeutralButNotLatched() {
        CommandValidator validator = new CommandValidator(IDENTITY);

        ValidationResult handshake = validator.validate(
                List.of(), 100, ControllerFault.STARTING, lookup(Map.of()));
        ValidationResult firstAction = validator.validate(
                List.of(), 100, ControllerFault.ACTIVE, lookup(Map.of()));

        assertEquals(ControllerFault.STARTING, handshake.state());
        assertEquals(ControllerFault.ACTIVE, firstAction.state());
        assertTrue(firstAction.appliedAction().isNeutral());
        assertFalse(firstAction.accepted());
    }

    @Test
    void missingFirstActionAfterTwoActiveTicksLatchesDeadlineFault() {
        // Pins both boundary sides of the cold-start first-action deadline. ADR 0018
        // extended this timeline from 103 to 108: COLD_START_GRACE_TICKS = 5 makes the
        // graced cold hold run through 107, so the tolerated frameless gap is the same
        // eight ticks the applied-anchor hold already tolerated. No frame ever arrives
        // here, so the pre-candidate first-action guard is the branch under test.
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> noReceipts = lookup(Map.of());

        ValidationResult first = validator.validate(
                List.of(), 100, ControllerFault.ACTIVE, noReceipts);
        for (long tick = 101; tick <= 107; tick++) {
            ValidationResult graced =
                    validator.validate(List.of(), tick, ControllerFault.ACTIVE, noReceipts);
            assertEquals(ControllerFault.ACTIVE, graced.state());
            assertTrue(graced.appliedAction().isNeutral());
            assertFalse(graced.accepted());
        }
        ValidationResult missed = validator.validate(
                List.of(), 108, ControllerFault.ACTIVE, noReceipts);

        assertEquals(ControllerFault.ACTIVE, first.state());
        assertEquals(ControllerFault.DEADLINE_MISSED, missed.state());
        assertEquals(ControllerFault.DEADLINE_MISSED, validator.state());
        assertEquals(-1, validator.lastCommandSequence());
        assertEquals(0, validator.dormancyReanchors());
    }

    @Test
    void firstActionIsAcceptedAtTheInclusiveStartupDeadline() {
        // Still true and still pinned after ADR 0018 — the pre-grace boundary tick
        // accepts exactly as before. The new inclusive boundary (107) is pinned by
        // coldStartFramelessGapToTheGracedHoldStillAppliesTheFirstFrame.
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(2L, 102L));
        validator.validate(List.of(), 100, ControllerFault.ACTIVE, receipts);

        ValidationResult boundary = validator.validate(
                List.of(frame(0, 2, 102, 102, 104, 0.5)),
                102,
                ControllerFault.ACTIVE,
                receipts);

        assertTrue(boundary.accepted());
        assertEquals(ControllerFault.ACTIVE, boundary.state());
        assertEquals(0, validator.lastCommandSequence());
    }

    @Test
    void lateFreshActionCannotMaskMissedFirstActionDeadline() {
        // ADR 0018 extended this timeline from 103 to 108. The pin is structural and
        // unchanged: once the cold-start hold has lapsed, a frame that is itself
        // perfectly fresh and targets the current tick still cannot revive the session,
        // because the first-action guard runs before the candidate loop. Intermediate
        // ticks are spelled out so the ADR 0012 dormancy re-anchor stays out of the
        // scenario (every gap is one tick).
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(7L, 107L));
        for (long tick = 100; tick <= 107; tick++) {
            // Asserted, not merely driven: the session must still be alive on every
            // tick up to the inclusive boundary, so the latch below is provably the
            // lapsed hold rather than an earlier one this timeline drove past.
            assertEquals(
                    ControllerFault.ACTIVE,
                    validator.validate(List.of(), tick, ControllerFault.ACTIVE, receipts).state());
        }

        ValidationResult missed = validator.validate(
                List.of(frame(0, 7, 107, 108, 109, 0.5)),
                108,
                ControllerFault.ACTIVE,
                receipts);

        assertEquals(ControllerFault.DEADLINE_MISSED, missed.state());
        assertFalse(missed.accepted());
        assertTrue(missed.appliedAction().isNeutral());
        assertEquals(-1, validator.lastCommandSequence());
        assertEquals(0, validator.pendingCommandCount());
    }

    @Test
    void missingCommandAfterStreamBeginsLatchesDeadlineFault() {
        // ADR 0013 extended this timeline: the applied hold now carries HOLD_GRACE_TICKS
        // of fail-neutral grace, so the latch moved from 103 to 109. Both boundary sides
        // of the graced hold are pinned here. The two-tick action budget moved the frame,
        // not the timeline: it answers observation 99 and is valid through its own target
        // tick 101, so the anchor and the grace (now 7) still latch at 109.
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 99L));
        validator.validate(
                List.of(frame(0, 0, 99, 101, 101, 0.0)),
                101,
                ControllerFault.ACTIVE,
                receipts);

        ValidationResult withinDeclaredWindow =
                validator.validate(List.of(), 102, ControllerFault.ACTIVE, receipts);
        for (long tick = 103; tick <= 107; tick++) {
            ValidationResult graced =
                    validator.validate(List.of(), tick, ControllerFault.ACTIVE, receipts);
            assertEquals(ControllerFault.ACTIVE, graced.state());
            assertTrue(graced.appliedAction().isNeutral());
        }
        ValidationResult graceBoundary =
                validator.validate(List.of(), 108, ControllerFault.ACTIVE, receipts);
        ValidationResult missing =
                validator.validate(List.of(), 109, ControllerFault.ACTIVE, receipts);

        assertEquals(ControllerFault.ACTIVE, withinDeclaredWindow.state());
        assertFalse(withinDeclaredWindow.accepted());
        assertTrue(withinDeclaredWindow.appliedAction().isNeutral());
        assertEquals(ControllerFault.ACTIVE, graceBoundary.state());
        assertFalse(graceBoundary.accepted());
        assertTrue(graceBoundary.appliedAction().isNeutral());
        assertEquals(ControllerFault.DEADLINE_MISSED, missing.state());
        assertEquals(ControllerFault.DEADLINE_MISSED, validator.state());
    }

    @Test
    void latestValueCoalescingCanRecoverAfterOneNeutralGap() {
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 100L, 2L, 102L));

        ValidationResult first = validator.validate(
                List.of(frame(0, 0, 100, 101, 102, 0.25)),
                101,
                ControllerFault.ACTIVE,
                receipts);
        ValidationResult coalescedGap =
                validator.validate(List.of(), 102, ControllerFault.ACTIVE, receipts);
        ValidationResult recovered = validator.validate(
                List.of(frame(1, 2, 102, 103, 104, 0.75)),
                103,
                ControllerFault.ACTIVE,
                receipts);

        assertTrue(first.accepted());
        assertEquals(ControllerFault.ACTIVE, coalescedGap.state());
        assertFalse(coalescedGap.accepted());
        assertTrue(coalescedGap.appliedAction().isNeutral());
        assertTrue(recovered.accepted());
        assertEquals(0.75, recovered.appliedAction().forwardThrust());
        assertEquals(1, validator.lastCommandSequence());
    }

    @Test
    void futureCommandCannotHideAnExpiredCurrentTickDeadline() {
        // ADR 0013 extended this timeline: the original arrival at 103 now falls
        // inside the graced hold (through 108), where reserving a next-tick frame is
        // the intended recovery. The pinned property is unchanged past the grace: a
        // frame that skips the current tick cannot mask a lapsed graced hold.
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 99L, 3L, 108L));

        validator.validate(
                List.of(frame(0, 0, 99, 101, 101, 0.25)),
                101,
                ControllerFault.ACTIVE,
                receipts);
        for (long tick = 102; tick <= 108; tick++) {
            ValidationResult graced =
                    validator.validate(List.of(), tick, ControllerFault.ACTIVE, receipts);
            assertEquals(ControllerFault.ACTIVE, graced.state());
        }
        ValidationResult skippedCurrentTick = validator.validate(
                List.of(frame(1, 3, 108, 110, 110, 0.75)),
                109,
                ControllerFault.ACTIVE,
                receipts);

        assertEquals(ControllerFault.DEADLINE_MISSED, skippedCurrentTick.state());
        assertFalse(skippedCurrentTick.accepted());
        assertTrue(skippedCurrentTick.appliedAction().isNeutral());
        assertEquals(0, validator.pendingCommandCount());
    }

    @Test
    void acceptedResetClearsReservedActionsButPreservesSessionSequence() {
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts =
                lookup(Map.of(0L, 100L, 1L, 101L, 2L, 102L));
        validator.validate(
                List.of(
                        frame(0, 0, 100, 101, 102, 0.25),
                        frame(1, 1, 101, 102, 103, 0.5)),
                101,
                ControllerFault.ACTIVE,
                receipts);

        validator.resetActuationWindow(101);
        ValidationResult afterReset = validator.validate(
                List.of(frame(2, 2, 102, 102, 104, 0.75)),
                102,
                ControllerFault.ACTIVE,
                receipts);

        assertEquals(0, validator.pendingCommandCount());
        assertTrue(afterReset.accepted());
        assertEquals(2, validator.lastCommandSequence());
    }

    @Test
    void acceptedResetRetiresOlderQueuedActionWithoutFailingClosed() {
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts =
                lookup(Map.of(0L, 100L, 1L, 101L, 2L, 104L));
        validator.validate(
                List.of(frame(0, 0, 100, 101, 102, 0.25)),
                101,
                ControllerFault.ACTIVE,
                receipts);

        validator.resetActuationWindow(103);
        ValidationResult retired = validator.validate(
                List.of(frame(1, 1, 101, 102, 103, 0.5)),
                103,
                ControllerFault.ACTIVE,
                receipts);
        ValidationResult fresh = validator.validate(
                List.of(frame(2, 2, 104, 104, 105, 0.75)),
                104,
                ControllerFault.ACTIVE,
                receipts);

        assertFalse(retired.accepted());
        assertEquals(ControllerFault.ACTIVE, retired.state());
        assertFalse(validator.state().isPaused());
        assertTrue(retired.appliedAction().isNeutral());
        assertEquals(0.0, retired.rawAction().forwardThrust());
        assertTrue(retired.command().isEmpty());
        assertEquals(0, validator.pendingCommandCount());
        assertTrue(fresh.accepted());
        assertEquals(0.75, fresh.appliedAction().forwardThrust());
        assertEquals(2, validator.lastCommandSequence());
    }

    @Test
    void actionObservedAtTheResetBoundaryStillValidatesNormally() {
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 100L, 1L, 103L));
        validator.validate(
                List.of(frame(0, 0, 100, 101, 102, 0.25)),
                101,
                ControllerFault.ACTIVE,
                receipts);

        validator.resetActuationWindow(103);
        ValidationResult boundary = validator.validate(
                List.of(frame(1, 1, 103, 103, 105, 0.5)),
                103,
                ControllerFault.ACTIVE,
                receipts);

        assertTrue(boundary.accepted());
        assertEquals(0.5, boundary.appliedAction().forwardThrust());
        assertEquals(1, validator.lastCommandSequence());
    }

    @Test
    void unverifiablePostResetActionStillLatchesFailClosed() {
        // ADR 0011 updated this pin: a fully verified late post-reset frame now retires,
        // so the fail-closed boundary is unverifiability. The frame's observation receipt
        // is absent from the ledger, which still latches STALE_COMMAND. Intermediate
        // ticks are spelled out so the ADR 0012 dormancy re-anchor stays out of this
        // pin's tick-continuous scenario.
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 100L));
        validator.validate(
                List.of(frame(0, 0, 100, 101, 102, 0.25)),
                101,
                ControllerFault.ACTIVE,
                receipts);

        validator.resetActuationWindow(103);
        validator.validate(List.of(), 103, ControllerFault.ACTIVE, receipts);
        validator.validate(List.of(), 104, ControllerFault.ACTIVE, receipts);
        ValidationResult stale = validator.validate(
                List.of(frame(1, 1, 103, 104, 105, 0.5)),
                105,
                ControllerFault.ACTIVE,
                receipts);

        assertEquals(ControllerFault.STALE_COMMAND, stale.state());
        assertEquals(ControllerFault.STALE_COMMAND, validator.state());
        assertFalse(stale.accepted());
        assertTrue(stale.appliedAction().isNeutral());
        assertEquals(0, validator.pendingCommandCount());
        assertEquals(0, validator.latenessRetirements());
        assertEquals(0, validator.dormancyReanchors());
    }

    @Test
    void verifiedLatePostResetFrameRetiresLikeAnyVerifiedLateFrame() {
        // Re-pins the exact scenario the pre-ADR-0011 test named: a fully
        // verified post-reset frame whose only defect is a passed target now
        // retires fail-neutral instead of latching STALE_COMMAND. Intermediate
        // ticks are spelled out so the ADR 0012 dormancy re-anchor stays out of
        // this pin's tick-continuous scenario.
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 100L, 1L, 103L));
        validator.validate(
                List.of(frame(0, 0, 100, 101, 102, 0.25)),
                101,
                ControllerFault.ACTIVE,
                receipts);

        validator.resetActuationWindow(103);
        validator.validate(List.of(), 103, ControllerFault.ACTIVE, receipts);
        validator.validate(List.of(), 104, ControllerFault.ACTIVE, receipts);
        ValidationResult retired = validator.validate(
                List.of(frame(1, 1, 103, 104, 105, 0.5)),
                105,
                ControllerFault.ACTIVE,
                receipts);

        assertEquals(ControllerFault.ACTIVE, retired.state());
        assertEquals(ControllerFault.ACTIVE, validator.state());
        assertFalse(retired.accepted());
        assertTrue(retired.appliedAction().isNeutral());
        assertEquals(0, validator.pendingCommandCount());
        assertEquals(1, validator.latenessRetirements());
        assertEquals(0, validator.dormancyReanchors());
    }

    @Test
    void totalLatenessRetirementsResetsToZero() {
        // The SERVER_STOPPING registration in NeuroCraftFly is a single frozen call
        // site, so the one reset hook zeroes every process-global validator counter:
        // lateness retirements (ADR 0011) and dormancy re-anchors (ADR 0012).
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 100L));
        validator.validate(
                List.of(frame(0, 0, 100, 101, 102, 0.25)),
                105,
                ControllerFault.ACTIVE,
                receipts);
        validator.validate(
                List.of(),
                105 + CommandValidator.DORMANCY_GAP_TICKS + 1,
                ControllerFault.ACTIVE,
                receipts);
        assertTrue(CommandValidator.totalLatenessRetirements() >= 1);
        assertTrue(CommandValidator.totalDormancyReanchors() >= 1);
        CommandValidator.resetTotalLatenessRetirements();
        assertEquals(0, CommandValidator.totalLatenessRetirements());
        assertEquals(0, CommandValidator.totalDormancyReanchors());
    }

    @Test
    void preResetActionAfterTheResetWindowRetiresButLapsedDeadlineStillLatches() {
        // ADR 0011 updated this pin: the verified pre-reset frame arriving after the
        // re-armed window used to latch STALE_COMMAND; its only defect is lateness, so it
        // now retires fail-neutral. The session still fails closed in the same pass
        // because the reset's actuation hold has lapsed with no applicable action:
        // deadline enforcement is unchanged in structure. ADR 0012 requires the
        // tick-continuity this scenario always assumed to be modeled explicitly, so the
        // intermediate empty ticks are spelled out. ADR 0013 extended the timeline
        // again: the re-armed reset hold is expressed through the applied anchor
        // (effective tick + 2) and therefore carries HOLD_GRACE_TICKS, so the frame
        // arrival moved from 106 to 112, past the graced hold. The two-tick action
        // budget moves this one timeline by one further tick, and deliberately: the
        // re-armed anchor is still effectiveTick + FIRST_ACTION_DEADLINE_TICKS, which is
        // now exactly the earliest tick a post-reset frame can target (observation 103
        // plus the budget), and the grace is 7. The hold therefore runs through 112 and
        // the arrival moves to 113. A post-reset session gets the same number of
        // attempts it had before, one tick later, because its first frame is due one
        // tick later.
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 99L, 1L, 100L));
        validator.validate(
                List.of(frame(0, 0, 99, 101, 101, 0.25)),
                101,
                ControllerFault.ACTIVE,
                receipts);

        validator.resetActuationWindow(103);
        for (long tick = 103; tick <= 112; tick++) {
            ValidationResult held =
                    validator.validate(List.of(), tick, ControllerFault.ACTIVE, receipts);
            assertEquals(ControllerFault.ACTIVE, held.state());
        }
        ValidationResult lapsed = validator.validate(
                List.of(frame(1, 1, 100, 102, 102, 0.5)),
                113,
                ControllerFault.ACTIVE,
                receipts);

        assertEquals(ControllerFault.DEADLINE_MISSED, lapsed.state());
        assertEquals(ControllerFault.DEADLINE_MISSED, validator.state());
        assertFalse(lapsed.accepted());
        assertTrue(lapsed.appliedAction().isNeutral());
        assertEquals(1, validator.lastCommandSequence());
        assertEquals(1, validator.latenessRetirements());
        assertEquals(0, validator.dormancyReanchors());
    }

    @Test
    void rejectsSequenceStaleFutureWrongBindingAndWrongIdentity() {
        assertRejected(
                ControllerFault.SEQUENCE_ERROR,
                frame(1, 0, 100, 101, 102, 0),
                101,
                Map.of(0L, 100L));
        // ADR 0011 updated this sub-case: with its receipt present the late frame
        // retires, so STALE_COMMAND is pinned on the unverifiable (missing-receipt)
        // form of the same frame.
        assertRejected(
                ControllerFault.STALE_COMMAND,
                frame(0, 0, 100, 101, 102, 0),
                101,
                Map.of());
        // One whole action budget ahead of the clock is legal now; two ticks past the
        // budget is the clock inconsistency.
        assertRejected(
                ControllerFault.FUTURE_COMMAND,
                frame(0, 0, 100, 102, 102, 0),
                99,
                Map.of(0L, 100L));
        assertRejected(
                ControllerFault.IDENTITY_MISMATCH,
                frame(0, 0, 99, 100, 101, 0),
                100,
                Map.of(0L, 100L));

        ControllerIdentity wrong = new ControllerIdentity(
                "run-1",
                7,
                "fly-01",
                "a".repeat(64),
                "2".repeat(64),
                "3".repeat(64),
                "4".repeat(64),
                "5".repeat(64));
        ActionFrame wrongIdentity = new ActionFrame(
                wrong,
                0,
                0,
                100,
                101,
                102,
                0,
                5,
                "6".repeat(64),
                RawPrimitiveAction.ZERO,
                true);
        assertRejected(
                ControllerFault.IDENTITY_MISMATCH,
                wrongIdentity,
                101,
                Map.of(0L, 100L));
    }

    @Test
    void firstFaultCannotSilentlyResume() {
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 100L));
        validator.validate(
                List.of(frame(1, 0, 100, 101, 102, 0)),
                101,
                ControllerFault.ACTIVE,
                receipts);

        ValidationResult later = validator.validate(
                List.of(frame(0, 0, 100, 101, 102, 0)),
                101,
                ControllerFault.ACTIVE,
                receipts);

        assertEquals(ControllerFault.SEQUENCE_ERROR, later.state());
        assertFalse(later.accepted());
    }

    @Test
    void verifiedFrameOlderThanMaxAgeRetiresAndFreshFrameStillApplies() {
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts =
                lookup(Map.of(0L, 100L, 1L, 101L, 2L, 104L));
        validator.validate(
                List.of(frame(0, 0, 100, 101, 102, 0.25)),
                101,
                ControllerFault.ACTIVE,
                receipts);

        ValidationResult burst = validator.validate(
                List.of(
                        frame(1, 1, 101, 102, 103, 0.5),
                        frame(2, 2, 104, 104, 106, 0.75)),
                104,
                ControllerFault.ACTIVE,
                receipts);

        assertTrue(burst.accepted());
        assertEquals(ControllerFault.ACTIVE, validator.state());
        assertEquals(0.75, burst.appliedAction().forwardThrust());
        assertEquals(2, validator.lastCommandSequence());
        assertEquals(1, validator.latenessRetirements());
        assertEquals(0, validator.pendingCommandCount());
    }

    @Test
    void verifiedFrameWithPassedTargetRetiresWithoutFault() {
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts =
                lookup(Map.of(0L, 100L, 1L, 101L, 2L, 103L));
        validator.validate(
                List.of(frame(0, 0, 100, 101, 102, 0.25)),
                101,
                ControllerFault.ACTIVE,
                receipts);

        ValidationResult retired = validator.validate(
                List.of(frame(1, 1, 101, 101, 103, 0.5)),
                102,
                ControllerFault.ACTIVE,
                receipts);
        ValidationResult fresh = validator.validate(
                List.of(frame(2, 2, 103, 103, 105, 0.75)),
                103,
                ControllerFault.ACTIVE,
                receipts);

        assertFalse(retired.accepted());
        assertEquals(ControllerFault.ACTIVE, retired.state());
        assertFalse(validator.state().isPaused());
        assertTrue(retired.appliedAction().isNeutral());
        assertEquals(1, validator.latenessRetirements());
        assertTrue(fresh.accepted());
        assertEquals(0.75, fresh.appliedAction().forwardThrust());
        assertEquals(2, validator.lastCommandSequence());
    }

    @Test
    void queuedFrameExpiredWithinHoldRetiresWithoutFault() {
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 100L, 1L, 100L));
        validator.validate(
                List.of(
                        frame(0, 0, 100, 100, 102, 0.25),
                        frame(1, 1, 100, 101, 102, 0.5)),
                100,
                ControllerFault.ACTIVE,
                receipts);

        ValidationResult afterBurst =
                validator.validate(List.of(), 102, ControllerFault.ACTIVE, receipts);

        assertFalse(afterBurst.accepted());
        assertEquals(ControllerFault.ACTIVE, afterBurst.state());
        assertFalse(validator.state().isPaused());
        assertEquals(1, validator.latenessRetirements());
        assertEquals(0, validator.pendingCommandCount());
        assertEquals(1, validator.lastCommandSequence());
    }

    @Test
    void queuedFrameExpiredBeyondHoldRetiresAndDeadlineStillLatches() {
        // ADR 0013 extended this timeline: the applied hold now reaches 107 with grace,
        // so the original single 3-tick skip landed inside the grace window. The
        // same-pass pin is preserved past the grace: a queued frame reserved late in the
        // grace window expires across a legal 3-tick validate skip, retires fail-neutral,
        // and the lapsed graced hold latches in that same pass. Both frames obey the
        // two-tick action budget, so each is valid through exactly its own target tick.
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 98L, 1L, 105L));
        validator.validate(
                List.of(frame(0, 0, 98, 100, 100, 0.25)),
                100,
                ControllerFault.ACTIVE,
                receipts);
        validator.validate(List.of(), 103, ControllerFault.ACTIVE, receipts);
        ValidationResult reserved = validator.validate(
                List.of(frame(1, 1, 105, 107, 107, 0.5)),
                106,
                ControllerFault.ACTIVE,
                receipts);
        assertEquals(ControllerFault.ACTIVE, reserved.state());
        assertEquals(1, validator.pendingCommandCount());

        ValidationResult lapsed =
                validator.validate(List.of(), 109, ControllerFault.ACTIVE, receipts);

        assertEquals(ControllerFault.DEADLINE_MISSED, lapsed.state());
        assertEquals(ControllerFault.DEADLINE_MISSED, validator.state());
        assertEquals(1, validator.latenessRetirements());
        assertEquals(0, validator.pendingCommandCount());
    }

    @Test
    void expiredQueuedHeadRetiresAndFreshFrameAppliesInTheSamePass() {
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts =
                lookup(Map.of(0L, 100L, 1L, 100L, 2L, 102L));
        validator.validate(
                List.of(
                        frame(0, 0, 100, 100, 102, 0.25),
                        frame(1, 1, 100, 101, 102, 0.5)),
                100,
                ControllerFault.ACTIVE,
                receipts);

        ValidationResult recovered = validator.validate(
                List.of(frame(2, 2, 102, 102, 104, 0.75)),
                102,
                ControllerFault.ACTIVE,
                receipts);

        assertTrue(recovered.accepted());
        assertEquals(0.75, recovered.appliedAction().forwardThrust());
        assertEquals(1, validator.latenessRetirements());
        assertEquals(0, validator.pendingCommandCount());
        assertEquals(2, validator.lastCommandSequence());
    }

    @Test
    void missingReceiptFrameStillLatchesStaleCommand() {
        CommandValidator validator = new CommandValidator(IDENTITY);

        ValidationResult stale = validator.validate(
                List.of(frame(0, 0, 100, 101, 102, 0.5)),
                101,
                ControllerFault.ACTIVE,
                lookup(Map.of()));

        assertEquals(ControllerFault.STALE_COMMAND, stale.state());
        assertEquals(ControllerFault.STALE_COMMAND, validator.state());
        assertEquals(0, validator.latenessRetirements());
        assertEquals(-1, validator.lastCommandSequence());
    }

    @Test
    void futureFramesStillLatchFutureCommand() {
        CommandValidator futureTarget = new CommandValidator(IDENTITY);
        ValidationResult target = futureTarget.validate(
                List.of(frame(0, 0, 100, 102, 102, 0.5)),
                99,
                ControllerFault.ACTIVE,
                lookup(Map.of(0L, 100L)));
        assertEquals(ControllerFault.FUTURE_COMMAND, target.state());
        assertEquals(ControllerFault.FUTURE_COMMAND, futureTarget.state());
        assertEquals(0, futureTarget.latenessRetirements());

        CommandValidator futureObservation = new CommandValidator(IDENTITY);
        ValidationResult observation = futureObservation.validate(
                List.of(frame(0, 0, 100, 101, 102, 0.5)),
                99,
                ControllerFault.ACTIVE,
                lookup(Map.of(0L, 100L)));
        assertEquals(ControllerFault.FUTURE_COMMAND, observation.state());
        assertEquals(0, futureObservation.latenessRetirements());
    }

    @Test
    void sequenceGapStillLatchesSequenceError() {
        CommandValidator validator = new CommandValidator(IDENTITY);

        ValidationResult gap = validator.validate(
                List.of(frame(1, 0, 100, 101, 102, 0.5)),
                101,
                ControllerFault.ACTIVE,
                lookup(Map.of(0L, 100L)));

        assertEquals(ControllerFault.SEQUENCE_ERROR, gap.state());
        assertEquals(ControllerFault.SEQUENCE_ERROR, validator.state());
        assertEquals(0, validator.latenessRetirements());
    }

    @Test
    void lateFrameWithNeuralStepRegressionStillLatchesSequenceError() {
        // The retirement gate requires the frame's sole defect to be lateness; a
        // neural-step regression is a stream inconsistency and still fails closed.
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 100L, 1L, 101L));
        validator.validate(
                List.of(frame(0, 0, 100, 101, 102, 0.25)),
                101,
                ControllerFault.ACTIVE,
                receipts);

        ActionFrame regressed = new ActionFrame(
                IDENTITY,
                1,
                1,
                101,
                102,
                103,
                3,
                8,
                "6".repeat(64),
                new RawPrimitiveAction(0.5, 0, 0, 0, 0),
                false);
        ValidationResult regression =
                validator.validate(List.of(regressed), 104, ControllerFault.ACTIVE, receipts);

        assertEquals(ControllerFault.SEQUENCE_ERROR, regression.state());
        assertEquals(ControllerFault.SEQUENCE_ERROR, validator.state());
        assertEquals(0, validator.latenessRetirements());
    }

    @Test
    void retirementDoesNotResetTheFirstActionDeadline() {
        // ADR 0018 extended this timeline from 103 to 108: the cold hold now runs
        // through 107. The pin itself is unchanged and is the reason the timeline can
        // move at all — a lateness retirement does not re-anchor anything, so the
        // deadline is still measured from activation (100), not from the retirement.
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 98L));
        validator.validate(List.of(), 100, ControllerFault.ACTIVE, receipts);

        ValidationResult retired = validator.validate(
                List.of(frame(0, 0, 98, 99, 100, 0.5)),
                101,
                ControllerFault.ACTIVE,
                receipts);
        for (long tick = 102; tick <= 107; tick++) {
            assertEquals(
                    ControllerFault.ACTIVE,
                    validator.validate(List.of(), tick, ControllerFault.ACTIVE, receipts)
                            .state());
        }
        ValidationResult missed =
                validator.validate(List.of(), 108, ControllerFault.ACTIVE, receipts);

        assertEquals(ControllerFault.ACTIVE, retired.state());
        assertEquals(1, validator.latenessRetirements());
        assertEquals(ControllerFault.DEADLINE_MISSED, missed.state());
        assertEquals(ControllerFault.DEADLINE_MISSED, validator.state());
        assertEquals(0, validator.dormancyReanchors());
    }

    @Test
    void latenessRetirementCounterCountsBothAcceptanceAndQueuePaths() {
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts =
                lookup(Map.of(0L, 100L, 1L, 101L, 2L, 104L, 3L, 104L));
        long totalBefore = CommandValidator.totalLatenessRetirements();
        validator.validate(
                List.of(frame(0, 0, 100, 101, 102, 0.25)),
                101,
                ControllerFault.ACTIVE,
                receipts);

        ValidationResult burst = validator.validate(
                List.of(
                        frame(1, 1, 101, 102, 103, 0.5),
                        frame(2, 2, 104, 104, 106, 0.75),
                        frame(3, 3, 104, 105, 106, 0.5)),
                104,
                ControllerFault.ACTIVE,
                receipts);
        ValidationResult queuedExpiry =
                validator.validate(List.of(), 106, ControllerFault.ACTIVE, receipts);

        assertTrue(burst.accepted());
        assertEquals(0.75, burst.appliedAction().forwardThrust());
        assertEquals(ControllerFault.ACTIVE, queuedExpiry.state());
        assertFalse(validator.state().isPaused());
        assertEquals(2, validator.latenessRetirements());
        assertEquals(totalBefore + 2, CommandValidator.totalLatenessRetirements());
    }

    @Test
    void gapAtTheDormancyBoundaryDoesNotReanchorAndStillLatchesDeadline() {
        // A gap of exactly DORMANCY_GAP_TICKS stays inside today's semantics: no
        // re-anchor, and the lapsed hold latches DEADLINE_MISSED against pre-gap
        // anchors. ADR 0013 extended this timeline: the single boundary gap used to
        // land past the hold (103) but now lands inside the grace window (through
        // 108), so three consecutive boundary gaps are chained — none re-anchors,
        // and the third lands past the graced hold and latches.
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 100L));
        validator.validate(
                List.of(frame(0, 0, 100, 101, 102, 0.25)),
                101,
                ControllerFault.ACTIVE,
                receipts);

        ValidationResult firstGap = validator.validate(
                List.of(), 104, ControllerFault.ACTIVE, receipts);
        ValidationResult secondGap = validator.validate(
                List.of(), 107, ControllerFault.ACTIVE, receipts);
        ValidationResult lapsed = validator.validate(
                List.of(), 110, ControllerFault.ACTIVE, receipts);

        assertEquals(ControllerFault.ACTIVE, firstGap.state());
        assertEquals(ControllerFault.ACTIVE, secondGap.state());
        assertEquals(ControllerFault.DEADLINE_MISSED, lapsed.state());
        assertEquals(ControllerFault.DEADLINE_MISSED, validator.state());
        assertEquals(0, validator.dormancyReanchors());
    }

    @Test
    void gapPastTheDormancyBoundaryReanchorsRetiresPendingAndResumesCleanly() {
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts =
                lookup(Map.of(0L, 100L, 1L, 100L, 2L, 105L));
        ValidationResult applied = validator.validate(
                List.of(
                        frame(0, 0, 100, 100, 102, 0.25),
                        frame(1, 1, 100, 101, 102, 0.5)),
                100,
                ControllerFault.ACTIVE,
                receipts);
        assertTrue(applied.accepted());
        assertEquals(1, validator.pendingCommandCount());

        ValidationResult resumed = validator.validate(
                List.of(),
                100 + CommandValidator.DORMANCY_GAP_TICKS + 1,
                ControllerFault.ACTIVE,
                receipts);
        ValidationResult fresh = validator.validate(
                List.of(frame(2, 2, 105, 105, 107, 0.75)),
                105,
                ControllerFault.ACTIVE,
                receipts);

        assertEquals(ControllerFault.ACTIVE, resumed.state());
        assertFalse(resumed.accepted());
        assertTrue(resumed.appliedAction().isNeutral());
        assertEquals(0, validator.pendingCommandCount());
        assertEquals(1, validator.dormancyReanchors());
        assertEquals(0, validator.latenessRetirements());
        assertTrue(fresh.accepted());
        assertEquals(0.75, fresh.appliedAction().forwardThrust());
        assertEquals(2, validator.lastCommandSequence());
    }

    @Test
    void reanchorRearmsTheFirstActionDeadlineAgainstPostResumeStarvation() {
        // The re-anchor is not a deadline waiver: it re-arms the first-action hold
        // from the re-anchor tick, so genuine post-resume starvation still fails
        // closed. ADR 0018 extended this timeline from 109 to 114 — the re-armed hold
        // is the cold branch, which now carries COLD_START_GRACE_TICKS, so it runs
        // through 113 (re-anchor at 106 + 2 + 5). The pin is the re-arming itself:
        // the deadline is measured from 106, not from the pre-gap anchors.
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 100L));
        validator.validate(
                List.of(frame(0, 0, 100, 101, 102, 0.25)),
                101,
                ControllerFault.ACTIVE,
                receipts);

        ValidationResult resumed =
                validator.validate(List.of(), 106, ControllerFault.ACTIVE, receipts);
        for (long tick = 107; tick <= 113; tick++) {
            assertEquals(
                    ControllerFault.ACTIVE,
                    validator.validate(List.of(), tick, ControllerFault.ACTIVE, receipts)
                            .state());
        }
        ValidationResult missed =
                validator.validate(List.of(), 114, ControllerFault.ACTIVE, receipts);

        assertEquals(ControllerFault.ACTIVE, resumed.state());
        assertEquals(ControllerFault.DEADLINE_MISSED, missed.state());
        assertEquals(ControllerFault.DEADLINE_MISSED, validator.state());
        assertEquals(1, validator.dormancyReanchors());
    }

    @Test
    void sequenceIdentityPersistsAcrossTheDormancyGap() {
        // The re-anchor resets hold anchors, never stream identity: the next frame
        // must still be exactly last+1, and a gapped sequence still latches.
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 100L, 2L, 106L));
        validator.validate(
                List.of(frame(0, 0, 100, 101, 102, 0.25)),
                101,
                ControllerFault.ACTIVE,
                receipts);

        ValidationResult gapped = validator.validate(
                List.of(frame(2, 2, 106, 106, 108, 0.75)),
                106,
                ControllerFault.ACTIVE,
                receipts);

        assertEquals(ControllerFault.SEQUENCE_ERROR, gapped.state());
        assertEquals(ControllerFault.SEQUENCE_ERROR, validator.state());
        assertFalse(gapped.accepted());
        assertEquals(1, validator.dormancyReanchors());
        assertEquals(0, validator.lastCommandSequence());
    }

    @Test
    void unverifiableInFlightFrameAfterReanchorStillLatchesStaleCommand() {
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 100L));
        validator.validate(
                List.of(frame(0, 0, 100, 101, 102, 0.25)),
                101,
                ControllerFault.ACTIVE,
                receipts);

        ValidationResult stale = validator.validate(
                List.of(frame(1, 1, 101, 102, 103, 0.5)),
                106,
                ControllerFault.ACTIVE,
                receipts);

        assertEquals(ControllerFault.STALE_COMMAND, stale.state());
        assertEquals(ControllerFault.STALE_COMMAND, validator.state());
        assertFalse(stale.accepted());
        assertTrue(stale.appliedAction().isNeutral());
        assertEquals(1, validator.dormancyReanchors());
        assertEquals(0, validator.latenessRetirements());
    }

    @Test
    void verifiedPreGapInFlightFrameRetiresUnderLatenessAfterReanchor() {
        // No observations are sent while the entity is dormant, so a pre-gap
        // in-flight frame's receipt is still in the bounded ledger: it is verified,
        // its only defect is lateness, and the ADR 0011 machinery retires it.
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 100L, 1L, 101L));
        validator.validate(
                List.of(frame(0, 0, 100, 101, 102, 0.25)),
                101,
                ControllerFault.ACTIVE,
                receipts);

        ValidationResult retired = validator.validate(
                List.of(frame(1, 1, 101, 102, 103, 0.5)),
                106,
                ControllerFault.ACTIVE,
                receipts);

        assertEquals(ControllerFault.ACTIVE, retired.state());
        assertFalse(retired.accepted());
        assertTrue(retired.appliedAction().isNeutral());
        assertEquals(1, validator.dormancyReanchors());
        assertEquals(1, validator.latenessRetirements());
        assertEquals(1, validator.lastCommandSequence());
    }

    @Test
    void framelessGapToExactlyTheGracedHoldStaysNeutralAndFreshFrameApplies() {
        // ADR 0013: with the applied frame valid through 102, the graced hold runs
        // through 108 (HOLD_GRACE_TICKS = 6, pinned by the literals here). Every
        // frameless tick through the inclusive boundary is fail-neutral
        // rejected-ACTIVE, and a fresh valid frame afterwards applies cleanly.
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 100L, 1L, 109L));
        validator.validate(
                List.of(frame(0, 0, 100, 101, 102, 0.25)),
                101,
                ControllerFault.ACTIVE,
                receipts);

        for (long tick = 102; tick <= 108; tick++) {
            ValidationResult graced =
                    validator.validate(List.of(), tick, ControllerFault.ACTIVE, receipts);
            assertEquals(ControllerFault.ACTIVE, graced.state());
            assertFalse(graced.accepted());
            assertTrue(graced.appliedAction().isNeutral());
        }
        ValidationResult fresh = validator.validate(
                List.of(frame(1, 1, 109, 109, 111, 0.75)),
                109,
                ControllerFault.ACTIVE,
                receipts);

        assertTrue(fresh.accepted());
        assertEquals(0.75, fresh.appliedAction().forwardThrust());
        assertEquals(ControllerFault.ACTIVE, validator.state());
        assertEquals(1, validator.lastCommandSequence());
        assertEquals(0, validator.latenessRetirements());
        assertEquals(0, validator.dormancyReanchors());
    }

    @Test
    void framelessGapPastTheGracedHoldLatchesDeadlineMissed() {
        // ADR 0013: one tick past the graced hold (108 + 1) with no applicable
        // action still fails closed as DEADLINE_MISSED.
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 99L));
        validator.validate(
                List.of(frame(0, 0, 99, 101, 101, 0.25)),
                101,
                ControllerFault.ACTIVE,
                receipts);

        for (long tick = 102; tick <= 108; tick++) {
            assertEquals(
                    ControllerFault.ACTIVE,
                    validator.validate(List.of(), tick, ControllerFault.ACTIVE, receipts)
                            .state());
        }
        ValidationResult starved =
                validator.validate(List.of(), 109, ControllerFault.ACTIVE, receipts);

        assertEquals(ControllerFault.DEADLINE_MISSED, starved.state());
        assertEquals(ControllerFault.DEADLINE_MISSED, validator.state());
        assertEquals(0, validator.dormancyReanchors());
    }

    @Test
    void dormancyGapDuringTheGraceWindowStillReanchorsAndDiscardsTheGrace() {
        // ADR 0013 x ADR 0012 interaction, both directions, restated under ADR 0018.
        // The dormancy gap test runs before the hold evaluation and is keyed on
        // validate-call gaps, which no grace alters, so a dormancy gap opening during a
        // grace window (here at 105, inside the applied grace through 108) still
        // re-anchors, and the re-anchor still clears the applied anchor.
        //
        // What ADR 0018 changes is the DIRECTION of the consequence, and this pin now
        // records it honestly rather than restating ADR 0013's claim. The re-anchor
        // lands on the cold branch, which now carries COLD_START_GRACE_TICKS of its
        // own, so the re-armed deadline latches at 113 (105 + 2 + 5) instead of at 108.
        // The discarded applied grace would have latched at 109, so a re-anchor no
        // longer necessarily SHORTENS the tolerated gap — here it lengthens it by four
        // ticks. ADR 0012's prose on that point is superseded by ADR 0018.
        //
        // The pin stays genuinely distinguishing: had the re-anchor kept the applied
        // anchor, the latch would be at 109, not 113.
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 100L));
        validator.validate(
                List.of(frame(0, 0, 100, 101, 102, 0.25)),
                101,
                ControllerFault.ACTIVE,
                receipts);

        ValidationResult resumed =
                validator.validate(List.of(), 105, ControllerFault.ACTIVE, receipts);
        for (long tick = 106; tick <= 112; tick++) {
            assertEquals(
                    ControllerFault.ACTIVE,
                    validator.validate(List.of(), tick, ControllerFault.ACTIVE, receipts)
                            .state());
        }
        ValidationResult missed =
                validator.validate(List.of(), 113, ControllerFault.ACTIVE, receipts);

        assertEquals(1, validator.dormancyReanchors());
        assertEquals(ControllerFault.ACTIVE, resumed.state());
        assertEquals(ControllerFault.DEADLINE_MISSED, missed.state());
        assertEquals(ControllerFault.DEADLINE_MISSED, validator.state());
    }

    @Test
    void coldStartFramelessGapToTheGracedHoldStillAppliesTheFirstFrame() {
        // ADR 0018: with COLD_START_GRACE_TICKS = 5 the cold hold runs through 107, so
        // a session gets seven application attempts instead of two. Every frameless
        // tick to the inclusive boundary is fail-neutral rejected-ACTIVE with neutral
        // motors — the identical state ticks 101 and 102 already occupied — and a first
        // frame arriving at that boundary applies cleanly. This is the failure the
        // operator's cold starts hit, and the budget it is measured against has since
        // doubled: the controller targets observation.game_tick + ACTION_TARGET_OFFSET_TICKS
        // and the mod applies only on targetTick == currentGameTick, so a cold service has
        // 100 ms rather than 50 to answer -- and, disclosed, one application attempt fewer,
        // because its first frame can only target firstActive + 2 while the cold hold still
        // ends at firstActive + 7.
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(7L, 106L));

        validator.validate(List.of(), 100, ControllerFault.ACTIVE, receipts);
        for (long tick = 101; tick <= 106; tick++) {
            ValidationResult graced =
                    validator.validate(List.of(), tick, ControllerFault.ACTIVE, receipts);
            assertEquals(ControllerFault.ACTIVE, graced.state());
            assertFalse(graced.accepted());
            assertTrue(graced.appliedAction().isNeutral());
        }
        ValidationResult firstAction = validator.validate(
                List.of(frame(0, 7, 106, 107, 108, 0.5)),
                107,
                ControllerFault.ACTIVE,
                receipts);

        assertTrue(firstAction.accepted());
        assertEquals(0.5, firstAction.appliedAction().forwardThrust());
        assertEquals(ControllerFault.ACTIVE, validator.state());
        assertEquals(0, validator.lastCommandSequence());
        assertEquals(0, validator.latenessRetirements());
        assertEquals(0, validator.dormancyReanchors());
    }

    @Test
    void coldStartToleratesExactlyTheSameFramelessGapAsTheAppliedHold() {
        // ADR 0018's declared invariant, pinned as a symmetry rather than as two
        // literals: a session tolerates the same eight-tick frameless gap whether or
        // not an action has yet applied. The cold validator activates at 100 and the
        // warm one applies at 101; both must stay ACTIVE through anchor + 7 and latch
        // at anchor + 8. Deriving both sides from the same loop is what makes this a
        // pin on the invariant instead of a restatement of the constants.
        CommandValidator cold = new CommandValidator(IDENTITY);
        CommandValidator warm = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 99L));

        long coldAnchor = 100;
        cold.validate(List.of(), coldAnchor, ControllerFault.ACTIVE, receipts);
        long warmAnchor = 101;
        assertTrue(warm.validate(
                        List.of(frame(0, 0, 99, 101, 101, 0.25)),
                        warmAnchor,
                        ControllerFault.ACTIVE,
                        receipts)
                .accepted());

        for (long offset = 1; offset <= 7; offset++) {
            assertEquals(
                    ControllerFault.ACTIVE,
                    cold.validate(
                                    List.of(),
                                    coldAnchor + offset,
                                    ControllerFault.ACTIVE,
                                    receipts)
                            .state());
            assertEquals(
                    ControllerFault.ACTIVE,
                    warm.validate(
                                    List.of(),
                                    warmAnchor + offset,
                                    ControllerFault.ACTIVE,
                                    receipts)
                            .state());
        }

        assertEquals(
                ControllerFault.DEADLINE_MISSED,
                cold.validate(List.of(), coldAnchor + 8, ControllerFault.ACTIVE, receipts)
                        .state());
        assertEquals(
                ControllerFault.DEADLINE_MISSED,
                warm.validate(List.of(), warmAnchor + 8, ControllerFault.ACTIVE, receipts)
                        .state());
        assertEquals(0, cold.dormancyReanchors());
        assertEquals(0, warm.dormancyReanchors());
    }

    @Test
    void lateFirstFrameRetiresAndTheServiceCatchingUpInsideTheGraceStillFlies() {
        // The live cold-start scenario end to end. Measured first-action latency sat in
        // (100, ~150] ms against a one-tick (50 ms) budget, so the first frames arrive
        // already past their targets and retire fail-neutral (ADR 0011). Before
        // ADR 0018 the session was dead at 103 regardless — the pre-candidate
        // first-action guard latched before the retired frame could even be examined.
        // With the grace the session survives the warm-up and applies the first frame
        // the service produces in time.
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 100L, 5L, 104L));

        validator.validate(List.of(), 100, ControllerFault.ACTIVE, receipts);
        validator.validate(List.of(), 101, ControllerFault.ACTIVE, receipts);
        validator.validate(List.of(), 102, ControllerFault.ACTIVE, receipts);
        ValidationResult retired = validator.validate(
                List.of(frame(0, 0, 100, 101, 102, 0.5)),
                103,
                ControllerFault.ACTIVE,
                receipts);
        ValidationResult held =
                validator.validate(List.of(), 104, ControllerFault.ACTIVE, receipts);
        ValidationResult applied = validator.validate(
                List.of(frame(1, 5, 104, 105, 106, 0.75)),
                105,
                ControllerFault.ACTIVE,
                receipts);

        assertEquals(ControllerFault.ACTIVE, retired.state());
        assertFalse(retired.accepted());
        assertTrue(retired.appliedAction().isNeutral());
        assertEquals(1, validator.latenessRetirements());
        assertEquals(ControllerFault.ACTIVE, held.state());
        assertTrue(applied.accepted());
        assertEquals(0.75, applied.appliedAction().forwardThrust());
        assertEquals(1, validator.lastCommandSequence());
        assertEquals(0, validator.dormancyReanchors());
    }

    @Test
    void gracedColdStartLeavesEveryAdmissionRuleUntouched() {
        // The grace must move exactly one thing: the tick at which a persistent
        // ABSENCE of applicable actions latches. Each frame below is presented at 105,
        // genuinely five ticks into a graced cold-start window that would have been
        // dead at 103 before ADR 0018, and must produce the identical fault it produces
        // anywhere else. Freshness, receipt verification, receipt-tick binding, and
        // sequence continuity are all exercised, so the grace cannot have widened
        // admission.
        assertRejectedInsideTheColdGrace(
                ControllerFault.STALE_COMMAND, frame(0, 9, 104, 105, 106, 0.5), Map.of());
        assertRejectedInsideTheColdGrace(
                ControllerFault.IDENTITY_MISMATCH,
                frame(0, 9, 104, 105, 106, 0.5),
                Map.of(9L, 103L));
        assertRejectedInsideTheColdGrace(
                ControllerFault.FUTURE_COMMAND,
                frame(0, 9, 106, 107, 108, 0.5),
                Map.of(9L, 106L));
        assertRejectedInsideTheColdGrace(
                ControllerFault.SEQUENCE_ERROR,
                frame(3, 9, 104, 105, 106, 0.5),
                Map.of(9L, 104L));

        // A frame older than MAX_COMMAND_AGE_TICKS is still never applied inside the
        // grace: it retires fail-neutral and reserves nothing.
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 100L));
        for (long tick = 100; tick <= 104; tick++) {
            validator.validate(List.of(), tick, ControllerFault.ACTIVE, receipts);
        }
        ValidationResult stale = validator.validate(
                List.of(frame(0, 0, 100, 101, 102, 0.5)),
                105,
                ControllerFault.ACTIVE,
                receipts);

        assertFalse(stale.accepted());
        assertTrue(stale.appliedAction().isNeutral());
        assertEquals(ControllerFault.ACTIVE, stale.state());
        assertEquals(0, validator.pendingCommandCount());
        assertEquals(1, validator.latenessRetirements());
    }

    /** Activates at 100, ticks frameless to 104, then presents one frame at 105. */
    private static void assertRejectedInsideTheColdGrace(
            ControllerFault expected, ActionFrame candidate, Map<Long, Long> receipts) {
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> ledger = lookup(receipts);
        for (long tick = 100; tick <= 104; tick++) {
            assertEquals(
                    ControllerFault.ACTIVE,
                    validator.validate(List.of(), tick, ControllerFault.ACTIVE, ledger).state());
        }
        ValidationResult result =
                validator.validate(List.of(candidate), 105, ControllerFault.ACTIVE, ledger);

        assertFalse(result.accepted());
        assertTrue(result.appliedAction().isNeutral());
        assertEquals(expected, result.state());
        assertEquals(expected, validator.state());
    }

    private static void assertRejected(
            ControllerFault expected,
            ActionFrame frame,
            long currentTick,
            Map<Long, Long> receipts) {
        CommandValidator validator = new CommandValidator(IDENTITY);
        ValidationResult result = validator.validate(
                List.of(frame), currentTick, ControllerFault.ACTIVE, lookup(receipts));
        assertFalse(result.accepted());
        assertTrue(result.appliedAction().isNeutral());
        assertEquals(expected, result.state());
        assertEquals(expected, validator.state());
    }

    @Test
    void neuralStepGapBetweenSentActionsIsAccepted() {
        // Both validators bound neural intervals with strict less-than, not equality:
        // a forward gap is legal and must apply cleanly, never fault or retire. This
        // pins that RULE, and no longer names a producer of gaps. The producer it used
        // to name -- the service dropping a computed-but-expired action after its
        // neural steps had already run -- is removed: that drop reused the dropped
        // frame's command sequence, which the controller had already recorded, and it
        // killed two live sessions. Every computed frame is now sent, so the service
        // emits contiguous intervals; the rule stays strict less-than regardless.
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 100L, 1L, 101L));
        validator.validate(
                List.of(frame(0, 0, 100, 101, 102, 0.25)),
                101,
                ControllerFault.ACTIVE,
                receipts);

        ActionFrame gapped = new ActionFrame(
                IDENTITY,
                1,
                1,
                101,
                102,
                103,
                10,
                15,
                "6".repeat(64),
                new RawPrimitiveAction(0.5, 0, 0, 0, 0),
                false);
        ValidationResult applied =
                validator.validate(List.of(gapped), 102, ControllerFault.ACTIVE, receipts);

        assertEquals(ControllerFault.ACTIVE, applied.state());
        assertTrue(applied.accepted());
        assertEquals(0.5, applied.appliedAction().forwardThrust(), 0.0);
        assertEquals(0, validator.latenessRetirements());
    }

    @Test
    void programLaneRefusesNonNeutralRawActionAndMotorLaneRefusesProgramFrames() {
        CommandValidator programSession = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 100L, 1L, 101L));

        ValidationResult firstProgram = programSession.validate(
                List.of(programFrame(0, 0, 100, 101, 102, BodyProgramId.CRUISE)),
                101,
                ControllerFault.ACTIVE,
                receipts);
        assertTrue(firstProgram.accepted());
        assertEquals(CommandValidator.FrameLane.PROGRAM, programSession.lane());

        ValidationResult mixed = programSession.validate(
                List.of(frame(1, 1, 101, 102, 103, 0.75)),
                102,
                ControllerFault.ACTIVE,
                receipts);
        assertFalse(mixed.accepted());
        assertEquals(ControllerFault.PROTOCOL_ERROR, mixed.state());
        assertTrue(mixed.appliedAction().isNeutral());
        assertEquals(ControllerFault.PROTOCOL_ERROR, programSession.state());

        CommandValidator motorSession = new CommandValidator(IDENTITY);
        ValidationResult firstMotor = motorSession.validate(
                List.of(frame(0, 0, 100, 101, 102, 0.25)),
                101,
                ControllerFault.ACTIVE,
                receipts);
        assertTrue(firstMotor.accepted());
        assertEquals(CommandValidator.FrameLane.MOTOR, motorSession.lane());

        ValidationResult intruder = motorSession.validate(
                List.of(programFrame(1, 1, 101, 102, 103, BodyProgramId.CRUISE)),
                102,
                ControllerFault.ACTIVE,
                receipts);
        assertFalse(intruder.accepted());
        assertEquals(ControllerFault.PROTOCOL_ERROR, intruder.state());
        assertTrue(intruder.appliedAction().isNeutral());
    }

    /**
     * A paused, replaying, or faulted service emits neutral actions on both lanes, so a neutral
     * action must neither decide the lane nor violate it. Without this the program lane would fail
     * closed the first time the operator paused the service.
     */
    @Test
    void neutralActionsNeverDecideTheLane() {
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 100L, 1L, 101L, 2L, 102L));

        ValidationResult neutral = validator.validate(
                List.of(frame(0, 0, 100, 101, 102, 0.0)),
                101,
                ControllerFault.ACTIVE,
                receipts);
        assertTrue(neutral.accepted());
        assertEquals(CommandValidator.FrameLane.UNDECIDED, validator.lane());

        ValidationResult program = validator.validate(
                List.of(programFrame(1, 1, 101, 102, 103, BodyProgramId.CRUISE)),
                102,
                ControllerFault.ACTIVE,
                receipts);
        assertTrue(program.accepted());
        assertEquals(CommandValidator.FrameLane.PROGRAM, validator.lane());

        ValidationResult pausedAgain = validator.validate(
                List.of(frame(2, 2, 102, 103, 104, 0.0)),
                103,
                ControllerFault.ACTIVE,
                receipts);
        assertTrue(pausedAgain.accepted());
        assertEquals(CommandValidator.FrameLane.PROGRAM, validator.lane());
        assertEquals(ControllerFault.ACTIVE, validator.state());
    }

    /** A RESET replaces the world, not the controller on the other end of the socket. */
    @Test
    void resetKeepsTheLatchedLane() {
        CommandValidator validator = new CommandValidator(IDENTITY);
        validator.validate(
                List.of(programFrame(0, 0, 100, 101, 102, BodyProgramId.CRUISE)),
                101,
                ControllerFault.ACTIVE,
                lookup(Map.of(0L, 100L)));
        assertEquals(CommandValidator.FrameLane.PROGRAM, validator.lane());

        validator.resetActuationWindow(105);

        assertEquals(CommandValidator.FrameLane.PROGRAM, validator.lane());
    }

    /**
     * Every admission rule is envelope-only, so a program frame meets exactly the rules an action
     * frame meets: neural-step continuity, the two-tick freshness bound, and the two-slot window.
     */
    @Test
    void programFrameParticipatesInNeuralStepContinuityFreshnessAndWindow() {
        CommandValidator overlapping = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 100L, 1L, 101L, 2L, 102L));
        overlapping.validate(
                List.of(programFrame(0, 0, 100, 101, 102, BodyProgramId.CRUISE)),
                101,
                ControllerFault.ACTIVE,
                receipts);
        ProgramActionFrame rewound = new ProgramActionFrame(
                IDENTITY, 1, 1, 101, 102, 103, 0, 5, "6".repeat(64),
                BodyProgramId.CRUISE,
                Modulation.NEUTRAL,
                ProgramParameters.NEUTRAL,
                ReadoutValues.ZERO,
                "7".repeat(64));
        ValidationResult overlap =
                overlapping.validate(List.of(rewound), 102, ControllerFault.ACTIVE, receipts);
        assertEquals(ControllerFault.SEQUENCE_ERROR, overlap.state());

        CommandValidator stale = new CommandValidator(IDENTITY);
        ValidationResult retired = stale.validate(
                List.of(programFrame(0, 0, 100, 101, 102, BodyProgramId.CRUISE)),
                103,
                ControllerFault.ACTIVE,
                receipts);
        assertFalse(retired.accepted());
        assertEquals(ControllerFault.ACTIVE, stale.state());
        assertEquals(1, stale.latenessRetirements());

        CommandValidator twoSlot = new CommandValidator(IDENTITY);
        ValidationResult windowed = twoSlot.validate(
                List.of(
                        programFrame(0, 0, 100, 101, 102, BodyProgramId.CRUISE),
                        programFrame(1, 1, 101, 102, 103, BodyProgramId.ESCAPE_JUMP)),
                101,
                ControllerFault.ACTIVE,
                receipts);
        assertTrue(windowed.accepted());
        assertEquals(1, twoSlot.pendingCommandCount());
    }

    /** A program frame requests no motors, so an admitted one applies the neutral vector. */
    @Test
    void anAdmittedProgramFrameCarriesNoMotorRequest() {
        CommandValidator validator = new CommandValidator(IDENTITY);
        ValidationResult result = validator.validate(
                List.of(programFrame(0, 0, 100, 101, 102, BodyProgramId.ESCAPE_JUMP)),
                101,
                ControllerFault.ACTIVE,
                lookup(Map.of(0L, 100L)));

        assertTrue(result.accepted());
        assertTrue(result.appliedAction().isNeutral());
        assertEquals(RawPrimitiveAction.ZERO, result.rawAction());
        assertEquals(
                BodyProgramId.ESCAPE_JUMP,
                ((ProgramActionFrame) result.command().orElseThrow()).program());
    }

    /**
     * The behaviour the two-tick budget exists for: a frame may arrive before its tick.
     *
     * <p>The service answers observation 100 with a frame targeting 102. If it gets back at
     * 101 -- the fast case the extra tick buys margin for -- the frame is admitted and then
     * <strong>waits</strong>: tick 101 is rejected-ACTIVE with neutral motors, nothing is
     * retired, and the frame applies on exactly its target tick. The applicability rule is
     * ADR 0018's, unchanged: a frame applies only on {@code targetTick == currentGameTick}.
     */
    @Test
    void aFrameArrivingATickEarlyWaitsAndAppliesOnItsTargetTick() {
        CommandValidator validator = new CommandValidator(IDENTITY);
        LongFunction<OptionalLong> receipts = lookup(Map.of(0L, 100L));

        ValidationResult early = validator.validate(
                List.of(budgetFrame(0, 0, 100, 0.5)), 101, ControllerFault.ACTIVE, receipts);

        assertFalse(early.accepted());
        assertEquals(ControllerFault.ACTIVE, early.state());
        assertTrue(early.appliedAction().isNeutral(), "a waiting frame actuates nothing");
        assertEquals(1, validator.pendingCommandCount());
        assertEquals(0, validator.latenessRetirements());

        ValidationResult due =
                validator.validate(List.of(), 102, ControllerFault.ACTIVE, receipts);

        assertTrue(due.accepted());
        assertEquals(0.5, due.appliedAction().forwardThrust());
        assertEquals(0, validator.pendingCommandCount());
        assertEquals(0, validator.latenessRetirements());
    }

    /** The on-time case: the frame arrives on its own target tick and applies in that pass. */
    @Test
    void aFrameArrivingOnItsTargetTickAppliesImmediately() {
        CommandValidator validator = new CommandValidator(IDENTITY);

        ValidationResult result = validator.validate(
                List.of(budgetFrame(0, 0, 100, 0.25)),
                102,
                ControllerFault.ACTIVE,
                lookup(Map.of(0L, 100L)));

        assertTrue(result.accepted());
        assertEquals(0.25, result.appliedAction().forwardThrust());
        assertEquals(0, validator.latenessRetirements());
    }

    /**
     * A tick later still, and the frame retires fail-neutral on the <em>unchanged</em> age
     * rule.
     *
     * <p>{@code MAX_COMMAND_AGE_TICKS} is deliberately not widened by this lane: under the
     * two-tick budget a frame's age at application is exactly 2, which the {@code age >}
     * comparison admits, and one tick later the same rule retires it. So the budget bought a
     * tick of compute margin without buying a tick of staleness.
     */
    @Test
    void aFrameArrivingATickLateRetiresFailNeutralOnTheUnchangedAgeRule() {
        CommandValidator validator = new CommandValidator(IDENTITY);

        ValidationResult late = validator.validate(
                List.of(budgetFrame(0, 0, 100, 0.75)),
                103,
                ControllerFault.ACTIVE,
                lookup(Map.of(0L, 100L)));

        assertFalse(late.accepted());
        assertEquals(ControllerFault.ACTIVE, late.state());
        assertTrue(late.appliedAction().isNeutral());
        assertEquals(1, validator.latenessRetirements());
        assertEquals(0, validator.pendingCommandCount());
        // The frame was consumed, so the next honest frame is not a SEQUENCE_ERROR.
        assertEquals(0, validator.lastCommandSequence());
    }

    /** A frame exactly as the service emits it: target = observation + the declared budget. */
    private static ActionFrame budgetFrame(
            long commandSequence, long observationSequence, long observationTick, double forward) {
        long target = observationTick + CommandValidator.ACTION_TARGET_OFFSET_TICKS;
        return frame(
                commandSequence, observationSequence, observationTick, target, target, forward);
    }

    private static ProgramActionFrame programFrame(
            long commandSequence,
            long observationSequence,
            long observationTick,
            long targetTick,
            long validThroughTick,
            BodyProgramId program) {
        return new ProgramActionFrame(
                IDENTITY,
                commandSequence,
                observationSequence,
                observationTick,
                targetTick,
                validThroughTick,
                commandSequence * 5,
                commandSequence * 5 + 5,
                "6".repeat(64),
                program,
                new Modulation(0.7, 1.0, 1.0, 0.0),
                new ProgramParameters(-0.25),
                new ReadoutValues(0.0, -0.25, 0.61),
                "7".repeat(64));
    }

    private static ActionFrame frame(
            long commandSequence,
            long observationSequence,
            long observationTick,
            long targetTick,
            long validThroughTick,
            double forward) {
        return new ActionFrame(
                IDENTITY,
                commandSequence,
                observationSequence,
                observationTick,
                targetTick,
                validThroughTick,
                commandSequence * 5,
                commandSequence * 5 + 5,
                "6".repeat(64),
                new RawPrimitiveAction(forward, 0, 0, 0, 0),
                forward == 0.0);
    }

    private static LongFunction<OptionalLong> lookup(Map<Long, Long> receipts) {
        return sequence -> {
            Long tick = receipts.get(sequence);
            return tick == null ? OptionalLong.empty() : OptionalLong.of(tick);
        };
    }
}
