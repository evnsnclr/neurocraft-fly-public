package dev.neurocraft.fly.control;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ProgramParameters;
import dev.neurocraft.fly.body.ReadoutValues;
import java.util.List;
import java.util.OptionalLong;
import org.junit.jupiter.api.Test;

/** Companion pins for the sent-observation ledger the command validator verifies against. */
final class SentObservationWindowTest {
    private static final ControllerIdentity IDENTITY = new ControllerIdentity(
            "retention-test", 7, "fly-01", "1".repeat(64), "2".repeat(64),
            "3".repeat(64), "4".repeat(64), "5".repeat(64));
    /** Every tick the ADR 0018 graced cold-start hold can keep a session alive for. */
    private static final long GRACED_SPAN =
            CommandValidator.FIRST_ACTION_DEADLINE_TICKS + CommandValidator.COLD_START_GRACE_TICKS;

    @Test
    void ledgerCoversEveryTickTheGracedColdStartHoldTolerates() {
        // ADR 0018 companion hardening. The grace lets a session survive GRACED_SPAN
        // frameless ticks, which makes frames that old reachable at all for the first
        // time. A frame whose observation receipt has aged out of this ledger is
        // unverifiable and latches STALE_COMMAND, so a ledger shorter than the graced
        // span would have converted the grace into a newly reachable hard fault. The
        // capacity is derived from the constants here, not written as a literal, so the
        // two cannot drift apart.
        SentObservationWindow window = new SentObservationWindow();
        for (long offset = 0; offset <= GRACED_SPAN; offset++) {
            window.record(offset, 100 + offset);
        }

        assertEquals(OptionalLong.of(100), window.tickFor(0));
        assertEquals(OptionalLong.of(100 + GRACED_SPAN), window.tickFor(GRACED_SPAN));
    }

    @Test
    void ledgerStillEvictsOnceTheGracedSpanIsExceeded() {
        // The ledger is still bounded: it is one entry longer than the graced span (the
        // current tick plus every graced tick behind it), and the entry that falls out
        // is one no live session can still be waiting on.
        SentObservationWindow window = new SentObservationWindow();
        for (long offset = 0; offset <= GRACED_SPAN + 1; offset++) {
            window.record(offset, 100 + offset);
        }

        assertEquals(OptionalLong.empty(), window.tickFor(0));
        assertEquals(OptionalLong.of(101), window.tickFor(1));
        assertEquals(OptionalLong.of(101 + GRACED_SPAN), window.tickFor(GRACED_SPAN + 1));
    }

    @Test
    void ledgerStillRefusesNonIncreasingAndNegativeReceipts() {
        SentObservationWindow window = new SentObservationWindow();
        window.record(4, 104);

        assertThrows(IllegalStateException.class, () -> window.record(4, 105));
        assertThrows(IllegalStateException.class, () -> window.record(3, 105));
        assertThrows(IllegalArgumentException.class, () -> window.record(-1, 105));
        assertThrows(IllegalArgumentException.class, () -> window.record(5, -1));

        window.clear();
        assertEquals(OptionalLong.empty(), window.tickFor(4));
    }

    @Test
    void fourThroughSevenNewerObservationsRetireLateProgramAndAdmitFreshSuccessor() {
        for (long newer = 4; newer <= 7; newer++) {
            long arrival = 100 + newer;
            SentObservationWindow window = new SentObservationWindow();
            CommandValidator validator = activeUntil(window, arrival);
            window.record(arrival - 99, arrival);
            ValidationResult late = validator.validate(List.of(program(1, 1, 100)), arrival,
                    ControllerFault.ACTIVE, window::tickFor);
            assertFalse(late.accepted());
            assertEquals(ControllerFault.ACTIVE, late.state());
            assertTrue(late.appliedAction().isNeutral());
            assertEquals(1, validator.lastCommandSequence());
            assertEquals(1, validator.latenessRetirements());
            assertEquals(0, validator.pendingCommandCount());

            window.record(arrival - 98, arrival + 1);
            ValidationResult waiting = validator.validate(
                    List.of(program(2, arrival - 99, arrival)), arrival + 1,
                    ControllerFault.ACTIVE, window::tickFor);
            assertFalse(waiting.accepted());
            assertTrue(waiting.appliedAction().isNeutral());
            window.record(arrival - 97, arrival + 2);
            ValidationResult fresh = validator.validate(List.of(), arrival + 2,
                    ControllerFault.ACTIVE, window::tickFor);
            assertTrue(fresh.accepted());
            assertEquals(2, validator.lastCommandSequence());
            assertEquals(1, validator.latenessRetirements());
            assertEquals(0, validator.dormancyReanchors());
        }
    }

    @Test
    void lateRetirementDoesNotExtendTheExistingAppliedStarvationHold() {
        SentObservationWindow window = new SentObservationWindow();
        CommandValidator validator = activeUntil(window, 107);
        window.record(8, 107);
        ValidationResult late = validator.validate(List.of(program(1, 1, 100)), 107,
                ControllerFault.ACTIVE, window::tickFor);
        assertEquals(ControllerFault.ACTIVE, late.state());
        assertFalse(late.accepted());
        assertTrue(late.appliedAction().isNeutral());
        assertEquals(1, validator.latenessRetirements());
        // Last application was 101, so grace still ends at 108 and failure is still at 109.
        window.record(9, 108);
        assertEquals(ControllerFault.ACTIVE,
                validator.validate(List.of(), 108, ControllerFault.ACTIVE, window::tickFor).state());
        window.record(10, 109);
        ValidationResult expired = validator.validate(List.of(), 109,
                ControllerFault.ACTIVE, window::tickFor);
        assertEquals(ControllerFault.DEADLINE_MISSED, expired.state());
        assertTrue(expired.appliedAction().isNeutral());
        assertEquals(0, validator.dormancyReanchors());
    }

    @Test
    void eighthNewerObservationStillMakesTheLateProgramUnverifiable() {
        SentObservationWindow window = new SentObservationWindow();
        CommandValidator validator = activeUntil(window, 108);
        window.record(9, 108);
        assertEquals(OptionalLong.empty(), window.tickFor(1));
        ValidationResult expired = validator.validate(List.of(program(1, 1, 100)), 108,
                ControllerFault.ACTIVE, window::tickFor);
        assertEquals(ControllerFault.STALE_COMMAND, expired.state());
        assertFalse(expired.accepted());
        assertTrue(expired.appliedAction().isNeutral());
        assertEquals(0, validator.latenessRetirements());
    }

    /** Tick continuously, with one real applied anchor at 101; no dormancy re-anchor shortcut. */
    private static CommandValidator activeUntil(SentObservationWindow window, long arrival) {
        CommandValidator validator = new CommandValidator(IDENTITY);
        window.record(0, 99);
        window.record(1, 100);
        validator.validate(List.of(), 100, ControllerFault.ACTIVE, window::tickFor);
        for (long tick = 101; tick < arrival; tick++) {
            window.record(tick - 99, tick);
            ValidationResult result = validator.validate(tick == 101 ? List.of(program(0, 0, 99))
                    : List.of(), tick, ControllerFault.ACTIVE, window::tickFor);
            assertEquals(tick == 101, result.accepted());
            assertEquals(ControllerFault.ACTIVE, result.state());
        }
        return validator;
    }

    private static ProgramActionFrame program(long sequence, long observation, long tick) {
        return new ProgramActionFrame(IDENTITY, sequence, observation, tick, tick + 2, tick + 2,
                sequence * 5, sequence * 5 + 5, "6".repeat(64), BodyProgramId.CRUISE,
                new Modulation(0.5, 1, 1, 0), new ProgramParameters(0),
                new ReadoutValues(0, 0, 0), "7".repeat(64));
    }
}
