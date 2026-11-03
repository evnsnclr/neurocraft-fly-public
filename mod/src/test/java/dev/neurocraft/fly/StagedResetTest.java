package dev.neurocraft.fly;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.neurocraft.fly.control.ControlAckFrame;
import dev.neurocraft.fly.control.ControlOperation;
import dev.neurocraft.fly.control.ControlState;
import dev.neurocraft.fly.control.ControllerIdentity;
import java.util.Optional;
import java.util.UUID;
import org.junit.jupiter.api.Test;

final class StagedResetTest {
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
     * The accepted window is {@code {budget, budget + 1}} and moved with the action budget:
     * the service stamps an accepted RESET with the governed action's {@code target_tick},
     * which is one budget after the observation the effect materialized against, and that
     * observation is the issuing tick or the one after it. It was {@code {1, 2}} under the
     * one-tick budget. The frame model still decodes a one-tick delay so acknowledgements
     * recorded then keep parsing; this transaction is where that permissiveness stops.
     */
    @Test
    void delayedAcceptedAckArmsOnlyItsExactEffectiveTick() {
        StagedReset reset = reset(100);

        assertEquals(StagedReset.Outcome.WAITING, reset.advance(100, Optional.empty()));
        assertEquals(
                StagedReset.Outcome.WAITING,
                reset.advance(100, Optional.of(ack(100, 103, true))));
        assertEquals(StagedReset.Outcome.WAITING, reset.advance(101, Optional.empty()));
        assertEquals(StagedReset.Outcome.WAITING, reset.advance(102, Optional.empty()));
        assertEquals(StagedReset.Outcome.READY, reset.advance(103, Optional.empty()));
        assertEquals(103, reset.effectiveTick());

        StagedReset earliest = reset(200);
        assertEquals(
                StagedReset.Outcome.READY,
                earliest.advance(202, Optional.of(ack(200, 202, true))));

        // The other end of the accepted window: an acknowledgement stamped one action budget
        // plus one after issue, first polled on that very tick, is READY rather than LATE_ACK.
        StagedReset latest = reset(250);
        assertEquals(
                StagedReset.Outcome.READY,
                latest.advance(253, Optional.of(ack(250, 253, true))));
        assertEquals(253, latest.effectiveTick());

        StagedReset shortOfTheBudget = reset(300);
        assertEquals(
                StagedReset.Outcome.INVALID,
                shortOfTheBudget.advance(301, Optional.of(ack(300, 301, true))));
        assertEquals("EFFECTIVE_TICK", shortOfTheBudget.failureCode());

        assertThrows(IllegalArgumentException.class, () -> ack(400, 400, true));
        assertThrows(IllegalArgumentException.class, () -> ack(400, 404, true));
    }

    @Test
    void rejectedAndMissingAcknowledgementsNeverBecomeReady() {
        StagedReset rejected = reset(100);
        assertEquals(
                StagedReset.Outcome.REJECTED,
                rejected.advance(100, Optional.of(ack(100, 100, false))));
        assertEquals("INVALID_STATE", rejected.failureCode());

        StagedReset missing = reset(100);
        assertEquals(StagedReset.Outcome.WAITING, missing.advance(102, Optional.empty()));
        assertEquals(StagedReset.Outcome.MISSED, missing.advance(103, Optional.empty()));
        assertEquals("ACK_DEADLINE", missing.failureCode());
    }

    @Test
    void slotRejectsOverlappingResetTransaction() {
        StagedReset first = reset(100);
        StagedReset second = reset(101);
        StagedResetSlot slot = new StagedResetSlot();

        assertTrue(slot.stage(first));
        assertFalse(slot.stage(second));
        assertEquals(first, slot.pending().orElseThrow());
        slot.clear(first);
        assertTrue(slot.stage(second));
    }

    @Test
    void missedTickStartAckKeepsInterleavedActionBlockedUntilFailureBoundary() {
        StagedReset reset = reset(100);
        StagedResetSlot slot = new StagedResetSlot();
        assertTrue(slot.stage(reset));

        // Tick 102 checks the reset before sensing. The ACK and its governed Action may arrive
        // immediately after this empty poll, but the pending slot must still gate actuation.
        assertEquals(StagedReset.Outcome.WAITING, reset.advance(102, Optional.empty()));
        assertTrue(slot.blocksActuation(ENTITY_UUID));
        assertFalse(slot.blocksActuation(
                UUID.fromString("00000000-0000-0000-0000-000000000002")));

        // At the next authoritative boundary the late ACK is terminal. No action was consumed
        // while waiting, and the runtime clears the slot only as it fails the bridge closed.
        assertEquals(
                StagedReset.Outcome.MISSED,
                reset.advance(103, Optional.of(ack(100, 102, true))));
        assertEquals("LATE_ACK", reset.failureCode());
        assertTrue(slot.blocksActuation(ENTITY_UUID));
        slot.clear(reset);
        assertFalse(slot.blocksActuation(ENTITY_UUID));
    }

    @Test
    void dormancyPastTheAcceptedEffectiveTickResolvesMissedInsteadOfWaitingForever() {
        // The ACK lands and sets the effective boundary, then the entity goes
        // chunk-unload dormant and never ticks on it. Left WAITING the slot
        // would block actuation - and the ADR 0012 re-anchor - forever.
        StagedReset reset = reset(100);
        StagedResetSlot slot = new StagedResetSlot();
        assertTrue(slot.stage(reset));

        assertEquals(
                StagedReset.Outcome.WAITING,
                reset.advance(101, Optional.of(ack(100, 102, true))));

        // Resume long after the boundary: the transaction resolves terminally.
        assertEquals(StagedReset.Outcome.MISSED, reset.advance(400, Optional.empty()));
        assertEquals("EFFECTIVE_TICK_PASSED", reset.failureCode());
        slot.clear(reset);
        assertFalse(slot.blocksActuation(ENTITY_UUID));
    }

    private static StagedReset reset(long issuedTick) {
        StagedReset reset = new StagedReset(ENTITY_UUID, issuedTick, 1, 2, 3, 90);
        reset.bindControlSequence(0);
        return reset;
    }

    private static ControlAckFrame ack(long issuedTick, long effectiveTick, boolean accepted) {
        return new ControlAckFrame(
                IDENTITY,
                0,
                issuedTick,
                ControlOperation.RESET,
                null,
                null,
                accepted,
                ControlState.RECORDING,
                effectiveTick,
                accepted ? null : "INVALID_STATE");
    }
}
