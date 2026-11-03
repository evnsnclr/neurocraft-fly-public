package dev.neurocraft.fly.body;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.neurocraft.fly.body.programs.CruiseProgram;
import dev.neurocraft.fly.body.programs.EscapeJumpProgram;
import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import java.util.OptionalLong;
import org.junit.jupiter.api.Test;

/** Trial determinism is about body inputs and owned ticks, not changing the world clock. */
final class BodyTrialSeedTest {
    private static final Modulation MODULATION = new Modulation(0.7, 0.8, 0.6, 0.0);
    private static final ProgramParameters PARAMETERS = new ProgramParameters(-0.25);
    private static final ReadoutValues READOUT = new ReadoutValues(0.2, -0.25, 0.61);
    private static final String POLICY = "7".repeat(64);

    @Test
    void oldConstructorsRetainEntryTickSeedsAndExplicitSeedAdvancesIndependently() {
        for (long tick : new long[] {0, 1, 72_001, Long.MAX_VALUE}) {
            var old = new BodyProgramState(BodyProgramId.CRUISE, tick, 17, 3);
            assertEquals(tick, old.wanderSeed());
            assertEquals(tick, BodyProgramState.entering(BodyProgramId.CRUISE, tick).wanderSeed());
            var seeded = BodyProgramState.entering(BodyProgramId.CRUISE, tick, 42).advanced(7);
            assertEquals(tick, seeded.enteredGameTick());
            assertEquals(42, seeded.wanderSeed());
            assertEquals(1, seeded.ticksInProgram());
            assertEquals(7, seeded.phaseIndex());
        }
        assertThrows(IllegalArgumentException.class,
                () -> BodyProgramState.entering(BodyProgramId.CRUISE, 10, -1));
        assertThrows(IllegalArgumentException.class,
                () -> BodyProgramState.entering(BodyProgramId.CRUISE, -1, 10));
    }

    @Test
    void sameSeedRepeatsCompleteExecutionsAcrossDifferentRealWorldTicksAndStalls() {
        for (long seed : new long[] {0, 1, 42, 72_001, Long.MAX_VALUE}) {
            var early = new BodyProgramRuntime();
            var late = new BodyProgramRuntime();
            early.resetForTrial(seed);
            late.resetForTrial(seed);
            for (int tick = 0; tick < 350; tick++) {
                int cycle = tick % 100;
                var program = cycle >= 50 && cycle < 74 ? BodyProgramId.ESCAPE_JUMP : BodyProgramId.CRUISE;
                if (tick % 17 == 16) {
                    var before = early.state();
                    assertTrue(early.advance(sample(100 + tick, tick), false, false).isEmpty());
                    assertTrue(late.advance(sample(90_000 + tick, tick), false, false).isEmpty());
                    assertEquals(before, early.state(), "a missed command must not advance the program");
                } else {
                    assertEquals(step(early, 100 + tick, tick, program),
                            step(late, 90_000 + tick, tick, program));
                }
                assertEquals(89_900, late.state().enteredGameTick() - early.state().enteredGameTick());
                assertEquals(early.state().ticksInProgram(), late.state().ticksInProgram());
                assertEquals(early.state().phaseIndex(), late.state().phaseIndex());
                assertEquals(seed, early.effectiveWanderSeed());
                assertEquals(seed, late.effectiveWanderSeed());
            }
            assertTrue(early.readoutStalledTicks() > 0);
            assertTrue(early.transitions() > 3);
        }
    }

    @Test
    void explicitSeedControlsAvoidanceTieEvenWhenWanderModulationIsZero() {
        var cruise = new CruiseProgram();
        var context = BodyProgramContext.sensorsOnly(sample(1_000, 1));
        var modulation = new Modulation(1, 1, 0, 0);
        var right = cruise.step(BodyProgramState.entering(BodyProgramId.CRUISE, 1_000, 0),
                context, modulation, ProgramParameters.NEUTRAL);
        var left = cruise.step(BodyProgramState.entering(BodyProgramId.CRUISE, 1_000, 1),
                context, modulation, ProgramParameters.NEUTRAL);
        assertTrue(right.motors().yawTorque() > 0);
        assertTrue(left.motors().yawTorque() < 0);
        assertNotEquals(right.motors(), left.motors(), "the seed must affect the actual motors");
        var moduloEquivalent = cruise.step(
                BodyProgramState.entering(BodyProgramId.CRUISE, 9_999, 72_001),
                context, modulation, ProgramParameters.NEUTRAL);
        assertEquals(left.motors(), moduloEquivalent.motors());
        assertEquals(72_001, moduloEquivalent.next().wanderSeed(), "ledger retains the raw seed");
    }

    @Test
    void successorAndAuthorityChangesKeepSeedButRetainActualEntryTimestamp() {
        var runtime = new BodyProgramRuntime();
        runtime.resetForTrial(42);
        for (int tick = 0; tick < EscapeJumpProgram.ESCAPE_JUMP_TICKS; tick++) {
            step(runtime, 500 + tick, 0, BodyProgramId.ESCAPE_JUMP);
        }
        long exitTick = 500 + EscapeJumpProgram.ESCAPE_JUMP_TICKS - 1;
        assertEquals(BodyProgramId.CRUISE, runtime.state().id());
        assertEquals(exitTick, runtime.state().enteredGameTick());
        assertEquals(42, runtime.effectiveWanderSeed());
        step(runtime, exitTick + 1, 0, BodyProgramId.CRUISE);
        assertEquals(exitTick, runtime.state().enteredGameTick(), "successor was already entered");
        runtime.releaseScripted(700, "trial-pause");
        assertEquals(700, runtime.state().enteredGameTick());
        runtime.engageScripted(BodyProgramId.CRUISE, MODULATION, SelectionSource.OPERATOR, 800);
        assertEquals(800, runtime.state().enteredGameTick());
        assertEquals(OptionalLong.of(42), runtime.trialWanderSeed());
        assertEquals(42, runtime.effectiveWanderSeed());
        assertTrue(runtime.advance(sample(800, 0), true, true).isPresent());
    }

    @Test
    void trialResetClearsAuthorityClocksAndBookkeepingWithoutGrantingAuthority() {
        var runtime = new BodyProgramRuntime();
        step(runtime, 100, 0, BodyProgramId.CRUISE);
        runtime.advance(sample(101, 0), false, false);
        step(runtime, 102, 0, BodyProgramId.FEED);
        runtime.resetForTrial(Long.MAX_VALUE);
        assertEquals(MotorAuthority.NEURAL, runtime.authority());
        assertEquals(SelectionSource.DEFAULT, runtime.selectionSource());
        assertEquals(Modulation.NEUTRAL, runtime.modulation());
        assertEquals(ProgramParameters.NEUTRAL, runtime.parameters());
        assertEquals(ReadoutValues.ZERO, runtime.lastReadout());
        assertEquals("", runtime.policyHash());
        assertEquals(BodyProgramId.HOVER, runtime.state().id());
        assertEquals(0, runtime.state().ticksInProgram());
        assertEquals(0, runtime.state().phaseIndex());
        assertEquals(0, runtime.presentationPhase());
        assertEquals(0, runtime.programOwnedTicks());
        assertEquals(0, runtime.readoutOwnedTicks());
        assertEquals(0, runtime.readoutStalledTicks());
        assertEquals(0, runtime.readoutRefusals());
        assertEquals(0, runtime.transitions());
        assertTrue(runtime.transitionLedger().isEmpty());
        assertEquals(OptionalLong.of(Long.MAX_VALUE), runtime.trialWanderSeed());
        assertEquals(Long.MAX_VALUE, runtime.effectiveWanderSeed());
        assertTrue(runtime.advance(sample(103, 0), true, true).isEmpty());
        step(runtime, 104, 0, BodyProgramId.CRUISE);
        assertEquals(1, runtime.transitions(), "old requested program must also be forgotten");
        assertEquals(104, runtime.state().enteredGameTick());
        runtime.resetForTrial(9);
        assertEquals(OptionalLong.of(9), runtime.trialWanderSeed());
    }

    @Test
    void ordinaryResetClearsOverrideAndRestoresFreshDefaultStreamExactly() {
        var runtime = new BodyProgramRuntime();
        runtime.resetForTrial(42);
        step(runtime, 100, 0, BodyProgramId.CRUISE);
        runtime.reset();
        var fresh = new BodyProgramRuntime();
        assertFalse(runtime.trialWanderSeed().isPresent());
        assertEquals(fresh.state(), runtime.state());
        for (int tick = 0; tick < 200; tick++) {
            var program = tick >= 50 && tick < 74 ? BodyProgramId.ESCAPE_JUMP : BodyProgramId.CRUISE;
            assertEquals(step(fresh, 90_000 + tick, tick, program),
                    step(runtime, 90_000 + tick, tick, program));
            assertEquals(fresh.state(), runtime.state());
            assertEquals(runtime.state().enteredGameTick(), runtime.effectiveWanderSeed());
            assertEquals(fresh.transitionLedger(), runtime.transitionLedger());
        }
    }

    @Test
    void invalidSeedFailsBeforeMutatingTheRunningProgram() {
        var runtime = new BodyProgramRuntime();
        runtime.resetForTrial(42);
        step(runtime, 500, 0, BodyProgramId.CRUISE);
        var state = runtime.state();
        var ledger = runtime.transitionLedger();
        assertThrows(IllegalArgumentException.class, () -> runtime.resetForTrial(-1));
        assertEquals(state, runtime.state());
        assertEquals(ledger, runtime.transitionLedger());
        assertEquals(MotorAuthority.READOUT, runtime.authority());
        assertEquals(1, runtime.readoutOwnedTicks());
        assertEquals(OptionalLong.of(42), runtime.trialWanderSeed());
    }

    private static ProgramExecution step(BodyProgramRuntime runtime, long tick, int shape, BodyProgramId id) {
        return runtime.advanceReadout(sample(tick, shape), id, MODULATION, PARAMETERS, READOUT, POLICY)
                .orElseThrow();
    }

    private static PhysicalSensorSnapshot sample(long tick, int shape) {
        int phase = shape % 5;
        return new PhysicalSensorSnapshot(tick, "trial-seed-fixture", 1, 64, -2,
                0, 0, 0, 0, 0, phase == 4, false, false, false, false, 12, 10,
                6, phase == 1 ? 0 : 6, 6, phase == 2 ? 0.2 : 6,
                phase == 3 ? 0.2 : 6, 6, 3);
    }
}
