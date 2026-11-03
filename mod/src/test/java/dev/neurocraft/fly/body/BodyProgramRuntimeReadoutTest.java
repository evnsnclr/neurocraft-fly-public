package dev.neurocraft.fly.body;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.neurocraft.fly.body.programs.EscapeJumpProgram;
import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/**
 * The readout lane's selection bookkeeping, driven as the pure state machine it is.
 *
 * <p>Every value here is a literal. What these tests pin is that a selection the authority cannot
 * honour is refused <em>once</em> and then held, rather than re-selected and re-refused every tick:
 * the ledger has to stay readable, and the fallback's own clock has to keep running, or HOVER stops
 * bobbing and starts climbing.
 */
final class BodyProgramRuntimeReadoutTest {
    private static final Modulation CRUISING = new Modulation(0.7, 1.0, 1.0, 0.0);
    private static final ProgramParameters BIASED = new ProgramParameters(-0.25);
    private static final ReadoutValues READOUT = new ReadoutValues(0.0, -0.25, 0.61);
    private static final String POLICY_HASH = "7".repeat(64);

    @Test
    void missingFoodOnGroundIsRefusedOnceAndDoesNotFreezeTheFallbacksClock() {
        BodyProgramRuntime runtime = new BodyProgramRuntime();
        List<Double> verticals = new ArrayList<>();

        for (int tick = 0; tick < 6; tick++) {
            var groundedSnapshot = new PhysicalSensorSnapshot(100 + tick, "grounded-food-test", 1, 2, 1,
                    0, 0, 0, 0, 0, false, true, true, false, false, 0, 0, 6, 6, 6, 6, 6, 6, 3);
            var missingFood = new BodyProgramContext(groundedSnapshot,
                    Optional.of(new BodyProgramContext.SurfaceBelow(0, true)),
                    Optional.empty(), Optional.empty());
            ProgramExecution execution = runtime.advanceReadout(
                            missingFood,
                            BodyProgramId.FEED,
                            CRUISING,
                            BIASED,
                            READOUT,
                            POLICY_HASH)
                    .orElseThrow();
            verticals.add(execution.emittedMotors().verticalThrust());
            // Re-derived every tick: the receipt must keep naming the refusal, not report the
            // fallback as though the mapping had asked for it.
            assertEquals(
                    Optional.of(ProgramRefusalReason.PRECONDITION),
                    execution.refusalReason(),
                    "tick " + tick);
            assertEquals(BodyProgramId.HOVER, execution.executedProgram(), "tick " + tick);
            assertEquals(CRUISING, execution.executedModulation(), "tick " + tick);
        }

        assertEquals(
                2,
                runtime.transitions(),
                "an unchanging refused request must log the selection and its refusal once: "
                        + runtime.transitionLedger());
        assertEquals(2, runtime.transitionLedger().size());
        assertTrue(runtime.transitionLedger().get(0).contains("HOVER -> FEED"));
        assertTrue(runtime.transitionLedger().get(0).contains("reason=selected"));
        assertTrue(runtime.transitionLedger().get(1).contains("FEED -> HOVER"));
        assertTrue(runtime.transitionLedger().get(1).contains("reason=refused"));
        assertEquals(
                6, runtime.state().ticksInProgram(), "the fallback's own clock must run");
        assertEquals(6, runtime.readoutRefusals());
        // The bob is a sine of ticksInProgram. A frozen clock emits one constant value forever --
        // a steady climb rather than a hover -- so the discriminating assertion is that the
        // emitted vertical actually moves.
        assertNotEquals(
                verticals.get(0), verticals.get(1), "HOVER's bob did not advance between ticks");
        assertEquals(
                6,
                verticals.stream().distinct().count(),
                "HOVER emitted a repeated vertical: " + verticals);
    }

    /** A changed request is a new selection, and is logged as one. */
    @Test
    void aChangedRequestReselects() {
        BodyProgramRuntime runtime = new BodyProgramRuntime();
        runtime.advanceReadout(
                snapshot(100), BodyProgramId.CRUISE, CRUISING, BIASED, READOUT, POLICY_HASH);
        runtime.advanceReadout(
                snapshot(101), BodyProgramId.CRUISE, CRUISING, BIASED, READOUT, POLICY_HASH);
        assertEquals(1, runtime.transitions(), "an unchanged applied request re-selected");
        assertEquals(2, runtime.state().ticksInProgram(), "the running program's clock ran");

        runtime.advanceReadout(
                snapshot(102), BodyProgramId.ESCAPE_JUMP, CRUISING, BIASED, READOUT, POLICY_HASH);

        assertEquals(2, runtime.transitions());
        assertEquals(BodyProgramId.ESCAPE_JUMP, runtime.state().id());
        assertTrue(runtime.transitionLedger().getLast().contains("CRUISE -> ESCAPE_JUMP"));
    }

    /** Releasing the lane forgets the request, so re-engaging on the same id logs a selection. */
    @Test
    void releasingTheLaneForgetsTheRequest() {
        BodyProgramRuntime runtime = new BodyProgramRuntime();
        runtime.advanceReadout(
                snapshot(100), BodyProgramId.CRUISE, CRUISING, BIASED, READOUT, POLICY_HASH);
        runtime.releaseScripted(101, "faulted");
        runtime.advanceReadout(
                snapshot(102), BodyProgramId.CRUISE, CRUISING, BIASED, READOUT, POLICY_HASH);

        assertEquals(3, runtime.transitions());
        assertEquals(MotorAuthority.READOUT, runtime.authority());
        assertTrue(runtime.transitionLedger().get(1).contains("reason=faulted"));
        assertTrue(runtime.transitionLedger().getLast().contains("HOVER -> CRUISE"));
        assertTrue(runtime.transitionLedger().getLast().contains("reason=selected"));
    }

    /**
     * A program that exits into its successor and is then asked for that successor is
     * <strong>not</strong> a new selection, and must not be logged as one.
     *
     * <p>This is the only reachable self-transition. ESCAPE_JUMP exits to CRUISE on its own
     * 24-tick clock, and the stage-4b policy holds ESCAPE_JUMP for exactly those 24 ticks and then
     * asks for CRUISE -- so on the very next tick the requested id differs from the last requested
     * id (a new request) while the runtime is already sitting in CRUISE at tick zero. The ledger
     * line that produced said {@code CRUISE -> CRUISE ... reason=selected}, which is not a
     * transition, and re-entering also rewrote the successor's entry tick one tick after it was
     * entered -- and that entry tick is CRUISE's wander seed.
     */
    @Test
    void aProgramRequestedAcrossItsOwnExitIsNotLoggedAsASelfTransition() {
        BodyProgramRuntime runtime = new BodyProgramRuntime();
        ProgramExecution exitTick = null;
        for (int tick = 0; tick < EscapeJumpProgram.ESCAPE_JUMP_TICKS; tick++) {
            exitTick = runtime.advanceReadout(
                            snapshot(200 + tick),
                            BodyProgramId.ESCAPE_JUMP,
                            CRUISING,
                            BIASED,
                            READOUT,
                            POLICY_HASH)
                    .orElseThrow();
        }
        long exitGameTick = 200 + EscapeJumpProgram.ESCAPE_JUMP_TICKS - 1;
        assertEquals(BodyProgramId.CRUISE, runtime.state().id(), "the jump did not exit to CRUISE");
        assertEquals(0, runtime.state().ticksInProgram());
        assertTrue(exitTick.applied(), "the exit tick was not the requested program applied");
        assertEquals(
                BodyProgramId.ESCAPE_JUMP,
                exitTick.executedProgram(),
                "the exit tick must report the program that actually ran on it");

        ProgramExecution afterExit = runtime.advanceReadout(
                        snapshot(exitGameTick + 1),
                        BodyProgramId.CRUISE,
                        CRUISING,
                        BIASED,
                        READOUT,
                        POLICY_HASH)
                .orElseThrow();

        assertTrue(afterExit.applied(), "the successor did not run as requested");
        assertEquals(BodyProgramId.CRUISE, afterExit.executedProgram());
        assertEquals(
                2,
                runtime.transitions(),
                "requesting a successor across an exit logged an extra transition: "
                        + runtime.transitionLedger());
        List<String> ledger = runtime.transitionLedger();
        assertEquals(2, ledger.size());
        assertTrue(ledger.get(0).contains("HOVER -> ESCAPE_JUMP"));
        assertTrue(ledger.get(0).contains("reason=selected"));
        assertTrue(ledger.get(1).contains("ESCAPE_JUMP -> CRUISE"), ledger.get(1));
        assertTrue(ledger.get(1).contains("reason=exit"), ledger.get(1));
        assertTrue(
                ledger.stream().noneMatch(line -> line.contains("CRUISE -> CRUISE")),
                "a self-transition was logged: " + ledger);
        // Not re-entered: the successor's own clock ran, and its entry tick -- which is CRUISE's
        // wander seed -- is still the tick it was actually entered on.
        assertEquals(1, runtime.state().ticksInProgram(), "the successor's clock was reset");
        assertEquals(
                exitGameTick,
                runtime.state().enteredGameTick(),
                "the successor's entry tick was rewritten a tick after it was entered");
    }

    @Test
    void completeContextPreservesExistingProgramOutputsAndBookkeeping() {
        for (BodyProgramId program : List.of(BodyProgramId.HOVER, BodyProgramId.CRUISE, BodyProgramId.ESCAPE_JUMP)) {
            BodyProgramRuntime legacy = new BodyProgramRuntime();
            BodyProgramRuntime complete = new BodyProgramRuntime();
            for (int tick = 0; tick < 40; tick++) {
                var sample = snapshot(500 + tick);
                var context = new BodyProgramContext(sample,
                        Optional.of(new BodyProgramContext.SurfaceBelow(2, true)), Optional.empty(), Optional.empty());
                assertEquals(legacy.advanceReadout(sample, program, CRUISING, BIASED, READOUT, POLICY_HASH),
                        complete.advanceReadout(context, program, CRUISING, BIASED, READOUT, POLICY_HASH));
                assertEquals(legacy.state(), complete.state());
                assertEquals(legacy.transitionLedger(), complete.transitionLedger());
            }
        }
    }

    @Test
    void landRefusalAndSupportLossReceiptsNameWhatActuallyExecuted() {
        BodyProgramRuntime runtime = new BodyProgramRuntime();
        var refused = runtime.advanceReadout(snapshot(100), BodyProgramId.LAND,
                CRUISING, BIASED, READOUT, POLICY_HASH).orElseThrow();
        assertEquals(BodyProgramId.HOVER, refused.executedProgram());
        assertEquals(Optional.of(ProgramRefusalReason.PRECONDITION), refused.refusalReason());
        var supported = new BodyProgramContext(snapshot(101),
                Optional.of(new BodyProgramContext.SurfaceBelow(2, true)), Optional.empty(), Optional.empty());
        var entered = runtime.advanceReadout(supported, BodyProgramId.LAND,
                CRUISING, BIASED, READOUT, POLICY_HASH).orElseThrow();
        assertEquals(BodyProgramId.LAND, entered.executedProgram());
        assertTrue(entered.applied());
        // The first lost-support tick still executes LAND's bounded brake, then exits HOVER.
        var exiting = runtime.advanceReadout(snapshot(102), BodyProgramId.LAND,
                CRUISING, BIASED, READOUT, POLICY_HASH).orElseThrow();
        assertEquals(BodyProgramId.LAND, exiting.executedProgram());
        assertEquals(BodyProgramId.HOVER, runtime.state().id());
        var stillRefused = runtime.advanceReadout(snapshot(103), BodyProgramId.LAND,
                CRUISING, BIASED, READOUT, POLICY_HASH).orElseThrow();
        assertEquals(BodyProgramId.HOVER, stillRefused.executedProgram());
        assertEquals(Optional.of(ProgramRefusalReason.PRECONDITION), stillRefused.refusalReason());
    }

    private static PhysicalSensorSnapshot snapshot(long gameTick) {
        return new PhysicalSensorSnapshot(
                gameTick,
                "00000000-0000-0000-0000-000000000001",
                1.0,
                64.0,
                -2.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                false,
                false,
                false,
                false,
                false,
                12,
                10,
                8.0,
                8.0,
                6.0,
                5.0,
                8.0,
                2.5,
                3.0);
    }
}
