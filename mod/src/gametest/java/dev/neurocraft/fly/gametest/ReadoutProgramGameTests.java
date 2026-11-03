package dev.neurocraft.fly.gametest;

import static dev.neurocraft.fly.gametest.ReadoutLaneFence.BIASED;
import static dev.neurocraft.fly.gametest.ReadoutLaneFence.CRUISING;
import static dev.neurocraft.fly.gametest.ReadoutLaneFence.POLICY_HASH;
import static dev.neurocraft.fly.gametest.ReadoutLaneFence.READOUT;

import dev.neurocraft.fly.NeuroCraftFly;
import dev.neurocraft.fly.StagedProgramAuthority;
import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.BodyProgramRuntime;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.MotorAuthority;
import dev.neurocraft.fly.body.ProgramExecution;
import dev.neurocraft.fly.body.ProgramParameters;
import dev.neurocraft.fly.body.ProgramRefusalReason;
import dev.neurocraft.fly.body.ReadoutChannel;
import dev.neurocraft.fly.body.ReadoutValues;
import dev.neurocraft.fly.body.SelectionSource;
import dev.neurocraft.fly.body.programs.CruiseProgram;
import dev.neurocraft.fly.control.CommandValidator;
import dev.neurocraft.fly.control.ControlOperation;
import dev.neurocraft.fly.control.ControlProtocol;
import dev.neurocraft.fly.control.ControllerFault;
import dev.neurocraft.fly.control.ControllerIdentity;
import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import dev.neurocraft.fly.control.ProgramActionFrame;
import dev.neurocraft.fly.control.RawPrimitiveAction;
import dev.neurocraft.fly.control.ValidationResult;
import dev.neurocraft.fly.entity.FlyEntity;
import dev.neurocraft.fly.entity.FlySensors;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.function.LongFunction;
import net.fabricmc.fabric.api.gametest.v1.GameTest;
import net.minecraft.gametest.framework.GameTestHelper;
import net.minecraft.network.chat.Component;
import net.minecraft.world.phys.Vec3;

/**
 * Stage 4b pins for the READOUT lane: a body program selected each tick by a declared hand-chosen
 * mapping over the connectome's labeled readouts (ADR 0019), executed on the authoritative server.
 *
 * <p><strong>Nothing here is evidence about the connectome.</strong> Every readout value below is a
 * literal (see {@link ReadoutLaneFence}). What these tests pin is the authority model and the
 * truthfulness of the receipt: who owns the body, who loses it to whom, and whether the frame the
 * mod sends back describes what actually ran.
 *
 * <p><strong>What this lane cannot drive, stated plainly.</strong> The default GameTest lane runs
 * with no bridge configured, so {@code FlyControllerRuntime.control} always rejects and
 * {@code recordActuation} returns before it reaches a bridge. These tests therefore drive
 * {@link BodyProgramRuntime} and {@link CommandValidator} directly, exactly as the operator-lane
 * and validator pins in {@code FlyEntityGameTests} do, and encode the receipt through the same
 * {@link ControlProtocol#encodeProgramReceipt} call the bridge makes. The wiring from an admitted
 * frame to that call is covered by {@code NeuralBridgeTest} and by the opt-in external-bridge lane.
 */
public final class ReadoutProgramGameTests {
        /**
     * The normal path: an admitted program frame engages READOUT authority, the selected program
     * runs, and the receipt reports exactly what ran.
     */
    @GameTest(maxTicks = 20)
    public void readoutSelectionExecutesWhileRunningAndIsReceipted(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 3.0, 1.5));
        BodyProgramRuntime runtime = fly.bodyPrograms();
        PhysicalSensorSnapshot snapshot = FlySensors.sample(helper.getLevel(), fly);

        Optional<ProgramExecution> executed = runtime.advanceReadout(
                snapshot, BodyProgramId.CRUISE, CRUISING, BIASED, READOUT, POLICY_HASH);

        helper.assertTrue(
                executed.isPresent(),
                Component.literal("an admitted program frame produced no execution"));
        ProgramExecution execution = executed.orElseThrow();
        helper.assertTrue(
                runtime.authority() == MotorAuthority.READOUT
                        && runtime.selectionSource() == SelectionSource.READOUT,
                Component.literal("the readout lane did not take the body: "
                        + runtime.authority() + "/" + runtime.selectionSource()));
        helper.assertTrue(
                runtime.state().id() == BodyProgramId.CRUISE,
                Component.literal("the selected program did not enter"));
        helper.assertTrue(
                execution.applied() && execution.executedProgram() == BodyProgramId.CRUISE,
                Component.literal("the execution did not report the requested program"));
        helper.assertTrue(
                execution.executedModulation().equals(CRUISING)
                        && execution.executedParameters().equals(BIASED),
                Component.literal("the execution did not run under the requested modulation"));
        helper.assertTrue(
                runtime.readoutOwnedTicks() == 1
                        && runtime.readoutStalledTicks() == 0
                        && runtime.readoutRefusals() == 0,
                Component.literal("readout tick accounting is wrong"));
        helper.assertTrue(
                runtime.lastReadout().equals(READOUT)
                        && runtime.policyHash().equals(POLICY_HASH),
                Component.literal("the driving readouts and their mapping were not retained"));
        helper.assertTrue(
                runtime.transitionLedger().getLast().contains("source=READOUT reason=selected")
                        && runtime.transitionLedger().getLast().contains("readout=ttmn:0.0000")
                        && runtime.transitionLedger().getLast().contains("policy=77777777"),
                Component.literal(
                        "the transition did not name its readouts and mapping: "
                                + runtime.transitionLedger()));

        ProgramActionFrame frame = ReadoutLaneFence.programFrame(
                fly, snapshot.gameTick(), 0, BodyProgramId.CRUISE, CRUISING, BIASED);
        String receipt = ControlProtocol.encodeProgramReceipt(
                ReadoutLaneFence.identity(fly), frame, snapshot, executed,
                ValidationResult.accepted(frame));
        helper.assertTrue(
                receipt.contains("\"validation\":\"PROGRAM_APPLIED\"")
                        && receipt.contains("\"executed_program\":\"CRUISE\"")
                        && receipt.contains("\"reason\":null")
                        && receipt.contains("\"fault_code\":null"),
                Component.literal("the receipt did not report an applied program: " + receipt));
        helper.succeed();
    }

    /**
     * One driving frame kind per session. A session that has admitted a program frame refuses a
     * non-neutral raw action, the tick applies neutral motors, and the readout lane never sees it.
     */
    @GameTest(maxTicks = 20)
    public void rawActionAndProgramActionCannotBothApplyOnOneTick(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 3.0, 1.5));
        BodyProgramRuntime runtime = fly.bodyPrograms();
        ControllerIdentity identity = ReadoutLaneFence.identity(fly);
        CommandValidator validator = new CommandValidator(identity);
        long tick = FlySensors.sample(helper.getLevel(), fly).gameTick();
        // Observation ticks strictly increase live, so the two sequences are one tick apart;
        // each frame below answers its own, one action budget before the tick it targets.
        LongFunction<OptionalLong> receipts =
                ReadoutLaneFence.observationReceipts(Map.of(0L, tick - 1, 1L, tick));

        ValidationResult program = validator.validate(
                List.of(ReadoutLaneFence.programFrame(
                        fly, tick + 1, 0, BodyProgramId.CRUISE, CRUISING, BIASED)),
                tick + 1,
                ControllerFault.ACTIVE,
                receipts);
        helper.assertTrue(
                program.accepted()
                        && validator.lane() == CommandValidator.FrameLane.PROGRAM,
                Component.literal("the program frame did not latch the PROGRAM lane"));
        helper.assertTrue(
                program.appliedAction().isNeutral()
                        && program.rawAction().equals(RawPrimitiveAction.ZERO),
                Component.literal("an admitted program frame carried a motor request"));

        ValidationResult mixed = validator.validate(
                List.of(ReadoutLaneFence.rawAction(identity, tick + 2, 1, 0.75)),
                tick + 2,
                ControllerFault.ACTIVE,
                receipts);

        helper.assertTrue(
                !mixed.accepted() && mixed.state() == ControllerFault.PROTOCOL_ERROR,
                Component.literal("a mixed session did not fail closed: " + mixed.state()));
        helper.assertTrue(
                mixed.appliedAction().isNeutral(),
                Component.literal("a refused lane-mixing tick applied motors"));
        helper.assertTrue(
                runtime.readoutOwnedTicks() == 0 && runtime.programOwnedTicks() == 0,
                Component.literal("a refused frame reached the body"));
        helper.succeed();
    }

    /**
     * The operator wins, visibly and from both sides, and the readout lane comes back on release.
     *
     * <p>The ledger must show the readout lane losing the body with {@code reason=operator} and the
     * operator lane taking it with {@code reason=selected}. One line that blended the two would be
     * exactly the defect this project must never ship.
     */
    @GameTest(maxTicks = 20)
    public void operatorProgramPreemptsReadoutAndReadoutResumesOnRelease(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 3.0, 1.5));
        BodyProgramRuntime runtime = fly.bodyPrograms();
        PhysicalSensorSnapshot snapshot = FlySensors.sample(helper.getLevel(), fly);
        long issuedTick = helper.getLevel().getGameTime();
        runtime.advanceReadout(
                snapshot, BodyProgramId.CRUISE, CRUISING, BIASED, READOUT, POLICY_HASH);
        helper.assertTrue(
                runtime.authority() == MotorAuthority.READOUT,
                Component.literal("the readout lane did not engage"));

        // The acknowledgement is stamped the way the service stamps it -- with the tick of the
        // observation it first answered paused -- and the transaction engages one action budget
        // later, at its engage tick, which is the tick the paused service's first neutral command
        // governs: the entity polls acknowledgements before it sends that observation, so no tick
        // can hold an acknowledgement on its own effective tick (StagedProgramAuthorityTest and
        // the opt-in ExternalBridgeProgramAuthorityGameTest carry that relation).
        StagedProgramAuthority pause = StagedProgramAuthority.engage(
                fly.getUUID(), issuedTick, BodyProgramId.HOVER, Modulation.NEUTRAL);
        pause.bindControlSequence(0);
        helper.assertTrue(
                pause.advance(
                                issuedTick,
                                Optional.of(ReadoutLaneFence.controlAck(
                                        fly, 0, issuedTick, ControlOperation.PAUSE)))
                        == StagedProgramAuthority.Outcome.WAITING,
                Component.literal("the operator PAUSE engaged on its own effective tick"));
        long pauseEngageTick = pause.engageTick();
        helper.assertTrue(
                pause.advance(pauseEngageTick, Optional.empty())
                        == StagedProgramAuthority.Outcome.READY,
                Component.literal("the operator PAUSE was not ready at its engage tick"));
        pause.apply(runtime, pauseEngageTick);

        helper.assertTrue(
                runtime.authority() == MotorAuthority.SCRIPTED
                        && runtime.selectionSource() == SelectionSource.OPERATOR
                        && runtime.state().id() == BodyProgramId.HOVER,
                Component.literal("the operator did not take the body from the readout lane"));
        List<String> ledger = runtime.transitionLedger();
        String lost = ledger.get(ledger.size() - 2);
        String taken = ledger.getLast();
        helper.assertTrue(
                lost.contains("source=READOUT reason=operator"),
                Component.literal("the readout lane did not lose the body on the record: " + lost));
        helper.assertTrue(
                taken.contains("source=OPERATOR reason=selected") && !taken.contains("readout="),
                Component.literal("the operator selection named a readout: " + taken));
        helper.assertTrue(
                runtime.lastReadout().equals(ReadoutValues.ZERO) && runtime.policyHash().isEmpty(),
                Component.literal("a readout kept driving the HUD after the operator took over"));
        // The operator command carries no signed parameter, so the operator lane must be running
        // at zero bias -- on the runtime and on the channel the HUD draws. A leftover turn bias
        // under an OPERATOR label would be a line that blends the two lanes.
        helper.assertTrue(
                runtime.parameters().isNeutral()
                        && runtime.channel().parameters().isNeutral(),
                Component.literal("the operator lane inherited the readout's turn bias: "
                        + runtime.parameters()));

        // A program frame arriving while the operator holds the body is refused, and the receipt
        // names the operator's own program rather than inventing a HOVER that did not run.
        Optional<ProgramExecution> refused = runtime.advanceReadout(
                snapshot, BodyProgramId.CRUISE, CRUISING, BIASED, READOUT, POLICY_HASH);
        helper.assertTrue(
                refused.isPresent()
                        && refused.orElseThrow().refusalReason()
                                .equals(Optional.of(ProgramRefusalReason.OPERATOR))
                        && refused.orElseThrow().executedProgram() == BodyProgramId.HOVER,
                Component.literal("the operator refusal did not report what actually ran"));
        helper.assertTrue(
                refused.orElseThrow().executedParameters().isNeutral(),
                Component.literal("a refused frame's turn bias was reported as having run: "
                        + refused.orElseThrow().executedParameters()));
        helper.assertTrue(
                runtime.authority() == MotorAuthority.SCRIPTED
                        && runtime.selectionSource() == SelectionSource.OPERATOR,
                Component.literal("a refused program frame moved the operator's authority"));

        long resumeIssuedTick = pauseEngageTick;
        StagedProgramAuthority resume =
                StagedProgramAuthority.release(fly.getUUID(), resumeIssuedTick);
        resume.bindControlSequence(1);
        helper.assertTrue(
                resume.advance(
                                resumeIssuedTick,
                                Optional.of(ReadoutLaneFence.controlAck(
                                        fly, 1, resumeIssuedTick, ControlOperation.RESUME)))
                        == StagedProgramAuthority.Outcome.WAITING,
                Component.literal("the operator RESUME released on its own effective tick"));
        long resumeEngageTick = resume.engageTick();
        helper.assertTrue(
                resume.advance(resumeEngageTick, Optional.empty())
                        == StagedProgramAuthority.Outcome.READY,
                Component.literal("the operator RESUME was not ready at its engage tick"));
        resume.apply(runtime, resumeEngageTick);
        helper.assertTrue(
                runtime.authority() == MotorAuthority.NEURAL,
                Component.literal("RESUME did not return the body to the neural lane"));

        runtime.advanceReadout(
                snapshot, BodyProgramId.CRUISE, CRUISING, BIASED, READOUT, POLICY_HASH);
        helper.assertTrue(
                runtime.authority() == MotorAuthority.READOUT
                        && runtime.selectionSource() == SelectionSource.READOUT
                        && runtime.transitionLedger().getLast()
                                .contains("source=READOUT reason=selected"),
                Component.literal("the readout lane did not resume after release"));
        helper.succeed();
    }

    /**
     * A stalled window is a neutral tick, not a free tick of the running program. Authority is
     * retained and the program's own clock is frozen, so a body under READOUT never advances a
     * program on a tick nothing selected one.
     */
    @GameTest(maxTicks = 20)
    public void emptyWindowUnderReadoutHoldsNeutralAndRetainsAuthority(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 3.0, 1.5));
        BodyProgramRuntime runtime = fly.bodyPrograms();
        PhysicalSensorSnapshot snapshot = FlySensors.sample(helper.getLevel(), fly);
        runtime.advanceReadout(
                snapshot, BodyProgramId.CRUISE, CRUISING, BIASED, READOUT, POLICY_HASH);
        int ticksInProgram = runtime.state().ticksInProgram();
        long ownedTicks = runtime.programOwnedTicks();

        helper.assertTrue(
                runtime.advance(snapshot, false, false).isEmpty(),
                Component.literal("a program owned a tick with no admitted command"));

        helper.assertTrue(
                runtime.authority() == MotorAuthority.READOUT
                        && runtime.selectionSource() == SelectionSource.READOUT,
                Component.literal("a stalled window dropped readout authority"));
        helper.assertTrue(
                runtime.readoutStalledTicks() == 1,
                Component.literal("the stalled tick was not counted"));
        helper.assertTrue(
                runtime.state().ticksInProgram() == ticksInProgram
                        && runtime.state().id() == BodyProgramId.CRUISE
                        && runtime.programOwnedTicks() == ownedTicks,
                Component.literal("a stalled tick advanced the program's own state"));
        helper.succeed();
    }

    /**
     * A neutral admitted command whose neural interval did not move is the paused service's
     * signature. The connectome is not stepping, so no readout can be selecting, and the body drops
     * to the neural lane rather than running a selection nothing is making.
     */
    @GameTest(maxTicks = 20)
    public void pausedServiceReleasesReadoutAuthority(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 3.0, 1.5));
        BodyProgramRuntime runtime = fly.bodyPrograms();
        PhysicalSensorSnapshot snapshot = FlySensors.sample(helper.getLevel(), fly);
        runtime.advanceReadout(
                snapshot, BodyProgramId.CRUISE, CRUISING, BIASED, READOUT, POLICY_HASH);

        helper.assertTrue(
                runtime.advance(snapshot, true, true).isEmpty(),
                Component.literal("a program owned a tick under a paused service"));

        helper.assertTrue(
                runtime.authority() == MotorAuthority.NEURAL
                        && runtime.selectionSource() == SelectionSource.DEFAULT
                        && runtime.state().id() == BodyProgramId.HOVER,
                Component.literal("a paused service did not release readout authority"));
        helper.assertTrue(
                runtime.transitionLedger().getLast().contains("source=READOUT reason=paused"),
                Component.literal("the release was not logged against the readout lane: "
                        + runtime.transitionLedger()));
        helper.succeed();
    }

    /** Missing food/support refuses registered FEED to HOVER with the actual precondition receipt. */
    @GameTest(maxTicks = 20)
    public void missingContextRefusesFeedToHoverWithAPreconditionReceipt(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 3.0, 1.5));
        BodyProgramRuntime runtime = fly.bodyPrograms();
        PhysicalSensorSnapshot snapshot = FlySensors.sample(helper.getLevel(), fly);
        helper.assertTrue(
                BodyProgramRuntime.registeredPrograms().contains(BodyProgramId.FEED),
                Component.literal("FEED must be registered before testing its precondition"));

        Optional<ProgramExecution> executed = runtime.advanceReadout(
                snapshot, BodyProgramId.FEED, CRUISING, BIASED, READOUT, POLICY_HASH);

        helper.assertTrue(executed.isPresent(), Component.literal("a refusal produced no receipt"));
        ProgramExecution execution = executed.orElseThrow();
        helper.assertTrue(
                execution.executedProgram() == BodyProgramId.HOVER
                        && execution.refusalReason()
                                .equals(Optional.of(ProgramRefusalReason.PRECONDITION)),
                Component.literal("the refusal did not fall back to HOVER"));
        helper.assertTrue(
                execution.executedModulation().equals(CRUISING)
                        && execution.executedParameters().equals(BIASED),
                Component.literal("the fallback did not run under the requested modulation"));
        helper.assertTrue(
                runtime.state().id() == BodyProgramId.HOVER
                        && runtime.authority() == MotorAuthority.READOUT
                        && runtime.readoutRefusals() == 1,
                Component.literal("the refusal left the runtime in the wrong state"));

        ProgramActionFrame frame = ReadoutLaneFence.programFrame(
                fly, snapshot.gameTick(), 0, BodyProgramId.FEED, CRUISING, BIASED);
        String receipt = ControlProtocol.encodeProgramReceipt(
                ReadoutLaneFence.identity(fly), frame, snapshot, executed,
                ValidationResult.accepted(frame));
        helper.assertTrue(
                receipt.contains("\"validation\":\"PROGRAM_REFUSED\"")
                        && receipt.contains("\"reason\":\"precondition\"")
                        && receipt.contains("\"requested_program\":\"FEED\"")
                        && receipt.contains("\"executed_program\":\"HOVER\""),
                Component.literal("the refusal receipt is not truthful: " + receipt));
        helper.succeed();
    }

    /**
     * <strong>A rejected program frame never reaches the body.</strong>
     *
     * <p>A {@code ValidationResult} that refused a frame still carries that frame's envelope --
     * that is how the authoritative REJECTED receipt binds what it refused -- so an arbitration
     * that keyed on "the decision carries a program frame" instead of "a command was admitted"
     * would hand every terminal rejection's selection straight to the readout lane: the body would
     * move on a command the validator had just refused, the ledger would log a selection that never
     * happened, and the REJECTED receipt would be unencodable and never sent.
     *
     * <p>This drives the real tick path -- {@code control()} into the real {@link CommandValidator}
     * and out through the real {@code recordActuation} -- against a recording bridge, because the
     * claim is about the wiring between the two, not about either end.
     */
    @GameTest(maxTicks = 20)
    public void aRejectedProgramFrameNeverReachesTheBody(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 3.0, 1.5));
        BodyProgramRuntime runtime = fly.bodyPrograms();
        Vec3 before = fly.position();
        long tick = helper.getLevel().getGameTime();

        try (RecordingBridgeFence.Installed installed = RecordingBridgeFence.install(fly)) {
            // command_seq 5 on a session whose last sequence is -1: a SEQUENCE_ERROR, refused
            // before the observation lookup and latched terminal.
            installed.bridge().offer(ReadoutLaneFence.programFrame(
                    fly, tick + 1, 5, BodyProgramId.CRUISE, CRUISING, BIASED));

            fly.tick();

            helper.assertTrue(
                    runtime.authority() == MotorAuthority.NEURAL
                            && runtime.selectionSource() == SelectionSource.DEFAULT,
                    Component.literal("a rejected program frame engaged the readout lane: "
                            + runtime.authority() + "/" + runtime.selectionSource()));
            helper.assertTrue(
                    runtime.transitions() == 0 && runtime.transitionLedger().isEmpty(),
                    Component.literal("a rejected program frame was logged as a selection: "
                            + runtime.transitionLedger()));
            helper.assertTrue(
                    runtime.readoutOwnedTicks() == 0 && runtime.programOwnedTicks() == 0,
                    Component.literal("a rejected program frame owned a tick"));
            helper.assertTrue(
                    fly.lastActuation().appliedAction().isNeutral(),
                    Component.literal("a rejected program frame applied motors: "
                            + fly.lastActuation().appliedAction()));
            helper.assertTrue(
                    fly.lastAcceptedCommand().isEmpty(),
                    Component.literal("a rejected frame was recorded as the last accepted command"));
            helper.assertTrue(
                    fly.position().distanceTo(before) < 1.0e-9,
                    Component.literal("a rejected program frame moved the fly "
                            + fly.position().distanceTo(before) + " blocks"));

            List<String> receipts = installed.bridge().programReceipts();
            helper.assertTrue(
                    receipts.size() == 1,
                    Component.literal("expected exactly one program receipt, got " + receipts));
            String receipt = receipts.getFirst();
            helper.assertTrue(
                    receipt.contains("\"validation\":\"REJECTED\"")
                            && receipt.contains("\"fault_code\":\"SEQUENCE\"")
                            && receipt.contains("\"executed_program\":null")
                            && receipt.contains("\"reason\":null"),
                    Component.literal("the terminal receipt is not an honest rejection: " + receipt));
        }
        helper.succeed();
    }

    /**
     * A latched controller fault ends the session, so it must also end READOUT authority.
     *
     * <p>Otherwise the body holds neutral forever with the HUD still drawing the green readout
     * lines under a red CONTROL line, and the operator is wedged: a RESET is refused while a body
     * program holds authority, and {@code program release} issues a RESUME that a service which was
     * never paused will not accept.
     */
    @GameTest(maxTicks = 20)
    public void aLatchedFaultReleasesReadoutAuthority(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 3.0, 1.5));
        BodyProgramRuntime runtime = fly.bodyPrograms();
        PhysicalSensorSnapshot snapshot = FlySensors.sample(helper.getLevel(), fly);
        runtime.advanceReadout(
                snapshot, BodyProgramId.CRUISE, CRUISING, BIASED, READOUT, POLICY_HASH);
        helper.assertTrue(
                runtime.authority() == MotorAuthority.READOUT,
                Component.literal("the readout lane did not engage"));

        // This lane has no bridge configured, so control() reports MISSING_CONFIGURATION: a
        // paused, terminal controller state, which is exactly the shape of any latched fault.
        fly.tick();

        helper.assertTrue(
                runtime.authority() == MotorAuthority.NEURAL
                        && runtime.selectionSource() == SelectionSource.DEFAULT
                        && runtime.state().id() == BodyProgramId.HOVER,
                Component.literal("a latched fault did not release readout authority: "
                        + runtime.authority()));
        helper.assertTrue(
                runtime.transitionLedger().getLast().contains("source=READOUT reason=faulted"),
                Component.literal("the release was not logged as a fault: "
                        + runtime.transitionLedger()));
        helper.assertTrue(
                fly.lastActuation().appliedAction().isNeutral(),
                Component.literal("a faulted tick applied motors"));
        helper.succeed();
    }

    /**
     * <strong>The cross-language sign pin: a {@code turn_bias} on the wire steers the real body,
     * and the direction it steers it is the one the policy declares.</strong>
     *
     * <p>D-077 declares {@code dna01_turn} positive to mean "toward the body's right", and until
     * this pin nothing measured that end of the chain: the mod carried the number, published it,
     * and reported it, and the only check on its sign was prose. This drives an admitted
     * {@code program_action} carrying {@code turn_bias = +1} and then one carrying {@code -1}
     * through the whole real path -- {@code FlyEntity.tick} into the real {@code CommandValidator},
     * the real arbitration, {@code advanceReadout}, CRUISE, the real actuator, the entity's own
     * move and drag -- and measures which way the body's world-space path curved, with "the body's
     * right" taken from the engine's own {@code Direction.getClockWise()} table by way of
     * {@link RaySideFence}.
     *
     * <p><strong>The whole drive runs inside one server tick, deliberately.</strong> The recording
     * bridge is a process-global field, and GameTests in a batch tick together, so an installation
     * held for {@link TurnBiasFence#CHAIN_TICKS} server ticks would be an ACTIVE bridge under every
     * other test in the batch -- including the one whose premise is that no bridge is configured.
     * Calling {@code fly.tick()} in a loop gives the body its own ticks (each one samples the world,
     * validates a frame, steps the program, actuates, and integrates the move and the drag) without
     * advancing the world clock, so the path is a real flown path and the exposure is a fraction of
     * a tick. What it does not exercise is anything that depends on the game clock advancing, which
     * for CRUISE is nothing: its own clock is {@code ticksInProgram}.
     *
     * <p>The run's premise -- that avoidance never took the wheel -- is asserted from the sampled
     * rays rather than assumed; see {@link TurnBiasFence#CHAIN_MIN_FORWARD_CLEARANCE_BLOCKS}.
     *
     * <p><strong>What it does not establish.</strong> The frame here is a test fixture built from
     * literals ({@link ReadoutLaneFence}), so this says nothing about whether the neural service
     * computes {@code dna01_turn} with the sign the readout census suggests -- that is the Python
     * half of D-077's chain and has its own tests. This closes the half that had none: a
     * {@code turn_bias} that arrives on the wire moves the body the way the declaration says it
     * will.
     *
     * <p><strong>And the world clock is frozen, so half the validator is exercised and half is
     * not.</strong> {@code streamedProgramFrame} stamps all sixty frames with
     * {@code observation_tick == target_tick ==} the one server tick this runs on, so the
     * <em>freshness</em> gates -- command age, the two-tick validity window, lateness retirement,
     * the cold-start and actuation holds -- are only ever seen in their degenerate age-zero case.
     * They are pinned where they belong, in {@code FlyEntityGameTests}' validator lane. What this
     * pin does genuinely drive is everything that does not depend on the clock advancing: the
     * identity check, the frame-lane latch, command-sequence continuity, the observation-receipt
     * lookup and its recorded-tick agreement, and neural-interval continuity, sixty frames deep.
     */
    @GameTest(maxTicks = 40)
    public void readoutTurnBiasSteersTheBodyThroughTheRealValidator(GameTestHelper helper) {
        double[] lateral = new double[2];
        double[] turned = new double[2];
        double[] biases = {1.0, -1.0};
        for (int index = 0; index < biases.length; index++) {
            FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(4.0, 4.0, 4.0));
            fly.setYRot(0.0F);
            List<CruiseFlightFence.Flight> flight =
                    driveAdmittedProgramFrames(helper, fly, new ProgramParameters(biases[index]));
            lateral[index] = TurnBiasFence.lateralOffsetTowardBodyRight(flight, 0.0F);
            turned[index] = TurnBiasFence.turnedTowardBodyRight(flight, 0.0F);
        }

        String measured = String.format(
                Locale.ROOT,
                "turn_bias +1 moved the body %+.4f blocks toward its own right and turned its path"
                        + " %+.1f degrees; turn_bias -1 moved it %+.4f and turned it %+.1f",
                lateral[0],
                turned[0],
                lateral[1],
                turned[1]);
        helper.assertTrue(
                lateral[0] >= TurnBiasFence.CHAIN_LATERAL_BLOCKS
                        && turned[0] >= TurnBiasFence.CHAIN_PATH_TURN_DEGREES,
                Component.literal(
                        "an admitted turn_bias of +1 must steer the body toward its right: "
                                + measured));
        helper.assertTrue(
                lateral[1] <= -TurnBiasFence.CHAIN_LATERAL_BLOCKS
                        && turned[1] <= -TurnBiasFence.CHAIN_PATH_TURN_DEGREES,
                Component.literal(
                        "an admitted turn_bias of -1 must steer the body toward its left: "
                                + measured));
        helper.succeed();
    }

    /**
     * Streams {@link TurnBiasFence#CHAIN_TICKS} admitted {@code program_action} frames into one
     * fly through the real tick path, and returns the world-space path it flew.
     *
     * <p>The installation is closed on every exit, including a failed assertion: seventy GameTests
     * share one server and a leaked bridge would poison every one that ran afterwards.
     */
    private static List<CruiseFlightFence.Flight> driveAdmittedProgramFrames(
            GameTestHelper helper, FlyEntity fly, ProgramParameters parameters) {
        BodyProgramRuntime runtime = fly.bodyPrograms();
        Vec3 origin = helper.absoluteVec(Vec3.ZERO);
        List<CruiseFlightFence.Flight> flight = new ArrayList<>();
        double minimumForward = FlySensors.RAY_LENGTH_BLOCKS;
        double minimumSide = FlySensors.RAY_LENGTH_BLOCKS;
        try (RecordingBridgeFence.Installed installed = RecordingBridgeFence.install(fly)) {
            // One priming tick with nothing queued, so the bridge records the observation the
            // first frame will answer. Asking the bridge which sequence it assigned is the point:
            // a stream anchored on an assumed counter would pass for the wrong reason.
            fly.tick();
            long tick = helper.getLevel().getGameTime();
            long observation = installed.bridge()
                    .observationSequenceAt(tick)
                    .orElseThrow(() -> new AssertionError(
                            "the priming tick recorded no observation at tick " + tick));

            for (int driven = 0; driven < TurnBiasFence.CHAIN_TICKS; driven++) {
                installed.bridge().offer(ReadoutLaneFence.streamedProgramFrame(
                        fly,
                        driven,
                        observation + driven,
                        tick,
                        tick,
                        BodyProgramId.CRUISE,
                        TurnBiasFence.CHAIN_MODULATION,
                        parameters));
                PhysicalSensorSnapshot seen = FlySensors.sample(helper.getLevel(), fly);
                minimumForward = Math.min(minimumForward, seen.rayForwardBlocks());
                minimumSide = Math.min(
                        minimumSide, Math.min(seen.rayLeftBlocks(), seen.rayRightBlocks()));
                fly.tick();
                flight.add(new CruiseFlightFence.Flight(
                        fly.getX() - origin.x,
                        fly.getZ() - origin.z,
                        fly.getYRot(),
                        seen.rayUpBlocks(),
                        seen.rayDownBlocks(),
                        seen.horizontalCollision(),
                        seen.verticalCollision()));
            }

            helper.assertTrue(
                    runtime.readoutOwnedTicks() == TurnBiasFence.CHAIN_TICKS
                            && runtime.readoutRefusals() == 0
                            && runtime.readoutStalledTicks() == 0,
                    Component.literal(String.format(
                            Locale.ROOT,
                            "the streamed frames did not all reach the body: owned %d, refused %d,"
                                    + " stalled %d, of %d",
                            runtime.readoutOwnedTicks(),
                            runtime.readoutRefusals(),
                            runtime.readoutStalledTicks(),
                            TurnBiasFence.CHAIN_TICKS)));
            List<String> receipts = installed.bridge().programReceipts();
            helper.assertTrue(
                    receipts.size() == TurnBiasFence.CHAIN_TICKS
                            && receipts.stream().allMatch(receipt ->
                                    receipt.contains("\"validation\":\"PROGRAM_APPLIED\"")
                                            && receipt.contains(
                                                    "\"executed_program\":\"CRUISE\"")),
                    Component.literal("the streamed frames were not all receipted as applied: "
                            + receipts.size() + " receipts"));
            helper.assertTrue(
                    minimumForward >= TurnBiasFence.CHAIN_MIN_FORWARD_CLEARANCE_BLOCKS
                            && minimumSide >= CruiseProgram.CRUISE_SIDE_CLEARANCE_BLOCKS
                            && CruiseFlightFence.wallContactTicks(flight) == 0,
                    Component.literal(String.format(
                            Locale.ROOT,
                            "the chain run left its declared premise: the forward ray fell to"
                                    + " %.3f (floor %.2f), the nearest side ray to %.3f (floor"
                                    + " %.2f), on %d contact ticks",
                            minimumForward,
                            TurnBiasFence.CHAIN_MIN_FORWARD_CLEARANCE_BLOCKS,
                            minimumSide,
                            CruiseProgram.CRUISE_SIDE_CLEARANCE_BLOCKS,
                            CruiseFlightFence.wallContactTicks(flight))));
        }
        return flight;
    }

    /**
     * The published channels carry what the HUD draws: the selection source, the program, the
     * signed turn bias, and the driving readouts, at their declared presentation resolution.
     */
    @GameTest(maxTicks = 20)
    public void theReadoutChannelsPublishTheSelectionAndItsDrivingReadouts(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 3.0, 1.5));
        BodyProgramRuntime runtime = fly.bodyPrograms();
        PhysicalSensorSnapshot snapshot = FlySensors.sample(helper.getLevel(), fly);
        runtime.advanceReadout(
                snapshot,
                BodyProgramId.ESCAPE_JUMP,
                new Modulation(1.0, 1.0, 1.0, 0.8),
                BIASED,
                new ReadoutValues(0.4213, -0.0120, 0.6149),
                POLICY_HASH);

        helper.assertTrue(
                runtime.channel().source() == SelectionSource.READOUT
                        && runtime.channel().program() == BodyProgramId.ESCAPE_JUMP,
                Component.literal("the body-program channel does not name the readout selection"));
        helper.assertTrue(
                Math.abs(runtime.channel().parameters().turnBias() + 0.25) < 1.0e-9,
                Component.literal("the signed turn bias did not reach the published channel"));
        ReadoutValues published =
                ReadoutChannel.decode(runtime.readoutChannel().packed()).readout();
        helper.assertTrue(
                Math.abs(published.ttmnJump() - 0.42) < 1.0e-9
                        && Math.abs(published.dna01Turn() + 0.01) < 1.0e-9
                        && Math.abs(published.wingCommon() - 0.61) < 1.0e-9,
                Component.literal(
                        "the readout channel did not publish at hundredths: " + published));
        helper.succeed();
    }
}
