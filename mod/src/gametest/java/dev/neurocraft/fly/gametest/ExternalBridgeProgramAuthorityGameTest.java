package dev.neurocraft.fly.gametest;

import dev.neurocraft.fly.FlyControllerRuntime;
import dev.neurocraft.fly.NeuroCraftFly;
import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.MotorAuthority;
import dev.neurocraft.fly.body.SelectionSource;
import dev.neurocraft.fly.control.ActionFrame;
import dev.neurocraft.fly.control.ControlProtocol;
import dev.neurocraft.fly.control.ControllerFault;
import dev.neurocraft.fly.entity.FlyEntity;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import net.fabricmc.fabric.api.gametest.v1.GameTest;
import net.minecraft.gametest.framework.GameTestHelper;
import net.minecraft.network.chat.Component;
import net.minecraft.world.entity.EntitySpawnReason;
import net.minecraft.world.phys.Vec3;

/**
 * Opt-in live probe of the ADR 0020 operator lane: the PAUSE-gated motor-authority transaction
 * driven end to end against a <strong>real</strong> neural service over the real bridge, with the
 * real control acknowledgements the service stamps. Absent from the default GameTest entrypoint
 * list; selected by {@code -Dneurocraft.fly.externalBridgeTest=true
 * -Dneurocraft.fly.externalBridgeScenario=program_authority} and launched by
 * {@code scripts/run_program_authority_gametest.py}.
 *
 * <p>It exists because the default-lane pins cannot express the one thing that decides whether
 * this lane is reachable at all: <em>when</em> the acknowledgement becomes visible to the entity
 * tick relative to the tick it names. Those pins hand-build acknowledgements with
 * {@code effectiveTick == currentTick}; the service never produces that relation at the mod's
 * poll, and before the fix pinned by {@code StagedProgramAuthorityTest} every real PAUSE failed
 * {@code MISSED/LATE_ACK} here. The probe issues the transaction exactly as
 * {@code /neurocraftfly program hover} does -- from a server-thread callback that runs after the
 * entity tick, with {@code issuedTick = level.getGameTime()} -- and asserts the whole lane:
 * connectome driving, PAUSE acknowledged and scripted authority engaged, program-owned ticks over a
 * frozen neural cursor, RESUME acknowledged and authority released by the transaction (not by the
 * preemption interlock), the connectome driving again from the same cursor, and finally an
 * operator RESET applied at its own accepted effective tick.
 *
 * <p><strong>What it cannot exercise.</strong> It runs against the raw-{@code action} fixture
 * profile, so the READOUT lane of ADR 0021 -- and its handover to the operator on PAUSE
 * ({@code reason=operator}) and back after RESUME -- is out of reach here: the only profile that
 * issues {@code program_action} frames is the full-artifact {@code full_mcns_loom_program_v1}.
 * That handover is pinned synthetically in {@code ReadoutProgramGameTests} and has not run live.
 *
 * <p><strong>Nothing here is a claim about the connectome.</strong> The body program is scripted
 * presentation; the connectome is asserted <em>not</em> to be driving while it runs. The receipt
 * this writes is engineering evidence about the transport and the transaction, nothing more.
 */
public final class ExternalBridgeProgramAuthorityGameTest {
    private static final int ACTIVE_DEADLINE_TICKS = 200;
    private static final int DRIVING_DEADLINE_TICKS = 200;
    /**
     * The transaction resolves by {@code issuedTick + ACK_DEADLINE_TICKS + 1} by construction; the
     * probe allows a little more so a slow tick reports the transaction's own verdict, not the
     * probe's.
     */
    private static final int TRANSACTION_DEADLINE_TICKS = 10;
    /** One second of scripted authority at 20 Hz. */
    private static final int REQUIRED_PROGRAM_OWNED_TICKS = 20;
    private static final int RESET_DEADLINE_TICKS = 10;
    private static final double POSITION_EPSILON = 1.0e-6;
    static final String SCHEMA = "neurocraft.external_bridge_program_authority.v1";

    @GameTest(maxTicks = 1200)
    public void liveServicePauseAcknowledgementEngagesTheOperatorProgramLane(
            GameTestHelper helper) {
        if (!Boolean.getBoolean("neurocraft.fly.externalBridgeTest")) {
            helper.fail(Component.literal(
                    "external bridge test class loaded without explicit opt-in"));
            return;
        }
        Optional<UUID> configuredUuid = FlyControllerRuntime.configuredEntityUuid();
        if (configuredUuid.isEmpty()) {
            helper.fail(Component.literal(
                    "external bridge test requires a valid configured Minecraft entity UUID"));
            return;
        }
        if (!helper.getLevel().getEntities(NeuroCraftFly.FLY, fly -> true).isEmpty()) {
            helper.fail(Component.literal("external bridge test requires an empty fly fixture"));
            return;
        }
        Probe probe = new Probe(
                configuredUuid.orElseThrow(),
                helper.absoluteVec(new Vec3(1.5, 3.0, 1.5)),
                helper.getLevel().getGameTime());
        helper.onEachTick(() -> probe.tick(helper));
    }

    private static final class Probe {
        private final UUID configuredUuid;
        private final Vec3 initialPosition;
        private final long initialGameTick;
        private final ExternalGameTestTickPacer tickPacer = ExternalGameTestTickPacer.system();
        private Stage stage = Stage.WAIT_ACTIVE;
        private FlyEntity fly;
        private long stageStartTick;

        private long drivingCommandSequence = -1;
        private long neuralCursorBeforePause = -1;

        private long pauseIssuedTick = -1;
        private long engagedTick = -1;
        private long ownedTicksAtEngage = -1;
        private String engageStatus;
        private String engageMessage;
        private String programFailure;

        private long pausedCommandsObserved;
        private long lastPausedCommandSequence = -1;
        private long frozenNeuralCursor = -1;
        private double maxAbsVerticalThrustWhileScripted;

        private long releaseIssuedTick = -1;
        private long releasedTick = -1;
        private long programOwnedTicksAtRelease = -1;
        private String releaseStatus;
        private String releaseMessage;
        private String releaseLedgerLine;

        private ActionFrame resumedCommand;
        private long commandsSkippedSincePause = -1;

        private long resetIssuedTick = -1;
        private long resetAppliedTick = -1;
        private Vec3 resetTarget;
        private String resetStatus;
        private String resetMessage;
        private String resetFailure;

        private CompletableFuture<Void> receiptWrite;

        private Probe(UUID configuredUuid, Vec3 initialPosition, long initialGameTick) {
            this.configuredUuid = configuredUuid;
            this.initialPosition = initialPosition;
            this.initialGameTick = initialGameTick;
            this.stageStartTick = initialGameTick;
        }

        private void tick(GameTestHelper helper) {
            try {
                tickPacer.pace();
            } catch (IllegalStateException exception) {
                helper.fail(Component.literal(exception.getMessage()));
                return;
            }
            ControllerFault bridgeState = FlyControllerRuntime.state();
            if (stage == Stage.WAIT_ACTIVE) {
                awaitActive(helper, bridgeState);
                return;
            }
            if (bridgeState != ControllerFault.ACTIVE) {
                helper.fail(Component.literal(
                        "external bridge left ACTIVE state: " + bridgeState
                                + " during " + stage
                                + "; program_authority=" + FlyControllerRuntime.programAuthorityStatus()
                                + " reset=" + FlyControllerRuntime.resetStatus()));
                return;
            }
            switch (stage) {
                case WAIT_DRIVING -> awaitDriving(helper);
                case WAIT_ENGAGED -> awaitEngaged(helper);
                case WAIT_PROGRAM_TICKS -> awaitProgramTicks(helper);
                case WAIT_RELEASED -> awaitReleased(helper);
                case WAIT_DRIVING_AGAIN -> awaitDrivingAgain(helper);
                case WAIT_RESET -> awaitReset(helper);
                case WAIT_RECEIPT_WRITE -> awaitReceiptWrite(helper);
                case WAIT_ACTIVE -> throw new IllegalStateException("stage already handled");
            }
        }

        private void enter(GameTestHelper helper, Stage next) {
            stage = next;
            stageStartTick = helper.getLevel().getGameTime();
        }

        private boolean pastDeadline(GameTestHelper helper, int ticks, String what) {
            if (helper.getLevel().getGameTime() - stageStartTick <= ticks) {
                return false;
            }
            helper.fail(Component.literal(what + " did not happen within " + ticks + " ticks"
                    + "; program_authority=" + FlyControllerRuntime.programAuthorityStatus()
                    + " reset=" + FlyControllerRuntime.resetStatus()
                    + " body=" + fly.bodyPrograms().authority()
                    + " ledger=" + fly.bodyPrograms().transitionLedger()));
            return true;
        }

        private void awaitActive(GameTestHelper helper, ControllerFault bridgeState) {
            if (bridgeState == ControllerFault.ACTIVE) {
                if (!spawnConfiguredFly(helper)) {
                    return;
                }
                enter(helper, Stage.WAIT_DRIVING);
                return;
            }
            if (bridgeState != ControllerFault.STARTING) {
                helper.fail(Component.literal(
                        "external bridge failed before handshake: " + bridgeState));
                return;
            }
            if (helper.getLevel().getGameTime() - initialGameTick > ACTIVE_DEADLINE_TICKS) {
                helper.fail(Component.literal("external bridge handshake exceeded 200 ticks"));
            }
        }

        private boolean spawnConfiguredFly(GameTestHelper helper) {
            FlyEntity candidate =
                    NeuroCraftFly.FLY.create(helper.getLevel(), EntitySpawnReason.COMMAND);
            if (candidate == null) {
                helper.fail(Component.literal("could not construct external bridge fly"));
                return false;
            }
            candidate.setUUID(configuredUuid);
            candidate.resetForFixture(
                    initialPosition.x, initialPosition.y, initialPosition.z, 0.0F);
            if (!helper.getLevel().addFreshEntity(candidate)) {
                helper.fail(Component.literal("could not spawn configured external bridge fly"));
                return false;
            }
            fly = candidate;
            return true;
        }

        /** The connectome must be stepping before the pause, or "paused" would prove nothing. */
        private void awaitDriving(GameTestHelper helper) {
            Optional<ActionFrame> latest = lastAcceptedAction();
            if (latest.isEmpty() || !stepped(latest.orElseThrow())) {
                pastDeadline(helper, DRIVING_DEADLINE_TICKS, "a stepped neural command");
                return;
            }
            ActionFrame command = latest.orElseThrow();
            drivingCommandSequence = command.commandSequence();
            neuralCursorBeforePause = command.neuralStepEnd();
            // Exactly what FlyCommands.program does: the request is made from the server thread
            // after this tick's entity tick, with the level's current game time as issuedTick.
            pauseIssuedTick = helper.getLevel().getGameTime();
            boolean queued = FlyControllerRuntime.requestProgramAuthority(
                    fly,
                    pauseIssuedTick,
                    BodyProgramId.HOVER,
                    Modulation.NEUTRAL,
                    message -> engageMessage = message,
                    message -> programFailure = message);
            if (!queued) {
                helper.fail(Component.literal("scripted body-program authority was not queued"));
                return;
            }
            enter(helper, Stage.WAIT_ENGAGED);
        }

        private void awaitEngaged(GameTestHelper helper) {
            if (programFailure != null) {
                helper.fail(Component.literal(
                        "the PAUSE transaction failed instead of engaging: " + programFailure
                                + " [status=" + FlyControllerRuntime.programAuthorityStatus() + "]"));
                return;
            }
            if (fly.bodyPrograms().authority() != MotorAuthority.SCRIPTED) {
                pastDeadline(helper, TRANSACTION_DEADLINE_TICKS, "scripted authority");
                return;
            }
            if (fly.bodyPrograms().selectionSource() != SelectionSource.OPERATOR
                    || fly.bodyPrograms().state().id() != BodyProgramId.HOVER) {
                helper.fail(Component.literal("scripted authority engaged the wrong lane: source="
                        + fly.bodyPrograms().selectionSource()
                        + " program=" + fly.bodyPrograms().state().id()));
                return;
            }
            engagedTick = helper.getLevel().getGameTime();
            ownedTicksAtEngage = fly.bodyPrograms().programOwnedTicks();
            engageStatus = FlyControllerRuntime.programAuthorityStatus();
            if (engageMessage == null) {
                helper.fail(Component.literal("authority engaged without its success message"));
                return;
            }
            enter(helper, Stage.WAIT_PROGRAM_TICKS);
            observeScriptedTick(helper);
        }

        private void awaitProgramTicks(GameTestHelper helper) {
            if (!observeScriptedTick(helper)) {
                return;
            }
            if (fly.bodyPrograms().programOwnedTicks() < REQUIRED_PROGRAM_OWNED_TICKS) {
                pastDeadline(
                        helper,
                        REQUIRED_PROGRAM_OWNED_TICKS * 2,
                        REQUIRED_PROGRAM_OWNED_TICKS + " program-owned ticks");
                return;
            }
            if (pausedCommandsObserved == 0) {
                helper.fail(Component.literal(
                        "the program owned ticks but no paused-neutral command was admitted"));
                return;
            }
            if (!(maxAbsVerticalThrustWhileScripted > 0.0)) {
                helper.fail(Component.literal(
                        "HOVER owned " + fly.bodyPrograms().programOwnedTicks()
                                + " ticks without presenting any vertical thrust"));
                return;
            }
            releaseIssuedTick = helper.getLevel().getGameTime();
            boolean queued = FlyControllerRuntime.releaseProgramAuthority(
                    fly,
                    releaseIssuedTick,
                    message -> releaseMessage = message,
                    message -> programFailure = message);
            if (!queued) {
                helper.fail(Component.literal("scripted body-program release was not queued"));
                return;
            }
            enter(helper, Stage.WAIT_RELEASED);
        }

        /**
         * While scripted: authority is held, every newly admitted command is the paused service's
         * neutral action with an empty neural interval at the frozen cursor, and the script is
         * actually presenting motion. Returns false after failing the test.
         */
        private boolean observeScriptedTick(GameTestHelper helper) {
            if (fly.bodyPrograms().authority() != MotorAuthority.SCRIPTED) {
                helper.fail(Component.literal("the program lost authority before release: "
                        + fly.bodyPrograms().transitionLedger()));
                return false;
            }
            Optional<ActionFrame> latest = lastAcceptedAction();
            if (latest.isPresent()) {
                ActionFrame command = latest.orElseThrow();
                // The validator applies a command only on its target tick, so target_tick is the
                // tick it was admitted on. Authority engaged at the start of engagedTick, before
                // that tick's control(); a command that applied earlier was admitted on the
                // neural lane and may legitimately be a stepped one still in flight when the
                // PAUSE was issued. Everything admitted from engagedTick on must be the paused
                // service's neutral action, all at one frozen cursor.
                if (command.targetTick() >= engagedTick
                        && command.commandSequence() != lastPausedCommandSequence) {
                    if (frozenNeuralCursor < 0) {
                        frozenNeuralCursor = command.neuralStepStart();
                    }
                    if (!command.neutral()
                            || command.neuralStepEnd() != command.neuralStepStart()
                            || command.neuralStepStart() != frozenNeuralCursor
                            || frozenNeuralCursor < neuralCursorBeforePause) {
                        helper.fail(Component.literal(
                                "a command admitted under scripted authority was not the paused"
                                        + " service's neutral action at the frozen cursor: seq="
                                        + command.commandSequence()
                                        + " target_tick=" + command.targetTick()
                                        + " neutral=" + command.neutral()
                                        + " neural_steps=" + command.neuralStepStart() + ".."
                                        + command.neuralStepEnd()
                                        + " frozen_cursor=" + frozenNeuralCursor
                                        + " cursor_before_pause=" + neuralCursorBeforePause));
                        return false;
                    }
                    lastPausedCommandSequence = command.commandSequence();
                    pausedCommandsObserved++;
                }
            }
            maxAbsVerticalThrustWhileScripted = Math.max(
                    maxAbsVerticalThrustWhileScripted,
                    Math.abs(fly.presentationAction().verticalThrust()));
            return true;
        }

        private void awaitReleased(GameTestHelper helper) {
            if (programFailure != null) {
                helper.fail(Component.literal(
                        "the RESUME transaction failed instead of releasing: " + programFailure
                                + " [status=" + FlyControllerRuntime.programAuthorityStatus() + "]"));
                return;
            }
            if (fly.bodyPrograms().authority() != MotorAuthority.NEURAL) {
                // Still the scripted window: the paused service keeps answering neutral until the
                // RESUME lands, and every one of those commands is still held to the frozen cursor.
                if (observeScriptedTick(helper)) {
                    pastDeadline(helper, TRANSACTION_DEADLINE_TICKS, "release of scripted authority");
                }
                return;
            }
            List<String> ledger = fly.bodyPrograms().transitionLedger();
            String last = ledger.isEmpty() ? "" : ledger.getLast();
            if (!last.contains("reason=released")) {
                // A resumed connectome's first non-neutral command drops the program through the
                // preemption interlock. That is the interlock working, not the transaction: it is
                // exactly what a missed RESUME acknowledgement looked like before the fix.
                helper.fail(Component.literal(
                        "authority returned to NEURAL by something other than the accepted RESUME: "
                                + ledger));
                return;
            }
            releasedTick = helper.getLevel().getGameTime();
            programOwnedTicksAtRelease = fly.bodyPrograms().programOwnedTicks();
            releaseStatus = FlyControllerRuntime.programAuthorityStatus();
            releaseLedgerLine = last;
            if (releaseMessage == null) {
                helper.fail(Component.literal("authority released without its success message"));
                return;
            }
            enter(helper, Stage.WAIT_DRIVING_AGAIN);
            // The RESUME acknowledgement and the first stepped action ride together, so the neural
            // lane usually applies that action on the release tick itself; look now, not next tick.
            awaitDrivingAgain(helper);
        }

        /**
         * The connectome resumes from the cursor it was frozen at. The first stepped command
         * admitted on or after the release tick is the resumed one; when it is the immediate
         * successor of the last paused command it must start exactly at the frozen cursor, and
         * when the service retired commands in between (a sequence gap, counted in the receipt)
         * the cursor may only have moved forward by those skipped steps.
         */
        private void awaitDrivingAgain(GameTestHelper helper) {
            if (stage != Stage.WAIT_DRIVING_AGAIN) {
                return;
            }
            Optional<ActionFrame> latest = lastAcceptedAction();
            if (latest.isEmpty()
                    || latest.orElseThrow().targetTick() < releasedTick
                    || !stepped(latest.orElseThrow())) {
                pastDeadline(helper, DRIVING_DEADLINE_TICKS, "a stepped command after release");
                return;
            }
            resumedCommand = latest.orElseThrow();
            commandsSkippedSincePause =
                    resumedCommand.commandSequence() - lastPausedCommandSequence - 1;
            boolean resumedFromFrozenCursor = commandsSkippedSincePause == 0
                    ? resumedCommand.neuralStepStart() == frozenNeuralCursor
                    : commandsSkippedSincePause > 0
                            && resumedCommand.neuralStepStart() >= frozenNeuralCursor;
            if (!resumedFromFrozenCursor) {
                helper.fail(Component.literal("the connectome did not resume from the frozen cursor: "
                        + "seq=" + resumedCommand.commandSequence()
                        + " neural_steps=" + resumedCommand.neuralStepStart() + ".."
                        + resumedCommand.neuralStepEnd()
                        + " frozen_cursor=" + frozenNeuralCursor
                        + " last_paused_seq=" + lastPausedCommandSequence));
                return;
            }
            resetTarget = initialPosition.add(1.0, 1.0, 1.0);
            resetIssuedTick = helper.getLevel().getGameTime();
            boolean queued = FlyControllerRuntime.requestReset(
                    fly,
                    resetIssuedTick,
                    resetTarget.x,
                    resetTarget.y,
                    resetTarget.z,
                    90.0F,
                    message -> resetMessage = message,
                    message -> resetFailure = message);
            if (!queued) {
                helper.fail(Component.literal("neural RESET was not queued"));
                return;
            }
            enter(helper, Stage.WAIT_RESET);
        }

        private void awaitReset(GameTestHelper helper) {
            if (resetFailure != null) {
                helper.fail(Component.literal("the RESET transaction failed: " + resetFailure
                        + " [status=" + FlyControllerRuntime.resetStatus() + "]"));
                return;
            }
            if (resetMessage == null) {
                pastDeadline(helper, RESET_DEADLINE_TICKS, "the accepted RESET");
                return;
            }
            if (fly.position().distanceTo(resetTarget) > POSITION_EPSILON) {
                helper.fail(Component.literal("accepted RESET reported applied but the fly is at "
                        + fly.position() + ", not " + resetTarget));
                return;
            }
            resetAppliedTick = helper.getLevel().getGameTime();
            resetStatus = FlyControllerRuntime.resetStatus();
            if (!tickPacer.hasVerifiedCadence()) {
                helper.fail(Component.literal(
                        "external GameTest pacer did not maintain its declared 20 Hz cadence"));
                return;
            }
            String configuredPath = System.getProperty("neurocraft.fly.externalBridgeReceipt");
            if (configuredPath == null || configuredPath.isBlank()) {
                helper.fail(Component.literal("external bridge receipt path is missing"));
                return;
            }
            Path output = Path.of(configuredPath).toAbsolutePath().normalize();
            String json = receiptJson(helper);
            receiptWrite = CompletableFuture.runAsync(() -> writeReceipt(output, json));
            enter(helper, Stage.WAIT_RECEIPT_WRITE);
        }

        private void awaitReceiptWrite(GameTestHelper helper) {
            if (!receiptWrite.isDone()) {
                return;
            }
            try {
                receiptWrite.join();
            } catch (CompletionException exception) {
                helper.fail(Component.literal(
                        "could not write external bridge receipt: "
                                + exception.getCause().getClass().getSimpleName()));
                return;
            }
            helper.succeed();
        }

        private static boolean stepped(ActionFrame command) {
            return command.neuralStepEnd() > command.neuralStepStart();
        }

        /**
         * The fixture profile drives this lane with raw {@code action} frames, so every admitted
         * command is an {@link ActionFrame}; a {@code program_action} frame (ADR 0021) would mean
         * the wrong profile is on the other end and is simply not seen here.
         */
        private Optional<ActionFrame> lastAcceptedAction() {
            return fly.lastAcceptedCommand()
                    .filter(ActionFrame.class::isInstance)
                    .map(ActionFrame.class::cast);
        }

        private String receiptJson(GameTestHelper helper) {
            return String.format(
                    Locale.ROOT,
                    "{\n"
                            + "  \"schema\": \"%s\",\n"
                            + "  \"protocol\": \"%s\",\n"
                            + "  \"entity_id\": \"%s\",\n"
                            + "  \"bridge_state\": \"%s\",\n"
                            + "  \"pacing_mode\": \"%s\",\n"
                            + "  \"pacing_period_ns\": %d,\n"
                            + "  \"driving_command_seq\": %d,\n"
                            + "  \"neural_cursor_before_pause\": %d,\n"
                            + "  \"pause\": {\"issued_tick\": %d, \"engaged_tick\": %d, "
                            + "\"program_owned_ticks_at_engage\": %d, \"status\": %s, "
                            + "\"message\": %s},\n"
                            + "  \"scripted_window\": {\"program\": \"HOVER\", "
                            + "\"source\": \"OPERATOR\", \"program_owned_ticks\": %d, "
                            + "\"paused_commands_observed\": %d, "
                            + "\"last_paused_command_seq\": %d, "
                            + "\"frozen_neural_cursor\": %d, "
                            + "\"max_abs_vertical_thrust\": %.9f},\n"
                            + "  \"release\": {\"issued_tick\": %d, \"released_tick\": %d, "
                            + "\"status\": %s, \"message\": %s, \"ledger_line\": %s},\n"
                            + "  \"resumed\": {\"command_seq\": %d, \"neural_step_start\": %d, "
                            + "\"neural_step_end\": %d, \"commands_skipped_since_pause\": %d},\n"
                            + "  \"reset\": {\"issued_tick\": %d, \"applied_tick\": %d, "
                            + "\"status\": %s, \"message\": %s, "
                            + "\"target_blocks\": [%.9f, %.9f, %.9f], "
                            + "\"position_blocks\": [%.9f, %.9f, %.9f]},\n"
                            + "  \"claim_boundary\": \"scripted presentation over a paused service;"
                            + " transport and transaction evidence only, not connectome behaviour\",\n"
                            + "  \"completed_game_tick\": %d\n"
                            + "}\n",
                    SCHEMA,
                    ControlProtocol.PROTOCOL_VERSION,
                    fly.getStringUUID(),
                    FlyControllerRuntime.state(),
                    ExternalGameTestTickPacer.MODE,
                    ExternalGameTestTickPacer.PERIOD_NANOS,
                    drivingCommandSequence,
                    neuralCursorBeforePause,
                    pauseIssuedTick,
                    engagedTick,
                    ownedTicksAtEngage,
                    quote(engageStatus),
                    quote(engageMessage),
                    programOwnedTicksAtRelease,
                    pausedCommandsObserved,
                    lastPausedCommandSequence,
                    frozenNeuralCursor,
                    maxAbsVerticalThrustWhileScripted,
                    releaseIssuedTick,
                    releasedTick,
                    quote(releaseStatus),
                    quote(releaseMessage),
                    quote(releaseLedgerLine),
                    resumedCommand.commandSequence(),
                    resumedCommand.neuralStepStart(),
                    resumedCommand.neuralStepEnd(),
                    commandsSkippedSincePause,
                    resetIssuedTick,
                    resetAppliedTick,
                    quote(resetStatus),
                    quote(resetMessage),
                    resetTarget.x,
                    resetTarget.y,
                    resetTarget.z,
                    fly.position().x,
                    fly.position().y,
                    fly.position().z,
                    helper.getLevel().getGameTime());
        }

        private static String quote(String value) {
            StringBuilder quoted = new StringBuilder("\"");
            for (int index = 0; index < value.length(); index++) {
                char character = value.charAt(index);
                switch (character) {
                    case '"' -> quoted.append("\\\"");
                    case '\\' -> quoted.append("\\\\");
                    case '\n' -> quoted.append("\\n");
                    default -> {
                        if (character < 0x20 || character > 0x7e) {
                            quoted.append(String.format(Locale.ROOT, "\\u%04x", (int) character));
                        } else {
                            quoted.append(character);
                        }
                    }
                }
            }
            return quoted.append('"').toString();
        }

        private static void writeReceipt(Path output, String json) {
            try {
                Files.createDirectories(output.getParent());
                Files.writeString(
                        output,
                        json,
                        StandardCharsets.UTF_8,
                        StandardOpenOption.CREATE_NEW,
                        StandardOpenOption.WRITE);
            } catch (IOException exception) {
                throw new CompletionException(exception);
            }
        }
    }

    private enum Stage {
        WAIT_ACTIVE,
        WAIT_DRIVING,
        WAIT_ENGAGED,
        WAIT_PROGRAM_TICKS,
        WAIT_RELEASED,
        WAIT_DRIVING_AGAIN,
        WAIT_RESET,
        WAIT_RECEIPT_WRITE
    }
}
