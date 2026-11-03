package dev.neurocraft.fly.gametest;

import dev.neurocraft.fly.FlyControllerRuntime;
import dev.neurocraft.fly.NeuroCraftFly;
import dev.neurocraft.fly.control.ActionFrame;
import dev.neurocraft.fly.control.CommandFrame;
import dev.neurocraft.fly.control.ControlAckFrame;
import dev.neurocraft.fly.control.ControlOperation;
import dev.neurocraft.fly.control.ControlProtocol;
import dev.neurocraft.fly.control.ControlState;
import dev.neurocraft.fly.control.ControllerFault;
import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import dev.neurocraft.fly.control.PrimitiveAction;
import dev.neurocraft.fly.entity.FlyEntity;
import dev.neurocraft.fly.entity.FlySensors;
import dev.neurocraft.fly.presentation.FlyTrailPresentation;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.regex.Pattern;
import net.fabricmc.fabric.api.gametest.v1.GameTest;
import net.minecraft.core.BlockPos;
import net.minecraft.gametest.framework.GameTestHelper;
import net.minecraft.network.chat.Component;
import net.minecraft.world.entity.EntitySpawnReason;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.phys.Vec3;

/** Opt-in causal bridge test; this class is absent from the default GameTest entrypoint list. */
public final class ExternalBridgeGameTest {
    private static final int ACTIVE_DEADLINE_TICKS = 200;
    private static final int POST_STOP_TICKS = 20;
    private static final int POST_STOP_OBSERVATIONS = 5;
    private static final double MOTION_EPSILON_SQUARED = 1.0e-10;
    private static final String RECORDING_ID = "external-bridge-gametest";
    private static final Pattern IDENTIFIER =
            Pattern.compile("[A-Za-z0-9][A-Za-z0-9._:-]{0,127}");
    private static final Pattern HASH = Pattern.compile("[0-9a-f]{64}");

    @GameTest(maxTicks = 600)
    public void liveServiceCausallyDrivesConfiguredFlyAndReceivesReceipt(GameTestHelper helper) {
        if (!Boolean.getBoolean("neurocraft.fly.externalBridgeTest")) {
            helper.fail(Component.literal(
                    "external bridge test class loaded without explicit opt-in"));
            return;
        }
        if (FlyTrailPresentation.enabled()) {
            helper.fail(Component.literal(
                    "cosmetic trail must stay disabled during the evidence bridge run"));
            return;
        }
        ExternalConfig externalConfig = externalConfig(helper);
        if (externalConfig == null) {
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
        Vec3 initialPosition = helper.absoluteVec(new Vec3(1.5, 3.0, 1.5));
        Probe probe = new Probe(
                configuredUuid.orElseThrow(),
                initialPosition,
                helper.getLevel().getGameTime(),
                externalConfig);
        helper.onEachTick(() -> probe.tick(helper));
    }

    private static ExternalConfig externalConfig(GameTestHelper helper) {
        String arenaHash = System.getProperty("neurocraft.fly.arenaMaterialHash");
        String stimulationId = System.getProperty("neurocraft.fly.stimulationId");
        String stimulationHash = System.getProperty("neurocraft.fly.stimulationHash");
        if (arenaHash == null || !HASH.matcher(arenaHash).matches()) {
            helper.fail(Component.literal(
                    "neurocraft.fly.arenaMaterialHash must be a lowercase SHA-256 digest"));
            return null;
        }
        if (stimulationId == null || !IDENTIFIER.matcher(stimulationId).matches()) {
            helper.fail(Component.literal(
                    "neurocraft.fly.stimulationId must be a protocol identifier"));
            return null;
        }
        if (stimulationHash == null || !HASH.matcher(stimulationHash).matches()) {
            helper.fail(Component.literal(
                    "neurocraft.fly.stimulationHash must be a lowercase SHA-256 digest"));
            return null;
        }
        return new ExternalConfig(arenaHash, stimulationId, stimulationHash);
    }

    private static final class Probe {
        private final UUID configuredUuid;
        private final Vec3 initialPosition;
        private final long initialGameTick;
        private final ExternalConfig config;
        private final ExternalGameTestTickPacer tickPacer = ExternalGameTestTickPacer.system();
        private final List<ActionFrame> recordStopBoundaryCandidates = new ArrayList<>();
        private Stage stage = Stage.WAIT_ACTIVE;
        private PhysicalSensorSnapshot openSensors;
        private PhysicalSensorSnapshot blockedSensors;
        private PhysicalSensorSnapshot obstaclePlacementSensors;
        private BlockPos obstacle;
        private long openCommandSequence = -1;
        private long recordStartActionSequence = -1;
        private long blockedCommandSequence = -1;
        private long blockedObservationTick = -1;
        private long obstaclePlacementTick = -1;
        private long blockedSensorSampleTick = -1;
        private long recordStopActionSequence = -1;
        private long postStopStartTick;
        private long postStopLastCommandSequence = -1;
        private ControlEffectFence stimulationApplyFence;
        private ControlEffectFence stimulationClearFence;
        private ObstacleObservationFence obstacleObservationFence;
        private TraceRecordCeiling traceRecordCeiling;
        private ActionFrame nonNeutralCommand;
        private ControlAckFrame recordStartAck;
        private ControlAckFrame stimulationApplyAck;
        private ControlAckFrame stimulationClearAck;
        private ControlAckFrame recordStopAck;
        private CompletableFuture<Void> receiptWrite;
        private FlyEntity fly;

        private Probe(
                UUID configuredUuid,
                Vec3 initialPosition,
                long initialGameTick,
                ExternalConfig config) {
            this.configuredUuid = configuredUuid;
            this.initialPosition = initialPosition;
            this.initialGameTick = initialGameTick;
            this.config = config;
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
                        "external bridge left ACTIVE state: " + bridgeState));
                return;
            }
            switch (stage) {
                case WAIT_RECORD_START_ACK -> awaitRecordStartAck(helper);
                case WAIT_OPEN_ACTION -> awaitOpenAction(helper);
                case WAIT_BLOCKED_ACTION -> awaitBlockedAction(helper);
                case WAIT_STIMULATION_APPLY_ACK -> awaitStimulationApplyAck(helper);
                case WAIT_STIMULATED_ACTION -> awaitStimulatedAction(helper);
                case WAIT_STIMULATED_RECEIPT -> awaitStimulatedReceipt(helper);
                case WAIT_STIMULATION_CLEAR_ACK -> awaitStimulationClearAck(helper);
                case WAIT_CLEARED_ACTION -> awaitClearedAction(helper);
                case WAIT_CLEARED_RECEIPT -> awaitClearedReceipt(helper);
                case WAIT_RECORD_STOP_ACK -> awaitRecordStopAck(helper);
                case WAIT_POST_STOP -> awaitPostStop(helper);
                case WAIT_RECEIPT_WRITE -> awaitReceiptWrite(helper);
                case WAIT_ACTIVE -> throw new IllegalStateException("stage already handled");
            }
        }

        private void awaitActive(GameTestHelper helper, ControllerFault bridgeState) {
            if (bridgeState == ControllerFault.ACTIVE) {
                if (!requestControl(helper, ControlOperation.RECORD_START, RECORDING_ID, null)) {
                    return;
                }
                stage = Stage.WAIT_RECORD_START_ACK;
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

        private void awaitRecordStartAck(GameTestHelper helper) {
            ControlAckFrame acknowledgement = acceptedAck(
                    helper, ControlOperation.RECORD_START, ControlState.RECORDING, 0);
            if (acknowledgement == null) {
                return;
            }
            recordStartAck = acknowledgement;
            if (!spawnConfiguredFly(helper)) {
                return;
            }
            recordStartActionSequence = currentActionSequence();
            openSensors = FlySensors.sample(helper.getLevel(), fly);
            if (Double.compare(openSensors.rayForwardBlocks(), FlySensors.RAY_LENGTH_BLOCKS) != 0) {
                helper.fail(Component.literal(
                        "external open arena did not expose the full forward ray"));
                return;
            }
            stage = Stage.WAIT_OPEN_ACTION;
        }

        private boolean spawnConfiguredFly(GameTestHelper helper) {
            if (fly != null) {
                helper.fail(Component.literal("external bridge fly was spawned more than once"));
                return false;
            }
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

        private void awaitOpenAction(GameTestHelper helper) {
            Optional<ActionFrame> latest = lastAcceptedAction();
            if (latest.isEmpty()) {
                return;
            }
            ActionFrame command = latest.orElseThrow();
            if (command.commandSequence() <= recordStartActionSequence) {
                return;
            }
            openCommandSequence = command.commandSequence();
            rememberNonNeutral(command);

            Vec3 origin = fly.getBoundingBox().getCenter();
            Vec3 forward = Vec3.directionFromRotation(fly.getXRot(), fly.getYRot()).normalize();
            obstacle = BlockPos.containing(origin.add(forward.scale(2.0)));
            helper.getLevel().setBlockAndUpdate(obstacle, Blocks.STONE.defaultBlockState());
            obstaclePlacementTick = helper.getLevel().getGameTime();
            obstaclePlacementSensors = FlySensors.sample(helper.getLevel(), fly);
            if (obstaclePlacementSensors.gameTick() != obstaclePlacementTick) {
                helper.fail(Component.literal(
                        "blocked sensor sample was not captured at obstacle placement"));
                return;
            }
            if (!(obstaclePlacementSensors.rayForwardBlocks() > 0.0
                    && obstaclePlacementSensors.rayForwardBlocks()
                            < openSensors.rayForwardBlocks())) {
                helper.fail(Component.literal(
                        "declared obstacle did not shorten the external forward ray"));
                return;
            }
            obstacleObservationFence =
                    new ObstacleObservationFence(openCommandSequence, obstaclePlacementTick);
            stage = Stage.WAIT_BLOCKED_ACTION;
        }

        private void awaitBlockedAction(GameTestHelper helper) {
            Optional<ActionFrame> latest = lastAcceptedAction();
            if (latest.isPresent()) {
                ActionFrame command = latest.orElseThrow();
                rememberNonNeutral(command);
                if (command.commandSequence() > openCommandSequence
                        && command.observationTick() > obstaclePlacementTick) {
                    Optional<PhysicalSensorSnapshot> exactSample =
                            fly.sensorSnapshotAt(command.observationTick());
                    if (exactSample.isEmpty()) {
                        helper.fail(Component.literal(
                                "blocked action did not retain its authoritative sensor sample"));
                        return;
                    }
                    PhysicalSensorSnapshot candidateSensors = exactSample.orElseThrow();
                    if (!(candidateSensors.rayForwardBlocks() > 0.0
                            && candidateSensors.rayForwardBlocks()
                                    < openSensors.rayForwardBlocks())) {
                        helper.fail(Component.literal(
                                "blocked action observation did not contain the shortened ray"));
                        return;
                    }
                    if (!obstacleObservationFence.tryLatch(command)) {
                        return;
                    }
                    blockedCommandSequence = obstacleObservationFence.blockedCommandSequence();
                    blockedObservationTick = obstacleObservationFence.blockedObservationTick();
                    blockedSensors = candidateSensors;
                    blockedSensorSampleTick = candidateSensors.gameTick();
                    if (blockedSensorSampleTick != blockedObservationTick) {
                        helper.fail(Component.literal(
                                "blocked receipt sample did not bind its action observation tick"));
                        return;
                    }
                }
            }
            if (blockedCommandSequence < 0
                    || nonNeutralCommand == null
                    || !worldTransitionObserved()) {
                return;
            }
            long receiptSequence = FlyControllerRuntime
                    .lastSentActuationReceiptCommandSequence()
                    .orElse(-1);
            if (receiptSequence < blockedCommandSequence) {
                return;
            }
            if (!requestControl(
                    helper,
                    ControlOperation.STIMULATION_APPLY,
                    config.stimulationId(),
                    config.stimulationHash())) {
                return;
            }
            stage = Stage.WAIT_STIMULATION_APPLY_ACK;
        }

        private void awaitStimulationApplyAck(GameTestHelper helper) {
            ControlAckFrame acknowledgement = acceptedAck(
                    helper, ControlOperation.STIMULATION_APPLY, ControlState.RECORDING, 1);
            if (acknowledgement == null) {
                return;
            }
            stimulationApplyAck = acknowledgement;
            stimulationApplyFence = new ControlEffectFence(
                    currentActionSequence(), acknowledgement.effectiveTick());
            stage = Stage.WAIT_STIMULATED_ACTION;
        }

        private void awaitStimulatedAction(GameTestHelper helper) {
            Optional<ActionFrame> latest = lastAcceptedAction();
            if (latest.isEmpty()) {
                return;
            }
            ActionFrame command = latest.orElseThrow();
            if (!stimulationApplyFence.tryLatch(command)) {
                return;
            }
            rememberNonNeutral(command);
            stage = Stage.WAIT_STIMULATED_RECEIPT;
        }

        private void awaitStimulatedReceipt(GameTestHelper helper) {
            long receiptSequence = FlyControllerRuntime
                    .lastSentActuationReceiptCommandSequence()
                    .orElse(-1);
            if (!stimulationApplyFence.receiptSentThrough(receiptSequence)) {
                return;
            }
            if (!requestControl(
                    helper,
                    ControlOperation.STIMULATION_CLEAR,
                    config.stimulationId(),
                    config.stimulationHash())) {
                return;
            }
            stage = Stage.WAIT_STIMULATION_CLEAR_ACK;
        }

        private void awaitStimulationClearAck(GameTestHelper helper) {
            ControlAckFrame acknowledgement = acceptedAck(
                    helper, ControlOperation.STIMULATION_CLEAR, ControlState.RECORDING, 2);
            if (acknowledgement == null) {
                return;
            }
            stimulationClearAck = acknowledgement;
            stimulationClearFence = new ControlEffectFence(
                    currentActionSequence(), acknowledgement.effectiveTick());
            stage = Stage.WAIT_CLEARED_ACTION;
        }

        private void awaitClearedAction(GameTestHelper helper) {
            Optional<ActionFrame> latest = lastAcceptedAction();
            if (latest.isEmpty()) {
                return;
            }
            ActionFrame command = latest.orElseThrow();
            if (!stimulationClearFence.tryLatch(command)) {
                return;
            }
            rememberNonNeutral(command);
            recordStopActionSequence = stimulationClearFence.latchedCommandSequence();
            recordStopBoundaryCandidates.add(command);
            stage = Stage.WAIT_CLEARED_RECEIPT;
        }

        private void awaitClearedReceipt(GameTestHelper helper) {
            if (!captureRecordStopBoundaryCandidate(helper)) {
                return;
            }
            long receiptSequence = FlyControllerRuntime
                    .lastSentActuationReceiptCommandSequence()
                    .orElse(-1);
            if (!stimulationClearFence.receiptSentThrough(receiptSequence)) {
                return;
            }
            if (!requestControl(helper, ControlOperation.RECORD_STOP, RECORDING_ID, null)) {
                return;
            }
            stage = Stage.WAIT_RECORD_STOP_ACK;
        }

        private void awaitRecordStopAck(GameTestHelper helper) {
            if (!captureRecordStopBoundaryCandidate(helper)) {
                return;
            }
            ControlAckFrame acknowledgement =
                    acceptedAck(helper, ControlOperation.RECORD_STOP, ControlState.RUNNING, 3);
            if (acknowledgement == null) {
                return;
            }
            if (acknowledgement.effectiveTick() != acknowledgement.issuedTick()) {
                helper.fail(Component.literal(
                        "record stop acknowledgement changed its atomic boundary tick"));
                return;
            }
            recordStopAck = acknowledgement;
            traceRecordCeiling = new TraceRecordCeiling(acknowledgement.effectiveTick());
            try {
                for (ActionFrame candidate : recordStopBoundaryCandidates) {
                    traceRecordCeiling.observe(candidate);
                }
            } catch (IllegalStateException exception) {
                helper.fail(Component.literal(exception.getMessage()));
                return;
            }
            postStopStartTick = helper.getLevel().getGameTime();
            postStopLastCommandSequence = recordStopActionSequence;
            stage = Stage.WAIT_POST_STOP;
        }

        private void awaitPostStop(GameTestHelper helper) {
            Optional<ActionFrame> latest = lastAcceptedAction();
            if (latest.isPresent()) {
                try {
                    traceRecordCeiling.observe(latest.orElseThrow());
                } catch (IllegalStateException exception) {
                    helper.fail(Component.literal(exception.getMessage()));
                    return;
                }
            }
            postStopLastCommandSequence = Math.max(
                    postStopLastCommandSequence, currentActionSequence());
            long postStopObservations = postStopLastCommandSequence - recordStopActionSequence;
            if (!traceRecordCeiling.isSealed()
                    || helper.getLevel().getGameTime() - postStopStartTick < POST_STOP_TICKS
                    || postStopObservations < POST_STOP_OBSERVATIONS) {
                return;
            }
            long receiptSequence = FlyControllerRuntime
                    .lastSentActuationReceiptCommandSequence()
                    .orElse(-1);
            if (FlyControllerRuntime.sentActuationReceiptCount() < 1
                    || receiptSequence < postStopLastCommandSequence) {
                return;
            }
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
            String json = receiptJson(helper, receiptSequence, postStopObservations);
            receiptWrite = CompletableFuture.runAsync(() -> writeReceipt(output, json));
            stage = Stage.WAIT_RECEIPT_WRITE;
        }

        private boolean captureRecordStopBoundaryCandidate(GameTestHelper helper) {
            Optional<ActionFrame> latest = lastAcceptedAction();
            if (latest.isEmpty()) {
                return true;
            }
            ActionFrame command = latest.orElseThrow();
            ActionFrame previous = recordStopBoundaryCandidates.getLast();
            if (command.commandSequence() == previous.commandSequence()) {
                return true;
            }
            if (command.commandSequence() < previous.commandSequence()) {
                helper.fail(Component.literal(
                        "record stop candidate actions are not sequence-monotonic"));
                return false;
            }
            recordStopBoundaryCandidates.add(command);
            return true;
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

        private boolean requestControl(
                GameTestHelper helper,
                ControlOperation operation,
                String contentId,
                String contentHash) {
            boolean queued = FlyControllerRuntime.requestControl(
                    helper.getLevel().getGameTime(), operation, contentId, contentHash);
            if (!queued) {
                helper.fail(Component.literal("could not queue external control " + operation));
            }
            return queued;
        }

        private static ControlAckFrame acceptedAck(
                GameTestHelper helper,
                ControlOperation expectedOperation,
                ControlState expectedState,
                long expectedSequence) {
            Optional<ControlAckFrame> available =
                    FlyControllerRuntime.pollControlAcknowledgement();
            if (available.isEmpty()) {
                return null;
            }
            ControlAckFrame acknowledgement = available.orElseThrow();
            if (acknowledgement.operation() != expectedOperation
                    || acknowledgement.controlSequence() != expectedSequence
                    || !acknowledgement.accepted()
                    || acknowledgement.state() != expectedState) {
                helper.fail(Component.literal("external control acknowledgement mismatch for "
                        + expectedOperation + ": " + acknowledgement));
                return null;
            }
            return acknowledgement;
        }

        private long currentActionSequence() {
            return fly.lastAcceptedCommand()
                    .map(CommandFrame::commandSequence)
                    .orElse(-1L);
        }

        /**
         * This lane is driven by raw {@code action} frames, so every admitted command is an
         * {@link ActionFrame}. The narrowing is written once and states that expectation, rather
         * than assumed at seven call sites.
         */
        private Optional<ActionFrame> lastAcceptedAction() {
            return fly.lastAcceptedCommand()
                    .filter(ActionFrame.class::isInstance)
                    .map(ActionFrame.class::cast);
        }

        private void rememberNonNeutral(ActionFrame command) {
            if (!command.rawAction().clamped().isNeutral()) {
                nonNeutralCommand = command;
            }
        }

        private boolean worldTransitionObserved() {
            return fly.position().distanceToSqr(initialPosition) > MOTION_EPSILON_SQUARED
                    || Math.abs(fly.getYRot()) > 1.0e-5F
                    || Math.abs(fly.getXRot()) > 1.0e-5F;
        }

        private String receiptJson(
                GameTestHelper helper, long receiptSequence, long postStopObservations) {
            PrimitiveAction motors = nonNeutralCommand.rawAction().clamped();
            Vec3 finalPosition = fly.position();
            long traceCeilingCommandSequence = traceRecordCeiling.commandSequence();
            long minimumRecordedActions =
                    traceCeilingCommandSequence - openCommandSequence + 1;
            return String.format(
                    Locale.ROOT,
                    "{\n"
                            + "  \"schema\": \"neurocraft.external_bridge_gametest.v1\",\n"
                            + "  \"protocol\": \"%s\",\n"
                            + "  \"entity_id\": \"%s\",\n"
                            + "  \"arena_material_hash\": \"%s\",\n"
                            + "  \"bridge_state\": \"%s\",\n"
                            + "  \"pacing_mode\": \"%s\",\n"
                            + "  \"pacing_period_ns\": %d,\n"
                            + "  \"open_forward_ray_blocks\": %.9f,\n"
                            + "  \"blocked_forward_ray_blocks\": %.9f,\n"
                            + "  \"obstacle_block\": [%d, %d, %d],\n"
                            + "  \"open_command_seq\": %d,\n"
                            + "  \"blocked_command_seq\": %d,\n"
                            + "  \"blocked_observation_tick\": %d,\n"
                            + "  \"non_neutral_command_seq\": %d,\n"
                            + "  \"neural_step_start\": %d,\n"
                            + "  \"neural_step_end\": %d,\n"
                            + "  \"state_digest\": \"%s\",\n"
                            + "  \"applied_motors\": {\"forward_thrust\": %.9f, "
                            + "\"lateral_thrust\": %.9f, \"vertical_thrust\": %.9f, "
                            + "\"yaw_torque\": %.9f, \"pitch_torque\": %.9f},\n"
                            + "  \"record_start\": {\"control_seq\": %d, \"issued_tick\": %d, "
                            + "\"effective_tick\": %d, \"state\": \"%s\"},\n"
                            + "  \"stimulation_apply\": {\"content_id\": \"%s\", "
                            + "\"content_hash\": \"%s\", \"control_seq\": %d, "
                            + "\"issued_tick\": %d, \"effective_tick\": %d},\n"
                            + "  \"stimulation_clear\": {\"content_id\": \"%s\", "
                            + "\"content_hash\": \"%s\", \"control_seq\": %d, "
                            + "\"issued_tick\": %d, \"effective_tick\": %d},\n"
                            + "  \"record_stop\": {\"control_seq\": %d, \"issued_tick\": %d, "
                            + "\"effective_tick\": %d, \"state\": \"%s\"},\n"
                            + "  \"trace_record_expectation\": {\"content_id\": \"%s\", "
                            + "\"first_command_seq\": %d, \"last_command_seq\": %d, "
                            + "\"minimum_action_records\": %d, "
                            + "\"contains_stimulation_round_trip\": true},\n"
                            + "  \"post_stop_observations\": %d,\n"
                            + "  \"initial_position_blocks\": [%.9f, %.9f, %.9f],\n"
                            + "  \"final_position_blocks\": [%.9f, %.9f, %.9f],\n"
                            + "  \"world_transition_observed\": true,\n"
                            + "  \"actuation_receipts_sent\": %d,\n"
                            + "  \"last_receipt_command_seq\": %d,\n"
                            + "  \"completed_game_tick\": %d\n"
                            + "}\n",
                    ControlProtocol.PROTOCOL_VERSION,
                    fly.getStringUUID(),
                    config.arenaMaterialHash(),
                    FlyControllerRuntime.state(),
                    ExternalGameTestTickPacer.MODE,
                    ExternalGameTestTickPacer.PERIOD_NANOS,
                    openSensors.rayForwardBlocks(),
                    blockedSensors.rayForwardBlocks(),
                    obstacle.getX(),
                    obstacle.getY(),
                    obstacle.getZ(),
                    openCommandSequence,
                    blockedCommandSequence,
                    blockedObservationTick,
                    nonNeutralCommand.commandSequence(),
                    nonNeutralCommand.neuralStepStart(),
                    nonNeutralCommand.neuralStepEnd(),
                    nonNeutralCommand.stateDigest(),
                    motors.forwardThrust(),
                    motors.lateralThrust(),
                    motors.verticalThrust(),
                    motors.yawTorque(),
                    motors.pitchTorque(),
                    recordStartAck.controlSequence(),
                    recordStartAck.issuedTick(),
                    recordStartAck.effectiveTick(),
                    recordStartAck.state(),
                    config.stimulationId(),
                    config.stimulationHash(),
                    stimulationApplyAck.controlSequence(),
                    stimulationApplyAck.issuedTick(),
                    stimulationApplyAck.effectiveTick(),
                    config.stimulationId(),
                    config.stimulationHash(),
                    stimulationClearAck.controlSequence(),
                    stimulationClearAck.issuedTick(),
                    stimulationClearAck.effectiveTick(),
                    recordStopAck.controlSequence(),
                    recordStopAck.issuedTick(),
                    recordStopAck.effectiveTick(),
                    recordStopAck.state(),
                    RECORDING_ID,
                    openCommandSequence,
                    traceCeilingCommandSequence,
                    minimumRecordedActions,
                    postStopObservations,
                    initialPosition.x,
                    initialPosition.y,
                    initialPosition.z,
                    finalPosition.x,
                    finalPosition.y,
                    finalPosition.z,
                    FlyControllerRuntime.sentActuationReceiptCount(),
                    receiptSequence,
                    helper.getLevel().getGameTime());
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

    private record ExternalConfig(
            String arenaMaterialHash, String stimulationId, String stimulationHash) {}

    private enum Stage {
        WAIT_ACTIVE,
        WAIT_RECORD_START_ACK,
        WAIT_OPEN_ACTION,
        WAIT_BLOCKED_ACTION,
        WAIT_STIMULATION_APPLY_ACK,
        WAIT_STIMULATED_ACTION,
        WAIT_STIMULATED_RECEIPT,
        WAIT_STIMULATION_CLEAR_ACK,
        WAIT_CLEARED_ACTION,
        WAIT_CLEARED_RECEIPT,
        WAIT_RECORD_STOP_ACK,
        WAIT_POST_STOP,
        WAIT_RECEIPT_WRITE
    }
}
