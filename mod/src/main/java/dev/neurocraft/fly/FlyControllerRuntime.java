package dev.neurocraft.fly;

import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.MotorAuthority;
import dev.neurocraft.fly.body.ProgramExecution;
import dev.neurocraft.fly.body.SelectionSource;
import dev.neurocraft.fly.control.ControllerIdentity;
import dev.neurocraft.fly.control.ActionFrame;
import dev.neurocraft.fly.control.BridgeConfig;
import dev.neurocraft.fly.control.BrainDisplayReader;
import dev.neurocraft.fly.control.EventDisplayReader;
import dev.neurocraft.fly.control.EventDisplayState;
import dev.neurocraft.fly.control.EventDisplayPresentation;
import dev.neurocraft.fly.control.BrainDisplayState;
import dev.neurocraft.fly.control.BrainDisplayPresentation;
import dev.neurocraft.fly.control.CommandFrame;
import dev.neurocraft.fly.control.CommandValidator;
import dev.neurocraft.fly.control.ComparisonLanePresentation;
import dev.neurocraft.fly.control.ComparisonLaneSession;
import dev.neurocraft.fly.control.ControllerBridge;
import dev.neurocraft.fly.control.ControllerFault;
import dev.neurocraft.fly.control.ControlAckFrame;
import dev.neurocraft.fly.control.ControlOperation;
import dev.neurocraft.fly.control.ControlProtocol;
import dev.neurocraft.fly.control.NeuralBridge;
import dev.neurocraft.fly.control.NeutralBridge;
import dev.neurocraft.fly.control.LaneCatalog;
import dev.neurocraft.fly.control.ProtocolException;
import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import dev.neurocraft.fly.control.ProgramActionFrame;
import dev.neurocraft.fly.control.ValidationResult;
import dev.neurocraft.fly.entity.FlyEntity;
import dev.neurocraft.fly.entity.ActuationReceipt;
import dev.neurocraft.fly.trial.MatchedTrialLedger;
import dev.neurocraft.fly.trial.MatchedTrialRunner;
import java.util.ArrayList;
import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.UUID;
import java.util.function.Consumer;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Owns one process bridge and per-entity command validators for a server lifecycle. */
public final class FlyControllerRuntime {
    private static final Logger LOGGER = LoggerFactory.getLogger("neurocraft-fly-controller");
    private static final Map<UUID, CommandValidator> VALIDATORS = new HashMap<>();
    private static final StagedResetSlot RESET_SLOT = new StagedResetSlot();

    private static ControllerBridge bridge =
            new NeutralBridge(ControllerFault.MISSING_CONFIGURATION);
    private static BridgeConfig config;
    private static ComparisonLaneSession comparison;
    private static Path epochDiagnosticsDirectory;
    private static MatchedTrialRunner matchedTrials;
    private static EventDisplayReader eventDisplayReader;
    private static EventDisplayState eventDisplayState;
    public static final String UNAVAILABLE_EVENT_DISPLAY = EventDisplayPresentation.unavailable().encode();
    private static BrainDisplayReader brainDisplayReader;
    private static BrainDisplayState brainDisplayState;
    public static final String UNAVAILABLE_BRAIN_DISPLAY = new BrainDisplayPresentation(
            BrainDisplayState.Status.UNAVAILABLE, 0, Optional.empty()).encode();
    private static Consumer<String> resetSuccess = ignored -> {};
    private static Consumer<String> resetFailure = ignored -> {};
    private static String resetStatus = "none";
    private static PendingOperatorControl pendingOperatorControl;
    private static StagedProgramAuthority pendingProgramAuthority;
    private static OperatorProgramGrant operatorProgramGrant;
    private static LocalProgramChange pendingLocalProgramChange;
    private static Consumer<String> programAuthoritySuccess = ignored -> {};
    private static Consumer<String> programAuthorityFailure = ignored -> {};
    private static String programAuthorityStatus = "none";

    private FlyControllerRuntime() {}

    public static void registerLifecycle() {
        ServerLifecycleEvents.SERVER_STARTED.register(server -> start());
        ServerLifecycleEvents.SERVER_STOPPING.register(server -> stop());
        ServerTickEvents.END_SERVER_TICK.register(server -> processPendingOperatorControl());
    }

    public static ValidationResult control(
            FlyEntity entity, PhysicalSensorSnapshot snapshot) {
        if (config == null) {
            return ValidationResult.rejected(bridge.state());
        }
        if (comparison != null) {
            comparison.presentation(); // Refresh authentication/failure before admitting commands.
        }
        if (!config.identity().entityId().equals(entity.getStringUUID())
                || !snapshot.entityId().equals(entity.getStringUUID())) {
            bridge.failClosed(ControllerFault.IDENTITY_MISMATCH);
            return ValidationResult.rejected(ControllerFault.IDENTITY_MISMATCH);
        }
        Optional<ControllerFault> pendingTerminalFault = bridge.pendingTerminalFault();
        if (pendingTerminalFault.isPresent()) {
            return ValidationResult.rejected(pendingTerminalFault.orElseThrow());
        }
        bridge.offerObservation(snapshot);
        // RESET acknowledgement and action frames arrive on the WebSocket thread, while
        // processPendingReset runs once at the server tick boundary. If the acknowledgement
        // misses that boundary, an action for the same tick may still be queued before this
        // method polls the action window. Keep sensing so the service can materialize the
        // transaction, but never consume or apply an action while its world reset is pending.
        // The next tick either applies the reset first or fails closed on the missed boundary.
        if (RESET_SLOT.blocksActuation(entity.getUUID())) {
            return ValidationResult.rejected(ControllerFault.ACTIVE);
        }
        CommandValidator validator = VALIDATORS.computeIfAbsent(
                entity.getUUID(), ignored -> new CommandValidator(config.identity()));
        List<CommandFrame> candidates =
                new ArrayList<>(dev.neurocraft.fly.control.BoundedActionWindow.CAPACITY);
        for (int index = 0;
                index < dev.neurocraft.fly.control.BoundedActionWindow.CAPACITY;
                index++) {
            var candidate = bridge.pollAction();
            if (candidate.isEmpty()) {
                break;
            }
            candidates.add(candidate.orElseThrow());
        }
        ValidationResult result = validator.validate(
                candidates,
                snapshot.gameTick(),
                bridge.state(),
                bridge::sentObservationTick);
        if (result.state().isPaused() && result.command().isEmpty()) {
            bridge.failClosed(result.state());
        }
        return result;
    }

    public static ControllerFault state() {
        if (comparison != null) {
            comparison.presentation();
        }
        return bridge.state();
    }

    public static ComparisonLanePresentation comparisonFor(FlyEntity entity) {
        return comparison == null || config == null
                || !config.identity().entityId().equals(entity.getStringUUID())
                ? ComparisonLanePresentation.unavailable() : comparison.presentation();
    }

    /** Same world/body, new authenticated neural epoch; called only on the server thread. */
    public static boolean requestComparison(List<FlyEntity> loadedFlies, String lane) {
        if (matchedTrialsActive()) { return false; }
        return changeComparison(loadedFlies, lane, () -> {});
    }

    private static boolean changeComparison(List<FlyEntity> loadedFlies, String lane, Runnable fixtureReset) {
        if (comparison == null || config == null || loadedFlies.size() != 1
                || !config.identity().entityId().equals(loadedFlies.getFirst().getStringUUID())
                || pendingOperatorControl != null || programTransactionPending()
                || RESET_SLOT.pending().isPresent() || comparison.pending()) {
            return false;
        }
        FlyEntity entity = loadedFlies.getFirst();
        try {
            comparison.switchTo(lane, () -> {
                stopBrainDisplay();
                stopEventDisplay();
                VALIDATORS.clear();
                RESET_SLOT.clear();
                resetSuccess = ignored -> {};
                resetFailure = ignored -> {};
                resetStatus = "none";
                pendingOperatorControl = null;
                pendingProgramAuthority = null;
                operatorProgramGrant = null;
                pendingLocalProgramChange = null;
                programAuthoritySuccess = ignored -> {};
                programAuthorityFailure = ignored -> {};
                programAuthorityStatus = "none";
                entity.resetNeuralSessionPreservingBody();
                fixtureReset.run();
            });
        } catch (IllegalArgumentException | IllegalStateException | ArithmeticException exception) {
            return false;
        }
        config = comparison.config();
        bridge = comparison.bridge();
        startBrainDisplay();
        startEventDisplay();
        entity.publishComparison(comparison.presentation());
        LOGGER.info("Comparison attempt lane={} epoch={} status={}", lane,
                config.identity().sessionEpoch(), comparison.presentation().status());
        return true;
    }

    public static boolean matchedTrialsActive() { return matchedTrials != null && matchedTrials.active(); }
    public static String matchedTrialStatus() { return matchedTrials == null ? "not started" : matchedTrials.status(); }

    /** Console arms the fixed plan; only subsequent authoritative ticks schedule stimuli. */
    public static boolean startMatchedTrials(List<FlyEntity> loadedFlies) {
        if (matchedTrialsActive() || comparison == null || config == null || epochDiagnosticsDirectory == null
                || comparison.pending() || state() != ControllerFault.ACTIVE || loadedFlies.size() != 1
                || loadedFlies.getFirst().level().dimension() != net.minecraft.world.level.Level.OVERWORLD
                || !config.identity().entityId().equals(loadedFlies.getFirst().getStringUUID())
                || pendingOperatorControl != null || programTransactionPending() || RESET_SLOT.pending().isPresent()) {
            return false;
        }
        MatchedTrialLedger ledger = null;
        try {
            Path path = epochDiagnosticsDirectory.resolveSibling("matched-trials-"
                    + Math.incrementExact(config.identity().sessionEpoch()) + ".ndjson");
            ledger = new MatchedTrialLedger(path);
            FlyEntity entity = loadedFlies.getFirst();
            matchedTrials = new MatchedTrialRunner(entity, ledger, new MatchedTrialRunner.Handover() {
                @Override public dev.neurocraft.fly.control.ControllerIdentity begin(String lane, Runnable reset) {
                    if (!changeComparison(List.of(entity), lane, reset)) {
                        throw new IllegalStateException("trial handover refused");
                    }
                    return config.identity();
                }
                @Override public ControllerFault state() { return FlyControllerRuntime.state(); }
                @Override public dev.neurocraft.fly.control.ControllerIdentity identity() { return config.identity(); }
            });
            LOGGER.info("Matched trial plan armed; diagnostic ledger={}", path);
            return true;
        } catch (IOException | RuntimeException exception) {
            if (ledger != null) { ledger.close(); }
            LOGGER.warn("Matched trial start refused: {}", exception.getMessage());
            return false;
        }
    }

    public static void stopMatchedTrials() {
        if (matchedTrials != null) { matchedTrials.stop("operator stopped; final lane retained"); }
    }

    public static void beforeMatchedTrialObservation(FlyEntity entity) {
        if (matchedTrialsActive() && matchedTrials.entity() == entity) { matchedTrials.beforeObservation(); }
    }

    public static void afterMatchedTrialPhysics(FlyEntity entity, PhysicalSensorSnapshot snapshot,
            ValidationResult decision, ActuationReceipt receipt, Optional<ProgramExecution> execution) {
        if (matchedTrialsActive() && matchedTrials.entity() == entity) {
            matchedTrials.afterPhysics(snapshot, decision, receipt, execution, bridge.sentProgramReceiptCount());
        }
    }

    /**
     * The driving frame kind this entity's session has latched, or empty before its first command.
     *
     * <p>This is the session's own latch, not an inference from the last admitted frame: a session
     * that latched PROGRAM and then failed closed on a lane-mixing refusal still reports PROGRAM,
     * which is the honest answer to "what was driving this fly".
     */
    public static Optional<CommandValidator.FrameLane> commandLane(FlyEntity entity) {
        CommandValidator validator = VALIDATORS.get(entity.getUUID());
        return validator == null ? Optional.empty() : Optional.of(validator.lane());
    }

    /** Count of canonical program receipt frames this bridge has handed to the WebSocket. */
    public static long sentProgramReceiptCount() {
        return bridge.sentProgramReceiptCount();
    }

    /**
     * Sends the authoritative actuation receipt for a command the world actually applied.
     *
     * <p>{@code programOwnedTick} suppresses the receipt when a scripted body program owned the
     * tick (ADR 0020 section 3). {@code encodeActuationReceipt} answers exactly one question --
     * for command N, what did the world do? -- and derives {@code validation} as
     * {@code APPLIED}/{@code CLAMPED} from the difference between requested and applied motors.
     * On a program-owned tick the applied motors did not come from the requested ones by clamping,
     * so both words would be false and there is no honest receipt to send.
     *
     * <p>The receipt-free tick path is safe <strong>because of ADR 0018</strong>: the service now
     * frees validated-window slots on {@code target_tick} from observation arrivals
     * ({@code packages/protocol/src/neurocraft_protocol/sequence.py:88}, {@code :136}), explicitly
     * for commands that produce no receipt. Before that change this design would have wedged the
     * window in two ticks and killed the session.
     *
     * <p><strong>A program frame is always receipted.</strong> It has no honest
     * {@code actuation_receipt}, but it has an honest {@code program_receipt}, which reports what
     * ran rather than a motor difference. So the readout lane drains the service's validated window
     * by receipt exactly as the raw lane does; only the operator lane's tick stays receipt-free.
     */
    public static void recordActuation(
            PhysicalSensorSnapshot worldBefore,
            ValidationResult decision,
            ActuationReceipt receipt,
            boolean programOwnedTick,
            Optional<ProgramExecution> readoutExecution) {
        if (config == null || decision.command().isEmpty()) {
            return;
        }
        CommandFrame command = decision.command().orElseThrow();
        long appliedTick = worldBefore.gameTick();
        if (command instanceof ProgramActionFrame program) {
            if (decision.accepted()
                    && (appliedTick < program.targetTick()
                            || appliedTick > program.validThroughTick())) {
                return;
            }
            bridge.offerProgramReceipt(program, worldBefore, readoutExecution, decision);
            if (decision.accepted() && readoutExecution.filter(ProgramExecution::applied).isPresent()) {
                recordDisplayedApplication(program);
            }
            return;
        }
        if (programOwnedTick || !(command instanceof ActionFrame action)) {
            return;
        }
        if (decision.accepted()
                && (appliedTick < action.targetTick()
                        || appliedTick > action.validThroughTick())) {
            return;
        }
        bridge.offerActuationReceipt(action, worldBefore, receipt, decision);
        if (decision.accepted() && !action.neutral()) {
            recordDisplayedApplication(action);
        }
    }

    private static void recordDisplayedApplication(CommandFrame command) {
        try {
            if (eventDisplayState != null) { eventDisplayState.recordApplied(command); }
        } catch (RuntimeException exception) { stopEventDisplay(); }
        try {
            if (brainDisplayState != null) {
                brainDisplayState.recordApplied(command);
            }
        } catch (RuntimeException exception) {
            stopBrainDisplay();
        }
    }

    /** Publish only after the entity has completed physics; no disk I/O on this thread. */
    public static String brainDisplayFor(FlyEntity entity, long gameTick) {
        try {
            if (brainDisplayReader == null || brainDisplayState == null || config == null
                    || !config.identity().entityId().equals(entity.getStringUUID())
                    || (comparison != null
                            && comparison.presentation().status() != ComparisonLanePresentation.Status.ACTIVE)) {
                return UNAVAILABLE_BRAIN_DISPLAY;
            }
            return BrainDisplayPresentation.fromView(brainDisplayState.view(
                    brainDisplayReader.snapshot(), System.nanoTime(), gameTick)).encode();
        } catch (RuntimeException exception) {
            stopBrainDisplay();
            return UNAVAILABLE_BRAIN_DISPLAY;
        }
    }

    public static void resetBrainDisplay(FlyEntity entity) {
        if (eventDisplayState != null && config != null
                && config.identity().entityId().equals(entity.getStringUUID())) {
            eventDisplayState.reset();
        }
        if (brainDisplayState != null && config != null
                && config.identity().entityId().equals(entity.getStringUUID())) {
            brainDisplayState.reset();
        }
    }

    /** Optional issued-event evidence, made visible only by the actual applied-program ledger. */
    public static String eventDisplayFor(FlyEntity entity, long gameTick) {
        if (config == null || !config.identity().entityId().equals(entity.getStringUUID())) { return ""; }
        boolean candidate = ControlProtocol.isEventObservationSchema(config.identity().observationSchemaHash());
        if (!candidate) { return ""; }
        try {
            if (eventDisplayReader == null || eventDisplayState == null
                    || (comparison != null && comparison.presentation().status()
                            != ComparisonLanePresentation.Status.ACTIVE)) {
                return UNAVAILABLE_EVENT_DISPLAY;
            }
            return EventDisplayPresentation.fromView(eventDisplayState.view(
                    eventDisplayReader.snapshot(), System.nanoTime(), gameTick)).encode();
        } catch (RuntimeException exception) {
            stopEventDisplay();
            return UNAVAILABLE_EVENT_DISPLAY;
        }
    }

    private static void startEventDisplay() {
        String path = System.getProperty("neurocraft.fly.eventDisplayPath");
        if (comparison != null && config != null && epochDiagnosticsDirectory != null) {
            path = epochDiagnosticsDirectory.resolve(Long.toString(config.identity().sessionEpoch()))
                    .resolve("event-selection-display.json").toString();
        }
        if (config == null || path == null
                || !ControlProtocol.isEventObservationSchema(config.identity().observationSchemaHash())) { return; }
        try {
            eventDisplayState = new EventDisplayState(config.identity());
            eventDisplayReader = new EventDisplayReader(Path.of(path), config.identity());
        } catch (RuntimeException exception) {
            stopEventDisplay();
            LOGGER.warn("Event display unavailable; command bridge continues");
        }
    }

    private static void stopEventDisplay() {
        try {
            if (eventDisplayReader != null) { eventDisplayReader.close(); }
        } catch (RuntimeException exception) {
            LOGGER.warn("Event display cleanup failed; command bridge is independent");
        } finally { eventDisplayReader = null; eventDisplayState = null; }
    }

    private static void startBrainDisplay() {
        String path = System.getProperty("neurocraft.fly.brainDisplayPath");
        if (comparison != null && config != null && epochDiagnosticsDirectory != null) {
            path = epochDiagnosticsDirectory.resolve(Long.toString(config.identity().sessionEpoch()))
                    .resolve("brain-display.json").toString();
        }
        if (config == null || path == null) {
            return;
        }
        try {
            brainDisplayState = new BrainDisplayState(config.identity());
            brainDisplayReader = new BrainDisplayReader(Path.of(path), config.identity());
        } catch (RuntimeException exception) {
            stopBrainDisplay();
            LOGGER.warn("Neural activity display unavailable; command bridge continues");
        }
    }

    private static void stopBrainDisplay() {
        try {
            if (brainDisplayReader != null) {
                brainDisplayReader.close();
            }
        } catch (RuntimeException exception) {
            LOGGER.warn("Neural activity display cleanup failed; command bridge is independent");
        } finally {
            brainDisplayReader = null;
            brainDisplayState = null;
        }
    }

    /** Enqueues a non-motor operator control request; callers must supply the authoritative tick. */
    public static boolean requestControl(
            long issuedTick, ControlOperation operation, String contentId, String contentHash) {
        if (programTransactionPending() || matchedTrialsActive()) { return false; }
        boolean requested = bridge.requestControl(issuedTick, operation, contentId, contentHash).isPresent();
        // This low-level API leaves ACK polling to its caller; it cannot retain a reusable
        // operator grant across an operation whose completion this runtime does not route.
        if (requested) { operatorProgramGrant = null; }
        return requested;
    }

    /** Routes an operator command's eventual ACK to a bounded server-thread consumer. */
    public static boolean requestOperatorControl(
            long issuedTick,
            ControlOperation operation,
            String contentId,
            String contentHash,
            Consumer<String> success,
            Consumer<String> failure) {
        Consumer<String> acceptedReporter =
                java.util.Objects.requireNonNull(success, "success");
        Consumer<String> rejectedReporter =
                java.util.Objects.requireNonNull(failure, "failure");
        if (matchedTrialsActive() || pendingOperatorControl != null
                || programTransactionPending()
                || RESET_SLOT.pending().isPresent() || (comparison != null && comparison.pending())) {
            return false;
        }
        OptionalLong requested =
                bridge.requestControl(issuedTick, operation, contentId, contentHash);
        if (requested.isEmpty()) {
            return false;
        }
        invalidateOperatorGrant(operation);
        pendingOperatorControl = new PendingOperatorControl(
                requested.orElseThrow(), operation, acceptedReporter, rejectedReporter);
        return true;
    }

    /** Stages the world mutation until the exact accepted RESET acknowledgement boundary. */
    public static boolean requestReset(
            FlyEntity entity,
            long issuedTick,
            double x,
            double y,
            double z,
            float yawDegrees,
            Consumer<String> success,
            Consumer<String> failure) {
        Consumer<String> acceptedReporter =
                java.util.Objects.requireNonNull(success, "success");
        Consumer<String> rejectedReporter =
                java.util.Objects.requireNonNull(failure, "failure");
        processPendingOperatorControl();
        if (matchedTrialsActive() || pendingOperatorControl != null || programTransactionPending()
                || (comparison != null && comparison.pending())) {
            return false;
        }
        MotorAuthority authority = entity.bodyPrograms().authority();
        if (authority != MotorAuthority.NEURAL) {
            // A reset would drop the running program and leave the two lanes disagreeing about who
            // owns the body. The remedy differs by lane, and naming the wrong one wedges the
            // operator: `program release` issues a RESUME, which an already-RUNNING service
            // refuses with INVALID_STATE, so it is the remedy only for the operator lane.
            rejectedReporter.accept(
                    "Refusing to reset while a body program holds motor authority ("
                            + authority
                            + "); "
                            + (authority == MotorAuthority.READOUT
                                    ? "pause the neural service first so it stops selecting"
                                            + " programs"
                                    : "run /neurocraftfly program release first"));
            return false;
        }
        StagedReset transaction = new StagedReset(
                entity.getUUID(), issuedTick, x, y, z, yawDegrees);
        if (!RESET_SLOT.stage(transaction)) {
            return false;
        }
        OptionalLong requested =
                bridge.requestControl(issuedTick, ControlOperation.RESET, null, null);
        if (requested.isEmpty()) {
            RESET_SLOT.clear(transaction);
            return false;
        }
        operatorProgramGrant = null;
        transaction.bindControlSequence(requested.orElseThrow());
        resetSuccess = acceptedReporter;
        resetFailure = rejectedReporter;
        resetStatus = "pending issued_tick=" + issuedTick;
        return true;
    }

    /** Called at the authoritative entity tick boundary before sensing or actuation. */
    public static void processPendingReset(FlyEntity entity, long currentTick) {
        Optional<StagedReset> available = RESET_SLOT.pending();
        if (available.isEmpty()) {
            return;
        }
        StagedReset transaction = available.orElseThrow();
        if (!transaction.targets(entity.getUUID())) {
            return;
        }
        StagedReset.Outcome outcome = transaction.advance(
                currentTick,
                bridge.pollControlAck(
                        transaction.controlSequence(), ControlOperation.RESET));
        switch (outcome) {
            case WAITING -> {
                return;
            }
            case READY -> {
                CommandValidator validator = VALIDATORS.get(entity.getUUID());
                if (validator != null) {
                    validator.resetActuationWindow(transaction.effectiveTick());
                }
                transaction.apply(entity, currentTick);
                resetStatus = "applied effective_tick=" + transaction.effectiveTick();
                finishReset(
                        transaction,
                        resetSuccess,
                        "Applied accepted neural RESET at effective tick "
                                + transaction.effectiveTick(),
                        null);
            }
            case REJECTED -> {
                resetStatus = "rejected fault=" + transaction.failureCode();
                finishReset(
                        transaction,
                        resetFailure,
                        "Neural RESET was rejected; world state was not changed: "
                                + transaction.failureCode(),
                        null);
            }
            case MISSED -> {
                resetStatus = "missed fault=" + transaction.failureCode();
                finishReset(
                        transaction,
                        resetFailure,
                        "Neural RESET missed its ACK boundary; world state was not changed",
                        ControllerFault.DEADLINE_MISSED);
            }
            case INVALID -> {
                resetStatus = "invalid fault=" + transaction.failureCode();
                finishReset(
                        transaction,
                        resetFailure,
                        "Neural RESET ACK was invalid; world state was not changed",
                        ControllerFault.PROTOCOL_ERROR);
            }
            case APPLIED -> throw new IllegalStateException("applied reset remained pending");
        }
    }

    public static String resetStatus() {
        return resetStatus;
    }

    /**
     * First acquires scripted authority through an accepted PAUSE (ADR 0020), or stages a new
     * selection for the next entity tick under that same retained, identity-bound authority.
     *
     * <p>Same two-phase shape as {@link #requestReset}: the control request goes out now, the
     * authority handover happens only when the bound acknowledgement has landed and the entity
     * reaches the tick it governs ({@link StagedProgramAuthority#engageTick()}, one action
     * budget after the observation tick the service stamped it with -- the tick the first
     * paused command applies on), and a rejected, invalid, or missed acknowledgement leaves the
     * fly on the neural lane.
     */
    public static boolean requestProgramAuthority(
            FlyEntity entity,
            long issuedTick,
            BodyProgramId program,
            Modulation modulation,
            Consumer<String> success,
            Consumer<String> failure) {
        java.util.Objects.requireNonNull(success, "success");
        java.util.Objects.requireNonNull(failure, "failure");
        java.util.Objects.requireNonNull(program, "program");
        java.util.Objects.requireNonNull(modulation, "modulation");
        processPendingOperatorControl();
        if (operatorProgramGrant != null && operatorProgramGrant.matches(entity)) {
            if (programChangeBlocked() || issuedTick < 0 || issuedTick == Long.MAX_VALUE
                    || issuedTick != entity.level().getGameTime()) {
                return false;
            }
            pendingLocalProgramChange = new LocalProgramChange(
                    operatorProgramGrant, issuedTick + 1, program, modulation);
            programAuthoritySuccess = success;
            programAuthorityFailure = failure;
            programAuthorityStatus = "pending local_program=" + program + " apply_tick=" + (issuedTick + 1)
                    + " pause_control_seq=" + operatorProgramGrant.pauseControlSequence();
            return true;
        }
        return stageProgramAuthority(
                StagedProgramAuthority.engage(entity.getUUID(), issuedTick, program, modulation),
                issuedTick,
                success,
                failure);
    }

    /** The mirror of {@link #requestProgramAuthority}: RESUME, then drop authority on the ACK. */
    public static boolean releaseProgramAuthority(
            FlyEntity entity, long issuedTick, Consumer<String> success, Consumer<String> failure) {
        return stageProgramAuthority(
                StagedProgramAuthority.release(entity.getUUID(), issuedTick),
                issuedTick,
                success,
                failure);
    }

    /** Called at the authoritative entity tick boundary before sensing or actuation. */
    public static void processPendingProgramAuthority(FlyEntity entity, long currentTick) {
        if (pendingLocalProgramChange != null) {
            processLocalProgramChange(entity, currentTick);
            return;
        }
        if (operatorProgramGrant != null && operatorProgramGrant.entity().getUUID().equals(entity.getUUID())
                && !operatorProgramGrant.matches(entity)) {
            operatorProgramGrant = null;
        }
        StagedProgramAuthority transaction = pendingProgramAuthority;
        if (transaction == null || !transaction.targets(entity.getUUID())) {
            return;
        }
        StagedProgramAuthority.Outcome outcome = transaction.advance(
                currentTick,
                bridge.pollControlAck(transaction.controlSequence(), transaction.operation()));
        switch (outcome) {
            case WAITING -> {
                return;
            }
            case READY -> {
                transaction.apply(entity.bodyPrograms(), currentTick);
                operatorProgramGrant = transaction.operation() == ControlOperation.PAUSE && config != null
                        ? new OperatorProgramGrant(entity, entity.level(), config.identity(), bridge,
                                transaction.controlSequence()) : null;
                programAuthorityStatus = "applied effective_tick=" + transaction.effectiveTick()
                        + " engaged_tick=" + transaction.engageTick();
                finishProgramAuthority(
                        programAuthoritySuccess,
                        "Applied accepted neural "
                                + transaction.operation()
                                + " (effective from observation tick "
                                + transaction.effectiveTick()
                                + ") at tick "
                                + transaction.engageTick()
                                + "; motor authority is now "
                                + entity.bodyPrograms().authority());
            }
            case REJECTED -> {
                programAuthorityStatus = "rejected fault=" + transaction.failureCode();
                finishProgramAuthority(
                        programAuthorityFailure,
                        "Neural " + transaction.operation() + " was rejected; motor authority is"
                                + " unchanged: " + transaction.failureCode());
            }
            case MISSED -> {
                programAuthorityStatus = "missed fault=" + transaction.failureCode();
                finishProgramAuthority(
                        programAuthorityFailure,
                        "Neural " + transaction.operation() + " missed its ACK boundary; motor"
                                + " authority is unchanged, but the service may have applied the"
                                + " operation anyway. If it is now paused, further program"
                                + " requests fail INVALID_STATE until /neurocraftfly resume");
            }
            case INVALID -> {
                programAuthorityStatus = "invalid fault=" + transaction.failureCode();
                finishProgramAuthority(
                        programAuthorityFailure,
                        "Neural " + transaction.operation() + " ACK was invalid; motor authority"
                                + " is unchanged");
            }
            case APPLIED -> throw new IllegalStateException("applied authority remained pending");
        }
    }

    private static boolean programTransactionPending() {
        return pendingProgramAuthority != null || pendingLocalProgramChange != null;
    }

    private static boolean programChangeBlocked() {
        return matchedTrialsActive() || pendingOperatorControl != null || programTransactionPending()
                || RESET_SLOT.pending().isPresent() || (comparison != null && comparison.pending());
    }

    private static void invalidateOperatorGrant(ControlOperation operation) {
        if (operation == ControlOperation.PAUSE || operation == ControlOperation.RESUME
                || operation == ControlOperation.RESET || operation == ControlOperation.SINGLE_STEP
                || operation == ControlOperation.REPLAY_START || operation == ControlOperation.REPLAY_STOP) {
            operatorProgramGrant = null;
        }
    }

    /** Selection only: the existing body tick still requires an admitted paused-neutral command. */
    private static void processLocalProgramChange(FlyEntity entity, long currentTick) {
        LocalProgramChange change = pendingLocalProgramChange;
        if (!change.grant().entity().getUUID().equals(entity.getUUID())) { return; }
        boolean valid = change.grant() == operatorProgramGrant && change.grant().matches(entity)
                && !matchedTrialsActive() && pendingOperatorControl == null && pendingProgramAuthority == null
                && RESET_SLOT.pending().isEmpty() && (comparison == null || !comparison.pending());
        if (!valid || currentTick > change.applyTick() || currentTick < change.applyTick() - 1) {
            programAuthorityStatus = "cancelled local_program=" + change.program()
                    + " reason=" + (valid ? "tick_boundary" : "authority_changed");
            if (!valid) { operatorProgramGrant = null; }
            finishProgramAuthority(programAuthorityFailure,
                    "Local scripted program change was cancelled; no new PAUSE was sent and the requested selection was not applied");
            return;
        }
        if (currentTick < change.applyTick()) { return; }
        entity.bodyPrograms().engageScripted(change.program(), change.modulation(), SelectionSource.OPERATOR, currentTick);
        programAuthorityStatus = "selected local_program=" + change.program() + " selected_tick=" + currentTick
                + " pause_control_seq=" + change.grant().pauseControlSequence();
        finishProgramAuthority(programAuthoritySuccess,
                "Selected scripted program " + change.program() + " at tick " + currentTick
                        + " under existing acknowledged SCRIPTED authority; no new PAUSE was sent."
                        + " Execution still requires an admitted neutral command and body preconditions");
    }

    public static String programAuthorityStatus() {
        return programAuthorityStatus;
    }

    private static boolean stageProgramAuthority(
            StagedProgramAuthority transaction,
            long issuedTick,
            Consumer<String> success,
            Consumer<String> failure) {
        Consumer<String> acceptedReporter = java.util.Objects.requireNonNull(success, "success");
        Consumer<String> rejectedReporter = java.util.Objects.requireNonNull(failure, "failure");
        processPendingOperatorControl();
        if (matchedTrialsActive() || pendingOperatorControl != null
                || programTransactionPending()
                || RESET_SLOT.pending().isPresent() || (comparison != null && comparison.pending())) {
            return false;
        }
        OptionalLong requested =
                bridge.requestControl(issuedTick, transaction.operation(), null, null);
        if (requested.isEmpty()) {
            return false;
        }
        transaction.bindControlSequence(requested.orElseThrow());
        // A new PAUSE/RESUME supersedes any earlier grant, even if its ACK is later refused.
        operatorProgramGrant = null;
        pendingProgramAuthority = transaction;
        programAuthoritySuccess = acceptedReporter;
        programAuthorityFailure = rejectedReporter;
        programAuthorityStatus =
                "pending operation=" + transaction.operation() + " issued_tick=" + issuedTick;
        return true;
    }

    private static void finishProgramAuthority(Consumer<String> reporter, String message) {
        pendingProgramAuthority = null;
        pendingLocalProgramChange = null;
        programAuthoritySuccess = ignored -> {};
        programAuthorityFailure = ignored -> {};
        try {
            reporter.accept(message);
        } catch (RuntimeException exception) {
            LOGGER.warn("Could not deliver staged program-authority completion message");
        }
    }

    private static void processPendingOperatorControl() {
        // A removed/unloaded entity must not retain the global local-selection slot forever.
        // This boundary only cancels; selection itself remains exclusive to the entity tick.
        if (pendingLocalProgramChange != null) {
            LocalProgramChange change = pendingLocalProgramChange;
            boolean valid = change.grant() == operatorProgramGrant && change.grant().matches(change.grant().entity());
            if (!valid || change.grant().level().getGameTime() > change.applyTick()) {
                programAuthorityStatus = "cancelled local_program=" + change.program() + " reason=authority_or_tick";
                if (!valid) { operatorProgramGrant = null; }
                finishProgramAuthority(programAuthorityFailure,
                        "Local scripted program change was cancelled; no new PAUSE was sent and the requested selection was not applied");
            }
        }
        PendingOperatorControl pending = pendingOperatorControl;
        if (pending == null) {
            return;
        }
        Optional<ControlAckFrame> available =
                bridge.pollControlAck(pending.controlSequence(), pending.operation());
        if (available.isEmpty()) {
            return;
        }
        ControlAckFrame acknowledgement = available.orElseThrow();
        pendingOperatorControl = null;
        Consumer<String> reporter =
                acknowledgement.accepted() ? pending.success() : pending.failure();
        String message = acknowledgement.accepted()
                ? "Accepted NeuroCraft control " + acknowledgement.operation()
                        + " at effective tick " + acknowledgement.effectiveTick()
                : "Rejected NeuroCraft control " + acknowledgement.operation()
                        + ": " + acknowledgement.faultCode();
        try {
            reporter.accept(message);
        } catch (RuntimeException exception) {
            LOGGER.warn("Could not deliver operator control acknowledgement message");
        }
    }

    private static void finishReset(
            StagedReset transaction,
            Consumer<String> reporter,
            String message,
            ControllerFault terminalFault) {
        RESET_SLOT.clear(transaction);
        resetSuccess = ignored -> {};
        resetFailure = ignored -> {};
        if (terminalFault != null) {
            bridge.failClosed(terminalFault);
        }
        try {
            reporter.accept(message);
        } catch (RuntimeException exception) {
            LOGGER.warn("Could not deliver staged RESET completion message");
        }
    }

    public static Optional<ControlAckFrame> pollControlAcknowledgement() {
        return bridge.pollControlAck();
    }

    public static long sentActuationReceiptCount() {
        return bridge.sentActuationReceiptCount();
    }

    public static java.util.OptionalLong lastSentActuationReceiptCommandSequence() {
        return bridge.lastSentActuationReceiptCommandSequence();
    }

    public static Optional<UUID> configuredEntityUuid() {
        if (config == null) {
            return Optional.empty();
        }
        try {
            return Optional.of(UUID.fromString(config.identity().entityId()));
        } catch (IllegalArgumentException exception) {
            return Optional.empty();
        }
    }

    /** Read-only decoder identity for cosmetic presentation; never feeds the handshake. */
    public static Optional<String> configuredDecoderHash() {
        if (config == null) {
            return Optional.empty();
        }
        return Optional.of(config.identity().decoderHash());
    }

    private static void start() {
        stop();
        try {
            config = BridgeConfig.load();
            bridge = new NeuralBridge(config);
            String catalogPath = System.getProperty("neurocraft.fly.laneCatalogPath");
            String catalogHash = System.getProperty("neurocraft.fly.laneCatalogSha256");
            String diagnostics = System.getProperty("neurocraft.fly.epochDiagnosticsDirectory");
            if (catalogPath != null || catalogHash != null || diagnostics != null) {
                if (catalogPath == null || catalogHash == null || diagnostics == null) {
                    throw new IllegalArgumentException("comparison requires catalog, digest and diagnostics");
                }
                LaneCatalog catalog = LaneCatalog.load(Path.of(catalogPath), catalogHash, config.identity());
                epochDiagnosticsDirectory = Path.of(diagnostics);
                comparison = new ComparisonLaneSession(catalog, config, bridge, NeuralBridge::new,
                        System::nanoTime);
            }
            bridge.start();
            LOGGER.info("NeuroCraft Fly bridge connecting to the configured loopback endpoint");
        } catch (IllegalArgumentException | IOException | ProtocolException exception) {
            bridge.close();
            comparison = null;
            epochDiagnosticsDirectory = null;
            config = null;
            bridge = new NeutralBridge(ControllerFault.MISSING_CONFIGURATION);
            LOGGER.warn("NeuroCraft Fly bridge configuration is missing or invalid; actuation is neutral");
        } catch (RuntimeException exception) {
            bridge.close();
            comparison = null;
            epochDiagnosticsDirectory = null;
            config = null;
            bridge = new NeutralBridge(ControllerFault.INTERNAL_ERROR);
            LOGGER.error("NeuroCraft Fly bridge initialization failed; actuation is neutral");
        }
        startBrainDisplay();
        startEventDisplay();
    }

    private static void stop() {
        if (matchedTrials != null) { matchedTrials.stop("server stopping"); matchedTrials = null; }
        stopBrainDisplay();
        stopEventDisplay();
        bridge.close();
        VALIDATORS.clear();
        RESET_SLOT.clear();
        resetSuccess = ignored -> {};
        resetFailure = ignored -> {};
        resetStatus = "none";
        pendingOperatorControl = null;
        pendingProgramAuthority = null;
        pendingLocalProgramChange = null;
        programAuthoritySuccess = ignored -> {};
        programAuthorityFailure = ignored -> {};
        programAuthorityStatus = "none";
        operatorProgramGrant = null;
        config = null;
        comparison = null;
        epochDiagnosticsDirectory = null;
        bridge = new NeutralBridge(ControllerFault.MISSING_CONFIGURATION);
    }

    /** Unforgeable outside this runtime; created only when its bound PAUSE actually applies. */
    private record OperatorProgramGrant(
            FlyEntity entity, net.minecraft.world.level.Level level, ControllerIdentity identity,
            ControllerBridge grantedBridge, long pauseControlSequence) {
        boolean matches(FlyEntity candidate) {
            return entity == candidate && !candidate.isRemoved() && candidate.isAlive()
                    && candidate.level() == level && config != null && identity.equals(config.identity())
                    && identity.entityId().equals(candidate.getStringUUID()) && grantedBridge == bridge
                    && bridge.state() == ControllerFault.ACTIVE && bridge.pendingTerminalFault().isEmpty()
                    && candidate.bodyPrograms().authority() == MotorAuthority.SCRIPTED;
        }
    }

    private record LocalProgramChange(
            OperatorProgramGrant grant, long applyTick, BodyProgramId program, Modulation modulation) {}

    private record PendingOperatorControl(
            long controlSequence,
            ControlOperation operation,
            Consumer<String> success,
            Consumer<String> failure) {}
}
