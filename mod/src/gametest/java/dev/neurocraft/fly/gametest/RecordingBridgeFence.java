package dev.neurocraft.fly.gametest;

import dev.neurocraft.fly.FlyControllerRuntime;
import dev.neurocraft.fly.body.ProgramExecution;
import dev.neurocraft.fly.control.ActionFrame;
import dev.neurocraft.fly.control.BridgeConfig;
import dev.neurocraft.fly.control.CommandFrame;
import dev.neurocraft.fly.control.ControlAckFrame;
import dev.neurocraft.fly.control.ControlOperation;
import dev.neurocraft.fly.control.ControlProtocol;
import dev.neurocraft.fly.control.ControllerBridge;
import dev.neurocraft.fly.control.ControllerFault;
import dev.neurocraft.fly.control.ControllerIdentity;
import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import dev.neurocraft.fly.control.ProgramActionFrame;
import dev.neurocraft.fly.control.ValidationResult;
import dev.neurocraft.fly.entity.ActuationReceipt;
import dev.neurocraft.fly.entity.FlyEntity;
import java.lang.reflect.Field;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.Properties;
import java.util.UUID;

/**
 * Installs a recording {@link ControllerBridge} into {@link FlyControllerRuntime} so a GameTest can
 * drive {@code FlyEntity.tick()} down its <em>real</em> path: {@code control()} into the real
 * {@code CommandValidator}, the real arbitration, and the real {@code recordActuation}.
 *
 * <p>The default GameTest lane has no bridge at all, so the accepted and rejected arms of the tick
 * are otherwise unreachable and every other pin in this package drives the runtime directly. Some
 * claims cannot be made that way — "a rejected program frame never reaches the body" is a claim
 * about the wiring between the validator and the body, not about either end — so this fence exists
 * for exactly those.
 *
 * <p>It reaches two private static fields by reflection. That is deliberate and contained:
 * {@link #install} always pairs with {@link Installed#close()} in a {@code finally}, restoring
 * whatever was there before, because seventy GameTests share one server and a leaked bridge would
 * poison every test that ran afterwards.
 */
final class RecordingBridgeFence {
    private static final String TOKEN = "0123456789abcdef0123456789abcdef";

    private RecordingBridgeFence() {}

    /**
     * A live installation. Always use it in try-with-resources.
     *
     * <p>{@link #close()} drops this fly's {@code CommandValidator} as well as restoring the two
     * fields. That entry is created lazily by {@code FlyControllerRuntime.control} and is keyed by
     * entity UUID in a map that lives for the whole server, so leaving it behind would leave a
     * session -- its latched frame lane, its sequence cursor, and any terminal fault it reached --
     * bound to a fly whose bridge no longer exists. It is not reachable by a later test today,
     * because every test spawns its own fly, but "not reachable today" is not the same as
     * restored, and this fence's whole contract is that it restores what it touched.
     */
    record Installed(
            RecordingBridge bridge, UUID flyId, Object previousConfig, Object previousBridge)
            implements AutoCloseable {
        @Override
        public void close() {
            try {
                configField().set(null, previousConfig);
                bridgeField().set(null, previousBridge);
                ((Map<?, ?>) validatorsField().get(null)).remove(flyId);
            } catch (ReflectiveOperationException exception) {
                throw new AssertionError("could not restore the controller runtime", exception);
            }
        }
    }

    static Installed install(FlyEntity fly) {
        return install(fly, ReadoutLaneFence.identity(fly));
    }

    static Installed install(FlyEntity fly, ControllerIdentity identity) {
        try {
            BridgeConfig config = configFor(fly, identity);
            RecordingBridge bridge = new RecordingBridge(config);
            Object previousConfig = configField().get(null);
            Object previousBridge = bridgeField().get(null);
            configField().set(null, config);
            bridgeField().set(null, bridge);
            return new Installed(bridge, fly.getUUID(), previousConfig, previousBridge);
        } catch (ReflectiveOperationException exception) {
            throw new AssertionError("could not install the recording bridge", exception);
        }
    }

    /**
     * The session identity, built from properties exactly as a real run builds it, and asserted
     * equal to the one {@link ReadoutLaneFence} stamps on its frames. A drift between the two would
     * turn every pin here into an IDENTITY_MISMATCH pin by accident.
     */
    private static BridgeConfig configFor(FlyEntity fly, ControllerIdentity expected) {
        Properties properties = new Properties();
        properties.setProperty("neurocraft.fly.runId", expected.runId());
        properties.setProperty(
                "neurocraft.fly.sessionEpoch", Long.toString(expected.sessionEpoch()));
        properties.setProperty("neurocraft.fly.entityId", expected.entityId());
        properties.setProperty("neurocraft.fly.graphHash", expected.graphHash());
        properties.setProperty("neurocraft.fly.adapterHash", expected.adapterHash());
        properties.setProperty("neurocraft.fly.dynamicsHash", expected.dynamicsHash());
        properties.setProperty("neurocraft.fly.decoderHash", expected.decoderHash());
        properties.setProperty(
                "neurocraft.fly.observationSchemaHash", expected.observationSchemaHash());
        BridgeConfig config = BridgeConfig.from(Map.of("NEUROCRAFT_RUN_TOKEN", TOKEN), properties);
        if (!expected.equals(config.identity())) {
            throw new AssertionError("the recording bridge's identity drifted from the fence's");
        }
        return config;
    }

    private static Field configField() throws ReflectiveOperationException {
        return accessible("config");
    }

    private static Field bridgeField() throws ReflectiveOperationException {
        return accessible("bridge");
    }

    private static Field validatorsField() throws ReflectiveOperationException {
        return accessible("VALIDATORS");
    }

    private static Field accessible(String name) throws ReflectiveOperationException {
        Field field = FlyControllerRuntime.class.getDeclaredField(name);
        field.setAccessible(true);
        return field;
    }

    /**
     * A bridge that hands the server thread whatever frames a test queued and keeps every receipt
     * the runtime offered it, encoded through the same {@link ControlProtocol} calls the real
     * bridge uses — so a receipt the runtime could not legally encode fails the test loudly instead
     * of vanishing.
     */
    static final class RecordingBridge implements ControllerBridge {
        private final BridgeConfig config;
        private final ArrayDeque<CommandFrame> inbound = new ArrayDeque<>();
        private final Map<Long, Long> sentObservations = new HashMap<>();
        private final List<String> programReceipts = new ArrayList<>();
        private final List<String> actuationReceipts = new ArrayList<>();
        private final List<String> observations = new ArrayList<>();
        private long nextObservationSequence;
        private boolean controlsEnabled;
        private final List<ControlRequest> controls = new ArrayList<>();
        private final ArrayDeque<ControlAckFrame> acknowledgements = new ArrayDeque<>();

        record ControlRequest(long sequence, long issuedTick, ControlOperation operation) {}

        void enableControls() { controlsEnabled = true; }
        List<ControlRequest> controls() { return List.copyOf(controls); }
        void acknowledge(ControlAckFrame frame) { acknowledgements.addLast(frame); }
        private ControllerFault state = ControllerFault.ACTIVE;

        RecordingBridge(BridgeConfig config) {
            this.config = config;
        }

        /** Queues a frame for the next {@code control()} poll, as the socket thread would. */
        void offer(CommandFrame frame) {
            inbound.addLast(frame);
        }

        List<String> observations() {
            return List.copyOf(observations);
        }

        List<String> programReceipts() {
            return List.copyOf(programReceipts);
        }

        List<String> actuationReceipts() {
            return List.copyOf(actuationReceipts);
        }

        /**
         * The observation sequence the bridge assigned to a snapshot at this tick, if any.
         *
         * <p>Callers that stream frames need it to anchor the first frame of a session on an
         * observation the bridge really recorded, rather than assuming the counter started where
         * they expect. <strong>When more than one observation was taken at the same game tick it
         * returns an arbitrary one of them</strong>, which is exactly what a caller anchoring a
         * stream wants and is not a lookup that can distinguish them.
         */
        OptionalLong observationSequenceAt(long gameTick) {
            return sentObservations.entrySet().stream()
                    .filter(entry -> entry.getValue() == gameTick)
                    .mapToLong(Map.Entry::getKey)
                    .findFirst();
        }

        @Override
        public void start() {}

        @Override
        public void offerObservation(PhysicalSensorSnapshot snapshot) {
            observations.add(ControlProtocol.encodeObservation(
                    config.identity(), snapshot, nextObservationSequence, 0, 0));
            sentObservations.put(nextObservationSequence++, snapshot.gameTick());
        }

        @Override
        public Optional<CommandFrame> pollAction() {
            return Optional.ofNullable(inbound.pollFirst());
        }

        @Override
        public OptionalLong sentObservationTick(long observationSequence) {
            Long tick = sentObservations.get(observationSequence);
            return tick == null ? OptionalLong.empty() : OptionalLong.of(tick);
        }

        @Override
        public void offerActuationReceipt(
                ActionFrame command,
                PhysicalSensorSnapshot worldBefore,
                ActuationReceipt receipt,
                ValidationResult decision) {
            actuationReceipts.add(ControlProtocol.encodeActuationReceipt(
                    config.identity(), command, worldBefore, receipt, decision));
        }

        @Override
        public void offerProgramReceipt(
                ProgramActionFrame command,
                PhysicalSensorSnapshot worldBefore,
                Optional<ProgramExecution> execution,
                ValidationResult decision) {
            programReceipts.add(ControlProtocol.encodeProgramReceipt(
                    config.identity(), command, worldBefore, execution, decision));
        }

        @Override
        public OptionalLong requestControl(
                long issuedTick, ControlOperation operation, String contentId, String contentHash) {
            if (!controlsEnabled || state != ControllerFault.ACTIVE) { return OptionalLong.empty(); }
            long sequence = controls.size();
            controls.add(new ControlRequest(sequence, issuedTick, operation));
            return OptionalLong.of(sequence);
        }

        @Override
        public Optional<ControlAckFrame> pollControlAck() {
            return Optional.ofNullable(acknowledgements.pollFirst());
        }

        @Override
        public Optional<ControlAckFrame> pollControlAck(
                long controlSequence, ControlOperation operation) {
            for (var iterator = acknowledgements.iterator(); iterator.hasNext();) {
                ControlAckFrame frame = iterator.next();
                if (frame.controlSequence() == controlSequence && frame.operation() == operation) {
                    iterator.remove();
                    return Optional.of(frame);
                }
            }
            return Optional.empty();
        }

        @Override
        public long sentActuationReceiptCount() {
            return actuationReceipts.size();
        }

        @Override
        public long sentProgramReceiptCount() {
            return programReceipts.size();
        }

        @Override
        public OptionalLong lastSentActuationReceiptCommandSequence() {
            return OptionalLong.empty();
        }

        @Override
        public Optional<ControllerFault> pendingTerminalFault() {
            return Optional.empty();
        }

        @Override
        public void failClosed(ControllerFault fault) {
            state = fault;
        }

        @Override
        public ControllerFault state() {
            return state;
        }

        @Override
        public void close() {}
    }
}
