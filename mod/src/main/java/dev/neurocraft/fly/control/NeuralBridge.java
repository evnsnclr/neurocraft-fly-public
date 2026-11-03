package dev.neurocraft.fly.control;

import dev.neurocraft.fly.body.ProgramExecution;
import dev.neurocraft.fly.entity.ActuationReceipt;
import java.net.http.HttpClient;
import java.net.http.WebSocket;
import java.nio.ByteBuffer;
import java.time.Duration;
import java.util.ArrayDeque;
import java.util.Iterator;
import java.util.Locale;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * JDK WebSocket client with bounded, coalescing handoffs. Parsing and serialization never run on
 * the Minecraft server thread.
 */
public final class NeuralBridge implements ControllerBridge, WebSocket.Listener {
    private static final Logger LOGGER = LoggerFactory.getLogger("neurocraft-fly-bridge");
    /**
     * How many actions of a session report their measured round trip. Bounded on purpose: the
     * cold-start question this answers is about the first actions, and a permanent per-action log
     * line would sit on the same single bridge thread that drains the socket.
     */
    private static final int FIRST_ACTION_LATENCY_SAMPLES = 8;

    private static final int OUTBOUND_WINDOW_CAPACITY = 2;
    private static final int CONTROL_WINDOW_CAPACITY = 1;
    private static final int CONTROL_ACK_WINDOW_CAPACITY = 2;
    private static final int TERMINAL_VALIDATION_CLOSE_STATUS = 1008;
    private static final char NO_PENDING_HIGH_SURROGATE = '\0';
    private static final char UTF8_TWO_BYTE_MIN = 0x80;
    private static final char UTF8_THREE_BYTE_MIN = 0x800;
    private static final int UTF8_UNPAIRED_SURROGATE_BYTES = 1;
    private static final int UTF8_SURROGATE_PAIR_BYTES = 4;

    private final BridgeConfig config;
    private final BoundedActionWindow actions = new BoundedActionWindow();
    private final SentObservationWindow sentObservations = new SentObservationWindow();
    private final CoalescingObservationSlot latestObservation = new CoalescingObservationSlot();
    private final FirstActionLatencyWindow firstActionLatency =
            new FirstActionLatencyWindow(FIRST_ACTION_LATENCY_SAMPLES);
    private final AtomicReference<ControllerFault> state =
            new AtomicReference<>(ControllerFault.STARTING);
    private final AtomicBoolean drainScheduled = new AtomicBoolean();
    private final AtomicBoolean closed = new AtomicBoolean();
    private final long createdNanos = System.nanoTime();
    private final AtomicLong sentActuationReceipts = new AtomicLong();
    private final AtomicLong sentProgramReceipts = new AtomicLong();
    private final AtomicLong lastSentReceiptCommandSequence = new AtomicLong(-1);
    private final ExecutorService ioExecutor;
    private final HttpClient httpClient;
    private final StringBuilder inboundText = new StringBuilder();
    private final Object outboundLock = new Object();
    private final ArrayDeque<ReceiptEnvelope> receipts =
            new ArrayDeque<>(OUTBOUND_WINDOW_CAPACITY);
    private final ArrayDeque<ControlCommandFrame> controls =
            new ArrayDeque<>(CONTROL_WINDOW_CAPACITY);
    private final ArrayDeque<ControlCommandFrame> pendingControls =
            new ArrayDeque<>(CONTROL_WINDOW_CAPACITY);
    private final ArrayDeque<ControlAckFrame> controlAcknowledgements =
            new ArrayDeque<>(CONTROL_ACK_WINDOW_CAPACITY);
    private long inboundUtf8Bytes;
    private char pendingHighSurrogate = NO_PENDING_HIGH_SURROGATE;
    private long nextObservationSequence;
    private long nextControlSequence;
    private long lastControlTick = -1;
    private ReceiptEnvelope terminalReceipt;
    private ControllerFault pendingTerminalFault;
    private volatile WebSocket socket;
    private volatile String clientNonce;

    public NeuralBridge(BridgeConfig config) {
        this.config = Objects.requireNonNull(config, "config");
        ThreadFactory threads = runnable -> {
            Thread thread = new Thread(runnable, "neurocraft-fly-bridge");
            thread.setDaemon(true);
            return thread;
        };
        ioExecutor = Executors.newSingleThreadExecutor(threads);
        httpClient = HttpClient.newBuilder()
                .executor(ioExecutor)
                .connectTimeout(BridgeConfig.CONNECT_TIMEOUT)
                .build();
    }

    @Override
    public void start() {
        if (closed.get() || state.get().isPaused()) {
            return;
        }
        diagnostic("connect_begin", null, -1, null);
        httpClient.newWebSocketBuilder()
                .connectTimeout(Duration.ofSeconds(3))
                .subprotocols(ControlProtocol.WEBSOCKET_SUBPROTOCOL)
                .buildAsync(config.uri(), this)
                .whenComplete((connected, error) -> {
                    if (error != null) {
                        latchFault(ControllerFault.DISCONNECTED, "connect_failed", error, -1);
                    } else if (closed.get()) {
                        connected.abort();
                    } else {
                        socket = connected;
                    }
                });
    }

    @Override
    public void offerObservation(PhysicalSensorSnapshot snapshot) {
        if (pendingTerminalFault().isPresent()) {
            return;
        }
        if (!latestObservation.offer(Objects.requireNonNull(snapshot, "snapshot"))) {
            latchFault(ControllerFault.QUEUE_OVERFLOW);
        } else {
            scheduleDrain();
        }
    }

    @Override
    public Optional<CommandFrame> pollAction() {
        return actions.poll();
    }

    @Override
    public OptionalLong sentObservationTick(long observationSequence) {
        return sentObservations.tickFor(observationSequence);
    }

    @Override
    public void offerActuationReceipt(
            ActionFrame command,
            PhysicalSensorSnapshot worldBefore,
            ActuationReceipt receipt,
            ValidationResult decision) {
        if (state.get() != ControllerFault.ACTIVE || closed.get()) {
            return;
        }
        enqueueReceipt(new ActuationEnvelope(
                Objects.requireNonNull(command, "command"),
                Objects.requireNonNull(worldBefore, "worldBefore"),
                Objects.requireNonNull(receipt, "receipt"),
                Objects.requireNonNull(decision, "decision")));
    }

    @Override
    public void offerProgramReceipt(
            ProgramActionFrame command,
            PhysicalSensorSnapshot worldBefore,
            Optional<ProgramExecution> execution,
            ValidationResult decision) {
        if (state.get() != ControllerFault.ACTIVE || closed.get()) {
            return;
        }
        enqueueReceipt(new ProgramEnvelope(
                Objects.requireNonNull(command, "command"),
                Objects.requireNonNull(worldBefore, "worldBefore"),
                Objects.requireNonNull(execution, "execution"),
                Objects.requireNonNull(decision, "decision")));
    }

    /**
     * The single outbound-receipt path, shared by both receipt kinds so a rejected program receipt
     * closes the session exactly as a rejected actuation receipt does: it becomes the terminal
     * receipt, latches its fault, and is sent before the close frame.
     */
    private void enqueueReceipt(ReceiptEnvelope envelope) {
        ValidationResult decision = envelope.decision();
        boolean overflow;
        synchronized (outboundLock) {
            if (pendingTerminalFault != null) {
                return;
            }
            if (decision.state().isPaused()) {
                terminalReceipt = envelope;
                pendingTerminalFault = decision.state();
                overflow = false;
            } else {
                overflow = receipts.size() == OUTBOUND_WINDOW_CAPACITY;
            }
            if (!overflow && !decision.state().isPaused()) {
                receipts.addLast(envelope);
            }
        }
        if (overflow) {
            latchFault(ControllerFault.QUEUE_OVERFLOW);
        } else {
            scheduleDrain();
        }
    }

    @Override
    public OptionalLong requestControl(
            long issuedTick, ControlOperation operation, String contentId, String contentHash) {
        if (state.get() != ControllerFault.ACTIVE || closed.get()) {
            return OptionalLong.empty();
        }
        ControlCommandFrame command;
        synchronized (outboundLock) {
            if (issuedTick <= lastControlTick
                    || pendingTerminalFault != null
                    || controls.size() + pendingControls.size() >= CONTROL_WINDOW_CAPACITY) {
                return OptionalLong.empty();
            }
            try {
                command = new ControlCommandFrame(
                        config.identity(),
                        nextControlSequence,
                        issuedTick,
                        operation,
                        contentId,
                        contentHash);
            } catch (IllegalArgumentException | NullPointerException exception) {
                return OptionalLong.empty();
            }
            nextControlSequence++;
            lastControlTick = issuedTick;
            controls.addLast(command);
        }
        scheduleDrain();
        return OptionalLong.of(command.controlSequence());
    }

    @Override
    public Optional<ControlAckFrame> pollControlAck() {
        synchronized (outboundLock) {
            return Optional.ofNullable(controlAcknowledgements.pollFirst());
        }
    }

    @Override
    public Optional<ControlAckFrame> pollControlAck(
            long controlSequence, ControlOperation operation) {
        Objects.requireNonNull(operation, "operation");
        synchronized (outboundLock) {
            Iterator<ControlAckFrame> acknowledgements = controlAcknowledgements.iterator();
            while (acknowledgements.hasNext()) {
                ControlAckFrame acknowledgement = acknowledgements.next();
                if (acknowledgement.controlSequence() == controlSequence
                        && acknowledgement.operation() == operation) {
                    acknowledgements.remove();
                    return Optional.of(acknowledgement);
                }
            }
            return Optional.empty();
        }
    }

    @Override
    public long sentActuationReceiptCount() {
        return sentActuationReceipts.get();
    }

    @Override
    public long sentProgramReceiptCount() {
        return sentProgramReceipts.get();
    }

    @Override
    public OptionalLong lastSentActuationReceiptCommandSequence() {
        long sequence = lastSentReceiptCommandSequence.get();
        return sequence < 0 ? OptionalLong.empty() : OptionalLong.of(sequence);
    }

    @Override
    public Optional<ControllerFault> pendingTerminalFault() {
        synchronized (outboundLock) {
            return Optional.ofNullable(pendingTerminalFault);
        }
    }

    @Override
    public void failClosed(ControllerFault fault) {
        latchFault(Objects.requireNonNull(fault, "fault"), "authority_fail_closed", null, -1);
    }

    @Override
    public ControllerFault state() {
        return state.get();
    }

    @Override
    public void onOpen(WebSocket webSocket) {
        if (closed.get()) {
            webSocket.abort();
            return;
        }
        socket = webSocket;
        diagnostic("socket_open", null, -1, null);
        if (!ControlProtocol.WEBSOCKET_SUBPROTOCOL.equals(webSocket.getSubprotocol())) {
            latchFault(ControllerFault.PROTOCOL_ERROR, "subprotocol_refused", null, -1);
            webSocket.abort();
            return;
        }
        webSocket.request(1);
        try {
            // The token stays local: the frame carries only this fresh nonce and an HMAC
            // proof over it. The nonce is retained so the service's hello_ack proof can be
            // verified before the bridge accepts a single command.
            String nonce = ControlProtocol.generateClientNonce();
            clientNonce = nonce;
            String hello =
                    ControlProtocol.encodeHello(config.identity(), config.token(), nonce);
            diagnostic("hello_write_begin", null, -1, null);
            webSocket.sendText(hello, true)
                    .whenComplete((ignored, error) -> {
                        if (error != null) {
                            latchFault(ControllerFault.DISCONNECTED, "hello_write_failed", error, -1);
                        } else {
                            diagnostic("hello_write_complete", null, -1, null);
                        }
                    });
        } catch (RuntimeException exception) {
            latchFault(ControllerFault.DISCONNECTED, "hello_setup_or_write_failed", exception, -1);
        }
    }

    @Override
    public CompletionStage<?> onText(WebSocket webSocket, CharSequence data, boolean last) {
        if (closed.get() || state.get().isPaused()) {
            webSocket.request(1);
            return null;
        }
        if (pendingTerminalFault().isPresent()) {
            resetInbound();
            webSocket.request(1);
            return null;
        }
        inboundUtf8Bytes += utf8ByteLength(data, pendingHighSurrogate);
        pendingHighSurrogate = trailingHighSurrogate(data, pendingHighSurrogate);
        if (inboundUtf8Bytes > ControlProtocol.MAX_MESSAGE_BYTES) {
            resetInbound();
            latchFault(ControllerFault.OVERSIZE_MESSAGE);
            webSocket.request(1);
            return null;
        }
        inboundText.append(data);
        if (last) {
            String message = inboundText.toString();
            resetInbound();
            acceptInbound(message);
        }
        webSocket.request(1);
        return null;
    }

    @Override
    public CompletionStage<?> onBinary(WebSocket webSocket, ByteBuffer data, boolean last) {
        latchFault(data.remaining() > ControlProtocol.MAX_MESSAGE_BYTES
                ? ControllerFault.OVERSIZE_MESSAGE
                : ControllerFault.PROTOCOL_ERROR);
        webSocket.request(1);
        return null;
    }

    @Override
    public CompletionStage<?> onClose(WebSocket webSocket, int statusCode, String reason) {
        if (!closed.get()) {
            latchFault(ControllerFault.DISCONNECTED, "peer_close", null, statusCode);
        } else {
            diagnostic("close_callback_after_local_close", null, statusCode, null);
        }
        return null;
    }

    @Override
    public void onError(WebSocket webSocket, Throwable error) {
        if (!closed.get()) {
            latchFault(ControllerFault.DISCONNECTED, "websocket_error", error, -1);
        }
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        diagnostic("local_close_begin", null, -1, null);
        actions.clear();
        sentObservations.clear();
        latestObservation.clear();
        clearOutbound();
        WebSocket current = socket;
        if (current != null) {
            try {
                current.sendClose(WebSocket.NORMAL_CLOSURE, "server stopping")
                        .whenComplete((ignored, error) -> {
                            diagnostic(error == null ? "close_write_complete" : "close_write_failed",
                                    error, WebSocket.NORMAL_CLOSURE, null);
                        })
                        .exceptionally(error -> {
                            current.abort();
                            return null;
                        });
            } catch (RuntimeException exception) {
                diagnostic("close_write_threw", exception, WebSocket.NORMAL_CLOSURE, null);
                current.abort();
            }
        }
        ioExecutor.shutdown();
        diagnostic("executor_shutdown_requested", null, -1, null);
    }

    /** Accumulated UTF-8 byte count of the in-flight fragmented text message. */
    long inboundUtf8Bytes() {
        return inboundUtf8Bytes;
    }

    /** High surrogate carried across the last fragment boundary, or {@code '\0'} when none. */
    char pendingHighSurrogate() {
        return pendingHighSurrogate;
    }

    /** Round-trip samples this session has still to report; zero once the log goes quiet. */
    int firstActionLatencySamplesRemaining() {
        return firstActionLatency.remaining();
    }

    private void resetInbound() {
        inboundText.setLength(0);
        inboundUtf8Bytes = 0;
        pendingHighSurrogate = NO_PENDING_HIGH_SURROGATE;
    }

    private void acceptInbound(String message) {
        try {
            InboundFrame frame = ControlProtocol.decodeInbound(message);
            if (frame instanceof HelloAck ack) {
                // An acknowledgement that cannot prove possession of the run token for this
                // session's fresh nonce never activates the bridge, so no command from an
                // unauthenticated peer is ever accepted.
                if (state.get() != ControllerFault.STARTING
                        || !config.identity().matches(ack)
                        || !ControlProtocol.verifyHelloAck(
                                ack, config.token(), config.identity(), clientNonce)) {
                    latchFault(ControllerFault.IDENTITY_MISMATCH, "hello_ack_refused", null, -1);
                    return;
                }
                state.set(ControllerFault.ACTIVE);
                diagnostic("hello_ack_verified", null, -1, null);
                scheduleDrain();
            } else if (frame instanceof CommandFrame command) {
                // Stamped before the handoff so the reading is the command's arrival, not the
                // cost of accepting it. One nanoTime() call on the bridge's own IO thread; the
                // Minecraft server thread is never involved. Both driving kinds take this path;
                // refusing a session that mixes them is the server-thread validator's job.
                long arrivedMonotonicNs = System.nanoTime();
                if (state.get() != ControllerFault.ACTIVE) {
                    latchFault(ControllerFault.PROTOCOL_ERROR);
                } else if (command.targetTick() - command.observationTick()
                        != CommandValidator.ACTION_TARGET_OFFSET_TICKS) {
                    // The declared action budget, enforced at the wire boundary rather than in
                    // the frame model (which stays permissive so historical frames still
                    // decode) or in CommandValidator (whose rules are keyed to the
                    // authoritative clock, not to the producer's convention). Every frame that
                    // crosses this socket comes from the service, whose own
                    // _validate_generated_action refuses to emit any other offset, so a frame
                    // failing here is a contract violation between the two halves and fails
                    // closed like any other. PROTOCOL_ERROR is reused for the same reason
                    // lane mixing reuses it: the launcher's terminal-fault pattern is
                    // regression-pinned against ControllerFault's source.
                    LOGGER.warn(
                            "action budget refused: command_seq={} observation_tick={} "
                            + "target_tick={} (expected observation_tick + {})",
                            command.commandSequence(),
                            command.observationTick(),
                            command.targetTick(),
                            CommandValidator.ACTION_TARGET_OFFSET_TICKS);
                    latchFault(ControllerFault.PROTOCOL_ERROR);
                } else if (!actions.offer(command)) {
                    latchFault(ControllerFault.QUEUE_OVERFLOW);
                } else {
                    logFirstActionLatency(command, arrivedMonotonicNs);
                }
            } else if (frame instanceof ControlAckFrame ack) {
                acceptControlAck(ack);
            } else if (frame instanceof FaultFrame fault) {
                if (!config.identity().equals(fault.identity())) {
                    latchFault(ControllerFault.IDENTITY_MISMATCH);
                } else {
                    latchFault(mapRemoteFault(fault.code()));
                }
            }
        } catch (ProtocolException exception) {
            latchFault(message.length() > ControlProtocol.MAX_MESSAGE_BYTES
                    ? ControllerFault.OVERSIZE_MESSAGE
                    : ControllerFault.PROTOCOL_ERROR);
        } catch (RuntimeException exception) {
            latchFault(ControllerFault.INTERNAL_ERROR);
        }
    }

    /**
     * Reports the round trip of one of a session's first actions, after the action itself is
     * already available to the server thread. Silent once the sample budget is spent, and silent
     * for any action whose observation has left the window, so the steady-state cost is one
     * bounded lookup.
     */
    private void logFirstActionLatency(CommandFrame action, long arrivedMonotonicNs) {
        Optional<FirstActionLatencyWindow.Sample> measured =
                firstActionLatency.completeArrival(action.observationSequence(), arrivedMonotonicNs);
        if (measured.isEmpty()) {
            return;
        }
        FirstActionLatencyWindow.Sample sample = measured.get();
        LOGGER.info(
                "NeuroCraft Fly action latency: command_seq={} observation_seq={} "
                        + "offer_to_action={} ms send_to_action={} ms",
                action.commandSequence(),
                sample.observationSequence(),
                milliseconds(sample.offerToActionNs()),
                milliseconds(sample.sendToActionNs()));
    }

    private static String milliseconds(long nanoseconds) {
        return String.format(Locale.ROOT, "%.2f", nanoseconds / 1_000_000.0);
    }

    private void scheduleDrain() {
        if (state.get() != ControllerFault.ACTIVE || closed.get()) {
            return;
        }
        if (drainScheduled.compareAndSet(false, true)) {
            ioExecutor.execute(this::drainOutbound);
        }
    }

    private void drainOutbound() {
        WebSocket current = socket;
        if (current == null || state.get() != ControllerFault.ACTIVE) {
            drainScheduled.set(false);
            rescheduleAfterRace();
            return;
        }
        ReceiptEnvelope receipt;
        ControlCommandFrame control;
        synchronized (outboundLock) {
            receipt = receipts.pollFirst();
            if (receipt == null && terminalReceipt != null) {
                receipt = terminalReceipt;
                terminalReceipt = null;
            }
            control = receipt == null ? controls.pollFirst() : null;
            if (control != null) {
                pendingControls.addLast(control);
            }
        }
        CoalescingObservationSlot.PendingObservation observation =
                receipt == null && control == null ? latestObservation.pollLatest() : null;
        if (receipt == null && control == null && observation == null) {
            drainScheduled.set(false);
            rescheduleAfterRace();
            return;
        }
        ReceiptEnvelope sentReceipt = receipt;
        String message;
        try {
            if (sentReceipt != null) {
                message = sentReceipt.encode(config.identity());
            } else if (control != null) {
                message = ControlProtocol.encodeControlCommand(control);
            } else {
                long observationSequence = nextObservationSequence;
                long sentNanos = System.nanoTime();
                long sentMonotonicNs = sentNanos & Long.MAX_VALUE;
                message = ControlProtocol.encodeObservation(
                        config.identity(),
                        observation.snapshot(),
                        observationSequence,
                        sentMonotonicNs,
                        observation.bridgeObservationsCoalesced());
                sentObservations.record(observationSequence, observation.snapshot().gameTick());
                firstActionLatency.recordSend(
                        observationSequence, observation.offeredMonotonicNs(), sentNanos);
                nextObservationSequence++;
            }
        } catch (RuntimeException exception) {
            latchFault(ControllerFault.INTERNAL_ERROR);
            drainScheduled.set(false);
            return;
        }
        boolean sendsReceipt = sentReceipt != null;
        boolean sendsTerminalReceipt =
                sentReceipt != null && sentReceipt.decision().state().isPaused();
        CompletionStage<WebSocket> send;
        try {
            send = Objects.requireNonNull(
                    current.sendText(message, true), "WebSocket.sendText completion");
        } catch (RuntimeException exception) {
            latchFault(ControllerFault.DISCONNECTED, "frame_write_threw", exception, -1);
            drainScheduled.set(false);
            return;
        }
        send.whenComplete((ignored, error) -> {
            if (error != null) {
                latchFault(ControllerFault.DISCONNECTED, "frame_write_failed", error, -1);
            } else if (sendsReceipt) {
                // The two kinds are counted apart on purpose: a session drives with one kind or the
                // other, and "actuation receipts sent" must not silently include frames that report
                // no motor request at all.
                if (sentReceipt instanceof ProgramEnvelope) {
                    sentProgramReceipts.incrementAndGet();
                } else {
                    sentActuationReceipts.incrementAndGet();
                }
                lastSentReceiptCommandSequence.set(sentReceipt.command().commandSequence());
                if (sendsTerminalReceipt) {
                    latchFault(sentReceipt.decision().state());
                    closeAfterTerminalReceipt(current);
                }
            }
            drainScheduled.set(false);
            if (!sendsTerminalReceipt) {
                rescheduleAfterRace();
            }
        });
    }

    private void closeAfterTerminalReceipt(WebSocket current) {
        try {
            current.sendClose(
                            TERMINAL_VALIDATION_CLOSE_STATUS,
                            "authoritative validation rejected action")
                    .exceptionally(closeError -> {
                        current.abort();
                        return null;
                    });
        } catch (RuntimeException exception) {
            current.abort();
        }
    }

    private void rescheduleAfterRace() {
        if (latestObservation.hasPending() || hasOutbound()) {
            scheduleDrain();
        }
    }

    private void acceptControlAck(ControlAckFrame acknowledgement) {
        if (!config.identity().equals(acknowledgement.identity())) {
            latchFault(ControllerFault.IDENTITY_MISMATCH);
            return;
        }
        boolean matches;
        boolean overflow = false;
        synchronized (outboundLock) {
            ControlCommandFrame expected = pendingControls.peekFirst();
            matches = expected != null && acknowledgement.binds(expected);
            if (matches) {
                pendingControls.removeFirst();
                if (controlAcknowledgements.size() == CONTROL_ACK_WINDOW_CAPACITY) {
                    overflow = true;
                } else {
                    controlAcknowledgements.addLast(acknowledgement);
                }
            }
        }
        if (!matches) {
            latchFault(ControllerFault.SEQUENCE_ERROR);
        } else if (overflow) {
            latchFault(ControllerFault.QUEUE_OVERFLOW);
        }
    }

    private boolean hasOutbound() {
        synchronized (outboundLock) {
            return !receipts.isEmpty() || terminalReceipt != null || !controls.isEmpty();
        }
    }

    private void clearOutbound() {
        synchronized (outboundLock) {
            receipts.clear();
            terminalReceipt = null;
            pendingTerminalFault = null;
            controls.clear();
            pendingControls.clear();
            controlAcknowledgements.clear();
        }
    }

    private void latchFault(ControllerFault fault) {
        latchFault(fault, "fault_latched", null, -1);
    }

    private void latchFault(ControllerFault fault, String stage, Throwable error, int closeCode) {
        if (!fault.isPaused()) {
            throw new IllegalArgumentException("fault must pause the bridge");
        }
        ControllerFault previous = state.getAndUpdate(current -> current.isPaused() ? current : fault);
        if (!previous.isPaused()) {
            diagnostic(stage, error, closeCode, fault);
        }
        actions.clear();
        sentObservations.clear();
        latestObservation.clear();
        firstActionLatency.clear();
        clearOutbound();
    }

    /** Local lifecycle elapsed time only: no per-frame log, handshake payload or peer error text. */
    private void diagnostic(String stage, Throwable error, int closeCode, ControllerFault fault) {
        try {
            String errorClass = error == null ? "none" : error.getClass().getSimpleName();
            LOGGER.warn("bridge_lifecycle epoch={} stage={} bridge_elapsed_ms={} error_class={} close_code={} fault={}",
                    config.identity().sessionEpoch(), stage, (System.nanoTime() - createdNanos) / 1_000_000L,
                    errorClass.substring(0, Math.min(errorClass.length(), 80)), closeCode,
                    fault == null ? "none" : fault.name());
        } catch (RuntimeException ignored) {
            // Observability must not acquire actuation or lifecycle authority.
        }
    }

    /**
     * UTF-8 byte growth contributed by {@code text} when appended to accumulated text whose
     * trailing high surrogate is {@code carry} ({@code '\0'} when there is none). A surrogate
     * pair costs four bytes even when the fragment boundary splits it: the high surrogate is
     * charged one byte on arrival (the replacement byte the JDK encoder emits for an unpaired
     * surrogate) and the completing low surrogate charges the remaining three. The running
     * total therefore equals the UTF-8 length of the accumulated text after every fragment,
     * at O(fragment) cost with no allocation.
     */
    private static long utf8ByteLength(CharSequence text, char carry) {
        int length = text.length();
        long bytes = 0;
        int index = 0;
        if (carry != NO_PENDING_HIGH_SURROGATE
                && length > 0
                && Character.isLowSurrogate(text.charAt(0))) {
            bytes += UTF8_SURROGATE_PAIR_BYTES - UTF8_UNPAIRED_SURROGATE_BYTES;
            index = 1;
        }
        for (; index < length; index++) {
            char current = text.charAt(index);
            if (Character.isHighSurrogate(current)
                    && index + 1 < length
                    && Character.isLowSurrogate(text.charAt(index + 1))) {
                bytes += UTF8_SURROGATE_PAIR_BYTES;
                index++;
            } else {
                bytes += utf8CharLength(current);
            }
        }
        return bytes;
    }

    private static char trailingHighSurrogate(CharSequence text, char carry) {
        if (text.length() == 0) {
            return carry;
        }
        char last = text.charAt(text.length() - 1);
        return Character.isHighSurrogate(last) ? last : NO_PENDING_HIGH_SURROGATE;
    }

    private static int utf8CharLength(char value) {
        if (value < UTF8_TWO_BYTE_MIN) {
            return 1;
        }
        if (value < UTF8_THREE_BYTE_MIN) {
            return 2;
        }
        return Character.isSurrogate(value) ? UTF8_UNPAIRED_SURROGATE_BYTES : 3;
    }

    private static ControllerFault mapRemoteFault(String code) {
        return switch (code) {
            case "IDENTITY_MISMATCH", "UNAUTHORIZED" -> ControllerFault.IDENTITY_MISMATCH;
            case "STALE" -> ControllerFault.STALE_COMMAND;
            case "FUTURE" -> ControllerFault.FUTURE_COMMAND;
            case "SEQUENCE" -> ControllerFault.SEQUENCE_ERROR;
            case "OVERSIZE" -> ControllerFault.OVERSIZE_MESSAGE;
            case "QUEUE_OVERFLOW" -> ControllerFault.QUEUE_OVERFLOW;
            case "DISCONNECTED" -> ControllerFault.DISCONNECTED;
            case "INTERNAL", "TRACE_FAILURE", "RECEIPT_MISMATCH" ->
                    ControllerFault.INTERNAL_ERROR;
            default -> ControllerFault.PROTOCOL_ERROR;
        };
    }

    /**
     * One queued outbound receipt, of either kind. Both kinds bind a command and a decision, and
     * both know how to encode themselves, so the drain loop and the terminal-close path stay
     * kind-agnostic.
     */
    private sealed interface ReceiptEnvelope {
        CommandFrame command();

        ValidationResult decision();

        String encode(ControllerIdentity identity);
    }

    private record ActuationEnvelope(
            ActionFrame command,
            PhysicalSensorSnapshot worldBefore,
            ActuationReceipt receipt,
            ValidationResult decision) implements ReceiptEnvelope {
        @Override
        public String encode(ControllerIdentity identity) {
            return ControlProtocol.encodeActuationReceipt(
                    identity, command, worldBefore, receipt, decision);
        }
    }

    private record ProgramEnvelope(
            ProgramActionFrame command,
            PhysicalSensorSnapshot worldBefore,
            Optional<ProgramExecution> execution,
            ValidationResult decision) implements ReceiptEnvelope {
        @Override
        public String encode(ControllerIdentity identity) {
            return ControlProtocol.encodeProgramReceipt(
                    identity, command, worldBefore, execution, decision);
        }
    }
}
