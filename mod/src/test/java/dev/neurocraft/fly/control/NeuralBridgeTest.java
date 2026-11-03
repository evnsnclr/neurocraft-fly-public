package dev.neurocraft.fly.control;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ProgramExecution;
import dev.neurocraft.fly.body.ProgramParameters;
import dev.neurocraft.fly.body.ReadoutValues;
import dev.neurocraft.fly.entity.ActuationReceipt;
import java.net.http.WebSocket;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.function.LongSupplier;
import org.junit.jupiter.api.Test;

final class NeuralBridgeTest {
    private static final String TOKEN = "0123456789abcdef0123456789abcdef";
    private static final String NONCE = "A".repeat(ControlProtocol.HANDSHAKE_NONCE_CHARS);

    @Test
    void lifecycleSeparatesHelloWriteFromVerifiedAckAndNeverLogsAuthenticationMaterial() throws Exception {
        try (var logs = new LifecycleLogCapture("neurocraft-fly-bridge")) {
            BridgeConfig config = config();
            NeuralBridge bridge = new NeuralBridge(config);
            FakeWebSocket socket = new FakeWebSocket(1);
            try {
                bridge.onOpen(socket);
                assertEquals(List.of("socket_open", "hello_write_begin", "hello_write_complete"), logs.stages());
                assertEquals(ControllerFault.STARTING, bridge.state());
                String nonce = sentClientNonce(socket);
                String ack = helloAcknowledgement(config, nonce);
                bridge.onText(socket, ack, true);
                assertEquals(ControllerFault.ACTIVE, bridge.state());
                assertEquals("hello_ack_verified", logs.stages().getLast());
                bridge.close();
                bridge.close();
                assertEquals(List.of("socket_open", "hello_write_begin", "hello_write_complete",
                        "hello_ack_verified", "local_close_begin", "close_write_complete",
                        "executor_shutdown_requested"), logs.stages());
                String all = String.join("\n", logs.messages());
                assertFalse(all.contains(TOKEN));
                assertFalse(all.contains(nonce));
                assertFalse(all.contains((String) StrictJson.parseObject(
                        ack, ControlProtocol.MAX_MESSAGE_BYTES).get("server_auth")));
                assertTrue(logs.messages().stream().allMatch(line ->
                        line.contains("epoch=" + config.identity().sessionEpoch() + " ")));
            } finally {
                bridge.close();
            }
        }
    }

    @Test
    void firstTerminalCauseDistinguishesPeerCloseFromTransportErrorAndOmitsPeerText() {
        for (boolean peerClose : List.of(true, false)) {
            try (var logs = new LifecycleLogCapture("neurocraft-fly-bridge")) {
                NeuralBridge bridge = new NeuralBridge(config());
                FakeWebSocket socket = new FakeWebSocket(1);
                try {
                    bridge.onOpen(socket);
                    if (peerClose) bridge.onClose(socket, 1001, TOKEN);
                    else bridge.onError(socket, new IllegalStateException(TOKEN));
                    for (int i = 0; i < 10; i++) bridge.onError(socket, new RuntimeException(TOKEN));
                    assertEquals(ControllerFault.DISCONNECTED, bridge.state());
                    var terminal = logs.messages().stream().filter(line -> line.contains("fault=DISCONNECTED")).toList();
                    assertEquals(1, terminal.size());
                    assertTrue(terminal.getFirst().contains(peerClose ? "stage=peer_close" : "stage=websocket_error"));
                    assertTrue(terminal.getFirst().contains(peerClose ? "close_code=1001" : "error_class=IllegalStateException"));
                    assertFalse(String.join("\n", logs.messages()).contains(TOKEN));
                } finally {
                    bridge.close();
                }
            }
        }
    }

    @Test
    void failedHelloWriteIsIdentifiedBeforeAnyVerifiedAcknowledgement() {
        try (var logs = new LifecycleLogCapture("neurocraft-fly-bridge")) {
            NeuralBridge bridge = new NeuralBridge(config());
            FakeWebSocket socket = new FakeWebSocket(1);
            socket.throwOnNextText = true;
            try {
                bridge.onOpen(socket);
                assertEquals(ControllerFault.DISCONNECTED, bridge.state());
                assertEquals(List.of("socket_open", "hello_write_begin", "hello_setup_or_write_failed"), logs.stages());
                assertTrue(logs.messages().getLast().contains("error_class=IllegalStateException"));
                assertFalse(String.join("\n", logs.messages()).contains("synthetic synchronous send failure"));
            } finally {
                bridge.close();
            }
        }
    }

    @Test
    void helloCarriesNoTokenAndBindsAFreshNonce() throws Exception {
        BridgeConfig config = config();
        NeuralBridge bridge = new NeuralBridge(config);
        FakeWebSocket socket = new FakeWebSocket(1);
        try {
            bridge.onOpen(socket);

            String hello = socket.textFrames().get(0);
            assertFalse(hello.contains(TOKEN));
            assertFalse(hello.contains("token"));
            Map<String, Object> sent =
                    StrictJson.parseObject(hello, ControlProtocol.MAX_MESSAGE_BYTES);
            assertEquals(
                    Set.of("type", "protocol", "identity", "role", "client_nonce", "auth"),
                    sent.keySet());
            String nonce = (String) sent.get("client_nonce");
            assertEquals(ControlProtocol.HANDSHAKE_NONCE_CHARS, nonce.length());
            assertEquals(
                    ControlProtocol.handshakeAuth(
                            TOKEN, "hello", config.identity(), "minecraft_server", nonce),
                    sent.get("auth"));

            // A second connection must not reuse the nonce of the first.
            NeuralBridge other = new NeuralBridge(config);
            FakeWebSocket otherSocket = new FakeWebSocket(1);
            try {
                other.onOpen(otherSocket);
                assertFalse(nonce.equals(sentClientNonce(otherSocket)));
            } finally {
                other.close();
            }
        } finally {
            bridge.close();
        }
    }

    @Test
    void unprovenHelloAckLatchesIdentityMismatch() throws Exception {
        for (String unproven : List.of("wrong-token", "wrong-nonce", "absent-proof")) {
            BridgeConfig config = config();
            NeuralBridge bridge = new NeuralBridge(config);
            FakeWebSocket socket = new FakeWebSocket(1);
            try {
                bridge.onOpen(socket);
                String nonce = sentClientNonce(socket);
                String acknowledgement = switch (unproven) {
                    // Right nonce and identity, but the peer does not hold the run token.
                    case "wrong-token" -> helloAcknowledgement(
                            config, nonce, "f".repeat(43));
                    // Right token, but replayed against a nonce this session never sent.
                    case "wrong-nonce" -> helloAcknowledgement(config, NONCE);
                    // Structurally valid digest that proves nothing at all.
                    default -> helloAcknowledgement(config, nonce)
                            .replaceAll("\"server_auth\":\"[0-9a-f]{64}\"",
                                    "\"server_auth\":\"" + "0".repeat(64) + "\"");
                };

                bridge.onText(socket, acknowledgement, true);

                assertEquals(ControllerFault.IDENTITY_MISMATCH, bridge.state(), unproven);
            } finally {
                bridge.close();
            }
        }
    }

    @Test
    void actionBeforeVerifiedAckLatchesProtocolError() throws Exception {
        BridgeConfig config = config();
        NeuralBridge bridge = new NeuralBridge(config);
        FakeWebSocket socket = new FakeWebSocket(1);
        try {
            bridge.onOpen(socket);
            assertEquals(ControllerFault.STARTING, bridge.state());

            bridge.onText(socket, actionFrame(config), true);

            assertEquals(ControllerFault.PROTOCOL_ERROR, bridge.state());
            assertTrue(bridge.pollAction().isEmpty());
        } finally {
            bridge.close();
        }
    }

    @Test
    void refusesSecondControlUntilFirstControlIsAcknowledged() throws Exception {
        OpenBridge open = openBridge(2);
        try {
            assertTrue(open.bridge()
                    .requestControl(100, ControlOperation.PAUSE, null, null)
                    .isPresent());
            assertTrue(open.socket().awaitExpectedText());
            assertTrue(open.bridge()
                    .requestControl(101, ControlOperation.RESUME, null, null)
                    .isEmpty());
            assertEquals(ControllerFault.ACTIVE, open.bridge().state());
        } finally {
            open.bridge().close();
        }
    }

    @Test
    void pauseOrRecordAckCannotPoisonBoundResetAck() throws Exception {
        for (ControlOperation ordinaryOperation :
                List.of(ControlOperation.PAUSE, ControlOperation.RECORD_START)) {
            OpenBridge open = openBridge(3);
            try {
                String contentId = ordinaryOperation == ControlOperation.RECORD_START
                        ? "trial-001"
                        : null;
                ControlState ordinaryState = ordinaryOperation == ControlOperation.PAUSE
                        ? ControlState.PAUSED
                        : ControlState.RECORDING;
                long ordinarySequence = open.bridge()
                        .requestControl(100, ordinaryOperation, contentId, null)
                        .orElseThrow();
                assertTrue(open.socket().awaitTextFrameCount(2), ordinaryOperation.toString());
                open.bridge()
                        .onText(
                                open.socket(),
                                controlAcknowledgement(
                                        open.config(),
                                        ordinarySequence,
                                        100,
                                        ordinaryOperation,
                                        contentId,
                                        ordinaryState,
                                        101),
                                true);

                long resetSequence = open.bridge()
                        .requestControl(102, ControlOperation.RESET, null, null)
                        .orElseThrow();
                assertTrue(open.socket().awaitExpectedText(), ordinaryOperation.toString());
                open.bridge()
                        .onText(
                                open.socket(),
                                controlAcknowledgement(
                                        open.config(),
                                        resetSequence,
                                        102,
                                        ControlOperation.RESET,
                                        null,
                                        ControlState.RECORDING,
                                        104),
                                true);

                ControlAckFrame reset = open.bridge()
                        .pollControlAck(resetSequence, ControlOperation.RESET)
                        .orElseThrow();
                assertEquals(resetSequence, reset.controlSequence());
                assertEquals(ControlOperation.RESET, reset.operation());
                ControlAckFrame surfacedOrdinary =
                        open.bridge().pollControlAck().orElseThrow();
                assertEquals(ordinarySequence, surfacedOrdinary.controlSequence());
                assertEquals(ordinaryOperation, surfacedOrdinary.operation());
                assertTrue(open.bridge().pollControlAck().isEmpty());
                assertEquals(ControllerFault.ACTIVE, open.bridge().state());
            } finally {
                open.bridge().close();
            }
        }
    }

    @Test
    void rejectedReceiptIsSentExactlyBeforeTerminalClose() throws Exception {
        for (TerminalCase terminal : List.of(
                new TerminalCase(ControllerFault.STALE_COMMAND, "STALE", 104),
                new TerminalCase(ControllerFault.FUTURE_COMMAND, "FUTURE", 100),
                new TerminalCase(ControllerFault.SEQUENCE_ERROR, "SEQUENCE", 102))) {
            OpenBridge open = openBridge(2);
            try {
                ActionFrame action = action(open.config().identity());
                PhysicalSensorSnapshot worldBefore = snapshot(
                        open.config().identity().entityId(), terminal.appliedTick());
                ActuationReceipt neutralReceipt = new ActuationReceipt(
                        action.rawAction(),
                        PrimitiveAction.NEUTRAL,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0);
                ValidationResult decision =
                        ValidationResult.rejected(action, terminal.fault());
                String expected = ControlProtocol.encodeActuationReceipt(
                        open.config().identity(), action, worldBefore, neutralReceipt, decision);

                open.bridge()
                        .offerActuationReceipt(action, worldBefore, neutralReceipt, decision);

                assertTrue(open.socket().awaitExpectedText(), terminal.fault().toString());
                assertTrue(open.socket().awaitClose(), terminal.fault().toString());
                assertEquals(
                        StrictJson.parseObject(expected, ControlProtocol.MAX_MESSAGE_BYTES),
                        StrictJson.parseObject(
                                open.socket().textFrames().get(1),
                                ControlProtocol.MAX_MESSAGE_BYTES));
                Map<String, Object> sent = StrictJson.parseObject(
                        open.socket().textFrames().get(1), ControlProtocol.MAX_MESSAGE_BYTES);
                assertEquals("REJECTED", sent.get("validation"));
                assertEquals(terminal.protocolFault(), sent.get("fault_code"));
                assertEquals(
                        terminal.appliedTick(),
                        ((java.math.BigDecimal) sent.get("applied_tick")).longValueExact());
                assertEquals(List.of("text", "text", "close"), open.socket().events());
                assertEquals(
                        NeuralBridgeTest.FakeWebSocket.TERMINAL_CLOSE_STATUS,
                        open.socket().closeStatus());
                assertEquals(terminal.fault(), open.bridge().state());
            } finally {
                open.bridge().close();
            }
        }
    }

    @Test
    void synchronousTerminalReceiptSendFailureFailsClosedAndReleasesDeferredState()
            throws Exception {
        OpenBridge open = openBridge(1);
        try {
            ActionFrame action = action(open.config().identity());
            PhysicalSensorSnapshot worldBefore =
                    snapshot(open.config().identity().entityId(), 104);
            ActuationReceipt neutralReceipt = new ActuationReceipt(
                    action.rawAction(),
                    PrimitiveAction.NEUTRAL,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0);
            open.socket().throwSynchronouslyOnNextText();

            open.bridge().offerActuationReceipt(
                    action,
                    worldBefore,
                    neutralReceipt,
                    ValidationResult.rejected(action, ControllerFault.STALE_COMMAND));

            assertTrue(open.socket().awaitSynchronousThrow());
            awaitBridgeState(open.bridge(), ControllerFault.DISCONNECTED);
            awaitTerminalReceiptClear(open.bridge());
            assertTrue(open.bridge()
                    .requestControl(104, ControlOperation.PAUSE, null, null)
                    .isEmpty());
            assertEquals(List.of("text"), open.socket().events());
        } finally {
            open.bridge().close();
        }
    }

    @Test
    void fragmentedMessageExactlyAtLimitAccepted() throws Exception {
        OpenBridge open = openBridge(1);
        try {
            String fragment = twoByteFragment(4);
            assertEquals(
                    ControlProtocol.MAX_MESSAGE_BYTES,
                    4L * fragment.getBytes(StandardCharsets.UTF_8).length);
            for (int index = 0; index < 3; index++) {
                open.bridge().onText(open.socket(), fragment, false);
            }
            assertEquals(
                    3L * fragment.getBytes(StandardCharsets.UTF_8).length,
                    open.bridge().inboundUtf8Bytes());
            assertEquals(ControllerFault.ACTIVE, open.bridge().state());

            open.bridge().onText(open.socket(), fragment, true);

            assertEquals(ControllerFault.PROTOCOL_ERROR, open.bridge().state());
            assertEquals(0L, open.bridge().inboundUtf8Bytes());
        } finally {
            open.bridge().close();
        }
    }

    @Test
    void fragmentedMessageOverLimitLatchesOversizeAtOffendingFragment() throws Exception {
        OpenBridge open = openBridge(1);
        try {
            String fragment = twoByteFragment(4);
            for (int index = 0; index < 4; index++) {
                open.bridge().onText(open.socket(), fragment, false);
                assertEquals(ControllerFault.ACTIVE, open.bridge().state());
            }
            assertEquals(
                    ControlProtocol.MAX_MESSAGE_BYTES, open.bridge().inboundUtf8Bytes());

            open.bridge().onText(open.socket(), "\u00e9", false);

            assertEquals(ControllerFault.OVERSIZE_MESSAGE, open.bridge().state());
            assertEquals(0L, open.bridge().inboundUtf8Bytes());
        } finally {
            open.bridge().close();
        }
    }

    @Test
    void surrogatePairSplitAcrossFragmentsCountsFourBytes() throws Exception {
        OpenBridge open = openBridge(1);
        try {
            open.bridge().onText(open.socket(), "\ud83d", false);
            assertEquals(
                    "\ud83d".getBytes(StandardCharsets.UTF_8).length,
                    open.bridge().inboundUtf8Bytes());
            assertEquals('\ud83d', open.bridge().pendingHighSurrogate());

            open.bridge().onText(open.socket(), "\ude00", false);

            assertEquals(
                    "\ud83d\ude00".getBytes(StandardCharsets.UTF_8).length,
                    open.bridge().inboundUtf8Bytes());
            assertEquals(4L, open.bridge().inboundUtf8Bytes());
            assertEquals('\0', open.bridge().pendingHighSurrogate());
            assertEquals(ControllerFault.ACTIVE, open.bridge().state());
        } finally {
            open.bridge().close();
        }
    }

    @Test
    void oversizeResetClearsRunningCounter() throws Exception {
        OpenBridge open = openBridge(1);
        try {
            open.bridge().onText(open.socket(), twoByteFragment(1), false);
            assertEquals(
                    ControlProtocol.MAX_MESSAGE_BYTES, open.bridge().inboundUtf8Bytes());

            open.bridge().onText(open.socket(), "\ud83d", false);

            assertEquals(ControllerFault.OVERSIZE_MESSAGE, open.bridge().state());
            assertEquals(0L, open.bridge().inboundUtf8Bytes());
            assertEquals('\0', open.bridge().pendingHighSurrogate());

            open.bridge().onText(open.socket(), "\ude00", true);

            assertEquals(0L, open.bridge().inboundUtf8Bytes());
            assertEquals('\0', open.bridge().pendingHighSurrogate());
            assertEquals(ControllerFault.OVERSIZE_MESSAGE, open.bridge().state());
        } finally {
            open.bridge().close();
        }
    }

    @Test
    void aFirstActionIsMeasuredAgainstTheObservationItIsBoundTo() throws Exception {
        OpenBridge open = openBridge(2);
        try {
            int budget = open.bridge().firstActionLatencySamplesRemaining();
            assertTrue(budget > 0);
            open.bridge().offerObservation(
                    snapshot(open.config().identity().entityId(), 100));
            assertTrue(open.socket().awaitExpectedText());

            open.bridge().onText(open.socket(), actionFrame(open.config()), true);

            assertEquals(budget - 1, open.bridge().firstActionLatencySamplesRemaining());
            assertTrue(open.bridge().pollAction().isPresent());
            assertEquals(ControllerFault.ACTIVE, open.bridge().state());
        } finally {
            open.bridge().close();
        }
    }

    @Test
    void anActionBoundToAnUnsentObservationConsumesNoSample() throws Exception {
        OpenBridge open = openBridge(2);
        try {
            int budget = open.bridge().firstActionLatencySamplesRemaining();
            open.bridge().offerObservation(
                    snapshot(open.config().identity().entityId(), 100));
            assertTrue(open.socket().awaitExpectedText());

            open.bridge().onText(open.socket(), actionFrame(open.config(), 3L), true);

            assertEquals(budget, open.bridge().firstActionLatencySamplesRemaining());
            assertTrue(open.bridge().pollAction().isPresent());
        } finally {
            open.bridge().close();
        }
    }

    private static String twoByteFragment(int fragmentsPerLimit) {
        return "\u00e9".repeat(ControlProtocol.MAX_MESSAGE_BYTES / (2 * fragmentsPerLimit));
    }

    /**
     * Waits, bounded, for a bridge counter to reach {@code expected}. Receipt counters are
     * incremented when the send future completes, which is after the fake socket has counted the
     * frame, so a test that has awaited the frame must still await the counter.
     */
    private static void awaitCount(LongSupplier counter, long expected, String what)
            throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (counter.getAsLong() != expected) {
            if (System.nanoTime() > deadline) {
                fail(what + " did not reach " + expected + "; it is " + counter.getAsLong());
            }
            Thread.sleep(1L);
        }
    }

    private static OpenBridge openBridge(int expectedTextFrames) throws Exception {
        BridgeConfig config = config();
        NeuralBridge bridge = new NeuralBridge(config);
        FakeWebSocket socket = new FakeWebSocket(expectedTextFrames);
        bridge.onOpen(socket);
        bridge.onText(socket, helloAcknowledgement(config, sentClientNonce(socket)), true);
        assertEquals(ControllerFault.ACTIVE, bridge.state());
        return new OpenBridge(config, bridge, socket);
    }

    private static void awaitBridgeState(NeuralBridge bridge, ControllerFault expected) {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
        while (bridge.state() != expected && System.nanoTime() < deadline) {
            Thread.onSpinWait();
        }
        assertEquals(expected, bridge.state());
    }

    private static void awaitTerminalReceiptClear(NeuralBridge bridge) {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
        while (bridge.pendingTerminalFault().isPresent() && System.nanoTime() < deadline) {
            Thread.onSpinWait();
        }
        assertTrue(bridge.pendingTerminalFault().isEmpty());
    }

    private static String helloAcknowledgement(BridgeConfig config, String clientNonce)
            throws Exception {
        return helloAcknowledgement(config, clientNonce, TOKEN);
    }

    /**
     * Build the service's reply, proving possession of {@code provingToken} over
     * {@code clientNonce}. Passing a token the bridge does not hold produces an unproven
     * acknowledgement that is structurally valid but cryptographically wrong.
     */
    private static String helloAcknowledgement(
            BridgeConfig config, String clientNonce, String provingToken) throws Exception {
        Map<String, Object> acknowledgement = new LinkedHashMap<>(StrictJson.parseObject(
                ControlProtocol.encodeHello(config.identity(), TOKEN, NONCE),
                ControlProtocol.MAX_MESSAGE_BYTES));
        acknowledgement.put("type", "hello_ack");
        acknowledgement.put("role", "neural_service");
        acknowledgement.remove("client_nonce");
        acknowledgement.remove("auth");
        acknowledgement.put(
                "server_auth",
                ControlProtocol.handshakeAuth(
                        provingToken,
                        "hello_ack",
                        config.identity(),
                        "neural_service",
                        clientNonce));
        acknowledgement.put("accepted", true);
        return StrictJson.encode(acknowledgement, ControlProtocol.MAX_MESSAGE_BYTES);
    }

    /** Read the {@code client_nonce} the bridge minted for its own hello frame. */
    private static String sentClientNonce(FakeWebSocket socket) throws Exception {
        Map<String, Object> hello = StrictJson.parseObject(
                socket.textFrames().get(0), ControlProtocol.MAX_MESSAGE_BYTES);
        return (String) hello.get("client_nonce");
    }

    /**
     * The declared action budget is a wire contract, enforced where frames arrive.
     *
     * <p>The frame model stays permissive so frames recorded under the one-tick budget still
     * decode, and {@code CommandValidator}'s rules are bounds against the authoritative clock
     * rather than a second copy of the producer's convention. This is the check that says the
     * service on the other end of the socket obeyed the budget it declared: a frame targeting
     * anything but {@code observation_tick + ACTION_TARGET_OFFSET_TICKS} fails the session
     * closed before it can reach the server thread.
     */
    @Test
    void anActionOffTheDeclaredBudgetIsRefusedAtTheWireBoundary() throws Exception {
        OpenBridge open = openBridge(2);
        try {
            open.bridge().onText(open.socket(), actionFrame(open.config(), 0L, 1L), true);

            assertEquals(ControllerFault.PROTOCOL_ERROR, open.bridge().state());
            assertTrue(
                    open.bridge().pollAction().isEmpty(),
                    "an off-budget frame must never reach the server-thread window");
        } finally {
            open.bridge().close();
        }
    }

    /** The same wire rule for the other driving kind: it is the envelope, not the payload. */
    @Test
    void aProgramActionOffTheDeclaredBudgetIsRefusedAtTheWireBoundary() throws Exception {
        OpenBridge open = openBridge(2);
        try {
            open.bridge()
                    .onText(
                            open.socket(),
                            programActionJson(open.config()).replace("\"target_tick\":102", "\"target_tick\":101"),
                            true);

            assertEquals(ControllerFault.PROTOCOL_ERROR, open.bridge().state());
            assertTrue(
                    open.bridge().pollAction().isEmpty(),
                    "an off-budget program frame must never reach the server-thread window");
        } finally {
            open.bridge().close();
        }
    }

    private static String actionFrame(BridgeConfig config) throws Exception {
        return actionFrame(config, 0L);
    }

    private static String actionFrame(BridgeConfig config, long observationSequence)
            throws Exception {
        return actionFrame(
                config, observationSequence, CommandValidator.ACTION_TARGET_OFFSET_TICKS);
    }

    private static String actionFrame(
            BridgeConfig config, long observationSequence, long targetOffsetTicks)
            throws Exception {
        Map<String, Object> motors = new LinkedHashMap<>();
        motors.put("forward_thrust", 0.5);
        motors.put("lateral_thrust", 0.0);
        motors.put("vertical_thrust", 0.0);
        motors.put("yaw_torque", 0.0);
        motors.put("pitch_torque", 0.0);
        Map<String, Object> message = new LinkedHashMap<>(StrictJson.parseObject(
                ControlProtocol.encodeHello(config.identity(), TOKEN, NONCE),
                ControlProtocol.MAX_MESSAGE_BYTES));
        message.put("type", "action");
        message.remove("role");
        message.remove("client_nonce");
        message.remove("auth");
        message.put("command_seq", 0L);
        message.put("observation_seq", observationSequence);
        message.put("observation_tick", 100L);
        message.put("target_tick", 100L + targetOffsetTicks);
        message.put("valid_through_tick", 102L);
        message.put("neural_step_start", 0L);
        message.put("neural_step_end", 5L);
        message.put("state_digest", "6".repeat(64));
        message.put("motors", motors);
        message.put("neutral", false);
        return StrictJson.encode(message, ControlProtocol.MAX_MESSAGE_BYTES);
    }

    private static String controlAcknowledgement(
            BridgeConfig config,
            long controlSequence,
            long issuedTick,
            ControlOperation operation,
            String contentId,
            ControlState state,
            long effectiveTick)
            throws Exception {
        Map<String, Object> acknowledgement = new LinkedHashMap<>(StrictJson.parseObject(
                ControlProtocol.encodeHello(config.identity(), TOKEN, NONCE),
                ControlProtocol.MAX_MESSAGE_BYTES));
        acknowledgement.put("type", "control_ack");
        acknowledgement.remove("role");
        acknowledgement.remove("client_nonce");
        acknowledgement.remove("auth");
        acknowledgement.put("control_seq", controlSequence);
        acknowledgement.put("issued_tick", issuedTick);
        acknowledgement.put("operation", operation.toString());
        acknowledgement.put("content_id", contentId);
        acknowledgement.put("content_hash", null);
        acknowledgement.put("accepted", true);
        acknowledgement.put("state", state.toString());
        acknowledgement.put("effective_tick", effectiveTick);
        acknowledgement.put("fault_code", null);
        return StrictJson.encode(acknowledgement, ControlProtocol.MAX_MESSAGE_BYTES);
    }

    /**
     * A program frame takes the same inbound path an action frame takes, and its receipt takes the
     * same outbound path -- including the terminal one: a REJECTED program receipt is sent before
     * the close frame, exactly as a rejected actuation receipt is, so a session that fails closed on
     * the program lane still delivers its authoritative last word.
     */
    @Test
    void programReceiptIsSentAndRejectedProgramReceiptIsTerminal() throws Exception {
        OpenBridge applied = openBridge(2);
        try {
            ProgramActionFrame frame = programAction(applied.config().identity());
            applied.bridge().onText(applied.socket(), programActionJson(applied.config()), true);
            assertEquals(
                    Optional.of(frame),
                    applied.bridge().pollAction(),
                    "a program frame did not reach the server-thread window");

            PhysicalSensorSnapshot worldBefore =
                    snapshot(applied.config().identity().entityId(), 102);
            Optional<ProgramExecution> execution = Optional.of(new ProgramExecution(
                    BodyProgramId.CRUISE,
                    frame.modulation(),
                    frame.parameters(),
                    PrimitiveAction.clamped(0.35, 0.0, 0.1, -0.05, 0.0),
                    Optional.empty()));
            applied.bridge()
                    .offerProgramReceipt(
                            frame,
                            worldBefore,
                            execution,
                            ValidationResult.accepted(frame));

            assertTrue(applied.socket().awaitExpectedText());
            Map<String, Object> sent = StrictJson.parseObject(
                    applied.socket().textFrames().get(1), ControlProtocol.MAX_MESSAGE_BYTES);
            assertEquals("program_receipt", sent.get("type"));
            assertEquals("PROGRAM_APPLIED", sent.get("validation"));
            // The counter is incremented in the send future's completion callback, which runs on
            // the drain thread after the fake socket has already released awaitExpectedText();
            // a loaded runner can read it first (CI run 33799330951), so wait for the counter's
            // own completion rather than for the frame.
            awaitCount(applied.bridge()::sentProgramReceiptCount, 1, "program receipt counter");
            assertEquals(
                    0,
                    applied.bridge().sentActuationReceiptCount(),
                    "a program receipt must not be counted as an actuation receipt");
            assertEquals(ControllerFault.ACTIVE, applied.bridge().state());
        } finally {
            applied.bridge().close();
        }

        OpenBridge rejected = openBridge(2);
        try {
            ProgramActionFrame frame = programAction(rejected.config().identity());
            PhysicalSensorSnapshot worldBefore =
                    snapshot(rejected.config().identity().entityId(), 104);
            ValidationResult decision =
                    ValidationResult.rejected(frame, ControllerFault.STALE_COMMAND);

            rejected.bridge()
                    .offerProgramReceipt(frame, worldBefore, Optional.empty(), decision);

            assertTrue(rejected.socket().awaitExpectedText());
            assertTrue(rejected.socket().awaitClose());
            Map<String, Object> sent = StrictJson.parseObject(
                    rejected.socket().textFrames().get(1), ControlProtocol.MAX_MESSAGE_BYTES);
            assertEquals("REJECTED", sent.get("validation"));
            assertEquals("STALE", sent.get("fault_code"));
            assertEquals(List.of("text", "text", "close"), rejected.socket().events());
            assertEquals(
                    NeuralBridgeTest.FakeWebSocket.TERMINAL_CLOSE_STATUS,
                    rejected.socket().closeStatus());
            assertEquals(ControllerFault.STALE_COMMAND, rejected.bridge().state());
        } finally {
            rejected.bridge().close();
        }
    }

    private static ProgramActionFrame programAction(ControllerIdentity identity) {
        return new ProgramActionFrame(
                identity,
                0,
                0,
                100,
                102,
                102,
                0,
                5,
                "6".repeat(64),
                BodyProgramId.CRUISE,
                new Modulation(0.7, 1.0, 1.0, 0.0),
                new ProgramParameters(-0.25),
                new ReadoutValues(0.0, -0.25, 0.61),
                "7".repeat(64));
    }

    private static String programActionJson(BridgeConfig config) {
        ControllerIdentity identity = config.identity();
        return "{"
                + "\"type\":\"program_action\","
                + "\"protocol\":\"neurocraft.protocol.v2\","
                + "\"identity\":" + identityJson(identity) + ","
                + "\"command_seq\":0,"
                + "\"observation_seq\":0,"
                + "\"observation_tick\":100,"
                + "\"target_tick\":102,"
                + "\"valid_through_tick\":102,"
                + "\"neural_step_start\":0,"
                + "\"neural_step_end\":5,"
                + "\"state_digest\":\"" + "6".repeat(64) + "\","
                + "\"program\":\"CRUISE\","
                + "\"modulation\":{\"speed\":0.7,\"turn_rate\":1.0,\"amplitude\":1.0,"
                + "\"urgency\":0.0},"
                + "\"parameters\":{\"turn_bias\":-0.25},"
                + "\"readout\":{\"ttmn_jump\":0.0,\"dna01_turn\":-0.25,\"wing_common\":0.61},"
                + "\"policy_hash\":\"" + "7".repeat(64) + "\"}";
    }

    private static String identityJson(ControllerIdentity identity) {
        return "{"
                + "\"run_id\":\"" + identity.runId() + "\","
                + "\"session_epoch\":" + identity.sessionEpoch() + ","
                + "\"entity_id\":\"" + identity.entityId() + "\","
                + "\"graph_hash\":\"" + identity.graphHash() + "\","
                + "\"adapter_hash\":\"" + identity.adapterHash() + "\","
                + "\"dynamics_hash\":\"" + identity.dynamicsHash() + "\","
                + "\"decoder_hash\":\"" + identity.decoderHash() + "\","
                + "\"observation_schema_hash\":\"" + identity.observationSchemaHash() + "\"}";
    }

    private static BridgeConfig config() {
        Properties properties = new Properties();
        properties.setProperty("neurocraft.fly.runId", "run-1");
        properties.setProperty("neurocraft.fly.sessionEpoch", "7");
        properties.setProperty(
                "neurocraft.fly.entityId", "00000000-0000-0000-0000-000000000001");
        properties.setProperty("neurocraft.fly.graphHash", "1".repeat(64));
        properties.setProperty("neurocraft.fly.adapterHash", "2".repeat(64));
        properties.setProperty("neurocraft.fly.dynamicsHash", "3".repeat(64));
        properties.setProperty("neurocraft.fly.decoderHash", "4".repeat(64));
        properties.setProperty("neurocraft.fly.observationSchemaHash", "5".repeat(64));
        return BridgeConfig.from(Map.of("NEUROCRAFT_RUN_TOKEN", TOKEN), properties);
    }

    private static ActionFrame action(ControllerIdentity identity) {
        return new ActionFrame(
                identity,
                0,
                0,
                100,
                102,
                102,
                0,
                5,
                "6".repeat(64),
                new RawPrimitiveAction(0.5, 0, 0, 0, 0),
                false);
    }

    private static PhysicalSensorSnapshot snapshot(String entityId, long gameTick) {
        return new PhysicalSensorSnapshot(
                gameTick,
                entityId,
                1,
                64,
                -2,
                0.005,
                0,
                -0.01,
                Math.toDegrees(0.25),
                Math.nextUp(Math.toDegrees(-0.1)),
                false,
                false,
                false,
                false,
                false,
                12,
                10,
                8,
                8,
                6,
                5,
                8,
                2.5,
                3);
    }

    private record TerminalCase(
            ControllerFault fault, String protocolFault, long appliedTick) {}

    private record OpenBridge(
            BridgeConfig config, NeuralBridge bridge, FakeWebSocket socket) {}

    private static final class FakeWebSocket implements WebSocket {
        private static final int TERMINAL_CLOSE_STATUS = 1008;

        private final CountDownLatch expectedText;
        private final CountDownLatch closed = new CountDownLatch(1);
        private final CountDownLatch synchronousThrow = new CountDownLatch(1);
        private final List<String> textFrames = new ArrayList<>();
        private final List<String> events = new ArrayList<>();
        private int closeStatus = -1;
        private boolean throwOnNextText;

        private FakeWebSocket(int expectedTextFrames) {
            expectedText = new CountDownLatch(expectedTextFrames);
        }

        @Override
        public CompletableFuture<WebSocket> sendText(CharSequence data, boolean last) {
            synchronized (this) {
                if (throwOnNextText) {
                    throwOnNextText = false;
                    synchronousThrow.countDown();
                    throw new IllegalStateException("synthetic synchronous send failure");
                }
                textFrames.add(data.toString());
                events.add("text");
                notifyAll();
            }
            expectedText.countDown();
            return CompletableFuture.completedFuture(this);
        }

        @Override
        public CompletableFuture<WebSocket> sendBinary(ByteBuffer data, boolean last) {
            return CompletableFuture.completedFuture(this);
        }

        @Override
        public CompletableFuture<WebSocket> sendPing(ByteBuffer message) {
            return CompletableFuture.completedFuture(this);
        }

        @Override
        public CompletableFuture<WebSocket> sendPong(ByteBuffer message) {
            return CompletableFuture.completedFuture(this);
        }

        @Override
        public CompletableFuture<WebSocket> sendClose(int statusCode, String reason) {
            synchronized (this) {
                closeStatus = statusCode;
                events.add("close");
            }
            closed.countDown();
            return CompletableFuture.completedFuture(this);
        }

        @Override
        public void request(long n) {}

        @Override
        public String getSubprotocol() {
            return ControlProtocol.WEBSOCKET_SUBPROTOCOL;
        }

        @Override
        public boolean isOutputClosed() {
            return false;
        }

        @Override
        public boolean isInputClosed() {
            return false;
        }

        @Override
        public void abort() {}

        private boolean awaitExpectedText() throws InterruptedException {
            return expectedText.await(2, TimeUnit.SECONDS);
        }

        private synchronized boolean awaitTextFrameCount(int expected) throws InterruptedException {
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
            long remaining = deadline - System.nanoTime();
            while (textFrames.size() < expected && remaining > 0) {
                TimeUnit.NANOSECONDS.timedWait(this, remaining);
                remaining = deadline - System.nanoTime();
            }
            return textFrames.size() >= expected;
        }

        private boolean awaitClose() throws InterruptedException {
            return closed.await(2, TimeUnit.SECONDS);
        }

        private boolean awaitSynchronousThrow() throws InterruptedException {
            return synchronousThrow.await(2, TimeUnit.SECONDS);
        }

        private synchronized void throwSynchronouslyOnNextText() {
            throwOnNextText = true;
        }

        private synchronized List<String> textFrames() {
            return List.copyOf(textFrames);
        }

        private synchronized List<String> events() {
            return List.copyOf(events);
        }

        private synchronized int closeStatus() {
            return closeStatus;
        }
    }
}
