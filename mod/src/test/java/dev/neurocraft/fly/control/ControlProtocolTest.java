package dev.neurocraft.fly.control;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ProgramExecution;
import dev.neurocraft.fly.body.ProgramParameters;
import dev.neurocraft.fly.body.ProgramRefusalReason;
import dev.neurocraft.fly.body.ReadoutValues;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

final class ControlProtocolTest {
    private static final String TOKEN = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
    private static final Pattern NEGATIVE_ZERO_TOKEN =
            Pattern.compile("[:\\[,]-0(\\.0*)?(?=[,\\]}])");
    private static final ControllerIdentity IDENTITY = new ControllerIdentity(
            "run-1",
            7,
            "fly-01",
            "1".repeat(64),
            "2".repeat(64),
            "3".repeat(64),
            "4".repeat(64),
            "5".repeat(64));

    @Test
    void rejectsUnknownDuplicateMissingWrongTypeAndLegacyFlatFields() {
        String action = actionJson("0.25");
        assertThrows(ProtocolException.class, () -> ControlProtocol.decodeInbound(
                action.replaceFirst("\\}$", ",\"steer_to_target\":0}")));
        assertThrows(ProtocolException.class, () -> ControlProtocol.decodeInbound(
                action.replace(
                        "\"run_id\":\"run-1\"",
                        "\"run_id\":\"run-1\",\"run_id\":\"run-2\"")));
        assertThrows(ProtocolException.class, () -> ControlProtocol.decodeInbound(
                action.replace(",\"state_digest\":\"" + "6".repeat(64) + "\"", "")));
        assertThrows(ProtocolException.class, () -> ControlProtocol.decodeInbound(
                action.replace("\"target_tick\":101", "\"target_tick\":\"101\"")));
        assertThrows(ProtocolException.class, () -> ControlProtocol.decodeInbound(
                action.replaceFirst(
                        "\"identity\":\\{",
                        "\"run_id\":\"run-1\",\"identity\":{")));
    }

    @Test
    void rejectsNonJsonNonFiniteOutOfRangeAndOversizeInput() {
        assertThrows(ProtocolException.class, () -> ControlProtocol.decodeInbound(
                actionJson("NaN")));
        assertThrows(ProtocolException.class, () -> ControlProtocol.decodeInbound(
                actionJson("Infinity")));
        assertThrows(ProtocolException.class, () -> ControlProtocol.decodeInbound(
                actionJson("1.0001")));
        assertThrows(ProtocolException.class, () -> ControlProtocol.decodeInbound("{broken"));
        assertThrows(ProtocolException.class, () -> ControlProtocol.decodeInbound(
                "{\"type\":\"" + "x".repeat(ControlProtocol.MAX_MESSAGE_BYTES) + "\"}"));
    }

    @Test
    void strictParserHandlesJsonNullInsideBoundedArrays() throws Exception {
        Map<String, Object> parsed = StrictJson.parseObject(
                "{\"values\":[null]}", ControlProtocol.MAX_MESSAGE_BYTES);

        assertEquals(1, ((List<?>) parsed.get("values")).size());
        assertNull(((List<?>) parsed.get("values")).getFirst());
    }

    @Test
    void decodedActionRetainsCanonicalPrimitiveMotorValues() throws Exception {
        ActionFrame decoded = assertInstanceOf(
                ActionFrame.class, ControlProtocol.decodeInbound(actionJson("0.25")));

        assertEquals(IDENTITY, decoded.identity());
        assertEquals(0.25, decoded.rawAction().forwardThrust());
        assertEquals(101, decoded.targetTick());
        assertFalse(decoded.neutral());
    }

    @Test
    void neutralActionMayKeepTheNeuralCursorStillButNonNeutralMayNot() throws Exception {
        String neutral = actionJson("0")
                .replace("\"lateral_thrust\":-0.1", "\"lateral_thrust\":0")
                .replace("\"yaw_torque\":0.05", "\"yaw_torque\":0")
                .replace("\"neural_step_end\":5", "\"neural_step_end\":0")
                .replace("\"neutral\":false", "\"neutral\":true");

        ActionFrame decoded = assertInstanceOf(
                ActionFrame.class, ControlProtocol.decodeInbound(neutral));

        assertTrue(decoded.neutral());
        assertEquals(decoded.neuralStepStart(), decoded.neuralStepEnd());
        assertThrows(
                ProtocolException.class,
                () -> ControlProtocol.decodeInbound(neutral.replace(
                        "\"neutral\":true", "\"neutral\":false")));
    }

    @Test
    void emitsEveryFrozenNonMotorControlOperationAndRejectsInvalidContentBindings() {
        Set<ControlOperation> simple = Set.of(
                ControlOperation.PAUSE,
                ControlOperation.RESUME,
                ControlOperation.SINGLE_STEP,
                ControlOperation.RESET);
        long sequence = 0;
        for (ControlOperation operation : simple) {
            String frame = ControlProtocol.encodeControlCommand(new ControlCommandFrame(
                    IDENTITY, sequence, 100 + sequence, operation, null, null));
            assertTrue(frame.contains("\"operation\":\"" + operation + "\""));
            sequence++;
        }
        for (ControlOperation operation : Set.of(
                ControlOperation.RECORD_START,
                ControlOperation.RECORD_STOP,
                ControlOperation.REPLAY_STOP)) {
            String frame = ControlProtocol.encodeControlCommand(new ControlCommandFrame(
                    IDENTITY, sequence, 100 + sequence, operation, "trial-001", null));
            assertTrue(frame.contains("\"content_id\":\"trial-001\""));
            sequence++;
        }
        String replay = ControlProtocol.encodeControlCommand(new ControlCommandFrame(
                IDENTITY,
                sequence,
                100 + sequence,
                ControlOperation.REPLAY_START,
                "trial-001",
                "a".repeat(64)));
        assertTrue(replay.contains("\"content_hash\":\"" + "a".repeat(64) + "\""));

        for (ControlOperation operation :
                Set.of(ControlOperation.LESION_APPLY, ControlOperation.STIMULATION_APPLY)) {
            String frame = ControlProtocol.encodeControlCommand(new ControlCommandFrame(
                    IDENTITY,
                    ++sequence,
                    100 + sequence,
                    operation,
                    "registry-entry-01",
                    "b".repeat(64)));
            assertTrue(frame.contains("\"operation\":\"" + operation + "\""));
            assertTrue(frame.contains("\"content_id\":\"registry-entry-01\""));
            assertTrue(frame.contains("\"content_hash\":\"" + "b".repeat(64) + "\""));
        }
        for (ControlOperation operation :
                Set.of(ControlOperation.LESION_CLEAR, ControlOperation.STIMULATION_CLEAR)) {
            String withoutHash = ControlProtocol.encodeControlCommand(new ControlCommandFrame(
                    IDENTITY, ++sequence, 100 + sequence, operation, "registry-entry-01", null));
            assertTrue(withoutHash.contains("\"content_hash\":null"));
            String withHash = ControlProtocol.encodeControlCommand(new ControlCommandFrame(
                    IDENTITY,
                    ++sequence,
                    100 + sequence,
                    operation,
                    "registry-entry-01",
                    "c".repeat(64)));
            assertTrue(withHash.contains("\"content_hash\":\"" + "c".repeat(64) + "\""));
        }

        assertThrows(
                IllegalArgumentException.class,
                () -> new ControlCommandFrame(
                        IDENTITY, 9, 109, ControlOperation.PAUSE, "semantic-target", null));
        assertThrows(
                IllegalArgumentException.class,
                () -> new ControlCommandFrame(
                        IDENTITY, 9, 109, ControlOperation.REPLAY_START, "trial-001", null));
        assertThrows(
                IllegalArgumentException.class,
                () -> new ControlCommandFrame(
                        IDENTITY, 9, 109, ControlOperation.LESION_APPLY, "lesion-a", null));
        assertThrows(
                IllegalArgumentException.class,
                () -> new ControlCommandFrame(
                        IDENTITY, 9, 109, ControlOperation.STIMULATION_CLEAR, null, null));
    }

    @Test
    void encodeHelloOmitsRawToken() {
        String nonce = ControlProtocol.generateClientNonce();
        String hello = ControlProtocol.encodeHello(IDENTITY, TOKEN, nonce);

        assertFalse(hello.contains(TOKEN));
        assertFalse(hello.contains("token"));
        assertTrue(hello.contains("\"client_nonce\":\"" + nonce + "\""));
        assertTrue(hello.contains("\"auth\":\""
                + ControlProtocol.handshakeAuth(
                        TOKEN, "hello", IDENTITY, "minecraft_server", nonce)
                + "\""));
        // The proof is derived from the secret, so a different token cannot produce it.
        assertNotEquals(
                ControlProtocol.handshakeAuth(
                        TOKEN, "hello", IDENTITY, "minecraft_server", nonce),
                ControlProtocol.handshakeAuth(
                        "B".repeat(43), "hello", IDENTITY, "minecraft_server", nonce));
        assertThrows(
                IllegalArgumentException.class,
                () -> ControlProtocol.encodeHello(IDENTITY, "short", nonce));
        assertThrows(
                IllegalArgumentException.class,
                () -> ControlProtocol.encodeHello(IDENTITY, TOKEN, "not-a-nonce"));
    }

    @Test
    void decodeHelloAckRejectsMissingServerAuth() throws Exception {
        String nonce = ControlProtocol.generateClientNonce();
        String serverAuth = ControlProtocol.handshakeAuth(
                TOKEN, "hello_ack", IDENTITY, "neural_service", nonce);
        String acknowledgement = helloAckJson("\"server_auth\":\"" + serverAuth + "\",");

        HelloAck decoded =
                assertInstanceOf(HelloAck.class, ControlProtocol.decodeInbound(acknowledgement));
        assertEquals(serverAuth, decoded.serverAuth());
        assertTrue(ControlProtocol.verifyHelloAck(decoded, TOKEN, IDENTITY, nonce));
        assertFalse(ControlProtocol.verifyHelloAck(decoded, TOKEN, IDENTITY, null));
        assertFalse(ControlProtocol.verifyHelloAck(
                decoded, TOKEN, IDENTITY, ControlProtocol.generateClientNonce()));
        assertFalse(ControlProtocol.verifyHelloAck(decoded, "C".repeat(43), IDENTITY, nonce));

        assertThrows(
                ProtocolException.class, () -> ControlProtocol.decodeInbound(helloAckJson("")));
        assertThrows(
                ProtocolException.class,
                () -> ControlProtocol.decodeInbound(
                        helloAckJson("\"server_auth\":\"not-a-digest\",")));
        assertThrows(
                ProtocolException.class,
                () -> ControlProtocol.decodeInbound(
                        helloAckJson("\"server_auth\":\"" + serverAuth.toUpperCase(Locale.ROOT)
                                + "\",")));
    }

    @Test
    void handshakeTranscriptMatchesThePinnedCrossLanguageVector() {
        // The same token, nonce, identity, and digests are asserted by
        // tests/protocol/test_messages.py so both endpoints are pinned to one layout.
        ControllerIdentity goldenIdentity = new ControllerIdentity(
                "demo-001",
                1,
                "fly-01",
                "1".repeat(64),
                "2".repeat(64),
                "3".repeat(64),
                "4".repeat(64),
                "5".repeat(64));
        String goldenToken = "A".repeat(43);
        byte[] rawNonce = new byte[ControlProtocol.HANDSHAKE_NONCE_BYTES];
        for (int index = 0; index < rawNonce.length; index++) {
            rawNonce[index] = (byte) index;
        }
        String nonce = Base64.getUrlEncoder().withoutPadding().encodeToString(rawNonce);
        assertEquals("AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8", nonce);

        byte[] transcript = ControlProtocol.handshakeTranscript(
                "hello", goldenIdentity, "minecraft_server", nonce);
        assertEquals(469, transcript.length);
        assertEquals(
                "000000166e6575726f63726166742e70726f746f636f6c2e76320000000568656c6c6f",
                hex(java.util.Arrays.copyOfRange(transcript, 0, 35)));
        assertEquals(
                "154741d2ddc60145534fbb783a5cce817882786ac28f14c31c5cb72559ffe73b",
                ControlProtocol.handshakeAuth(
                        goldenToken, "hello", goldenIdentity, "minecraft_server", nonce));
        assertEquals(
                "ac0e2401342ca596b6b67f43140a652bc96dabeecfaa600a917beab809dd57ee",
                ControlProtocol.handshakeAuth(
                        goldenToken, "hello_ack", goldenIdentity, "neural_service", nonce));
    }

    @Test
    void encodedFramesAreBoundedAndAuthenticationAppearsOnlyInHello() {
        PhysicalSensorSnapshot snapshot = new PhysicalSensorSnapshot(
                100,
                "fly-01",
                1,
                2,
                3,
                0.1,
                -0.2,
                0.3,
                45,
                -10,
                true,
                false,
                false,
                false,
                false,
                6,
                15,
                6,
                4,
                3,
                2,
                1,
                5,
                0.5);

        String nonce = ControlProtocol.generateClientNonce();
        String hello = ControlProtocol.encodeHello(IDENTITY, TOKEN, nonce);
        String observation = ControlProtocol.encodeObservation(IDENTITY, snapshot, 2, 123, 0);

        assertTrue(hello.contains("\"protocol\":\"neurocraft.protocol.v2\""));
        assertTrue(hello.contains("\"client_nonce\":\"" + nonce + "\""));
        assertTrue(observation.contains("\"ray_distances_m\":[4.0,3.0,2.0,1.0,5.0,0.5]"));
        assertTrue(observation.contains("\"bridge_observations_coalesced\":0"));
        assertFalse(observation.contains("token"));
        assertFalse(observation.contains("x_blocks"));
        assertTrue(observation.getBytes(StandardCharsets.UTF_8).length
                <= ControlProtocol.MAX_MESSAGE_BYTES);
        assertThrows(
                IllegalArgumentException.class,
                () -> ControlProtocol.encodeObservation(IDENTITY, snapshot, 2, 123, -1));
    }

    @Test
    void observationReportsCanonicalRepresentationalClamps() throws Exception {
        PhysicalSensorSnapshot snapshot = new PhysicalSensorSnapshot(
                100,
                "fly-01",
                30_000_001,
                2,
                3,
                100,
                0,
                0,
                450,
                100,
                false,
                false,
                false,
                false,
                false,
                0,
                0,
                6,
                7,
                6,
                6,
                6,
                6,
                6);

        String json = ControlProtocol.encodeObservation(IDENTITY, snapshot, 0, 1, 0);

        assertTrue(json.contains("\"clamped_fields\":[\"position_m\",\"velocity_m_s\","
                + "\"yaw_rad\",\"pitch_rad\",\"ray_distances_m\"]"));
        assertEquals(
                "observation",
                StrictJson.parseObject(json, ControlProtocol.MAX_MESSAGE_BYTES).get("type"));
    }

    private static String actionJson(String forward) {
        return "{" +
                "\"type\":\"action\"," +
                "\"protocol\":\"neurocraft.protocol.v2\"," +
                "\"identity\":{" +
                "\"run_id\":\"run-1\"," +
                "\"session_epoch\":7," +
                "\"entity_id\":\"fly-01\"," +
                "\"graph_hash\":\"" + "1".repeat(64) + "\"," +
                "\"adapter_hash\":\"" + "2".repeat(64) + "\"," +
                "\"dynamics_hash\":\"" + "3".repeat(64) + "\"," +
                "\"decoder_hash\":\"" + "4".repeat(64) + "\"," +
                "\"observation_schema_hash\":\"" + "5".repeat(64) + "\"}," +
                "\"command_seq\":0," +
                "\"observation_seq\":0," +
                "\"observation_tick\":100," +
                "\"target_tick\":101," +
                "\"valid_through_tick\":102," +
                "\"neural_step_start\":0," +
                "\"neural_step_end\":5," +
                "\"state_digest\":\"" + "6".repeat(64) + "\"," +
                "\"motors\":{" +
                "\"forward_thrust\":" + forward + "," +
                "\"lateral_thrust\":-0.1," +
                "\"vertical_thrust\":0," +
                "\"yaw_torque\":0.05," +
                "\"pitch_torque\":0}," +
                "\"neutral\":false}";
    }

    @Test
    void decodesProgramActionAndRefusesUnknownProgramMissingKeysAndEmptyInterval()
            throws Exception {
        ProgramActionFrame frame = assertInstanceOf(
                ProgramActionFrame.class, ControlProtocol.decodeInbound(programActionJson()));
        assertEquals(BodyProgramId.CRUISE, frame.program());
        assertEquals(new Modulation(0.7, 1.0, 1.0, 0.0), frame.modulation());
        assertEquals(new ProgramParameters(-0.25), frame.parameters());
        assertEquals(new ReadoutValues(0.0, -0.25, 0.61), frame.readout());

        String json = programActionJson();
        // A program frame is never neutral, so an empty neural interval is refused outright --
        // strictly, unlike an action frame, which may carry one while neutral.
        assertThrows(ProtocolException.class, () -> ControlProtocol.decodeInbound(
                json.replace("\"neural_step_end\":5", "\"neural_step_end\":0")));
        // Case-sensitive, closed program vocabulary.
        assertThrows(ProtocolException.class, () -> ControlProtocol.decodeInbound(
                json.replace("\"program\":\"CRUISE\"", "\"program\":\"cruise\"")));
        assertThrows(ProtocolException.class, () -> ControlProtocol.decodeInbound(
                json.replace("\"program\":\"CRUISE\"", "\"program\":\"SOAR\"")));
        // The exact-key rule, at the top level and at every nesting level.
        assertThrows(ProtocolException.class, () -> ControlProtocol.decodeInbound(
                json.replaceFirst("\\}$", ",\"motors\":{}}")));
        assertThrows(ProtocolException.class, () -> ControlProtocol.decodeInbound(
                json.replace(",\"policy_hash\":\"" + "7".repeat(64) + "\"", "")));
        assertThrows(ProtocolException.class, () -> ControlProtocol.decodeInbound(
                json.replace("\"turn_bias\":-0.25", "\"turn_bias\":-0.25,\"turn_gain\":0")));
        assertThrows(ProtocolException.class, () -> ControlProtocol.decodeInbound(
                json.replace("\"wing_common\":0.61", "\"wing\":0.61")));
        // Declared ranges are wire rules, not something the Java clamp absorbs later.
        assertThrows(ProtocolException.class, () -> ControlProtocol.decodeInbound(
                json.replace("\"speed\":0.7", "\"speed\":1.5")));
        assertThrows(ProtocolException.class, () -> ControlProtocol.decodeInbound(
                json.replace("\"turn_bias\":-0.25", "\"turn_bias\":-1.5")));
        assertThrows(ProtocolException.class, () -> ControlProtocol.decodeInbound(
                json.replace("\"ttmn_jump\":0.0", "\"ttmn_jump\":-0.1")));
        // Duplicate keys at any depth, and the non-numbers JSON has no token for.
        assertThrows(ProtocolException.class, () -> ControlProtocol.decodeInbound(
                json.replace("\"speed\":0.7", "\"speed\":0.7,\"speed\":0.7")));
        assertThrows(ProtocolException.class, () -> ControlProtocol.decodeInbound(
                json.replace("\"speed\":0.7", "\"speed\":NaN")));
        assertThrows(ProtocolException.class, () -> ControlProtocol.decodeInbound(
                json.replace("\"wing_common\":0.61", "\"wing_common\":Infinity")));
        // Mislabelling the frame fails closed in both directions.
        assertThrows(ProtocolException.class, () -> ControlProtocol.decodeInbound(
                json.replace("\"type\":\"program_action\"", "\"type\":\"action\"")));
        assertThrows(ProtocolException.class, () -> ControlProtocol.decodeInbound(
                actionJson("0.25").replace("\"type\":\"action\"", "\"type\":\"program_action\"")));
    }

    /** Integer JSON is a legal float on this wire, exactly as the Python side accepts it. */
    @Test
    void programActionAcceptsJsonIntegersForItsFloatFields() throws Exception {
        ProgramActionFrame frame = assertInstanceOf(
                ProgramActionFrame.class,
                ControlProtocol.decodeInbound(programActionJson()
                        .replace("\"amplitude\":1.0", "\"amplitude\":1")
                        .replace("\"turn_bias\":-0.25", "\"turn_bias\":-1")
                        .replace("\"ttmn_jump\":0.0", "\"ttmn_jump\":0")));

        assertEquals(1.0, frame.modulation().amplitude());
        assertEquals(-1.0, frame.parameters().turnBias());
        assertEquals(0.0, frame.readout().ttmnJump());
    }

    @Test
    void encodesProgramReceiptValidationMatrix() throws Exception {
        ProgramActionFrame frame = assertInstanceOf(
                ProgramActionFrame.class, ControlProtocol.decodeInbound(programActionJson()));
        PhysicalSensorSnapshot world = snapshot(101);

        Map<String, Object> applied = decode(ControlProtocol.encodeProgramReceipt(
                IDENTITY,
                frame,
                world,
                Optional.of(new ProgramExecution(
                        BodyProgramId.CRUISE,
                        frame.modulation(),
                        frame.parameters(),
                        PrimitiveAction.clamped(0.35, 0.0, 0.1, -0.05, 0.0),
                        Optional.empty())),
                ValidationResult.accepted(frame)));
        assertEquals(18, applied.size());
        assertEquals("PROGRAM_APPLIED", applied.get("validation"));
        assertNull(applied.get("reason"));
        assertNull(applied.get("fault_code"));
        assertEquals("CRUISE", applied.get("executed_program"));
        assertEquals(applied.get("requested_modulation"), applied.get("executed_modulation"));
        assertEquals(applied.get("requested_parameters"), applied.get("executed_parameters"));

        Map<String, Object> refused = decode(ControlProtocol.encodeProgramReceipt(
                IDENTITY,
                frame,
                world,
                Optional.of(new ProgramExecution(
                        BodyProgramId.HOVER,
                        frame.modulation(),
                        frame.parameters(),
                        PrimitiveAction.clamped(0.0, 0.0, 0.08, 0.0, 0.0),
                        Optional.of(ProgramRefusalReason.PRECONDITION))),
                ValidationResult.accepted(frame)));
        assertEquals("PROGRAM_REFUSED", refused.get("validation"));
        assertEquals("precondition", refused.get("reason"));
        assertEquals("HOVER", refused.get("executed_program"));
        assertNull(refused.get("fault_code"));

        Map<String, Object> rejected = decode(ControlProtocol.encodeProgramReceipt(
                IDENTITY,
                frame,
                snapshot(104),
                Optional.empty(),
                ValidationResult.rejected(frame, ControllerFault.STALE_COMMAND)));
        assertEquals("REJECTED", rejected.get("validation"));
        assertEquals("STALE", rejected.get("fault_code"));
        assertNull(rejected.get("reason"));
        assertNull(rejected.get("executed_program"));
        assertNull(rejected.get("executed_modulation"));
        assertNull(rejected.get("executed_parameters"));
        assertTrue(rejected.containsKey("executed_program"), "nullable keys are never omitted");
    }

    @Test
    void programReceiptRefusesUntruthfulAndUnboundCombinations() throws Exception {
        ProgramActionFrame frame = assertInstanceOf(
                ProgramActionFrame.class, ControlProtocol.decodeInbound(programActionJson()));
        PhysicalSensorSnapshot world = snapshot(101);
        ValidationResult acceptance = ValidationResult.accepted(frame);

        // An applied receipt may not name a program other than the requested one.
        Optional<ProgramExecution> wrongProgram = Optional.of(new ProgramExecution(
                BodyProgramId.HOVER,
                frame.modulation(),
                frame.parameters(),
                PrimitiveAction.NEUTRAL,
                Optional.empty()));
        assertThrows(IllegalArgumentException.class, () -> ControlProtocol.encodeProgramReceipt(
                IDENTITY, frame, world, wrongProgram, acceptance));

        // An unregistered or precondition refusal must have run the declared HOVER fallback.
        Optional<ProgramExecution> fabricatedFallback = Optional.of(new ProgramExecution(
                BodyProgramId.WALK,
                frame.modulation(),
                frame.parameters(),
                PrimitiveAction.NEUTRAL,
                Optional.of(ProgramRefusalReason.UNREGISTERED)));
        assertThrows(IllegalArgumentException.class, () -> ControlProtocol.encodeProgramReceipt(
                IDENTITY, frame, world, fabricatedFallback, acceptance));

        // A rejected decision executed nothing, and an accepted one executed something.
        Optional<ProgramExecution> ran = Optional.of(new ProgramExecution(
                BodyProgramId.CRUISE,
                frame.modulation(),
                frame.parameters(),
                PrimitiveAction.NEUTRAL,
                Optional.empty()));
        assertThrows(IllegalArgumentException.class, () -> ControlProtocol.encodeProgramReceipt(
                IDENTITY,
                frame,
                world,
                ran,
                ValidationResult.rejected(frame, ControllerFault.STALE_COMMAND)));
        assertThrows(IllegalArgumentException.class, () -> ControlProtocol.encodeProgramReceipt(
                IDENTITY, frame, world, Optional.empty(), acceptance));

        // An applied receipt outside the frame's two-tick window is refused.
        assertThrows(IllegalArgumentException.class, () -> ControlProtocol.encodeProgramReceipt(
                IDENTITY, frame, snapshot(104), ran, acceptance));
    }

    /** Negative zero is written as {@code 0.0} on every float this frame carries. */
    @Test
    void programReceiptEncodesNegativeZeroAsZero() throws Exception {
        ProgramActionFrame frame = assertInstanceOf(
                ProgramActionFrame.class,
                ControlProtocol.decodeInbound(
                        programActionJson().replace("\"turn_bias\":-0.25", "\"turn_bias\":-0.0")));
        String encoded = ControlProtocol.encodeProgramReceipt(
                IDENTITY,
                frame,
                snapshot(101),
                Optional.of(new ProgramExecution(
                        BodyProgramId.CRUISE,
                        frame.modulation(),
                        frame.parameters(),
                        PrimitiveAction.clamped(-0.0, 0.0, 0.0, -0.0, 0.0),
                        Optional.empty())),
                ValidationResult.accepted(frame));

        // A JSON number token that is a negative zero, anywhere at any depth. Matching the token
        // rather than the substring keeps identifiers such as "fly-01" out of the assertion.
        assertFalse(
                NEGATIVE_ZERO_TOKEN.matcher(encoded).find(),
                "no encoded float may carry a negative zero: " + encoded);
    }

    private static Map<String, Object> decode(String json) throws ProtocolException {
        return StrictJson.parseObject(json, ControlProtocol.MAX_MESSAGE_BYTES);
    }

    private static PhysicalSensorSnapshot snapshot(long gameTick) {
        return new PhysicalSensorSnapshot(
                gameTick,
                "fly-01",
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

    private static String programActionJson() {
        return "{" +
                "\"type\":\"program_action\"," +
                "\"protocol\":\"neurocraft.protocol.v2\"," +
                "\"identity\":{" +
                "\"run_id\":\"run-1\"," +
                "\"session_epoch\":7," +
                "\"entity_id\":\"fly-01\"," +
                "\"graph_hash\":\"" + "1".repeat(64) + "\"," +
                "\"adapter_hash\":\"" + "2".repeat(64) + "\"," +
                "\"dynamics_hash\":\"" + "3".repeat(64) + "\"," +
                "\"decoder_hash\":\"" + "4".repeat(64) + "\"," +
                "\"observation_schema_hash\":\"" + "5".repeat(64) + "\"}," +
                "\"command_seq\":0," +
                "\"observation_seq\":0," +
                "\"observation_tick\":100," +
                "\"target_tick\":101," +
                "\"valid_through_tick\":102," +
                "\"neural_step_start\":0," +
                "\"neural_step_end\":5," +
                "\"state_digest\":\"" + "6".repeat(64) + "\"," +
                "\"program\":\"CRUISE\"," +
                "\"modulation\":{" +
                "\"speed\":0.7," +
                "\"turn_rate\":1.0," +
                "\"amplitude\":1.0," +
                "\"urgency\":0.0}," +
                "\"parameters\":{\"turn_bias\":-0.25}," +
                "\"readout\":{" +
                "\"ttmn_jump\":0.0," +
                "\"dna01_turn\":-0.25," +
                "\"wing_common\":0.61}," +
                "\"policy_hash\":\"" + "7".repeat(64) + "\"}";
    }

    private static String helloAckJson(String serverAuthClause) {
        return "{" +
                "\"type\":\"hello_ack\"," +
                "\"protocol\":\"neurocraft.protocol.v2\"," +
                "\"identity\":{" +
                "\"run_id\":\"run-1\"," +
                "\"session_epoch\":7," +
                "\"entity_id\":\"fly-01\"," +
                "\"graph_hash\":\"" + "1".repeat(64) + "\"," +
                "\"adapter_hash\":\"" + "2".repeat(64) + "\"," +
                "\"dynamics_hash\":\"" + "3".repeat(64) + "\"," +
                "\"decoder_hash\":\"" + "4".repeat(64) + "\"," +
                "\"observation_schema_hash\":\"" + "5".repeat(64) + "\"}," +
                "\"role\":\"neural_service\"," +
                serverAuthClause +
                "\"accepted\":true}";
    }

    private static String hex(byte[] raw) {
        StringBuilder output = new StringBuilder(raw.length * 2);
        for (byte value : raw) {
            output.append(String.format(Locale.ROOT, "%02x", value));
        }
        return output.toString();
    }
}
