package dev.neurocraft.fly.control;

import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ProgramExecution;
import dev.neurocraft.fly.body.ProgramParameters;
import dev.neurocraft.fly.body.ProgramRefusalReason;
import dev.neurocraft.fly.body.ReadoutValues;
import dev.neurocraft.fly.entity.ActuationReceipt;
import java.math.BigDecimal;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Pattern;
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;

/** Exact Java implementation of the canonical transport-neutral protocol v2 schema. */
public final class ControlProtocol {
    public static final String PROTOCOL_VERSION = "neurocraft.protocol.v2";
    public static final String WEBSOCKET_SUBPROTOCOL = "neurocraft.v2";
    public static final int MAX_MESSAGE_BYTES = 65_536;
    public static final int RAY_COUNT = 6;
    /** Random bytes carried by {@code hello.client_nonce}, before base64url encoding. */
    public static final int HANDSHAKE_NONCE_BYTES = 32;
    /** Length of the unpadded base64url encoding of {@link #HANDSHAKE_NONCE_BYTES} bytes. */
    public static final int HANDSHAKE_NONCE_CHARS = 43;

    private static final String HELLO_LABEL = "hello";
    private static final String HELLO_ACK_LABEL = "hello_ack";
    private static final String MINECRAFT_SERVER_ROLE = "minecraft_server";
    private static final String NEURAL_SERVICE_ROLE = "neural_service";
    private static final String HANDSHAKE_MAC_ALGORITHM = "HmacSHA256";
    private static final int HANDSHAKE_LENGTH_PREFIX_BYTES = 4;
    private static final char[] LOWERCASE_HEX = "0123456789abcdef".toCharArray();
    private static final SecureRandom NONCE_SOURCE = new SecureRandom();

    private static final Pattern IDENTIFIER =
            Pattern.compile("[A-Za-z0-9][A-Za-z0-9._:-]{0,127}");
    private static final Pattern HASH = Pattern.compile("[0-9a-f]{64}");
    private static final Pattern TOKEN = Pattern.compile("[A-Za-z0-9_-]{32,256}");
    private static final Pattern NONCE =
            Pattern.compile("[A-Za-z0-9_-]{" + HANDSHAKE_NONCE_CHARS + "}");
    private static final Set<String> IDENTITY_KEYS = Set.of(
            "run_id",
            "session_epoch",
            "entity_id",
            "graph_hash",
            "adapter_hash",
            "dynamics_hash",
            "decoder_hash",
            "observation_schema_hash");
    private static final Set<String> ACK_KEYS =
            Set.of("type", "protocol", "identity", "role", "server_auth", "accepted");
    private static final Set<String> ACTION_KEYS = Set.of(
            "type",
            "protocol",
            "identity",
            "command_seq",
            "observation_seq",
            "observation_tick",
            "target_tick",
            "valid_through_tick",
            "neural_step_start",
            "neural_step_end",
            "state_digest",
            "motors",
            "neutral");
    private static final Set<String> PROGRAM_ACTION_KEYS = Set.of(
            "type",
            "protocol",
            "identity",
            "command_seq",
            "observation_seq",
            "observation_tick",
            "target_tick",
            "valid_through_tick",
            "neural_step_start",
            "neural_step_end",
            "state_digest",
            "program",
            "modulation",
            "parameters",
            "readout",
            "policy_hash");
    private static final Set<String> MODULATION_KEYS =
            Set.of("speed", "turn_rate", "amplitude", "urgency");
    private static final Set<String> PARAMETER_KEYS = Set.of("turn_bias");
    private static final Set<String> READOUT_KEYS =
            Set.of("ttmn_jump", "dna01_turn", "wing_common");
    private static final Set<String> MOTOR_KEYS = Set.of(
            "forward_thrust",
            "lateral_thrust",
            "vertical_thrust",
            "yaw_torque",
            "pitch_torque");
    private static final Set<String> FAULT_KEYS = Set.of(
            "type",
            "protocol",
            "identity",
            "code",
            "message",
            "game_tick",
            "related_observation_seq",
            "neutral_applied");
    private static final Set<String> CONTROL_ACK_KEYS = Set.of(
            "type",
            "protocol",
            "identity",
            "control_seq",
            "issued_tick",
            "operation",
            "content_id",
            "content_hash",
            "accepted",
            "state",
            "effective_tick",
            "fault_code");
    private static final Set<String> FAULT_CODES = Set.of(
            "MALFORMED",
            "OVERSIZE",
            "UNKNOWN_FIELD",
            "UNAUTHORIZED",
            "IDENTITY_MISMATCH",
            "SEQUENCE",
            "STALE",
            "FUTURE",
            "QUEUE_OVERFLOW",
            "TRACE_FAILURE",
            "INTERNAL",
            "DISCONNECTED",
            "CONTROL_UNAVAILABLE",
            "INVALID_STATE",
            "RECEIPT_MISMATCH");

    private ControlProtocol() {}

    public static InboundFrame decodeInbound(String json) throws ProtocolException {
        Map<String, Object> values = StrictJson.parseObject(json, MAX_MESSAGE_BYTES);
        String type = requireString(values, "type");
        if (!PROTOCOL_VERSION.equals(requireString(values, "protocol"))) {
            throw new ProtocolException("unsupported protocol version");
        }
        return switch (type) {
            case "hello_ack" -> decodeAck(values);
            case "action" -> decodeAction(values);
            case "program_action" -> decodeProgramAction(values);
            case "control_ack" -> decodeControlAck(values);
            case "fault" -> decodeFault(values);
            default -> throw new ProtocolException("message type is missing or unknown");
        };
    }

    /**
     * Encode the opening handshake frame. The run token never reaches the wire: the frame
     * carries a fresh {@code client_nonce} and an {@code auth} proof computed as
     * {@code HMAC-SHA256(token, transcript)} over the canonical layout documented on
     * {@link #handshakeTranscript}. Use {@link #generateClientNonce()} for the nonce and keep
     * it so the matching {@code hello_ack} proof can be verified.
     */
    public static String encodeHello(
            ControllerIdentity identity, String token, String clientNonce) {
        String auth = handshakeAuth(
                token, HELLO_LABEL, identity, MINECRAFT_SERVER_ROLE, clientNonce);
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("type", HELLO_LABEL);
        message.put("protocol", PROTOCOL_VERSION);
        message.put("identity", encodeIdentity(identity));
        message.put("role", MINECRAFT_SERVER_ROLE);
        message.put("client_nonce", clientNonce);
        message.put("auth", auth);
        return StrictJson.encode(message, MAX_MESSAGE_BYTES);
    }

    /** Return 32 fresh {@link SecureRandom} bytes encoded as unpadded base64url. */
    public static String generateClientNonce() {
        byte[] raw = new byte[HANDSHAKE_NONCE_BYTES];
        NONCE_SOURCE.nextBytes(raw);
        return Base64.getUrlEncoder().withoutPadding().encodeToString(raw);
    }

    /**
     * Constant-time check that {@code acknowledgement} proves possession of {@code token} for
     * the nonce this bridge sent. Returns {@code false} for a null or malformed nonce, which is
     * exactly the state of a bridge that has not yet sent its {@code hello}.
     */
    public static boolean verifyHelloAck(
            HelloAck acknowledgement,
            String token,
            ControllerIdentity identity,
            String clientNonce) {
        if (acknowledgement == null
                || token == null
                || identity == null
                || clientNonce == null
                || !NONCE.matcher(clientNonce).matches()
                || !TOKEN.matcher(token).matches()) {
            return false;
        }
        String expected = handshakeAuth(
                token, HELLO_ACK_LABEL, identity, NEURAL_SERVICE_ROLE, clientNonce);
        return MessageDigest.isEqual(
                expected.getBytes(StandardCharsets.UTF_8),
                acknowledgement.serverAuth().getBytes(StandardCharsets.UTF_8));
    }

    /**
     * Return the canonical handshake transcript bytes that both endpoints authenticate.
     *
     * <p>The layout is an unambiguous, length-prefixed concatenation of exactly twelve UTF-8
     * strings in a frozen order. Every element is preceded by its byte length as a 4-byte
     * big-endian unsigned integer, so no choice of field content can shift a boundary:
     *
     * <pre>
     * transcript := element(protocol) || element(label)
     *            || element(run_id) || element(session_epoch)
     *            || element(entity_id) || element(graph_hash)
     *            || element(adapter_hash) || element(dynamics_hash)
     *            || element(decoder_hash) || element(observation_schema_hash)
     *            || element(role) || element(client_nonce)
     * element(s) := uint32_be(len(utf8(s))) || utf8(s)
     * </pre>
     *
     * <p>{@code protocol} is {@link #PROTOCOL_VERSION} and {@code label} is the message type
     * ({@code hello} or {@code hello_ack}), which together domain-separate this transcript from
     * every other protocol revision and from the opposite handshake direction.
     * {@code session_epoch} is rendered as its canonical base-10 ASCII form with no sign and no
     * leading zeros.
     *
     * <p>The Python package produces byte-identical transcripts in
     * {@code neurocraft_protocol.messages.handshake_transcript}; {@code ControlProtocolTest} and
     * {@code tests/protocol/test_messages.py} pin the same vector on both sides.
     */
    public static byte[] handshakeTranscript(
            String label, ControllerIdentity identity, String role, String clientNonce) {
        String[] elements = {
            PROTOCOL_VERSION,
            label,
            identity.runId(),
            Long.toString(identity.sessionEpoch()),
            identity.entityId(),
            identity.graphHash(),
            identity.adapterHash(),
            identity.dynamicsHash(),
            identity.decoderHash(),
            identity.observationSchemaHash(),
            role,
            clientNonce
        };
        int total = 0;
        byte[][] encoded = new byte[elements.length][];
        for (int index = 0; index < elements.length; index++) {
            encoded[index] = elements[index].getBytes(StandardCharsets.UTF_8);
            total += HANDSHAKE_LENGTH_PREFIX_BYTES + encoded[index].length;
        }
        ByteBuffer transcript = ByteBuffer.allocate(total);
        for (byte[] element : encoded) {
            transcript.putInt(element.length);
            transcript.put(element);
        }
        return transcript.array();
    }

    /** Return {@code HMAC-SHA256(token, transcript)} as a lowercase hex digest. */
    public static String handshakeAuth(
            String token,
            String label,
            ControllerIdentity identity,
            String role,
            String clientNonce) {
        if (token == null || !TOKEN.matcher(token).matches()) {
            throw new IllegalArgumentException("run token violates the canonical token contract");
        }
        if (clientNonce == null || !NONCE.matcher(clientNonce).matches()) {
            throw new IllegalArgumentException(
                    "handshake nonce must be 32 random bytes in unpadded base64url");
        }
        try {
            Mac mac = Mac.getInstance(HANDSHAKE_MAC_ALGORITHM);
            mac.init(new SecretKeySpec(
                    token.getBytes(StandardCharsets.UTF_8), HANDSHAKE_MAC_ALGORITHM));
            return lowercaseHex(
                    mac.doFinal(handshakeTranscript(label, identity, role, clientNonce)));
        } catch (GeneralSecurityException exception) {
            throw new IllegalStateException("HmacSHA256 is required by every Java 21 runtime",
                    exception);
        }
    }

    private static String lowercaseHex(byte[] raw) {
        char[] hex = new char[raw.length * 2];
        for (int index = 0; index < raw.length; index++) {
            int value = raw[index] & 0xff;
            hex[index * 2] = LOWERCASE_HEX[value >>> 4];
            hex[index * 2 + 1] = LOWERCASE_HEX[value & 0x0f];
        }
        return new String(hex);
    }

    /** SHA-256 of the declared typed-ray contract, pinned by the Python/Java golden. */
    public static final String OBSERVATION_SCHEMA_V2_SHA256 =
            "032b6a74d57b37f9fc1883b4865c89ec2eeccc5c9084831e29cd986738559308";

    /** SHA-256 of the versioned event contract, pinned against the Python declaration. */
    public static final String OBSERVATION_SCHEMA_V3_SHA256 =
            "d7b135a1ce79d034454394d8ee73eb5b9f5fe46bb79ee924da465b8cec62ba80";

    /** SHA-256 of the same-hit actor metadata contract, pinned against the Python declaration. */
    public static final String OBSERVATION_SCHEMA_V4_SHA256 =
            "e89419ad6b9c174e243e12180d9ffe0f131256a2e0cc2ec14693df0367d601f1";

    /** Exactly the declared observation versions carrying event-display inputs. */
    public static boolean isEventObservationSchema(String schemaHash) {
        return OBSERVATION_SCHEMA_V3_SHA256.equals(schemaHash)
                || OBSERVATION_SCHEMA_V4_SHA256.equals(schemaHash);
    }

    public static String encodeObservation(
            ControllerIdentity identity,
            PhysicalSensorSnapshot snapshot,
            long observationSequence,
            long sentMonotonicNs,
            long bridgeObservationsCoalesced) {
        if (!identity.entityId().equals(snapshot.entityId())) {
            throw new IllegalArgumentException("snapshot entity does not match the session identity");
        }
        if (observationSequence < 0
                || sentMonotonicNs < 0
                || bridgeObservationsCoalesced < 0) {
            throw new IllegalArgumentException(
                    "observation sequence, monotonic time, and bridge coalescence must be non-negative");
        }

        List<String> clampedFields = new ArrayList<>();
        List<Double> position = clampedVector(
                clampedFields,
                "position_m",
                -30_000_000.0,
                30_000_000.0,
                snapshot.xBlocks(),
                snapshot.yBlocks(),
                snapshot.zBlocks());
        List<Double> velocity = clampedVector(
                clampedFields,
                "velocity_m_s",
                -1_024.0,
                1_024.0,
                snapshot.velocityXBlocksPerTick() * 20.0,
                snapshot.velocityYBlocksPerTick() * 20.0,
                snapshot.velocityZBlocksPerTick() * 20.0);
        double rawYawRadians = Math.toRadians(snapshot.yawDegrees());
        double yawRadians = rawYawRadians >= -Math.PI && rawYawRadians <= Math.PI
                ? rawYawRadians
                : Math.atan2(Math.sin(rawYawRadians), Math.cos(rawYawRadians));
        if (Double.compare(rawYawRadians, yawRadians) != 0) {
            clampedFields.add("yaw_rad");
        }
        double rawPitchRadians = Math.toRadians(snapshot.pitchDegrees());
        double pitchRadians = clamp(rawPitchRadians, -Math.PI / 2.0, Math.PI / 2.0);
        if (Double.compare(rawPitchRadians, pitchRadians) != 0) {
            clampedFields.add("pitch_rad");
        }
        double rayMaximum = snapshot.rayMaxDistanceBlocks();
        List<Double> rays = clampedVector(
                clampedFields,
                "ray_distances_m",
                0.0,
                rayMaximum,
                // Frozen order: forward, back, left, right, body-up, body-down.
                snapshot.rayForwardBlocks(),
                snapshot.rayBackBlocks(),
                snapshot.rayLeftBlocks(),
                snapshot.rayRightBlocks(),
                snapshot.rayUpBlocks(),
                snapshot.rayDownBlocks());

        Map<String, Object> sensors = new LinkedHashMap<>();
        sensors.put("position_m", position);
        sensors.put("velocity_m_s", velocity);
        sensors.put("yaw_rad", yawRadians == 0.0 ? 0.0 : yawRadians);
        sensors.put("pitch_rad", pitchRadians == 0.0 ? 0.0 : pitchRadians);
        sensors.put("on_ground", snapshot.onGround());
        sensors.put("horizontal_collision", snapshot.horizontalCollision());
        sensors.put("vertical_collision", snapshot.verticalCollision());
        sensors.put("in_fluid", snapshot.inWater() || snapshot.inLava());
        sensors.put("local_light", Math.max(snapshot.blockLight(), snapshot.skyLight()));
        sensors.put("ray_max_distance_m", rayMaximum);
        sensors.put("ray_distances_m", rays);
        // The authenticated session identity selects the shape. Existing identities keep
        // their historical wire bytes; only declared v2/v3/v4 hashes opt into typed rays.
        boolean actorsV4 = identity.observationSchemaHash().equals(OBSERVATION_SCHEMA_V4_SHA256);
        boolean eventsV3 = identity.observationSchemaHash().equals(OBSERVATION_SCHEMA_V3_SHA256);
        String sensorVersion = actorsV4 ? "v4" : eventsV3 ? "v3" : "v2";
        if (actorsV4 || eventsV3 || identity.observationSchemaHash().equals(OBSERVATION_SCHEMA_V2_SHA256)) {
            if (snapshot.rayHitKinds() == null) {
                throw new IllegalArgumentException(sensorVersion
                        + " observation requires measured ray hit kinds");
            }
            sensors.put("sensor_schema", "neurocraft.sensors." + sensorVersion);
            sensors.put("ray_hit_kinds", snapshot.rayHitKinds());
        }
        if (actorsV4) {
            if (snapshot.rayActorKinds() == null) {
                throw new IllegalArgumentException("v4 observation requires ray actor kinds");
            }
            sensors.put("ray_actor_kinds", snapshot.rayActorKinds());
        }
        if (actorsV4 || eventsV3) {
            EventSample sample = snapshot.events();
            if (sample == null) {
                throw new IllegalArgumentException(sensorVersion + " observation requires measured events");
            }
            Map<String, Object> events = new LinkedHashMap<>();
            events.put("food_contact", sample.foodContact());
            events.put("food_near", sample.foodNear());
            events.put("food", encodeEventCounter(sample.food()));
            events.put("brush", encodeEventCounter(sample.brush()));
            events.put("attack_attempt", encodeEventCounter(sample.attackAttempt()));
            sensors.put("events", events);
        }

        Map<String, Object> message = new LinkedHashMap<>();
        message.put("type", "observation");
        message.put("protocol", PROTOCOL_VERSION);
        message.put("identity", encodeIdentity(identity));
        message.put("observation_seq", observationSequence);
        message.put("game_tick", snapshot.gameTick());
        message.put("sent_monotonic_ns", sentMonotonicNs);
        message.put("bridge_observations_coalesced", bridgeObservationsCoalesced);
        message.put("sensors", sensors);
        message.put("missing_fields", List.of());
        message.put("clamped_fields", List.copyOf(clampedFields));
        return StrictJson.encode(message, MAX_MESSAGE_BYTES);
    }

    private static Map<String, Object> encodeEventCounter(EventSample.Counter counter) {
        Map<String, Object> value = new LinkedHashMap<>();
        value.put("count", counter.count());
        value.put("last_game_tick", counter.lastGameTick());
        return value;
    }

    public static String encodeActuationReceipt(
            ControllerIdentity identity,
            ActionFrame command,
            PhysicalSensorSnapshot worldBefore,
            ActuationReceipt receipt,
            ValidationResult decision) {
        if (worldBefore == null) {
            throw new NullPointerException("worldBefore");
        }
        long appliedTick = worldBefore.gameTick();
        if (!identity.equals(command.identity())
                || !identity.entityId().equals(worldBefore.entityId())
                || !receipt.rawAction().equals(command.rawAction())
                || decision.command().isEmpty()
                || !decision.command().orElseThrow().equals(command)) {
            throw new IllegalArgumentException("actuation receipt does not bind its command");
        }
        if (decision.accepted()
                && (appliedTick < command.targetTick() || appliedTick - command.targetTick() > 2)) {
            throw new IllegalArgumentException("actuation receipt is outside its two-tick window");
        }
        List<String> clampFields = changedMotorFields(
                receipt.rawAction(), receipt.appliedAction());
        String validation;
        String faultCode;
        if (decision.accepted()) {
            validation = clampFields.isEmpty() ? "APPLIED" : "CLAMPED";
            faultCode = null;
        } else {
            if (!receipt.appliedAction().isNeutral()) {
                throw new IllegalArgumentException("rejected receipt must apply neutral motors");
            }
            validation = "REJECTED";
            clampFields = List.of();
            faultCode = protocolFault(decision.state());
        }

        Map<String, Object> message = new LinkedHashMap<>();
        message.put("type", "actuation_receipt");
        message.put("protocol", PROTOCOL_VERSION);
        message.put("identity", encodeIdentity(identity));
        message.put("command_seq", command.commandSequence());
        message.put("observation_seq", command.observationSequence());
        message.put("target_tick", command.targetTick());
        message.put("applied_tick", appliedTick);
        message.put("world_before", encodeWorldBefore(worldBefore));
        message.put("requested_motors", encodeMotors(receipt.rawAction()));
        message.put("applied_motors", encodeMotors(receipt.appliedAction()));
        message.put("validation", validation);
        message.put("clamp_fields", clampFields);
        message.put("fault_code", faultCode);
        return StrictJson.encode(message, MAX_MESSAGE_BYTES);
    }

    /**
     * Encodes the authoritative report for one admitted {@code program_action}.
     *
     * <p>Eighteen keys, always present: the five nullable ones are written as JSON {@code null}
     * rather than omitted, exactly as {@code actuation_receipt} writes its {@code fault_code}.
     *
     * <p>The validation matrix mirrors the Python {@code ProgramReceipt} rules exactly.
     *
     * <ul>
     *   <li>{@code PROGRAM_APPLIED} — the requested program ran; the executed fields echo the
     *       request, and there is neither a reason nor a fault.
     *   <li>{@code PROGRAM_REFUSED} — a reason is required and no fault is allowed. For
     *       {@code unregistered} and {@code precondition} the declared HOVER fallback ran, so the
     *       executed program is pinned to HOVER. For {@code operator} the requested program never
     *       reached the body and the operator's own program ran instead: the receipt reports
     *       <em>that</em> program, under its own modulation and parameters, with the motors it
     *       actually emitted. Pinning HOVER there would make the receipt assert a program ran that
     *       did not, which is the one thing a receipt may never do.
     *   <li>{@code REJECTED} — the envelope failed validation, nothing ran, all three executed
     *       fields are null, the emitted motors are neutral, and a fault code names the failure.
     * </ul>
     *
     * @param execution what actually ran, or empty exactly when the decision rejected the frame
     */
    public static String encodeProgramReceipt(
            ControllerIdentity identity,
            ProgramActionFrame command,
            PhysicalSensorSnapshot worldBefore,
            Optional<ProgramExecution> execution,
            ValidationResult decision) {
        if (worldBefore == null) {
            throw new NullPointerException("worldBefore");
        }
        if (execution == null) {
            throw new NullPointerException("execution");
        }
        long appliedTick = worldBefore.gameTick();
        if (!identity.equals(command.identity())
                || !identity.entityId().equals(worldBefore.entityId())
                || decision.command().isEmpty()
                || !decision.command().orElseThrow().equals(command)) {
            throw new IllegalArgumentException("program receipt does not bind its command");
        }
        if (decision.accepted() != execution.isPresent()) {
            throw new IllegalArgumentException(
                    "an admitted program frame executed something and a rejected one executed nothing");
        }
        if (decision.accepted()
                && (appliedTick < command.targetTick() || appliedTick - command.targetTick() > 2)) {
            throw new IllegalArgumentException("program receipt is outside its two-tick window");
        }

        String validation;
        String reason;
        String faultCode;
        if (execution.isPresent()) {
            ProgramExecution ran = execution.orElseThrow();
            Optional<ProgramRefusalReason> refusal = ran.refusalReason();
            validation = refusal.isEmpty() ? "PROGRAM_APPLIED" : "PROGRAM_REFUSED";
            reason = refusal.map(ProgramRefusalReason::wireValue).orElse(null);
            faultCode = null;
            if (refusal.isEmpty()
                    && (ran.executedProgram() != command.program()
                            || !ran.executedModulation().equals(command.modulation())
                            || !ran.executedParameters().equals(command.parameters()))) {
                throw new IllegalArgumentException(
                        "an applied program receipt must execute exactly what was requested");
            }
            if (refusal.isPresent()
                    && refusal.orElseThrow() != ProgramRefusalReason.OPERATOR
                    && ran.executedProgram() != BodyProgramId.HOVER) {
                throw new IllegalArgumentException(
                        "an unregistered or precondition refusal must fall back to HOVER");
            }
        } else {
            validation = "REJECTED";
            reason = null;
            faultCode = protocolFault(decision.state());
        }

        Map<String, Object> message = new LinkedHashMap<>();
        message.put("type", "program_receipt");
        message.put("protocol", PROTOCOL_VERSION);
        message.put("identity", encodeIdentity(identity));
        message.put("command_seq", command.commandSequence());
        message.put("observation_seq", command.observationSequence());
        message.put("target_tick", command.targetTick());
        message.put("applied_tick", appliedTick);
        message.put("world_before", encodeWorldBefore(worldBefore));
        message.put("requested_program", command.program().name());
        message.put("requested_modulation", encodeModulation(command.modulation()));
        message.put("requested_parameters", encodeParameters(command.parameters()));
        message.put(
                "executed_program",
                execution.map(ran -> ran.executedProgram().name()).orElse(null));
        message.put(
                "executed_modulation",
                execution.map(ran -> encodeModulation(ran.executedModulation())).orElse(null));
        message.put(
                "executed_parameters",
                execution.map(ran -> encodeParameters(ran.executedParameters())).orElse(null));
        message.put(
                "emitted_motors",
                encodeEmittedMotors(execution.map(ProgramExecution::emittedMotors)
                        .orElse(PrimitiveAction.NEUTRAL)));
        message.put("validation", validation);
        message.put("reason", reason);
        message.put("fault_code", faultCode);
        return StrictJson.encode(message, MAX_MESSAGE_BYTES);
    }

    public static String encodeControlCommand(ControlCommandFrame command) {
        validateControlContent(
                command.operation(), command.contentId(), command.contentHash(), "control_command");
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("type", "control_command");
        message.put("protocol", PROTOCOL_VERSION);
        message.put("identity", encodeIdentity(command.identity()));
        message.put("control_seq", command.controlSequence());
        message.put("issued_tick", command.issuedTick());
        message.put("operation", command.operation().name());
        message.put("content_id", command.contentId());
        message.put("content_hash", command.contentHash());
        return StrictJson.encode(message, MAX_MESSAGE_BYTES);
    }

    private static HelloAck decodeAck(Map<String, Object> values) throws ProtocolException {
        requireExactKeys(values, ACK_KEYS, "hello_ack");
        if (!HELLO_ACK_LABEL.equals(requireString(values, "type"))) {
            throw new ProtocolException("message type must be hello_ack");
        }
        if (!NEURAL_SERVICE_ROLE.equals(requireString(values, "role"))) {
            throw new ProtocolException("hello_ack role must be neural_service");
        }
        if (!requireBoolean(values, "accepted")) {
            throw new ProtocolException("rejected handshakes must use a fault frame");
        }
        // Structural only: possession of the run token is proven against the bridge's own
        // stored nonce in NeuralBridge, which is the only place the token is available.
        return new HelloAck(
                decodeIdentity(requireObject(values, "identity")),
                requireHash(values, "server_auth"));
    }

    private static ActionFrame decodeAction(Map<String, Object> values) throws ProtocolException {
        requireExactKeys(values, ACTION_KEYS, "action");
        if (!"action".equals(requireString(values, "type"))) {
            throw new ProtocolException("message type must be action");
        }
        Map<String, Object> motors = requireObject(values, "motors");
        requireExactKeys(motors, MOTOR_KEYS, "motors");
        RawPrimitiveAction rawAction = new RawPrimitiveAction(
                requireFiniteNumber(motors, "forward_thrust", -1.0, 1.0),
                requireFiniteNumber(motors, "lateral_thrust", -1.0, 1.0),
                requireFiniteNumber(motors, "vertical_thrust", -1.0, 1.0),
                requireFiniteNumber(motors, "yaw_torque", -1.0, 1.0),
                requireFiniteNumber(motors, "pitch_torque", -1.0, 1.0));
        try {
            return new ActionFrame(
                    decodeIdentity(requireObject(values, "identity")),
                    requireNonNegativeLong(values, "command_seq"),
                    requireNonNegativeLong(values, "observation_seq"),
                    requireNonNegativeLong(values, "observation_tick"),
                    requireNonNegativeLong(values, "target_tick"),
                    requireNonNegativeLong(values, "valid_through_tick"),
                    requireNonNegativeLong(values, "neural_step_start"),
                    requireNonNegativeLong(values, "neural_step_end"),
                    requireHash(values, "state_digest"),
                    rawAction,
                    requireBoolean(values, "neutral"));
        } catch (IllegalArgumentException exception) {
            throw new ProtocolException("action violates the canonical protocol contract", exception);
        }
    }

    /**
     * Decodes a body-program selection.
     *
     * <p>Exactly sixteen keys, checked as a set, and every nested object closed the same way: a
     * frame labelled {@code "action"} that carries program fields fails the exact-key check, and so
     * does a {@code program_action} carrying {@code motors}. The three nested objects are validated
     * with their own exact key sets and their own declared ranges, so an out-of-range attenuator is
     * a wire refusal rather than something {@link Modulation}'s clamp quietly absorbs.
     *
     * <p>{@code StrictJson} refuses duplicate keys at every nesting depth and has no NaN or Infinity
     * token at all, so neither can reach these constructors; {@code ControlProtocolTest} pins both.
     */
    private static ProgramActionFrame decodeProgramAction(Map<String, Object> values)
            throws ProtocolException {
        requireExactKeys(values, PROGRAM_ACTION_KEYS, "program_action");
        if (!"program_action".equals(requireString(values, "type"))) {
            throw new ProtocolException("message type must be program_action");
        }
        Map<String, Object> modulation = requireObject(values, "modulation");
        requireExactKeys(modulation, MODULATION_KEYS, "modulation");
        Map<String, Object> parameters = requireObject(values, "parameters");
        requireExactKeys(parameters, PARAMETER_KEYS, "parameters");
        Map<String, Object> readout = requireObject(values, "readout");
        requireExactKeys(readout, READOUT_KEYS, "readout");
        try {
            return new ProgramActionFrame(
                    decodeIdentity(requireObject(values, "identity")),
                    requireNonNegativeLong(values, "command_seq"),
                    requireNonNegativeLong(values, "observation_seq"),
                    requireNonNegativeLong(values, "observation_tick"),
                    requireNonNegativeLong(values, "target_tick"),
                    requireNonNegativeLong(values, "valid_through_tick"),
                    requireNonNegativeLong(values, "neural_step_start"),
                    requireNonNegativeLong(values, "neural_step_end"),
                    requireHash(values, "state_digest"),
                    requireEnum(values, "program", BodyProgramId.class, "body program"),
                    new Modulation(
                            requireFiniteNumber(modulation, "speed", 0.0, 1.0),
                            requireFiniteNumber(modulation, "turn_rate", 0.0, 1.0),
                            requireFiniteNumber(modulation, "amplitude", 0.0, 1.0),
                            requireFiniteNumber(modulation, "urgency", 0.0, 1.0)),
                    new ProgramParameters(
                            requireFiniteNumber(parameters, "turn_bias", -1.0, 1.0)),
                    new ReadoutValues(
                            requireFiniteNumber(readout, "ttmn_jump", 0.0, 1.0),
                            requireFiniteNumber(readout, "dna01_turn", -1.0, 1.0),
                            requireFiniteNumber(readout, "wing_common", -1.0, 1.0)),
                    requireHash(values, "policy_hash"));
        } catch (IllegalArgumentException exception) {
            throw new ProtocolException(
                    "program_action violates the canonical protocol contract", exception);
        }
    }

    private static ControlAckFrame decodeControlAck(Map<String, Object> values)
            throws ProtocolException {
        requireExactKeys(values, CONTROL_ACK_KEYS, "control_ack");
        if (!"control_ack".equals(requireString(values, "type"))) {
            throw new ProtocolException("message type must be control_ack");
        }
        ControlOperation operation = requireEnum(
                values, "operation", ControlOperation.class, "control operation");
        String contentId = requireNullableString(values, "content_id");
        String contentHash = requireNullableString(values, "content_hash");
        try {
            validateControlContent(operation, contentId, contentHash, "control_ack");
        } catch (IllegalArgumentException exception) {
            throw new ProtocolException("control_ack content identity is invalid", exception);
        }
        boolean accepted = requireBoolean(values, "accepted");
        String faultCode = requireNullableString(values, "fault_code");
        if (faultCode != null && !FAULT_CODES.contains(faultCode)) {
            throw new ProtocolException("control_ack fault code is unknown");
        }
        if (accepted != (faultCode == null)) {
            throw new ProtocolException("control_ack acceptance and fault code disagree");
        }
        try {
            return new ControlAckFrame(
                    decodeIdentity(requireObject(values, "identity")),
                    requireNonNegativeLong(values, "control_seq"),
                    requireNonNegativeLong(values, "issued_tick"),
                    operation,
                    contentId,
                    contentHash,
                    accepted,
                    requireEnum(values, "state", ControlState.class, "control state"),
                    requireNonNegativeLong(values, "effective_tick"),
                    faultCode);
        } catch (IllegalArgumentException exception) {
            throw new ProtocolException("control_ack violates the canonical contract", exception);
        }
    }

    private static FaultFrame decodeFault(Map<String, Object> values) throws ProtocolException {
        requireExactKeys(values, FAULT_KEYS, "fault");
        if (!"fault".equals(requireString(values, "type"))) {
            throw new ProtocolException("message type must be fault");
        }
        String code = requireString(values, "code");
        if (!FAULT_CODES.contains(code)) {
            throw new ProtocolException("fault code is unknown");
        }
        String message = requireText(values, "message", 256);
        Object related = values.get("related_observation_seq");
        Long relatedSequence = related == null
                ? null
                : requireNonNegativeLongValue(related, "related_observation_seq");
        if (!requireBoolean(values, "neutral_applied")) {
            throw new ProtocolException("fault must declare neutral_applied=true");
        }
        return new FaultFrame(
                decodeIdentity(requireObject(values, "identity")),
                code,
                message,
                requireNonNegativeLong(values, "game_tick"),
                relatedSequence);
    }

    private static ControllerIdentity decodeIdentity(Map<String, Object> values)
            throws ProtocolException {
        requireExactKeys(values, IDENTITY_KEYS, "identity");
        try {
            return new ControllerIdentity(
                    requireIdentifier(values, "run_id"),
                    requireNonNegativeLong(values, "session_epoch"),
                    requireIdentifier(values, "entity_id"),
                    requireHash(values, "graph_hash"),
                    requireHash(values, "adapter_hash"),
                    requireHash(values, "dynamics_hash"),
                    requireHash(values, "decoder_hash"),
                    requireHash(values, "observation_schema_hash"));
        } catch (IllegalArgumentException exception) {
            throw new ProtocolException("identity violates the canonical protocol contract", exception);
        }
    }

    private static Map<String, Object> encodeIdentity(ControllerIdentity identity) {
        Map<String, Object> value = new LinkedHashMap<>();
        value.put("run_id", identity.runId());
        value.put("session_epoch", identity.sessionEpoch());
        value.put("entity_id", identity.entityId());
        value.put("graph_hash", identity.graphHash());
        value.put("adapter_hash", identity.adapterHash());
        value.put("dynamics_hash", identity.dynamicsHash());
        value.put("decoder_hash", identity.decoderHash());
        value.put("observation_schema_hash", identity.observationSchemaHash());
        return value;
    }

    private static Map<String, Object> encodeMotors(RawPrimitiveAction motors) {
        Map<String, Object> value = new LinkedHashMap<>();
        value.put("forward_thrust", motors.forwardThrust());
        value.put("lateral_thrust", motors.lateralThrust());
        value.put("vertical_thrust", motors.verticalThrust());
        value.put("yaw_torque", motors.yawTorque());
        value.put("pitch_torque", motors.pitchTorque());
        return value;
    }

    private static Map<String, Object> encodeMotors(PrimitiveAction motors) {
        Map<String, Object> value = new LinkedHashMap<>();
        value.put("forward_thrust", motors.forwardThrust());
        value.put("lateral_thrust", motors.lateralThrust());
        value.put("vertical_thrust", motors.verticalThrust());
        value.put("yaw_torque", motors.yawTorque());
        value.put("pitch_torque", motors.pitchTorque());
        return value;
    }

    /**
     * The program's own motors. Unlike {@code actuation_receipt}'s two motor objects, which echo a
     * wire value the peer sent, these are computed on this side, so negative zero is normalized
     * here for the same reason the decoder normalizes it on the way in.
     */
    private static Map<String, Object> encodeEmittedMotors(PrimitiveAction motors) {
        Map<String, Object> value = new LinkedHashMap<>();
        value.put("forward_thrust", zeroNormalized(motors.forwardThrust()));
        value.put("lateral_thrust", zeroNormalized(motors.lateralThrust()));
        value.put("vertical_thrust", zeroNormalized(motors.verticalThrust()));
        value.put("yaw_torque", zeroNormalized(motors.yawTorque()));
        value.put("pitch_torque", zeroNormalized(motors.pitchTorque()));
        return value;
    }

    private static Map<String, Object> encodeModulation(Modulation modulation) {
        Map<String, Object> value = new LinkedHashMap<>();
        value.put("speed", zeroNormalized(modulation.speed()));
        value.put("turn_rate", zeroNormalized(modulation.turnRate()));
        value.put("amplitude", zeroNormalized(modulation.amplitude()));
        value.put("urgency", zeroNormalized(modulation.urgency()));
        return value;
    }

    private static Map<String, Object> encodeParameters(ProgramParameters parameters) {
        Map<String, Object> value = new LinkedHashMap<>();
        value.put("turn_bias", zeroNormalized(parameters.turnBias()));
        return value;
    }

    /** Negative zero is written as {@code 0.0}, exactly as {@link #requireFiniteNumber} reads it. */
    private static double zeroNormalized(double value) {
        return value == 0.0 ? 0.0 : value;
    }

    private static Map<String, Object> encodeWorldBefore(PhysicalSensorSnapshot snapshot) {
        Map<String, Object> value = new LinkedHashMap<>();
        value.put(
                "position_m",
                boundedVector(
                        "world_before.position_m",
                        -30_000_000.0,
                        30_000_000.0,
                        snapshot.xBlocks(),
                        snapshot.yBlocks(),
                        snapshot.zBlocks()));
        value.put(
                "velocity_m_s",
                boundedVector(
                        "world_before.velocity_m_s",
                        -1_024.0,
                        1_024.0,
                        snapshot.velocityXBlocksPerTick() * 20.0,
                        snapshot.velocityYBlocksPerTick() * 20.0,
                        snapshot.velocityZBlocksPerTick() * 20.0));
        value.put(
                "yaw_rad",
                boundedNumber(
                        "world_before.yaw_rad",
                        Math.toRadians(snapshot.yawDegrees()),
                        -Math.PI,
                        Math.PI));
        value.put(
                "pitch_rad",
                boundedNumber(
                        "world_before.pitch_rad",
                        Math.toRadians(snapshot.pitchDegrees()),
                        -Math.PI / 2.0,
                        Math.PI / 2.0));
        value.put("on_ground", snapshot.onGround());
        value.put("horizontal_collision", snapshot.horizontalCollision());
        value.put("vertical_collision", snapshot.verticalCollision());
        value.put("in_fluid", snapshot.inWater() || snapshot.inLava());
        return value;
    }

    private static List<String> changedMotorFields(
            RawPrimitiveAction requested, PrimitiveAction applied) {
        List<String> fields = new ArrayList<>(5);
        if (Double.compare(requested.forwardThrust(), applied.forwardThrust()) != 0) {
            fields.add("forward_thrust");
        }
        if (Double.compare(requested.lateralThrust(), applied.lateralThrust()) != 0) {
            fields.add("lateral_thrust");
        }
        if (Double.compare(requested.verticalThrust(), applied.verticalThrust()) != 0) {
            fields.add("vertical_thrust");
        }
        if (Double.compare(requested.yawTorque(), applied.yawTorque()) != 0) {
            fields.add("yaw_torque");
        }
        if (Double.compare(requested.pitchTorque(), applied.pitchTorque()) != 0) {
            fields.add("pitch_torque");
        }
        return List.copyOf(fields);
    }

    private static String protocolFault(ControllerFault fault) {
        return switch (fault) {
            case MISSING_CONFIGURATION -> "CONTROL_UNAVAILABLE";
            case DISCONNECTED -> "DISCONNECTED";
            case IDENTITY_MISMATCH -> "IDENTITY_MISMATCH";
            case STALE_COMMAND, DEADLINE_MISSED -> "STALE";
            case FUTURE_COMMAND -> "FUTURE";
            case SEQUENCE_ERROR -> "SEQUENCE";
            case OVERSIZE_MESSAGE -> "OVERSIZE";
            case QUEUE_OVERFLOW -> "QUEUE_OVERFLOW";
            case INTERNAL_ERROR -> "INTERNAL";
            case PROTOCOL_ERROR -> "MALFORMED";
            case STARTING, ACTIVE -> "INVALID_STATE";
        };
    }

    private static void validateControlContent(
            ControlOperation operation, String contentId, String contentHash, String name) {
        if (contentId != null && !IDENTIFIER.matcher(contentId).matches()) {
            throw new IllegalArgumentException(name + " content_id is invalid");
        }
        if (contentHash != null && !HASH.matcher(contentHash).matches()) {
            throw new IllegalArgumentException(name + " content_hash is invalid");
        }
        switch (operation) {
            case PAUSE, RESUME, SINGLE_STEP, RESET -> {
                if (contentId != null || contentHash != null) {
                    throw new IllegalArgumentException(operation + " cannot carry content identity");
                }
            }
            case REPLAY_START, LESION_APPLY, STIMULATION_APPLY -> {
                if (contentId == null || contentHash == null) {
                    throw new IllegalArgumentException(
                            operation + " requires content identity and hash");
                }
            }
            case LESION_CLEAR, STIMULATION_CLEAR -> {
                if (contentId == null) {
                    throw new IllegalArgumentException(operation + " requires content_id");
                }
            }
            default -> {
                if (contentId == null) {
                    throw new IllegalArgumentException(operation + " requires content_id");
                }
            }
        }
    }

    private static List<Double> clampedVector(
            List<String> clampedFields,
            String field,
            double minimum,
            double maximum,
            double... rawValues) {
        List<Double> values = new ArrayList<>(rawValues.length);
        boolean changed = false;
        for (double raw : rawValues) {
            if (!Double.isFinite(raw)) {
                throw new IllegalArgumentException(field + " must be finite");
            }
            double value = clamp(raw, minimum, maximum);
            changed |= Double.compare(raw, value) != 0;
            values.add(value == 0.0 ? 0.0 : value);
        }
        if (changed) {
            clampedFields.add(field);
        }
        return List.copyOf(values);
    }

    private static List<Double> boundedVector(
            String field, double minimum, double maximum, double... rawValues) {
        List<Double> values = new ArrayList<>(rawValues.length);
        for (double raw : rawValues) {
            values.add(boundedNumber(field, raw, minimum, maximum));
        }
        return List.copyOf(values);
    }

    private static double boundedNumber(
            String field, double value, double minimum, double maximum) {
        if (!Double.isFinite(value) || value < minimum || value > maximum) {
            throw new IllegalArgumentException(
                    field + " must be finite and in [" + minimum + ", " + maximum + "]");
        }
        return value == 0.0 ? 0.0 : value;
    }

    private static double clamp(double value, double minimum, double maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }

    private static void requireExactKeys(
            Map<String, Object> values, Set<String> expected, String name)
            throws ProtocolException {
        if (!values.keySet().equals(expected)) {
            throw new ProtocolException(name + " has missing or unknown fields");
        }
    }

    private static Map<String, Object> requireObject(Map<String, Object> values, String key)
            throws ProtocolException {
        Object value = values.get(key);
        if (!(value instanceof Map<?, ?> raw)
                || !raw.keySet().stream().allMatch(String.class::isInstance)) {
            throw new ProtocolException(key + " must be an object");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> result = (Map<String, Object>) raw;
        return result;
    }

    private static String requireString(Map<String, Object> values, String key)
            throws ProtocolException {
        Object value = values.get(key);
        if (!(value instanceof String string)) {
            throw new ProtocolException(key + " must be a string");
        }
        return string;
    }

    private static String requireNullableString(Map<String, Object> values, String key)
            throws ProtocolException {
        Object value = values.get(key);
        if (value == null) {
            return null;
        }
        if (!(value instanceof String string)) {
            throw new ProtocolException(key + " must be a string or null");
        }
        return string;
    }

    private static <E extends Enum<E>> E requireEnum(
            Map<String, Object> values, String key, Class<E> enumType, String name)
            throws ProtocolException {
        String value = requireString(values, key);
        try {
            return Enum.valueOf(enumType, value);
        } catch (IllegalArgumentException exception) {
            throw new ProtocolException(name + " is unknown", exception);
        }
    }

    private static String requireText(Map<String, Object> values, String key, int maximum)
            throws ProtocolException {
        String value = requireString(values, key);
        if (value.isEmpty()
                || value.length() > maximum
                || value.chars().anyMatch(character -> character < 32 || character == 127)) {
            throw new ProtocolException(key + " must be non-empty bounded text without controls");
        }
        return value;
    }

    private static String requireIdentifier(Map<String, Object> values, String key)
            throws ProtocolException {
        String value = requireString(values, key);
        if (!IDENTIFIER.matcher(value).matches()) {
            throw new ProtocolException(key + " is not a valid identifier");
        }
        return value;
    }

    private static String requireHash(Map<String, Object> values, String key)
            throws ProtocolException {
        String value = requireString(values, key);
        if (!HASH.matcher(value).matches()) {
            throw new ProtocolException(key + " must be a lowercase SHA-256 digest");
        }
        return value;
    }

    private static boolean requireBoolean(Map<String, Object> values, String key)
            throws ProtocolException {
        Object value = values.get(key);
        if (!(value instanceof Boolean bool)) {
            throw new ProtocolException(key + " must be boolean");
        }
        return bool;
    }

    private static long requireNonNegativeLong(Map<String, Object> values, String key)
            throws ProtocolException {
        return requireNonNegativeLongValue(values.get(key), key);
    }

    private static long requireNonNegativeLongValue(Object value, String key)
            throws ProtocolException {
        if (!(value instanceof BigDecimal number)) {
            throw new ProtocolException(key + " must be an integer");
        }
        try {
            long result = number.longValueExact();
            if (result < 0) {
                throw new ProtocolException(key + " must be non-negative");
            }
            return result;
        } catch (ArithmeticException exception) {
            throw new ProtocolException(key + " is outside the integer range", exception);
        }
    }

    private static double requireFiniteNumber(
            Map<String, Object> values, String key, double minimum, double maximum)
            throws ProtocolException {
        Object value = values.get(key);
        if (!(value instanceof BigDecimal number)) {
            throw new ProtocolException(key + " must be numeric");
        }
        double result = number.doubleValue();
        if (!Double.isFinite(result) || result < minimum || result > maximum) {
            throw new ProtocolException(
                    key + " must be finite and in [" + minimum + ", " + maximum + "]");
        }
        return result == 0.0 ? 0.0 : result;
    }
}
