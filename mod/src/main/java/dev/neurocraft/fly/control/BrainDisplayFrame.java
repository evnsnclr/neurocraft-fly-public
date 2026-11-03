package dev.neurocraft.fly.control;

import java.math.BigDecimal;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/** Six raw model-activity means from a local diagnostic file; never a motor command. */
public record BrainDisplayFrame(
        ControllerIdentity identity,
        long commandSequence,
        long observationTick,
        long neuralStepStart,
        long neuralStepEnd,
        String stateDigest,
        String encoderSchema,
        String bindingHash,
        Map<String, Population> populations) {
    public static final String SCHEMA = "neurocraft.brain_display.v1";
    public static final int MAX_BYTES = 16 * 1024;
    public static final List<String> POPULATION_NAMES = List.of(
            "lc4_left", "lc4_right", "dnp01_left", "dnp01_right", "ttmn_left", "ttmn_right");
    private static final Set<String> KEYS = Set.of(
            "schema", "presentation_only", "identity", "command_seq", "observation_tick",
            "neural_step_start", "neural_step_end", "state_digest", "encoder_schema",
            "binding_hash", "populations");
    private static final Set<String> IDENTITY_KEYS = Set.of(
            "run_id", "session_epoch", "entity_id", "graph_hash", "adapter_hash",
            "dynamics_hash", "decoder_hash", "observation_schema_hash");

    public record Population(double mean, int count) {
        public Population {
            if (!Double.isFinite(mean) || mean < -1.0 || mean > 1.0) {
                throw new IllegalArgumentException("population mean must be finite in [-1,1]");
            }
            if (count < 1 || count > 166_700) {
                throw new IllegalArgumentException("population count must be within 1..166700");
            }
        }
    }

    public BrainDisplayFrame {
        Objects.requireNonNull(identity, "identity");
        if (commandSequence < 0 || observationTick < 0 || neuralStepStart < 0
                || neuralStepEnd <= neuralStepStart) {
            throw new IllegalArgumentException("display frame requires a completed neural step");
        }
        if (stateDigest == null || !stateDigest.matches("[0-9a-f]{64}")
                || bindingHash == null || !bindingHash.matches("[0-9a-f]{64}")) {
            throw new IllegalArgumentException("display hashes must be lowercase SHA-256");
        }
        if (encoderSchema == null || !encoderSchema.matches("[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")) {
            throw new IllegalArgumentException("invalid encoder schema identifier");
        }
        populations = Map.copyOf(populations);
        if (!populations.keySet().equals(Set.copyOf(POPULATION_NAMES))) {
            throw new IllegalArgumentException("display must name exactly the six populations");
        }
    }

    /** Strict bounded decoding uses the existing codec, without touching command validation. */
    public static BrainDisplayFrame parse(String json) throws ProtocolException {
        Map<String, Object> root = StrictJson.parseObject(json, MAX_BYTES);
        exact(root, KEYS);
        if (!SCHEMA.equals(root.get("schema")) || !Boolean.TRUE.equals(root.get("presentation_only"))) {
            throw new ProtocolException("not a supported presentation-only brain display frame");
        }
        Map<String, Object> id = object(root, "identity");
        exact(id, IDENTITY_KEYS);
        Map<String, Object> rawPopulations = object(root, "populations");
        exact(rawPopulations, Set.copyOf(POPULATION_NAMES));
        try {
            ControllerIdentity identity = new ControllerIdentity(
                    text(id, "run_id"), integer(id, "session_epoch"), text(id, "entity_id"),
                    text(id, "graph_hash"), text(id, "adapter_hash"), text(id, "dynamics_hash"),
                    text(id, "decoder_hash"), text(id, "observation_schema_hash"));
            Map<String, Population> values = new LinkedHashMap<>();
            for (String name : POPULATION_NAMES) {
                Map<String, Object> population = object(rawPopulations, name);
                exact(population, Set.of("mean", "count"));
                Object mean = population.get("mean");
                if (!(mean instanceof BigDecimal number)) {
                    throw new ProtocolException("population mean must be numeric");
                }
                values.put(name, new Population(number.doubleValue(),
                        Math.toIntExact(integer(population, "count"))));
            }
            return new BrainDisplayFrame(identity, integer(root, "command_seq"),
                    integer(root, "observation_tick"), integer(root, "neural_step_start"),
                    integer(root, "neural_step_end"), text(root, "state_digest"),
                    text(root, "encoder_schema"), text(root, "binding_hash"), values);
        } catch (IllegalArgumentException | ArithmeticException exception) {
            throw new ProtocolException("invalid brain display fields", exception);
        }
    }

    /** Equality of recorded envelope fields is a display consistency check, not authentication. */
    public boolean matches(CommandFrame command) {
        return identity.equals(command.identity()) && commandSequence == command.commandSequence()
                && observationTick == command.observationTick()
                && neuralStepStart == command.neuralStepStart()
                && neuralStepEnd == command.neuralStepEnd()
                && stateDigest.equals(command.stateDigest());
    }

    Map<String, Object> toMap() {
        Map<String, Object> id = Map.of(
                "run_id", identity.runId(), "session_epoch", identity.sessionEpoch(),
                "entity_id", identity.entityId(), "graph_hash", identity.graphHash(),
                "adapter_hash", identity.adapterHash(), "dynamics_hash", identity.dynamicsHash(),
                "decoder_hash", identity.decoderHash(),
                "observation_schema_hash", identity.observationSchemaHash());
        Map<String, Object> values = new LinkedHashMap<>();
        values.put("schema", SCHEMA);
        values.put("presentation_only", true);
        values.put("identity", id);
        values.put("command_seq", commandSequence);
        values.put("observation_tick", observationTick);
        values.put("neural_step_start", neuralStepStart);
        values.put("neural_step_end", neuralStepEnd);
        values.put("state_digest", stateDigest);
        values.put("encoder_schema", encoderSchema);
        values.put("binding_hash", bindingHash);
        Map<String, Object> means = new LinkedHashMap<>();
        for (String name : POPULATION_NAMES) {
            Population population = populations.get(name);
            means.put(name, Map.of("mean", population.mean(), "count", population.count()));
        }
        values.put("populations", means);
        return values;
    }

    static void exact(Map<String, Object> values, Set<String> keys) throws ProtocolException {
        if (!values.keySet().equals(keys)) {
            throw new ProtocolException("brain display object has missing or unknown fields");
        }
    }

    static Map<String, Object> object(Map<String, Object> parent, String key)
            throws ProtocolException {
        if (!(parent.get(key) instanceof Map<?, ?> raw)) {
            throw new ProtocolException(key + " must be an object");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> result = (Map<String, Object>) raw;
        return result;
    }

    static String text(Map<String, Object> values, String key) throws ProtocolException {
        if (!(values.get(key) instanceof String value)) {
            throw new ProtocolException(key + " must be a string");
        }
        return value;
    }

    static long integer(Map<String, Object> values, String key) throws ProtocolException {
        if (!(values.get(key) instanceof BigDecimal value)) {
            throw new ProtocolException(key + " must be an integer");
        }
        try {
            long result = value.longValueExact();
            if (result < 0) {
                throw new ProtocolException(key + " must be nonnegative");
            }
            return result;
        } catch (ArithmeticException exception) {
            throw new ProtocolException(key + " exceeds the integer range", exception);
        }
    }
}
