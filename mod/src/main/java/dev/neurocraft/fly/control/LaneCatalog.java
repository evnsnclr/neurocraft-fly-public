package dev.neurocraft.fly.control;

import java.io.IOException;
import java.math.BigDecimal;
import java.nio.ByteBuffer;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/** A digest-pinned local lane catalog, loaded once; peer authentication remains the HMAC handshake. */
public final class LaneCatalog {
    public static final String SCHEMA = "neurocraft.lane_catalog.v1";
    public static final int MAX_BYTES = 16 * 1024;
    public static final Set<String> NAMES = Set.of("original", "shuffled_weight", "no_input");
    private static final Set<String> ROOT_KEYS = Set.of("schema", "initial_identity", "lanes");
    private static final Set<String> ENTRY_KEYS = Set.of(
            "name", "identity", "control_kind", "control_sha256", "seed");
    private static final Set<String> IDENTITY_KEYS = Set.of(
            "run_id", "session_epoch", "entity_id", "graph_hash", "adapter_hash", "dynamics_hash",
            "decoder_hash", "observation_schema_hash");
    private static final String MAX_SEED = "18446744073709551615";

    private final Map<String, Entry> entries;

    private LaneCatalog(Map<String, Entry> entries) {
        this.entries = Map.copyOf(entries);
    }

    public static LaneCatalog load(
            Path path, String expectedSha256, ControllerIdentity initialIdentity)
            throws IOException, ProtocolException {
        Objects.requireNonNull(initialIdentity, "initialIdentity");
        requireHash(expectedSha256, "catalog digest");
        byte[] bytes;
        try (var stream = Files.newInputStream(path)) {
            bytes = stream.readNBytes(MAX_BYTES + 1);
        }
        if (bytes.length > MAX_BYTES) {
            throw new ProtocolException("lane catalog exceeds 16 KiB");
        }
        try {
            String actual = HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
            if (!actual.equals(expectedSha256)) {
                throw new ProtocolException("lane catalog digest does not match configured bytes");
            }
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException("JDK SHA-256 is unavailable", exception);
        }
        String json = StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
                .decode(ByteBuffer.wrap(bytes)).toString();
        Map<String, Object> root = StrictJson.parseObject(json, MAX_BYTES);
        requireIntegerEpochTokens(json);
        exact(root, ROOT_KEYS);
        if (!SCHEMA.equals(root.get("schema"))) {
            throw new ProtocolException("unsupported lane catalog schema");
        }
        if (!identity(root.get("initial_identity")).equals(initialIdentity)) {
            throw new ProtocolException("lane catalog initial identity does not match bridge");
        }
        if (!(root.get("lanes") instanceof List<?> lanes) || lanes.size() != NAMES.size()) {
            throw new ProtocolException("lane catalog requires exactly three lanes");
        }
        Map<String, Entry> entries = new HashMap<>();
        Set<String> dynamics = new HashSet<>();
        for (Object lane : lanes) {
            Map<String, Object> row = object(lane);
            exact(row, ENTRY_KEYS);
            String name = text(row, "name");
            if (!NAMES.contains(name) || entries.containsKey(name)) {
                throw new ProtocolException("unknown or duplicate catalog lane");
            }
            ControllerIdentity identity = identity(row.get("identity"));
            if (!sameBoundary(initialIdentity, identity)
                    || identity.sessionEpoch() != initialIdentity.sessionEpoch()) {
                throw new ProtocolException("catalog lane changes a fixed identity field");
            }
            if (!dynamics.add(identity.dynamicsHash())) {
                throw new ProtocolException("catalog lanes require distinct dynamics identities");
            }
            String kind = text(row, "control_kind");
            String controlHash = null;
            String seed = null;
            if (name.equals("original")) {
                if (!kind.equals("faithful") || !identity.equals(initialIdentity)
                        || row.get("control_sha256") != null || row.get("seed") != null) {
                    throw new ProtocolException("original lane must retain the initial faithful identity");
                }
            } else {
                if (!kind.equals(name)) {
                    throw new ProtocolException("control kind does not match lane name");
                }
                controlHash = text(row, "control_sha256");
                requireHash(controlHash, "control digest");
                seed = text(row, "seed");
                if (!seed.matches("0|[1-9][0-9]{0,19}")
                        || (seed.length() == MAX_SEED.length() && seed.compareTo(MAX_SEED) > 0)) {
                    throw new ProtocolException("control seed must be a canonical decimal uint64 string");
                }
            }
            entries.put(name, new Entry(name, identity, kind, controlHash, seed));
        }
        return new LaneCatalog(entries);
    }

    public Entry entry(String name) {
        Entry result = entries.get(name);
        if (result == null) {
            throw new IllegalArgumentException("unknown catalog lane");
        }
        return result;
    }

    public ControllerIdentity resolve(String name, long epoch) {
        return entry(name).withEpoch(epoch);
    }

    static boolean sameBoundary(ControllerIdentity first, ControllerIdentity second) {
        return first.runId().equals(second.runId()) && first.entityId().equals(second.entityId())
                && first.graphHash().equals(second.graphHash())
                && first.adapterHash().equals(second.adapterHash())
                && first.decoderHash().equals(second.decoderHash())
                && first.observationSchemaHash().equals(second.observationSchemaHash());
    }

    /** Immutable verified declaration; only the session epoch can change at handover. */
    public static final class Entry {
        private final String name;
        private final ControllerIdentity identity;
        private final String controlKind;
        private final String controlSha256;
        private final String seed;

        private Entry(String name, ControllerIdentity identity, String kind, String hash, String seed) {
            this.name = name;
            this.identity = identity;
            this.controlKind = kind;
            this.controlSha256 = hash;
            this.seed = seed;
        }

        public String name() { return name; }
        public ControllerIdentity identity() { return identity; }
        public String controlKind() { return controlKind; }
        public String controlSha256() { return controlSha256; }
        public String seed() { return seed; }

        public ControllerIdentity withEpoch(long epoch) {
            if (epoch <= identity.sessionEpoch()) {
                throw new IllegalArgumentException("handover epoch must exceed the catalog base epoch");
            }
            return new ControllerIdentity(identity.runId(), epoch, identity.entityId(),
                    identity.graphHash(), identity.adapterHash(), identity.dynamicsHash(),
                    identity.decoderHash(), identity.observationSchemaHash());
        }
    }

    private static ControllerIdentity identity(Object value) throws ProtocolException {
        Map<String, Object> id = object(value);
        exact(id, IDENTITY_KEYS);
        if (!(id.get("session_epoch") instanceof BigDecimal epoch)) {
            throw new ProtocolException("session epoch must be an integer");
        }
        try {
            return new ControllerIdentity(text(id, "run_id"), epoch.longValueExact(),
                    text(id, "entity_id"), text(id, "graph_hash"), text(id, "adapter_hash"),
                    text(id, "dynamics_hash"), text(id, "decoder_hash"), text(id, "observation_schema_hash"));
        } catch (IllegalArgumentException | ArithmeticException exception) {
            throw new ProtocolException("invalid catalog identity", exception);
        }
    }

    private static void requireIntegerEpochTokens(String json) throws ProtocolException {
        // StrictJson intentionally normalizes BigDecimal values. Catalog epochs
        // must also match Python's integer token type, so inspect the already
        // validated JSON tokens without changing the shared wire codec.
        for (int position = 0; position < json.length(); position++) {
            if (json.charAt(position) != '"') {
                continue;
            }
            int start = position++;
            while (json.charAt(position) != '"') {
                if (json.charAt(position) == '\\') {
                    position++;
                }
                position++;
            }
            int end = position + 1;
            int colon = skipWhitespace(json, end);
            if (colon >= json.length() || json.charAt(colon) != ':') {
                continue;
            }
            // Reuse the bounded decoder for escaped key names as well; an
            // escaped underscore must not bypass the lexical integer check.
            Map<String, Object> key = StrictJson.parseObject(
                    "{" + json.substring(start, end) + ":null}", MAX_BYTES);
            if (!key.containsKey("session_epoch")) {
                continue;
            }
            int numberStart = skipWhitespace(json, colon + 1);
            int numberEnd = numberStart;
            while (numberEnd < json.length()
                    && !Character.isWhitespace(json.charAt(numberEnd))
                    && ",}]".indexOf(json.charAt(numberEnd)) < 0) {
                numberEnd++;
            }
            if (!json.substring(numberStart, numberEnd).matches("-?(0|[1-9][0-9]*)")) {
                throw new ProtocolException("session epoch must use an integer JSON token");
            }
        }
    }

    private static int skipWhitespace(String json, int position) {
        while (position < json.length() && Character.isWhitespace(json.charAt(position))) {
            position++;
        }
        return position;
    }

    private static Map<String, Object> object(Object value) throws ProtocolException {
        if (!(value instanceof Map<?, ?> raw)) {
            throw new ProtocolException("catalog value must be an object");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> result = (Map<String, Object>) raw;
        return result;
    }

    private static void exact(Map<String, Object> value, Set<String> keys) throws ProtocolException {
        if (!value.keySet().equals(keys)) {
            throw new ProtocolException("catalog object has missing or unknown fields");
        }
    }

    private static String text(Map<String, Object> value, String key) throws ProtocolException {
        if (!(value.get(key) instanceof String text)) {
            throw new ProtocolException("catalog " + key + " must be a string");
        }
        return text;
    }

    private static void requireHash(String value, String name) throws ProtocolException {
        if (value == null || !value.matches("[0-9a-f]{64}")) {
            throw new ProtocolException(name + " must be a lowercase SHA-256 digest");
        }
    }
}
