package dev.neurocraft.fly.control;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

final class LaneCatalogTest {
    @TempDir Path directory;
    private static final ControllerIdentity INITIAL = new ControllerIdentity(
            "lane-run", 7, "00000000-0000-0000-0000-000000000001", "1".repeat(64),
            "2".repeat(64), "3".repeat(64), "4".repeat(64), "5".repeat(64));

    @Test
    void verifiesAllLanesOnceAndResolvesWithoutReopeningTheFile() throws Exception {
        Path path = write(material());
        LaneCatalog catalog = LaneCatalog.load(path, digest(Files.readAllBytes(path)), INITIAL);
        Files.delete(path);
        assertEquals(INITIAL, catalog.entry("original").identity());
        assertEquals("faithful", catalog.entry("original").controlKind());
        assertNull(catalog.entry("original").controlSha256());
        assertNull(catalog.entry("original").seed());
        assertEquals("0", catalog.entry("shuffled_weight").seed());
        assertEquals("18446744073709551615", catalog.entry("no_input").seed());
        assertEquals("8".repeat(64), catalog.entry("no_input").controlSha256());
        assertEquals("no_input", catalog.entry("no_input").name());
        ControllerIdentity next = catalog.resolve("shuffled_weight", 8);
        assertEquals(8, next.sessionEpoch());
        assertEquals("6".repeat(64), next.dynamicsHash());
        assertEquals(new ControllerIdentity(INITIAL.runId(), 9, INITIAL.entityId(),
                INITIAL.graphHash(), INITIAL.adapterHash(), INITIAL.dynamicsHash(),
                INITIAL.decoderHash(), INITIAL.observationSchemaHash()), catalog.resolve("original", 9));
        assertEquals(INITIAL, catalog.entry("original").identity());
        assertEquals(Long.MAX_VALUE, catalog.entry("no_input").withEpoch(Long.MAX_VALUE).sessionEpoch());
        for (long epoch : new long[] {-1, 0, 7}) {
            assertThrows(IllegalArgumentException.class, () -> catalog.resolve("original", epoch));
        }
        assertThrows(IllegalArgumentException.class, () -> catalog.entry("unknown"));
        assertThrows(UnsupportedOperationException.class, () -> LaneCatalog.NAMES.add("other"));
    }

    @Test
    void rejectsDigestMismatchOversizeMalformedUtf8AndDuplicateKeys() throws Exception {
        Path path = write(material());
        String json = Files.readString(path);
        String originalDigest = digest(Files.readAllBytes(path));
        assertThrows(ProtocolException.class, () -> LaneCatalog.load(path, "0".repeat(64), INITIAL));
        assertThrows(ProtocolException.class, () -> LaneCatalog.load(path, "ABC", INITIAL));
        Files.writeString(path, json + " ".repeat(LaneCatalog.MAX_BYTES - json.length()));
        assertThrows(ProtocolException.class, () -> LaneCatalog.load(path, originalDigest, INITIAL));
        assertEquals(INITIAL, LaneCatalog.load(path, digest(Files.readAllBytes(path)), INITIAL)
                .entry("original").identity());
        Files.writeString(path, " ".repeat(LaneCatalog.MAX_BYTES + 1));
        String oversizedDigest = digest(Files.readAllBytes(path));
        assertThrows(ProtocolException.class, () -> LaneCatalog.load(path, oversizedDigest, INITIAL));
        Files.write(path, new byte[] {(byte) 0xc3, (byte) 0x28});
        String invalidUtf8Digest = digest(Files.readAllBytes(path));
        assertThrows(java.io.IOException.class, () -> LaneCatalog.load(path, invalidUtf8Digest, INITIAL));
        for (String malformed : List.of(
                json.replaceFirst("\\{", "{\"schema\":\"neurocraft.lane_catalog.v1\","),
                json.replace("\"name\":\"original\"", "\"name\":\"original\",\"name\":\"original\""),
                json + "{}")) {
            Files.writeString(path, malformed);
            String hash = digest(Files.readAllBytes(path));
            assertThrows(ProtocolException.class, () -> LaneCatalog.load(path, hash, INITIAL));
        }
    }

    @Test
    void rejectsMissingUnknownDuplicateAndMislabeledLanes() throws Exception {
        List<Consumer<Map<String, Object>>> mutations = List.of(
                root -> root.put("extra", true),
                root -> root.remove("schema"),
                root -> root.put("schema", "neurocraft.lane_catalog.v2"),
                root -> root.put("lanes", List.of()),
                root -> root.put("lanes", List.of(row(root, 0), row(root, 1))),
                root -> row(root, 2).put("name", "shuffled_weight"),
                root -> row(root, 2).put("name", "rewired"),
                root -> row(root, 0).put("control_kind", "original"),
                root -> row(root, 1).put("control_kind", "no_input"),
                root -> row(root, 0).put("seed", "7"),
                root -> row(root, 0).put("control_sha256", "9".repeat(64)),
                root -> row(root, 1).remove("seed"),
                root -> row(root, 1).put("extra", 0),
                root -> laneIdentity(root, 0).put("dynamics_hash", "9".repeat(64)),
                root -> laneIdentity(root, 1).put("dynamics_hash", INITIAL.dynamicsHash()),
                root -> laneIdentity(root, 2).put("dynamics_hash", "6".repeat(64)));
        for (Consumer<Map<String, Object>> mutation : mutations) {
            reject(mutation);
        }
    }

    @Test
    void rejectsEveryChangedFixedBoundaryAndInvalidEpoch() throws Exception {
        for (String field : List.of("run_id", "entity_id", "graph_hash", "adapter_hash",
                "decoder_hash", "observation_schema_hash")) {
            String different = field.endsWith("_hash") ? "a".repeat(64) : "other";
            reject(root -> laneIdentity(root, 1).put(field, different));
            reject(root -> object(root.get("initial_identity")).put(field, different));
        }
        for (Object epoch : List.of(true, "7", -1, 8, new BigDecimal("7.5"),
                new BigDecimal("9223372036854775808"))) {
            reject(root -> laneIdentity(root, 1).put("session_epoch", epoch));
            reject(root -> object(root.get("initial_identity")).put("session_epoch", epoch));
        }
        reject(root -> laneIdentity(root, 1).remove("graph_hash"));
        reject(root -> laneIdentity(root, 1).put("extra", false));
    }

    @Test
    void requiresExactControlDigestAndDecimalUint64String() throws Exception {
        for (Object seed : List.of(7, true, "", "-1", "+1", "01", "1.0", "1e2", " 7",
                "18446744073709551616", "999999999999999999999")) {
            reject(root -> row(root, 1).put("seed", seed));
        }
        reject(root -> row(root, 1).put("seed", null));
        for (Object hash : List.of(7, "", "a".repeat(63), "A".repeat(64), "g".repeat(64))) {
            reject(root -> row(root, 1).put("control_sha256", hash));
        }
        reject(root -> row(root, 1).put("control_sha256", null));
    }

    @Test
    void refusesIntegralDecimalAndExponentEpochTokensIncludingEscapedKeys() throws Exception {
        Path path = write(material());
        String json = Files.readString(path);
        for (String key : List.of("session_epoch", "session\\u005fepoch")) {
            for (String token : List.of("7.0", "7e0", "7E+0")) {
                Files.writeString(path, json.replace("\"session_epoch\":7", "\"" + key + "\":" + token));
                String sha256 = digest(Files.readAllBytes(path));
                assertThrows(ProtocolException.class, () -> LaneCatalog.load(path, sha256, INITIAL));
            }
        }
        Files.writeString(path, json.replace("session_epoch", "session\\u005fepoch"));
        assertEquals(INITIAL, LaneCatalog.load(path, digest(Files.readAllBytes(path)), INITIAL)
                .entry("original").identity());
    }

    private void reject(Consumer<Map<String, Object>> mutation) throws Exception {
        Map<String, Object> root = material();
        mutation.accept(root);
        Path path = write(root);
        String sha256 = digest(Files.readAllBytes(path));
        assertThrows(ProtocolException.class, () -> LaneCatalog.load(path, sha256, INITIAL));
    }

    private Path write(Map<String, Object> root) throws Exception {
        Path path = directory.resolve("catalog.json");
        Files.writeString(path, StrictJson.encode(root, LaneCatalog.MAX_BYTES), StandardCharsets.UTF_8);
        return path;
    }

    private static String digest(byte[] bytes) throws Exception {
        return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
    }

    private static Map<String, Object> material() {
        List<Object> lanes = new ArrayList<>();
        for (String name : List.of("original", "shuffled_weight", "no_input")) {
            boolean original = name.equals("original");
            Map<String, Object> identity = identity();
            identity.put("dynamics_hash", original ? INITIAL.dynamicsHash()
                    : (name.equals("shuffled_weight") ? "6" : "7").repeat(64));
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("name", name);
            row.put("identity", identity);
            row.put("control_kind", original ? "faithful" : name);
            row.put("control_sha256", original ? null : "8".repeat(64));
            row.put("seed", original ? null : name.equals("shuffled_weight")
                    ? "0" : "18446744073709551615");
            lanes.add(row);
        }
        return new LinkedHashMap<>(Map.of(
                "schema", LaneCatalog.SCHEMA, "initial_identity", identity(), "lanes", lanes));
    }

    private static Map<String, Object> identity() {
        return new LinkedHashMap<>(Map.of(
                "run_id", INITIAL.runId(), "session_epoch", INITIAL.sessionEpoch(),
                "entity_id", INITIAL.entityId(), "graph_hash", INITIAL.graphHash(),
                "adapter_hash", INITIAL.adapterHash(), "dynamics_hash", INITIAL.dynamicsHash(),
                "decoder_hash", INITIAL.decoderHash(), "observation_schema_hash", INITIAL.observationSchemaHash()));
    }

    private static Map<String, Object> row(Map<String, Object> root, int index) {
        return object(((List<?>) root.get("lanes")).get(index));
    }

    private static Map<String, Object> laneIdentity(Map<String, Object> root, int index) {
        return object(row(root, index).get("identity"));
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> object(Object value) {
        return (Map<String, Object>) value;
    }
}
