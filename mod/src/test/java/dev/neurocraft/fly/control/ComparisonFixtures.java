package dev.neurocraft.fly.control;

import java.lang.reflect.Proxy;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Properties;

final class ComparisonFixtures {
    static final long EPOCH = 1_783_000_000_000_000_007L;
    static final ControllerIdentity INITIAL = new ControllerIdentity("comparison-test", EPOCH,
            "00000000-0000-0000-0000-000000000001", "1".repeat(64), "2".repeat(64),
            "3".repeat(64), "4".repeat(64), "5".repeat(64));

    static LaneCatalog catalog(Path directory) throws Exception {
        List<Object> rows = new ArrayList<>();
        for (String name : List.of("original", "shuffled_weight", "no_input")) {
            boolean original = name.equals("original");
            Map<String, Object> identity = new LinkedHashMap<>(id());
            identity.put("dynamics_hash", (original ? "3" : name.equals("no_input") ? "7" : "6").repeat(64));
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("name", name);
            row.put("identity", identity);
            row.put("control_kind", original ? "faithful" : name);
            row.put("control_sha256", original ? null : "8".repeat(64));
            row.put("seed", original ? null : "18446744073709551615");
            rows.add(row);
        }
        byte[] bytes = StrictJson.encode(Map.of("schema", LaneCatalog.SCHEMA,
                "initial_identity", id(), "lanes", rows), LaneCatalog.MAX_BYTES)
                .getBytes(java.nio.charset.StandardCharsets.UTF_8);
        Path path = directory.resolve("catalog.json");
        Files.write(path, bytes);
        return LaneCatalog.load(path, HexFormat.of().formatHex(
                MessageDigest.getInstance("SHA-256").digest(bytes)), INITIAL);
    }

    static BridgeConfig config() {
        Properties properties = new Properties();
        Map<String, String> values = Map.of("runId", INITIAL.runId(), "sessionEpoch", Long.toString(EPOCH),
                "entityId", INITIAL.entityId(), "graphHash", INITIAL.graphHash(),
                "adapterHash", INITIAL.adapterHash(), "dynamicsHash", INITIAL.dynamicsHash(),
                "decoderHash", INITIAL.decoderHash(), "observationSchemaHash", INITIAL.observationSchemaHash());
        values.forEach((key, value) -> properties.setProperty("neurocraft.fly." + key, value));
        return BridgeConfig.from(Map.of("NEUROCRAFT_RUN_TOKEN", "T".repeat(43)), properties);
    }

    static ActionFrame command(ControllerIdentity identity) {
        return new ActionFrame(identity, 0, 0, 100, 102, 102, 0, 5,
                "a".repeat(64), new RawPrimitiveAction(0.2, 0, 0, 0, 0), false);
    }

    static BrainDisplayPresentation display(ControllerIdentity identity) {
        Map<String, BrainDisplayFrame.Population> populations = new LinkedHashMap<>();
        for (String name : BrainDisplayFrame.POPULATION_NAMES) {
            populations.put(name, new BrainDisplayFrame.Population(0.1, 1));
        }
        return new BrainDisplayPresentation(BrainDisplayState.Status.MATCHED, 0,
                Optional.of(new BrainDisplayFrame(identity, 0, 100, 0, 5,
                        "a".repeat(64), "test", "b".repeat(64), populations)));
    }

    private static Map<String, Object> id() {
        return Map.of("run_id", INITIAL.runId(), "session_epoch", EPOCH, "entity_id", INITIAL.entityId(),
                "graph_hash", INITIAL.graphHash(), "adapter_hash", INITIAL.adapterHash(),
                "dynamics_hash", INITIAL.dynamicsHash(), "decoder_hash", INITIAL.decoderHash(),
                "observation_schema_hash", INITIAL.observationSchemaHash());
    }

    static final class Probe {
        ControllerFault state = ControllerFault.STARTING;
        boolean closed;
        boolean failStart;
        final ArrayDeque<CommandFrame> commands = new ArrayDeque<>();
        final ControllerBridge bridge = (ControllerBridge) Proxy.newProxyInstance(
                ControllerBridge.class.getClassLoader(), new Class<?>[] {ControllerBridge.class},
                (proxy, method, arguments) -> switch (method.getName()) {
                    case "state" -> state;
                    case "pendingTerminalFault" -> Optional.empty();
                    case "pollAction" -> Optional.ofNullable(commands.pollFirst());
                    case "start" -> {
                        if (failStart) { throw new IllegalStateException("synthetic start failure"); }
                        yield null;
                    }
                    case "close" -> { closed = true; commands.clear(); yield null; }
                    case "failClosed" -> { state = (ControllerFault) arguments[0]; yield null; }
                    default -> throw new AssertionError("unexpected bridge method: " + method.getName());
                });
    }
}
