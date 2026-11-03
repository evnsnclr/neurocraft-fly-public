package dev.neurocraft.fly.gametest;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import dev.neurocraft.fly.FlyControllerRuntime;
import dev.neurocraft.fly.NeuroCraftFly;
import dev.neurocraft.fly.control.BridgeConfig;
import dev.neurocraft.fly.control.CommandFrame;
import dev.neurocraft.fly.control.ComparisonLaneSession;
import dev.neurocraft.fly.control.ControlProtocol;
import dev.neurocraft.fly.control.EventDisplayFrame;
import dev.neurocraft.fly.control.EventDisplayPresentation;
import dev.neurocraft.fly.control.EventDisplayReader;
import dev.neurocraft.fly.control.EventDisplayState;
import dev.neurocraft.fly.control.LaneCatalog;
import dev.neurocraft.fly.control.ProgramActionFrame;
import dev.neurocraft.fly.entity.FlyEntity;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import net.minecraft.gametest.framework.GameTestHelper;
import net.minecraft.network.chat.Component;
import net.minecraft.world.phys.Vec3;

/** Real runtime reader startup and presentation, using fixture commands rather than live neural evidence. */
final class EventDisplayRuntimeFence {
    private static final String PROPERTY = "neurocraft.fly.eventDisplayPath";

    private EventDisplayRuntimeFence() {}

    static void verify(GameTestHelper helper) {
        Map<Field, Object> saved = new LinkedHashMap<>();
        String previousPath = System.getProperty(PROPERTY);
        Path directory = null;
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(4, 3, 4));
        FlyEntity foreign = helper.spawn(NeuroCraftFly.FLY, new Vec3(6, 3, 4));
        try {
            for (String name : List.of("eventDisplayReader", "eventDisplayState", "comparison",
                    "epochDiagnosticsDirectory")) {
                Field field = runtimeField(name);
                saved.put(field, field.get(null));
                field.set(null, null);
            }
            directory = Files.createTempDirectory("neurocraft-event-runtime-");
            for (String hash : List.of(ControlProtocol.OBSERVATION_SCHEMA_V3_SHA256,
                    ControlProtocol.OBSERVATION_SCHEMA_V4_SHA256, "5".repeat(64),
                    ControlProtocol.OBSERVATION_SCHEMA_V2_SHA256, "f".repeat(64))) {
                String resource = hash.equals(ControlProtocol.OBSERVATION_SCHEMA_V4_SHA256)
                        ? "interaction-display-v2.json" : "event-display.json";
                JsonObject root;
                try (var stream = EventDisplayRuntimeFence.class.getResourceAsStream("/neurocraft/" + resource)) {
                    if (stream == null) { throw new AssertionError("missing event display fixture"); }
                    root = JsonParser.parseString(new String(stream.readAllBytes(), StandardCharsets.UTF_8)).getAsJsonObject();
                }
                for (JsonObject identity : List.of(root.getAsJsonObject("identity"),
                        root.getAsJsonObject("issued_trigger").getAsJsonObject("identity"),
                        root.getAsJsonObject("issued_trigger").getAsJsonObject("issued_command").getAsJsonObject("identity"))) {
                    identity.addProperty("entity_id", fly.getStringUUID());
                    identity.addProperty("observation_schema_hash", hash);
                }
                EventDisplayFrame frame = EventDisplayFrame.parse(root.toString());
                boolean supported = hash.equals(ControlProtocol.OBSERVATION_SCHEMA_V3_SHA256)
                        || hash.equals(ControlProtocol.OBSERVATION_SCHEMA_V4_SHA256);
                for (boolean epochPath : List.of(false, true)) {
                    try (var installed = RecordingBridgeFence.install(fly, frame.identity())) {
                        BridgeConfig config = (BridgeConfig) runtimeField("config").get(null);
                        runtimeField("comparison").set(null, epochPath ? new ComparisonLaneSession(
                                catalog(directory, root), config, installed.bridge(),
                                ignored -> { throw new AssertionError("fixture must not open a socket"); },
                                System::nanoTime) : null);
                        runtimeField("epochDiagnosticsDirectory").set(null, epochPath ? directory : null);
                        Path path = epochPath ? directory.resolve(Long.toString(frame.identity().sessionEpoch()))
                                .resolve("event-selection-display.json") : directory.resolve("standalone.json");
                        Files.createDirectories(path.getParent());
                        Files.writeString(path, root.toString());
                        System.setProperty(PROPERTY, (epochPath ? directory.resolve("wrong-path.json") : path).toString());
                        invokeRuntime("startEventDisplay");
                        EventDisplayReader reader = (EventDisplayReader) runtimeField("eventDisplayReader").get(null);
                        check((reader != null) == supported, hash + ": reader initialization eligibility");
                        if (!supported) {
                            check(FlyControllerRuntime.eventDisplayFor(fly, frame.command().targetTick()).isEmpty(),
                                    "legacy/unknown observation schema enabled event presentation");
                            continue;
                        }
                        pollOnWorker(reader);
                        check(reader.snapshot().status() == EventDisplayReader.Status.AVAILABLE, "actual sidecar not read");
                        check(status(fly, frame.command().targetTick()) == EventDisplayState.Status.WAITING_FOR_APPLIED_COMMAND,
                                "read file claimed application without a matching command");
                        Method record = FlyControllerRuntime.class.getDeclaredMethod("recordDisplayedApplication", CommandFrame.class);
                        record.setAccessible(true);
                        ProgramActionFrame command = frame.command();
                        record.invoke(null, new ProgramActionFrame(command.identity(), command.commandSequence(),
                                command.observationSequence(), command.observationTick(), command.targetTick(),
                                command.validThroughTick(), command.neuralStepStart(), command.neuralStepEnd(),
                                "0".repeat(64), command.program(), command.modulation(), command.parameters(),
                                command.readout(), command.policyHash()));
                        check(status(fly, command.targetTick()) == EventDisplayState.Status.ENVELOPE_MISMATCH,
                                "same sequence with a different state digest claimed an applied match");
                        ((EventDisplayState) runtimeField("eventDisplayState").get(null)).reset();
                        record.invoke(null, frame.command());
                        check(status(fly, frame.command().targetTick()) == EventDisplayState.Status.MATCHED,
                                "exact application did not reach runtime presentation");
                        check(FlyControllerRuntime.eventDisplayFor(foreign, frame.command().targetTick()).isEmpty(),
                                "another entity received the event presentation");
                        check(status(fly, frame.command().observationTick() + 21) == EventDisplayState.Status.STALE,
                                "observation freshness fence lost at runtime");
                        Files.delete(path);
                        pollOnWorker(reader);
                        check(status(fly, frame.command().targetTick()) == EventDisplayState.Status.UNAVAILABLE,
                                "missing sidecar must remain unavailable");
                    } finally { invokeRuntime("stopEventDisplay"); }
                }
            }
        } catch (Exception | AssertionError exception) {
            Throwable cause = exception;
            while (cause.getCause() != null) { cause = cause.getCause(); }
            helper.fail(Component.literal("runtime event display regression: " + cause));
        } finally {
            try {
                invokeRuntime("stopEventDisplay");
                for (var entry : saved.entrySet()) { entry.getKey().set(null, entry.getValue()); }
                if (previousPath == null) { System.clearProperty(PROPERTY); }
                else { System.setProperty(PROPERTY, previousPath); }
                if (directory != null) {
                    try (var files = Files.walk(directory)) {
                        for (Path path : files.sorted(Comparator.reverseOrder()).toList()) { Files.delete(path); }
                    }
                }
            } catch (Exception exception) { throw new AssertionError("failed to restore runtime fixture", exception); }
            finally { fly.discard(); foreign.discard(); }
        }
    }

    private static LaneCatalog catalog(Path directory, JsonObject frame) throws Exception {
        JsonObject root = new JsonObject();
        root.addProperty("schema", LaneCatalog.SCHEMA);
        root.add("initial_identity", frame.getAsJsonObject("identity").deepCopy());
        JsonArray rows = new JsonArray();
        for (String lane : List.of("original", "shuffled_weight", "no_input")) {
            boolean original = lane.equals("original");
            JsonObject row = new JsonObject();
            JsonObject identity = frame.getAsJsonObject("identity").deepCopy();
            if (!original) { identity.addProperty("dynamics_hash", (lane.equals("no_input") ? "8" : "9").repeat(64)); }
            row.add("identity", identity);
            row.addProperty("name", lane);
            row.addProperty("control_kind", original ? "faithful" : lane);
            row.addProperty("control_sha256", original ? null : "7".repeat(64));
            row.addProperty("seed", original ? null : "7");
            rows.add(row);
        }
        root.add("lanes", rows);
        Path path = directory.resolve("catalog.json");
        byte[] bytes = root.toString().getBytes(StandardCharsets.UTF_8);
        Files.write(path, bytes);
        return LaneCatalog.load(path, HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes)),
                EventDisplayFrame.parse(frame.toString()).identity());
    }

    private static void pollOnWorker(EventDisplayReader reader) throws Exception {
        Field field = EventDisplayReader.class.getDeclaredField("executor");
        field.setAccessible(true);
        Method poll = EventDisplayReader.class.getDeclaredMethod("pollOnce");
        poll.setAccessible(true);
        ((ScheduledExecutorService) field.get(reader)).submit(() -> poll.invoke(reader)).get(2, TimeUnit.SECONDS);
    }

    private static Field runtimeField(String name) throws ReflectiveOperationException {
        Field field = FlyControllerRuntime.class.getDeclaredField(name);
        field.setAccessible(true);
        return field;
    }

    private static void invokeRuntime(String name) throws ReflectiveOperationException {
        Method method = FlyControllerRuntime.class.getDeclaredMethod(name);
        method.setAccessible(true);
        method.invoke(null);
    }

    private static EventDisplayState.Status status(FlyEntity fly, long tick) {
        return EventDisplayPresentation.parseOrInvalid(FlyControllerRuntime.eventDisplayFor(fly, tick)).status();
    }

    private static void check(boolean condition, String message) {
        if (!condition) { throw new AssertionError(message); }
    }
}
