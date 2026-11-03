package dev.neurocraft.fly.gametest;

import dev.neurocraft.fly.FlyControllerRuntime;
import dev.neurocraft.fly.NeuroCraftFly;
import dev.neurocraft.fly.body.BodyProgramContextSampler;
import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.MotorAuthority;
import dev.neurocraft.fly.body.SelectionSource;
import dev.neurocraft.fly.body.programs.LandProgram;
import dev.neurocraft.fly.control.ActionFrame;
import dev.neurocraft.fly.control.BridgeConfig;
import dev.neurocraft.fly.control.ControlAckFrame;
import dev.neurocraft.fly.control.ControlOperation;
import dev.neurocraft.fly.control.ControlState;
import dev.neurocraft.fly.control.ControllerFault;
import dev.neurocraft.fly.control.ControllerIdentity;
import dev.neurocraft.fly.control.ComparisonLaneSession;
import dev.neurocraft.fly.control.LaneCatalog;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.HexFormat;
import dev.neurocraft.fly.control.RawPrimitiveAction;
import dev.neurocraft.fly.entity.FlyEntity;
import dev.neurocraft.fly.entity.FlySensors;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import net.minecraft.core.BlockPos;
import net.minecraft.gametest.framework.GameTestHelper;
import net.minecraft.network.chat.Component;
import net.minecraft.world.entity.item.ItemEntity;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.storage.ServerLevelData;
import net.minecraft.world.phys.Vec3;

/** Synthetic transport/clock, real validators, entity ticks, world collisions and program phases. */
final class OperatorProgramSwitchFence {
    private static final Modulation LAND = new Modulation(1, 1, 1, 0);
    private static final Modulation WALK = new Modulation(0.3, 1, 1, 0);
    private static final Modulation STILL = new Modulation(0, 1, 1, 0);

    private OperatorProgramSwitchFence() {}

    static void groundChain(GameTestHelper helper) {
        for (int x = 1; x <= 7; x++) {
            for (int z = 1; z <= 7; z++) { helper.setBlock(new BlockPos(x, 1, z), Blocks.STONE); }
        }
        try (Harness h = new Harness(helper, new Vec3(4.5, 2.5, 3.5))) {
            double initialHeight = h.fly.getY();
            h.acquire(BodyProgramId.LAND);
            int descent = 0;
            while (!h.grounded() && descent++ < LandProgram.MAX_DESCENT_TICKS) { h.tick(true); }
            h.check(h.grounded() && initialHeight - h.fly.getY() > 0.4,
                    "LAND must descend to actual supported contact before local switching");
            for (int i = 0; i < 4; i++) { h.tick(true); }
            double floorY = h.fly.getY();
            h.select(BodyProgramId.WALK, WALK);
            h.check(h.fly.bodyPrograms().state().id() == BodyProgramId.LAND,
                    "request changed LAND before the next entity tick");
            long walkEntry = h.now() + 1;
            h.tick(true);
            h.check(h.fly.bodyPrograms().state().id() == BodyProgramId.WALK
                            && h.fly.bodyPrograms().state().enteredGameTick() == walkEntry
                            && h.fly.bodyPrograms().state().ticksInProgram() == 1 && h.grounded(),
                    "WALK did not enter at its actual next tick on retained support");
            Vec3 walkStart = h.fly.position();
            for (int i = 0; i < 7; i++) {
                h.tick(true);
                h.check(h.grounded() && Math.abs(h.fly.getY() - floorY) < 1e-9,
                        "local switch inserted a cruise/lift tick or lost the floor");
            }
            h.check(h.fly.position().distanceTo(walkStart) > 0.001
                            && h.fly.bodyPrograms().state().ticksInProgram() == 8,
                    "WALK must actually move and advance eight owned phase ticks");
            ItemEntity food = new ItemEntity(helper.getLevel(), h.fly.getX(), h.fly.getY() + 0.05,
                    h.fly.getZ() + 0.35, new ItemStack(Items.SUGAR, 3));
            food.setNoGravity(true);
            helper.getLevel().addFreshEntity(food);
            try {
                h.select(BodyProgramId.FEED, STILL);
                long feedEntry = h.now() + 1;
                boolean[] phases = new boolean[4];
                for (int i = 0; i < 60; i++) {
                    h.tick(true);
                    h.check(h.fly.bodyPrograms().state().id() == BodyProgramId.FEED && h.grounded(),
                            "FEED was refused or lost support under unchanged operator authority");
                    phases[Math.min(3, (int) (h.fly.bodyPrograms().presentationPhase() * 4))] = true;
                }
                h.check(h.fly.bodyPrograms().state().enteredGameTick() == feedEntry
                                && h.fly.bodyPrograms().state().ticksInProgram() == 60,
                        "FEED reused an old phase clock");
                for (boolean phase : phases) { h.check(phase, "FEED did not render every phase quarter"); }
                h.tick(true);
                h.check(h.fly.bodyPrograms().state().id() == BodyProgramId.HOVER
                                && h.fly.bodyPrograms().authority() == MotorAuthority.SCRIPTED,
                        "the real FEED duration/automatic successor changed");
                h.check(food.getItem().getCount() == 3, "operator sequence consumed food");
            } finally { food.discard(); }
            h.check(h.bridge.controls().size() == 1
                            && h.bridge.controls().getFirst().operation() == ControlOperation.PAUSE,
                    "LAND/WALK/FEED must share exactly one PAUSE, with no RESUME between them");
            h.check(h.messages.size() == 3 && h.failures.isEmpty()
                            && h.messages.get(1).contains("existing acknowledged SCRIPTED authority")
                            && h.messages.get(2).contains("no new PAUSE was sent"),
                    "local selection completion invented another PAUSE ACK");
            h.check(h.fly.bodyPrograms().transitionLedger().stream()
                            .noneMatch(row -> row.contains("source=READOUT")),
                    "ground chain crossed into neural program selection");
        }
        helper.succeed();
    }

    static void admissionAndPreconditions(GameTestHelper helper) {
        try (Harness h = new Harness(helper, new Vec3(4.5, 5, 4.5))) {
            h.acquire(BodyProgramId.HOVER);
            h.select(BodyProgramId.GROOM, STILL);
            long owned = h.fly.bodyPrograms().programOwnedTicks();
            h.tick(false);
            h.check(h.fly.bodyPrograms().state().id() == BodyProgramId.GROOM
                            && h.fly.bodyPrograms().state().ticksInProgram() == 0
                            && h.fly.bodyPrograms().programOwnedTicks() == owned
                            && h.fly.presentationAction().isNeutral(),
                    "local selection bypassed the no-admitted-command neutral interlock");
            h.tick(true);
            h.check(h.fly.bodyPrograms().state().ticksInProgram() == 1,
                    "a stalled selection must begin its phase clock on the first admitted neutral tick");
            h.select(BodyProgramId.WALK, WALK);
            h.tick(true);
            h.check(h.fly.bodyPrograms().state().id() == BodyProgramId.HOVER
                            && h.fly.bodyPrograms().selectionSource() == SelectionSource.OPERATOR
                            && h.fly.bodyPrograms().transitionLedger().getLast().contains("reason=refused"),
                    "airborne local WALK bypassed its actual ground precondition");
            // A fallback still holds the same paused operator grant; selecting another program is local.
            h.select(BodyProgramId.GROOM, STILL);
            h.tickDriving();
            h.check(h.fly.bodyPrograms().authority() == MotorAuthority.NEURAL
                            && h.fly.bodyPrograms().transitionLedger().getLast().contains("reason=preempted"),
                    "local selection displaced an admitted driving command");
            h.check(h.bridge.controls().size() == 1, "local selections issued duplicate PAUSE requests");
        }
        helper.succeed();
    }

    static void exclusionAndInvalidation(GameTestHelper helper) {
        for (String changed : List.of("reset", "identity", "bridge", "removed", "late", "rewind", "entity")) {
            try (Harness h = new Harness(helper, new Vec3(4.5, 5, 4.5))) {
                h.acquire(BodyProgramId.HOVER);
                h.enableComparison();
                h.select(BodyProgramId.GROOM, STILL);
                h.check(!h.request(BodyProgramId.WALK, WALK), "a simultaneous local selection was accepted");
                h.check(!FlyControllerRuntime.releaseProgramAuthority(h.fly, h.now(), h.success, h.failure),
                        "RESUME overlapped a pending local selection");
                h.check(!FlyControllerRuntime.requestOperatorControl(h.now(), ControlOperation.RESUME,
                                null, null, h.success, h.failure)
                                && !FlyControllerRuntime.requestControl(h.now(), ControlOperation.RESUME, null, null),
                        "another operator control overlapped local selection");
                h.check(!FlyControllerRuntime.requestReset(h.fly, h.now(), 0, 0, 0, 0, h.success, h.failure)
                                && h.failures.isEmpty(),
                        "RESET reached the body mutation/precondition path during local selection");
                h.check(!FlyControllerRuntime.requestComparison(List.of(h.fly), "original")
                                && !FlyControllerRuntime.startMatchedTrials(List.of(h.fly)),
                        "comparison/trials overlapped local selection");
                switch (changed) {
                    case "reset" -> h.fly.resetNeuralSessionPreservingBody();
                    case "identity" -> {
                        BridgeConfig config = (BridgeConfig) read("config");
                        ControllerIdentity old = config.identity();
                        write("config", config.withIdentity(new ControllerIdentity(old.runId(), old.sessionEpoch() + 1,
                                old.entityId(), old.graphHash(), old.adapterHash(), old.dynamicsHash(),
                                old.decoderHash(), old.observationSchemaHash())));
                    }
                    case "bridge" -> write("bridge", new RecordingBridgeFence.RecordingBridge((BridgeConfig) read("config")));
                    case "removed" -> h.fly.discard();
                    case "late" -> h.clock.setGameTime(h.now() + 2);
                    case "rewind" -> h.clock.setGameTime(h.now() - 1);
                    case "entity" -> {
                        FlyEntity replacement = helper.spawn(NeuroCraftFly.FLY, new Vec3(6.5, 5, 6.5));
                        replacement.setUUID(h.fly.getUUID());
                        try { FlyControllerRuntime.processPendingProgramAuthority(replacement, h.now() + 1); }
                        finally { replacement.discard(); }
                    }
                    default -> throw new AssertionError(changed);
                }
                long before = h.fly.bodyPrograms().transitions();
                if (changed.equals("removed") || changed.equals("late")) {
                    // Exercise the server-end expiry path when the target no longer ticks.
                    serverBoundary();
                } else { FlyControllerRuntime.processPendingProgramAuthority(h.fly, h.now()); }
                h.check(h.fly.bodyPrograms().transitions() == before
                                && h.fly.bodyPrograms().state().id() != BodyProgramId.GROOM
                                && h.failures.size() == 1
                                && FlyControllerRuntime.programAuthorityStatus().startsWith("cancelled local_program="),
                        "local selection survived " + changed + " or mutated the body while cancelling");
                h.check(h.bridge.controls().size() == 1, "cancellation sent a PAUSE retry");
            }
        }
        helper.succeed();
    }

    static void pendingRpcAndRejectedPause(GameTestHelper helper) {
        try (Harness h = new Harness(helper, new Vec3(4.5, 5, 4.5))) {
            h.check(h.request(BodyProgramId.HOVER, STILL), "initial PAUSE was not requested");
            h.check(!h.request(BodyProgramId.GROOM, STILL), "pending initial PAUSE admitted another request");
            h.acknowledge(false);
            h.tick(true);
            h.check(h.fly.bodyPrograms().authority() == MotorAuthority.NEURAL && h.failures.size() == 1,
                    "rejected PAUSE granted operator authority");
            h.check(h.request(BodyProgramId.HOVER, STILL), "fresh PAUSE after rejection was refused");
            h.acknowledge(true);
            h.tick(true);
            h.tick(true);
            h.check(h.fly.bodyPrograms().authority() == MotorAuthority.SCRIPTED, "second valid PAUSE did not apply");
            h.check(FlyControllerRuntime.requestOperatorControl(h.now(), ControlOperation.RESUME,
                            null, null, h.success, h.failure), "ordinary RESUME was refused");
            h.check(!h.request(BodyProgramId.GROOM, STILL), "a pending ordinary RESUME retained a local grant");
            h.acknowledge(true);
            // Drain the ordinary ACK through the production request path; no local grant remains even
            // before a driving frame has had a chance to preempt SCRIPTED body authority.
            h.check(h.request(BodyProgramId.GROOM, STILL), "existing fresh-PAUSE RPC path was lost after RESUME");
            h.check(h.bridge.controls().size() == 4
                            && h.bridge.controls().getLast().operation() == ControlOperation.PAUSE
                            && FlyControllerRuntime.programAuthorityStatus().contains("operation=PAUSE"),
                    "ordinary RESUME completion incorrectly reused a stale PAUSE grant");
        }
        try (Harness h = new Harness(helper, new Vec3(4.5, 5, 4.5))) {
            h.acquire(BodyProgramId.HOVER);
            h.fly.resetNeuralSessionPreservingBody();
            h.check(FlyControllerRuntime.requestReset(h.fly, h.now(), h.fly.getX(), h.fly.getY(),
                            h.fly.getZ(), 0, h.success, h.failure), "valid reset fixture failed to stage");
            h.check(!h.request(BodyProgramId.GROOM, STILL) && h.bridge.controls().size() == 2,
                    "pending RESET admitted a program/PAUSE request");
        }
        try (Harness h = new Harness(helper, new Vec3(4.5, 5, 4.5))) {
            h.acquire(BodyProgramId.HOVER);
            h.enableComparison();
            h.check(FlyControllerRuntime.requestComparison(List.of(h.fly), "no_input"),
                    "comparison positive control was not configured to permit switching");
            h.check(!h.request(BodyProgramId.GROOM, STILL), "comparison handshake admitted a program request");
        }
        try (Harness h = new Harness(helper, new Vec3(4.5, 5, 4.5))) {
            h.acquire(BodyProgramId.HOVER);
            h.enableComparison();
            h.check(FlyControllerRuntime.startMatchedTrials(List.of(h.fly)) && FlyControllerRuntime.matchedTrialsActive(),
                    "matched-trial positive control failed to arm");
            h.check(!h.request(BodyProgramId.GROOM, STILL), "active matched trials admitted local selection");
            FlyControllerRuntime.stopMatchedTrials();
        }
        helper.succeed();
    }

    private static final class Harness implements AutoCloseable {
        private static final List<String> FIELDS = List.of("pendingOperatorControl", "pendingProgramAuthority",
                "operatorProgramGrant", "pendingLocalProgramChange", "programAuthoritySuccess",
                "programAuthorityFailure", "programAuthorityStatus", "comparison", "epochDiagnosticsDirectory",
                "matchedTrials", "resetSuccess", "resetFailure", "resetStatus");
        final GameTestHelper helper;
        final ServerLevelData clock;
        final long originalTick;
        final FlyEntity fly;
        final RecordingBridgeFence.Installed installed;
        final RecordingBridgeFence.RecordingBridge bridge;
        final Map<String, Object> previous = new LinkedHashMap<>();
        final List<String> messages = new ArrayList<>();
        final List<String> failures = new ArrayList<>();
        final Consumer<String> success = messages::add;
        final Consumer<String> failure = failures::add;
        long nextCommand;
        final Object previousReset;
        final Object resetSlot;

        Harness(GameTestHelper helper, Vec3 position) {
            this.helper = helper;
            clock = (ServerLevelData) helper.getLevel().getLevelData();
            originalTick = clock.getGameTime();
            resetSlot = read("RESET_SLOT");
            previousReset = instanceRead(resetSlot, "pending");
            for (String name : FIELDS) { previous.put(name, read(name)); }
            write("pendingOperatorControl", null);
            write("pendingProgramAuthority", null);
            write("operatorProgramGrant", null);
            write("pendingLocalProgramChange", null);
            fly = helper.spawn(NeuroCraftFly.FLY, position);
            fly.setYRot(0);
            installed = RecordingBridgeFence.install(fly);
            bridge = installed.bridge();
            bridge.enableControls();
            fly.tick(); // Real priming observation: later frames answer the recorded sequence.
        }

        void enableComparison() {
            try {
                ControllerIdentity identity = ReadoutLaneFence.identity(fly);
                Map<String, Object> base = Map.of("run_id", identity.runId(), "session_epoch", identity.sessionEpoch(),
                        "entity_id", identity.entityId(), "graph_hash", identity.graphHash(),
                        "adapter_hash", identity.adapterHash(), "dynamics_hash", identity.dynamicsHash(),
                        "decoder_hash", identity.decoderHash(), "observation_schema_hash", identity.observationSchemaHash());
                List<Object> lanes = new ArrayList<>();
                for (String name : List.of("original", "shuffled_weight", "no_input")) {
                    boolean original = name.equals("original");
                    Map<String, Object> laneIdentity = new LinkedHashMap<>(base);
                    laneIdentity.put("dynamics_hash", (original ? "3" : name.equals("no_input") ? "7" : "6").repeat(64));
                    Map<String, Object> lane = new LinkedHashMap<>();
                    lane.put("name", name);
                    lane.put("identity", laneIdentity);
                    lane.put("control_kind", original ? "faithful" : name);
                    lane.put("control_sha256", original ? null : "8".repeat(64));
                    lane.put("seed", original ? null : "7");
                    lanes.add(lane);
                }
                byte[] bytes = new com.google.gson.GsonBuilder().serializeNulls().create().toJson(
                        Map.of("schema", LaneCatalog.SCHEMA, "initial_identity", base, "lanes", lanes))
                        .getBytes(StandardCharsets.UTF_8);
                Path directory = Files.createTempDirectory("neurocraft-operator-switch-");
                Path path = directory.resolve("catalog.json");
                Files.write(path, bytes);
                LaneCatalog catalog = LaneCatalog.load(path,
                        HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes)), identity);
                BridgeConfig config = (BridgeConfig) read("config");
                write("comparison", new ComparisonLaneSession(catalog, config, bridge,
                        next -> {
                            var pending = new RecordingBridgeFence.RecordingBridge(next);
                            pending.failClosed(ControllerFault.STARTING);
                            return pending;
                        }, System::nanoTime));
                write("epochDiagnosticsDirectory", directory.resolve("epochs"));
            } catch (Exception exception) { throw new AssertionError("comparison fixture failed", exception); }
        }
        long now() { return clock.getGameTime(); }
        boolean request(BodyProgramId id, Modulation modulation) {
            return FlyControllerRuntime.requestProgramAuthority(fly, now(), id, modulation, success, failure);
        }
        void select(BodyProgramId id, Modulation modulation) {
            check(request(id, modulation), "program request was refused: " + id);
            check(FlyControllerRuntime.programAuthorityStatus().startsWith("pending local_program="),
                    "acknowledged SCRIPTED authority did not stage a local selection");
        }
        void acquire(BodyProgramId id) {
            check(request(id, LAND), "first PAUSE was not requested");
            acknowledge(true);
            tick(true);
            check(fly.bodyPrograms().authority() == MotorAuthority.NEURAL,
                    "PAUSE granted authority before the acknowledged action boundary");
            tick(true);
            check(fly.bodyPrograms().authority() == MotorAuthority.SCRIPTED, "accepted PAUSE did not engage");
        }
        void acknowledge(boolean accepted) {
            var request = bridge.controls().getLast();
            bridge.acknowledge(new ControlAckFrame(ReadoutLaneFence.identity(fly), request.sequence(),
                    request.issuedTick(), request.operation(), null, null, accepted,
                    request.operation() == ControlOperation.PAUSE ? ControlState.PAUSED : ControlState.RUNNING,
                    request.issuedTick(), accepted ? null : "INVALID_STATE"));
        }
        void tick(boolean admit) { tick(admit, false); }
        void tickDriving() { tick(true, true); }
        private void tick(boolean admit, boolean driving) {
            clock.setGameTime(now() + 1);
            var observation = bridge.observationSequenceAt(now() - 2);
            if (admit && observation.isPresent()) {
                bridge.offer(new ActionFrame(ReadoutLaneFence.identity(fly), nextCommand++,
                        observation.orElseThrow(), now() - 2, now(), now(), 0, driving ? 5 : 0,
                        "6".repeat(64), driving ? new RawPrimitiveAction(0.1, 0, 0, 0, 0) : RawPrimitiveAction.ZERO,
                        !driving));
            }
            fly.tick();
            check(FlyControllerRuntime.state() == ControllerFault.ACTIVE, "fixture bridge faulted");
        }
        boolean grounded() {
            return LandProgram.hasGroundContact(BodyProgramContextSampler.sample(helper.getLevel(), fly,
                    FlySensors.sample(helper.getLevel(), fly)));
        }
        void check(boolean condition, String message) { helper.assertTrue(condition, Component.literal(message)); }
        @Override public void close() {
            installed.close();
            FlyControllerRuntime.stopMatchedTrials();
            previous.forEach(OperatorProgramSwitchFence::write);
            instanceWrite(resetSlot, "pending", previousReset);
            clock.setGameTime(originalTick);
            fly.discard();
        }
    }

    private static void serverBoundary() {
        try {
            var method = FlyControllerRuntime.class.getDeclaredMethod("processPendingOperatorControl");
            method.setAccessible(true);
            method.invoke(null);
        } catch (ReflectiveOperationException exception) { throw new AssertionError(exception); }
    }
    private static Object instanceRead(Object instance, String name) {
        try {
            var field = instance.getClass().getDeclaredField(name);
            field.setAccessible(true);
            return field.get(instance);
        } catch (ReflectiveOperationException exception) { throw new AssertionError(exception); }
    }
    private static void instanceWrite(Object instance, String name, Object value) {
        try {
            var field = instance.getClass().getDeclaredField(name);
            field.setAccessible(true);
            field.set(instance, value);
        } catch (ReflectiveOperationException exception) { throw new AssertionError(exception); }
    }
    private static Object read(String name) {
        try { return field(name).get(null); }
        catch (ReflectiveOperationException exception) { throw new AssertionError(exception); }
    }
    private static void write(String name, Object value) {
        try { field(name).set(null, value); }
        catch (ReflectiveOperationException exception) { throw new AssertionError(exception); }
    }
    private static Field field(String name) throws ReflectiveOperationException {
        Field field = FlyControllerRuntime.class.getDeclaredField(name);
        field.setAccessible(true);
        return field;
    }
}
