package dev.neurocraft.fly.control;

import dev.neurocraft.fly.NeuroCraftFly;
import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.SelectionSource;
import dev.neurocraft.fly.entity.FlyEntity;
import dev.neurocraft.fly.entity.FlySensors;
import dev.neurocraft.fly.entity.PrimitiveActuator;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import net.fabricmc.fabric.api.client.gametest.v1.FabricClientGameTest;
import net.fabricmc.fabric.api.client.gametest.v1.context.ClientGameTestContext;
import net.fabricmc.fabric.api.client.gametest.v1.context.TestSingleplayerContext;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.minecraft.client.gui.GuiGraphics;
import net.minecraft.client.gui.screens.ChatScreen;
import net.minecraft.client.gui.screens.Overlay;
import net.minecraft.client.gui.screens.PauseScreen;
import net.minecraft.commands.arguments.EntityAnchorArgument;
import net.minecraft.core.BlockPos;
import net.minecraft.world.entity.EntitySpawnReason;
import net.minecraft.world.entity.EntityType;
import net.minecraft.world.entity.animal.Cow;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.phys.EntityHitResult;
import net.minecraft.world.phys.Vec3;
import org.lwjgl.glfw.GLFW;

/** Real native-test-input and public gameMode packets both reach the authoritative event counters. */
public final class DiagnosticClientActionsGameTest implements FabricClientGameTest {
    @Override public void runTest(ClientGameTestContext context) {
        try { run(context); }
        catch (IOException failure) { throw new UncheckedIOException(failure); }
    }

    private static void run(ClientGameTestContext context) throws IOException {
        verifyParserAndFreshFiles();
        var registrations = new AtomicInteger();
        DiagnosticClientActions.register(null, tick -> registrations.incrementAndGet(),
                close -> registrations.incrementAndGet());
        DiagnosticClientActions.register(" ", tick -> registrations.incrementAndGet(),
                close -> registrations.incrementAndGet());
        check(registrations.get() == 0, "absent property registered diagnostic hooks");

        Path directory = Files.createTempDirectory("neurocraft-client-actions-test-");
        Path path = directory.resolve("diagnostic-client-actions.ndjson");
        Files.createFile(path);
        var inbox = new DiagnosticClientActions(path);
        long[] clock = {System.nanoTime()};
        context.runOnClient(client -> inbox.tick(client, clock[0]));
        check(Files.size(results(path)) == 0, "inbox acted without a world");

        WalkProbe departingWalk;
        try (TestSingleplayerContext world = context.worldBuilder().create()) {
            world.getServer().runOnServer(server -> {
                for (int x = -4; x <= 4; x++) {
                    for (int z = -4; z <= 6; z++) {
                        server.overworld().setBlockAndUpdate(new BlockPos(x, 79, z),
                                Blocks.STONE.defaultBlockState());
                    }
                }
            });
            world.getServer().runCommand("gamemode creative @p");
            world.getServer().runCommand("tp @p 0 80 0 0 37");
            world.getServer().runCommand("item replace entity @p weapon.offhand with minecraft:air");
            world.getClientWorld().waitForChunksRender();
            UUID flyId = world.getServer().computeOnServer(server -> {
                FlyEntity fly = NeuroCraftFly.FLY.create(server.overworld(), EntitySpawnReason.COMMAND);
                check(fly != null, "cannot create interaction fixture fly");
                fly.resetForFixture(0, 80, 2, 0);
                check(server.overworld().addFreshEntity(fly), "cannot spawn interaction fixture fly");
                return fly.getUUID();
            });
            context.waitTicks(20);
            aim(context, flyId);
            pulse(context, inbox, clock);
            correctClientClock(context, inbox, clock, path, 0);
            expectCounts(world, context, flyId, 0, 0);

            // A genuine client mouse callback -> vanilla interaction -> actual network/server hook.
            equip(world, context, "minecraft:brush");
            context.getInput().pressMouse(GLFW.GLFW_MOUSE_BUTTON_RIGHT);
            expectCounts(world, context, flyId, 1, 0);
            world.getServer().runOnServer(server -> {
                var brush = server.getPlayerList().getPlayers().getFirst().getMainHandItem();
                check(brush.getCount() == 1 && brush.getDamageValue() == 0,
                        "native brush use consumed or damaged the brush");
            });
            equip(world, context, "minecraft:air");
            aim(context, flyId);
            context.getInput().pressMouse(GLFW.GLFW_MOUSE_BUTTON_LEFT);
            expectCounts(world, context, flyId, 1, 1);

            equip(world, context, "minecraft:brush");
            aim(context, flyId);
            long tick = clientTick(context);
            append(path, request(0, "brush", flyId, tick, tick + 40));
            pulse(context, inbox, clock);
            expectCounts(world, context, flyId, 2, 1);
            pulse(context, inbox, clock);
            expectCounts(world, context, flyId, 2, 1); // A consumed row never repeats.
            String brushRow = Files.readAllLines(results(path)).getFirst();
            check(brushRow.contains("\"status\":\"DISPATCHED\"")
                            && brushRow.contains("gameMode.interactAt(MAIN_HAND)")
                            && brushRow.contains("gameMode.interact(MAIN_HAND)")
                            && brushRow.contains("\"main_hand_item\":\"minecraft:brush\"")
                            && brushRow.contains("\"crosshair_entity_uuid\":\"" + flyId + "\""),
                    "brush dispatch metadata omits actual methods, target or held item: " + brushRow);

            // A later same-world correction also cannot replay the consumed brush
            // or prevent the next separately sequenced native attack request.
            correctClientClock(context, inbox, clock, path, 1);
            expectCounts(world, context, flyId, 2, 1);

            equip(world, context, "minecraft:air");
            aim(context, flyId);
            tick = clientTick(context);
            String attack = request(1, "attack", flyId, tick, tick + 40);
            Files.writeString(path, attack, StandardCharsets.UTF_8, StandardOpenOption.APPEND);
            pulse(context, inbox, clock);
            check(Files.readAllLines(results(path)).size() == 1, "partial record was consumed");
            append(path, ""); // Finish exactly that record.
            pulse(context, inbox, clock);
            expectCounts(world, context, flyId, 2, 2);

            // Stale/future/wrong target/item/player gates each consume a valid sequence once.
            tick = clientTick(context);
            append(path, request(2, "attack", flyId, tick - 2, tick - 1));
            pulse(context, inbox, clock);
            assertLastReason(path, "expired_request");
            tick = clientTick(context);
            append(path, request(3, "attack", flyId, tick + 10, tick + 30));
            pulse(context, inbox, clock);
            assertLastReason(path, "future_request");
            tick = clientTick(context);
            append(path, request(4, "attack", new UUID(0, 1), tick, tick + 40));
            pulse(context, inbox, clock);
            assertLastReason(path, "wrong_target_uuid");
            tick = clientTick(context);
            append(path, request(5, "brush", flyId, tick, tick + 40));
            pulse(context, inbox, clock);
            assertLastReason(path, "main_hand_is_not_brush");
            world.getServer().runCommand("gamemode spectator @p");
            context.waitFor(client -> client.player != null && client.player.isSpectator());
            tick = clientTick(context);
            append(path, request(6, "attack", flyId, tick, tick + 40));
            pulse(context, inbox, clock);
            assertLastReason(path, "player_not_live_nonspectator");
            expectCounts(world, context, flyId, 2, 2);
            world.getServer().runCommand("gamemode creative @p");
            context.waitFor(client -> client.player != null && !client.player.isSpectator());
            aim(context, flyId);

            // Real intervening geometry causes renderer targeting/LOS refusal, never retargeting.
            world.getServer().runCommand("setblock 0 80 1 minecraft:stone");
            world.getServer().runCommand("setblock 0 81 1 minecraft:stone");
            context.waitTicks(4);
            context.runOnClient(client -> client.gameRenderer.pick(1));
            tick = clientTick(context);
            append(path, request(7, "attack", flyId, tick, tick + 40));
            pulse(context, inbox, clock);
            check(Files.readAllLines(results(path)).getLast().contains("\"status\":\"REFUSED\""),
                    "occluded target dispatched");
            expectCounts(world, context, flyId, 2, 2);

            verifyBoundedRefusals(context, flyId);
            verifyInvalidFileDisables(context);
            world.getServer().runOnServer(server -> {
                var fly = (FlyEntity) server.overworld().getEntity(flyId);
                check(fly != null && !fly.isRemoved(), "attacks damaged or removed the fly");
                check(fly.getX() == 0 && fly.getZ() == 2
                                && fly.getDeltaMovement().x == 0 && fly.getDeltaMovement().z == 0,
                        "interaction packets introduced knockback");
            });
            verifyAiming(world, context, flyId);
            verifyResumeMenu(world, context, flyId);
            verifyTracking(world, context, flyId);
            departingWalk = verifyWalking(world, context, flyId);
        }
        pulse(context, departingWalk.inbox(), departingWalk.clock());
        check(departingWalk.inbox().isClosed(), "world departure left walking inbox open");
        context.runOnClient(client -> check(!client.options.keyUp.isDown(),
                "world departure left the synthetic forward key held"));
        var departureRows = Files.readAllLines(results(departingWalk.path()));
        check(departureRows.size() == 3
                        && departureRows.get(1).contains("\"status\":\"WALK_ABORTED\"")
                        && departureRows.get(1).contains("world_or_connection_changed")
                        && departureRows.get(2).contains("\"status\":\"DISABLED\""),
                "world departure did not preserve walking abort and lifecycle evidence");
        pulse(context, inbox, clock);
        check(inbox.isClosed(), "world departure did not close the inbox");
        check(Files.readAllLines(results(path)).getLast().contains("world_or_connection_changed"),
                "world departure was not recorded");
        System.out.println("Client interaction request diagnostics retained " + directory);
    }

    private static void resumePulse(ClientGameTestContext context, WalkProbe probe, UUID flyId, long elapsed) {
        probe.clock()[0] += elapsed;
        context.runOnClient(client -> {
            var fly = findFly(client, flyId);
            Vec3 position = client.player.position(), velocity = client.player.getDeltaMovement();
            Vec3 flyPosition = fly.position(), flyVelocity = fly.getDeltaMovement();
            float yaw = client.player.getYRot(), pitch = client.player.getXRot();
            var hit = client.hitResult;
            var keys = java.util.Arrays.stream(client.options.keyMappings).map(key -> key.isDown()).toList();
            boolean focusOption = client.options.pauseOnLostFocus;
            probe.inbox().tick(client, probe.clock()[0]);
            check(position.equals(client.player.position()) && velocity.equals(client.player.getDeltaMovement())
                            && flyPosition.equals(fly.position()) && flyVelocity.equals(fly.getDeltaMovement())
                            && yaw == client.player.getYRot() && pitch == client.player.getXRot()
                            && hit == client.hitResult && focusOption == client.options.pauseOnLostFocus
                            && keys.equals(java.util.Arrays.stream(client.options.keyMappings)
                                    .map(key -> key.isDown()).toList()),
                    "UI resume changed actor motion, view, picker, keys or focus option");
        });
    }

    private static void expectUi(Path path, String status, String reason, boolean attempted, boolean clear)
            throws IOException {
        var row = com.google.gson.JsonParser.parseString(Files.readAllLines(results(path)).getLast()).getAsJsonObject();
        check(row.get("status").getAsString().equals(status) && row.get("reason").getAsString().equals(reason)
                        && row.get("action").getAsString().equals("resume_game_menu")
                        && !row.has("walk") && !row.has("aim"), "UI resume result differs: " + row);
        var methods = row.getAsJsonArray("attempted_methods");
        check(attempted ? methods.size() == 1 && methods.get(0).getAsString().equals("Minecraft.setScreen(null)")
                        : methods.isEmpty(), "UI resume invented a method: " + row);
        var ui = row.getAsJsonObject("ui");
        check(ui.get("screen_clear").getAsBoolean() == clear
                        && ui.get("before_window_active").getAsJsonPrimitive().isBoolean()
                        && ui.get("after_window_active").getAsJsonPrimitive().isBoolean()
                        && ui.get("before_pause_on_lost_focus").getAsJsonPrimitive().isBoolean()
                        && ui.get("after_pause_on_lost_focus").getAsJsonPrimitive().isBoolean(),
                "UI resume lacks actual runtime context: " + ui);
        if (attempted) {
            check(ui.get("before_screen").getAsString().equals(PauseScreen.class.getName())
                            && ui.get("after_screen").isJsonNull() && ui.get("before_overlay").isJsonNull()
                            && ui.get("after_overlay").isJsonNull(), "resume did not close the actual PauseScreen");
        }
    }

    private static void verifyResumeMenu(TestSingleplayerContext world, ClientGameTestContext context,
            UUID flyId) throws IOException {
        prepareWalking(world, context, flyId);
        world.getServer().runCommand("gamemode spectator @p");
        world.getServer().runOnServer(server ->
                ((FlyEntity) server.overworld().getEntity(flyId)).resetForFixture(0, 80.8, 9, 180));
        context.waitFor(client -> client.player.isSpectator() && findFly(client, flyId).getZ() > 8.9);
        wrongView(context);
        var events = world.getServer().computeOnServer(server ->
                ((FlyEntity) server.overworld().getEntity(flyId)).eventSample());
        boolean originalFocusOption = context.computeOnClient(client -> client.options.pauseOnLostFocus);
        WalkProbe resumed = freshWalking(context);
        try {
            context.runOnClient(client -> {
                check(client.player.getEyePosition().distanceToSqr(findFly(client, flyId).position()) > 36
                                && !(client.hitResult instanceof EntityHitResult hit
                                        && hit.getEntity().getUUID().equals(flyId)),
                        "UI-only test accidentally satisfied gameplay reach/picker");
                client.options.pauseOnLostFocus = false;
                client.setScreen(new PauseScreen(true));
            });
            long tick = clientTick(context);
            append(resumed.path(), request(0, "resume_game_menu", flyId, tick, tick + 100));
            resumePulse(context, resumed, flyId, DiagnosticClientActions.DISPATCH_INTERVAL_NS);
            expectUi(resumed.path(), "RESUMED", "pause_screen_closed", true, true);
            context.runOnClient(client -> check(client.screen == null, "actual pause menu remained open"));
            resumePulse(context, resumed, flyId, DiagnosticClientActions.DISPATCH_INTERVAL_NS);
            check(Files.readAllLines(results(resumed.path())).size() == 1, "resume replayed without a request");
            tick = clientTick(context);
            append(resumed.path(), request(1, "resume_game_menu", flyId, tick, tick + 100));
            resumePulse(context, resumed, flyId, DiagnosticClientActions.DISPATCH_INTERVAL_NS);
            expectUi(resumed.path(), "ALREADY_CLEAR", "game_screen_already_clear", false, true);
            // The observed no-op still consumes its sequence and monotonic rate budget.
            tick = clientTick(context);
            append(resumed.path(), request(2, "resume_game_menu", flyId, tick, tick + 100));
            resumePulse(context, resumed, flyId, DiagnosticClientActions.POLL_NS);
            expectUi(resumed.path(), "REFUSED", "rate_limited", false, true);
            context.runOnClient(client -> client.setScreen(new PauseScreen(true)));
            append(resumed.path(), request(2, "resume_game_menu", flyId, tick, tick + 100));
            resumePulse(context, resumed, flyId, DiagnosticClientActions.DISPATCH_INTERVAL_NS);
            expectUi(resumed.path(), "REFUSED", "unexpected_sequence", false, false);
            check(resumed.inbox().isClosed(), "duplicate resume did not close inbox");
            context.runOnClient(client -> check(client.screen.getClass() == PauseScreen.class,
                    "duplicate resume silently closed menu"));
        } finally {
            context.runOnClient(client -> {
                client.setScreen(null);
                client.options.pauseOnLostFocus = originalFocusOption;
                resumed.inbox().close();
            });
        }
        expectCounts(world, context, flyId, events.brush().count(), events.attackAttempt().count());

        for (String reason : List.of("screen_is_not_pause_menu", "pause_subclass", "overlay_open",
                "player_busy", "conflicting_movement_input", "conflicting_key_input", "target_unavailable",
                "expired_request", "future_request", "world_or_connection_changed")) {
            WalkProbe probe = freshWalking(context);
            long tick = clientTick(context);
            append(probe.path(), request(0, "resume_game_menu", reason.equals("target_unavailable")
                    ? new UUID(0, 1) : flyId, reason.equals("future_request") ? tick + 10 : tick - 2,
                    reason.equals("expired_request") ? tick - 1 : tick + 100 - 2));
            try {
                context.runOnClient(client -> {
                    switch (reason) {
                        case "screen_is_not_pause_menu" -> client.setScreen(new ChatScreen(""));
                        case "pause_subclass" -> client.setScreen(new PauseScreen(true) {});
                        default -> client.setScreen(new PauseScreen(true));
                    }
                    switch (reason) {
                        case "overlay_open" -> client.setOverlay(new Overlay() {
                            @Override public void render(GuiGraphics graphics, int mouseX, int mouseY, float delta) {}
                        });
                        case "player_busy" -> client.options.keyUse.setDown(true);
                        case "conflicting_movement_input" -> client.options.keyUp.setDown(true);
                        case "conflicting_key_input" -> client.options.keyInventory.setDown(true);
                        default -> { }
                    }
                });
                if (reason.equals("world_or_connection_changed")) {
                    context.runOnClient(client -> {
                        var level = client.level;
                        var screen = client.screen;
                        client.level = null;
                        try {
                            probe.clock()[0] += DiagnosticClientActions.DISPATCH_INTERVAL_NS;
                            probe.inbox().tick(client, probe.clock()[0]);
                            check(client.screen == screen && probe.inbox().isClosed(),
                                    "lifecycle-changed resume closed UI or retained inbox");
                        } finally { client.level = level; }
                    });
                    var disabled = com.google.gson.JsonParser.parseString(
                            Files.readAllLines(results(probe.path())).getLast()).getAsJsonObject();
                    check(disabled.get("status").getAsString().equals("DISABLED")
                                    && disabled.get("reason").getAsString().equals(reason)
                                    && disabled.get("seq").isJsonNull() && disabled.get("action").isJsonNull(),
                            "lifecycle fence did not disable before UI request");
                } else {
                    resumePulse(context, probe, flyId, DiagnosticClientActions.DISPATCH_INTERVAL_NS);
                    expectUi(probe.path(), "REFUSED", reason.equals("pause_subclass")
                            ? "screen_is_not_pause_menu" : reason, false, false);
                    context.runOnClient(client -> check(client.screen != null, "refused resume closed a GUI"));
                }
            } finally {
                context.runOnClient(client -> {
                    client.options.keyUse.setDown(false);
                    client.options.keyUp.setDown(false);
                    client.options.keyInventory.setDown(false);
                    client.setOverlay(null);
                    client.setScreen(null);
                    probe.inbox().close();
                });
            }
        }
        for (boolean opensPause : new boolean[] {false, true}) {
            prepareWalking(world, context, flyId);
            WalkProbe walking = freshWalking(context);
            startWalking(context, walking, flyId);
            long tick = clientTick(context);
            append(walking.path(), request(1, "resume_game_menu", flyId, tick, tick + 100));
            context.runOnClient(client -> {
                if (opensPause) { client.setScreen(new PauseScreen(true)); }
                walking.clock()[0] += DiagnosticClientActions.POLL_NS;
                walking.inbox().tick(client, walking.clock()[0]);
                check(opensPause ? client.screen != null && !client.options.keyUp.isDown()
                                : client.screen == null && client.options.keyUp.isDown(),
                        "UI request changed normal walking/GUI lifecycle behavior");
            });
            expectUi(walking.path(), "REFUSED", "walk_in_progress", false, !opensPause);
            context.runOnClient(client -> { client.setScreen(null); walking.inbox().close(); });
        }
        WalkProbe capped = freshWalking(context);
        for (int seq = 0; seq < DiagnosticClientActions.COMMAND_LIMIT; seq++) {
            long tick = clientTick(context);
            append(capped.path(), request(seq, "resume_game_menu", flyId, tick, tick + 100));
            resumePulse(context, capped, flyId, DiagnosticClientActions.DISPATCH_INTERVAL_NS);
            expectUi(capped.path(), "ALREADY_CLEAR", "game_screen_already_clear", false, true);
        }
        check(capped.inbox().isClosed() && Files.readAllLines(results(capped.path())).size() == 32,
                "UI requests escaped the existing request cap");
        System.out.println("Explicit pause-menu resume proof: native PauseScreen closure and no-op; "
                + "other GUI/overlay/key/walk/lifecycle/time/sequence/rate refusals retained; "
                + "client UI evidence only, no focus or footage claim; " + resumed.path());
    }

    private static void prepareAim(TestSingleplayerContext world, ClientGameTestContext context,
            UUID flyId) {
        world.getServer().runCommand("setblock 0 80 1 minecraft:air");
        world.getServer().runCommand("setblock 0 81 1 minecraft:air");
        world.getServer().runCommand("gamemode creative @p");
        world.getServer().runCommand("tp @p 0.0 80.0 0.0 90 0");
        world.getServer().runOnServer(server -> {
            var fly = (FlyEntity) server.overworld().getEntity(flyId);
            fly.resetForFixture(0, 80, 2, 0);
        });
        context.waitTicks(6);
        context.waitFor(client -> client.player != null && !client.player.isSpectator()
                && Math.abs(client.player.getX()) < 0.01 && Math.abs(client.player.getZ()) < 0.01
                && findFly(client, flyId) != null && Math.abs(findFly(client, flyId).getX()) < 0.01
                && Math.abs(findFly(client, flyId).getZ() - 2.0) < 0.01);
    }

    private static void wrongView(ClientGameTestContext context) {
        context.runOnClient(client -> {
            client.player.setYRot(90);
            client.player.setXRot(0);
            client.gameRenderer.pick(1.0F);
        });
    }

    /** Same callback: the new actions may rotate the view, but never author actor translation. */
    private static void pulseWithoutMovingActors(ClientGameTestContext context, WalkProbe probe,
            UUID flyId) {
        probe.clock()[0] += DiagnosticClientActions.DISPATCH_INTERVAL_NS;
        context.runOnClient(client -> {
            var fly = findFly(client, flyId);
            Vec3 playerPosition = client.player.position(), playerVelocity = client.player.getDeltaMovement();
            Vec3 flyPosition = fly.position(), flyVelocity = fly.getDeltaMovement();
            probe.inbox().tick(client, probe.clock()[0]);
            check(client.player.position().equals(playerPosition)
                            && client.player.getDeltaMovement().equals(playerVelocity)
                            && fly.position().equals(flyPosition) && fly.getDeltaMovement().equals(flyVelocity),
                    "aim/click authored player or fly position/velocity");
        });
    }

    private static void aimRefusedWithoutViewChange(ClientGameTestContext context, WalkProbe probe,
            int seq, String action, UUID flyId, long tick, long through, String reason) throws IOException {
        append(probe.path(), request(seq, action, flyId, tick, through));
        probe.clock()[0] += DiagnosticClientActions.DISPATCH_INTERVAL_NS;
        context.runOnClient(client -> {
            float yaw = client.player.getYRot(), pitch = client.player.getXRot();
            probe.inbox().tick(client, probe.clock()[0]);
            check(client.player.getYRot() == yaw && client.player.getXRot() == pitch,
                    "invalid aimed request changed the view: " + reason);
        });
        assertLastReason(probe.path(), reason);
        String row = Files.readAllLines(results(probe.path())).getLast();
        check(row.contains("\"aim\":null") && row.contains("\"attempted_methods\":[]"),
                "preaim refusal claimed an executed aim: " + row);
    }

    private static void verifyAiming(TestSingleplayerContext world, ClientGameTestContext context,
            UUID flyId) throws IOException {
        prepareAim(world, context, flyId);
        equip(world, context, "minecraft:brush");
        WalkProbe probe = freshWalking(context); // The fixture helper opens an action-neutral inbox.
        try {
            aim(context, flyId);
            world.getServer().runOnServer(server -> {
                // Preparation-only velocity: actual entity ticks move the fly after the old aim.
                var fly = (FlyEntity) server.overworld().getEntity(flyId);
                fly.setDeltaMovement(0.12, 0, 0);
            });
            context.waitFor(client -> findFly(client, flyId).getX() > 0.3);
            context.runOnClient(client -> {
                client.gameRenderer.pick(1.0F);
                check(!(client.hitResult instanceof EntityHitResult hit)
                                || !hit.getEntity().getUUID().equals(flyId),
                        "moving-target preparation did not invalidate the earlier aim");
            });
            long tick = clientTick(context);
            append(probe.path(), request(0, "brush", flyId, tick, tick + 40));
            pulse(context, probe.inbox(), probe.clock());
            assertLastReason(probe.path(), "crosshair_is_not_fly");
            check(!Files.readAllLines(results(probe.path())).getLast().contains("\"aim\":"),
                    "strict legacy brush acquired synthetic aiming");
            tick = clientTick(context);
            append(probe.path(), request(1, "aim_and_brush", flyId, tick, tick + 40));
            pulseWithoutMovingActors(context, probe, flyId);
            expectCounts(world, context, flyId, 3, 2);
            String row = Files.readAllLines(results(probe.path())).getLast();
            check(row.contains("\"status\":\"DISPATCHED\"") && row.contains("before_client_view")
                            && row.contains("after_client_view") && row.contains("target_client_position_blocks")
                            && row.contains("target_client_center_blocks") && row.contains("eye_client_position_blocks")
                            && row.contains("\"type\":\"ENTITY\"") && row.contains("\"entity_uuid\":\"" + flyId + "\"")
                            && row.indexOf("player.lookAt(") < row.indexOf("gameRenderer.pick(")
                            && row.indexOf("gameRenderer.pick(") < row.indexOf("gameMode.interactAt("),
                    "aimed brush omitted actual ordered aim/picker/click evidence: " + row);
            pulse(context, probe.inbox(), probe.clock());
            expectCounts(world, context, flyId, 3, 2);
            check(Files.readAllLines(results(probe.path())).size() == 2, "aimed brush replayed");

            prepareAim(world, context, flyId);
            equip(world, context, "minecraft:air");
            wrongView(context);
            tick = clientTick(context);
            append(probe.path(), request(2, "aim_and_attack", flyId, tick, tick + 40));
            pulseWithoutMovingActors(context, probe, flyId);
            expectCounts(world, context, flyId, 3, 3);
            row = Files.readAllLines(results(probe.path())).getLast();
            check(row.contains("\"status\":\"DISPATCHED\"")
                            && row.indexOf("gameRenderer.pick(") < row.indexOf("gameMode.attack")
                            && row.contains("player.swing(MAIN_HAND)"), "aimed attack did not use native methods: " + row);

            // The aim itself consumes the rate interval. Rejection must not restore or change view.
            wrongView(context);
            tick = clientTick(context);
            append(probe.path(), request(3, "aim_and_attack", flyId, tick, tick + 40));
            context.runOnClient(client -> {
                float yaw = client.player.getYRot(), pitch = client.player.getXRot();
                probe.clock()[0] += DiagnosticClientActions.POLL_NS;
                probe.inbox().tick(client, probe.clock()[0]);
                check(yaw == client.player.getYRot() && pitch == client.player.getXRot(),
                        "rate-limited request changed view");
            });
            assertLastReason(probe.path(), "rate_limited");
            tick = clientTick(context);
            aimRefusedWithoutViewChange(context, probe, 4, "aim_and_attack", new UUID(0, 1), tick, tick + 40,
                    "target_unavailable");
            tick = clientTick(context);
            aimRefusedWithoutViewChange(context, probe, 5, "aim_and_attack", flyId, tick + 20, tick + 40,
                    "future_request");
            tick = clientTick(context);
            aimRefusedWithoutViewChange(context, probe, 6, "aim_and_attack", flyId, tick - 2, tick - 1,
                    "expired_request");
            tick = clientTick(context);
            aimRefusedWithoutViewChange(context, probe, 7, "aim_and_brush", flyId, tick, tick + 40,
                    "main_hand_is_not_brush");
            context.runOnClient(client -> client.options.keyLeft.setDown(true));
            tick = clientTick(context);
            aimRefusedWithoutViewChange(context, probe, 8, "aim_and_attack", flyId, tick, tick + 40,
                    "conflicting_movement_input");
            context.runOnClient(client -> {
                check(client.options.keyLeft.isDown(), "aim refusal changed an unowned movement key");
                client.options.keyLeft.setDown(false);
                client.setCameraEntity(findFly(client, flyId));
            });
            tick = clientTick(context);
            aimRefusedWithoutViewChange(context, probe, 9, "aim_and_attack", flyId, tick, tick + 40,
                    "camera_is_not_player");
            context.runOnClient(client -> client.setCameraEntity(client.player));

            world.getServer().runCommand("setblock 0 80 1 minecraft:stone");
            world.getServer().runCommand("setblock 0 81 1 minecraft:stone");
            context.waitTicks(4);
            tick = clientTick(context);
            aimRefusedWithoutViewChange(context, probe, 10, "aim_and_attack", flyId, tick, tick + 40,
                    "blocked_line_of_sight");
            prepareAim(world, context, flyId);
            UUID cowId = world.getServer().computeOnServer(server -> {
                Cow cow = EntityType.COW.create(server.overworld(), EntitySpawnReason.COMMAND);
                check(cow != null, "cannot create picker obstruction");
                cow.setPos(0, 80, 1); cow.setNoAi(true); cow.setNoGravity(true);
                check(server.overworld().addFreshEntity(cow), "cannot add picker obstruction");
                return cow.getUUID();
            });
            context.waitTicks(6);
            tick = clientTick(context);
            append(probe.path(), request(11, "aim_and_attack", flyId, tick, tick + 40));
            pulseWithoutMovingActors(context, probe, flyId);
            row = Files.readAllLines(results(probe.path())).getLast();
            check(row.contains("\"status\":\"REFUSED\"") && !row.contains("\"aim\":null")
                            && row.contains("\"entity_uuid\":\"" + cowId + "\"")
                            && row.contains("gameRenderer.pick(1.0F)") && !row.contains("gameMode.attack"),
                    "intervening entity was clicked or not reported by actual picker: " + row);
            world.getServer().runOnServer(server -> {
                Cow cow = (Cow) server.overworld().getEntity(cowId);
                check(cow != null && cow.getHealth() == cow.getMaxHealth(), "other entity was attacked");
                cow.discard();
            });
            context.runOnClient(client -> {
                probe.clock()[0] += DiagnosticClientActions.POLL_NS;
                probe.inbox().tick(client, probe.clock()[0]);
            });
            check(Files.readAllLines(results(probe.path())).size() == 12, "refused aim retried");
            tick = clientTick(context);
            append(probe.path(), request(12, "aim_and_attack", flyId, tick, tick + 40));
            context.runOnClient(client -> {
                float yaw = client.player.getYRot(), pitch = client.player.getXRot();
                probe.clock()[0] += DiagnosticClientActions.POLL_NS;
                probe.inbox().tick(client, probe.clock()[0]);
                check(yaw == client.player.getYRot() && pitch == client.player.getXRot(),
                        "postpicker refusal did not charge the aim rate interval");
            });
            assertLastReason(probe.path(), "rate_limited");

            world.getServer().runCommand("gamemode survival @p");
            world.getServer().runOnServer(server ->
                    ((FlyEntity) server.overworld().getEntity(flyId)).resetForFixture(0, 80, 4.5, 0));
            context.waitFor(client -> client.gameMode.getPlayerMode() == net.minecraft.world.level.GameType.SURVIVAL
                    && findFly(client, flyId).getZ() > 4.4
                    // The game-mode packet can precede the native reach-attribute update.
                    && client.player.entityInteractionRange() == 3.0
                    && !client.player.canInteractWithEntity(findFly(client, flyId), 0.0));
            context.runOnClient(client -> {
                var fly = findFly(client, flyId);
                double distance = client.player.getEyePosition().distanceTo(fly.getBoundingBox().getCenter());
                check(distance < DiagnosticClientActions.AIM_TARGET_DISTANCE_BLOCKS
                                && !client.player.canInteractWithEntity(fly, 0.0),
                        "native-reach fixture is not inside diagnostic cap and outside actual reach: distance="
                                + distance + " range=" + client.player.entityInteractionRange());
            });
            tick = clientTick(context);
            aimRefusedWithoutViewChange(context, probe, 13, "aim_and_attack", flyId, tick, tick + 40,
                    "out_of_reach");
            world.getServer().runOnServer(server ->
                    ((FlyEntity) server.overworld().getEntity(flyId)).resetForFixture(0, 80, 7, 0));
            context.waitFor(client -> findFly(client, flyId).getZ() > 6.9);
            tick = clientTick(context);
            aimRefusedWithoutViewChange(context, probe, 14, "aim_and_attack", flyId, tick, tick + 40,
                    "aim_target_out_of_range");
            prepareAim(world, context, flyId);
            equip(world, context, "minecraft:brush");
            tick = clientTick(context);
            aimRefusedWithoutViewChange(context, probe, 15, "aim_and_attack", flyId, tick, tick + 40,
                    "attack_requires_empty_main_hand");
            equip(world, context, "minecraft:air");
            context.runOnClient(client -> client.setScreen(new net.minecraft.client.gui.screens.ChatScreen("")));
            tick = clientTick(context);
            aimRefusedWithoutViewChange(context, probe, 16, "aim_and_attack", flyId, tick, tick + 40,
                    "screen_or_overlay_open");
            context.runOnClient(client -> {
                client.setScreen(null);
                client.options.keyAttack.setDown(true);
            });
            tick = clientTick(context);
            aimRefusedWithoutViewChange(context, probe, 17, "aim_and_attack", flyId, tick, tick + 40,
                    "player_busy");
            context.runOnClient(client -> client.options.keyAttack.setDown(false));
            expectCounts(world, context, flyId, 3, 3);
        } finally { context.runOnClient(client -> probe.inbox().close()); }
    }

    private record WalkProbe(DiagnosticClientActions inbox, Path path, long[] clock) {}

    private static WalkProbe freshWalking(ClientGameTestContext context) throws IOException {
        Path path = Files.createTempDirectory("neurocraft-native-walking-").resolve("input.ndjson");
        Files.createFile(path);
        var probe = new WalkProbe(new DiagnosticClientActions(path), path, new long[] {System.nanoTime()});
        context.runOnClient(client -> probe.inbox().tick(client, probe.clock()[0]));
        return probe;
    }

    private static void prepareWalking(TestSingleplayerContext world, ClientGameTestContext context,
            UUID flyId) {
        // All pose/world preparation precedes the key request. No movement is authored after onset.
        world.getServer().runCommand("setblock 0 80 1 minecraft:air");
        world.getServer().runCommand("setblock 0 81 1 minecraft:air");
        world.getServer().runCommand("gamemode survival @p");
        world.getServer().runCommand("tp @p 0.0 80.0 0.0 0 0");
        world.getServer().runOnServer(server -> {
            FlyEntity fly = (FlyEntity) server.overworld().getEntity(flyId);
            check(fly != null, "walking fixture lost fly");
            fly.resetForFixture(0, 80.8, 5.5, 180);
        });
        context.waitFor(client -> client.player != null && client.player.onGround()
                && !client.player.getAbilities().flying && !client.player.isSpectator()
                && Math.abs(client.player.getZ()) < 0.01 && client.screen == null);
        context.waitTicks(4);
    }

    private static void startWalking(ClientGameTestContext context, WalkProbe probe, UUID flyId)
            throws IOException {
        startWalking(context, probe, flyId, "walk_forward");
    }

    private static void startWalking(ClientGameTestContext context, WalkProbe probe, UUID flyId, String action)
            throws IOException {
        long tick = clientTick(context);
        append(probe.path(), request(0, action, flyId, tick, tick + 100));
        if (action.equals("walk_track_target") || action.equals("walk_approach_target")) {
            pulseWithoutMovingActors(context, probe, flyId);
        }
        else { pulse(context, probe.inbox(), probe.clock()); }
        var rows = Files.readAllLines(results(probe.path()));
        check(rows.size() == 1 && rows.getFirst().contains("\"status\":\"WALK_STARTED\""),
                "walking did not start: " + rows);
        context.runOnClient(client -> check(client.options.keyUp.isDown(), "forward key was not held"));
    }

    /** Actual scripted CRUISE, real actuator/physics, actual client W and authoritative sensors. */
    private static void verifyTracking(TestSingleplayerContext world, ClientGameTestContext context,
            UUID flyId) throws IOException {
        AtomicReference<FlyEntity> driven = new AtomicReference<>();
        ServerTickEvents.START_SERVER_TICK.register(server -> {
            FlyEntity fly = driven.get();
            if (fly == null || fly.isRemoved() || fly.level().getServer() != server) { return; }
            var sample = FlySensors.sample(server.overworld(), fly);
            var step = fly.bodyPrograms().advance(sample, true, true).orElseThrow();
            PrimitiveActuator.apply(fly, RawPrimitiveAction.ZERO, step.motors());
        });
        try {
            world.getServer().runOnServer(server -> {
                for (int x = -10; x <= 10; x++) {
                    for (int z = -10; z <= 12; z++) {
                        server.overworld().setBlockAndUpdate(new BlockPos(x, 79, z), Blocks.STONE.defaultBlockState());
                    }
                }
            });
            int[] delays = {0, 2, 4};
            long[] phases = {0, 23, 47};
            for (int scenario = 0; scenario < delays.length; scenario++) {
                prepareWalking(world, context, flyId);
                long phase = phases[scenario];
                world.getServer().runOnServer(server -> {
                    FlyEntity fly = (FlyEntity) server.overworld().getEntity(flyId);
                    fly.resetForFixture(0, 81.14, 5.5, 180);
                    fly.bodyPrograms().engageScripted(BodyProgramId.CRUISE, Modulation.NEUTRAL,
                            SelectionSource.OPERATOR, phase);
                    driven.set(fly);
                });
                context.waitFor(client -> findFly(client, flyId).getY() > 81.0);
                context.waitTicks(delays[scenario]);
                WalkProbe probe = freshWalking(context);
                Vec3 beforePlayer = serverPlayerPosition(world);
                Vec3 beforeFly = serverFlyPosition(world, flyId);
                PhysicalSensorSnapshot previous = walkingSnapshot(world, flyId);
                startWalking(context, probe, flyId, "walk_track_target");
                int sameRayDecrease = 0, front = 0, left = 0, right = 0;
                for (int tick = 1; tick <= DiagnosticClientActions.WALK_CLIENT_TICKS; tick++) {
                    walkingTick(context, probe);
                    PhysicalSensorSnapshot current = walkingSnapshot(world, flyId);
                    for (int ray : new int[] {0, 2, 3}) {
                        if (current.rayActorKinds().get(ray) == 1 && previous.rayActorKinds().get(ray) == 1
                                && rayDistance(current, ray) < rayDistance(previous, ray) - 1.0e-6) {
                            sameRayDecrease++;
                            if (ray == 0) { front++; } else if (ray == 2) { left++; } else { right++; }
                        }
                    }
                    previous = current;
                }
                Vec3 afterPlayer = serverPlayerPosition(world), afterFly = serverFlyPosition(world, flyId);
                var rows = Files.readAllLines(results(probe.path()));
                String last = rows.getLast();
                check(rows.size() == 2 && last.contains("\"status\":\"WALK_COMPLETED\"")
                                && last.contains("\"aim_count\":20")
                                && last.contains("\"observed_client_ticks\":20")
                                && last.contains("initial_observation") && last.contains("last_observation"),
                        "tracking did not complete exactly twenty aims/callbacks: " + rows);
                check(beforePlayer.distanceTo(afterPlayer) > 1.0 && beforeFly.distanceTo(afterFly) > 0.5
                                && sameRayDecrease >= 2,
                        "moving CRUISE tracking lacked actual actor motion/same-ray approach: player="
                                + beforePlayer.distanceTo(afterPlayer) + " fly=" + beforeFly.distanceTo(afterFly)
                                + " same-ray=" + sameRayDecrease);
                context.runOnClient(client -> {
                    check(!client.options.keyUp.isDown(), "tracking finish retained forward");
                    float yaw = client.player.getYRot(), pitch = client.player.getXRot();
                    probe.clock()[0] += DiagnosticClientActions.DISPATCH_INTERVAL_NS;
                    probe.inbox().tick(client, probe.clock()[0]);
                    check(yaw == client.player.getYRot() && pitch == client.player.getXRot(), "late tracking aim");
                    probe.inbox().close();
                });
                check(Files.readAllLines(results(probe.path())).size() == 2, "tracking replayed");
                System.out.println("Synthetic tracking CRUISE proof: phase=" + phase + " dispatch_delay_client_ticks="
                        + delays[scenario] + " server_player_displacement=" + beforePlayer.distanceTo(afterPlayer)
                        + " server_fly_displacement=" + beforeFly.distanceTo(afterFly)
                        + " same_ray_decreasing_player_samples=" + sameRayDecrease + " front=" + front
                        + " left=" + left + " right=" + right + "; scripted body, no neural selection; " + probe.path());
                driven.set(null);
            }
            verifyApproachMoving(world, context, flyId, driven);
        } finally { driven.set(null); }
        verifyApproachEntry(world, context, flyId);
        for (String action : List.of("walk_track_target", "walk_approach_target")) {
            verifyTrackingRefusals(world, context, flyId, action);
            verifyTrackingAborts(world, context, flyId, action);
        }
    }

    private static void verifyApproachMoving(TestSingleplayerContext world, ClientGameTestContext context,
            UUID flyId, AtomicReference<FlyEntity> driven) throws IOException {
        int[] delays = {0, 2, 4};
        long[] phases = {0, 23, 47};
        for (int scenario = 0; scenario < 13; scenario++) {
            // Three off-axis 4.0/1.2 approach candidates; fourth is a separately declared
            // moving-away target that exercises the twenty-callback limit without arrival.
            // Nine additional exploratory cells retain every outcome at lateral .4/.6/.8,
            // each with the same three declared phase/delay pairs. The measured .6 candidate
            // now protects off-ray admission and repeated front-ray observations as a regression.
            boolean retreating = scenario == 3;
            boolean exploratory = scenario >= 4;
            int phaseIndex = exploratory ? (scenario - 4) % 3 : scenario % 3;
            double lateral = exploratory ? new double[] {0.4, 0.6, 0.8}[(scenario - 4) / 3] : 1.2;
            boolean selectedCandidate = exploratory && lateral == 0.6;
            long phase = phases[phaseIndex];
            int delay = delays[phaseIndex];
            prepareWalking(world, context, flyId);
            if (!retreating) {
                // One preparation-only placement above the real floor makes vanilla gravity
                // produce a fresh landing packet. Exact-floor teleport alone can leave the
                // server's onGround flag stale while a motionless client sends no update.
                world.getServer().runCommand("tp @p " + lateral + " 80.25 0.0 0 0");
                context.waitFor(client -> Math.abs(client.player.getX() - lateral) < 0.01 && client.player.getY() > 80.0);
                int groundedStreak = 0;
                for (int settle = 0; settle < 10; settle++) {
                    context.waitTick();
                    boolean grounded = world.getServer().computeOnServer(server -> {
                        var player = server.getPlayerList().getPlayers().getFirst();
                        check(server.overworld().getBlockState(BlockPos.containing(player.getX(), 79, player.getZ()))
                                        .is(Blocks.STONE), "off-axis preparation lacks its actual stone support");
                        return player.onGround() && Math.abs(player.getY() - 80.0) < 1.0e-6
                                && !player.getAbilities().flying && !player.isSpectator();
                    });
                    grounded &= context.computeOnClient(client -> client.player.onGround()
                            && Math.abs(client.player.getY() - 80.0) < 1.0e-6 && !client.player.getAbilities().flying);
                    groundedStreak = grounded ? groundedStreak + 1 : 0;
                    if (groundedStreak == 3) { break; }
                }
                check(groundedStreak == 3, "ordinary landing did not establish three matched grounded observations");
            }
            world.getServer().runOnServer(server -> {
                FlyEntity fly = (FlyEntity) server.overworld().getEntity(flyId);
                fly.resetForFixture(0, 81.14, 4, retreating ? 0 : 180);
                fly.bodyPrograms().engageScripted(BodyProgramId.CRUISE, Modulation.NEUTRAL,
                        SelectionSource.OPERATOR, phase);
                driven.set(fly);
            });
            context.waitFor(client -> findFly(client, flyId).getY() > 81.0);
            context.waitTicks(delay);
            WalkProbe probe = freshWalking(context);
            try {
                Vec3 playerBefore = serverPlayerPosition(world), flyBefore = serverFlyPosition(world, flyId);
                PhysicalSensorSnapshot previous = walkingSnapshot(world, flyId);
                List<Integer> prestartActors = previous.rayActorKinds();
                check(!selectedCandidate || prestartActors.stream().noneMatch(actor -> actor == 1),
                        "selected approach preparation already exposed a Player ray");
                if (exploratory) {
                    long requestTick = clientTick(context);
                    append(probe.path(), request(0, "walk_approach_target", flyId, requestTick, requestTick + 100));
                    pulseWithoutMovingActors(context, probe, flyId);
                    var first = com.google.gson.JsonParser.parseString(Files.readAllLines(results(probe.path())).getFirst())
                            .getAsJsonObject();
                    if (!first.get("status").getAsString().equals("WALK_STARTED")) {
                        check(!selectedCandidate, "selected approach candidate refused before walking: " + first);
                        System.out.println("Synthetic bounded approach sweep: lateral=" + lateral + " phase=" + phase
                                + " delay=" + delay + " prestart_actor_kinds=" + prestartActors
                                + " outcome=" + first + "; no successful gesture inferred; " + probe.path());
                        continue;
                    }
                } else { startWalking(context, probe, flyId, "walk_approach_target"); }
                int decreasing = 0, hitObservations = 0, firstHitCallback = -1;
                int[] decreasesByRay = new int[4];
                for (int callback = 1; callback <= 20; callback++) {
                    walkingTick(context, probe);
                    var current = walkingSnapshot(world, flyId);
                    boolean playerHit = false;
                    for (int ray : new int[] {0, 2, 3}) {
                        playerHit |= current.rayActorKinds().get(ray) == 1;
                        if (current.rayActorKinds().get(ray) == 1 && previous.rayActorKinds().get(ray) == 1
                                && rayDistance(current, ray) < rayDistance(previous, ray) - 1.0e-6) {
                            decreasing++; decreasesByRay[ray]++;
                        }
                    }
                    if (playerHit) { hitObservations++; if (firstHitCallback < 0) { firstHitCallback = callback; } }
                    previous = current;
                    if (Files.readAllLines(results(probe.path())).size() > 1) { break; }
                }
                var rows = Files.readAllLines(results(probe.path()));
                var last = com.google.gson.JsonParser.parseString(rows.getLast()).getAsJsonObject();
                boolean completed = last.get("status").getAsString().equals("WALK_COMPLETED");
                check(rows.size() == 2 && (completed || exploratory
                                && last.get("status").getAsString().equals("WALK_ABORTED")),
                        "bounded approach did not complete: " + rows);
                var walk = last.getAsJsonObject("walk");
                int callbacks = walk.get("observed_client_ticks").getAsInt();
                double stoppedDistance = walk.getAsJsonObject("terminal_geometry")
                        .get("horizontal_distance_blocks").getAsDouble();
                check(!walk.has("fixed_client_ticks") && walk.get("max_client_ticks").getAsInt() == 20
                                && walk.get("stop_horizontal_distance_blocks").getAsDouble() == 0.65
                                && walk.getAsJsonObject("tracking").get("aim_count").getAsInt() == callbacks
                                && callbacks >= 1 && callbacks <= 20,
                        "bounded approach misrepresented count or stop rule: " + walk);
                String completion = completed ? walk.get("completion_reason").getAsString()
                        : "ABORTED:" + last.get("reason").getAsString();
                if (retreating) {
                    check(completion.equals("callback_limit_reached") && callbacks == 20 && stoppedDistance > 0.65,
                            "retreating fixture did not exercise the declared cap: " + walk);
                } else if (!exploratory || selectedCandidate) {
                    check(completion.equals("stop_distance_reached") && stoppedDistance > 0.05 && stoppedDistance <= 0.65,
                            "arrival did not retain valid finite stop geometry: " + walk);
                }
                Vec3 atRelease = serverPlayerPosition(world), flyAtRelease = serverFlyPosition(world, flyId);
                check(exploratory && !selectedCandidate
                                || playerBefore.distanceTo(atRelease) > 0.5 && flyBefore.distanceTo(flyAtRelease) > 0.3,
                        "approach lacked actual grounded player and moving CRUISE displacement");
                check(!selectedCandidate || decreasesByRay[0] >= 2,
                        "selected approach candidate lost repeated decreasing front-Player observations");
                float[] finalView = new float[2];
                context.runOnClient(client -> {
                    check(!client.options.keyUp.isDown(), "approach completion retained forward");
                    finalView[0] = client.player.getYRot(); finalView[1] = client.player.getXRot();
                });
                double minimumAfterRelease = horizontalSeparation(atRelease, flyAtRelease);
                for (int coast = 0; coast < 10; coast++) {
                    walkingTick(context, probe);
                    context.runOnClient(client -> check(!client.options.keyUp.isDown()
                                    && client.player.getYRot() == finalView[0] && client.player.getXRot() == finalView[1],
                            "approach aimed/repressed forward after key release"));
                    minimumAfterRelease = Math.min(minimumAfterRelease,
                            horizontalSeparation(serverPlayerPosition(world), serverFlyPosition(world, flyId)));
                }
                double coast = atRelease.distanceTo(serverPlayerPosition(world));
                check(exploratory && !selectedCandidate || coast > 0.001,
                        "native momentum was unexpectedly replaced by an immediate stop");
                check(Files.readAllLines(results(probe.path())).size() == 2, "approach request replayed");
                System.out.println("Synthetic bounded approach " + (exploratory ? "sweep" : "proof")
                        + ": lateral=" + (retreating ? 0.0 : lateral) + " phase=" + phase + " delay=" + delay
                        + " retreating=" + retreating + " finish=" + completion + " callbacks=" + callbacks
                        + " client_stop_request_distance=" + stoppedDistance
                        + " server_player_displacement=" + playerBefore.distanceTo(atRelease)
                        + " server_fly_displacement=" + flyBefore.distanceTo(flyAtRelease)
                        + " decreasing_same_player_ray_samples=" + decreasing + " postrelease_server_coast=" + coast
                        + " prestart_actor_kinds=" + prestartActors + " player_ray_hit_observations=" + hitObservations
                        + " first_player_ray_callback=" + firstHitCallback + " decreasing_front=" + decreasesByRay[0]
                        + " decreasing_left=" + decreasesByRay[2] + " decreasing_right=" + decreasesByRay[3]
                        + " minimum_server_separation_during_coast=" + minimumAfterRelease
                        + "; scripted CRUISE/native gameplay, no neural selection; " + probe.path());
            } finally {
                context.runOnClient(client -> probe.inbox().close());
                driven.set(null);
            }
        }
    }

    private static double horizontalSeparation(Vec3 a, Vec3 b) {
        return Math.hypot(a.x - b.x, a.z - b.z);
    }

    private static void verifyApproachEntry(TestSingleplayerContext world, ClientGameTestContext context,
            UUID flyId) throws IOException {
        check(DiagnosticClientActions.insideApproachStopRadius(Vec3.ZERO, new Vec3(0, 0, 0.65))
                        && DiagnosticClientActions.insideApproachStopRadius(Vec3.ZERO, new Vec3(0, 0, Math.nextDown(0.65)))
                        && !DiagnosticClientActions.insideApproachStopRadius(Vec3.ZERO, new Vec3(0, 0, Math.nextUp(0.65))),
                "arrival radius is not inclusive at the exact declared boundary");
        for (double distance : new double[] {0.64, 0.66}) {
            prepareWalking(world, context, flyId);
            world.getServer().runOnServer(server ->
                    ((FlyEntity) server.overworld().getEntity(flyId)).resetForFixture(0, 81.14, distance, 0));
            context.waitFor(client -> Math.abs(findFly(client, flyId).getZ() - distance) < 0.001);
            WalkProbe probe = freshWalking(context);
            long tick = clientTick(context);
            append(probe.path(), request(0, "walk_approach_target", flyId, tick, tick + 40));
            context.runOnClient(client -> {
                float yaw = client.player.getYRot(), pitch = client.player.getXRot();
                probe.clock()[0] += DiagnosticClientActions.DISPATCH_INTERVAL_NS;
                probe.inbox().tick(client, probe.clock()[0]);
                if (distance < 0.65) {
                    check(!client.options.keyUp.isDown() && yaw == client.player.getYRot() && pitch == client.player.getXRot(),
                            "already-close approach changed view/key");
                } else { check(client.options.keyUp.isDown(), "outside-radius approach did not begin"); }
                probe.inbox().close();
            });
            var row = Files.readAllLines(results(probe.path())).getFirst();
            check(row.contains(distance < 0.65 ? "walking_target_already_within_stop_distance" : "WALK_STARTED"),
                    "approach entry radius test used wrong result: " + row);
        }
    }

    private static double rayDistance(PhysicalSensorSnapshot sample, int ray) {
        return switch (ray) {
            case 0 -> sample.rayForwardBlocks();
            case 2 -> sample.rayLeftBlocks();
            case 3 -> sample.rayRightBlocks();
            default -> throw new IllegalArgumentException("undeclared approach ray");
        };
    }

    private static Vec3 serverPlayerPosition(TestSingleplayerContext world) {
        return world.getServer().computeOnServer(server -> {
            var player = server.getPlayerList().getPlayers().getFirst();
            check(player.onGround() && !player.getAbilities().flying && !player.isSpectator(),
                    "authoritative tracking motion was not grounded, nonflying gameplay: onGround="
                            + player.onGround() + " flying=" + player.getAbilities().flying
                            + " spectator=" + player.isSpectator() + " position=" + player.position());
            return player.position();
        });
    }

    private static Vec3 serverFlyPosition(TestSingleplayerContext world, UUID flyId) {
        return world.getServer().computeOnServer(server -> server.overworld().getEntity(flyId).position());
    }

    private static void verifyTrackingRefusals(TestSingleplayerContext world, ClientGameTestContext context,
            UUID flyId, String action) throws IOException {
        check("invalid_tracking_geometry".equals(DiagnosticClientActions.trackingGeometryRefusal(
                        new Vec3(Double.NaN, 0, 0), Vec3.ZERO)), "nonfinite eye accepted");
        check("invalid_tracking_geometry".equals(DiagnosticClientActions.trackingGeometryRefusal(
                        Vec3.ZERO, new Vec3(0, Double.POSITIVE_INFINITY, 1))), "nonfinite target accepted");
        check("tracking_horizontal_heading_undefined".equals(DiagnosticClientActions.trackingGeometryRefusal(
                        Vec3.ZERO, Vec3.ZERO)), "coincident target accepted");
        for (Vec3 offset : new Vec3[] {Vec3.ZERO, new Vec3(0, 0.5, 0), new Vec3(0, 0.5, 0.04)}) {
            prepareWalking(world, context, flyId);
            Vec3 center = world.getServer().computeOnServer(server -> {
                FlyEntity fly = (FlyEntity) server.overworld().getEntity(flyId);
                Vec3 target = server.getPlayerList().getPlayers().getFirst().getEyePosition().add(offset);
                fly.resetForFixture(target.x, target.y - fly.getBbHeight() / 2.0, target.z, 0);
                return target;
            });
            context.waitFor(client -> findFly(client, flyId).getBoundingBox().getCenter().distanceTo(center) < 0.005);
            context.runOnClient(client -> check(findFly(client, flyId).getBoundingBox().getCenter()
                            .subtract(client.player.getEyePosition()).distanceTo(offset) < 0.005,
                    "coincident/vertical tracking fixture did not match actual eyes and target"));
            WalkProbe probe = freshWalking(context);
            long tick = clientTick(context);
            append(probe.path(), request(0, action, flyId, tick, tick + 40));
            context.runOnClient(client -> {
                float yaw = client.player.getYRot(), pitch = client.player.getXRot();
                probe.clock()[0] += DiagnosticClientActions.DISPATCH_INTERVAL_NS;
                probe.inbox().tick(client, probe.clock()[0]);
                check(yaw == client.player.getYRot() && pitch == client.player.getXRot()
                                && !client.options.keyUp.isDown(), "undefined heading changed view/key");
            });
            assertLastReason(probe.path(), "tracking_horizontal_heading_undefined");
            check(!Files.readAllLines(results(probe.path())).getLast().contains("\"walk\":"),
                    "refusal invented tracking observations");
            context.runOnClient(client -> probe.inbox().close());
        }
    }

    private static void assertTrackingView(net.minecraft.client.Minecraft client, UUID flyId) {
        var target = findFly(client, flyId);
        Vec3 direction = target.getBoundingBox().getCenter().subtract(client.player.getEyePosition()).normalize();
        check(client.player.getViewVector(1.0F).dot(direction) > 0.999999,
                "tracking view is not aimed at the actual fresh center");
        check(Float.isFinite(client.player.getYRot()) && Math.abs(client.player.getYRot()) <= 180.0001F
                        && Float.isFinite(client.player.getXRot()) && Math.abs(client.player.getXRot()) <= 90,
                "native tracking produced an invalid/unwrapped view");
    }

    private static void verifyTrackingAborts(TestSingleplayerContext world, ClientGameTestContext context,
            UUID flyId, String action) throws IOException {
        // A target behind the player, then raised by actual ESCAPE body thrust, exercises yaw wrap
        // and changed pitch without the diagnostic authoring either actor's translation.
        prepareWalking(world, context, flyId);
        world.getServer().runOnServer(server ->
                ((FlyEntity) server.overworld().getEntity(flyId)).resetForFixture(0, 81.14, -5.5, 0));
        context.waitFor(client -> findFly(client, flyId).getZ() < -5.4);
        WalkProbe jumping = freshWalking(context);
        startWalking(context, jumping, flyId, action);
        context.runOnClient(client -> {
            assertTrackingView(client, flyId);
            check(Math.abs(client.player.getYRot()) > 170, "behind target did not wrap native yaw");
        });
        double startHeight = serverFlyPosition(world, flyId).y;
        world.getServer().runOnServer(server -> ((FlyEntity) server.overworld().getEntity(flyId))
                .bodyPrograms().engageScripted(BodyProgramId.ESCAPE_JUMP, Modulation.NEUTRAL,
                        SelectionSource.OPERATOR, server.overworld().getGameTime()));
        // Neutral ESCAPE starts with six crouch steps. Observe through six actual launch steps.
        for (int i = 0; i < 12; i++) {
            world.getServer().runOnServer(server -> {
                FlyEntity fly = (FlyEntity) server.overworld().getEntity(flyId);
                var step = fly.bodyPrograms().advance(FlySensors.sample(server.overworld(), fly), true, true)
                        .orElseThrow();
                PrimitiveActuator.apply(fly, RawPrimitiveAction.ZERO, step.motors());
            });
            walkingTick(context, jumping);
            context.runOnClient(client -> assertTrackingView(client, flyId));
        }
        double endHeight = serverFlyPosition(world, flyId).y;
        check(endHeight > startHeight + 0.1,
                "scripted escape did not raise target after six crouch/six launch steps: start="
                        + startHeight + " end=" + endHeight);
        context.runOnClient(client -> jumping.inbox().close());
        check(Files.readAllLines(results(jumping.path())).getLast().contains("\"aim_count\":13"),
                "jump tracking did not retain exactly entry plus twelve observed aims");

        for (String reason : List.of("conflicting_movement_input", "screen_or_overlay_open",
                "walking_wall_timeout", "blocked_line_of_sight", "walking_target_out_of_range",
                "camera_is_not_player", "world_or_connection_changed", "diagnostic_closed",
                "result_write_failure")) {
            prepareWalking(world, context, flyId);
            WalkProbe probe = freshWalking(context);
            startWalking(context, probe, flyId, action);
            if (reason.equals("blocked_line_of_sight")) {
                world.getServer().runCommand("setblock 0 80 1 minecraft:stone");
                world.getServer().runCommand("setblock 0 81 1 minecraft:stone");
                context.waitTicks(2);
            } else if (reason.equals("walking_target_out_of_range")) {
                world.getServer().runOnServer(server ->
                        ((FlyEntity) server.overworld().getEntity(flyId)).resetForFixture(0, 80.8, 10, 180));
                context.waitFor(client -> findFly(client, flyId).getZ() > 9.9);
            }
            context.runOnClient(client -> {
                var level = client.level;
                float yaw = client.player.getYRot(), pitch = client.player.getXRot();
                switch (reason) {
                    case "conflicting_movement_input" -> client.options.keyLeft.setDown(true);
                    case "screen_or_overlay_open" -> client.setScreen(new net.minecraft.client.gui.screens.ChatScreen(""));
                    case "camera_is_not_player" -> client.setCameraEntity(findFly(client, flyId));
                    case "world_or_connection_changed" -> client.level = null; // Fixture lifecycle loss in this callback.
                    case "result_write_failure" -> {
                        try {
                            var field = DiagnosticClientActions.class.getDeclaredField("results");
                            field.setAccessible(true);
                            ((java.nio.channels.FileChannel) field.get(probe.inbox())).close();
                        } catch (ReflectiveOperationException | IOException failure) { throw new AssertionError(failure); }
                    }
                    default -> { }
                }
                probe.clock()[0] += reason.equals("walking_wall_timeout") || reason.equals("result_write_failure")
                        ? DiagnosticClientActions.WALK_TIMEOUT_NS : 50_000_000L;
                try {
                    if (reason.equals("diagnostic_closed")) { probe.inbox().close(); }
                    else { probe.inbox().tick(client, probe.clock()[0]); }
                    check(!client.options.keyUp.isDown(), "abort left owned key held: " + reason);
                    check(yaw == client.player.getYRot() && pitch == client.player.getXRot(),
                            "abort performed a late aim: " + reason);
                    if (reason.equals("conflicting_movement_input")) {
                        check(client.options.keyLeft.isDown(), "abort overwrote unowned key");
                    }
                } finally {
                    client.level = level;
                    client.setCameraEntity(client.player);
                    client.options.keyLeft.setDown(false);
                    if (client.screen != null) { client.setScreen(null); }
                    probe.inbox().close();
                }
                probe.clock()[0] += DiagnosticClientActions.DISPATCH_INTERVAL_NS;
                probe.inbox().tick(client, probe.clock()[0]);
                check(yaw == client.player.getYRot() && pitch == client.player.getXRot(),
                        "closed tracking diagnostic aimed again: " + reason);
            });
            var rows = Files.readAllLines(results(probe.path()));
            if (reason.equals("result_write_failure")) {
                check(rows.size() == 1 && probe.inbox().isClosed(), "failed result write remained active");
            } else {
                check(rows.stream().anyMatch(row -> row.contains("\"status\":\"WALK_ABORTED\"")
                                && row.contains(reason) && row.contains("\"aim_count\":1")
                                && row.contains("\"forward_key_released\":true")),
                        "abort omitted bounded last aim or release: " + rows);
                if (action.equals("walk_approach_target")) {
                    var aborted = rows.stream().map(row -> com.google.gson.JsonParser.parseString(row).getAsJsonObject())
                            .filter(row -> row.get("status").getAsString().equals("WALK_ABORTED")).findFirst().orElseThrow();
                    var walk = aborted.getAsJsonObject("walk");
                    check(walk.get("completion_reason").isJsonNull() && !walk.has("fixed_client_ticks")
                                    && walk.get("max_client_ticks").getAsInt() == 20,
                            "approach abort claimed completion or fixed twenty ticks");
                    check(walk.get("terminal_geometry").isJsonNull() == reason.equals("world_or_connection_changed"),
                            "approach terminal geometry crossed lifecycle or disappeared while available");
                }
            }
        }
    }

    private static void walkingTick(ClientGameTestContext context, WalkProbe probe) {
        context.waitTick();
        probe.clock()[0] += 50_000_000L;
        context.runOnClient(client -> probe.inbox().tick(client, probe.clock()[0]));
    }

    private static PhysicalSensorSnapshot walkingSnapshot(TestSingleplayerContext world, UUID flyId) {
        return world.getServer().computeOnServer(server -> {
            FlyEntity fly = (FlyEntity) server.overworld().getEntity(flyId);
            check(fly != null, "walking fixture lost fly");
            return FlySensors.sample(server.overworld(), fly);
        });
    }

    private static WalkProbe verifyWalking(TestSingleplayerContext world, ClientGameTestContext context,
            UUID flyId) throws IOException {
        prepareWalking(world, context, flyId);
        WalkProbe moving = freshWalking(context);
        var before = walkingSnapshot(world, flyId);
        check(before.rayActorKinds().getFirst() == 1 && before.rayHitKinds().getFirst() == 2,
                "standing Player is absent from the low fly's forward ray");
        double startZ = world.getServer().computeOnServer(server ->
                server.getPlayerList().getPlayers().getFirst().getZ());
        startWalking(context, moving, flyId);
        double previous = before.rayForwardBlocks();
        int decreasing = 0;
        for (int tick = 1; tick <= DiagnosticClientActions.WALK_CLIENT_TICKS; tick++) {
            walkingTick(context, moving);
            var sample = walkingSnapshot(world, flyId);
            if (sample.rayActorKinds().getFirst() == 1 && sample.rayHitKinds().getFirst() == 2
                    && sample.rayForwardBlocks() < previous) { decreasing++; }
            previous = sample.rayForwardBlocks();
            if (tick < DiagnosticClientActions.WALK_CLIENT_TICKS) {
                check(Files.readAllLines(results(moving.path())).size() == 1,
                        "walking completed before twenty client ticks");
            }
        }
        var rows = Files.readAllLines(results(moving.path()));
        check(rows.size() == 2 && rows.getLast().contains("\"status\":\"WALK_COMPLETED\"")
                        && rows.getLast().contains("\"observed_client_ticks\":20")
                        && rows.getLast().contains("\"forward_key_released\":true")
                        && !rows.getLast().contains("\"tracking\":")
                        && rows.getLast().contains("start_client_pose")
                        && rows.getLast().contains("end_client_pose"),
                "walking completion/release evidence missing: " + rows);
        context.runOnClient(client -> check(!client.options.keyUp.isDown(), "completed key stayed held"));
        double endZ = world.getServer().computeOnServer(server -> {
            var player = server.getPlayerList().getPlayers().getFirst();
            check(player.onGround() && !player.getAbilities().flying, "movement was not grounded walking");
            check(Math.abs(player.getX()) < 0.05, "forward walking drifted sideways");
            return player.getZ();
        });
        check(endZ > startZ + 1.0 && decreasing >= 5,
                "native movement did not produce authoritative displacement and decreasing Player rays");
        System.out.println("Synthetic walking proof: server_start_z=" + startZ + " server_end_z=" + endZ
                + " decreasing_player_ray_samples=" + decreasing + " client_key_ticks=20");
        for (int tick = 0; tick < 10; tick++) { walkingTick(context, moving); }
        check(Files.readAllLines(results(moving.path())).size() == 2, "walking request replayed");
        context.runOnClient(client -> check(!client.options.keyUp.isDown(), "walking input restarted"));
        context.runOnClient(client -> moving.inbox().close());

        // A competing input aborts and releases only the key owned by this diagnostic.
        prepareWalking(world, context, flyId);
        WalkProbe conflict = freshWalking(context);
        startWalking(context, conflict, flyId);
        context.runOnClient(client -> {
            client.options.keyLeft.setDown(true);
            conflict.clock()[0] += 50_000_000L;
            conflict.inbox().tick(client, conflict.clock()[0]);
            check(!client.options.keyUp.isDown() && client.options.keyLeft.isDown(),
                    "conflicting input was overwritten or forward was not released");
            client.options.keyLeft.setDown(false);
            conflict.inbox().close();
        });
        check(Files.readAllLines(results(conflict.path())).getLast().contains("conflicting_movement_input"),
                "competing input did not produce an abort diagnostic");

        prepareWalking(world, context, flyId);
        WalkProbe screen = freshWalking(context);
        startWalking(context, screen, flyId);
        context.runOnClient(client -> {
            client.setScreen(new net.minecraft.client.gui.screens.ChatScreen(""));
            screen.clock()[0] += 50_000_000L;
            screen.inbox().tick(client, screen.clock()[0]);
            check(!client.options.keyUp.isDown(), "screen opening left forward held");
            client.setScreen(null);
            screen.inbox().close();
        });
        check(Files.readAllLines(results(screen.path())).getLast().contains("screen_or_overlay_open"),
                "screen opening omitted the walking abort");

        prepareWalking(world, context, flyId);
        WalkProbe timed = freshWalking(context);
        startWalking(context, timed, flyId);
        context.runOnClient(client -> {
            timed.clock()[0] += DiagnosticClientActions.WALK_TIMEOUT_NS;
            timed.inbox().tick(client, timed.clock()[0]);
            check(!client.options.keyUp.isDown(), "wall timeout left forward held");
            timed.inbox().close();
        });
        check(Files.readAllLines(results(timed.path())).getLast().contains("walking_wall_timeout"),
                "monotonic wall bound was not enforced");

        prepareWalking(world, context, flyId);
        WalkProbe closing = freshWalking(context);
        startWalking(context, closing, flyId);
        context.runOnClient(client -> {
            closing.inbox().close();
            check(!client.options.keyUp.isDown(), "close left forward held");
        });
        check(Files.readAllLines(results(closing.path())).getLast().contains("diagnostic_closed"),
                "close omitted the active walking abort");

        verifyWalkingRefusals(world, context, flyId);
        prepareWalking(world, context, flyId);
        WalkProbe departing = freshWalking(context);
        startWalking(context, departing, flyId);
        return departing;
    }

    private static void verifyWalkingRefusals(TestSingleplayerContext world, ClientGameTestContext context,
            UUID flyId) throws IOException {
        prepareWalking(world, context, flyId);
        WalkProbe probe = freshWalking(context);
        long tick = clientTick(context);
        append(probe.path(), request(0, "walk_forward", new UUID(0, 1), tick, tick + 40));
        pulse(context, probe.inbox(), probe.clock());
        assertLastReason(probe.path(), "target_unavailable");
        context.runOnClient(client -> client.options.keyUp.setDown(true));
        tick = clientTick(context);
        append(probe.path(), request(1, "walk_forward", flyId, tick, tick + 40));
        pulse(context, probe.inbox(), probe.clock());
        assertLastReason(probe.path(), "conflicting_movement_input");
        context.runOnClient(client -> {
            check(client.options.keyUp.isDown(), "refusal released a forward key the diagnostic did not own");
            client.options.keyUp.setDown(false);
        });
        world.getServer().runCommand("setblock 0 80 1 minecraft:stone");
        world.getServer().runCommand("setblock 0 81 1 minecraft:stone");
        context.waitTicks(4);
        tick = clientTick(context);
        append(probe.path(), request(2, "walk_forward", flyId, tick, tick + 40));
        pulse(context, probe.inbox(), probe.clock());
        assertLastReason(probe.path(), "blocked_line_of_sight");
        prepareWalking(world, context, flyId);
        world.getServer().runCommand("tp @p 0.0 80.0 -3.0 0 0");
        context.waitFor(client -> client.player != null && client.player.getZ() < -2.9);
        tick = clientTick(context);
        append(probe.path(), request(3, "walk_forward", flyId, tick, tick + 40));
        pulse(context, probe.inbox(), probe.clock());
        assertLastReason(probe.path(), "walking_target_out_of_range");
        prepareWalking(world, context, flyId);
        context.getInput().holdKeyFor(GLFW.GLFW_KEY_SPACE, 2);
        context.waitFor(client -> !client.player.onGround());
        tick = clientTick(context);
        append(probe.path(), request(4, "walk_forward", flyId, tick, tick + 40));
        pulse(context, probe.inbox(), probe.clock());
        assertLastReason(probe.path(), "walking_requires_grounded_nonflying_player");
        context.runOnClient(client -> {
            check(!client.options.keyUp.isDown(), "refused geometry pressed forward");
            probe.inbox().close();
        });
    }

    private static void correctClientClock(ClientGameTestContext context,
            DiagnosticClientActions inbox, long[] clock, Path path, int expectedResults)
            throws IOException {
        context.runOnClient(client -> {
            check(client.level != null, "clock correction fixture has no client world");
            // Establish the latest tick and apply the packet handler's actual
            // setter in the same callback. Seven ticks is an explicit test
            // perturbation, not a claim about the recorded startup correction.
            clock[0] += DiagnosticClientActions.DISPATCH_INTERVAL_NS;
            inbox.tick(client, clock[0]);
            long before = client.level.getGameTime();
            check(before >= 7, "clock correction fixture started too early");
            client.level.setTimeFromServer(before - 7, client.level.getDayTime(), true);
            check(client.level.getGameTime() == before - 7, "client clock did not correct backwards");
            clock[0] += DiagnosticClientActions.DISPATCH_INTERVAL_NS;
            inbox.tick(client, clock[0]);
            check(!inbox.isClosed(), "same-world clock correction closed the action inbox");
        });
        check(Files.readAllLines(results(path)).size() == expectedResults,
                "clock correction itself dispatched, replayed or wrote a refusal");
    }

    private static void verifyParserAndFreshFiles() throws IOException {
        String valid = request(0, "brush", new UUID(0, 1), 100, 140);
        try { check(DiagnosticClientActions.Request.parse(valid).seq() == 0, "valid request refused"); }
        catch (ProtocolException bad) { throw new AssertionError(bad); }
        try {
            check(DiagnosticClientActions.Request.parse(valid.replace("brush", "walk_forward"))
                    .action().equals("walk_forward"), "fixed walking action was refused");
            var tracking = DiagnosticClientActions.Request.parse(valid.replace("brush", "walk_track_target"));
            check(tracking.walking() && tracking.tracking() && !tracking.aimed() && !tracking.brush(),
                    "explicit tracking walk misclassified");
            var approach = DiagnosticClientActions.Request.parse(valid.replace("brush", "walk_approach_target"));
            check(approach.approaching() && approach.walking() && approach.tracking() && !approach.aimed()
                            && !approach.brush() && !tracking.approaching(), "bounded approach action misclassified");
            var resume = DiagnosticClientActions.Request.parse(valid.replace("brush", "resume_game_menu"));
            check(resume.resumingMenu() && !resume.walking() && !resume.tracking() && !resume.aimed()
                            && !resume.brush() && !approach.resumingMenu(), "UI resume entered a gameplay path");
            for (String action : List.of("aim_and_brush", "aim_and_attack")) {
                var parsed = DiagnosticClientActions.Request.parse(valid.replace("brush", action));
                check(parsed.action().equals(action) && parsed.aimed(), "explicit aimed action rejected");
                check(parsed.brush() == action.equals("aim_and_brush"), "aimed gesture misclassified");
            }
        } catch (ProtocolException bad) { throw new AssertionError(bad); }
        for (String invalid : List.of(valid.replace("\"seq\":0", "\"seq\":32"),
                valid.replace("\"seq\":0", "\"seq\":true"),
                valid.replace("\"seq\":0", "\"seq\":0.5"),
                valid.replace("\"seq\":0", "\"seq\":0,\"seq\":0"),
                valid.replace("brush", "teleport"), valid.replace("140", "201"),
                valid.replace("100", "-1"), valid.replace("140", "100"),
                valid.replace("140", "9223372036854775808"), valid.replace("}", ",\"extra\":1}"))) {
            try { DiagnosticClientActions.Request.parse(invalid); throw new AssertionError("accepted " + invalid); }
            catch (ProtocolException | IllegalArgumentException expected) { /* Correct refusal. */ }
        }
        Path directory = Files.createTempDirectory("neurocraft-client-actions-file-test-");
        Path file = Files.createFile(directory.resolve("input.ndjson"));
        Path link = directory.resolve("link.ndjson");
        Files.createSymbolicLink(link, file);
        try { new DiagnosticClientActions(link); throw new AssertionError("symlink accepted"); }
        catch (IOException expected) { /* Correct refusal. */ }
        Files.writeString(file, "{}\n");
        try { new DiagnosticClientActions(file); throw new AssertionError("nonempty inbox accepted"); }
        catch (IOException expected) { /* Correct refusal. */ }
        Files.writeString(file, "");
        try (var first = new DiagnosticClientActions(file)) { check(!first.isClosed(), "fresh inbox closed"); }
        byte[] original = Files.readAllBytes(results(file));
        try { new DiagnosticClientActions(file); throw new AssertionError("result evidence overwritten"); }
        catch (IOException expected) { /* Correct refusal. */ }
        check(java.util.Arrays.equals(original, Files.readAllBytes(results(file))), "result file changed");
    }

    private static void verifyBoundedRefusals(ClientGameTestContext context, UUID flyId) throws IOException {
        Path path = Files.createTempDirectory("neurocraft-client-actions-cap-").resolve("input.ndjson");
        Files.createFile(path);
        try (var inbox = new DiagnosticClientActions(path)) {
            long[] clock = {System.nanoTime()};
            pulse(context, inbox, clock);
            for (int seq = 0; seq < 32; seq++) {
                long tick = clientTick(context);
                append(path, request(seq, "attack", flyId, tick + 20, tick + 40));
                pulse(context, inbox, clock);
            }
            check(inbox.isClosed(), "32 requests did not close the inbox");
            check(Files.readAllLines(results(path)).size() == 32, "request bound or one-shot accounting failed");
        }
    }

    private static void verifyInvalidFileDisables(ClientGameTestContext context) throws IOException {
        for (boolean oversized : new boolean[] {false, true}) {
            Path path = Files.createTempDirectory("neurocraft-client-actions-invalid-").resolve("input.ndjson");
            Files.createFile(path);
            try (var inbox = new DiagnosticClientActions(path)) {
                Files.write(path, oversized ? new byte[16_385] : new byte[] {(byte) 0xff, '\n'});
                context.runOnClient(client -> inbox.tick(client, System.nanoTime()));
                check(inbox.isClosed(), "invalid file left inbox active");
            }
        }
    }

    private static void aim(ClientGameTestContext context, UUID flyId) {
        context.waitFor(client -> client.level != null && findFly(client, flyId) != null);
        context.runOnClient(client -> {
            var fly = findFly(client, flyId);
            client.player.lookAt(EntityAnchorArgument.Anchor.EYES, fly.getBoundingBox().getCenter());
        });
        context.waitTicks(2);
        context.runOnClient(client -> client.gameRenderer.pick(1));
        context.waitFor(client -> client.hitResult instanceof EntityHitResult hit
                && hit.getEntity().getUUID().equals(flyId));
    }

    private static FlyEntity findFly(net.minecraft.client.Minecraft client, UUID id) {
        for (var entity : client.level.entitiesForRendering()) {
            if (entity instanceof FlyEntity fly && fly.getUUID().equals(id)) { return fly; }
        }
        return null;
    }

    private static void equip(TestSingleplayerContext world, ClientGameTestContext context, String item) {
        world.getServer().runCommand("item replace entity @p weapon.mainhand with " + item);
        context.waitFor(client -> client.player != null &&
                net.minecraft.core.registries.BuiltInRegistries.ITEM
                        .getKey(client.player.getMainHandItem().getItem()).toString().equals(item));
        context.waitTicks(2);
    }

    private static void expectCounts(TestSingleplayerContext world, ClientGameTestContext context,
            UUID flyId, long brush, long attack) {
        for (int tick = 0; tick < 40; tick++) {
            var events = world.getServer().computeOnServer(server ->
                    ((FlyEntity) server.overworld().getEntity(flyId)).eventSample());
            if (events.brush().count() == brush && events.attackAttempt().count() == attack) { return; }
            context.waitTick();
        }
        throw new AssertionError("server counters did not reach brush=" + brush + " attack=" + attack);
    }

    private static long clientTick(ClientGameTestContext context) {
        return context.computeOnClient(client -> client.level.getGameTime());
    }

    private static void pulse(ClientGameTestContext context, DiagnosticClientActions inbox, long[] clock) {
        clock[0] += DiagnosticClientActions.DISPATCH_INTERVAL_NS;
        context.runOnClient(client -> inbox.tick(client, clock[0]));
    }

    private static String request(int seq, String action, UUID fly, long tick, long through) {
        return "{\"seq\":" + seq + ",\"action\":\"" + action + "\",\"entity_uuid\":\"" + fly
                + "\",\"game_tick\":" + tick + ",\"valid_through_tick\":" + through + "}";
    }

    private static void append(Path path, String line) throws IOException {
        Files.writeString(path, line + "\n", StandardCharsets.UTF_8, StandardOpenOption.APPEND);
    }

    private static Path results(Path input) { return input.resolveSibling("diagnostic-client-actions-results.ndjson"); }
    private static void assertLastReason(Path path, String reason) throws IOException {
        String row = Files.readAllLines(results(path)).getLast();
        check(row.contains("\"status\":\"REFUSED\"") && row.contains("\"reason\":\"" + reason + "\""), row);
    }
    private static void check(boolean condition, String detail) { if (!condition) { throw new AssertionError(detail); } }
}
