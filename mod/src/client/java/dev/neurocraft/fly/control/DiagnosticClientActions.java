package dev.neurocraft.fly.control;

import dev.neurocraft.fly.entity.FlyEntity;
import java.io.IOException;
import java.math.BigDecimal;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;
import java.util.function.Consumer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientLifecycleEvents;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.minecraft.client.Minecraft;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.gui.screens.PauseScreen;
import net.minecraft.commands.arguments.EntityAnchorArgument;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.level.ClipContext;
import net.minecraft.world.phys.EntityHitResult;
import net.minecraft.world.phys.HitResult;
import net.minecraft.world.phys.Vec3;
import net.minecraft.world.item.Items;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Opt-in synthetic client interactions through vanilla gameMode and its real network connection.
 * Dispatch is not human input, a server event receipt, or evidence of a neural program being applied.
 * This client-only class shares the strict codec's package, not any controller authority.
 */
public final class DiagnosticClientActions implements AutoCloseable {
    public static final String PATH_PROPERTY = "neurocraft.fly.clientActionsPath";
    static final int BYTE_LIMIT = 16_384;
    static final int COMMAND_LIMIT = 32;
    static final long POLL_NS = 200_000_000L;
    static final long DISPATCH_INTERVAL_NS = 1_000_000_000L;
    static final int WALK_CLIENT_TICKS = 20;
    static final long WALK_TIMEOUT_NS = 3_000_000_000L;
    static final double WALK_TARGET_DISTANCE_BLOCKS = 6.0;
    static final double AIM_TARGET_DISTANCE_BLOCKS = 6.0;
    static final double TRACK_MIN_HORIZONTAL_BLOCKS = 0.05;
    static final double APPROACH_STOP_HORIZONTAL_BLOCKS = 0.65;
    private static final String TRACK_METHOD = "player.lookAt(EYES,target.getBoundingBox().getCenter())";
    private static final Logger LOGGER = LoggerFactory.getLogger("neurocraft-client-actions");
    private final Path path;
    private final Object fileKey;
    private final FileChannel results;
    private byte[] consumed = new byte[0];
    private int nextSequence;
    private long lastPoll = Long.MIN_VALUE;
    private long lastDispatch = Long.MIN_VALUE;
    private Object world;
    private Object connection;
    private UUID player;
    private WalkHold walk;
    private boolean closed;

    DiagnosticClientActions(Path path) throws IOException {
        this.path = path.toAbsolutePath().normalize();
        BasicFileAttributes attributes = attributes();
        if (attributes.size() != 0) {
            throw new IOException("client action inbox must begin empty");
        }
        fileKey = attributes.fileKey();
        results = FileChannel.open(this.path.resolveSibling("diagnostic-client-actions-results.ndjson"),
                StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE, LinkOption.NOFOLLOW_LINKS);
    }

    public static void register() {
        register(System.getProperty(PATH_PROPERTY), ClientTickEvents.END_CLIENT_TICK::register,
                ClientLifecycleEvents.CLIENT_STOPPING::register);
    }

    static void register(String configured, Consumer<ClientTickEvents.EndTick> ticks,
            Consumer<ClientLifecycleEvents.ClientStopping> stopping) {
        if (configured == null || configured.isBlank()) { return; }
        try {
            var inbox = new DiagnosticClientActions(Path.of(configured));
            ticks.accept(client -> inbox.tick(client, System.nanoTime()));
            stopping.accept(client -> inbox.close());
        } catch (IOException | RuntimeException invalid) {
            LOGGER.warn("Diagnostic client actions disabled: invalid fresh inbox", invalid);
        }
    }

    void tick(Minecraft client, long now) {
        if (closed) { return; }
        // Inspect lifecycle every tick, even between file polls. Never carry requests into a new world.
        if (world != null && (world != client.level || connection != client.getConnection()
                || client.player == null || !player.equals(client.player.getUUID()))) {
            disable("world_or_connection_changed", client);
            return;
        }
        if (client.level == null || client.player == null || client.gameMode == null
                || client.getConnection() == null) {
            if (walk != null) { disable("client_unavailable", client); }
            return;
        }
        long tick = client.level.getGameTime();
        // ClientLevel predicts this clock locally; ordinary server time packets
        // can correct it backwards within the same world. It is a request-window
        // clock, not a lifecycle identity or monotonic dispatch-rate clock.
        // Keep the world/connection/player fence above and validate every request
        // against the current tick below. Consumed sequences never replay.
        world = client.level;
        connection = client.getConnection();
        player = client.player.getUUID();
        boolean walkAtCallbackEntry = walk != null;
        if (walk != null) {
            try { advanceWalk(client, now, tick); }
            catch (IOException | RuntimeException invalid) {
                disable("walk_result_write_failed", client);
                return;
            }
            if (walk == null && nextSequence == COMMAND_LIMIT) { close(); return; }
        }
        if (lastPoll != Long.MIN_VALUE && now - lastPoll < POLL_NS) { return; }
        lastPoll = now;
        try {
            String line = nextLine();
            if (line == null) { return; }
            Request request = Request.parse(line);
            if (request.seq() != nextSequence) {
                writeResult(client, request, "REFUSED", "unexpected_sequence", List.of());
                close();
                return;
            }
            nextSequence++;
            List<String> methods = new ArrayList<>();
            FlyEntity aimTarget = request.aimed() ? findFly(client, request.entityId()) : null;
            String refusal = walk != null || request.resumingMenu() && walkAtCallbackEntry ? "walk_in_progress"
                    : request.resumingMenu() ? resumeMenuRefusal(client, request, tick)
                    : request.aimed() ? aimRefusal(client, request, tick, aimTarget)
                    : refusal(client, request, tick);
            if (refusal == null && lastDispatch != Long.MIN_VALUE
                    && now - lastDispatch < DISPATCH_INTERVAL_NS) {
                refusal = "rate_limited";
            }
            if (refusal != null) {
                writeResult(client, request, "REFUSED", refusal, methods);
            } else if (request.resumingMenu()) {
                lastDispatch = now;
                UiEvidence ui = new UiEvidence(client);
                if (client.screen == null) {
                    writeMenuResult(client, request, "ALREADY_CLEAR", "game_screen_already_clear", methods, ui);
                } else {
                    methods.add("Minecraft.setScreen(null)");
                    try {
                        // Vanilla setScreen(null) invokes sound resume and mouse capture handling.
                        // This is a one-shot UI request, never an automatic gameplay action.
                        client.setScreen(null);
                        boolean cleared = client.screen == null && client.getOverlay() == null;
                        writeMenuResult(client, request, cleared ? "RESUMED" : "DISPATCH_ERROR",
                                cleared ? "pause_screen_closed" : "pause_screen_did_not_clear", methods, ui);
                        if (!cleared) { close(); }
                    } catch (RuntimeException failure) {
                        writeMenuResult(client, request, "DISPATCH_ERROR", "ui_resume_result_unknown", methods, ui);
                        close();
                    }
                }
            } else if (request.walking()) {
                lastDispatch = now;
                walk = new WalkHold(request, client, now, tick);
                if (request.tracking()) {
                    trackTarget(client, walk, tick);
                    methods.add(TRACK_METHOD);
                }
                walk.forward.setDown(true);
                methods.add("options.keyUp.setDown(true)");
                writeResult(client, request, "WALK_STARTED", "server_movement_unconfirmed",
                        methods, walk, now, false);
            } else {
                // A view change is itself a rate-limited attempt, even if the real picker refuses.
                lastDispatch = now;
                AimEvidence aim = null;
                try {
                    if (request.aimed()) {
                        aim = new AimEvidence(client, aimTarget);
                        methods.add("player.lookAt(EYES,target.getBoundingBox().getCenter())");
                        try {
                            client.player.lookAt(EntityAnchorArgument.Anchor.EYES,
                                    aimTarget.getBoundingBox().getCenter());
                        } finally { aim.afterView = clientView(client); }
                        methods.add("gameRenderer.pick(1.0F)");
                        client.gameRenderer.pick(1.0F);
                        aim.pickerResult = pickerResult(client.hitResult);
                        // Same callback, real native picker, same strict click gates. A nearer
                        // entity or a miss never becomes an authored/fabricated hit or a retry.
                        refusal = refusal(client, request, tick);
                    }
                    if (refusal != null) {
                        writeAimedResult(client, request, "REFUSED", refusal, methods, aim);
                    } else {
                        EntityHitResult hit = (EntityHitResult) client.hitResult;
                        if (request.brush()) {
                            methods.add("gameMode.interactAt(MAIN_HAND)");
                            var result = client.gameMode.interactAt(client.player, hit.getEntity(), hit,
                                    InteractionHand.MAIN_HAND);
                            if (!result.consumesAction()) {
                                methods.add("gameMode.interact(MAIN_HAND)");
                                client.gameMode.interact(client.player, hit.getEntity(), InteractionHand.MAIN_HAND);
                            }
                        } else {
                            methods.add("gameMode.attack");
                            client.gameMode.attack(client.player, hit.getEntity());
                            methods.add("player.swing(MAIN_HAND)");
                            client.player.swing(InteractionHand.MAIN_HAND);
                        }
                        writeAimedResult(client, request, "DISPATCHED", "server_result_unconfirmed", methods, aim);
                    }
                } catch (RuntimeException failure) {
                    // A method may have sent a packet before throwing. Never retry or call it REFUSED.
                    writeAimedResult(client, request, "DISPATCH_ERROR", "server_result_unknown", methods, aim);
                    close();
                }
            }
            if (nextSequence == COMMAND_LIMIT && walk == null) { close(); }
        } catch (IOException | ProtocolException | RuntimeException invalid) {
            disable("invalid_inbox_or_result_write", client);
        }
    }

    private BasicFileAttributes attributes() throws IOException {
        var attributes = Files.readAttributes(path, BasicFileAttributes.class, LinkOption.NOFOLLOW_LINKS);
        if (!attributes.isRegularFile() || attributes.isSymbolicLink() || attributes.size() > BYTE_LIMIT) {
            throw new IOException("client action inbox must be a bounded regular nonsymlink file");
        }
        return attributes;
    }

    private String nextLine() throws IOException {
        var attributes = attributes();
        if (!Objects.equals(fileKey, attributes.fileKey())) { throw new IOException("inbox replaced"); }
        byte[] bytes;
        try (var input = Files.newInputStream(path, LinkOption.NOFOLLOW_LINKS)) {
            bytes = input.readNBytes(BYTE_LIMIT + 1);
        }
        if (bytes.length > BYTE_LIMIT || bytes.length < consumed.length
                || !Arrays.equals(consumed, Arrays.copyOf(bytes, consumed.length))) {
            throw new IOException("inbox exceeded its bound or changed consumed bytes");
        }
        int end = consumed.length;
        while (end < bytes.length && bytes[end] != '\n') { end++; }
        if (end == bytes.length) { return null; }
        String line = StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
                .decode(ByteBuffer.wrap(bytes, consumed.length, end - consumed.length)).toString();
        consumed = Arrays.copyOf(bytes, end + 1);
        return line;
    }

    private static String commonRefusal(Minecraft client, Request request, long tick) {
        if (tick < request.gameTick()) { return "future_request"; }
        if (tick > request.validThroughTick()) { return "expired_request"; }
        if (client.screen != null || client.getOverlay() != null) { return "screen_or_overlay_open"; }
        if (!client.player.isAlive() || client.player.isSpectator()) { return "player_not_live_nonspectator"; }
        if (client.player.isHandsBusy() || client.player.isUsingItem() || client.gameMode.isDestroying()
                || client.options.keyAttack.isDown() || client.options.keyUse.isDown()) { return "player_busy"; }
        return null;
    }

    private static String resumeMenuRefusal(Minecraft client, Request request, long tick) {
        if (tick < request.gameTick()) { return "future_request"; }
        if (tick > request.validThroughTick()) { return "expired_request"; }
        if (client.getOverlay() != null) { return "overlay_open"; }
        if (client.screen != null && client.screen.getClass() != PauseScreen.class) {
            return "screen_is_not_pause_menu";
        }
        if (!client.player.isAlive()) { return "player_not_alive"; }
        if (client.player.isHandsBusy() || client.player.isUsingItem() || client.gameMode.isDestroying()
                || client.options.keyAttack.isDown() || client.options.keyUse.isDown()) { return "player_busy"; }
        if (client.options.keyUp.isDown() || client.options.keyDown.isDown()
                || client.options.keyLeft.isDown() || client.options.keyRight.isDown()
                || client.options.keyJump.isDown() || client.options.keyShift.isDown()
                || client.options.keySprint.isDown()) { return "conflicting_movement_input"; }
        for (KeyMapping key : client.options.keyMappings) {
            if (key.isDown()) { return "conflicting_key_input"; }
        }
        FlyEntity fly = findFly(client, request.entityId());
        if (fly == null || !fly.isAlive() || fly.isRemoved() || fly.level() != client.level) {
            return "target_unavailable";
        }
        return null;
    }

    private static String refusal(Minecraft client, Request request, long tick) {
        String common = commonRefusal(client, request, tick);
        if (common != null) { return common; }
        if (request.walking()) { return walkRefusal(client, request, false); }
        if (!(client.hitResult instanceof EntityHitResult hit)
                || !(hit.getEntity() instanceof FlyEntity fly)) { return "crosshair_is_not_fly"; }
        if (!fly.getUUID().equals(request.entityId())) { return "wrong_target_uuid"; }
        if (fly.isRemoved() || fly.level() != client.level || !fly.isPickable()) { return "target_unavailable"; }
        if (!client.player.canInteractWithEntity(fly, 0.0)) { return "out_of_reach"; }
        if (!client.level.getWorldBorder().isWithinBounds(fly.blockPosition())) { return "outside_world_border"; }
        var eye = client.player.getEyePosition();
        var point = hit.getLocation();
        if (!fly.getBoundingBox().inflate(1.0e-5).contains(point)) { return "stale_hit_location"; }
        var obstruction = client.level.clip(new ClipContext(eye, point,
                ClipContext.Block.COLLIDER, ClipContext.Fluid.NONE, client.player));
        if (obstruction.getType() != HitResult.Type.MISS) { return "blocked_line_of_sight"; }
        return equipmentRefusal(client, request, fly);
    }

    private static String equipmentRefusal(Minecraft client, Request request, FlyEntity fly) {
        if (request.brush()) {
            if (!client.player.getMainHandItem().is(Items.BRUSH)) { return "main_hand_is_not_brush"; }
            if (fly.isInWater() || fly.isInLava()) { return "fly_in_fluid"; }
        } else if (!client.player.getMainHandItem().isEmpty()) { return "attack_requires_empty_main_hand"; }
        return null;
    }

    private static FlyEntity findFly(Minecraft client, UUID id) {
        for (var entity : client.level.entitiesForRendering()) {
            if (entity instanceof FlyEntity fly && fly.getUUID().equals(id)) { return fly; }
        }
        return null;
    }

    private static String aimRefusal(Minecraft client, Request request, long tick, FlyEntity fly) {
        String common = commonRefusal(client, request, tick);
        if (common != null) { return common; }
        if (client.getCameraEntity() != client.player) { return "camera_is_not_player"; }
        if (client.options.keyUp.isDown() || client.options.keyDown.isDown()
                || client.options.keyLeft.isDown() || client.options.keyRight.isDown()
                || client.options.keyJump.isDown() || client.options.keyShift.isDown()
                || client.options.keySprint.isDown()) { return "conflicting_movement_input"; }
        if (fly == null || !fly.isAlive() || fly.isRemoved() || fly.level() != client.level
                || !fly.isPickable()) { return "target_unavailable"; }
        Vec3 eye = client.player.getEyePosition(), point = fly.getBoundingBox().getCenter();
        if (!finite(eye) || !finite(point)) { return "invalid_target_geometry"; }
        if (eye.distanceToSqr(point) > AIM_TARGET_DISTANCE_BLOCKS * AIM_TARGET_DISTANCE_BLOCKS) {
            return "aim_target_out_of_range";
        }
        if (!client.player.canInteractWithEntity(fly, 0.0)) { return "out_of_reach"; }
        if (!client.level.getWorldBorder().isWithinBounds(fly.blockPosition())) {
            return "outside_world_border";
        }
        if (client.level.clip(new ClipContext(eye, point, ClipContext.Block.COLLIDER,
                ClipContext.Fluid.NONE, client.player)).getType() != HitResult.Type.MISS) {
            return "blocked_line_of_sight";
        }
        return equipmentRefusal(client, request, fly);
    }

    private static boolean finite(Vec3 value) {
        return Double.isFinite(value.x) && Double.isFinite(value.y) && Double.isFinite(value.z);
    }

    private static Map<String, Object> clientView(Minecraft client) {
        return Map.of("yaw_degrees", client.player.getYRot(), "pitch_degrees", client.player.getXRot());
    }

    private static Map<String, Object> pickerResult(HitResult hit) {
        if (hit == null) { return null; }
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("type", hit.getType().name());
        result.put("entity_uuid", hit instanceof EntityHitResult entityHit
                ? entityHit.getEntity().getStringUUID() : null);
        result.put("hit_position_blocks", coordinates(hit.getLocation()));
        return result;
    }

    private static List<Double> coordinates(Vec3 position) {
        return List.of(position.x, position.y, position.z);
    }

    private static final class AimEvidence {
        final Map<String, Object> beforeView;
        final List<Double> eyePosition;
        final List<Double> targetPosition;
        final List<Double> targetCentre;
        Map<String, Object> afterView;
        Map<String, Object> pickerResult;

        AimEvidence(Minecraft client, FlyEntity target) {
            beforeView = clientView(client);
            eyePosition = coordinates(client.player.getEyePosition());
            targetPosition = coordinates(target.position());
            targetCentre = coordinates(target.getBoundingBox().getCenter());
        }

        Map<String, Object> material() {
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("before_client_view", beforeView);
            result.put("after_client_view", afterView);
            result.put("eye_client_position_blocks", eyePosition);
            result.put("target_client_position_blocks", targetPosition);
            result.put("target_client_center_blocks", targetCentre);
            result.put("picker_result", pickerResult);
            return result;
        }
    }

    private static final class UiEvidence {
        final String beforeScreen;
        final String beforeOverlay;
        final boolean beforeWindowActive;
        final boolean beforePauseOnLostFocus;

        UiEvidence(Minecraft client) {
            beforeScreen = client.screen == null ? null : client.screen.getClass().getName();
            beforeOverlay = client.getOverlay() == null ? null : client.getOverlay().getClass().getName();
            beforeWindowActive = client.isWindowActive();
            beforePauseOnLostFocus = client.options.pauseOnLostFocus;
        }

        Map<String, Object> material(Minecraft client) {
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("before_screen", beforeScreen);
            result.put("after_screen", client.screen == null ? null : client.screen.getClass().getName());
            result.put("before_overlay", beforeOverlay);
            result.put("after_overlay", client.getOverlay() == null ? null : client.getOverlay().getClass().getName());
            result.put("screen_clear", client.screen == null && client.getOverlay() == null);
            result.put("before_window_active", beforeWindowActive);
            result.put("after_window_active", client.isWindowActive());
            result.put("before_pause_on_lost_focus", beforePauseOnLostFocus);
            result.put("after_pause_on_lost_focus", client.options.pauseOnLostFocus);
            return result;
        }
    }

    private static String walkRefusal(Minecraft client, Request request, boolean ownsForward) {
        if (request.tracking() && client.getCameraEntity() != client.player) {
            return "camera_is_not_player";
        }
        if (!client.player.onGround() || client.player.getAbilities().flying) {
            return "walking_requires_grounded_nonflying_player";
        }
        if (client.options.keyDown.isDown() || client.options.keyLeft.isDown()
                || client.options.keyRight.isDown() || client.options.keyJump.isDown()
                || client.options.keyShift.isDown() || client.options.keySprint.isDown()
                || (!ownsForward && client.options.keyUp.isDown())) {
            return "conflicting_movement_input";
        }
        if (ownsForward && !client.options.keyUp.isDown()) { return "forward_input_released"; }
        FlyEntity target = null;
        for (var entity : client.level.entitiesForRendering()) {
            if (entity.getUUID().equals(request.entityId()) && entity instanceof FlyEntity fly) {
                target = fly;
                break;
            }
        }
        if (target == null || !target.isAlive() || target.isRemoved()
                || target.level() != client.level || !target.isPickable()) {
            return "target_unavailable";
        }
        if (!client.level.getWorldBorder().isWithinBounds(target.blockPosition())) {
            return "outside_world_border";
        }
        var eye = client.player.getEyePosition();
        var point = target.getBoundingBox().getCenter();
        if (request.tracking()) {
            String invalid = trackingGeometryRefusal(eye, point);
            if (invalid != null) { return invalid; }
        }
        if (eye.distanceToSqr(point) > WALK_TARGET_DISTANCE_BLOCKS * WALK_TARGET_DISTANCE_BLOCKS) {
            return "walking_target_out_of_range";
        }
        if (client.level.clip(new ClipContext(eye, point, ClipContext.Block.COLLIDER,
                ClipContext.Fluid.NONE, client.player)).getType() != HitResult.Type.MISS) {
            return "blocked_line_of_sight";
        }
        if (request.approaching() && !ownsForward && insideApproachStopRadius(eye, point)) {
            return "walking_target_already_within_stop_distance";
        }
        return null;
    }

    private static double horizontalDistance(Vec3 eye, Vec3 point) {
        return Math.hypot(point.x - eye.x, point.z - eye.z);
    }

    static boolean insideApproachStopRadius(Vec3 eye, Vec3 point) {
        return horizontalDistance(eye, point) <= APPROACH_STOP_HORIZONTAL_BLOCKS;
    }

    /** A vertical/coincident target has no stable walking heading; never retain an arbitrary yaw. */
    static String trackingGeometryRefusal(Vec3 eye, Vec3 point) {
        if (!finite(eye) || !finite(point)) { return "invalid_tracking_geometry"; }
        double dx = point.x - eye.x, dz = point.z - eye.z;
        if (dx * dx + dz * dz <= TRACK_MIN_HORIZONTAL_BLOCKS * TRACK_MIN_HORIZONTAL_BLOCKS) {
            return "tracking_horizontal_heading_undefined";
        }
        return null;
    }

    private static void trackTarget(Minecraft client, WalkHold held, long tick) {
        FlyEntity target = findFly(client, held.request.entityId());
        Map<String, Object> observation = new LinkedHashMap<>();
        observation.put("game_tick", tick);
        observation.put("before_client_view", clientView(client));
        observation.put("eye_client_position_blocks", coordinates(client.player.getEyePosition()));
        observation.put("target_client_position_blocks", coordinates(target.position()));
        observation.put("target_client_center_blocks", coordinates(target.getBoundingBox().getCenter()));
        // Native lookAt computes wrapped yaw and pitch from the fresh center. Only view fields change.
        client.player.lookAt(EntityAnchorArgument.Anchor.EYES, target.getBoundingBox().getCenter());
        observation.put("after_client_view", clientView(client));
        held.trackingAimCount++;
        if (held.firstTrackingObservation == null) { held.firstTrackingObservation = observation; }
        held.lastTrackingObservation = observation;
    }

    private void advanceWalk(Minecraft client, long now, long tick) throws IOException {
        WalkHold held = walk;
        held.observedClientTicks++;
        String reason = now - held.startedNs >= WALK_TIMEOUT_NS ? "walking_wall_timeout" : null;
        if (reason == null && (client.player != held.playerObject || client.options.keyUp != held.forward)) {
            reason = "player_or_key_binding_changed";
        }
        if (reason == null && (tick < held.request.gameTick() || tick > held.request.validThroughTick())) {
            reason = "walking_request_window_closed";
        }
        if (reason == null && (client.screen != null || client.getOverlay() != null)) {
            reason = "screen_or_overlay_open";
        }
        if (reason == null && (!client.player.isAlive() || client.player.isSpectator())) {
            reason = "player_not_live_nonspectator";
        }
        if (reason == null && (client.player.isHandsBusy() || client.player.isUsingItem()
                || client.gameMode.isDestroying() || client.options.keyAttack.isDown()
                || client.options.keyUse.isDown())) { reason = "player_busy"; }
        if (reason == null) { reason = walkRefusal(client, held.request, true); }
        if (reason != null) { finishWalk(client, now, "WALK_ABORTED", reason); }
        else if (held.request.approaching() && insideApproachStopRadius(client.player.getEyePosition(),
                findFly(client, held.request.entityId()).getBoundingBox().getCenter())) {
            finishWalk(client, now, "WALK_COMPLETED", "walking_stop_distance_reached");
        }
        else if (held.observedClientTicks >= WALK_CLIENT_TICKS) {
            finishWalk(client, now, "WALK_COMPLETED", held.request.approaching()
                    ? "walking_callback_limit_reached" : "key_hold_complete_server_movement_unconfirmed");
        } else if (held.request.tracking()) {
            trackTarget(client, held, tick);
        }
    }

    private void finishWalk(Minecraft client, long now, String status, String reason) throws IOException {
        WalkHold held = walk;
        if (held == null) { return; }
        try {
            if (held.request.approaching()) { held.terminalGeometry = approachTerminalGeometry(client, held); }
        } finally {
            // Geometry/log failures must still relinquish the only key this diagnostic owns.
            held.forward.setDown(false);
            walk = null;
        }
        writeResult(client, held.request, status, reason, List.of("options.keyUp.setDown(false)"),
                held, now, true);
    }

    private static Map<String, Object> approachTerminalGeometry(Minecraft client, WalkHold held) {
        if (client.level != held.level || client.player != held.playerObject) { return null; }
        FlyEntity target = findFly(client, held.request.entityId());
        if (target == null || target.isRemoved() || target.level() != held.level) { return null; }
        Vec3 eye = client.player.getEyePosition(), center = target.getBoundingBox().getCenter();
        double horizontal = horizontalDistance(eye, center);
        if (!finite(eye) || !finite(center) || !finite(target.position()) || !Double.isFinite(horizontal)) {
            return null;
        }
        return Map.of("game_tick", client.level.getGameTime(),
                "eye_client_position_blocks", coordinates(eye),
                "target_client_position_blocks", coordinates(target.position()),
                "target_client_center_blocks", coordinates(center),
                "horizontal_distance_blocks", horizontal);
    }

    private static Map<String, Object> playerPose(Minecraft client) {
        var player = client.player;
        return Map.of("position_blocks", List.of(player.getX(), player.getY(), player.getZ()),
                "yaw_degrees", player.getYRot(), "pitch_degrees", player.getXRot(),
                "on_ground", player.onGround(), "flying", player.getAbilities().flying);
    }

    private static final class WalkHold {
        final Request request;
        final Minecraft client;
        final KeyMapping forward;
        final Object playerObject;
        final Object level;
        final long startedNs;
        final long startedGameTick;
        final Map<String, Object> startPose;
        int observedClientTicks;
        int trackingAimCount;
        Map<String, Object> firstTrackingObservation;
        Map<String, Object> lastTrackingObservation;
        Map<String, Object> terminalGeometry;

        WalkHold(Request request, Minecraft client, long now, long tick) {
            this.request = request;
            this.client = client;
            forward = client.options.keyUp;
            playerObject = client.player;
            level = client.level;
            startedNs = now;
            startedGameTick = tick;
            startPose = playerPose(client);
        }
    }

    private void writeResult(Minecraft client, Request request, String status, String reason,
            List<String> methods) throws IOException {
        writeResult(client, request, status, reason, methods, null, 0, false);
    }

    private void writeAimedResult(Minecraft client, Request request, String status, String reason,
            List<String> methods, AimEvidence aim) throws IOException {
        writeResult(client, request, status, reason, methods, null, 0, false, aim);
    }

    private void writeMenuResult(Minecraft client, Request request, String status, String reason,
            List<String> methods, UiEvidence ui) throws IOException {
        writeResult(client, request, status, reason, methods, null, 0, false, null, ui);
    }

    private void writeResult(Minecraft client, Request request, String status, String reason,
            List<String> methods, WalkHold held, long now, boolean ended) throws IOException {
        writeResult(client, request, status, reason, methods, held, now, ended, null);
    }

    private void writeResult(Minecraft client, Request request, String status, String reason,
            List<String> methods, WalkHold held, long now, boolean ended, AimEvidence aim) throws IOException {
        writeResult(client, request, status, reason, methods, held, now, ended, aim, null);
    }

    private void writeResult(Minecraft client, Request request, String status, String reason,
            List<String> methods, WalkHold held, long now, boolean ended, AimEvidence aim,
            UiEvidence ui) throws IOException {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("schema", "neurocraft.client_action_dispatch.v1");
        row.put("scope", request != null && request.resumingMenu()
                ? "explicit synthetic pause-menu resume; observed client UI only, not gameplay, "
                        + "human input, neural resume or controller authority"
                : request != null && request.aimed()
                ? "synthetic client view aiming and native gameplay request; not human input, "
                        + "server event receipt or neural application"
                : request != null && request.approaching()
                ? "synthetic bounded target approach through native gameplay; stopping requests key release, "
                        + "not immediate rest; client poses are not human input, authoritative server movement or neural application"
                : request != null && request.tracking()
                ? "synthetic target-tracking view and forward key through native gameplay; client poses are not "
                        + "human input, authoritative server movement or neural application"
                : request != null && request.walking()
                ? "synthetic forward key hold through native gameplay; client poses are not human input, "
                        + "authoritative server movement or neural application"
                : "synthetic client request; not human input, server event receipt or neural application");
        row.put("seq", request == null ? null : request.seq());
        row.put("action", request == null ? null : request.action());
        row.put("requested_entity_uuid", request == null ? null : request.entityId().toString());
        row.put("request_game_tick", request == null ? null : request.gameTick());
        row.put("valid_through_tick", request == null ? null : request.validThroughTick());
        row.put("client_world_game_tick", client.level == null ? null : client.level.getGameTime());
        row.put("dimension", client.level == null ? null : client.level.dimension().location().toString());
        row.put("player_uuid", client.player == null ? null : client.player.getStringUUID());
        row.put("game_mode", client.gameMode == null ? null : client.gameMode.getPlayerMode().getName());
        row.put("main_hand_item", client.player == null ? null
                : BuiltInRegistries.ITEM.getKey(client.player.getMainHandItem().getItem()).toString());
        row.put("off_hand_item", client.player == null ? null
                : BuiltInRegistries.ITEM.getKey(client.player.getOffhandItem().getItem()).toString());
        row.put("crosshair_entity_uuid", client.hitResult instanceof EntityHitResult hit
                ? hit.getEntity().getStringUUID() : null);
        row.put("status", status);
        row.put("reason", reason);
        row.put("attempted_methods", List.copyOf(methods));
        if (request != null && request.resumingMenu()) {
            row.put("ui", (ui == null ? new UiEvidence(client) : ui).material(client));
        }
        if (request != null && request.aimed()) { row.put("aim", aim == null ? null : aim.material()); }
        if (held != null) {
            Map<String, Object> evidence = new LinkedHashMap<>();
            boolean samePlayer = client.player == held.playerObject && client.level == held.level;
            evidence.put(held.request.approaching() ? "max_client_ticks" : "fixed_client_ticks", WALK_CLIENT_TICKS);
            evidence.put("wall_timeout_ns", WALK_TIMEOUT_NS);
            evidence.put("observed_client_ticks", held.observedClientTicks);
            evidence.put("elapsed_ns", now >= held.startedNs ? now - held.startedNs : null);
            evidence.put("start_game_tick", held.startedGameTick);
            evidence.put("start_client_pose", held.startPose);
            evidence.put("end_game_tick", ended && samePlayer ? client.level.getGameTime() : null);
            evidence.put("end_client_pose", ended && samePlayer ? playerPose(client) : null);
            evidence.put("forward_key_released", ended && !held.forward.isDown());
            if (held.request.approaching()) {
                evidence.put("stop_horizontal_distance_blocks", APPROACH_STOP_HORIZONTAL_BLOCKS);
                evidence.put("completion_reason", status.equals("WALK_COMPLETED")
                        ? reason.equals("walking_stop_distance_reached") ? "stop_distance_reached" : "callback_limit_reached"
                        : null);
                evidence.put("release_scope", "forward key release; native momentum and subsequent coasting remain");
                evidence.put("terminal_geometry", held.terminalGeometry);
            }
            if (held.request.tracking()) {
                Map<String, Object> tracking = new LinkedHashMap<>();
                tracking.put("mode", "direct_fresh_target_center");
                tracking.put("rule", "lookAt(EYES,target bounding-box center) before forward key-down and each "
                        + "continuing held callback; no aim on finish or abort");
                tracking.put("aim_count", held.trackingAimCount);
                tracking.put("initial_observation", held.firstTrackingObservation);
                tracking.put("last_observation", held.lastTrackingObservation);
                evidence.put("tracking", tracking);
            }
            row.put("walk", evidence);
        }
        ByteBuffer encoded = StandardCharsets.UTF_8.encode(StrictJson.encode(row, 4096) + "\n");
        while (encoded.hasRemaining()) { results.write(encoded); }
    }

    private void disable(String reason, Minecraft client) {
        try {
            finishWalk(client, System.nanoTime(), "WALK_ABORTED", reason);
            writeResult(client, null, "DISABLED", reason, List.of());
        }
        catch (IOException | RuntimeException ignored) { LOGGER.warn("Diagnostic client action result unavailable"); }
        finally { close(); }
    }

    boolean isClosed() { return closed; }

    @Override public void close() {
        if (closed) { return; }
        closed = true;
        if (walk != null) {
            try { finishWalk(walk.client, System.nanoTime(), "WALK_ABORTED", "diagnostic_closed"); }
            catch (IOException | RuntimeException failure) {
                LOGGER.warn("Diagnostic walking completion could not be recorded", failure);
            }
        }
        try { results.close(); }
        catch (IOException failure) { LOGGER.warn("Diagnostic client action log close failed", failure); }
    }

    record Request(int seq, String action, UUID entityId, long gameTick, long validThroughTick) {
        boolean resumingMenu() { return action.equals("resume_game_menu"); }
        boolean aimed() { return action.equals("aim_and_brush") || action.equals("aim_and_attack"); }
        boolean brush() { return action.equals("brush") || action.equals("aim_and_brush"); }
        boolean approaching() { return action.equals("walk_approach_target"); }
        boolean tracking() { return action.equals("walk_track_target") || approaching(); }
        boolean walking() { return action.equals("walk_forward") || tracking(); }

        static Request parse(String line) throws ProtocolException {
            var fields = StrictJson.parseObject(line, 512);
            if (!fields.keySet().equals(Set.of("seq", "action", "entity_uuid", "game_tick", "valid_through_tick"))) {
                throw new IllegalArgumentException("unexpected client action keys");
            }
            long seq = integer(fields.get("seq"));
            String action = fields.get("action") instanceof String text ? text : "";
            if (seq >= COMMAND_LIMIT || !(action.equals("brush") || action.equals("attack")
                    || action.equals("walk_forward") || action.equals("aim_and_brush")
                    || action.equals("aim_and_attack") || action.equals("walk_track_target")
                    || action.equals("walk_approach_target") || action.equals("resume_game_menu"))) {
                throw new IllegalArgumentException("unsupported client action or sequence");
            }
            if (!(fields.get("entity_uuid") instanceof String text)) {
                throw new IllegalArgumentException("entity_uuid must be a UUID string");
            }
            UUID entity = UUID.fromString(text);
            if (!entity.toString().equals(text)) { throw new IllegalArgumentException("noncanonical UUID"); }
            long tick = integer(fields.get("game_tick")), through = integer(fields.get("valid_through_tick"));
            if (through <= tick || through - tick > 100) { throw new IllegalArgumentException("invalid tick window"); }
            return new Request((int) seq, action, entity, tick, through);
        }

        private static long integer(Object value) {
            if (!(value instanceof BigDecimal number)) { throw new IllegalArgumentException("integer required"); }
            try {
                long result = number.longValueExact();
                if (result < 0) { throw new IllegalArgumentException("negative integer"); }
                return result;
            } catch (ArithmeticException invalid) { throw new IllegalArgumentException("integer out of range", invalid); }
        }
    }
}
