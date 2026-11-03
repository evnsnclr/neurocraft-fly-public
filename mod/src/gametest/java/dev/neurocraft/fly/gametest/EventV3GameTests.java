package dev.neurocraft.fly.gametest;

import com.mojang.authlib.GameProfile;
import dev.neurocraft.fly.NeuroCraftFly;
import dev.neurocraft.fly.body.BodyProgramContextSampler;
import dev.neurocraft.fly.body.programs.FeedProgram;
import dev.neurocraft.fly.control.ControlProtocol;
import dev.neurocraft.fly.control.ControllerIdentity;
import dev.neurocraft.fly.control.EventSample;
import dev.neurocraft.fly.control.PrimitiveAction;
import dev.neurocraft.fly.control.RawPrimitiveAction;
import dev.neurocraft.fly.entity.FlyEntity;
import dev.neurocraft.fly.entity.FlySensors;
import dev.neurocraft.fly.entity.FoodEventSampler;
import dev.neurocraft.fly.entity.PrimitiveActuator;
import io.netty.channel.embedded.EmbeddedChannel;
import java.util.UUID;
import net.fabricmc.fabric.api.gametest.v1.GameTest;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.gametest.framework.GameTestHelper;
import net.minecraft.network.Connection;
import net.minecraft.network.chat.Component;
import net.minecraft.network.protocol.PacketFlow;
import net.minecraft.nbt.CompoundTag;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.server.network.CommonListenerCookie;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.InteractionResult;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EntityType;
import net.minecraft.world.entity.item.ItemEntity;
import net.minecraft.world.entity.projectile.Arrow;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.level.GameType;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.phys.Vec3;

/** Real authoritative hooks and sampled food edges. No health, item, neural or motor effect is inferred. */
public final class EventV3GameTests {
    @GameTest(maxTicks = 20)
    public void eventDisplayRuntimeStartsOnlyForV3AndV4AndKeepsAppliedMatching(GameTestHelper helper) {
        EventDisplayRuntimeFence.verify(helper);
        helper.succeed();
    }

    @GameTest(maxTicks = 20)
    public void nearFoodHasIndependentThreeBlockRangeCardinalConeNearestTieAndOcclusionRules(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(4.5, 3.39, 4.5));
        ItemEntity item = food(helper, fly);
        for (Direction facing : new Direction[] {Direction.SOUTH, Direction.WEST, Direction.NORTH, Direction.EAST}) {
            fly.setYRot(switch (facing) { case SOUTH -> 0; case WEST -> 90; case NORTH -> 180; default -> -90; });
            Vec3 centre = fly.getBoundingBox().getCenter();
            Vec3 forward = new Vec3(facing.getStepX(), 0, facing.getStepZ());
            moveCentre(item, centre.add(forward.scale(3)));
            check(helper, FoodEventSampler.foodNear(helper.getLevel(), fly), "inclusive 3-block near range " + facing);
            check(helper, BodyProgramContextSampler.sample(helper.getLevel(), fly,
                    FlySensors.sample(helper.getLevel(), fly)).food().isEmpty(), "near sensing widened body food range");
            moveCentre(item, centre.add(forward.scale(3.001)));
            check(helper, !FoodEventSampler.foodNear(helper.getLevel(), fly), "food beyond 3 blocks admitted");
            moveCentre(item, centre.subtract(forward));
            check(helper, !FoodEventSampler.foodNear(helper.getLevel(), fly), "food behind fly admitted");
        }
        fly.setYRot(0);
        Vec3 centre = fly.getBoundingBox().getCenter();
        moveCentre(item, centre.add(0, 0, 2));
        ItemEntity second = food(helper, fly);
        moveCentre(second, centre.add(0, 0, -1));
        check(helper, !FoodEventSampler.foodNear(helper.getLevel(), fly), "nearer off-cone item did not mask farther food");
        moveCentre(second, centre.add(0, 0, -2));
        check(helper, FoodEventSampler.foodNear(helper.getLevel(), fly) == (item.getUUID().compareTo(second.getUUID()) < 0),
                "exact-distance near-food tie ignored UUID order");
        second.discard();
        helper.setBlock(new BlockPos(4, 3, 5), Blocks.STONE);
        check(helper, !FoodEventSampler.foodNear(helper.getLevel(), fly), "collider failed to occlude near food");
        helper.setBlock(new BlockPos(4, 3, 5), Blocks.AIR);
        item.setItem(new ItemStack(Items.STONE));
        check(helper, !FoodEventSampler.foodNear(helper.getLevel(), fly), "non-whitelisted item became nearby food");
        item.setItem(new ItemStack(Items.SUGAR, 3));
        fly.tick();
        check(helper, fly.eventSample().foodNear() && !fly.eventSample().foodContact()
                        && fly.eventSample().food().count() == 1,
                "airborne near food did not reach the event sample independently of contact");
        // Actual entity NBT preserves the already-seen near edge even while not in contact.
        CompoundTag saved = fly.saveWithoutId(new CompoundTag());
        FlyEntity loaded = new FlyEntity(NeuroCraftFly.FLY, helper.getLevel());
        loaded.load(saved);
        loaded.tick();
        check(helper, loaded.eventSample().foodNear() && loaded.eventSample().food().count() == 1,
                "same-tick save/load replayed the near-food edge");
        helper.succeed();
    }

    @GameTest(maxTicks = 20)
    public void savedEntityReloadRetainsCountersAndMalformedEventNbtStaysUnavailable(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(4, 3, 4));
        ServerPlayer player = player(helper);
        try {
            player.setItemInHand(InteractionHand.MAIN_HAND, new ItemStack(Items.BRUSH));
            fly.interact(player, InteractionHand.MAIN_HAND);
            fly.hurtServer(helper.getLevel(), helper.getLevel().damageSources().playerAttack(player), 1);
            fly.tick();
            EventSample original = fly.eventSample();
            CompoundTag saved = fly.saveWithoutId(new CompoundTag());
            fly.discard();
            FlyEntity reloaded = new FlyEntity(NeuroCraftFly.FLY, helper.getLevel());
            reloaded.load(saved);
            check(helper, reloaded.getUUID().equals(fly.getUUID()) && original.equals(reloaded.eventSample()),
                    "same-UUID save/load lost cumulative events");
            helper.getLevel().addFreshEntity(reloaded);
            reloaded.interact(player, InteractionHand.MAIN_HAND);
            check(helper, reloaded.eventSample().brush().count() == 2, "reload restarted the brush counter");
            CompoundTag corrupt = reloaded.saveWithoutId(new CompoundTag());
            corrupt.getCompoundOrEmpty("neurocraft_events").remove("food");
            reloaded.load(corrupt);
            check(helper, reloaded.eventSample() == null, "partial supplied NBT silently became zero counters");
            check(helper, reloaded.interact(player, InteractionHand.MAIN_HAND) == InteractionResult.PASS,
                    "invalid event state accepted a new brush count");
            reloaded.tick();
            check(helper, FlySensors.sample(helper.getLevel(), reloaded).events() == null,
                    "unavailable event data was fabricated into the sensor sample");
            var old = ReadoutLaneFence.identity(reloaded);
            var v3 = new ControllerIdentity(old.runId(), old.sessionEpoch(), old.entityId(), old.graphHash(),
                    old.adapterHash(), old.dynamicsHash(), old.decoderHash(), ControlProtocol.OBSERVATION_SCHEMA_V3_SHA256);
            boolean refused = false;
            try { ControlProtocol.encodeObservation(v3, FlySensors.sample(helper.getLevel(), reloaded), 0, 0, 0); }
            catch (IllegalArgumentException expected) { refused = true; }
            check(helper, refused, "V3 accepted unavailable required events");
            ControlProtocol.encodeObservation(old, FlySensors.sample(helper.getLevel(), reloaded), 0, 0, 0);
            CompoundTag invalidSavedAgain = reloaded.saveWithoutId(new CompoundTag());
            reloaded.load(invalidSavedAgain);
            check(helper, reloaded.eventSample() == null, "save/reload repaired corrupt events by inventing zeros");
            CompoundTag legacy = saved.copy(); legacy.remove("neurocraft_events");
            FlyEntity legacyLoaded = new FlyEntity(NeuroCraftFly.FLY, helper.getLevel());
            legacyLoaded.load(legacy);
            check(helper, EventSample.EMPTY.equals(legacyLoaded.eventSample()), "missing legacy data did not initialize zero");
        } finally { remove(helper, player); }
        helper.succeed();
    }

    @GameTest(maxTicks = 20)
    public void realPlayerBrushAndAttackHooksCountWithoutChangingTheBodyOrItem(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(4, 3, 4));
        ServerPlayer player = player(helper);
        try {
            player.setPos(fly.getX(), fly.getY(), fly.getZ() + 1);
            ItemStack brush = new ItemStack(Items.BRUSH);
            player.setItemInHand(InteractionHand.MAIN_HAND, brush);
            fly.setDeltaMovement(0.03, -0.01, 0.02);
            Vec3 position = fly.position(), velocity = fly.getDeltaMovement();
            check(helper, fly.isPickable(), "fly cannot be targeted by real interaction");
            check(helper, player.interactOn(fly, InteractionHand.MAIN_HAND) == InteractionResult.SUCCESS_SERVER,
                    "real player interaction did not enter the server brush hook");
            player.attack(fly);
            EventSample events = fly.eventSample();
            check(helper, events.brush().count() == 1 && events.attackAttempt().count() == 1,
                    "real brush/attack paths did not each count once: " + events);
            check(helper, events.brush().lastGameTick() == helper.getLevel().getGameTime()
                            && events.attackAttempt().lastGameTick() == helper.getLevel().getGameTime(),
                    "interaction timestamps were not the current world tick");
            check(helper, position.equals(fly.position()) && velocity.equals(fly.getDeltaMovement()),
                    "event hooks changed fly position or velocity");
            check(helper, !fly.isRemoved() && brush.getCount() == 1 && brush.getDamageValue() == 0,
                    "events consumed/damaged the brush or removed the fly");
            fly.resetNeuralSessionPreservingBody();
            check(helper, events.equals(fly.eventSample()), "neural epoch reset erased world events");
            fly.resetForFixture(fly.getX(), fly.getY(), fly.getZ(), 0);
            check(helper, events.equals(fly.eventSample()), "fixture reset erased world counters");
        } finally { remove(helper, player); }
        helper.succeed();
    }

    @GameTest(maxTicks = 20)
    public void brushRequiresTheInvokedBrushHandAndLiveNonSpectatorPlayerOutsideFlyFluid(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(4, 3, 4));
        ServerPlayer player = player(helper);
        try {
            player.setItemInHand(InteractionHand.MAIN_HAND, new ItemStack(Items.STICK));
            player.setItemInHand(InteractionHand.OFF_HAND, new ItemStack(Items.BRUSH));
            check(helper, fly.interact(player, InteractionHand.MAIN_HAND) == InteractionResult.PASS,
                    "a brush in the other hand created an event");
            check(helper, fly.interact(player, InteractionHand.OFF_HAND) == InteractionResult.SUCCESS_SERVER,
                    "off-hand brush was ignored");
            player.setGameMode(GameType.SPECTATOR);
            fly.interact(player, InteractionHand.OFF_HAND);
            player.setGameMode(GameType.CREATIVE);
            player.setHealth(0);
            fly.interact(player, InteractionHand.OFF_HAND);
            player.setHealth(20);
            helper.setBlock(new BlockPos(4, 3, 4), Blocks.WATER);
            fly.tick();
            check(helper, fly.isInWater(), "water fixture did not immerse fly");
            fly.interact(player, InteractionHand.OFF_HAND);
            check(helper, fly.eventSample().brush().count() == 1,
                    "spectator, dead player or fluid brush interaction created an event");
        } finally { remove(helper, player); }
        helper.succeed();
    }

    @GameTest(maxTicks = 20)
    public void onlyFinitePositiveLivePlayerOriginHurtRequestsCountIncludingOwnedProjectiles(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(4, 3, 4));
        ServerPlayer player = player(helper);
        try {
            var level = helper.getLevel();
            var source = level.damageSources().playerAttack(player);
            for (float amount : new float[] {0, -1, Float.NaN, Float.POSITIVE_INFINITY}) {
                check(helper, !fly.hurtServer(level, source, amount), "fly accepted health damage");
            }
            fly.hurtServer(level, level.damageSources().onFire(), 1);
            var cow = helper.spawn(EntityType.COW, new Vec3(2, 3, 2));
            fly.hurtServer(level, level.damageSources().mobAttack(cow), 1);
            player.setGameMode(GameType.SPECTATOR);
            fly.hurtServer(level, source, 1);
            player.setGameMode(GameType.CREATIVE);
            player.setHealth(0);
            fly.hurtServer(level, source, 1);
            player.setHealth(20);
            check(helper, fly.eventSample().attackAttempt().count() == 0,
                    "invalid amount, environmental/NPC origin or inactive player created an attack event");
            check(helper, !fly.hurtServer(level, source, 1), "positive attempt became health damage");
            Arrow arrow = new Arrow(EntityType.ARROW, level);
            arrow.setOwner(player);
            check(helper, !fly.hurtServer(level, level.damageSources().arrow(arrow, player), 1),
                    "owned projectile became health damage");
            check(helper, fly.eventSample().attackAttempt().count() == 2, "player-origin attempts were lost");
        } finally { remove(helper, player); }
        helper.succeed();
    }

    @GameTest(maxTicks = 30)
    public void foodContactUsesActualFeedPreconditionAndCountsRemovalThenReturnAsOneNewEdge(GameTestHelper helper) {
        for (int x = 3; x <= 5; x++) {
            for (int z = 3; z <= 5; z++) { helper.setBlock(new BlockPos(x, 1, z), Blocks.STONE); }
        }
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(4.5, 2, 4.5));
        fly.setYRot(0);
        ItemEntity[] food = {contactFood(helper, fly)};
        Vec3[] foodPosition = {food[0].position()};
        int[] tick = {0};
        helper.onEachTick(() -> {
            int index = tick[0]++;
            if (!food[0].isRemoved()) {
                assertStationaryContactFood(helper, fly, food[0], foodPosition[0]);
            }
            PrimitiveActuator.apply(fly, RawPrimitiveAction.ZERO, PrimitiveAction.clamped(0, 0, -0.02, 0, 0));
            if (index == 3) {
                var context = BodyProgramContextSampler.sample(helper.getLevel(), fly,
                        FlySensors.sample(helper.getLevel(), fly));
                check(helper, new FeedProgram().canEnter(context) && fly.eventSample().foodContact(),
                        "expected eligible FEED and sampled contact: " + foodContactDiagnostic(helper, fly, food[0]));
                check(helper, fly.eventSample().food().count() == 1, "persistent contact was recounted");
                food[0].discard();
            } else if (index == 5) {
                check(helper, !fly.eventSample().foodContact() && fly.eventSample().food().count() == 1,
                        "food removal did not end contact or erased its counter");
                food[0] = contactFood(helper, fly);
                foodPosition[0] = food[0].position();
            } else if (index == 8) {
                check(helper, fly.eventSample().foodContact() && fly.eventSample().food().count() == 2,
                        "food return did not create exactly one new edge: "
                                + foodContactDiagnostic(helper, fly, food[0]));
                check(helper, food[0].getItem().getCount() == 3, "food sensing consumed an item");
                helper.succeed();
            }
        });
    }

    @GameTest(maxTicks = 20)
    public void measuredEventCountersTravelThroughTheRealTickAndRecordingBridge(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(4, 3, 4));
        ServerPlayer player = player(helper);
        try {
            player.setItemInHand(InteractionHand.MAIN_HAND, new ItemStack(Items.BRUSH));
            fly.interact(player, InteractionHand.MAIN_HAND);
            var old = ReadoutLaneFence.identity(fly);
            var identity = new ControllerIdentity(old.runId(), old.sessionEpoch(), old.entityId(),
                    old.graphHash(), old.adapterHash(), old.dynamicsHash(), old.decoderHash(),
                    ControlProtocol.OBSERVATION_SCHEMA_V3_SHA256);
            try (RecordingBridgeFence.Installed installed = RecordingBridgeFence.install(fly, identity)) {
                fly.tick();
                String frame = installed.bridge().observations().getLast();
                check(helper, frame.contains("\"sensor_schema\":\"neurocraft.sensors.v3\"")
                                && frame.contains("\"ray_hit_kinds\":") && frame.contains("\"events\":"),
                        "real tick lost v3 transport fields: " + frame);
                long tick = helper.getLevel().getGameTime();
                check(helper, frame.contains("\"brush\":{\"count\":1,\"last_game_tick\":" + tick + "}"),
                        "actual brush event did not reach the wire");
                check(helper, fly.sensorSnapshotAt(tick).orElseThrow().events().equals(fly.eventSample()),
                        "retained causal snapshot differs from the measured events");
            }
        } finally { remove(helper, player); }
        helper.succeed();
    }

    /**
     * Contact fixture with ordinary item collision physics. Matching the 0.22-high fly's centre
     * placed a 0.25-high item's feet 0.015 inside the floor; NoGravity does not prevent vanilla
     * collision ejection. Keep this item's feet 0.02 above the actually sampled support instead.
     */
    private static ItemEntity contactFood(GameTestHelper helper, FlyEntity fly) {
        var snapshot = FlySensors.sample(helper.getLevel(), fly);
        var context = BodyProgramContextSampler.sample(helper.getLevel(), fly, snapshot);
        check(helper, context.surfaceBelow().isPresent() && context.surfaceBelow().orElseThrow().sturdy(),
                "contact food fixture has no complete actual support: " + context);
        double floorY = fly.getBoundingBox().minY - context.surfaceBelow().orElseThrow().distanceBlocks();
        ItemEntity item = food(helper, fly);
        item.setPos(item.getX(), floorY + 0.02, item.getZ());
        assertStationaryContactFood(helper, fly, item, item.position());
        return item;
    }

    /** Every observed present-item tick checks the fixture before checking event semantics. */
    private static void assertStationaryContactFood(
            GameTestHelper helper, FlyEntity fly, ItemEntity item, Vec3 expectedPosition) {
        var context = BodyProgramContextSampler.sample(helper.getLevel(), fly,
                FlySensors.sample(helper.getLevel(), fly));
        check(helper, context.surfaceBelow().isPresent(), "fixture support disappeared: " + context);
        double floorY = fly.getBoundingBox().minY - context.surfaceBelow().orElseThrow().distanceBlocks();
        check(helper, item.getBoundingBox().minY > floorY
                        && helper.getLevel().noCollision(item, item.getBoundingBox())
                        && item.position().equals(expectedPosition) && item.getDeltaMovement().equals(Vec3.ZERO),
                "contact food fixture embedded or moved; expected=" + expectedPosition
                        + " floorY=" + floorY + ": " + foodContactDiagnostic(helper, fly, item));
        check(helper, context.food().isPresent()
                        && context.food().orElseThrow().distanceBlocks() <= FeedProgram.REACH_BLOCKS
                        && Math.abs(context.food().orElseThrow().bearingDegrees()) <= FeedProgram.FORWARD_CONE_DEGREES,
                "contact fixture left actual FEED food reach: " + foodContactDiagnostic(helper, fly, item));
    }

    private static String foodContactDiagnostic(GameTestHelper helper, FlyEntity fly, ItemEntity item) {
        var snapshot = FlySensors.sample(helper.getLevel(), fly);
        var context = BodyProgramContextSampler.sample(helper.getLevel(), fly, snapshot);
        return "itemBox=" + item.getBoundingBox() + " itemVelocity=" + item.getDeltaMovement()
                + " itemPosition=" + item.position() + " flyPosition=" + fly.position()
                + " context=" + context + " canFeed=" + new FeedProgram().canEnter(context)
                + " sampledContact=" + fly.eventSample().foodContact();
    }

    private static ItemEntity food(GameTestHelper helper, FlyEntity fly) {
        ItemEntity item = helper.spawnItem(Items.SUGAR, new Vec3(4, 3, 4));
        item.setItem(new ItemStack(Items.SUGAR, 3));
        item.setNoGravity(true);
        Vec3 centre = fly.getBoundingBox().getCenter().add(0, 0, 0.5);
        item.setPos(centre.x, centre.y - item.getBoundingBox().getYsize() / 2, centre.z);
        item.setDeltaMovement(Vec3.ZERO);
        return item;
    }

    private static void moveCentre(ItemEntity item, Vec3 centre) {
        item.setPos(centre.x, centre.y - item.getBoundingBox().getYsize() / 2, centre.z);
        item.setDeltaMovement(Vec3.ZERO);
    }

    private static ServerPlayer player(GameTestHelper helper) {
        var profile = new GameProfile(UUID.randomUUID(), "event-source-test");
        var cookie = CommonListenerCookie.createInitial(profile, false);
        var level = helper.getLevel();
        ServerPlayer player = new ServerPlayer(level.getServer(), level, profile, cookie.clientInformation());
        Connection connection = new Connection(PacketFlow.SERVERBOUND);
        new EmbeddedChannel(connection);
        level.getServer().getPlayerList().placeNewPlayer(connection, player, cookie);
        player.setGameMode(GameType.CREATIVE);
        player.setNoGravity(true);
        player.setDeltaMovement(Vec3.ZERO);
        return player;
    }

    private static void remove(GameTestHelper helper, ServerPlayer player) {
        helper.getLevel().removePlayerImmediately(player, Entity.RemovalReason.DISCARDED);
    }

    private static void check(GameTestHelper helper, boolean condition, String message) {
        helper.assertTrue(condition, Component.literal(message));
    }
}
