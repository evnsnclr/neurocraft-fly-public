package dev.neurocraft.fly.gametest;

import com.mojang.authlib.GameProfile;
import dev.neurocraft.fly.NeuroCraftFly;
import dev.neurocraft.fly.control.ControlProtocol;
import dev.neurocraft.fly.control.ControllerIdentity;
import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import dev.neurocraft.fly.entity.FlyEntity;
import dev.neurocraft.fly.entity.FlySensors;
import io.netty.channel.embedded.EmbeddedChannel;
import java.util.List;
import java.util.UUID;
import net.fabricmc.fabric.api.gametest.v1.GameTest;
import net.minecraft.core.BlockPos;
import net.minecraft.gametest.framework.GameTestHelper;
import net.minecraft.network.Connection;
import net.minecraft.network.chat.Component;
import net.minecraft.network.protocol.PacketFlow;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.server.network.CommonListenerCookie;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EntityType;
import net.minecraft.world.entity.animal.Cow;
import net.minecraft.world.entity.decoration.ArmorStand;
import net.minecraft.world.entity.projectile.ProjectileUtil;
import net.minecraft.world.level.ClipContext;
import net.minecraft.world.level.GameType;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.HitResult;
import net.minecraft.world.phys.Vec3;

/** Actor classes are world metadata; these tests establish no downstream response or biological sense. */
public final class ActorSensorGameTests {
    @GameTest(maxTicks = 20, skyAccess = true)
    public void playerCowAndOtherPickableKeepTheSameMeasuredHitAndOcclusion(GameTestHelper helper) {
        FlyEntity fly = fly(helper);
        Vec3 origin = fly.getBoundingBox().getCenter();
        check(helper, FlySensors.sample(helper.getLevel(), fly).rayHitKinds().getFirst() == 0,
                "forward fixture is not initially clear");
        ServerPlayer player = player(helper, UUID.randomUUID());
        try {
            Cow cow = helper.spawn(EntityType.COW, new Vec3(4.5, 4, 7.5));
            cow.setNoAi(true);
            ArmorStand stand = helper.spawn(EntityType.ARMOR_STAND, new Vec3(4.5, 4, 7.5));
            stand.setNoGravity(true);
            List<Entity> actors = List.of(player, cow, stand);
            for (int index = 0; index < actors.size(); index++) {
                for (Entity actor : actors) { centre(actor, origin.add(3, 0, 0)); }
                Entity target = actors.get(index);
                centre(target, origin.add(0, 0, 3));
                PhysicalSensorSnapshot seen = FlySensors.sample(helper.getLevel(), fly);
                check(helper, seen.rayHitKinds().getFirst() == 2 && seen.rayActorKinds().getFirst() == index + 1,
                        "wrong actor class for " + target.getType() + ": " + seen.rayActorKinds());
                check(helper, Math.abs(seen.rayForwardBlocks() - (target.getBoundingBox().minZ - origin.z)) < 1e-7,
                        "actor classification changed uninflated near-face distance");
                checkVanillaDistances(helper, fly);
                helper.setBlock(new BlockPos(4, 4, 2), Blocks.STONE);
                PhysicalSensorSnapshot blocked = FlySensors.sample(helper.getLevel(), fly);
                check(helper, blocked.rayHitKinds().getFirst() == 1 && blocked.rayActorKinds().getFirst() == 0,
                        "actor metadata leaked through a nearer collider");
                checkVanillaDistances(helper, fly);
                helper.setBlock(new BlockPos(4, 4, 2), Blocks.AIR);
            }
            cow.discard(); stand.discard();
            centre(player, origin.add(0, 0, 3));
            player.setGameMode(GameType.SPECTATOR);
            check(helper, FlySensors.sample(helper.getLevel(), fly).rayActorKinds().getFirst() == 0,
                    "spectator became an actor hit");
        } finally { remove(helper, player); }
        helper.succeed();
    }

    @GameTest(maxTicks = 20, skyAccess = true)
    public void exactEntityTiesUseSignedUuidNaturalOrderIndependentOfInsertion(GameTestHelper helper) {
        FlyEntity fly = fly(helper);
        Vec3 origin = fly.getBoundingBox().getCenter();
        // Natural UUID ordering is signed, unlike textual UUID ordering, in both 64-bit halves.
        for (int half = 0; half < 2; half++) {
            UUID low = half == 0 ? new UUID(-1L, 0L) : new UUID(0L, -1L);
            UUID high = half == 0 ? new UUID(1L, 0L) : new UUID(0L, 1L);
            for (boolean playerFirst : List.of(false, true)) {
                Cow cow = new Cow(EntityType.COW, helper.getLevel());
                cow.setUUID(half == 0 ? high : low); cow.setNoAi(true); cow.setNoGravity(true);
                centre(cow, origin.add(0, 0, 3));
                if (!playerFirst) { helper.getLevel().addFreshEntity(cow); }
                ServerPlayer player = player(helper, half == 0 ? low : high);
                try {
                    if (playerFirst) { helper.getLevel().addFreshEntity(cow); }
                    centre(player, origin.add(0, 0, 3));
                    AABB identical = new AABB(origin.x - 0.25, origin.y - 0.25, origin.z + 2,
                            origin.x + 0.25, origin.y + 0.25, origin.z + 3);
                    player.setBoundingBox(identical); cow.setBoundingBox(identical);
                    PhysicalSensorSnapshot seen = FlySensors.sample(helper.getLevel(), fly);
                    check(helper, seen.rayForwardBlocks() == 2 && seen.rayActorKinds().getFirst() == (half == 0 ? 1 : 2),
                            "equal entity hits ignored natural UUID order or depended on insertion order");
                    checkVanillaDistances(helper, fly);
                } finally { cow.discard(); remove(helper, player); }
            }
        }
        helper.succeed();
    }

    @GameTest(maxTicks = 20, skyAccess = true)
    public void exactBlockTieAndEndpointInsideAndOffRayGeometryRetainVanillaDistances(GameTestHelper helper) {
        FlyEntity fly = fly(helper);
        Vec3 origin = fly.getBoundingBox().getCenter();
        Cow cow = helper.spawn(EntityType.COW, new Vec3(4.5, 4, 7));
        cow.setNoAi(true); cow.setNoGravity(true);
        helper.setBlock(new BlockPos(4, 4, 3), Blocks.STONE);
        double blockNearFace = helper.absolutePos(new BlockPos(4, 4, 3)).getZ();
        cow.setBoundingBox(new AABB(origin.x - 0.25, origin.y - 0.25, blockNearFace,
                origin.x + 0.25, origin.y + 0.25, blockNearFace + 1));
        PhysicalSensorSnapshot tie = FlySensors.sample(helper.getLevel(), fly);
        check(helper, tie.rayForwardBlocks() == blockNearFace - origin.z
                        && tie.rayHitKinds().getFirst() == 2 && tie.rayActorKinds().getFirst() == 2,
                "exact block/entity tie did not preserve entity actor");
        checkVanillaDistances(helper, fly);
        helper.setBlock(new BlockPos(4, 4, 3), Blocks.AIR);
        for (double near : new double[] {5.999, 6.0, 6.001, -0.25}) {
            centre(cow, origin.add(0, 0, near + 0.5));
            cow.setBoundingBox(new AABB(origin.x - 0.25, origin.y - 0.25, origin.z + near,
                    origin.x + 0.25, origin.y + 0.25, origin.z + near + 1));
            checkVanillaDistances(helper, fly);
            PhysicalSensorSnapshot seen = FlySensors.sample(helper.getLevel(), fly);
            check(helper, near == 5.999 ? seen.rayHitKinds().getFirst() == 2
                            : seen.rayHitKinds().getFirst() == 0,
                    "vanilla endpoint or origin-inside clipping semantics changed at " + near);
        }
        // This body misses the actual line by 0.01 blocks. A silent vanilla 0.3 inflation hits it.
        centre(cow, origin.add(0.5, 0, 3));
        cow.setBoundingBox(new AABB(origin.x + 0.01, origin.y - 0.25, origin.z + 2,
                origin.x + 0.51, origin.y + 0.25, origin.z + 3));
        check(helper, FlySensors.sample(helper.getLevel(), fly).rayHitKinds().getFirst() == 0,
                "off-ray entity was admitted through hitbox inflation");
        checkVanillaDistances(helper, fly);
        helper.succeed();
    }

    @GameTest(maxTicks = 20, skyAccess = true)
    public void sameTickActorMetadataTravelsThroughV4RecordingBridgeOnly(GameTestHelper helper) {
        FlyEntity fly = fly(helper);
        Cow cow = helper.spawn(EntityType.COW, new Vec3(4.5, 4, 7.5));
        cow.setNoAi(true); cow.setNoGravity(true);
        var expected = FlySensors.sample(helper.getLevel(), fly);
        check(helper, expected.rayActorKinds().getFirst() == 2, "cow is not on forward ray");
        var old = ReadoutLaneFence.identity(fly);
        var typed = new ControllerIdentity(old.runId(), old.sessionEpoch(), old.entityId(), old.graphHash(),
                old.adapterHash(), old.dynamicsHash(), old.decoderHash(), ControlProtocol.OBSERVATION_SCHEMA_V4_SHA256);
        try (RecordingBridgeFence.Installed installed = RecordingBridgeFence.install(fly, typed)) {
            fly.tick();
            var frames = installed.bridge().observations();
            check(helper, !frames.isEmpty(), "real tick emitted no observation");
            String frame = frames.getLast();
            check(helper, frame.contains("\"sensor_schema\":\"neurocraft.sensors.v4\""), "V4 schema lost");
            check(helper, frame.contains("\"ray_actor_kinds\":" + expected.rayActorKinds().toString().replace(" ", "")),
                    "actual actor classes did not reach the V4 bridge frame");
        }
        for (String hash : List.of(old.observationSchemaHash(), ControlProtocol.OBSERVATION_SCHEMA_V2_SHA256,
                ControlProtocol.OBSERVATION_SCHEMA_V3_SHA256)) {
            var identity = new ControllerIdentity(old.runId(), old.sessionEpoch(), old.entityId(), old.graphHash(),
                    old.adapterHash(), old.dynamicsHash(), old.decoderHash(), hash);
            check(helper, !ControlProtocol.encodeObservation(identity, expected, 0, 0, 0).contains("ray_actor_kinds"),
                    "V4 actor metadata leaked into an older profile");
        }
        helper.succeed();
    }

    /** Differential against the bundled old ray picker; explicit face assertions above are independent. */
    private static void checkVanillaDistances(GameTestHelper helper, FlyEntity fly) {
        PhysicalSensorSnapshot seen = FlySensors.sample(helper.getLevel(), fly);
        Vec3 origin = fly.getBoundingBox().getCenter();
        Vec3 forward = Vec3.directionFromRotation(fly.getXRot(), fly.getYRot()).normalize();
        Vec3 right = forward.cross(new Vec3(0, 1, 0)).normalize();
        Vec3 up = right.cross(forward).normalize();
        List<Vec3> directions = List.of(forward, forward.reverse(), right.reverse(), right, up, up.reverse());
        List<Double> distances = List.of(seen.rayForwardBlocks(), seen.rayBackBlocks(), seen.rayLeftBlocks(),
                seen.rayRightBlocks(), seen.rayUpBlocks(), seen.rayDownBlocks());
        for (int index = 0; index < 6; index++) {
            Vec3 end = origin.add(directions.get(index).scale(FlySensors.RAY_LENGTH_BLOCKS));
            var block = helper.getLevel().clip(new ClipContext(origin, end,
                    ClipContext.Block.COLLIDER, ClipContext.Fluid.NONE, fly));
            var entity = ProjectileUtil.getEntityHitResult(helper.getLevel(), fly, origin, end,
                    new AABB(origin, end), FlySensors::obstructsSensorRay, 0.0F);
            double distance = FlySensors.RAY_LENGTH_BLOCKS;
            int kind = 0;
            if (block.getType() != HitResult.Type.MISS && origin.distanceTo(block.getLocation()) <= distance) {
                distance = origin.distanceTo(block.getLocation()); kind = 1;
            }
            if (entity != null && origin.distanceTo(entity.getLocation()) <= distance) {
                distance = origin.distanceTo(entity.getLocation()); kind = 2;
            }
            check(helper, distance == distances.get(index) && kind == seen.rayHitKinds().get(index),
                    "legacy geometry changed in ray " + index + ": " + distances.get(index) + " vs " + distance);
            check(helper, (kind == 2) == (seen.rayActorKinds().get(index) != 0), "actor/kind mismatch");
        }
    }

    private static FlyEntity fly(GameTestHelper helper) {
        // The forward six-block segment stays inside the eight-block-wide test enclosure.
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(4.5, 4, 1.5));
        fly.setYRot(0); fly.setXRot(0);
        return fly;
    }

    private static void centre(Entity entity, Vec3 centre) {
        entity.setPos(centre.x, centre.y - entity.getBoundingBox().getYsize() / 2, centre.z);
        entity.setDeltaMovement(Vec3.ZERO);
    }

    private static ServerPlayer player(GameTestHelper helper, UUID uuid) {
        var profile = new GameProfile(uuid, "actor-source-test");
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
