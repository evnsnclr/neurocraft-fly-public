package dev.neurocraft.fly.gametest;

import com.mojang.authlib.GameProfile;
import dev.neurocraft.fly.NeuroCraftFly;
import dev.neurocraft.fly.body.BodyProgram;
import dev.neurocraft.fly.body.BodyProgramContext;
import dev.neurocraft.fly.body.BodyProgramContextSampler;
import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.BodyProgramState;
import dev.neurocraft.fly.body.BodyProgramStep;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ProgramParameters;
import dev.neurocraft.fly.body.programs.ApproachProgram;
import dev.neurocraft.fly.body.programs.BankProgram;
import dev.neurocraft.fly.body.programs.FleeProgram;
import dev.neurocraft.fly.body.programs.HoverProgram;
import dev.neurocraft.fly.control.RawPrimitiveAction;
import dev.neurocraft.fly.entity.FlyEntity;
import dev.neurocraft.fly.entity.FlySensors;
import dev.neurocraft.fly.entity.PrimitiveActuator;
import io.netty.channel.embedded.EmbeddedChannel;
import java.util.Optional;
import java.util.UUID;
import net.fabricmc.fabric.api.gametest.v1.GameTest;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.gametest.framework.GameTestHelper;
import net.minecraft.network.Connection;
import net.minecraft.network.chat.Component;
import net.minecraft.network.protocol.PacketFlow;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.server.network.CommonListenerCookie;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.level.GameType;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.phys.Vec3;

/**
 * Real server ticks, real player entities and real rays. Motion is driven through the primitive
 * actuator, as in the CRUISE flight pins, without acquiring program authority or involving a
 * neural service. Minecraft Direction supplies the independent cardinal reference. These are
 * scripted engineering tests; no assertion concerns neural program selection or biology.
 */
public final class TargetFlightProgramGameTests {
    private static final Direction[] CARDINALS = {Direction.SOUTH, Direction.WEST, Direction.NORTH, Direction.EAST};

    @GameTest(structure = FloorBandFence.TALL_ARENA, maxTicks = 20)
    public void playerContextHasCardinalBearingsUuidTiesAndNoGroundDependency(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(4, 9.39, 4));
        ServerPlayer first = player(helper);
        ServerPlayer second = player(helper);
        try {
            for (Direction direction : CARDINALS) {
                fly.setYRot(yaw(direction));
                Vec3 centre = fly.getBoundingBox().getCenter();
                moveCentre(first, centre.add(vector(direction.getClockWise()).scale(2)));
                moveCentre(second, centre.add(vector(direction).scale(3)));
                var context = context(helper, fly);
                check(helper, context.surfaceBelow().isEmpty(), "the fly must have no floor within six blocks");
                check(helper, context.target().isPresent(), "lack of ground suppressed the target");
                near(helper, 90, context.target().orElseThrow().bearingDegrees(), 0.0001, "right bearing " + direction);
                near(helper, 2, context.target().orElseThrow().distanceBlocks(), 0.0001, "nearest distance");
                // Equal distance, opposite sides. Expected tie order does not depend on list insertion.
                moveCentre(second, centre.add(vector(direction.getCounterClockWise()).scale(2)));
                double expected = first.getUUID().compareTo(second.getUUID()) < 0 ? 90 : -90;
                near(helper, expected, context(helper, fly).target().orElseThrow().bearingDegrees(),
                        0.0001, "UUID tie " + direction);
            }
        } finally {
            remove(helper, first);
            remove(helper, second);
        }
        helper.succeed();
    }

    @GameTest(structure = FloorBandFence.TALL_ARENA, maxTicks = 20)
    public void targetRangeLosSpectatorDeathAndRemovalAreAppliedToRealPlayers(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(3.5, 9.39, 3.5));
        ServerPlayer player = player(helper);
        try {
            Vec3 centre = fly.getBoundingBox().getCenter();
            moveCentre(player, centre.add(0, 12, 0));
            check(helper, context(helper, fly).target().isPresent(), "inclusive 12-block target boundary");
            moveCentre(player, centre.add(0, 12.001, 0));
            check(helper, context(helper, fly).target().isEmpty(), "out-of-range player remained a target");
            moveCentre(player, centre.add(3, 0, 0));
            check(helper, context(helper, fly).target().isPresent(), "clear LOS was rejected");
            helper.setBlock(new BlockPos(5, 9, 3), Blocks.STONE);
            check(helper, context(helper, fly).target().isEmpty(), "collider wall did not occlude player");
            helper.setBlock(new BlockPos(5, 9, 3), Blocks.AIR);
            player.setGameMode(GameType.SPECTATOR);
            check(helper, player.isSpectator() && context(helper, fly).target().isEmpty(), "spectator target accepted");
            player.setGameMode(GameType.CREATIVE);
            player.setHealth(0);
            check(helper, !player.isAlive() && context(helper, fly).target().isEmpty(), "dead target accepted");
            player.setHealth(20);
            check(helper, context(helper, fly).target().isPresent(), "restored live target missing");
            remove(helper, player);
            check(helper, context(helper, fly).target().isEmpty(), "removed target retained");
        } finally {
            if (!player.isRemoved()) { remove(helper, player); }
        }
        helper.succeed();
    }

    @GameTest(maxTicks = 200)
    public void bankMovesToItsDeclaredSideAtEveryCardinalHeading(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(4, CruiseFlightFence.IN_BAND_CLEARANCE_BLOCKS - 0.11, 4));
        BankProgram program = new BankProgram();
        int[] tick = {0};
        int[] trial = {0};
        Vec3[] origin = {Vec3.ZERO};
        BodyProgramState[] state = {BodyProgramState.entering(program.id(), 0)};
        helper.onEachTick(() -> {
            if (tick[0] == 16) {
                Direction facing = CARDINALS[trial[0] % 4];
                double sign = trial[0] < 4 ? 1 : -1;
                Vec3 delta = fly.position().subtract(origin[0]);
                check(helper, delta.dot(vector(facing.getClockWise())) * sign > 0.25,
                        "BANK did not move to declared side: " + facing + " sign=" + sign + " delta=" + delta);
                check(helper, delta.dot(vector(facing)) > 0.45, "BANK lacked a forward arc: " + delta);
                near(helper, origin[0].y, fly.getY(), 0.02, "BANK added vertical lift");
                trial[0]++;
                tick[0] = 0;
                if (trial[0] == 8) { helper.succeed(); return; }
            }
            if (tick[0] == 0) {
                reset(helper, fly, new Vec3(4, CruiseFlightFence.IN_BAND_CLEARANCE_BLOCKS - 0.11, 4), yaw(CARDINALS[trial[0] % 4]));
                origin[0] = fly.position();
                state[0] = BodyProgramState.entering(program.id(), helper.getLevel().getGameTime());
            }
            var step = program.step(state[0], context(helper, fly), Modulation.NEUTRAL,
                    new ProgramParameters(trial[0] < 4 ? 0 : -1));
            state[0] = step.next();
            PrimitiveActuator.apply(fly, RawPrimitiveAction.ZERO, step.motors());
            tick[0]++;
        });
    }

    @GameTest(maxTicks = 650)
    public void approachTurnsTowardThenStopsNearPlayerAtEveryCardinalHeading(GameTestHelper helper) {
        trajectory(helper, new ApproachProgram(), 140);
    }

    @GameTest(maxTicks = 420)
    public void fleeTurnsAwayBurstsAndExpiresAtEveryCardinalHeading(GameTestHelper helper) {
        trajectory(helper, new FleeProgram(), 90);
    }

    @GameTest(maxTicks = 20)
    public void removedTargetExitsOnThatTickAndVisibleObstacleBrakes(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(4, 2.89, 4));
        ServerPlayer player = player(helper);
        moveCentre(player, fly.getBoundingBox().getCenter().add(3, 0, 2));
        fly.setYRot(0);
        helper.setBlock(new BlockPos(4, 2, 5), Blocks.STONE);
        helper.setBlock(new BlockPos(4, 3, 5), Blocks.STONE);
        var obstructed = context(helper, fly);
        check(helper, obstructed.target().isPresent(), "diagonal player should remain visible past side of block");
        check(helper, obstructed.snapshot().rayForwardBlocks() < 1.5, "test failed to put obstacle in front ray");
        var approach = new ApproachProgram().step(BodyProgramState.entering(BodyProgramId.APPROACH, 0),
                obstructed, Modulation.NEUTRAL, ProgramParameters.NEUTRAL);
        check(helper, approach.motors().forwardThrust() == 0 && approach.motors().yawTorque() < 0,
                "visible obstacle did not brake and steer left toward clearance");
        remove(helper, player);
        fly.setDeltaMovement(0, 0, 0.1);
        for (BodyProgram program : new BodyProgram[] {new FleeProgram(), new ApproachProgram()}) {
            var result = program.step(new BodyProgramState(program.id(), 0, 8, 1), context(helper, fly),
                    Modulation.NEUTRAL, ProgramParameters.NEUTRAL);
            check(helper, result.exitTo().equals(Optional.of(BodyProgramId.HOVER)), "target loss did not request HOVER");
            check(helper, result.motors().forwardThrust() < 0 && result.motors().yawTorque() == 0,
                    "target loss did not brake on the same tick");
        }
        helper.succeed();
    }

    @GameTest(structure = FloorBandFence.TALL_ARENA, maxTicks = 650)
    public void repeatedProgramSelectionsGiveBackAltitudeWithoutAnUpwardRatchet(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(4, 11.89, 4));
        ServerPlayer player = player(helper);
        moveCentre(player, helper.absoluteVec(new Vec3(6.5, 3, 6.5)));
        BodyProgram[] programs = {new BankProgram(), new FleeProgram(), new ApproachProgram()};
        BodyProgram[] running = {programs[0]};
        BodyProgramState[] state = {BodyProgramState.entering(running[0].id(), 0)};
        int[] tick = {0};
        double originY = helper.absoluteVec(Vec3.ZERO).y;
        helper.onEachTick(() -> {
            if (tick[0] == 600) {
                double down = FlySensors.sample(helper.getLevel(), fly).rayDownBlocks();
                check(helper, FloorBandFence.settledInBand(down),
                        "repeated scripts did not return to the declared floor band: down=" + down);
                remove(helper, player);
                helper.succeed();
                return;
            }
            check(helper, fly.getY() - originY <= 12.1, "repeated entries added an upward ratchet");
            if (tick[0] % 40 == 0) {
                running[0] = programs[(tick[0] / 40) % programs.length];
                state[0] = BodyProgramState.entering(running[0].id(), helper.getLevel().getGameTime());
            }
            BodyProgramStep result = running[0].step(state[0], context(helper, fly), Modulation.NEUTRAL, ProgramParameters.NEUTRAL);
            PrimitiveActuator.apply(fly, RawPrimitiveAction.ZERO, result.motors());
            state[0] = result.next();
            if (result.exitTo().isPresent()) {
                running[0] = new HoverProgram();
                state[0] = BodyProgramState.entering(BodyProgramId.HOVER, helper.getLevel().getGameTime());
            }
            tick[0]++;
        });
    }

    private static void trajectory(GameTestHelper helper, BodyProgram program, int duration) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(4, 2.89, 4));
        ServerPlayer player = player(helper);
        BodyProgram[] running = {program};
        BodyProgramState[] state = {BodyProgramState.entering(program.id(), 0)};
        int[] tick = {0};
        int[] trial = {0};
        boolean[] exited = {false};
        double[] minDistance = {100};
        double[] maxDistance = {0};
        boolean approach = program.id() == BodyProgramId.APPROACH;
        helper.onEachTick(() -> {
            if (tick[0] == duration) {
                check(helper, exited[0], "program did not exit: " + program.id() + " " + CARDINALS[trial[0]]);
                double distance = fly.position().subtract(player.position()).horizontalDistance();
                if (approach) {
                    check(helper, distance >= 2.2 && distance <= 2.7 && minDistance[0] >= 2.2,
                            "APPROACH stand-off violation " + CARDINALS[trial[0]] + " end=" + distance + " min=" + minDistance[0]);
                } else {
                    check(helper, maxDistance[0] >= 2.75, "FLEE did not increase distance " + CARDINALS[trial[0]]
                            + " max=" + maxDistance[0]);
                }
                trial[0]++;
                tick[0] = 0;
                if (trial[0] == 4) { remove(helper, player); helper.succeed(); return; }
            }
            if (tick[0] == 0) {
                Direction facing = CARDINALS[trial[0]];
                Vec3 centre = new Vec3(4, 3, 4);
                Vec3 start = approach ? centre.subtract(vector(facing).scale(2)) : centre;
                reset(helper, fly, start.add(0, -0.11, 0), yaw(facing) - (approach ? 90 : 0));
                moveCentre(player, helper.absoluteVec(centre.add(vector(facing).scale(2))));
                running[0] = program;
                state[0] = BodyProgramState.entering(program.id(), helper.getLevel().getGameTime());
                exited[0] = false;
                minDistance[0] = 100;
                maxDistance[0] = 0;
            }
            double distance = fly.position().subtract(player.position()).horizontalDistance();
            minDistance[0] = Math.min(minDistance[0], distance);
            maxDistance[0] = Math.max(maxDistance[0], distance);
            var result = running[0].step(state[0], context(helper, fly), Modulation.NEUTRAL, ProgramParameters.NEUTRAL);
            PrimitiveActuator.apply(fly, RawPrimitiveAction.ZERO, result.motors());
            state[0] = result.next();
            if (result.exitTo().isPresent()) {
                exited[0] = true;
                running[0] = new HoverProgram();
                state[0] = BodyProgramState.entering(BodyProgramId.HOVER, helper.getLevel().getGameTime());
            }
            tick[0]++;
        });
    }

    private static BodyProgramContext context(GameTestHelper helper, FlyEntity fly) {
        return BodyProgramContextSampler.sample(helper.getLevel(), fly, FlySensors.sample(helper.getLevel(), fly));
    }

    private static ServerPlayer player(GameTestHelper helper) {
        // Use an ordinary ServerPlayer with Minecraft's embedded-channel login fixture.
        // GameTestHelper's deprecated subclass hard-codes CREATIVE and cannot exercise spectators.
        var profile = new GameProfile(UUID.randomUUID(), "body-target-test");
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

    private static void moveCentre(ServerPlayer player, Vec3 centre) {
        player.setPos(centre.x, centre.y - player.getBoundingBox().getYsize() / 2, centre.z);
        player.setDeltaMovement(Vec3.ZERO);
    }

    private static void remove(GameTestHelper helper, ServerPlayer player) {
        helper.getLevel().removePlayerImmediately(player, Entity.RemovalReason.DISCARDED);
    }

    private static void reset(GameTestHelper helper, FlyEntity fly, Vec3 relative, float yaw) {
        fly.setPos(helper.absoluteVec(relative));
        fly.setDeltaMovement(Vec3.ZERO);
        fly.setYRot(yaw);
        fly.setXRot(0);
    }

    private static Vec3 vector(Direction direction) {
        return new Vec3(direction.getStepX(), 0, direction.getStepZ());
    }

    private static float yaw(Direction direction) {
        return switch (direction) {
            case SOUTH -> 0;
            case WEST -> 90;
            case NORTH -> 180;
            case EAST -> -90;
            default -> throw new IllegalArgumentException("horizontal direction required");
        };
    }

    private static void near(GameTestHelper helper, double expected, double actual, double tolerance, String label) {
        check(helper, Math.abs(expected - actual) <= tolerance, label + " expected=" + expected + " actual=" + actual);
    }

    private static void check(GameTestHelper helper, boolean condition, String message) {
        helper.assertTrue(condition, Component.literal(message));
    }
}
