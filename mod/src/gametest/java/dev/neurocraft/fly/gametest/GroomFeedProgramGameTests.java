package dev.neurocraft.fly.gametest;

import dev.neurocraft.fly.NeuroCraftFly;
import dev.neurocraft.fly.body.BodyProgramContext;
import dev.neurocraft.fly.body.BodyProgramContextSampler;
import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.BodyProgramState;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ProgramParameters;
import dev.neurocraft.fly.body.programs.FeedProgram;
import dev.neurocraft.fly.body.programs.GroomProgram;
import dev.neurocraft.fly.body.programs.LandProgram;
import dev.neurocraft.fly.control.PrimitiveAction;
import dev.neurocraft.fly.control.RawPrimitiveAction;
import dev.neurocraft.fly.entity.FlyEntity;
import dev.neurocraft.fly.entity.FlySensors;
import dev.neurocraft.fly.entity.PrimitiveActuator;
import java.util.Optional;
import net.fabricmc.fabric.api.gametest.v1.GameTest;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.gametest.framework.GameTestHelper;
import net.minecraft.network.chat.Component;
import net.minecraft.world.entity.item.ItemEntity;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.phys.Vec3;

/** Real block contact, item entities and passive physics; no neural food selection or consumption. */
public final class GroomFeedProgramGameTests {
    private static final Direction[] CARDINALS = {Direction.SOUTH, Direction.WEST, Direction.NORTH, Direction.EAST};

    @GameTest(structure = FloorBandFence.TALL_ARENA, maxTicks = 20)
    public void foodWhitelistRangeRemovalAndAirborneContextAreExplicit(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(4, 9, 4));
        Vec3 centre = fly.getBoundingBox().getCenter();
        ItemEntity item = food(helper, Items.SUGAR, centre.add(0, 0, 0.5));
        for (Item allowed : new Item[] {Items.SUGAR, Items.APPLE, Items.SWEET_BERRIES, Items.GLOW_BERRIES}) {
            item.setItem(new ItemStack(allowed, 3));
            var context = context(helper, fly);
            check(helper, context.surfaceBelow().isEmpty(), "airborne fixture must have no floor within search range");
            check(helper, context.food().isPresent(), "a declared food item was not sampled: " + allowed);
            check(helper, !new FeedProgram().canEnter(context), "airborne FEED was admitted");
            check(helper, item.getItem().getCount() == 3, "sampling consumed the item");
        }
        item.setItem(new ItemStack(Items.STONE));
        check(helper, context(helper, fly).food().isEmpty(), "non-whitelisted stone was sampled as food");
        item.setItem(new ItemStack(Items.SUGAR));
        moveCentre(item, centre.add(0, 0, 0.75));
        check(helper, context(helper, fly).food().isPresent(), "inclusive 0.75-block range refused");
        moveCentre(item, centre.add(0, 0, 0.751));
        check(helper, context(helper, fly).food().isEmpty(), "out-of-range food was retained");
        moveCentre(item, centre.add(0, 0, 0.5));
        item.setItem(ItemStack.EMPTY);
        check(helper, context(helper, fly).food().isEmpty(), "empty item stack became food");
        item.setItem(new ItemStack(Items.SUGAR));
        item.discard();
        check(helper, context(helper, fly).food().isEmpty(), "removed food was remembered");
        helper.succeed();
    }

    @GameTest(maxTicks = 20)
    public void foodBearingAndNearestUuidTiesUseTheMinecraftCardinalBasis(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(4, 3, 4));
        Vec3 centre = fly.getBoundingBox().getCenter();
        ItemEntity first = food(helper, Items.SUGAR, centre);
        ItemEntity second = food(helper, Items.APPLE, centre);
        for (Direction facing : CARDINALS) {
            fly.setYRot(yaw(facing));
            moveCentre(first, centre.add(vector(facing.getClockWise()).scale(0.5)));
            moveCentre(second, centre.add(vector(facing).scale(0.7)));
            var target = context(helper, fly).food().orElseThrow();
            near(helper, 90, target.bearingDegrees(), 0.0001, "body-right food bearing " + facing);
            near(helper, 0.5, target.distanceBlocks(), 0.0001, "nearest food distance");
            moveCentre(second, centre.add(vector(facing.getCounterClockWise()).scale(0.5)));
            String expectedId = first.getUUID().compareTo(second.getUUID()) < 0 ? "minecraft:sugar" : "minecraft:apple";
            check(helper, context(helper, fly).food().orElseThrow().itemId().equals(expectedId),
                    "equal-distance item UUID ordering changed with list order");
        }
        helper.succeed();
    }

    @GameTest(maxTicks = 20)
    public void collidingBlockShapeOccludesFoodWithoutChangingTheSensorSnapshot(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(4.25, 3.39, 4.5));
        food(helper, Items.SUGAR, fly.getBoundingBox().getCenter().add(0.5, 0, 0));
        check(helper, context(helper, fly).food().isPresent(), "clear food target was missing");
        // Thin iron bars separate two free spaces less than the declared reach apart.
        helper.setBlock(new BlockPos(4, 3, 4), Blocks.IRON_BARS);
        var snapshot = FlySensors.sample(helper.getLevel(), fly);
        var blocked = BodyProgramContextSampler.sample(helper.getLevel(), fly, snapshot);
        check(helper, blocked.snapshot() == snapshot, "food lookup replaced the neural sensory snapshot");
        check(helper, blocked.food().isEmpty(), "iron-bar collider did not block close food LOS");
        helper.succeed();
    }

    @GameTest(maxTicks = 100)
    public void feedHoldsActualGroundAcrossAllPhasesAndLeavesTheItemUntouched(GameTestHelper helper) {
        platform(helper, 3, 3);
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(4.5, 2, 4.5));
        fly.setYRot(0);
        ItemEntity item = food(helper, Items.SUGAR, fly.getBoundingBox().getCenter().add(0, 0, 0.5));
        item.setItem(new ItemStack(Items.SUGAR, 3));
        Vec3 origin = fly.position();
        FeedProgram program = new FeedProgram();
        BodyProgramState[] state = {BodyProgramState.entering(program.id(), 0)};
        boolean[] phases = new boolean[4];
        int[] tick = {0};
        helper.onEachTick(() -> {
            if (tick[0]++ == 0) {
                fly.setDeltaMovement(0.04, 0, 0.03);
                contact(fly);
                return;
            }
            var context = context(helper, fly);
            check(helper, LandProgram.hasGroundContact(context), "FEED lost real full-footprint ground contact");
            check(helper, program.canEnter(context), "grounded food within reach was refused");
            var result = program.step(state[0], context, Modulation.NEUTRAL, ProgramParameters.NEUTRAL);
            check(helper, result.motors().verticalThrust() == -0.02, "FEED added a bob or lost contact force");
            check(helper, result.motors().yawTorque() == 0 && result.motors().pitchTorque() == 0,
                    "FEED turned the body to fake a gesture");
            phases[Math.min(3, (int) (result.presentationPhase() * 4))] = true;
            PrimitiveActuator.apply(fly, RawPrimitiveAction.ZERO, result.motors());
            state[0] = result.next();
            if (result.exitTo().isPresent()) {
                check(helper, result.exitTo().equals(Optional.of(BodyProgramId.HOVER)), "FEED named an unexpected successor");
                for (boolean visited : phases) { check(helper, visited, "FEED skipped a presentation quarter"); }
                check(helper, fly.position().distanceTo(origin) < 0.25, "FEED failed to brake initial drift");
                check(helper, fly.getDeltaMovement().horizontalDistance() < 0.001, "FEED kept drifting after settling");
                check(helper, !item.isRemoved() && item.getItem().is(Items.SUGAR) && item.getItem().getCount() == 3,
                        "FEED consumed or altered world food");
                helper.succeed();
            }
        });
    }

    @GameTest(maxTicks = 30)
    public void groundedFeedRejectsMissingFoodAndExitsImmediatelyWhenFoodOrSupportDisappears(GameTestHelper helper) {
        platform(helper, 3, 3);
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(4.5, 2, 4.5));
        FeedProgram feed = new FeedProgram();
        contact(fly);
        helper.runAfterDelay(1, () -> {
            var grounded = context(helper, fly);
            check(helper, LandProgram.hasGroundContact(grounded), "missing-food test did not establish real ground");
            check(helper, grounded.food().isEmpty() && !feed.canEnter(grounded), "ground alone admitted FEED");
            ItemEntity item = food(helper, Items.APPLE, fly.getBoundingBox().getCenter().add(0, 0, 0.5));
            var valid = context(helper, fly);
            check(helper, feed.canEnter(valid), "test food did not make FEED admissible");
            item.discard();
            var missing = feed.step(new BodyProgramState(BodyProgramId.FEED, 0, 25, 2),
                    context(helper, fly), Modulation.NEUTRAL, ProgramParameters.NEUTRAL);
            check(helper, missing.exitTo().equals(Optional.of(BodyProgramId.HOVER)), "food removal did not exit immediately");
            food(helper, Items.APPLE, fly.getBoundingBox().getCenter().add(0, 0, 0.5));
            helper.setBlock(new BlockPos(4, 1, 4), Blocks.AIR);
            var unsupported = context(helper, fly);
            check(helper, !LandProgram.hasGroundContact(unsupported), "support-removal fixture retained a complete floor");
            var lostFloor = feed.step(new BodyProgramState(BodyProgramId.FEED, 0, 25, 2),
                    unsupported, Modulation.NEUTRAL, ProgramParameters.NEUTRAL);
            check(helper, lostFloor.exitTo().equals(Optional.of(BodyProgramId.HOVER)), "support loss did not exit immediately");
            check(helper, lostFloor.motors().verticalThrust() == 0, "lost support forced the fly downward");
            helper.succeed();
        });
    }

    @GameTest(maxTicks = 100)
    public void groomingBrakesAirAndGroundDriftWithoutAHeightRatchet(GameTestHelper helper) {
        platform(helper, 4, 4);
        FlyEntity air = helper.spawn(NeuroCraftFly.FLY, new Vec3(2, 4, 2));
        FlyEntity ground = helper.spawn(NeuroCraftFly.FLY, new Vec3(5.5, 2, 5.5));
        air.setYRot(71);
        ground.setYRot(-111);
        air.setDeltaMovement(0.06, 0.04, -0.05);
        ground.setDeltaMovement(0.06, 0, -0.05);
        Vec3[] origins = {air.position(), ground.position()};
        FlyEntity[] flies = {air, ground};
        GroomProgram program = new GroomProgram();
        BodyProgramState[] states = {BodyProgramState.entering(program.id(), 0), BodyProgramState.entering(program.id(), 0)};
        int[] tick = {0};
        helper.onEachTick(() -> {
            if (tick[0]++ == 0) { contact(ground); return; }
            for (int index = 0; index < flies.length; index++) {
                FlyEntity fly = flies[index];
                var context = context(helper, fly);
                check(helper, index == 0 || LandProgram.hasGroundContact(context), "ground grooming lost support");
                var result = program.step(states[index], context, Modulation.NEUTRAL, ProgramParameters.NEUTRAL);
                PrimitiveActuator.apply(fly, RawPrimitiveAction.ZERO, result.motors());
                states[index] = result.next();
                check(helper, fly.position().distanceTo(origins[index]) < 0.4, "grooming failed to brake inherited drift");
                if (result.exitTo().isPresent()) {
                    check(helper, fly.getDeltaMovement().length() < 0.001, "grooming continued drifting after settling");
                    check(helper, fly.getY() <= origins[index].y + 0.1, "grooming introduced an upward ratchet");
                }
            }
            if (states[0].ticksInProgram() > 60) { helper.succeed(); }
        });
    }

    private static void platform(GameTestHelper helper, int x, int z) {
        for (int dx = 0; dx < 3; dx++) {
            for (int dz = 0; dz < 3; dz++) { helper.setBlock(new BlockPos(x + dx, 1, z + dz), Blocks.STONE); }
        }
    }

    private static void contact(FlyEntity fly) {
        PrimitiveActuator.apply(fly, RawPrimitiveAction.ZERO, PrimitiveAction.clamped(0, 0, -0.02, 0, 0));
    }

    private static ItemEntity food(GameTestHelper helper, Item item, Vec3 absoluteCentre) {
        ItemEntity entity = helper.spawnItem(item, new Vec3(4, 3, 4));
        entity.setNoGravity(true);
        moveCentre(entity, absoluteCentre);
        return entity;
    }

    private static void moveCentre(ItemEntity item, Vec3 centre) {
        item.setPos(centre.x, centre.y - item.getBoundingBox().getYsize() / 2, centre.z);
        item.setDeltaMovement(Vec3.ZERO);
    }

    private static BodyProgramContext context(GameTestHelper helper, FlyEntity fly) {
        return BodyProgramContextSampler.sample(helper.getLevel(), fly, FlySensors.sample(helper.getLevel(), fly));
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

    private static void near(GameTestHelper helper, double expected, double actual, double tolerance, String message) {
        check(helper, Math.abs(expected - actual) <= tolerance, message + " expected=" + expected + " actual=" + actual);
    }

    private static void check(GameTestHelper helper, boolean condition, String message) {
        helper.assertTrue(condition, Component.literal(message));
    }
}
