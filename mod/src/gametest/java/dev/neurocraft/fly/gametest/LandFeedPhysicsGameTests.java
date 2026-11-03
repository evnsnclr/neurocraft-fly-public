package dev.neurocraft.fly.gametest;

import dev.neurocraft.fly.NeuroCraftFly;
import dev.neurocraft.fly.body.BodyProgramContext;
import dev.neurocraft.fly.body.BodyProgramContextSampler;
import dev.neurocraft.fly.body.BodyProgramState;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ProgramParameters;
import dev.neurocraft.fly.body.programs.FeedProgram;
import dev.neurocraft.fly.body.programs.LandProgram;
import dev.neurocraft.fly.control.RawPrimitiveAction;
import dev.neurocraft.fly.control.PrimitiveAction;
import dev.neurocraft.fly.entity.FlyEntity;
import dev.neurocraft.fly.entity.FlySensors;
import dev.neurocraft.fly.entity.PrimitiveActuator;
import net.fabricmc.fabric.api.gametest.v1.GameTest;
import net.minecraft.core.BlockPos;
import net.minecraft.gametest.framework.GameTestHelper;
import net.minecraft.network.chat.Component;
import net.minecraft.world.entity.item.ItemEntity;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.phys.Vec3;

/**
 * The candidate's declared LAND/FEED commands against real body physics and world food.
 * Selection is explicit test input; this does not claim a neural event, command admission or
 * autonomous LAND-to-FEED transition. No ground flag, contact sample or descent is injected.
 */
public final class LandFeedPhysicsGameTests {
    // Exact candidate modulation, deliberately independent of Modulation.NEUTRAL's turn rate.
    private static final Modulation CANDIDATE = new Modulation(1.0, 0.0, 1.0, 0.0);
    private static final ProgramParameters PARAMETERS = new ProgramParameters(0.0);
    private static final int LAND_DEADLINE_TICKS = 200;
    private static final int FEED_CONTACT_TICKS = 5;

    @GameTest(maxTicks = 220)
    public void feedingRemainsEligibleAtTargetTickAfterTwoNeutralTicks(GameTestHelper helper) {
        feedAfterNeutralGap(helper, 2);
    }

    @GameTest(maxTicks = 220)
    public void feedingRemainsEligibleAfterEightStalledTicks(GameTestHelper helper) {
        feedAfterNeutralGap(helper, 8);
    }

    @GameTest(maxTicks = 20)
    public void restingSupportDoesNotInventContactAboveAnEdgeOrDuringLift(GameTestHelper helper) {
        for (int x = 3; x <= 5; x++) {
            for (int z = 3; z <= 5; z++) {
                helper.setBlock(new BlockPos(x, 1, z), Blocks.STONE);
            }
        }
        FlyEntity hovering = helper.spawn(NeuroCraftFly.FLY, new Vec3(3.5, 2.001, 4.5));
        check(helper, !LandProgram.hasGroundContact(context(helper, hovering)),
                "a positive air gap must not count as resting support");
        FlyEntity edge = helper.spawn(NeuroCraftFly.FLY, new Vec3(5.98, 2, 4.5));
        var partial = context(helper, edge);
        check(helper, !partial.surfaceBelow().orElseThrow().sturdy(), "edge fixture needs a partial footprint");
        check(helper, !LandProgram.hasGroundContact(partial), "partial footprint became resting support");
        FlyEntity rising = helper.spawn(NeuroCraftFly.FLY, new Vec3(4.5, 2, 4.5));
        PrimitiveActuator.apply(rising, RawPrimitiveAction.ZERO, PrimitiveAction.clamped(0, 0, 0.02, 0, 0));
        check(helper, !LandProgram.hasGroundContact(context(helper, rising)),
                "an upward commanded body must not be retained as resting support");
        helper.succeed();
    }

    private static void feedAfterNeutralGap(GameTestHelper helper, int gapTicks) {
        for (int x = 3; x <= 5; x++) {
            for (int z = 3; z <= 5; z++) {
                helper.setBlock(new BlockPos(x, 1, z), Blocks.STONE);
            }
        }
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(4.5, 3, 4.5));
        fly.setYRot(0);
        ItemEntity sugar = helper.spawnItem(Items.SUGAR, new Vec3(4.5, 2.5, 5));
        sugar.setDeltaMovement(Vec3.ZERO);
        double floorY = helper.absolutePos(new BlockPos(4, 2, 4)).getY();
        LandProgram land = new LandProgram();
        FeedProgram feed = new FeedProgram();
        BodyProgramState[] landState = {BodyProgramState.entering(land.id(), helper.getLevel().getGameTime())};
        BodyProgramState[] feedState = {null};
        long[] observedContactTick = {-1};
        int[] feedTicks = {0};
        helper.onEachTick(() -> {
            BodyProgramContext context = context(helper, fly);
            long tick = helper.getLevel().getGameTime();
            if (observedContactTick[0] < 0) {
                // Require a collision-produced touchdown before withholding all subsequent
                // actuation. Neither position, velocity nor Minecraft contact flags are reset.
                if (!context.snapshot().onGround() || !context.snapshot().verticalCollision()) {
                    var step = land.step(landState[0], context, CANDIDATE, PARAMETERS);
                    check(helper, step.exitTo().isEmpty(), "LAND exited before the gap fixture touched down");
                    landState[0] = step.next();
                    PrimitiveActuator.apply(fly, RawPrimitiveAction.ZERO, step.motors());
                    return;
                }
                check(helper, feed.canEnter(context), "observed real touchdown must initially permit FEED");
                observedContactTick[0] = tick;
            }
            check(helper, Math.abs(fly.getY() - floorY) < 1.0e-6, "neutral gap moved the fly off its floor");
            if (tick - observedContactTick[0] < gapTicks) {
                // This is the late/no-command path: no body program gets an actuator write.
                return;
            }
            if (feedState[0] == null) {
                check(helper, !context.snapshot().onGround() && !context.snapshot().verticalCollision(),
                        "gap must exercise Minecraft's lost movement-contact flags");
                check(helper, context.snapshot().velocityYBlocksPerTick() == 0,
                        "fixture must rest physically at zero vertical velocity");
                check(helper, context.surfaceBelow().orElseThrow().sturdy()
                                && context.surfaceBelow().orElseThrow().distanceBlocks() < 1.0e-6,
                        "actual full-footprint floor support disappeared during the gap");
                check(helper, context.food().isPresent() && sugar.onGround(),
                        "dropped food must remain physically eligible after the gap");
                check(helper, LandProgram.hasGroundContact(context) && feed.canEnter(context),
                        "real supported FEED contact was lost across " + gapTicks + " neutral ticks");
                // A fresh command's application-time predicate, not retrospective acceptance
                // of a late command. Two ticks model the existing observation-to-target gap.
                feedState[0] = BodyProgramState.entering(feed.id(), tick);
            }
            check(helper, feed.canEnter(context), "FEED lost support after resuming valid actuation");
            var step = feed.step(feedState[0], context, CANDIDATE, PARAMETERS);
            check(helper, step.exitTo().isEmpty(), "FEED refused a supported resumed tick");
            PrimitiveActuator.apply(fly, RawPrimitiveAction.ZERO, step.motors());
            feedState[0] = step.next();
            if (++feedTicks[0] == FEED_CONTACT_TICKS) { helper.succeed(); }
        });
    }

    @GameTest(maxTicks = 220)
    public void candidateLandDescendsFromRestAndCanFeedAfterPhysicalTouchdown(GameTestHelper helper) {
        for (int x = 3; x <= 5; x++) {
            for (int z = 3; z <= 5; z++) {
                helper.setBlock(new BlockPos(x, 1, z), Blocks.STONE);
            }
        }
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(4.5, 5.5, 4.5));
        fly.setYRot(0.0F);
        double initialY = fly.getY();
        double floorY = helper.absolutePos(new BlockPos(4, 2, 4)).getY();
        long enteredTick = helper.getLevel().getGameTime();
        check(helper, fly.getDeltaMovement().lengthSqr() == 0.0, "LAND must start stationary");

        // A real dropped stack settles under vanilla item gravity. Only its initial random
        // throw velocity is removed; neither the fly nor the item is repositioned thereafter.
        ItemEntity sugar = helper.spawnItem(Items.SUGAR, new Vec3(4.5, 2.5, 5.0));
        sugar.setItem(new ItemStack(Items.SUGAR, 3));
        sugar.setDeltaMovement(Vec3.ZERO);
        check(helper, !sugar.isNoGravity(), "food fixture must retain real item gravity");

        LandProgram land = new LandProgram();
        FeedProgram feed = new FeedProgram();
        BodyProgramState[] landState = {BodyProgramState.entering(land.id(), enteredTick)};
        BodyProgramState[] feedState = {null};
        int[] drivenLandTicks = {0};
        int[] drivenFeedTicks = {0};
        helper.onEachTick(() -> {
            BodyProgramContext context = context(helper, fly);
            if (feedState[0] == null) {
                check(helper, land.canEnter(context), "LAND lost a real sturdy floor within its probe range");
                check(helper, helper.getLevel().getGameTime() - enteredTick < LAND_DEADLINE_TICKS,
                        "candidate did not reach supported ground before its 200-tick deadline");
                var step = land.step(landState[0], context, CANDIDATE, PARAMETERS);
                check(helper, step.exitTo().isEmpty(), "LAND exited before physical touchdown");
                check(helper, Math.abs(step.motors().verticalThrust()) <= 0.35,
                        "LAND exceeded its declared vertical thrust bound");
                if (drivenLandTicks[0] == 0) {
                    check(helper, !context.snapshot().onGround(), "fixture must begin airborne");
                    check(helper, !feed.canEnter(context), "FEED must refuse the airborne fixture");
                    check(helper, Math.abs(context.surfaceBelow().orElseThrow().distanceBlocks() - 3.5) < 1.0e-6,
                            "LAND entry clearance must be measured from feet to physical floor");
                    // A zero-speed candidate formerly requested zero descent in the no-gravity
                    // body. Pin the required positive-speed outcome, not that faulty outcome.
                    check(helper, step.motors().verticalThrust() < 0.0,
                            "candidate LAND must command descent from stationary airborne rest");
                }
                if (drivenLandTicks[0] >= 5) {
                    check(helper, fly.getY() < initialY - 0.05,
                            "downward LAND command did not produce real vertical movement");
                }
                landState[0] = step.next();
                drivenLandTicks[0]++;
                if (!LandProgram.hasGroundContact(context)) {
                    PrimitiveActuator.apply(fly, RawPrimitiveAction.ZERO, step.motors());
                    return;
                }
                check(helper, landState[0].phaseIndex() == LandProgram.GROUNDED_HOLD,
                        "LAND must enter its distinct supported hold at touchdown");
                check(helper, Math.abs(fly.getY() - floorY) < 1.0e-6,
                        "touchdown did not land on the physical floor surface");
                check(helper, initialY - fly.getY() > 3.0, "fixture did not execute the airborne descent");
                check(helper, sugar.onGround(), "dropped sugar must have settled on the floor");
                check(helper, context.food().isPresent()
                                && context.food().orElseThrow().itemId().equals("minecraft:sugar"),
                        "actual dropped sugar was not sampled within grounded feeding reach");
                // Explicit new selection after observed contact; LAND itself never selects FEED.
                feedState[0] = BodyProgramState.entering(feed.id(), helper.getLevel().getGameTime());
            }

            check(helper, LandProgram.hasGroundContact(context), "FEED lost actual supported ground contact");
            check(helper, feed.canEnter(context), "candidate FEED refused actual reachable dropped food");
            var step = feed.step(feedState[0], context, CANDIDATE, PARAMETERS);
            check(helper, step.exitTo().isEmpty(), "FEED immediately exited its eligible grounded state");
            check(helper, step.motors().verticalThrust() < 0.0
                            && Math.abs(step.motors().verticalThrust()) <= 0.35,
                    "FEED must retain bounded contact pressure in the no-gravity body");
            PrimitiveActuator.apply(fly, RawPrimitiveAction.ZERO, step.motors());
            feedState[0] = step.next();
            if (++drivenFeedTicks[0] == FEED_CONTACT_TICKS) {
                check(helper, !sugar.isRemoved() && sugar.getItem().is(Items.SUGAR)
                                && sugar.getItem().getCount() == 3,
                        "scripted feeding must not consume or alter world food");
                helper.succeed();
            }
        });
    }

    private static BodyProgramContext context(GameTestHelper helper, FlyEntity fly) {
        return BodyProgramContextSampler.sample(helper.getLevel(), fly, FlySensors.sample(helper.getLevel(), fly));
    }

    private static void check(GameTestHelper helper, boolean condition, String message) {
        helper.assertTrue(condition, Component.literal(message));
    }
}
