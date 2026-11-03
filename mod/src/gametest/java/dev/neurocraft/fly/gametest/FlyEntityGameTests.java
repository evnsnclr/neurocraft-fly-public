package dev.neurocraft.fly.gametest;

import dev.neurocraft.fly.NeuroCraftFly;
import dev.neurocraft.fly.StagedProgramAuthority;
import dev.neurocraft.fly.StagedReset;
import dev.neurocraft.fly.body.BodyProgramChannel;
import dev.neurocraft.fly.body.BodyProgramContext;
import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.BodyProgramRuntime;
import dev.neurocraft.fly.body.BodyProgramState;
import dev.neurocraft.fly.body.BodyProgramStep;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ProgramParameters;
import dev.neurocraft.fly.body.MotorAuthority;
import dev.neurocraft.fly.body.SelectionSource;
import dev.neurocraft.fly.body.programs.HoverProgram;
import dev.neurocraft.fly.control.ActionFrame;
import dev.neurocraft.fly.control.CommandValidator;
import dev.neurocraft.fly.control.ControlAckFrame;
import dev.neurocraft.fly.control.ControlOperation;
import dev.neurocraft.fly.control.ControlProtocol;
import dev.neurocraft.fly.control.ControlState;
import dev.neurocraft.fly.control.ControllerFault;
import dev.neurocraft.fly.control.ControllerIdentity;
import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import dev.neurocraft.fly.control.PrimitiveAction;
import dev.neurocraft.fly.control.RawPrimitiveAction;
import dev.neurocraft.fly.control.ValidationResult;
import dev.neurocraft.fly.entity.ActuationReceipt;
import dev.neurocraft.fly.entity.BodyFrameMath;
import dev.neurocraft.fly.entity.FlyEntity;
import dev.neurocraft.fly.entity.FlySensors;
import dev.neurocraft.fly.entity.PrimitiveActuator;
import dev.neurocraft.fly.presentation.FlyTrailPresentation;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.LongFunction;
import net.fabricmc.fabric.api.gametest.v1.GameTest;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.gametest.framework.GameTestHelper;
import net.minecraft.network.chat.Component;
import net.minecraft.tags.EntityTypeTags;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EntityType;
import net.minecraft.world.entity.Mob;
import net.minecraft.world.entity.decoration.ArmorStand;
import net.minecraft.world.entity.item.ItemEntity;
import net.minecraft.world.entity.projectile.Arrow;
import net.minecraft.world.entity.projectile.windcharge.WindCharge;
import net.minecraft.world.item.Items;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.Vec3;

/** Dedicated-server checks for entity ownership, neutral failure, sensing, reset, and primitives. */
public final class FlyEntityGameTests {
    @GameTest(maxTicks = 20)
    public void acknowledgedOperatorAuthorityRetainsGroundAcrossLandWalkFeed(GameTestHelper helper) {
        OperatorProgramSwitchFence.groundChain(helper);
    }

    @GameTest(maxTicks = 20)
    public void localOperatorSelectionPreservesNeutralAdmissionAndPreconditions(GameTestHelper helper) {
        OperatorProgramSwitchFence.admissionAndPreconditions(helper);
    }

    @GameTest(maxTicks = 20)
    public void localOperatorSelectionExcludesTransactionsAndRechecksAuthority(GameTestHelper helper) {
        OperatorProgramSwitchFence.exclusionAndInvalidation(helper);
    }

    @GameTest(maxTicks = 20)
    public void operatorSelectionCannotReuseRejectedOrResumedPause(GameTestHelper helper) {
        OperatorProgramSwitchFence.pendingRpcAndRejectedPause(helper);
    }

    @GameTest(maxTicks = 20)
    public void neuralSessionResetPreservesMovingBodyAndClearsBothProgramAuthorities(GameTestHelper helper) {
        for (boolean readout : new boolean[] {false, true}) {
            FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(2.5, 4.0, 2.5));
            try (RecordingBridgeFence.Installed installed = RecordingBridgeFence.install(fly)) {
                fly.tick(); // Record the actual first observation before offering its command.
                long tick = helper.getLevel().getGameTime();
                if (readout) {
                    installed.bridge().offer(ReadoutLaneFence.streamedProgramFrame(fly, 0, 0,
                            tick, tick, BodyProgramId.CRUISE, ReadoutLaneFence.CRUISING,
                            ReadoutLaneFence.BIASED));
                } else {
                    fly.bodyPrograms().engageScripted(BodyProgramId.HOVER, Modulation.NEUTRAL,
                            SelectionSource.OPERATOR, tick);
                    installed.bridge().offer(new ActionFrame(ReadoutLaneFence.identity(fly),
                            0, 0, tick, tick, tick, 0, 0, "6".repeat(64),
                            RawPrimitiveAction.ZERO, true));
                }
                fly.tick();
                helper.assertTrue(fly.lastAcceptedCommand().isPresent()
                                && fly.sensorSnapshotAt(tick).isPresent(),
                        Component.literal("test must first publish a real command and sensor sample"));
                helper.assertTrue(fly.bodyPrograms().authority()
                                == (readout ? MotorAuthority.READOUT : MotorAuthority.SCRIPTED),
                        Component.literal("test did not establish the requested active program authority"));
                fly.setYRot(37.0F);
                fly.setXRot(-12.0F);
                fly.setDeltaMovement(new Vec3(0.11, -0.07, 0.19));
                Vec3 position = fly.position();
                Vec3 velocity = fly.getDeltaMovement();
                int age = fly.tickCount;
                fly.resetNeuralSessionPreservingBody();
                helper.assertTrue(fly.position().equals(position)
                                && fly.getDeltaMovement().equals(velocity)
                                && fly.getYRot() == 37.0F && fly.getXRot() == -12.0F
                                && fly.tickCount == age && helper.getLevel().getGameTime() == tick,
                        Component.literal("neural session reset changed pose, velocity, body age or world time"));
                helper.assertTrue(fly.bodyPrograms().authority() == MotorAuthority.NEURAL
                                && fly.bodyPrograms().selectionSource() == SelectionSource.DEFAULT
                                && fly.lastAcceptedCommand().isEmpty()
                                && fly.sensorSnapshotAt(tick).isEmpty()
                                && fly.readoutChannel().equals(dev.neurocraft.fly.body.ReadoutChannel.DEFAULT)
                                && fly.presentationAction().equals(PrimitiveAction.NEUTRAL),
                        Component.literal("neural session reset retained old authority or telemetry"));
                helper.assertTrue(fly.brainDisplayPresentation().frame().isEmpty(),
                        Component.literal("neural session reset retained a display frame"));
            }
            fly.discard();
        }
        helper.succeed();
    }

    @GameTest(maxTicks = 20)
    public void externalPacerEnforcesTwentyHertzAndNeverCatchesUp(GameTestHelper helper) {
        AtomicLong nowNanos = new AtomicLong();
        AtomicLong parkedNanos = new AtomicLong();
        ExternalGameTestTickPacer pacer = new ExternalGameTestTickPacer(
                nowNanos::get,
                duration -> {
                    parkedNanos.addAndGet(duration);
                    nowNanos.addAndGet(duration);
                });

        pacer.pace();
        nowNanos.addAndGet(75_000_000L);
        pacer.pace();
        pacer.pace();

        helper.assertTrue(
                pacer.pacedTickCount() == 3,
                Component.literal("external pacer did not record all paced ticks"));
        helper.assertTrue(
                pacer.minimumIntervalNanos() >= ExternalGameTestTickPacer.PERIOD_NANOS,
                Component.literal("external pacer released ticks faster than 20 Hz"));
        helper.assertTrue(
                parkedNanos.get() == 2 * ExternalGameTestTickPacer.PERIOD_NANOS,
                Component.literal("external pacer attempted catch-up after an overshoot"));
        helper.assertTrue(
                pacer.hasVerifiedCadence(),
                Component.literal("external pacer did not verify its measured cadence"));
        helper.succeed();
    }

    @GameTest(maxTicks = 20)
    public void controlEffectFenceRejectsInFlightActionAndLatchesReceiptTarget(
            GameTestHelper helper) {
        ControllerIdentity identity = new ControllerIdentity(
                "control-effect-fence",
                0,
                "fly-01",
                "1".repeat(64),
                "2".repeat(64),
                "3".repeat(64),
                "4".repeat(64),
                "5".repeat(64));
        ControlEffectFence fence = new ControlEffectFence(7, 100);
        ActionFrame staleSequenceAtEffectiveTick = action(identity, 7, 100);
        ActionFrame wrongTickAfterAcknowledgement = action(identity, 8, 101);
        ActionFrame firstQualified = action(identity, 9, 100);
        ActionFrame laterQualified = action(identity, 10, 100);

        helper.assertFalse(
                fence.tryLatch(staleSequenceAtEffectiveTick),
                Component.literal("control fence accepted an action from the ack-time snapshot"));
        helper.assertFalse(
                fence.tryLatch(wrongTickAfterAcknowledgement),
                Component.literal("control fence accepted an action outside the ACK effective tick"));
        helper.assertTrue(
                fence.tryLatch(firstQualified),
                Component.literal("control fence rejected its first ACK-governed action"));
        helper.assertFalse(
                fence.tryLatch(laterQualified),
                Component.literal("control fence chased a later action while awaiting receipt"));
        helper.assertTrue(
                fence.latchedCommandSequence() == 9,
                Component.literal("control fence changed its latched receipt target"));
        helper.assertFalse(
                fence.receiptSentThrough(8),
                Component.literal("control fence advanced before its latched receipt"));
        helper.assertTrue(
                fence.receiptSentThrough(9),
                Component.literal("control fence did not advance on its exact latched receipt"));
        helper.succeed();
    }

    @GameTest(maxTicks = 20)
    public void obstacleFenceRequiresFirstPostPlacementObservation(GameTestHelper helper) {
        ControllerIdentity identity = new ControllerIdentity(
                "obstacle-observation-fence",
                0,
                "fly-01",
                "1".repeat(64),
                "2".repeat(64),
                "3".repeat(64),
                "4".repeat(64),
                "5".repeat(64));
        ObstacleObservationFence fence = new ObstacleObservationFence(7, 100);

        helper.assertFalse(
                fence.tryLatch(action(identity, 8, 100)),
                Component.literal("obstacle fence accepted a pre-placement observation"));
        helper.assertFalse(
                fence.tryLatch(action(identity, 7, 101)),
                Component.literal("obstacle fence accepted the open command sequence"));
        helper.assertTrue(
                fence.tryLatch(action(identity, 9, 101)),
                Component.literal("obstacle fence rejected the next authoritative sample"));
        helper.assertFalse(
                fence.tryLatch(action(identity, 10, 102)),
                Component.literal("obstacle fence changed after its first qualified action"));
        helper.assertTrue(
                fence.blockedCommandSequence() == 9,
                Component.literal("obstacle fence changed its blocked command sequence"));
        helper.assertTrue(
                fence.blockedObservationTick() == 101,
                Component.literal("obstacle fence did not retain the exact observation tick"));
        helper.succeed();
    }

    @GameTest(maxTicks = 20)
    public void traceRecordCeilingIncludesEffectiveTickAndSealsAtFirstLaterTarget(
            GameTestHelper helper) {
        ControllerIdentity identity = new ControllerIdentity(
                "trace-record-ceiling",
                0,
                "fly-01",
                "1".repeat(64),
                "2".repeat(64),
                "3".repeat(64),
                "4".repeat(64),
                "5".repeat(64));
        TraceRecordCeiling ceiling = new TraceRecordCeiling(17);

        helper.assertFalse(
                ceiling.observe(actionWithTargetTick(identity, 6, 16)),
                Component.literal("trace ceiling sealed before the stop boundary"));
        helper.assertFalse(
                ceiling.observe(actionWithTargetTick(identity, 7, 17)),
                Component.literal("effective-tick action was incorrectly excluded"));
        helper.assertTrue(
                ceiling.observe(actionWithTargetTick(identity, 8, 18)),
                Component.literal("first later target did not seal the trace ceiling"));
        helper.assertFalse(
                ceiling.observe(actionWithTargetTick(identity, 9, 19)),
                Component.literal("trace ceiling moved after its first later target"));
        helper.assertTrue(
                ceiling.commandSequence() == 7,
                Component.literal("trace ceiling did not include the effective-tick action"));
        helper.assertTrue(
                ceiling.firstLaterCommandSequence() == 8,
                Component.literal("trace ceiling did not retain its first later witness"));
        helper.succeed();
    }

    @GameTest(maxTicks = 20)
    public void operatorCommandTreeExposesLifecycleAndNamedRegistryControls(
            GameTestHelper helper) {
        var root = helper.getLevel()
                .getServer()
                .getCommands()
                .getDispatcher()
                .getRoot()
                .getChild("neurocraftfly");
        helper.assertTrue(root != null, Component.literal("operator command root is missing"));
        for (String child : new String[] {
            "spawn",
            "pause",
            "resume",
            "single_step",
            "reset",
            "record_start",
            "record_stop",
            "replay_start",
            "replay_stop",
            "lesion_apply",
            "lesion_clear",
            "stimulation_apply",
            "stimulation_clear",
            "trail",
            "program",
            "status",
            "inspect"
        }) {
            helper.assertTrue(
                    root.getChild(child) != null,
                    Component.literal("operator command is missing: " + child));
        }
        var trail = root.getChild("trail");
        for (String child : new String[] {"on", "off", "interval"}) {
            helper.assertTrue(
                    trail.getChild(child) != null,
                    Component.literal("trail operator command is missing: " + child));
        }
        var program = root.getChild("program");
        for (String child : new String[] {"release", "status", "program"}) {
            helper.assertTrue(
                    program.getChild(child) != null,
                    Component.literal("body-program operator command is missing: " + child));
        }
        var scalars = program.getChild("program");
        for (String child : new String[] {"speed", "turn", "amplitude", "urgency"}) {
            helper.assertTrue(
                    scalars.getChild(child) != null,
                    Component.literal("modulation argument is missing: " + child));
            scalars = scalars.getChild(child);
        }
        helper.assertTrue(
                trail.getCommand() != null,
                Component.literal("bare trail operator command must be executable (toggle)"));
        // Lane L (2026-09-04): bare `program` answered "Unknown or incomplete command" and `status`
        // hid the HUD behind twenty chat lines. Bare `program` now executes (usage), `status`
        // executes (compact) and carries `full`; the chat text itself is pinned in
        // OperatorCommandClientGameTest, which can arrange the single fly the text reports.
        helper.assertTrue(
                program.getCommand() != null,
                Component.literal("bare program operator command must be executable (usage)"));
        var status = root.getChild("status");
        helper.assertTrue(
                status.getCommand() != null,
                Component.literal("bare status operator command must be executable (compact)"));
        helper.assertTrue(
                status.getChild("full") != null,
                Component.literal("operator command is missing: status full"));
        helper.succeed();
    }

    @GameTest(maxTicks = 20)
    public void projectOwnedEntityHasNoMobOrNavigationInheritance(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 2.0, 1.5));

        helper.assertTrue(
                fly.getClass().getSuperclass() == Entity.class,
                Component.literal("FlyEntity must extend Entity directly"));
        helper.assertFalse(
                Mob.class.isAssignableFrom(fly.getClass()),
                Component.literal("FlyEntity must not inherit Mob goals or navigation"));
        helper.succeed();
    }

    @GameTest(maxTicks = 20)
    public void missingBridgeIsNeutralAndHasNoHorizontalIntent(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 3.0, 1.5));
        Vec3 initial = fly.position();

        helper.runAfterDelay(3, () -> {
            helper.assertTrue(
                    fly.controllerState() == ControllerFault.MISSING_CONFIGURATION,
                    Component.literal("missing bridge configuration must fail closed"));
            helper.assertTrue(
                    fly.presentationAction().isNeutral(),
                    Component.literal("faulted entity must expose neutral applied action"));
            helper.assertTrue(
                    Math.abs(fly.getX() - initial.x) < 1.0e-9
                            && Math.abs(fly.getZ() - initial.z) < 1.0e-9,
                    Component.literal("neutral controller added horizontal motion"));
            helper.succeed();
        });
    }

    @GameTest(maxTicks = 20)
    public void trailIsDisabledByDefaultAndEmitsNothingWithoutOperatorOptIn(
            GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 3.0, 1.5));
        long emittedBefore = FlyTrailPresentation.emittedParticleCount();

        helper.assertFalse(
                FlyTrailPresentation.enabled(),
                Component.literal("cosmetic trail must be disabled by default"));
        helper.assertTrue(
                FlyTrailPresentation.intervalTicks()
                        == FlyTrailPresentation.DEFAULT_INTERVAL_TICKS,
                Component.literal("cosmetic trail default interval changed"));
        helper.runAfterDelay(10, () -> {
            helper.assertTrue(
                    FlyTrailPresentation.emittedParticleCount() == emittedBefore,
                    Component.literal("disabled trail emitted particles"));
            helper.succeed();
        });
    }

    @GameTest(maxTicks = 100)
    public void trailEmissionNeverPerturbsAuthoritativeEntityState(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 3.0, 1.5));
        Vec3 start = fly.position();
        int phaseTicks = 8;
        // The trail-on phase starts only after every default-off assertion in this
        // concurrently ticking batch has passed its own 20-tick deadline. The
        // trail toggle is process-global: any new test in this class that reads
        // FlyTrailPresentation.enabled() must finish before tick 40, or this
        // margin must grow with it.
        int trailPhaseStartTick = 40;
        long[] baseline = new long[phaseTicks * STATE_WORDS];
        long[] withTrail = new long[phaseTicks * STATE_WORDS];
        int[] tickCounter = {0};
        long[] emittedAtPhaseStart = {0};

        helper.onEachTick(() -> {
            int tick = tickCounter[0]++;
            if (tick == 0) {
                seedComparablePhase(fly, start);
                return;
            }
            if (tick <= phaseTicks) {
                recordAuthoritativeState(fly, baseline, tick - 1);
                return;
            }
            if (tick < trailPhaseStartTick) {
                return;
            }
            if (tick == trailPhaseStartTick) {
                seedComparablePhase(fly, start);
                FlyTrailPresentation.setIntervalTicks(
                        FlyTrailPresentation.MIN_INTERVAL_TICKS);
                FlyTrailPresentation.setEnabled(true);
                emittedAtPhaseStart[0] = FlyTrailPresentation.emittedParticleCount();
                return;
            }
            if (tick > trailPhaseStartTick + phaseTicks) {
                return;
            }
            recordAuthoritativeState(fly, withTrail, tick - trailPhaseStartTick - 1);
            if (tick < trailPhaseStartTick + phaseTicks) {
                return;
            }
            FlyTrailPresentation.setEnabled(false);
            FlyTrailPresentation.setIntervalTicks(FlyTrailPresentation.DEFAULT_INTERVAL_TICKS);
            helper.assertTrue(
                    FlyTrailPresentation.emittedParticleCount() > emittedAtPhaseStart[0],
                    Component.literal("trail-on comparison phase recorded no emissions"));
            helper.assertTrue(
                    Arrays.equals(baseline, withTrail),
                    Component.literal("trail emission perturbed authoritative entity state"));
            helper.succeed();
        });
    }

    @GameTest(maxTicks = 20)
    public void fixtureResetIsServerOwnedAndClearsVelocity(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 2.0, 1.5));
        Vec3 reset = helper.absoluteVec(new Vec3(2.5, 3.0, 2.5));
        fly.setDeltaMovement(0.2, 0.3, 0.4);

        fly.resetForFixture(reset.x, reset.y, reset.z, 90.0F);

        helper.assertTrue(
                fly.position().distanceTo(reset) < 1.0e-9,
                Component.literal("fixture reset position mismatch"));
        helper.assertTrue(
                fly.getDeltaMovement().equals(Vec3.ZERO),
                Component.literal("fixture reset must clear velocity"));
        helper.assertTrue(
                fly.presentationAction().isNeutral(),
                Component.literal("fixture reset must clear applied action"));
        helper.succeed();
    }

    @GameTest(maxTicks = 20)
    public void stagedResetWaitsForAcceptedRecordingAckEffectiveTick(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 2.0, 1.5));
        Vec3 initial = fly.position();
        Vec3 target = helper.absoluteVec(new Vec3(2.5, 3.0, 2.5));
        long issuedTick = helper.getLevel().getGameTime();
        StagedReset reset = new StagedReset(
                fly.getUUID(), issuedTick, target.x, target.y, target.z, 90);
        reset.bindControlSequence(0);

        helper.assertTrue(
                reset.advance(issuedTick, Optional.empty()) == StagedReset.Outcome.WAITING,
                Component.literal("reset advanced before an acknowledgement"));
        helper.assertTrue(
                reset.advance(
                                issuedTick,
                                Optional.of(resetAck(fly, issuedTick, issuedTick + 2, true)))
                        == StagedReset.Outcome.WAITING,
                Component.literal("delayed reset applied before its effective tick"));
        helper.assertTrue(
                fly.position().equals(initial),
                Component.literal("staged reset changed world state before its effective tick"));

        helper.runAfterDelay(2, () -> {
            long currentTick = helper.getLevel().getGameTime();
            helper.assertTrue(
                    reset.advance(currentTick, Optional.empty()) == StagedReset.Outcome.READY,
                    Component.literal("accepted reset was not ready at its effective tick"));
            reset.apply(fly, currentTick);
            PhysicalSensorSnapshot firstGovernedSample =
                    FlySensors.sample(helper.getLevel(), fly);
            helper.assertTrue(
                    fly.position().distanceTo(target) < 1.0e-9,
                    Component.literal("accepted reset did not apply at its effective tick"));
            helper.assertTrue(
                    firstGovernedSample.gameTick() == currentTick
                            && firstGovernedSample.xBlocks() == target.x
                            && firstGovernedSample.yBlocks() == target.y
                            && firstGovernedSample.zBlocks() == target.z,
                    Component.literal("reset was not visible to the effective-tick sensor sample"));
            helper.succeed();
        });
    }

    @GameTest(maxTicks = 20)
    public void rejectedRecordingResetLeavesWorldUntouched(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 2.0, 1.5));
        Vec3 initial = fly.position();
        long issuedTick = helper.getLevel().getGameTime();
        StagedReset reset = new StagedReset(
                fly.getUUID(), issuedTick, initial.x + 4, initial.y + 4, initial.z + 4, 90);
        reset.bindControlSequence(0);

        helper.assertTrue(
                reset.advance(
                                issuedTick,
                                Optional.of(resetAck(fly, issuedTick, issuedTick, false)))
                        == StagedReset.Outcome.REJECTED,
                Component.literal("rejected reset did not surface its terminal outcome"));
        helper.assertTrue(
                fly.position().equals(initial),
                Component.literal("rejected reset changed authoritative world state"));
        helper.succeed();
    }

    // ADR 0014 (D-064) regression pin: under the symmetric hover envelope, full
    // positive vertical thrust from a grounded start must produce liftoff. Under
    // the removed ambient-gravity term this exact case was impossible — gravity
    // equaled the maximum vertical thrust authority, so vy' = 0.98 * vy at full
    // command and the floor was an absorbing state — and this test fails there.
    @GameTest(maxTicks = 40)
    public void fullVerticalThrustFromGroundedStartProducesLiftoff(GameTestHelper helper) {
        helper.setBlock(new BlockPos(1, 1, 1), Blocks.STONE);
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 2.0, 1.5));
        fly.setYRot(0.0F);
        double floorTopY = fly.getY();
        int[] tickCounter = {0};

        helper.onEachTick(() -> {
            int tick = tickCounter[0]++;
            if (tick < 4) {
                PrimitiveActuator.apply(
                        fly,
                        new RawPrimitiveAction(0, 0, -1, 0, 0),
                        new PrimitiveAction(0, 0, -1, 0, 0));
                return;
            }
            if (tick == 4) {
                helper.assertTrue(
                        fly.onGround(),
                        Component.literal("commanded descent did not ground the fly"));
                helper.assertTrue(
                        Math.abs(fly.getY() - floorTopY) < 1.0e-6,
                        Component.literal("grounded fly is not resting on the floor"));
            }
            if (tick <= 17) {
                PrimitiveActuator.apply(
                        fly,
                        new RawPrimitiveAction(0, 0, 1, 0, 0),
                        new PrimitiveAction(0, 0, 1, 0, 0));
                return;
            }
            helper.assertTrue(
                    fly.getY() > floorTopY + 0.5,
                    Component.literal("full positive vertical thrust did not lift the fly"));
            helper.assertFalse(
                    fly.onGround(),
                    Component.literal("climbing fly still reports ground contact"));
            helper.succeed();
        });
    }

    // ADR 0014 (D-064) regression pin: neutral motors hold altitude. The hover
    // envelope has no ambient vertical force, so an airborne fly with zero
    // velocity stays exactly put — fail-neutral means hover-hold, not free-fall.
    // Fails on the removed physics, where the fly sank 0.04 blocks/tick^2.
    @GameTest(maxTicks = 20)
    public void neutralMotorsHoldAltitudeWithoutAmbientSink(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 3.5, 1.5));
        Vec3 initial = fly.position();

        helper.runAfterDelay(10, () -> {
            helper.assertTrue(
                    fly.position().distanceTo(initial) < 1.0e-9,
                    Component.literal("neutral motors did not hold altitude"));
            helper.assertTrue(
                    fly.getDeltaMovement().equals(Vec3.ZERO),
                    Component.literal("neutral motors accumulated velocity"));
            helper.assertFalse(
                    fly.onGround(),
                    Component.literal("hovering fly reported ground contact"));
            helper.succeed();
        });
    }

    // ADR 0014 (D-064) regression pin: landing stays possible — a commanded
    // descent still reaches the floor, grounds the fly, and leaves it at rest.
    @GameTest(maxTicks = 60)
    public void commandedDescentStillReachesTheFloorAndGrounds(GameTestHelper helper) {
        helper.setBlock(new BlockPos(1, 1, 1), Blocks.STONE);
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 3.5, 1.5));
        double floorTopY = helper.absoluteVec(new Vec3(1.5, 2.0, 1.5)).y;

        helper.onEachTick(() -> {
            if (!fly.onGround()) {
                PrimitiveActuator.apply(
                        fly,
                        new RawPrimitiveAction(0, 0, -1, 0, 0),
                        new PrimitiveAction(0, 0, -1, 0, 0));
                return;
            }
            helper.assertTrue(
                    Math.abs(fly.getY() - floorTopY) < 1.0e-6,
                    Component.literal("landed fly did not rest on the floor surface"));
            helper.assertTrue(
                    fly.getDeltaMovement().y == 0.0,
                    Component.literal("grounded fly retained vertical velocity"));
            helper.succeed();
        });
    }

    @GameTest(maxTicks = 20)
    public void primitiveActuatorAppliesBodyFrameThrustWithoutTeleport(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 3.0, 1.5));
        fly.setYRot(0.0F);
        Vec3 beforePosition = fly.position();

        PrimitiveActuator.apply(
                fly,
                new RawPrimitiveAction(1, 0, 0, 0, 0),
                new PrimitiveAction(1, 0, 0, 0, 0));

        helper.assertTrue(
                fly.position().equals(beforePosition),
                Component.literal("actuator must not teleport or set absolute position"));
        helper.assertTrue(
                Math.abs(fly.getDeltaMovement().x) < 1.0e-9
                        && fly.getDeltaMovement().z > 0.0,
                Component.literal("yaw-zero forward thrust must be +Z velocity delta"));
        helper.succeed();
    }

    @GameTest(maxTicks = 20)
    public void physicalForwardRayReportsObstacleDistance(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 2.0, 1.5));
        fly.setYRot(0.0F);
        fly.setXRot(0.0F);
        PhysicalSensorSnapshot open = FlySensors.sample(helper.getLevel(), fly);

        BlockPos obstacle = new BlockPos(1, 2, 3);
        helper.setBlock(obstacle, Blocks.STONE);
        PhysicalSensorSnapshot blocked = FlySensors.sample(helper.getLevel(), fly);

        helper.assertTrue(
                open.rayForwardBlocks() == FlySensors.RAY_LENGTH_BLOCKS,
                Component.literal("open arena did not report the declared ray maximum"));
        helper.assertTrue(
                blocked.rayForwardBlocks() > 0.0
                        && blocked.rayForwardBlocks() < open.rayForwardBlocks(),
                Component.literal("block perturbation did not shorten the forward physical ray"));
        helper.succeed();
    }

    // D-068 lateral handedness probe, ADR 0015. The two lateral rays are body-relative, so the
    // slot the protocol calls "right" must be the one that shortens when an obstacle
    // sits on the body's right. Handedness is taken from the engine's own table
    // (Direction.getClockWise()), never from a constant chosen here; RaySideFence
    // carries the worked derivation at yaw 0 and yaw 90.
    @GameTest(maxTicks = 20)
    public void declaredRightRayReportsObstacleOnTheBodysRightSide(GameTestHelper helper) {
        float yaw = 0.0F;
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, LATERAL_PROBE_SPAWN);
        fly.setYRot(yaw);
        fly.setXRot(0.0F);
        Direction bodyRight = RaySideFence.bodyRight(yaw);

        helper.assertTrue(
                RaySideFence.facing(yaw) == Direction.SOUTH && bodyRight == Direction.WEST,
                Component.literal("engine yaw table changed: yaw 0 must face south, right west"));
        helper.assertTrue(
                RaySideFence.obstacleLiesOn(
                        bodyRight, fly.position(), helper.absolutePos(WEST_OBSTACLE)),
                Component.literal("probe obstacle is not on the body's right in world coordinates"));

        PhysicalSensorSnapshot open = FlySensors.sample(helper.getLevel(), fly);
        helper.setBlock(WEST_OBSTACLE, Blocks.STONE);
        PhysicalSensorSnapshot blocked = FlySensors.sample(helper.getLevel(), fly);
        RaySideFence.LateralSlot shortened = RaySideFence.shortenedSlot(open, blocked);

        // Prove the probe measures what it claims before trusting which slot moved:
        // the obstacle is in range and seen on exactly one lateral slot, that slot
        // reports the exact block-face distance, nothing else moved, and the body is
        // still at the pinned yaw.
        helper.assertTrue(
                shortened == RaySideFence.LateralSlot.LEFT
                        || shortened == RaySideFence.LateralSlot.RIGHT,
                Component.literal(
                        "lateral obstacle was not seen on exactly one declared slot: " + shortened));
        helper.assertTrue(
                Math.abs(RaySideFence.reading(blocked, shortened)
                                - RaySideFence.EXPECTED_OBSTACLE_DISTANCE_BLOCKS)
                        < RaySideFence.TOLERANCE_BLOCKS,
                Component.literal("shortened slot did not report the exact block-face distance"));
        helper.assertTrue(
                RaySideFence.nonLateralRaysUnchanged(open, blocked),
                Component.literal("a purely lateral obstacle perturbed a non-lateral ray"));
        helper.assertTrue(
                blocked.yawDegrees() == yaw,
                Component.literal("body yaw drifted away from the pinned probe yaw"));

        helper.assertTrue(
                shortened == RaySideFence.expectedSlot(yaw, bodyRight),
                Component.literal(
                        "lateral sensor rays are swapped: at yaw " + yaw + " the body faces "
                                + RaySideFence.facing(yaw) + " so its right is " + bodyRight
                                + ", but the obstacle on that side shortened the declared "
                                + shortened + " ray"));
        helper.succeed();
    }

    // D-068 lateral handedness probe, ADR 0015, controls. Flipping the obstacle to the opposite
    // side, and flipping the body 180 degrees under a fixed obstacle, must each move
    // the evidence to the other slot. Those relative checks hold under a consistent
    // left/right transposition, so they pass either way and prove the sensor really
    // does resolve the two sides; only the absolute mapping below can go red.
    @GameTest(maxTicks = 20)
    public void lateralRaySlotsFollowTheBodyUnderSideSwapAndYawFlip(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, LATERAL_PROBE_SPAWN);
        fly.setXRot(0.0F);
        fly.setYRot(0.0F);

        PhysicalSensorSnapshot open = FlySensors.sample(helper.getLevel(), fly);
        helper.setBlock(WEST_OBSTACLE, Blocks.STONE);
        RaySideFence.LateralSlot westSlot =
                RaySideFence.shortenedSlot(open, FlySensors.sample(helper.getLevel(), fly));
        helper.setBlock(WEST_OBSTACLE, Blocks.AIR);
        helper.setBlock(EAST_OBSTACLE, Blocks.STONE);
        RaySideFence.LateralSlot eastSlot =
                RaySideFence.shortenedSlot(open, FlySensors.sample(helper.getLevel(), fly));

        helper.setBlock(EAST_OBSTACLE, Blocks.AIR);
        fly.setYRot(180.0F);
        PhysicalSensorSnapshot flippedOpen = FlySensors.sample(helper.getLevel(), fly);
        helper.setBlock(EAST_OBSTACLE, Blocks.STONE);
        RaySideFence.LateralSlot flippedSlot = RaySideFence.shortenedSlot(
                flippedOpen, FlySensors.sample(helper.getLevel(), fly));

        helper.assertTrue(
                westSlot != RaySideFence.LateralSlot.NEITHER
                        && eastSlot != RaySideFence.LateralSlot.NEITHER
                        && flippedSlot != RaySideFence.LateralSlot.NEITHER,
                Component.literal("a lateral obstacle went undetected by both declared slots"));
        helper.assertTrue(
                westSlot != eastSlot,
                Component.literal("obstacles on opposite sides reported the same declared slot"));
        helper.assertTrue(
                flippedSlot != eastSlot,
                Component.literal("a 180 degree yaw flip did not move the fixed obstacle's slot"));

        helper.assertTrue(
                eastSlot == RaySideFence.expectedSlot(0.0F, Direction.EAST)
                        && flippedSlot == RaySideFence.expectedSlot(180.0F, Direction.EAST),
                Component.literal(
                        "lateral slots are transposed: an east obstacle is on the body's left at "
                                + "yaw 0 and its right at yaw 180, but reported slots were "
                                + eastSlot + " and " + flippedSlot));
        helper.succeed();
    }

    // D-068 lateral handedness probe, ADR 0015, through the real tick path. FlySensors.sample is
    // the same call FlyEntity.tick() makes, but this pins the defect in the bounded
    // snapshot history that is actually published to the neural service: the fence
    // admits only a snapshot sampled strictly after the obstacle was placed, and the
    // yaw is re-checked on that exact snapshot rather than assumed.
    @GameTest(maxTicks = 20)
    public void tickPathSnapshotCarriesTheSameLateralAssignment(GameTestHelper helper) {
        float yaw = 0.0F;
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, LATERAL_PROBE_SPAWN);
        fly.setYRot(yaw);
        fly.setXRot(0.0F);
        PhysicalSensorSnapshot open = FlySensors.sample(helper.getLevel(), fly);
        long placementTick = helper.getLevel().getGameTime();
        helper.setBlock(WEST_OBSTACLE, Blocks.STONE);
        RaySideFence fence = new RaySideFence(placementTick);

        helper.runAfterDelay(3, () -> {
            Optional<PhysicalSensorSnapshot> retained =
                    fence.latestPostPlacementSnapshot(fly, helper.getLevel().getGameTime());
            helper.assertTrue(
                    retained.isPresent(),
                    Component.literal("no retained snapshot was sampled after obstacle placement"));
            PhysicalSensorSnapshot blocked = retained.orElseThrow();

            helper.assertTrue(
                    blocked.gameTick() > placementTick,
                    Component.literal("admitted a snapshot from before the obstacle existed"));
            helper.assertTrue(
                    blocked.yawDegrees() == yaw,
                    Component.literal("body yaw was rewritten before the measured tick"));
            helper.assertTrue(
                    blocked.xBlocks() == open.xBlocks() && blocked.zBlocks() == open.zBlocks(),
                    Component.literal("body drifted, so the open baseline is not comparable"));

            RaySideFence.LateralSlot shortened = RaySideFence.shortenedSlot(open, blocked);
            helper.assertTrue(
                    shortened == RaySideFence.LateralSlot.LEFT
                            || shortened == RaySideFence.LateralSlot.RIGHT,
                    Component.literal(
                            "published snapshot saw no single lateral slot: " + shortened));
            helper.assertTrue(
                    Math.abs(RaySideFence.reading(blocked, shortened)
                                    - RaySideFence.EXPECTED_OBSTACLE_DISTANCE_BLOCKS)
                            < RaySideFence.TOLERANCE_BLOCKS,
                    Component.literal("published snapshot lost the exact block-face distance"));
            helper.assertTrue(
                    shortened == RaySideFence.expectedSlot(yaw, RaySideFence.bodyRight(yaw)),
                    Component.literal(
                            "published snapshot carries swapped lateral rays: obstacle on the "
                                    + "body's " + RaySideFence.bodyRight(yaw)
                                    + " side (its right) shortened the declared " + shortened
                                    + " ray"));
            helper.succeed();
        });
    }

    // D-068, ADR 0015, actuation side. BodyFrameMath is the motor twin of the FlySensors
    // lateral axis and repeated its convention. It was latent *only under the faithful
    // pinned projection*, which hardwires lateral_thrust to 0 (controller.py). It was not
    // latent under the TRM demo projection: TrainedReadoutController._project_motor_frame
    // passes lateral through, and EXTERNAL-BRIDGE-06 records that path driving this method
    // against the real mod across nine APPLIED actuation records, each carrying a non-zero
    // lateral_thrust between 0.5758 and 1.0. So
    // the wrong-handed axis did execute, and the corrected axis changes TRM demo behavior.
    // Pure component check, in the same world-free style as the pacer and fence tests above.
    @GameTest(maxTicks = 20)
    public void bodyFrameLateralAxisMatchesEngineHandedness(GameTestHelper helper) {
        for (float yaw : new float[] {0.0F, 90.0F}) {
            Direction facing = RaySideFence.facing(yaw);
            Direction bodyRight = RaySideFence.bodyRight(yaw);
            BodyFrameMath.HorizontalVector forwardOnly = BodyFrameMath.toWorld(yaw, 1.0, 0.0);
            BodyFrameMath.HorizontalVector lateralOnly = BodyFrameMath.toWorld(yaw, 0.0, 1.0);

            // Anchor: the forward axis already agrees with the engine, which shows this
            // check uses the engine's yaw convention and isolates the defect laterally.
            helper.assertTrue(
                    Math.abs(forwardOnly.x() - facing.getStepX()) < 1.0e-9
                            && Math.abs(forwardOnly.z() - facing.getStepZ()) < 1.0e-9,
                    Component.literal(
                            "BodyFrameMath forward axis disagrees with the engine at yaw " + yaw));
            helper.assertTrue(
                    Math.abs(lateralOnly.x() - bodyRight.getStepX()) < 1.0e-9
                            && Math.abs(lateralOnly.z() - bodyRight.getStepZ()) < 1.0e-9,
                    Component.literal(
                            "BodyFrameMath lateral axis is the body's left: at yaw " + yaw
                                    + " the body faces " + facing + " so its right is " + bodyRight
                                    + " " + vector(bodyRight) + ", but unit lateral thrust gives ("
                                    + lateralOnly.x() + ", " + lateralOnly.z() + ")"));
        }
        helper.succeed();
    }

    // ADR 0015, vertical axis. The up/down pair was accidentally correct before this lane:
    // FlySensors built bodyUp as forward x right with a left-handed "right", and the two
    // sign errors cancelled. Correcting the lateral axis alone would have inverted them,
    // and nothing in the repository read rayUpBlocks or rayDownBlocks. These two probes
    // are the evidence that the vertical axis survived the correction, rather than a
    // reader's confidence that it did. Expectations come from RayVerticalFence, which
    // derives body-up from the engine's own look-direction table and never from the cross
    // products under test.
    @GameTest(maxTicks = 20)
    public void verticalRaySlotsMatchTheBodyUpAxisAtLevelPitch(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, VERTICAL_PROBE_SPAWN);
        fly.setYRot(0.0F);
        fly.setXRot(0.0F);
        double originY = fly.getBoundingBox().getCenter().y;

        helper.assertTrue(
                RayVerticalFence.liesInColumnOf(fly.position(), helper.absolutePos(ABOVE_OBSTACLE))
                        && RayVerticalFence.liesInColumnOf(
                                fly.position(), helper.absolutePos(BELOW_OBSTACLE)),
                Component.literal("probe obstacles are not in the body's own block column"));
        helper.assertTrue(
                RayVerticalFence.bodyUp(0.0F, 0.0F).y > 0.999,
                Component.literal("engine look table changed: body-up at level pitch is not world up"));

        PhysicalSensorSnapshot open = FlySensors.sample(helper.getLevel(), fly);
        helper.setBlock(ABOVE_OBSTACLE, Blocks.STONE);
        PhysicalSensorSnapshot ceiling = FlySensors.sample(helper.getLevel(), fly);
        helper.setBlock(ABOVE_OBSTACLE, Blocks.AIR);
        helper.setBlock(BELOW_OBSTACLE, Blocks.STONE);
        PhysicalSensorSnapshot floor = FlySensors.sample(helper.getLevel(), fly);

        assertVerticalHit(
                helper,
                open,
                ceiling,
                RayVerticalFence.VerticalSlot.UP,
                RayVerticalFence.distanceUpToBottomFace(
                        originY, helper.absolutePos(ABOVE_OBSTACLE)),
                "a block directly overhead");
        assertVerticalHit(
                helper,
                open,
                floor,
                RayVerticalFence.VerticalSlot.DOWN,
                RayVerticalFence.distanceDownToTopFace(originY, helper.absolutePos(BELOW_OBSTACLE)),
                "a block directly underfoot");
        helper.succeed();
    }

    // ADR 0015, vertical axis under pitch. The level-pitch probe above separates up from
    // down, but world-locked up would also pass it. This one pitches the nose up 40
    // degrees and measures the ceiling through the body-up ray: the correct ray leans
    // north by exactly the pitch angle and so travels 1/cos(40 deg) further to the same
    // horizontal plane. World-locked up would report the plain vertical distance and an
    // inverted axis would look at the floor -- both more than half a block away.
    @GameTest(maxTicks = 20)
    public void bodyUpRayTiltsWithPitchInsteadOfTrackingWorldUp(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, VERTICAL_PROBE_SPAWN);
        fly.setYRot(0.0F);
        fly.setXRot(0.0F);
        double originY = fly.getBoundingBox().getCenter().y;
        double ceilingFaceY = helper.absolutePos(ABOVE_OBSTACLE).getY();
        for (int x = CEILING_MIN; x <= CEILING_MAX; x++) {
            for (int z = CEILING_MIN; z <= CEILING_MAX; z++) {
                helper.setBlock(new BlockPos(x, ABOVE_OBSTACLE.getY(), z), Blocks.STONE);
            }
        }

        double level = FlySensors.sample(helper.getLevel(), fly).rayUpBlocks();
        fly.setXRot(PROBE_PITCH_DEGREES);
        PhysicalSensorSnapshot pitched = FlySensors.sample(helper.getLevel(), fly);

        Vec3 bodyUp = RayVerticalFence.bodyUp(PROBE_PITCH_DEGREES, 0.0F);
        double expectedLevel = ceilingFaceY - originY;
        double expectedPitched = expectedLevel / bodyUp.y;
        helper.assertTrue(
                bodyUp.y > 0.0 && bodyUp.z < -0.5,
                Component.literal(
                        "engine look table changed: body-up at pitch " + PROBE_PITCH_DEGREES
                                + " must lean up and north, got " + bodyUp));
        helper.assertTrue(
                Math.abs(level - expectedLevel) < RayVerticalFence.TOLERANCE_BLOCKS,
                Component.literal(
                        "level-pitch ceiling reading " + level + " is not the vertical distance "
                                + expectedLevel));
        helper.assertTrue(
                Math.abs(pitched.rayUpBlocks() - expectedPitched)
                        < RayVerticalFence.PITCHED_TOLERANCE_BLOCKS,
                Component.literal(
                        "body-up ray does not follow the body: at pitch " + PROBE_PITCH_DEGREES
                                + " the ceiling is " + expectedPitched + " blocks along body-up "
                                + bodyUp + ", but the declared up ray reports "
                                + pitched.rayUpBlocks() + " (world-locked up would report "
                                + expectedLevel + ")"));
        helper.assertTrue(
                pitched.rayUpBlocks() > level + 0.5,
                Component.literal("pitching the body did not lengthen the path to the ceiling"));
        helper.succeed();
    }

    private static void assertVerticalHit(
            GameTestHelper helper,
            PhysicalSensorSnapshot open,
            PhysicalSensorSnapshot blocked,
            RayVerticalFence.VerticalSlot expected,
            double expectedDistance,
            String what) {
        RayVerticalFence.VerticalSlot shortened = RayVerticalFence.shortenedSlot(open, blocked);
        helper.assertTrue(
                RayVerticalFence.nonVerticalRaysUnchanged(open, blocked),
                Component.literal(what + " perturbed a non-vertical ray"));
        helper.assertTrue(
                shortened == expected,
                Component.literal(
                        "vertical sensor rays are inverted: " + what
                                + " must shorten the declared " + expected
                                + " ray, but the slot that moved was " + shortened));
        helper.assertTrue(
                Math.abs(RayVerticalFence.reading(blocked, shortened) - expectedDistance)
                        < RayVerticalFence.TOLERANCE_BLOCKS,
                Component.literal(
                        "declared " + expected + " ray reports "
                                + RayVerticalFence.reading(blocked, shortened)
                                + " for " + what + ", not the exact block-face distance "
                                + expectedDistance));
    }

    // ---------------------------------------------------------------- D-070 entity rays, ADR 0017
    //
    // Before this lane the six rays were cast with a single level.clip(...) under
    // ClipContext.Block.COLLIDER, which consults blocks only. An operator standing in the fly's
    // path produced a bit-identical sensor vector to an empty arena. The rays now sweep entities
    // over the SAME segment and keep the nearer of the block hit and the entity hit. What widened
    // is only what counts as an obstruction; the declared length, the body-aligned directions
    // (ADR 0015) and the returned meaning are untouched.
    //
    // Every probe below spawns the body at DEPTH_PROBE_SPAWN facing yaw 0, so forward is +Z (south)
    // and back is -Z (north) per the engine yaw table derived in RaySideFence. Expected distances
    // come from RayEntityFence, which does plain arithmetic on the target's own bounding box or
    // BlockPos and never calls AABB.clip -- the routine the production sweep uses internally.

    /**
     * A ray must shorten on a living body placed in its path. This is the direct pin on the defect:
     * before ADR 0017 this test's blocked sample was bit-identical to its open sample.
     */
    @GameTest(maxTicks = 20)
    public void declaredRayShortensOnALivingBodyInItsPath(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, DEPTH_PROBE_SPAWN);
        fly.setYRot(0.0F);
        fly.setXRot(0.0F);
        helper.assertTrue(
                RaySideFence.facing(0.0F) == Direction.SOUTH,
                Component.literal("engine yaw table changed: yaw 0 must face south"));

        Vec3 origin = fly.getBoundingBox().getCenter();
        PhysicalSensorSnapshot open = FlySensors.sample(helper.getLevel(), fly);

        ArmorStand body = helper.spawn(EntityType.ARMOR_STAND, FORWARD_NEAR_BODY);
        AABB box = body.getBoundingBox();
        PhysicalSensorSnapshot blocked = FlySensors.sample(helper.getLevel(), fly);

        // Prove the probe measures what it claims before trusting the reading: the body really
        // straddles the forward ray line, and the declared predicate really admits it.
        helper.assertTrue(
                RayEntityFence.straddlesDepthRay(origin, box),
                Component.literal("probe body is not astride the forward ray line"));
        helper.assertTrue(
                FlySensors.obstructsSensorRay(body),
                Component.literal("declared predicate refused a non-marker living body"));

        double expected = RayEntityFence.distanceSouthToNearFace(origin, box);
        helper.assertTrue(
                expected > 0.0 && expected < open.rayForwardBlocks(),
                Component.literal(
                        "probe body is not strictly nearer than the open-arena forward reading "
                                + open.rayForwardBlocks() + ", so the probe would have no teeth"));
        helper.assertTrue(
                blocked.rayForwardBlocks() < open.rayForwardBlocks(),
                Component.literal(
                        "a living body in the ray path did not shorten the forward ray: still "
                                + blocked.rayForwardBlocks()
                                + ". The rays are blocks-only again."));
        helper.assertTrue(
                RayEntityFence.reads(blocked.rayForwardBlocks(), expected),
                Component.literal(
                        "forward ray reports " + blocked.rayForwardBlocks()
                                + " for a body whose exact near face is at " + expected
                                + "; a non-zero hitbox inflation would show up here"));
        helper.assertTrue(
                RayEntityFence.nonForwardRaysUnchanged(open, blocked),
                Component.literal("a body on the forward ray alone perturbed another ray"));
        helper.succeed();
    }

    /**
     * The sensing fly must not see itself, and must not see another fly. Two different mechanisms:
     * self is excluded by {@code Level.getEntities(Entity except, ...)}, a second fly by the
     * declared predicate's {@code FlyEntity} clause. The ray origin is the body's own bounding-box
     * centre, so a self-hit would collapse all six rays to nearly zero at once.
     */
    @GameTest(maxTicks = 20)
    public void sensorRaysIgnoreTheSensingFlyAndEveryOtherFly(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, DEPTH_PROBE_SPAWN);
        fly.setYRot(0.0F);
        fly.setXRot(0.0F);
        Vec3 origin = fly.getBoundingBox().getCenter();

        PhysicalSensorSnapshot alone = FlySensors.sample(helper.getLevel(), fly);
        helper.assertTrue(
                RayEntityFence.straddlesDepthRay(origin, fly.getBoundingBox()),
                Component.literal("the ray origin is not inside the sensing body's own hitbox"));
        helper.assertTrue(
                RayEntityFence.allRaysAtLeast(alone, SELF_HIT_FLOOR_BLOCKS),
                Component.literal(
                        "the body sees itself: a ray collapsed below " + SELF_HIT_FLOOR_BLOCKS
                                + " in an unobstructed structure (forward read "
                                + alone.rayForwardBlocks() + ")"));
        helper.assertFalse(
                FlySensors.obstructsSensorRay(fly),
                Component.literal("declared predicate admitted a fly"));

        FlyEntity second = helper.spawn(NeuroCraftFly.FLY, FORWARD_SHALLOW_BODY);
        PhysicalSensorSnapshot withSecond = FlySensors.sample(helper.getLevel(), fly);
        helper.assertTrue(
                RayEntityFence.straddlesDepthRay(origin, second.getBoundingBox()),
                Component.literal("the second fly is not astride the forward ray line"));
        helper.assertFalse(
                FlySensors.obstructsSensorRay(second),
                Component.literal("declared predicate admitted a second fly"));
        helper.assertTrue(
                RayEntityFence.allRaysUnchanged(alone, withSecond),
                Component.literal(
                        "a second fly astride the forward ray moved a reading to "
                                + withSecond.rayForwardBlocks()
                                + "; flies are declared not to sense each other"));
        helper.succeed();
    }

    /**
     * The nearer of a block hit and an entity hit wins, proven in both orders and only after each
     * candidate has been shown to be seen on its own. The forward ray takes a block first and then
     * a nearer body; the back ray takes a body first and then a nearer block.
     */
    @GameTest(maxTicks = 20)
    public void nearerOfBlockAndEntityWinsInBothOrders(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, DEPTH_PROBE_SPAWN);
        fly.setYRot(0.0F);
        fly.setXRot(0.0F);
        Vec3 origin = fly.getBoundingBox().getCenter();

        // Forward: a block alone, then a body nearer than it.
        helper.setBlock(FORWARD_FAR_BLOCK, Blocks.STONE);
        BlockPos forwardBlock = helper.absolutePos(FORWARD_FAR_BLOCK);
        double blockDistance = RayEntityFence.distanceSouthToNearFace(origin, forwardBlock);
        PhysicalSensorSnapshot blockOnly = FlySensors.sample(helper.getLevel(), fly);
        helper.assertTrue(blockOnly.rayHitKinds().get(0) == 1,
                Component.literal("wrong nearest-hit kind for blockOnly"));
        helper.assertTrue(
                RayEntityFence.reads(blockOnly.rayForwardBlocks(), blockDistance),
                Component.literal(
                        "forward block alone reports " + blockOnly.rayForwardBlocks()
                                + ", not its exact face distance " + blockDistance));

        ArmorStand nearBody = helper.spawn(EntityType.ARMOR_STAND, FORWARD_NEAR_BODY);
        double nearBodyDistance =
                RayEntityFence.distanceSouthToNearFace(origin, nearBody.getBoundingBox());
        helper.assertTrue(
                RayEntityFence.straddlesDepthRay(origin, nearBody.getBoundingBox())
                        && nearBodyDistance < blockDistance,
                Component.literal("forward probe body is not astride the ray and nearer"));
        PhysicalSensorSnapshot entityNearer = FlySensors.sample(helper.getLevel(), fly);
        helper.assertTrue(entityNearer.rayHitKinds().get(0) == 2,
                Component.literal("wrong nearest-hit kind for entityNearer"));
        helper.assertTrue(
                RayEntityFence.reads(entityNearer.rayForwardBlocks(), nearBodyDistance),
                Component.literal(
                        "a body nearer than a block did not win the forward ray: reported "
                                + entityNearer.rayForwardBlocks() + ", expected "
                                + nearBodyDistance + " (block was at " + blockDistance + ")"));

        // Back: a body alone, then a block nearer than it.
        ArmorStand farBody = helper.spawn(EntityType.ARMOR_STAND, BACK_FAR_BODY);
        double farBodyDistance =
                RayEntityFence.distanceNorthToNearFace(origin, farBody.getBoundingBox());
        helper.assertTrue(
                RayEntityFence.straddlesDepthRay(origin, farBody.getBoundingBox()),
                Component.literal("back probe body is not astride the back ray line"));
        PhysicalSensorSnapshot bodyOnly = FlySensors.sample(helper.getLevel(), fly);
        helper.assertTrue(bodyOnly.rayHitKinds().get(1) == 2,
                Component.literal("wrong nearest-hit kind for bodyOnly"));
        helper.assertTrue(
                RayEntityFence.reads(bodyOnly.rayBackBlocks(), farBodyDistance),
                Component.literal(
                        "back body alone reports " + bodyOnly.rayBackBlocks()
                                + ", not its exact near-face distance " + farBodyDistance));

        helper.setBlock(BACK_NEAR_BLOCK, Blocks.STONE);
        BlockPos backBlock = helper.absolutePos(BACK_NEAR_BLOCK);
        double backBlockDistance = RayEntityFence.distanceNorthToNearFace(origin, backBlock);
        helper.assertTrue(
                backBlockDistance < farBodyDistance,
                Component.literal("back probe block is not nearer than the back probe body"));
        PhysicalSensorSnapshot blockNearer = FlySensors.sample(helper.getLevel(), fly);
        helper.assertTrue(blockNearer.rayHitKinds().get(1) == 1,
                Component.literal("wrong nearest-hit kind for blockNearer"));
        helper.assertTrue(
                RayEntityFence.reads(blockNearer.rayBackBlocks(), backBlockDistance),
                Component.literal(
                        "a block nearer than a body did not win the back ray: reported "
                                + blockNearer.rayBackBlocks() + ", expected " + backBlockDistance
                                + " (body was at " + farBodyDistance + ")"));
        helper.assertTrue(
                RayEntityFence.reads(blockNearer.rayForwardBlocks(), nearBodyDistance),
                Component.literal("the back arrangement disturbed the forward reading"));
        helper.succeed();
    }

    /**
     * The declared predicate's exclusions, exercised through the real sweep. An item entity
     * straddling the ray must leave every reading bit-identical: it is not ray-pickable, and the
     * fly's bounded actuation passes straight through it, so a reading for it would be a phantom.
     */
    @GameTest(maxTicks = 20)
    public void nonPickableEntitiesDoNotObstructAnyRay(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, DEPTH_PROBE_SPAWN);
        fly.setYRot(0.0F);
        fly.setXRot(0.0F);
        Vec3 origin = fly.getBoundingBox().getCenter();
        PhysicalSensorSnapshot open = FlySensors.sample(helper.getLevel(), fly);

        ItemEntity item = helper.spawnItem(Items.STICK, FORWARD_SHALLOW_BODY);
        PhysicalSensorSnapshot withItem = FlySensors.sample(helper.getLevel(), fly);

        // Without this the test would pass merely because the item drifted out of the ray.
        helper.assertTrue(
                RayEntityFence.straddlesDepthRay(origin, item.getBoundingBox()),
                Component.literal("probe item is not astride the forward ray line"));
        helper.assertTrue(
                RayEntityFence.distanceSouthToNearFace(origin, item.getBoundingBox())
                        < FlySensors.RAY_LENGTH_BLOCKS,
                Component.literal("probe item is beyond the declared ray length"));
        helper.assertFalse(
                FlySensors.obstructsSensorRay(item),
                Component.literal("declared predicate admitted a non-pickable item entity"));
        helper.assertTrue(
                RayEntityFence.allRaysUnchanged(open, withItem),
                Component.literal(
                        "an item entity astride the forward ray moved a reading to "
                                + withItem.rayForwardBlocks()));
        helper.succeed();
    }

    /**
     * The projectile half of the declared predicate, pinned on both sides of the tag that decides
     * it. {@code Projectile} <em>does</em> override {@code isPickable()}, to
     * {@code getType().is(EntityTypeTags.REDIRECTABLE_PROJECTILE)}, and that tag is exactly
     * {@code {fireball, wind_charge, breeze_wind_charge}}. So a wind charge obstructs a ray and an
     * arrow does not, and ADR 0017 accepts the former deliberately: a redirectable projectile is a
     * thing that physically interacts with the fly's body, which is what the predicate is for.
     *
     * <p>The two targets sit on opposite rays so neither can mask the other, and the probe asserts
     * the tag membership itself rather than only the outcome — a datapack or version change that
     * moved {@code minecraft:wind_charge} out of the tag would then name its own cause instead of
     * surfacing as an unexplained distance.
     *
     * <p><b>Pinned against projectile motion.</b> No world tick elapses between the spawns and the
     * samples, and both targets additionally have their delta movement zeroed and gravity disabled
     * before the reading is taken, so neither can drift, fall, or strike anything while the probe
     * runs. Both are discarded before {@code succeed()} so no live wind charge outlives the test.
     */
    @GameTest(maxTicks = 20)
    public void taggedRedirectableProjectileObstructsWhileUntaggedProjectileDoesNot(
            GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, DEPTH_PROBE_SPAWN);
        fly.setYRot(0.0F);
        fly.setXRot(0.0F);
        Vec3 origin = fly.getBoundingBox().getCenter();
        PhysicalSensorSnapshot open = FlySensors.sample(helper.getLevel(), fly);

        WindCharge charge = helper.spawn(EntityType.WIND_CHARGE, FORWARD_SHALLOW_BODY);
        charge.setDeltaMovement(Vec3.ZERO);
        charge.setNoGravity(true);
        Arrow arrow = helper.spawn(EntityType.ARROW, BACK_SHALLOW_BODY);
        arrow.setDeltaMovement(Vec3.ZERO);
        arrow.setNoGravity(true);
        AABB chargeBox = charge.getBoundingBox();
        AABB arrowBox = arrow.getBoundingBox();
        PhysicalSensorSnapshot withProjectiles = FlySensors.sample(helper.getLevel(), fly);

        // Both targets must really be astride their rays, or either verdict would be vacuous.
        helper.assertTrue(
                RayEntityFence.straddlesDepthRay(origin, chargeBox),
                Component.literal("probe wind charge is not astride the forward ray line"));
        helper.assertTrue(
                RayEntityFence.straddlesDepthRay(origin, arrowBox),
                Component.literal("probe arrow is not astride the back ray line"));

        // The tag is the mechanism, so pin the tag and not just its consequence.
        helper.assertTrue(
                charge.getType().is(EntityTypeTags.REDIRECTABLE_PROJECTILE),
                Component.literal(
                        "minecraft:wind_charge is no longer in REDIRECTABLE_PROJECTILE; "
                                + "Projectile.isPickable() reads that tag, so ADR 0017's declared "
                                + "resolution has moved"));
        helper.assertFalse(
                arrow.getType().is(EntityTypeTags.REDIRECTABLE_PROJECTILE),
                Component.literal(
                        "minecraft:arrow entered REDIRECTABLE_PROJECTILE; the declared split "
                                + "between obstructing and non-obstructing projectiles has moved"));

        helper.assertTrue(
                FlySensors.obstructsSensorRay(charge),
                Component.literal("declared predicate refused a redirectable projectile"));
        helper.assertFalse(
                FlySensors.obstructsSensorRay(arrow),
                Component.literal("declared predicate admitted a non-redirectable projectile"));

        double expected = RayEntityFence.distanceSouthToNearFace(origin, chargeBox);
        helper.assertTrue(
                expected > 0.0 && expected < open.rayForwardBlocks(),
                Component.literal(
                        "probe wind charge is not strictly nearer than the open-arena forward "
                                + "reading " + open.rayForwardBlocks()
                                + ", so the probe would have no teeth"));
        helper.assertTrue(
                RayEntityFence.reads(withProjectiles.rayForwardBlocks(), expected),
                Component.literal(
                        "a redirectable projectile astride the forward ray reported "
                                + withProjectiles.rayForwardBlocks() + ", not its exact near face "
                                + expected + "; ADR 0017 accepts it as an obstruction"));
        double arrowDistance = RayEntityFence.distanceNorthToNearFace(origin, arrowBox);
        helper.assertTrue(
                arrowDistance > 0.0 && arrowDistance < open.rayBackBlocks(),
                Component.literal(
                        "probe arrow at " + arrowDistance + " is not strictly nearer than the "
                                + "open-arena back reading " + open.rayBackBlocks()
                                + ", so its exclusion would be vacuous"));
        helper.assertTrue(
                RayEntityFence.nonForwardRaysUnchanged(open, withProjectiles),
                Component.literal(
                        "an arrow astride the back ray moved a reading to "
                                + withProjectiles.rayBackBlocks()
                                + "; it is not in REDIRECTABLE_PROJECTILE and must not obstruct"));

        charge.discard();
        arrow.discard();
        helper.succeed();
    }

    /**
     * Regression pin for the block-only path. With no qualifying entity anywhere in the world, the
     * six readings must be exactly what the pre-ADR-0017 single {@code level.clip} produced: the
     * exact block-face distance forward and the declared maximum on the other five.
     */
    @GameTest(maxTicks = 20)
    public void blockOnlyReadingsAreUnchangedWhenNoEntityIsPresent(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, DEPTH_PROBE_SPAWN);
        fly.setYRot(0.0F);
        fly.setXRot(0.0F);
        Vec3 origin = fly.getBoundingBox().getCenter();
        PhysicalSensorSnapshot open = FlySensors.sample(helper.getLevel(), fly);

        helper.setBlock(FORWARD_FAR_BLOCK, Blocks.STONE);
        PhysicalSensorSnapshot blocked = FlySensors.sample(helper.getLevel(), fly);
        double expected =
                RayEntityFence.distanceSouthToNearFace(origin, helper.absolutePos(FORWARD_FAR_BLOCK));

        helper.assertTrue(
                RayEntityFence.reads(blocked.rayForwardBlocks(), expected),
                Component.literal(
                        "block-only forward reading moved to " + blocked.rayForwardBlocks()
                                + "; the exact face distance is " + expected));
        helper.assertTrue(
                RayEntityFence.nonForwardRaysUnchanged(open, blocked),
                Component.literal("the entity sweep perturbed an unobstructed ray"));
        helper.assertTrue(
                blocked.rayMaxDistanceBlocks() == FlySensors.RAY_LENGTH_BLOCKS,
                Component.literal("the declared ray length changed"));
        helper.succeed();
    }

    /** Probe body centred in the 8x8x8 test structure, clear of every wall by 3.5 blocks. */
    private static final Vec3 LATERAL_PROBE_SPAWN = new Vec3(4.5, 2.0, 4.5);

    /**
     * D-070 depth-probe body: centred in x and z, raised so a 1.975-block-tall armor stand standing
     * at y = 2.0 straddles the ray line at the fly's bounding-box centre (y = 3.0 + 0.22/2 = 3.11).
     */
    private static final Vec3 DEPTH_PROBE_SPAWN = new Vec3(4.5, 3.0, 4.5);

    /** Living body on the forward (+Z) ray, nearer than FORWARD_FAR_BLOCK. */
    private static final Vec3 FORWARD_NEAR_BODY = new Vec3(4.5, 2.0, 5.5);

    /**
     * Forward (+Z) probe position for targets too short to reach the ray line from the floor: a
     * second fly is 0.22 blocks tall and an item entity 0.25, against an armor stand's 1.975. These
     * spawn at the sensing body's own feet height so their hitboxes straddle the ray at y = 3.11.
     */
    private static final Vec3 FORWARD_SHALLOW_BODY = new Vec3(4.5, 3.0, 5.5);

    /**
     * Back (-Z) counterpart of FORWARD_SHALLOW_BODY: an arrow is 0.5 blocks tall, so spawning it at
     * the sensing body's own feet height puts its hitbox astride the ray line at y = 3.11. It sits
     * on the opposite ray from the wind charge so the two verdicts cannot mask each other.
     */
    private static final Vec3 BACK_SHALLOW_BODY = new Vec3(4.5, 3.0, 3.5);

    /** Block on the forward (+Z) ray, farther than FORWARD_NEAR_BODY. */
    private static final BlockPos FORWARD_FAR_BLOCK = new BlockPos(4, 3, 7);

    /** Living body on the back (-Z) ray, farther than BACK_NEAR_BLOCK. */
    private static final Vec3 BACK_FAR_BODY = new Vec3(4.5, 2.0, 1.0);

    /** Block on the back (-Z) ray, nearer than BACK_FAR_BODY. */
    private static final BlockPos BACK_NEAR_BLOCK = new BlockPos(4, 3, 2);

    /**
     * Floor separating "the body did not see itself" from "it did". The fly is 0.38 wide by 0.22
     * tall and the ray origin is its own bounding-box centre, so a self-hit reports at most 0.19.
     * The walled 8x8x8 structure reads 3.5 at its nearest from DEPTH_PROBE_SPAWN, so 1.0 sits an
     * order of magnitude clear of a self-hit and well below every real reading.
     */
    private static final double SELF_HIT_FLOOR_BLOCKS = 1.0;

    /** Obstacle 1.5 blocks west of the probe body, y/z aligned with the lateral rays. */
    private static final BlockPos WEST_OBSTACLE = new BlockPos(2, 2, 4);

    /** Obstacle 1.5 blocks east of the probe body, the mirror control of WEST_OBSTACLE. */
    private static final BlockPos EAST_OBSTACLE = new BlockPos(6, 2, 4);

    /** Probe body raised clear of the structure floor so both vertical rays are unobstructed. */
    private static final Vec3 VERTICAL_PROBE_SPAWN = new Vec3(4.5, 4.0, 4.5);

    /** Block in the body's own column, overhead; bottom face is the expected up-ray hit. */
    private static final BlockPos ABOVE_OBSTACLE = new BlockPos(4, 6, 4);

    /** Block in the body's own column, underfoot; top face is the expected down-ray hit. */
    private static final BlockPos BELOW_OBSTACLE = new BlockPos(4, 2, 4);

    /** Nose-up pitch for the tilt probe; negative is up in Minecraft's pitch convention. */
    private static final float PROBE_PITCH_DEGREES = -40.0F;

    /** Ceiling slab bounds, wide enough to catch the pitched ray's northward excursion. */
    private static final int CEILING_MIN = 1;

    private static final int CEILING_MAX = 7;

    private static String vector(Direction direction) {
        return "(" + direction.getStepX() + ", " + direction.getStepZ() + ")";
    }

    @GameTest(maxTicks = 20)
    public void actuationProducesCanonicalReceiptFrame(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 3.0, 1.5));
        long tick = helper.getLevel().getGameTime();
        ControllerIdentity identity = new ControllerIdentity(
                "gametest",
                0,
                fly.getStringUUID(),
                "1".repeat(64),
                "2".repeat(64),
                "3".repeat(64),
                "4".repeat(64),
                "5".repeat(64));
        RawPrimitiveAction requested = new RawPrimitiveAction(0.5, -0.25, 0.1, 0.2, -0.1);
        ActionFrame command = new ActionFrame(
                identity,
                0,
                0,
                tick,
                tick,
                tick + 2,
                0,
                1,
                "6".repeat(64),
                requested,
                false);

        PhysicalSensorSnapshot worldBefore = FlySensors.sample(helper.getLevel(), fly);
        ActuationReceipt applied = PrimitiveActuator.apply(fly, requested, requested.clamped());
        String frame = ControlProtocol.encodeActuationReceipt(
                identity,
                command,
                worldBefore,
                applied,
                ValidationResult.accepted(command));

        helper.assertTrue(
                frame.contains("\"type\":\"actuation_receipt\""),
                Component.literal("actuation receipt type was not emitted"));
        helper.assertTrue(
                frame.contains("\"validation\":\"APPLIED\""),
                Component.literal("receipt did not retain the applied validation state"));
        helper.assertTrue(
                frame.contains("\"requested_motors\":{\"forward_thrust\":0.5"),
                Component.literal("receipt did not retain requested primitive motors"));
        helper.assertTrue(
                frame.contains("\"world_before\":{\"position_m\":["),
                Component.literal("receipt did not retain authoritative pre-actuation world state"));
        helper.succeed();
    }

    // The default GameTest lane runs with no bridge configured, and the real WebSocket
    // bridge path is exercised only by the opt-in ExternalBridgeGameTest against a live
    // service, so a late frame cannot be injected through NeuralBridge here. This test
    // therefore drives the authoritative server-thread validator directly against real
    // level game ticks and real entity actuation, in the same component style as the
    // fence and receipt tests above; the full lateness-retirement decision table is
    // covered in CommandValidatorTest. CommandValidator.totalLatenessRetirements()
    // is process-global (reset only at SERVER_STOPPING): a future test in this
    // concurrently ticking batch must never assert it against an absolute value.
    @GameTest(maxTicks = 20)
    public void verifiedLateFrameRetiresWithoutKillingSessionAndFreshFrameActuates(
            GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 3.0, 1.5));
        fly.setYRot(0.0F);
        ControllerIdentity identity = new ControllerIdentity(
                "lateness-retirement-gametest",
                0,
                fly.getStringUUID(),
                "1".repeat(64),
                "2".repeat(64),
                "3".repeat(64),
                "4".repeat(64),
                "5".repeat(64));
        CommandValidator validator = new CommandValidator(identity);
        Map<Long, Long> receipts = new HashMap<>();
        LongFunction<OptionalLong> receiptLookup = sequence -> {
            Long tick = receipts.get(sequence);
            return tick == null ? OptionalLong.empty() : OptionalLong.of(tick);
        };
        long firstTick = helper.getLevel().getGameTime();
        receipts.put(0L, firstTick);

        ValidationResult first = validator.validate(
                List.of(sequencedFrame(identity, 0, firstTick, firstTick, 0.5)),
                firstTick,
                ControllerFault.ACTIVE,
                receiptLookup);
        helper.assertTrue(
                first.accepted(),
                Component.literal("fresh frame was not applied before the simulated burst"));
        PrimitiveActuator.apply(fly, first.rawAction(), first.appliedAction());

        helper.runAfterDelay(3, () -> {
            long burstTick = helper.getLevel().getGameTime();
            helper.assertTrue(
                    burstTick - firstTick > CommandValidator.MAX_COMMAND_AGE_TICKS,
                    Component.literal("burst delay did not exceed the command age bound"));
            receipts.put(1L, firstTick);
            receipts.put(2L, burstTick);
            double forwardBefore = fly.getDeltaMovement().z;
            ValidationResult afterBurst = validator.validate(
                    List.of(
                            sequencedFrame(identity, 1, firstTick, firstTick + 1, 0.5),
                            sequencedFrame(identity, 2, burstTick, burstTick, 0.75)),
                    burstTick,
                    ControllerFault.ACTIVE,
                    receiptLookup);
            helper.assertTrue(
                    validator.latenessRetirements() == 1,
                    Component.literal("verified late frame was not retired"));
            helper.assertFalse(
                    validator.state().isPaused(),
                    Component.literal("late frame latched a fault instead of retiring"));
            helper.assertTrue(
                    afterBurst.accepted(),
                    Component.literal("fresh frame after the retirement was not applied"));
            PrimitiveActuator.apply(fly, afterBurst.rawAction(), afterBurst.appliedAction());
            helper.assertTrue(
                    fly.getDeltaMovement().z > forwardBefore,
                    Component.literal("post-retirement fresh frame did not actuate the fly"));
            helper.succeed();
        });
    }

    // ---- ADR 0020 body programs: global pins -------------------------------------------------

    /**
     * The no-hidden-autonomy pin. With no operator command the fly must sit on the neural lane
     * forever: HOVER, source DEFAULT, authority NEURAL, and not one tick owned by a program.
     *
     * <p>A body program that decided when to run would be exactly the vanilla-mob-AI-shaped thing
     * {@code projectOwnedEntityHasNoMobOrNavigationInheritance} exists to forbid, wearing a
     * different name.
     */
    @GameTest(maxTicks = 60)
    public void noProgramRunsWithoutASelection(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 3.0, 1.5));
        Vec3 initial = fly.position();

        helper.runAfterDelay(45, () -> {
            BodyProgramRuntime runtime = fly.bodyPrograms();
            helper.assertTrue(
                    runtime.authority() == MotorAuthority.NEURAL,
                    Component.literal("a program took motor authority without a selection"));
            helper.assertTrue(
                    runtime.selectionSource() == SelectionSource.DEFAULT,
                    Component.literal("selection source moved without a selection"));
            helper.assertTrue(
                    runtime.state().id() == BodyProgramId.HOVER,
                    Component.literal("the default program is not HOVER"));
            helper.assertTrue(
                    runtime.programOwnedTicks() == 0,
                    Component.literal("a program owned a tick without a selection"));
            helper.assertTrue(
                    runtime.transitions() == 0,
                    Component.literal("a transition happened without a selection"));
            helper.assertTrue(
                    fly.bodyProgramChannel().equals(BodyProgramChannel.DEFAULT),
                    Component.literal("the published channel is not the declared default"));
            helper.assertTrue(
                    fly.position().distanceTo(initial) < 1.0e-9,
                    Component.literal("the unselected fly moved"));
            helper.succeed();
        });
    }

    /**
     * The inertness pin, as a **differential against `main`**.
     *
     * <p>{@code neurocraft/fly-tick-baseline-main.txt} was recorded by running an unmodified
     * `main` tree (418d902) in a scratch worktree and letting {@code FlyEntity.tick()} run for
     * sixteen authoritative ticks with nothing driven. This test reproduces that scenario exactly
     * on this branch and asserts the same values, tick by tick, so the arbitration branch, the
     * extra {@code processPendingProgramAuthority} call, the widened {@code recordActuation}
     * signature and the two new synchronized fields are proven to change nothing the world can
     * see.
     *
     * <p><strong>Ten of the thirteen words are compared bit-for-bit; the three position words are
     * compared to 1e-6.</strong> A GameTest structure is placed at a different absolute position
     * on every run — the observed runs sat near |x| = 3e6 — and {@code Entity.move} works in
     * absolute coordinates, so {@code getX() - origin.x} keeps only about 5e-10 of resolution and
     * its low bits are a function of where the batch happened to land, not of the code. Velocity,
     * rotation and the receipt are computed from small numbers and are placement-independent, so
     * they are held to exact bit patterns; that includes the grounded vertical-velocity clamp,
     * which is the interesting part of the second half of the scenario.
     *
     * <p><strong>What this does not pin, stated plainly.</strong> The GameTest lane runs with no
     * bridge configured, so {@code FlyControllerRuntime.control} always rejects and only the
     * neutral arm of {@code tick()} executes. The accepted arm cannot be driven here at all. Its
     * inertness rests on two other things instead: the expression that reaches the actuator is
     * textually unchanged from main ({@code PrimitiveActuator.apply(this, decision.rawAction(),
     * decision.appliedAction())}, with the published action still {@code decision.appliedAction()}),
     * and {@code advance} is asserted below to decline every tick under all four
     * (admitted, neutral) combinations while authority is NEURAL.
     */
    @GameTest(maxTicks = 60)
    public void neuralAuthorityIsByteIdenticalToThePreProgramPath(GameTestHelper helper) {
        List<String> baseline = readBaseline(helper);
        helper.assertTrue(
                baseline.size() == BASELINE_TICKS,
                Component.literal("baseline fixture has " + baseline.size() + " ticks"));
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, BASELINE_SPAWN);
        Vec3 origin = helper.absoluteVec(BASELINE_SPAWN);
        for (int x = 0; x <= 2; x++) {
            for (int z = 0; z <= 2; z++) {
                helper.setBlock(new BlockPos(x, 1, z), Blocks.STONE);
            }
        }
        fly.resetForFixture(origin.x, origin.y, origin.z, BASELINE_YAW);
        fly.setXRot(BASELINE_PITCH);
        fly.setDeltaMovement(BASELINE_VELOCITY);
        BodyProgramRuntime runtime = fly.bodyPrograms();
        int[] compared = {0};

        helper.onEachTick(() -> {
            int tick = compared[0];
            if (tick >= BASELINE_TICKS) {
                helper.assertTrue(
                        runtime.programOwnedTicks() == 0 && runtime.transitions() == 0,
                        Component.literal("the arbitration branch mutated state while NEURAL"));
                helper.succeed();
                return;
            }
            assertMatchesBaseline(helper, tick, baseline.get(tick), authoritativeWords(fly, origin));
            PhysicalSensorSnapshot snapshot = FlySensors.sample(helper.getLevel(), fly);
            for (boolean admitted : new boolean[] {true, false}) {
                for (boolean neutral : new boolean[] {true, false}) {
                    helper.assertTrue(
                            runtime.advance(snapshot, admitted, neutral).isEmpty(),
                            Component.literal("the arbitration branch claimed a NEURAL tick"));
                }
            }
            compared[0]++;
        });
    }

    /**
     * Scripted authority is reachable only through an accepted PAUSE acknowledgement, and every
     * other outcome of that transaction leaves the fly on the neural lane.
     *
     * <p>This drives {@link StagedProgramAuthority} directly for the same reason
     * {@code verifiedLateFrameRetiresWithoutKillingSessionAndFreshFrameActuates} drives the
     * validator directly: the default GameTest lane runs with no bridge configured, so an
     * acknowledgement cannot be injected through the WebSocket path.
     *
     * <p>The accepted acknowledgement below is stamped the way the service stamps it -- with the
     * tick of the observation it first answered paused -- and the transaction engages one action
     * budget later, at {@link StagedProgramAuthority#engageTick()}, which is the tick the paused
     * service's first neutral command governs. <strong>The sequence this pin feeds
     * is impossible live</strong>: an acknowledgement stamped with tick {@code X} is derived from
     * the observation the entity sends <em>after</em> its tick-{@code X} acknowledgement poll, so
     * no entity tick can ever hold an acknowledgement on its own effective tick. It is kept here
     * only to pin that the transaction refuses to engage on it (WAITING, and {@code apply}
     * throws). An earlier version of this pin expected READY at that point; it passed while every
     * live PAUSE failed {@code MISSED/LATE_ACK}. The real relation -- the acknowledgement first
     * visible at {@code effectiveTick + 1}, engaging at {@code effectiveTick +
     * ACTION_TARGET_OFFSET_TICKS} -- is carried by {@code StagedProgramAuthorityTest} (unit) and
     * by the opt-in {@code ExternalBridgeProgramAuthorityGameTest} against a real service.
     */
    @GameTest(maxTicks = 20)
    public void scriptedAuthorityRequiresAnAcceptedPause(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 3.0, 1.5));
        BodyProgramRuntime runtime = fly.bodyPrograms();
        long issuedTick = helper.getLevel().getGameTime();
        PhysicalSensorSnapshot snapshot = FlySensors.sample(helper.getLevel(), fly);

        StagedProgramAuthority unacknowledged = StagedProgramAuthority.engage(
                fly.getUUID(), issuedTick, BodyProgramId.HOVER, Modulation.NEUTRAL);
        unacknowledged.bindControlSequence(0);
        helper.assertTrue(
                unacknowledged.advance(issuedTick, Optional.empty())
                        == StagedProgramAuthority.Outcome.WAITING,
                Component.literal("authority advanced before its acknowledgement"));
        helper.assertTrue(
                runtime.authority() == MotorAuthority.NEURAL,
                Component.literal("authority engaged before an accepted PAUSE"));
        helper.assertTrue(
                runtime.advance(snapshot, true, true).isEmpty(),
                Component.literal("a program ran before an accepted PAUSE"));
        boolean refusedEarlyApply = false;
        try {
            unacknowledged.apply(runtime, issuedTick);
        } catch (IllegalStateException expected) {
            refusedEarlyApply = true;
        }
        helper.assertTrue(
                refusedEarlyApply,
                Component.literal("an unacknowledged authority handover applied anyway"));

        StagedProgramAuthority rejected = StagedProgramAuthority.engage(
                fly.getUUID(), issuedTick, BodyProgramId.HOVER, Modulation.NEUTRAL);
        rejected.bindControlSequence(1);
        helper.assertTrue(
                rejected.advance(
                                issuedTick,
                                Optional.of(controlAck(
                                        fly,
                                        1,
                                        issuedTick,
                                        issuedTick,
                                        ControlOperation.PAUSE,
                                        false)))
                        == StagedProgramAuthority.Outcome.REJECTED,
                Component.literal("a rejected PAUSE did not surface its terminal outcome"));
        helper.assertTrue(
                runtime.authority() == MotorAuthority.NEURAL,
                Component.literal("a rejected PAUSE still handed over motor authority"));

        StagedProgramAuthority accepted = StagedProgramAuthority.engage(
                fly.getUUID(), issuedTick, BodyProgramId.HOVER, Modulation.NEUTRAL);
        accepted.bindControlSequence(2);
        helper.assertTrue(
                accepted.advance(
                                issuedTick,
                                Optional.of(controlAck(
                                        fly,
                                        2,
                                        issuedTick,
                                        issuedTick,
                                        ControlOperation.PAUSE,
                                        true)))
                        == StagedProgramAuthority.Outcome.WAITING,
                Component.literal("an accepted PAUSE engaged on its own effective tick, which the"
                        + " entity can never observe"));
        boolean refusedEffectiveTickApply = false;
        try {
            accepted.apply(runtime, issuedTick);
        } catch (IllegalStateException expected) {
            refusedEffectiveTickApply = true;
        }
        helper.assertTrue(
                refusedEffectiveTickApply,
                Component.literal("an accepted PAUSE applied before its engage tick"));
        long engageTick = accepted.engageTick();
        helper.assertTrue(
                engageTick == issuedTick + CommandValidator.ACTION_TARGET_OFFSET_TICKS,
                Component.literal("engage tick is not one action budget past the acknowledged"
                        + " effective tick, so it is not the tick the paused service's first"
                        + " neutral command governs"));
        helper.assertTrue(
                accepted.advance(issuedTick + 1, Optional.empty())
                        == StagedProgramAuthority.Outcome.WAITING,
                Component.literal("an accepted PAUSE engaged before the first paused command"));
        helper.assertTrue(
                accepted.advance(engageTick, Optional.empty())
                        == StagedProgramAuthority.Outcome.READY,
                Component.literal("an accepted PAUSE did not become ready at its engage tick"));
        accepted.apply(runtime, engageTick);
        helper.assertTrue(
                runtime.authority() == MotorAuthority.SCRIPTED
                        && runtime.selectionSource() == SelectionSource.OPERATOR
                        && runtime.state().id() == BodyProgramId.HOVER,
                Component.literal("an accepted PAUSE did not engage the operator lane"));
        helper.assertTrue(
                runtime.advance(snapshot, true, true).isPresent(),
                Component.literal("the engaged program did not own its neutral tick"));
        helper.succeed();
    }

    /**
     * The connectome always wins. A non-neutral admitted command drops the program that instant,
     * the neural command is applied as normal, and the transition is logged {@code reason=preempted}.
     */
    @GameTest(maxTicks = 20)
    public void aNonNeutralCommandPreemptsAnyProgram(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 3.0, 1.5));
        BodyProgramRuntime runtime = fly.bodyPrograms();
        PhysicalSensorSnapshot snapshot = FlySensors.sample(helper.getLevel(), fly);
        runtime.engageScripted(
                BodyProgramId.HOVER,
                Modulation.NEUTRAL,
                SelectionSource.OPERATOR,
                snapshot.gameTick());

        helper.assertTrue(
                runtime.advance(snapshot, true, true).isPresent(),
                Component.literal("the engaged program did not own its neutral tick"));
        helper.assertTrue(
                runtime.programOwnedTicks() == 1,
                Component.literal("owned-tick accounting is wrong"));

        helper.assertTrue(
                runtime.advance(snapshot, true, false).isEmpty(),
                Component.literal("a program owned a tick carrying a non-neutral command"));
        helper.assertTrue(
                runtime.authority() == MotorAuthority.NEURAL
                        && runtime.selectionSource() == SelectionSource.DEFAULT
                        && runtime.state().id() == BodyProgramId.HOVER,
                Component.literal("preemption did not return the fly to the neural lane"));
        helper.assertTrue(
                runtime.programOwnedTicks() == 1,
                Component.literal("a preempted tick was counted as owned"));
        List<String> ledger = runtime.transitionLedger();
        helper.assertTrue(
                ledger.getLast().contains("reason=preempted"),
                Component.literal("preemption was not logged: " + ledger));
        helper.succeed();
    }

    /**
     * The snapshot gains typed rays, while historical identities retain the exact old wire shape.
     */
    @GameTest(maxTicks = 20)
    public void theObservationSchemaIsUnchanged(GameTestHelper helper) {
        List<String> components = Arrays.stream(PhysicalSensorSnapshot.class.getRecordComponents())
                .map(component -> component.getName())
                .toList();
        helper.assertTrue(
                components.equals(DECLARED_SNAPSHOT_COMPONENTS),
                Component.literal("the sensory vector's shape changed: " + components));

        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 3.0, 1.5));
        ControllerIdentity identity = new ControllerIdentity(
                "observation-schema-gametest",
                0,
                fly.getStringUUID(),
                "1".repeat(64),
                "2".repeat(64),
                "3".repeat(64),
                "4".repeat(64),
                "5".repeat(64));
        String frame = ControlProtocol.encodeObservation(
                identity, FlySensors.sample(helper.getLevel(), fly), 0, 0, 0);

        helper.assertTrue(
                objectKeys(frame, frame.indexOf('{')).equals(DECLARED_OBSERVATION_KEYS),
                Component.literal("the observation frame's key set changed: " + frame));
        helper.assertTrue(
                objectKeys(frame, frame.indexOf("\"sensors\":{") + "\"sensors\":".length())
                        .equals(DECLARED_SENSOR_KEYS),
                Component.literal("the sensor key set changed: " + frame));
        helper.succeed();
    }

    @GameTest(maxTicks = 20)
    public void typedRayKindsTravelThroughTheRealTickAndRecordingBridge(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, DEPTH_PROBE_SPAWN);
        fly.setYRot(0.0F);
        fly.setXRot(0.0F);
        helper.spawn(EntityType.ARMOR_STAND, FORWARD_NEAR_BODY);
        PhysicalSensorSnapshot expected = FlySensors.sample(helper.getLevel(), fly);
        helper.assertTrue(expected.rayHitKinds().getFirst() == 2,
                Component.literal("the entity stimulus does not intersect the forward ray"));
        ControllerIdentity old = ReadoutLaneFence.identity(fly);
        ControllerIdentity typed = new ControllerIdentity(old.runId(), old.sessionEpoch(),
                old.entityId(), old.graphHash(), old.adapterHash(), old.dynamicsHash(),
                old.decoderHash(), ControlProtocol.OBSERVATION_SCHEMA_V2_SHA256);
        try (RecordingBridgeFence.Installed installed = RecordingBridgeFence.install(fly, typed)) {
            fly.tick();
            List<String> frames = installed.bridge().observations();
            helper.assertTrue(!frames.isEmpty(), Component.literal("real tick emitted no observation"));
            String frame = frames.getLast();
            helper.assertTrue(frame.contains("\"sensor_schema\":\"neurocraft.sensors.v2\""),
                    Component.literal("typed sensor discriminator was lost: " + frame));
            String kinds = "\"ray_hit_kinds\":" + expected.rayHitKinds().toString().replace(" ", "");
            helper.assertTrue(frame.contains(kinds),
                    Component.literal("the six measured ray kinds did not reach the frame: " + frame));
        }
        helper.succeed();
    }

    // ---- ADR 0020 body programs: HOVER ---------------------------------------------------------

    /** HOVER is unconditional; all ten programs exist and missing-context FEED refuses on its first tick. */
    @GameTest(maxTicks = 20)
    public void hoverEntersUnconditionallyAndAllTenProgramsHaveConcreteImplementations(GameTestHelper helper) {
        helper.setBlock(new BlockPos(1, 1, 1), Blocks.STONE);
        FlyEntity grounded = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 2.0, 1.5));
        FlyEntity airborne = helper.spawn(NeuroCraftFly.FLY, new Vec3(3.5, 4.0, 3.5));
        HoverProgram hover = new HoverProgram();
        for (FlyEntity fly : List.of(grounded, airborne)) {
            helper.assertTrue(
                    hover.canEnter(bodyContext(helper, fly)),
                    Component.literal("HOVER refused a context; it is the universal fallback"));
        }

        BodyProgramRuntime runtime = airborne.bodyPrograms();
        long tick = helper.getLevel().getGameTime();
        runtime.engageScripted(
                BodyProgramId.FEED, Modulation.NEUTRAL, SelectionSource.OPERATOR, tick);
        runtime.advance(FlySensors.sample(helper.getLevel(), airborne), true, true);
        helper.assertTrue(
                runtime.state().id() == BodyProgramId.HOVER,
                Component.literal("FEED ran without ground or food context"));
        helper.assertTrue(
                runtime.transitionLedger().getLast().contains("reason=refused"),
                Component.literal("a refusal was not logged: " + runtime.transitionLedger()));
        helper.assertTrue(
                BodyProgramRuntime.registeredPrograms().size() == BodyProgramId.values().length
                        && BodyProgramRuntime.registeredPrograms().containsAll(List.of(BodyProgramId.values())),
                Component.literal("every declared program must now have a concrete implementation"));
        helper.succeed();
    }

    /** HOVER never self-exits: only a new selection can leave it. */
    @GameTest(maxTicks = 20)
    public void hoverNeverSelfExits(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 3.0, 1.5));
        PhysicalSensorSnapshot snapshot = FlySensors.sample(helper.getLevel(), fly);
        HoverProgram hover = new HoverProgram();
        BodyProgramState state = BodyProgramState.entering(BodyProgramId.HOVER, snapshot.gameTick());

        for (int tick = 0; tick < 400; tick++) {
            BodyProgramStep step = hover.step(
                    state,
                    bodyContext(helper, fly),
                    Modulation.NEUTRAL,
                    ProgramParameters.NEUTRAL);
            helper.assertTrue(
                    step.exitTo().isEmpty(),
                    Component.literal("HOVER named a successor at tick " + tick));
            helper.assertTrue(
                    step.next().id() == BodyProgramId.HOVER,
                    Component.literal("HOVER left itself at tick " + tick));
            state = step.next();
        }
        helper.assertTrue(
                state.ticksInProgram() == 400,
                Component.literal("HOVER's tick counter did not advance monotonically"));
        helper.succeed();
    }

    /**
     * HOVER stays inside its own declared envelope, and therefore inside the actuator rails, for
     * every modulation — including modulation values a caller asked for outside {@code [0, 1]}.
     * That last part is the whole of the "a modulation scalar can never widen a bound" guarantee.
     */
    @GameTest(maxTicks = 20)
    public void hoverStaysInsideTheActuationEnvelope(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 3.0, 1.5));
        fly.setYRot(37.0F);
        HoverProgram hover = new HoverProgram();
        List<Modulation> modulations = List.of(
                Modulation.NEUTRAL,
                new Modulation(0.0, 0.0, 0.0, 0.0),
                new Modulation(0.5, 0.5, 0.5, 0.5),
                new Modulation(5.0, 5.0, 5.0, 5.0),
                new Modulation(-3.0, -3.0, -3.0, -3.0));

        for (Modulation modulation : modulations) {
            fly.setDeltaMovement(0.31, 0.0, -0.24);
            BodyProgramState state =
                    BodyProgramState.entering(BodyProgramId.HOVER, helper.getLevel().getGameTime());
            for (int tick = 0; tick < HoverProgram.HOVER_BOB_PERIOD_TICKS * 2; tick++) {
                BodyProgramStep step = hover.step(
                        state,
                        bodyContext(helper, fly),
                        modulation,
                        ProgramParameters.NEUTRAL);
                PrimitiveAction motors = step.motors();
                String where = " at tick " + tick + " under " + modulation;
                assertUnitRange(helper, motors.forwardThrust(), "forward" + where);
                assertUnitRange(helper, motors.lateralThrust(), "lateral" + where);
                assertUnitRange(helper, motors.verticalThrust(), "vertical" + where);
                assertUnitRange(helper, motors.yawTorque(), "yaw" + where);
                assertUnitRange(helper, motors.pitchTorque(), "pitch" + where);
                helper.assertTrue(
                        Math.abs(motors.verticalThrust())
                                <= HoverProgram.HOVER_BOB_AMPLITUDE + 1.0e-12,
                        Component.literal("HOVER exceeded its declared bob amplitude" + where));
                helper.assertTrue(
                        Math.hypot(motors.forwardThrust(), motors.lateralThrust())
                                <= HoverProgram.HOVER_SETTLE_THRUST + 1.0e-12,
                        Component.literal("HOVER exceeded its declared settle thrust" + where));
                helper.assertTrue(
                        motors.yawTorque() == 0.0 && motors.pitchTorque() == 0.0,
                        Component.literal("HOVER commanded a torque" + where));

                float yawBefore = fly.getYRot();
                PrimitiveActuator.apply(fly, RawPrimitiveAction.ZERO, motors);
                helper.assertTrue(
                        fly.getDeltaMovement().length()
                                <= PrimitiveActuator.MAX_SPEED_BLOCKS_PER_TICK + 1.0e-9,
                        Component.literal("HOVER drove the body past the speed cap" + where));
                helper.assertTrue(
                        Math.abs(fly.getYRot() - yawBefore)
                                <= PrimitiveActuator.MAX_YAW_DELTA_DEGREES_PER_TICK + 1.0e-6,
                        Component.literal("HOVER turned faster than the yaw rate" + where));
                state = step.next();
            }
        }
        helper.succeed();
    }

    /**
     * HOVER is a pure function of state, context, and modulation, and it reads only
     * {@code amplitude}: varying {@code speed}, {@code turnRate}, or {@code urgency} produces a
     * bit-identical motor sequence.
     */
    @GameTest(maxTicks = 20)
    public void hoverIsDeterministic(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 3.0, 1.5));
        fly.setYRot(-118.0F);
        fly.setDeltaMovement(0.05, 0.0, 0.07);
        int ticks = HoverProgram.HOVER_BOB_PERIOD_TICKS * 3;

        long[] first = hoverMotorWords(helper, fly, Modulation.NEUTRAL, ticks);
        long[] second = hoverMotorWords(helper, fly, Modulation.NEUTRAL, ticks);
        helper.assertTrue(
                Arrays.equals(first, second),
                Component.literal("HOVER is not deterministic across identical runs"));

        long[] unreadScalars = hoverMotorWords(
                helper, fly, new Modulation(0.0, 0.25, 1.0, 1.0), ticks);
        helper.assertTrue(
                Arrays.equals(first, unreadScalars),
                Component.literal("HOVER read a modulation scalar it does not declare"));

        long[] readScalar = hoverMotorWords(
                helper, fly, new Modulation(1.0, 1.0, 0.5, 0.0), ticks);
        helper.assertFalse(
                Arrays.equals(first, readScalar),
                Component.literal("HOVER ignored the amplitude scalar it does declare"));
        helper.succeed();
    }

    /**
     * The bob is <strong>actuation</strong>: it moves the authoritative body. So it must return
     * that body to where it found it, and this is the pin that says so.
     *
     * <p>It is also the pin that found the defect. A plain {@code sin(2*pi*t/40)} bob leaves the
     * vertical velocity filter with a nonzero steady-state offset at entry, and the position
     * integral of that transient climbs about 0.83 blocks over the first period — a hovering fly
     * that ascends forever. {@code HoverProgram} therefore enters at the phase whose steady-state
     * velocity is zero, derived from the declared vertical drag. The bound is 1e-6 rather than the
     * originally proposed 0.01 because a 0.01-radian phase error passes at 0.01 and the derived
     * phase measures below 1e-12; the pin has to be tight enough to defend the derivation.
     */
    @GameTest(maxTicks = 90)
    public void hoverBobIsZeroMeanOverOnePeriod(GameTestHelper helper) {
        FlyEntity fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(1.5, 4.0, 1.5));
        BodyProgramRuntime runtime = fly.bodyPrograms();
        runtime.engageScripted(
                BodyProgramId.HOVER,
                Modulation.NEUTRAL,
                SelectionSource.OPERATOR,
                helper.getLevel().getGameTime());
        int[] driven = {0};
        double[] entryY = {Double.NaN};

        helper.onEachTick(() -> {
            int tick = driven[0];
            if (tick == 0) {
                entryY[0] = fly.getY();
            }
            if (tick == HoverProgram.HOVER_BOB_PERIOD_TICKS) {
                double drift = Math.abs(fly.getY() - entryY[0]);
                helper.assertTrue(
                        drift < 1.0e-6,
                        Component.literal("the hover bob is not zero-mean: drift " + drift));
                helper.assertTrue(
                        runtime.programOwnedTicks() == HoverProgram.HOVER_BOB_PERIOD_TICKS,
                        Component.literal("the bob did not own every tick of its period"));
                helper.succeed();
                return;
            }
            PhysicalSensorSnapshot snapshot = FlySensors.sample(helper.getLevel(), fly);
            Optional<BodyProgramStep> step = runtime.advance(snapshot, true, true);
            helper.assertTrue(
                    step.isPresent(),
                    Component.literal("the engaged program did not own tick " + tick));
            PrimitiveActuator.apply(fly, RawPrimitiveAction.ZERO, step.orElseThrow().motors());
            driven[0]++;
        });
    }

    /** Bit-comparable words per recorded tick: position, velocity, rotation, receipt. */
    private static final int STATE_WORDS = 13;

    private static void seedComparablePhase(FlyEntity fly, Vec3 start) {
        fly.resetForFixture(start.x, start.y, start.z, 0.0F);
        fly.setDeltaMovement(0.08, 0.03, -0.05);
    }

    private static void recordAuthoritativeState(FlyEntity fly, long[] sink, int index) {
        ActuationReceipt receipt = fly.lastActuation();
        long[] words = {
            Double.doubleToLongBits(fly.getX()),
            Double.doubleToLongBits(fly.getY()),
            Double.doubleToLongBits(fly.getZ()),
            Double.doubleToLongBits(fly.getDeltaMovement().x),
            Double.doubleToLongBits(fly.getDeltaMovement().y),
            Double.doubleToLongBits(fly.getDeltaMovement().z),
            Float.floatToIntBits(fly.getYRot()),
            Float.floatToIntBits(fly.getXRot()),
            Double.doubleToLongBits(receipt.velocityAfterX()),
            Double.doubleToLongBits(receipt.velocityAfterY()),
            Double.doubleToLongBits(receipt.velocityAfterZ()),
            Float.floatToIntBits(receipt.yawAfterDegrees()),
            Float.floatToIntBits(receipt.pitchAfterDegrees()),
        };
        System.arraycopy(words, 0, sink, index * STATE_WORDS, STATE_WORDS);
    }

    private static ActionFrame action(
            ControllerIdentity identity, long commandSequence, long observationTick) {
        return new ActionFrame(
                identity,
                commandSequence,
                commandSequence,
                observationTick,
                observationTick + 1,
                observationTick + 2,
                commandSequence,
                commandSequence + 1,
                "6".repeat(64),
                RawPrimitiveAction.ZERO,
                true);
    }

    private static ActionFrame sequencedFrame(
            ControllerIdentity identity,
            long commandSequence,
            long observationTick,
            long targetTick,
            double forwardThrust) {
        return new ActionFrame(
                identity,
                commandSequence,
                commandSequence,
                observationTick,
                targetTick,
                observationTick + 2,
                commandSequence * 5,
                commandSequence * 5 + 5,
                "6".repeat(64),
                new RawPrimitiveAction(forwardThrust, 0, 0, 0, 0),
                false);
    }

    private static ActionFrame actionWithTargetTick(
            ControllerIdentity identity, long commandSequence, long targetTick) {
        return new ActionFrame(
                identity,
                commandSequence,
                commandSequence,
                targetTick - 1,
                targetTick,
                targetTick + 1,
                commandSequence,
                commandSequence + 1,
                "6".repeat(64),
                RawPrimitiveAction.ZERO,
                true);
    }

    private static ControlAckFrame resetAck(
            FlyEntity fly, long issuedTick, long effectiveTick, boolean accepted) {
        ControllerIdentity identity = new ControllerIdentity(
                "staged-reset-gametest",
                0,
                fly.getStringUUID(),
                "1".repeat(64),
                "2".repeat(64),
                "3".repeat(64),
                "4".repeat(64),
                "5".repeat(64));
        return new ControlAckFrame(
                identity,
                0,
                issuedTick,
                ControlOperation.RESET,
                null,
                null,
                accepted,
                ControlState.RECORDING,
                effectiveTick,
                accepted ? null : "INVALID_STATE");
    }

    /** Component names of the sensory vector the Python encoder consumes, frozen order. */
    private static final List<String> DECLARED_SNAPSHOT_COMPONENTS = List.of(
            "gameTick",
            "entityId",
            "xBlocks",
            "yBlocks",
            "zBlocks",
            "velocityXBlocksPerTick",
            "velocityYBlocksPerTick",
            "velocityZBlocksPerTick",
            "yawDegrees",
            "pitchDegrees",
            "horizontalCollision",
            "verticalCollision",
            "onGround",
            "inWater",
            "inLava",
            "blockLight",
            "skyLight",
            "rayMaxDistanceBlocks",
            "rayForwardBlocks",
            "rayBackBlocks",
            "rayLeftBlocks",
            "rayRightBlocks",
            "rayUpBlocks",
            "rayDownBlocks",
            "rayHitKinds",
            "rayActorKinds",
            "events");

    private static final List<String> DECLARED_OBSERVATION_KEYS = List.of(
            "type",
            "protocol",
            "identity",
            "observation_seq",
            "game_tick",
            "sent_monotonic_ns",
            "bridge_observations_coalesced",
            "sensors",
            "missing_fields",
            "clamped_fields");

    private static final List<String> DECLARED_SENSOR_KEYS = List.of(
            "position_m",
            "velocity_m_s",
            "yaw_rad",
            "pitch_rad",
            "on_ground",
            "horizontal_collision",
            "vertical_collision",
            "in_fluid",
            "local_light",
            "ray_max_distance_m",
            "ray_distances_m");

    /** Ticks in the committed main-tree baseline (neurocraft/fly-tick-baseline-main.txt). */
    private static final int BASELINE_TICKS = 16;

    private static final Vec3 BASELINE_SPAWN = new Vec3(1.5, 3.5, 1.5);
    private static final Vec3 BASELINE_VELOCITY = new Vec3(0.05, -0.21, -0.03);
    private static final float BASELINE_YAW = 41.0F;
    private static final float BASELINE_PITCH = -17.0F;
    private static final String BASELINE_RESOURCE = "/neurocraft/fly-tick-baseline-main.txt";

    private static List<String> readBaseline(GameTestHelper helper) {
        try (InputStream stream = FlyEntityGameTests.class.getResourceAsStream(BASELINE_RESOURCE)) {
            helper.assertTrue(
                    stream != null,
                    Component.literal("missing baseline fixture " + BASELINE_RESOURCE));
            String text = new String(
                    Objects.requireNonNull(stream).readAllBytes(), StandardCharsets.UTF_8);
            return text.lines()
                    .map(String::strip)
                    .filter(line -> !line.isEmpty() && !line.startsWith("#"))
                    .toList();
        } catch (IOException exception) {
            throw new IllegalStateException("could not read " + BASELINE_RESOURCE, exception);
        }
    }

    /**
     * The same thirteen raw bit patterns the main-tree recorder emitted, in the same order.
     * Position is relative to the spawn because a GameTest structure's absolute origin moves.
     */
    private static String authoritativeWords(FlyEntity fly, Vec3 origin) {
        ActuationReceipt receipt = fly.lastActuation();
        long[] words = {
            Double.doubleToRawLongBits(fly.getX() - origin.x),
            Double.doubleToRawLongBits(fly.getY() - origin.y),
            Double.doubleToRawLongBits(fly.getZ() - origin.z),
            Double.doubleToRawLongBits(fly.getDeltaMovement().x),
            Double.doubleToRawLongBits(fly.getDeltaMovement().y),
            Double.doubleToRawLongBits(fly.getDeltaMovement().z),
            Float.floatToRawIntBits(fly.getYRot()),
            Float.floatToRawIntBits(fly.getXRot()),
            Double.doubleToRawLongBits(receipt.velocityAfterX()),
            Double.doubleToRawLongBits(receipt.velocityAfterY()),
            Double.doubleToRawLongBits(receipt.velocityAfterZ()),
            Float.floatToRawIntBits(receipt.yawAfterDegrees()),
            Float.floatToRawIntBits(receipt.pitchAfterDegrees()),
        };
        StringBuilder line = new StringBuilder();
        for (long word : words) {
            if (line.length() > 0) {
                line.append(' ');
            }
            line.append(String.format(Locale.ROOT, "%016x", word));
        }
        return line.toString();
    }

    /** Position words are placement-quantized; everything else must match bit-for-bit. */
    private static final int BASELINE_POSITION_WORDS = 3;

    /** Bound on the position comparison: ~2000x the observed placement quantization. */
    private static final double BASELINE_POSITION_TOLERANCE_BLOCKS = 1.0e-6;

    private static void assertMatchesBaseline(
            GameTestHelper helper, int tick, String expected, String actual) {
        String[] want = expected.split(" ");
        String[] got = actual.split(" ");
        helper.assertTrue(
                want.length == got.length,
                Component.literal("baseline row " + tick + " has the wrong width"));
        String context = "\n  main:   " + expected + "\n  branch: " + actual;
        for (int index = 0; index < want.length; index++) {
            if (index < BASELINE_POSITION_WORDS) {
                double expectedValue = Double.longBitsToDouble(Long.parseUnsignedLong(want[index], 16));
                double actualValue = Double.longBitsToDouble(Long.parseUnsignedLong(got[index], 16));
                helper.assertTrue(
                        Math.abs(expectedValue - actualValue)
                                < BASELINE_POSITION_TOLERANCE_BLOCKS,
                        Component.literal("tick " + tick + " position word " + index
                                + " diverged from main's FlyEntity.tick()" + context));
                continue;
            }
            helper.assertTrue(
                    want[index].equals(got[index]),
                    Component.literal("tick " + tick + " word " + index
                            + " diverged from main's FlyEntity.tick()" + context));
        }
    }

    private static BodyProgramContext bodyContext(GameTestHelper helper, FlyEntity fly) {
        return BodyProgramContext.sensorsOnly(FlySensors.sample(helper.getLevel(), fly));
    }

    private static void assertUnitRange(GameTestHelper helper, double value, String label) {
        helper.assertTrue(
                value >= -1.0 && value <= 1.0,
                Component.literal("motor left the rails: " + label + " = " + value));
    }

    /** Raw bits of HOVER's five motor channels over {@code ticks} steps from a fixed context. */
    private static long[] hoverMotorWords(
            GameTestHelper helper, FlyEntity fly, Modulation modulation, int ticks) {
        HoverProgram hover = new HoverProgram();
        BodyProgramContext context = bodyContext(helper, fly);
        BodyProgramState state = BodyProgramState.entering(BodyProgramId.HOVER, 0L);
        long[] words = new long[ticks * 5];
        for (int tick = 0; tick < ticks; tick++) {
            BodyProgramStep step =
                    hover.step(state, context, modulation, ProgramParameters.NEUTRAL);
            PrimitiveAction motors = step.motors();
            words[tick * 5] = Double.doubleToRawLongBits(motors.forwardThrust());
            words[tick * 5 + 1] = Double.doubleToRawLongBits(motors.lateralThrust());
            words[tick * 5 + 2] = Double.doubleToRawLongBits(motors.verticalThrust());
            words[tick * 5 + 3] = Double.doubleToRawLongBits(motors.yawTorque());
            words[tick * 5 + 4] = Double.doubleToRawLongBits(motors.pitchTorque());
            state = step.next();
        }
        return words;
    }

    private static ControlAckFrame controlAck(
            FlyEntity fly,
            long controlSequence,
            long issuedTick,
            long effectiveTick,
            ControlOperation operation,
            boolean accepted) {
        ControllerIdentity identity = new ControllerIdentity(
                "program-authority-gametest",
                0,
                fly.getStringUUID(),
                "1".repeat(64),
                "2".repeat(64),
                "3".repeat(64),
                "4".repeat(64),
                "5".repeat(64));
        return new ControlAckFrame(
                identity,
                controlSequence,
                issuedTick,
                operation,
                null,
                null,
                accepted,
                accepted ? ControlState.PAUSED : ControlState.RUNNING,
                effectiveTick,
                accepted ? null : "INVALID_STATE");
    }

    /**
     * Top-level keys of the JSON object beginning at {@code start}, in emitted order. The encoder
     * writes canonical JSON with no whitespace, so a hand-rolled walk is exact and needs no parser
     * on the GameTest classpath. Nested objects and arrays are skipped by depth.
     */
    private static List<String> objectKeys(String json, int start) {
        List<String> keys = new ArrayList<>();
        int depth = 0;
        int index = start;
        boolean expectKey = false;
        while (index < json.length()) {
            char character = json.charAt(index);
            if (character == '"') {
                int end = index + 1;
                while (json.charAt(end) != '"' || json.charAt(end - 1) == '\\') {
                    end++;
                }
                if (depth == 1 && expectKey) {
                    keys.add(json.substring(index + 1, end));
                    expectKey = false;
                }
                index = end + 1;
                continue;
            }
            if (character == '{' || character == '[') {
                depth++;
                expectKey = character == '{' && depth == 1;
            } else if (character == '}' || character == ']') {
                depth--;
                if (depth == 0) {
                    break;
                }
            } else if (character == ',' && depth == 1) {
                expectKey = true;
            }
            index++;
        }
        return List.copyOf(keys);
    }
}
