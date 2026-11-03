package dev.neurocraft.fly.gametest;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import dev.neurocraft.fly.FlyControllerRuntime;
import dev.neurocraft.fly.NeuroCraftFly;
import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ProgramExecution;
import dev.neurocraft.fly.body.ProgramParameters;
import dev.neurocraft.fly.body.ReadoutValues;
import dev.neurocraft.fly.control.ControlProtocol;
import dev.neurocraft.fly.control.ControllerFault;
import dev.neurocraft.fly.control.ControllerIdentity;
import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import dev.neurocraft.fly.control.PrimitiveAction;
import dev.neurocraft.fly.control.ProgramActionFrame;
import dev.neurocraft.fly.control.RawPrimitiveAction;
import dev.neurocraft.fly.control.ValidationResult;
import dev.neurocraft.fly.entity.FlyEntity;
import dev.neurocraft.fly.entity.FlySensors;
import dev.neurocraft.fly.entity.PrimitiveActuator;
import dev.neurocraft.fly.trial.MatchedTrialLedger;
import dev.neurocraft.fly.trial.MatchedTrialPlan;
import dev.neurocraft.fly.trial.MatchedTrialRunner;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import net.fabricmc.fabric.api.gametest.v1.GameTest;
import net.minecraft.core.BlockPos;
import net.minecraft.gametest.framework.GameTestHelper;
import net.minecraft.network.chat.Component;
import net.minecraft.world.entity.animal.Cow;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.phys.Vec3;

/** Real world ticks, rays and body resets; explicitly synthetic commands and handover state.
 * Authentication is tested by ComparisonLaneSession/bridge tests, not simulated as live evidence here.
 */
public final class MatchedTrialGameTests {
    private static final long BASE_EPOCH = 9007199254740999L;
    private static final ProgramParameters PARAMETERS = new ProgramParameters(0);
    private static final ProgramExecution HOVER = new ProgramExecution(BodyProgramId.HOVER,
            Modulation.NEUTRAL, PARAMETERS, PrimitiveAction.NEUTRAL, Optional.empty());

    @GameTest(maxTicks = 370)
    public void realEntityAndBlockFrontFacesFollowWorldTicksAndRestoreOnlyAfterActive(GameTestHelper helper) throws Exception {
        Harness harness = new Harness(helper, List.of(cell(MatchedTrialPlan.Stimulus.ENTITY), cell(MatchedTrialPlan.Stimulus.BLOCK)));
        int[] approachSamples = {0};
        long[] restorationSeen = {-1};
        helper.onEachTick(() -> {
            try {
                harness.runner.beforeObservation();
                if (harness.handover.begins > 2) {
                    check(helper, harness.fly.bodyPrograms().trialWanderSeed().isEmpty(), "ordinary reset must actually clear the seed");
                    if (restorationSeen[0] < 0) { restorationSeen[0] = helper.getLevel().getGameTime(); }
                    if (helper.getLevel().getGameTime() - restorationSeen[0] < 3) {
                        check(helper, harness.runner.active(), "restoration was claimed before ACTIVE");
                        return;
                    }
                    harness.handover.state = ControllerFault.ACTIVE;
                    harness.runner.beforeObservation();
                    check(helper, !harness.runner.active() && harness.runner.status().equals("complete; original restored"),
                            "exact ACTIVE restoration did not complete: " + harness.runner.status());
                    var rows = harness.rows();
                    JsonObject header = rows.getFirst();
                    check(helper, MatchedTrialPlan.sha256(header.get("plan_json").getAsString())
                                    .equals(header.get("plan_sha256").getAsString()), "exact serialized plan hash changed");
                    check(helper, JsonParser.parseString(header.get("plan_json").getAsString()).equals(header.get("plan")),
                            "human-readable plan and hashed bytes differ");
                    check(helper, count(rows, "trial_end") == 2 && count(rows, "original_restored") == 1,
                            "ledger must contain two results and exactly one confirmed restoration");
                    check(helper, approachSamples[0] == 120, "expected all 60 approach samples for each stimulus");
                    for (var end : rows.stream().filter(r -> kind(r).equals("trial_end")).toList()) {
                        check(helper, end.get("outcome").getAsString().equals("complete"), "a real stimulus ray was not attributed");
                    }
                    var ticks = rows.stream().filter(r -> kind(r).equals("tick") && !r.get("anchor_offset").isJsonNull()).toList();
                    check(helper, ticks.size() == 320, "each anchored trial must have exactly 160 tick rows");
                    check(helper, ticks.stream().filter(r -> r.get("anchor_offset").getAsInt() == 0).count() == 2,
                            "each trial requires its own command-0 anchor");
                    check(helper, ticks.getFirst().has("initial_observation_zero"), "anchor lost actual observation 0");
                    check(helper, ticks.getFirst().getAsJsonObject("identity").get("sessionEpoch").getAsLong() == BASE_EPOCH + 1,
                            "large epoch lost integer precision");
                    check(helper, ticks.getFirst().getAsJsonObject("program_receipt_offered").get("command_seq").getAsLong() == 0,
                            "applied anchor receipt was not retained");
                    harness.close(); helper.succeed(); return;
                }
                check(helper, harness.runner.active(), "runner failed before completing schedule: " + harness.runner.status());
                PhysicalSensorSnapshot sample = FlySensors.sample(helper.getLevel(), harness.fly);
                long sequence = sample.gameTick() - harness.handover.resetTick - 2;
                int front = MatchedTrialPlan.frontPlane(sequence);
                if (front >= 0) {
                    int expectedKind = harness.handover.begins == 1 ? 2 : 1;
                    check(helper, sample.rayHitKinds().getFirst() == expectedKind, "stimulus was not installed before the actual ray sample");
                    check(helper, Math.abs(sample.rayForwardBlocks() - (front - .5)) < 1.0e-6,
                            "entity/block front-face distances differ from the declared world plane");
                    approachSamples[0]++;
                    if (expectedKind == 2) {
                        var cows = helper.getLevel().getEntitiesOfClass(Cow.class, harness.fly.getBoundingBox().inflate(8));
                        check(helper, cows.size() == 1, "one persistent cow is required per trial");
                        // Deliberate stimulus drift after this observation: the next scheduled
                        // tick must restore its world plane, without resetting the fly.
                        cows.getFirst().setDeltaMovement(new Vec3(.2, 0, .2));
                    }
                }
                harness.observe(sample, sequence, 0);
            } catch (RuntimeException | Error failure) { harness.close(); throw failure; }
        });
    }

    @GameTest(maxTicks = 15)
    public void missingCommandZeroDoesNotReanchorToOne(GameTestHelper helper) throws Exception {
        Harness harness = new Harness(helper, List.of(cell(MatchedTrialPlan.Stimulus.ENTITY)));
        helper.onEachTick(() -> {
            try {
                harness.runner.beforeObservation();
                var sample = FlySensors.sample(helper.getLevel(), harness.fly);
                long sequence = sample.gameTick() - harness.handover.resetTick - 2;
                harness.observe(sample, sequence, 1);
                if (sequence >= 0) {
                    harness.runner.stop("test end after missing command 0");
                    var rows = harness.rows();
                    var result = rows.stream().filter(r -> kind(r).equals("trial_end")).findFirst().orElseThrow();
                    check(helper, result.get("outcome").getAsString().equals("invalid_anchor"), "missing command 0 was not invalidated");
                    check(helper, result.get("anchor_game_tick").isJsonNull(), "command 1 became an anchor");
                    check(helper, rows.stream().filter(r -> kind(r).equals("tick")).allMatch(r -> r.get("anchor_offset").isJsonNull()),
                            "an invalid trial received an anchored timeline");
                    check(helper, rows.indexOf(result) > rows.indexOf(rows.stream().filter(r -> r.has("command")
                                    && !r.get("command").isJsonNull()).findFirst().orElseThrow()),
                            "result preceded the causal command row");
                    harness.close(); helper.succeed();
                }
            } catch (RuntimeException | Error failure) { harness.close(); throw failure; }
        });
    }

    @GameTest(maxTicks = 20)
    public void handshakeDriftAndFalseAirborneResetAreRefused(GameTestHelper helper) throws Exception {
        try (Harness harness = new Harness(helper, List.of(cell(MatchedTrialPlan.Stimulus.ENTITY)))) {
            harness.runner.beforeObservation();
            harness.fly.setDeltaMovement(new Vec3(.01, 0, 0));
            harness.runner.beforeObservation();
            harness.runner.stop("inspect drift refusal");
            check(helper, harness.rows().stream().anyMatch(r -> kind(r).equals("trial_end")
                    && r.get("outcome").getAsString().equals("invalid_initial_state")), "handshake drift was silently accepted");
        }
        try (Harness harness = new Harness(helper, List.of(cell(MatchedTrialPlan.Stimulus.BLOCK)))) {
            // The reset clears movement flags, but a physical surface at the feet must still
            // refuse the declared two-block-airborne initial state.
            helper.setBlock(new BlockPos(3, 3, 0), Blocks.STONE);
            harness.handover.swallowResetFailure = true;
            harness.runner.beforeObservation();
            check(helper, !harness.runner.active(), "cleared flags invented airborne support");
            check(helper, harness.handover.begins == 1, "failed initial support must not retry");
            check(helper, count(harness.rows(), "trial_start") == 0, "swallowed reset failure produced verified trial_start");
            JsonObject end = harness.rows().getLast();
            check(helper, end.getAsJsonObject("observed_current_identity").get("sessionEpoch").getAsLong() == BASE_EPOCH + 1,
                    "consumed failed epoch disappeared from the diagnostic");
        }
        helper.succeed();
    }

    @GameTest(maxTicks = 20)
    public void terminalFaultAndWriterFailureStopWithoutAnotherEpoch(GameTestHelper helper) throws Exception {
        for (boolean writerFailure : List.of(false, true)) {
            try (Harness harness = new Harness(helper, List.of(cell(MatchedTrialPlan.Stimulus.ENTITY)))) {
                harness.runner.beforeObservation();
                if (writerFailure) {
                    JsonObject oversized = new JsonObject(); oversized.addProperty("bad", "x".repeat(32768));
                    try { harness.ledger.append(oversized); throw new AssertionError("oversized row accepted"); }
                    catch (IllegalStateException expected) { /* explicitly unavailable, no silent loss */ }
                } else { harness.handover.state = ControllerFault.DEADLINE_MISSED; }
                harness.runner.beforeObservation();
                check(helper, !harness.runner.active(), "terminal condition did not stop the plan");
                check(helper, harness.handover.begins == 1, "terminal condition started a recovery epoch");
                check(helper, harness.fly.bodyPrograms().trialWanderSeed().isPresent(), "stop hid its retained trial seed");
                check(helper, count(harness.rows(), "original_restoration_attempt") == 0, "stop silently restored a lane");
            }
        }
        helper.succeed();
    }

    @GameTest(maxTicks = 10)
    public void missingObservedWorldTickStopsInsteadOfCompletingAPartialSchedule(GameTestHelper helper) throws Exception {
        Harness harness = new Harness(helper, List.of(cell(MatchedTrialPlan.Stimulus.ENTITY)));
        int[] stage = {0};
        helper.onEachTick(() -> {
            try {
                if (stage[0]++ == 0) {
                    harness.runner.beforeObservation();
                    harness.observe(FlySensors.sample(helper.getLevel(), harness.fly), -1, 0);
                } else if (stage[0] > 2) {
                    // One actual world tick passed without a runner observation, as when an
                    // entity stops ticking. World time itself is never altered by this test.
                    harness.runner.beforeObservation();
                    check(helper, !harness.runner.active() && harness.runner.status().contains("skipped or duplicate"),
                            "an observed-tick gap was allowed to label a partial schedule complete");
                    check(helper, harness.handover.begins == 1 && count(harness.rows(), "trial_end") == 0,
                            "a gap produced a completed trial or a new epoch");
                    harness.close(); helper.succeed();
                }
            } catch (RuntimeException | Error failure) { harness.close(); throw failure; }
        });
    }

    @GameTest(maxTicks = 20)
    public void entityHookResetsBeforeRealBridgeObservationAndRecordsAfterPhysics(GameTestHelper helper) throws Exception {
        try (Harness harness = new Harness(helper, List.of(cell(MatchedTrialPlan.Stimulus.ENTITY)))) {
            Field runnerField = FlyControllerRuntime.class.getDeclaredField("matchedTrials");
            runnerField.setAccessible(true);
            Object previous = runnerField.get(null);
            try (var installed = RecordingBridgeFence.install(harness.fly, harness.handover.identityAt(BASE_EPOCH + 1))) {
                runnerField.set(null, harness.runner);
                harness.fly.setPos(harness.fly.getX() + 1, harness.fly.getY(), harness.fly.getZ());
                harness.fly.tick();
                check(helper, installed.bridge().observations().size() == 1, "actual bridge did not receive the observation");
                var snapshot = FlySensors.sample(helper.getLevel(), harness.fly);
                check(helper, Math.abs(snapshot.xBlocks() - (harness.origin.x + .5)) < 1.0e-9,
                        "fixture reset happened after sampling or not at all");
                harness.runner.stop("hook test finished");
                var ticks = harness.rows().stream().filter(r -> kind(r).equals("tick")).toList();
                check(helper, ticks.size() == 1 && ticks.getFirst().has("world_after"), "post-physics hook did not record exactly one tick");
                check(helper, ticks.getFirst().getAsJsonObject("world_before").get("xBlocks").getAsDouble() == snapshot.xBlocks(),
                        "recorded observation preceded the reset");
            } finally { runnerField.set(null, previous); }
        }
        helper.succeed();
    }

    private static MatchedTrialPlan.Trial cell(MatchedTrialPlan.Stimulus stimulus) {
        return new MatchedTrialPlan.Trial(stimulus.ordinal(), 0, 7919, stimulus, "original");
    }
    private static String kind(JsonObject row) { return row.get("kind").getAsString(); }
    private static long count(List<JsonObject> rows, String kind) { return rows.stream().filter(r -> kind(r).equals(kind)).count(); }
    private static void check(GameTestHelper helper, boolean value, String message) { helper.assertTrue(value, Component.literal(message)); }

    private static final class Harness implements AutoCloseable {
        final FlyEntity fly;
        final Vec3 origin;
        final Path path;
        final MatchedTrialLedger ledger;
        final TestHandover handover;
        final MatchedTrialRunner runner;
        Harness(GameTestHelper helper, List<MatchedTrialPlan.Trial> plan) throws IOException {
            for (int x = 0; x < 8; x++) {
                for (int z = 0; z < 8; z++) { helper.setBlock(new BlockPos(x, 1, z), Blocks.STONE); }
            }
            BlockPos base = helper.absolutePos(BlockPos.ZERO);
            origin = new Vec3(base.getX() + 3, base.getY() - 63, base.getZ());
            fly = helper.spawn(NeuroCraftFly.FLY, new Vec3(3.5, 4, .5));
            path = Files.createTempDirectory("matched-trial-gametest-").resolve("ledger.ndjson");
            ledger = new MatchedTrialLedger(path);
            handover = new TestHandover(fly, plan.size());
            runner = new MatchedTrialRunner(fly, ledger, handover, origin, plan);
        }
        void observe(PhysicalSensorSnapshot sample, long sequence, long skip) {
            ValidationResult decision = ValidationResult.rejected(ControllerFault.ACTIVE);
            Optional<ProgramExecution> execution = Optional.empty();
            if (sequence >= 0) {
                long command = sequence + skip;
                var frame = new ProgramActionFrame(handover.identity, command, command,
                        sample.gameTick() - 2, sample.gameTick(), sample.gameTick(), command * 5, command * 5 + 5,
                        "6".repeat(64), BodyProgramId.HOVER, Modulation.NEUTRAL, PARAMETERS,
                        new ReadoutValues(0, 0, 0), "7".repeat(64));
                decision = ValidationResult.accepted(frame); execution = Optional.of(HOVER);
            }
            var motors = PrimitiveActuator.apply(fly, RawPrimitiveAction.ZERO, PrimitiveAction.NEUTRAL);
            runner.afterPhysics(sample, decision, motors, execution, 0);
        }
        List<JsonObject> rows() {
            try { return Files.readAllLines(path).stream().map(s -> JsonParser.parseString(s).getAsJsonObject()).toList(); }
            catch (IOException exception) { throw new UncheckedIOException(exception); }
        }
        @Override public void close() { runner.stop("test cleanup"); fly.discard(); }
    }

    private static final class TestHandover implements MatchedTrialRunner.Handover {
        final FlyEntity fly;
        final int trials;
        int begins;
        long resetTick;
        ControllerIdentity identity;
        ControllerFault state = ControllerFault.ACTIVE;
        boolean swallowResetFailure;
        TestHandover(FlyEntity fly, int trials) { this.fly = fly; this.trials = trials; identity = identityAt(BASE_EPOCH); }
        ControllerIdentity identityAt(long epoch) {
            return new ControllerIdentity("matched-gametest-synthetic", epoch, fly.getStringUUID(), "1".repeat(64),
                    "2".repeat(64), "3".repeat(64), "4".repeat(64), ControlProtocol.OBSERVATION_SCHEMA_V2_SHA256);
        }
        @Override public ControllerIdentity begin(String lane, Runnable reset) {
            begins++;
            identity = identityAt(BASE_EPOCH + begins);
            try { reset.run(); }
            catch (RuntimeException exception) {
                if (!swallowResetFailure) { throw exception; }
                state = ControllerFault.INTERNAL_ERROR;
                return identity; // Same consumed-epoch/neutral return shape as ComparisonLaneSession.
            }
            resetTick = fly.level().getGameTime();
            state = begins > trials ? ControllerFault.STARTING : ControllerFault.ACTIVE;
            return identity;
        }
        @Override public ControllerIdentity identity() { return identity; }
        @Override public ControllerFault state() { return state; }
    }
}
