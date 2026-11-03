package dev.neurocraft.fly.trial;

import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import dev.neurocraft.fly.body.ProgramExecution;
import dev.neurocraft.fly.body.BodyProgramContextSampler;
import dev.neurocraft.fly.body.programs.LandProgram;
import dev.neurocraft.fly.control.ControlProtocol;
import dev.neurocraft.fly.control.ControllerFault;
import dev.neurocraft.fly.control.ControllerIdentity;
import dev.neurocraft.fly.control.EventSample;
import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import dev.neurocraft.fly.control.ProgramActionFrame;
import dev.neurocraft.fly.control.ValidationResult;
import dev.neurocraft.fly.entity.ActuationReceipt;
import dev.neurocraft.fly.entity.FlyEntity;
import dev.neurocraft.fly.entity.FlySensors;
import java.util.ArrayDeque;
import java.util.List;
import java.util.Optional;
import net.minecraft.core.BlockPos;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.entity.EntitySpawnReason;
import net.minecraft.world.entity.EntityType;
import net.minecraft.world.entity.animal.Cow;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.Vec3;

/**
 * Server-thread experiment scheduler. It moves only its declared stimulus after a one-time
 * fixture reset; the existing authenticated command path alone moves the fly during a trial.
 * The ledger reports commands/receipts observed here, not unseen service-issued commands.
 */
public final class MatchedTrialRunner {
    public interface Handover {
        /** Must retire old ownership, run reset, then start the fresh authenticated bridge. */
        ControllerIdentity begin(String lane, Runnable reset);
        ControllerFault state();
        ControllerIdentity identity();
    }

    private final FlyEntity fly;
    private final ServerLevel level;
    private final MatchedTrialLedger ledger;
    private final Handover handover;
    private final Vec3 origin;
    private final List<MatchedTrialPlan.Trial> plan;
    private final ArrayDeque<PhysicalSensorSnapshot> history = new ArrayDeque<>();
    private int cell = -1;
    private boolean active = true;
    private boolean next = true;
    private boolean restoring;
    private String status = "armed";
    private ControllerIdentity identity;
    private long resetTick;
    private long anchor = -1;
    private boolean stimulusSeen;
    private long commandMissTicks;
    private String anchorFailure;
    private Cow cow;
    private BlockPos block;
    private int plane = -1;

    public MatchedTrialRunner(FlyEntity fly, MatchedTrialLedger ledger, Handover handover) {
        this(fly, ledger, handover, Vec3.ZERO, MatchedTrialPlan.trials());
    }

    /** Translated/subset plans are for bounded server tests; the console uses the fixed plan. */
    public MatchedTrialRunner(FlyEntity fly, MatchedTrialLedger ledger, Handover handover,
            Vec3 origin, List<MatchedTrialPlan.Trial> plan) {
        this.fly = fly;
        this.level = (ServerLevel) fly.level();
        this.ledger = ledger;
        this.handover = handover;
        this.origin = origin;
        this.plan = List.copyOf(plan);
        if (plan.isEmpty() || plan.size() > 60) { throw new IllegalArgumentException("trial count outside 1..60"); }
        JsonObject header = row("plan");
        JsonObject document = MatchedTrialPlan.document();
        String serializedPlan = MatchedTrialLedger.JSON.toJson(document);
        header.add("plan", document);
        header.addProperty("plan_json", serializedPlan);
        header.addProperty("plan_sha256", MatchedTrialPlan.sha256(serializedPlan));
        header.addProperty("plan_hash_scope", "SHA-256 of plan_json UTF-8 bytes, no trailing newline; neurocraft.matched_trial_plan.v1");
        header.add("world_origin", MatchedTrialLedger.JSON.toJsonTree(origin));
        header.addProperty("cell_count", plan.size());
        header.add("executed_cells", MatchedTrialLedger.JSON.toJsonTree(this.plan));
        header.addProperty("source_provenance", "operator release manifest must bind source; per-epoch metadata provides catalog control provenance");
        header.addProperty("receipt_scope", "authoritative program receipt offered to bridge; transport delivery counted separately");
        header.addProperty("command_missing_scope", "local validator retained no command; includes unanchored/handshake ticks, not an issued or late-retired count");
        ledger.append(header);
    }

    public boolean active() { return active; }
    public String status() { return status; }
    public FlyEntity entity() { return fly; }

    /** Called before the fly samples rays/events, exactly once per authoritative world tick. */
    public void beforeObservation() {
        if (!active) { return; }
        try {
            checkWriter();
            if (handover.state().isPaused()) { stop("terminal controller fault: " + handover.state()); return; }
            if (restoring) {
                if (!identity.equals(handover.identity())) { throw new IllegalStateException("restoration identity changed"); }
                if (handover.state() == ControllerFault.ACTIVE) {
                    if (fly.bodyPrograms().trialWanderSeed().isPresent()) {
                        throw new IllegalStateException("ordinary restoration retained the trial seed");
                    }
                    JsonObject restored = row("original_restored");
                    restored.addProperty("authenticated_active", true);
                    restored.addProperty("trial_seed_cleared", true);
                    ledger.append(restored);
                    stop("complete; original restored");
                } else if (level.getGameTime() - resetTick >= MatchedTrialPlan.ANCHOR_WAIT_TICKS) {
                    stop("original restoration timed out");
                }
                return;
            }
            if (next) { beginNext(); }
            if (!active || restoring) { return; }
            if (!identity.equals(handover.identity())) { throw new IllegalStateException("trial identity changed"); }
            assertNoForeignEntities();
            long tick = level.getGameTime();
            if (!history.isEmpty() && tick != history.getLast().gameTick() + 1) {
                throw new IllegalStateException("skipped or duplicate observation tick; scheduled trial incomplete");
            }
            if (anchor < 0) {
                if (!atInitialPose()) {
                    finishTrial("invalid_initial_state", "body changed before applied command 0");
                    beginNext();
                    return;
                }
                if (tick - resetTick >= MatchedTrialPlan.ANCHOR_WAIT_TICKS) {
                    finishTrial("missing_command_zero", "no applied program command 0 inside anchor budget");
                    beginNext();
                }
                return;
            }
            long elapsed = tick - anchor;
            if (elapsed >= MatchedTrialPlan.TOTAL_TICKS) {
                finishTrial(stimulusSeen ? "complete" : "complete_stimulus_ray_miss",
                        stimulusSeen ? "declared stimulus matched forward ray geometry" : "no matching forward-ray stimulus hit observed");
                beginNext();
                return;
            }
            setStimulus(MatchedTrialPlan.frontPlane(elapsed));
        } catch (RuntimeException exception) {
            stop("trial failure: " + exception.getMessage());
        }
    }

    /** Called after normal physics; never requests, replaces or reapplies a motor command. */
    public void afterPhysics(PhysicalSensorSnapshot before, ValidationResult decision,
            ActuationReceipt motors, Optional<ProgramExecution> execution, long sentProgramReceipts) {
        if (!active || next || restoring || identity == null) { return; }
        try {
            checkWriter();
            if (!history.isEmpty() && before.gameTick() != history.getLast().gameTick() + 1) {
                throw new IllegalStateException("skipped or duplicate recorded observation tick; ledger incomplete");
            }
            if (history.size() == 3) { history.removeFirst(); }
            history.addLast(before);
            JsonObject tick = row("tick");
            tick.addProperty("game_tick", before.gameTick());
            tick.addProperty("anchor_offset", anchor < 0 ? null : before.gameTick() - anchor);
            tick.add("world_before", MatchedTrialLedger.JSON.toJsonTree(before));
            tick.add("world_after", bodyState());
            tick.add("actuation", MatchedTrialLedger.JSON.toJsonTree(motors));
            tick.addProperty("selection_source", fly.bodyPrograms().selectionSource().name());
            tick.addProperty("body_seed", fly.bodyPrograms().effectiveWanderSeed());
            tick.addProperty("controller_state", decision.state().name());
            tick.addProperty("command_admitted", decision.accepted());
            tick.addProperty("command_missing", decision.command().isEmpty());
            tick.addProperty("program_receipts_sent_total", sentProgramReceipts);
            tick.add("command", decision.command().isEmpty() ? JsonNull.INSTANCE
                    : MatchedTrialLedger.JSON.toJsonTree(decision.command().orElseThrow()));
            JsonObject executed = null;
            if (execution.isPresent()) {
                ProgramExecution actual = execution.orElseThrow();
                executed = new JsonObject();
                executed.addProperty("program", actual.executedProgram().name());
                executed.add("modulation", MatchedTrialLedger.JSON.toJsonTree(actual.executedModulation()));
                executed.add("parameters", MatchedTrialLedger.JSON.toJsonTree(actual.executedParameters()));
                executed.add("motors", MatchedTrialLedger.JSON.toJsonTree(actual.emittedMotors()));
                executed.addProperty("refusal", actual.refusalReason().map(Enum::name).orElse(null));
            }
            tick.add("program_execution", executed == null ? JsonNull.INSTANCE : executed);
            tick.add("stimulus", stimulusState());
            if (decision.command().isEmpty()) { commandMissTicks++; }
            if (decision.command().orElse(null) instanceof ProgramActionFrame command) {
                if (!identity.equals(command.identity())) { throw new IllegalStateException("foreign trial command identity"); }
                // Mirror recordActuation's window guard. This is the encoded local offer,
                // not an acknowledgement of transport delivery or service receipt.
                if (!decision.accepted() || (before.gameTick() >= command.targetTick()
                        && before.gameTick() <= command.validThroughTick())) {
                    tick.add("program_receipt_offered", JsonParser.parseString(ControlProtocol.encodeProgramReceipt(
                            identity, command, before, execution, decision)));
                }
                if (anchor < 0) { tryAnchor(command, before.gameTick(), decision, execution, tick); }
            }
            boolean matchedHit = matchesStimulusRay(before);
            stimulusSeen |= matchedHit;
            tick.addProperty("stimulus_forward_ray_matches_geometry", matchedHit);
            ledger.append(tick);
            if (decision.state().isPaused()) { stop("terminal controller fault: " + decision.state()); }
            else if (anchorFailure != null) { finishTrial("invalid_anchor", anchorFailure); }
        } catch (RuntimeException exception) {
            stop("trial failure: " + exception.getMessage());
        }
    }

    private void tryAnchor(ProgramActionFrame command, long tick, ValidationResult decision,
            Optional<ProgramExecution> execution, JsonObject row) {
        if (command.commandSequence() != 0 || !decision.accepted()
                || execution.filter(ProgramExecution::applied).isEmpty()) {
            anchorFailure = "first observed program command was not applied command 0";
            return;
        }
        PhysicalSensorSnapshot initial = history.stream().filter(sample -> sample.gameTick() == command.observationTick())
                .findFirst().orElse(null);
        if (command.observationSequence() != 0 || command.neuralStepStart() != 0 || command.neuralStepEnd() != 5
                || initial == null || !validInitial(initial)) {
            anchorFailure = "command 0 did not bind the declared actual observation 0";
            return;
        }
        anchor = tick;
        status = "running cell=" + cell + " lane=" + current().lane() + " epoch=" + identity.sessionEpoch();
        row.addProperty("anchor_offset", 0);
        row.add("initial_observation_zero", MatchedTrialLedger.JSON.toJsonTree(initial));
    }

    private void beginNext() {
        clearStimulus();
        if (++cell >= plan.size()) {
            restoring = true;
            identity = handover.begin("original", fly::resetNeuralSessionPreservingBody);
            resetTick = level.getGameTime();
            status = "original restoration attempted epoch=" + identity.sessionEpoch();
            JsonObject attempt = row("original_restoration_attempt");
            attempt.addProperty("trial_seed_cleared", fly.bodyPrograms().trialWanderSeed().isEmpty());
            ledger.append(attempt);
            return;
        }
        assertFixtureClear();
        next = false;
        anchor = -1;
        history.clear();
        stimulusSeen = false;
        commandMissTicks = 0;
        anchorFailure = null;
        MatchedTrialPlan.Trial trial = current();
        JsonObject bodyBeforeReset = bodyState();
        boolean[] resetCompleted = {false};
        identity = handover.begin(trial.lane(), () -> {
            fly.resetForTrial(origin.x + MatchedTrialPlan.START_X, origin.y + MatchedTrialPlan.START_Y,
                    origin.z + MatchedTrialPlan.START_Z, trial.bodySeed());
            if (!atInitialPose()) { throw new IllegalStateException("fixture reset did not establish the declared initial body"); }
            resetCompleted[0] = true;
        });
        if (!resetCompleted[0] || handover.state().isPaused() || !atInitialPose()) {
            throw new IllegalStateException("trial handover/reset failed before an initial observation");
        }
        resetTick = level.getGameTime();
        status = "waiting_command_zero cell=" + cell + " lane=" + trial.lane() + " epoch=" + identity.sessionEpoch();
        JsonObject row = row("trial_start");
        row.addProperty("reset_game_tick", resetTick);
        row.add("body_before_reset", bodyBeforeReset);
        row.add("initial_body", bodyState());
        row.addProperty("verified_initial_feet_to_floor_blocks", MatchedTrialPlan.INITIAL_FLOOR_CLEARANCE_BLOCKS);
        row.addProperty("verified_initial_ground_contact", false);
        ledger.append(row);
    }

    private boolean validInitial(PhysicalSensorSnapshot sample) {
        return close(sample.xBlocks(), origin.x + MatchedTrialPlan.START_X)
                && close(sample.yBlocks(), origin.y + MatchedTrialPlan.START_Y)
                && close(sample.zBlocks(), origin.z + MatchedTrialPlan.START_Z)
                && sample.velocityXBlocksPerTick() == 0 && sample.velocityYBlocksPerTick() == 0
                && sample.velocityZBlocksPerTick() == 0 && sample.yawDegrees() == 0 && sample.pitchDegrees() == 0
                && !sample.onGround() && !sample.horizontalCollision() && !sample.verticalCollision()
                && !sample.inWater() && !sample.inLava()
                && (sample.events() == null || sample.events().equals(EventSample.EMPTY));
    }

    private boolean atInitialPose() {
        return close(fly.getX(), origin.x + MatchedTrialPlan.START_X)
                && close(fly.getY(), origin.y + MatchedTrialPlan.START_Y)
                && close(fly.getZ(), origin.z + MatchedTrialPlan.START_Z)
                && fly.getDeltaMovement().lengthSqr() == 0 && fly.getYRot() == 0 && fly.getXRot() == 0
                && !fly.onGround() && !fly.horizontalCollision && !fly.verticalCollision
                && !fly.isInWater() && !fly.isInLava() && fly.eventSample().equals(EventSample.EMPTY)
                && fly.bodyPrograms().trialWanderSeed().isPresent()
                && fly.bodyPrograms().trialWanderSeed().getAsLong() == current().bodySeed()
                && hasDeclaredInitialSupport();
    }

    private boolean hasDeclaredInitialSupport() {
        var context = BodyProgramContextSampler.sample(level, fly, FlySensors.sample(level, fly));
        return context.surfaceBelow().filter(surface -> surface.sturdy()
                && Math.abs(surface.distanceBlocks() - MatchedTrialPlan.INITIAL_FLOOR_CLEARANCE_BLOCKS)
                        <= BodyProgramContextSampler.SUPPORT_HEIGHT_TOLERANCE_BLOCKS).isPresent()
                && !LandProgram.hasGroundContact(context);
    }

    private void assertNoForeignEntities() {
        AABB region = new AABB(origin.x - 6, origin.y + 64, origin.z - 6,
                origin.x + 7, origin.y + 74, origin.z + 8);
        if (!level.getEntities(fly, region, entity -> entity != cow && !entity.isRemoved() && !entity.isSpectator()).isEmpty()) {
            throw new IllegalStateException("foreign entity in matched fixture; spectator/clear fixture first");
        }
    }

    private void assertFixtureClear() {
        assertNoForeignEntities();
        for (int z = 1; z <= 6; z++) {
            if (!level.getBlockState(stimulusBlock(z)).isAir()) { throw new IllegalStateException("stimulus path is not clear air"); }
        }
        if (!level.noCollision(fly, fly.getBoundingBox().move(
                origin.x + MatchedTrialPlan.START_X - fly.getX(), origin.y + MatchedTrialPlan.START_Y - fly.getY(),
                origin.z + MatchedTrialPlan.START_Z - fly.getZ()))) {
            throw new IllegalStateException("initial body volume is obstructed");
        }
    }

    private void setStimulus(int requestedPlane) {
        if (requestedPlane < 0) { clearStimulus(); return; }
        if (block != null && !level.getBlockState(block).is(Blocks.STONE)) {
            throw new IllegalStateException("stimulus block externally changed");
        }
        if (requestedPlane == plane && block != null) { return; }
        if (block != null) {
            level.setBlockAndUpdate(block, Blocks.AIR.defaultBlockState());
            block = null;
        }
        plane = requestedPlane;
        if (current().stimulus() == MatchedTrialPlan.Stimulus.BLOCK) {
            block = stimulusBlock(plane);
            if (!level.getBlockState(block).isAir()) { throw new IllegalStateException("stimulus cell externally changed"); }
            level.setBlockAndUpdate(block, Blocks.STONE.defaultBlockState());
        } else {
            boolean fresh = cow == null;
            if (fresh) {
                cow = EntityType.COW.create(level, EntitySpawnReason.COMMAND);
                if (cow == null) { throw new IllegalStateException("could not allocate the declared cow stimulus"); }
                cow.setNoAi(true);
                cow.setNoGravity(true);
                cow.setInvulnerable(true);
            } else if (cow.isRemoved()) { throw new IllegalStateException("stimulus entity externally removed"); }
            cow.setPos(origin.x + 0.5, origin.y + 67, origin.z + plane + cow.getBbWidth() / 2.0);
            cow.setDeltaMovement(Vec3.ZERO);
            if (fresh && !level.addFreshEntity(cow)) { throw new IllegalStateException("stimulus entity insertion failed"); }
        }
    }

    private BlockPos stimulusBlock(int front) { return BlockPos.containing(origin.x, origin.y + 67, origin.z + front); }

    private void clearStimulus() {
        if (cow != null) { cow.discard(); cow = null; }
        if (block != null) {
            if (level.getBlockState(block).is(Blocks.STONE)) { level.setBlockAndUpdate(block, Blocks.AIR.defaultBlockState()); }
            block = null;
        }
        plane = -1;
    }

    private JsonObject stimulusState() {
        JsonObject value = new JsonObject();
        value.addProperty("present", plane >= 0);
        value.addProperty("front_plane_z", plane < 0 ? null : origin.z + plane);
        value.addProperty("entity_uuid", cow == null ? null : cow.getStringUUID());
        value.add("entity_bounds", cow == null ? JsonNull.INSTANCE : MatchedTrialLedger.JSON.toJsonTree(cow.getBoundingBox()));
        value.add("block_position", block == null ? JsonNull.INSTANCE : MatchedTrialLedger.JSON.toJsonTree(block));
        return value;
    }

    private boolean matchesStimulusRay(PhysicalSensorSnapshot sample) {
        if (plane < 0) { return false; }
        int expectedKind = current().stimulus() == MatchedTrialPlan.Stimulus.ENTITY ? 2 : 1;
        if (sample.rayHitKinds().getFirst() != expectedKind) { return false; }
        AABB bounds = cow == null ? new AABB(block) : cow.getBoundingBox();
        Vec3 from = new Vec3(sample.xBlocks(), sample.yBlocks() + fly.getBbHeight() / 2.0, sample.zBlocks());
        Vec3 to = from.add(Vec3.directionFromRotation((float) sample.pitchDegrees(), (float) sample.yawDegrees())
                .normalize().scale(sample.rayMaxDistanceBlocks()));
        // Same zero-inflation segment as FlySensors, evaluated against only our owned shape.
        // A type alone is insufficient to attribute a distant arena-wall hit to the stimulus.
        Optional<Vec3> hit = bounds.clip(from, to);
        return hit.isPresent() && Math.abs(from.distanceTo(hit.orElseThrow()) - sample.rayForwardBlocks()) < 1.0e-6;
    }

    private JsonObject bodyState() {
        JsonObject value = new JsonObject();
        value.add("position", MatchedTrialLedger.JSON.toJsonTree(fly.position()));
        value.add("velocity", MatchedTrialLedger.JSON.toJsonTree(fly.getDeltaMovement()));
        value.addProperty("yaw", fly.getYRot()); value.addProperty("pitch", fly.getXRot());
        value.addProperty("on_ground", fly.onGround());
        value.addProperty("vertical_collision", fly.verticalCollision);
        value.addProperty("horizontal_collision", fly.horizontalCollision);
        value.addProperty("body_seed", fly.bodyPrograms().effectiveWanderSeed());
        value.addProperty("trial_seed_latched", fly.bodyPrograms().trialWanderSeed().isPresent());
        value.addProperty("trial_wander_seed", fly.bodyPrograms().trialWanderSeed().isPresent()
                ? fly.bodyPrograms().trialWanderSeed().getAsLong() : null);
        value.addProperty("body_program", fly.bodyPrograms().state().id().name());
        value.addProperty("entered_game_tick", fly.bodyPrograms().state().enteredGameTick());
        value.addProperty("ticks_in_program", fly.bodyPrograms().state().ticksInProgram());
        value.addProperty("phase_index", fly.bodyPrograms().state().phaseIndex());
        value.addProperty("program_owned_ticks", fly.bodyPrograms().programOwnedTicks());
        value.addProperty("readout_owned_ticks", fly.bodyPrograms().readoutOwnedTicks());
        value.addProperty("readout_stalled_ticks", fly.bodyPrograms().readoutStalledTicks());
        value.addProperty("in_water", fly.isInWater());
        value.addProperty("in_lava", fly.isInLava());
        value.add("events", MatchedTrialLedger.JSON.toJsonTree(fly.eventSample()));
        return value;
    }

    private JsonObject row(String kind) {
        JsonObject value = new JsonObject();
        value.addProperty("schema", "neurocraft.matched_trial_ledger.v1");
        value.addProperty("diagnostic_only", true);
        value.addProperty("kind", kind);
        value.add("identity", identity == null ? JsonNull.INSTANCE : MatchedTrialLedger.JSON.toJsonTree(identity));
        value.add("trial", cell < 0 || cell >= plan.size() ? JsonNull.INSTANCE : MatchedTrialLedger.JSON.toJsonTree(current()));
        return value;
    }

    private MatchedTrialPlan.Trial current() { return plan.get(cell); }
    private void checkWriter() {
        if (ledger.failure() != null) { throw new IllegalStateException(ledger.failure()); }
    }
    private static boolean close(double a, double b) { return Math.abs(a - b) <= 1.0e-9; }

    private void finishTrial(String outcome, String reason) {
        JsonObject value = row("trial_end");
        value.addProperty("outcome", outcome); value.addProperty("reason", reason);
        value.addProperty("game_tick", level.getGameTime());
        value.addProperty("anchor_game_tick", anchor < 0 ? null : anchor);
        value.addProperty("command_missing_ticks", commandMissTicks);
        value.addProperty("stimulus_forward_ray_hit", stimulusSeen);
        ledger.append(value);
        clearStimulus();
        next = true;
    }

    public void stop(String reason) {
        if (!active) { return; }
        active = false;
        status = reason;
        clearStimulus();
        try {
            JsonObject end = row("run_end"); end.addProperty("reason", reason);
            end.add("observed_current_identity", MatchedTrialLedger.JSON.toJsonTree(handover.identity()));
            end.add("final_body", bodyState());
            end.addProperty("last_recorded_game_tick", history.isEmpty() ? null : history.getLast().gameTick());
            end.addProperty("cell_in_progress", cell >= 0 && cell < plan.size() && !next);
            end.addProperty("anchor_game_tick", anchor < 0 ? null : anchor);
            end.addProperty("trial_seed_latched", fly.bodyPrograms().trialWanderSeed().isPresent());
            end.addProperty("game_tick", level.getGameTime()); ledger.append(end);
        } catch (RuntimeException ignored) { status += "; ledger incomplete"; }
        ledger.close();
        if (ledger.failure() != null) { status += "; " + ledger.failure(); }
    }
}
