package dev.neurocraft.fly.entity;

import dev.neurocraft.fly.FlyControllerRuntime;
import dev.neurocraft.fly.body.BodyProgramChannel;
import dev.neurocraft.fly.body.BodyProgramContext;
import dev.neurocraft.fly.body.BodyProgramContextSampler;
import dev.neurocraft.fly.body.BodyProgramRuntime;
import dev.neurocraft.fly.body.BodyProgramStep;
import dev.neurocraft.fly.body.MotorAuthority;
import dev.neurocraft.fly.body.ProgramExecution;
import dev.neurocraft.fly.body.ReadoutChannel;
import dev.neurocraft.fly.body.programs.FeedProgram;
import dev.neurocraft.fly.control.ActionFrame;
import dev.neurocraft.fly.control.BrainDisplayPresentation;
import dev.neurocraft.fly.control.EventDisplayPresentation;
import dev.neurocraft.fly.control.ComparisonLaneClientCache;
import dev.neurocraft.fly.control.ComparisonLanePresentation;
import dev.neurocraft.fly.control.CommandFrame;
import dev.neurocraft.fly.control.ControllerFault;
import dev.neurocraft.fly.control.EventSample;
import dev.neurocraft.fly.control.ProgramActionFrame;
import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import dev.neurocraft.fly.control.PrimitiveAction;
import dev.neurocraft.fly.control.RawPrimitiveAction;
import dev.neurocraft.fly.control.ValidationResult;
import java.util.ArrayDeque;
import java.util.Iterator;
import java.util.Optional;
import net.minecraft.nbt.CompoundTag;
import net.minecraft.network.syncher.EntityDataAccessor;
import net.minecraft.network.syncher.EntityDataSerializers;
import net.minecraft.network.syncher.SynchedEntityData;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.util.Mth;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.InteractionResult;
import net.minecraft.world.damagesource.DamageSource;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EntityType;
import net.minecraft.world.entity.MoverType;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.Items;
import net.minecraft.world.level.Level;
import net.minecraft.world.phys.Vec3;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Project-owned server-authoritative fly. It extends Entity directly, so no goal selector,
 * navigation, target selector, or vanilla mob fallback exists in its inheritance path.
 */
public final class FlyEntity extends Entity {
    private static final Logger EVENT_LOGGER = LoggerFactory.getLogger("neurocraft-fly-events");
    private static final int SENSOR_HISTORY_CAPACITY = 3;
    private static final EntityDataAccessor<Integer> CONTROLLER_STATE =
            SynchedEntityData.defineId(FlyEntity.class, EntityDataSerializers.INT);
    private static final EntityDataAccessor<Float> FORWARD_THRUST =
            SynchedEntityData.defineId(FlyEntity.class, EntityDataSerializers.FLOAT);
    private static final EntityDataAccessor<Float> LATERAL_THRUST =
            SynchedEntityData.defineId(FlyEntity.class, EntityDataSerializers.FLOAT);
    private static final EntityDataAccessor<Float> VERTICAL_THRUST =
            SynchedEntityData.defineId(FlyEntity.class, EntityDataSerializers.FLOAT);
    private static final EntityDataAccessor<Float> YAW_TORQUE =
            SynchedEntityData.defineId(FlyEntity.class, EntityDataSerializers.FLOAT);
    private static final EntityDataAccessor<Float> PITCH_TORQUE =
            SynchedEntityData.defineId(FlyEntity.class, EntityDataSerializers.FLOAT);

    // ADR 0020 presentation channels: the packed body-program word (selection source, program id,
    // and the modulation at its published tenths resolution) and the program's normalized phase.
    // Both are drawn and reported; neither is transmitted, traced, hashed, or read by the
    // controller, and neither alters the command the actuator applies.
    private static final EntityDataAccessor<Integer> BODY_PROGRAM =
            SynchedEntityData.defineId(FlyEntity.class, EntityDataSerializers.INT);
    private static final EntityDataAccessor<Float> BODY_PROGRAM_PHASE =
            SynchedEntityData.defineId(FlyEntity.class, EntityDataSerializers.FLOAT);

    // The three labeled readout scalars the mapping consumed, packed at their published hundredths
    // resolution. Drawn and reported only: nothing here is transmitted, traced, hashed, or read by
    // the controller, and it never alters the command the actuator applies. The values travel
    // service -> mod on a program frame; nothing mod-side computes or returns them, so the sensory
    // vector and observation_schema_hash are untouched.
    private static final EntityDataAccessor<Integer> READOUT_CHANNEL =
            SynchedEntityData.defineId(FlyEntity.class, EntityDataSerializers.INT);
    private static final EntityDataAccessor<String> BRAIN_DISPLAY =
            SynchedEntityData.defineId(FlyEntity.class, EntityDataSerializers.STRING);
    private static final EntityDataAccessor<String> COMPARISON_LANE =
            SynchedEntityData.defineId(FlyEntity.class, EntityDataSerializers.STRING);

    private static final EntityDataAccessor<String> EVENT_DISPLAY =
            SynchedEntityData.defineId(FlyEntity.class, EntityDataSerializers.STRING);

    private static final double AIR_HORIZONTAL_DRAG = 0.91;

    /**
     * Vertical drag applied after the move (ADR 0014). Public because a body program derives its
     * own bob entry phase from it: the vertical axis is a first-order filter and the phase whose
     * steady-state velocity is zero is a closed form in this constant.
     */
    public static final double AIR_VERTICAL_DRAG = 0.98;

    private static final double WATER_DRAG = 0.80;

    private ActuationReceipt lastActuation = neutralReceipt();
    private CommandFrame lastAcceptedCommand;
    private final ComparisonLaneClientCache presentationCache = new ComparisonLaneClientCache();
    private final BodyProgramRuntime bodyPrograms = new BodyProgramRuntime();
    private EventAccumulator events = new EventAccumulator();
    private boolean eventDataWarningLogged;
    private static final FeedProgram FOOD_CONTACT_PREDICATE = new FeedProgram();
    private final ArrayDeque<PhysicalSensorSnapshot> sensorHistory =
            new ArrayDeque<>(SENSOR_HISTORY_CAPACITY);

    public FlyEntity(EntityType<? extends FlyEntity> type, Level level) {
        super(type, level);
    }

    @Override
    protected void defineSynchedData(SynchedEntityData.Builder builder) {
        builder.define(CONTROLLER_STATE, ControllerFault.STARTING.ordinal());
        builder.define(FORWARD_THRUST, 0.0F);
        builder.define(LATERAL_THRUST, 0.0F);
        builder.define(VERTICAL_THRUST, 0.0F);
        builder.define(YAW_TORQUE, 0.0F);
        builder.define(PITCH_TORQUE, 0.0F);
        builder.define(BODY_PROGRAM, BodyProgramChannel.DEFAULT.packed());
        builder.define(BODY_PROGRAM_PHASE, 0.0F);
        builder.define(READOUT_CHANNEL, ReadoutChannel.DEFAULT.packed());
        builder.define(BRAIN_DISPLAY, FlyControllerRuntime.UNAVAILABLE_BRAIN_DISPLAY);
        builder.define(EVENT_DISPLAY, "");
        builder.define(COMPARISON_LANE, ComparisonLanePresentation.unavailable().encode());
    }

    @Override
    public void tick() {
        super.tick();
        if (!(level() instanceof ServerLevel serverLevel)) {
            return;
        }
        FlyControllerRuntime.processPendingReset(this, serverLevel.getGameTime());
        FlyControllerRuntime.processPendingProgramAuthority(this, serverLevel.getGameTime());
        FlyControllerRuntime.beforeMatchedTrialObservation(this);
        PhysicalSensorSnapshot snapshot = FlySensors.sample(serverLevel, this);
        BodyProgramContext bodyContext = BodyProgramContextSampler.sample(serverLevel, this, snapshot);
        // Only this authoritative tick evaluates food edges. V3 exposes food-near and exact
        // FEED-admissible contact booleans; no bearing, selected item identity or motor is sent.
        if (events != null) {
            snapshot = snapshot.withEvents(events.sampleFood(FoodEventSampler.foodNear(serverLevel, this),
                    FOOD_CONTACT_PREDICATE.canEnter(bodyContext), snapshot.gameTick()));
        }
        bodyContext = new BodyProgramContext(snapshot, bodyContext.surfaceBelow(),
                bodyContext.target(), bodyContext.food());
        rememberSensorSnapshot(snapshot);
        ValidationResult decision = FlyControllerRuntime.control(this, snapshot);
        // ADR 0020 arbitration. control() is still called on every entity tick -- the result is
        // discarded, never the call -- so CommandValidator's DORMANCY_GAP_TICKS premise is
        // untouched and no third dormancy window is opened.
        boolean commandAdmitted = decision.accepted();
        // The neutral flag alone is not a sound proxy for "the service is paused": a controller
        // can legitimately emit a neutral action while RUNNING. A paused service additionally
        // leaves its neural cursor where it was, and ActionFrame permits an empty neural interval
        // only for a neutral action, so requiring both is strictly tighter and is the same second
        // signature scripts/verify_paused_window.py demands of a trace.
        boolean commandNeutral = commandAdmitted
                && decision.command().orElseThrow() instanceof ActionFrame raw
                && raw.neutral()
                && raw.neuralStepEnd() == raw.neuralStepStart();
        // A latched controller fault ends the session: the validator will never admit another
        // frame, so nothing is selecting a program any more. Without this the body would keep
        // READOUT authority forever -- neutral, but with the HUD still drawing the green readout
        // lines -- and the operator would be wedged, because a RESET is refused while a body
        // program holds authority and `program release` issues a RESUME the service will not
        // accept. Lane A's SCRIPTED authority deliberately keeps its retention: that lane is
        // entered against an already-paused service and the operator is expected to release it.
        if (decision.state().isPaused()
                && bodyPrograms.authority() == MotorAuthority.READOUT) {
            bodyPrograms.releaseScripted(snapshot.gameTick(), "faulted");
        }
        // A program frame is a selection, not a motor request, so it drives the readout lane
        // instead of the operator lane's advance(). The two are exclusive on any one tick by
        // construction: at most one of the two calls below happens.
        //
        // The `commandAdmitted` guard is load-bearing, not defensive. A rejected ValidationResult
        // still carries its command envelope -- that is how the authoritative REJECTED receipt
        // binds the frame it refuses -- so without the guard every terminal rejection would hand
        // the refused frame's selection to the readout lane and move the body on a command the
        // validator just refused.
        Optional<ProgramActionFrame> programCommand = commandAdmitted
                ? decision.command()
                        .filter(ProgramActionFrame.class::isInstance)
                        .map(ProgramActionFrame.class::cast)
                : Optional.empty();
        Optional<ProgramExecution> readoutExecution = Optional.empty();
        Optional<BodyProgramStep> programStep = Optional.empty();
        if (programCommand.isPresent()) {
            ProgramActionFrame frame = programCommand.orElseThrow();
            readoutExecution = bodyPrograms.advanceReadout(
                    bodyContext,
                    frame.program(),
                    frame.modulation(),
                    frame.parameters(),
                    frame.readout(),
                    frame.policyHash());
        } else {
            programStep = bodyPrograms.advance(bodyContext, commandAdmitted, commandNeutral);
        }
        PrimitiveAction presented;
        if (readoutExecution.isPresent()) {
            // A body program owns this tick because an admitted program frame selected one. The
            // frame requests no motors of its own, so nothing the connectome asked for as a motor
            // vector is displaced; the receipt below reports what actually ran.
            lastAcceptedCommand = programCommand.orElseThrow();
            presented = readoutExecution.orElseThrow().emittedMotors();
            lastActuation = PrimitiveActuator.apply(this, RawPrimitiveAction.ZERO, presented);
        } else if (programStep.isPresent()) {
            // A scripted body program owns this tick. It can only get here while the neural
            // service is paused and the admitted command is neutral, so nothing the connectome
            // asked for is being displaced.
            lastAcceptedCommand = decision.command().orElseThrow();
            presented = programStep.orElseThrow().motors();
            lastActuation = PrimitiveActuator.apply(this, RawPrimitiveAction.ZERO, presented);
        } else if (commandAdmitted) {
            lastAcceptedCommand = decision.command().orElseThrow();
            presented = decision.appliedAction();
            lastActuation = PrimitiveActuator.apply(
                    this, decision.rawAction(), decision.appliedAction());
        } else {
            presented = decision.appliedAction();
            lastActuation = neutralReceipt(this, decision.rawAction());
        }
        updatePresentation(decision.state(), presented);
        FlyControllerRuntime.recordActuation(
                snapshot, decision, lastActuation, programStep.isPresent(), readoutExecution);
        integratePassivePhysics();
        FlyControllerRuntime.afterMatchedTrialPhysics(this, snapshot, decision, lastActuation, readoutExecution);
        publishComparison(FlyControllerRuntime.comparisonFor(this));
        if (snapshot.gameTick() % 4 == 0) {
            entityData.set(BRAIN_DISPLAY, FlyControllerRuntime.brainDisplayFor(this, snapshot.gameTick()));
            entityData.set(EVENT_DISPLAY, FlyControllerRuntime.eventDisplayFor(this, snapshot.gameTick()));
        }
    }

    /** The renderer reads one complete synchronized sample; it never opens the local file. */
    public BrainDisplayPresentation brainDisplayPresentation() {
        return presentationCache.display(System.nanoTime());
    }

    public EventDisplayPresentation eventDisplayPresentation() {
        return presentationCache.events(System.nanoTime());
    }

    public ComparisonLanePresentation comparisonPresentation() {
        return presentationCache.lane();
    }

    public void publishComparison(ComparisonLanePresentation presentation) {
        entityData.set(COMPARISON_LANE, presentation.encode());
    }

    @Override
    public void onSyncedDataUpdated(EntityDataAccessor<?> key) {
        super.onSyncedDataUpdated(key);
        if (BRAIN_DISPLAY.equals(key)) {
            presentationCache.receiveDisplay(BrainDisplayPresentation.parseOrInvalid(
                    entityData.get(BRAIN_DISPLAY)), System.nanoTime());
        } else if (EVENT_DISPLAY.equals(key)) {
            presentationCache.receiveEvents(EventDisplayPresentation.parseOrInvalid(
                    entityData.get(EVENT_DISPLAY)), System.nanoTime());
        } else if (COMPARISON_LANE.equals(key)) {
            presentationCache.receiveLane(ComparisonLanePresentation.parseOrUnavailable(
                    entityData.get(COMPARISON_LANE)), System.nanoTime());
        }
    }

    @Override
    public boolean hurtServer(ServerLevel level, DamageSource source, float amount) {
        if (events != null && level == level() && Float.isFinite(amount) && amount > 0
                && source.getEntity() instanceof ServerPlayer player
                && player.level() == level && player.isAlive() && !player.isSpectator()
                && !isRemoved()) {
            events.recordAttackAttempt(level.getGameTime());
        }
        // An attack attempt is recorded, but this Entity still has no health or damage effect.
        return false;
    }

    @Override public boolean isPickable() { return !isRemoved(); }

    @Override
    public InteractionResult interact(Player player, InteractionHand hand) {
        if (events != null && level() instanceof ServerLevel serverLevel && player instanceof ServerPlayer
                && player.level() == serverLevel && player.isAlive() && !player.isSpectator()
                && !isRemoved() && !isInWater() && !isInLava()
                && player.getItemInHand(hand).is(Items.BRUSH)
                && events.recordBrush(serverLevel.getGameTime())) {
            return InteractionResult.SUCCESS_SERVER;
        }
        return InteractionResult.PASS;
    }

    /** Immutable current event counters; sampling here neither consumes nor creates an event. */
    public EventSample eventSample() { return events == null ? null : events.snapshot(); }

    @Override
    protected void readAdditionalSaveData(CompoundTag tag) {
        releaseAndResetBodyPrograms();
        updatePresentation(ControllerFault.STARTING, PrimitiveAction.NEUTRAL);
        lastActuation = neutralReceipt();
        lastAcceptedCommand = null;
        sensorHistory.clear();
        try {
            events = tag.contains(EventNbt.KEY)
                    ? EventNbt.read(tag.get(EventNbt.KEY), level().getGameTime())
                    : new EventAccumulator();
        } catch (IllegalArgumentException invalid) {
            events = null;
            if (!eventDataWarningLogged) {
                eventDataWarningLogged = true;
                EVENT_LOGGER.warn("NeuroCraft Fly event sampling unavailable for {}: {}",
                        getStringUUID(), invalid.getMessage());
            }
        }
    }

    @Override
    protected void addAdditionalSaveData(CompoundTag tag) {
        // Run/session observation sequencing belongs to the live transport, not persistent NBT.
        tag.put(EventNbt.KEY, EventNbt.write(events));
    }

    public ControllerFault controllerState() {
        int ordinal = entityData.get(CONTROLLER_STATE);
        ControllerFault[] values = ControllerFault.values();
        return ordinal >= 0 && ordinal < values.length
                ? values[ordinal]
                : ControllerFault.INTERNAL_ERROR;
    }

    public PrimitiveAction presentationAction() {
        return new PrimitiveAction(
                entityData.get(FORWARD_THRUST),
                entityData.get(LATERAL_THRUST),
                entityData.get(VERTICAL_THRUST),
                entityData.get(YAW_TORQUE),
                entityData.get(PITCH_TORQUE));
    }

    public ActuationReceipt lastActuation() {
        return lastActuation;
    }

    /**
     * The server-owned scripted body-program lane (ADR 0020). Server-side state: the client's copy
     * of this entity never ticks it, and the renderer reads the synchronized channel below instead.
     */
    public BodyProgramRuntime bodyPrograms() {
        return bodyPrograms;
    }

    /** Synchronized presentation channel: selection source, program id, published modulation. */
    public BodyProgramChannel bodyProgramChannel() {
        return BodyProgramChannel.decode(entityData.get(BODY_PROGRAM));
    }

    /** Synchronized normalized program phase in {@code [0, 1)}; presentation only. */
    public float bodyProgramPhase() {
        return entityData.get(BODY_PROGRAM_PHASE);
    }

    /** Synchronized driving readout scalars at their published hundredths; presentation only. */
    public ReadoutChannel readoutChannel() {
        return ReadoutChannel.decode(entityData.get(READOUT_CHANNEL));
    }

    /** Last validated neural command of either kind, exposed only for read-only inspection. */
    public Optional<CommandFrame> lastAcceptedCommand() {
        return Optional.ofNullable(lastAcceptedCommand);
    }

    /** Exact recent authoritative sample retained only for causal test/evidence correlation. */
    public Optional<PhysicalSensorSnapshot> sensorSnapshotAt(long gameTick) {
        Iterator<PhysicalSensorSnapshot> snapshots = sensorHistory.descendingIterator();
        while (snapshots.hasNext()) {
            PhysicalSensorSnapshot snapshot = snapshots.next();
            if (snapshot.gameTick() == gameTick) {
                return Optional.of(snapshot);
            }
        }
        return Optional.empty();
    }

    /** Server-only explicit fixture reset; ordinary movement never calls absolute positioning. */
    public void resetForFixture(double x, double y, double z, float yawDegrees) {
        if (level().isClientSide()) {
            throw new IllegalStateException("fixture reset is server-only");
        }
        setPos(x, y, z);
        setYRot(Mth.wrapDegrees(yawDegrees));
        setXRot(0.0F);
        setDeltaMovement(Vec3.ZERO);
        releaseAndResetBodyPrograms();
        updatePresentation(ControllerFault.STARTING, PrimitiveAction.NEUTRAL);
        lastActuation = neutralReceipt();
        lastAcceptedCommand = null;
        sensorHistory.clear();
    }

    /** One-time matched-fixture boundary, invoked after old bridge retirement and before new start. */
    public void resetForTrial(double x, double y, double z, long bodySeed) {
        if (bodySeed < 0) { throw new IllegalArgumentException("body trial seed must be nonnegative"); }
        resetForFixture(x, y, z, 0.0F);
        horizontalCollision = false;
        verticalCollision = false;
        verticalCollisionBelow = false;
        setOnGround(false);
        events = new EventAccumulator();
        eventDataWarningLogged = false;
        bodyPrograms.resetForTrial(bodySeed);
    }

    /** Clear only session ownership/history. Position, orientation, velocity and world time persist. */
    public void resetNeuralSessionPreservingBody() {
        if (level().isClientSide()) {
            throw new IllegalStateException("neural session reset is server-only");
        }
        releaseAndResetBodyPrograms();
        updatePresentation(ControllerFault.STARTING, PrimitiveAction.NEUTRAL);
        lastActuation = neutralReceipt(this, RawPrimitiveAction.ZERO);
        lastAcceptedCommand = null;
        sensorHistory.clear();
    }

    /**
     * ADR 0014: the body declares a symmetric hover envelope. There is no ambient
     * gravity term, so the vertical axis carries exactly the same bounded
     * commanded-acceleration authority as the horizontal axes and neutral motors
     * drift to a stop and hover. No vanilla path re-adds gravity: this entity
     * extends {@code Entity} directly, {@code Entity.tick()} never calls
     * {@code applyGravity()}, and the inherited {@code getDefaultGravity()} is 0.
     */
    private void integratePassivePhysics() {
        Vec3 velocity = getDeltaMovement();
        move(MoverType.SELF, velocity);
        Vec3 movedVelocity = getDeltaMovement();
        if (onGround() && movedVelocity.y < 0.0) {
            // Defensive floor invariant: a grounded fly never carries downward
            // velocity, even across blocks whose collision response does not
            // zero it. Reachable only through commanded descent into contact.
            movedVelocity = new Vec3(movedVelocity.x, 0.0, movedVelocity.z);
        }
        double horizontalDrag = isInWater() ? WATER_DRAG : AIR_HORIZONTAL_DRAG;
        setDeltaMovement(
                movedVelocity.x * horizontalDrag,
                movedVelocity.y * (isInWater() ? WATER_DRAG : AIR_VERTICAL_DRAG),
                movedVelocity.z * horizontalDrag);
    }

    /**
     * Returns the body-program lane to its spawn state, logging the loss of scripted authority
     * first so a reset or a chunk reload can never drop a running program silently. The neural
     * service is not resumed by this: it stays paused until the operator resumes it, and
     * {@code /neurocraftfly status} shows the mismatch.
     */
    private void releaseAndResetBodyPrograms() {
        if (!level().isClientSide) {
            FlyControllerRuntime.resetBrainDisplay(this);
        }
        entityData.set(BRAIN_DISPLAY, FlyControllerRuntime.UNAVAILABLE_BRAIN_DISPLAY);
        entityData.set(EVENT_DISPLAY, "");
        if (bodyPrograms.authority() != MotorAuthority.NEURAL) {
            bodyPrograms.releaseScripted(level().getGameTime(), "reset");
        }
        bodyPrograms.reset();
    }

    private void rememberSensorSnapshot(PhysicalSensorSnapshot snapshot) {
        if (sensorHistory.size() == SENSOR_HISTORY_CAPACITY) {
            sensorHistory.removeFirst();
        }
        sensorHistory.addLast(snapshot);
    }

    private void updatePresentation(ControllerFault state, PrimitiveAction action) {
        entityData.set(CONTROLLER_STATE, state.ordinal());
        entityData.set(FORWARD_THRUST, (float) action.forwardThrust());
        entityData.set(LATERAL_THRUST, (float) action.lateralThrust());
        entityData.set(VERTICAL_THRUST, (float) action.verticalThrust());
        entityData.set(YAW_TORQUE, (float) action.yawTorque());
        entityData.set(PITCH_TORQUE, (float) action.pitchTorque());
        entityData.set(BODY_PROGRAM, bodyPrograms.channel().packed());
        entityData.set(BODY_PROGRAM_PHASE, bodyPrograms.presentationPhase());
        entityData.set(READOUT_CHANNEL, bodyPrograms.readoutChannel().packed());
    }

    private static ActuationReceipt neutralReceipt() {
        return neutralReceipt(RawPrimitiveAction.ZERO);
    }

    private static ActuationReceipt neutralReceipt(RawPrimitiveAction rawAction) {
        return new ActuationReceipt(
                rawAction,
                PrimitiveAction.NEUTRAL,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0);
    }

    private static ActuationReceipt neutralReceipt(
            FlyEntity entity, RawPrimitiveAction rawAction) {
        Vec3 velocity = entity.getDeltaMovement();
        return new ActuationReceipt(
                rawAction,
                PrimitiveAction.NEUTRAL,
                velocity.x,
                velocity.y,
                velocity.z,
                velocity.x,
                velocity.y,
                velocity.z,
                entity.getYRot(),
                entity.getYRot(),
                entity.getXRot(),
                entity.getXRot());
    }
}
