package dev.neurocraft.fly.body;

import dev.neurocraft.fly.body.programs.ApproachProgram;
import dev.neurocraft.fly.body.programs.BankProgram;
import dev.neurocraft.fly.body.programs.CruiseProgram;
import dev.neurocraft.fly.body.programs.EscapeJumpProgram;
import dev.neurocraft.fly.body.programs.FeedProgram;
import dev.neurocraft.fly.body.programs.FleeProgram;
import dev.neurocraft.fly.body.programs.GroomProgram;
import dev.neurocraft.fly.body.programs.HoverProgram;
import dev.neurocraft.fly.body.programs.LandProgram;
import dev.neurocraft.fly.body.programs.WalkProgram;
import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import java.util.ArrayDeque;
import java.util.EnumMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Per-entity holder of the running body program, its modulation, and the motor authority
 * (ADR 0020 section 1). The only mutable thing in this package.
 *
 * <p><strong>The lanes are mutually exclusive by construction.</strong> {@link #advance} drives the
 * operator lane and returns empty unless scripted authority is held <em>and</em> the command
 * admitted on this tick carries {@code neutral = true}. Scripted authority itself is reachable only
 * through an accepted {@code PAUSE} acknowledgement staged by {@code StagedProgramAuthority}, and an
 * admitted non-neutral <em>{@code action}</em> drops the program the instant it arrives: against a
 * requested motor vector, the connectome always wins.
 *
 * <p>An admitted {@code program_action} is the one command that does <strong>not</strong> take the
 * body from the operator lane, because it requests no motors to be displaced by and has an honest
 * receipt of its own; see {@link #advanceReadout} and {@link MotorAuthority}.
 *
 * <p>{@link #advanceReadout} drives the readout lane and is reachable only from an admitted
 * {@code program_action} frame, which exists only while the neural service is RUNNING and stepping.
 * A neutral admitted command — the paused service's signature — releases that authority instead.
 *
 * <p><strong>Nothing here selects a program.</strong> Selection arrives from outside: an operator
 * command, or a declared hand-chosen mapping over labeled readouts running in the neural service
 * (ADR 0019). This class executes a selection and reports what it executed; it never makes one.
 * {@link BodyProgram#canEnter} is a precondition that can only refuse a selection someone else made,
 * and a refusal falls back to HOVER.
 */
public final class BodyProgramRuntime {
    /** Bound on the retained transition ledger; it is an operator aid, not evidence. */
    public static final int TRANSITION_LEDGER_CAPACITY = 32;

    private static final Logger LOGGER = LoggerFactory.getLogger("neurocraft-fly-body");

    /**
     * The registry. Programs are stateless pure functions, so one shared instance per program is
     * correct and every fly resolves the same object.
     */
    private static final Map<BodyProgramId, BodyProgram> PROGRAMS = buildRegistry();

    private final ArrayDeque<String> transitionLedger = new ArrayDeque<>();

    private MotorAuthority authority = MotorAuthority.NEURAL;
    private SelectionSource selectionSource = SelectionSource.DEFAULT;
    private Modulation modulation = Modulation.NEUTRAL;
    private ProgramParameters parameters = ProgramParameters.NEUTRAL;
    private ReadoutValues lastReadout = ReadoutValues.ZERO;
    private BodyProgramId lastRequestedProgram;
    private String policyHash = "";
    private BodyProgramState state = BodyProgramState.entering(BodyProgramId.HOVER, 0L);
    private float presentationPhase;
    private long programOwnedTicks;
    private long readoutOwnedTicks;
    private long readoutStalledTicks;
    private long readoutRefusals;
    private long transitions;
    private OptionalLong trialWanderSeed = OptionalLong.empty();

    /** The programs that actually exist. The rest of {@link BodyProgramId} is declared, not built. */
    public static Set<BodyProgramId> registeredPrograms() {
        return PROGRAMS.keySet();
    }

    public MotorAuthority authority() {
        return authority;
    }

    public SelectionSource selectionSource() {
        return selectionSource;
    }

    public BodyProgramState state() {
        return state;
    }

    /** Explicit trial override, absent during ordinary execution. */
    public OptionalLong trialWanderSeed() {
        return trialWanderSeed;
    }

    /**
     * Current raw phase seed for the diagnostic ledger. CRUISE reduces this modulo its declared
     * 72,000-tick seed period; this value is not a replacement for the actual entry timestamp.
     */
    public long effectiveWanderSeed() {
        return state.wanderSeed();
    }

    /** The authoritative modulation, exact. The synchronized channel publishes it quantized. */
    public Modulation modulation() {
        return modulation;
    }

    /**
     * The authoritative signed program parameters, exact. The synchronized channel publishes the
     * turn bias quantized to tenths.
     *
     * <p>It is the readout lane's input alone: {@link #advance} hands every program
     * {@link ProgramParameters#NEUTRAL}, and {@link #engageScripted} restores it, so an operator
     * program can never inherit a bias the mapping asked for. What a program does with it is the
     * program's declaration — CRUISE reads it, HOVER and ESCAPE_JUMP ignore it.
     */
    public ProgramParameters parameters() {
        return parameters;
    }

    /** The readout scalars the last admitted program frame carried; zero when none is driving. */
    public ReadoutValues lastReadout() {
        return lastReadout;
    }

    /** Identity of the declared mapping that selected the running program; empty when none is. */
    public String policyHash() {
        return policyHash;
    }

    public float presentationPhase() {
        return presentationPhase;
    }

    /** Ticks a scripted program has actually owned since this runtime was created or reset. */
    public long programOwnedTicks() {
        return programOwnedTicks;
    }

    /** Ticks the readout lane has actually owned; a subset of {@link #programOwnedTicks()}. */
    public long readoutOwnedTicks() {
        return readoutOwnedTicks;
    }

    /** Ticks READOUT authority was held but no command was admitted, so the body held neutral. */
    public long readoutStalledTicks() {
        return readoutStalledTicks;
    }

    /** Admitted program frames whose requested program did not run as asked. */
    public long readoutRefusals() {
        return readoutRefusals;
    }

    public long transitions() {
        return transitions;
    }

    /** Most recent transitions, oldest first. Bounded, human-readable, never an evidence surface. */
    public List<String> transitionLedger() {
        return List.copyOf(transitionLedger);
    }

    /** What the entity publishes this tick. */
    public BodyProgramChannel channel() {
        return new BodyProgramChannel(selectionSource, state.id(), modulation, parameters);
    }

    /** The driving readout scalars the entity publishes this tick, quantized to hundredths. */
    public ReadoutChannel readoutChannel() {
        return new ReadoutChannel(lastReadout);
    }

    /**
     * Engages scripted authority.
     *
     * <p>Initial authority comes from {@code StagedProgramAuthority.apply}, which refuses to run before an
     * accepted {@code PAUSE} acknowledgement has landed and the entity reaches the tick that
     * acknowledgement governs -- one action budget after its observation-stamped effective tick,
     * which is the tick the paused service's first neutral command applies on and is at or after
     * the first tick the acknowledgement can be polled at all. The GameTest
     * {@code scriptedAuthorityRequiresAnAcceptedPause} pins that gate. Subsequent operator changes
     * may use the runtime's retained, identity-bound PAUSE grant at an exact next-tick boundary
     * (ADR 0029); execution still requires an admitted neutral command. An unregistered program is
     * refused here as well, so no caller can engage a program that does not exist.
     */
    public void engageScripted(
            BodyProgramId program, Modulation requested, SelectionSource source, long gameTick) {
        Objects.requireNonNull(program, "program");
        Objects.requireNonNull(requested, "requested");
        Objects.requireNonNull(source, "source");
        if (source == SelectionSource.DEFAULT) {
            throw new IllegalArgumentException("DEFAULT is not a selection; it is the fallback");
        }
        authority = MotorAuthority.SCRIPTED;
        selectionSource = source;
        modulation = requested;
        // The operator command carries no signed parameter, so the operator lane runs at zero bias
        // by construction rather than by relying on whoever held the body last having released it.
        // Without this, a PAUSE taken straight from READOUT would leave the mapping's turn bias on
        // the runtime, steering the operator's program and publishing that bias on the HUD under an
        // OPERATOR label -- a blended lane, which is the one defect this project must never ship.
        parameters = ProgramParameters.NEUTRAL;
        if (!PROGRAMS.containsKey(program)) {
            transition(BodyProgramId.HOVER, gameTick, source, "refused");
            return;
        }
        transition(program, gameTick, source, "selected");
    }

    /**
     * Drops scripted authority and returns the body to the neural lane and the HOVER default.
     *
     * <p>The transition is logged against the source that was <em>losing</em> authority, not
     * against the DEFAULT it falls back to, so a {@code reason=preempted} line names the lane the
     * connectome took the body away from.
     */
    public void releaseScripted(long gameTick, String reason) {
        Objects.requireNonNull(reason, "reason");
        SelectionSource releasedFrom = selectionSource;
        authority = MotorAuthority.NEURAL;
        selectionSource = SelectionSource.DEFAULT;
        modulation = Modulation.NEUTRAL;
        parameters = ProgramParameters.NEUTRAL;
        lastReadout = ReadoutValues.ZERO;
        lastRequestedProgram = null;
        policyHash = "";
        transition(BodyProgramId.HOVER, gameTick, releasedFrom, reason);
    }

    /**
     * Returns the runtime to its spawn state; used by fixture reset and NBT reload.
     *
     * <p>This clears the in-memory ledger, so callers that may be holding scripted authority must
     * call {@link #releaseScripted} first — {@code FlyEntity} does — and the release line survives
     * on the server logger even though the ledger does not.
     */
    public void reset() {
        trialWanderSeed = OptionalLong.empty();
        authority = MotorAuthority.NEURAL;
        selectionSource = SelectionSource.DEFAULT;
        modulation = Modulation.NEUTRAL;
        parameters = ProgramParameters.NEUTRAL;
        lastReadout = ReadoutValues.ZERO;
        lastRequestedProgram = null;
        policyHash = "";
        state = BodyProgramState.entering(BodyProgramId.HOVER, 0L);
        presentationPhase = 0.0F;
        programOwnedTicks = 0L;
        readoutOwnedTicks = 0L;
        readoutStalledTicks = 0L;
        readoutRefusals = 0L;
        transitions = 0L;
        transitionLedger.clear();
    }

    /**
     * Performs the ordinary complete body-runtime reset and latches a deterministic trial seed.
     *
     * <p>Every subsequent program entry retains its real game tick while carrying this independent
     * seed. Only CRUISE's existing wander and obstacle tie-break consume it. Program exits and
     * authority changes retain the latch; an ordinary {@link #reset()} clears it. The caller must
     * arrange the physical and neural trial reset separately, then call this after other body
     * resets and before admitting the new trial's commands. No authority is granted here.
     *
     * @param wanderSeed non-negative raw seed, before CRUISE's existing modulo reduction
     */
    public void resetForTrial(long wanderSeed) {
        if (wanderSeed < 0) {
            throw new IllegalArgumentException("wanderSeed must be non-negative");
        }
        reset();
        trialWanderSeed = OptionalLong.of(wanderSeed);
        state = BodyProgramState.entering(BodyProgramId.HOVER, 0L, wanderSeed);
    }

    /**
     * Advances the operator lane for one authoritative tick, and resolves what a tick with no
     * admitted program frame means for the readout lane.
     *
     * <p>Under {@link MotorAuthority#NEURAL} it declines every combination of its two flags, which
     * is what makes the pre-program tick path byte-identical; that is pinned tick by tick against a
     * fixture recorded on {@code main}.
     *
     * @param snapshot this tick's authoritative sensor sample
     * @param commandAdmitted whether the validator admitted a command on this tick
     * @param commandNeutral whether that admitted command is a neutral {@code action}
     * @return the step whose motors own this tick, or empty when the neural lane owns it
     */
    public Optional<BodyProgramStep> advance(
            PhysicalSensorSnapshot snapshot, boolean commandAdmitted, boolean commandNeutral) {
        return advance(BodyProgramContext.sensorsOnly(snapshot), commandAdmitted, commandNeutral);
    }

    /** Complete scripted world context; the caller samples it alongside this tick's sensors. */
    public Optional<BodyProgramStep> advance(
            BodyProgramContext context, boolean commandAdmitted, boolean commandNeutral) {
        Objects.requireNonNull(context, "context");
        PhysicalSensorSnapshot snapshot = context.snapshot();
        if (authority == MotorAuthority.READOUT) {
            // The readout lane never owns a tick from here: it is driven by advanceReadout, from an
            // admitted program frame. Reaching this method under READOUT means no program frame was
            // admitted on this tick, and the three cases below are the whole of what that can mean.
            if (commandAdmitted && commandNeutral) {
                // A neutral admitted command that left the neural cursor where it was is the paused
                // service's signature. The connectome is not stepping, so no readout can be
                // selecting, and a body still running a readout-selected program would be moving on
                // a selection nothing is making. Drop to the neural lane.
                releaseScripted(snapshot.gameTick(), "paused");
                return Optional.empty();
            }
            if (commandAdmitted) {
                // Unreachable by construction: the validator's lane latch refuses a non-neutral raw
                // action in a PROGRAM session. Kept, and pinned, because if it ever were reached the
                // answer is the same one the operator lane gives -- the connectome always wins.
                releaseScripted(snapshot.gameTick(), "preempted");
                return Optional.empty();
            }
            // No admitted command at all: authority is retained and the program's own state is
            // frozen, so a stalled window costs a neutral tick rather than a tick of a program
            // nobody selected this tick.
            readoutStalledTicks++;
            return Optional.empty();
        }
        if (authority != MotorAuthority.SCRIPTED) {
            return Optional.empty();
        }
        if (commandAdmitted && !commandNeutral) {
            // The hard interlock. A non-neutral command means the connectome is driving, and a
            // program that displaced it would leave encodeActuationReceipt with no honest word
            // for the difference between requested and applied motors.
            releaseScripted(snapshot.gameTick(), "preempted");
            return Optional.empty();
        }
        if (!commandAdmitted) {
            // No admitted command, so no admitted command carries neutral = true. Authority is
            // retained; this tick is simply not owned and the body holds neutral.
            return Optional.empty();
        }

        // ProgramParameters.NEUTRAL, not the field: the operator lane has no signed input, and
        // passing the literal makes that structural rather than an invariant to be maintained.
        return Optional.of(runOwnedTick(context, ProgramParameters.NEUTRAL).step());
    }

    /**
     * Executes one readout-selected body program for one authoritative tick.
     *
     * <p><strong>This method selects nothing.</strong> Every input below arrived on an admitted
     * {@code program_action} frame, chosen by a declared hand-chosen mapping over labeled readouts
     * running in the neural service (ADR 0019). What happens here is execution and honest
     * reporting: the returned {@link ProgramExecution} states what actually ran, under what, and
     * with which motors, so the receipt can never assert a program ran that did not.
     *
     * <p>There is deliberately <strong>no</strong> mod-side dwell or hysteresis. Anti-flapping is
     * the mapping's job; two anti-flapping layers would make the observed behaviour a function of
     * neither.
     *
     * <p>The three refusal shapes:
     *
     * <ul>
     *   <li>an operator hold owns the body — the operator's own program runs this tick and is what
     *       the returned execution reports, with reason {@code operator}. Authority is untouched:
     *       the operator lane wins, and it wins visibly;
     *   <li>the requested program has no registry entry — the declared HOVER fallback runs, reason
     *       {@code unregistered};
     *   <li>the requested program declines to enter — HOVER again, reason {@code precondition}.
     * </ul>
     *
     * @param policyHash identity of the declared mapping that made the selection, carried only so
     *     the transition log and the operator status line can name it
     * @return what ran on this tick; empty is never returned, because an admitted program frame
     *     always produces an honest receipt
     */
    public Optional<ProgramExecution> advanceReadout(
            PhysicalSensorSnapshot snapshot,
            BodyProgramId requestedProgram,
            Modulation requestedModulation,
            ProgramParameters requestedParameters,
            ReadoutValues readout,
            String policyHash) {
        return advanceReadout(BodyProgramContext.sensorsOnly(snapshot), requestedProgram,
                requestedModulation, requestedParameters, readout, policyHash);
    }

    /** Complete scripted context does not enter the controller's observation or select a program. */
    public Optional<ProgramExecution> advanceReadout(
            BodyProgramContext context,
            BodyProgramId requestedProgram,
            Modulation requestedModulation,
            ProgramParameters requestedParameters,
            ReadoutValues readout,
            String policyHash) {
        Objects.requireNonNull(context, "context");
        PhysicalSensorSnapshot snapshot = context.snapshot();
        Objects.requireNonNull(requestedProgram, "requestedProgram");
        Objects.requireNonNull(requestedModulation, "requestedModulation");
        Objects.requireNonNull(requestedParameters, "requestedParameters");
        Objects.requireNonNull(readout, "readout");
        Objects.requireNonNull(policyHash, "policyHash");
        long gameTick = snapshot.gameTick();
        if (authority == MotorAuthority.SCRIPTED) {
            // The operator lane holds the body. The requested program never reaches it, so the
            // operator's program owns this tick exactly as it would have without the frame, and the
            // receipt reports that program, its modulation, and the motors it actually emitted.
            // The frame's signed parameter is refused with the rest of it: the operator's own
            // NEUTRAL is what runs, and the receipt echoes the value the tick was actually run
            // under rather than the field, which would be truthful only by invariant.
            OwnedTick operatorTick = runOwnedTick(context, ProgramParameters.NEUTRAL);
            readoutRefusals++;
            return Optional.of(new ProgramExecution(
                    operatorTick.ranProgram(),
                    modulation,
                    operatorTick.ranParameters(),
                    operatorTick.step().motors(),
                    Optional.of(ProgramRefusalReason.OPERATOR)));
        }
        boolean engaging = authority != MotorAuthority.READOUT;
        // The selection is the REQUESTED program, not the running one. Keying re-selection on
        // `state.id()` instead flapped twice per tick under a persistently refused request: the
        // fallback leaves HOVER running, HOVER is not the requested id, so the next identical
        // frame re-selected and was refused again -- two ledger lines per tick, and HOVER's own
        // clock reset every tick, which turned its bob into a constant climb.
        boolean newRequest = engaging || lastRequestedProgram != requestedProgram;
        authority = MotorAuthority.READOUT;
        selectionSource = SelectionSource.READOUT;
        // Set before the checks below, so a refused frame runs the HOVER fallback under the
        // modulation the mapping asked for. The receipt echoes exactly that.
        modulation = requestedModulation;
        parameters = requestedParameters;
        lastReadout = readout;
        this.policyHash = policyHash;
        lastRequestedProgram = requestedProgram;
        // Re-derived every tick against the REQUESTED program, so a receipt for an unchanging
        // refused request keeps saying PROGRAM_REFUSED with its reason instead of quietly
        // reporting the fallback as an applied selection.
        Optional<ProgramRefusalReason> refusal =
                refusalFor(requestedProgram, context, newRequest);
        if (!alreadyEnteredAsSuccessor(requestedProgram, engaging)
                && (newRequest || (refusal.isEmpty() && state.id() != requestedProgram))) {
            // Either the mapping asked for something new, or the program it is still asking for
            // has exited on its own clock and must be re-entered. A repeated request that is
            // still refused logs nothing further: the first refusal already said so.
            transition(requestedProgram, gameTick, SelectionSource.READOUT, "selected", readout);
        }
        OwnedTick owned = runOwnedTick(context, parameters);
        readoutOwnedTicks++;
        Optional<ProgramRefusalReason> reported =
                refusal.isPresent() ? refusal : owned.refusal();
        if (reported.isPresent()) {
            readoutRefusals++;
        }
        return Optional.of(new ProgramExecution(
                owned.ranProgram(),
                modulation,
                owned.ranParameters(),
                owned.step().motors(),
                reported));
    }

    /**
     * Whether the requested program is already entered because its predecessor exited into it on
     * the previous tick, so re-selecting it would log {@code X -> X} and say nothing.
     *
     * <p>This is the exit-into-successor case and only that case. Reaching {@link #advanceReadout}
     * with {@code ticksInProgram == 0} under retained READOUT authority has exactly one cause: an
     * exit transition ran at the <em>end</em> of the previous owned tick, after
     * {@link #runOwnedTick} had already advanced the state. Every other transition — a refusal, a
     * selection — is followed by a step that advances the counter off zero within the same tick.
     * The concrete sequence is ESCAPE_JUMP exiting to CRUISE on its own 24-tick clock while the
     * mapping's hold expires on the same boundary and asks for CRUISE next tick.
     *
     * <p>{@code engaging} is excluded deliberately. Engaging from the neural lane at HOVER and
     * asking for HOVER is also a self-transition, but there the ledger line is the record that the
     * readout lane took the body, and dropping it would remove an honesty surface rather than a
     * duplicate.
     */
    private boolean alreadyEnteredAsSuccessor(BodyProgramId requestedProgram, boolean engaging) {
        return !engaging
                && state.id() == requestedProgram
                && state.ticksInProgram() == 0;
    }

    /**
     * Why the requested program cannot run as asked, if it cannot.
     *
     * <p>Asked about the <em>requested</em> program rather than the running one, and asked again on
     * every admitted frame, so the answer tracks the world: a precondition that stops holding
     * releases the fallback on the next tick, and one that keeps holding keeps the receipt honest
     * without re-entering anything. A program already part-way through its own run is never asked
     * {@code canEnter} again, exactly as the operator lane never asks it.
     */
    private Optional<ProgramRefusalReason> refusalFor(
            BodyProgramId requestedProgram, BodyProgramContext context, boolean newRequest) {
        BodyProgram program = PROGRAMS.get(requestedProgram);
        if (program == null) {
            return Optional.of(ProgramRefusalReason.UNREGISTERED);
        }
        boolean entering =
                newRequest || state.id() != requestedProgram || state.ticksInProgram() == 0;
        if (entering && !program.canEnter(context)) {
            return Optional.of(ProgramRefusalReason.PRECONDITION);
        }
        return Optional.empty();
    }

    /**
     * The one place a body program is resolved and stepped, shared by both driving lanes so the
     * refusal rule, the state advance, the phase, the owned-tick count, and the exit transition are
     * written once. The refusal it reports is the same refusal the operator lane has always logged;
     * only the readout lane needs to name it on a wire receipt.
     *
     * <p>{@code programParameters} is an argument rather than the field because the two lanes
     * disagree about it and must keep disagreeing: the operator lane passes
     * {@link ProgramParameters#NEUTRAL}, the readout lane passes the admitted frame's.
     */
    private OwnedTick runOwnedTick(
            BodyProgramContext context, ProgramParameters programParameters) {
        PhysicalSensorSnapshot snapshot = context.snapshot();
        BodyProgram program = PROGRAMS.get(state.id());
        Optional<ProgramRefusalReason> refusal = Optional.empty();
        if (program == null) {
            transition(BodyProgramId.HOVER, snapshot.gameTick(), selectionSource, "refused");
            program = PROGRAMS.get(BodyProgramId.HOVER);
            refusal = Optional.of(ProgramRefusalReason.UNREGISTERED);
        } else if (state.ticksInProgram() == 0 && !program.canEnter(context)) {
            transition(BodyProgramId.HOVER, snapshot.gameTick(), selectionSource, "refused");
            program = PROGRAMS.get(BodyProgramId.HOVER);
            refusal = Optional.of(ProgramRefusalReason.PRECONDITION);
        }
        BodyProgramId ranProgram = state.id();
        BodyProgramStep step =
                Objects.requireNonNull(program).step(state, context, modulation, programParameters);

        state = step.next();
        presentationPhase = step.presentationPhase();
        programOwnedTicks++;
        Optional<BodyProgramId> successor = step.exitTo();
        if (successor.isPresent()) {
            transition(successor.orElseThrow(), snapshot.gameTick(), selectionSource, "exit");
        }
        return new OwnedTick(ranProgram, programParameters, step, refusal);
    }

    /**
     * One executed tick: the program that actually ran, the signed parameters it actually ran
     * under, its step, and why it was not the asked-for one.
     *
     * <p>{@code ranParameters} is carried rather than read back off the field because the two lanes
     * pass different values: the receipt has to be able to say what <em>this</em> tick ran under,
     * and reading the field would make an operator refusal's receipt truthful by invariant instead
     * of by construction.
     */
    private record OwnedTick(
            BodyProgramId ranProgram,
            ProgramParameters ranParameters,
            BodyProgramStep step,
            Optional<ProgramRefusalReason> refusal) {}

    private void transition(
            BodyProgramId to, long gameTick, SelectionSource source, String reason) {
        transition(to, gameTick, source, reason, null);
    }

    private void transition(
            BodyProgramId to,
            long gameTick,
            SelectionSource source,
            String reason,
            ReadoutValues readout) {
        BodyProgramId from = state.id();
        state = BodyProgramState.entering(to, gameTick, trialWanderSeed.orElse(gameTick));
        presentationPhase = 0.0F;
        transitions++;
        String line = String.format(
                Locale.ROOT,
                "BODY PROGRAM %s -> %s tick=%d source=%s reason=%s",
                from,
                to,
                gameTick,
                source,
                reason);
        if (readout != null) {
            // The readout lane names the numbers it was handed and the mapping that consumed them,
            // on the same line as the transition, so a log grep can never show a readout-selected
            // program without the declared mapping that selected it.
            line = line
                    + String.format(
                            Locale.ROOT,
                            " readout=ttmn:%.4f,dna01:%+.4f,wing:%+.4f policy=%s",
                            readout.ttmnJump(),
                            readout.dna01Turn(),
                            readout.wingCommon(),
                            abbreviatedPolicyHash());
        }
        if (transitionLedger.size() == TRANSITION_LEDGER_CAPACITY) {
            transitionLedger.removeFirst();
        }
        transitionLedger.addLast(line);
        LOGGER.info("{}", line);
    }

    /** First eight hex characters of the declared mapping identity; the status line reports all 64. */
    private String abbreviatedPolicyHash() {
        return policyHash.length() <= 8 ? policyHash : policyHash.substring(0, 8);
    }

    private static Map<BodyProgramId, BodyProgram> buildRegistry() {
        Map<BodyProgramId, BodyProgram> registry = new EnumMap<>(BodyProgramId.class);
        register(registry, new HoverProgram());
        register(registry, new CruiseProgram());
        register(registry, new EscapeJumpProgram());
        register(registry, new LandProgram());
        register(registry, new WalkProgram());
        register(registry, new BankProgram());
        register(registry, new FleeProgram());
        register(registry, new ApproachProgram());
        register(registry, new GroomProgram());
        register(registry, new FeedProgram());
        return Map.copyOf(registry);
    }

    private static void register(Map<BodyProgramId, BodyProgram> registry, BodyProgram program) {
        if (registry.put(program.id(), program) != null) {
            throw new IllegalStateException("duplicate body program registration");
        }
    }
}
