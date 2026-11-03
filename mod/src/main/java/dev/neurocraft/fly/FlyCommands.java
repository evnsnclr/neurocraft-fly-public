package dev.neurocraft.fly;

import static net.minecraft.commands.Commands.argument;
import static net.minecraft.commands.Commands.literal;

import com.mojang.brigadier.CommandDispatcher;
import com.mojang.brigadier.arguments.DoubleArgumentType;
import com.mojang.brigadier.arguments.IntegerArgumentType;
import com.mojang.brigadier.arguments.StringArgumentType;
import com.mojang.brigadier.builder.LiteralArgumentBuilder;
import com.mojang.brigadier.builder.RequiredArgumentBuilder;
import com.mojang.brigadier.context.CommandContext;
import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.BodyProgramRuntime;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ReadoutValues;
import dev.neurocraft.fly.control.CommandValidator;
import dev.neurocraft.fly.control.ControlOperation;
import dev.neurocraft.fly.control.ComparisonLanePresentation;
import dev.neurocraft.fly.control.LaneCatalog;
import dev.neurocraft.fly.entity.FlyEntity;
import dev.neurocraft.fly.presentation.FlyTrailPresentation;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.stream.Collectors;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.minecraft.commands.CommandSourceStack;
import net.minecraft.commands.SharedSuggestionProvider;
import net.minecraft.network.chat.Component;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.entity.EntitySpawnReason;
import net.minecraft.world.phys.Vec3;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Operator-only experiment lifecycle commands; none can issue semantic movement intent. */
public final class FlyCommands {
    private static final Logger LOGGER = LoggerFactory.getLogger("neurocraft-fly-commands");
    /** How many hex digits of the mapping's identity the compact status shows. */
    private static final int POLICY_HASH_PREFIX_LENGTH = 12;

    private FlyCommands() {}

    public static void register() {
        CommandRegistrationCallback.EVENT.register(
                (dispatcher, registryAccess, environment) -> register(dispatcher));
    }

    private static void register(CommandDispatcher<CommandSourceStack> dispatcher) {
        dispatcher.register(literal("neurocraftfly")
                .requires(source -> source.hasPermission(2))
                .then(literal("spawn").executes(FlyCommands::spawn))
                .then(literal("trials")
                        // withSource returns this only when the underlying source is exactly
                        // the server console; player and command-block sources are excluded.
                        .requires(source -> source.getEntity() == null
                                && source.withSource(source.getServer()) == source)
                        .executes(FlyCommands::trialStatus)
                        .then(literal("start").executes(FlyCommands::trialStart))
                        .then(literal("stop").executes(FlyCommands::trialStop)))
                .then(literal("control")
                        .then(argument("lane", StringArgumentType.word())
                                .suggests((context, builder) -> SharedSuggestionProvider.suggest(
                                        LaneCatalog.NAMES, builder))
                                .executes(FlyCommands::comparison)))
                .then(control("pause", ControlOperation.PAUSE))
                .then(control("resume", ControlOperation.RESUME))
                .then(control("single_step", ControlOperation.SINGLE_STEP))
                .then(literal("reset").executes(FlyCommands::reset))
                .then(optionalHashContentControl("record_start", ControlOperation.RECORD_START))
                .then(optionalHashContentControl("record_stop", ControlOperation.RECORD_STOP))
                .then(literal("replay_start")
                        .then(argument("content_id", StringArgumentType.word())
                                .then(argument("content_hash", StringArgumentType.word())
                                        .executes(context -> request(
                                                context,
                                                ControlOperation.REPLAY_START,
                                                StringArgumentType.getString(
                                                        context, "content_id"),
                                                StringArgumentType.getString(
                                                        context, "content_hash"))))))
                .then(optionalHashContentControl("replay_stop", ControlOperation.REPLAY_STOP))
                .then(hashedContentControl("lesion_apply", ControlOperation.LESION_APPLY))
                .then(optionalHashContentControl("lesion_clear", ControlOperation.LESION_CLEAR))
                .then(hashedContentControl(
                        "stimulation_apply", ControlOperation.STIMULATION_APPLY))
                .then(optionalHashContentControl(
                        "stimulation_clear", ControlOperation.STIMULATION_CLEAR))
                .then(literal("trail")
                        .executes(FlyCommands::trailToggle)
                        .then(literal("on").executes(FlyCommands::trailOn))
                        .then(literal("off").executes(FlyCommands::trailOff))
                        .then(literal("interval")
                                .then(argument("ticks", IntegerArgumentType.integer(
                                                FlyTrailPresentation.MIN_INTERVAL_TICKS,
                                                FlyTrailPresentation.MAX_INTERVAL_TICKS))
                                        .executes(FlyCommands::trailInterval))))
                .then(programTree())
                .then(literal("status")
                        .executes(FlyCommands::statusCompact)
                        .then(literal("full").executes(FlyCommands::statusFull)))
                .then(literal("inspect").executes(FlyCommands::statusFull)));
    }

    /**
     * {@code /neurocraftfly program <id> [speed] [turn] [amplitude] [urgency]}, {@code release},
     * {@code status} (ADR 0020); bare {@code program} prints that usage.
     *
     * <p>This is the <strong>scripted</strong> lane and it is the declared negative control for
     * stage 4: it runs a body program at its written form so a readout's effect is measurable as a
     * departure from this baseline. Selecting a program pauses the neural service through the
     * existing acknowledged control channel and engages only on the accepted acknowledgement;
     * nothing here is a movement command to the connectome, and the connectome is not driving
     * while a program runs.
     */
    private static LiteralArgumentBuilder<CommandSourceStack> programTree() {
        return literal("program")
                .executes(FlyCommands::programUsage)
                .then(literal("release").executes(FlyCommands::programRelease))
                .then(literal("status").executes(FlyCommands::programStatus))
                .then(argument("program", StringArgumentType.word())
                        .suggests((context, builder) -> SharedSuggestionProvider.suggest(
                                BodyProgramRuntime.registeredPrograms().stream()
                                        .map(id -> id.name().toLowerCase(Locale.ROOT))
                                        .toList(),
                                builder))
                        .executes(context -> program(context, 0))
                        .then(scalar("speed")
                                .executes(context -> program(context, 1))
                                .then(scalar("turn")
                                        .executes(context -> program(context, 2))
                                        .then(scalar("amplitude")
                                                .executes(context -> program(context, 3))
                                                .then(scalar("urgency")
                                                        .executes(context -> program(context, 4)))))));
    }

    private static RequiredArgumentBuilder<CommandSourceStack, Double> scalar(String name) {
        return argument(name, DoubleArgumentType.doubleArg(0.0, 1.0));
    }

    /**
     * Brigadier has no optional arguments, so the four modulation scalars nest and each level is
     * executable. {@code supplied} is how many of them this level parsed; the rest keep their
     * neutral defaults, which are "the program exactly as written".
     */
    private static Modulation modulation(CommandContext<CommandSourceStack> context, int supplied) {
        Modulation neutral = Modulation.NEUTRAL;
        return new Modulation(
                supplied >= 1 ? DoubleArgumentType.getDouble(context, "speed") : neutral.speed(),
                supplied >= 2 ? DoubleArgumentType.getDouble(context, "turn") : neutral.turnRate(),
                supplied >= 3
                        ? DoubleArgumentType.getDouble(context, "amplitude")
                        : neutral.amplitude(),
                supplied >= 4
                        ? DoubleArgumentType.getDouble(context, "urgency")
                        : neutral.urgency());
    }

    /**
     * Bare {@code /neurocraftfly program} prints usage instead of Brigadier's "Unknown or
     * incomplete command, see below for error", which is what the operator got when they typed
     * exactly that (lane L, 2026-09-04). The names come from the registry, so the text cannot
     * drift from it.
     */
    private static int programUsage(CommandContext<CommandSourceStack> context) {
        String usage = programUsageText();
        context.getSource().sendSuccess(() -> Component.literal(usage), false);
        return 1;
    }

    /** The registered program names and the two verbs, so the operator never has to guess them. */
    private static String programUsageText() {
        String names = BodyProgramRuntime.registeredPrograms().stream()
                .sorted()
                .map(id -> id.name().toLowerCase(Locale.ROOT))
                .collect(Collectors.joining("|"));
        return "Usage: /neurocraftfly program <" + names + "> [speed] [turn] [amplitude] [urgency]"
                + " — run that scripted body program (first selection queues neural PAUSE; later selections"
                + " reuse acknowledged SCRIPTED authority; the connectome is"
                + " not driving while it runs) · /neurocraftfly program release — hand the body"
                + " back (queues a neural RESUME) · /neurocraftfly program status — the exact"
                + " server-side body-program state";
    }

    private static int program(CommandContext<CommandSourceStack> context, int suppliedScalars) {
        CommandSourceStack source = context.getSource();
        String requested = StringArgumentType.getString(context, "program");
        Optional<BodyProgramId> program = BodyProgramRuntime.registeredPrograms().stream()
                .filter(id -> id.name().equalsIgnoreCase(requested))
                .findFirst();
        if (program.isEmpty()) {
            source.sendFailure(Component.literal(
                    "No such body program is implemented: " + requested + "; implemented="
                            + BodyProgramRuntime.registeredPrograms()));
            return 0;
        }
        List<FlyEntity> flies = loadedFlies(source);
        if (flies.size() != 1) {
            source.sendFailure(Component.literal(
                    "Body programs require exactly one loaded NeuroCraft fly; found "
                            + flies.size()));
            return 0;
        }
        Modulation modulation = modulation(context, suppliedScalars);
        if (!FlyControllerRuntime.requestProgramAuthority(
                flies.getFirst(),
                source.getLevel().getGameTime(),
                program.orElseThrow(),
                modulation,
                message -> source.sendSuccess(() -> Component.literal(message), true),
                message -> source.sendFailure(Component.literal(message)))) {
            source.sendFailure(Component.literal(
                    "Scripted body-program authority was not queued; another control transaction"
                            + " is pending or the current authority/bridge refused the request"));
            return 0;
        }
        source.sendSuccess(
                () -> Component.literal("Queued scripted body program " + program.orElseThrow()
                        + "; " + FlyControllerRuntime.programAuthorityStatus()
                        + ". This is scripted presentation: the connectome is not driving while it runs."),
                true);
        return 1;
    }

    private static int programRelease(CommandContext<CommandSourceStack> context) {
        CommandSourceStack source = context.getSource();
        List<FlyEntity> flies = loadedFlies(source);
        if (flies.size() != 1) {
            source.sendFailure(Component.literal(
                    "Body programs require exactly one loaded NeuroCraft fly; found "
                            + flies.size()));
            return 0;
        }
        if (!FlyControllerRuntime.releaseProgramAuthority(
                flies.getFirst(),
                source.getLevel().getGameTime(),
                message -> source.sendSuccess(() -> Component.literal(message), true),
                message -> source.sendFailure(Component.literal(message)))) {
            source.sendFailure(
                    Component.literal("Scripted body-program release was not queued"));
            return 0;
        }
        source.sendSuccess(
                () -> Component.literal(
                        "Queued neural RESUME; scripted authority drops on the accepted ACK"),
                true);
        return 1;
    }

    private static int programStatus(CommandContext<CommandSourceStack> context) {
        CommandSourceStack source = context.getSource();
        List<FlyEntity> flies = loadedFlies(source);
        if (flies.size() != 1) {
            source.sendSuccess(
                    () -> Component.literal("NeuroCraft body programs: loaded_flies="
                            + flies.size() + " implemented="
                            + BodyProgramRuntime.registeredPrograms()),
                    false);
            return 1;
        }
        String report = bodyProgramReport(flies.getFirst());
        source.sendSuccess(() -> Component.literal("NeuroCraft " + report), false);
        return 1;
    }

    /**
     * The exact server-side body-program state. The HUD shows a quantized rendering of this.
     *
     * <p>The readout fields are full precision, unlike the channel's published hundredths, and they
     * name the declared mapping that consumed them. A readout-selected program is never reported
     * without its {@code policy_hash}: the mapping is engineering (ADR 0019), and the hash is what
     * makes that checkable rather than a claim.
     */
    private static String bodyProgramReport(FlyEntity fly) {
        BodyProgramRuntime runtime = fly.bodyPrograms();
        Modulation modulation = runtime.modulation();
        ReadoutValues readout = runtime.lastReadout();
        return String.format(
                Locale.ROOT,
                "authority=%s program=%s source=%s since_tick=%d transitions=%d "
                        + "program_owned_ticks=%d modulation=(speed=%.6f,turn=%.6f,"
                        + "amplitude=%.6f,urgency=%.6f) authority_txn=%s lane=%s "
                        + "readout_owned_ticks=%d readout_stalled_ticks=%d readout_refusals=%d "
                        + "last_readout=(ttmn=%.6f,dna01=%.6f,wing=%.6f) turn_bias=%.6f "
                        + "policy_hash=%s program_receipts_sent=%d",
                runtime.authority(),
                runtime.state().id(),
                runtime.selectionSource(),
                runtime.state().enteredGameTick(),
                runtime.transitions(),
                runtime.programOwnedTicks(),
                modulation.speed(),
                modulation.turnRate(),
                modulation.amplitude(),
                modulation.urgency(),
                FlyControllerRuntime.programAuthorityStatus(),
                commandLane(fly),
                runtime.readoutOwnedTicks(),
                runtime.readoutStalledTicks(),
                runtime.readoutRefusals(),
                readout.ttmnJump(),
                readout.dna01Turn(),
                readout.wingCommon(),
                runtime.parameters().turnBias(),
                runtime.policyHash().isEmpty() ? "none" : runtime.policyHash(),
                FlyControllerRuntime.sentProgramReceiptCount());
    }

    /** The session's latched driving frame kind; {@code none} before its first command. */
    private static String commandLane(FlyEntity fly) {
        return FlyControllerRuntime.commandLane(fly)
                .map(Enum::name)
                .orElse("none");
    }

    private static LiteralArgumentBuilder<CommandSourceStack> control(
            String name, ControlOperation operation) {
        return literal(name).executes(context -> request(context, operation, null, null));
    }

    private static LiteralArgumentBuilder<CommandSourceStack> hashedContentControl(
            String name, ControlOperation operation) {
        return literal(name).then(argument("content_id", StringArgumentType.word())
                .then(argument("content_hash", StringArgumentType.word())
                        .executes(context -> request(
                                context,
                                operation,
                                StringArgumentType.getString(context, "content_id"),
                                StringArgumentType.getString(context, "content_hash")))));
    }

    private static LiteralArgumentBuilder<CommandSourceStack> optionalHashContentControl(
            String name, ControlOperation operation) {
        return literal(name).then(argument("content_id", StringArgumentType.word())
                .executes(context -> request(
                        context,
                        operation,
                        StringArgumentType.getString(context, "content_id"),
                        null))
                .then(argument("content_hash", StringArgumentType.word())
                        .executes(context -> request(
                                context,
                                operation,
                                StringArgumentType.getString(context, "content_id"),
                                StringArgumentType.getString(context, "content_hash")))));
    }

    private static int spawn(CommandContext<CommandSourceStack> context) {
        CommandSourceStack source = context.getSource();
        ServerLevel level = source.getLevel();
        boolean flyExists = false;
        for (ServerLevel candidate : source.getServer().getAllLevels()) {
            if (!candidate.getEntities(NeuroCraftFly.FLY, fly -> true).isEmpty()) {
                flyExists = true;
                break;
            }
        }
        if (flyExists) {
            source.sendFailure(Component.literal(
                    "Refusing to spawn: a loaded dimension already contains a NeuroCraft fly"));
            return 0;
        }
        FlyEntity fly = NeuroCraftFly.FLY.create(level, EntitySpawnReason.COMMAND);
        if (fly == null) {
            source.sendFailure(Component.literal("Could not construct the NeuroCraft fly"));
            return 0;
        }
        FlyControllerRuntime.configuredEntityUuid().ifPresent(fly::setUUID);
        Vec3 position = source.getPosition();
        fly.resetForFixture(position.x, position.y + 1.0, position.z, source.getRotation().y);
        if (!level.addFreshEntity(fly)) {
            source.sendFailure(Component.literal("Could not add the NeuroCraft fly to the level"));
            return 0;
        }
        source.sendSuccess(
                () -> Component.literal("Spawned one NeuroCraft fly with entity_id "
                        + fly.getStringUUID()),
                true);
        return 1;
    }

    private static int reset(CommandContext<CommandSourceStack> context) {
        CommandSourceStack source = context.getSource();
        ServerLevel level = source.getLevel();
        List<? extends FlyEntity> flies = level.getEntities(NeuroCraftFly.FLY, fly -> true);
        if (flies.size() != 1) {
            source.sendFailure(Component.literal(
                    "Reset requires exactly one loaded NeuroCraft fly; found " + flies.size()));
            return 0;
        }
        Vec3 position = source.getPosition();
        if (!FlyControllerRuntime.requestReset(
                flies.getFirst(),
                level.getGameTime(),
                position.x,
                position.y + 1.0,
                position.z,
                source.getRotation().y,
                message -> source.sendSuccess(() -> Component.literal(message), true),
                message -> source.sendFailure(Component.literal(message)))) {
            source.sendFailure(Component.literal("Neural reset request was not queued"));
            return 0;
        }
        source.sendSuccess(
                () -> Component.literal("Queued neural RESET; world mutation awaits its ACK"),
                true);
        return 1;
    }

    /**
     * {@code /neurocraftfly status}: the compact form, sized to sit beside the HUD rather than over
     * it. The full line ({@link #fullStatusLine}) runs to about 1,100 characters and wraps to some
     * twenty chat lines, which covered the HUD for ten seconds each time the operator asked (lane
     * L, 2026-09-04). It is still written to the server log here, and {@code status full} or
     * {@code inspect} sends it to chat on request.
     */
    private static int statusCompact(CommandContext<CommandSourceStack> context) {
        CommandSourceStack source = context.getSource();
        List<FlyEntity> flies = loadedFlies(source);
        if (flies.size() != 1) {
            return reportFlyCount(source, flies.size());
        }
        FlyEntity fly = flies.getFirst();
        LOGGER.info("{} ran /neurocraftfly status: {}", source.getTextName(), fullStatusLine(fly));
        String compact = compactStatusLine(fly);
        source.sendSuccess(() -> Component.literal(compact), false);
        return 1;
    }

    /** {@code /neurocraftfly status full} and {@code inspect}: the whole line, in chat. */
    private static int statusFull(CommandContext<CommandSourceStack> context) {
        CommandSourceStack source = context.getSource();
        List<FlyEntity> flies = loadedFlies(source);
        if (flies.size() != 1) {
            return reportFlyCount(source, flies.size());
        }
        String full = fullStatusLine(flies.getFirst());
        source.sendSuccess(() -> Component.literal(full), false);
        return 1;
    }

    private static int reportFlyCount(CommandSourceStack source, int loadedFlies) {
        source.sendSuccess(
                () -> Component.literal("NeuroCraft bridge=" + FlyControllerRuntime.state()
                        + " loaded_flies=" + loadedFlies),
                false);
        return 1;
    }

    /**
     * The operator's glance: who owns the body, which program, selected by what, since which tick,
     * how many handovers, how many frames the authority has retired as late, which frame kind the
     * session latched, and the first {@value #POLICY_HASH_PREFIX_LENGTH} hex digits of the declared
     * mapping's identity. Everything else is in {@link #fullStatusLine}.
     */
    private static String compactStatusLine(FlyEntity fly) {
        BodyProgramRuntime runtime = fly.bodyPrograms();
        String policyHash = runtime.policyHash();
        return String.format(
                Locale.ROOT,
                "NeuroCraft bridge=%s authority=%s program=%s source=%s since_tick=%d "
                        + "transitions=%d lateness_retired=%d lane=%s policy_hash_prefix=%s",
                FlyControllerRuntime.state(),
                runtime.authority(),
                runtime.state().id(),
                runtime.selectionSource(),
                runtime.state().enteredGameTick(),
                runtime.transitions(),
                CommandValidator.totalLatenessRetirements(),
                commandLane(fly),
                policyHash.isEmpty()
                        ? "none"
                        : policyHash.substring(
                                0, Math.min(POLICY_HASH_PREFIX_LENGTH, policyHash.length())));
    }

    /** The whole picture in one line: the bridge report and the exact body-program report. */
    private static String fullStatusLine(FlyEntity fly) {
        var action = fly.lastActuation().appliedAction();
        String neural = fly.lastAcceptedCommand()
                .map(command -> "command_seq=" + command.commandSequence() + " neural_steps="
                        + command.neuralStepStart() + ".." + command.neuralStepEnd()
                        + " state_digest=" + command.stateDigest())
                .orElse("command_seq=none neural_steps=none state_digest=none");
        String report = String.format(
                Locale.ROOT,
                "NeuroCraft bridge=%s loaded_flies=1 entity_id=%s dimension=%s "
                        + "position=(%.6f,%.6f,%.6f) applied_motors=(forward=%.6f,lateral=%.6f,"
                        + "vertical=%.6f,yaw=%.6f,pitch=%.6f) reset=%s %s lateness_retired=%d"
                        + " dormancy_reanchors=%d",
                FlyControllerRuntime.state(),
                fly.getStringUUID(),
                fly.level().dimension().location(),
                fly.getX(),
                fly.getY(),
                fly.getZ(),
                action.forwardThrust(),
                action.lateralThrust(),
                action.verticalThrust(),
                action.yawTorque(),
                action.pitchTorque(),
                FlyControllerRuntime.resetStatus(),
                neural,
                CommandValidator.totalLatenessRetirements(),
                CommandValidator.totalDormancyReanchors());
        ComparisonLanePresentation comparison = FlyControllerRuntime.comparisonFor(fly);
        return report + " " + bodyProgramReport(fly) + " comparison_status=" + comparison.status()
                + " attempted_lane=" + comparison.attemptedLane() + " epoch=" + comparison.epoch()
                + " active_lane=" + comparison.activeLane() + " control_kind=" + comparison.controlKind()
                + " control_sha256=" + comparison.controlSha256() + " control_seed=" + comparison.seed();
    }

    private static List<FlyEntity> loadedFlies(CommandSourceStack source) {
        List<FlyEntity> flies = new ArrayList<>();
        for (ServerLevel level : source.getServer().getAllLevels()) {
            flies.addAll(level.getEntities(NeuroCraftFly.FLY, fly -> true));
        }
        return List.copyOf(flies);
    }

    private static int comparison(CommandContext<CommandSourceStack> context) {
        CommandSourceStack source = context.getSource();
        String lane = StringArgumentType.getString(context, "lane");
        List<FlyEntity> flies = loadedFlies(source);
        if (!FlyControllerRuntime.requestComparison(flies, lane)) {
            source.sendFailure(Component.literal(
                    "Comparison refused: requires a trusted catalog, exactly the configured fly, "
                            + "a valid lane and no pending control or handshake"));
            return 0;
        }
        ComparisonLanePresentation status = FlyControllerRuntime.comparisonFor(flies.getFirst());
        source.sendSuccess(() -> Component.literal("Comparison attempt " + status.attemptedLane()
                + " epoch=" + status.epoch() + " " + status.status()
                + "; active lane is shown only after authentication; body/world preserved"), true);
        return 1;
    }

    private static int trialStart(CommandContext<CommandSourceStack> context) {
        if (!FlyControllerRuntime.startMatchedTrials(loadedFlies(context.getSource()))) {
            context.getSource().sendFailure(Component.literal(
                    "Matched trials refused: need an ACTIVE comparison session, one configured fly, and no pending transaction."));
            return 0;
        }
        context.getSource().sendSuccess(() -> Component.literal(
                "Armed 60 declared matched trials; fresh epoch per cell, explicit misses, no automatic retries."), false);
        return 1;
    }

    private static int trialStop(CommandContext<CommandSourceStack> context) {
        FlyControllerRuntime.stopMatchedTrials();
        return trialStatus(context);
    }

    private static int trialStatus(CommandContext<CommandSourceStack> context) {
        context.getSource().sendSuccess(() -> Component.literal(FlyControllerRuntime.matchedTrialStatus()), false);
        return 1;
    }

    /** Bare {@code /neurocraftfly trail} toggles the trail instead of erroring. */
    private static int trailToggle(CommandContext<CommandSourceStack> context) {
        boolean enabled = !FlyTrailPresentation.enabled();
        FlyTrailPresentation.setEnabled(enabled);
        context.getSource().sendSuccess(
                () -> Component.literal((enabled ? "Enabled" : "Disabled")
                        + " the cosmetic NeuroCraft fly trail (interval="
                        + FlyTrailPresentation.intervalTicks() + " ticks)"),
                true);
        return 1;
    }

    private static int trailOn(CommandContext<CommandSourceStack> context) {
        FlyTrailPresentation.setEnabled(true);
        context.getSource().sendSuccess(
                () -> Component.literal("Enabled the cosmetic NeuroCraft fly trail (interval="
                        + FlyTrailPresentation.intervalTicks() + " ticks)"),
                true);
        return 1;
    }

    private static int trailOff(CommandContext<CommandSourceStack> context) {
        FlyTrailPresentation.setEnabled(false);
        context.getSource().sendSuccess(
                () -> Component.literal("Disabled the cosmetic NeuroCraft fly trail"), true);
        return 1;
    }

    private static int trailInterval(CommandContext<CommandSourceStack> context) {
        int ticks = IntegerArgumentType.getInteger(context, "ticks");
        FlyTrailPresentation.setIntervalTicks(ticks);
        context.getSource().sendSuccess(
                () -> Component.literal(
                        "Set the cosmetic NeuroCraft fly trail interval to " + ticks + " ticks"),
                true);
        return 1;
    }

    private static int request(
            CommandContext<CommandSourceStack> context,
            ControlOperation operation,
            String contentId,
            String contentHash) {
        CommandSourceStack source = context.getSource();
        boolean queued = FlyControllerRuntime.requestOperatorControl(
                source.getLevel().getGameTime(),
                operation,
                contentId,
                contentHash,
                message -> source.sendSuccess(() -> Component.literal(message), true),
                message -> source.sendFailure(Component.literal(message)));
        if (!queued) {
            source.sendFailure(Component.literal(operation + " control request was not queued"));
            return 0;
        }
        source.sendSuccess(
                () -> Component.literal("Queued NeuroCraft control request " + operation), true);
        return 1;
    }
}
