package dev.neurocraft.fly.client;

import dev.neurocraft.fly.body.BodyProgramChannel;
import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ProgramParameters;
import dev.neurocraft.fly.body.ReadoutValues;
import dev.neurocraft.fly.body.SelectionSource;
import dev.neurocraft.fly.control.ControllerFault;
import dev.neurocraft.fly.control.BrainDisplayFrame;
import dev.neurocraft.fly.control.BrainDisplayPresentation;
import dev.neurocraft.fly.control.EventDisplayFrame;
import dev.neurocraft.fly.control.EventDisplayPresentation;
import dev.neurocraft.fly.control.EventDisplayState;
import dev.neurocraft.fly.control.PrimitiveAction;
import dev.neurocraft.fly.control.ComparisonLanePresentation;
import dev.neurocraft.fly.entity.FlyEntity;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.function.ToDoubleFunction;
import java.util.stream.Collectors;
import net.minecraft.client.DeltaTracker;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.Font;
import net.minecraft.client.gui.GuiGraphics;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.world.entity.Entity;

/** Read-only overlay reconstructed entirely from server-synchronized presentation fields. */
public final class NeuroCraftHud {
    private static final int NEUTRAL_COLOR = 0xFFD7E0E7;
    private static final int ACTIVE_COLOR = 0xFF72E6A5;
    private static final int PAUSED_COLOR = 0xFFFF5B5B;
    private static final int TITLE_COLOR = 0xFFF2D38A;
    private static final int PANEL_COLOR = 0xB0101418;

    private static final int LINE_HEIGHT = 10;
    private static final int PANEL_LEFT = 6;
    private static final int PANEL_TOP = 6;
    private static final int FIRST_LINE_TOP = 11;
    private static final int PANEL_BOTTOM_PADDING = 5;
    /** Clear space kept between the widest line's last pixel and the panel's right edge. */
    private static final int TEXT_RIGHT_PADDING = 6;
    /**
     * The panel's right edge never falls short of this, so a short frame keeps the familiar box.
     * It is a floor, not a size: the panel grows to the widest line it actually draws.
     */
    private static final int PANEL_MIN_RIGHT = 300;
    /** Where every line's first pixel is drawn; public so a test measures from the same origin. */
    public static final int TEXT_LEFT = 12;

    /** One drawn HUD line: its text and the colour it is drawn in. */
    public record Line(String text, int color) {}

    private NeuroCraftHud() {}

    public static void render(GuiGraphics graphics, DeltaTracker tickCounter) {
        Minecraft minecraft = Minecraft.getInstance();
        ClientLevel level = minecraft.level;
        if (level == null) {
            return;
        }
        FlyEntity fly = firstFly(level);
        if (fly == null) {
            return;
        }
        RecordingHud.render(graphics, minecraft.font, fly);
    }

    /** Full diagnostic text remains available through the model-details screen. */
    public static List<Line> detailsLines(FlyEntity fly) {
        return showcaseLines(
                fly.controllerState(),
                fly.bodyProgramChannel(),
                fly.readoutChannel().readout(),
                fly.presentationAction(),
                fly.brainDisplayPresentation(),
                fly.level().tickRateManager().tickrate(), fly.comparisonPresentation(),
                fly.eventDisplayPresentation());
    }

    /** Compact filmed view: actual samples, source/authority labels, and all three goal facts. */
    public static List<Line> showcaseLines(
            ControllerFault state, BodyProgramChannel channel, ReadoutValues readout,
            PrimitiveAction action, BrainDisplayPresentation display, float tickRate) {
        return showcaseLines(state, channel, readout, action, display, tickRate,
                ComparisonLanePresentation.unavailable());
    }

    public static List<Line> showcaseLines(
            ControllerFault state, BodyProgramChannel channel, ReadoutValues readout,
            PrimitiveAction action, BrainDisplayPresentation display, float tickRate,
            ComparisonLanePresentation comparison) {
        List<Line> lines = new ArrayList<>();
        lines.add(new Line("NEUROCRAFT FLY · SERVER AUTHORITY", TITLE_COLOR));
        lines.add(new Line(playbackRateLine(tickRate), tickRate == 20.0F ? NEUTRAL_COLOR : PAUSED_COLOR));
        lines.add(new Line("CONTROL: " + state.name(), state.isPaused() ? PAUSED_COLOR : ACTIVE_COLOR));
        if (comparison.status() != ComparisonLanePresentation.Status.UNAVAILABLE) {
            boolean matched = comparison.status() == ComparisonLanePresentation.Status.ACTIVE;
            lines.add(new Line((matched ? "LANE ACTIVE: " : comparison.status() + " → ")
                    + comparison.attemptedLane() + (matched ? " · ENGINEERED" : ""),
                    matched ? ACTIVE_COLOR : PAUSED_COLOR));
            lines.add(new Line("EPOCH " + comparison.epoch()
                    + (matched ? " · DYN " + comparison.activeIdentity().orElseThrow()
                            .dynamicsHash().substring(0, 8) : ""),
                    NEUTRAL_COLOR));
        }
        lines.add(new Line("BODY: " + channel.program() + " · " + channel.source()
                + (channel.source() == SelectionSource.READOUT
                        && drivenBy(channel.program()).isEmpty() ? " · FALLBACK" : ""),
                bodyProgramColor(channel.source())));
        if (channel.source() == SelectionSource.OPERATOR) {
            lines.add(new Line("OPERATOR: SCRIPTED · CONNECTOME NOT DRIVING", PAUSED_COLOR));
        } else if (channel.source() == SelectionSource.READOUT) {
            lines.add(new Line("HAND-CHOSEN (ADR 0019) → SCRIPTED PROGRAMS", NEUTRAL_COLOR));
            lines.add(new Line("NOT THE FLY DECIDING", NEUTRAL_COLOR));
            lines.add(new Line(String.format(Locale.ROOT, "READOUT: TTMn %.2f  DNa01 %+.2f  WING %+.2f",
                    readout.ttmnJump(), readout.dna01Turn(), readout.wingCommon()), NEUTRAL_COLOR));
        } else {
            lines.add(new Line("Program label: DEFAULT · selection unavailable", NEUTRAL_COLOR));
        }
        if (channel.source() != SelectionSource.READOUT) {
            lines.add(new Line("ARCHITECTURE: READOUTS → SCRIPTED PROGRAMS", NEUTRAL_COLOR));
            lines.add(new Line("HAND-CHOSEN MAPPING · NOT FLY DECIDING", NEUTRAL_COLOR));
        }
        String modelHeading = display.frame().isPresent() ? "MODEL MEANS · ENGINEERED SENSES" : "MODEL DISPLAY";
        if (comparison.status() == ComparisonLanePresentation.Status.ACTIVE && display.frame().isPresent()) {
            BrainDisplayFrame sample = display.frame().orElseThrow();
            modelHeading = "MODEL · GRAPH " + sample.identity().graphHash().substring(0, 8)
                    + " · ENC " + sample.encoderSchema().replace("neurocraft.eem.encoder.", "");
        }
        lines.add(new Line(modelHeading, TITLE_COLOR));
        if (comparison.status() == ComparisonLanePresentation.Status.ACTIVE && display.frame().isEmpty()) {
            var identity = comparison.activeIdentity().orElseThrow();
            lines.add(new Line("GRAPH " + identity.graphHash().substring(0, 8)
                    + " · ADAPTER " + identity.adapterHash().substring(0, 8), NEUTRAL_COLOR));
        }
        if (channel.source() == SelectionSource.OPERATOR) {
            lines.add(new Line("PAUSED · CONNECTOME NOT STEPPING", PAUSED_COLOR));
        } else if (state.isPaused()) {
            lines.add(new Line("MODEL SESSION FAULTED · DISPLAY UNAVAILABLE", PAUSED_COLOR));
        } else if (display.frame().isPresent()) {
            BrainDisplayFrame sample = display.frame().orElseThrow();
            if (comparison.status() == ComparisonLanePresentation.Status.UNAVAILABLE) {
                lines.add(new Line("GRAPH " + sample.identity().graphHash().substring(0, 8)
                        + " · DYNAMICS " + sample.identity().dynamicsHash().substring(0, 8), NEUTRAL_COLOR));
                lines.add(new Line("ENCODER " + sample.encoderSchema(), NEUTRAL_COLOR));
            }
            lines.add(new Line(String.format(Locale.ROOT, "SAMPLE #%d · obs %d · received %.1fs",
                    sample.commandSequence(), sample.observationTick(), display.ageMillis() / 1000.0),
                    NEUTRAL_COLOR));
            for (String[] population : new String[][] {
                    {"LC4", "lc4"}, {"DNp01", "dnp01"}, {"TTMn", "ttmn"}}) {
                BrainDisplayFrame.Population left = sample.populations().get(population[1] + "_left");
                BrainDisplayFrame.Population right = sample.populations().get(population[1] + "_right");
                lines.add(new Line(String.format(Locale.ROOT, "%s L %+.1e  R %+.1e [%d/%d]",
                        population[0], left.mean(), right.mean(), left.count(), right.count()), ACTIVE_COLOR));
            }
            lines.add(new Line("DNp01: display only · values are model activity", NEUTRAL_COLOR));
        } else {
            lines.add(new Line("SAMPLE: " + display.status().name().replace('_', ' '), PAUSED_COLOR));
            lines.add(new Line("LC4 / DNp01 / TTMn: no current matched sample", NEUTRAL_COLOR));
        }
        lines.add(new Line("128-NEURON FIXTURE · SYNTHETIC IMITATION", TITLE_COLOR));
        lines.add(new Line("MSE 95.66x raw / 403.70x bounded (ADR 0019)", NEUTRAL_COLOR));
        lines.add(new Line("OFFLINE ESCAPE-ARC-01:", TITLE_COLOR));
        lines.add(new Line("DNp01 ~300x wing; TTMn 35.1x baseline", NEUTRAL_COLOR));
        lines.add(new Line("4 neural steps · NOT LIVE REACTION TIMING", NEUTRAL_COLOR));
        lines.add(new Line(String.format(Locale.ROOT, "APPLIED F %+.2f  V %+.2f  YAW %+.2f",
                action.forwardThrust(), action.verticalThrust(), action.yawTorque()), NEUTRAL_COLOR));
        return List.copyOf(lines);
    }

    /** Candidate event view stays compact enough to retain the facts at 320x240 GUI pixels. */
    public static List<Line> showcaseLines(
            ControllerFault state, BodyProgramChannel channel, ReadoutValues readout,
            PrimitiveAction action, BrainDisplayPresentation display, float tickRate,
            ComparisonLanePresentation comparison, EventDisplayPresentation events) {
        if (events.status() == EventDisplayState.Status.DISABLED) {
            return showcaseLines(state, channel, readout, action, display, tickRate, comparison);
        }
        List<Line> lines = new ArrayList<>();
        boolean comparisonActive = comparison.status() == ComparisonLanePresentation.Status.ACTIVE;
        boolean running = state == ControllerFault.ACTIVE && channel.source() == SelectionSource.READOUT
                && (comparisonActive || comparison.status() == ComparisonLanePresentation.Status.UNAVAILABLE);
        lines.add(new Line("NEUROCRAFT · ENGINEERED EVENT CANDIDATE", TITLE_COLOR));
        lines.add(new Line(playbackRateLine(tickRate), NEUTRAL_COLOR));
        lines.add(new Line("CONTROL " + state + " · BODY " + channel.program()
                + " · " + channel.source(), running ? ACTIVE_COLOR : PAUSED_COLOR));
        if (comparison.status() != ComparisonLanePresentation.Status.UNAVAILABLE) {
            lines.add(new Line((comparisonActive ? "LANE ACTIVE: " : comparison.status() + " → ")
                    + comparison.attemptedLane(), comparisonActive ? ACTIVE_COLOR : PAUSED_COLOR));
            lines.add(new Line("EPOCH " + comparison.epoch(), NEUTRAL_COLOR));
        }
        lines.add(new Line("HAND-CHOSEN READOUTS → SCRIPTED PROGRAMS", NEUTRAL_COLOR));
        lines.add(new Line(channel.source() == SelectionSource.OPERATOR
                ? "OPERATOR · PAUSED · CONNECTOME NOT DRIVING"
                : "NOT THE FLY DECIDING · MODEL ACTIVITY", NEUTRAL_COLOR));
        if (running && display.frame().isPresent()) {
            var sample = display.frame().orElseThrow();
            lines.add(new Line(String.format(Locale.ROOT, "MODEL %s · #%d · read %.1fs",
                    sample.identity().graphHash().substring(0, 8), sample.commandSequence(),
                    display.ageMillis() / 1000.0), TITLE_COLOR));
            for (String[] pop : new String[][] {{"LC4", "lc4"}, {"DNp01*", "dnp01"}, {"TTMn", "ttmn"}}) {
                lines.add(new Line(String.format(Locale.ROOT, "%s L %+.1e  R %+.1e", pop[0],
                        sample.populations().get(pop[1] + "_left").mean(),
                        sample.populations().get(pop[1] + "_right").mean()), ACTIVE_COLOR));
            }
        } else {
            lines.add(new Line("MODEL UNAVAILABLE · " + (running ? display.status() : "PAUSED/FAULTED"), PAUSED_COLOR));
            lines.add(new Line("LC4 / DNp01* / TTMn: unavailable", NEUTRAL_COLOR));
        }
        if (running && events.frame().isPresent()) {
            EventDisplayFrame sample = events.frame().orElseThrow();
            if (sample.interaction() != null) {
                interactionLines(lines, sample, events.ageMillis());
            } else {
                lines.add(new Line(String.format(Locale.ROOT, "EVENT #%d APPLIED %s · read %.1fs",
                        sample.commandSequence(), sample.command().program(), events.ageMillis() / 1000.0), TITLE_COLOR));
                lines.add(new Line(String.format(Locale.ROOT, "MN9 L %+.1e R %+.1e [1/1]",
                        sample.mn9().left(), sample.mn9().right()), ACTIVE_COLOR));
                lines.add(new Line(String.format(Locale.ROOT, "Ant. motor L %+.1e R %+.1e [6/7]",
                        sample.antenna().left(), sample.antenna().right()), ACTIVE_COLOR));
                lines.add(new Line("WHY " + sample.reason().replace('_', ' '), NEUTRAL_COLOR));
                var trigger = sample.trigger();
                lines.add(new Line(trigger == null ? "TRIGGER: none scored"
                        : String.format(Locale.ROOT, "ISSUED #%d %s score %.1e",
                                trigger.command().commandSequence(), trigger.readout().equals("mn9") ? "MN9" : "ANT M",
                                trigger.score().score()), NEUTRAL_COLOR));
            }
        } else {
            if (!running) {
                lines.add(new Line("EVENT UNAVAILABLE · PAUSED/FAULTED", PAUSED_COLOR));
            } else if (events.status() == EventDisplayState.Status.WAITING_FOR_APPLIED_COMMAND) {
                // Missing application evidence says nothing about whether an event occurred.
                lines.add(new Line("EVENT DISPLAY · awaiting applied match", NEUTRAL_COLOR));
            } else {
                lines.add(new Line("EVENT DISPLAY · " + events.status(), TITLE_COLOR));
            }
            lines.add(new Line("MN9 / antennal motors: unavailable", NEUTRAL_COLOR));
        }
        lines.add(new Line("*DNp01 display only · signed model activity", NEUTRAL_COLOR));
        lines.add(new Line("128-NEURON FIXTURE · SYNTHETIC IMITATION", TITLE_COLOR));
        lines.add(new Line("MSE 95.66x raw / 403.70x bounded (ADR 0019)", NEUTRAL_COLOR));
        lines.add(new Line("OFFLINE ESCAPE-ARC-01:", TITLE_COLOR));
        lines.add(new Line("DNp01 ~300x wing; TTMn 35.1x baseline", NEUTRAL_COLOR));
        lines.add(new Line("4 neural steps · NOT LIVE REACTION TIMING", NEUTRAL_COLOR));
        return List.copyOf(lines);
    }

    /** V2 uses the same five detail lines as V1; source metadata does not imply recognition. */
    private static void interactionLines(List<Line> lines, EventDisplayFrame sample, long ageMillis) {
        var details = sample.interaction();
        String source = details.source();
        lines.add(new Line(String.format(Locale.ROOT, "READOUT #%d APPLIED %s · read %.1fs",
                sample.commandSequence(), sample.command().program(), ageMillis / 1000.0), TITLE_COLOR));
        if (source == null) {
            lines.add(new Line("SOURCE none · fallback " + sample.command().program(), NEUTRAL_COLOR));
            lines.add(new Line(String.format(Locale.ROOT, "TTMn raw %+.1e · light %d",
                    details.rawTtmn(), details.localLight()), ACTIVE_COLOR));
            lines.add(new Line("GATE: none selected · " + sample.reason().replace('_', ' '), NEUTRAL_COLOR));
            lines.add(new Line("TRIGGER: none issued", NEUTRAL_COLOR));
            return;
        }
        String context = switch (source) {
            case "attack" -> "ATTACK ATTEMPT · tactile proxy";
            case "player_approach" -> "PLAYER approach · Minecraft category";
            case "mob_approach" -> "MOB approach · Minecraft category";
            case "food" -> "SWEET FOOD · engineered contact/near";
            case "brush" -> "BRUSH · engineered touch event";
            case "light" -> "LIGHT " + details.localLight() + " · Minecraft level";
            default -> throw new IllegalArgumentException("unsupported parsed source");
        };
        lines.add(new Line(context, NEUTRAL_COLOR));
        String raw = switch (source) {
            case "food" -> String.format(Locale.ROOT, "MN9 L %+.1e R %+.1e [1/1]",
                    sample.mn9().left(), sample.mn9().right());
            case "brush" -> String.format(Locale.ROOT, "Ant. motor L %+.1e R %+.1e [6/7]",
                    sample.antenna().left(), sample.antenna().right());
            case "light" -> String.format(Locale.ROOT, "DNa01 L %+.1e R %+.1e",
                    details.dna01Left(), details.dna01Right());
            default -> String.format(Locale.ROOT, "TTMn raw %+.1e · jump %.1e",
                    details.rawTtmn(), sample.command().readout().ttmnJump());
        };
        lines.add(new Line(raw, ACTIVE_COLOR));
        var gate = details.scores().get(source);
        var trigger = sample.trigger();
        // Source/context eligibility is not a fresh neural threshold crossing.
        // A held program retains its original qualifying issued trigger.
        String phase = trigger.command().commandSequence() == sample.commandSequence() ? "onset" : "held";
        lines.add(new Line(String.format(Locale.ROOT, "NOW %s %.1e · %s",
                formulaLabel(gate.formula()), gate.value().score(),
                gate.value().baseline() == null ? "no baseline" : phase), NEUTRAL_COLOR));
        lines.add(new Line(String.format(Locale.ROOT, "ISSUED #%d %s %s %.1e",
                trigger.command().commandSequence(), trigger.command().program(),
                formulaLabel(trigger.formula()), trigger.score().score()), NEUTRAL_COLOR));
    }

    private static String formulaLabel(String formula) {
        return switch (formula) {
            case "absolute_change" -> "abs delta";
            case "positive_change" -> "positive delta";
            case "absolute_readout" -> "readout";
            default -> throw new IllegalArgumentException("unsupported parsed formula");
        };
    }

    /**
     * Every line the HUD draws for one synchronized frame, top to bottom, each with its colour.
     *
     * <p>Pure in the frame's decoded words, so a test can lay out exactly the lines a viewer would
     * see and measure them with the client's real font. The CONTROL: line goes red whenever the
     * body is scripted as well as whenever the controller is paused, so a frame grab never shows a
     * green "ACTIVE" controller above a red scripted body and invites the reader to take the green
     * half as the whole story. The readout lane stays green: the connectome is stepping, and the
     * two lines beneath it say exactly what its readouts are and are not doing.
     */
    public static List<Line> panelLines(
            ControllerFault controllerState,
            BodyProgramChannel channel,
            ReadoutValues readout,
            PrimitiveAction action) {
        Modulation modulation = channel.modulation();
        boolean modulated = !modulation.isNeutral() || !channel.parameters().isNeutral();
        int stateColor = controllerState.isPaused() || channel.source() == SelectionSource.OPERATOR
                ? PAUSED_COLOR
                : ACTIVE_COLOR;
        List<Line> lines = new ArrayList<>();
        lines.add(new Line("NEUROCRAFT FLY · SERVER AUTHORITY", TITLE_COLOR));
        lines.add(new Line("CONTROL: " + controllerState.name(), stateColor));
        lines.add(new Line(bodyProgramLine(channel, readout), bodyProgramColor(channel.source())));
        if (channel.source() == SelectionSource.READOUT) {
            for (String text : readoutMappingLines(channel.program(), readout)) {
                lines.add(new Line(text, NEUTRAL_COLOR));
            }
        }
        lines.add(new Line(
                String.format(
                        Locale.ROOT,
                        "APPLIED PRIMITIVES  F %+1.2f  L %+1.2f  V %+1.2f",
                        action.forwardThrust(),
                        action.lateralThrust(),
                        action.verticalThrust()),
                NEUTRAL_COLOR));
        lines.add(new Line(
                String.format(
                        Locale.ROOT,
                        "TORQUE  YAW %+1.2f  PITCH %+1.2f",
                        action.yawTorque(),
                        action.pitchTorque()),
                NEUTRAL_COLOR));
        if (modulated) {
            lines.add(new Line(modulationLine(modulation, channel.parameters()), NEUTRAL_COLOR));
        }
        return List.copyOf(lines);
    }

    public static String playbackRateLine(float tickRate) {
        return String.format(Locale.ROOT, "WORLD RATE: %.1f TPS · %s", tickRate,
                tickRate == 20.0F ? "NORMAL TARGET"
                        : tickRate == 10.0F ? "HALF-SPEED DIAGNOSTIC" : "NONSTANDARD TARGET");
    }

    /**
     * The panel's right edge for these lines: the widest line as the font actually measures it,
     * plus clear space, and never narrower than {@link #PANEL_MIN_RIGHT}.
     *
     * <p>Sized from the measurement rather than from a constant because the constant clipped. The
     * readout lane's mapping line was drawn into a fixed 380-px panel; at the default GUI scale in
     * a 1,708-px window it overran the panel and the window edge cut it after {@code TTMn 0.01
     * DNa}, so the DNa01 and WING values were never on screen (lane L, 2026-09-04). Every line is now wrapped to the actual GUI width before sizing
     * the panel, and {@code HudHonestyClientGameTest} measures the panel
     * against the GUI-scaled window width at the scale it runs at.
     */
    public static int panelRight(Font font, List<Line> lines, int guiWidth) {
        int widest = lines.stream().mapToInt(line -> font.width(line.text())).max().orElse(0);
        return Math.min(guiWidth - PANEL_LEFT,
                Math.max(PANEL_MIN_RIGHT, TEXT_LEFT + widest + TEXT_RIGHT_PADDING));
    }

    /** Wrap every line to the actual GUI width, preserving its words and lane colour. */
    public static List<Line> wrapLines(Font font, List<Line> lines, int guiWidth) {
        int available = Math.max(1, guiWidth - PANEL_LEFT - TEXT_LEFT - TEXT_RIGHT_PADDING);
        List<Line> wrapped = new ArrayList<>();
        for (Line line : lines) {
            String remaining = line.text();
            while (font.width(remaining) > available) {
                String prefix = font.plainSubstrByWidth(remaining, available);
                if (prefix.isEmpty()) {
                    throw new IllegalArgumentException("HUD viewport cannot fit one character");
                }
                int space = prefix.lastIndexOf(' ');
                int split = space > 0 ? space : prefix.length();
                wrapped.add(new Line(remaining.substring(0, split), line.color()));
                remaining = remaining.substring(split).stripLeading();
            }
            wrapped.add(new Line(remaining, line.color()));
        }
        return List.copyOf(wrapped);
    }

    /** The panel's bottom edge for this many lines; five lines give the original 86-px panel. */
    public static int panelBottom(int lineCount) {
        return FIRST_LINE_TOP + LINE_HEIGHT * lineCount + PANEL_BOTTOM_PADDING;
    }

    /**
     * The body-program line, with its disclaimer <strong>inside the string</strong> rather than
     * beside it. That is the direct lesson of D-074 / ADR 0016: a frame grab travels without its
     * caption, so a screenshot of a scripted fly has to say so on its face.
     *
     * <p>Under READOUT the arrow names the labeled readout the declared mapping consults for this
     * program, and shows the value that crossed the wire with the selection. It is a label, not a
     * second copy of the mapping: the client computes no selection, and the lines beneath it say in
     * words that the mapping is hand-chosen and that the fly is not deciding.
     */
    public static String bodyProgramLine(BodyProgramChannel channel, ReadoutValues readout) {
        return switch (channel.source()) {
            case DEFAULT -> "BODY PROGRAM: " + channel.program() + " · DEFAULT";
            case OPERATOR -> "BODY PROGRAM: " + channel.program()
                    + " · OPERATOR — SCRIPTED, CONNECTOME NOT DRIVING";
            case READOUT -> drivenBy(channel.program())
                    .map(driver -> "BODY PROGRAM: " + channel.program() + " · READOUT ← "
                            + driver.render(readout))
                    // No arrow, because no readout selected this program: it is the declared
                    // fallback the authority ran after refusing the mapping's request. An arrow
                    // here would claim a readout chose HOVER, which is the opposite of the truth.
                    .orElse("BODY PROGRAM: " + channel.program() + " · READOUT · FALLBACK");
        };
    }

    /**
     * The READOUT lane's second and third lines: what the mapping is, what it is not, and the
     * readout scalars the program line did not name. Every clause is required. "HAND-CHOSEN (ADR
     * 0019)" says the mapping is engineering; "NOT THE FLY DECIDING" says the thing a viewer would
     * otherwise assume.
     *
     * <p>Two lines rather than one because one did not fit; see {@link #panelRight}. The honesty
     * clauses own the first line in full, so the words a frame grab must carry are never the words
     * a narrow window trims, and the values own the second. All three scalars are visible across
     * the program line and this pair in every case, including the fallback case, where the program
     * line names none of them.
     */
    public static List<String> readoutMappingLines(BodyProgramId program, ReadoutValues readout) {
        String remainder = Arrays.stream(Driver.values())
                .filter(driver -> !drivenBy(program).equals(Optional.of(driver)))
                .map(driver -> driver.render(readout))
                .collect(Collectors.joining("  "));
        return List.of(
                "READOUT MAPPING: HAND-CHOSEN (ADR 0019) · NOT THE FLY DECIDING", remainder);
    }

    /** The published modulation and the declared signed turn bias, at the channel's resolution. */
    public static String modulationLine(Modulation modulation, ProgramParameters parameters) {
        return String.format(
                Locale.ROOT,
                "MODULATION: SPD %.1f TRN %.1f AMP %.1f URG %.1f BIAS %+.1f",
                modulation.speed(),
                modulation.turnRate(),
                modulation.amplitude(),
                modulation.urgency(),
                parameters.turnBias());
    }

    /** Operator red, readout green, default neutral, so a scripted fly is unmistakable on camera. */
    static int bodyProgramColor(SelectionSource source) {
        return switch (source) {
            case DEFAULT -> NEUTRAL_COLOR;
            case OPERATOR -> PAUSED_COLOR;
            case READOUT -> ACTIVE_COLOR;
        };
    }

    /**
     * The three labeled readout scalars, and how each is written on the HUD. {@code TTMn} is a
     * rectified term so it is unsigned; the other two are signed differences.
     */
    private enum Driver {
        TTMN("TTMn %.2f", ReadoutValues::ttmnJump),
        DNA01("DNa01 %+.2f", ReadoutValues::dna01Turn),
        WING("WING %+.2f", ReadoutValues::wingCommon);

        private final String format;
        private final ToDoubleFunction<ReadoutValues> value;

        Driver(String format, ToDoubleFunction<ReadoutValues> value) {
            this.format = format;
            this.value = value;
        }

        String render(ReadoutValues readout) {
            return String.format(Locale.ROOT, format, value.applyAsDouble(readout));
        }
    }

    /**
     * Which labeled readout the declared mapping consults <em>to select this program</em>, if any.
     *
     * <p>A declared presentation label keyed on the program id, written once here; the mapping
     * itself runs in the neural service and its identity is reported by
     * {@code /neurocraftfly status}. Only the two programs the mapping can actually select have a
     * driver. Everything else on this lane is the declared HOVER fallback after a refusal, and
     * naming a readout beside it would assert a selection that never happened.
     */
    private static Optional<Driver> drivenBy(BodyProgramId program) {
        return switch (program) {
            case ESCAPE_JUMP -> Optional.of(Driver.TTMN);
            case CRUISE -> Optional.of(Driver.WING);
            default -> Optional.empty();
        };
    }

    private static FlyEntity firstFly(ClientLevel level) {
        for (Entity entity : level.entitiesForRendering()) {
            if (entity instanceof FlyEntity fly) {
                return fly;
            }
        }
        return null;
    }
}
