package dev.neurocraft.fly.client;

import dev.neurocraft.fly.body.BodyProgramChannel;
import dev.neurocraft.fly.body.SelectionSource;
import dev.neurocraft.fly.control.BrainDisplayPresentation;
import dev.neurocraft.fly.control.ComparisonLanePresentation;
import dev.neurocraft.fly.control.ControllerFault;
import dev.neurocraft.fly.control.EventDisplayPresentation;
import dev.neurocraft.fly.control.EventDisplayState;
import dev.neurocraft.fly.entity.FlyEntity;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import net.minecraft.client.gui.Font;
import net.minecraft.client.gui.GuiGraphics;

/** Small, read-only recording overlay. Detail disclosure never controls the simulation. */
public final class RecordingHud {
    public static final int TEXT = 0xFFE9EFF2;
    public static final int MUTED = 0xFFA1AFBA;
    public static final int ACCENT = 0xFF8DE0CE;
    public static final int ATTENTION = 0xFFF1C17D;
    public static final int FAULT = 0xFFFF9898;
    private static final int BACKGROUND = 0xDC111A21;
    private static final int BORDER = 0x80435460;
    private static final int LEFT = 10;
    private static final int TOP = 10;
    private static final int PADDING = 10;
    private static final int ROW_HEIGHT = 12;
    public static final int MAX_WIDTH = 244;

    public record Row(String label, String value, int color) {}
    public record View(String status, int statusColor, String lane, String authority,
                       List<Row> rows, String modelStatus, String footer) {}

    private RecordingHud() {}

    public static View view(ControllerFault state, BodyProgramChannel body,
                            BrainDisplayPresentation brain, EventDisplayPresentation event,
                            ComparisonLanePresentation comparison, float tickRate) {
        boolean operator = body.source() == SelectionSource.OPERATOR;
        boolean laneReady = comparison.status() == ComparisonLanePresentation.Status.ACTIVE
                || comparison.status() == ComparisonLanePresentation.Status.UNAVAILABLE;
        boolean stepping = state == ControllerFault.ACTIVE && !operator && laneReady;
        boolean neural = stepping && body.source() == SelectionSource.READOUT;
        String status = state.isPaused() ? "Fault" : operator ? "Paused"
                : state == ControllerFault.ACTIVE && laneReady ? "Live" : "Starting";
        int statusColor = state.isPaused() ? FAULT : stepping ? ACCENT : ATTENTION;
        String authority = operator ? "Operator control" : body.source() == SelectionSource.READOUT
                ? "Readout control" : "Default body";
        String lane = comparison.status() == ComparisonLanePresentation.Status.UNAVAILABLE
                ? "Lane unavailable" : laneName(comparison.attemptedLane());
        if (!laneReady) {
            lane = title(comparison.status().name()) + ": " + lane;
        }
        List<Row> rows = new ArrayList<>();
        rows.add(new Row("Body now", title(body.program().name()), TEXT));
        if (state.isPaused()) {
            rows.add(new Row("Controller", title(state.name()), FAULT));
        }
        if (Float.compare(tickRate, 20.0F) != 0) {
            rows.add(new Row("World target", String.format(Locale.ROOT, "%.0f TPS", tickRate), ATTENTION));
        }
        // Keep this block explicitly separate from Body now: display samples may lag that field.
        // Only already-matched presentation frames can supply source, program or neural values.
        if (neural && event.frame().isPresent()) {
            var sample = event.frame().orElseThrow();
            var interaction = sample.interaction();
            String source = interaction == null ? null : interaction.source();
            rows.add(new Row("Applied sample", String.format(Locale.ROOT, "#%d · read %.1fs",
                    sample.commandSequence(), event.ageMillis() / 1000.0), MUTED));
            rows.add(new Row("Input", interaction == null ? "Source unavailable"
                    : source == null ? "No selected event" : sourceName(source), TEXT));
            rows.add(new Row("Program", title(sample.command().program().name()), ACCENT));
            if (source != null) {
                String label;
                double left;
                double right;
                switch (source) {
                    case "food" -> {
                        label = "MN9";
                        left = sample.mn9().left();
                        right = sample.mn9().right();
                    }
                    case "brush" -> {
                        label = "Ant. motor";
                        left = sample.antenna().left();
                        right = sample.antenna().right();
                    }
                    case "light" -> {
                        label = "DNa01";
                        left = interaction.dna01Left();
                        right = interaction.dna01Right();
                    }
                    default -> {
                        label = "TTMn raw";
                        left = interaction.rawTtmn();
                        right = Double.NaN;
                    }
                }
                rows.add(new Row(label, Double.isNaN(right) ? signed(left) : bilateral(left, right), ACCENT));
            }
        } else {
            rows.add(new Row("Applied sample", operator ? "Operator selected"
                    : !stepping ? "Unavailable" : event.status() == EventDisplayState.Status.WAITING_FOR_APPLIED_COMMAND
                            ? "Awaiting match" : title(event.status().name()), MUTED));
        }
        String modelStatus;
        if (stepping && brain.frame().isPresent()) {
            var sample = brain.frame().orElseThrow();
            rows.add(new Row("Model sample", String.format(Locale.ROOT, "#%d · read %.1fs",
                    sample.commandSequence(), brain.ageMillis() / 1000.0), MUTED));
            modelStatus = "Signed model activity";
            for (String key : List.of("lc4", "ttmn")) {
                var left = sample.populations().get(key + "_left");
                var right = sample.populations().get(key + "_right");
                rows.add(new Row(key.equals("lc4") ? "LC4" : "TTMn",
                        bilateral(left.mean(), right.mean()), ACCENT));
            }
        } else {
            modelStatus = operator ? "Neural simulation paused"
                    : "Model sample · " + (stepping ? title(brain.status().name()) : "unavailable");
        }
        return new View(status, statusColor, lane, authority, List.copyOf(rows), modelStatus,
                operator ? "Operator selected · scripted motion"
                        : body.source() == SelectionSource.READOUT ? "Neural readouts · scripted motion"
                        : "Default body · scripted motion");
    }

    public static void render(GuiGraphics graphics, Font font, FlyEntity fly) {
        View view = view(fly.controllerState(), fly.bodyProgramChannel(), fly.brainDisplayPresentation(),
                fly.eventDisplayPresentation(), fly.comparisonPresentation(), fly.level().tickRateManager().tickrate());
        render(graphics, font, view);
    }

    public static void render(GuiGraphics graphics, Font font, View view) {
        // Keep the center clear even at the smallest supported Minecraft GUI width.
        int width = Math.min(MAX_WIDTH, (graphics.guiWidth() - 30) / 2);
        // A tiny window keeps state and lane visible without overflowing below the viewport.
        List<NeuroCraftHud.Line> lines = lines(view);
        int available = width - 2 * PADDING;
        lines = fitLines(font, lines, available, graphics.guiHeight() - 2 * TOP);
        int bottom = TOP + 2 * PADDING + lines.size() * ROW_HEIGHT;
        graphics.fill(LEFT, TOP, LEFT + width, bottom, BORDER);
        graphics.fill(LEFT + 1, TOP + 1, LEFT + width - 1, bottom - 1, BACKGROUND);
        graphics.fill(LEFT, TOP + 12, LEFT + 2, TOP + 30, view.statusColor());
        graphics.enableScissor(LEFT, TOP, LEFT + width, Math.min(bottom, graphics.guiHeight() - TOP));
        int y = TOP + PADDING;
        boolean aligned = lines.size() == lines(view).size()
                && view.rows().stream().allMatch(row -> font.width(row.label()) + 12
                        + font.width(row.value()) <= available);
        if (aligned) {
            graphics.drawString(font, "NeuroCraft Fly", LEFT + PADDING, y, TEXT, false);
            graphics.drawString(font, view.status(), LEFT + width - PADDING - font.width(view.status()),
                    y, view.statusColor(), false);
            y += ROW_HEIGHT;
            graphics.drawString(font, view.lane() + " · " + view.authority(), LEFT + PADDING, y, MUTED, false);
            y += ROW_HEIGHT;
            for (Row row : view.rows()) {
                graphics.drawString(font, row.label(), LEFT + PADDING, y, MUTED, false);
                graphics.drawString(font, row.value(), LEFT + width - PADDING - font.width(row.value()),
                        y, row.color(), false);
                y += ROW_HEIGHT;
            }
            for (String text : List.of(view.modelStatus(), view.footer(), detailsHint())) {
                graphics.drawString(font, text, LEFT + PADDING, y, MUTED, false);
                y += ROW_HEIGHT;
            }
        } else {
            for (var line : lines) {
                graphics.drawString(font, line.text(), LEFT + PADDING, y, line.color(), false);
                y += ROW_HEIGHT;
            }
        }
        graphics.disableScissor();
    }

    /** Content is testable with real font metrics; clipping never creates an invented value. */
    public static List<NeuroCraftHud.Line> lines(View view) {
        List<NeuroCraftHud.Line> lines = new ArrayList<>();
        lines.add(new NeuroCraftHud.Line("NeuroCraft Fly   /   " + view.status(), view.statusColor()));
        lines.add(new NeuroCraftHud.Line(view.lane() + " · " + view.authority(), MUTED));
        for (Row row : view.rows()) {
            lines.add(new NeuroCraftHud.Line(row.label() + "   " + row.value(), row.color()));
        }
        lines.add(new NeuroCraftHud.Line(view.modelStatus(), MUTED));
        lines.add(new NeuroCraftHud.Line(view.footer(), MUTED));
        lines.add(new NeuroCraftHud.Line(detailsHint(), MUTED));
        return List.copyOf(lines);
    }

    public static String detailsHint() {
        return "[" + EventReplayControls.REPLAY.getTranslatedKeyMessage().getString() + "] Replay · ["
                + RecordingHudControls.DETAILS.getTranslatedKeyMessage().getString() + "] Model details";
    }

    public static List<NeuroCraftHud.Line> fitLines(Font font, List<NeuroCraftHud.Line> lines,
                                                   int available, int height) {
        List<NeuroCraftHud.Line> fitted = new ArrayList<>();
        // On narrow windows wrap instead of dropping the active lane or authority suffix.
        for (var line : lines) {
            String remaining = line.text();
            while (font.width(remaining) > available && available > 12) {
                String prefix = font.plainSubstrByWidth(remaining, available);
                int space = prefix.lastIndexOf(' ');
                int split = space > 0 ? space : prefix.length();
                fitted.add(new NeuroCraftHud.Line(remaining.substring(0, split), line.color()));
                remaining = remaining.substring(split).stripLeading();
            }
            fitted.add(new NeuroCraftHud.Line(remaining, line.color()));
        }
        if (2 * PADDING + fitted.size() * ROW_HEIGHT > height) {
            // Fall back to identity + current body/state + disclosure; the screen has all samples.
            List<NeuroCraftHud.Line> compact = new ArrayList<>(lines.subList(0, Math.min(4, lines.size())));
            compact.addAll(lines.subList(lines.size() - 2, lines.size()));
            if (compact.size() < lines.size()) {
                List<NeuroCraftHud.Line> wrapped = fitLines(font, compact, available, Integer.MAX_VALUE);
                int budget = Math.max(0, (height - 2 * PADDING) / ROW_HEIGHT);
                if (wrapped.size() <= budget) { return wrapped; }
                // A very short viewport cannot show all identity/detail text. Bound the panel,
                // retain its top state/identity rows, and make the overflow explicit.
                List<NeuroCraftHud.Line> bounded = new ArrayList<>(wrapped.subList(0,
                        Math.max(0, Math.min(wrapped.size(), budget - 1))));
                if (budget > 0) {
                    bounded.add(new NeuroCraftHud.Line(font.plainSubstrByWidth("More in model details", available), MUTED));
                }
                return List.copyOf(bounded);
            }
        }
        return List.copyOf(fitted);
    }

    public static String laneName(String lane) {
        return switch (lane) {
            case "original" -> "Original weights";
            case "shuffled_weight" -> "Shuffled weights";
            case "no_input" -> "No sensory input";
            default -> title(lane);
        };
    }

    public static String sourceName(String source) {
        return switch (source) {
            case "player_approach" -> "Player approach";
            case "mob_approach" -> "Mob approach";
            case "attack" -> "Attack attempt";
            case "food" -> "Sweet food";
            case "brush" -> "Brush contact";
            case "light" -> "Light change";
            default -> title(source);
        };
    }

    private static String bilateral(double left, double right) {
        return "L " + signed(left) + "  R " + signed(right);
    }

    private static String signed(double value) {
        return String.format(Locale.ROOT, "%+.1e", value);
    }

    private static String title(String value) {
        String words = value.toLowerCase(Locale.ROOT).replace('_', ' ');
        return words.isEmpty() ? words : Character.toUpperCase(words.charAt(0)) + words.substring(1);
    }
}
