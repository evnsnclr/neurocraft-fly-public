package dev.neurocraft.fly.client;

import dev.neurocraft.fly.control.EventReplayHistory;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.function.Supplier;
import net.minecraft.client.gui.GuiGraphics;
import net.minecraft.client.gui.components.Button;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.network.chat.Component;
import org.lwjgl.glfw.GLFW;

/** Historical display playback only. Screen navigation never pauses, resets or replays the model. */
public final class EventReplayScreen extends Screen {
    private final Supplier<List<EventReplayHistory.Sample>> source;
    private List<EventReplayHistory.Sample> samples;
    private int index;
    private int scroll;
    private int maxScroll;
    private boolean playing;
    private int playTicks;
    private Button playButton;

    public EventReplayScreen(Supplier<List<EventReplayHistory.Sample>> source) {
        super(Component.literal("Circuit Playground · event replay"));
        this.source = source;
        refresh();
    }

    private void refresh() {
        samples = List.copyOf(source.get());
        index = Math.max(0, samples.size() - 1);
        for (int cursor = samples.size() - 1; cursor >= 0; cursor--) {
            if (samples.get(cursor).inspectedSource() != null) { index = cursor; break; }
        }
        playing = false;
        scroll = 0;
    }

    @Override protected void init() {
        int size = Math.min(90, (width - 44) / 4);
        int left = (width - 4 * size - 12) / 2;
        addRenderableWidget(Button.builder(Component.literal("Previous"), button -> step(-1))
                .bounds(left, height - 28, size, 20).build());
        playButton = addRenderableWidget(Button.builder(Component.literal("Play samples"), button -> togglePlayback())
                .bounds(left + size + 4, height - 28, size, 20).build());
        addRenderableWidget(Button.builder(Component.literal("Next"), button -> step(1))
                .bounds(left + 2 * (size + 4), height - 28, size, 20).build());
        addRenderableWidget(Button.builder(Component.literal("Load latest"), button -> {
            refresh(); updatePlayButton();
        }).bounds(left + 3 * (size + 4), height - 28, size, 20).build());
        updatePlayButton();
    }

    private void step(int amount) {
        playing = false;
        index = Math.max(0, Math.min(samples.size() - 1, index + amount));
        updatePlayButton();
    }

    private void togglePlayback() {
        if (samples.isEmpty()) { return; }
        if (index >= samples.size() - 1) { index = 0; }
        playing = !playing;
        playTicks = 0;
        updatePlayButton();
    }

    private void updatePlayButton() {
        if (playButton != null) {
            playButton.setMessage(Component.literal(playing ? "Stop playback" : "Play samples"));
            playButton.active = !samples.isEmpty();
        }
    }

    @Override public void tick() {
        if (playing && ++playTicks >= 4) {
            playTicks = 0;
            if (index < samples.size() - 1) { index++; }
            else { playing = false; updatePlayButton(); }
        }
    }

    @Override public boolean isPauseScreen() { return false; }

    @Override public void renderBackground(GuiGraphics graphics, int x, int y, float delta) {
        // Own opaque background; vanilla blur would obscure text painted before super.render.
    }

    @Override public boolean keyPressed(int keyCode, int scanCode, int modifiers) {
        if (EventReplayControls.REPLAY.matches(keyCode, scanCode)) { onClose(); return true; }
        if (keyCode == GLFW.GLFW_KEY_LEFT) { step(-1); return true; }
        if (keyCode == GLFW.GLFW_KEY_RIGHT) { step(1); return true; }
        if (keyCode == GLFW.GLFW_KEY_SPACE) { togglePlayback(); return true; }
        return super.keyPressed(keyCode, scanCode, modifiers);
    }

    @Override public boolean mouseClicked(double x, double y, int button) {
        if (EventReplayControls.REPLAY.matchesMouse(button)) { onClose(); return true; }
        return super.mouseClicked(x, y, button);
    }

    @Override public boolean mouseScrolled(double x, double y, double horizontal, double vertical) {
        scroll = Math.max(0, Math.min(maxScroll, scroll - (int) (vertical * 30)));
        return true;
    }

    public static List<NeuroCraftHud.Line> sampleLines(EventReplayHistory.Sample sample) {
        var lines = new ArrayList<NeuroCraftHud.Line>();
        var frame = sample.frame();
        var details = frame.interaction();
        String source = sample.inspectedSource();
        lines.add(line("HISTORICAL DISPLAY SAMPLE · not the current world", RecordingHud.ATTENTION));
        lines.add(line("Condition: " + RecordingHud.laneName(sample.lane())
                + (sample.seed() == null ? "" : " · seed " + sample.seed())
                + " · epoch " + frame.identity().sessionEpoch(), RecordingHud.TEXT));
        lines.add(line("Run " + frame.identity().runId() + " · command #" + frame.commandSequence()
                + " · observation tick " + frame.observationTick(), RecordingHud.MUTED));
        lines.add(line("1 · SOURCE: " + EventReplayHistory.sourceLabel(source) + (source == null ? ""
                : sample.selectedSource() ? " · selected" : " · context eligible; no source selected"), RecordingHud.TEXT));
        if (source != null) {
            String values = switch (source) {
                case "brush" -> bilateral("Antennal motor", frame.antenna().left(), frame.antenna().right());
                case "food" -> bilateral("MN9", frame.mn9().left(), frame.mn9().right());
                case "light" -> bilateral("DNa01", details.dna01Left(), details.dna01Right());
                default -> String.format(Locale.ROOT, "TTMn raw %+.3e · rectified jump %.3e",
                        details.rawTtmn(), frame.command().readout().ttmnJump());
            };
            lines.add(line("2 · MODELED READOUT: " + values, RecordingHud.ACCENT));
            var score = details.scores().get(source);
            lines.add(line(String.format(Locale.ROOT, "%s · raw %+.3e · baseline %s · score %.3e",
                    formulaLabel(score.formula()), score.value().raw(), score.value().baseline() == null ? "unavailable"
                            : String.format(Locale.ROOT, "%+.3e", score.value().baseline()), score.value().score()), RecordingHud.TEXT));
            lines.add(line("Source context eligible: " + score.value().eligible()
                    + " · selection: " + frame.reason().replace('_', ' '), RecordingHud.MUTED));
        } else {
            lines.add(line("No eligible event readout to inspect; missing event evidence is not a zero response.", RecordingHud.MUTED));
        }
        lines.add(line("3 · ISSUED / MATCHED APPLICATION", RecordingHud.ACCENT));
        var trigger = frame.trigger();
        lines.add(line(trigger == null ? "No event onset is attached to this command."
                : String.format(Locale.ROOT, "Issued onset #%d: %s · %s score %.3e (issued)",
                        trigger.command().commandSequence(), trigger.command().program(), formulaLabel(trigger.formula()), trigger.score().score()),
                RecordingHud.TEXT));
        lines.add(line("Applied match #" + frame.commandSequence() + ": " + frame.command().program()
                + " · exact command, non-refused application", RecordingHud.ACCENT));
        lines.add(line("Only the matched command is confirmed here; an earlier onset remains issued. Held programs retain it.", RecordingHud.MUTED));
        lines.add(line("Numeric threshold and executed sensory vector: unavailable in this display.", RecordingHud.MUTED));
        if ("no_input".equals(sample.lane())) {
            lines.add(line("No-input condition declares zero sensory drive; baseline Cruise may remain.", RecordingHud.MUTED));
        }
        lines.add(line("Signed dimensionless model activity; hand-chosen readouts select scripted motion.", RecordingHud.MUTED));
        return List.copyOf(lines);
    }

    public static List<NeuroCraftHud.Line> interactionHelp() {
        return List.of(
                line("TRY A NATIVE INTERACTION", RecordingHud.ACCENT),
                line("Brush: hold a brush, aim within reach and right-click the fly. Close this panel first.", RecordingHud.TEXT),
                line("Creative/operator setup: /give @s minecraft:brush 1 and /give @s minecraft:sugar 8.", RecordingHud.MUTED),
                line("Food: drop sugar in front of the fly near a supported landing surface. Food is not automatically sought.", RecordingHud.TEXT),
                line("Approach: walk toward a front/side ray. Light: change illumination actually reaching the fly.", RecordingHud.TEXT),
                line("Pads are staging marks; they neither stimulate neurons nor force body programs. Close this panel, then press the model-details key (default H).", RecordingHud.MUTED));
    }

    private static NeuroCraftHud.Line line(String text, int color) { return new NeuroCraftHud.Line(text, color); }
    private static String bilateral(String name, double left, double right) {
        return String.format(Locale.ROOT, "%s L %+.3e / R %+.3e", name, left, right);
    }
    private static String formulaLabel(String formula) {
        return switch (formula) {
            case "absolute_change" -> "Absolute change";
            case "positive_change" -> "Positive change";
            case "absolute_readout" -> "Rectified readout";
            default -> throw new IllegalArgumentException("unknown source formula");
        };
    }

    @Override public void render(GuiGraphics graphics, int mouseX, int mouseY, float delta) {
        graphics.fill(0, 0, width, height, 0xFF101920);
        int left = Math.max(14, (width - 620) / 2);
        int right = width - left;
        graphics.drawString(font, font.plainSubstrByWidth(title.getString(), right - left), left, 12, RecordingHud.TEXT, false);
        String status = "Snapshot · world continues · " + samples.size() + "/" + EventReplayHistory.MAX_SAMPLES + " received samples";
        graphics.drawString(font, font.plainSubstrByWidth(status, right - left), left, 27, RecordingHud.ATTENTION, false);
        String cursor = samples.isEmpty() ? "No matched interaction samples yet. Try an interaction, then Load latest."
                : "Sample " + (index + 1) + "/" + samples.size() + " · arrows step · Space plays · scroll for details";
        var text = new ArrayList<NeuroCraftHud.Line>();
        text.add(line(cursor, RecordingHud.MUTED));
        if (!samples.isEmpty()) { text.addAll(sampleLines(samples.get(index))); }
        text.add(line("Display playback: 5 samples/s; gaps are not timed. Esc or the replay key returns to the fly.", RecordingHud.MUTED));
        text.add(line("Sparse display capture can omit commands/events/refusals. This is not a complete causal trace or reaction timer.", RecordingHud.MUTED));
        text.addAll(interactionHelp());
        var lines = RecordingHud.fitLines(font, text, right - left - 12, Integer.MAX_VALUE);
        int top = 47, bottom = height - 44;
        maxScroll = Math.max(0, lines.size() * 13 - (bottom - top));
        scroll = Math.min(scroll, maxScroll);
        graphics.enableScissor(left, top, right, bottom);
        int y = top - scroll;
        for (var row : lines) { graphics.drawString(font, row.text(), left, y, row.color(), false); y += 13; }
        graphics.disableScissor();
        if (!samples.isEmpty()) {
            int length = Math.max(1, right - left);
            graphics.fill(left, height - 38, right, height - 36, 0xFF33424D);
            int marker = left + (samples.size() < 2 ? 0 : (length - 3) * index / (samples.size() - 1));
            graphics.fill(marker, height - 39, marker + 3, height - 35, RecordingHud.ACCENT);
        }
        super.render(graphics, mouseX, mouseY, delta);
    }
}
