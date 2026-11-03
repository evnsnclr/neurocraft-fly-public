package dev.neurocraft.fly.client;

import dev.neurocraft.fly.entity.FlyEntity;
import java.util.ArrayList;
import java.util.List;
import net.minecraft.client.gui.GuiGraphics;
import net.minecraft.client.gui.components.Button;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.network.chat.Component;

/** Scrollable explanation and live diagnostics. Opening this screen never pauses the model. */
public final class ModelDetailsScreen extends Screen {
    private final FlyEntity fly;
    private int scroll;
    private int maxScroll;

    public ModelDetailsScreen(FlyEntity fly) {
        super(Component.literal("Model & experiment details"));
        this.fly = fly;
    }

    @Override
    protected void init() {
        addRenderableWidget(Button.builder(Component.literal("Back to fly"), button -> onClose())
                .bounds(width / 2 - 60, height - 28, 120, 20).build());
    }

    @Override
    public boolean isPauseScreen() {
        return false;
    }

    @Override
    public void renderBackground(GuiGraphics graphics, int mouseX, int mouseY, float delta) {
        // This screen paints its own solid panel. Screen.render calls this after our text;
        // vanilla's default blur would otherwise blur the explanatory content itself.
    }

    @Override
    public boolean keyPressed(int keyCode, int scanCode, int modifiers) {
        if (RecordingHudControls.DETAILS.matches(keyCode, scanCode)) {
            onClose();
            return true;
        }
        return super.keyPressed(keyCode, scanCode, modifiers);
    }

    @Override
    public boolean mouseScrolled(double mouseX, double mouseY, double horizontal, double vertical) {
        scroll = Math.max(0, Math.min(maxScroll, scroll - (int) (vertical * 30)));
        return true;
    }

    @Override
    public boolean mouseClicked(double mouseX, double mouseY, int button) {
        if (RecordingHudControls.DETAILS.matchesMouse(button)) {
            onClose();
            return true;
        }
        return super.mouseClicked(mouseX, mouseY, button);
    }

    public static List<NeuroCraftHud.Line> explanation() {
        int text = RecordingHud.TEXT;
        int muted = RecordingHud.MUTED;
        int accent = RecordingHud.ACCENT;
        return List.of(
                new NeuroCraftHud.Line("How this model works", accent),
                new NeuroCraftHud.Line("Minecraft events stimulate named sensory populations through engineered mappings. "
                        + "Hand-chosen neural readouts select and modulate scripted body programs.", text),
                new NeuroCraftHud.Line("Activity values are signed model quantities, not recorded spike rates. "
                        + "Minecraft supplies player and mob categories. DNp01 is displayed for inspection only.", muted),
                new NeuroCraftHud.Line("Reading the overlay", accent),
                new NeuroCraftHud.Line("Body now is the current synchronized movement. Applied sample pairs a historical "
                        + "input and program with the same received command. Model samples have their own sequence. "
                        + "Received age is file-read age, not reaction latency. Missing samples remain unavailable.", text),
                new NeuroCraftHud.Line("Comparison controls", accent),
                new NeuroCraftHud.Line("Original weights, shuffled weights and no sensory input are distinct conditions. "
                        + "A switch starts a fresh neural epoch. Repeat the same stimulus after it becomes active. "
                        + "Shuffling preserves connection endpoints and signs. No-input can still run baseline Cruise.", text),
                new NeuroCraftHud.Line("Historical measurements", accent),
                new NeuroCraftHud.Line("128-neuron fixture, synthetic imitation task: 95.66x raw and 403.70x bounded "
                        + "MSE relative to the matched direct controller (ADR 0019; scope in ADR 0038). These "
                        + "historical results do not measure full-network control or current gameplay.", text),
                new NeuroCraftHud.Line("Offline ESCAPE-ARC-01: DNp01 approximately 300x the wing-pool response; TTMn "
                        + "35.1x its near-silent baseline; propagation in 4 neural steps. These are offline model "
                        + "measurements, not live reaction timing or biological validation.", text),
                new NeuroCraftHud.Line("Live diagnostic values", accent));
    }

    @Override
    public void render(GuiGraphics graphics, int mouseX, int mouseY, float delta) {
        graphics.fill(0, 0, width, height, 0xFF101920);
        int left = Math.max(18, (width - 560) / 2);
        int right = width - left;
        graphics.drawString(font, title, left, 14, RecordingHud.TEXT, false);
        String hint = "Scroll to inspect · Esc to return · panel does not pause the world";
        graphics.drawString(font, font.plainSubstrByWidth(hint, right - left), left, 29,
                RecordingHud.MUTED, false);
        List<NeuroCraftHud.Line> source = new ArrayList<>(explanation());
        if (!fly.isRemoved()) {
            // Keep evidence and original strings accessible, without the old repeated warning banners.
            for (var line : NeuroCraftHud.detailsLines(fly)) {
                if (line.text().contains("HAND-CHOSEN") || line.text().contains("NOT THE FLY DECIDING")
                        || line.text().startsWith("128-NEURON") || line.text().startsWith("MSE 95.66")
                        || line.text().startsWith("OFFLINE ") || line.text().startsWith("DNp01 ~")
                        || line.text().startsWith("4 neural steps")) {
                    continue;
                }
                source.add(new NeuroCraftHud.Line(line.text(), RecordingHud.MUTED));
            }
        } else {
            source.add(new NeuroCraftHud.Line("Fly unavailable; live values are no longer shown.", RecordingHud.ATTENTION));
        }
        List<NeuroCraftHud.Line> lines = RecordingHud.fitLines(font, source, right - left - 12, Integer.MAX_VALUE);
        int top = 50;
        int bottom = height - 38;
        maxScroll = Math.max(0, lines.size() * 13 - (bottom - top));
        scroll = Math.min(scroll, maxScroll);
        graphics.enableScissor(left, top, right, bottom);
        int y = top - scroll;
        for (var line : lines) {
            graphics.drawString(font, line.text(), left, y, line.color(), false);
            y += 13;
        }
        graphics.disableScissor();
        if (maxScroll > 0) {
            int track = Math.max(1, bottom - top);
            int thumb = Math.max(12, track * track / (maxScroll + track));
            int thumbTop = top + (track - thumb) * scroll / maxScroll;
            graphics.fill(right - 3, top, right - 1, bottom, 0xFF33424D);
            graphics.fill(right - 3, thumbTop, right - 1, thumbTop + thumb, RecordingHud.ACCENT);
        }
        super.render(graphics, mouseX, mouseY, delta);
    }
}
