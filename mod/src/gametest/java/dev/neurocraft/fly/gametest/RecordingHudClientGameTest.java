package dev.neurocraft.fly.gametest;

import com.mojang.blaze3d.platform.InputConstants;
import dev.neurocraft.fly.NeuroCraftFly;
import dev.neurocraft.fly.body.*;
import dev.neurocraft.fly.client.ModelDetailsScreen;
import dev.neurocraft.fly.client.EventReplayScreen;
import dev.neurocraft.fly.client.NeuroCraftHud;
import dev.neurocraft.fly.client.RecordingHud;
import dev.neurocraft.fly.control.*;
import dev.neurocraft.fly.entity.FlyEntity;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Optional;
import net.fabricmc.fabric.api.client.gametest.v1.FabricClientGameTest;
import net.fabricmc.fabric.api.client.gametest.v1.context.ClientGameTestContext;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.gui.GuiGraphics;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.network.chat.Component;
import net.minecraft.world.entity.EntitySpawnReason;
import org.lwjgl.glfw.GLFW;

/** Rendering/interaction checks with explicitly synthetic display samples; no neural benchmark. */
public final class RecordingHudClientGameTest implements FabricClientGameTest {
    @Override
    public void runTest(ClientGameTestContext context) {
        final EventDisplayFrame event;
        try (var stream = getClass().getResourceAsStream("/neurocraft/interaction-display-v2.json")) {
            if (stream == null) { throw new AssertionError("missing display fixture"); }
            event = EventDisplayFrame.parse(new String(stream.readAllBytes(), StandardCharsets.UTF_8));
        } catch (Exception exception) { throw new AssertionError(exception); }
        var populations = new LinkedHashMap<String, BrainDisplayFrame.Population>();
        for (String name : BrainDisplayFrame.POPULATION_NAMES) {
            populations.put(name, new BrainDisplayFrame.Population(-0.00001, 1));
        }
        var frame = new BrainDisplayFrame(event.identity(), 5, 5, 25, 30, "e".repeat(64),
                "neurocraft.events.encoder.v1", "3".repeat(64), populations);
        var brain = new BrainDisplayPresentation(BrainDisplayState.Status.MATCHED, 100, Optional.of(frame));
        var applied = new EventDisplayPresentation(EventDisplayState.Status.MATCHED, 100, Optional.of(event));
        var body = new BodyProgramChannel(SelectionSource.READOUT, BodyProgramId.CRUISE,
                Modulation.NEUTRAL, ProgramParameters.NEUTRAL);
        var lane = new ComparisonLanePresentation(ComparisonLanePresentation.Status.ACTIVE,
                "original", event.identity().sessionEpoch(), Optional.of(event.identity()), "original",
                "faithful", null, null);
        var view = RecordingHud.view(ControllerFault.ACTIVE, body, brain, applied, lane, 20);
        context.runOnClient(client -> {
            String live = text(view);
            // Current body and matched sample intentionally disagree: never merge them.
            for (String required : List.of("Body now   Cruise", "Applied sample   #26", "Program   Hover",
                    "Light change", "DNa01", "Model sample   #5", "-1.0e-05", "Original weights", "scripted motion")) {
                if (!live.contains(required)) { throw new AssertionError("missing " + required + ": " + live); }
            }
            for (var state : ControllerFault.values()) {
                for (var authority : SelectionSource.values()) {
                    var channel = new BodyProgramChannel(authority, BodyProgramId.CRUISE,
                            Modulation.NEUTRAL, ProgramParameters.NEUTRAL);
                    for (var comparison : List.of(lane,
                            ComparisonLanePresentation.waiting(ComparisonLanePresentation.Status.SWITCHING,
                                    "no_input", event.identity().sessionEpoch()),
                            ComparisonLanePresentation.waiting(ComparisonLanePresentation.Status.FAILED,
                                    "shuffled_weight", event.identity().sessionEpoch()))) {
                        String actual = text(RecordingHud.view(state, channel, brain, applied, comparison, 20));
                        boolean permitted = state == ControllerFault.ACTIVE && authority == SelectionSource.READOUT
                                && comparison.status() == ComparisonLanePresentation.Status.ACTIVE;
                        boolean modelPermitted = state == ControllerFault.ACTIVE && authority != SelectionSource.OPERATOR
                                && comparison.status() == ComparisonLanePresentation.Status.ACTIVE;
                        if (actual.contains("-1.0e-05") != modelPermitted || actual.contains("Light change") != permitted) {
                            throw new AssertionError("stale state leaked into recording HUD: " + actual);
                        }
                    }
                }
            }
            var absent = RecordingHud.view(ControllerFault.ACTIVE, body,
                    new BrainDisplayPresentation(BrainDisplayState.Status.STALE, 1000, Optional.empty()),
                    EventDisplayPresentation.unavailable(), lane, 10);
            if (text(absent).contains("0.0e") || !text(absent).contains("10 TPS")) {
                throw new AssertionError("missing values became zero or changed world target was hidden");
            }
            for (int width : List.of(125, 145, 180, 224, 300)) {
                var lines = RecordingHud.fitLines(client.font, RecordingHud.lines(view), width, 220);
                for (var line : lines) {
                    if (client.font.width(line.text()) > width) {
                        throw new AssertionError("recording HUD clips " + line.text());
                    }
                }
                if (20 + lines.size() * 12 > 220) { throw new AssertionError("recording HUD exceeds viewport"); }
                var shortLines = RecordingHud.fitLines(client.font, RecordingHud.lines(view), width, 100);
                if (20 + shortLines.size() * 12 > 100) { throw new AssertionError("compact fallback exceeds viewport"); }
            }
            String details = String.join(" ", ModelDetailsScreen.explanation().stream().map(NeuroCraftHud.Line::text).toList());
            for (String required : List.of("Hand-chosen", "scripted body programs", "128-neuron fixture",
                    "synthetic imitation", "95.66x", "403.70x",
                    "300x", "35.1x", "4 neural steps", "not live reaction timing")) {
                if (!details.contains(required)) { throw new AssertionError("details lost " + required); }
            }
            client.setScreen(new Screen(Component.literal("Synthetic HUD layout fixture")) {
                @Override public void render(GuiGraphics graphics, int x, int y, float delta) {
                    graphics.fill(0, 0, width, height, 0xFF334C49);
                    RecordingHud.render(graphics, font, view);
                    graphics.drawString(font, "Synthetic UI fixture / no live experiment", 10, height - 16,
                            RecordingHud.TEXT, false);
                }
            });
        });
        context.waitTicks(2);
        context.takeScreenshot("recording-hud-synthetic-preview");
        context.runOnClient(client -> client.setScreen(new EventReplayScreen(() -> List.of(
                new EventReplayHistory.Sample(event, "original", null)))));
        context.waitTicks(2);
        context.takeScreenshot("event-replay-synthetic-preview");
        context.runOnClient(client -> client.setScreen(null));
        try (var world = context.worldBuilder().create()) {
            world.getServer().runOnServer(server -> {
                FlyEntity fly = NeuroCraftFly.FLY.create(server.overworld(), EntitySpawnReason.COMMAND);
                if (fly == null) { throw new AssertionError("could not create details-key fixture"); }
                fly.resetForFixture(0, 81.5, 4, 180);
                server.overworld().addFreshEntity(fly);
            });
            world.getServer().runCommand("tp @p 0 80 0 0 0");
            context.waitFor(client -> client.level != null
                    && java.util.stream.StreamSupport.stream(client.level.entitiesForRendering().spliterator(), false)
                            .anyMatch(entity -> entity instanceof FlyEntity));
            // Exercise the registered key callback against a real client level.
            context.runOnClient(client -> KeyMapping.click(InputConstants.Type.KEYSYM.getOrCreate(GLFW.GLFW_KEY_H)));
            context.waitFor(client -> client.screen instanceof ModelDetailsScreen);
            context.runOnClient(client -> {
                if (client.screen.isPauseScreen()) { throw new AssertionError("details pauses simulation"); }
                client.screen.mouseScrolled(100, 100, 0, -12);
            });
            context.waitTicks(2);
            context.takeScreenshot("recording-hud-details-preview");
            context.runOnClient(client -> client.screen.keyPressed(GLFW.GLFW_KEY_H, 0, 0));
            context.waitFor(client -> client.screen == null);
            context.runOnClient(client -> KeyMapping.click(InputConstants.Type.KEYSYM.getOrCreate(GLFW.GLFW_KEY_R)));
            context.waitFor(client -> client.screen instanceof EventReplayScreen);
            context.runOnClient(client -> {
                if (client.screen.isPauseScreen()) { throw new AssertionError("replay pauses the world"); }
                client.screen.keyPressed(GLFW.GLFW_KEY_RIGHT, 0, 0);
                client.screen.keyPressed(GLFW.GLFW_KEY_SPACE, 0, 0);
            });
            context.waitTicks(2);
            context.takeScreenshot("event-replay-empty-preview");
            context.runOnClient(client -> client.screen.keyPressed(GLFW.GLFW_KEY_R, 0, 0));
            context.waitFor(client -> client.screen == null);
        }
    }

    private static String text(RecordingHud.View view) {
        return String.join(" ", RecordingHud.lines(view).stream().map(NeuroCraftHud.Line::text).toList());
    }
}
