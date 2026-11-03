package dev.neurocraft.fly.client;

import com.mojang.blaze3d.platform.InputConstants;
import dev.neurocraft.fly.control.EventReplayHistory;
import dev.neurocraft.fly.entity.FlyEntity;
import java.util.List;
import java.util.UUID;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.multiplayer.ClientLevel;
import org.lwjgl.glfw.GLFW;

/** Read-only replay collection and remappable screen key, independent of transport and authority. */
public final class EventReplayControls {
    public static final KeyMapping REPLAY = new KeyMapping("key.neurocraft_fly.replay",
            InputConstants.Type.KEYSYM, GLFW.GLFW_KEY_R, "key.categories.neurocraft_fly");
    private static final EventReplayHistory HISTORY = new EventReplayHistory();
    private static ClientLevel trackedLevel;
    private static UUID trackedFly;

    private EventReplayControls() {}

    public static void register() {
        KeyBindingHelper.registerKeyBinding(REPLAY);
        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            FlyEntity fly = null;
            if (client.level != null) {
                for (var entity : client.level.entitiesForRendering()) {
                    if (entity instanceof FlyEntity candidate && !candidate.isRemoved()) {
                        fly = candidate;
                        break;
                    }
                }
            }
            if (client.level != trackedLevel || fly == null || !fly.getUUID().equals(trackedFly)) {
                HISTORY.clear();
                trackedLevel = client.level;
                trackedFly = fly == null ? null : fly.getUUID();
            }
            if (fly != null) {
                HISTORY.record(fly.eventDisplayPresentation(), fly.comparisonPresentation(),
                        fly.controllerState(), fly.bodyProgramChannel().source());
            }
            while (REPLAY.consumeClick()) {
                if (client.level != null && client.screen == null) {
                    client.setScreen(new EventReplayScreen(HISTORY::snapshot));
                }
            }
        });
    }

    public static List<EventReplayHistory.Sample> snapshot() { return HISTORY.snapshot(); }
}
