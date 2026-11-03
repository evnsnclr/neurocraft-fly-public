package dev.neurocraft.fly.client;

import com.mojang.blaze3d.platform.InputConstants;
import dev.neurocraft.fly.entity.FlyEntity;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.KeyMapping;
import org.lwjgl.glfw.GLFW;

/** A remappable client-only details key; never sends commands or changes program authority. */
public final class RecordingHudControls {
    public static final KeyMapping DETAILS = new KeyMapping("key.neurocraft_fly.details",
            InputConstants.Type.KEYSYM, GLFW.GLFW_KEY_H, "key.categories.neurocraft_fly");

    private RecordingHudControls() {}

    public static void register() {
        KeyBindingHelper.registerKeyBinding(DETAILS);
        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            while (DETAILS.consumeClick()) {
                if (client.level == null || client.screen != null) {
                    continue;
                }
                for (var entity : client.level.entitiesForRendering()) {
                    if (entity instanceof FlyEntity fly) {
                        client.setScreen(new ModelDetailsScreen(fly));
                        break;
                    }
                }
            }
        });
    }
}
