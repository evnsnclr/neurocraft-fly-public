package dev.neurocraft.fly.client;

import dev.neurocraft.fly.NeuroCraftFly;
import dev.neurocraft.fly.control.DiagnosticClientActions;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.rendering.v1.EntityModelLayerRegistry;
import net.fabricmc.fabric.api.client.rendering.v1.EntityRendererRegistry;
import net.fabricmc.fabric.api.client.rendering.v1.HudLayerRegistrationCallback;
import net.fabricmc.fabric.api.client.rendering.v1.IdentifiedLayer;
import net.minecraft.client.model.geom.ModelLayerLocation;
import net.minecraft.resources.ResourceLocation;

/** Client rendering and opt-in diagnostic interactions; no access to the bridge runtime. */
public final class NeuroCraftFlyClient implements ClientModInitializer {
    public static final ModelLayerLocation FLY_MODEL_LAYER = new ModelLayerLocation(
            ResourceLocation.fromNamespaceAndPath(NeuroCraftFly.MOD_ID, "fly"), "main");
    private static final ResourceLocation HUD_LAYER = ResourceLocation.fromNamespaceAndPath(
            NeuroCraftFly.MOD_ID, "controller_status");

    @Override
    public void onInitializeClient() {
        EntityModelLayerRegistry.registerModelLayer(FLY_MODEL_LAYER, FlyModel::createBodyLayer);
        EntityRendererRegistry.register(NeuroCraftFly.FLY, FlyRenderer::new);
        RecordingHudControls.register();
        EventReplayControls.register();
        HudLayerRegistrationCallback.EVENT.register(layeredDrawer -> layeredDrawer.attachLayerAfter(
                IdentifiedLayer.MISC_OVERLAYS, HUD_LAYER, NeuroCraftHud::render));
        LiveFrameCapture.register();
        DiagnosticClientActions.register();
    }
}
