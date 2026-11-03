package dev.neurocraft.fly;

import dev.neurocraft.fly.control.CommandValidator;
import dev.neurocraft.fly.entity.FlyEntity;
import dev.neurocraft.fly.presentation.FlyTrailPresentation;
import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.minecraft.core.Registry;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.core.registries.Registries;
import net.minecraft.resources.ResourceKey;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.world.entity.EntityType;
import net.minecraft.world.entity.MobCategory;

/** Common mod entrypoint; contains no client classes. */
public final class NeuroCraftFly implements ModInitializer {
    public static final String MOD_ID = "neurocraft_fly";
    public static final ResourceLocation FLY_ID =
            ResourceLocation.fromNamespaceAndPath(MOD_ID, "fly");
    public static final EntityType<FlyEntity> FLY = registerFly();

    @Override
    public void onInitialize() {
        FlyControllerRuntime.registerLifecycle();
        FlyTrailPresentation.registerLifecycle();
        // The validator stays Fabric-free, so its process-global counter's
        // stop-time reset registers here, mirroring the trail convention.
        ServerLifecycleEvents.SERVER_STOPPING.register(
                server -> CommandValidator.resetTotalLatenessRetirements());
        FlyCommands.register();
    }

    private static EntityType<FlyEntity> registerFly() {
        ResourceKey<EntityType<?>> key = ResourceKey.create(Registries.ENTITY_TYPE, FLY_ID);
        EntityType<FlyEntity> type = EntityType.Builder.<FlyEntity>of(FlyEntity::new, MobCategory.MISC)
                .sized(0.38F, 0.22F)
                .eyeHeight(0.11F)
                .clientTrackingRange(8)
                .updateInterval(1)
                .noLootTable()
                .build(key);
        return Registry.register(BuiltInRegistries.ENTITY_TYPE, key, type);
    }
}
