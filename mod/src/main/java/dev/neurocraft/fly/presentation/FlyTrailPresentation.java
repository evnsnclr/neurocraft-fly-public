package dev.neurocraft.fly.presentation;

import dev.neurocraft.fly.FlyControllerRuntime;
import dev.neurocraft.fly.NeuroCraftFly;
import dev.neurocraft.fly.entity.FlyEntity;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.minecraft.core.particles.DustParticleOptions;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.phys.Vec3;

/**
 * Cosmetic server-side dust trail behind loaded flies. Presentation only: emission runs at
 * END_SERVER_TICK, strictly after every entity tick and control decision, reads entity state
 * through public accessors without writing any field, uses no RNG, and never touches sensing,
 * actuation, protocol frames, or trace identities.
 */
public final class FlyTrailPresentation {
    public static final int DEFAULT_INTERVAL_TICKS = 3;
    public static final int MIN_INTERVAL_TICKS = 1;
    public static final int MAX_INTERVAL_TICKS = 40;
    /** Hard emission budget per server tick regardless of loaded fly count. */
    public static final int MAX_PARTICLES_PER_TICK = 64;
    private static final float PARTICLE_SCALE = 0.8F;

    private static boolean enabled;
    private static int intervalTicks = DEFAULT_INTERVAL_TICKS;
    private static long emittedParticleCount;

    private FlyTrailPresentation() {}

    public static void registerLifecycle() {
        ServerTickEvents.END_SERVER_TICK.register(FlyTrailPresentation::emit);
        ServerLifecycleEvents.SERVER_STOPPING.register(server -> resetToDefaults());
    }

    public static boolean enabled() {
        return enabled;
    }

    public static void setEnabled(boolean value) {
        enabled = value;
    }

    public static int intervalTicks() {
        return intervalTicks;
    }

    public static void setIntervalTicks(int ticks) {
        if (ticks < MIN_INTERVAL_TICKS || ticks > MAX_INTERVAL_TICKS) {
            throw new IllegalArgumentException("trail interval must be within "
                    + MIN_INTERVAL_TICKS + ".." + MAX_INTERVAL_TICKS + " ticks");
        }
        intervalTicks = ticks;
    }

    /** Total particles handed to the packet layer; observable by headless tests. */
    public static long emittedParticleCount() {
        return emittedParticleCount;
    }

    private static void emit(MinecraftServer server) {
        // Deterministic cadence: a pure function of the authoritative server tick counter.
        if (!enabled || server.getTickCount() % intervalTicks != 0) {
            return;
        }
        DustParticleOptions particle = new DustParticleOptions(
                FlyControllerRuntime.configuredDecoderHash()
                        .map(FlyTrailColor::fromDecoderHash)
                        .orElse(FlyTrailColor.FALLBACK_RGB),
                PARTICLE_SCALE);
        int budget = MAX_PARTICLES_PER_TICK;
        for (ServerLevel level : server.getAllLevels()) {
            for (FlyEntity fly : level.getEntities(NeuroCraftFly.FLY, candidate -> true)) {
                if (budget < 1) {
                    return;
                }
                // Exact bounding-box center with zero spread and zero speed: the emitted
                // packet carries no server-derived randomness and mutates no world state.
                Vec3 center = fly.getBoundingBox().getCenter();
                level.sendParticles(
                        particle, center.x, center.y, center.z, 1, 0.0, 0.0, 0.0, 0.0);
                budget -= 1;
                emittedParticleCount += 1;
            }
        }
    }

    private static void resetToDefaults() {
        enabled = false;
        intervalTicks = DEFAULT_INTERVAL_TICKS;
        emittedParticleCount = 0;
    }
}
