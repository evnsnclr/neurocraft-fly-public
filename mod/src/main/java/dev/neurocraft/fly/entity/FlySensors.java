package dev.neurocraft.fly.entity;

import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import dev.neurocraft.fly.control.SensorRayHit;
import java.util.List;
import java.util.Optional;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.Mob;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.level.ClipContext;
import net.minecraft.world.level.LightLayer;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.EntityHitResult;
import net.minecraft.world.phys.HitResult;
import net.minecraft.world.phys.Vec3;

/** Samples a fixed raw-physical vector at the start of one authoritative server entity tick. */
public final class FlySensors {
    public static final double RAY_LENGTH_BLOCKS = 6.0;

    private static final Vec3 WORLD_UP = new Vec3(0.0, 1.0, 0.0);

    /**
     * Entity hitbox inflation for the sensor sweep, declared at zero (ADR 0017).
     *
     * <p>This constant exists to be zero <em>explicitly</em>. Minecraft's six-argument
     * {@code ProjectileUtil.getEntityHitResult(Level, Entity, Vec3, Vec3, AABB, Predicate)} does not
     * take an inflation; it hard-codes {@code 0.3F} and forwards to the seven-argument form
     * (verified from the shipped 1.21.5 bytecode: {@code ldc #21 // float 0.3f}). That default is
     * projectile aim leniency -- it makes arrows feel generous -- and it is wrong here twice over.
     * It would report an obstruction 0.3 blocks nearer than the body actually is, and the reported
     * scalar is consumed by the Python encoder as {@code proximity = clip(1 - d/6, 0, 1)}, so a
     * silent 0.3 would become a silent +5% standing proximity bias on every entity reading. The
     * seven-argument overload's geometry is therefore retained with this zero. Its minimum-hit
     * selection is restated below solely to resolve entity/entity ties by UUID for actor metadata.
     */
    private static final float ENTITY_HITBOX_INFLATION_BLOCKS = 0.0F;

    private FlySensors() {}

    /**
     * The declared predicate for what counts as an obstruction on a sensor ray (ADR 0017).
     *
     * <p>Widening the rays from blocks-only to blocks-and-entities requires saying which entities
     * are part of the world the fly can sense. This is a declared engineering choice, exactly as the
     * ray directions and the 6.0-block length are (ADR 0015); nothing about it is derived from the
     * connectome or from fly biology. An entity obstructs a sensor ray if and only if all four
     * clauses hold:
     *
     * <ol>
     *   <li><b>It is not a {@link FlyEntity}.</b> Flies do not sense each other. That is the
     *       conservative choice: a second fly is a degenerate configuration (the operator commands
     *       one per dimension), and mutual sensing would close a feedback loop between two
     *       connectome runtimes that no measurement in this repository accounts for.
     *       <p><b>This clause is currently redundant, measured rather than assumed.</b> The sensing
     *       fly is already excluded upstream by {@code Level.getEntities(Entity except, ...)}, and
     *       a <em>second</em> fly is excluded by clause 4, because {@code FlyEntity} extends
     *       {@code Entity} directly and so inherits {@code isPickable() == false}. Deleting this
     *       clause and re-running the server GameTest lane leaves all 32 tests passing, so <b>no
     *       test binds it</b>. It is kept as declared intent that survives a future change making
     *       the fly pickable -- letting an operator target it, say -- which would otherwise switch
     *       fly-to-fly sensing on silently as a side effect.
     *   <li><b>It is not removed.</b> A despawned or killed entity is not in the world.
     *   <li><b>It is not a spectator.</b> Spectators are observers, not bodies: vanilla excludes
     *       them from every collision and targeting path. An operator watching a run must not
     *       perturb the run by being looked at.
     *   <li><b>It is ray-pickable</b> ({@link Entity#isPickable()}). This is vanilla's own contract
     *       for "a ray cast can hit this", which is precisely this sensor's question, so it is used
     *       rather than restated. Read from the shipped 1.21.5 bytecode, the five overrides that
     *       decide the cases named here are {@code Entity.isPickable()} returning {@code false};
     *       {@code LivingEntity} overriding it to {@code !isRemoved()}; {@code Projectile}
     *       overriding it to {@code getType().is(EntityTypeTags.REDIRECTABLE_PROJECTILE)};
     *       {@code AbstractArrow} refining that to {@code Projectile.isPickable() &&
     *       !isInGround()}; and {@code ShulkerBullet} returning an unconditional {@code true}. The
     *       tag file {@code data/minecraft/tags/entity_type/redirectable_projectile.json} holds
     *       exactly {@code minecraft:fireball} (the ghast's {@code LargeFireball}),
     *       {@code minecraft:wind_charge} and {@code minecraft:breeze_wind_charge}. So:
     *       <ul>
     *         <li><b>Players and mobs count</b> -- the whole point of this change.
     *         <li><b>Item entities and experience orbs do not.</b> Both extend {@code Entity}
     *             directly and declare no {@code isPickable}. That exclusion is wanted on its own
     *             terms: the fly's bounded actuation passes straight through a dropped item, and a
     *             sensory signal for something that cannot obstruct the body is a phantom reading.
     *         <li><b>Four projectile types count and the rest do not.</b> A ghast fireball, a wind
     *             charge and a breeze wind charge resolve {@code true} through the tag, and a
     *             shulker bullet through its own unconditional {@code true}; arrows, snowballs,
     *             ender pearls, eggs, small and dragon fireballs, thrown potions and fishing
     *             bobbers all resolve {@code false}, arrows twice over since they are also outside
     *             the tag. <b>Admitting the four is a declared choice, not an accident of
     *             inheritance</b> (ADR 0017): this predicate exists to admit bodies the fly's
     *             bounded actuation cannot pass through, and a redirectable projectile is exactly a
     *             thing that physically interacts with the body -- a wind charge knocks the fly
     *             around -- so it also gives the operator a legitimate fast-approaching stimulus.
     *             Pinned by the GameTest
     *             {@code taggedRedirectableProjectileObstructsWhileUntaggedProjectileDoesNot}.
     *       </ul>
     * </ol>
     *
     * <p><b>Not used, deliberately:</b> {@link Entity#canBeCollidedWith()}. It reads like the
     * symmetric partner of the block side's {@code ClipContext.Block.COLLIDER}, but in 1.21.5 it is
     * {@code false} on {@code Entity} and overridden by neither {@code LivingEntity} nor
     * {@code Player} (hard entity-entity collision belongs to boats and shulkers). It would have
     * excluded the operator, which is the one case this change exists to fix.
     */
    public static boolean obstructsSensorRay(Entity candidate) {
        return !(candidate instanceof FlyEntity)
                && !candidate.isRemoved()
                && !candidate.isSpectator()
                && candidate.isPickable();
    }

    public static PhysicalSensorSnapshot sample(ServerLevel level, FlyEntity entity) {
        Vec3 position = entity.position();
        Vec3 velocity = entity.getDeltaMovement();
        Vec3 origin = entity.getBoundingBox().getCenter();
        // Body frame, ADR 0015. In Minecraft's basis (+X east, +Y up, +Z south) a body's
        // right is Direction.getClockWise() of its heading: facing south, right is west.
        // At yaw 0 forward is (0,0,+1) south, so right must be (-1,0,0) west, which is
        // forward x up -- NOT up x forward, which yields east and is the body's left.
        // bodyUp is then right x forward; forward x right would be down. Pitch is clamped
        // to +/-89 degrees (PrimitiveActuator.MAX_ABS_PITCH_DEGREES), so |forward x up| is
        // at worst cos(89 deg) ~= 0.017, far above Vec3.normalize's 1.0e-5 zero threshold.
        Vec3 forward = Vec3.directionFromRotation(entity.getXRot(), entity.getYRot()).normalize();
        Vec3 right = forward.cross(WORLD_UP).normalize();
        Vec3 bodyUp = right.cross(forward).normalize();
        SensorRayHit[] rays = {
            rayHit(level, entity, origin, forward),
            rayHit(level, entity, origin, forward.reverse()),
            rayHit(level, entity, origin, right.reverse()),
            rayHit(level, entity, origin, right),
            rayHit(level, entity, origin, bodyUp),
            rayHit(level, entity, origin, bodyUp.reverse())
        };
        return new PhysicalSensorSnapshot(
                level.getGameTime(),
                entity.getStringUUID(),
                position.x,
                position.y,
                position.z,
                velocity.x,
                velocity.y,
                velocity.z,
                entity.getYRot(),
                entity.getXRot(),
                entity.horizontalCollision,
                entity.verticalCollision,
                entity.onGround(),
                entity.isInWater(),
                entity.isInLava(),
                level.getBrightness(LightLayer.BLOCK, entity.blockPosition()),
                level.getBrightness(LightLayer.SKY, entity.blockPosition()),
                RAY_LENGTH_BLOCKS,
                rays[0].distance(), rays[1].distance(), rays[2].distance(),
                rays[3].distance(), rays[4].distance(), rays[5].distance(),
                List.of(rays[0].kind(), rays[1].kind(), rays[2].kind(),
                        rays[3].kind(), rays[4].kind(), rays[5].kind()),
                List.of(rays[0].actorKind(), rays[1].actorKind(), rays[2].actorKind(),
                        rays[3].actorKind(), rays[4].actorKind(), rays[5].actorKind()), entity.eventSample());
    }

    /**
     * Distance to the nearest obstruction along one declared ray, clamped to the ray length.
     *
     * <p>ADR 0017. The ray is cast twice over the <em>same</em> segment -- once against blocks,
     * once against entities admitted by {@link #obstructsSensorRay} -- and the nearer hit wins. The
     * declared length, the body-aligned directions (ADR 0015), the bounding-box-centre origin and
     * the returned meaning are all unchanged; only the set of things that can obstruct is wider.
     *
     * <p><b>Determinism.</b> The minimum distance remains unchanged. The kind records which
     * sweep produced it: 0 none, 1 block, 2 entity. An exact block/entity distance tie
     * selects entity. Exact entity/entity squared-distance ties select the smallest UUID under
     * {@link java.util.UUID#compareTo}; all fields describe that same winning hit.
     *
     * <p><b>Cost.</b> One {@code Level.getEntities} broad-phase per ray, six rays per server tick.
     * The search box is {@code new AABB(origin, end)} -- the segment's own bounding box, degenerate
     * in two dimensions for an axis-aligned ray -- not an inflated or unioned box, so the swept
     * volume is the tightest one that can still contain every entity the segment passes through.
     */
    private static SensorRayHit rayHit(
            ServerLevel level, FlyEntity entity, Vec3 origin, Vec3 direction) {
        Vec3 end = origin.add(direction.scale(RAY_LENGTH_BLOCKS));
        Double blockDistance = null;

        HitResult blockHit = level.clip(new ClipContext(
                origin, end, ClipContext.Block.COLLIDER, ClipContext.Fluid.NONE, entity));
        if (blockHit.getType() != HitResult.Type.MISS) {
            blockDistance = origin.distanceTo(blockHit.getLocation());
        }

        EntityHitResult entityHit = nearestEntityHit(level, entity, origin, end);
        Double entityDistance = entityHit == null ? null : origin.distanceTo(entityHit.getLocation());
        int actorKind = entityHit == null ? 0
                : entityHit.getEntity() instanceof Player ? 1
                : entityHit.getEntity() instanceof Mob ? 2 : 3;
        return SensorRayHit.nearest(RAY_LENGTH_BLOCKS, blockDistance, entityDistance, actorKind);
    }

    /**
     * Minecraft 1.21.5's seven-argument ProjectileUtil geometry, with only its exact tie changed.
     * In particular AABB.clip retains its own endpoint and inside-box behavior; the separate
     * overload's origin-inside shortcut, pick radius and vehicle exclusions are not introduced.
     */
    private static EntityHitResult nearestEntityHit(
            ServerLevel level, FlyEntity entity, Vec3 origin, Vec3 end) {
        double bestSquaredDistance = Double.MAX_VALUE;
        Entity winner = null;
        Vec3 location = null;
        for (Entity candidate : level.getEntities(entity, new AABB(origin, end), FlySensors::obstructsSensorRay)) {
            Optional<Vec3> clipped = candidate.getBoundingBox()
                    .inflate(ENTITY_HITBOX_INFLATION_BLOCKS).clip(origin, end);
            if (clipped.isEmpty()) { continue; }
            double squaredDistance = origin.distanceToSqr(clipped.get());
            if (squaredDistance < bestSquaredDistance
                    || (squaredDistance == bestSquaredDistance && winner != null
                            && candidate.getUUID().compareTo(winner.getUUID()) < 0)) {
                winner = candidate;
                location = clipped.get();
                bestSquaredDistance = squaredDistance;
            }
        }
        return winner == null ? null : new EntityHitResult(winner, location);
    }
}
