package dev.neurocraft.fly.control;

/** Nearest sweep and actor class: 0 none, 1 player, 2 mob, 3 other entity; null is unmeasured. */
public record SensorRayHit(double distance, int kind, Integer actorKind) {
    /** Older typed fixtures know only that an entity was hit, not its actor class. */
    public SensorRayHit(double distance, int kind) {
        this(distance, kind, kind == 2 ? null : 0);
    }

    public SensorRayHit {
        if (kind < 0 || kind > 2 || (actorKind != null && (actorKind < 0 || actorKind > 3))
                || (kind == 2 ? Integer.valueOf(0).equals(actorKind) : !Integer.valueOf(0).equals(actorKind))) {
            throw new IllegalArgumentException("ray hit and actor kinds disagree");
        }
    }

    public static SensorRayHit nearest(double maximum, Double block, Double entity) {
        return nearest(maximum, block, entity, entity == null ? 0 : null);
    }

    /** Exact block/entity ties retain the entity and its same-hit actor class. */
    public static SensorRayHit nearest(double maximum, Double block, Double entity, Integer entityActorKind) {
        if ((entity == null ? !Integer.valueOf(0).equals(entityActorKind) : Integer.valueOf(0).equals(entityActorKind))
                || (entityActorKind != null && (entityActorKind < 0 || entityActorKind > 3))) {
            throw new IllegalArgumentException("entity distance and actor kind disagree");
        }
        double distance = maximum;
        int kind = 0;
        if (block != null && block <= distance) {
            distance = StrictMath.min(distance, block);
            kind = 1;
        }
        if (entity != null && entity <= distance) {
            distance = StrictMath.min(distance, entity);
            kind = 2;
        }
        return new SensorRayHit(distance, kind, kind == 2 ? entityActorKind : Integer.valueOf(0));
    }
}
