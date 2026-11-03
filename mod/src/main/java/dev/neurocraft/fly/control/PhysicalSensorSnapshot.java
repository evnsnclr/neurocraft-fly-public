package dev.neurocraft.fly.control;

import java.util.List;
import java.util.Objects;

/** One immutable server-tick sample of raw physical quantities and body-relative ray distances. */
public record PhysicalSensorSnapshot(
        long gameTick,
        String entityId,
        double xBlocks,
        double yBlocks,
        double zBlocks,
        double velocityXBlocksPerTick,
        double velocityYBlocksPerTick,
        double velocityZBlocksPerTick,
        double yawDegrees,
        double pitchDegrees,
        boolean horizontalCollision,
        boolean verticalCollision,
        boolean onGround,
        boolean inWater,
        boolean inLava,
        int blockLight,
        int skyLight,
        double rayMaxDistanceBlocks,
        double rayForwardBlocks,
        double rayBackBlocks,
        double rayLeftBlocks,
        double rayRightBlocks,
        double rayUpBlocks,
        double rayDownBlocks,
        List<Integer> rayHitKinds,
        List<Integer> rayActorKinds,
        EventSample events) {

    /** Older typed snapshots carry no measured actor classification and cannot alone produce V4. */
    public PhysicalSensorSnapshot(
            long gameTick, String entityId, double xBlocks, double yBlocks, double zBlocks,
            double velocityXBlocksPerTick, double velocityYBlocksPerTick, double velocityZBlocksPerTick,
            double yawDegrees, double pitchDegrees, boolean horizontalCollision, boolean verticalCollision,
            boolean onGround, boolean inWater, boolean inLava, int blockLight, int skyLight,
            double rayMaxDistanceBlocks, double rayForwardBlocks, double rayBackBlocks,
            double rayLeftBlocks, double rayRightBlocks, double rayUpBlocks, double rayDownBlocks,
            List<Integer> rayHitKinds, EventSample events) {
        this(gameTick, entityId, xBlocks, yBlocks, zBlocks,
                velocityXBlocksPerTick, velocityYBlocksPerTick, velocityZBlocksPerTick,
                yawDegrees, pitchDegrees, horizontalCollision, verticalCollision, onGround,
                inWater, inLava, blockLight, skyLight, rayMaxDistanceBlocks, rayForwardBlocks,
                rayBackBlocks, rayLeftBlocks, rayRightBlocks, rayUpBlocks, rayDownBlocks,
                rayHitKinds, null, events);
    }

    /** Existing typed-ray snapshots do not infer an event history. */
    public PhysicalSensorSnapshot(
            long gameTick, String entityId, double xBlocks, double yBlocks, double zBlocks,
            double velocityXBlocksPerTick, double velocityYBlocksPerTick, double velocityZBlocksPerTick,
            double yawDegrees, double pitchDegrees, boolean horizontalCollision, boolean verticalCollision,
            boolean onGround, boolean inWater, boolean inLava, int blockLight, int skyLight,
            double rayMaxDistanceBlocks, double rayForwardBlocks, double rayBackBlocks,
            double rayLeftBlocks, double rayRightBlocks, double rayUpBlocks, double rayDownBlocks,
            List<Integer> rayHitKinds) {
        this(gameTick, entityId, xBlocks, yBlocks, zBlocks,
                velocityXBlocksPerTick, velocityYBlocksPerTick, velocityZBlocksPerTick,
                yawDegrees, pitchDegrees, horizontalCollision, verticalCollision, onGround,
                inWater, inLava, blockLight, skyLight, rayMaxDistanceBlocks, rayForwardBlocks,
                rayBackBlocks, rayLeftBlocks, rayRightBlocks, rayUpBlocks, rayDownBlocks, rayHitKinds, null);
    }

    public PhysicalSensorSnapshot withEvents(EventSample sample) {
        return new PhysicalSensorSnapshot(gameTick, entityId, xBlocks, yBlocks, zBlocks,
                velocityXBlocksPerTick, velocityYBlocksPerTick, velocityZBlocksPerTick,
                yawDegrees, pitchDegrees, horizontalCollision, verticalCollision, onGround,
                inWater, inLava, blockLight, skyLight, rayMaxDistanceBlocks, rayForwardBlocks,
                rayBackBlocks, rayLeftBlocks, rayRightBlocks, rayUpBlocks, rayDownBlocks, rayHitKinds,
                rayActorKinds,
                Objects.requireNonNull(sample, "sample"));
    }

    /** Legacy synthetic snapshots carry no inferred hit classifications. */
    public PhysicalSensorSnapshot(
        long gameTick,
        String entityId,
        double xBlocks,
        double yBlocks,
        double zBlocks,
        double velocityXBlocksPerTick,
        double velocityYBlocksPerTick,
        double velocityZBlocksPerTick,
        double yawDegrees,
        double pitchDegrees,
        boolean horizontalCollision,
        boolean verticalCollision,
        boolean onGround,
        boolean inWater,
        boolean inLava,
        int blockLight,
        int skyLight,
        double rayMaxDistanceBlocks,
        double rayForwardBlocks,
        double rayBackBlocks,
        double rayLeftBlocks,
        double rayRightBlocks,
        double rayUpBlocks,
        double rayDownBlocks) {
        this(
                gameTick,
                entityId,
                xBlocks,
                yBlocks,
                zBlocks,
                velocityXBlocksPerTick,
                velocityYBlocksPerTick,
                velocityZBlocksPerTick,
                yawDegrees,
                pitchDegrees,
                horizontalCollision,
                verticalCollision,
                onGround,
                inWater,
                inLava,
                blockLight,
                skyLight,
                rayMaxDistanceBlocks,
                rayForwardBlocks,
                rayBackBlocks,
                rayLeftBlocks,
                rayRightBlocks,
                rayUpBlocks,
                rayDownBlocks,
                null);
    }

    public PhysicalSensorSnapshot {
        Objects.requireNonNull(entityId, "entityId");
        if (rayHitKinds != null) {
            rayHitKinds = List.copyOf(rayHitKinds);
            if (rayHitKinds.size() != 6
                    || rayHitKinds.stream().anyMatch(kind -> kind < 0 || kind > 2)) {
                throw new IllegalArgumentException("ray hit kinds must be six integers in [0, 2]");
            }
        }
        if (rayHitKinds == null && rayActorKinds != null) {
            throw new IllegalArgumentException("ray actor kinds require measured hit kinds");
        }
        if (rayActorKinds != null) {
            rayActorKinds = List.copyOf(rayActorKinds);
            if (rayActorKinds.size() != 6
                    || rayActorKinds.stream().anyMatch(kind -> kind < 0 || kind > 3)) {
                throw new IllegalArgumentException("ray actor kinds must be six integers in [0, 3]");
            }
            for (int index = 0; index < 6; index++) {
                if ((rayHitKinds.get(index) == 2) != (rayActorKinds.get(index) != 0)) {
                    throw new IllegalArgumentException("ray hit and actor kinds disagree");
                }
            }
        }
        if (gameTick < 0) {
            throw new IllegalArgumentException("tick must be non-negative");
        }
        if (events != null) { events.validateAt(gameTick); }
        if (blockLight < 0 || blockLight > 15 || skyLight < 0 || skyLight > 15) {
            throw new IllegalArgumentException("light levels must be in [0, 15]");
        }
        double[] finiteValues = {
            xBlocks,
            yBlocks,
            zBlocks,
            velocityXBlocksPerTick,
            velocityYBlocksPerTick,
            velocityZBlocksPerTick,
            yawDegrees,
            pitchDegrees,
            rayMaxDistanceBlocks,
            rayForwardBlocks,
            rayBackBlocks,
            rayLeftBlocks,
            rayRightBlocks,
            rayUpBlocks,
            rayDownBlocks
        };
        for (double value : finiteValues) {
            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException("sensor values must be finite");
            }
        }
        if (rayMaxDistanceBlocks <= 0.0
                || rayMaxDistanceBlocks > 64.0
                || rayForwardBlocks < 0.0
                || rayBackBlocks < 0.0
                || rayLeftBlocks < 0.0
                || rayRightBlocks < 0.0
                || rayUpBlocks < 0.0
                || rayDownBlocks < 0.0) {
            throw new IllegalArgumentException("ray range must be in (0, 64] and distances non-negative");
        }
    }
}
