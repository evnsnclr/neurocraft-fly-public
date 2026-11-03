package dev.neurocraft.fly.body;

import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import java.util.Objects;
import java.util.Optional;

/**
 * The world inputs a body program may read on one tick (ADR 0020 section 5).
 *
 * <p><strong>This is a body-program input. It is never merged into
 * {@link PhysicalSensorSnapshot}, and it is structurally unable to reach the bridge.</strong>
 * The sensory vector the Python encoder consumes is hashed into {@code observation_schema_hash};
 * adding a field to it rotates that identity and every downstream binding. More importantly,
 * handing the connectome a bearing to a target would be navigation policy hidden inside the
 * encoder, which the Faithful Organism Mode purity rule forbids: a ray proximity is a sensor
 * reading, a bearing to the nearest player is a goal.
 *
 * <p>In lane A of stage 3 the three optional world inputs are always empty; the surface, target,
 * and food sources land with the programs that need them (LAND, APPROACH/FLEE, FEED).
 */
public record BodyProgramContext(
        PhysicalSensorSnapshot snapshot,
        Optional<SurfaceBelow> surfaceBelow,
        Optional<BodyTargetVector> target,
        Optional<ReachableFood> food) {
    public BodyProgramContext {
        Objects.requireNonNull(snapshot, "snapshot");
        Objects.requireNonNull(surfaceBelow, "surfaceBelow");
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(food, "food");
    }

    /** The context available in lane A: the authoritative sensor sample and nothing else. */
    public static BodyProgramContext sensorsOnly(PhysicalSensorSnapshot snapshot) {
        return new BodyProgramContext(
                snapshot, Optional.empty(), Optional.empty(), Optional.empty());
    }

    /** Distance to, and landability of, the surface under the body. */
    public record SurfaceBelow(double distanceBlocks, boolean sturdy) {
        public SurfaceBelow {
            requireFinite(distanceBlocks, "distanceBlocks");
        }
    }

    /** A body-frame bearing to a target entity. Never reaches the sensory vector. */
    public record BodyTargetVector(
            double bearingDegrees, double elevationDegrees, double distanceBlocks) {
        public BodyTargetVector {
            requireFinite(bearingDegrees, "bearingDegrees");
            requireFinite(elevationDegrees, "elevationDegrees");
            requireFinite(distanceBlocks, "distanceBlocks");
        }
    }

    /** A body-frame bearing to the nearest item entity from the declared edible set. */
    public record ReachableFood(double bearingDegrees, double distanceBlocks, String itemId) {
        public ReachableFood {
            requireFinite(bearingDegrees, "bearingDegrees");
            requireFinite(distanceBlocks, "distanceBlocks");
            Objects.requireNonNull(itemId, "itemId");
        }
    }

    private static void requireFinite(double value, String name) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(name + " must be finite");
        }
    }
}
