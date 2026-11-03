package dev.neurocraft.fly.client;

import dev.neurocraft.fly.control.PrimitiveAction;

/**
 * Client-only presentation memory for one fly: the low-pass-filtered motor channels and the
 * integrated wing-beat phase that {@link FlyModel} draws from (ADR 0016).
 *
 * <p><strong>Render-only.</strong> In production this object is owned by {@link FlyRenderer}, lives
 * entirely on the client, and holds no reference to an entity or a level. It is written only from
 * {@code extractRenderState} and read only by the model and the pose. Nothing it stores is sent
 * anywhere, recorded in a trace, folded into an identity, or visible to the server.
 *
 * <p>It is {@code public} so that a client GameTest can drive the real filter with a synthetic
 * command sequence rather than reimplementing it — {@code escapeJumpProfileSurvivesTheClientSmoother}
 * needs the actual τ = {@value FlyAnimation#SMOOTHING_TAU_TICKS}-tick response, not a copy of it.
 * The type stays in the client source set, so it is not on the server classpath at all and the
 * render-only guarantee above is structural rather than a matter of visibility.
 *
 * <p>It is mutable by necessity: smoothing and phase integration are stateful across frames, and
 * Minecraft reuses a single render state per renderer, so there is nowhere else for a per-entity
 * carry to live. The mutation is confined to this class.
 */
public final class FlyPresentationSmoother {
    private boolean started;
    private float lastAgeInTicks;
    private float forwardThrust;
    private float verticalThrust;
    private float yawTorque;
    private float wingBeatPhase;

    /**
     * Advances the filter to {@code ageInTicks} against the currently synchronized command.
     *
     * @param command the entity's published presentation action; read, never written
     * @param ageInTicks the entity's own tick count plus the frame's partial tick
     */
    public void advance(PrimitiveAction command, float ageInTicks) {
        float target = (float) command.forwardThrust();
        float targetVertical = (float) command.verticalThrust();
        float targetYaw = (float) command.yawTorque();

        if (!started || !Float.isFinite(ageInTicks) || ageInTicks < lastAgeInTicks) {
            // First frame for this entity, or the clock went backwards (respawn, world reload).
            // Snap rather than sweep in from a stale carry.
            started = true;
            lastAgeInTicks = Float.isFinite(ageInTicks) ? ageInTicks : 0.0F;
            forwardThrust = target;
            verticalThrust = targetVertical;
            yawTorque = targetYaw;
            return;
        }

        float elapsedTicks = FlyAnimation.boundElapsedTicks(ageInTicks - lastAgeInTicks);
        lastAgeInTicks = ageInTicks;
        forwardThrust = FlyAnimation.smoothTowards(forwardThrust, target, elapsedTicks);
        verticalThrust = FlyAnimation.smoothTowards(verticalThrust, targetVertical, elapsedTicks);
        yawTorque = FlyAnimation.smoothTowards(yawTorque, targetYaw, elapsedTicks);
        wingBeatPhase = FlyAnimation.advanceWingBeatPhase(
                wingBeatPhase, elapsedTicks, FlyAnimation.wingBeatRate(thrustEffort()));
    }

    /** Copies the derived presentation values into a render state. Writes nothing else. */
    public void writeTo(FlyRenderState state) {
        state.smoothedForwardThrust = forwardThrust;
        state.smoothedVerticalThrust = verticalThrust;
        state.thrustEffort = thrustEffort();
        state.visualYawTorque = FlyAnimation.visualYawTorque(yawTorque);
        state.bankRotationDegrees = FlyAnimation.bankRotationDegrees(state.visualYawTorque);
        state.wingBeatPhase = wingBeatPhase;
    }

    private float thrustEffort() {
        return FlyAnimation.thrustEffort(forwardThrust, verticalThrust);
    }
}
