package dev.neurocraft.fly.client;

import dev.neurocraft.fly.body.BodyProgramChannel;
import dev.neurocraft.fly.control.ControllerFault;
import dev.neurocraft.fly.control.PrimitiveAction;
import net.minecraft.client.renderer.entity.state.EntityRenderState;

/** Client-only immutable-by-convention presentation snapshot; never feeds authoritative control. */
public final class FlyRenderState extends EntityRenderState {
    public float yawDegrees;
    public float pitchDegrees;
    public ControllerFault controllerState = ControllerFault.STARTING;
    public PrimitiveAction appliedAction = PrimitiveAction.NEUTRAL;

    // ---- ADR 0016 presentation channels -------------------------------------------------
    // Derived from appliedAction by FlyPresentationSmoother during extractRenderState. They are
    // drawn and nothing else: no value below is transmitted, traced, hashed, or read by the
    // server, and none of them alters the command the actuator applies.

    /** Low-pass-filtered forward thrust; drives abdomen pitch. */
    public float smoothedForwardThrust;

    /** Low-pass-filtered vertical thrust; drives leg tuck and extension. */
    public float smoothedVerticalThrust;

    /** Smoothed forward/vertical thrust magnitude in [0, 1]; drives wing beat rate and amplitude. */
    public float thrustEffort;

    /**
     * Smoothed yaw torque rescaled by the declared {@link FlyAnimation#YAW_TORQUE_VISUAL_REFERENCE}
     * and clamped to [-1, +1]. Positive is toward the body's right. A display scaling only.
     */
    public float visualYawTorque;

    /** Render-only body bank, in degrees, for {@code Axis.ZP}. The entity has no roll axis. */
    public float bankRotationDegrees;

    /** Integrated wing-beat phase in radians, wrapped into [0, 2*pi). */
    public float wingBeatPhase;

    // ---- ADR 0020 body-program channels -------------------------------------------------
    // Copied verbatim from the entity's two synchronized presentation fields. They name what the
    // body is doing and who chose it, and they exist so a frame grab of a scripted fly says so on
    // its face (the D-074 lesson). Nothing here influences an action.

    /** The published selection source, program id, and quantized modulation. */
    public BodyProgramChannel bodyProgram = BodyProgramChannel.DEFAULT;

    /** The published normalized program phase in [0, 1). */
    public float bodyProgramPhase;

    /** Existing client entity ground-contact flag, read only; gates folded-wing poses. */
    public boolean onGround;
}
