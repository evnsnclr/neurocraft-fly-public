package dev.neurocraft.fly.client;

import com.mojang.blaze3d.vertex.PoseStack;
import com.mojang.blaze3d.vertex.VertexConsumer;
import com.mojang.math.Axis;
import dev.neurocraft.fly.NeuroCraftFly;
import dev.neurocraft.fly.entity.FlyEntity;
import java.util.HashMap;
import java.util.Map;
import net.minecraft.client.renderer.MultiBufferSource;
import net.minecraft.client.renderer.entity.EntityRenderer;
import net.minecraft.client.renderer.entity.EntityRendererProvider;
import net.minecraft.client.renderer.texture.OverlayTexture;
import net.minecraft.resources.ResourceLocation;

/**
 * Client-only original renderer; it consumes synchronized presentation fields and mutates no
 * entity state.
 *
 * <p>ADR 0016 added animation driven by the fly's motor vector, and the invariant above is what
 * bounds it. Every call this class makes on {@link FlyEntity} is a getter
 * ({@code getYRot}, {@code getXRot}, {@code controllerState}, {@code presentationAction},
 * {@code bodyProgramChannel}, {@code bodyProgramPhase}, {@code onGround}, {@code tickCount},
 * {@code getId}); the per-entity {@link FlyPresentationSmoother} it keeps is
 * client-side presentation memory that no other subsystem can read. Nothing here writes entity
 * state, feeds the controller, enters a trace, or participates in any connectome, adapter,
 * decoder, encoder, calibration, arena, pilot or embodiment identity. (The
 * neurocraft.gametest_code_bundle.v1 code_hash hashes all of mod/src and therefore does
 * include these files; see ADR 0016.)
 */
public final class FlyRenderer extends EntityRenderer<FlyEntity, FlyRenderState> {
    private static final ResourceLocation TEXTURE = ResourceLocation.fromNamespaceAndPath(
            NeuroCraftFly.MOD_ID, "textures/entity/fly.png");

    /**
     * Bound on the presentation-smoother cache. The fly is a singleton in every declared scenario,
     * so this exists only so a client-side cache cannot grow without limit; on overflow the cache
     * is dropped whole, which costs at most one smoothing restart and nothing else.
     */
    private static final int SMOOTHER_CACHE_LIMIT = 64;

    private final FlyModel model;
    private final Map<Integer, FlyPresentationSmoother> smoothers = new HashMap<>();

    public FlyRenderer(EntityRendererProvider.Context context) {
        super(context);
        model = new FlyModel(context.bakeLayer(NeuroCraftFlyClient.FLY_MODEL_LAYER));
        shadowRadius = 0.12F;
        shadowStrength = 0.35F;
    }

    @Override
    public FlyRenderState createRenderState() {
        return new FlyRenderState();
    }

    @Override
    public void extractRenderState(FlyEntity entity, FlyRenderState state, float partialTick) {
        super.extractRenderState(entity, state, partialTick);
        state.yawDegrees = entity.getYRot(partialTick);
        state.pitchDegrees = entity.getXRot(partialTick);
        state.controllerState = entity.controllerState();
        state.appliedAction = entity.presentationAction();
        state.bodyProgram = entity.bodyProgramChannel();
        state.bodyProgramPhase = entity.bodyProgramPhase();
        state.onGround = entity.onGround();
        // Read-only from here down. The smoother is advanced against the entity's own interpolated
        // tick clock so the animation is frame-rate independent, and it writes only into `state`.
        if (smoothers.size() >= SMOOTHER_CACHE_LIMIT && !smoothers.containsKey(entity.getId())) {
            smoothers.clear();
        }
        FlyPresentationSmoother smoother =
                smoothers.computeIfAbsent(entity.getId(), id -> new FlyPresentationSmoother());
        smoother.advance(state.appliedAction, entity.tickCount + partialTick);
        smoother.writeTo(state);
    }

    @Override
    public void render(
            FlyRenderState state,
            PoseStack poses,
            MultiBufferSource buffers,
            int packedLight) {
        poses.pushPose();
        poses.translate(0.0, state.boundingBoxHeight * 0.5F, 0.0);
        applyBodyPose(poses, state.yawDegrees, state.pitchDegrees, state.bankRotationDegrees);
        model.setupAnim(state);
        VertexConsumer vertices = buffers.getBuffer(model.renderType(TEXTURE));
        model.renderToBuffer(poses, vertices, packedLight, OverlayTexture.NO_OVERLAY);
        poses.popPose();
        super.render(state, poses, buffers, packedLight);
    }

    /**
     * Orients and scales the pose for the fly body. Yaw and pitch are the authoritative entity
     * rotations; the bank is a render-only rotation with no server-side counterpart at all
     * (ADR 0016).
     */
    public static void applyBodyPose(
            PoseStack poses, float yawDegrees, float pitchDegrees, float bankRotationDegrees) {
        poses.mulPose(Axis.YP.rotationDegrees(180.0F - yawDegrees));
        poses.mulPose(Axis.XP.rotationDegrees(pitchDegrees));
        poses.mulPose(Axis.ZP.rotationDegrees(bankRotationDegrees));
        // ModelPart cube coordinates are converted from sixteenths internally; this scale keeps
        // the body near its 0.38-block authoritative hitbox while leaving the wings visible.
        poses.scale(-0.42F, -0.42F, 0.42F);
    }
}
