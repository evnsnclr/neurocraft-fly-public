package dev.neurocraft.fly.client;

import dev.neurocraft.fly.body.BodyProgramId;
import java.util.List;
import net.minecraft.client.model.EntityModel;
import net.minecraft.client.model.geom.ModelPart;
import net.minecraft.client.model.geom.PartPose;
import net.minecraft.client.model.geom.builders.CubeListBuilder;
import net.minecraft.client.model.geom.builders.LayerDefinition;
import net.minecraft.client.model.geom.builders.MeshDefinition;
import net.minecraft.client.model.geom.builders.PartDefinition;

/** Original low-poly fly with scripted display poses, not a claim of anatomical reconstruction. */
public final class FlyModel extends EntityModel<FlyRenderState> {
    /** Part names of the three legs on the body's left side, in front-to-back order. */
    public static final List<String> LEFT_LEG_PARTS =
            List.of("left_front_leg", "left_mid_leg", "left_hind_leg");

    /** Part names of the three legs on the body's right side, in front-to-back order. */
    public static final List<String> RIGHT_LEG_PARTS =
            List.of("right_front_leg", "right_mid_leg", "right_hind_leg");

    private final ModelPart abdomen;
    private final ModelPart leftWing;
    private final ModelPart rightWing;
    private final ModelPart leftAntenna;
    private final ModelPart rightAntenna;
    private final ModelPart proboscis;
    private final List<ModelPart> leftLegs;
    private final List<ModelPart> rightLegs;

    public FlyModel(ModelPart root) {
        super(root);
        abdomen = root.getChild("abdomen");
        leftWing = root.getChild("left_wing");
        rightWing = root.getChild("right_wing");
        leftAntenna = root.getChild("left_antenna");
        rightAntenna = root.getChild("right_antenna");
        proboscis = root.getChild("proboscis");
        leftLegs = LEFT_LEG_PARTS.stream().map(root::getChild).toList();
        rightLegs = RIGHT_LEG_PARTS.stream().map(root::getChild).toList();
    }

    public static LayerDefinition createBodyLayer() {
        MeshDefinition mesh = new MeshDefinition();
        PartDefinition root = mesh.getRoot();
        root.addOrReplaceChild(
                "thorax",
                CubeListBuilder.create().texOffs(0, 18).addBox(-2.5F, -2.0F, -2.0F, 5.0F, 4.0F, 5.0F),
                PartPose.ZERO);
        root.addOrReplaceChild(
                "abdomen",
                CubeListBuilder.create().texOffs(0, 0).addBox(-2.0F, -1.7F, 2.0F, 4.0F, 3.4F, 8.0F),
                PartPose.offsetAndRotation(0.0F, 0.0F, 0.0F, -0.10F, 0.0F, 0.0F));
        root.addOrReplaceChild(
                "head",
                CubeListBuilder.create().texOffs(24, 18).addBox(-2.2F, -2.0F, -5.4F, 4.4F, 4.0F, 3.6F),
                PartPose.ZERO);
        root.addOrReplaceChild(
                "left_eye",
                CubeListBuilder.create().texOffs(48, 0).addBox(1.7F, -1.65F, -5.1F, 0.8F, 3.3F, 2.8F),
                PartPose.ZERO);
        root.addOrReplaceChild(
                "right_eye",
                CubeListBuilder.create().texOffs(48, 0).addBox(-2.5F, -1.65F, -5.1F, 0.8F, 3.3F, 2.8F),
                PartPose.ZERO);
        root.addOrReplaceChild(
                "left_wing",
                CubeListBuilder.create().texOffs(32, 0).addBox(0.0F, -0.35F, -1.0F, 8.0F, 0.7F, 7.0F),
                PartPose.offsetAndRotation(1.6F, -1.2F, 0.0F, -0.08F, -0.24F, -0.25F));
        root.addOrReplaceChild(
                "right_wing",
                CubeListBuilder.create().texOffs(32, 0).mirror().addBox(-8.0F, -0.35F, -1.0F, 8.0F, 0.7F, 7.0F),
                PartPose.offsetAndRotation(-1.6F, -1.2F, 0.0F, -0.08F, 0.24F, 0.25F));

        addLeg(root, "left_front_leg", 1.8F, -1.0F, 32, 18);
        addLeg(root, "left_mid_leg", 2.0F, 1.0F, 36, 18);
        addLeg(root, "left_hind_leg", 1.7F, 3.0F, 40, 18);
        addLeg(root, "right_front_leg", -2.8F, -1.0F, 32, 18);
        addLeg(root, "right_mid_leg", -3.0F, 1.0F, 36, 18);
        addLeg(root, "right_hind_leg", -2.7F, 3.0F, 40, 18);

        root.addOrReplaceChild(
                "left_antenna",
                CubeListBuilder.create().texOffs(48, 18).addBox(-0.25F, -0.25F, -4.0F, 0.5F, 0.5F, 4.0F),
                PartPose.offsetAndRotation(1.0F, -1.2F, -4.6F, 0.25F, -0.18F, 0.0F));
        root.addOrReplaceChild(
                "right_antenna",
                CubeListBuilder.create().texOffs(48, 18).addBox(-0.25F, -0.25F, -4.0F, 0.5F, 0.5F, 4.0F),
                PartPose.offsetAndRotation(-1.0F, -1.2F, -4.6F, 0.25F, 0.18F, 0.0F));
        // Original simplified proboscis; reuses the owned brown leg palette at (32,18).
        // Its length is a declared presentation constant, never an anatomical measurement.
        PartDefinition mouth = root.addOrReplaceChild(
                "proboscis",
                CubeListBuilder.create().texOffs(32, 18).addBox(-0.35F, -0.35F,
                        -FlyAnimation.PROBOSCIS_LENGTH, 0.7F, 0.7F, FlyAnimation.PROBOSCIS_LENGTH),
                PartPose.offsetAndRotation(0.0F, 1.0F, -5.1F, FlyAnimation.PROBOSCIS_PITCH, 0.0F, 0.0F));
        mouth.addOrReplaceChild("proboscis_tip",
                CubeListBuilder.create().texOffs(36, 18).addBox(-0.65F, -0.35F, -0.6F, 1.3F, 0.7F, 0.8F),
                PartPose.offset(0.0F, 0.0F, -FlyAnimation.PROBOSCIS_LENGTH));
        return LayerDefinition.create(mesh, 64, 64);
    }

    /**
     * ADR 0016: the base pose uses the synchronized motor vector, carried here as the
     * already-smoothed presentation channels on {@code state}.
     * Scripted program overlays additionally consume the published phase/modulation and existing
     * ground-contact flag. These are presentation engineering, not additional neural readouts.
     *
     * <p>This method writes only {@link ModelPart} transforms and visibility. {@code super.setupAnim}
     * restores every part to its baked pose first, so each channel is set absolutely and no frame
     * can accumulate into the next.
     */
    @Override
    public void setupAnim(FlyRenderState state) {
        super.setupAnim(state);
        BodyProgramId program = state.bodyProgram.program();
        float phase = FlyAnimation.programPhase(state.bodyProgramPhase);
        float gesture = (float) state.bodyProgram.modulation().amplitude();
        boolean grounded = FlyAnimation.groundPose(program, state.onGround);
        // resetPose does not reset visibility. Set both properties on every frame, including exit.
        proboscis.visible = false;
        proboscis.zScale = 1.0F;

        // Wings: amplitude and rate rise with thrust effort, and the outside wing of a turn works
        // harder than the inside one. Positive visual yaw is a command toward the body's right
        // (ADR 0015), so the LEFT wing is the outside wing and gains stroke.
        float stroke = (float) Math.sin(state.wingBeatPhase);
        float amplitude = FlyAnimation.wingBeatAmplitude(state.thrustEffort);
        float differential = FlyAnimation.clampSigned(state.visualYawTorque);
        float leftAmplitude = amplitude + FlyAnimation.WING_DIFFERENTIAL_AMPLITUDE * differential;
        float rightAmplitude = amplitude - FlyAnimation.WING_DIFFERENTIAL_AMPLITUDE * differential;
        leftWing.zRot = -FlyAnimation.WING_REST_SPLAY - stroke * leftAmplitude;
        rightWing.zRot = FlyAnimation.WING_REST_SPLAY + stroke * rightAmplitude;
        leftWing.xRot = FlyAnimation.WING_REST_PITCH - FlyAnimation.WING_DIFFERENTIAL_PITCH * differential;
        rightWing.xRot = FlyAnimation.WING_REST_PITCH + FlyAnimation.WING_DIFFERENTIAL_PITCH * differential;
        if (grounded) {
            leftWing.zRot = -FlyAnimation.GROUND_WING_SPLAY;
            rightWing.zRot = FlyAnimation.GROUND_WING_SPLAY;
            leftWing.yRot = -FlyAnimation.GROUND_WING_FOLD_YAW;
            rightWing.yRot = FlyAnimation.GROUND_WING_FOLD_YAW;
            leftWing.xRot = rightWing.xRot = FlyAnimation.WING_REST_PITCH;
        }

        // Legs: a climb tucks them back and in, a descent extends them forward and out.
        float tuck = FlyAnimation.clampSigned(state.smoothedVerticalThrust);
        if (state.bodyProgram.program() == BodyProgramId.CRUISE && tuck >= 0.0F) {
            // ADR 0020 lane B, presentation only. Level cruise commands zero vertical thrust, so
            // the mapping above would draw a flying fly with parked legs. The overlay is a floor
            // on the tuck rather than a replacement, so a CRUISE climbing away from a floor still
            // draws the full climb pose the vertical channel asked for.
            //
            // It is applied only to a non-negative tuck. A descending CRUISE -- the clearance loop
            // pushing the body down off a ceiling -- keeps the ADR 0016 extended-leg descent pose,
            // which the unguarded floor would have overridden with a climb-shaped tuck.
            tuck = Math.max(tuck, FlyAnimation.CRUISE_LEG_TUCK);
        }
        if (program == BodyProgramId.LAND) {
            tuck = Math.min(tuck, FlyAnimation.LAND_LEG_EXTENSION);
        } else if ((program == BodyProgramId.FLEE || program == BodyProgramId.APPROACH)
                && tuck >= 0.0F) {
            tuck = Math.max(tuck, program == BodyProgramId.FLEE
                    ? FlyAnimation.FLEE_LEG_TUCK : FlyAnimation.APPROACH_LEG_TUCK);
        } else if (grounded) {
            tuck = 0.0F;
        }
        float legPitch = FlyAnimation.LEG_REST_PITCH + FlyAnimation.LEG_TUCK_SWING * tuck;
        float legSplay = FlyAnimation.LEG_REST_SPLAY - FlyAnimation.LEG_TUCK_SPREAD * tuck;
        poseLegs(leftLegs, legPitch, -legSplay);
        poseLegs(rightLegs, legPitch, legSplay);

        // Abdomen: rides up under forward thrust and drops when the fly brakes.
        abdomen.xRot = FlyAnimation.ABDOMEN_REST_PITCH
                + FlyAnimation.ABDOMEN_PITCH_SWING
                        * FlyAnimation.clampSigned(state.smoothedForwardThrust);

        // Antennae: an idle sweep only. Presentation with no motor input, unchanged.
        float antennaSweep = (float) Math.sin(state.ageInTicks * FlyAnimation.ANTENNA_IDLE_RATE)
                * FlyAnimation.ANTENNA_IDLE_SWEEP;
        leftAntenna.yRot = -FlyAnimation.ANTENNA_REST_YAW + antennaSweep;
        rightAntenna.yRot = FlyAnimation.ANTENNA_REST_YAW - antennaSweep;
        if (program == BodyProgramId.FLEE || program == BodyProgramId.APPROACH) {
            float spread = program == BodyProgramId.FLEE
                    ? FlyAnimation.FLEE_ANTENNA_SPREAD : FlyAnimation.APPROACH_ANTENNA_SPREAD;
            leftAntenna.yRot = -spread + antennaSweep;
            rightAntenna.yRot = spread - antennaSweep;
        }
        if (program == BodyProgramId.WALK && grounded) {
            float gait = gesture * (float) state.bodyProgram.modulation().speed();
            for (int index = 0; index < leftLegs.size(); index++) {
                walkLeg(leftLegs.get(index), phase, index, true, gait);
                walkLeg(rightLegs.get(index), phase, index, false, gait);
            }
        } else if (program == BodyProgramId.GROOM) {
            float reach = gesture * FlyAnimation.groomingReach(phase);
            float rub = gesture * FlyAnimation.groomingRub(phase, 1);
            float antennaRub = gesture * FlyAnimation.groomingRub(phase, 2);
            float antennaReach = gesture * FlyAnimation.GROOM_ANTENNA_REACH_EXTRA
                    * FlyAnimation.groomingAntennaReach(phase);
            leftLegs.getFirst().xRot -= FlyAnimation.GROOM_FORELEG_REACH * reach
                    + antennaReach + FlyAnimation.GROOM_RUB_SWING * (rub + antennaRub);
            rightLegs.getFirst().xRot -= FlyAnimation.GROOM_FORELEG_REACH * reach
                    + antennaReach - FlyAnimation.GROOM_RUB_SWING * (rub + antennaRub);
            leftLegs.getFirst().zRot += FlyAnimation.GROOM_FORELEG_INWARD_ROLL * reach;
            rightLegs.getFirst().zRot -= FlyAnimation.GROOM_FORELEG_INWARD_ROLL * reach;
            leftAntenna.yRot += FlyAnimation.GROOM_ANTENNA_SWEEP * antennaRub;
            rightAntenna.yRot -= FlyAnimation.GROOM_ANTENNA_SWEEP * antennaRub;
        } else if (program == BodyProgramId.FEED && grounded) {
            float extension = gesture * FlyAnimation.feedingExtension(phase);
            proboscis.visible = extension > 0.0F;
            proboscis.zScale = extension > 0.0F ? extension : 1.0F;
        }
    }

    private static void walkLeg(ModelPart leg, float phase, int index, boolean left, float amplitude) {
        float swing = FlyAnimation.walkSwing(phase, index, left);
        leg.xRot += FlyAnimation.WALK_LEG_SWING * amplitude * swing;
        float rollSwing = FlyAnimation.WALK_LEG_ROLL_SWING * amplitude * Math.max(0.0F, swing);
        leg.zRot += left ? rollSwing : -rollSwing;
    }

    private static void poseLegs(List<ModelPart> legs, float pitch, float roll) {
        for (ModelPart leg : legs) {
            leg.xRot = pitch;
            leg.zRot = roll;
        }
    }

    private static void addLeg(
            PartDefinition root, String name, float x, float z, int textureX, int textureY) {
        float roll = x > 0.0F ? -0.65F : 0.65F;
        root.addOrReplaceChild(
                name,
                CubeListBuilder.create().texOffs(textureX, textureY).addBox(-0.3F, 0.0F, -0.3F, 0.6F, 5.0F, 0.6F),
                PartPose.offsetAndRotation(x, 1.3F, z, -0.18F, 0.0F, roll));
    }
}
