package dev.neurocraft.fly.gametest;

import com.mojang.blaze3d.platform.Lighting;
import com.mojang.blaze3d.platform.NativeImage;
import com.mojang.math.Axis;
import dev.neurocraft.fly.body.BodyProgramChannel;
import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ProgramParameters;
import dev.neurocraft.fly.body.SelectionSource;
import dev.neurocraft.fly.client.FlyAnimation;
import dev.neurocraft.fly.client.FlyModel;
import dev.neurocraft.fly.client.FlyRenderState;
import dev.neurocraft.fly.client.FlyRenderer;
import dev.neurocraft.fly.control.PrimitiveAction;
import dev.neurocraft.fly.entity.FlyEntity;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import net.fabricmc.fabric.api.client.gametest.v1.context.ClientGameTestContext;
import net.minecraft.client.gui.GuiGraphics;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.client.model.geom.ModelPart;
import net.minecraft.client.renderer.LightTexture;
import net.minecraft.network.chat.Component;
import net.minecraft.world.entity.Entity;

/** Synthetic poses through the actual client renderer, not observed body behavior. */
final class BodyProgramPoseClientChecks {
    private static final float EPSILON = 1.0E-5F;

    private BodyProgramPoseClientChecks() {}

    static void assertPoses(FlyModel model) {
        FlyRenderState land = pose(BodyProgramId.LAND, 0.5F, false, 1.0);
        model.setupAnim(land);
        expect(model.root().getChild("left_front_leg").xRot < FlyAnimation.LEG_REST_PITCH - 0.3F,
                "LAND did not visibly extend its legs before contact");
        float airborneWing = model.root().getChild("left_wing").zRot;
        land.wingBeatPhase += (float) Math.PI;
        model.setupAnim(land);
        expect(Math.abs(model.root().getChild("left_wing").zRot - airborneWing) > 0.1F,
                "airborne LAND lost its flight wing beat");
        land.onGround = true;
        model.setupAnim(land);
        float groundedWing = model.root().getChild("left_wing").zRot;
        land.wingBeatPhase += (float) Math.PI;
        model.setupAnim(land);
        expect(model.root().getChild("left_wing").zRot == groundedWing,
                "grounded LAND still beats wings instead of the declared rest pose");
        expect(Math.abs(model.root().getChild("left_wing").yRot) > 0.8F,
                "grounded wings did not fold toward the abdomen");

        FlyRenderState walk = pose(BodyProgramId.WALK, 0.25F, true, 1.0);
        model.setupAnim(walk);
        float front = part(model, "left_front_leg").xRot;
        expect(front > part(model, "left_mid_leg").xRot + 0.5F
                        && front == part(model, "left_hind_leg").xRot
                        && front == part(model, "right_mid_leg").xRot,
                "WALK did not alternate the declared tripods");
        walk.bodyProgramPhase = 0.75F;
        model.setupAnim(walk);
        expect(part(model, "left_front_leg").xRot < part(model, "left_mid_leg").xRot - 0.5F,
                "half a cycle did not swap the walking tripods");
        model.setupAnim(pose(BodyProgramId.WALK, 0.25F, true, 0.5));
        expect(close(part(model, "left_front_leg").xRot - FlyAnimation.LEG_REST_PITCH,
                        0.5F * (front - FlyAnimation.LEG_REST_PITCH)),
                "walking gesture does not attenuate with amplitude");
        model.setupAnim(pose(BodyProgramId.WALK, 0.25F, true, 0.0));
        expect(part(model, "left_front_leg").xRot == FlyAnimation.LEG_REST_PITCH,
                "zero amplitude retained a walking gesture");

        FlyRenderState groom = pose(BodyProgramId.GROOM, 0.4375F, true, 1.0);
        model.setupAnim(groom);
        float foreleg = part(model, "left_front_leg").xRot;
        float antenna = part(model, "left_antenna").yRot;
        expect(foreleg < part(model, "left_mid_leg").xRot - 0.7F,
                "foreleg grooming did not lift the front legs independently");
        groom.bodyProgramPhase = 0.59375F;
        model.setupAnim(groom);
        expect(Math.abs(part(model, "left_antenna").yRot - antenna) > 0.1F,
                "antennal quarter did not differ from the foreleg quarter");
        expect(part(model, "left_front_leg").xRot < foreleg - 0.4F,
                "antennal grooming did not raise the forelegs above the foreleg-rubbing pose");
        model.setupAnim(pose(BodyProgramId.GROOM, 0.59375F, true, 0.0));
        expect(part(model, "left_front_leg").xRot == FlyAnimation.LEG_REST_PITCH,
                "zero amplitude retained a grooming gesture");
        groom.onGround = false;
        model.setupAnim(groom);
        expect(Math.abs(part(model, "left_wing").yRot) < 0.3F,
                "airborne grooming folded flight wings");

        model.setupAnim(pose(BodyProgramId.FEED, 0.125F, true, 1.0));
        expect(!part(model, "proboscis").visible, "proboscis appeared before the extend phase");
        model.setupAnim(pose(BodyProgramId.FEED, 0.375F, true, 1.0));
        expect(part(model, "proboscis").visible && close(part(model, "proboscis").zScale, 0.5F),
                "FEED did not visibly extend halfway");
        model.setupAnim(pose(BodyProgramId.FEED, 0.625F, true, 1.0));
        expect(close(part(model, "proboscis").zScale, 1.0F), "FEED hold did not fully extend");
        model.setupAnim(pose(BodyProgramId.FEED, 0.875F, true, 1.0));
        expect(close(part(model, "proboscis").zScale, 0.5F), "FEED did not retract halfway");
        model.setupAnim(pose(BodyProgramId.FEED, 0.625F, true, 0.5));
        expect(close(part(model, "proboscis").zScale, 0.5F), "FEED amplitude did not attenuate extent");
        for (FlyRenderState hidden : List.of(pose(BodyProgramId.FEED, 0.625F, true, 0.0),
                pose(BodyProgramId.FEED, 0.625F, false, 1.0))) {
            model.setupAnim(hidden);
            expect(!part(model, "proboscis").visible, "zero gesture or airborne FEED drew a proboscis");
        }

        // Every scripted pose must leave the next frame identical to a fresh model frame.
        FlyRenderState neutral = pose(BodyProgramId.HOVER, 0.0F, false, 1.0);
        FlyModel fresh = new FlyModel(FlyModel.createBodyLayer().bakeRoot());
        fresh.setupAnim(neutral);
        List<Transform> neutralPose = transforms(fresh);
        HashSet<List<Transform>> distinct = new HashSet<>();
        for (BodyProgramId id : BodyProgramId.values()) {
            FlyRenderState state = pose(id, 0.59375F, FlyAnimation.groundPose(id, true), 1.0);
            if (id == BodyProgramId.BANK) state.visualYawTorque = 1.0F;
            if (id == BodyProgramId.ESCAPE_JUMP) state.smoothedVerticalThrust = 1.0F;
            BodyProgramChannel label = state.bodyProgram;
            model.setupAnim(state);
            expect(state.bodyProgram.equals(label) && state.appliedAction.equals(PrimitiveAction.NEUTRAL)
                            && state.bodyProgramPhase == 0.59375F,
                    "model mutated the authority label, applied action or published phase");
            distinct.add(transforms(model));
            model.setupAnim(neutral);
            expect(transforms(model).equals(neutralPose), "pose leaked across program exit: " + id);
        }
        expect(distinct.size() == BodyProgramId.values().length,
                "representative scripted poses were indistinguishable");
    }

    static void observe(ClientGameTestContext context) {
        List<Example> examples = List.of(
                new Example("land-air", BodyProgramId.LAND, 0.5F, false),
                new Example("land-ground", BodyProgramId.LAND, 0.5F, true),
                new Example("walk-tripod-a", BodyProgramId.WALK, 0.25F, true),
                new Example("walk-tripod-b", BodyProgramId.WALK, 0.75F, true),
                new Example("groom-forelegs", BodyProgramId.GROOM, 0.4375F, true),
                new Example("groom-antennae", BodyProgramId.GROOM, 0.59375F, true),
                new Example("feed-retracted", BodyProgramId.FEED, 0.125F, true),
                new Example("feed-extended", BodyProgramId.FEED, 0.625F, true));
        List<Path> images = new ArrayList<>();
        try {
            for (Example example : examples) {
                context.runOnClient(client -> {
                    FlyEntity fly = null;
                    if (client.level != null) {
                        for (Entity entity : client.level.entitiesForRendering()) {
                            if (entity instanceof FlyEntity candidate) fly = candidate;
                        }
                    }
                    if (fly == null || !(client.getEntityRenderDispatcher().getRenderer(fly)
                            instanceof FlyRenderer renderer)) {
                        throw new AssertionError("actual fly renderer unavailable for pose observation");
                    }
                    client.setScreen(new PoseScreen(renderer, example));
                });
                context.waitTicks(2);
                Path image = context.takeScreenshot("scripted-pose-" + example.name());
                images.add(image);
                System.out.println("Scripted renderer pose screenshot: " + image);
            }
            for (int index = 0; index < images.size(); index += 2) {
                expect(centralDifference(images.get(index), images.get(index + 1)) > 40,
                        "actual renderer did not distinguish poses " + examples.get(index).name());
            }
        } finally {
            context.runOnClient(client -> client.setScreen(null));
        }
    }

    private static int centralDifference(Path first, Path second) {
        try (var aStream = Files.newInputStream(first); var bStream = Files.newInputStream(second);
                NativeImage a = NativeImage.read(aStream); NativeImage b = NativeImage.read(bStream)) {
            expect(a.getWidth() == b.getWidth() && a.getHeight() == b.getHeight(), "image dimensions changed");
            int changed = 0;
            // Exclude captions, so two text labels cannot satisfy the renderer assertion.
            for (int y = a.getHeight() / 4; y < a.getHeight() * 4 / 5; y++) {
                for (int x = a.getWidth() / 8; x < a.getWidth() * 7 / 8; x++) {
                    if (a.getPixel(x, y) != b.getPixel(x, y)) changed++;
                }
            }
            return changed;
        } catch (IOException exception) {
            throw new AssertionError("could not inspect scripted pose screenshots", exception);
        }
    }

    private static FlyRenderState pose(BodyProgramId id, float phase, boolean ground, double amplitude) {
        FlyRenderState state = new FlyRenderState();
        state.bodyProgram = new BodyProgramChannel(SelectionSource.OPERATOR, id,
                new Modulation(1.0, 1.0, amplitude, 0.0), ProgramParameters.NEUTRAL);
        state.bodyProgramPhase = phase;
        state.onGround = ground;
        state.ageInTicks = 100.0F;
        state.wingBeatPhase = (float) (Math.PI / 2.0);
        state.boundingBoxHeight = 0.22F;
        state.boundingBoxWidth = 0.38F;
        state.yawDegrees = 220.0F;
        state.pitchDegrees = -12.0F;
        return state;
    }

    private static List<Transform> transforms(FlyModel model) {
        List<String> names = new ArrayList<>(List.of("abdomen", "left_wing", "right_wing",
                "left_antenna", "right_antenna", "proboscis"));
        names.addAll(FlyModel.LEFT_LEG_PARTS);
        names.addAll(FlyModel.RIGHT_LEG_PARTS);
        return names.stream().map(name -> Transform.of(part(model, name))).toList();
    }

    private static ModelPart part(FlyModel model, String name) { return model.root().getChild(name); }
    private static boolean close(float a, float b) { return Math.abs(a - b) < EPSILON; }
    private static void expect(boolean value, String message) {
        if (!value) throw new AssertionError(message);
    }

    private record Transform(float x, float y, float z, float pitch, float yaw, float roll,
            float xScale, float yScale, float zScale, boolean visible) {
        static Transform of(ModelPart part) {
            return new Transform(part.x, part.y, part.z, part.xRot, part.yRot, part.zRot,
                    part.xScale, part.yScale, part.zScale, part.visible);
        }
    }

    private record Example(String name, BodyProgramId program, float phase, boolean grounded) {}

    private static final class PoseScreen extends Screen {
        private final FlyRenderer renderer;
        private final Example example;
        private final FlyRenderState state;

        PoseScreen(FlyRenderer renderer, Example example) {
            super(Component.literal("Scripted renderer pose"));
            this.renderer = renderer;
            this.example = example;
            state = pose(example.program(), example.phase(), example.grounded(), 1.0);
        }

        @Override
        public void render(GuiGraphics graphics, int mouseX, int mouseY, float delta) {
            graphics.fill(0, 0, width, height, 0xffece6da);
            graphics.drawCenteredString(font, "SCRIPTED DISPLAY POSE · NOT ANATOMY", width / 2, 12, 0xff302b25);
            graphics.drawCenteredString(font, String.format(Locale.ROOT, "%s · OPERATOR · phase %.5f",
                    example.program(), example.phase()), width / 2, 28, 0xff302b25);
            graphics.drawCenteredString(font, "Magnified renderer check · synthetic state · ground="
                    + example.grounded(), width / 2, height - 22, 0xff302b25);
            graphics.flush();
            Lighting.setupForEntityInInventory();
            graphics.pose().pushPose();
            graphics.pose().translate(width / 2.0, height * 0.65, 50.0);
            float scale = Math.min(width, height) * 1.8F;
            graphics.pose().scale(scale, scale, -scale);
            graphics.pose().mulPose(Axis.ZP.rotationDegrees(180.0F));
            graphics.drawSpecial(buffers -> renderer.render(state, graphics.pose(), buffers, LightTexture.FULL_BRIGHT));
            graphics.flush();
            graphics.pose().popPose();
            Lighting.setupFor3DItems();
        }

        @Override
        public boolean isPauseScreen() { return false; }
    }
}
