package dev.neurocraft.fly.gametest;

import com.mojang.blaze3d.vertex.PoseStack;
import dev.neurocraft.fly.NeuroCraftFly;
import dev.neurocraft.fly.body.BodyProgramChannel;
import dev.neurocraft.fly.body.BodyProgramContext;
import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.BodyProgramState;
import dev.neurocraft.fly.body.BodyProgramStep;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ProgramParameters;
import dev.neurocraft.fly.body.SelectionSource;
import dev.neurocraft.fly.body.programs.EscapeJumpProgram;
import dev.neurocraft.fly.client.FlyAnimation;
import dev.neurocraft.fly.client.FlyModel;
import dev.neurocraft.fly.client.FlyPresentationSmoother;
import dev.neurocraft.fly.client.FlyRenderState;
import dev.neurocraft.fly.client.FlyRenderer;
import dev.neurocraft.fly.control.ControllerFault;
import dev.neurocraft.fly.control.PrimitiveAction;
import dev.neurocraft.fly.entity.FlyEntity;
import dev.neurocraft.fly.entity.FlySensors;
import java.util.Locale;
import net.fabricmc.fabric.api.client.gametest.v1.FabricClientGameTest;
import net.fabricmc.fabric.api.client.gametest.v1.context.ClientGameTestContext;
import net.minecraft.client.model.geom.ModelPart;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.client.renderer.entity.EntityRenderer;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EntitySpawnReason;
import net.minecraft.world.phys.Vec3;
import org.joml.Vector3f;

/**
 * ADR 0016 presentation pins: the fly's drawn body must be a function of its synchronized motor
 * vector, not of {@code ageInTicks} alone, and the render path must remain incapable of touching
 * entity state.
 *
 * <p><strong>Discrimination.</strong> Every assertion in {@link #assertWingBeatTracksThrust},
 * {@link #assertWingDifferentialTracksYawTorque} and {@link #assertLegsAndAbdomenTrackTheirAxes}
 * holds {@code ageInTicks} and the wing-beat phase fixed and varies only the motor channels. The
 * pre-ADR-0016 {@code setupAnim} was a function of {@code ageInTicks} alone and never touched a
 * leg or the abdomen at all, so it produced identical output for both members of every pair here
 * and fails each of these assertions by construction.
 *
 * <p>This is an automated renderer-state test, not a manual gameplay observation, and it makes no
 * claim about what the fly does — only about how what it does is drawn.
 */
public final class FlyAnimationClientGameTest implements FabricClientGameTest {
    private static final float TOLERANCE = 1.0E-5F;

    /** Half the wing span in model units; the tip of the left wing sits at {@code +x}. */
    private static final float WING_TIP_MODEL_X = 9.6F;

    /**
     * Ticks of settled neutral command fed to the smoother before the jump starts.
     *
     * <p>Production never reaches ESCAPE_JUMP with a cold filter: HOVER has been running. A cold
     * {@code advance} <em>snaps</em> to its target, which would draw the crouch at 100% of its
     * commanded depth and flatter every measurement below. Warming to a settled zero is the
     * honest starting condition and is what a hovering body's smoothed vertical averages to.
     */
    private static final int SMOOTHER_WARMUP_TICKS = 20;

    /**
     * Fraction of the commanded crouch the legs must actually reach, at zero urgency.
     * Measured on the declared 6/8 profile: <strong>0.8646647</strong>.
     */
    private static final float SMOOTHER_CROUCH_FRACTION = 0.75F;

    /**
     * Fraction of the commanded launch the legs must actually reach, at zero urgency.
     * Measured on the declared 6/8 profile: <strong>0.9124926</strong>.
     */
    private static final float SMOOTHER_LAUNCH_FRACTION = 0.85F;

    /**
     * The same two floors at full urgency, where the phases are at their declared 4/5 minima.
     *
     * <p>They are lower, and saying so is the point: full urgency buys a faster reflex and pays for
     * it in legibility. The floors are where that trade stops — a phase shorter than the declared
     * minimum draws so little that the crouch and the launch stop reading as separate events.
     * Measured on the 4/5 floor profile: crouch <strong>0.7364029</strong>, launch
     * <strong>0.7693979</strong>.
     */
    private static final float SMOOTHER_CROUCH_FRACTION_AT_FULL_URGENCY = 0.70F;

    private static final float SMOOTHER_LAUNCH_FRACTION_AT_FULL_URGENCY = 0.75F;

    /**
     * Fraction of one ideal filter step the crouch-to-launch boundary must draw in a single tick.
     *
     * <p><strong>This is a corroborating check, not the load-bearing one</strong>, and the review
     * was right to say so. One filter step from a drawn crouch is
     * {@code alpha * (launch - drawnCrouch)} with {@code alpha = 1 - e^(-1/3)}, so the ratio it
     * asserts is a deterministic function of the crouch fraction alone —
     * {@code (1 + 0.3 * cf) / 1.3}, which clears 0.9 for any {@code cf >= 0.567}. Both per-phase
     * floors below are already stricter than that, so this can never be the first to fail. It is
     * kept because it names the <em>snap</em> directly: a reader should be able to see that the
     * legs kick at the boundary rather than drift, without deriving it from the two fractions.
     *
     * <p>Measured against an ideal step of 0.3685093: <strong>0.9687690</strong> of it at zero
     * urgency and <strong>0.9391701</strong> at full.
     */
    private static final float SMOOTHER_BOUNDARY_JUMP_FRACTION = 0.9F;

    @Override
    public void runTest(ClientGameTestContext context) {
        context.runOnClient(client -> {
            FlyModel model = new FlyModel(FlyModel.createBodyLayer().bakeRoot());
            assertWingBeatTracksThrust(model);
            assertWingBeatStaysAliveAtZeroCommand(model);
            assertWingDifferentialTracksYawTorque(model);
            assertLegsAndAbdomenTrackTheirAxes(model);
            assertCruiseTucksTheLegsWhileAirborne(model);
            assertEscapeJumpProfileSurvivesTheClientSmoother(model);
            assertBankIsZeroSignedAndClamped();
            assertBankLowersTheInsideOfTheTurn();
            BodyProgramPoseClientChecks.assertPoses(model);
        });
        assertRenderExtractionLeavesEntityStateUntouched(context);
    }

    // ---- model animation ----------------------------------------------------------------

    private static void assertWingBeatTracksThrust(FlyModel model) {
        // Peak of the down stroke, identical clock and phase in both cases.
        float hovering = wingDeflection(model, animated(0.0F, 0.0F, 0.0F, 0.0F), true);
        float working = wingDeflection(model, animated(1.0F, 0.0F, 0.0F, 0.0F), true);
        expect(
                working > hovering + TOLERANCE,
                "wing stroke must widen with thrust effort at a fixed phase, but hovering="
                        + hovering + " working=" + working);

        float expectedHovering = FlyAnimation.wingBeatAmplitude(0.0F);
        float expectedWorking = FlyAnimation.wingBeatAmplitude(1.0F);
        expect(
                Math.abs(hovering - expectedHovering) <= TOLERANCE
                        && Math.abs(working - expectedWorking) <= TOLERANCE,
                "wing stroke must equal the declared amplitude mapping");

        // The vertical axis feeds the same effort channel as the forward axis.
        float climbing = wingDeflection(model, animated(0.0F, 1.0F, 0.0F, 0.0F), true);
        expect(
                Math.abs(climbing - expectedWorking) <= TOLERANCE,
                "vertical thrust must drive wing effort exactly as forward thrust does");

        // Beat rate rises with effort, so the phase itself is command-dependent.
        expect(
                FlyAnimation.wingBeatRate(1.0F) > FlyAnimation.wingBeatRate(0.0F) + TOLERANCE,
                "wing beat rate must rise with thrust effort");
        float idleAdvance = FlyAnimation.advanceWingBeatPhase(
                0.0F, 1.0F, FlyAnimation.wingBeatRate(0.0F));
        float workingAdvance = FlyAnimation.advanceWingBeatPhase(
                0.0F, 1.0F, FlyAnimation.wingBeatRate(1.0F));
        expect(
                workingAdvance > idleAdvance + TOLERANCE,
                "one tick of beat must advance further under thrust than at rest");
    }

    private static void assertWingBeatStaysAliveAtZeroCommand(FlyModel model) {
        float downStroke = wingDeflection(model, animated(0.0F, 0.0F, 0.0F, 0.0F), true);
        float upStroke = wingDeflection(model, animated(0.0F, 0.0F, 0.0F, 0.0F), false);
        expect(
                downStroke > TOLERANCE && upStroke < -TOLERANCE,
                "a fly with no command must still beat its wings through a full stroke, but the"
                        + " stroke was down=" + downStroke + " up=" + upStroke);
        expect(
                Math.abs((downStroke - upStroke) - 2.0F * FlyAnimation.WING_BEAT_AMPLITUDE_IDLE)
                        <= TOLERANCE,
                "the resting stroke must span exactly the declared idle amplitude");
    }

    private static void assertWingDifferentialTracksYawTorque(FlyModel model) {
        ModelPart leftWing = model.root().getChild("left_wing");
        ModelPart rightWing = model.root().getChild("right_wing");

        model.setupAnim(animated(0.5F, 0.0F, 0.0F, 0.0F));
        float symmetricLeft = -leftWing.zRot - FlyAnimation.WING_REST_SPLAY;
        float symmetricRight = rightWing.zRot - FlyAnimation.WING_REST_SPLAY;
        expect(
                Math.abs(symmetricLeft - symmetricRight) <= TOLERANCE
                        && Math.abs(leftWing.xRot - rightWing.xRot) <= TOLERANCE,
                "at zero yaw torque both wings must work equally hard");

        // Positive yaw torque commands the body toward its own right (ADR 0015), so the LEFT
        // wing is the outside wing and must work harder.
        model.setupAnim(animated(0.5F, 0.0F, 1.0F, 0.0F));
        float turningLeft = -leftWing.zRot - FlyAnimation.WING_REST_SPLAY;
        float turningRight = rightWing.zRot - FlyAnimation.WING_REST_SPLAY;
        float leftPitch = leftWing.xRot;
        float rightPitch = rightWing.xRot;
        expect(
                turningLeft > turningRight + TOLERANCE,
                "in a right turn the outside (left) wing must out-stroke the inside one, but"
                        + " left=" + turningLeft + " right=" + turningRight);
        expect(
                Math.abs(leftPitch - rightPitch) > TOLERANCE,
                "the two wings must differ in stroke-plane pitch during a turn");
        expect(
                turningRight > TOLERANCE,
                "the inside wing must keep a positive stroke rather than inverting");

        // The mirrored command must mirror the body exactly.
        model.setupAnim(animated(0.5F, 0.0F, -1.0F, 0.0F));
        float mirroredLeft = -leftWing.zRot - FlyAnimation.WING_REST_SPLAY;
        float mirroredRight = rightWing.zRot - FlyAnimation.WING_REST_SPLAY;
        expect(
                Math.abs(mirroredLeft - turningRight) <= TOLERANCE
                        && Math.abs(mirroredRight - turningLeft) <= TOLERANCE
                        && Math.abs(leftWing.xRot - rightPitch) <= TOLERANCE
                        && Math.abs(rightWing.xRot - leftPitch) <= TOLERANCE,
                "a mirrored yaw torque must mirror the wing differential exactly");
    }

    private static void assertLegsAndAbdomenTrackTheirAxes(FlyModel model) {
        ModelPart abdomen = model.root().getChild("abdomen");
        ModelPart leftLeg = model.root().getChild(FlyModel.LEFT_LEG_PARTS.get(0));
        ModelPart rightLeg = model.root().getChild(FlyModel.RIGHT_LEG_PARTS.get(0));

        model.setupAnim(animated(0.0F, 1.0F, 0.0F, 0.0F));
        float climbPitch = leftLeg.xRot;
        float climbSplay = Math.abs(leftLeg.zRot);
        expect(
                Math.abs(leftLeg.zRot + rightLeg.zRot) <= TOLERANCE
                        && climbSplay > TOLERANCE,
                "the two leg banks must stay mirrored about the body midline, got left="
                        + leftLeg.zRot + " right=" + rightLeg.zRot);

        model.setupAnim(animated(0.0F, -1.0F, 0.0F, 0.0F));
        float descentPitch = leftLeg.xRot;
        float descentSplay = Math.abs(leftLeg.zRot);

        expect(
                climbPitch > descentPitch + TOLERANCE,
                "a climbing fly must tuck its legs back relative to a descending one, but"
                        + " climb=" + climbPitch + " descent=" + descentPitch);
        expect(
                climbSplay < descentSplay - TOLERANCE,
                "a climbing fly must draw its legs in and a descending one splay them, but"
                        + " climb=" + climbSplay + " descent=" + descentSplay);

        // Every leg on a side moves together.
        model.setupAnim(animated(0.0F, 1.0F, 0.0F, 0.0F));
        for (String name : FlyModel.LEFT_LEG_PARTS) {
            ModelPart leg = model.root().getChild(name);
            expect(
                    Math.abs(leg.xRot - climbPitch) <= TOLERANCE
                            && Math.abs(Math.abs(leg.zRot) - climbSplay) <= TOLERANCE,
                    "all legs on one side must share the tuck pose, but " + name + " did not");
        }

        model.setupAnim(animated(1.0F, 0.0F, 0.0F, 0.0F));
        float acceleratingAbdomen = abdomen.xRot;
        model.setupAnim(animated(-1.0F, 0.0F, 0.0F, 0.0F));
        float brakingAbdomen = abdomen.xRot;
        expect(
                acceleratingAbdomen > brakingAbdomen + TOLERANCE,
                "abdomen pitch must track forward thrust, but accelerating="
                        + acceleratingAbdomen + " braking=" + brakingAbdomen);
        model.setupAnim(animated(0.0F, 0.0F, 0.0F, 0.0F));
        expect(
                Math.abs(abdomen.xRot - FlyAnimation.ABDOMEN_REST_PITCH) <= TOLERANCE,
                "an uncommanded abdomen must sit at its declared rest pitch");
    }

    /**
     * ADR 0020 lane B: a fly running CRUISE is drawn with its legs tucked, because level cruise
     * commands zero vertical thrust and the ADR 0016 mapping would otherwise draw it parked.
     *
     * <p>Render-only, and discriminating in both directions: the overlay fires only for the CRUISE
     * program id, it is a <em>floor</em> on the tuck rather than a replacement, and the pose it
     * produces is the declared constant exactly. Every other program still gets the unmodified
     * ADR 0016 mapping, which the assertions at the end check.
     */
    private static void assertCruiseTucksTheLegsWhileAirborne(FlyModel model) {
        ModelPart leftLeg = model.root().getChild(FlyModel.LEFT_LEG_PARTS.get(0));
        ModelPart rightLeg = model.root().getChild(FlyModel.RIGHT_LEG_PARTS.get(0));

        // Level flight under the default channel: the ADR 0016 mapping alone, at rest pose.
        model.setupAnim(animated(0.5F, 0.0F, 0.0F, 0.0F));
        float restPitch = leftLeg.xRot;
        float restSplay = Math.abs(leftLeg.zRot);
        expect(
                Math.abs(restPitch - FlyAnimation.LEG_REST_PITCH) <= TOLERANCE
                        && Math.abs(restSplay - FlyAnimation.LEG_REST_SPLAY) <= TOLERANCE,
                "level flight with no program overlay must draw the declared rest pose, got pitch="
                        + restPitch + " splay=" + restSplay);

        // The same level flight, with CRUISE published on the body-program channel.
        model.setupAnim(cruising(0.5F, 0.0F));
        float cruisePitch = leftLeg.xRot;
        float cruiseSplay = Math.abs(leftLeg.zRot);
        expect(
                cruisePitch > restPitch + TOLERANCE && cruiseSplay < restSplay - TOLERANCE,
                "a cruising fly must tuck its legs back and in rather than dangle them, but"
                        + " pitch went " + restPitch + " -> " + cruisePitch
                        + " and splay " + restSplay + " -> " + cruiseSplay);
        expect(
                Math.abs(
                                cruisePitch
                                        - (FlyAnimation.LEG_REST_PITCH
                                                + FlyAnimation.LEG_TUCK_SWING
                                                        * FlyAnimation.CRUISE_LEG_TUCK))
                                <= TOLERANCE
                        && Math.abs(
                                        cruiseSplay
                                                - (FlyAnimation.LEG_REST_SPLAY
                                                        - FlyAnimation.LEG_TUCK_SPREAD
                                                                * FlyAnimation.CRUISE_LEG_TUCK))
                                <= TOLERANCE,
                "the cruise tuck must equal the declared overlay fraction exactly");
        expect(
                Math.abs(leftLeg.zRot + rightLeg.zRot) <= TOLERANCE,
                "the tucked legs must stay mirrored about the body midline, got left="
                        + leftLeg.zRot + " right=" + rightLeg.zRot);
        for (String name : FlyModel.LEFT_LEG_PARTS) {
            ModelPart leg = model.root().getChild(name);
            expect(
                    Math.abs(leg.xRot - cruisePitch) <= TOLERANCE
                            && Math.abs(Math.abs(leg.zRot) - cruiseSplay) <= TOLERANCE,
                    "all legs on one side must share the cruise tuck, but " + name + " did not");
        }

        // The overlay is a floor, not a replacement: a CRUISE climbing off a floor still draws
        // the full climb pose the vertical-thrust mapping asks for.
        model.setupAnim(cruising(0.0F, 1.0F));
        expect(
                Math.abs(
                                leftLeg.xRot
                                        - (FlyAnimation.LEG_REST_PITCH
                                                + FlyAnimation.LEG_TUCK_SWING))
                        <= TOLERANCE,
                "a climbing CRUISE must keep the stronger of the two tucks, got " + leftLeg.xRot);

        // And the ADR 0016 mapping is untouched for every other program: a descending fly on the
        // default channel still extends its legs past the rest pose.
        model.setupAnim(animated(0.0F, -1.0F, 0.0F, 0.0F));
        float descentPitch = leftLeg.xRot;
        float descentSplay = Math.abs(leftLeg.zRot);
        expect(
                descentPitch < restPitch - TOLERANCE && descentSplay > restSplay + TOLERANCE,
                "the ADR 0016 descent pose must be unchanged for programs other than CRUISE");

        // A DESCENDING cruise keeps that same descent pose. The overlay is a floor on a
        // non-negative tuck, so it must not turn a body being pushed down off a ceiling into a
        // body drawn as though it were climbing.
        model.setupAnim(cruising(0.0F, -1.0F));
        expect(
                Math.abs(leftLeg.xRot - descentPitch) <= TOLERANCE
                        && Math.abs(Math.abs(leftLeg.zRot) - descentSplay) <= TOLERANCE,
                "a descending CRUISE must keep the ADR 0016 extended-leg descent pose, but pitch"
                        + " went " + descentPitch + " -> " + leftLeg.xRot
                        + " and splay " + descentSplay + " -> " + Math.abs(leftLeg.zRot));
    }

    // ---- ADR 0020 lane D: the escape profile has to survive the filter ------------------

    /**
     * {@code escapeJumpProfileSurvivesTheClientSmoother}: ESCAPE_JUMP's declared phase durations
     * are long enough that each phase is still <em>drawn</em> after the client's tau = 3-tick
     * exponential filter, at both ends of the urgency range.
     *
     * <p>This drives the <em>real</em> {@link FlyPresentationSmoother} with the <em>real</em>
     * program output, one game tick per step, warmed first, and then reads the leg rotation the
     * real {@link FlyModel} produces. Nothing about the filter or the leg mapping is restated here.
     *
     * <p>It measures three things rather than one span, because a span is not legibility: how much
     * of the commanded crouch the legs reach, how much of the commanded launch they reach, and how
     * much of the crouch-to-launch step lands in a single tick. A split like 1/21/2 passes a span
     * test and fails all three of these.
     *
     * <p><strong>Presentation only.</strong> This says nothing about what the body does; it says
     * the body's declared motion is still legible once drawn.
     */
    private static void assertEscapeJumpProfileSurvivesTheClientSmoother(FlyModel model) {
        assertEscapeProfileIsDrawn(
                model,
                Modulation.NEUTRAL,
                SMOOTHER_CROUCH_FRACTION,
                SMOOTHER_LAUNCH_FRACTION);
        assertEscapeProfileIsDrawn(
                model,
                new Modulation(1.0, 1.0, 1.0, 1.0),
                SMOOTHER_CROUCH_FRACTION_AT_FULL_URGENCY,
                SMOOTHER_LAUNCH_FRACTION_AT_FULL_URGENCY);
    }

    private static void assertEscapeProfileIsDrawn(
            FlyModel model, Modulation modulation, float crouchFloor, float launchFloor) {
        int percent = EscapeJumpProgram.urgencyPercent(modulation);
        int crouchTicks = EscapeJumpProgram.crouchTicks(percent);
        int launchTicks = EscapeJumpProgram.launchTicks(percent);
        String lane = " at urgency " + percent + "% (" + crouchTicks + "/" + launchTicks + ")";

        FlyPresentationSmoother smoother = new FlyPresentationSmoother();
        FlyRenderState state = new FlyRenderState();
        state.appliedAction = PrimitiveAction.NEUTRAL;
        // The published channel says ESCAPE_JUMP, so the CRUISE leg overlay is provably not what
        // is being measured here: the drawn tuck is the ADR 0016 mapping alone.
        state.bodyProgram = new BodyProgramChannel(
                SelectionSource.READOUT,
                BodyProgramId.ESCAPE_JUMP,
                modulation,
                ProgramParameters.NEUTRAL);
        float age = 0.0F;
        for (int tick = 0; tick < SMOOTHER_WARMUP_TICKS; tick++) {
            smoother.advance(PrimitiveAction.NEUTRAL, age);
            age += 1.0F;
        }

        EscapeJumpProgram program = new EscapeJumpProgram();
        BodyProgramContext context = BodyProgramContext.sensorsOnly(
                EscapeProfileFence.raySnapshot(
                        FlySensors.RAY_LENGTH_BLOCKS,
                        FlySensors.RAY_LENGTH_BLOCKS,
                        FlySensors.RAY_LENGTH_BLOCKS));
        BodyProgramState programState =
                BodyProgramState.entering(BodyProgramId.ESCAPE_JUMP, 0L);
        float lastCrouchDrawn = Float.NaN;
        float firstLaunchDrawn = Float.NaN;
        float lastLaunchDrawn = Float.NaN;
        for (int tick = 0; tick < EscapeJumpProgram.ESCAPE_JUMP_TICKS; tick++) {
            BodyProgramStep step =
                    program.step(programState, context, modulation, ProgramParameters.NEUTRAL);
            smoother.advance(step.motors(), age);
            age += 1.0F;
            smoother.writeTo(state);
            state.ageInTicks = age;
            model.setupAnim(state);
            float drawn = (model.root().getChild(FlyModel.LEFT_LEG_PARTS.get(0)).xRot
                            - FlyAnimation.LEG_REST_PITCH)
                    / FlyAnimation.LEG_TUCK_SWING;
            if (tick == crouchTicks - 1) {
                lastCrouchDrawn = drawn;
            } else if (tick == crouchTicks) {
                firstLaunchDrawn = drawn;
            } else if (tick == crouchTicks + launchTicks - 1) {
                lastLaunchDrawn = drawn;
            }
            programState = step.next();
        }

        float crouchFraction =
                lastCrouchDrawn / (float) EscapeJumpProgram.ESCAPE_CROUCH_VERTICAL;
        float launchFraction =
                lastLaunchDrawn / (float) EscapeJumpProgram.ESCAPE_LAUNCH_VERTICAL;
        float boundaryJump = firstLaunchDrawn - lastCrouchDrawn;
        float idealStep = (1.0F - (float) Math.exp(-1.0 / FlyAnimation.SMOOTHING_TAU_TICKS))
                * (float) (EscapeJumpProgram.ESCAPE_LAUNCH_VERTICAL
                        - EscapeJumpProgram.ESCAPE_CROUCH_VERTICAL);

        expect(
                crouchFraction >= crouchFloor,
                "the drawn crouch reached only " + crouchFraction + " of its commanded depth"
                        + lane + ", below the declared " + crouchFloor);
        expect(
                launchFraction >= launchFloor,
                "the drawn launch reached only " + launchFraction + " of its commanded height"
                        + lane + ", below the declared " + launchFloor);
        expect(
                boundaryJump >= SMOOTHER_BOUNDARY_JUMP_FRACTION * idealStep,
                "the crouch-to-launch snap drew only " + boundaryJump + " in one tick" + lane
                        + ", below " + SMOOTHER_BOUNDARY_JUMP_FRACTION + " of the ideal step "
                        + idealStep);
    }

    // ---- bank -------------------------------------------------------------------------

    private static void assertBankIsZeroSignedAndClamped() {
        expect(
                bankFor(0.0F) == 0.0F,
                "zero yaw torque must produce exactly zero bank, got " + bankFor(0.0F));

        // The live band. ADR 0015 records the eem-v1 sided boundary at +0.0054 / -0.0346 and
        // D-067 records the measured yaw as 0.5-3.5% of the rail; these must be visible, which is
        // the entire purpose of the declared visual reference amplitude.
        float smallest = bankFor(0.0054F);
        float largest = bankFor(0.0346F);
        expect(
                Math.abs(smallest) >= 3.0F,
                "the smallest recorded live yaw torque must still bank the body several degrees,"
                        + " got " + smallest);
        expect(
                Math.abs(largest) > Math.abs(smallest),
                "a larger yaw torque must bank the body further");

        // Sign: a positive yaw torque and its negation must bank in opposite directions.
        expect(
                Math.signum(bankFor(0.02F)) == -Math.signum(bankFor(-0.02F))
                        && bankFor(0.02F) != 0.0F,
                "opposite yaw torques must bank in opposite directions");

        // Clamp: the rails, and well past them, saturate at exactly the declared maximum.
        for (float extreme : new float[] {1.0F, -1.0F, 0.5F, -0.5F}) {
            float bank = bankFor(extreme);
            expect(
                    Math.abs(Math.abs(bank) - FlyAnimation.MAX_BANK_DEGREES) <= TOLERANCE,
                    "yaw torque " + extreme + " must clamp to the declared "
                            + FlyAnimation.MAX_BANK_DEGREES + " degree bank, got " + bank);
        }
        expect(
                Math.abs(
                                Math.abs(bankFor(FlyAnimation.YAW_TORQUE_VISUAL_REFERENCE))
                                        - FlyAnimation.MAX_BANK_DEGREES)
                        <= TOLERANCE,
                "the declared reference amplitude must map to exactly full bank");
    }

    private static void assertBankLowersTheInsideOfTheTurn() {
        // End-to-end through the real pose the renderer builds, so the sign is pinned through the
        // yaw offset and the negative model scale rather than trusted from the mapping in
        // isolation. Positive yaw torque commands a turn toward the body's right (ADR 0015), and
        // a body banking into that turn drops its right side.
        float level = wingTipHeightDifference(0.0F);
        expect(
                Math.abs(level) <= TOLERANCE,
                "with no bank the wing tips must sit at the same height, got " + level);

        float rightTurn = wingTipHeightDifference(bankFor(0.0346F));
        expect(
                rightTurn > TOLERANCE,
                "banking into a right turn must put the left wing tip above the right one, but"
                        + " the left-minus-right height was " + rightTurn);

        float leftTurn = wingTipHeightDifference(bankFor(-0.0346F));
        expect(
                leftTurn < -TOLERANCE
                        && Math.abs(leftTurn + rightTurn) <= TOLERANCE,
                "the mirrored command must mirror the bank exactly, got " + leftTurn);
    }

    /** World-space height of the left wing tip minus that of the right, under a given bank. */
    private static float wingTipHeightDifference(float bankRotationDegrees) {
        PoseStack poses = new PoseStack();
        FlyRenderer.applyBodyPose(poses, 0.0F, 0.0F, bankRotationDegrees);
        Vector3f left = new Vector3f(WING_TIP_MODEL_X, 0.0F, 0.0F);
        Vector3f right = new Vector3f(-WING_TIP_MODEL_X, 0.0F, 0.0F);
        poses.last().pose().transformPosition(left);
        poses.last().pose().transformPosition(right);
        return left.y() - right.y();
    }

    private static float bankFor(float yawTorque) {
        return FlyAnimation.bankRotationDegrees(FlyAnimation.visualYawTorque(yawTorque));
    }

    // ---- render-path isolation ---------------------------------------------------------

    private static void assertRenderExtractionLeavesEntityStateUntouched(
            ClientGameTestContext context) {
        try (var singleplayer = context.worldBuilder().create()) {
            singleplayer.getServer().runCommand("gamemode spectator @p");
            singleplayer.getServer().runCommand("tp @p 0 80 0 0 0");
            singleplayer.getServer().runOnServer(server -> {
                FlyEntity fly =
                        NeuroCraftFly.FLY.create(server.overworld(), EntitySpawnReason.COMMAND);
                if (fly == null) {
                    throw new AssertionError("could not construct render-isolation fly");
                }
                fly.resetForFixture(0.0, 81.5, 4.0, 180.0F);
                if (!server.overworld().addFreshEntity(fly)) {
                    throw new AssertionError("could not spawn render-isolation fly");
                }
            });
            context.waitFor(client -> client.level != null && findFly(client.level) != null);
            context.waitTicks(5);

            context.runOnClient(client -> {
                ClientLevel level = client.level;
                if (level == null) {
                    throw new AssertionError("client level vanished before render isolation check");
                }
                FlyEntity fly = findFly(level);
                if (fly == null) {
                    throw new AssertionError("render-isolation fly never reached the client");
                }
                EntityRenderer<? super FlyEntity, ?> renderer =
                        client.getEntityRenderDispatcher().getRenderer(fly);
                if (!(renderer instanceof FlyRenderer flyRenderer)) {
                    throw new AssertionError(
                            "the fly must be drawn by the project renderer, got " + renderer);
                }

                FlyRenderState state = flyRenderer.createRenderState();
                EntitySnapshot before = EntitySnapshot.of(fly);
                float firstPhase = Float.NaN;
                float lastPhase = Float.NaN;
                for (int frame = 0; frame < 40; frame++) {
                    flyRenderer.extractRenderState(fly, state, frame / 40.0F);
                    if (frame == 0) {
                        firstPhase = state.wingBeatPhase;
                    }
                    lastPhase = state.wingBeatPhase;
                }
                EntitySnapshot after = EntitySnapshot.of(fly);

                if (!before.equals(after)) {
                    throw new AssertionError(
                            "the render path mutated entity state: before=" + before
                                    + " after=" + after);
                }
                if (!(Math.abs(lastPhase - firstPhase) > TOLERANCE)) {
                    throw new AssertionError(String.format(
                            Locale.ROOT,
                            "the live render path must advance the wing beat across frames,"
                                    + " but the phase stayed at %.9f",
                            firstPhase));
                }
                if (state.bankRotationDegrees != 0.0F || state.thrustEffort != 0.0F) {
                    throw new AssertionError(
                            "an uncommanded fly must extract to a level body at rest, got bank="
                                    + state.bankRotationDegrees
                                    + " effort=" + state.thrustEffort);
                }
                if (state.onGround != fly.onGround()) {
                    throw new AssertionError("render state did not copy actual ground contact");
                }
            });
            BodyProgramPoseClientChecks.observe(context);
        }
    }

    private static FlyEntity findFly(ClientLevel level) {
        for (Entity entity : level.entitiesForRendering()) {
            if (entity instanceof FlyEntity fly) {
                return fly;
            }
        }
        return null;
    }

    // ---- helpers ------------------------------------------------------------------------

    /**
     * A render state at a fixed clock and a fixed wing-beat phase, so that any difference between
     * two of them is attributable to the motor vector alone.
     */
    private static FlyRenderState animated(
            float forwardThrust, float verticalThrust, float visualYawTorque, float wingPhase) {
        FlyRenderState state = new FlyRenderState();
        state.ageInTicks = 100.0F;
        state.appliedAction = PrimitiveAction.NEUTRAL;
        state.smoothedForwardThrust = forwardThrust;
        state.smoothedVerticalThrust = verticalThrust;
        state.thrustEffort = FlyAnimation.thrustEffort(forwardThrust, verticalThrust);
        state.visualYawTorque = visualYawTorque;
        state.bankRotationDegrees = FlyAnimation.bankRotationDegrees(visualYawTorque);
        state.wingBeatPhase = wingPhase == 0.0F ? (float) (Math.PI / 2.0) : wingPhase;
        return state;
    }

    /** A render state at a fixed clock whose body-program channel publishes a running CRUISE. */
    private static FlyRenderState cruising(float forwardThrust, float verticalThrust) {
        FlyRenderState state = animated(forwardThrust, verticalThrust, 0.0F, 0.0F);
        state.bodyProgram = new BodyProgramChannel(
                SelectionSource.OPERATOR,
                BodyProgramId.CRUISE,
                Modulation.NEUTRAL,
                ProgramParameters.NEUTRAL);
        return state;
    }

    /** Stroke deflection of the left wing away from its rest splay, at the peak of a stroke. */
    private static float wingDeflection(FlyModel model, FlyRenderState state, boolean downStroke) {
        state.wingBeatPhase = downStroke
                ? (float) (Math.PI / 2.0)
                : (float) (3.0 * Math.PI / 2.0);
        model.setupAnim(state);
        return -model.root().getChild("left_wing").zRot - FlyAnimation.WING_REST_SPLAY;
    }

    private static void expect(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }

    private record EntitySnapshot(
            double x,
            double y,
            double z,
            float yRot,
            float xRot,
            Vec3 deltaMovement,
            PrimitiveAction action,
            ControllerFault controllerState,
            int tickCount,
            BodyProgramChannel bodyProgram,
            float programPhase,
            boolean onGround) {
        static EntitySnapshot of(FlyEntity fly) {
            return new EntitySnapshot(
                    fly.getX(),
                    fly.getY(),
                    fly.getZ(),
                    fly.getYRot(),
                    fly.getXRot(),
                    fly.getDeltaMovement(),
                    fly.presentationAction(),
                    fly.controllerState(),
                    fly.tickCount,
                    fly.bodyProgramChannel(),
                    fly.bodyProgramPhase(),
                    fly.onGround());
        }
    }
}
