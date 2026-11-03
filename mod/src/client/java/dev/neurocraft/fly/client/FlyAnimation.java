package dev.neurocraft.fly.client;

import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.control.PrimitiveAction;

/**
 * Declared presentation mapping from synchronized motors and scripted program channels onto the
 * renderer's visual degrees of freedom (ADRs 0016 and 0020).
 *
 * <p><strong>Render-only.</strong> Every member here is a pure function of primitives and of the
 * already-published {@link PrimitiveAction}. This class holds no reference to an entity, a level,
 * or the bridge; nothing it returns re-enters the controller, the actuator, the trace, or any
 * declared identity. It changes how the fly is <em>drawn</em> and nothing else.
 *
 * <p><strong>The visual gain is declared presentation engineering, not amplification.</strong>
 * {@link #YAW_TORQUE_VISUAL_REFERENCE} rescales the yaw channel for the screen only. The command
 * that reaches {@code PrimitiveActuator} is untouched and unscaled. A fly drawn at full visual
 * bank is executing a yaw torque of about 3.5% of its rail, and a rendered bank must never be read
 * or reported as the fly steering hard.
 */
public final class FlyAnimation {
    /**
     * Exponential-smoothing time constant, in game ticks, applied to every motor channel before it
     * reaches the model. Commands arrive at 20 Hz and the client draws faster, so the raw channel
     * is a staircase. 3.0 ticks (150 ms) is slow enough to erase the step and fast enough that a
     * command change is on screen inside a tenth of a second. Declared presentation constant.
     */
    public static final float SMOOTHING_TAU_TICKS = 3.0F;

    /**
     * Upper bound, in ticks, on the elapsed time a single smoothing step may integrate. It bounds
     * the catch-up after a pause, a chunk reload, or a dropped frame so the body eases in rather
     * than snapping. Declared presentation constant.
     */
    public static final float MAX_SMOOTHING_STEP_TICKS = 20.0F;

    /**
     * Yaw-torque magnitude drawn at full visual deflection. The committed live band is small:
     * ADR 0015 records the eem-v1 sided boundary at {@code +0.0054} and {@code -0.0346}, and D-067
     * records the measured yaw as 0.5-3.5% of the +/-1 rail. {@code 0.035} is the top of that band,
     * so a real command at its recorded maximum reaches full bank and anything larger clamps.
     * A linear map would leave the entire live band inside one degree of roll and the screen would
     * stay a level brick. Declared presentation constant; it scales pixels, never the command.
     */
    public static final float YAW_TORQUE_VISUAL_REFERENCE = 0.035F;

    /**
     * Maximum render-only body bank, in degrees. The entity has no roll axis at all server-side;
     * this rotation exists only in the {@link com.mojang.blaze3d.vertex.PoseStack}. 25 degrees is
     * a bank a viewer reads as "leaning into the turn" without the model clipping its own shadow.
     * Declared presentation constant.
     */
    public static final float MAX_BANK_DEGREES = 25.0F;

    /**
     * Wing-beat angular rate at zero command, in radians per tick. This is the pre-ADR-0016 rate
     * verbatim, so a hovering fly beats at exactly the rate it always did; its amplitude does
     * change (see WING_BEAT_AMPLITUDE_IDLE, below the pre-ADR-0016 fixed 0.48). Declared
     * presentation constant.
     */
    public static final float WING_BEAT_RATE_IDLE = 2.4F;

    /**
     * Additional wing-beat rate at full thrust effort, in radians per tick. A fly working its
     * hardest beats 1.5x as fast as one hovering. Declared presentation constant.
     */
    public static final float WING_BEAT_RATE_GAIN = 1.2F;

    /**
     * Wing-beat stroke amplitude at zero command, in radians (about 21 degrees). Below the
     * pre-ADR-0016 fixed 0.48, so a hovering fly reads as calm rather than as frantic.
     * Declared presentation constant.
     */
    public static final float WING_BEAT_AMPLITUDE_IDLE = 0.36F;

    /**
     * Additional stroke amplitude at full thrust effort, in radians. Declared presentation
     * constant; the idle-plus-gain sum (0.60 rad, about 34 degrees) is the maximum stroke.
     */
    public static final float WING_BEAT_AMPLITUDE_GAIN = 0.24F;

    /**
     * Stroke-amplitude difference between the outside and inside wing at full visual yaw, in
     * radians. Kept strictly below {@link #WING_BEAT_AMPLITUDE_IDLE} so the inside wing never
     * inverts its stroke. Declared presentation constant.
     */
    public static final float WING_DIFFERENTIAL_AMPLITUDE = 0.18F;

    /**
     * Stroke-plane pitch difference between the two wings at full visual yaw, in radians.
     * Declared presentation constant.
     */
    public static final float WING_DIFFERENTIAL_PITCH = 0.20F;

    /** Wing rest splay, in radians: the pre-ADR-0016 model value, retained verbatim. */
    public static final float WING_REST_SPLAY = 0.25F;

    /** Wing rest pitch, in radians: the pre-ADR-0016 baked pose value, retained verbatim. */
    public static final float WING_REST_PITCH = -0.08F;

    /** Leg rest pitch, in radians: the pre-ADR-0016 baked pose value, retained verbatim. */
    public static final float LEG_REST_PITCH = -0.18F;

    /** Leg rest splay magnitude, in radians: the pre-ADR-0016 baked pose value, retained. */
    public static final float LEG_REST_SPLAY = 0.65F;

    /**
     * Fore/aft leg swing at full vertical thrust, in radians. Positive vertical thrust (a climb)
     * swings the legs back and up into a tuck; negative (a descent) swings them forward into the
     * extended landing posture. Declared presentation constant.
     */
    public static final float LEG_TUCK_SWING = 0.55F;

    /**
     * Leg splay change at full vertical thrust, in radians. A climbing fly draws its legs in under
     * the body; a descending one splays them. Kept below {@link #LEG_REST_SPLAY} so the legs never
     * cross the midline. Declared presentation constant.
     */
    public static final float LEG_TUCK_SPREAD = 0.22F;

    /**
     * Leg-tuck fraction drawn while the CRUISE body program is running, on the same {@code [-1, +1]}
     * tuck channel {@link #LEG_TUCK_SWING} and {@link #LEG_TUCK_SPREAD} already scale (ADR 0020
     * lane B).
     *
     * <p><strong>A declared presentation overlay, and it exists because of a gap in the ADR 0016
     * mapping's coverage rather than anything wrong with CRUISE.</strong> Level cruise commands
     * zero vertical thrust — that is exactly what its clearance loop is for — so
     * {@code smoothedVerticalThrust} is zero and the leg mapping below draws a cruising fly at its
     * full rest splay, legs dangling as though it were parked. 0.6 is most of the tuck a full climb
     * draws, so a cruising fly reads as airborne without reading as climbing.
     *
     * <p><strong>It is a floor on a non-negative tuck, not a replacement.</strong> A climbing CRUISE
     * keeps the full climb pose the vertical channel asked for, and a <em>descending</em> CRUISE --
     * the clearance loop pushing the body down off a ceiling -- keeps the ADR 0016 extended-leg
     * descent pose untouched, because a landing-shaped posture is the honest drawing of a body
     * moving down.
     *
     * <p>It scales pixels and nothing else. No motor, no command, no trace and no declared identity
     * sees it, and the synchronized program id it keys on is a presentation channel: a tucked leg
     * is not anatomy and not a connectome output.
     */
    public static final float CRUISE_LEG_TUCK = 0.6F;

    /** Abdomen rest pitch, in radians: the pre-ADR-0016 baked pose value, retained verbatim. */
    public static final float ABDOMEN_REST_PITCH = -0.10F;

    /**
     * Abdomen pitch swing at full forward thrust, in radians. Positive forward thrust raises the
     * abdomen; braking drops it. Declared presentation constant.
     */
    public static final float ABDOMEN_PITCH_SWING = 0.30F;

    /** Idle antenna sweep rate, in radians per tick: the pre-ADR-0016 value, retained verbatim. */
    public static final float ANTENNA_IDLE_RATE = 0.18F;

    /** Idle antenna sweep amplitude, in radians: the pre-ADR-0016 value, retained verbatim. */
    public static final float ANTENNA_IDLE_SWEEP = 0.08F;

    /** Antenna rest yaw magnitude, in radians: the pre-ADR-0016 baked pose value, retained. */
    public static final float ANTENNA_REST_YAW = 0.18F;

    // Scripted program poses: display units/radians, chosen for legibility, not anatomy.
    public static final float LAND_LEG_EXTENSION = -0.8F;
    public static final float FLEE_LEG_TUCK = 0.85F;
    public static final float APPROACH_LEG_TUCK = 0.3F;
    public static final float GROUND_WING_FOLD_YAW = 0.95F;
    public static final float GROUND_WING_SPLAY = 0.10F;
    public static final float WALK_LEG_SWING = 0.65F;
    public static final float WALK_LEG_ROLL_SWING = 0.25F;
    public static final float GROOM_FORELEG_REACH = 1.25F;
    public static final float GROOM_ANTENNA_REACH_EXTRA = 0.75F;
    public static final float GROOM_FORELEG_INWARD_ROLL = 0.85F;
    public static final float GROOM_RUB_SWING = 0.16F;
    public static final float GROOM_ANTENNA_SWEEP = 0.35F;
    public static final float PROBOSCIS_PITCH = 0.55F;
    public static final float PROBOSCIS_LENGTH = 3.4F;
    public static final float APPROACH_ANTENNA_SPREAD = 0.32F;
    public static final float FLEE_ANTENNA_SPREAD = 0.06F;

    private static final float TWO_PI = (float) (Math.PI * 2.0);

    private FlyAnimation() {}

    /** Rest wings only on actual ground contact, never merely from a program's name. */
    public static boolean groundPose(BodyProgramId program, boolean onGround) {
        return onGround && (program == BodyProgramId.LAND || program == BodyProgramId.WALK
                || program == BodyProgramId.GROOM || program == BodyProgramId.FEED);
    }

    /** Published normalized phase; malformed presentation cannot produce non-finite transforms. */
    public static float programPhase(float value) {
        return Float.isFinite(value) ? Math.max(0.0F, Math.min(Math.nextDown(1.0F), value)) : 0.0F;
    }

    /** Opposing tripods: left front/hind plus right middle, then the other three legs. */
    public static float walkSwing(float phase, int legIndex, boolean left) {
        boolean firstTripod = (legIndex % 2 == 0) == left;
        return (float) Math.sin(TWO_PI * programPhase(phase)) * (firstTripod ? 1.0F : -1.0F);
    }

    /** FEED fixed quarters: settle, extend, hold, retract; urgency is already in published phase. */
    public static float feedingExtension(float phase) {
        float quarters = programPhase(phase) * 4.0F;
        if (quarters < 1.0F) return 0.0F;
        if (quarters < 2.0F) return smoothUnit(quarters - 1.0F);
        if (quarters < 3.0F) return 1.0F;
        return 1.0F - smoothUnit(quarters - 3.0F);
    }

    /** GROOM fixed quarters: settle, forelegs, antennae, recover. */
    public static float groomingReach(float phase) {
        float quarters = programPhase(phase) * 4.0F;
        if (quarters < 1.0F) return 0.0F;
        if (quarters < 2.0F) return smoothUnit(quarters - 1.0F);
        if (quarters < 3.0F) return 1.0F;
        return 1.0F - smoothUnit(quarters - 3.0F);
    }

    /** Two bounded rub cycles inside the named active quarter; zero outside it. */
    public static float groomingRub(float phase, int quarter) {
        float local = programPhase(phase) * 4.0F - quarter;
        if (local < 0.0F || local >= 1.0F) return 0.0F;
        return (float) (Math.sin(local * TWO_PI * 2.0F) * Math.sin(local * Math.PI));
    }

    /** Raise the forelegs above the face only during the antenna quarter. */
    public static float groomingAntennaReach(float phase) {
        float local = programPhase(phase) * 4.0F - 2.0F;
        return local < 0.0F || local >= 1.0F ? 0.0F : (float) Math.sin(local * Math.PI);
    }

    private static float smoothUnit(float value) {
        return value * value * (3.0F - 2.0F * value);
    }

    /**
     * How hard the body is working, in [0, 1]: the magnitude of the commanded thrust in the
     * forward/vertical plane, saturated at unit. Lateral thrust is deliberately excluded — it is
     * hardwired to zero under the pinned faithful projection and carries no wing-effort reading.
     */
    public static float thrustEffort(float forwardThrust, float verticalThrust) {
        float magnitude =
                (float) Math.sqrt(forwardThrust * forwardThrust + verticalThrust * verticalThrust);
        return clampUnitInterval(magnitude);
    }

    /**
     * The yaw channel rescaled for the screen: the commanded torque divided by the declared
     * {@link #YAW_TORQUE_VISUAL_REFERENCE} and clamped to [-1, +1]. Positive means the body is
     * commanded toward its own right (ADR 0015: positive yaw torque increases yaw, and increasing
     * yaw walks the heading SOUTH -> WEST -> NORTH -> EAST, i.e. clockwise, i.e. rightward).
     *
     * <p>This is a display scaling of about 28.6x. It changes no command and adds no information:
     * the same number, drawn larger.
     */
    public static float visualYawTorque(float yawTorque) {
        return clampSigned(yawTorque / YAW_TORQUE_VISUAL_REFERENCE);
    }

    /**
     * The render-only bank angle to hand to {@code Axis.ZP.rotationDegrees}, in degrees, for a
     * given {@linkplain #visualYawTorque visual yaw}.
     *
     * <p>Sign, derived once and pinned end-to-end by a GameTest rather than trusted here: in
     * {@code FlyModel} coordinates, after the renderer's {@code (-1, -1, +1)} scale, model {@code
     * +X} is the body's left, model {@code +Y} is world-down and model {@code +Z} is the body's
     * back. A positive rotation about model {@code +Z} therefore lowers the body's <em>left</em>
     * side. Banking into a right turn requires lowering the <em>right</em> side, so a positive
     * visual yaw maps to a negative {@code ZP} angle.
     */
    public static float bankRotationDegrees(float visualYawTorque) {
        return -clampSigned(visualYawTorque) * MAX_BANK_DEGREES;
    }

    /** Wing-beat angular rate in radians per tick for a given {@linkplain #thrustEffort effort}. */
    public static float wingBeatRate(float thrustEffort) {
        return WING_BEAT_RATE_IDLE + WING_BEAT_RATE_GAIN * clampUnitInterval(thrustEffort);
    }

    /** Wing-beat stroke amplitude in radians for a given {@linkplain #thrustEffort effort}. */
    public static float wingBeatAmplitude(float thrustEffort) {
        return WING_BEAT_AMPLITUDE_IDLE
                + WING_BEAT_AMPLITUDE_GAIN * clampUnitInterval(thrustEffort);
    }

    /**
     * Frame-rate-independent exponential-smoothing weight for an elapsed interval, in [0, 1).
     * Using {@code 1 - exp(-dt/tau)} rather than a per-frame constant keeps the smoothed value on
     * the same trajectory whether the client draws at 30 or 300 frames per second.
     */
    public static float smoothingWeight(float elapsedTicks) {
        float bounded = boundElapsedTicks(elapsedTicks);
        return 1.0F - (float) Math.exp(-bounded / SMOOTHING_TAU_TICKS);
    }

    /** One exponential-smoothing step of {@code current} toward {@code target}. */
    public static float smoothTowards(float current, float target, float elapsedTicks) {
        return current + (target - current) * smoothingWeight(elapsedTicks);
    }

    /** Advances a wing-beat phase by an elapsed interval, wrapped into [0, 2*pi). */
    public static float advanceWingBeatPhase(float phase, float elapsedTicks, float rate) {
        float advanced = phase + boundElapsedTicks(elapsedTicks) * rate;
        float wrapped = advanced % TWO_PI;
        return wrapped < 0.0F ? wrapped + TWO_PI : wrapped;
    }

    /** Clamps a raw elapsed interval into the declared {@code [0, MAX_SMOOTHING_STEP_TICKS]}. */
    public static float boundElapsedTicks(float elapsedTicks) {
        if (!Float.isFinite(elapsedTicks) || elapsedTicks <= 0.0F) {
            return 0.0F;
        }
        return Math.min(elapsedTicks, MAX_SMOOTHING_STEP_TICKS);
    }

    /** Clamps to [-1, +1]. */
    public static float clampSigned(float value) {
        return Math.max(-1.0F, Math.min(1.0F, value));
    }

    /** Clamps to [0, 1]. */
    public static float clampUnitInterval(float value) {
        return Math.max(0.0F, Math.min(1.0F, value));
    }
}
