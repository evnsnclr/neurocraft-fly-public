package dev.neurocraft.fly.body.programs;

import dev.neurocraft.fly.body.BodyProgram;
import dev.neurocraft.fly.body.BodyProgramContext;
import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.BodyProgramState;
import dev.neurocraft.fly.body.BodyProgramStep;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ProgramParameters;
import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import dev.neurocraft.fly.control.PrimitiveAction;
import java.util.Optional;

/**
 * ESCAPE_JUMP: a short, bounded, self-arresting jump directed away from the nearest obstacle
 * (ADR 0020, programs table entry 8).
 *
 * <p><strong>This is scripted presentation and engineering. It is never anatomy, and it is never
 * connectome output.</strong> Every constant below was chosen by a person and is named here so it
 * can be argued with. Nothing in this file is derived from the MaleCNS connectome, from fly
 * anatomy, or from any measurement of fly behaviour, and a fly seen jumping is not evidence that
 * the connectome did anything. The readout that will select this program in stage 4 is a
 * hand-chosen mapping from a labeled population to a program identifier (ADR 0019); the jump it
 * selects is this script, unchanged, and the selection must be reported as exactly that.
 *
 * <p><strong>Why this program exists.</strong> The first surviving live session recorded the neural
 * lane's "escape" as a directionless push into the obstacle that lasted 83 seconds, and before that
 * a climb that ended in the ceiling. A reflex that neither points away from the threat nor stops is
 * not an escape. This program answers all three: the profile ends on its own fixed clock, the yaw
 * turns the body away from the shortest horizontal ray, and the last phase actively brakes the
 * vertical axis so the body is not still climbing when the program hands over.
 *
 * <p><strong>Three latches, and why they exist.</strong> The away side, the entry urgency, and the
 * two launch clearance scales are all decided once and carried in
 * {@link BodyProgramState#phaseIndex()} — see {@link #latch} for the bit layout. Re-deciding them
 * every tick was a measured defect, not a hypothetical one: with one obstacle sitting just outside
 * the side ray at entry, turning toward the open side sweeps that ray onto the obstacle, the
 * comparison flips, and the body dithers instead of escaping. A reflex commits.
 *
 * <p><strong>What it reads.</strong> Four ray distances, the vertical velocity, and {@code urgency}.
 * It reads no other modulation scalar, so varying {@code speed}, {@code turnRate}, or
 * {@code amplitude} produces a bit-identical motor sequence. It holds no field, no clock, and no
 * random source: {@link #step} is a function of {@code (state, context, modulation)} alone.
 *
 * <p><strong>Entry is unconditional</strong> — a reflex has no hidden gate — and the last tick names
 * {@link BodyProgramId#CRUISE}, which is registered, so the runtime enters it for real and CRUISE's
 * own vertical-rate loop holds whatever this program hands it.
 */
public final class EscapeJumpProgram implements BodyProgram {
    /**
     * Profile length in game ticks: 24, 1.2 s at 20 Hz, <strong>for every urgency</strong>.
     *
     * <p>Fixed and not extendable, and — since the review — not shortenable either. A jump that
     * lasts is not a jump, and the 83-second push this program replaces is what an unbounded one
     * looks like. Stage 4b's selection policy holds ESCAPE_JUMP for {@code ESCAPE_HOLD_TICKS = 24}
     * and re-sends its modulation every frame; a duration that varied with the urgency of the
     * frame would let the program exit early and be re-selected into a second jump inside one
     * hold. <strong>The policy's hold and this program's duration are equal by construction.</strong>
     */
    public static final int ESCAPE_JUMP_TICKS = 24;

    /** Crouch length at zero urgency: ticks 0-5 of the declared profile. */
    public static final int ESCAPE_CROUCH_TICKS = 6;

    /** Launch length at zero urgency: ticks 6-13 of the declared profile. */
    public static final int ESCAPE_LAUNCH_TICKS = 8;

    /**
     * Shortest crouch the urgency reshaping may produce, in ticks.
     *
     * <p>{@code FlyAnimation}'s client smoother runs at {@code SMOOTHING_TAU_TICKS = 3.0}, so a
     * step held for fewer than about four ticks is largely gone before it is drawn: the legs would
     * never reach the crouch and the launch snap would be a smear.
     * {@code escapeJumpProfileSurvivesTheClientSmoother} measures the drawn depth of each phase at
     * both ends of the urgency range, and the executed 2/2/2 negative control is what says these
     * floors are derived rather than picked. <strong>Do not shorten the phases and do not touch the
     * smoothing constant</strong>; the latter is pinned by ADR 0016's client GameTests.
     */
    public static final int ESCAPE_CROUCH_MIN_TICKS = 4;

    /** Shortest launch the reshaping may produce, in ticks. See {@link #ESCAPE_CROUCH_MIN_TICKS}. */
    public static final int ESCAPE_LAUNCH_MIN_TICKS = 5;

    /**
     * Commanded vertical thrust during the crouch, as a fraction of the vertical rail.
     *
     * <p>Negative, so a grounded fly presses into the floor rather than sinking through it, and
     * {@code FlyAnimation.LEG_TUCK_SWING} draws the legs extended and splayed. 0.30 is deep enough
     * to read as a crouch on screen and shallow enough that an airborne fly loses about a tenth of
     * a block before the launch.
     */
    public static final double ESCAPE_CROUCH_VERTICAL = -0.30;

    /** Commanded vertical thrust during the launch, before up-clearance scaling: the whole rail. */
    public static final double ESCAPE_LAUNCH_VERTICAL = 1.0;

    /**
     * Peak commanded forward thrust during the launch, before front-clearance scaling.
     *
     * <p>0.60 rather than 1.0 because the jump's point is to leave, not to travel.
     */
    public static final double ESCAPE_LAUNCH_FORWARD = 0.60;

    /**
     * Ray distance, in blocks, inside which a ray counts as a threat, and the divisor of both
     * clearance scalings.
     *
     * <p>Half the 6.0-block declared ray length ({@code FlySensors.RAY_LENGTH_BLOCKS}), so a ray
     * reading its maximum is unambiguously open. One constant serves both roles deliberately: the
     * distance at which the fly stops treating what is in front of or above it as a wall is the
     * same distance at which it is willing to thrust into it at full declared force.
     */
    public static final double ESCAPE_THREAT_RANGE_BLOCKS = 3.0;

    /**
     * Bound on the commanded yaw torque of the away-turn, as a fraction of the yaw rail.
     *
     * <p>0.6 of {@code MAX_YAW_DELTA_DEGREES_PER_TICK = 6.0} is 3.6 degrees per tick, so crouch and
     * launch together can turn the body far enough to point the jump away from a wall without
     * spinning it.
     */
    public static final double ESCAPE_TURN_TORQUE = 0.6;

    /**
     * Vertical speed, in blocks/tick, at which the brake phase saturates the vertical rail.
     *
     * <p>It equals the actuator's {@code MAX_THRUST_DELTA_BLOCKS_PER_TICK}, which is the speed one
     * tick of full thrust can remove, so the brake is deadbeat: it commands exactly the thrust that
     * would null the body's vertical velocity this tick, clamped to the rail. Restated here rather
     * than imported because {@code BodyPackageBoundaryTest} does not admit
     * {@code PrimitiveActuator} into this package; {@code escapeJumpBrakeReferenceMatchesTheRail}
     * pins the two equal so the restatement cannot drift.
     *
     * <p>This constant is the fix for the defect the review found: with drag 0.98 and no ambient
     * gravity, a body left at the launch's exit velocity coasts roughly {@code v / 0.02} further
     * blocks, which in the demo arena means a floor jump still ends in the ceiling — the exact live
     * failure this program exists to cure.
     */
    public static final double ESCAPE_BRAKE_REFERENCE_SPEED_BLOCKS_PER_TICK = 0.04;

    /** Latched away side: no threat has been seen yet. */
    public static final int SIDE_NONE = 0;

    /** Latched away side: turn toward the body's right, i.e. positive yaw torque (ADR 0015). */
    public static final int SIDE_RIGHT = 1;

    /** Latched away side: turn toward the body's left. */
    public static final int SIDE_LEFT = 2;

    /** Resolution of the two latched clearance scales, as parts per {@code SCALE_UNIT}. */
    public static final int SCALE_UNIT = 1000;

    private static final int SIDE_SHIFT = 0;
    private static final int SIDE_MASK = 0x3;
    private static final int URGENCY_SHIFT = 2;
    private static final int URGENCY_MASK = 0x7F;
    private static final int FORWARD_SHIFT = 9;
    private static final int VERTICAL_SHIFT = 19;
    private static final int SCALE_MASK = 0x3FF;

    @Override
    public BodyProgramId id() {
        return BodyProgramId.ESCAPE_JUMP;
    }

    @Override
    public boolean canEnter(BodyProgramContext context) {
        // Unconditional, deliberately. A reflex that could decline to fire would be a policy
        // decision hidden inside a precondition, and the program that must never be second-guessed
        // is the one the escape readout selects.
        return true;
    }

    /**
     * ESCAPE_JUMP reads {@code urgency} alone. It ignores the signed {@link ProgramParameters}
     * entirely: the away-turn side is latched from the rays at first sighting, and a selector that
     * could steer a reflex mid-flight would be reshaping the profile under the body -- exactly what
     * the one-way latch word exists to prevent. {@code escapeJumpIgnoresTheTurnBias} pins that
     * varying it produces a bit-identical motor sequence.
     */
    @Override
    public BodyProgramStep step(
            BodyProgramState state,
            BodyProgramContext context,
            Modulation modulation,
            ProgramParameters parameters) {
        PhysicalSensorSnapshot snapshot = context.snapshot();
        int tick = Math.min(Math.max(state.ticksInProgram(), 0), ESCAPE_JUMP_TICKS - 1);
        int word = latch(state, snapshot, modulation);
        int crouchTicks = crouchTicks(urgencyPercent(word));
        int launchTicks = launchTicks(urgencyPercent(word));

        double vertical;
        double forward;
        double yaw;
        if (tick < crouchTicks) {
            vertical = ESCAPE_CROUCH_VERTICAL;
            forward = 0.0;
            yaw = awayYawTorque(snapshot, side(word));
        } else if (tick < crouchTicks + launchTicks) {
            vertical = ESCAPE_LAUNCH_VERTICAL * scale(word, VERTICAL_SHIFT);
            forward = ESCAPE_LAUNCH_FORWARD * scale(word, FORWARD_SHIFT);
            yaw = awayYawTorque(snapshot, side(word));
        } else {
            // The brake. Forward eases linearly to exactly zero on the last tick, from the scale
            // latched at launch start rather than from a re-read ray, so "eases to zero" is true
            // even when the wall the jump was aimed away from leaves the forward ray mid-flight.
            // The away-turn is not part of this phase: the body is airborne and already pointed,
            // and holding torque would keep rotating it through the arc it just flew.
            int brakeTicks = ESCAPE_JUMP_TICKS - crouchTicks - launchTicks;
            double remaining = 1.0 - (tick - crouchTicks - launchTicks + 1) / (double) brakeTicks;
            vertical = brakeVertical(snapshot.velocityYBlocksPerTick());
            forward = ESCAPE_LAUNCH_FORWARD * scale(word, FORWARD_SHIFT) * remaining;
            yaw = 0.0;
        }

        return new BodyProgramStep(
                PrimitiveAction.clamped(forward, 0.0, vertical, yaw, 0.0),
                state.advanced(word),
                (float) (tick / (double) ESCAPE_JUMP_TICKS),
                state.ticksInProgram() >= ESCAPE_JUMP_TICKS - 1
                        ? Optional.of(BodyProgramId.CRUISE)
                        : Optional.empty());
    }

    // ---- the latched state word ------------------------------------------------------------

    /**
     * The program's <strong>private state word</strong>, carried in {@code phaseIndex}.
     *
     * <p>{@code BodyProgramState.phaseIndex} is the program's own scratch integer: nothing outside
     * this package reads it, and what the HUD and the renderer draw is
     * {@code BodyProgramStep.presentationPhase}, which this program publishes separately as the
     * fraction of the profile elapsed. The bit layout is:
     *
     * <pre>
     *   bits  0-1   away side          0 none, 1 right, 2 left     latched at first sighting
     *   bits  2-8   urgency            percent, 0..100             latched at entry (tick 0)
     *   bits  9-18  launch forward     parts per SCALE_UNIT        latched at the first launch tick
     *   bits 19-28  launch vertical    parts per SCALE_UNIT        latched at the first launch tick
     * </pre>
     *
     * <p>Every field is a one-way latch, so the word is monotone over a jump and the profile cannot
     * be reshaped under the body once it has started.
     */
    private static int latch(
            BodyProgramState state, PhysicalSensorSnapshot snapshot, Modulation modulation) {
        int tick = state.ticksInProgram();
        int word = tick == 0
                ? urgencyPercent(modulation) << URGENCY_SHIFT
                : state.phaseIndex();
        int crouchTicks = crouchTicks(urgencyPercent(word));
        int launchTicks = launchTicks(urgencyPercent(word));

        if (side(word) == SIDE_NONE && tick < crouchTicks + launchTicks) {
            word |= threatSide(snapshot) << SIDE_SHIFT;
        }
        if (tick == crouchTicks) {
            word |= permille(clearanceScale(snapshot.rayForwardBlocks())) << FORWARD_SHIFT;
            word |= permille(clearanceScale(snapshot.rayUpBlocks())) << VERTICAL_SHIFT;
        }
        return word;
    }

    /** The latched away side of a state word. */
    public static int side(int word) {
        return (word >> SIDE_SHIFT) & SIDE_MASK;
    }

    /** The latched entry urgency of a state word, in percent. */
    public static int urgencyPercent(int word) {
        return Math.min(100, (word >> URGENCY_SHIFT) & URGENCY_MASK);
    }

    private static double scale(int word, int shift) {
        return Math.min(SCALE_UNIT, (word >> shift) & SCALE_MASK) / (double) SCALE_UNIT;
    }

    private static int permille(double scale) {
        return (int) Math.round(scale * SCALE_UNIT);
    }

    // ---- the pure profile functions ---------------------------------------------------------

    /**
     * The entry urgency quantized to percent, which is what the state word carries.
     *
     * <p>One percent is far finer than one tick of any phase — the whole reshaping spans two ticks
     * of crouch and three of launch over the full range — so the quantization is invisible in the
     * profile and exists only to make the latch a small integer.
     */
    public static int urgencyPercent(Modulation modulation) {
        return (int) Math.round(modulation.urgency() * 100.0);
    }

    /** Crouch length at an urgency percent, in ticks. */
    public static int crouchTicks(int urgencyPercent) {
        return reshape(ESCAPE_CROUCH_TICKS, ESCAPE_CROUCH_MIN_TICKS, urgencyPercent);
    }

    /** Launch length at an urgency percent, in ticks. */
    public static int launchTicks(int urgencyPercent) {
        return reshape(ESCAPE_LAUNCH_TICKS, ESCAPE_LAUNCH_MIN_TICKS, urgencyPercent);
    }

    /**
     * Brake length at an urgency percent, in ticks: whatever the fixed profile has left.
     *
     * <p>10 ticks at zero urgency and 15 at full. Urgency buys a faster launch, and it spends the
     * ticks it saves on braking rather than on ending the jump sooner.
     */
    public static int brakeTicks(int urgencyPercent) {
        return ESCAPE_JUMP_TICKS - crouchTicks(urgencyPercent) - launchTicks(urgencyPercent);
    }

    /**
     * Linear reshaping of one phase toward its floor:
     * {@code declared - round(percent/100 * (declared - floor))}. Monotone non-increasing in the
     * percent because {@code Math.round} is, and exact at both ends.
     */
    public static int reshape(int declared, int floor, int urgencyPercent) {
        int bounded = Math.min(100, Math.max(0, urgencyPercent));
        return declared - (int) Math.round(bounded / 100.0 * (declared - floor));
    }

    /**
     * Deadbeat vertical brake: the thrust that would null this vertical velocity in one tick,
     * clamped to the rail by {@code PrimitiveAction.clamped} downstream.
     *
     * <p>Positive velocity gives negative thrust and vice versa, so the brake is symmetric and a
     * body that arrives descending is caught just as a climbing one is.
     */
    public static double brakeVertical(double velocityYBlocksPerTick) {
        // The trailing "+ 0.0" is the whole of IEEE-754 negative zero: -0.0 / x is -0.0, and a
        // stopped body would then publish -0.0 on the vertical channel for every brake tick, which
        // the HUD prints as "V -0.00". Adding zero maps -0.0 to +0.0 and leaves every other value
        // bit-identical. Pinned by escapeJumpBrakeAtRestIsPositiveZero.
        return -velocityYBlocksPerTick / ESCAPE_BRAKE_REFERENCE_SPEED_BLOCKS_PER_TICK + 0.0;
    }

    /**
     * Clearance scale for one ray: {@code clamp(distance / ESCAPE_THREAT_RANGE_BLOCKS, 0, 1)}.
     *
     * <p>Applied to the launch's forward thrust from the forward ray and to its vertical thrust
     * from the up ray, identically, so the jump neither rams a wall it is facing nor drives itself
     * into a ceiling it is under. This is scripted obstacle avoidance and must be described as
     * exactly that: it reads one ray, it has no memory beyond the launch-start latch, and it is
     * one line.
     */
    public static double clearanceScale(double rayBlocks) {
        return Math.max(0.0, Math.min(1.0, rayBlocks / ESCAPE_THREAT_RANGE_BLOCKS));
    }

    /**
     * Which side the body would commit to if it latched on this sample, or {@link #SIDE_NONE}.
     *
     * <p>The rule, over the forward, left, and right rays:
     *
     * <ul>
     *   <li>No ray inside {@link #ESCAPE_THREAT_RANGE_BLOCKS}: {@link #SIDE_NONE}. An open arena
     *       produces no turn at all, which is what makes a turn attributable to a threat.
     *   <li>The <em>front</em> ray is the shortest: commit toward the more open side.
     *   <li>A <em>side</em> ray is the shortest: commit away from that side.
     * </ul>
     *
     * <p>Every tie resolves to the right, deterministically and by construction: the front case
     * wins ties against a side and breaks its own left/right tie rightward, and a left/right tie
     * makes the left ray the threat, which is turned away from rightward. Handedness is a coin flip
     * in an exactly symmetric world; what matters is that it is the same coin flip every time.
     */
    public static int threatSide(PhysicalSensorSnapshot snapshot) {
        double front = snapshot.rayForwardBlocks();
        double left = snapshot.rayLeftBlocks();
        double right = snapshot.rayRightBlocks();
        if (Math.min(front, Math.min(left, right)) >= ESCAPE_THREAT_RANGE_BLOCKS) {
            return SIDE_NONE;
        }
        if (front <= left && front <= right) {
            return right >= left ? SIDE_RIGHT : SIDE_LEFT;
        }
        return left <= right ? SIDE_RIGHT : SIDE_LEFT;
    }

    /**
     * Commanded yaw torque of the away-turn, in {@code [-ESCAPE_TURN_TORQUE, ESCAPE_TURN_TORQUE]}.
     *
     * <p><strong>Sign, from ADR 0015 and nowhere else.</strong> Positive yaw torque increases yaw,
     * and increasing yaw walks the heading SOUTH to WEST to NORTH to EAST — clockwise seen from
     * above, i.e. toward the body's own right. So {@link #SIDE_RIGHT} is a positive torque.
     *
     * <p><strong>The side is latched; only the magnitude is live.</strong> The magnitude ramps with
     * proximity, {@code 1 - shortest / ESCAPE_THREAT_RANGE_BLOCKS}, so a threat at the edge of range
     * produces a nudge, one at contact produces the declared maximum, and a threat that has left
     * every ray produces exactly zero without un-committing the side.
     */
    public static double awayYawTorque(PhysicalSensorSnapshot snapshot, int latchedSide) {
        if (latchedSide != SIDE_RIGHT && latchedSide != SIDE_LEFT) {
            return 0.0;
        }
        double shortest = Math.min(
                snapshot.rayForwardBlocks(),
                Math.min(snapshot.rayLeftBlocks(), snapshot.rayRightBlocks()));
        double proximity = Math.max(0.0, 1.0 - shortest / ESCAPE_THREAT_RANGE_BLOCKS);
        return (latchedSide == SIDE_RIGHT ? 1.0 : -1.0) * ESCAPE_TURN_TORQUE * proximity;
    }
}
