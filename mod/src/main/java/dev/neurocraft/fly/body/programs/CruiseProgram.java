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
 * CRUISE: self-sufficient scripted locomotion (ADR 0020, programs table entry 2).
 *
 * <p><strong>Every constant below is a declared engineering constant.</strong> None is derived from
 * the connectome, from fly anatomy, or from any measurement of fly behaviour, and a running CRUISE
 * is never evidence that the connectome did anything. Several are derived from the Minecraft body's
 * own physics — the actuator rails and the two drag coefficients — and where they are, the
 * arithmetic is written out so it can be checked and so it moves correctly if a rail moves.
 *
 * <p><strong>Provenance of the numbers in this file.</strong> Three kinds appear, and they are
 * labelled every time: <em>arithmetic</em> on declared constants, which anyone can redo;
 * <em>in-world</em>, meaning a value a GameTest in this repository produced and which a reviewer
 * can reproduce by running the lane; and <em>scratch simulation</em>, meaning a value from an
 * uncommitted Python replica of the actuator and drag that is <strong>not reproducible from this
 * repository</strong> and is reported only because it is what a choice was made against. A scratch
 * number is never presented as a measurement of this code.
 *
 * <p><strong>Why this program exists.</strong> The first live session that survived under the
 * current stack (2026-09-02) recorded the neural lane climbing into the glass ceiling 62 s after
 * spawn and then spending 83 s wedged in a top corner, pushing forward 0.25 and vertical 0.36 with
 * a yaw torque of −0.003. CRUISE is the body-side answer to exactly that: it holds a vertical
 * clearance instead of climbing, it turns away from what its rays see instead of pushing into it,
 * and it never sits still. It is scripted presentation and engineering; it fixes what the body
 * does, not what the readout asks for.
 *
 * <p><strong>Why it holds a floor band.</strong> The program lane's first live session (2026-09-04,
 * lane L, session {@code interactive-20260904T155926Z-25199c93}) showed the other half of the
 * problem. Every ESCAPE_JUMP banks about a block of altitude — launch vertical 1.0 scaled by up
 * clearance, then the deadbeat brake nulling the vertical velocity wherever the body happens to be
 * — and a CRUISE that flew level wherever it found itself kept every block: y 67.0 at spawn, 68.18
 * after the first jump, 84.7 after 19, then 85.4–85.8 under the glass ceiling for the last three
 * minutes, out of the operator's reach (R-31). The rays clamp at 6.0, so a floor more than six
 * blocks down was invisible and nothing ever commanded a descent. CRUISE now holds a floor band,
 * {@link #CRUISE_MIN_FLOOR_CLEARANCE_BLOCKS} to {@link #CRUISE_MAX_FLOOR_CLEARANCE_BLOCKS} of
 * floor clearance, and descends gently above it, so the altitude a jump banks is given back.
 * ESCAPE_JUMP is unchanged. This too is engineering, not anatomy.
 *
 * <p><strong>Three channels, one shared sign.</strong> Vertical thrust is a rate command derived
 * from the two vertical clearances and the floor band — a climb under the minimum clearance, level
 * flight inside the band, a gentle descent above it; forward thrust and yaw torque come from the
 * forward and side clearances; a deterministic wander adds a slow heading drift when nothing is
 * near; and the
 * selector's one signed input, {@code turnBias}, adds a steady turn beside that wander. Lateral
 * thrust and pitch torque are exactly zero: a coordinated bank is BANK's job and carries its own
 * handedness test, and nothing in this project has ever pitched the body.
 *
 * <p><strong>Handedness is not asserted from prose.</strong> {@code BodyFrameMath} puts the body's
 * right at {@code (-cos, -sin)} and positive yaw torque walks the heading SOUTH → WEST → NORTH →
 * EAST, i.e. toward the body's right (ADR 0015). {@code FlySensors} casts the right ray along that
 * same vector and the left ray along its reverse. So "turn toward the more open side" is
 * {@code sign(rayRight − rayLeft)}, and it is pinned in-world by
 * {@code cruiseTurnsAwayFromAWallInsteadOfPushingIntoIt} and {@code cruiseLeavesACorner} measuring
 * where the body actually goes, never by asserting the sign of a variable.
 *
 * <p>A positive {@code turnBias} is declared to mean "toward the body's right" (stage-4b design
 * §3.1, D-077), and it is held to that declaration the same way: the sign is never checked against
 * a variable but against the world-space path the body flies, with "the body's right" derived from
 * the engine's own {@code Direction.getClockWise()} table rather than restated here
 * ({@code cruiseTurnBiasCurvesTheDeclaredWay} for this program, and
 * {@code readoutTurnBiasSteersTheBodyThroughTheRealValidator} for the whole chain, from an admitted
 * {@code program_action} frame carrying {@code turn_bias} to the path the body flew).
 *
 * <p>CRUISE reads {@code speed} and {@code turnRate}. It ignores {@code amplitude} and
 * {@code urgency} structurally, and {@code cruiseAmplitudeAndUrgencyAreStructurallyIgnored} pins
 * that varying either produces a bit-identical motor sequence. <strong>That pairing is a declared
 * departure from ADR 0020 §4's modulation table</strong>, which gives CRUISE {@code speed} alone
 * and lists {@code turnRate} for BANK, WALK, FLEE and APPROACH; ADRs are append-only, so the
 * departure is recorded in the CHANGELOG rather than by editing the table. CRUISE has no duration
 * cap and never self-exits: like HOVER it runs until it is released or preempted.
 *
 * <p>It also reads the one signed program input, {@link ProgramParameters#turnBias}, through
 * {@link #CRUISE_BIAS_TORQUE}. That is a second declared departure from ADR 0020 §4, which
 * describes modulation as four attenuating scalars and nothing else; a signed direction request is
 * a different kind of object and travels beside {@code Modulation} rather than inside it (stage-4b
 * design §3.4). It is recorded in the CHANGELOG for the same append-only reason.
 */
public final class CruiseProgram implements BodyProgram {
    /**
     * Forward thrust at full {@code speed}, as a fraction of the horizontal rail.
     *
     * <p>Arithmetic: terminal horizontal speed is {@code thrust * 0.04 * 0.91 / (1 − 0.91)}, so
     * 0.55 settles at about 0.2224 blocks/tick — 4.4 blocks/s, a brisk but readable cruise, and
     * well under the actuator's 0.6 blocks/tick speed cap.
     */
    public static final double CRUISE_THRUST = 0.55;

    /** Side clearance, in blocks, below which a lateral ray adds avoidance yaw. */
    public static final double CRUISE_SIDE_CLEARANCE_BLOCKS = 1.5;

    /**
     * Forward clearance, in blocks, at which avoidance reaches full authority and forward thrust
     * its floor. Bound to {@link #CRUISE_SIDE_CLEARANCE_BLOCKS} rather than restated, so the
     * forward ray hands over to the side rays at exactly the distance they begin defending and one
     * edit moves both.
     */
    public static final double CRUISE_HARD_DISTANCE_BLOCKS = CRUISE_SIDE_CLEARANCE_BLOCKS;

    /**
     * Forward clearance, in blocks, at which slowing and turning begin.
     *
     * <p>Arithmetic: at {@link #CRUISE_AVOID_TORQUE} the body turns 5.4 degrees per tick, so at the
     * terminal cruise speed above its turn radius is {@code 0.2224 / radians(5.4)} = 2.36 blocks
     * and a 90-degree turn takes 16.7 ticks. 4.0 blocks is 1.7 turn radii: enough room to complete
     * the turn from full speed.
     *
     * <p>3.0 was tried first and rejected. That comparison was made in an uncommitted scratch
     * simulation and is <strong>not reproducible from this repository</strong>; what is
     * reproducible is that at 4.0 the in-world pin
     * {@code cruiseTurnsAwayFromAWallInsteadOfPushingIntoIt} holds, and its own Javadoc records the
     * contact-tick distribution the value actually produces.
     */
    public static final double CRUISE_SLOW_DISTANCE_BLOCKS = 4.0;

    /**
     * Fraction of {@link #CRUISE_THRUST} the body keeps at {@link #CRUISE_HARD_DISTANCE_BLOCKS}.
     *
     * <p>Deliberately nonzero. A body that stops dead in front of an obstacle is the wedged fly this
     * program exists to replace; a body that keeps a tenth of its thrust while pivoting leaves on
     * its own. Arithmetic: at 0.10 the terminal speed against a near wall is about 0.022
     * blocks/tick, slow enough that the turn wins.
     */
    public static final double CRUISE_SLOW_FLOOR_FRACTION = 0.10;

    /**
     * Ceiling clearance, in blocks. The historical public name is retained for compatibility;
     * the floor now uses {@link #CRUISE_MIN_FLOOR_CLEARANCE_BLOCKS} independently.
     */
    public static final double CRUISE_MIN_VERTICAL_CLEARANCE_BLOCKS = 2.5;

    /**
     * Bottom of the declared 1.0–1.5-block floor band, measured from the ray origin at the body
     * centre. This brings settled CRUISE within the height of a grounded player. It is a scripted
     * presentation choice, not an anatomical inference. The ceiling clearance remains 2.5.
     */
    public static final double CRUISE_MIN_FLOOR_CLEARANCE_BLOCKS = 1.0;

    /**
     * Top of the floor band. Above this clearance, including a clamped 6.0-block floor ray,
     * CRUISE commands the existing gentle descent. The half-block band leaves room for its
     * approximately 0.3-block stopping coast; real body tests check the resulting clearance.
     *
     * <p>The previous 2.5–4.0 band measured settled clearances of 3.7015 from the ceiling,
     * 3.7268 from mid-air and 3.7348 after three jumps. Those are historical measurements of the
     * old band, which placed horizontal sensing rays above a standing player's bounding box.
     * ESCAPE_JUMP, the ceiling rule, rates, thrust and horizontal channels are unchanged.
     */
    public static final double CRUISE_MAX_FLOOR_CLEARANCE_BLOCKS = 1.5;

    /**
     * Vertical rate, in blocks/tick, commanded above the floor band (1.2 blocks/s): half of
     * {@link #CRUISE_CLIMB_RATE_BLOCKS_PER_TICK}, a ratio {@code CruiseFloorBandTest} pins so the
     * prose cannot drift from the number.
     *
     * <p><strong>Gentle on purpose.</strong> The descent is a return, not a reflex: it should read
     * on screen as the fly drifting back down after a jump, slower than the climb the same loop
     * commands away from a floor. It goes through the same rate controller
     * ({@link #CRUISE_ALTITUDE_THRUST}), so the same arithmetic applies. Arithmetic on the declared
     * constants — gain 3.75 of thrust per block/tick of rate error, 0.04 blocks/tick² per unit
     * thrust, drag 0.98: the loop is proportional, so the settled descent is
     * {@code 0.15 / (0.15 + 0.02 / 0.98)} = 0.880 of the command, 0.0528 blocks/tick (1.06
     * blocks/s), and once the command drops to zero at the band top the vertical velocity decays by
     * {@code 0.98 × 0.85} = 0.833 per tick, so the body travels a further
     * {@code 0.0528 / (1 − 0.833)} ≈ 0.32 blocks into the band and stops. Historically, with the 2.5–4.0 band, measured by
     * {@code cruiseDescendsFromTheCeilingIntoTheFloorBand} in the launcher's arena height: entered
     * 2.39 under the lid with the floor ray at its clamp (live y 85.5), the floor ray is inside the
     * band on tick 332 — 17.6 blocks in 332 ticks, 0.053 blocks/tick over the run against the
     * 0.0528 above — the lowest reading over 600 ticks is 3.7015, the body settles at 3.7015 with
     * zero excursion over the last 100 ticks, and no tick reports a real vertical contact.
     *
     * <p><strong>A constant above the band, not a ramp toward its top, deliberately.</strong> A
     * command that faded to zero at the band top would approach the band's top edge from above without ever
     * being inside the band, and the point is to be inside it. The command therefore steps from
     * {@code -0.06} to zero at the band top; the rate loop turns that step into the exponential stop
     * above, and the body's position is continuous throughout.
     *
     * <p><strong>Where it does and does not apply.</strong> It applies whenever the floor ray reads
     * above the band top, the clamped 6.0 of open air included, and it adds to — never opposes —
     * the ceiling rule, which pushes the same way. It never coexists with the climb rule: the climb
     * is live only under 1.0 and the descent only over 1.5, so "the minimum-clearance climb always
     * wins over the descent" is structural rather than tuned, and the ceiling contribution stays independent of the floor band.
     */
    public static final double CRUISE_DESCENT_RATE_BLOCKS_PER_TICK = 0.06;

    /**
     * Vertical rate, in blocks/tick, commanded at full encroachment (2.4 blocks/s).
     *
     * <p>It is a <em>rate</em> and not a thrust on purpose. The vertical axis is a first-order
     * filter with drag 0.98, so it coasts with a time constant of about 49 ticks; a
     * clearance-proportional <em>acceleration</em> would be fighting that coast rather than using
     * it, and would have to be tuned against it. Commanding a rate puts the drag inside the loop.
     * No numeric damping figure is claimed for that rejected alternative: none was ever built.
     */
    public static final double CRUISE_CLIMB_RATE_BLOCKS_PER_TICK = 0.12;

    /**
     * Maximum vertical thrust of the clearance term, as a fraction of the vertical rail.
     *
     * <p><strong>The rate loop's gain is derived from this and not chosen separately:</strong> full
     * vertical thrust is commanded when the vertical rate is off target by the full
     * {@link #CRUISE_CLIMB_RATE_BLOCKS_PER_TICK}, i.e. a gain of {@code 0.45 / 0.12} = 3.75 of
     * thrust per block/tick of rate error.
     *
     * <p>The old symmetric 2.5-block clearance gave the linearized single-surface equation
     * {@code ÿ = −0.0072·e − 0.1702·ė}, natural frequency 0.0849 and damping ratio 1.003.
     * That historical calculation still describes the unchanged ceiling ramp. The new 1.0-block
     * floor ramp is steeper and does not inherit the old critical-damping/no-overshoot claim.
     * High and low entry, a raised floor, narrow-gap and repeated escape tests measure its
     * settling and contact directly with the actual body actuator.
     *
     * <p>The floor band's descent runs through this same loop at a constant command of
     * {@link #CRUISE_DESCENT_RATE_BLOCKS_PER_TICK}, so it retains this rate-loop gain and
     * needs no constant of its own beyond its rate: one more target for a loop that already
     * existed, not a second controller.
     */
    public static final double CRUISE_ALTITUDE_THRUST = 0.45;

    /**
     * Deadband, in blocks, on the left-minus-right clearance comparison.
     *
     * <p><strong>This is the anti-dither device and it is load-bearing.</strong> The turn side is
     * "toward the more open side", which is undefined in a symmetric corner — exactly the situation
     * that wedged the fly live. Inside this deadband the side instead comes from the sign of the
     * wander, which varies on a 137-tick period and therefore holds one sign for tens of ticks:
     * long enough to turn 180 degrees. Without it, a near-tie flips the commanded side tick by tick
     * and the body dithers in place at zero net yaw.
     */
    public static final double CRUISE_SIDE_TIE_BLOCKS = 0.25;

    /**
     * Maximum avoidance yaw torque, as a fraction of the yaw rail: 5.4 of the actuator's 6.0
     * degrees per tick.
     *
     * <p><strong>Avoidance is never attenuated by modulation.</strong> {@code turnRate} scales the
     * wander and nothing else, so no selector — operator today, labeled readout in stage 4 — can
     * turn the obstacle response down. A fly that can be modulated into a wall is a defect.
     */
    public static final double CRUISE_AVOID_TORQUE = 0.9;

    /**
     * Maximum wander yaw torque, as a fraction of the yaw rail: 1.5 degrees per tick.
     *
     * <p>Small on purpose. Its job is to keep a cruising fly from flying in a straight line forever,
     * not to steer it. {@code cruiseWandersInOpenSpaceDeterministically} pins the resulting
     * open-air heading range against a declared floor of 45 degrees over 300 ticks.
     */
    public static final double CRUISE_WANDER_TORQUE = 0.25;

    /**
     * Maximum turn-bias yaw torque, as a fraction of the yaw rail: 2.1 of the actuator's 6.0
     * degrees per tick. <strong>A declared engineering constant, like every other number here.</strong>
     * It is the size of the steering authority a selector's signed request is given; it is not
     * derived from the connectome, and a curving fly is not evidence that one is steering it.
     *
     * <p>Arithmetic, on the declared constants: at full bias in open air the body turns 2.1
     * degrees per tick, so at the terminal cruise speed of 0.2224 blocks/tick its turn circle has a
     * radius of {@code 0.2224 / radians(2.1)} = 6.07 blocks and closes in {@code 360 / 2.1} = 171
     * ticks (8.6 s). That is a visible arc over a demo-length window and a slower turn than the
     * 2.36-block avoidance circle, which is the intended ordering: steering is a suggestion, and
     * avoidance is not.
     *
     * <p><strong>It sits inside the wander's gate, not beside it.</strong> The bias is added to the
     * wander term and the sum is multiplied by {@code (1 - avoid)}, exactly as {@code turnRate}'s
     * wander is, so the total stays bounded by {@link #CRUISE_AVOID_TORQUE}:
     * {@code 0.9*a + (0.25 + 0.35)*(1 - a) <= 0.9} for every {@code a} in {@code [0, 1]}, because
     * 0.60 is below 0.90. At full avoidance authority — a contact tick, or an obstruction at the
     * hard distance — the bias contributes <em>exactly zero</em>.
     *
     * <p><strong>What that does and does not guarantee, stated rather than implied.</strong>
     * Between those extremes the two terms compose: a bias pointed into the obstruction reduces the
     * avoidance turn, and the bias term is the larger one until {@code 0.9*a = 0.35*(1 - a)}, i.e.
     * {@code a = 0.28} — a side clearance of {@code 1.5 * 0.72} = 1.08 blocks, or a forward
     * clearance of {@code 4.0 - 0.28 * 2.5} = 3.30 blocks. Inside that the avoidance term wins and
     * keeps winning, which is why a full bias aimed at a wall closes to a standoff and turns away
     * rather than arriving; {@code cruiseTurnBiasNeverOverridesAvoidance} measures where the body
     * actually goes. The wander already had the same property at a smaller amplitude (its crossover
     * is {@code a = 0.217}); this constant makes the same trade with 1.4x the authority.
     *
     * <p><strong>The visible consequence, named so nobody has to discover it in a demo.</strong> A
     * bias held toward a wall does not end in contact and does not end in a clean turn away: it
     * settles into <em>wall-following</em> at roughly the crossover clearance, where the two terms
     * balance. An independent review probe of the wall scenario over 300 ticks put a biased body in
     * the 1.0-to-1.5-block band for 62 of those ticks against 42 for an unbiased one; <em>that is
     * the reviewer's measurement, recorded as reported rather than as this lane's own, and it is
     * one run of one scenario in an 8-block arena, so it sizes the effect rather than establishing
     * it.</em> It is a body-program behaviour, not anatomy, and a fly seen skirting a wall is not
     * evidence that anything decided to.
     *
     * <p><strong>It is deliberately not scaled by {@code turnRate}.</strong> {@code turnRate}
     * attenuates the wander and only the wander. A selector that could cancel its own steering
     * request by turning down an unrelated attenuator would make the two inputs interact in a way
     * neither declares, and the bias is a direction rather than an amount: {@code |turnBias| <= 1}
     * already multiplies this maximum.
     */
    public static final double CRUISE_BIAS_TORQUE = 0.35;

    /** First wander period, in ticks (6.85 s). Prime, and coprime with the second. */
    public static final int CRUISE_WANDER_PERIOD_A_TICKS = 137;

    /**
     * Second wander period, in ticks (15.55 s). Also prime, so the two-sinusoid sum has period
     * 137 × 311 = 42,607 ticks (about 35 minutes) and reads as unrepeating over any demo.
     */
    public static final int CRUISE_WANDER_PERIOD_B_TICKS = 311;

    /**
     * Phase offset per seed unit, in turns, applied to the first sinusoid: the reciprocal of the
     * golden ratio. An irrational offset means two entries a few ticks apart get unrelated wanders
     * under the ordinary entry-tick seed, while each single entry stays exactly reproducible.
     */
    public static final double CRUISE_WANDER_SEED_TURNS_A = 0.6180339887498949;

    /** Phase offset per seed unit for the second sinusoid: {@code sqrt(2) − 1}. */
    public static final double CRUISE_WANDER_SEED_TURNS_B = 0.41421356237309515;

    /**
     * Modulus applied to the raw seed before it becomes a phase; the ordinary seed is the entry
     * tick, so this corresponds to one hour at 20 Hz. Explicit trial seeds use the same reduction.
     *
     * <p>Purely numerical hygiene: a game tick can be in the billions, and multiplying that by an
     * irrational turn count would leave the fractional part — the only part that matters — with
     * almost no resolution left. Reducing the seed first keeps about 1e-11 of a turn. It costs a
     * repeat every hour of world time, which no session reaches.
     */
    public static final int CRUISE_WANDER_SEED_MODULUS_TICKS = 72000;

    /**
     * Period, in ticks, of the published presentation phase (one second at 20 Hz).
     *
     * <p><strong>What it encodes: elapsed time inside a one-second cycle, and nothing else.</strong>
     * CRUISE's own visible motion comes from the ADR 0016 motor mapping — thrust drives the wing
     * beat and the abdomen, yaw drives the render bank — and the client's CRUISE leg-tuck overlay
     * keys on the published program <em>id</em>, not on this phase. It is published so the HUD and
     * a later overlay have a monotone, deterministic clock that says the program is running.
     */
    public static final int CRUISE_PHASE_PERIOD_TICKS = 20;

    @Override
    public BodyProgramId id() {
        return BodyProgramId.CRUISE;
    }

    @Override
    public boolean canEnter(BodyProgramContext context) {
        // Unconditional. CRUISE is intended to become the demo's default locomotion, and a
        // default that can refuse would need a fallback of its own.
        return true;
    }

    @Override
    public BodyProgramStep step(
            BodyProgramState state,
            BodyProgramContext context,
            Modulation modulation,
            ProgramParameters parameters) {
        PhysicalSensorSnapshot snapshot = context.snapshot();
        int tick = state.ticksInProgram();
        double wander = wander(state.wanderSeed(), tick);
        double turnSign = turnSign(snapshot, wander);

        double approach = approach(snapshot.rayForwardBlocks());
        double sideThreat = Math.max(
                encroachment(snapshot.rayLeftBlocks(), CRUISE_SIDE_CLEARANCE_BLOCKS),
                encroachment(snapshot.rayRightBlocks(), CRUISE_SIDE_CLEARANCE_BLOCKS));
        // A contact tick is full authority by declaration: the rays have already been overruled
        // by the world, so there is nothing left to be proportional to.
        double avoid = snapshot.horizontalCollision() ? 1.0 : Math.max(approach, sideThreat);

        // The wander and the selector's signed bias share one gate: both yield to avoidance rather
        // than being added on top of it, so the total yaw is bounded by CRUISE_AVOID_TORQUE and
        // neither can steer the body into the wall the avoidance term is turning away from at the
        // authority where that matters. CRUISE_BIAS_TORQUE states exactly what "at the authority
        // where that matters" means, including where the two terms cross over.
        double yawTorque = CRUISE_AVOID_TORQUE * avoid * turnSign
                + (CRUISE_WANDER_TORQUE * modulation.turnRate() * wander
                                + CRUISE_BIAS_TORQUE * parameters.turnBias())
                        * (1.0 - avoid);
        double forward = snapshot.horizontalCollision()
                ? 0.0
                : CRUISE_THRUST
                        * modulation.speed()
                        * (1.0 - (1.0 - CRUISE_SLOW_FLOOR_FRACTION) * approach);

        int phaseIndex = Math.floorMod(tick + 1, CRUISE_PHASE_PERIOD_TICKS);
        float presentationPhase = (float)
                (Math.floorMod(tick, CRUISE_PHASE_PERIOD_TICKS)
                        / (double) CRUISE_PHASE_PERIOD_TICKS);
        return new BodyProgramStep(
                PrimitiveAction.clamped(forward, 0.0, verticalThrust(snapshot), yawTorque, 0.0),
                state.advanced(phaseIndex),
                presentationPhase,
                Optional.empty());
    }

    /**
     * The vertical channel: a rate command from the two vertical clearances and the floor band,
     * through one proportional rate loop.
     *
     * <p>The commanded rate is {@link #targetVerticalRate}. <strong>The rate error term stays
     * active inside the band, and that is a deliberate deviation from "zero vertical thrust
     * whenever both clearances are at or above the minimum".</strong> Holding zero vertical rate
     * <em>is</em> level flight, which is the thing the first live failure could not do, and it also
     * brakes a body that arrives carrying vertical velocity from somewhere else — the real case on
     * release from the neural lane, and at every ESCAPE_JUMP hand-over.
     *
     * <p><strong>Historical fast-release limits, measured with the old 2.5–4.0 band.</strong>
     * {@code cruiseBrakesADescentEnteredFromTheNeuralLane} drives the case this deviation was
     * argued from: entering at 0.4 blocks/tick downward with 4.0 blocks of floor clearance, the
     * body's lowest reading is 0.657 blocks and it never touches. <em>That is the good end of a
     * narrow window, and the window is:</em> in-world over a 300-tick drive, entering at 0.4
     * blocks/tick the body stays off the floor from 4.0 blocks of clearance (lowest 0.657) and from
     * 3.5 (lowest 0.158), but touches down from 3.25, 3.0 and 2.5; and entering at the actuator's
     * 0.6 blocks/tick speed cap it touches down from every entry clearance tried, including 6.0.
     * <strong>So this term does not rescue a fast release, and it must not be described as
     * doing so.</strong> Those probes entered inside the old band or at its top edge. The current lower floor
     * band must be measured again; none of these old values establishes a new rescue envelope.
     *
     * <p>The gated alternative — vertical exactly zero whenever both clearances are at or above the
     * minimum — was compared against this one only in an uncommitted scratch simulation and that
     * comparison is <strong>not reproducible from this repository</strong>. What is reproducible is
     * the historical scenario above, retained as a regression test.
     *
     * <p>Floor and ceiling encroachments can oppose each other in a narrow gap. The actual
     * gap tests report both clearances and contact; the prior 6-block-gap outcomes of 3.4227 and
     * 2.5000 floor clearance belong to the old 2.5–4.0 band, not this lower band.
     *
     * <p>Public and static because it is a pure function of the sample. The unit tests pin
     * the band, rates, independent ceiling rule and unchanged rate-to-thrust conversion.
     */
    public static double verticalThrust(PhysicalSensorSnapshot snapshot) {
        double rateError = targetVerticalRate(snapshot.rayDownBlocks(), snapshot.rayUpBlocks())
                - snapshot.velocityYBlocksPerTick();
        return clampSymmetric(
                CRUISE_ALTITUDE_THRUST * rateError / CRUISE_CLIMB_RATE_BLOCKS_PER_TICK,
                CRUISE_ALTITUDE_THRUST);
    }

    /**
     * The commanded vertical rate, in blocks/tick, from the floor and ceiling clearances.
     *
     * <p>Three rules on disjoint ranges of the floor ray, and one on the ceiling ray:
     *
     * <ul>
     *   <li>floor ray under {@link #CRUISE_MIN_FLOOR_CLEARANCE_BLOCKS}: climb, proportional to
     *       the encroachment, up to {@link #CRUISE_CLIMB_RATE_BLOCKS_PER_TICK};
     *   <li>floor ray inside the band, both edges included: nothing from the floor;
     *   <li>floor ray above {@link #CRUISE_MAX_FLOOR_CLEARANCE_BLOCKS}, the clamped 6.0 included:
     *       descend at {@link #CRUISE_DESCENT_RATE_BLOCKS_PER_TICK};
     *   <li>ceiling ray under the minimum clearance: push down, proportional, up to the climb rate,
     *       added to whichever floor rule is live.
     * </ul>
     *
     * <p>The floor climb and descent have disjoint ranges. The independent ceiling rule and
     * rate loop retain their previous constants; only the two floor boundaries change.
     */
    public static double targetVerticalRate(double rayDownBlocks, double rayUpBlocks) {
        double floorPush = encroachment(rayDownBlocks, CRUISE_MIN_FLOOR_CLEARANCE_BLOCKS);
        double ceilingPush = encroachment(rayUpBlocks, CRUISE_MIN_VERTICAL_CLEARANCE_BLOCKS);
        double targetRate = CRUISE_CLIMB_RATE_BLOCKS_PER_TICK * (floorPush - ceilingPush);
        if (rayDownBlocks > CRUISE_MAX_FLOOR_CLEARANCE_BLOCKS) {
            targetRate -= CRUISE_DESCENT_RATE_BLOCKS_PER_TICK;
        }
        return targetRate;
    }

    /**
     * The heading wander, in {@code [-1, 1]}: two sinusoids on incommensurate periods, phased from
     * the state's declared wander seed (the entry tick during ordinary execution).
     *
     * <p><strong>It is deterministic and periodic, not random.</strong> There is no random source in
     * this package — {@code BodyPackageBoundaryTest} pins the absence of every one — and the value
     * here is a pure function of {@code (wanderSeed, ticksInProgram)}. The ordinary seed remains
     * {@code enteredGameTick}; a matched trial can hold the seed fixed across different real entry
     * ticks without changing any clock. With identical other inputs, equal seeds and owned ticks
     * therefore produce bit-identical motor sequences, and
     * {@code cruiseWandersInOpenSpaceDeterministically} pins that.
     *
     * <p>{@link StrictMath#sin} rather than {@link Math#sin}: the JLS permits {@code Math.sin} to
     * differ by up to 1 ulp between implementations, and a bit-identity claim that only holds on
     * one JDK build is not the claim this program makes. {@code StrictMath} is reproducible by
     * specification.
     */
    private static double wander(long wanderSeed, int ticksInProgram) {
        int seed = Math.floorMod(wanderSeed, CRUISE_WANDER_SEED_MODULUS_TICKS);
        double turnsA = ticksInProgram / (double) CRUISE_WANDER_PERIOD_A_TICKS
                + seed * CRUISE_WANDER_SEED_TURNS_A;
        double turnsB = ticksInProgram / (double) CRUISE_WANDER_PERIOD_B_TICKS
                + seed * CRUISE_WANDER_SEED_TURNS_B;
        return 0.5
                * (StrictMath.sin(2.0 * Math.PI * turnsA)
                        + StrictMath.sin(2.0 * Math.PI * turnsB));
    }

    /**
     * Which way the body turns when it has to: toward the more open side, with the wander's sign as
     * the tie-break inside {@link #CRUISE_SIDE_TIE_BLOCKS}.
     *
     * <p>The wander used here is the <em>unmodulated</em> one, so the tie-break survives
     * {@code turnRate = 0} and the commanded side is the same at every modulation.
     */
    private static double turnSign(PhysicalSensorSnapshot snapshot, double wander) {
        double difference = snapshot.rayRightBlocks() - snapshot.rayLeftBlocks();
        if (difference > CRUISE_SIDE_TIE_BLOCKS) {
            return 1.0;
        }
        if (difference < -CRUISE_SIDE_TIE_BLOCKS) {
            return -1.0;
        }
        return wander >= 0.0 ? 1.0 : -1.0;
    }

    /** Forward approach fraction: 0 at the slow distance, 1 at the hard distance and inside it. */
    private static double approach(double forwardClearanceBlocks) {
        return clampUnitInterval(
                (CRUISE_SLOW_DISTANCE_BLOCKS - forwardClearanceBlocks)
                        / (CRUISE_SLOW_DISTANCE_BLOCKS - CRUISE_HARD_DISTANCE_BLOCKS));
    }

    /** How far inside a declared clearance an obstruction is, in {@code [0, 1]}. */
    private static double encroachment(double clearanceBlocks, double declaredBlocks) {
        return clampUnitInterval((declaredBlocks - clearanceBlocks) / declaredBlocks);
    }

    private static double clampUnitInterval(double value) {
        return Math.max(0.0, Math.min(1.0, value));
    }

    private static double clampSymmetric(double value, double bound) {
        return Math.max(-bound, Math.min(bound, value));
    }
}
