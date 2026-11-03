package dev.neurocraft.fly.body.programs;

import dev.neurocraft.fly.body.BodyProgram;
import dev.neurocraft.fly.body.BodyProgramContext;
import dev.neurocraft.fly.body.BodyProgramContextSampler;
import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.BodyProgramState;
import dev.neurocraft.fly.body.BodyProgramStep;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ProgramParameters;

/**
 * Grounded scripted feeding gesture. It neither consumes an item nor changes any world food,
 * health or energy state; extension/hold/retraction are presentation, not digestion.
 * Requires current complete sturdy contact and the sampler's whitelisted visible food within
 * 0.75 blocks and 60 degrees of forward. It never seeks or aligns to food. Loss of a precondition
 * brakes immediately and requests HOVER with the existing generic exit reason.
 * Settle 10, extend 10, hold 30..10, retract 10 ticks: urgency compresses the hold only.
 * Presentation quarters name those phases. Amplitude is renderer-only; other scalars and signed
 * parameters are ignored, and every motor is the same bounded stationary brake/contact law.
 */
public final class FeedProgram implements BodyProgram {
    public static final int SETTLE_TICKS = 10;
    public static final int EXTEND_TICKS = 10;
    public static final int MAX_HOLD_TICKS = 30;
    public static final int MIN_HOLD_TICKS = 10;
    public static final int RETRACT_TICKS = 10;
    public static final double REACH_BLOCKS = 0.75;
    public static final double FORWARD_CONE_DEGREES = 60.0;
    public static final int SETTLE = 0;
    public static final int EXTEND = 1;
    public static final int HOLD = 2;
    public static final int RETRACT = 3;

    @Override public BodyProgramId id() { return BodyProgramId.FEED; }

    @Override
    public boolean canEnter(BodyProgramContext context) {
        return LandProgram.hasGroundContact(context) && context.food().filter(food ->
                BodyProgramContextSampler.FOOD_ITEM_IDS.contains(food.itemId())
                        && food.distanceBlocks() >= 0.0 && food.distanceBlocks() <= REACH_BLOCKS
                        && Math.abs(food.bearingDegrees()) <= FORWARD_CONE_DEGREES).isPresent();
    }

    @Override
    public BodyProgramStep step(BodyProgramState state, BodyProgramContext context,
            Modulation modulation, ProgramParameters parameters) {
        return GroomProgram.phasedStep(state, context, !canEnter(context),
                new int[] {SETTLE_TICKS, EXTEND_TICKS, holdTicks(modulation.urgency()), RETRACT_TICKS});
    }

    public static int holdTicks(double urgency) {
        return MAX_HOLD_TICKS - (int) Math.floor((MAX_HOLD_TICKS - MIN_HOLD_TICKS) * urgency);
    }
}
