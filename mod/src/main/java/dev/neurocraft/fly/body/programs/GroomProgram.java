package dev.neurocraft.fly.body.programs;

import dev.neurocraft.fly.body.BodyProgram;
import dev.neurocraft.fly.body.BodyProgramContext;
import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.BodyProgramState;
import dev.neurocraft.fly.body.BodyProgramStep;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ProgramParameters;
import dev.neurocraft.fly.control.PrimitiveAction;
import java.util.Optional;

/**
 * Scripted stationary grooming gesture, not anatomy or a neural selection policy.
 * Settle 10 ticks, forelegs 20..10, antenna 20..10, recover 10: urgency compresses only the
 * two active phases. The four presentation quarters name those phases at every duration.
 * Speed, turnRate, signed parameters and amplitude do not change motors or the clock;
 * amplitude belongs to the renderer's gesture extent. No hidden force creates the pose.
 * Airborne grooming brakes all velocity without a bob or altitude target; complete sturdy
 * ground contact uses LAND's small downward contact force. Water/lava refuses or exits HOVER.
 */
public final class GroomProgram implements BodyProgram {
    public static final int SETTLE_TICKS = 10;
    public static final int MAX_STROKE_TICKS = 20;
    public static final int MIN_STROKE_TICKS = 10;
    public static final int RECOVER_TICKS = 10;
    public static final int SETTLE = 0;
    public static final int FORELEGS = 1;
    public static final int ANTENNA = 2;
    public static final int RECOVER = 3;

    @Override public BodyProgramId id() { return BodyProgramId.GROOM; }

    @Override
    public boolean canEnter(BodyProgramContext context) {
        return !context.snapshot().inWater() && !context.snapshot().inLava();
    }

    @Override
    public BodyProgramStep step(BodyProgramState state, BodyProgramContext context,
            Modulation modulation, ProgramParameters parameters) {
        int stroke = strokeTicks(modulation.urgency());
        return phasedStep(state, context, !canEnter(context),
                new int[] {SETTLE_TICKS, stroke, stroke, RECOVER_TICKS});
    }

    public static int strokeTicks(double urgency) {
        return MAX_STROKE_TICKS - (int) Math.floor(
                (MAX_STROKE_TICKS - MIN_STROKE_TICKS) * urgency);
    }

    /** Zero-rate feedback never increases the braking envelope when modulation is attenuated. */
    static PrimitiveAction park(BodyProgramContext context) {
        double vertical = LandProgram.hasGroundContact(context)
                ? -LandProgram.GROUND_CONTACT_THRUST
                : LandProgram.verticalThrust(context.snapshot().velocityYBlocksPerTick(), 0.0);
        return LandProgram.brakeHorizontal(context.snapshot(), vertical);
    }

    /** Four explicit presentation quarters; local phase never reaches the next quarter early. */
    static BodyProgramStep phasedStep(BodyProgramState state, BodyProgramContext context,
            boolean refused, int[] durations) {
        int tick = state.ticksInProgram();
        int total = durations[0] + durations[1] + durations[2] + durations[3];
        if (refused || tick >= total) {
            return new BodyProgramStep(park(context), state.advanced(0), 0.0F,
                    Optional.of(BodyProgramId.HOVER));
        }
        int phase = 0;
        int localTick = tick;
        while (localTick >= durations[phase]) {
            localTick -= durations[phase++];
        }
        float presentation = (float) ((phase + localTick / (double) durations[phase]) / 4.0);
        int nextPhase = localTick + 1 == durations[phase] ? Math.min(3, phase + 1) : phase;
        return new BodyProgramStep(park(context), state.advanced(nextPhase), presentation,
                Optional.empty());
    }
}
