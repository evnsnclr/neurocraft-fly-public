package dev.neurocraft.fly.gametest;

import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ProgramParameters;
import dev.neurocraft.fly.body.ReadoutValues;
import dev.neurocraft.fly.control.ActionFrame;
import dev.neurocraft.fly.control.CommandValidator;
import dev.neurocraft.fly.control.ControlAckFrame;
import dev.neurocraft.fly.control.ControlOperation;
import dev.neurocraft.fly.control.ControlState;
import dev.neurocraft.fly.control.ControllerIdentity;
import dev.neurocraft.fly.control.ProgramActionFrame;
import dev.neurocraft.fly.control.RawPrimitiveAction;
import dev.neurocraft.fly.entity.FlyEntity;
import java.util.Map;
import java.util.OptionalLong;
import java.util.function.LongFunction;

/**
 * The synthetic inputs {@link ReadoutProgramGameTests} drives the readout lane with.
 *
 * <p><strong>Every value here is a literal typed into this file.</strong> No readout below came
 * from the connectome, no modulation came from a mapping, and a frame built here is a test fixture,
 * not a recording. Keeping them in one place is why the pins beside them read as assertions about
 * the authority model rather than as claims about a fly.
 */
final class ReadoutLaneFence {
    /** A plausible cruise selection: mild speed attenuation, a leftward bias, no urgency. */
    static final Modulation CRUISING = new Modulation(0.7, 1.0, 1.0, 0.0);

    static final ProgramParameters BIASED = new ProgramParameters(-0.25);

    static final ReadoutValues READOUT = new ReadoutValues(0.0, -0.25, 0.61);

    /** A stand-in mapping identity. It identifies nothing; it is 64 sevens. */
    static final String POLICY_HASH = "7".repeat(64);

    private ReadoutLaneFence() {}

    static ControllerIdentity identity(FlyEntity fly) {
        return new ControllerIdentity(
                "readout-program-gametest",
                0,
                fly.getStringUUID(),
                "1".repeat(64),
                "2".repeat(64),
                "3".repeat(64),
                "4".repeat(64),
                "5".repeat(64));
    }

    static ProgramActionFrame programFrame(
            FlyEntity fly,
            long targetTick,
            long commandSequence,
            BodyProgramId program,
            Modulation modulation,
            ProgramParameters parameters) {
        return new ProgramActionFrame(
                identity(fly),
                commandSequence,
                commandSequence,
                targetTick - CommandValidator.ACTION_TARGET_OFFSET_TICKS,
                targetTick,
                targetTick,
                commandSequence * 5,
                commandSequence * 5 + 5,
                "6".repeat(64),
                program,
                modulation,
                parameters,
                READOUT,
                POLICY_HASH);
    }

    /**
     * A frame in a <em>streamed</em> session: the caller states the observation the frame answers
     * and the tick it targets, instead of taking both from the command sequence.
     *
     * <p>{@link #programFrame} is written for a single frame in a session that has taken one
     * observation per game tick. A pin that drives many frames against one authoritative tick --
     * the chain pin, which runs its whole drive inside one server tick so the recording bridge is
     * installed for a fraction of a tick rather than for a minute -- needs to say
     * {@code observationTick == targetTick == the tick the server is actually on}.
     */
    static ProgramActionFrame streamedProgramFrame(
            FlyEntity fly,
            long commandSequence,
            long observationSequence,
            long observationTick,
            long targetTick,
            BodyProgramId program,
            Modulation modulation,
            ProgramParameters parameters) {
        return new ProgramActionFrame(
                identity(fly),
                commandSequence,
                observationSequence,
                observationTick,
                targetTick,
                targetTick + 1,
                commandSequence * 5,
                commandSequence * 5 + 5,
                "6".repeat(64),
                program,
                modulation,
                parameters,
                READOUT,
                POLICY_HASH);
    }

    static ActionFrame rawAction(
            ControllerIdentity identity, long targetTick, long commandSequence, double forward) {
        return new ActionFrame(
                identity,
                commandSequence,
                commandSequence,
                targetTick - CommandValidator.ACTION_TARGET_OFFSET_TICKS,
                targetTick,
                targetTick,
                commandSequence * 5,
                commandSequence * 5 + 5,
                "6".repeat(64),
                new RawPrimitiveAction(forward, 0, 0, 0, 0),
                false);
    }

    /** An accepted acknowledgement effective on the tick it was issued. */
    static ControlAckFrame controlAck(
            FlyEntity fly, long controlSequence, long issuedTick, ControlOperation operation) {
        return new ControlAckFrame(
                identity(fly),
                controlSequence,
                issuedTick,
                operation,
                null,
                null,
                true,
                operation == ControlOperation.PAUSE ? ControlState.PAUSED : ControlState.RUNNING,
                issuedTick,
                null);
    }

    static LongFunction<OptionalLong> observationReceipts(Map<Long, Long> receipts) {
        return sequence -> {
            Long tick = receipts.get(sequence);
            return tick == null ? OptionalLong.empty() : OptionalLong.of(tick);
        };
    }
}
