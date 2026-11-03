package dev.neurocraft.fly.control;

import java.util.ArrayDeque;
import java.util.Optional;

/**
 * A synchronized bounded handoff between the socket callback and Minecraft server thread.
 *
 * <p>It holds either driving frame kind. The window is transport plumbing and applies no policy;
 * refusing a session that mixes the two kinds is {@code CommandValidator}'s job, on the server
 * thread, where the session's latched lane lives.
 *
 * <p><strong>The capacity is derived from the longest window in which the server thread sends
 * observations but never polls, and it counts arrivals rather than ticks.</strong> That window is
 * a staged RESET: {@code control()} offers the observation and then returns at
 * {@code RESET_SLOT.blocksActuation} without polling, while the service keeps answering one frame
 * per observation.
 *
 * <p>An operator RESET is issued at tick {@code I} after that tick's entity tick has already
 * polled, so the last poll before the block is at {@code I}; the block covers {@code I+1} through
 * {@code E-1} and the poll at the accepted effective tick {@code E} — at worst {@code I+3}, one
 * action budget after the observation at {@code I+1} that the effect materialized against — sees
 * everything that landed since. Two things can land there: the frame that was already in flight
 * at the poll at {@code I} (the answer to observation {@code I-1}, if its latency exceeds a tick
 * — and the lane's measured latencies are 43–75 ms against a 50 ms tick, so it routinely does),
 * plus one answer for each of the observations at {@code I}, {@code I+1} and {@code I+2}. Four,
 * which is {@code ACTION_TARGET_OFFSET_TICKS + 2}.
 *
 * <p><strong>Why not five.</strong> A fifth arrival would need the service to answer two
 * observations inside one tick. It cannot: the mod sends one observation per tick, the service's
 * latest-value queue keeps only the newest, and its controller step runs one at a time under the
 * execution fence, so at most one answer per tick plus the one already in flight can arrive. A
 * frame more than two ticks late is not an extra arrival either — it is the in-flight one,
 * counted above, and it will be retired fail-neutral rather than applied.
 *
 * <p>It was two frames while the budget was one tick and the block was two ticks long. A live
 * program-authority probe overflowed this window with {@code QUEUE_OVERFLOW during WAIT_RESET}
 * the moment the budget grew; the review then showed that three is exact only for the probe's
 * roughly 1 ms frames and that four is the bound under the latencies this lane was built for.
 * The capacity is a consequence of the budget and of the reset's own timeline, not an
 * independent choice, and an overflow is terminal — it latches {@code QUEUE_OVERFLOW} and kills
 * the session on an operator reset.
 */
public final class BoundedActionWindow {
    public static final int CAPACITY = (int) CommandValidator.ACTION_TARGET_OFFSET_TICKS + 2;

    private final ArrayDeque<CommandFrame> frames = new ArrayDeque<>(CAPACITY);

    public synchronized boolean offer(CommandFrame frame) {
        if (frames.size() == CAPACITY) {
            return false;
        }
        frames.addLast(frame);
        return true;
    }

    public synchronized Optional<CommandFrame> poll() {
        return Optional.ofNullable(frames.pollFirst());
    }

    public synchronized int size() {
        return frames.size();
    }

    public synchronized void clear() {
        frames.clear();
    }
}
