package dev.neurocraft.fly.control;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.SeekableByteChannel;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.LongSupplier;

/**
 * Read-only local-file display polling on a daemon, separate from the command socket.
 * A single immutable slot replaces old reads; file/parse failures affect only this display.
 */
public final class EventDisplayReader implements AutoCloseable {
    public static final long POLL_PERIOD_MILLIS = 200;
    public enum Status { AVAILABLE, UNAVAILABLE, INVALID, CLOSED }

    public record Snapshot(Status status, Optional<EventDisplayFrame> frame, long firstSeenNanos) {
        public Snapshot {
            Objects.requireNonNull(status, "status");
            Objects.requireNonNull(frame, "frame");
            if ((status == Status.AVAILABLE) != frame.isPresent()) {
                throw new IllegalArgumentException("only AVAILABLE snapshots carry a frame");
            }
        }
    }

    private final Path path;
    private final ControllerIdentity identity;
    private final LongSupplier clock;
    private final ScheduledExecutorService executor;
    private final AtomicReference<Snapshot> latest = new AtomicReference<>(
            empty(Status.UNAVAILABLE));
    // Accessed only by the polling worker (or a deterministic test without a worker).
    private EventDisplayFrame previous;
    private long firstSeenNanos;
    private volatile boolean closed;

    public EventDisplayReader(Path path, ControllerIdentity identity) {
        this(path, identity, System::nanoTime, true);
    }

    EventDisplayReader(Path path, ControllerIdentity identity, LongSupplier clock, boolean startWorker) {
        this.path = Objects.requireNonNull(path, "path");
        this.identity = Objects.requireNonNull(identity, "identity");
        this.clock = Objects.requireNonNull(clock, "clock");
        executor = Executors.newSingleThreadScheduledExecutor(task -> {
            Thread worker = new Thread(task, "neurocraft-event-display");
            worker.setDaemon(true);
            return worker;
        });
        if (startWorker) {
            executor.scheduleWithFixedDelay(this::pollOnce, 0, POLL_PERIOD_MILLIS,
                    TimeUnit.MILLISECONDS);
        }
    }

    /** Constant-time, no disk access; safe on the game/render thread. */
    public Snapshot snapshot() {
        return closed ? empty(Status.CLOSED) : latest.get();
    }

    void pollOnce() {
        if (closed) {
            return;
        }
        Snapshot result;
        try {
            if (!Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS)) {
                result = empty(Status.UNAVAILABLE);
            } else {
                ByteBuffer bytes = ByteBuffer.allocate(EventDisplayFrame.MAX_BYTES + 1);
                try (SeekableByteChannel channel = Files.newByteChannel(
                        path, StandardOpenOption.READ, LinkOption.NOFOLLOW_LINKS)) {
                    while (bytes.hasRemaining() && channel.read(bytes) != -1) {
                        // A regular local file, read to EOF or the fixed byte ceiling.
                    }
                }
                if (bytes.position() > EventDisplayFrame.MAX_BYTES) {
                    throw new ProtocolException("event display file exceeds byte ceiling");
                }
                bytes.flip();
                String json = StandardCharsets.UTF_8.newDecoder()
                        .onMalformedInput(CodingErrorAction.REPORT)
                        .onUnmappableCharacter(CodingErrorAction.REPORT).decode(bytes).toString();
                EventDisplayFrame frame = EventDisplayFrame.parse(json);
                // A reused standalone path may still contain the previous run's file.
                // Reject it before adopting any sequence or freshness state, so the
                // current run's first sample can recover without restarting the reader.
                if (!identity.equals(frame.identity())) {
                    throw new ProtocolException("display identity does not match the configured run");
                }
                // Each file belongs to one session epoch. Replacing it with an older command
                // must not turn that command's first-read age back to zero.
                if (previous != null && (!previous.identity().equals(frame.identity())
                        || frame.commandSequence() < previous.commandSequence())) {
                    throw new ProtocolException("display identity changed or sequence regressed");
                }
                boolean sameCommand = previous != null && previous.identity().equals(frame.identity())
                        && previous.commandSequence() == frame.commandSequence();
                if (sameCommand && !previous.equals(frame)) {
                    throw new ProtocolException("display contents changed for the same command");
                }
                if (!sameCommand) {
                    firstSeenNanos = clock.getAsLong();
                    previous = frame;
                }
                result = new Snapshot(Status.AVAILABLE, Optional.of(frame), firstSeenNanos);
            }
        } catch (IOException exception) {
            result = empty(Status.UNAVAILABLE);
        } catch (ProtocolException | RuntimeException exception) {
            result = empty(Status.INVALID);
        }
        // close may race an in-flight read; CLOSED always wins in that case.
        latest.set(closed ? empty(Status.CLOSED) : result);
    }

    @Override
    public void close() {
        closed = true;
        executor.shutdownNow();
        latest.set(empty(Status.CLOSED));
    }

    private static Snapshot empty(Status status) {
        return new Snapshot(status, Optional.empty(), 0);
    }
}
