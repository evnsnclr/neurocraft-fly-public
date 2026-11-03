package dev.neurocraft.fly.client;

import com.mojang.blaze3d.platform.NativeImage;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.AtomicInteger;
import net.fabricmc.fabric.api.client.gametest.v1.FabricClientGameTest;
import net.fabricmc.fabric.api.client.gametest.v1.context.ClientGameTestContext;
import org.lwjgl.stb.STBImageWrite;

/** Scheduler checks and one actual framebuffer write; not a live performance measurement. */
public final class LiveFrameCaptureClientGameTest implements FabricClientGameTest {
    @Override
    public void runTest(ClientGameTestContext context) {
        verifySchedule();
        verifyRecordingSchedule();
        verifyPhaseSlotJitter();
        verifyDroppedSlotsAndMicroburstGuard();
        verifyCumulativeCeiling();
        verifyPngWriter();
        verifyImageOwnership();
        AtomicInteger registrations = new AtomicInteger();
        LiveFrameCapture.register(null, ignored -> registrations.incrementAndGet());
        LiveFrameCapture.register("  ", ignored -> registrations.incrementAndGet());
        if (registrations.get() != 0) {
            throw new AssertionError("absent capture property registered a client hook");
        }

        Path directory;
        try {
            directory = Files.createTempDirectory("neurocraft-live-frame-test-");
        } catch (IOException exception) {
            throw new UncheckedIOException(exception);
        }
        LiveFrameCapture capture = new LiveFrameCapture(directory);
        context.runOnClient(client -> capture.tick(client, System.nanoTime(),
                System.currentTimeMillis()));
        if (Files.exists(directory.resolve("frames.ndjson"))) {
            throw new AssertionError("capture wrote before a world and player existed");
        }

        try (var world = context.worldBuilder().create()) {
            world.getClientWorld().waitForChunksRender();
            context.waitTicks(20);
            long started = System.nanoTime();
            context.runOnClient(client -> capture.tick(client, started,
                    System.currentTimeMillis()));
            context.waitFor(client -> System.nanoTime() - started >= LiveFrameCapture.INTERVAL_NS);
            context.runOnClient(client -> capture.tick(client, System.nanoTime(),
                    System.currentTimeMillis()));
            context.waitFor(client -> saved(directory));
            Path screenshot = directory.resolve("screenshots/frame-0001.png");
            context.runOnClient(client -> {
                try (var stream = Files.newInputStream(screenshot);
                        NativeImage image = NativeImage.read(stream)) {
                    if (image.getWidth() != client.getMainRenderTarget().width
                            || image.getHeight() != client.getMainRenderTarget().height) {
                        throw new AssertionError("saved PNG differs from actual framebuffer size");
                    }
                } catch (IOException exception) {
                    throw new UncheckedIOException(exception);
                }
            });
            try {
                var metadata = Files.readAllLines(directory.resolve("frames.ndjson"));
                if (metadata.size() != 2 || !metadata.getFirst().contains("\"status\":\"requested\"")) {
                    throw new AssertionError("capture must append request and completion metadata");
                }
                verifyFirstSlotMetadata(metadata);
                if (!metadata.getFirst().contains("\"request_game_tick\":")
                        || !metadata.getFirst().contains("\"request_wall_elapsed_ns\":")) {
                    throw new AssertionError("capture metadata omits request timing");
                }
                if (!metadata.getLast().contains("\"png_writer\":\"minecraft_stb\"")
                        || !metadata.getLast().contains("\"png_compression\":null")
                        || !metadata.getLast().contains("\"frame_bytes\":" + Files.size(screenshot))) {
                    throw new AssertionError("default capture metadata invents compression or omits bytes");
                }
                byte[] original = Files.readAllBytes(screenshot);
                String originalMetadata = Files.readString(directory.resolve("frames.ndjson"));
                LiveFrameCapture reused = new LiveFrameCapture(directory);
                long reusedStart = System.nanoTime();
                context.runOnClient(client -> {
                    reused.tick(client, reusedStart, System.currentTimeMillis());
                    reused.tick(client, reusedStart + LiveFrameCapture.INTERVAL_NS,
                            System.currentTimeMillis());
                    reused.tick(client, reusedStart + 2 * LiveFrameCapture.INTERVAL_NS,
                            System.currentTimeMillis());
                });
                if (!java.util.Arrays.equals(original, Files.readAllBytes(screenshot))
                        || !originalMetadata.equals(Files.readString(directory.resolve("frames.ndjson")))) {
                    throw new AssertionError("reused capture directory overwrote original evidence");
                }
                // Retain the actual PNG and its metadata, even after this automated test ends.
                System.out.println("Live framebuffer capture test retained " + directory);
            } catch (IOException exception) {
                throw new UncheckedIOException(exception);
            }
            verifyConfiguredFramebuffer(context);
        }
    }

    private static void verifyConfiguredFramebuffer(ClientGameTestContext context) {
        try {
            Path directory = Files.createTempDirectory("neurocraft-configured-png-test-");
            var capture = new LiveFrameCapture(directory, new LiveFrameCapture.Settings(20, 2, 1));
            long started = System.nanoTime();
            context.runOnClient(client -> capture.tick(client, started, System.currentTimeMillis()));
            context.waitFor(client -> System.nanoTime() - started >= LiveFrameCapture.INTERVAL_NS);
            context.runOnClient(client -> capture.tick(client, System.nanoTime(), System.currentTimeMillis()));
            context.waitFor(client -> saved(directory));
            Path screenshot = directory.resolve("screenshots/frame-0001.png");
            context.runOnClient(client -> {
                try (var stream = Files.newInputStream(screenshot);
                        NativeImage image = NativeImage.read(stream)) {
                    if (image.getWidth() != client.getMainRenderTarget().width
                            || image.getHeight() != client.getMainRenderTarget().height) {
                        throw new AssertionError("configured writer changed framebuffer dimensions");
                    }
                } catch (IOException exception) {
                    throw new UncheckedIOException(exception);
                }
            });
            var metadata = Files.readAllLines(directory.resolve("frames.ndjson"));
            verifyFirstSlotMetadata(metadata);
            String row = metadata.getLast();
            if (!row.contains("\"png_writer\":\"jdk_deflater_rgba_filter0_v1\"")
                    || !row.contains("\"png_compression\":1")
                    || !row.contains("\"frame_bytes\":" + Files.size(screenshot))
                    || !row.contains("\"total_frame_bytes\":" + Files.size(screenshot))
                    || !row.contains("\"request_to_readback_complete_elapsed_ns\":")
                    || !row.contains("\"native_readback_elapsed_ns\":")
                    || !row.contains("\"png_pixel_copy_elapsed_ns\":")
                    || !row.contains("\"png_encode_write_elapsed_ns\":")
                    || row.contains("elapsed_ns\":null")) {
                throw new AssertionError("configured framebuffer metadata omits measured writer details");
            }
            System.out.println("Configured PNG framebuffer test retained " + directory);
        } catch (IOException exception) {
            throw new UncheckedIOException(exception);
        }
    }

    private static void verifyFirstSlotMetadata(java.util.List<String> metadata) {
        for (String row : metadata) {
            if (!row.contains("\"schedule_kind\":\"fixed_slots_half_period_v1\"")
                    || !row.contains("\"schedule_slot\":0,")
                    || !row.contains("\"missed_slots_before_request\":0,")) {
                throw new AssertionError("request/completion did not retain the same first-slot identity");
            }
        }
    }

    private static void verifyPngWriter() {
        int stbCompression = STBImageWrite.stbi_write_png_compression_level.get(0);
        try (NativeImage source = new NativeImage(67, 35, false)) {
            Path directory = Files.createTempDirectory("neurocraft-png-pixels-test-");
            // Asymmetric coordinates, independent RGB channels, and non-opaque alpha catch flips
            // and ABGR/RGBA mistakes that a dimensions-only framebuffer assertion cannot catch.
            for (int y = 0; y < source.getHeight(); y++) {
                for (int x = 0; x < source.getWidth(); x++) {
                    int rgba = ((x * 7 + y * 13) & 255) << 24 | ((x * 11) & 255) << 16
                            | ((y * 9) & 255) << 8 | ((x + y * 3) & 255);
                    source.setPixelABGR(x, y, rgba);
                }
            }
            Path vanilla = directory.resolve("vanilla.png");
            source.writeToFile(vanilla);
            try (NativeImage expected = NativeImage.read(Files.readAllBytes(vanilla))) {
                for (int compression : new int[] {0, 1, 9}) {
                    Path output = directory.resolve("configured-" + compression + ".png");
                    var result = LiveFrameCapture.writePng(source, output, compression,
                            LiveFrameCapture.BYTE_LIMIT);
                    try (NativeImage actual = NativeImage.read(Files.readAllBytes(output))) {
                        if (actual.getWidth() != source.getWidth() || actual.getHeight() != source.getHeight()
                                || !Arrays.equals(source.getPixelsABGR(), actual.getPixelsABGR())
                                || !Arrays.equals(expected.getPixelsABGR(), actual.getPixelsABGR())) {
                            throw new AssertionError("configured PNG differs from original/vanilla RGBA pixels");
                        }
                    }
                    if (result.bytes() != Files.size(output) || result.pixelCopyNanos() < 0
                            || result.encodeWriteNanos() < 0) {
                        throw new AssertionError("PNG accounting differs from actual file");
                    }
                    byte[] original = Files.readAllBytes(output);
                    try {
                        LiveFrameCapture.writePng(source, output, compression, LiveFrameCapture.BYTE_LIMIT);
                        throw new AssertionError("configured PNG overwrote evidence");
                    } catch (IOException expectedFailure) {
                        if (!Arrays.equals(original, Files.readAllBytes(output))) {
                            throw new AssertionError("failed output changed existing evidence");
                        }
                    }
                }
            }
            Path limited = directory.resolve("limited.png");
            try {
                LiveFrameCapture.writePng(source, limited, 0, 100);
                throw new AssertionError("configured PNG exceeded remaining byte allowance");
            } catch (IOException expectedFailure) {
                if (Files.size(limited) > 100) {
                    throw new AssertionError("failed PNG exceeded its hard byte bound");
                }
            }
            if (STBImageWrite.stbi_write_png_compression_level.get(0) != stbCompression) {
                throw new AssertionError("capture writer modified process-global STB compression");
            }
        } catch (IOException exception) {
            throw new UncheckedIOException(exception);
        }
    }

    private static void verifyImageOwnership() {
        for (int mode = 0; mode < 3; mode++) {
            final int outcome = mode;
            NativeImage image = new NativeImage(1, 1, false);
            AtomicInteger completions = new AtomicInteger();
            AtomicInteger writes = new AtomicInteger();
            LiveFrameCapture.writeOwnedAsync(image, job -> {
                if (outcome == 2) {
                    throw new RejectedExecutionException("test rejection");
                }
                job.run();
            }, owned -> {
                writes.incrementAndGet();
                if (owned.getPointer() == 0) {
                    throw new AssertionError("native image closed before writer ran");
                }
                if (outcome == 1) {
                    throw new IOException("test writer failure");
                }
                return new LiveFrameCapture.PngResult(1, 0, 0);
            }, result -> {
                completions.incrementAndGet();
                if (result.success() != (outcome == 0) || image.getPointer() != 0) {
                    throw new AssertionError("completion success/closed image does not match job outcome");
                }
            });
            if (completions.get() != 1 || writes.get() != (mode == 2 ? 0 : 1)
                    || image.getPointer() != 0) {
                throw new AssertionError("capture job ownership leaked on success/failure/rejection");
            }
        }
    }

    private static boolean saved(Path directory) {
        Path metadata = directory.resolve("frames.ndjson");
        try {
            return Files.exists(metadata)
                    && Files.readString(metadata, StandardCharsets.UTF_8).contains("\"status\":\"saved\"");
        } catch (IOException exception) {
            throw new UncheckedIOException(exception);
        }
    }

    private static void verifySchedule() {
        var schedule = new LiveFrameCapture.Schedule();
        if (!schedule.request(0) || schedule.request(LiveFrameCapture.INTERVAL_NS)) {
            throw new AssertionError("capture allowed overlapping jobs");
        }
        if (schedule.expired(LiveFrameCapture.TIMEOUT_NS - 1)
                || !schedule.expired(LiveFrameCapture.TIMEOUT_NS)) {
            throw new AssertionError("capture timeout is not bounded");
        }
        schedule.complete();
        if (schedule.request(LiveFrameCapture.INTERVAL_NS - 1)) {
            throw new AssertionError("capture exceeded one request per wall-clock second");
        }
        for (int frame = 1; frame < LiveFrameCapture.FRAME_LIMIT; frame++) {
            if (!schedule.request(frame * LiveFrameCapture.INTERVAL_NS)) {
                throw new AssertionError("capture stopped below the 120-frame limit");
            }
            schedule.complete();
        }
        if (schedule.request(LiveFrameCapture.FRAME_LIMIT * LiveFrameCapture.INTERVAL_NS)) {
            throw new AssertionError("capture exceeded the 120-frame limit");
        }
    }

    private static void verifyRecordingSchedule() {
        var schedule = new LiveFrameCapture.Schedule(new LiveFrameCapture.Settings(20, 2));
        if (!schedule.request(0)) {
            throw new AssertionError("recording failed to request its first frame");
        }
        schedule.complete();
        if (schedule.request(49_999_999L) || !schedule.request(50_000_000L)) {
            throw new AssertionError("recording exceeded its requested frame rate");
        }
        schedule.complete();
        // A slow write produces a gap. The scheduler never catches up with a burst or extends time.
        if (!schedule.request(1_900_000_000L)) {
            throw new AssertionError("recording failed to resume after a gap");
        }
        schedule.complete();
        if (schedule.request(2_000_000_000L)) {
            throw new AssertionError("recording exceeded its wall-time window");
        }
        for (int[] invalid : new int[][] {{0, 1}, {21, 1}, {1, 0}, {1, 721}}) {
            try {
                new LiveFrameCapture.Settings(invalid[0], invalid[1]);
                throw new AssertionError("recording accepted unbounded settings");
            } catch (IllegalArgumentException expected) {
                // Validation must happen before a render hook or filesystem output is created.
            }
        }
        if (new LiveFrameCapture.Settings(1, 120).pngCompression() != null
                || LiveFrameCapture.configuredCompression(null) != null) {
            throw new AssertionError("absent compression property must preserve vanilla writer");
        }
        for (int compression : new int[] {-1, 10}) {
            try {
                new LiveFrameCapture.Settings(20, 2, compression);
                throw new AssertionError("capture accepted invalid PNG compression");
            } catch (IllegalArgumentException expected) {
                // Invalid optional configuration cannot start capture.
            }
        }
    }

    private static void verifyPhaseSlotJitter() {
        var schedule = new LiveFrameCapture.Schedule(new LiveFrameCapture.Settings(20, 2));
        long[] requests = {0, 50_100_000L, 100_000_000L, 150_100_000L};
        for (int slot = 0; slot < requests.length; slot++) {
            if (!schedule.request(requests[slot]) || schedule.lastSlot() != slot
                    || schedule.missedSlotsBeforeRequest() != 0) {
                throw new AssertionError("ordinary 50.1/49.9ms jitter dropped a current capture slot");
            }
            schedule.complete();
            if (slot == 0 && schedule.request(49_999_999L)) {
                throw new AssertionError("first slot ended before its cumulative FPS boundary");
            }
        }
    }

    private static void verifyDroppedSlotsAndMicroburstGuard() {
        var backlog = new LiveFrameCapture.Schedule(new LiveFrameCapture.Settings(20, 2));
        if (!backlog.request(0)) { throw new AssertionError("missing initial request"); }
        for (long now : new long[] {50_000_000L, 100_000_000L, 200_000_000L}) {
            if (backlog.request(now)) { throw new AssertionError("overlapping outstanding capture"); }
        }
        backlog.complete();
        if (!backlog.request(210_000_000L) || backlog.lastSlot() != 4
                || backlog.missedSlotsBeforeRequest() != 3) {
            throw new AssertionError("resumption must drop three missed slots and take only slot four");
        }
        // Timeout belongs to this actual request, not its nominal slot or the first request.
        if (backlog.expired(210_000_000L + LiveFrameCapture.TIMEOUT_NS - 1)
                || !backlog.expired(210_000_000L + LiveFrameCapture.TIMEOUT_NS)) {
            throw new AssertionError("phase schedule moved the outstanding-write timeout");
        }
        backlog.complete();
        if (backlog.expired(210_000_000L + LiveFrameCapture.TIMEOUT_NS)
                || backlog.request(210_000_000L) || backlog.request(249_999_999L)
                || !backlog.request(250_000_000L) || backlog.missedSlotsBeforeRequest() != 0) {
            throw new AssertionError("missed slots were replayed or a completed write stayed outstanding");
        }

        var boundary = new LiveFrameCapture.Schedule(new LiveFrameCapture.Settings(20, 2));
        if (!boundary.request(0)) { throw new AssertionError("missing initial request"); }
        boundary.complete();
        if (!boundary.request(99_900_000L)) { throw new AssertionError("missing late slot one"); }
        boundary.complete();
        if (boundary.request(100_100_000L) || boundary.request(124_899_999L)
                || !boundary.request(124_900_000L) || boundary.lastSlot() != 2) {
            throw new AssertionError("adjacent phase slots bypassed the half-period spacing guard");
        }
        boundary.complete();
        if (boundary.request(149_999_999L) || !boundary.request(150_000_000L)) {
            throw new AssertionError("one slot admitted twice or phase resumed late after guarded request");
        }
    }

    private static void verifyCumulativeCeiling() {
        for (int fps : new int[] {1, 3, 7, 20}) {
            var schedule = new LiveFrameCapture.Schedule(new LiveFrameCapture.Settings(fps, 2));
            // Deliberately cross nanoTime's signed wrap while the bounded elapsed time stays small.
            long origin = Long.MAX_VALUE - 1_000_000_000L;
            int admitted = 0;
            long previous = -1;
            for (long elapsed = 0; elapsed <= 2_100_000_000L; elapsed += 1_000_003L) {
                if (schedule.request(origin + elapsed)) {
                    admitted++;
                    if (admitted > 1 + elapsed * fps / 1_000_000_000L
                            || elapsed >= 2_000_000_000L
                            || (previous >= 0 && (elapsed - previous) * 2 * fps < 1_000_000_000L)) {
                        throw new AssertionError("capture exceeded cumulative/time/minimum-spacing bounds");
                    }
                    previous = elapsed;
                    schedule.complete();
                }
            }
            if (admitted != fps * 2 || schedule.request(origin + 3_000_000_000L)) {
                throw new AssertionError("capture stopped early or extended its frame/time window");
            }
        }
    }
}
