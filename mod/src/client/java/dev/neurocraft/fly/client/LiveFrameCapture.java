package dev.neurocraft.fly.client;

import com.mojang.blaze3d.platform.NativeImage;
import com.mojang.logging.LogUtils;
import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Locale;
import java.util.concurrent.Executor;
import java.util.function.Consumer;
import java.util.zip.CRC32;
import java.util.zip.Deflater;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.minecraft.Util;
import net.minecraft.client.Minecraft;
import net.minecraft.client.Screenshot;
import net.minecraft.network.chat.Component;
import net.minecraft.network.chat.contents.TranslatableContents;
import org.lwjgl.system.MemoryUtil;
import org.slf4j.Logger;

/**
 * Opt-in diagnostic samples of the actual last rendered framebuffer, including its HUD.
 * Request timestamps describe capture scheduling, not neural latency or an exact image tick.
 * This is neither continuous video nor evidence of sustained real-time performance.
 */
public final class LiveFrameCapture {
    public static final String DIRECTORY_PROPERTY = "neurocraft.fly.captureDirectory";
    public static final String FPS_PROPERTY = "neurocraft.fly.captureFps";
    public static final String SECONDS_PROPERTY = "neurocraft.fly.captureSeconds";
    public static final String PNG_COMPRESSION_PROPERTY = "neurocraft.fly.capturePngCompression";
    static final long INTERVAL_NS = 1_000_000_000L;
    static final long TIMEOUT_NS = 10_000_000_000L;
    static final int FRAME_LIMIT = 120;
    static final long BYTE_LIMIT = 4L * 1024 * 1024 * 1024;
    private static final Logger LOGGER = LogUtils.getLogger();

    private final Path directory;
    private final long startedNanos = System.nanoTime();
    private final Settings settings;
    private final Schedule schedule;
    private boolean metadataCreated;
    private boolean disabled;
    private long readySince = Long.MIN_VALUE;
    private Request outstanding;
    private long savedBytes;

    LiveFrameCapture(Path directory) {
        this(directory, new Settings(1, 120));
    }

    LiveFrameCapture(Path directory, Settings settings) {
        this.directory = directory.toAbsolutePath().normalize();
        this.settings = settings;
        this.schedule = new Schedule(settings);
    }

    public static void register() {
        register(System.getProperty(DIRECTORY_PROPERTY), ClientTickEvents.END_CLIENT_TICK::register);
    }

    static void register(String configured, Consumer<ClientTickEvents.EndTick> registrar) {
        if (configured == null || configured.isBlank()) {
            return;
        }
        try {
            LiveFrameCapture capture = new LiveFrameCapture(Path.of(configured), new Settings(
                    Integer.parseInt(System.getProperty(FPS_PROPERTY, "1")),
                    Integer.parseInt(System.getProperty(SECONDS_PROPERTY, "120")),
                    configuredCompression(System.getProperty(PNG_COMPRESSION_PROPERTY))));
            registrar.accept(capture::tick);
        } catch (RuntimeException exception) {
            LOGGER.warn("Diagnostic framebuffer capture disabled: invalid configuration", exception);
        }
    }

    private void tick(Minecraft client) {
        tick(client, System.nanoTime(), System.currentTimeMillis());
    }

    synchronized void tick(Minecraft client, long nowNanos, long utcMillis) {
        if (disabled) {
            return;
        }
        if (schedule.expired(nowNanos)) {
            disable("timed_out", nowNanos);
            return;
        }
        if (client.level == null || client.player == null) {
            readySince = Long.MIN_VALUE;
            return;
        }
        // Allow a real world frame to be drawn before sampling the previous framebuffer.
        if (readySince == Long.MIN_VALUE) {
            readySince = nowNanos;
            return;
        }
        if (nowNanos - readySince < INTERVAL_NS || !schedule.request(nowNanos)) {
            return;
        }
        Request request = new Request(
                schedule.frames, nowNanos, nowNanos - startedNanos, utcMillis,
                client.level.getGameTime(), schedule.lastSlot, schedule.missedSlotsBeforeRequest);
        outstanding = request;
        try {
            // Bound optional disk output even if the requested window is long or very large.
            if (savedBytes >= BYTE_LIMIT
                    || client.getMainRenderTarget().width > 4096
                    || client.getMainRenderTarget().height > 2160) {
                disable("storage_or_resolution_limit", nowNanos);
                return;
            }
            Path screenshot = directory.resolve("screenshots").resolve(request.filename());
            if (Files.exists(screenshot)) {
                throw new IOException("capture filename already exists");
            }
            if (!metadataCreated) {
                Files.createDirectories(directory);
                Files.writeString(directory.resolve("frames.ndjson"), "", StandardCharsets.UTF_8,
                        StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE);
                metadataCreated = true;
            }
            append(request, "requested", nowNanos);
            if (settings.pngCompression() == null) {
                Screenshot.grab(directory.toFile(), request.filename(), client.getMainRenderTarget(),
                        result -> complete(request, result, null));
            } else {
                long remainingBytes = BYTE_LIMIT - savedBytes;
                long nativeReadbackStarted = System.nanoTime();
                Screenshot.takeScreenshot(client.getMainRenderTarget(), image -> {
                    long readbackCompleted = System.nanoTime();
                    long readbackElapsed = readbackCompleted - request.startedNanos();
                    // Includes the async GPU wait and native-image copy, excluding request setup.
                    long nativeReadbackElapsed = readbackCompleted - nativeReadbackStarted;
                    writeOwnedAsync(image, Util.ioPool(), owned -> writePng(owned, screenshot,
                            settings.pngCompression(), remainingBytes), result -> complete(request,
                                    Component.translatable(result.success()
                                            ? "screenshot.success" : "screenshot.failure"),
                                    new WriteTiming(readbackElapsed, nativeReadbackElapsed,
                                            result.png(), result.storageLimit())));
                });
            }
        } catch (IOException | RuntimeException exception) {
            disable("request_failed", nowNanos);
        }
    }

    private synchronized void complete(Request request, Component result, WriteTiming timing) {
        boolean success;
        String failure = timing != null && timing.storageLimit() ? "storage_limit" : "save_failed";
        long now = System.nanoTime();
        try {
            success = result.getContents() instanceof TranslatableContents translated
                    && "screenshot.success".equals(translated.getKey());
            Path screenshot = directory.resolve("screenshots").resolve(request.filename());
            long frameBytes = Files.isRegularFile(screenshot) ? Files.size(screenshot) : 0;
            success = success && frameBytes > 0;
            savedBytes += frameBytes; // Account for a partial failed output too.
            append(request, success ? (disabled ? "saved_after_disable" : "saved") : failure,
                    now, frameBytes, timing);
        } catch (IOException | RuntimeException exception) {
            success = false;
        }
        outstanding = null;
        schedule.complete();
        if (!success && !disabled) {
            disable(failure, now);
        }
    }

    private void disable(String status, long nowNanos) {
        disabled = true;
        if (metadataCreated && outstanding != null) {
            try {
                append(outstanding, status, nowNanos);
            } catch (IOException | RuntimeException ignored) {
                // A broken diagnostic destination must never fault the client or bridge.
            }
        }
        LOGGER.warn("Diagnostic framebuffer capture disabled: {}", status);
    }

    private void append(Request request, String status, long nowNanos) throws IOException {
        append(request, status, nowNanos, null, null);
    }

    private void append(Request request, String status, long nowNanos, Long frameBytes,
                        WriteTiming timing) throws IOException {
        String row = String.format(Locale.ROOT,
                "{\"schema\":\"neurocraft.live_frame_capture.v1\","
                        + "\"diagnostic_sampling\":true,\"requested_fps\":%d,"
                        + "\"schedule_kind\":\"fixed_slots_half_period_v1\","
                        + "\"schedule_slot\":%d,\"missed_slots_before_request\":%d,"
                        + "\"requested_duration_seconds\":%d,\"frame\":%d,"
                        + "\"file\":\"screenshots/%s\","
                        + "\"request_wall_elapsed_ns\":%d,\"request_utc_epoch_ms\":%d,"
                        + "\"request_game_tick\":%d,\"status_elapsed_since_request_ns\":%d,"
                        + "\"png_writer\":\"%s\",\"png_compression\":%s,"
                        + "\"frame_bytes\":%s,\"total_frame_bytes\":%d,"
                        + "\"request_to_readback_complete_elapsed_ns\":%s,"
                        + "\"native_readback_elapsed_ns\":%s,"
                        + "\"png_pixel_copy_elapsed_ns\":%s,\"png_encode_write_elapsed_ns\":%s,"
                        + "\"status\":\"%s\"}%n",
                settings.fps(), request.scheduleSlot(), request.missedSlotsBeforeRequest(),
                settings.seconds(), request.frame(), request.filename(),
                request.elapsedNanos(), request.utcMillis(),
                request.gameTick(), nowNanos - request.startedNanos(),
                settings.pngCompression() == null ? "minecraft_stb" : "jdk_deflater_rgba_filter0_v1",
                settings.pngCompression(), frameBytes, savedBytes,
                timing == null ? null : timing.readbackNanos(),
                timing == null ? null : timing.nativeReadbackNanos(),
                timing == null || timing.png() == null ? null : timing.png().pixelCopyNanos(),
                timing == null || timing.png() == null ? null : timing.png().encodeWriteNanos(), status);
        Files.writeString(directory.resolve("frames.ndjson"), row, StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);
    }

    static Integer configuredCompression(String value) {
        return value == null ? null : Integer.valueOf(value);
    }

    /** Ownership transfers at submission; every successful or rejected job closes its image. */
    static void writeOwnedAsync(NativeImage image, Executor executor, ImageWriter writer,
                                Consumer<WriteResult> completion) {
        try {
            executor.execute(() -> {
                WriteResult result;
                try (image) {
                    result = new WriteResult(true, writer.write(image), false);
                } catch (IOException | RuntimeException exception) {
                    result = new WriteResult(false, null, exception instanceof PngLimitException);
                }
                notifyWrite(completion, result);
            });
        } catch (RuntimeException exception) {
            image.close();
            notifyWrite(completion, new WriteResult(false, null, false));
        }
    }

    private static void notifyWrite(Consumer<WriteResult> completion, WriteResult result) {
        try {
            completion.accept(result);
        } catch (RuntimeException exception) {
            LOGGER.warn("Diagnostic framebuffer completion failed", exception);
        }
    }

    /**
     * Capture-only streaming PNG. STB's compression knob is process-global and cannot safely be
     * changed around a shared-pool screenshot. Each image instead owns one JDK compressor; no
     * STB state is read or modified. NativeImage RGBA rows are already top-down after readback.
     * Filter 0 is explicit, with bounded row/chunk buffers and no full-image Java pixel copy.
     */
    static PngResult writePng(NativeImage image, Path path, int compression, long byteBudget)
            throws IOException {
        if (compression < 0 || compression > 9 || image.format() != NativeImage.Format.RGBA
                || image.getWidth() < 1 || image.getWidth() > 4096
                || image.getHeight() < 1 || image.getHeight() > 2160 || image.getPointer() == 0) {
            throw new IOException("invalid diagnostic PNG bounds");
        }
        if (byteBudget < 57) {
            throw new PngLimitException();
        }
        long started = System.nanoTime();
        long copyNanos = 0;
        int stride = image.getWidth() * 4;
        ByteBuffer pixels = MemoryUtil.memByteBuffer(image.getPointer(), stride * image.getHeight());
        byte[] row = new byte[stride + 1]; // first byte is the declared PNG filter 0
        byte[] compressed = new byte[64 * 1024];
        Deflater deflater = new Deflater(compression);
        long bytes = 8;
        try {
            Files.createDirectories(path.getParent());
            try (var output = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(
                    path, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)))) {
                output.writeLong(0x89504e470d0a1a0aL);
                byte[] header = ByteBuffer.allocate(13).putInt(image.getWidth())
                        .putInt(image.getHeight()).put((byte) 8).put((byte) 6)
                        .put((byte) 0).put((byte) 0).put((byte) 0).array();
                bytes += writeChunk(output, "IHDR", header, header.length, byteBudget - bytes);
                for (int y = 0; y < image.getHeight(); y++) {
                    long copyStarted = System.nanoTime();
                    pixels.get(y * stride, row, 1, stride);
                    copyNanos += System.nanoTime() - copyStarted;
                    deflater.setInput(row);
                    while (!deflater.needsInput()) {
                        int size = deflater.deflate(compressed);
                        if (size > 0) {
                            bytes += writeChunk(output, "IDAT", compressed, size, byteBudget - bytes);
                        }
                    }
                }
                deflater.finish();
                while (!deflater.finished()) {
                    int size = deflater.deflate(compressed);
                    if (size > 0) {
                        bytes += writeChunk(output, "IDAT", compressed, size, byteBudget - bytes);
                    }
                }
                bytes += writeChunk(output, "IEND", new byte[0], 0, byteBudget - bytes);
            }
        } finally {
            deflater.end();
        }
        return new PngResult(bytes, copyNanos, System.nanoTime() - started - copyNanos);
    }

    private static long writeChunk(DataOutputStream output, String type, byte[] data, int size,
                                   long byteBudget) throws IOException {
        if (size + 12L > byteBudget) {
            throw new PngLimitException();
        }
        byte[] kind = type.getBytes(StandardCharsets.US_ASCII);
        CRC32 crc = new CRC32();
        crc.update(kind);
        crc.update(data, 0, size);
        output.writeInt(size);
        output.write(kind);
        output.write(data, 0, size);
        output.writeInt((int) crc.getValue());
        return size + 12L;
    }

    @FunctionalInterface
    interface ImageWriter {
        PngResult write(NativeImage image) throws IOException;
    }

    record PngResult(long bytes, long pixelCopyNanos, long encodeWriteNanos) {}
    record WriteResult(boolean success, PngResult png, boolean storageLimit) {}
    private record WriteTiming(long readbackNanos, long nativeReadbackNanos,
                               PngResult png, boolean storageLimit) {}

    private static final class PngLimitException extends IOException {
        private static final long serialVersionUID = 1L;

        PngLimitException() {
            super("diagnostic PNG byte limit reached");
        }
    }

    private record Request(
            int frame, long startedNanos, long elapsedNanos, long utcMillis, long gameTick,
            long scheduleSlot, long missedSlotsBeforeRequest) {
        String filename() {
            return String.format(Locale.ROOT, "frame-%04d.png", frame);
        }
    }

    /**
     * Pure fixed-slot schedule, with no filesystem, render, or controller authority.
     * Requested FPS is a cumulative ceiling from the first request: at elapsed seconds t,
     * at most 1 + floor(t * fps) requests exist. Adjacent intervals may jitter, but never below
     * half a period. Missed slots are discarded; no loop or backlog replays them after a stall.
     * END_CLIENT_TICK jitter therefore does not repeatedly restart a full-period deadline.
     */
    static final class Schedule {
        private final Settings settings;
        private int frames;
        private boolean outstanding;
        private long firstRequest;
        private long lastRequest;
        private long lastSlot = -1;
        private long missedSlotsBeforeRequest;

        Schedule() {
            this(new Settings(1, 120));
        }

        Schedule(Settings settings) {
            this.settings = settings;
        }

        boolean request(long nowNanos) {
            if (outstanding || frames >= settings.fps() * settings.seconds()) {
                return false;
            }
            long elapsed = frames == 0 ? 0 : nowNanos - firstRequest;
            if (elapsed < 0 || elapsed >= settings.seconds() * 1_000_000_000L) {
                return false;
            }
            // Check the bounded recording window before multiplication, avoiding clock overflow.
            long slot = elapsed * settings.fps() / 1_000_000_000L;
            long halfPeriod = (1_000_000_000L + 2L * settings.fps() - 1) / (2L * settings.fps());
            if (frames > 0 && (slot <= lastSlot || nowNanos - lastRequest < halfPeriod)) {
                return false;
            }
            if (frames == 0) {
                firstRequest = nowNanos;
            }
            frames++;
            outstanding = true;
            missedSlotsBeforeRequest = slot - lastSlot - 1;
            lastSlot = slot;
            lastRequest = nowNanos;
            return true;
        }

        long lastSlot() {
            return lastSlot;
        }

        long missedSlotsBeforeRequest() {
            return missedSlotsBeforeRequest;
        }

        void complete() {
            outstanding = false;
        }

        boolean expired(long nowNanos) {
            return outstanding && nowNanos - lastRequest >= TIMEOUT_NS;
        }
    }

    /** Requested cumulative FPS ceiling with half-period jitter spacing, never an achieved rate. */
    record Settings(int fps, int seconds, Integer pngCompression) {
        Settings(int fps, int seconds) {
            this(fps, seconds, null);
        }

        Settings {
            if (fps < 1 || fps > 20 || seconds < 1 || seconds > 720) {
                throw new IllegalArgumentException("capture requires 1..20 fps and 1..720 seconds");
            }
            if (pngCompression != null && (pngCompression < 0 || pngCompression > 9)) {
                throw new IllegalArgumentException("capture PNG compression requires 0..9");
            }
        }
    }
}
