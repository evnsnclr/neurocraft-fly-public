package dev.neurocraft.fly.trial;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;

/** Exclusive diagnostic NDJSON with a bounded queue; a lost row invalidates the run. */
public final class MatchedTrialLedger implements AutoCloseable {
    public static final Gson JSON = new GsonBuilder().serializeNulls().disableHtmlEscaping().create();
    private static final int MAX_ROWS = 25000;
    private static final int MAX_ROW_CHARS = 32768;
    private final ArrayBlockingQueue<String> queue = new ArrayBlockingQueue<>(256);
    private final BufferedWriter writer;
    private final Thread thread;
    private final Path path;
    private volatile boolean closing;
    private volatile String failure;
    private int rows;

    public MatchedTrialLedger(Path path) throws IOException {
        this.path = path;
        writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE);
        thread = new Thread(this::writeRows, "matched-trial-ledger");
        thread.setDaemon(true);
        thread.start();
    }

    public Path path() { return path; }
    public String failure() { return failure; }

    public void append(JsonObject value) {
        if (closing || failure != null) { throw new IllegalStateException("trial ledger unavailable: " + failure); }
        String line = JSON.toJson(value);
        if (++rows > MAX_ROWS || line.length() > MAX_ROW_CHARS || !queue.offer(line)) {
            failure = "trial ledger bound exceeded; row unavailable";
            throw new IllegalStateException(failure);
        }
    }

    private void writeRows() {
        try (writer) {
            while (!closing || !queue.isEmpty()) {
                String row = queue.poll(100, TimeUnit.MILLISECONDS);
                if (row != null) { writer.write(row); writer.newLine(); }
            }
        } catch (IOException | InterruptedException exception) {
            failure = "trial ledger write failed: " + exception.getClass().getSimpleName();
            if (exception instanceof InterruptedException) { Thread.currentThread().interrupt(); }
        }
    }

    @Override public void close() {
        closing = true;
        try { thread.join(1000); }
        catch (InterruptedException exception) { Thread.currentThread().interrupt(); failure = "ledger close interrupted"; }
        if (thread.isAlive()) { failure = "ledger close exceeded one second"; }
    }
}
