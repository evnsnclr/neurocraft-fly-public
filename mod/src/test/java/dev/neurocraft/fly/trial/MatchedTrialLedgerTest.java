package dev.neurocraft.fly.trial;

import static org.junit.jupiter.api.Assertions.*;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

final class MatchedTrialLedgerTest {
    @TempDir Path directory;

    @Test void drainsInOrderPreservesExactLargeEpochAndNeverOverwrites() throws Exception {
        Path path = directory.resolve("run.ndjson");
        try (var ledger = new MatchedTrialLedger(path)) {
            for (int index = 0; index < 20; index++) {
                JsonObject row = new JsonObject();
                row.addProperty("index", index);
                row.addProperty("epoch", 9007199254740999L + index);
                ledger.append(row);
            }
            assertThrows(FileAlreadyExistsException.class, () -> new MatchedTrialLedger(path));
        }
        var lines = Files.readAllLines(path);
        assertEquals(20, lines.size());
        for (int index = 0; index < 20; index++) {
            var row = JsonParser.parseString(lines.get(index)).getAsJsonObject();
            assertEquals(index, row.get("index").getAsInt());
            assertEquals(9007199254740999L + index, row.get("epoch").getAsLong());
        }
    }

    @Test void oversizedRowIsVisibleFailureAndDoesNotSilentlyContinue() throws Exception {
        Path path = directory.resolve("bounded.ndjson");
        try (var ledger = new MatchedTrialLedger(path)) {
            JsonObject good = new JsonObject(); good.addProperty("before", true); ledger.append(good);
            JsonObject bad = new JsonObject(); bad.addProperty("large", "x".repeat(32768));
            assertThrows(IllegalStateException.class, () -> ledger.append(bad));
            assertNotNull(ledger.failure());
            assertThrows(IllegalStateException.class, () -> ledger.append(good));
        }
        assertEquals(1, Files.readAllLines(path).size());
    }

    @Test void unavailableDirectoryFailsBeforeAnyRunnerCanStart() {
        assertThrows(java.io.IOException.class, () -> new MatchedTrialLedger(directory.resolve("absent/run.ndjson")));
    }
}
