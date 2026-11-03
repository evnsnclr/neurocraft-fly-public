package dev.neurocraft.fly;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

final class RuntimeLibrariesTest {
    @Test
    void resolvedRuntimeContainsThePinnedPatchedBytes() {
        assertDoesNotThrow(RuntimeLibraries::verify);
    }

    @Test
    void modifiedLibraryIsRejected(@TempDir Path directory) throws Exception {
        var file = directory.resolve("dependency.jar");
        Files.writeString(file, "different dependency bytes");
        assertThrows(Exception.class, () -> RuntimeLibraries.verifyFile(file, "0".repeat(64)));
    }
}
