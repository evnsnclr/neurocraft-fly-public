package dev.neurocraft.fly;

import com.google.gson.JsonParser;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import net.fabricmc.loader.api.entrypoint.PreLaunchEntrypoint;

/** Refuse a launcher that resolves older or shadowed libraries before gameplay starts. */
public final class RuntimeLibraries implements PreLaunchEntrypoint {
    @Override
    public void onPreLaunch() {
        verify();
    }

    public static void verify() {
        try (var stream = RuntimeLibraries.class.getResourceAsStream("/neurocraft-java-runtime-pins.json")) {
            if (stream == null) throw new IOException("Missing Java library policy");
            var policy = JsonParser.parseReader(new InputStreamReader(stream, StandardCharsets.UTF_8))
                .getAsJsonObject();
            for (var item : policy.getAsJsonArray("libraries")) {
                var pin = item.getAsJsonObject();
                var anchor = Class.forName(pin.get("anchor").getAsString(), false,
                    RuntimeLibraries.class.getClassLoader());
                var source = anchor.getProtectionDomain().getCodeSource();
                if (source == null || !"file".equals(source.getLocation().getProtocol())) {
                    throw new IOException("Unverifiable Java library " + anchor.getName());
                }
                verifyFile(Path.of(source.getLocation().toURI()), pin.get("sha256").getAsString());
                System.out.println("NeuroCraft verified Java library " + pin.get("name").getAsString());
            }
        } catch (Exception error) {
            throw new IllegalStateException("NeuroCraft Fly needs its patched launcher version. "
                + "Install the generated NeuroCraft client version (from source: the install step of "
                + "scripts/java_runtime.py in the installation guide; private Companion testers: "
                + "Install client files) and select it in your launcher. "
                + "The Java library check failed.", error);
        }
    }

    static void verifyFile(Path path, String expected) throws Exception {
        if (!Files.isRegularFile(path) || Files.isSymbolicLink(path)) {
            throw new IOException("Java library is not a regular JAR: " + path.getFileName());
        }
        var digest = MessageDigest.getInstance("SHA-256");
        try (var stream = Files.newInputStream(path)) {
            byte[] buffer = new byte[65536];
            int count;
            while ((count = stream.read(buffer)) != -1) digest.update(buffer, 0, count);
        }
        if (!HexFormat.of().formatHex(digest.digest()).equals(expected)) {
            throw new IOException("Unapproved Java library bytes: " + path.getFileName());
        }
    }
}
