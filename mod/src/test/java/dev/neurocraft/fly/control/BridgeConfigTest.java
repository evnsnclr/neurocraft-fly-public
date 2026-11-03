package dev.neurocraft.fly.control;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.net.URI;
import java.util.Map;
import java.util.Properties;
import org.junit.jupiter.api.Test;

final class BridgeConfigTest {
    private static final String TOKEN = "0123456789abcdef0123456789abcdef";

    @Test
    void acceptsOnlyExplicitLoopbackWebSocketAndRedactsToken() {
        Properties properties = validProperties();
        properties.setProperty("neurocraft.fly.bridgeUri", "ws://[::1]:8765/control");

        BridgeConfig config = BridgeConfig.from(Map.of("NEUROCRAFT_RUN_TOKEN", TOKEN), properties);

        assertEquals(URI.create("ws://[::1]:8765/control"), config.uri());
        assertFalse(config.toString().contains(TOKEN));
        assertTrue(config.toString().contains("<redacted>"));
    }

    @Test
    void rejectsNonLoopbackAmbiguousOrCredentialBearingUris() {
        for (String uri : new String[] {
            "ws://0.0.0.0:8765/control",
            "ws://localhost:8765/control",
            "ws://example.com:8765/control",
            "wss://127.0.0.1:8765/control",
            "ws://user@127.0.0.1:8765/control",
            "ws://127.0.0.1:8765/other",
            "ws://127.0.0.1:8765/control?token=x"
        }) {
            Properties properties = validProperties();
            properties.setProperty("neurocraft.fly.bridgeUri", uri);
            assertThrows(
                    IllegalArgumentException.class,
                    () -> BridgeConfig.from(Map.of("NEUROCRAFT_RUN_TOKEN", TOKEN), properties),
                    uri);
        }
    }

    @Test
    void requiresSafeNontrivialTokenAndEveryIdentity() {
        assertThrows(
                IllegalArgumentException.class,
                () -> BridgeConfig.from(Map.of("NEUROCRAFT_RUN_TOKEN", "short"), validProperties()));
        Properties legacyPropertyToken = validProperties();
        legacyPropertyToken.setProperty("neurocraft.fly.token", TOKEN);
        assertThrows(
                IllegalArgumentException.class,
                () -> BridgeConfig.from(Map.of(), legacyPropertyToken));
        Properties missingHash = validProperties();
        missingHash.remove("neurocraft.fly.graphHash");
        assertThrows(
                IllegalArgumentException.class,
                () -> BridgeConfig.from(Map.of("NEUROCRAFT_RUN_TOKEN", TOKEN), missingHash));
        Properties nonMinecraftIdentity = validProperties();
        nonMinecraftIdentity.setProperty("neurocraft.fly.entityId", "fly-01");
        assertThrows(
                IllegalArgumentException.class,
                () -> BridgeConfig.from(
                        Map.of("NEUROCRAFT_RUN_TOKEN", TOKEN), nonMinecraftIdentity));
    }

    private static Properties validProperties() {
        Properties properties = new Properties();
        properties.setProperty("neurocraft.fly.runId", "run-1");
        properties.setProperty("neurocraft.fly.sessionEpoch", "7");
        properties.setProperty(
                "neurocraft.fly.entityId", "00000000-0000-0000-0000-000000000001");
        properties.setProperty("neurocraft.fly.graphHash", "1".repeat(64));
        properties.setProperty("neurocraft.fly.adapterHash", "2".repeat(64));
        properties.setProperty("neurocraft.fly.dynamicsHash", "3".repeat(64));
        properties.setProperty("neurocraft.fly.decoderHash", "4".repeat(64));
        properties.setProperty("neurocraft.fly.observationSchemaHash", "5".repeat(64));
        return properties;
    }
}
