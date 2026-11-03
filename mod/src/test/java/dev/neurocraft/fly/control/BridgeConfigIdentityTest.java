package dev.neurocraft.fly.control;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Map;
import java.util.Properties;
import org.junit.jupiter.api.Test;

final class BridgeConfigIdentityTest {
    private static final String TOKEN = "0123456789abcdef0123456789abcdef";

    @Test
    void switchesOnlyDynamicsAndEpochWhileRetainingTransportAndSecret() {
        BridgeConfig original = config();
        ControllerIdentity initial = original.identity();
        ControllerIdentity selected = new ControllerIdentity(initial.runId(), 8, initial.entityId(),
                initial.graphHash(), initial.adapterHash(), "a".repeat(64), initial.decoderHash(),
                initial.observationSchemaHash());
        BridgeConfig switched = original.withIdentity(selected);
        assertEquals(selected, switched.identity());
        assertEquals(original.uri(), switched.uri());
        assertEquals(TOKEN, switched.token());
        assertEquals(7, original.identity().sessionEpoch());
        assertFalse(switched.toString().contains(TOKEN));
        assertEquals(initial, original.withIdentity(initial).identity());
    }

    @Test
    void refusesEachFixedIdentityFieldAndNull() {
        BridgeConfig original = config();
        ControllerIdentity id = original.identity();
        for (int field : new int[] {0, 1, 2, 3, 4, 5}) {
            ControllerIdentity selected = new ControllerIdentity(
                    field == 0 ? "other" : id.runId(), 8,
                    field == 1 ? "00000000-0000-0000-0000-000000000002" : id.entityId(),
                    field == 2 ? "a".repeat(64) : id.graphHash(),
                    field == 3 ? "a".repeat(64) : id.adapterHash(), "b".repeat(64),
                    field == 4 ? "a".repeat(64) : id.decoderHash(),
                    field == 5 ? "a".repeat(64) : id.observationSchemaHash());
            assertThrows(IllegalArgumentException.class, () -> original.withIdentity(selected));
        }
        assertThrows(NullPointerException.class, () -> original.withIdentity(null));
    }

    private static BridgeConfig config() {
        Properties properties = new Properties();
        properties.setProperty("neurocraft.fly.runId", "lane-run");
        properties.setProperty("neurocraft.fly.sessionEpoch", "7");
        properties.setProperty("neurocraft.fly.entityId", "00000000-0000-0000-0000-000000000001");
        properties.setProperty("neurocraft.fly.graphHash", "1".repeat(64));
        properties.setProperty("neurocraft.fly.adapterHash", "2".repeat(64));
        properties.setProperty("neurocraft.fly.dynamicsHash", "3".repeat(64));
        properties.setProperty("neurocraft.fly.decoderHash", "4".repeat(64));
        properties.setProperty("neurocraft.fly.observationSchemaHash", "5".repeat(64));
        return BridgeConfig.from(Map.of("NEUROCRAFT_RUN_TOKEN", TOKEN), properties);
    }
}
