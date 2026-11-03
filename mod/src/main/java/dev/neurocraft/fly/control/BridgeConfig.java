package dev.neurocraft.fly.control;

import java.net.URI;
import java.time.Duration;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.UUID;
import java.util.regex.Pattern;

/** Strict local-only bridge configuration. Secret material is intentionally omitted from toString. */
public final class BridgeConfig {
    public static final Duration CONNECT_TIMEOUT = Duration.ofSeconds(3);
    private static final Pattern TOKEN = Pattern.compile("[A-Za-z0-9_-]{32,256}");

    private final URI uri;
    private final String token;
    private final ControllerIdentity identity;

    private BridgeConfig(URI uri, String token, ControllerIdentity identity) {
        this.uri = uri;
        this.token = token;
        this.identity = identity;
    }

    public static BridgeConfig load() {
        return from(System.getenv(), System.getProperties());
    }

    public static BridgeConfig from(Map<String, String> environment, Properties properties) {
        Objects.requireNonNull(environment, "environment");
        Objects.requireNonNull(properties, "properties");
        URI uri = URI.create(properties.getProperty(
                "neurocraft.fly.bridgeUri", "ws://127.0.0.1:8765/control"));
        validateUri(uri);
        String token = environment.get("NEUROCRAFT_RUN_TOKEN");
        if (token == null || !TOKEN.matcher(token).matches()) {
            throw new IllegalArgumentException(
                    "NEUROCRAFT_RUN_TOKEN must contain 32-256 URL-safe characters");
        }
        ControllerIdentity identity = new ControllerIdentity(
                required(properties, "neurocraft.fly.runId"),
                parseNonNegativeLong(properties, "neurocraft.fly.sessionEpoch"),
                required(properties, "neurocraft.fly.entityId"),
                required(properties, "neurocraft.fly.graphHash"),
                required(properties, "neurocraft.fly.adapterHash"),
                required(properties, "neurocraft.fly.dynamicsHash"),
                required(properties, "neurocraft.fly.decoderHash"),
                required(properties, "neurocraft.fly.observationSchemaHash"));
        try {
            UUID.fromString(identity.entityId());
        } catch (IllegalArgumentException exception) {
            throw new IllegalArgumentException(
                    "neurocraft.fly.entityId must be the UUID of the authoritative Minecraft entity",
                    exception);
        }
        return new BridgeConfig(uri, token, identity);
    }

    public URI uri() {
        return uri;
    }

    public ControllerIdentity identity() {
        return identity;
    }

    /** Retain the transport secret and fixed boundary while selecting dynamics and epoch. */
    public BridgeConfig withIdentity(ControllerIdentity selected) {
        Objects.requireNonNull(selected, "selected");
        if (!LaneCatalog.sameBoundary(identity, selected)) {
            throw new IllegalArgumentException("bridge identity can change only dynamics and epoch");
        }
        return new BridgeConfig(uri, token, selected);
    }

    String token() {
        return token;
    }

    @Override
    public String toString() {
        return "BridgeConfig[uri=" + uri + ", token=<redacted>, identity=" + identity + "]";
    }

    private static void validateUri(URI uri) {
        String host = uri.getHost();
        if (!"ws".equals(uri.getScheme())
                || host == null
                || !("127.0.0.1".equals(host) || "::1".equals(host) || "[::1]".equals(host))
                || uri.getPort() < 1
                || uri.getPort() > 65_535
                || uri.getUserInfo() != null
                || uri.getQuery() != null
                || uri.getFragment() != null
                || !"/control".equals(uri.getPath())) {
            throw new IllegalArgumentException(
                    "bridge URI must be ws://127.0.0.1:<port>/control or ws://[::1]:<port>/control");
        }
    }

    private static String required(Properties properties, String key) {
        String value = properties.getProperty(key);
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("missing required property " + key);
        }
        return value;
    }

    private static long parseNonNegativeLong(Properties properties, String key) {
        try {
            long value = Long.parseLong(required(properties, key));
            if (value < 0) {
                throw new IllegalArgumentException(key + " must be non-negative");
            }
            return value;
        } catch (NumberFormatException exception) {
            throw new IllegalArgumentException(key + " must be an integer", exception);
        }
    }

}
