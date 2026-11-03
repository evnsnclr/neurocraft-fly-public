package dev.neurocraft.fly.control;

import java.util.Objects;

/**
 * Server acknowledgement for one authenticated session epoch.
 *
 * <p>{@code serverAuth} is the service's HMAC proof over the bridge's own handshake nonce. It is
 * carried unverified out of the decoder and checked by {@link NeuralBridge}, which is the only
 * component holding the run token; see
 * {@link ControlProtocol#verifyHelloAck(HelloAck, String, ControllerIdentity, String)}.
 */
public record HelloAck(ControllerIdentity identity, String serverAuth) implements InboundFrame {
    public HelloAck {
        Objects.requireNonNull(identity, "identity");
        Objects.requireNonNull(serverAuth, "serverAuth");
        if (!serverAuth.matches("[0-9a-f]{64}")) {
            throw new IllegalArgumentException("server_auth must be a 64-character lowercase hex digest");
        }
    }
}
