package dev.neurocraft.fly.control;

import java.util.Objects;
import java.util.function.Function;
import java.util.function.LongSupplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Server-thread owner of exact epochs and one replaceable authenticated bridge. */
public final class ComparisonLaneSession {
    private static final Logger LOGGER = LoggerFactory.getLogger("neurocraft-fly-comparison");
    private final LaneCatalog catalog;
    private final Function<BridgeConfig, ControllerBridge> bridgeFactory;
    private final LongSupplier clock;
    private BridgeConfig config;
    private ControllerBridge bridge;
    private LaneCatalog.Entry attempted;
    private ComparisonLanePresentation.Status phase = ComparisonLanePresentation.Status.STARTING;
    private long startedNanos;

    public ComparisonLaneSession(LaneCatalog catalog, BridgeConfig initial, ControllerBridge bridge,
            Function<BridgeConfig, ControllerBridge> bridgeFactory, LongSupplier clock) {
        this.catalog = Objects.requireNonNull(catalog, "catalog");
        this.config = Objects.requireNonNull(initial, "initial");
        this.bridge = Objects.requireNonNull(bridge, "bridge");
        this.bridgeFactory = Objects.requireNonNull(bridgeFactory, "bridgeFactory");
        this.clock = Objects.requireNonNull(clock, "clock");
        attempted = catalog.entry("original");
        if (!initial.identity().equals(attempted.identity())) {
            throw new IllegalArgumentException("initial bridge must match the verified original lane");
        }
        startedNanos = clock.getAsLong();
    }

    public BridgeConfig config() { return config; }
    public ControllerBridge bridge() { return bridge; }

    public boolean pending() {
        refresh();
        return phase == ComparisonLanePresentation.Status.STARTING
                || phase == ComparisonLanePresentation.Status.SWITCHING;
    }

    public ComparisonLanePresentation presentation() {
        refresh();
        return phase == ComparisonLanePresentation.Status.ACTIVE
                ? ComparisonLanePresentation.active(attempted, config.identity())
                : ComparisonLanePresentation.waiting(phase, attempted.name(), config.identity().sessionEpoch());
    }

    /** Validate before retiring; after epoch consumption, every failure remains neutral. */
    public void switchTo(String name, Runnable clearSessionState) {
        LaneCatalog.Entry next = catalog.entry(name);
        if (pending()) {
            throw new IllegalStateException("another comparison handshake is pending");
        }
        long epoch = Math.incrementExact(config.identity().sessionEpoch());
        BridgeConfig nextConfig = config.withIdentity(next.withEpoch(epoch));
        ControllerBridge previous = bridge;
        config = nextConfig; // Consume before close, state reset, allocation or connection attempt.
        attempted = next;
        phase = ComparisonLanePresentation.Status.SWITCHING;
        startedNanos = clock.getAsLong();
        bridge = new NeutralBridge(ControllerFault.INTERNAL_ERROR);
        diagnostic("switch_begin", null, null);
        try {
            try {
                diagnostic("previous_bridge_close_begin", null, null);
                previous.close(); // NeuralBridge.close is nonblocking and retires its own queues.
                diagnostic("previous_bridge_close_returned", null, null);
            } finally {
                diagnostic("body_session_reset_begin", null, null);
                clearSessionState.run();
                diagnostic("body_session_reset_complete", null, null);
            }
            bridge = Objects.requireNonNull(bridgeFactory.apply(nextConfig), "new bridge");
            diagnostic("new_bridge_start_begin", null, null);
            bridge.start();
            diagnostic("new_bridge_start_returned", null, null);
        } catch (RuntimeException exception) {
            diagnostic("switch_failed", exception, ControllerFault.INTERNAL_ERROR);
            try {
                bridge.close();
            } catch (RuntimeException ignored) {
                // The new owner below is neutral even if transport cleanup failed.
            } finally {
                bridge = new NeutralBridge(ControllerFault.INTERNAL_ERROR);
                phase = ComparisonLanePresentation.Status.FAILED;
            }
        }
    }

    private void refresh() {
        if (phase == ComparisonLanePresentation.Status.FAILED) {
            return;
        }
        ControllerFault state = bridge.pendingTerminalFault().orElse(bridge.state());
        if (state.isPaused()) {
            diagnostic("bridge_terminal_observed", null, state);
            phase = ComparisonLanePresentation.Status.FAILED;
        } else if (state == ControllerFault.ACTIVE) {
            // NeuralBridge publishes ACTIVE only after full identity and HMAC verification.
            if (phase != ComparisonLanePresentation.Status.ACTIVE) {
                diagnostic("bridge_active_observed", null, state);
            }
            phase = ComparisonLanePresentation.Status.ACTIVE;
        } else if (clock.getAsLong() - startedNanos >= BridgeConfig.CONNECT_TIMEOUT.toNanos()) {
            diagnostic("handshake_watchdog_expired", null, ControllerFault.DISCONNECTED);
            bridge.failClosed(ControllerFault.DISCONNECTED);
            bridge.close();
            phase = ComparisonLanePresentation.Status.FAILED;
        }
    }

    /** Same monotonic origin as the existing handshake watchdog, never a neural latency. */
    private void diagnostic(String stage, RuntimeException error, ControllerFault fault) {
        try {
            String errorClass = error == null ? "none" : error.getClass().getSimpleName();
            LOGGER.warn("comparison_lifecycle epoch={} stage={} handover_elapsed_ms={} error_class={} fault={}",
                    config.identity().sessionEpoch(), stage, (clock.getAsLong() - startedNanos) / 1_000_000L,
                    errorClass.substring(0, Math.min(errorClass.length(), 80)),
                    fault == null ? "none" : fault.name());
        } catch (RuntimeException ignored) {
            // A diagnostic sink cannot change session ownership or the fail-closed result.
        }
    }
}
