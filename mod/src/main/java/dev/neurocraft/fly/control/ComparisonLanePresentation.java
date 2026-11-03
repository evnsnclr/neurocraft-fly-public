package dev.neurocraft.fly.control;

import java.math.BigInteger;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/** One atomic presentation value; active metadata exists only after authenticated HelloAck. */
public record ComparisonLanePresentation(
        Status status, String attemptedLane, long epoch, Optional<ControllerIdentity> activeIdentity,
        String activeLane, String controlKind, String controlSha256, String seed) {
    public enum Status { UNAVAILABLE, STARTING, SWITCHING, ACTIVE, FAILED }
    private static final String SCHEMA = "neurocraft.comparison_status.v1";
    private static final int MAX_BYTES = 4096;
    private static final Set<String> ID_KEYS = Set.of("run_id", "session_epoch", "entity_id",
            "graph_hash", "adapter_hash", "dynamics_hash", "decoder_hash", "observation_schema_hash");

    public ComparisonLanePresentation {
        Objects.requireNonNull(status, "status");
        Objects.requireNonNull(activeIdentity, "activeIdentity");
        if (epoch < 0 || (status == Status.UNAVAILABLE ? !"".equals(attemptedLane)
                : !LaneCatalog.NAMES.contains(attemptedLane))) {
            throw new IllegalArgumentException("invalid comparison status envelope");
        }
        if (status == Status.ACTIVE) {
            if (activeIdentity.isEmpty() || activeIdentity.orElseThrow().sessionEpoch() != epoch
                    || !attemptedLane.equals(activeLane)
                    || !("original".equals(activeLane) ? "faithful" : activeLane).equals(controlKind)) {
                throw new IllegalArgumentException("active comparison metadata must match its lane");
            }
            if ("original".equals(activeLane)) {
                if (controlSha256 != null || seed != null) {
                    throw new IllegalArgumentException("original wiring carries no control overlay");
                }
            } else if (controlSha256 == null || !controlSha256.matches("[0-9a-f]{64}")
                    || seed == null || !seed.matches("0|[1-9][0-9]{0,19}")
                    || new BigInteger(seed).bitLength() > 64) {
                throw new IllegalArgumentException("invalid active control metadata");
            }
        } else if (activeIdentity.isPresent() || activeLane != null || controlKind != null
                || controlSha256 != null || seed != null) {
            throw new IllegalArgumentException("unmatched status cannot carry active lane metadata");
        }
    }

    public static ComparisonLanePresentation unavailable() {
        return new ComparisonLanePresentation(Status.UNAVAILABLE, "", 0, Optional.empty(),
                null, null, null, null);
    }

    public static ComparisonLanePresentation waiting(Status status, String lane, long epoch) {
        return new ComparisonLanePresentation(status, lane, epoch, Optional.empty(),
                null, null, null, null);
    }

    public static ComparisonLanePresentation active(LaneCatalog.Entry entry, ControllerIdentity id) {
        return new ComparisonLanePresentation(Status.ACTIVE, entry.name(), id.sessionEpoch(),
                Optional.of(id), entry.name(), entry.controlKind(), entry.controlSha256(), entry.seed());
    }

    public String encode() {
        Map<String, Object> values = new LinkedHashMap<>();
        values.put("schema", SCHEMA);
        values.put("status", status.name());
        values.put("attempted_lane", attemptedLane);
        values.put("epoch", epoch);
        values.put("active_identity", activeIdentity.map(id -> Map.<String, Object>of(
                "run_id", id.runId(), "session_epoch", id.sessionEpoch(), "entity_id", id.entityId(),
                "graph_hash", id.graphHash(), "adapter_hash", id.adapterHash(),
                "dynamics_hash", id.dynamicsHash(), "decoder_hash", id.decoderHash(),
                "observation_schema_hash", id.observationSchemaHash())).orElse(null));
        values.put("active_lane", activeLane);
        values.put("control_kind", controlKind);
        values.put("control_sha256", controlSha256);
        values.put("seed", seed);
        return StrictJson.encode(values, MAX_BYTES);
    }

    public static ComparisonLanePresentation parseOrUnavailable(String text) {
        try {
            Map<String, Object> values = StrictJson.parseObject(text, MAX_BYTES);
            BrainDisplayFrame.exact(values, Set.of("schema", "status", "attempted_lane", "epoch",
                    "active_identity", "active_lane", "control_kind", "control_sha256", "seed"));
            if (!SCHEMA.equals(values.get("schema"))) {
                return unavailable();
            }
            Optional<ControllerIdentity> identity = Optional.empty();
            if (values.get("active_identity") != null) {
                Map<String, Object> id = BrainDisplayFrame.object(values, "active_identity");
                BrainDisplayFrame.exact(id, ID_KEYS);
                identity = Optional.of(new ControllerIdentity(
                        BrainDisplayFrame.text(id, "run_id"), BrainDisplayFrame.integer(id, "session_epoch"),
                        BrainDisplayFrame.text(id, "entity_id"), BrainDisplayFrame.text(id, "graph_hash"),
                        BrainDisplayFrame.text(id, "adapter_hash"), BrainDisplayFrame.text(id, "dynamics_hash"),
                        BrainDisplayFrame.text(id, "decoder_hash"), BrainDisplayFrame.text(id, "observation_schema_hash")));
            }
            return new ComparisonLanePresentation(
                    Status.valueOf(BrainDisplayFrame.text(values, "status")),
                    BrainDisplayFrame.text(values, "attempted_lane"), BrainDisplayFrame.integer(values, "epoch"),
                    identity, nullable(values, "active_lane"), nullable(values, "control_kind"),
                    nullable(values, "control_sha256"), nullable(values, "seed"));
        } catch (ProtocolException | RuntimeException exception) {
            return unavailable();
        }
    }

    private static String nullable(Map<String, Object> values, String key) throws ProtocolException {
        return values.get(key) == null ? null : BrainDisplayFrame.text(values, key);
    }
}
