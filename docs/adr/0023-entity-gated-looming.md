# ADR 0023: Entity-gated looming

_Sensory contract, accepted 2026-09-04_

**Status:** Implemented on the delivery branch (`0440d0e`, integrated as
`6a9cbb4`). Offline verification is recorded in the
[encoder declaration](../encoders/eem-loom-v2.md). Combined delivery verification
and controlled live acceptance remain separate work.

## Context

Loom-v1 applies temporal contrast to ray proximity. Those distances cannot
distinguish the fly approaching a wall from an entity entering or approaching a
ray. The demo needs an explicit entity filter while preserving the original
encoder for comparison. This is an engineering choice about the demo's threat
input, not a biological model of looming.

## Decision

1. Add the closed `neurocraft.sensors.v2` shape with six required integer
   `ray_hit_kinds`: `0` miss, `1` block, `2` entity, ordered forward, back, left,
   right, up, down. Distance and kind come from the same nearest-hit calculation.
   Exact block/entity ties select entity; a hit at maximum range retains its kind.
   ADR 0017's entity predicate, zero hitbox inflation, and numerical distances
   remain unchanged. The frame carries no entity identity or player label.
2. Select the nested sensor shape using the authenticated session's
   `observation_schema_hash`. The new contract hash is
   `032b6a74d57b37f9fc1883b4865c89ec2eeccc5c9084831e29cd986738559308`.
   The service refuses a mismatched shape before queueing; the full-brain
   controller also refuses it before stepping. Legacy `SensorFrame`, its wire
   shape, and existing observation identities remain intact. Missing kinds are
   never inferred. Protocol/trace envelopes remain v2.
3. In `full_mcns_loom_v2` and `full_mcns_loom_program_v2`, admit raw front/left/right
   threat proximity only for an entity hit. Block and miss values become exactly
   zero **before** the inherited rectified temporal contrast. Initialization,
   adaptation, front scaling, population placement, touch, light, and the original
   physical features keep their numerical rules. Wall distances still reach the
   physical features; zero threat input does not imply zero network activity or
   zero jump readout. An entity crossing a ray causes a step and decay; movement
   toward a stationary entity also counts.
4. Preserve the v1 profiles and calibration. V2 declares distinct observation,
   encoder, configuration, stimulation, calibration, decoder, and policy bindings.
   The calibration uses the existing numeric stream and seed with explicit entity
   kinds; its measurements and wing/jump scales reproduce v1 exactly. Loaders
   refuse cross-generation calibration documents. Graph, dynamics, TTMn and DNa01
   memberships, decoder arithmetic, and policy thresholds remain unchanged.

`make live-demo-loom` and `make live-demo-program` select the v2 profiles; explicit
v1 profiles remain available. ADR 0021's hand-chosen readouts selecting scripted
programs and ADR 0022's action budget are unchanged.

## Evidence and limits

The [encoder declaration](../encoders/eem-loom-v2.md#measured-offline-probes)
records synthetic full-artifact diagnostics, their identities, and reproduction
commands. The original single-front wall trial did **not** cross the v1 escape
threshold: peak jump `0.0090688221` was below `ESCAPE_ON = 0.0113890810`.
Additional corner trials crossed it in v1. V2 threat drive was zero in these wall
trials, but peak jump remained approximately `0.00155–0.00158`, below threshold.
The entity ramp retained the v1 peak `0.02277816208187006`, with every reported
float matching by `float.hex()`. The held-wall policy battery requested CRUISE
for all 30 ticks of each condition; these are policy requests, not live receipts.

Tests cover strict wire shapes and kinds, legacy/new goldens, nearest-hit
occlusion and ties, real-tick transport, refusal before stepping, all three threat
channels, entity equivalence, and calibration cross-refusal. These establish the
declared transport and filter. They do not establish player-specific sensing,
natural behavior, or a successful live interaction.
