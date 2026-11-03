# EEM escape-reflex identity — TTMn jump readout, v1

- **Status: recorded engineering-demonstration identity, not a statistical
  protocol — frozen and committed BEFORE any measurement.** This document
  exists to satisfy ADR 0008's rule that Engineered-Encoder Mode identities
  are recorded before anything runs under them, and the D-058/D-059
  freeze-then-measure process rule: the identity text and the tunable code
  are frozen and committed first, and every measurement follows in a separate
  phase whose results are recorded separately and never edit this document.
  It declares a *demonstration*: no hypothesis test, no preregistered power,
  no confirmatory claim. As of this document's commit, **no escape-profile
  measurement of any kind has been run**: no calibration, no probe, no live
  session, and no TTMn selection against the real table under this lane's
  code. The one prior quantitative input — a scratchpad feasibility
  measurement recorded in D-060 — is disclosed there as *not*
  content-addressed evidence; phase 2 re-derives it under a committed
  receipt.
- **Identity:** `eem-escape-v1`, service profile `full_mcns_escape_v1`.
  Mode: Engineered-Encoder Mode (ADR 0008, Mode 3 of 3). Never described as
  Faithful Organism Mode; every artifact, overlay, and caption produced under
  this identity carries the engineered-encoder label, and the configuration
  identity pins `mode_taxonomy: "engineered_encoder"` under the schema
  `neurocraft.controller.full_brain_escape_config.v1`.

## Motivation — the recorded D-060 observations and pivot

Two frozen encoder designs (eem v1, D-058; eem v2, D-059/D-060) both measure
continuous steering through the 52-neuron wing readout an order of magnitude
below visually evident amplitude: v1's sided yaw signature is
+0.0054/−0.0346, v2's approach ramp peaks at yaw ≈ 0.016 before contact and
its held conditions adapt toward a non-sided residual (−0.0143/−0.0207),
while touch remains the strongest response (+0.32 forward). On those
observations D-060 records the engineering decision this document executes:
the showcase lane pivots to the escape reflex arc. Informing that choice,
D-060 also records a scratchpad feasibility measurement (disclosed as not
content-addressed evidence; re-derived under a committed receipt in
phase 2): under the already-declared looming encoder, DNp01 giant-fiber
activity rises ~300× above the wing-pool means, the TTMn jump-motoneuron
stage attenuates by 1.40% — matching the census's 1.49% anatomical fraction
almost exactly — and a TTMn-pair readout under the existing calibration
methodology would show 45–76× peak-over-baseline event contrast (vs ~7×
through the wings), with the full arc completing within one game tick.
Every other quantity the scratchpad run observed is deliberately NOT
restated here: the D-060 record above is the only quantitative motivation
this document rests on, and phase 2's feasibility receipt re-derives the
full set under committed evidence.

**Encoder choice, declared with its rationale:** the escape profile is
defined over the **eem v1 ABSOLUTE encoder — v1, not v2**. The escape arc
scores a looming event against a near-silent motor stage; v2's temporal
contrast adapts a held threat away (that is what D-060 measured), which is
the wrong boundary for an arc whose interesting signal is the event itself
arriving at an otherwise quiet readout.

## Frozen core (identical to the eem-v1 profile)

The verified full MaleCNS runtime artifact (manifest `b33bdda8…4f67c98c`,
166,700 neurons / 25,582,938 edges), the disclosed
`coarse_presynaptic_consensus` sign overlay (`SIGN-OVERLAY-02`), the
array-native backend under its own dynamics identity, the two faithful
sensory entries via the pinned channel builder (retained byte-identically),
the declared 25+27 somaSide wing-motor pools with a-priori gain 8.0 / dead
zone 0.0, and the pinned primitive projection. The **sensory boundary is
exactly eem v1's**: the same populations, selectors, counts, and membership
(hashed population binding
`bf45222dd478ecb5dcb35df8ed34e3dd2193449f0d8fbd5f931a24319d023bae`), the
same five absolute channels with all gains 1.0 under the same v1 encoder
document (schema `neurocraft.eem.encoder.v1`, identity
`e1bb7a1fa9838e890e543e1f83b0e605c04657bd65d9051cb5ddba208e25d901`), the
same ray-side convention (front = 0, left = 2, right = 3) and per-capita
drive semantics, and the same v1 per-step stimulation provenance
(`neurocraft.full_brain.encoded_stimulation.eem.v1`). Because the sensory
boundary is unchanged, the **adapter identity is byte-identical to
eem-v1's by construction** — the rotation lives entirely in the decoder and
configuration identities below. Nothing about the dynamics, the wing readout
weights, the faithful channels, or the population membership is new,
trained, or tuned; no cross-tick encoder state exists. The wing, eem-v1, and
eem-v2 profiles are regression-pinned byte-unchanged by
`tests/controller/test_embodied_senses.py`,
`tests/controller/test_embodied_senses_v2.py`, and
`tests/controller/test_escape_reflex.py`.

## The escape addition (the declared change)

**Census anatomy** (D-050-era readout census, `READOUT-CENSUS-01`, restated
in D-060): all 126 LC4 looming detectors project to the DNp01 giant-fiber
pair, which projects to the TTMn jump motoneurons. **DNp01 is never read
directly** — it stays interior recurrent context; the declared readout is
the motor stage.

**TTMn pool selection** — from the pinned body-annotations table
`body-annotations-male-cns-v1.0-minconf-0.5.feather` (manifest-declared
SHA-256 `2177e246…a9a3b2`), joined through the artifact's
`annotation_source_rows` with the same integrity gates the embodied
populations use (digest match, complete coverage, bitwise bodyId identity,
unique in-range pointers), fail-closed on any drift:

| pool | selector | expected |
|---|---|---|
| `jump_motor` | `type == 'TTMn'` | EXACTLY 2 members, somaSide split 1 L / 1 R |

The census records bodies **800146 (R)** and **804642 (L)**, class
`vnc_motor` / subclass `wm`; the loader refuses any count or side drift
rather than renormalizing, and additionally refuses membership overlap with
the 52 declared wing-motor bodies, with every embodied sensory population,
and with the two faithful sensory entries. Neither TTMn body is a member of
the current 52-neuron wing readout. Membership is frozen into the hashed
pool-binding document (schema `neurocraft.escape.ttmn_pool_binding.v1`);
its identity depends on the artifact's dense indices and is recorded with
the phase-2 calibration, not guessed here.

**Jump readout** — unit weights, mean of the pair, the a-priori gain, one
declared scale (verbatim from
`neurocraft_controller.escape_reflex.ESCAPE_JUMP_FORMULA`):

```
jump_raw = motor_pool_gain * (fsum(TTMn activity) / 2) * jump_scale
jump_value = clip(jump_raw, -1, 1), zeroed when |clip(jump_raw, -1, 1)| < motor_dead_zone
```

with `motor_pool_gain` = 8.0 and `motor_dead_zone` = 0.0 exactly as the wing
pools declare, and `jump_scale` measured once by this profile's own
calibration (below). The arithmetic mirrors the wing decode bit-for-bit
(mean member activity × gain × scale, then the pinned clip and dead zone).

**Declared projection extension** — the pinned primitive projection
(`neurocraft.primitive_projection.v1`, untouched) produces
(forward, lateral, vertical ≡ forward, yaw, pitch) from the two wing pools.
The escape profile's vertical axis becomes

```
vertical_thrust = clip(vertical_projection + jump_value, -1, 1)
```

**one declared additive term on one axis; forward, lateral, yaw, and pitch
are passed through bit-for-bit.** The wing MotorFrame itself is unchanged;
the jump value is recorded per step in the controller evidence
(`jump_value`), and the per-step evidence subset (`state_after_activity`)
appends the two TTMn members after the wing pools (56 values instead of 54
for the real artifact) — a disclosed, escape-only extension.

**Identity chaining.** The configuration identity is under
`neurocraft.controller.full_brain_escape_config.v1` with the v1 encoder
identity and the declared `jump_scale` chained in; the uncalibrated
configuration identity is
`54814d90b296949c2e92e712ab4c5228d4fd430d4a18e6fe03874f5af8b987da`
(derivable without the artifact; the calibrated configuration and decoder
identities incorporate the phase-2 calibration digest and TTMn membership
and will be recorded with it). The decoder identity is under
`neurocraft.full_brain.decoder.escape.v1` and chains: the untouched wing
readout document (which itself binds the sign overlay, the pinned
projection, the wing scale, and the calibration digest) + the TTMn pool
membership + the jump mapping formula verbatim + the declared `jump_scale` +
the calibration document digest + the encoder identity. A different
membership, formula, scale, encoder, or calibration is a different decoder.
An escape configuration without the escape binding, or the reverse, or an
escape binding over the v2 encoder document, refuses construction.

## Phase-2 measurement protocol (declared here, not yet run)

1. **Feasibility receipt.** Re-derive D-060's scratchpad feasibility
   numbers from the frozen tree under a committed content-addressed receipt
   in `docs/receipts/` (a new tracked script, `scripts/measure_escape_arc.py`,
   byte-idempotent evidence like the sign-overlay censuses), so the pivot's
   quantitative motivation stops resting on a scratchpad. Pinned method:
   30-tick holds from fresh reset for baseline all-open (rays 6.0, light 15,
   no collision), blocked-front (front ray 0.5), and all-rays-0.5, on the
   calibrated escape controller, recording per-tick DNp01 L/R and TTMn L/R
   activity, the LC4 population mean, and the wing pool means; plus a
   150-step direct-drive diagnostic (LC4 = 1.0 dense stimulation on a raw
   engine, labeled diagnostic-only). Quantities, defined exactly:
   DNp01 response ratio = |mean DNp01 pair activity over the last 10 ticks,
   condition minus baseline| / |same for the wing-pool mean|; TTMn
   attenuation = TTMn-pool mean / DNp01-pair mean at the all-rays condition;
   TTMn event contrast = TTMn-pool |mean| at condition peak / baseline
   TTMn-pool |mean|; arc latency = neural steps from LC4 commit to first
   nonzero TTMn response in the direct phase. The receipt's DNp01 read is
   offline and diagnostic-only — the live decoder never reads DNp01, and
   this item does not weaken that claim boundary. Whatever the re-derivation
   shows is recorded.
2. **Boundary calibration**, measured once by
   `scripts/calibrate_full_readout.py --profile escape` under the stream
   `full-readout-calibration-escape-v1`: byte layout and declared
   distributions IDENTICAL to the eem-v1 stream — rays from two digest bytes
   each, light from byte 12 (uniform integers 0..15), horizontal/vertical
   collisions from digest bytes 13:15 / 15:17 as declared
   Bernoulli(6554/65536 ≈ 0.1000061) — under the new stream id as seed
   prefix, because the escape sensory boundary IS eem v1's. **ONE run** of
   50 warmup + 600 measured ticks measures BOTH pool groups — the pooled
   wing magnitudes AND the TTMn pool magnitudes — and derives both scales by
   the standard derivation, `scale = 0.5 / (p95 × motor_pool_gain)` per pool
   group: `wing_scale` (re-measured under this profile so one document binds
   the whole boundary) and `jump_scale`. Frozen write-or-verify into
   `data/calibration/full-readout-calibration-escape-v1.json` under the
   schema `neurocraft.full_readout_calibration.escape.v1`. Cross-binding:
   the document names this profile's encoder identity AND TTMn pool identity
   and refuses every other boundary, and every other boundary refuses it —
   the wing/eem-v1/eem-v2 loader rejects the escape schema and the escape
   loader rejects theirs, in both directions.
3. **The declared probe battery**, with the calibrated escape controller
   and, for the honest A/B, the calibrated eem-v1 and wing controllers —
   each condition held 30 ticks from a fresh reset, reporting **ALL motor
   axes including the new jump-augmented vertical** (for eem-v1 and wing the
   vertical is the pinned projection's, with no jump term):
   - **The frozen six conditions at the pinned ray values** (the values
     D-059 froze: the blocked ray held at 0.5 m, every other ray at 6.0 m,
     `ray_max_distance_m` = 6.0): all-open bright (light 15), all-open dark
     (light 0), blocked-left, blocked-right, blocked-front, collision-on.
   - **The approach ramp** (the same ramp eem v2 measured, for
     comparability): light 15, all rays open except the left ray, which
     shrinks linearly from 6.0 m at tick 0 to 0.5 m at tick 19
     (`ray_left(t) = 6.0 − (6.0 − 0.5)·t/19`), then holds at 0.5 m for the
     remaining 10 ticks.

**No pass/fail criterion is declared — explicitly.** Whatever is measured is
reported, in full, against this identity. There is no threshold whose
crossing would make this demonstration "succeed" or "fail", and any go/no-go
decision about a live demo under this profile is an engineering decision
made and recorded AFTER the measurements, on the record — the D-060 lesson,
where an independent verification had to expunge a criterion that existed
nowhere in the frozen record. No parameter is changed in response to what
any phase-2 measurement shows; the escape profile is what this document
says it is. Phase-2 results are recorded in a separate results document
referencing this identity.

## Live wiring

Opt-in service profile `full_mcns_escape_v1` (refusals identical to
`full_mcns_eem_v1`: no traces, no lesions, no fixture, no presentation
telemetry; local-only artifact and body-annotations table required), loading
the escape calibration by default and refusing every other profile's
calibration; until the phase-2 calibration exists the service refuses to
start, naming `scripts/calibrate_full_readout.py --profile escape`
(`--full-uncalibrated` opts out explicitly). Launcher row with fixture
`None` and description "166,700-neuron full MaleCNS with declared embodied
senses and escape-reflex jump readout (EEM escape, local-only)";
`make live-demo-escape`. The wing, eem-v1, eem-v2, TRM, steering, and pair
profiles are byte-identical before and after this change.

## Claim boundary — what is and is not claimed

Supported by a successful demonstration: *"a declared, engineered sensory
front end drives labeled sensory populations of the measured circuit, whose
untouched dynamics are read at a declared jump-motoneuron stage through a
declared boundary mapping onto one motor axis"* — Engineered-Encoder Mode
engineering evidence, nothing more.

Never supported, and disclosed plainly:

- **"Jump" names a declared boundary mapping** from jump-motoneuron pool
  activity to Minecraft vertical thrust — an engineering convention, hashed
  before any measurement. It is **not** a claim that the organism decided,
  intended, chose, or performed an escape, and no such language attaches to
  this identity.
- **Body policy, disclosed:** the fly's Minecraft body renders a TTMn
  activity spike as upward thrust. The real tergotrochanteral muscles
  extend the mesothoracic legs; a Minecraft fly has no legs to extend, and
  "upward thrust" is this project's declared body-model convention for the
  jump motor stage, not physiology.
- **DNp01 is never read directly.** The giant fiber stays interior
  recurrent context; only the TTMn motor stage is read, through the declared
  pool. No claim about giant-fiber spiking, latency classes, or the
  biological escape pathway's dynamics is made or supported.
- **No biological encoding claim.** Ray proximity is not looming (LC4's
  biological selectivity is for expanding retinal images), uniform
  per-capita drive is not retinotopy, and the eem v1 caveats apply verbatim
  to the unchanged sensory boundary.
- **No faithful-mode claim.** The sentence "nothing else computes in the
  loop" belongs to Faithful Organism Mode only and is never blended with
  this identity (ADR 0008 headline-language rule).
- **No behavior/performance claim in advance.** That the escape arc
  "should" produce a visible response is the motivation for the design, not
  a claim; a fly that never visibly jumps is a recorded observation, exactly
  as the v1 straight-line fly was.
- The coarse sign overlay's caveats (`SIGN-OVERLAY-02`) apply unchanged.
