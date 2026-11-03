# EEM embodied-senses identity — labeled-population drive, v1

- **Status: recorded engineering-demonstration identity, not a statistical
  protocol.** This document exists to satisfy ADR 0008's rule that
  Engineered-Encoder Mode identities are recorded before anything runs under
  them. It declares a *demonstration*: no hypothesis test, no preregistered
  power, no confirmatory claim. The only measurements it promises are
  descriptive offline probes and one boundary calibration, reported as
  observed.
- **Identity:** `eem-embodied-senses-v1`, service profile `full_mcns_eem_v1`.
  Mode: Engineered-Encoder Mode (ADR 0008, Mode 3 of 3). Never described as
  Faithful Organism Mode; every artifact, overlay, and caption produced under
  this identity carries the engineered-encoder label, and the configuration
  identity itself pins `mode_taxonomy: "engineered_encoder"` under the schema
  `neurocraft.controller.full_brain_eem_config.v1`.
- **Purpose:** the live full-brain wing profile (`full_mcns_wing_v1`) senses
  the world through exactly two mechanosensory entries (JO-EV3 wind body
  77185 and SNpp10 proprioceptor body 805722) and, in sustained live sessions
  (2026-08-30), circles leftward. This identity gives the same untouched
  brain and the same untouched wing readout a declared, engineered sensory
  front end over the artifact's *labeled* sensory populations, so Minecraft's
  existing observations reach the parts of the brain the source annotations
  label as sensory. Whether behavior changes is what the demonstration
  measures — nothing here presumes it.

## Frozen core (identical to the wing profile)

The verified full MaleCNS runtime artifact (manifest
`b33bdda8…4f67c98c`, 166,700 neurons / 25,582,938 edges), the disclosed
`coarse_presynaptic_consensus` sign overlay (`SIGN-OVERLAY-02`), the
array-native backend under its own dynamics identity, the two faithful
sensory entries via the pinned channel builder (retained byte-identically and
placed exactly as today), the declared 25+27 somaSide wing-motor pools with
a-priori gain 8.0 / dead zone 0.0, and the pinned primitive projection.
Nothing about the dynamics, the readout weights, or the faithful channels is
new, trained, or tuned. The wing profile itself is regression-pinned: its
adapter, decoder, and configuration identities are asserted byte-unchanged by
`tests/controller/test_embodied_senses.py`.

## The engineered encoder (the declared addition)

**Populations** — selected from the pinned body-annotations table
`body-annotations-male-cns-v1.0-minconf-0.5.feather` (manifest-declared
SHA-256 `2177e246…a9a3b2`), joined through the artifact's
`annotation_source_rows` with the census integrity gates (digest match,
bitwise bodyId identity, unique in-range pointers), fail-closed on any count
drift:

| population | selector | count |
|---|---|---|
| `lc4_left` | `type == 'LC4' and somaSide == 'L'` | 71 |
| `lc4_right` | `type == 'LC4' and somaSide == 'R'` | 55 |
| `visual` | `class == 'visual'` | 6,091 |
| `bristle` | `subclass == 'mechanosensory bristle'` | 2,206 |

The side column is **somaSide**, not the census-guessed `rootSide`:
measured against the pinned table, `rootSide` is null for all 126 LC4 rows
while `somaSide` carries exactly the 71 L / 55 R split (the `instance`
column concurs: `LC4_L` 71 / `LC4_R` 55). This deviation from the plan's
guess is recorded here. The four populations are pairwise disjoint (measured
pairwise overlaps: all zero) and contain neither faithful entry body; any
overlap refuses construction. Membership is frozen into the hashed
population-binding document (schema `neurocraft.eem.population_binding.v1`,
identity `bf45222dd478ecb5dcb35df8ed34e3dd2193449f0d8fbd5f931a24319d023bae`).

**Ray-side convention** — protocol ray order is
`forward, back, left, right, up, down`; verified in source:
`FlySensors.java` casts `forward`, `forward.reverse()`, `right.reverse()`
(labeled left), `right`, `bodyUp`, `bodyUp.reverse()`, and the pinned
`PhysicalFeatureEncoder` schema pins the same order. Declared indices:
front = 0, left = 2, right = 3. Each side's "max over side rays" is a max
over that singleton set.

**Channels** (schema `neurocraft.eem.encoder.v1`, identity
`e1bb7a1fa9838e890e543e1f83b0e605c04657bd65d9051cb5ddba208e25d901`), all
values in [0, 1], all per-channel gains **1.0** (v1 declares no tuning; the
half amplitude of `threat_front` is part of its formula, not a hidden gain):
The `/2` reflects that a frontal object partially occupies both visual
hemifields rather than either fully — an engineering choice fixed before any
measurement, declared here rather than justified from physiology.

| channel | target(s) | formula |
|---|---|---|
| `threat_left` | `lc4_left` | `clip(1 − ray[2]/ray_max, 0, 1)` |
| `threat_right` | `lc4_right` | `clip(1 − ray[3]/ray_max, 0, 1)` |
| `threat_front` | `lc4_left` + `lc4_right` | `clip(1 − ray[0]/ray_max, 0, 1) / 2`, added to both sides |
| `light` | `visual` | `local_light / 15` |
| `touch` | `bristle` | `1.0 if horizontal_collision or vertical_collision else 0.0` |

**Drive semantics** — per-capita: every member of a target population
receives the identical channel value times the declared gain; contributions
to one population are summed and clipped to [0, 1]; there are no per-member
weights. Values are placed on the dense sensory vector exactly as the
faithful channels are (the engine applies its uniform sensory gain inside
the step). The faithful JO-EV3/SNpp10 channels are retained unchanged and
placed exactly as today; EEM adds on top of disjoint indices.

**Identity chaining** — the adapter identity becomes a derived SHA-256 over
{pinned feature schema, faithful sensory binding, population binding,
encoder document} under schema `neurocraft.boundary_adapter.identity.eem.v1`;
the per-step stimulation provenance chains the faithful assignments, the
channel values, and the per-population drives under
`neurocraft.full_brain.encoded_stimulation.eem.v1`. The per-step evidence
subset (`state_after_activity`) is deliberately unchanged from the wing
profile — copying ~8.4k population members per tick is not cheap, membership
is pinned by the binding hash, and the drives are recorded per step under
`eem:`/`eem_drive:` keys.

## Boundary calibration (declared, measured once)

The EEM drive moves the readout operating range (6k+ driven neurons vs 2),
so the profile carries its own declared readout calibration, measured by
`scripts/calibrate_full_readout.py --profile eem` under the stream
`full-readout-calibration-eem-v1`: rays and light exactly as the wing stream
(two digest bytes per ray, byte 12 for light — uniform integers 0..15),
plus collision flags from digest bytes 13:15 / 15:17 as declared
Bernoulli(6554/65536 ≈ 0.1000061, the closest two-byte approximation of the
declared p = 0.1). Same derivation as the wing document:
`readout_scale = 0.5 / (p95 × motor_pool_gain)`, one scalar, both pools,
frozen write-or-verify into the committed
`data/calibration/full-readout-calibration-eem-v1.json`.

Measured 2026-08-30 (reported as observed): n=1200 pooled magnitudes,
p50 = 3.728112e-04, p95 = 8.183196e-04, p99 = 9.715750e-04,
max = 1.094947e-03, derived `readout_scale` = 76.376026 (saturating scale
114.160758 disclosed, not applied); document SHA-256
`146abef92a54b405291c2d52ffa37ce46fc52096c5463b8a2d4a626c585def98`. The
document carries an `encoder` section binding the encoder-document identity,
and `assert_binds` refuses a wing calibration on the EEM boundary and an EEM
calibration on the wing boundary, in both directions.

## Offline behavior probes (descriptive, declared before measurement)

With the calibrated EEM controller and, for the honest A/B, the calibrated
wing controller: hold each condition 30 ticks from a fresh reset and report
all motor axes — all-open bright (light 15), all-open dark (light 0),
blocked-left (left ray short), blocked-right, blocked-front, collision-on.
The question of interest, reported as observed and not tuned against: does
yaw differ in sign or strongly in magnitude between blocked-left and
blocked-right (sided LC4 drive breaking the circling symmetry), and does
light level modulate anything? No parameter is changed in response to what
the probes show; v1 is what it is.

## Live wiring

Opt-in service profile `full_mcns_eem_v1` (same refusals as
`full_mcns_wing_v1`: no traces, no lesions, no fixture, no presentation
telemetry; local-only artifact required), loading the EEM calibration by
default and refusing the wing calibration; launcher row with fixture `None`
and description "166,700-neuron full MaleCNS with declared embodied senses
(EEM, local-only)"; `make live-demo-eem`. The wing, TRM, steering, and pair
profiles are byte-identical before and after this change.

## Claim boundary — what is and is not claimed

Supported by a successful demonstration: *"a declared, engineered sensory
front end drives labeled sensory populations of the measured circuit, whose
untouched dynamics and untouched declared readout then actuate the fly"* —
Engineered-Encoder Mode engineering evidence, nothing more.

Never supported, and disclosed plainly:

- **No biological encoding claim.** Ray proximity is not looming: LC4's
  biological selectivity is for expanding retinal images, and
  `1 − distance/ray_max` is a static proximity fraction with no expansion
  dynamics. Uniform per-capita drive is not retinotopy: real LC4 units tile
  visual space, and this encoder drives all 71 (or 55) identically.
  `local_light/15` is not phototransduction, and a collision bit is not a
  bristle deflection model. The labels locate populations; the encoder's
  meaning is engineering convention, declared and hashed.
- **No faithful-mode claim.** The sentence "nothing else computes in the
  loop" belongs to Faithful Organism Mode only and is never blended with
  this identity (ADR 0008 headline-language rule).
- **No behavior/performance claim in advance.** A fly that still circles is
  a recorded observation, exactly as the circling wing-profile fly was.
- The coarse sign overlay's caveats (`SIGN-OVERLAY-02`) apply unchanged.
