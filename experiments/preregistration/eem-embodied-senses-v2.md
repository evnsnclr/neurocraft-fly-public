# EEM embodied-senses identity — temporal-contrast drive, v2

- **Status: recorded engineering-demonstration identity, not a statistical
  protocol — frozen and committed BEFORE any measurement.** This document
  exists to satisfy ADR 0008's rule that Engineered-Encoder Mode identities
  are recorded before anything runs under them, and the D-058 process rule
  that EEM identity text is frozen and committed before any measurement is
  taken (v1's prose was finalized minutes after its calibration ran; this
  document closes that gap by construction). It declares a *demonstration*:
  no hypothesis test, no preregistered power, no confirmatory claim. The only
  measurements it promises are the phase-2 descriptive offline probes and one
  boundary calibration, reported as observed. As of this document's commit,
  **no eem-v2 measurement of any kind has been run**: no calibration, no
  probe, no live session. Phase-2 results will be recorded separately and
  will never edit this document.
- **Identity:** `eem-embodied-senses-v2`, service profile `full_mcns_eem_v2`.
  Mode: Engineered-Encoder Mode (ADR 0008, Mode 3 of 3). Never described as
  Faithful Organism Mode; every artifact, overlay, and caption produced under
  this identity carries the engineered-encoder label, and the configuration
  identity pins `mode_taxonomy: "engineered_encoder"` under the schema
  `neurocraft.controller.full_brain_eem_v2_config.v1`.

## Motivation — the recorded v1 diagnosis

EEM v1 (`full_mcns_eem_v1`, D-058,
`experiments/preregistration/eem-embodied-senses-v1.md`) drives its labeled
populations with ABSOLUTE signal levels. Live observation (2026-08-31, three
fault-free sessions): the fly flies in a near-straight line with
imperceptible responses — the live motors match the offline all-open probe
bit-for-bit (yaw −0.0188). Diagnosis: constant max light pins the 6,091
visual cells at drive 1.0 (a huge standing DC input), and the calibration
stream's varied light/collisions set a p95 far above the live dynamic range,
compressing real responses roughly twentyfold. The principled fix declared
here: TEMPORAL-CONTRAST channels — each channel drives its population with
the deviation of its signal from a leaky running mean (adaptation), not the
absolute level, so standing conditions adapt away and events stand out.

## Frozen core (identical to the wing and eem-v1 profiles)

The verified full MaleCNS runtime artifact (manifest `b33bdda8…4f67c98c`,
166,700 neurons / 25,582,938 edges), the disclosed
`coarse_presynaptic_consensus` sign overlay (`SIGN-OVERLAY-02`), the
array-native backend under its own dynamics identity, the two faithful
sensory entries via the pinned channel builder (retained byte-identically),
the declared 25+27 somaSide wing-motor pools with a-priori gain 8.0 / dead
zone 0.0, and the pinned primitive projection. The **populations, selectors,
counts, and membership are exactly v1's**: the hashed population-binding
document is unchanged (schema `neurocraft.eem.population_binding.v1`,
identity `bf45222dd478ecb5dcb35df8ed34e3dd2193449f0d8fbd5f931a24319d023bae`),
as are the ray-side convention (front = 0, left = 2, right = 3) and the
per-capita drive semantics. Nothing about the dynamics, the readout weights,
the faithful channels, or the membership is new, trained, or tuned. The wing
and eem-v1 profiles are regression-pinned byte-unchanged by
`tests/controller/test_embodied_senses.py` and
`tests/controller/test_embodied_senses_v2.py`.

## The temporal-contrast encoding (the declared change)

**Channels** (schema `neurocraft.eem.encoder.v2`, identity
`31c4cd68cbbe3d7afb32d53d963a237f3fe27da8023e2bfc4dc818aaf11c9cd6`): the same
five channels as v1 — same names, units, target populations, and all
per-channel gains **1.0** (v2 declares no tuning). The v1 formulas are
retained byte-identically as the RAW value `raw_t` in [0, 1]; what changes is
the encoding of the drive:

- For the four **contrast channels** `threat_left`, `threat_right`,
  `threat_front`, `light`, a leaky running mean is maintained per channel:

  ```
  m_t = (1 − α) · m_(t−1) + α · raw_t        with declared α = 1/8 per game tick
  drive_t = clip(|raw_t − m_(t−1)|, 0, 1)
  ```

  The baseline `m_(t−1)` is initialized to `raw_0` on the first tick after
  construction or `reset()`, so `drive_0 = 0` and the update yields
  `m_0 = raw_0`. The drive is scored against the PRE-update mean, and the
  mean updates once per game tick (never per neural sub-step).
- The **`touch` channel stays ABSOLUTE** (declared exception): a collision
  flag is already an event, not a standing field; contrast of a binary would
  fire only on its edges, and a sustained contact would vanish from the drive
  within a few ticks.

**Rationales, one line each, declared before any measurement:**

- **α = 1/8**: at 20 game ticks/s this is a ~0.4 s adaptation time constant —
  standing conditions adapt away in seconds while one-tick events are scored
  against an almost-unmoved baseline — and 1/8 is dyadic, so the update is
  exact in binary floating point.
- **The absolute value**: v2 declares TWO drives per contrast channel,
  on-drive `clip(raw_t − m_(t−1), 0, 1)` and off-drive
  `clip(m_(t−1) − raw_t, 0, 1)`, but maps BOTH onto the SAME population
  (summed, clipped) for v2 simplicity; because at most one of the two is
  nonzero, the executed drive is exactly `clip(|raw_t − m_(t−1)|, 0, 1)`.
  **Disclosed plainly: real ON and OFF pathways are separate cell types that
  this encoder does not distinguish** — this is an engineering
  simplification, not a model of ON/OFF physiology.

**State, determinism, and reset.** The running means are the only cross-tick
encoder state and live on the controller. They are a pure function of the
observation history: the same observation sequence reproduces the same means,
the same drives, and — on the same backend — the same neural trajectory, so
same-backend replay remains exact. `reset()` clears the means together with
the neural state: the next step is a declared first tick again. The per-step
stimulation provenance (schema
`neurocraft.full_brain.encoded_stimulation.eem.v2`) chains the raw channel
values, the exact baselines used, the encoded contrast values, and the
per-population drives, so an auditor can recompute the encoding and the
leaky-mean update from the digested material alone; per-step evidence records
`eem:` (encoded drive), `eem_raw:`, `eem_mean:`, and `eem_drive:` keys.

**Identity chaining.** The adapter identity keeps the v1 chain shape (schema
`neurocraft.boundary_adapter.identity.eem.v1` over {pinned feature schema,
faithful sensory binding, population binding, encoder document}) and rotates
because the encoder document is v2's. The configuration identity is under
`neurocraft.controller.full_brain_eem_v2_config.v1` with the v2 encoder
identity chained in; the uncalibrated configuration identity is
`63cfde2cb15627dec2907df250504b3b652407f7be2af2e83b1110ab0604ec94` (derivable
without the artifact; the calibrated configuration and decoder identities
incorporate the phase-2 calibration digest and will be recorded with it). A
v1 configuration with a v2 encoder document, or the reverse, refuses
construction.

## Phase-2 measurement protocol (declared here, not yet run)

1. **Boundary calibration**, measured once by
   `scripts/calibrate_full_readout.py --profile eem-v2` under the stream
   `full-readout-calibration-eem-v2`: byte layout and declared distributions
   IDENTICAL to the eem-v1 stream — rays from two digest bytes each, light
   from byte 12 (uniform integers 0..15), horizontal/vertical collisions from
   digest bytes 13:15 / 15:17 as declared Bernoulli(6554/65536 ≈ 0.1000061)
   — under the new stream id as seed prefix, so the contrast dynamics see the
   same realistic variation rather than a new distribution choice. Same
   derivation as v1: 50 warmup + 600 measured ticks,
   `readout_scale = 0.5 / (p95 × motor_pool_gain)`, one scalar, both pools,
   frozen write-or-verify into
   `data/calibration/full-readout-calibration-eem-v2.json`. The document's
   `encoder` section binds the v2 encoder identity, so the wing, eem-v1, and
   eem-v2 calibrations all refuse each other's boundaries in every direction.
2. **The six-condition probe battery from v1** with the calibrated eem-v2
   controller (and, for the honest A/B, the calibrated eem-v1 and wing
   controllers): hold each condition 30 ticks from a fresh reset and report
   all motor axes — all-open bright (light 15), all-open dark (light 0),
   blocked-left, blocked-right, blocked-front, collision-on. Exact ray
   values, closing a distance the v1 document left only in its reference
   script: the blocked ray is held at 0.5 m and every other ray at 6.0 m
   (`ray_max_distance_m` = 6.0); open conditions hold all six rays at 6.0 m.
   Under contrast encoding the held conditions are expected to adapt toward
   zero drive; that expectation is declared, not assumed — whatever is
   measured is reported.
3. **Two contrast-specific probes**, the events contrast encoding should
   amplify (declared exactly, 30 ticks each from a fresh reset):
   - **Light step 15→3 mid-hold**: all rays open, light 15 for ticks 0–14,
     light 3 from tick 15 onward (0-indexed).
   - **Approach ramp**: light 15, all rays open except the left ray, which
     shrinks linearly from 6.0 m at tick 0 to 0.5 m at tick 19
     (`ray_left(t) = 6.0 − (6.0 − 0.5)·t/19`), then holds at 0.5 m for the
     remaining 10 ticks.

No parameter is changed in response to what any phase-2 measurement shows;
v2 is what this document says it is. Phase-2 results are recorded in a
separate results document referencing this identity.

## Live wiring

Opt-in service profile `full_mcns_eem_v2` (same refusals as
`full_mcns_eem_v1`: no traces, no lesions, no fixture, no presentation
telemetry; local-only artifact and body-annotations table required), loading
the eem-v2 calibration by default and refusing the wing and eem-v1
calibrations; until the phase-2 calibration exists the service refuses to
start, naming `scripts/calibrate_full_readout.py --profile eem-v2`
(`--full-uncalibrated` opts out explicitly). Launcher row with fixture `None`
and description "166,700-neuron full MaleCNS with declared temporal-contrast
embodied senses (EEM v2, local-only)"; `make live-demo-eem-v2`. The wing,
eem-v1, TRM, steering, and pair profiles are byte-identical before and after
this change.

## Claim boundary — what is and is not claimed

Supported by a successful demonstration: *"a declared, engineered
temporal-contrast front end drives labeled sensory populations of the
measured circuit, whose untouched dynamics and untouched declared readout
then actuate the fly"* — Engineered-Encoder Mode engineering evidence,
nothing more.

Never supported, and disclosed plainly:

- **No biological encoding claim.** A leaky running mean is not
  photoreceptor adaptation, the absolute deviation is not the separate ON/OFF
  pathways of the real visual system (declared above), ray-proximity contrast
  is not looming, uniform per-capita drive is not retinotopy, and a collision
  bit is not a bristle deflection model. The labels locate populations; the
  encoder's meaning is engineering convention, declared and hashed.
- **No faithful-mode claim.** The sentence "nothing else computes in the
  loop" belongs to Faithful Organism Mode only and is never blended with this
  identity (ADR 0008 headline-language rule).
- **No behavior/performance claim in advance.** That contrast encoding
  "should" break the straight line is the motivation for the design, not a
  claim; a fly that still flies straight is a recorded observation, exactly
  as the v1 straight-line fly was.
- The coarse sign overlay's caveats (`SIGN-OVERLAY-02`) apply unchanged.
