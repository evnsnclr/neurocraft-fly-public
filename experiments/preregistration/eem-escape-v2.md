# EEM escape-reflex identity — transient touch and rectified jump, v2

- **Status: recorded engineering-demonstration identity, not a statistical
  protocol — frozen and committed BEFORE any measurement.** This document
  satisfies ADR 0008's rule that Engineered-Encoder Mode identities are
  recorded before anything runs under them, and the D-058/D-059
  freeze-then-measure process rule: the identity text and the tunable code
  are frozen and committed first, and every measurement follows in a
  separate phase whose results are recorded separately and never edit this
  document. It declares a *demonstration*: no hypothesis test, no
  preregistered power, no confirmatory claim. As of this document's commit,
  **no escape-v2 measurement of any kind has been run**: no calibration, no
  probe, and no live session under this lane's code.
- **Identity:** `eem-escape-v2`, service profile `full_mcns_escape_v2`.
  Mode: Engineered-Encoder Mode (ADR 0008, Mode 3 of 3). Never described as
  Faithful Organism Mode; every artifact, overlay, and caption produced
  under this identity carries the engineered-encoder label, and the
  configuration identity pins `mode_taxonomy: "engineered_encoder"` under
  the schema `neurocraft.controller.full_brain_escape_config.v2`.

## Motivation — the recorded escape-v1 live observations

The escape-v1 live sessions (recorded in D-063 and D-064) showed the D-062
collision measurement operating as a continuous state rather than an event.
The causal chain, established from session logs and code trace and recorded
in D-064: ground or wall contact re-asserts the entity's collision flags on
every authoritative tick; the v1 touch channel is absolute and
non-adapting, so all 2,206 bristle members receive the full 1.0 per-capita
drive on every tick of sustained contact; and the measured collision
response — forward +0.27 (D-058) with the jump term at the −1.0 clip rail
(vertical −0.723, D-062) — therefore holds continuously, pressing the fly
into the floor it is touching. The committed record states the outcome
plainly: the fly "collision-drove its own jump axis to the −1.0 rail and
self-trapped against a floor corner" (`STATUS.md`, escape-reflex row; the
D-063 decision row records the same observation in its own words), and the
eem-v1 session
`interactive-20260831T042655Z-77d432b9` held vertical positive for 85 s
across 43 status samples without ever leaving y=64.000000 (D-064).
Separately, D-064 (ADR 0014) replaced the body's actuation envelope — under
which no vertical output could produce a climb — with the symmetric hover
envelope, making measured vertical amplitudes visible for the first time.
On those recorded observations, this identity executes the follow-up
engineering decision: the touch channel becomes transient and the jump
mapping is rectified to match the upward-thrust body policy that
`eem-escape-v1.md` already declared.

**No parameter here was chosen against any measurement of what "looks
better".** Both changes are structural: one channel's declared treatment
switches to the already-frozen D-059 temporal-contrast formula, and the
jump clip range changes to the non-negative half of its declared codomain.

## Frozen core (identical to eem-escape-v1)

The verified full MaleCNS runtime artifact (manifest `b33bdda8…4f67c98c`,
166,700 neurons / 25,582,938 edges), the disclosed
`coarse_presynaptic_consensus` sign overlay (`SIGN-OVERLAY-02`), the
array-native backend under its own dynamics identity, the two faithful
sensory entries via the pinned channel builder (retained byte-identically),
the declared 25+27 somaSide wing-motor pools with a-priori gain 8.0 / dead
zone 0.0, the pinned primitive projection, and the TTMn pool selection
exactly as escape-v1 declares it (`type == 'TTMn'`, exactly 2, 1 L / 1 R,
bodies 800146 R / 804642 L per the census, all overlap refusals, hashed
binding under `neurocraft.escape.ttmn_pool_binding.v1`). DNp01 is never
read directly. The wing, eem-v1, eem-v2, and escape-v1 profiles are
regression-pinned byte-unchanged by the existing suites plus this lane's
new pins.

## Declared change 1 — the touch channel becomes transient

The sensory boundary keeps eem-v1's populations, selectors, counts,
membership (hashed population binding `bf45222d…3bae`), ray-side
convention, and per-capita drive semantics. Four of the five channels —
`threat_left`, `threat_right`, `threat_front`, `light` — keep the v1
ABSOLUTE treatment verbatim, for the recorded D-060 reason: temporal
contrast adapts a held threat away, which is the wrong boundary for an arc
that scores looming events against a quiet motor stage.

The `touch` channel alone moves under the D-059 temporal-contrast formula,
verbatim from the frozen eem-v2 identity (which applied it to every channel
EXCEPT touch; this identity applies it to touch ONLY):

```
baseline_t = m_{t-1}                          (first tick: baseline_0 = raw_0, so drive_0 = 0)
drive_t    = clip(|raw_t − baseline_t|, 0, 1)
m_t        = (1 − 1/8) · baseline_t + (1/8) · raw_t   (α = 1/8 per tick, dyadic, bit-exact)
```

(the drive is scored against the PRE-update mean, exactly as
`apply_temporal_contrast` implements it — verbatim reuse of the frozen
D-059 machinery with `contrast_channels = ('touch',)`)

Rationale, declared: `touch` is the one channel whose raw signal is a
*state* (contact held true by the physics for as long as the body rests on
a surface) while the engineering-relevant signal is the *event* (impact or
release). Under this treatment a contact onset drives the bristles near
full amplitude and decays with the declared τ ≈ 8 ticks; sustained resting
contact adapts toward zero; a release produces an OFF transient of the
same sign (ON/OFF non-distinction disclosed, exactly as D-059 disclosed
it). Cross-tick encoder state now exists for this one channel: the running
mean is lock-scoped, cleared on controller reset, and never serialized —
the D-059 precedent verbatim.

This rotates the adapter and encoder identities (escape-v1's were
byte-identical to eem-v1's; escape-v2's are not). New encoder document
schema `neurocraft.eem.encoder.escape.v2`, identity
`1cb591a222806ea5e1b687a1ccdb092809a9a7223f5ebc1d523556478d82ea9c`
(derivable without the artifact: the encoder document hashes only declared
material plus the already-committed population-binding identity
`bf45222d…3bae`, exactly as the eem-v1 and eem-v2 encoder identities are);
per-step stimulation provenance
`neurocraft.full_brain.encoded_stimulation.eem.escape.v2`.

## Declared change 2 — the jump mapping is rectified upward

`eem-escape-v1.md` declares the body policy: *"the fly's Minecraft body
renders a TTMn activity spike as upward thrust."* Its formula nonetheless
admitted the full signed codomain, and D-062 measured collision driving
`jump_raw` to the −1.0 rail — a downward slam that contradicts the
declared takeoff semantics and, live, wedged the fly against the floor
(D-063/D-064). The v2 formula makes the mapping match the declared
convention (verbatim from
`neurocraft_controller.escape_reflex.ESCAPE_JUMP_FORMULA_V2`):

```
jump_raw = motor_pool_gain * (fsum(TTMn activity) / 2) * jump_scale
jump_value = clip(jump_raw, 0, 1), zeroed when clip(jump_raw, 0, 1) < motor_dead_zone
```

with `motor_pool_gain` = 8.0 and `motor_dead_zone` = 0.0 exactly as
escape-v1 declares, and `jump_scale` measured once by this profile's own
phase-2 calibration. Negative pooled TTMn activity maps to zero — no
anti-jump — and the vertical axis remains

```
vertical_thrust = clip(vertical_projection + jump_value, -1, 1)
```

one declared additive term on one axis; forward, lateral, yaw, and pitch
pass through bit-for-bit. The `jump_value` per-step evidence field and the
56-value `state_after_activity` extension carry over unchanged.

**Identity chaining.** Configuration identity under
`neurocraft.controller.full_brain_escape_config.v2` (uncalibrated identity
`32624431aefac74248edf51b58824dc97a37116abb26e9e90633bb2dcc0142c2`,
derivable without the artifact the same way escape-v1's `54814d90…b987da`
is: the configuration hashes only declared material plus the v2 encoder
identity above; the calibrated configuration identity incorporates the
phase-2 calibration digest and is recorded with it), chaining the v2
encoder identity and the declared `jump_scale`. Decoder identity under
`neurocraft.full_brain.decoder.escape.v2`, chaining the untouched wing
readout document + the TTMn pool membership + the v2 jump formula verbatim
+ the declared `jump_scale` + the calibration document digest + the v2
encoder identity; it depends on the artifact's dense indices, so — exactly
as escape-v1 handled the TTMn binding — it is recorded with the phase-2
calibration, not guessed here. A different membership, formula, scale,
encoder, or calibration is a different decoder. An escape-v2 configuration
over the v1 encoder document, or a v1 configuration over the v2 encoder, or
either over the eem-v2 all-channel-contrast document, refuses construction.

## Phase-2 measurement protocol (declared here, not yet run)

1. **Boundary calibration**, measured once by
   `scripts/calibrate_full_readout.py --profile escape-v2` under the stream
   `full-readout-calibration-escape-v2`: byte layout and declared
   distributions IDENTICAL to the escape-v1 stream (rays from two digest
   bytes each, light from byte 12 as uniform integers 0..15,
   horizontal/vertical collisions from digest bytes 13:15 / 15:17 as
   declared Bernoulli(6554/65536 ≈ 0.1000061)) under the new stream id as
   seed prefix. The stream drives RAW channel values; the touch contrast
   transforms them inside the encoder exactly as live. ONE run of 50
   warmup + 600 measured ticks measures BOTH pool groups and derives both
   scales by the standard derivation `scale = 0.5 / (p95 × motor_pool_gain)`
   per pool group. Frozen write-or-verify into
   `data/calibration/full-readout-calibration-escape-v2.json` under schema
   `neurocraft.full_readout_calibration.escape.v2`. Cross-binding: the
   document names this profile's encoder identity AND TTMn pool identity
   and refuses every other boundary, and every other boundary refuses it,
   in both directions.
2. **The declared probe battery**, with the calibrated escape-v2
   controller and, for the honest A/B, the calibrated escape-v1 controller
   — each condition held 30 ticks from a fresh reset, reporting ALL motor
   axes:
   - **The frozen six conditions at the pinned ray values** (blocked ray
     0.5 m, all others 6.0 m, `ray_max_distance_m` = 6.0): all-open bright
     (light 15), all-open dark (light 0), blocked-left, blocked-right,
     blocked-front, collision-on (note: under this identity "collision-on"
     is a held raw state whose drive adapts — the per-tick trajectory is
     the result, not a steady state).
   - **The approach ramp** (identical to the v1/v2 ramp for
     comparability): light 15, all rays open except the left ray shrinking
     linearly 6.0 → 0.5 m over ticks 0–19, held at 0.5 m for ticks 20–29.
   - **NEW: the touch-transient probe** — 30 ticks, all rays 6.0, light
     15; collision raw FALSE for ticks 0–4, TRUE for ticks 5–24, FALSE for
     ticks 25–29; reported per-tick on all axes. This captures the onset
     transient, the adaptation trajectory, and the release transient —
     the exact dynamics this identity exists to create.

**No pass/fail criterion is declared — explicitly.** Whatever is measured
is reported, in full, against this identity. Any go/no-go decision about a
live demo under this profile is an engineering decision made and recorded
AFTER the measurements. No parameter changes in response to what phase 2
shows.

## Live wiring

Opt-in service profile `full_mcns_escape_v2` (refusals identical to
`full_mcns_escape_v1`), loading the escape-v2 calibration by default and
refusing every other profile's calibration; until the phase-2 calibration
exists the service refuses to start, naming
`scripts/calibrate_full_readout.py --profile escape-v2`. Launcher row and
`make live-demo-escape-v2`. The wing, eem-v1, eem-v2, and escape-v1
profiles are byte-identical before and after this change.

## Claim boundary — what is and is not claimed

Everything `eem-escape-v1.md` declares in its claim boundary applies
verbatim, plus:

- **The rectification is a declared body-model convention**, aligning the
  formula with the upward-thrust body policy v1 already declared. It is
  not a claim that the organism cannot express downward motion, nor that
  negative TTMn pool activity is biologically meaningless — it is a
  declared choice about what one motor axis of a Minecraft body renders.
- **The touch transient is a declared engineering treatment** (events, not
  states), not a model of mechanosensory adaptation physiology. Its α and
  formula are inherited from the frozen D-059 identity, not fit to any
  observation.
- **No behavior/performance claim in advance.** A fly that still traps,
  still plows, or never visibly jumps is a recorded observation.
