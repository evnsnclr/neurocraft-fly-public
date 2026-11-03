# TRM live-demo identity — cloned-readout flight, v2

- **Status: recorded engineering-demonstration identity, not a statistical
  protocol.** This document exists to satisfy ADR 0008's rule that
  Trained-Readout Mode identities are recorded before anything runs under
  them. It declares a *demonstration*: no hypothesis test, no preregistered
  power, no confirmatory claim. The only measurements it promises are
  descriptive fit metrics on held-out training scenarios, reported as
  observed.
- **Supersession:** `trm-live-demo-v2` supersedes `trm-live-demo-v1` for all
  future runs of the `trm_cloned_readout_v1` bridge profile. The v1 identity,
  its artifact (`data/readouts/trm-cloned-readout-v1.json`, SHA-256
  `527eb4f7…9dc3c6`), and every receipt bound to it (D-047, D-048,
  `EXTERNAL-BRIDGE-06`) remain recorded history, exactly as written; nothing
  in v1 is edited or retrained in place.
- **Identity:** `trm-live-demo-v2`. Mode: Trained-Readout Mode (ADR 0008,
  Mode 2 of 3). Never described as Faithful Organism Mode; every artifact,
  overlay, and video caption produced under this identity carries the
  trained-readout label.
- **Purpose:** recalibrate the cloned readout for the live loop's actual input
  distribution. `EXTERNAL-BRIDGE-06` recorded, unretouched, that the v1
  readout demonstrably reacts to the world but saturates at the clip outside
  transitions, with the labeled hypothesis: *the v1 training streams vary
  every tick while live Minecraft scenes are quasi-static, and the live loop
  has no washout, so near-constant inputs drive the network toward states
  outside the v1 training distribution.* v2 changes exactly the training
  distribution to cover that regime — openly, under its own identity, never
  as a silent patch.

## Frozen core (identical to v1 and to the live faithful controller)

`mcns-v1.0-flight-steering-v1` (runtime identity `ebc50c0d…01c26`),
`coarse_presynaptic_consensus` signs (`SIGN-OVERLAY-01`), the pinned reference
engine, the pinned `PhysicalFeatureEncoder` (forward/left/right clearance
fractions + light/15), the pinned `SensoryAdapter` channels, and exactly five
engine steps per tick with the encoded stimulation held — byte-for-byte the
faithful sensory path. Nothing about the dynamics is trained, and nothing
about the core changes between v1 and v2.

## The trained readout (form unchanged from v1)

- **Form:** ridge regression with intercept over the full 128-node hidden
  state, three output channels: `forward`, `lateral`, `yaw`.
- **Cloning target:** the committed `scripted_baseline` restricted to its
  ray-only components — `forward = clip(ray0/2)`,
  `lateral = clip((ray3 − ray2)/4)`, `yaw = clip((ray3 − ray2)/2)`. The
  baseline's vertical component remains **excluded by information
  availability**: it reads `on_ground` and vertical velocity, which never
  enter the circuit, so no readout of the hidden state could clone it
  honestly.
- **Action projection (declared, unchanged):** `forward_thrust =
  clip(forward)`, `lateral_thrust = clip(lateral)`, `yaw_torque = clip(yaw)`,
  `vertical_thrust = forward_thrust`, `pitch_torque = 0`. Clip is to the
  protocol's [−1, 1].
- **Free parameters:** 3 × 129 weights, declared; the l2 grid is E1's
  {1e-6, 1e-4, 1e-2, 1.0}, selected on the validation scenarios.

## Training distribution v2 (declared generative process — the recorded fix)

All streams derive from SHA-256 counters under the protocol id
`trm-live-demo-v2` (the E1 derivation discipline, role-separated), so no v2
stream can ever collide with a v1 stream. Three changes, each aimed at the
recorded hypothesis; everything else is identical to v1 (ray bounds
[0.3 m, 6.0 m], walk step 0.6 m, approach decay 0.85, open drift 0.2 m,
light 8, 200 ticks per scenario, 50-tick segments, five engine steps per
tick, expert labels, l2 grid):

1. **Hold segments (quasi-static coverage).** Segment modes are drawn from
   six options — `approach_forward`, `approach_left`, `approach_right`,
   `open`, `hold`, `hold` — so a hold segment is drawn with probability 1/3.
   In a `hold` segment the rays are **frozen for the whole segment** at the
   value they carried entering it: no walk step, no drift, no decay. The
   network sees constant input long enough to approach its driven fixed
   point, and the expert's label on that constant input is what the readout
   must reproduce there.
2. **No washout (`washout_ticks = 0`).** Every tick from the fresh engine's
   cold start is kept as training data, because the live loop has no washout
   either: the bridge starts a fresh engine and the readout acts from tick
   one. Cold-start transient states are in-distribution by construction.
3. **Eight pure-constant scenarios (indices 60–67).** Rays fixed for the
   entire scenario at declared values covering the open, blocked-forward,
   left-blocked, and right-blocked cases, one milder and one sharper variant
   each, in this order (ray0 = forward, ray2 = left, ray3 = right, metres):
   `(6.0, 6.0, 6.0)` open; `(0.5, 6.0, 6.0)` blocked forward;
   `(2.0, 1.0, 2.4)` left-blocked; `(2.0, 2.4, 1.0)` right-blocked;
   `(4.0, 5.0, 5.0)` open; `(0.8, 3.0, 3.0)` blocked forward;
   `(0.9, 0.6, 3.0)` blocked forward + left-blocked;
   `(0.9, 3.0, 0.6)` blocked forward + right-blocked.

**Scenarios and splits (exact indices, declared).** 68 scenarios: 60 walk
scenarios (0–59, the v1 mechanics plus hold segments) and the 8 constants.
Extending the v1 40/10/10 split proportionally:

- **Train:** walk 0–39 plus constants 60–63 (44 scenarios).
- **Validation:** walk 40–49 plus constants 64–65 (12 scenarios).
- **Test:** walk 50–59 plus constants 66–67 (12 scenarios).

Each tick: encode frame → adapter → five engine steps → record (post-step
hidden state, expert action on the same frame). This trains the sensor→action
mapping through the circuit's real dynamics; it is open-loop behavioral
cloning and is disclosed as such — closed-loop behavior in Minecraft is what
the demonstration itself shows, not something this document claims in
advance.

## Artifact and identity binding

Training writes `data/readouts/trm-cloned-readout-v2.json` (tracked, written
once) under the same artifact **format** schema
`neurocraft.trm.cloned_readout.v1` — the schema names the file format, not
the demo round — with `"demo_id": "trm-live-demo-v2"` naming this identity.
It carries: weights and intercept in float hex, graph identity, the full v2
generator parameters including the segment-mode table, the constant-scenario
ray values, and the exact split indices, and these descriptive metrics,
reported as observed:

- per-channel held-out MSE and R² on the test scenarios;
- the v1-pattern yaw directional-sanity rate (fraction of blocked-forward
  test frames, forward ray < 1.0 m, where the readout's yaw sign matches the
  expert's, counted only where the expert's |yaw| exceeds the disclosed
  0.05 epsilon) — reported, not gated;
- **pre-clip saturation fractions** (the number the fix is judged by): the
  fraction of held-out test frames whose raw readout value exceeds 1.0 in
  magnitude before the [−1, 1] clip, per channel and for any channel,
  reported **separately** for the v1-like varying frames (non-hold segments
  of walk scenarios) and for the quasi-static frames (hold segments plus the
  constant test scenarios). Reported, not gated.

The artifact's SHA-256 binds into `TrmDemoConfiguration`
(schema `neurocraft.controller.trm_demo_config.v1`) and the trained decoder's
identity (schema `neurocraft.decoder.identity.trm_demo.v1`) exactly as in v1,
so the bridge trace manifest carries the trained readout's exact provenance
and a plain v1 or wing identity can never describe a trained-readout run.

## Live wiring

Unchanged machinery, re-pointed artifact: the existing opt-in
`TrainedReadoutController`, bridge launch-v3 controller profile
`trm_cloned_readout_v1`, run id `external-bridge-trm-demo-v1`, and pinned
entity id all stay as recorded. The profile row's `readout_artifact_path` now
names the v2 artifact; because every configuration, decoder, and launch
identity derives from the artifact's SHA-256 at read time, the switch rotates
those identities automatically and visibly. Historical receipts
(`EXTERNAL-BRIDGE-06`) remain bound to the v1 artifact hash and are correct
as history. No pilot-pinned byte is touched; the shipped default and wing
profiles are byte-identical before and after.

## Claim boundary

What a successful demonstration supports: *"a small trained readout over the
frozen measured circuit's dynamics steers the fly"* — Trained-Readout Mode
engineering evidence. Improved saturation metrics support only: *"the v2
training distribution covers quasi-static inputs the v1 distribution did
not."* What it never supports: any faithful-mode, biological,
learning-in-the-brain, or gate claim. A fly that still fails to steer is a
recorded observation, exactly as the circling faithful fly and the saturating
v1 readout were.
