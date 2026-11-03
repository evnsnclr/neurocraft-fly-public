# TRM live-demo identity — cloned-readout flight, v1

- **Status: recorded engineering-demonstration identity, not a statistical
  protocol.** This document exists to satisfy ADR 0008's rule that
  Trained-Readout Mode identities are recorded before anything runs under
  them. It declares a *demonstration*: no hypothesis test, no preregistered
  power, no confirmatory claim. The only measurements it promises are
  descriptive fit metrics on held-out training scenarios, reported as
  observed.
- **Identity:** `trm-live-demo-v1`. Mode: Trained-Readout Mode (ADR 0008,
  Mode 2 of 3). Never described as Faithful Organism Mode; every artifact,
  overlay, and video caption produced under this identity carries the
  trained-readout label.
- **Purpose:** demonstrate visibly purposeful, obstacle-responsive flight from
  the same frozen measured circuit whose biological readout circles — the
  "after" half of the before/after demo, motivated by the recorded 2026-08-27
  manual observation and licensed by `TRM-E1-01`'s finding that the circuit's
  hidden state carries structure-dependent, linearly decodable information.

## Frozen core (identical to the live faithful controller)

`mcns-v1.0-flight-steering-v1` (runtime identity `ebc50c0d…01c26`),
`coarse_presynaptic_consensus` signs (`SIGN-OVERLAY-01`), the pinned reference
engine, the pinned `PhysicalFeatureEncoder` (forward/left/right clearance
fractions + light/15), the pinned `SensoryAdapter` channels, and exactly five
engine steps per tick with the encoded stimulation held — byte-for-byte the
faithful sensory path. Nothing about the dynamics is trained.

## The trained readout

- **Form:** ridge regression with intercept over the full 128-node hidden
  state, three output channels: `forward`, `lateral`, `yaw`.
- **Cloning target:** the committed `scripted_baseline` restricted to its
  ray-only components — `forward = clip(ray0/2)`,
  `lateral = clip((ray3 − ray2)/4)`, `yaw = clip((ray3 − ray2)/2)`. The
  baseline's vertical component is **excluded by information availability**:
  it reads `on_ground` and vertical velocity, which never enter the circuit,
  so no readout of the hidden state could clone it honestly.
- **Action projection (declared):** `forward_thrust = clip(forward)`,
  `lateral_thrust = clip(lateral)`, `yaw_torque = clip(yaw)`,
  `vertical_thrust = forward_thrust` — mirroring the faithful projection's
  common-mode→vertical coupling so altitude behavior matches the working
  demo — and `pitch_torque = 0`. Clip is to the protocol's [−1, 1].
- **Free parameters:** 3 × 129 weights, declared; the l2 grid is E1's
  {1e-6, 1e-4, 1e-2, 1.0}, selected on the validation scenarios.

## Training distribution (declared generative process)

Synthetic sensor trajectories, not physics: SHA-256 counter streams (the
E1 derivation discipline, role-separated) generate per-scenario ray
trajectories for rays 0/2/3 as bounded bounded-step random walks on
[0.3 m, 6.0 m] with interleaved wall-approach ramps that exercise every
branch of the expert (blocked forward, left-blocked, right-blocked, open);
other rays fixed at 6.0 m, light fixed at 8, all other frame fields benign
constants. Each tick: encode frame → adapter → five engine steps → record
(post-step hidden state, expert action on the same frame). Scenarios split
train/validation/test by index. This trains the sensor→action mapping through
the circuit's real dynamics; it is open-loop behavioral cloning and is
disclosed as such — closed-loop behavior in Minecraft is what the
demonstration itself shows, not something this document claims in advance.

## Artifact and identity binding

Training writes `data/readouts/trm-cloned-readout-v1.json` (tracked, written
once): weights and intercept in float hex, graph identity, this document's
purpose id, generator parameters, per-channel held-out MSE/R², and a
descriptive directional-sanity rate (fraction of blocked-forward test frames
where the readout's yaw sign matches the expert's — reported, not gated).
The artifact's SHA-256 binds into `TrmDemoConfiguration`
(schema `neurocraft.controller.trm_demo_config.v1`) and the trained decoder's
identity (schema `neurocraft.decoder.identity.trm_demo.v1`), so the bridge
trace manifest carries the trained readout's exact provenance and a plain v1
or wing identity can never describe a trained-readout run.

## Live wiring

A new opt-in `TrainedReadoutController` (new module; no pilot-pinned byte
touched; the shipped defaults and the flight profile unchanged), exposed to
the bridge as launch-v2 controller profile `trm_cloned_readout_v1` with its
own run id `external-bridge-trm-demo-v1` and uuid5-derived entity id,
following the EXTERNAL-BRIDGE-05 profile pattern including the profile-pinned
identity labels.

## Claim boundary

What a successful demonstration supports: *"a small trained readout over the
frozen measured circuit's dynamics steers the fly"* — Trained-Readout Mode
engineering evidence. What it never supports: any faithful-mode, biological,
learning-in-the-brain, or gate claim. A fly that fails to steer is a recorded
observation, exactly as the circling faithful fly was.
