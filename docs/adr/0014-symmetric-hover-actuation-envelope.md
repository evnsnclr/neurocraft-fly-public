# ADR 0014: Symmetric hover actuation envelope for the fly body

_Body physics-envelope declaration, drafted 2026-08-31_

---

| Field | Value |
|---|---|
| **Status** | Implemented; GameTest-pinned with an executed negative control |
| **Date** | 2026-08-31 |
| **Authority** | `FlyEntity.integratePassivePhysics` — the fly body's declared passive-physics envelope |
| **Precedent** | ADR 0003 bounded actuation; ADR 0008 mode taxonomy (the envelope is body, not policy); D-052 declared boundary calibration |
| **Evidence** | Three GameTest pins in `FlyEntityGameTests` (liftoff from grounded start, neutral hover-hold, commanded landing) plus an executed negative control: on the reverted old physics the liftoff and hover pins fail exactly as predicted while all 18 other tests pass |

## Context

The fly body declared two constants that were exactly equal:

- `FlyEntity.GRAVITY_BLOCKS_PER_TICK_SQUARED = 0.04`, applied unconditionally
  every tick before `move()`;
- `PrimitiveActuator.MAX_THRUST_DELTA_BLOCKS_PER_TICK = 0.04`, the full
  vertical thrust authority (`clamp(action, [-1, 1]) * 0.04`, pitch hardwired
  to 0, forward thrust horizontal-only).

Ambient gravity therefore exactly cancelled maximum commanded vertical thrust.
Per tick, with commanded vertical `v` in `[-1, 1]`:

```
vy' = ((vy + 0.04 * v) - 0.04) * 0.98
```

At the +1.0 rail this is `vy' = 0.98 * vy`: a grounded fly (`vy = 0`) can never
achieve positive vertical velocity, so **the floor was an absorbing state no
neural output could exit**. Every profile, mode, and control was subject to it;
every session ever run was floor-bound within about a second of spawn (the
live launcher's fly starts three blocks above the floor — spawn placement is
one block above the command source — and the unopposed sink covers that in
under a second).

Live proof, on the record: in session
`artifacts/live-interactive/interactive-20260831T042655Z-77d432b9` the eem-v1
profile held vertical thrust **positive** (+0.27..+0.53, equal in magnitude to
its forward channel) for 85 s across 43 status samples, and the fly never left
`y=64.000000`. The equality also masks every committed response amplitude: the
D-062 loom response (total vertical +0.222) sits far below the `v > 1.0` a net
climb required, so the first event-locked responses at visible scale in any
profile were physically invisible in the world.

## Decision

Replace the body's declared actuation envelope with a **symmetric hover
envelope**: remove the ambient gravity term entirely, so the vertical axis has
exactly the same authority and semantics as the horizontal axes — ±1 maps to
±0.04 blocks/tick of commanded acceleration — with the existing drag retained
(0.91 horizontal, 0.98 vertical, 0.80 in water). Neutral motors drift to a stop
and hover: fail-neutral becomes hover-hold, not free-fall, which makes the
ADR 0013 description of a session inside a grace window ("the fly hovers on
neutral motors") literally true for the first time.

This is a physics-envelope constant of the body, chosen for **actuation
legibility**: motion now maps to motor commands and to nothing else. Under
hover the committed amplitudes become visible instead of sub-threshold — the
D-062 loom response (total vertical +0.222) commands `0.222 × 0.04 ≈ 0.0089`
blocks/tick² of upward acceleration — a steady-state climb rate of ≈0.435
blocks/tick under the 0.98 vertical drag — instead of a slightly-slowed sink.
It is applied
uniformly to every profile, mode, and control; it is not mob AI, not a policy,
and no observation or actuation semantics change (the wire protocol carries no
gravity or passive-force field — `passive_forces` exists only in the Python
fixture-trace schema, written by the standalone harness's own declared arena).

Implementation, precisely:

- `GRAVITY_BLOCKS_PER_TICK_SQUARED` is **deleted**, along with its application
  and the now-vacuous `isNoGravity()` guard and pre-move `setDeltaMovement`.
- **No vanilla path re-adds gravity**, verified against the mapped 1.21.5
  bytecode: `FlyEntity` extends `Entity` directly (GameTest-pinned),
  `Entity.tick()` calls only `baseTick()` which never invokes `applyGravity()`,
  and the inherited `getDefaultGravity()` returns `0.0` (unoverridden), so even
  an unexpected `applyGravity()` call would be a no-op. Explicit
  `setNoGravity(true)` is therefore **not** needed and is not set: the flag no
  longer has any reader on this entity.
- The grounded downward-velocity zeroing is **kept** as a defensive floor
  invariant. It is reachable only through commanded descent into contact, and
  on default blocks it is redundant with vanilla collision response
  (`Entity.move` invokes `Block.updateEntityMovementAfterFallOn`, which zeroes
  vertical velocity), but it guarantees the invariant across blocks whose
  collision response does not zero it, and it never suppresses an upward
  response (it fires only on `vy < 0`).
- Spawn placement is unchanged and now means what it says: `neurocraftfly
  spawn` places the fly one block above the command source and the interactive
  launcher spawns at `SPAWN_Y + 1 = 67` (`scripts/run_interactive_demo.py`,
  untouched in this change) — under hover the fly holds that height until
  motors act, instead of sinking to the floor and into the touch loop within a
  second.

## Alternatives considered

**Reduced gravity at a declared fraction of thrust authority** (half, or a
tenth, of `0.04`): rejected. Any nonzero ambient force keeps baseline-level
motor output — measured at roughly +0.001..+0.005 across the committed probe
batteries — below the hover threshold, so the fly still ends grounded at rest
and the touch-collision loop re-engages on contact. The failure would merely be
slower, and the envelope would need a second declared constant whose only
defensible value is the one that restores symmetry anyway. (The touch loop
itself — the escape profile collision-driving its own jump axis to the −1.0
rail — is a separate frozen-identity concern, addressed in its own lane, not
here.)

**Raising thrust authority above gravity** (e.g. `MAX_THRUST = 0.08` against
gravity `0.04`): rejected. It changes the horizontal axes' declared authority
as a side effect (or splits vertical from horizontal authority, breaking axis
symmetry in the opposite direction), and it keeps a hidden bias — neutral
motors still sink, so fail-neutral still lands in the absorbing corner case the
live sessions demonstrated.

## Regression pins and test adjustments

Three new GameTest pins in `FlyEntityGameTests` (server lane 17 → 20 tests):

- `fullVerticalThrustFromGroundedStartProducesLiftoff` — the exact case the old
  physics made impossible: commanded descent grounds the fly on a placed stone
  floor, then full positive vertical thrust must lift it more than half a block.
- `neutralMotorsHoldAltitudeWithoutAmbientSink` — an airborne fly with neutral
  motors holds position exactly (no ambient sink), with zero velocity and no
  ground contact, over 10 ticks.
- `commandedDescentStillReachesTheFloorAndGrounds` — landing stays possible: a
  commanded descent reaches the placed floor, grounds, and rests at exactly the
  floor surface with zero vertical velocity.

**Negative control, executed** (not merely reasoned): the old gravity term was
temporarily re-added and the suite run once — the liftoff and hover pins failed
with their own messages ("full positive vertical thrust did not lift the fly",
"neutral motors did not hold altitude") and all 18 other tests passed,
confirming both that the pins bind the defect and that no other test secretly
depends on either physics. The revert was then removed; all 20 pass on the
hover envelope.

Adjustments to existing tests: **no pinned timeline, position, or assertion
changed**. Eleven now-inert `setNoGravity(true)` shims were deleted (nine in
`FlyEntityGameTests`, one in `FlyClientGameTest`, one in
`ExternalBridgeGameTest`): they existed solely to opt those tests out of the
removed gravity term, and after this change the flag has no reader anywhere in
the entity's physics, so keeping them would misrepresent the envelope. The
negative-control run doubles as evidence the deletions are behaviorally inert
in every passing test.

## Consequences

- The envelope is uniform across modes and controls — Faithful Organism Mode,
  EEM, TRM, and the shuffle/rewire/no-input controls all fly the same body; no
  profile-specific physics exists.
- **Escape-v1's measured collision response can still ground-pin the fly.** The
  D-062 record measured collision driving the jump term to the −1.0 rail
  (vertical −0.723): on contact, that commanded descent holds the fly down
  until the touch input releases — hover removes the *ambient* absorber but a
  profile can still command itself onto the floor. This is honest: it is the
  organism's measured response, visible for exactly what it is, and the
  escape-v2 identity work addresses it in its own recorded lane.
- **No Python-side identity rotates.** Verified against the hash-construction
  code, not assumed: the configuration identities
  (`FullBrainConfiguration.to_identity_material` and the eem/escape variants in
  `packages/controller-runtime/src/neurocraft_controller/full_brain_controller.py`)
  cover gain, readout scales, calibration digests, and the encoder identity;
  the sensory-binding and motor-binding documents cover channels, populations,
  pools, weights, the sign-overlay identity, and calibration digests; the EEM
  encoder identity (`embodied_senses.py`) covers formulas, gains, populations,
  and ray semantics. No identity document anywhere binds a mod physics
  constant, so every committed adapter/decoder/configuration hash, calibration,
  and receipt is byte-unchanged by this ADR.
- **The standalone fixture harness still models the old envelope.**
  `scripts/fixture_harness.py` declares its own Minecraft-free arena with
  `GRAVITY = 0.04` and records it as `passive_forces` in its traces, bound into
  its own `code_hash`; it is untouched here (this change is mod-only) and its
  historical traces remain correct as self-consistent artifacts of that
  declared arena. Its arena now differs from the live body; reconciling it is
  recorded follow-up work, and any future comparison between harness traces and
  live behavior must account for the envelope difference. The pinned
  fixture-physics pilot arena (`neurocraft_experiments/pilot.py`,
  `gravity_m_s2`) is likewise a separate declared non-Minecraft physics and is
  untouched.
- The change touches `mod/src` control-adjacent body code, so any evidence
  bundle that enumerates mod control sources is superseded for future candidate
  runs and must be regenerated on the fixed tree, exactly as ADR 0006, 0011,
  0012, and 0013 recorded for their `mod/src` changes.
- Live re-observation is pending: nothing about live behavior under hover is
  claimed until a session is observed and recorded.
