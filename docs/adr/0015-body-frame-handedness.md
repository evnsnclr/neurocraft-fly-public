# ADR 0015: Body-frame handedness for the fly's lateral and vertical axes

_Body sensor/actuator frame correction, drafted 2026-08-31_

---

| Field | Value |
|---|---|
| **Status** | Implemented; GameTest-pinned in both languages with three executed negative controls |
| **Date** | 2026-08-31 |
| **Authority** | `FlySensors.sample` and `BodyFrameMath.toWorld` — the fly body's declared body frame — and `ReferenceEmbodiment.observe` / `._substep`, its Python twin's sensing and actuation halves |
| **Precedent** | ADR 0003 bounded actuation and ray semantics; ADR 0008 mode taxonomy (the frame is body, not policy); ADR 0014 (the previous body-envelope correction, same "declared engineering, fix the body not the connectome" posture) |
| **Evidence** | Six GameTests in `FlyEntityGameTests` (four lateral, two vertical) anchored on the engine's own `Direction.getClockWise()` and `Vec3.directionFromRotation` tables, plus `BodyFrameMathTest` and two parametrised Python handedness tests (sensing and actuation); three executed negative controls (see *Regression pins*) |

## Context

The fly's six sensor rays are declared **body-relative**: the protocol names
them `forward, back, left, right, up, down`, and the engineered encoder binds
slot 2 to the left LC4 looming population and slot 3 to the right one
(`embodied_senses.py`, `threat_left` / `threat_right`). The two lateral slots
were transposed. The slot the protocol calls "left" carried the body's **right**
at every yaw, and vice versa.

### The derivation, from shipped 1.21.5 bytecode

Minecraft's basis is `+X` east, `+Y` up, `+Z` south. Disassembling
`Vec3.directionFromRotation(float xRot, float yRot)` from the mapped
`minecraft-common` jar gives, for
`f = cos(-yaw_rad - PI)`, `g = sin(-yaw_rad - PI)`, `h = -cos(-pitch_rad)`,
`i = sin(-pitch_rad)`, the vector `(g*h, i, f*h)`:

```
yaw  0, pitch 0:  f = cos(-PI)    = -1,  g = sin(-PI)    =  0,  h = -1
                  forward = (g*h, i, f*h) = ( 0, 0, +1)  = SOUTH
yaw 90, pitch 0:  f = cos(-3PI/2) =  0,  g = sin(-3PI/2) = +1,  h = -1
                  forward = (g*h, i, f*h) = (-1, 0,  0)  = WEST
```

A body's right is a 90-degree clockwise turn from its heading, which is exactly
`Direction.getClockWise()`. Its shipped `tableswitch` (ordinals `NORTH=2`,
`SOUTH=3`, `WEST=4`, `EAST=5`) maps `NORTH->EAST`, `SOUTH->WEST`, `WEST->NORTH`,
`EAST->SOUTH`. Face north and your right hand points east; face south and it
points west. Therefore:

```
yaw  0: facing SOUTH -> right = WEST  = (-1, 0,  0)
yaw 90: facing WEST  -> right = NORTH = ( 0, 0, -1)
```

`Vec3.cross` disassembles to the standard right-handed
`(y*vz - z*vy, z*vx - x*vz, x*vy - y*vx)`, so with `up = (0,1,0)`:

```
yaw  0: forward x up = ( 0,0, 1) x (0,1,0) = (-1, 0,  0)  WEST   <- the body's right
        up x forward = (0,1,0) x ( 0,0, 1) = (+1, 0,  0)  EAST   <- the body's LEFT
yaw 90: forward x up = (-1,0, 0) x (0,1,0) = ( 0, 0, -1)  NORTH  <- the body's right
        up x forward = (0,1,0) x (-1,0, 0) = ( 0, 0, +1)  SOUTH  <- the body's LEFT
```

`FlySensors` computed `Vec3 right = WORLD_UP.cross(forward)` — the second line
in each pair. The variable named `right` was the body's left at every yaw, and
the slot assignments feeding `rayLeftBlocks` and `rayRightBlocks` were therefore
transposed. `BodyFrameMath` repeated the identical convention with
`rightX = cos(yaw)`, `rightZ = sin(yaw)`: `(+1, 0)` east at yaw 0 and `(0, +1)`
south at yaw 90, the body's left in both cases.

### Why the vertical axis was accidentally correct

`bodyUp` was built as `forward.cross(right)` with that left-handed `right`.
Because `forward x left == -(forward x right) == +up`, **the two sign errors
cancelled** and the up/down pair read correctly at every attitude. Correcting
the lateral axis alone would have silently inverted the vertical pair — and
nothing in the repository would have noticed, because before this lane **no
test in either language read `rayUpBlocks` or `rayDownBlocks` at all**. The fix
is therefore two lines, not one, and it ships with the vertical coverage that
was missing.

## Decision

Correct the body frame at all four sites that declare it, and pin every axis
with a test anchored on an engine table rather than on a hand-chosen constant.

- `FlySensors.java` — `right = forward.cross(WORLD_UP)`, and correspondingly
  `bodyUp = right.cross(forward)`.
- `BodyFrameMath.java` — `rightX = -cos(yaw)`, `rightZ = -sin(yaw)`. Latent
  **under the faithful pinned projection only**, which hardwires
  `lateral_thrust` to 0 (`controller.py:440`). It is *not* latent under the TRM
  demo projection — see *The lateral axis did execute* below — and it is the
  same defect in the actuation twin either way.
- `packages/embodiment-runtime/.../reference.py`, **sensing** — the Python
  reference body shares Minecraft's basis and yaw convention and carried the
  identical transposition (`left = (-cos, 0, -sin)`, which is west at yaw 0).
  Its two lateral direction tuples are swapped to match. Left uncorrected, the
  same connectome would fly mirrored in the two worlds, which is precisely the
  cross-body comparison the reference body exists to support.
- `packages/embodiment-runtime/.../reference.py`, **actuation** — `_substep`
  accelerated along `(cos, sin)`, east at yaw 0: character-for-character the
  `rightX = cos, rightZ = sin` that `BodyFrameMath` is corrected away from
  above. It is now `(-cos, -sin)`.

  This site is called out separately because the first draft of this lane
  missed it, and the way it was missed is the point. Before the lane
  `reference.py` was *uniformly* mirrored — sensing and actuation both inverted
  — so a controller coupled through both halves flew a consistent mirror image
  and nothing disagreed with itself. Correcting the rays alone would have left
  the file sensing right-handed and actuating left-handed: a new internal
  inconsistency introduced by the fix, worse than the defect it replaced.
  Nothing caught it, because the only tests that drove `lateral_thrust`
  (`tests/embodiment/test_arena_envelope.py:58`,
  `tests/embodiment/test_reference_embodiment.py:187`) asserted envelope
  containment, never direction. The rule the lane is really enforcing is
  therefore narrower than "the actuation twin is not left behind": **an axis
  with no assertion on it does not stay correct**, which is also why the
  vertical rays needed new tests and why `BodyFrameMathTest` had been asserting
  the defect. Every corrected axis in this ADR now carries a direction
  assertion.

**The ray directions, and what counts as an obstruction, are declared
engineering and always were.** Nothing here is derived from the connectome or
from fly biology: six body-aligned rays of 6.0 blocks — body-aligned, not
world-axis-aligned; they rotate with yaw and pitch, which is the whole subject
of this ADR — `ClipContext.Block.COLLIDER` with fluids ignored, sampled from the
bounding-box centre. This ADR does not
change that declaration, add a ray, or alter what a ray reports. It corrects
which *body side* each already-declared slot is wired to, so that the label the
protocol carries is the side the world actually presents. No connectome
parameter, gain, population binding, calibration, or encoder formula is touched.

Also declared, because the fix depends on it: the frame is never degenerate.
`PrimitiveActuator.MAX_ABS_PITCH_DEGREES = 89.0`, so `|forward x up|` is at
worst `cos(89 deg) ~= 0.0175` — about 1700x above the `1.0e-5` zero threshold in
`Vec3.normalize`, read from the same disassembly.

## The lateral axis did execute

`BodyFrameMath`'s lateral defect is **latent only under the faithful pinned
projection**. `ConnectomeController`'s pinned wing-differential projection sets
`lateral_thrust=0.0` unconditionally (`controller.py:440`), so on that path
`BodyFrameMath`'s lateral column is multiplied by zero and the handedness never
reaches the world.

There is a second projection. `TrainedReadoutController._project_motor_frame`
(`packages/controller-runtime/.../trained_readout_controller.py:330`) sets
`lateral_thrust=clip(values["lateral"])`, passing the readout's lateral channel
straight through. `docs/receipts/EXTERNAL-BRIDGE-06-trace.ndjson` records that
path running against the real mod on Minecraft 1.21.5: **nine actuation
records, every one `"validation":"APPLIED"` and every one carrying a non-zero
`lateral_thrust`** — `0.5758053547…`, `0.8748423482…`, six at exactly `1.0`, and
`0.9054827230…`. (Counted per record. The raw file contains 45 `lateral_thrust`
and 18 `"validation":"APPLIED"` occurrences because each record repeats them
across five sites — `applied_action`, `decoder_output`, `issued_action`, and the
`actuation_receipt`'s `applied_motors` and `requested_motors`;
those are duplications of the same nine commands, not eighteen or forty-five
distinct ones.) **The wrong-handed lateral axis therefore executed in
committed evidence, and the fly was pushed to the side opposite the one the
readout named.** Any statement that the `BodyFrameMath` defect was simply
"latent" is true only of the faithful pinned projection and is narrowed
accordingly wherever it appears in `STATUS.md`, `CHANGELOG.md`, `DECISIONS.md`
and the GameTest comments.

## The interpretation correction

Every previously recorded sided measurement is **numerically correct and no
recorded number changes**. The offline probes build `SensorFrame`s directly by
slot (`scripts/measure_escape_arc.py`, `scripts/calibrate_full_readout.py`);
they never construct a body frame, so they were never exposed to this defect and
reproduce bit-for-bit across the change (verified — see *Regression pins*).

What changes is the **world interpretation** of those rows. Because slot 2
physically carried the body's right and slot 3 its left, every condition the
records call `blocked_left` described, in the live world, an obstacle on the
fly's **right**, and every `blocked_right` an obstacle on its **left**.

The consequence is sharper than a relabelling, and it runs against the direction
one might assume. Measured on the committed eem-v1 boundary (baseline `-0.0189`,
slot-2-blocked `+0.0054`, slot-3-blocked `-0.0346` — the two sided figures
reproduce D-058's exactly; see *On the baseline figure* below), and combined
with the actuation sign — positive
`yaw_torque` increases yaw, and increasing yaw walks the heading
`SOUTH -> WEST -> NORTH -> EAST`, i.e. clockwise, i.e. toward the body's right:

- **Read against the protocol slots the connectome is declared against**, the
  wing differential has the correct avoidance sign in both directions: an
  obstacle on the declared left commands a right turn, one on the declared right
  commands a left turn. That is true of the recorded numbers and always was.
- **Read against the wiring that was actually in effect**, the same numbers
  commanded a turn *toward* the obstructed side on both sides. The body was
  delivering a correct neural response to the wrong side.

So the defect never lay in the connectome's differential; it lay in the body
that fed it. The fix makes the recorded numbers mean, in the world, what they
have always meant in the protocol. Affected rows are annotated in `DECISIONS.md`
and `STATUS.md` without altering a single measurement.

**On the baseline figure.** D-058 records the two *sided* conditions and no
all-open baseline row; the baseline quoted above belongs to this measurement,
not to D-058, and the earlier claim that it "reproduces the D-058 signature
exactly" is narrowed to the two sided figures accordingly. Four committed places
write that baseline as `-0.0188` — `STATUS.md`'s EEM row, D-059, D-067, and the
frozen `experiments/preregistration/eem-embodied-senses-v2.md` — where this ADR
writes `-0.0189`. They are the same measurement at four decimal places: the
adversarial review re-measured the underlying value as `-0.018850894…`, which
rounds to `-0.0189` and truncates to `-0.0188` (that re-measurement is the
reviewer's, recorded here as reported rather than as this lane's own
measurement). The rounded form is used here because it is the correct rounding;
**no older record's number is changed** — the preregistration is frozen and
would be untouchable in any case — and
nothing downstream depends on the fourth decimal — D-067's turn-circle sizing
cites the figure only as an illustrative worst case and its conclusion is
unmoved either way.

**Scope bound.** Only the *sided* readings are reinterpreted. The persistent
turn bias — the "left circle" of D-054, D-056, D-058 and the turn circle sized in
D-067 — comes from the open-arena baseline, where both lateral slots read 6.0
and are therefore invariant under the transposition; the yaw actuation sign is
untouched by this ADR. Frontal, collision, light, and vertical channels are
likewise unaffected.

**D-056 is downgraded.** That row records a manually observed "sustained circling
flight that visibly dodged placed blocks". It was never sided-verified against a
trace, and the same row states the flight matched "the calibrated offline
prediction of a persistent left bias with **~30% sensory modulation**" — quoted
in full because that 30% is the strongest form of the counter-argument and
paraphrasing it away would be avoiding it. The counter-argument does not
survive, but on its own terms: a persistent bias with amplitude modulation is
satisfied by a modulated turn rate, which requires no *sided* steering at all,
and an unaided observer cannot separate a 30%-modulated left circle that dodges
from one that does not. It is now recorded as **non-evidence for or against
the swap in either direction**, and it is not cited as support for this ADR. The
measurement above is the reason it cannot be: under the wiring in effect at the
time, the sided differential commanded a turn toward the obstructed side, so
that observation cannot have been produced by sided avoidance, and equally
cannot be used to argue the swap was harmless.

## The fixture harness is deliberately untouched

`scripts/fixture_harness.py` declares a `HarnessWorld` that is an **abstract
world with no body frame**. Its lateral rays are `x + HALF_WIDTH` and
`HALF_WIDTH - x` — distances to the two walls of a corridor, computed from world
`x` with **no yaw term whatsoever** — and its up/down slots are hardwired to
`RAY_MAX` and `0.0`. There is no handedness there to correct, and its labels are
world-axis names rather than body sides.

It is left byte-identical on purpose. `fixture_harness.replay_implementation_sha256`
folds in `Path(__file__)`, so a one-byte edit rotates the harness code identity
and every harness trace stops replaying against it (`fixture_harness.py:340`
refuses a trace whose `code_hash` no longer matches).

**What that identity is not bound to, checked rather than assumed.** An earlier
draft of this ADR said a one-byte edit would rotate "fixture-physics-pilot-v3 and
`BENCH-FIXTURE-01`". Both are false: pilot-v3's `implementation_hashes` is an
explicit eleven-entry file map that does not list `fixture_harness.py`, and
`BENCH-FIXTURE-01`'s `implementation.script_sha256` is `8a608c52…5edd06`, the
hash of `scripts/benchmark_controller.py`. Nor is it the
`replay_implementation_sha256` field carried by `EXTERNAL-BRIDGE-02…06` or by
`EMBODIMENT-02`: **three different quantities share that key name**, and only the
first is the harness's —

| Emitter | Schema | Quantity |
|---|---|---|
| `fixture_harness.py:412` | harness trace | four package `src` roots + the harness file (`ec8bce73…41bf1`) |
| `replay_minecraft_trace.py:429` | `neurocraft.minecraft_trace_replay.v1` | that script's own file hash (`EXTERNAL-BRIDGE-02…06`) |
| `loop.py:629` | `neurocraft.embodiment.replay.v2` | `embodiment_implementation_sha256` (`EMBODIMENT-02`, `4d6d369e…f72b21`) |

The harness's own value `ec8bce73…41bf1` is pinned by **no committed receipt** —
harness traces are written under the untracked `artifacts/` tree. What actually
argues for leaving the file alone is therefore the trace-replay gate above and
the fact that `AB11-MUTANTS-01` names the function as a
declared code-identity boundary (`EMBODIMENT-01` does not: its only reference to
`scripts/fixture_harness.py` is an unrelated disclosure about replay scope) — not any receipt value. The conclusion is
unchanged; only the mechanism was wrong.

The harness identity's four hashed roots are `connectome-data`,
`controller-runtime`, `neural-runtime` and `protocol` — `embodiment-runtime` is
not among them, so this lane's Python edits provably cannot move it either
(measured: `ec8bce73…41bf1` before and after).

**And it still carries `GRAVITY = 0.04`, the pre-ADR-0014 physics.** Its arena
therefore differs from the live body in both the vertical envelope and the ray
semantics. **No harness result may be read as a prediction about the live body.**
Its traces remain correct as self-consistent artifacts of its own declared
arena, and reconciling it stays recorded follow-up work, exactly as ADR 0014
left it.

## A fifth site is knowingly left wrong: `pilot.py`

Unlike the harness, this one *does* have a body frame, and it *is* transposed.
It is carved out rather than fixed, and the carve-out is stated here because
silence about a known wrong-signed control would be the worse record.

**The site.** `packages/experiment-sdk/src/neurocraft_experiments/pilot.py`
builds its heading as `forward = (cos(yaw), sin(yaw))` in `(x, z)` — its own yaw
convention, not Minecraft's — for which `forward x up = (-forward_z, forward_x)`
is the body's right. That exact expression appears twice, under two different
labels: `_step_physics` (`pilot.py:801-802`) calls it `right_x, right_z` and is
therefore **correct**, while `_sensor_frame` (`pilot.py:734-735`) calls it
`left_x, left_z` and fills slot 2 from it, with slot 3 taking the negation. Its
lateral slots are transposed in exactly the way `FlySensors`'s were, and — as in
`reference.py` before this lane — sensing and actuation disagree, here in the
opposite direction.

**The consequence, stated plainly.** Those slots become `left_clearance` and
`right_clearance`, which `baseline.py:65-69` reads as
`asymmetry = right_clearance - left_clearance` to set both `yaw` and `lateral`.
So **the scripted baseline — the low-capacity non-neural comparator — steers
toward the obstructed side.** A control that is wrong-signed is not a neutral
defect: any comparison drawn against it is affected, in a direction that
flatters whatever it is compared with.

**Why it is not fixed here.** `pilot.py` is one of the eleven entries in
fixture-physics-pilot-v3's `implementation_hashes`
(`experiments/results/fixture-physics-pilot-v3.json`), as is `baseline.py`.
Editing either rotates a committed pilot identity and obliges a pilot
amendment and re-execution — a separate lane with its own preregistration
discipline, not a rider on a body-frame correction. This lane's own scope bound
("no experiment identity rotates") is what makes the carve-out necessary and is
also what makes it a debt.

**Status: disclosed open work.** Until it is fixed, no comparison against the
`scripted_baseline` control may be presented without this defect stated
alongside it.

## Regression pins and test adjustments

Server GameTest lane 20 -> 26. **All six probes, and both fence classes, are new
to this lane** — `git log --all -S` finds no commit in the repository
introducing any of the six method names, `RaySideFence`, or `RayVerticalFence`,
and the count moving by exactly six says the same thing. An earlier draft of
this ADR said the four lateral probes "arrived with the preceding lane"; that
was false and is corrected here.

- `declaredRightRayReportsObstacleOnTheBodysRightSide`,
  `lateralRaySlotsFollowTheBodyUnderSideSwapAndYawFlip`,
  `tickPathSnapshotCarriesTheSameLateralAssignment`,
  `bodyFrameLateralAxisMatchesEngineHandedness` — the lateral axis, through the
  direct sampler, the side-swap and yaw-flip controls, the published bounded
  snapshot history, and `BodyFrameMath`. Expectations come from
  `RaySideFence`, which derives handedness from `Direction.getClockWise()`.
- `verticalRaySlotsMatchTheBodyUpAxisAtLevelPitch` — a block directly overhead
  must shorten the declared `UP` ray to the exact bottom-face distance, and one
  directly underfoot the `DOWN` ray, with the non-vertical rays bit-identical.
- `bodyUpRayTiltsWithPitchInsteadOfTrackingWorldUp` — with the nose pitched up
  40 degrees the up ray must reach a ceiling slab at `1/cos(40 deg)` times the
  vertical distance, so the axis is proven to *follow the body* rather than
  merely to point upward. Its expectation comes from `RayVerticalFence`, which
  derives body-up independently of the cross products under test as
  `Vec3.directionFromRotation(pitch - 90, yaw)` — the direction the body would
  look if it pitched 90 degrees further up.
- `BodyFrameMathTest` — the two pinned yaws now assert south/west and
  west/north, and a new sweep asserts the lateral axis equals `forward x up`
  (`(-forward.z, forward.x)`) at every 15 degrees, so a future edit cannot
  satisfy two yaws by hand while leaving the frame wrong between them.
- `test_lateral_ray_slots_follow_the_body_handedness` — the Python twin's
  *sensing*, parametrised over an arena seed whose single pillar sits due west
  (the body's right at spawn) and one due east, asserting the matching slot
  shortens to the analytic cylinder distance.
- `test_lateral_thrust_accelerates_toward_the_bodys_declared_right` — the Python
  twin's *actuation*, at three yaws (spawn, and after 5 and 13 ticks of
  saturated yaw torque, 0.1 rad each). Unit `lateral_thrust` from rest must
  produce a horizontal velocity parallel to `(-cos yaw, -sin yaw)`; at spawn the
  case additionally pins it to due west, the same side seed 298's pillar
  occupies in the sensing test above, so the two halves of the frame are held to
  one answer rather than two independently plausible ones.

**Three negative controls, executed** (not merely reasoned):

1. **The vertical inversion.** `bodyUp` was reverted to `forward.cross(right)`
   against the corrected `right` — precisely the state a one-line fix would have
   produced. Exactly the two new vertical tests failed
   (`"a block directly overhead must shorten the declared UP ray, but the slot
   that moved was DOWN"`; the pitched probe read `4.10999...`, the floor
   distance, instead of `1.89000...`), and all 24 others passed — confirming both
   that the new pins bind the inversion and that nothing else guards that axis.
2. **The Python transposition.** `reference.py` was reverted to the swapped
   tuples and the new parametrised test failed on both sides
   (`"a pillar on the body's left must shorten the declared left ray; slots that
   moved were [3]"`). Before this lane the full Python suite passed identically
   with the swap in place — 1127 tests, unchanged — which is the direct evidence
   that the Python site had no coverage at all.
3. **The Python actuation inversion.** `_substep`'s lateral axis was reverted to
   `(cos, sin)` and the whole Python suite re-run: **exactly the three new
   actuation cases failed and all 1130 other tests passed.** That is the direct
   measurement behind the "an unasserted axis does not stay correct" rule above
   — the two tests that already drove `lateral_thrust=1.0`
   (`test_arena_envelope.py:58`, `test_reference_embodiment.py:187`) assert
   envelope containment and stayed green with the axis pointing the wrong way.

**Test adjustment, declared.** `BodyFrameMathTest.yawZeroMapsForwardToPositiveZAndRightToPositiveX`
asserted `right == (+1, 0)` — east, the body's left. It was the unit-test half of
the defect, and it is corrected rather than deleted. No other existing
assertion, timeline, or pinned value changed.

## Consequences

- **One Python code identity rotates, and no declared identity does.** Measured,
  not assumed: `embodiment_implementation_sha256` moves
  `2ba65340...11cdd` -> `37a8af7c...1bff59`, because it hashes the *source bytes* of
  five package trees including `embodiment-runtime`. (Both the sensing and the
  actuation corrections land in that hash; the intermediate value the first draft
  of this ADR quoted, `0eada044...45842`, was the sensing-only tree and no longer
  exists on any branch.) The three pinned reference
  identities are unmoved and their literals in `tests/embodiment/test_identity.py`
  are untouched — `REFERENCE_EMBODIMENT_SHA256 = 3103af43...b19ed`,
  `REFERENCE_SENSORY_BOUNDARY_SHA256 = 4efc3e44...49efe`,
  `REFERENCE_MOTOR_BOUNDARY_SHA256 = aca18951...9148e3` — because
  `EmbodimentIdentity.to_identity_material` folds in declared metadata (channels,
  arena constants, determinism level), never source bytes. No test literal pins
  the rotated value; `test_embodiment_trace.py` asserts only that it differs from
  the harness hash and is 64 characters. Committed receipts *do* record it, in at
  least four files across two receipt families: `EMBODIMENT-02-replay.json` and
  `EMBODIMENT-02-trace.ndjson` carry `4d6d369e...f72b21` (under
  `replay_implementation_sha256` at `loop.py:629` and under `code_hash`
  respectively), `EMBODIMENT-02.md` restates it, and `EMBODIMENT-01.md` records a
  still older value, `185d5b40...b710b`, under the plain key `code_hash`. Both
  recorded values had already rotated away before this lane began — HEAD's was
  `2ba65340...11cdd` — so the quantity has moved at least twice before now. Those
  receipts stand as history of the trees they were taken on; this lane neither
  edits them nor is the first change to move past them.
- **No receipt is edited.** `EMBODIMENT-01`/`-02` and every sided measurement
  receipt stand as history. `ESCAPE-ARC-01` was re-derived to `3d5560bb...ffd59`,
  byte-identical to the committed evidence, on the sensing-only tree. The
  actuation correction added afterwards was **not** re-executed against it and
  does not need to be: nothing in `scripts/measure_escape_arc.py`'s import
  closure reaches `neurocraft_embodiment` at all — `reference.py` is imported
  only from inside `embodiment-runtime`, `scripts/run_flygym_embodiment.py`, and
  the `tests/embodiment/` suites (none of which is in either probe's closure) —
  so that path is structurally incapable of seeing the change. Stated as an
  import-closure argument, not as a second re-derivation.
- **`EXTERNAL-BRIDGE-06`'s lateral motion is not reproducible on the fixed tree,
  and this is a behavior change for the TRM demo profile.** That receipt records
  the trained-readout projection driving the real mod with a non-zero
  `lateral_thrust` (`0.5758…` to `1.0`) in every one of its nine `APPLIED`
  actuation records, through the wrong-handed `BodyFrameMath` axis. Re-running
  the same trace against the corrected axis will push the body to the opposite
  side, so the recorded positions will not reproduce. The receipt is **not
  edited** — it is correct as history of the tree it was taken on — but it is
  superseded for any future comparison, and this ADR is the forward-looking
  disclosure of that. The faithful pinned profile is unaffected, because its
  projection sets `lateral_thrust = 0.0`.
- **No connectome-side identity rotates.** No adapter, decoder, sign-overlay,
  configuration, calibration, or encoder identity binds a mod source file or the
  embodiment package; `replay_implementation_sha256` is unmoved
  (`ec8bce73...41bf1`) and the fixture-pilot-v3 snapshot re-executes exactly.
- The change touches `mod/src` control-adjacent body code, so any evidence
  bundle enumerating mod control sources is superseded for future candidate runs
  and must be regenerated on the fixed tree, exactly as ADR 0006, 0011, 0012,
  0013 and 0014 recorded for their `mod/src` changes.
- **Live re-observation is pending.** No claim is made about live behavior under
  the corrected frame until a session is observed and recorded. In particular,
  whether the corrected sided differential produces visible avoidance at the
  measured amplitudes (`+0.0054` / `-0.0346` yaw) is unknown; those magnitudes are
  small, and ADR 0014's hover envelope is what makes them expressible at all.
