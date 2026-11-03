# ADR 0017: Entity-aware sensor rays

_Sensory boundary widening, drafted 2026-08-31_

---

| Field | Value |
|---|---|
| **Status** | Implemented; GameTest-pinned with four executed negative controls. Adversarial review returned LAND-WITH-FIXES (1 HIGH, 3 lesser); all four findings are addressed in this revision — see *The declared predicate* for the corrected projectile resolution and *Consequences* for the ADR 0015 supersession |
| **Date** | 2026-08-31 |
| **Authority** | `mod/src/main/java/dev/neurocraft/fly/entity/FlySensors.java` — `obstructsSensorRay` (the declared predicate) and `rayDistance` (the two-sweep nearest-hit rule) |
| **Precedent** | ADR 0003 (bounded actuation and the original ray semantics); ADR 0008 (mode taxonomy — this is the body, not policy); ADR 0015 (the same "declared engineering, fix the body not the connectome" posture, and the ADR that declared what a ray was in the first place) |
| **Evidence** | Six server GameTests in `FlyEntityGameTests` with the `RayEntityFence` expectation class; the server lane moves **26 -> 32** and all 32 pass; four executed negative controls (see *Regression pins*) |

## Context

The operator's standing observation is that the fly does not react to him. One enum
value explains all of it.

`FlySensors.rayDistance` cast each of the six body-aligned rays with exactly one
call:

```java
HitResult result = level.clip(new ClipContext(
        origin, end, ClipContext.Block.COLLIDER, ClipContext.Fluid.NONE, entity));
```

`Level.clip` walks **blocks**. Entities are not in its search at any
`ClipContext.Block` value — the enum selects *which block shapes* count
(`COLLIDER`, `OUTLINE`, `VISUAL`, `FALLDAMAGE_RESETTING`), never whether entities
are consulted at all. There is no setting of that argument that would have made a
player visible.

So a player standing in front of the fly, walking at it, waving, or attacking it
produced a **bit-identical** sensor vector to an empty arena. The ray passed
straight through the body and reported whatever wall lay behind it. Every
downstream channel inherited that: `threat_left`, `threat_right` and
`threat_front` in the EEM encoder are `clip(1 - d/6, 0, 1)` over those same
distances, so the operator read as vacuum on all three. The one channel that
could have noticed him — `touch`, driven by
`horizontal_collision or vertical_collision` — only fires on contact the fly
itself initiates.

**This is a body defect, not a connectome defect**, in the same sense as ADR 0014
and ADR 0015: the neural side was answering correctly about a world that had been
described to it incompletely.

## Decision

Sweep the same segment twice — once against blocks, once against entities — and
keep the nearer hit.

```java
Vec3 end = origin.add(direction.scale(RAY_LENGTH_BLOCKS));
double nearest = RAY_LENGTH_BLOCKS;
// blocks: unchanged call
// entities: ProjectileUtil.getEntityHitResult(..., new AABB(origin, end),
//           FlySensors::obstructsSensorRay, ENTITY_HITBOX_INFLATION_BLOCKS)
return nearest;  // min of whichever hit
```

`RAY_LENGTH_BLOCKS` is unchanged at 6.0. The body-aligned directions are
unchanged (ADR 0015). The origin is still the bounding-box centre. The returned
value still means *"distance to the nearest obstruction along this ray, clamped
to the ray length"*, so the Python encoder's `proximity = clip(1 - d/6, 0, 1)`
keeps its exact meaning — the set of things that can be an obstruction is the
only thing that got wider.

### The declared predicate

An entity obstructs a sensor ray if and only if all four clauses hold. This is
declared engineering exactly as the ray count, length and directions are; nothing
in it is derived from the connectome or from fly biology.

| Clause | Effect |
|---|---|
| not a `FlyEntity` | flies do not sense each other |
| not removed | a despawned entity is not in the world |
| not a spectator | an observer must not perturb the run by being looked at |
| `isPickable()` | vanilla's own "a ray cast can hit this" contract |

Clause 4 carries the weight, and it is stated here rather than left implicit
because its resolution is not obvious from the name. Read from the shipped
1.21.5 bytecode — the whole `isPickable()` override set under
`net.minecraft.world.entity` is fifteen declarations, of which these five decide
the cases named below:

| Declaring class | `isPickable()` body |
|---|---|
| `Entity` | `iconst_0; ireturn` — false |
| `LivingEntity` | `!isRemoved()` |
| `Projectile` | `getType().is(EntityTypeTags.REDIRECTABLE_PROJECTILE)` |
| `AbstractArrow` | `Projectile.isPickable() && !isInGround()` |
| `ShulkerBullet` | `iconst_1; ireturn` — true |

and `data/minecraft/tags/entity_type/redirectable_projectile.json` holds exactly
three values: `minecraft:fireball` (the ghast's `LargeFireball`),
`minecraft:wind_charge`, `minecraft:breeze_wind_charge`. The other ten overrides
are not claimed about below, but were read too rather than left as an unexamined
remainder: `Interaction`, `BlockAttachedEntity` (item frames, paintings, leash
knots), `EnderDragonPart` and `EndCrystal` return an unconditional `true`;
`AbstractBoat`, `AbstractMinecart`, `PrimedTnt` and `FallingBlockEntity` return
`!isRemoved()`; `ArmorStand` is `LivingEntity.isPickable() && !isMarker()`; and
`EnderDragon` itself returns **`false`** — the dragon is sensed through its
parts, not its body. Therefore

- **players and mobs count.** `Player` overrides neither `isPickable` nor
  `canBeCollidedWith`, so the operator's body reaches the predicate through that
  `LivingEntity` override. This is the case the whole change exists for.
- **item entities and experience orbs do not count.** `ItemEntity` and
  `ExperienceOrb` extend `Entity` directly and declare no `isPickable`, so they
  take the `false`. This exclusion is wanted on its own terms rather than merely
  inherited: the fly's bounded actuation passes straight through a dropped item,
  and a proximity reading for something that cannot obstruct the body would be a
  phantom threat.
- **four projectile types do count, and that is accepted deliberately.** A
  ghast fireball, a wind charge and a breeze wind charge resolve `true` through
  the tag; a shulker bullet resolves `true` through its own
  `iconst_1; ireturn`. Everything else that extends `Projectile` — arrows,
  snowballs, ender pearls, eggs, small and dragon fireballs, thrown potions,
  fishing bobbers — resolves `false`, arrows twice over, since they are not in
  the tag and `AbstractArrow` additionally requires `!isInGround()`. **Accepting
  the four is a declared engineering choice, not an accident of inheritance.**
  The predicate's declared purpose is to admit bodies the fly's bounded
  actuation cannot pass through, and a redirectable projectile is precisely a
  thing that physically interacts with the body: a wind charge knocks the fly
  around. Admitting it is consistent with that purpose, and it gives the
  operator a legitimate way to present a fast-approaching stimulus without
  standing in the arena. It is pinned by
  `taggedRedirectableProjectileObstructsWhileUntaggedProjectileDoesNot`, which
  asserts the tag membership as well as the distance so that a datapack or
  version change moving `minecraft:wind_charge` out of the tag names its own
  cause.

  **This paragraph is a correction, recorded rather than quietly rewritten.**
  The first draft of this ADR, of the `FlySensors` javadoc, of D-070, of
  `STATUS.md` and of the changelog all asserted that in-flight projectiles could
  not obstruct a ray because they "never override `isPickable`". That was false:
  `Projectile` does override it. Adversarial review caught it from the same
  bytecode and demonstrated it by execution, spawning a `minecraft:wind_charge`
  on the forward ray and watching the reading drop from 3.5 to 0.84375. Both the
  disassembly and that number were then **re-derived independently** while
  applying this fix — negative control 4 below — rather than accepted on the
  reviewer's word. All five sites now state the resolution the tag actually
  produces.

**`Entity.canBeCollidedWith()` was considered and rejected.** It reads like the
symmetric partner of the block side's `ClipContext.Block.COLLIDER`, and that
symmetry is why it was the first candidate. It is wrong: in 1.21.5 it is `false`
on `Entity` and overridden by neither `LivingEntity` nor `Player` (hard
entity-entity collision belongs to boats and shulkers). It would have excluded
the operator — the one case this change exists to fix — while looking principled.

**Clause 1 is currently redundant, and that is measured, not assumed.** The
sensing fly is already excluded upstream by
`Level.getEntities(Entity except, AABB, Predicate)`, whose first argument the
shipped bytecode captures into the filtering lambda. A *second* fly is excluded
by clause 4, because `FlyEntity extends Entity` directly and so inherits
`isPickable() == false`. Deleting clause 1 and re-running the server lane leaves
**all 32 tests passing**, so no test binds it. It is kept as declared intent that
survives a future change making the fly pickable — to let an operator target it,
say — which would otherwise switch fly-to-fly sensing on silently as a side
effect. Recorded as a declared clause with no coverage, not as a pinned one.

### The hitbox inflation is declared at zero

`ProjectileUtil` has a six-argument `getEntityHitResult` that looks like the
natural call. It hard-codes `0.3F` and forwards to the seven-argument form
(verified in the disassembly: `ldc #21 // float 0.3f`). That constant is
projectile aim leniency — it is what makes arrows feel generous — and inheriting
it silently would have been wrong twice over: the ray would report an obstruction
0.3 blocks nearer than the body actually is, and because the scalar is consumed
as `clip(1 - d/6, 0, 1)`, that would have become a standing **+5% proximity bias**
on every entity reading, visible nowhere in any identity document. The
seven-argument overload is called directly with an explicit `0.0F`, and negative
control 2 below pins it.

## Identity analysis: nothing declared rotates

The question this lane had to answer before proceeding is whether *"what counts
as an obstruction"* is part of any hashed identity. If it were, this would need a
new frozen identity under freeze-then-measure rather than an in-place edit. It is
not. Four independent checks, each verified rather than reasoned:

1. **The EEM encoder identity does not name it.**
   `EemEncoderDocument.to_identity_material()` hashes a `ray_semantics` block
   containing exactly `front_ray_index`, `left_ray_index`, `right_ray_index`,
   `ray_order`, and a `source` string. That string names the *order* in which the
   mod casts the rays, not what they hit. No obstruction predicate, no
   `ClipContext`, no ray length.
2. **The observation-schema identity does not name it.**
   `observation_schema_hash` is `canonical_sha256(PhysicalFeatureEncoder.SCHEMA_MATERIAL)`
   (`controller.py:188`), and that material is four channels — each a slot index
   plus `x / ray_max_distance_m` — plus `ray_order` and a schema string.
3. **No document in the repository names it.** A grep for
   `collider|obstruction|clipcontext` across `packages/`, `experiments/`,
   `docs/`, `services/`, `scripts/` and `tests/` returns exactly two hits
   outside this ADR, both in ADR 0015's prose (lines 118 and 122) — the same
   sentence that declares the ray semantics as engineering. Prose in an ADR is
   not hashed material, so nothing rotates. **It is, however, a standing
   declaration, and this lane changes it**: see the ADR 0015 supersession bullet
   under *Consequences*, which records that those exact lines are superseded on
   the obstruction clause and on nothing else.
4. **The frozen preregistrations cannot move, structurally.** Every EEM
   measurement builds `SensorFrame`s directly by slot from a digest stream — the
   eem-v1/v2 and escape-v2 preregistrations specify "rays from two digest bytes
   each" and hold "the blocked ray at 0.5 m and every other ray at 6.0 m". None
   of them runs Minecraft, so no mod-side ray change can reach a frozen number.

**Conclusion: this is an in-place edit, not a new frozen identity.** No adapter,
decoder, encoder, dynamics, arena, control-matrix or pilot identity binds the
mod's obstruction rule.

### The one identity that does rotate

`neurocraft.gametest_code_bundle.v1` hashes six source roots wholesale, `mod/src`
among them (`prepare_gametest_demo.py:320-339`), so it moves with any `mod/src`
edit. **Measured like-for-like, not assumed:**

| Tree | `code_hash` | files |
|---|---|---|
| HEAD `83cd774` | `c9d4006d…4a047` | 125 |
| this lane, first draft | `0a0bd82e…3259c2` | 126 |
| this lane, after adversarial-review fixes | `bee3c162…4afe5` | 126 |

The `+1` is `RayEntityFence.java`. The third row is the current tree and the live
one: the review fixes added a sixth GameTest method and rewrote two javadocs
inside `mod/src`, which moves the hash without adding a file. The second row is
kept so the file-count reasoning stays checkable, not as a live identity. All
three were measured on a `_spmv`-bearing tree, so the comparison is
like-for-like (see the finding below). As D-068 and D-069 already recorded for
their own `mod/src` changes, this quantity had already drifted away from every
committed receipt before this lane began, so nothing is newly invalidated; any
future evidence bundle enumerating mod sources must be regenerated on this tree.

**A finding surfaced while measuring it, disclosed rather than filed away.** That
bundle is **not a pure function of the git tree**. Two gitignored local build
products — `packages/neural-runtime/src/neurocraft_neural/_spmv.so` and
`_spmv.build.json` — sit *inside* a hashed source root, so a developer machine
that has built the native extension and a clean checkout hash the same commit
differently (`c9d4006d…4a047` with them, `b2f48ef7…31702e` without: 125 files
against 123). The first measurement attempt in this lane compared a
`_spmv`-bearing working tree against a clean worktree and produced a spurious
`+3` file delta; the table above is the corrected like-for-like pair. This is a
pre-existing property of the bundle definition, not something this lane
introduced, and it is recorded as open follow-up rather than fixed here — the
fix belongs with whoever owns the bundle's file selection.

## The Python reference twin is deliberately left alone

`packages/embodiment-runtime/.../reference.py` implements the same sensory
boundary for the non-Minecraft body, and ADR 0015 corrected it in lockstep with
`FlySensors` because both had the same transposed body frame. This lane does
**not** mirror the change, and the asymmetry is deliberate.

`ReferenceEmbodiment` is a single point mass in a bounded box with a static field
of vertical cylinders. `_ray_distance` intersects exactly two things: the six
planes of the arena box, and the pillars, analytically. **There is no entity
concept in that arena to mirror** — no second body, no items, no observer, no
spawning of any kind. Adding entity-awareness there would mean inventing an
entity system first, and then the divergence would be between one real world and
one imagined one rather than between two worlds with different contents.

So the two bodies now differ in what their rays can see, and that is correct:
they always differed in *what is there to see*. What ADR 0015 had to keep aligned
was the **frame** — the same connectome must not fly mirrored in the two worlds —
and the frame is untouched here. The rule this lane follows is the narrower one:
mirror a convention, do not mirror a world.

**Checked, not assumed:** the full Python suite was re-run on this tree
(`make test`, see *Gates*), including the cross-language pins in
`tests/embodiment/` and `tests/protocol/`. No golden, parity, or identity test
depends on the mod's obstruction rule, which is the same conclusion the identity
analysis reaches from the other direction. `embodiment_implementation_sha256` is
unmoved because no Python source byte changed in this lane.

## Determinism and cost

**Determinism.** `ProjectileUtil.getEntityHitResult` keeps the strict minimum of
`from.distanceToSqr(clipPoint)` over `Level.getEntities`, whose iteration order is
chunk-section order and is not a declared invariant; the disassembled comparison
is `dcmpg; ifge`, so a tie keeps whichever entity the list yields **first**. That
tie-break is real, and it cannot reach this sensor's result: the entity object is
discarded and only `origin.distanceTo(hit.getLocation())` is returned, and two
clip points that tie on `distanceToSqr` from `origin` are by definition
equidistant from it. `Vec3.distanceTo` and `Vec3.distanceToSqr` disassemble to
the identical `dx, dy, dz` expression differing only by the final `Math.sqrt`, so
bit-equal squared distances give bit-equal distances. **No tie-break needs
declaring, because no tie is observable in the scalar the protocol carries.**

**Cost.** One `Level.getEntities` broad phase per ray, six per server tick. The
search box is `new AABB(origin, end)` — the segment's own bounding box,
degenerate in two dimensions for an axis-aligned ray — not an inflated box and
not a single unioned 12x12x12 cube around the body, which would have swept eight
times the volume in the worst case for one fewer query. A point strictly inside
an entity's box is inside both boxes, so the tight box admits every entity the
segment actually passes through; only exact face tangency is excluded, where the
reported distance would be unchanged anyway.

## Regression pins and negative controls

Server GameTest lane **20 -> 26 -> 32**. The six new probes are
`declaredRayShortensOnALivingBodyInItsPath`,
`sensorRaysIgnoreTheSensingFlyAndEveryOtherFly`,
`nearerOfBlockAndEntityWinsInBothOrders`,
`nonPickableEntitiesDoNotObstructAnyRay`,
`taggedRedirectableProjectileObstructsWhileUntaggedProjectileDoesNot`, and
`blockOnlyReadingsAreUnchangedWhenNoEntityIsPresent`. They live in
`FlyEntityGameTests` because `mod/build.gradle:159-166` expands a **single**
`fabric-gametest` entrypoint into `fabric.mod.json`; a `@GameTest` method in any
other server class would never run. **Execution is confirmed by count**, not
assumed: the runner reports `32 tests are now running` against 26 on HEAD, and
`All 32 required tests passed`, twice, at two independently randomized test
positions.

**The projectile probe is pinned against projectile motion**, which is the one
way it could have been flaky. No world tick elapses between the spawns and the
samples — a `@GameTest` body runs to `succeed()` inside a single tick — and both
targets additionally have `setDeltaMovement(Vec3.ZERO)` and `setNoGravity(true)`
applied before the reading, so neither can drift, fall, or strike anything while
the probe runs; both are `discard()`ed before `succeed()` so no live wind charge
outlives the test. It asserts the tag membership of both targets as well as the
distances, so a datapack or version change that moved `minecraft:wind_charge`
out of `REDIRECTABLE_PROJECTILE` would name its own cause instead of surfacing
as an unexplained number.

Expectations come from `RayEntityFence`, which computes entry distances by plain
arithmetic on the target's own `getBoundingBox()` or `BlockPos` and never calls
`AABB.clip` — the routine the production sweep uses internally — so the pins
cannot agree with a consistent error in the code under test. Every probe asserts
its own geometry first (`straddlesDepthRay`), including the negative ones, so
"the reading did not change" can never pass because a target drifted out of the
ray.

**An armor stand stands in for the operator**, because the dedicated GameTest
server has no player. `ArmorStand.isPickable()` is
`super.isPickable() && !isMarker()` over `LivingEntity.isPickable()`, which is the
identical clause a `Player` takes, and a non-marker armor stand is motionless and
AI-free so the geometry stays exact. The substitution is in the predicate's
argument, not in the predicate.

**Four negative controls, executed** (not merely reasoned). Controls 1 and 2 were
taken on the 31-test lane, before the projectile probe existed; their counts are
recorded as measured rather than restated against the current 32.

1. **The sweep removed.** `rayDistance` reverted to the block-only single clip.
   Exactly two probes failed —
   `"a living body in the ray path did not shorten the forward ray: still 3.5.
   The rays are blocks-only again."` and `"a body nearer than a block did not win
   the forward ray: reported 2.5, expected 0.75"` — and the other 29 passed,
   including the three exclusion/regression probes, which is correct: with
   nothing seen, "nothing is seen" is trivially true.
2. **The vanilla inflation restored.** `ENTITY_HITBOX_INFLATION_BLOCKS` set to
   `0.3F`. The exact-distance assertions caught it immediately: the forward ray
   reported `0.44999998807907104` against an exact near face of `0.75` — short by
   exactly the inflation. This is the direct evidence that the zero is pinned
   rather than merely written down.
3. **Clause 1 deleted.** `!(candidate instanceof FlyEntity)` removed from the
   predicate: **all 32 tests still passed** (re-executed on the current lane),
   which is the measurement behind the "currently redundant, no test binds it"
   disclosure above. It is reported as a null result rather than quietly
   omitted, because the alternative would be letting the ADR imply coverage that
   does not exist.
4. **Projectiles excluded from the predicate.** A fifth clause
   `&& !(candidate instanceof Projectile)` added temporarily. **Exactly one test
   failed** — the new probe, at `"declared predicate refused a redirectable
   projectile"` — and the other 31 passed, so the probe binds the accepted
   behavior and nothing else in the lane depends on it. The distances behind
   that verdict were then read out directly, by making the same probe report
   them: with the predicate intact the forward ray reads **`3.5` open and
   `0.84375` with a `minecraft:wind_charge` on it**, against
   `RayEntityFence`'s independent `0.84375` (charge `minZ` 0.15625 beyond the
   probe position, origin at the body's box centre), while the back ray reads
   `4.5` both with and without a `minecraft:arrow` astride it. `0.84375` is the
   same figure the adversarial review measured independently.

**No existing test changed.** All 26 prior tests pass untouched, and the risk to
them was structurally nil: every one of them spawns only `FlyEntity` and no other
entity type, so no qualifying entity exists anywhere in their worlds.

**One test-geometry note, declared.** The default 8x8x8 GameTest structure is
walled, so a body centred at `(4.5, *, 4.5)` reads its own shell at 3.5 and 4.5
blocks rather than the 6.0 maximum. This was discovered by execution, not
foreseen: the first draft of three probes asserted an absolute 6.0 open-arena
reading and failed. They now assert **deltas against an open baseline taken in
the same structure**, which is both correct and stronger. The single remaining
absolute check separates a self-hit (at most 0.19, the fly's half-width) from any
real reading, with a floor of 1.0.

## Consequences

- **This widens what the fly can sense. It makes no claim about how the fly will
  behave.** The connectome, the encoder, the decoder, every gain and every
  population binding are untouched. What changes is that a channel which
  previously read a constant now varies when a body is near.
- **"The fly reacts to you" must not be asserted.** Not by this ADR, not in
  `STATUS.md`, not in the changelog. A sensory channel that can now move is a
  necessary condition for a visible response, not evidence of one. The measured
  sided yaw amplitudes on the committed EEM boundary are small (`+0.0054` /
  `-0.0346`, ADR 0015), and whether an operator standing in front of the fly
  produces any perceptible change is **unknown and unmeasured**. It becomes
  claimable when a live session is observed and recorded, and not before.
- **ADR 0015's obstruction clause is superseded, on that clause alone.** ADR 0015
  lines 118–124 are the standing declaration of the sensory boundary, and they
  name `ClipContext.Block.COLLIDER` as *what counts as an obstruction* in the
  same sentence that declares the six body-aligned rays, their 6.0-block length
  and their bounding-box-centre origin. **This ADR supersedes exactly the
  obstruction half of that declaration** and nothing else: the ray count, the
  body-aligned directions, the length and the origin stand unchanged, and the
  block sweep itself is still `ClipContext.Block.COLLIDER` with fluids ignored.
  What is superseded is the implication that a block clip is the *whole* of what
  a ray can hit; the declaration is now the block clip plus the entity sweep
  under `obstructsSensorRay`, nearer hit wins. **ADR 0015 is not edited** — it is
  correct as the declaration of the tree it was written on. That is the
  repository's established idiom: ADR 0015 left `EXTERNAL-BRIDGE-06` unedited and
  superseded it forward from its own Consequences ("the forward-looking
  disclosure of that"), ADR 0006, 0011 and 0014 supersede prior evidence bundles
  the same way, and ADR 0003 supersedes its own pre-implementation protocol
  wording in an addendum rather than by rewriting the sections it replaces. This
  bullet is the forward-looking disclosure of that supersession, and it is why
  the identity analysis above can cite ADR 0015 lines 118 and 122 as evidence
  without leaving the reader to discover unaided that those very lines are the
  ones this lane changes.
- **No receipt is edited.** `EXTERNAL-BRIDGE-*`, `EMBODIMENT-*`, `LIVE-CIRCUIT-01`
  and every sided-measurement receipt stand as history of the trees they were
  taken on.
- **One recorded figure's provenance predates this lane, and nothing rotates
  because of it.** D-059 records that "the calibration's varied-stream p95 sits
  ~20× above the live dynamic range", and `STATUS.md`'s EEM row carries the same
  ~20× compression. That observation was taken on live sessions whose rays were
  **blocks-only**, so the live ray-value distribution it describes is the
  narrower, pre-ADR-0017 one; this lane widens that distribution by letting
  bodies shorten a ray. **No identity or number moves**: the eem-v1 and eem-v2
  calibrations are driven by a digest stream, not by Minecraft, and the frozen
  preregistrations build `SensorFrame`s by slot without running the game (see
  *Identity analysis*, check 4). The clause is recorded so that a future
  comparison does not read the ~20× figure as a measurement of the current
  sensory boundary. Re-measuring it against entity-aware rays is open work, not
  a claim made here.
- **Any evidence bundle enumerating mod control sources is superseded** and must
  be regenerated on this tree, exactly as ADR 0006, 0011, 0012, 0013, 0014, 0015
  and 0016 recorded for their own `mod/src` changes.
- **The code-bundle purity finding above is open follow-up**, tracked here and
  not fixed in this lane.
- **Live re-observation is pending.** No live session has been run against this
  change.
