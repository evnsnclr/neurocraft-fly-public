# ADR 0030: same-hit Minecraft actor metadata

Status: implemented, independently reviewed and focused-tested; new live behavior pending.
Date: 2026-09-05

## Context

The recorded showcase distinguishes blocks from generic entities, but cannot
choose separate player and mob responses from its observation contract. An entity
ray can also hit another pickable entity. Guessing its class from a distance, an
attack event or an older snapshot would invent a measurement.

## Decision

Add `neurocraft.sensors.v4`, inheriting every v3 event and physical field, with six
`ray_actor_kinds` in the existing ray order. Values are 0 for a non-entity hit,
1 for Player, 2 for Mob, and 3 for another entity admitted by the existing ray
predicate. Actor 0 is required exactly when the corresponding hit kind is not 2.
The values describe the same winning hit as the distance and hit kind. They are
engineered Minecraft class metadata, not neural recognition, intention or identity.

The authenticated observation declaration hash is
`e89419ad6b9c174e243e12180d9ffe0f131256a2e0cc2ec14693df0367d601f1`.
Only that identity selects v4. Old identities retain their old serialized shape
and contract hashes. Legacy snapshots leave actor metadata unavailable; merely
changing their identity cannot upgrade them to v4. Both endpoints reject malformed
or inconsistent fields. The Python parser refuses booleans and float spellings as
actor integers; JSON Schema retains its standard mathematical integer semantics.

Minecraft retains its six-block ray range, collider occlusion, broad-phase box,
zero hitbox inflation, entity predicate and block/entity tie rule. The bundled
1.21.5 seven-argument ProjectileUtil entity-picker geometry is restated only to
resolve exact entity/entity squared-distance ties by the smallest Java UUID under
its signed natural ordering. The inside-box and endpoint behavior remains that of
`AABB.clip`; no pick-radius inflation or inside-box shortcut is added.

This revision supplies a later policy with declared actor context. It adds no
body program, neural classification mechanism, trained readout or service profile.
Source-identified new controller bindings must explicitly opt in; recorded v3
results remain evidence for their own build.

## Verification and limits

The Python author passed 368 protocol tests, including 142 new actor cases, plus
protocol type/lint/format checks. The Java author passed 43 selected JUnit tests,
one existing sensor-shape GameTest and four new actor GameTests. These exercise
real Player/Cow/ArmorStand hits, occlusion, spectator exclusion, both signed UUID
halves and insertion orders, differential distances against bundled Minecraft,
edge geometry and real-tick bridge serialization. The shared v4 golden is checked
at both endpoints; historical v1/v2/v3 golden bytes remain unchanged.

Local Java logs, retained outside Git:

- `artifacts/delivery/actor-v4-java-compatibility-final.log`, SHA-256
  `efd7fb8ae5bfd6e2c5a55e3ab48a2a6a616cb5ba505855d7042ddb5dfea5fda6`.
- `artifacts/delivery/actor-v4-java-gametest-contained-fixture.log`, SHA-256
  `55af027212eecbfae9815786e3c8767736c1256c2b012a4f6d7c181bea0cd762`.

Earlier failed fixture/selection attempts remain in their own logs and are not
passes: one registration race, two runs selecting zero tests, and a fixture whose
wall lay inside the forward ray. The corrected fixture contains the full segment.
Independent Python and Java source reviews found no blocking issues; the Java
reviewer also inspected the bundled picker bytecode. Reviewers did not rerun those
author checks. Full integrated verification and a live distinct-response session
remain due; no merge or publication is implied.
