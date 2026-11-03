# Encoder identity `eem-loom-v2` — entity-gated approach contrast

This declares `full_mcns_loom_v2` and `full_mcns_loom_program_v2`. The three
loom-v1 threat channels admit proximity only when that ray's nearest hit is an
entity. Blocks and misses contribute exactly zero to those channels. The
program lane still uses hand-chosen neural readouts to select scripted body
programs. This change is an engineering filter for the demo's threat sense,
not a biological model of looming or evidence of natural behavior.

## Sensor contract and compatibility

`SensorFrame` and its legacy JSON shape remain unchanged. `SensorFrameV2` adds
the discriminator `sensor_schema: "neurocraft.sensors.v2"` and the required
`ray_hit_kinds` array. It has exactly six integer elements, each `0` (none), `1`
(block), or `2` (entity), in the distance array's order: forward, back, left,
right, up, down. Booleans, floats, missing kinds, wrong lengths, unknown
discriminators, and extra fields are refused. Historical frames never acquire
inferred kinds.

The mod derives distance and kind from the same nearest-hit calculation, with
the ADR 0017 entity predicate and zero hitbox inflation unchanged. An entity
wins an exact block/entity distance tie. A hit at maximum range keeps its
kind; only a double miss is `none`. An entity hidden behind a closer block is
therefore a block hit. This transports no entity identity or player label.

`observation_schema_hash` in the authenticated controller identity selects the
wire shape: the newly declared hash below requires v2; all existing hashes,
including opaque historical fixture identities, retain the exact legacy
shape. The service enforces that choice before queueing an observation, and
the full-brain controller enforces it again before stepping. Neither a frame's
discriminator nor a supplied kind can upgrade a legacy session. The mod emits
the old shape for old profiles even though its sampler now also measures
kinds. Protocol and trace envelopes stay v2; this is a versioned nested sensor
contract, not a migration of old recordings.

The v2 observation identity hashes `SENSOR_V2_DECLARATION` in
`packages/protocol/src/neurocraft_protocol/sensor_contract.py`: the frozen
four-feature schema hash, ray order, typed-kind semantics, tie rule, and
predicate declaration. The existing four numerical feature values are
unchanged. Keeping the original `SensorFrame` dataclass fields also preserves
the older embodiment seam's width declaration and identities.

## Formula

For front ray 0, left ray 2, and right ray 3:

```text
raw = the unchanged loom-v1 proximity formula when ray_hit_kinds[ray] == 2
raw = 0.0 otherwise

drive_t = clip(raw_t - m_(t-1), 0, 1)
m_t = (1 - 1/8) * m_(t-1) + (1/8) * raw_t
```

The gate acts before temporal contrast. Front proximity retains its factor
of 1/2. Left/right formulas, first-tick initialization (`m0 = raw0`), adaptation
rate, population placement, and gains retain the v1 arithmetic. Touch retains
two-sided contrast and light remains absolute. Graph, dynamics, TTMn pool,
and DNa01 steering membership remain unchanged.

A stationary entity crossing into a ray at 4 m produces a proximity step,
then decay; this is not exclusive detection of approaching entities. The fly
moving toward a stationary entity also counts. An entity stepping aside to
expose a wall causes no new positive threat drive. Wall distances still enter
the original physical feature encoder, and light still enters the network:
zero gated threat does **not** imply zero neural state, jump readout, or motor
output.

## Identities and calibration

| What | v2 SHA256 |
|---|---|
| observation contract | `032b6a74d57b37f9fc1883b4865c89ec2eeccc5c9084831e29cd986738559308` |
| encoder document | `9c41662421849ebf0542c5dbfd54f3f364eda819f2e504ea33484de39a9bf1c3` |
| adapter | `5fbb0a04f246fd657753c7bd41fd262e1e79bf8f3046095dfeeab6e06d5037a6` |
| calibrated decoder | `bfc5dd6a798cc23082f814488613869f2bcff7272c8fde7d9cc606466c23c692` |
| calibration document | `b3232f10d6bbd3dbdda7b6ab51c693dbbcae06aae67ff0783c8fce25639bfea0` |
| body-program policy | `8140773d7ded100b13742121c89f65270b62800a4a13388cd0b8663b88ed27e3` |
| graph, unchanged | `b33bdda8b1a9abfc6b23f7346197442546f7a2aec0a03fc7c07fbdfe4f67c98c` |
| dynamics, unchanged | `5da651e72477a186c82a6091afa0f415b7e4a182fcd8f014870922143d526d8e` |
| DNa01 pair binding, unchanged | `01f0469567ee41c637cbac282c939ab3dc46f830bd30dd4a5354dca91fece1a2` |

The old observation hash remains
`fbfdd38a650f1496dbf946eea7ab686f039f271e5250139259a3931f95335356`.
All v1 profile and historical calibration identities remain intact. The new
decoder and policy hashes deliberately rotate: the decoder binds the encoder
and calibration bytes, and the policy binds those readouts. Their numerical
rules did not change. Suppressing those rotations would misstate the binding.

`data/calibration/full-readout-calibration-loom-v2.json` was measured on the
full 166,700-node / 25,582,938-edge artifact, using the existing loom-v1 numeric
stream and seed, with all six kinds explicitly declared `entity`. This is a
declared calibration stimulus, not a recorded Minecraft scene. Its wing/jump
sample digests and all distribution statistics match the committed v1
measurement exactly. The independently regenerated scales therefore remain:

| Scale | Decimal | Float hex |
|---|---:|---|
| wing | 104.24708186405311 | `0x1.a0fd0307362bep+6` |
| jump | 6.821555884690886 | `0x1.b4945f22255b6p+2` |

The new schema is `neurocraft.full_readout_calibration.loom.v2`. V1 and v2
loaders refuse each other's calibration. The encoder, configuration,
stimulation, and observation declarations each distinguish v2.

## Measured offline probes

Paired full-artifact runs, reset between conditions, same numeric sensors,
8 kernel threads. Kinds are explicitly assigned by the probe; these are
synthetic diagnostics and not exact live sensor replays. Every reported float
in the entity, touch, and receding sequences matched v1 by its `float.hex()`
representation, including TTMn, DNp01 pair mean, motor axes, touch, and LC4
population drives.

| Condition | v1 peak jump | v2 peak jump | v2 front threat |
|---|---:|---:|---|
| single front wall, closes 0.088 m/tick | 0.009068822120896608 | 0.001584184183895226 | exactly 0 |
| front + left corner, same closing ramp | 0.012524462532277214 | 0.001552118083269564 | exactly 0 |
| front + left + right corner, same ramp | 0.023612138348766638 | 0.001555266111171368 | exactly 0 |
| front entity, original 20-tick ramp | 0.02277816208187006 | 0.02277816208187006 | positive, same series |
| stationary entity enters ray at 4 m | 0.015062638329578132 | 0.015062638329578132 | step, then decay |

The wall diagnostics run 80 ticks from 6 m, clamped at 1.5 m. The two additional
corner diagnostics were added after the single-front trial failed to reproduce
the brief's expected v1 threshold crossing. **That original reproduction
failed:** 0.0090688221 is below `ESCAPE_ON`. The corner diagnostics cross it in
v1; all three v2 diagnostics stay below it. The brief's prediction of exactly
zero v2 jump also failed. Acceptance is zero gated wall threat, entity response
equivalence, unchanged touch/light, and no selected escape in the controlled
wall trials, not zero whole-network activity.

The entity ramp independently reproduces the unchanged policy constants:
`ESCAPE_ON = 0.01138908104093503`, `ESCAPE_OFF = 0.005694540520467515`, and
`ESCAPE_URGENCY_SPAN = 0.01138908104093503`. No thresholds or gains were changed.

The separate 30-tick program-policy battery selects CRUISE on all 30 ticks in
each held front/left/right wall condition. The front entity ramp selects
ESCAPE_JUMP on 22 ticks, starting at tick 8; left/right entity ramps select it
on 12/24 ticks, starting at 18/6. Their peak signed DNa01 biases are
+0.880676 / -0.844783 under the existing hand-chosen sign convention. These
measure the policy's requests; they do not establish live body behavior.

Reproduce with the local artifact and pinned annotation files present:

```sh
UV_CACHE_DIR=.uv-cache uv run --frozen python scripts/calibrate_full_readout.py --profile loom-v2 --threads 8
UV_CACHE_DIR=.uv-cache uv run --frozen python scripts/probe_loom_encoder.py --loom-version 2 --threads 8 --json-out artifacts/loom-v2/probe.json
UV_CACHE_DIR=.uv-cache uv run --frozen python scripts/probe_program_policy.py --loom-version 2 --threads 8 --json artifacts/loom-v2/policy-probe.json
```

The probe returns failure if entity equivalence, zero wall front threat,
below-threshold wall jump, or policy-constant reproduction fails. Encoder unit
tests separately cover all three threat rays. Full per-tick diagnostics are
local artifacts; no historical receipt was edited.

## Verification and limits

Tests cover both closed wire shapes, strict kind typing, cross-language v1/v2
goldens, identity-specific downgrade/upgrade refusal before neural stepping,
all three threat rays, crossing/step-aside behavior, calibration pins and
cross-refusal, and a real fly tick delivering typed rays to a recording bridge.
Legacy sensor blocks in all seven checked-in traces round-trip exactly. Six
current trace-v2 receipts pass the existing full trace verifier; the older
EXTERNAL-BRIDGE-01 trace-v1 envelope was already unsupported and is not migrated.

Executed negative controls deliberately inverted the entity predicate and
dropped the wire kinds; the corresponding tests failed and source was
restored. Mutation work must invalidate generated Python bytecode too: a
same-second, same-length restore can otherwise reuse the mutant's `.pyc`.
The final narrow tests were rerun after removing that generated file.

This establishes the transport and declared event filter on offline inputs.
Interactive acceptance, timing under the user's current CPU load, final
integration, and the video remain separate verification work.
