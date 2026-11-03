# Encoder identity `eem-loom-v1` — approach-gated looming

Sensory declaration for the `full_mcns_loom_v1` controller profile. First item of
`PLAN.md` stage 5 (event-gated senses). This document is the declaration; the
executable truth is
`packages/controller-runtime/src/neurocraft_controller/loom_senses.py`, whose
`EemEncoderDocumentLoomV1.to_identity_material()` is what the identity below hashes.

**This is an engineering event gate chosen so a fly can live in a room. It is not
anatomy, and it is not a model of looming.** See *What this does not establish*.

## Why this identity exists

The first live session that survived under the current stack — 2026-09-02, session
`interactive-20260903T025643Z-adbe2bf2`, `make live-demo-escape-v2` on tree `418d902`,
operator-observed and screen-recorded, recorded in the `STATUS.md` pivot section —
ended with the fly wedged in a top corner for its last 83 s, pushing
`forward 0.25 / vertical 0.36` at yaw −0.003 until the operator quit. `STATUS.md`
records three causes. This encoder addresses one of them:

> the escape-v2 encoder deliberately keeps the three threat channels absolute
> (`ESCAPE_V2_ABSOLUTE_CHANNEL_NAMES`, D-060), so a standing wall in the frontal ray
> is a standing LC4 drive and a standing escape — the `ESCAPE-ARC-01` blocked-ray
> stimulus, presented continuously by the world.

D-060's reason for keeping those three channels absolute (temporal contrast adapts a
held threat away) was right for *scoring that arc offline* and wrong for a fly that
has to live in a room. Under this identity a threat channel fires on the **rise** of
proximity, not on its level.

Probe condition 1 below reproduces the live defect offline: under escape-v2 a wall
held in the frontal ray for 600 ticks leaves a **permanent** `forward +0.146 /
vertical +0.207` offset above the profile's own all-open baseline. Under loom-v1 the
same wall leaves `+8.2e-04 / +6.6e-04` — 178x and 315x smaller.

## What changes against `eem-escape-v2`

| channel | escape-v2 | loom-v1 |
|---|---|---|
| `threat_left` | absolute | **rectified approach contrast** |
| `threat_right` | absolute | **rectified approach contrast** |
| `threat_front` | absolute | **rectified approach contrast** |
| `light` | absolute | absolute (unchanged) |
| `touch` | temporal contrast (two-sided) | temporal contrast (two-sided), byte-identical |

Nothing else moves. Channel names, units, target populations
(`lc4_left` / `lc4_right` / `visual` / `bristle`), per-capita placement, the 1.0
gains, the population selectors and their fail-closed counts, the ray-side
convention, and the **raw** formulas are all carried over from
`eem-embodied-senses-v1` verbatim. The decoder is the escape-v2 decoder document
unchanged (`ESCAPE_JUMP_FORMULA_V2`: the rectified TTMn jump term plus the wing
common term); only the `encoder_document_sha256` chained into it differs, which is
what separates the two decoder identities.

## Exact formulas

Raw values (byte-identical to v1, `encode_channel_values`):

```
raw threat_left  = clip(1 - ray_distances_m[2] / ray_max_distance_m, 0, 1)
raw threat_right = clip(1 - ray_distances_m[3] / ray_max_distance_m, 0, 1)
raw threat_front = clip(1 - ray_distances_m[0] / ray_max_distance_m, 0, 1) / 2
raw light        = local_light / 15
raw touch        = 1.0 if horizontal_collision or vertical_collision else 0.0
```

Encoding, one update per **game tick**, never per neural sub-step:

```
rectified (threat_left, threat_right, threat_front):
    drive_t = clip(raw_t - m_(t-1), 0, 1)
    m_t     = (1 - alpha) * m_(t-1) + alpha * raw_t

two-sided (touch):                       # the frozen D-059 formula, verbatim
    drive_t = clip(|raw_t - m_(t-1)|, 0, 1)
    m_t     = (1 - alpha) * m_(t-1) + alpha * raw_t

absolute (light):
    drive_t = raw_t
```

`alpha = 1/8` **exactly** — dyadic, so the update is exact in binary floating point.
Inherited verbatim from the frozen D-059 identity; a different rate is a different
(undeclared) encoder and `EemEncoderDocumentLoomV1` refuses it. At 20 game ticks/s
this is a ~0.4 s adaptation time constant (tau ~= 8 ticks).

Baseline initialization: on the first tick after construction or `reset()` the
baseline is `raw_0` itself, so `drive_0 = 0` exactly and the update yields
`m_0 = raw_0`. Per-population drives are summed and clipped to `[0, 1]`, unchanged.

The running means are the only cross-tick state: pure functions of the observation
history, updated once per game tick, cleared by `reset()`, never serialized.

## Identities

| what | value |
|---|---|
| encoder schema | `neurocraft.eem.encoder.loom.v1` |
| encoder document sha256 | `9f4b8876a2d3a50380aaf0903d16b157e33cfdb776338d53ac5097432289b126` |
| population binding sha256 | `bf45222dd478ecb5dcb35df8ed34e3dd2193449f0d8fbd5f931a24319d023bae` (unchanged) |
| configuration schema | `neurocraft.controller.full_brain_loom_config.v1` |
| configuration sha256 (uncalibrated) | `7240492264f71b1af276e5c9ec763eecdc8c73e9aebace2da6e4c2417b9e193c` |
| configuration sha256 (calibrated) | `251f8f144995e7683fb41dd86d0027b65dc7f2c51e47c9adcb40990a5d142964` |
| adapter hash | `fed84fea82ef9d6ca216524311dde4008041f79c5d5bf39e11c055087955645d` |
| decoder hash (calibrated) | `774837935998f9077f4bb6b91a1136d9125d350194d40779e26dbbbac1732e97` |
| per-step stimulation schema | `neurocraft.full_brain.encoded_stimulation.eem.loom.v1` |
| calibration schema | `neurocraft.full_readout_calibration.loom.v1` |
| calibration document sha256 | `4529bf17e48a19200d5a34bb0d1aeb19b17ef38ee99f3d2b534b9f47e91fe324` |

The encoder and uncalibrated-configuration digests are pure hashing over declared
material and are pinned in `tests/controller/test_loom_senses.py`, so they are
verified on CI without the artifact. Every other identity in the table requires the
local-only artifact.

## Calibration

Measured once by `scripts/calibrate_full_readout.py --profile loom-v1` under the
declared `full-readout-calibration-loom-v1` stream (byte layout and declared
distributions identical to the escape-v2 stream, own seed prefix), committed at
`data/calibration/full-readout-calibration-loom-v1.json`, and re-run
**byte-identically** (the second run reported `verified`, same document sha256).

| | loom-v1 | escape-v2 (for comparison) |
|---|---|---|
| `wing_scale` | 104.24708186405311 (`0x1.a0fd0307362bep+6`) | 76.83907030409712 |
| `jump_scale` | 6.821555884690886 (`0x1.b4945f22255b6p+2`) | 7.650125700320479 |
| wing p95 pool magnitude | 5.995372e-04 | 8.133883e-04 |
| jump p95 pool magnitude | 9.162133e-03 | 8.169800e-03 |
| wing saturating scale (disclosed, not applied) | 151.326950 | 121.817383 |
| jump saturating scale (disclosed, not applied) | 12.649936 | 13.894849 |

Derivation is the unchanged declared one: `scale = 0.5 / (p95 * motor_pool_gain)`.
Nothing was searched or fitted, and the measurement is the standard stream's own.
The encoder declaration was written before the run, but this lane is one commit:
there is no separate freeze commit to point at, and ADR 0019 retired
freeze-then-measure, so take that as a statement of method rather than as
something the repository can show you. `measured_under`
records the uncalibrated configuration `7240492264f7…`, `readout_scale 1.0`,
`jump_scale 1.0`, and `dynamics_hash 5da651e7…` (the C kernel every committed
calibration in this repository carries).

The wing scale is 1.3567x escape-v2's because the rectified threat channels drive the
wing pools less under the calibration stream, so the measured p95 is smaller. That
factor propagates to every applied wing amplitude — see probe condition 4.

## Probe battery

`scripts/probe_loom_encoder.py`, run against the full artifact on the calibrated
`full_mcns_loom_v1` and `full_mcns_escape_v2` controllers over identical sensor
sequences. `ESCAPE-ARC-01`'s geometry is reused verbatim (`ray_max 6.0 m`, blocked
ray `0.5 m`, so `raw threat_front = 0.458333333333`). Every "baseline" below is the
same controller's **paired all-open control run** at the same tick index, not a
remembered constant. The conditions and thresholds were written before the run (one
commit, no separate freeze commit; ADR 0019 retired freeze-then-measure), and the
numbers are reported as measured.

### 1. Standing blocked front ray — one open tick, then 0.5 m held

| | loom-v1 | escape-v2 |
|---|---|---|
| `threat_front` drive at onset (t=1) | 0.458333 | 0.458333 |
| `threat_front` drive at t=80 | 1.202e-05 | 0.458333 |
| `threat_front` drive first exactly 0.0 | t=265 | never |
| peak \|Δforward\| (tick) | 0.131160 (t=7) | rises to a plateau |
| peak \|Δvertical\| (tick) | 0.168459 (t=6) | rises to a plateau |
| \|Δforward\| at t=80 | 8.295e-04 | 0.145720 |
| \|Δvertical\| at t=80 | 6.697e-04 | 0.207050 |
| **settle tick, forward** (\|Δ\| < 1e-3 to the end) | **59** | never |
| **settle tick, vertical** | **55** | never |
| **settle tick, yaw** | **never** | never |
| settle tick, lateral | 0 | 0 |
| residual at t=600: forward / vertical / yaw | 8.198e-04 / 6.576e-04 / 2.226e-03 | 0.145720 / 0.207050 / 3.732e-03 |
| reduction (escape-v2 / loom) at t=600 | forward **177.7x**, vertical **314.9x**, yaw 1.7x | — |

Reported straight, including the part that is not what the requirement asked for:
**the drive returns to exactly 0 but the motor output does not return to baseline on
every axis.** Forward and vertical fall under 1e-3 at ticks 59 and 55 and stay there;
yaw settles to a *permanent* 2.226e-03 residual and never crosses 1e-3, even with the
drive at exactly 0.0 from tick 265 and a 600-tick hold. The whole-brain recurrent
state does not come back to the control trajectory after seeing the transient — it
settles into a nearby but different fixed point of the same dynamics under identical
(zero) input. Escape-v2's yaw residual on the same probe is 3.732e-03, larger, so this
is a property of the recurrent network and not something the loom encoder introduced.
The encoder removed the *standing escape*; it did not make the brain forgetful.

### 2. Approach ramp — proximity 0 → blocked over 20 ticks, then held 60

| | loom-v1 | escape-v2 |
|---|---|---|
| ramp ticks with drive > 0 | **20/20** | n/a (absolute) |
| peak drive (tick) | 0.170645 (t=20) | 0.458333 (flat) |
| drive after the ramp: first / last | 0.149314 / 5.657e-05 | 0.458333 / 0.458333 |
| **peak jump (tick)** | **0.022778 (t=21)** | 0.063126 (t=36) |
| **peak vertical (tick)** | **0.088478 (t=21)** | 0.201369 (t=80) |
| peak forward (tick) | 0.065920 (t=22) | 0.138243 (t=80) |

The loom drive is positive on every rising tick and peaks at the end of the approach;
loom's motor peaks land one tick later (t=21/22) and then decay. Escape-v2's peaks
land at the *end of the 60-tick hold* — it is still climbing when the probe stops,
which is the standing-escape signature again. Loom's peak jump and vertical are
2.8x and 2.3x smaller than escape-v2's: an approach transient is a smaller, briefer
drive than a held level, and the calibration did not compensate for that (it is one
p95-derived scalar over the standard stream, not a per-condition gain).

### 3. Recede — proximity blocked → 0 over 20 ticks

| | loom-v1 | escape-v2 |
|---|---|---|
| ticks with `threat_front` drive > 0 | **0/21** | 20/21 |
| max `threat_front` drive | **0** | 0.458333 |
| peak jump | 0.001552 | 0.051079 |
| peak vertical | 0.009176 | 0.161739 |

The threat drive is identically zero on every tick, as declared. Loom's nonzero peak
jump and vertical here are **not** a threat response: with the threat drive at exactly
zero they are the profile's own standing baseline — the absolute `light` channel
driving all 6,091 visual cells plus the two faithful sensory entries — and the
fresh-reset transient. The all-open control run produces output of the same order.

### 4. Touch onset, four held ticks, release

| tick | raw collision | loom-v1 touch / bristle drive | escape-v2 touch / bristle drive |
|---|---|---|---|
| 0 | false | 0 / 0 | 0 / 0 |
| 1 | true | 1.000000 / 1.000000 | 1.000000 / 1.000000 |
| 2 | true | 0.875000 / 0.875000 | 0.875000 / 0.875000 |
| 3 | true | 0.765625 / 0.765625 | 0.765625 / 0.765625 |
| 4 | true | 0.669922 / 0.669922 | 0.669922 / 0.669922 |
| 5 | true | 0.586182 / 0.586182 | 0.586182 / 0.586182 |
| 6 | false | 0.487091 / 0.487091 | 0.487091 / 0.487091 |
| 7 | false | 0.426205 / 0.426205 | 0.426205 / 0.426205 |

* touch channel value: **bitwise equal** to escape-v2 on all 8 ticks.
* bristle population drive: **bitwise equal**.
* pre-gain, pre-scale TTMn pool mean: **bitwise equal** — the neural response is
  byte-identical, as intended.
* applied motors: **not** bitwise equal. Peak vertical is 0.660095 (loom, t=4) against
  0.486547 (escape-v2, t=4), a ratio of 1.3566936904817626 — exactly
  `104.24708186405311 / 76.83907030409712`, the two profiles' independently measured
  wing scales. The difference is the calibration, not the encoder.
* the jump term is 0.0 on every tick in **both** profiles: during a collision the
  pooled TTMn activity is negative and the escape-v2 rectified mapping sends it to
  zero. Unchanged behaviour, disclosed here because it is easy to misread the table
  as "touch drives the jump".

### 5. `ESCAPE-ARC-01` reference stimulus — blocked frontal ray, fresh reset, 30-tick hold

| | loom-v1 | escape-v2 |
|---|---|---|
| `threat_front` drive, all 30 ticks | **0.000000** | 0.458333 |
| DNp01 \|mean\| at tick 0 (x paired control) | 2.000e-07 (**0.913x**) | 0.010157 (46361.5x) |
| TTMn \|mean\| at tick 0 (x control) | 1.959e-07 (**0.913x**) | 7.095e-06 (33.07x) |
| DNp01 \|mean\| last-10 (x control) | 9.380e-04 (**0.996x**) | 0.070956 (75.28x) |
| TTMn \|mean\| last-10 (x control) | 2.638e-05 (**0.899x**) | 0.001031 (35.14x) |
| vertical tick 0 / last | +0.001334 / −0.007885 | +0.001850 / +0.201369 |
| jump tick 0 / last | 0 / 0.001440 | 0.000434 / 0.063126 |

Every "x control" above is a ratio against the **paired all-open control run** of the
same controller. For TTMn that is exactly the `ESCAPE-ARC-01` quantity ("summaries are
means over the last 10 held ticks", baseline = all-open) — see below. For DNp01 it is
**not**: the receipt's "DNp01 ratio 303x" is DNp01's response against the **wing
pool's**, a different comparison, and nothing here should be read as agreeing or
disagreeing with it. The tick-0 DNp01 ratio (46361.5x) divides by a near-zero control
(2.19e-07) and is numerically unstable; it is listed for completeness, not leaned on.

**This is the finding a reviewer should read first, and it is not flattering.** Under
loom-v1 the `ESCAPE-ARC-01` reference stimulus produces **no arc response at all** —
every ratio is at or *below* its paired control. The reason is structural, not a bug:
that stimulus is presented as an *initial condition* (blocked ray from tick 0 of a
fresh reset), so the declared baseline initialization sets `m_0 := raw_0 = 0.458333`
on the very first tick and no later tick is a rise. The arc is not attenuated by this
encoder; it is never triggered by this particular stimulus presentation.

**The table's TTMn ratio is not the receipt's quantity.** The two differ in the
numerator:

| | numerator | denominator | blocked-front value |
|---|---|---|---|
| `ESCAPE-ARC-01` "TTMn event contrast" | **peak** over the 30 held ticks of the per-tick \|pool mean\| | \|last-10 mean\| of the all-open hold | **35.135851** |
| this probe's table row | **\|last-10 mean\|** of the held condition | same | **35.135820** |

They agree to seven figures only because the escape-v2 series has plateaued by the
end of the hold; on a series that had not, they would not. The receipt's definition
is `scripts/measure_escape_arc.py:343-347` and the receipt JSON's own
`"TTMn-pool |mean| at condition peak … / baseline TTMn-pool |mean| (|summary-window
mean| …)"`.

What actually establishes that this probe measures the receipt's arc, rather than a
lookalike, is stronger and mechanical: **the escape-v2 lane's TTMn and DNp01 series
and the all-open control's are bitwise identical to the receipt's `blocked_front` and
`baseline_all_open` series** — four series, 30 ticks each, max absolute difference
exactly `0.0`. `scripts/probe_loom_encoder.py` recomputes that comparison from the
committed `docs/receipts/ESCAPE-ARC-01.json` float hexes on every run and prints it
under condition 5, alongside both contrast definitions, so the claim is checked rather
than asserted. (The receipt's hold was measured under the escape-**v1** calibration,
but both series are pre-gain, pre-scale pool means and this condition has no
collisions, so the sensory drives and the neural trajectory are identical; a bitwise
match is the expectation, not a coincidence.)

The TTMn jump readout does respond under loom-v1 when the geometry arrives as a
**transition**: probe 1's onset drives `threat_front` to 0.458333 — bit-for-bit the
same value escape-v2 holds — and probe 2 reaches peak jump 0.022778 at t=21. The
arc's DNp01/TTMn contrast under a transition presentation is **unmeasured**: no probe
here computes it, and nothing above should be read as having measured it.

Two consequences follow and are stated rather than hidden:

1. **`ESCAPE-ARC-01`'s method cannot measure the arc under this identity.** Any future
   arc measurement on the loom profile must present the blocked ray as a transition
   from an open ray, not as an initial condition. `ESCAPE-ARC-01` itself is unedited
   history and remains valid for the identity it was measured under.
2. **A threat that is already present at spawn or after `/neurocraftfly reset` is
   invisible until it changes.** If the fly spawns facing a wall, that wall is its
   baseline. This is the deliberate trade for removing the standing escape, and it is
   the first thing to check in the live session.

### Why the baseline is `raw_0` and not 0

The alternative was `m_0 = 0`, which would make the very first tick a rise from
nothing and so would fire on a threat already present at reset — recovering the
`ESCAPE-ARC-01` measurement and removing consequence 2 above. It was **not** taken.
Under `m_0 = 0` a fly spawned or reset beside a wall, or with a player already
standing in front of it, fires a one-shot escape at spawn for a stimulus that is not
approaching — the same false positive this identity exists to remove, just moved to
tick 0 and to every `/neurocraftfly reset`. `m_0 = raw_0` is also the inherited
frozen D-059 rule, so keeping it leaves one initialization rule across every
identity that has a baseline at all (eem-v2, escape-v2, loom-v1) instead of two. The cost is the one recorded above: a standing
threat present at reset is invisible until it changes, which is what an approach gate
means. If the live session shows the spawn-facing-a-wall case matters more than the
spawn-beside-a-wall case, this is the decision to revisit, and it is one constant.

## What this does not establish

* **No claim that this is how LC4 computes.** Real LC4 responds to looming — angular
  expansion on the retina. This encoder computes no angular size, no expansion rate,
  and no retinotopy: it takes the first difference of a single scalar ray proximity
  against a leaky mean. It is an event gate, chosen by hand, for a demo.
* The standing caveats carry forward verbatim: **proximity is not looming** and
  **uniform per-capita drive is not retinotopy** (D-058, `ESCAPE-ARC-01`).
* **The rectification is an engineering choice, disclosed.** Real ON (approach) and
  OFF (recede) pathways are separate cell types. This identity keeps the ON half of
  the three threat channels and **drops the OFF half entirely**, rather than mapping
  it onto the same population the way the escape-v2 `touch` channel does. That is a
  simplification, not ON/OFF physiology.
* **No behaviour claim.** Nothing above is a live observation. Probes 1–5 are offline
  dynamics measurements over declared boundaries on synthetic sensor sequences. No
  live session has been run under `full_mcns_loom_v1`; whether the fly stops wedging
  itself into corners is unobserved.
* **Nothing here is a performance result.** The calibration changes the units of what
  the brain already produced. Loom's smaller peak amplitudes (probe 2) are not
  evidence that the response got worse, and the 178x/315x standing-offset reduction
  (probe 1) is not evidence that behaviour improved — it is evidence that the standing
  drive is gone.
* The motor output does not return to baseline on all axes after a transient
  (probe 1, yaw). That limitation is measured, not designed away.
* The `light` channel is untouched and still absolute, so the constant maximum light
  of an outdoor Minecraft day still pins all 6,091 visual cells at drive 1.0. Light
  events are a later stage.
* The DNp01 reads in probe 5 are offline and diagnostic-only. The live decoder never
  reads DNp01, and nothing above changes that.

## Reproducing

```bash
make sync
make spmv-build                              # REQUIRED: see the note below
UV_CACHE_DIR=.uv-cache uv run --frozen python -B scripts/calibrate_full_readout.py \
    --profile loom-v1                       # write-or-verify; must stay byte-identical
UV_CACHE_DIR=.uv-cache uv run --frozen python -B scripts/probe_loom_encoder.py \
    --json-out artifacts/analysis/loom-v1-probe.json
UV_CACHE_DIR=.uv-cache uv run --frozen pytest -q tests/controller/test_loom_senses.py
make live-demo-loom                          # local EULA + the local-only artifact
```

`make spmv-build` is not optional here. The C kernel's identity binds the compiled
library's own bytes, so `dynamics_hash` depends on it: without **both**
`packages/neural-runtime/src/neurocraft_neural/_spmv.so` and its `_spmv.build.json`
sidecar the loader falls back to the numpy kernel, `dynamics_hash` becomes
`0ebff623…` instead of the `5da651e7…` this calibration was measured under, the
measured distributions differ, and `write_or_verify` reports that the existing
document "differs" rather than "verified". Both files are gitignored build products,
so a fresh checkout or a fresh worktree has neither.

The probe's `--standing-hold-ticks` flag lengthens condition 1; the 600-tick residuals
in the table above came from `--standing-hold-ticks 600`. `--escape-arc-receipt`
points the condition-5 bitwise cross-check at a different receipt. The probe reads the
artifact and writes only to `artifacts/`, which is never tracked.
