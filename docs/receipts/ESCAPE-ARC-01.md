# ESCAPE-ARC-01 — escape-arc feasibility quantities under committed evidence

- **Artifact ID:** `male-cns-v1.0-superclass-v1` (manifest file SHA-256
  `b33bdda8b1a9abfc6b23f7346197442546f7a2aec0a03fc7c07fbdfe4f67c98c`, the same
  identity recorded in `RUNTIME-MCNS-01`)
- **Identity document:** `experiments/preregistration/eem-escape-v1.md`
  (frozen and committed before any measurement; content SHA-256
  `70b1206f0f685ad21b41fb677364441e678ebb711f5e2a2e986e95b523a20e9a`), phase-2
  item 1, executed exactly as pinned.
- **Evidence date:** 2026-08-30
- **Evidence state:** measured, offline dynamics measurement over declared
  boundaries only; deterministically reproduced (second run byte-identical,
  enforced by `write_or_verify`)
- **Machine scope:** measured on the local development machine that holds the
  gitignored artifact (`data/processed/male-cns-v1.0-superclass-v1`) and the
  pinned source cache (`data/cache/male-cns-v1.0`). Wall time (~7.2 s for the
  measurement phase) is an environment-scoped observation and is deliberately
  outside the canonical evidence document.
- **Scope:** re-derivation of the D-060 scratchpad feasibility numbers as
  content-addressed evidence, on the calibrated `full_mcns_escape_v1`
  controller. **The DNp01 read here is offline and diagnostic-only**: the
  live decoder never reads DNp01 (the declared readout is the TTMn motor
  stage), and nothing in this receipt weakens that claim boundary. No
  behavior, performance, or biological claim: ray proximity is not looming,
  per-capita drive is not retinotopy, the coarse sign overlay's caveats
  (`SIGN-OVERLAY-02`) apply unchanged, and the frozen identity declares **no
  pass/fail criterion** — the numbers below are reported as measured, and any
  go/no-go on a live demo is a separate engineering decision recorded after
  the fact.

## Why this receipt exists

D-060's pivot to the escape arc rested partly on a scratchpad feasibility
measurement disclosed there as *not* content-addressed evidence. The frozen
escape identity (D-061) pins, as phase-2 item 1, a re-derivation of those
quantities from the frozen tree under committed evidence — this receipt —
so the pivot's quantitative motivation stops resting on a scratchpad. The
scratchpad run is superseded by this receipt.

## Executed order

The frozen item-1 method names the **calibrated** escape controller, so
phase-2 item 2 (the boundary calibration,
`data/calibration/full-readout-calibration-escape-v1.json`, SHA-256
`6028ccd15b4c9c15f29c3b58136dd4793542608e243ecee02d2d115826ba8bec`,
wing_scale 76.915733 / jump_scale 10.186536, itself byte-idempotent across
reruns) was measured immediately before this receipt — the calibration is
itself a measurement the frozen document orders. The executed order is also
recorded inside the evidence JSON (`executed_order`). Calibration touches
only the decode stage; the activity series below are pre-decode quantities.

## Command

```
UV_CACHE_DIR=.uv-cache uv run --frozen python scripts/measure_escape_arc.py
```

The artifact manifest pin defaults to the reviewed module constant recorded
by `RUNTIME-MCNS-01`; the calibration document is loaded through the escape
loader, which cross-binds the encoder identity and the TTMn pool identity in
both directions before anything runs.

## Measured result

Evidence file: `docs/receipts/ESCAPE-ARC-01.json`, SHA-256
`3d5560bbb50ec730dcf242365d59dadcb1971279e7d4c0bdd6e0593f5a9ffd59`, produced
by `scripts/measure_escape_arc.py` as `artifacts/analysis/escape-arc-v1.json`
and copied here byte-for-byte. A second run reproduced the same bytes
(`write_or_verify` refuses anything else).

Selections (fail-closed, digest-verified join): DNp01 pair = bodies 10001
(R, dense 0) / 10010 (L, dense 6), diagnostic-only; TTMn pool = bodies
800146 (R, dense 144432) / 804642 (L, dense 147865), the declared
`jump_motor` pool (binding identity
`b5481a1385caefa57b63acf60bb25a544ec8156f38be791ca8a1b944f655c622`); LC4
population = the 126 pinned members. Conditions: 30-tick holds from fresh
reset, light 15, no collision, blocked ray 0.5 m against `ray_max` 6.0 m;
summaries are means over the last 10 held ticks (the window the frozen text
names for the response ratio, applied uniformly and disclosed in the
evidence).

| Quantity (frozen definition) | Measured | D-060 scratchpad comparator |
|---|---|---|
| DNp01 response ratio, blocked-front | **303.303** (DNp01 pair Δ 7.190e-2 / wing-pool Δ 2.371e-4) | ~300× |
| DNp01 response ratio, all-rays-0.5 | **300.532** (1.248e-1 / 4.152e-4) | ~300× |
| TTMn attenuation (all-rays) | **0.014160** (TTMn 1.754e-3 / DNp01 1.238e-1) | 1.40% (census anatomical fraction 1.49%) |
| TTMn event contrast, blocked-front | **35.136** (peak 1.031e-3 / baseline 2.936e-5) | 45–76× |
| TTMn event contrast, all-rays-0.5 | **59.734** (peak 1.754e-3 / baseline 2.936e-5) | 45–76× |
| Arc latency (direct-drive diagnostic) | **4 neural steps** (LC4 commit step 1 → first nonzero TTMn at step 5; DNp01 first nonzero at step 3), within one 5-sub-step game tick | full arc within one game tick |

Reported as measured, against a record that declares no criterion: the
response ratios and attenuation land where the scratchpad run put them
(attenuation 1.4160% vs the census's 1.49% anatomical fraction); the
blocked-front event contrast (35.1×) measures **below** the 45–76× band the
scratchpad recorded while the all-rays contrast (59.7×) sits inside it; the
arc completes in 4 neural steps — within one game tick. Baseline TTMn pool
activity is near-silent (|mean| 2.936e-5).

The direct-drive phase (150 steps, LC4 = 1.0 dense stimulation on the raw
engine, no encoder, no decode) is **diagnostic-only** and so labeled inside
the evidence; it exists to time the anatomical arc in neural steps and is
not part of any live profile.

## Verification

```sh
UV_CACHE_DIR=.uv-cache uv run --frozen python scripts/measure_escape_arc.py
shasum -a 256 docs/receipts/ESCAPE-ARC-01.json
UV_CACHE_DIR=.uv-cache uv run --frozen python scripts/calibrate_full_readout.py --profile escape
```

The first command requires the gitignored artifact and the pinned annotation
tables; it refuses to overwrite a differing evidence file. The calibration
command reports `verified` against the committed document.

## What this receipt does not establish

No live Minecraft behavior, no visible jump, no gate movement, and no
biological claim. A large offline event contrast at the TTMn readout says
nothing about what the fly's body will visibly do under the declared
jump-to-vertical-thrust body policy; that is what the phase-3 live
observation would measure, and the frozen identity commits to reporting it
whatever it shows.
