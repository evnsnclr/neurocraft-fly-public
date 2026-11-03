# READOUT-CENSUS-01 — motor readout population and latency-bounded comparison

- **Evidence date:** 2026-08-18
- **Analyzed artifact:** `male-cns-v1.0-superclass-v1` (G1, manifest
  `b33bdda8…f67c98c`), cross-referenced against the pinned annotations table
  (`2177e246…9a3b2`) through `annotation_source_rows`
- **Evidence state:** measured, unit-tested, locally executed with a
  byte-identical hashed analysis body across runs; reproduces `REACHABILITY-01`'s
  headline figure exactly as a built-in guard
- **Scope:** graph topology and annotation census only. No dynamical,
  participation, behavioral, or biological claim. The word "causal" does not
  appear in the hashed body; every function reading is confined to an unhashed
  `literature_inferred_notes` sidecar

## Evidence boundary

Reachability bounds participation and never demonstrates it. Path support
measures what the retained edge support permits, not what the assigned
leaky-rate dynamics do — a path may carry vanishing, cancelling, or sign-inverted
influence, and this analysis executes no dynamics. Nothing here promotes a gate
or assigns a behavior to a neuron.

## Integrity gates

The report refuses to emit unless all five pass, and all five passed: the
recomputed annotations digest matches the manifest-declared source hash; the
`bodyId` column taken through `annotation_source_rows` is bitwise identical to
`body_ids.bin` across all 166,700 values; per-superclass counts reconcile with
the manifest across all 27 keys with zero mismatches; annotation source rows are
unique and in range against the 211,577-row source table; and retained edge
weights sum to 124,177,617, matching the manifest.

## What the current readout actually is

Motor bodies 800241 and 800917 are the **bilateral `hg1 MN` pair** — one shared
type, subclass `wm` (wing motor), soma neuromere T2, exit nerve ADMN, soma sides
L and R, sharing `mancGroup` and `mcnsSerial`. They are not arbitrary neurons.
Sensory body 805722 is `SNpp10` (subclass `campaniform sensilla`, class
`mechanosensory_proprioceptive`, entry nerve ADMN); sensory body 77185 is
`JO-EV3` (subclass `wind_gravity`, class `mechanosensory`, entry nerve AN).

## Readout-eligible population

`descending_neuron` 1,314 · `descending_neuron_tbc` 2 · `efferent_descending` 4 ·
`sensory_descending` 12 · `vnc_motor` 708 · `cb_motor` 107 · `vnc_efferent` 94.
**No superclass string is literally `descending`** — an equality filter on that
word matches zero nodes, which is a trap worth recording.

`vnc_motor` by subclass: abdominal 214, front leg 135, hind leg 130, middle leg
116, **wing motor 67**, neck 24, haltere 16, other 6. The 67 wing-motor cells
comprise 25 named types plus one untyped, including the steering muscles
(`b1`–`b3`, `i1`, `i2`, `iii1`, `iii3`, `hg1`–`hg4`, `ps1`, `ps2`, `tp1`, `tp2`,
`tpn`) and the power muscles (`DLMn`, `DVMn`), with `TTMn` and `STTMm`.

Descending neurons: 480 distinct types over 1,310 typed cells (4 untyped). 386
types have exactly two cells, of which **384 are exactly one left and one
right**. Soma sides L 656 / R 648 / M 10. VNC-target subclass is led by `xn` 441,
`ut` 276, `xl` 275.

`DNg07` has 16 cells (8 L / 8 R) in the full artifact, of which the shipped
fixture uses 2; `DNp18` has 2 cells, of which the fixture uses 1.

## Latency-bounded readout comparison

Injection sites held fixed at bodies 77185 and 805722. One hop is one neural
step; `neural_steps_per_game_tick` is 5, so L=5 is one game tick.

| Readout set | Sites | L=5 nodes | L=5 fraction | Unbounded fraction |
|---|---:|---:|---:|---:|
| shipped 2 × `hg1 MN` | 2 | 45,610 | 27.3605% | 99.1686% |
| `vnc_motor` subclass `wm` | 67 | 58,699 | 35.2124% | 99.1686% |
| `wm` + `nm` + `hm` | 107 | 60,118 | 36.0636% | 99.1686% |
| all `vnc_motor` | 708 | 64,195 | 38.5093% | 99.1890% |
| **all `descending_neuron`** | 1,314 | **152,983** | **91.7714%** | 99.1686% |

The shipped row reproduces `REACHABILITY-01` exactly (45,610 / 27.3605%,
unbounded 165,314, shortest path 2), and the script refuses to write a report if
it drifts. Unbounded path support stays at ~99.17% for every set — the degeneracy
`REACHABILITY-01` documented — so **all of the usable signal is in the
latency-bounded column**.

Reading out at the descending-neuron layer raises one-tick path support by a
factor of **3.35**, against 1.29–1.41 for widening within `vnc_motor`. Forward
hop distances explain the mechanism: 35 descending neurons are 1 hop from the
injection sites and 1,178 are 2 hops, whereas reaching a wing-motor neuron costs
an additional hop through the descending layer, which is expensive inside a
five-step budget. This is a topological statement about path length and is not
evidence that any neuron participates.

## Descending-to-motor connectivity

Direct `descending_neuron → vnc_motor` connectivity totals **17,645 edges and
229,565 weight — 8.5922%** of all incoming `vnc_motor` weight. **1,090 of 1,314
descending neurons (82.95%)** have at least one direct motor target, covering 700
of 708 motor cells and 421 of 480 descending types.

Selected per-type measurements (wing = subclass `wm`; leg = front, middle and
hind leg; laterality is (ipsilateral − contralateral)/(sum) over soma side):

| Type | Weight | Wing | Leg | Laterality |
|---|---:|---:|---:|---:|
| `DNp31` | 4,071 | 93.3% | 0.8% | −0.037 |
| `DNa08` | 2,350 | 99.9% | 0.1% | +0.074 |
| `DNg02_a` | 1,509 | 95.4% | 3.2% | +0.062 |
| **`DNa02`** | 1,501 | **5.3%** | **80.8%** | +0.852 |
| `DNg110` | 1,329 | 100.0% | 0.0% | +0.073 |
| `DNp03` | 1,284 | 94.2% | 0.3% | −0.321 |
| `DNg07` | 349 | 98.6% | 0.0% | −0.966 |
| `DNg13` | 310 | 0.0% | 99.4% | −0.974 |
| **`DNp01`** | 94 | 100.0% | 0.0% | +1.000 |

Two measurements carry corrections to characterisations made during design work:

- **`DNp01` is not "TTMn-only."** It has the sparsest motor output of any named
  descending type — 3 retained edges, 94 weight, entirely wing-motor — but across
  **two** target types: `TTMn` at 90 units (95.74%) and `DVMn 1a-c` at 4 units.
  The report records `distinct_target_types`, the dominant type, and its weight
  fraction so the distinction is auditable without recomputation.
- **`DNa02`'s wing fraction depends on the wing definition.** It is 5.3% with
  wing defined as subclass `wm` alone and 11.2% if neck and haltere motor
  neurons are included. The report records the full per-subclass weight vector so
  either convention can be recovered. Under both conventions the substantive
  point holds: the descending neuron most associated with turning in the walking
  literature sends the large majority of its motor weight to legs in this
  connectome.

## Tracked artifacts

| Generated source | Tracked evidence | SHA-256 |
|---|---|---|
| `artifacts/analysis/readout-census.json` | `docs/receipts/READOUT-CENSUS-01.json` | `ae83890ed0066cf9e10650be44325de171e1582439966f4b1b65ec994a6d814b` |

Schema `neurocraft.analysis.readout.v1`. The hashed `analysis` body
(`analysis_sha256 = aceed3df11cabde9ad8a6bcb466ce616004c986dda91b970846744404e6f3181`)
is byte-identical across runs; the unhashed `execution` envelope records elapsed
time and peak RSS, and a separate unhashed `literature_inferred_notes` block
carries every function reading so no literature inference enters a measured
figure. Implementation: `scripts/analyze_readout.py` with siblings
`readout_annotations.py`, `readout_census.py`, `readout_connectivity.py`.

Measured run: 4.35 s wall, peak RSS 1,665,089,536 B (1.55 GiB), macOS-arm64 —
environment-scoped figures, not portable claims.

## Verification

```sh
UV_CACHE_DIR=.uv-cache uv run --frozen python scripts/analyze_readout.py
UV_CACHE_DIR=.uv-cache uv run --frozen pytest -q tests/analysis
shasum -a 256 docs/receipts/READOUT-CENSUS-01.json
```

The analysis requires the gitignored G1 artifact and the pinned annotations
table; the 30 unit tests use synthetic data and skip the artifact-gated cases
when it is absent. Latency support is cross-checked against an independent
pure-Python breadth-first traversal and the connectivity table against a
per-edge Python accumulation.

## What this receipt does not establish

No dynamics, no participation, no behavior, no biology, no gate movement, and no
statement that any descending neuron drives any Minecraft axis. It measures what
the retained graph contains and what its edge support permits within a bounded
number of steps.
