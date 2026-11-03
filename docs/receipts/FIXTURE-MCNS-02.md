# FIXTURE-MCNS-02 — flight-steering fixture, a second curated circuit

- **Fixture ID:** `mcns-v1.0-flight-steering-v1`
- **Evidence date:** 2026-08-18
- **Evidence state:** extracted, deterministically reproduced, loader-verified,
  and unit-tested. It **coexists** with `mcns-v1.0-flight-power-v1`; that fixture
  remains the authority for the full artifact's boundary role overlay and its
  identity is unchanged
- **Scope:** a curated circuit extracted from the pinned MaleCNS v1.0 source
  tables. No dynamical, behavioral, or biological claim is made here

## Why a second fixture exists

The controller stack re-verifies the graph and all four overlays on **every**
neural step, and that re-verification is a declared AB-11 anti-bypass defense
rather than removable overhead. Its cost is linear in edge count, which caps a
runnable graph at roughly 2,000 edges regardless of how fast the arithmetic is.
The 166,700-node G1 artifact is therefore permanently outside the causal path on
this architecture, and the array-native backend recorded in `FULL-ENGINE-02`
stays offline for that reason. A second, larger, curated fixture is the only
route to a richer runnable circuit, and this is it.

## Circuit and selection policy

The circuit is a flight-steering pathway: the existing mechanosensory entries,
descending neurons with predominantly wing-motor output, and the wing motor
neurons they contact.

Selection rules, frozen in the extractor and hashed into the document:

1. **Sensory entries** — bodies 77185 and 805722, each re-verified against the
   v1 fixture's own rule rather than accepted on its identifier: type beginning
   `JO-E`, or subclass `campaniform sensilla` with entry nerve in {ADMN, DMetaN}.
2. **Descending set** — the exact types `DNg02_a` through `DNg02_g`, `DNp03`, and
   `DNp31`. A declared type that is missing is a hard failure, never a silent
   substitution.
3. **Motor readout** — every `vnc_motor` body of subclass `wm` receiving at least
   one direct source edge from a declared descending body at weight ≥ 5.
4. **Path support** — for each ordered (sensory entry, descending body) pair, one
   simple directed path of 1–3 source edges over the weight ≥ 10 subgraph, chosen
   by the v1 extractor's frozen objective: maximise the minimum raw source
   weight, tie-broken by the lexicographically smallest ordered body-ID tuple.
   Pairs with no qualifying path are **recorded and skipped**, never substituted.
5. **Closure** — all source edges of weight ≥ 5 between selected bodies, with a
   declared refusal at or above a 2,000-edge ceiling.

Edge signs remain `unknown`, exactly as in v1; sign is a separate runtime overlay
decision and is not baked into fixture data.

## Measured result

| Quantity | Value |
|---|---:|
| Nodes | **128** |
| Edges (weight ≥ 5, emitted) | **887** |
| Unthresholded induced edges | 1,981 (below the declared 2,000 ceiling) |
| Soma sides | 71 right / 57 left, zero null |
| Resolved sensory→descending pairs | 62 of 66; 4 unresolved and recorded |

Composition by superclass: `vnc_motor` 52, `descending_neuron` 36 (33 declared
targets plus 3 appearing as path intermediates), `cb_intrinsic` 27,
`ascending_neuron` 7, `vnc_intrinsic` 3, `visual_projection` 1, `cb_sensory` 1,
`vnc_sensory` 1. Roles: motor 52, intermediate 41, descending 33, sensory 2.

Against the shipped `mcns-v1.0-flight-power-v1` fixture (14 nodes, 22 edges),
this is roughly 9× the nodes and 40× the edges while remaining runnable.

**Deviation from the design estimate, recorded.** A feasibility probe estimated
110–125 nodes and 778–797 edges. Taken literally — every node on any ≤3-hop
weight ≥ 10 path — the rule yields 288 nodes and 2,775 emitted edges, 39% above
the ceiling and therefore not implementable. The probe's window in fact brackets
a path threshold near 22–23 rather than the stated 10. Rather than fit a
threshold to hit a target, the extractor kept the stated thresholds and adopted
the repository's existing precedent of one maximum-bottleneck path per pair,
landing slightly above the estimate and comfortably below the ceiling.

## Identities

| Field | Value |
|---|---|
| File SHA-256 | `71d9a56c81914d02af8286745a23f320e37531d040b8249ff50719a26871b05b` |
| File bytes | 284,949 |
| Canonical graph SHA-256 | `adb6f1690a13a515d3ad88b18463540d4390a6170beb8ad0cd6400c4ccb775f8` |
| Runtime identity SHA-256 | `ebc50c0db92f7ec0e0eeb737af3d6d446cd569a488564a5682a0f255af401c26` |
| Loader implementation SHA-256 | `1909ff88521e11d9af1857c3486f748217e2d7a045dd7546e9b3326583827368` |
| Extractor code SHA-256 | `e7fc5f15202d4765bd0f9beec3e2f80d755ce3f17ea3e0d4e56c616f6d8e24c1` |
| Configuration SHA-256 | `e5feb443053ba88568d7e5b6438e4656f268ca1f3bea5fd1575d2b19558093c0` |

The loader implementation hash is **identical to the one the v1 fixture resolves
today**, which is the direct evidence that `loader.py` and `model.py` were not
touched — a one-byte edit to either would rotate every fixture identity in the
repository. The v1 fixture's runtime identity remains
`49001caafff67ab5dfc303b169155136482b7574fd42e93736bdbbfa6f66fc07`, verified
after this extraction. The new pin lives in a new module,
`packages/connectome-data/src/neurocraft_connectome/flight_pin.py`, and is passed
through `load_malecns_fixture`'s existing `expected_pin` parameter.

Sources were re-hashed before parsing and match
`data/manifests/male-cns-v1.0.sources.json`: annotations `2177e246…9a3b2`,
neurotransmitters `95c92892…879621`, weights `e35da783…56afc1`. Zero duplicate
directed pairs across all 151,856,684 source rows.

## Determinism

Three-way byte-identical: two fresh extractions into separate scratch
directories and the committed artifact all hash to `71d9a56c…71b05b`. Splitting
the extractor into three modules to satisfy the file-size rule left the
document's semantic hash unchanged, confirming that only lineage and receipt
fields moved.

## Latent hazard recorded

`model.canonical_json_bytes` encodes with `ensure_ascii=True` while the v1
extractor writes with `ensure_ascii=False`. That difference is invisible for v1,
whose content is pure ASCII, but it broke the loader's recomputed digests on a
first attempt here because the rights prose contained em-dashes. This extractor's
canonical encoder now matches the loader's exactly and the emitted fixture is
pure ASCII. **Any future fixture whose source strings carry accented characters
will hit this**, and the mismatch should be resolved before that happens.

## Rights

The document carries the same CC BY 4.0 attribution, citation, provenance and
change notice that the v1 fixture carries, with a change statement describing
this extraction. Redistribution posture is unchanged from the v1 decision.

## Verification

```sh
UV_CACHE_DIR=.uv-cache uv run --frozen python scripts/extract_flight_fixture.py --output <path>
shasum -a 256 data/fixtures/mcns-v1.0-flight-steering-v1.json
UV_CACHE_DIR=.uv-cache uv run --frozen pytest -q tests/connectome
```

Extraction requires the gitignored pinned source Feathers; the 13 fixture tests
load the committed document and run without them. A test asserts that the full
artifact's boundary role overlay still resolves to the v1 fixture.

## What this receipt does not establish

No dynamics, no behavior, no biology, no gate movement, and no claim that any
descending neuron drives any Minecraft axis. It records that a second curated
circuit was deterministically extracted from the pinned sources, loads under the
strict loader, and is small enough to run.
