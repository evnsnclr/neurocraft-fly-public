# RUNTIME-MCNS-01: full MaleCNS superclass runtime

_Observed deterministic build and structural-provenance receipt, 2026-08-17_

## Receipt status

| Field | Value |
|---|---|
| Receipt ID | `RUNTIME-MCNS-01` |
| Dataset | `male-cns:v1.0` |
| Named artifact | `male-cns-v1.0-superclass-v1` |
| Local path | `data/processed/male-cns-v1.0-superclass-v1` |
| Evidence state | Built twice, byte-identical, hash-verified, source-spot-audited, scoped-unit-tested, and engineering-smoked |
| G1 recommendation | Pass only for manifest `b33bdda8b1a9abfc6b23f7346197442546f7a2aec0a03fc7c07fbdfe4f67c98c` |
| Claim boundary | G2 and G3 do not follow; the unsigned smoke is not faithful and the full graph is not real-time on this machine |

The build retains annotation rows whose `superclass` is non-null and has nonzero UTF-8 length, then retains each directed weight row only when both endpoints are retained. It applies no weight threshold and no aggregation; duplicate directed pairs are refused, source self-edges are preserved, node IDs and weights remain `uint64`, and node/edge source-row lineage remains explicit.

The eight packed arrays contain `619,324,920` bytes (`590.63 MiB`, conventionally `591 MiB` rounded); the manifest brings the directory to `619,331,728` bytes. These derived arrays remain ignored under `data/processed/` and are not committed or approved for redistribution by this receipt.

## Exact identities

| Identity | SHA-256 |
|---|---|
| Source manifest | `9412105f54884a57521bbdf7fe46659caab559ef7877e6a62af95f50e1ee3768` |
| Artifact manifest file | `b33bdda8b1a9abfc6b23f7346197442546f7a2aec0a03fc7c07fbdfe4f67c98c` |
| Artifact payload | `599d1f1867ff4e9409b718674702ea5e918cdd0770bde92e97b64630b98168ce` |
| Structural graph | `21e0a82f398263702ebb213b021e2e7355745eccbdfc2d5e21c46ebb51198b25` |
| Provenance graph | `e19d549c2b3c339e82d5faa4f5c1eb96891742284b2ac025fb2ccfff34604968` |
| Builder configuration | `009725389bb72942ce63d7288c3e81b451be70400524bb97e28961c8c0899629` |
| Builder implementation composite | `3ff7163b1e85c4bbb700485da824f38647fc903fd5c97db35ac764a0b76a5764` |
| Boundary-role overlay | `94c9c009516e0cf654df32bc9f54fda17d094e90395d4a14810141042cf960d2` |

The builder composite binds `scripts/build_malecns_runtime.py` (`ab13f3b…`), `scripts/extract_malecns_fixture.py` (`2eb9fc3…`), and the strict full-artifact loader (`e1750a3…`). The machine-readable [JSON receipt](./RUNTIME-MCNS-01.json) records all component and per-array hashes.

## Counts and conservation

| Measure | Source | Retained | Excluded |
|---|---:|---:|---:|
| Annotation rows / nodes | 211,577 | 166,700 | 44,877 null superclass; 0 empty superclass |
| Directed edge rows | 151,856,684 | 25,582,938 | 126,273,746 |
| Raw weight sum | 311,833,243 | 124,177,617 | 187,655,626 |
| Self-edge rows | 123 | 101 | 22 |
| Self-edge weight sum | 542 | 474 | 68 |
| Transmitter rows / retained-node matches | 1,835,518 | 166,522 | 1,668,996 source rows excluded; 178 retained nodes have no transmitter row |

The excluded edge partition is exact: `112,633,588` pre-only, `4,802,752` post-only, and `8,837,406` neither-endpoint rows. Annotation IDs, transmitter IDs, and directed edge pairs each had zero duplicates. Raw and retained weights span `1..2591`. The retained arrays have stable body-ID order and target-major edge order; body IDs span `10001..1471062202`, source edge rows span `0..151856571`, and maximum incoming degree is `11,526`.

## Lineage and controller boundary

A SHA-256-derived audit bound to the structural identity checked 64 node positions and 256 edge positions against the re-verified Feather tables. Annotation body/superclass, transmitter body when present, edge endpoints, raw weight, and source rows all matched. Audit SHA-256: `586ae8533f3866e735f9181f6211cc56ba411ff5874cf74023ff7c65a0927c70`.

The fixture-authorized named boundary overlay retains:

| Role | Body ID | Superclass | Annotation row | Dense index |
|---|---:|---|---:|---:|
| Sensory | 77185 | `cb_sensory` | 59,342 | 58,119 |
| Sensory | 805722 | `vnc_sensory` | 173,573 | 148,743 |
| Motor | 800241 | `vnc_motor` | 169,326 | 144,496 |
| Motor | 800917 | `vnc_motor` | 169,829 | 144,999 |

## Repeatability and measured performance

Two cold builds from the same verified sources produced byte-identical manifests and arrays across all nine artifact files. Recursive byte comparison returned no differences.

| Observation | Build 1 | Build 2 |
|---|---:|---:|
| Cold build | 40.9386 s | 39.8532 s |
| Verified hash load | 0.2862 s | 0.2988 s |
| Full NumPy layout verification | 0.9501 s | 0.8366 s |
| Peak builder-process RSS | 6,050,217,984 B | 6,387,580,928 B |

The measured cold build is below the G1 target of 30 minutes and verified warm load is below the 10-second target. Peak RSS is whole-builder process RSS, not an incremental steady-state runtime measurement.

The second unsigned all-edge engineering smoke took `0.239217`, `0.223034`, and `0.225021` seconds per step (mean `0.229091 s`) and reproduced activity digest `ca37d96b23fb4e152e7cce7e7013f72c849b2d94530904a6c45aacc5227e31d1`. It used all 25,582,938 retained edges but no transmitter-derived signs. It is therefore not an ADR-0002 faithful neural run, not behavioral evidence, and not real-time full-graph execution on this machine.

## Command, environment, and checks

The observed build environment was Python `3.13.5`, NumPy `2.2.5`, PyArrow `21.0.0`, and arm64 `macOS 15.6`. From the repository root, both builds used this path-sanitized exact invocation:

```text
UV_CACHE_DIR=<temporary-uv-cache> PYTHONPATH=packages/connectome-data/src uv run --isolated --no-project --offline --python 3.13.5 --with numpy==2.2.5 --with pyarrow==21.0.0 python scripts/build_malecns_runtime.py --scratch-dir <temporary-scratch> --receipt <temporary-receipt> --smoke-steps 3
```

Only operating-system temporary paths were replaced by placeholders. Repository-relative arguments and all option values are unchanged.

Final scoped verification passed `48/48` tests in `0.89 s`; Ruff passed all owned package/script/test paths; strict mypy passed eight public source files. Fresh wheels for `neurocraft-connectome==0.1.0` and `neurocraft-neural==0.1.0` both include `py.typed`, and a strict downstream mypy check against only those installed wheels passed. Whole-repository integration remains coordinator-owned and is recorded after cross-lane integration freezes.

## Recommendation and non-claims

G1 may pass for the exact manifest hash above: node and edge lineage is complete, count semantics reconcile, source/config/builder/array identities are bound, mismatches refuse loading, and independent builds are byte-identical. This does not pass G2 or G3. In particular, the unsigned smoke is not Faithful Organism Mode, and the measured full graph is not real-time on this machine.
