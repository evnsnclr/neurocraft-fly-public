# MaleCNS v1.0 data provenance

_Source acquisition, local inspection, and transformation boundaries as observed 2026-08-17_

---

## 📋 Evidence summary

| Field | Recorded value |
|---|---|
| **Dataset target** | Janelia `male-cns:v1.0` |
| **Official release** | `v1.0`, 2026-06-08[^1] |
| **Acquisition receipt** | [`ACQ-MCNS-01`](./receipts/ACQ-MCNS-01.md) |
| **Canonical source manifest** | SHA-256 `9412105f54884a57521bbdf7fe46659caab559ef7877e6a62af95f50e1ee3768` |
| **Source license** | CC BY 4.0[^2][^3] |
| **Local storage** | `data/cache/male-cns-v1.0/`; ignored by Git |
| **Observed source state** | Three official Feather objects acquired and locally integrity-checked |
| **Derived state** | Attributed 14-node/22-edge `mcns-v1.0-flight-power-v1` CI fixture and 128-node/887-edge `mcns-v1.0-flight-steering-v1` fixture extracted ([FIXTURE-MCNS-02](./receipts/FIXTURE-MCNS-02.md)); ignored `male-cns-v1.0-superclass-v1` full artifact built twice byte-identically |
| **Gate state** | G0 passed; G1 passed only for the evidence bundle comprising full-artifact manifest `b33bdda8…f67c98c` plus companion `RUNTIME-MCNS-01` |

The acquisition receipt establishes that the named bytes were present and inspected.
[FIXTURE-MCNS-01](./receipts/FIXTURE-MCNS-01.md) separately establishes the bounded
circuit extraction. Neither of those two receipts alone establishes an executable
neural runtime or canonical full graph. G1 for the named full artifact instead relies
on its content-addressed manifest together with companion
[RUNTIME-MCNS-01](./receipts/RUNTIME-MCNS-01.md). None of these artifacts establishes
behavioral validity or biological physiology.

## 📦 Official source objects

The official download index places the flat connectome files under `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/` and links the three HTTPS objects below.[^2]

| Artifact | Exact acquisition URL | Bytes | Remote object identity | Local SHA-256 |
|---|---|---:|---|---|
| **Annotations** | `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14,483,314 | generation `1780494878811468`; MD5/ETag `50a7718770c57220f160ba4f431ab89e` | `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2` |
| **Neurotransmitters** | `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather` | 43,282,834 | generation `1780894899156750`; MD5/ETag `3d842b12fe5c49eefade528d7dd24a1f` | `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621` |
| **Connection weights** | `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1,051,241,946 | generation `1780494887545976`; MD5/ETag `f30e9dcca25cfd021bf1e7b3d975599e` | `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1` |

For each object, the locally calculated MD5 matched the observed HTTP MD5/ETag and the local byte length matched the observed object length. SHA-256 is the project's local content identity; it is not represented as an upstream SHA-256 claim.

## 🔍 Observed schemas and row counts

Schema inspection read the locally verified Feather objects. Arrow/Pandas metadata exists in the files but is not treated as a biological field.

### Body annotations

**Rows:** 211,577

```text
assignedOlHex1: float64
assignedOlHex2: float64
bodyId: int64
flywireType: string
group: float64
instance: string
somaSide: string
statusLabel: dictionary<string, int8, ordered>
superclass: string
type: string
vfbId: string
hemibrainType: string
itoleeHl: string
supertype: string
birthtime: string
mancBodyid: float64
mancGroup: float64
mancType: string
subclass: string
synonyms: string
class: string
rootSide: string
somaNeuromere: string
trumanHl: string
dimorphism: string
matchingNotes: string
entryNerve: string
mancSerial: float64
mcnsSerial: float64
serialMotif: string
fruDsx: string
exitNerve: string
receptorType: string
somaLocation: list<int64>
tosomaLocation: list<int64>
status: string
```

All 211,577 observed `bodyId` values were unique. Of those records, 166,700 had non-null `superclass`; that subset remains a retention-policy candidate rather than an accepted graph population.

### Body neurotransmitters

**Rows:** 1,835,518

```text
body: int64
cell_type: string
total_nt_predictions: int32
predicted_nt_confidence: float64
predicted_nt: string
ground_truth: string
celltype_total_nt_predictions: int32
celltype_predicted_nt: string
celltype_predicted_nt_confidence: float64
consensus_nt: string
```

All 1,835,518 observed `body` values were unique. The table covers many segment IDs outside the candidate annotated population; uniqueness does not imply that every annotation body has a transmitter row.

Within the 166,700-body non-null-`superclass` candidate, 166,522 IDs had a neurotransmitter record and 178 did not. Missing rows must remain explicit rather than receive a silently invented transmitter.

| Transmitter label | `predicted_nt` count | `consensus_nt` count |
|---|---:|---:|
| Acetylcholine | 94,947 (`acetylcholine`) | 103,720 (`ACh`) |
| Glutamate | 28,058 (`glutamate`) | 29,302 (`glutamate`) |
| GABA | 20,231 (`gaba`) | 22,069 (`GABA`) |
| Unclear | 16,151 (`unclear`) | 2,999 (`unclear`) |
| Dopamine | 4,443 (`dopamine`) | 392 (`dopamine`) |
| Histamine | 2,126 (`histamine`) | 7,891 (`histamine`) |
| Serotonin | 465 (`serotonin`) | 48 (`serotonin`) |
| Octopamine | 101 (`octopamine`) | 101 (`octopamine`) |

Each distribution totals 166,522 covered candidate IDs. These are local join/group-by results, not counts published by the source.

### Connection weights

**Rows:** 151,856,684

```text
body_pre: int64
body_post: int64
weight: int64
```

The canonical direction is `body_pre -> body_post`. A later transform must test positive integer weights, duplicates, self-edges, endpoint coverage, and aggregation semantics before producing a runtime graph.

Observed whole-table invariants were: `weight` sum 311,833,243; range 1 through 2,591; zero or negative rows 0; self-edge rows 123; and 88,384,522 unique segment IDs across the union of endpoints. The rows were not lexicographically sorted by `(body_pre, body_post)`. No globally duplicate-free claim is made until an explicit grouped duplicate audit is recorded.

## 📊 Candidate retention inspection

These values began as **local derived inspection results**, not official MaleCNS
totals. The inspected body set is defined by non-null `superclass` in the annotation
table. The edge set contains source weight rows for which both `body_pre` and
`body_post` are in that body set. Threshold counts are then evaluated within the
both-end set. The later `male-cns-v1.0-superclass-v1` artifact froze the no-threshold,
both-endpoint retention rule through its separately identified manifest and companion
receipt; this historical inspection did not itself freeze that policy.

| Quantity | Observed candidate value |
|---|---:|
| Annotation rows/body IDs with non-null `superclass` | 166,700 |
| Source connection rows | 151,856,684 |
| Source `weight` sum | 311,833,243 |
| Both-end candidate connection rows | 25,582,938 |
| Both-end candidate `weight` sum | 124,177,617 |
| Both-end candidate rows with `weight >= 5` | 6,242,118 |
| Both-end candidate rows with `weight >= 10` | 2,753,975 |

No threshold, superclass requirement, endpoint policy, duplicate policy, or self-edge
policy was accepted merely because these counts were computed. The later G1 bundle
records the versioned configuration, deterministic implementation, tests, attrition,
and derived identities for the exact named artifact only.

## 🔄 Source-to-derived contract

### Source evidence

Source evidence consists only of the exact URLs, object generations, byte lengths, remote MD5/ETags, local hashes, captured governing pages, schemas, and observed row counts recorded here and in `ACQ-MCNS-01`.

Source files remain immutable inputs. A transform may not overwrite `bodyId`, `body_pre`, `body_post`, or raw `weight`, and it may not silently replace missing or ambiguous identifiers.

### G1/runtime/release derived-evidence contract

Every future artifact promoted to G1, used as a runtime graph, or distributed as a
release requires a two-part evidence bundle rather than a claim that one manifest
contains every observed fact.

The content-addressed artifact manifest must bind:

1. the exact source-manifest identity;
2. transformation-code and configuration identities;
3. input/output schema and row counts;
4. inclusion, exclusion, aggregation, threshold, and tie-break rules;
5. stable source-ID mappings and source-row lineage;
6. raw and retained weight sums plus endpoint, duplicate, and self-edge invariants;
7. every derived array/file hash and the canonical artifact identity.

The companion build/audit receipt must bind the manifest and record receipt-only
evidence such as exact commands, observed environment and dependencies, dependency
lock where applicable, creation/evidence time, measured resource use, independent
audits, repeatability, attribution, and redistribution status. A claim must cite the
bundle and must not describe companion-only fields as literal manifest content.

The present `mcns-v1.0-flight-power-v1` milestone fixture satisfies the G0
source-lineage, deterministic-selection, and rights subset of this contract for its
bounded 14-node/22-edge circuit. Its exact types, body IDs, paths, source rows,
extraction policy, hashes, observed dependency versions, and rights disposition are
recorded in the artifact and
[FIXTURE-MCNS-01](./receipts/FIXTURE-MCNS-01.md). It is not promoted as a G1 full
runtime artifact. The tracked `mcns-v1.0-flight-steering-v1` fixture (128 neurons,
887 edges, SHA-256 `71d9a56c81914d02af8286745a23f320e37531d040b8249ff50719a26871b05b`)
carries the same embedded attribution, citation, change notice and
redistribution decision; [FIXTURE-MCNS-02](./receipts/FIXTURE-MCNS-02.md) records
its selection rule and identities.

The ignored `male-cns-v1.0-superclass-v1` artifact separately passed G1 through
manifest `b33bdda8b1a9abfc6b23f7346197442546f7a2aec0a03fc7c07fbdfe4f67c98c`
plus [RUNTIME-MCNS-01](./receipts/RUNTIME-MCNS-01.md). That bundle binds 166,700
nodes, 25,582,938 directed source rows, retained raw weight 124,177,617, per-array
identities, complete source-row lineage, mismatch refusal, audits, and two
byte-identical builds. It passes structural provenance for that artifact only; it
does not pass full-graph G2, real-time performance, G3, or redistribution.

## 🔐 Rights and redistribution boundary

The official download and release pages label the dataset CC BY and link to the CC BY 4.0 license.[^1][^2][^3] That license requires appropriate credit, a license link, and an indication of changes for adapted material.[^3]

- Keep the six current cache files ignored and out of commits.
- Treat every extracted fixture and graph as third-party-derived data, even when small.
- Include source attribution, the CC BY 4.0 link, and a precise change statement with any distributed derivative.
- Do not imply that a future project software license relicenses the source data.
- Do not publish an additional source or derived artifact until its manifest contains a file-level redistribution decision. The present CI fixture's decision is `permitted_with_attribution_and_change_notice`.
- Preserve notices and provenance independently of code packaging.

Suggested attribution:

> MaleCNS v1.0 connectome data by FlyEM (HHMI Janelia), the University of Cambridge Department of Zoology, the MRC Laboratory of Molecular Biology, and Google Research; released 2026-06-08; licensed under CC BY 4.0. Source: https://male-cns.janelia.org/download/. Changes: [describe the transformation and identify the derived artifact].

Scientific citation: Berg et al., “Sexual dimorphism in the complete Drosophila male central nervous system connectome,” Cell 189(18):5504–5526.e15 (2026), https://doi.org/10.1016/j.cell.2026.08.015.[^4] The pinned acquisition manifest records the bioRxiv v2 preprint (2025-10-30, https://doi.org/10.1101/2025.10.09.680999) that was current at acquisition.[^5]

## 🚧 Gate impact

`ACQ-MCNS-01` closes the bounded source-acquisition question;
`FIXTURE-MCNS-01` closes exact bilateral fixture extraction and file-level
redistribution. Together with the reconciled architecture/transport decisions they
pass G0. The distinct manifest-plus-`RUNTIME-MCNS-01` evidence bundle passes G1 only
for `male-cns-v1.0-superclass-v1`; it must not be generalized to another filter,
artifact, runtime, or scientific gate.

## 🔗 References

[^1]: HHMI Janelia FlyEM. (2026). “MaleCNS release notes.” https://male-cns.janelia.org/release/

[^2]: HHMI Janelia FlyEM. (2026). “MaleCNS dataset download.” https://male-cns.janelia.org/download/

[^3]: Creative Commons. “Attribution 4.0 International legal code.” https://creativecommons.org/licenses/by/4.0/legalcode.en

[^4]: Berg et al. “Sexual dimorphism in the complete Drosophila male central nervous system connectome.” Cell 189(18):5504–5526.e15, 2026. https://doi.org/10.1016/j.cell.2026.08.015

[^5]: Berg et al. “Sexual dimorphism in the complete connectome of the Drosophila male central nervous system.” bioRxiv v2, 2025-10-30. https://doi.org/10.1101/2025.10.09.680999
