# FIXTURE-MCNS-01: MaleCNS flight-power fixture

_Observed deterministic extraction and redistribution receipt, 2026-08-17_

---

## 📋 Receipt status

| Field | Value |
|---|---|
| **Receipt ID** | `FIXTURE-MCNS-01` |
| **Fixture identity** | `mcns-v1.0-flight-power-v1` |
| **Dataset identity** | `male-cns:v1.0` |
| **Evidence state** | Extracted twice from separate newly created verified cache directories; outputs byte-identical |
| **Milestone artifact** | `data/fixtures/mcns-v1.0-flight-power-v1.json`; tracked in the G0 milestone |
| **Artifact bytes / SHA-256** | `36,707` / `3076c22fd0c87041c601b865c9fc8e913466200491f80775060e28412a087ff8` |
| **Internal payload SHA-256** | `69b9a929edd84fa02584de2c18d4bc1cacd1e3fe20ff64be6fe0ca8657bd9888` |
| **Semantic SHA-256** | `8360e414a3df4004fe7bfcf03bdfba8d2fb4d188c6cdd93ef30b7760055f9232` |
| **Gate effect** | Closes the exact-fixture and O-03 prerequisites for G0; does not pass G1 or prove executable dynamics |

## 🔗 Exact inputs and implementation

| Input | SHA-256 / identity |
|---|---|
| **Canonical source manifest** | `9412105f54884a57521bbdf7fe46659caab559ef7877e6a62af95f50e1ee3768` |
| **Annotations source** | generation `1780494878811468`; `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2` |
| **Neurotransmitter source** | generation `1780894899156750`; `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621` |
| **Connection-weight source** | generation `1780494887545976`; `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1` |
| **Extractor** | `scripts/extract_malecns_fixture.py`; `2eb9fc3c4cdd813131835f9c477140e81d4e2f13d6bf31e0164f353f4d7266c4` |
| **Extraction policy/configuration** | `9926d8b3e7bb0266b9b6ff02486b00b00b798e87554e6f00a64ab7724a195b11` |
| **Observed extraction environment** | Python `3.10.16`; NumPy `2.2.5`; PyArrow `21.0.0`; arm64 macOS `15.6` |

The extractor re-hashed every source before parsing. It rejected null/invalid IDs and weights, any directed-pair duplicate across all 151,856,684 weight rows, any endpoint that could not be losslessly encoded for the duplicate audit, fuzzy role substitutions, role/side ambiguity, missing bilateral roles, and incomplete directed paths.

## 🧭 Frozen extraction contract

- Sensory candidates are exact `JO-E*` types or exact `campaniform sensilla` records entering through `ADMN`/`DMetaN`.
- Named candidates are exact `DNg07` and `AN27X008` types.
- Motor candidates are exact `superclass=vnc_motor`, `subclass=wm` records.
- Each side independently maximizes the minimum raw source weight over a sensory-to-named segment and named-to-motor segment, with one to three directed edges per segment and every path edge weight at least 10.
- Ties use the lexicographically smallest complete ordered source body-ID tuple.
- The fixture closes over every source edge of weight at least 5 whose endpoints are selected path nodes.
- Required-role sides must agree across exact `rootSide`, `somaSide`, and `_L`/`_R` instance evidence. Intermediate nodes are retained even when their annotation side differs from the endpoint side; that declared policy is visible in the artifact.
- No transmitter-to-sign mapping is applied during extraction. Aggregate transmitter evidence is copied and all 22 edge signs remain explicitly `unknown`.

## 🧬 Exact resolution and selected paths

| Role candidate | Left | Right |
|---|---:|---:|
| Sensory union | 360 | 320 |
| `DNg07` | 8 | 8 |
| `AN27X008` | 1 | 1 |
| Wing motor | 33 | 34 |

Both exact `AN27X008` candidates had complete qualifying routes, with maximum complete bottlenecks 25 on the left and 22 on the right. They were not silently dropped or fuzzy-matched; the higher-bottleneck `DNg07` routes won the frozen objective.

### Left

`805722 (SNpp10, campaniform sensory, ADMN) -> 10180 -> 10069 -> 27143 (DNg07_L) -> 10236 -> 73753 -> 800241 (hg1 MN_L)`

- complete bottleneck: `26`
- upstream source-row/weight pairs: `299920/43`, `8535/210`, `674441/26`
- downstream source-row/weight pairs: `626877/28`, `105127/74`, `238705/48`

### Right

`77185 (JO-EV3_R) -> 536501 -> 29142 -> 28949 (DNg07_R) -> 10691 -> 10026 -> 800917 (hg1 MN_R)`

- complete bottleneck: `25`
- upstream source-row/weight pairs: `450501/34`, `50613/104`, `572925/29`
- downstream source-row/weight pairs: `732967/25`, `5163/249`, `92518/78`

The union contains 14 distinct source neurons and 22 induced directed edges with total raw weight 1,516. Every selected node has an aggregate transmitter row; transmitter disagreements remain visible and do not become an extraction-time sign.

## 🔁 Independent repeatability observation

Two new cache directories were created, each populated with independently re-hashed copies of the three canonical source objects. The extractor was run once against each cache using separate scratch directories. The observed coordinator wall times were approximately `15.46 s` and `15.71 s`; these are extraction observations, not neural-runtime benchmarks.

Both runs produced:

- artifact SHA-256 `3076c22fd0c87041c601b865c9fc8e913466200491f80775060e28412a087ff8`;
- payload SHA-256 `69b9a929edd84fa02584de2c18d4bc1cacd1e3fe20ff64be6fe0ca8657bd9888`;
- semantic SHA-256 `8360e414a3df4004fe7bfcf03bdfba8d2fb4d188c6cdd93ef30b7760055f9232`; and
- 14 nodes, 22 edges, the same source rows, and the same selected body-ID tuples.

Byte comparison among both fresh-cache outputs and the milestone artifact succeeded. The temporary cache copies were then removed; the ignored canonical cache and attributed fixture remain.

## 🔐 Rights and redistribution decision

The official MaleCNS release distributes the source under CC BY 4.0.[^1][^2] The 36.7 KB fixture is adapted dataset content. It is approved for commit and redistribution only with the attribution, source/license links, scientific citation, full provenance, and explicit change notice embedded in the artifact and repeated in the repository data documentation.

> MaleCNS v1.0 connectome data by FlyEM (HHMI Janelia), the University of Cambridge Department of Zoology, the MRC Laboratory of Molecular Biology, and Google Research; released 2026-06-08; licensed under CC BY 4.0. Source: https://male-cns.janelia.org/download/. Changes: deterministically selected 14 source neurons and all 22 directed source edges with weight at least 5 among the bilateral maximum-bottleneck flight-power paths; copied selected annotation and aggregate transmitter evidence; retained every edge sign as unknown; did not modify source values.

Scientific citation: Berg et al., “Sexual dimorphism in the complete connectome of the Drosophila male central nervous system,” bioRxiv v2 (2025-10-30).[^3]

The bulk Feather objects and captured evidence pages remain ignored and are not approved for commit here. A future canonical/full runtime graph requires its own transformation, attrition, rights, and runtime-identity receipt.

## 🚧 Conclusions and non-claims

This receipt establishes a deterministic, source-row-traceable, attributed MaleCNS-derived CI fixture and closes the previously unresolved cross-dataset type hypothesis. It does not establish a canonical full runtime graph, receptor-resolved signs, membrane physiology, executed neural dynamics, Minecraft control, behavioral validity, full-MaleCNS execution, or real-time performance. G1 and every stronger claim remain open.

## 🔗 References

[^1]: HHMI Janelia FlyEM. (2026). “MaleCNS dataset download.” https://male-cns.janelia.org/download/

[^2]: Creative Commons. “Attribution 4.0 International legal code.” https://creativecommons.org/licenses/by/4.0/legalcode.en

[^3]: Berg et al. “Sexual dimorphism in the complete connectome of the Drosophila male central nervous system.” bioRxiv v2, 2025-10-30. https://doi.org/10.1101/2025.10.09.680999
