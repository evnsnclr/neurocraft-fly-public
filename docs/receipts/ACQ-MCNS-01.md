# ACQ-MCNS-01: MaleCNS v1.0 source acquisition

_Observed local acquisition and integrity receipt, 2026-08-17_

---

## 📋 Receipt status

| Field | Value |
|---|---|
| **Receipt ID** | `ACQ-MCNS-01` |
| **Evidence state** | Official source objects acquired; local integrity and schema inspection observed |
| **Dataset identity** | `male-cns:v1.0` |
| **Release evidence** | Official release notes: v1.0, 2026-06-08[^1] |
| **License evidence** | Official download/release pages link CC BY 4.0[^2][^3] |
| **Cache location** | `data/cache/male-cns-v1.0/` |
| **Git state** | Cache ignored by `data/cache/`; no source object committed |
| **Canonical source manifest** | `data/manifests/male-cns-v1.0.sources.json`; SHA-256 `9412105f54884a57521bbdf7fe46659caab559ef7877e6a62af95f50e1ee3768` |
| **Acquisition code** | `scripts/acquire_malecns.py`; SHA-256 `0b2e5637d0683a3fb4cac8945cc09a6afc5141ee656c01a61bf3ad228b4c19a3` |
| **Independent verify-only observation** | `2026-08-17T04:40:17.731037Z`; six of six resources verified |
| **Gate effect** | Acquisition prerequisite satisfied; this receipt alone does not pass G0 or G1 |

No credential-gated API or mutable `latest` alias is the provenance anchor. The three static HTTPS objects linked by the official download index are the acquired byte sources.[^2]

The acquisition code independently pins the canonical manifest digest; a caller-selected file cannot redefine the v1.0 generation, byte, or SHA-256 identities while retaining the dataset label. The observed command was `python3 -B scripts/acquire_malecns.py --verify-only --include-metadata`. Its machine-readable receipts bound each generation-qualified URL, expected and observed size/hash, manifest digest, acquirer digest, and observation time.

## 📦 Acquired object receipts

### Annotations

- **URL:** `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather`
- **Object generation:** `1780494878811468`
- **Observed/local bytes:** `14,483,314`
- **Observed HTTP MD5/ETag:** `50a7718770c57220f160ba4f431ab89e`
- **Local MD5:** `50a7718770c57220f160ba4f431ab89e` — match
- **Local SHA-256:** `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2`
- **Observed rows:** `211,577`

### Neurotransmitters

- **URL:** `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather`
- **Object generation:** `1780894899156750`
- **Observed/local bytes:** `43,282,834`
- **Observed HTTP MD5/ETag:** `3d842b12fe5c49eefade528d7dd24a1f`
- **Local MD5:** `3d842b12fe5c49eefade528d7dd24a1f` — match
- **Local SHA-256:** `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621`
- **Observed rows:** `1,835,518`

### Connection weights

- **URL:** `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather`
- **Object generation:** `1780494887545976`
- **Observed/local bytes:** `1,051,241,946`
- **Observed HTTP MD5/ETag:** `f30e9dcca25cfd021bf1e7b3d975599e`
- **Local MD5:** `f30e9dcca25cfd021bf1e7b3d975599e` — match
- **Local SHA-256:** `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1`
- **Observed rows:** `151,856,684`

The matching MD5 values bind the local files to the observed remote MD5/ETags. The local SHA-256 values are the stronger project-side content identities used for future manifests.

## 🔗 Governing-page captures

These ignored HTML captures preserve what was observed on the acquisition date. They are supporting evidence, not immutable upstream authorities and not neural-runtime inputs.

| Capture | Exact URL | Bytes | Local SHA-256 | Observed fact |
|---|---|---:|---|---|
| **Download index** | `https://male-cns.janelia.org/download/` | 44,208 | `51144922891ccdc5cda2490d55b39ef69222134ed6944bbff2766379e403ad3c` | Exact object links and CC BY link |
| **Release notes** | `https://male-cns.janelia.org/release/` | 21,431 | `b2a4d40f0ae376f8c887cfbdbbc3dd10a6a165380b658fce8a20fd153a0a16aa` | `v1.0 (June 8, 2026)` |
| **License legal code** | `https://creativecommons.org/licenses/by/4.0/legalcode.en` | 48,970 | `6d55b998ed5c54f43426d059a8c549ed58a3321e5463e6a6af1c6b56ab78c333` | CC BY 4.0 terms |

## 🔍 Local inspection results

The locally verified Feather objects were opened for schema and row-count inspection. Full field schemas are recorded in [Data provenance](../DATA_PROVENANCE.md).

| File | Rows | Key observed fields |
|---|---:|---|
| **Annotations** | 211,577 | `bodyId`, hierarchy/type fields, sides, nerves, receptor metadata, locations, status |
| **Neurotransmitters** | 1,835,518 | `body`, prediction counts/confidences, predicted/ground-truth/cell-type/consensus transmitter fields |
| **Connection weights** | 151,856,684 | `body_pre`, `body_post`, `weight` |

Local byte length, MD5, and SHA-256 were recalculated after acquisition. `git check-ignore -v` identified the cache through the repository's `data/cache/` rule. No bulk file was staged or committed.

## 📊 Unfrozen retention candidate

One read-only inspection formed a candidate body set from annotation records with non-null `superclass`, then kept connection rows whose `body_pre` and `body_post` were both in that set.

| Result | Value |
|---|---:|
| Candidate bodies | 166,700 |
| Source connection rows | 151,856,684 |
| Source `weight` sum | 311,833,243 |
| Both-end candidate rows | 25,582,938 |
| Both-end candidate `weight` sum | 124,177,617 |
| Candidate rows with `weight >= 5` | 6,242,118 |
| Candidate rows with `weight >= 10` | 2,753,975 |

These are derived inspection results, not official source counts. The `superclass` filter and weight thresholds are **not frozen**, and no result table or graph artifact was written.

## 🔐 Attribution and redistribution

The source is governed by CC BY 4.0.[^2][^3] A distributed derivative must credit the MaleCNS collaborators, link the license and source, and state the changes. This repository continues to keep bulk source data in ignored local storage. Any future fixture or graph needs a separate file-level redistribution decision and manifest; a project software license cannot silently replace the data license.

The canonical scientific citation is Berg et al., “Sexual dimorphism in the complete connectome of the Drosophila male central nervous system,” bioRxiv v2 (2025-10-30), https://doi.org/10.1101/2025.10.09.680999.[^4] The manuscript was a preprint at this receipt date; data rights derive from the official release terms rather than the article alone.

## 🚧 Conclusions and non-claims

This receipt supports only these observed statements:

- the three named official objects were acquired on 2026-08-17;
- their local bytes, MD5 values, and SHA-256 values were recorded;
- local MD5 values matched the observed HTTP MD5/ETags;
- their schemas and row counts were inspected; and
- a retention-policy candidate was counted without writing a derived artifact.

This acquisition receipt does **not by itself** establish fixture extraction, a canonical/runtime graph, an accepted full-graph retention policy, G1, or use by a controller. Fixture extraction and overall G0 promotion are recorded separately in [FIXTURE-MCNS-01](./FIXTURE-MCNS-01.md) and the current governance files.

## 🔗 References

[^1]: HHMI Janelia FlyEM. (2026). “MaleCNS release notes.” https://male-cns.janelia.org/release/

[^2]: HHMI Janelia FlyEM. (2026). “MaleCNS dataset download.” https://male-cns.janelia.org/download/

[^3]: Creative Commons. “Attribution 4.0 International legal code.” https://creativecommons.org/licenses/by/4.0/legalcode.en

[^4]: Berg et al. “Sexual dimorphism in the complete connectome of the Drosophila male central nervous system.” bioRxiv v2, 2025-10-30. https://doi.org/10.1101/2025.10.09.680999
