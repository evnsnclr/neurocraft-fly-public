# NeuroCraft Fly data workspace

_Local data policy and MaleCNS acquisition state, observed 2026-08-17_

---

## 📋 Current evidence state

Three official `male-cns:v1.0` Feather files were acquired into the ignored local cache and verified by byte length, remote MD5/ETag, local MD5, and local SHA-256. Their schemas and row counts were inspected. The official download index identifies these files as annotations, aggregate neuron neurotransmitter predictions, and the full segment-to-segment connection graph.[^1]

The source acquisition is accompanied by one attributed, deterministic CI derivative:
`mcns-v1.0-flight-power-v1`, containing 14 source neurons and 22
source-row-traceable directed edges. Two fresh-cache builds were byte-identical. G0
is passed after the exact fixture and rights review. G1 is separately passed only for
ignored named artifact `male-cns-v1.0-superclass-v1` through its content-addressed
manifest `b33bdda8…f67c98c` plus companion
[RUNTIME-MCNS-01](../docs/receipts/RUNTIME-MCNS-01.md); its 166,700 nodes and
25,582,938 directed rows were built twice byte-identically with mismatch refusal and
source-row lineage. These ignored arrays are not approved for redistribution (the
G-numbered gate vocabulary here is historical; ADR 0019 retired the ladder). See
[ACQ-MCNS-01](../docs/receipts/ACQ-MCNS-01.md),
[FIXTURE-MCNS-01](../docs/receipts/FIXTURE-MCNS-01.md), and
[Data provenance](../docs/DATA_PROVENANCE.md).

## 📦 Local layout

| Path | Purpose | Commit policy |
|---|---|---|
| `data/cache/male-cns-v1.0/` | Six observed source and rights-evidence files | Ignored; never commit |
| `data/manifests/` | Reviewed machine-readable source and future derived manifests | Commit only reviewed metadata, never bulk data |
| `data/fixtures/` | Small reviewed synthetic or licensed biological fixtures | `mcns-v1.0-flight-power-v1.json` (14 neurons, 22 edges) and `mcns-v1.0-flight-steering-v1.json` (128 neurons, 887 edges) are approved with embedded CC BY attribution, citation, provenance, and change notice |
| `data/calibration/` | Declared boundary calibrations measured once under a named deterministic stream | Commit only the small canonical measurement document, never bulk data; `full-readout-calibration-v1.json` binds the artifact manifest and sign-overlay identities it was measured on |
| `data/raw/` | Future immutable source material | Ignored |
| `data/processed/` | Generated artifacts, including local `male-cns-v1.0-superclass-v1` arrays | Ignored; never infer redistribution approval from local presence |

The current ignored cache contains:

- `body-annotations-male-cns-v1.0-minconf-0.5.feather` — 211,577 rows
- `body-neurotransmitters-male-cns-v1.0.feather` — 1,835,518 rows
- `connectome-weights-male-cns-v1.0-minconf-0.5.feather` — 151,856,684 rows
- `male-cns-v1.0-download-index.html`, `male-cns-v1.0-release-notes.html`, and `cc-by-4.0-legalcode.en.html` — dated local evidence captures, not runtime inputs

## 🔐 License and attribution

The official site links the MaleCNS data to Creative Commons Attribution 4.0 International.[^1][^2] Source files and derivatives remain third-party data; the project's MIT license covers project-owned software and documentation and does not replace their data license.

Use this attribution template for every distributed MaleCNS-derived artifact, then replace the bracketed text with the actual transformation:

> MaleCNS v1.0 connectome data by FlyEM (HHMI Janelia), the University of Cambridge Department of Zoology, the MRC Laboratory of Molecular Biology, and Google Research; released 2026-06-08; licensed under CC BY 4.0. Source: https://male-cns.janelia.org/download/. Changes: [describe transformation and identify the derived artifact].

Scientific citation: Berg et al., “Sexual dimorphism in the complete Drosophila male central nervous system connectome,” Cell 189(18):5504–5526.e15 (2026), https://doi.org/10.1016/j.cell.2026.08.015.[^3] The pinned acquisition manifest records the bioRxiv v2 preprint (2025-10-30, https://doi.org/10.1101/2025.10.09.680999) that was current at acquisition.[^4]

Before distributing any additional fixture or derived graph, record its exact source hashes, code/configuration hashes, changes, artifact hash, attribution, license link, citation, and redistribution decision. The present fixture carries those fields in its `rights`, `input_receipt`, `extraction_receipt`, and `hashes` objects.

## 🚫 Evidence boundaries

- A present cache file is not an incorporated repository artifact.
- A schema/count inspection is not a frozen transformation.
- The non-null-`superclass` filter is accepted only as the retention policy for named
  G1 artifact `male-cns-v1.0-superclass-v1`; it is not a universal canonical graph,
  a full-runtime G2/performance result, or permission to redistribute the arrays.
- The present circuit preserves source body IDs and row lineage; a hand-authored substitute cannot be labelled MaleCNS-derived.
- Bulk source files and generated caches remain ignored even where the source license permits sharing.

## 🔗 References

[^1]: HHMI Janelia FlyEM. (2026). “MaleCNS dataset download.” https://male-cns.janelia.org/download/

[^2]: Creative Commons. “Attribution 4.0 International legal code.” https://creativecommons.org/licenses/by/4.0/legalcode.en

[^3]: Berg et al. “Sexual dimorphism in the complete Drosophila male central nervous system connectome.” Cell 189(18):5504–5526.e15, 2026. https://doi.org/10.1016/j.cell.2026.08.015

[^4]: Berg et al. “Sexual dimorphism in the complete connectome of the Drosophila male central nervous system.” bioRxiv v2, 2025-10-30. https://doi.org/10.1101/2025.10.09.680999
