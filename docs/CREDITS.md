# Credits and provenance

NeuroCraft Fly is developed by Evan Smith, with development history preserved in
the private repository. Implementation and reviews were assisted by AI coding
tools (Claude and Codex models); that assistance does not replace the source,
receipts and observed runs that support a result.

## Connectome data

MaleCNS v1.0 data are credited to **FlyEM (HHMI Janelia), the University of
Cambridge Department of Zoology, the MRC Laboratory of Molecular Biology,
and Google Research**. The pinned source release is 2026-06-08, under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The local `male-cns-v1.0-superclass-v1` derivative retains neurons with non-null
superclass annotations and connections whose two endpoints are retained,
with source IDs and row lineage preserved. It contains 166,700 nodes and
25,582,938 directed edges. Runtime signs, dynamics, sensory/readout mappings,
and Minecraft body programs are declared downstream modeling choices.

Use [DATA_PROVENANCE.md](DATA_PROVENANCE.md) for source-object identities,
attribution and change notices, and
[`data/manifests/male-cns-v1.0.sources.json`](../data/manifests/male-cns-v1.0.sources.json)
for the acquisition manifest. Bulk source tables and the full derived runtime
are acquired/built locally; these documents do not authorize redistributing them.
The small tracked MaleCNS-derived fixture carries its own attribution and scope.

Scientific citation: Berg et al., *Sexual dimorphism in the complete Drosophila male central nervous system connectome*, Cell 189(18):5504–5526.e15 (2026),
[doi:10.1016/j.cell.2026.08.015](https://doi.org/10.1016/j.cell.2026.08.015).
The pinned acquisition manifest records the bioRxiv v2 preprint that was current
when the data were acquired (2025-10-30,
[doi:10.1101/2025.10.09.680999](https://doi.org/10.1101/2025.10.09.680999)).

## Software and inspiration

Minecraft Java Edition supplies the game; Fabric supplies the mod platform.
Java, Python, uv, Gradle, NumPy, PyArrow and the other pinned dependencies retain
their respective terms. The [third-party inventory](../THIRD_PARTY.md),
lockfiles and recorded audit scopes identify incorporated material separately
from historical candidates.

EON Systems' fruit-fly demonstration inspired the inspectable loop and explicit
hand-chosen mapping described in [ADR 0019](adr/0019-pivot-to-interactive-connectome-demo.md).
NeuroCraft Fly is an independent project; this credit does not imply shared
implementation, scientific equivalence, partnership or endorsement. FlyGym,
FlyWire and other referenced work are not thereby dependencies of this Minecraft
runtime.

Project software and associated documentation use the [MIT License](../LICENSE).
Third-party data/software permissions remain separate. The project code license
does not relicense source data, Minecraft material or other assets.

NeuroCraft Fly is not an official Minecraft product and is not approved by or
associated with Mojang or Microsoft.
