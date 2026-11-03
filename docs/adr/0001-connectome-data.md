# ADR 0001: Use MaleCNS v1.0 as the structural target

_Dataset selection for Faithful Organism Mode, accepted 2026-08-16_

---

| Field | Value |
|---|---|
| **Status** | Accepted; G0 and the named G1 structural artifact verified 2026-08-17 |
| **Date** | 2026-08-16 |
| **Target** | Janelia `male-cns:v1.0` |
| **Fallback** | BANC v888, only through a superseding ADR |
| **Research basis** | [Connectome dataset decision](../research/connectome_dataset_decision.md) |

## 📋 Context

Faithful Organism Mode requires one frozen adult specimen with a connected central brain, optic lobes, neck connective, ventral nerve cord, documented boundary neurons, and motor outputs. Brain-only and VNC-only resources cannot alone provide the desired source-to-motor structural chain. Counts from manuscripts, portals, and runtime filters use different semantics and cannot be treated as interchangeable constants.

MaleCNS remains a CNS reconstruction rather than a whole body. It omits known peripheral/visual structures, does not provide complete receptor-resolved signs or physiology, and its primary manuscript was still a preprint at the research cutoff. Dataset choice therefore constrains topology and provenance; it does not make an executable or biologically complete fly.

## 🎯 Decision

Use the public frozen release `male-cns:v1.0` as the only primary structural target for Faithful Organism Mode.

Acquisition must use canonical static v1.0 bulk files rather than mutable `latest` aliases or live neuPrint queries. The manifest must bind exact URLs, retrieval dates, byte lengths, SHA-256 hashes, release identity, source terms, required attribution, schema, connection direction, confidence/filter policy, and computed counts.

At minimum, the source set must cover:

- body/neuron annotations
- neuron-level transmitter predictions and confidence/provenance
- the full directed connection-weight table
- release notes and license/attribution text

The canonical normalized graph preserves stable source body IDs, raw directed synapse counts, all inclusion/exclusion reasons, metadata provenance, and deterministic dense-index mappings. Sign, normalization, delay, threshold, and experimental masks are derived overlays with separate identities; they never overwrite the canonical source representation.

## ⚠️ Artifact boundary at decision time

No MaleCNS v1.0 file has been downloaded, hashed, parsed, or normalized in this repository. No derived graph exists. No exact node, pair-edge, retained-weight, exclusion, unknown-sign, or boundary-role count is established.

The proposed fixture name is `mcns-v1.0-flight-power-v1`, but no such extracted fixture exists yet. The proposed type names and bilateral pathway are extraction hypotheses until exact MaleCNS v1.0 body IDs and source rows are resolved. Implementation must fail on missing or ambiguous exact matches rather than substitute fuzzy labels.

Synthetic fixtures may be used immediately for loader schemas, equations, protocol, and trace tests. They must be labelled synthetic and cannot support the claim “derived from MaleCNS v1.0” or satisfy G1.

## ✅ Evidence update: 2026-08-17

`ACQ-MCNS-01` binds the three generation-qualified source objects and governing evidence pages to an independently pinned manifest. `FIXTURE-MCNS-01` records exact bilateral sensory, `DNg07`, `AN27X008`, and wing-motor resolution plus two byte-identical fresh-cache extractions. The present attributed derivative contains 14 source neurons and 22 source-row-traceable edges; its file SHA-256 is `3076c22fd0c87041c601b865c9fc8e913466200491f80775060e28412a087ff8`.

At the time of this first update, the evidence passed G0 only and G1 remained open. The historical no-artifact text above is retained as the decision-time evidence boundary.

The later `RUNTIME-MCNS-01` evidence bundle closes G1 for the named
`male-cns-v1.0-superclass-v1` structural artifact: 166,700 nodes, 25,582,938
directed edges, and raw retained weight 124,177,617. This is structural
provenance and runtime identity evidence, not physiology, full behavioral
validation, or a promotion of the full graph through G2.

## ⚙️ Fixture extraction contract

The first biological fixture must be deterministically extracted from verified MaleCNS files and must:

1. resolve allowed named sensory, intermediate/ascending/descending, and motor roles without fuzzy substitution;
2. identify left and right directed sensor-to-motor paths using the frozen threshold and tie-break policy from the dataset research;
3. retain original body IDs, source weights/rows, sides, types, neuropils, transmitter probabilities, and sign status;
4. preserve unknown signs instead of inventing certainty;
5. record source-file, extraction-code, extraction-config, and result hashes;
6. report exact nodes, edges, summed raw weight, exclusions, and role coverage;
7. stop with an explicit mismatch receipt if the proposed cross-dataset type hypothesis is not present in MaleCNS.

The extracted fixture remains dataset content. Redistribution requires a file-level rights decision and the applicable CC BY attribution; small size is not an exemption.

## ⚡ Consequences

### Positive

- One structural source spans the target adult male CNS boundary rather than combining specimens.
- Frozen public files can become the provenance authority without a live API in the control loop.
- Stable body IDs and raw counts can remain traceable through preprocessing and runtime views.

### Negative

- Acquisition and exact fixture extraction become hard prerequisites for structural claims.
- Missing periphery, signs, physiology, and specimen variation remain explicit limitations.
- Source-file size and full-graph performance must be measured rather than inferred.

## 🔍 Rejected alternatives

- **BANC v888 as primary:** retained as the peer-reviewed fallback and annotation cross-check, but not silently merged with MaleCNS or substituted without a new ADR.
- **FlyWire FAFB v783:** strong brain resource, but no connected VNC/motor boundary for the selected embodied target.
- **Hemibrain, MANC, FANC, or larval connectome:** valuable audits or subsystem resources, but each changes anatomical coverage, sex, life stage, or source schema.
- **Hand-authored “biological” fixture:** rejected because it cannot provide source-row lineage.

## 🧪 Promotion evidence

This ADR selected the target and did not itself pass G1. The later named G1
artifact passed only after verified source downloads, per-file rights,
canonical and derived hashes, computed count/attrition reports, direction and
duplicate tests, source-row lineage, runtime refusal on identity mismatch, and
the attributed fixture were recorded in the companion receipts.

## 🔗 References

- [Architecture overview](../ARCHITECTURE.md)
- [ADR 0005: Fixture and claim gates](./0005-fixture-and-claim-gates.md)
- [Connectome dataset decision](../research/connectome_dataset_decision.md)
- [Scientific red-team review](../research/scientific_red_team.md)
