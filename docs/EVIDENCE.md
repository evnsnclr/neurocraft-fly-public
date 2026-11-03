# Evidence availability

This source package includes runnable software, a reproducible small example and
selected historical records. It does not include every development artifact or
reproduce every reported historical result. Start with the
[model card](MODEL_CARD.md) for current assumptions and the
[research quickstart](RESEARCH_QUICKSTART.md) for an experiment you can run.

## Included evidence

| Material | What is included | What a reader can establish |
| --- | --- | --- |
| Research example | [Three notebooks](../notebooks/), [dataset, schema and lineage](../research/example-v1/README.md), regeneration and execution tools | Reproduce this 14-neuron, 22-edge synthetic-input experiment. It is not the full gameplay network. |
| Source acquisition and graph construction | [ACQ-MCNS-01](receipts/ACQ-MCNS-01.md), [RUNTIME-MCNS-01](receipts/RUNTIME-MCNS-01.md), its [machine-readable receipt](receipts/RUNTIME-MCNS-01.json), [source manifest](../data/manifests/male-cns-v1.0.sources.json) and builder | Inspect recorded counts, transformations and hashes; obtain the source tables separately to repeat construction. The bulk tables and arrays are absent. |
| Fixture construction | [FIXTURE-MCNS-01](receipts/FIXTURE-MCNS-01.md) and the [attributed fixture](../data/fixtures/mcns-v1.0-flight-power-v1.json) | Inspect the selected source IDs, edge values, lineage and selection rule. |
| Escape probe | [ESCAPE-ARC-01](receipts/ESCAPE-ARC-01.md) and [JSON receipt](receipts/ESCAPE-ARC-01.json) | Inspect the historical modeled pathway result and its configuration. The approximately 300× selectivity, 35.1× TTMn baseline ratio and four-step propagation are not live behavior or biological validation. |
| Trained-readout negative result | [ADR 0019](adr/0019-pivot-to-interactive-connectome-demo.md), [ADR 0038](adr/0038-trained-readout-result-scope.md) and [FIXTURE-MCNS-02](receipts/FIXTURE-MCNS-02.md) | Inspect its recorded 95.66× raw and 403.70× projected error ratios and artifact identity. The artifact's graph identity is the 128-neuron, 887-edge fixture, and the task was synthetic open-loop imitation; it is not a full-network or closed-loop navigation benchmark. The earlier training dataset and complete training run are not distributed as a reproducible benchmark. |
| Current inputs, policy and presentation | [Architecture](ARCHITECTURE.md), source/tests and [model card](MODEL_CARD.md) | Inspect the hand-chosen mappings, modeled dynamics, issued-program policy, applied-action protocol and compact HUD. Tests of these contracts are distinct from a live experiment. |

Every historical receipt retains its original source identity and scope. A newer
package version does not update those results. Machine-readable hashes establish
consistency with the named bytes, not biological validity or independent replication.

One disclosed defect concerns the exported fixture-physics pilot results
(`experiments/results/fixture-physics-pilot-v1.json` to `-v3.json`). Until
2026-11-03 the Python pilot world cast its lateral sensor rays with the body's right
in the `left` slot (risk R-27; ADR 0015 records it as the fifth, then still-open site).
The stored results ran that way: the scripted comparator steered toward the
obstructed side, and the fixture controller's left/right channels received the
opposite sides in every input-driven condition. The rays are corrected in the
exported `pilot.py`, with `tests/experiments/test_pilot_handedness.py` as the
regression; the stored results are unchanged history that is not reexecuted. No
comparison drawn from those files should be presented without this statement.

## Live demonstrations and qualification

The model card summarizes earlier same-development-host recordings and software
verification. Their large raw session traces, recordings, local installation state
and complete private development history are not in this source snapshot. Neither
the notebooks nor a successful package build independently reproduces those live
results. The primary older recording had excluded control stimuli, and its separate
60-trial schedule missed all forward-ray targets; those misses establish no neural
suppression.

Any new release qualification must name its actual source revision, conditions,
delivered stimuli and applied actions, retain failed attempts, and separately record
installation evidence on the claimed platform. See the
[experiment guide](EXPERIMENTS.md) and [release scope](RELEASE_SCOPE.md). Follow
the result actually measured; shuffling is not required to abolish a response.

## Historical references

ADRs and the dated 2026-08-18 architecture/claims snapshots retain their original
text. Some of their local links point to internal planning, research reviews or
support records intentionally omitted from this curated package. Those historical
links can therefore be unavailable. They are not hidden runnable release steps or
additional published evidence.

Historical ADRs, the model card and one preregistration also name private
working records (`STATUS.md`, `PLAN.md`, `RISKS.md`, `BLOCKERS.md`,
`DECISIONS.md`, `CHANGELOG.md`, `GOAL.md`, `AGENTS.md`, `CLAUDE.md`), private
development branches and dated showcase reports in prose. None of those is in
this package, and none is needed to run or reproduce what it ships.

The omitted references include the early bridge/optic-census support records,
dataset/model/Minecraft decision reviews, performance and red-team plans,
reference-project forensics, reuse/code reviews, the source log and the earlier
matched-trial guide. The [current architecture](ARCHITECTURE.md),
[model card](MODEL_CARD.md) and [experiment guide](EXPERIMENTS.md) are the supported
entry points. The [early resource assessment](FULL_GRAPH_RESOURCES.md) is explicitly
historical, including its obsolete goal ladder and initial unsigned timings.

The source export audits Markdown local links against the actual exported paths.
The accompanying `RELEASE_MANIFEST.json` reports every allowed historical omission
under `documentation_links.historical_omissions`. Each exception is reviewed for
an exact historical source hash and a specific omitted destination. New unresolved
links fail export; changing historical bytes invalidates their exceptions. This
audit checks file/directory targets, not external availability or heading anchors.
