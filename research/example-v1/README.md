# Synthetic pulses through a MaleCNS fixture

This is a generated, headless teaching dataset: 14 source-derived neurons and
22 directed edges from the pinned MaleCNS v1.0 flight-power fixture. It is not a
whole-brain experiment, an escape-circuit experiment, a learned policy, a
Minecraft session, or a biological measurement. Signed activity is dimensionless
model activity, not firing rate. Negative values are permitted.

From the repository root, after the documented environment setup:

```sh
uv run --locked python scripts/research_example.py verify
uv run --locked python scripts/research_example.py generate --output artifacts/my-example
uv run --locked python scripts/research_example.py verify --dataset artifacts/my-example
```

Generation refuses an existing destination. The checked-in example is preserved;
use a new directory for another run. `verify` checks the complete file inventory,
all payload hashes, code hashes, pinned source identity, schemas, conditions and
configuration, then replays the experiment. Default replay requires identical
bytes. `--tolerant` permits only finite numerical differences satisfying
abs(stored - replay) <= max(1e-12, 1e-10 * max(abs(stored), abs(replay)))
(Python math.isclose semantics) in declared CSV floating columns and floating
summary values. Identifiers, configuration, source and code
hashes remain exact. The stored payload must still match its own manifest hashes.
This accommodates possible Python/libm differences; it does not certify a platform.
Checksums detect corruption and replay checks consistency; neither is a signature
or proof that an arbitrary external producer is trustworthy.

## Experiment

Each condition begins with zero current activity, zero delayed history, no lesion,
no noise and zero baseline. The recurrence is the existing `NeuralEngine` reference
backend (10 ms step, 50 ms leak time, recurrent gain 0.9, tanh, float64). An edge of
numerical delay d reads a[n-d] when computing a[n+1]; default d=1 therefore does
not mean an instantaneous contribution from a[n]. All steps commit synchronously.

The engineered direct drive p[n] is +0.8 on source body 805722 (SNpp10, L) at
step_before 10 through 29, then +0.8 on body 77185 (JO-EV3, R) at step_before 50
through 69. Other neurons receive zero direct drive, and no sensory adapter is
used. These pulses do not implement measured touch or wind stimuli. Both target
rows are present in inputs.csv for every interval, including zero intervals.

- `original`: source endpoints and raw counts with declared modeled signs.
- `shuffled_weights`: same endpoints, sign and delay overlays, but raw-count
  weights permuted across edges by `WeightOverlay.shuffled`, seed 7. Incoming L1
  normalization is recomputed from the assigned counts. This is not rewiring and
  does not preserve per-target weight totals or normalized weight distributions.
- `no_input`: original weights with the identical requested schedule recorded,
  but every applied drive zero. This is an input-ablation control.

The coarse presynaptic consensus sign policy assigns +1 for acetylcholine,
dopamine, octopamine, serotonin and tyramine; -1 for GABA, glutamate and histamine;
0 for other/unknown labels. These are engineered assumptions, not receptor-level
measurements. Every canonical source-edge sign remains `unknown`.

Readouts are explicitly chosen singleton inspection pools: sensory_L=805722,
sensory_R=77185, DNg07_L=27143, DNg07_R=28949, hg1_motor_L=800241,
hg1_motor_R=800917. Each readout is the unscaled signed arithmetic mean of its
members. These are not the Minecraft program-selection pools and emit no action.
The pool names and source IDs are in manifest.json; no behavior is inferred.

## Files and schema

- nodes.csv: one row per source neuron, ascending body ID; roles separated by `;`.
  Blank cells mean missing source metadata, not an inferred annotation.
- edges.csv: one row per condition and source edge; target-major edge order.
  `raw_count` and `source_row` always describe the unchanged source edge.
  `overlay_source_edge_index` identifies the donor edge for `overlay_count`.
  `effective_weight` is assigned_sign * overlay_count / incoming overlay-count sum.
- inputs.csv: one row per condition, interval and target; `step_before` names n
  before p[n] is applied, and time_ms = n * 10.
- activity.csv: every neuron at committed steps 0..100, including the zero reset;
  time_ms = step * 10. Values are signed dimensionless activity.
- readouts.csv: every named pool at those same committed steps.
- summary.json: measured peak absolute activity, first peak step and signed value
  at that peak; a rectangle sum of absolute samples at steps 1..100 times 10 ms;
  and original/shuffled differences. Peaks are within this finite observation
  window; a final-sample peak does not establish settling or a global maximum.
  No success criterion or cherry-picked trial.
- manifest.json: schemas, dynamics, exact source/overlay identities, source rights,
  pool and input definitions, code hashes, payload hashes and verification policy.
  It has no timestamp, machine path, Git HEAD dependency or self-referential hash.

Source rows refer to the original source objects recorded in the fixture manifest,
not row numbers in these CSVs. Source and overlay identities are separate.
Compare the measured controls even if shuffling retains the response. This small,
selected graph cannot establish a full-network advantage or biological necessity.

## Attribution and rights

Copied neuron/edge structure and annotations retain the upstream CC BY 4.0 terms;
retain attribution and this change notice. Project-generated simulated responses,
input schedules, summaries, code and associated documentation are released under
the project MIT license. This does not relicense underlying third-party data.
The mixed edges.csv and manifest.json preserve the separate source-data terms.

MaleCNS v1.0 connectome data by FlyEM (HHMI Janelia), the University of Cambridge
Department of Zoology, the MRC Laboratory of Molecular Biology, and Google Research.
Berg et al., "Sexual dimorphism in the complete connectome of the Drosophila male
central nervous system", bioRxiv v2 (2025-10-30),
https://doi.org/10.1101/2025.10.09.680999, the reference current at acquisition;
published as "Sexual dimorphism in the complete Drosophila male central nervous
system connectome", Cell 189(18):5504-5526.e15 (2026),
https://doi.org/10.1016/j.cell.2026.08.015.
Source: https://male-cns.janelia.org/download/
License: https://creativecommons.org/licenses/by/4.0/legalcode.en

Changes: transcribed the already selected 14 neurons and 22 edges into CSV without
changing source values; added declared sign/count/delay overlays, synthetic input
histories, simulated activity and engineered inspection summaries. The upstream
fixture selection and bulk-source hashes are preserved in manifest.json. No bulk
source objects, Minecraft binaries or account material are included.
