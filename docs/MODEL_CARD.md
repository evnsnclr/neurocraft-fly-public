# NeuroCraft Fly model card

Updated 2026-11-03. Maintainer: Evan Smith. Software: NeuroCraft Fly
0.2.0-rc.6 consolidated candidate. Current interactive profile: `full_mcns_interactions_program_v2`.

The paragraph below records the 2026-10-24 portable-builder rerun for the earlier
Companion candidate; the consolidated candidate uses the same manifest and
calibrations and has not rerun them. The portable builder's candidate manifest is
`0485f7fe5aeede0a23e71d8191825d5b8129abc46a38c10563c94c604a840546`.
Its eight graph arrays were freshly rehashed and reproduce the preceding artifact
byte-for-byte. All seven calibration streams were actually rerun on macOS arm64
with the portable native kernel; scales and sampled-value hashes are unchanged,
with new honest source/binary/dynamics identities. Historical results below keep
their original source scope. A Windows build has its own kernel identity and
reuses these explicitly Mac-measured scales until Windows measurements exist.
Native Windows gameplay and independent installation remain unqualified.

NeuroCraft Fly is an interactive, inspectable connectome-derived model in
Minecraft. **Engineered inputs drive modeled activity, and hand-chosen readouts
select and modulate scripted body programs.** It provides a way to explore this
implementation and its assumptions. It has no demonstrated biological validity
or general-purpose control advantage.

For a runnable introduction, start with the
[research quickstart](RESEARCH_QUICKSTART.md). The
[architecture](ARCHITECTURE.md) explains software authority, and
[scientific claims](SCIENTIFIC_CLAIMS.md) defines the evidence boundaries.
The [evidence index](EVIDENCE.md) distinguishes material included in this source
package from historical support records and private raw sessions that are absent.

## Two distinct execution scopes

| Scope | Graph and inputs | Outputs and use |
| --- | --- | --- |
| Interactive local alpha | Retained MaleCNS full artifact: 166,700 nodes and 25,582,938 directed edges; engineered Minecraft observations and events | Named model means, program selections, actual applied commands and a rendered fly. Requires separately obtained bulk data and Minecraft. |
| Research starter example | Attributed 14-node, 22-edge `mcns-v1.0-flight-power-v1` fixture; prescribed synthetic pulses to named fixture entries | Headless Python model activity and fixed-input comparisons. No Minecraft, full-network interaction policy, biological recordings or trained controller. |

The small example teaches loading, stimulation and controlled comparison. It is
not a reduced implementation of all six Minecraft interactions. Its graph is
MaleCNS-derived; its pulse history and simulated activity are generated example
data. The fixture's `DNg07` and wing-motor populations must not be relabeled as
the full profile's LC4, TTMn, MN9 or antennal populations.

## Data and provenance

The source is the pinned Janelia MaleCNS `male-cns:v1.0` release from one adult
male specimen, spanning central brain, optic lobes, neck connective and ventral
nerve cord. The runtime builder retains annotation rows with a nonempty,
non-null `superclass`, then keeps directed connection rows whose two endpoints
are retained. It applies no weight threshold or aggregation. Source body IDs,
raw counts, directed rows and annotation/transmitter joins retain lineage.

The full derivative contains **166,700 nodes, 25,582,938 directed edges and a raw
retained weight sum of 124,177,617**. These are this filter's counts; edge rows
are not synapses, and these counts are not interchangeable with manuscript or
portal counts. See [acquisition](receipts/ACQ-MCNS-01.md),
[full artifact construction](receipts/RUNTIME-MCNS-01.md), and
[data provenance](DATA_PROVENANCE.md).

The distributed fixture selects 14 source neurons and 22 source edges using a
frozen bilateral path rule. It preserves the original source values and their
attribution; its stored edge signs remain `unknown`. The runtime separately
assigns signs from copied transmitter predictions. Its original artifact SHA-256
is `3076c22fd0c87041c601b865c9fc8e913466200491f80775060e28412a087ff8`.
See the [fixture receipt](receipts/FIXTURE-MCNS-01.md) and
[fixture file](../data/fixtures/mcns-v1.0-flight-power-v1.json).

Structural reconstruction does not supply a complete living nervous system,
peripheral organs, physiological state, sensory transduction or muscles. A
population's annotation name identifies source membership; it does not validate
our chosen stimulus or movement mapping. Repeated model trials are not biological
replicates of different animals.

## Dynamics

Each retained neuron has one signed, dimensionless activity-deviation value
`a_i[n]`. With raw count `C_ij` from neuron `j` to neuron `i`, modeled sign
`sigma_ij`, and incoming normalization:

```text
W_ij = sigma_ij * C_ij / max(1, sum_k C_ik)
lambda = exp(-neural_step_ms / tau_ms)
z_i[n] = g_rec * sum_j W_ij * a_j[n-1] + g_sens * s_i[n] + p_i[n] + b_i
a_i[n+1] = lambda * a_i[n] + (1 - lambda) * tanh(z_i[n])
```

`s_i[n]` is delivered sensory drive after the relevant encoder/control gate;
`p_i[n]` is separately declared direct stimulation, normally zero in the demo.
Zero initial activity and zero prehistory are the default. The displayed
formula is the unablated uniform-delay-one case. The reference engine also
supports explicit delay/intervention experiments under separate identities.

| Parameter | Current default | Interpretation |
| --- | --- | --- |
| Neural step | 10 ms simulated | Assigned integration interval |
| Leak time `tau` | 50 ms | Assigned memory scale; `lambda = exp(-0.2)` |
| Recurrent gain | 0.9 | Assigned propagation scale |
| Sensory gain | 1.0 | Scale of already encoded input |
| Baseline / noise | 0 / none | No spontaneous drive or stochastic noise |
| Activation / precision | tanh / float64 | Bounded signed model activity |
| Full-backend delay | Uniform one step | At committed `n`, recurrence reads `a[n-1]`; not measured axonal delay |

The coarse presynaptic consensus policy maps acetylcholine, dopamine,
octopamine, serotonin and tyramine to +1; GABA, glutamate and histamine to -1;
and unknown or unrecognized predictions to 0. These are contestable modeling
assumptions, not receptor-resolved connection signs. A retained edge with sign
zero contributes no recurrent drive; retained graph size does not mean every
neuron is active or every contact functionally contributes.

Implementation: [reference configuration](../packages/neural-runtime/src/neurocraft_neural/model.py),
[array engine](../packages/neural-runtime/src/neurocraft_neural/full_engine.py),
[normalized view](../packages/neural-runtime/src/neurocraft_neural/full_view.py),
and [full sign policy](../packages/neural-runtime/src/neurocraft_neural/full_signs.py).
Activity is not voltage, calcium, a spike count, firing rate, perception or thought.

## Inputs, readouts and body mapping

The current sensory encoder extends the entity-gated approach and absolute-light
components with separate event pulses. Memberships are exact annotation queries
on the verified retained table and are hashed with the source and configuration.
The main input groups are LC4 (126, split 71 left/55 right by `somaSide`), visual
class (6,091), mechanosensory bristles (2,206), labellar gustatory cells (163) and
grooming mechanosensory afferents (65). Uniform population stimulation is not
retinotopy or reconstructed receptor transduction.

The table is in arbitration priority order. A neural value alone is insufficient:
its observed source must also be eligible.

| Minecraft source | Declared input | Readout gate | Requested scripted program |
| --- | --- | --- | --- |
| Fresh player attack attempt | Amplitude-1 pulse to the 2,206 bristle cells | Absolute change in signed TTMn pair mean ≥0.002 | FLEE |
| Observed player approach | Entity-gated approach contrast to sided LC4 groups | Existing calibrated, rectified TTMn jump scalar ≥0.01138908104093503 | ESCAPE_JUMP |
| Observed mob approach | Same approach encoding; actor category supplied by Minecraft | Same TTMn scalar and threshold | BANK |
| Fresh food-near event, still near | Amplitude-1 pulse to 163 labellar gustatory cells | Positive change in MN9 pair mean ≥0.0001 | LAND; FEED when contact permits |
| Fresh brush event, outside fluid | Amplitude-1 pulse to 65 grooming afferents | Positive change in 13 antennal motor cells' mean ≥0.0003 | GROOM |
| Observed local-light change | Existing absolute-light drive to visual class | Absolute change in the mean of the two DNa01 activities ≥0.000001 | HOVER |

Generic physical contact can also stimulate bristle inputs; it is not a separate
six-source policy event. Attack and contact contributions can overlap on those
cells. Brushing and attack attempts are different native counters. Attack records
an attempted hit; the entity has no injury, health or nociception model.

Change scores freeze the preceding four completed observations' mean. Sources
are eligible for eight game ticks. Approach requires decreasing distances on
consecutive observations with the same player/mob category in the front, left
or right ray; merely seeing an actor is insufficient. Player and mob histories
that differ only in actor category can therefore have identical neural input and
activity while the source-aware policy requests different programs.

Holds and cooldowns are engineered: escape requests 24 ticks; BANK, FLEE and
light HOVER up to 40; GROOM and FEED up to 60; food landing has a 200-tick
limit; the global cooldown is 120 ticks. Java enforces actual body preconditions
and command validity, so issued durations need not equal applied durations.
Baseline CRUISE provides scripted clearance control, obstacle avoidance and
wander. Programs can query world geometry and eligible targets. FLEE can target
a visible player other than the attacker.

TTMn, MN9 and antennal readouts are source-labeled motor populations; DNa01 is a
descending-neuron pair. A separate 52-cell wing readout and DNa01-derived steering
also enter declared modulation. These selected channels and development thresholds
were chosen by hand. They are not learned natural motor decoding.

See [event inputs](adr/0031-separate-interaction-inputs.md),
[six-source policy](adr/0032-six-interaction-program-policy.md), and the
[actual selection implementation](../packages/controller-runtime/src/neurocraft_controller/interaction_program_policy.py).
The earlier [event-input ADR](adr/0026-engineered-world-events.md) describes a
previous version where brush and attack shared an input; that is historical.

## Controls and evaluation

Original wiring, shuffled weights and no input use the same declared live
encoder, readout calibration and program policy. Weight shuffling permutes raw
counts across fixed directed edge slots, preserves signs/endpoints and recomputes
incoming normalization. It does not shuffle neuron IDs or rewire connectivity.
No input zeroes delivered sensory drive after all encoder additions. With zero
initial activity/baseline/noise, that lane stays at zero neural activity; the
scripted baseline policy can still move the Minecraft body.

The fixture example uses fixed synthetic inputs and explicit seeds with the
reference engine. Its shuffle algorithm differs from the full-array shuffle;
identical seed numbers do not imply identical transformations across engines.
Keep the saved graph, sign, input, control and configuration identities with
results. One shuffle realization is an illustrative comparison, not a null
distribution. Do not assume shuffling must remove a response, and do not treat
fixed-input replay as a new closed-loop trial.

The live full profile currently accepts original, shuffled-weight and no-input
conditions. Runtime tools also contain other controls, but this does not make
live lesions or rewiring available through this profile.

## Evidence and limits

| Recorded result | What it supports and what it does not |
| --- | --- |
| Historical trained readout: raw held-out MSE 95.66× and projected MSE 403.70× the matched direct controller | Synthetic open-loop imitation on a 128-neuron, 887-edge fixture, identified by the trained artifact's graph binding and [FIXTURE-MCNS-02](receipts/FIXTURE-MCNS-02.md) ([ADR 0038](adr/0038-trained-readout-result-scope.md)). It did not benchmark the full network, closed-loop navigation or the present scripted-program policy, and it is not a general impossibility result. Values and artifact provenance remain in [ADR 0019](adr/0019-pivot-to-interactive-connectome-demo.md). |
| Historical escape probe: DNp01 response approximately 300× wing-pool response; TTMn 35.1× baseline; four neural steps | A modeled pathway result under the receipt's older encoder and configuration. The ~300× value is a cross-population selectivity ratio. TTMn's near-silent baseline inflates ratios; blocked-front 35.1× fell below the expected 45–76× band. All-rays TTMn was 59.7×. Four steps is LC4 commit 1 to TTMn first nonzero at 5. This was not a live behavior measurement. See [ESCAPE-ARC-01](receipts/ESCAPE-ARC-01.md). |
| Source `9f7b13e`, 2026-09-05 primary live recording: 18/18 original trials, three per source, completed qualified responses | Same-development-host demonstration of the six declared interactions. Six of twelve control stimuli failed qualification; among qualified controls, shuffled light retained its HOVER response. These are not complete matched biological controls. The separate fixed-60 schedule had sixty forward-ray misses and establishes no suppression. |
| Source `9f7b13e`, primary service timings: p50 28.611 ms, p99 44.140 ms, maximum 169.184 ms across 14,632 samples | Recorded compute durations at that build, including paused/neutral frames and excluding transport, application and rendering. 77 samples exceeded 50 ms. The normal 20 TPS target is not a universal real-time guarantee. |
| MIT alpha source `cab4082`, 2026-09-06 exported verification | Fresh same-host exported environment passed 2,832 Python tests, 294 JUnit tests, 126 required server GameTests and the client suite, with 35 prerequisite skips and two documented private archival exclusions. Software verification does not reproduce neural outcomes or qualify another machine. |

The live rows summarize the retained project status and the dated private
reports `showcase-current-live-2026-09-05.md` and
`showcase-current-fixed60-2026-09-05.md`, which are not in this package; their
large private session traces and recordings are not included either. The notebooks
reproduce their own small fixture experiment, not those historical results.
These rows are historical evidence summaries, not claims that a reader can
regenerate all underlying live results from the distributed files. See the
[availability index](EVIDENCE.md) before using a historical result in new work.

The installed production client has shown same-host compatibility with live
model samples. That earlier smoke is separately scoped from later source/license
rebuilds. Independent first-user setup, official-launcher account flow, broad
platform support and arbitrary multiplayer use remain unobserved. Stronger
biological conclusions would need independent organism measurements and explicit
sensitivity to signs, normalization, dynamics, sensor mapping and readout choices.

## Intended use, presentation and licensing

Use the kit to learn the architecture, inspect source labels, reproduce numerical
experiments and propose controlled changes. The current interactive profile has
no learned recurrent weights or online learning. Do not present it as recovered
fly physiology, natural behavior, intelligence, consciousness, an uploaded animal
or a validated biological control model.

The compact HUD says `Neural readouts · scripted motion`; H opens Model &
experiment details. It names the current condition and authority, distinguishes
applied and model samples, and treats missing/stale values as unavailable rather
than zero. [ADR 0033](adr/0033-compact-recording-hud.md) records these presentation
semantics. Animation and display gains provide legibility, not biological evidence.

Project software and associated documentation are [MIT](../LICENSE), following
[ADR 0035](adr/0035-project-code-mit-license.md). The original MaleCNS data and
attributed fixture are CC BY 4.0 material with their own attribution, citation and
change notices. Generated example data must retain its declared fixture lineage
and identify synthetic inputs and simulated outputs. Dependencies and Minecraft
retain separate terms; no game files, account material or bulk connectome is
bundled. See [third-party terms](../THIRD_PARTY.md).
