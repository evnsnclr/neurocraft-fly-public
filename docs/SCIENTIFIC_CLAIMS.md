# Scientific claims and evidence boundaries

Current guidance: 2026-11-03. The objective is an interactive demo and an
inspectable software release. [ADR 0019](adr/0019-pivot-to-interactive-connectome-demo.md)
retired the paper-oriented gate ladder as a project goal. The
[earlier claims document](SCIENTIFIC_CLAIMS_2026-08-18.md), whose policy date was
2026-08-18 and which includes a later pivot notice, is preserved byte-for-byte as
history. Its then-current execution limits and primitive-only control requirements
must not be used as the present architecture.

**Describe what the implementation does, what was measured, and the scope of each
measurement.** Keep structural provenance, modeled dynamics, engineered inputs,
readout policy, body programs and presentation separate.

## A current description

> NeuroCraft Fly is an interactive Minecraft demo using simulated dynamics on a
> retained graph derived from the MaleCNS fruit-fly reconstruction. Minecraft
> events stimulate declared populations. Hand-chosen readouts select and modulate
> scripted body programs, while a HUD shows actual model activity and the current
> network condition. Players can compare original wiring, shuffled weights and
> no input. The equations, event encoding, readout mapping and Minecraft body are
> engineered assumptions.

For the headless notebooks, use a different scope:

> This example runs deterministic leaky-rate dynamics on an attributed
> MaleCNS-derived fixture containing 14 source neurons and 22 directed edges.
> Synthetic pulses drive named entries; original, shuffled-weight and no-input
> conditions expose their modeled responses under fixed inputs. These outputs
> are simulated fixture data, not live Minecraft trials or neural recordings.

Use the [model card](MODEL_CARD.md) for exact parameters, mappings and limitations,
and the [research quickstart](RESEARCH_QUICKSTART.md) to run the example.

## Five separate kinds of evidence

| Dimension | Supported questions | Conclusions it cannot supply on its own |
| --- | --- | --- |
| Structural provenance | Do IDs, directed rows, raw counts, annotations and transforms match the pinned source? | Correct physiology, complete nervous system or behavior |
| Numerical dynamics | Do the declared equations, delays, signs, interventions and same-build replay execute as specified? | Recovered membrane voltage, spikes, calcium, receptor action or natural dynamics |
| Sensor and movement engineering | Which Minecraft fields drive which populations; which source and neural gates select which programs? | Biological transduction, actor recognition, natural motor decoding or biomechanics |
| Observed behavior | Which issued commands were applied in a qualified recorded scene, and how do matched controls differ? | General competence, natural fly behavior, population generality or biological mechanism |
| Presentation | Does the fly, HUD, animation or film clearly show the recorded state? | New neural or behavioral evidence |

The current policy explicitly uses Minecraft source categories alongside neural
readouts. Body programs contain motion logic and can query world geometry or
eligible targets. Therefore neither “the network alone recognizes the player” nor
“the network generates every movement” describes this implementation. The
[architecture diagram](ARCHITECTURE.md#current-control-path) shows those paths.

## Facts that stay accessible

Three facts remain in the README and the HUD's Model & experiment details screen.
The compact overlay need not repeat historical measurements throughout play;
[ADR 0033](adr/0033-compact-recording-hud.md) makes them accessible through H.

1. **One 128-neuron fixture readout performed poorly on synthetic imitation.** Its
   raw held-out MSE was 95.66× the matched direct controller's error and its
   projected MSE was 403.70×. [ADR 0019](adr/0019-pivot-to-interactive-connectome-demo.md)
   records the artifact, [FIXTURE-MCNS-02](receipts/FIXTURE-MCNS-02.md) identifies its
   128-neuron, 887-edge graph and [ADR 0038](adr/0038-trained-readout-result-scope.md)
   fixes the scope. This was not a full-network or closed-loop navigation test. Do not
   generalize it to all connectome control, and do not imply the present
   scripted-program policy won that benchmark.
2. **The historical escape probe has a precise, limited result.** DNp01's response
   was approximately 300× the wing pool's response, a selectivity ratio rather
   than signal-to-background. TTMn reached 35.1× its near-silent baseline in the
   blocked-front condition, below that receipt's expected 45–76× band; the
   all-rays result was 59.7×. A direct-drive diagnostic propagated from LC4 commit
   step 1 to first nonzero TTMn at step 5, four neural steps. These are modeled
   responses under the older configuration, not observed live reflexes. Cite
   [ESCAPE-ARC-01](receipts/ESCAPE-ARC-01.md).
3. **Hand-chosen readouts select and modulate scripted body programs.** Event
   inputs, population mappings, thresholds, source arbitration and movement
   programs are explicit engineering. A source name such as TTMn is evidence
   about annotation membership, not proof that its selected Minecraft output is
   natural behavior.

## Current observations and their bounds

Full retained-network execution in Minecraft has been observed. The September 5
primary recording on source `9f7b13e` completed 18 qualified original-condition
responses, three for each of six sources. Six of its twelve control stimuli
failed qualification and remain excluded. Among the qualified controls, shuffled
light retained its HOVER response. A separate sixty-cell schedule missed the
forward ray in every cell. It supplies no evidence that neural responses were
suppressed. See the [model card evidence table](MODEL_CARD.md#evidence-and-limits)
for the source/date, performance and packaging scope.

Keep the following distinctions in reports and figures:

- A requested, issued, accepted and actually applied program are different states.
  Name the applied-receipt evidence before claiming an executed response.
- A failed or missed stimulus is an exclusion, not a zero neural response.
  Report exclusions, retained responses and denominators.
- A fixed input replay tests the response to that history. Live control changes
  movement and can change the later history; nominally equal scenes need not
  deliver equal inputs.
- Weight shuffling changes counts on fixed endpoints and re-normalizes incoming
  weights. It is not rewiring. One shuffle is not a null distribution.
- No-input activity can remain zero while scripted baseline CRUISE moves the
  Minecraft body. “No input makes the fly stop” is not the policy's contract.
- Service compute time, end-to-end action latency, game TPS and video FPS are
  separate measurements. Attach hardware, source/configuration, sample count and
  tail/miss information to a performance claim. A launch target is not an
  achieved rate.
- Same-build determinism does not prove bitwise cross-hardware parity. Record
  environment and source identities; declare numerical tolerances where needed.

The small research dataset is independently reproducible from the distributed
fixture. It does not reproduce the historical escape experiment, trained-readout
benchmark or six-source full-network live recording. Those records keep their
original artifact and configuration identities.

## Dataset and biological limits

The retained full graph has 166,700 nodes, 25,582,938 directed edge rows and raw
retained weight 124,177,617 under the declared superclass filter. See
[RUNTIME-MCNS-01](receipts/RUNTIME-MCNS-01.md). Do not substitute those quantities
for source-table, manuscript or portal counts, and do not call directed edge rows
individual synapses. The distributed fixture contains 14 source neurons and 22
source edges under its own selection policy;
[FIXTURE-MCNS-01](receipts/FIXTURE-MCNS-01.md) establishes its provenance.

MaleCNS is a reconstruction from one male specimen and includes the VNC. It is
not a whole body or a complete physiological record. Some visual periphery is
retained, but source coverage and column annotations are incomplete; the
engineered visual input is not a reconstructed eye. Predicted transmitters and
coarse assigned signs do not establish receptor-resolved excitation/inhibition.
Model repetition is not biological replication across animals, sexes or ages.

Natural-behavior or biological-mechanism claims would need separate organism
measurements, appropriate matched controls and sensitivity to the assigned
parameters and mappings. Success in a Minecraft scene cannot establish those
claims. This model provides no evidence of recovered memories, individual
identity, cognition, consciousness or an uploaded animal.

## Labels, rights and research changes

Name the executed scope and condition: fixture or retained full graph; original,
shuffled-weight, no-input, rewired, or another declared intervention. The live
profile's internal `faithful` condition string means original graph condition;
the composite still uses the engineered encoder and scripted-program policy.
It must not inherit the historical Faithful Organism Mode's primitive-only claims.
Operator tours are labeled scripted presentation and remain separate from READOUT
control.

HUD and plot values are **model activity** or **simulated state**. Missing or stale
samples are unavailable, not zero. Animation amplitude, turn gains and movement
programs are presentation/control choices, not anatomical measurements. Explain
these details once clearly in the walkthrough and keep the details accessible.

Project software and associated documentation are [MIT](../LICENSE). Data,
Minecraft and other third-party material retain separate terms and attribution.
Publication, permission to reuse code, successful installation and scientific
validation are separate facts.

For a new experiment, record its source/configuration, actual inputs, state/reset
policy, population membership, control construction, seeds and outcomes. Report
failures and negative results. Link to the exact evidence supporting a new claim;
never overwrite an old receipt or silently carry a measurement across a changed
model identity.
