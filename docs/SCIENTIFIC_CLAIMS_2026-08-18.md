# Scientific claims and language policy

> **Status under ADR 0019 (2026-09-02).** The project is now an interactive demo, not a
> paper. The *claim ladder* (G0–G6) and the *approved wording by evidence gate* sections
> below are retired as policy and kept as history. What remains in force is the one rule
> in `GOAL.md` — never claim the connectome does something it doesn't — and, as its
> detail, the sections **Five independent fidelity dimensions**, **Statements prohibited
> by present evidence**, and **Why this is not consciousness or brain uploading**. The
> three facts every public surface carries are stated in `GOAL.md`.

- **Policy date:** 2026-08-18
- **Current maximum promoted claim:** **G3 — causal closed loop, MaleCNS-derived fixture circuit scope**
- **Current evidence:** G1 is passed for named full artifact `male-cns-v1.0-superclass-v1` through its manifest plus `RUNTIME-MCNS-01` companion receipt. Fixture-scope G2 is passed for the deterministic source-derived fixture controller through exact replay and named actual-fixture zero-recurrence (sign-ablation) and matched-ray mutations. Fixture-scope G3 is passed (D-027) through `EXTERNAL-BRIDGE-03` — exact replay of nine Minecraft-derived records with eight pre-actuation and seven next-tick world cross-links and a 9/9 zero-recurrence state/action divergence to neutral — together with the complete `AB11-MUTANTS-01` test-power suite, whose independent adversarial review verified no skipped/zero-count path. `EXTERNAL-BRIDGE-01` and `EXTERNAL-BRIDGE-02` are historical. The promoted claim inherits the receipts' named bounds; G3-R, G4, G5, and G6 remain open

This policy governs README text, UI labels, logs, demonstrations, talks, manuscripts, repository descriptions, and release notes. A stronger statement is prohibited until its cumulative gate passes with inspectable evidence.

## Claim ladder

| Gate | Exact scope that becomes supportable | Minimum evidence |
|---|---|---|
| **G0 — research resolved** | “NeuroCraft Fly targets the named pinned dataset, fixture, model, Minecraft tuple, protocol, controls, rights posture, and compute envelope.” | One authoritative design; exact MaleCNS fixture extraction or explicit fallback receipt; resolved required data rights |
| **G1 — structural provenance** | “This named runtime graph artifact is derived from MaleCNS v1.0.” | Evidence bundle of content-addressed artifact manifest plus companion build/audit receipt; hashed sources; source-row lineage; direction/count/filter/attrition tests; startup refusal on mismatch |
| **G2 — deterministic neural engine** | “The declared deterministic dynamics execute on this named source-derived fixture.” | Analytic/dense-oracle tests; same-build full fixture trace replay; named actual-fixture zero-recurrence (sign-ablation) and matched-input perturbation mutations; sign/delay/lesion/stimulation observability |
| **G3 — causal closed loop** | “Minecraft observations causally affect model state, whose named motor readout exclusively controls one custom fly.” | Raw-to-world authoritative trace; all anti-bypass tests; zero/frozen propagation changes the action path; primitive actuator only |
| **G4 — connectome-dependent result** | “The selected source topology/weights contribute to this preregistered Minecraft result under these adapters.” | Held-out paired trials; fair hierarchical nulls; uncertainty/effect sizes; all trials disclosed |
| **G5 — limited biological correspondence** | “This named model result is consistent with this named Drosophila response under stated assumptions.” | Independent compatible fly target; a priori prediction; matched lesions/controls; assumption sensitivity; preferably independent dataset/specimen |
| **G6 — reproducible artifact/demo** | “The declared artifact and demonstration were reproduced under these named conditions.” | Clean acquisition/build/run; complete tests/benchmarks/replays; SBOM/rights/notices; continuous launcher trial; critical/high findings closed |

Claims are cumulative. A circuit can pass a gate without the full graph, but every statement must say **MaleCNS-derived circuit**. “Full MaleCNS” requires the exact full artifact and run receipt. Fixture-scope G2 does not imply that every AB-01 through AB-12 mutation, any Minecraft-specific AB-11 case, full-graph latency, or G3 passed.

“Deterministic” is environment-scoped unless a receipt says otherwise. Current exact
fixture/controller replay and pilot-v2 result hashes are supported for the named
build and execution environment; removing platform metadata from a result envelope
does not prove bitwise cross-hardware parity. That stronger claim requires declared
tolerances or a separately identified portable numerical backend.

### Real-time qualifier

“Real time on `<named hardware>`” is a qualifier for G3 or higher, not a separate biological claim. The following are **acceptance targets, not measurements**:

- 10,000-interval soak;
- each 50 ms full-model neural interval below 35 ms p99 and 45 ms p99.9;
- observation-to-applied-action below 40 ms p99;
- Fabric callback below 1 ms p99 and 2 ms p99.9;
- no growing backlog, unmarked deadline miss, dropped edge, or skipped neural step.

The fixture controller has a separate benchmark, while the unsigned full-graph engineering smoke averaged about `0.229` seconds per neural step. Full-graph latency remains an open full-runtime subgate. Neither result supports G3-R.

## Five independent fidelity dimensions

Success in one column never implies another.

| Dimension | What can be tested | What it does not establish |
|---|---|---|
| **Structural fidelity** | Runtime IDs/edges/directions/raw counts trace to the pinned source and declared filters. | Correct physiology, complete nervous system, or behavior |
| **Dynamical assumptions** | Deterministic equations, parameters, convergence, sensitivity, and replay are explicit. | Recovered membrane voltage, firing rate, receptor action, or natural dynamics |
| **Sensor/motor assumptions** | Minecraft fields stimulate named boundary populations; named motor pools produce primitive actions. | Biological transduction, muscles, biomechanics, vision, or proprioception |
| **Behavioral evidence** | Preregistered Minecraft effects and, separately, agreement with a named fly response. | General fly intelligence, natural behavior, individual identity, or consciousness |
| **Artistic presentation** | Custom model, texture, animation, particles, HUD, sound, video. | Any scientific fidelity or causality |

The HUD must label internal values **model activity** or **simulated state**, never neural recording, voltage, calcium, spike rate, or thought unless a separately validated model supplies that unit.

Protocol suffixes `_m` and `_m_s` name engineering coordinates only. The bridge
treats one Minecraft block as one engineering metre and converts velocity from
blocks per tick by multiplying by the 20 Hz game-tick rate. This supports a
consistent software interface; it is not physical calibration and must not be
described as measured SI motion, fly biomechanics, or biological scale.

## Dataset boundary

The selected target is Janelia `male-cns:v1.0`, a frozen CC BY release from one adult **male** specimen. Research sources describe a connected central nervous system spanning central brain, optic lobes, neck connective, and ventral nerve cord, with sensory afferents entering nerves and motor outputs. Three generation-qualified source objects are present only in the ignored cache; the repository contains the attributed bounded fixture, not the bulk dataset.

It is not:

- a whole body, muscles, peripheral sensory organs, or neuromuscular system;
- a complete retina/visual periphery. Corrected 2026-08-18 by measurement ([`OPTIC-CENSUS-01`](./receipts/OPTIC-CENSUS-01.md)); the earlier wording here claimed the lamina and ocellar ganglion were absent, which was a BANC property mistakenly transferred to MaleCNS. The retained artifact **contains** a partial visual periphery: 6,098 `ol_sensory` nodes including 3,377 `R1-R6`, 1,300 R7 and 1,329 R8 variants, plus lamina monopolars `L1`-`L5` (1,770-1,787 each), 86 `Lai`, and 24 `Lat1`-`Lat5`. The incompleteness is measured, not assumed: only 45.6%/45.8%/44.1%/36.9%/2.2% of `L1`/`L2`/`L3`/`L4`/`L5` receive any retained `R1-R6` input, and optic-lobe column identity (`assignedOlHex1`/`assignedOlHex2`) is populated for 14.23% of retained nodes, absent for every `R1-R6`, every `L4`, and every left-side `L3`, `C2`, and `Tm4`. Column coverage is bilaterally asymmetric: 797 of 892 right-eye columns are complete across all 15 columnar types against 0 of 879 left-eye columns. Ocellar **photoreceptors** are absent; ocellar-ganglion projection and centrifugal neurons (46 retained `OCG*`/`OCC*`) are present. Any pixel-to-photoreceptor or retinotopic assignment is therefore **inferred**, never source-observed;
- a complete physiological/signalling record: receptor identity, gap junctions, neuropeptides, modulation, internal state, and many signs/kinetics are absent or incomplete;
- a population sample: Minecraft trials repeat one structural specimen and are not biological replicates.

The manuscript reports 166,691 neurons and 11,691 types; the official v1.0 Cell Type Explorer snapshot reports 164,838 catalogued neurons, 11,751 types, and 171,901,440 synapses. These are different filtered count semantics. The source weight table has 151,856,684 directed rows; that is not a neuron or synapse count. The named G1 artifact `male-cns-v1.0-superclass-v1` has its own declared retention semantics: 166,700 nodes, 25,582,938 directed rows, and raw retained weight 124,177,617. Those retained counts must not be substituted for manuscript, portal, or synapse counts.

The CI circuit `mcns-v1.0-flight-power-v1` is a valid attributed MaleCNS-derived structural fixture: 14 source neurons and 22 directed source edges selected by the frozen bilateral path policy. Exact `DNg07`, `AN27X008`, boundary, and motor classes resolved without fuzzy substitution. Its embedded CC BY attribution/citation/change notice permits redistribution. Deterministic assigned dynamics executed and replayed exactly on this fixture; named actual-fixture tests remove recurrent sign influence and perturb a matched forward ray from identical recurrent state. That bounded evidence does not turn it into a physiological model or full graph.

## Provenance of model choices

| Element | Status in the baseline | Required wording |
|---|---|---|
| Source neuron identities, directed chemical contacts, raw counts, annotations | source-observed for the fixture and named G1 full artifact; exact retention/filter identities remain mandatory | “the fixture/full artifact is derived from pinned sources under the named manifest and companion receipt” |
| Transmitter predictions | source prediction with confidence/provenance | “predicted transmitter identity” |
| Connection sign | coarse inference from transmitter/research policy; unknown retained explicitly | “inferred sign policy,” not biological sign |
| Count-to-weight mapping and incoming normalization | assigned engineering transformation | “synapse-count-based normalized weight” |
| Leak, recurrent/sensory gain, bounded activation, one-step delay, zero baseline | assigned engineering dynamics | “deterministic leaky-rate model,” not physiology |
| Sensory mapping | manually configured sparse mapping to named boundary populations | “engineered sensory adapter” |
| Motor readout | manually configured named pool mean/gain/dead-zone/clip | “engineered motor adapter” |
| `_m` / `_m_s` protocol coordinates | one block treated as one engineering metre; blocks/tick multiplied by 20 | “engineering coordinates,” not physically calibrated SI units |
| Minecraft collision/gravity/physics | game environment | “Minecraft physics,” not fly biomechanics |
| Learning/training | none in baseline Faithful mode | say “untrained baseline”; never imply learned biology |
| Later bounded calibration | permitted only for allow-listed global/type/adapter parameters using external evidence | list every fitted parameter, data split, seed, and uncertainty |

Forbidden in Faithful mode: learned internal edge weights, latent neuron embeddings, MLP/GRU/attention dynamics, arbitrary all-neuron readout, imitation, reinforcement learning, a hidden conventional policy, or tuning on final Minecraft reward. A model using those components must have a distinct mode name such as `learned-connectome-architecture` and cannot inherit Faithful claims.

## Currently supportable statements

The following wording is accurate now:

> NeuroCraft Fly is a privately published research prototype with G0, G1, fixture-scope G2, and fixture-scope G3 passed at named scopes. The G1 evidence bundle for `male-cns-v1.0-superclass-v1` consists of its content-addressed manifest plus `RUNTIME-MCNS-01`; assigned sparse leaky-rate dynamics execute on the attributed 14-node/22-edge MaleCNS-derived fixture, exactly replay its full fixture trace, and respond to named actual-fixture zero-recurrence (sign-ablation) and matched-ray mutations. In the verified `EXTERNAL-BRIDGE-03` run, Minecraft observations stimulated named model populations of that MaleCNS-derived circuit and the resulting named motor-pool readout exclusively controlled one custom fly through primitive actions; nine records exactly replayed with eight pre-actuation and seven next-tick world cross-links, and all nine state/action paths diverged to neutral actions under zero recurrence. The complete mandatory AB-11 test-power suite is recorded in `AB11-MUTANTS-01` with an independent adversarial review. Separately, `LIVE-CIRCUIT-01` observed that fixture circuit advancing one custom fly in a normal headless development server at about 20 commands/s, followed by intentional fail-neutral disconnect. This does not imply full-graph dynamics, full-runtime latency, real-time (G3-R) qualification, production-JAR/manual-client observation, Minecraft-physics replay, a long soak, G4 behavioral results, or full-toolchain reproduction. Historical `EXTERNAL-BRIDGE-01` and `EXTERNAL-BRIDGE-02` remain bound to their own commits.

Research reports may compare source facts and planning calculations if each is cited and labelled. Architecture selection does not mean dependency incorporation, build success, dataset availability, runtime execution, or scientific validation.

## Statements prohibited by present evidence

Do not claim or imply that:

- a biological fly brain, complete fly, recovered physiology, or uploaded individual controls Minecraft;
- historical `EXTERNAL-BRIDGE-01`, current `EXTERNAL-BRIDGE-02`, `LIVE-CIRCUIT-01`, or any future bounded candidate is already a promoted G3 result, a production-JAR/manual-client demonstration, or a full-MaleCNS live controller;
- the full graph runs in the Minecraft loop or in real time; the authenticated Minecraft run used the 14-node/22-edge MaleCNS-derived fixture;
- full MaleCNS executes in real time or fits any measured memory/latency budget;
- the model recreates biological voltages, spikes, learning, internal state, perception, muscles, flight, or natural behavior;
- all source neurons/synapses participate, unless a runtime manifest and trace prove the exact count;
- a successful Minecraft task validates the dynamics or adapters biologically;
- a lesion establishes a biological mechanism without preregistration and matched controls;
- one connectome represents all flies, females, ages, conditions, or individuals;
- the artifact is a whole fly, digital fly, exact fly, uploaded brain, uploaded consciousness, living organism, or sentient entity;
- private publication implies scientific promotion, public reuse rights, a reproducible release, or G6; production-JAR/manual-client observation, cold reproduction, compliance, security remediation, and remaining gate work are still open.
- deterministic execution on one platform implies bitwise-identical floats, stream hashes, or pilot results on another platform.

## Why this is not consciousness or brain uploading

A connectome is a reconstruction of structural connectivity, not a complete physical or functional state. It omits or incompletely represents membrane and receptor physiology, electrical coupling, neuromodulation, plasticity, molecular state, ongoing activity, peripheral organs, body, development, experience, and environment. The proposed dynamics and interfaces add engineered assumptions; they do not recover the source animal's state, memories, identity, subjective experience, or consciousness. NeuroCraft Fly has no validated consciousness measure or theory. “Uploading” or “bringing a fly to life” is scientifically unsupported and prohibited.

## Required causal and negative evidence

G3 and stronger claims require all hard anti-bypass criteria in `PLAN.md`, including:

- no raw observation access outside the sensory adapter;
- no observation-to-action route around the recurrent state;
- no vanilla AI, navigation, hidden script/LLM/policy, operator control, or stale fallback;
- primitive bounded actuation and reset-only teleport;
- graph/hash/source-row identity enforcement;
- named, low-capacity sensory and motor adapters;
- fail-closed protocol behavior;
- authoritative observation-to-world trace;
- presentation isolation;
- mutations proving tests detect zero/frozen/decorative neural computation and injected hidden policies.

G4 additionally requires no-input, zero/frozen graph, weight/sign shuffle, matched directed rewiring, sensor/motor permutation, lesion/sham/matched lesions, direct low-capacity baseline, scripted baseline, and passive-physics controls under identical interfaces. Active `control-matrix-v2` declares 640 unexecuted Minecraft executions: 16 conditions, two arenas, ten seeds, and two reproducibility repeats. The repeats are not independent and are aggregated within arena-seed-condition before arena-seed paired analysis. A difference from zero shows internal dynamics matter; a difference from matched rewiring shows the broken structural property matters; neither alone shows biological reproduction. G4 remains unexecuted.

## Approved wording by evidence gate

### README / repository description

- **Now:** use the current supportable statement above and retain the historical/quarantined external-receipt and open-G3 qualifiers.
- **After G1:** “The named graph artifact was deterministically derived from hashed MaleCNS v1.0 files using the documented filters; this establishes structural provenance only.”
- **After fixture G2:** “Deterministic leaky-rate dynamics execute on the named source-derived fixture, reproduce the same-build full fixture trace, and respond to the named fixture mutations; the dynamics are assigned, not measured physiology.”
- **After G3:** “In the verified run, Minecraft observations stimulated named model populations and the resulting named motor-pool readout exclusively controlled one custom fly through primitive actions.”
- **After G4:** “Under fixed adapters and preregistered arenas, the selected connectome property contributed to the reported Minecraft outcome relative to the named matched controls.”
- **After G6:** append the exact commit, artifact hashes, hardware, rates, claim level, reproduction boundary, and private-publication state.

### Demo narration after G3

> This custom Minecraft fly is controlled by a deterministic model whose graph is derived from the named MaleCNS v1.0 artifact. Minecraft sensors are mapped by an engineered adapter to named boundary populations. Model state evolves under disclosed leaky-rate equations, and a constrained decoder reads named motor populations to produce primitive Minecraft actions. The source supplies structural connectivity; the dynamics, signs, sensor mapping, motor mapping, and Minecraft body are modelling assumptions. This is not a biological fly, a recovered brain state, or consciousness.

If only the fixture runs, replace “graph” with “MaleCNS-derived circuit” and name its node/edge count from the manifest. If the full graph runs, state its retained count and filter/hash. Do not use “real time” unless the measured qualifier passes.

### Publication-style wording after the relevant gates

> We constructed a deterministic controller on a graph derived from the pinned MaleCNS v1.0 chemical-connectome release. A documented Minecraft sensor adapter stimulated named CNS boundary populations, synchronous leaky-rate dynamics propagated state over immutable directed edge support with synapse-count-derived weights, and a constrained named motor-pool adapter produced primitive actions. We evaluated causal dependence with zero/frozen, shuffled, rewired, lesion, adapter, baseline, and passive-physics controls. The model does not reconstruct cellular physiology, peripheral transduction, muscles, natural flight, an individual fly's functional state, or consciousness.

Use “consistent with a named Drosophila response” only after G5 and include the exact stimulus/outcome, preregistered direction, controls, effect/uncertainty, specimen boundary, and sensitivity failures.

## Promotion audit

Before changing any claim:

1. Identify the exact gate and artifact/run hash.
2. Link the required test, benchmark, trace, rights, and review receipts.
3. Confirm circuit versus full-graph scope and measured versus target status.
4. Confirm every critical/high risk affecting the claim is closed.
5. Run the prohibited-language search across README, UI, narration, docs, metadata, and release notes.
6. Record the promotion in `STATUS.md`, `DECISIONS.md`, the relevant ADR, and this file.

If evidence later fails, downgrade the wording immediately; preserve the failed receipt rather than rewriting history.
