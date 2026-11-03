# ADR 0002: Use deterministic sparse leaky-rate dynamics

_Neural execution model for Faithful Organism Mode, accepted 2026-08-16_

---

| Field | Value |
|---|---|
| **Status** | Accepted and implemented; fixture-scoped G2 verified, stronger scopes open |
| **Date** | 2026-08-16 |
| **Baseline** | Synchronous fixed-step sparse leaky-rate recurrence |
| **Reference backend** | Deterministic single-thread CPU |
| **Research basis** | [Executable neural-system decision](../research/neural_model_decision.md) |

The status above is current. At decision time this dynamics model was
unimplemented and unbenchmarked. The 2026-08-17 evidence implements and tests
the equations, one-step delay, adapters, replay, and causal mutations on the
14-node/22-edge MaleCNS fixture. It does not promote faithful full-graph
runtime, Minecraft closed-loop causality, or biological behavior.

## 📋 Context

A chemical connectome supplies structural contacts and metadata, not membrane equations, complete signs, delays, basal activity, plasticity, muscles, or a uniquely identified controller. Minecraft success cannot recover those missing quantities. The first runtime must therefore minimize hidden capacity, expose every assigned assumption, preserve source topology/counts, and support exact causal intervention and replay.

Whole-graph spiking, learned message-passing policies, and high-capacity adapters introduce unsupported parameters or a runtime envelope that has not been established for this project. The neural state must be active computation rather than decorative telemetry.

## 🎯 Decision

Use one dimensionless activity-deviation state `a_i[n]` per retained source neuron and a deterministic synchronous sparse recurrence. Let `C_ij` be the immutable raw synapse count from source neuron `j` to target neuron `i`, and let `sigma_ij` be a separately versioned sign/provenance policy. The initial effective recurrent weight is:

$$
W_{ij} = \sigma_{ij}\frac{C_{ij}}{\max(1,\sum_k C_{ik})}.
$$

For sparse sensory adapter `B`, raw sensor vector `u[n]`, intervention `p[n]`, baseline `b_i`, ablation mask `m_i`, and integer delay `d_ij >= 1`:

$$
z_i[n] = g_{rec}\sum_j W_{ij}a_j[n-d_{ij}] + g_{sens}(Bu[n])_i + p_i[n] + b_i,
$$

$$
a_i[n+1] = m_i\left(\lambda_i a_i[n] + (1-\lambda_i)\tanh(z_i[n])\right).
$$

Every update reads the previously committed state and delay buffers, then commits all neurons together. Ablation clamps the selected neuron, suppresses its outgoing contribution, and clears its delayed history at the declared intervention step.

Initial values are engineering assumptions rather than biological measurements:

| Parameter | Initial value | Interpretation |
|---|---:|---|
| **Neural step** | `10 ms` | Fixed simulated integration interval |
| **Global leak time** | `50 ms` | Assigned memory scale |
| **Edge delay** | `1` step | Numerical causal delay, not measured conduction |
| **Recurrent gain** | `0.9` | Conservative baseline to test |
| **Sensory gain** | `1.0` | Adapter scale baseline |
| **Baseline** | `0` | Zero activity deviation |
| **Noise** | none | Deterministic Faithful baseline |

## 🔌 Adapter and decoder limits

The sensory adapter consumes only the versioned raw physical sensor schema. Each nonzero mapping targets a named, source-traceable sensory boundary population and uses only declared affine scaling, offset, clipping, and saturation. It cannot classify objects, select targets, remember action state, or emit motor values.

Motor channel `k` reads only a named motor/descending population `P_k`:

$$
y_k[n] = \operatorname{clip}\left(g_k\frac{\sum_{i\in P_k}q_{ki}a_i[n]}{\sum_{i\in P_k}|q_{ki}|},-1,1\right).
$$

The decoder may apply only declared population weights, gain, dead-zone, clipping, and optional first-order smoothing. It must reject an empty pool or zero denominator. It has no raw-observation reference and cannot contain obstacle avoidance, target selection, sequencing, PID/heading hold, or semantic movement macros.

The Java actuator accepts only bounded primitive channels such as thrust and torque. It never receives semantic behavior names.

## ⚙️ Determinism and provenance

- Canonical body-ID ordering, edge ordering, and accumulation order are stable.
- Source edge support and raw counts are immutable and separately hashed from model overlays.
- Activity advances by integer neural steps, never variable wall-clock deltas.
- Faithful mode has no unseeded randomness; a later noise experiment must be separately named and counter-seeded by run, neuron, and step.
- Interventions, sign status, delay state, and adapter/pool values are inspectable and included in the trace identity.
- A float64 dense oracle validates small graphs; the runtime may use float32 only after parity tests.
- Same-build single-thread replay is byte-identical. Faster backends must pass declared numerical and action parity.

The initial sign policy is a disclosed coarse inference, not receptor-resolved biology: ACh positive; GABA, glutamate, and histamine negative; monoamines positive; unknown/unclassified zero. Mandatory controls include unsigned, glutamate-flip, monoamine-zero, sign-shuffled, and unknown-sign alternatives.

## 🚫 Forbidden learning

Faithful Organism Mode forbids internal edge training, topology edits, latent neuron embeddings, MLP/GRU/attention updates, arbitrary all-neuron readouts, imitation, reinforcement learning, and Minecraft-reward fitting.

Later calibration may change only an explicit allow-list of global or source-type-shared dynamics parameters and sparse boundary-adapter parameters, using independent biological evidence or development-only conditions. Before/after manifests must prove source edge support and raw counts are unchanged. A learned graph policy requires a different mode name and claim.

## ⚡ Consequences

### Positive

- The structural graph remains necessary and auditable.
- Synthetic equations, delays, interventions, and replay have direct golden tests.
- Sparse matrix execution is a plausible candidate for ordinary hardware without implying measured speed.

### Negative

- Activity is not voltage, calcium, spikes, or firing rate.
- Incoming normalization, signs, gains, leak, and delays are modelling choices requiring sensitivity analysis.
- Zero baseline/noise may produce quiescent or stimulus-locked behavior.
- A point-rate model omits morphology, electrical synapses, modulation, plasticity, and biomechanics.

## 🔍 Rejected or deferred alternatives

- **Whole-graph LIF as baseline:** deferred because its parameters and live compute cost are not established. A small-circuit Shiu-style LIF backend may serve as an audit after reference parity.
- **Event-driven or GPU execution:** deferred until measured activity/hardware results beat the reference while preserving determinism.
- **Graph neural policy or reservoir with arbitrary learned readout:** rejected for Faithful mode because behavior can reside outside the source-derived recurrence.
- **Direct observation-to-action baseline:** allowed only as a separately labelled control using identical sensors and primitive actuators.

## 🧪 Promotion evidence

G2 requires analytic synthetic tests, a genuinely MaleCNS-derived fixture trace after G1, synchronous/delay/sign/ablation/stimulation tests, anti-bypass mutations, snapshot/replay, observability, and full-graph benchmark receipts. A passing synthetic suite alone proves software algebra, not source-derived execution.

## 🔗 References

- [Architecture overview](../ARCHITECTURE.md)
- [ADR 0001: Connectome data](./0001-connectome-data.md)
- [ADR 0004: Causal trace and replay](./0004-causal-trace-and-replay.md)
- [Executable neural-system decision](../research/neural_model_decision.md)
- [Systems and performance plan](../research/performance_plan.md)
