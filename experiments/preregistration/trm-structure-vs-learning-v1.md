# TRM preregistration draft — Structure vs. Learning v1

- **Status: `draft_not_frozen`.** No result may be generated under this
  document. Freezing requires: every hash slot below filled, an adversarial
  review of the frozen text, and a DECISIONS entry recording the freeze —
  exactly the discipline D-014/D-029 established for the fixture-physics
  pilots. This draft satisfies the "recorded before any result" requirement of
  ADR 0008 for *design*, not for execution.
- **Mode:** Trained-Readout Mode (TRM), ADR 0008 Mode 2 of 3. Not Faithful
  Organism Mode; no FOM language may describe any run under this protocol.
- **Question:** does the measured connectome circuit's structure — its
  topology, weights, and transmitter-derived sign placement — contribute to
  closed-form readout performance on declared tasks, relative to matched
  nulls, when the recurrent core is frozen and only the readout is trained?
  Any outcome, including the null, is a reportable result.

## Substrate

| Element | Binding |
|---|---|
| Fixture | `mcns-v1.0-flight-steering-v1`, runtime identity `ebc50c0d…01c26` (`FIXTURE-MCNS-02`) |
| Sign coverage | `SIGN-OVERLAY-01` (D-042): 869/887 edges signed under `coarse_presynaptic_consensus`; the coarse whole-transmitter mapping is a disclosed modeling choice, not physiology |
| Engine | The pinned reference `NeuralEngine` (deterministic leaky-rate, bounded tanh), consumed byte-identically as pinned by pilot v3 |
| Harness | `neurocraft_experiments.nulls` + `neurocraft_experiments.reservoir` (new modules; unit-tested; the freeze binds their exact bytes) |
| Run identity | `neurocraft.trm.reservoir_run.v1` (`reservoir_run_identity`) |

## Conditions (implemented; identities distinct by construction)

1. `connectome` — consensus signs, canonical weights and topology.
2. `sign_permuted` — the sign multiset preserved, placement permuted
   (`nulls.permuted_sign_overlay`; value/provenance pairs travel together).
3. `sign_ablated_zero` — all signs zero; the zero-recurrence control.
4. `weight_shuffled` — canonical topology, weight multiset permuted
   (pinned `WeightOverlay.shuffled`).
5. `degree_preserving_rewire` — pinned `TopologyOverlay.degree_preserving_random`.

Open decisions the freeze must settle, recorded here so they cannot be settled
silently after results:

- **Rewire swap count.** The pilot used `requested_swaps=1`; a reservoir null
  needs enough swaps to destroy structure. Candidate: swaps equal to edge
  count, with the accepted-swap count reported. Must be fixed at freeze.
- **Signs under rewire.** The harness's declared default keeps each sign with
  its edge slot. The alternative — re-deriving signs from the rewired
  presynaptic neuron — requires a new derivation path and its own review.
- **Strength-matched weight null.** Not implementable today without amending
  a pilot-pinned file (`WeightOverlay` accepts only its two known kinds).
  Either the freeze drops the condition with this reason, or a pilot-v4-class
  amendment adds a new overlay kind first. It must not be hacked around.
- **Per-condition seed derivation.** Adversarial review measured that the same
  integer seed produces the *identical permutation* in the sign-permutation
  generator and the pinned `WeightOverlay.shuffled` (both use the same
  seeded-SHA-256 key sort over the same index domain). No current condition
  combines them in one engine, but the freeze must declare a per-condition
  seed-derivation rule so nominally independent nulls never share a
  permutation.
- **Run-identity hardening.** `reservoir_run_identity` binds every overlay,
  the dynamics, the graph, and the drive — but the review demonstrated three
  gaps the freeze must close or accept in writing: the `condition` label is
  caller-asserted (an engine built under one condition can be hashed under
  another label; the overlay hashes keep the content honest, the label can
  lie), engine freshness is unbound (a pre-stepped engine hashes identically
  to a fresh one), and a prospective identity can name an undriveable drive.
- **Readout-weight hashing.** `ridge_readout` output is `np.linalg.solve`
  output; last-ulp BLAS variation makes cross-platform byte-hashing of the
  weights fragile. The freeze must either declare weight hashes same-build
  scoped or avoid hashing solver output.

## Power and arms

- **Ridge arm (primary):** every condition, at least **100 null samples per
  stochastic condition** and at least **10 drive seeds**, closed-form ridge
  (`reservoir.ridge_readout`) over the full hidden state, l2 grid declared at
  freeze. Rationale: with 20 nulls the minimum achievable empirical p (~1/21)
  cannot survive multiple-comparison correction across conditions and metrics.
- **RL arm (secondary, restricted):** at most 3 named conditions, seed and
  budget declared at freeze; absent that declaration the arm is out of scope.
- **Biological-readout regime (descriptive):** the 2-pool wing-motor readout
  reported alongside the full-state readout, as a second regime, never as the
  headline.

## Tasks (candidates; the freeze must pick and bind exact generators)

- Memory capacity / delayed-signal reconstruction from the two mechanosensory
  entries.
- A declared nonlinear temporal benchmark over the same entries.
- Sensor-to-action mapping distilled from recorded FOM bridge traces (only if
  the traces and their identities are bound at freeze).

## Analysis and attribution

Primary metric per task declared at freeze (held-out MSE / decoding accuracy).
Attribution axes: sign placement (1 vs 2), recurrence (1 vs 3), weight
placement (1 vs 4), wiring (1 vs 5). Effect direction and uncertainty via the
null distributions; no post-hoc condition additions.

## Non-claims

No behavioral, biological, participation, or Faithful-mode claim. No Minecraft
execution occurs under this protocol. Results describe this fixture, this
engine, and these declared tasks.

## Freeze checklist

- [x] frozen protocol document:
  `trm-structure-vs-learning-v1-frozen.json`, SHA-256
  `c98b151296447920e9022b1c5e4fafe3e018fce260e463fd354d6f7d64e6bf01`
- [x] implementation hashes bound inside the frozen document: `nulls.py`,
  `reservoir.py`, `trm_tasks.py`, `trm_e1.py`, `scripts/run_trm_e1.py`
- [x] tasks bound (delayed reconstruction + NARMA-10; trace-distillation
  dropped with reason), l2 grid {1e-6, 1e-4, 1e-2, 1.0}, per-run validation
  selection, intercept column declared; the 2-pool biological-readout regime
  dropped with a recorded reason and the 1-vs-3 axis marked descriptive-only
- [x] power fixed by review: 199 nulls per stochastic condition (min p 1/200
  = 0.005 < 0.05/6), replacing the draft's >=100 after the freeze review
  proved N=100 could never reject under Holm-Bonferroni
- [x] rewire swaps = 887 (accepted count reported per run); signs travel with
  edge slots; strength-matched null dropped with the pinned-file reason;
  per-condition SHA-256 seed derivation; RL arm out of scope (E3)
- [x] run-identity hardening closed structurally in `trm_e1.built_run`
  (atomic engine+identity, fresh-engine guard, construction-validated drives);
  readout weights never hashed into identities
- [x] adversarial review recorded (freeze review, 2026-08-27, verdict
  fix-first with one critical statistical finding and three record findings;
  all remediated and the refrozen text delta-reviewed before the freeze
  commit)
- [x] DECISIONS entry: D-045
