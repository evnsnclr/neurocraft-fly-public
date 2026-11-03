# ADR 0038: Scope of the trained-readout negative result

Date: 2026-11-03
Status: accepted; supersedes the inference, not the figures, in ADR 0019

## Context

ADR 0019 records a trained-readout measurement: raw held-out MSE 95.66× and
bounded-projection MSE 403.70× a capacity-matched direct controller, with the
readout artifact pinned at SHA-256
`2090345fea54ddaa92a2ea2dd8f573ab2f851c5849047f94c269e0a60ee8912e`. It then
infers that "the connectome is a poor policy network", and that sentence was
copied onto every current surface: `README.md`, `GOAL.md`, `CLAUDE.md`,
`AGENTS.md`, the scientific-claims guide, the model card, the public README and
release scope, the HUD details screen and the GameTests that pin its strings.

The 2026-10-25 scientific audit questioned the scope, and the 2026-11-03
publishable-state review verified it against the artifact itself. The pinned
artifact's `graph_runtime_identity_sha256` is
`ebc50c0db92f7ec0e0eeb737af3d6d446cd569a488564a5682a0f255af401c26`, which is
exactly the runtime identity that [`FIXTURE-MCNS-02`](../receipts/FIXTURE-MCNS-02.md)
records for the **128-neuron, 887-edge** fixture. The artifact declares 32 input
bodies, 33 readout bodies, a zero-bias ridge fit (`ridge_l2` 1e-4) and its held-out
MSE metrics; the shelved training script on `codex/p0-p4-program-2026-09-01`
defines the synthetic direct-feature teacher it imitated, open loop. It never
touched the 166,700-node runtime, and it measured imitation of a formula, not
closed-loop navigation.

The project's first rule is to never claim the connectome does something it
doesn't. A negative claim stated too broadly violates the same rule as a
positive one: a reader was being told that the full network had been tested as
a controller and had failed.

## Decision

1. The figures stand unchanged, with their provenance in ADR 0019, which is not
   edited. What changes is the sentence around them. The defensible statement is:
   *one readout trained on the 128-neuron, 887-edge fixture performed about
   95.66× (raw) and 403.70× (bounded projection) worse than a capacity-matched
   direct controller on a synthetic open-loop imitation task.*
2. Every current guidance surface (listed under Consequences) says that, names
   the fixture and the task, and states
   that this was not a full-network or closed-loop navigation benchmark and is
   neither a general control advantage nor an impossibility result. The HUD
   details lines carry `128-NEURON FIXTURE · SYNTHETIC IMITATION`, and the client
   GameTests pin those strings alongside the figures.
3. The connectome's value in this project is still stated as its real labeled
   wiring. What is withdrawn is the claim that the full network has been shown
   to be a poor controller; what remains open is any claim that it is a good
   one. The continuous-control lane, if it ever reports a full-network control
   result, reports it under its own identities and receipts.
4. Historical receipts, dated snapshots, design notes and ADR 0019 keep their
   original wording. ADR 0033's and the 2026-08-18 documents' mentions are
   history, not current guidance.

## Consequences

- `README.md`, `GOAL.md`, `CLAUDE.md`, `AGENTS.md`, `docs/SCIENTIFIC_CLAIMS.md`,
  `docs/MODEL_CARD.md`, `docs/EVIDENCE.md`, `docs/RECORD_IT_YOURSELF.md`,
  `docs/public/README.md`, `docs/public/RELEASE_SCOPE.md`, the blog draft, the
  current operator guide, the public release runbook, the closing cell of the
  third notebook, the private agent prompts, `NeuroCraftHud`, `ModelDetailsScreen`
  and three client GameTests change in the same commit as this ADR.
- The curated public export includes this ADR and `FIXTURE-MCNS-02` so that a
  reader can check the fixture side of the identity match; the readout artifact
  itself stays on the shelved private branch and is not exported.
- No model, calibration, binding, receipt or graph identity changes. No new
  measurement is claimed. A full-network control benchmark remains unmeasured.
