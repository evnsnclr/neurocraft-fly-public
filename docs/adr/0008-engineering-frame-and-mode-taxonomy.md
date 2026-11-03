# ADR 0008: Adopt an engineering/ML program frame and a three-mode taxonomy

_Program direction, accepted 2026-08-26_

---

| Field | Value |
|---|---|
| **Status** | Accepted as direction; Faithful Organism Mode unchanged; no Trained-Readout or Engineered-Encoder implementation exists yet |
| **Date** | 2026-08-26 |
| **Authority** | Project owner decision (Evan Smith), recorded from the 2026-08-26 strategy session |
| **Risk lineage** | Reception/credibility risks documented in `RISKS.md`; D-033/D-035/D-037 refusals |
| **Evidence** | `SIGN-OVERLAY-01` (first receipt produced under this frame). The strategy synthesis and its two independent adversarial critiques are session artifacts, cited as context rather than committed evidence |

## Context

The project's claims were framed biologically: Faithful Organism Mode (FOM) as
the headline, with G5 pointing at biological correspondence. Three refusals
protect that framing — D-033 (no unsigned full graph in the causal path),
D-035 (no invented descending-neuron-to-axis mapping in the decoder), and
D-037 (no policy-in-the-encoder optic-flow front end). They are correct for
FOM and remain so.

The project owner has redirected the program: the subject of the claims is now
**engineering, machine learning, reinforcement learning, and vision**. Biology
supplies structure — topology, sparsity, module identity — and the
claim-discipline guardrails, not the claims themselves. Under that frame,
trained readouts and engineered encoders are not violations to be refused but
experimental conditions to be declared. The charter already requires
non-faithful modes to carry separate identities; this ADR extends that
machinery rather than weakening it.

## Decision

1. **FOM is unchanged and demoted from sole frame to Mode 1 of 3.** Only FOM
   may use faithful language ("nothing else computes in the loop"). Its
   refusals, ceiling, and verification protocol are untouched.
2. **Two mode families are planned as first-class identities:**
   - **Trained-Readout Mode (TRM):** the measured circuit is the frozen
     recurrent core; a declared, trained readout (regression or small RL) maps
     hidden state to actuation. Nothing else about the dynamics is learned.
   - **Engineered-Encoder Mode (EEM):** a declared, engineered sensory front
     end (for example an optic-flow encoder) feeds a measured circuit whose
     inputs biologically expect that signal.
   Each mode requires its own identity, preregistration document, and
   verification protocol, recorded **before** any result is generated under
   it. Neither exists yet; this ADR authorizes their design, not their
   results.
3. **Refusal scoping.** D-033 binds every live causal path in every mode: the
   unsigned full graph stays out, and any offline use of it must be labeled
   `unsigned topology reservoir`, never "the connectome". D-035 and D-037
   continue to bind FOM exactly as written; TRM and EEM may implement trained
   readouts and engineered encoders respectively because their identities
   disclose exactly the assignment those refusals exist to keep out of FOM.
   This scoping is enacted here, by recorded decision — not by
   reinterpretation of the original entries, which remain correct history.
4. **Headline-language rule.** FOM's purity sentence and TRM/EEM's
   trained-boundary sentences are never blended in any document, README,
   video, or paper. A mixed sentence is treated as a claim-discipline defect.

## Consequences

- The planned research program ("Structure vs. Learning": connectome vs.
  matched-null frozen reservoirs with trained readouts, plus attribution
  analysis) becomes chartered work instead of a rules conflict.
- Sign coverage stopped being an assumption: `SIGN-OVERLAY-01` measures the
  coarse-consensus overlay at 869/887 signed edges (97.97%) on
  `mcns-v1.0-flight-steering-v1`, with all 18 zeroed edges from `unclear`
  consensus NT. Experiments that lean on signed dynamics cite that receipt,
  including its disclosed policy caveats.
- New null-condition generators and TRM/EEM modules land as new code with
  their own identities; the byte-bound and identity-bound files are untouched.
- `GOAL.md`/`PLAN.md` language moves to the engineering frame in a later
  documentation change that references this ADR; gate mechanics are handled in
  ADR 0009.
