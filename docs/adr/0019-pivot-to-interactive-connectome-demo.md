# ADR 0019: Pivot to an interactive connectome demo

_Program direction, accepted 2026-09-02_

---

| Field | Value |
|---|---|
| **Status** | Accepted as direction. No implementation claim: the interactive demo described below is planned, and the roadmap that reaches it is `PLAN.md` |
| **Date** | 2026-09-02 |
| **Authority** | Project owner decision (Evan Smith), 2026-09-02 |
| **Precedent** | ADR 0008 (engineering/ML frame, three-mode taxonomy) and ADR 0009 (charter right-sizing) already moved the program away from biological claims; this ADR completes that move and drops the paper |
| **Supersedes** | The paper-oriented charter: `GOAL.md`'s "reproducible research prototype … whose primary demonstration is a scientifically honest Faithful Organism Mode" objective, `CLAUDE.md`'s "Deliverables: a research paper, a runnable demo, and a public open-source repository", ADR 0005's G3–G6 promotion ladder **as a project goal**, and the plan of 2026-08-18. ADR 0005's gate *definitions* and every promotion made under them remain valid history |
| **Evidence** | `ESCAPE-ARC-01`; D-051; D-058; D-062; D-066; D-069; D-070; D-071 / ADR 0018; and the P1 readout measurement recorded below, whose only record is the shelved `codex/p0-p4-program-2026-09-01` branch. Nothing new is measured here — this is a direction record |

## Context

### What the operator actually wants

Evan is a machine-learning engineer, and he wants something remarkable to show people.
Not a paper: no venue, no reviewers, no submission. He does not want biological claims.
He wants a Minecraft experience where the real fruit-fly connectome runs live and the
fly reacts to him — to being approached, touched, hit, fed, lit, and crowded by other
mobs — with its real labeled pathways visible firing on a HUD.

The charter that has been driving this repository was built for a different deliverable.
It optimized for auditability against a paper's reviewers: freeze-then-measure, a
content-addressed receipt for every change, a `DECISIONS.md` row and an ADR per
constant, a preregistration before each measurement, and an enforced claim vocabulary.
That machinery is excellent at what it was for. It is also why the interactive premise —
does the fly react to a person at all? — went untested for months: the sensor rays could
not see a player until ADR 0017 / D-070, and the client model was a metronome
independent of the motor vector until ADR 0016 / D-069.

### EON as the model

The reference is EON Systems' fruit-fly demo
(<https://eon.systems/updates/weve-uploaded-a-fruit-fly>): a connectome running live,
driving a body, reacting to a viewer. Its architecture is labeled-population readouts
that **select and modulate scripted body programs**, over hand-chosen mappings and
event-gated sensory inputs. EON are candid about it — they describe their
descending-neuron-to-controller mapping as "somewhat arbitrarily chosen by hand" (EON
Systems) — and claim no intelligence for the result.

EON is a **new** reference. The repository's existing reference-project forensics
(REF-02, `docs/research/original_project_forensics.md`) concern a different, unrelated
Instagram demonstration, and the EON facts cited here come from EON's own page rather
than from a repository record.

Two things separate what we can build from what they built:

1. **We have found no prior instance of a full adult-fly connectome running in real
   time inside a game engine, closed-loop, with a person interacting with it.** This is
   a survey claim, not a proof, and it is written that way everywhere it appears. What
   *is* measured is the latency it rests on: the whole 166,700-node artifact steps
   inside the tick budget on the development machine — p50 24.4 ms, p99 32.0 ms against
   a 50 ms game tick, maximum 75.6 ms disclosed, with no real-time claim attached
   (D-051).
2. **MaleCNS includes the ventral nerve cord.** It spans central brain, optic lobes,
   neck connective, and VNC in one reconstruction (ADR 0001,
   `docs/SCIENTIFIC_CLAIMS.md`). EON's demo runs a FlyWire brain connectome; a FlyWire
   FAFB reconstruction is the brain and optic lobes without a connected VNC or
   peripheral motor system (`docs/research/original_project_forensics.md`), which is
   exactly the deficiency ADR 0001 cites when it rules out brain-only sources for a
   source-to-motor chain. We have the fly's actual motor neurons, so a readout can be
   taken at the motor stage — TTMn, proboscis and leg motor neurons — instead of
   invented above it. `ESCAPE-ARC-01` already exercises exactly that path.

### The P1 negative result, which decides the framing

The P1 lane measured a readout trained on the connectome against a capacity-matched
direct controller on the same navigation task. Raw held-out MSE `0.007157117334320995`
against `0.00007482049765503581` — about **95.66x worse**. Under the actual bounded
runtime projection, `0.00711924880856344` against `0.000017635134511335958` — about
**403.70x worse**. The readout artifact is pinned at SHA-256
`2090345fea54ddaa92a2ea2dd8f573ab2f851c5849047f94c269e0a60ee8912e` and reproduces
offline.

**The connectome is a poor policy network.** That result is not a setback to be managed
— it is the fact that determines how this project is allowed to talk about itself. The
connectome's value here is its **real labeled wiring**, not its competence at control,
and any pitch that implies otherwise is refuted by our own measurement.

**Where this record lives, stated plainly.** The P1 lane is part of the uncommitted
P0–P4 tree, which this ADR shelves to `codex/p0-p4-program-2026-09-01` unmerged. Its
decision row is numbered D-073 on that branch, and on `main` D-073 is one of the chip
commits instead. So `main` carries no other record of this measurement, and **this ADR
is its record**: cite ADR 0019 for the 95.66x / 403.70x figures, not a decision row.
If the P1 lane is ever cherry-picked, its row is renumbered and this paragraph is
superseded.

## Decision

### 1. The objective changes

From "a scientifically honest Faithful Organism Mode prototype plus a research paper" to
**an interactive Minecraft experience in which the real MaleCNS connectome runs live and
drives a fly that reacts to the player, to items, to light, to damage, and to other
mobs, with its real labeled neural pathways visible firing on a HUD.** `GOAL.md` is
rewritten accordingly; "done" is a stable, filmed, interactable session, not a gate.

### 2. The architecture is EON's, described as EON's

1. **Body programs** on the mod entity — hover, cruise, bank, land, walk, groom, feed
   with the proboscis, escape-jump, flee, approach — scripted motion and animation that
   the connectome selects among.
2. **Labeled readouts** drive that selection and modulate it. The escape arc is already
   measured (LC4 -> DNp01 -> TTMn); steering DNs, grooming DNs, the proboscis motor
   neurons, and leg motor neurons follow a census of the MaleCNS annotations, using the
   existing fail-closed census tooling so population names come from the artifact rather
   than from memory.
3. **Event-gated senses** from Minecraft events: player looming -> LC4, touch ->
   bristles, damage -> nociception, nearby sweet food -> gustatory, block light ->
   visual, other mobs as further looming and visual sources. The entity-aware sensor
   rays of ADR 0017 are the foundation; before them a player produced a bit-identical
   sensor vector to an empty arena.
4. **A live brain HUD** showing the labeled populations firing, building on
   `apps/brain-viewer` and `NeuroCraftHud`, read-only as both already are.
5. **A shuffle-brain toggle**: swap the live graph for a shuffled one inside a running
   session and watch the reflexes die, then swap back. The control identities already
   exist (`shuffled_weights`, `degree_preserving_rewire`, `RunMode.SHUFFLED_WEIGHT` /
   `DEGREE_PRESERVING_RANDOM` / `NO_INPUT`).
6. **Optionally later**, LIF spiking dynamics with Shiu et al. 2024 parameters, as a
   second declared dynamics identity beside the rate model, never a silent replacement.

**This architecture is described as exactly what it is** — hand-chosen labeled readouts
selecting and modulating scripted body programs over event-gated inputs — in the README,
in the HUD, and in anything filmed. It is not "the fly deciding".

### 3. The process changes

Feature branch -> tests -> one review -> merge. CI stays. The deterministic runtime and
the hash-verified artifact stay. Data hygiene stays. A `CHANGELOG.md` entry per feature;
a `DECISIONS.md` row only for a genuine direction decision; an ADR only for durable
architectural consequences.

### 4. One rule survives, and it is the important one

**Never claim the connectome does something it doesn't.** Kept not as science overhead
but because it is what makes this a real thing rather than a puppet show, and because it
is the first thing a skeptical engineering audience probes. Three facts stay visible
wherever the project is shown: the 95.66x / 403.70x readout result recorded above, the
measured escape arc with its stated bounds (`ESCAPE-ARC-01`), and the hand-chosen
readouts-over-scripted-programs architecture. Every lane and toggle is labeled honestly
— Faithful Organism Mode, engineered, shuffled, scripted, no-input — on the HUD and in
code, and a sentence that blends two of them is a defect.

## What is cut, and why

| Cut | Why |
|---|---|
| The research paper | Not the deliverable. No venue, no reviewers, no submission |
| The G3–G6 promotion ladder as a goal | It measures progress toward a paper's claims. Progress is now measured by what a person can see and do in a session |
| Freeze-then-measure | Its purpose was to make a measurement admissible to a reviewer. There is no reviewer, and it costs a full protocol cycle per experiment |
| A receipt for every change | Receipts exist to make a claim auditable. Ordinary features no longer make claims that need auditing. Existing receipts remain, unedited |
| A `DECISIONS.md` row and an ADR per constant | The record became a bottleneck on the work it was recording. Direction decisions still get rows; durable architecture still gets ADRs |
| Claim-vocabulary enforcement | The vocabulary was a proxy for honesty. The honesty rule above is kept directly, and it is stronger and cheaper |
| Preregistrations | Same reason as freeze-then-measure. Existing preregistrations stay as history and are never edited |
| The Nature / N0 / NF1 program | Dead. See disposition below |
| Byte-bound evidence files as a prohibition | Demoted to a practical caution: editing them rotates identities and invalidates calibrations, so re-run the calibrations in the same change |

**The Codex role configurations are retired too.** The seven read-only role definitions
under `.codex/agents/` — `connectome_researcher`, `license_security_reviewer`,
`minecraft_architect`, `neural_systems_engineer`, `prior_art_researcher`,
`scientific_reviewer`, and `test_reviewer`, all dated 2026-08-16 — were written for the
paper program and steer toward it by construction: `scientific_reviewer` is instructed to
read "like a skeptical connectomics and computational-neuroscience peer reviewer",
`connectome_researcher` and `prior_art_researcher` are dataset- and literature-facing
rather than demo-facing, and none of them has a mandate over body programs,
presentation, or whether a live session survives. That is not a neutral gap. The
uncommitted P0–P4 tree rewrote the project's own charter without an operator decision —
it appended a second objective ("NeuroArena") to `GOAL.md` and edited `CLAUDE.md`'s
non-negotiables — which is exactly the failure mode a role set with no owner for
direction produces. (To be exact about the blast radius: that tree did **not** touch the
identity-bound `loader.py`/`model.py` pair or the ruff-excluded acquisition scripts; it
did touch `packages/neural-runtime/.../model.py` and
`packages/connectome-data/.../__init__.py`.) `license_security_reviewer` is retained; its mandate is orthogonal to the paper program. Six replacement roles suited to the demo land with this change — body programs, readout census, sensory encoder, presentation/HUD, demo QA, and honesty review — each carrying the honesty rule and a prohibition on editing `GOAL.md`, `CLAUDE.md`, `AGENTS.md`, receipts, or ADRs.

## What is kept

- The honesty rule, in full, as stated above.
- CI, `make verify`, and the test suites.
- The deterministic runtime, the hash-verified artifact, exact replay, and the
  fail-closed protocol/validator boundaries — they are why sessions are debuggable and
  why the shuffle toggle proves anything.
- Data hygiene: no bulk source data, artifacts, worlds, EULA files, account material, or
  secrets in the repository.
- Every existing receipt, preregistration, and ADR, unedited, as history.
- Separate identities for controls. A shuffled run is never presented as Faithful
  Organism Mode.

## Disposition of the three work streams

**1. The Codex P0–P4 tree (uncommitted).** Preserved on branch
`codex/p0-p4-program-2026-09-01`, unmerged, with tests red. It contains transactional
neural proposals and delay-semantics v2, a playable-navigation lane, an articulated
five-axis presentation and HUD, a haltere causal-evaluation fixture, and a direct-goal
positive control. **P0a's transactional proposal lifecycle is a candidate cherry-pick
later** — the two-phase `proposal_seq` / `command_seq` split is the right shape for a
long interactive session and is independent of the paper program. Nothing from that tree
is claimed as merged, and its own ADRs are numbered 0019–0023, which collide with this
document; any cherry-pick renumbers them.

**2. The five `codex/*` Nature/FlyGym branches.** Deleted, with content preserved under
the `archive/codex-*` tags: `codex/flygym-cosim-v1`,
`codex/flygym-minecraft-guided-showcase-v1`, `codex/flygym-minecraft-live-body-studio-v1`,
`codex/nf1-expert-pilot-activation-v1`, and `codex/nf1-map-redesign-v1`. The Nature / N0 /
NF1 program is dead as a project goal.

**3. The three chip commits.** Merged as D-072, D-073, and D-074: the GameTest code
bundle made a function of the git tree, latency-only instrumentation on both sides of the
bridge, and the presentation visual-gain marker in the renderer receipt. All three serve
the demo — the first keeps evidence bundles honest without hand-enumeration, the second
measures exactly the cold-start latency ADR 0018 reasons about, and the third keeps the
declared display gain visible where it is applied.

## Consequences

- **The G0–G2 evidence stands, and so does fixture-scope G3.** G0 (D-009), G1 (D-010),
  fixture-scope G2 (D-011), and the fixture-scope G3 promotion on `EXTERNAL-BRIDGE-03`
  (D-027) are recorded evidence about recorded builds. They are unaffected by this ADR.
- **G3 at wider scope, G3-R, G4, G5, and G6 are retired as goals — not falsified.**
  Nothing about them is claimed to be false; they simply stop being what this project is
  trying to reach. `STATUS.md` keeps the gate table as evidence history under a heading
  that says so.
- **Prior forward-looking language becomes historical only.** Every "live run pending",
  "qualification blocked", "G4 execution", and "before promotion" clause in D-014,
  D-017, D-026, D-051 through D-066, and the control-matrix preregistrations now
  describes a program that is not being pursued. The rows are correct as history and are
  not edited.
- **`RISKS.md` entries scoped to publication, reviewer reception, and gate promotion lose
  their forward obligations.** Risks about session stability, honesty of presentation,
  and data rights do not.
- **The receipt corpus stops growing for ordinary work**, which means future readers
  cannot assume "no receipt" implies "not done". `STATUS.md` and `CHANGELOG.md` carry
  that weight instead, and they must be kept honest about what has actually been
  observed.
- **Stage 2 of `PLAN.md` gates everything.** No live session has yet survived under the
  current stack: the last known-good session died after 2,538 applied commands, three
  more died at cold start, and ADR 0018's fixes for both are unit- and
  integration-tested but have never been run live. ADR 0018 says so itself. Nothing in
  this ADR changes that, and no demo claim is available until a session is observed.
