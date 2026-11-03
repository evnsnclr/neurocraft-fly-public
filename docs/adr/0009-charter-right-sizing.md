# ADR 0009: Right-size the charter — G4 rewrite path, G5 off the critical path, G6 venue review, G3-R disposition

_Gate mechanics, accepted 2026-08-26_

---

| Field | Value |
|---|---|
| **Status** | Accepted; G4 amendment itself remains to be drafted and frozen before execution |
| **Date** | 2026-08-26 |
| **Authority** | Project owner decision (Evan Smith); amendment mechanics per D-014 |
| **Risk lineage** | Solo-scale schedule risk; reviewer-recruitment dead end (ADR 0010) |
| **Evidence** | The one load-bearing external fact is primary-source verified and quoted in the decision (JOSS public-history requirement, joss.readthedocs.io, 2026-08-26, independently re-fetched during adversarial review). The strategy synthesis and venue-verification report are session artifacts, cited as context rather than committed evidence |

## Context

The charter was sized for a funded lab: G4 preregisters 640 Minecraft
executions, G5 asks for a biological-correspondence result, and G6 requires an
independent scientific review the project would have to recruit by itself. As
a solo program under the engineering frame (ADR 0008), each of these is either
oversized, off-frame, or unfillable as written. G0–G3's evidence machinery is
none of those things and is untouched.

## Decision

1. **G4 — rewrite via D-014's declared amendment path.** The active
   `control-matrix-v2` (640 executions) is preserved as immutable design
   history. Before any G4 execution, a successor preregistration will declare:
   - an **offline condition matrix** run in the fixture harness: connectome
     versus matched-null conditions, at least 100 null samples per condition
     on the regression-readout arm, at least 10 seeds per condition, with the
     costly RL-readout arm restricted to a small named condition subset,
     declared before execution;
   - a **reduced traced Minecraft evaluation matrix** (order of 6 conditions x
     2 arenas x 10 seeds) once a batch evaluation runner exists.
   The amendment must be frozen, hash-bound, and adversarially reviewed before
   its first trial, exactly as D-014 requires. Nothing is executed under this
   ADR alone.
2. **G5 — explicitly off the durable-goal critical path.** G5 was already
   "optional until a defensible biological target is preregistered"; under the
   engineering frame no biological-correspondence work is planned, and G5
   non-completion no longer blocks any other gate or the durable goal. The
   gate definition is retained unmodified in case a future collaboration
   revives it.
3. **G6 — review clause satisfied by venue peer review.** The
   "independent scientific/anti-bypass review" requirement is satisfied by
   completed external venue peer review — a closed JOSS review and/or
   acceptance with review at a named workshop, conference, or journal —
   together with the unchanged cold-reproduction, security-rescan, rights,
   SBOM, and publication-receipt requirements. Verified 2026-08-26: JOSS
   requires more than six months of public, actively developed repository
   history before submission, so the public-release date starts that clock.
   Self-recruited expert review is retired (ADR 0010).
4. **G3-R — kept open with a named disposition.** G3-R (real-time/long-soak)
   is neither passed nor silently dropped: it is scheduled to be measured as a
   soak recording alongside the live-demo lane, or explicitly deferred by a
   future decision if that lane closes without it.

## Consequences

- `PLAN.md`'s gate table remains the detailed reference; its G4/G5/G6 rows are
  to be annotated to point at this ADR in the next documentation pass, and the
  G4 successor preregistration document implements item 1.
- The 640-execution matrix stops being the project's implied centerpiece;
  statistical power moves to the offline matrix where samples are cheap.
- The reviewer gap stops being a blocker: review is obtained where venues
  already provide it, and outreach to directly relevant research groups
  replaces recruitment.
- No gate is promoted by this ADR; it changes what future evidence must show,
  never what existing evidence is claimed to have shown.
