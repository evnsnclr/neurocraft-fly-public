# ADR 0010: Cancel the recruited-expert review workstream

_Workstream cancellation, accepted 2026-08-26_

---

| Field | Value |
|---|---|
| **Status** | Accepted and executed (archive + working-tree removal); branch deletion deferred |
| **Date** | 2026-08-26 |
| **Authority** | Project owner decision (Evan Smith) |
| **Risk lineage** | G6 review clause (now ADR 0009); solo-recruitment infeasibility |
| **Evidence** | Archive tarball SHA-256 `8e8a6171fb5e089ae4ed12c31bd06fffde7076820acd980a460d48463be8ba3f` |

## Context

G6 originally required an independent scientific/anti-bypass review, and a
workstream grew around recruiting that review directly: 31 untracked JSON
schemas under `data/schemas/nature-nf1-expert-pilot-activation/` (reviewer
enrollment, credentials, rights determinations, review packets, session
receipts) and two related unmerged branches
(`codex/nf1-expert-pilot-activation-v1`, `codex/nf1-map-redesign-v1`). No
reviewer was ever enrolled, no schema was ever referenced from a tracked file
on `main`, and the project has no realistic path to recruiting domain experts
directly. ADR 0009 replaces the requirement with venue peer review.

## Decision

The recruited-expert review workstream is **cancelled**.

- The untracked schema directory was archived before removal:
  `nf1-expert-pilot-activation-schemas-20260826.tar.gz` (33 entries), SHA-256
  `8e8a6171fb5e089ae4ed12c31bd06fffde7076820acd980a460d48463be8ba3f`, stored
  outside the repository in the sibling `neurocraft-archives/` directory. The
  working-tree copy was then deleted. No tracked file changed.
- All five unmerged `codex/*` branches (the two nf1 branches and the three
  FlyGym showcase branches) are preserved as `archive/codex-*` tags. **Branch
  deletion is deferred** to the same later gate that decides the FlyGym
  showcase lane's fate, so nothing consumable is lost while the archive tar
  outlives every branch that references the schemas.

## Consequences

- G6's review requirement is carried entirely by ADR 0009's venue-review
  clause; nothing in the charter depends on recruited reviewers any more.
- The archived schemas remain recoverable byte-exactly from the tarball if a
  future collaborative phase revives structured external review.
- This is a workstream cancellation recorded per house rules — not a quiet
  deletion.
