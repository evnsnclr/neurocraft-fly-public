# ADR 0012: Re-anchor the command validator after entity-tick dormancy

_Dormancy availability semantics, drafted 2026-08-30_

---

| Field | Value |
|---|---|
| **Status** | Implemented; unit-tested at validator scope |
| **Date** | 2026-08-30 |
| **Authority** | Server-side `CommandValidator` admission and application rules |
| **Precedent** | ADR 0006 / D-025 reset-boundary retirement; ADR 0011 lateness retirement |
| **Evidence** | Dormancy suite in `CommandValidatorTest` (boundary no-op, re-anchor, re-armed deadline, sequence continuity, unverifiable and verified in-flight regressions) |

## Context

Every timing rule in the validator assumes `validate()` runs once per game tick,
because it is invoked only from `FlyEntity.tick()` via
`FlyControllerRuntime.control`. That assumption is about the *entity* clock, not
the *world* clock: when the fly leaves loaded chunks, the entity stops ticking
while `ServerLevel.getGameTime()` keeps advancing. The two clocks were conflated.

This was observed live on 2026-08-30: a whole-brain fly flew sustained, visibly
dodged obstacles, escaped the glass arena, and out-flew the loaded chunks. On
re-entry the first `validate()` call compared the current game tick against hold
anchors frozen hundreds of ticks earlier — `actuationHoldThroughTick()` had long
lapsed — and latched `DEADLINE_MISSED` immediately, killing the session. Nothing
had starved while the entity was actually live; the authority simply was not
running it. As with ADR 0006 and ADR 0011, this is an availability defect only:
no unsafe actuation path exists, because rejected frames always apply neutral
motors.

## Decision

The validator now tracks `lastValidatedGameTick`, updated at the end of every
`validate()` call. After the bridge-state gating and before the first-action
deadline and hold checks, a gap test runs:

```
lastValidatedGameTick >= 0
    && currentGameTick - lastValidatedGameTick > DORMANCY_GAP_TICKS
```

with `DORMANCY_GAP_TICKS = FIRST_ACTION_DEADLINE_TICKS + 1` (= 3). While the
entity is live, `validate()` runs on every tick and consecutive calls are one
tick apart, so a gap above this bound is impossible for a tick-continuous
session and can only mean the authority stopped ticking the entity.

When the gap test fires, the validator performs a **dormancy re-anchor** instead
of evaluating deadlines against pre-gap state:

- Every reserved frame is dropped fail-neutral — its target passed while no
  authority could apply it — and a dedicated `dormancyReanchors` counter
  increments once per re-anchor event (up to `MAX_PENDING_COMMANDS` reserved
  frames are dropped per event).
- The hold anchors reset exactly as a fresh activation would:
  `firstActiveGameTick` becomes the current tick and
  `lastAppliedValidThroughTick` returns to its unset sentinel. The first-action
  deadline is therefore re-armed, not waived: a re-anchored session that
  produces no applicable action within `FIRST_ACTION_DEADLINE_TICKS` of the
  re-anchor tick still latches `DEADLINE_MISSED`.
- No fault latches at the re-anchor itself.

Sequence state (`lastCommandSequence`, `lastNeuralStepEnd`) **persists**: the
stream identity is unbroken by dormancy. In-flight frames that reference
pre-gap observations then meet the unchanged ADR 0011 machinery in the same
pass. This is sound against `SentObservationWindow`: its receipts are recorded
only when the bridge encodes and sends an observation, observations are offered
only from the entity tick path, and a dormant entity offers nothing — so the
bounded three-entry ledger still holds the most recent pre-gap receipts at
resumption (at most the resume tick's own observation has been recorded ahead
of the first post-gap `validate()`). A pre-gap in-flight frame is therefore
still verifiable: identity, sequence, receipt, and recorded tick all check out,
its only defect is lateness, and it retires fail-neutral under ADR 0011. A
frame whose receipt is genuinely absent remains unverifiable and still latches
`STALE_COMMAND`.

Decision table, with `g = currentGameTick - lastValidatedGameTick`:

| Gap | Behavior |
|---|---|
| `g <= 1` (tick-continuous) | Byte-identical to before this ADR |
| `1 < g <= DORMANCY_GAP_TICKS` | Byte-identical to before this ADR: deadlines are evaluated against pre-gap anchors, so a lapsed hold latches `DEADLINE_MISSED`; the exact boundary `g = DORMANCY_GAP_TICKS` is pinned as still latching |
| `g > DORMANCY_GAP_TICKS` | Dormancy re-anchor as above; candidates in the same pass then flow through the unchanged admission rules |

Everything that is not the pre-gap hold evaluation still latches exactly as
before: `IDENTITY_MISMATCH`, `SEQUENCE_ERROR` (a gapped post-resume sequence
included), `FUTURE_COMMAND`, `STALE_COMMAND` on unverifiable frames, and
`DEADLINE_MISSED` on genuine post-resume starvation. ADR 0011 retirement
semantics are untouched.

Operator visibility: the single-fly `neurocraftfly status` line now ends with
`dormancy_reanchors=N` after `lateness_retired=N`, a process-lifetime count
across validators. The single stop-time reset hook registered in
`NeuroCraftFly.registerLifecycle` (a frozen call site) zeroes both process-global
counters through `resetTotalLatenessRetirements`.

## Consequences

- A fly whose chunks unload mid-session resumes cleanly when they load again:
  the session re-anchors, pending pre-gap actuation is dropped fail-neutral,
  and the stream continues at the next command sequence. Deadlines now measure
  live-authority starvation only, which is what they were always meant to bound.
- Genuine starvation is still fail-closed with the same constants: within any
  continuously-ticking stretch — including the stretch that begins at a
  re-anchor — an absent applicable action latches `DEADLINE_MISSED` within the
  same bounds as before.
- The causal record is not weakened: frames dropped at a re-anchor were never
  applied, every applied action retains its full verified provenance chain, and
  re-anchors are counted and disclosed via `dormancy_reanchors` in the operator
  status line.
- The change touches `mod/src` control code, so any evidence bundle that
  enumerates bridge/controller sources is superseded for future candidate runs
  and must be regenerated on the fixed tree, as with ADR 0006 and ADR 0011.
- The re-anchor decision is validator-scope unit evidence only: chunk-unload
  dormancy is not cheaply modelable in a GameTest (the entity must stop ticking
  under a live authority), and a live whole-brain session observation of a
  survived unload/reload cycle remains to be recorded.
