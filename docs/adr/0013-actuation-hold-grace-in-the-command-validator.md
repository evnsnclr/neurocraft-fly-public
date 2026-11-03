# ADR 0013: Declared actuation-hold grace in the command validator

_Burst availability semantics, drafted 2026-08-31_

---

| Field | Value |
|---|---|
| **Status** | Implemented; unit-tested at validator scope |
| **Date** | 2026-08-31 |
| **Authority** | Server-side `CommandValidator` admission and application rules |
| **Precedent** | ADR 0006 / D-025 reset-boundary retirement; ADR 0011 lateness retirement; ADR 0012 dormancy re-anchor |
| **Evidence** | Grace suite in `CommandValidatorTest` (inclusive-boundary no-latch with fresh-frame recovery, past-boundary latch, cold-start pin, dormancy-during-grace interaction) plus the extended timelines listed below |

## Context

ADR 0011 made an isolated verified-late frame survivable (retire fail-neutral,
deadline unchanged) and ADR 0012 made entity dormancy survivable (re-anchor on a
validate-call gap the live entity cannot produce). A residual failure class sits
exactly between them: a **live-load latency burst** in which the entity ticks
continuously (so ADR 0012 never triggers) but two or more consecutive ticks pass
without an applicable frame. The hold rule latched `DEADLINE_MISSED` as soon as
`currentGameTick` passed the last applied frame's `validThroughTick` — effectively
one tick of tolerance — so a burst of two late frames killed the session even
though every frame in the stream was honest and verifiable.

This was observed live in escape-profile session
`interactive-20260831T032630Z-a84f8300`: ACTIVE at 23:26:51, `DEADLINE_MISSED`
in the status line at 23:27:09, with `command_seq=242`, `lateness_retired=3`,
and receipt supersession having fired twice. Every prior stability mechanism
functioned as designed; the latch came from the burst class none of them covers.
As with ADR 0006, 0011, and 0012, this is an availability defect only: no unsafe
actuation path exists, because rejected frames always apply neutral motors.

## Decision

Declare a fail-neutral grace on the applied-anchor hold:

```java
public static final long HOLD_GRACE_TICKS = 6;
```

`actuationHoldThroughTick()` returns `lastAppliedValidThroughTick +
HOLD_GRACE_TICKS` whenever the applied anchor is set. The value is a **declared
boundary constant, not a tuning knob**: six ticks is 300 ms at 20 TPS — roughly
four times the worst disclosed live full-brain per-tick compute time (p50
24.4 ms / p99 32.0 ms / max 75.6 ms at 8 threads, per the D-051 benchmark row
in `STATUS.md`), leaving margin for tick jitter on top of the worst recorded
spike — while still failing closed within a third of a second of genuine
starvation. It was chosen from those recorded quantities, not adjusted against
any observation of which behavior "looks better".

Scope of the grace, precisely:

- **Applied hold only.** Once a frame has applied, the frameless hold runs
  through `validThroughTick + HOLD_GRACE_TICKS`. During grace ticks the fly is
  fail-neutral — rejected-ACTIVE with neutral motors — exactly as within the
  validity window itself; nothing is actuated and nothing latches.
- **The re-armed post-RESET hold carries the same grace**, because
  `resetActuationWindow` expresses that hold through the same applied anchor
  (`effectiveTick + FIRST_ACTION_DEADLINE_TICKS`). A reset is a warm-path
  intervention mid-session under the same live-load burst risk, so this is
  intended, and the session still fails closed within
  `FIRST_ACTION_DEADLINE_TICKS + HOLD_GRACE_TICKS` ticks of a reset that
  produces no applicable action. The pre-reset frame retirement window
  (`retiresAtResetBoundary`, `<= FIRST_ACTION_DEADLINE_TICKS`) is untouched.
- **The cold-start first-action deadline is byte-identical.** Before any applied
  action the hold remains `firstActiveGameTick + FIRST_ACTION_DEADLINE_TICKS`
  with no grace: the warm path owns cold starts, and a controller that never
  produces a first action still latches within two ticks of activation.
- **`DORMANCY_GAP_TICKS` is byte-identical.** Its bound reasons about
  *validate-call* gaps — `validate()` runs on every entity tick while the fly is
  live, so a gap above the bound can only mean the authority stopped ticking the
  entity — not about *frame* gaps. Grace changes how long the session tolerates
  absent frames; it does not change how often `validate()` is called, so the
  "a live tick-continuous world can never produce g > 3" argument is unaffected.

Decision table, with `v` = last applied frame's `validThroughTick` and `t` =
`currentGameTick`, no applicable frame available:

| Gap | Before this ADR | After this ADR |
|---|---|---|
| `t <= v` (within declared validity) | rejected-ACTIVE, neutral motors | Byte-identical |
| `v < t <= v + 6` (grace window) | `DEADLINE_MISSED` latched | rejected-ACTIVE, neutral motors; a fresh valid frame applies cleanly |
| `t > v + 6` | `DEADLINE_MISSED` latched | `DEADLINE_MISSED` latched |
| Cold start / post-re-anchor, `t - firstActiveGameTick > 2`, no first action | `DEADLINE_MISSED` latched | Byte-identical |
| Post-RESET, frameless past `effectiveTick + 2` | `DEADLINE_MISSED` latched | latches at `effectiveTick + 9` (hold runs through `effectiveTick + 8`: same anchor, plus grace) |

A reserved frame whose target is still ahead of the clock follows the same hold:
it may wait through the grace window, and a lapsed graced hold latches in the
same pass even with such a frame reserved (a future frame still cannot mask
starvation).

## Interaction with ADR 0011 and ADR 0012

- **Grace never suppresses a legitimate re-anchor.** The dormancy gap test runs
  before the hold evaluation on every `validate()` call and is keyed on
  `lastValidatedGameTick`, which grace does not touch. A dormancy gap that opens
  *during* a grace window therefore still re-anchors on resumption.
- **A re-anchor discards any outstanding grace.** `reanchorAfterDormancy`
  returns `lastAppliedValidThroughTick` to its unset sentinel, so the
  post-resume hold is the cold-start branch with no grace: grace cannot leak
  across dormancy, and genuine post-resume starvation latches within
  `FIRST_ACTION_DEADLINE_TICKS` of the re-anchor tick — even on ticks the
  discarded grace window would still have covered. This is pinned by
  `dormancyGapDuringTheGraceWindowStillReanchorsAndDiscardsTheGrace`.
- **A re-anchor never suppresses grace going forward**: the first applied
  post-resume action re-establishes the applied anchor and its grace.
- **ADR 0011 retirement semantics are untouched.** Verified-late frames retire
  and unverifiable frames latch exactly as before; grace only moves the tick at
  which a *persistent absence* of applicable actions latches. Retirements still
  never extend or reset any deadline — the anchors move only on application,
  reset, or re-anchor.

Everything that is not the frameless (or future-reserved) hold evaluation still
latches exactly as before: `IDENTITY_MISMATCH`, `SEQUENCE_ERROR`,
`FUTURE_COMMAND`, `STALE_COMMAND` on unverifiable frames, and `DEADLINE_MISSED`
past the graced hold.

Operator visibility: no status surface prints hold information (the
`neurocraftfly status` line reports bridge state, applied motors, reset state,
sequence bookkeeping, and the ADR 0011/0012 counters), so no surface changes.
A session inside a grace window reports `bridge=ACTIVE` with neutral applied
motors, which is exactly what it is.

## Test timeline adjustments

Five pre-existing `CommandValidatorTest` pins latched inside the new grace
window and had their timelines extended, per the D-056 precedent; one gained an
explanatory pin comment:

- `missingCommandAfterStreamBeginsLatchesDeadlineFault` — latch moved from 103
  to 109 (`v = 102`, grace through 108); both boundary sides of the graced hold
  are now pinned in place.
- `futureCommandCannotHideAnExpiredCurrentTickDeadline` — the original arrival
  at 103 falls inside the grace window, where reserving a next-tick frame is the
  intended recovery; the scenario now arrives at 109 targeting 110 and pins that
  a current-tick-skipping frame cannot mask the lapsed graced hold.
- `queuedFrameExpiredBeyondHoldRetiresAndDeadlineStillLatches` — rebuilt so the
  queued frame is reserved late in the grace window and expires across a legal
  3-tick validate skip, preserving the original same-pass retire-and-latch pin
  past the graced hold.
- `preResetActionAfterTheResetWindowRetiresButLapsedDeadlineStillLatches` — the
  re-armed reset hold carries grace (through `effectiveTick + 8` = 111), so the
  late pre-reset frame now arrives at 112; retire-and-latch in the same pass is
  preserved.
- `gapAtTheDormancyBoundaryDoesNotReanchorAndStillLatchesDeadline` — a single
  boundary-sized gap now lands inside the grace window, so three consecutive
  boundary gaps are chained: none re-anchors, and the third lands past the
  graced hold and latches.
- `missingFirstActionAfterTwoActiveTicksLatchesDeadlineFault` — unchanged
  behavior, now carries the explicit pin that the cold-start path is
  byte-identical under this ADR.

## Consequences

- A live-load burst of up to six frameless ticks after the last applied
  validity window is now survivable fail-neutral; the fly hovers on neutral
  motors and resumes on the next verified frame. Genuine starvation still fails
  closed, one declared constant later.
- The causal record is not weakened: nothing is applied during grace ticks, so
  every applied action retains its full verified provenance chain, and the
  ADR 0011/0012 disclosure counters are unchanged in meaning.
- The change touches `mod/src` control code, so any evidence bundle that
  enumerates bridge/controller sources is superseded for future candidate runs
  and must be regenerated on the fixed tree, as with ADR 0006, 0011, and 0012.
- The grace decision is validator-scope unit evidence only; a live whole-brain
  escape-profile session observation of a survived burst remains to be
  recorded.
