# ADR 0018: Cold-start actuation grace and the target-tick window boundary

_Availability at both ends of a session, drafted 2026-08-31_

---

| Field | Value |
|---|---|
| **Status** | Implemented; unit-tested at validator and protocol scope, integration-tested at service scope. **Adversarial review pending.** No live session has been run against this change |
| **Date** | 2026-08-31 |
| **Authority** | `CommandValidator` cold-start hold and `SentObservationWindow` capacity (mod side); `SessionValidator.retire_expired_actions` and the pre-send drop in `NeuralService.process_latest` (service side) |
| **Precedent** | ADR 0006 / D-025 reset-boundary retirement; ADR 0011 lateness retirement; ADR 0012 dormancy re-anchor; ADR 0013 actuation-hold grace |
| **Supersedes** | ADR 0013's "the warm path owns cold starts" (falsified below, on three grounds); ADR 0012:58-61 ("a re-anchored session that produces no applicable action within `FIRST_ACTION_DEADLINE_TICKS` of the re-anchor tick still latches `DEADLINE_MISSED`" — the bound is now `FIRST_ACTION_DEADLINE_TICKS + COLD_START_GRACE_TICKS`). Note ADR 0012 predates ADR 0013 and never mentions grace at all |
| **Evidence** | Four new `CommandValidatorTest` pins plus three `SentObservationWindowTest` pins, two new Python pins, six adjusted timelines, and **four executed negative controls** (see *Regression pins*). Validator lane 40 → 44 methods; mod unit lane 92 → 99 tests; Python 1,132 → 1,134 |

## Context

The operator's live demo died three times at cold start on 2026-08-31, and a
fourth session — the last known-good one,
`interactive-20260831T164331Z-9295f2f3` — died after 2,538 successfully applied
commands. Two independent defects account for all four.

Both are availability defects only. Nothing unsafe was ever actuated: a rejected
frame applies neutral motors, and every change below leaves that property exactly
where it was.

### The correction that matters most: the budget is one tick, not two

D-053 and ADR 0013 both reason about a "100 ms first-action deadline". **That
number is wrong, and it is the single most important correction to the mental
model this lane carries.** The real budget for a first action is **one game tick,
50 ms**, and it is enforced in three places that agree with each other:

| Where | Rule |
|---|---|
| `full_brain_controller.py:1823` | `target_tick = observation.game_tick + 1`, unconditionally |
| `server.py:197-198` (`_validate_generated_action`) | refuses any action whose `target_tick != observation.game_tick + 1` |
| `CommandValidator.java` (application branch) | applies **only** when `next.targetTick() == currentGameTick` |

`FIRST_ACTION_DEADLINE_TICKS = 2` is therefore not two chances at the first
action. `FlyControllerRuntime.control()` offers the observation for tick `T`
(`bridge.offerObservation`) and polls the action window (`bridge.pollAction`)
**microseconds apart in the same call**, so the action derived from tick `T`'s
observation targets `T+1` and must be back before the poll one tick later. Miss
that and the frame is late by construction; it can then only retire fail-neutral
(ADR 0011), never apply. The two ticks of deadline buy two *attempts*, each of
which is a fresh 50 ms race.

First-action latency in the failing sessions was bracketed at **(100, ~150] ms** — derived from the operator-observed live logs of 2026-09-01, which are gitignored local artifacts and not committed evidence, exactly like the session ids and the 2,538-command figure quoted below —
against a disclosed warm **maximum of 75.6 ms** (D-051, 8 threads). Both attempts
lose.

### Defect A — the cold start gets two attempts where steady state gets eight

Under ADR 0013 the applied-anchor hold runs through `validThrough +
HOLD_GRACE_TICKS`. With `validThrough = t + 1` for a frame applied at `t`, the
session latches at `t + 8`. The cold branch had no grace at all and latched at
`firstActive + 3`. **A 2.7× asymmetry, applied at the moment the system is
provably least warm** — the JVM has just finished minutes of Minecraft boot, the
`asyncio.to_thread` executor has not been created, and the first `engine.step`
calls have never run.

**ADR 0013's justifying sentence — "the warm path owns cold starts" — is
falsified on three independent grounds:**

1. **The warm path does not cover the work that dominates a cold first action.**
   `FullBrainController._warm_locked` warms `engine.kernel.matvec` plus
   `np.tanh`, once per pass. It does not warm the five `engine.step` calls that a
   tick actually performs, nor their `_verify_step`, nor the two SHA-256 activity
   digests each of them computes, nor the encoder/decoder chain, nor the first
   `asyncio.to_thread` executor creation.
2. **Even the part it does cover is not the whole cost.** One warm pass is a
   single `matvec + tanh`; a tick performs five full `engine.step` calls, so a
   perfectly warm kernel still leaves a substantial share of the per-tick cost
   cold on the first pass. **The exact share is not re-measured here** — the
   diagnosis that opened this lane put it near 63% of p50, and that figure has
   no provenance anywhere in this repository, so it is recorded as unverified
   and nothing below depends on it.
3. **Decisively, and independently of 1 and 2: no amount of warming makes a
   disclosed 75.6 ms maximum fit inside a 50 ms budget.** The claim is
   arithmetically unavailable regardless of how good the warm path is.

**Ground 3 is the one that settles it, and it is self-contained**: it needs only
the disclosed 75.6 ms and the 50 ms tick, both of which are recorded quantities.
Grounds 1 and 2 explain why the observed latency was worse still. Ground 1 was
verified by reading `_warm_locked`, which warms `kernel.matvec` and `np.tanh`
over locally allocated scratch and asserts it never advances `step_index`.

### Defect B — the validated-command window backs up and fails closed

`SessionValidator.retire_expired_actions` popped the window head only while
`pending[0].valid_through_tick < current_tick`, against a hard-capped
`max_command_window = 2`. When the authority retires a frame fail-neutral
(ADR 0011, or the ADR 0006 reset boundary) it produces **no receipt**, so nothing
else frees the slot. Two un-receipted commands then wedge the window and the
third raises `SequenceError("validated command window is full")` — the service
fails closed and the run ends.

The arithmetic is exact and reproduces on demand. Actions arrive one per
observation with `target = obs + 1`, `validThrough = obs + 2`:

| Observation | Retired under `valid_through` | Window after |
|---|---|---|
| 100 | — | `[a100]` |
| 101 | `validThrough 102 < 101`? no | `[a100, a101]` |
| 102 | `validThrough 102 < 102`? **no** | full → `accept_action` raises |

Under a `target_tick` boundary the same trace retires `a100` at observation 102
(`target 101 < 102`) and stays at two slots indefinitely.

**This is the defect that matters most for a long demo run.** It killed the
2,538-command session, and it killed one of the three cold-start sessions *after*
that session had already recovered from defect A.

## Decision

### A. `COLD_START_GRACE_TICKS = 5`

Declared, and **derived rather than chosen**. Steady state latches eight ticks
after the last applied tick. For the cold branch to latch eight ticks after
activation:

```
hold = firstActive + FIRST_ACTION_DEADLINE_TICKS + G = firstActive + 7   ⟹   G = 5
```

The declared invariant is: **a session tolerates the same 8-tick (400 ms)
frameless gap whether or not an action has yet applied.** Seven application
attempts instead of two.

**The naive variant is explicitly refused.** Raising `FIRST_ACTION_DEADLINE_TICKS`
would have produced the same latch tick with a one-line diff, and it is wrong:
that constant is aliased to `MAX_COMMAND_AGE_TICKS`
(`FIRST_ACTION_DEADLINE_TICKS = MAX_COMMAND_AGE_TICKS`), so raising it would
**widen the freshness bound — permitting the application of a genuinely stale
frame — and shift `DORMANCY_GAP_TICKS` with it**. That is a real safety
weakening dressed as a one-character edit. The grace is an additive term on the
hold and on nothing else.

**Applied at both cold-start hold evaluations, not one.** This is a correction to
the lane's own plan, recorded rather than quietly absorbed. The plan specified
the grace in the unset-applied-anchor branch of `actuationHoldThroughTick()`
only. Implementing exactly that and running the suite showed
`missingFirstActionAfterTwoActiveTicksLatchesDeadlineFault` and
`lateFreshActionCannotMaskMissedFirstActionDeadline` **still passing** — because
a *second* cold-start guard sits earlier in `validateChecked`:

```java
if (lastCommandSequence < 0 && currentGameTick > coldStartHoldThroughTick()) {
    return reject(ControllerFault.DEADLINE_MISSED);
}
```

It runs **before the candidate loop**, so whenever no frame has arrived at all —
which is precisely the observed cold-start failure — it decides the session at
`firstActive + 3` and the hold rule never sees a frame. Gracing only the hold
would have left the reported failure mode untouched and made the ADR's declared
invariant false. Both sites now read the same private
`coldStartHoldThroughTick()` helper, so they cannot drift apart.

### B. `target_tick` is the applicability boundary on the service side

Two one-expression changes:

- `SessionValidator.retire_expired_actions` retires the head while
  `pending[0].target_tick < current_tick`.
- `NeuralService.process_latest` drops a freshly computed action when
  `newest_game_tick > action.target_tick`.

**Strictly tighter, verified rather than asserted.** `Action.__post_init__`
refuses any action with `valid_through_tick < target_tick`, so
`target_tick <= valid_through_tick` holds for every action that exists at all.
Therefore `newest > valid_through ⟹ newest > target`: the new rule retires and
drops a **superset**, one tick sooner. It can never keep a frame the old rule
dropped, so it cannot weaken anything by construction.

It is also the *correct* question. The authority applies on exactly `target_tick`;
once the world is past it the frame can only be retired. The
`valid_through_tick` slack was dead window occupancy — real for
`accept_receipt`'s expiry check, which is untouched, but meaningless for
applicability.

The ordering argument the old docstring rested on survives intact and gets
tighter: within tick `T`, `FlyControllerRuntime.control()` sends observation `T`
*before* it polls and applies, so a receipt for a frame targeting `T` is written
during tick `T`, ahead of observation `T+1`. On an ordered transport the receipt
is always parsed before the observation that would retire its action.

### C. Companion hardening: the observation ledger grows 3 → 8

`SentObservationWindow.CAPACITY` becomes
`FIRST_ACTION_DEADLINE_TICKS + COLD_START_GRACE_TICKS + 1`, derived from the
constants in code so it cannot drift.

Without this, the grace would convert a previously-**masked** hard fault into a
reachable one. A frame whose observation receipt has aged out of the ledger is
*unverifiable* and latches `STALE_COMMAND` — no lateness retirement is available
to it. Before this ADR a session was dead at `firstActive + 3` anyway, so frames
older than three ticks were unreachable. With seven graced ticks they are
reachable, and a three-entry ledger would have met them with a latch.

**The trade is declared, not discovered.** A replayed or very late frame whose
observation was sent three to seven ticks ago (the bound is in ledger *entries*, so it spans more wall-clock ticks whenever observations coalesce) is now *verified* and then refused
fail-neutral by the unchanged `age > MAX_COMMAND_AGE_TICKS` rule, instead of
latching `STALE_COMMAND`. **The loss is adversarial *signal*, not integrity.**
Application remains impossible — the age rule retires it without reserving
anything — and an actual replay still latches `SEQUENCE_ERROR` on the unchanged
`commandSequence == lastCommandSequence + 1` continuity check, while a forged
observation tick still latches `IDENTITY_MISMATCH` on the receipt-tick
comparison.

## Every admission rule is byte-identical

The grace moves exactly one thing: **the tick at which a persistent *absence* of
applicable actions latches `DEADLINE_MISSED`.** Nothing about which frames may be
admitted or applied changes.

This is demonstrated structurally, not asserted: `acceptCandidate` — the method
that contains every admission rule — has **no executable statement changed by this lane** (one inline comment inside its `recordedTick.isEmpty()` branch is reworded; nothing else). `git diff`
over `CommandValidator.java` shows changes to the constant block, the
first-action guard, `actuationHoldThroughTick()`, the new helper, and three
javadocs, plus that one comment; `acceptCandidate`'s executable body is unmodified. So all of the following are
byte-identical:

| Rule | Site | Status |
|---|---|---|
| `age <= MAX_COMMAND_AGE_TICKS` | `acceptCandidate` | untouched |
| Receipt verification (`recordedTick.isEmpty()` → latch) | `acceptCandidate` | untouched |
| Receipt-tick binding (`recordedTick != observationTick` → `IDENTITY_MISMATCH`) | `acceptCandidate` | untouched |
| Sequence continuity (`commandSequence == last + 1`) | `acceptCandidate` | untouched |
| Neural-step monotonicity (`neuralStepStart < lastNeuralStepEnd`) | `acceptCandidate` | untouched |
| `target ∈ {current, current + 1}` | `acceptCandidate` + application branch | untouched |
| `MAX_COMMAND_AGE_TICKS`, `FIRST_ACTION_DEADLINE_TICKS`, `DORMANCY_GAP_TICKS`, `MAX_PENDING_COMMANDS` | constant block | unmodified values |

Behaviourally it is pinned as well as structurally, by
`gracedColdStartLeavesEveryAdmissionRuleUntouched`, which drives a session five
ticks into a graced window — a window that did not exist before this ADR — and
presents four malformed frames plus one honest-but-stale frame, asserting that
each produces the identical fault (`STALE_COMMAND`, `IDENTITY_MISMATCH`,
`FUTURE_COMMAND`, `SEQUENCE_ERROR`) or the identical fail-neutral retirement it
produces anywhere else.

Every graced tick is **rejected-ACTIVE with neutral motors** — the identical
state ticks `T0+1` and `T0+2` already occupied — asserted tick by tick in
`coldStartFramelessGapToTheGracedHoldStillAppliesTheFirstFrame`.

## Residual risk: the dormancy interaction, declared not discovered

`reanchorAfterDormancy` clears the applied anchor, so a re-anchored session lands
on the cold branch — **which now carries a grace of its own.** The consequence
changes direction, and ADR 0012's prose on this point is superseded:

| | Before | After |
|---|---|---|
| Applied at 101, grace through 108 | latch 109 | latch 109 |
| Dormancy re-anchor at 105 | latch **108** (re-anchor shortens) | latch **113** (re-anchor lengthens) |

ADR 0012 and ADR 0013 both state in prose that discarding the grace makes the
post-resume deadline *tighter* — "genuine post-resume starvation latches within
`FIRST_ACTION_DEADLINE_TICKS` of the re-anchor tick, even on ticks the discarded
grace window would still have covered". **That is no longer true.** A re-anchor
still discards the applied grace and still re-arms from the resume tick, but the
re-armed bound is now `+7` rather than `+2`, which in the pinned scenario is four
ticks *later* than the grace it discarded.

This is accepted, on the same reasoning that motivates the whole lane: a
post-dormancy resume is a cold start — chunks have just reloaded and the entity
has just started ticking again — and it deserves the same budget as any other
cold start. `dormancyGapDuringTheGraceWindowStillReanchorsAndDiscardsTheGrace`
is retained and rewritten to record this honestly rather than to restate
ADR 0013's claim; it remains genuinely distinguishing, because a re-anchor that
had *kept* the applied anchor would latch at 109, not 113.

Two properties that do **not** change, and are still pinned: a dormancy gap
opening inside a grace window still re-anchors (the gap test is keyed on
`lastValidatedGameTick`, which no grace touches), and the re-anchor still clears
the applied anchor.

## Regression pins and negative controls

Validator lane **40 → 44** test methods; mod unit lane **92 → 99** tests
(`SentObservationWindowTest` is new, three methods); Python **1,132 → 1,134** as reported by `make verify`'s pytest stage
(`1,134 passed, 1 skipped`); a standalone `pytest -q tests/` afterwards reports
`1,135 passed`, because `tests/compliance/test_build_jvm_inventory.py::test_real_emitted_graph_satisfies_the_license_policy`
is gated on `mod/build/reports/jvm-dependency-graph.json`, which `make verify` writes only in its
later `fabric-sbom` stage — so it skips inside verify and runs in any standalone run after it.
Collection is identical either way (`pyproject.toml` sets `testpaths = ["tests"]`); the
attributable delta is +2.

**New pins**

| Pin | Binds |
|---|---|
| `coldStartFramelessGapToTheGracedHoldStillAppliesTheFirstFrame` | every graced tick is rejected-ACTIVE/neutral; the first frame applies at the inclusive boundary (107) |
| `coldStartToleratesExactlyTheSameFramelessGapAsTheAppliedHold` | the declared invariant as a **symmetry** — a cold and a warm validator driven from one loop must both survive anchor+7 and latch at anchor+8 |
| `lateFirstFrameRetiresAndTheServiceCatchingUpInsideTheGraceStillFlies` | the live scenario end to end: a late first frame retires fail-neutral, and a frame the warming service gets back in time applies |
| `gracedColdStartLeavesEveryAdmissionRuleUntouched` | four admission faults plus one fail-neutral retirement, all evaluated inside a graced window |
| `SentObservationWindowTest` (3 methods) | ledger covers the full graced span; still evicts past it; still refuses non-increasing and negative receipts |
| `test_a_long_run_of_unreceipted_commands_never_wedges_the_window` | twelve consecutive receipt-less commands stay inside the two-slot window |
| `test_action_whose_target_the_world_passed_is_dropped_before_acceptance` | the pre-send drop fires at `target_tick`, one tick before `valid_through_tick` |

**Four executed negative controls.** Each reverts one half, runs the suite,
confirms exactly the intended tests fail, and restores byte-identically —
verified by SHA-256, not by inspection.

1. **`COLD_START_GRACE_TICKS` 5 → 0** (behaviourally exact: the cold hold returns
   to `firstActive + 2` and the derived ledger capacity returns to 3).
   **Exactly 9 of 99 failed**, all cold-start timelines:
   `coldStartFramelessGapToTheGracedHoldStillAppliesTheFirstFrame`,
   `coldStartToleratesExactlyTheSameFramelessGapAsTheAppliedHold`,
   `lateFirstFrameRetiresAndTheServiceCatchingUpInsideTheGraceStillFlies`,
   `gracedColdStartLeavesEveryAdmissionRuleUntouched`,
   `missingFirstActionAfterTwoActiveTicksLatchesDeadlineFault`,
   `lateFreshActionCannotMaskMissedFirstActionDeadline`,
   `retirementDoesNotResetTheFirstActionDeadline`,
   `reanchorRearmsTheFirstActionDeadlineAgainstPostResumeStarvation`,
   `dormancyGapDuringTheGraceWindowStillReanchorsAndDiscardsTheGrace`.
   `CommandValidator.java` restored to `df1a5d11…5408a43`.

   **A finding this control produced, disclosed rather than filed away.** On its
   first run only 8 failed: `lateFreshActionCannotMaskMissedFirstActionDeadline`
   passed under both settings, because its adjusted timeline drove eight ticks
   without asserting that the session was still alive on any of them, so an
   early latch was invisible to it. It now asserts `ACTIVE` on every tick to the
   boundary and is distinguishing. **The control found a weak test, which is what
   controls are for**; the 9-failure figure above is the re-run after the fix.

2. **`SentObservationWindow.CAPACITY` pinned back to a literal 3 with the grace
   left at 5** — isolating hardening C from grace A, which control 1 cannot do
   because the capacity is derived from the same constant. **Exactly 2 failed**,
   both `SentObservationWindowTest` methods; all 97 others passed. Restored to
   `efebf19b…b98fcc`.

3. **`sequence.py` retirement boundary reverted to `valid_through_tick`.**
   **Exactly 4 failed**:
   `test_a_long_run_of_unreceipted_commands_never_wedges_the_window`,
   `test_unreceipted_expired_actions_do_not_wedge_the_command_window`,
   `test_retirement_only_drops_what_the_authority_can_no_longer_apply`,
   `test_a_receipt_for_a_retired_action_still_fails_closed`. The first failed
   with the live fault verbatim —
   `neurocraft_protocol.errors.SequenceError: validated command window is full`
   — which is the direct evidence that this pin reproduces the defect rather
   than merely describing it. Restored to `4997c0b2…0286cd`.

4. **`server.py` drop boundary reverted to `valid_through_tick`.** **Exactly 1
   failed**: `test_action_whose_target_the_world_passed_is_dropped_before_acceptance`.
   Restored to `59b270ff…3b2333`.

**Six adjusted timelines**, each carrying its justification comment inline, per
the D-056 / ADR 0013 precedent. **None crosses `DORMANCY_GAP_TICKS`**: every
extended timeline advances one tick per `validate()` call, and the two that
deliberately contain a gap (`reanchorRearms…` at 101→106,
`dormancyGapDuringTheGrace…` at 101→105) assert `dormancyReanchors() == 1`
exactly once, with every subsequent tick consecutive. Timelines with no intended
re-anchor now assert `dormancyReanchors() == 0` explicitly.

- `missingFirstActionAfterTwoActiveTicksLatchesDeadlineFault` — latch 103 → 108.
- `lateFreshActionCannotMaskMissedFirstActionDeadline` — arrival 103 → 108;
  strengthened per control 1.
- `retirementDoesNotResetTheFirstActionDeadline` — latch 103 → 108. The pin
  itself is *why* the timeline may move: a lateness retirement re-anchors
  nothing, so the deadline is still measured from activation.
- `reanchorRearmsTheFirstActionDeadlineAgainstPostResumeStarvation` — latch
  109 → 114.
- `dormancyGapDuringTheGraceWindowStillReanchorsAndDiscardsTheGrace` — latch
  108 → 113, and rewritten to record the reversed direction (see above).
- `test_deblinding.py::test_receiptless_accepts_retire_and_drop_instead_of_filling_the_window`
  — the mid-step observation moved 102 → 101 so action 0 is deterministically
  not dropped regardless of send/release interleaving, and its synchronisation
  barrier now waits for the retirement *at tick 105* by name. Under the tighter
  rule an unrelated earlier retirement satisfied the bare substring match and
  released the gate early, which would have made the test **flaky rather than
  failing** — found by execution, and worth recording as the more dangerous of
  the two outcomes.

Three further Python pins had their expected retirement counts corrected (1 → 2)
because retirement now fires a tick sooner. One of them,
`test_retirement_only_drops_what_the_authority_can_no_longer_apply`, carried a
comment asserting *"At tick 102 command 0 is still applicable
(valid_through_tick == 102)"* — **which was never true**: the authority applies
only at `target_tick`, which was 101. The test's name was always right and its
comment was always wrong; it now probes the real boundary at 101 through a
direct `retire_expired_actions` call and pins both sides.

## Consequences

- A cold session survives seven frameless ticks instead of two, hovering on
  neutral motors, and applies the first frame the service delivers in time.
  Genuine starvation still fails closed, one declared constant later.
- A run of receipt-less commands can no longer wedge the validated-command
  window. This is the change that matters for a long demo run.
- **No live claim is made.** The three failed cold starts and the
  2,538-command session are the evidence *for* the diagnosis; nothing here is
  evidence that the live demo now works. That is an observation nobody has made.
  It becomes claimable when a live session is run and recorded, and not before.
- The causal record is not weakened: nothing is applied during graced ticks, so
  every applied action retains its full verified provenance chain, and the
  ADR 0011/0012 disclosure counters are unchanged in meaning.
- **ADR 0013's cold-start clause is superseded** — its "the cold-start
  first-action deadline is byte-identical" row and its "the warm path owns cold
  starts" rationale no longer describe the tree. ADR 0013 is **not edited**; it
  is correct as the declaration of the tree it was written on, and this document
  supersedes it forward, per the ADR 0015 → 0017 idiom. Its `HOLD_GRACE_TICKS`
  decision, its applied-hold decision table, and its post-RESET reasoning all
  stand unchanged.
- **One residual guard disagreement, declared rather than discovered.** After
  `resetActuationWindow` with `lastCommandSequence < 0` — reachable, because
  `RESET_SLOT.blocksActuation` returns from `control()` before `pollAction`, so no
  frame need ever have been consumed — the first-action guard takes the cold branch
  and latches at `effectiveTick + 8`, while `actuationHoldThroughTick()` takes the
  applied branch and would latch at `effectiveTick + 9`. ADR 0013's decision-table
  row for that state ("latches at `effectiveTick + 9`") is therefore one tick
  optimistic. The direction is fail-closed, so nothing is weakened, and this lane
  *narrows* a pre-existing disagreement from six ticks (`+3` vs `+9`) to one. Not
  fixed here; recorded so the next reader of that table is not surprised.
- **Two admission rows rest on the diff, not on the behavioural pin.** Neural-step
  monotonicity and the `target > current + 1` bound are covered structurally by the
  unchanged-body argument above; `gracedColdStartLeavesEveryAdmissionRuleUntouched`
  exercises the other five, its FUTURE_COMMAND case tripping on `age < 0` first.
- **ADR 0012's post-resume-tightening prose is superseded** on that clause
  alone; its re-anchor mechanism, its `DORMANCY_GAP_TICKS` derivation, and its
  sequence-identity guarantee are untouched.
- D-053's and ADR 0013's "100 ms" framing of the first-action deadline is
  corrected to one game tick throughout. No recorded measurement moves — the
  D-051 figures (p50 24.4 ms / p99 32.0 ms / max 75.6 ms) are unaffected; what
  changes is the budget they are compared against.
- The change touches `mod/src` control code, so `neurocraft.gametest_code_bundle.v1`
  moves and any evidence bundle enumerating bridge/controller sources is
  superseded for future candidate runs and must be regenerated on the fixed
  tree, exactly as ADR 0006, 0011, 0012, 0013, 0014, 0015, 0016 and 0017
  recorded for their own `mod/src` changes. As D-068 through D-070 already
  recorded, that quantity had drifted from every committed receipt before this
  lane began, so nothing is newly invalidated. **No receipt is edited.**
- **Adversarial review is pending**, and this ADR's status says so. The two most
  review-worthy judgment calls are stated in the open above rather than left to
  be found: the extension of the grace to a second guard site (a deviation from
  the lane's plan, forced by execution), and the reversed dormancy consequence.
