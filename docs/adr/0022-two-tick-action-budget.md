# ADR 0022: A two-tick action budget

_Control-loop timing, 2026-09-04_


---

| Field | Value |
|---|---|
| **Status** | **Implemented on `main` at `04bd8d7` (2026-09-04); not yet run live.** Decided after the first four live sessions of the program lane (`make live-demo-program`, 2026-09-03) died within about two seconds of engaging, two of them on `DEADLINE_MISSED` after 7- and 13-tick bursts of late frames |
| **Date** | 2026-09-04 |
| **Authority** | `neurocraft_protocol.ACTION_TARGET_OFFSET_TICKS` (Python) and its Java twin; every command-frame producer (`services/neural-engine/.../controller.py`, `server.py` `_neutral_action`, `FullBrainController.step`, `ProgramSelectingController.step`); `_validate_generated_action`; the mod's `CommandValidator` acceptance and its apply rule; the cross-language goldens |
| **Precedent** | ADR 0018 (found the budget was one tick and made `target_tick` the retirement boundary), ADR 0013 (reasoned from a 100 ms first-action deadline), ADR 0011/0012 (fail-neutral retirement family), D-051 (the full-brain step benchmark: p50 24.4 ms / p99 32.0 / max 75.6 ms at 8 kernel threads, on a quiet machine), D-079 (PAUSE/RESUME acknowledgements engage at `effective_tick + 1` — observation ticks, untouched here), D-080 (the service never withholds a computed action) |
| **Supersedes** | ADR 0018's finding "the budget is one tick, not two" as a *design*: the one-tick budget is retained as a fact about what the stack did, and replaced as the rule the stack follows. ADR 0013's "100 ms first-action deadline" premise becomes true by construction rather than by accident. `target_tick` remains the retirement boundary on both ends (ADR 0018 decision B, as amended by D-080) |
| **Evidence** | The measurement in `STATUS.md` (lane L record): action latency 43–75 ms on the mod's first eight samples in every live session, surviving and dying alike; offline on the operator's machine under realistic load the raw `FullBrainController.step` is p50 38.9 / p90 41.5 / p99 56.7 / max 62.9 ms over 300 steps, and the program wrapper adds +0.13 ms; the surviving raw-lane session (2026-09-02) retired 25 frames in 4,200 on a faster day. Tests: Python 1,395 → 1,400 passed (35 skipped); mod unit tests 163 → 167; server GameTests 76 with the differential fixture reproducing; goldens regenerated in both languages. `make verify` (all stages, the historicized pilot stage included), `make demo` (`status: verified`, 12 records), and `make program-authority-gametest` green on the merged tree; Python 1,395 → 1,402, mod unit tests 163 → 169, 76 server GameTests with the differential fixture reproducing. **No live session under the two-tick budget yet** |

## The claim this ADR is allowed to make

**A command frame computed from observation `T` is applied at tick `T + 2`, giving the service
100 ms rather than 50 ms to compute, encode, and deliver it, at the cost of one tick of control
lag.** That is all. It does not make the fly react faster or better; it makes the session survive
the laptop it runs on. The one-tick figure in ADR 0018 was a correct measurement of what the
stack did; it was never a requirement, and D-051's numbers that made it look survivable were
taken on a quiet machine.

## Context

The full-brain step at 8 kernel threads costs about 39 ms at the median on the operator's M4 Max
while the Minecraft client and server run beside it, and its tail reaches 57–63 ms. Encoding,
the loopback WebSocket, JVM decoding, and the mod's once-per-tick poll take the rest of a 50 ms
tick. Under a one-tick budget roughly half of all frames therefore arrive after the tick they
target; the mod retires each fail-neutral, and a burst of about eight consecutive misses latches
`DEADLINE_MISSED`. The program lane's first live sessions showed exactly that (bursts of 7 and 13),
and the surviving raw-lane session of the night before survived only because its worst burst was
5. The program wrapper is not the cause (+0.13 ms); the machine is.

## Decision

1. **One declared constant per language**, `ACTION_TARGET_OFFSET_TICKS = 2`. Every producer sets
   `target_tick = observation.game_tick + 2` and `valid_through_tick = observation.game_tick + 2`
   (already the protocol's maximum), including the paused-service neutral action.
2. **The frame rule stays permissive** (`observation < target <= valid_through`), so every committed
   trace and golden still parses; the offset is enforced where frames are *accepted*: the service's
   `_validate_generated_action` and `SessionValidator`, and the mod's `CommandValidator`.
3. **The mod still applies a frame only on its `target_tick`.** A frame that arrives at `T + 1` waits
   one tick in the validated window and applies at `T + 2`; one that arrives at `T + 3` is retired
   fail-neutral exactly as today. `MAX_COMMAND_AGE_TICKS = 2` is unchanged and is met at the
   boundary; the cold-start grace and both observation ledgers are unchanged, and the hold
   grace moves by exactly the added tick (item 6).
4. **PAUSE / RESUME effective ticks are observation ticks and are untouched** (D-079); RESET is
   stamped with the neutral action's `target_tick`, so its bounds move with the offset: the service's effect window and `StagedReset`'s
   accepted delay become `{issued + 2, issued + 3}` (were `{issued + 1, issued + 2}`), the
   acknowledgement frames widen permissively to a delay of 1–3, and `StagedReset`'s two-tick
   no-acknowledgement deadline is deliberately unchanged because it bounds the validation gap
   that `DORMANCY_GAP_TICKS = 3` is calibrated against.
5. **The service's observation ledger grows from three entries to four
   (`ACTION_TARGET_OFFSET_TICKS + 2`).** The review found that with three the on-time boundary sat
   exactly on the ledger's cliff: a step outrun by two observations was on time and by three the
   service lost the observation's binding and the session **failed closed** on `STALE` — not a
   fail-neutral retirement, as an earlier comment claimed, because the service accepts its own
   frames at age zero and its target and age rules never fire there. With four entries a frame one
   tick late is sent and the mod retires it fail-neutral, which is what D-080 intends; a step outrun
   by four observations still ends the session, and the comments now say so.
6. **Derived constants move with the offset, none by tuning.** The actuation hold grace becomes
   7 ticks (was 6) because the applied anchor's `valid_through` now equals its target, so the
   grace absorbs exactly the one added tick and the latch tick — ADR 0018's eight-tick invariant —
   does not move. The service's command window and the trace recorder's pending capacity grow from 2 to
   3. The mod's bounded action window grows from 2 to **4** (`ACTION_TARGET_OFFSET_TICKS + 2`): a
   real overflow under a staged RESET in the probe forced it past 2, and the review's timeline at
   the measured 43–75 ms latencies — an operator RESET blocks polling for up to two ticks while
   frames from the observation before the reset, the reset tick, and the next two ticks can all
   land — showed that 3 dies on an operator reset under live jitter. Four absorbs that span plus one
   straggler; a two-tick-late straggler beyond it is retired, not queued. The operator transaction of D-079 engages at `effective_tick + 2`
   (was `+ 1`): with `+ 1` the probe showed the program selected and preempted on the same tick,
   because an in-flight driving command still governed it and ADR 0020 §3.3 lets a program own
   only a neutral tick — D-079's relation, engage on the first tick the paused service's neutral
   command governs, is intact; its number is not.
7. **Goldens are regenerated** in both languages; they are test vectors, not receipts. The
   `EXTERNAL-BRIDGE-*` receipts keep their one-tick timings as history.

8. **Two consequences the arithmetic forces, disclosed rather than hidden.** Cold-start
   application attempts drop from seven to six (the first frame now targets `T0 + 2` while the cold
   hold still ends at `T0 + 7`; steady state keeps seven, post-RESET keeps eight; the cold-start
   grace is not moved because the pinned eight-tick symmetry matters more than one attempt). The
   post-RESET guard disagreement ADR 0018 narrowed to one tick widens to two (re-armed anchor
   `E + 2` plus a seven-tick grace holds to `E + 9` and latches at `E + 10`, while the cold guard
   still latches at `E + 8`) — in the fail-closed direction.
9. **The `DORMANCY_GAP_TICKS = 3` calibration holds** because operator commands run after the level
   tick: the last validation at `I`, blocked `I + 1 .. I + 2`, resumed at `I + 3`. A RESET staged
   before its issue tick's entity tick would open a gap of four and re-anchor spuriously, in the
   fail-closed direction.
10. **The frozen fixture-physics pilot's same-environment re-execution is historicized**, as its
   implementation-hash pin was at `539d182`: the v3 result core last reproduced bit-for-bit at
   tree `a89d236` in the recorded environment; the platform-neutral byte pins and the
   `protocol_sha256` binding stay; the rebuilt core is no longer compared against the working
   tree, because the only leaves that move are the 100 action- and trace-stream digests that carry
   the envelope, and the result is write-once. **The loss, stated plainly:** after this, nothing
   bit-pins the fixture stack's numbers on any platform (Linux never did); a regenerable engineering
   baseline, disclosed on regeneration like the mod's differential fixture, is the honest
   replacement if one is wanted. `RISKS.md` R-27's control statement changes with it.

## What moves and what does not

- Rotates: `replay_implementation_sha256` (harness traces; none committed) and the
  `neurocraft.gametest_code_bundle.v1` `code_hash` (wholesale `mod/src`).
- Unchanged: every calibration, `RunIdentity`, `observation_schema_hash`, the differential fixture
  (no admitted command), the D-079 timeline (verified by the real-timing probe).

## Consequences

- Control lag grows by exactly 50 ms. For a fly whose body programs run on 24-tick scripts this is
  below anything a viewer can see; it is stated here so nobody discovers it later.
- Sustained latency above 100 ms still ends the session on the mod, and a compute burst that lets
  four observations pass during one step still fails closed on the service. Neither is hidden.
- The mod's validator keeps clock-bound rules rather than a strict `target == observation + 2`
  rule: the offset is enforced at the producers, in the service validator, and at the mod's wire
  boundary, which covers every frame that crosses the socket, and the 85 validator test call sites
  and 49 hand-reasoned timelines that are ADR 0006–0018's evidence keep their one-tick shape
  (a follow-up sweep is filed).
- A further lever exists and is not taken here: the kernel runs 8 threads by default on a 12
  performance-core machine; more threads would cut compute without rotating any identity
  (thread count is declared not to affect results). Measure under realistic load before using it.
- The honest test is lane L: a live session under this budget. Until it runs, the only claim is
  the arithmetic above and the offline measurement.
