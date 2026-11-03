# ADR 0011: Retire fully-verified late action frames in the command validator

_Lateness availability semantics, drafted 2026-08-30_

---

| Field | Value |
|---|---|
| **Status** | Implemented; unit-tested at validator scope with a component-scope GameTest |
| **Date** | 2026-08-30 |
| **Authority** | Server-side `CommandValidator` admission and application rules |
| **Precedent** | ADR 0006 / D-025 reset-boundary retirement |
| **Evidence** | Lateness-retirement suite in `CommandValidatorTest` (retirement, hold-rule, counter, and still-latching regressions) plus `verifiedLateFrameRetiresWithoutKillingSessionAndFreshFrameActuates` in `FlyEntityGameTests` |

## Context

The validator's timing rules were designed around fixture profiles that compute an
action in roughly one millisecond. The full-brain profile computes each action in
25–150 ms, the neural service runs its steps inline on its event loop and is
therefore blind to game-tick advance while computing, and Minecraft additionally
runs silent catch-up tick bursts under load. The combination occasionally produces
an action frame that is fully verified — correct identity, exact next command
sequence, present observation receipt, matching recorded tick — but whose target
tick has already passed when the server thread examines it.

`CommandValidator.acceptCandidate` treated that frame as `STALE_COMMAND`, and the
bridge latches every such fault permanently (`NeuralBridge.latchFault`), so a
single late frame terminated the session. This was observed live on 2026-08-30:
326 whole-brain commands applied with real flight, then one stale-ARRIVING frame
latched `STALE_COMMAND` and killed the session. As with the reset-boundary defect
in ADR 0006, this is an availability defect only — no unsafe actuation path
exists, because rejected frames always apply neutral motors.

## Decision

Extend ADR 0006's retirement precedent from the reset boundary to pure lateness.
In `acceptCandidate`, after identity, sequence continuity, reset-boundary
retirement, observation-receipt lookup success, recorded-tick match, non-negative
age, non-future target, neural-step continuity, and target ordering against the
reserved queue have all passed — i.e. the frame is provably the next honest frame
of this session's stream — the three pure-lateness rejections become fail-neutral
retirements instead of latched faults:

1. `age > MAX_COMMAND_AGE_TICKS` (observation older than two ticks),
2. `currentGameTick > validThroughTick` (declared validity window passed),
3. `targetTick < currentGameTick` (target tick already passed at acceptance).

Retirement mirrors the reset-boundary mechanics exactly: the frame is consumed
(`lastCommandSequence` and `lastNeuralStepEnd` advance so the next fresh frame is
not a `SEQUENCE_ERROR`), nothing is reserved, no motors are applied, no fault
latches, and a `latenessRetirements` counter increments.

In the application path, a frame that was valid at acceptance but whose target
passed while it sat reserved (a tick burst) is the same lateness class: it is
retired (consumed, counted, no fault) and the validator examines the next reserved
frame in the same pass, preserving single-application-per-tick semantics. The hold
rule is unchanged and now expressed once: past the validity window of the last
applied action — or, before any application, past the inclusive first-action
deadline — an absent applicable action still latches `DEADLINE_MISSED`. Because
the first-action hold is anchored to the first ACTIVE tick rather than to command
sequences, a retirement never resets or extends the first-action deadline.

Everything that is not pure lateness on a fully verified frame still latches:

- `IDENTITY_MISMATCH` — wrong identity, or a receipt whose recorded tick
  contradicts the frame's claimed observation tick.
- `SEQUENCE_ERROR` — command-sequence gap, neural-step regression, target-order
  violation against reserved frames, or reservation overflow. A late frame with a
  neural-step regression or target-order violation latches `SEQUENCE_ERROR`: the
  retirement gate requires the frame's sole defect to be lateness. (Previously the
  staleness check ran first and such frames latched `STALE_COMMAND`; the fault
  identity changed but the fail-closed outcome did not.)
- `FUTURE_COMMAND` — observation or target ahead of the authoritative clock
  (`targetTick > currentGameTick + 1`). A future frame is a clock inconsistency,
  not lateness.
- `STALE_COMMAND` — missing observation receipt. An absent receipt leaves the
  frame unverifiable, and with `MAX_COMMAND_AGE_TICKS = 2` bounding honest
  lateness, an honest late-but-verifiable frame cannot have aged its receipt out
  of the bounded ledger; only the reset-boundary retirement of ADR 0006, whose
  blocked-actuation window can legitimately outlive receipts, forgives a missing
  receipt.
- `DEADLINE_MISSED` — persistent absence of an applicable action beyond the hold
  window, exactly as before.

Operator visibility: the single-fly `neurocraftfly status` line now ends with
`lateness_retired=N`, a process-lifetime count across validators, so silent
lateness is disclosed rather than absorbed.

## Consequences

- A verified late frame now costs one skipped actuation tick fail-neutral instead
  of terminating the session; deadline enforcement is unchanged, so a stream that
  stops producing applicable actions still fails closed within the same bounds as
  before.
- The causal record is not weakened: retired frames were never applied, so every
  applied action retains its full verified provenance chain, and retirements are
  counted and disclosed via `lateness_retired` in the operator status line.
- Two ADR 0006 boundary pins were deliberately updated: a verified late
  post-reset frame and a verified pre-reset frame arriving after the re-armed
  window now retire instead of latching `STALE_COMMAND` (the latter still fails
  closed in the same pass via the lapsed reset hold, as `DEADLINE_MISSED`). The
  fail-closed pins were repointed at unverifiable frames, which still latch.
- The change touches `mod/src` control code, so any evidence bundle that
  enumerates bridge/controller sources is superseded for future candidate runs
  and must be regenerated on the fixed tree, as with ADR 0006.
- The retirement decision is validator-scope unit evidence plus a component-scope
  GameTest; the real WebSocket bridge path is only exercised by the opt-in
  external bridge lane, and a live whole-brain session observation of a survived
  late frame remains to be recorded.
