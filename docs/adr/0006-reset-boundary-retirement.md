# ADR 0006: Retire pre-reset actions fail-neutral at the reset boundary

_Reset availability semantics, accepted 2026-08-17_

---

| Field | Value |
|---|---|
| **Status** | Accepted and implemented; unit-tested at validator scope |
| **Date** | 2026-08-17 |
| **Authority** | Server-side `CommandValidator` admission rules |
| **Risk lineage** | `RISKS.md` R-24 availability defect |
| **Evidence** | Named regression `acceptedResetRetiresOlderQueuedActionWithoutFailingClosed` plus three boundary tests in `CommandValidatorTest` |

## Context

While an accepted RESET is staged, `FlyControllerRuntime.control` deliberately
refuses to consume bridge actions, so an action frame produced for the pre-reset
world can survive in the bounded bridge window. On the reset's effective tick the
validator's actuation window is re-armed, the surviving frame is drained, and its
pre-reset observation tick made it stale: the validator latched `STALE_COMMAND`
and the bridge failed closed. The accepted RESET therefore terminated its own
session. This was an availability defect only; no unsafe actuation path existed,
because rejected frames always applied neutral motors.

## Decision

`CommandValidator.resetActuationWindow` records the reset-boundary effective
tick. A candidate frame that passes the unchanged identity and sequence checks
but was observed before the boundary, while the current tick is still inside the
re-armed first-action window, is **retired fail-neutral**: the frame is consumed,
the session command sequence and neural-step bookkeeping advance so the next
fresh frame is not a `SEQUENCE_ERROR`, nothing is reserved for actuation, no
motors from the retired frame are ever applied, and no fault latches. Every rule
outside that boundary window is byte-for-byte unchanged: post-boundary frames
validate normally, genuinely stale post-boundary frames still latch
`STALE_COMMAND`, and a pre-boundary frame arriving after the window closes still
latches `STALE_COMMAND`.

## Consequences

- An accepted RESET no longer terminates its own session when an older queued
  action crosses the boundary; availability is preserved without any hold-last
  or replayed actuation path, keeping AB-05 fail-closed semantics intact.
- The retirement decision is unit-tested validator evidence. Live-lane RESET use
  remains conservative until an integration-lane exercise is recorded; the
  launcher guidance in `docs/RUNBOOK.md` reflects this scope.
- The change touches `mod/src` control code, so evidence bundles that enumerate
  bridge/controller sources (`EXTERNAL-BRIDGE-02`) are superseded for any future
  candidate run and must be regenerated on the fixed tree before G3 promotion
  uses a current-tree candidate.
