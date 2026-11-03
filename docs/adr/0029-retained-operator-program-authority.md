# ADR 0029: retain acknowledged operator authority between program selections

Status: implemented, independently reviewed and focused-tested; live acceptance pending.
Date: 2026-09-04

## Context

Every operator program request previously sent PAUSE, including when the service
was already paused. That second request was rejected with `INVALID_STATE`.
Releasing to neural CRUISE between LAND, WALK and FEED could also lose the grounded
body precondition. A program selection needs the existing authority, not another
authority handover.

## Decision

Keep ADR 0020's accepted PAUSE transaction for the first operator selection and
accepted RESUME for the final handback. After the initial transaction applies,
retain a grant bound to the exact entity object, level, full configured identity,
bridge and PAUSE control sequence. It is usable only while the bridge is ACTIVE,
has no pending terminal fault, and the body still has SCRIPTED authority.

A subsequent request stages a local selection for exactly the next authoritative
entity tick. Recheck the grant and all transaction exclusions at that boundary.
The selection sends no control RPC and is never described as a new PAUSE ACK.
Normal admitted paused-neutral commands and physical body preconditions still
govern motor execution. A selected program is not proof that its motors applied.

Reset, comparison handover, lifecycle replacement and authority loss invalidate
the grant. Pending local selections exclude other control/reset/comparison/trial
transactions; removal, missed ticks and invalidated grants cancel pending work.
The existing lower-level control API also invalidates a grant when it accepts a
request because this runtime does not route that API's eventual ACK.

## Verification boundary

The implementation passed 27 focused JUnit tests and 48 selected server GameTests
in the author's isolated worktree and received independent source review.
The tests exercise retained selection, cancellation and transaction exclusions;
a negative control restoring repeated PAUSE behavior fails the relevant tests.
The integrated root run passed 11 selected JUnit tests and all 120 required server
GameTests (`artifacts/delivery/verify-operator-integrated.log`).
These results do not establish a live ten-program demonstration. That acceptance
requires visible OPERATOR labels, actual execution and a verified neural handback
in the recorded Minecraft session.
