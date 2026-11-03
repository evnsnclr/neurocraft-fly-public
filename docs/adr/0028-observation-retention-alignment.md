# ADR 0028: align bounded observation retention

Status: implemented, independently reviewed and focused-tested; live verification pending.
Date: 2026-09-04

## Context

The live event session on `70959db` applied three feeding sequences, then failed
with `STALE (action references an unavailable observation)`. Python retained four
accepted observations, while Minecraft's sent-observation window retained eight.
The asynchronous receiver could evict an in-flight command's observation before
Python validated that completed command. Minecraft could otherwise validate and
retire that expired frame through its existing neutral lateness path.

The final event sidecar describes completed command 3217 for observation 3218 at
world tick 3521, targeting 3523. It never reached the issued-decision/latency logs.
The server fault appeared at tick 3526; its last applied command was at tick 3521.
These records identify the exhausted Python binding, not the cause of the backlog.
Worker time, receiver scheduling and buffered arrivals are not separable here.

## Decision

Retain eight accepted observation bindings in Python, matching the existing Java
window. This supersedes only ADR 0022's declared four-observation retention cliff.
That ADR and its historical measurements remain unchanged.

Preserve full identity, sequence, observation-tick and action-envelope validation.
A retained expired command keeps its original command number and target tick;
it is never retimed, replayed as fresh or assigned a reused number. Minecraft's
existing validator retires it without applying motors. After eight newer accepted
observations, an unavailable binding still faults closed.

The two-tick action deadline, command-window capacity, starvation hold, neural
update count, authentication and control identities remain unchanged. Retaining a
binding permits validation and normal retirement; it grants no extra actuation
time. Fault diagnostics may name the affected command, observation and tick lag,
but never include authentication material.

## Verification boundary

Require delayed-worker tests at four and seven newer observations with contiguous
late/fresh command handling, the eight-newer-observation failure, and retained
identity/tick rejection checks. Minecraft must still retire an expired frame
neutral and accept its valid fresh successor under the existing deadline.
Implementation `f4ee5e4` passed 393 protocol/service Python tests and 58 focused
Java tests, including those boundaries. Focused tests alone do not establish
sustained live stability; the final session
and its unchanged timing settings must be recorded separately.
