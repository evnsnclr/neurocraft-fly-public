# ADR 0025: authenticated comparison sessions in one world

Status: accepted; implemented and focused-tested, live integration pending.
Date: 2026-09-04

## Context

The showcase needs original wiring, shuffled weights and no-input controls, followed
by restoration of the original lane. Replacing only a HUD label would misrepresent
the model. Reusing a command window or an asynchronous worker across controller
resets could apply a command from the previous model state. Relaunching Minecraft
for each comparison also changes the world and prevents a clear live demonstration.

## Decision

The opt-in service preloads three actual full-artifact controllers. All retain the
same original calibration, labeled readouts and program policy. A control identity
binds its executed view; no-input zeroes the final input vector after encoding.
Graph identity is retained and the declared control changes dynamics identity.
There is no required direction of behavioral change. The first offline probe found
escape under both original and shuffled weights; no-input selected CRUISE.

Before listening, the service writes a three-entry local catalog. The launcher
reads bounded bytes, checks the original identity, records their SHA-256 and passes
the path and digest to the server. The server verifies the exact digest, strict
bounded schema, three fixed lane names, original identity and fixed source/input/
readout fields before opening its bridge. Seeds are decimal strings, preserving
all unsigned 64-bit values. Epochs are exact nonnegative Java-range integers.
The catalog is immutable in memory after loading; a later switch does no catalog I/O.
It is a trusted local artifact pin, not a signed scientific trace.

`/neurocraftfly control original|shuffled_weight|no_input` starts a fresh epoch for
the configured fly. A pending reset, operator transaction, program-authority
transaction or lane handshake refuses another switch. The server consumes the next
epoch before retiring the old bridge, clears validators, control queues, neural
history, body-program ownership and display state, and opens a fresh bridge.
Position, orientation, velocity and world time persist. Motors remain neutral
until the existing full-identity/HMAC HelloAck is accepted. The normal action
deadline stays two game ticks. An unanswered comparison Hello is bounded by the
existing three-second connection timeout; failed attempts remain neutral and do
not make their epoch reusable.

On the service, invalid or unauthenticated requests cannot evict a live lane.
Authenticated epochs are globally increasing, including consumed failed attempts.
Handover closes the previous connection and drains tracked step, receipt, control,
telemetry and sink work before resetting or warming the selected controller.
Cancellation cannot release reset/shutdown ownership while an older worker is still
running. A genuinely wedged Python worker can therefore prevent the next handover;
the implementation does not pretend that a Python thread can be forcibly stopped.

Each prepared epoch gets an exclusive directory containing `epoch.json` and its
own optional decision, latency and display files. Preparation metadata alone does
not establish activation or an applied program. Diagnostic summarization verifies
the pinned catalog and sidecar and never joins command numbers across epochs.
Individual NDJSON lines do not carry the full epoch identity: their directory and
sidecar are part of their attribution and must stay together.

The client receives a separate atomic comparison status. Attempted lane/epoch can
be shown while switching or failed; actual lane, full identity, control hash and
seed are available only after authentication. Optional model displays must match
that exact active identity, and old-epoch values cannot become current. The display
remains independent of commands and retains its existing age limits.

## Consequences and verification

Switching preserves the Minecraft body, but resets model state. It is not by itself
a matched experiment: starting pose, stimulus geometry, timing and declared reset
procedure still need control. CRUISE is a scripted program and can move the body
even in the no-input lane. The HUD and recording must explain that distinction.

Focused service tests exercise four real loopback epochs, stale/invalid requests,
worker draining, cancellation and exclusive diagnostics. Java tests exercise
consumed epochs, stale-display rejection, timeout and failure behavior. A Minecraft
test establishes real command/program history before verifying preservation of
pose and velocity for both operator and readout ownership. Client tests check the
comparison HUD with actual font metrics at widths 320, 347 and the test window.
The full-artifact offline comparison restores original canonical command streams
exactly. These checks do not substitute for a same-world live comparison or the
final release verification.
