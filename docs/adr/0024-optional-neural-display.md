# ADR 0024: Optional neural display

_Presentation architecture, accepted 2026-09-04_

**Status:** Implemented in the delivery working tree, with focused producer,
service, parser, matching, and freshness tests. Combined delivery verification
and live display acceptance remain separate work.

## Context

The demo should show actual model activity from named populations beside the
command that was applied. Motor values and animation alone do not supply that
information. The display must remain optional and must not acquire control
authority. ADR 0019's claim boundaries and ADR 0021's hand-chosen mapping remain
visible.

## Decision

1. Display six signed means of dimensionless model activity: LC4 left/right
   (71/55 members), DNp01 left/right (1/1), and TTMn left/right (1/1). The binding
   hashes the verified graph and annotation join, source body IDs and dense
   indices, and existing sensory/TTMn bindings. LC4 and TTMn membership must match
   the controller's bindings. DNp01 is interior model activity for display only;
   this adds no driving readout. Missing or invalid populations are unavailable,
   never invented zeros. These quantities are not measured spikes or spike rates.
2. Capture after a completed controller step in the **same serialized compute
   worker**, under the service execution fence. The producer checks the completed
   evidence and engine state against the command's identity, sequence,
   observation tick, neural interval, and state digest before taking means from
   an immutable activity copy. It checks the engine digest again after capture.
   Capture runs every four commands by default; its cost remains included in the
   service's compute measurement. No claim of zero overhead follows.
3. Send only the immutable six-mean frame to a daemon writer with one coalescing
   slot and a nonblocking offer. The writer atomically replaces a local file using
   a temporary file in the same directory. The closed
   `neurocraft.brain_display.v1` JSON frame has a 16 KiB ceiling and carries the
   full run identity, command envelope fields, state digest, encoder schema,
   binding hash, six finite means in `[-1, 1]`, and positive counts. This file is
   independent of the authoritative command WebSocket and its queues.
4. Read the file only on a dedicated-server polling worker, every 200 ms, into
   one immutable slot. Bound bytes and strict parsing, including the configured
   identity, precede freshness adoption. The game thread performs no display
   disk I/O. A 128-entry ledger records commands actually applied; neutral,
   rejected, and refused commands do not qualify. A visible sample must match
   that ledger exactly on full identity, command sequence, observation tick,
   neural start/end, and state digest. First matched encoder/binding metadata is
   retained until reset; drift, regression, or changed contents for the same
   command hide the sample.
5. Enforce two server freshness bounds: at most one second since the frame's
   **first file read**, and at most 20 authoritative game ticks since its source
   observation. Repeated reads do not refresh that age. An old command first
   discovered late therefore cannot appear current solely because it was just
   read. Reset clears the application ledger, accepted display, and synchronized
   payload; shutdown closes the reader.
6. After physics, every fourth server tick, synchronize one complete bounded
   presentation string containing status, elapsed age in milliseconds, and an
   optional matched frame. The client never reads the file. Its cache parses on
   synchronized-data receipt, anchors its own monotonic receive time, and adds
   elapsed local time when rendering, expiring the sample after the remaining
   one-second budget even if server updates stop or the HUD was hidden. Server
   and client `nanoTime` timestamps are never compared. This age is receipt/read
   age, not production age or end-to-end latency.

The program launcher enables the file and server reader for program profiles,
including the default `full_mcns_loom_program_v2` target. The standalone service
opts in through `--brain-display-out`; the server receives
`neurocraft.fly.brainDisplayPath`. Nominal sampling and synchronization are five
times per second at 20 commands/ticks per second, not a measured live rate.

## Failure behavior and claim boundary

Display startup, capture, write, parse, and matching failures make the display
unavailable or stale. An escaping observer exception disables that observer;
subsequent commands continue unchanged. Display data never enters an actuator,
command admission, or policy decision. The HUD suppresses activity during
operator pause or controller fault and labels DNp01 as display-only model
activity. Historical escape-arc ratios remain explicitly offline measurements.

The local file is **not an authenticated causal trace**. Matching its envelope
and digest to an applied command prevents accidental stale or unrelated display;
it does not cryptographically authenticate the file's means or prove that a
population caused the observed behavior. The binding hash records provenance for
inspection. Display enablement does not rotate controller identities or change
the model's sensory, recurrent, or motor rules.

Implementation authority is `neurocraft_controller/brain_display.py`, the service's
`completed_step_observer`, `FlyControllerRuntime`, `FlyEntity`, and the
`BrainDisplayFrame/Reader/State/Presentation/ClientCache` classes. Focused tests
cover binding integrity, actual means, bounded transport, application matching,
reset and expiry, and authenticated motor/program command-byte equivalence with
display off, on, and failing. The CLI tests also cover optional startup cleanup.
Those checks do not qualify sustained live timing, biological performance, or
the filmed demo.
