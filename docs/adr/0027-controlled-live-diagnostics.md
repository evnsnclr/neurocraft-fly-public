# ADR 0027: controlled live diagnostics

Status: accepted implementation; live acceptance remains unestablished by this record.
Date: 2026-09-04

## Context

[ADR 0019](0019-pivot-to-interactive-connectome-demo.md) requires a playable,
honestly labeled demonstration. [ADR 0025](0025-comparison-session-handover.md)
provides authenticated same-world lane changes, but preserving a body's current
pose does not produce matched starting conditions. [ADR 0026](0026-engineered-world-events.md)
also needs a reproducible way to exercise real client interaction requests without
mistaking scripted input for a person, a server event, or neural execution.

## Decision

Keep both facilities opt-in and outside the neural controller. The server owns a
fixed 60-cell plan: ten declared body seeds, each crossed with entity/block stimuli
and `original`, `shuffled_weight`, `no_input` lanes. The neural controller and
physical body remain in a closed loop; the stimulus follows a fixed world-coordinate
schedule and never follows the fly or adapts to a readout.

Each cell consumes a fresh epoch through the existing authenticated handover.
After retiring the previous bridge, its callback resets pose, velocity, body
ownership/history, event counters and the independent body wander seed before
starting the next bridge. Clearing event counters is an explicit fixture-reset
exception to ordinary neural resets, which retain them. Later program entries keep
their actual world ticks; reset retains the existing HOVER@0 sentinel. The paired
seed only replaces CRUISE's phase seed. Both the
callback and its caller verify the declared dry, stationary, airborne body and
complete floor support two blocks below its feet.

Only an applied program command 0, bound to actual observation 0 and neural steps
0..5, anchors a cell. Its 160 world ticks comprise 20 baseline, 60 approach and
80 recovery ticks. Six front planes, z=6 through z=1, last ten ticks each. A native
cow's front face and a stone block's front face share that plane; their complete
shapes differ. Stimulus updates precede sensing, and ordinary authenticated
commands retain exclusive motor authority. Timing remains the existing two-tick
action budget; normal-speed measurement uses 20 TPS, without changing deadlines.

Missing/refused command 0 never becomes command 1's anchor. Invalid cells and
stimulus misses remain in the declared denominator, with no automatic retries.
Terminal faults, writer failure, or skipped/duplicate observation ticks stop the
plan. Completion attempts a fresh original epoch; restoration is confirmed only
at exact-identity ACTIVE with the trial seed observed cleared. Console stop retains
its actual final lane and seed state instead of claiming restoration.

An exclusive, bounded NDJSON ledger lives beside `epochs/`. Its header stores the
exact plan JSON and its UTF-8 SHA-256. Rows retain actual identities, pre/post-reset
body state, program clocks, observations, applied actions, stimulus geometry and
local program-receipt offers. Offers and cumulative transport-sent counts remain
separate from service-issued decisions and received receipts. Per-epoch sidecars
supply catalog/control provenance; an operator release manifest must separately
bind source/build and run artifacts. This local diagnostic is not a signed trace.

Separately, a bounded client inbox accepts only explicit brush/attack requests
against the currently aimed, reachable fly. It calls vanilla client interaction
methods on the current connection; it never aims, equips, teleports, writes counters,
or changes neural/motor state. Its result means dispatched, refused, or uncertain
client action. Server event counters and exact applied program receipts are the
subsequent evidence boundaries.

## Consequences

These are engineered diagnostics supporting a demo, not biological stimuli,
anatomical body control, human-input evidence, or a live performance qualification.
No-input may still move through scripted CRUISE, and shuffled wiring need not lose
an observed response. Implementation and focused tests at runner `59b6e41` and
client inbox `0cf3561` establish bounded behavior, not successful live trials or
filming acceptance. [MATCHED_TRIALS.md](../MATCHED_TRIALS.md) specifies operation and
how missing evidence remains visible.
