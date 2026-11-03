# ADR 0026: explicit world events and candidate feeding/grooming readouts

Status: accepted direction; protocol implemented, controller and world integration in progress.
Date: 2026-09-04

## Context

The showcase needs visible interactions beyond approach-triggered escape. An
isolated full-artifact probe found responses from 163 labeled labellar gustatory
inputs to the two MN9 cells and from 65 grooming mechanosensory inputs to 13
antennal motor cells. The latter sources are sensory cells, not grooming
descending neurons. All memberships come from the verified retained annotations.

A second probe added those pulses to the actual loom-v2 encoder's synthetic
light/contact/approach histories. Ordinary contact alone raised MN9 to about
0.000348, above isolated food's roughly 0.000280 peak. An absolute MN9 threshold
therefore does not identify food. Shuffling increased the antennal response;
these observations cannot justify a faithful-only grooming claim. The paired
counterfactual differences computed offline do not exist as live observations.

The existing Minecraft fly also has no health system and rejected damage. A new
interaction hook cannot honestly be described as biological nociception.

## Decision

Introduce the separate candidate `full_mcns_events_program_v1` profile. Keep old
profiles, dynamics and v1/v2 sensor bytes unchanged. New sensor revision v3 adds
an exact events object with current `food_near` and `food_contact` booleans and
independent food, brush and player attack-attempt counters. Each counter carries
its cumulative count and latest game tick; count zero requires a null tick.
Integers fit signed Java longs, and event ticks cannot exceed the observation.

Food-near means the nearest visible whitelisted dropped item within three blocks,
followed by the 60-degree forward-bearing check. Food-contact keeps the exact
grounded FEED precondition within 0.75 blocks; contact implies near. Both use
collider line of sight and deterministic UUID tie-breaking. Food events count
false-to-true near transitions once per authoritative game tick. No target bearing
or navigation goal is sent to the model. The wider near predicate is necessary
for an airborne fly to begin an explicit landing sequence before feeding.

A live non-spectator player can brush the pickable fly with the brush item outside
water/lava. A finite positive player-origin hurt request records an attack attempt,
including a player-owned projectile. Hooks require player and unremoved fly in the
same server level. They change counters, not health, velocity or item durability.
Attack handling still returns false. These are engineered interaction proxies.

Counters persist across neural resets and are retained in bounded versioned entity
NBT across save/load. Supplied malformed event NBT makes event sampling unavailable
instead of silently restarting a monotonic stream. Missing legacy NBT may start
zero. First observed counters initialize a fresh encoder baseline, absorbing
already-present events rather than replaying them. Coalesced counter jumps become
one bounded pulse, not amplified stimulation or fabricated intermediate events.

The new observation declaration hash is
`d7b135a1ce79d034454394d8ee73eb5b9f5fe46bb79ee924da465b8cec62ba80`.
Its provisional earlier v3 definitions were revised before any live profile launch;
the v1/v2 definitions and historical receipts are unchanged.

## Candidate neural mapping

Preserve the loom-v2 component arithmetic, then add at most one observation's
amplitude-1 pulse per selected source member: food to the 163 gustatory cells;
brush/attack attempts to the 65 mechanosensory cells. Both mechanical sources
occurring together still produce one pulse. Five neural updates execute that
observation. Ignore events older than eight game ticks. Apply no-input after
all additions, so no component can escape the control gate.

After four completed observations establish a baseline, freeze their mean before
an event and inspect positive raw-mean change over an eight-tick response window.
This is a declared pre/post score, not the diagnostic probe's paired difference.
The provisional thresholds are 0.0001 for MN9 and 0.0003 for antennal motors.
They must be tested against the actual candidate and controls before live adoption;
this ADR does not claim those tests passed or those values are biological.

ESCAPE keeps priority and discards pending food/grooming events. A scored food
event selects FEED if current contact permits; otherwise it selects scripted LAND
for at most 200 game ticks, retaining the original scored-event identity. During
that bounded landing interval, actual food contact permits FEED. Food loss, fluid,
timeout, escape or reset cancels the sequence. Contact changes cannot renew the
event. Simultaneous eligible food/grooming resolves to food by a declared rule.
The two gestures latch entry modulation with urgency zero for 60 ticks, followed
by a 120-tick cooldown and a required fresh event. Java still enforces body
preconditions and may refuse a requested program. Late commands may not apply.

Use the existing loom-v2 decoder/calibration as an explicitly identified component,
not as validation of the added events. The new composite binding hashes its source,
population membership, pulses, event gates, score formulas and arbitration. The
candidate remains unvalidated until its new numerical and live checks are measured.

## Evidence and compatibility

Keep ProgramAction's actual TTMn/DNa01/wing fields unchanged. New named evidence
records MN9/antennal means, readiness, frozen baselines, scores, eligibility and
event ages with the full command/state/identity envelope and binding/policy hashes.
Only the existing applied-command ledger can upgrade an issued selection to an
applied display. Missing, stale or malformed display data never changes actuation.
Scientific notation prevents small values from appearing as rounded zero.

The transport tests pin old goldens, closed shapes, large counters, tick ordering,
sensor-identity refusal and the published JSON Schema. Further checks must establish
actual world counters, save/load continuity, no-input zero scores, coalesced/reset
events, cancellation and cooldown, live LAND→FEED and GROOM receipts, and final-build
timing under the unchanged action deadline. All three GOAL facts stay visible.
