# NeuroCraft Fly architecture

Current overview: 2026-09-06. The implemented showcase profile is
`full_mcns_interactions_program_v2`.

NeuroCraft Fly connects a source-derived fruit-fly neural network to an
interactive Minecraft body. Engineered sensory inputs drive recurrent model
activity. **Hand-chosen readouts select and modulate scripted body programs.**
The neural network, event policy and body programs each contribute to what a
player sees.

The [model card](MODEL_CARD.md) specifies the equations, populations and limits.
The [research quickstart](RESEARCH_QUICKSTART.md) provides a small headless
example. The earlier [gate-oriented architecture](ARCHITECTURE_2026-08-18.md)
is preserved unchanged as history; its primitive-only movement rules and
then-pending full-network execution statements do not describe this profile.
[ADR 0019](adr/0019-pivot-to-interactive-connectome-demo.md) records the pivot.

## Current control path

```mermaid
flowchart LR
    world[Authoritative Minecraft world] --> observation[Physical fields and typed events]
    observation --> encoder[Engineered population inputs]
    graph[Retained MaleCNS graph and modeled signs] --> neural[Leaky-rate dynamics]
    encoder --> neural
    neural --> readout[Named population readouts]
    readout --> policy[Source-aware program selection]
    observation --> policy
    policy --> body[Scripted body programs]
    world --> body
    body --> action[Bounded thrust and torque]
    action --> world
    neural --> hud[Read-only model display]
    policy --> hud
    action --> hud
```

The direct arrows from observations to the policy and from the world to body
programs are intentional. Minecraft supplies actor categories and event
eligibility; the network does not infer those categories. Programs use declared
world geometry, contact and eligible targets to execute motion. For example,
CRUISE performs clearance control and obstacle avoidance, and FLEE uses a visible
player target. A neural gate selects a program; it does not generate its movement
trajectory or animation.

## Components and authority

| Component | Implemented responsibility |
| --- | --- |
| Data acquisition and graph construction | Verify pinned MaleCNS sources; retain stable neuron IDs, directed source rows and raw counts under a declared filter; hash transformation lineage. Bulk inputs stay outside the source package. |
| Python neural runtime | Apply an explicit transmitter-to-sign policy and incoming normalization; advance deterministic float64 leaky-rate state on the retained 166,700-node, 25,582,938-edge graph. |
| Engineered encoder | Map typed ray approach, contact, light, food, brush and attack-attempt fields to declared populations. These are software proxies, not reconstructed sensory transduction. |
| Python interaction policy | Combine observed source eligibility with TTMn, DNa01, MN9 and antennal motor readouts; enforce priority, baselines, holds, cooldowns and cancellation. |
| Java body programs | Execute HOVER, CRUISE, BANK, LAND, WALK, GROOM, FEED, ESCAPE_JUMP, FLEE and APPROACH as scripted motion. The six-source neural policy selects a subset; the separate operator tour can demonstrate all ten. |
| Minecraft logical server | Own entity state, sensor sampling, program preconditions, bounded actuation and physics. An issued program can be refused or expire before application. |
| Bridge | Authenticate loopback sessions; bind graph/controller identities, sequences and ticks; reject invalid or stale commands before mutation. I/O and neural execution stay off the game thread. |
| Client renderer and HUD | Render the custom fly and actual model samples. Show network condition and control authority; distinguish current body state, applied command samples and separately sequenced neural samples. |

The ordinary showcase is an **engineered encoder and program-policy mode**.
The internal original-condition string `faithful` identifies its unchanged graph
condition; it does not convert this composite into the historical
Faithful Organism Mode. Original wiring, shuffled weights and no input retain
separate identities. OPERATOR authority is separately labeled scripted control.

## Numerical and timing contract

The default dynamics use a 10 ms simulated neural step, 50 ms leak time,
recurrent gain 0.9, sensory gain 1, zero baseline, tanh activation and no noise.
The full-array backend uses a uniform numerical delay of one step; at committed
index `n`, recurrence reads `a[n-1]` while leak uses `a[n]`.
See the [equations and implementation links](MODEL_CARD.md#dynamics).

The interactive profile executes five neural updates per observation against a
normal 20 TPS game target. The existing two-tick command deadline is an
application window, not a promise of a 50 ms end-to-end response. The public
launcher requests eight native kernel threads by default; the actual thread
choice, source, library identity and measured timings belong with each run.
An older slow unsigned graph smoke does not characterize this backend, and an
earlier fast profile does not qualify a later build. Target TPS, achieved
throughput, service compute time and video frame rate are different quantities.

The Python reference engine supports the small fixture used in the notebooks.
That headless pathway does not launch the full graph, invoke the Minecraft
interaction policy or reproduce its six responses.

## Observations and experiment records

The bridge and runtime bind observations, neural state digests, declared
population readouts, issued commands and applied receipts to run identities.
Compare an issued selection with its actual applied receipt before calling it
an executed movement. A visualization is explanatory; it cannot establish an
application that the underlying records do not show.

Offline replay or prescribed-input probes hold an input history fixed. A live
closed loop can produce different later observations after different movement.
Those are different experiment designs. Weight shuffling preserves endpoints
and changes magnitudes; it is not rewiring. No input zeroes delivered sensory
drive, but baseline scripted CRUISE remains available in the live policy.

The [current evidence boundaries](SCIENTIFIC_CLAIMS.md) distinguish historical
full-network sessions, numerical probes, automated software checks and the new
fixture example. Old receipts and ADRs remain bound to their original versions;
their original pending-work clauses are historical.

## Local-alpha deployment

The alpha has a Fabric mod and an external Python companion process. The Java
mod connects to an authenticated loopback WebSocket server. The installed path
prepares a separate Fabric server and the user joins with their own matching
Fabric client. Runtime preparation does not supply a Minecraft account or
redistribute Minecraft.

The pinned game tuple is Minecraft Java 1.21.5, Java 21, Fabric Loader 0.16.10,
Fabric API 0.128.2+1.21.5 and Loom 1.10.5. The current mod version is
0.2.0-rc.5. This is a local developer alpha, not established arbitrary-world or public
multiplayer support. Same-host production-client compatibility and fresh exported
source verification were recorded for earlier candidates (September 2026); the
consolidated candidate has development-worktree verification and an export preflight
only. Independent first-user installation and official-launcher account flow remain
unobserved.

Project-owned software and associated documentation are [MIT](../LICENSE).
MaleCNS-derived data and dependencies retain their own terms. See
[data provenance](DATA_PROVENANCE.md) and [third-party notices](../THIRD_PARTY.md).

## Design and source pointers

- [Six-source policy decision](adr/0032-six-interaction-program-policy.md) and
  [implementation](../packages/controller-runtime/src/neurocraft_controller/interaction_program_policy.py).
- [Separate brush and attack inputs](adr/0031-separate-interaction-inputs.md) and
  [full controller](../packages/controller-runtime/src/neurocraft_controller/full_brain_controller.py).
- [Array neural engine](../packages/neural-runtime/src/neurocraft_neural/full_engine.py)
  and [full-graph controls](../packages/neural-runtime/src/neurocraft_neural/full_controls.py).
- [Compact HUD and model details](adr/0033-compact-recording-hud.md),
  [local-alpha distribution](adr/0034-public-local-alpha-distribution.md) and
  [MIT decision](adr/0035-project-code-mit-license.md).
