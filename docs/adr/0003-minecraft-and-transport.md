# ADR 0003: Use Fabric 1.21.5 and a loopback JDK WebSocket bridge

_Minecraft embodiment and cross-process transport, accepted 2026-08-16_

---

| Field | Value |
|---|---|
| **Status** | Accepted; implemented and automated-test verified; normal launcher unverified |
| **Date** | 2026-08-16 |
| **Minecraft** | `1.21.5` |
| **Transport** | JDK WebSocket client to Python `websockets` server |
| **Research basis** | [Minecraft architecture decision](../research/minecraft_architecture_decision.md) |

## Current implementation addendum (2026-08-17)

The architecture below was written before implementation. The current strict wire is
`neurocraft.protocol.v2` over WebSocket subprotocol `neurocraft.v2`; it supersedes
the pre-implementation "protocol v1" wording while retaining the same authority,
identity, bounded-queue, primitive-actuation, and fail-closed decisions. Strict builds,
server/client GameTests, and one authenticated external GameTest now exist. A normal
interactive launcher run, long soak, and G3 promotion still do not.

## 📋 Context

The embodiment must remove vanilla action selection, sample exact server state, apply bounded primitive motors, render an original Drosophila, expose a read-only overlay, and support automated server/client tests. The observed machine has Java 21 and a vanilla 1.21.5 launcher profile but no system Gradle. That profile does not prove Fabric installation or launcher compatibility.

An external Python process is required for scientific tooling and fault isolation. The bridge is a small local control channel rather than a neural-state or visualization stream.

## 🎯 Decision

Build a custom Java mod on Fabric with this authoritative tuple:

| Component | Pin |
|---|---:|
| **Minecraft Java Edition** | `1.21.5` |
| **Java** | `21` |
| **Fabric Loader** | `0.16.10` |
| **Fabric API** | `0.128.2+1.21.5` |
| **Fabric Loom** | `1.10.5` |
| **Gradle wrapper** | `8.12` |
| **Java WebSocket** | JDK 21 `java.net.http.WebSocket` |
| **Python WebSocket** | `websockets==17.0.1` |

Use Loom 1.10's stable `fabric-loom` plugin configuration rather than the newer moving snapshot configuration in the inspected example branch. Commit and verify the Gradle wrapper; system Gradle is neither required nor an accepted hidden prerequisite.

This tuple supersedes provisional research suggestions for Minecraft/Fabric 1.21.1 and `Java-WebSocket`. No third-party Java WebSocket library is selected.

## 🏗️ Entity and rendering split

Register a project-owned fly entity. The logical server owns spawn/reset, raw sensors, experiment state, bridge sequencing, motor validation, physics, collision, pose, and authoritative telemetry. The entity has no goals, targets, navigation, pathfinding, or vanilla fallback intelligence. If its inheritance introduces those facilities, tests must prove they remain empty and unused.

The client source set owns only the original Drosophila model, texture, renderer, animation, and HUD. It receives compact presentation state and bounded activity summaries. Closing the client/HUD or corrupting a presentation packet cannot change server actions.

A bee or other vanilla proxy may appear only in a clearly labelled developer smoke that is incapable of producing acceptance evidence. Paper-only plugins, Mineflayer players, data packs, and RCON cannot meet the combined custom-entity, renderer, sensor, and anti-bypass requirements.

## 🔐 Protocol and failure contract

The Java mod connects as a WebSocket client to a Python server bound only to loopback. The versioned protocol must enforce:

- negotiated subprotocol and a random per-run token that is never logged
- explicit protocol, run, session epoch, entity, graph, adapter, dynamics, decoder, and schema identities
- monotonic observation/command sequences, game ticks, neural-step ranges, and deadlines
- finite numeric values, fixed enums, known fields, bounded arrays, and a 64 KiB message limit
- compression disabled and bounded receive/write buffers
- a one-slot latest observation handoff and at most a two-command validated window
- rejection of malformed, duplicate, future, wrong-session, wrong-hash, and more-than-two-ticks-stale messages
- immutable parsing off-thread and final application on the Minecraft server thread

The server tick never blocks on sockets, Python, serialization, or disk. On timeout, disconnect, invalid state, queue/trace failure, or identity mismatch, the next tick enters `PAUSED_FAULT`, applies neutral action, clears controller-owned actuation state, and leaves vanilla AI disabled. Reconnect creates a new session epoch; old commands remain invalid.

## 📥 Raw sensors and primitive motors

Faithful sensors are versioned server-authoritative physical measurements: entity pose/velocity, collision/contact/fluid flags, declared local light or environmental quantities, and deterministic body-relative ray-cast results. Exact target vectors, semantic object labels, global world queries, camera pixels, and task-oracle channels are excluded from Faithful behavioral evidence.

The server samples all fields at one declared tick boundary and records raw plus encoded values. Motor commands contain only declared bounded primitive axes such as forward/lateral/vertical thrust and yaw/pitch torque. Minecraft collision, gravity, drag, and fluid response remain physics, not hidden policy. Absolute placement is limited to logged reset operations.

## ⚡ Consequences

### Positive

- One Fabric mod provides server authority, custom rendering, HUD, packets, and GameTests.
- Java 21 supplies the WebSocket client without another mod classpath dependency.
- The committed wrapper fits the local no-system-Gradle environment.
- The Python runtime remains isolated, restartable, and independently testable.

### Negative

- Both Java and Python implementations must share golden protocol fixtures.
- The launcher needs a Fabric 1.21.5 profile beyond the observed vanilla profile.
- Client rendering still requires a real launcher acceptance run; headless success is insufficient.
- JSON is inspectable but requires strict schema and resource bounds.

## 🔍 Rejected or deferred alternatives

- **NeoForge:** capable but redundant for v1; it adds a second ecosystem without a requirement-serving advantage.
- **Paper plugin:** cannot alone provide the custom client renderer/HUD and encourages version-fragile proxy techniques.
- **Mineflayer:** controls a player, not the custom server-owned fly; optional offline observer only.
- **Data packs:** useful for static arenas/fixtures, not the controller or socket.
- **Embedded Python:** rejected due to JVM/native packaging and shared crash boundary.
- **gRPC:** reserved for measured schema/throughput needs; Protobuf/Netty complexity is unnecessary for the bounded v1 channel.

## 🧪 Promotion evidence

Required evidence includes a clean wrapper build, dependency verification, server GameTests, client GameTests/Xvfb, deliberately failing harness self-tests, cross-language golden messages, malformed/fault tests, a multi-process soak, and a real launcher run of the same built mod/config. At decision time this ADR claimed none of those results; current bounded evidence is indexed in [the source log](../SOURCE_LOG.md), with the remaining gaps above still open.

## 🔗 References

- [Architecture overview](../ARCHITECTURE.md)
- [ADR 0004: Causal trace and replay](./0004-causal-trace-and-replay.md)
- [Minecraft architecture decision](../research/minecraft_architecture_decision.md)
- [Systems and performance plan](../research/performance_plan.md)
- [Scientific red-team review](../research/scientific_red_team.md)
