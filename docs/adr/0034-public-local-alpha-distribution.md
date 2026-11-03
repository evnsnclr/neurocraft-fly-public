# ADR 0034: Public local-alpha distribution

Date: 2026-09-05
Status: accepted for local implementation; publication and code license pending

## Context

The owner requested implementation of public readiness after the Minecraft demo
attracted a large audience. The development launcher, local account/world files,
bulk graph artifacts and complete Git history are not a consumer distribution.
ADR 0019's interactive-demo direction and ADR 0033's compact presentation remain.

## Decision

Prepare a curated, explicit source export from one committed revision. Preserve
selected evidence bytes; omit private operational records and raw local sessions
instead of rewriting historical receipts. Record source identity and every exported
file's digest. The export has public-facing documentation and CI/Make targets;
the private Git audit and one test requiring seven undistributed historical raw
traces are explicitly outside that public verification scope.

Ship only the remapped project mod JAR, with its code license and notice. Keep
the full neural runtime in a local companion process using the existing controller
construction, authenticated bridge, exact identities and comparison semantics.
Add a private runtime-preparation command using the pinned official Fabric
installer (1.1.2, Apache-2.0) and the already pinned Loader/API/game tuple. Installer
and API downloads have pinned SHA-256 values. The official installer resolves
Minecraft and loader libraries; their local inventory is subsequently checked on
each copy, which detects changes but does not authenticate an arbitrary inventory.

The installed mode starts an isolated normal Fabric server. The user joins from
their own normal Fabric client with the matching two mod JARs. It never reads
Minecraft account tokens or modifies launcher profiles. Each launch creates fresh
world and neural state; the runtime template itself is never started or published.
Manual disconnect, timeout and launcher interruption clean up managed children.

The bootstrap doctor is read-only; setup prints ordered steps. Downloads, graph
derivation and runtime installation are explicit commands. Existing pinned NumPy
and PyArrow versions become runtime dependencies of actual importing packages,
without rotating graph/model numerical source or widening scientific claims.

## Consequences

This is a local alpha with an external Python process, not arbitrary-world or
public-multiplayer support. macOS development evidence, Linux fixture CI and
Windows portability remain separate. A source-only export and same-host clean
environment test do not prove first-user success. Code licensing and publication
remain owner decisions. The runtime directory contains Minecraft and must never
be bundled with the public artifact.
