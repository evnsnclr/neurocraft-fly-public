# ADR 0037: Patched native and Java dependencies in the Companion

Date: 2026-10-30
Status: accepted for private implementation; public release unapproved

## Context

The owner requested patched SQLite and affected Java libraries in a rebuilt
private candidate. The latest selected PBS archive still embeds SQLite 3.53.1.
Gradle dependency overrides alone do not affect libraries unpacked by Minecraft's
server bundle or inherited by a normal player launcher.

## Decision

Build a documented PBS derivative using pinned SQLite source. Reconstruct both
POSIX Python engines from the full archive's objects using the matching compiler;
rebuild Windows's SQLite DLL with the existing MSVC toolchain. Bind the source,
recipe and native hashes in a derivative receipt. Verify the loaded engine and
stdlib functionality before packaging; preserve the original upstream notices.

Use one pinned Guava/failureaccess/Log4j policy for Gradle, an explicit isolated
server classpath and a separately generated launcher client version. Verify the
actual loaded JAR bytes before mod startup. Keep Minecraft 1.21.5, Fabric's tuple
and neural model unchanged. Existing player profiles and account data remain
under the player; selecting the generated version is part of setup.

## Consequences

Release builders now maintain and verify the native derivative and Java library
selection. Players receive the completed Python runtime and install the generated
client version through their own launcher. A copied mod JAR alone cannot upgrade
Minecraft's libraries and is refused with an old/mixed library selection.

Exact candidate tests, package hashes, remaining dependency exceptions and player
qualification are reported separately. The original rc.2 receipts remain intact.
This decision authorizes no publication or broader biological claim.
