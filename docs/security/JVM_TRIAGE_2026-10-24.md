# JVM advisory triage — 2026-10-24

Scope: the 15 previously unexcepted OSV findings from the private companion
candidate's `mod/gradle.lockfile`. This is a targeted static assessment, followed
by a managed-JVM temporary-directory mitigation and focused regression checks.
It is not an exhaustive security review, a library upgrade, Windows gameplay
qualification, or authorization to publish.

The baseline report is `artifacts/audit/jvm-advisories.json`, produced on
2026-10-24 with 47 advisories, 32 existing exact exceptions and 15 new findings.
The private source base was `6724541`; the final source manifest must identify
the exact candidate containing this delta. Lockfile SHA-256: `cb7ddee344b93eb2e4efb1fbdd1bfe537c87a4ef676086998c7b6b1aa259c786`.
The existing 32 exception entries and their original expiration dates remain
unchanged. The 15 new exact entries expire **2026-11-24**.

## Decision and limits

Retain the official Minecraft/Fabric dependency tuple. A Gradle-only library
upgrade would change development classpaths but leave the companion's official
Fabric-installed Minecraft libraries unchanged. The mod JAR does not bundle
these libraries. No dependency version, lockfile or verification hash changes
are proposed by this triage.

Fourteen findings are `not_actionable` for the stated managed application route
because their necessary API/configuration or lower-trust input is absent. The
shared-temp JNI finding was initially `needs_review` (rank 1): loopback does not
remove a different local user's ability to attack a shared temporary directory.
The added private-JVM-temp enforcement removes that precondition for managed
interactive launches. This supports a bounded exception after the focused
checks; it does not clear the affected package for arbitrary embedding.

The security policy is the root `SECURITY.md`, as resolved for `mod/`;
there was no nearer scoped policy. Source trust includes untrusted player and
HTTP input, so merely calling the application local was not used to dismiss a
reachable parser. Existing-world imports, arbitrary server joins, extra mods,
shared writable installations and user-edited compression/JVM settings require
re-triage. Normal external Fabric clients remain owned by their launcher; this
project does not change their account files, JVM flags or profiles.

## Finding-by-finding assessment

The table preserves the baseline report's input order. Every link identifies
the specific primary/vendor advisory reproduced by GitHub's advisory database;
its API requirements were compared with the concrete consumers below. All 15
upstream package ranges remain affected. New exceptions cover only the exact
advisory/package/version/configuration combinations listed in the ledger.

| Item | Advisory / CVE | Locked component | Necessary path and assessed boundary |
|---|---|---|---|
| 1 | [GHSA-7hhh-6rmp-j9qf](https://github.com/advisories/GHSA-7hhh-6rmp-j9qf) / CVE-2026-89425 | `com.fasterxml.jackson.core:jackson-core:2.13.4` | DataInput JSON error-token growth. Requires createParser(DataInput). The mod/Minecraft do not use Jackson; inherited MSAL JsonHelper binds String JSON through a default ObjectMapper, and Log4j JSON configuration reads streams/trees. No DataInput-backed Jackson parser was found in these consumers. |
| 2 | [GHSA-5gvw-p9qm-jgwh](https://github.com/advisories/GHSA-5gvw-p9qm-jgwh) / CVE-2026-59889 | `com.fasterxml.jackson.core:jackson-databind:2.13.4.2` | Unwrapped property view bypass. Requires an active JsonView together with a JsonUnwrapped container authorization boundary. No JsonView references occur in the inspected non-Jackson consumers; Log4j has a JsonUnwrapped logging-output DTO but no view-based input authorization. MSAL DTOs do not use either feature. |
| 3 | [GHSA-cxp5-3px4-pw24](https://github.com/advisories/GHSA-cxp5-3px4-pw24) / CVE-2026-91777 | `com.fasterxml.jackson.core:jackson-databind:2.13.4.2` | Quadratic identity forward references. Requires identity-enabled collection/map deserialization. No JsonIdentityInfo references occur in the inspected mod, Minecraft or non-Jackson consumers; MSAL uses ordinary DTOs/maps, not object-ID graphs. |
| 4 | [GHSA-gx83-3vf8-gh7j](https://github.com/advisories/GHSA-gx83-3vf8-gh7j) / CVE-2026-83557 | `com.fasterxml.jackson.core:jackson-databind:2.13.4.2` | Comparable polymorphic validator gap. Requires polymorphic Comparable targets under DefaultBaseTypeLimitingValidator. Neither JsonTypeInfo nor default-typing activation occurs in the inspected consumers. MSAL JsonHelper initializes a plain ObjectMapper without polymorphic typing. |
| 5 | [GHSA-vvgp-rfg2-7rr6](https://github.com/advisories/GHSA-vvgp-rfg2-7rr6) / CVE-2026-77310 | `com.fasterxml.jackson.core:jackson-databind:2.13.4.2` | InetAddress deserialization DNS lookup. Requires a JSON target typed InetAddress or enabled polymorphic binding to that type. The inspected MSAL InetAddress references belong to its local HTTP listener/interactive networking, not Jackson DTOs; JSON helper binds fixed DTOs/String maps and enables no polymorphic typing. The mod performs no Jackson binding. |
| 6 | [GHSA-wjgm-6hv5-3cvf](https://github.com/advisories/GHSA-wjgm-6hv5-3cvf) / CVE-2026-19032 | `com.fasterxml.jackson.core:jackson-databind:2.13.4.2` | Path deserialization provider selection. Requires JSON binding into java.nio.file.Path and a side-effecting filesystem provider. No such JSON target is present in the inspected consumers. MSAL uses Path for Azure Arc token-file I/O, separate from JSON DTO binding. The mod performs no Jackson binding. |
| 7 | [GHSA-wv8q-qhhj-9h54](https://github.com/advisories/GHSA-wv8q-qhhj-9h54) / CVE-2026-91776 | `com.fasterxml.jackson.core:jackson-databind:2.13.4.2` | Unknown polymorphic type-ID retention. Requires name-based polymorphism with defaultImpl/fallback and a long-lived mapper accepting attacker IDs. No JsonTypeInfo or default-typing activation is present in the inspected consumers; plain MSAL String/DTO binding does not supply that configuration. |
| 8 | [GHSA-c4c3-7fpv-j4q5](https://github.com/advisories/GHSA-c4c3-7fpv-j4q5) / CVE-2026-75595 | `io.netty:netty-handler:4.1.118.Final` | TLS SNI route/mTLS bypass. Requires SniHandler/SslClientHelloHandler TLS termination with differing default/per-name SslContexts. Neither Minecraft client/common bytecode nor project source uses those handlers. Minecraft game connections use its packet protocol; the bridge is an outbound JDK WebSocket client. No SNI authorization boundary exists here. |
| 9 | [GHSA-fccg-mwvh-qqg4](https://github.com/advisories/GHSA-fccg-mwvh-qqg4) / CVE-2026-75596 | `io.netty:netty-handler:4.1.118.Final` | Quadratic TLS ClientHello aggregation. Requires the SniHandler/SslClientHelloHandler pre-handshake TLS aggregation path. Those handlers are absent from the inspected Minecraft/project consumers; adding an arbitrary TLS proxy is outside the companion route and requires re-triage. |
| 10 | [GHSA-5q95-hrpc-m3w3](https://github.com/advisories/GHSA-5q95-hrpc-m3w3) / CVE-2026-77420 | `org.jline:jline-reader:3.20.0` | JLine history-ignore regex. Requires control over the HISTORY_IGNORE reader configuration. This component is on Loom development-console classpaths, not the Minecraft library family used by the official installed server. Project code never sets HISTORY_IGNORE or exposes JLine variables through player/HTTP commands. Developer JVM/config modification is same-privilege trusted configuration. |
| 11 | [GHSA-343h-94h5-c4wr](https://github.com/advisories/GHSA-343h-94h5-c4wr) / CVE-2026-106449 | `org.lz4:lz4-java:1.8.0` | LZ4 empty-block recursion. Requires LZ4BlockInputStream stopOnEmptyBlock=false. Minecraft RegionFileVersion uses the InputStream-only constructor, whose 1.8.0 bytecode delegates with stopOnEmptyBlock=true. The managed fresh-world launcher additionally pins region-file-compression=deflate. |
| 12 | [GHSA-4v53-57pg-c464](https://github.com/advisories/GHSA-4v53-57pg-c464) / CVE-2026-106452 | `org.lz4:lz4-java:1.8.0` | LZ4 compressed-length allocation. Requires attacker-controlled LZ4BlockInputStream headers. The managed server creates a new world, explicitly selects deflate, and copy_runtime rejects imported world/log state. Minecraft packet compression does not use this block-stream path. Imported LZ4 worlds and modified compression settings are excluded from this exception. |
| 13 | [GHSA-6cx8-rjf8-pr8g](https://github.com/advisories/GHSA-6cx8-rjf8-pr8g) / CVE-2026-106453 | `org.lz4:lz4-java:1.8.0` | LZ4 length-wrapper allocation. Requires LZ4DecompressorWithLength. No reference to that API appears in project, Minecraft common/client or the inspected non-LZ4 locked consumers. Minecraft uses LZ4BlockInputStream for optional region compression, not this length-wrapper API. |
| 14 | [GHSA-gm45-99xc-r7wv](https://github.com/advisories/GHSA-gm45-99xc-r7wv) / CVE-2026-106450 | `org.lz4:lz4-java:1.8.0` | LZ4 frame buffer amplification. Requires LZ4FrameInputStream, a distinct frame format. No reference to that API appears in the inspected project/Minecraft/non-LZ4 consumers; optional Minecraft region compression uses the block-stream API, and managed worlds select deflate. |
| 15 | [GHSA-mcr4-qmvw-px4g](https://github.com/advisories/GHSA-mcr4-qmvw-px4g) / CVE-2026-106451 | `org.lz4:lz4-java:1.8.0` | LZ4 JNI shared-temp replacement race. Requires another local user writing the native-extraction directory. Managed installed and development interactive JVMs now receive an exclusive per-run java.io.tmpdir created mode 0700; POSIX owner/permission checks are enforced, and Windows requires CPython 3.13+, whose mkdir(0700) applies a protected user/admin ACL. Gradle explicitly forwards the path to game JVMs. Generic unmanaged JVMs/shared writable installations are excluded; fresh GameTest worlds use default DEFLATE. Windows ACL execution remains a platform qualification check, not an observed result. |

## Inspected static evidence

- Project Java source contains no Jackson or LZ4 API calls, no JLine variable
  configuration and no SNI handler. The outbound neural bridge uses the JDK
  WebSocket client; Minecraft packet handlers are separate from Netty TLS SNI.
- Cached mapped Minecraft 1.21.5 common/client class pools contain no Jackson
  or SNI handler references. `RegionFileVersion` is their LZ4 caller: its optional
  LZ4 wrapper invokes `LZ4BlockInputStream(InputStream)` and its `DEFAULT` is
  `VERSION_DEFLATE`. Static `javap` inspection did not run Minecraft classes.
- In locked `lz4-java:1.8.0`, that constructor delegates to the variant with
  `stopOnEmptyBlock=true`. No non-LZ4 locked consumer references
  `LZ4FrameInputStream` or `LZ4DecompressorWithLength`.
- MSAL 1.17.2 `JsonHelper` constructs a default `ObjectMapper`, then changes
  unknown-property/control-character/null-output options. Its binding calls use
  String input. It does not enable default typing or views. Inspection found no
  `JsonView`, `JsonIdentityInfo` or `JsonTypeInfo` references in its DTOs.
  Its networking `InetAddress` and Azure Arc file `Path` uses are not JSON DTO
  targets. Log4j's `JsonUnwrapped` output DTO is not a view-gated input boundary.
- `scripts/run_interactive_demo.py` creates a fresh run, rejects legacy state
  through the installed-runtime copier, explicitly writes
  `region-file-compression=deflate`, creates a new `jvm-tmp` directory and supplies
  `-Djava.io.tmpdir` to installed Java and development Gradle commands.
  `mod/build.gradle` explicitly forwards this location to development game JVMs;
  changing Gradle's own temporary directory alone would not suffice.
- `scripts/companion_app.py` Java discovery is a version/property probe and
  `scripts/installed_mod.py` uses the official Fabric installer; neither hardcodes
  a shared public Java temp directory. The reported LZ4 extraction sink is in the
  subsequent Minecraft JVM, where the managed launcher supplies the private path.

## Private temporary-directory enforcement

The original shared-temp replacement attack is described in the
[upstream LZ4 advisory](https://github.com/yawkat/lz4-java/security/advisories/GHSA-mcr4-qmvw-px4g).
The launcher uses exclusive `mkdir(mode=0o700)` and never reuses an existing
scratch path. On POSIX it verifies current ownership and the absence of group
or other permissions. Windows requires CPython 3.13+ for this route;
[Python's documented mkdir behavior](https://docs.python.org/3.13/library/os.html#os.mkdir)
applies an ACL restricted to the current user and administrators for mode 0700.
This is Windows ACL behavior, not a claim that Windows implements POSIX chmod.
The enclosing fresh session directory is also created with the restrictive mode.
The mitigation does not change ordinary non-interactive Gradle/GameTest JVM
configuration or a player's separately launched Fabric client. Those JVMs must
not be described as protected by this patch. A same-user process or administrator
can already replace this installation and is outside the cross-user boundary.

Native Windows ACL behavior and complete Windows gameplay have not been observed
on this Mac. They remain required Windows candidate qualification checks. The
source regression proves path propagation and fail-closed rejection of Windows
Python versions without the selected documented mechanism; POSIX mode checks are
observed locally. No exploit payload, native-library injection or Minecraft
execution was used to establish these static findings.

## Verification record

Focused launcher/installed-runtime regression run: **155 passed**. The new tests
check an exclusive private scratch directory, POSIX owner-only modes, propagation
to installed and development server/client commands, and rejection of older
Windows Python. Ruff and strict mypy passed for the changed Python implementation.
The complete Gradle/game checks and a fresh OSV scan are recorded separately by
the coordinator; this document does not assume their results.
