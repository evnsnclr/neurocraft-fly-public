# JVM advisory follow-up — October 30, 2026

The required scan of candidate source b476c0c reported three newly unexcepted
component/advisory instances. This follow-up preserves that failed scan and the
October 24 assessment. It changes no library version or game tuple.

| Scanner instance | Static assessment for the supported route |
| --- | --- |
| Guava 33.3.1-jre / GHSA-xxph-c9ww-hj94 / CVE-2026-102554 | The affected native Java deserialization entry is not exposed by the inspected project, Minecraft, Mojang-library or MSAL routes. The library remains affected. |
| Log4j Core 2.24.1 / GHSA-h383-gmxw-35v2 / CVE-2026-34479 | Apache identifies the separate log4j-1.2-api bridge and its Log4j1XmlLayout as the required component/configuration. Neither is in the locked runtime. |
| Log4j Core 2.8.1 / the same advisory | Only loomDevelopmentDependencies; the bridge/layout is likewise absent from the locked artifacts. This is not a shipped mod dependency. |

These are **not actionable within the inspected bounded route**, rather than
upstream repairs or an assertion that the entire dependency tree is safe.
Three exact advisory/package/version/classpath exceptions expire **2026-11-13**.
Existing entries retain their original expiry; nothing is silently renewed.

## Evidence and boundary

The [Guava owning advisory](https://github.com/google/guava/security/advisories/GHSA-xxph-c9ww-hj94)
requires ObjectInputStream reading attacker-controlled native Java bytes and can
allocate excessive heap from a collection capacity. Guava 33.7.2 fixes it.
Project Java has no ObjectInputStream/readObject entry. Static class-pool
inspection found no ObjectInputStream reference in mapped Minecraft common/client,
the inspected Mojang libraries or MSAL 1.17.2. Those Minecraft JAR hashes were
486a2ac94dfe672e53b4b248ad68fc8780c5bf304abc170cb4f27e3a5f7df97c and
f26ad0101c57b403d08ebae462bb20fed942f1b1783d97cd14c8bbe82ea401f9.

Log4j does contain native serialization helpers. The inspected Loom logging
configuration uses PatternLayout and no serialized-log receiver; it does not
establish an attacker stream route to those helpers. The model bridge uses its
bounded JSON protocol, and managed sessions create a fresh world on a loopback
game server with query and RCON disabled. The acquisition/runtime copier rejects
imported world/log state. These are source/configuration facts, not proof about
all reflective or transitive paths.

[Apache's owning advisory](https://logging.apache.org/security.html#CVE-2026-34479)
identifies log4j-1.2-api, fixed in 2.25.4, rather than an unconditional Core
logging defect. The lock contains no bridge artifact, project/configuration has
no affected layout selection, and inspected Core/API/SLF4J artifacts contain no
Log4j1XmlLayout class. Core 2.24.1's SHA-256 is
00bcf388472ca80a687014181763b66d777177f22cbbf179fd60e1b1ac9bc9b0.

SECURITY.md forbids untrusted Java native serialization. These exceptions apply
only to the declared local showcase and private development checks. Imported
worlds, additional mods, arbitrary servers, custom logging/serialization
configuration, changed dependencies or new database/network consumers require
re-triage. The player's external launcher/client remains under its owner.
No exploit reproduction, complete call graph or public-platform qualification
is claimed by this static assessment.
