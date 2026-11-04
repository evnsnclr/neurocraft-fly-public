# JVM advisory re-triage — November 3, 2026

Scope: all 34 entries of `docs/security/jvm-advisory-exceptions.json`, re-triaged
for source candidate 0.2.0-rc.6. Fifteen entries from
[the October 24 triage](JVM_TRIAGE_2026-10-24.md) were due to expire on 2026-11-24
and the 19 entries introduced with the fail-closed scan on 2026-08-18 on 2026-12-31; from
2026-11-25 `make jvm-audit`, `make verify` and the CI workflow of the published
candidate 0.2.0-rc.5 fail on any push or pull request. This record states what was
re-checked, finding by finding, and the new expiry. It is a static re-triage with
fresh justification, as `SECURITY.md` requires: every entry was replaced by a new
entry whose justification records this re-check; no entry was renewed by editing
its date, none was broadened, none was dropped, and no library version, lockfile,
verification hash or game tuple changed. It is not an exhaustive security review,
a library upgrade, an exploit reproduction, Windows qualification or authorization
to publish.

## What was re-checked

1. **OSV.** The fail-closed scan ran live against `api.osv.dev` on 2026-11-03,
   first with the expiring ledger (ledger SHA-256
   `ec8c6e3d61de417818388b92f3e13c56e5e1ca68a73dfe7f6c81cf826cd4e75b`, report
   SHA-256 `ca05ec594ded7a8c5d7be724ff3f7cbbd734d8c1cc387943bc351847927eff47`),
   then with the rewritten one (ledger SHA-256
   `32eac872572c2fe3ae5b501c02c6bacf1597a3eff75d266f9fb067e8a43d66b9`, report
   SHA-256 `232a6429bf2db68c4308abe600424e00af1267b68bfc5097162c4058aa4d0357`).
   Both passed: 73 Maven-Central components scanned, 140 unscannable coordinates
   inventoried, the same 34 advisories across the same 9 components, each matched
   by one exact entry whose aliases OSV confirms. Lockfile SHA-256
   `64377a8f19d2d385b288db08c07c6d3b8d100b2068d97be7c89d2a8a24f86dcf`. The fixed
   versions OSV lists per coordinate and each record's last-modified date are in
   the table below; for the 15 October 24 entries they equal what those entries
   already stated, and the org.lz4 coordinate still has no fixed version for the
   five October advisories and the two coordinator-triaged ones that said so.
2. **Pins.** The cached Mojang version document for 1.21.5 (piston-meta package
   `5ca2012338100e4dbfef0d2114a5fa66959ab1e5`, released 2025-03-25, cached copy
   SHA-256 `ea7ff2dbea92fba08b1a7d15412d912d9bcac68c610fe968882c512a8e934196`)
   still lists jackson-core 2.13.4, jackson-databind 2.13.4.2, netty-codec and
   netty-handler 4.1.118.Final, lz4-java 1.8.0, nimbus-jose-jwt 9.40, json-smart
   2.5.1, commons-lang3 3.17.0, msal4j 1.17.2 and authlib 6.0.58, the versions in
   `mod/gradle.lockfile`. JLine 3.20.0 is still resolved only by Loom 1.10.5
   (`mod/gradle.properties`) through terminalconsoleappender 1.3.0, whose POM
   declares jline-reader: it sits on `loomDevelopmentDependencies` and on the four
   runtime classpaths Loom launches development and GameTest JVMs with, never on a
   compile or `minecraft*` configuration; it is not a Mojang-manifest pin. The lockfile was last changed by the rc.3 patch commit
   `75fe931` (Guava, failureaccess and the Log4j family only); every coordinate
   in this ledger has the same lock line it had at the October 24 triage.
3. **Shipped classpath.** `mod/build.gradle` declares no jar-in-jar `include`.
   The candidate 0.2.0-rc.5 production JAR (`neurocraft-fly-0.2.0-rc.5.jar`,
   SHA-256 `e75ada15…b116`) contains 166 classes, all under `dev.neurocraft`, plus
   `fabric.mod.json`, `LICENSE`, `NOTICE`, the language file, the texture and the
   runtime-pins resource, and no third-party library; the rc.6 JAR is inspected the
   same way in the candidate record. Every affected library reaches a running game
   only through Minecraft's own libraries or Loom's development classpaths.
4. **Project source.** A token search over `mod/src` (main, client, test and
   GameTest) finds no reference to `com.fasterxml.jackson`, `net.jpountz`,
   `org.lz4`, `org.apache.commons.lang3`, `com.nimbusds`, `net.minidev`,
   `org.jline`, `ObjectMapper`, `SslContext`, `SniHandler`, `LZ4` or
   `HISTORY_IGNORE`. The only Netty reference is
   `io.netty.channel.embedded.EmbeddedChannel`, imported by four GameTests that
   build in-process connections. The neural bridge (`NeuralBridge.java`) is a
   `java.net.http.WebSocket` client over the loopback-only authenticated socket.
5. **Inspected class pools.** The Loom-mapped Minecraft 1.21.5 jars are the same
   bytes the October 30 follow-up inspected: common SHA-256
   `486a2ac94dfe672e53b4b248ad68fc8780c5bf304abc170cb4f27e3a5f7df97c` (6,557
   classes) and client-only SHA-256
   `f26ad0101c57b403d08ebae462bb20fed942f1b1783d97cd14c8bbe82ea401f9` (2,591
   classes). A constant-pool scan of both finds no `com/fasterxml/jackson/`,
   `io/netty/handler/ssl/SniHandler`, `SslClientHelloHandler`, `SslContext`,
   `io/netty/handler/ipfilter/IpSubnetFilterRule`,
   `io/netty/handler/codec/compression/`, `net/jpountz/lz4/LZ4FrameInputStream`,
   `LZ4DecompressorWithLength`, `net/jpountz/xxhash/`,
   `org/apache/commons/lang3/ClassUtils`, `com/nimbusds/`, `net/minidev/json/`,
   `org/jline/`, `HISTORY_IGNORE` or `java/io/ObjectInputStream` token; the one
   `net/jpountz/lz4/LZ4BlockInputStream` reference is
   `net.minecraft.world.level.chunk.storage.RegionFileVersion`. In the locked
   `lz4-java-1.8.0.jar`, `javap` shows `LZ4BlockInputStream(InputStream)`
   delegating to `(InputStream, LZ4FastDecompressor)`, which calls the four-argument
   constructor with `stopOnEmptyBlock` pushed as `iconst_1` (true). MSAL4j 1.17.2
   (SHA-256 `797530d79d750700da2ecb70a27718d747e59c44ed3829a6d18f06c915721e1b`,
   247 classes) has 23 Jackson-referencing classes and no `JsonView`,
   `JsonIdentityInfo`, `JsonTypeInfo`, `JsonUnwrapped`, `activateDefaultTyping`,
   `enableDefaultTyping`, `PolymorphicTypeValidator` or `java/io/DataInput` token;
   `java/net/InetAddress` appears only in `HttpListener` and `InteractiveRequest`,
   `java/nio/file/Path` only in `AzureArcManagedIdentitySource`, and
   `ObjectMapper` in `ClaimsRequest`, `JsonHelper` and `TokenCache`, as the
   October 24 assessment described. authlib 6.0.58 (SHA-256
   `7bea5444e83c8d343e11fb9e45939721f0db4321c5056ac846072e4a0bbe1321`, 94 classes)
   references none of the tokens.
6. **Launcher mitigations.** `scripts/run_interactive_demo.py` still creates the
   run directory and a `jvm-tmp` directory with mode 0700, writes
   `region-file-compression=deflate` into the fresh server's properties and passes
   `-Djava.io.tmpdir=<run>/jvm-tmp` to both server and client JVMs;
   `mod/build.gradle` still forwards a private `java.io.tmpdir` to the game JVMs
   it launches. Windows ACL behavior remains an unobserved platform check.

## Decision and limits

Every entry is re-issued with the same exact advisory, package, version and
configuration scope and a new expiry of **2027-01-31**; the first day the scan
fails again is 2027-02-01. One cohort replaces the two earlier ones so that the
next re-triage is one event before the next candidate. The 15 October 24 entries
keep that triage as their assessment evidence and this record as the re-triage;
the 19 earlier entries keep their 2026-08-18 reasoning with this record as the
re-triage. These remain scoped reachability and mitigation acceptances for the
private offline GameTest, the managed loopback demo and the source-built developer
alpha: not patched libraries, not online or public-multiplayer clearance, and not
proof about every reflective or transitive path. Imported worlds, extra mods,
arbitrary servers, custom logging, serialization or compression configuration,
unmanaged JVMs, shared writable installations, changed dependencies or new
consumers require re-triage before the expiry.

## Finding-by-finding re-check

OSV facts are as returned on 2026-11-03. "Unchanged" means the necessary API,
configuration or input boundary that the earlier assessment identified is still
absent from the project source, the inspected class pools and the launcher paths
listed above.

| Advisory | Locked component | First triage | OSV on 2026-11-03 (fixed; record modified) | Re-check | Expiry |
|---|---|---|---|---|---|
| GHSA-7hhh-6rmp-j9qf / CVE-2026-89425 | jackson-core 2.13.4 | 2026-10-24 | 2.18.11, 2.21.7, 2.22.3; 2026-10-01 | Unchanged: no DataInput-backed Jackson parser in MSAL, Minecraft or project code | 2027-01-31 |
| GHSA-h46c-h94j-95f3 / CVE-2025-52999 | jackson-core 2.13.4 | 2026-08-18 | 2.15.0; 2026-09-10 | Unchanged: the project constructs no Jackson parser; protocol frames use the mod's bounded reader | 2027-01-31 |
| GHSA-r7wm-3cxj-wff9 / CVE-2026-68494 | jackson-core 2.13.4 | 2026-08-18 | 2.18.8, 2.21.4; 2026-10-02 | Unchanged: no NonBlockingJsonParser or any Jackson parser in project code | 2027-01-31 |
| GHSA-3wrr-7qpf-2prh / CVE-2026-50193 | jackson-databind 2.13.4.2 | 2026-08-18 | 2.14.0; 2026-09-10 | Unchanged: no Jackson data binding or JsonNode in project code | 2027-01-31 |
| GHSA-5gvw-p9qm-jgwh / CVE-2026-59889 | jackson-databind 2.13.4.2 | 2026-10-24 | 2.18.9, 2.21.5, 2.22.1; 2026-10-19 | Unchanged: no JsonView or JsonUnwrapped token in MSAL; none in project code | 2027-01-31 |
| GHSA-5jmj-h7xm-6q6v / CVE-2026-54515 | jackson-databind 2.13.4.2 | 2026-08-18 | 2.18.9, 2.21.5, 2.22.1, 3.1.4; 2026-09-10 | Unchanged: databind is reached only by Mojang/Microsoft sign-in code that the offline and loopback paths never exercise | 2027-01-31 |
| GHSA-cxp5-3px4-pw24 / CVE-2026-91777 | jackson-databind 2.13.4.2 | 2026-10-24 | 2.18.11, 2.21.7, 2.22.3; 2026-09-30 | Unchanged: no JsonIdentityInfo token in MSAL; none in project code | 2027-01-31 |
| GHSA-gx83-3vf8-gh7j / CVE-2026-83557 | jackson-databind 2.13.4.2 | 2026-10-24 | 2.18.10, 2.21.6, 2.22.2; 2026-09-28 | Unchanged: no JsonTypeInfo, default typing or PolymorphicTypeValidator token in MSAL; none in project code | 2027-01-31 |
| GHSA-hgj6-7826-r7m5 / CVE-2026-54514 | jackson-databind 2.13.4.2 | 2026-08-18 | 2.18.8, 2.21.4, 3.1.4; 2026-09-10 | Unchanged: no Jackson binding in project code; offline and loopback paths never authenticate | 2027-01-31 |
| GHSA-j3rv-43j4-c7qm / CVE-2026-54512 | jackson-databind 2.13.4.2 | 2026-08-18 | 2.18.8, 2.21.4, 3.1.4; 2026-10-28 | Unchanged: no ObjectMapper, default typing or PolymorphicTypeValidator in project code; MSAL enables none | 2027-01-31 |
| GHSA-rmj7-2vxq-3g9f / CVE-2026-54513 | jackson-databind 2.13.4.2 | 2026-08-18 | 2.18.8, 2.21.4, 3.1.4; 2026-09-11 | Unchanged: no BasicPolymorphicTypeValidator configuration anywhere inspected | 2027-01-31 |
| GHSA-vvgp-rfg2-7rr6 / CVE-2026-77310 | jackson-databind 2.13.4.2 | 2026-10-24 | 2.18.9, 2.21.5, 2.22.1; 2026-09-28 | Unchanged: MSAL's InetAddress uses are its local listener and interactive request, not JSON targets | 2027-01-31 |
| GHSA-wjgm-6hv5-3cvf / CVE-2026-19032 | jackson-databind 2.13.4.2 | 2026-10-24 | 2.18.10, 2.21.6, 2.22.2; 2026-09-28 | Unchanged: MSAL's Path use is Azure Arc file I/O, not a JSON target | 2027-01-31 |
| GHSA-wv8q-qhhj-9h54 / CVE-2026-91776 | jackson-databind 2.13.4.2 | 2026-10-24 | 2.18.11, 2.21.7, 2.22.3; 2026-09-30 | Unchanged: no name-based polymorphism or default typing in MSAL or project code | 2027-01-31 |
| GHSA-xwmg-2g98-w7v9 / CVE-2025-53864 | nimbus-jose-jwt 9.40 | 2026-08-18 | 9.37.4, 10.0.2; 2026-09-10 | Unchanged: reached only through msal4j sign-in; no com.nimbusds reference in project code | 2027-01-31 |
| GHSA-3p8m-j85q-pgmj / CVE-2025-58057 | netty-codec 4.1.118.Final | 2026-08-18 | 4.1.125.Final; 2026-09-10 | Unchanged: no Netty decompression decoder token in Minecraft class pools or project code | 2027-01-31 |
| GHSA-558v-64gr-wgg4 / CVE-2026-59901 | netty-codec 4.1.118.Final | 2026-08-18 | 4.1.136.Final; 2026-09-10 | Unchanged: no Bzip2Decoder (no compression-codec token) in the inspected consumers | 2027-01-31 |
| GHSA-mj4r-2hfc-f8p6 / CVE-2026-42583 | netty-codec 4.1.118.Final | 2026-08-18 | 4.1.133.Final; 2026-09-10 | Unchanged: no Lz4FrameDecoder (no compression-codec token) in the inspected consumers | 2027-01-31 |
| GHSA-3qp7-7mw8-wx86 / CVE-2026-44249 | netty-handler 4.1.118.Final | 2026-08-18 | 4.1.135.Final, 4.2.15.Final; 2026-09-10 | Unchanged: no IpSubnetFilterRule token in Minecraft class pools or project code | 2027-01-31 |
| GHSA-c4c3-7fpv-j4q5 / CVE-2026-75595 | netty-handler 4.1.118.Final | 2026-10-24 | 4.1.137.Final, 4.2.17.Final; 2026-09-08 | Unchanged: no SniHandler or SslClientHelloHandler token; GameTests use EmbeddedChannel only | 2027-01-31 |
| GHSA-c653-97m9-rcg9 / CVE-2026-50010 | netty-handler 4.1.118.Final | 2026-08-18 | 4.1.135.Final, 4.2.15.Final; 2026-09-10 | Unchanged: no SslContext token in Minecraft class pools or project code | 2027-01-31 |
| GHSA-fccg-mwvh-qqg4 / CVE-2026-75596 | netty-handler 4.1.118.Final | 2026-10-24 | 4.1.137.Final, 4.2.17.Final; 2026-09-08 | Unchanged: same absent pre-handshake SNI aggregation path | 2027-01-31 |
| GHSA-x4gw-5cx5-pgmh / CVE-2026-45416 | netty-handler 4.1.118.Final | 2026-08-18 | 4.1.135.Final, 4.2.15.Final; 2026-09-10 | Unchanged: no server-side SniHandler; no public TLS listener | 2027-01-31 |
| GHSA-pq2g-wx69-c263 / CVE-2024-57699 | json-smart 2.5.1 | 2026-08-18 | 2.5.2; 2026-09-10 | Unchanged: transitive under nimbus-jose-jwt in the sign-in chain; one MSAL class references it; none in project code | 2027-01-31 |
| GHSA-j288-q9x7-2f5v / CVE-2025-48924 | commons-lang3 3.17.0 | 2026-08-18 | 3.18.0; 2026-09-10 | Unchanged: no ClassUtils token in Minecraft class pools; no commons-lang3 call in project code | 2027-01-31 |
| GHSA-5q95-hrpc-m3w3 / CVE-2026-77420 | jline-reader 3.20.0 | 2026-10-24 | 3.30.15, 4.3.1; 2026-09-23 | Unchanged: Loom development-console classpaths only; no HISTORY_IGNORE anywhere inspected | 2027-01-31 |
| GHSA-343h-94h5-c4wr / CVE-2026-106449 | lz4-java 1.8.0 | 2026-10-24 | none for org.lz4; 2026-10-07 | Unchanged: RegionFileVersion uses the one-argument constructor, which reaches stopOnEmptyBlock=true; managed worlds select deflate | 2027-01-31 |
| GHSA-4v53-57pg-c464 / CVE-2026-106452 | lz4-java 1.8.0 | 2026-10-24 | none for org.lz4; 2026-10-07 | Unchanged: fresh managed worlds with deflate; imported worlds excluded | 2027-01-31 |
| GHSA-6cx8-rjf8-pr8g / CVE-2026-106453 | lz4-java 1.8.0 | 2026-10-24 | none for org.lz4; 2026-10-07 | Unchanged: no LZ4DecompressorWithLength token anywhere inspected | 2027-01-31 |
| GHSA-cmp6-m4wj-q63q / CVE-2025-66566 | lz4-java 1.8.0 | 2026-08-18 | none for org.lz4; 2026-09-10 | Unchanged: locally produced chunk data only in the offline and loopback paths | 2027-01-31 |
| GHSA-gm45-99xc-r7wv / CVE-2026-106450 | lz4-java 1.8.0 | 2026-10-24 | none for org.lz4; 2026-10-07 | Unchanged: no LZ4FrameInputStream token anywhere inspected | 2027-01-31 |
| GHSA-mcr4-qmvw-px4g / CVE-2026-106451 | lz4-java 1.8.0 | 2026-10-24 | none for org.lz4; 2026-10-07 | Unchanged: private mode-0700 `jvm-tmp` and `-Djava.io.tmpdir` in both managed launch paths and in Gradle's game JVMs; unmanaged JVMs excluded; Windows ACL unobserved | 2027-01-31 |
| GHSA-vqf4-7m7x-wgfc / CVE-2025-12183 | lz4-java 1.8.0 | 2026-08-18 | 1.8.1; 2026-09-10 | Unchanged: upstream 1.8.1 exists and only Mojang's pin bars it; locally produced LZ4 input only | 2027-01-31 |
| GHSA-xx22-p4ch-683r / CVE-2026-59949 | lz4-java 1.8.0 | 2026-08-18 | none for org.lz4; 2026-09-10 | Unchanged: no xxhash token in Minecraft class pools; no lz4-java call in project code | 2027-01-31 |

## Verification record

After the rewrite: `make jvm-audit` passed live on 2026-11-03 with the receipt
values in item 1; `tests/security` (53 tests, including the one that loads the
real ledger against the real lockfile) passed; `make audit` (`pip-audit` with its
one documented exception) passed. The candidate's fresh-extraction `make verify`
and CI runs are recorded in the candidate record, not here.
