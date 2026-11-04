# Security policy

NeuroCraft Fly is being prepared as a local developer alpha. It is not a hardened
public multiplayer server distribution. This policy does not announce publication
or certify a current release as secure.

## Report a vulnerability

The route is the public repository's **Security → Report a vulnerability**
(GitHub private vulnerability reporting,
`https://github.com/evnsnclr/neurocraft-fly-public/security/advisories/new`).
It was enabled on 2026-11-03 and verified the same day: the Security tab shows the
Report a vulnerability entry to a logged-out visitor. Include the affected
revision, component, minimal reproduction, impact and relevant sanitized evidence
there. Do not include credentials or unrelated personal information. Do not open a
public issue with exploit details.

If that private action is unavailable to you, use **Issues → New issue → Private
contact request** to request a confidential security-report destination. Give no
technical exploit details in that public request and wait for a private channel.
No personal email address or other confidential service is implied by this
document. There is no promised response time or bug-bounty program.

## Security boundaries

- Bind development services to loopback by default.
- Authenticate or disable any non-loopback control endpoint.
- Validate protocol versions, message sizes, numeric ranges, session identifiers, and state transitions.
- Never deserialize untrusted pickle, Java native serialization, or executable configuration.
- Verify download checksums and use HTTPS primary sources.
- Do not commit access tokens, Minecraft credentials, cookies, worlds containing private data, or restricted datasets.
- Faithful mode must fail closed to a paused entity when the neural controller disconnects.

Security work targets the latest reviewed development revision. Older snapshots
remain historical evidence rather than maintained release lines. When reporting,
name the exact revision; there is no current long-term support commitment.

## Handling local evidence

Attaching logs is optional. Before sharing, remove tokens, cookies, account
identifiers, personal directory prefixes, private network addresses and world/player
data. Keep the unchanged original privately. Do not upload entire run directories,
launcher profiles, dependency caches, source data or an environment dump.

The scan and advisory sections below preserve their recorded revisions and dates.
They are historical findings and bounded audit decisions, not a new audit of the
public-alpha export. New release checks must retain their own source identity.

## Snapshot security scan

These historical scan receipts are retained privately and are not included in
the curated public source export. Their identifiers and recorded conclusions
remain below; this section is not a new export audit.

`SECURITY-SCAN-01` records registered standard
scan `f4b1864b-6db5-44db-b0bc-fbf9054e0a09` for immutable revision
`6c3a7cbcfc1461c58fa751d5364d76faa3fc5eab`. Coverage was complete for that snapshot;
seven findings remain open (four medium, three low), with no critical/high findings.

The snapshot is immutable historical evidence. All seven of its findings were
remediated and adversarially confirmed at commit `ea9ae95` by
`SECURITY-SCAN-02`, an independent full-tree
review that found no critical, high, or medium issues and four low
consistency gaps in newer tooling (one corrected in the recording commit, three
open with disclosure). The remediations are the mutual-HMAC handshake (ADR 0007,
token never on the wire), fd-bound size gates before reads, pilot work bounds,
digest-over-mapping mmap binding, pilot path containment, incremental
fragmented-frame accounting, and arena-byte binding in pilot v3. Accepted
residuals — the same-inode concurrent mmap writer and the byte-bound acquisition
pipeline's frozen read patterns — are disclosed in the new receipt. Neither scan
supports a production-hardening claim by itself.

## Audited dependency exception

`pyarrow==21.0.0` is retained for the hash-bound MaleCNS acquisition and
derivation environment. `pip-audit` maps it to `PYSEC-2026-113` /
`CVE-2026-25087`, an Arrow C++ IPC-file-reader use-after-free. Apache's
[23.0.1 security release](https://arrow.apache.org/blog/2026/02/16/23.0.1-release/)
states that the issue requires explicitly calling the C++
`RecordBatchFileReader::PreBufferMetadata` API; pre-buffering is disabled by
default. The advisory further states that this functionality is not exposed by
the Python bindings. This repository uses only the Python bindings and contains
no `PreBufferMetadata` or `pre_buffer` call.

The local audit therefore carries one exact exception:
`--ignore-vuln PYSEC-2026-113`. Do not broaden it, treat it as proof that other
Arrow findings are inapplicable, or expose Arrow's C++ IPC pre-buffering path.
Remove the exception when the source-derivation environment is deliberately
re-pinned and all affected data, runtime, trace, benchmark, and reproduction
receipts have been regenerated.

## JVM advisory exceptions

`make jvm-audit` (also a CI step) scans `mod/gradle.lockfile` against OSV.dev,
fail-closed. Of the 213 locked components, 73 Maven-Central components are
scanned; the 140 Mojang-repo, Fabric-maven, and Loom-generated coordinates are
unscannable by OSV and are inventoried, not cleared. The only permitted
exceptions live in `docs/security/jvm-advisory-exceptions.json`: each entry is
an exact advisory/package/version match with a mandatory expiry, a
configurations subset validated against the lockfile, and a justification
naming the classpath scope and the reachability boundary. A stale or expired
entry fails the scan.

The October 24 companion candidate added 15 exact, scoped exceptions, assessed in
[the October 24 JVM triage](docs/security/JVM_TRIAGE_2026-10-24.md). The upstream
package ranges remain affected; managed-launch constraints and private Java
temporary directories are part of that assessment. On 2026-11-03 all 34 entries
were re-triaged for source candidate 0.2.0-rc.6
([the November 3 re-triage](docs/security/JVM_TRIAGE_2026-11-03.md)): OSV was
re-queried, the Mojang 1.21.5 manifest and Loom 1.10.5 pins, the unbundled mod JAR
and the inspected consumers were re-confirmed, and every entry was replaced by a
new entry carrying that re-check and expiring **2027-01-31**. Arbitrary worlds,
extra mods, unmanaged JVMs and shared writable installations require re-triage.
The separately launched player client is controlled by its own launcher.

The rc.3 candidate replaces Guava with **33.7.2-jre**, failureaccess with **1.0.3**,
and the Log4j API/Core/SLF4J2 family with **2.25.5**. This includes developer
classpaths, the isolated installed server and a separately generated client
version for the player's launcher. A prelaunch check verifies actual loaded JAR
hashes and refuses old/mixed selections. The obsolete Guava/Log4j exceptions
are removed; [the rc.2 JVM follow-up](docs/security/JVM_TRIAGE_2026-10-30.md)
remains historical evidence.

Other exact exceptions remain for inherited Jackson, Netty, lz4-java, json-smart,
nimbus-jose-jwt and commons-lang3 components. Their documented input boundaries
and expiries still apply. Two lz4-java advisories have no published upstream fix.
The patched runtime does not establish that every dependency is vulnerability-free.

The packaged Python also uses a [pinned SQLite 3.54.1 rebuild](docs/compliance/SQLITE_RUNTIME.md).
Its source/recipe/native hashes and actual loaded version are checked before
packaging, and the bundled self-test checks the engine after relocation.

Exceptions expire on their recorded dates, 2027-01-31 for all 34 entries after the
November 3 re-triage, and must be re-triaged, not silently renewed; from
2027-02-01 the scan fails until that happens. The
2026-10-24 Circuit Playground verification triage is a private record; its JVM
entries were re-assessed in the companion triages above, and its JNI-extraction
exception still requires explicit owner-only JVM temporary directories, with
loopback alone insufficient. JLine is a Loom development dependency, not a
Mojang-manifest pin. These are scoped reachability/workaround acceptances, not
patched libraries or public/online-use clearance.
