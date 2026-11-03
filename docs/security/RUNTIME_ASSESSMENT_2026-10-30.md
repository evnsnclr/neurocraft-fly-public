# Private runtime assessment — October 30, 2026

Candidate 0.2.0-rc.2 upgrades CPython to 3.13.18 / python-build-standalone
(PBS) 20261026, uv to 0.13.6 and the Windows Temurin download to 21.0.13+9,
matching the other configured Java downloads. Production Python package pins
and wheel hashes are unchanged. Windows x64 and Apple Silicon are private
player-test targets; Linux configuration does not establish player support.

## Relevant fixes and evidence

[Python 3.13.18](https://www.python.org/downloads/release/python-31318/)
blocks the compromised eMudhra intermediate keys and includes Expat and temporary
directory fixes. Post-3.13.5 archive fixes matter to setup's ZIP/TAR handling.
[uv's certificate advisory](https://github.com/astral-sh/uv/security/advisories/GHSA-hq85-5cpj-mr62)
affects the old 0.7.8 pin; 0.13.6 is beyond its 0.13.4 fixed boundary.
The [PBS release](https://github.com/astral-sh/python-build-standalone/releases/tag/20261026)
supplies the matching Python archives. Downloads were checked against upstream
SHA-256 values and exact sizes; full-archive metadata/notices retain original bytes.

The local Mac interpreter reports Python 3.13.18, OpenSSL 3.5.9,
Expat 2.9.0, SQLite 3.53.1 and OS-linked zlib 1.2.12. Native bundle self-tests
record these loaded versions per platform; Mac results cannot qualify Windows.
Original Windows metadata contains old OpenSSL DLL names despite the
[build selecting OpenSSL 3.5](https://github.com/astral-sh/python-build-standalone/blob/20261026/cpython-windows/build.py).
Preserve that metadata; use actual native probes rather than inferring loaded versions.

[OpenSSL 3.5.9](https://www.openssl-library.org/news/vulnerabilities-3.5/index.html),
[Expat 2.9.0](https://github.com/libexpat/libexpat/releases/tag/R_2_9_0) and
[XZ 5.8.4](https://github.com/tukaani-project/xz/releases/tag/v5.8.4)
include the reviewed upstream fixes. XZ is identified by versioned build inputs;
Python's lzma API supplies no native version probe.

The required audit also found CVE-2026-114613 in development-tool MessagePack
1.2.1. The C decoder retained about 4 MiB after sixteen rejected 256-KiB values
through each of its two decoding APIs. The lock changes only MessagePack to
[the fixed 1.2.3 release](https://github.com/msgpack/msgpack-python/releases/tag/v1.2.3);
the same probes retained under 2 KiB and ordinary decoding still worked.
The actual CacheControl malformed-cache path retained 1,333 bytes across sixteen
rejected entries, and an ordinary response/status/header/body round-trip passed.
This is a development audit-cache path, not the neural runtime: response bodies
are serialized as byte values in a string-keyed envelope, so a malformed-key
trigger needs a crafted/corrupted cache envelope. The scan finding and reproduced
1.2.1 behavior are separate from upstream's unresolved earliest-affected-version
range. Production dependencies are unchanged; MessagePack needed no new exception.

The complete research-group audit additionally identified CVE-2026-116577,
CVE-2026-115927 and CVE-2026-116578 in optional Tornado 6.5.9. The lock updates
only that package to [the fixed 6.6 release](https://www.tornadoweb.org/en/stable/releases/v6.6.0.html).
Its Python 3.11 minimum fits this project's supported range, and the Jupyter
parents accept it. The production Companion excludes Tornado/Jupyter. Focused
upstream parsing/redirect tests and actual fresh-kernel notebook execution are
the compatibility checks; no additional audit exception is used.

## Scoped retention and qualification limits

**SQLite is not patched for CVE-2026-11822/11824.** The
[owning assessment](https://www.sqlite.org/cves.html) identifies 3.53.2 as
fixed and requires attacker SQL, FTS5 and disabled defensive mode.
Inspection found no SQL/database entry point in the packaged Companion, setup,
launcher or neural service. Retention is accepted only for that route; adding
database consumers or accepting external SQL/databases requires re-assessment.
Older optional Tcl/Tk, ncurses and Berkeley DB inputs are dormant-component
limits, not claims that every library is current or vulnerability-free.

Setup uses default TLS verification, approved redirect hosts, exact download
hashes/sizes, bounded extraction and cancellation. The production HTTP interface
uses loopback, Host/Origin validation, fresh bearer authorization and bounded
fixed-route requests. These controls constrain inputs; they do not replace patches.
Keep the OS and a user-selected Java runtime current. The Java probe checks
major version 21, not every vendor build's security patch level.

Focused tests reject old Python, mismatched PBS notices, changed source/notices
and foreign kernels; existing tests cover malicious archives/downloads.
Relocated native tests exercise imports and unfused sparse arithmetic.
The builder inventories a supplied prefix and deliberately reports
`upstream_authenticity_verified: false`; a separate verified-download receipt
binds the upstream archive. An inventory alone does not authenticate its origin.

The [separate JVM follow-up](JVM_TRIAGE_2026-10-30.md) records three new exact,
expiring reachability exceptions. This runtime assessment does not establish
exhaustive advisory coverage, full-network Windows calibration, independent
installation, sustained human gameplay or representative footage. The public
playable release remains on hold. Final check results belong to the exact
frozen candidate and its release notes.
