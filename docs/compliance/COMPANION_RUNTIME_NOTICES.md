# Companion runtime inventory and necessity

The private 0.2.0-rc.3 companion uses no new application framework or production
Python dependency. Its browser UI, HTTP server, download verification, archive
handling and process interface use the standard library. The pinned production
NumPy, PyArrow and websockets packages retain their existing lockfile hashes and
licenses; only hashed wheels are installed into a bundle, with their notices.

The packaged interpreter is CPython **3.13.18**, Astral python-build-standalone
build **20261026**, with a [reproducible SQLite 3.54.1 rebuild](SQLITE_RUNTIME.md).
Exact per-platform base archive URLs, sizes and SHA-256 values are
in `release/companion-runtime-pins.json`, obtained from the owning GitHub release
and checksum sidecars. Preserve Python's `LICENSE.txt` and all included component
license material. PSF licensing does not replace OpenSSL, libffi, compression
library or other component terms. Bundle inventories bind the actual supplied
runtime prefix; a separate verified download receipt binds the upstream archive.
Do not infer upstream authenticity from a locally supplied prefix alone.

The upstream `install_only` archive omits the full archive's component notices.
`release/runtime-legal/` supplies each platform's original `PYTHON.json` and every
license path it references, copied from its hash-verified matching full archive.
The inventory records full-archive URLs, SHA-256 values, file hashes and sizes.
Every referenced notice is present in the matching full archives; the earlier
candidate's supplemental zlib-ng acquisition is no longer needed.
The bundler refuses missing or changed component notices. These upstream terms
remain separate from the project's MIT license; metadata may conservatively
reference multiple library variants. See the owning
[distribution format](https://docs.astral.sh/python-build-standalone/distributions/)
and [build inputs](https://github.com/astral-sh/python-build-standalone/blob/20261026/pythonbuild/downloads.json).

Java 21 is required for the isolated Fabric server. The app first probes a
user-selected or installed runtime and reads its actual `java.home`; launcher
shims are not Java home directories. Where needed, it obtains the pinned Eclipse
Temurin JRE archive directly for the user, retaining its included `legal` material
and GPLv2 + Classpath Exception terms. Java is acquired during setup rather than
included in the companion ZIP. The existing Fabric installer/API pins and
Minecraft terms remain as documented in `THIRD_PARTY.md`.

The Python distribution, Eclipse Temurin and the existing dependencies are
actively maintained upstream projects; this version intentionally pins its
runtime inputs for reproducibility. Recheck advisories and platform compatibility
before a public candidate. The [rc.2 scoped assessment](../security/RUNTIME_ASSESSMENT_2026-10-30.md)
retains its historical scope; rc.3 replaces its SQLite/Guava/Log4j versions.
Other documented component limits remain. Downloads enforce their declared sizes/hashes and
approved HTTPS redirect hosts; extraction rejects escaping paths and links.
The app creates a separate patched client version when installing client files;
it does not overwrite existing launcher profiles, accounts or global PATH, and does not
send automatic telemetry. Reports are generated only at the player's request.

Native Windows threading uses the OS API and adds no pthread runtime dependency.
Private CI uses pinned official GitHub/Astral actions, read-only repository
permissions and a private-repository guard. Artifact upload does not publish a
GitHub release. Signing, public availability and independent platform testing
are separate release decisions/evidence.
