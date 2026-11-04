# Build and run from source

**This is the Play route of the 0.2.0-rc.6 source release:** build the mod and the
local neural service from source on Apple Silicon macOS. The packaged Companion in
[the Companion guide](COMPANION.md) is private testing material outside this release.
Do not mix source, mod, model or native-kernel files from different candidates.

Run commands from the intended source revision's repository root.
On macOS/Linux, extract the source ZIP with an archive utility that preserves
executable permissions, such as `unzip`. The verifier checks executable scripts
as well as file bytes; Python's `zipfile.extractall()` alone loses these bits.

For an unaltered curated export, verify its inventory first:

```sh
python3 -I scripts/build_public_release.py --verify-source .
```

This detects missing/changed included bytes; it does not authenticate a publisher.
Project software uses [MIT](../LICENSE); Minecraft and third-party material keep
[their separate terms](../THIRD_PARTY.md).

## Platform and tools

Apple Silicon macOS has recorded full-network development runs and is the only
platform this release describes; installation and automated comparison runs are
so far qualified only on the development Mac, for the earlier candidate `c1ff8b1`.
Native Windows portability is implemented for private testing, with full-network
calibration, performance and independent gameplay still to qualify, and it is not
a supported platform of this release. Linux is experimental; fixture CI is not
evidence of full-network manual gameplay on any platform.

Use the Python version in `.python-version` (3.13.18), uv **0.13.6** exactly
(`pyproject.toml` sets `required-version`, so another uv refuses to run; install
it with the standalone installer for that version,
`https://astral.sh/uv/0.13.6/install.sh`, or with `pip install uv==0.13.6` inside
a virtual environment, since a bare `pip install` fails on Homebrew's externally
managed Python; then let uv fetch the pinned Python),
Java 21, a supported C compiler, Git and Make. The
Gradle wrapper supplies its pinned Gradle version. A source build requires a
JDK, and Node.js 18 or newer only for the Companion UI tests that `make verify`
includes; playing from source needs neither Node.js nor the Companion.

```sh
python3 --version
uv --version
java -version
python3 -I scripts/public_alpha.py doctor
python3 -I scripts/public_alpha.py setup
```

`doctor` is read-only. `setup` prints the ordered commands; it does not execute
them. Follow that revision's printed plan rather than copying commands from a
different candidate. If Java 21 is not the default, select its JDK with `JAVA_HOME`.

These shell examples use POSIX syntax. Windows contributors can use the
corresponding Python executable and `mod/gradlew.bat` with a supported native
toolchain, with no qualification claim; the private [packaged route](COMPANION.md)
is not part of this release.

## Prepare dependencies and model

Follow the setup plan in order:

1. Synchronize the frozen production Python workspace. The project Python pin is
   selected from `.python-version`. `make sync` also adds contributor/test tools.
2. Acquire and verify the source tables in
   `data/manifests/male-cns-v1.0.sources.json`. Full interactions need annotations,
   neurotransmitters and weights, not the tiny tutorial fixture.
3. Build `data/processed/male-cns-v1.0-superclass-v1/` using the printed derivation
   command and a fresh receipt path. The builder refuses an existing graph output;
   preserve verified artifacts rather than deleting them to bypass a refusal.
4. Compile the native kernel using the supported build route. On POSIX, use
   `make spmv-build`. Keep its generated identity sidecar with the library.
5. Read the Minecraft EULA and record your own acceptance in the indicated local
   file if you agree. Another machine's acceptance does not transfer.

See [data provenance](DATA_PROVENANCE.md) for acquisition and derivation details.
Source downloads total about 1.1 GB and the derived graph about 591 MiB. Historical
derivation memory is not simultaneous game memory; leave extra room for Python,
Java, source tables and caches. Measure total gameplay use on the actual candidate
before recommending hardware.

A new native build has its own identity. Historical calibration `measured_under`
identities describe earlier measurements; reusing their scales does not establish
a new calibration. Keep graph/readout/calibration bindings coherent. Never edit
hashes or disable dependency verification to get past a mismatch.

## Run a normal Fabric client

The pinned game tuple is **Minecraft 1.21.5 / Fabric Loader 0.16.10 / Fabric API
0.128.2+1.21.5**. Build the production mod and stage a new private runtime:

```sh
cd mod
./gradlew --no-daemon --dependency-verification strict build
cd ..
python3 -I scripts/installed_mod.py --output artifacts/installed-runtime --mod-jar mod/build/libs/neurocraft-fly-0.2.0-rc.6.jar
```

Use the candidate's actual versioned production JAR, not the sources JAR. The
helper downloads the pinned official installer/dependencies, inventories the
runtime and refuses an existing output. It does not accept the EULA or edit
launcher profiles. **Do not distribute the resulting runtime: it contains Minecraft.**

Create an isolated game folder through your own launcher/account.
Copy the two JARs in `artifacts/installed-runtime/server/mods/` into its otherwise
empty `mods` directory. Generate the matching patched launcher version:

```sh
python3 -I scripts/java_runtime.py --installed-runtime artifacts/installed-runtime --launcher-root "/path/to/minecraft-launcher-folder" --mods "/path/to/isolated-client/mods"
```

The launcher folder contains `versions` and `libraries`; it can differ from the
isolated game folder. Select the printed `neurocraft-fly-1.21.5-…` version in your
normal launcher's installation settings, with the isolated game directory.
This creates a new version and pinned libraries without editing existing profiles
or account files. The mod refuses unpatched or modified loaded Java libraries
before gameplay. Start with GUI scale 2 and a 1280×720 or larger window.
Replace the example mods path and player name below with your own:

```sh
python3 -I scripts/public_alpha.py doctor --installed-runtime artifacts/installed-runtime --client-mods-dir "/path/to/isolated-client/mods"
python3 -I scripts/public_alpha.py prepare --installed-runtime artifacts/installed-runtime --manual-client-name YourJavaName --client-mods-dir "/path/to/isolated-client/mods"
python3 -I scripts/public_alpha.py run --installed-runtime artifacts/installed-runtime --manual-client-name YourJavaName --client-mods-dir "/path/to/isolated-client/mods"
```

`prepare` creates a session plan; `run` starts the local server/model. The checker
compares the runtime and client to this source directory's built JAR. Renaming a
JAR is allowed, replacing it with different bytes is refused. The generated
client version and its patched library bytes are also checked. Omitting the
client mods argument leaves that installation unverified; the in-game library
guard still applies. No account files are inspected.

Join the **newly printed local address** in the generated NeuroCraft client version. Wait for
**Live · Original weights · Readout control**, then try the
[first interaction](EXPERIMENTS.md#your-first-interaction). This launcher does not
manage your normal client, settings or authentication. The earlier same-host
production-client smoke used cached assets and an offline QA identity; it did
not qualify independent installation or the official account flow.

## Development client

For the managed development client instead:

```sh
python3 -I scripts/public_alpha.py doctor
python3 -I scripts/public_alpha.py prepare
python3 -I scripts/public_alpha.py run
```

This uses `full_mcns_interactions_program_v2`, original/shuffled/no-input lanes
and shuffle seed 7. Add `--circuit-playground` to `prepare` or `run` for the close-up
scene. The option also works with the installed-client arguments. It changes
staging, not neural dynamics or condition. Requested 20 TPS / 60 FPS settings are
not achieved-performance measurements. The launcher uses authenticated loopback
services; do not port-forward them or treat the internal protocol as a public API.

## Stop, restart and retain evidence

Disconnect the installed client or press Ctrl-C in the launcher and wait for its
managed processes to stop. Close your normal client separately. Closing the
managed development client also ends that session. Keep the failed session if
something goes wrong; a new launch does not repair its outcome.

Launch again using the same command, join the new address and try the brush after
the HUD becomes live. Do not start the installed runtime template directly or
copy old worlds/tokens into a new session. See
[safe reporting](TROUBLESHOOTING.md#ask-for-help) before sharing evidence.

## Qualify an extracted candidate

Use a fresh extraction and its own `.venv`, native build and remapped mod. Verify
source hashes before and after checks; do not import editable packages from the
original checkout. Record source/archive/mod hashes, platform, setup/build times,
cache use, failures and acquisition/build receipts. A same-host rehearsal with
verified caches is not an independent fresh-download result.

Run the exported `make verify` and execute the
[research quickstart](RESEARCH_QUICKSTART.md) from that extraction. The public
checks explicitly exclude the private Git-governance wrapper and one replay
requiring seven undistributed historical traces. The other runtime tests remain
enabled. `make verify-source` checks the unmodified inventory; after editing
source, a mismatch is expected. Platform qualification for this release means an
independent installation from this guide and a sustained manual session on the
exact candidate; the packaged route's [private test record](COMPANION.md#record-a-private-test-result)
belongs to the later Companion milestone.
