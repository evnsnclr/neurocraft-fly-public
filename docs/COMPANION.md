# Companion (private testing route)

**Private testing route, not part of the 0.2.0-rc.7 source release.** The last
Companion bundles were the unsigned `0.2.0-rc.3` private testing candidates for
Windows x64 and Apple Silicon macOS; no bundle has been built from this candidate
and none is published. Testers use the exact platform archive and checksum
supplied for their test. Independent installation, Windows full-network
performance and sustained manual play remain unqualified. Players of the source
release use the [source installation guide](INSTALL.md).

## Before you start

You need your own **Minecraft Java Edition account**, an isolated game folder
for the supplied **Minecraft 1.21.5 / Fabric Loader 0.16.10** version, an internet connection
for setup, and a writable local folder. Companion includes Python and the neural
kernel; you do not need to install Python, a compiler, Git or Make. Java 21 can
be selected locally or downloaded privately during setup.

Model source downloads total about 1.1 GB plus metadata. Setup derives the graph
locally and obtains the Fabric server. Leave additional disk space and memory
for those files and the game; a measured total-memory requirement is not yet
established. Minecraft, accounts, worlds and bulk connectome data are not bundled.

## Install and start

1. Extract the **entire archive** into a short, writable local path, such as
   `C:\NeuroCraftTest` on Windows. Keep `app` and `runtime` beside the start
   shortcut. Avoid Program Files, cloud-synced folders and Minecraft profiles.
2. Double-click **Start NeuroCraft Fly.cmd** on Windows or **Start NeuroCraft
   Fly.command** on macOS. Companion opens in your browser. The unsigned Mac
   launcher may need an explicit Open action; do not disable system security.
3. Read the download information and enable its download checkbox if you agree. Enter a Java 21 home or leave it blank for discovery/private download.
   Read the Minecraft EULA and check acceptance yourself if you agree.
   Select **Prepare companion** and wait for setup to complete. Keep any failed
   attempt for diagnosis.
4. Enter the isolated game folder's empty `mods` directory and your **Minecraft
   launcher folder** (the folder containing `versions` and `libraries`). On Windows
   this is usually `%APPDATA%\.minecraft`; on macOS it is usually
   `~/Library/Application Support/minecraft`. Click **Install client files**.
   This copies the project mod and **Fabric API 0.128.2+1.21.5**, and creates a
   new version named `neurocraft-fly-1.21.5-…` with patched Java libraries.
5. Restart your normal launcher if needed. In its installation settings, select
   the generated NeuroCraft version shown in Companion's progress panel and set
   its game directory to the isolated folder containing `mods`. Existing launcher
   profiles and account files are not edited by Companion. Enter your Java player
   name in Companion and launch that installation. Select **Start
   playground** in Companion. Wait for readiness, then use Minecraft's
   **Multiplayer → Direct Connection** to join the displayed local address.
   Keep Companion running while playing; each session has a new join address.

The mod checks the actual loaded library bytes before gameplay and refuses an
older or modified library selection. A plain Fabric profile with only the two
mod JARs is insufficient for this candidate.

Start with a 1280×720 or larger window and GUI scale 2. These are viewing settings,
not a performance guarantee. Wait for **Live · Original weights · Readout control**
on the HUD, then try the [first brush interaction](EXPERIMENTS.md#your-first-interaction).
Gameplay needs no hosted AI account or API key. No automatic usage reports are sent.

## Interact and compare

Follow the [six interaction recipes](EXPERIMENTS.md#six-sources-six-declared-responses).
**H** opens model details; **R** opens a read-only history of collected HUD samples.
Both leave the world running, so close them before interacting again.

In Companion, request **Original weights**, **Shuffled weights** or **No input**.
Wait for the in-game HUD to acknowledge each condition before repeating an input.
Finish by restoring **Original weights**. A shuffle may retain responses; no-input
can still show fallback cruising. See [comparison instructions](EXPERIMENTS.md#compare-the-conditions).

## Stop and restart

Use **Stop session** to end a game or cancel setup. A stalled download can take
until its read timeout to stop. Your ordinary Minecraft client remains yours to
close separately. To exit Companion, select **Stop and close companion** before
closing its browser tab; closing only the tab leaves Companion running.

Reopen the start shortcut for a new launch. Join the **newly displayed** address
and wait for Live / Original weights / Readout control again. A terminal session
fault needs a new session; it is not repaired by reopening a menu.

## Record a private test result

Keep the archive hash, OS, CPU, RAM, graphics hardware, cache use and setup/retry
outcomes. Test all six interactions and comparisons, restore original weights,
and complete a full client/Companion restart with clean exit. Record a 20-minute
normal-speed session, achieved TPS/FPS, total memory and any late/stalled commands.
Retain missed inputs and failures. A build self-test is separate from this result.

**Save sanitized support report** exports platform/runtime/status categories
without player names, paths, tokens, raw logs or account files. Share that report
and your observations privately during testing. Inspect extra logs before sharing;
never upload entire setup/session folders. See [troubleshooting](TROUBLESHOOTING.md),
[third-party notices](../THIRD_PARTY.md) and the [model card](MODEL_CARD.md).
