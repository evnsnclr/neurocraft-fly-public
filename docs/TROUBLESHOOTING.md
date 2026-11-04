# Troubleshooting

Players of the source release: start with the [installation guide](INSTALL.md),
run `python3 -I scripts/public_alpha.py doctor` from the source root and follow
that revision's printed setup plan; keep the first error and the failed attempt
when asking for help. Fix the reported prerequisite; do not weaken identity or
checksum checks. Private Companion testers start with its progress panel's first
error and **Save sanitized support report** instead.

## Source setup

| Symptom | Next step |
| --- | --- |
| Missing tool or unsupported build | Check the project-pinned toolchain and platform scope. A Windows port or Linux fixture test does not establish full-network playability. |
| Missing graph/annotations/transmitters | Follow the pinned acquisition and derivation plan. The tutorial fixture cannot replace the full interaction graph. |
| Gradle verification fails | Keep the coordinate and checksum error. Do not switch to lenient verification or update a lock just to pass. |
| Installed runtime differs from source | Build the intended production JAR, stage a new runtime and install its matching client files and generated launcher version. Do not rewrite an old inventory to accept different bytes. |

## Gameplay (source and Companion routes)

Rows that name Companion buttons, shortcuts or its progress panel apply only to
private Companion testers; the source route uses the launch command and the
in-game commands from the guides above.

| Symptom | Next step |
| --- | --- |
| Java library check failed / needs patched launcher version | Install the generated client version (from source: the `scripts/java_runtime.py` install step in the [installation guide](INSTALL.md); private Companion testers: **Install client files**), restart the launcher, and select the generated `neurocraft-fly-1.21.5-…` version for the isolated game folder. A plain Fabric version does not supply the patched libraries. |
| Start shortcut fails | Extract the entire matching platform archive into a short, writable local path. Keep `app` and `runtime` beside the shortcut. Do not run inside the ZIP. Keep the OS error; do not disable system security globally. |
| Java is missing | Leave Java home blank for discovery/private download, or enter a working Java 21 home. Allow the displayed downloads if you agree. A launcher's private Java may not be on PATH. |
| Setup refuses the EULA | Read it and check acceptance yourself if you agree. Someone else's accepted state is not transferable. |
| Download, data build or setup fails | Keep the first error and retry outcome. Check available disk space and the selected writable folder. Stop before retrying; do not delete a verified graph or change source hashes. |
| Kernel or calibration mismatch | Keep the candidate's library and generated sidecar together. Use one complete matching candidate; do not borrow another build's sidecar or edit identities. |
| Client mod mismatch or extra mods | Use an isolated matching Fabric profile and **Install matching mod files** with its empty `mods` folder. Keep other mod collections in their own profiles. |
| Second launch will not connect | Stop the old session, start again and join the newly displayed address. A previous session's port/identity is not reused. |
| HUD stays Starting or reports a terminal fault | Keep the first progress-panel error and failed session. Stop and start a fresh session; reopening H or R cannot repair a terminal fault. |
| Closing the browser did not stop Companion | Reopen its local page and use **Stop and close companion**. Closing only the browser tab leaves it running. Your own Minecraft client closes separately. |
| Brush/attack does nothing | Close menus, aim within reach with a clear line of sight, and use the correct item/button. Stay a living non-spectator player and wait through hold/cooldown. |
| Walk/cow does not trigger | The same front/side ray needs decreasing actor distance on consecutive observations. Camera visibility alone is insufficient; fly movement also matters. |
| Food never reaches Feed | Drop it forward and near touchdown; feeding needs support and food within 0.75 blocks. A nearer off-cone item can mask an offering. There is no automatic food seeking. |
| Torch does not cause Hover | Measured block/sky light may be unchanged in daylight. Use the [controlled baseline](EXPERIMENTS.md#controlled-light-preparation) and inspect light in H details. |
| Applied sample is Awaiting match, Stale or Unavailable | Matching evidence is unavailable, not zero. Check **Body now** separately; its sequence can differ from the model/applied samples. |
| H or R does not open | Check Controls → Key Binds → NeuroCraft Fly and close other screens. F1 hides the HUD. Details and replay leave the world running. |
| Small window or oversized HUD | Use GUI scale 2, enlarge the window and keep the fly away from the left card. Remaining samples are accessible in H details. |
| Low FPS, late commands or stalls | Stop competing builds/exports and rehearse without recording first. Keep normal world speed for normal-speed acceptance; record actual performance rather than requested FPS. |

## Ask for help

After a public release, use the repository's setup/bug issue template when
available. Private Companion testers share the sanitized support report and a
short description through the agreed private test channel; no invitation or
external sharing is authorized by these instructions. Include:

- candidate/archive or source revision, OS, CPU architecture and RAM;
- condition, expected result, actual result and first relevant error;
- minimal reproduction steps, including open menus, resets and retry outcomes.

Inspect any extra excerpt before sharing. Remove personal paths, player/account
identifiers, tokens, cookies and private world/network data. Never upload entire
session/setup folders, launcher profiles, environment dumps, caches or source data.
Keep the unchanged original privately; a sanitized excerpt has different bytes.
For suspected vulnerabilities, follow [SECURITY.md](../SECURITY.md), use its
private reporting route and keep exploit details out of public issues.
