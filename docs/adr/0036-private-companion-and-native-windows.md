# ADR 0036: Packaged local companion and native Windows candidate

Date: 2026-10-24
Status: accepted for private implementation and testing; publication unapproved

## Context

The owner requested a general-player install including Windows, then explicitly
authorized companion/Windows implementation and a private testing candidate.
ADR 0034's external neural service and isolated Fabric server remain suitable;
terminal-driven source setup is insufficient for that audience.

## Decision

Package the pinned CPython runtime, locked production wheels, project source and
matching native kernel as a relocatable per-platform bundle. A double-click
shortcut opens a loopback browser interface implemented with Python's standard
library. It manages setup, native launcher lifecycle, matching client mod copies,
condition requests, cancellation and sanitized local support reports. No hosted
service, frontend runtime dependency or Minecraft account integration is added.

Keep the ordinary Fabric client/account under the player's own launcher. Acquire
Minecraft, Java where needed and model sources locally with declared provenance;
do not redistribute Minecraft, worlds, accounts or unapproved bulk data. Explicit
EULA acceptance belongs to the player. The interface uses a fresh local bearer
token, exact Host/Origin checks, fixed actions and bounded requests.

Use owned Windows Job Objects and native Win32 kernel threads/DLL exports while
retaining POSIX process groups/pthreads. Preserve row arithmetic, bounds checking,
protocol authentication and independent condition identities. Each binary has a
genuine build/source identity. The portable builder uses canonical paths and
platform-safe filesystem operations; its changed implementation receives a new
manifest and coherently regenerated calibration/binding pins.

## Consequences

The new Mac graph arrays reproduce byte-for-byte and actual recalibration is
recorded separately from historical receipts. A new Windows binary can have a
different dynamics identity and uses explicitly disclosed Mac-measured component
scales until independently measured. Neither portability tests nor a successful
bundle build establish Windows gameplay or biological validity.

Freeze committed source, verify exact packages, perform independent review and
prepare Windows/Apple Silicon player tests. Private CI artifacts are testing
material only. Public repositories, releases, invitations and posts remain
unchanged until the owner approves a specific publication candidate.
