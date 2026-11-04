# Roadmap

The next milestone has two entrances: **Play from source** and **Explore the
model in Python**. The [public project home](https://github.com/evnsnclr/neurocraft-fly-public)
already contains the demo and citation. Candidate `0.2.0-rc.7` is a source
release candidate; the earlier `0.2.0-rc.3` Companion archives remain private
testing material, and no public software release is announced.

## Current scope

Play builds the Fabric mod and the local neural service from source on Apple
Silicon macOS, generates a fresh local showcase world and offers six documented
interactions; it is a developer alpha with installation and automated comparison
runs so far qualified only on the development Mac, for the earlier candidate `c1ff8b1`. The packaged Companion, its
portable runtime/kernel and the native Windows port are implemented in this tree as
private testing material and stay outside this release until independent
installation and sustained normal-client gameplay qualify them. Start with the
[source installation guide](INSTALL.md).

Explore supplies three Python notebooks, a 14-neuron / 22-edge teaching fixture
and a model card. It needs no Minecraft or bulk MaleCNS download. Full gameplay
uses the separately acquired 166,700-neuron network. Start with the
[research quickstart](RESEARCH_QUICKSTART.md); exact scope is in
[RELEASE_SCOPE.md](RELEASE_SCOPE.md).

## Remaining release work

| Area | Evidence needed for the selected candidate |
| --- | --- |
| One candidate | Matching source, mod, runtime pins, model identities, instructions and checksums; required automated checks and independent review. Companion bundles belong to a later milestone. |
| Installation | Independent Apple Silicon setup from the source guide through the normal Minecraft account/client route, without undocumented help. Record downloads, setup time, failures and resource use. Windows qualification belongs to the later Companion milestone. |
| Playability | A 20-minute normal-speed manual session per claimed platform; all six inputs, condition switching, original restoration, full restart, fault recovery and clean exit. Measure actual TPS/FPS, memory and stalls. |
| Comparisons | Verify delivered stimuli and applied responses, disclose condition/seed/reset rules, retain every attempt and restore original. Retained shuffled responses and missed inputs remain visible. |
| Research | Regenerate the fixture and execute all notebooks from the selected export in a clean environment; report exact versus tolerant reproduction separately. |
| Presentation | Representative human-operated footage of that candidate with a readable fly/HUD and labeled automation/operator segments. Existing edited automated video keeps its original scope. |
| Publication | Review the exact public diff, downloads, licenses, current runtime/dependency assessment and support route. Publish only the specifically approved batch. |

A research-only release can precede Play if its own checks pass. It must not imply
that a playable download or independent installation has been qualified. Release
timing follows evidence, with separate release-or-hold recommendations for Play
and Explore.

## Later work

Prioritize reproducible setup failures, crashes, confusing experiments and
accessibility from private tester reports. Batch reviewed fixes into versioned
releases. The packaged Companion and native Windows release, Linux full-network
support, live circuit silencing, stable extension APIs and multiplayer need their
own implementation and qualification.

Multiple shuffle seeds, sensitivity analyses and held-out baselines can support
broader model experiments. New dynamics and mappings retain their own identities
and evidence. A software release establishes no recovered physiology, natural
behavior or consciousness; see the [model card](MODEL_CARD.md).
