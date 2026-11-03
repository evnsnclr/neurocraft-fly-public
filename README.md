# NeuroCraft Fly

**An interactive fruit-fly connectome model in Minecraft.**

Walk toward the fly, brush it, offer sugar, change the light, or bring a cow close.
Minecraft events stimulate named model populations; hand-chosen neural readouts
select and modulate scripted body programs. The retained MaleCNS network has
**166,700 neurons and 25,582,938 directed edges**. A compact HUD shows the active
condition and modeled activity; **H** opens model and experiment details.

```text
Minecraft inputs → modeled neural activity → labeled readouts
                 → scripted body programs → Minecraft movement
```

## Demo

[![Automated brush input followed by grooming, with original weights and the compact neural HUD](https://raw.githubusercontent.com/evnsnclr/neurocraft-fly-public/main/media/brush-demo.gif)](https://github.com/evnsnclr/neurocraft-fly-public/blob/main/media/neurocraft-demo.mp4)

**[Watch or download the full demo · 2:27 · 1080p / 30 FPS](https://raw.githubusercontent.com/evnsnclr/neurocraft-fly-public/main/media/neurocraft-demo.mp4)**

Real Minecraft footage, edited from selected automated takes. The full video
includes six inputs, comparison conditions, restored-original brushing and a
separately labeled operator tour. It is not an uninterrupted human-played session.
[Recording notes and exclusions](https://github.com/evnsnclr/neurocraft-fly-public/blob/main/media/README.md).
These existing public media describe recorded source `87ebb97`, not qualification
of the later software candidate.

## Choose your starting point

**Candidate `0.2.0-rc.5` is a source release candidate.** This repository holds the Fabric
mod and local neural service sources, the research kit and the curated documents;
it ships no packaged downloads. The Play route builds from source on Apple Silicon
macOS as a developer alpha, with installation and automated comparison runs so far
qualified only on the development Mac, for the earlier candidate `c1ff8b1`. The packaged Companion and its native
Windows port are implemented in this tree as private testing material: their bundles
are unqualified and unpublished. Read the
[release scope](docs/RELEASE_SCOPE.md) before setup.

| | Play in Minecraft (developer alpha) | Explore the model in Python |
| --- | --- | --- |
| Start | [Build and run from source](docs/INSTALL.md) | [Research quickstart](docs/RESEARCH_QUICKSTART.md) |
| What you use | The Fabric mod, the local Python neural service and a fresh showcase world, built and launched from this source tree | Three notebooks and a reproducible example dataset |
| Network | Full retained MaleCNS: 166,700 neurons / 25,582,938 directed edges, derived locally from the separately acquired source tables | Attributed teaching fixture: 14 neurons / 22 directed edges |
| Platform scope | Apple Silicon macOS, with same-host installation and automated runs so far qualified only on the development Mac for the earlier candidate `c1ff8b1`; Linux experimental; Windows unqualified | Tested on the development Mac; broader qualification remains open |
| You need | Your own Minecraft Java account and matching Fabric client, plus the pinned Python and uv toolchain, Java 21, a C compiler, Git and Make | Pinned Python dependencies; no Minecraft, Java, native build or bulk download |

The [Companion guide](docs/COMPANION.md) describes the private packaged route for
completeness; its bundles are not part of this release.

After joining the playground, try the [first brush interaction](docs/EXPERIMENTS.md#your-first-interaction),
then [compare conditions](docs/EXPERIMENTS.md#compare-the-conditions) and restore
original weights. Use [troubleshooting](docs/TROUBLESHOOTING.md) if setup fails.
Gameplay runs locally without a hosted LLM or OpenAI API key. Initial dependency
and data acquisition use the network. This alpha targets the documented local
showcase; arbitrary worlds and public multiplayer are outside its scope.

For Explore, inspect source annotations, stimulate the reference engine, then
compare original weights, one seeded weight shuffle and no input. The
[model card](docs/MODEL_CARD.md) separates the teaching example from the full
model and its engineered Minecraft interfaces. The notebook outputs are model
experiments, not biological recordings or a reproduction of the full demo.

## What has been observed

Full-network gameplay has been recorded on the development Mac. A production-JAR
connection/HUD smoke also ran on that host. Independent first-user installation,
normal account/launcher flow and sustained manual play on the final candidate
remain release checks, not established outcomes.

The compact-HUD showcase has selected automated takes across original,
shuffled-weight and no-input conditions. In those takes, seed-7 shuffling retained
five of six target responses; none of the six corresponding target responses
appeared with sensory input disabled. Baseline cruise can still move the body.
Histories differ between takes; failed preparations and stalled capture sections
were retained privately and disclosed in the recording notes. This is an
illustration, not a repeated matched-input benchmark or biological validation.

Earlier source `9f7b13e` completed 18 qualified original trials and a
720.050-second walkthrough. Six native-input control trials were excluded;
shuffled light retained its response among qualified controls. Ten late
retirements and thirteen sampled readout stalls occurred, without a recorded
terminal bridge fault. A separate fixed-60 schedule missed every forward sensor
ray and establishes no suppression. Those results retain their original build
and scope; the [model card](docs/MODEL_CARD.md#evidence-and-limits) summarizes them.

## Read the model honestly

The source reconstruction supplies connectivity and neuron annotations.
Sensors, dynamics, modeled signs, readout mappings and Minecraft movement are
declared modeling and engineering choices.

- A historical readout on a **128-neuron, 887-edge fixture** had **95.66× raw /
  403.70× bounded MSE** relative to a matched direct controller on synthetic
  open-loop imitation. This did not test full-network or closed-loop navigation
  ([ADR 0038](docs/adr/0038-trained-readout-result-scope.md)).
- The **offline** escape diagnostic measured DNp01 about **300×** the wing-pool
  response and TTMn **35.1×** its near-silent baseline in **four neural steps**.
  This is not measured live reaction timing.
- **Hand-chosen readouts select and modulate scripted body programs.** The HUD
  separates applied samples from model samples and labels the active condition
  and control authority. Missing matching evidence stays unavailable.

See H details, [ADR 0019](docs/adr/0019-pivot-to-interactive-connectome-demo.md)
and the [offline escape receipt](docs/receipts/ESCAPE-ARC-01.md). The
[evidence guide](docs/EVIDENCE.md) explains retained history and omitted private
support material. No recovered physiology, natural behavior, intelligence or
consciousness is established.

## Cite this project

Smith, E. (2026). *NeuroCraft Fly*. [Project landing page](https://github.com/evnsnclr/neurocraft-fly-public).

```bibtex
@misc{smith2026neurocraftfly,
  author = {Smith, Evan},
  title = {{NeuroCraft Fly}},
  year = {2026},
  url = {https://github.com/evnsnclr/neurocraft-fly-public},
  note = {Project landing page; software release is in preparation}
}
```

[Download the project BibTeX](https://github.com/evnsnclr/neurocraft-fly-public/blob/main/CITATION.bib)
or use [CITATION.cff](CITATION.cff). This landing-page citation does not identify
a published software release. When using candidate code, record its version and
the exact source revision too. Cite the underlying MaleCNS reconstruction
separately as described in [credits](docs/CREDITS.md).

## Participate

Created by [Evan Smith](https://github.com/evnsnclr). Start with the
[roadmap](docs/ROADMAP.md), [contributing guide](CONTRIBUTING.md) and
[community code](CODE_OF_CONDUCT.md). Questions and reproducible bug reports
belong in repository issues when enabled. For sensitive findings, follow
[SECURITY.md](SECURITY.md); the private reporting route must be verified before
submitting sensitive material.

Project software and associated documentation use the [MIT License](LICENSE).
Third-party software, data and assets retain their separate terms; see
[credits](docs/CREDITS.md) and [third-party notices](THIRD_PARTY.md). Minecraft
binaries, account material and bulk connectome data are not included.

NeuroCraft Fly is not an official Minecraft product and is not approved by or
associated with Mojang or Microsoft.
