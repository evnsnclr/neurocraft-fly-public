# NeuroCraft Fly

**An interactive fruit-fly connectome model in Minecraft.**

NeuroCraft Fly lets people interact with a simulated neural network, inspect its
activity and explore how changing the network changes its responses. The local
demo runs a retained MaleCNS graph with **166,700 neurons and 25,582,938 directed
edges**, connected to a custom fly and a live activity display.

```text
Minecraft inputs → modeled neural activity → labeled readouts
                 → scripted body programs → Minecraft movement
```

The source reconstruction supplies connectivity and neuron annotations. Sensory
mappings and dynamics are modeled, and hand-chosen readouts select and modulate
scripted body programs. This makes the connection visible and testable; it does
not establish recovered fly physiology or natural behavior.

## Release status

This repository is the project's public landing page. The interactive demo has
been implemented and recorded locally. The mod, companion source and research
starter kit are being prepared for distribution; **downloads and runnable assets
are not available here yet**.

The first release is being prepared around two entry points:

- **Play:** a Fabric mod, local neural companion and installation guide.
- **Explore:** three Python notebooks, a small reproducible example dataset and
  a model card explaining the implementation and its assumptions.

See the [roadmap](ROADMAP.md) for the release sequence. Watch this repository for
release updates. Questions and suggestions about experiments, installation and
research documentation are welcome through repository issues when enabled.

## What the experiments mean

The recorded demo exercises six declared interactions: player approach, nearby
mobs, brushing, attack attempts, food and light. Network comparisons are still
being qualified; a response surviving shuffled weights is a result to report.

Two historical findings help explain the design. A trained readout had 95.66×
higher raw error and 403.70× higher projected error than a matched direct
controller on its navigation task. A separate offline escape-path probe found
DNp01 changes about 300× the wing-pool changes, TTMn activity 35.1× its near-silent
baseline, and propagation over four neural steps. These concern their original
configurations, not biological validation or a general control advantage. The
release documentation will include their provenance and limits.

## Credits and license

Created by [Evan Smith](https://github.com/evnsnclr).

The project builds on the [MaleCNS reconstruction](https://male-cns.janelia.org/)
by FlyEM at HHMI Janelia, the University of Cambridge Department of Zoology, the
MRC Laboratory of Molecular Biology and Google Research. Please credit the
underlying scientific work: [Berg et al., *Sexual dimorphism in the complete
connectome of the Drosophila male central nervous
system*](https://doi.org/10.1101/2025.10.09.680999).

Project-owned documentation in this repository uses the [MIT License](LICENSE).
MaleCNS data retains its [upstream terms and attribution](https://male-cns.janelia.org/download/).
Minecraft and third-party materials retain their own rights. No game files or
bulk connectome data are distributed here.

NeuroCraft Fly is not an official Minecraft product and is not approved by or
associated with Mojang or Microsoft. To reference this project, use the metadata
in [CITATION.cff](CITATION.cff), and cite MaleCNS separately when using its data.
