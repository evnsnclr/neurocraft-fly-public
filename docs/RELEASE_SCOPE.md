# Release scope: 0.2.0-rc.6 candidate

This describes the 0.2.0-rc.6 source candidate: a Play route built from source on
Apple Silicon macOS as a developer alpha, and the separate research kit. The
packaged Companion and its native Windows port are present in the source, but
their bundles are private testing material outside this release. It does not
announce public software availability or mark qualification complete.
The existing public landing page, demo and project citation remain available.
The source inventory and verification record must identify the exact revision
and bytes tested; earlier packages and videos keep their original identities.

## Play in Minecraft

| Included scope | Contract |
| --- | --- |
| Platform | Apple Silicon macOS, built from source, with installation and automated comparison runs so far qualified only on the development Mac, for the earlier candidate `c1ff8b1`. Linux is experimental. Native Windows is implemented in this tree as private testing material, unqualified, and not a supported platform of this release. Independent installation and sustained manual play remain separate checks. |
| Components | Matching Fabric mod and the local Python neural service, both built from this source tree, plus a fresh showcase world generated at launch. The packaged Companion in this tree is not part of the release; installing the JAR alone is insufficient. |
| Game tuple | Minecraft 1.21.5, Fabric Loader 0.16.10, Fabric API 0.128.2+1.21.5; project mod 0.2.0-rc.6. |
| Player requirements | A matching Fabric client, the player's own Minecraft Java account, a writable setup folder and network access for setup, plus the pinned Python and uv toolchain, Java 21, a supported C compiler, Git and Make to build the kernel and run the service. |
| Model | `full_mcns_interactions_program_v2`, retained MaleCNS v1.0 derivative: 166,700 neurons and 25,582,938 directed edges. Graph, sign, dynamics, encoder, readout and control identities travel with results. |
| Interactions | Player approach, nearby mob, brush, attack attempt, food and light. Each uses a declared source gate, named modeled populations and scripted body programs. Attack is a tactile proxy with no injury or nociception model. |
| Comparisons | Original weights, shuffled weights and no input, with explicit labels and original-state restoration. Weight shuffling permutes counts on existing edges; it does not rewire endpoints. |
| Interface | Compact neural HUD and H-key model/experiment details. Model samples, applied commands, body state and control authority stay distinguishable. |
| Acquired separately | User-owned Minecraft setup, dependency downloads, pinned MaleCNS source tables and locally derived full graph. No hosted LLM or OpenAI API key is needed during play. |

Use the [source installation guide](INSTALL.md) to play. The [Companion guide](COMPANION.md)
describes the private packaged route and is not an entry point of this release. Resource numbers
must be measured on the qualified candidate and hardware; historical graph-build
memory or controller timings are not minimum-system requirements. Normal
account/launcher use remains an owner-operated step; the source candidate does not include
accounts, credentials, Minecraft binaries, existing worlds or bulk source data.

Companion bundles for Windows x64 and Apple Silicon are private testing
candidates outside this release. Linux is experimental; WSL is not independently qualified. Arbitrary worlds, public
multiplayer, a hosted service, automatic launcher-profile changes, online learning
and new species are outside this release.

## Explore the model in Python

| Included scope | Contract |
| --- | --- |
| Entry point | [Research quickstart](RESEARCH_QUICKSTART.md), three runnable notebooks and the [model card](MODEL_CARD.md). |
| Teaching data | One reproducible example derived from the attributed 14-neuron / 22-edge MaleCNS fixture; includes source identities, synthetic inputs and modeled outputs. It is not the full gameplay network or a biological recording. |
| Experiment | Inspect the graph, apply prescribed pulses, and compare original weights, one seeded shuffled-weight control and no input using the Python reference engine. |
| Reproduction | A frozen configuration, explicit seed, schema, hashes and regeneration commands. Report byte-exact and numerical-tolerance verification separately; retain failed executions. |
| Requirements | The project-pinned Python and uv toolchain and optional research group, as described in the quickstart. No Minecraft, Java, native compilation, account or bulk-data acquisition is needed for the tutorial. |
| Platform evidence | Executed on the development Mac; a wider independently tested platform matrix has not been established. |
| Advanced route | Separately acquired full network and declared headless tools; dynamic probes also need the native kernel, while annotation census does not. Running the tutorial does not qualify that route or reproduce full-network gameplay outcomes. |

## Claims, license and readiness

Connectivity and annotations come from the MaleCNS reconstruction. Modeled
signs, dynamics, event encoding, hand-chosen readouts and scripted movement
are assumptions and engineering. Neither candidate establishes recovered
physiology, natural fly behavior, intelligence or consciousness.

The historical trained-readout disadvantage (95.66× raw / 403.70× projected MSE,
measured on a 128-neuron, 887-edge fixture's synthetic imitation task rather than
the full network) and the offline escape findings (DNp01 about 300× wing-pool
response, TTMn 35.1× its near-silent baseline, four neural steps) remain visible in the
[overview](../README.md#read-the-model-honestly) and model card, with their
original measurement scope. They are not benchmarks of the new candidate.

Project-owned software and documentation use MIT. MaleCNS data and the attributed
fixture retain their separate terms and attribution. Minecraft and third-party
components retain their own rights; see [third-party notices](../THIRD_PARTY.md).

Play needs end-to-end installation and experiment checks on its exact package,
independent-user outcomes, sustained measured performance and representative
footage. Explore needs clean regeneration and notebook execution on its exact
package. Both need relevant automated checks, an independent review and a
documented release-or-hold recommendation. Owner authorization is required before
changing the public repository. The research kit may be proposed separately;
that decision cannot imply playable-mod readiness.
