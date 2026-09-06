# Roadmap

The interactive demo exists locally. This repository is the public landing page;
the software and research package are being prepared for release. Milestones
below describe the next public deliverables, not downloads already available.

## 1. Make the local alpha usable

Publish the Fabric mod and its Python neural companion together with matching
versions, checksums, setup instructions and troubleshooting. Explain hardware
requirements, separate data acquisition and which platforms have actually been
tested. Verify installation with someone outside the development environment.

## 2. Make the model approachable

Release a current model card and three headless Python notebooks:

1. **Meet the network:** inspect source IDs, annotations and connections.
2. **Stimulate and observe:** apply a repeatable input and plot modeled activity.
3. **Change the network and compare:** repeat fixed inputs with original weights,
   shuffled weights and no input.

The introductory dataset uses the attributed **14-neuron, 22-edge MaleCNS-derived
fixture**, with synthetic inputs and simulated outputs. It will include the
configuration, seeds, provenance and regeneration instructions. This small
example is separate from the full-network, six-interaction Minecraft demo.

## 3. Strengthen the comparisons

Validate stimulus delivery and actual body application across the live comparison
conditions. Keep failed stimuli and retained responses visible. Extend fixed-input
experiments with multiple shuffle seeds, explicit baselines and sensitivity to
modeling assumptions before making stronger claims about network contributions.

## 4. Build around community feedback

Prioritize reproducible installation reports, clear experiments and documented
extension points. Invite contributions that improve usability, visualization,
control comparisons or understanding of the model. Broader platform and
multiplayer support follow evidence from the local alpha.

The public scope is an inspectable model and interactive demo. Biological
correspondence would require separate evidence beyond successful Minecraft
interactions. Updates will appear in the [project overview](README.md) and future
release notes; no release date is promised here.
