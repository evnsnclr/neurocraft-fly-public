# Contributing to NeuroCraft Fly

Useful contributions make the local demo easier to run, understand or reproduce.
Start with the [public-alpha overview](README.md) and
[roadmap](docs/ROADMAP.md).

Project software and associated documentation use the [MIT License](LICENSE).
Submit only work you have the right to contribute under those terms. Third-party
software, data and assets retain their own licenses and attribution requirements.
Setup reports, documentation questions and proposals can be discussed through
this repository's **Issues → New issue** without sharing restricted material.

## Report or propose a change

Choose **Setup help or bug**, **Feature or documentation proposal**, or
**Private contact request**. Describe a concrete trigger and observed result.
Logs are optional; redact them as described in
[troubleshooting](docs/TROUBLESHOOTING.md#ask-for-help). Vulnerabilities
belong in the private route described in [SECURITY.md](SECURITY.md).

For substantial changes, agree on scope in an issue before implementing. Good
first targets include clearer setup errors, reproducible platform reports,
camera/HUD accessibility and smaller verified examples. Do not promise that
Linux/Windows or a fresh-machine install works without a corresponding run.

## Prepare a reviewable patch

1. Work on a focused branch and avoid unrelated cleanup.
2. Read [architecture](docs/ARCHITECTURE.md), [experiments](docs/EXPERIMENTS.md),
   and [ADR 0019](docs/adr/0019-pivot-to-interactive-connectome-demo.md), plus
   decisions relevant to your change. A changed claim needs evidence.
3. Run the narrow checks relevant to the change. Record exact commands and actual
   outcomes, including tests you could not run. `make verify` is required before
   merge; it includes graphical client checks and is not a lightweight live-session
   command. Do not run builds/tests during a measured Minecraft take.
4. Update the relevant public documentation for a feature. Add an ADR only for a durable
   architecture decision. Preserve existing receipts; supersede rather than edit them.
5. Open a pull request using the template. Name the behavior change, validation
   and remaining limits. A draft PR is not a merged or released result.

Keep the controller and presentation responsibilities clear. HUD, animation and
camera changes must not silently select body programs. Original, shuffled,
no-input and operator-selected conditions must remain explicit. Never relabel
a fixture, command request, selected state or screenshot as an applied neural
response without the matching evidence.

Do not add bulk data, generated worlds/caches, account material or unreviewed
third-party assets. New dependencies need an exact pin, necessity review,
license/notice inventory and security assessment. Changing graph-loading or
model code can rotate downstream identities; include the required regenerated
bindings while preserving historical artifacts.
For a change affecting Faithful Organism Mode, state whether it alters topology,
dynamics, sensory mapping, motor mapping, calibration, or presentation.

Maintainers review and merge according to available capacity; no response-time
or platform-support SLA is promised. Publishing official project releases or
changing the project's repository visibility requires explicit owner approval.
Follow the
[Code of Conduct](CODE_OF_CONDUCT.md) throughout the discussion.
