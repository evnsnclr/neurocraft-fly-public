# Changelog

## 0.2.0-rc.6 (source frozen 2026-11-03)

- Re-triage all 34 scoped JVM advisory exceptions in
  `docs/security/jvm-advisory-exceptions.json` for this candidate
  (`docs/security/JVM_TRIAGE_2026-11-03.md`): OSV re-queried, the Minecraft
  1.21.5 and Loom 1.10.5 pins and the unbundled mod JAR re-confirmed, and every
  entry replaced by one that records the re-check and expires 2027-01-31. The
  15 entries that were due to expire on 2026-11-24 no longer fail `make jvm-audit`,
  `make verify` and the CI workflow from 2026-11-25. No library version, lockfile
  or game tuple changed.
- List the standalone uv installer, or `pip install uv==0.13.6` inside a virtual
  environment, before a bare pip install in the installation guide and the
  research quickstart; a bare pip install fails on Homebrew's externally managed
  Python.
- State in `SECURITY.md`, the README and the troubleshooting guide that private
  vulnerability reporting is enabled on this repository (verified 2026-11-03).
- Refresh the citation note in `CITATION.cff`, `CITATION.bib` and the README's
  BibTeX: source release candidates are published as GitHub pre-releases.
- Widen the external evidence runner's `java -version` probe bound from 10 to 60
  seconds, with a regression test; a GitHub-hosted runner exceeded the old bound
  in one rc.5 CI run.
- Mark the gate vocabulary in the data provenance and third-party inventory as
  the retired research-gate ladder's (ADR 0019).
- Play remains a developer alpha built from source on Apple Silicon macOS, with
  installation and automated comparison runs so far qualified only on the
  development Mac for the earlier candidate `c1ff8b1`; Explore is the research
  kit. No packaged download, Windows support, playable claim or biological claim.

## 0.2.0-rc.5 (source frozen 2026-11-03)

- Publish the curated source of candidate 0.2.0-rc.5, exported from private
  revision `bee8ca2`: the Fabric 1.21.5 mod, the local Python neural service,
  the research kit (three notebooks, a 14-neuron example dataset and the model
  card), the curated documents and `RELEASE_MANIFEST.json` for inventory
  verification. No packaged download, Minecraft file or bulk connectome data is
  included.
- Replace the landing-page overview with the release README. Play builds from
  source on Apple Silicon macOS as a developer alpha whose installation and
  automated comparison runs are so far qualified only on the development Mac,
  for the earlier candidate `c1ff8b1`; Explore is the research kit. The packaged
  Companion and the native Windows port remain private testing material.
- Record the candidate version and the published MaleCNS article (Berg et al.,
  Cell 2026) in `CITATION.cff`. The landing-page citation and BibTeX are unchanged.
- Keep the demo media and recording notes. The roadmap moves to
  `docs/ROADMAP.md`; a pointer stays at the old path.

## 2026-09-06

- Add a seven-second grooming GIF and the 1080p demo video, with recording notes,
  source attribution and checksums.
- Add a visible project citation, downloadable BibTeX and matching CFF metadata.
- Keep the software release status separate from published demo media.
