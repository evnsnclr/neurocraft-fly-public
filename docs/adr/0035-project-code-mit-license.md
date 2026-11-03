# ADR 0035: MIT license for project software

Date: 2026-09-06
Status: accepted by the owner

## Context

The public-alpha implementation and source/mod review package were prepared
under the existing all-rights-reserved notice. D-021 deferred any permissive
license to an explicit owner decision. On 2026-09-06 Evan Smith approved MIT
as the public code license.

## Decision

Apply the standard MIT License to project-owned software and its associated
documentation, with copyright 2026 Evan Smith. Keep the license text unmodified;
NOTICE and THIRD_PARTY.md explain the separate rights for dependencies, data
and Minecraft. Fabric and Python metadata identify the project software as MIT.

Rebuild the source export and both mod JARs so their license text, metadata and
checksums agree. The exporter may identify a complete standard MIT grant but
must not infer publication approval or publisher authenticity from that grant.
Unknown or incomplete license text remains unresolved by the exporter.

This supersedes D-021 for project software and resolves the code-license decision
left open by ADR 0034. Earlier ADRs, source packages, receipts and their hashes
remain unchanged historical artifacts. The current package receives new hashes.

## Consequences

The software may be used and redistributed under MIT's conditions. This does
not relicense third-party dependencies, MaleCNS source/derived data, Minecraft,
or independently licensed material. Existing attribution and source-provenance
requirements remain in force; bulk data and Minecraft are not bundled.

Repository visibility, publishing downloads, external invitations and outreach
remain separately authorized actions. Independent first-user setup and broader
platform qualification remain unobserved and explicitly described as such.

The standard license text is published by the
[Open Source Initiative](https://opensource.org/license/mit).
