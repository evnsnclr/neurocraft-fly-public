# ADR 0007: Prove token possession mutually; never transmit the token

_Control-channel handshake revision, accepted 2026-08-18_

---

| Field | Value |
|---|---|
| **Status** | Accepted and implemented; unit- and integration-tested on both endpoints |
| **Date** | 2026-08-18 |
| **Authority** | `neurocraft.protocol.v2` handshake frames on both in-repo endpoints |
| **Finding lineage** | `SECURITY-SCAN-01` finding 1 (medium): the client disclosed the bearer token before any server proof |
| **Evidence** | Regenerated `packages/protocol/golden/hello.json` / `hello_ack.json` and `schema/protocol-v2.schema.json`; cross-language pinned transcript vector asserted byte-identically in Python and Java; independent security review of the change found no critical/high issue |

## Context

The previous handshake sent the raw run token inside the `hello` frame and
accepted a `hello_ack` on public identity fields alone. A process that could
bind the loopback port before the service could therefore receive the bearer
token without ever proving it held anything, and the bridge would activate
against an unproven peer.

## Decision

Both endpoints now prove possession of the run token with HMAC-SHA256 over a
canonical handshake transcript; the token itself never crosses the wire.

- `hello` carries a fresh 32-byte `client_nonce` (base64url, 43 characters)
  plus `auth = HMAC-SHA256(token, transcript("hello"))`.
- `hello_ack` carries `server_auth = HMAC-SHA256(token, transcript("hello_ack"))`,
  bound to the same client nonce; the bridge verifies it before entering
  `ACTIVE` and latches `IDENTITY_MISMATCH` otherwise. No pre-verification state
  transition, queue admission, or session-epoch claim occurs on either side.
- The transcript is an unambiguous length-prefixed concatenation of exactly
  twelve UTF-8 elements in a frozen order — protocol version, message label,
  run id, session epoch, entity id, the five identity hashes, peer role, and
  client nonce — each preceded by a 4-byte big-endian length. The label and
  role fields domain-separate the two directions, so neither proof can be
  reflected as the other; the session epoch inside the transcript composes
  with the existing one-shot session-epoch claim to bound replay.
- Comparisons are constant-time on both sides (`hmac.compare_digest`;
  `MessageDigest.isEqual` over fixed 64-byte operands). Java uses only JDK
  `javax.crypto.Mac`; no dependency changed.
- The wire subprotocol remains `neurocraft.v2`: both endpoints live in this
  repository, no external consumer exists, and the schema, goldens, and both
  codecs moved in the same change (D-028). Protocol-v1 remains historical.

## Consequences

- The bearer token is no longer disclosed to an unauthenticated listener, and
  each endpoint activates only against a peer that proved possession.
- The handshake frames are a breaking change to `neurocraft.protocol.v2`'s
  hello/hello_ack shapes; every enumerated evidence bundle that binds protocol
  or bridge sources is superseded for current-tree candidacy and is
  regenerated as `EXTERNAL-BRIDGE-04`. Earlier receipts remain historical
  evidence bound to their own commits.
- A pinned 469-byte cross-language transcript vector with both digests is
  asserted on both endpoints; any divergence in canonicalization fails tests
  rather than production handshakes.
