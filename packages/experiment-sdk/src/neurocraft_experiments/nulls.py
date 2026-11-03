"""Sign-space null generators for Trained-Readout Mode experiments (ADR 0008).

This module lives in its own file on purpose: fixture-physics pilot v3 pins the
exact bytes of eleven implementation files (including `neural_runtime.model` and
`neurocraft_experiments.pilot`), so new controls must arrive as new modules that
leave every pinned file byte-identical. Sign permutation needs no pinned-file
change because :meth:`SignOverlay.explicit` already accepts arbitrary values
with named provenance.

The permutation preserves the base overlay's exact sign multiset and keeps each
value/provenance pair together, so the control isolates sign *placement* from
sign *statistics*. Determinism uses the same seeded-SHA-256 key sort as the
pinned ``WeightOverlay.shuffled`` rather than ``random``, for the same reason:
one integer seed, one permutation, on every platform.
"""

from __future__ import annotations

import struct
from hashlib import sha256

from neurocraft_connectome import RuntimeGraph
from neurocraft_neural.model import SignOverlay

PERMUTED_SIGN_PROVENANCE_PREFIX = "sign_permutation_control"

_SEED_MAX = 2**64 - 1


def _validated_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= _SEED_MAX:
        raise ValueError("sign permutation seed must be an integer in [0, 2**64)")
    return seed


def permuted_sign_pairs(
    base: SignOverlay, *, seed: int
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[int, ...]]:
    """Deterministically permute the base overlay's (value, provenance) pairs.

    Returns ``(values, provenance, permutation)`` where ``permutation[k]`` is the
    base edge slot whose pair now sits at slot ``k``. The pair travels together so
    every permuted sign still names the transmitter inference it came from.
    """

    resolved_seed = _validated_seed(seed)
    seed_bytes = struct.pack(">Q", resolved_seed)
    permutation = tuple(
        sorted(
            range(len(base.values)),
            key=lambda index: (sha256(seed_bytes + struct.pack(">Q", index)).digest(), index),
        )
    )
    values = tuple(base.values[source] for source in permutation)
    provenance = tuple(base.provenance[source] for source in permutation)
    return values, provenance, permutation


def permuted_sign_overlay(
    graph: RuntimeGraph,
    base: SignOverlay,
    *,
    seed: int,
    protocol_id: str,
) -> SignOverlay:
    """Build the explicit permuted overlay, bound to ``graph`` and labelled.

    Every provenance entry records the protocol, seed, and the base slot the
    sign came from, so a trace carrying this overlay can never be mistaken for
    the faithful consensus assignment.
    """

    if not isinstance(protocol_id, str) or not protocol_id:
        raise ValueError("sign permutation requires a non-empty protocol_id")
    base.assert_compatible(graph)
    values, provenance, permutation = permuted_sign_pairs(base, seed=seed)
    labelled = tuple(
        f"{PERMUTED_SIGN_PROVENANCE_PREFIX}:{protocol_id}:seed={seed}:from_slot={source}:{entry}"
        for source, entry in zip(permutation, provenance, strict=True)
    )
    return SignOverlay.explicit(graph, values, labelled)
