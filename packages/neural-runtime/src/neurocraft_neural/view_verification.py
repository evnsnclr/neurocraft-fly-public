"""Verification policy for the array-native runtime view.

Why this module exists
----------------------
The reference :class:`~neurocraft_neural.engine.NeuralEngine` re-derives the
graph and all four overlays from first principles on every step. Measured at
fixture scale that is 213 us of a 258 us step; extrapolated to the full
25,582,938-edge graph it is roughly 149 s of a 152 s step, i.e. about 98% of
the cost. It cannot be in the hot loop of a real-time backend.

What the per-step rebuild actually defends against is post-construction
mutation of in-process Python objects. This module replaces it with a policy
that defends against mutation of *the bytes the kernel dereferences*, which is
a strictly larger surface, at O(1) per step:

1. **Load-time, bytes bound to the mapping.** Already implemented upstream:
   ``MappedUnsignedArray`` opens ``O_NOFOLLOW``, ``fstat``-gates regular-file,
   ``st_nlink == 1`` and the exact declared byte length, maps *that
   descriptor*, and hashes the mapping itself. There is no verify-then-reopen
   window. The derived view adds ``view_sha256`` over
   ``indptr || indices || weights`` in that declared order.
2. **Freeze by construction.** All three buffers get
   ``flags.writeable = False`` and the C kernel takes ``const`` pointers.
3. **Per-step, O(1).** :func:`assert_buffers_pinned` re-checks each buffer's base
   address, byte length, dtype, shape and writeable flag against the token
   captured at verification time. This catches rebinding, resizing, dtype
   swaps, re-enabled writeability and whole-view substitution.
4. **Checkpointed re-verification.** :class:`IncrementalViewDigest` re-hashes
   the view in bounded per-step slices. Each slice is compared against the
   digest that slice had at verification time *as it is hashed*, not only when
   a whole pass finishes. This is what catches in-place mutation through a
   retained alias or a same-inode writer on disk -- the one residual the
   upstream mmap docstring already discloses.

Detection semantics, stated exactly
-----------------------------------
The byte stream is cut into ``ceil(total_bytes / bytes_per_step)`` fixed
slices, and slice ``i`` is re-hashed on every step congruent to ``i`` modulo
the slice count. Therefore:

* **Persistent mutation** -- bytes that are changed and left changed -- is
  detected within ``full_coverage_period_steps`` steps of the change, because
  the slice containing them is re-hashed at least once in any window of that
  length. This is a real bound, not an average.
* **Transient mutation** -- bytes changed and restored before the slice
  containing them is next re-hashed -- is **not detected at all**. Nothing in
  an amortised scheme can detect it: no byte is read on the steps between its
  own re-hashes. A run that wants zero-latency coverage must set
  ``mode="load_only"`` and rebuild the view per step, or accept this bound.
  The steps affected by such a mutation are indistinguishable from clean steps
  in the trace, so the amortised policy must never be described as covering
  them.
* An earlier revision compared only the completed-pass digest. That reported a
  persistent mutation no earlier than the *end of the next pass*, i.e. up to
  ``2 * full_coverage_period_steps - 1`` steps, which is not what the policy
  declared. Per-slice comparison is what makes the declared bound true.

Why AB-05 and AB-11 survive
---------------------------
*AB-05 (fail closed)* is about commands: invalid, late, missing, duplicate,
future, wrong-session or wrong-hash commands must produce neutral/pause. That
path lives entirely in the protocol and controller layers and is untouched
here. What changes for graph-array tampering is the *detection latency* for a
persistent change, from 0 steps to at most ``full_coverage_period_steps``, and
the loss of coverage for a transient change entirely. The outcome for what is
detected is unchanged
because detection retroactively invalidates every record back to
``verified_through_step``: a tampered run can never be presented as valid
evidence. The cadence itself is hashed into
``verification_policy_sha256`` and carried in the trace, so a weaker cadence
cannot be substituted silently.

*AB-06 (executed graph identity)* is strengthened, not weakened: the reference
verifies a Python object graph, while this verifies the exact bytes the kernel
reads, and records that digest on every step.

*AB-11 (test power)* keeps each committed mutant's named detector, and the
bound kernel object is pinned by object identity so the runtime-substitution
idiom (rebinding an engine's kernel mid-run while the published identity stays
put) faults on the next step. Byte
substitution is still caught by ``load_malecns_fixture`` /
``load_full_runtime_artifact``; runtime-object substitution is still caught by
the O(1) bound-identity comparison in the controller, now joined by
:func:`assert_buffers_pinned`; trace-provenance substitution is still caught by
the replayer.

Honest boundary
---------------
An adversary who can already execute arbitrary Python in this process can flip
``writeable`` back, or monkeypatch ``assert_integrity`` in the reference engine
just as easily. Freezing is not a weaker guarantee than the per-step rebuild;
it is the same guarantee at a different cost. What is genuinely traded is
detection latency for on-disk or aliased in-place mutation, and that trade is
declared, hashed and bounded rather than assumed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from neurocraft_connectome.model import sha256_json

DEFAULT_FULL_COVERAGE_PERIOD_STEPS: Final = 200
#: Every trigger listed here is invoked by :class:`.ArrayNativeEngine`; the
#: list is hashed into ``verification_policy_sha256`` and must therefore
#: describe behaviour that exists. ``ArrayNativeEngine`` has a named call site
#: for each entry, and ``test_n1_every_declared_trigger_has_a_call_site``
#: asserts that.
UNCONDITIONAL_VERIFICATION_TRIGGERS: Final[tuple[str, ...]] = (
    "run_start",
    "run_end",
    "snapshot_restore",
    "lesion_change",
    "first_fault",
)

#: Triggers that a previous revision declared and hashed into the policy
#: identity but never invoked. They are recorded here rather than silently
#: dropped, with the reason each was withdrawn.
WITHDRAWN_VERIFICATION_TRIGGERS: Final[dict[str, str]] = {
    "stimulation_change": (
        "a full re-hash costs O(view bytes); in a closed loop the stimulation "
        "vector changes every tick, so this trigger would put a full-graph "
        "re-hash in the hot path. Stimulation identity is instead derived from "
        "the executed arrays inside the step and bound into every record."
    ),
    "trace_flush": (
        "the engine does not flush traces; buffering and flushing live in the "
        "controller and service layers, which have their own verification."
    ),
}


class ViewIntegrityError(RuntimeError):
    """A runtime view was rebound, resized, unfrozen or mutated in place."""


@dataclass(frozen=True, slots=True)
class BufferPin:
    """Everything about a buffer that must not change between steps."""

    name: str
    data_ptr: int
    nbytes: int
    dtype_str: str
    shape: tuple[int, ...]

    @classmethod
    def capture(cls, name: str, array: np.ndarray[Any, Any]) -> BufferPin:
        return cls(
            name=name,
            data_ptr=int(array.__array_interface__["data"][0]),
            nbytes=int(array.nbytes),
            dtype_str=str(array.dtype.str),
            shape=tuple(int(size) for size in array.shape),
        )


@dataclass(frozen=True, slots=True)
class VerificationPolicy:
    """Declared, hashable cadence for full re-verification."""

    mode: str = "amortized_in_thread"
    full_coverage_period_steps: int = DEFAULT_FULL_COVERAGE_PERIOD_STEPS
    unconditional_triggers: tuple[str, ...] = UNCONDITIONAL_VERIFICATION_TRIGGERS

    def __post_init__(self) -> None:
        if self.mode not in {"amortized_in_thread", "load_only"}:
            raise ViewIntegrityError(f"unknown verification mode {self.mode!r}")
        if self.full_coverage_period_steps < 1:
            raise ViewIntegrityError("full coverage period must be at least one step")

    @property
    def identity_sha256(self) -> str:
        return sha256_json(
            {
                "full_coverage_period_steps": self.full_coverage_period_steps,
                "mode": self.mode,
                "unconditional_triggers": list(self.unconditional_triggers),
            }
        )

    def bytes_per_step(self, total_bytes: int) -> int:
        if self.mode == "load_only":
            return 0
        return -(-total_bytes // self.full_coverage_period_steps)


@dataclass(frozen=True, slots=True)
class VerifiedView:
    """Token proving which bytes were verified, and when."""

    view_sha256: str
    verified_at_step: int
    buffers: tuple[BufferPin, ...]


def capture_pins(named_arrays: Sequence[tuple[str, np.ndarray[Any, Any]]]) -> tuple[BufferPin, ...]:
    return tuple(BufferPin.capture(name, array) for name, array in named_arrays)


def assert_buffers_pinned(
    named_arrays: Sequence[tuple[str, np.ndarray[Any, Any]]],
    token: VerifiedView,
) -> None:
    """O(1) check that the verified buffers are still exactly those buffers.

    Costs a few microseconds regardless of graph size. It does *not* detect
    in-place mutation through a retained alias; :class:`IncrementalViewDigest`
    is what covers that, within the bound stated in the module docstring.
    """

    if len(named_arrays) != len(token.buffers):
        raise ViewIntegrityError("runtime view buffer count changed")
    for (name, array), pin in zip(named_arrays, token.buffers, strict=True):
        if name != pin.name:
            raise ViewIntegrityError(f"runtime view buffer order changed at {name!r}")
        if array.flags.writeable:
            raise ViewIntegrityError(f"runtime view buffer {name!r} was made writeable")
        observed = BufferPin.capture(name, array)
        if observed != pin:
            raise ViewIntegrityError(f"runtime view buffer {name!r} was rebound or reshaped")


def hash_buffers(named_arrays: Sequence[tuple[str, np.ndarray[Any, Any]]]) -> str:
    """SHA-256 over the concatenated buffers in declared order."""

    digest = hashlib.sha256()
    for _, array in named_arrays:
        with memoryview(array).cast("B") as block:
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SliceVerification:
    """Outcome of re-hashing one amortised slice."""

    slice_index: int
    start_offset: int
    end_offset: int
    matched: bool
    completed_view_sha256: str | None


class IncrementalViewDigest:
    """Amortised full re-verification, bounded and deterministic per step.

    The view is treated as one logical byte stream in declared buffer order and
    cut into fixed slices of ``bytes_per_step``. At construction each slice's
    digest is recorded, together with the digest of the whole stream. Each call
    to :meth:`advance` re-hashes exactly one slice and compares it against the
    recorded value, so a mismatch is reported on the step that reads the
    changed bytes rather than at the end of the pass. The per-step cost is one
    slice hash -- the same work the previous pass-completion-only version did,
    with no second hash added.

    Slice ``i`` is visited on every step congruent to ``i`` modulo
    :attr:`slice_count`, so no byte goes unread for longer than
    :attr:`full_coverage_steps` steps and the detection bound for a persistent
    mutation equals the declared coverage period. A mutation reverted inside
    one such window is not detected; see the module docstring.

    There is no background thread, so which slice is read on which step is a
    deterministic function of the step index and is bindable into a trace.
    """

    def __init__(
        self,
        named_arrays: Sequence[tuple[str, np.ndarray[Any, Any]]],
        *,
        bytes_per_step: int,
    ) -> None:
        if bytes_per_step < 1:
            raise ViewIntegrityError("amortised verification needs a positive byte budget")
        self._arrays = tuple(named_arrays)
        self._bytes_per_step = bytes_per_step
        self._sizes = tuple(int(array.nbytes) for _, array in self._arrays)
        self._total_bytes = sum(self._sizes)
        if self._total_bytes < 1:
            raise ViewIntegrityError("amortised verification needs a non-empty view")
        self._starts = tuple(range(0, self._total_bytes, bytes_per_step))
        whole = hashlib.sha256()
        expected: list[str] = []
        for start in self._starts:
            length = min(bytes_per_step, self._total_bytes - start)
            expected.append(self._hash_range(start, length, whole))
        self._expected = tuple(expected)
        self._view_sha256 = whole.hexdigest()
        self._index = 0

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def slice_count(self) -> int:
        return len(self._starts)

    @property
    def full_coverage_steps(self) -> int:
        """Steps between two reads of the same byte; the true detection bound."""

        return len(self._starts)

    @property
    def offset(self) -> int:
        """Byte offset of the slice the next :meth:`advance` will read."""

        return self._starts[self._index]

    @property
    def expected_view_sha256(self) -> str:
        """Digest of the whole stream as it stood when this object was built."""

        return self._view_sha256

    @property
    def expected_slice_digests(self) -> tuple[str, ...]:
        return self._expected

    def restart(self) -> None:
        """Rewind to slice zero without re-recording expectations."""

        self._index = 0

    def _hash_range(self, start: int, length: int, also: Any = None) -> str:
        """Digest ``length`` bytes of the logical stream starting at ``start``.

        The bytes are fed straight from the buffers' memoryviews -- no copy --
        into a fresh slice hasher and, at construction, into the whole-stream
        hasher as well. A slice that spans a buffer boundary is simply fed in
        two pieces; the logical stream is contiguous by definition of the
        declared buffer order.
        """

        slice_digest = hashlib.sha256()
        base = 0
        remaining = length
        offset = start
        for index, size in enumerate(self._sizes):
            if remaining <= 0:
                break
            if offset >= base + size:
                base += size
                continue
            local = offset - base
            take = min(size - local, remaining)
            with (
                memoryview(self._arrays[index][1]).cast("B") as view,
                view[local : local + take] as block,
            ):
                slice_digest.update(block)
                if also is not None:
                    also.update(block)
            offset += take
            remaining -= take
            base += size
        return slice_digest.hexdigest()

    def advance(self) -> SliceVerification:
        """Re-hash the next slice and compare it against its recorded digest."""

        index = self._index
        start = self._starts[index]
        length = min(self._bytes_per_step, self._total_bytes - start)
        observed = self._hash_range(start, length)
        self._index += 1
        completed = self._index >= len(self._starts)
        if completed:
            self._index = 0
        return SliceVerification(
            slice_index=index,
            start_offset=start,
            end_offset=start + length,
            matched=observed == self._expected[index],
            completed_view_sha256=self._view_sha256 if completed else None,
        )
