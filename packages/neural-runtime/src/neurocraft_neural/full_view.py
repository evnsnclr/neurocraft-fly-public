"""Array-native CSR runtime view over a connectome graph.

The view holds exactly the four buffers the runtime dereferences:

===============  ==========  ==============================================
buffer           dtype       meaning
===============  ==========  ==============================================
``indptr``       ``int64``   CSR row offsets, length ``node_count + 1``
``indices``      ``uint32``  presynaptic dense index per edge
``weights``      ``float64`` effective weight per edge
``empty_rows``   ``int64``   indices of the in-degree-zero rows
===============  ==========  ==============================================

For the full MaleCNS artifact that is 294.0 MiB in total, against 7.8 GB of
Python objects for the same graph in the reference engine.

``empty_rows`` is in that list because it is *read by the numpy kernel*:
``NumpyReduceatKernel.matvec`` ends with ``out[empty_rows] = 0.0``. An earlier
revision hashed and pinned only the first three, so pointing ``empty_rows`` at
a non-empty row silently zeroed that row's entire recurrent drive with no
fault, no digest change and no bound violated. Every buffer the runtime
dereferences is hashed, pinned and re-verified; that is the rule, and
``empty_rows`` is a buffer the runtime dereferences.

Bit-exact weight derivation
---------------------------
``weights[k] = sign[k] * count[k] / max(1, sum of count over incoming edges of
post[k])``, which is character for character the reference engine's
``sign_overlay.values[k] * weight_overlay.values[k] / max(1,
incoming_denominator[post])``. The result is bit-identical to the reference,
not merely close, and the tests assert ``==`` rather than ``allclose``:

* raw counts are exact integers (max 2,591 in the real artifact), so their
  float64 images are exact;
* denominators are accumulated in **int64**, never in floating point, and the
  construction refuses any graph whose count total reaches ``2**53``, so the
  denominator's float64 image is exact too;
* ``sign * count`` is exact for ``sign`` in ``{-1, 0, +1}``;
* one correctly-rounded IEEE-754 division then produces each weight.

Row-L1 bound, stated to the ulp
-------------------------------
The reference's row-L1 normalisation makes every row's absolute weight sum
equal to 1.0 *in exact arithmetic* whenever the row is non-empty: the weights
of a row are ``count_k / sum(count)``. In float64 the individual divisions are
each correctly rounded and their sum is then rounded again, so the realised
row L1 can exceed 1.0 by a few ulps. The measured maximum over the real
25,582,938-edge artifact is ``1.0000000000000004`` -- two ulps above one, not
at or below one. The contraction argument is unaffected: the sup-norm
Lipschitz constant is ``leak + (1 - leak) * recurrent_gain * max_row_l1``,
which at the pinned parameters is ``0.98187...`` whether the row L1 is 1.0 or
1.0000000000000004, so the divergence against the reference still plateaus.
What is *not* true, and was previously written here, is that the derivation
bounds the sum at exactly 1.0. :data:`ROW_L1_ROUNDING_TOLERANCE` names the
slack that the derivation actually permits.

The empty-row trap
------------------
The real graph has 370 in-degree-zero rows. ``np.add.reduceat`` returns
``operand[start]`` for a zero-length segment, not ``0.0``, so the naive
segmented reduction silently injects a foreign edge's product into those rows,
stays inside ``[-1, 1]``, and never raises. Both the denominator derivation
here and the numpy fallback kernel therefore zero ``empty_rows`` explicitly,
and a named test exercises leading, trailing and consecutive empty rows.
A trailing empty row is a second, separate trap: it makes the last start
offset equal the edge count, which ``reduceat`` rejects outright. The operand
is padded with one neutral element rather than having its offsets clipped --
clipping would shorten the last non-empty row's segment and drop an edge.

Scope
-----
Phase 1 is uniform delay 1 only. Heterogeneous delays turn the gather into a
2-D read over a ``(D, N)`` history buffer and require bucketing edges by delay
class, which re-permutes the CSR and therefore changes the summation order and
the dynamics identity. That is a separate backend identity and a separate
benchmark row, not a flag on this one.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from neurocraft_connectome import RuntimeGraph
from neurocraft_connectome.model import sha256_json
from numpy.typing import NDArray

from neurocraft_neural.model import (
    ZERO_SHA256,
    DelayOverlay,
    NeuralConfigurationError,
    SignOverlay,
    TopologyOverlay,
    WeightOverlay,
)

#: The exact-integer ceiling for float64. Denominators are accumulated in
#: int64 and refused at or above this bound so their float64 images stay exact.
EXACT_INTEGER_LIMIT: Final = 1 << 53
UINT32_LIMIT: Final = 1 << 32

#: Declared hashing and pinning order. Every buffer the runtime dereferences
#: appears here, ``empty_rows`` included.
BUFFER_ORDER: Final[tuple[str, ...]] = ("indptr", "indices", "weights", "empty_rows")

#: Rounding slack permitted above a row L1 of 1.0. Each weight is one correctly
#: rounded division and the row sum is one more rounding, so a few ulps of
#: excess is arithmetic, not a derivation error. Eight ulps of 1.0 leaves ample
#: margin over the ``1.0000000000000004`` measured on the real artifact while
#: still refusing any weight overlay that is not row-L1 normalised.
ROW_L1_ROUNDING_TOLERANCE: Final = 8.0 * float(np.finfo(np.float64).eps)

_MODULE_PATH: Final = Path(__file__).resolve()


def view_construction_sha256() -> str:
    """SHA-256 of this module, byte-binding the weight derivation to identity."""

    return hashlib.sha256(_MODULE_PATH.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class ViewProvenance:
    """Where the view's numbers came from, in the repo's digest vocabulary."""

    origin: str
    graph_identity_sha256: str
    sign_overlay_sha256: str = ZERO_SHA256
    delay_overlay_sha256: str = ZERO_SHA256
    weight_overlay_sha256: str = ZERO_SHA256
    topology_overlay_sha256: str = ZERO_SHA256
    source_manifest_sha256: str = ZERO_SHA256

    @property
    def identity_sha256(self) -> str:
        return sha256_json(
            {
                "delay_overlay_sha256": self.delay_overlay_sha256,
                "graph_identity_sha256": self.graph_identity_sha256,
                "origin": self.origin,
                "sign_overlay_sha256": self.sign_overlay_sha256,
                "source_manifest_sha256": self.source_manifest_sha256,
                "topology_overlay_sha256": self.topology_overlay_sha256,
                "weight_overlay_sha256": self.weight_overlay_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class ViewValidation:
    """The structural facts checked once at construction, in hashable form.

    ``max_observed_index`` is the fact that licenses the sparse kernels'
    gather. It is recomputed from the buffers by the engine at load and hashed
    into every step record, so a hand-built :class:`FullRuntimeView` cannot
    assert a bound it does not satisfy. It is deliberately *not* recomputed per
    step: that is an O(E) scan, which is the cost this backend exists to avoid.
    Corruption after construction is contained by the kernels themselves, which
    range-check every offset and index they dereference.
    """

    node_count: int
    edge_count: int
    max_observed_index: int
    index_upper_bound_exclusive: int
    empty_row_count: int
    max_in_degree: int

    @classmethod
    def derive(
        cls,
        *,
        indptr: NDArray[np.int64],
        indices: NDArray[np.uint32],
        empty_rows: NDArray[np.int64],
    ) -> ViewValidation:
        node_count = int(indptr.shape[0]) - 1
        edge_count = int(indices.shape[0])
        degrees = np.diff(indptr)
        return cls(
            node_count=node_count,
            edge_count=edge_count,
            max_observed_index=int(indices.max()) if edge_count else -1,
            index_upper_bound_exclusive=node_count,
            empty_row_count=int(empty_rows.shape[0]),
            max_in_degree=int(degrees.max()) if node_count else 0,
        )

    @property
    def identity_sha256(self) -> str:
        return sha256_json(
            {
                "edge_count": self.edge_count,
                "empty_row_count": self.empty_row_count,
                "index_upper_bound_exclusive": self.index_upper_bound_exclusive,
                "max_in_degree": self.max_in_degree,
                "max_observed_index": self.max_observed_index,
                "node_count": self.node_count,
            }
        )


@dataclass(frozen=True, slots=True)
class FullRuntimeView:
    """Frozen CSR arrays plus the digest that binds them."""

    indptr: NDArray[np.int64]
    indices: NDArray[np.uint32]
    weights: NDArray[np.float64]
    empty_rows: NDArray[np.int64]
    node_count: int
    edge_count: int
    max_in_degree: int
    view_sha256: str
    provenance: ViewProvenance
    validation: ViewValidation

    def named_buffers(self) -> tuple[tuple[str, np.ndarray[Any, Any]], ...]:
        """Every buffer the runtime dereferences, in the one declared order."""

        return (
            ("indptr", self.indptr),
            ("indices", self.indices),
            ("weights", self.weights),
            ("empty_rows", self.empty_rows),
        )

    @property
    def buffer_bytes(self) -> int:
        return int(sum(array.nbytes for _, array in self.named_buffers()))

    @property
    def max_row_l1(self) -> float:
        """Largest row absolute weight sum.

        Equals 1.0 in exact arithmetic for every non-empty row; in float64 it
        may exceed 1.0 by a few ulps. See :data:`ROW_L1_ROUNDING_TOLERANCE`.
        """

        if self.edge_count == 0 or self.node_count == 0:
            return 0.0
        magnitudes = np.abs(self.weights)
        sums = np.add.reduceat(_padded_for_reduceat(magnitudes, self.indptr), self.indptr[:-1])
        sums[self.empty_rows] = 0.0
        return float(sums.max())


def _freeze(array: NDArray[Any]) -> NDArray[Any]:
    array.flags.writeable = False
    return array


def _row_denominators(
    counts: NDArray[np.int64],
    indptr: NDArray[np.int64],
    empty_rows: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Exact int64 per-row sums, with the two ``reduceat`` traps handled."""

    node_count = int(indptr.shape[0]) - 1
    edge_count = int(counts.shape[0])
    if edge_count == 0:
        return np.zeros(node_count, dtype=np.int64)
    denominators = np.add.reduceat(_padded_for_reduceat(counts, indptr), indptr[:-1])
    denominators = denominators.astype(np.int64, copy=False)
    denominators[empty_rows] = 0
    return denominators


def _padded_for_reduceat(operand: NDArray[Any], indptr: NDArray[np.int64]) -> NDArray[Any]:
    """Append one neutral element when the last rows are empty.

    A trailing empty row leaves the final start offset equal to the element
    count, which ``reduceat`` rejects as out of range. Clipping the offsets is
    *not* a fix: it would shorten the last non-empty row's segment and silently
    drop its final edge. Appending a zero keeps every offset in range, leaves
    every segment intact, and contributes an additively neutral term.
    """

    count = int(operand.shape[0])
    if int(indptr[-2]) < count:
        return operand
    return np.concatenate([operand, np.zeros(1, dtype=operand.dtype)])


def build_view(
    *,
    indptr: Sequence[int] | NDArray[Any],
    pre_indices: Sequence[int] | NDArray[Any],
    raw_counts: Sequence[int] | NDArray[Any],
    signs: Sequence[int] | NDArray[Any],
    provenance: ViewProvenance,
) -> FullRuntimeView:
    """Validate, derive effective weights, freeze, and digest.

    Every check here runs exactly once. After it returns, the buffers are
    read-only and the engine re-pins them (address, length, dtype, shape,
    writeability) on every step instead of re-deriving anything.
    """

    if sys.byteorder != "little":
        raise NeuralConfigurationError(
            "the array-native backend declares little-endian buffer digests"
        )
    offsets = np.ascontiguousarray(indptr, dtype=np.int64)
    if offsets.ndim != 1 or offsets.shape[0] < 1:
        raise NeuralConfigurationError("indptr must be a one-dimensional CSR offset array")
    node_count = int(offsets.shape[0]) - 1
    columns = np.ascontiguousarray(pre_indices, dtype=np.uint32)
    counts = np.ascontiguousarray(raw_counts, dtype=np.int64)
    sign_values = np.ascontiguousarray(signs, dtype=np.int64)
    edge_count = int(columns.shape[0])
    if counts.shape[0] != edge_count or sign_values.shape[0] != edge_count:
        raise NeuralConfigurationError("edge arrays must all have one entry per edge")
    if node_count >= UINT32_LIMIT:
        raise NeuralConfigurationError("dense indices must fit in uint32")
    if int(offsets[0]) != 0 or int(offsets[-1]) != edge_count:
        raise NeuralConfigurationError("indptr endpoints do not reconcile with the edge count")
    degrees = np.diff(offsets)
    if bool((degrees < 0).any()):
        raise NeuralConfigurationError("indptr is not non-decreasing")
    if edge_count and int(columns.max()) >= node_count:
        raise NeuralConfigurationError("presynaptic dense index is out of range")
    if edge_count and int(counts.min()) < 1:
        raise NeuralConfigurationError("edge raw counts must be positive integers")
    if not bool(np.isin(sign_values, (-1, 0, 1)).all()):
        raise NeuralConfigurationError("edge signs must be exactly -1, 0, or +1")
    if edge_count and int(counts.sum()) >= EXACT_INTEGER_LIMIT:
        raise NeuralConfigurationError("raw count total exceeds the exact float64 integer range")

    empty_rows = np.flatnonzero(degrees == 0).astype(np.int64, copy=False)
    denominators = _row_denominators(counts, offsets, empty_rows)
    counts_float = counts.astype(np.float64)
    numerators = sign_values.astype(np.float64) * counts_float
    divisors = np.repeat(np.maximum(denominators, 1).astype(np.float64), degrees)
    weights = numerators / divisors if edge_count else np.zeros(0, dtype=np.float64)
    if not bool(np.isfinite(weights).all()):
        raise NeuralConfigurationError("effective weight derivation produced a non-finite value")

    frozen_offsets = _freeze(offsets)
    frozen_columns = _freeze(columns)
    frozen_weights = _freeze(np.ascontiguousarray(weights, dtype=np.float64))
    frozen_empty_rows = _freeze(empty_rows)
    digest = hashlib.sha256()
    for array in (frozen_offsets, frozen_columns, frozen_weights, frozen_empty_rows):
        with memoryview(array).cast("B") as block:
            digest.update(block)
    return FullRuntimeView(
        indptr=frozen_offsets,
        indices=frozen_columns,
        weights=frozen_weights,
        empty_rows=frozen_empty_rows,
        node_count=node_count,
        edge_count=edge_count,
        max_in_degree=int(degrees.max()) if node_count else 0,
        view_sha256=digest.hexdigest(),
        provenance=provenance,
        validation=ViewValidation.derive(
            indptr=frozen_offsets,
            indices=frozen_columns,
            empty_rows=frozen_empty_rows,
        ),
    )


def view_from_runtime_graph(
    graph: RuntimeGraph,
    *,
    sign_overlay: SignOverlay,
    weight_overlay: WeightOverlay | None = None,
    topology_overlay: TopologyOverlay | None = None,
    delay_overlay: DelayOverlay | None = None,
) -> FullRuntimeView:
    """Build the view from the same objects the reference engine consumes.

    This is the path the fidelity tests use: identical graph, identical
    overlays, two engines. Overlay defaults mirror
    :class:`~neurocraft_neural.engine.NeuralEngine` exactly.

    Only CSR-ordered topologies are accepted. The reference sums each row in
    ascending edge index; that equals CSR stored order only when the post
    indices are non-decreasing. A rewired-topology control breaks that, so it
    is refused here rather than silently re-permuted into a different
    summation order.
    """

    graph.assert_integrity()
    sign_overlay.assert_compatible(graph)
    resolved_weight = weight_overlay or WeightOverlay.canonical(graph)
    resolved_weight.assert_compatible(graph)
    resolved_topology = topology_overlay or TopologyOverlay.canonical(graph)
    resolved_topology.assert_compatible(graph)
    resolved_delay = delay_overlay or DelayOverlay.uniform(
        graph, 1, provenance="dynamics_config_uniform_default"
    )
    resolved_delay.assert_compatible(graph)
    if any(step != 1 for step in resolved_delay.steps):
        raise NeuralConfigurationError(
            "the array-native Phase 1 backend implements uniform delay 1 only"
        )

    node_count = len(graph.nodes)
    post_indices = np.asarray(resolved_topology.post_indices, dtype=np.int64)
    if post_indices.shape[0] and bool((np.diff(post_indices) < 0).any()):
        raise NeuralConfigurationError(
            "the array-native backend requires target-major (CSR) edge order"
        )
    degrees = np.bincount(post_indices, minlength=node_count).astype(np.int64, copy=False)
    indptr = np.zeros(node_count + 1, dtype=np.int64)
    np.cumsum(degrees, out=indptr[1:])
    return build_view(
        indptr=indptr,
        pre_indices=np.asarray(resolved_topology.pre_indices, dtype=np.int64),
        raw_counts=np.asarray(resolved_weight.values, dtype=np.int64),
        signs=np.asarray(sign_overlay.values, dtype=np.int64),
        provenance=ViewProvenance(
            origin="runtime_graph",
            graph_identity_sha256=graph.manifest.runtime_identity_sha256,
            sign_overlay_sha256=sign_overlay.identity_sha256,
            delay_overlay_sha256=resolved_delay.identity_sha256,
            weight_overlay_sha256=resolved_weight.identity_sha256,
            topology_overlay_sha256=resolved_topology.identity_sha256,
            source_manifest_sha256=graph.manifest.source_manifest_sha256,
        ),
    )


def view_from_full_runtime_graph(
    graph: Any,
    *,
    signs: Sequence[int] | NDArray[Any] | None = None,
    sign_provenance: str = "uniform_positive_placeholder_pending_transmitter_overlay",
    zero_copy_indices: bool = False,
) -> FullRuntimeView:
    """Build the view from the verified memory-mapped full artifact.

    ``graph`` is a ``neurocraft_connectome.full_runtime.FullRuntimeGraph``; it
    is typed loosely so that importing this module never drags in the full
    artifact machinery.

    The loader binds each array's SHA-256 to its *mapping*, so the bytes read
    here are the bytes that were verified; :func:`_verified_mapped_array`
    re-checks that digest against the manifest spec before handing the buffer
    over, which also makes the zero-copy access to the mapping self-validating
    rather than trusted.

    ``signs`` must be supplied by the caller. The default placeholder is ``+1``
    everywhere and is explicitly *not* a transmitter-derived sign overlay; its
    provenance string says so, and a run using it must never be presented as
    Faithful Organism Mode dynamics.

    With ``zero_copy_indices`` the 102 MB presynaptic index array aliases the
    mapping instead of being copied. The view then keeps the mapping exported,
    so ``FullRuntimeGraph.close()`` will raise ``BufferError`` until the view is
    dropped, and the on-disk residual disclosed by the loader becomes reachable
    at runtime. The default copies.

    A previous revision claimed the amortised checkpoint mitigated that
    residual. For *value* corruption within the declared index range it does,
    within the bound in :mod:`neurocraft_neural.view_verification`, and only
    for a persistent change. For an *out-of-range* index it did not and could
    not: the kernel dereferences the buffer at the top of the step, and the
    checkpoint runs afterwards, so the bad read happens first. That class is
    now contained in the kernels instead -- both range-check every offset and
    index they dereference and refuse the call -- and this option's honest
    description is "cheaper load, an on-disk write window that the checkpoint
    detects late and only while it persists", not "mitigated".
    """

    offsets = _verified_mapped_array(graph, "incoming_offsets", "<u8").astype(np.int64)
    columns = _verified_mapped_array(graph, "edge_pre_indices", "<u4")
    if not zero_copy_indices:
        columns = columns.copy()
    counts = _verified_mapped_array(graph, "edge_raw_counts", "<u8").astype(np.int64)
    edge_count = int(counts.shape[0])
    sign_values = np.ones(edge_count, dtype=np.int64) if signs is None else signs
    return build_view(
        indptr=offsets,
        pre_indices=columns,
        raw_counts=counts,
        signs=sign_values,
        provenance=ViewProvenance(
            origin=f"full_runtime_artifact:{sign_provenance}",
            graph_identity_sha256=graph.manifest.provenance_graph_sha256,
            sign_overlay_sha256=hashlib.sha256(sign_provenance.encode("utf-8")).hexdigest(),
            delay_overlay_sha256=hashlib.sha256(b"uniform_delay_1").hexdigest(),
            weight_overlay_sha256=hashlib.sha256(b"canonical_source_raw_counts").hexdigest(),
            topology_overlay_sha256=graph.manifest.structural_graph_sha256,
            source_manifest_sha256=graph.manifest.source_manifest_sha256,
        ),
    )


def _verified_mapped_array(graph: Any, name: str, dtype: str) -> NDArray[Any]:
    """Return the mapped bytes of one artifact array, re-checked against its spec.

    ``MappedUnsignedArray`` exposes only a per-element ``Sequence[int]``
    interface, which would cost 25.6 M Python integer boxes per array. The
    underlying mapping is read directly instead, and the digest comparison
    below is what licenses that: if the private attribute ever stops being the
    verified mapping, or its length stops matching the manifest, construction
    fails closed instead of quietly reading something else.
    """

    specs = {spec.name: spec for spec in graph.manifest.arrays}
    try:
        spec = specs[name]
    except KeyError as exc:  # pragma: no cover - manifest schema is validated upstream
        raise NeuralConfigurationError(f"full artifact has no array {name!r}") from exc
    mapped = getattr(graph, name)
    mapping = getattr(mapped, "_mmap", None)
    if mapping is None:  # pragma: no cover - upstream layout change
        raise NeuralConfigurationError(f"array {name!r} does not expose a verified mapping")
    buffer = memoryview(mapping)
    if buffer.nbytes != spec.byte_length:
        raise NeuralConfigurationError(f"array {name!r} mapping length does not match the manifest")
    if hashlib.sha256(buffer).hexdigest() != spec.sha256:
        raise NeuralConfigurationError(f"array {name!r} mapping digest does not match the manifest")
    values: NDArray[Any] = np.frombuffer(buffer, dtype=dtype)
    if values.shape[0] != spec.count:
        raise NeuralConfigurationError(f"array {name!r} element count does not match the manifest")
    return values
