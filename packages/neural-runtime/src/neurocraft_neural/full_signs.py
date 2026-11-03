"""Transmitter-derived coarse-consensus edge signs over the full MaleCNS artifact.

Array-native duplicate of the fixture-scope policy
``neurocraft_neural.model.SignOverlay.coarse_presynaptic_consensus``: the sign
of every edge is the sign class of its *presynaptic* node's consensus
transmitter, casefolded, with anything outside the two disclosed sets (missing
rows included) mapping to zero. That fixture-scope implementation is pinned
(its bytes participate in downstream identities), so the transmitter sets are
**duplicated** here rather than imported from a shared constant. The
duplication is itself pinned by the policy-equality regression tests in
``tests/neural/test_full_signs.py``, which run the pinned overlay and this
module on the same graph and require edge-for-edge equality.

Where the consensus transmitter comes from
------------------------------------------
The full artifact stores no transmitter strings. It stores
``transmitter_source_rows``: one ``uint64`` per retained node pointing into the
pinned ``body-neurotransmitters-male-cns-v1.0.feather`` source table (source id
``neurotransmitters`` in the artifact manifest's ``sources[]``), with
``2**64 - 1`` as the sentinel for a node with no transmitter row. That table's
``consensus_nt`` column is exactly the field the fixture extraction records as
``transmitter.consensus_nt`` and the pinned fixture loader materialises as
``node.consensus_nt`` — the very value the pinned policy reads. The sibling
``annotation_source_rows`` array points at the body-annotations table, which
carries ``superclass``/``type``/... and no transmitter column, and is not used
here.

Fail-closed integrity gates
---------------------------
Before any sign is computed:

1. the annotations file on disk must hash to the SHA-256 (and byte length) the
   verified artifact manifest declares for the ``neurotransmitters`` source —
   the digest chain is external: ``load_full_runtime_artifact`` only accepts a
   manifest whose ``sources[]`` equal the trusted acquisition pins;
2. the source table's row count must equal the manifest's declared
   ``transmitter_source_rows`` count;
3. the non-sentinel pointer rows must be unique, in range, and equal in number
   to the manifest's ``retained_nodes_with_transmitter_row``;
4. the ``body`` column taken through the pointers must be bitwise identical to
   the artifact's ``body_ids`` for those nodes — one misaligned pointer would
   silently relabel transmitters, so the join is proven, not trusted.

Claim discipline
----------------
This is an annotation census plus derived arrays, nothing more. The coarse
whole-transmitter mapping (for example glutamate uniformly inhibitory) is a
disclosed, contestable modeling choice; nothing physiological or behavioral is
claimed by any number this module produces.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pyarrow.feather as feather  # type: ignore[import-untyped]
from neurocraft_connectome.full_runtime import FullRuntimeGraph
from neurocraft_connectome.model import canonical_json_bytes
from numpy.typing import NDArray

#: Policy identity. The implementation of record is the pinned fixture-scope
#: method; this module is its array-native duplicate, pinned by regression test.
SIGN_POLICY_ID: Final = "coarse_presynaptic_consensus"
SIGN_POLICY_IMPLEMENTATION: Final = (
    "neurocraft_neural.model.SignOverlay.coarse_presynaptic_consensus"
)

#: Duplicated from the pinned ``SignOverlay.coarse_presynaptic_consensus``.
#: Do not edit one copy without the other; the equality tests will refuse.
POSITIVE_TRANSMITTERS: Final = frozenset(
    {"acetylcholine", "dopamine", "octopamine", "serotonin", "tyramine"}
)
NEGATIVE_TRANSMITTERS: Final = frozenset({"gaba", "glutamate", "histamine"})

#: The pinned policy's fallback bucket for an empty/absent consensus value.
UNKNOWN_TRANSMITTER: Final = "unknown"

TRANSMITTER_SOURCE_ID: Final = "neurotransmitters"
BODY_COLUMN: Final = "body"
CONSENSUS_COLUMN: Final = "consensus_nt"
MISSING_TRANSMITTER_ROW: Final = (1 << 64) - 1
SIGN_LABELS: Final[dict[int, str]] = {1: "positive", -1: "negative", 0: "zero"}
_HASH_BLOCK_BYTES: Final = 8 * 1024 * 1024


class FullSignOverlayError(RuntimeError):
    """Raised when the overlay cannot be computed and verified exactly."""


def normalized_transmitter(value: str | None) -> str:
    """Casefold one consensus value; empty/absent becomes ``"unknown"``.

    Character for character the pinned policy's normalisation:
    ``transmitter.casefold() if transmitter else "unknown"``.
    """

    return value.casefold() if value else UNKNOWN_TRANSMITTER


def transmitter_sign(normalized: str) -> int:
    """Sign class of one normalized transmitter under the coarse policy."""

    if normalized in POSITIVE_TRANSMITTERS:
        return 1
    if normalized in NEGATIVE_TRANSMITTERS:
        return -1
    return 0


def node_signs_from_transmitters(
    transmitters: Sequence[str | None],
) -> NDArray[np.int8]:
    """Per-node int8 sign array from raw (or already normalized) consensus values."""

    signs = np.fromiter(
        (transmitter_sign(normalized_transmitter(value)) for value in transmitters),
        dtype=np.int8,
        count=len(transmitters),
    )
    signs.setflags(write=False)
    return signs


def edge_signs_from_node_signs(
    node_signs: NDArray[np.int8],
    edge_pre_indices: NDArray[Any],
) -> NDArray[np.int8]:
    """Gather per-edge signs as ``node_signs[edge_pre_indices]``, refusing bad input."""

    if node_signs.ndim != 1 or node_signs.dtype != np.int8:
        raise FullSignOverlayError("node signs must be a one-dimensional int8 array")
    if node_signs.size == 0:
        raise FullSignOverlayError("node sign array must not be empty")
    if not np.isin(node_signs, (-1, 0, 1)).all():
        raise FullSignOverlayError("node signs must be exactly -1, 0, or +1")
    if edge_pre_indices.ndim != 1 or not np.issubdtype(edge_pre_indices.dtype, np.integer):
        raise FullSignOverlayError("edge pre indices must be a one-dimensional integer array")
    if edge_pre_indices.size and (
        int(edge_pre_indices.min()) < 0 or int(edge_pre_indices.max()) >= node_signs.size
    ):
        raise FullSignOverlayError("edge pre index is outside the node sign array")
    edge_signs: NDArray[np.int8] = node_signs[edge_pre_indices]
    edge_signs = np.ascontiguousarray(edge_signs, dtype=np.int8)
    edge_signs.setflags(write=False)
    return edge_signs


def file_sha256(path: Path) -> str:
    """Block-wise SHA-256 of a file on disk."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(_HASH_BLOCK_BYTES)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FullSignOverlayResult:
    """Frozen census of one executed full-graph coarse-consensus overlay.

    ``edge_signs`` and ``node_signs`` are read-only int8 arrays. The counts are
    plain dictionaries for JSON friendliness; treat them as immutable.
    """

    policy_id: str
    policy_implementation: str
    artifact_manifest_sha256: str
    annotations_sha256: str
    node_signs: NDArray[np.int8]
    edge_signs: NDArray[np.int8]
    sign_counts: dict[str, int]
    per_transmitter_edge_signs: dict[str, dict[str, int]]
    node_transmitter_counts: dict[str, int]
    zeroing_transmitters: tuple[str, ...]
    zeroing_presynaptic_node_count: int
    nodes_missing_transmitter_row: int
    sign_array_sha256: str
    identity_sha256: str

    @property
    def edge_count(self) -> int:
        return int(self.edge_signs.size)

    @property
    def signed_edge_count(self) -> int:
        return self.sign_counts["positive"] + self.sign_counts["negative"]


def _verified_graph_array(graph: FullRuntimeGraph, name: str, dtype: str) -> NDArray[Any]:
    """Return one artifact array as numpy, re-checked against its manifest spec.

    Mirrors the licensed pattern in :mod:`neurocraft_neural.full_view`: the
    per-element ``Sequence[int]`` interface would box 25.6 M Python integers,
    so the verified mapping is read directly, and the digest re-check below is
    what makes that read self-validating rather than trusted.
    """

    specs = {spec.name: spec for spec in graph.manifest.arrays}
    try:
        spec = specs[name]
    except KeyError as exc:  # pragma: no cover - manifest schema is validated upstream
        raise FullSignOverlayError(f"full artifact has no array {name!r}") from exc
    mapped = getattr(graph, name)
    mapping = getattr(mapped, "_mmap", None)
    if mapping is None:  # pragma: no cover - upstream layout change
        raise FullSignOverlayError(f"array {name!r} does not expose a verified mapping")
    buffer = memoryview(mapping)
    if buffer.nbytes != spec.byte_length:
        raise FullSignOverlayError(f"array {name!r} mapping length does not match the manifest")
    if hashlib.sha256(buffer).hexdigest() != spec.sha256:
        raise FullSignOverlayError(f"array {name!r} mapping digest does not match the manifest")
    values: NDArray[Any] = np.frombuffer(buffer, dtype=dtype)
    if values.shape[0] != spec.count:
        raise FullSignOverlayError(f"array {name!r} element count does not match the manifest")
    return values


def _verified_annotations_table(graph: FullRuntimeGraph, annotations_path: Path) -> tuple[Any, str]:
    """Open the neurotransmitters table only after its declared digest matches."""

    by_id = {source.source_id: source for source in graph.manifest.source_artifacts}
    try:
        declared = by_id[TRANSMITTER_SOURCE_ID]
    except KeyError as exc:
        raise FullSignOverlayError(
            f"artifact manifest declares no {TRANSMITTER_SOURCE_ID!r} source"
        ) from exc
    try:
        observed_bytes = annotations_path.stat().st_size
    except OSError as exc:
        raise FullSignOverlayError(f"cannot read annotations table: {exc}") from exc
    if observed_bytes != declared.byte_length:
        raise FullSignOverlayError(
            "annotations table byte length mismatch: "
            f"declared {declared.byte_length}, observed {observed_bytes}"
        )
    observed_sha256 = file_sha256(annotations_path)
    if observed_sha256 != declared.sha256:
        raise FullSignOverlayError(
            "annotations table SHA-256 mismatch: "
            f"declared {declared.sha256}, observed {observed_sha256}"
        )
    table = feather.read_table(
        annotations_path, columns=[BODY_COLUMN, CONSENSUS_COLUMN], memory_map=True
    )
    declared_rows = graph.manifest.counts["transmitter_source_rows"]
    if int(table.num_rows) != declared_rows:
        raise FullSignOverlayError(
            "annotations table row count mismatch: "
            f"declared {declared_rows}, observed {int(table.num_rows)}"
        )
    return table, observed_sha256


def _node_transmitters(
    graph: FullRuntimeGraph,
    table: Any,
    transmitter_rows: NDArray[Any],
    body_ids: NDArray[Any],
) -> NDArray[np.object_]:
    """Normalized per-node consensus transmitters, join proven bitwise."""

    matched = transmitter_rows != np.uint64(MISSING_TRANSMITTER_ROW)
    matched_rows = transmitter_rows[matched].astype(np.int64)
    if matched_rows.size:
        if int(matched_rows.min()) < 0 or int(matched_rows.max()) >= int(table.num_rows):
            raise FullSignOverlayError("transmitter source row is out of table range")
        if np.unique(matched_rows).size != matched_rows.size:
            raise FullSignOverlayError("transmitter source rows are not unique")
    declared_matched = graph.manifest.counts["retained_nodes_with_transmitter_row"]
    if int(matched_rows.size) != declared_matched:
        raise FullSignOverlayError(
            "matched transmitter rows do not reconcile with the manifest: "
            f"declared {declared_matched}, observed {int(matched_rows.size)}"
        )
    table_bodies = table.column(BODY_COLUMN).combine_chunks().to_numpy(zero_copy_only=False)
    taken_bodies = table_bodies[matched_rows].astype(np.uint64)
    if not np.array_equal(taken_bodies, body_ids[matched].astype(np.uint64)):
        raise FullSignOverlayError(
            "body column taken through transmitter_source_rows is not bitwise "
            "identical to the artifact body IDs; refusing a misaligned join"
        )
    consensus_values = table.column(CONSENSUS_COLUMN).to_pylist()
    normalized: NDArray[np.object_] = np.asarray(
        [normalized_transmitter(value) for value in consensus_values], dtype=object
    )
    node_transmitters: NDArray[np.object_] = np.full(
        len(graph.body_ids), UNKNOWN_TRANSMITTER, dtype=object
    )
    node_transmitters[matched] = normalized[matched_rows]
    return node_transmitters


def compute_full_sign_overlay(
    graph: FullRuntimeGraph, annotations_path: Path
) -> FullSignOverlayResult:
    """Compute coarse-consensus signs for every edge of the full artifact.

    Vectorized end to end: the per-node sign lookup is composed with
    ``edge_pre_indices`` in one numpy gather. All integrity gates described in
    the module docstring run first and fail closed.
    """

    table, annotations_sha256 = _verified_annotations_table(graph, annotations_path)
    transmitter_rows = _verified_graph_array(graph, "transmitter_source_rows", "<u8")
    body_ids = _verified_graph_array(graph, "body_ids", "<u8")
    edge_pre_indices = _verified_graph_array(graph, "edge_pre_indices", "<u4")

    node_transmitters = _node_transmitters(graph, table, transmitter_rows, body_ids)
    node_signs = node_signs_from_transmitters(node_transmitters.tolist())
    edge_signs = edge_signs_from_node_signs(node_signs, edge_pre_indices)
    if edge_signs.size != graph.manifest.edge_count:
        raise FullSignOverlayError("edge sign count does not match the manifest edge count")

    raw_counts = np.bincount((edge_signs.astype(np.int64) + 1), minlength=3)
    sign_counts = {
        "positive": int(raw_counts[2]),
        "negative": int(raw_counts[0]),
        "zero": int(raw_counts[1]),
    }

    uniques, inverse, node_counts = np.unique(
        node_transmitters, return_inverse=True, return_counts=True
    )
    transmitter_names = [str(name) for name in uniques.tolist()]
    edge_codes = inverse.astype(np.int64)[edge_pre_indices]
    per_transmitter_edges = np.bincount(edge_codes, minlength=len(transmitter_names))
    per_transmitter_edge_signs = {
        name: {SIGN_LABELS[transmitter_sign(name)]: int(per_transmitter_edges[code])}
        for code, name in enumerate(transmitter_names)
        if int(per_transmitter_edges[code])
    }
    node_transmitter_counts = {
        name: int(count)
        for name, count in zip(transmitter_names, node_counts.tolist(), strict=True)
    }
    zeroing_transmitters = tuple(
        sorted(
            name for name, counts in per_transmitter_edge_signs.items() if counts.get("zero", 0) > 0
        )
    )
    zeroing_presynaptic_node_count = int(np.unique(edge_pre_indices[edge_signs == 0]).size)

    sign_array_sha256 = hashlib.sha256(edge_signs.tobytes()).hexdigest()
    identity_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "artifact_manifest_sha256": graph.manifest.manifest_file_sha256,
                "policy_id": SIGN_POLICY_ID,
                "sign_array_sha256": sign_array_sha256,
            }
        )
    ).hexdigest()
    return FullSignOverlayResult(
        policy_id=SIGN_POLICY_ID,
        policy_implementation=SIGN_POLICY_IMPLEMENTATION,
        artifact_manifest_sha256=graph.manifest.manifest_file_sha256,
        annotations_sha256=annotations_sha256,
        node_signs=node_signs,
        edge_signs=edge_signs,
        sign_counts=sign_counts,
        per_transmitter_edge_signs=per_transmitter_edge_signs,
        node_transmitter_counts=node_transmitter_counts,
        zeroing_transmitters=zeroing_transmitters,
        zeroing_presynaptic_node_count=zeroing_presynaptic_node_count,
        nodes_missing_transmitter_row=graph.manifest.counts[
            "retained_nodes_missing_transmitter_row"
        ],
        sign_array_sha256=sign_array_sha256,
        identity_sha256=identity_sha256,
    )
