"""Immutable public graph and provenance types.

The canonical graph stores source support and raw counts. Runtime dynamics must
refer to these arrays; signs, delays, gains, lesions, and stimulation are
separate overlays owned by the neural package.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from enum import StrEnum
from hashlib import sha256
from typing import Any

UINT64_MAX = (1 << 64) - 1


class FixtureTier(StrEnum):
    """Machine-readable evidence tier; never derived from a display label."""

    SYNTHETIC_ALGEBRA = "SYNTHETIC_ALGEBRA"
    MALECNS_FIXTURE = "MALECNS_FIXTURE"


@dataclass(frozen=True, slots=True)
class SourceArtifact:
    """A generation-qualified source object bound into a graph identity."""

    source_id: str
    filename: str
    byte_length: int
    sha256: str
    generation: str
    url: str


@dataclass(frozen=True, slots=True)
class FixturePin:
    """Independent trusted identity for a tracked fixture."""

    dataset_id: str
    fixture_id: str
    schema_version: int
    file_sha256: str
    payload_sha256: str
    semantic_sha256: str
    source_manifest_sha256: str
    source_artifacts: tuple[SourceArtifact, ...]


@dataclass(frozen=True, slots=True)
class RuntimeNode:
    """A source neuron with a stable runtime index and row lineage."""

    dense_index: int
    body_id: int
    roles: tuple[str, ...]
    side: str | None
    type_name: str | None
    instance: str | None
    superclass: str | None
    annotation_source_row: int
    transmitter_source_row: int | None
    predicted_nt: str | None
    consensus_nt: str | None
    ground_truth_nt: str | None
    canonical_sign_status: str


@dataclass(frozen=True, slots=True)
class RuntimeEdge:
    """One immutable directed source edge in target-major sparse order."""

    dense_index: int
    pre_index: int
    post_index: int
    body_pre: int
    body_post: int
    raw_count: int
    source_row: int
    canonical_sign_status: str


@dataclass(frozen=True, slots=True)
class RuntimeGraphManifest:
    """Content-addressed identity of the exact arrays used at runtime."""

    schema_version: int
    dataset_id: str
    fixture_id: str
    fixture_tier: FixtureTier
    fixture_file_sha256: str
    fixture_payload_sha256: str
    fixture_semantic_sha256: str
    source_manifest_sha256: str
    extraction_code_sha256: str
    extraction_configuration_sha256: str
    loader_name: str
    loader_version: str
    loader_implementation_sha256: str
    loader_configuration_sha256: str
    node_count: int
    edge_count: int
    raw_count_sum: int
    body_ids_sha256: str
    edge_support_sha256: str
    raw_counts_sha256: str
    source_rows_sha256: str
    lineage_sha256: str
    canonical_graph_sha256: str
    runtime_identity_sha256: str
    source_artifacts: tuple[SourceArtifact, ...]


@dataclass(frozen=True, slots=True)
class RuntimeGraph:
    """Immutable graph arrays plus their independently checkable manifest."""

    manifest: RuntimeGraphManifest
    nodes: tuple[RuntimeNode, ...]
    edges: tuple[RuntimeEdge, ...]

    @property
    def body_ids(self) -> tuple[int, ...]:
        return tuple(node.body_id for node in self.nodes)

    @property
    def edge_support_sha256(self) -> str:
        return self.manifest.edge_support_sha256

    @property
    def raw_counts_sha256(self) -> str:
        return self.manifest.raw_counts_sha256

    def index_of(self, body_id: int) -> int:
        """Resolve an exact source body ID without fuzzy name substitution."""

        lo = 0
        hi = len(self.nodes)
        while lo < hi:
            mid = (lo + hi) // 2
            candidate = self.nodes[mid].body_id
            if candidate < body_id:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(self.nodes) or self.nodes[lo].body_id != body_id:
            raise KeyError(f"body ID {body_id} is not present in graph")
        return lo

    def node_for(self, body_id: int) -> RuntimeNode:
        return self.nodes[self.index_of(body_id)]

    def role_body_ids(self, role: str) -> tuple[int, ...]:
        return tuple(node.body_id for node in self.nodes if role in node.roles)

    def assert_integrity(self) -> None:
        """Recompute all hot-array and manifest identities, refusing mutation."""

        from neurocraft_connectome.loader import GraphIntegrityError

        if self.manifest.node_count != len(self.nodes):
            raise GraphIntegrityError("runtime node count does not match manifest")
        if self.manifest.edge_count != len(self.edges):
            raise GraphIntegrityError("runtime edge count does not match manifest")
        if tuple(node.dense_index for node in self.nodes) != tuple(range(len(self.nodes))):
            raise GraphIntegrityError("runtime node dense indices are not canonical")
        body_ids = self.body_ids
        if body_ids != tuple(sorted(body_ids)) or len(set(body_ids)) != len(body_ids):
            raise GraphIntegrityError("runtime node body IDs are not unique ascending uint64")
        if any(body_id <= 0 or body_id > UINT64_MAX for body_id in body_ids):
            raise GraphIntegrityError("runtime node body ID is outside uint64")
        expected_edge_order = tuple(
            sorted(
                self.edges,
                key=lambda edge: (edge.body_post, edge.body_pre, edge.source_row),
            )
        )
        if self.edges != expected_edge_order:
            raise GraphIntegrityError("runtime edges are not in canonical target-major order")
        if tuple(edge.dense_index for edge in self.edges) != tuple(range(len(self.edges))):
            raise GraphIntegrityError("runtime edge dense indices are not canonical")
        seen_pairs: set[tuple[int, int]] = set()
        seen_rows: set[int] = set()
        for edge in self.edges:
            if not 0 <= edge.pre_index < len(self.nodes) or not 0 <= edge.post_index < len(
                self.nodes
            ):
                raise GraphIntegrityError("runtime edge dense endpoint is out of range")
            if self.nodes[edge.pre_index].body_id != edge.body_pre:
                raise GraphIntegrityError("runtime edge pre-index/body-ID mismatch")
            if self.nodes[edge.post_index].body_id != edge.body_post:
                raise GraphIntegrityError("runtime edge post-index/body-ID mismatch")
            if edge.raw_count <= 0 or edge.raw_count > UINT64_MAX:
                raise GraphIntegrityError("runtime raw edge count is outside positive uint64")
            if edge.source_row < 0 or edge.source_row > UINT64_MAX:
                raise GraphIntegrityError("runtime edge source row is outside uint64")
            if edge.canonical_sign_status != "unknown":
                raise GraphIntegrityError("canonical edge sign status was overwritten")
            pair = (edge.body_pre, edge.body_post)
            if pair in seen_pairs:
                raise GraphIntegrityError("runtime graph contains a duplicate directed pair")
            if edge.source_row in seen_rows:
                raise GraphIntegrityError("runtime graph contains a duplicate source row")
            seen_pairs.add(pair)
            seen_rows.add(edge.source_row)
        if sum(edge.raw_count for edge in self.edges) != self.manifest.raw_count_sum:
            raise GraphIntegrityError("runtime raw count sum does not match manifest")

        observed = graph_hashes(self.nodes, self.edges)
        expected = {
            "body_ids_sha256": self.manifest.body_ids_sha256,
            "edge_support_sha256": self.manifest.edge_support_sha256,
            "raw_counts_sha256": self.manifest.raw_counts_sha256,
            "source_rows_sha256": self.manifest.source_rows_sha256,
            "lineage_sha256": self.manifest.lineage_sha256,
            "canonical_graph_sha256": self.manifest.canonical_graph_sha256,
        }
        if observed != expected:
            mismatches = sorted(name for name, value in expected.items() if observed[name] != value)
            raise GraphIntegrityError(
                "runtime graph arrays do not match manifest: " + ", ".join(mismatches)
            )
        identity = runtime_manifest_identity(self.manifest)
        if identity != self.manifest.runtime_identity_sha256:
            raise GraphIntegrityError("runtime graph manifest identity mismatch")


@dataclass(frozen=True, slots=True)
class SyntheticNode:
    """Explicitly synthetic node used only for algebra and harness tests."""

    body_id: int
    roles: tuple[str, ...] = ("intermediate",)
    side: str | None = None
    type_name: str | None = None
    superclass: str | None = None
    predicted_nt: str | None = None
    consensus_nt: str | None = None


@dataclass(frozen=True, slots=True)
class SyntheticEdge:
    """Explicitly synthetic directed raw-count edge."""

    body_pre: int
    body_post: int
    raw_count: int
    source_row: int


def canonical_json_bytes(value: Any) -> bytes:
    """Encode identity material without timestamps or platform formatting."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def _uint64_stream(rows: Iterable[tuple[int, ...]]) -> bytes:
    encoded = bytearray()
    for row in rows:
        for value in row:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("graph identity values must be integers")
            if value < 0 or value > UINT64_MAX:
                raise ValueError(f"value {value!r} is outside uint64")
            encoded.extend(struct.pack(">Q", value))
    return bytes(encoded)


def graph_hashes(nodes: tuple[RuntimeNode, ...], edges: tuple[RuntimeEdge, ...]) -> dict[str, str]:
    """Compute stable hashes over canonical topology, counts, and lineage."""

    body_ids_sha256 = sha256(_uint64_stream((node.body_id,) for node in nodes)).hexdigest()
    edge_support_sha256 = sha256(
        _uint64_stream((edge.body_pre, edge.body_post) for edge in edges)
    ).hexdigest()
    raw_counts_sha256 = sha256(
        _uint64_stream((edge.body_pre, edge.body_post, edge.raw_count) for edge in edges)
    ).hexdigest()
    source_rows_sha256 = sha256(_uint64_stream((edge.source_row,) for edge in edges)).hexdigest()
    lineage = [
        {
            "annotation_source_row": node.annotation_source_row,
            "body_id": node.body_id,
            "canonical_sign_status": node.canonical_sign_status,
            "consensus_nt": node.consensus_nt,
            "ground_truth_nt": node.ground_truth_nt,
            "instance": node.instance,
            "predicted_nt": node.predicted_nt,
            "roles": list(node.roles),
            "side": node.side,
            "superclass": node.superclass,
            "transmitter_source_row": node.transmitter_source_row,
            "type_name": node.type_name,
        }
        for node in nodes
    ]
    lineage_sha256 = sha256_json(lineage)
    canonical_graph_sha256 = sha256_json(
        {
            "body_ids_sha256": body_ids_sha256,
            "edge_support_sha256": edge_support_sha256,
            "lineage_sha256": lineage_sha256,
            "raw_counts_sha256": raw_counts_sha256,
            "source_rows_sha256": source_rows_sha256,
        }
    )
    return {
        "body_ids_sha256": body_ids_sha256,
        "edge_support_sha256": edge_support_sha256,
        "raw_counts_sha256": raw_counts_sha256,
        "source_rows_sha256": source_rows_sha256,
        "lineage_sha256": lineage_sha256,
        "canonical_graph_sha256": canonical_graph_sha256,
    }


def runtime_manifest_identity(manifest: RuntimeGraphManifest) -> str:
    material = {
        field.name: _identity_value(getattr(manifest, field.name))
        for field in fields(manifest)
        if field.name != "runtime_identity_sha256"
    }
    return sha256_json(material)


def _identity_value(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, tuple):
        return [_identity_value(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {field.name: _identity_value(getattr(value, field.name)) for field in fields(value)}
    return value


def manifest_identity_material(
    values: Mapping[str, Any], *, runtime_identity_sha256: str = ""
) -> RuntimeGraphManifest:
    """Construct a typed manifest and fill its self-excluding identity."""

    provisional = RuntimeGraphManifest(
        **values,
        runtime_identity_sha256=runtime_identity_sha256,
    )
    if runtime_identity_sha256:
        return provisional
    return RuntimeGraphManifest(
        **values,
        runtime_identity_sha256=runtime_manifest_identity(provisional),
    )
