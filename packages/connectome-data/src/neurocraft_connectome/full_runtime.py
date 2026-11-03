"""Verified memory-mapped access to the full MaleCNS runtime artifact."""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import re
import stat
import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, overload

from neurocraft_connectome.loader import MALECNS_FIXTURE_V1_PIN
from neurocraft_connectome.model import SourceArtifact, canonical_json_bytes

FULL_ARTIFACT_ID = "male-cns-v1.0-superclass-v1"
FULL_ARTIFACT_SCHEMA_VERSION = 1
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_HASH_BLOCK_BYTES = 8 * 1024 * 1024
_NO_FOLLOW_FLAG: int = getattr(os, "O_NOFOLLOW", 0)
MALECNS_SOURCE_MANIFEST_SHA256 = "9412105f54884a57521bbdf7fe46659caab559ef7877e6a62af95f50e1ee3768"
ARRAY_LAYOUT: dict[str, tuple[str, int]] = {
    "annotation_source_rows": ("uint64-le", 8),
    "body_ids": ("uint64-le", 8),
    "edge_post_indices": ("uint32-le", 4),
    "edge_pre_indices": ("uint32-le", 4),
    "edge_raw_counts": ("uint64-le", 8),
    "edge_source_rows": ("uint64-le", 8),
    "incoming_offsets": ("uint64-le", 8),
    "transmitter_source_rows": ("uint64-le", 8),
}
FULL_RETENTION_POLICY: dict[str, Any] = {
    "annotation_retention": "superclass is non-null and UTF-8 length > 0",
    "directed_pair_policy": "preserve each source row; reject duplicate pairs",
    "edge_order": ["post_dense_index", "pre_dense_index"],
    "edge_retention": "both body_pre and body_post are retained nodes",
    "node_order": ["body_id ascending"],
    "self_edge_policy": "preserve",
    "threshold": None,
    "weight_policy": "preserve positive raw source weight as uint64",
}
BOUNDARY_ROLE_BODY_IDS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("motor", (800241, 800917)),
    ("sensory", (77185, 805722)),
)
BOUNDARY_BODY_SUPERCLASSES: tuple[tuple[int, str], ...] = (
    (77185, "cb_sensory"),
    (800241, "vnc_motor"),
    (800917, "vnc_motor"),
    (805722, "vnc_sensory"),
)
COUNT_KEYS = {
    "annotation_duplicate_body_ids",
    "annotation_source_rows",
    "excluded_edge_rows",
    "excluded_neither_endpoint_rows",
    "excluded_post_only_rows",
    "excluded_pre_only_rows",
    "excluded_self_edge_rows",
    "excluded_self_edge_weight_sum",
    "excluded_superclass_empty_rows",
    "excluded_superclass_null_rows",
    "excluded_transmitter_rows",
    "excluded_weight_sum",
    "raw_duplicate_directed_pairs",
    "raw_edge_rows",
    "raw_self_edge_rows",
    "raw_self_edge_weight_sum",
    "raw_weight_max",
    "raw_weight_min",
    "raw_weight_sum",
    "retained_duplicate_directed_pairs",
    "retained_edge_rows",
    "retained_node_rows",
    "retained_self_edge_rows",
    "retained_self_edge_weight_sum",
    "retained_nodes_missing_transmitter_row",
    "retained_nodes_with_transmitter_row",
    "retained_weight_max",
    "retained_weight_min",
    "retained_weight_sum",
    "transmitter_duplicate_body_ids",
    "transmitter_source_rows",
}
_MANIFEST_KEYS = {
    "array_encoding",
    "arrays",
    "artifact_id",
    "boundary_role_overlay",
    "builder_configuration_sha256",
    "builder_implementation_sha256",
    "counts",
    "dataset",
    "dataset_id",
    "edge_count",
    "hashes",
    "node_count",
    "provenance_graph_sha256",
    "raw_count_sum",
    "retained_by_superclass",
    "retention_policy",
    "schema_version",
    "source_manifest_sha256",
    "sources",
    "structural_graph_sha256",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class FullArtifactError(RuntimeError):
    """Base error for a refused full runtime artifact."""


class FullArtifactIdentityError(FullArtifactError):
    """Artifact content does not match an external trusted identity."""


class FullArtifactSchemaError(FullArtifactError):
    """Artifact manifest or array layout is structurally invalid."""


@dataclass(frozen=True, slots=True)
class RuntimeArraySpec:
    name: str
    filename: str
    dtype: str
    count: int
    byte_length: int
    sha256: str


@dataclass(frozen=True, slots=True)
class BoundaryRoleNode:
    body_id: int
    dense_index: int
    annotation_source_row: int
    superclass: str


@dataclass(frozen=True, slots=True)
class BoundaryRoleOverlay:
    fixture_id: str
    fixture_file_sha256: str
    fixture_semantic_sha256: str
    overlay_sha256: str
    roles: dict[str, tuple[BoundaryRoleNode, ...]]


@dataclass(frozen=True, slots=True)
class FullRuntimeManifest:
    schema_version: int
    dataset_id: str
    artifact_id: str
    source_manifest_sha256: str
    payload_sha256: str
    structural_graph_sha256: str
    provenance_graph_sha256: str
    builder_implementation_sha256: str
    builder_configuration_sha256: str
    node_count: int
    edge_count: int
    raw_count_sum: int
    manifest_file_sha256: str
    arrays: tuple[RuntimeArraySpec, ...]
    source_artifacts: tuple[SourceArtifact, ...]
    boundary_role_overlay: BoundaryRoleOverlay
    counts: dict[str, int]
    retained_by_superclass: dict[str, int]
    retention_policy: dict[str, Any]


def _open_no_follow(path: Path) -> BinaryIO:
    """Open one read-only descriptor, refusing a final-component symlink.

    The caller owns the returned handle for as long as it needs the bytes; every
    check and every read is issued against this one descriptor.
    """

    descriptor = os.open(path, os.O_RDONLY | _NO_FOLLOW_FLAG)
    try:
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _read_bounded_single_link_file(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    """Return whole-file bytes read from the descriptor that was size-gated.

    One descriptor answers every question: ``fstat`` proves the file is regular
    and singly linked and that its length is within ``maximum_bytes``, and the
    read that follows uses that same descriptor. No caller ever hashes one path
    and then reopens it.
    """

    with _open_no_follow(path) as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise FullArtifactIdentityError(f"{label} must be a single-link regular file")
        if info.st_size > maximum_bytes:
            raise FullArtifactIdentityError(
                f"{label} exceeds the {maximum_bytes}-byte limit: {info.st_size} bytes"
            )
        payload = handle.read(info.st_size + 1)
        if len(payload) != info.st_size:
            raise FullArtifactIdentityError(
                f"{label} length changed during the gated read: read {len(payload)} bytes"
            )
    return payload


def _sha256_mapping(mapping: mmap.mmap, byte_length: int) -> str:
    """Hash the mapped bytes themselves, in blocks, without copying the file."""

    digest = hashlib.sha256()
    with memoryview(mapping) as view:
        for start in range(0, byte_length, _HASH_BLOCK_BYTES):
            with view[start : start + _HASH_BLOCK_BYTES] as block:
                digest.update(block)
    return digest.hexdigest()


class MappedUnsignedArray(Sequence[int]):
    """Read-only little-endian unsigned integer array backed by a verified mmap.

    Verification is bound to the mapped bytes. A single descriptor is opened
    without following a symlink, ``fstat`` gates regular-file type, a link count
    of one and the exact declared byte length, the descriptor is mapped, and the
    SHA-256 comparison then runs over the mapping itself. The bytes that were
    verified are therefore the same bytes that every later read observes; there
    is no verify-then-reopen window for a path swap to exploit.

    Accepted residual: the mapping is ``ACCESS_READ`` and therefore shared, so a
    writer that already holds the same inode can still mutate the file after the
    digest is computed. The descriptor-bound checks close the rename, symlink,
    hard-link and replace vectors; concurrent same-inode writers remain out of
    scope and are disclosed rather than defended against.
    """

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        count: int,
        byte_length: int,
        expected_sha256: str,
        label: str,
    ) -> None:
        if width not in {4, 8}:
            raise ValueError("mapped unsigned array width must be 4 or 8")
        self._path = path
        self._width = width
        self._count = count
        self._struct = struct.Struct("<I" if width == 4 else "<Q")
        self._handle = _open_no_follow(path)
        try:
            info = os.fstat(self._handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise FullArtifactIdentityError(f"{label} must be a single-link regular file")
            if info.st_size != byte_length:
                raise FullArtifactIdentityError(f"{label} file size mismatch")
            self._mmap = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)
        except BaseException:
            self._handle.close()
            raise
        try:
            if _sha256_mapping(self._mmap, byte_length) != expected_sha256:
                raise FullArtifactIdentityError(f"{label} SHA-256 mismatch")
        except BaseException:
            self._mmap.close()
            self._handle.close()
            raise

    def __len__(self) -> int:
        return self._count

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[int, ...]: ...

    def __getitem__(self, index: int | slice) -> int | tuple[int, ...]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._count)
            return tuple(self[position] for position in range(start, stop, step))
        normalized = index + self._count if index < 0 else index
        if normalized < 0 or normalized >= self._count:
            raise IndexError(index)
        return int(self._struct.unpack_from(self._mmap, normalized * self._width)[0])

    def __iter__(self) -> Iterator[int]:
        for (value,) in struct.iter_unpack(self._struct.format, self._mmap):
            yield value

    def close(self) -> None:
        self._mmap.close()
        self._handle.close()


@dataclass(slots=True)
class FullRuntimeGraph:
    """Compact full-graph arrays; call ``close`` when no longer needed."""

    manifest: FullRuntimeManifest
    annotation_source_rows: MappedUnsignedArray
    body_ids: MappedUnsignedArray
    edge_post_indices: MappedUnsignedArray
    edge_pre_indices: MappedUnsignedArray
    edge_raw_counts: MappedUnsignedArray
    edge_source_rows: MappedUnsignedArray
    incoming_offsets: MappedUnsignedArray
    transmitter_source_rows: MappedUnsignedArray

    def close(self) -> None:
        for name in ARRAY_LAYOUT:
            getattr(self, name).close()

    def __enter__(self) -> FullRuntimeGraph:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def index_of(self, body_id: int) -> int:
        lo = 0
        hi = len(self.body_ids)
        while lo < hi:
            middle = (lo + hi) // 2
            candidate = self.body_ids[middle]
            if candidate < body_id:
                lo = middle + 1
            else:
                hi = middle
        if lo == len(self.body_ids) or self.body_ids[lo] != body_id:
            raise KeyError(f"body ID {body_id} is not present in full runtime artifact")
        return lo

    def role_body_ids(self, role: str) -> tuple[int, ...]:
        """Return exact G0-fixture-derived body IDs from the named overlay."""

        try:
            return tuple(node.body_id for node in self.manifest.boundary_role_overlay.roles[role])
        except KeyError as exc:
            raise KeyError(f"unknown full-runtime boundary role {role!r}") from exc

    def transmitter_source_row(self, body_id: int) -> int | None:
        """Return exact transmitter lineage, or ``None`` for a source-row miss."""

        value = self.transmitter_source_rows[self.index_of(body_id)]
        return None if value == (1 << 64) - 1 else value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FullArtifactSchemaError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise FullArtifactSchemaError(f"non-finite JSON number is forbidden: {value}")


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FullArtifactSchemaError(f"{label} must be an object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise FullArtifactSchemaError(f"{label} must be an array")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise FullArtifactSchemaError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    digest = _require_string(value, label)
    if _SHA256_RE.fullmatch(digest) is None:
        raise FullArtifactSchemaError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FullArtifactSchemaError(f"{label} must be an integer >= {minimum}")
    return value


def _source_from_document(value: Any) -> SourceArtifact:
    source = _require_dict(value, "sources[]")
    if set(source) != {"bytes", "filename", "generation", "id", "sha256", "url"}:
        raise FullArtifactSchemaError("source object fields are not canonical")
    return SourceArtifact(
        source_id=_require_string(source.get("id"), "source.id"),
        filename=_require_string(source.get("filename"), "source.filename"),
        byte_length=_require_int(source.get("bytes"), "source.bytes", minimum=1),
        sha256=_require_sha256(source.get("sha256"), "source.sha256"),
        generation=_require_string(source.get("generation"), "source.generation"),
        url=_require_string(source.get("url"), "source.url"),
    )


def load_full_runtime_artifact(
    directory: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_source_manifest_sha256: str = MALECNS_SOURCE_MANIFEST_SHA256,
    trusted_sources: tuple[SourceArtifact, ...] = MALECNS_FIXTURE_V1_PIN.source_artifacts,
    verify_layout: bool = False,
) -> FullRuntimeGraph:
    """Verify every byte before mapping a full runtime graph.

    The manifest hash is deliberately mandatory and external. A colocated
    manifest can therefore never authorize its own mutation.
    """

    root = Path(directory)
    manifest_path = root / "manifest.json"
    try:
        manifest_bytes = _read_bounded_single_link_file(
            manifest_path,
            maximum_bytes=MAX_MANIFEST_BYTES,
            label="full artifact manifest",
        )
    except OSError as exc:
        raise FullArtifactIdentityError(f"cannot read full artifact manifest: {exc}") from exc
    observed_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if observed_manifest_sha256 != expected_manifest_sha256:
        raise FullArtifactIdentityError(
            "full artifact manifest SHA-256 mismatch: "
            f"expected {expected_manifest_sha256}, observed {observed_manifest_sha256}"
        )
    try:
        document = json.loads(
            manifest_bytes,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FullArtifactSchemaError(f"manifest is not strict UTF-8 JSON: {exc}") from exc
    document = _require_dict(document, "manifest")
    if set(document) != _MANIFEST_KEYS:
        raise FullArtifactSchemaError("full artifact manifest fields are not canonical")
    hashes = _require_dict(document.get("hashes"), "hashes")
    if set(hashes) != {"payload_scope", "payload_sha256"}:
        raise FullArtifactSchemaError("full artifact hash fields are not canonical")
    if hashes.get("payload_scope") != "canonical JSON document with hashes object omitted":
        raise FullArtifactSchemaError("full artifact payload hash scope is not canonical")
    payload = {key: value for key, value in document.items() if key != "hashes"}
    observed_payload_sha256 = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    declared_payload_sha256 = _require_sha256(hashes.get("payload_sha256"), "hashes.payload_sha256")
    if observed_payload_sha256 != declared_payload_sha256:
        raise FullArtifactIdentityError("full artifact payload SHA-256 mismatch")
    if document.get("schema_version") != FULL_ARTIFACT_SCHEMA_VERSION:
        raise FullArtifactSchemaError("unsupported full artifact schema version")
    dataset_id = _require_string(document.get("dataset_id"), "dataset_id")
    artifact_id = _require_string(document.get("artifact_id"), "artifact_id")
    if dataset_id != "male-cns:v1.0" or artifact_id != FULL_ARTIFACT_ID:
        raise FullArtifactIdentityError("full artifact dataset/artifact identity mismatch")
    dataset = _require_dict(document.get("dataset"), "dataset")
    if dataset.get("id") != dataset_id:
        raise FullArtifactIdentityError("full artifact dataset metadata mismatch")
    if document.get("array_encoding") != ("packed little-endian unsigned integers; no header"):
        raise FullArtifactSchemaError("full artifact array encoding is not canonical")
    source_manifest_sha256 = _require_sha256(
        document.get("source_manifest_sha256"), "source_manifest_sha256"
    )
    if source_manifest_sha256 != expected_source_manifest_sha256:
        raise FullArtifactIdentityError("full artifact source manifest identity mismatch")
    sources = tuple(
        _source_from_document(item) for item in _require_list(document.get("sources"), "sources")
    )
    if sources != trusted_sources:
        raise FullArtifactIdentityError("full artifact source objects do not match trusted pins")

    arrays_document = _require_dict(document.get("arrays"), "arrays")
    if set(arrays_document) != set(ARRAY_LAYOUT):
        raise FullArtifactSchemaError("full artifact array set is not canonical")
    specs: list[RuntimeArraySpec] = []
    mapped: dict[str, MappedUnsignedArray] = {}
    try:
        for name, (expected_dtype, width) in ARRAY_LAYOUT.items():
            raw_spec = _require_dict(arrays_document.get(name), f"arrays.{name}")
            if set(raw_spec) != {"bytes", "count", "dtype", "filename", "sha256"}:
                raise FullArtifactSchemaError(f"array {name} fields are not canonical")
            spec = RuntimeArraySpec(
                name=name,
                filename=_require_string(raw_spec.get("filename"), f"arrays.{name}.filename"),
                dtype=_require_string(raw_spec.get("dtype"), f"arrays.{name}.dtype"),
                count=_require_int(raw_spec.get("count"), f"arrays.{name}.count"),
                byte_length=_require_int(raw_spec.get("bytes"), f"arrays.{name}.bytes"),
                sha256=_require_sha256(raw_spec.get("sha256"), f"arrays.{name}.sha256"),
            )
            if spec.dtype != expected_dtype:
                raise FullArtifactSchemaError(f"array {name} dtype mismatch")
            if spec.filename != f"{name}.bin":
                raise FullArtifactSchemaError(f"array {name} filename is not canonical")
            if spec.byte_length != spec.count * width:
                raise FullArtifactSchemaError(f"array {name} byte length does not reconcile")
            specs.append(spec)
            mapped[name] = MappedUnsignedArray(
                root / spec.filename,
                width=width,
                count=spec.count,
                byte_length=spec.byte_length,
                expected_sha256=spec.sha256,
                label=f"array {name}",
            )

        counts = _require_dict(document.get("counts"), "counts")
        if set(counts) != COUNT_KEYS:
            raise FullArtifactSchemaError("full artifact count fields are not canonical")
        integer_counts = {
            key: _require_int(value, f"counts.{key}") for key, value in counts.items()
        }
        node_count = _require_int(document.get("node_count"), "node_count", minimum=1)
        edge_count = _require_int(document.get("edge_count"), "edge_count", minimum=1)
        raw_count_sum = _require_int(document.get("raw_count_sum"), "raw_count_sum", minimum=1)
        expected_lengths = {
            "annotation_source_rows": node_count,
            "body_ids": node_count,
            "edge_post_indices": edge_count,
            "edge_pre_indices": edge_count,
            "edge_raw_counts": edge_count,
            "edge_source_rows": edge_count,
            "incoming_offsets": node_count + 1,
            "transmitter_source_rows": node_count,
        }
        for name, expected_count in expected_lengths.items():
            if len(mapped[name]) != expected_count:
                raise FullArtifactSchemaError(f"array {name} count does not reconcile")
        if integer_counts.get("retained_node_rows") != node_count:
            raise FullArtifactSchemaError("retained node count does not reconcile")
        if integer_counts.get("retained_edge_rows") != edge_count:
            raise FullArtifactSchemaError("retained edge count does not reconcile")
        if integer_counts.get("retained_weight_sum") != raw_count_sum:
            raise FullArtifactSchemaError("retained weight sum does not reconcile")

        if (
            integer_counts["retained_node_rows"]
            + integer_counts["excluded_superclass_null_rows"]
            + integer_counts["excluded_superclass_empty_rows"]
            != integer_counts["annotation_source_rows"]
        ):
            raise FullArtifactSchemaError("annotation attrition counts do not reconcile")
        if (
            integer_counts["retained_edge_rows"] + integer_counts["excluded_edge_rows"]
            != integer_counts["raw_edge_rows"]
        ):
            raise FullArtifactSchemaError("edge attrition counts do not reconcile")
        if (
            integer_counts["retained_weight_sum"] + integer_counts["excluded_weight_sum"]
            != integer_counts["raw_weight_sum"]
        ):
            raise FullArtifactSchemaError("edge attrition weights do not reconcile")
        if (
            integer_counts["excluded_pre_only_rows"]
            + integer_counts["excluded_post_only_rows"]
            + integer_counts["excluded_neither_endpoint_rows"]
            != integer_counts["excluded_edge_rows"]
        ):
            raise FullArtifactSchemaError("endpoint exclusion counts do not reconcile")
        if (
            integer_counts["retained_self_edge_rows"] + integer_counts["excluded_self_edge_rows"]
            != integer_counts["raw_self_edge_rows"]
        ):
            raise FullArtifactSchemaError("self-edge counts do not reconcile")
        if (
            integer_counts["retained_self_edge_weight_sum"]
            + integer_counts["excluded_self_edge_weight_sum"]
            != integer_counts["raw_self_edge_weight_sum"]
        ):
            raise FullArtifactSchemaError("self-edge weights do not reconcile")
        if (
            integer_counts["raw_duplicate_directed_pairs"] != 0
            or integer_counts["retained_duplicate_directed_pairs"] != 0
        ):
            raise FullArtifactIdentityError("duplicate directed pairs are forbidden")
        if integer_counts["annotation_duplicate_body_ids"] != 0:
            raise FullArtifactIdentityError("duplicate annotation body IDs are forbidden")
        if integer_counts["transmitter_duplicate_body_ids"] != 0:
            raise FullArtifactIdentityError("duplicate transmitter body IDs are forbidden")
        if (
            integer_counts["retained_nodes_with_transmitter_row"]
            + integer_counts["retained_nodes_missing_transmitter_row"]
            != node_count
        ):
            raise FullArtifactSchemaError("retained transmitter lineage does not reconcile")
        if (
            integer_counts["retained_nodes_with_transmitter_row"]
            + integer_counts["excluded_transmitter_rows"]
            != integer_counts["transmitter_source_rows"]
        ):
            raise FullArtifactSchemaError("transmitter source attrition does not reconcile")

        retained_by_superclass_document = _require_dict(
            document.get("retained_by_superclass"), "retained_by_superclass"
        )
        retained_by_superclass = {
            _require_string(key, "retained_by_superclass key"): _require_int(
                value, f"retained_by_superclass.{key}", minimum=1
            )
            for key, value in retained_by_superclass_document.items()
        }
        if sum(retained_by_superclass.values()) != node_count:
            raise FullArtifactSchemaError("superclass counts do not reconcile")
        retention_policy = _require_dict(document.get("retention_policy"), "retention_policy")
        if retention_policy != FULL_RETENTION_POLICY:
            raise FullArtifactIdentityError("full artifact retention policy mismatch")

        overlay_document = _require_dict(
            document.get("boundary_role_overlay"), "boundary_role_overlay"
        )
        if set(overlay_document) != {"authority", "roles", "sha256"}:
            raise FullArtifactSchemaError("boundary role overlay fields are not canonical")
        authority = _require_dict(
            overlay_document.get("authority"), "boundary_role_overlay.authority"
        )
        if set(authority) != {
            "fixture_file_sha256",
            "fixture_id",
            "fixture_semantic_sha256",
        }:
            raise FullArtifactSchemaError("boundary role authority fields are not canonical")
        fixture_id = _require_string(authority.get("fixture_id"), "authority.fixture_id")
        fixture_file_sha256 = _require_sha256(
            authority.get("fixture_file_sha256"), "authority.fixture_file_sha256"
        )
        fixture_semantic_sha256 = _require_sha256(
            authority.get("fixture_semantic_sha256"),
            "authority.fixture_semantic_sha256",
        )
        if (
            fixture_id != MALECNS_FIXTURE_V1_PIN.fixture_id
            or fixture_file_sha256 != MALECNS_FIXTURE_V1_PIN.file_sha256
            or fixture_semantic_sha256 != MALECNS_FIXTURE_V1_PIN.semantic_sha256
        ):
            raise FullArtifactIdentityError("boundary role authority mismatch")
        roles_document = _require_dict(overlay_document.get("roles"), "boundary_role_overlay.roles")
        expected_roles = dict(BOUNDARY_ROLE_BODY_IDS)
        expected_superclasses = dict(BOUNDARY_BODY_SUPERCLASSES)
        if set(roles_document) != set(expected_roles):
            raise FullArtifactSchemaError("boundary role names are not canonical")
        parsed_roles: dict[str, tuple[BoundaryRoleNode, ...]] = {}
        seen_boundary_bodies: set[int] = set()
        for role, expected_body_ids in expected_roles.items():
            entries = _require_list(roles_document.get(role), f"roles.{role}")
            parsed_entries: list[BoundaryRoleNode] = []
            for raw_entry in entries:
                entry = _require_dict(raw_entry, f"roles.{role}[]")
                if set(entry) != {
                    "annotation_source_row",
                    "body_id",
                    "dense_index",
                    "superclass",
                }:
                    raise FullArtifactSchemaError(
                        f"boundary role {role} entry fields are not canonical"
                    )
                parsed = BoundaryRoleNode(
                    body_id=_require_int(entry.get("body_id"), f"roles.{role}.body_id", minimum=1),
                    dense_index=_require_int(entry.get("dense_index"), f"roles.{role}.dense_index"),
                    annotation_source_row=_require_int(
                        entry.get("annotation_source_row"),
                        f"roles.{role}.annotation_source_row",
                    ),
                    superclass=_require_string(entry.get("superclass"), f"roles.{role}.superclass"),
                )
                if parsed.dense_index >= node_count:
                    raise FullArtifactSchemaError("boundary dense index is out of range")
                if mapped["body_ids"][parsed.dense_index] != parsed.body_id:
                    raise FullArtifactIdentityError("boundary body-ID/dense-index mismatch")
                if (
                    mapped["annotation_source_rows"][parsed.dense_index]
                    != parsed.annotation_source_row
                ):
                    raise FullArtifactIdentityError("boundary annotation lineage mismatch")
                if expected_superclasses.get(parsed.body_id) != parsed.superclass:
                    raise FullArtifactIdentityError("boundary superclass mismatch")
                if parsed.body_id in seen_boundary_bodies:
                    raise FullArtifactSchemaError("boundary body belongs to multiple roles")
                seen_boundary_bodies.add(parsed.body_id)
                parsed_entries.append(parsed)
            if tuple(entry.body_id for entry in parsed_entries) != expected_body_ids:
                raise FullArtifactIdentityError(f"boundary role {role} body IDs mismatch")
            parsed_roles[role] = tuple(parsed_entries)
        overlay_sha256 = _require_sha256(
            overlay_document.get("sha256"), "boundary_role_overlay.sha256"
        )
        observed_overlay_sha256 = hashlib.sha256(
            canonical_json_bytes({"authority": authority, "roles": roles_document})
        ).hexdigest()
        if overlay_sha256 != observed_overlay_sha256:
            raise FullArtifactIdentityError("boundary role overlay SHA-256 mismatch")
        boundary_role_overlay = BoundaryRoleOverlay(
            fixture_id=fixture_id,
            fixture_file_sha256=fixture_file_sha256,
            fixture_semantic_sha256=fixture_semantic_sha256,
            overlay_sha256=overlay_sha256,
            roles=parsed_roles,
        )

        spec_by_name = {spec.name: spec for spec in specs}
        structural_graph_sha256 = _require_sha256(
            document.get("structural_graph_sha256"), "structural_graph_sha256"
        )
        observed_structural_graph_sha256 = hashlib.sha256(
            canonical_json_bytes(
                {
                    "artifact_id": FULL_ARTIFACT_ID,
                    "body_ids": spec_by_name["body_ids"].sha256,
                    "edge_post_indices": spec_by_name["edge_post_indices"].sha256,
                    "edge_pre_indices": spec_by_name["edge_pre_indices"].sha256,
                    "edge_raw_counts": spec_by_name["edge_raw_counts"].sha256,
                    "retention_policy": retention_policy,
                }
            )
        ).hexdigest()
        if observed_structural_graph_sha256 != structural_graph_sha256:
            raise FullArtifactIdentityError("structural graph SHA-256 mismatch")
        provenance_graph_sha256 = _require_sha256(
            document.get("provenance_graph_sha256"), "provenance_graph_sha256"
        )
        observed_provenance_graph_sha256 = hashlib.sha256(
            canonical_json_bytes(
                {
                    "annotation_source_rows": spec_by_name["annotation_source_rows"].sha256,
                    "boundary_role_overlay": overlay_sha256,
                    "edge_source_rows": spec_by_name["edge_source_rows"].sha256,
                    "source_manifest_sha256": source_manifest_sha256,
                    "structural_graph_sha256": structural_graph_sha256,
                    "transmitter_source_rows": spec_by_name["transmitter_source_rows"].sha256,
                }
            )
        ).hexdigest()
        if observed_provenance_graph_sha256 != provenance_graph_sha256:
            raise FullArtifactIdentityError("provenance graph SHA-256 mismatch")
        builder_configuration_sha256 = _require_sha256(
            document.get("builder_configuration_sha256"),
            "builder_configuration_sha256",
        )
        expected_builder_configuration = {
            "boundary_role_body_ids": {
                role: list(body_ids) for role, body_ids in BOUNDARY_ROLE_BODY_IDS
            },
            "retention_policy": retention_policy,
        }
        if (
            builder_configuration_sha256
            != hashlib.sha256(canonical_json_bytes(expected_builder_configuration)).hexdigest()
        ):
            raise FullArtifactIdentityError("builder configuration SHA-256 mismatch")

        manifest = FullRuntimeManifest(
            schema_version=FULL_ARTIFACT_SCHEMA_VERSION,
            dataset_id=dataset_id,
            artifact_id=artifact_id,
            source_manifest_sha256=source_manifest_sha256,
            payload_sha256=declared_payload_sha256,
            structural_graph_sha256=structural_graph_sha256,
            provenance_graph_sha256=provenance_graph_sha256,
            builder_implementation_sha256=_require_sha256(
                document.get("builder_implementation_sha256"),
                "builder_implementation_sha256",
            ),
            builder_configuration_sha256=builder_configuration_sha256,
            node_count=node_count,
            edge_count=edge_count,
            raw_count_sum=raw_count_sum,
            manifest_file_sha256=observed_manifest_sha256,
            arrays=tuple(specs),
            source_artifacts=sources,
            boundary_role_overlay=boundary_role_overlay,
            counts=integer_counts,
            retained_by_superclass=retained_by_superclass,
            retention_policy=retention_policy,
        )
        graph = FullRuntimeGraph(manifest=manifest, **mapped)
        if verify_layout:
            _verify_mapped_layout(graph)
        return graph
    except BaseException:
        for array in mapped.values():
            array.close()
        raise


def _verify_mapped_layout(graph: FullRuntimeGraph) -> None:
    """Exhaustive dependency-free layout check, intended for small/test artifacts."""

    if graph.body_ids[0] <= 0:
        raise FullArtifactSchemaError("body IDs must be positive")
    previous_body = graph.body_ids[0]
    for index in range(1, len(graph.body_ids)):
        body_id = graph.body_ids[index]
        if body_id <= previous_body:
            raise FullArtifactSchemaError("body IDs are not strictly ascending")
        previous_body = body_id
    transmitter_rows: set[int] = set()
    transmitter_row_count = graph.manifest.counts["transmitter_source_rows"]
    missing_sentinel = (1 << 64) - 1
    for source_row in graph.transmitter_source_rows:
        if source_row == missing_sentinel:
            continue
        if source_row >= transmitter_row_count or source_row in transmitter_rows:
            raise FullArtifactSchemaError("transmitter source lineage is invalid")
        transmitter_rows.add(source_row)
    if len(transmitter_rows) != graph.manifest.counts["retained_nodes_with_transmitter_row"]:
        raise FullArtifactSchemaError("transmitter source lineage count mismatch")
    if graph.incoming_offsets[0] != 0 or graph.incoming_offsets[-1] != len(graph.edge_pre_indices):
        raise FullArtifactSchemaError("incoming offsets have invalid endpoints")
    previous_offset = 0
    for offset in graph.incoming_offsets:
        if offset < previous_offset:
            raise FullArtifactSchemaError("incoming offsets are not monotonic")
        previous_offset = offset
    previous_pair: tuple[int, int] | None = None
    weight_sum = 0
    for index in range(len(graph.edge_pre_indices)):
        pre = graph.edge_pre_indices[index]
        post = graph.edge_post_indices[index]
        if pre >= len(graph.body_ids) or post >= len(graph.body_ids):
            raise FullArtifactSchemaError("edge dense index is out of range")
        pair = (post, pre)
        if previous_pair is not None and pair <= previous_pair:
            raise FullArtifactSchemaError("edges are not unique target-major pairs")
        previous_pair = pair
        weight = graph.edge_raw_counts[index]
        if weight <= 0:
            raise FullArtifactSchemaError("edge raw count is not positive")
        weight_sum += weight
    if weight_sum != graph.manifest.raw_count_sum:
        raise FullArtifactSchemaError("edge raw count sum does not match manifest")


def manifest_sha256(directory: str | Path) -> str:
    """Return the external startup pin for an already written artifact."""

    return hashlib.sha256(
        _read_bounded_single_link_file(
            Path(directory) / "manifest.json",
            maximum_bytes=MAX_MANIFEST_BYTES,
            label="full artifact manifest",
        )
    ).hexdigest()
