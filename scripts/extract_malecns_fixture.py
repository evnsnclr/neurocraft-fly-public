#!/usr/bin/env python3
"""Deterministically extract the reserved MaleCNS v1.0 flight-power fixture.

The extractor is deliberately strict.  It consumes only the three immutable
Feather objects named by ``male-cns-v1.0.sources.json``, verifies their exact
byte lengths and SHA-256 digests before parsing, rejects duplicate directed
edge rows, and preserves zero-based source-row lineage in the result.

This module does not assign synaptic signs.  MaleCNS neurotransmitter records
are copied as evidence, while every emitted edge keeps an explicit unknown
sign status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.feather as feather
except ImportError as exc:  # pragma: no cover - exercised by the CLI environment
    raise SystemExit(
        "extract_malecns_fixture.py requires pyarrow (and its NumPy dependency)"
    ) from exc


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    REPOSITORY_ROOT / "data" / "manifests" / "male-cns-v1.0.sources.json"
)
DEFAULT_CACHE = REPOSITORY_ROOT / "data" / "cache" / "male-cns-v1.0"
DATASET_ID = "male-cns:v1.0"
FIXTURE_ID = "mcns-v1.0-flight-power-v1"
EXPECTED_MANIFEST_SHA256 = (
    "9412105f54884a57521bbdf7fe46659caab559ef7877e6a62af95f50e1ee3768"
)
REQUIRED_SOURCE_IDS = ("annotations", "neurotransmitters", "weights")
PATH_MIN_WEIGHT = 10
INDUCED_MIN_WEIGHT = 5
MAX_SEGMENT_HOPS = 3
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
INSTANCE_SIDE_RE = re.compile(r"_([LR])\Z")
UINT32_MAX = np.iinfo(np.uint32).max

REQUIRED_ANNOTATION_COLUMNS = (
    "bodyId",
    "type",
    "instance",
    "rootSide",
    "somaSide",
    "superclass",
    "subclass",
    "entryNerve",
)
TRANSMITTER_EVIDENCE_FIELDS = (
    "ground_truth",
    "consensus_nt",
    "predicted_nt",
    "celltype_predicted_nt",
)


class FixtureExtractionError(RuntimeError):
    """Stable, user-facing extraction failure."""


class ManifestError(FixtureExtractionError):
    """The source manifest does not satisfy the pinned contract."""


class SourceVerificationError(FixtureExtractionError):
    """A source object does not match its manifest pin."""


class SourceDataError(FixtureExtractionError):
    """A verified Feather object violates the extraction contract."""


class CircuitResolutionError(FixtureExtractionError):
    """The exact bilateral circuit cannot be resolved."""


@dataclass(frozen=True, slots=True)
class SourceSpec:
    source_id: str
    filename: str
    expected_bytes: int
    sha256: str
    url: str
    generation: str | None
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Manifest:
    raw_bytes: bytes
    raw_sha256: str
    required_sha256: str
    schema_version: int
    dataset: Mapping[str, Any]
    sources: Mapping[str, SourceSpec]


@dataclass(frozen=True, slots=True)
class AnnotationRecord:
    source_row: int
    body_id: int
    record: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Edge:
    body_pre: int
    body_post: int
    weight: int
    source_row: int


@dataclass(frozen=True, slots=True)
class DirectedPath:
    nodes: tuple[int, ...]
    edges: tuple[Edge, ...]
    bottleneck: int


@dataclass(frozen=True, slots=True)
class SideSelection:
    side: str
    named_body_id: int
    named_type: str
    upstream: DirectedPath
    downstream: DirectedPath
    complete_nodes: tuple[int, ...]
    bottleneck: int
    upstream_candidate_count: int
    downstream_candidate_count: int


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON key in manifest: {key!r}")
        result[key] = value
    return result


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be a JSON object")
    return value


def _require_string(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{label}.{key} must be a non-empty string")
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(
    path: Path,
    *,
    expected_sha256: str = EXPECTED_MANIFEST_SHA256,
) -> Manifest:
    """Load the exact acquisition manifest schema used by this repository."""

    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise ManifestError(f"cannot read source manifest {path}: {exc}") from exc

    observed_manifest_sha256 = _sha256_bytes(raw_bytes)
    if SHA256_RE.fullmatch(expected_sha256) is None:
        raise ManifestError("expected manifest SHA-256 is not a valid lowercase digest")
    if observed_manifest_sha256 != expected_sha256:
        raise ManifestError(
            f"source manifest SHA-256 mismatch: expected {expected_sha256}, "
            f"observed {observed_manifest_sha256}"
        )

    try:
        document = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"invalid UTF-8 JSON source manifest: {exc}") from exc

    root = _require_mapping(document, "manifest")
    if root.get("schema_version") != 1:
        raise ManifestError("manifest.schema_version must equal 1")
    dataset = _require_mapping(root.get("dataset"), "manifest.dataset")
    if dataset.get("id") != DATASET_ID:
        raise ManifestError(
            f"manifest.dataset.id must equal {DATASET_ID!r}, got {dataset.get('id')!r}"
        )

    raw_sources = root.get("sources")
    if not isinstance(raw_sources, list):
        raise ManifestError("manifest.sources must be a JSON array")

    sources: dict[str, SourceSpec] = {}
    for index, raw_source in enumerate(raw_sources):
        label = f"manifest.sources[{index}]"
        source = _require_mapping(raw_source, label)
        source_id = _require_string(source, "id", label)
        if source_id in sources:
            raise ManifestError(f"duplicate source ID in manifest: {source_id!r}")

        filename = _require_string(source, "filename", label)
        if Path(filename).name != filename or filename in {".", ".."}:
            raise ManifestError(f"{label}.filename must be a safe basename")
        expected_bytes = source.get("bytes")
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes <= 0
        ):
            raise ManifestError(f"{label}.bytes must be a positive integer")
        sha256 = _require_string(source, "sha256", label)
        if SHA256_RE.fullmatch(sha256) is None:
            raise ManifestError(f"{label}.sha256 must be 64 lowercase hexadecimal digits")
        url = _require_string(source, "url", label)
        generation = source.get("generation")
        if generation is not None and (
            not isinstance(generation, str) or not generation
        ):
            raise ManifestError(f"{label}.generation must be null or a non-empty string")
        sources[source_id] = SourceSpec(
            source_id=source_id,
            filename=filename,
            expected_bytes=expected_bytes,
            sha256=sha256,
            url=url,
            generation=generation,
            raw=source,
        )

    if set(sources) != set(REQUIRED_SOURCE_IDS):
        raise ManifestError(
            "manifest.sources must contain exactly the IDs "
            f"{list(REQUIRED_SOURCE_IDS)!r}; got {sorted(sources)!r}"
        )

    return Manifest(
        raw_bytes=raw_bytes,
        raw_sha256=observed_manifest_sha256,
        required_sha256=expected_sha256,
        schema_version=1,
        dataset=dataset,
        sources=sources,
    )


def verify_sources(manifest: Manifest, cache_dir: Path) -> dict[str, Path]:
    """Verify all three source bytes before allowing any Feather parsing."""

    paths: dict[str, Path] = {}
    for source_id in REQUIRED_SOURCE_IDS:
        spec = manifest.sources[source_id]
        path = cache_dir / spec.filename
        try:
            stat = path.stat()
        except OSError as exc:
            raise SourceVerificationError(
                f"cannot stat required source {source_id!r} at {path}: {exc}"
            ) from exc
        if not path.is_file():
            raise SourceVerificationError(
                f"required source {source_id!r} is not a regular file: {path}"
            )
        if stat.st_size != spec.expected_bytes:
            raise SourceVerificationError(
                f"source {source_id!r} byte mismatch: expected {spec.expected_bytes}, "
                f"observed {stat.st_size}"
            )
        observed_sha256 = _sha256_file(path)
        if observed_sha256 != spec.sha256:
            raise SourceVerificationError(
                f"source {source_id!r} SHA-256 mismatch: expected {spec.sha256}, "
                f"observed {observed_sha256}"
            )
        paths[source_id] = path
    return paths


def _read_feather(path: Path, label: str) -> pa.Table:
    try:
        return feather.read_table(path, memory_map=True)
    except Exception as exc:  # pyarrow exposes several format/IO exception types
        raise SourceDataError(f"cannot read verified {label} Feather table: {exc}") from exc


def _require_columns(table: pa.Table, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(table.column_names))
    if missing:
        raise SourceDataError(f"{label} table is missing required columns: {missing!r}")


def _require_int64(table: pa.Table, column: str, label: str) -> None:
    if not pa.types.is_int64(table.schema.field(column).type):
        raise SourceDataError(f"{label}.{column} must have Arrow int64 type")


def _normalize_json(value: Any) -> Any:
    """Convert Arrow/Python values to an explicit, canonical JSON domain."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"nonfinite_float": "NaN"}
        if math.isinf(value):
            return {"nonfinite_float": "Infinity" if value > 0 else "-Infinity"}
        return value
    if isinstance(value, bytes):
        return {"hex_bytes": value.hex()}
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if hasattr(value, "item"):
        return _normalize_json(value.item())
    return str(value)


def load_annotations(path: Path) -> tuple[dict[int, AnnotationRecord], int]:
    table = _read_feather(path, "annotations")
    _require_columns(table, REQUIRED_ANNOTATION_COLUMNS, "annotations")
    _require_int64(table, "bodyId", "annotations")
    if table["bodyId"].null_count:
        raise SourceDataError("annotations.bodyId contains null values")

    records: dict[int, AnnotationRecord] = {}
    for source_row, raw in enumerate(table.to_pylist()):
        body_id = raw.get("bodyId")
        if isinstance(body_id, bool) or not isinstance(body_id, int) or body_id <= 0:
            raise SourceDataError(
                f"annotations row {source_row} has invalid bodyId {body_id!r}"
            )
        if body_id in records:
            previous = records[body_id].source_row
            raise SourceDataError(
                f"duplicate annotations.bodyId {body_id} at zero-based source rows "
                f"{previous} and {source_row}"
            )
        records[body_id] = AnnotationRecord(
            source_row=source_row,
            body_id=body_id,
            record=_normalize_json(raw),
        )
    return records, table.num_rows


def load_transmitters(path: Path) -> pa.Table:
    table = _read_feather(path, "neurotransmitters")
    _require_columns(table, ("body",), "neurotransmitters")
    _require_int64(table, "body", "neurotransmitters")
    if table["body"].null_count:
        raise SourceDataError("neurotransmitters.body contains null values")

    bodies = table["body"].combine_chunks().to_numpy(zero_copy_only=False)
    if np.any(bodies <= 0):
        source_row = int(np.flatnonzero(bodies <= 0)[0])
        raise SourceDataError(
            f"neurotransmitters row {source_row} has invalid body "
            f"{int(bodies[source_row])!r}"
        )
    sorted_bodies = np.sort(bodies, kind="quicksort")
    duplicate_offsets = np.flatnonzero(sorted_bodies[1:] == sorted_bodies[:-1])
    if duplicate_offsets.size:
        duplicate_body = int(sorted_bodies[int(duplicate_offsets[0])])
        rows = [int(row) for row in np.flatnonzero(bodies == duplicate_body)]
        raise SourceDataError(
            f"duplicate neurotransmitters.body {duplicate_body} at zero-based "
            f"source rows {rows}"
        )
    return table


def select_transmitters(
    table: pa.Table, selected_body_ids: set[int]
) -> dict[int, tuple[int, Mapping[str, Any]]]:
    """Materialize only selected transmitter rows, retaining global row indices."""

    bodies = table["body"].combine_chunks().to_numpy(zero_copy_only=False)
    selected = np.asarray(sorted(selected_body_ids), dtype=np.int64)
    source_rows = np.flatnonzero(np.isin(bodies, selected)).astype(np.int64, copy=False)
    if source_rows.size == 0:
        return {}
    selected_table = table.take(pa.array(source_rows, type=pa.int64()))
    return {
        int(raw["body"]): (int(source_row), _normalize_json(raw))
        for source_row, raw in zip(source_rows, selected_table.to_pylist(), strict=True)
    }


def _duplicate_source_rows(table: pa.Table, body_pre: int, body_post: int) -> list[int]:
    matches: list[int] = []
    offset = 0
    for batch in table.to_batches(max_chunksize=65536):
        pre = batch.column(batch.schema.get_field_index("body_pre")).to_numpy(
            zero_copy_only=False
        )
        post = batch.column(batch.schema.get_field_index("body_post")).to_numpy(
            zero_copy_only=False
        )
        local = np.flatnonzero((pre == body_pre) & (post == body_post))
        matches.extend(int(offset + index) for index in local)
        offset += batch.num_rows
    return matches


def _first_duplicate_key(sorted_keys: np.memmap) -> int | None:
    block_size = 1_000_000
    previous: int | None = None
    for start in range(0, sorted_keys.shape[0], block_size):
        block = np.asarray(sorted_keys[start : start + block_size])
        if block.size == 0:
            continue
        if previous is not None and previous == int(block[0]):
            return previous
        duplicate_offsets = np.flatnonzero(block[1:] == block[:-1])
        if duplicate_offsets.size:
            return int(block[int(duplicate_offsets[0])])
        previous = int(block[-1])
    return None


def load_weights(
    path: Path,
    *,
    scratch_dir: Path | None = None,
) -> tuple[pa.Table, Mapping[str, int]]:
    """Validate every weight row and retain the >=5 subgraph with lineage.

    Duplicate checking is exact and bounded-memory: each directed pair is
    encoded as one uint64 key in a temporary memory-mapped array, sorted in
    place, and compared adjacently.  MaleCNS v1.0 body IDs fit uint32; rejecting
    a future out-of-range ID prevents key collisions instead of truncating it.
    """

    table = _read_feather(path, "weights")
    if table.column_names != ["body_pre", "body_post", "weight"]:
        raise SourceDataError(
            "weights table columns must be exactly ['body_pre', 'body_post', 'weight']"
        )
    for column in table.column_names:
        _require_int64(table, column, "weights")
        if table[column].null_count:
            raise SourceDataError(f"weights.{column} contains null values")
    if table.num_rows == 0:
        raise SourceDataError("weights table is empty")

    if scratch_dir is not None:
        scratch_dir.mkdir(parents=True, exist_ok=True)

    retained_batches: list[pa.RecordBatch] = []
    raw_weight_sum = 0
    path_edge_count = 0
    path_weight_sum = 0
    induced_edge_count = 0
    induced_weight_sum = 0
    offset = 0

    with tempfile.TemporaryDirectory(
        prefix="malecns-edge-keys-", dir=scratch_dir
    ) as temporary:
        key_path = Path(temporary) / "directed-pairs.u64"
        keys = np.memmap(key_path, dtype=np.uint64, mode="w+", shape=(table.num_rows,))
        try:
            for batch in table.to_batches(max_chunksize=65536):
                pre = batch.column(0).to_numpy(zero_copy_only=False)
                post = batch.column(1).to_numpy(zero_copy_only=False)
                weight = batch.column(2).to_numpy(zero_copy_only=False)

                if np.any(pre <= 0) or np.any(post <= 0):
                    raise SourceDataError(
                        f"weights rows {offset}:{offset + batch.num_rows} contain "
                        "non-positive body IDs"
                    )
                if np.any(weight <= 0):
                    raise SourceDataError(
                        f"weights rows {offset}:{offset + batch.num_rows} contain "
                        "non-positive weights"
                    )
                if np.any(pre > UINT32_MAX) or np.any(post > UINT32_MAX):
                    raise SourceDataError(
                        "weights contain a body ID above uint32; the exact duplicate-key "
                        "encoding is not valid for this source version"
                    )

                pre_u64 = pre.astype(np.uint64, copy=False)
                post_u64 = post.astype(np.uint64, copy=False)
                keys[offset : offset + batch.num_rows] = (pre_u64 << np.uint64(32)) | post_u64

                raw_weight_sum += int(weight.sum(dtype=np.int64))
                path_mask = weight >= PATH_MIN_WEIGHT
                induced_mask = weight >= INDUCED_MIN_WEIGHT
                path_edge_count += int(np.count_nonzero(path_mask))
                path_weight_sum += int(weight[path_mask].sum(dtype=np.int64))
                induced_edge_count += int(np.count_nonzero(induced_mask))
                induced_weight_sum += int(weight[induced_mask].sum(dtype=np.int64))

                if np.any(induced_mask):
                    local_rows = np.flatnonzero(induced_mask).astype(np.int64, copy=False)
                    retained_batches.append(
                        pa.record_batch(
                            [
                                pa.array(pre[induced_mask], type=pa.int64()),
                                pa.array(post[induced_mask], type=pa.int64()),
                                pa.array(weight[induced_mask], type=pa.int64()),
                                pa.array(local_rows + offset, type=pa.int64()),
                            ],
                            names=("body_pre", "body_post", "weight", "source_row"),
                        )
                    )
                offset += batch.num_rows

            keys.flush()
            keys.sort(kind="quicksort")
            duplicate_key = _first_duplicate_key(keys)
        finally:
            del keys

    if duplicate_key is not None:
        body_pre = duplicate_key >> 32
        body_post = duplicate_key & int(UINT32_MAX)
        rows = _duplicate_source_rows(table, body_pre, body_post)
        raise SourceDataError(
            f"duplicate directed edge ({body_pre}, {body_post}) at zero-based "
            f"source rows {rows}"
        )

    retained_schema = pa.schema(
        [
            ("body_pre", pa.int64()),
            ("body_post", pa.int64()),
            ("weight", pa.int64()),
            ("source_row", pa.int64()),
        ]
    )
    retained = (
        pa.Table.from_batches(retained_batches, schema=retained_schema)
        if retained_batches
        else pa.Table.from_batches([], schema=retained_schema)
    )
    counts = {
        "raw_edge_rows": table.num_rows,
        "raw_weight_sum": raw_weight_sum,
        "weight_gte_5_edge_rows": induced_edge_count,
        "weight_gte_5_weight_sum": induced_weight_sum,
        "weight_gte_10_edge_rows": path_edge_count,
        "weight_gte_10_weight_sum": path_weight_sum,
        "duplicate_directed_pairs": 0,
    }
    return retained, counts


class EdgeIndex:
    """Deterministic adjacency index over the source edges with weight >= 5."""

    def __init__(self, retained: pa.Table, annotated_body_ids: set[int]) -> None:
        self.pre = retained["body_pre"].combine_chunks().to_numpy(zero_copy_only=False)
        self.post = retained["body_post"].combine_chunks().to_numpy(zero_copy_only=False)
        self.weight = retained["weight"].combine_chunks().to_numpy(zero_copy_only=False)
        self.source_row = retained["source_row"].combine_chunks().to_numpy(
            zero_copy_only=False
        )

        annotated = np.asarray(sorted(annotated_body_ids), dtype=np.int64)
        path_indices = np.flatnonzero(
            (self.weight >= PATH_MIN_WEIGHT)
            & np.isin(self.pre, annotated)
            & np.isin(self.post, annotated)
        )
        self.path_indices = path_indices.astype(np.int64, copy=False)
        path_pre = self.pre[self.path_indices]
        path_post = self.post[self.path_indices]
        self.out_order = self.path_indices[np.lexsort((path_post, path_pre))]
        self.in_order = self.path_indices[np.lexsort((path_pre, path_post))]
        self.out_keys = self.pre[self.out_order]
        self.in_keys = self.post[self.in_order]

    def edge(self, index: int) -> Edge:
        return Edge(
            body_pre=int(self.pre[index]),
            body_post=int(self.post[index]),
            weight=int(self.weight[index]),
            source_row=int(self.source_row[index]),
        )

    def outgoing_indices(self, body_id: int) -> np.ndarray:
        left = int(np.searchsorted(self.out_keys, body_id, side="left"))
        right = int(np.searchsorted(self.out_keys, body_id, side="right"))
        return self.out_order[left:right]

    def incoming_indices(self, body_id: int) -> np.ndarray:
        left = int(np.searchsorted(self.in_keys, body_id, side="left"))
        right = int(np.searchsorted(self.in_keys, body_id, side="right"))
        return self.in_order[left:right]

    def induced_edges(self, selected_nodes: set[int]) -> list[Edge]:
        if not selected_nodes:
            return []
        ordered_nodes = np.asarray(sorted(selected_nodes), dtype=np.int64)
        mask = np.isin(self.pre, ordered_nodes) & np.isin(self.post, ordered_nodes)
        indices = np.flatnonzero(mask)
        edges = [self.edge(int(index)) for index in indices]
        return sorted(edges, key=lambda edge: (edge.body_pre, edge.body_post))


def _annotation_matches(record: Mapping[str, Any]) -> set[str]:
    matches: set[str] = set()
    neuron_type = record.get("type")
    if isinstance(neuron_type, str) and neuron_type.startswith("JO-E"):
        matches.add("sensory")
    if (
        record.get("subclass") == "campaniform sensilla"
        and record.get("entryNerve") in {"ADMN", "DMetaN"}
    ):
        matches.add("sensory")
    if neuron_type == "DNg07":
        matches.add("named:DNg07")
    if neuron_type == "AN27X008":
        matches.add("named:AN27X008")
    if record.get("superclass") == "vnc_motor" and record.get("subclass") == "wm":
        matches.add("motor")
    return matches


def derive_side(record: AnnotationRecord, *, required: bool) -> tuple[str | None, dict[str, str]]:
    evidence: dict[str, str] = {}
    for field in ("rootSide", "somaSide"):
        value = record.record.get(field)
        if value in {"L", "R"}:
            evidence[field] = value
    instance = record.record.get("instance")
    if isinstance(instance, str):
        match = INSTANCE_SIDE_RE.search(instance)
        if match is not None:
            evidence["instance_suffix"] = match.group(1)

    observed = set(evidence.values())
    if len(observed) > 1:
        raise CircuitResolutionError(
            f"side mismatch for body {record.body_id}: {evidence}"
        )
    if not observed:
        if required:
            raise CircuitResolutionError(
                f"incomplete side for required role body {record.body_id}: no exact "
                "rootSide, somaSide, or _L/_R instance suffix"
            )
        return None, evidence
    return next(iter(observed)), evidence


def resolve_candidates(
    annotations: Mapping[int, AnnotationRecord],
) -> tuple[dict[str, dict[str, set[int]]], dict[int, set[str]]]:
    candidates = {
        "L": {"sensory": set(), "DNg07": set(), "AN27X008": set(), "motor": set()},
        "R": {"sensory": set(), "DNg07": set(), "AN27X008": set(), "motor": set()},
    }
    roles_by_body: dict[int, set[str]] = {}

    for body_id, annotation in annotations.items():
        matches = _annotation_matches(annotation.record)
        if not matches:
            continue
        broad_roles = {
            "named" if match.startswith("named:") else match for match in matches
        }
        if len(broad_roles) > 1:
            raise CircuitResolutionError(
                f"ambiguous role annotation for body {body_id}: {sorted(matches)!r}"
            )
        side, _ = derive_side(annotation, required=True)
        assert side is not None
        roles_by_body[body_id] = matches
        for match in matches:
            if match == "sensory":
                candidates[side]["sensory"].add(body_id)
            elif match == "motor":
                candidates[side]["motor"].add(body_id)
            else:
                candidates[side][match.split(":", 1)[1]].add(body_id)

    for side in ("L", "R"):
        for role in ("sensory", "DNg07", "AN27X008", "motor"):
            if not candidates[side][role]:
                raise CircuitResolutionError(
                    f"missing exact {role} candidate on side {side}"
                )
    return candidates, roles_by_body


def _enumerate_upstream(
    graph: EdgeIndex,
    named_body_id: int,
    sensory: set[int],
) -> list[DirectedPath]:
    paths: list[DirectedPath] = []

    def visit(
        current: int,
        reverse_nodes: tuple[int, ...],
        reverse_edges: tuple[Edge, ...],
        visited: frozenset[int],
    ) -> None:
        hops = len(reverse_edges)
        if hops >= 1 and current in sensory:
            edges = tuple(reversed(reverse_edges))
            paths.append(
                DirectedPath(
                    nodes=tuple(reversed(reverse_nodes)),
                    edges=edges,
                    bottleneck=min(edge.weight for edge in edges),
                )
            )
        if hops == MAX_SEGMENT_HOPS:
            return
        for index_value in graph.incoming_indices(current):
            edge = graph.edge(int(index_value))
            predecessor = edge.body_pre
            if predecessor in visited:
                continue
            visit(
                predecessor,
                reverse_nodes + (predecessor,),
                reverse_edges + (edge,),
                visited | {predecessor},
            )

    visit(named_body_id, (named_body_id,), (), frozenset({named_body_id}))
    return paths


def _enumerate_downstream(
    graph: EdgeIndex,
    named_body_id: int,
    motor: set[int],
) -> list[DirectedPath]:
    paths: list[DirectedPath] = []

    def visit(
        current: int,
        nodes: tuple[int, ...],
        edges: tuple[Edge, ...],
        visited: frozenset[int],
    ) -> None:
        hops = len(edges)
        if hops >= 1 and current in motor:
            paths.append(
                DirectedPath(
                    nodes=nodes,
                    edges=edges,
                    bottleneck=min(edge.weight for edge in edges),
                )
            )
        if hops == MAX_SEGMENT_HOPS:
            return
        for index_value in graph.outgoing_indices(current):
            edge = graph.edge(int(index_value))
            successor = edge.body_post
            if successor in visited:
                continue
            visit(
                successor,
                nodes + (successor,),
                edges + (edge,),
                visited | {successor},
            )

    visit(named_body_id, (named_body_id,), (), frozenset({named_body_id}))
    return paths


def select_side_path(
    side: str,
    candidates: Mapping[str, set[int]],
    annotations: Mapping[int, AnnotationRecord],
    graph: EdgeIndex,
) -> tuple[SideSelection, list[dict[str, Any]]]:
    sensory = candidates["sensory"]
    motor = candidates["motor"]
    named_ids = sorted(candidates["DNg07"] | candidates["AN27X008"])
    viable: list[SideSelection] = []
    evaluations: list[dict[str, Any]] = []

    for named_body_id in named_ids:
        upstream = _enumerate_upstream(graph, named_body_id, sensory)
        downstream = _enumerate_downstream(graph, named_body_id, motor)
        named_type = str(annotations[named_body_id].record.get("type"))
        evaluation: dict[str, Any] = {
            "body_id": named_body_id,
            "type": named_type,
            "upstream_path_count": len(upstream),
            "downstream_path_count": len(downstream),
            "complete": bool(upstream and downstream),
        }
        if upstream and downstream:
            complete_bottleneck = min(
                max(path.bottleneck for path in upstream),
                max(path.bottleneck for path in downstream),
            )
            chosen_upstream = min(
                (path for path in upstream if path.bottleneck >= complete_bottleneck),
                key=lambda path: path.nodes,
            )
            chosen_downstream = min(
                (path for path in downstream if path.bottleneck >= complete_bottleneck),
                key=lambda path: path.nodes,
            )
            complete_nodes = chosen_upstream.nodes + chosen_downstream.nodes[1:]
            viable.append(
                SideSelection(
                    side=side,
                    named_body_id=named_body_id,
                    named_type=named_type,
                    upstream=chosen_upstream,
                    downstream=chosen_downstream,
                    complete_nodes=complete_nodes,
                    bottleneck=complete_bottleneck,
                    upstream_candidate_count=len(upstream),
                    downstream_candidate_count=len(downstream),
                )
            )
            evaluation["maximum_complete_bottleneck"] = complete_bottleneck
        evaluations.append(evaluation)

    if not viable:
        raise CircuitResolutionError(
            f"no complete directed sensory->named->motor route on side {side} with "
            f"each segment <= {MAX_SEGMENT_HOPS} edges and every path edge weight "
            f">= {PATH_MIN_WEIGHT}"
        )
    maximum = max(selection.bottleneck for selection in viable)
    selected = min(
        (selection for selection in viable if selection.bottleneck == maximum),
        key=lambda selection: selection.complete_nodes,
    )
    return selected, evaluations


def _edge_document(edge: Edge) -> dict[str, Any]:
    return {
        "body_post": edge.body_post,
        "body_pre": edge.body_pre,
        "sign_status": "unknown",
        "source_row": edge.source_row,
        "weight": edge.weight,
    }


def _path_document(path: DirectedPath) -> dict[str, Any]:
    return {
        "body_id_tuple": list(path.nodes),
        "bottleneck_weight": path.bottleneck,
        "edge_count": len(path.edges),
        "edges": [_edge_document(edge) for edge in path.edges],
    }


def _candidate_document(
    candidates: Mapping[str, Mapping[str, set[int]]]
) -> dict[str, Any]:
    by_side: dict[str, Any] = {}
    for side in ("L", "R"):
        by_side[side] = {}
        for role in ("sensory", "DNg07", "AN27X008", "motor"):
            body_ids = sorted(candidates[side][role])
            by_side[side][role] = {"body_ids": body_ids, "count": len(body_ids)}
    return {
        "by_side": by_side,
        "total_unique_role_candidates": len(
            set().union(
                *(candidates[side][role] for side in ("L", "R") for role in candidates[side])
            )
        ),
    }


def _transmitter_document(
    body_id: int,
    transmitters: Mapping[int, tuple[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    item = transmitters.get(body_id)
    if item is None:
        return {
            "record": None,
            "sign_status": "unknown",
            "source_row": None,
            "status": "missing_source_row",
        }
    source_row, record = item
    evidence = [record.get(field) for field in TRANSMITTER_EVIDENCE_FIELDS if field in record]

    def is_resolved(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip().lower() not in {"", "unclear", "unknown", "n/a", "na"}
        return True

    reported = any(is_resolved(value) for value in evidence)
    return {
        "record": record,
        "sign_status": "unknown",
        "source_row": source_row,
        "status": "reported_unmapped" if reported else "unknown_transmitter",
    }


def _selection_policy() -> dict[str, Any]:
    return {
        "candidate_resolution": {
            "ambiguous_role_policy": "fail",
            "motor": {"subclass_exact": "wm", "superclass_exact": "vnc_motor"},
            "named_types_exact": ["AN27X008", "DNg07"],
            "sensory_union": [
                {"type_prefix": "JO-E"},
                {
                    "entryNerve_exact_any": ["ADMN", "DMetaN"],
                    "subclass_exact": "campaniform sensilla",
                },
            ],
            "string_matching": "case-sensitive exact except the declared JO-E prefix",
        },
        "complete_selected_path_criterion": {
            "definition": (
                "For each side independently, concatenate one simple directed sensory-to-"
                "named segment and one simple directed named-to-motor segment, omitting the "
                "repeated named join node. Each segment has 1..3 source edges. Role endpoints "
                "must have that exact derived side; intermediate nodes are not side-filtered."
            ),
            "direction": "body_pre -> body_post",
            "duplicate_directed_pair_policy": "fail across every source weight row",
            "maximum_edges_per_segment": MAX_SEGMENT_HOPS,
            "objective": "maximize the minimum raw source weight over both segments",
            "path_edge_minimum_weight_inclusive": PATH_MIN_WEIGHT,
            "path_vertex_eligibility": "every path body must have a source annotation row",
            "segment_vertex_repetition": "forbidden within each segment",
            "tie_break": (
                "lexicographically smallest complete ordered source body-ID tuple among "
                "all maximum-bottleneck complete paths"
            ),
        },
        "fixture_closure": {
            "definition": "all source edges whose endpoints are both selected path nodes",
            "minimum_weight_inclusive": INDUCED_MIN_WEIGHT,
            "self_edges": "included when present",
        },
        "side_derivation": {
            "allowed_evidence": ["rootSide exact L/R", "somaSide exact L/R", "instance _L/_R suffix"],
            "conflict": "fail",
            "missing_required_role_side": "fail",
            "unresolved_intermediate_side": "retain as null",
        },
        "transmitter_and_sign": {
            "mapping": "none",
            "policy": "copy the exact aggregate transmitter row when present; retain unknown edge signs",
        },
    }


def _strip_source_rows(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_source_rows(item)
            for key, item in value.items()
            if key != "source_row"
        }
    if isinstance(value, list):
        return [_strip_source_rows(item) for item in value]
    return value


def _semantic_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    projection = {
        key: value
        for key, value in payload.items()
        if key not in {"input_receipt", "extraction_receipt"}
    }
    return _strip_source_rows(projection)


def _rights_document(
    manifest: Manifest,
    *,
    node_count: int,
    edge_count: int,
) -> dict[str, Any]:
    citation = _require_mapping(
        manifest.dataset.get("citation"), "manifest.dataset.citation"
    )
    return {
        "attribution": (
            "MaleCNS v1.0 connectome data by FlyEM (HHMI Janelia), the "
            "University of Cambridge Department of Zoology, the MRC Laboratory "
            "of Molecular Biology, and Google Research."
        ),
        "changes": (
            f"Deterministically selected {node_count} source neurons and all "
            f"{edge_count} directed source edges with weight >= 5 among the "
            "bilateral maximum-bottleneck flight-power paths; copied selected "
            "annotation and aggregate transmitter evidence; retained every edge "
            "sign as unknown; did not modify source values."
        ),
        "citation": _normalize_json(citation),
        "license": "CC BY 4.0",
        "license_spdx": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/legalcode.en",
        "redistribution_decision": "permitted_with_attribution_and_change_notice",
        "redistribution_scope": (
            "This small derived JSON fixture may be committed and redistributed "
            "with this attribution, license link, citation, provenance, and change "
            "notice. The three bulk source objects remain excluded from Git."
        ),
        "source": "https://male-cns.janelia.org/download/",
        "source_dataset_remains_third_party": True,
    }


def build_fixture(
    manifest: Manifest,
    annotations: Mapping[int, AnnotationRecord],
    annotation_row_count: int,
    transmitter_table: pa.Table,
    retained_weights: pa.Table,
    weight_counts: Mapping[str, int],
) -> dict[str, Any]:
    candidates, roles_by_body = resolve_candidates(annotations)
    graph = EdgeIndex(retained_weights, set(annotations))

    selections: dict[str, SideSelection] = {}
    evaluations: dict[str, list[dict[str, Any]]] = {}
    for side in ("L", "R"):
        selection, side_evaluations = select_side_path(
            side, candidates[side], annotations, graph
        )
        selections[side] = selection
        evaluations[side] = side_evaluations

    selected_ids = set().union(
        *(set(selection.complete_nodes) for selection in selections.values())
    )
    missing_annotations = sorted(selected_ids - set(annotations))
    if missing_annotations:
        raise CircuitResolutionError(
            "selected path contains bodies without annotation rows: "
            f"{missing_annotations!r}"
        )

    transmitters = select_transmitters(transmitter_table, selected_ids)

    side_details: dict[int, tuple[str | None, dict[str, str]]] = {}
    for body_id in selected_ids:
        required = body_id in roles_by_body
        side_details[body_id] = derive_side(annotations[body_id], required=required)

    nodes: list[dict[str, Any]] = []
    unknown_transmitter_count = 0
    for body_id in sorted(selected_ids):
        annotation = annotations[body_id]
        side, side_evidence = side_details[body_id]
        transmitter = _transmitter_document(body_id, transmitters)
        if transmitter["status"] != "reported_unmapped":
            unknown_transmitter_count += 1
        roles = sorted(roles_by_body.get(body_id, {"intermediate"}))
        nodes.append(
            {
                "annotation": {
                    "record": annotation.record,
                    "source_row": annotation.source_row,
                },
                "body_id": body_id,
                "roles": roles,
                "side": side,
                "side_evidence": side_evidence,
                "transmitter": transmitter,
            }
        )

    induced_edges = graph.induced_edges(selected_ids)
    policy = _selection_policy()
    source_receipts = []
    for source_id in REQUIRED_SOURCE_IDS:
        source = manifest.sources[source_id]
        source_receipts.append(
            {
                "bytes": source.expected_bytes,
                "filename": source.filename,
                "generation": source.generation,
                "id": source.source_id,
                "sha256": source.sha256,
                "url": source.url,
            }
        )

    selection_document: dict[str, Any] = {}
    for side in ("L", "R"):
        selection = selections[side]
        selection_document[side] = {
            "bottleneck_weight": selection.bottleneck,
            "complete_body_id_tuple": list(selection.complete_nodes),
            "downstream": _path_document(selection.downstream),
            "downstream_candidate_path_count_for_selected_named": selection.downstream_candidate_count,
            "named_body_id": selection.named_body_id,
            "named_type": selection.named_type,
            "upstream": _path_document(selection.upstream),
            "upstream_candidate_path_count_for_selected_named": selection.upstream_candidate_count,
        }

    script_path = Path(__file__).resolve()
    payload: dict[str, Any] = {
        "candidate_resolution": _candidate_document(candidates),
        "counts": {
            "annotation_source_rows": annotation_row_count,
            "emitted_edge_rows": len(induced_edges),
            "emitted_node_rows": len(nodes),
            "emitted_weight_sum": sum(edge.weight for edge in induced_edges),
            "neurotransmitter_source_rows": transmitter_table.num_rows,
            "selected_nodes_without_reported_transmitter": unknown_transmitter_count,
            **dict(weight_counts),
        },
        "dataset_id": DATASET_ID,
        "edges": [_edge_document(edge) for edge in induced_edges],
        "extraction_receipt": {
            "code_sha256": _sha256_file(script_path),
            "configuration_sha256": _sha256_bytes(_canonical_bytes(policy)),
            "implementation": script_path.name,
        },
        "fixture_id": FIXTURE_ID,
        "input_receipt": {
            "dataset": _normalize_json(manifest.dataset),
            "manifest": {
                "bytes": len(manifest.raw_bytes),
                "required_sha256": manifest.required_sha256,
                "sha256": manifest.raw_sha256,
            },
            "sources": source_receipts,
        },
        "named_candidate_path_evaluations": evaluations,
        "nodes": nodes,
        "rights": _rights_document(
            manifest,
            node_count=len(nodes),
            edge_count=len(induced_edges),
        ),
        "schema_version": 1,
        "selected_paths": selection_document,
        "selection_policy": policy,
        "transmitter_sign_boundary": (
            "No transmitter-to-sign mapping was applied; every emitted edge sign remains unknown."
        ),
    }

    semantic_sha256 = _sha256_bytes(_canonical_bytes(_semantic_projection(payload)))
    payload_sha256 = _sha256_bytes(_canonical_bytes(payload))
    payload["hashes"] = {
        "payload_sha256": payload_sha256,
        "payload_scope": "canonical JSON document with the hashes object omitted",
        "semantic_sha256": semantic_sha256,
        "semantic_scope": (
            "canonical payload excluding input/extraction receipts and every source_row; "
            "this identifies selection semantics while payload_sha256 preserves exact lineage"
        ),
    }
    return payload


def write_fixture(document: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_bytes(document) + b"\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def extract_fixture(
    manifest_path: Path,
    cache_dir: Path,
    output_path: Path,
    *,
    scratch_dir: Path | None = None,
    expected_manifest_sha256: str = EXPECTED_MANIFEST_SHA256,
) -> dict[str, Any]:
    """Verify, extract, atomically write, and return a fixture receipt."""

    manifest = load_manifest(
        manifest_path, expected_sha256=expected_manifest_sha256
    )
    source_paths = verify_sources(manifest, cache_dir)
    annotations, annotation_count = load_annotations(source_paths["annotations"])
    transmitter_table = load_transmitters(source_paths["neurotransmitters"])
    retained, weight_counts = load_weights(
        source_paths["weights"], scratch_dir=scratch_dir
    )
    document = build_fixture(
        manifest,
        annotations,
        annotation_count,
        transmitter_table,
        retained,
        weight_counts,
    )
    write_fixture(document, output_path)
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract a deterministic, source-row-traceable MaleCNS v1.0 fixture"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=None,
        help=(
            "directory for the exact duplicate-pair sort (about 8 bytes per source edge; "
            "system temporary storage by default)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        document = extract_fixture(
            args.manifest,
            args.cache_dir,
            args.output,
            scratch_dir=args.scratch_dir,
        )
    except FixtureExtractionError as exc:
        print(f"fixture extraction failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "edge_count": document["counts"]["emitted_edge_rows"],
                "fixture_id": document["fixture_id"],
                "node_count": document["counts"]["emitted_node_rows"],
                "output": str(args.output),
                "payload_sha256": document["hashes"]["payload_sha256"],
                "semantic_sha256": document["hashes"]["semantic_sha256"],
                "status": "written",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
