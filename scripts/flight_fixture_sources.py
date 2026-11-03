"""Verified MaleCNS v1.0 source access for the flight-steering extractor.

This module owns everything between the pinned bytes on disk and validated
in-memory arrays: the acquisition manifest contract, the byte/digest gate over
the three immutable Feather objects, and the row-level validation of the
annotation, neurotransmitter, and connection-weight tables.

It contains no circuit policy. Thresholds are arguments, so the frozen selection
rules live in exactly one place — ``scripts/extract_flight_fixture.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.feather as feather  # type: ignore[import-untyped]

IntArray = npt.NDArray[np.int64]

DATASET_ID = "male-cns:v1.0"
EXPECTED_MANIFEST_SHA256 = "9412105f54884a57521bbdf7fe46659caab559ef7877e6a62af95f50e1ee3768"
REQUIRED_SOURCE_IDS = ("annotations", "neurotransmitters", "weights")
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

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
UINT32_MAX = int(np.iinfo(np.uint32).max)


class FixtureExtractionError(RuntimeError):
    """Stable, user-facing extraction failure."""


class ManifestError(FixtureExtractionError):
    """The source manifest does not satisfy the pinned contract."""


class SourceVerificationError(FixtureExtractionError):
    """A source object does not match its manifest pin."""


class SourceDataError(FixtureExtractionError):
    """A verified Feather object violates the extraction contract."""


class CircuitResolutionError(FixtureExtractionError):
    """The declared flight-steering circuit cannot be resolved exactly."""


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """One generation-qualified source object named by the manifest."""

    source_id: str
    filename: str
    expected_bytes: int
    sha256: str
    url: str
    generation: str | None


@dataclass(frozen=True, slots=True)
class Manifest:
    """The verified acquisition manifest and its own content hash."""

    raw_bytes: bytes
    raw_sha256: str
    required_sha256: str
    dataset: Mapping[str, Any]
    sources: Mapping[str, SourceSpec]


@dataclass(frozen=True, slots=True)
class AnnotationRecord:
    """One annotation row with zero-based source lineage."""

    source_row: int
    body_id: int
    record: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RetainedWeights:
    """The closure-eligible weight rows plus whole-table audit counts."""

    body_pre: IntArray
    body_post: IntArray
    weight: IntArray
    source_row: IntArray
    counts: Mapping[str, int]


def canonical_bytes(value: object) -> bytes:
    """Canonical JSON bytes, byte-for-byte identical to the loader's encoder.

    ``neurocraft_connectome.model.canonical_json_bytes`` recomputes the embedded
    payload and semantic digests at load time, so this encoder must match it
    exactly, including ``ensure_ascii=True``. Escaping non-ASCII keeps a future
    source string with an accent or a dash from silently splitting the extractor
    identity from the loader identity.
    """

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be a JSON object")
    return value


def normalize_json(value: Any) -> Any:
    """Convert Arrow/Python values into an explicit, canonical JSON domain."""

    if value is None or isinstance(value, str | bool | int):
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
        return {str(key): normalize_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [normalize_json(item) for item in value]
    if hasattr(value, "item"):
        return normalize_json(value.item())
    return str(value)


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON key in manifest: {key!r}")
        result[key] = value
    return result


def _require_string(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{label}.{key} must be a non-empty string")
    return value


def _source_spec(source: Mapping[str, Any], label: str) -> SourceSpec:
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
    generation = source.get("generation")
    if generation is not None and (not isinstance(generation, str) or not generation):
        raise ManifestError(f"{label}.generation must be null or a non-empty string")
    return SourceSpec(
        source_id=_require_string(source, "id", label),
        filename=filename,
        expected_bytes=expected_bytes,
        sha256=sha256,
        url=_require_string(source, "url", label),
        generation=generation,
    )


def load_manifest(path: Path, *, expected_sha256: str = EXPECTED_MANIFEST_SHA256) -> Manifest:
    """Load the exact acquisition manifest schema used by this repository."""

    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise ManifestError(f"cannot read source manifest {path}: {exc}") from exc

    observed = sha256_bytes(raw_bytes)
    if SHA256_RE.fullmatch(expected_sha256) is None:
        raise ManifestError("expected manifest SHA-256 is not a valid lowercase digest")
    if observed != expected_sha256:
        raise ManifestError(
            f"source manifest SHA-256 mismatch: expected {expected_sha256}, observed {observed}"
        )

    try:
        document = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"invalid UTF-8 JSON source manifest: {exc}") from exc

    root = require_mapping(document, "manifest")
    if root.get("schema_version") != 1:
        raise ManifestError("manifest.schema_version must equal 1")
    dataset = require_mapping(root.get("dataset"), "manifest.dataset")
    if dataset.get("id") != DATASET_ID:
        raise ManifestError(f"manifest.dataset.id must equal {DATASET_ID!r}")
    raw_sources = root.get("sources")
    if not isinstance(raw_sources, list):
        raise ManifestError("manifest.sources must be a JSON array")

    sources: dict[str, SourceSpec] = {}
    for index, raw_source in enumerate(raw_sources):
        label = f"manifest.sources[{index}]"
        spec = _source_spec(require_mapping(raw_source, label), label)
        if spec.source_id in sources:
            raise ManifestError(f"duplicate source ID in manifest: {spec.source_id!r}")
        sources[spec.source_id] = spec
    if set(sources) != set(REQUIRED_SOURCE_IDS):
        raise ManifestError(
            f"manifest.sources must contain exactly {list(REQUIRED_SOURCE_IDS)!r}; "
            f"got {sorted(sources)!r}"
        )

    return Manifest(
        raw_bytes=raw_bytes,
        raw_sha256=observed,
        required_sha256=expected_sha256,
        dataset=dataset,
        sources=sources,
    )


def verify_sources(manifest: Manifest, cache_dir: Path) -> dict[str, Path]:
    """Verify all three source objects before allowing any Feather parsing."""

    paths: dict[str, Path] = {}
    for source_id in REQUIRED_SOURCE_IDS:
        spec = manifest.sources[source_id]
        path = cache_dir / spec.filename
        try:
            info = path.stat()
        except OSError as exc:
            raise SourceVerificationError(
                f"cannot stat required source {source_id!r} at {path}: {exc}"
            ) from exc
        if not path.is_file():
            raise SourceVerificationError(
                f"required source {source_id!r} is not a regular file: {path}"
            )
        if info.st_size != spec.expected_bytes:
            raise SourceVerificationError(
                f"source {source_id!r} byte mismatch: expected {spec.expected_bytes}, "
                f"observed {info.st_size}"
            )
        observed = sha256_file(path)
        if observed != spec.sha256:
            raise SourceVerificationError(
                f"source {source_id!r} SHA-256 mismatch: expected {spec.sha256}, "
                f"observed {observed}"
            )
        paths[source_id] = path
    return paths


def _read_feather(path: Path, label: str) -> Any:
    try:
        return feather.read_table(path, memory_map=True)
    except Exception as exc:  # pyarrow raises several format/IO exception types
        raise SourceDataError(f"cannot read verified {label} Feather table: {exc}") from exc


def _require_columns(table: Any, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(table.column_names))
    if missing:
        raise SourceDataError(f"{label} table is missing required columns: {missing!r}")


def _require_int64(table: Any, column: str, label: str) -> None:
    if not pa.types.is_int64(table.schema.field(column).type):
        raise SourceDataError(f"{label}.{column} must have Arrow int64 type")


def _int_column(table: Any, name: str) -> IntArray:
    column = table[name].combine_chunks().to_numpy(zero_copy_only=False)
    typed: IntArray = column.astype(np.int64, copy=False)
    return typed


def load_annotations(path: Path) -> tuple[dict[int, AnnotationRecord], int]:
    """Index every annotation row by body ID, refusing duplicates."""

    table = _read_feather(path, "annotations")
    _require_columns(table, REQUIRED_ANNOTATION_COLUMNS, "annotations")
    _require_int64(table, "bodyId", "annotations")
    if table["bodyId"].null_count:
        raise SourceDataError("annotations.bodyId contains null values")

    records: dict[int, AnnotationRecord] = {}
    for source_row, raw in enumerate(table.to_pylist()):
        body_id = raw.get("bodyId")
        if isinstance(body_id, bool) or not isinstance(body_id, int) or body_id <= 0:
            raise SourceDataError(f"annotations row {source_row} has invalid bodyId {body_id!r}")
        if body_id in records:
            previous = records[body_id].source_row
            raise SourceDataError(
                f"duplicate annotations.bodyId {body_id} at zero-based source rows "
                f"{previous} and {source_row}"
            )
        records[body_id] = AnnotationRecord(
            source_row=source_row,
            body_id=body_id,
            record=normalize_json(raw),
        )
    return records, int(table.num_rows)


def load_transmitters(path: Path) -> Any:
    """Validate the aggregate transmitter table without materializing rows."""

    table = _read_feather(path, "neurotransmitters")
    _require_columns(table, ("body",), "neurotransmitters")
    _require_int64(table, "body", "neurotransmitters")
    if table["body"].null_count:
        raise SourceDataError("neurotransmitters.body contains null values")

    bodies = _int_column(table, "body")
    if bool(np.any(bodies <= 0)):
        row = int(np.flatnonzero(bodies <= 0)[0])
        raise SourceDataError(f"neurotransmitters row {row} has invalid body {int(bodies[row])!r}")
    ordered = np.sort(bodies, kind="stable")
    duplicates = np.flatnonzero(ordered[1:] == ordered[:-1])
    if duplicates.size:
        duplicate = int(ordered[int(duplicates[0])])
        rows = [int(row) for row in np.flatnonzero(bodies == duplicate)]
        raise SourceDataError(
            f"duplicate neurotransmitters.body {duplicate} at zero-based source rows {rows}"
        )
    return table


def select_transmitters(table: Any, selected: set[int]) -> dict[int, tuple[int, Mapping[str, Any]]]:
    """Materialize only selected transmitter rows, retaining global row indices."""

    bodies = _int_column(table, "body")
    wanted = np.asarray(sorted(selected), dtype=np.int64)
    rows = np.flatnonzero(np.isin(bodies, wanted)).astype(np.int64, copy=False)
    if rows.size == 0:
        return {}
    chosen = table.take(pa.array(rows, type=pa.int64()))
    return {
        int(raw["body"]): (int(row), normalize_json(raw))
        for row, raw in zip(rows, chosen.to_pylist(), strict=True)
    }


def _duplicate_source_rows(table: Any, body_pre: int, body_post: int) -> list[int]:
    matches: list[int] = []
    offset = 0
    for batch in table.to_batches(max_chunksize=65536):
        pre = batch.column(0).to_numpy(zero_copy_only=False)
        post = batch.column(1).to_numpy(zero_copy_only=False)
        local = np.flatnonzero((pre == body_pre) & (post == body_post))
        matches.extend(int(offset + index) for index in local)
        offset += batch.num_rows
    return matches


def _first_duplicate_key(sorted_keys: Any) -> int | None:
    block_size = 1_000_000
    previous: int | None = None
    for start in range(0, int(sorted_keys.shape[0]), block_size):
        block = np.asarray(sorted_keys[start : start + block_size])
        if block.size == 0:
            continue
        if previous is not None and previous == int(block[0]):
            return previous
        duplicates = np.flatnonzero(block[1:] == block[:-1])
        if duplicates.size:
            return int(block[int(duplicates[0])])
        previous = int(block[-1])
    return None


def _validate_weight_batch(pre: Any, post: Any, weight: Any, span: str) -> None:
    if bool(np.any(pre <= 0)) or bool(np.any(post <= 0)):
        raise SourceDataError(f"weights rows {span} contain non-positive body IDs")
    if bool(np.any(weight <= 0)):
        raise SourceDataError(f"weights rows {span} contain non-positive weights")
    if bool(np.any(pre > UINT32_MAX)) or bool(np.any(post > UINT32_MAX)):
        raise SourceDataError(
            "weights contain a body ID above uint32; the exact duplicate-key encoding "
            "is not valid for this source version"
        )


def load_weights(
    path: Path,
    *,
    path_min_weight: int,
    induced_min_weight: int,
    scratch_dir: Path | None = None,
) -> RetainedWeights:
    """Validate every weight row and retain the closure-eligible subgraph.

    Duplicate checking is exact and bounded-memory: each directed pair becomes
    one uint64 key in a temporary memory-mapped array that is sorted in place and
    compared adjacently. MaleCNS v1.0 body IDs fit uint32; a future out-of-range
    ID is refused rather than truncated into a colliding key.
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

    parts: dict[str, list[IntArray]] = {"pre": [], "post": [], "weight": [], "row": []}
    raw_weight_sum = 0
    path_edges = 0
    path_weight_sum = 0
    induced_edges = 0
    induced_weight_sum = 0
    offset = 0

    with tempfile.TemporaryDirectory(prefix="malecns-flight-keys-", dir=scratch_dir) as temporary:
        keys = np.memmap(
            Path(temporary) / "directed-pairs.u64",
            dtype=np.uint64,
            mode="w+",
            shape=(int(table.num_rows),),
        )
        try:
            for batch in table.to_batches(max_chunksize=65536):
                pre = batch.column(0).to_numpy(zero_copy_only=False)
                post = batch.column(1).to_numpy(zero_copy_only=False)
                weight = batch.column(2).to_numpy(zero_copy_only=False)
                _validate_weight_batch(pre, post, weight, f"{offset}:{offset + batch.num_rows}")
                keys[offset : offset + batch.num_rows] = (
                    pre.astype(np.uint64, copy=False) << np.uint64(32)
                ) | post.astype(np.uint64, copy=False)

                raw_weight_sum += int(weight.sum(dtype=np.int64))
                path_mask = weight >= path_min_weight
                induced_mask = weight >= induced_min_weight
                path_edges += int(np.count_nonzero(path_mask))
                path_weight_sum += int(weight[path_mask].sum(dtype=np.int64))
                induced_edges += int(np.count_nonzero(induced_mask))
                induced_weight_sum += int(weight[induced_mask].sum(dtype=np.int64))
                if bool(np.any(induced_mask)):
                    local = np.flatnonzero(induced_mask).astype(np.int64, copy=False)
                    parts["pre"].append(pre[induced_mask].astype(np.int64, copy=False))
                    parts["post"].append(post[induced_mask].astype(np.int64, copy=False))
                    parts["weight"].append(weight[induced_mask].astype(np.int64, copy=False))
                    parts["row"].append(local + offset)
                offset += batch.num_rows

            keys.flush()
            keys.sort(kind="stable")
            duplicate_key = _first_duplicate_key(keys)
        finally:
            del keys

    if duplicate_key is not None:
        body_pre = duplicate_key >> 32
        body_post = duplicate_key & UINT32_MAX
        rows = _duplicate_source_rows(table, body_pre, body_post)
        raise SourceDataError(
            f"duplicate directed edge ({body_pre}, {body_post}) at zero-based source rows {rows}"
        )

    def joined(name: str) -> IntArray:
        values = parts[name]
        return np.concatenate(values) if values else np.zeros(0, dtype=np.int64)

    return RetainedWeights(
        body_pre=joined("pre"),
        body_post=joined("post"),
        weight=joined("weight"),
        source_row=joined("row"),
        counts={
            "raw_edge_rows": int(table.num_rows),
            "raw_weight_sum": raw_weight_sum,
            "weight_gte_5_edge_rows": induced_edges,
            "weight_gte_5_weight_sum": induced_weight_sum,
            "weight_gte_10_edge_rows": path_edges,
            "weight_gte_10_weight_sum": path_weight_sum,
            "duplicate_directed_pairs": 0,
        },
    )
