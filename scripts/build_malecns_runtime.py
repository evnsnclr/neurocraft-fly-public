#!/usr/bin/env python3
"""Build the deterministic full MaleCNS superclass-retained runtime artifact.

The source tables are verified before parsing. The transformation retains each
annotation row whose ``superclass`` is non-null and non-empty, then retains each
directed source weight row only when both endpoints are retained. Source rows
are already directed pairs in MaleCNS v1.0; duplicates are rejected rather than
silently aggregated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:  # Windows does not provide the POSIX resource module.
    resource = None  # type: ignore[assignment]

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.feather as feather
except ImportError as exc:  # pragma: no cover - CLI dependency boundary
    raise SystemExit("build_malecns_runtime.py requires numpy==2.2.5 and pyarrow==21.0.0") from exc

from extract_malecns_fixture import (
    EXPECTED_MANIFEST_SHA256,
    FixtureExtractionError,
    Manifest,
    load_manifest,
    verify_sources,
)
from neurocraft_connectome import (
    BOUNDARY_BODY_SUPERCLASSES,
    BOUNDARY_ROLE_BODY_IDS,
    FULL_ARTIFACT_ID,
    FULL_RETENTION_POLICY,
    MALECNS_FIXTURE_V1_PIN,
    load_full_runtime_artifact,
)
from neurocraft_connectome.model import canonical_json_bytes

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "data/manifests/male-cns-v1.0.sources.json"
DEFAULT_CACHE = REPOSITORY_ROOT / "data/cache/male-cns-v1.0"
DEFAULT_OUTPUT = REPOSITORY_ROOT / f"data/processed/{FULL_ARTIFACT_ID}"
DEFAULT_RECEIPT = Path(tempfile.gettempdir()) / "neurocraft-male-cns-g1-receipt.json"
DATASET_ID = "male-cns:v1.0"
SCHEMA_VERSION = 1
UINT32_MAX = np.iinfo(np.uint32).max
UINT64_MAX = np.iinfo(np.uint64).max
PAIR_KEY_DTYPE = np.dtype([("body_pre", "<u8"), ("body_post", "<u8")], align=False)
EDGE_RECORD_DTYPE = np.dtype(
    [
        ("post_index", "<u4"),
        ("pre_index", "<u4"),
        ("raw_count", "<u8"),
        ("source_row", "<u8"),
    ],
    align=False,
)
BUILD_CONFIGURATION = FULL_RETENTION_POLICY
ARTIFACT_BUILD_CONFIGURATION = {
    "boundary_role_body_ids": {role: list(body_ids) for role, body_ids in BOUNDARY_ROLE_BODY_IDS},
    "retention_policy": BUILD_CONFIGURATION,
}


class RuntimeBuildError(RuntimeError):
    """A source or transformation invariant blocked artifact construction."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    paths = (
        Path(__file__).resolve(),
        REPOSITORY_ROOT / "scripts/extract_malecns_fixture.py",
        REPOSITORY_ROOT / "packages/connectome-data/src/neurocraft_connectome/full_runtime.py",
    )
    for path in paths:
        raw = path.read_bytes()
        relative = path.relative_to(REPOSITORY_ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


@contextmanager
def _managed_memmap(path: Path, **kwargs: Any) -> Iterator[np.memmap]:
    """Release OS mappings even while a failing validation retains its traceback.

    Windows refuses removal or promotion of files with open mappings. Merely
    dropping a local reference leaves them open when an exception retains that
    frame, so scratch/build cleanup must close the mapping explicitly.
    """
    mapped = np.memmap(path, **kwargs)
    try:
        yield mapped
    finally:
        mapped._mmap.close()


def _source_receipts(manifest: Manifest) -> list[dict[str, Any]]:
    receipts = []
    for source_id in ("annotations", "neurotransmitters", "weights"):
        source = manifest.sources[source_id]
        if not isinstance(source.generation, str) or not source.generation:
            raise RuntimeBuildError(f"source {source_id!r} is not bound to a non-empty generation")
        receipts.append(
            {
                "bytes": source.expected_bytes,
                "filename": source.filename,
                "generation": source.generation,
                "id": source.source_id,
                "sha256": source.sha256,
                "url": source.url,
            }
        )
    return receipts


def _read_feather(path: Path, columns: list[str], label: str) -> pa.Table:
    try:
        table = feather.read_table(path, columns=columns, memory_map=True)
    except Exception as exc:
        raise RuntimeBuildError(f"cannot read {label} Feather table: {exc}") from exc
    if table.column_names != columns:
        raise RuntimeBuildError(f"{label} columns do not match required order {columns}")
    return table


def _annotation_selection(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, Any]]:
    table = _read_feather(path, ["bodyId", "superclass"], "annotations")
    body_column = table["bodyId"]
    superclass_column = table["superclass"]
    if not pa.types.is_int64(body_column.type) or not pa.types.is_string(superclass_column.type):
        raise RuntimeBuildError("annotations bodyId/superclass types must be int64/string")
    if body_column.null_count:
        raise RuntimeBuildError("annotations.bodyId contains null values")
    body_ids = body_column.combine_chunks().to_numpy(zero_copy_only=False)
    if body_ids.size == 0 or np.any(body_ids <= 0):
        raise RuntimeBuildError("annotations body IDs must be non-empty and positive")
    sorted_all = np.sort(body_ids.astype(np.uint64, copy=True), kind="quicksort")
    duplicate_offsets = np.flatnonzero(sorted_all[1:] == sorted_all[:-1])
    if duplicate_offsets.size:
        duplicate = int(sorted_all[int(duplicate_offsets[0])])
        raise RuntimeBuildError(f"duplicate annotation body ID {duplicate}")

    superclasses = superclass_column.to_pylist()
    retained_rows: list[int] = []
    retained_bodies: list[int] = []
    retained_superclasses: list[str] = []
    retained_by_superclass: Counter[str] = Counter()
    excluded_null = 0
    excluded_empty = 0
    for source_row, (body_id, superclass) in enumerate(zip(body_ids, superclasses, strict=True)):
        if superclass is None:
            excluded_null += 1
            continue
        if not isinstance(superclass, str):
            raise RuntimeBuildError(f"annotation row {source_row} superclass is not a string")
        if len(superclass) == 0:
            excluded_empty += 1
            continue
        retained_rows.append(source_row)
        retained_bodies.append(int(body_id))
        retained_superclasses.append(superclass)
        retained_by_superclass[superclass] += 1
    if not retained_bodies:
        raise RuntimeBuildError("retention policy selected zero nodes")
    order = np.argsort(np.asarray(retained_bodies, dtype=np.uint64), kind="stable")
    selected_bodies = np.asarray(retained_bodies, dtype="<u8")[order]
    selected_rows = np.asarray(retained_rows, dtype="<u8")[order]
    selected_superclasses = tuple(retained_superclasses[int(index)] for index in order)
    if selected_bodies.size > UINT32_MAX:
        raise RuntimeBuildError("retained node count exceeds uint32 dense-index capacity")
    stats = {
        "annotation_duplicate_body_ids": 0,
        "annotation_source_rows": int(table.num_rows),
        "excluded_superclass_empty_rows": excluded_empty,
        "excluded_superclass_null_rows": excluded_null,
        "retained_node_rows": int(selected_bodies.size),
        "retained_by_superclass": dict(sorted(retained_by_superclass.items())),
    }
    if retained_bodies and (
        len(retained_bodies) + excluded_null + excluded_empty != table.num_rows
    ):
        raise RuntimeBuildError("annotation attrition counts do not reconcile")
    return selected_bodies, selected_rows, selected_superclasses, stats


def _boundary_role_overlay(
    selected_bodies: np.ndarray,
    selected_rows: np.ndarray,
    selected_superclasses: tuple[str, ...],
) -> dict[str, Any]:
    expected_superclasses = dict(BOUNDARY_BODY_SUPERCLASSES)
    roles: dict[str, list[dict[str, Any]]] = {}
    for role, body_ids in BOUNDARY_ROLE_BODY_IDS:
        entries = []
        for body_id in body_ids:
            dense_index = int(np.searchsorted(selected_bodies, np.uint64(body_id)))
            if dense_index >= selected_bodies.size or int(selected_bodies[dense_index]) != body_id:
                raise RuntimeBuildError(
                    f"required boundary role {role!r} body {body_id} was not retained"
                )
            superclass = selected_superclasses[dense_index]
            if superclass != expected_superclasses[body_id]:
                raise RuntimeBuildError(
                    f"boundary body {body_id} superclass mismatch: expected "
                    f"{expected_superclasses[body_id]!r}, observed {superclass!r}"
                )
            entries.append(
                {
                    "annotation_source_row": int(selected_rows[dense_index]),
                    "body_id": body_id,
                    "dense_index": dense_index,
                    "superclass": superclass,
                }
            )
        roles[role] = entries
    authority = {
        "fixture_file_sha256": MALECNS_FIXTURE_V1_PIN.file_sha256,
        "fixture_id": MALECNS_FIXTURE_V1_PIN.fixture_id,
        "fixture_semantic_sha256": MALECNS_FIXTURE_V1_PIN.semantic_sha256,
    }
    overlay_payload = {"authority": authority, "roles": roles}
    return {
        **overlay_payload,
        "sha256": hashlib.sha256(canonical_json_bytes(overlay_payload)).hexdigest(),
    }


def _transmitter_lineage(
    path: Path, selected_bodies: np.ndarray
) -> tuple[np.ndarray, dict[str, int]]:
    table = _read_feather(path, ["body"], "neurotransmitters")
    body_column = table["body"]
    if not pa.types.is_int64(body_column.type) or body_column.null_count:
        raise RuntimeBuildError("neurotransmitters.body must be non-null int64")
    bodies = body_column.combine_chunks().to_numpy(zero_copy_only=False)
    if bodies.size == 0 or np.any(bodies <= 0):
        raise RuntimeBuildError("neurotransmitter body IDs must be non-empty and positive")
    order = np.argsort(bodies.astype(np.uint64, copy=False), kind="stable")
    sorted_bodies = bodies[order].astype(np.uint64, copy=False)
    duplicate_offsets = np.flatnonzero(sorted_bodies[1:] == sorted_bodies[:-1])
    if duplicate_offsets.size:
        duplicate = int(sorted_bodies[int(duplicate_offsets[0])])
        raise RuntimeBuildError(f"duplicate neurotransmitter body ID {duplicate}")
    positions = np.searchsorted(sorted_bodies, selected_bodies)
    within = positions < sorted_bodies.size
    matches = np.zeros(selected_bodies.shape, dtype=np.bool_)
    if np.any(within):
        matches[within] = sorted_bodies[positions[within]] == selected_bodies[within]
    lineage = np.full(selected_bodies.shape, UINT64_MAX, dtype="<u8")
    lineage[matches] = order[positions[matches]].astype(np.uint64, copy=False)
    matched = int(np.count_nonzero(matches))
    return lineage, {
        "excluded_transmitter_rows": int(table.num_rows) - matched,
        "retained_nodes_missing_transmitter_row": int(selected_bodies.size) - matched,
        "retained_nodes_with_transmitter_row": matched,
        "transmitter_duplicate_body_ids": 0,
        "transmitter_source_rows": int(table.num_rows),
    }


def _retained_indices(
    values: np.ndarray, selected_bodies: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(selected_bodies, values.astype(np.uint64, copy=False))
    within = positions < selected_bodies.size
    matches = np.zeros(values.shape, dtype=np.bool_)
    if np.any(within):
        matches[within] = selected_bodies[positions[within]] == values[within]
    return positions, matches


def _first_duplicate_pair(sorted_keys: np.memmap, *, packed_uint32: bool) -> tuple[int, int] | None:
    previous: tuple[int, int] | None = None
    for start in range(0, sorted_keys.shape[0], 1_000_000):
        block = np.asarray(sorted_keys[start : start + 1_000_000])
        if block.size == 0:
            continue
        if packed_uint32:
            first_key = int(block[0])
            first = (first_key >> 32, first_key & int(UINT32_MAX))
            duplicate_offsets = np.flatnonzero(block[1:] == block[:-1])
            last_key = int(block[-1])
            last = (last_key >> 32, last_key & int(UINT32_MAX))
        else:
            first = (int(block["body_pre"][0]), int(block["body_post"][0]))
            duplicate_offsets = np.flatnonzero(
                (block["body_pre"][1:] == block["body_pre"][:-1])
                & (block["body_post"][1:] == block["body_post"][:-1])
            )
            last = (int(block["body_pre"][-1]), int(block["body_post"][-1]))
        if previous is not None and previous == first:
            return previous
        if duplicate_offsets.size:
            index = int(duplicate_offsets[0])
            if packed_uint32:
                key = int(block[index])
                return key >> 32, key & int(UINT32_MAX)
            return int(block["body_pre"][index]), int(block["body_post"][index])
        previous = last
    return None


def _exact_positive_uint64_sum(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    maximum = int(values.max())
    if maximum <= int(UINT64_MAX) // int(values.size):
        return int(values.sum(dtype=np.uint64))
    total = sum(int(value) for value in values)
    if total > int(UINT64_MAX):
        raise RuntimeBuildError("source weight sum exceeds uint64")
    return total


def _edge_selection(
    path: Path,
    selected_bodies: np.ndarray,
    scratch_root: Path,
) -> tuple[Path, dict[str, int]]:
    table = _read_feather(path, ["body_pre", "body_post", "weight"], "weights")
    for name in table.column_names:
        column = table[name]
        if not pa.types.is_int64(column.type) or column.null_count:
            raise RuntimeBuildError(f"weights.{name} must be non-null int64")
    if table.num_rows == 0:
        raise RuntimeBuildError("weights table is empty")

    largest_body_id = max(
        int(pc.max(table["body_pre"]).as_py()),
        int(pc.max(table["body_post"]).as_py()),
    )
    packed_uint32 = largest_body_id <= int(UINT32_MAX)
    keys_path = scratch_root / "all-directed-pairs.bin"
    records_path = scratch_root / "retained-edge-records.bin"
    keys = np.memmap(
        keys_path,
        dtype="<u8" if packed_uint32 else PAIR_KEY_DTYPE,
        mode="w+",
        shape=(table.num_rows,),
    )
    counts: dict[str, int] = {
        "raw_edge_rows": int(table.num_rows),
        "raw_weight_sum": 0,
        "raw_self_edge_rows": 0,
        "raw_self_edge_weight_sum": 0,
        "raw_duplicate_directed_pairs": 0,
        "retained_edge_rows": 0,
        "retained_weight_sum": 0,
        "retained_self_edge_rows": 0,
        "retained_self_edge_weight_sum": 0,
        "excluded_edge_rows": 0,
        "excluded_weight_sum": 0,
        "excluded_pre_only_rows": 0,
        "excluded_post_only_rows": 0,
        "excluded_neither_endpoint_rows": 0,
        "excluded_self_edge_rows": 0,
        "excluded_self_edge_weight_sum": 0,
    }
    raw_weight_min: int | None = None
    raw_weight_max: int | None = None
    retained_weight_min: int | None = None
    retained_weight_max: int | None = None
    offset = 0
    try:
        with records_path.open("xb") as records_handle:
            for batch in table.to_batches(max_chunksize=262_144):
                pre = batch.column(0).to_numpy(zero_copy_only=False)
                post = batch.column(1).to_numpy(zero_copy_only=False)
                weight = batch.column(2).to_numpy(zero_copy_only=False)
                if np.any(pre <= 0) or np.any(post <= 0):
                    raise RuntimeBuildError(
                        f"weights rows {offset}:{offset + batch.num_rows} contain non-positive IDs"
                    )
                if np.any(weight <= 0):
                    raise RuntimeBuildError(
                        f"weights rows {offset}:{offset + batch.num_rows} contain "
                        "non-positive weights"
                    )
                pre_u64 = pre.astype(np.uint64, copy=False)
                post_u64 = post.astype(np.uint64, copy=False)
                if packed_uint32:
                    keys[offset : offset + batch.num_rows] = (pre_u64 << np.uint64(32)) | post_u64
                else:
                    keys["body_pre"][offset : offset + batch.num_rows] = pre_u64
                    keys["body_post"][offset : offset + batch.num_rows] = post_u64

                batch_weight_sum = _exact_positive_uint64_sum(weight)
                counts["raw_weight_sum"] += batch_weight_sum
                if counts["raw_weight_sum"] > int(UINT64_MAX):
                    raise RuntimeBuildError("source weight sum exceeds uint64")
                self_edges = pre == post
                counts["raw_self_edge_rows"] += int(np.count_nonzero(self_edges))
                counts["raw_self_edge_weight_sum"] += _exact_positive_uint64_sum(weight[self_edges])
                local_min = int(weight.min())
                local_max = int(weight.max())
                raw_weight_min = (
                    local_min if raw_weight_min is None else min(raw_weight_min, local_min)
                )
                raw_weight_max = (
                    local_max if raw_weight_max is None else max(raw_weight_max, local_max)
                )

                pre_positions, pre_matches = _retained_indices(pre, selected_bodies)
                post_positions, post_matches = _retained_indices(post, selected_bodies)
                retained = pre_matches & post_matches
                pre_only = pre_matches & ~post_matches
                post_only = ~pre_matches & post_matches
                neither = ~pre_matches & ~post_matches
                retained_count = int(np.count_nonzero(retained))
                excluded_count = batch.num_rows - retained_count
                counts["retained_edge_rows"] += retained_count
                counts["excluded_edge_rows"] += excluded_count
                counts["excluded_pre_only_rows"] += int(np.count_nonzero(pre_only))
                counts["excluded_post_only_rows"] += int(np.count_nonzero(post_only))
                counts["excluded_neither_endpoint_rows"] += int(np.count_nonzero(neither))
                if retained_count:
                    retained_weights = weight[retained]
                    retained_sum = _exact_positive_uint64_sum(retained_weights)
                    counts["retained_weight_sum"] += retained_sum
                    retained_self_edges = self_edges & retained
                    counts["retained_self_edge_rows"] += int(np.count_nonzero(retained_self_edges))
                    counts["retained_self_edge_weight_sum"] += _exact_positive_uint64_sum(
                        weight[retained_self_edges]
                    )
                    local_retained_min = int(retained_weights.min())
                    local_retained_max = int(retained_weights.max())
                    retained_weight_min = (
                        local_retained_min
                        if retained_weight_min is None
                        else min(retained_weight_min, local_retained_min)
                    )
                    retained_weight_max = (
                        local_retained_max
                        if retained_weight_max is None
                        else max(retained_weight_max, local_retained_max)
                    )
                    local_rows = np.flatnonzero(retained).astype(np.uint64, copy=False)
                    records = np.empty(retained_count, dtype=EDGE_RECORD_DTYPE)
                    records["post_index"] = post_positions[retained].astype(np.uint32, copy=False)
                    records["pre_index"] = pre_positions[retained].astype(np.uint32, copy=False)
                    records["raw_count"] = retained_weights.astype(np.uint64, copy=False)
                    records["source_row"] = local_rows + np.uint64(offset)
                    records.tofile(records_handle)
                if excluded_count:
                    counts["excluded_weight_sum"] += _exact_positive_uint64_sum(weight[~retained])
                    excluded_self_edges = self_edges & ~retained
                    counts["excluded_self_edge_rows"] += int(np.count_nonzero(excluded_self_edges))
                    counts["excluded_self_edge_weight_sum"] += _exact_positive_uint64_sum(
                        weight[excluded_self_edges]
                    )
                offset += batch.num_rows
            records_handle.flush()
            os.fsync(records_handle.fileno())
        keys.flush()
        if packed_uint32:
            keys.sort(kind="quicksort")
        else:
            keys.sort(order=("body_pre", "body_post"), kind="quicksort")
        duplicate_pair = _first_duplicate_pair(keys, packed_uint32=packed_uint32)
    finally:
        keys._mmap.close()
        del keys
    if duplicate_pair is not None:
        body_pre, body_post = duplicate_pair
        raise RuntimeBuildError(
            f"duplicate directed source pair {body_pre}->{body_post}; aggregation is forbidden"
        )
    keys_path.unlink()
    if offset != table.num_rows:
        raise RuntimeBuildError("weight source row accounting does not reconcile")
    if counts["retained_edge_rows"] == 0:
        raise RuntimeBuildError("retention policy selected zero edges")
    if counts["retained_edge_rows"] + counts["excluded_edge_rows"] != table.num_rows:
        raise RuntimeBuildError("edge attrition row counts do not reconcile")
    if counts["retained_weight_sum"] + counts["excluded_weight_sum"] != counts["raw_weight_sum"]:
        raise RuntimeBuildError("edge attrition weight sums do not reconcile")
    if (
        counts["excluded_pre_only_rows"]
        + counts["excluded_post_only_rows"]
        + counts["excluded_neither_endpoint_rows"]
        != counts["excluded_edge_rows"]
    ):
        raise RuntimeBuildError("endpoint exclusion counts do not reconcile")
    if (
        counts["retained_self_edge_rows"] + counts["excluded_self_edge_rows"]
        != counts["raw_self_edge_rows"]
    ):
        raise RuntimeBuildError("self-edge row counts do not reconcile")
    if (
        counts["retained_self_edge_weight_sum"] + counts["excluded_self_edge_weight_sum"]
        != counts["raw_self_edge_weight_sum"]
    ):
        raise RuntimeBuildError("self-edge weight sums do not reconcile")
    counts.update(
        {
            "raw_weight_min": int(raw_weight_min or 0),
            "raw_weight_max": int(raw_weight_max or 0),
            "retained_weight_min": int(retained_weight_min or 0),
            "retained_weight_max": int(retained_weight_max or 0),
            "retained_duplicate_directed_pairs": 0,
        }
    )
    return records_path, counts


def _array_chunks(values: np.ndarray, *, size: int = 1_000_000) -> Iterator[np.ndarray]:
    for start in range(0, values.shape[0], size):
        yield np.asarray(values[start : start + size])


def _write_array(
    output_root: Path,
    name: str,
    values: np.ndarray,
    *,
    dtype: str,
) -> dict[str, Any]:
    canonical_dtype = np.dtype(dtype)
    path = output_root / f"{name}.bin"
    digest = hashlib.sha256()
    byte_length = 0
    with path.open("xb") as handle:
        for chunk in _array_chunks(values):
            encoded = np.asarray(chunk, dtype=canonical_dtype).tobytes(order="C")
            handle.write(encoded)
            digest.update(encoded)
            byte_length += len(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    expected_bytes = int(values.shape[0]) * canonical_dtype.itemsize
    if byte_length != expected_bytes:
        raise RuntimeBuildError(f"array {name} byte count does not reconcile")
    dtype_label = "uint32-le" if canonical_dtype.itemsize == 4 else "uint64-le"
    return {
        "bytes": byte_length,
        "count": int(values.shape[0]),
        "dtype": dtype_label,
        "filename": path.name,
        "sha256": digest.hexdigest(),
    }


def _sort_and_write_edges(
    records_path: Path,
    edge_count: int,
    node_count: int,
    output_root: Path,
) -> dict[str, dict[str, Any]]:
    records = np.memmap(
        records_path,
        dtype=EDGE_RECORD_DTYPE,
        mode="r+",
        shape=(edge_count,),
    )
    try:
        records.sort(order=("post_index", "pre_index"), kind="quicksort")
        records.flush()
        duplicate = np.flatnonzero(
            (records["post_index"][1:] == records["post_index"][:-1])
            & (records["pre_index"][1:] == records["pre_index"][:-1])
        )
        if duplicate.size:
            raise RuntimeBuildError("retained edge sort exposed a duplicate directed pair")
        if (
            int(records["pre_index"].max()) >= node_count
            or int(records["post_index"].max()) >= node_count
        ):
            raise RuntimeBuildError("retained dense edge index is out of range")
        incoming_counts = np.bincount(records["post_index"], minlength=node_count).astype(
            "<u8", copy=False
        )
        offsets = np.empty(node_count + 1, dtype="<u8")
        offsets[0] = 0
        np.cumsum(incoming_counts, dtype=np.uint64, out=offsets[1:])
        arrays = {
            "edge_post_indices": _write_array(
                output_root,
                "edge_post_indices",
                records["post_index"],
                dtype="<u4",
            ),
            "edge_pre_indices": _write_array(
                output_root,
                "edge_pre_indices",
                records["pre_index"],
                dtype="<u4",
            ),
            "edge_raw_counts": _write_array(
                output_root,
                "edge_raw_counts",
                records["raw_count"],
                dtype="<u8",
            ),
            "edge_source_rows": _write_array(
                output_root,
                "edge_source_rows",
                records["source_row"],
                dtype="<u8",
            ),
            "incoming_offsets": _write_array(
                output_root,
                "incoming_offsets",
                offsets,
                dtype="<u8",
            ),
        }
    finally:
        records._mmap.close()
        del records
    return arrays


def _write_manifest(root: Path, payload: dict[str, Any]) -> tuple[str, str]:
    payload_sha256 = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    document = {
        **payload,
        "hashes": {
            "payload_scope": "canonical JSON document with hashes object omitted",
            "payload_sha256": payload_sha256,
        },
    }
    raw = canonical_json_bytes(document) + b"\n"
    path = root / "manifest.json"
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return payload_sha256, hashlib.sha256(raw).hexdigest()


def _fsync_directory(path: Path) -> None:
    # Python's Windows os.open cannot open a directory for fsync. Every artifact
    # file was already flushed/fsynced, and same-parent os.replace still provides
    # atomic visibility there; POSIX additionally syncs the directory entries.
    if sys.platform == "win32":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_runtime_artifact(
    manifest_path: Path,
    cache_dir: Path,
    output_dir: Path,
    *,
    scratch_dir: Path | None = None,
    expected_manifest_sha256: str = EXPECTED_MANIFEST_SHA256,
    progress: bool = False,
) -> dict[str, Any]:
    """Verify sources and atomically create one cold full runtime artifact."""

    started = time.perf_counter()
    if output_dir.exists() or output_dir.is_symlink():
        raise RuntimeBuildError(f"refusing to replace existing output {output_dir}")
    try:
        manifest = load_manifest(manifest_path, expected_sha256=expected_manifest_sha256)
        if manifest.dataset.get("id") != DATASET_ID:
            raise RuntimeBuildError("source manifest dataset identity is not male-cns:v1.0")
        if progress:
            print("[1/4] verifying pinned MaleCNS source objects", file=sys.stderr, flush=True)
        source_paths = verify_sources(manifest, cache_dir)
    except FixtureExtractionError as exc:
        raise RuntimeBuildError(str(exc)) from exc
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    build_root = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    scratch_parent = scratch_dir
    if scratch_parent is not None:
        scratch_parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix="neurocraft-full-runtime-", dir=scratch_parent
        ) as scratch_name:
            scratch_root = Path(scratch_name)
            if progress:
                print(
                    "[2/4] applying the exact superclass node-retention policy",
                    file=sys.stderr,
                    flush=True,
                )
            (
                selected_bodies,
                annotation_rows,
                selected_superclasses,
                annotation_stats,
            ) = _annotation_selection(source_paths["annotations"])
            boundary_role_overlay = _boundary_role_overlay(
                selected_bodies, annotation_rows, selected_superclasses
            )
            transmitter_rows, transmitter_stats = _transmitter_lineage(
                source_paths["neurotransmitters"], selected_bodies
            )
            if progress:
                print(
                    "[3/4] scanning, retaining, and target-major sorting source edges",
                    file=sys.stderr,
                    flush=True,
                )
            records_path, edge_stats = _edge_selection(
                source_paths["weights"], selected_bodies, scratch_root
            )
            arrays = {
                "body_ids": _write_array(build_root, "body_ids", selected_bodies, dtype="<u8"),
                "annotation_source_rows": _write_array(
                    build_root,
                    "annotation_source_rows",
                    annotation_rows,
                    dtype="<u8",
                ),
                "transmitter_source_rows": _write_array(
                    build_root,
                    "transmitter_source_rows",
                    transmitter_rows,
                    dtype="<u8",
                ),
            }
            arrays.update(
                _sort_and_write_edges(
                    records_path,
                    edge_stats["retained_edge_rows"],
                    int(selected_bodies.size),
                    build_root,
                )
            )
        counts = {
            key: int(value)
            for key, value in {
                **annotation_stats,
                **transmitter_stats,
                **edge_stats,
            }.items()
            if key != "retained_by_superclass"
        }
        structural_graph_sha256 = hashlib.sha256(
            canonical_json_bytes(
                {
                    "artifact_id": FULL_ARTIFACT_ID,
                    "body_ids": arrays["body_ids"]["sha256"],
                    "edge_post_indices": arrays["edge_post_indices"]["sha256"],
                    "edge_pre_indices": arrays["edge_pre_indices"]["sha256"],
                    "edge_raw_counts": arrays["edge_raw_counts"]["sha256"],
                    "retention_policy": BUILD_CONFIGURATION,
                }
            )
        ).hexdigest()
        provenance_graph_sha256 = hashlib.sha256(
            canonical_json_bytes(
                {
                    "annotation_source_rows": arrays["annotation_source_rows"]["sha256"],
                    "boundary_role_overlay": boundary_role_overlay["sha256"],
                    "edge_source_rows": arrays["edge_source_rows"]["sha256"],
                    "source_manifest_sha256": manifest.raw_sha256,
                    "structural_graph_sha256": structural_graph_sha256,
                    "transmitter_source_rows": arrays["transmitter_source_rows"]["sha256"],
                }
            )
        ).hexdigest()
        payload = {
            "array_encoding": "packed little-endian unsigned integers; no header",
            "arrays": dict(sorted(arrays.items())),
            "artifact_id": FULL_ARTIFACT_ID,
            "boundary_role_overlay": boundary_role_overlay,
            "builder_configuration_sha256": hashlib.sha256(
                canonical_json_bytes(ARTIFACT_BUILD_CONFIGURATION)
            ).hexdigest(),
            "builder_implementation_sha256": _implementation_sha256(),
            "counts": dict(sorted(counts.items())),
            "dataset": manifest.dataset,
            "dataset_id": DATASET_ID,
            "edge_count": edge_stats["retained_edge_rows"],
            "node_count": int(selected_bodies.size),
            "provenance_graph_sha256": provenance_graph_sha256,
            "raw_count_sum": edge_stats["retained_weight_sum"],
            "retained_by_superclass": annotation_stats["retained_by_superclass"],
            "retention_policy": BUILD_CONFIGURATION,
            "schema_version": SCHEMA_VERSION,
            "source_manifest_sha256": manifest.raw_sha256,
            "sources": _source_receipts(manifest),
            "structural_graph_sha256": structural_graph_sha256,
        }
        payload_sha256, manifest_file_sha256 = _write_manifest(build_root, payload)
        if progress:
            print(
                "[4/4] exhaustively validating arrays before atomic promotion",
                file=sys.stderr,
                flush=True,
            )
        _verify_numpy_layout(build_root, payload)
        _fsync_directory(build_root)
        os.replace(build_root, output_dir)
        _fsync_directory(output_dir.parent)
        return {
            "artifact_dir": str(output_dir),
            "artifact_id": FULL_ARTIFACT_ID,
            "build_seconds": time.perf_counter() - started,
            "counts": counts,
            "manifest_file_sha256": manifest_file_sha256,
            "payload_sha256": payload_sha256,
            "provenance_graph_sha256": provenance_graph_sha256,
            "retained_by_superclass": annotation_stats["retained_by_superclass"],
            "structural_graph_sha256": structural_graph_sha256,
        }
    except BaseException:
        if build_root.exists():
            shutil.rmtree(build_root)
        raise


def _verify_numpy_layout(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    arrays = payload["arrays"]
    node_count = int(payload["node_count"])
    edge_count = int(payload["edge_count"])
    with ExitStack() as mappings:
        def mapped(name: str, dtype: str) -> np.memmap:
            return mappings.enter_context(
                _managed_memmap(root / arrays[name]["filename"], dtype=dtype, mode="r")
            )

        body_ids = mapped("body_ids", "<u8")
        annotation_rows = mapped("annotation_source_rows", "<u8")
        transmitter_rows = mapped("transmitter_source_rows", "<u8")
        pre = mapped("edge_pre_indices", "<u4")
        post = mapped("edge_post_indices", "<u4")
        weight = mapped("edge_raw_counts", "<u8")
        source_rows = mapped("edge_source_rows", "<u8")
        offsets = mapped("incoming_offsets", "<u8")
        if not (
            body_ids.size == annotation_rows.size == transmitter_rows.size == node_count
            and pre.size == post.size == weight.size == source_rows.size == edge_count
            and offsets.size == node_count + 1
        ):
            raise RuntimeBuildError("written array shapes do not reconcile")
        if np.any(body_ids[1:] <= body_ids[:-1]):
            raise RuntimeBuildError("written body IDs are not strictly ascending")
        if np.any(annotation_rows >= payload["counts"]["annotation_source_rows"]):
            raise RuntimeBuildError("annotation lineage row is out of source range")
        if np.unique(annotation_rows).size != annotation_rows.size:
            raise RuntimeBuildError("annotation lineage rows are not unique")
        matched_transmitter_rows = transmitter_rows[transmitter_rows != UINT64_MAX]
        if np.any(matched_transmitter_rows >= payload["counts"]["transmitter_source_rows"]):
            raise RuntimeBuildError("transmitter lineage row is out of source range")
        if np.unique(matched_transmitter_rows).size != matched_transmitter_rows.size:
            raise RuntimeBuildError("transmitter lineage rows are not unique")
        if matched_transmitter_rows.size != payload["counts"]["retained_nodes_with_transmitter_row"]:
            raise RuntimeBuildError("transmitter lineage counts do not reconcile")
        if int(pre.max()) >= node_count or int(post.max()) >= node_count:
            raise RuntimeBuildError("written dense edge index is out of range")
        ordered = (post[1:] > post[:-1]) | ((post[1:] == post[:-1]) & (pre[1:] > pre[:-1]))
        if not bool(np.all(ordered)):
            raise RuntimeBuildError("written edges are not unique target-major pairs")
        if np.any(weight == 0):
            raise RuntimeBuildError("written edge count contains zero")
        if np.any(source_rows >= payload["counts"]["raw_edge_rows"]):
            raise RuntimeBuildError("edge lineage row is out of source range")
        if np.unique(source_rows).size != source_rows.size:
            raise RuntimeBuildError("edge lineage rows are not unique")
        observed_weight_sum = int(weight.sum(dtype=np.uint64))
        if observed_weight_sum != payload["raw_count_sum"]:
            raise RuntimeBuildError("written edge weight sum does not reconcile")
        if int(offsets[0]) != 0 or int(offsets[-1]) != edge_count or np.any(offsets[1:] < offsets[:-1]):
            raise RuntimeBuildError("written incoming offsets are invalid")
        observed_degrees = np.bincount(post, minlength=node_count).astype(np.uint64)
        if not np.array_equal(np.diff(offsets), observed_degrees):
            raise RuntimeBuildError("written incoming offsets do not match edge targets")
        return {
            "body_id_max": int(body_ids[-1]),
            "body_id_min": int(body_ids[0]),
            "edge_source_row_max": int(source_rows.max()),
            "edge_source_row_min": int(source_rows.min()),
            "incoming_degree_max": int(observed_degrees.max()),
            "weight_sum": observed_weight_sum,
        }


def _deterministic_sample_positions(
    count: int, desired: int, *, identity: str, domain: str
) -> np.ndarray:
    sample_count = min(count, desired)
    selected: set[int] = set()
    counter = 0
    while len(selected) < sample_count:
        block = hashlib.sha256(f"{identity}:{domain}:{counter}".encode("ascii")).digest()
        selected.add(int.from_bytes(block[:8], "big") % count)
        counter += 1
    return np.asarray(sorted(selected), dtype=np.int64)


def source_spot_audit(
    artifact_dir: Path,
    manifest_path: Path,
    cache_dir: Path,
    *,
    expected_manifest_sha256: str = EXPECTED_MANIFEST_SHA256,
) -> dict[str, Any]:
    """Bind deterministic runtime samples back to verified Feather source rows."""

    try:
        source_manifest = load_manifest(manifest_path, expected_sha256=expected_manifest_sha256)
        source_paths = verify_sources(source_manifest, cache_dir)
    except FixtureExtractionError as exc:
        raise RuntimeBuildError(str(exc)) from exc
    artifact_manifest = json.loads((artifact_dir / "manifest.json").read_text())
    identity = str(artifact_manifest["structural_graph_sha256"])
    arrays = artifact_manifest["arrays"]
    body_ids = np.memmap(artifact_dir / arrays["body_ids"]["filename"], dtype="<u8", mode="r")
    annotation_rows = np.memmap(
        artifact_dir / arrays["annotation_source_rows"]["filename"],
        dtype="<u8",
        mode="r",
    )
    transmitter_rows = np.memmap(
        artifact_dir / arrays["transmitter_source_rows"]["filename"],
        dtype="<u8",
        mode="r",
    )
    pre = np.memmap(artifact_dir / arrays["edge_pre_indices"]["filename"], dtype="<u4", mode="r")
    post = np.memmap(
        artifact_dir / arrays["edge_post_indices"]["filename"],
        dtype="<u4",
        mode="r",
    )
    weight = np.memmap(artifact_dir / arrays["edge_raw_counts"]["filename"], dtype="<u8", mode="r")
    edge_rows = np.memmap(
        artifact_dir / arrays["edge_source_rows"]["filename"], dtype="<u8", mode="r"
    )
    node_positions = _deterministic_sample_positions(
        body_ids.size, 64, identity=identity, domain="nodes"
    )
    edge_positions = _deterministic_sample_positions(
        pre.size, 256, identity=identity, domain="edges"
    )

    annotations = _read_feather(
        source_paths["annotations"], ["bodyId", "superclass"], "annotations"
    ).take(pa.array(np.asarray(annotation_rows[node_positions], dtype=np.int64)))
    transmitters = _read_feather(source_paths["neurotransmitters"], ["body"], "neurotransmitters")
    weights = _read_feather(
        source_paths["weights"], ["body_pre", "body_post", "weight"], "weights"
    ).take(pa.array(np.asarray(edge_rows[edge_positions], dtype=np.int64)))
    audit_rows: list[dict[str, int | str]] = []
    for sample_index, runtime_index in enumerate(node_positions):
        source_body_id = annotations["bodyId"][sample_index].as_py()
        superclass = annotations["superclass"][sample_index].as_py()
        if source_body_id != int(body_ids[runtime_index]):
            raise RuntimeBuildError("annotation lineage spot audit found a body-ID mismatch")
        if not isinstance(superclass, str) or not superclass:
            raise RuntimeBuildError("annotation lineage spot audit found a non-retained superclass")
        audit_rows.append(
            {
                "body_id": int(source_body_id),
                "kind": "node",
                "runtime_index": int(runtime_index),
                "source_row": int(annotation_rows[runtime_index]),
                "transmitter_source_row": int(transmitter_rows[runtime_index]),
            }
        )
        transmitter_source_row = int(transmitter_rows[runtime_index])
        if transmitter_source_row != int(UINT64_MAX) and (
            transmitters["body"][transmitter_source_row].as_py() != source_body_id
        ):
            raise RuntimeBuildError("transmitter lineage spot audit found a body-ID mismatch")
    for sample_index, runtime_index in enumerate(edge_positions):
        source_pre = weights["body_pre"][sample_index].as_py()
        source_post = weights["body_post"][sample_index].as_py()
        source_weight = weights["weight"][sample_index].as_py()
        expected_pre = int(body_ids[int(pre[runtime_index])])
        expected_post = int(body_ids[int(post[runtime_index])])
        if (source_pre, source_post, source_weight) != (
            expected_pre,
            expected_post,
            int(weight[runtime_index]),
        ):
            raise RuntimeBuildError("edge lineage spot audit found a source-row mismatch")
        audit_rows.append(
            {
                "body_post": expected_post,
                "body_pre": expected_pre,
                "kind": "edge",
                "raw_count": int(source_weight),
                "runtime_index": int(runtime_index),
                "source_row": int(edge_rows[runtime_index]),
            }
        )
    return {
        "audit_sha256": hashlib.sha256(canonical_json_bytes(audit_rows)).hexdigest(),
        "edge_samples": int(edge_positions.size),
        "node_samples": int(node_positions.size),
        "sampling": "SHA-256-derived positions bound to structural graph identity",
        "source_manifest_sha256": source_manifest.raw_sha256,
        "status": "matched",
    }


def benchmark_artifact(artifact_dir: Path, manifest_file_sha256: str) -> dict[str, Any]:
    load_started = time.perf_counter()
    with load_full_runtime_artifact(
        artifact_dir,
        expected_manifest_sha256=manifest_file_sha256,
    ) as graph:
        verified_load_seconds = time.perf_counter() - load_started
        payload = json.loads((artifact_dir / "manifest.json").read_text())
        layout_started = time.perf_counter()
        layout = _verify_numpy_layout(artifact_dir, payload)
        layout_seconds = time.perf_counter() - layout_started
        return {
            "edge_count": graph.manifest.edge_count,
            "layout": layout,
            "numpy_layout_verification_seconds": layout_seconds,
            "node_count": graph.manifest.node_count,
            "verified_hash_load_seconds": verified_load_seconds,
        }


def unsigned_neural_smoke(
    artifact_dir: Path,
    *,
    steps: int,
) -> dict[str, Any]:
    """Run an explicitly unsigned engineering smoke over every retained edge."""

    if steps < 3:
        raise RuntimeBuildError("full-graph neural smoke needs at least three steps")
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    arrays = manifest["arrays"]
    body_ids = np.memmap(artifact_dir / arrays["body_ids"]["filename"], dtype="<u8", mode="r")
    pre = np.memmap(artifact_dir / arrays["edge_pre_indices"]["filename"], dtype="<u4", mode="r")
    post = np.memmap(artifact_dir / arrays["edge_post_indices"]["filename"], dtype="<u4", mode="r")
    counts = np.memmap(artifact_dir / arrays["edge_raw_counts"]["filename"], dtype="<u8", mode="r")
    offsets = np.memmap(
        artifact_dir / arrays["incoming_offsets"]["filename"], dtype="<u8", mode="r"
    )
    cumulative = np.empty(counts.size + 1, dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(counts, dtype=np.float64, out=cumulative[1:])
    denominators = cumulative[offsets[1:]] - cumulative[offsets[:-1]]
    denominators[denominators == 0] = 1.0
    probe_indices = np.asarray([0, body_ids.size - 1], dtype=np.int64)
    probe_body_ids = tuple(int(body_ids[index]) for index in probe_indices)
    activity = np.zeros(body_ids.size, dtype=np.float64)
    timings: list[float] = []
    for _ in range(steps):
        started = time.perf_counter()
        recurrent = np.zeros_like(activity)
        for start in range(0, pre.size, 1_000_000):
            stop = min(pre.size, start + 1_000_000)
            local_pre = np.asarray(pre[start:stop])
            local_post = np.asarray(post[start:stop])
            contribution = (
                np.asarray(counts[start:stop], dtype=np.float64)
                * activity[local_pre]
                / denominators[local_post]
            )
            recurrent += np.bincount(local_post, weights=contribution, minlength=activity.size)
        drive = 0.9 * recurrent
        drive[probe_indices[0]] += 1.0
        drive[probe_indices[1]] += 0.5
        next_activity = math.exp(-10.0 / 50.0) * activity + (
            1.0 - math.exp(-10.0 / 50.0)
        ) * np.tanh(drive)
        activity = next_activity
        timings.append(time.perf_counter() - started)
    return {
        "activity_max": float(activity.max()),
        "activity_min": float(activity.min()),
        "final_activity_sha256": hashlib.sha256(
            np.asarray(activity, dtype="<f8").tobytes(order="C")
        ).hexdigest(),
        "mode": "unsigned_engineering_smoke_not_faithful",
        "nonzero_activity_count": int(np.count_nonzero(activity)),
        "probe_body_ids": probe_body_ids,
        "step_seconds": timings,
        "steps": steps,
        "uses_all_retained_edges": True,
    }


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    raw = canonical_json_bytes(receipt) + b"\n"
    with temporary.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _peak_rss_bytes() -> int | None:
    if resource is None:
        return None
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return observed if sys.platform == "darwin" else observed * 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the full MaleCNS v1.0 superclass-retained runtime graph"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scratch-dir", type=Path, default=None)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--smoke-steps", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        build = build_runtime_artifact(
            args.manifest,
            args.cache_dir,
            args.output,
            scratch_dir=args.scratch_dir,
            progress=True,
        )
        print("[post-build] verifying mapped load and full array layout", file=sys.stderr)
        benchmark = benchmark_artifact(args.output, build["manifest_file_sha256"])
        print("[post-build] auditing deterministic samples to source rows", file=sys.stderr)
        spot_audit = source_spot_audit(
            args.output,
            args.manifest,
            args.cache_dir,
        )
        print("[post-build] executing unsigned all-edge neural smoke", file=sys.stderr)
        smoke = unsigned_neural_smoke(args.output, steps=args.smoke_steps)
    except RuntimeBuildError as exc:
        print(f"full runtime build failed: {exc}", file=sys.stderr)
        return 2
    receipt = {
        "artifact": build,
        "benchmark": benchmark,
        "builder_implementation_sha256": _implementation_sha256(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "environment": {
            "machine": platform.machine(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "pyarrow": pa.__version__,
            "python": platform.python_version(),
            "peak_rss_bytes": _peak_rss_bytes(),
        },
        "neural_smoke": smoke,
        "source_spot_audit": spot_audit,
        "status": "built_verified_and_smoked",
    }
    _write_receipt(args.receipt, receipt)
    print(
        json.dumps(
            {
                "artifact": str(args.output),
                "edge_count": build["counts"]["retained_edge_rows"],
                "manifest_file_sha256": build["manifest_file_sha256"],
                "node_count": build["counts"]["retained_node_rows"],
                "receipt": str(args.receipt),
                "status": receipt["status"],
                "structural_graph_sha256": build["structural_graph_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
