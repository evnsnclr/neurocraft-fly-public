from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path, PureWindowsPath
from typing import Any

import pyarrow as pa
import pyarrow.feather as feather
import pytest
from neurocraft_connectome import (
    FullArtifactIdentityError,
    SourceArtifact,
    load_full_runtime_artifact,
)
from neurocraft_connectome.model import canonical_json_bytes

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import build_malecns_runtime as builder  # noqa: E402

ANNOTATIONS_FILENAME = "annotations.feather"
TRANSMITTERS_FILENAME = "neurotransmitters.feather"
WEIGHTS_FILENAME = "weights.feather"
LARGE_BODY = 2**40 + 4
LARGER_BODY = 2**40 + 5


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_annotations() -> list[dict[str, Any]]:
    return [
        {"bodyId": 1, "superclass": "sensory"},
        {"bodyId": 2, "superclass": None},
        {"bodyId": 3, "superclass": ""},
        {"bodyId": LARGE_BODY, "superclass": "descending"},
        {"bodyId": LARGER_BODY, "superclass": "motor"},
        {"bodyId": 77185, "superclass": "cb_sensory"},
        {"bodyId": 800241, "superclass": "vnc_motor"},
        {"bodyId": 800917, "superclass": "vnc_motor"},
        {"bodyId": 805722, "superclass": "vnc_sensory"},
    ]


def _base_weights() -> list[tuple[int, int, int | None]]:
    return [
        (1, LARGE_BODY, 5),
        (LARGE_BODY, LARGER_BODY, 7),
        (LARGER_BODY, LARGER_BODY, 2),
        (1, 2, 11),
        (2, LARGE_BODY, 13),
        (2, 3, 17),
        (2, 2, 19),
    ]


class SyntheticFullSource:
    def __init__(
        self,
        root: Path,
        *,
        annotations: list[dict[str, Any]] | None = None,
        weights: list[tuple[int, int, int | None]] | None = None,
        transmitter_bodies: list[int | None] | None = None,
    ) -> None:
        self.root = root
        self.cache = root / "cache"
        self.cache.mkdir(parents=True)
        self.manifest_path = root / "male-cns-v1.0.sources.json"
        self.annotations = annotations or _base_annotations()
        self.weights = weights or _base_weights()
        self.transmitter_bodies = [1] if transmitter_bodies is None else transmitter_bodies
        self._write_sources()

    def _write_sources(self) -> None:
        feather.write_feather(
            pa.Table.from_pylist(
                self.annotations,
                schema=pa.schema([("bodyId", pa.int64()), ("superclass", pa.string())]),
            ),
            self.cache / ANNOTATIONS_FILENAME,
            compression="uncompressed",
        )
        feather.write_feather(
            pa.table({"body": pa.array(self.transmitter_bodies, type=pa.int64())}),
            self.cache / TRANSMITTERS_FILENAME,
            compression="uncompressed",
        )
        feather.write_feather(
            pa.Table.from_pylist(
                [
                    {"body_pre": pre, "body_post": post, "weight": weight}
                    for pre, post, weight in self.weights
                ],
                schema=pa.schema(
                    [
                        ("body_pre", pa.int64()),
                        ("body_post", pa.int64()),
                        ("weight", pa.int64()),
                    ]
                ),
            ),
            self.cache / WEIGHTS_FILENAME,
            compression="uncompressed",
        )
        sources = []
        for source_id, filename in (
            ("annotations", ANNOTATIONS_FILENAME),
            ("neurotransmitters", TRANSMITTERS_FILENAME),
            ("weights", WEIGHTS_FILENAME),
        ):
            path = self.cache / filename
            sources.append(
                {
                    "bytes": path.stat().st_size,
                    "filename": filename,
                    "generation": f"synthetic-{source_id}-generation",
                    "id": source_id,
                    "sha256": _sha256(path),
                    "url": f"https://example.invalid/{filename}",
                }
            )
        document = {
            "dataset": {
                "id": "male-cns:v1.0",
                "license": "CC BY 4.0",
                "license_spdx": "CC-BY-4.0",
                "release_date": "2026-06-08",
            },
            "default_cache_dir": "cache",
            "schema_version": 1,
            "sources": sources,
        }
        self.manifest_path.write_bytes(
            json.dumps(document, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )

    @property
    def manifest_sha256(self) -> str:
        return _sha256(self.manifest_path)

    @property
    def trusted_sources(self) -> tuple[SourceArtifact, ...]:
        document = json.loads(self.manifest_path.read_text())
        by_id = {item["id"]: item for item in document["sources"]}
        result = []
        for source_id in ("annotations", "neurotransmitters", "weights"):
            item = by_id[source_id]
            result.append(
                SourceArtifact(
                    source_id=source_id,
                    filename=item["filename"],
                    byte_length=item["bytes"],
                    sha256=item["sha256"],
                    generation=item["generation"],
                    url=item["url"],
                )
            )
        return tuple(result)

    def build(self, output: Path) -> dict[str, Any]:
        return builder.build_runtime_artifact(
            self.manifest_path,
            self.cache,
            output,
            scratch_dir=self.root / "scratch",
            expected_manifest_sha256=self.manifest_sha256,
        )

    def load(self, output: Path, receipt: dict[str, Any]):
        return load_full_runtime_artifact(
            output,
            expected_manifest_sha256=receipt["manifest_file_sha256"],
            expected_source_manifest_sha256=self.manifest_sha256,
            trusted_sources=self.trusted_sources,
            verify_layout=True,
        )


def _rewrite_manifest_with_valid_payload_hash(path: Path, mutate: Any) -> str:
    document = json.loads(path.read_text())
    mutate(document)
    payload = {key: value for key, value in document.items() if key != "hashes"}
    document["hashes"]["payload_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    raw = canonical_json_bytes(document) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_exact_counts_uint64_lineage_layout_and_byte_reproduction(tmp_path: Path) -> None:
    source = SyntheticFullSource(tmp_path / "source")
    first = tmp_path / "artifact-first"
    second = tmp_path / "artifact-second"
    receipt = source.build(first)
    second_receipt = source.build(second)

    assert receipt["manifest_file_sha256"] == second_receipt["manifest_file_sha256"]
    assert {path.name for path in first.iterdir()} == {
        "annotation_source_rows.bin",
        "body_ids.bin",
        "edge_post_indices.bin",
        "edge_pre_indices.bin",
        "edge_raw_counts.bin",
        "edge_source_rows.bin",
        "incoming_offsets.bin",
        "manifest.json",
        "transmitter_source_rows.bin",
    }
    for first_path in first.iterdir():
        assert first_path.read_bytes() == (second / first_path.name).read_bytes()

    expected_counts = {
        "annotation_duplicate_body_ids": 0,
        "annotation_source_rows": 9,
        "excluded_edge_rows": 4,
        "excluded_neither_endpoint_rows": 2,
        "excluded_post_only_rows": 1,
        "excluded_pre_only_rows": 1,
        "excluded_self_edge_rows": 1,
        "excluded_self_edge_weight_sum": 19,
        "excluded_superclass_empty_rows": 1,
        "excluded_superclass_null_rows": 1,
        "excluded_transmitter_rows": 0,
        "excluded_weight_sum": 60,
        "raw_duplicate_directed_pairs": 0,
        "raw_edge_rows": 7,
        "raw_self_edge_rows": 2,
        "raw_self_edge_weight_sum": 21,
        "raw_weight_max": 19,
        "raw_weight_min": 2,
        "raw_weight_sum": 74,
        "retained_duplicate_directed_pairs": 0,
        "retained_edge_rows": 3,
        "retained_node_rows": 7,
        "retained_self_edge_rows": 1,
        "retained_self_edge_weight_sum": 2,
        "retained_nodes_missing_transmitter_row": 6,
        "retained_nodes_with_transmitter_row": 1,
        "retained_weight_max": 7,
        "retained_weight_min": 2,
        "retained_weight_sum": 14,
        "transmitter_duplicate_body_ids": 0,
        "transmitter_source_rows": 1,
    }
    assert receipt["counts"] == expected_counts
    with source.load(first, receipt) as graph:
        assert graph.body_ids[:] == (
            1,
            77185,
            800241,
            800917,
            805722,
            LARGE_BODY,
            LARGER_BODY,
        )
        assert graph.annotation_source_rows[:] == (0, 5, 6, 7, 8, 3, 4)
        assert graph.transmitter_source_rows[:] == (
            0,
            2**64 - 1,
            2**64 - 1,
            2**64 - 1,
            2**64 - 1,
            2**64 - 1,
            2**64 - 1,
        )
        assert graph.transmitter_source_row(1) == 0
        assert graph.transmitter_source_row(77185) is None
        assert graph.edge_pre_indices[:] == (0, 5, 6)
        assert graph.edge_post_indices[:] == (5, 6, 6)
        assert graph.edge_raw_counts[:] == (5, 7, 2)
        assert graph.edge_source_rows[:] == (0, 1, 2)
        assert graph.incoming_offsets[:] == (0, 0, 0, 0, 0, 0, 1, 3)
        assert graph.index_of(LARGE_BODY) == 5
        assert graph.role_body_ids("sensory") == (77185, 805722)
        assert graph.role_body_ids("motor") == (800241, 800917)
        assert tuple(
            node.annotation_source_row
            for node in graph.manifest.boundary_role_overlay.roles["sensory"]
        ) == (5, 8)
        assert graph.manifest.counts == expected_counts
        assert graph.manifest.retained_by_superclass == {
            "cb_sensory": 1,
            "descending": 1,
            "motor": 1,
            "sensory": 1,
            "vnc_motor": 2,
            "vnc_sensory": 1,
        }

    audit = builder.source_spot_audit(
        first,
        source.manifest_path,
        source.cache,
        expected_manifest_sha256=source.manifest_sha256,
    )
    assert audit["status"] == "matched"
    assert audit["node_samples"] == 7
    assert audit["edge_samples"] == 3
    with pytest.raises(builder.RuntimeBuildError, match="refusing to replace"):
        source.build(first)


def test_row_order_changes_only_lineage_and_provenance_identity(tmp_path: Path) -> None:
    first_source = SyntheticFullSource(tmp_path / "first-source")
    second_source = SyntheticFullSource(
        tmp_path / "second-source",
        annotations=list(reversed(_base_annotations())),
        weights=list(reversed(_base_weights())),
    )
    first_output = tmp_path / "first-artifact"
    second_output = tmp_path / "second-artifact"
    first = first_source.build(first_output)
    second = second_source.build(second_output)
    first_manifest = json.loads((first_output / "manifest.json").read_text())
    second_manifest = json.loads((second_output / "manifest.json").read_text())

    assert first["structural_graph_sha256"] == second["structural_graph_sha256"]
    assert first["provenance_graph_sha256"] != second["provenance_graph_sha256"]
    for name in (
        "body_ids",
        "edge_post_indices",
        "edge_pre_indices",
        "edge_raw_counts",
        "incoming_offsets",
        "transmitter_source_rows",
    ):
        assert first_manifest["arrays"][name]["sha256"] == second_manifest["arrays"][name]["sha256"]
    for name in ("annotation_source_rows", "edge_source_rows"):
        assert first_manifest["arrays"][name]["sha256"] != second_manifest["arrays"][name]["sha256"]


def test_loader_refuses_external_array_and_internal_identity_mismatches(
    tmp_path: Path,
) -> None:
    source = SyntheticFullSource(tmp_path / "source")
    artifact = tmp_path / "artifact"
    receipt = source.build(artifact)
    with pytest.raises(FullArtifactIdentityError, match="manifest SHA-256 mismatch"):
        load_full_runtime_artifact(
            artifact,
            expected_manifest_sha256="0" * 64,
            expected_source_manifest_sha256=source.manifest_sha256,
            trusted_sources=source.trusted_sources,
        )

    corrupted_array = tmp_path / "corrupted-array"
    shutil.copytree(artifact, corrupted_array)
    array_path = corrupted_array / "edge_raw_counts.bin"
    raw = bytearray(array_path.read_bytes())
    raw[0] ^= 1
    array_path.write_bytes(raw)
    with pytest.raises(FullArtifactIdentityError, match="array edge_raw_counts SHA-256"):
        source.load(corrupted_array, receipt)

    corrupted_identity = tmp_path / "corrupted-identity"
    shutil.copytree(artifact, corrupted_identity)
    new_pin = _rewrite_manifest_with_valid_payload_hash(
        corrupted_identity / "manifest.json",
        lambda document: document.update({"structural_graph_sha256": "0" * 64}),
    )
    with pytest.raises(FullArtifactIdentityError, match="structural graph SHA-256"):
        load_full_runtime_artifact(
            corrupted_identity,
            expected_manifest_sha256=new_pin,
            expected_source_manifest_sha256=source.manifest_sha256,
            trusted_sources=source.trusted_sources,
        )


def test_duplicate_pair_and_invalid_source_values_fail_without_partial_output(
    tmp_path: Path,
) -> None:
    cases = {
        "duplicate directed source pair": [*_base_weights(), (1, LARGE_BODY, 99)],
        "non-positive weights": [(1, LARGE_BODY, 0)],
        "must be non-null int64": [(1, LARGE_BODY, None)],
        "non-positive IDs": [(-1, LARGE_BODY, 1)],
    }
    for case_index, (message, weights) in enumerate(cases.items()):
        case_root = tmp_path / f"case-{case_index}"
        source = SyntheticFullSource(case_root, weights=weights)
        output = case_root / "artifact"
        with pytest.raises(builder.RuntimeBuildError, match=message):
            source.build(output)
        assert not output.exists()
        assert not list(case_root.glob(".artifact.*"))


def test_duplicate_annotation_and_uint64_weight_sum_overflow_refuse(tmp_path: Path) -> None:
    duplicate_root = tmp_path / "duplicate-body"
    duplicate_annotations = [*_base_annotations(), {"bodyId": 1, "superclass": "duplicate"}]
    duplicate = SyntheticFullSource(duplicate_root, annotations=duplicate_annotations)
    with pytest.raises(builder.RuntimeBuildError, match="duplicate annotation body ID 1"):
        duplicate.build(duplicate_root / "artifact")

    duplicate_transmitter_root = tmp_path / "duplicate-transmitter"
    duplicate_transmitter = SyntheticFullSource(
        duplicate_transmitter_root, transmitter_bodies=[1, 1]
    )
    with pytest.raises(builder.RuntimeBuildError, match="duplicate neurotransmitter body ID 1"):
        duplicate_transmitter.build(duplicate_transmitter_root / "artifact")

    overflow_root = tmp_path / "overflow"
    overflow = SyntheticFullSource(
        overflow_root,
        weights=[
            (1, LARGE_BODY, 2**63 - 1),
            (LARGE_BODY, LARGER_BODY, 2**63 - 1),
            (LARGER_BODY, LARGER_BODY, 2),
        ],
    )
    with pytest.raises(builder.RuntimeBuildError, match="weight sum exceeds uint64"):
        overflow.build(overflow_root / "artifact")


def test_verified_source_mismatch_fails_before_artifact_creation(tmp_path: Path) -> None:
    source = SyntheticFullSource(tmp_path / "source")
    weights_path = source.cache / WEIGHTS_FILENAME
    weights_path.write_bytes(weights_path.read_bytes() + b"corrupt")
    output = tmp_path / "artifact"
    with pytest.raises(builder.RuntimeBuildError, match="source 'weights' byte mismatch"):
        source.build(output)
    assert not output.exists()


def test_windows_path_separators_and_build_io_preserve_artifact_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identical sources must not acquire a different graph pin on Windows."""
    source = SyntheticFullSource(tmp_path / "source")
    first = tmp_path / "artifact-posix"
    first_receipt = source.build(first)

    class WindowsRelativePath(type(Path())):
        def relative_to(self, *other: Any, **kwargs: Any) -> PureWindowsPath:
            return PureWindowsPath(super().relative_to(*other, **kwargs))

    monkeypatch.setattr(builder, "Path", WindowsRelativePath)
    monkeypatch.setattr(builder, "REPOSITORY_ROOT", WindowsRelativePath(REPOSITORY_ROOT))
    monkeypatch.setattr(builder.sys, "platform", "win32")
    second = tmp_path / "artifact-windows-paths"
    second_receipt = source.build(second)

    assert second_receipt["manifest_file_sha256"] == first_receipt["manifest_file_sha256"]
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    assert not list(tmp_path.glob(".artifact-windows-paths.*"))


def test_builder_imports_without_posix_resource_and_uses_system_temp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "resource", None)
    spec = importlib.util.spec_from_file_location("portable_builder_probe", builder.__file__)
    assert spec is not None and spec.loader is not None
    portable = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(portable)

    assert portable._peak_rss_bytes() is None
    assert portable.DEFAULT_RECEIPT.parent == Path(tempfile.gettempdir())


def test_directory_sync_keeps_posix_failures_and_skips_windows_directory_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_directory_open(*args: Any, **kwargs: Any) -> int:
        raise OSError("directory sync failed")

    monkeypatch.setattr(builder.os, "open", unavailable_directory_open)
    monkeypatch.setattr(builder.sys, "platform", "linux")
    with pytest.raises(OSError, match="directory sync failed"):
        builder._fsync_directory(tmp_path)
    monkeypatch.setattr(builder.sys, "platform", "win32")
    builder._fsync_directory(tmp_path)


def test_validation_failure_closes_mappings_before_transaction_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = SyntheticFullSource(tmp_path / "source")
    output = tmp_path / "artifact"
    real_memmap = builder.np.memmap
    real_verify = builder._verify_numpy_layout
    opened = []

    def track_memmap(*args: Any, **kwargs: Any) -> Any:
        mapped = real_memmap(*args, **kwargs)
        opened.append(mapped)
        return mapped

    def corrupt_before_validation(root: Path, payload: dict[str, Any]) -> Any:
        path = root / payload["arrays"]["edge_pre_indices"]["filename"]
        with path.open("r+b") as handle:
            handle.write((payload["node_count"] + 1).to_bytes(4, "little"))
            handle.flush()
            os.fsync(handle.fileno())
        return real_verify(root, payload)

    monkeypatch.setattr(builder.np, "memmap", track_memmap)
    monkeypatch.setattr(builder, "_verify_numpy_layout", corrupt_before_validation)
    with pytest.raises(builder.RuntimeBuildError, match="dense edge index") as failure:
        source.build(output)

    # Keep the exception/traceback alive: Windows cleanup must not depend on GC.
    assert failure.value.__traceback__ is not None
    assert len(opened) == 10
    assert all(mapped._mmap.closed for mapped in opened)
    assert not output.exists()
    assert not list(tmp_path.glob(".artifact.*"))
    assert not list((source.root / "scratch").iterdir())
