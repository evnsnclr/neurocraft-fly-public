"""Full-runtime verification must be bound to the bytes that are actually used."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO

import pytest
from neurocraft_connectome import FullArtifactIdentityError, load_full_runtime_artifact
from neurocraft_connectome import full_runtime as full_runtime_module

from tests.data.test_build_malecns_runtime import SyntheticFullSource

ARRAY_NAMES = tuple(full_runtime_module.ARRAY_LAYOUT)


class _CountingOpen:
    """Count descriptors the loader opens, and swap bytes under any reopen.

    A verify-then-reopen implementation opens each verified path twice: once to
    hash it and once to use it. Serving different bytes to that second open makes
    the window observable, and a count of one per path proves it is closed.
    """

    def __init__(self, decoys: dict[Path, Path] | None = None) -> None:
        self.counts: Counter[str] = Counter()
        self._decoys = {} if decoys is None else decoys
        self._open = full_runtime_module._open_no_follow

    def __call__(self, path: Path) -> BinaryIO:
        self.counts[path.name] += 1
        decoy = self._decoys.get(path)
        if decoy is not None and self.counts[path.name] > 1:
            return self._open(decoy)
        return self._open(path)


def _artifact(tmp_path: Path) -> tuple[SyntheticFullSource, Path, dict[str, Any]]:
    source = SyntheticFullSource(tmp_path / "source")
    output = tmp_path / "artifact"
    return source, output, source.build(output)


def test_full_runtime_manifest_size_cap_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, artifact, receipt = _artifact(tmp_path)
    manifest_bytes = (artifact / "manifest.json").stat().st_size
    monkeypatch.setattr(full_runtime_module, "MAX_MANIFEST_BYTES", manifest_bytes - 1)

    with pytest.raises(FullArtifactIdentityError, match=r"exceeds the \d+-byte limit") as refusal:
        source.load(artifact, receipt)
    assert "SHA-256" not in str(refusal.value)

    monkeypatch.setattr(full_runtime_module, "MAX_MANIFEST_BYTES", manifest_bytes)
    with source.load(artifact, receipt) as graph:
        assert graph.manifest.manifest_file_sha256 == receipt["manifest_file_sha256"]


def test_full_runtime_hardlinked_array_refused(tmp_path: Path) -> None:
    source, artifact, receipt = _artifact(tmp_path)
    linked = tmp_path / "linked"
    shutil.copytree(artifact, linked)
    target = linked / "edge_raw_counts.bin"
    original = target.read_bytes()
    target.unlink()
    os.link(artifact / "edge_raw_counts.bin", target)
    assert target.read_bytes() == original

    with pytest.raises(
        FullArtifactIdentityError,
        match="array edge_raw_counts must be a single-link regular file",
    ):
        source.load(linked, receipt)


def test_full_runtime_array_hash_binds_mapped_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, artifact, receipt = _artifact(tmp_path)
    swapped = tmp_path / "edge_raw_counts-swapped.bin"
    corrupted = bytearray((artifact / "edge_raw_counts.bin").read_bytes())
    corrupted[0] ^= 1
    swapped.write_bytes(corrupted)
    opener = _CountingOpen({artifact / "edge_raw_counts.bin": swapped})
    monkeypatch.setattr(full_runtime_module, "_open_no_follow", opener)

    with source.load(artifact, receipt) as graph:
        assert graph.edge_raw_counts[:] == (5, 7, 2)
    assert [opener.counts[f"{name}.bin"] for name in ARRAY_NAMES] == [1] * len(ARRAY_NAMES)

    corrupted_artifact = tmp_path / "corrupted"
    shutil.copytree(artifact, corrupted_artifact)
    (corrupted_artifact / "edge_raw_counts.bin").write_bytes(corrupted)
    with pytest.raises(FullArtifactIdentityError, match="array edge_raw_counts SHA-256 mismatch"):
        source.load(corrupted_artifact, receipt)


def test_full_runtime_manifest_parsed_bytes_equal_hashed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, artifact, receipt = _artifact(tmp_path)
    decoy = tmp_path / "decoy-manifest.json"
    document = json.loads((artifact / "manifest.json").read_text())
    document["node_count"] = document["node_count"] + 1
    decoy.write_bytes(json.dumps(document).encode("utf-8"))
    opener = _CountingOpen({artifact / "manifest.json": decoy})
    monkeypatch.setattr(full_runtime_module, "_open_no_follow", opener)

    with source.load(artifact, receipt) as graph:
        assert graph.manifest.node_count == 7
    assert opener.counts["manifest.json"] == 1


def test_full_runtime_manifest_symlink_refused(tmp_path: Path) -> None:
    source, artifact, receipt = _artifact(tmp_path)
    indirect = tmp_path / "indirect"
    shutil.copytree(artifact, indirect)
    manifest = indirect / "manifest.json"
    manifest.unlink()
    manifest.symlink_to(artifact / "manifest.json")

    with pytest.raises(FullArtifactIdentityError, match="cannot read full artifact manifest"):
        load_full_runtime_artifact(
            indirect,
            expected_manifest_sha256=receipt["manifest_file_sha256"],
            expected_source_manifest_sha256=source.manifest_sha256,
            trusted_sources=source.trusted_sources,
        )
