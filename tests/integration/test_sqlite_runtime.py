"""The release must reject stale, mixed, or altered native SQLite runtimes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import build_sqlite_runtime as sqlite
from scripts import fetch_companion_runtime


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(sqlite, "platform_key", lambda: "macos-arm64")
    policy = json.loads((sqlite.ROOT / sqlite.POLICY).read_bytes())
    files = {}
    for name in sqlite._modified_files("macos-arm64"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"native fixture " + name.encode())
        files[name] = sqlite.sha256(path)
    receipt = {
        "schema": "neurocraft.sqlite-build.v1",
        "platform": "macos-arm64",
        "policy_sha256": sqlite.sha256(sqlite.ROOT / sqlite.POLICY),
        "recipe_sha256": sqlite.sha256(Path(sqlite.__file__)),
        "sqlite_source_sha256": policy["source"]["sha256"],
        "modified_files": files,
    }
    (tmp_path / sqlite.RECEIPT).write_text(json.dumps(receipt))
    monkeypatch.setattr(
        sqlite,
        "probe",
        lambda prefix: {
            "version": policy["version"],
            "source_id": policy["source_id"],
            "checks": ["fixture"],
        },
    )
    return tmp_path


def test_matching_derivative_is_accepted(runtime: Path) -> None:
    assert sqlite.verify_runtime(runtime)["version"] == "3.54.1"


@pytest.mark.parametrize("name", ["bin/python3.13", "lib/libpython3.13.dylib"])
def test_both_embedded_engines_are_bound(runtime: Path, name: str) -> None:
    (runtime / name).write_bytes(b"old SQLite engine")
    with pytest.raises(ValueError, match="native payload changed"):
        sqlite.verify_runtime(runtime)


@pytest.mark.parametrize(
    "field", ["policy_sha256", "recipe_sha256", "sqlite_source_sha256", "platform"]
)
def test_stale_or_foreign_receipt_is_rejected(runtime: Path, field: str) -> None:
    receipt = json.loads((runtime / sqlite.RECEIPT).read_bytes())
    receipt[field] = "other candidate"
    (runtime / sqlite.RECEIPT).write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="recipe/provenance"):
        sqlite.verify_runtime(runtime)


def test_missing_shared_engine_cannot_be_hidden_in_receipt(runtime: Path) -> None:
    receipt = json.loads((runtime / sqlite.RECEIPT).read_bytes())
    del receipt["modified_files"]["lib/libpython3.13.dylib"]
    (runtime / sqlite.RECEIPT).write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="inventory is incomplete"):
        sqlite.verify_runtime(runtime)


@pytest.mark.parametrize("field,value", [("version", "3.53.1"), ("source_id", "foreign build")])
def test_native_engine_must_agree_with_recipe(
    runtime: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: str
) -> None:
    native: dict[str, Any] = sqlite.probe(runtime)
    native[field] = value
    monkeypatch.setattr(sqlite, "probe", lambda prefix: native)
    with pytest.raises(ValueError, match="exact patched engine"):
        sqlite.verify_runtime(runtime)


def test_existing_output_is_untouched(tmp_path: Path) -> None:
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(ValueError, match="new path"):
        sqlite.build(tmp_path)
    assert sentinel.read_text() == "keep"


def test_fetch_keeps_prior_candidate_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_receipt = tmp_path / "python-runtime-provenance.json"
    old_receipt.write_text("prior immutable candidate")
    output = tmp_path / "new candidate"
    monkeypatch.setattr(fetch_companion_runtime, "build", lambda path, cache: path.mkdir())
    monkeypatch.setattr(fetch_companion_runtime.sys, "argv", ["fetch", "--output", str(output)])
    fetch_companion_runtime.main()
    assert old_receipt.read_text() == "prior immutable candidate"
    assert json.loads((output / "python-runtime-provenance.json").read_bytes())["sqlite_build"]
