from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "audit_repository.py"


def load_module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("audit_repository", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_hygiene_and_g0_identities() -> None:
    receipt = load_module().audit()
    assert receipt["status"] == "ok", receipt["failures"]
    assert receipt["verified_g0_hashes"] == load_module().EXPECTED
    assert load_module().ALLOWED_BINARY_HASHES == {
        "mod/gradle/wrapper/gradle-wrapper.jar": (
            "2db75c40782f5e8ba1fc278a5574bab070adccb2d21ca5a6e5ed840888448046"
        )
    }


def test_secret_patterns_detect_examples_without_real_credentials() -> None:
    module = load_module()
    examples = {
        "AWS access key": b"AK" + b"IA" + b"ABCDEFGHIJKLMNOP",
        "GitHub token": b"gh" + b"p_" + b"abcdefghijklmnopqrstuvwxyz012345",
        "OpenAI key": b"s" + b"k-proj-" + b"abcdefghijklmnopqrstuvwxyz",
        "private key": b"-----BEGIN " + b"PRIVATE KEY-----",
    }
    for label, example in examples.items():
        assert module.SECRET_PATTERNS[label].search(example)


def test_prohibited_paths_are_rejected_at_any_depth() -> None:
    module = load_module()
    for path in (
        "artifacts/run.json",
        "nested/data/raw/source.bin",
        "mod/run/world.dat",
        "nested/world_nether/region/r.0.0.mca",
    ):
        assert module.is_prohibited_path(path)
    assert not module.is_prohibited_path("docs/research/run-design.md")
    assert not module.is_prohibited_path("data/fixtures/fixture.json")


def test_secret_scanner_streams_files_larger_than_old_limit(tmp_path: Path) -> None:
    module = load_module()
    candidate = tmp_path / "large-source.txt"
    prefix = b"x" * (5_000_000 + module.SECRET_SCAN_CHUNK_BYTES - 8)
    candidate.write_bytes(prefix + b"gh" + b"p_" + b"abcdefghijklmnopqrstuvwxyz012345")
    assert module.scan_secrets(candidate) == ("GitHub token",)
