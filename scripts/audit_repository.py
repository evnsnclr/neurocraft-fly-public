#!/usr/bin/env python3
"""Fail-closed repository hygiene and immutable G0 identity checks."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "data/manifests/male-cns-v1.0.sources.json": (
        "9412105f54884a57521bbdf7fe46659caab559ef7877e6a62af95f50e1ee3768"
    ),
    "data/fixtures/mcns-v1.0-flight-power-v1.json": (
        "3076c22fd0c87041c601b865c9fc8e913466200491f80775060e28412a087ff8"
    ),
    "data/fixtures/mcns-v1.0-flight-steering-v1.json": (
        "71d9a56c81914d02af8286745a23f320e37531d040b8249ff50719a26871b05b"
    ),
}
PROHIBITED_PATH_COMPONENTS = (
    ("artifacts",),
    ("data", "cache"),
    ("data", "processed"),
    ("data", "raw"),
    ("logs",),
    ("recordings",),
    ("run",),
    ("world",),
    ("world_nether",),
    ("world_the_end",),
)
PROHIBITED_SUFFIXES = (".class", ".jar", ".key", ".mcworld", ".pem")
ALLOWED_BINARY_HASHES = {
    "mod/gradle/wrapper/gradle-wrapper.jar": (
        "2db75c40782f5e8ba1fc278a5574bab070adccb2d21ca5a6e5ed840888448046"
    )
}
SECRET_PATTERNS = {
    "AWS access key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "GitHub token": re.compile(rb"gh[pousr]_[A-Za-z0-9_]{20,}"),
    "OpenAI key": re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
SECRET_SCAN_CHUNK_BYTES = 1024 * 1024
SECRET_SCAN_OVERLAP_BYTES = 4096


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_paths(*, include_untracked: bool) -> tuple[str, ...]:
    command = ["git", "ls-files", "-z"]
    if include_untracked:
        command.extend(("--cached", "--others", "--exclude-standard"))
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return tuple(path.decode("utf-8") for path in result.stdout.split(b"\0") if path)


def is_prohibited_path(path: str) -> bool:
    parts = tuple(part for part in Path(path).parts if part not in {".", ""})
    for prohibited in PROHIBITED_PATH_COMPONENTS:
        width = len(prohibited)
        if any(
            parts[index : index + width] == prohibited for index in range(len(parts) - width + 1)
        ):
            return True
    return False


def scan_secrets(path: Path) -> tuple[str, ...]:
    """Stream every candidate, retaining overlap for tokens split across chunks."""

    matches: set[str] = set()
    carry = b""
    with path.open("rb") as handle:
        while chunk := handle.read(SECRET_SCAN_CHUNK_BYTES):
            window = carry + chunk
            for label, pattern in SECRET_PATTERNS.items():
                if label not in matches and pattern.search(window):
                    matches.add(label)
            carry = window[-SECRET_SCAN_OVERLAP_BYTES:]
    return tuple(sorted(matches))


def audit() -> dict[str, object]:
    failures: list[str] = []
    tracked = git_paths(include_untracked=False)
    candidates = git_paths(include_untracked=True)
    tracked_set = set(tracked)
    for path in candidates:
        if is_prohibited_path(path):
            failures.append(f"prohibited repository path: {path}")
        if path.lower().endswith(PROHIBITED_SUFFIXES) and path not in ALLOWED_BINARY_HASHES:
            failures.append(f"prohibited tracked file type: {path}")
        basename = Path(path).name
        if (basename == ".env" or basename.startswith(".env.")) and basename != ".env.example":
            failures.append(f"environment secret file tracked: {path}")
        absolute = ROOT / path
        if not absolute.is_file():
            continue
        for label in scan_secrets(absolute):
            failures.append(f"possible {label} in {path}")
        expected_binary_hash = ALLOWED_BINARY_HASHES.get(path)
        if expected_binary_hash is not None:
            observed_binary_hash = sha256(absolute)
            if observed_binary_hash != expected_binary_hash:
                failures.append(
                    f"allowed binary identity mismatch for {path}: "
                    f"{observed_binary_hash} != {expected_binary_hash}"
                )
    observed_hashes: dict[str, str] = {}
    for path, expected in EXPECTED.items():
        if path not in tracked_set:
            failures.append(f"required G0 evidence is not tracked: {path}")
            continue
        observed = sha256(ROOT / path)
        observed_hashes[path] = observed
        if observed != expected:
            failures.append(f"identity mismatch for {path}: {observed} != {expected}")
    fixture_path = ROOT / "data/fixtures/mcns-v1.0-flight-power-v1.json"
    if fixture_path.is_file():
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        if fixture.get("dataset_id") != "male-cns:v1.0":
            failures.append("fixture dataset identity is not male-cns:v1.0")
        if fixture.get("fixture_id") != "mcns-v1.0-flight-power-v1":
            failures.append("fixture identity mismatch")
        if any(edge.get("sign_status") != "unknown" for edge in fixture.get("edges", [])):
            failures.append("G0 fixture contains a sign presented as observed/known")
    return {
        "status": "failed" if failures else "ok",
        "scanned_files": len(candidates),
        "tracked_files": len(tracked),
        "verified_g0_hashes": observed_hashes,
        "failures": failures,
    }


def main() -> int:
    try:
        receipt = audit()
    except (OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        receipt = {"status": "error", "error_type": type(error).__name__, "message": str(error)}
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
