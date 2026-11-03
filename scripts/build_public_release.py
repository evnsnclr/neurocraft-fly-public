#!/usr/bin/env python3
"""Export an explicitly reviewed committed tree without its private Git history.

This stdlib-only tool stages a release candidate. It does not upload, publish,
grant a license, assert fresh-machine compatibility, or sign the output. Source
bytes are read from Git objects, never the potentially dirty working directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "neurocraft.public-source-candidate.v1"
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_SOURCE_BYTES = 96 * 1024 * 1024
MANIFEST_NAME = "RELEASE_MANIFEST.json"
REQUIRED_SOURCE_ROOTS = (
    "packages/",
    "services/",
    "scripts/",
    "tests/",
    "mod/",
    "apps/",
    "data/",
    "experiments/",
    "notebooks/",
    "research/",
)
FORBIDDEN_PARTS = frozenset(
    {
        ".git",
        ".codex",
        ".claude",
        ".ssh",
        ".venv",
        ".gradle",
        ".uv-cache",
        "__pycache__",
        "node_modules",
        "artifacts",
        "recordings",
        "logs",
        "run",
        "world",
        "world_nether",
        "world_the_end",
        "launcher_accounts.json",
        "launcher_profiles.json",
        "usercache.json",
        "session.lock",
        ".npmrc",
        ".pypirc",
        ".netrc",
        "credentials.json",
        "accounts.json",
    }
)
FORBIDDEN_SUFFIXES = frozenset(
    {
        ".bundle",
        ".zip",
        ".tar",
        ".gz",
        ".feather",
        ".parquet",
        ".npy",
        ".npz",
        ".key",
        ".pem",
        ".p12",
        ".pfx",
        ".class",
        ".mcworld",
        ".mca",
        ".dat",
        ".log",
        ".ndjson",
        ".sqlite",
        ".db",
        ".dylib",
        ".so",
        ".dll",
        ".exe",
    }
)
WRAPPER_PATH = "mod/gradle/wrapper/gradle-wrapper.jar"
WRAPPER_SHA256 = "2db75c40782f5e8ba1fc278a5574bab070adccb2d21ca5a6e5ed840888448046"
SECRET_PATTERNS = {
    "AWS access key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "GitHub token": re.compile(rb"gh[pousr]_[A-Za-z0-9_]{20,}"),
    "OpenAI key": re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
PERSONAL_PATH = re.compile(
    rb"(?:/(?:Users|home)/[^\s/\"'<>]+/|[A-Za-z]:\\{1,2}Users\\{1,2}[^\s\\\"'<>]+\\)"
)
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
MIT_COPYRIGHT = "Copyright (c) 2026 Evan Smith"
MIT_TERMS = """\
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


class ReleaseError(ValueError):
    """Candidate refused before publication or partial output creation."""


@dataclass(frozen=True)
class SourceFile:
    source: str
    destination: str
    blob: str
    mode: int
    data: bytes


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_standard_mit_license(data: bytes) -> bool:
    """Recognize the owner-approved MIT text, never a title or SPDX label alone.

    Permit whitespace changes to the approved copyright header and standard terms.
    Other headers, grants or added restrictions require separate review, even if
    they mention MIT.
    """

    try:
        lines = [line.strip() for line in data.decode("utf-8").splitlines() if line.strip()]
    except UnicodeError:
        return False
    return (
        len(lines) >= 3
        and lines[0] == "MIT License"
        and " ".join(lines[1].split()) == MIT_COPYRIGHT
        and " ".join(" ".join(lines[2:]).split()) == " ".join(MIT_TERMS.split())
    )


def relative_path(value: object) -> str:
    """Require a portable exact relative path, including on Windows extractors."""

    if not isinstance(value, str) or not value:
        raise ReleaseError("every release path must be a nonempty string")
    if any(ord(char) < 32 or char in "\\:" for char in value):
        raise ReleaseError("release path contains a control character, backslash, or colon")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ReleaseError("release path must be relative without empty or dot components")
    if any(part.endswith((" ", ".")) for part in path.parts):
        raise ReleaseError("release path has a nonportable trailing space or dot")
    devices = {"con", "prn", "aux", "nul"} | {
        f"{prefix}{number}" for prefix in ("com", "lpt") for number in range(1, 10)
    }
    if any(part.split(".")[0].lower() in devices for part in path.parts):
        raise ReleaseError("release path uses a reserved device name")
    return value


def check_category(path: str) -> None:
    parts = PurePosixPath(path).parts
    if any(part.lower() in FORBIDDEN_PARTS for part in parts):
        raise ReleaseError(f"forbidden release path category: {path}")
    for index, part in enumerate(parts[:-1]):
        if part.lower() == "data" and parts[index + 1].lower() in {"raw", "cache", "processed"}:
            raise ReleaseError(f"bulk or generated data is excluded: {path}")
    name = parts[-1].lower()
    if name.startswith(".env") and name != ".env.example":
        raise ReleaseError(f"environment or credential file is excluded: {path}")
    suffix = PurePosixPath(name).suffix
    if suffix in FORBIDDEN_SUFFIXES or (suffix == ".jar" and path != WRAPPER_PATH):
        raise ReleaseError(f"forbidden release file type: {path}")


def git(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "--no-replace-objects", *args], cwd=root, capture_output=True, check=False
    )
    if result.returncode:
        # Do not echo Git stderr: it can contain the operator's local filesystem paths.
        raise ReleaseError(f"Git operation failed ({args[0]})")
    return result.stdout


def committed_tree(root: Path, revision: str) -> tuple[str, dict[str, tuple[str, str]]]:
    if not revision or revision.startswith("-") or "\x00" in revision:
        raise ReleaseError("invalid committed revision")
    commit = git(root, "rev-parse", "--verify", "--end-of-options", revision + "^{commit}")
    sha = commit.decode("ascii").strip()
    if not HEX40.fullmatch(sha):
        raise ReleaseError("expected a SHA-1 Git commit identity")
    tree: dict[str, tuple[str, str]] = {}
    for record in git(root, "ls-tree", "-rz", "--full-tree", sha).split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, kind, blob = metadata.decode("ascii").split(" ")
        path = raw_path.decode("utf-8")
        tree[path] = (mode if kind == "blob" else "unsupported", blob)
    return sha, tree


def read_blob(root: Path, tree: dict[str, tuple[str, str]], path: str) -> tuple[str, int, bytes]:
    if path not in tree:
        raise ReleaseError(f"allowlisted source is absent from the committed revision: {path}")
    mode, blob = tree[path]
    if mode not in {"100644", "100755"}:
        raise ReleaseError(f"symlink, submodule, or unsupported Git mode refused: {path}")
    size = int(git(root, "cat-file", "-s", blob))
    if size > MAX_FILE_BYTES:
        raise ReleaseError(f"source file exceeds bounded release size: {path}")
    return blob, int(mode, 8), git(root, "cat-file", "blob", blob)


def load_config(data: bytes) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ReleaseError(f"duplicate configuration key: {key}")
            output[key] = value
        return output

    try:
        config = json.loads(data, object_pairs_hook=unique)
    except (ValueError, UnicodeError) as exc:
        raise ReleaseError("release configuration is not valid unique-key JSON") from exc
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ReleaseError("unsupported release allowlist schema")
    if not isinstance(config.get("files"), list) or not config["files"]:
        raise ReleaseError("release allowlist must contain files")
    if not isinstance(config.get("exclusions"), list) or not config["exclusions"]:
        raise ReleaseError("release allowlist must declare exclusions")
    return config


def scan_source(file: SourceFile, exceptions: dict[str, str]) -> None:
    for label, pattern in SECRET_PATTERNS.items():
        if pattern.search(file.data):
            raise ReleaseError(f"possible {label} in {file.source}; secret bytes omitted")
    if file.source == WRAPPER_PATH:
        if file.destination != WRAPPER_PATH or digest(file.data) != WRAPPER_SHA256:
            raise ReleaseError("Gradle wrapper does not match the reviewed binary identity")
        return
    if file.source.endswith(".png"):
        if not file.data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ReleaseError(f"invalid PNG signature: {file.source}")
        return
    try:
        file.data.decode("utf-8")
    except UnicodeError as exc:
        raise ReleaseError(f"unreviewed binary source file: {file.source}") from exc
    if b"\x00" in file.data:
        raise ReleaseError(f"NUL byte in purported textual source: {file.source}")
    if PERSONAL_PATH.search(file.data) and exceptions.get(file.source) != digest(file.data):
        raise ReleaseError(f"absolute personal path in {file.source}; no source bytes changed")


def selected_files(
    root: Path, tree: dict[str, tuple[str, str]], config: dict[str, Any]
) -> list[SourceFile]:
    exceptions: dict[str, str] = {}
    for entry in config.get("reviewed_text_exceptions", []):
        source = relative_path(entry["source"])
        sha = entry["sha256"]
        if not isinstance(sha, str) or not HEX64.fullmatch(sha) or not entry.get("reason"):
            raise ReleaseError("text exception needs a reviewed hash and reason")
        exceptions[source] = sha
    sources: set[str] = set()
    destinations: set[str] = {MANIFEST_NAME.casefold()}
    files: list[SourceFile] = []
    total_size = 0
    for entry in config["files"]:
        if not isinstance(entry, dict) or set(entry) != {"source", "destination"}:
            raise ReleaseError("each allowlist entry needs exactly source and destination")
        source = relative_path(entry["source"])
        destination = relative_path(entry["destination"])
        check_category(source)
        check_category(destination)
        if source in sources or destination.casefold() in destinations:
            raise ReleaseError(f"duplicate or case-colliding release path: {destination}")
        sources.add(source)
        destinations.add(destination.casefold())
        blob, mode, data = read_blob(root, tree, source)
        total_size += len(data)
        if total_size > MAX_SOURCE_BYTES:
            raise ReleaseError("source candidate exceeds total bounded release size")
        file = SourceFile(source, destination, blob, mode, data)
        scan_source(file, exceptions)
        files.append(file)
    # ZIP readers must never choose between a file and a directory of the same name.
    for destination in destinations:
        if any(str(parent) in destinations for parent in PurePosixPath(destination).parents):
            raise ReleaseError("release path has a file/directory prefix collision")
    for required in ("README.md", "LICENSE", "NOTICE", "pyproject.toml", "uv.lock"):
        if required.casefold() not in destinations:
            raise ReleaseError(f"required release file missing: {required}")
    omitted_source = sorted(
        path for path in tree if path.startswith(REQUIRED_SOURCE_ROOTS) and path not in sources
    )
    if omitted_source:
        raise ReleaseError(
            "application source or test file is not explicitly allowlisted: " + omitted_source[0]
        )
    return sorted(files, key=lambda file: file.destination)


def markdown_link_targets(markdown: str) -> list[str]:
    """Read inline links and reference definitions, excluding Markdown code spans.

    This is a bounded local-file audit, not a CommonMark renderer or an external
    URL/heading checker. Reference definitions are checked even when unused.
    """

    markdown = re.sub(r"(?ms)^ *(`{3,}|~{3,}).*?^ *\1 *$", "", markdown)
    markdown = re.sub(r"(?s)`+[^`]*`+", "", markdown)
    targets = [
        match.group(1) or match.group(2)
        for match in re.finditer(r"\]\(\s*(?:<([^>\n]+)>|([^\s)]+))", markdown)
    ]
    targets.extend(
        match.group(1) or match.group(2)
        for match in re.finditer(r"(?m)^ {0,3}\[(?!\^)[^]\n]+\]:\s*(?:<([^>\n]+)>|(\S+))", markdown)
    )
    return targets


def audit_documentation_links(files: list[SourceFile], config: dict[str, Any]) -> dict[str, Any]:
    """Resolve documentation links against export destinations, never local files.

    Immutable historical records may reference omitted private support material.
    Exceptions pin the original source bytes and every omitted destination, so a
    new broken current-facing link cannot inherit an unrelated historical waiver.
    """

    by_source = {file.source: file for file in files}
    destinations = {file.destination for file in files} | {MANIFEST_NAME}
    directories = {str(parent) for name in destinations for parent in PurePosixPath(name).parents}
    exceptions: dict[str, set[str]] = {}
    reasons: dict[str, str] = {}
    for entry in config.get("historical_link_exceptions", []):
        if not isinstance(entry, dict) or set(entry) != {
            "source",
            "sha256",
            "missing_destinations",
            "reason",
        }:
            raise ReleaseError("historical link exception needs source, hash, targets and reason")
        source = relative_path(entry["source"])
        targets = entry["missing_destinations"]
        reason = entry["reason"]
        if (
            source in exceptions
            or source not in by_source
            or entry["sha256"] != digest(by_source[source].data)
            or not isinstance(reason, str)
            or not reason.strip()
            or not isinstance(targets, list)
            or not targets
        ):
            raise ReleaseError("historical link exception has stale bytes or invalid review fields")
        normalized = {relative_path(target) for target in targets}
        if len(normalized) != len(targets):
            raise ReleaseError("duplicate target in historical link exception")
        exceptions[source], reasons[source] = normalized, reason
    used: dict[str, set[str]] = {}
    omitted: list[dict[str, str]] = []
    checked_documents = local_links = 0
    for file in files:
        if file.destination.endswith(".md"):
            markdown = file.data.decode("utf-8")
        elif file.destination.endswith(".ipynb"):
            try:
                notebook = json.loads(file.data)
                markdown = "\n\n".join(
                    "".join(cell["source"])
                    for cell in notebook["cells"]
                    if cell["cell_type"] == "markdown"
                )
            except (ValueError, TypeError, KeyError) as exc:
                raise ReleaseError(f"cannot audit notebook Markdown: {file.destination}") from exc
        else:
            continue
        checked_documents += 1
        for target in markdown_link_targets(markdown):
            try:
                url = urlsplit(target)
            except ValueError as exc:
                raise ReleaseError(f"invalid documentation link in {file.destination}") from exc
            if url.scheme or url.netloc or not url.path:
                continue
            local_links += 1
            destination = posixpath.normpath(
                posixpath.join(posixpath.dirname(file.destination), unquote(url.path))
            )
            if destination in destinations or destination in directories:
                continue
            if destination not in exceptions.get(file.source, set()):
                raise ReleaseError(
                    f"unresolved local documentation link: {file.destination} -> {destination}"
                )
            used.setdefault(file.source, set()).add(destination)
            record = {
                "document": file.destination,
                "omitted_destination": destination,
                "reason": reasons[file.source],
            }
            if record not in omitted:
                omitted.append(record)
    if used != exceptions:
        raise ReleaseError("historical link exceptions contain unused targets; review them again")
    return {
        "status": "passed-with-declared-historical-omissions" if omitted else "passed",
        "checked_documents": checked_documents,
        "local_link_occurrences": local_links,
        "scope": "Markdown inline links and reference definitions, including notebook Markdown; "
        "exported file/directory existence only; no heading or external URL validation",
        "historical_omissions": omitted,
    }


def make_manifest(
    sha: str, config_path: str, config_data: bytes, config: dict[str, Any], files: list[SourceFile]
) -> dict[str, Any]:
    license_file = next(file for file in files if file.destination == "LICENSE")
    mit_verified = is_standard_mit_license(license_file.data)
    return {
        "schema": SCHEMA,
        "status": "candidate-pending-release-review"
        if mit_verified
        else "candidate-pending-license",
        "distribution_ready": False,
        "publication_performed": False,
        "license": {
            "status": "standard-mit-grant-verified" if mit_verified else "owner-decision-required",
            "spdx_id": "MIT" if mit_verified else None,
            "sha256": digest(license_file.data),
            "notice": "This builder does not grant permissions or approve distribution.",
        },
        "source": {
            "commit": sha,
            "selection_path": config_path,
            "selection_sha256": digest(config_data),
            "content_origin": "unchanged committed Git blobs; selected documents are path-mapped",
            "history_included": False,
            "working_directory_changes_included": False,
        },
        "scope": {
            "kind": "curated application source snapshot",
            "installer": False,
            "mod_binary_included": False,
            "minecraft_included": False,
            "bulk_connectome_data_included": False,
            "fresh_machine_playtest": "not established by this packaging operation",
            "verification": config.get("verification_scope", "See public installation guide."),
        },
        "exclusions": config["exclusions"],
        "reviewed_text_exceptions": config.get("reviewed_text_exceptions", []),
        "documentation_links": audit_documentation_links(files, config),
        "integrity": {
            "algorithm": "sha256",
            "signature": None,
            "notice": "Hashes detect changes; they do not establish publisher authenticity.",
        },
        "files": [
            {
                "path": file.destination,
                "source_path": file.source,
                "source_blob": file.blob,
                "mode": f"{file.mode:o}",
                "bytes": len(file.data),
                "sha256": digest(file.data),
            }
            for file in files
        ],
    }


def write_zip(path: Path, files: list[SourceFile], manifest: bytes) -> None:
    members = [(file.destination, file.data, file.mode) for file in files]
    members.append((MANIFEST_NAME, manifest, stat.S_IFREG | 0o644))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data, mode in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = mode << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ReleaseError("new source archive failed CRC validation")
        if len(archive.infolist()) != len(members):
            raise ReleaseError("new source archive has an unexpected member count")
        for name, data, _ in members:
            if archive.read(name) != data:
                raise ReleaseError(f"new source archive failed byte verification: {name}")


def build_release(root: Path, revision: str, output: Path, config_path: str) -> dict[str, Any]:
    config_path = relative_path(config_path)
    if output.exists() or output.is_symlink():
        raise ReleaseError("output must be a new path; existing artifacts are never overwritten")
    sha, tree = committed_tree(root, revision)
    config_blob, config_mode, config_data = read_blob(root, tree, config_path)
    check_category(config_path)
    scan_source(SourceFile(config_path, config_path, config_blob, config_mode, config_data), {})
    config = load_config(config_data)
    files = selected_files(root, tree, config)
    manifest = make_manifest(sha, config_path, config_data, config, files)
    manifest_bytes = canonical_json(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".public-release-", dir=output.parent))
    try:
        archive = stage / "source.zip"
        write_zip(archive, files, manifest_bytes)
        (stage / "manifest.json").write_bytes(manifest_bytes)
        result = {
            "schema": "neurocraft.public-source-build.v1",
            "status": manifest["status"],
            "distribution_ready": False,
            "publication_performed": False,
            "license": manifest["license"],
            "source_commit": sha,
            "files": len(files),
            "source_zip": {"path": "source.zip", "sha256": digest(archive.read_bytes())},
            "manifest": {"path": "manifest.json", "sha256": digest(manifest_bytes)},
        }
        (stage / "build-result.json").write_bytes(canonical_json(result))
        # Refuse a destination created after preflight as well.
        if output.exists() or output.is_symlink():
            raise ReleaseError("output appeared while staging; refusing to overwrite")
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return result


def verify_source_snapshot(root: Path) -> dict[str, Any]:
    """Check extracted bytes and POSIX executable scripts without private Git.

    This detects corruption relative to the accompanying manifest. An independently
    trusted archive checksum is still needed to establish the publisher's identity.
    Generated caches/worlds are outside the immutable source manifest and are not
    read, enumerated, or included in the verification result.
    """

    root = root.resolve()
    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_symlink():
        raise ReleaseError("source manifest must not be a symlink")
    if manifest_path.stat().st_size > MAX_FILE_BYTES:
        raise ReleaseError("source manifest exceeds bounded size")
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (ValueError, UnicodeError) as exc:
        raise ReleaseError("invalid source manifest JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ReleaseError("unsupported source manifest schema")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ReleaseError("source manifest needs a nonempty file list")
    seen: set[str] = set()
    for record in files:
        if not isinstance(record, dict):
            raise ReleaseError("invalid source manifest file entry")
        name = relative_path(record.get("path"))
        check_category(name)
        if name.casefold() in seen:
            raise ReleaseError("duplicate path in source manifest")
        seen.add(name.casefold())
        path = root
        for part in PurePosixPath(name).parts:
            path /= part
            if path.is_symlink():
                raise ReleaseError(f"symlink in extracted source: {name}")
        if not path.is_file() or path.stat().st_size != record.get("bytes"):
            raise ReleaseError(f"extracted source missing or size changed: {name}")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ReleaseError(f"extracted source exceeds bounded size: {name}")
        if digest(path.read_bytes()) != record.get("sha256"):
            raise ReleaseError(f"extracted source hash mismatch: {name}")
        if (
            os.name != "nt"
            and record.get("mode") == "100755"
            and not path.stat().st_mode & stat.S_IXUSR
        ):
            raise ReleaseError(
                f"extracted script is not executable: {name}; "
                "extract the source ZIP with a tool that preserves executable permissions"
            )
    return {
        "schema": "neurocraft.public-source-integrity.v1",
        "status": "verified-against-accompanying-manifest",
        "files": len(files),
        "publisher_authenticity_verified": False,
        "distribution_ready": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", help="Committed source revision; no dirty files")
    parser.add_argument("--output", type=Path, help="New local candidate directory")
    parser.add_argument("--config", default="release/public-files.json", help="Committed allowlist")
    parser.add_argument("--verify-source", type=Path, help="Verify an extracted source candidate")
    args = parser.parse_args(argv)
    if args.verify_source is not None and (args.revision is not None or args.output is not None):
        parser.error("--verify-source cannot be combined with --revision or --output")
    if args.verify_source is None and (args.revision is None or args.output is None):
        parser.error("building requires --revision and --output")
    try:
        result = (
            verify_source_snapshot(args.verify_source)
            if args.verify_source is not None
            else build_release(ROOT, args.revision, args.output, args.config)
        )
    except (ReleaseError, OSError) as exc:
        # OSError filenames can reveal local paths; only the bounded custom reason is printed.
        message = str(exc) if isinstance(exc, ReleaseError) else "local I/O operation failed"
        print(json.dumps({"status": "refused", "reason": message}))
        return 1
    print(canonical_json(result).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
