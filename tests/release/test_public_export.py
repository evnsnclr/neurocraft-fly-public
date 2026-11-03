"""Exercise candidate boundaries against real temporary Git repositories."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import zipfile
from pathlib import Path

import pytest

from scripts.build_public_release import (
    MANIFEST_NAME,
    ReleaseError,
    SourceFile,
    audit_documentation_links,
    build_release,
    canonical_json,
    digest,
    relative_path,
    verify_source_snapshot,
)

MIT_LICENSE = """\
MIT License

Copyright (c) 2026 Evan Smith

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


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def commit(root: Path) -> str:
    git(root, "add", ".")
    git(root, "commit", "-qm", "fixture source")
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def source_repo(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "example@example.invalid")
    git(root, "config", "user.name", "Release test fixture")
    # A private history-only file must never enter the curated candidate.
    (root / "private-history.txt").write_text("PRIVATE HISTORICAL MATERIAL")
    commit(root)
    (root / "private-history.txt").unlink()
    originals = {
        "docs/public/README.md": "# Candidate\n",
        "LICENSE": "Copyright 2026 Example. All rights reserved.\n",
        "NOTICE": "Attribution fixture.\n",
        "pyproject.toml": "[project]\nname = 'example'\n",
        "uv.lock": "version = 1\n",
        "scripts/run.sh": "#!/bin/sh\nexit 0\n",
    }
    for path, text in originals.items():
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
    (root / "scripts/run.sh").chmod(0o755)
    entries = [
        {"source": path, "destination": "README.md" if path.endswith("README.md") else path}
        for path in originals
    ]
    config = {
        "schema_version": 1,
        "files": entries,
        "exclusions": [{"path": ".git", "reason": "Private history is excluded."}],
    }
    (root / "release").mkdir()
    (root / "release/public-files.json").write_bytes(canonical_json(config))
    commit(root)
    return root


def mutate_allowlist(root: Path, source: str, destination: str | None = None) -> str:
    path = root / "release/public-files.json"
    config = json.loads(path.read_bytes())
    config["files"].append({"source": source, "destination": destination or source})
    path.write_bytes(canonical_json(config))
    return commit(root)


def build(root: Path, output: Path, revision: str = "HEAD") -> dict[str, object]:
    return build_release(root, revision, output, "release/public-files.json")


def test_export_is_deterministic_and_ignores_dirty_files_and_private_history(
    source_repo: Path, tmp_path: Path
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    result = build(source_repo, first)
    (source_repo / "docs/public/README.md").write_text("UNCOMMITTED EDIT")
    (source_repo / ".env").write_text("PRIVATE UNTRACKED CREDENTIAL")
    build(source_repo, second)
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    assert result["status"] == "candidate-pending-license"
    assert result["distribution_ready"] is False
    assert result["publication_performed"] is False
    license_metadata = result["license"]
    assert isinstance(license_metadata, dict)
    assert license_metadata["status"] == "owner-decision-required"
    archive_bytes = (first / "source.zip").read_bytes()
    assert b"PRIVATE HISTORICAL MATERIAL" not in archive_bytes
    assert b"PRIVATE UNTRACKED CREDENTIAL" not in archive_bytes
    assert b"UNCOMMITTED EDIT" not in archive_bytes
    with zipfile.ZipFile(first / "source.zip") as archive:
        assert archive.read("README.md") == b"# Candidate\n"
        assert not any(".git" in name or "private" in name for name in archive.namelist())
        assert archive.getinfo("scripts/run.sh").external_attr >> 16 == 0o100755
        manifest = json.loads(archive.read(MANIFEST_NAME))
        assert archive.read(MANIFEST_NAME) == (first / "manifest.json").read_bytes()
        assert manifest["source"]["commit"] == git(source_repo, "rev-parse", "HEAD")
        for record in manifest["files"]:
            assert digest(archive.read(record["path"])) == record["sha256"]


@pytest.mark.parametrize(
    "path",
    ["../outside", "/absolute", "a/../b", "a//b", "a\\b", "C:relative", "a\x00b", "a/NUL.txt"],
)
def test_nonportable_and_traversing_paths_are_refused(path: str) -> None:
    with pytest.raises(ReleaseError):
        relative_path(path)


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".git/config",
        "data/raw/graph.txt",
        "artifacts/run.json",
        "backup.bundle",
        "trace.ndjson",
    ],
)
def test_forbidden_material_cannot_be_explicitly_allowlisted(
    source_repo: Path, tmp_path: Path, path: str
) -> None:
    # Category rejection precedes Git object access, including .git/config.
    mutate_allowlist(source_repo, path)
    output = tmp_path / "refused"
    with pytest.raises(ReleaseError, match=r"forbidden|excluded|credential"):
        build(source_repo, output)
    assert not output.exists()


def test_symlink_source_is_never_followed(source_repo: Path, tmp_path: Path) -> None:
    outside = tmp_path / "private.txt"
    outside.write_text("PRIVATE")
    (source_repo / "linked.txt").symlink_to(outside)
    mutate_allowlist(source_repo, "linked.txt")
    with pytest.raises(ReleaseError, match="symlink"):
        build(source_repo, tmp_path / "refused")


@pytest.mark.parametrize("kind", ["personal_path", "credential", "binary"])
def test_content_scan_refuses_without_echoing_sensitive_bytes(
    source_repo: Path, tmp_path: Path, kind: str
) -> None:
    values = {
        "personal_path": "/".join(["", "Users", "actual-person", "private-project"]).encode(),
        "credential": b"ghp_" + b"X" * 40,
        "binary": b"\xff\x00binary",
    }
    secret = values[kind]
    (source_repo / "source.txt").write_bytes(secret)
    mutate_allowlist(source_repo, "source.txt")
    with pytest.raises(ReleaseError) as caught:
        build(source_repo, tmp_path / "refused")
    assert secret.decode("utf-8", errors="replace") not in str(caught.value)


def test_case_collision_is_refused(source_repo: Path, tmp_path: Path) -> None:
    (source_repo / "other.txt").write_text("another")
    mutate_allowlist(source_repo, "other.txt", "readme.MD")
    with pytest.raises(ReleaseError, match="case-colliding"):
        build(source_repo, tmp_path / "refused")


def test_file_directory_collision_is_refused(source_repo: Path, tmp_path: Path) -> None:
    (source_repo / "other.txt").write_text("another")
    mutate_allowlist(source_repo, "other.txt", "README.md/inside.txt")
    with pytest.raises(ReleaseError, match="prefix collision"):
        build(source_repo, tmp_path / "refused")


def test_existing_output_is_not_changed(source_repo: Path, tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("KEEP")
    with pytest.raises(ReleaseError, match="never overwritten"):
        build(source_repo, output)
    assert (output / "keep.txt").read_text() == "KEEP"


def test_missing_committed_file_does_not_fall_back_to_worktree(
    source_repo: Path, tmp_path: Path
) -> None:
    mutate_allowlist(source_repo, "new.txt")
    (source_repo / "new.txt").write_text("Only in working directory")
    with pytest.raises(ReleaseError, match="absent from the committed revision"):
        build(source_repo, tmp_path / "refused")


@pytest.mark.parametrize("line_endings", ["\n", "\r\n"])
def test_complete_mit_grant_is_recognized_without_approving_publication(
    source_repo: Path, tmp_path: Path, line_endings: str
) -> None:
    git(source_repo, "config", "core.autocrlf", "false")
    license_bytes = MIT_LICENSE.replace("\n", line_endings).encode()
    (source_repo / "LICENSE").write_bytes(license_bytes)
    commit(source_repo)
    result = build(source_repo, tmp_path / "candidate")
    manifest = json.loads((tmp_path / "candidate/manifest.json").read_bytes())
    for metadata in (result, manifest):
        assert metadata["status"] == "candidate-pending-release-review"
        assert metadata["distribution_ready"] is False
        assert metadata["publication_performed"] is False
        assert metadata["license"]["status"] == "standard-mit-grant-verified"
        assert metadata["license"]["spdx_id"] == "MIT"
        assert metadata["license"]["sha256"] == digest(license_bytes)
    with zipfile.ZipFile(tmp_path / "candidate/source.zip") as archive:
        assert archive.read("LICENSE") == license_bytes


@pytest.mark.parametrize(
    "license_text",
    [
        "Example license text\n",
        "SPDX-License-Identifier: MIT\n",
        "MIT License\nCopyright (c) 2026 Example Author\nTerms available on request.\n",
        MIT_LICENSE.replace("free of charge", "for noncommercial use only"),
        MIT_LICENSE + "Additional condition: commercial use is prohibited.\n",
        MIT_LICENSE + "All rights reserved.\n",
        MIT_LICENSE.split("THE SOFTWARE IS PROVIDED")[0],
        MIT_LICENSE.replace("Copyright (c) 2026 Evan Smith\n", ""),
        MIT_LICENSE.replace("Copyright (c) 2026 Evan Smith", "Copyright (c) [year] [fullname]"),
        MIT_LICENSE.replace(
            "Copyright (c) 2026 Evan Smith",
            "Copyright (c) 2026 Evan Smith. Commercial use is prohibited.",
        ),
        MIT_LICENSE.replace("Copyright (c) 2026 Evan Smith", "Copyright (c) 2027 Evan Smith"),
        MIT_LICENSE.replace("Copyright (c) 2026 Evan Smith", "Copyright (c) 2026 Another Owner"),
    ],
    ids=[
        "unknown",
        "spdx-only",
        "title-only",
        "changed-grant",
        "extra-restriction",
        "reserved-rights",
        "missing-disclaimer",
        "missing-copyright",
        "template-placeholder",
        "inline-copyright-restriction",
        "unreviewed-copyright-year",
        "unreviewed-copyright-owner",
    ],
)
def test_unknown_or_altered_license_remains_unresolved(
    source_repo: Path, tmp_path: Path, license_text: str
) -> None:
    (source_repo / "LICENSE").write_text(license_text)
    commit(source_repo)
    result = build(source_repo, tmp_path / "candidate")
    manifest = json.loads((tmp_path / "candidate/manifest.json").read_bytes())
    assert result["status"] == "candidate-pending-license"
    assert result["distribution_ready"] is False
    assert result["publication_performed"] is False
    assert manifest["license"]["status"] == "owner-decision-required"
    assert manifest["license"]["spdx_id"] is None


def test_uncommitted_mit_grant_does_not_change_exported_legacy_license(
    source_repo: Path, tmp_path: Path
) -> None:
    (source_repo / "LICENSE").write_text(MIT_LICENSE)
    result = build(source_repo, tmp_path / "candidate")
    assert result["status"] == "candidate-pending-license"
    assert result["distribution_ready"] is False
    with zipfile.ZipFile(tmp_path / "candidate/source.zip") as archive:
        assert archive.read("LICENSE") == b"Copyright 2026 Example. All rights reserved.\n"


def test_text_exception_is_bound_to_exact_reviewed_file_bytes(
    source_repo: Path, tmp_path: Path
) -> None:
    data = "/".join(["", "Users", "example", "fictional"]).encode()
    (source_repo / "example.txt").write_bytes(data)
    mutate_allowlist(source_repo, "example.txt")
    path = source_repo / "release/public-files.json"
    config = json.loads(path.read_bytes())
    config["reviewed_text_exceptions"] = [
        {"source": "example.txt", "sha256": digest(data), "reason": "Synthetic test fixture"}
    ]
    path.write_bytes(canonical_json(config))
    commit(source_repo)
    build(source_repo, tmp_path / "accepted")
    (source_repo / "example.txt").write_bytes(data + b" CHANGED")
    commit(source_repo)
    with pytest.raises(ReleaseError, match="absolute personal path"):
        build(source_repo, tmp_path / "refused")


def test_extracted_snapshot_verifies_without_git_and_detects_modified_source(
    source_repo: Path, tmp_path: Path
) -> None:
    candidate = tmp_path / "candidate"
    extracted = tmp_path / "extracted"
    build(source_repo, candidate)
    with zipfile.ZipFile(candidate / "source.zip") as archive:
        archive.extractall(extracted)
        for entry in archive.infolist():
            if os.name != "nt" and entry.external_attr >> 16 & stat.S_IXUSR:
                (extracted / entry.filename).chmod(stat.S_IMODE(entry.external_attr >> 16))
    result = verify_source_snapshot(extracted)
    assert result["status"] == "verified-against-accompanying-manifest"
    assert result["publisher_authenticity_verified"] is False
    assert not (extracted / ".git").exists()
    # Generated runtime state is not public source and is not scanned or exported here.
    (extracted / "artifacts").mkdir()
    (extracted / "artifacts/private.json").write_text("LOCAL ONLY")
    assert verify_source_snapshot(extracted) == result
    (extracted / "README.md").write_bytes(b"# Tampered!\n")
    with pytest.raises(ReleaseError, match=r"changed|mismatch"):
        verify_source_snapshot(extracted)


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable permissions do not apply on Windows")
def test_snapshot_refuses_lost_execute_permission_without_changing_source_bytes(
    source_repo: Path, tmp_path: Path
) -> None:
    candidate, extracted = tmp_path / "candidate", tmp_path / "extracted"
    build(source_repo, candidate)
    with zipfile.ZipFile(candidate / "source.zip") as archive:
        archive.extractall(extracted)
    script = extracted / "scripts/run.sh"
    original = script.read_bytes()
    script.chmod(0o644)
    with pytest.raises(ReleaseError, match="not executable"):
        verify_source_snapshot(extracted)
    script.chmod(0o755)
    assert script.read_bytes() == original
    assert verify_source_snapshot(extracted)["status"] == "verified-against-accompanying-manifest"


def test_extracted_snapshot_refuses_symlinked_directories(
    source_repo: Path, tmp_path: Path
) -> None:
    candidate = tmp_path / "candidate"
    extracted = tmp_path / "extracted"
    build(source_repo, candidate)
    with zipfile.ZipFile(candidate / "source.zip") as archive:
        archive.extractall(extracted)
    (extracted / "scripts").rename(tmp_path / "other-scripts")
    (extracted / "scripts").symlink_to(tmp_path / "other-scripts", target_is_directory=True)
    with pytest.raises(ReleaseError, match="symlink"):
        verify_source_snapshot(extracted)


@pytest.mark.parametrize(
    "new_source",
    ["scripts/new_runtime.py", "notebooks/new_example.ipynb", "research/new_sample.csv"],
)
def test_new_application_source_requires_explicit_allowlist_review(
    source_repo: Path, tmp_path: Path, new_source: str
) -> None:
    path = source_repo / new_source
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# New reviewed source prerequisite\n")
    commit(source_repo)
    with pytest.raises(ReleaseError, match="not explicitly allowlisted"):
        build(source_repo, tmp_path / "refused")


def documentation_file(source: str, destination: str, markdown: str) -> SourceFile:
    return SourceFile(source, destination, "0" * 40, 0o100644, markdown.encode())


def test_export_checks_links_after_public_path_mapping(source_repo: Path, tmp_path: Path) -> None:
    readme = source_repo / "docs/public/README.md"
    readme.write_text("[runtime](../../scripts/run.sh)\n")
    commit(source_repo)
    # This link exists from the source location but leaves the exported root.
    with pytest.raises(ReleaseError, match="unresolved local documentation link"):
        build(source_repo, tmp_path / "bad-mapped-link")
    readme.write_text("[runtime](scripts/run.sh)\n")
    commit(source_repo)
    candidate = tmp_path / "good-mapped-link"
    build(source_repo, candidate)
    report = json.loads((candidate / "manifest.json").read_bytes())["documentation_links"]
    assert report["status"] == "passed"
    assert report["local_link_occurrences"] == 1


def test_link_audit_handles_encoded_paths_references_and_ignores_code_and_footnotes() -> None:
    files = [
        documentation_file(
            "docs/public/README.md",
            "README.md",
            """[guide](<docs/My Guide.md#heading>)
[directory](docs/)
[reference]: docs/My%20Guide.md?view=1#heading
[external](https://example.invalid/unavailable)
[mail](mailto:maintainer@example.invalid)
[local heading](#heading)
`[example](not-a-link.md)`
```md
[example](not-a-link.md)
```
~~~markdown
[example](not-a-link.md)
~~~
[^source]: Historical prose with a https://example.invalid source.
""",
        ),
        documentation_file("docs/My Guide.md", "docs/My Guide.md", "# Guide"),
    ]
    report = audit_documentation_links(files, {})
    assert report["local_link_occurrences"] == 3
    assert report["historical_omissions"] == []


def historical_link_config(file: SourceFile, target: str) -> dict[str, object]:
    return {
        "historical_link_exceptions": [
            {
                "source": file.source,
                "sha256": digest(file.data),
                "missing_destinations": [target],
                "reason": "Immutable history; supporting material is not distributed.",
            }
        ]
    }


def test_historical_omission_is_reported_and_does_not_authorize_other_links() -> None:
    historical = documentation_file(
        "docs/adr/old.md", "docs/adr/old.md", "[support](../internal.md)"
    )
    config = historical_link_config(historical, "docs/internal.md")
    report = audit_documentation_links([historical], config)
    assert report["status"] == "passed-with-declared-historical-omissions"
    assert report["historical_omissions"][0]["omitted_destination"] == "docs/internal.md"
    current = documentation_file("docs/public/README.md", "README.md", "[guide](missing.md)")
    with pytest.raises(ReleaseError, match="unresolved local documentation link"):
        audit_documentation_links([historical, current], config)


def test_changed_historical_bytes_invalidate_link_exception() -> None:
    original = documentation_file("docs/adr/old.md", "docs/adr/old.md", "[support](../old.md)")
    config = historical_link_config(original, "docs/old.md")
    changed = documentation_file(original.source, original.destination, "[new](../different.md)")
    with pytest.raises(ReleaseError, match="stale bytes"):
        audit_documentation_links([changed], config)


def test_historical_exception_requires_exact_target_and_removal_when_resolved() -> None:
    historical = documentation_file("docs/adr/old.md", "docs/adr/old.md", "[support](../old.md)")
    with pytest.raises(ReleaseError, match="unresolved local documentation link"):
        audit_documentation_links([historical], historical_link_config(historical, "docs/new.md"))
    restored = documentation_file("docs/old.md", "docs/old.md", "# Restored support")
    with pytest.raises(ReleaseError, match="unused targets"):
        audit_documentation_links(
            [historical, restored], historical_link_config(historical, "docs/old.md")
        )


def test_notebook_markdown_is_checked_but_code_is_not_treated_as_links() -> None:
    notebook = {
        "cells": [
            {"cell_type": "code", "source": ["print('[code](missing.py)')"]},
            {"cell_type": "markdown", "source": ["[missing figure](../research/lost.png)"]},
        ]
    }
    file = documentation_file("notebooks/demo.ipynb", "notebooks/demo.ipynb", json.dumps(notebook))
    with pytest.raises(ReleaseError, match=r"research/lost\.png"):
        audit_documentation_links([file], {})


def test_a_private_worktree_file_does_not_satisfy_exported_link(
    source_repo: Path, tmp_path: Path
) -> None:
    (source_repo / "private-support.md").write_text("Private implementation notes")
    (source_repo / "docs/public/README.md").write_text("[support](private-support.md)")
    commit(source_repo)
    with pytest.raises(ReleaseError, match="unresolved local documentation link"):
        build(source_repo, tmp_path / "refused-private-reference")
