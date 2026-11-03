#!/usr/bin/env python3
"""Safely acquire or verify the pinned MaleCNS v1.0 source files.

This module deliberately uses only the Python standard library. It never reads
credentials, follows an HTTP URL, or accepts an object outside the explicit
official host/path allowlist below. Downloads are written to a resumable
``.part`` file and promoted atomically only after exact size and SHA-256 checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import ssl
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = (
    REPOSITORY_ROOT / "data" / "manifests" / "male-cns-v1.0.sources.json"
)
EXPECTED_CACHE_RELATIVE = Path("data/cache/male-cns-v1.0")
BUFFER_BYTES = 1024 * 1024
USER_AGENT = "NeuroCraftFly-MaleCNS-Acquirer/1"
CANONICAL_MANIFEST_SHA256 = (
    "9412105f54884a57521bbdf7fe46659caab559ef7877e6a62af95f50e1ee3768"
)


@dataclass(frozen=True, slots=True)
class ResourceRule:
    host: str
    path: str
    filename: str
    generation_required: bool
    kind: str


OFFICIAL_RESOURCES: Mapping[str, ResourceRule] = {
    "annotations": ResourceRule(
        host="storage.googleapis.com",
        path=(
            "/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
            "body-annotations-male-cns-v1.0-minconf-0.5.feather"
        ),
        filename="body-annotations-male-cns-v1.0-minconf-0.5.feather",
        generation_required=True,
        kind="bulk",
    ),
    "neurotransmitters": ResourceRule(
        host="storage.googleapis.com",
        path=(
            "/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
            "body-neurotransmitters-male-cns-v1.0.feather"
        ),
        filename="body-neurotransmitters-male-cns-v1.0.feather",
        generation_required=True,
        kind="bulk",
    ),
    "weights": ResourceRule(
        host="storage.googleapis.com",
        path=(
            "/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
            "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
        ),
        filename="connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        generation_required=True,
        kind="bulk",
    ),
    "download_index": ResourceRule(
        host="male-cns.janelia.org",
        path="/download/",
        filename="male-cns-v1.0-download-index.html",
        generation_required=False,
        kind="metadata",
    ),
    "release_notes": ResourceRule(
        host="male-cns.janelia.org",
        path="/release/",
        filename="male-cns-v1.0-release-notes.html",
        generation_required=False,
        kind="metadata",
    ),
    "license_legalcode": ResourceRule(
        host="creativecommons.org",
        path="/licenses/by/4.0/legalcode.en",
        filename="cc-by-4.0-legalcode.en.html",
        generation_required=False,
        kind="metadata",
    ),
}

EXPECTED_DATASET_FIELDS: Mapping[str, str] = {
    "id": "male-cns:v1.0",
    "release_date": "2026-06-08",
    "license": "CC BY 4.0",
    "license_spdx": "CC-BY-4.0",
    "official_download_url": "https://male-cns.janelia.org/download/",
    "official_release_notes_url": "https://male-cns.janelia.org/release/",
}
EXPECTED_CITATION_FIELDS: Mapping[str, str] = {
    "authors": "Berg et al.",
    "title": (
        "Sexual dimorphism in the complete connectome of the Drosophila male "
        "central nervous system"
    ),
    "doi": "10.1101/2025.10.09.680999",
    "url": "https://doi.org/10.1101/2025.10.09.680999",
    "version": "bioRxiv v2, 2025-10-30",
    "status": "preprint",
}

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
CONTENT_RANGE_RE = re.compile(r"bytes (\d+)-(\d+)/(\d+)\Z")


class AcquisitionError(RuntimeError):
    """Base class for stable, user-facing acquisition failures."""

    code = "acquisition"


class ManifestError(AcquisitionError):
    code = "manifest"


class UnsafeSourceError(AcquisitionError):
    code = "unsafe_source"


class VerificationError(AcquisitionError):
    code = "verification"


class DownloadError(AcquisitionError):
    code = "download"


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    resource_id: str
    kind: str
    filename: str
    url: str
    generation: str | None
    expected_bytes: int
    sha256: str
    purpose: str


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    schema_version: int
    dataset_id: str
    default_cache_dir: str
    sources: tuple[ResourceSpec, ...]
    metadata_documents: tuple[ResourceSpec, ...]
    manifest_path: str = ""
    manifest_sha256: str = ""

    def selected_resources(self, include_metadata: bool) -> tuple[ResourceSpec, ...]:
        if include_metadata:
            return self.sources + self.metadata_documents
        return self.sources


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    resource: ResourceSpec
    path: Path
    status: str
    resumed_from: int

    def receipt(
        self,
        *,
        acquirer_sha256: str,
        manifest_sha256: str,
        observed_at: str,
    ) -> dict[str, Any]:
        return {
            "acquirer_sha256": acquirer_sha256,
            "event": "resource",
            "expected_bytes": self.resource.expected_bytes,
            "expected_sha256": self.resource.sha256,
            "generation": self.resource.generation,
            "kind": self.resource.kind,
            "manifest_sha256": manifest_sha256,
            "observed_at": observed_at,
            "observed_bytes": self.path.stat().st_size,
            "observed_sha256": self.resource.sha256,
            "path": str(self.path),
            "resource_id": self.resource.resource_id,
            "resumed_from": self.resumed_from,
            "status": self.status,
            "url": self.resource.url,
        }


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be a JSON object")
    return value


def _require_string(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{label}.{key} must be a non-empty string")
    return value


def validate_source_url(
    url: str,
    *,
    resource_id: str | None = None,
    generation: str | None = None,
) -> str:
    """Validate an exact HTTPS resource and return its allowlisted ID."""

    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise UnsafeSourceError(f"invalid source URL: {url!r}") from exc

    if parsed.scheme != "https":
        raise UnsafeSourceError("source URL must use HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeSourceError("source URL must not contain credentials")
    if port not in (None, 443):
        raise UnsafeSourceError("source URL must not select a non-default port")
    if parsed.fragment:
        raise UnsafeSourceError("source URL must not contain a fragment")

    candidates: Iterable[tuple[str, ResourceRule]]
    if resource_id is None:
        candidates = OFFICIAL_RESOURCES.items()
    else:
        rule = OFFICIAL_RESOURCES.get(resource_id)
        if rule is None:
            raise UnsafeSourceError(f"resource ID is not allowlisted: {resource_id!r}")
        candidates = ((resource_id, rule),)

    matched_id: str | None = None
    matched_rule: ResourceRule | None = None
    for candidate_id, candidate_rule in candidates:
        if parsed.hostname == candidate_rule.host and parsed.path == candidate_rule.path:
            matched_id = candidate_id
            matched_rule = candidate_rule
            break

    if matched_id is None or matched_rule is None:
        raise UnsafeSourceError(
            f"source host/path is not allowlisted: {parsed.hostname!r}{parsed.path!r}"
        )

    if parsed.query:
        try:
            query = urllib.parse.parse_qs(
                parsed.query, keep_blank_values=True, strict_parsing=True
            )
        except ValueError as exc:
            raise UnsafeSourceError("source URL has an invalid query string") from exc
    else:
        query = {}

    if matched_rule.generation_required:
        values = query.get("generation")
        if set(query) != {"generation"} or values is None or len(values) != 1:
            raise UnsafeSourceError(
                "bulk source URL must contain exactly one generation query parameter"
            )
        if not values[0].isdigit():
            raise UnsafeSourceError("object generation must contain only decimal digits")
        if generation is not None and values[0] != generation:
            raise UnsafeSourceError(
                f"URL generation {values[0]} does not match manifest generation {generation}"
            )
    elif parsed.query:
        raise UnsafeSourceError("metadata source URL must not contain a query string")

    return matched_id


class SameResourceRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Allow redirects only within the same exact allowlisted resource."""

    def redirect_request(  # type: ignore[override]
        self,
        req: urllib.request.Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> urllib.request.Request | None:
        original_id = validate_source_url(req.full_url)
        redirected_id = validate_source_url(newurl)
        original = urllib.parse.urlsplit(req.full_url)
        redirected = urllib.parse.urlsplit(newurl)
        if (
            original_id != redirected_id
            or original.hostname != redirected.hostname
            or original.path != redirected.path
            or original.query != redirected.query
        ):
            raise UnsafeSourceError(
                f"redirect left the original official resource: {newurl!r}"
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def build_safe_opener() -> urllib.request.OpenerDirector:
    """Build a credential-free opener with system proxies explicitly disabled."""

    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
        SameResourceRedirectHandler(),
    )


def _parse_resource(raw: object, *, kind: str, label: str) -> ResourceSpec:
    mapping = _require_mapping(raw, label)
    resource_id = _require_string(mapping, "id", label)
    filename = _require_string(mapping, "filename", label)
    url = _require_string(mapping, "url", label)
    sha256 = _require_string(mapping, "sha256", label)
    purpose = _require_string(mapping, "purpose", label)
    generation_value = mapping.get("generation")
    generation: str | None

    if Path(filename).name != filename or filename in {".", ".."}:
        raise ManifestError(f"{label}.filename must be a plain basename")
    if not SHA256_RE.fullmatch(sha256):
        raise ManifestError(f"{label}.sha256 must be 64 lowercase hexadecimal digits")

    expected_bytes = mapping.get("bytes")
    if (
        not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes <= 0
    ):
        raise ManifestError(f"{label}.bytes must be a positive integer")

    if generation_value is None:
        generation = None
    elif isinstance(generation_value, str) and generation_value.isdigit():
        generation = generation_value
    else:
        raise ManifestError(f"{label}.generation must be a decimal string or null")

    rule = OFFICIAL_RESOURCES.get(resource_id)
    if rule is None or rule.kind != kind:
        raise ManifestError(f"{label}.id is not an allowlisted {kind} resource")
    validate_source_url(url, resource_id=resource_id, generation=generation)
    if rule.filename != filename:
        raise ManifestError(
            f"{label}.filename does not match the allowlisted source path"
        )
    if rule.generation_required and generation is None:
        raise ManifestError(f"{label}.generation is required")
    if not rule.generation_required and generation is not None:
        raise ManifestError(f"{label}.generation must be null for metadata")

    return ResourceSpec(
        resource_id=resource_id,
        kind=kind,
        filename=filename,
        url=url,
        generation=generation,
        expected_bytes=expected_bytes,
        sha256=sha256,
        purpose=purpose,
    )


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> DatasetManifest:
    try:
        manifest_bytes = path.read_bytes()
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if manifest_sha256 != CANONICAL_MANIFEST_SHA256:
            raise ManifestError(
                "canonical manifest SHA-256 mismatch: expected "
                f"{CANONICAL_MANIFEST_SHA256}, observed {manifest_sha256} at {path}"
            )
        payload = json.loads(manifest_bytes)
    except ManifestError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc

    root = _require_mapping(payload, "manifest")
    if root.get("schema_version") != 1:
        raise ManifestError("manifest.schema_version must equal 1")

    dataset = _require_mapping(root.get("dataset"), "manifest.dataset")
    for key, expected in EXPECTED_DATASET_FIELDS.items():
        observed = dataset.get(key)
        if observed != expected:
            raise ManifestError(
                f"manifest.dataset.{key} must equal {expected!r}, got {observed!r}"
            )
    citation = _require_mapping(dataset.get("citation"), "manifest.dataset.citation")
    for key, expected in EXPECTED_CITATION_FIELDS.items():
        observed = citation.get(key)
        if observed != expected:
            raise ManifestError(
                "manifest.dataset.citation."
                f"{key} must equal {expected!r}, got {observed!r}"
            )

    default_cache_dir = _require_string(
        root, "default_cache_dir", "manifest"
    )
    if Path(default_cache_dir) != EXPECTED_CACHE_RELATIVE:
        raise ManifestError(
            "manifest.default_cache_dir must be data/cache/male-cns-v1.0"
        )

    raw_sources = root.get("sources")
    raw_metadata = root.get("metadata_documents")
    if not isinstance(raw_sources, list):
        raise ManifestError("manifest.sources must be a JSON array")
    if not isinstance(raw_metadata, list):
        raise ManifestError("manifest.metadata_documents must be a JSON array")

    sources = tuple(
        _parse_resource(raw, kind="bulk", label=f"manifest.sources[{index}]")
        for index, raw in enumerate(raw_sources)
    )
    metadata = tuple(
        _parse_resource(
            raw, kind="metadata", label=f"manifest.metadata_documents[{index}]"
        )
        for index, raw in enumerate(raw_metadata)
    )

    expected_bulk_ids = tuple(
        resource_id
        for resource_id, rule in OFFICIAL_RESOURCES.items()
        if rule.kind == "bulk"
    )
    expected_metadata_ids = tuple(
        resource_id
        for resource_id, rule in OFFICIAL_RESOURCES.items()
        if rule.kind == "metadata"
    )
    if tuple(resource.resource_id for resource in sources) != expected_bulk_ids:
        raise ManifestError(
            f"manifest.sources must contain exactly, in order: {expected_bulk_ids}"
        )
    if tuple(resource.resource_id for resource in metadata) != expected_metadata_ids:
        raise ManifestError(
            "manifest.metadata_documents must contain exactly, in order: "
            f"{expected_metadata_ids}"
        )

    return DatasetManifest(
        schema_version=1,
        dataset_id=EXPECTED_DATASET_FIELDS["id"],
        default_cache_dir=default_cache_dir,
        sources=sources,
        metadata_documents=metadata,
        manifest_path=str(path),
        manifest_sha256=manifest_sha256,
    )


def _validate_cache_directory(path: Path) -> None:
    if path.is_symlink():
        raise VerificationError(f"refusing symbolic-link cache directory: {path}")
    try:
        metadata = path.stat(follow_symlinks=False)
    except FileNotFoundError as exc:
        raise VerificationError(f"cache directory is missing: {path}") from exc
    except OSError as exc:
        raise VerificationError(f"cannot inspect cache directory {path}: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise VerificationError(f"cache path is not a directory: {path}")
    if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
        raise VerificationError(f"cache directory is not owned by the current user: {path}")
    if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise VerificationError(f"cache directory is group/world writable: {path}")


def _open_readonly_no_follow(path: Path) -> BinaryIO:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise VerificationError(f"cannot safely open {path}: {exc}") from exc
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise VerificationError(
            f"artifact must be a single-link regular file: {path}"
        )
    return os.fdopen(descriptor, "rb")


def _open_part(path: Path, *, append: bool) -> BinaryIO:
    flags = os.O_WRONLY | os.O_CREAT
    flags |= os.O_APPEND if append else os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise DownloadError(f"cannot safely open partial file {path}: {exc}") from exc
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise DownloadError(
            f"partial artifact must be a single-link regular file: {path}"
        )
    return os.fdopen(descriptor, "ab" if append else "wb")


def _hash_stream(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = handle.read(BUFFER_BYTES)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


def verify_path(path: Path, resource: ResourceSpec) -> str:
    if path.is_symlink():
        raise VerificationError(f"refusing symbolic-link artifact: {path}")
    try:
        handle = _open_readonly_no_follow(path)
    except FileNotFoundError as exc:
        raise VerificationError(f"required artifact is missing: {path}") from exc
    with handle:
        observed_bytes = os.fstat(handle.fileno()).st_size
        if observed_bytes != resource.expected_bytes:
            raise VerificationError(
                f"size mismatch for {resource.resource_id}: expected "
                f"{resource.expected_bytes}, observed {observed_bytes} at {path}"
            )
        observed_sha256 = _hash_stream(handle)
    if observed_sha256 != resource.sha256:
        raise VerificationError(
            f"SHA-256 mismatch for {resource.resource_id}: expected "
            f"{resource.sha256}, observed {observed_sha256} at {path}"
        )
    return observed_sha256


def _resource_paths(cache_dir: Path, resource: ResourceSpec) -> tuple[Path, Path]:
    target = cache_dir / resource.filename
    partial = cache_dir / f"{resource.filename}.part"
    return target, partial


def verify_source(cache_dir: Path, resource: ResourceSpec) -> AcquisitionResult:
    _validate_cache_directory(cache_dir)
    target, _ = _resource_paths(cache_dir, resource)
    verify_path(target, resource)
    return AcquisitionResult(resource, target, "verified", 0)


def _response_status(response: Any) -> int:
    status = getattr(response, "status", None)
    if status is None:
        status = response.getcode()
    if not isinstance(status, int):
        raise DownloadError("HTTP response did not provide an integer status")
    return status


def _header(response: Any, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get(name)
    return str(value) if value is not None else None


def _validate_response(
    response: Any, resource: ResourceSpec, *, requested_offset: int
) -> tuple[int, bool]:
    final_url = response.geturl()
    validate_source_url(
        final_url, resource_id=resource.resource_id, generation=resource.generation
    )
    status = _response_status(response)
    encoding = _header(response, "Content-Encoding")
    if encoding is not None and encoding.lower() != "identity":
        raise DownloadError(f"unexpected Content-Encoding: {encoding!r}")

    append = status == 206
    if append:
        if requested_offset <= 0:
            raise DownloadError("server returned partial content without a range request")
        content_range = _header(response, "Content-Range")
        match = CONTENT_RANGE_RE.fullmatch(content_range or "")
        if match is None:
            raise DownloadError("partial response has an invalid Content-Range")
        start, end, total = (int(value) for value in match.groups())
        if (
            start != requested_offset
            or end != resource.expected_bytes - 1
            or total != resource.expected_bytes
        ):
            raise DownloadError(
                "partial response range does not match the requested offset and "
                "manifest size"
            )
        expected_response_bytes = resource.expected_bytes - requested_offset
    elif status == 200:
        expected_response_bytes = resource.expected_bytes
    else:
        raise DownloadError(f"unexpected HTTP status {status}")

    content_length = _header(response, "Content-Length")
    if content_length is not None:
        try:
            parsed_length = int(content_length)
        except ValueError as exc:
            raise DownloadError("HTTP Content-Length is not an integer") from exc
        if parsed_length != expected_response_bytes:
            raise DownloadError(
                f"HTTP Content-Length mismatch: expected {expected_response_bytes}, "
                f"observed {parsed_length}"
            )
    return status, append


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _promote(partial: Path, target: Path, resource: ResourceSpec) -> None:
    try:
        os.replace(partial, target)
    except OSError as exc:
        raise VerificationError(
            f"verified artifact could not be atomically promoted to {target}: {exc}"
        ) from exc
    _fsync_directory(target.parent)
    verify_path(target, resource)


def acquire_source(
    cache_dir: Path,
    resource: ResourceSpec,
    *,
    opener: Any,
    timeout: float,
) -> AcquisitionResult:
    validate_source_url(
        resource.url, resource_id=resource.resource_id, generation=resource.generation
    )
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DownloadError(f"cannot create cache directory {cache_dir}: {exc}") from exc
    _validate_cache_directory(cache_dir)

    target, partial = _resource_paths(cache_dir, resource)
    if target.exists() or target.is_symlink():
        return verify_source(cache_dir, resource)
    if partial.is_symlink():
        raise VerificationError(f"refusing symbolic-link partial file: {partial}")
    if partial.exists() and not partial.is_file():
        raise VerificationError(f"partial path is not a regular file: {partial}")

    try:
        requested_offset = partial.stat().st_size
    except FileNotFoundError:
        requested_offset = 0
    except OSError as exc:
        raise DownloadError(f"cannot stat partial file {partial}: {exc}") from exc
    if requested_offset > resource.expected_bytes:
        raise VerificationError(
            f"partial file is larger than the manifest size: {requested_offset} > "
            f"{resource.expected_bytes} at {partial}"
        )
    if requested_offset == resource.expected_bytes:
        verify_path(partial, resource)
        _promote(partial, target, resource)
        return AcquisitionResult(
            resource, target, "promoted-part", requested_offset
        )

    headers = {
        "Accept": "*/*",
        "Accept-Encoding": "identity",
        "User-Agent": USER_AGENT,
    }
    if requested_offset:
        headers["Range"] = f"bytes={requested_offset}-"
    request = urllib.request.Request(
        resource.url, headers=headers, method="GET"
    )

    try:
        response = opener.open(request, timeout=timeout)
        with response:
            status, append = _validate_response(
                response, resource, requested_offset=requested_offset
            )
            write_offset = requested_offset if append else 0
            with _open_part(partial, append=append) as handle:
                written = write_offset
                while True:
                    chunk = response.read(BUFFER_BYTES)
                    if not chunk:
                        break
                    if written + len(chunk) > resource.expected_bytes:
                        raise DownloadError(
                            "response exceeds the manifest byte count; extra bytes were "
                            "not written"
                        )
                    handle.write(chunk)
                    written += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
    except AcquisitionError:
        raise
    except urllib.error.HTTPError as exc:
        raise DownloadError(f"HTTP error {exc.code} for {resource.resource_id}") from exc
    except urllib.error.URLError as exc:
        raise DownloadError(
            f"network error for {resource.resource_id}: {exc.reason}"
        ) from exc
    except OSError as exc:
        raise DownloadError(f"I/O error for {resource.resource_id}: {exc}") from exc

    try:
        observed_bytes = partial.stat().st_size
    except OSError as exc:
        raise VerificationError(f"cannot stat completed partial file: {exc}") from exc
    if observed_bytes != resource.expected_bytes:
        raise DownloadError(
            f"incomplete response for {resource.resource_id}: expected "
            f"{resource.expected_bytes}, retained {observed_bytes} bytes in {partial}"
        )

    verify_path(partial, resource)
    _promote(partial, target, resource)
    if status == 206:
        result_status = "resumed"
    elif requested_offset:
        result_status = "restarted"
    else:
        result_status = "downloaded"
    return AcquisitionResult(resource, target, result_status, requested_offset)


def _safe_stat_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def error_receipt(
    resource: ResourceSpec, cache_dir: Path, error: AcquisitionError
) -> dict[str, Any]:
    target, partial = _resource_paths(cache_dir, resource)
    return {
        "error_type": error.code,
        "event": "resource",
        "expected_bytes": resource.expected_bytes,
        "expected_sha256": resource.sha256,
        "kind": resource.kind,
        "message": str(error),
        "part_bytes": _safe_stat_size(partial),
        "path": str(target),
        "resource_id": resource.resource_id,
        "status": "error",
        "target_bytes": _safe_stat_size(target),
    }


def process_resources(
    manifest: DatasetManifest,
    cache_dir: Path,
    *,
    verify_only: bool,
    include_metadata: bool,
    opener: Any | None,
    timeout: float,
) -> tuple[list[dict[str, Any]], int]:
    selected = manifest.selected_resources(include_metadata)
    observed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    acquirer_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if not verify_only and opener is None:
        opener = build_safe_opener()

    receipts: list[dict[str, Any]] = []
    failures = 0
    for resource in selected:
        try:
            if verify_only:
                result = verify_source(cache_dir, resource)
            else:
                result = acquire_source(
                    cache_dir, resource, opener=opener, timeout=timeout
                )
            receipts.append(
                result.receipt(
                    acquirer_sha256=acquirer_sha256,
                    manifest_sha256=manifest.manifest_sha256,
                    observed_at=observed_at,
                )
            )
        except AcquisitionError as exc:
            failures += 1
            receipts.append(error_receipt(resource, cache_dir, exc))
    return receipts, failures


def _emit(receipt: Mapping[str, Any], *, error: bool = False) -> None:
    stream = sys.stderr if error else sys.stdout
    print(json.dumps(receipt, sort_keys=True), file=stream, flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download or verify the three pinned MaleCNS v1.0 Feather sources. "
            "Official metadata documents are opt-in."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="pinned source manifest",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="artifact cache (default: data/cache/male-cns-v1.0)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="perform no network I/O or writes; verify existing cached files",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="also cache/verify the download index, release notes, and license",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="per-operation network timeout in seconds (default: 60)",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, opener: Any | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not (args.timeout > 0):
        _emit(
            {
                "error_type": "arguments",
                "event": "acquisition",
                "message": "--timeout must be greater than zero",
                "status": "error",
            },
            error=True,
        )
        return 2

    try:
        manifest = load_manifest(args.manifest)
    except AcquisitionError as exc:
        _emit(
            {
                "error_type": exc.code,
                "event": "manifest",
                "message": str(exc),
                "path": str(args.manifest),
                "status": "error",
            },
            error=True,
        )
        return 2

    cache_dir = (
        args.cache_dir
        if args.cache_dir is not None
        else REPOSITORY_ROOT / manifest.default_cache_dir
    )
    receipts, failures = process_resources(
        manifest,
        cache_dir,
        verify_only=args.verify_only,
        include_metadata=args.include_metadata,
        opener=opener,
        timeout=args.timeout,
    )
    for receipt in receipts:
        _emit(receipt, error=receipt["status"] == "error")

    selected_count = len(manifest.selected_resources(args.include_metadata))
    summary = {
        "acquirer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cache_dir": str(cache_dir),
        "dataset": manifest.dataset_id,
        "event": "summary",
        "failed": failures,
        "include_metadata": args.include_metadata,
        "manifest_path": manifest.manifest_path,
        "manifest_sha256": manifest.manifest_sha256,
        "mode": "verify-only" if args.verify_only else "acquire",
        "selected": selected_count,
        "status": "ok" if failures == 0 else "error",
        "succeeded": selected_count - failures,
    }
    _emit(summary, error=failures != 0)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
