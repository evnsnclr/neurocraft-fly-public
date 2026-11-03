"""Hash-verified, bounded runtime downloads and contained archive extraction."""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import stat
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

ALLOWED_HOSTS = frozenset(
    {
        "github.com",
        "release-assets.githubusercontent.com",
        "objects.githubusercontent.com",
        "www.sqlite.org",
    }
)


def platform_key() -> str:
    system = {"Darwin": "macos", "Windows": "windows", "Linux": "linux"}.get(platform.system())
    machine = platform.machine().lower()
    arch = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "arm64", "aarch64": "arm64"}.get(
        machine
    )
    if system is None or arch is None:
        raise ValueError("This operating system or processor has no companion bundle.")
    return f"{system}-{arch}"


def _allowed_url(url: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in ALLOWED_HOSTS
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
    ):
        raise ValueError("Runtime URL must use an approved HTTPS upstream host.")


class RuntimeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> urllib.request.Request | None:
        _allowed_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(
    pin: Mapping[str, Any],
    destination: Path,
    *,
    progress: Callable[[str], None] = print,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    """Download an exact pinned archive; a failed attempt never becomes the cache."""
    url, expected, size = pin["url"], pin["sha256"], pin["bytes"]
    if (
        not isinstance(url, str)
        or not isinstance(expected, str)
        or not re.fullmatch("[a-f0-9]{64}", expected)
    ):
        raise ValueError("Invalid runtime URL/hash pin.")
    if type(size) is not int or not 0 < size <= 512 * 1024 * 1024:
        raise ValueError("Runtime size pin is missing or exceeds the download bound.")
    _allowed_url(url)
    _cancelled(cancelled)
    if destination.is_symlink():
        raise ValueError("Runtime cache must not be a symlink.")
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise ValueError("Runtime cache must be a regular file.")
        with destination.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if destination.is_symlink() or destination.stat().st_size != size or actual != expected:
            raise ValueError(
                "Existing runtime archive differs from the pin; keep it for diagnosis."
            )
        progress("Reusing a hash-verified runtime archive.")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".download-", dir=destination.parent)
    temporary = Path(name)
    try:
        digest = hashlib.sha256()
        total = 0
        request = urllib.request.Request(url, headers={"User-Agent": "NeuroCraftFly-Companion/0.1"})
        deadline = time.monotonic() + 1200
        with (
            os.fdopen(fd, "wb") as output,
            urllib.request.build_opener(RuntimeRedirect()).open(request, timeout=10) as response,
        ):
            _allowed_url(response.url)
            while block := response.read(1024 * 1024):
                _cancelled(cancelled)
                if time.monotonic() > deadline:
                    raise TimeoutError("Runtime download exceeded 20 minutes; retry setup.")
                total += len(block)
                if total > size:
                    raise ValueError("Runtime download exceeds its pinned size.")
                output.write(block)
                digest.update(block)
                if total % (16 * 1024 * 1024) == 0:
                    progress(
                        f"Runtime download: {total // (1024 * 1024)} / {size // (1024 * 1024)} MiB"
                    )
            output.flush()
            os.fsync(output.fileno())
        _cancelled(cancelled)
        if total != size or digest.hexdigest() != expected:
            raise ValueError("Runtime archive checksum or size does not match its pin.")
        if destination.exists():
            raise ValueError("Runtime cache appeared during download; refusing to overwrite it.")
        temporary.rename(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _member(name: str) -> None:
    path = PurePosixPath(name)
    if "\\" in name or ":" in name or path.is_absolute() or ".." in path.parts:
        raise ValueError("Archive member escapes the installation directory.")


def _cancelled(callback: Callable[[], bool] | None) -> None:
    if callback is not None and callback():
        raise InterruptedError("Runtime setup cancelled.")


def extract(
    archive: Path, destination: Path, *, cancelled: Callable[[], bool] | None = None
) -> None:
    """Extract a verified upstream archive into a fresh, atomically promoted folder."""
    _cancelled(cancelled)
    if destination.exists() or destination.is_symlink():
        raise ValueError("Runtime extraction requires a new directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".runtime-", dir=destination.parent))
    try:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as package:
                for entry in package.infolist():
                    _member(entry.filename)
                    mode = entry.external_attr >> 16
                    if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                        raise ValueError("Special ZIP entries are not supported.")
                if sum(e.file_size for e in package.infolist()) > 2 * 1024**3:
                    raise ValueError("Runtime archive expands beyond its size bound.")
                for entry in package.infolist():
                    _cancelled(cancelled)
                    package.extract(entry, stage)
        else:
            with tarfile.open(archive, "r:*") as tar_package:
                entries = tar_package.getmembers()
                for tar_entry in entries:
                    _member(tar_entry.name)
                    if (tar_entry.issym() or tar_entry.islnk()) and (
                        "\\" in tar_entry.linkname or ":" in tar_entry.linkname
                    ):
                        raise ValueError("Invalid runtime link path.")
                if sum(e.size for e in entries) > 2 * 1024**3:
                    raise ValueError("Runtime archive expands beyond its size bound.")

                def safe_filter(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo | None:
                    _cancelled(cancelled)
                    return tarfile.data_filter(member, path)

                tar_package.extractall(stage, filter=safe_filter)
        _cancelled(cancelled)
        stage.rename(destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
