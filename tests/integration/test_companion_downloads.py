from __future__ import annotations

import hashlib
import io
import os
import socket
import stat
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import companion_downloads as downloads


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    # Each download fixture replaces urlopen. Preserve that transport seam when
    # production uses an opener with its redirect validator as well.
    monkeypatch.setattr(
        downloads.urllib.request,
        "build_opener",
        lambda *handlers: SimpleNamespace(
            open=lambda *args, **kwargs: downloads.urllib.request.urlopen(*args, **kwargs)
        ),
    )
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *args, **kwargs: pytest.fail("download fixture attempted real networking"),
    )


class Response(io.BytesIO):
    def __init__(self, data: bytes, url: str = "https://github.com/example/runtime") -> None:
        super().__init__(data)
        self.url = url


def pin(data: bytes) -> dict[str, Any]:
    return {
        "url": "https://github.com/example/runtime",
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def test_download_promotes_only_exact_pinned_bytes_and_reuses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"small test runtime archive\n"
    target = tmp_path / "cache/runtime.archive"
    calls = []

    def request(url: Any, **kwargs: Any) -> Response:
        calls.append(url.full_url)
        return Response(payload)

    monkeypatch.setattr(downloads.urllib.request, "urlopen", request)
    downloads.download(pin(payload), target, progress=lambda message: None)
    assert target.read_bytes() == payload
    assert not list(target.parent.glob(".download-*"))
    timestamp = target.stat().st_mtime_ns
    downloads.download(pin(payload), target, progress=lambda message: None)
    assert calls == [pin(payload)["url"]], "verified cache reuse attempted another request"
    assert target.stat().st_mtime_ns == timestamp


@pytest.mark.parametrize("failure", ["hash", "short", "long", "network", "redirect"])
def test_failed_download_never_becomes_cache_or_leaves_partial_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    expected = b"verified"
    actual = {"hash": b"modified", "short": b"short", "long": b"too many bytes"}.get(
        failure, expected
    )
    target = tmp_path / "cache/runtime.archive"

    def request(*args: Any, **kwargs: Any) -> Response:
        if failure == "network":
            raise OSError("synthetic offline fixture")
        return Response(
            actual,
            "https://unapproved.invalid/file" if failure == "redirect" else pin(expected)["url"],
        )

    monkeypatch.setattr(downloads.urllib.request, "urlopen", request)
    with pytest.raises((ValueError, OSError)):
        downloads.download(pin(expected), target, progress=lambda message: None)
    assert not target.exists()
    assert not list(target.parent.glob(".download-*"))


@pytest.mark.parametrize("existing", [b"same size differs", b"short"])
def test_cache_mismatch_is_retained_and_never_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bytes
) -> None:
    target = tmp_path / "runtime.archive"
    target.write_bytes(existing)
    before = target.stat().st_mtime_ns
    monkeypatch.setattr(
        downloads.urllib.request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("mismatched cache caused a network request"),
    )
    with pytest.raises(ValueError, match="differs"):
        downloads.download(pin(b"expected runtime"), target, progress=lambda message: None)
    assert target.read_bytes() == existing
    assert target.stat().st_mtime_ns == before
    assert not list(tmp_path.glob(".download-*"))


@pytest.mark.parametrize(
    "mutation",
    [
        {"bytes": True},
        {"bytes": 0},
        {"bytes": -1},
        {"bytes": 10**12},
        {"sha256": "not a digest"},
        {"url": "http://github.com/runtime"},
        {"url": "https://github.com.evil.invalid/runtime"},
        {"url": "https://user:password@github.com/runtime"},
    ],
)
def test_invalid_pin_has_no_network_or_filesystem_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: dict[str, Any]
) -> None:
    metadata = pin(b"fixture") | mutation
    monkeypatch.setattr(
        downloads.urllib.request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("invalid pin attempted a download"),
    )
    with pytest.raises(ValueError):
        downloads.download(metadata, tmp_path / "absent/runtime.archive")
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(os.name == "nt", reason="creating this symlink requires Windows Developer Mode")
@pytest.mark.parametrize("dangling", [False, True])
def test_cache_symlinks_are_refused_without_reading_or_replacing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dangling: bool
) -> None:
    outside = tmp_path / "outside"
    if not dangling:
        outside.write_bytes(b"fixture")
    target = tmp_path / "runtime.archive"
    target.symlink_to(outside)
    original_open = Path.open

    def refuse_link_read(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == target:
            pytest.fail("the cache opened a symlink before rejecting it")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", refuse_link_read)
    monkeypatch.setattr(
        downloads.urllib.request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("symlink cache attempted a download"),
    )
    with pytest.raises(ValueError):
        downloads.download(pin(b"fixture"), target)
    assert target.is_symlink()
    assert outside.exists() is (not dangling)
    if not dangling:
        assert outside.read_bytes() == b"fixture"


def test_cache_created_during_download_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"runtime"
    target = tmp_path / "runtime.archive"

    class RacingResponse(Response):
        def read(self, size: int = -1) -> bytes:
            if not target.exists():
                target.write_bytes(b"another operation owns this file")
            return super().read(size)

    monkeypatch.setattr(
        downloads.urllib.request, "urlopen", lambda *args, **kwargs: RacingResponse(payload)
    )
    with pytest.raises(ValueError):
        downloads.download(pin(payload), target)
    assert target.read_bytes() == b"another operation owns this file"
    assert not list(tmp_path.glob(".download-*"))


@pytest.mark.parametrize("kind", ["zip", "tar"])
def test_extract_promotes_a_complete_contained_runtime(tmp_path: Path, kind: str) -> None:
    archive = tmp_path / ("runtime." + kind)
    if kind == "zip":
        with zipfile.ZipFile(archive, "w") as package:
            package.writestr("runtime/bin/tool", b"executable fixture")
            package.writestr("runtime/LICENSE", b"fixture license")
    else:
        with tarfile.open(archive, "w") as package:
            for name, data in (
                ("runtime/bin/tool", b"executable fixture"),
                ("runtime/LICENSE", b"fixture license"),
            ):
                entry = tarfile.TarInfo(name)
                entry.size = len(data)
                entry.mode = 0o755 if name.endswith("tool") else 0o644
                package.addfile(entry, io.BytesIO(data))
    target = tmp_path / "installed"
    downloads.extract(archive, target)
    assert (target / "runtime/bin/tool").read_bytes() == b"executable fixture"
    assert (target / "runtime/LICENSE").read_bytes() == b"fixture license"
    assert not list(tmp_path.glob(".runtime-*"))
    with pytest.raises(ValueError):
        downloads.extract(archive, target)
    assert (target / "runtime/bin/tool").read_bytes() == b"executable fixture"


@pytest.mark.parametrize("kind", ["zip", "tar"])
@pytest.mark.parametrize(
    "member", ["../outside", "/outside", "C:/outside", "safe/../outside", "safe\\..\\outside"]
)
def test_archive_traversal_has_no_promoted_or_outside_files(
    tmp_path: Path, kind: str, member: str
) -> None:
    archive = tmp_path / ("runtime." + kind)
    if kind == "zip":
        with zipfile.ZipFile(archive, "w") as package:
            package.writestr("safe/first", b"must not survive rejection")
            package.writestr(member, b"escape")
    else:
        with tarfile.open(archive, "w") as package:
            entry = tarfile.TarInfo(member)
            entry.size = 6
            package.addfile(entry, io.BytesIO(b"escape"))
    with pytest.raises(ValueError):
        downloads.extract(archive, tmp_path / "installed")
    assert not (tmp_path / "installed").exists()
    assert not (tmp_path / "outside").exists()
    assert not list(tmp_path.glob(".runtime-*"))


def test_zip_symlink_and_device_entries_are_refused(tmp_path: Path) -> None:
    for index, mode in enumerate((stat.S_IFLNK | 0o777, stat.S_IFCHR | 0o600)):
        archive = tmp_path / f"special-{index}.zip"
        with zipfile.ZipFile(archive, "w") as package:
            entry = zipfile.ZipInfo("bin/redirect")
            entry.create_system = 3
            entry.external_attr = mode << 16
            package.writestr(entry, "../../outside")
        with pytest.raises(ValueError):
            downloads.extract(archive, tmp_path / "installed")
        assert not (tmp_path / "installed").exists()
        assert not list(tmp_path.glob(".runtime-*"))


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE])
@pytest.mark.parametrize("target", ["../../outside", "/outside", "C:/outside"])
def test_tar_links_cannot_escape_stage(tmp_path: Path, kind: bytes, target: str) -> None:
    archive = tmp_path / "link.tar"
    with tarfile.open(archive, "w") as package:
        entry = tarfile.TarInfo("bin/redirect")
        entry.type = kind
        entry.linkname = target
        package.addfile(entry)
    with pytest.raises((ValueError, tarfile.TarError)):
        downloads.extract(archive, tmp_path / "installed")
    assert not (tmp_path / "installed").exists()
    assert not (tmp_path / "outside").exists()
    assert not list(tmp_path.glob(".runtime-*"))


@pytest.mark.parametrize(
    "system,machine,expected",
    [
        ("Windows", "AMD64", "windows-x86_64"),
        ("Darwin", "arm64", "macos-arm64"),
        ("Linux", "x86_64", "linux-x86_64"),
        ("Linux", "aarch64", "linux-arm64"),
    ],
)
def test_platform_key_normalizes_runtime_pin_lookup(
    monkeypatch: pytest.MonkeyPatch, system: str, machine: str, expected: str
) -> None:
    monkeypatch.setattr(downloads.platform, "system", lambda: system)
    monkeypatch.setattr(downloads.platform, "machine", lambda: machine)
    assert downloads.platform_key() == expected


@pytest.mark.parametrize("system,machine", [("Unknown", "x86_64"), ("Windows", "mips")])
def test_unknown_platform_cannot_select_a_different_architecture_pin(
    monkeypatch: pytest.MonkeyPatch, system: str, machine: str
) -> None:
    monkeypatch.setattr(downloads.platform, "system", lambda: system)
    monkeypatch.setattr(downloads.platform, "machine", lambda: machine)
    with pytest.raises(ValueError):
        downloads.platform_key()


@pytest.mark.parametrize(
    "url",
    [
        "https://unapproved.invalid/runtime",
        "http://github.com/runtime",
        "https://user@github.com/runtime",
    ],
)
def test_redirect_policy_rejects_unapproved_destination_before_following(url: str) -> None:
    request = downloads.urllib.request.Request("https://github.com/example/runtime")
    with pytest.raises(ValueError):
        downloads.RuntimeRedirect().redirect_request(request, None, 302, "Found", {}, url)


def test_cancelled_download_creates_no_cache_or_network_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        downloads.urllib.request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("cancelled download attempted networking"),
    )
    with pytest.raises(InterruptedError):
        downloads.download(pin(b"fixture"), tmp_path / "absent/runtime", cancelled=lambda: True)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("when", ["partial_stream", "before_promotion"])
def test_download_cancellation_removes_temporary_bytes_and_prevents_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, when: str
) -> None:
    payload = b"complete fixture payload"
    target = tmp_path / "runtime.archive"
    stopped = [False]

    class CancelledResponse(Response):
        def read(self, size: int = -1) -> bytes:
            data = super().read(min(size, 5))
            if (when == "partial_stream" and self.tell() > 5) or (
                when == "before_promotion" and data == b""
            ):
                stopped[0] = True
            return data

    monkeypatch.setattr(
        downloads.urllib.request, "urlopen", lambda *args, **kwargs: CancelledResponse(payload)
    )
    with pytest.raises(InterruptedError):
        downloads.download(pin(payload), target, cancelled=lambda: stopped[0])
    assert not target.exists()
    assert not list(tmp_path.glob(".download-*"))


@pytest.mark.parametrize("kind", ["zip", "tar"])
def test_extraction_cancelled_after_writing_member_never_promotes_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    archive = tmp_path / ("runtime." + kind)
    stopped = [False]
    if kind == "zip":
        with zipfile.ZipFile(archive, "w") as package:
            package.writestr("runtime/member", b"fixture")
        original_zip = zipfile.ZipFile.extract

        def extract_zip(package: zipfile.ZipFile, *args: Any, **kwargs: Any) -> str:
            result = original_zip(package, *args, **kwargs)
            stopped[0] = True
            return result

        monkeypatch.setattr(zipfile.ZipFile, "extract", extract_zip)
    else:
        with tarfile.open(archive, "w") as package:
            entry = tarfile.TarInfo("runtime/member")
            entry.size = 7
            package.addfile(entry, io.BytesIO(b"fixture"))
        original_tar = tarfile.TarFile.extractall

        def extract_tar(package: tarfile.TarFile, *args: Any, **kwargs: Any) -> None:
            original_tar(package, *args, **kwargs)
            stopped[0] = True

        monkeypatch.setattr(tarfile.TarFile, "extractall", extract_tar)
    with pytest.raises(InterruptedError):
        downloads.extract(archive, tmp_path / "installed", cancelled=lambda: stopped[0])
    assert not (tmp_path / "installed").exists()
    assert not list(tmp_path.glob(".runtime-*"))


@pytest.mark.parametrize("kind", ["zip", "tar"])
def test_excessive_declared_expansion_is_refused_before_extracting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    archive = tmp_path / ("runtime." + kind)
    if kind == "zip":
        with zipfile.ZipFile(archive, "w") as package:
            package.writestr("runtime/oversized", b"small fixture")
        original = zipfile.ZipFile.infolist

        def oversized(package: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
            entries = original(package)
            entries[0].file_size = 3 * 1024**3
            return entries

        monkeypatch.setattr(zipfile.ZipFile, "infolist", oversized)
        monkeypatch.setattr(
            zipfile.ZipFile, "extract", lambda *args, **kwargs: pytest.fail("expanded unsafe ZIP")
        )
    else:
        with tarfile.open(archive, "w"):
            pass
        entry = tarfile.TarInfo("runtime/oversized")
        entry.size = 3 * 1024**3
        monkeypatch.setattr(tarfile.TarFile, "getmembers", lambda package: [entry])
        monkeypatch.setattr(
            tarfile.TarFile,
            "extractall",
            lambda *args, **kwargs: pytest.fail("expanded unsafe TAR"),
        )
    with pytest.raises(ValueError):
        downloads.extract(archive, tmp_path / "installed")
    assert not (tmp_path / "installed").exists()
    assert not list(tmp_path.glob(".runtime-*"))


@pytest.mark.skipif(os.name == "nt", reason="Windows test users may not have symlink privilege")
def test_tar_internal_relative_runtime_link_is_preserved_and_contained(tmp_path: Path) -> None:
    archive = tmp_path / "runtime.tar"
    with tarfile.open(archive, "w") as package:
        entry = tarfile.TarInfo("lib/java-real")
        entry.size = 7
        package.addfile(entry, io.BytesIO(b"fixture"))
        link = tarfile.TarInfo("bin/java")
        link.type = tarfile.SYMTYPE
        link.linkname = "../lib/java-real"
        package.addfile(link)
    target = tmp_path / "installed"
    downloads.extract(archive, target)
    assert (target / "bin/java").is_symlink()
    assert (target / "bin/java").read_bytes() == b"fixture"
    assert (target / "bin/java").resolve().is_relative_to(target)
