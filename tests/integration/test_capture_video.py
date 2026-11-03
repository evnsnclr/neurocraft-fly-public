from __future__ import annotations

import json
import shutil
import struct
import subprocess
import zlib
from pathlib import Path
from typing import Any

import pytest

from scripts.export_capture_video import capture_manifest, concat_document, export


def _capture(directory: Path) -> list[dict[str, Any]]:
    images = directory / "screenshots"
    images.mkdir()
    rows = []
    for frame, nanos in enumerate((1_000_000_000, 1_050_000_000, 1_600_000_000), 1):
        filename = f"frame-{frame:04d}.png"
        # Header-only fixture exercises manifest validation; FFmpeg does full image decoding.
        (images / filename).write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR" + struct.pack(">II", 32, 32)
        )
        request = {
            "schema": "neurocraft.live_frame_capture.v1",
            "frame": frame,
            "file": f"screenshots/{filename}",
            "request_wall_elapsed_ns": nanos,
            "request_game_tick": frame + 100,
            "status": "requested",
            "status_elapsed_since_request_ns": 0,
        }
        rows.extend([request, {**request, "status": "saved", "status_elapsed_since_request_ns": 5}])
    _write(directory, rows)
    return rows


def _write(directory: Path, rows: list[dict[str, Any]]) -> None:
    (directory / "frames.ndjson").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_real_gaps_remain_in_timeline_without_interpolation(tmp_path: Path) -> None:
    _capture(tmp_path)
    manifest = capture_manifest(tmp_path)
    assert manifest["frame_count"] == 3
    assert manifest["max_interval_seconds"] == 0.55
    assert manifest["intervals_over_250ms"] == 1
    assert manifest["expected_duration_seconds"] == pytest.approx(0.9)
    assert manifest["observed_average_fps"] == pytest.approx(2 / 0.6)
    document = concat_document(manifest)
    assert "duration 0.050000" in document
    assert "duration 0.550000" in document
    assert document.count("option framerate 1000") == 4  # Three frames plus terminal hold.


def _writer_metadata(directory: Path) -> list[dict[str, Any]]:
    rows = _capture(directory)
    total = 0
    for request, saved in zip(rows[::2], rows[1::2], strict=True):
        byte_count = (directory / request["file"]).stat().st_size
        fixed = {"png_writer": "jdk_deflater_rgba_filter0_v1", "png_compression": 1}
        phases = {
            "request_to_readback_complete_elapsed_ns": 2,
            "native_readback_elapsed_ns": 1,
            "png_pixel_copy_elapsed_ns": 1,
            "png_encode_write_elapsed_ns": 2,
        }
        request.update(fixed, **dict.fromkeys(phases), frame_bytes=None, total_frame_bytes=total)
        total += byte_count
        saved.update(fixed, **phases, frame_bytes=byte_count, total_frame_bytes=total)
    _write(directory, rows)
    return rows


def test_completed_png_writer_metrics_preserve_request_timeline(tmp_path: Path) -> None:
    _writer_metadata(tmp_path)
    manifest = capture_manifest(tmp_path)
    assert manifest["frame_count"] == 3
    assert manifest["observed_average_fps"] == pytest.approx(2 / 0.6)
    assert manifest["expected_duration_seconds"] == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("png_writer", "minecraft_stb"), ("png_compression", 9), ("frame_bytes", 1)],
)
def test_writer_provenance_and_saved_size_must_match(
    tmp_path: Path, field: str, replacement: object
) -> None:
    rows = _writer_metadata(tmp_path)
    rows[1][field] = replacement
    _write(tmp_path, rows)
    with pytest.raises(ValueError):
        capture_manifest(tmp_path)


@pytest.mark.parametrize("mutation", ["duplicate", "timestamp", "path", "reverse", "orphan"])
def test_ambiguous_capture_records_are_refused(tmp_path: Path, mutation: str) -> None:
    rows = _capture(tmp_path)
    if mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "timestamp":
        rows[1]["request_wall_elapsed_ns"] += 1
    elif mutation == "path":
        rows[0]["file"] = rows[1]["file"] = "../outside.png"
    elif mutation == "reverse":
        rows[2]["request_wall_elapsed_ns"] = rows[3]["request_wall_elapsed_ns"] = 1
    else:
        rows.pop(0)
    _write(tmp_path, rows)
    with pytest.raises(ValueError):
        capture_manifest(tmp_path)


def test_unsaved_request_is_reported_and_paths_are_quoted(tmp_path: Path) -> None:
    directory = tmp_path / "Evan's capture"
    directory.mkdir()
    rows = _capture(directory)
    rows.append({**rows[0], "frame": 4, "file": "screenshots/frame-0004.png"})
    _write(directory, rows)
    manifest = capture_manifest(directory)
    assert manifest["unsaved_requested_frames"] == [4]
    assert "Evan'\\''s capture" in concat_document(manifest)


def test_long_fractional_microsecond_cadence_never_accumulates_roundoff() -> None:
    count, interval_ns = 14_400, 50_000_500
    manifest = {
        "frames": [
            {"path": "/fixture.png", "request_wall_elapsed_ns": i * interval_ns}
            for i in range(count)
        ],
        "expected_duration_seconds": count * interval_ns / 1e9,
    }
    document = concat_document(manifest)
    durations_us = [
        round(float(line.split()[1]) * 1_000_000)
        for line in document.splitlines()
        if line.startswith("duration ")
    ]
    boundary = 0
    for i, duration in enumerate(durations_us, 1):
        boundary += duration
        assert abs(boundary * 1000 - i * interval_ns) <= 500


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="offline FFmpeg is optional")
@pytest.mark.parametrize("corrupt_middle", [False, True])
def test_real_encoder_requires_every_frame_to_decode(tmp_path: Path, corrupt_middle: bool) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _capture(source)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    # Minimal valid RGBA test image; these are fixture pixels, never delivery footage.
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress((b"\x00" + b"\xff\x00\x00\xff" * 2) * 2))
        + chunk(b"IEND", b"")
    )
    for path in (source / "screenshots").glob("*.png"):
        path.write_bytes(png)
    if corrupt_middle:
        (source / "screenshots/frame-0002.png").write_bytes(png[:24])
        with pytest.raises(subprocess.CalledProcessError):
            export(source, tmp_path / "output")
        assert not (tmp_path / "output/manifest.json").exists()
    else:
        video = export(source, tmp_path / "output")
        assert video.is_file()
        result = json.loads((tmp_path / "output/manifest.json").read_text())
        assert int(result["probe"]["streams"][0]["nb_frames"]) == 4
        assert [float(packet["pts_time"]) for packet in result["probe"]["packets"]] == (
            pytest.approx([0, 0.05, 0.6, 0.9], abs=0.0011)
        )
        assert result["encoded_duration_seconds"] == pytest.approx(0.9, abs=0.01)
