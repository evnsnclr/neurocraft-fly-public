#!/usr/bin/env python3
"""Export actual Minecraft framebuffer samples with their recorded request-time spacing.

Uses local FFmpeg (delivery environment: 8.1) after capture, never during control.
There is no invented motion or interpolation. Gaps hold the previous frame and are reported.
Request times approximate capture scheduling; they are not exact image or neural timestamps.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import struct
import subprocess
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from typing import Any


def capture_manifest(directory: Path) -> dict[str, Any]:
    metadata = directory / "frames.ndjson"
    metadata_bytes = metadata.read_bytes()
    requests: dict[int, dict[str, Any]] = {}
    saved: dict[int, dict[str, Any]] = {}
    failures = []
    for line in metadata_bytes.decode("utf-8").splitlines():
        row = json.loads(line)
        if row.get("schema") != "neurocraft.live_frame_capture.v1":
            raise ValueError("unsupported capture schema")
        frame = row.get("frame")
        if type(frame) is not int or frame < 1:
            raise ValueError("invalid frame number")
        status = row.get("status")
        if status == "requested":
            if frame in requests:
                raise ValueError("duplicate frame request")
            requests[frame] = row
        elif status in ("saved", "saved_after_disable"):
            if frame in saved:
                raise ValueError("duplicate frame completion")
            saved[frame] = row
        else:
            failures.append(row)
    frames = []
    dimensions = set()
    previous = -1
    for number in sorted(saved):
        if number not in requests:
            raise ValueError("saved frame has no request")
        request, completion = requests[number], saved[number]
        fixed = set(request) - {
            "status",
            "status_elapsed_since_request_ns",
            "frame_bytes",
            "total_frame_bytes",
            "request_to_readback_complete_elapsed_ns",
            "native_readback_elapsed_ns",
            "png_pixel_copy_elapsed_ns",
            "png_encode_write_elapsed_ns",
        }
        if any(request[key] != completion.get(key) for key in fixed):
            raise ValueError("frame completion differs from request")
        relative = f"screenshots/frame-{number:04d}.png"
        if request.get("file") != relative:
            raise ValueError("unexpected screenshot path")
        elapsed = request.get("request_wall_elapsed_ns")
        if type(elapsed) is not int or elapsed < 0 or elapsed <= previous:
            raise ValueError("capture request times must strictly increase")
        previous = elapsed
        path = directory / relative
        if path.is_symlink():
            raise ValueError("screenshot symlink is not a captured frame")
        data = path.read_bytes()
        if "frame_bytes" in completion and (
            type(completion["frame_bytes"]) is not int or completion["frame_bytes"] != len(data)
        ):
            raise ValueError("completion frame bytes differ from the captured PNG")
        if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
            raise ValueError("captured file is not a PNG")
        dimensions.add(struct.unpack(">II", data[16:24]))
        frames.append(
            {
                "frame": number,
                "path": str(path.resolve()),
                "sha256": sha256(data).hexdigest(),
                "request_wall_elapsed_ns": elapsed,
                "request_game_tick": request["request_game_tick"],
            }
        )
    if len(frames) < 2 or len(dimensions) != 1:
        raise ValueError("export requires at least two frames with one unchanged resolution")
    gaps = [
        (b["request_wall_elapsed_ns"] - a["request_wall_elapsed_ns"]) / 1e9
        for a, b in pairwise(frames)
    ]
    tail = statistics.median(gaps)
    return {
        "schema": "neurocraft.capture_video_export.v1",
        "diagnostic_only": True,
        "claim_boundary": "Actual framebuffer samples with request-time spacing. Gaps hold the "
        "previous image; no interpolation. Request time is not an exact image or neural timestamp.",
        "source_metadata_sha256": sha256(metadata_bytes).hexdigest(),
        "dimensions": list(next(iter(dimensions))),
        "frames": frames,
        "frame_count": len(frames),
        "median_interval_seconds": tail,
        "max_interval_seconds": max(gaps),
        "observed_average_fps": len(gaps) / sum(gaps),
        "intervals_over_250ms": sum(gap > 0.25 for gap in gaps),
        "final_frame_hold_seconds": tail,
        "expected_duration_seconds": sum(gaps) + tail,
        "unsaved_requested_frames": sorted(requests.keys() - saved.keys()),
        "capture_failure_records": failures,
    }


def concat_document(manifest: dict[str, Any]) -> str:
    frames = manifest["frames"]
    lines = ["ffconcat version 1.0"]
    origin = frames[0]["request_wall_elapsed_ns"]
    boundaries_us = [
        (frame["request_wall_elapsed_ns"] - origin + 500) // 1000 for frame in frames
    ] + [round(manifest["expected_duration_seconds"] * 1_000_000)]

    def quote(path: str) -> str:
        if "\n" in path or "\r" in path:
            raise ValueError("capture path contains a line break")
        return "'" + path.replace("'", "'\\''") + "'"

    for index, frame in enumerate(frames):
        lines.extend([f"file {quote(frame['path'])}", "option framerate 1000"])
        duration_us = boundaries_us[index + 1] - boundaries_us[index]
        lines.append(f"duration {duration_us // 1_000_000}.{duration_us % 1_000_000:06d}")
    # concat needs a terminal copy to retain the final frame's declared hold.
    lines.extend([f"file {quote(frames[-1]['path'])}", "option framerate 1000"])
    return "\n".join(lines) + "\n"


def export(directory: Path, output_directory: Path) -> Path:
    manifest = capture_manifest(directory)
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("local FFmpeg and ffprobe are required for offline video export")
    output_directory.mkdir(parents=True, exist_ok=False)
    manifest["ffmpeg_version"] = subprocess.check_output(
        [ffmpeg, "-version"], text=True
    ).splitlines()[0]
    concat = output_directory / "frames.ffconcat"
    concat.write_text(concat_document(manifest))
    video = output_directory / "capture.mp4"
    command = [
        ffmpeg,
        "-nostdin",
        "-xerror",
        "-n",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat),
        "-fps_mode",
        "vfr",
        "-c:v",
        "libx264",
        "-bf",
        "0",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "2",
        "-movflags",
        "+faststart",
        str(video),
    ]
    with (output_directory / "ffmpeg.log").open("x") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    probe = json.loads(
        subprocess.check_output(
            [
                ffprobe,
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-show_packets",
                "-of",
                "json",
                str(video),
            ],
            text=True,
        )
    )
    duration = float(probe["format"]["duration"])
    if abs(duration - manifest["expected_duration_seconds"]) > 0.1:
        raise RuntimeError("encoded duration differs from the captured request timeline")
    streams = [stream for stream in probe["streams"] if stream.get("codec_type") == "video"]
    if len(streams) != 1 or int(streams[0].get("nb_frames", 0)) != manifest["frame_count"] + 1:
        raise RuntimeError("encoded frame count differs from captured frames plus terminal hold")
    origin = manifest["frames"][0]["request_wall_elapsed_ns"]
    expected_times = [
        (frame["request_wall_elapsed_ns"] - origin) / 1e9 for frame in manifest["frames"]
    ] + [manifest["expected_duration_seconds"]]
    encoded_times = [float(packet["pts_time"]) for packet in probe["packets"]]
    if len(encoded_times) != len(expected_times) or any(
        abs(actual - expected) > 0.0011
        for actual, expected in zip(encoded_times, expected_times, strict=True)
    ):
        raise RuntimeError("encoded timestamps differ from the captured request timeline")
    manifest.update(
        video_sha256=sha256(video.read_bytes()).hexdigest(),
        encoded_duration_seconds=duration,
        probe=probe,
    )
    with (output_directory / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return video


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_directory", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    print(export(args.capture_directory, args.output_directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
