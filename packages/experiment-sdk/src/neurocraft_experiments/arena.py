"""Strict arena specifications shared by setup tools and experiment receipts."""

from __future__ import annotations

import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .identity import canonical_sha256

MAX_ARENA_BYTES = 64 * 1024


def read_bounded_file_bytes(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    """Return whole-file bytes read from the descriptor that was size-gated.

    Shared fd-bound reader for every experiment document. ``fstat`` proves the
    file is regular and no larger than ``maximum_bytes`` before any content is
    allocated, and the read that follows uses the same descriptor.
    """

    with path.open("rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{label} must be a regular file")
        if info.st_size > maximum_bytes:
            raise ValueError(
                f"{label} exceeds the {maximum_bytes}-byte limit: {info.st_size} bytes"
            )
        payload = handle.read(info.st_size + 1)
    if len(payload) != info.st_size:
        raise ValueError(f"{label} length changed during the gated read: read {len(payload)} bytes")
    return payload


def _triplet(value: Any, field: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field} must be a three-number array")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field} must be finite")
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ArenaSpec:
    arena_id: str
    version: int
    description: str
    min_corner: tuple[float, float, float]
    max_corner: tuple[float, float, float]
    spawn: tuple[float, float, float]
    yaw_degrees: float
    required_sensors: tuple[str, ...]
    metrics: tuple[str, ...]
    raw: dict[str, Any]

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.raw)


def load_arena(path: str | Path) -> ArenaSpec:
    raw = read_bounded_file_bytes(Path(path), maximum_bytes=MAX_ARENA_BYTES, label="arena")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("arena root must be an object")
    allowed = {
        "arena_id",
        "version",
        "description",
        "bounds",
        "spawn",
        "yaw_degrees",
        "required_sensors",
        "metrics",
        "geometry",
    }
    unknown = set(payload).difference(allowed)
    if unknown:
        raise ValueError(f"unknown arena fields: {sorted(unknown)}")
    bounds = payload.get("bounds")
    if not isinstance(bounds, dict) or set(bounds) != {"min", "max"}:
        raise ValueError("bounds must contain exactly min and max")
    minimum = _triplet(bounds["min"], "bounds.min")
    maximum = _triplet(bounds["max"], "bounds.max")
    spawn = _triplet(payload.get("spawn"), "spawn")
    if any(low >= high for low, high in zip(minimum, maximum, strict=True)):
        raise ValueError("each arena minimum must be below its maximum")
    inside_bounds = all(
        low <= point <= high for low, point, high in zip(minimum, spawn, maximum, strict=True)
    )
    if not inside_bounds:
        raise ValueError("spawn must lie inside bounds")
    required_sensors = payload.get("required_sensors")
    metrics = payload.get("metrics")
    if not isinstance(required_sensors, list) or not required_sensors:
        raise ValueError("required_sensors must be a non-empty list")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("metrics must be a non-empty list")
    if len(set(required_sensors)) != len(required_sensors) or len(set(metrics)) != len(metrics):
        raise ValueError("sensor and metric names must be unique")
    return ArenaSpec(
        arena_id=str(payload["arena_id"]),
        version=int(payload["version"]),
        description=str(payload["description"]),
        min_corner=minimum,
        max_corner=maximum,
        spawn=spawn,
        yaw_degrees=float(payload["yaw_degrees"]),
        required_sensors=tuple(str(item) for item in required_sensors),
        metrics=tuple(str(item) for item in metrics),
        raw=payload,
    )
