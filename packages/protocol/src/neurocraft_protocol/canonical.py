"""Bounded JSON parsing and the project's deterministic JSON encoding."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass
from typing import Any

from .errors import OversizeFrameError, ProtocolError

MAX_FRAME_BYTES = 65_536
MAX_JSON_DEPTH = 24


def _reject_constant(value: str) -> None:
    raise ProtocolError(f"non-finite JSON number is forbidden: {value}")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _validate_json_tree(value: Any, *, depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise ProtocolError(f"JSON nesting exceeds {MAX_JSON_DEPTH}")
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolError("non-finite JSON number is forbidden")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProtocolError("JSON object keys must be strings")
            _validate_json_tree(item, depth=depth + 1)
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for item in value:
            _validate_json_tree(item, depth=depth + 1)
        return
    raise ProtocolError(f"unsupported JSON value type: {type(value).__name__}")


def parse_json_frame(frame: str | bytes, *, max_bytes: int = MAX_FRAME_BYTES) -> dict[str, Any]:
    """Parse one UTF-8 JSON object with hard size, duplicate-key, and number checks."""

    if isinstance(frame, bytes):
        raw = frame
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ProtocolError("frame is not valid UTF-8") from exc
    elif isinstance(frame, str):
        text = frame
        raw = frame.encode("utf-8")
    else:
        raise ProtocolError("frame must be text or bytes")
    if len(raw) > max_bytes:
        raise OversizeFrameError(f"frame exceeds {max_bytes} UTF-8 bytes")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except ProtocolError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("frame is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ProtocolError("top-level JSON value must be an object")
    _validate_json_tree(value)
    return value


def _json_ready(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _json_ready(value.to_dict())
    if is_dataclass(value):
        raise ProtocolError("dataclass must expose an explicit to_dict method")
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolError("non-finite JSON number is forbidden")
        # Make negative zero canonical instead of preserving an invisible sign bit.
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProtocolError("JSON object keys must be strings")
            result[key] = _json_ready(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_ready(item) for item in value]
    raise ProtocolError(f"unsupported JSON value type: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize using the repository's stable, compact, UTF-8 JSON profile."""

    ready = _json_ready(value)
    _validate_json_tree(ready)
    return json.dumps(
        ready,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def encode_json_frame(value: Any, *, max_bytes: int = MAX_FRAME_BYTES) -> str:
    raw = canonical_json_bytes(value)
    if len(raw) > max_bytes:
        raise OversizeFrameError(f"encoded frame exceeds {max_bytes} UTF-8 bytes")
    return raw.decode("utf-8")
