"""Canonical run identities that prevent mode and evidence-tier ambiguity."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")


class RunMode(StrEnum):
    """Mutually exclusive controller/control modes recorded in every run."""

    FAITHFUL = "faithful"
    SHUFFLED_WEIGHT = "shuffled_weight"
    DEGREE_PRESERVING_RANDOM = "degree_preserving_random"
    LESION = "lesion"
    NULL_GRAPH = "null_graph"
    FROZEN_STATE = "frozen_state"
    SCRIPTED_BASELINE = "scripted_baseline"
    SYNTHETIC = "synthetic"
    NO_INPUT = "no_input"
    PASSIVE_PHYSICS = "passive_physics"
    SIGN_SENSITIVITY = "sign_sensitivity"
    ADAPTER_PERMUTATION = "adapter_permutation"
    DECODER_PERMUTATION = "decoder_permutation"


class DataTier(StrEnum):
    """Artifact identity shown in traces, dashboards, and reports."""

    SYNTHETIC = "SYNTHETIC"
    MALECNS_FIXTURE = "MALECNS_FIXTURE"
    MALECNS_CIRCUIT = "MALECNS_CIRCUIT"
    MALECNS_FULL = "MALECNS_FULL"


def _reject_non_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical evidence cannot contain non-finite numbers")
    if isinstance(value, dict):
        for item in value.values():
            _reject_non_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_non_finite(item)


def canonical_bytes(value: Any) -> bytes:
    """Encode strict, stable JSON suitable for evidence hashing."""

    _reject_non_finite(value)
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class TrialIdentity:
    """All immutable identities required to compare or replay one trial."""

    run_id: str
    mode: RunMode
    data_tier: DataTier
    seed: int
    graph_sha256: str
    adapter_sha256: str
    dynamics_sha256: str
    decoder_sha256: str
    arena_sha256: str
    protocol_version: int = 1

    def __post_init__(self) -> None:
        if not _ID_RE.fullmatch(self.run_id):
            raise ValueError("run_id must be a lowercase stable identifier")
        if self.seed < 0 or self.seed > (2**63 - 1):
            raise ValueError("seed must fit a non-negative signed 64-bit integer")
        if self.protocol_version != 1:
            raise ValueError("only protocol version 1 is supported")
        for field_name in (
            "graph_sha256",
            "adapter_sha256",
            "dynamics_sha256",
            "decoder_sha256",
            "arena_sha256",
        ):
            if not _HASH_RE.fullmatch(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
        if self.mode is RunMode.SYNTHETIC and self.data_tier is not DataTier.SYNTHETIC:
            raise ValueError("synthetic mode must use the SYNTHETIC data tier")
        if self.mode is RunMode.FAITHFUL and self.data_tier is DataTier.SYNTHETIC:
            raise ValueError("faithful mode cannot be labelled with synthetic data")

    @property
    def identity_sha256(self) -> str:
        payload = asdict(self)
        payload["mode"] = self.mode.value
        payload["data_tier"] = self.data_tier.value
        return canonical_sha256(payload)
