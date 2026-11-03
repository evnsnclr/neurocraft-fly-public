"""Deterministic non-Minecraft fixture-physics pilot and evidence report.

This module is deliberately separate from the Minecraft evaluation protocol.
Its tiny physics world is useful for exercising source-derived controller and
control identities, but it cannot establish Minecraft behavior or biological
validity.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from neurocraft_connectome import RuntimeGraph, load_malecns_fixture
from neurocraft_controller import ConnectomeController, ControllerConfiguration
from neurocraft_neural import TopologyOverlay, WeightOverlay
from neurocraft_protocol import MotorVector, Observation, SensorFrame

from .arena import MAX_ARENA_BYTES, ArenaSpec, load_arena, read_bounded_file_bytes
from .baseline import RawPhysicalObservation, scripted_baseline
from .identity import DataTier, RunMode, TrialIdentity, canonical_bytes, canonical_sha256

PILOT_SCHEMA = "neurocraft.fixture_physics_pilot.v3"
RESULT_SCHEMA = "neurocraft.fixture_physics_result.v3"
SUPERSEDED_PILOT_SCHEMA = "neurocraft.fixture_physics_pilot.v2"
ACTIVE_PROTOCOL_ID = "fixture-physics-pilot-v3"

# Fail-closed read caps: every pilot document is size-gated on the descriptor
# that is subsequently read (see arena.read_bounded_file_bytes).
MAX_PROTOCOL_BYTES = 1024 * 1024
MAX_PINNED_FILE_BYTES = 1024 * 1024
MAX_RESULT_BYTES = 16 * 1024 * 1024

# Frozen work-amplification bounds. All limits sit comfortably above the
# executed v2 design (tick_limit 120, 5 seeds, 1 replicate, 5000 bootstrap
# samples, 2 lesioned bodies, 6 conditions, 60 planned trials).
MAX_TICK_LIMIT = 10_000
MAX_SEEDS = 64
MAX_REPLICATES = 16
MAX_BOOTSTRAP_SAMPLES = 100_000
MAX_LESION_BODY_IDS = 4_096
MAX_CONDITIONS = 16
MAX_PLANNED_TRIAL_COUNT = 100_000

# Executed v2 lineage, pinned exactly like the v1 lineage was pinned in v2.
EXECUTED_V2_PROTOCOL_FILE_SHA256 = (
    "18878d8beed0e8da004c8d0b927456d8b20bba37b6f311ef1ece43a8992f3457"
)
EXECUTED_V2_PROTOCOL_PAYLOAD_SHA256 = (
    "aeca3fc5d90c63590750e18b85d25298e214c047ce8f0735c7fbc4776c4fda36"
)
EXECUTED_V2_RESULT_FILE_SHA256 = "c5c5023298a55ebbda51abef2e7f8420c9178d04dba09d6cbfb30489f2eb80e2"

# Executed v3 lineage, pinned exactly like the v1 and v2 lineages. The v3 protocol was frozen,
# executed, and its result snapshot committed, so these bytes are frozen historical evidence and
# the preregistration is append-only: it can never be rewritten to describe a later tree.
EXECUTED_V3_PROTOCOL_FILE_SHA256 = (
    "c1621757efd6ea4abdb250b02597663529b0f008223076b808b3a23b8fc18701"
)
EXECUTED_V3_RESULT_FILE_SHA256 = "62458225913c11901a7d9ff609d53a3cc648761ae08679973748b377a273354b"

#: The last commit at which every ``implementation_hashes`` entry of the v3 preregistration was
#: observed to match the working tree. It is recorded here because those hashes are now read as
#: recorded evidence rather than re-verified against the working tree; see
#: :func:`_recorded_implementation_hashes` for why, and what still holds instead.
IMPLEMENTATION_HASH_TREE_COMMIT = "87a4243a9246bd4e0486f28de8c9665c60590ee5"

#: The last commit at which the v3 **result core** reproduced bit for bit under deterministic
#: reexecution, in the recorded execution environment (macOS-15.6-arm64 / CPython 3.13.5).
#:
#: The reexecution comparison used to run on every same-environment verification. It could not
#: survive the two-tick action budget: every trial's ``hashes.action_stream_sha256`` and
#: ``hashes.trace_stream_sha256`` digest the issued ``Action`` envelopes, which now carry
#: ``target_tick = observation_tick + 2``. Rebuilding the core on the tree of that change moved
#: exactly 100 leaf fields -- 50 action-stream and 50 trace-stream digests across 50 of the 60
#: trials -- and **no metric at all**. Since the 2026-11-03 correction of the lateral rays
#: (R-27) a rebuild also changes metrics in 47 of the 60 trials, every input-driven condition
#: and no no-input trial, without changing the signs or ordering of the paired comparisons;
#: no pin covers this module, and the stored result keeps the transposed-ray history.
#:
#: A result is write-once by construction (:func:`write_result_no_clobber` refuses to overwrite)
#: and a preregistration is append-only, so the recorded core cannot be refreshed; writing a v4
#: whose only purpose is to satisfy a check would be a preregistration written for a test, which
#: ADR 0019 retired. So the reexecution comparison joins the implementation-hash pin as history,
#: recorded against this tree and no longer run against the working tree.
#:
#: **What is lost, stated plainly:** after this, nothing bit-pins the fixture stack's numbers on
#: any platform. Linux never re-executed (the environment gate never matched), so the loss is the
#: macOS check alone -- but it was the only one there was.
#:
#: **What still holds:** every platform-neutral byte pin. The snapshot file's SHA-256, the
#: envelope's internal ``result_sha256`` over its own core, the recorded execution environment,
#: and the ``protocol_sha256`` binding to the frozen preregistration are all still verified on
#: every platform by ``scripts/verify_fixture_pilot_snapshot.py``, so neither the stored numbers
#: nor the protocol they were produced under can be edited without failing that check.
RESULT_CORE_TREE_COMMIT = "a89d2364dbe4c3a63390098a0121c49380f9068e"

ZERO_SHA256 = "0" * 64
_SHA256_CHARS = frozenset("0123456789abcdef")
_POLICY_FIELDS = frozenset(
    {
        "forward_clearance",
        "left_clearance",
        "right_clearance",
        "vertical_velocity",
        "on_ground",
        "local_light",
    }
)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value).issubset(_SHA256_CHARS)


def _strict_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be an object with string keys")
    return value


def _exact_keys(value: Mapping[str, Any], required: set[str], field: str) -> None:
    if set(value) != required:
        raise ValueError(f"{field} fields must be exactly {sorted(required)}")


def _strict_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _strict_int(value: object, field: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field} must be an integer <= {maximum}")
    return value


def _contained_repository_path(root: Path, relative: str, field: str) -> Path:
    """Resolve a declared repository-relative path and refuse every escape.

    Absolute paths and ``..`` components are rejected before resolution, and
    ``resolve()`` + ``is_relative_to`` defeats symlink escapes. ``root`` must
    already be resolved by the caller.
    """

    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"{field} must be repository-relative")
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"{field} escapes the repository root")
    return candidate


def _verified_arena(root: Path, arena_path: str, expected_sha256: str) -> ArenaSpec:
    """Verify declared arena bytes against the frozen hash before loading."""

    candidate = _contained_repository_path(root, arena_path, "arenas[].path")
    payload = read_bounded_file_bytes(candidate, maximum_bytes=MAX_ARENA_BYTES, label="arena")
    observed = sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise ValueError(f"arena file hash mismatch for {arena_path}")
    return load_arena(candidate)


def _bounded_planned_trial_count(
    arena_count: int, seed_count: int, replicates: int, condition_count: int
) -> int:
    """Gate the full trial product so no field combination amplifies work."""

    planned = arena_count * seed_count * replicates * condition_count
    if planned > MAX_PLANNED_TRIAL_COUNT:
        raise ValueError(
            f"planned_trial_count {planned} exceeds the {MAX_PLANNED_TRIAL_COUNT}-trial limit"
        )
    return planned


def _strict_float(value: object, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{field} must be a finite number >= {minimum}")
    return result


@dataclass(frozen=True, slots=True)
class PilotCondition:
    """One unambiguous control label in the fixture-only pilot."""

    condition_id: str
    mode: RunMode
    policy: str
    intervention: str
    receives_adapter_inputs: bool
    lesion_body_ids: tuple[int, ...]
    overlay: str

    @classmethod
    def from_raw(cls, raw: object) -> PilotCondition:
        value = _strict_mapping(raw, "condition")
        _exact_keys(
            value,
            {
                "condition_id",
                "mode",
                "policy",
                "intervention",
                "receives_adapter_inputs",
                "lesion_body_ids",
                "overlay",
            },
            "condition",
        )
        policy = _strict_string(value["policy"], "condition.policy")
        if policy not in {"connectome_controller", "scripted_baseline"}:
            raise ValueError("condition.policy is unsupported")
        overlay = _strict_string(value["overlay"], "condition.overlay")
        if overlay not in {"canonical", "shuffled_weight", "degree_preserving_random"}:
            raise ValueError("condition.overlay is unsupported")
        raw_lesions = value["lesion_body_ids"]
        if not isinstance(raw_lesions, list):
            raise ValueError("condition.lesion_body_ids must be an array")
        if len(raw_lesions) > MAX_LESION_BODY_IDS:
            raise ValueError(
                f"condition.lesion_body_ids must contain at most {MAX_LESION_BODY_IDS} entries"
            )
        lesions = tuple(
            _strict_int(item, "condition.lesion_body_ids[]", minimum=1) for item in raw_lesions
        )
        if len(set(lesions)) != len(lesions):
            raise ValueError("condition.lesion_body_ids contains duplicates")
        receives = value["receives_adapter_inputs"]
        if not isinstance(receives, bool):
            raise ValueError("condition.receives_adapter_inputs must be boolean")
        condition = cls(
            condition_id=_strict_string(value["condition_id"], "condition.condition_id"),
            mode=RunMode(_strict_string(value["mode"], "condition.mode")),
            policy=policy,
            intervention=_strict_string(value["intervention"], "condition.intervention"),
            receives_adapter_inputs=receives,
            lesion_body_ids=lesions,
            overlay=overlay,
        )
        condition.assert_semantics()
        return condition

    def assert_semantics(self) -> None:
        if self.mode is RunMode.SCRIPTED_BASELINE:
            if self.policy != "scripted_baseline" or self.overlay != "canonical":
                raise ValueError("scripted baseline must be explicit and use no graph overlay")
        elif self.policy != "connectome_controller":
            raise ValueError("only scripted_baseline mode may use the scripted policy")
        expected_overlay = {
            RunMode.SHUFFLED_WEIGHT: "shuffled_weight",
            RunMode.DEGREE_PRESERVING_RANDOM: "degree_preserving_random",
        }.get(self.mode, "canonical")
        if self.overlay != expected_overlay:
            raise ValueError("condition mode and overlay disagree")
        if self.mode is RunMode.LESION:
            if not self.lesion_body_ids or self.intervention == "none":
                raise ValueError("lesion mode requires exact body IDs and a named intervention")
        elif self.lesion_body_ids:
            raise ValueError("only lesion mode may declare lesion body IDs")
        if self.mode is RunMode.NO_INPUT:
            if (
                self.receives_adapter_inputs
                or self.intervention != "zero_all_four_adapter_channels"
            ):
                raise ValueError("no-input must be separately labelled with zero adapter channels")
        elif not self.receives_adapter_inputs:
            raise ValueError("only no-input may disable adapter inputs")
        if self.mode is RunMode.FAITHFUL and (
            self.intervention != "none" or self.overlay != "canonical"
        ):
            raise ValueError("faithful condition cannot carry a control intervention")

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(
            {
                "condition_id": self.condition_id,
                "intervention": self.intervention,
                "lesion_body_ids": list(self.lesion_body_ids),
                "mode": self.mode.value,
                "overlay": self.overlay,
                "policy": self.policy,
                "receives_adapter_inputs": self.receives_adapter_inputs,
            }
        )


@dataclass(frozen=True, slots=True)
class PhysicsConfiguration:
    tick_seconds: float
    tick_limit: int
    ray_max_distance_m: float
    horizontal_acceleration_m_s2: float
    vertical_acceleration_m_s2: float
    gravity_m_s2: float
    velocity_retention: float
    yaw_radians_per_tick: float
    body_radius_m: float

    @classmethod
    def from_raw(cls, raw: object) -> PhysicsConfiguration:
        value = _strict_mapping(raw, "physics")
        _exact_keys(
            value,
            {
                "tick_seconds",
                "tick_limit",
                "ray_max_distance_m",
                "horizontal_acceleration_m_s2",
                "vertical_acceleration_m_s2",
                "gravity_m_s2",
                "velocity_retention",
                "yaw_radians_per_tick",
                "body_radius_m",
            },
            "physics",
        )
        result = cls(
            tick_seconds=_strict_float(
                value["tick_seconds"], "physics.tick_seconds", minimum=0.001
            ),
            tick_limit=_strict_int(
                value["tick_limit"], "physics.tick_limit", minimum=1, maximum=MAX_TICK_LIMIT
            ),
            ray_max_distance_m=_strict_float(
                value["ray_max_distance_m"], "physics.ray_max_distance_m", minimum=0.001
            ),
            horizontal_acceleration_m_s2=_strict_float(
                value["horizontal_acceleration_m_s2"],
                "physics.horizontal_acceleration_m_s2",
                minimum=0.0,
            ),
            vertical_acceleration_m_s2=_strict_float(
                value["vertical_acceleration_m_s2"],
                "physics.vertical_acceleration_m_s2",
                minimum=0.0,
            ),
            gravity_m_s2=_strict_float(value["gravity_m_s2"], "physics.gravity_m_s2", minimum=0.0),
            velocity_retention=_strict_float(
                value["velocity_retention"], "physics.velocity_retention", minimum=0.0
            ),
            yaw_radians_per_tick=_strict_float(
                value["yaw_radians_per_tick"],
                "physics.yaw_radians_per_tick",
                minimum=0.0,
            ),
            body_radius_m=_strict_float(
                value["body_radius_m"], "physics.body_radius_m", minimum=0.001
            ),
        )
        if result.velocity_retention > 1.0:
            raise ValueError("physics.velocity_retention must be <= 1")
        if result.ray_max_distance_m > 64.0:
            raise ValueError("physics.ray_max_distance_m exceeds protocol range")
        return result

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(
            {
                field: float(value).hex() if isinstance(value, float) else value
                for field, value in (
                    ("body_radius_m", self.body_radius_m),
                    ("gravity_m_s2", self.gravity_m_s2),
                    ("horizontal_acceleration_m_s2", self.horizontal_acceleration_m_s2),
                    ("ray_max_distance_m", self.ray_max_distance_m),
                    ("tick_limit", self.tick_limit),
                    ("tick_seconds", self.tick_seconds),
                    ("velocity_retention", self.velocity_retention),
                    ("vertical_acceleration_m_s2", self.vertical_acceleration_m_s2),
                    ("yaw_radians_per_tick", self.yaw_radians_per_tick),
                )
            }
        )


@dataclass(frozen=True, slots=True)
class PilotProtocol:
    protocol_id: str
    status: str
    claim_boundary: str
    fixture_path: str
    fixture_file_sha256: str
    arena_paths: tuple[str, ...]
    arena_file_hashes: tuple[str, ...]
    seeds: tuple[int, ...]
    replicates: int
    conditions: tuple[PilotCondition, ...]
    physics: PhysicsConfiguration
    primary_metric: str
    bootstrap_samples: int
    analysis_seed: int
    implementation_hashes: Mapping[str, str]
    raw: Mapping[str, Any]

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.raw)

    @property
    def planned_trial_count(self) -> int:
        return len(self.arena_paths) * len(self.seeds) * self.replicates * len(self.conditions)


def _require_v3_schema(raw: Mapping[str, Any]) -> None:
    schema = raw.get("schema")
    if schema == SUPERSEDED_PILOT_SCHEMA:
        raise ValueError(
            "fixture-physics-pilot-v2 documents are frozen historical evidence; "
            f"the active loader executes only {PILOT_SCHEMA}"
        )
    if schema != PILOT_SCHEMA:
        raise ValueError(f"pilot protocol schema must be {PILOT_SCHEMA}")


def _validated_v3_amendment(raw: Mapping[str, Any]) -> None:
    amendment = _strict_mapping(raw["amendment"], "amendment")
    _exact_keys(
        amendment,
        {
            "supersedes",
            "prior_protocol_file_sha256",
            "prior_protocol_payload_sha256",
            "prior_result_file_sha256",
            "timing",
            "reason",
        },
        "amendment",
    )
    if (
        amendment["supersedes"] != "fixture-physics-pilot-v2"
        or amendment["prior_protocol_file_sha256"] != EXECUTED_V2_PROTOCOL_FILE_SHA256
        or amendment["prior_protocol_payload_sha256"] != EXECUTED_V2_PROTOCOL_PAYLOAD_SHA256
        or amendment["prior_result_file_sha256"] != EXECUTED_V2_RESULT_FILE_SHA256
        or amendment["timing"] != "after_v2_execution_before_v3_execution"
        or not isinstance(amendment["reason"], str)
        or not amendment["reason"]
    ):
        raise ValueError("fixture-pilot v3 amendment lineage is invalid")


def _validated_arena_entries(raw: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    arena_values = raw["arenas"]
    if not isinstance(arena_values, list) or len(arena_values) != 2:
        raise ValueError("pilot must name exactly two arenas")
    arena_paths: list[str] = []
    arena_hashes: list[str] = []
    for item in arena_values:
        entry = _strict_mapping(item, "arenas[]")
        _exact_keys(entry, {"path", "file_sha256"}, "arenas[]")
        arena_paths.append(_strict_string(entry["path"], "arenas[].path"))
        entry_hash = _strict_string(entry["file_sha256"], "arenas[].file_sha256")
        if not _is_sha256(entry_hash):
            raise ValueError("arenas[].file_sha256 must be lowercase SHA-256")
        arena_hashes.append(entry_hash)
    if len(set(arena_paths)) != 2:
        raise ValueError("pilot arena paths must be unique")
    return tuple(arena_paths), tuple(arena_hashes)


def _recorded_implementation_hashes(raw: Mapping[str, Any], root: Path) -> dict[str, str]:
    """Return the preregistration's recorded implementation hashes, as recorded evidence.

    **These hashes describe the tree at** :data:`IMPLEMENTATION_HASH_TREE_COMMIT`, **not the
    working tree, and are no longer compared against it.** The comparison used to run on every
    load. It could not survive, for two reasons that are both structural rather than incidental:

    * The pin includes this module's own bytes, so any edit to ``pilot.py`` — including the edit
      that would refresh the pin — breaks it. It was self-sealing from the moment it was written.
    * A preregistration is append-only, so the recorded hashes cannot be refreshed when the
      pinned files legitimately change. The first such change was the ``program_action`` and
      ``program_receipt`` frames inside protocol v2 (ADR 0020 section 7), which necessarily edit
      ``packages/protocol/src/neurocraft_protocol/messages.py``.

    Rewriting the preregistration is forbidden, and writing a v4 whose only purpose is to satisfy
    a test would be a preregistration written for a test — ADR 0019 retired the practice. So v3
    joins v1 and v2 as history: its bytes stay frozen and pinned
    (:data:`EXECUTED_V3_PROTOCOL_FILE_SHA256`, :data:`EXECUTED_V3_RESULT_FILE_SHA256`), and this
    function validates their shape and refuses path escapes without reading a byte of the tree.

    **What is lost:** a load no longer proves the pinned implementation files are byte-identical
    to the ones that produced ``experiments/results/fixture-physics-pilot-v3.json``.

    **What still holds:** the result envelope binds ``protocol_sha256``, so the recorded hashes
    cannot be edited without breaking ``scripts/verify_fixture_pilot_snapshot.py``.

    **What no longer holds:** that script used to re-execute the pilot in the recorded
    environment and compare the result core bit for bit -- a statement about the numbers the code
    produces rather than about the bytes of the code. The two-tick action budget moved the
    per-trial stream digests (they carry the action envelope) without moving a single metric, and
    a write-once result cannot be refreshed, so that comparison is now history too. See
    :data:`RESULT_CORE_TREE_COMMIT`.
    """

    hashes_raw = _strict_mapping(raw["implementation_hashes"], "implementation_hashes")
    if not hashes_raw:
        raise ValueError("implementation_hashes cannot be empty")
    hashes: dict[str, str] = {}
    for relative_path, expected_hash in sorted(hashes_raw.items()):
        if not _is_sha256(expected_hash):
            raise ValueError("implementation hash must be lowercase SHA-256")
        _contained_repository_path(root, relative_path, "implementation_hashes path")
        hashes[relative_path] = expected_hash
    return hashes


def load_pilot_protocol(path: str | Path, *, repository_root: str | Path) -> PilotProtocol:
    """Load and validate the frozen fixture-only pilot protocol."""

    document = read_bounded_file_bytes(
        Path(path), maximum_bytes=MAX_PROTOCOL_BYTES, label="pilot protocol"
    )
    raw = _strict_mapping(json.loads(document.decode("utf-8")), "pilot protocol")
    _require_v3_schema(raw)
    _exact_keys(
        raw,
        {
            "schema",
            "protocol_id",
            "status",
            "amendment",
            "evidence_scope",
            "claim_boundary",
            "fixture",
            "arenas",
            "seeds",
            "replicates",
            "conditions",
            "physics",
            "analysis",
            "implementation_hashes",
        },
        "pilot protocol",
    )
    if raw["protocol_id"] != ACTIVE_PROTOCOL_ID:
        raise ValueError(f"current pilot loader accepts only {ACTIVE_PROTOCOL_ID}")
    if raw["status"] != "frozen_before_fixture_pilot_execution":
        raise ValueError("pilot protocol is not frozen in the required pre-execution state")
    if raw["evidence_scope"] != "NON_MINECRAFT_LOCAL_FIXTURE_PHYSICS_ONLY":
        raise ValueError("pilot evidence scope must explicitly exclude Minecraft")
    _validated_v3_amendment(raw)

    fixture = _strict_mapping(raw["fixture"], "fixture")
    _exact_keys(fixture, {"path", "file_sha256", "data_tier"}, "fixture")
    if fixture["data_tier"] != DataTier.MALECNS_FIXTURE.value:
        raise ValueError("pilot must use the tracked MaleCNS fixture tier")
    fixture_hash = _strict_string(fixture["file_sha256"], "fixture.file_sha256")
    if not _is_sha256(fixture_hash):
        raise ValueError("fixture.file_sha256 must be lowercase SHA-256")

    arena_paths, arena_hashes = _validated_arena_entries(raw)

    seed_values = raw["seeds"]
    if not isinstance(seed_values, list) or not seed_values:
        raise ValueError("pilot seeds must be a non-empty array")
    if len(seed_values) > MAX_SEEDS:
        raise ValueError(f"seeds must contain at most {MAX_SEEDS} entries")
    seeds = tuple(_strict_int(item, "seeds[]") for item in seed_values)
    if len(set(seeds)) != len(seeds):
        raise ValueError("pilot seeds contain duplicates")
    replicates = _strict_int(raw["replicates"], "replicates", minimum=1, maximum=MAX_REPLICATES)

    condition_values = raw["conditions"]
    if not isinstance(condition_values, list):
        raise ValueError("conditions must be an array")
    if len(condition_values) > MAX_CONDITIONS:
        raise ValueError(f"conditions must contain at most {MAX_CONDITIONS} entries")
    conditions = tuple(PilotCondition.from_raw(item) for item in condition_values)
    if len({condition.condition_id for condition in conditions}) != len(conditions):
        raise ValueError("condition IDs contain duplicates")
    required_modes = {
        RunMode.FAITHFUL,
        RunMode.SHUFFLED_WEIGHT,
        RunMode.DEGREE_PRESERVING_RANDOM,
        RunMode.LESION,
        RunMode.SCRIPTED_BASELINE,
        RunMode.NO_INPUT,
    }
    if {condition.mode for condition in conditions} != required_modes:
        raise ValueError("pilot conditions must contain exactly the six required modes")
    _bounded_planned_trial_count(len(arena_paths), len(seeds), replicates, len(conditions))

    analysis = _strict_mapping(raw["analysis"], "analysis")
    _exact_keys(
        analysis,
        {"primary_metric", "paired_comparisons", "bootstrap_samples", "analysis_seed"},
        "analysis",
    )
    if analysis["primary_metric"] != "normalized_forward_progress":
        raise ValueError("fixture pilot primary metric is fixed")
    comparisons = analysis["paired_comparisons"]
    expected_comparisons = [
        "faithful_vs_shuffled-weight",
        "faithful_vs_degree-preserving",
        "faithful_vs_no-input",
    ]
    if comparisons != expected_comparisons:
        raise ValueError("paired comparisons are not the frozen primary set")
    bootstrap_samples = _strict_int(
        analysis["bootstrap_samples"],
        "analysis.bootstrap_samples",
        minimum=100,
        maximum=MAX_BOOTSTRAP_SAMPLES,
    )
    analysis_seed = _strict_int(analysis["analysis_seed"], "analysis.analysis_seed")

    root = Path(repository_root).resolve()
    hashes = _recorded_implementation_hashes(raw, root)

    fixture_path = _strict_string(fixture["path"], "fixture.path")
    fixture_file = _contained_repository_path(root, fixture_path, "fixture.path")
    fixture_payload = read_bounded_file_bytes(
        fixture_file, maximum_bytes=MAX_PINNED_FILE_BYTES, label="pilot fixture"
    )
    if sha256(fixture_payload).hexdigest() != fixture_hash:
        raise ValueError("pilot fixture file hash mismatch")
    for arena_path, arena_hash in zip(arena_paths, arena_hashes, strict=True):
        _verified_arena(root, arena_path, arena_hash)

    return PilotProtocol(
        protocol_id=_strict_string(raw["protocol_id"], "protocol_id"),
        status=_strict_string(raw["status"], "status"),
        claim_boundary=_strict_string(raw["claim_boundary"], "claim_boundary"),
        fixture_path=fixture_path,
        fixture_file_sha256=fixture_hash,
        arena_paths=arena_paths,
        arena_file_hashes=arena_hashes,
        seeds=seeds,
        replicates=replicates,
        conditions=conditions,
        physics=PhysicsConfiguration.from_raw(raw["physics"]),
        primary_metric="normalized_forward_progress",
        bootstrap_samples=bootstrap_samples,
        analysis_seed=analysis_seed,
        implementation_hashes=hashes,
        raw=raw,
    )


@dataclass(slots=True)
class _World:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    yaw: float
    on_ground: bool = False
    horizontal_collision: bool = False
    vertical_collision: bool = False
    collision_count: int = 0
    fall_count: int = 0
    path_length: float = 0.0


@dataclass(frozen=True, slots=True)
class PolicyStep:
    motors: MotorVector
    observation_sha256: str
    action_sha256: str
    state_sha256: str
    feature_values: Mapping[str, float]


def policy_sensor_fields(sensors: SensorFrame) -> Mapping[str, float | bool]:
    """Expose the complete and target-free policy boundary for audit tests."""

    values: dict[str, float | bool] = {
        "forward_clearance": sensors.ray_distances_m[0],
        "left_clearance": sensors.ray_distances_m[2],
        "right_clearance": sensors.ray_distances_m[3],
        "vertical_velocity": sensors.velocity_m_s[1],
        "on_ground": sensors.on_ground,
        "local_light": float(sensors.local_light),
    }
    if set(values) != _POLICY_FIELDS:
        raise RuntimeError("pilot policy boundary schema drifted")
    return values


def select_policy_action(
    condition: PilotCondition,
    controller: ConnectomeController | None,
    sensors: SensorFrame,
    *,
    sequence: int,
) -> PolicyStep:
    """Select an action from physical sensors without arena/target state access."""

    boundary = policy_sensor_fields(sensors)
    if condition.mode is RunMode.SCRIPTED_BASELINE:
        if controller is not None:
            raise ValueError("scripted baseline must not execute a neural controller")
        primitive = scripted_baseline(
            RawPhysicalObservation(
                forward_clearance=float(boundary["forward_clearance"]),
                left_clearance=float(boundary["left_clearance"]),
                right_clearance=float(boundary["right_clearance"]),
                vertical_velocity=float(boundary["vertical_velocity"]),
                on_ground=bool(boundary["on_ground"]),
            )
        )
        motors = MotorVector(
            forward_thrust=primitive.forward_thrust,
            lateral_thrust=primitive.lateral_thrust,
            vertical_thrust=primitive.vertical_thrust,
            yaw_torque=primitive.yaw_torque,
            pitch_torque=primitive.pitch_torque,
        )
        action_material = {
            "condition_id": condition.condition_id,
            "motors": motors.to_dict(),
            "policy": "scripted_baseline",
            "sequence": sequence,
        }
        return PolicyStep(
            motors=motors,
            observation_sha256=canonical_sha256(boundary),
            action_sha256=canonical_sha256(action_material),
            state_sha256=ZERO_SHA256,
            feature_values={name: float(value) for name, value in boundary.items()},
        )
    if controller is None:
        raise ValueError("connectome condition requires a controller")
    executed_sensors = sensors
    if condition.mode is RunMode.NO_INPUT:
        executed_sensors = SensorFrame(
            position_m=sensors.position_m,
            velocity_m_s=sensors.velocity_m_s,
            yaw_rad=sensors.yaw_rad,
            pitch_rad=sensors.pitch_rad,
            on_ground=sensors.on_ground,
            horizontal_collision=sensors.horizontal_collision,
            vertical_collision=sensors.vertical_collision,
            in_fluid=sensors.in_fluid,
            local_light=0,
            ray_max_distance_m=sensors.ray_max_distance_m,
            ray_distances_m=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
    observation = Observation(
        identity=controller.identity,
        observation_seq=sequence,
        game_tick=sequence,
        sent_monotonic_ns=sequence * 50_000_000,
        sensors=executed_sensors,
    )
    action = controller.step(observation, command_seq=sequence)
    evidence = controller.last_evidence
    if evidence is None:
        raise RuntimeError("controller produced no evidence")
    return PolicyStep(
        motors=action.motors,
        observation_sha256=evidence.observation_sha256,
        action_sha256=evidence.action_sha256,
        state_sha256=evidence.state_after_sha256,
        feature_values=dict(evidence.feature_values),
    )


def _ray_to_bounds(
    world: _World,
    arena: ArenaSpec,
    direction_x: float,
    direction_z: float,
    maximum: float,
) -> float:
    distances: list[float] = []
    if direction_x > 1e-15:
        distances.append((arena.max_corner[0] - world.x) / direction_x)
    elif direction_x < -1e-15:
        distances.append((arena.min_corner[0] - world.x) / direction_x)
    if direction_z > 1e-15:
        distances.append((arena.max_corner[2] - world.z) / direction_z)
    elif direction_z < -1e-15:
        distances.append((arena.min_corner[2] - world.z) / direction_z)
    positive = [distance for distance in distances if distance >= 0.0]
    if not positive:
        return maximum
    return min(maximum, max(0.0, min(positive)))


def _body_right(yaw: float) -> tuple[float, float]:
    """The body's right in the arena plane, shared by sensing and actuation (ADR 0015).

    A clockwise quarter turn from the heading ``(cos yaw, sin yaw)`` viewed from +Y: facing
    east (+X) the right points south (+Z). One expression for both callers, because the R-27
    defect was this expression duplicated under two labels.
    """
    return (-math.sin(yaw), math.cos(yaw))


def _sensor_frame(world: _World, arena: ArenaSpec, physics: PhysicsConfiguration) -> SensorFrame:
    forward_x = math.cos(world.yaw)
    forward_z = math.sin(world.yaw)
    # Until 2026-11-03 this frame put the right-hand clearance in the ``left`` slot (R-27).
    right_x, right_z = _body_right(world.yaw)
    ray_max = physics.ray_max_distance_m
    forward = _ray_to_bounds(world, arena, forward_x, forward_z, ray_max)
    backward = _ray_to_bounds(world, arena, -forward_x, -forward_z, ray_max)
    left = _ray_to_bounds(world, arena, -right_x, -right_z, ray_max)
    right = _ray_to_bounds(world, arena, right_x, right_z, ray_max)
    upward = min(ray_max, max(0.0, arena.max_corner[1] - world.y))
    downward = min(ray_max, max(0.0, world.y - arena.min_corner[1]))
    if arena.arena_id == "light-corridor-v1":
        span = arena.max_corner[0] - arena.min_corner[0]
        local_light = round(15.0 * (world.x - arena.min_corner[0]) / span)
        local_light = min(15, max(0, local_light))
    else:
        local_light = 8
    return SensorFrame(
        position_m=(world.x, world.y, world.z),
        velocity_m_s=(world.vx, world.vy, world.vz),
        yaw_rad=world.yaw,
        pitch_rad=0.0,
        on_ground=world.on_ground,
        horizontal_collision=world.horizontal_collision,
        vertical_collision=world.vertical_collision,
        in_fluid=False,
        local_light=local_light,
        ray_max_distance_m=ray_max,
        ray_distances_m=(forward, backward, left, right, upward, downward),
    )


def _crosswind_acceleration(world: _World, arena: ArenaSpec) -> tuple[float, float, float]:
    geometry = _strict_mapping(arena.raw["geometry"], "arena.geometry")
    if arena.arena_id != "crosswind-gap-v1":
        return (0.0, 0.0, 0.0)
    zone = _strict_mapping(geometry["force_zone"], "arena.geometry.force_zone")
    minimum = zone["min"]
    maximum = zone["max"]
    force = zone["force"]
    if not (
        isinstance(minimum, list)
        and isinstance(maximum, list)
        and isinstance(force, list)
        and len(minimum) == len(maximum) == len(force) == 3
    ):
        raise ValueError("crosswind force zone must contain three-vectors")
    position = (world.x, world.y, world.z)
    inside = all(
        float(low) <= point <= float(high)
        for low, point, high in zip(minimum, position, maximum, strict=True)
    )
    if not inside:
        return (0.0, 0.0, 0.0)
    # Arena values are treated as per-tick velocity impulses in this pilot.
    return (float(force[0]), float(force[1]), float(force[2]))


def _step_physics(
    world: _World,
    motors: MotorVector,
    arena: ArenaSpec,
    physics: PhysicsConfiguration,
) -> None:
    before = (world.x, world.y, world.z)
    world.yaw += motors.yaw_torque * physics.yaw_radians_per_tick
    world.yaw = math.atan2(math.sin(world.yaw), math.cos(world.yaw))
    forward_x = math.cos(world.yaw)
    forward_z = math.sin(world.yaw)
    right_x, right_z = _body_right(world.yaw)
    dt = physics.tick_seconds
    horizontal = physics.horizontal_acceleration_m_s2
    world.vx += (
        (motors.forward_thrust * forward_x + motors.lateral_thrust * right_x) * horizontal * dt
    )
    world.vz += (
        (motors.forward_thrust * forward_z + motors.lateral_thrust * right_z) * horizontal * dt
    )
    world.vy += (
        motors.vertical_thrust * physics.vertical_acceleration_m_s2 - physics.gravity_m_s2
    ) * dt
    wind_x, wind_y, wind_z = _crosswind_acceleration(world, arena)
    world.vx += wind_x
    world.vy += wind_y
    world.vz += wind_z
    world.vx *= physics.velocity_retention
    world.vy *= physics.velocity_retention
    world.vz *= physics.velocity_retention
    world.x += world.vx * dt
    world.y += world.vy * dt
    world.z += world.vz * dt

    radius = physics.body_radius_m
    world.horizontal_collision = False
    world.vertical_collision = False
    x_min = arena.min_corner[0] + radius
    z_min = arena.min_corner[2] + radius
    z_max = arena.max_corner[2] - radius
    if world.x < x_min:
        world.x = x_min
        world.vx = max(0.0, world.vx)
        world.horizontal_collision = True
    if world.z < z_min:
        world.z = z_min
        world.vz = max(0.0, world.vz)
        world.horizontal_collision = True
    elif world.z > z_max:
        world.z = z_max
        world.vz = min(0.0, world.vz)
        world.horizontal_collision = True

    y_min = arena.min_corner[1] + radius
    y_max = arena.max_corner[1] - radius
    if world.y < y_min:
        world.y = y_min
        world.vy = max(0.0, world.vy)
        world.on_ground = True
        world.vertical_collision = True
        if arena.arena_id == "crosswind-gap-v1":
            geometry = _strict_mapping(arena.raw["geometry"], "arena.geometry")
            gap_x = geometry["gap_x"]
            if (
                isinstance(gap_x, list)
                and len(gap_x) == 2
                and float(gap_x[0]) <= world.x <= float(gap_x[1])
            ):
                world.fall_count += 1
    else:
        world.on_ground = False
    if world.y > y_max:
        world.y = y_max
        world.vy = min(0.0, world.vy)
        world.vertical_collision = True
    if world.horizontal_collision or world.vertical_collision:
        world.collision_count += 1
    world.path_length += math.dist(before, (world.x, world.y, world.z))


def _exit_x(arena: ArenaSpec) -> float:
    geometry = _strict_mapping(arena.raw["geometry"], "arena.geometry")
    return _strict_float(geometry["exit_plane_x"], "arena.geometry.exit_plane_x")


def _controller_for_condition(
    graph: RuntimeGraph,
    condition: PilotCondition,
    *,
    protocol_id: str,
    seed: int,
    run_id: str,
) -> tuple[ConnectomeController | None, Mapping[str, str]]:
    weight_overlay = None
    topology_overlay = None
    if condition.mode is RunMode.SHUFFLED_WEIGHT:
        weight_overlay = WeightOverlay.shuffled(
            graph,
            seed=seed,
            provenance=f"{protocol_id}-shuffled-weight",
        )
    elif condition.mode is RunMode.DEGREE_PRESERVING_RANDOM:
        topology_overlay = TopologyOverlay.degree_preserving_random(
            graph,
            seed=seed,
            requested_swaps=1,
            provenance=f"{protocol_id}-degree-preserving",
        )
    configuration_mode = "lesion" if condition.mode is RunMode.LESION else "faithful"
    controller = ConnectomeController.create(
        graph,
        run_id=run_id,
        entity_id="fixture-physics-pilot-fly",
        configuration=ControllerConfiguration(mode=configuration_mode),
        weight_overlay=weight_overlay,
        topology_overlay=topology_overlay,
    )
    if condition.mode is RunMode.LESION:
        controller.set_named_lesion(
            condition.intervention,
            condition.lesion_body_ids,
        )
    hashes = {
        "adapter_sha256": controller.identity.adapter_hash,
        "controller_run_identity_sha256": canonical_sha256(controller.identity.to_dict()),
        "decoder_sha256": controller.identity.decoder_hash,
        "dynamics_sha256": controller.identity.dynamics_hash,
        "topology_overlay_sha256": controller.engine.topology_overlay.identity_sha256,
        "weight_overlay_sha256": controller.engine.weight_overlay.identity_sha256,
    }
    if condition.mode is RunMode.SCRIPTED_BASELINE:
        # The reference controller is constructed only to bind the fixed adapter,
        # graph, dynamics, and decoder context. It is never stepped.
        return None, hashes
    return controller, hashes


def _trial_run_id(arena: ArenaSpec, condition: PilotCondition, seed: int, replicate: int) -> str:
    arena_label = "corridor" if arena.arena_id == "light-corridor-v1" else "crosswind"
    return f"fxp-{arena_label}-{condition.condition_id}-{seed}-r{replicate}"


def _rolling_hash(previous: str, value: Mapping[str, Any]) -> str:
    return canonical_sha256({"previous": previous, "value": value})


def run_pilot_trial(
    graph: RuntimeGraph,
    protocol: PilotProtocol,
    arena: ArenaSpec,
    condition: PilotCondition,
    *,
    seed: int,
    replicate: int,
) -> dict[str, Any]:
    """Execute one deterministic local fixture-physics trial."""

    run_id = _trial_run_id(arena, condition, seed, replicate)
    controller, controller_hashes = _controller_for_condition(
        graph, condition, protocol_id=protocol.protocol_id, seed=seed, run_id=run_id
    )
    trial_identity = TrialIdentity(
        run_id=run_id,
        mode=condition.mode,
        data_tier=DataTier.MALECNS_FIXTURE,
        seed=seed,
        graph_sha256=graph.manifest.runtime_identity_sha256,
        adapter_sha256=controller_hashes["adapter_sha256"],
        dynamics_sha256=controller_hashes["dynamics_sha256"],
        decoder_sha256=controller_hashes["decoder_sha256"],
        arena_sha256=arena.sha256,
    )
    experiment_identity = canonical_sha256(
        {
            "condition_sha256": condition.identity_sha256,
            "physics_sha256": protocol.physics.identity_sha256,
            "protocol_sha256": protocol.sha256,
            "replicate": replicate,
            "trial_identity_sha256": trial_identity.identity_sha256,
        }
    )
    rng = random.Random(seed ^ (replicate << 32))
    world = _World(
        x=arena.spawn[0],
        y=arena.spawn[1],
        z=arena.spawn[2],
        vx=0.0,
        vy=0.0,
        vz=0.0,
        yaw=math.radians(arena.yaw_degrees) + rng.uniform(-0.02, 0.02),
    )
    exit_x = _exit_x(arena)
    start_x = world.x
    observations_sha256 = ZERO_SHA256
    actions_sha256 = ZERO_SHA256
    states_sha256 = ZERO_SHA256
    trace_sha256 = ZERO_SHA256
    first_non_neutral_tick: int | None = None
    motor_energy = 0.0
    ticks_elapsed = 0
    completed = False
    for tick in range(protocol.physics.tick_limit):
        sensors = _sensor_frame(world, arena, protocol.physics)
        policy_step = select_policy_action(
            condition,
            controller,
            sensors,
            sequence=tick,
        )
        observations_sha256 = _rolling_hash(
            observations_sha256,
            {"observation_sha256": policy_step.observation_sha256, "tick": tick},
        )
        actions_sha256 = _rolling_hash(
            actions_sha256,
            {"action_sha256": policy_step.action_sha256, "tick": tick},
        )
        states_sha256 = _rolling_hash(
            states_sha256,
            {"state_sha256": policy_step.state_sha256, "tick": tick},
        )
        trace_sha256 = _rolling_hash(
            trace_sha256,
            {
                "action_sha256": policy_step.action_sha256,
                "observation_sha256": policy_step.observation_sha256,
                "state_sha256": policy_step.state_sha256,
                "tick": tick,
            },
        )
        if first_non_neutral_tick is None and not policy_step.motors.is_neutral:
            first_non_neutral_tick = tick
        motor_energy += sum(value * value for value in policy_step.motors.to_dict().values())
        _step_physics(world, policy_step.motors, arena, protocol.physics)
        ticks_elapsed = tick + 1
        if world.x >= exit_x:
            completed = True
            break

    denominator = exit_x - start_x
    if denominator <= 0.0:
        raise ValueError("arena exit must be beyond spawn on the x axis")
    progress = (world.x - start_x) / denominator
    path_efficiency = max(0.0, world.x - start_x) / max(world.path_length, 1e-12)
    metrics: dict[str, Any] = {
        "collision_count": world.collision_count,
        "completion": int(completed),
        "fall_count": world.fall_count,
        "final_position_m": [world.x, world.y, world.z],
        "first_non_neutral_tick": first_non_neutral_tick,
        "motor_energy": motor_energy,
        "normalized_forward_progress": progress,
        "path_efficiency": path_efficiency,
        "path_length_m": world.path_length,
        "ticks_elapsed": ticks_elapsed,
    }
    return {
        "arena_id": arena.arena_id,
        "arena_sha256": arena.sha256,
        "condition_id": condition.condition_id,
        "condition_mode": condition.mode.value,
        "controller_execution": (
            "not_executed_scripted_baseline"
            if condition.mode is RunMode.SCRIPTED_BASELINE
            else (
                "canonical_recurrence_with_zeroed_adapter_inputs"
                if condition.mode is RunMode.NO_INPUT
                else controller.run_mode.value
                if controller is not None
                else "invalid"
            )
        ),
        "controller_hashes": dict(controller_hashes),
        "experiment_identity_sha256": experiment_identity,
        "hashes": {
            "action_stream_sha256": actions_sha256,
            "observation_stream_sha256": observations_sha256,
            "state_stream_sha256": states_sha256,
            "trace_stream_sha256": trace_sha256,
        },
        "intervention": condition.intervention,
        "metrics": metrics,
        "replicate": replicate,
        "run_id": run_id,
        "seed": seed,
        "status": "completed_fixture_physics_trial",
        "trial_identity_sha256": trial_identity.identity_sha256,
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty value sequence")
    return math.fsum(values) / len(values)


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of no values")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _paired_bootstrap(differences: Sequence[float], *, samples: int, seed: int) -> dict[str, Any]:
    if not differences:
        return {
            "n_pairs": 0,
            "mean_difference_faithful_minus_control": None,
            "bootstrap_95_percent_interval": None,
        }
    rng = random.Random(seed)
    count = len(differences)
    bootstrapped = sorted(
        _mean(tuple(differences[rng.randrange(count)] for _ in range(count)))
        for _ in range(samples)
    )
    return {
        "n_pairs": count,
        "mean_difference_faithful_minus_control": _mean(differences),
        "bootstrap_95_percent_interval": [
            _percentile(bootstrapped, 0.025),
            _percentile(bootstrapped, 0.975),
        ],
        "bootstrap_samples": samples,
    }


def summarize_trials(
    trials: Sequence[Mapping[str, Any]], protocol: PilotProtocol
) -> dict[str, Any]:
    completed = [
        trial for trial in trials if trial.get("status") == "completed_fixture_physics_trial"
    ]
    aggregate: list[dict[str, Any]] = []
    for arena_path in protocol.arena_paths:
        arena_id = Path(arena_path).stem
        for condition in protocol.conditions:
            members = [
                trial
                for trial in completed
                if trial["arena_id"] == arena_id and trial["condition_id"] == condition.condition_id
            ]
            metric_values = [
                _strict_mapping(trial["metrics"], "trial.metrics") for trial in members
            ]
            aggregate.append(
                {
                    "arena_id": arena_id,
                    "condition_id": condition.condition_id,
                    "completion_rate": (
                        _mean([float(value["completion"]) for value in metric_values])
                        if metric_values
                        else None
                    ),
                    "mean_collision_count": (
                        _mean([float(value["collision_count"]) for value in metric_values])
                        if metric_values
                        else None
                    ),
                    "mean_normalized_forward_progress": (
                        _mean([float(value[protocol.primary_metric]) for value in metric_values])
                        if metric_values
                        else None
                    ),
                    "mean_path_length_m": (
                        _mean([float(value["path_length_m"]) for value in metric_values])
                        if metric_values
                        else None
                    ),
                    "n_completed_trials": len(members),
                }
            )

    comparison_ids = (
        ("shuffled-weight", "faithful_vs_shuffled-weight"),
        ("degree-preserving", "faithful_vs_degree-preserving"),
        ("no-input", "faithful_vs_no-input"),
    )
    paired: list[dict[str, Any]] = []
    for arena_path in protocol.arena_paths:
        arena_id = Path(arena_path).stem
        by_cell = {
            (str(trial["condition_id"]), int(trial["seed"]), int(trial["replicate"])): trial
            for trial in completed
            if trial["arena_id"] == arena_id
        }
        for control_id, comparison_id in comparison_ids:
            differences: list[float] = []
            missing: list[str] = []
            for seed in protocol.seeds:
                for replicate in range(protocol.replicates):
                    faithful = by_cell.get(("faithful", seed, replicate))
                    control = by_cell.get((control_id, seed, replicate))
                    cell_id = f"{arena_id}:{seed}:r{replicate}"
                    if faithful is None or control is None:
                        missing.append(cell_id)
                        continue
                    faithful_metrics = _strict_mapping(faithful["metrics"], "faithful.metrics")
                    control_metrics = _strict_mapping(control["metrics"], "control.metrics")
                    differences.append(
                        float(faithful_metrics[protocol.primary_metric])
                        - float(control_metrics[protocol.primary_metric])
                    )
            summary = _paired_bootstrap(
                differences,
                samples=protocol.bootstrap_samples,
                seed=protocol.analysis_seed + len(paired),
            )
            paired.append(
                {
                    "arena_id": arena_id,
                    "comparison_id": comparison_id,
                    "missing_or_failed_cells": missing,
                    **summary,
                }
            )
    return {
        "aggregate_by_arena_condition": aggregate,
        "completed_trial_count": len(completed),
        "failed_trial_count": len(trials) - len(completed),
        "paired_primary_comparisons": paired,
        "uncertainty_boundary": (
            "Descriptive paired bootstrap intervals over five fixed fixture-pilot seeds; "
            "not inferential biological or Minecraft uncertainty."
        ),
    }


def build_pilot_result(protocol: PilotProtocol, *, repository_root: str | Path) -> dict[str, Any]:
    """Run every frozen pilot cell and return a self-hashing result envelope."""

    root = Path(repository_root).resolve()
    fixture_file = _contained_repository_path(root, protocol.fixture_path, "fixture.path")
    graph = load_malecns_fixture(fixture_file)
    arenas = [
        _verified_arena(root, path, digest)
        for path, digest in zip(protocol.arena_paths, protocol.arena_file_hashes, strict=True)
    ]
    trials: list[dict[str, Any]] = []
    for arena in arenas:
        for seed in protocol.seeds:
            for replicate in range(protocol.replicates):
                for condition in protocol.conditions:
                    try:
                        trial = run_pilot_trial(
                            graph,
                            protocol,
                            arena,
                            condition,
                            seed=seed,
                            replicate=replicate,
                        )
                    except Exception as exc:
                        trial = {
                            "arena_id": arena.arena_id,
                            "condition_id": condition.condition_id,
                            "condition_mode": condition.mode.value,
                            "error": {
                                "message": str(exc),
                                "type": type(exc).__name__,
                            },
                            "replicate": replicate,
                            "run_id": _trial_run_id(arena, condition, seed, replicate),
                            "seed": seed,
                            "status": "failed_fixture_physics_trial",
                        }
                    trials.append(trial)
    core: dict[str, Any] = {
        "claim_boundary": protocol.claim_boundary,
        "evidence_scope": "NON_MINECRAFT_LOCAL_FIXTURE_PHYSICS_ONLY",
        "fixture": {
            "canonical_graph_sha256": graph.manifest.canonical_graph_sha256,
            "edge_count": graph.manifest.edge_count,
            "file_sha256": protocol.fixture_file_sha256,
            "fixture_id": graph.manifest.fixture_id,
            "node_count": graph.manifest.node_count,
            "runtime_identity_sha256": graph.manifest.runtime_identity_sha256,
        },
        "planned_trial_count": protocol.planned_trial_count,
        "implementation_hashes": dict(protocol.implementation_hashes),
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256,
        "schema": RESULT_SCHEMA,
        "summary": summarize_trials(trials, protocol),
        "trials": trials,
    }
    return {
        "execution_environment": {
            "platform": platform.platform(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
        },
        "result": core,
        "result_sha256": canonical_sha256(core),
    }


def result_bytes(result: Mapping[str, Any]) -> bytes:
    """Canonical newline-terminated result bytes."""

    return canonical_bytes(result) + b"\n"


def write_result_no_clobber(path: str | Path, result: Mapping[str, Any]) -> str:
    """Write one result atomically with exclusive creation; never overwrite."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = result_bytes(result)
    if target.exists():
        raise FileExistsError(target)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        temporary.unlink()
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    return sha256(payload).hexdigest()


def verify_result_envelope(result: object) -> Mapping[str, Any]:
    value = _strict_mapping(result, "result envelope")
    _exact_keys(
        value,
        {"execution_environment", "result", "result_sha256"},
        "result envelope",
    )
    environment = _strict_mapping(value["execution_environment"], "execution_environment")
    _exact_keys(
        environment,
        {"platform", "python_implementation", "python_version"},
        "execution_environment",
    )
    for key, item in environment.items():
        _strict_string(item, f"execution_environment.{key}")
    core = _strict_mapping(value["result"], "result")
    expected = value["result_sha256"]
    if not _is_sha256(expected) or canonical_sha256(core) != expected:
        raise ValueError("result envelope SHA-256 mismatch")
    return value


def verify_reexecution(
    path: str | Path,
    protocol: PilotProtocol,
    *,
    repository_root: str | Path,
) -> str:
    """Reexecute all cells and require an exact platform-neutral result core.

    **This compares a stored result against the tree it is called on**, so it holds only while
    that tree still produces the recorded core. It is the check ``scripts/run_fixture_experiment``
    applies to a result it has just written; it is deliberately no longer part of the frozen-v3
    snapshot gate, which verifies recorded bytes on every platform instead. See
    :data:`RESULT_CORE_TREE_COMMIT`.
    """

    target = Path(path)
    existing = read_bounded_file_bytes(target, maximum_bytes=MAX_RESULT_BYTES, label="pilot result")
    parsed = verify_result_envelope(json.loads(existing))
    rebuilt = verify_result_envelope(build_pilot_result(protocol, repository_root=repository_root))
    if parsed["result"] != rebuilt["result"] or parsed["result_sha256"] != rebuilt["result_sha256"]:
        raise ValueError("stored pilot result core differs from deterministic reexecution")
    return sha256(existing).hexdigest()


def runtime_description() -> str:
    return f"{platform.python_implementation()} {sys.version.split()[0]}"
