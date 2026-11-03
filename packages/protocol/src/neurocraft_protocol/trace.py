"""Strict causal records and an immutable SHA-256-linked NDJSON container."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, ClassVar, cast

from .canonical import canonical_json_bytes, parse_json_frame
from .errors import ProtocolError, TraceIntegrityError
from .messages import (
    Action,
    ActuationReceipt,
    ActuationValidation,
    FaultCode,
    MotorVector,
    Observation,
    RunIdentity,
    _boolean,
    _enum,
    _exact_keys,
    _finite_number,
    _hash,
    _identifier,
    _mapping,
    _number_tuple,
    _strict_int,
    _text,
)

TRACE_SCHEMA = "neurocraft.trace.v2"
TRACE_RECORD_SCHEMA = "neurocraft.trace.tick.v2"
ZERO_HASH = "0" * 64
MAX_TRACE_LINE_BYTES = 1_048_576
_PROHIBITED_TRACE_KEYS = {
    "token",
    "authorization",
    "password",
    "secret",
    "chat",
    "account_material",
}


class RunMode(StrEnum):
    FAITHFUL = "faithful"
    SHUFFLED_WEIGHT = "shuffled_weight"
    DEGREE_PRESERVING_RANDOM = "degree_preserving_random"
    LESION = "lesion"
    SCRIPTED_BASELINE = "scripted_baseline"
    SYNTHETIC = "synthetic"
    NO_INPUT = "no_input"


class DataTier(StrEnum):
    SYNTHETIC = "SYNTHETIC"
    MALECNS_FIXTURE = "MALECNS_FIXTURE"
    MALECNS_CIRCUIT = "MALECNS_CIRCUIT"
    MALECNS_FULL = "MALECNS_FULL"


class ClaimScope(StrEnum):
    DESIGN_ONLY = "DESIGN_ONLY"
    G0 = "G0"
    G1 = "G1"
    G2 = "G2"
    G3 = "G3"
    G4 = "G4"
    G5 = "G5"
    G6 = "G6"


class ValidationResult(StrEnum):
    APPLIED = "APPLIED"
    NEUTRAL_FAULT = "NEUTRAL_FAULT"
    REJECTED = "REJECTED"


class FloatPrecision(StrEnum):
    FLOAT64 = "float64"
    FLOAT32 = "float32"


class DeterminismLevel(StrEnum):
    SAME_BUILD_EXACT = "same_build_exact"
    NUMERICAL_TOLERANCE = "numerical_tolerance"


@dataclass(frozen=True, slots=True)
class ArtifactIdentities:
    source_hash: str
    canonical_graph_hash: str
    runtime_graph_hash: str
    fixture_hash: str
    code_hash: str
    dependency_lock_hash: str
    adapter_hash: str
    dynamics_hash: str
    sign_hash: str
    decoder_hash: str
    arena_hash: str
    configuration_hash: str
    observation_schema_hash: str

    FIELDS: ClassVar[set[str]] = {
        "source_hash",
        "canonical_graph_hash",
        "runtime_graph_hash",
        "fixture_hash",
        "code_hash",
        "dependency_lock_hash",
        "adapter_hash",
        "dynamics_hash",
        "sign_hash",
        "decoder_hash",
        "arena_hash",
        "configuration_hash",
        "observation_schema_hash",
    }

    def __post_init__(self) -> None:
        for name in self.FIELDS:
            _hash(getattr(self, name), f"artifacts.{name}")

    @classmethod
    def from_dict(cls, raw: Any) -> ArtifactIdentities:
        value = _mapping(raw, "artifacts")
        _exact_keys(value, required=cls.FIELDS, name="artifacts")
        return cls(**{name: _hash(value[name], f"artifacts.{name}") for name in cls.FIELDS})

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in sorted(self.FIELDS)}


@dataclass(frozen=True, slots=True)
class TraceManifest:
    run_id: str
    initial_session_epoch: int
    mode: RunMode
    data_tier: DataTier
    claim_scope: ClaimScope
    artifacts: ArtifactIdentities
    game_tick_hz: float
    neural_step_ms: float
    precision: FloatPrecision
    backend: str
    thread_count: int
    seed: int
    determinism_level: DeterminismLevel
    machine: str
    os: str
    python: str
    java: str
    minecraft: str
    fabric: str
    mod: str
    privacy_policy: str
    expected_chunks: int
    checkpoint_interval_ticks: int

    FIELDS: ClassVar[set[str]] = {
        "trace_schema",
        "protocol_version",
        "run_id",
        "initial_session_epoch",
        "mode",
        "data_tier",
        "claim_scope",
        "artifacts",
        "game_tick_hz",
        "neural_step_ms",
        "precision",
        "backend",
        "thread_count",
        "seed",
        "determinism_level",
        "machine",
        "os",
        "python",
        "java",
        "minecraft",
        "fabric",
        "mod",
        "privacy_policy",
        "expected_chunks",
        "checkpoint_interval_ticks",
    }

    def __post_init__(self) -> None:
        _identifier(self.run_id, "manifest.run_id")
        _strict_int(self.initial_session_epoch, "manifest.initial_session_epoch")
        if not isinstance(self.mode, RunMode):
            raise ProtocolError("manifest.mode must be a RunMode")
        if not isinstance(self.data_tier, DataTier):
            raise ProtocolError("manifest.data_tier must be a DataTier")
        if not isinstance(self.claim_scope, ClaimScope):
            raise ProtocolError("manifest.claim_scope must be a ClaimScope")
        if not isinstance(self.artifacts, ArtifactIdentities):
            raise ProtocolError("manifest.artifacts must be ArtifactIdentities")
        _finite_number(self.game_tick_hz, "manifest.game_tick_hz", minimum=1, maximum=1000)
        _finite_number(self.neural_step_ms, "manifest.neural_step_ms", minimum=0.001, maximum=1000)
        if not isinstance(self.precision, FloatPrecision):
            raise ProtocolError("manifest.precision must be FloatPrecision")
        _identifier(self.backend, "manifest.backend")
        _strict_int(self.thread_count, "manifest.thread_count", minimum=1, maximum=1024)
        _strict_int(self.seed, "manifest.seed")
        if not isinstance(self.determinism_level, DeterminismLevel):
            raise ProtocolError("manifest.determinism_level must be DeterminismLevel")
        for name in ("machine", "os", "python", "java", "minecraft", "fabric", "mod"):
            _text(getattr(self, name), f"manifest.{name}", maximum=256)
        _identifier(self.privacy_policy, "manifest.privacy_policy")
        _strict_int(self.expected_chunks, "manifest.expected_chunks", minimum=1, maximum=1_000_000)
        _strict_int(
            self.checkpoint_interval_ticks,
            "manifest.checkpoint_interval_ticks",
            minimum=1,
        )

    @classmethod
    def from_dict(cls, raw: Any) -> TraceManifest:
        value = _mapping(raw, "manifest")
        _exact_keys(value, required=cls.FIELDS, name="manifest")
        if value["trace_schema"] != TRACE_SCHEMA:
            raise ProtocolError(f"manifest.trace_schema must be {TRACE_SCHEMA}")
        from .messages import PROTOCOL_VERSION

        if value["protocol_version"] != PROTOCOL_VERSION:
            raise ProtocolError(f"manifest.protocol_version must be {PROTOCOL_VERSION}")
        return cls(
            run_id=_identifier(value["run_id"], "manifest.run_id"),
            initial_session_epoch=_strict_int(
                value["initial_session_epoch"], "manifest.initial_session_epoch"
            ),
            mode=_enum(value["mode"], RunMode, "manifest.mode"),  # type: ignore[arg-type]
            data_tier=_enum(value["data_tier"], DataTier, "manifest.data_tier"),  # type: ignore[arg-type]
            claim_scope=_enum(value["claim_scope"], ClaimScope, "manifest.claim_scope"),  # type: ignore[arg-type]
            artifacts=ArtifactIdentities.from_dict(value["artifacts"]),
            game_tick_hz=_finite_number(
                value["game_tick_hz"], "manifest.game_tick_hz", minimum=1, maximum=1000
            ),
            neural_step_ms=_finite_number(
                value["neural_step_ms"], "manifest.neural_step_ms", minimum=0.001, maximum=1000
            ),
            precision=_enum(value["precision"], FloatPrecision, "manifest.precision"),  # type: ignore[arg-type]
            backend=_identifier(value["backend"], "manifest.backend"),
            thread_count=_strict_int(
                value["thread_count"], "manifest.thread_count", minimum=1, maximum=1024
            ),
            seed=_strict_int(value["seed"], "manifest.seed"),
            determinism_level=_enum(
                value["determinism_level"], DeterminismLevel, "manifest.determinism_level"
            ),  # type: ignore[arg-type]
            machine=_text(value["machine"], "manifest.machine", maximum=256),
            os=_text(value["os"], "manifest.os", maximum=256),
            python=_text(value["python"], "manifest.python", maximum=256),
            java=_text(value["java"], "manifest.java", maximum=256),
            minecraft=_text(value["minecraft"], "manifest.minecraft", maximum=256),
            fabric=_text(value["fabric"], "manifest.fabric", maximum=256),
            mod=_text(value["mod"], "manifest.mod", maximum=256),
            privacy_policy=_identifier(value["privacy_policy"], "manifest.privacy_policy"),
            expected_chunks=_strict_int(
                value["expected_chunks"], "manifest.expected_chunks", minimum=1, maximum=1_000_000
            ),
            checkpoint_interval_ticks=_strict_int(
                value["checkpoint_interval_ticks"],
                "manifest.checkpoint_interval_ticks",
                minimum=1,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        from .messages import PROTOCOL_VERSION

        return {
            "trace_schema": TRACE_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "run_id": self.run_id,
            "initial_session_epoch": self.initial_session_epoch,
            "mode": self.mode.value,
            "data_tier": self.data_tier.value,
            "claim_scope": self.claim_scope.value,
            "artifacts": self.artifacts.to_dict(),
            "game_tick_hz": self.game_tick_hz,
            "neural_step_ms": self.neural_step_ms,
            "precision": self.precision.value,
            "backend": self.backend,
            "thread_count": self.thread_count,
            "seed": self.seed,
            "determinism_level": self.determinism_level.value,
            "machine": self.machine,
            "os": self.os,
            "python": self.python,
            "java": self.java,
            "minecraft": self.minecraft,
            "fabric": self.fabric,
            "mod": self.mod,
            "privacy_policy": self.privacy_policy,
            "expected_chunks": self.expected_chunks,
            "checkpoint_interval_ticks": self.checkpoint_interval_ticks,
        }


@dataclass(frozen=True, slots=True)
class WorldState:
    position_m: tuple[float, float, float]
    velocity_m_s: tuple[float, float, float]
    yaw_rad: float
    pitch_rad: float
    on_ground: bool
    horizontal_collision: bool
    vertical_collision: bool
    in_fluid: bool

    FIELDS: ClassVar[set[str]] = {
        "position_m",
        "velocity_m_s",
        "yaw_rad",
        "pitch_rad",
        "on_ground",
        "horizontal_collision",
        "vertical_collision",
        "in_fluid",
    }

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_m",
            _number_tuple(
                self.position_m,
                "world.position_m",
                length=3,
                minimum=-30_000_000,
                maximum=30_000_000,
            ),
        )
        object.__setattr__(
            self,
            "velocity_m_s",
            _number_tuple(
                self.velocity_m_s,
                "world.velocity_m_s",
                length=3,
                minimum=-1024,
                maximum=1024,
            ),
        )
        _finite_number(self.yaw_rad, "world.yaw_rad", minimum=-3.141593, maximum=3.141593)
        _finite_number(self.pitch_rad, "world.pitch_rad", minimum=-1.570797, maximum=1.570797)
        for name in ("on_ground", "horizontal_collision", "vertical_collision", "in_fluid"):
            _boolean(getattr(self, name), f"world.{name}")

    @classmethod
    def from_dict(cls, raw: Any, *, name: str = "world") -> WorldState:
        value = _mapping(raw, name)
        _exact_keys(value, required=cls.FIELDS, name=name)
        return cls(
            position_m=_number_tuple(
                value["position_m"],
                f"{name}.position_m",
                length=3,
                minimum=-30_000_000,
                maximum=30_000_000,
            ),  # type: ignore[arg-type]
            velocity_m_s=_number_tuple(
                value["velocity_m_s"], f"{name}.velocity_m_s", length=3, minimum=-1024, maximum=1024
            ),  # type: ignore[arg-type]
            yaw_rad=_finite_number(
                value["yaw_rad"], f"{name}.yaw_rad", minimum=-3.141593, maximum=3.141593
            ),
            pitch_rad=_finite_number(
                value["pitch_rad"], f"{name}.pitch_rad", minimum=-1.570797, maximum=1.570797
            ),
            on_ground=_boolean(value["on_ground"], f"{name}.on_ground"),
            horizontal_collision=_boolean(
                value["horizontal_collision"], f"{name}.horizontal_collision"
            ),
            vertical_collision=_boolean(value["vertical_collision"], f"{name}.vertical_collision"),
            in_fluid=_boolean(value["in_fluid"], f"{name}.in_fluid"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_m": list(self.position_m),
            "velocity_m_s": list(self.velocity_m_s),
            "yaw_rad": self.yaw_rad,
            "pitch_rad": self.pitch_rad,
            "on_ground": self.on_ground,
            "horizontal_collision": self.horizontal_collision,
            "vertical_collision": self.vertical_collision,
            "in_fluid": self.in_fluid,
        }


def _receipt_world_state(receipt: ActuationReceipt) -> WorldState:
    frame = receipt.world_before
    return WorldState(
        position_m=frame.position_m,
        velocity_m_s=frame.velocity_m_s,
        yaw_rad=frame.yaw_rad,
        pitch_rad=frame.pitch_rad,
        on_ground=frame.on_ground,
        horizontal_collision=frame.horizontal_collision,
        vertical_collision=frame.vertical_collision,
        in_fluid=frame.in_fluid,
    )


def _named_values(raw: Any, name: str) -> dict[str, float]:
    value = _mapping(raw, name)
    if len(value) > 4096:
        raise ProtocolError(f"{name} exceeds 4096 entries")
    result: dict[str, float] = {}
    for key, item in value.items():
        result[_identifier(key, f"{name} key")] = _finite_number(
            item, f"{name}.{key}", minimum=-1_000_000, maximum=1_000_000
        )
    return result


def _activity_array(raw: Any, name: str) -> list[Any]:
    if not isinstance(raw, list) or not 1 <= len(raw) <= 4096:
        raise ProtocolError(f"record.{name} must be a bounded non-empty array or null")
    return raw


@dataclass(frozen=True, slots=True)
class CausalRecord:
    identity: RunIdentity
    record_seq: int
    game_tick: int
    observation_age_ticks: int
    observation: Observation
    encoded_stimulation: Mapping[str, float]
    direct_stimulation: Mapping[str, float]
    direct_stimulation_sha256: str
    neural_step_start: int
    neural_step_end: int
    state_before_digest: str
    state_after_digest: str
    state_before_activity: tuple[float, ...] | None
    state_after_activity: tuple[float, ...] | None
    interventions: tuple[str, ...]
    delay_state_digest: str
    checkpoint_reference: str | None
    motor_pool_values: Mapping[str, float]
    pool_membership_hash: str
    decoder_output: MotorVector
    issued_action: Action
    actuation_receipt: ActuationReceipt | None
    command_seq: int | None
    validation: ValidationResult
    clamp_fields: tuple[str, ...]
    applied_action: MotorVector
    neutral_reason: str | None
    world_before: WorldState
    world_after: WorldState
    world_after_observation_seq: int
    world_after_game_tick: int
    passive_forces: tuple[float, float, float]
    external_forces: tuple[float, float, float]
    queue_coalesced: int
    queue_overruns: int
    compute_ns: int
    serialization_ns: int
    write_ns: int
    fault_code: FaultCode | None

    FIELDS: ClassVar[set[str]] = {
        "record_schema",
        "identity",
        "record_seq",
        "game_tick",
        "observation_age_ticks",
        "observation",
        "encoded_stimulation",
        "direct_stimulation",
        "direct_stimulation_sha256",
        "neural_step_start",
        "neural_step_end",
        "state_before_digest",
        "state_after_digest",
        "state_before_activity",
        "state_after_activity",
        "interventions",
        "delay_state_digest",
        "checkpoint_reference",
        "motor_pool_values",
        "pool_membership_hash",
        "decoder_output",
        "issued_action",
        "actuation_receipt",
        "command_seq",
        "validation",
        "clamp_fields",
        "applied_action",
        "neutral_reason",
        "world_before",
        "world_after",
        "world_after_observation_seq",
        "world_after_game_tick",
        "passive_forces",
        "external_forces",
        "queue_coalesced",
        "queue_overruns",
        "compute_ns",
        "serialization_ns",
        "write_ns",
        "fault_code",
    }

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RunIdentity):
            raise ProtocolError("record.identity must be RunIdentity")
        _strict_int(self.record_seq, "record.record_seq")
        _strict_int(self.game_tick, "record.game_tick")
        _strict_int(self.observation_age_ticks, "record.observation_age_ticks")
        if not isinstance(self.observation, Observation):
            raise ProtocolError("record.observation must be Observation")
        if (
            self.observation.identity != self.identity
            or self.observation.game_tick != self.game_tick
        ):
            raise ProtocolError("record observation identity/tick does not match record")
        object.__setattr__(
            self,
            "encoded_stimulation",
            MappingProxyType(_named_values(self.encoded_stimulation, "record.encoded_stimulation")),
        )
        object.__setattr__(
            self,
            "direct_stimulation",
            MappingProxyType(_named_values(self.direct_stimulation, "record.direct_stimulation")),
        )
        _hash(self.direct_stimulation_sha256, "record.direct_stimulation_sha256")
        if bool(self.direct_stimulation) == (self.direct_stimulation_sha256 == ZERO_HASH):
            raise ProtocolError(
                "record direct stimulation values and identity must be present together"
            )
        _strict_int(self.neural_step_start, "record.neural_step_start")
        _strict_int(self.neural_step_end, "record.neural_step_end")
        if self.neural_step_end < self.neural_step_start:
            raise ProtocolError("record neural interval moves backward")
        for name in ("state_before_digest", "state_after_digest", "delay_state_digest"):
            _hash(getattr(self, name), f"record.{name}")
        if (self.state_before_activity is None) != (self.state_after_activity is None):
            raise ProtocolError("record full neural states must be present or absent together")
        for name in ("state_before_activity", "state_after_activity"):
            activity = getattr(self, name)
            if activity is None:
                continue
            if not isinstance(activity, tuple) or not 1 <= len(activity) <= 4096:
                raise ProtocolError(f"record.{name} must be a bounded non-empty tuple")
            object.__setattr__(
                self,
                name,
                tuple(
                    _finite_number(value, f"record.{name}[{index}]", minimum=-1.0, maximum=1.0)
                    for index, value in enumerate(activity)
                ),
            )
        if (
            self.state_before_activity is not None
            and self.state_after_activity is not None
            and len(self.state_before_activity) != len(self.state_after_activity)
        ):
            raise ProtocolError("record full neural state sizes differ")
        if not isinstance(self.interventions, tuple) or len(self.interventions) > 256:
            raise ProtocolError("record.interventions must be a bounded tuple")
        for item in self.interventions:
            _identifier(item, "record.intervention")
        if self.checkpoint_reference is not None:
            _identifier(self.checkpoint_reference, "record.checkpoint_reference")
        object.__setattr__(
            self,
            "motor_pool_values",
            MappingProxyType(_named_values(self.motor_pool_values, "record.motor_pool_values")),
        )
        _hash(self.pool_membership_hash, "record.pool_membership_hash")
        if not isinstance(self.decoder_output, MotorVector) or not isinstance(
            self.applied_action, MotorVector
        ):
            raise ProtocolError("record actions must be MotorVector values")
        if self.command_seq is None:
            raise ProtocolError("receipt-bound causal record requires a command sequence")
        if not isinstance(self.issued_action, Action):
            raise ProtocolError("record.issued_action must be Action")
        action = self.issued_action
        if (
            action.identity != self.identity
            or action.command_seq != self.command_seq
            or action.observation_seq != self.observation.observation_seq
            or action.observation_tick != self.observation.game_tick
            or action.neural_step_start != self.neural_step_start
            or action.neural_step_end != self.neural_step_end
            or action.state_digest != self.state_after_digest
            or action.motors != self.decoder_output
        ):
            raise ProtocolError("record issued action does not bind the causal fields")
        receipt = self.actuation_receipt
        if receipt is not None:
            if not isinstance(receipt, ActuationReceipt):
                raise ProtocolError("record.actuation_receipt must be ActuationReceipt or null")
            if (
                receipt.identity != self.identity
                or receipt.command_seq != action.command_seq
                or receipt.observation_seq != action.observation_seq
                or receipt.target_tick != action.target_tick
                or receipt.requested_motors != action.motors
                or receipt.applied_motors != self.applied_action
                or _receipt_world_state(receipt) != self.world_before
            ):
                raise ProtocolError("record actuation receipt does not bind action/world fields")
            if receipt.applied_tick != self.game_tick + self.observation_age_ticks:
                raise ProtocolError("record actuation receipt tick does not match observation age")
        if self.command_seq is not None:
            _strict_int(self.command_seq, "record.command_seq")
        if not isinstance(self.validation, ValidationResult):
            raise ProtocolError("record.validation must be ValidationResult")
        if not isinstance(self.clamp_fields, tuple) or len(set(self.clamp_fields)) != len(
            self.clamp_fields
        ):
            raise ProtocolError("record.clamp_fields must be a unique tuple")
        if any(item not in MotorVector.FIELDS for item in self.clamp_fields):
            raise ProtocolError("record.clamp_fields contains an unknown motor axis")
        if self.neutral_reason is not None:
            _identifier(self.neutral_reason, "record.neutral_reason")
        if self.validation is ValidationResult.APPLIED:
            if self.observation_age_ticks > 2:
                raise ProtocolError("applied action exceeds the two-tick freshness bound")
            if self.neutral_reason is not None or self.fault_code is not None:
                raise ProtocolError("applied record cannot carry neutral or fault metadata")
            decoder_values = self.decoder_output.to_dict()
            applied_values = self.applied_action.to_dict()
            if not self.clamp_fields and applied_values != decoder_values:
                raise ProtocolError("unclamped applied record must equal decoder output")
            for axis in MotorVector.FIELDS - set(self.clamp_fields):
                if applied_values[axis] != decoder_values[axis]:
                    raise ProtocolError("applied record changed an undeclared motor axis")
            if receipt is not None and receipt.validation not in {
                ActuationValidation.APPLIED,
                ActuationValidation.CLAMPED,
            }:
                raise ProtocolError("applied record carries a non-applied receipt")
        else:
            if not self.applied_action.is_neutral:
                raise ProtocolError("rejected or faulted record must apply neutral action")
            if self.clamp_fields:
                raise ProtocolError("rejected or faulted record cannot carry clamp fields")
            if self.neutral_reason is None or self.fault_code is None:
                raise ProtocolError("rejected or faulted record requires reason and fault code")
            if receipt is not None:
                expected_receipt_validation = (
                    ActuationValidation.NEUTRAL_FAULT
                    if self.validation is ValidationResult.NEUTRAL_FAULT
                    else ActuationValidation.REJECTED
                )
                if receipt.validation is not expected_receipt_validation:
                    raise ProtocolError("faulted record receipt validation is inconsistent")
        if not isinstance(self.world_before, WorldState) or not isinstance(
            self.world_after, WorldState
        ):
            raise ProtocolError("record world values must be WorldState")
        _strict_int(
            self.world_after_observation_seq,
            "record.world_after_observation_seq",
        )
        _strict_int(self.world_after_game_tick, "record.world_after_game_tick")
        if self.world_after_observation_seq <= self.observation.observation_seq:
            raise ProtocolError("record world-after observation sequence must move forward")
        applied_tick = self.game_tick + self.observation_age_ticks
        if self.world_after_game_tick != applied_tick + 1:
            raise ProtocolError("record world-after game tick must be the next authoritative tick")
        object.__setattr__(
            self,
            "passive_forces",
            _number_tuple(
                self.passive_forces,
                "record.passive_forces",
                length=3,
                minimum=-1_000_000,
                maximum=1_000_000,
            ),
        )
        object.__setattr__(
            self,
            "external_forces",
            _number_tuple(
                self.external_forces,
                "record.external_forces",
                length=3,
                minimum=-1_000_000,
                maximum=1_000_000,
            ),
        )
        for name in (
            "queue_coalesced",
            "queue_overruns",
            "compute_ns",
            "serialization_ns",
            "write_ns",
        ):
            _strict_int(getattr(self, name), f"record.{name}")
        if self.fault_code is not None and not isinstance(self.fault_code, FaultCode):
            raise ProtocolError("record.fault_code must be FaultCode or null")

    @classmethod
    def from_dict(cls, raw: Any) -> CausalRecord:
        value = _mapping(raw, "record")
        _exact_keys(value, required=cls.FIELDS, name="record")
        if value["record_schema"] != TRACE_RECORD_SCHEMA:
            raise ProtocolError(f"record.record_schema must be {TRACE_RECORD_SCHEMA}")
        interventions = value["interventions"]
        clamps = value["clamp_fields"]
        if not isinstance(interventions, list) or not all(
            isinstance(item, str) for item in interventions
        ):
            raise ProtocolError("record.interventions must be a string array")
        if not isinstance(clamps, list) or not all(isinstance(item, str) for item in clamps):
            raise ProtocolError("record.clamp_fields must be a string array")
        command_seq = value["command_seq"]
        checkpoint_reference = value["checkpoint_reference"]
        neutral_reason = value["neutral_reason"]
        fault_code = value["fault_code"]
        return cls(
            identity=RunIdentity.from_dict(value["identity"]),
            record_seq=_strict_int(value["record_seq"], "record.record_seq"),
            game_tick=_strict_int(value["game_tick"], "record.game_tick"),
            observation_age_ticks=_strict_int(
                value["observation_age_ticks"], "record.observation_age_ticks"
            ),
            observation=Observation.from_dict(value["observation"]),
            encoded_stimulation=_named_values(
                value["encoded_stimulation"], "record.encoded_stimulation"
            ),
            direct_stimulation=_named_values(
                value["direct_stimulation"], "record.direct_stimulation"
            ),
            direct_stimulation_sha256=_hash(
                value["direct_stimulation_sha256"],
                "record.direct_stimulation_sha256",
            ),
            neural_step_start=_strict_int(value["neural_step_start"], "record.neural_step_start"),
            neural_step_end=_strict_int(value["neural_step_end"], "record.neural_step_end"),
            state_before_digest=_hash(value["state_before_digest"], "record.state_before_digest"),
            state_after_digest=_hash(value["state_after_digest"], "record.state_after_digest"),
            state_before_activity=(
                None
                if value["state_before_activity"] is None
                else tuple(
                    _finite_number(
                        item,
                        f"record.state_before_activity[{index}]",
                        minimum=-1.0,
                        maximum=1.0,
                    )
                    for index, item in enumerate(
                        _activity_array(value["state_before_activity"], "state_before_activity")
                    )
                )
            ),
            state_after_activity=(
                None
                if value["state_after_activity"] is None
                else tuple(
                    _finite_number(
                        item,
                        f"record.state_after_activity[{index}]",
                        minimum=-1.0,
                        maximum=1.0,
                    )
                    for index, item in enumerate(
                        _activity_array(value["state_after_activity"], "state_after_activity")
                    )
                )
            ),
            interventions=tuple(interventions),
            delay_state_digest=_hash(value["delay_state_digest"], "record.delay_state_digest"),
            checkpoint_reference=(
                None
                if checkpoint_reference is None
                else _identifier(checkpoint_reference, "record.checkpoint_reference")
            ),
            motor_pool_values=_named_values(value["motor_pool_values"], "record.motor_pool_values"),
            pool_membership_hash=_hash(
                value["pool_membership_hash"], "record.pool_membership_hash"
            ),
            decoder_output=MotorVector.from_dict(
                value["decoder_output"], name="record.decoder_output"
            ),
            issued_action=Action.from_dict(value["issued_action"]),
            actuation_receipt=(
                None
                if value["actuation_receipt"] is None
                else ActuationReceipt.from_dict(value["actuation_receipt"])
            ),
            command_seq=(
                None if command_seq is None else _strict_int(command_seq, "record.command_seq")
            ),
            validation=_enum(value["validation"], ValidationResult, "record.validation"),  # type: ignore[arg-type]
            clamp_fields=tuple(clamps),
            applied_action=MotorVector.from_dict(
                value["applied_action"], name="record.applied_action"
            ),
            neutral_reason=(
                None
                if neutral_reason is None
                else _identifier(neutral_reason, "record.neutral_reason")
            ),
            world_before=WorldState.from_dict(value["world_before"], name="record.world_before"),
            world_after=WorldState.from_dict(value["world_after"], name="record.world_after"),
            world_after_observation_seq=_strict_int(
                value["world_after_observation_seq"],
                "record.world_after_observation_seq",
            ),
            world_after_game_tick=_strict_int(
                value["world_after_game_tick"], "record.world_after_game_tick"
            ),
            passive_forces=_number_tuple(
                value["passive_forces"],
                "record.passive_forces",
                length=3,
                minimum=-1_000_000,
                maximum=1_000_000,
            ),  # type: ignore[arg-type]
            external_forces=_number_tuple(
                value["external_forces"],
                "record.external_forces",
                length=3,
                minimum=-1_000_000,
                maximum=1_000_000,
            ),  # type: ignore[arg-type]
            queue_coalesced=_strict_int(value["queue_coalesced"], "record.queue_coalesced"),
            queue_overruns=_strict_int(value["queue_overruns"], "record.queue_overruns"),
            compute_ns=_strict_int(value["compute_ns"], "record.compute_ns"),
            serialization_ns=_strict_int(value["serialization_ns"], "record.serialization_ns"),
            write_ns=_strict_int(value["write_ns"], "record.write_ns"),
            fault_code=(
                None
                if fault_code is None
                else cast(FaultCode, _enum(fault_code, FaultCode, "record.fault_code"))
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_schema": TRACE_RECORD_SCHEMA,
            "identity": self.identity.to_dict(),
            "record_seq": self.record_seq,
            "game_tick": self.game_tick,
            "observation_age_ticks": self.observation_age_ticks,
            "observation": self.observation.to_dict(),
            "encoded_stimulation": dict(self.encoded_stimulation),
            "direct_stimulation": dict(self.direct_stimulation),
            "direct_stimulation_sha256": self.direct_stimulation_sha256,
            "neural_step_start": self.neural_step_start,
            "neural_step_end": self.neural_step_end,
            "state_before_digest": self.state_before_digest,
            "state_after_digest": self.state_after_digest,
            "state_before_activity": (
                None if self.state_before_activity is None else list(self.state_before_activity)
            ),
            "state_after_activity": (
                None if self.state_after_activity is None else list(self.state_after_activity)
            ),
            "interventions": list(self.interventions),
            "delay_state_digest": self.delay_state_digest,
            "checkpoint_reference": self.checkpoint_reference,
            "motor_pool_values": dict(self.motor_pool_values),
            "pool_membership_hash": self.pool_membership_hash,
            "decoder_output": self.decoder_output.to_dict(),
            "issued_action": self.issued_action.to_dict(),
            "actuation_receipt": (
                None if self.actuation_receipt is None else self.actuation_receipt.to_dict()
            ),
            "command_seq": self.command_seq,
            "validation": self.validation.value,
            "clamp_fields": list(self.clamp_fields),
            "applied_action": self.applied_action.to_dict(),
            "neutral_reason": self.neutral_reason,
            "world_before": self.world_before.to_dict(),
            "world_after": self.world_after.to_dict(),
            "world_after_observation_seq": self.world_after_observation_seq,
            "world_after_game_tick": self.world_after_game_tick,
            "passive_forces": list(self.passive_forces),
            "external_forces": list(self.external_forces),
            "queue_coalesced": self.queue_coalesced,
            "queue_overruns": self.queue_overruns,
            "compute_ns": self.compute_ns,
            "serialization_ns": self.serialization_ns,
            "write_ns": self.write_ns,
            "fault_code": None if self.fault_code is None else self.fault_code.value,
        }


def _scan_trace_privacy(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key.lower() in _PROHIBITED_TRACE_KEYS:
                raise ProtocolError(f"prohibited secret/private field at {path}.{key}")
            _scan_trace_privacy(item, path=f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _scan_trace_privacy(item, path=f"{path}[{index}]")
    elif isinstance(value, str) and value.startswith(("/Users/", "/home/", "C:\\Users\\")):
        raise ProtocolError(f"absolute user path is prohibited at {path}")


def _entry_hash(kind: str, previous_hash: str, payload: Mapping[str, Any]) -> str:
    material = {"kind": kind, "payload": payload, "previous_hash": previous_hash}
    return hashlib.sha256(canonical_json_bytes(material)).hexdigest()


@dataclass(frozen=True, slots=True)
class TraceVerification:
    path: Path
    record_count: int
    first_tick: int | None
    last_tick: int | None
    manifest_hash: str
    final_hash: str
    file_sha256: str
    manifest: TraceManifest


_RECEIPT_GATED_CLAIM_SCOPES = frozenset(
    {ClaimScope.G3, ClaimScope.G4, ClaimScope.G5, ClaimScope.G6}
)


def _validate_record_policy(manifest: TraceManifest, record: CausalRecord) -> None:
    """Apply manifest-dependent gates identically while writing and verifying."""

    if manifest.data_tier in {DataTier.SYNTHETIC, DataTier.MALECNS_FIXTURE} and (
        record.state_before_activity is None or record.state_after_activity is None
    ):
        raise ProtocolError("small-fixture traces require full neural state")
    if manifest.claim_scope in _RECEIPT_GATED_CLAIM_SCOPES:
        if record.actuation_receipt is None:
            raise ProtocolError("G3-or-higher traces require the canonical actuation receipt frame")
        timings = (record.compute_ns, record.serialization_ns, record.write_ns)
        if any(value <= 0 for value in timings):
            raise ProtocolError("G3-or-higher traces require positive measured operational timings")


class TraceWriter:
    """Write one complete trace to a private temp file, then publish without overwrite."""

    def __init__(self, path: str | Path, manifest: TraceManifest) -> None:
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(f"trace already exists: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        os.chmod(temporary, 0o600)
        self._temporary = Path(temporary)
        self._stream: BinaryIO = os.fdopen(descriptor, "wb")
        self.manifest = manifest
        self._previous_hash = ZERO_HASH
        self._manifest_hash = ""
        self._record_count = 0
        self._last_record_hash = ""
        self._last_tick = -1
        self._last_session = manifest.initial_session_epoch
        self._closed = False
        try:
            self._manifest_hash = self._write_entry("manifest", manifest.to_dict())
        except Exception:
            self._stream.close()
            self._temporary.unlink(missing_ok=True)
            self._closed = True
            raise

    def _write_entry(self, kind: str, payload: Mapping[str, Any]) -> str:
        _scan_trace_privacy(payload)
        digest = _entry_hash(kind, self._previous_hash, payload)
        envelope = {
            "kind": kind,
            "payload": payload,
            "previous_hash": self._previous_hash,
            "entry_hash": digest,
        }
        raw = canonical_json_bytes(envelope)
        if len(raw) > MAX_TRACE_LINE_BYTES:
            raise ProtocolError(f"trace entry exceeds {MAX_TRACE_LINE_BYTES} bytes")
        self._stream.write(raw + b"\n")
        self._previous_hash = digest
        return digest

    def append(self, record: CausalRecord) -> str:
        if self._closed:
            raise ProtocolError("trace writer is closed")
        if record.record_seq != self._record_count:
            raise ProtocolError(
                f"record sequence must be {self._record_count}, got {record.record_seq}"
            )
        if record.identity.run_id != self.manifest.run_id:
            raise ProtocolError("record run ID does not match manifest")
        if record.identity.session_epoch < self._last_session:
            raise ProtocolError("record session epoch moved backward")
        if record.game_tick <= self._last_tick:
            raise ProtocolError("record game tick must increase")
        expected = self.manifest.artifacts
        identity = record.identity
        if (
            identity.graph_hash != expected.runtime_graph_hash
            or identity.adapter_hash != expected.adapter_hash
            or identity.dynamics_hash != expected.dynamics_hash
            or identity.decoder_hash != expected.decoder_hash
            or identity.observation_schema_hash != expected.observation_schema_hash
        ):
            raise ProtocolError("record artifact identity does not match manifest")
        _validate_record_policy(self.manifest, record)
        digest = self._write_entry("record", record.to_dict())
        self._record_count += 1
        self._last_record_hash = digest
        self._last_tick = record.game_tick
        self._last_session = record.identity.session_epoch
        return digest

    def close(self) -> Path:
        if self._closed:
            return self.path
        footer = {
            "record_count": self._record_count,
            "last_record_hash": self._last_record_hash or self._manifest_hash,
            "manifest_hash": self._manifest_hash,
            "final_record_seq": None if self._record_count == 0 else self._record_count - 1,
        }
        self._write_entry("footer", footer)
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        try:
            # Atomic, no-clobber publication. The temp link is removed only after
            # the durable final name exists.
            os.link(self._temporary, self.path)
            self._temporary.unlink()
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except Exception:
            self._temporary.unlink(missing_ok=True)
            raise
        self._closed = True
        return self.path

    def abort(self) -> None:
        if self._closed:
            return
        self._stream.close()
        self._temporary.unlink(missing_ok=True)
        self._closed = True

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def verify_trace(path: str | Path) -> TraceVerification:
    trace_path = Path(path)
    previous_hash = ZERO_HASH
    manifest: TraceManifest | None = None
    manifest_hash = ""
    record_count = 0
    first_tick: int | None = None
    last_tick: int | None = None
    last_session = -1
    footer_seen = False
    file_hasher = hashlib.sha256()

    with trace_path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            file_hasher.update(raw_line)
            if len(raw_line) > MAX_TRACE_LINE_BYTES + 1:
                raise TraceIntegrityError(f"line {line_number} exceeds the trace bound")
            if not raw_line.endswith(b"\n"):
                raise TraceIntegrityError(f"line {line_number} is not newline-terminated")
            raw = raw_line[:-1]
            if not raw:
                raise TraceIntegrityError(f"line {line_number} is blank")
            try:
                envelope = parse_json_frame(raw, max_bytes=MAX_TRACE_LINE_BYTES)
                _exact_keys(
                    envelope,
                    required={"kind", "payload", "previous_hash", "entry_hash"},
                    name=f"trace line {line_number}",
                )
                if canonical_json_bytes(envelope) != raw:
                    raise TraceIntegrityError(f"line {line_number} is not canonical JSON")
                kind = envelope["kind"]
                payload = _mapping(envelope["payload"], f"line {line_number}.payload")
                linked_previous = _hash(
                    envelope["previous_hash"], f"line {line_number}.previous_hash"
                )
                observed_hash = _hash(envelope["entry_hash"], f"line {line_number}.entry_hash")
            except ProtocolError as exc:
                if isinstance(exc, TraceIntegrityError):
                    raise
                raise TraceIntegrityError(f"invalid trace line {line_number}: {exc}") from exc
            if footer_seen:
                raise TraceIntegrityError("content appears after the footer")
            if linked_previous != previous_hash:
                raise TraceIntegrityError(f"chain link mismatch at line {line_number}")
            computed_hash = _entry_hash(kind, previous_hash, payload)
            if observed_hash != computed_hash:
                raise TraceIntegrityError(f"entry hash mismatch at line {line_number}")
            _scan_trace_privacy(payload)

            if line_number == 1:
                if kind != "manifest":
                    raise TraceIntegrityError("first trace entry must be the manifest")
                manifest = TraceManifest.from_dict(payload)
                manifest_hash = observed_hash
                last_session = manifest.initial_session_epoch
            elif kind == "record":
                if manifest is None:
                    raise TraceIntegrityError("record precedes manifest")
                try:
                    record = CausalRecord.from_dict(payload)
                    _validate_record_policy(manifest, record)
                except ProtocolError as exc:
                    raise TraceIntegrityError(
                        f"invalid trace record at line {line_number}: {exc}"
                    ) from exc
                if record.record_seq != record_count:
                    raise TraceIntegrityError(
                        "record sequence is missing, duplicated, or reordered"
                    )
                if record.identity.run_id != manifest.run_id:
                    raise TraceIntegrityError("record run ID does not match manifest")
                if record.identity.session_epoch < last_session:
                    raise TraceIntegrityError("record session epoch moved backward")
                if last_tick is not None and record.game_tick <= last_tick:
                    raise TraceIntegrityError("record game tick did not increase")
                artifacts = manifest.artifacts
                identity = record.identity
                if (
                    identity.graph_hash != artifacts.runtime_graph_hash
                    or identity.adapter_hash != artifacts.adapter_hash
                    or identity.dynamics_hash != artifacts.dynamics_hash
                    or identity.decoder_hash != artifacts.decoder_hash
                    or identity.observation_schema_hash != artifacts.observation_schema_hash
                ):
                    raise TraceIntegrityError("record artifact identity does not match manifest")
                first_tick = record.game_tick if first_tick is None else first_tick
                last_tick = record.game_tick
                last_session = record.identity.session_epoch
                record_count += 1
            elif kind == "footer":
                if manifest is None:
                    raise TraceIntegrityError("footer precedes manifest")
                _exact_keys(
                    payload,
                    required={
                        "record_count",
                        "last_record_hash",
                        "manifest_hash",
                        "final_record_seq",
                    },
                    name="footer",
                )
                if _strict_int(payload["record_count"], "footer.record_count") != record_count:
                    raise TraceIntegrityError("footer record count mismatch")
                expected_last = manifest_hash if record_count == 0 else previous_hash
                if _hash(payload["last_record_hash"], "footer.last_record_hash") != expected_last:
                    raise TraceIntegrityError("footer last-record hash mismatch")
                if _hash(payload["manifest_hash"], "footer.manifest_hash") != manifest_hash:
                    raise TraceIntegrityError("footer manifest hash mismatch")
                expected_seq = None if record_count == 0 else record_count - 1
                if payload["final_record_seq"] != expected_seq:
                    raise TraceIntegrityError("footer final record sequence mismatch")
                footer_seen = True
            else:
                raise TraceIntegrityError(f"unexpected entry kind at line {line_number}")
            previous_hash = observed_hash

    if manifest is None:
        raise TraceIntegrityError("trace is empty")
    if not footer_seen:
        raise TraceIntegrityError("trace footer is missing")
    return TraceVerification(
        path=trace_path,
        record_count=record_count,
        first_tick=first_tick,
        last_tick=last_tick,
        manifest_hash=manifest_hash,
        final_hash=previous_hash,
        file_sha256=file_hasher.hexdigest(),
        manifest=manifest,
    )
