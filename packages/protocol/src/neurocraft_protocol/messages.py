"""Strict dataclass model for the NeuroCraft Fly transport-neutral protocol v2."""

from __future__ import annotations

import base64
import hashlib
import hmac
import math
import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Any, ClassVar, cast

from .budget import ACTION_TARGET_OFFSET_TICKS
from .errors import ProtocolError, UnknownFieldError
from .event_sensors import EventSample
from .sensor_contract import (
    SENSOR_SCHEMA_V1,
    SENSOR_SCHEMA_V2,
    SENSOR_SCHEMA_V3,
    SENSOR_SCHEMA_V4,
    sensor_schema_for_identity,
)

PROTOCOL_VERSION = "neurocraft.protocol.v2"
WEBSOCKET_SUBPROTOCOL = "neurocraft.v2"
RAY_COUNT = 6
MAX_FAULT_MESSAGE_CHARS = 256
MAX_TOKEN_CHARS = 256
HANDSHAKE_NONCE_BYTES = 32
HANDSHAKE_NONCE_CHARS = 43
HANDSHAKE_LENGTH_PREFIX_BYTES = 4

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_\-]{32,256}$")
_NONCE_RE = re.compile(rf"^[A-Za-z0-9_\-]{{{HANDSHAKE_NONCE_CHARS}}}$")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ProtocolError(f"{name} keys must be strings")
    return cast(Mapping[str, Any], value)


def _exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    name: str,
) -> None:
    optional = optional or set()
    missing = required - value.keys()
    if missing:
        raise ProtocolError(f"{name} is missing fields: {', '.join(sorted(missing))}")
    unknown = value.keys() - required - optional
    if unknown:
        raise UnknownFieldError(f"{name} has unknown fields: {', '.join(sorted(unknown))}")


def _strict_int(value: Any, name: str, *, minimum: int = 0, maximum: int = 2**63 - 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ProtocolError(f"{name} must be between {minimum} and {maximum}")
    return int(value)


def _finite_number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ProtocolError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ProtocolError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ProtocolError(f"{name} must be at most {maximum}")
    return 0.0 if result == 0.0 else result


def _text(value: Any, name: str, *, maximum: int = 128) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ProtocolError(f"{name} must be a non-empty string of at most {maximum} characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ProtocolError(f"{name} contains control characters")
    return value


def _identifier(value: Any, name: str) -> str:
    result = _text(value, name)
    if not _ID_RE.fullmatch(result):
        raise ProtocolError(f"{name} is not a valid protocol identifier")
    return result


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise ProtocolError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ProtocolError(f"{name} must be boolean")
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> Enum:
    if not isinstance(value, str):
        raise ProtocolError(f"{name} must be a string enum")
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise ProtocolError(f"{name} must be one of: {allowed}") from exc


def _number_tuple(
    value: Any,
    name: str,
    *,
    length: int,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[float, ...]:
    if not isinstance(value, list | tuple) or len(value) != length:
        raise ProtocolError(f"{name} must contain exactly {length} numbers")
    return tuple(
        _finite_number(item, f"{name}[{index}]", minimum=minimum, maximum=maximum)
        for index, item in enumerate(value)
    )


def _protocol_header(value: Mapping[str, Any], expected_type: str) -> None:
    if value.get("type") != expected_type:
        raise ProtocolError(f"message type must be {expected_type}")
    if value.get("protocol") != PROTOCOL_VERSION:
        raise ProtocolError(f"protocol must be {PROTOCOL_VERSION}")


class PeerRole(StrEnum):
    MINECRAFT_SERVER = "minecraft_server"
    NEURAL_SERVICE = "neural_service"


class FaultCode(StrEnum):
    MALFORMED = "MALFORMED"
    OVERSIZE = "OVERSIZE"
    UNKNOWN_FIELD = "UNKNOWN_FIELD"
    UNAUTHORIZED = "UNAUTHORIZED"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
    SEQUENCE = "SEQUENCE"
    STALE = "STALE"
    FUTURE = "FUTURE"
    QUEUE_OVERFLOW = "QUEUE_OVERFLOW"
    TRACE_FAILURE = "TRACE_FAILURE"
    INTERNAL = "INTERNAL"
    DISCONNECTED = "DISCONNECTED"
    CONTROL_UNAVAILABLE = "CONTROL_UNAVAILABLE"
    INVALID_STATE = "INVALID_STATE"
    RECEIPT_MISMATCH = "RECEIPT_MISMATCH"


class ActuationValidation(StrEnum):
    APPLIED = "APPLIED"
    CLAMPED = "CLAMPED"
    NEUTRAL_FAULT = "NEUTRAL_FAULT"
    REJECTED = "REJECTED"


class ControlOperation(StrEnum):
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    SINGLE_STEP = "SINGLE_STEP"
    RESET = "RESET"
    RECORD_START = "RECORD_START"
    RECORD_STOP = "RECORD_STOP"
    REPLAY_START = "REPLAY_START"
    REPLAY_STOP = "REPLAY_STOP"
    LESION_APPLY = "LESION_APPLY"
    LESION_CLEAR = "LESION_CLEAR"
    STIMULATION_APPLY = "STIMULATION_APPLY"
    STIMULATION_CLEAR = "STIMULATION_CLEAR"


class ControlState(StrEnum):
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STEP_ARMED = "STEP_ARMED"
    RECORDING = "RECORDING"
    REPLAYING = "REPLAYING"
    FAULTED = "FAULTED"


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """Immutable identities repeated on every causal protocol frame."""

    run_id: str
    session_epoch: int
    entity_id: str
    graph_hash: str
    adapter_hash: str
    dynamics_hash: str
    decoder_hash: str
    observation_schema_hash: str

    FIELDS: ClassVar[set[str]] = {
        "run_id",
        "session_epoch",
        "entity_id",
        "graph_hash",
        "adapter_hash",
        "dynamics_hash",
        "decoder_hash",
        "observation_schema_hash",
    }

    def __post_init__(self) -> None:
        _identifier(self.run_id, "identity.run_id")
        _strict_int(self.session_epoch, "identity.session_epoch")
        _identifier(self.entity_id, "identity.entity_id")
        _hash(self.graph_hash, "identity.graph_hash")
        _hash(self.adapter_hash, "identity.adapter_hash")
        _hash(self.dynamics_hash, "identity.dynamics_hash")
        _hash(self.decoder_hash, "identity.decoder_hash")
        _hash(self.observation_schema_hash, "identity.observation_schema_hash")

    @classmethod
    def from_dict(cls, raw: Any) -> RunIdentity:
        value = _mapping(raw, "identity")
        _exact_keys(value, required=cls.FIELDS, name="identity")
        return cls(
            run_id=_identifier(value["run_id"], "identity.run_id"),
            session_epoch=_strict_int(value["session_epoch"], "identity.session_epoch"),
            entity_id=_identifier(value["entity_id"], "identity.entity_id"),
            graph_hash=_hash(value["graph_hash"], "identity.graph_hash"),
            adapter_hash=_hash(value["adapter_hash"], "identity.adapter_hash"),
            dynamics_hash=_hash(value["dynamics_hash"], "identity.dynamics_hash"),
            decoder_hash=_hash(value["decoder_hash"], "identity.decoder_hash"),
            observation_schema_hash=_hash(
                value["observation_schema_hash"], "identity.observation_schema_hash"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_epoch": self.session_epoch,
            "entity_id": self.entity_id,
            "graph_hash": self.graph_hash,
            "adapter_hash": self.adapter_hash,
            "dynamics_hash": self.dynamics_hash,
            "decoder_hash": self.decoder_hash,
            "observation_schema_hash": self.observation_schema_hash,
        }


@dataclass(frozen=True, slots=True)
class SensorFrame:
    """Versioned physical-only sensor vector; six rays are body-relative distances."""

    position_m: tuple[float, float, float]
    velocity_m_s: tuple[float, float, float]
    yaw_rad: float
    pitch_rad: float
    on_ground: bool
    horizontal_collision: bool
    vertical_collision: bool
    in_fluid: bool
    local_light: int
    ray_max_distance_m: float
    ray_distances_m: tuple[float, float, float, float, float, float]

    FIELDS: ClassVar[set[str]] = {
        "position_m",
        "velocity_m_s",
        "yaw_rad",
        "pitch_rad",
        "on_ground",
        "horizontal_collision",
        "vertical_collision",
        "in_fluid",
        "local_light",
        "ray_max_distance_m",
        "ray_distances_m",
    }

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_m",
            _number_tuple(
                self.position_m,
                "sensors.position_m",
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
                "sensors.velocity_m_s",
                length=3,
                minimum=-1024,
                maximum=1024,
            ),
        )
        _finite_number(self.yaw_rad, "sensors.yaw_rad", minimum=-math.pi, maximum=math.pi)
        _finite_number(
            self.pitch_rad, "sensors.pitch_rad", minimum=-math.pi / 2, maximum=math.pi / 2
        )
        _boolean(self.on_ground, "sensors.on_ground")
        _boolean(self.horizontal_collision, "sensors.horizontal_collision")
        _boolean(self.vertical_collision, "sensors.vertical_collision")
        _boolean(self.in_fluid, "sensors.in_fluid")
        _strict_int(self.local_light, "sensors.local_light", maximum=15)
        ray_max = _finite_number(
            self.ray_max_distance_m,
            "sensors.ray_max_distance_m",
            minimum=0.001,
            maximum=64.0,
        )
        object.__setattr__(
            self,
            "ray_distances_m",
            _number_tuple(
                self.ray_distances_m,
                "sensors.ray_distances_m",
                length=RAY_COUNT,
                minimum=0.0,
                maximum=ray_max,
            ),
        )

    @classmethod
    def from_dict(cls, raw: Any) -> SensorFrame:
        value = _mapping(raw, "sensors")
        _exact_keys(value, required=cls.FIELDS, name="sensors")
        ray_max = _finite_number(
            value["ray_max_distance_m"],
            "sensors.ray_max_distance_m",
            minimum=0.001,
            maximum=64.0,
        )
        return cls(
            position_m=_number_tuple(
                value["position_m"],
                "sensors.position_m",
                length=3,
                minimum=-30_000_000,
                maximum=30_000_000,
            ),  # type: ignore[arg-type]
            velocity_m_s=_number_tuple(
                value["velocity_m_s"],
                "sensors.velocity_m_s",
                length=3,
                minimum=-1024,
                maximum=1024,
            ),  # type: ignore[arg-type]
            yaw_rad=_finite_number(
                value["yaw_rad"], "sensors.yaw_rad", minimum=-math.pi, maximum=math.pi
            ),
            pitch_rad=_finite_number(
                value["pitch_rad"], "sensors.pitch_rad", minimum=-math.pi / 2, maximum=math.pi / 2
            ),
            on_ground=_boolean(value["on_ground"], "sensors.on_ground"),
            horizontal_collision=_boolean(
                value["horizontal_collision"], "sensors.horizontal_collision"
            ),
            vertical_collision=_boolean(value["vertical_collision"], "sensors.vertical_collision"),
            in_fluid=_boolean(value["in_fluid"], "sensors.in_fluid"),
            local_light=_strict_int(value["local_light"], "sensors.local_light", maximum=15),
            ray_max_distance_m=ray_max,
            ray_distances_m=_number_tuple(
                value["ray_distances_m"],
                "sensors.ray_distances_m",
                length=RAY_COUNT,
                minimum=0.0,
                maximum=ray_max,
            ),  # type: ignore[arg-type]
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
            "local_light": self.local_light,
            "ray_max_distance_m": self.ray_max_distance_m,
            "ray_distances_m": list(self.ray_distances_m),
        }


@dataclass(frozen=True, slots=True)
class SensorFrameV2(SensorFrame):
    """Typed rays for new profiles; the historical base frame stays byte-stable."""

    ray_hit_kinds: tuple[int, int, int, int, int, int]

    FIELDS: ClassVar[set[str]] = SensorFrame.FIELDS | {"sensor_schema", "ray_hit_kinds"}

    def __post_init__(self) -> None:
        SensorFrame.__post_init__(self)
        kinds = self.ray_hit_kinds
        if not isinstance(kinds, list | tuple) or len(kinds) != RAY_COUNT:
            raise ProtocolError("sensors.ray_hit_kinds must contain exactly six integers")
        object.__setattr__(
            self,
            "ray_hit_kinds",
            tuple(
                _strict_int(kind, f"sensors.ray_hit_kinds[{index}]", maximum=2)
                for index, kind in enumerate(kinds)
            ),
        )

    @classmethod
    def from_dict(cls, raw: Any) -> SensorFrameV2:
        value = _mapping(raw, "sensors")
        _exact_keys(value, required=cls.FIELDS, name="sensors")
        if value["sensor_schema"] != SENSOR_SCHEMA_V2:
            raise ProtocolError(f"sensors.sensor_schema must be {SENSOR_SCHEMA_V2}")
        base = SensorFrame.from_dict({name: value[name] for name in SensorFrame.FIELDS})
        return cls.from_sensor_frame(base, value["ray_hit_kinds"])

    @classmethod
    def from_sensor_frame(cls, frame: SensorFrame, kinds: Any) -> SensorFrameV2:
        """Attach explicitly supplied kinds; never infer them from distances."""

        return cls(
            **{name: getattr(frame, name) for name in SensorFrame.FIELDS}, ray_hit_kinds=kinds
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **SensorFrame.to_dict(self),
            "sensor_schema": SENSOR_SCHEMA_V2,
            "ray_hit_kinds": list(self.ray_hit_kinds),
        }


@dataclass(frozen=True, slots=True)
class SensorFrameV3(SensorFrameV2):
    """Typed rays and explicit world-event counters; old frames remain byte-stable."""

    events: EventSample

    FIELDS: ClassVar[set[str]] = SensorFrameV2.FIELDS | {"events"}

    def __post_init__(self) -> None:
        SensorFrameV2.__post_init__(self)
        if not isinstance(self.events, EventSample):
            raise ProtocolError("sensors.events must be an EventSample")

    @classmethod
    def from_dict(cls, raw: Any) -> SensorFrameV3:
        value = _mapping(raw, "sensors")
        _exact_keys(value, required=cls.FIELDS, name="sensors")
        if value["sensor_schema"] != SENSOR_SCHEMA_V3:
            raise ProtocolError(f"sensors.sensor_schema must be {SENSOR_SCHEMA_V3}")
        base = SensorFrame.from_dict({name: value[name] for name in SensorFrame.FIELDS})
        return cls(
            **{name: getattr(base, name) for name in SensorFrame.FIELDS},
            ray_hit_kinds=value["ray_hit_kinds"],
            events=EventSample.from_dict(value["events"]),
        )

    @classmethod
    def from_typed_frame(cls, frame: SensorFrameV2, events: EventSample) -> SensorFrameV3:
        return cls(
            **{name: getattr(frame, name) for name in SensorFrame.FIELDS},
            ray_hit_kinds=frame.ray_hit_kinds,
            events=events,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **SensorFrameV2.to_dict(self),
            "sensor_schema": SENSOR_SCHEMA_V3,
            "events": self.events.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SensorFrameV4(SensorFrameV3):
    """Actor class of each winning entity ray hit; existing event semantics remain intact."""

    ray_actor_kinds: tuple[int, int, int, int, int, int]

    FIELDS: ClassVar[set[str]] = SensorFrameV3.FIELDS | {"ray_actor_kinds"}

    def __post_init__(self) -> None:
        SensorFrameV3.__post_init__(self)
        actors = self.ray_actor_kinds
        if not isinstance(actors, list | tuple) or len(actors) != RAY_COUNT:
            raise ProtocolError("sensors.ray_actor_kinds must contain exactly six integers")
        checked = tuple(
            _strict_int(actor, f"sensors.ray_actor_kinds[{index}]", maximum=3)
            for index, actor in enumerate(actors)
        )
        for index, (actor, hit) in enumerate(zip(checked, self.ray_hit_kinds, strict=True)):
            if (actor == 0) != (hit != 2):
                raise ProtocolError(
                    f"sensors.ray_actor_kinds[{index}] must be 0 iff ray_hit_kinds[{index}] "
                    "is not 2; entity hits require actor kind 1, 2 or 3"
                )
        object.__setattr__(self, "ray_actor_kinds", checked)

    @classmethod
    def from_dict(cls, raw: Any) -> SensorFrameV4:
        value = _mapping(raw, "sensors")
        _exact_keys(value, required=cls.FIELDS, name="sensors")
        if value["sensor_schema"] != SENSOR_SCHEMA_V4:
            raise ProtocolError(f"sensors.sensor_schema must be {SENSOR_SCHEMA_V4}")
        base = SensorFrameV3.from_dict(
            {
                **{name: value[name] for name in SensorFrameV3.FIELDS},
                "sensor_schema": SENSOR_SCHEMA_V3,
            }
        )
        return cls.from_event_frame(base, value["ray_actor_kinds"])

    @classmethod
    def from_event_frame(cls, frame: SensorFrameV3, actors: Any) -> SensorFrameV4:
        """Attach supplied actor metadata; never infer entity class from a distance or event."""

        if not isinstance(frame, SensorFrameV3):
            raise ProtocolError("actor metadata requires a SensorFrameV3 event frame")
        return cls(
            **{name: getattr(frame, name) for name in SensorFrame.FIELDS},
            ray_hit_kinds=frame.ray_hit_kinds,
            events=frame.events,
            ray_actor_kinds=actors,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **SensorFrameV3.to_dict(self),
            "sensor_schema": SENSOR_SCHEMA_V4,
            "ray_actor_kinds": list(self.ray_actor_kinds),
        }


def parse_sensor_frame(raw: Any) -> SensorFrame:
    """Closed shapes; a missing/unknown discriminator never supplies default values."""

    value = _mapping(raw, "sensors")
    if value.get("sensor_schema") == SENSOR_SCHEMA_V4:
        return SensorFrameV4.from_dict(value)
    if value.get("sensor_schema") == SENSOR_SCHEMA_V3:
        return SensorFrameV3.from_dict(value)
    if "sensor_schema" in value:
        return SensorFrameV2.from_dict(value)
    return SensorFrame.from_dict(value)


def require_sensor_contract(sensors: SensorFrame, observation_schema_hash: str) -> None:
    """Refuse a frame whose shape differs from the trusted session contract."""

    expected = sensor_schema_for_identity(observation_schema_hash)
    observed = (
        SENSOR_SCHEMA_V4
        if isinstance(sensors, SensorFrameV4)
        else SENSOR_SCHEMA_V3
        if isinstance(sensors, SensorFrameV3)
        else SENSOR_SCHEMA_V2
        if isinstance(sensors, SensorFrameV2)
        else SENSOR_SCHEMA_V1
    )
    if observed != expected:
        raise ProtocolError(f"session requires {expected}, observation carries {observed}")


@dataclass(frozen=True, slots=True)
class MotorVector:
    """Only bounded primitive axes; semantic actions are intentionally absent."""

    forward_thrust: float = 0.0
    lateral_thrust: float = 0.0
    vertical_thrust: float = 0.0
    yaw_torque: float = 0.0
    pitch_torque: float = 0.0

    FIELDS: ClassVar[set[str]] = {
        "forward_thrust",
        "lateral_thrust",
        "vertical_thrust",
        "yaw_torque",
        "pitch_torque",
    }

    def __post_init__(self) -> None:
        for name in self.FIELDS:
            _finite_number(getattr(self, name), f"motors.{name}", minimum=-1.0, maximum=1.0)

    @classmethod
    def neutral(cls) -> MotorVector:
        return cls()

    @classmethod
    def from_dict(cls, raw: Any, *, name: str = "motors") -> MotorVector:
        value = _mapping(raw, name)
        _exact_keys(value, required=cls.FIELDS, name=name)
        return cls(
            **{
                field_name: _finite_number(
                    value[field_name], f"{name}.{field_name}", minimum=-1.0, maximum=1.0
                )
                for field_name in cls.FIELDS
            }
        )

    @property
    def is_neutral(self) -> bool:
        return all(getattr(self, name) == 0.0 for name in self.FIELDS)

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in sorted(self.FIELDS)}


def validate_run_token(token: Any) -> str:
    """Return ``token`` after enforcing the canonical shared-secret contract.

    The run token authenticates the control channel but is never transmitted: it is only
    ever used as an HMAC key. See :func:`handshake_transcript`.
    """

    if not isinstance(token, str) or not _TOKEN_RE.fullmatch(token):
        raise ProtocolError("run token must be 32-256 URL-safe characters")
    return token


def _nonce(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _NONCE_RE.fullmatch(value):
        raise ProtocolError(
            f"{name} must be {HANDSHAKE_NONCE_BYTES} random bytes in unpadded base64url"
        )
    return value


def generate_client_nonce() -> str:
    """Return a fresh unpadded base64url nonce over 32 cryptographically random bytes."""

    raw = secrets.token_bytes(HANDSHAKE_NONCE_BYTES)
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def handshake_transcript(
    *,
    label: str,
    identity: RunIdentity,
    role: PeerRole,
    client_nonce: str,
) -> bytes:
    """Return the canonical handshake transcript bytes that both endpoints authenticate.

    The layout is an unambiguous, length-prefixed concatenation of exactly twelve UTF-8
    strings in a frozen order. Every element is preceded by its byte length as a 4-byte
    big-endian unsigned integer, so no choice of field content can shift a boundary::

        transcript := element(protocol) || element(label)
                   || element(run_id) || element(session_epoch)
                   || element(entity_id) || element(graph_hash)
                   || element(adapter_hash) || element(dynamics_hash)
                   || element(decoder_hash) || element(observation_schema_hash)
                   || element(role) || element(client_nonce)
        element(s) := uint32_be(len(utf8(s))) || utf8(s)

    ``protocol`` is :data:`PROTOCOL_VERSION` and ``label`` is the message type (``hello``
    or ``hello_ack``), which together domain-separate this transcript from every other
    protocol revision and from the opposite handshake direction. ``session_epoch`` is
    rendered as its canonical base-10 ASCII form with no sign and no leading zeros.

    The Java bridge produces byte-identical transcripts in
    ``ControlProtocol.handshakeTranscript``; ``tests/protocol/test_messages.py`` and
    ``ControlProtocolTest`` pin the same vector on both sides.
    """

    elements = (
        PROTOCOL_VERSION,
        label,
        identity.run_id,
        str(identity.session_epoch),
        identity.entity_id,
        identity.graph_hash,
        identity.adapter_hash,
        identity.dynamics_hash,
        identity.decoder_hash,
        identity.observation_schema_hash,
        role.value,
        _nonce(client_nonce, "handshake.client_nonce"),
    )
    transcript = bytearray()
    for element in elements:
        encoded = element.encode("utf-8")
        transcript += len(encoded).to_bytes(HANDSHAKE_LENGTH_PREFIX_BYTES, "big")
        transcript += encoded
    return bytes(transcript)


def handshake_auth(
    token: str,
    *,
    label: str,
    identity: RunIdentity,
    role: PeerRole,
    client_nonce: str,
) -> str:
    """Return ``HMAC-SHA256(token, transcript)`` as a lowercase hex digest."""

    key = validate_run_token(token).encode("utf-8")
    transcript = handshake_transcript(
        label=label, identity=identity, role=role, client_nonce=client_nonce
    )
    return hmac.new(key, transcript, hashlib.sha256).hexdigest()


def verify_handshake_auth(
    token: str,
    candidate: Any,
    *,
    label: str,
    identity: RunIdentity,
    role: PeerRole,
    client_nonce: str,
) -> bool:
    """Constant-time check of a handshake proof against the locally derived expectation."""

    if not isinstance(candidate, str) or not _HASH_RE.fullmatch(candidate):
        return False
    expected = handshake_auth(
        token, label=label, identity=identity, role=role, client_nonce=client_nonce
    )
    return hmac.compare_digest(expected, candidate)


@dataclass(frozen=True, slots=True)
class Hello:
    """Opening handshake frame. It carries no token material of any kind.

    ``client_nonce`` is 32 fresh random bytes in unpadded base64url and ``auth`` is
    ``HMAC-SHA256(token, handshake_transcript(label="hello", ...))`` as lowercase hex.
    Possession of the run token is therefore proven without the token ever reaching the
    wire, and the proof is bound to this session's identity, role, and fresh nonce.
    """

    identity: RunIdentity
    role: PeerRole
    client_nonce: str
    auth: str

    TYPE: ClassVar[str] = "hello"
    FIELDS: ClassVar[set[str]] = {
        "type",
        "protocol",
        "identity",
        "role",
        "client_nonce",
        "auth",
    }

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RunIdentity):
            raise ProtocolError("hello.identity must be a RunIdentity")
        if not isinstance(self.role, PeerRole):
            raise ProtocolError("hello.role must be a PeerRole")
        _nonce(self.client_nonce, "hello.client_nonce")
        _hash(self.auth, "hello.auth")

    @classmethod
    def create(
        cls,
        *,
        identity: RunIdentity,
        role: PeerRole,
        token: str,
        client_nonce: str | None = None,
    ) -> Hello:
        """Build a hello that proves possession of ``token`` without disclosing it."""

        nonce = generate_client_nonce() if client_nonce is None else client_nonce
        return cls(
            identity=identity,
            role=role,
            client_nonce=nonce,
            auth=handshake_auth(
                token, label=cls.TYPE, identity=identity, role=role, client_nonce=nonce
            ),
        )

    def verify(self, token: str) -> bool:
        """Constant-time check that ``auth`` was produced by a holder of ``token``."""

        return verify_handshake_auth(
            token,
            self.auth,
            label=self.TYPE,
            identity=self.identity,
            role=self.role,
            client_nonce=self.client_nonce,
        )

    @classmethod
    def from_dict(cls, raw: Any) -> Hello:
        value = _mapping(raw, "hello")
        _exact_keys(value, required=cls.FIELDS, name="hello")
        _protocol_header(value, cls.TYPE)
        return cls(
            identity=RunIdentity.from_dict(value["identity"]),
            role=_enum(value["role"], PeerRole, "hello.role"),  # type: ignore[arg-type]
            client_nonce=_nonce(value["client_nonce"], "hello.client_nonce"),
            auth=_hash(value["auth"], "hello.auth"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.TYPE,
            "protocol": PROTOCOL_VERSION,
            "identity": self.identity.to_dict(),
            "role": self.role.value,
            "client_nonce": self.client_nonce,
            "auth": self.auth,
        }


@dataclass(frozen=True, slots=True)
class HelloAck:
    """Handshake reply proving the service also holds the run token.

    ``server_auth`` is ``HMAC-SHA256(token, handshake_transcript(label="hello_ack", ...))``
    over the client's nonce, so a peer that cannot compute it cannot impersonate the
    service. The nonce is deliberately not echoed: the client must already hold it, which
    makes a replayed acknowledgement useless against a fresh session.
    """

    identity: RunIdentity
    role: PeerRole
    server_auth: str
    accepted: bool = True

    TYPE: ClassVar[str] = "hello_ack"
    FIELDS: ClassVar[set[str]] = {
        "type",
        "protocol",
        "identity",
        "role",
        "server_auth",
        "accepted",
    }

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RunIdentity):
            raise ProtocolError("hello_ack.identity must be a RunIdentity")
        if self.role is not PeerRole.NEURAL_SERVICE:
            raise ProtocolError("hello_ack.role must be neural_service")
        _hash(self.server_auth, "hello_ack.server_auth")
        if self.accepted is not True:
            raise ProtocolError("rejected handshakes use a fault frame, not hello_ack")

    @classmethod
    def create(
        cls,
        *,
        identity: RunIdentity,
        role: PeerRole = PeerRole.NEURAL_SERVICE,
        token: str,
        client_nonce: str,
    ) -> HelloAck:
        """Build an acknowledgement that proves possession of ``token`` for this nonce."""

        return cls(
            identity=identity,
            role=role,
            server_auth=handshake_auth(
                token, label=cls.TYPE, identity=identity, role=role, client_nonce=client_nonce
            ),
            accepted=True,
        )

    def verify(self, token: str, *, client_nonce: str) -> bool:
        """Constant-time check that the service proved possession for ``client_nonce``."""

        return verify_handshake_auth(
            token,
            self.server_auth,
            label=self.TYPE,
            identity=self.identity,
            role=self.role,
            client_nonce=client_nonce,
        )

    @classmethod
    def from_dict(cls, raw: Any) -> HelloAck:
        value = _mapping(raw, "hello_ack")
        _exact_keys(value, required=cls.FIELDS, name="hello_ack")
        _protocol_header(value, cls.TYPE)
        return cls(
            identity=RunIdentity.from_dict(value["identity"]),
            role=_enum(value["role"], PeerRole, "hello_ack.role"),  # type: ignore[arg-type]
            server_auth=_hash(value["server_auth"], "hello_ack.server_auth"),
            accepted=_boolean(value["accepted"], "hello_ack.accepted"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.TYPE,
            "protocol": PROTOCOL_VERSION,
            "identity": self.identity.to_dict(),
            "role": self.role.value,
            "server_auth": self.server_auth,
            "accepted": self.accepted,
        }


@dataclass(frozen=True, slots=True)
class Observation:
    identity: RunIdentity
    observation_seq: int
    game_tick: int
    sent_monotonic_ns: int
    sensors: SensorFrame
    bridge_observations_coalesced: int = 0
    missing_fields: tuple[str, ...] = ()
    clamped_fields: tuple[str, ...] = ()

    TYPE: ClassVar[str] = "observation"
    FIELDS: ClassVar[set[str]] = {
        "type",
        "protocol",
        "identity",
        "observation_seq",
        "game_tick",
        "sent_monotonic_ns",
        "bridge_observations_coalesced",
        "sensors",
        "missing_fields",
        "clamped_fields",
    }

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RunIdentity):
            raise ProtocolError("observation.identity must be a RunIdentity")
        _strict_int(self.observation_seq, "observation.observation_seq")
        _strict_int(self.game_tick, "observation.game_tick")
        _strict_int(self.sent_monotonic_ns, "observation.sent_monotonic_ns")
        _strict_int(
            self.bridge_observations_coalesced,
            "observation.bridge_observations_coalesced",
        )
        if not isinstance(self.sensors, SensorFrame):
            raise ProtocolError("observation.sensors must be a SensorFrame")
        if isinstance(self.sensors, SensorFrameV3):
            self.sensors.events.require_observation_tick(self.game_tick)
        for list_name, values in (
            ("missing_fields", self.missing_fields),
            ("clamped_fields", self.clamped_fields),
        ):
            if not isinstance(values, tuple) or len(values) > len(self.sensors.FIELDS):
                raise ProtocolError(f"observation.{list_name} must be a bounded tuple")
            if len(set(values)) != len(values) or any(
                item not in self.sensors.FIELDS or item == "sensor_schema" for item in values
            ):
                raise ProtocolError(
                    f"observation.{list_name} contains an invalid or duplicate field"
                )

    @classmethod
    def from_dict(cls, raw: Any) -> Observation:
        value = _mapping(raw, "observation")
        _exact_keys(value, required=cls.FIELDS, name="observation")
        _protocol_header(value, cls.TYPE)
        lists: dict[str, tuple[str, ...]] = {}
        for list_name in ("missing_fields", "clamped_fields"):
            raw_list = value[list_name]
            if not isinstance(raw_list, list) or not all(
                isinstance(item, str) for item in raw_list
            ):
                raise ProtocolError(f"observation.{list_name} must be a string array")
            lists[list_name] = tuple(raw_list)
        return cls(
            identity=RunIdentity.from_dict(value["identity"]),
            observation_seq=_strict_int(value["observation_seq"], "observation.observation_seq"),
            game_tick=_strict_int(value["game_tick"], "observation.game_tick"),
            sent_monotonic_ns=_strict_int(
                value["sent_monotonic_ns"], "observation.sent_monotonic_ns"
            ),
            sensors=parse_sensor_frame(value["sensors"]),
            bridge_observations_coalesced=_strict_int(
                value["bridge_observations_coalesced"],
                "observation.bridge_observations_coalesced",
            ),
            missing_fields=lists["missing_fields"],
            clamped_fields=lists["clamped_fields"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.TYPE,
            "protocol": PROTOCOL_VERSION,
            "identity": self.identity.to_dict(),
            "observation_seq": self.observation_seq,
            "game_tick": self.game_tick,
            "sent_monotonic_ns": self.sent_monotonic_ns,
            "bridge_observations_coalesced": self.bridge_observations_coalesced,
            "sensors": self.sensors.to_dict(),
            "missing_fields": list(self.missing_fields),
            "clamped_fields": list(self.clamped_fields),
        }


@dataclass(frozen=True, slots=True)
class Action:
    identity: RunIdentity
    command_seq: int
    observation_seq: int
    observation_tick: int
    target_tick: int
    valid_through_tick: int
    neural_step_start: int
    neural_step_end: int
    state_digest: str
    motors: MotorVector
    neutral: bool = False

    TYPE: ClassVar[str] = "action"
    FIELDS: ClassVar[set[str]] = {
        "type",
        "protocol",
        "identity",
        "command_seq",
        "observation_seq",
        "observation_tick",
        "target_tick",
        "valid_through_tick",
        "neural_step_start",
        "neural_step_end",
        "state_digest",
        "motors",
        "neutral",
    }

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RunIdentity):
            raise ProtocolError("action.identity must be a RunIdentity")
        for name in (
            "command_seq",
            "observation_seq",
            "observation_tick",
            "target_tick",
            "valid_through_tick",
            "neural_step_start",
            "neural_step_end",
        ):
            _strict_int(getattr(self, name), f"action.{name}")
        if self.target_tick < self.observation_tick:
            raise ProtocolError("action.target_tick precedes its observation")
        if self.valid_through_tick < self.target_tick:
            raise ProtocolError("action.valid_through_tick precedes target_tick")
        if self.valid_through_tick > self.observation_tick + ACTION_TARGET_OFFSET_TICKS:
            raise ProtocolError("action validity exceeds the declared action budget")
        if self.neural_step_end < self.neural_step_start:
            raise ProtocolError("action neural interval cannot move backward")
        _hash(self.state_digest, "action.state_digest")
        if not isinstance(self.motors, MotorVector):
            raise ProtocolError("action.motors must be a MotorVector")
        _boolean(self.neutral, "action.neutral")
        if self.neutral and not self.motors.is_neutral:
            raise ProtocolError("neutral action must contain all-zero primitive motors")
        if not self.neutral and self.neural_step_end == self.neural_step_start:
            raise ProtocolError("non-neutral action neural interval must be non-empty")

    @classmethod
    def from_dict(cls, raw: Any) -> Action:
        value = _mapping(raw, "action")
        _exact_keys(value, required=cls.FIELDS, name="action")
        _protocol_header(value, cls.TYPE)
        return cls(
            identity=RunIdentity.from_dict(value["identity"]),
            command_seq=_strict_int(value["command_seq"], "action.command_seq"),
            observation_seq=_strict_int(value["observation_seq"], "action.observation_seq"),
            observation_tick=_strict_int(value["observation_tick"], "action.observation_tick"),
            target_tick=_strict_int(value["target_tick"], "action.target_tick"),
            valid_through_tick=_strict_int(
                value["valid_through_tick"], "action.valid_through_tick"
            ),
            neural_step_start=_strict_int(value["neural_step_start"], "action.neural_step_start"),
            neural_step_end=_strict_int(value["neural_step_end"], "action.neural_step_end"),
            state_digest=_hash(value["state_digest"], "action.state_digest"),
            motors=MotorVector.from_dict(value["motors"]),
            neutral=_boolean(value["neutral"], "action.neutral"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.TYPE,
            "protocol": PROTOCOL_VERSION,
            "identity": self.identity.to_dict(),
            "command_seq": self.command_seq,
            "observation_seq": self.observation_seq,
            "observation_tick": self.observation_tick,
            "target_tick": self.target_tick,
            "valid_through_tick": self.valid_through_tick,
            "neural_step_start": self.neural_step_start,
            "neural_step_end": self.neural_step_end,
            "state_digest": self.state_digest,
            "motors": self.motors.to_dict(),
            "neutral": self.neutral,
        }


@dataclass(frozen=True, slots=True)
class WorldFrame:
    """Authoritative physical world state sampled immediately before actuation."""

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
                "world_frame.position_m",
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
                "world_frame.velocity_m_s",
                length=3,
                minimum=-1024,
                maximum=1024,
            ),
        )
        _finite_number(self.yaw_rad, "world_frame.yaw_rad", minimum=-math.pi, maximum=math.pi)
        _finite_number(
            self.pitch_rad,
            "world_frame.pitch_rad",
            minimum=-math.pi / 2,
            maximum=math.pi / 2,
        )
        for name in ("on_ground", "horizontal_collision", "vertical_collision", "in_fluid"):
            _boolean(getattr(self, name), f"world_frame.{name}")

    @classmethod
    def from_dict(cls, raw: Any, *, name: str = "world_frame") -> WorldFrame:
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
                value["velocity_m_s"],
                f"{name}.velocity_m_s",
                length=3,
                minimum=-1024,
                maximum=1024,
            ),  # type: ignore[arg-type]
            yaw_rad=_finite_number(
                value["yaw_rad"], f"{name}.yaw_rad", minimum=-math.pi, maximum=math.pi
            ),
            pitch_rad=_finite_number(
                value["pitch_rad"],
                f"{name}.pitch_rad",
                minimum=-math.pi / 2,
                maximum=math.pi / 2,
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


@dataclass(frozen=True, slots=True)
class ActuationReceipt:
    """Authoritative Minecraft report of how one issued primitive action was handled.

    Successful application is confined to the action's two-tick window. Faulted receipts use
    ``applied_tick`` as the exact authoritative neutral-application or validation tick, which can
    lie outside that window for stale or future commands.
    """

    identity: RunIdentity
    command_seq: int
    observation_seq: int
    target_tick: int
    applied_tick: int
    world_before: WorldFrame
    requested_motors: MotorVector
    applied_motors: MotorVector
    validation: ActuationValidation
    clamp_fields: tuple[str, ...] = ()
    fault_code: FaultCode | None = None

    TYPE: ClassVar[str] = "actuation_receipt"
    FIELDS: ClassVar[set[str]] = {
        "type",
        "protocol",
        "identity",
        "command_seq",
        "observation_seq",
        "target_tick",
        "applied_tick",
        "world_before",
        "requested_motors",
        "applied_motors",
        "validation",
        "clamp_fields",
        "fault_code",
    }

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RunIdentity):
            raise ProtocolError("actuation_receipt.identity must be a RunIdentity")
        for name in ("command_seq", "observation_seq", "target_tick", "applied_tick"):
            _strict_int(getattr(self, name), f"actuation_receipt.{name}")
        if not isinstance(self.world_before, WorldFrame):
            raise ProtocolError("actuation_receipt.world_before must be a WorldFrame")
        if not isinstance(self.requested_motors, MotorVector) or not isinstance(
            self.applied_motors, MotorVector
        ):
            raise ProtocolError("actuation_receipt motor fields must be MotorVector values")
        if not isinstance(self.validation, ActuationValidation):
            raise ProtocolError("actuation_receipt.validation must be ActuationValidation")
        if self.validation in (ActuationValidation.APPLIED, ActuationValidation.CLAMPED) and (
            self.applied_tick < self.target_tick or self.applied_tick > self.target_tick + 2
        ):
            raise ProtocolError(
                "successful actuation_receipt.applied_tick must be within two target ticks"
            )
        if not isinstance(self.clamp_fields, tuple) or len(set(self.clamp_fields)) != len(
            self.clamp_fields
        ):
            raise ProtocolError("actuation_receipt.clamp_fields must be a unique tuple")
        if any(item not in MotorVector.FIELDS for item in self.clamp_fields):
            raise ProtocolError("actuation_receipt.clamp_fields contains an unknown motor axis")
        if self.fault_code is not None and not isinstance(self.fault_code, FaultCode):
            raise ProtocolError("actuation_receipt.fault_code must be FaultCode or null")
        if self.validation is ActuationValidation.APPLIED:
            if self.clamp_fields or self.fault_code is not None:
                raise ProtocolError("APPLIED receipt cannot contain clamps or a fault")
            if self.applied_motors != self.requested_motors:
                raise ProtocolError("APPLIED receipt must apply the requested motors exactly")
        elif self.validation is ActuationValidation.CLAMPED:
            if not self.clamp_fields or self.fault_code is not None:
                raise ProtocolError("CLAMPED receipt requires clamp fields and no fault")
            for axis in MotorVector.FIELDS - set(self.clamp_fields):
                if getattr(self.applied_motors, axis) != getattr(self.requested_motors, axis):
                    raise ProtocolError("CLAMPED receipt changed an undeclared motor axis")
        else:
            if not self.applied_motors.is_neutral or self.fault_code is None:
                raise ProtocolError(
                    "faulted or rejected receipt requires neutral motors and a fault"
                )

    @classmethod
    def from_dict(cls, raw: Any) -> ActuationReceipt:
        value = _mapping(raw, "actuation_receipt")
        _exact_keys(value, required=cls.FIELDS, name="actuation_receipt")
        _protocol_header(value, cls.TYPE)
        clamp_fields = value["clamp_fields"]
        if not isinstance(clamp_fields, list) or not all(
            isinstance(item, str) for item in clamp_fields
        ):
            raise ProtocolError("actuation_receipt.clamp_fields must be a string array")
        fault_code = value["fault_code"]
        return cls(
            identity=RunIdentity.from_dict(value["identity"]),
            command_seq=_strict_int(value["command_seq"], "actuation_receipt.command_seq"),
            observation_seq=_strict_int(
                value["observation_seq"], "actuation_receipt.observation_seq"
            ),
            target_tick=_strict_int(value["target_tick"], "actuation_receipt.target_tick"),
            applied_tick=_strict_int(value["applied_tick"], "actuation_receipt.applied_tick"),
            world_before=WorldFrame.from_dict(
                value["world_before"], name="actuation_receipt.world_before"
            ),
            requested_motors=MotorVector.from_dict(
                value["requested_motors"], name="actuation_receipt.requested_motors"
            ),
            applied_motors=MotorVector.from_dict(
                value["applied_motors"], name="actuation_receipt.applied_motors"
            ),
            validation=cast(
                ActuationValidation,
                _enum(value["validation"], ActuationValidation, "actuation_receipt.validation"),
            ),
            clamp_fields=tuple(clamp_fields),
            fault_code=(
                None
                if fault_code is None
                else cast(
                    FaultCode,
                    _enum(fault_code, FaultCode, "actuation_receipt.fault_code"),
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.TYPE,
            "protocol": PROTOCOL_VERSION,
            "identity": self.identity.to_dict(),
            "command_seq": self.command_seq,
            "observation_seq": self.observation_seq,
            "target_tick": self.target_tick,
            "applied_tick": self.applied_tick,
            "world_before": self.world_before.to_dict(),
            "requested_motors": self.requested_motors.to_dict(),
            "applied_motors": self.applied_motors.to_dict(),
            "validation": self.validation.value,
            "clamp_fields": list(self.clamp_fields),
            "fault_code": None if self.fault_code is None else self.fault_code.value,
        }


def _control_content(
    operation: ControlOperation,
    content_id: str | None,
    content_hash: str | None,
    *,
    name: str,
) -> None:
    if content_id is not None:
        _identifier(content_id, f"{name}.content_id")
    if content_hash is not None:
        _hash(content_hash, f"{name}.content_hash")
    no_content = {
        ControlOperation.PAUSE,
        ControlOperation.RESUME,
        ControlOperation.SINGLE_STEP,
        ControlOperation.RESET,
    }
    apply_intervention = {
        ControlOperation.LESION_APPLY,
        ControlOperation.STIMULATION_APPLY,
    }
    clear_intervention = {
        ControlOperation.LESION_CLEAR,
        ControlOperation.STIMULATION_CLEAR,
    }
    if operation in no_content:
        if content_id is not None or content_hash is not None:
            raise ProtocolError(f"{operation.value} cannot carry content identity")
    elif operation in apply_intervention:
        if content_id is None or content_hash is None:
            raise ProtocolError(f"{operation.value} requires content_id and content_hash")
    elif operation in clear_intervention:
        if content_id is None:
            raise ProtocolError(f"{operation.value} requires content_id")
    elif content_id is None:
        raise ProtocolError(f"{operation.value} requires content_id")
    if operation is ControlOperation.REPLAY_START and content_hash is None:
        raise ProtocolError("REPLAY_START requires content_hash")


@dataclass(frozen=True, slots=True)
class ControlCommand:
    identity: RunIdentity
    control_seq: int
    issued_tick: int
    operation: ControlOperation
    content_id: str | None = None
    content_hash: str | None = None

    TYPE: ClassVar[str] = "control_command"
    FIELDS: ClassVar[set[str]] = {
        "type",
        "protocol",
        "identity",
        "control_seq",
        "issued_tick",
        "operation",
        "content_id",
        "content_hash",
    }

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RunIdentity):
            raise ProtocolError("control_command.identity must be a RunIdentity")
        _strict_int(self.control_seq, "control_command.control_seq")
        _strict_int(self.issued_tick, "control_command.issued_tick")
        if not isinstance(self.operation, ControlOperation):
            raise ProtocolError("control_command.operation must be ControlOperation")
        _control_content(self.operation, self.content_id, self.content_hash, name="control_command")

    @classmethod
    def from_dict(cls, raw: Any) -> ControlCommand:
        value = _mapping(raw, "control_command")
        _exact_keys(value, required=cls.FIELDS, name="control_command")
        _protocol_header(value, cls.TYPE)
        content_id = value["content_id"]
        content_hash = value["content_hash"]
        if content_id is not None and not isinstance(content_id, str):
            raise ProtocolError("control_command.content_id must be a string or null")
        if content_hash is not None and not isinstance(content_hash, str):
            raise ProtocolError("control_command.content_hash must be a string or null")
        return cls(
            identity=RunIdentity.from_dict(value["identity"]),
            control_seq=_strict_int(value["control_seq"], "control_command.control_seq"),
            issued_tick=_strict_int(value["issued_tick"], "control_command.issued_tick"),
            operation=cast(
                ControlOperation,
                _enum(value["operation"], ControlOperation, "control_command.operation"),
            ),
            content_id=content_id,
            content_hash=content_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.TYPE,
            "protocol": PROTOCOL_VERSION,
            "identity": self.identity.to_dict(),
            "control_seq": self.control_seq,
            "issued_tick": self.issued_tick,
            "operation": self.operation.value,
            "content_id": self.content_id,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True, slots=True)
class ControlAck:
    identity: RunIdentity
    control_seq: int
    issued_tick: int
    operation: ControlOperation
    content_id: str | None
    content_hash: str | None
    accepted: bool
    state: ControlState
    effective_tick: int
    fault_code: FaultCode | None

    TYPE: ClassVar[str] = "control_ack"
    FIELDS: ClassVar[set[str]] = {
        "type",
        "protocol",
        "identity",
        "control_seq",
        "issued_tick",
        "operation",
        "content_id",
        "content_hash",
        "accepted",
        "state",
        "effective_tick",
        "fault_code",
    }

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RunIdentity):
            raise ProtocolError("control_ack.identity must be a RunIdentity")
        _strict_int(self.control_seq, "control_ack.control_seq")
        _strict_int(self.issued_tick, "control_ack.issued_tick")
        _strict_int(self.effective_tick, "control_ack.effective_tick")
        if not isinstance(self.operation, ControlOperation):
            raise ProtocolError("control_ack.operation must be ControlOperation")
        _control_content(self.operation, self.content_id, self.content_hash, name="control_ack")
        _boolean(self.accepted, "control_ack.accepted")
        tick_delta = self.effective_tick - self.issued_tick
        if self.accepted and self.operation is ControlOperation.RESET:
            # An accepted RESET is stamped with the governed action's target_tick, so its
            # delay is the action budget plus the observation the effect was materialized
            # against: issued + ACTION_TARGET_OFFSET_TICKS or one more. The frame rule stays
            # permissive from 1 so acknowledgements recorded under the one-tick budget still
            # parse; the exact pair is enforced where the acknowledgement is produced
            # (NeuralService.process_latest) and where it is consumed (StagedReset).
            if tick_delta < 1 or tick_delta > ACTION_TARGET_OFFSET_TICKS + 1:
                raise ProtocolError(
                    "accepted RESET control_ack.effective_tick is outside the reset tick window"
                )
        elif tick_delta < 0 or tick_delta > 1:
            raise ProtocolError("control_ack.effective_tick must be issued_tick or the next tick")
        if not isinstance(self.state, ControlState):
            raise ProtocolError("control_ack.state must be ControlState")
        if self.fault_code is not None and not isinstance(self.fault_code, FaultCode):
            raise ProtocolError("control_ack.fault_code must be FaultCode or null")
        if self.accepted and self.fault_code is not None:
            raise ProtocolError("accepted control ack cannot carry a fault")
        if not self.accepted and self.fault_code is None:
            raise ProtocolError("rejected control ack requires a fault")

    @classmethod
    def from_dict(cls, raw: Any) -> ControlAck:
        value = _mapping(raw, "control_ack")
        _exact_keys(value, required=cls.FIELDS, name="control_ack")
        _protocol_header(value, cls.TYPE)
        content_id = value["content_id"]
        content_hash = value["content_hash"]
        fault_code = value["fault_code"]
        if content_id is not None and not isinstance(content_id, str):
            raise ProtocolError("control_ack.content_id must be a string or null")
        if content_hash is not None and not isinstance(content_hash, str):
            raise ProtocolError("control_ack.content_hash must be a string or null")
        return cls(
            identity=RunIdentity.from_dict(value["identity"]),
            control_seq=_strict_int(value["control_seq"], "control_ack.control_seq"),
            issued_tick=_strict_int(value["issued_tick"], "control_ack.issued_tick"),
            operation=cast(
                ControlOperation,
                _enum(value["operation"], ControlOperation, "control_ack.operation"),
            ),
            content_id=content_id,
            content_hash=content_hash,
            accepted=_boolean(value["accepted"], "control_ack.accepted"),
            state=cast(ControlState, _enum(value["state"], ControlState, "control_ack.state")),
            effective_tick=_strict_int(value["effective_tick"], "control_ack.effective_tick"),
            fault_code=(
                None
                if fault_code is None
                else cast(FaultCode, _enum(fault_code, FaultCode, "control_ack.fault_code"))
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.TYPE,
            "protocol": PROTOCOL_VERSION,
            "identity": self.identity.to_dict(),
            "control_seq": self.control_seq,
            "issued_tick": self.issued_tick,
            "operation": self.operation.value,
            "content_id": self.content_id,
            "content_hash": self.content_hash,
            "accepted": self.accepted,
            "state": self.state.value,
            "effective_tick": self.effective_tick,
            "fault_code": None if self.fault_code is None else self.fault_code.value,
        }


@dataclass(frozen=True, slots=True)
class Fault:
    identity: RunIdentity
    code: FaultCode
    message: str
    game_tick: int
    related_observation_seq: int | None = None
    neutral_applied: bool = True

    TYPE: ClassVar[str] = "fault"
    FIELDS: ClassVar[set[str]] = {
        "type",
        "protocol",
        "identity",
        "code",
        "message",
        "game_tick",
        "related_observation_seq",
        "neutral_applied",
    }

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RunIdentity):
            raise ProtocolError("fault.identity must be a RunIdentity")
        if not isinstance(self.code, FaultCode):
            raise ProtocolError("fault.code must be a FaultCode")
        _text(self.message, "fault.message", maximum=MAX_FAULT_MESSAGE_CHARS)
        _strict_int(self.game_tick, "fault.game_tick")
        if self.related_observation_seq is not None:
            _strict_int(self.related_observation_seq, "fault.related_observation_seq")
        if self.neutral_applied is not True:
            raise ProtocolError("fault frames must declare neutral_applied=true")

    @classmethod
    def from_dict(cls, raw: Any) -> Fault:
        value = _mapping(raw, "fault")
        _exact_keys(value, required=cls.FIELDS, name="fault")
        _protocol_header(value, cls.TYPE)
        related = value["related_observation_seq"]
        return cls(
            identity=RunIdentity.from_dict(value["identity"]),
            code=_enum(value["code"], FaultCode, "fault.code"),  # type: ignore[arg-type]
            message=_text(value["message"], "fault.message", maximum=MAX_FAULT_MESSAGE_CHARS),
            game_tick=_strict_int(value["game_tick"], "fault.game_tick"),
            related_observation_seq=(
                None if related is None else _strict_int(related, "fault.related_observation_seq")
            ),
            neutral_applied=_boolean(value["neutral_applied"], "fault.neutral_applied"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.TYPE,
            "protocol": PROTOCOL_VERSION,
            "identity": self.identity.to_dict(),
            "code": self.code.value,
            "message": self.message,
            "game_tick": self.game_tick,
            "related_observation_seq": self.related_observation_seq,
            "neutral_applied": self.neutral_applied,
        }
