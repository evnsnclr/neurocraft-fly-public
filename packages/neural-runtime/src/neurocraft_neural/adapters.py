"""Sparse, named, low-capacity sensory and motor boundaries."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from neurocraft_connectome import RuntimeGraph
from neurocraft_connectome.model import sha256_json

from neurocraft_neural.model import (
    NeuralConfigurationError,
    NeuralIntegrityError,
    NeuralState,
    _finite_float,
    _float_hex,
    _float_tuple_digest,
)

MAX_SENSOR_CHANNELS = 32
MAX_SENSOR_TARGETS_TOTAL = 64
MAX_MOTOR_CHANNELS = 16
MAX_MOTOR_MEMBERS_TOTAL = 64
MAX_ABSOLUTE_ADAPTER_WEIGHT = 8.0
MAX_ABSOLUTE_ADAPTER_PARAMETER = 8.0


@dataclass(frozen=True, slots=True)
class SensorTarget:
    body_id: int
    weight: float


@dataclass(frozen=True, slots=True)
class SensorChannel:
    """One declared physical scalar mapped only to sensory boundary neurons."""

    name: str
    unit: str
    targets: tuple[SensorTarget, ...]
    gain: float = 1.0
    offset: float = 0.0
    clip_min: float = -1.0
    clip_max: float = 1.0
    saturation: str = "none"


@dataclass(frozen=True, slots=True)
class EncodedStimulation:
    """Sparse-adapter result accepted by the neural engine."""

    graph_identity_sha256: str
    adapter_sha256: str
    vector: tuple[float, ...]
    assignments: tuple[tuple[str, int, float], ...]
    identity_sha256: str

    def assert_integrity(self) -> None:
        for value in self.vector:
            if not math.isfinite(value):
                raise NeuralIntegrityError("encoded sensory vector contains non-finite value")
        material = {
            "adapter_sha256": self.adapter_sha256,
            "assignments": [
                [name, index, _float_hex(value)] for name, index, value in self.assignments
            ],
            "graph_identity_sha256": self.graph_identity_sha256,
            "vector_sha256": _float_tuple_digest(self.vector),
        }
        if sha256_json(material) != self.identity_sha256:
            raise NeuralIntegrityError("encoded stimulation identity mismatch")


class SensoryAdapter:
    """Compiled observation boundary with no motor or recurrent reference."""

    def __init__(self, graph: RuntimeGraph, channels: Sequence[SensorChannel]) -> None:
        graph.assert_integrity()
        channel_values = tuple(channels)
        if not 1 <= len(channel_values) <= MAX_SENSOR_CHANNELS:
            raise NeuralConfigurationError(
                f"sensory adapter requires 1..{MAX_SENSOR_CHANNELS} channels"
            )
        names = [channel.name for channel in channel_values]
        if any(not isinstance(name, str) or not name for name in names):
            raise NeuralConfigurationError("sensor channels require non-empty names")
        if len(set(names)) != len(names):
            raise NeuralConfigurationError("sensor channel names must be unique")
        if sum(len(channel.targets) for channel in channel_values) > MAX_SENSOR_TARGETS_TOTAL:
            raise NeuralConfigurationError("sensory adapter exceeds sparse target budget")

        compiled: list[tuple[SensorChannel, tuple[tuple[int, float], ...]]] = []
        identity_channels: list[dict[str, object]] = []
        for channel in channel_values:
            if not isinstance(channel.unit, str) or not channel.unit:
                raise NeuralConfigurationError(f"sensor {channel.name!r} requires a physical unit")
            if not channel.targets:
                raise NeuralConfigurationError(
                    f"sensor {channel.name!r} has an empty boundary population"
                )
            gain = _finite_float(channel.gain, f"sensor {channel.name}.gain")
            offset = _finite_float(channel.offset, f"sensor {channel.name}.offset")
            clip_min = _finite_float(channel.clip_min, f"sensor {channel.name}.clip_min")
            clip_max = _finite_float(channel.clip_max, f"sensor {channel.name}.clip_max")
            if clip_min >= clip_max:
                raise NeuralConfigurationError(
                    f"sensor {channel.name!r} clip_min must be below clip_max"
                )
            if any(
                abs(value) > MAX_ABSOLUTE_ADAPTER_PARAMETER
                for value in (gain, offset, clip_min, clip_max)
            ):
                raise NeuralConfigurationError(
                    f"sensor {channel.name!r} exceeds declared parameter magnitude budget"
                )
            if channel.saturation not in {"none", "tanh"}:
                raise NeuralConfigurationError(
                    f"sensor {channel.name!r} saturation must be 'none' or 'tanh'"
                )
            target_indices: set[int] = set()
            targets: list[tuple[int, float]] = []
            for target in channel.targets:
                try:
                    index = graph.index_of(target.body_id)
                except KeyError as exc:
                    raise NeuralConfigurationError(str(exc)) from exc
                node = graph.nodes[index]
                if "sensory" not in node.roles:
                    raise NeuralConfigurationError(
                        f"sensor {channel.name!r} target {target.body_id} "
                        "is not a named sensory node"
                    )
                if index in target_indices:
                    raise NeuralConfigurationError(
                        f"sensor {channel.name!r} repeats target {target.body_id}"
                    )
                target_indices.add(index)
                weight = _finite_float(
                    target.weight, f"sensor {channel.name}.target[{target.body_id}]"
                )
                if weight == 0 or abs(weight) > MAX_ABSOLUTE_ADAPTER_WEIGHT:
                    raise NeuralConfigurationError(
                        "sensor target weight must be nonzero and at most "
                        f"{MAX_ABSOLUTE_ADAPTER_WEIGHT} in magnitude"
                    )
                targets.append((index, weight))
            targets.sort()
            normalized_channel = SensorChannel(
                name=channel.name,
                unit=channel.unit,
                targets=channel.targets,
                gain=gain,
                offset=offset,
                clip_min=clip_min,
                clip_max=clip_max,
                saturation=channel.saturation,
            )
            compiled.append((normalized_channel, tuple(targets)))
            identity_channels.append(
                {
                    "clip_max": _float_hex(clip_max),
                    "clip_min": _float_hex(clip_min),
                    "gain": _float_hex(gain),
                    "name": channel.name,
                    "offset": _float_hex(offset),
                    "saturation": channel.saturation,
                    "targets": [
                        [graph.nodes[index].body_id, _float_hex(weight)]
                        for index, weight in targets
                    ],
                    "unit": channel.unit,
                }
            )
        self._graph_identity = graph.manifest.runtime_identity_sha256
        self._node_count = len(graph.nodes)
        self._channels = tuple(compiled)
        self._expected_names = frozenset(names)
        self._identity = sha256_json(
            {
                "channels": identity_channels,
                "graph_identity_sha256": self._graph_identity,
                "schema": "neurocraft-sensory-adapter-v1",
            }
        )

    @property
    def identity_sha256(self) -> str:
        return self._identity

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(channel.name for channel, _ in self._channels)

    def encode(self, observations: Mapping[str, float]) -> EncodedStimulation:
        if not isinstance(observations, Mapping):
            raise NeuralConfigurationError("observations must be a numeric mapping")
        if any(not isinstance(name, str) for name in observations):
            raise NeuralConfigurationError("observation field names must be strings")
        if set(observations) != self._expected_names:
            missing = sorted(self._expected_names - set(observations))
            unknown = sorted(set(observations) - self._expected_names)
            raise NeuralConfigurationError(
                f"observation schema mismatch; missing={missing}, unknown={unknown}"
            )
        vector = [0.0] * self._node_count
        assignments: list[tuple[str, int, float]] = []
        for channel, targets in self._channels:
            raw = _finite_float(observations[channel.name], f"observation[{channel.name}]")
            value = channel.gain * raw + channel.offset
            value = min(channel.clip_max, max(channel.clip_min, value))
            if channel.saturation == "tanh":
                value = math.tanh(value)
            for index, weight in targets:
                contribution = value * weight
                vector[index] += contribution
                assignments.append((channel.name, index, contribution))
        if any(not math.isfinite(value) for value in vector):
            raise NeuralConfigurationError("encoded sensory vector is non-finite")
        assignment_tuple = tuple(assignments)
        material = {
            "adapter_sha256": self._identity,
            "assignments": [
                [name, index, _float_hex(value)] for name, index, value in assignment_tuple
            ],
            "graph_identity_sha256": self._graph_identity,
            "vector_sha256": _float_tuple_digest(vector),
        }
        encoded = EncodedStimulation(
            graph_identity_sha256=self._graph_identity,
            adapter_sha256=self._identity,
            vector=tuple(vector),
            assignments=assignment_tuple,
            identity_sha256=sha256_json(material),
        )
        encoded.assert_integrity()
        return encoded


@dataclass(frozen=True, slots=True)
class PoolMember:
    body_id: int
    weight: float


@dataclass(frozen=True, slots=True)
class MotorPool:
    name: str
    members: tuple[PoolMember, ...]
    gain: float = 1.0
    dead_zone: float = 0.0


@dataclass(frozen=True, slots=True)
class MotorFrame:
    graph_identity_sha256: str
    decoder_sha256: str
    neural_state_sha256: str
    values: tuple[tuple[str, float], ...]
    identity_sha256: str


class MotorDecoder:
    """Stateless readout accepting only a verified neural state."""

    def __init__(self, graph: RuntimeGraph, pools: Sequence[MotorPool]) -> None:
        graph.assert_integrity()
        pool_values = tuple(pools)
        if not 1 <= len(pool_values) <= MAX_MOTOR_CHANNELS:
            raise NeuralConfigurationError(f"motor decoder requires 1..{MAX_MOTOR_CHANNELS} pools")
        if sum(len(pool.members) for pool in pool_values) > MAX_MOTOR_MEMBERS_TOTAL:
            raise NeuralConfigurationError("motor decoder exceeds sparse member budget")
        names = [pool.name for pool in pool_values]
        if any(not isinstance(name, str) or not name for name in names):
            raise NeuralConfigurationError("motor pools require non-empty names")
        if len(set(names)) != len(names):
            raise NeuralConfigurationError("motor pool names must be unique")

        compiled: list[tuple[str, tuple[tuple[int, float], ...], float, float, float]] = []
        identity_pools: list[dict[str, object]] = []
        for pool in pool_values:
            if not pool.members:
                raise NeuralConfigurationError(f"motor pool {pool.name!r} is empty")
            gain = _finite_float(pool.gain, f"motor pool {pool.name}.gain")
            dead_zone = _finite_float(pool.dead_zone, f"motor pool {pool.name}.dead_zone")
            if gain < 0 or gain > MAX_ABSOLUTE_ADAPTER_PARAMETER or not 0 <= dead_zone < 1:
                raise NeuralConfigurationError(
                    "motor gain must be within the declared budget and dead_zone in [0, 1)"
                )
            members: list[tuple[int, float]] = []
            seen: set[int] = set()
            for member in pool.members:
                try:
                    index = graph.index_of(member.body_id)
                except KeyError as exc:
                    raise NeuralConfigurationError(str(exc)) from exc
                node = graph.nodes[index]
                is_named_output = "motor" in node.roles or node.superclass == "descending_neuron"
                if not is_named_output:
                    raise NeuralConfigurationError(
                        f"motor pool {pool.name!r} member {member.body_id} is not motor/descending"
                    )
                if index in seen:
                    raise NeuralConfigurationError(
                        f"motor pool {pool.name!r} repeats member {member.body_id}"
                    )
                seen.add(index)
                weight = _finite_float(
                    member.weight, f"motor pool {pool.name}.member[{member.body_id}]"
                )
                if weight == 0 or abs(weight) > MAX_ABSOLUTE_ADAPTER_WEIGHT:
                    raise NeuralConfigurationError(
                        "motor member weight must be nonzero and at most "
                        f"{MAX_ABSOLUTE_ADAPTER_WEIGHT} in magnitude"
                    )
                members.append((index, weight))
            members.sort()
            denominator = math.fsum(abs(weight) for _, weight in members)
            if denominator == 0:
                raise NeuralConfigurationError(f"motor pool {pool.name!r} has zero denominator")
            compiled.append((pool.name, tuple(members), denominator, gain, dead_zone))
            identity_pools.append(
                {
                    "dead_zone": _float_hex(dead_zone),
                    "gain": _float_hex(gain),
                    "members": [
                        [graph.nodes[index].body_id, _float_hex(weight)]
                        for index, weight in members
                    ],
                    "name": pool.name,
                }
            )
        self._graph_identity = graph.manifest.runtime_identity_sha256
        self._node_count = len(graph.nodes)
        self._pools = tuple(compiled)
        self._identity = sha256_json(
            {
                "graph_identity_sha256": self._graph_identity,
                "pools": identity_pools,
                "schema": "neurocraft-motor-decoder-v1",
            }
        )

    @property
    def identity_sha256(self) -> str:
        return self._identity

    def decode(self, state: NeuralState) -> MotorFrame:
        if not isinstance(state, NeuralState):
            raise NeuralConfigurationError("motor decoder accepts only NeuralState")
        state.assert_integrity()
        if state.graph_identity_sha256 != self._graph_identity:
            raise NeuralIntegrityError("motor decoder state is from a different graph")
        if len(state.activity) != self._node_count:
            raise NeuralIntegrityError("motor decoder state length mismatch")
        values: list[tuple[str, float]] = []
        for name, members, denominator, gain, dead_zone in self._pools:
            raw = (
                gain
                * math.fsum(weight * state.activity[index] for index, weight in members)
                / denominator
            )
            if not math.isfinite(raw):
                raise NeuralIntegrityError("motor decoder produced a non-finite raw value")
            value = min(1.0, max(-1.0, raw))
            if abs(value) < dead_zone:
                value = 0.0
            values.append((name, value))
        value_tuple = tuple(values)
        identity = sha256_json(
            {
                "decoder_sha256": self._identity,
                "graph_identity_sha256": self._graph_identity,
                "neural_state_sha256": state.state_sha256,
                "values": [[name, _float_hex(value)] for name, value in value_tuple],
            }
        )
        return MotorFrame(
            graph_identity_sha256=self._graph_identity,
            decoder_sha256=self._identity,
            neural_state_sha256=state.state_sha256,
            values=value_tuple,
            identity_sha256=identity,
        )
