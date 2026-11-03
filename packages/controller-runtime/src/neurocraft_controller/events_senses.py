"""Candidate engineered event pulses over unchanged loom-v2 background inputs.

Food proximity/contact are engineered Java predicates, not gustatory inferences.
Brush and player attack attempts share grooming afferents as an explicit proxy;
no damage, nociception, hunger, consumption, or biological behavior is modeled.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from neurocraft_protocol import EventSample, SensorFrame, SensorFrameV3
from neurocraft_protocol.sensor_contract import OBSERVATION_SCHEMA_V3_SHA256
from numpy.typing import NDArray

from .config import canonical_sha256
from .embodied_senses import _string_column, _verified_graph_array
from .features import EncodedFeatures, EntityRayFeatureEncoder

if TYPE_CHECKING:
    from neurocraft_connectome.full_runtime import FullRuntimeGraph

ENCODER_SCHEMA = "neurocraft.events.encoder.v1"
BINDING_SCHEMA = "neurocraft.events.populations.v1"
MAX_EVENT_AGE_TICKS = 8
PULSE_AMPLITUDE = 1.0
# Exact retained annotation selectors. Sensory side is rootSide, not somaSide.
SPECS = (
    ("food", (("class", "gustatory"), ("subclass", "labellar bristle")), "rootSide", (82, 81)),
    ("groom", (("class", "mechanosensory"), ("subclass", "grooming")), "rootSide", (40, 25)),
    ("mn9", (("type", "MN9"),), "somaSide", (1, 1)),
    ("antenna", (("superclass", "cb_motor"), ("subclass", "am")), "somaSide", (6, 7)),
)


class EventSensesError(ValueError):
    """An event boundary cannot be bound or executed as declared."""


def _hash(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise EventSensesError("expected a lowercase SHA-256")


def _integer(value: int, *, minimum: int = 0, maximum: int = 2**63 - 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise EventSensesError("expected a bounded exact integer")


@dataclass(frozen=True, slots=True)
class EventPopulation:
    name: str
    # body ID, dense index, declared source side; sorted by body ID.
    members: tuple[tuple[int, int, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "members", tuple(tuple(row) for row in self.members))

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(row[1] for row in self.members)


@dataclass(frozen=True, slots=True)
class EventPopulationBinding:
    graph_hash: str
    annotations_hash: str
    node_count: int
    populations: tuple[EventPopulation, ...]

    def __post_init__(self) -> None:
        _hash(self.graph_hash)
        _hash(self.annotations_hash)
        _integer(self.node_count, minimum=1, maximum=166_700)
        object.__setattr__(self, "populations", tuple(self.populations))
        if tuple(p.name for p in self.populations) != tuple(spec[0] for spec in SPECS):
            raise EventSensesError("exact food/groom/MN9/antenna populations required in order")
        bodies: set[int] = set()
        indices: set[int] = set()
        for pop, (_, _, _, counts) in zip(self.populations, SPECS, strict=True):
            if tuple(sorted(pop.members)) != pop.members:
                raise EventSensesError("event members must ascend by source body ID")
            if tuple(sum(row[2] == side for row in pop.members) for side in "LR") != counts:
                raise EventSensesError(f"{pop.name} source counts or sides drifted")
            if len(pop.members) != sum(counts):
                raise EventSensesError("event population has unknown sides")
            for body, index, _ in pop.members:
                _integer(body, minimum=1)
                _integer(index, maximum=self.node_count - 1)
                if body in bodies or index in indices:
                    raise EventSensesError("event populations must be internally disjoint")
                bodies.add(body)
                indices.add(index)

    def population(self, name: str) -> EventPopulation:
        return next(pop for pop in self.populations if pop.name == name)

    def to_identity_material(self) -> dict[str, Any]:
        return {
            "schema": BINDING_SCHEMA,
            "graph_hash": self.graph_hash,
            "annotations_hash": self.annotations_hash,
            "node_count": self.node_count,
            "populations": [
                {
                    "name": pop.name,
                    "selector": dict(terms),
                    "side_column": side,
                    "expected_lr": list(counts),
                    "members_body_dense_side": list(pop.members),
                }
                for pop, (_, terms, side, counts) in zip(self.populations, SPECS, strict=True)
            ],
            "readout_quantity": "unweighted mean of signed dimensionless model activity",
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.to_identity_material())


def select_event_populations(
    *,
    columns: Mapping[str, Sequence[str | None]],
    body_ids: Sequence[int],
    index_of: Callable[[int], int],
    graph_hash: str,
    annotations_hash: str,
    node_count: int,
) -> EventPopulationBinding:
    required = {key for _, terms, side, _ in SPECS for key in (*dict(terms), side)}
    if len(body_ids) != node_count or len(set(body_ids)) != node_count:
        raise EventSensesError("annotation body IDs must cover the unique dense node space")
    if any(key not in columns or len(columns[key]) != node_count for key in required):
        raise EventSensesError("required annotation columns must cover the retained dense space")
    selected = []
    for name, terms, side, _ in SPECS:
        members = []
        for index, body in enumerate(body_ids):
            if all(columns[key][index] == value for key, value in terms):
                if index_of(body) != index:
                    raise EventSensesError(
                        "event annotation join differs from artifact dense order"
                    )
                members.append((body, index, str(columns[side][index])))
        selected.append(EventPopulation(name, tuple(sorted(members))))
    return EventPopulationBinding(graph_hash, annotations_hash, node_count, tuple(selected))


def load_event_populations(
    graph: FullRuntimeGraph,
    body_annotations_path: Path,
) -> EventPopulationBinding:
    """Verify source bytes and complete retained body-ID join before selecting labels."""
    # The old embodied loader intentionally projects only its five columns. Read
    # this boundary's complete columns from the SAME bytes whose digest is checked.
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.feather as feather  # type: ignore[import-untyped]

    declared = next(
        (source for source in graph.manifest.source_artifacts if source.source_id == "annotations"),
        None,
    )
    if declared is None:
        raise EventSensesError("artifact declares no annotations source")
    raw = body_annotations_path.read_bytes()
    annotations_hash = sha256(raw).hexdigest()
    if len(raw) != declared.byte_length or annotations_hash != declared.sha256:
        raise EventSensesError("event annotation source size or digest mismatch")
    keys = {key for _, terms, side, _ in SPECS for key in (*dict(terms), side)}
    table = feather.read_table(pa.BufferReader(raw), columns=sorted({"bodyId", *keys}))
    if table.num_rows != graph.manifest.counts["annotation_source_rows"]:
        raise EventSensesError("event annotation source row count mismatch")
    # Do not retain exported mmap pointers in an exception traceback: graph.close
    # must succeed even if a source membership or join check below fails.
    rows = _verified_graph_array(graph, "annotation_source_rows", "<u8").copy()
    bodies = _verified_graph_array(graph, "body_ids", "<u8").copy()
    if rows.size == 0 or bool((rows >= table.num_rows).any()) or np.unique(rows).size != rows.size:
        raise EventSensesError("annotation source rows are missing, repeated, or out of range")
    taken = table.take(rows.astype(np.int64))
    if not np.array_equal(np.asarray(taken.column("bodyId").to_pylist(), dtype=np.uint64), bodies):
        raise EventSensesError("annotation body-ID join differs from artifact dense order")
    keys = {key for _, terms, side, _ in SPECS for key in (*dict(terms), side)}
    return select_event_populations(
        columns={key: _string_column(taken, key) for key in keys},
        body_ids=bodies.tolist(),
        index_of=graph.index_of,
        graph_hash=graph.manifest.manifest_file_sha256,
        annotations_hash=annotations_hash,
        node_count=graph.manifest.node_count,
    )


class EventFeatureEncoder(EntityRayFeatureEncoder):
    """Frozen four physical feature values under the new event transport contract."""

    def __init__(self) -> None:
        super().__init__()
        self._identity = OBSERVATION_SCHEMA_V3_SHA256

    def encode(self, sensors: SensorFrame) -> EncodedFeatures:
        if not isinstance(sensors, SensorFrameV3):
            raise EventSensesError("event profile requires SensorFrameV3")
        return super().encode(sensors)


@dataclass(frozen=True, slots=True)
class EventPulse:
    food: bool
    groom: bool
    food_age: int | None
    groom_age: int | None
    food_delta: int
    brush_delta: int
    attack_delta: int
    baseline_absorbed: bool

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class EventEncoderBinding:
    populations: EventPopulationBinding
    background_encoder_sha256: str
    component_decoder_sha256: str
    component_calibration_sha256: str

    def __post_init__(self) -> None:
        _hash(self.background_encoder_sha256)
        _hash(self.component_decoder_sha256)
        if self.component_calibration_sha256:
            _hash(self.component_calibration_sha256)

    def to_identity_material(self) -> dict[str, Any]:
        return {
            "schema": ENCODER_SCHEMA,
            "status": "candidate engineered encoder; not behavior-validated",
            "observation_schema_sha256": OBSERVATION_SCHEMA_V3_SHA256,
            "background_encoder_sha256": self.background_encoder_sha256,
            "component_decoder_sha256": self.component_decoder_sha256,
            "component_calibration_sha256": self.component_calibration_sha256,
            "population_binding_sha256": self.populations.identity_sha256,
            "pulse_amplitude_per_member": PULSE_AMPLITUDE.hex(),
            "pulse_neural_steps": 5,
            "max_event_age_game_ticks": MAX_EVENT_AGE_TICKS,
            "formula": "add amplitude to each selected member after loom-v2; no clipping",
            "food": "one pulse on new fresh food counter; body food-near rising edge",
            "groom": "one pulse on fresh brush OR player attack-attempt; no summing duplicates",
            "proxy": "brush and attack-attempt drive grooming afferents; not nociception",
            "counters": (
                "first observation absorbed; monotone; jumps coalesced once; stale consumed"
            ),
            "no_input": "zero the FINAL composite after every old and event assignment",
            "calibration_scope": (
                "existing calibration describes background/decoder component only; "
                "it does not validate the composite event encoder"
            ),
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.to_identity_material())


class EventInputState:
    def __init__(self, binding: EventEncoderBinding) -> None:
        self.binding = binding
        self._previous: EventSample | None = None

    def reset(self) -> None:
        self._previous = None

    def prepare(self, events: EventSample, game_tick: int) -> EventPulse:
        """Validate all counters before mutation; consume each changed count at most once."""
        _integer(game_tick)
        previous = self._previous
        deltas: list[int] = []
        ages: list[int | None] = []
        for name in ("food", "brush", "attack_attempt"):
            counter = getattr(events, name)
            _integer(counter.count)
            if (counter.count == 0) != (counter.last_game_tick is None):
                raise EventSensesError("zero event count iff timestamp absent")
            if counter.last_game_tick is not None:
                _integer(counter.last_game_tick, maximum=game_tick)
            if previous is None:
                delta = 0
            else:
                old = getattr(previous, name)
                if counter.count < old.count:
                    raise EventSensesError("event count moved backwards")
                if counter.count == old.count and counter.last_game_tick != old.last_game_tick:
                    raise EventSensesError("event timestamp changed without a new count")
                if (
                    old.last_game_tick is not None
                    and counter.last_game_tick is not None
                    and counter.last_game_tick < old.last_game_tick
                ):
                    raise EventSensesError("event timestamp moved backwards")
                delta = counter.count - old.count
            age = None if counter.last_game_tick is None else game_tick - counter.last_game_tick
            deltas.append(delta)
            ages.append(age)
        fresh = [
            delta > 0 and age is not None and age <= MAX_EVENT_AGE_TICKS
            for delta, age in zip(deltas, ages, strict=True)
        ]
        groom_ages = [
            age
            for eligible, age in zip(fresh[1:], ages[1:], strict=True)
            if eligible and age is not None
        ]
        result = EventPulse(
            fresh[0],
            any(fresh[1:]),
            ages[0] if fresh[0] else None,
            min(groom_ages) if groom_ages else None,
            deltas[0],
            deltas[1],
            deltas[2],
            baseline_absorbed=previous is None,
        )
        self._previous = events
        return result

    def apply(self, vector: NDArray[np.float64], pulse: EventPulse) -> None:
        for name, active in (("food", pulse.food), ("groom", pulse.groom)):
            if active:
                vector[list(self.binding.populations.population(name).indices)] += PULSE_AMPLITUDE


@dataclass(frozen=True, slots=True)
class CompletedEventReadout:
    """Only scalar copies from the actual immutable completed compute boundary."""

    pulse: EventPulse
    mn9_raw: float
    antenna_raw: float
    mn9_left: float
    mn9_right: float
    antenna_left: float
    antenna_right: float


def completed_event_readout(
    activity: NDArray[np.float64],
    binding: EventPopulationBinding,
    pulse: EventPulse,
) -> CompletedEventReadout:
    means = []
    for name in ("mn9", "antenna"):
        members = binding.population(name).members
        means.append(math.fsum(float(activity[index]) for _, index, _ in members) / len(members))
    sides = []
    for name in ("mn9", "antenna"):
        for side in "LR":
            indices = [
                index
                for _, index, source_side in binding.population(name).members
                if source_side == side
            ]
            sides.append(math.fsum(float(activity[index]) for index in indices) / len(indices))
    if any(not math.isfinite(value) or not -1 <= value <= 1 for value in (*means, *sides)):
        raise EventSensesError("completed event readout must contain finite model activity")
    return CompletedEventReadout(pulse, *means, *sides)
