"""Optional model-activity display, independent of actuation and transport queues.

DNp01 is display-only interior recurrent context, never another motor readout.
Values are means of dimensionless model activity, not spike rates. Capture must
run immediately after a completed controller step under the service execution
fence; only immutable six-population snapshots reach the background file writer.
"""

from __future__ import annotations

import math
import os
import re
import tempfile
import threading
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from neurocraft_protocol import Action, ProgramAction, RunIdentity, canonical_json_bytes

from .config import canonical_sha256
from .embodied_senses import (
    EXPECTED_LC4_LEFT_NODES,
    EXPECTED_LC4_RIGHT_NODES,
    SIDE_COLUMN,
    TYPE_COLUMN,
    EmbodiedPopulationBinding,
    _string_column,
    _verified_annotation_table,
    _verified_graph_array,
    faithful_entry_body_ids,
)
from .escape_reflex import TtmnPoolBinding, select_ttmn_pool
from .flight_pools import LEFT_WING_MOTOR_BODY_IDS, RIGHT_WING_MOTOR_BODY_IDS

if TYPE_CHECKING:
    from neurocraft_connectome.full_runtime import FullRuntimeGraph

    from .full_brain_controller import FullBrainController
    from .program_policy import ProgramSelectingController

SCHEMA = "neurocraft.brain_display.v1"
BINDING_SCHEMA = "neurocraft.brain_display_binding.v1"
MAX_FRAME_BYTES = 16 * 1024
POPULATION_NAMES = ("lc4_left", "lc4_right", "dnp01_left", "dnp01_right", "ttmn_left", "ttmn_right")
_COUNTS = (EXPECTED_LC4_LEFT_NODES, EXPECTED_LC4_RIGHT_NODES, 1, 1, 1, 1)
_HASH = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


class BrainDisplayError(ValueError):
    """Display data is unavailable or inconsistent; never a controller fault."""


def _integer(value: int, name: str, *, minimum: int = 0, maximum: int = 2**63 - 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise BrainDisplayError(f"{name} must be an integer in [{minimum}, {maximum}]")


def _hash(value: str, name: str) -> None:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise BrainDisplayError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class DisplayPopulation:
    name: str
    body_ids: tuple[int, ...]
    dense_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "body_ids", tuple(self.body_ids))
        object.__setattr__(self, "dense_indices", tuple(self.dense_indices))


@dataclass(frozen=True, slots=True)
class BrainDisplayBinding:
    graph_hash: str
    annotations_hash: str
    sensory_binding_hash: str
    ttmn_binding_hash: str
    node_count: int
    populations: tuple[DisplayPopulation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "populations", tuple(self.populations))
        for name in ("graph_hash", "annotations_hash", "sensory_binding_hash", "ttmn_binding_hash"):
            _hash(getattr(self, name), name)
        _integer(self.node_count, "node_count", minimum=1, maximum=166_700)
        if tuple(pop.name for pop in self.populations) != POPULATION_NAMES:
            raise BrainDisplayError(
                "display binding requires the six populations in declared order"
            )
        all_bodies: set[int] = set()
        all_indices: set[int] = set()
        for pop, count in zip(self.populations, _COUNTS, strict=True):
            if len(pop.body_ids) != count or len(pop.dense_indices) != count:
                raise BrainDisplayError(f"{pop.name} count drifted from {count}")
            if tuple(sorted(set(pop.body_ids))) != pop.body_ids:
                raise BrainDisplayError(f"{pop.name} body IDs must be unique and sorted")
            for body, index in zip(pop.body_ids, pop.dense_indices, strict=True):
                _integer(body, "body_id", minimum=1)
                _integer(index, "dense_index", maximum=self.node_count - 1)
                if body in all_bodies or index in all_indices:
                    raise BrainDisplayError(
                        "display populations overlap in body IDs or dense indices"
                    )
                all_bodies.add(body)
                all_indices.add(index)

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(
            {
                "schema": BINDING_SCHEMA,
                "graph_hash": self.graph_hash,
                "annotations_hash": self.annotations_hash,
                "sensory_binding_hash": self.sensory_binding_hash,
                "ttmn_binding_hash": self.ttmn_binding_hash,
                "node_count": self.node_count,
                "populations": [
                    {
                        "name": pop.name,
                        "members": list(zip(pop.body_ids, pop.dense_indices, strict=True)),
                    }
                    for pop in self.populations
                ],
                "quantity": "mean dimensionless model activity; not spike rates",
                "dnp01_role": "display only; not a readout",
            }
        )


def select_brain_display_binding(
    *,
    type_column: Sequence[str | None],
    soma_side_column: Sequence[str | None],
    body_ids: Sequence[int],
    index_of: Callable[[int], int],
    node_count: int,
    graph_hash: str,
    annotations_hash: str,
    populations: EmbodiedPopulationBinding,
    ttmn_pool: TtmnPoolBinding,
) -> BrainDisplayBinding:
    """Pure selection over dense-ordered, verified annotation columns."""
    if not (len(type_column) == len(soma_side_column) == len(body_ids) == node_count):
        raise BrainDisplayError("annotation columns must cover the complete dense node space")
    if len(set(body_ids)) != node_count:
        raise BrainDisplayError("annotation body IDs must be unique")
    for binding in (populations, ttmn_pool):
        if (binding.graph_manifest_sha256, binding.annotations_sha256, binding.node_count) != (
            graph_hash,
            annotations_hash,
            node_count,
        ):
            raise BrainDisplayError(
                "population bindings name a different artifact or annotation join"
            )
    selected: list[DisplayPopulation] = []
    for label, prefix in (("LC4", "lc4"), ("DNp01", "dnp01"), ("TTMn", "ttmn")):
        rows = [i for i, value in enumerate(type_column) if value == label]
        if any(soma_side_column[i] not in {"L", "R"} for i in rows):
            raise BrainDisplayError(f"{label} has a member outside the declared L/R split")
        for side, suffix in (("L", "left"), ("R", "right")):
            members = sorted((body_ids[i], i) for i in rows if soma_side_column[i] == side)
            for body, index in members:
                if index_of(body) != index:
                    raise BrainDisplayError("annotation members are not in artifact dense order")
            pop = DisplayPopulation(
                f"{prefix}_{suffix}",
                tuple(body for body, _ in members),
                tuple(index for _, index in members),
            )
            if label == "LC4":
                existing = populations.population(pop.name)
                if (pop.body_ids, pop.dense_indices) != (existing.body_ids, existing.dense_indices):
                    raise BrainDisplayError(
                        "LC4 selection differs from the bound sensory population"
                    )
            elif label == "TTMn":
                expected = sorted(
                    (member.body_id, member.dense_index)
                    for member in ttmn_pool.members
                    if member.soma_side == side
                )
                if members != expected:
                    raise BrainDisplayError(
                        "TTMn selection differs from the bound motor-stage pair"
                    )
            selected.append(pop)
    forbidden = set(LEFT_WING_MOTOR_BODY_IDS + RIGHT_WING_MOTOR_BODY_IDS)
    forbidden.update(faithful_entry_body_ids())
    for bound_population in populations.populations:
        forbidden.update(bound_population.body_ids)
    forbidden.update(member.body_id for member in ttmn_pool.members)
    if any(forbidden.intersection(pop.body_ids) for pop in selected[2:4]):
        raise BrainDisplayError(
            "DNp01 overlaps a driven population or declared motor/entry binding"
        )
    return BrainDisplayBinding(
        graph_hash,
        annotations_hash,
        populations.identity_sha256,
        ttmn_pool.identity_sha256,
        node_count,
        tuple(selected),
    )


def load_brain_display_binding(
    graph: FullRuntimeGraph,
    body_annotations_path: Path,
    populations: EmbodiedPopulationBinding,
    ttmn_binding_hash: str,
) -> BrainDisplayBinding:
    """Reverify the annotation digest and complete body-ID join, as the motor selectors do."""
    table, annotations_hash = _verified_annotation_table(graph, body_annotations_path)
    rows = _verified_graph_array(graph, "annotation_source_rows", "<u8")
    bodies = _verified_graph_array(graph, "body_ids", "<u8")
    if rows.size == 0 or bool((rows >= table.num_rows).any()) or np.unique(rows).size != rows.size:
        raise BrainDisplayError("annotation source rows are missing, repeated, or out of range")
    taken = table.take(rows.astype(np.int64))
    if not np.array_equal(np.asarray(taken.column("bodyId").to_pylist(), dtype=np.uint64), bodies):
        raise BrainDisplayError("annotation bodyId join is not identical to artifact dense order")
    arguments = dict(
        type_column=_string_column(taken, TYPE_COLUMN),
        soma_side_column=_string_column(taken, SIDE_COLUMN),
        body_ids=bodies.tolist(),
        index_of=graph.index_of,
        node_count=graph.manifest.node_count,
        graph_manifest_sha256=graph.manifest.manifest_file_sha256,
        annotations_sha256=annotations_hash,
        populations=populations,
    )
    ttmn = select_ttmn_pool(**arguments)
    if ttmn.identity_sha256 != ttmn_binding_hash:
        raise BrainDisplayError("verified TTMn selection differs from the controller's binding")
    return select_brain_display_binding(
        type_column=arguments["type_column"],
        soma_side_column=arguments["soma_side_column"],
        body_ids=bodies.tolist(),
        index_of=graph.index_of,
        node_count=graph.manifest.node_count,
        graph_hash=graph.manifest.manifest_file_sha256,
        annotations_hash=annotations_hash,
        populations=populations,
        ttmn_pool=ttmn,
    )


@dataclass(frozen=True, slots=True)
class PopulationMean:
    name: str
    mean: float
    count: int

    def __post_init__(self) -> None:
        if self.name not in POPULATION_NAMES:
            raise BrainDisplayError("unknown display population")
        if isinstance(self.mean, bool) or not math.isfinite(self.mean) or not -1 <= self.mean <= 1:
            raise BrainDisplayError("population mean must be finite model activity in [-1, 1]")
        _integer(self.count, "population count", minimum=1, maximum=166_700)


@dataclass(frozen=True, slots=True)
class BrainDisplayFrame:
    identity: RunIdentity
    command_seq: int
    observation_tick: int
    neural_step_start: int
    neural_step_end: int
    state_digest: str
    encoder_schema: str
    binding_hash: str
    populations: tuple[PopulationMean, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "populations", tuple(self.populations))
        if not isinstance(self.identity, RunIdentity):
            raise BrainDisplayError("display frame requires a validated RunIdentity")
        for name in ("command_seq", "observation_tick", "neural_step_start", "neural_step_end"):
            _integer(getattr(self, name), name)
        if self.neural_step_end <= self.neural_step_start:
            raise BrainDisplayError("display frame must name a completed, nonempty neural interval")
        _hash(self.state_digest, "state_digest")
        _hash(self.binding_hash, "binding_hash")
        if (
            not isinstance(self.encoder_schema, str)
            or _IDENTIFIER.fullmatch(self.encoder_schema) is None
        ):
            raise BrainDisplayError(
                "encoder_schema must be a safe identifier of at most 128 characters"
            )
        if tuple(pop.name for pop in self.populations) != POPULATION_NAMES:
            raise BrainDisplayError("display frame requires all six named means in declared order")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "presentation_only": True,
            "identity": self.identity.to_dict(),
            "command_seq": self.command_seq,
            "observation_tick": self.observation_tick,
            "neural_step_start": self.neural_step_start,
            "neural_step_end": self.neural_step_end,
            "state_digest": self.state_digest,
            "encoder_schema": self.encoder_schema,
            "binding_hash": self.binding_hash,
            "populations": {
                pop.name: {"mean": pop.mean, "count": pop.count} for pop in self.populations
            },
        }


class DisplayDocument(Protocol):
    """Frozen display value, serialized only by the bounded background writer."""

    def to_dict(self) -> dict[str, Any]: ...


class AtomicBrainDisplayPublisher:
    """One coalescing slot and one daemon writer; no I/O or serialization in offer()."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.coalesced_count = 0
        self.dropped_count = 0
        self.published_count = 0
        self.last_error: str | None = None
        self._pending: DisplayDocument | None = None
        self._condition = threading.Condition()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="brain-display-writer", daemon=True)
        self._thread.start()

    def offer(self, frame: DisplayDocument) -> bool:
        if not self._condition.acquire(blocking=False):
            self.dropped_count += 1
            return False
        try:
            if self._closed or self.last_error is not None:
                self.dropped_count += 1
                return False
            self.coalesced_count += self._pending is not None
            self._pending = frame
            self._condition.notify()
            return True
        finally:
            self._condition.release()

    def close(self, *, timeout: float = 1.0) -> bool:
        with self._condition:
            self._closed = True
            self._condition.notify()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            self.last_error = "display writer did not stop inside its close budget"
            return False
        return self.last_error is None

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending is not None or self._closed)
                frame, self._pending = self._pending, None
                if frame is None:
                    return
            try:
                self._write(frame)
                self.published_count += 1
            except Exception as exc:
                with self._condition:
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    self._pending = None
                return

    def _write(self, frame: DisplayDocument) -> None:
        data = canonical_json_bytes(frame.to_dict()) + b"\n"
        if len(data) > MAX_FRAME_BYTES:
            raise BrainDisplayError("display frame exceeds the 16 KiB file ceiling")
        descriptor, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
            os.replace(temporary, self.path)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary)


class BrainDisplayProducer:
    """Best-effort completed-step observer; capture failures never escape to the controller.

    Call capture_completed_step(controller, command) in the SAME serialized compute
    worker immediately after controller.step returned. Do not schedule it in the
    writer thread or a service telemetry callback: the next neural step can already
    be running there. The writer receives only six frozen means, never the engine.
    """

    def __init__(
        self,
        binding: BrainDisplayBinding,
        publisher: AtomicBrainDisplayPublisher,
        *,
        every_n_commands: int = 4,
    ) -> None:
        _integer(every_n_commands, "every_n_commands", minimum=1)
        self.binding = binding
        self.publisher = publisher
        self.every_n_commands = every_n_commands
        self.last_error: str | None = None
        self.dropped_count = 0
        self._binding_hash = binding.identity_sha256
        self._verified_binding_objects: tuple[object, object] | None = None

    def capture_completed_step(
        self,
        controller: FullBrainController | ProgramSelectingController,
        command: Action | ProgramAction,
    ) -> bool:
        try:
            return self._capture(controller, command)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.dropped_count += 1
            return False

    def _capture(
        self,
        controller: FullBrainController | ProgramSelectingController,
        command: Action | ProgramAction,
    ) -> bool:
        from .program_policy import ProgramSelectingController

        if command.command_seq % self.every_n_commands or self.publisher.last_error is not None:
            return False
        core = (
            controller.controller
            if isinstance(controller, ProgramSelectingController)
            else controller
        )
        evidence = core.last_evidence
        embodied, escape = core.embodied_senses, core.escape_binding
        if embodied is None or escape is None or evidence is None:
            raise BrainDisplayError(
                "completed full-brain step and sensory/TTMn bindings are required"
            )
        if (
            core.identity != command.identity
            or command.identity.graph_hash != self.binding.graph_hash
        ):
            raise BrainDisplayError("controller/command identity differs from the display binding")
        verified = self._verified_binding_objects
        if verified is None or verified[0] is not embodied.populations or verified[1] is not escape:
            if (
                embodied.populations.identity_sha256 != self.binding.sensory_binding_hash
                or escape.ttmn_pool_identity_sha256 != self.binding.ttmn_binding_hash
            ):
                raise BrainDisplayError("controller sensory/TTMn bindings differ from the display")
            # Both bindings are frozen. Keep strong references, and avoid rehashing
            # thousands of sensory members on every display capture.
            self._verified_binding_objects = (embodied.populations, escape)
        if (
            evidence.command_seq,
            evidence.observation.game_tick,
            evidence.neural_step_start,
            evidence.neural_step_end,
            evidence.state_after_sha256,
        ) != (
            command.command_seq,
            command.observation_tick,
            command.neural_step_start,
            command.neural_step_end,
            command.state_digest,
        ):
            raise BrainDisplayError("completed evidence does not match the command being displayed")
        if core.engine.current_state().state_sha256 != command.state_digest:
            raise BrainDisplayError("the engine advanced beyond the completed command")
        activity = core.engine.activity()
        if activity.shape != (self.binding.node_count,) or activity.flags.writeable:
            raise BrainDisplayError(
                "activity must be an immutable copy of the complete committed state"
            )
        means = tuple(
            PopulationMean(
                pop.name,
                math.fsum(float(activity[i]) for i in pop.dense_indices) / len(pop.body_ids),
                len(pop.body_ids),
            )
            for pop in self.binding.populations
        )
        if core.engine.current_state().state_sha256 != command.state_digest:
            raise BrainDisplayError("the engine advanced during display capture")
        frame = BrainDisplayFrame(
            command.identity,
            command.command_seq,
            command.observation_tick,
            command.neural_step_start,
            command.neural_step_end,
            command.state_digest,
            (core.event_encoder or embodied.encoder).to_identity_material()["schema"],
            self._binding_hash,
            means,
        )
        return self.publisher.offer(frame)
