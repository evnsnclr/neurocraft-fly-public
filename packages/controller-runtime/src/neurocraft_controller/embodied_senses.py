"""Engineered-Encoder Mode sensory front end over labeled MaleCNS populations.

This module is the sensory half of the ``full_mcns_eem_v1`` profile (ADR 0008
Mode 3, Engineered-Encoder Mode -- **never** Faithful Organism Mode): a
declared, engineered encoder maps Minecraft's existing physical observations
onto three labeled sensory populations of the verified full MaleCNS artifact,
and everything about that mapping -- selectors, membership, ray-side
convention, formulas, per-channel gains -- is written into hashed identity
documents before anything runs.

Populations (annotation selectors, fail-closed on count drift)
--------------------------------------------------------------
* ``lc4_left`` / ``lc4_right``: ``type == 'LC4'`` split by ``somaSide``
  (L/R). The census guess of ``rootSide`` was checked against the pinned
  table and found null for all 126 LC4 rows; ``somaSide`` carries the
  measured 71 L / 55 R split and is the declared side column.
* ``visual``: ``class == 'visual'`` (6,091 nodes; photoreceptors and other
  cells the source labels visual).
* ``bristle``: ``subclass == 'mechanosensory bristle'`` (2,206 nodes).

The four populations are required to be pairwise disjoint and disjoint from
the two faithful sensory entries (SNpp10 805722 / JO-EV3 77185); on the
pinned table the measured pairwise overlaps are all zero, and any nonzero
overlap refuses construction rather than being summed away.

Ray-side convention (declared, verified in source)
--------------------------------------------------
Protocol ray order is ``forward, back, left, right, up, down``
(``PhysicalFeatureEncoder.SCHEMA_MATERIAL['ray_order']``), and the mod casts
the rays in exactly that order (``FlySensors.java``: ``forward``,
``forward.reverse()``, ``right.reverse()`` labeled left, ``right``,
``bodyUp``, ``bodyUp.reverse()``). The encoder therefore reads index 0 as the
front ray, index 2 as the left ray, and index 3 as the right ray; each side's
"max over side rays" is a max over that singleton set, declared as such.

Claim discipline
----------------
Proximity is not looming, uniform per-capita drive is not retinotopy, and a
light fraction is not a photoreceptor transduction model. Nothing here claims
biological encoding; the claim is only that the declared mapping was executed
exactly. The brain and the wing readout are untouched -- this module writes a
dense sensory vector and identity documents, nothing else.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from neurocraft_protocol import SensorFrame

from .config import ControllerConfiguration, canonical_sha256

if TYPE_CHECKING:
    from neurocraft_connectome.full_runtime import FullRuntimeGraph
    from numpy.typing import NDArray

EEM_POPULATION_BINDING_SCHEMA: Final = "neurocraft.eem.population_binding.v1"
EEM_ENCODER_SCHEMA: Final = "neurocraft.eem.encoder.v1"

#: The ADR 0008 mode-taxonomy label every EEM identity carries.
EEM_MODE_TAXONOMY: Final = "engineered_encoder"

#: Declared ray-side mapping, verified against ``FlySensors.java`` and the
#: pinned ``PhysicalFeatureEncoder`` ray order (forward, back, left, right,
#: up, down).
FRONT_RAY_INDEX: Final = 0
LEFT_RAY_INDEX: Final = 2
RIGHT_RAY_INDEX: Final = 3

#: Annotation source and columns, exactly as the pinned table spells them.
ANNOTATION_SOURCE_ID: Final = "annotations"
BODY_COLUMN: Final = "bodyId"
TYPE_COLUMN: Final = "type"
CLASS_COLUMN: Final = "class"
SUBCLASS_COLUMN: Final = "subclass"
SIDE_COLUMN: Final = "somaSide"

#: Fail-closed population sizes, verified against the pinned annotations
#: table (sha 2177e246...a9a3b2) joined through ``annotation_source_rows``.
#: A different count is treated as drift and refused, never renormalized.
EXPECTED_LC4_NODES: Final = 126
EXPECTED_LC4_LEFT_NODES: Final = 71
EXPECTED_LC4_RIGHT_NODES: Final = 55
EXPECTED_VISUAL_NODES: Final = 6091
EXPECTED_BRISTLE_NODES: Final = 2206

_MISSING_ANNOTATION_ROW: Final = (1 << 64) - 1
_HASH_BLOCK_BYTES: Final = 8 * 1024 * 1024
_SHA256_HEX_LENGTH: Final = 64


class EmbodiedSensesError(RuntimeError):
    """The embodied-senses binding could not be built exactly as declared."""


def faithful_entry_body_ids() -> frozenset[int]:
    """The two pinned faithful sensory entry bodies, read from the pinned config."""

    configuration = ControllerConfiguration()
    return frozenset((configuration.left_sensor_body_id, configuration.right_sensor_body_id))


# ------------------------------------------------------------------ populations


@dataclass(frozen=True, slots=True)
class EmbodiedPopulation:
    """One selected sensory population with sorted members and dense indices."""

    name: str
    selector: str
    body_ids: tuple[int, ...]
    dense_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        # Frozen dataclasses do not freeze caller-supplied lists. Copy membership
        # containers before any enclosing document may cache their identity.
        object.__setattr__(self, "body_ids", tuple(self.body_ids))
        object.__setattr__(self, "dense_indices", tuple(self.dense_indices))
        if not isinstance(self.name, str) or not isinstance(self.selector, str):
            raise EmbodiedSensesError("population name and selector must be strings")
        if any(type(value) is not int for value in (*self.body_ids, *self.dense_indices)):
            raise EmbodiedSensesError("population members must be exact immutable integers")
        if not self.body_ids:
            raise EmbodiedSensesError(f"population {self.name!r} is empty")
        if len(self.body_ids) != len(self.dense_indices):
            raise EmbodiedSensesError(f"population {self.name!r} members are misaligned")
        pairs = zip(self.body_ids, self.body_ids[1:], strict=False)
        if any(later <= earlier for earlier, later in pairs):
            raise EmbodiedSensesError(f"population {self.name!r} body IDs must ascend strictly")


@dataclass(frozen=True, slots=True)
class EmbodiedPopulationBinding:
    """The canonical population-membership document, hashed into identities."""

    graph_manifest_sha256: str
    annotations_sha256: str
    node_count: int
    populations: tuple[EmbodiedPopulation, ...]
    _identity_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.graph_manifest_sha256, str)
            or not isinstance(self.annotations_sha256, str)
            or type(self.node_count) is not int
        ):
            raise EmbodiedSensesError(
                "binding metadata must contain immutable strings and an integer"
            )
        object.__setattr__(self, "populations", tuple(self.populations))
        if any(not isinstance(population, EmbodiedPopulation) for population in self.populations):
            raise EmbodiedSensesError("binding populations must be immutable population documents")
        # The declaration bytes stay identical; only their repeated reconstruction
        # is removed. Per-step stimulation and activity hashing remain separate.
        object.__setattr__(self, "_identity_sha256", canonical_sha256(self.to_identity_material()))

    def population(self, name: str) -> EmbodiedPopulation:
        for population in self.populations:
            if population.name == name:
                return population
        raise KeyError(f"unknown embodied population {name!r}")

    def to_identity_material(self) -> dict[str, Any]:
        return {
            "annotations_sha256": self.annotations_sha256,
            "graph_manifest_sha256": self.graph_manifest_sha256,
            "node_count": self.node_count,
            "populations": [
                {
                    "members": [
                        [body_id, dense_index]
                        for body_id, dense_index in zip(
                            population.body_ids, population.dense_indices, strict=True
                        )
                    ],
                    "name": population.name,
                    "selector": population.selector,
                }
                for population in self.populations
            ],
            "schema": EEM_POPULATION_BINDING_SCHEMA,
        }

    @property
    def identity_sha256(self) -> str:
        return self._identity_sha256


def _require_sha256(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EmbodiedSensesError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _object_array(values: Sequence[str | None], label: str, expected: int) -> NDArray[np.object_]:
    if len(values) != expected:
        raise EmbodiedSensesError(
            f"annotation column {label!r} has {len(values)} rows; expected {expected}"
        )
    column: NDArray[np.object_] = np.asarray(list(values), dtype=object)
    return column


def _require_count(name: str, observed: int, expected: int) -> None:
    if observed != expected:
        raise EmbodiedSensesError(
            f"population {name!r} count drifted: selected {observed} nodes, the "
            f"declared selector expects exactly {expected}; refusing to bind"
        )


def _build_population(
    name: str,
    selector: str,
    mask: NDArray[np.bool_],
    *,
    body_ids: NDArray[np.uint64],
    index_of: Callable[[int], int],
    node_count: int,
) -> EmbodiedPopulation:
    positions = np.flatnonzero(mask)
    members: list[tuple[int, int]] = []
    for position in positions.tolist():
        body_id = int(body_ids[position])
        try:
            dense_index = index_of(body_id)
        except KeyError as exc:
            raise EmbodiedSensesError(
                f"population {name!r} body {body_id} is not present in the artifact"
            ) from exc
        if dense_index != position:
            raise EmbodiedSensesError(
                f"population {name!r} body {body_id} resolves to dense index "
                f"{dense_index} but sits at annotation row {position}; the "
                "annotation rows are not in dense body order"
            )
        if not 0 <= dense_index < node_count:
            raise EmbodiedSensesError(
                f"population {name!r} dense index {dense_index} is out of range"
            )
        members.append((body_id, dense_index))
    members.sort()
    return EmbodiedPopulation(
        name=name,
        selector=selector,
        body_ids=tuple(body_id for body_id, _ in members),
        dense_indices=tuple(dense_index for _, dense_index in members),
    )


def _refuse_overlaps(
    populations: Sequence[EmbodiedPopulation], faithful_bodies: Collection[int]
) -> None:
    """Refuse any pairwise population overlap and any faithful-entry capture.

    Measured on the pinned table all pairwise overlaps are zero; a nonzero
    overlap means the selectors no longer partition and per-capita drives
    would silently sum, so it is refused rather than clipped away.
    """

    for earlier_index, earlier in enumerate(populations):
        earlier_set = set(earlier.body_ids)
        captured = sorted(earlier_set.intersection(faithful_bodies))
        if captured:
            raise EmbodiedSensesError(
                f"population {earlier.name!r} contains faithful sensory entry "
                f"bodies {captured}; the faithful channels must stay exclusive"
            )
        for later in populations[earlier_index + 1 :]:
            overlap = earlier_set.intersection(later.body_ids)
            if overlap:
                raise EmbodiedSensesError(
                    f"populations {earlier.name!r} and {later.name!r} overlap on "
                    f"{len(overlap)} bodies (e.g. {sorted(overlap)[:5]}); the "
                    "declared populations must be pairwise disjoint"
                )


def select_embodied_populations(
    *,
    type_column: Sequence[str | None],
    class_column: Sequence[str | None],
    subclass_column: Sequence[str | None],
    soma_side_column: Sequence[str | None],
    body_ids: Sequence[int],
    index_of: Callable[[int], int],
    node_count: int,
    graph_manifest_sha256: str,
    annotations_sha256: str,
    faithful_bodies: Collection[int] | None = None,
) -> EmbodiedPopulationBinding:
    """Select the declared populations from dense-ordered annotation columns.

    Pure over its inputs so the CI tests can drive it with a synthetic
    annotation frame; :func:`load_embodied_population_binding` supplies the
    verified real columns. Every count gate fails closed.
    """

    _require_sha256(graph_manifest_sha256, "graph_manifest_sha256")
    _require_sha256(annotations_sha256, "annotations_sha256")
    row_count = len(body_ids)
    if row_count == 0 or row_count > node_count:
        raise EmbodiedSensesError("annotation rows must be nonempty and within the node count")
    bodies: NDArray[np.uint64] = np.asarray(list(body_ids), dtype=np.uint64)
    types = _object_array(type_column, TYPE_COLUMN, row_count)
    classes = _object_array(class_column, CLASS_COLUMN, row_count)
    subclasses = _object_array(subclass_column, SUBCLASS_COLUMN, row_count)
    sides = _object_array(soma_side_column, SIDE_COLUMN, row_count)

    lc4_mask: NDArray[np.bool_] = np.asarray(types == "LC4", dtype=bool)
    _require_count("lc4", int(lc4_mask.sum()), EXPECTED_LC4_NODES)
    lc4_sides = {value for value in sides[lc4_mask].tolist()}
    if not lc4_sides.issubset({"L", "R"}):
        raise EmbodiedSensesError(
            f"LC4 somaSide values {sorted(str(value) for value in lc4_sides - {'L', 'R'})} "
            "are outside the declared L/R split; refusing to bind"
        )
    left_mask: NDArray[np.bool_] = lc4_mask & np.asarray(sides == "L", dtype=bool)
    right_mask: NDArray[np.bool_] = lc4_mask & np.asarray(sides == "R", dtype=bool)
    _require_count("lc4_left", int(left_mask.sum()), EXPECTED_LC4_LEFT_NODES)
    _require_count("lc4_right", int(right_mask.sum()), EXPECTED_LC4_RIGHT_NODES)
    visual_mask: NDArray[np.bool_] = np.asarray(classes == "visual", dtype=bool)
    _require_count("visual", int(visual_mask.sum()), EXPECTED_VISUAL_NODES)
    bristle_mask: NDArray[np.bool_] = np.asarray(subclasses == "mechanosensory bristle", dtype=bool)
    _require_count("bristle", int(bristle_mask.sum()), EXPECTED_BRISTLE_NODES)

    populations = (
        _build_population(
            "lc4_left",
            f"{TYPE_COLUMN} == 'LC4' and {SIDE_COLUMN} == 'L'",
            left_mask,
            body_ids=bodies,
            index_of=index_of,
            node_count=node_count,
        ),
        _build_population(
            "lc4_right",
            f"{TYPE_COLUMN} == 'LC4' and {SIDE_COLUMN} == 'R'",
            right_mask,
            body_ids=bodies,
            index_of=index_of,
            node_count=node_count,
        ),
        _build_population(
            "visual",
            f"{CLASS_COLUMN} == 'visual'",
            visual_mask,
            body_ids=bodies,
            index_of=index_of,
            node_count=node_count,
        ),
        _build_population(
            "bristle",
            f"{SUBCLASS_COLUMN} == 'mechanosensory bristle'",
            bristle_mask,
            body_ids=bodies,
            index_of=index_of,
            node_count=node_count,
        ),
    )
    _refuse_overlaps(
        populations,
        faithful_entry_body_ids() if faithful_bodies is None else faithful_bodies,
    )
    return EmbodiedPopulationBinding(
        graph_manifest_sha256=graph_manifest_sha256,
        annotations_sha256=annotations_sha256,
        node_count=node_count,
        populations=populations,
    )


# ---------------------------------------------------------------- verified load


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(_HASH_BLOCK_BYTES)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _verified_graph_array(graph: FullRuntimeGraph, name: str, dtype: str) -> NDArray[Any]:
    """One artifact array as numpy, re-checked against its manifest spec.

    Mirrors the licensed pattern in :mod:`neurocraft_neural.full_signs` (which
    duplicates it from :mod:`neurocraft_neural.full_view`): the per-element
    integer interface would box 166,700 Python integers, so the verified
    mapping is read directly and the digest re-check is what makes that read
    self-validating rather than trusted.
    """

    specs = {spec.name: spec for spec in graph.manifest.arrays}
    try:
        spec = specs[name]
    except KeyError as exc:  # pragma: no cover - manifest schema validated upstream
        raise EmbodiedSensesError(f"full artifact has no array {name!r}") from exc
    mapped = getattr(graph, name)
    mapping = getattr(mapped, "_mmap", None)
    if mapping is None:  # pragma: no cover - upstream layout change
        raise EmbodiedSensesError(f"array {name!r} does not expose a verified mapping")
    buffer = memoryview(mapping)
    if buffer.nbytes != spec.byte_length:
        raise EmbodiedSensesError(f"array {name!r} mapping length does not match the manifest")
    if hashlib.sha256(buffer).hexdigest() != spec.sha256:
        raise EmbodiedSensesError(f"array {name!r} mapping digest does not match the manifest")
    values: NDArray[Any] = np.frombuffer(buffer, dtype=dtype)
    if values.shape[0] != spec.count:
        raise EmbodiedSensesError(f"array {name!r} element count does not match the manifest")
    return values


def _verified_annotation_table(
    graph: FullRuntimeGraph, body_annotations_path: Path
) -> tuple[Any, str]:
    """Open the body-annotations table only after its declared digest matches."""

    by_id = {source.source_id: source for source in graph.manifest.source_artifacts}
    try:
        declared = by_id[ANNOTATION_SOURCE_ID]
    except KeyError as exc:
        raise EmbodiedSensesError(
            f"artifact manifest declares no {ANNOTATION_SOURCE_ID!r} source"
        ) from exc
    try:
        observed_bytes = body_annotations_path.stat().st_size
    except OSError as exc:
        raise EmbodiedSensesError(f"cannot read the body-annotations table: {exc}") from exc
    if observed_bytes != declared.byte_length:
        raise EmbodiedSensesError(
            "body-annotations table byte length mismatch: "
            f"declared {declared.byte_length}, observed {observed_bytes}"
        )
    observed_sha256 = _file_sha256(body_annotations_path)
    if observed_sha256 != declared.sha256:
        raise EmbodiedSensesError(
            "body-annotations table SHA-256 mismatch: "
            f"declared {declared.sha256}, observed {observed_sha256}"
        )
    import pyarrow.feather as feather  # type: ignore[import-untyped]

    table = feather.read_table(
        body_annotations_path,
        columns=[BODY_COLUMN, TYPE_COLUMN, CLASS_COLUMN, SUBCLASS_COLUMN, SIDE_COLUMN],
        memory_map=True,
    )
    declared_rows = graph.manifest.counts["annotation_source_rows"]
    if int(table.num_rows) != declared_rows:
        raise EmbodiedSensesError(
            "body-annotations table row count mismatch: "
            f"declared {declared_rows}, observed {int(table.num_rows)}"
        )
    return table, observed_sha256


def _string_column(table: Any, name: str) -> list[str | None]:
    return [None if value is None else str(value) for value in table.column(name).to_pylist()]


def load_embodied_population_binding(
    graph: FullRuntimeGraph, body_annotations_path: Path
) -> EmbodiedPopulationBinding:
    """Join the pinned body-annotations table and select the declared populations.

    The join discipline is the one ``full_signs.py`` and the readout census
    use: pointer rows must be unique, in range, sentinel-free, and the
    ``bodyId`` column taken through them must be bitwise identical to the
    artifact's ``body_ids`` before any selector runs.
    """

    table, annotations_sha256 = _verified_annotation_table(graph, body_annotations_path)
    annotation_rows = _verified_graph_array(graph, "annotation_source_rows", "<u8")
    body_ids = _verified_graph_array(graph, "body_ids", "<u8")
    if bool((annotation_rows == np.uint64(_MISSING_ANNOTATION_ROW)).any()):
        raise EmbodiedSensesError(
            "a retained node has no annotation row; the population selectors "
            "require complete annotation coverage"
        )
    rows = annotation_rows.astype(np.int64)
    if int(rows.min()) < 0 or int(rows.max()) >= int(table.num_rows):
        raise EmbodiedSensesError("annotation source row is out of table range")
    if np.unique(rows).size != rows.size:
        raise EmbodiedSensesError("annotation source rows are not unique")
    taken = table.take(rows)
    taken_bodies = np.asarray(taken.column(BODY_COLUMN).to_pylist(), dtype=np.uint64)
    if not np.array_equal(taken_bodies, body_ids):
        raise EmbodiedSensesError(
            "bodyId column taken through annotation_source_rows is not bitwise "
            "identical to the artifact body IDs; refusing a misaligned join"
        )
    return select_embodied_populations(
        type_column=_string_column(taken, TYPE_COLUMN),
        class_column=_string_column(taken, CLASS_COLUMN),
        subclass_column=_string_column(taken, SUBCLASS_COLUMN),
        soma_side_column=_string_column(taken, SIDE_COLUMN),
        body_ids=body_ids.tolist(),
        index_of=graph.index_of,
        node_count=graph.manifest.node_count,
        graph_manifest_sha256=graph.manifest.manifest_file_sha256,
        annotations_sha256=annotations_sha256,
    )


# -------------------------------------------------------------------- encoder


@dataclass(frozen=True, slots=True)
class EemChannel:
    """One declared encoder channel: a formula, target populations, one gain."""

    name: str
    unit: str
    populations: tuple[str, ...]
    formula: str
    gain: float = 1.0

    def __post_init__(self) -> None:
        if not self.populations:
            raise EmbodiedSensesError(f"channel {self.name!r} targets no population")
        if not np.isfinite(self.gain) or self.gain <= 0.0:
            raise EmbodiedSensesError(f"channel {self.name!r} gain must be finite and positive")


#: The five declared channels. All gains default to 1.0: v1 declares no
#: tuning, and the half amplitude of ``threat_front`` is part of its formula,
#: not a hidden gain. Formulas are the executable truth
#: (:func:`encode_channel_values` / :func:`population_drives`); the strings
#: here are the human-readable declaration hashed into the encoder identity.
EEM_CHANNELS: Final[tuple[EemChannel, ...]] = (
    EemChannel(
        name="threat_left",
        unit="fraction",
        populations=("lc4_left",),
        formula=(
            "max over left-side rays {ray_distances_m[2]} of "
            "clip(1 - distance / ray_max_distance_m, 0, 1)"
        ),
    ),
    EemChannel(
        name="threat_right",
        unit="fraction",
        populations=("lc4_right",),
        formula=(
            "max over right-side rays {ray_distances_m[3]} of "
            "clip(1 - distance / ray_max_distance_m, 0, 1)"
        ),
    ),
    EemChannel(
        name="threat_front",
        unit="fraction",
        populations=("lc4_left", "lc4_right"),
        formula=(
            "clip(1 - ray_distances_m[0] / ray_max_distance_m, 0, 1) / 2, added to "
            "both LC4 sides; the per-population sum is clipped to [0, 1]"
        ),
    ),
    EemChannel(
        name="light",
        unit="fraction",
        populations=("visual",),
        formula="local_light / 15",
    ),
    EemChannel(
        name="touch",
        unit="fraction",
        populations=("bristle",),
        formula="1.0 if horizontal_collision or vertical_collision else 0.0",
    ),
)


@dataclass(frozen=True, slots=True)
class EemEncoderDocument:
    """The canonical encoder declaration, hashed into the adapter identity."""

    population_binding_sha256: str
    channels: tuple[EemChannel, ...] = EEM_CHANNELS

    def to_identity_material(self) -> dict[str, Any]:
        return {
            "channels": [
                {
                    "formula": channel.formula,
                    "gain": float(channel.gain).hex(),
                    "name": channel.name,
                    "populations": list(channel.populations),
                    "unit": channel.unit,
                }
                for channel in self.channels
            ],
            "drive_semantics": (
                "per-capita: every member of a target population receives the "
                "identical channel value times the declared gain; contributions "
                "from multiple channels to one population are summed and clipped "
                "to [0, 1]; no per-member weights exist"
            ),
            "population_binding_sha256": self.population_binding_sha256,
            "ray_semantics": {
                "front_ray_index": FRONT_RAY_INDEX,
                "left_ray_index": LEFT_RAY_INDEX,
                "ray_order": ["forward", "back", "left", "right", "up", "down"],
                "right_ray_index": RIGHT_RAY_INDEX,
                "source": (
                    "mod FlySensors.java casts rays forward, back, left "
                    "(right.reverse()), right, up, down; the pinned "
                    "PhysicalFeatureEncoder schema pins the same order"
                ),
            },
            "schema": EEM_ENCODER_SCHEMA,
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.to_identity_material())


@dataclass(frozen=True, slots=True)
class EmbodiedSensesBinding:
    """Populations plus encoder, cross-checked, ready for the controller."""

    populations: EmbodiedPopulationBinding
    encoder: EemEncoderDocument

    def __post_init__(self) -> None:
        if self.encoder.population_binding_sha256 != self.populations.identity_sha256:
            raise EmbodiedSensesError(
                "encoder document does not bind the supplied population binding"
            )
        known = {population.name for population in self.populations.populations}
        for channel in self.encoder.channels:
            unknown = [name for name in channel.populations if name not in known]
            if unknown:
                raise EmbodiedSensesError(
                    f"channel {channel.name!r} targets unknown populations {unknown}"
                )


def bind_embodied_senses(populations: EmbodiedPopulationBinding) -> EmbodiedSensesBinding:
    """Build the declared v1 encoder document over one population binding."""

    return EmbodiedSensesBinding(
        populations=populations,
        encoder=EemEncoderDocument(population_binding_sha256=populations.identity_sha256),
    )


def _clip_unit(value: float) -> float:
    return min(1.0, max(0.0, value))


def encode_channel_values(sensors: SensorFrame) -> tuple[tuple[str, float], ...]:
    """The five declared channel values for one sensor frame, all in [0, 1]."""

    if not isinstance(sensors, SensorFrame):
        raise EmbodiedSensesError("the embodied encoder accepts only a protocol SensorFrame")
    ray_max = sensors.ray_max_distance_m

    def proximity(slot: int) -> float:
        return _clip_unit(1.0 - sensors.ray_distances_m[slot] / ray_max)

    values = (
        ("threat_left", proximity(LEFT_RAY_INDEX)),
        ("threat_right", proximity(RIGHT_RAY_INDEX)),
        ("threat_front", proximity(FRONT_RAY_INDEX) / 2.0),
        ("light", sensors.local_light / 15.0),
        (
            "touch",
            1.0 if (sensors.horizontal_collision or sensors.vertical_collision) else 0.0,
        ),
    )
    for name, value in values:
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise EmbodiedSensesError(f"channel {name!r} produced {value!r} outside [0, 1]")
    return values


def population_drives(
    channel_values: Mapping[str, float], channels: Sequence[EemChannel]
) -> tuple[tuple[str, float], ...]:
    """Per-population drive: sum of gain * channel value, clipped to [0, 1]."""

    totals: dict[str, float] = {}
    for channel in channels:
        try:
            value = channel_values[channel.name]
        except KeyError as exc:
            raise EmbodiedSensesError(f"channel {channel.name!r} has no encoded value") from exc
        for population in channel.populations:
            totals[population] = totals.get(population, 0.0) + channel.gain * value
    return tuple((name, _clip_unit(total)) for name, total in totals.items())


# ------------------------------------------------- v2 encoder: temporal contrast


EEM_ENCODER_V2_SCHEMA: Final = "neurocraft.eem.encoder.v2"

#: Declared leaky-mean rate of the v2 temporal-contrast encoder: 1/8 per game
#: tick, applied once per controller step (never per neural sub-step).
#: Rationale, declared before any measurement: at 20 game ticks/s this is a
#: ~0.4 s adaptation time constant -- fast enough that a standing condition
#: (the constant max light that pinned v1's 6,091 visual cells at drive 1.0)
#: adapts away within seconds, slow enough that a one-tick event is scored
#: against an almost-unmoved baseline -- and 1/8 is dyadic, so the update is
#: exact in binary floating point.
EEM_CONTRAST_ALPHA: Final = 0.125

#: Channels encoded as temporal contrast in v2: the drive is the deviation of
#: the v1 raw value from its leaky running mean, not the absolute level.
EEM_V2_CONTRAST_CHANNEL_NAMES: Final = ("threat_left", "threat_right", "threat_front", "light")

#: The declared exception: ``touch`` stays ABSOLUTE in v2. A collision flag is
#: already an event, not a standing field; contrast of a binary would fire
#: only on its edges, and a sustained contact would vanish from the drive
#: within a few ticks.
EEM_V2_ABSOLUTE_CHANNEL_NAMES: Final = ("touch",)

_CONTRAST_UPDATE_DECLARATION: Final = (
    "drive_t = clip(|raw_t - m_(t-1)|, 0, 1); m_t = (1 - alpha) * m_(t-1) + "
    "alpha * raw_t; one update per game tick, never per neural sub-step"
)
_CONTRAST_INITIALIZATION_DECLARATION: Final = (
    "the baseline m_(t-1) is initialized to raw_0 on the first tick after "
    "construction or reset(), so drive_0 = 0 and the update yields m_0 = raw_0"
)


def _contrast_formula(raw_formula: str) -> str:
    """The declared v2 formula string: temporal contrast over the v1 raw value."""

    return (
        f"temporal contrast of the v1 raw value raw_t = {raw_formula}: "
        f"{_CONTRAST_UPDATE_DECLARATION}, alpha = 1/8; "
        f"{_CONTRAST_INITIALIZATION_DECLARATION}"
    )


#: The five v2 channels: identical names, units, target populations, and 1.0
#: gains as v1; only the encoding of the value changes. The four contrast
#: channels wrap the byte-identical v1 raw formulas in the declared contrast
#: encoding; ``touch`` is carried over verbatim as the declared absolute
#: exception.
EEM_V2_CHANNELS: Final[tuple[EemChannel, ...]] = tuple(
    channel
    if channel.name in EEM_V2_ABSOLUTE_CHANNEL_NAMES
    else EemChannel(
        name=channel.name,
        unit=channel.unit,
        populations=channel.populations,
        formula=_contrast_formula(channel.formula),
        gain=channel.gain,
    )
    for channel in EEM_CHANNELS
)


@dataclass(frozen=True, slots=True)
class EemEncoderDocumentV2(EemEncoderDocument):
    """The canonical v2 (temporal-contrast) encoder declaration.

    Motivated by the recorded v1 live observation (three fault-free sessions,
    2026-08-31): near-straight flight with imperceptible responses, live
    motors matching the offline all-open probe bit-for-bit. Diagnosis:
    absolute signal levels put a huge DC drive on the labeled populations
    (constant max light pinned all 6,091 visual cells at 1.0) and the
    calibration stream's varied signals set a p95 far above the live dynamic
    range, compressing real responses ~20x. v2 therefore drives each
    population with the DEVIATION of its signal from a declared leaky running
    mean; ``touch`` stays absolute (a collision is already an event).

    The rectification choice is disclosed plainly: real ON (brightening/
    approach) and OFF (dimming/recede) pathways are separate cell types; v2
    declares on-drive = clip(raw_t - m_(t-1), 0, 1) and off-drive =
    clip(m_(t-1) - raw_t, 0, 1) but maps BOTH onto the SAME population
    (summed, clipped), which -- because at most one of the two is nonzero --
    is exactly drive_t = clip(|raw_t - m_(t-1)|, 0, 1). This is an
    engineering simplification, not a model of ON/OFF physiology.
    """

    channels: tuple[EemChannel, ...] = EEM_V2_CHANNELS
    alpha: float = EEM_CONTRAST_ALPHA
    contrast_channels: tuple[str, ...] = EEM_V2_CONTRAST_CHANNEL_NAMES
    absolute_channels: tuple[str, ...] = EEM_V2_ABSOLUTE_CHANNEL_NAMES

    def __post_init__(self) -> None:
        if self.alpha != EEM_CONTRAST_ALPHA:
            raise EmbodiedSensesError(
                "the v2 encoder declares alpha = 1/8 exactly; a different rate "
                "is a different (undeclared) encoder"
            )
        names = {channel.name for channel in self.channels}
        contrast = set(self.contrast_channels)
        absolute = set(self.absolute_channels)
        if contrast & absolute or (contrast | absolute) != names:
            raise EmbodiedSensesError(
                "the v2 contrast/absolute channel sets must partition the declared channels exactly"
            )

    def to_identity_material(self) -> dict[str, Any]:
        encoding = {name: "temporal_contrast" for name in self.contrast_channels}
        encoding.update(dict.fromkeys(self.absolute_channels, "absolute"))
        return {
            "channels": [
                {
                    "encoding": encoding[channel.name],
                    "formula": channel.formula,
                    "gain": float(channel.gain).hex(),
                    "name": channel.name,
                    "populations": list(channel.populations),
                    "unit": channel.unit,
                }
                for channel in self.channels
            ],
            "contrast": {
                "alpha_float_hex": float(self.alpha).hex(),
                "alpha_rationale": (
                    "1/8 per game tick (~0.4 s at 20 ticks/s): standing "
                    "conditions adapt away in seconds while one-tick events "
                    "are scored against an almost-unmoved baseline; dyadic, "
                    "so exact in binary floating point"
                ),
                "initialization": _CONTRAST_INITIALIZATION_DECLARATION,
                "on_off_disclosure": (
                    "real ON and OFF pathways are separate cell types; v2 "
                    "declares on-drive = clip(raw_t - m_(t-1), 0, 1) and "
                    "off-drive = clip(m_(t-1) - raw_t, 0, 1) but maps both "
                    "onto the SAME population (summed, clipped), which equals "
                    "clip(|raw_t - m_(t-1)|, 0, 1) because at most one is "
                    "nonzero; an engineering simplification, not ON/OFF "
                    "physiology"
                ),
                "rectification_rationale": (
                    "the absolute value keeps both signal onsets and offsets "
                    "visible to the same population without doubling the "
                    "declared channel count"
                ),
                "state": (
                    "the running means are the only cross-tick state: pure "
                    "functions of the observation history (same observation "
                    "sequence reproduces the same means, drives, and -- on "
                    "the same backend -- the same trajectory), updated once "
                    "per game tick, cleared by reset()"
                ),
                "update": _CONTRAST_UPDATE_DECLARATION,
            },
            "drive_semantics": (
                "per-capita: every member of a target population receives the "
                "identical channel value times the declared gain; contributions "
                "from multiple channels to one population are summed and clipped "
                "to [0, 1]; no per-member weights exist"
            ),
            "population_binding_sha256": self.population_binding_sha256,
            "ray_semantics": {
                "front_ray_index": FRONT_RAY_INDEX,
                "left_ray_index": LEFT_RAY_INDEX,
                "ray_order": ["forward", "back", "left", "right", "up", "down"],
                "right_ray_index": RIGHT_RAY_INDEX,
                "source": (
                    "mod FlySensors.java casts rays forward, back, left "
                    "(right.reverse()), right, up, down; the pinned "
                    "PhysicalFeatureEncoder schema pins the same order"
                ),
            },
            "schema": EEM_ENCODER_V2_SCHEMA,
            "touch_exception": (
                "touch stays absolute: a collision flag is already an event, "
                "and contrast of a binary would fire only on its edges while "
                "a sustained contact vanished from the drive within ticks"
            ),
        }


def bind_embodied_senses_v2(populations: EmbodiedPopulationBinding) -> EmbodiedSensesBinding:
    """Build the declared v2 (temporal-contrast) encoder over one population binding."""

    return EmbodiedSensesBinding(
        populations=populations,
        encoder=EemEncoderDocumentV2(population_binding_sha256=populations.identity_sha256),
    )


# --------------------------------------- escape-v2 encoder: transient touch


EEM_ENCODER_ESCAPE_V2_SCHEMA: Final = "neurocraft.eem.encoder.escape.v2"

#: Channels encoded as temporal contrast in the escape-v2 encoder: ``touch``
#: ONLY. The eem v2 encoder applied the frozen D-059 contrast formula to every
#: channel EXCEPT touch; the escape-v2 identity applies it to touch alone.
ESCAPE_V2_CONTRAST_CHANNEL_NAMES: Final = ("touch",)

#: The four channels that keep the v1 ABSOLUTE treatment verbatim, for the
#: recorded D-060 reason: temporal contrast adapts a held threat away, which
#: is the wrong boundary for an arc that scores looming events against a
#: quiet motor stage.
ESCAPE_V2_ABSOLUTE_CHANNEL_NAMES: Final = (
    "threat_left",
    "threat_right",
    "threat_front",
    "light",
)

#: The five escape-v2 channels: identical names, units, target populations,
#: and 1.0 gains as v1; only ``touch`` changes encoding, wrapping the
#: byte-identical v1 raw formula in the declared contrast encoding. The four
#: absolute channels are carried over from v1 verbatim.
ESCAPE_V2_CHANNELS: Final[tuple[EemChannel, ...]] = tuple(
    EemChannel(
        name=channel.name,
        unit=channel.unit,
        populations=channel.populations,
        formula=_contrast_formula(channel.formula),
        gain=channel.gain,
    )
    if channel.name in ESCAPE_V2_CONTRAST_CHANNEL_NAMES
    else channel
    for channel in EEM_CHANNELS
)


@dataclass(frozen=True, slots=True)
class EemEncoderDocumentEscapeV2(EemEncoderDocumentV2):
    """The canonical escape-v2 (transient-touch) encoder declaration.

    Identity ``eem-escape-v2`` (frozen in
    ``experiments/preregistration/eem-escape-v2.md``). Motivated by the
    recorded escape-v1 live observations (D-063/D-064): sustained contact
    holds the raw touch flag true on every authoritative tick, the v1
    absolute touch channel therefore drives all bristle members at the full
    per-capita amplitude continuously, and the measured collision response
    holds as a continuous state rather than an event. Under this identity
    the ``touch`` channel alone moves under the frozen D-059
    temporal-contrast formula (verbatim :func:`apply_temporal_contrast` with
    ``contrast_channels = ('touch',)``); the four other channels keep the v1
    ABSOLUTE treatment verbatim. Nothing here is fit to any measurement: the
    formula and alpha are inherited from the frozen D-059 identity.
    """

    channels: tuple[EemChannel, ...] = ESCAPE_V2_CHANNELS
    contrast_channels: tuple[str, ...] = ESCAPE_V2_CONTRAST_CHANNEL_NAMES
    absolute_channels: tuple[str, ...] = ESCAPE_V2_ABSOLUTE_CHANNEL_NAMES

    def __post_init__(self) -> None:
        if self.contrast_channels != ESCAPE_V2_CONTRAST_CHANNEL_NAMES:
            raise EmbodiedSensesError(
                "the escape-v2 encoder applies temporal contrast to the touch "
                "channel only; a different contrast set is a different "
                "(undeclared) encoder"
            )
        # The v2 gates apply unchanged: alpha is pinned to 1/8 exactly and the
        # contrast/absolute sets must partition the declared channels.
        EemEncoderDocumentV2.__post_init__(self)

    def to_identity_material(self) -> dict[str, Any]:
        encoding = {name: "temporal_contrast" for name in self.contrast_channels}
        encoding.update(dict.fromkeys(self.absolute_channels, "absolute"))
        return {
            "absolute_exception": (
                "threat_left, threat_right, threat_front, and light keep the "
                "v1 ABSOLUTE treatment verbatim, for the recorded D-060 "
                "reason: temporal contrast adapts a held threat away, which "
                "is the wrong boundary for an arc that scores looming events "
                "against a quiet motor stage"
            ),
            "channels": [
                {
                    "encoding": encoding[channel.name],
                    "formula": channel.formula,
                    "gain": float(channel.gain).hex(),
                    "name": channel.name,
                    "populations": list(channel.populations),
                    "unit": channel.unit,
                }
                for channel in self.channels
            ],
            "contrast": {
                "alpha_float_hex": float(self.alpha).hex(),
                "alpha_rationale": (
                    "1/8 per game tick (~0.4 s at 20 ticks/s), inherited "
                    "verbatim from the frozen D-059 identity: a contact onset "
                    "drives the bristles near full amplitude and decays with "
                    "tau ~= 8 ticks, sustained resting contact adapts toward "
                    "zero within seconds; dyadic, so exact in binary floating "
                    "point"
                ),
                "initialization": _CONTRAST_INITIALIZATION_DECLARATION,
                "on_off_disclosure": (
                    "a release produces an OFF transient of the same sign as "
                    "the onset: on-drive = clip(raw_t - m_(t-1), 0, 1) and "
                    "off-drive = clip(m_(t-1) - raw_t, 0, 1) both map onto "
                    "the SAME bristle population (summed, clipped), which "
                    "equals clip(|raw_t - m_(t-1)|, 0, 1) because at most one "
                    "is nonzero; an engineering simplification, not ON/OFF "
                    "physiology, exactly as D-059 disclosed it"
                ),
                "state": (
                    "the touch running mean is the only cross-tick state: a "
                    "pure function of the observation history (same "
                    "observation sequence reproduces the same mean, drives, "
                    "and -- on the same backend -- the same trajectory), "
                    "updated once per game tick, cleared by reset(), never "
                    "serialized"
                ),
                "update": _CONTRAST_UPDATE_DECLARATION,
            },
            "drive_semantics": (
                "per-capita: every member of a target population receives the "
                "identical channel value times the declared gain; contributions "
                "from multiple channels to one population are summed and clipped "
                "to [0, 1]; no per-member weights exist"
            ),
            "population_binding_sha256": self.population_binding_sha256,
            "ray_semantics": {
                "front_ray_index": FRONT_RAY_INDEX,
                "left_ray_index": LEFT_RAY_INDEX,
                "ray_order": ["forward", "back", "left", "right", "up", "down"],
                "right_ray_index": RIGHT_RAY_INDEX,
                "source": (
                    "mod FlySensors.java casts rays forward, back, left "
                    "(right.reverse()), right, up, down; the pinned "
                    "PhysicalFeatureEncoder schema pins the same order"
                ),
            },
            "schema": EEM_ENCODER_ESCAPE_V2_SCHEMA,
            "touch_rationale": (
                "touch is the one channel whose raw signal is a state "
                "(contact held true by the physics for as long as the body "
                "rests on a surface) while the engineering-relevant signal is "
                "the event (impact or release); under the declared contrast "
                "treatment a contact onset drives the bristles near full "
                "amplitude, sustained resting contact adapts toward zero, and "
                "a release produces an OFF transient of the same sign"
            ),
        }


def bind_embodied_senses_escape_v2(
    populations: EmbodiedPopulationBinding,
) -> EmbodiedSensesBinding:
    """Build the declared escape-v2 (transient-touch) encoder over one binding."""

    return EmbodiedSensesBinding(
        populations=populations,
        encoder=EemEncoderDocumentEscapeV2(population_binding_sha256=populations.identity_sha256),
    )


def apply_temporal_contrast(
    raw_values: Sequence[tuple[str, float]],
    previous_means: Mapping[str, float] | None,
    *,
    alpha: float = EEM_CONTRAST_ALPHA,
    contrast_channels: Collection[str] = EEM_V2_CONTRAST_CHANNEL_NAMES,
    rectified_channels: Collection[str] = (),
) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, float], ...], dict[str, float]]:
    """Encode raw channel values as temporal contrast against leaky means.

    Pure over its inputs: given the same raw values and the same previous
    means it returns the same result, so the controller's running means stay
    a pure function of the observation history. Returns
    ``(encoded_values, means_used, new_means)`` where ``encoded_values``
    carries every channel (contrast channels as ``clip(|raw - mean|, 0, 1)``,
    absolute channels passed through verbatim), ``means_used`` records the
    exact baseline each contrast channel was scored against this tick, and
    ``new_means`` is the post-update leaky mean per contrast channel.

    ``previous_means=None`` is the declared first tick: the baseline is the
    raw value itself, so every contrast drive is 0 and the update yields
    ``m_0 = raw_0``.

    ``rectified_channels`` must be a subset of ``contrast_channels`` and is
    empty by default, so every caller that does not pass it is byte-unchanged.
    A rectified channel takes the NON-NEGATIVE HALF of the deviation --
    ``clip(raw - mean, 0, 1)`` instead of ``clip(|raw - mean|, 0, 1)`` -- so a
    rise above the baseline fires and a fall does not. Which channels are
    rectified is a property of the bound encoder document
    (:class:`neurocraft_controller.loom_senses.EemEncoderDocumentLoomV1`
    declares it), never of this function's default; the leaky-mean update
    itself is identical in both cases.
    """

    if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise EmbodiedSensesError("contrast alpha must be finite and in (0, 1]")
    if not set(rectified_channels) <= set(contrast_channels):
        raise EmbodiedSensesError(
            "every rectified channel must also be a contrast channel; a "
            "rectified absolute channel has no baseline to rectify against"
        )
    tracked = sorted(name for name, _ in raw_values if name in contrast_channels)
    if previous_means is not None and sorted(previous_means) != tracked:
        raise EmbodiedSensesError(
            "previous contrast means must cover exactly the contrast channels; "
            f"expected {tracked}, got {sorted(previous_means)}"
        )
    encoded: list[tuple[str, float]] = []
    means_used: list[tuple[str, float]] = []
    new_means: dict[str, float] = {}
    for name, raw in raw_values:
        if not np.isfinite(raw) or not 0.0 <= raw <= 1.0:
            raise EmbodiedSensesError(f"channel {name!r} raw value {raw!r} is outside [0, 1]")
        if name not in contrast_channels:
            encoded.append((name, raw))
            continue
        baseline = raw if previous_means is None else float(previous_means[name])
        if not np.isfinite(baseline) or not 0.0 <= baseline <= 1.0:
            raise EmbodiedSensesError(
                f"channel {name!r} running mean {baseline!r} is outside [0, 1]"
            )
        deviation = raw - baseline
        # ``_clip_unit`` already clamps the low side to 0.0, so the rectified
        # branch IS clip(raw - mean, 0, 1); the two-sided branch is unchanged.
        encoded.append(
            (name, _clip_unit(deviation if name in rectified_channels else abs(deviation)))
        )
        means_used.append((name, baseline))
        new_means[name] = (1.0 - alpha) * baseline + alpha * raw
    return tuple(encoded), tuple(means_used), new_means
