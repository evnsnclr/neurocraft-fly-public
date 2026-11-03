"""Escape-reflex TTMn selection for the ``full_mcns_escape_v1`` profile.

This module is the anatomy half of the escape-reflex Engineered-Encoder
profile (ADR 0008 Mode 3, identity recorded in
``experiments/preregistration/eem-escape-v1.md`` -- **never** Faithful
Organism Mode): the sensory boundary is exactly the eem v1 absolute encoder
over the pinned labeled populations, and the declared addition is one more
motor-stage readout -- the TTMn jump-motoneuron pair -- selected from the
same digest-verified body-annotations join the embodied-senses populations
use, plus a declared jump mapping onto the vertical thrust axis.

Census anatomy (D-050-era readout census, recorded in D-060)
------------------------------------------------------------
All 126 LC4 looming detectors project to the DNp01 giant-fiber pair, which
projects to the TTMn jump motoneurons. The pinned body-annotations table
carries exactly two rows with ``type == 'TTMn'``: bodies 800146 (somaSide R)
and 804642 (somaSide L), class ``vnc_motor`` / subclass ``wm``. Neither body
is a member of the declared 52-neuron wing-motor readout. DNp01 itself is
never read directly: the declared readout is the motor stage, and the giant
fiber stays interior recurrent context.

Fail-closed selection
---------------------
The selector requires EXACTLY two TTMn rows split exactly 1 L / 1 R, refuses
any count or side drift, refuses membership overlap with the 52 declared
wing-motor bodies, with every embodied sensory population, and with the two
faithful sensory entries. Membership is frozen into a hashed pool-binding
document that the escape decoder identity chains.

Claim discipline
----------------
"Jump" names a declared boundary mapping from jump-motoneuron pool activity
to Minecraft vertical thrust -- an engineering convention, hashed before any
measurement -- not a claim that the organism decided, intended, or performed
anything. Nothing here reads DNp01, alters the dynamics, or touches the wing
readout; this module selects two bodies and writes identity documents.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from .config import canonical_sha256
from .embodied_senses import (
    _MISSING_ANNOTATION_ROW,
    SIDE_COLUMN,
    TYPE_COLUMN,
    EmbodiedPopulationBinding,
    EmbodiedSensesError,
    _object_array,
    _require_sha256,
    _string_column,
    _verified_annotation_table,
    _verified_graph_array,
    faithful_entry_body_ids,
)
from .flight_pools import LEFT_WING_MOTOR_BODY_IDS, RIGHT_WING_MOTOR_BODY_IDS

if TYPE_CHECKING:
    from neurocraft_connectome.full_runtime import FullRuntimeGraph
    from numpy.typing import NDArray

ESCAPE_TTMN_POOL_SCHEMA: Final = "neurocraft.escape.ttmn_pool_binding.v1"

#: Declared selector and fail-closed expectations, verified against the
#: pinned body-annotations table (census: bodies 800146 R / 804642 L).
TTMN_TYPE: Final = "TTMn"
TTMN_SELECTOR: Final = f"{TYPE_COLUMN} == '{TTMN_TYPE}'"
EXPECTED_TTMN_NODES: Final = 2
EXPECTED_TTMN_LEFT_NODES: Final = 1
EXPECTED_TTMN_RIGHT_NODES: Final = 1

#: The declared jump mapping, hashed verbatim into the escape decoder
#: identity. One additive term on one axis; every other axis is untouched.
ESCAPE_JUMP_FORMULA: Final = (
    "jump_raw = motor_pool_gain * (fsum(TTMn activity) / 2) * jump_scale; "
    "jump_value = clip(jump_raw, -1, 1), zeroed when |clip(jump_raw, -1, 1)| < "
    "motor_dead_zone; vertical_thrust = clip(vertical_projection + jump_value, "
    "-1, 1); forward, lateral, yaw, and pitch are untouched"
)

#: The escape-v2 jump mapping (identity recorded in
#: ``experiments/preregistration/eem-escape-v2.md``), hashed verbatim into the
#: escape-v2 decoder identity: identical pooled readout, with the clip range
#: rectified to the non-negative half of the declared codomain so the mapping
#: matches the upward-thrust body policy ``eem-escape-v1.md`` already declared.
#: Negative pooled TTMn activity maps to zero -- no anti-jump. D-062 measured
#: collision driving the v1 formula to the -1.0 rail, a downward slam that
#: contradicted the declared takeoff semantics; the rectification is a
#: declared body-model convention, frozen before any measurement under it.
ESCAPE_JUMP_FORMULA_V2: Final = (
    "jump_raw = motor_pool_gain * (fsum(TTMn activity) / 2) * jump_scale; "
    "jump_value = clip(jump_raw, 0, 1), zeroed when clip(jump_raw, 0, 1) < "
    "motor_dead_zone; vertical_thrust = clip(vertical_projection + jump_value, "
    "-1, 1); forward, lateral, yaw, and pitch are untouched"
)


class EscapeReflexError(RuntimeError):
    """The escape-reflex binding could not be built exactly as declared."""


@dataclass(frozen=True, slots=True)
class TtmnMember:
    """One selected TTMn jump motoneuron with its dense index and soma side."""

    body_id: int
    dense_index: int
    soma_side: str


@dataclass(frozen=True, slots=True)
class TtmnPoolBinding:
    """The canonical TTMn pool-membership document, hashed into identities."""

    graph_manifest_sha256: str
    annotations_sha256: str
    node_count: int
    members: tuple[TtmnMember, ...]

    def __post_init__(self) -> None:
        if len(self.members) != EXPECTED_TTMN_NODES:
            raise EscapeReflexError(
                f"the TTMn pool binding must carry exactly {EXPECTED_TTMN_NODES} members"
            )
        pairs = zip(self.members, self.members[1:], strict=False)
        if any(later.body_id <= earlier.body_id for earlier, later in pairs):
            raise EscapeReflexError("TTMn pool member body IDs must ascend strictly")

    def to_identity_material(self) -> dict[str, Any]:
        return {
            "annotations_sha256": self.annotations_sha256,
            "expected": {
                "left": EXPECTED_TTMN_LEFT_NODES,
                "right": EXPECTED_TTMN_RIGHT_NODES,
                "total": EXPECTED_TTMN_NODES,
            },
            "graph_manifest_sha256": self.graph_manifest_sha256,
            "members": [
                [member.body_id, member.dense_index, member.soma_side] for member in self.members
            ],
            "node_count": self.node_count,
            "schema": ESCAPE_TTMN_POOL_SCHEMA,
            "selector": TTMN_SELECTOR,
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.to_identity_material())


def _refuse_ttmn_overlaps(
    members: Sequence[TtmnMember],
    *,
    populations: EmbodiedPopulationBinding,
    wing_body_ids: Collection[int],
    faithful_bodies: Collection[int],
) -> None:
    """Refuse any TTMn capture of the wing readout, a population, or an entry."""

    member_ids = {member.body_id for member in members}
    wing_overlap = sorted(member_ids.intersection(wing_body_ids))
    if wing_overlap:
        raise EscapeReflexError(
            f"TTMn pool bodies {wing_overlap} are members of the declared 52-neuron "
            "wing-motor readout; the jump pool must stay disjoint from the wing pools"
        )
    faithful_overlap = sorted(member_ids.intersection(faithful_bodies))
    if faithful_overlap:
        raise EscapeReflexError(
            f"TTMn pool bodies {faithful_overlap} are faithful sensory entry bodies; "
            "the jump pool must stay disjoint from the faithful channels"
        )
    for population in populations.populations:
        overlap = sorted(member_ids.intersection(population.body_ids))
        if overlap:
            raise EscapeReflexError(
                f"TTMn pool bodies {overlap} are members of the sensory population "
                f"{population.name!r}; the jump pool must stay disjoint from every "
                "driven population"
            )


def select_ttmn_pool(
    *,
    type_column: Sequence[str | None],
    soma_side_column: Sequence[str | None],
    body_ids: Sequence[int],
    index_of: Callable[[int], int],
    node_count: int,
    graph_manifest_sha256: str,
    annotations_sha256: str,
    populations: EmbodiedPopulationBinding,
    wing_body_ids: Collection[int] | None = None,
    faithful_bodies: Collection[int] | None = None,
) -> TtmnPoolBinding:
    """Select the declared TTMn pair from dense-ordered annotation columns.

    Pure over its inputs so the CI tests can drive it with a synthetic
    annotation frame; :func:`load_escape_ttmn_binding` supplies the verified
    real columns. Every count, side, and overlap gate fails closed.
    """

    _require_sha256(graph_manifest_sha256, "graph_manifest_sha256")
    _require_sha256(annotations_sha256, "annotations_sha256")
    if populations.graph_manifest_sha256 != graph_manifest_sha256:
        raise EscapeReflexError(
            "the supplied population binding was derived from a different artifact"
        )
    if populations.annotations_sha256 != annotations_sha256:
        raise EscapeReflexError(
            "the supplied population binding was derived from a different annotations table"
        )
    if populations.node_count != node_count:
        raise EscapeReflexError(
            "the supplied population binding node count does not match this selection"
        )
    row_count = len(body_ids)
    if row_count == 0 or row_count > node_count:
        raise EscapeReflexError("annotation rows must be nonempty and within the node count")
    try:
        types = _object_array(type_column, TYPE_COLUMN, row_count)
        sides = _object_array(soma_side_column, SIDE_COLUMN, row_count)
    except EmbodiedSensesError as exc:
        raise EscapeReflexError(str(exc)) from exc
    mask: NDArray[np.bool_] = np.asarray(types == TTMN_TYPE, dtype=bool)
    observed = int(mask.sum())
    if observed != EXPECTED_TTMN_NODES:
        raise EscapeReflexError(
            f"TTMn count drifted: selected {observed} nodes, the declared selector "
            f"expects exactly {EXPECTED_TTMN_NODES}; refusing to bind"
        )
    ttmn_sides = {value for value in sides[mask].tolist()}
    if not ttmn_sides.issubset({"L", "R"}):
        raise EscapeReflexError(
            f"TTMn somaSide values "
            f"{sorted(str(value) for value in ttmn_sides - {'L', 'R'})} are outside "
            "the declared L/R split; refusing to bind"
        )
    left = int((mask & np.asarray(sides == "L", dtype=bool)).sum())
    right = int((mask & np.asarray(sides == "R", dtype=bool)).sum())
    if left != EXPECTED_TTMN_LEFT_NODES or right != EXPECTED_TTMN_RIGHT_NODES:
        raise EscapeReflexError(
            f"TTMn side split drifted: selected {left} L / {right} R, the declared "
            f"selector expects exactly {EXPECTED_TTMN_LEFT_NODES} L / "
            f"{EXPECTED_TTMN_RIGHT_NODES} R; refusing to bind"
        )
    members: list[TtmnMember] = []
    for position in np.flatnonzero(mask).tolist():
        body_id = int(body_ids[position])
        try:
            dense_index = index_of(body_id)
        except KeyError as exc:
            raise EscapeReflexError(f"TTMn body {body_id} is not present in the artifact") from exc
        if dense_index != position:
            raise EscapeReflexError(
                f"TTMn body {body_id} resolves to dense index {dense_index} but sits "
                f"at annotation row {position}; the annotation rows are not in dense "
                "body order"
            )
        if not 0 <= dense_index < node_count:
            raise EscapeReflexError(f"TTMn dense index {dense_index} is out of range")
        members.append(
            TtmnMember(body_id=body_id, dense_index=dense_index, soma_side=str(sides[position]))
        )
    members.sort(key=lambda member: member.body_id)
    _refuse_ttmn_overlaps(
        members,
        populations=populations,
        wing_body_ids=(
            frozenset(LEFT_WING_MOTOR_BODY_IDS + RIGHT_WING_MOTOR_BODY_IDS)
            if wing_body_ids is None
            else wing_body_ids
        ),
        faithful_bodies=(faithful_entry_body_ids() if faithful_bodies is None else faithful_bodies),
    )
    return TtmnPoolBinding(
        graph_manifest_sha256=graph_manifest_sha256,
        annotations_sha256=annotations_sha256,
        node_count=node_count,
        members=tuple(members),
    )


def load_escape_ttmn_binding(
    graph: FullRuntimeGraph,
    body_annotations_path: Path,
    populations: EmbodiedPopulationBinding,
) -> TtmnPoolBinding:
    """Join the pinned body-annotations table and select the declared TTMn pair.

    The join discipline is byte-for-byte the one
    :func:`neurocraft_controller.embodied_senses.load_embodied_population_binding`
    executes (digest match, complete coverage, unique in-range pointers,
    bitwise ``bodyId`` identity); it is re-run here rather than trusted from
    the population load so the TTMn selection fails closed on its own.
    """

    table, annotations_sha256 = _verified_annotation_table(graph, body_annotations_path)
    annotation_rows = _verified_graph_array(graph, "annotation_source_rows", "<u8")
    body_ids = _verified_graph_array(graph, "body_ids", "<u8")
    if bool((annotation_rows == np.uint64(_MISSING_ANNOTATION_ROW)).any()):
        raise EscapeReflexError(
            "a retained node has no annotation row; the TTMn selector requires "
            "complete annotation coverage"
        )
    rows = annotation_rows.astype(np.int64)
    if int(rows.min()) < 0 or int(rows.max()) >= int(table.num_rows):
        raise EscapeReflexError("annotation source row is out of table range")
    if np.unique(rows).size != rows.size:
        raise EscapeReflexError("annotation source rows are not unique")
    taken = table.take(rows)
    taken_bodies = np.asarray(taken.column("bodyId").to_pylist(), dtype=np.uint64)
    if not np.array_equal(taken_bodies, body_ids):
        raise EscapeReflexError(
            "bodyId column taken through annotation_source_rows is not bitwise "
            "identical to the artifact body IDs; refusing a misaligned join"
        )
    return select_ttmn_pool(
        type_column=_string_column(taken, TYPE_COLUMN),
        soma_side_column=_string_column(taken, SIDE_COLUMN),
        body_ids=body_ids.tolist(),
        index_of=graph.index_of,
        node_count=graph.manifest.node_count,
        graph_manifest_sha256=graph.manifest.manifest_file_sha256,
        annotations_sha256=annotations_sha256,
        populations=populations,
    )
