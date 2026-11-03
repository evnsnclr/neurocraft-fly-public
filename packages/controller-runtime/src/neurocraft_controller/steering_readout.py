"""DNa01 steering-pair selection for the readout-to-program selection policy.

This module is the anatomy half of one hand-chosen engineering mapping. It selects two
labeled bodies and writes an identity document; it decides nothing, drives nothing, and
adds no term to any motor axis. The policy that reads it
(:mod:`neurocraft_controller.program_policy`) is declared engineering under ADR 0019 --
labeled readouts selecting scripted body programs -- and every sentence about it says so.

Census anatomy (``docs/research/readout-census-2026-09-02.md``)
--------------------------------------------------------------
The pinned body-annotations table carries exactly two rows with ``type == 'DNa01'``:
bodies 10442 and 10760, one ``somaSide`` L and one R, class ``descending_neuron``. The
census records the pair as strictly sided, disjoint from every runtime population, from
the declared 52-body wing readout, from the TTMn jump pair, and from the two faithful
sensory entries (census candidate table and overlap column).

Taxonomy: this reads a DESCENDING neuron, not a motor stage
-----------------------------------------------------------
``escape_reflex.py`` declares, for the jump readout, that "the declared readout is the
motor stage, and the giant fiber stays interior recurrent context". Binding DNa01
departs from that declaration on purpose, and the census says so in as many words: it
"is a taxonomy change and needs its own decision, not just a new selector". The decision
is recorded as a ``DECISIONS.md`` row alongside this module. The alternatives considered
were the motor-stage ones: the wing pools' own left/right difference (the pinned
projection's yaw, measured at most 0.036 in the one observed live session) and
``cb_neck_motor_pool`` (20 bodies, 10 L / 10 R, a genuine motor stage).

**The honest reason cb_neck was not chosen, stated correctly.** It is *not* that its sign
is inconsistent -- an earlier draft of this docstring said that and it was wrong. Under the
escape-v2 census its sided means are all positive on both lateral conditions, and the
declared turn formula reads them as +0.2238 at ``blocked_left`` and -0.1565 at
``blocked_right``: the same "away from the blocked side" direction DNa01 gives. The two
real reasons are **magnitude** -- DNa01 gives +0.9418 / -0.9244 on the same rows, about
four times larger, and the 0.10 deadband would swallow far more of cb_neck's range (its
own baseline row already sits at -0.0746, inside that deadband) -- and that **neither pool
has been measured under the loom-v1 encoder**, which is the encoder this profile runs. A
larger separation from the deadband under a shared, unmeasured encoder is the whole of the
argument; it is a preference between two candidates, not a defect found in one.

Fail-closed selection
---------------------
The selector requires EXACTLY two ``DNa01`` rows split exactly 1 L / 1 R, refuses any
count or side drift, and refuses membership overlap with the declared 52 wing-motor
bodies, with the TTMn jump pair, with every embodied sensory population, and with the two
faithful sensory entries. Membership is frozen into a hashed binding document whose
identity the selection policy's ``policy_hash`` chains -- and which enters no
``RunIdentity`` field, so binding it rotates no adapter, decoder, or graph identity.

Claim discipline
----------------
Nothing here measures, establishes, or implies that DNa01 steers a fly, that its left and
right activity difference means "turn", or that the sign convention the policy declares
matches any organism. It selects two bodies by label and hashes the selection.
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
from .escape_reflex import TtmnPoolBinding
from .flight_pools import LEFT_WING_MOTOR_BODY_IDS, RIGHT_WING_MOTOR_BODY_IDS

if TYPE_CHECKING:
    from neurocraft_connectome.full_runtime import FullRuntimeGraph
    from numpy.typing import NDArray

STEERING_DNA01_PAIR_SCHEMA: Final = "neurocraft.steering.dna01_pair_binding.v1"

#: Declared selector and fail-closed expectations, verified against the pinned
#: body-annotations table (census: bodies 10442 and 10760, one L and one R).
DNA01_TYPE: Final = "DNa01"
DNA01_SELECTOR: Final = f"{TYPE_COLUMN} == '{DNA01_TYPE}'"
EXPECTED_DNA01_NODES: Final = 2
EXPECTED_DNA01_LEFT_NODES: Final = 1
EXPECTED_DNA01_RIGHT_NODES: Final = 1


class SteeringReadoutError(RuntimeError):
    """The DNa01 steering binding could not be built exactly as declared."""


@dataclass(frozen=True, slots=True)
class SteeringMember:
    """One selected DNa01 descending neuron with its dense index and soma side."""

    body_id: int
    dense_index: int
    soma_side: str


@dataclass(frozen=True, slots=True)
class SteeringPairBinding:
    """The canonical DNa01 pair-membership document, hashed into ``policy_hash``.

    Deliberately NOT hashed into ``RunIdentity``: the eight identity fields
    (``graph_hash``, ``adapter_hash``, ``dynamics_hash``, ``decoder_hash``,
    ``observation_schema_hash`` and the three run labels) describe the sensory boundary,
    the dynamics, and the motor decoder. This binding is read by a selection policy that
    adds no term to any motor axis, so putting it in the decoder identity would rotate
    every calibration cross-binding for a readout the decoder never sees.
    """

    graph_manifest_sha256: str
    annotations_sha256: str
    node_count: int
    members: tuple[SteeringMember, ...]

    def __post_init__(self) -> None:
        if len(self.members) != EXPECTED_DNA01_NODES:
            raise SteeringReadoutError(
                f"the DNa01 pair binding must carry exactly {EXPECTED_DNA01_NODES} members"
            )
        pairs = zip(self.members, self.members[1:], strict=False)
        if any(later.body_id <= earlier.body_id for earlier, later in pairs):
            raise SteeringReadoutError("DNa01 pair member body IDs must ascend strictly")
        sides = [member.soma_side for member in self.members]
        if sorted(sides) != ["L", "R"]:
            raise SteeringReadoutError("the DNa01 pair binding must carry exactly one L and one R")

    @property
    def left(self) -> SteeringMember:
        """The L-side member; the pair is pinned to exactly one."""

        return next(member for member in self.members if member.soma_side == "L")

    @property
    def right(self) -> SteeringMember:
        """The R-side member; the pair is pinned to exactly one."""

        return next(member for member in self.members if member.soma_side == "R")

    def to_identity_material(self) -> dict[str, Any]:
        return {
            "annotations_sha256": self.annotations_sha256,
            "expected": {
                "left": EXPECTED_DNA01_LEFT_NODES,
                "right": EXPECTED_DNA01_RIGHT_NODES,
                "total": EXPECTED_DNA01_NODES,
            },
            "graph_manifest_sha256": self.graph_manifest_sha256,
            "members": [
                [member.body_id, member.dense_index, member.soma_side] for member in self.members
            ],
            "node_count": self.node_count,
            "schema": STEERING_DNA01_PAIR_SCHEMA,
            "selector": DNA01_SELECTOR,
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.to_identity_material())


def _refuse_dna01_overlaps(
    members: Sequence[SteeringMember],
    *,
    populations: EmbodiedPopulationBinding,
    wing_body_ids: Collection[int],
    ttmn_body_ids: Collection[int],
    faithful_bodies: Collection[int],
) -> None:
    """Refuse any DNa01 capture of the wing readout, the TTMn pair, a population, or an entry."""

    member_ids = {member.body_id for member in members}
    wing_overlap = sorted(member_ids.intersection(wing_body_ids))
    if wing_overlap:
        raise SteeringReadoutError(
            f"DNa01 pair bodies {wing_overlap} are members of the declared 52-neuron "
            "wing-motor readout; the steering pair must stay disjoint from the wing pools"
        )
    ttmn_overlap = sorted(member_ids.intersection(ttmn_body_ids))
    if ttmn_overlap:
        raise SteeringReadoutError(
            f"DNa01 pair bodies {ttmn_overlap} are members of the declared TTMn jump "
            "pool; the steering pair must stay disjoint from the jump readout"
        )
    faithful_overlap = sorted(member_ids.intersection(faithful_bodies))
    if faithful_overlap:
        raise SteeringReadoutError(
            f"DNa01 pair bodies {faithful_overlap} are faithful sensory entry bodies; "
            "the steering pair must stay disjoint from the faithful channels"
        )
    for population in populations.populations:
        overlap = sorted(member_ids.intersection(population.body_ids))
        if overlap:
            raise SteeringReadoutError(
                f"DNa01 pair bodies {overlap} are members of the sensory population "
                f"{population.name!r}; the steering pair must stay disjoint from every "
                "driven population"
            )


def select_dna01_pair(
    *,
    type_column: Sequence[str | None],
    soma_side_column: Sequence[str | None],
    body_ids: Sequence[int],
    index_of: Callable[[int], int],
    node_count: int,
    graph_manifest_sha256: str,
    annotations_sha256: str,
    populations: EmbodiedPopulationBinding,
    ttmn_pool: TtmnPoolBinding,
    wing_body_ids: Collection[int] | None = None,
    faithful_bodies: Collection[int] | None = None,
) -> SteeringPairBinding:
    """Select the declared DNa01 pair from dense-ordered annotation columns.

    Pure over its inputs so the CI tests can drive it with a synthetic annotation frame;
    :func:`load_steering_binding` supplies the verified real columns. Every count, side,
    and overlap gate fails closed, exactly as ``escape_reflex.select_ttmn_pool`` does.
    """

    _require_sha256(graph_manifest_sha256, "graph_manifest_sha256")
    _require_sha256(annotations_sha256, "annotations_sha256")
    if populations.graph_manifest_sha256 != graph_manifest_sha256:
        raise SteeringReadoutError(
            "the supplied population binding was derived from a different artifact"
        )
    if populations.annotations_sha256 != annotations_sha256:
        raise SteeringReadoutError(
            "the supplied population binding was derived from a different annotations table"
        )
    if populations.node_count != node_count:
        raise SteeringReadoutError(
            "the supplied population binding node count does not match this selection"
        )
    if ttmn_pool.graph_manifest_sha256 != graph_manifest_sha256:
        raise SteeringReadoutError(
            "the supplied TTMn pool binding was derived from a different artifact"
        )
    if ttmn_pool.annotations_sha256 != annotations_sha256:
        raise SteeringReadoutError(
            "the supplied TTMn pool binding was derived from a different annotations table"
        )
    row_count = len(body_ids)
    if row_count == 0 or row_count > node_count:
        raise SteeringReadoutError("annotation rows must be nonempty and within the node count")
    try:
        types = _object_array(type_column, TYPE_COLUMN, row_count)
        sides = _object_array(soma_side_column, SIDE_COLUMN, row_count)
    except EmbodiedSensesError as exc:
        raise SteeringReadoutError(str(exc)) from exc
    mask: NDArray[np.bool_] = np.asarray(types == DNA01_TYPE, dtype=bool)
    observed = int(mask.sum())
    if observed != EXPECTED_DNA01_NODES:
        raise SteeringReadoutError(
            f"DNa01 count drifted: selected {observed} nodes, the declared selector "
            f"expects exactly {EXPECTED_DNA01_NODES}; refusing to bind"
        )
    pair_sides = {value for value in sides[mask].tolist()}
    if not pair_sides.issubset({"L", "R"}):
        raise SteeringReadoutError(
            f"DNa01 somaSide values "
            f"{sorted(str(value) for value in pair_sides - {'L', 'R'})} are outside "
            "the declared L/R split; refusing to bind"
        )
    left = int((mask & np.asarray(sides == "L", dtype=bool)).sum())
    right = int((mask & np.asarray(sides == "R", dtype=bool)).sum())
    if left != EXPECTED_DNA01_LEFT_NODES or right != EXPECTED_DNA01_RIGHT_NODES:
        raise SteeringReadoutError(
            f"DNa01 side split drifted: selected {left} L / {right} R, the declared "
            f"selector expects exactly {EXPECTED_DNA01_LEFT_NODES} L / "
            f"{EXPECTED_DNA01_RIGHT_NODES} R; refusing to bind"
        )
    members: list[SteeringMember] = []
    for position in np.flatnonzero(mask).tolist():
        body_id = int(body_ids[position])
        try:
            dense_index = index_of(body_id)
        except KeyError as exc:
            raise SteeringReadoutError(
                f"DNa01 body {body_id} is not present in the artifact"
            ) from exc
        if dense_index != position:
            raise SteeringReadoutError(
                f"DNa01 body {body_id} resolves to dense index {dense_index} but sits "
                f"at annotation row {position}; the annotation rows are not in dense "
                "body order"
            )
        if not 0 <= dense_index < node_count:
            raise SteeringReadoutError(f"DNa01 dense index {dense_index} is out of range")
        members.append(
            SteeringMember(body_id=body_id, dense_index=dense_index, soma_side=str(sides[position]))
        )
    members.sort(key=lambda member: member.body_id)
    _refuse_dna01_overlaps(
        members,
        populations=populations,
        wing_body_ids=(
            frozenset(LEFT_WING_MOTOR_BODY_IDS + RIGHT_WING_MOTOR_BODY_IDS)
            if wing_body_ids is None
            else wing_body_ids
        ),
        ttmn_body_ids=frozenset(member.body_id for member in ttmn_pool.members),
        faithful_bodies=(faithful_entry_body_ids() if faithful_bodies is None else faithful_bodies),
    )
    return SteeringPairBinding(
        graph_manifest_sha256=graph_manifest_sha256,
        annotations_sha256=annotations_sha256,
        node_count=node_count,
        members=tuple(members),
    )


def load_steering_binding(
    graph: FullRuntimeGraph,
    body_annotations_path: Path,
    populations: EmbodiedPopulationBinding,
    ttmn_pool: TtmnPoolBinding,
) -> SteeringPairBinding:
    """Join the pinned body-annotations table and select the declared DNa01 pair.

    The join discipline is byte-for-byte the one
    :func:`neurocraft_controller.embodied_senses.load_embodied_population_binding` executes
    (digest match, complete coverage, unique in-range pointers, bitwise ``bodyId``
    identity); it is re-run here rather than trusted from the population or TTMn load so
    the DNa01 selection fails closed on its own.
    """

    table, annotations_sha256 = _verified_annotation_table(graph, body_annotations_path)
    annotation_rows = _verified_graph_array(graph, "annotation_source_rows", "<u8")
    body_ids = _verified_graph_array(graph, "body_ids", "<u8")
    if bool((annotation_rows == np.uint64(_MISSING_ANNOTATION_ROW)).any()):
        raise SteeringReadoutError(
            "a retained node has no annotation row; the DNa01 selector requires "
            "complete annotation coverage"
        )
    rows = annotation_rows.astype(np.int64)
    if int(rows.min()) < 0 or int(rows.max()) >= int(table.num_rows):
        raise SteeringReadoutError("annotation source row is out of table range")
    if np.unique(rows).size != rows.size:
        raise SteeringReadoutError("annotation source rows are not unique")
    taken = table.take(rows)
    taken_bodies = np.asarray(taken.column("bodyId").to_pylist(), dtype=np.uint64)
    if not np.array_equal(taken_bodies, body_ids):
        raise SteeringReadoutError(
            "bodyId column taken through annotation_source_rows is not bitwise "
            "identical to the artifact body IDs; refusing a misaligned join"
        )
    return select_dna01_pair(
        type_column=_string_column(taken, TYPE_COLUMN),
        soma_side_column=_string_column(taken, SIDE_COLUMN),
        body_ids=body_ids.tolist(),
        index_of=graph.index_of,
        node_count=graph.manifest.node_count,
        graph_manifest_sha256=graph.manifest.manifest_file_sha256,
        annotations_sha256=annotations_sha256,
        populations=populations,
        ttmn_pool=ttmn_pool,
    )
