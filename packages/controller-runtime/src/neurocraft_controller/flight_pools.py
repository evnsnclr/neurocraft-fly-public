"""Motor pools for the 128-node ``mcns-v1.0-flight-steering-v1`` circuit.

The shipped 14-node circuit reads out through one bilateral pair, ``hg1 MN``
(bodies 800241 and 800917), and projects it as common mode to thrust and
differential mode to yaw. The flight-steering fixture does not contain 800241 at
all, so that decoder cannot be built on it; it carries 52 wing motor neurons
instead.

This module widens each side of the existing pair into every wing motor neuron on
that side, and changes nothing else. The projection in
``ConnectomeController._project_motor_frame`` is untouched: two pools named
``left_motor`` and ``right_motor``, common mode to thrust, differential mode to
yaw. Scale is preserved because ``MotorDecoder`` divides by the summed absolute
member weight, so unit weights make each pool the mean activity of its side —
exactly what a single unit-weight member computed before.

**The split is the dataset's own ``somaSide`` annotation and nothing else.** No
neuron is assigned to a Minecraft axis by function, and no muscle class is
treated as steering or power here, because the connectome does not say that. The
52 neurons include both power (``DLMn``, ``DVMn``) and steering (``i1``, ``i2``,
``ps1``, ``hg3``, ``hg4``, ``tp2``, ``b2``, ``MNwm36``) motor neurons, and
splitting them by role would require a literature-derived functional claim of the
kind ``DECISIONS.md`` D-035 declined for descending neurons. Anyone wanting that
mapping should preregister it as a separate, named decoder.

Membership is declared here and re-derived from the graph at construction. A
mismatch is a hard failure: a fixture that no longer yields exactly this set must
not silently produce a different readout under the same pool name.
"""

from __future__ import annotations

from neurocraft_connectome import RuntimeGraph
from neurocraft_neural.adapters import MotorPool, PoolMember

WING_MOTOR_POOL_PROFILE = "wing_motor_soma_side_v1"

WING_MOTOR_POOL_RULE = (
    "every node carrying the fixture's motor role, partitioned by the pinned "
    "annotation table's somaSide value, with unit weight per member"
)

LEFT_WING_MOTOR_BODY_IDS: tuple[int, ...] = (
    800184,
    800381,
    800412,
    800718,
    800847,
    800890,
    800928,
    801021,
    801137,
    801185,
    801220,
    801292,
    801895,
    801949,
    801970,
    802315,
    803013,
    803589,
    803982,
    804131,
    805165,
    807799,
    906612,
    932322,
    932796,
)

RIGHT_WING_MOTOR_BODY_IDS: tuple[int, ...] = (
    800056,
    800573,
    800637,
    800696,
    800743,
    800899,
    800906,
    800917,
    801295,
    801350,
    801815,
    801816,
    801998,
    802028,
    802120,
    802544,
    802837,
    803048,
    803129,
    803599,
    803891,
    804301,
    807924,
    903852,
    932321,
    1050014552,
    1050045395,
)


class FlightPoolError(RuntimeError):
    """The declared wing-motor membership does not match the supplied graph."""


def _motor_body_ids_by_side(graph: RuntimeGraph) -> dict[str, tuple[int, ...]]:
    sides: dict[str, list[int]] = {}
    for node in graph.nodes:
        if "motor" not in node.roles:
            continue
        if node.side is None:
            raise FlightPoolError(
                f"wing motor body {node.body_id} carries no somaSide annotation, so it "
                "cannot be assigned to a side without inventing one"
            )
        sides.setdefault(node.side, []).append(node.body_id)
    return {side: tuple(sorted(ids)) for side, ids in sides.items()}


def derive_wing_motor_pools(
    graph: RuntimeGraph, *, gain: float, dead_zone: float
) -> tuple[MotorPool, ...]:
    """Build the two side pools, refusing any graph that is not the declared circuit."""

    observed = _motor_body_ids_by_side(graph)
    unexpected = set(observed) - {"L", "R"}
    if unexpected:
        raise FlightPoolError(
            f"unexpected soma sides among wing motor neurons: {sorted(unexpected)}"
        )
    for side, declared in (("L", LEFT_WING_MOTOR_BODY_IDS), ("R", RIGHT_WING_MOTOR_BODY_IDS)):
        found = observed.get(side, ())
        if found != declared:
            missing = sorted(set(declared) - set(found))
            extra = sorted(set(found) - set(declared))
            raise FlightPoolError(
                f"soma side {side} wing motor membership does not match the declared set: "
                f"{len(found)} found against {len(declared)} declared; "
                f"missing={missing}; unexpected={extra}"
            )
    return (
        MotorPool(
            "left_motor",
            tuple(PoolMember(body_id, 1.0) for body_id in LEFT_WING_MOTOR_BODY_IDS),
            gain=gain,
            dead_zone=dead_zone,
        ),
        MotorPool(
            "right_motor",
            tuple(PoolMember(body_id, 1.0) for body_id in RIGHT_WING_MOTOR_BODY_IDS),
            gain=gain,
            dead_zone=dead_zone,
        ),
    )
