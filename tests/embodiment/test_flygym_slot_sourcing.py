"""Per-slot sourcing for the FlyGym body: what moves, what does not, and both claims.

``FLYGYM_SENSORY_CHANNELS`` states, for each of the twenty scalars a protocol
``SensorFrame`` carries, whether this body's ``observe`` varies it. Twelve are
declared dynamic and eight are not, and that split is hashed into the sensory
boundary identity -- it is the number a reviewer reads to learn how much of this
body's frame the connectome can actually see.

Both directions of that claim are checked here, because only one used to be. See
:func:`_slot_sample_states` for why the sample set had to be rebuilt before either
check could mean much.

Nothing in this file runs FlyGym. The body under test is a kinematic stub that
integrates no equation of motion; flygym is not installed and has never been executed.
"""

from __future__ import annotations

from neurocraft_connectome import RuntimeGraph
from neurocraft_controller import ConnectomeController
from neurocraft_embodiment import FlyGymEmbodiment
from neurocraft_embodiment.seam import sensor_frame_widths

from .test_flygym_body import _command, _kinematic_body


def _sensor_scalars(
    body: FlyGymEmbodiment,
    controller: ConnectomeController,
    sequence: int,
) -> tuple[float, ...]:
    """One observation flattened to the twenty protocol scalars, in wire order."""

    payload = body.observe(identity=controller.identity, sequence=sequence).sensors.to_dict()
    flat: list[float] = []
    for name, width in sensor_frame_widths():
        value = payload[name]
        flat.extend(float(item) for item in value) if width > 1 else flat.append(float(value))
    return tuple(flat)


def _slot_sample_states(
    graph: RuntimeGraph,
    controller: ConnectomeController,
) -> list[tuple[float, ...]]:
    """Sample frames spanning every reading this body's ``observe`` can vary.

    The previous sample set was three frames from two bodies that differed only in leg
    contact and in MuJoCo +x. Nine of the twelve slots the boundary declares *dynamic*
    were therefore identical across every sample -- both remaining velocity
    components, both angles, the height-derived light level, the down ray and two
    position components -- so the "declared constant never moves" check was passing
    over slots it never moved, and the converse ("declared dynamic really can move")
    could not be stated at all. The stub's pose is scriptable now, and each body below
    exists to move a named group of slots.
    """

    samples: list[tuple[float, ...]] = []

    # Leg contact: on_ground and vertical_collision.
    resting = _kinematic_body(
        graph,
        contact_found=(0.3,) * 6,
        contact_forces=((0.0, 0.0, 0.4),) * 6,
    )
    samples.append(_sensor_scalars(resting, controller, 0))

    # Out of contact, and after a tick: position_m[0] and velocity_m_s[0].
    airborne = _kinematic_body(graph)
    samples.append(_sensor_scalars(airborne, controller, 0))
    airborne.apply(_command(controller, 0))
    samples.append(_sensor_scalars(airborne, controller, 1))

    # A different height and lateral offset: position_m[1], position_m[2],
    # local_light and ray_distances_m[5], none of which any earlier sample moved.
    lifted = _kinematic_body(graph, spawn_position_mm=(0.0, 1.0, 2.0))
    samples.append(_sensor_scalars(lifted, controller, 0))

    # Motion in all three MuJoCo axes: velocity_m_s[1] and velocity_m_s[2].
    drifting = _kinematic_body(graph, step_delta_mm=(0.01, 0.02, 0.03))
    _sensor_scalars(drifting, controller, 0)
    drifting.apply(_command(controller, 0))
    samples.append(_sensor_scalars(drifting, controller, 1))

    # A rotated thorax: yaw_rad and pitch_rad.
    turned = _kinematic_body(graph, quaternion=(0.9, 0.1, 0.3, 0.3))
    samples.append(_sensor_scalars(turned, controller, 0))
    return samples


def test_every_declared_constant_slot_really_is_constant(
    graph: RuntimeGraph,
    controller: ConnectomeController,
) -> None:
    """MEDIUM-3/MEDIUM-5: the per-slot table is a claim about ``observe``'s output.

    A declaration that no test compares against the frame is a comment with a hash
    around it.
    """

    from neurocraft_embodiment import CHANNEL_SOURCE_DYNAMIC, FLYGYM_SENSORY_CHANNELS

    samples = _slot_sample_states(graph, controller)
    moved = {
        index
        for index, _entry in enumerate(FLYGYM_SENSORY_CHANNELS)
        if len({sample[index] for sample in samples}) > 1
    }
    constant_but_moved = [
        FLYGYM_SENSORY_CHANNELS[index].name
        for index in moved
        if FLYGYM_SENSORY_CHANNELS[index].source != CHANNEL_SOURCE_DYNAMIC
    ]
    assert constant_but_moved == []
    assert moved, "no slot moved at all; the sample states were not distinct"


def test_every_declared_dynamic_slot_really_can_move(
    graph: RuntimeGraph,
    controller: ConnectomeController,
) -> None:
    """The converse, which nothing asserted: ``dynamic`` is a claim too.

    ``declared_dynamic_channels`` is the number a reviewer reads off the identity to
    learn how much of this body's frame the connectome can actually see. One direction
    of that claim was checked -- a slot called constant must not move -- and the other
    was not, so a slot could be advertised as varying while being as pinned as the
    five saturated rays beside it, and the tally hashed into the sensory boundary would
    have overstated the input the graph receives.

    "Can move" is the honest reading of ``dynamic``: the sample states here are
    constructed to exercise each source, not sampled from a run, so this says the slot
    is wired to a reading rather than that any particular run moves it.
    """

    from neurocraft_embodiment import CHANNEL_SOURCE_DYNAMIC, FLYGYM_SENSORY_CHANNELS

    samples = _slot_sample_states(graph, controller)
    stuck = [
        entry.name
        for index, entry in enumerate(FLYGYM_SENSORY_CHANNELS)
        if entry.source == CHANNEL_SOURCE_DYNAMIC
        and len({sample[index] for sample in samples}) == 1
    ]
    assert stuck == []
