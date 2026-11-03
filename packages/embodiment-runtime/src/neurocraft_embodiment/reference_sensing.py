"""What the reference body's boundaries carry, and what parameterises them.

Split out of ``reference`` so that module stays inside the project's file-size
limit, and so the per-slot channel table sits beside the parameter classification it
shares a purpose with: both exist so a reviewer can read the boundary's shape and
capacity off the hashed identity instead of off the integrator.

Mirrors ``flygym_sensing`` for the second body.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from .identity import (
    CHANNEL_SOURCE_CONFIG_CONSTANT,
    CHANNEL_SOURCE_DYNAMIC,
    CHANNEL_SOURCE_LITERAL_CONSTANT,
    ChannelDeclaration,
)
from .seam import sensor_frame_channels

#: The five bounded actuation axes the primitive ``MotorVector`` carries, in the
#: order the motor boundary declares them.
REFERENCE_MOTOR_AXES: tuple[str, ...] = (
    "forward_thrust",
    "lateral_thrust",
    "vertical_thrust",
    "yaw_torque",
    "pitch_torque",
)

#: How this body sources every scalar of the protocol ``SensorFrame``, per slot.
#:
#: Eighteen of the twenty vary with the body's state. Exactly two do not, and both
#: are named here rather than left for a reviewer to discover in ``observe``:
#: ``ray_max_distance_m`` restates a hashed arena constant, and ``in_fluid`` is a
#: source literal that no configuration value can change, because this arena has no
#: fluid to be in.
REFERENCE_SENSORY_SOURCING: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        **{
            f"position_m[{index}]": (CHANNEL_SOURCE_DYNAMIC, "reference_body_state_position_m")
            for index in range(3)
        },
        **{
            f"velocity_m_s[{index}]": (
                CHANNEL_SOURCE_DYNAMIC,
                "reference_body_state_velocity_m_s",
            )
            for index in range(3)
        },
        "yaw_rad": (CHANNEL_SOURCE_DYNAMIC, "reference_body_state_yaw_rad"),
        "pitch_rad": (CHANNEL_SOURCE_DYNAMIC, "reference_body_state_pitch_rad"),
        "on_ground": (CHANNEL_SOURCE_DYNAMIC, "floor_contact_of_the_final_substep"),
        "horizontal_collision": (
            CHANNEL_SOURCE_DYNAMIC,
            "wall_or_pillar_contact_accumulated_over_substeps",
        ),
        "vertical_collision": (
            CHANNEL_SOURCE_DYNAMIC,
            "floor_or_ceiling_contact_accumulated_over_substeps",
        ),
        "in_fluid": (CHANNEL_SOURCE_LITERAL_CONSTANT, "false_reference_arena_has_no_fluid"),
        "local_light": (
            CHANNEL_SOURCE_DYNAMIC,
            "linear_height_gradient_floor_to_ceiling_16_levels",
        ),
        "ray_max_distance_m": (CHANNEL_SOURCE_CONFIG_CONSTANT, "arena_ray_max_distance_m"),
        **{
            f"ray_distances_m[{index}]": (
                CHANNEL_SOURCE_DYNAMIC,
                "pillar_and_bound_intersection_yaw_rotated_body_frame",
            )
            for index in range(6)
        },
    }
)

#: The twenty sensory declarations, ordered and completeness-checked by the protocol.
REFERENCE_SENSORY_CHANNELS: tuple[ChannelDeclaration, ...] = sensor_frame_channels(
    REFERENCE_SENSORY_SOURCING
)

#: How many sensory scalars vary with the body's state. Declared at the call site
#: and asserted against the records; it is the one number here nothing else derives.
REFERENCE_DYNAMIC_SENSORY_CHANNELS = 18

#: The five motor declarations. Every axis is driven by the graph readout, so a
#: future decoder that pins one to a constant has to say so here.
REFERENCE_MOTOR_CHANNELS: tuple[ChannelDeclaration, ...] = tuple(
    ChannelDeclaration(
        name=axis,
        source=CHANNEL_SOURCE_DYNAMIC,
        derivation="bounded_axis_to_declared_acceleration_or_rate",
    )
    for axis in REFERENCE_MOTOR_AXES
)

# Parameter classification, defined by dereference and nothing else
# -----------------------------------------------------------------
# Each tuple below names exactly the ``ReferenceArena`` attributes one code path
# reads. "Reads" means the path evaluates ``arena.<name>``; a constant that only
# appears in ``__post_init__`` or ``to_identity_material`` is not in any path. An
# earlier hand-maintained split was wrong in both directions -- it listed
# ``contact_epsilon_m`` as sensory when only the collision tie-break touches it, and
# omitted seven constants ``_substep`` dereferences every step -- so a test now
# re-derives these sets from the module's own AST and asserts the contents, not just
# the counts.
#
# The AST test used to take a hand-picked list of four functions and was therefore
# blind to anything nobody had thought to name -- which is how ``spawn_height_m``,
# read by ``_initial_state`` on every construction and every reset, ended up in no
# group at all and left both free-parameter counts one short. The test now scans
# every dereference in the module, so the list of paths is derived too.

#: Read by ``ReferenceEmbodiment._initial_state``, which ``__init__`` and ``reset``
#: both call. It fixes where the body starts, so it parameterises every observation
#: and every actuation correction that follows -- both boundaries, like the geometry.
REFERENCE_INITIAL_STATE_PARAMETERS: tuple[str, ...] = ("spawn_height_m",)

#: Construction-time geometry: the static pillar field both paths intersect.
#: Read by ``_generate_pillars`` and by the declaration guard ``pillar_crowding``.
REFERENCE_GEOMETRY_PARAMETERS: tuple[str, ...] = (
    "half_extent_x_m",
    "half_extent_z_m",
    "pillar_count",
    "pillar_exclusion_radius_m",
    "pillar_radius_m",
    "seed",
)

#: Read by ``ReferenceEmbodiment.observe`` and ``_ray_distance``.
REFERENCE_OBSERVATION_PARAMETERS: tuple[str, ...] = (
    "ceiling_y_m",
    "direction_epsilon",
    "floor_y_m",
    "half_extent_x_m",
    "half_extent_z_m",
    "pillar_radius_m",
    "ray_max_distance_m",
)

#: Read by ``ReferenceEmbodiment.apply`` and ``ReferenceEmbodiment._substep``.
REFERENCE_ACTUATION_PARAMETERS: tuple[str, ...] = (
    "ceiling_y_m",
    "contact_epsilon_m",
    "floor_y_m",
    "gravity_m_s2",
    "half_extent_x_m",
    "half_extent_z_m",
    "lateral_accel_m_s2",
    "linear_drag_per_s",
    "max_pitch_rad",
    "max_speed_m_s",
    "physics_steps_per_control_tick",
    "physics_timestep_s",
    "pillar_radius_m",
    "pitch_rate_rad_s",
    "thrust_accel_m_s2",
    "yaw_rate_rad_s",
)


#: Read by ``ReferenceEmbodiment._build_identity``, which hashes rather than acts.
#:
#: This group is deliberately **not** unioned into either boundary's parameter list.
#: The classification rule is "constants a code path turns into behaviour", which is
#: why ``ReferenceArena.__post_init__`` (validates) and ``to_identity_material``
#: (hashes) are excluded; ``_build_identity`` hashes too, and counting it as capacity
#: would charge the boundaries for scalars they do not act on.
#:
#: It is named anyway because it was the last path covered by nothing but arithmetic
#: luck: every constant it reads happens to be in another group, so the whole-module
#: union test could not have noticed if one were not. A test now asserts this path's
#: own dereferences and asserts the containment, so the coverage is stated instead of
#: incidental -- and if ``_build_identity`` ever reaches for a constant no behaviour
#: path reads, that test fails rather than the number quietly staying right.
REFERENCE_IDENTITY_PARAMETERS: tuple[str, ...] = (
    "physics_steps_per_control_tick",
    "physics_timestep_s",
    "seed",
)


def _classified(*groups: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(set().union(*groups)))


#: What the sensory boundary is parameterised by: the observation path, the geometry
#: that decides what its rays can hit, and the state the body starts in.
REFERENCE_SENSORY_PARAMETERS: tuple[str, ...] = _classified(
    REFERENCE_GEOMETRY_PARAMETERS,
    REFERENCE_INITIAL_STATE_PARAMETERS,
    REFERENCE_OBSERVATION_PARAMETERS,
)

#: What the motor boundary is parameterised by: the actuation path, the same
#: geometry, which its collision correction pushes out of, and the same start state.
REFERENCE_MOTOR_PARAMETERS: tuple[str, ...] = _classified(
    REFERENCE_GEOMETRY_PARAMETERS,
    REFERENCE_INITIAL_STATE_PARAMETERS,
    REFERENCE_ACTUATION_PARAMETERS,
)
