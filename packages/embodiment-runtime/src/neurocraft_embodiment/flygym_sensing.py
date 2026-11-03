"""What the FlyGym body's sensory boundary carries, slot by slot.

Split out of ``flygym_body`` so the per-slot record and the group-level mapping sit
next to each other and cross-check each other, and so neither file has to grow past
the size the project allows.

Two records, deliberately
-------------------------
:data:`FLYGYM_SENSORY_MAPPING` describes the mapping by *group* -- the position
read, the velocity method, the light model. Its keys are not ``SensorFrame`` field
names (``yaw_pitch`` covers two fields, ``ray_other`` covers five), so nothing
mechanical could ever check it for completeness, and three fields escaped it
entirely for as long as it existed.

:data:`FLYGYM_SENSORY_CHANNELS` describes the same boundary by *slot*: one record
per scalar, completeness enforced against the protocol by
``seam.sensor_frame_channels``. It is what makes the constants visible -- a reviewer
reading the identity can see that most of this body's frame does not move.

The two are cross-checked below in both directions at import: every derivation a
channel declares must be a value the mapping states, and every value the mapping
states must be used by some channel. Neither can drift without the other.
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
from .seam import EmbodimentError, sensor_frame_channels

FLYGYM_SENSORY_MAPPING_SCHEMA = "neurocraft.embodiment.flygym.sensory-mapping.v1"

#: The Design-A sensory mapping, by group, exactly as the boundary identity records
#: it. Immutable: it reaches into hashed identity material, and a plain ``dict``
#: exported in ``__all__`` let a caller rotate every subsequently built FlyGym
#: sensory boundary hash from outside the package.
#:
#: ``vertical_collision`` deliberately does *not* claim a segment filter. The reading
#: is ``FlyGymEmbodiment._contacts``' unfiltered "any contact-force row with a
#: nonzero magnitude"; whether FlyGym's contact arrays even expose the segment
#: identity such a filter would need cannot be confirmed without a real install, so
#: the honest description is recorded rather than the intended one. A filter, if one
#: is added later, is a new mapping string and a new identity.
FLYGYM_SENSORY_MAPPING: Mapping[str, str] = MappingProxyType(
    {
        "horizontal_collision": "false_flat_ground_world_has_no_vertical_geometry",
        "in_fluid": "false_neuromechfly_has_no_fluid_model",
        "local_light": "thorax_height_over_light_span_mm",
        "on_ground": "any_leg_contact_found_gt_zero",
        "position_m": "thorax_body_position_mm_div_1000",
        "ray_down": "thorax_height_above_ground",
        "ray_max": "config_ray_max_distance_m",
        "ray_other": "saturated_flat_ground_no_occluders",
        "velocity_m_s": "finite_difference_between_consecutive_samples_of_the_same_path",
        "vertical_collision": "any_nonzero_ground_contact_force_row_no_segment_filter",
        "yaw_pitch": "zyx_euler_from_thorax_quaternion",
    }
)

#: The config attributes ``FlyGymEmbodiment.observe`` dereferences. These are the
#: sensory boundary's free parameters -- the scalars a designer chose -- and the
#: count was previously a hand-written ``2`` with no rule tying it to anything.
FLYGYM_SENSORY_PARAMETERS: tuple[str, ...] = ("light_span_mm", "ray_max_distance_m")

_M = FLYGYM_SENSORY_MAPPING

#: Where every scalar of this body's protocol ``SensorFrame`` comes from.
#:
#: Twelve of the twenty vary. The other eight are stated here rather than left for a
#: reviewer to find in ``observe``: on ``FlatGroundWorld`` five of the six rays
#: saturate at maximum range because there is nothing to occlude them, the sixth ray
#: is the thorax height, and two flags are source literals about a world with no
#: vertical geometry and no fluid model. Nothing in the previous identity said any of
#: that, and the gap is the reviewer-facing fact: this body feeds the connectome a
#: world with no lateral range sense at all.
FLYGYM_SENSORY_SOURCING: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        **{
            f"position_m[{index}]": (CHANNEL_SOURCE_DYNAMIC, _M["position_m"]) for index in range(3)
        },
        **{
            f"velocity_m_s[{index}]": (CHANNEL_SOURCE_DYNAMIC, _M["velocity_m_s"])
            for index in range(3)
        },
        "yaw_rad": (CHANNEL_SOURCE_DYNAMIC, _M["yaw_pitch"]),
        "pitch_rad": (CHANNEL_SOURCE_DYNAMIC, _M["yaw_pitch"]),
        "on_ground": (CHANNEL_SOURCE_DYNAMIC, _M["on_ground"]),
        "horizontal_collision": (CHANNEL_SOURCE_LITERAL_CONSTANT, _M["horizontal_collision"]),
        "vertical_collision": (CHANNEL_SOURCE_DYNAMIC, _M["vertical_collision"]),
        "in_fluid": (CHANNEL_SOURCE_LITERAL_CONSTANT, _M["in_fluid"]),
        "local_light": (CHANNEL_SOURCE_DYNAMIC, _M["local_light"]),
        "ray_max_distance_m": (CHANNEL_SOURCE_CONFIG_CONSTANT, _M["ray_max"]),
        **{
            f"ray_distances_m[{index}]": (CHANNEL_SOURCE_CONFIG_CONSTANT, _M["ray_other"])
            for index in range(5)
        },
        "ray_distances_m[5]": (CHANNEL_SOURCE_DYNAMIC, _M["ray_down"]),
    }
)

#: The twenty declarations, ordered and completeness-checked against the protocol.
FLYGYM_SENSORY_CHANNELS: tuple[ChannelDeclaration, ...] = sensor_frame_channels(
    FLYGYM_SENSORY_SOURCING
)

#: How many of this body's sensory scalars vary. Declared here, asserted against the
#: records by ``BoundaryIdentity``; nothing else in the repository derives it.
FLYGYM_DYNAMIC_SENSORY_CHANNELS = 12


def _assert_mapping_and_channels_agree() -> None:
    """Both records describe one boundary, so neither may name anything the other does not."""

    declared = {entry.derivation for entry in FLYGYM_SENSORY_CHANNELS}
    stated = set(FLYGYM_SENSORY_MAPPING.values())
    if declared != stated:
        raise EmbodimentError(
            "the FlyGym sensory mapping and its per-slot channel table disagree; "
            f"only in the channels={sorted(declared - stated)}, "
            f"only in the mapping={sorted(stated - declared)}"
        )
    tally = sum(1 for entry in FLYGYM_SENSORY_CHANNELS if entry.source == CHANNEL_SOURCE_DYNAMIC)
    if tally != FLYGYM_DYNAMIC_SENSORY_CHANNELS:
        raise EmbodimentError(
            f"the FlyGym channel table tallies {tally} dynamic slots but "
            f"{FLYGYM_DYNAMIC_SENSORY_CHANNELS} are declared"
        )


_assert_mapping_and_channels_agree()
