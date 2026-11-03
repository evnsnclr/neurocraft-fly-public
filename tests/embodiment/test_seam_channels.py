"""The channel convention is derived from the protocol; this pins it independently.

Two layers, deliberately in different files.

``seam.sensor_frame_widths`` derives every width from ``SensorFrame``'s own
annotations, so there is no number in ``seam.py`` left to edit. Derivation removes
the drift but it also removes the tripwire: a protocol field added, renamed,
reordered or rewidened now flows through silently and rotates every
``EmbodimentIdentity``, because both the channel names and the channel count are
hashed into ``BoundaryIdentity.to_identity_material``.

This file restores the tripwire from outside the derivation: the 20 names are
written out by hand, digested, and compared. A protocol author has no reason to edit
this file while editing the protocol, which is exactly what makes it evidence.
"""

from __future__ import annotations

import dataclasses

import pytest
from neurocraft_embodiment.identity import canonical_sha256
from neurocraft_embodiment.seam import (
    SENSOR_FRAME_CHANNELS,
    SENSOR_FRAME_SCALAR_COUNT,
    EmbodimentError,
    sensor_frame_widths,
)
from neurocraft_protocol import RAY_COUNT, SensorFrame

#: Written out by hand, in protocol field order. Not derived from anything.
PINNED_SENSOR_FRAME_CHANNELS: tuple[str, ...] = (
    "position_m[0]",
    "position_m[1]",
    "position_m[2]",
    "velocity_m_s[0]",
    "velocity_m_s[1]",
    "velocity_m_s[2]",
    "yaw_rad",
    "pitch_rad",
    "on_ground",
    "horizontal_collision",
    "vertical_collision",
    "in_fluid",
    "local_light",
    "ray_max_distance_m",
    "ray_distances_m[0]",
    "ray_distances_m[1]",
    "ray_distances_m[2]",
    "ray_distances_m[3]",
    "ray_distances_m[4]",
    "ray_distances_m[5]",
)

#: ``sha256`` over the canonical JSON of the list above.
PINNED_SENSOR_FRAME_CHANNEL_DIGEST = (
    "0c6d6e46086368805036da3dbd21ce91d557ae70406d3d554bc3fcdfcc354779"
)


def test_the_derived_channel_list_matches_the_hand_written_pin() -> None:
    """Any protocol width, rename, addition or reorder must land here first.

    Every name and the count are hashed into ``BoundaryIdentity``, so a change that
    passes this test silently is a change that rotates every embodiment identity in
    the repository without anyone deciding to.
    """

    assert SENSOR_FRAME_CHANNELS == PINNED_SENSOR_FRAME_CHANNELS
    assert SENSOR_FRAME_SCALAR_COUNT == len(PINNED_SENSOR_FRAME_CHANNELS) == 20
    assert canonical_sha256(list(SENSOR_FRAME_CHANNELS)) == PINNED_SENSOR_FRAME_CHANNEL_DIGEST


def test_the_derived_widths_survive_a_round_trip_through_the_protocol_validators() -> None:
    """Widths come from annotations; the validators are a second, independent source.

    An all-ones frame is used rather than an all-zeros one because
    ``ray_max_distance_m`` has a positive minimum. If an annotation and its validator
    ever disagreed about a width, this construction is what would notice.
    """

    widths = dict(sensor_frame_widths())
    hints = {field.name: field for field in dataclasses.fields(SensorFrame)}
    assert set(widths) == set(hints)

    values: dict[str, object] = {}
    for name, width in widths.items():
        if width > 1:
            values[name] = (1.0,) * width
        elif name in {"on_ground", "horizontal_collision", "vertical_collision", "in_fluid"}:
            values[name] = False
        elif name == "local_light":
            values[name] = 1
        else:
            values[name] = 1.0

    payload = SensorFrame(**values).to_dict()  # type: ignore[arg-type]
    for name, width in widths.items():
        encoded = payload[name]
        measured = len(encoded) if isinstance(encoded, list) else 1
        assert measured == width, f"{name}: annotation says {width}, wire carries {measured}"
    assert sum(widths.values()) == SENSOR_FRAME_SCALAR_COUNT


def test_the_ray_width_is_cross_checked_against_the_protocol_validator() -> None:
    """The annotation and ``RAY_COUNT`` are two sources; the seam refuses a split."""

    assert dict(sensor_frame_widths())["ray_distances_m"] == RAY_COUNT


def test_the_convention_refuses_a_field_shape_it_cannot_widen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A variadic or non-float tuple has no fixed scalar width; guessing one is worse.

    Patched on the seam's own ``typing`` handle rather than on the protocol, so no
    protocol object is mutated and no other test can observe the change.
    """

    from neurocraft_embodiment import seam

    real = seam.typing.get_type_hints

    def variadic(obj: object, *args: object, **kwargs: object) -> dict[str, object]:
        hints = dict(real(obj))  # type: ignore[arg-type]
        hints["ray_distances_m"] = tuple[float, ...]
        return hints

    monkeypatch.setattr(seam.typing, "get_type_hints", variadic)
    with pytest.raises(EmbodimentError, match="variadic or not all-float"):
        seam.sensor_frame_widths()

    def stringy(obj: object, *args: object, **kwargs: object) -> dict[str, object]:
        hints = dict(real(obj))  # type: ignore[arg-type]
        hints["local_light"] = str
        return hints

    monkeypatch.setattr(seam.typing, "get_type_hints", stringy)
    with pytest.raises(EmbodimentError, match="does not know how to widen"):
        seam.sensor_frame_widths()
