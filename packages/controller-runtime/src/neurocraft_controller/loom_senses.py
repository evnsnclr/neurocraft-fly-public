"""Approach-gated looming encoder (identity ``eem-loom-v1``).

The sensory half of the ``full_mcns_loom_v1`` profile. Same populations, same
channel names, same units, same target populations, same 1.0 gains, and the
byte-identical v1 raw formulas as :mod:`neurocraft_controller.embodied_senses`
-- only the *encoding* of the three threat channels changes, from an absolute
proximity level to a **rectified approach contrast**.

Why this identity exists (a recorded live observation, not a hypothesis)
-----------------------------------------------------------------------
In the first live session that survived under the current stack (2026-09-02,
session ``interactive-20260903T025643Z-adbe2bf2``, recorded in ``STATUS.md``)
the fly spent its last 83 s wedged in a top corner pushing ``forward 0.25 /
vertical 0.36`` at yaw -0.003 until the operator quit. One of the three
recorded causes is this module's subject: the escape-v2 encoder keeps
``threat_left``/``threat_right``/``threat_front`` ABSOLUTE
(``ESCAPE_V2_ABSOLUTE_CHANNEL_NAMES``), so a standing wall in the frontal ray
is a standing LC4 drive and therefore a standing escape -- the
``ESCAPE-ARC-01`` blocked-ray stimulus, presented continuously by the world.
D-060's rationale for keeping them absolute (temporal contrast adapts a held
threat away) was right for *scoring that arc* and wrong for a fly that has to
live in a room.

Under this identity a threat channel fires on the *rise* of proximity and not
on its level: an approaching object drives LC4, a standing or receding one does
not. ``touch`` keeps the escape-v2 transient encoding byte-identically, and
``light`` stays absolute (light events are a later stage).

The baseline is initialized to ``raw_0`` (the inherited D-059 rule) rather than
to 0. Under ``m_0 = 0`` a fly spawned or reset beside a wall, or with a player
already standing in front of it, would fire a one-shot escape at spawn for a
stimulus that is not approaching. Under ``m_0 = raw_0`` a threat already
present at reset is invisible until it changes, which is what an approach gate
is for. The recorded consequence is that ``ESCAPE-ARC-01``'s step-from-reset
method cannot measure the arc under this identity; see
``docs/encoders/eem-loom-v1.md``.

Claim discipline
----------------
This is an **engineering event gate chosen for a living demo, not anatomy.**
Real LC4 responds to looming -- angular expansion on the retina -- and this
encoder computes no angular size, no expansion rate, and no retinotopy: it
takes the first difference of a single scalar ray proximity against a leaky
mean. Nothing here claims that this is how LC4 computes, only that the
declared mapping is executed exactly as declared. The standing D-058 and
``ESCAPE-ARC-01`` caveats carry forward verbatim: proximity is not looming and
uniform per-capita drive is not retinotopy.

The rectification is likewise a declared engineering choice, disclosed as
such: real ON (approach) and OFF (recede) pathways are separate cell types,
and this encoder drops the OFF half of the threat channels entirely rather
than mapping it onto the same population the way the escape-v2 touch channel
does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from .embodied_senses import (
    _CONTRAST_INITIALIZATION_DECLARATION,
    _CONTRAST_UPDATE_DECLARATION,
    EEM_CHANNELS,
    FRONT_RAY_INDEX,
    LEFT_RAY_INDEX,
    RIGHT_RAY_INDEX,
    EemChannel,
    EemEncoderDocumentV2,
    EmbodiedPopulationBinding,
    EmbodiedSensesBinding,
    EmbodiedSensesError,
    _contrast_formula,
)

#: Schema of the loom-v1 encoder declaration. Its own value, so a loom-v1 run
#: can never be described by a v1, eem-v2, or escape-v2 encoder identity.
LOOM_V1_ENCODER_SCHEMA: Final = "neurocraft.eem.encoder.loom.v1"

#: The three channels encoded as RECTIFIED approach contrast: the drive is the
#: non-negative rise of the v1 raw proximity above its leaky running mean.
LOOM_V1_RECTIFIED_CHANNEL_NAMES: Final = ("threat_left", "threat_right", "threat_front")

#: Every channel that moves against a leaky running mean under this identity:
#: the three rectified threat channels plus ``touch``, which keeps the
#: escape-v2 (two-sided) transient encoding byte-identically.
LOOM_V1_CONTRAST_CHANNEL_NAMES: Final = (*LOOM_V1_RECTIFIED_CHANNEL_NAMES, "touch")

#: The one channel that keeps the v1 ABSOLUTE treatment verbatim. Light events
#: (a torch placed, a cave entered) are a later stage; until an event encoder
#: for them is declared, the absolute level is carried over unchanged rather
#: than silently re-encoded.
LOOM_V1_ABSOLUTE_CHANNEL_NAMES: Final = ("light",)

#: The declared rectified update, hashed verbatim into the encoder identity.
#: Identical to the frozen D-059 leaky-mean update in every respect except the
#: rectification: the deviation is taken as its non-negative half instead of
#: its absolute value, so a rise fires and a fall does not.
_RECTIFIED_UPDATE_DECLARATION: Final = (
    "drive_t = clip(raw_t - m_(t-1), 0, 1); m_t = (1 - alpha) * m_(t-1) + "
    "alpha * raw_t; one update per game tick, never per neural sub-step"
)


def _approach_formula(raw_formula: str) -> str:
    """The declared loom-v1 formula string for one rectified threat channel."""

    return (
        f"rectified approach contrast of the v1 raw value raw_t = {raw_formula}: "
        f"{_RECTIFIED_UPDATE_DECLARATION}, alpha = 1/8; "
        f"{_CONTRAST_INITIALIZATION_DECLARATION}"
    )


def _loom_channel(channel: EemChannel) -> EemChannel:
    """One v1 channel re-declared under the loom-v1 encoding for its name.

    The raw formula, name, unit, target populations, and gain are carried over
    from :data:`EEM_CHANNELS` unchanged; only the declared wrapper differs.
    """

    if channel.name in LOOM_V1_RECTIFIED_CHANNEL_NAMES:
        formula = _approach_formula(channel.formula)
    elif channel.name in LOOM_V1_CONTRAST_CHANNEL_NAMES:
        # ``touch``: the escape-v2 wrapper, byte-identical.
        formula = _contrast_formula(channel.formula)
    else:
        return channel
    return EemChannel(
        name=channel.name,
        unit=channel.unit,
        populations=channel.populations,
        formula=formula,
        gain=channel.gain,
    )


#: The five loom-v1 channels: identical names, units, target populations, and
#: 1.0 gains as v1; only the encoding of each value changes. ``light`` is the
#: v1 channel object verbatim, ``touch`` is the escape-v2 channel object
#: verbatim, and the three threat channels wrap the byte-identical v1 raw
#: formulas in the declared rectified-approach encoding.
LOOM_V1_CHANNELS: Final[tuple[EemChannel, ...]] = tuple(
    _loom_channel(channel) for channel in EEM_CHANNELS
)


@dataclass(frozen=True, slots=True)
class EemEncoderDocumentLoomV1(EemEncoderDocumentV2):
    """The canonical loom-v1 (approach-gated) encoder declaration.

    Rides the frozen D-059 leaky-mean machinery -- one update per game tick,
    ``alpha`` pinned to 1/8 exactly, baseline initialized to ``raw_0`` so
    ``drive_0`` is exactly 0 -- with the three threat channels taking the
    RECTIFIED half of the deviation (``clip(raw_t - m_(t-1), 0, 1)``) instead
    of its absolute value. Nothing here is fit to a measurement: alpha and the
    update are inherited from the frozen D-059 identity, and the rectification
    was declared before anything was measured under it.

    ``rectified_channels`` must be exactly the three threat channels and must
    be a subset of ``contrast_channels``; every other combination is a
    different (undeclared) encoder and is refused.
    """

    channels: tuple[EemChannel, ...] = LOOM_V1_CHANNELS
    contrast_channels: tuple[str, ...] = LOOM_V1_CONTRAST_CHANNEL_NAMES
    absolute_channels: tuple[str, ...] = LOOM_V1_ABSOLUTE_CHANNEL_NAMES
    rectified_channels: tuple[str, ...] = LOOM_V1_RECTIFIED_CHANNEL_NAMES

    def __post_init__(self) -> None:
        if self.rectified_channels != LOOM_V1_RECTIFIED_CHANNEL_NAMES:
            raise EmbodiedSensesError(
                "the loom-v1 encoder rectifies exactly the three threat "
                "channels; a different rectified set is a different "
                "(undeclared) encoder"
            )
        # The v2 gate below only requires that contrast/absolute PARTITION the
        # declared channels, which would still admit (say) light moved into
        # contrast or touch moved into absolute under this schema. This
        # identity declares one exact partition, so both tuples are pinned.
        if (
            self.contrast_channels != LOOM_V1_CONTRAST_CHANNEL_NAMES
            or self.absolute_channels != LOOM_V1_ABSOLUTE_CHANNEL_NAMES
        ):
            raise EmbodiedSensesError(
                "the loom-v1 encoder declares exactly the contrast set "
                f"{LOOM_V1_CONTRAST_CHANNEL_NAMES} and the absolute set "
                f"{LOOM_V1_ABSOLUTE_CHANNEL_NAMES}; a different partition is a "
                "different (undeclared) encoder"
            )
        if not set(self.rectified_channels) <= set(self.contrast_channels):
            raise EmbodiedSensesError(
                "every rectified channel must also be a contrast channel; a "
                "rectified absolute channel has no baseline to rectify against"
            )
        # The v2 gates apply unchanged: alpha is pinned to 1/8 exactly and the
        # contrast/absolute sets must partition the declared channels.
        EemEncoderDocumentV2.__post_init__(self)

    def to_identity_material(self) -> dict[str, Any]:
        encoding: dict[str, str] = dict.fromkeys(self.absolute_channels, "absolute")
        encoding.update(
            {
                name: (
                    "rectified_approach_contrast"
                    if name in self.rectified_channels
                    else "temporal_contrast"
                )
                for name in self.contrast_channels
            }
        )
        return {
            "absolute_exception": (
                "light keeps the v1 ABSOLUTE treatment verbatim: light events "
                "are a later stage, and re-encoding the level before an event "
                "encoder for it is declared would change a channel nothing "
                "here has measured"
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
                    "verbatim from the frozen D-059 identity: an approach "
                    "drives the threat channels while proximity is rising and "
                    "a standing obstacle adapts toward zero with tau ~= 8 "
                    "ticks; dyadic, so exact in binary floating point"
                ),
                "initialization": _CONTRAST_INITIALIZATION_DECLARATION,
                "rectification_disclosure": (
                    "real ON (approach) and OFF (recede) pathways are "
                    "separate cell types; this identity declares the threat "
                    "drive as the ON half ALONE -- clip(raw_t - m_(t-1), 0, "
                    "1) -- and drops the OFF half entirely rather than "
                    "mapping it onto the same population the way the "
                    "escape-v2 touch channel does. An engineering event gate, "
                    "not ON/OFF physiology, and not a model of angular "
                    "expansion: no angular size, no expansion rate, and no "
                    "retinotopy is computed anywhere"
                ),
                "rectified_channels": list(self.rectified_channels),
                "state": (
                    "the four running means are the only cross-tick state: "
                    "pure functions of the observation history (same "
                    "observation sequence reproduces the same means, drives, "
                    "and -- on the same backend -- the same trajectory), "
                    "updated once per game tick, cleared by reset(), never "
                    "serialized"
                ),
                "touch_update": _CONTRAST_UPDATE_DECLARATION,
                "update": _RECTIFIED_UPDATE_DECLARATION,
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
            "schema": LOOM_V1_ENCODER_SCHEMA,
            "threat_rationale": (
                "the three threat channels are the ones whose raw signal is a "
                "STANDING field the world holds true for as long as the body "
                "sits near a surface, while the engineering-relevant signal "
                "is the approach. Under the escape-v2 identity a standing "
                "wall in the frontal ray was a standing LC4 drive and a "
                "standing, directionless escape -- observed live on "
                "2026-09-02, where the fly spent 83 s wedged in a corner "
                "pushing forward 0.25 / vertical 0.36. Under this identity an "
                "approaching object fires and a standing or receding one does "
                "not"
            ),
            "touch_rationale": (
                "touch carries the escape-v2 (two-sided) transient encoding "
                "byte-identically: a contact onset drives the bristles near "
                "full amplitude, sustained resting contact adapts toward "
                "zero, and a release produces an OFF transient of the same "
                "sign. That channel's raw signal is already an event and this "
                "identity changes nothing about it"
            ),
        }


def bind_embodied_senses_loom_v1(
    populations: EmbodiedPopulationBinding,
) -> EmbodiedSensesBinding:
    """Build the declared loom-v1 (approach-gated) encoder over one binding."""

    return EmbodiedSensesBinding(
        populations=populations,
        encoder=EemEncoderDocumentLoomV1(population_binding_sha256=populations.identity_sha256),
    )
