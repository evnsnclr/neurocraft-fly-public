"""The embodiment seam: the smallest interface any body must satisfy.

Scope, stated once so it cannot drift:

* An embodiment *observes* — it turns its own physical state into the protocol
  ``Observation`` the pinned controller already accepts.
* An embodiment *applies* one bounded command and advances time deterministically.
* An embodiment *records provenance* — a hashed :class:`EmbodimentIdentity`.

An embodiment contains no policy. It has no goal, no target, no reward, no plan,
no oscillator and no scripted fallback. It may not read the controller's internal
state for anything except the motor currency it declared, and it may not choose
what to do; it only integrates what it was given. Faithful Organism Mode purity
lives or dies on that boundary, so the seam makes the boundary the type system's
problem rather than a reviewer's.

Why the command carries both currencies
---------------------------------------
``ConnectomeController.step`` projects the graph readout down to five bounded
primitive axes. That projection is exactly right for a block entity and exactly
wrong for a 42-DOF musculoskeletal body: forcing a six-legged fly through five
axes would need a 5 -> 42 expansion that no hashed decoder can justify. So the
command carries *both* the projected ``MotorVector`` and the verified post-step
``NeuralState``, and every embodiment declares in its identity which one it
consumes. Neither is a policy input; both are readouts of the same tick.
"""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from neurocraft_neural import NeuralState
from neurocraft_protocol import (
    RAY_COUNT,
    MotorVector,
    Observation,
    RunIdentity,
    SensorFrame,
    WorldState,
)

from .identity import (
    CONTROL_HZ,
    CONTROL_PERIOD_NS,
    ChannelDeclaration,
    EmbodimentIdentity,
)

# ``CONTROL_HZ`` / ``CONTROL_PERIOD_NS`` are defined in ``identity`` and re-exported
# here, where every body already imports them from. They moved because
# ``EmbodimentIdentity`` has to check a body's declared ``control_hz`` against the
# rate ``observation_timing`` below stamps unconditionally, and ``identity`` cannot
# import ``seam`` -- this module imports it. Nothing else changed: the contract is
# still one number, still frozen, still an ADR-level act to change.
__all__ = [
    "CONTROL_HZ",
    "CONTROL_PERIOD_NS",
    "SENSOR_FRAME_CHANNELS",
    "SENSOR_FRAME_SCALAR_COUNT",
    "Embodiment",
    "EmbodimentCommand",
    "EmbodimentError",
    "EmbodimentTransition",
    "observation_timing",
    "sensor_frame_channels",
    "sensor_frame_widths",
]

#: The annotations a ``SensorFrame`` field may carry and still be one scalar.
_SCALAR_ANNOTATIONS: frozenset[Any] = frozenset({bool, int, float})


class EmbodimentError(RuntimeError):
    """A body failed closed: bad command, bad state, or missing prerequisites."""


def sensor_frame_widths() -> tuple[tuple[str, int], ...]:
    """Every ``SensorFrame`` field and its width in scalars, read off the protocol.

    This is *the* channel convention for a sensory boundary: a channel is one scalar
    the controller reads, never a vector counted as one.

    The widths used to be a hand-written table here, cross-checked against
    ``SensorFrame.FIELDS`` by name only. A reviewer changed ``ray_distances_m`` from
    6 to 5 and the package imported cleanly with a 19-channel convention, because the
    only thing checking the seam's table was the seam's own table. Nothing is
    declared here now: the field list, its **order** and every width come from
    ``dataclasses.fields`` and the resolved annotations, so there is no number left
    for anyone to edit. ``SensorFrame.FIELDS`` is a hand-maintained set inside the
    protocol and cannot see a reorder, so it is kept as a cross-check of the protocol
    against itself rather than as the source.

    Fails closed on anything the convention cannot widen honestly -- a variadic
    tuple, a non-float tuple member, or a scalar annotation outside
    ``{bool, int, float}`` -- rather than guessing a width.
    """

    hints: dict[str, Any] = typing.get_type_hints(SensorFrame)
    widths: list[tuple[str, int]] = []
    for field in dataclasses.fields(SensorFrame):
        annotation = hints[field.name]
        if typing.get_origin(annotation) is tuple:
            args = typing.get_args(annotation)
            if not args or Ellipsis in args or any(arg is not float for arg in args):
                raise EmbodimentError(
                    f"SensorFrame.{field.name} is annotated {annotation!r}, which is "
                    "variadic or not all-float; the embodiment channel convention "
                    "cannot derive a fixed scalar width from it"
                )
            widths.append((field.name, len(args)))
        elif annotation in _SCALAR_ANNOTATIONS:
            widths.append((field.name, 1))
        else:
            raise EmbodimentError(
                f"SensorFrame.{field.name} is annotated {annotation!r}, which the "
                "embodiment channel convention does not know how to widen"
            )
    derived = tuple(widths)
    if {name for name, _ in derived} != SensorFrame.FIELDS:
        raise EmbodimentError(
            "SensorFrame.FIELDS disagrees with the dataclass fields; the protocol "
            "contradicts itself, so no channel convention over it can be honest"
        )
    ray_width = dict(derived)["ray_distances_m"]
    if ray_width != RAY_COUNT:
        raise EmbodimentError(
            f"SensorFrame.ray_distances_m is annotated with {ray_width} floats but "
            f"the protocol validates RAY_COUNT={RAY_COUNT}; the annotation and the "
            "validator disagree, so the ray width is not a single fact"
        )
    return derived


def _sensor_frame_channels() -> tuple[str, ...]:
    channels: list[str] = []
    for name, width in sensor_frame_widths():
        if width == 1:
            channels.append(name)
        else:
            channels.extend(f"{name}[{index}]" for index in range(width))
    return tuple(channels)


#: The 20 scalar channels a protocol ``SensorFrame`` carries, in field order.
SENSOR_FRAME_CHANNELS: tuple[str, ...] = _sensor_frame_channels()

#: How many scalars one protocol ``SensorFrame`` carries. Derived, never declared.
SENSOR_FRAME_SCALAR_COUNT: int = len(SENSOR_FRAME_CHANNELS)


def sensor_frame_channels(
    sourcing: Mapping[str, tuple[str, str]],
) -> tuple[ChannelDeclaration, ...]:
    """Turn a body's own per-slot sourcing table into the boundary's declarations.

    The keys are the body's; the required key *set* and the emitted *order* are the
    protocol's, taken from :data:`SENSOR_FRAME_CHANNELS`. That is what makes the
    boundary's channel list an independent statement rather than a restatement: a
    body cannot declare a 19-slot frame, cannot silently inherit a slot it never
    thought about, and cannot reorder the wire. Writing the table forces its author
    to say, for every one of the twenty scalars, where that scalar comes from --
    which is precisely where a saturated ray or a hardcoded flag stops being
    invisible.
    """

    expected = set(SENSOR_FRAME_CHANNELS)
    missing = [name for name in SENSOR_FRAME_CHANNELS if name not in sourcing]
    unknown = sorted(set(sourcing) - expected)
    if missing or unknown:
        raise EmbodimentError(
            "a boundary emitting a protocol SensorFrame must source every slot "
            f"exactly once; missing={missing}, unknown={unknown}"
        )
    return tuple(
        ChannelDeclaration(name=name, source=sourcing[name][0], derivation=sourcing[name][1])
        for name in SENSOR_FRAME_CHANNELS
    )


@dataclass(frozen=True, slots=True)
class EmbodimentCommand:
    """One tick of verified controller output handed to a body.

    Both fields describe the *same* neural step. ``motors`` is the primitive
    projection an embodiment with five actuation axes consumes; ``neural_state`` is
    the verified state a wide-boundary embodiment decodes through its own hashed
    motor decoder. An embodiment reads exactly one of them, as declared.
    """

    command_seq: int
    motors: MotorVector
    neural_state: NeuralState

    def __post_init__(self) -> None:
        if isinstance(self.command_seq, bool) or not isinstance(self.command_seq, int):
            raise EmbodimentError("embodiment command sequence must be an integer")
        if self.command_seq < 0:
            raise EmbodimentError("embodiment command sequence must be non-negative")
        if not isinstance(self.motors, MotorVector):
            raise EmbodimentError("embodiment command requires a protocol MotorVector")
        if not isinstance(self.neural_state, NeuralState):
            raise EmbodimentError("embodiment command requires a verified NeuralState")
        # MotorVector already bounds every axis to [-1, 1] in its own __post_init__;
        # NeuralState carries a self-verifying digest. Re-check the latter here so a
        # forged state cannot reach a body.
        self.neural_state.assert_integrity()


@dataclass(frozen=True, slots=True)
class EmbodimentTransition:
    """The authoritative world transition a body produced for one control tick.

    ``passive_forces`` and ``external_forces`` are *declared*, not inferred: the
    trace's ``DeclaredTransitionForces`` refuses to guess, so a body must state what
    it applied.
    """

    world_after: WorldState
    passive_forces: tuple[float, float, float]
    external_forces: tuple[float, float, float]
    physics_steps: int
    simulated_time_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.world_after, WorldState):
            raise EmbodimentError("embodiment transition requires a protocol WorldState")
        for name in ("passive_forces", "external_forces"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or len(value) != 3:
                raise EmbodimentError(f"{name} must be a 3-tuple")
        if isinstance(self.physics_steps, bool) or not isinstance(self.physics_steps, int):
            raise EmbodimentError("physics_steps must be an integer")
        if self.physics_steps < 1:
            raise EmbodimentError("a control tick must advance at least one physics step")
        if self.simulated_time_s <= 0.0:
            raise EmbodimentError("a control tick must advance simulated time")


@runtime_checkable
class Embodiment(Protocol):
    """A body the pinned connectome controller can drive.

    Implementations must be deterministic given ``(identity, seed, command
    sequence)``: two runs of the same body over the same commands must produce the
    same transitions on the same build.
    """

    @property
    def identity(self) -> EmbodimentIdentity:
        """Hashed provenance for this body, its config, its boundaries and its seed."""

    def reset(self) -> None:
        """Return the body to its declared initial state. Never randomizes."""

    def observe(self, *, identity: RunIdentity, sequence: int) -> Observation:
        """Encode current physical state as a protocol Observation for ``sequence``."""

    def apply(self, command: EmbodimentCommand) -> EmbodimentTransition:
        """Apply one bounded command and advance physics by exactly one control tick."""

    def provenance(self) -> Mapping[str, str]:
        """Environment facts the run loop records beside the run.

        This used to say "recorded alongside the run" while ``run_embodiment`` had
        zero call sites for it, so a stub-backed body's ``flygym_version:
        "2.1.0-stub"`` lived only inside the identity's hash preimage and was
        unreadable from the trace or the receipt. ``run_embodiment`` now calls this,
        cross-checks it against the hashed identity, resolves every declared
        ``<distribution>_version`` against the installed set, and writes both the
        mapping and that classification into the receipt and the trace manifest.
        """


def observation_timing(sequence: int) -> tuple[int, int]:
    """The frozen (game_tick, sent_monotonic_ns) binding for a control sequence.

    Shared by every embodiment so two bodies cannot disagree about what tick they
    are on. 20 Hz, 50 ms per tick, matching the Minecraft lane exactly.
    """

    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise EmbodimentError("observation sequence must be a non-negative integer")
    return sequence, sequence * CONTROL_PERIOD_NS
