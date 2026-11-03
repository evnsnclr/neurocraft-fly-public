"""Wide sensory/motor boundaries sized for the NeuroMechFly body.

Why these are siblings and not edits
------------------------------------
The reason is **schema separation**, and it is worth stating precisely because an
earlier version of this docstring gave a reason that is false. It claimed that
raising ``neurocraft_neural.adapters``' 32-channel / 16-pool budgets would invalidate
already-recorded identities. It would not: ``SensoryAdapter`` hashes only
``{channels, graph_identity_sha256, schema}`` and ``MotorDecoder`` only
``{graph_identity_sha256, pools, schema}``, so those constants are nowhere in any
recorded hash and raising them would leave every existing digest byte-identical.

The real argument is that a boundary of a different shape is a different boundary. A
42-DOF musculoskeletal readout is not the Minecraft entity's five-axis readout with a
bigger number attached; it deserves its own schema string, so that a trace can never
be ambiguous about which boundary produced it, and its own identity, so the two can
never collide. Widening the shipped classes in place would let one schema string mean
two different things -- which *is* how a recorded claim gets silently weakened, just
not by way of the constants.

The capacity argument, stated plainly
-------------------------------------
High dimension is acceptable in this project; high parametric capacity is not. A
42-DOF body legitimately needs 42 output channels, but each channel is allowed only
a handful of declared scalars (its member weights, a gain, a dead zone, a neutral
angle and a half range). The constructor computes the total number of scalars a
designer could have chosen and asserts it equals the number the caller declared, so a
caller cannot quietly widen the boundary without changing a number that appears in
the identity. It then asserts that total against
:data:`MAX_FLYGYM_SENSORY_FREE_PARAMETERS` / :data:`MAX_FLYGYM_MOTOR_FREE_PARAMETERS`,
which is not redundant with the shape budgets even though those budgets reject first
for every counting rule shipped today -- see :func:`_assert_declared`.

One free parameter per target/member is the right rule *for this boundary*, where a
designer really does choose every weight. It is not a general rule: a retinotopic
adapter would make per-column weights structural rather than free and would need its
own counting rule and its own schema.

This module imports nothing from ``flygym``. It is pure graph/arithmetic code and
is fully testable on the CI fixture.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from neurocraft_connectome import RuntimeGraph
from neurocraft_connectome.model import sha256_json
from neurocraft_neural import (
    EncodedStimulation,
    MotorFrame,
    MotorPool,
    NeuralConfigurationError,
    NeuralIntegrityError,
    NeuralState,
    SensorChannel,
)

# The identity helpers below are the same private encoders the shipped adapters use
# (``neurocraft_neural.adapters`` imports them from ``model`` in exactly this way).
# They must be byte-identical here or the EncodedStimulation/MotorFrame integrity
# checks in the engine and decoder would reject this boundary's output.
from neurocraft_neural.model import _finite_float, _float_hex, _float_tuple_digest

from .identity import (
    CHANNEL_SOURCE_CONFIG_CONSTANT,
    CHANNEL_SOURCE_DYNAMIC,
    ChannelDeclaration,
)

MAX_FLYGYM_SENSOR_CHANNELS = 96
MAX_FLYGYM_SENSOR_TARGETS_TOTAL = 192
MAX_FLYGYM_MOTOR_CHANNELS = 48
MAX_FLYGYM_MOTOR_MEMBERS_TOTAL = 96
MAX_ABSOLUTE_FLYGYM_WEIGHT = 8.0
MAX_ABSOLUTE_FLYGYM_PARAMETER = 8.0

# The per-unit counting rules, named once and used both by the constructors that do
# the counting and by the maxima below. They were previously a bare ``4`` inside the
# sensory loop, a ``2`` inside the motor loop, a ``2`` inside the joint loop, and two
# independently written ``4``s in the maxima -- five literals for three rules, with
# nothing tying any of them together. A change to a per-channel count would have
# silently falsified numbers already inside the hashed identity material.
#: gain, offset, clip_min, clip_max.
FLYGYM_SENSOR_CHANNEL_PARAMETERS = 4
#: gain, dead_zone.
FLYGYM_MOTOR_POOL_PARAMETERS = 2
#: neutral_rad, half_range_rad.
FLYGYM_JOINT_READOUT_PARAMETERS = 2

# The capacity limits, derived from the shape budgets rather than declared, so the
# two can never disagree. A previous hand-picked ceiling of 640 was removed because it
# could never fire -- the shape budgets bind at 576 and 288 -- and a fail-closed check
# that cannot fire reads as protection that is not there.
#
# Removing it removed the wrong half. What made 640 useless was the *number*, not the
# comparison: with no comparison left at all, nothing at runtime related a computed
# free-parameter count to the maximum being stamped into the identity beside it. A
# reviewer then added four lines to the sensory loop --
#
#     if channel.saturation == "tanh":
#         free_parameters += 1
#
# -- and built a boundary that recorded ``max_free_parameters: 576`` while actually
# carrying 672. Every test passed, because no shipped test constructs a saturating
# channel and because the shape budgets only bind the *shape*. The maxima are derived
# from the budgets here, and ``_assert_declared`` compares against them.
#: Sensory: one free parameter per target, plus the per-channel affine parameters.
#: 192 + 4 * 96 = 576.
MAX_FLYGYM_SENSORY_FREE_PARAMETERS = (
    MAX_FLYGYM_SENSOR_TARGETS_TOTAL + FLYGYM_SENSOR_CHANNEL_PARAMETERS * MAX_FLYGYM_SENSOR_CHANNELS
)
#: Motor: one per member, plus the per-pool parameters, plus the per-joint readout
#: parameters (a joint is a pool, so joints are capped by the pool budget).
#: 96 + (2 + 2) * 48 = 288.
MAX_FLYGYM_MOTOR_FREE_PARAMETERS = (
    MAX_FLYGYM_MOTOR_MEMBERS_TOTAL
    + (FLYGYM_MOTOR_POOL_PARAMETERS + FLYGYM_JOINT_READOUT_PARAMETERS) * MAX_FLYGYM_MOTOR_CHANNELS
)

FLYGYM_SENSORY_SCHEMA = "neurocraft.flygym.sensory-adapter.v1"
FLYGYM_MOTOR_SCHEMA = "neurocraft.flygym.motor-decoder.v1"

#: Leg identifiers in FlyGym's own order: left/right x front/middle/hind.
NEUROMECHFLY_LEG_ORDER: tuple[str, ...] = ("lf", "lm", "lh", "rf", "rm", "rh")

#: The seven actuated joint DOFs per leg under ``ActuatedDOFPreset.LEGS_ACTIVE_ONLY``.
#: Coxa keeps all three axes; trochanterfemur drops yaw; everything distal is pitch
#: only. 7 x 6 legs = 42 actuated DOFs.
NEUROMECHFLY_DOFS_PER_LEG: tuple[str, ...] = (
    "thorax_coxa_pitch",
    "thorax_coxa_roll",
    "thorax_coxa_yaw",
    "coxa_trochanterfemur_pitch",
    "coxa_trochanterfemur_roll",
    "trochanterfemur_tibia_pitch",
    "tibia_tarsus1_pitch",
)

#: The full 42-entry actuated DOF ordering, leg-major.
NEUROMECHFLY_ACTUATED_DOF_ORDER: tuple[str, ...] = tuple(
    f"{leg}_{dof}" for leg in NEUROMECHFLY_LEG_ORDER for dof in NEUROMECHFLY_DOFS_PER_LEG
)

NEUROMECHFLY_ACTUATED_DOF_COUNT = 42
NEUROMECHFLY_ADHESION_COUNT = 6


class FlyGymBoundaryError(NeuralConfigurationError):
    """A wide boundary was declared with the wrong shape or too much capacity."""


def _assert_declared(computed: int, declared: int, label: str, maximum: int) -> int:
    """Force the caller to state the capacity, refuse a wrong statement, and cap it.

    Two checks with different jobs, and the second one used to be missing.

    The equality is about *honesty*: the caller must name the capacity it is taking,
    and a boundary whose declaration disagrees with what the constructor charged is
    refused. That number is hashed, so it cannot be walked back later.

    The ceiling is about *ordering*. Under every counting rule shipped today it is
    unreachable through these constructors -- the channel/pool and target/member
    budgets reject an over-wide boundary first, at 576 and 288 -- and an earlier
    version of this function said so and left the ceiling out. That reasoning holds
    only as long as the per-unit charges are the ones the maxima were derived from.
    Add one conditional charge inside a counting loop and the shape budgets keep
    passing while the computed total walks past the maximum being stamped into the
    identity beside it, silently: a four-line ``saturation == "tanh"`` surcharge
    produced a boundary carrying 672 free parameters under a recorded ceiling of 576.

    So the ceiling is here, and it is a backstop rather than a reachable public path.
    Its own test calls this function directly, because a check that no test can reach
    is a check nobody has read.
    """

    if isinstance(declared, bool) or not isinstance(declared, int):
        raise FlyGymBoundaryError(f"{label} declared_free_parameters must be an integer")
    if computed != declared:
        raise FlyGymBoundaryError(
            f"{label} free-parameter count mismatch: computed {computed}, declared {declared}"
        )
    if computed > maximum:
        raise FlyGymBoundaryError(
            f"{label} free-parameter budget exceeded: computed {computed}, maximum "
            f"{maximum}; the shape budgets admitted this boundary, so a per-unit "
            "charge no longer matches the rule the maximum is derived from"
        )
    return computed


class FlyGymSensoryAdapter:
    """Wide, sparse observation boundary with no motor or recurrent reference.

    Mirrors :class:`neurocraft_neural.SensoryAdapter`'s arithmetic exactly (affine,
    clip, optional tanh, then a sparse scatter onto named sensory boundary neurons)
    and produces the same ``EncodedStimulation`` the engine already validates. Only
    the budgets, the schema string and the capacity assertion differ.
    """

    def __init__(
        self,
        graph: RuntimeGraph,
        channels: Sequence[SensorChannel],
        *,
        declared_free_parameters: int,
    ) -> None:
        graph.assert_integrity()
        channel_values = tuple(channels)
        if not 1 <= len(channel_values) <= MAX_FLYGYM_SENSOR_CHANNELS:
            raise FlyGymBoundaryError(
                f"flygym sensory adapter requires 1..{MAX_FLYGYM_SENSOR_CHANNELS} channels"
            )
        names = [channel.name for channel in channel_values]
        if any(not isinstance(name, str) or not name for name in names):
            raise FlyGymBoundaryError("flygym sensor channels require non-empty names")
        if len(set(names)) != len(names):
            raise FlyGymBoundaryError("flygym sensor channel names must be unique")
        target_total = sum(len(channel.targets) for channel in channel_values)
        if target_total > MAX_FLYGYM_SENSOR_TARGETS_TOTAL:
            raise FlyGymBoundaryError("flygym sensory adapter exceeds sparse target budget")

        compiled: list[tuple[SensorChannel, tuple[tuple[int, float], ...]]] = []
        identity_channels: list[dict[str, object]] = []
        free_parameters = 0
        for channel in channel_values:
            if not isinstance(channel.unit, str) or not channel.unit:
                raise FlyGymBoundaryError(f"sensor {channel.name!r} requires a physical unit")
            if not channel.targets:
                raise FlyGymBoundaryError(
                    f"sensor {channel.name!r} has an empty boundary population"
                )
            gain = _finite_float(channel.gain, f"sensor {channel.name}.gain")
            offset = _finite_float(channel.offset, f"sensor {channel.name}.offset")
            clip_min = _finite_float(channel.clip_min, f"sensor {channel.name}.clip_min")
            clip_max = _finite_float(channel.clip_max, f"sensor {channel.name}.clip_max")
            if clip_min >= clip_max:
                raise FlyGymBoundaryError(
                    f"sensor {channel.name!r} clip_min must be below clip_max"
                )
            if any(
                abs(value) > MAX_ABSOLUTE_FLYGYM_PARAMETER
                for value in (gain, offset, clip_min, clip_max)
            ):
                raise FlyGymBoundaryError(
                    f"sensor {channel.name!r} exceeds declared parameter magnitude budget"
                )
            if channel.saturation not in {"none", "tanh"}:
                raise FlyGymBoundaryError(
                    f"sensor {channel.name!r} saturation must be 'none' or 'tanh'"
                )
            seen: set[int] = set()
            targets: list[tuple[int, float]] = []
            for target in channel.targets:
                try:
                    index = graph.index_of(target.body_id)
                except KeyError as exc:
                    raise FlyGymBoundaryError(str(exc)) from exc
                node = graph.nodes[index]
                if "sensory" not in node.roles:
                    raise FlyGymBoundaryError(
                        f"sensor {channel.name!r} target {target.body_id} "
                        "is not a named sensory node"
                    )
                if index in seen:
                    raise FlyGymBoundaryError(
                        f"sensor {channel.name!r} repeats target {target.body_id}"
                    )
                seen.add(index)
                weight = _finite_float(
                    target.weight, f"sensor {channel.name}.target[{target.body_id}]"
                )
                if weight == 0 or abs(weight) > MAX_ABSOLUTE_FLYGYM_WEIGHT:
                    raise FlyGymBoundaryError(
                        "sensor target weight must be nonzero and at most "
                        f"{MAX_ABSOLUTE_FLYGYM_WEIGHT} in magnitude"
                    )
                targets.append((index, weight))
            targets.sort()
            # Free parameters for one channel: every target weight, plus the
            # per-channel affine parameters MAX_FLYGYM_SENSORY_FREE_PARAMETERS counts.
            free_parameters += len(targets) + FLYGYM_SENSOR_CHANNEL_PARAMETERS
            compiled.append((channel, tuple(targets)))
            identity_channels.append(
                {
                    "clip_max": _float_hex(clip_max),
                    "clip_min": _float_hex(clip_min),
                    "gain": _float_hex(gain),
                    "name": channel.name,
                    "offset": _float_hex(offset),
                    "saturation": channel.saturation,
                    "targets": [
                        [graph.nodes[index].body_id, _float_hex(weight)]
                        for index, weight in targets
                    ],
                    "unit": channel.unit,
                }
            )

        self._free_parameters = _assert_declared(
            free_parameters,
            declared_free_parameters,
            "flygym sensory adapter",
            MAX_FLYGYM_SENSORY_FREE_PARAMETERS,
        )
        self._graph_identity = graph.manifest.runtime_identity_sha256
        self._node_count = len(graph.nodes)
        self._channels = tuple(compiled)
        self._expected_names = frozenset(names)
        self._identity = sha256_json(
            {
                "channels": identity_channels,
                "free_parameters": self._free_parameters,
                "graph_identity_sha256": self._graph_identity,
                "max_channels": MAX_FLYGYM_SENSOR_CHANNELS,
                "max_free_parameters": MAX_FLYGYM_SENSORY_FREE_PARAMETERS,
                "max_targets_total": MAX_FLYGYM_SENSOR_TARGETS_TOTAL,
                "schema": FLYGYM_SENSORY_SCHEMA,
            }
        )

    @property
    def identity_sha256(self) -> str:
        return self._identity

    @property
    def schema(self) -> str:
        return FLYGYM_SENSORY_SCHEMA

    @property
    def free_parameters(self) -> int:
        return self._free_parameters

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(channel.name for channel, _ in self._channels)

    def encode(self, observations: Mapping[str, float]) -> EncodedStimulation:
        if not isinstance(observations, Mapping):
            raise FlyGymBoundaryError("observations must be a numeric mapping")
        if set(observations) != self._expected_names:
            missing = sorted(self._expected_names - set(observations))
            unknown = sorted(set(observations) - self._expected_names)
            raise FlyGymBoundaryError(
                f"observation schema mismatch; missing={missing}, unknown={unknown}"
            )
        vector = [0.0] * self._node_count
        assignments: list[tuple[str, int, float]] = []
        for channel, targets in self._channels:
            raw = _finite_float(observations[channel.name], f"observation[{channel.name}]")
            value = channel.gain * raw + channel.offset
            value = min(channel.clip_max, max(channel.clip_min, value))
            if channel.saturation == "tanh":
                value = math.tanh(value)
            for index, weight in targets:
                contribution = value * weight
                vector[index] += contribution
                assignments.append((channel.name, index, contribution))
        if any(not math.isfinite(value) for value in vector):
            raise FlyGymBoundaryError("encoded sensory vector is non-finite")
        assignment_tuple = tuple(assignments)
        material = {
            "adapter_sha256": self._identity,
            "assignments": [
                [name, index, _float_hex(value)] for name, index, value in assignment_tuple
            ],
            "graph_identity_sha256": self._graph_identity,
            "vector_sha256": _float_tuple_digest(vector),
        }
        encoded = EncodedStimulation(
            graph_identity_sha256=self._graph_identity,
            adapter_sha256=self._identity,
            vector=tuple(vector),
            assignments=assignment_tuple,
            identity_sha256=sha256_json(material),
        )
        encoded.assert_integrity()
        return encoded


@dataclass(frozen=True, slots=True)
class JointBinding:
    """One actuated DOF: a motor pool, a neutral angle and a symmetric half range.

    The readout is ``target = neutral + half_range * clip(pool, -1, 1)``. Affine and
    clipped, nothing else. No phase, no oscillator, no integrator, no memory.
    """

    dof: str
    pool: MotorPool
    neutral_rad: float
    half_range_rad: float


@dataclass(frozen=True, slots=True)
class AdhesionBinding:
    """One leg's adhesion channel: ``state = clip(0.5 * (1 + pool), 0, 1)``."""

    leg: str
    pool: MotorPool


@dataclass(frozen=True, slots=True)
class FlyGymActuation:
    """The bounded actuation a NeuroMechFly body accepts for one control tick."""

    joint_targets_rad: tuple[float, ...]
    adhesion: tuple[float, ...]
    frame: MotorFrame


class FlyGymMotorDecoder:
    """Stateless wide readout: N motor pools -> N joint targets + M adhesion states.

    Deliberately *not* a pattern generator. A CPG driven by a couple of descending
    channels would make the fly walk beautifully and would prove nothing about the
    connectome, because the gait would be the CPG's. This decoder therefore holds no
    state between calls: ``decode`` is a pure function of the neural state.
    """

    def __init__(
        self,
        graph: RuntimeGraph,
        joints: Sequence[JointBinding],
        adhesion: Sequence[AdhesionBinding],
        *,
        declared_free_parameters: int,
    ) -> None:
        graph.assert_integrity()
        joint_values = tuple(joints)
        adhesion_values = tuple(adhesion)
        if not joint_values:
            raise FlyGymBoundaryError("flygym motor decoder requires at least one joint binding")
        total_pools = len(joint_values) + len(adhesion_values)
        if total_pools > MAX_FLYGYM_MOTOR_CHANNELS:
            raise FlyGymBoundaryError(
                f"flygym motor decoder requires at most {MAX_FLYGYM_MOTOR_CHANNELS} pools"
            )
        dofs = [binding.dof for binding in joint_values]
        legs = [binding.leg for binding in adhesion_values]
        if len(set(dofs)) != len(dofs):
            raise FlyGymBoundaryError("flygym joint DOF names must be unique")
        if len(set(legs)) != len(legs):
            raise FlyGymBoundaryError("flygym adhesion leg names must be unique")

        pools = [binding.pool for binding in joint_values] + [
            binding.pool for binding in adhesion_values
        ]
        names = [pool.name for pool in pools]
        if len(set(names)) != len(names):
            raise FlyGymBoundaryError("flygym motor pool names must be unique")
        if sum(len(pool.members) for pool in pools) > MAX_FLYGYM_MOTOR_MEMBERS_TOTAL:
            raise FlyGymBoundaryError("flygym motor decoder exceeds sparse member budget")

        compiled: list[tuple[str, tuple[tuple[int, float], ...], float, float, float]] = []
        identity_pools: list[dict[str, object]] = []
        free_parameters = 0
        # Every output channel must read a disjoint population. Letting one neuron
        # drive several DOFs would manufacture body-wide coordinated motion out of a
        # single channel -- a gait generated by the binding rather than by the graph.
        claimed: dict[int, str] = {}
        for pool in pools:
            if not isinstance(pool.name, str) or not pool.name:
                raise FlyGymBoundaryError("flygym motor pools require non-empty names")
            if not pool.members:
                raise FlyGymBoundaryError(f"flygym motor pool {pool.name!r} is empty")
            gain = _finite_float(pool.gain, f"motor pool {pool.name}.gain")
            dead_zone = _finite_float(pool.dead_zone, f"motor pool {pool.name}.dead_zone")
            if gain < 0 or gain > MAX_ABSOLUTE_FLYGYM_PARAMETER or not 0 <= dead_zone < 1:
                raise FlyGymBoundaryError(
                    "flygym motor gain must be within budget and dead_zone in [0, 1)"
                )
            members: list[tuple[int, float]] = []
            seen: set[int] = set()
            for member in pool.members:
                try:
                    index = graph.index_of(member.body_id)
                except KeyError as exc:
                    raise FlyGymBoundaryError(str(exc)) from exc
                node = graph.nodes[index]
                if "motor" not in node.roles and node.superclass != "descending_neuron":
                    raise FlyGymBoundaryError(
                        f"flygym motor pool {pool.name!r} member {member.body_id} "
                        "is not motor/descending"
                    )
                if index in seen:
                    raise FlyGymBoundaryError(
                        f"flygym motor pool {pool.name!r} repeats member {member.body_id}"
                    )
                seen.add(index)
                owner = claimed.get(index)
                if owner is not None:
                    raise FlyGymBoundaryError(
                        f"flygym motor pool {pool.name!r} reuses member {member.body_id}, "
                        f"already read by pool {owner!r}; output channels must read "
                        "disjoint populations"
                    )
                claimed[index] = pool.name
                weight = _finite_float(
                    member.weight, f"motor pool {pool.name}.member[{member.body_id}]"
                )
                if weight == 0 or abs(weight) > MAX_ABSOLUTE_FLYGYM_WEIGHT:
                    raise FlyGymBoundaryError(
                        "flygym motor member weight must be nonzero and at most "
                        f"{MAX_ABSOLUTE_FLYGYM_WEIGHT} in magnitude"
                    )
                members.append((index, weight))
            members.sort()
            denominator = math.fsum(abs(weight) for _, weight in members)
            if denominator == 0:
                raise FlyGymBoundaryError(f"flygym motor pool {pool.name!r} has zero denominator")
            # Free parameters for one pool: every member weight, plus the per-pool
            # parameters MAX_FLYGYM_MOTOR_FREE_PARAMETERS counts.
            free_parameters += len(members) + FLYGYM_MOTOR_POOL_PARAMETERS
            compiled.append((pool.name, tuple(members), denominator, gain, dead_zone))
            identity_pools.append(
                {
                    "dead_zone": _float_hex(dead_zone),
                    "gain": _float_hex(gain),
                    "members": [
                        [graph.nodes[index].body_id, _float_hex(weight)]
                        for index, weight in members
                    ],
                    "name": pool.name,
                }
            )

        identity_joints: list[dict[str, object]] = []
        neutral: list[float] = []
        half_range: list[float] = []
        for binding in joint_values:
            neutral_rad = _finite_float(binding.neutral_rad, f"joint {binding.dof}.neutral_rad")
            half = _finite_float(binding.half_range_rad, f"joint {binding.dof}.half_range_rad")
            if abs(neutral_rad) > math.pi or not 0.0 < half <= math.pi / 2:
                raise FlyGymBoundaryError(
                    f"joint {binding.dof!r} pose bounds are outside the declared envelope"
                )
            neutral.append(neutral_rad)
            half_range.append(half)
            # Free parameters for one DOF's affine readout, same constant the motor
            # maximum multiplies by the pool budget.
            free_parameters += FLYGYM_JOINT_READOUT_PARAMETERS
            identity_joints.append(
                {
                    "dof": binding.dof,
                    "half_range_rad": _float_hex(half),
                    "neutral_rad": _float_hex(neutral_rad),
                    "pool": binding.pool.name,
                }
            )

        self._free_parameters = _assert_declared(
            free_parameters,
            declared_free_parameters,
            "flygym motor decoder",
            MAX_FLYGYM_MOTOR_FREE_PARAMETERS,
        )
        self._graph_identity = graph.manifest.runtime_identity_sha256
        self._node_count = len(graph.nodes)
        self._pools = tuple(compiled)
        self._joint_count = len(joint_values)
        self._adhesion_count = len(adhesion_values)
        self._dof_order = tuple(dofs)
        self._leg_order = tuple(legs)
        self._channel_names = tuple(names)
        self._neutral = tuple(neutral)
        self._half_range = tuple(half_range)
        self._identity = sha256_json(
            {
                "adhesion": [
                    {"leg": binding.leg, "pool": binding.pool.name} for binding in adhesion_values
                ],
                "adhesion_readout": "clip_0_1_of_half_one_plus_pool",
                "free_parameters": self._free_parameters,
                "graph_identity_sha256": self._graph_identity,
                "joint_readout": "neutral_plus_half_range_times_clipped_pool",
                "joints": identity_joints,
                "max_free_parameters": MAX_FLYGYM_MOTOR_FREE_PARAMETERS,
                "max_members_total": MAX_FLYGYM_MOTOR_MEMBERS_TOTAL,
                "max_pools": MAX_FLYGYM_MOTOR_CHANNELS,
                "pools": identity_pools,
                "schema": FLYGYM_MOTOR_SCHEMA,
                "stateful": False,
            }
        )

    @property
    def identity_sha256(self) -> str:
        return self._identity

    @property
    def schema(self) -> str:
        return FLYGYM_MOTOR_SCHEMA

    @property
    def free_parameters(self) -> int:
        return self._free_parameters

    @property
    def channel_count(self) -> int:
        return self._joint_count + self._adhesion_count

    @property
    def channel_names(self) -> tuple[str, ...]:
        """One name per output channel: joint pools first, then adhesion pools."""

        return self._channel_names

    @property
    def channels(self) -> tuple[ChannelDeclaration, ...]:
        """One declaration per output channel, in the order ``decode`` emits them.

        ``gain`` is only bounded below by zero, and a pool with ``gain == 0.0``
        produces exactly ``0.0`` however the graph moves -- pinning its DOF to the
        neutral angle, or its leg to half adhesion, forever. That is a real way to
        put a constant inside a boundary that claims to be graph-driven, so it is
        read off the compiled pool and declared rather than left invisible. The
        readout strings are the ones the decoder already hashes.
        """

        declarations: list[ChannelDeclaration] = []
        for index, (name, _members, _denominator, gain, _dead_zone) in enumerate(self._pools):
            joint = index < self._joint_count
            if gain == 0.0:
                source = CHANNEL_SOURCE_CONFIG_CONSTANT
                derivation = (
                    "zero_gain_pool_pins_joint_to_neutral"
                    if joint
                    else "zero_gain_pool_pins_adhesion_to_half"
                )
            else:
                source = CHANNEL_SOURCE_DYNAMIC
                derivation = (
                    "neutral_plus_half_range_times_clipped_pool"
                    if joint
                    else "clip_0_1_of_half_one_plus_pool"
                )
            declarations.append(ChannelDeclaration(name=name, source=source, derivation=derivation))
        return tuple(declarations)

    @property
    def dynamic_channel_count(self) -> int:
        """Output channels the graph can actually move. Derived from the pool gains."""

        return sum(1 for entry in self.channels if entry.source == CHANNEL_SOURCE_DYNAMIC)

    @property
    def dof_order(self) -> tuple[str, ...]:
        """The actuated DOF names, in the order ``decode`` emits joint targets.

        ``FlyGymEmbodiment`` hands ``joint_targets_rad`` to ``set_actuator_inputs``
        positionally, so this order must equal the order the running skeleton
        reports. The body cross-validates it at construction; nothing here can check
        it, because this module imports nothing from ``flygym``.
        """

        return self._dof_order

    @property
    def leg_order(self) -> tuple[str, ...]:
        """The adhesion leg names, in the order ``decode`` emits adhesion states."""

        return self._leg_order

    @property
    def neutral_pose_rad(self) -> tuple[float, ...]:
        return self._neutral

    def decode(self, state: NeuralState) -> FlyGymActuation:
        if not isinstance(state, NeuralState):
            raise FlyGymBoundaryError("flygym motor decoder accepts only NeuralState")
        state.assert_integrity()
        if state.graph_identity_sha256 != self._graph_identity:
            raise NeuralIntegrityError("flygym decoder state is from a different graph")
        if len(state.activity) != self._node_count:
            raise NeuralIntegrityError("flygym decoder state length mismatch")
        values: list[tuple[str, float]] = []
        for name, members, denominator, gain, dead_zone in self._pools:
            raw = (
                gain
                * math.fsum(weight * state.activity[index] for index, weight in members)
                / denominator
            )
            if not math.isfinite(raw):
                raise NeuralIntegrityError("flygym decoder produced a non-finite raw value")
            value = min(1.0, max(-1.0, raw))
            if abs(value) < dead_zone:
                value = 0.0
            values.append((name, value))
        value_tuple = tuple(values)
        joint_targets = tuple(
            self._neutral[index] + self._half_range[index] * value_tuple[index][1]
            for index in range(self._joint_count)
        )
        adhesion = tuple(
            min(1.0, max(0.0, 0.5 * (1.0 + value_tuple[self._joint_count + index][1])))
            for index in range(self._adhesion_count)
        )
        frame = MotorFrame(
            graph_identity_sha256=self._graph_identity,
            decoder_sha256=self._identity,
            neural_state_sha256=state.state_sha256,
            values=value_tuple,
            identity_sha256=sha256_json(
                {
                    "decoder_sha256": self._identity,
                    "graph_identity_sha256": self._graph_identity,
                    "neural_state_sha256": state.state_sha256,
                    "values": [[name, _float_hex(value)] for name, value in value_tuple],
                }
            ),
        )
        return FlyGymActuation(
            joint_targets_rad=joint_targets,
            adhesion=adhesion,
            frame=frame,
        )
