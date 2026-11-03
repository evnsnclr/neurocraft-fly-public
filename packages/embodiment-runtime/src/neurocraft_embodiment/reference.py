"""A deterministic reference body with fully declared physics.

This exists so the seam is exercised, tested and traceable without adding any
dependency. It is deliberately *not* a copy of the Minecraft physics or of
``scripts/fixture_harness.py``'s ``HarnessWorld``: that world integrates in
blocks-per-tick with Minecraft's drag constants, whereas this one is SI
(metres, seconds, m/s^2), integrates with semi-implicit Euler over a declared
number of substeps per control tick, and lives in a seeded static obstacle field.
Having a second, differently parameterised world is the point — a result that
survives both bodies is not an artifact of one integrator.

Everything numeric is a named constant on :class:`ReferenceArena`, and the arena
geometry is generated once at construction from a declared SplitMix64 stream.
Nothing is sampled while stepping, so the body is a pure function of
``(config, seed, command sequence)``.

There is no policy here. The body does not know where it is going, has no target,
no reward and no scripted behaviour. It integrates the bounded command it was
handed and reports the resulting world state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from neurocraft_protocol import (
    DeterminismLevel,
    MotorVector,
    Observation,
    ProtocolError,
    RunIdentity,
    SensorFrame,
    WorldState,
)

from .identity import (
    CONSUMES_PRIMITIVE_MOTOR_VECTOR,
    BoundaryIdentity,
    EmbodimentIdentity,
    canonical_sha256,
    float_hex,
    sorted_provenance,
)
from .reference_sensing import (
    REFERENCE_DYNAMIC_SENSORY_CHANNELS,
    REFERENCE_MOTOR_AXES,
    REFERENCE_MOTOR_CHANNELS,
    REFERENCE_MOTOR_PARAMETERS,
    REFERENCE_SENSORY_CHANNELS,
    REFERENCE_SENSORY_PARAMETERS,
)
from .seam import (
    CONTROL_HZ,
    SENSOR_FRAME_SCALAR_COUNT,
    EmbodimentCommand,
    EmbodimentError,
    EmbodimentTransition,
    observation_timing,
)

REFERENCE_EMBODIMENT_ID = "reference-arena-pointmass"
REFERENCE_EMBODIMENT_VERSION = "v1"
REFERENCE_ARENA_SCHEMA = "neurocraft.embodiment.reference.arena.v1"
REFERENCE_SENSORY_SCHEMA = "neurocraft.embodiment.reference.sensory.v1"
REFERENCE_MOTOR_SCHEMA = "neurocraft.embodiment.reference.motor.v1"

#: Draw budget per declared pillar. Bounded so generation always terminates;
#: exhausting it raises rather than truncating (see :func:`_generate_pillars`).
#:
#: Raised from 64 alongside :data:`MAX_PILLAR_CROWDING`. 64 was sized when a
#: candidate could only be rejected by the spawn exclusion disc; the pairwise
#: overlap rejection added later made it too small for part of the envelope
#: ``__post_init__`` still admitted. Widening the budget cannot move an existing
#: field: :func:`_generate_pillars` stops the moment it has placed the declared
#: count, so a larger budget only extends a stream that previously ran out.
PILLAR_DRAWS_PER_PILLAR = 256

#: Ceiling on :func:`pillar_crowding`, the declaration-time packing guard.
#:
#: Rejection sampling has no closed-form completeness proof, so the guard is a
#: *necessary* condition sized from measurement rather than a sufficient one. Over a
#: sweep of 5500+ configurations (seeds, counts 2..64, radii 0.05..2.0, three arena
#: sizes, four exclusion radii) the sampler needed at most 5.5 draws per pillar at
#: crowding <= 1.0 and did not fail anywhere below crowding 2.2. Admitting at 1.0
#: against a 256-draw budget therefore leaves roughly a 2x margin in crowding and a
#: 45x margin in draws. :func:`_generate_pillars` keeps its shortfall error as the
#: fail-closed backstop for anything this necessary condition lets through.
MAX_PILLAR_CROWDING = 1.0

_UINT64_MASK = (1 << 64) - 1
_SPLITMIX64_GAMMA = 0x9E3779B97F4A7C15


def _splitmix64(state: int) -> tuple[int, int]:
    """One SplitMix64 draw: returns ``(next_state, uint64_output)``.

    Written out rather than taken from ``random`` so the arena geometry does not
    depend on any interpreter's RNG implementation. The constants are the published
    SplitMix64 ones; this is a declared, reproducible integer stream.
    """

    state = (state + _SPLITMIX64_GAMMA) & _UINT64_MASK
    mixed = state
    mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return state, mixed ^ (mixed >> 31)


@dataclass(frozen=True, slots=True)
class ReferenceArena:
    """Every constant the reference body uses. All SI. All hashed."""

    schema: str = REFERENCE_ARENA_SCHEMA
    seed: int = 0
    half_extent_x_m: float = 8.0
    half_extent_z_m: float = 8.0
    floor_y_m: float = 0.0
    ceiling_y_m: float = 6.0
    spawn_height_m: float = 1.5
    pillar_count: int = 8
    pillar_radius_m: float = 0.35
    pillar_exclusion_radius_m: float = 2.0
    ray_max_distance_m: float = 6.0
    physics_steps_per_control_tick: int = 25
    physics_timestep_s: float = 0.002
    thrust_accel_m_s2: float = 6.0
    lateral_accel_m_s2: float = 4.0
    gravity_m_s2: float = 9.81
    linear_drag_per_s: float = 1.6
    max_speed_m_s: float = 4.0
    yaw_rate_rad_s: float = 2.0
    pitch_rate_rad_s: float = 1.2
    max_pitch_rad: float = 1.4
    contact_epsilon_m: float = 1.0e-6
    # Dimensionless: compared against unit-vector components, not against a length.
    direction_epsilon: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.schema != REFERENCE_ARENA_SCHEMA:
            raise EmbodimentError("unsupported reference arena schema")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise EmbodimentError("reference arena seed must be a non-negative integer")
        positives = (
            self.half_extent_x_m,
            self.half_extent_z_m,
            self.ray_max_distance_m,
            self.physics_timestep_s,
            self.thrust_accel_m_s2,
            self.lateral_accel_m_s2,
            self.gravity_m_s2,
            self.max_speed_m_s,
            self.yaw_rate_rad_s,
            self.pitch_rate_rad_s,
            self.pillar_radius_m,
            self.pillar_exclusion_radius_m,
            self.contact_epsilon_m,
            self.direction_epsilon,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positives):
            raise EmbodimentError("reference arena scalars must be finite and positive")
        if self.linear_drag_per_s < 0.0 or not math.isfinite(self.linear_drag_per_s):
            raise EmbodimentError("linear drag must be finite and non-negative")
        if self.ceiling_y_m <= self.floor_y_m:
            raise EmbodimentError("arena ceiling must sit above its floor")
        if not self.floor_y_m < self.spawn_height_m < self.ceiling_y_m:
            raise EmbodimentError("spawn height must lie strictly inside the arena")
        if not 0.0 < self.max_pitch_rad < math.pi / 2:
            raise EmbodimentError("max pitch must lie strictly inside the protocol pitch range")
        if (
            isinstance(self.pillar_count, bool)
            or not isinstance(self.pillar_count, int)
            or not 0 <= self.pillar_count <= 64
        ):
            raise EmbodimentError("pillar count must be an integer in [0, 64]")
        # Checked whether or not any pillar is declared: ``pillar_radius_m`` is
        # hashed into the arena identity unconditionally, so leaving it unvalidated
        # at ``pillar_count == 0`` would let an impossible constant into the hash.
        if self.pillar_radius_m >= min(self.half_extent_x_m, self.half_extent_z_m):
            raise EmbodimentError("a pillar must be narrower than the arena half extent")
        if (
            isinstance(self.physics_steps_per_control_tick, bool)
            or not isinstance(self.physics_steps_per_control_tick, int)
            or self.physics_steps_per_control_tick < 1
        ):
            raise EmbodimentError("physics steps per control tick must be a positive integer")
        if self.ray_max_distance_m > 64.0:
            raise EmbodimentError("ray range exceeds the protocol sensor bound")
        crowding = pillar_crowding(self)
        if crowding > MAX_PILLAR_CROWDING:
            raise EmbodimentError(
                f"declared pillar field crowding {crowding!r} exceeds "
                f"{MAX_PILLAR_CROWDING!r}; the sampler could not reliably place "
                f"{self.pillar_count} pillars of radius {self.pillar_radius_m!r} m "
                f"outside a {self.pillar_exclusion_radius_m!r} m spawn exclusion in "
                "this arena. Widen the arena, or reduce pillar_count, "
                "pillar_radius_m or pillar_exclusion_radius_m"
            )
        self._assert_control_clock_closes()
        self._assert_protocol_envelope()

    def _assert_control_clock_closes(self) -> None:
        """The substep product must fill one control tick, refused *here* not later.

        ``EmbodimentIdentity`` already enforces this, and it remains the authority.
        The defect was one of *ordering*: ``__post_init__`` validated
        ``physics_steps_per_control_tick >= 1`` and ``physics_timestep_s > 0``
        independently, so of the 200 step counts in ``1..200`` at the default 2 ms
        timestep it admitted all 200 while exactly one -- 25 -- can be carried by a
        body. ``ReferenceArena(physics_steps_per_control_tick=1)`` and
        ``ReferenceArena(physics_timestep_s=0.001)`` both constructed cleanly and then
        raised ``EmbodimentIdentityError`` out of ``ReferenceEmbodiment.__init__``.

        An arena is a *declaration*. Admitting one that no body can carry means the
        declaration and the thing it declares disagree, and the error surfaces from a
        constructor that did not make the mistake. The rule is stated once here in the
        arena's own vocabulary and once in the identity's; a test asserts the two
        agree over a grid of ``(steps, timestep)``, so the restatement is checked
        rather than trusted.
        """

        control_period_s = 1.0 / CONTROL_HZ
        simulated_s = self.physics_steps_per_control_tick * self.physics_timestep_s
        if not math.isclose(simulated_s, control_period_s, rel_tol=1e-9, abs_tol=1e-15):
            raise EmbodimentError(
                "physics_steps_per_control_tick * physics_timestep_s must fill exactly "
                f"one {CONTROL_HZ!r} Hz control tick: "
                f"{self.physics_steps_per_control_tick} x {self.physics_timestep_s!r} s "
                f"= {simulated_s!r} s, not {control_period_s!r} s"
            )

    def _assert_protocol_envelope(self) -> None:
        """Every frame this arena's extremes can produce must be a legal protocol frame.

        Three constants were bounded by this dataclass and then bounded *again*, more
        tightly, by the protocol that receives every reading:

        * ``ray_max_distance_m`` was admitted anywhere in ``(0, 64]``, but
          ``SensorFrame`` requires at least ``0.001``. ``ray_max_distance_m=0.0009999``
          constructed, and then every single ``observe()`` raised ``ProtocolError``.
        * ``half_extent_*``, ``floor_y_m``, ``ceiling_y_m`` and ``spawn_height_m`` were
          unbounded above; positions are clamped to ``+/- 3e7``.
        * ``max_speed_m_s`` was unbounded above; velocity components are clamped to
          ``+/- 1024``. Found by widening the constructibility sweep to vary every
          field rather than four of them, and reachable: an arena wide enough for the
          body to keep accelerating reports the speed it was allowed to reach.

        Nothing here restates a protocol limit. The arena builds the two frames it can
        ever emit -- a ``SensorFrame`` and a ``WorldState`` -- at the corners of its own
        declared envelope and lets the protocol's own validators judge them, so this
        guard cannot drift from the bounds it is enforcing. It is deliberately a
        *necessary* condition on the geometry, not a claim that every frame in between
        is reachable.
        """

        for sign in (1.0, -1.0):
            position = (
                sign * self.half_extent_x_m,
                self.ceiling_y_m if sign > 0.0 else self.floor_y_m,
                sign * self.half_extent_z_m,
            )
            speed = sign * self.max_speed_m_s
            velocities = (speed, speed, speed)
            yaw = sign * math.pi
            pitch = sign * self.max_pitch_rad
            light = 15 if sign > 0.0 else 0
            ray = self.ray_max_distance_m if sign > 0.0 else 0.0
            try:
                SensorFrame(
                    position_m=position,
                    velocity_m_s=velocities,
                    yaw_rad=yaw,
                    pitch_rad=pitch,
                    on_ground=False,
                    horizontal_collision=False,
                    vertical_collision=False,
                    in_fluid=False,
                    local_light=light,
                    ray_max_distance_m=self.ray_max_distance_m,
                    ray_distances_m=(ray, ray, ray, ray, ray, ray),
                )
                WorldState(
                    position_m=position,
                    velocity_m_s=velocities,
                    yaw_rad=yaw,
                    pitch_rad=pitch,
                    on_ground=False,
                    horizontal_collision=False,
                    vertical_collision=False,
                    in_fluid=False,
                )
            except ProtocolError as exc:
                raise EmbodimentError(
                    "this reference arena declares an envelope whose extremes the "
                    f"protocol refuses, so a body in it could not report its own state: {exc}"
                ) from exc

    def to_identity_material(self) -> dict[str, Any]:
        floats = {
            name: float_hex(getattr(self, name))
            for name in (
                "ceiling_y_m",
                "contact_epsilon_m",
                "direction_epsilon",
                "floor_y_m",
                "gravity_m_s2",
                "half_extent_x_m",
                "half_extent_z_m",
                "lateral_accel_m_s2",
                "linear_drag_per_s",
                "max_pitch_rad",
                "max_speed_m_s",
                "physics_timestep_s",
                "pillar_exclusion_radius_m",
                "pillar_radius_m",
                "pitch_rate_rad_s",
                "ray_max_distance_m",
                "spawn_height_m",
                "thrust_accel_m_s2",
                "yaw_rate_rad_s",
            )
        }
        return {
            **floats,
            "integrator": "semi_implicit_euler_fixed_substep",
            "physics_steps_per_control_tick": self.physics_steps_per_control_tick,
            "pillar_count": self.pillar_count,
            "prng": "splitmix64_construction_time_only",
            "schema": self.schema,
            "seed": self.seed,
            "units": "si_metres_seconds",
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.to_identity_material())


def pillar_crowding(arena: ReferenceArena) -> float:
    """How much of the usable centre region a declared pillar field lays claim to.

    ``_generate_pillars`` draws candidate *centres* uniformly from the rectangle
    ``[-usable_x, usable_x] x [-usable_z, usable_z]`` and rejects a candidate inside
    the spawn exclusion disc or within ``2 * pillar_radius_m`` of a placed centre. So
    by the time the last pillar is drawn, the forbidden centre region is the spawn
    disc plus one disc of radius ``2 * pillar_radius_m`` per already-placed pillar.

    This returns that forbidden area over the rectangle's area, counting every disc
    at full area and as if wholly inside the rectangle. Both approximations
    *over*-count, which is the safe direction for a rejection guard: the true
    acceptance region is never smaller than this number implies.

    Returns ``0.0`` for a pillar-free arena, which forbids nothing.
    """

    if arena.pillar_count == 0:
        return 0.0
    usable_x = arena.half_extent_x_m - arena.pillar_radius_m
    usable_z = arena.half_extent_z_m - arena.pillar_radius_m
    if usable_x <= 0.0 or usable_z <= 0.0:
        return math.inf
    forbidden = (arena.pillar_count - 1) * 4.0 * math.pi * arena.pillar_radius_m**2
    forbidden += math.pi * arena.pillar_exclusion_radius_m**2
    return forbidden / (4.0 * usable_x * usable_z)


@dataclass(frozen=True, slots=True)
class ReferenceBodyState:
    """Immutable body state; every substep returns a new instance."""

    x_m: float
    y_m: float
    z_m: float
    velocity_x_m_s: float
    velocity_y_m_s: float
    velocity_z_m_s: float
    yaw_rad: float
    pitch_rad: float
    on_ground: bool
    horizontal_collision: bool
    vertical_collision: bool


def _wrap_radians(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _generate_pillars(arena: ReferenceArena) -> tuple[tuple[float, float], ...]:
    """Place exactly ``arena.pillar_count`` static pillars, or fail closed.

    Rejection sampling from the declared SplitMix64 stream. A candidate is discarded
    when it lands inside the spawn exclusion disc, and when it would overlap a pillar
    already placed. The second rule is what makes ``_substep``'s per-pillar push-out
    well defined: overlapping cylinders have no single nearest surface, so the
    correction would silently depend on iteration order.

    The draw budget is bounded, so the function always terminates and always returns
    the same field for a given seed. Exhausting it is an *error*, never a silent
    truncation: an arena that placed fewer pillars than it declared would be hashed
    under an identity claiming a world it does not contain.
    """

    if arena.pillar_count == 0:
        return ()
    state = arena.seed & _UINT64_MASK
    placed: list[tuple[float, float]] = []
    separation = 2.0 * arena.pillar_radius_m
    max_draws = PILLAR_DRAWS_PER_PILLAR * arena.pillar_count
    usable_x = arena.half_extent_x_m - arena.pillar_radius_m
    usable_z = arena.half_extent_z_m - arena.pillar_radius_m
    for _ in range(max_draws):
        if len(placed) == arena.pillar_count:
            break
        state, raw_x = _splitmix64(state)
        state, raw_z = _splitmix64(state)
        # 53-bit mantissa draw in [0, 1), then mapped onto the usable arena square.
        unit_x = (raw_x >> 11) * (2.0**-53)
        unit_z = (raw_z >> 11) * (2.0**-53)
        candidate_x = (2.0 * unit_x - 1.0) * usable_x
        candidate_z = (2.0 * unit_z - 1.0) * usable_z
        if math.hypot(candidate_x, candidate_z) < arena.pillar_exclusion_radius_m:
            continue
        if any(
            math.hypot(candidate_x - other_x, candidate_z - other_z) < separation
            for other_x, other_z in placed
        ):
            continue
        placed.append((candidate_x, candidate_z))
    if len(placed) != arena.pillar_count:
        raise EmbodimentError(
            f"reference arena declares {arena.pillar_count} pillars but the sampler "
            f"placed {len(placed)} in {max_draws} draws; widen the arena, or reduce "
            "pillar_exclusion_radius_m, pillar_radius_m or pillar_count"
        )
    return tuple(placed)


def _ray_distance(
    arena: ReferenceArena,
    pillars: tuple[tuple[float, float], ...],
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
) -> float:
    """Analytic distance to the arena box or the nearest pillar, clipped to range."""

    limit = arena.ray_max_distance_m
    bounds = (
        (origin[0], direction[0], -arena.half_extent_x_m, arena.half_extent_x_m),
        (origin[1], direction[1], arena.floor_y_m, arena.ceiling_y_m),
        (origin[2], direction[2], -arena.half_extent_z_m, arena.half_extent_z_m),
    )
    for start, delta, low, high in bounds:
        if abs(delta) <= arena.direction_epsilon:
            continue
        target = high if delta > 0.0 else low
        distance = (target - start) / delta
        if 0.0 <= distance < limit:
            limit = distance

    # Pillars are vertical cylinders; intersect in the xz-plane only.
    dir_x, dir_z = direction[0], direction[2]
    planar = dir_x * dir_x + dir_z * dir_z
    if planar > arena.direction_epsilon:
        for pillar_x, pillar_z in pillars:
            offset_x = origin[0] - pillar_x
            offset_z = origin[2] - pillar_z
            half_b = offset_x * dir_x + offset_z * dir_z
            c = offset_x * offset_x + offset_z * offset_z - arena.pillar_radius_m**2
            discriminant = half_b * half_b - planar * c
            if discriminant < 0.0:
                continue
            root = math.sqrt(discriminant)
            for numerator in (-half_b - root, -half_b + root):
                distance = numerator / planar
                if 0.0 <= distance < limit:
                    limit = distance
    return max(0.0, min(arena.ray_max_distance_m, limit))


class ReferenceEmbodiment:
    """Seeded point-mass body in a bounded box with a static pillar field.

    Consumes the primitive ``MotorVector``; the ``NeuralState`` on the command is
    ignored, exactly as declared by ``motor_currency`` in the identity.
    """

    def __init__(self, arena: ReferenceArena | None = None) -> None:
        self._arena = arena or ReferenceArena()
        self._pillars = _generate_pillars(self._arena)
        self._state = self._initial_state()
        self._identity = self._build_identity()

    def _initial_state(self) -> ReferenceBodyState:
        return ReferenceBodyState(
            x_m=0.0,
            y_m=self._arena.spawn_height_m,
            z_m=0.0,
            velocity_x_m_s=0.0,
            velocity_y_m_s=0.0,
            velocity_z_m_s=0.0,
            yaw_rad=0.0,
            pitch_rad=0.0,
            on_ground=False,
            horizontal_collision=False,
            vertical_collision=False,
        )

    def _build_identity(self) -> EmbodimentIdentity:
        arena_hash = self._arena.identity_sha256
        sensory = BoundaryIdentity(
            role="sensory",
            schema=REFERENCE_SENSORY_SCHEMA,
            identity_sha256=canonical_sha256(
                {
                    "arena_sha256": arena_hash,
                    "channels": [
                        entry.to_identity_material() for entry in REFERENCE_SENSORY_CHANNELS
                    ],
                    "light_model": "linear_height_gradient_floor_to_ceiling_16_levels",
                    "parameters": list(REFERENCE_SENSORY_PARAMETERS),
                    "ray_frame": "yaw_rotated_body_frame_pitch_not_applied",
                    "ray_order": ["forward", "back", "left", "right", "up", "down"],
                    "schema": REFERENCE_SENSORY_SCHEMA,
                }
            ),
            # The width comes from the protocol frame; the records come from this
            # body's own sourcing table. Two derivations, so the equality can fire.
            channel_count=SENSOR_FRAME_SCALAR_COUNT,
            channels=REFERENCE_SENSORY_CHANNELS,
            declared_dynamic_channels=REFERENCE_DYNAMIC_SENSORY_CHANNELS,
            free_parameters=len(REFERENCE_SENSORY_PARAMETERS),
        )
        motor = BoundaryIdentity(
            role="motor",
            schema=REFERENCE_MOTOR_SCHEMA,
            identity_sha256=canonical_sha256(
                {
                    "arena_sha256": arena_hash,
                    "axes": list(REFERENCE_MOTOR_AXES),
                    "channels": [
                        entry.to_identity_material() for entry in REFERENCE_MOTOR_CHANNELS
                    ],
                    "mapping": "bounded_axis_to_declared_acceleration_or_rate",
                    "parameters": list(REFERENCE_MOTOR_PARAMETERS),
                    "schema": REFERENCE_MOTOR_SCHEMA,
                    "stateful": False,
                }
            ),
            # The width comes from the protocol MotorVector; the records come from
            # this body's own axis ordering. A renamed or added axis fires here.
            channel_count=len(MotorVector.FIELDS),
            channels=REFERENCE_MOTOR_CHANNELS,
            declared_dynamic_channels=len(REFERENCE_MOTOR_AXES),
            free_parameters=len(REFERENCE_MOTOR_PARAMETERS),
        )
        return EmbodimentIdentity(
            embodiment_id=REFERENCE_EMBODIMENT_ID,
            embodiment_version=REFERENCE_EMBODIMENT_VERSION,
            configuration_sha256=arena_hash,
            sensory_boundary=sensory,
            motor_boundary=motor,
            motor_currency=CONSUMES_PRIMITIVE_MOTOR_VECTOR,
            seed=self._arena.seed,
            control_hz=CONTROL_HZ,
            physics_steps_per_control_tick=self._arena.physics_steps_per_control_tick,
            physics_timestep_s=self._arena.physics_timestep_s,
            determinism_level=DeterminismLevel.SAME_BUILD_EXACT,
            provenance=sorted_provenance(
                {
                    "arena_sha256": arena_hash,
                    "integrator": "semi_implicit_euler_fixed_substep",
                    "pillar_count": str(len(self._pillars)),
                    "prng": "splitmix64",
                    "units": "si_metres_seconds",
                }
            ),
        )

    @property
    def identity(self) -> EmbodimentIdentity:
        return self._identity

    @property
    def arena(self) -> ReferenceArena:
        return self._arena

    @property
    def pillars(self) -> tuple[tuple[float, float], ...]:
        return self._pillars

    @property
    def state(self) -> ReferenceBodyState:
        return self._state

    def reset(self) -> None:
        self._state = self._initial_state()

    def provenance(self) -> Mapping[str, str]:
        return self._identity.provenance_mapping()

    def observe(self, *, identity: RunIdentity, sequence: int) -> Observation:
        game_tick, sent_monotonic_ns = observation_timing(sequence)
        state = self._state
        origin = (state.x_m, state.y_m, state.z_m)
        cos_yaw = math.cos(state.yaw_rad)
        sin_yaw = math.sin(state.yaw_rad)
        # Protocol ray order: forward, back, left, right, up, down.
        #
        # Handedness, ADR 0015. This body shares the Minecraft basis (+x east, +y up,
        # +z south) and its yaw convention, so it must share its handedness with
        # ``FlySensors``: at yaw 0 forward is (0, 0, +1) south, and the body's right is
        # then west, (-1, 0, 0) -- ``forward x up``. The left slot previously held
        # (-cos, 0, -sin), which is west at yaw 0, so the two lateral slots were
        # transposed here in exactly the way they were in the Java body.
        directions = (
            (-sin_yaw, 0.0, cos_yaw),
            (sin_yaw, 0.0, -cos_yaw),
            (cos_yaw, 0.0, sin_yaw),
            (-cos_yaw, 0.0, -sin_yaw),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
        )
        rays = tuple(
            _ray_distance(self._arena, self._pillars, origin, direction) for direction in directions
        )
        span = self._arena.ceiling_y_m - self._arena.floor_y_m
        light_fraction = (state.y_m - self._arena.floor_y_m) / span
        local_light = max(0, min(15, round(15.0 * light_fraction)))
        return Observation(
            identity=identity,
            observation_seq=sequence,
            game_tick=game_tick,
            sent_monotonic_ns=sent_monotonic_ns,
            sensors=SensorFrame(
                position_m=(state.x_m, state.y_m, state.z_m),
                velocity_m_s=(
                    state.velocity_x_m_s,
                    state.velocity_y_m_s,
                    state.velocity_z_m_s,
                ),
                yaw_rad=state.yaw_rad,
                pitch_rad=state.pitch_rad,
                on_ground=state.on_ground,
                horizontal_collision=state.horizontal_collision,
                vertical_collision=state.vertical_collision,
                in_fluid=False,
                local_light=local_light,
                ray_max_distance_m=self._arena.ray_max_distance_m,
                ray_distances_m=(rays[0], rays[1], rays[2], rays[3], rays[4], rays[5]),
            ),
        )

    def apply(self, command: EmbodimentCommand) -> EmbodimentTransition:
        if not isinstance(command, EmbodimentCommand):
            raise EmbodimentError("reference embodiment accepts only an EmbodimentCommand")
        arena = self._arena
        state = self._state
        horizontal_collision = False
        vertical_collision = False
        for _ in range(arena.physics_steps_per_control_tick):
            state, hit_wall, hit_floor = self._substep(state, command.motors)
            horizontal_collision = horizontal_collision or hit_wall
            vertical_collision = vertical_collision or hit_floor
        state = replace(
            state,
            horizontal_collision=horizontal_collision,
            vertical_collision=vertical_collision,
        )
        self._state = state
        return EmbodimentTransition(
            world_after=WorldState(
                position_m=(state.x_m, state.y_m, state.z_m),
                velocity_m_s=(
                    state.velocity_x_m_s,
                    state.velocity_y_m_s,
                    state.velocity_z_m_s,
                ),
                yaw_rad=state.yaw_rad,
                pitch_rad=state.pitch_rad,
                on_ground=state.on_ground,
                horizontal_collision=state.horizontal_collision,
                vertical_collision=state.vertical_collision,
                in_fluid=False,
            ),
            passive_forces=(0.0, -arena.gravity_m_s2, 0.0),
            external_forces=(0.0, 0.0, 0.0),
            physics_steps=arena.physics_steps_per_control_tick,
            simulated_time_s=arena.physics_steps_per_control_tick * arena.physics_timestep_s,
        )

    def _substep(
        self,
        state: ReferenceBodyState,
        motors: MotorVector,
    ) -> tuple[ReferenceBodyState, bool, bool]:
        arena = self._arena
        dt = arena.physics_timestep_s
        yaw = _wrap_radians(state.yaw_rad + arena.yaw_rate_rad_s * motors.yaw_torque * dt)
        pitch = max(
            -arena.max_pitch_rad,
            min(
                arena.max_pitch_rad,
                state.pitch_rad + arena.pitch_rate_rad_s * motors.pitch_torque * dt,
            ),
        )
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        # Handedness, ADR 0015 -- the actuation half of the same frame ``observe`` senses
        # in, and it must use the same axes. Forward is (-sin, 0, cos), south at yaw 0.
        # The body's right is ``forward x up`` = (-cos, 0, -sin), west at yaw 0 and north
        # at yaw 90. The lateral axis here was (cos, sin) -- east at yaw 0, the body's
        # *left*, character-for-character the convention ``BodyFrameMath`` was corrected
        # away from. Positive ``lateral_thrust`` now accelerates the body toward its
        # declared right, matching ``BodyFrameMath.toWorld`` and slot 3 of ``observe``.
        accel_x = (
            -sin_yaw * motors.forward_thrust * arena.thrust_accel_m_s2
            - cos_yaw * motors.lateral_thrust * arena.lateral_accel_m_s2
        )
        accel_z = (
            cos_yaw * motors.forward_thrust * arena.thrust_accel_m_s2
            - sin_yaw * motors.lateral_thrust * arena.lateral_accel_m_s2
        )
        accel_y = motors.vertical_thrust * arena.thrust_accel_m_s2 - arena.gravity_m_s2
        drag = arena.linear_drag_per_s
        velocity_x = state.velocity_x_m_s + (accel_x - drag * state.velocity_x_m_s) * dt
        velocity_y = state.velocity_y_m_s + (accel_y - drag * state.velocity_y_m_s) * dt
        velocity_z = state.velocity_z_m_s + (accel_z - drag * state.velocity_z_m_s) * dt
        speed = math.sqrt(velocity_x**2 + velocity_y**2 + velocity_z**2)
        if speed > arena.max_speed_m_s:
            scale = arena.max_speed_m_s / speed
            velocity_x *= scale
            velocity_y *= scale
            velocity_z *= scale

        x = state.x_m + velocity_x * dt
        y = state.y_m + velocity_y * dt
        z = state.z_m + velocity_z * dt

        hit_wall = False
        if x < -arena.half_extent_x_m or x > arena.half_extent_x_m:
            x = max(-arena.half_extent_x_m, min(arena.half_extent_x_m, x))
            velocity_x = 0.0
            hit_wall = True
        if z < -arena.half_extent_z_m or z > arena.half_extent_z_m:
            z = max(-arena.half_extent_z_m, min(arena.half_extent_z_m, z))
            velocity_z = 0.0
            hit_wall = True

        for pillar_x, pillar_z in self._pillars:
            offset_x = x - pillar_x
            offset_z = z - pillar_z
            distance = math.hypot(offset_x, offset_z)
            if distance >= arena.pillar_radius_m:
                continue
            hit_wall = True
            if distance <= arena.contact_epsilon_m:
                # Exactly on the axis: push out along +x, a declared tie-break.
                x = pillar_x + arena.pillar_radius_m
                velocity_x = 0.0
                continue
            normal_x = offset_x / distance
            normal_z = offset_z / distance
            x = pillar_x + normal_x * arena.pillar_radius_m
            z = pillar_z + normal_z * arena.pillar_radius_m
            radial = velocity_x * normal_x + velocity_z * normal_z
            if radial < 0.0:
                velocity_x -= radial * normal_x
                velocity_z -= radial * normal_z

        hit_floor = False
        on_ground = False
        if y <= arena.floor_y_m:
            y = arena.floor_y_m
            velocity_y = 0.0
            hit_floor = True
            on_ground = True
        elif y >= arena.ceiling_y_m:
            y = arena.ceiling_y_m
            velocity_y = 0.0
            hit_floor = True

        return (
            ReferenceBodyState(
                x_m=x,
                y_m=y,
                z_m=z,
                velocity_x_m_s=velocity_x,
                velocity_y_m_s=velocity_y,
                velocity_z_m_s=velocity_z,
                yaw_rad=yaw,
                pitch_rad=pitch,
                on_ground=on_ground,
                horizontal_collision=hit_wall,
                vertical_collision=hit_floor,
            ),
            hit_wall,
            hit_floor,
        )
