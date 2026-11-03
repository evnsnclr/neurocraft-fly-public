from __future__ import annotations

import math

import pytest
from neurocraft_connectome import RuntimeGraph
from neurocraft_controller import ConnectomeController
from neurocraft_embodiment import (
    CHANNEL_SOURCE_CONFIG_CONSTANT,
    CHANNEL_SOURCE_DYNAMIC,
    CHANNEL_SOURCE_LITERAL_CONSTANT,
    CONSUMES_PRIMITIVE_MOTOR_VECTOR,
    Embodiment,
    EmbodimentCommand,
    EmbodimentError,
    ReferenceArena,
    ReferenceEmbodiment,
)
from neurocraft_protocol import DeterminismLevel, MotorVector, Observation


def drive(
    controller: ConnectomeController,
    embodiment: ReferenceEmbodiment,
    ticks: int,
) -> list[tuple[float, float, float]]:
    positions: list[tuple[float, float, float]] = []
    for tick in range(ticks):
        observation = embodiment.observe(identity=controller.identity, sequence=tick)
        action = controller.step(observation, command_seq=tick)
        transition = embodiment.apply(
            EmbodimentCommand(
                command_seq=tick,
                motors=action.motors,
                neural_state=controller.engine.current_state(),
            )
        )
        positions.append(transition.world_after.position_m)
    return positions


def test_declared_capacity_counts_only_real_parameters(
    embodiment: ReferenceEmbodiment,
) -> None:
    """Schema strings and unit labels are not free parameters.

    The counts are asserted alongside the exact contents, because counts alone were
    what let the classification drift: 11 and 4 were both stable and both wrong. See
    ``test_every_arena_constant_any_code_path_reads_is_classified`` for the
    whole-module AST check that keeps the contents honest, and
    ``test_parameter_classification_matches_what_each_path_dereferences`` for the
    per-path split.
    """

    from neurocraft_embodiment.reference_sensing import (
        REFERENCE_MOTOR_PARAMETERS,
        REFERENCE_SENSORY_PARAMETERS,
    )

    arena = embodiment.arena
    for name in (*REFERENCE_SENSORY_PARAMETERS, *REFERENCE_MOTOR_PARAMETERS):
        assert isinstance(getattr(arena, name), (int, float))
        assert not isinstance(getattr(arena, name), bool)
    assert REFERENCE_SENSORY_PARAMETERS == (
        "ceiling_y_m",
        "direction_epsilon",
        "floor_y_m",
        "half_extent_x_m",
        "half_extent_z_m",
        "pillar_count",
        "pillar_exclusion_radius_m",
        "pillar_radius_m",
        "ray_max_distance_m",
        "seed",
        "spawn_height_m",
    )
    assert REFERENCE_MOTOR_PARAMETERS == (
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
        "pillar_count",
        "pillar_exclusion_radius_m",
        "pillar_radius_m",
        "pitch_rate_rad_s",
        "seed",
        "spawn_height_m",
        "thrust_accel_m_s2",
        "yaw_rate_rad_s",
    )
    assert embodiment.identity.sensory_boundary.free_parameters == 11
    assert embodiment.identity.motor_boundary.free_parameters == 20


def test_reference_embodiment_satisfies_the_seam(embodiment: ReferenceEmbodiment) -> None:
    assert isinstance(embodiment, Embodiment)
    identity = embodiment.identity
    assert identity.motor_currency == CONSUMES_PRIMITIVE_MOTOR_VECTOR
    assert identity.determinism_level is DeterminismLevel.SAME_BUILD_EXACT
    assert identity.control_hz == 20.0
    assert len(identity.identity_sha256) == 64
    assert "integrator" in embodiment.provenance()


def test_observation_is_a_valid_protocol_frame(
    controller: ConnectomeController,
    embodiment: ReferenceEmbodiment,
) -> None:
    observation = embodiment.observe(identity=controller.identity, sequence=7)
    assert isinstance(observation, Observation)
    assert observation.observation_seq == 7
    assert observation.game_tick == 7
    assert observation.sent_monotonic_ns == 7 * 50_000_000
    sensors = observation.sensors
    assert len(sensors.ray_distances_m) == 6
    assert all(0.0 <= ray <= sensors.ray_max_distance_m for ray in sensors.ray_distances_m)
    assert 0 <= sensors.local_light <= 15


def test_run_is_deterministic_for_a_fixed_seed(
    graph: RuntimeGraph,
) -> None:
    first = drive(
        ConnectomeController.create(graph, run_id="det-a", entity_id="fly"),
        ReferenceEmbodiment(ReferenceArena(seed=11)),
        24,
    )
    second = drive(
        ConnectomeController.create(graph, run_id="det-a", entity_id="fly"),
        ReferenceEmbodiment(ReferenceArena(seed=11)),
        24,
    )
    assert first == second


def test_seed_changes_the_arena_and_therefore_the_identity() -> None:
    a = ReferenceEmbodiment(ReferenceArena(seed=0))
    b = ReferenceEmbodiment(ReferenceArena(seed=1))
    assert a.pillars != b.pillars
    assert a.identity.identity_sha256 != b.identity.identity_sha256


def test_pillar_field_is_generated_once_and_respects_the_spawn_exclusion() -> None:
    arena = ReferenceArena(seed=5)
    body = ReferenceEmbodiment(arena)
    assert len(body.pillars) == arena.pillar_count
    assert body.pillars == ReferenceEmbodiment(arena).pillars
    for x, z in body.pillars:
        assert math.hypot(x, z) >= arena.pillar_exclusion_radius_m
        assert abs(x) <= arena.half_extent_x_m
        assert abs(z) <= arena.half_extent_z_m


def test_reset_returns_the_body_to_its_declared_initial_state(
    controller: ConnectomeController,
    embodiment: ReferenceEmbodiment,
) -> None:
    initial = embodiment.state
    drive(controller, embodiment, 5)
    assert embodiment.state != initial
    embodiment.reset()
    assert embodiment.state == initial


def test_body_stays_inside_the_arena_under_saturated_command(
    embodiment: ReferenceEmbodiment,
    graph: RuntimeGraph,
) -> None:
    controller = ConnectomeController.create(graph, run_id="bounds", entity_id="fly")
    arena = embodiment.arena
    # Saturated on every axis, but this asserts containment only: it stayed green with the
    # lateral axis pointing at the body's left (ADR 0015). Direction is asserted by
    # test_lateral_thrust_accelerates_toward_the_bodys_declared_right.
    for tick in range(60):
        observation = embodiment.observe(identity=controller.identity, sequence=tick)
        controller.step(observation, command_seq=tick)
        transition = embodiment.apply(
            EmbodimentCommand(
                command_seq=tick,
                motors=MotorVector(
                    forward_thrust=1.0,
                    lateral_thrust=1.0,
                    vertical_thrust=1.0,
                    yaw_torque=1.0,
                    pitch_torque=1.0,
                ),
                neural_state=controller.engine.current_state(),
            )
        )
        x, y, z = transition.world_after.position_m
        assert -arena.half_extent_x_m - 1e-9 <= x <= arena.half_extent_x_m + 1e-9
        assert arena.floor_y_m - 1e-9 <= y <= arena.ceiling_y_m + 1e-9
        assert -arena.half_extent_z_m - 1e-9 <= z <= arena.half_extent_z_m + 1e-9


def test_body_falls_under_declared_gravity_with_no_command(
    controller: ConnectomeController,
    embodiment: ReferenceEmbodiment,
) -> None:
    start = embodiment.state.y_m
    for tick in range(10):
        observation = embodiment.observe(identity=controller.identity, sequence=tick)
        controller.step(observation, command_seq=tick)
        transition = embodiment.apply(
            EmbodimentCommand(
                command_seq=tick,
                motors=MotorVector.neutral(),
                neural_state=controller.engine.current_state(),
            )
        )
    assert transition.world_after.position_m[1] < start
    assert transition.passive_forces == (0.0, -embodiment.arena.gravity_m_s2, 0.0)


def test_transition_advances_exactly_one_control_tick(
    controller: ConnectomeController,
    embodiment: ReferenceEmbodiment,
) -> None:
    observation = embodiment.observe(identity=controller.identity, sequence=0)
    controller.step(observation, command_seq=0)
    transition = embodiment.apply(
        EmbodimentCommand(
            command_seq=0,
            motors=MotorVector.neutral(),
            neural_state=controller.engine.current_state(),
        )
    )
    assert transition.physics_steps == embodiment.arena.physics_steps_per_control_tick
    assert math.isclose(transition.simulated_time_s, 1.0 / 20.0, rel_tol=1e-12)


def test_apply_rejects_anything_but_an_embodiment_command(
    embodiment: ReferenceEmbodiment,
) -> None:
    with pytest.raises(EmbodimentError, match="EmbodimentCommand"):
        embodiment.apply(MotorVector.neutral())  # type: ignore[arg-type]


def test_arena_rejects_a_configuration_outside_the_protocol_envelope() -> None:
    with pytest.raises(EmbodimentError, match="ray range"):
        ReferenceArena(ray_max_distance_m=128.0)
    with pytest.raises(EmbodimentError, match="ceiling"):
        ReferenceArena(floor_y_m=2.0, ceiling_y_m=1.0)
    with pytest.raises(EmbodimentError, match="spawn height"):
        ReferenceArena(spawn_height_m=99.0)


def test_body_never_penetrates_a_pillar(graph: RuntimeGraph) -> None:
    """Exercises the cylinder collision branch and asserts the hard invariant.

    Seed 4 places the single pillar close enough to the spawn that a saturated
    forward+yaw command drives the body into it. The body must be pushed exactly to
    the surface and never inside it.
    """

    arena = ReferenceArena(
        seed=4,
        pillar_count=1,
        pillar_radius_m=1.2,
        pillar_exclusion_radius_m=1.3,
        half_extent_x_m=3.0,
        half_extent_z_m=3.0,
    )
    body = ReferenceEmbodiment(arena)
    controller = ConnectomeController.create(graph, run_id="pillar", entity_id="fly")
    pillar_x, pillar_z = body.pillars[0]
    controller.step(body.observe(identity=controller.identity, sequence=0), command_seq=0)
    state = controller.engine.current_state()

    contacted = False
    minimum_distance = math.inf
    for tick in range(120):
        transition = body.apply(
            EmbodimentCommand(
                command_seq=tick,
                motors=MotorVector(forward_thrust=1.0, vertical_thrust=1.0, yaw_torque=1.0),
                neural_state=state,
            )
        )
        x, _, z = transition.world_after.position_m
        distance = math.hypot(x - pillar_x, z - pillar_z)
        minimum_distance = min(minimum_distance, distance)
        assert distance >= arena.pillar_radius_m - 1e-9
        contacted = contacted or transition.world_after.horizontal_collision
    assert contacted
    assert minimum_distance == pytest.approx(arena.pillar_radius_m, abs=1e-9)


def test_rays_shorten_when_a_pillar_is_directly_ahead(graph: RuntimeGraph) -> None:
    """The forward ray must see an occluder, not just the arena wall."""

    empty = ReferenceArena(seed=0, pillar_count=0, half_extent_x_m=3.0, half_extent_z_m=3.0)
    occluded = ReferenceArena(
        seed=4,
        pillar_count=1,
        pillar_radius_m=1.2,
        pillar_exclusion_radius_m=1.3,
        half_extent_x_m=3.0,
        half_extent_z_m=3.0,
    )
    controller = ConnectomeController.create(graph, run_id="rays", entity_id="fly")
    open_rays = (
        ReferenceEmbodiment(empty)
        .observe(identity=controller.identity, sequence=0)
        .sensors.ray_distances_m
    )
    blocked_rays = (
        ReferenceEmbodiment(occluded)
        .observe(identity=controller.identity, sequence=0)
        .sensors.ray_distances_m
    )
    assert blocked_rays != open_rays
    assert min(blocked_rays[:4]) < min(open_rays[:4])


#: Ray slots in protocol order: forward, back, left, right, up, down.
_LEFT_SLOT = 2
_RIGHT_SLOT = 3


def _lateral_pillar_arena(seed: int) -> ReferenceArena:
    """One fat pillar, found by search to sit almost exactly on the body's lateral axis."""

    return ReferenceArena(
        seed=seed,
        pillar_count=1,
        pillar_radius_m=1.2,
        pillar_exclusion_radius_m=1.3,
        half_extent_x_m=6.0,
        half_extent_z_m=6.0,
    )


@pytest.mark.parametrize(
    ("seed", "expected_slot", "expected_sign", "side"),
    [(298, _RIGHT_SLOT, -1.0, "right"), (366, _LEFT_SLOT, 1.0, "left")],
)
def test_lateral_ray_slots_follow_the_body_handedness(
    graph: RuntimeGraph, seed: int, expected_slot: int, expected_sign: float, side: str
) -> None:
    """ADR 0015. The lateral slots must name the body's own sides, as Minecraft's do.

    This body shares Minecraft's basis (+x east, +y up, +z south) and its yaw
    convention, and it is the Python twin of the Java ``FlySensors`` body, so the two
    must agree on which slot is which or the same connectome would fly mirrored in the
    two worlds. At the spawn state -- origin, yaw 0 -- forward is ``(-sin 0, 0, cos 0)``
    ``= (0, 0, +1)``, south. A body facing south has its right to the west: that is
    ``forward x up = (0,0,1) x (0,1,0) = (-1, 0, 0)``, and it is what the engine's own
    ``Direction.SOUTH.getClockWise()`` table returns. So a pillar at negative x is on the
    body's RIGHT and a pillar at positive x is on its LEFT.

    Before this lane the left slot carried ``(-cos, 0, -sin)``, which is west at yaw 0,
    so both these cases reported the opposite slot. No test read the two slots against a
    known-side occluder, which is why the transposition survived here as long as it did
    in the Java body.
    """

    arena = _lateral_pillar_arena(seed)
    body = ReferenceEmbodiment(arena)
    pillar_x, pillar_z = body.pillars[0]

    # The probe measures what it claims: the pillar really is squarely lateral, on the
    # side the case says, and near enough to be seen inside the ray range.
    assert math.copysign(1.0, pillar_x) == expected_sign
    assert abs(pillar_z) < 0.35, "pillar is not squarely on the lateral axis"
    assert abs(pillar_x) - arena.pillar_radius_m < arena.ray_max_distance_m

    controller = ConnectomeController.create(graph, run_id="handedness", entity_id="fly")
    open_rays = (
        ReferenceEmbodiment(
            ReferenceArena(seed=seed, pillar_count=0, half_extent_x_m=6.0, half_extent_z_m=6.0)
        )
        .observe(identity=controller.identity, sequence=0)
        .sensors.ray_distances_m
    )
    rays = body.observe(identity=controller.identity, sequence=0).sensors.ray_distances_m

    shortened = {slot for slot in (_LEFT_SLOT, _RIGHT_SLOT) if rays[slot] < open_rays[slot] - 1e-9}
    assert shortened == {expected_slot}, (
        f"a pillar on the body's {side} must shorten the declared {side} ray; "
        f"slots that moved were {sorted(shortened)}"
    )
    # The reading is the analytic distance to the cylinder, not merely "something smaller".
    assert rays[expected_slot] == pytest.approx(
        abs(pillar_x) - math.sqrt(arena.pillar_radius_m**2 - pillar_z**2), abs=1e-9
    )
    assert rays[:2] == open_rays[:2] and rays[4:] == open_rays[4:]


@pytest.mark.parametrize("turn_ticks", [0, 5, 13])
def test_lateral_thrust_accelerates_toward_the_bodys_declared_right(
    controller: ConnectomeController, turn_ticks: int
) -> None:
    """ADR 0015, the actuation half of the same frame the test above senses in.

    Sensing and actuation must name the same right. ``_substep`` builds forward as
    ``(-sin, 0, cos)`` -- south at yaw 0 -- so its lateral axis must be
    ``forward x up = (-cos, 0, -sin)``, west at yaw 0 and north at yaw 90, the same
    ``Direction.getClockWise()`` answer ``BodyFrameMath.toWorld`` returns. It was
    ``(cos, sin)`` instead: east at yaw 0, the body's *left*. The whole file was
    uniformly mirrored before this lane, so a coupled controller merely flew a
    consistent mirror image; correcting the rays alone would have left this site
    sensing right-handed and actuating left-handed.

    Three yaws, because a single one can be satisfied by a sign that is wrong
    everywhere else -- ``turn_ticks`` of saturated yaw torque advance the heading by
    ``yaw_rate_rad_s * dt * substeps = 0.1 rad`` each, with every thrust axis at zero
    so the horizontal velocity is still exactly zero when the lateral tick begins.
    """

    arena = ReferenceArena(pillar_count=0, half_extent_x_m=24.0, half_extent_z_m=24.0)
    body = ReferenceEmbodiment(arena)

    yaw = body.state.yaw_rad
    for tick in range(turn_ticks):
        transition = body.apply(
            EmbodimentCommand(
                command_seq=tick,
                motors=MotorVector(yaw_torque=1.0),
                neural_state=controller.engine.current_state(),
            )
        )
        yaw = transition.world_after.yaw_rad
    assert body.state.velocity_x_m_s == 0.0
    assert body.state.velocity_z_m_s == 0.0

    lateral = body.apply(
        EmbodimentCommand(
            command_seq=turn_ticks,
            motors=MotorVector(lateral_thrust=1.0),
            neural_state=controller.engine.current_state(),
        )
    )
    velocity_x, _, velocity_z = lateral.world_after.velocity_m_s
    speed = math.hypot(velocity_x, velocity_z)
    assert speed > 1.0e-6, "unit lateral thrust must move the body horizontally"
    assert velocity_x / speed == pytest.approx(-math.cos(yaw), abs=1.0e-9)
    assert velocity_z / speed == pytest.approx(-math.sin(yaw), abs=1.0e-9)

    if turn_ticks == 0:
        # Spawn yaw is 0, so the body faces south and its right is due west: negative x.
        # That is the same side seed 298's pillar occupies in the sensing test above,
        # where it shortens slot 3 -- the declared *right* ray. The two halves of the
        # frame now agree on which way the body's right points.
        assert velocity_x < 0.0
        assert velocity_z == pytest.approx(0.0, abs=1.0e-12)


def test_declared_pillar_count_that_cannot_be_placed_fails_closed() -> None:
    """HIGH-2: the identity must never claim an arena the world does not contain.

    Reviewer's demonstration: ``pillar_exclusion_radius_m = 100.0`` is finite and
    passes every constructor check, yet no candidate in the 8 m arena ever clears
    the exclusion disc. The sampler exhausted its draw budget, returned zero
    pillars, raised nothing, and ``to_identity_material`` still reported eight.

    LOW-4: this used to assert ``match="pillar"``, which the arena's own unrelated
    "pillar count must be an integer in [0, 64]" error satisfies just as well -- so
    the test could not tell which guard had fired, or whether the right one existed.
    The second half below constructs that unrelated error on purpose to show the old
    matcher could not have distinguished them.
    """

    with pytest.raises(EmbodimentError) as excinfo:
        ReferenceEmbodiment(ReferenceArena(pillar_exclusion_radius_m=100.0))
    message = str(excinfo.value)
    assert "crowding" in message
    assert "must be an integer in" not in message

    with pytest.raises(EmbodimentError) as unrelated:
        ReferenceArena(pillar_count=65)
    other = str(unrelated.value)
    assert "pillar" in other and "pillar" in message  # what the old matcher saw
    assert "crowding" not in other  # what it could not see


def test_shortfall_error_names_the_declared_and_placed_counts() -> None:
    """A fail-closed error is only useful if it says what it could not do.

    LOW-4: this used to assert ``"8" in message`` and ``"0" in message``. Both are
    digit-substring matches that a message mentioning ``0.35 m`` satisfies by
    accident, so they passed almost regardless of what the message said. The counts
    are parsed out and compared as numbers now, and the message must name the knobs
    that would fix the arena.

    Reaching this at all takes deliberate effort since MEDIUM-1: the declaration
    guard rejects every over-dense arena before the sampler runs, so the shortfall is
    the fail-closed backstop for whatever that necessary condition lets through
    rather than a reachable public path. The guard is stepped past here on purpose --
    ``object.__setattr__`` on the frozen arena -- because a backstop that no test can
    reach is a backstop nobody has read.
    """

    import re

    from neurocraft_embodiment.reference import PILLAR_DRAWS_PER_PILLAR, _generate_pillars

    arena = ReferenceArena(pillar_count=8)
    object.__setattr__(arena, "pillar_exclusion_radius_m", 100.0)
    with pytest.raises(EmbodimentError) as excinfo:
        _generate_pillars(arena)
    message = str(excinfo.value)
    match = re.search(
        r"declares (\d+) pillars but the sampler placed (\d+) in (\d+) draws", message
    )
    assert match is not None, message
    declared, placed, draws = (int(group) for group in match.groups())
    assert declared == arena.pillar_count == 8
    assert placed == 0
    assert draws == PILLAR_DRAWS_PER_PILLAR * arena.pillar_count
    for knob in ("pillar_exclusion_radius_m", "pillar_radius_m", "pillar_count"):
        assert knob in message


def test_arena_rejects_a_negative_spawn_exclusion_radius() -> None:
    """MEDIUM-3: a negative exclusion radius defeated the guarantee it names.

    ``pillar_exclusion_radius_m`` was absent from the positivity block, so a negative
    value constructed cleanly and every candidate cleared an exclusion disc of
    negative radius -- the reviewer observed pillars roughly 0.9 m from the spawn
    origin, inside a body the docstring promises is kept clear.
    """

    with pytest.raises(EmbodimentError, match="finite and positive"):
        ReferenceArena(pillar_exclusion_radius_m=-1.0)
    with pytest.raises(EmbodimentError, match="finite and positive"):
        ReferenceArena(pillar_exclusion_radius_m=0.0)


def test_arena_rejects_a_non_positive_contact_epsilon() -> None:
    """The collision tie-break divides by the distance it guards.

    LOW-5 corrects the rationale this docstring used to give. ``_substep`` tests
    ``distance <= arena.contact_epsilon_m``, so ``contact_epsilon_m == 0.0`` is
    *safe* at ``distance == 0.0``: the comparison holds and the +x tie-break runs.
    Only a **negative** epsilon lets an exact ``distance == 0.0`` fall through to
    ``offset_x / distance`` and raise ``ZeroDivisionError`` instead of a clean
    ``EmbodimentError``.

    Rejecting ``0.0`` is still right, for the other reason: an epsilon of zero makes
    the tie-break fire only on exact float equality, so a body a single ULP off the
    pillar axis divides by a denormal and is thrown an arbitrary distance. The
    constant is validated rather than trusted either way.
    """

    with pytest.raises(EmbodimentError, match="finite and positive"):
        ReferenceArena(contact_epsilon_m=-1.0e-6)
    with pytest.raises(EmbodimentError, match="finite and positive"):
        ReferenceArena(contact_epsilon_m=0.0)


def test_the_declaration_guard_rejects_a_field_that_cannot_be_packed() -> None:
    """The guard is a necessary condition, so it must actually reject something."""

    with pytest.raises(EmbodimentError, match="crowding"):
        ReferenceArena(pillar_count=64, pillar_radius_m=0.8)
    with pytest.raises(EmbodimentError, match="crowding"):
        ReferenceArena(pillar_count=16, pillar_radius_m=1.5, pillar_exclusion_radius_m=4.0)


def test_the_default_seed_zero_pillar_field_is_byte_identical() -> None:
    """Widening the draw budget must not move a single existing pillar.

    ``_generate_pillars`` stops as soon as it has placed the declared count, so a
    larger budget can only extend a stream that previously ran out. This pins the
    shipped field so that claim is checked rather than argued.
    """

    assert ReferenceEmbodiment(ReferenceArena()).pillars == (
        (5.864655365668733, -1.0476216451577975),
        (-7.245563294633254, 7.204494265753576),
        (-6.02289561902165, -2.641915807462676),
        (-4.989821650816853, 4.1546623118729755),
        (-3.89095908274599, 6.916069577927745),
        (-1.584039972879153, 3.993826650902692),
        (3.1858017215150625, 0.28277741431526293),
        (-0.1696061536176744, 4.052644018316567),
    )
    assert (
        ReferenceArena().identity_sha256
        == "e11488ffab91c7de818b998685a7a1f7823c09f80f8d6c514982ab838634b120"
    )


def test_arena_rejects_a_pillar_wider_than_the_arena() -> None:
    """LOW-5: the guard used to be skipped entirely when no pillar was declared.

    ``pillar_radius_m`` is hashed into the arena identity whether or not any pillar
    exists, so an arena declaring ``pillar_count=0`` could carry an impossible radius
    into the hash unchallenged. The check no longer asks how many pillars there are.
    """

    with pytest.raises(EmbodimentError, match="narrower than the arena"):
        ReferenceArena(pillar_radius_m=9.0)
    with pytest.raises(EmbodimentError, match="narrower than the arena"):
        ReferenceArena(pillar_count=0, pillar_radius_m=9.0)


def test_generated_pillars_never_overlap_one_another() -> None:
    """LOW-8: overlapping cylinders have no single nearest surface.

    ``_substep`` corrects against each pillar in turn, so an overlap would make the
    push-out depend on iteration order. The sampler now rejects a candidate within
    two radii of one already placed.
    """

    for seed in (0, 1, 5, 11):
        arena = ReferenceArena(seed=seed)
        pillars = ReferenceEmbodiment(arena).pillars
        assert len(pillars) == arena.pillar_count
        for index, (x, z) in enumerate(pillars):
            for other_x, other_z in pillars[index + 1 :]:
                separation = math.hypot(x - other_x, z - other_z)
                assert separation >= 2.0 * arena.pillar_radius_m


def test_reference_boundary_channel_counts_match_their_channel_lists(
    embodiment: ReferenceEmbodiment,
) -> None:
    """MEDIUM-4: the sensory boundary declared 11 while listing 10 channels."""

    from neurocraft_embodiment import SENSOR_FRAME_CHANNELS
    from neurocraft_embodiment.reference_sensing import REFERENCE_MOTOR_AXES

    sensory = embodiment.identity.sensory_boundary
    motor = embodiment.identity.motor_boundary
    assert sensory.channel_names == SENSOR_FRAME_CHANNELS
    assert sensory.channel_count == len(sensory.channel_names)
    assert motor.channel_names == REFERENCE_MOTOR_AXES
    assert motor.channel_count == len(motor.channel_names)

    # MEDIUM-3: the tally that actually distinguishes this body from the FlyGym one.
    assert sensory.dynamic_channel_count == 18
    assert sensory.constant_channel_count == 2
    tagged = dict(zip(sensory.channel_names, sensory.channels, strict=True))
    assert tagged["in_fluid"].source == CHANNEL_SOURCE_LITERAL_CONSTANT
    assert tagged["in_fluid"].derivation == "false_reference_arena_has_no_fluid"
    assert tagged["ray_max_distance_m"].source == CHANNEL_SOURCE_CONFIG_CONSTANT
    assert tagged["ray_distances_m[0]"].source == CHANNEL_SOURCE_DYNAMIC
    assert motor.dynamic_channel_count == len(REFERENCE_MOTOR_AXES)
