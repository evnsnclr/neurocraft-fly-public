"""MEDIUM-1: what ``ReferenceArena.__post_init__`` admits, a body must be able to run.

An arena is a declaration, and its hash is recorded as the configuration identity of
every run made in it. A declaration that no body can carry is therefore not a harmless
inconvenience: the error surfaces out of a constructor that did not make the mistake,
or -- worse for the three classes below -- out of the *first observation*, after the
run has started.

The previous sweep varied four of the twenty-three fields (seed, pillar count, pillar
radius, spawn exclusion) and asserted only that the pillar sampler could fill the
field. All three demonstrated admitted-but-unbuildable classes lived in the other
nineteen, and a fourth turned up as soon as the sweep was widened:

* ``physics_steps_per_control_tick`` and ``physics_timestep_s`` were validated
  independently (``>= 1`` and ``> 0``) while ``EmbodimentIdentity`` requires their
  product to equal ``1 / control_hz``. Of the 200 step counts in ``1..200`` at the
  default 2 ms timestep, the arena admitted all 200 and exactly one -- 25 -- could be
  built.
* ``ray_max_distance_m`` was admitted anywhere in ``(0, 64]``; the protocol's floor is
  ``0.001``, so ``0.0009999`` constructed and then every ``observe()`` raised.
* extents, floor, ceiling and spawn height were unbounded above; the protocol clamps
  positions to ``+/- 3e7``.
* ``max_speed_m_s`` was unbounded above; the protocol clamps velocity components to
  ``+/- 1024``. Not in the review that prompted this file -- it was found by widening
  the sweep to every field, which is the point of widening it.

The sweep below is therefore over every field, and "buildable" now means what it has
to mean: the body constructs, observes, applies a saturated command, and observes
again. Anything an arena is admitted for, a run in it can do.
"""

from __future__ import annotations

import dataclasses
import itertools
import math

import pytest
from neurocraft_connectome import RuntimeGraph
from neurocraft_controller import ConnectomeController
from neurocraft_embodiment import (
    EmbodimentCommand,
    EmbodimentError,
    EmbodimentIdentityError,
    ReferenceArena,
    ReferenceEmbodiment,
)
from neurocraft_protocol import MotorVector

from .conftest import PACKAGE_SRC
from .test_identity import make_identity

#: Every corner of the five bounded axes is overkill here; one saturated command
#: exercises thrust, lateral, vertical, yaw and pitch at once, which is what the
#: sweep needs to drive a body away from its spawn state.
#:
#: It says nothing about *direction*: this constant stayed green throughout the period
#: the lateral axis pointed at the body's left (ADR 0015). Handedness is asserted in
#: ``tests/embodiment/test_reference_embodiment.py``, not here.
SATURATED = MotorVector(
    forward_thrust=1.0,
    lateral_thrust=1.0,
    vertical_thrust=1.0,
    yaw_torque=1.0,
    pitch_torque=1.0,
)


def usable(arena: ReferenceArena, controller: ConnectomeController) -> None:
    """Everything a run does with an arena, in the order a run does it."""

    body = ReferenceEmbodiment(arena)
    body.observe(identity=controller.identity, sequence=0)
    body.apply(
        EmbodimentCommand(
            command_seq=0,
            motors=SATURATED,
            neural_state=controller.engine.current_state(),
        )
    )
    body.observe(identity=controller.identity, sequence=1)


#: One candidate list per field, chosen to straddle whatever bound the field has.
#: Values are deliberately extreme on both sides: the sweep's job is to find the
#: boundary between admitted and refused and check that nothing survives on the wrong
#: side of it.
FIELD_CANDIDATES: dict[str, tuple[object, ...]] = {
    "schema": ("neurocraft.embodiment.reference.arena.v1", "some.other.schema.v1"),
    "seed": (0, 1, 7, 2**63),
    "half_extent_x_m": (0.5, 3.0, 8.0, 64.0, 1.0e6, 4.0e7),
    "half_extent_z_m": (0.5, 3.0, 8.0, 64.0, 1.0e6, 4.0e7),
    "floor_y_m": (-1.0e8, -5.0, 0.0, 1.0, 2.0),
    "ceiling_y_m": (1.0, 1.6, 6.0, 64.0, 2.0e8),
    "spawn_height_m": (0.0, 0.001, 1.5, 5.999, 1.0e8),
    "pillar_count": (0, 1, 8, 32, 64, 65),
    "pillar_radius_m": (0.0, 0.001, 0.35, 1.5, 9.0),
    "pillar_exclusion_radius_m": (-1.0, 0.5, 2.0, 4.0, 100.0),
    "ray_max_distance_m": (0.0, 0.0009999, 0.001, 0.5, 6.0, 64.0, 128.0),
    "physics_steps_per_control_tick": (0, 1, 10, 25, 50, 200),
    "physics_timestep_s": (0.0005, 0.001, 0.002, 0.0025, 0.05),
    "thrust_accel_m_s2": (0.0, 0.1, 6.0, 1.0e5),
    "lateral_accel_m_s2": (0.0, 0.1, 4.0, 1.0e5),
    "gravity_m_s2": (0.0, 0.01, 9.81, 100.0),
    "linear_drag_per_s": (-1.0, 0.0, 1.6, 50.0),
    "max_speed_m_s": (0.0, 0.01, 4.0, 1024.0, 1024.1, 2000.0),
    "yaw_rate_rad_s": (0.0, 0.1, 2.0, 100.0),
    "pitch_rate_rad_s": (0.0, 0.1, 1.2, 100.0),
    "max_pitch_rad": (0.0, 0.01, 1.4, 1.5707963, math.pi / 2),
    "contact_epsilon_m": (0.0, 1.0e-12, 1.0e-6, 0.1),
    "direction_epsilon": (0.0, 1.0e-18, 1.0e-12, 0.5),
}


def test_the_candidate_table_covers_every_declared_field() -> None:
    """A sweep over a stale field list is a sweep with a hole in it.

    The previous one covered four fields by hand. This asserts the table against the
    dataclass, so a new arena constant cannot be added without a decision about what
    values it is allowed to take.
    """

    declared = {field.name for field in dataclasses.fields(ReferenceArena)}
    assert set(FIELD_CANDIDATES) == declared
    assert len(declared) > 20


def test_every_admitted_arena_can_actually_be_run(
    graph: RuntimeGraph,
) -> None:
    """MEDIUM-1: admitted must imply observable and actuable, over every field."""

    controller = ConnectomeController.create(graph, run_id="envelope", entity_id="fly")
    admitted = 0
    refused = 0
    broken: list[tuple[str, object, str]] = []
    for field, candidates in sorted(FIELD_CANDIDATES.items()):
        for candidate in candidates:
            try:
                arena = ReferenceArena(**{field: candidate})  # type: ignore[arg-type]
            except EmbodimentError:
                refused += 1
                continue
            admitted += 1
            try:
                usable(arena, controller)
            except Exception as exc:
                broken.append((field, candidate, f"{type(exc).__name__}: {exc}"))
    assert broken == []
    # The guard must filter the envelope, not empty it, and not wave it all through.
    assert admitted >= 40
    assert refused >= 20


def test_the_pillar_field_sweep_still_holds_over_combinations(
    graph: RuntimeGraph,
) -> None:
    """LOW-8's sweep, kept: one-at-a-time cannot see a sampler that needs two knobs.

    The pairwise-overlap rejection was added without touching the draw budget, so the
    arena kept admitting fields ``_generate_pillars`` could no longer place -- a defect
    that only appears at particular *combinations* of count, radius and exclusion.
    """

    controller = ConnectomeController.create(graph, run_id="pillars", entity_id="fly")
    admitted = 0
    unbuildable: list[tuple[int, int, float, float]] = []
    for seed, count, radius, exclusion in itertools.product(
        (0, 1, 2, 3, 4, 5), (0, 1, 8, 16, 32, 64), (0.35, 0.8, 1.5), (1.0, 2.0, 4.0)
    ):
        try:
            arena = ReferenceArena(
                seed=seed,
                pillar_count=count,
                pillar_radius_m=radius,
                pillar_exclusion_radius_m=exclusion,
            )
        except EmbodimentError:
            continue
        admitted += 1
        try:
            pillars = ReferenceEmbodiment(arena).pillars
            usable(arena, controller)
        except EmbodimentError:
            unbuildable.append((seed, count, radius, exclusion))
            continue
        assert len(pillars) == count
    assert unbuildable == []
    assert admitted >= 120


# -- the four classes, named individually ---------------------------------------


def test_the_arena_refuses_a_substep_product_that_does_not_fill_a_control_tick() -> None:
    """M1-A: 200 step counts admitted, one buildable."""

    with pytest.raises(EmbodimentError, match="fill exactly one"):
        ReferenceArena(physics_steps_per_control_tick=1)
    with pytest.raises(EmbodimentError, match="fill exactly one"):
        ReferenceArena(physics_timestep_s=0.001)

    buildable = []
    for steps in range(1, 201):
        try:
            ReferenceArena(physics_steps_per_control_tick=steps)
        except EmbodimentError:
            continue
        buildable.append(steps)
    assert buildable == [25]


def test_the_arena_and_the_identity_agree_about_the_clock() -> None:
    """The arena restates a rule ``EmbodimentIdentity`` owns, so the two are compared.

    ``identity.py`` remains the authority; the arena refuses earlier so that a
    declaration is never admitted that a body cannot carry. A restatement is only
    honest if something checks it, so this asserts the two verdicts are the same over
    a grid that contains both closing and non-closing clocks.
    """

    closing = 0
    for steps, timestep in itertools.product(
        (1, 5, 10, 20, 25, 40, 50, 100), (0.0005, 0.001, 0.002, 0.0025, 0.005, 0.01)
    ):
        try:
            ReferenceArena(
                physics_steps_per_control_tick=steps,
                physics_timestep_s=timestep,
            )
            arena_admits = True
        except EmbodimentError:
            arena_admits = False
        try:
            make_identity(
                physics_steps_per_control_tick=steps,
                physics_timestep_s=timestep,
            )
            identity_admits = True
        except EmbodimentIdentityError:
            identity_admits = False
        assert arena_admits is identity_admits, (steps, timestep)
        closing += int(arena_admits)
    # Non-vacuous in both directions.
    assert 0 < closing < 48


def test_the_arena_refuses_a_ray_range_below_the_protocol_floor() -> None:
    """M1-B: ``0.0009999`` constructed, then every ``observe()`` raised."""

    with pytest.raises(EmbodimentError, match="ray_max_distance_m"):
        ReferenceArena(ray_max_distance_m=0.0009999)
    # The floor itself is admitted, so the guard is a bound and not a widening.
    assert ReferenceArena(ray_max_distance_m=0.001).ray_max_distance_m == 0.001


def test_the_arena_refuses_geometry_the_protocol_position_bound_rejects() -> None:
    """M1-C: an arena taller than ``3e7`` built, and then could not report itself."""

    with pytest.raises(EmbodimentError, match="position_m"):
        ReferenceArena(ceiling_y_m=2.0e8, spawn_height_m=1.0e8)
    with pytest.raises(EmbodimentError, match="position_m"):
        ReferenceArena(half_extent_x_m=4.0e7)
    with pytest.raises(EmbodimentError, match="position_m"):
        ReferenceArena(floor_y_m=-1.0e8)


def test_the_arena_refuses_a_top_speed_the_protocol_velocity_bound_rejects(
    graph: RuntimeGraph,
) -> None:
    """The fourth class, found by widening the sweep rather than by the review.

    Reachable, not theoretical: an arena wide enough for the body to keep accelerating
    reports the speed it was allowed to reach, and ``world.velocity_m_s`` is clamped to
    ``+/- 1024``. The refused declaration below is exactly the one that used to build a
    body whose first ``apply`` raised ``ProtocolError`` out of ``WorldState``.
    """

    with pytest.raises(EmbodimentError, match="velocity_m_s"):
        ReferenceArena(max_speed_m_s=2000.0)
    with pytest.raises(EmbodimentError, match="velocity_m_s"):
        ReferenceArena(max_speed_m_s=1024.1)
    # At the bound the arena is admitted, and a body in it really does run.
    controller = ConnectomeController.create(graph, run_id="topspeed", entity_id="fly")
    usable(
        ReferenceArena(
            max_speed_m_s=1024.0,
            thrust_accel_m_s2=1.0e5,
            half_extent_x_m=1.0e6,
            half_extent_z_m=1.0e6,
            ceiling_y_m=1.0e6,
        ),
        controller,
    )


def test_the_envelope_guard_is_derived_from_the_protocol_not_restated() -> None:
    """The guard must not contain the numbers it enforces.

    ``0.001``, ``3e7`` and ``1024`` live in ``neurocraft_protocol`` and nowhere else.
    ``ReferenceArena`` builds the two frames it can emit at the corners of its own
    declared envelope and lets the protocol's validators judge them, so tightening a
    protocol bound tightens this guard with no edit here. A copy of the numbers would
    silently keep admitting arenas the protocol had started refusing.
    """

    import ast

    tree = ast.parse((PACKAGE_SRC / "reference.py").read_text(encoding="utf-8"))
    numbers = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
    }
    for bound in (0.001, 30_000_000, 1024, -1024, -30_000_000):
        assert bound not in numbers, f"the arena restates the protocol bound {bound}"
