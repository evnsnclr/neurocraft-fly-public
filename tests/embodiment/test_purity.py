"""Structural guards on Faithful Organism Mode purity for the embodiment lane.

The single biggest risk in coupling a connectome to NeuroMechFly is scope creep
from "second embodiment" into "we made a fly walk". FlyGym ships
``HybridTurningController`` and friends in ``flygym_demo``; they are one import
away, they would produce a beautiful tripod gait, and the gait would be theirs.
These tests make that failure structural rather than a matter of discipline.

Both halves of the boundary, because only one used to be covered
---------------------------------------------------------------
Every guard in this file used to sit on the *motor* half: forbidden imports,
currency isolation, a neutral-pose spot check at the arena centre, reset
determinism. Two policies walked straight through all four:

* **A wall-avoidance term in ``_substep``**, firing only within 0.5 m of a wall.
  The neutral-pose check starts at x = z = 0, eight metres from every surface, so
  it never entered the region where the policy exists. Caught now by
  :func:`test_in_free_space_the_body_cannot_tell_what_world_it_is_in`.
* **A scripted sine oscillator in ``local_light``**, an encoder-visible sensory
  channel. There was no sensory guard at all. Caught now by
  :func:`test_the_sensor_frame_does_not_depend_on_the_observation_counter` and,
  structurally, by :func:`test_the_observation_counter_never_reaches_the_sensor_frame`.

Both were applied to ``reference.py`` and the whole suite reported 112 passed.

The sensory half is not a lesser case. ``PhysicalFeatureEncoder`` reads exactly
four scalars -- ``ray_distances_m[0]``, ``[2]``, ``[3]`` and ``local_light`` -- so a
fabricated oscillator in one of them is not one input among twenty; on a body whose
other three encoder-visible slots are pinned constants it is 100% of the
time-varying drive the graph receives, and every visible motion would be the
oscillator's period laundered through the connectome. ``seam``'s own module
docstring already says an embodiment "has no goal, no target, no reward, no plan,
no oscillator and no scripted fallback"; the only thing missing was a test.
"""

from __future__ import annotations

import ast
import inspect
import math
import textwrap
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from neurocraft_connectome import RuntimeGraph
from neurocraft_controller import ConnectomeController
from neurocraft_controller.features import PhysicalFeatureEncoder
from neurocraft_embodiment import (
    REFERENCE_SENSORY_CHANNELS,
    EmbodimentCommand,
    FlyGymEmbodiment,
    ReferenceArena,
    ReferenceEmbodiment,
)
from neurocraft_neural.model import make_neural_state
from neurocraft_protocol import MotorVector, SensorFrame

from .conftest import PACKAGE_SRC, REPO_ROOT, kinematic_flygym_body

#: Names that would signal a hand-written locomotion policy inside the boundary.
FORBIDDEN_IMPORTS = (
    "flygym_demo",
    "CPGController",
    "RuleBasedController",
    "HybridController",
    "HybridTurningController",
)

#: Deliberately not an arithmetic progression: a period-8 oscillator sampled at
#: {0, 8, 16} is invisible. Any *fixed* probe set can be aliased by choosing the
#: period, which is why this test is necessary and not sufficient -- the structural
#: guard below carries the general case.
SEQUENCE_PROBE = (0, 1, 2, 3, 5, 7, 11, 13, 17)

#: Every corner of the five bounded actuation axes.
SATURATED_COMMAND_GRID = tuple(
    MotorVector(
        forward_thrust=forward,
        lateral_thrust=lateral,
        vertical_thrust=vertical,
        yaw_torque=yaw,
        pitch_torque=pitch,
    )
    for forward in (-1.0, 0.0, 1.0)
    for lateral in (-1.0, 0.0, 1.0)
    for vertical in (-1.0, 0.0, 1.0)
    for yaw in (-1.0, 0.0, 1.0)
    for pitch in (-1.0, 0.0, 1.0)
)

#: Clearance every probe keeps from every surface. One control tick is 25 x 2 ms =
#: 50 ms and ``max_speed_m_s`` is 4.0, so displacement per tick is at most 0.2 m;
#: from rest under full thrust it is under 0.01 m. A probe at this clearance cannot
#: reach a surface within the tick, and the collision-flag assertions make the test
#: fail closed rather than silently if that ever stops holding.
PROBE_CLEARANCE_M = 0.25


def package_sources() -> list[Path]:
    sources = sorted(path for path in PACKAGE_SRC.rglob("*.py") if "__pycache__" not in path.parts)
    assert len(sources) >= 7, "package source scan found too few files to be meaningful"
    return sources


def neural_state(graph: RuntimeGraph, run_id: str) -> Any:
    controller = ConnectomeController.create(graph, run_id=run_id, entity_id="fly")
    body = ReferenceEmbodiment(ReferenceArena())
    controller.step(body.observe(identity=controller.identity, sequence=0), command_seq=0)
    return controller.engine.current_state()


def body_at(arena: ReferenceArena, x_m: float, z_m: float) -> ReferenceEmbodiment:
    """A body placed at rest at ``(x, z, spawn_height)``.

    Reaches into ``_state`` because the seam exposes no way to place a body, and
    this file may not change ``reference.py``. Placement is the whole experiment:
    the defect these tests exist for lives in a region the public API cannot reach.
    """

    body = ReferenceEmbodiment(arena)
    body._state = replace(body.state, x_m=x_m, z_m=z_m, y_m=arena.spawn_height_m)
    return body


def is_free(
    arena: ReferenceArena, pillars: tuple[tuple[float, float], ...], x: float, z: float
) -> bool:
    if abs(x) > arena.half_extent_x_m - PROBE_CLEARANCE_M:
        return False
    if abs(z) > arena.half_extent_z_m - PROBE_CLEARANCE_M:
        return False
    return all(
        math.hypot(x - pillar_x, z - pillar_z) >= arena.pillar_radius_m + PROBE_CLEARANCE_M
        for pillar_x, pillar_z in pillars
    )


def free_space_probes(
    arena: ReferenceArena,
    pillars: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    """Deliberately includes the near-surface annulus a centre-only probe misses."""

    candidates: list[tuple[float, float]] = []
    for offset in (1.0, 0.7, 0.45, 0.3):
        for sign in (-1.0, 1.0):
            candidates.append((sign * (arena.half_extent_x_m - offset), 0.0))
            candidates.append((0.0, sign * (arena.half_extent_z_m - offset)))
    for pillar_x, pillar_z in pillars:
        radius = arena.pillar_radius_m + 0.3
        for bearing in range(4):
            angle = bearing * math.pi / 2.0
            candidates.append(
                (pillar_x + radius * math.cos(angle), pillar_z + radius * math.sin(angle))
            )
    probes = tuple(point for point in candidates if is_free(arena, pillars, *point))
    assert len(probes) >= 24, f"probe set collapsed to {len(probes)} points"
    return probes


def test_no_source_imports_flygym_demo_or_a_bundled_controller() -> None:
    offenders: list[str] = []
    for path in package_sources():
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if not (stripped.startswith("import ") or stripped.startswith("from ")):
                continue
            if any(name in stripped for name in FORBIDDEN_IMPORTS):
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {stripped}")
    assert offenders == []


def test_only_flygym_itself_is_imported_and_only_lazily() -> None:
    """``import flygym`` must appear inside ``require_flygym``, never at module scope."""

    text = (PACKAGE_SRC / "flygym_body.py").read_text(encoding="utf-8")
    module_scope = [
        line
        for line in text.splitlines()
        if (line.startswith("import ") or line.startswith("from "))
        # Relative imports are this package's own modules, not the third-party one.
        and not line.startswith("from .")
        and ("flygym" in line or "mujoco" in line)
        and "neurocraft" not in line
    ]
    assert module_scope == []
    assert "    import flygym.anatomy as flygym_anatomy" in text


def test_flygym_body_actuates_only_through_the_hashed_decoder() -> None:
    """Its declared currency is the neural state; the primitive axes are unused."""

    text = (PACKAGE_SRC / "flygym_body.py").read_text(encoding="utf-8")
    assert "self._decoder.decode(command.neural_state)" in text
    assert "command.motors" not in text


def test_reference_body_consumes_only_its_declared_currency(
    graph: RuntimeGraph,
) -> None:
    """Same motors, different neural state -> identical transition.

    The reference body declares ``primitive_motor_vector``. If the neural state
    could influence it, the body would be reading a channel it never declared.
    """

    controller = ConnectomeController.create(graph, run_id="currency", entity_id="fly")
    body_a = ReferenceEmbodiment(ReferenceArena())
    body_b = ReferenceEmbodiment(ReferenceArena())
    observation = body_a.observe(identity=controller.identity, sequence=0)
    controller.step(observation, command_seq=0)
    live = controller.engine.current_state()
    zeroed = make_neural_state(
        step_index=live.step_index,
        graph_identity_sha256=live.graph_identity_sha256,
        dynamics_identity_sha256=live.dynamics_identity_sha256,
        sign_overlay_sha256=live.sign_overlay_sha256,
        delay_overlay_sha256=live.delay_overlay_sha256,
        weight_overlay_sha256=live.weight_overlay_sha256,
        topology_overlay_sha256=live.topology_overlay_sha256,
        activity=[0.0] * len(live.activity),
    )
    motors = MotorVector(forward_thrust=0.3, yaw_torque=-0.2)
    first = body_a.apply(EmbodimentCommand(command_seq=0, motors=motors, neural_state=live))
    second = body_b.apply(EmbodimentCommand(command_seq=0, motors=motors, neural_state=zeroed))
    assert first.world_after == second.world_after


def test_neutral_command_at_the_arena_centre_leaves_pose_untouched(graph: RuntimeGraph) -> None:
    """A spot check at exactly one point, kept for what it is.

    x = z = 0 is eight metres from every wall and outside the spawn exclusion disc,
    so this cannot see any policy keyed to proximity. The free-space invariance
    tests below carry the general claim.
    """

    controller = ConnectomeController.create(graph, run_id="neutral", entity_id="fly")
    body = ReferenceEmbodiment(ReferenceArena())
    observation = body.observe(identity=controller.identity, sequence=0)
    controller.step(observation, command_seq=0)
    for tick in range(12):
        body.apply(
            EmbodimentCommand(
                command_seq=tick,
                motors=MotorVector.neutral(),
                neural_state=controller.engine.current_state(),
            )
        )
    assert body.state.yaw_rad == 0.0
    assert body.state.pitch_rad == 0.0
    assert body.state.x_m == 0.0
    assert body.state.z_m == 0.0


def test_reference_body_holds_no_hidden_state_between_identical_runs(
    graph: RuntimeGraph,
) -> None:
    controller = ConnectomeController.create(graph, run_id="reset", entity_id="fly")
    body = ReferenceEmbodiment(ReferenceArena())
    observation = body.observe(identity=controller.identity, sequence=0)
    controller.step(observation, command_seq=0)
    state = controller.engine.current_state()
    motors = MotorVector(forward_thrust=0.5)

    first = body.apply(EmbodimentCommand(command_seq=0, motors=motors, neural_state=state))
    body.reset()
    second = body.apply(EmbodimentCommand(command_seq=1, motors=motors, neural_state=state))
    assert first.world_after == second.world_after


# -- the motor half: position may enter the update only through contact ---------


def test_in_free_space_the_body_cannot_tell_what_world_it_is_in(graph: RuntimeGraph) -> None:
    """The discriminator between a collision response and a policy, made structural.

    A collision response fires when the body is *outside* the free region and only
    ever removes motion. A policy fires when the body is *inside* it and adds
    motion. So: hold every physics constant equal, delete the geometry, and require
    the free-space update to be bitwise identical.

    This is exact, with no tolerance, and that is not luck. In free space
    ``_substep`` touches ``half_extent_*`` only inside two untaken clamp
    comparisons, and the pillar loop either has no iterations or ``continue``s on
    its first test, so every surviving operand is identical. Nobody can later
    "fix flakiness" by loosening it.

    Why an AST scan cannot do this job: the wall clamp at ``reference.py:625-632``
    reads ``arena.half_extent_x_m`` and ``state.x_m`` too, so no name-level predicate
    admits the clamp and rejects a policy -- and a policy written as a ``max(0.0,
    0.5 - clearance)`` gain has no branch to find at all.
    """

    arena = ReferenceArena()
    pillars = ReferenceEmbodiment(arena).pillars
    open_arena = replace(arena, half_extent_x_m=64.0, half_extent_z_m=64.0, pillar_count=0)
    state = neural_state(graph, "ablation")
    probes = free_space_probes(arena, pillars)

    for x_m, z_m in probes:
        walled = body_at(arena, x_m, z_m)
        open_world = body_at(open_arena, x_m, z_m)
        origin = walled.state
        for index, motors in enumerate(SATURATED_COMMAND_GRID):
            walled._state = origin
            open_world._state = origin
            command = EmbodimentCommand(command_seq=index, motors=motors, neural_state=state)
            in_arena = walled.apply(command).world_after
            in_open = open_world.apply(command).world_after
            # Fail closed: the probe must actually have been in free space.
            assert in_arena.horizontal_collision is False
            assert in_open.horizontal_collision is False
            assert in_arena == in_open, (x_m, z_m, motors)


def test_the_free_space_response_does_not_depend_on_where_the_body_is(
    graph: RuntimeGraph,
) -> None:
    """Covers what the ablation cannot: a policy keyed to a hardcoded coordinate.

    Enlarging the arena does not move the literal in ``if abs(x) > 7.5``, so the
    ablation is blind to it; translating the body is not.

    Position equality is deliberately **not** asserted: ``(x + dx) + v*dt`` and
    ``(x + v*dt) + dx`` are not bitwise equal, and a tolerance on position would be
    the seam through which this test gets weakened later. Every quantity below
    carries no position term at all, and any policy that alters the trajectory has
    to alter one of them.
    """

    arena = ReferenceArena()
    open_arena = replace(arena, half_extent_x_m=64.0, half_extent_z_m=64.0, pillar_count=0)
    state = neural_state(graph, "equivariance")
    offsets = ((3.0, 0.0), (-3.0, 0.0), (0.0, 7.5), (0.0, -7.5), (20.0, -20.0), (-40.0, 40.0))

    for index, motors in enumerate(SATURATED_COMMAND_GRID):
        command = EmbodimentCommand(command_seq=index, motors=motors, neural_state=state)
        base = body_at(open_arena, 0.0, 0.0).apply(command).world_after
        for offset_x, offset_z in offsets:
            moved = body_at(open_arena, offset_x, offset_z).apply(command).world_after
            assert moved.velocity_m_s == base.velocity_m_s, (offset_x, offset_z, motors)
            assert moved.yaw_rad == base.yaw_rad
            assert moved.pitch_rad == base.pitch_rad
            assert moved.on_ground == base.on_ground
            assert moved.horizontal_collision is False


# -- the sensory half: the frame is a function of body state, nothing else ------


def test_the_sensor_frame_does_not_depend_on_the_observation_counter(
    graph: RuntimeGraph,
    controller: ConnectomeController,
) -> None:
    """The counter is a wire header. A frame that moves without the body moving is
    reporting something that is not a measurement of anything.

    ``reset()`` between samples because the FlyGym body's ``observe`` refuses any
    call where the elapsed control ticks are not exactly one; resetting makes each
    call a first observation and returns the stub to its declared pose.
    """

    for body in (ReferenceEmbodiment(ReferenceArena()), kinematic_flygym_body(graph)):
        frames: list[SensorFrame] = []
        for sequence in SEQUENCE_PROBE:
            body.reset()
            frames.append(body.observe(identity=controller.identity, sequence=sequence).sensors)
        assert len(set(frames)) == 1, f"{type(body).__name__} frame moved without the body moving"


def uses_of_parameter(function: Any, name: str) -> set[tuple[str, str]]:
    """Classify every load of ``name`` inside ``function`` by the role it plays."""

    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    uses: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or node.id != name:
            continue
        if isinstance(node.ctx, ast.Store):
            uses.add(("assigned", name))
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.keyword):
            uses.add(("keyword", parent.arg or "**kwargs"))
        elif isinstance(parent, ast.Call) and any(arg is node for arg in parent.args):
            target = parent.func
            uses.add(("call_arg", target.id if isinstance(target, ast.Name) else ast.dump(target)))
        else:
            uses.add(("other", type(parent).__name__))
    return uses


def test_the_observation_counter_never_reaches_the_sensor_frame() -> None:
    """The general case the probe set cannot cover: constrain the parameter's *role*.

    ``sequence`` is a wire header. Exactly two uses are legal -- it names the tick
    and it is echoed into the observation header. Anything else (arithmetic, a
    comparison, a subscript, a modulus) is the counter being turned into sensor
    material, and no rewriting of a scripted oscillator evades that.
    """

    legal = {("call_arg", "observation_timing"), ("keyword", "observation_seq")}
    for observe in (FlyGymEmbodiment.observe, ReferenceEmbodiment.observe):
        assert uses_of_parameter(observe, "sequence") == legal, observe.__qualname__


def test_no_internal_phase_survives_reset(graph: RuntimeGraph) -> None:
    """The sibling failure: an oscillator driven by the body's own accumulator.

    This does **not** catch a counter-keyed oscillator -- the caller's sequence
    repeats across the two passes, so a ``sin(2*pi*sequence/8)`` term reproduces
    perfectly here. It is a different violation class and is named for it.

    The sweep length is prime for a measured reason. At eight ticks this test
    passed against an accumulator-driven period-8 oscillator: the second pass ran
    phases 9..16, which alias exactly onto 1..8. A prime length aliases only against
    periods that divide it, so five is the cheapest length that does not hand a
    period-2, -4 or -8 oscillator a free pass.
    """

    controller = ConnectomeController.create(graph, run_id="phase", entity_id="fly")
    body = ReferenceEmbodiment(ReferenceArena())
    state = neural_state(graph, "phase-state")
    motors = MotorVector(forward_thrust=0.4, yaw_torque=0.3)

    def sweep() -> list[SensorFrame]:
        frames = []
        for tick in range(5):
            frames.append(body.observe(identity=controller.identity, sequence=tick).sensors)
            body.apply(EmbodimentCommand(command_seq=tick, motors=motors, neural_state=state))
        return frames

    first = sweep()
    body.reset()
    assert sweep() == first


def encoder_visible_channels() -> tuple[str, ...]:
    """The sensory slots the pinned encoder reads, derived from its own schema."""

    return tuple(
        entry["source"].removeprefix("sensors.")
        for entry in PhysicalFeatureEncoder.SCHEMA_MATERIAL["channels"]
    )


def test_the_encoder_reads_exactly_four_of_the_twenty_sensory_scalars() -> None:
    """Read off the encoder's hashed schema, so the number cannot be stale."""

    assert encoder_visible_channels() == (
        "ray_distances_m[0]",
        "ray_distances_m[2]",
        "ray_distances_m[3]",
        "local_light",
    )


def test_the_encoder_visible_light_channel_matches_the_derivation_it_publishes(
    graph: RuntimeGraph,
    controller: ConnectomeController,
) -> None:
    """``derivation`` is hashed into the boundary identity, so it is a testable claim.

    Reimplementing a formula in a test is normally a smell; here it is the point,
    because the test checks the identity's published claim rather than the code's
    behaviour. Anyone changing the mapping must change the derivation string, which
    rotates the sensory boundary hash -- which is what makes that hash load-bearing
    instead of decorative.

    Scope, stated because it is narrower than the section heading: this covers
    ``local_light``, the only encoder-visible reference slot that is not a ray cast.
    The three rays are geometric and reimplementing them here would be a second
    implementation of the body rather than a check on it.
    """

    declared = {entry.name: entry.derivation for entry in REFERENCE_SENSORY_CHANNELS}
    assert declared["local_light"] == "linear_height_gradient_floor_to_ceiling_16_levels"

    arena = ReferenceArena()
    body = ReferenceEmbodiment(arena)
    state = neural_state(graph, "light")
    span = arena.ceiling_y_m - arena.floor_y_m
    observed_levels = set()
    for tick, thrust in enumerate((1.0,) * 30 + (-1.0,) * 40):
        frame = body.observe(identity=controller.identity, sequence=tick).sensors
        expected = max(0, min(15, round(15.0 * (body.state.y_m - arena.floor_y_m) / span)))
        assert frame.local_light == expected, (tick, body.state.y_m)
        observed_levels.add(frame.local_light)
        body.apply(
            EmbodimentCommand(
                command_seq=tick,
                motors=MotorVector(vertical_thrust=thrust),
                neural_state=state,
            )
        )
    # Non-vacuous: the trajectory really moves the channel through several levels.
    assert len(observed_levels) >= 3


@pytest.mark.parametrize("slot", ["ray_distances_m", "local_light"])
def test_every_encoder_visible_slot_is_declared_by_the_sensory_boundary(slot: str) -> None:
    """No encoder-visible scalar may be missing from the body's sourcing table."""

    declared = {entry.name for entry in REFERENCE_SENSORY_CHANNELS}
    visible = [name for name in encoder_visible_channels() if name.startswith(slot)]
    assert visible
    assert set(visible) <= declared
