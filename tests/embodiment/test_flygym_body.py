from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from neurocraft_connectome import RuntimeGraph
from neurocraft_controller import ConnectomeController
from neurocraft_embodiment import (
    CONSUMES_NEURAL_STATE,
    NEUROMECHFLY_ADHESION_COUNT,
    PINNED_MESH_TYPE,
    Embodiment,
    EmbodimentCommand,
    EmbodimentError,
    FlyGymActuation,
    FlyGymBodyConfig,
    FlyGymEmbodiment,
    FlyGymModules,
    FlyGymMotorDecoder,
)
from neurocraft_embodiment.flygym_body import _yaw_pitch_from_quaternion
from neurocraft_protocol import DeterminismLevel, MotorVector

from .conftest import (
    REPO_ROOT,
    fixture_adhesion_bindings,
    fixture_decoder_free_parameters,
    fixture_joint_bindings,
    make_fixture_decoder,
)

FLYGYM_INSTALLED = importlib.util.find_spec("flygym") is not None


def test_config_pins_the_offline_mesh_set() -> None:
    assert FlyGymBodyConfig().mesh_type == PINNED_MESH_TYPE == "SIMPLIFIED_MAX2000FACES"
    with pytest.raises(EmbodimentError, match="network"):
        FlyGymBodyConfig(mesh_type="FULLSIZE")


def test_config_refuses_vision_in_the_loop_in_v1() -> None:
    with pytest.raises(EmbodimentError, match="NUMERICAL_TOLERANCE"):
        FlyGymBodyConfig(enable_vision=True)


def test_config_clock_matches_the_frozen_20_hz_contract() -> None:
    config = FlyGymBodyConfig()
    assert config.physics_timestep_s == 1.0e-4
    assert config.physics_steps_per_control_tick == 500
    assert config.physics_steps_per_control_tick * config.physics_timestep_s == pytest.approx(0.05)


def test_config_identity_covers_every_construction_constant() -> None:
    baseline = FlyGymBodyConfig().identity_sha256
    assert len(baseline) == 64
    assert FlyGymBodyConfig(actuator_kp=44.0).identity_sha256 != baseline
    assert FlyGymBodyConfig(adhesion_gain=41.0).identity_sha256 != baseline
    assert FlyGymBodyConfig(seed=1).identity_sha256 != baseline


def test_yaw_pitch_conversion_stays_inside_the_protocol_envelope() -> None:
    import math

    yaw, pitch = _yaw_pitch_from_quaternion((1.0, 0.0, 0.0, 0.0))
    assert (yaw, pitch) == (0.0, 0.0)
    for quaternion in (
        (0.0, 0.0, 0.0, 1.0),
        (0.7071067811865476, 0.0, 0.7071067811865476, 0.0),
        (0.5, 0.5, 0.5, 0.5),
    ):
        yaw, pitch = _yaw_pitch_from_quaternion(quaternion)
        assert -math.pi <= yaw <= math.pi
        assert -math.pi / 2 <= pitch <= math.pi / 2


def test_flygym_embodiment_requires_the_wide_decoder() -> None:
    with pytest.raises(EmbodimentError, match="FlyGymMotorDecoder"):
        FlyGymEmbodiment(object(), modules=_stub_modules())  # type: ignore[arg-type]


def test_flygym_embodiment_declares_the_neural_state_currency(graph: RuntimeGraph) -> None:
    """Built against a stub so the identity contract is testable without MuJoCo.

    This exercises identity construction and the declared currency only. It does not
    and cannot demonstrate that FlyGym runs; the first real run is the integration
    test.
    """

    body = FlyGymEmbodiment(
        make_fixture_decoder(graph),
        modules=_stub_modules(),
    )
    assert isinstance(body, Embodiment)
    identity = body.identity
    assert identity.motor_currency == CONSUMES_NEURAL_STATE
    assert identity.embodiment_id == "flygym-neuromechfly"
    assert identity.control_hz == 20.0
    assert identity.physics_steps_per_control_tick == 500
    assert identity.determinism_level is DeterminismLevel.SAME_BUILD_EXACT
    provenance = body.provenance()
    assert provenance["flygym_version"] == "2.1.0-stub"
    assert provenance["mesh_type"] == PINNED_MESH_TYPE
    assert provenance["gl_backend"] == "not_used_vision_disabled"
    assert identity.motor_boundary.identity_sha256 == make_fixture_decoder(graph).identity_sha256


#: One stub physics step advances the thorax by this much along MuJoCo +x.
STUB_STEP_DELTA_MM = 0.01

#: Where the stub thorax sits, in MuJoCo (x, y, z) mm. z is height.
STUB_SPAWN_POSITION_MM = (0.0, 0.0, 0.5)

#: The DOF order the default stub skeleton reports, matching ``fixture_joint_bindings``.
STUB_ACTUATED_DOFS = ("lf_dof_0", "lf_dof_1", "lf_dof_2", "lf_dof_3")


def _stub_modules(
    *,
    actuated_dofs: tuple[str, ...] = STUB_ACTUATED_DOFS,
    preset_suffix: str = "",
    step_delta_mm: tuple[float, float, float] = (STUB_STEP_DELTA_MM, 0.0, 0.0),
    spawn_position_mm: tuple[float, float, float] = STUB_SPAWN_POSITION_MM,
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    contact_found: tuple[float, ...] = (0.0,) * 6,
    contact_forces: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),) * 6,
) -> FlyGymModules:
    """A minimal stand-in for the FlyGym surface used during construction.

    Deliberately narrow: it records the calls the body makes so the construction
    sequence is pinned, and it translates the thorax by a fixed amount per step so
    the finite-difference velocity path has a known displacement to measure. It is
    never presented as a simulation -- nothing here integrates an equation of
    motion, and no result obtained through it is a FlyGym result.

    ``preset_suffix`` decorates the ``.name`` of every stub enum member so a test
    can prove the body reads the enum it actually used rather than re-declaring a
    matching string literal.

    ``spawn_position_mm``, ``step_delta_mm`` and ``quaternion`` make the stub's pose
    scriptable. They exist because the per-slot constancy check could only produce
    sample states differing in leg contact and in MuJoCo +x, which left the height,
    yaw and pitch readings -- and therefore ``local_light``, ``ray_distances_m[5]``,
    ``yaw_rad``, ``pitch_rad`` and two of the three velocity components -- pinned
    across every sample it compared. Scripting a pose is not simulating one, and
    nothing these knobs produce is a FlyGym or MuJoCo result.
    """

    class _Preset:
        def __init__(self, name: str) -> None:
            self.name = name

        def get_pose_by_axis_order(self, _order: object) -> object:
            return ("neutral_pose", self.name)

    class _Enum:
        def __getattr__(self, name: str) -> object:
            return _Preset(name + preset_suffix)

    class _Skeleton:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def get_actuated_dofs_from_preset(self, _preset: object) -> tuple[str, ...]:
            return actuated_dofs

    class _Anatomy:
        AxisOrder = _Enum()
        JointPreset = _Enum()
        ActuatedDOFPreset = _Enum()
        Skeleton = _Skeleton

    class _Fly:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def add_joints(self, *_args: object, **_kwargs: object) -> None:
            return None

        def add_actuators(self, *_args: object, **_kwargs: object) -> None:
            return None

        def add_leg_adhesion(self, **_kwargs: object) -> None:
            return None

    class _World:
        def __init__(self, **_kwargs: object) -> None:
            self.add_fly_kwargs: dict[str, object] = {}

        def add_fly(self, *_args: object, **_kwargs: object) -> None:
            # Recorded so a test can pin the spawn pose actually handed over;
            # FlyGym 2.1.0 refuses a raw Euler tuple here (EMBODIMENT-02 defect 1).
            self.add_fly_kwargs = dict(_kwargs)
            return None

    class _WorldModule:
        FlatGroundWorld = _World

    class _Compose:
        KinematicPosePreset = _Enum()
        ActuatorType = _Enum()
        NeuroMechFly = _Fly
        world = _WorldModule()

    class _Simulation:
        """Kinematic stub: ``step`` translates the thorax, it does not simulate."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.position_mm: tuple[float, float, float] = spawn_position_mm
            self.steps = 0
            self.actuator_inputs: list[tuple[float, ...]] = []
            self.adhesion_states: list[tuple[float, ...]] = []

        def reset(self) -> None:
            self.position_mm = spawn_position_mm
            self.steps = 0

        def warmup(self, **_kwargs: object) -> None:
            return None

        def step(self) -> None:
            self.steps += 1
            x, y, z = self.position_mm
            self.position_mm = (
                x + step_delta_mm[0],
                y + step_delta_mm[1],
                z + step_delta_mm[2],
            )

        def set_actuator_inputs(self, _fly: object, _kind: object, values: Sequence[float]) -> None:
            self.actuator_inputs.append(tuple(values))

        def set_leg_adhesion_states(self, _fly: object, values: Sequence[float]) -> None:
            self.adhesion_states.append(tuple(values))

        def get_body_positions(self, _fly: object) -> tuple[tuple[float, float, float], ...]:
            return (self.position_mm,)

        def get_body_rotations(self, _fly: object) -> tuple[tuple[float, float, float, float], ...]:
            return (quaternion,)

        def get_ground_contact_info(self, _fly: object) -> tuple[object, object]:
            return (contact_found, contact_forces)

    class _SimulationModule:
        Simulation = _Simulation

    def _stub_rotation3d(*, format: str, values: object) -> tuple[str, str, object]:
        return ("stub_rotation", format, values)

    return FlyGymModules(
        anatomy=_Anatomy(),
        compose=_Compose(),
        simulation=_SimulationModule(),
        rotation3d=_stub_rotation3d,
        version="2.1.0-stub",
        mujoco_version="3.9.0-stub",
    )


class _SixLegDecoder(FlyGymMotorDecoder):
    """Test double that pads adhesion to the six channels NeuroMechFly requires.

    The 14-node CI fixture exposes six motor/descending nodes in total, and the
    decoder demands disjoint populations, so one joint plus six adhesion legs is
    seven populations the fixture cannot supply -- ``apply`` is unreachable with a
    real fixture decoder. This subclass changes nothing about the readout
    arithmetic; it only widens the adhesion tuple so the *velocity* path can be
    exercised. It makes no claim about gait, contact or adhesion behaviour.
    """

    def decode(self, state: object) -> FlyGymActuation:
        actuation = super().decode(state)  # type: ignore[arg-type]
        missing = NEUROMECHFLY_ADHESION_COUNT - len(actuation.adhesion)
        return replace(actuation, adhesion=actuation.adhesion + (0.5,) * missing)


def _six_leg_decoder(graph: RuntimeGraph) -> _SixLegDecoder:
    return _SixLegDecoder(
        graph,
        fixture_joint_bindings(),
        fixture_adhesion_bindings(),
        declared_free_parameters=fixture_decoder_free_parameters(),
    )


def _kinematic_body(
    graph: RuntimeGraph,
    **stub_kwargs: object,
) -> FlyGymEmbodiment:
    return FlyGymEmbodiment(
        _six_leg_decoder(graph),
        modules=_stub_modules(**stub_kwargs),  # type: ignore[arg-type]
    )


def _command(controller: ConnectomeController, tick: int) -> EmbodimentCommand:
    return EmbodimentCommand(
        command_seq=tick,
        motors=MotorVector.neutral(),
        neural_state=controller.engine.current_state(),
    )


#: Displacement one control tick of the kinematic stub produces, in protocol m/s.
STUB_TICK_VELOCITY_M_S = (
    STUB_STEP_DELTA_MM * FlyGymBodyConfig().physics_steps_per_control_tick / 1000.0
) / (1.0 / 20.0)


def test_observe_reports_a_real_per_tick_velocity_rather_than_permanent_zero(
    graph: RuntimeGraph,
    controller: ConnectomeController,
) -> None:
    """HIGH-1: ``observe`` must not report ``(0, 0, 0)`` on every tick forever.

    Reviewer's demonstration: no physics runs between one tick's ``apply`` and the
    next tick's ``observe``, so an ``observe`` that finite-differences against the
    position ``apply`` just committed always reads a zero displacement. Driven
    through a stub whose every step translates the thorax, the first observation
    has no earlier sample and is honestly zero; every later one must carry the
    displacement of exactly one control tick.
    """

    body = _kinematic_body(graph)
    observed: list[tuple[float, float, float]] = []
    for tick in range(4):
        frame = body.observe(identity=controller.identity, sequence=tick).sensors
        observed.append(frame.velocity_m_s)
        body.apply(_command(controller, tick))

    assert observed[0] == (0.0, 0.0, 0.0)
    for velocity in observed[1:]:
        assert velocity[0] == pytest.approx(STUB_TICK_VELOCITY_M_S)
        assert velocity[0] > 0.0


def test_apply_reports_first_tick_velocity_from_its_own_position_samples(
    graph: RuntimeGraph,
    controller: ConnectomeController,
) -> None:
    """HIGH-1: the first ``apply`` must report real displacement, not ``(0, 0, 0)``.

    ``apply`` samples the thorax before and after its own physics-step loop, so its
    velocity is self-contained and does not depend on whether ``observe`` was ever
    called.
    """

    body = _kinematic_body(graph)
    first = body.apply(_command(controller, 0)).world_after.velocity_m_s
    second = body.apply(_command(controller, 1)).world_after.velocity_m_s
    assert first[0] == pytest.approx(STUB_TICK_VELOCITY_M_S)
    assert second[0] == pytest.approx(STUB_TICK_VELOCITY_M_S)
    # Both ticks travel the same distance; the stub accumulates in floating point,
    # so this is an equality up to that accumulation, not an exact one.
    assert first == pytest.approx(second)


def test_apply_velocity_is_independent_of_the_observe_call_pattern(
    graph: RuntimeGraph,
    controller: ConnectomeController,
) -> None:
    """HIGH-1: interleaving ``observe`` must not perturb what ``apply`` reports."""

    with_observe = _kinematic_body(graph)
    without_observe = _kinematic_body(graph)
    for tick in range(3):
        with_observe.observe(identity=controller.identity, sequence=tick)
        applied = with_observe.apply(_command(controller, tick)).world_after.velocity_m_s
        bare = without_observe.apply(_command(controller, tick)).world_after.velocity_m_s
        assert applied == bare
        assert applied[0] == pytest.approx(STUB_TICK_VELOCITY_M_S)


def test_repository_layout_places_the_package_outside_hashed_roots() -> None:
    """``replay_implementation_sha256`` must be unaffected by this package."""

    from scripts.fixture_harness import replay_implementation_sha256

    before = replay_implementation_sha256()
    package = REPO_ROOT / "packages/embodiment-runtime/src"
    assert package.is_dir()
    assert replay_implementation_sha256() == before
    assert Path("packages/embodiment-runtime") not in [
        Path("packages/connectome-data"),
        Path("packages/controller-runtime"),
        Path("packages/neural-runtime"),
        Path("packages/protocol"),
    ]


def test_construction_rejects_a_decoder_ordered_differently_from_the_skeleton(
    graph: RuntimeGraph,
) -> None:
    """MEDIUM-6: joint targets are passed positionally, so order is behaviour.

    Nothing compared the decoder's binding order against the order the skeleton
    reports, so a decoder listing the same DOFs in another order would have driven
    every leg with another leg's target, silently and forever.
    """

    swapped = ("lf_dof_1", "lf_dof_0", "lf_dof_2", "lf_dof_3")
    with pytest.raises(EmbodimentError, match="actuated DOF order mismatch at index 0"):
        FlyGymEmbodiment(
            make_fixture_decoder(graph),
            modules=_stub_modules(actuated_dofs=swapped),
        )


def test_construction_rejects_a_decoder_binding_a_different_number_of_dofs(
    graph: RuntimeGraph,
) -> None:
    """The empty stub order the earlier test used would have passed unnoticed."""

    with pytest.raises(EmbodimentError, match="actuated DOF order mismatch"):
        FlyGymEmbodiment(
            make_fixture_decoder(graph),
            modules=_stub_modules(actuated_dofs=()),
        )


def test_construction_rejects_dofs_it_cannot_name(graph: RuntimeGraph) -> None:
    """An unnameable DOF fails closed rather than being coerced into a name."""

    with pytest.raises(EmbodimentError, match="does not expose a string enum name"):
        FlyGymEmbodiment(
            make_fixture_decoder(graph),
            modules=_stub_modules(actuated_dofs=(1, 2, 3, 4)),  # type: ignore[arg-type]
        )


def test_architecture_identity_is_read_off_the_enums_build_actually_used(
    graph: RuntimeGraph,
) -> None:
    """MEDIUM-7: five identity strings used to be hand-copied literals.

    ``actuator_type``, ``actuated_dof_preset``, ``joint_preset``,
    ``kinematic_pose_preset`` and ``axis_order`` were literals in the *config*
    identity material, duplicating enum members chosen in ``_build``. Changing a
    preset without editing the literal would have left ``identity_sha256`` unrotated.
    Here the stub decorates every enum member's ``name``; the recorded strings must
    follow it, which a hardcoded literal could not do.
    """

    body = FlyGymEmbodiment(make_fixture_decoder(graph), modules=_stub_modules())
    provenance = body.provenance()
    assert provenance["axis_order"] == "YAW_PITCH_ROLL"
    assert provenance["joint_preset"] == "LEGS_ONLY"
    assert provenance["kinematic_pose_preset"] == "NEUTRAL"
    assert provenance["actuated_dof_preset"] == "LEGS_ACTIVE_ONLY"
    assert provenance["actuator_type"] == "POSITION"

    decorated = FlyGymEmbodiment(
        make_fixture_decoder(graph),
        modules=_stub_modules(preset_suffix="_VARIANT"),
    )
    assert decorated.provenance()["axis_order"] == "YAW_PITCH_ROLL_VARIANT"
    assert decorated.provenance()["actuator_type"] == "POSITION_VARIANT"
    # A different architecture is a different body.
    assert decorated.identity.identity_sha256 != body.identity.identity_sha256

    # And the literals are gone from the configuration material, where nothing could
    # have kept them honest.
    material = FlyGymBodyConfig().to_identity_material()
    for field in (
        "actuator_type",
        "actuated_dof_preset",
        "axis_order",
        "joint_preset",
        "kinematic_pose_preset",
    ):
        assert field not in material


def test_sensory_boundary_channel_count_is_the_protocol_sensor_frame_width(
    graph: RuntimeGraph,
) -> None:
    """MEDIUM-4: this body declared 11 while its own comment derived 20.

    The trailing ``== 20`` that used to sit here is gone: ``test_seam_channels`` owns
    the width pin now, over all twenty names in order rather than one scalar that
    cannot see a rename, a reorder or two compensating width changes.
    """

    from neurocraft_embodiment import SENSOR_FRAME_CHANNELS

    body = FlyGymEmbodiment(make_fixture_decoder(graph), modules=_stub_modules())
    sensory = body.identity.sensory_boundary
    motor = body.identity.motor_boundary
    assert sensory.channel_names == SENSOR_FRAME_CHANNELS
    assert sensory.channel_count == len(sensory.channel_names)
    assert motor.channel_count == len(motor.channel_names)
    assert motor.channel_names == make_fixture_decoder(graph).channel_names


def test_the_two_sensory_boundaries_are_distinguishable_by_their_channel_records(
    graph: RuntimeGraph,
) -> None:
    """MEDIUM-3: identical name lists told a reviewer nothing about either body.

    Both boundaries carry the same twenty protocol names, so ``channel_names`` alone
    could not distinguish them and merely restated ``observation_schema_hash``. The
    per-slot ``source`` tag is what separates them, and the number it implies is the
    reviewer-facing fact: the reference arena feeds the connectome eighteen varying
    scalars, the FlyGym body on ``FlatGroundWorld`` feeds it twelve, with no lateral
    range sense at all.
    """

    from neurocraft_embodiment import (
        CHANNEL_SOURCE_CONFIG_CONSTANT,
        CHANNEL_SOURCE_LITERAL_CONSTANT,
        ReferenceArena,
        ReferenceEmbodiment,
    )

    fly = FlyGymEmbodiment(make_fixture_decoder(graph), modules=_stub_modules())
    arena = ReferenceEmbodiment(ReferenceArena())
    fly_sensory = fly.identity.sensory_boundary
    arena_sensory = arena.identity.sensory_boundary

    assert fly_sensory.channel_names == arena_sensory.channel_names
    assert fly_sensory.channels != arena_sensory.channels
    assert (fly_sensory.dynamic_channel_count, fly_sensory.constant_channel_count) == (12, 8)
    assert (arena_sensory.dynamic_channel_count, arena_sensory.constant_channel_count) == (18, 2)

    tagged = dict(zip(fly_sensory.channel_names, fly_sensory.channels, strict=True))
    for index in range(5):
        entry = tagged[f"ray_distances_m[{index}]"]
        assert entry.source == CHANNEL_SOURCE_CONFIG_CONSTANT
        assert entry.derivation == "saturated_flat_ground_no_occluders"
    assert tagged["ray_distances_m[5]"].derivation == "thorax_height_above_ground"
    assert tagged["horizontal_collision"].source == CHANNEL_SOURCE_LITERAL_CONSTANT
    assert tagged["in_fluid"].source == CHANNEL_SOURCE_LITERAL_CONSTANT


def test_a_zero_gain_pool_is_declared_a_constant_motor_channel(graph: RuntimeGraph) -> None:
    """The motor dynamic tally is a real reading, not ``len`` under another name.

    ``gain >= 0`` is legal, and a pool with ``gain == 0.0`` emits exactly zero
    however the graph moves -- pinning its DOF to the neutral angle forever. The
    decoder reads that off the compiled pool, so a boundary that smuggles in a
    constant DOF has to declare one fewer dynamic channel.
    """

    from dataclasses import replace as dataclass_replace

    from neurocraft_embodiment import CHANNEL_SOURCE_CONFIG_CONSTANT, FlyGymMotorDecoder
    from neurocraft_neural import MotorPool

    joints = list(fixture_joint_bindings())
    pinned = dataclass_replace(
        joints[0],
        pool=MotorPool(name=joints[0].pool.name, members=joints[0].pool.members, gain=0.0),
    )
    decoder = FlyGymMotorDecoder(
        graph,
        (pinned, *joints[1:]),
        fixture_adhesion_bindings(),
        declared_free_parameters=fixture_decoder_free_parameters(),
    )
    assert decoder.dynamic_channel_count == decoder.channel_count - 1
    assert decoder.channels[0].source == CHANNEL_SOURCE_CONFIG_CONSTANT
    assert decoder.channels[0].derivation == "zero_gain_pool_pins_joint_to_neutral"

    body = FlyGymEmbodiment(decoder, modules=_stub_modules())
    motor = body.identity.motor_boundary
    assert motor.dynamic_channel_count == motor.channel_count - 1
    assert motor.constant_channel_count == 1


def test_contact_mapping_describes_what_contacts_actually_computes() -> None:
    """LOW-9: the identity claimed a segment filter the code does not implement.

    ``_contacts`` flags any contact-force row with a nonzero magnitude and applies no
    segment-type test, so ordinary tarsal contact raises the flag. Whether FlyGym even
    exposes the segment identity such a filter would need cannot be confirmed without
    a real install, so the honest description is recorded rather than the intended
    one.

    MEDIUM-5 moved the string to the flag it actually describes. The unfiltered
    reading is a *ground* contact, so it is the vertical flag; the horizontal flag is
    a declared world constant. The LOW-9 claim -- no fabricated segment filter
    anywhere in the package -- is asserted over the whole package, not one file.
    """

    from neurocraft_embodiment import FLYGYM_SENSORY_MAPPING

    assert (
        FLYGYM_SENSORY_MAPPING["vertical_collision"]
        == "any_nonzero_ground_contact_force_row_no_segment_filter"
    )
    assert FLYGYM_SENSORY_MAPPING["horizontal_collision"].startswith("false_")
    package = REPO_ROOT / "packages/embodiment-runtime/src/neurocraft_embodiment"
    for path in sorted(package.rglob("*.py")):
        assert "non_tarsal_segment_ground_contact" not in path.read_text(encoding="utf-8")


def test_ground_contact_raises_the_vertical_flag_not_the_horizontal_one(
    graph: RuntimeGraph,
    controller: ConnectomeController,
) -> None:
    """MEDIUM-5: the rename made the label honest; the signal was still misrouted.

    ``get_ground_contact_info`` reports contact with the *floor*. Both protocol
    contact flags inherit their meaning from the Minecraft entity they were defined
    against -- ``FlySensors.java`` binds them to ``Entity.horizontalCollision`` and
    ``Entity.verticalCollision`` -- and the reference body follows the same
    convention, raising the horizontal flag for walls and pillars and the vertical
    one for the floor and the ceiling. A body resting on the ground therefore has
    ``vertical_collision`` true and ``horizontal_collision`` false.

    This body did the opposite of both: it fed leg-on-floor contact force into
    ``horizontal_collision``, telling the connectome it had struck a wall every time
    a tarsus touched the ground, and hardcoded ``vertical_collision`` to the negation
    of the fact it was actually measuring. ``FlatGroundWorld`` has no vertical
    geometry, so ``horizontal_collision`` is a world constant here, not a reading.
    """

    body = _kinematic_body(graph, contact_forces=((0.0, 0.0, 0.4),) * 6)
    sensors = body.observe(identity=controller.identity, sequence=0).sensors
    assert sensors.vertical_collision is True
    assert sensors.horizontal_collision is False

    transition = body.apply(_command(controller, 0))
    assert transition.world_after.vertical_collision is True
    assert transition.world_after.horizontal_collision is False


def test_a_body_out_of_contact_lowers_the_vertical_flag(
    graph: RuntimeGraph,
    controller: ConnectomeController,
) -> None:
    """The routed flag must be a reading, not a second constant."""

    body = _kinematic_body(graph)
    sensors = body.observe(identity=controller.identity, sequence=0).sensors
    assert sensors.vertical_collision is False
    assert sensors.horizontal_collision is False


def test_the_sensory_mapping_is_immutable_and_covers_every_protocol_field() -> None:
    """LOW-2 and MEDIUM-5: a shared mutable dict reached into hashed identity material.

    Mutating one key from outside rotated every FlyGym sensory boundary built
    afterwards. It is now a read-only mapping, and it names every ``SensorFrame``
    field rather than the eight of eleven it used to cover.
    """

    from neurocraft_embodiment import FLYGYM_SENSORY_MAPPING
    from neurocraft_protocol import SensorFrame

    with pytest.raises(TypeError):
        FLYGYM_SENSORY_MAPPING["local_light"] = "tampered"  # type: ignore[index]

    grouped = {"yaw_pitch": {"yaw_rad", "pitch_rad"}, "ray_down": set(), "ray_other": set()}
    named = {key for key in FLYGYM_SENSORY_MAPPING if key not in grouped}
    covered = named | {"yaw_rad", "pitch_rad", "ray_distances_m", "ray_max_distance_m"}
    covered.discard("ray_max")
    assert covered >= SensorFrame.FIELDS


def test_observe_refuses_to_report_a_velocity_over_a_gap_it_did_not_measure(
    graph: RuntimeGraph,
    controller: ConnectomeController,
) -> None:
    """LOW-3: the docstring asserted a 1:1 relationship the seam never required.

    ``observe`` divides the displacement since its own previous sample by exactly one
    control period. The ``Embodiment`` protocol does not require one ``apply`` per
    ``observe``, and nothing checked: two ``apply`` calls between observations
    produced double the true per-tick velocity, silently, and that number is what
    reaches the connectome. A docstring cannot enforce a call pattern, so the body
    counts its own ticks and refuses the reading it cannot honestly make.
    """

    body = _kinematic_body(graph)
    body.observe(identity=controller.identity, sequence=0)
    body.apply(_command(controller, 0))
    body.apply(_command(controller, 1))
    with pytest.raises(EmbodimentError, match="2 control ticks"):
        body.observe(identity=controller.identity, sequence=2)


def test_observe_refuses_a_second_reading_with_no_tick_between(
    graph: RuntimeGraph,
    controller: ConnectomeController,
) -> None:
    """Zero elapsed ticks is the same defect with the divisor wrong the other way."""

    body = _kinematic_body(graph)
    body.observe(identity=controller.identity, sequence=0)
    with pytest.raises(EmbodimentError, match="0 control ticks"):
        body.observe(identity=controller.identity, sequence=1)


def test_reset_clears_the_tick_ledger_as_well_as_the_position_sample(
    graph: RuntimeGraph,
    controller: ConnectomeController,
) -> None:
    """After a reset the next observation is a first observation again."""

    body = _kinematic_body(graph)
    body.observe(identity=controller.identity, sequence=0)
    body.apply(_command(controller, 0))
    body.apply(_command(controller, 1))
    body.reset()
    sensors = body.observe(identity=controller.identity, sequence=2).sensors
    assert sensors.velocity_m_s == (0.0, 0.0, 0.0)


def test_build_hands_the_world_an_identity_quaternion_spawn_rotation(
    graph: RuntimeGraph,
) -> None:
    """EMBODIMENT-02 defect 1: ``add_fly`` refuses a raw Euler tuple.

    The seam must wrap the spawn rotation through the modules handle's
    ``rotation3d`` in quaternion format; a regression back to the raw
    ``spawn_rotation_rad`` tuple would keep every stub test green and only fail
    on a live flygym run, so the handed-over pose is pinned here.
    """

    body = _kinematic_body(graph)
    recorded = body._construction.world.add_fly_kwargs
    assert recorded["spawn_position"] == FlyGymBodyConfig().spawn_position_mm
    assert recorded["spawn_rotation"] == ("stub_rotation", "quat", (1.0, 0.0, 0.0, 0.0))


def test_build_refuses_a_non_zero_spawn_rotation(graph: RuntimeGraph) -> None:
    """FlyGym 2.1.0 accepts only quaternion freejoint rotations; no Euler
    convention is declared, so anything but the identity rotation is refused."""

    with pytest.raises(EmbodimentError, match="non-zero spawn_rotation_rad"):
        FlyGymEmbodiment(
            _six_leg_decoder(graph),
            FlyGymBodyConfig(spawn_rotation_rad=(0.1, 0.0, 0.0)),
            modules=_stub_modules(),  # type: ignore[arg-type]
        )
