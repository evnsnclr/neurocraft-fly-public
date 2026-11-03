"""NeuroMechFly (FlyGym 2.x) as a second body behind the same seam.

Status: executed once, at fixture scale. The historical receipt ``EMBODIMENT-02``
records three real control ticks and exact same-build replay on 2026-08-26 against
FlyGym 2.1.0. That smoke used a 128-neuron fixture with arbitrary sorted-ID
wing-neuron-to-leg assignments; it established construction and replay, not an
anatomical leg controller or a walking gait. ``flygym`` remains an optional extra
that the core lane does not install, and this module's current bytes have not
received a newer simulator execution. Every FlyGym API call below is written
against FlyGym 2.1.0 as documented.

Purity boundary
---------------
This module must never import ``flygym_demo``. That is where ``CPGController``,
``RuleBasedController``, ``HybridController`` and ``HybridTurningController`` live.
A CPG driven by a descending channel would produce a beautiful tripod gait that
belongs to the CPG, not to the connectome, and it would destroy the project's
central claim while making the demo look better. Only ``flygym`` itself is
imported, and only for anatomy, composition and simulation. A governance test
enforces this by scanning this package's sources.

Offline hygiene
---------------
``MeshType.SIMPLIFIED_MAX2000FACES`` is pinned. Those STLs ship inside the wheel.
``MeshType.FULLSIZE`` is the only path that reaches ``assets_lazy_loading`` and
downloads MD5-verified meshes over the network; selecting it would put a network
fetch inside the causal chain, so the config refuses it outright.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version as package_version
from typing import Any

from neurocraft_protocol import (
    DeterminismLevel,
    Observation,
    RunIdentity,
    SensorFrame,
    WorldState,
)

from .flygym_boundary import (
    FLYGYM_MOTOR_SCHEMA,
    NEUROMECHFLY_ADHESION_COUNT,
    FlyGymMotorDecoder,
)
from .flygym_sensing import (
    FLYGYM_DYNAMIC_SENSORY_CHANNELS,
    FLYGYM_SENSORY_CHANNELS,
    FLYGYM_SENSORY_MAPPING,
    FLYGYM_SENSORY_MAPPING_SCHEMA,
    FLYGYM_SENSORY_PARAMETERS,
)
from .identity import (
    CONSUMES_NEURAL_STATE,
    BoundaryIdentity,
    EmbodimentIdentity,
    canonical_sha256,
    float_hex,
    sorted_provenance,
)
from .seam import (
    CONTROL_HZ,
    SENSOR_FRAME_SCALAR_COUNT,
    EmbodimentCommand,
    EmbodimentError,
    EmbodimentTransition,
    observation_timing,
)

FLYGYM_EMBODIMENT_ID = "flygym-neuromechfly"
FLYGYM_EMBODIMENT_VERSION = "v1"
FLYGYM_BODY_SCHEMA = "neurocraft.embodiment.flygym.body.v1"

#: The only mesh set permitted. Anything else can trigger a network fetch.
PINNED_MESH_TYPE = "SIMPLIFIED_MAX2000FACES"

#: Exact pin for the extra. Kept in one place so the error message, the identity
#: and ``pyproject.toml``'s optional dependency cannot drift apart.
REQUIRED_FLYGYM_VERSION = "2.1.0"

INSTALL_COMMAND = f'UV_CACHE_DIR=.uv-cache uv pip install "flygym=={REQUIRED_FLYGYM_VERSION}"'

_MIN_PYTHON = (3, 12)


def _enum_label(value: Any, field: str) -> str:
    """The enum member's own ``name``, so an identity string cannot drift from it.

    The architecture choices below used to be hand-copied string literals sitting in
    :meth:`FlyGymBodyConfig.to_identity_material`, duplicating enum members chosen in
    :meth:`FlyGymEmbodiment._build`. Changing a preset without editing the matching
    literal would have left ``identity_sha256`` unrotated while the body behaved
    differently. They are now read off the objects ``_build`` actually used.
    """

    label = value if isinstance(value, str) else getattr(value, "name", None)
    if not isinstance(label, str) or not label:
        raise EmbodimentError(f"{field} does not expose a string enum name to hash")
    return label


def _actuated_dof_names(reported: Any) -> tuple[str, ...]:
    """Normalize whatever the skeleton reports for its actuated DOFs into names.

    FlyGym 2.1.0's exact return type is not pinned by this repository and has never
    been observed here, so an entry that is neither a string nor an object with a
    string ``name`` fails closed instead of being coerced into one.
    """

    try:
        entries = list(reported)
    except TypeError as exc:
        raise EmbodimentError("the actuated DOF preset did not return a sequence of DOFs") from exc
    return tuple(_enum_label(entry, "actuated DOF") for entry in entries)


def _assert_actuator_order(reported: tuple[str, ...], declared: tuple[str, ...]) -> None:
    """Refuse to run when the decoder's joint order is not the skeleton's order.

    :meth:`FlyGymEmbodiment.apply` hands ``joint_targets_rad`` to
    ``set_actuator_inputs`` positionally. If the decoder's bindings are ordered
    differently from the actuators the skeleton actually built, every leg is driven
    by another leg's target and nothing reports an error -- the fly simply moves
    wrongly. That is exactly the failure this project cannot afford to explain away
    afterwards, so it is a construction-time refusal.
    """

    if reported == declared:
        return
    if len(reported) != len(declared):
        raise EmbodimentError(
            "actuated DOF order mismatch: the skeleton reports "
            f"{len(reported)} actuated DOFs, the decoder binds {len(declared)}; "
            "joint targets are passed positionally, so the legs would be misdriven"
        )
    index = next(
        position
        for position, (left, right) in enumerate(zip(reported, declared, strict=True))
        if left != right
    )
    raise EmbodimentError(
        f"actuated DOF order mismatch at index {index}: the skeleton reports "
        f"{reported[index]!r}, the decoder binds {declared[index]!r}; "
        "joint targets are passed positionally, so the legs would be misdriven"
    )


def _velocity_m_s(
    previous_mm: tuple[float, float, float],
    current_mm: tuple[float, float, float],
) -> tuple[float, float, float]:
    """One control tick of displacement, MuJoCo z-up mm -> protocol y-up m/s."""

    period_s = 1.0 / CONTROL_HZ
    return (
        (current_mm[0] - previous_mm[0]) / 1000.0 / period_s,
        (current_mm[2] - previous_mm[2]) / 1000.0 / period_s,
        (current_mm[1] - previous_mm[1]) / 1000.0 / period_s,
    )


class FlyGymUnavailableError(EmbodimentError):
    """FlyGym is not installed, or the interpreter cannot support it.

    Raised instead of a bare ``ModuleNotFoundError`` so the message names the exact
    install command and the compliance consequences of running it.
    """


@dataclass(frozen=True, slots=True)
class FlyGymConstruction:
    """What ``_build`` actually constructed, read back off the objects it used.

    ``architecture`` and ``actuated_dof_order`` are recorded here rather than
    re-declared elsewhere: they are the two things that decide how the body behaves
    and that no configuration field can pin.
    """

    fly: Any
    world: Any
    simulation: Any
    architecture: tuple[tuple[str, str], ...]
    actuated_dof_order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FlyGymModules:
    """The lazily imported FlyGym surface, kept behind one typed handle."""

    anatomy: Any
    compose: Any
    simulation: Any
    #: ``flygym.utils.math.Rotation3D`` (or a stub): ``world.add_fly`` refuses a raw
    #: Euler tuple, so the spawn rotation must be wrapped before construction.
    rotation3d: Any
    version: str
    mujoco_version: str


def require_flygym() -> FlyGymModules:
    """Import FlyGym on demand, at the exact pinned version, with an actionable failure.

    Importing this module never triggers this call; only constructing a
    :class:`FlyGymEmbodiment` does. That keeps ``neurocraft_embodiment`` importable,
    lintable and type-checkable in an environment that has no MuJoCo.

    The version refusal, and why it is the cheapest guard this lane has
    ------------------------------------------------------------------
    :data:`REQUIRED_FLYGYM_VERSION` used to be compared against exactly one thing: the
    string in ``pyproject.toml``, by a test. Nothing compared it to an imported
    package. This function read ``package_version("flygym")`` and wrote whatever came
    back straight into the identity's provenance, so a 2.0 or a 3.0 install would have
    been driven with an API surface written against 2.1.0 and recorded under a hash
    that faithfully reported the mismatch nobody was checking.

    Every FlyGym call in this module is written from documentation and has never been
    executed here. That is precisely the situation in which a silently different
    version is most dangerous and a hard refusal costs least: there is no accumulated
    evidence that any other version behaves the same way, because there is no
    accumulated evidence at all.

    ``mujoco`` deliberately carries no equivalent pin. This repository declares no
    mujoco version -- it arrives as a FlyGym dependency -- so there is nothing to
    compare against, and inventing a pin here would be a claim rather than a check.
    Its version is recorded in provenance and left as a record.
    """

    if sys.version_info < _MIN_PYTHON:
        raise FlyGymUnavailableError(
            f"flygym {REQUIRED_FLYGYM_VERSION} requires Python >= 3.12 "
            f"(this interpreter is {sys.version_info.major}.{sys.version_info.minor}). "
            "The workspace floor is 3.11; use the pinned 3.13 interpreter for this lane."
        )
    try:
        import flygym.anatomy as flygym_anatomy
        import flygym.compose as flygym_compose
        import flygym.simulation as flygym_simulation
        import flygym.utils.math as flygym_math
        import mujoco
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise FlyGymUnavailableError(
            "the FlyGym embodiment requires the optional 'flygym' extra, which the "
            "core lane deliberately does not install.\n"
            f"  install: {INSTALL_COMMAND}\n"
            "  note:    this pulls mujoco, numba/llvmlite, matplotlib, pillow and "
            "imageio-ffmpeg (which bundles an FFmpeg binary whose license PyPI does "
            "not declare). Run the FlyGym lane's own SBOM/licence/audit scope after "
            "installing; do not fold it into the core lane's compliance claim."
        ) from exc
    installed = package_version("flygym")
    if installed != REQUIRED_FLYGYM_VERSION:
        raise FlyGymUnavailableError(
            f"this lane is pinned to flygym {REQUIRED_FLYGYM_VERSION} and found "
            f"{installed!r} installed. Every FlyGym call in this module is written "
            f"against {REQUIRED_FLYGYM_VERSION}'s documented API and has never been "
            "executed against any version, so a different one is not a smaller risk "
            "than no install at all -- it is the same risk with the failure moved "
            "past the point where it is attributable.\n"
            f"  install: {INSTALL_COMMAND}"
        )
    return FlyGymModules(
        anatomy=flygym_anatomy,
        compose=flygym_compose,
        simulation=flygym_simulation,
        rotation3d=flygym_math.Rotation3D,
        version=installed,
        mujoco_version=str(mujoco.__version__),
    )


@dataclass(frozen=True, slots=True)
class FlyGymBodyConfig:
    """Every pinned NeuroMechFly construction and physics constant.

    Defaults reproduce FlyGym 2.1.0's own documented values so that a change to the
    body is a change to this dataclass, and therefore to the embodiment identity.

    The architecture choices (axis order, joint preset, kinematic pose preset,
    actuated-DOF preset, actuator type) are deliberately *not* fields here and not
    hashed here. They are enum members selected in :meth:`FlyGymEmbodiment._build`,
    and they are read back off those objects into the embodiment identity's
    provenance, so that changing one in ``_build`` rotates the hash without anyone
    having to remember to edit a matching literal.
    """

    schema: str = FLYGYM_BODY_SCHEMA
    fly_name: str = "nmf"
    world_name: str = "flat_ground"
    mesh_type: str = PINNED_MESH_TYPE
    # mujoco_globals.yaml: timestep 1e-4 s, Euler integrator, Newton solver.
    physics_timestep_s: float = 1.0e-4
    physics_steps_per_control_tick: int = 500
    warmup_duration_s: float = 0.05
    joint_stiffness: float = 0.05
    joint_damping: float = 0.06
    actuator_kp: float = 45.0
    actuator_force_min: float = -65.0
    actuator_force_max: float = 65.0
    adhesion_gain: float = 40.0
    spawn_position_mm: tuple[float, float, float] = (0.0, 0.0, 0.5)
    spawn_rotation_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Design-A sensory mapping constants (all GL-free; see the class docstring).
    ray_max_distance_m: float = 6.0
    light_span_mm: float = 5.0
    seed: int = 0
    enable_vision: bool = False

    def __post_init__(self) -> None:
        if self.schema != FLYGYM_BODY_SCHEMA:
            raise EmbodimentError("unsupported flygym body schema")
        if self.mesh_type != PINNED_MESH_TYPE:
            raise EmbodimentError(
                "mesh_type is pinned to SIMPLIFIED_MAX2000FACES; FULLSIZE reaches "
                "flygym.utils.assets_lazy_loading and downloads meshes over the "
                "network, which must never sit inside the causal chain"
            )
        if self.enable_vision:
            raise EmbodimentError(
                "vision-in-the-loop is not enabled in v1: GL rendering is not "
                "bit-reproducible across drivers/backends, so it requires a Tier-2 "
                "DeterminismLevel.NUMERICAL_TOLERANCE declaration, recorded GL "
                "vendor/renderer strings and a preregistered replay tolerance"
            )
        if self.physics_timestep_s <= 0.0 or not math.isfinite(self.physics_timestep_s):
            raise EmbodimentError("physics timestep must be finite and positive")
        if (
            isinstance(self.physics_steps_per_control_tick, bool)
            or not isinstance(self.physics_steps_per_control_tick, int)
            or self.physics_steps_per_control_tick < 1
        ):
            raise EmbodimentError("physics steps per control tick must be a positive integer")
        if self.actuator_force_min >= self.actuator_force_max:
            raise EmbodimentError("actuator force range is inverted")
        if self.light_span_mm <= 0.0 or self.ray_max_distance_m <= 0.0:
            raise EmbodimentError("sensory mapping spans must be positive")

    def to_identity_material(self) -> dict[str, Any]:
        return {
            "actuator_force_max": float_hex(self.actuator_force_max),
            "actuator_force_min": float_hex(self.actuator_force_min),
            "actuator_kp": float_hex(self.actuator_kp),
            "adhesion_gain": float_hex(self.adhesion_gain),
            "enable_vision": self.enable_vision,
            "fly_name": self.fly_name,
            "joint_damping": float_hex(self.joint_damping),
            "joint_stiffness": float_hex(self.joint_stiffness),
            "light_span_mm": float_hex(self.light_span_mm),
            "mesh_type": self.mesh_type,
            "physics_steps_per_control_tick": self.physics_steps_per_control_tick,
            "physics_timestep_s": float_hex(self.physics_timestep_s),
            "ray_max_distance_m": float_hex(self.ray_max_distance_m),
            "schema": self.schema,
            "seed": self.seed,
            "spawn_position_mm": [float_hex(value) for value in self.spawn_position_mm],
            "spawn_rotation_rad": [float_hex(value) for value in self.spawn_rotation_rad],
            "warmup_duration_s": float_hex(self.warmup_duration_s),
            "world_name": self.world_name,
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.to_identity_material())


class FlyGymEmbodiment:
    """NeuroMechFly v2 behind the embodiment seam.

    Consumes ``EmbodimentCommand.neural_state`` through its own hashed
    :class:`FlyGymMotorDecoder`; the primitive ``MotorVector`` on the command is
    ignored, as declared by ``motor_currency``. That is the whole reason the seam
    carries both: squeezing a six-legged body through five thrust/torque axes would
    need a 5 -> 42 expansion that no honest hash can justify.

    Sensory mapping (Design A, GL-free)
    -----------------------------------
    The controller's ``PhysicalFeatureEncoder`` is untouched, so this body must
    speak the existing ``SensorFrame``. State that costs, out loud: 1442 ommatidia,
    66 joint angles and six leg contact wrenches are discarded to produce four
    scalars, one of them quantised to sixteen levels. On ``FlatGroundWorld`` there
    are no occluders either, so the forward/left/right rays saturate at maximum
    range and the only varying inputs are the down ray and the height-derived light
    level. Eight of the twenty scalars this body emits do not move at all; which
    eight, and why, is declared per slot in ``flygym_sensing`` and hashed into the
    sensory boundary identity rather than left in this paragraph.

    The ``PhysicalFeatureEncoder`` the pinned controller uses reads only
    ``ray_distances_m[0]``, ``[2]``, ``[3]`` and ``local_light``. Three of those four
    are declared constants on this world, so the encoder-visible input to the graph
    may not vary at all here. That is a limit on what a run in this configuration
    can support, not a defect to be papered over, and it is now readable off the
    identity instead of only from this file. This is deliberately the
    *comparability* configuration: identical
    ``adapter_hash`` and ``observation_schema_hash`` to the Minecraft lane, with only
    the world and the decoder differing. The wide boundary in ``flygym_boundary``
    is what a body can actually tell the brain; wiring it to the sensory side needs
    a controller that accepts a different encoder.
    """

    def __init__(
        self,
        decoder: FlyGymMotorDecoder,
        config: FlyGymBodyConfig | None = None,
        *,
        modules: FlyGymModules | None = None,
    ) -> None:
        if not isinstance(decoder, FlyGymMotorDecoder):
            raise EmbodimentError("flygym embodiment requires a FlyGymMotorDecoder")
        self._config = config or FlyGymBodyConfig()
        self._decoder = decoder
        self._modules = modules or require_flygym()
        self._previous_observed_position_mm: tuple[float, float, float] | None = None
        self._applied_ticks = 0
        self._ticks_at_previous_observation = 0
        self._construction = self._build()
        self._simulation = self._construction.simulation
        self._identity = self._build_identity()

    # -- construction ----------------------------------------------------------

    def _build(self) -> FlyGymConstruction:
        config = self._config
        anatomy = self._modules.anatomy
        compose = self._modules.compose
        simulation = self._modules.simulation

        axis_order = anatomy.AxisOrder.YAW_PITCH_ROLL
        joint_preset = anatomy.JointPreset.LEGS_ONLY
        kinematic_pose_preset = compose.KinematicPosePreset.NEUTRAL
        actuated_dof_preset = anatomy.ActuatedDOFPreset.LEGS_ACTIVE_ONLY
        actuator_type = compose.ActuatorType.POSITION

        neutral_pose = kinematic_pose_preset.get_pose_by_axis_order(axis_order)
        skeleton = anatomy.Skeleton(axis_order=axis_order, joint_preset=joint_preset)
        fly = compose.NeuroMechFly(name=config.fly_name)
        fly.add_joints(
            skeleton,
            neutral_pose=neutral_pose,
            stiffness=config.joint_stiffness,
            damping=config.joint_damping,
        )
        actuated = skeleton.get_actuated_dofs_from_preset(actuated_dof_preset)
        fly.add_actuators(
            actuated,
            actuator_type,
            neutral_input=neutral_pose,
            kp=config.actuator_kp,
            forcerange=(config.actuator_force_min, config.actuator_force_max),
        )
        actuated_dof_order = _actuated_dof_names(actuated)
        _assert_actuator_order(actuated_dof_order, self._decoder.dof_order)
        fly.add_leg_adhesion(gain=config.adhesion_gain)

        world = compose.world.FlatGroundWorld(name=config.world_name)
        # FlyGym 2.1.0's freejoint accepts the spawn rotation only in quaternion
        # format. The identity rotation converts exactly; anything else would need
        # a declared Euler convention, so it is refused instead of guessed.
        if config.spawn_rotation_rad != (0.0, 0.0, 0.0):
            raise EmbodimentError(
                "non-zero spawn_rotation_rad is not supported yet: FlyGym 2.1.0 "
                "requires a quaternion freejoint rotation and this lane declares "
                "no Euler-to-quaternion convention"
            )
        world.add_fly(
            fly,
            spawn_position=config.spawn_position_mm,
            spawn_rotation=self._modules.rotation3d(format="quat", values=(1.0, 0.0, 0.0, 0.0)),
        )
        sim = simulation.Simulation(world, timestep=config.physics_timestep_s)
        sim.reset()
        sim.warmup(duration_s=config.warmup_duration_s)
        return FlyGymConstruction(
            fly=fly,
            world=world,
            simulation=sim,
            architecture=(
                ("actuated_dof_preset", _enum_label(actuated_dof_preset, "ActuatedDOFPreset")),
                ("actuator_type", _enum_label(actuator_type, "ActuatorType")),
                ("axis_order", _enum_label(axis_order, "AxisOrder")),
                ("joint_preset", _enum_label(joint_preset, "JointPreset")),
                (
                    "kinematic_pose_preset",
                    _enum_label(kinematic_pose_preset, "KinematicPosePreset"),
                ),
            ),
            actuated_dof_order=actuated_dof_order,
        )

    def _build_identity(self) -> EmbodimentIdentity:
        config = self._config
        sensory = BoundaryIdentity(
            role="sensory",
            schema=FLYGYM_SENSORY_MAPPING_SCHEMA,
            identity_sha256=canonical_sha256(
                {
                    "config_sha256": config.identity_sha256,
                    "discarded": [
                        "ommatidia_readouts",
                        "joint_angles",
                        "joint_velocities",
                        "contact_wrenches",
                    ],
                    "channels": [entry.to_identity_material() for entry in FLYGYM_SENSORY_CHANNELS],
                    "gl_used": False,
                    "light_model": "linear_thorax_height_gradient_16_levels",
                    "mapping": dict(FLYGYM_SENSORY_MAPPING),
                    "parameters": list(FLYGYM_SENSORY_PARAMETERS),
                    "schema": FLYGYM_SENSORY_MAPPING_SCHEMA,
                    "world_frame": "mujoco_zup_mm_to_protocol_yup_m",
                }
            ),
            # The width is the protocol frame's; the records are this body's own
            # sourcing table, which says per slot which scalars actually move.
            channel_count=SENSOR_FRAME_SCALAR_COUNT,
            channels=FLYGYM_SENSORY_CHANNELS,
            declared_dynamic_channels=FLYGYM_DYNAMIC_SENSORY_CHANNELS,
            free_parameters=len(FLYGYM_SENSORY_PARAMETERS),
        )
        motor = BoundaryIdentity(
            role="motor",
            schema=FLYGYM_MOTOR_SCHEMA,
            identity_sha256=self._decoder.identity_sha256,
            channel_count=self._decoder.channel_count,
            channels=self._decoder.channels,
            declared_dynamic_channels=self._decoder.dynamic_channel_count,
            free_parameters=self._decoder.free_parameters,
        )
        return EmbodimentIdentity(
            embodiment_id=FLYGYM_EMBODIMENT_ID,
            embodiment_version=FLYGYM_EMBODIMENT_VERSION,
            configuration_sha256=config.identity_sha256,
            sensory_boundary=sensory,
            motor_boundary=motor,
            motor_currency=CONSUMES_NEURAL_STATE,
            seed=config.seed,
            control_hz=CONTROL_HZ,
            physics_steps_per_control_tick=config.physics_steps_per_control_tick,
            physics_timestep_s=config.physics_timestep_s,
            determinism_level=DeterminismLevel.SAME_BUILD_EXACT,
            provenance=sorted_provenance(
                {
                    **dict(self._construction.architecture),
                    "flygym_version": self._modules.version,
                    "gl_backend": "not_used_vision_disabled",
                    "mesh_type": config.mesh_type,
                    "mujoco_version": self._modules.mujoco_version,
                    "numba_threads": "1",
                    "world": "FlatGroundWorld",
                }
            ),
        )

    # -- seam ------------------------------------------------------------------

    @property
    def identity(self) -> EmbodimentIdentity:
        return self._identity

    @property
    def config(self) -> FlyGymBodyConfig:
        return self._config

    @property
    def simulation(self) -> Any:
        return self._simulation

    def provenance(self) -> Mapping[str, str]:
        return self._identity.provenance_mapping()

    def reset(self) -> None:
        self._simulation.reset()
        self._simulation.warmup(duration_s=self._config.warmup_duration_s)
        self._previous_observed_position_mm = None
        self._applied_ticks = 0
        self._ticks_at_previous_observation = 0

    def observe(self, *, identity: RunIdentity, sequence: int) -> Observation:
        """Encode the current thorax state as a protocol ``Observation``.

        Velocity choice, stated because it is the honest one and not the obvious one:
        this method keeps its **own** previous position sample, updated here, and
        reports the displacement between consecutive *observations* divided by one
        control period. It does not return whatever ``apply`` last measured.

        That divisor is only correct if exactly one ``apply`` happened in between, and
        the ``Embodiment`` protocol does not require a caller to alternate. This used
        to be asserted in this docstring and enforced nowhere: two ``apply`` calls
        between observations reported double the true per-tick velocity, with no
        error, and that number is what the connectome integrates. A docstring cannot
        enforce a call pattern, so the body counts its own control ticks and refuses
        to report a velocity over an interval it did not measure. The first
        observation has no earlier sample and is honestly ``(0, 0, 0)``; ``reset``
        makes the next observation a first observation again.

        The rejected alternative was sharing one previous-position slot with
        ``apply``. That is what this code did, and it made the reported velocity
        permanently zero: no physics runs between a tick's ``apply`` and the next
        tick's ``observe``, so the observation differenced the position ``apply`` had
        just stored against itself, every tick, forever.
        """

        game_tick, sent_monotonic_ns = observation_timing(sequence)
        config = self._config
        position_mm = self._thorax_position_mm()
        quaternion = self._thorax_quaternion()
        yaw_rad, pitch_rad = _yaw_pitch_from_quaternion(quaternion)
        velocity = self._observed_velocity_m_s(position_mm)
        contact_found, ground_contact = self._contacts()
        # MuJoCo NeuroMechFly is z-up in millimetres; the protocol frame is y-up in
        # metres. The swap is declared in the sensory boundary identity.
        position_m = (
            position_mm[0] / 1000.0,
            position_mm[2] / 1000.0,
            position_mm[1] / 1000.0,
        )
        height_mm = max(0.0, position_mm[2])
        ray_max = config.ray_max_distance_m
        down_ray = max(0.0, min(ray_max, height_mm / 1000.0))
        local_light = max(0, min(15, round(15.0 * min(1.0, height_mm / config.light_span_mm))))
        return Observation(
            identity=identity,
            observation_seq=sequence,
            game_tick=game_tick,
            sent_monotonic_ns=sent_monotonic_ns,
            sensors=SensorFrame(
                position_m=position_m,
                velocity_m_s=velocity,
                yaw_rad=yaw_rad,
                pitch_rad=pitch_rad,
                on_ground=contact_found,
                horizontal_collision=False,
                vertical_collision=ground_contact,
                in_fluid=False,
                local_light=local_light,
                ray_max_distance_m=ray_max,
                ray_distances_m=(ray_max, ray_max, ray_max, ray_max, ray_max, down_ray),
            ),
        )

    def apply(self, command: EmbodimentCommand) -> EmbodimentTransition:
        if not isinstance(command, EmbodimentCommand):
            raise EmbodimentError("flygym embodiment accepts only an EmbodimentCommand")
        config = self._config
        compose = self._modules.compose
        actuation = self._decoder.decode(command.neural_state)
        if len(actuation.adhesion) != NEUROMECHFLY_ADHESION_COUNT:
            raise EmbodimentError(
                "NeuroMechFly requires exactly six adhesion channels, one per leg"
            )
        # Sampled before the step loop and differenced against the sample after it,
        # so the reported velocity is exactly this tick's displacement whatever the
        # caller's observe() pattern is -- including on the very first tick, which
        # used to report (0, 0, 0) because no previous sample existed yet.
        position_before_mm = self._thorax_position_mm()
        self._simulation.set_actuator_inputs(
            self._config.fly_name,
            compose.ActuatorType.POSITION,
            actuation.joint_targets_rad,
        )
        self._simulation.set_leg_adhesion_states(self._config.fly_name, actuation.adhesion)
        for _ in range(config.physics_steps_per_control_tick):
            self._simulation.step()
        self._applied_ticks += 1

        position_mm = self._thorax_position_mm()
        quaternion = self._thorax_quaternion()
        yaw_rad, pitch_rad = _yaw_pitch_from_quaternion(quaternion)
        velocity = _velocity_m_s(position_before_mm, position_mm)
        contact_found, ground_contact = self._contacts()
        return EmbodimentTransition(
            world_after=WorldState(
                position_m=(
                    position_mm[0] / 1000.0,
                    position_mm[2] / 1000.0,
                    position_mm[1] / 1000.0,
                ),
                velocity_m_s=velocity,
                yaw_rad=yaw_rad,
                pitch_rad=pitch_rad,
                on_ground=contact_found,
                horizontal_collision=False,
                vertical_collision=ground_contact,
                in_fluid=False,
            ),
            # mujoco_globals.yaml gravity is [0, 0, -9810] mm/s^2 = -9.81 m/s^2,
            # reported in the protocol's y-up frame.
            passive_forces=(0.0, -9.81, 0.0),
            external_forces=(0.0, 0.0, 0.0),
            physics_steps=config.physics_steps_per_control_tick,
            simulated_time_s=config.physics_steps_per_control_tick * config.physics_timestep_s,
        )

    # -- MuJoCo readers --------------------------------------------------------

    def _thorax_position_mm(self) -> tuple[float, float, float]:
        positions = self._simulation.get_body_positions(self._config.fly_name)
        row = positions[0]
        return (float(row[0]), float(row[1]), float(row[2]))

    def _thorax_quaternion(self) -> tuple[float, float, float, float]:
        rotations = self._simulation.get_body_rotations(self._config.fly_name)
        row = rotations[0]
        return (float(row[0]), float(row[1]), float(row[2]), float(row[3]))

    def _observed_velocity_m_s(
        self,
        position_mm: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Displacement since the previous observation, over exactly one control tick.

        ``_velocity_m_s`` divides by one control period, so any other number of ticks
        between two observations would make the reported velocity wrong by that
        factor. Refused rather than scaled: a caller observing on a different cadence
        than it applies has a bug about which tick the reading belongs to, and
        silently returning an interval-averaged velocity would hide it inside the
        hash-chained trace.
        """

        previous = self._previous_observed_position_mm
        elapsed = self._applied_ticks - self._ticks_at_previous_observation
        self._previous_observed_position_mm = position_mm
        self._ticks_at_previous_observation = self._applied_ticks
        if previous is None:
            return (0.0, 0.0, 0.0)
        if elapsed != 1:
            raise EmbodimentError(
                f"observe() was called {elapsed} control ticks after the previous "
                "observation; the reported velocity divides by exactly one control "
                "period, so it can only be measured one tick at a time"
            )
        return _velocity_m_s(previous, position_mm)

    def _contacts(self) -> tuple[bool, bool]:
        """Contact with the ground, exactly as recorded in ``FLYGYM_SENSORY_MAPPING``.

        Returns ``(contact_found, ground_contact)``. Both are unfiltered readings of
        the same call: the first is any nonzero entry of the contact array, the second
        is any contact-force row with a nonzero magnitude. There is no segment-type
        test here, so the two will usually agree, and ordinary tarsal contact raises
        both. The mapping string says so.

        What this reading is *not*
        --------------------------
        It is not a horizontal collision, and it used to be wired to that flag. The
        call reports contact with the floor; the protocol's two contact flags carry
        the Minecraft entity's meanings (``FlySensors.java`` binds them to
        ``Entity.horizontalCollision`` and ``Entity.verticalCollision``), and the
        reference body follows the same convention -- walls and pillars raise the
        horizontal flag, the floor and ceiling raise the vertical one. So a fly
        standing on the ground has ``vertical_collision`` true, and this is the
        reading that says so. ``horizontal_collision`` is false on
        ``FlatGroundWorld`` because that world has no vertical geometry to strike; it
        is declared a literal constant in the boundary identity rather than dressed
        up as a measurement.

        None of this has been run against a real FlyGym install. Unverified, stated
        plainly: the arity, row shape, frame and row identity of
        ``get_ground_contact_info``'s return, and whether a resting fly's solver
        forces are large enough that ``> 0.0`` is a threshold rather than a
        float-equality test. The first real run is the integration test, and the
        empirical force distribution it produces is what a declared threshold would
        have to be sized from.
        """

        info = self._simulation.get_ground_contact_info(self._config.fly_name)
        # 2.0.2 renamed contact_active -> contact_found and made it a raw float.
        contact_found = bool(any(float(value) > 0.0 for value in info[0]))
        forces = info[1]
        ground_contact = bool(
            any(
                math.sqrt(float(row[0]) ** 2 + float(row[1]) ** 2 + float(row[2]) ** 2) > 0.0
                for row in forces
            )
        )
        return contact_found, ground_contact


def _yaw_pitch_from_quaternion(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float]:
    """ZYX Euler yaw/pitch from a MuJoCo (w, x, y, z) quaternion, protocol-clamped."""

    w, x, y, z = quaternion
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi
    limit = math.pi / 2
    return yaw, max(-limit, min(limit, pitch))
