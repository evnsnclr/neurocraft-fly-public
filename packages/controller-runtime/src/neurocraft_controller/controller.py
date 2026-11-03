"""Causal connectome-to-primitive-action controller."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar

from neurocraft_connectome import FixtureTier, RuntimeGraph
from neurocraft_neural import (
    DelayOverlay,
    DynamicsConfig,
    MotorDecoder,
    MotorFrame,
    MotorPool,
    NeuralEngine,
    NeuralSnapshot,
    NeuralStepRecord,
    PoolMember,
    SensorChannel,
    SensorTarget,
    SensoryAdapter,
    SignOverlay,
    Stimulation,
    TopologyOverlay,
    WeightOverlay,
)
from neurocraft_protocol import (
    ACTION_TARGET_OFFSET_TICKS,
    Action,
    MotorVector,
    Observation,
    RunIdentity,
    RunMode,
)

from .config import ControllerConfiguration, canonical_sha256
from .features import PhysicalFeatureEncoder


class ControllerIntegrityError(RuntimeError):
    """Identity, sequence, or graph evidence failed closed."""


@dataclass(frozen=True, slots=True)
class ControllerStepEvidence:
    observation: Observation
    observation_sha256: str
    feature_schema_sha256: str
    feature_values: Mapping[str, float]
    feature_values_sha256: str
    encoded_stimulation: Mapping[str, float]
    encoded_stimulation_sha256: str
    direct_stimulation: Mapping[str, float]
    direct_stimulation_sha256: str
    neural_step_records: tuple[NeuralStepRecord, ...]
    state_before_sha256: str
    state_after_sha256: str
    state_before_activity: tuple[float, ...]
    state_after_activity: tuple[float, ...]
    neural_step_start: int
    neural_step_end: int
    motor_frame: MotorFrame
    command_seq: int
    motors: MotorVector
    action_sha256: str
    interventions: tuple[str, ...]


class ConnectomeController:
    """The only faithful path from physical sensors to primitive motors.

    No raw observation reaches the decoder. The decoder accepts a verified
    NeuralState produced after five deterministic updates of the bound graph.
    """

    PROJECTION_MATERIAL: ClassVar[dict[str, Any]] = {
        "input_order": ["left_motor", "right_motor"],
        "output_order": [
            "forward_thrust",
            "lateral_thrust",
            "vertical_thrust",
            "yaw_torque",
            "pitch_torque",
        ],
        "matrix": [
            ["0x1.0000000000000p-1", "0x1.0000000000000p-1"],
            ["0x0.0p+0", "0x0.0p+0"],
            ["0x1.0000000000000p-1", "0x1.0000000000000p-1"],
            ["-0x1.0000000000000p-1", "0x1.0000000000000p-1"],
            ["0x0.0p+0", "0x0.0p+0"],
        ],
        "nonlinearity": "componentwise_clip_to_minus1_plus1",
        "schema": "neurocraft.primitive_projection.v1",
    }

    def __init__(
        self,
        graph: RuntimeGraph,
        *,
        run_id: str,
        entity_id: str,
        session_epoch: int = 0,
        configuration: ControllerConfiguration | None = None,
        weight_overlay: WeightOverlay | None = None,
        topology_overlay: TopologyOverlay | None = None,
    ) -> None:
        graph.assert_integrity()
        config = configuration or ControllerConfiguration()
        is_synthetic = graph.manifest.fixture_tier is FixtureTier.SYNTHETIC_ALGEBRA
        if config.mode == "synthetic" and not is_synthetic:
            raise ControllerIntegrityError("synthetic mode requires an explicitly synthetic graph")
        if config.mode != "synthetic" and is_synthetic:
            raise ControllerIntegrityError("synthetic graph cannot replace a connectome controller")

        self.graph = graph
        self.configuration = config
        self.features = PhysicalFeatureEncoder()
        self.adapter = SensoryAdapter(graph, self._sensor_channels(config))
        self.sign_overlay = self._sign_overlay(graph, config)
        self.dynamics = DynamicsConfig()
        self.delay_overlay = DelayOverlay.uniform(
            graph,
            self.dynamics.default_delay_steps,
            provenance="controller_v1_uniform_fixed_step_delay",
        )
        self.engine = NeuralEngine(
            graph,
            sign_overlay=self.sign_overlay,
            config=self.dynamics,
            delay_overlay=self.delay_overlay,
            weight_overlay=weight_overlay,
            topology_overlay=topology_overlay,
            sensory_adapter_sha256=self.adapter.identity_sha256,
        )
        has_weight_control = self.engine.weight_overlay.identity_sha256 != "0" * 64
        has_topology_control = self.engine.topology_overlay.identity_sha256 != "0" * 64
        if has_weight_control and has_topology_control:
            raise ControllerIntegrityError(
                "weight and topology controls must run separately for causal attribution"
            )
        if has_weight_control:
            self.run_mode = RunMode.SHUFFLED_WEIGHT
        elif has_topology_control:
            self.run_mode = RunMode.DEGREE_PRESERVING_RANDOM
        elif config.mode == "lesion":
            self.run_mode = RunMode.LESION
        elif config.mode == "synthetic":
            self.run_mode = RunMode.SYNTHETIC
        else:
            self.run_mode = RunMode.FAITHFUL
        self.decoder = MotorDecoder(graph, self._motor_pools(config))

        self.adapter_identity_sha256 = canonical_sha256(
            {
                "feature_schema_sha256": self.features.identity_sha256,
                "neural_adapter_sha256": self.adapter.identity_sha256,
                "schema": "neurocraft.boundary_adapter.identity.v1",
            }
        )
        self.dynamics_identity_sha256 = canonical_sha256(
            {
                "delay_overlay_sha256": self.delay_overlay.identity_sha256,
                "dynamics_config_sha256": self.dynamics.identity_sha256,
                "engine_control_sha256": self.engine.control_identity_sha256,
                "schema": "neurocraft.dynamics.identity.v1",
                "sign_overlay_sha256": self.sign_overlay.identity_sha256,
                "topology_overlay_sha256": self.engine.topology_overlay.identity_sha256,
                "weight_overlay_sha256": self.engine.weight_overlay.identity_sha256,
            }
        )
        self.projection_identity_sha256 = canonical_sha256(self.PROJECTION_MATERIAL)
        self.decoder_identity_sha256 = canonical_sha256(
            {
                "motor_decoder_sha256": self.decoder.identity_sha256,
                "primitive_projection_sha256": self.projection_identity_sha256,
                "schema": "neurocraft.decoder.identity.v1",
            }
        )
        self.identity = RunIdentity(
            run_id=run_id,
            session_epoch=session_epoch,
            entity_id=entity_id,
            graph_hash=graph.manifest.runtime_identity_sha256,
            adapter_hash=self.adapter_identity_sha256,
            dynamics_hash=self.dynamics_identity_sha256,
            decoder_hash=self.decoder_identity_sha256,
            observation_schema_hash=self.features.identity_sha256,
        )
        self._last_observation_seq = -1
        self._last_game_tick = -1
        self._last_command_seq = -1
        self._neural_step_offset = 0
        self._stimulation: Stimulation | None = None
        self._named_lesion_ids: set[str] = set()
        self._last_evidence: ControllerStepEvidence | None = None

    @classmethod
    def create(
        cls,
        graph: RuntimeGraph,
        *,
        run_id: str,
        entity_id: str,
        session_epoch: int = 0,
        configuration: ControllerConfiguration | None = None,
        weight_overlay: WeightOverlay | None = None,
        topology_overlay: TopologyOverlay | None = None,
    ) -> ConnectomeController:
        return cls(
            graph,
            run_id=run_id,
            entity_id=entity_id,
            session_epoch=session_epoch,
            configuration=configuration,
            weight_overlay=weight_overlay,
            topology_overlay=topology_overlay,
        )

    @property
    def last_evidence(self) -> ControllerStepEvidence | None:
        return self._last_evidence

    @property
    def active_stimulation_id(self) -> str | None:
        return None if self._stimulation is None else self._stimulation.intervention_id

    @property
    def active_named_lesions(self) -> tuple[str, ...]:
        return tuple(sorted(self._named_lesion_ids))

    def reset(self) -> None:
        """Reset neural/intervention state without rewinding the live session clocks."""

        self._neural_step_offset += self.engine.step_index
        self.engine.reset()
        self._stimulation = None
        self._named_lesion_ids.clear()
        self._last_evidence = None

    def snapshot(self) -> NeuralSnapshot:
        return self.engine.snapshot()

    def set_lesion(self, body_ids: Sequence[int], *, ablated: bool = True) -> None:
        if self.configuration.mode not in {"lesion", "synthetic"}:
            raise ControllerIntegrityError("lesions require an explicit lesion/synthetic run mode")
        self.engine.set_ablated(body_ids, ablated=ablated)

    def set_named_lesion(
        self,
        intervention_id: str,
        body_ids: Sequence[int],
        *,
        ablated: bool = True,
    ) -> None:
        if not isinstance(intervention_id, str) or not intervention_id:
            raise ControllerIntegrityError("named lesion requires an intervention ID")
        self.set_lesion(body_ids, ablated=ablated)
        if ablated:
            self._named_lesion_ids.add(intervention_id)
        else:
            self._named_lesion_ids.discard(intervention_id)

    def set_stimulation(self, intervention_id: str, values: Mapping[int, float]) -> None:
        self._stimulation = Stimulation.from_body_values(
            self.graph,
            intervention_id,
            values,
        )

    def clear_stimulation(self) -> None:
        self._stimulation = None

    def step(self, observation: Observation, *, command_seq: int) -> Action:
        self._validate_observation(observation, command_seq)
        state_before = self.engine.current_state()
        features = self.features.encode(observation.sensors)
        encoded = self.adapter.encode(features.values)
        encoded_values = MappingProxyType(
            {
                f"{channel}:{self.graph.nodes[index].body_id}": value
                for channel, index, value in encoded.assignments
            }
        )
        records: list[NeuralStepRecord] = []
        direct_stimulation_values = MappingProxyType(
            {}
            if self._stimulation is None
            else {
                str(self.graph.nodes[index].body_id): value
                for index, value in self._stimulation.assignments
            }
        )
        result = None
        for _ in range(self.configuration.neural_steps_per_game_tick):
            result = self.engine.step(encoded, stimulation=self._stimulation)
            records.append(result.record)
        assert result is not None
        motor_frame = self.decoder.decode(result.state)
        motors = self._project_motor_frame(motor_frame)
        action = Action(
            identity=self.identity,
            command_seq=command_seq,
            observation_seq=observation.observation_seq,
            observation_tick=observation.game_tick,
            target_tick=observation.game_tick + ACTION_TARGET_OFFSET_TICKS,
            valid_through_tick=observation.game_tick + ACTION_TARGET_OFFSET_TICKS,
            neural_step_start=self._neural_step_offset + state_before.step_index,
            neural_step_end=self._neural_step_offset + result.state.step_index,
            state_digest=result.state.state_sha256,
            motors=motors,
            neutral=False,
        )
        action_sha256 = canonical_sha256(action.to_dict())
        intervention_ids = tuple(sorted(self._named_lesion_ids))
        if self._stimulation is not None:
            intervention_ids += (self._stimulation.intervention_id,)
        self._last_evidence = ControllerStepEvidence(
            observation=observation,
            observation_sha256=canonical_sha256(observation.to_dict()),
            feature_schema_sha256=self.features.identity_sha256,
            feature_values=MappingProxyType(dict(features.values)),
            feature_values_sha256=features.identity_sha256,
            encoded_stimulation=encoded_values,
            encoded_stimulation_sha256=encoded.identity_sha256,
            direct_stimulation=direct_stimulation_values,
            direct_stimulation_sha256=records[-1].direct_stimulation_sha256,
            neural_step_records=tuple(records),
            state_before_sha256=state_before.state_sha256,
            state_after_sha256=result.state.state_sha256,
            state_before_activity=state_before.activity,
            state_after_activity=result.state.activity,
            neural_step_start=self._neural_step_offset + state_before.step_index,
            neural_step_end=self._neural_step_offset + result.state.step_index,
            motor_frame=motor_frame,
            command_seq=command_seq,
            motors=motors,
            action_sha256=action_sha256,
            interventions=intervention_ids,
        )
        self._last_observation_seq = observation.observation_seq
        self._last_game_tick = observation.game_tick
        self._last_command_seq = command_seq
        return action

    def _validate_observation(self, observation: Observation, command_seq: int) -> None:
        if not isinstance(observation, Observation):
            raise ControllerIntegrityError("controller accepts only protocol Observation")
        if observation.identity != self.identity:
            raise ControllerIntegrityError("observation identity does not match controller")
        if observation.missing_fields:
            raise ControllerIntegrityError("faithful input has declared missing sensor fields")
        if observation.observation_seq <= self._last_observation_seq:
            raise ControllerIntegrityError("observation sequence is stale or duplicated")
        if observation.game_tick <= self._last_game_tick:
            raise ControllerIntegrityError("observation tick is stale or duplicated")
        if isinstance(command_seq, bool) or not isinstance(command_seq, int):
            raise ControllerIntegrityError("command sequence must be an integer")
        if command_seq <= self._last_command_seq:
            raise ControllerIntegrityError("command sequence is stale or duplicated")
        if self.identity.graph_hash != self.graph.manifest.runtime_identity_sha256:
            raise ControllerIntegrityError("bound graph identity changed")

    @staticmethod
    def _sign_overlay(graph: RuntimeGraph, configuration: ControllerConfiguration) -> SignOverlay:
        if configuration.sign_policy == "coarse_presynaptic_consensus":
            return SignOverlay.coarse_presynaptic_consensus(graph)
        return SignOverlay.unknown_zero(graph)

    @staticmethod
    def _sensor_channels(
        configuration: ControllerConfiguration,
    ) -> tuple[SensorChannel, ...]:
        left = configuration.left_sensor_body_id
        right = configuration.right_sensor_body_id
        return (
            SensorChannel(
                "forward_clearance_fraction",
                "fraction",
                (SensorTarget(left, 0.5), SensorTarget(right, 0.5)),
                clip_min=0.0,
                clip_max=1.0,
            ),
            SensorChannel(
                "left_clearance_fraction",
                "fraction",
                (SensorTarget(left, 1.0),),
                clip_min=0.0,
                clip_max=1.0,
            ),
            SensorChannel(
                "right_clearance_fraction",
                "fraction",
                (SensorTarget(right, 1.0),),
                clip_min=0.0,
                clip_max=1.0,
            ),
            SensorChannel(
                "local_light_fraction",
                "fraction",
                (SensorTarget(left, 0.25), SensorTarget(right, 0.25)),
                clip_min=0.0,
                clip_max=1.0,
            ),
        )

    @staticmethod
    def _motor_pools(
        configuration: ControllerConfiguration,
    ) -> tuple[MotorPool, ...]:
        return (
            MotorPool(
                "left_motor",
                (PoolMember(configuration.left_motor_body_id, 1.0),),
                gain=configuration.motor_pool_gain,
                dead_zone=configuration.motor_dead_zone,
            ),
            MotorPool(
                "right_motor",
                (PoolMember(configuration.right_motor_body_id, 1.0),),
                gain=configuration.motor_pool_gain,
                dead_zone=configuration.motor_dead_zone,
            ),
        )

    @staticmethod
    def _project_motor_frame(frame: MotorFrame) -> MotorVector:
        values = dict(frame.values)
        if set(values) != {"left_motor", "right_motor"}:
            raise ControllerIntegrityError("motor frame does not match projection inputs")
        left = values["left_motor"]
        right = values["right_motor"]

        def clip(value: float) -> float:
            return min(1.0, max(-1.0, value))

        common = 0.5 * left + 0.5 * right
        return MotorVector(
            forward_thrust=clip(common),
            lateral_thrust=0.0,
            vertical_thrust=clip(common),
            yaw_torque=clip(-0.5 * left + 0.5 * right),
            pitch_torque=0.0,
        )
