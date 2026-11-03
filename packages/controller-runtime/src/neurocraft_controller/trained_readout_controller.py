"""Opt-in trained-readout controller for the TRM live demo (trm-live-demo-v1).

Trained-Readout Mode, ADR 0008 Mode 2 of 3 — never Faithful Organism Mode. The
frozen core (steering fixture, consensus signs, pinned engine, pinned sensory
path, five steps per tick) is byte-for-byte the faithful controller's; the only
new computation is a ridge readout with intercept, loaded from a committed
artifact whose SHA-256 this configuration binds, mapping the full hidden state
to three declared channels (forward, lateral, yaw). The declared projection
mirrors the faithful common-mode→vertical coupling (`vertical = forward`).

New module by the same rule as `flight_controller.py`: pilot-v3 pins
`config.py`/`controller.py` bytes and the bridge receipts pin the default
identities, so everything here is additive and explicitly opt-in.

Disclosure carried from `EMBODIMENT-02`: trace manifests stamp the pinned
controller's ``run_mode`` (derived from ``configuration.mode`` — the controls
taxonomy: faithful/lesion/shuffled/etc.); under this profile that field is
the controls-taxonomy value only. The configuration schema, decoder identity,
and launch profile mark a trace as Trained-Readout Mode, and they make the v1
identities unreachable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from neurocraft_connectome import RuntimeGraph
from neurocraft_connectome.model import canonical_json_bytes, sha256_json
from neurocraft_neural.adapters import MotorFrame, MotorPool, PoolMember
from neurocraft_neural.model import NeuralState
from neurocraft_protocol import MotorVector, RunIdentity

from .config import ControllerConfiguration, canonical_sha256
from .controller import ConnectomeController, ControllerIntegrityError
from .flight_pools import LEFT_WING_MOTOR_BODY_IDS, RIGHT_WING_MOTOR_BODY_IDS

TRM_DEMO_CONFIG_SCHEMA = "neurocraft.controller.trm_demo_config.v1"
TRM_DEMO_DECODER_SCHEMA = "neurocraft.decoder.identity.trm_demo.v1"
TRM_DEMO_CHANNELS = ("forward", "lateral", "yaw")
READOUT_ARTIFACT_SCHEMA = "neurocraft.trm.cloned_readout.v1"

#: The declared TRM projection: readout channels to bounded primitive axes.
#: ``vertical = forward`` mirrors the faithful projection's common-mode
#: coupling so altitude behavior matches the working demo.
TRM_PROJECTION_MATERIAL: dict[str, Any] = {
    "channels": list(TRM_DEMO_CHANNELS),
    "forward_thrust": "clip(forward)",
    "lateral_thrust": "clip(lateral)",
    "vertical_thrust": "clip(forward)",
    "yaw_torque": "clip(yaw)",
    "pitch_torque": "0.0",
    "clip": "[-1, 1]",
    "schema": "neurocraft.trm.projection.v1",
}


@dataclass(frozen=True, slots=True)
class TrmDemoConfiguration(ControllerConfiguration):
    """The trained-readout demo boundary, under its own schema.

    ``readout_artifact_sha256`` binds the exact committed weights file; the
    inherited single motor ids are inert (the readout replaces the pool
    decoder entirely) and are frozen at the v1 defaults exactly as the flight
    profile freezes them.
    """

    schema: str = TRM_DEMO_CONFIG_SCHEMA
    readout_artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema != TRM_DEMO_CONFIG_SCHEMA:
            raise ValueError("unsupported trm demo configuration schema")
        if (
            not isinstance(self.readout_artifact_sha256, str)
            or len(self.readout_artifact_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.readout_artifact_sha256)
        ):
            raise ValueError("readout_artifact_sha256 must be lowercase SHA-256 hex")
        if self.mode not in {"faithful", "lesion"}:
            raise ValueError("trm demo controls-taxonomy mode must be faithful or lesion")
        if self.sign_policy not in {"coarse_presynaptic_consensus", "unknown_zero"}:
            raise ValueError("unsupported sign policy")
        if self.neural_steps_per_game_tick != 5:
            raise ValueError("the v1 clock contract requires exactly five neural steps per tick")
        for name in ("left_sensor_body_id", "right_sensor_body_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive source body ID")
        if self.left_sensor_body_id == self.right_sensor_body_id:
            raise ValueError("sensor body IDs must be distinct")
        defaults = ControllerConfiguration()
        inert_ok = all(
            not isinstance(value, bool) and isinstance(value, int) and value == expected
            for value, expected in (
                (self.left_motor_body_id, defaults.left_motor_body_id),
                (self.right_motor_body_id, defaults.right_motor_body_id),
            )
        )
        if not inert_ok:
            raise ValueError(
                "the single motor body IDs are inert under the trained-readout "
                "profile; customizing them would imply a boundary this controller "
                "does not read"
            )
        if not math.isfinite(self.motor_pool_gain) or not 0.0 <= self.motor_pool_gain <= 8.0:
            raise ValueError("motor_pool_gain must be finite and in [0, 8]")
        if not math.isfinite(self.motor_dead_zone) or not 0.0 <= self.motor_dead_zone < 1.0:
            raise ValueError("motor_dead_zone must be finite and in [0, 1)")

    def to_identity_material(self) -> dict[str, Any]:
        return {
            "boundary": {
                "left_sensor_body_id": self.left_sensor_body_id,
                "right_sensor_body_id": self.right_sensor_body_id,
            },
            "channels": list(TRM_DEMO_CHANNELS),
            "mode": self.mode,
            "neural_steps_per_game_tick": self.neural_steps_per_game_tick,
            "projection": TRM_PROJECTION_MATERIAL,
            "readout_artifact_sha256": self.readout_artifact_sha256,
            "schema": self.schema,
            "sign_policy": self.sign_policy,
        }


class TrainedReadout:
    """The parsed, validated cloned-readout artifact."""

    def __init__(self, document: dict[str, Any], *, graph: RuntimeGraph) -> None:
        if document.get("schema") != READOUT_ARTIFACT_SCHEMA:
            raise ControllerIntegrityError("unsupported readout artifact schema")
        if document.get("graph_runtime_identity_sha256") != (
            graph.manifest.runtime_identity_sha256
        ):
            raise ControllerIntegrityError("readout artifact is bound to a different graph")
        if tuple(document.get("channels", ())) != TRM_DEMO_CHANNELS:
            raise ControllerIntegrityError("readout artifact channels changed")
        raw = document.get("weights_float_hex")
        if not isinstance(raw, list) or not raw:
            raise ControllerIntegrityError("readout artifact carries no weights")
        expected_width = len(graph.nodes) + 1
        if len(raw) != expected_width:
            raise ControllerIntegrityError(
                f"readout weight rows ({len(raw)}) do not match hidden state + "
                f"intercept ({expected_width})"
            )
        weights: list[tuple[float, ...]] = []
        for row in raw:
            if not isinstance(row, list) or len(row) != len(TRM_DEMO_CHANNELS):
                raise ControllerIntegrityError("readout weight row shape changed")
            values = tuple(float.fromhex(value) for value in row)
            if any(not math.isfinite(value) for value in values):
                raise ControllerIntegrityError("readout weights must be finite")
            weights.append(values)
        recomputed = sha256_json([[float(v).hex() for v in row] for row in weights])
        if recomputed != document.get("weights_identity_sha256"):
            raise ControllerIntegrityError("readout weights identity mismatch")
        self.weights: tuple[tuple[float, ...], ...] = tuple(weights)
        self.weights_identity_sha256: str = recomputed
        self.demo_id: str = str(document.get("demo_id", ""))


class TrainedReadoutDecoder:
    """Full-hidden-state linear readout presented through the MotorFrame shape."""

    def __init__(
        self, graph: RuntimeGraph, readout: TrainedReadout, *, artifact_sha256: str
    ) -> None:
        graph.assert_integrity()
        self._graph_identity = graph.manifest.runtime_identity_sha256
        self._node_count = len(graph.nodes)
        self._readout = readout
        self._identity = sha256_json(
            {
                "artifact_sha256": artifact_sha256,
                "channels": list(TRM_DEMO_CHANNELS),
                "graph_identity_sha256": self._graph_identity,
                "intercept": True,
                "readout_form": "ridge_full_hidden_state_clip_-1_1",
                "schema": TRM_DEMO_DECODER_SCHEMA,
                "weights_identity_sha256": readout.weights_identity_sha256,
            }
        )

    @property
    def identity_sha256(self) -> str:
        return self._identity

    def decode(self, state: NeuralState) -> MotorFrame:
        if not isinstance(state, NeuralState):
            raise ControllerIntegrityError("trained readout accepts only NeuralState")
        state.assert_integrity()
        if state.graph_identity_sha256 != self._graph_identity:
            raise ControllerIntegrityError("trained readout state is from a different graph")
        if len(state.activity) != self._node_count:
            raise ControllerIntegrityError("trained readout state length mismatch")
        augmented = (*state.activity, 1.0)
        values: list[tuple[str, float]] = []
        for out, name in enumerate(TRM_DEMO_CHANNELS):
            raw = math.fsum(
                augmented[feature] * self._readout.weights[feature][out]
                for feature in range(len(augmented))
            )
            if not math.isfinite(raw):
                raise ControllerIntegrityError("trained readout produced a non-finite value")
            values.append((name, min(1.0, max(-1.0, raw))))
        value_tuple = tuple(values)
        identity = sha256_json(
            {
                "decoder_sha256": self._identity,
                "graph_identity_sha256": self._graph_identity,
                "neural_state_sha256": state.state_sha256,
                "values": [[name, float(value).hex()] for name, value in value_tuple],
            }
        )
        return MotorFrame(
            graph_identity_sha256=self._graph_identity,
            decoder_sha256=self._identity,
            neural_state_sha256=state.state_sha256,
            values=value_tuple,
            identity_sha256=identity,
        )


class TrainedReadoutController(ConnectomeController):
    """The faithful core with the cloned readout as its only new computation."""

    def __init__(
        self,
        graph: RuntimeGraph,
        *,
        run_id: str,
        entity_id: str,
        artifact_bytes: bytes,
        artifact_document: dict[str, Any],
        session_epoch: int = 0,
        configuration: ControllerConfiguration | None = None,
    ) -> None:
        if not isinstance(configuration, TrmDemoConfiguration):
            raise ControllerIntegrityError(
                "the trained-readout controller requires a TrmDemoConfiguration"
            )
        digest = sha256(artifact_bytes).hexdigest()
        if digest != configuration.readout_artifact_sha256:
            raise ControllerIntegrityError(
                "readout artifact bytes do not match the configured SHA-256"
            )
        # The document must BE the parsed bytes: a self-consistent forged
        # document alongside honest bytes would otherwise execute arbitrary
        # weights under the committed artifact's hash (adversarial-review
        # finding, demonstrated).
        if canonical_json_bytes(artifact_document) + b"\n" != artifact_bytes:
            raise ControllerIntegrityError(
                "readout artifact document does not match the artifact bytes"
            )
        readout = TrainedReadout(artifact_document, graph=graph)
        super().__init__(
            graph,
            run_id=run_id,
            entity_id=entity_id,
            session_epoch=session_epoch,
            configuration=configuration,
        )
        # Replace the construction-time pool decoder with the trained readout
        # and rebind every identity that described it, so the trace manifest
        # carries the trained readout's exact provenance.
        self.decoder = TrainedReadoutDecoder(  # type: ignore[assignment]
            graph, readout, artifact_sha256=digest
        )
        self.projection_identity_sha256 = canonical_sha256(TRM_PROJECTION_MATERIAL)
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

    # The parent's pool decoder must be constructible during super().__init__;
    # the default single motor ids are v1 bodies absent from the steering
    # fixture, so a valid throwaway pair from the wing constants stands in
    # until the trained decoder replaces it (the same staticmethod-to-instance
    # override necessity flight_controller documents).
    def _motor_pools(  # type: ignore[override]
        self,
        configuration: ControllerConfiguration,
    ) -> tuple[MotorPool, ...]:
        return (
            MotorPool(
                "left_motor",
                (PoolMember(LEFT_WING_MOTOR_BODY_IDS[0], 1.0),),
                gain=configuration.motor_pool_gain,
                dead_zone=configuration.motor_dead_zone,
            ),
            MotorPool(
                "right_motor",
                (PoolMember(RIGHT_WING_MOTOR_BODY_IDS[0], 1.0),),
                gain=configuration.motor_pool_gain,
                dead_zone=configuration.motor_dead_zone,
            ),
        )

    def _project_motor_frame(  # type: ignore[override]
        self,
        frame: MotorFrame,
    ) -> MotorVector:
        values = dict(frame.values)
        if set(values) != set(TRM_DEMO_CHANNELS):
            raise ControllerIntegrityError("motor frame does not match the TRM projection")

        def clip(value: float) -> float:
            return min(1.0, max(-1.0, value))

        forward = clip(values["forward"])
        return MotorVector(
            forward_thrust=forward,
            lateral_thrust=clip(values["lateral"]),
            vertical_thrust=forward,
            yaw_torque=clip(values["yaw"]),
            pitch_torque=0.0,
        )
