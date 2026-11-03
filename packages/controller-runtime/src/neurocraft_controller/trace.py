"""Translate verified controller evidence into the canonical causal trace schema."""

from __future__ import annotations

import hashlib
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from neurocraft_protocol import (
    ACTION_TARGET_OFFSET_TICKS,
    Action,
    ActuationReceipt,
    ActuationValidation,
    CausalRecord,
    ClaimScope,
    ControlCommand,
    ControlOperation,
    FaultCode,
    Observation,
    TraceWriter,
    ValidationResult,
    WorldFrame,
    WorldState,
    canonical_json_bytes,
)

from .config import canonical_sha256
from .controller import ConnectomeController, ControllerIntegrityError, ControllerStepEvidence


@dataclass(frozen=True, slots=True)
class ReceiptBinding:
    """Exact issued-action/receipt/trace binding retained by the live recorder."""

    command_seq: int
    observation_seq: int
    target_tick: int
    applied_tick: int
    action_sha256: str
    receipt_sha256: str
    trace_entry_sha256: str


class TraceRecordingState(StrEnum):
    IDLE = "IDLE"
    RECORDING = "RECORDING"
    STOPPING = "STOPPING"
    CLOSED = "CLOSED"
    ABORTED = "ABORTED"


class TraceControlError(ControllerIntegrityError):
    def __init__(self, code: FaultCode, message: str) -> None:
        if code not in {FaultCode.CONTROL_UNAVAILABLE, FaultCode.INVALID_STATE}:
            raise ValueError("trace control rejection must use a control fault code")
        super().__init__(message)
        self.code = code.value


@dataclass(frozen=True, slots=True)
class DeclaredTransitionForces:
    """Explicit arena-bound forces; never inferred or silently defaulted."""

    passive_forces: tuple[float, float, float]
    external_forces: tuple[float, float, float]
    provenance: str
    arena_material_hash: str
    identity_sha256: str

    @classmethod
    def create(
        cls,
        *,
        passive_forces: tuple[float, float, float],
        external_forces: tuple[float, float, float],
        provenance: str,
        arena_material_hash: str,
    ) -> DeclaredTransitionForces:
        if not isinstance(provenance, str) or not provenance:
            raise ValueError("transition forces require named provenance")
        if len(arena_material_hash) != 64 or any(
            character not in "0123456789abcdef" for character in arena_material_hash
        ):
            raise ValueError("arena material hash must be lowercase SHA-256")
        resolved_forces: list[tuple[float, float, float]] = []
        for name, values in (
            ("passive_forces", passive_forces),
            ("external_forces", external_forces),
        ):
            if not isinstance(values, tuple) or len(values) != 3:
                raise ValueError(f"{name} must be a three-value tuple")
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float)) for value in values
            ):
                raise ValueError(f"{name} must contain finite numbers")
            resolved = (float(values[0]), float(values[1]), float(values[2]))
            if any(not math.isfinite(number) for number in resolved):
                raise ValueError(f"{name} must contain finite numbers")
            resolved_forces.append(resolved)
        resolved_passive, resolved_external = resolved_forces
        material = {
            "arena_material_hash": arena_material_hash,
            "external_forces": [value.hex() for value in resolved_external],
            "passive_forces": [value.hex() for value in resolved_passive],
            "provenance": provenance,
            "schema": "neurocraft.declared_transition_forces.v1",
        }
        return cls(
            passive_forces=resolved_passive,
            external_forces=resolved_external,
            provenance=provenance,
            arena_material_hash=arena_material_hash,
            identity_sha256=canonical_sha256(material),
        )


@dataclass(frozen=True, slots=True)
class PendingTraceState:
    """Non-secret state explaining why a receipt-bound trace cannot finalize yet."""

    issued_without_receipt: tuple[int, ...]
    receipts_waiting_for_world_after: tuple[int, ...]
    actions_without_controller_evidence: tuple[int, ...]
    actions_without_operational_telemetry: tuple[int, ...]

    @property
    def is_clear(self) -> bool:
        return not (
            self.issued_without_receipt
            or self.receipts_waiting_for_world_after
            or self.actions_without_controller_evidence
            or self.actions_without_operational_telemetry
        )


@dataclass(frozen=True, slots=True)
class _OperationalTelemetry:
    queue_coalesced: int
    queue_overruns: int
    compute_ns: int
    serialization_ns: int
    write_ns: int


@dataclass(slots=True)
class _PendingAction:
    action: Action
    action_sha256: str
    evidence: ControllerStepEvidence | None
    receipt: ActuationReceipt | None = None
    telemetry: _OperationalTelemetry | None = None


def world_state_from_observation(observation: Observation) -> WorldState:
    sensors = observation.sensors
    return WorldState(
        position_m=sensors.position_m,
        velocity_m_s=sensors.velocity_m_s,
        yaw_rad=sensors.yaw_rad,
        pitch_rad=sensors.pitch_rad,
        on_ground=sensors.on_ground,
        horizontal_collision=sensors.horizontal_collision,
        vertical_collision=sensors.vertical_collision,
        in_fluid=sensors.in_fluid,
    )


def world_state_from_frame(frame: WorldFrame) -> WorldState:
    """Convert an authenticated pre-actuation protocol frame to trace state."""
    return WorldState(
        position_m=frame.position_m,
        velocity_m_s=frame.velocity_m_s,
        yaw_rad=frame.yaw_rad,
        pitch_rad=frame.pitch_rad,
        on_ground=frame.on_ground,
        horizontal_collision=frame.horizontal_collision,
        vertical_collision=frame.vertical_collision,
        in_fluid=frame.in_fluid,
    )


def causal_record_from_applied_action(
    controller: ConnectomeController,
    *,
    record_seq: int,
    observation: Observation,
    world_after: WorldState,
    world_after_observation_seq: int,
    world_after_game_tick: int,
    observation_age_ticks: int = 0,
    clamp_fields: tuple[str, ...] = (),
    passive_forces: tuple[float, float, float] = (0.0, 0.0, 0.0),
    external_forces: tuple[float, float, float] = (0.0, 0.0, 0.0),
    queue_coalesced: int = 0,
    queue_overruns: int = 0,
    compute_ns: int = 0,
    serialization_ns: int = 0,
    write_ns: int = 0,
) -> CausalRecord:
    """Create an APPLIED record only after the caller has observed application.

    This function cannot itself prove that Minecraft applied the command. A caller
    must invoke it only from an authenticated application receipt or a clearly
    labelled deterministic harness that supplies ``world_after``.
    """

    evidence: ControllerStepEvidence | None = controller.last_evidence
    if evidence is None:
        raise ControllerIntegrityError("no controller step evidence is available")
    if evidence.observation_sha256 == "":
        raise ControllerIntegrityError("controller evidence lacks observation identity")
    action_sequence = evidence.command_seq
    if action_sequence < 0:
        raise ControllerIntegrityError("controller has no generated command")
    state_records = evidence.neural_step_records
    if not state_records:
        raise ControllerIntegrityError("controller evidence has no neural steps")

    # CausalRecord validates that its observation belongs to the controller identity.
    if observation.identity != controller.identity:
        raise ControllerIntegrityError("trace observation/controller identity mismatch")
    if canonical_sha256(observation.to_dict()) != evidence.observation_sha256:
        raise ControllerIntegrityError("trace observation does not match executed controller input")
    issued_action = Action(
        identity=controller.identity,
        command_seq=action_sequence,
        observation_seq=observation.observation_seq,
        observation_tick=observation.game_tick,
        target_tick=observation.game_tick + ACTION_TARGET_OFFSET_TICKS,
        valid_through_tick=observation.game_tick + ACTION_TARGET_OFFSET_TICKS,
        neural_step_start=evidence.neural_step_start,
        neural_step_end=evidence.neural_step_end,
        state_digest=evidence.state_after_sha256,
        motors=evidence.motors,
        neutral=False,
    )
    if canonical_sha256(issued_action.to_dict()) != evidence.action_sha256:
        raise ControllerIntegrityError("reconstructed issued action does not match evidence")
    return CausalRecord(
        identity=controller.identity,
        record_seq=record_seq,
        game_tick=observation.game_tick,
        observation_age_ticks=observation_age_ticks,
        observation=observation,
        encoded_stimulation=dict(evidence.encoded_stimulation),
        direct_stimulation=dict(evidence.direct_stimulation),
        direct_stimulation_sha256=evidence.direct_stimulation_sha256,
        neural_step_start=evidence.neural_step_start,
        neural_step_end=evidence.neural_step_end,
        state_before_digest=evidence.state_before_sha256,
        state_after_digest=evidence.state_after_sha256,
        state_before_activity=evidence.state_before_activity,
        state_after_activity=evidence.state_after_activity,
        interventions=evidence.interventions,
        delay_state_digest=state_records[-1].delay_after_sha256,
        checkpoint_reference=None,
        motor_pool_values=dict(evidence.motor_frame.values),
        pool_membership_hash=controller.decoder.identity_sha256,
        decoder_output=evidence.motors,
        issued_action=issued_action,
        actuation_receipt=None,
        command_seq=action_sequence,
        validation=ValidationResult.APPLIED,
        clamp_fields=clamp_fields,
        applied_action=evidence.motors,
        neutral_reason=None,
        world_before=world_state_from_observation(observation),
        world_after=world_after,
        world_after_observation_seq=world_after_observation_seq,
        world_after_game_tick=world_after_game_tick,
        passive_forces=passive_forces,
        external_forces=external_forces,
        queue_coalesced=queue_coalesced,
        queue_overruns=queue_overruns,
        compute_ns=compute_ns,
        serialization_ns=serialization_ns,
        write_ns=write_ns,
        fault_code=None,
    )


def causal_record_from_receipt(
    controller: ConnectomeController,
    *,
    record_seq: int,
    action: Action,
    evidence: ControllerStepEvidence,
    receipt: ActuationReceipt,
    world_after_observation: Observation,
    passive_forces: tuple[float, float, float] = (0.0, 0.0, 0.0),
    external_forces: tuple[float, float, float] = (0.0, 0.0, 0.0),
    queue_coalesced: int = 0,
    queue_overruns: int = 0,
    compute_ns: int = 0,
    serialization_ns: int = 0,
    write_ns: int = 0,
) -> CausalRecord:
    """Build a causal record from an exact receipt and a later world observation.

    The receipt proves actuation; the later observation supplies the authoritative
    post-actuation world state. Neither input is inferred from the requested motors.
    """

    if action.identity != controller.identity or receipt.identity != controller.identity:
        raise ControllerIntegrityError("receipt/action/controller identity mismatch")
    if evidence.observation.identity != controller.identity:
        raise ControllerIntegrityError("controller evidence observation identity mismatch")
    action_sha256 = canonical_sha256(action.to_dict())
    if evidence.action_sha256 != action_sha256:
        raise ControllerIntegrityError("issued action hash does not match controller evidence")
    if canonical_sha256(evidence.observation.to_dict()) != evidence.observation_sha256:
        raise ControllerIntegrityError("controller observation hash does not match evidence")
    if (
        receipt.command_seq != action.command_seq
        or receipt.observation_seq != action.observation_seq
        or receipt.target_tick != action.target_tick
        or receipt.requested_motors != action.motors
    ):
        raise ControllerIntegrityError("receipt does not bind the exact issued action")
    if (
        receipt.validation in {ActuationValidation.APPLIED, ActuationValidation.CLAMPED}
        and receipt.applied_tick > action.valid_through_tick
    ):
        raise ControllerIntegrityError("receipt reports application after action expiry")
    if world_after_observation.identity != controller.identity:
        raise ControllerIntegrityError("world-after observation identity mismatch")
    if world_after_observation.game_tick != receipt.applied_tick + 1:
        raise ControllerIntegrityError(
            "world-after observation must be the exact next tick after receipt application"
        )

    if receipt.validation in {ActuationValidation.APPLIED, ActuationValidation.CLAMPED}:
        validation = ValidationResult.APPLIED
        neutral_reason = None
    elif receipt.validation is ActuationValidation.NEUTRAL_FAULT:
        validation = ValidationResult.NEUTRAL_FAULT
        neutral_reason = "minecraft_neutral_fault"
    else:
        validation = ValidationResult.REJECTED
        neutral_reason = "minecraft_rejected"

    state_records = evidence.neural_step_records
    if not state_records:
        raise ControllerIntegrityError("controller evidence has no neural steps")
    return CausalRecord(
        identity=controller.identity,
        record_seq=record_seq,
        game_tick=evidence.observation.game_tick,
        observation_age_ticks=receipt.applied_tick - evidence.observation.game_tick,
        observation=evidence.observation,
        encoded_stimulation=dict(evidence.encoded_stimulation),
        direct_stimulation=dict(evidence.direct_stimulation),
        direct_stimulation_sha256=evidence.direct_stimulation_sha256,
        neural_step_start=evidence.neural_step_start,
        neural_step_end=evidence.neural_step_end,
        state_before_digest=evidence.state_before_sha256,
        state_after_digest=evidence.state_after_sha256,
        state_before_activity=evidence.state_before_activity,
        state_after_activity=evidence.state_after_activity,
        interventions=evidence.interventions,
        delay_state_digest=state_records[-1].delay_after_sha256,
        checkpoint_reference=None,
        motor_pool_values=dict(evidence.motor_frame.values),
        pool_membership_hash=controller.decoder.identity_sha256,
        decoder_output=action.motors,
        issued_action=action,
        actuation_receipt=receipt,
        command_seq=action.command_seq,
        validation=validation,
        clamp_fields=receipt.clamp_fields,
        applied_action=receipt.applied_motors,
        neutral_reason=neutral_reason,
        world_before=world_state_from_frame(receipt.world_before),
        world_after=world_state_from_observation(world_after_observation),
        world_after_observation_seq=world_after_observation.observation_seq,
        world_after_game_tick=world_after_observation.game_tick,
        passive_forces=passive_forces,
        external_forces=external_forces,
        queue_coalesced=queue_coalesced,
        queue_overruns=queue_overruns,
        compute_ns=compute_ns,
        serialization_ns=serialization_ns,
        write_ns=write_ns,
        fault_code=receipt.fault_code,
    )


class MinecraftReceiptTraceRecorder:
    """Gate Minecraft-labelled trace writes on exact authenticated receipts.

    ``accept_issued_action`` is the service's issued-action callback and
    ``accept_receipt`` is its authenticated receipt callback. A record is written
    only after both callbacks agree and a later controller input supplies an
    authoritative world-after observation. ``close`` refuses incomplete traces.
    """

    def __init__(
        self,
        controller: ConnectomeController,
        writer: TraceWriter,
        *,
        forces: DeclaredTransitionForces,
        max_pending: int = ACTION_TARGET_OFFSET_TICKS + 1,
    ) -> None:
        if max_pending < 1 or max_pending > ACTION_TARGET_OFFSET_TICKS + 1:
            # Issued actions awaiting their record. A record is committed once an observation
            # later than the action's applied tick arrives, and an action applies
            # ACTION_TARGET_OFFSET_TICKS after the observation it answers, so at the moment
            # the action for observation T is cached, the actions for T - 1 and T - 2 are
            # still in flight: ACTION_TARGET_OFFSET_TICKS + 1 in total. It was two under the
            # one-tick budget by the same derivation.
            raise ValueError("max_pending must be between one and the action budget plus one")
        manifest = writer.manifest
        labels = (manifest.java, manifest.minecraft, manifest.fabric, manifest.mod)
        if any(not label.strip() or "not_used" in label.lower() for label in labels):
            raise ControllerIntegrityError(
                "Minecraft receipt recorder requires explicit Java/Minecraft/Fabric/mod labels"
            )
        if manifest.run_id != controller.identity.run_id:
            raise ControllerIntegrityError("trace manifest run ID does not match controller")
        if manifest.initial_session_epoch != controller.identity.session_epoch:
            raise ControllerIntegrityError("trace manifest session epoch does not match controller")
        artifacts = manifest.artifacts
        identity = controller.identity
        if (
            artifacts.runtime_graph_hash != identity.graph_hash
            or artifacts.adapter_hash != identity.adapter_hash
            or artifacts.dynamics_hash != identity.dynamics_hash
            or artifacts.decoder_hash != identity.decoder_hash
            or artifacts.observation_schema_hash != identity.observation_schema_hash
            or artifacts.sign_hash != controller.sign_overlay.identity_sha256
            or artifacts.configuration_hash != controller.configuration.identity_sha256
        ):
            raise ControllerIntegrityError(
                "trace manifest artifact identities do not match controller"
            )
        if not isinstance(forces, DeclaredTransitionForces):
            raise ControllerIntegrityError("Minecraft receipt recorder requires declared forces")
        if manifest.artifacts.arena_hash != forces.identity_sha256:
            raise ControllerIntegrityError(
                "trace manifest arena hash does not bind the declared force model"
            )
        self.controller = controller
        self.writer = writer
        self.forces = forces
        self.max_pending = max_pending
        self._requires_operational_telemetry = manifest.claim_scope in {
            ClaimScope.G3,
            ClaimScope.G4,
            ClaimScope.G5,
            ClaimScope.G6,
        }
        self._pending: OrderedDict[int, _PendingAction] = OrderedDict()
        self._bindings: list[ReceiptBinding] = []
        self._record_count = 0
        self._issued_high_water: int | None = None
        self._last_target_tick: int | None = None
        self._recording_floor: int | None = None
        self._recording_ceiling: int | None = None
        self._stop_tick: int | None = None
        self._boundary_crossed = False
        self._latest_world_observation: Observation | None = None
        self._detached_after_stop: OrderedDict[int, _PendingAction] = OrderedDict()
        self._recording_id: str | None = None
        self._recording_hash: str | None = None
        self._state = TraceRecordingState.IDLE
        self._closed = False
        self._lock = threading.RLock()

    @property
    def bindings(self) -> tuple[ReceiptBinding, ...]:
        with self._lock:
            return tuple(self._bindings)

    @property
    def record_count(self) -> int:
        with self._lock:
            return self._record_count

    @property
    def state(self) -> TraceRecordingState:
        with self._lock:
            return self._state

    @property
    def pending_state(self) -> PendingTraceState:
        with self._lock:
            missing = tuple(
                sequence for sequence, pending in self._pending.items() if pending.receipt is None
            )
            waiting = tuple(
                sequence
                for sequence, pending in self._pending.items()
                if pending.receipt is not None and pending.evidence is not None
            )
            untraceable = tuple(
                sequence for sequence, pending in self._pending.items() if pending.evidence is None
            )
            missing_telemetry = tuple(
                sequence for sequence, pending in self._pending.items() if pending.telemetry is None
            )
            return PendingTraceState(missing, waiting, untraceable, missing_telemetry)

    def accept_issued_action(self, action: Action) -> None:
        """Cache one service-validated Action before it is serialized to the peer."""

        with self._lock:
            if self._state is TraceRecordingState.CLOSED:
                return
            self._ensure_open()
            if action.identity != self.controller.identity:
                raise ControllerIntegrityError("issued action identity does not match recorder")
            expected_sequence = (
                0 if self._issued_high_water is None else self._issued_high_water + 1
            )
            if action.command_seq != expected_sequence:
                raise ControllerIntegrityError("issued action sequence is not contiguous")
            if self._last_target_tick is not None and action.target_tick <= self._last_target_tick:
                raise ControllerIntegrityError("issued action target ticks are not increasing")
            self._issued_high_water = action.command_seq
            self._last_target_tick = action.target_tick

            # Actions issued while idle remain outside the recorded interval, but
            # their high-water mark makes a later in-flight receipt distinguishable
            # from a missing post-start issued-action callback.
            if self._state is TraceRecordingState.IDLE:
                return

            evidence = self.controller.last_evidence
            if evidence is not None and evidence.command_seq == action.command_seq:
                if evidence.action_sha256 != canonical_sha256(action.to_dict()):
                    raise ControllerIntegrityError("issued action hash does not match evidence")
                self._latest_world_observation = evidence.observation
                self.flush(evidence.observation)
                bound_evidence: ControllerStepEvidence | None = evidence
            else:
                # The service can originate fail-closed neutral actions while paused
                # or faulted. They are still paired exactly with receipts, but cannot
                # become a neural causal record without the raw Observation/evidence.
                bound_evidence = None

            pending = _PendingAction(
                action=action,
                action_sha256=canonical_sha256(action.to_dict()),
                evidence=bound_evidence,
                telemetry=(
                    None
                    if self._requires_operational_telemetry
                    else _OperationalTelemetry(0, 0, 0, 0, 0)
                ),
            )
            if self._state is TraceRecordingState.STOPPING:
                assert self._stop_tick is not None
                if action.target_tick > self._stop_tick:
                    self._boundary_crossed = True
                    self._detach_after_stop(pending)
                    self._try_finalize()
                    return
                if self._boundary_crossed:
                    raise ControllerIntegrityError(
                        "issued action moved back inside the crossed record-stop boundary"
                    )

            if len(self._pending) >= self.max_pending:
                raise ControllerIntegrityError("receipt trace pending window is full")
            self._pending[action.command_seq] = pending
            self._recording_ceiling = action.command_seq

    def accept_action_telemetry(
        self,
        *,
        command_seq: int,
        action_sha256: str,
        queue_coalesced: int,
        queue_overruns: int,
        compute_ns: int,
        serialization_ns: int,
        write_ns: int,
    ) -> None:
        """Bind measured service telemetry to one exact issued Action."""

        values = {
            "command_seq": command_seq,
            "queue_coalesced": queue_coalesced,
            "queue_overruns": queue_overruns,
            "compute_ns": compute_ns,
            "serialization_ns": serialization_ns,
            "write_ns": write_ns,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ControllerIntegrityError(
                    f"operational telemetry {name} must be a non-negative integer"
                )
        for name in ("compute_ns", "serialization_ns", "write_ns"):
            if values[name] == 0:
                raise ControllerIntegrityError(
                    f"operational telemetry {name} must be measured and positive"
                )
        if len(action_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in action_sha256
        ):
            raise ControllerIntegrityError(
                "operational telemetry action hash must be lowercase SHA-256"
            )

        with self._lock:
            if self._state in {TraceRecordingState.IDLE, TraceRecordingState.CLOSED}:
                return
            self._ensure_open()
            if self._recording_floor is None:
                raise ControllerIntegrityError(
                    "operational telemetry recording floor is unavailable"
                )
            if command_seq < self._recording_floor:
                return
            detached = self._detached_after_stop.get(command_seq)
            if detached is not None and not self._requires_operational_telemetry:
                return
            pending = self._pending.get(command_seq) if detached is None else detached
            if pending is None:
                raise ControllerIntegrityError("operational telemetry has no pending issued action")
            if pending.action_sha256 != action_sha256:
                raise ControllerIntegrityError(
                    "operational telemetry does not bind the issued action"
                )
            if pending.telemetry is not None:
                raise ControllerIntegrityError("operational telemetry was already recorded")
            pending.telemetry = _OperationalTelemetry(
                queue_coalesced=queue_coalesced,
                queue_overruns=queue_overruns,
                compute_ns=compute_ns,
                serialization_ns=serialization_ns,
                write_ns=write_ns,
            )
            if detached is not None:
                if detached.receipt is not None:
                    del self._detached_after_stop[command_seq]
                return
            if self._latest_world_observation is not None:
                self.flush(self._latest_world_observation)
            self._try_finalize()

    def accept_receipt(self, receipt: ActuationReceipt) -> None:
        """Bind an authenticated receipt to the oldest issued action, without writing yet."""

        with self._lock:
            if self._state in {TraceRecordingState.IDLE, TraceRecordingState.CLOSED}:
                return
            self._ensure_open()
            if receipt.identity != self.controller.identity:
                raise ControllerIntegrityError("receipt identity does not match recorder")
            if self._recording_floor is None:
                raise ControllerIntegrityError("receipt trace recording floor is unavailable")
            if receipt.command_seq < self._recording_floor:
                # The service's SessionValidator has already authenticated and
                # exactly bound this receipt. Its sequence proves that it belongs
                # to an action issued before RECORD_START acquired this lock.
                return

            detached = self._detached_after_stop.get(receipt.command_seq)
            if detached is not None:
                self._validate_receipt_binding(receipt, detached.action)
                if detached.receipt is not None:
                    raise ControllerIntegrityError("receipt was already recorded")
                detached.receipt = receipt
                if detached.telemetry is not None:
                    del self._detached_after_stop[receipt.command_seq]
                return

            unreceipted = next(
                (
                    (sequence, pending)
                    for sequence, pending in self._pending.items()
                    if pending.receipt is None
                ),
                None,
            )
            if unreceipted is None:
                raise ControllerIntegrityError("receipt has no pending issued action")
            sequence, pending = unreceipted
            action = pending.action
            try:
                self._validate_receipt_binding(receipt, action)
            except ControllerIntegrityError as exc:
                raise ControllerIntegrityError(
                    "receipt does not bind oldest unreceipted issued action"
                ) from exc
            if receipt.command_seq != sequence:
                raise ControllerIntegrityError(
                    "receipt does not bind oldest unreceipted issued action"
                )
            pending.receipt = receipt
            if self._latest_world_observation is not None:
                self.flush(self._latest_world_observation)
            self._try_finalize()

    def flush(self, world_after_observation: Observation) -> tuple[ReceiptBinding, ...]:
        """Commit all oldest receipt-bound transitions observed by this later frame."""

        with self._lock:
            self._ensure_open()
            if world_after_observation.identity != self.controller.identity:
                raise ControllerIntegrityError("flush observation identity does not match recorder")
            committed: list[ReceiptBinding] = []
            while self._pending:
                sequence, pending = next(iter(self._pending.items()))
                receipt = pending.receipt
                if receipt is None or receipt.applied_tick >= world_after_observation.game_tick:
                    break
                if pending.evidence is None:
                    break
                telemetry = pending.telemetry
                if telemetry is None:
                    break
                record = causal_record_from_receipt(
                    self.controller,
                    record_seq=self._record_count,
                    action=pending.action,
                    evidence=pending.evidence,
                    receipt=receipt,
                    world_after_observation=world_after_observation,
                    passive_forces=self.forces.passive_forces,
                    external_forces=self.forces.external_forces,
                    queue_coalesced=telemetry.queue_coalesced,
                    queue_overruns=telemetry.queue_overruns,
                    compute_ns=telemetry.compute_ns,
                    serialization_ns=telemetry.serialization_ns,
                    write_ns=telemetry.write_ns,
                )
                trace_hash = self.writer.append(record)
                binding = ReceiptBinding(
                    command_seq=sequence,
                    observation_seq=receipt.observation_seq,
                    target_tick=receipt.target_tick,
                    applied_tick=receipt.applied_tick,
                    action_sha256=pending.action_sha256,
                    receipt_sha256=hashlib.sha256(
                        canonical_json_bytes(receipt.to_dict())
                    ).hexdigest(),
                    trace_entry_sha256=trace_hash,
                )
                self._bindings.append(binding)
                self._record_count += 1
                del self._pending[sequence]
                committed.append(binding)
            return tuple(committed)

    def handle_control(self, command: ControlCommand) -> None:
        """Implement honest RECORD_START/STOP; replay is deliberately unavailable."""

        with self._lock:
            self._ensure_open()
            if command.operation is ControlOperation.RECORD_START:
                if self._state is not TraceRecordingState.IDLE:
                    raise TraceControlError(FaultCode.INVALID_STATE, "receipt trace is not idle")
                assert command.content_id is not None
                self._recording_id = command.content_id
                self._recording_hash = command.content_hash
                self._recording_floor = (
                    0 if self._issued_high_water is None else self._issued_high_water + 1
                )
                self._recording_ceiling = self._recording_floor - 1
                self._state = TraceRecordingState.RECORDING
                return
            if command.operation is ControlOperation.RECORD_STOP:
                if self._state is not TraceRecordingState.RECORDING:
                    raise TraceControlError(
                        FaultCode.INVALID_STATE, "receipt trace is not recording"
                    )
                if command.content_id != self._recording_id or (
                    command.content_hash is not None
                    and self._recording_hash is not None
                    and command.content_hash != self._recording_hash
                ):
                    raise TraceControlError(
                        FaultCode.CONTROL_UNAVAILABLE,
                        "record stop content identity does not match active recording",
                    )
                if any(binding.target_tick > command.issued_tick for binding in self._bindings):
                    raise TraceControlError(
                        FaultCode.INVALID_STATE,
                        "record stop tick predates an already committed action",
                    )
                self._stop_tick = command.issued_tick
                self._state = TraceRecordingState.STOPPING
                for sequence, pending in tuple(self._pending.items()):
                    if pending.action.target_tick <= command.issued_tick:
                        continue
                    self._detach_after_stop(pending)
                    del self._pending[sequence]
                    self._boundary_crossed = True
                if self._pending:
                    self._recording_ceiling = next(reversed(self._pending))
                elif self._bindings:
                    self._recording_ceiling = self._bindings[-1].command_seq
                else:
                    assert self._recording_floor is not None
                    self._recording_ceiling = self._recording_floor - 1
                if self._latest_world_observation is not None:
                    self.flush(self._latest_world_observation)
                self._try_finalize()
                return
            raise TraceControlError(
                FaultCode.CONTROL_UNAVAILABLE,
                "receipt trace handler supports recording, not replay",
            )

    def close(self) -> Path:
        """Publish only a complete trace; pending terminal transitions fail closed."""

        with self._lock:
            if self._state is TraceRecordingState.CLOSED:
                return self.writer.path
            self._ensure_open()
            if self._state is not TraceRecordingState.STOPPING:
                raise ControllerIntegrityError("trace can close only after RECORD_STOP")
            self._try_finalize()
            if self._closed:
                return self.writer.path
            if not self._boundary_crossed:
                raise ControllerIntegrityError(
                    "cannot finalize trace before the record-stop boundary is crossed"
                )
            state = self.pending_state
            if not state.is_clear:
                raise ControllerIntegrityError(
                    "cannot finalize trace with missing receipt or world-after evidence"
                )
            raise ControllerIntegrityError("record-stop boundary could not finalize the trace")

    def abort(self) -> None:
        with self._lock:
            if self._closed:
                return
            self.writer.abort()
            self._closed = True
            self._state = TraceRecordingState.ABORTED

    def _ensure_open(self) -> None:
        if self._closed:
            raise ControllerIntegrityError("receipt trace recorder is closed")

    def _detach_after_stop(self, pending: _PendingAction) -> None:
        # The peer can acknowledge a socket write before its asynchronous
        # telemetry callback runs. Keep both bindings until both have arrived,
        # including callbacks accepted while this action was still in-bound.
        if pending.receipt is not None and pending.telemetry is not None:
            return
        if len(self._detached_after_stop) >= self.max_pending:
            raise ControllerIntegrityError("receipt trace detached window is full")
        self._detached_after_stop[pending.action.command_seq] = pending

    @staticmethod
    def _validate_receipt_binding(receipt: ActuationReceipt, action: Action) -> None:
        if (
            receipt.command_seq != action.command_seq
            or receipt.observation_seq != action.observation_seq
            or receipt.target_tick != action.target_tick
            or receipt.requested_motors != action.motors
        ):
            raise ControllerIntegrityError("receipt does not bind issued action")
        if (
            receipt.validation in {ActuationValidation.APPLIED, ActuationValidation.CLAMPED}
            and receipt.applied_tick > action.valid_through_tick
        ):
            raise ControllerIntegrityError("receipt reports application after action expiry")

    def _try_finalize(self) -> None:
        if self._state is not TraceRecordingState.STOPPING or not self._boundary_crossed:
            return
        if self._latest_world_observation is not None:
            self.flush(self._latest_world_observation)
        if not self.pending_state.is_clear:
            return
        assert self._recording_floor is not None
        assert self._recording_ceiling is not None
        expected_sequences = tuple(range(self._recording_floor, self._recording_ceiling + 1))
        actual_sequences = tuple(binding.command_seq for binding in self._bindings)
        if actual_sequences != expected_sequences:
            raise ControllerIntegrityError(
                "recorded command sequences do not cover the atomic record-stop interval"
            )
        self.writer.close()
        self._closed = True
        self._state = TraceRecordingState.CLOSED
