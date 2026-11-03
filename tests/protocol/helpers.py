from __future__ import annotations

from neurocraft_protocol import (
    ACTION_TARGET_OFFSET_TICKS,
    Action,
    ActuationReceipt,
    ActuationValidation,
    ArtifactIdentities,
    CausalRecord,
    ClaimScope,
    ControlAck,
    ControlCommand,
    ControlOperation,
    ControlState,
    DataTier,
    DeterminismLevel,
    FaultCode,
    FloatPrecision,
    MotorVector,
    Observation,
    ProgramAction,
    ProgramId,
    ProgramModulation,
    ProgramParameters,
    ProgramReceipt,
    ProgramRefusalReason,
    ProgramValidation,
    ReadoutValues,
    RunIdentity,
    RunMode,
    SensorFrame,
    TraceManifest,
    ValidationResult,
    WorldState,
)
from neurocraft_protocol.messages import WorldFrame


def identity(*, session_epoch: int = 1, graph_hash: str = "c" * 64) -> RunIdentity:
    return RunIdentity(
        run_id="test-run",
        session_epoch=session_epoch,
        entity_id="fly-01",
        graph_hash=graph_hash,
        adapter_hash="2" * 64,
        dynamics_hash="3" * 64,
        decoder_hash="4" * 64,
        observation_schema_hash="5" * 64,
    )


def sensors() -> SensorFrame:
    return SensorFrame(
        position_m=(1.0, 64.0, -2.0),
        velocity_m_s=(0.1, 0.0, -0.2),
        yaw_rad=0.25,
        pitch_rad=-0.1,
        on_ground=False,
        horizontal_collision=False,
        vertical_collision=False,
        in_fluid=False,
        local_light=12,
        ray_max_distance_m=8.0,
        ray_distances_m=(8.0, 6.0, 5.0, 8.0, 2.5, 3.0),
    )


def observation(
    *,
    sequence: int = 0,
    tick: int = 100,
    bridge_observations_coalesced: int = 0,
    run_identity: RunIdentity | None = None,
) -> Observation:
    return Observation(
        identity=run_identity or identity(),
        observation_seq=sequence,
        game_tick=tick,
        sent_monotonic_ns=123_456_789 + sequence,
        sensors=sensors(),
        bridge_observations_coalesced=bridge_observations_coalesced,
    )


def action(
    *,
    command_sequence: int = 0,
    observation_sequence: int = 0,
    observation_tick: int = 100,
    target_tick: int | None = None,
    valid_through_tick: int | None = None,
    neural_start: int = 0,
    state_digest: str = "6" * 64,
    run_identity: RunIdentity | None = None,
    motors: MotorVector | None = None,
    neutral: bool = False,
) -> Action:
    chosen_motors = motors or MotorVector(forward_thrust=0.25)
    return Action(
        identity=run_identity or identity(),
        command_seq=command_sequence,
        observation_seq=observation_sequence,
        observation_tick=observation_tick,
        target_tick=(
            observation_tick + ACTION_TARGET_OFFSET_TICKS if target_tick is None else target_tick
        ),
        valid_through_tick=(
            observation_tick + ACTION_TARGET_OFFSET_TICKS
            if valid_through_tick is None
            else valid_through_tick
        ),
        neural_step_start=neural_start,
        neural_step_end=neural_start + 5,
        state_digest=state_digest,
        motors=chosen_motors,
        neutral=neutral,
    )


def modulation(
    *,
    speed: float = 0.7,
    turn_rate: float = 1.0,
    amplitude: float = 1.0,
    urgency: float = 0.0,
) -> ProgramModulation:
    return ProgramModulation(speed=speed, turn_rate=turn_rate, amplitude=amplitude, urgency=urgency)


def parameters(*, turn_bias: float = -0.25) -> ProgramParameters:
    return ProgramParameters(turn_bias=turn_bias)


def readout(
    *, ttmn_jump: float = 0.0, dna01_turn: float = -0.25, wing_common: float = 0.61
) -> ReadoutValues:
    return ReadoutValues(ttmn_jump=ttmn_jump, dna01_turn=dna01_turn, wing_common=wing_common)


def program_action(
    *,
    command_sequence: int = 0,
    observation_sequence: int = 0,
    observation_tick: int = 100,
    target_tick: int | None = None,
    valid_through_tick: int | None = None,
    neural_start: int = 0,
    neural_end: int | None = None,
    state_digest: str = "6" * 64,
    run_identity: RunIdentity | None = None,
    program: ProgramId = ProgramId.CRUISE,
    program_modulation: ProgramModulation | None = None,
    program_parameters: ProgramParameters | None = None,
    readout_values: ReadoutValues | None = None,
    policy_hash: str = "7" * 64,
) -> ProgramAction:
    return ProgramAction(
        identity=run_identity or identity(),
        command_seq=command_sequence,
        observation_seq=observation_sequence,
        observation_tick=observation_tick,
        target_tick=(
            observation_tick + ACTION_TARGET_OFFSET_TICKS if target_tick is None else target_tick
        ),
        valid_through_tick=(
            observation_tick + ACTION_TARGET_OFFSET_TICKS
            if valid_through_tick is None
            else valid_through_tick
        ),
        neural_step_start=neural_start,
        neural_step_end=neural_start + 5 if neural_end is None else neural_end,
        state_digest=state_digest,
        program=program,
        modulation=program_modulation or modulation(),
        parameters=program_parameters or parameters(),
        readout=readout_values or readout(),
        policy_hash=policy_hash,
    )


def program_receipt(
    issued: ProgramAction | None = None,
    *,
    validation: ProgramValidation = ProgramValidation.PROGRAM_APPLIED,
    applied_tick: int | None = None,
    emitted_motors: MotorVector | None = None,
    reason: ProgramRefusalReason | None = None,
    fault_code: FaultCode | None = None,
    executed_program: ProgramId | None = None,
    executed_modulation: ProgramModulation | None = None,
    executed_parameters: ProgramParameters | None = None,
) -> ProgramReceipt:
    """Build a receipt for ``issued``; the ``executed_*`` overrides build the operator shape.

    By default a refusal echoes the request under the declared HOVER fallback, which is the
    ``unregistered``/``precondition`` convention. An ``operator`` refusal ran the operator's own
    program under the operator's own modulation, so it supplies all three overrides.
    """

    frame = issued or program_action()
    rejected = validation is ProgramValidation.REJECTED
    refused = validation is ProgramValidation.PROGRAM_REFUSED
    if executed_program is None:
        executed_program = None if rejected else (ProgramId.HOVER if refused else frame.program)
    if executed_modulation is None:
        executed_modulation = None if rejected else frame.modulation
    if executed_parameters is None:
        executed_parameters = None if rejected else frame.parameters
    return ProgramReceipt(
        identity=frame.identity,
        command_seq=frame.command_seq,
        observation_seq=frame.observation_seq,
        target_tick=frame.target_tick,
        applied_tick=frame.target_tick if applied_tick is None else applied_tick,
        world_before=world_frame(),
        requested_program=frame.program,
        requested_modulation=frame.modulation,
        requested_parameters=frame.parameters,
        executed_program=executed_program,
        executed_modulation=executed_modulation,
        executed_parameters=executed_parameters,
        emitted_motors=(
            MotorVector.neutral()
            if rejected
            else (emitted_motors or MotorVector(forward_thrust=0.35))
        ),
        validation=validation,
        reason=(ProgramRefusalReason.UNREGISTERED if refused and reason is None else reason),
        fault_code=(FaultCode.STALE if rejected and fault_code is None else fault_code),
    )


def receipt(issued: Action | None = None) -> ActuationReceipt:
    action_value = issued or action()
    return ActuationReceipt(
        identity=action_value.identity,
        command_seq=action_value.command_seq,
        observation_seq=action_value.observation_seq,
        target_tick=action_value.target_tick,
        applied_tick=action_value.target_tick,
        world_before=world_frame(),
        requested_motors=action_value.motors,
        applied_motors=action_value.motors,
        validation=ActuationValidation.APPLIED,
    )


def world_frame(*, x: float = 1.0) -> WorldFrame:
    return WorldFrame(
        position_m=(x, 64.0, -2.0),
        velocity_m_s=(0.1, 0.0, -0.2),
        yaw_rad=0.25,
        pitch_rad=-0.1,
        on_ground=False,
        horizontal_collision=False,
        vertical_collision=False,
        in_fluid=False,
    )


def control_command(
    *,
    sequence: int = 0,
    tick: int = 100,
    operation: ControlOperation = ControlOperation.PAUSE,
    content_id: str | None = None,
    content_hash: str | None = None,
) -> ControlCommand:
    return ControlCommand(
        identity=identity(),
        control_seq=sequence,
        issued_tick=tick,
        operation=operation,
        content_id=content_id,
        content_hash=content_hash,
    )


def control_ack(
    command: ControlCommand | None = None,
    *,
    accepted: bool = True,
    state: ControlState = ControlState.PAUSED,
) -> ControlAck:
    command_value = command or control_command()
    return ControlAck(
        identity=command_value.identity,
        control_seq=command_value.control_seq,
        issued_tick=command_value.issued_tick,
        operation=command_value.operation,
        content_id=command_value.content_id,
        content_hash=command_value.content_hash,
        accepted=accepted,
        state=state,
        effective_tick=command_value.issued_tick,
        fault_code=None if accepted else FaultCode.INVALID_STATE,
    )


def artifacts() -> ArtifactIdentities:
    return ArtifactIdentities(
        source_hash="a" * 64,
        canonical_graph_hash="b" * 64,
        runtime_graph_hash="c" * 64,
        fixture_hash="d" * 64,
        code_hash="e" * 64,
        dependency_lock_hash="f" * 64,
        adapter_hash="2" * 64,
        dynamics_hash="3" * 64,
        sign_hash="7" * 64,
        decoder_hash="4" * 64,
        arena_hash="8" * 64,
        configuration_hash="9" * 64,
        observation_schema_hash="5" * 64,
    )


def manifest() -> TraceManifest:
    return TraceManifest(
        run_id="test-run",
        initial_session_epoch=1,
        mode=RunMode.SYNTHETIC,
        data_tier=DataTier.SYNTHETIC,
        claim_scope=ClaimScope.G0,
        artifacts=artifacts(),
        game_tick_hz=20.0,
        neural_step_ms=10.0,
        precision=FloatPrecision.FLOAT64,
        backend="synthetic-reference",
        thread_count=1,
        seed=20260817,
        determinism_level=DeterminismLevel.SAME_BUILD_EXACT,
        machine="test-machine",
        os="test-os",
        python="3.13",
        java="21",
        minecraft="1.21.5",
        fabric="0.16.10",
        mod="0.1.0",
        privacy_policy="scientific-trace-v1",
        expected_chunks=1,
        checkpoint_interval_ticks=100,
    )


def world(*, x: float = 1.0) -> WorldState:
    return WorldState(
        position_m=(x, 64.0, -2.0),
        velocity_m_s=(0.1, 0.0, -0.2),
        yaw_rad=0.25,
        pitch_rad=-0.1,
        on_ground=False,
        horizontal_collision=False,
        vertical_collision=False,
        in_fluid=False,
    )


def causal_record(*, sequence: int = 0, tick: int = 100) -> CausalRecord:
    obs = observation(sequence=sequence, tick=tick)
    motors = MotorVector(forward_thrust=0.1, yaw_torque=0.02)
    issued = action(
        command_sequence=sequence,
        observation_sequence=sequence,
        observation_tick=tick,
        neural_start=sequence * 5,
        state_digest="b" * 64,
        run_identity=obs.identity,
        motors=motors,
    )
    return CausalRecord(
        identity=obs.identity,
        record_seq=sequence,
        game_tick=tick,
        # The record's action applies one action budget after the observation it answers, so
        # the world is ACTION_TARGET_OFFSET_TICKS on from the observation when it is applied
        # and one further on in world_after_game_tick.
        observation_age_ticks=ACTION_TARGET_OFFSET_TICKS,
        observation=obs,
        encoded_stimulation={"805722": 0.4, "77185": 0.2},
        direct_stimulation={},
        direct_stimulation_sha256="0" * 64,
        neural_step_start=sequence * 5,
        neural_step_end=sequence * 5 + 5,
        state_before_digest="a" * 64,
        state_after_digest="b" * 64,
        state_before_activity=(0.0, 0.1),
        state_after_activity=(0.1, 0.2),
        interventions=(),
        delay_state_digest="c" * 64,
        checkpoint_reference=None,
        motor_pool_values={"DNg07_L": 0.15, "DNg07_R": -0.1},
        pool_membership_hash="d" * 64,
        decoder_output=motors,
        issued_action=issued,
        actuation_receipt=None,
        command_seq=sequence,
        validation=ValidationResult.APPLIED,
        clamp_fields=(),
        applied_action=motors,
        neutral_reason=None,
        world_before=world(x=1.0 + sequence),
        world_after=world(x=1.1 + sequence),
        world_after_observation_seq=sequence + 1,
        world_after_game_tick=tick + ACTION_TARGET_OFFSET_TICKS + 1,
        passive_forces=(0.0, -0.08, 0.0),
        external_forces=(0.0, 0.0, 0.0),
        queue_coalesced=0,
        queue_overruns=0,
        compute_ns=1_000_000,
        serialization_ns=20_000,
        write_ns=30_000,
        fault_code=None,
    )
