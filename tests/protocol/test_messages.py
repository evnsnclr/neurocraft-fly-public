from __future__ import annotations

import base64
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
from neurocraft_protocol import (
    MAX_FRAME_BYTES,
    Action,
    ActuationReceipt,
    ActuationValidation,
    ControlAck,
    ControlCommand,
    ControlOperation,
    ControlState,
    Fault,
    FaultCode,
    Hello,
    HelloAck,
    MotorVector,
    Observation,
    OversizeFrameError,
    PeerRole,
    ProgramAction,
    ProgramId,
    ProgramReceipt,
    ProgramRefusalReason,
    ProgramValidation,
    ProtocolError,
    RunIdentity,
    UnknownFieldError,
    canonical_json_bytes,
    encode_message,
    generate_client_nonce,
    handshake_auth,
    handshake_transcript,
    parse_json_frame,
    parse_message,
)

from .helpers import action, identity, observation, sensors, world_frame

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "packages" / "protocol" / "golden"


@pytest.mark.parametrize(
    "fixture_name,expected_type",
    [
        ("hello.json", Hello),
        ("hello_ack.json", HelloAck),
        ("observation.json", Observation),
        ("action.json", Action),
        ("program_action.json", ProgramAction),
        ("actuation_receipt.json", ActuationReceipt),
        ("actuation_receipt_rejected.json", ActuationReceipt),
        ("program_receipt.json", ProgramReceipt),
        ("program_receipt_refused.json", ProgramReceipt),
        ("program_receipt_refused_operator.json", ProgramReceipt),
        ("program_receipt_rejected.json", ProgramReceipt),
        ("control_command.json", ControlCommand),
        ("control_ack.json", ControlAck),
        ("intervention_control_command.json", ControlCommand),
        ("intervention_control_ack.json", ControlAck),
        ("reset_control_ack.json", ControlAck),
        ("fault.json", Fault),
    ],
)
def test_golden_messages_round_trip(fixture_name: str, expected_type: type[object]) -> None:
    first = parse_message((GOLDEN / fixture_name).read_bytes())
    assert isinstance(first, expected_type)
    encoded = encode_message(first)
    assert len(encoded.encode("utf-8")) <= MAX_FRAME_BYTES
    assert parse_message(encoded) == first
    assert encoded == canonical_json_bytes(first).decode("utf-8")


def test_nested_object_parser_returns_validated_mapping_regression() -> None:
    parsed = Observation.from_dict(observation().to_dict())
    assert parsed.identity.run_id == "test-run"
    assert parsed.sensors.position_m == (1.0, 64.0, -2.0)


def test_schema_document_is_valid_json_and_closes_objects() -> None:
    schema = json.loads((ROOT / "packages/protocol/schema/protocol-v2.schema.json").read_text())
    assert schema["$id"] == "urn:neurocraft:protocol:v2"
    for definition in (
        "identity",
        "sensors",
        "motorVector",
        "worldFrame",
        "hello",
        "helloAck",
        "observation",
        "action",
        "programAction",
        "actuationReceipt",
        "programReceipt",
        "programModulation",
        "programParameters",
        "readoutValues",
        "controlCommand",
        "controlAck",
        "fault",
    ):
        assert schema["$defs"][definition]["additionalProperties"] is False
    observation_schema = schema["$defs"]["observation"]
    assert "bridge_observations_coalesced" in observation_schema["required"]
    assert observation_schema["properties"]["bridge_observations_coalesced"] == {
        "$ref": "#/$defs/nonnegativeInteger"
    }
    receipt_schema = schema["$defs"]["actuationReceipt"]
    assert "world_before" in receipt_schema["required"]
    assert receipt_schema["properties"]["world_before"] == {"$ref": "#/$defs/worldFrame"}
    # The body-program frames are siblings inside protocol v2, not a version bump
    # (ADR 0020 section 7), so they are members of the same one-of and their
    # objects close exactly as every other frame's do.
    one_of = [entry["$ref"].rsplit("/", 1)[-1] for entry in schema["oneOf"]]
    assert "programAction" in one_of and "programReceipt" in one_of
    program_action_schema = schema["$defs"]["programAction"]
    assert "motors" not in program_action_schema["properties"]
    assert "neutral" not in program_action_schema["properties"]
    assert set(program_action_schema["required"]) == ProgramAction.FIELDS
    assert set(schema["$defs"]["programReceipt"]["required"]) == ProgramReceipt.FIELDS


def test_unknown_message_field_is_rejected() -> None:
    raw = observation().to_dict()
    raw["semantic_target"] = "diamond"
    with pytest.raises(UnknownFieldError):
        parse_message(json.dumps(raw))


def test_unknown_sensor_and_motor_fields_are_rejected() -> None:
    raw_observation = observation().to_dict()
    raw_observation["sensors"]["nearest_goal"] = [1, 2, 3]
    with pytest.raises(UnknownFieldError):
        parse_message(json.dumps(raw_observation))

    raw_action = action().to_dict()
    raw_action["motors"]["navigate_to"] = 1
    with pytest.raises(UnknownFieldError):
        parse_message(json.dumps(raw_action))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_numbers_are_rejected(value: float) -> None:
    with pytest.raises(ProtocolError):
        MotorVector(forward_thrust=value)
    raw = '{"type":"x","value":NaN}'
    with pytest.raises(ProtocolError):
        parse_json_frame(raw)


def test_duplicate_json_keys_are_rejected() -> None:
    with pytest.raises(ProtocolError, match="duplicate JSON key"):
        parse_json_frame('{"type":"hello","type":"observation"}')


def test_invalid_utf8_and_non_object_json_are_rejected() -> None:
    with pytest.raises(ProtocolError, match="UTF-8"):
        parse_json_frame(b"\xff")
    with pytest.raises(ProtocolError, match="top-level"):
        parse_json_frame("[]")


def test_frame_limit_uses_utf8_bytes() -> None:
    with pytest.raises(OversizeFrameError):
        parse_json_frame('"' + "é" * (MAX_FRAME_BYTES // 2) + '"')


@pytest.mark.parametrize("axis", sorted(MotorVector.FIELDS))
@pytest.mark.parametrize("value", [-1.00001, 1.00001])
def test_every_motor_axis_is_bounded(axis: str, value: float) -> None:
    kwargs = {name: 0.0 for name in MotorVector.FIELDS}
    kwargs[axis] = value
    with pytest.raises(ProtocolError):
        MotorVector(**kwargs)


def test_boolean_is_not_accepted_as_number_or_integer() -> None:
    with pytest.raises(ProtocolError):
        RunIdentity(
            run_id="run",
            session_epoch=True,
            entity_id="fly",
            graph_hash="1" * 64,
            adapter_hash="2" * 64,
            dynamics_hash="3" * 64,
            decoder_hash="4" * 64,
            observation_schema_hash="5" * 64,
        )
    with pytest.raises(ProtocolError):
        MotorVector(forward_thrust=True)


def test_sensor_ray_count_and_distance_bound_are_enforced() -> None:
    with pytest.raises(ProtocolError):
        replace(sensors(), ray_distances_m=(1.0, 2.0))
    with pytest.raises(ProtocolError):
        replace(sensors(), ray_distances_m=(9.0, 1.0, 1.0, 1.0, 1.0, 1.0))


def test_sensor_missing_and_clamped_names_are_strict() -> None:
    with pytest.raises(ProtocolError):
        replace(observation(), missing_fields=("task_target",))
    with pytest.raises(ProtocolError):
        replace(observation(), clamped_fields=("yaw_rad", "yaw_rad"))


def test_observation_bridge_coalescence_is_required_nonnegative_int64() -> None:
    assert observation(bridge_observations_coalesced=2).bridge_observations_coalesced == 2
    with pytest.raises(ProtocolError, match="bridge_observations_coalesced"):
        replace(observation(), bridge_observations_coalesced=-1)
    with pytest.raises(ProtocolError, match="bridge_observations_coalesced"):
        replace(observation(), bridge_observations_coalesced=True)
    with pytest.raises(ProtocolError, match="bridge_observations_coalesced"):
        replace(observation(), bridge_observations_coalesced=2**63)

    missing = observation().to_dict()
    del missing["bridge_observations_coalesced"]
    with pytest.raises(ProtocolError, match="missing fields"):
        Observation.from_dict(missing)


def test_action_temporal_contract_is_enforced() -> None:
    with pytest.raises(ProtocolError, match="declared action budget"):
        action(valid_through_tick=103)
    with pytest.raises(ProtocolError, match="non-empty"):
        replace(action(), neural_step_end=0)
    with pytest.raises(ProtocolError, match="precedes"):
        action(target_tick=99, valid_through_tick=100)


def test_neutral_flag_requires_zero_primitive_vector() -> None:
    with pytest.raises(ProtocolError, match="all-zero"):
        action(neutral=True)
    neutral = action(motors=MotorVector.neutral(), neutral=True)
    assert neutral.motors.is_neutral


def test_hello_carries_no_token_material() -> None:
    secret = "A" * 43
    hello = Hello.create(
        identity=identity(),
        role=PeerRole.MINECRAFT_SERVER,
        token=secret,
    )
    encoded = encode_message(hello)

    assert secret not in encoded
    assert "token" not in encoded
    assert secret not in repr(hello)
    assert "token=" not in repr(hello)
    assert set(json.loads(encoded)) == Hello.FIELDS
    # The proof must actually be derived from the secret, not merely absent from the frame.
    assert hello.verify(secret)
    assert not hello.verify("B" * 43)


def test_hello_auth_hmac_mismatch_refused() -> None:
    hello = Hello.create(identity=identity(), role=PeerRole.MINECRAFT_SERVER, token="A" * 43)

    assert not replace(hello, auth="0" * 64).verify("A" * 43)
    assert not replace(hello, client_nonce=generate_client_nonce()).verify("A" * 43)
    assert not replace(hello, identity=identity(session_epoch=2)).verify("A" * 43)
    assert not replace(hello, role=PeerRole.NEURAL_SERVICE).verify("A" * 43)
    # A hello_ack proof for the same nonce must not authenticate a hello: the transcript
    # label domain-separates the two handshake directions.
    crossed = handshake_auth(
        "A" * 43,
        label=HelloAck.TYPE,
        identity=hello.identity,
        role=hello.role,
        client_nonce=hello.client_nonce,
    )
    assert not replace(hello, auth=crossed).verify("A" * 43)

    raw = hello.to_dict()
    raw["auth"] = "not-a-digest"
    with pytest.raises(ProtocolError, match="auth"):
        parse_message(json.dumps(raw))


def test_hello_ack_requires_server_auth() -> None:
    nonce = generate_client_nonce()
    ack = HelloAck.create(identity=identity(), token="A" * 43, client_nonce=nonce)

    assert ack.verify("A" * 43, client_nonce=nonce)
    assert not ack.verify("A" * 43, client_nonce=generate_client_nonce())
    assert not ack.verify("B" * 43, client_nonce=nonce)

    raw = ack.to_dict()
    del raw["server_auth"]
    with pytest.raises(ProtocolError, match="server_auth"):
        parse_message(json.dumps(raw))


def test_handshake_transcript_matches_the_pinned_cross_language_vector() -> None:
    """Pin the exact bytes and digests that ControlProtocolTest asserts on the Java side."""

    golden_identity = RunIdentity(
        run_id="demo-001",
        session_epoch=1,
        entity_id="fly-01",
        graph_hash="1" * 64,
        adapter_hash="2" * 64,
        dynamics_hash="3" * 64,
        decoder_hash="4" * 64,
        observation_schema_hash="5" * 64,
    )
    token = "A" * 43
    nonce = base64.urlsafe_b64encode(bytes(range(32))).decode("ascii").rstrip("=")

    assert nonce == "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8"
    transcript = handshake_transcript(
        label=Hello.TYPE,
        identity=golden_identity,
        role=PeerRole.MINECRAFT_SERVER,
        client_nonce=nonce,
    )
    assert len(transcript) == 469
    assert transcript.startswith(b"\x00\x00\x00\x16neurocraft.protocol.v2\x00\x00\x00\x05hello")
    assert transcript.endswith(b"\x00\x00\x00\x2b" + nonce.encode("ascii"))
    assert (
        handshake_auth(
            token,
            label=Hello.TYPE,
            identity=golden_identity,
            role=PeerRole.MINECRAFT_SERVER,
            client_nonce=nonce,
        )
        == "154741d2ddc60145534fbb783a5cce817882786ac28f14c31c5cb72559ffe73b"
    )
    assert (
        handshake_auth(
            token,
            label=HelloAck.TYPE,
            identity=golden_identity,
            role=PeerRole.NEURAL_SERVICE,
            client_nonce=nonce,
        )
        == "ac0e2401342ca596b6b67f43140a652bc96dabeecfaa600a917beab809dd57ee"
    )

    golden_hello = parse_message((GOLDEN / "hello.json").read_bytes())
    golden_ack = parse_message((GOLDEN / "hello_ack.json").read_bytes())
    assert isinstance(golden_hello, Hello)
    assert isinstance(golden_ack, HelloAck)
    assert golden_hello.verify(token)
    assert golden_ack.verify(token, client_nonce=nonce)


def test_identity_hashes_are_lowercase_sha256() -> None:
    with pytest.raises(ProtocolError):
        replace(identity(), graph_hash="A" * 64)
    with pytest.raises(ProtocolError):
        replace(identity(), graph_hash="1" * 63)


def test_canonical_json_normalizes_negative_zero_and_is_stable() -> None:
    first = canonical_json_bytes({"z": -0.0, "a": [2, 1]})
    second = canonical_json_bytes({"a": [2, 1], "z": 0.0})
    assert first == second == b'{"a":[2,1],"z":0.0}'


def test_fault_must_declare_neutral_applied() -> None:
    raw = json.loads((GOLDEN / "fault.json").read_text())
    raw["neutral_applied"] = False
    with pytest.raises(ProtocolError, match="neutral_applied"):
        parse_message(json.dumps(raw))


def test_actuation_receipt_semantics_are_strict() -> None:
    requested = MotorVector(forward_thrust=0.5, yaw_torque=0.2)
    with pytest.raises(ProtocolError, match="exactly"):
        ActuationReceipt(
            identity=identity(),
            command_seq=0,
            observation_seq=0,
            target_tick=101,
            applied_tick=101,
            world_before=world_frame(),
            requested_motors=requested,
            applied_motors=MotorVector(forward_thrust=0.4, yaw_torque=0.2),
            validation=ActuationValidation.APPLIED,
        )
    clamped = ActuationReceipt(
        identity=identity(),
        command_seq=0,
        observation_seq=0,
        target_tick=101,
        applied_tick=101,
        world_before=world_frame(),
        requested_motors=requested,
        applied_motors=MotorVector(forward_thrust=0.4, yaw_torque=0.2),
        validation=ActuationValidation.CLAMPED,
        clamp_fields=("forward_thrust",),
    )
    assert clamped.validation is ActuationValidation.CLAMPED
    assert clamped.world_before == world_frame()
    missing_world = clamped.to_dict()
    del missing_world["world_before"]
    with pytest.raises(ProtocolError, match="missing fields"):
        ActuationReceipt.from_dict(missing_world)
    forged_world = clamped.to_dict()
    forged_world["world_before"]["semantic_target"] = "diamond"
    with pytest.raises(UnknownFieldError, match="world_before"):
        ActuationReceipt.from_dict(forged_world)
    with pytest.raises(ProtocolError, match="neutral"):
        replace(
            clamped,
            validation=ActuationValidation.NEUTRAL_FAULT,
            clamp_fields=(),
            fault_code=FaultCode.STALE,
        )
    with pytest.raises(ProtocolError, match="within two target ticks"):
        replace(clamped, applied_tick=104)
    delayed_rejection = ActuationReceipt(
        identity=identity(),
        command_seq=0,
        observation_seq=0,
        target_tick=101,
        applied_tick=104,
        world_before=world_frame(),
        requested_motors=requested,
        applied_motors=MotorVector.neutral(),
        validation=ActuationValidation.REJECTED,
        fault_code=FaultCode.STALE,
    )
    assert delayed_rejection.applied_tick == 104
    future_rejection = replace(delayed_rejection, applied_tick=100, fault_code=FaultCode.FUTURE)
    assert future_rejection.applied_tick == 100


def test_control_content_is_identifier_hash_only_and_operation_bound() -> None:
    with pytest.raises(ProtocolError, match="cannot carry"):
        ControlCommand(
            identity=identity(),
            control_seq=0,
            issued_tick=100,
            operation=ControlOperation.PAUSE,
            content_id="trace-1",
        )
    with pytest.raises(ProtocolError, match="requires content_hash"):
        ControlCommand(
            identity=identity(),
            control_seq=0,
            issued_tick=100,
            operation=ControlOperation.REPLAY_START,
            content_id="trace-1",
        )
    command = ControlCommand(
        identity=identity(),
        control_seq=0,
        issued_tick=100,
        operation=ControlOperation.REPLAY_START,
        content_id="trace-1",
        content_hash="a" * 64,
    )
    ack = ControlAck(
        identity=identity(),
        control_seq=0,
        issued_tick=100,
        operation=ControlOperation.REPLAY_START,
        content_id="trace-1",
        content_hash="a" * 64,
        accepted=True,
        state=ControlState.REPLAYING,
        effective_tick=100,
        fault_code=None,
    )
    assert parse_message(encode_message(command)) == command
    assert parse_message(encode_message(ack)) == ack


@pytest.mark.parametrize(
    "operation",
    [ControlOperation.LESION_APPLY, ControlOperation.STIMULATION_APPLY],
)
def test_intervention_apply_requires_named_hash_bound_content(
    operation: ControlOperation,
) -> None:
    with pytest.raises(ProtocolError, match="content_id and content_hash"):
        ControlCommand(
            identity=identity(),
            control_seq=0,
            issued_tick=100,
            operation=operation,
            content_id="declared-intervention",
        )
    command = ControlCommand(
        identity=identity(),
        control_seq=0,
        issued_tick=100,
        operation=operation,
        content_id="declared-intervention",
        content_hash="b" * 64,
    )
    assert parse_message(encode_message(command)) == command


@pytest.mark.parametrize(
    "operation",
    [ControlOperation.LESION_CLEAR, ControlOperation.STIMULATION_CLEAR],
)
def test_intervention_clear_requires_id_and_permits_echoed_hash(
    operation: ControlOperation,
) -> None:
    with pytest.raises(ProtocolError, match="requires content_id"):
        ControlCommand(
            identity=identity(),
            control_seq=0,
            issued_tick=100,
            operation=operation,
        )
    without_hash = ControlCommand(
        identity=identity(),
        control_seq=0,
        issued_tick=100,
        operation=operation,
        content_id="declared-intervention",
    )
    with_hash = replace(without_hash, content_hash="b" * 64)
    assert parse_message(encode_message(without_hash)) == without_hash
    assert parse_message(encode_message(with_hash)) == with_hash


def test_rejected_control_ack_requires_fault() -> None:
    with pytest.raises(ProtocolError, match="requires a fault"):
        ControlAck(
            identity=identity(),
            control_seq=0,
            issued_tick=100,
            operation=ControlOperation.PAUSE,
            content_id=None,
            content_hash=None,
            accepted=False,
            state=ControlState.PAUSED,
            effective_tick=100,
            fault_code=None,
        )


def test_only_accepted_reset_ack_may_bind_second_future_tick() -> None:
    reset = ControlAck(
        identity=identity(),
        control_seq=0,
        issued_tick=100,
        operation=ControlOperation.RESET,
        content_id=None,
        content_hash=None,
        accepted=True,
        state=ControlState.RECORDING,
        effective_tick=102,
        fault_code=None,
    )
    assert reset.effective_tick == 102
    assert replace(reset, effective_tick=101).effective_tick == 101
    with pytest.raises(ProtocolError, match="accepted RESET"):
        replace(reset, effective_tick=100)
    with pytest.raises(ProtocolError, match="next tick"):
        replace(reset, operation=ControlOperation.PAUSE)
    with pytest.raises(ProtocolError, match="next tick"):
        replace(reset, accepted=False, fault_code=FaultCode.INVALID_STATE)


def test_program_receipt_goldens_pin_the_two_refusal_shapes() -> None:
    """The vectors lane M builds its Java encoder against.

    A `unregistered` or `precondition` refusal ran the declared HOVER fallback, and it ran under
    the REQUESTED modulation and parameters: `BodyProgramRuntime` sets those before the registry
    lookup and the `canEnter` check, so the echo is what actually happened. An `operator` refusal
    is a different event — the frame never reached the body, the operator's own program was
    running, and its modulation and parameters are the operator lane's, unrelated to the request.

    `emitted_motors` in every program golden is a CONSTRUCTED vector, not simulated output of the
    named program: nothing in this repository runs HOVER or CRUISE to produce these numbers, and
    the values are illustrative. Lane M should build its Java vectors the same way, through
    `ValidationResult`-style construction, and must not expect them to match a program it runs.
    """

    fallback = parse_message((GOLDEN / "program_receipt_refused.json").read_bytes())
    assert isinstance(fallback, ProgramReceipt)
    assert fallback.reason is ProgramRefusalReason.UNREGISTERED
    assert fallback.executed_program is ProgramId.HOVER
    assert fallback.executed_modulation == fallback.requested_modulation
    assert fallback.executed_parameters == fallback.requested_parameters

    operator = parse_message((GOLDEN / "program_receipt_refused_operator.json").read_bytes())
    assert isinstance(operator, ProgramReceipt)
    assert operator.reason is ProgramRefusalReason.OPERATOR
    assert operator.requested_program is ProgramId.ESCAPE_JUMP
    assert operator.executed_program is ProgramId.CRUISE
    assert operator.executed_modulation != operator.requested_modulation
    assert operator.validation is ProgramValidation.PROGRAM_REFUSED
    assert operator.fault_code is None
