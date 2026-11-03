"""Strictness of the body-program command and receipt frames (ADR 0020 section 7)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from neurocraft_protocol import (
    FaultCode,
    MotorVector,
    ProgramAction,
    ProgramId,
    ProgramReceipt,
    ProgramRefusalReason,
    ProgramValidation,
    ProtocolError,
    UnknownFieldError,
    parse_message,
)

from .helpers import (
    modulation,
    parameters,
    program_action,
    program_receipt,
    readout,
)


def test_program_action_refuses_empty_neural_interval() -> None:
    """A program frame is never neutral, so the connectome must have stepped for it."""

    with pytest.raises(ProtocolError, match="neural interval must be non-empty"):
        program_action(neural_start=7, neural_end=7)
    with pytest.raises(ProtocolError, match="neural interval must be non-empty"):
        program_action(neural_start=7, neural_end=6)


def test_program_action_refuses_unknown_program() -> None:
    raw = program_action().to_dict()
    raw["program"] = "PIROUETTE"
    with pytest.raises(ProtocolError, match=r"program_action\.program must be one of"):
        parse_message(json.dumps(raw))


def test_program_action_refuses_out_of_range_modulation_parameters_and_readout() -> None:
    """Modulations attenuate in [0, 1]; the signed parameter and readouts have their own bounds."""

    for field, value in (
        ("speed", -0.01),
        ("turn_rate", 1.01),
        ("amplitude", 2.0),
        ("urgency", -1.0),
    ):
        with pytest.raises(ProtocolError, match=rf"modulation\.{field}"):
            program_action(program_modulation=replace(modulation(), **{field: value}))
    for value in (-1.01, 1.01):
        with pytest.raises(ProtocolError, match=r"parameters\.turn_bias"):
            program_action(program_parameters=parameters(turn_bias=value))
    with pytest.raises(ProtocolError, match=r"readout\.ttmn_jump"):
        program_action(readout_values=readout(ttmn_jump=-0.001))
    with pytest.raises(ProtocolError, match=r"readout\.dna01_turn"):
        program_action(readout_values=readout(dna01_turn=1.5))
    with pytest.raises(ProtocolError, match=r"readout\.wing_common"):
        program_action(readout_values=readout(wing_common=-1.5))


@pytest.mark.parametrize(
    "container,field",
    [
        ("", "policy_confidence"),
        ("modulation", "aggression"),
        ("parameters", "pitch_bias"),
        ("readout", "dnp01_jump"),
    ],
)
def test_program_action_refuses_unknown_field(container: str, field: str) -> None:
    raw = program_action().to_dict()
    target = raw if container == "" else raw[container]
    assert isinstance(target, dict)
    target[field] = 0.5
    with pytest.raises(UnknownFieldError):
        parse_message(json.dumps(raw))


@pytest.mark.parametrize(
    "field",
    [
        "program",
        "modulation",
        "parameters",
        "readout",
        "policy_hash",
        "state_digest",
        "neural_step_end",
    ],
)
def test_program_action_refuses_a_missing_field(field: str) -> None:
    raw = program_action().to_dict()
    del raw[field]
    with pytest.raises(ProtocolError, match=f"missing fields: {field}"):
        parse_message(json.dumps(raw))


def test_program_action_carries_neither_motors_nor_a_neutral_flag() -> None:
    """A program frame requests no motors; the paused service keeps emitting neutral actions."""

    assert "motors" not in ProgramAction.FIELDS
    assert "neutral" not in ProgramAction.FIELDS
    raw = program_action().to_dict()
    raw["motors"] = MotorVector.neutral().to_dict()
    with pytest.raises(UnknownFieldError):
        parse_message(json.dumps(raw))


def test_program_action_keeps_the_action_envelope_rules() -> None:
    with pytest.raises(ProtocolError, match="target_tick precedes its observation"):
        program_action(observation_tick=100, target_tick=99, valid_through_tick=101)
    with pytest.raises(ProtocolError, match="valid_through_tick precedes target_tick"):
        program_action(observation_tick=100, target_tick=101, valid_through_tick=100)
    with pytest.raises(ProtocolError, match="declared action budget"):
        program_action(observation_tick=100, target_tick=101, valid_through_tick=103)


def test_program_action_refuses_a_non_sha256_policy_hash() -> None:
    with pytest.raises(ProtocolError, match=r"program_action\.policy_hash"):
        program_action(policy_hash="not-a-hash")


def test_program_receipt_validation_matrix() -> None:
    """The three validation states and the nullable executed values they imply."""

    frame = program_action()
    applied = program_receipt(frame)
    assert applied.executed_program is frame.program
    assert applied.executed_modulation == frame.modulation
    assert applied.reason is None and applied.fault_code is None

    refused = program_receipt(frame, validation=ProgramValidation.PROGRAM_REFUSED)
    assert refused.executed_program is ProgramId.HOVER
    assert refused.reason is ProgramRefusalReason.UNREGISTERED
    assert refused.fault_code is None

    rejected = program_receipt(frame, validation=ProgramValidation.REJECTED, applied_tick=104)
    assert rejected.executed_program is None
    assert rejected.executed_modulation is None
    assert rejected.executed_parameters is None
    assert rejected.emitted_motors.is_neutral
    assert rejected.fault_code is FaultCode.STALE
    assert rejected.reason is None


def test_program_receipt_refuses_executed_values_on_a_rejection() -> None:
    rejected = program_receipt(validation=ProgramValidation.REJECTED, applied_tick=104)
    for field in ("executed_program", "executed_modulation", "executed_parameters"):
        raw = rejected.to_dict()
        raw[field] = program_receipt().to_dict()[field]
        with pytest.raises(ProtocolError, match="REJECTED program_receipt executed nothing"):
            parse_message(json.dumps(raw))


def test_program_receipt_requires_executed_values_when_something_ran() -> None:
    for validation in (ProgramValidation.PROGRAM_APPLIED, ProgramValidation.PROGRAM_REFUSED):
        raw = program_receipt(validation=validation).to_dict()
        raw["executed_program"] = None
        with pytest.raises(ProtocolError, match="requires executed values"):
            parse_message(json.dumps(raw))


def test_applied_program_receipt_must_execute_exactly_what_was_requested() -> None:
    raw = program_receipt().to_dict()
    raw["executed_program"] = ProgramId.HOVER.value
    with pytest.raises(ProtocolError, match="must execute what was requested"):
        parse_message(json.dumps(raw))
    raw = program_receipt().to_dict()
    raw["executed_modulation"] = modulation(speed=0.1).to_dict()
    with pytest.raises(ProtocolError, match="must execute what was requested"):
        parse_message(json.dumps(raw))
    raw = program_receipt().to_dict()
    raw["executed_parameters"] = parameters(turn_bias=0.5).to_dict()
    with pytest.raises(ProtocolError, match="must execute what was requested"):
        parse_message(json.dumps(raw))


def test_applied_program_receipt_refuses_a_reason_or_a_fault() -> None:
    raw = program_receipt().to_dict()
    raw["reason"] = ProgramRefusalReason.OPERATOR.value
    with pytest.raises(ProtocolError, match="cannot carry a reason or a fault"):
        parse_message(json.dumps(raw))
    raw = program_receipt().to_dict()
    raw["fault_code"] = FaultCode.STALE.value
    with pytest.raises(ProtocolError, match="cannot carry a reason or a fault"):
        parse_message(json.dumps(raw))


@pytest.mark.parametrize(
    "reason", [ProgramRefusalReason.UNREGISTERED, ProgramRefusalReason.PRECONDITION]
)
def test_fallback_refusal_must_execute_hover(reason: ProgramRefusalReason) -> None:
    """`unregistered` and `precondition` both mean the declared HOVER fallback ran."""

    refused = program_receipt(validation=ProgramValidation.PROGRAM_REFUSED, reason=reason)
    assert refused.executed_program is ProgramId.HOVER
    raw = refused.to_dict()
    raw["executed_program"] = ProgramId.CRUISE.value
    with pytest.raises(ProtocolError, match="must fall back to HOVER"):
        parse_message(json.dumps(raw))


@pytest.mark.parametrize("executed", list(ProgramId))
def test_operator_refusal_may_execute_any_program(executed: ProgramId) -> None:
    """An operator hold ran the operator's own program, not the fallback.

    Pinning HOVER here would make the receipt assert a program ran that did not. The frame
    never reached the body at all, so the executed modulation and parameters are the operator
    lane's and need not echo the request.
    """

    frame = program_action(program=ProgramId.ESCAPE_JUMP)
    receipt = program_receipt(
        frame,
        validation=ProgramValidation.PROGRAM_REFUSED,
        reason=ProgramRefusalReason.OPERATOR,
        executed_program=executed,
        executed_modulation=modulation(speed=0.5, turn_rate=0.5),
        executed_parameters=parameters(turn_bias=0.0),
    )
    assert receipt.executed_program is executed
    assert receipt.requested_program is ProgramId.ESCAPE_JUMP
    assert receipt.executed_modulation != receipt.requested_modulation
    assert ProgramReceipt.from_dict(receipt.to_dict()) == receipt


def test_refused_program_receipt_requires_a_declared_reason() -> None:
    raw = program_receipt(validation=ProgramValidation.PROGRAM_REFUSED).to_dict()
    raw["reason"] = None
    with pytest.raises(ProtocolError, match="requires a reason and no fault"):
        parse_message(json.dumps(raw))
    raw = program_receipt(validation=ProgramValidation.PROGRAM_REFUSED).to_dict()
    raw["reason"] = "vibes"
    with pytest.raises(ProtocolError, match=r"program_receipt\.reason must be one of"):
        parse_message(json.dumps(raw))
    raw = program_receipt(validation=ProgramValidation.PROGRAM_REFUSED).to_dict()
    raw["fault_code"] = FaultCode.STALE.value
    with pytest.raises(ProtocolError, match="requires a reason and no fault"):
        parse_message(json.dumps(raw))


def test_operator_refusal_still_requires_an_executed_program() -> None:
    """ "Any of the ten" is not "none": a refusal always ran something."""

    raw = program_receipt(
        validation=ProgramValidation.PROGRAM_REFUSED,
        reason=ProgramRefusalReason.OPERATOR,
        executed_program=ProgramId.CRUISE,
    ).to_dict()
    raw["executed_program"] = None
    with pytest.raises(ProtocolError, match="requires executed values"):
        parse_message(json.dumps(raw))


def test_rejected_program_receipt_requires_a_fault_and_neutral_motors() -> None:
    raw = program_receipt(validation=ProgramValidation.REJECTED, applied_tick=104).to_dict()
    raw["fault_code"] = None
    with pytest.raises(ProtocolError, match="requires a fault and no reason"):
        parse_message(json.dumps(raw))
    raw = program_receipt(validation=ProgramValidation.REJECTED, applied_tick=104).to_dict()
    raw["emitted_motors"] = MotorVector(forward_thrust=0.2).to_dict()
    with pytest.raises(ProtocolError, match="requires neutral emitted motors"):
        parse_message(json.dumps(raw))


def test_program_receipt_confines_a_successful_or_refused_tick_to_the_frame_window() -> None:
    """A rejection keeps its real validation tick; an executed tick cannot claim one outside."""

    frame = program_action()
    for validation in (ProgramValidation.PROGRAM_APPLIED, ProgramValidation.PROGRAM_REFUSED):
        program_receipt(frame, validation=validation, applied_tick=frame.target_tick + 2)
        with pytest.raises(ProtocolError, match="within two target ticks"):
            program_receipt(frame, validation=validation, applied_tick=frame.target_tick + 3)
        with pytest.raises(ProtocolError, match="within two target ticks"):
            program_receipt(frame, validation=validation, applied_tick=frame.target_tick - 1)
    assert (
        program_receipt(
            frame, validation=ProgramValidation.REJECTED, applied_tick=frame.target_tick + 9
        ).applied_tick
        == frame.target_tick + 9
    )


def test_program_receipt_refuses_unknown_and_missing_fields() -> None:
    raw = program_receipt().to_dict()
    raw["executed_semantic_goal"] = "diamond"
    with pytest.raises(UnknownFieldError):
        parse_message(json.dumps(raw))
    raw = program_receipt().to_dict()
    del raw["emitted_motors"]
    with pytest.raises(ProtocolError, match="missing fields: emitted_motors"):
        parse_message(json.dumps(raw))


def test_program_frames_are_rejected_under_the_wrong_message_type() -> None:
    """The header check is per-type, so a mislabeled frame fails closed on both sides."""

    raw = program_action().to_dict()
    raw["type"] = "action"
    with pytest.raises(ProtocolError):
        parse_message(json.dumps(raw))
    raw = program_receipt().to_dict()
    raw["protocol"] = "neurocraft.protocol.v3"
    with pytest.raises(ProtocolError, match=r"protocol must be neurocraft\.protocol\.v2"):
        parse_message(json.dumps(raw))


def test_program_frames_are_frozen_and_immutable() -> None:
    frame = program_action()
    with pytest.raises(AttributeError):
        frame.program = ProgramId.HOVER  # type: ignore[misc]
    assert replace(frame, program=ProgramId.ESCAPE_JUMP).program is ProgramId.ESCAPE_JUMP
    assert frame.program is ProgramId.CRUISE


def test_program_receipt_parses_from_its_own_dict_for_every_validation_state() -> None:
    for validation in ProgramValidation:
        applied_tick = 104 if validation is ProgramValidation.REJECTED else None
        original = program_receipt(validation=validation, applied_tick=applied_tick)
        assert ProgramReceipt.from_dict(original.to_dict()) == original
