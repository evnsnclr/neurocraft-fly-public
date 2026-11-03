"""The program-decision sink records selections and receipts, and never faults the loop.

This writer sits on the LIVE path: the service hands it every issued command through
``issued_action_sink`` and every receipt through ``receipt_sink``, and it treats a sink
exception as a terminal trace fault. So the two things that matter are that it writes what
it claims to write, and that a broken writer disables itself rather than killing a session.

Everything it emits is diagnostic: not content-addressed, not hash-linked, not evidence.
The tests pin that too -- there is no action hash, no neural activity, and no trace binding
in an emitted line.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from neurocraft_controller.interaction_program_controller import (
    InteractionProgramSelectingController,
)
from neurocraft_protocol import (
    Action,
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
    WorldFrame,
)

from scripts.probe_interaction_inputs import sensor_frame
from scripts.program_decision_telemetry import (
    PROGRAM_DECISION_SCHEMA,
    ProgramDecisionWriter,
)
from scripts.run_release_comparisons import input_matches
from tests.controller.test_interaction_senses import paired_controllers

IDENTITY = RunIdentity(
    run_id="run-1",
    session_epoch=1,
    entity_id="fly-01",
    graph_hash="11" * 32,
    adapter_hash="22" * 32,
    dynamics_hash="33" * 32,
    decoder_hash="44" * 32,
    observation_schema_hash="55" * 32,
)
POLICY_HASH = "77" * 32

#: Every key an ``issued`` line may carry. Widening it is a boundary change: this writer is
#: permitted on the live path precisely because it copies no neural activity.
ISSUED_KEYS = {
    "schema",
    "diagnostic_only",
    "run_id",
    "controller_profile",
    "kind",
    "command_seq",
    "observation_seq",
    "observation_tick",
    "target_tick",
    "state_digest",
    "neural_step_start",
    "neural_step_end",
    "program",
    "modulation",
    "parameters",
    "readout",
    "policy_hash",
    "policy_state_bound",
    "policy_state_before",
    "policy_state_after",
    "driving_readout",
}
RECEIPT_KEYS = {
    "schema",
    "diagnostic_only",
    "run_id",
    "controller_profile",
    "kind",
    "command_seq",
    "observation_seq",
    "target_tick",
    "applied_tick",
    "requested_program",
    "executed_program",
    "validation",
    "reason",
    "fault_code",
}


def _frame(command_seq: int = 0, program: ProgramId = ProgramId.CRUISE) -> ProgramAction:
    return ProgramAction(
        identity=IDENTITY,
        command_seq=command_seq,
        observation_seq=command_seq,
        observation_tick=100 + command_seq,
        target_tick=101 + command_seq,
        valid_through_tick=102 + command_seq,
        neural_step_start=5 * command_seq,
        neural_step_end=5 * command_seq + 5,
        state_digest="66" * 32,
        program=program,
        modulation=ProgramModulation(speed=0.7, turn_rate=1.0, amplitude=1.0, urgency=0.0),
        parameters=ProgramParameters(turn_bias=-0.25),
        readout=ReadoutValues(ttmn_jump=0.0, dna01_turn=-0.25, wing_common=0.61),
        policy_hash=POLICY_HASH,
    )


def _receipt(
    *,
    command_seq: int = 0,
    validation: ProgramValidation = ProgramValidation.PROGRAM_APPLIED,
    executed: ProgramId | None = ProgramId.CRUISE,
    reason: ProgramRefusalReason | None = None,
) -> ProgramReceipt:
    modulation = ProgramModulation(speed=0.7, turn_rate=1.0, amplitude=1.0, urgency=0.0)
    parameters = ProgramParameters(turn_bias=-0.25)
    return ProgramReceipt(
        identity=IDENTITY,
        command_seq=command_seq,
        observation_seq=command_seq,
        target_tick=101 + command_seq,
        applied_tick=101 + command_seq,
        world_before=WorldFrame(
            position_m=(0.0, 66.0, 0.0),
            velocity_m_s=(0.0, 0.0, 0.0),
            yaw_rad=0.0,
            pitch_rad=0.0,
            on_ground=False,
            horizontal_collision=False,
            vertical_collision=False,
            in_fluid=False,
        ),
        requested_program=ProgramId.CRUISE,
        requested_modulation=modulation,
        requested_parameters=parameters,
        executed_program=executed,
        executed_modulation=None if executed is None else modulation,
        executed_parameters=None if executed is None else parameters,
        emitted_motors=MotorVector.neutral(),
        validation=validation,
        reason=reason,
    )


class _FakeDecision:
    def __init__(self) -> None:
        self.state_before = _FakeState("CRUISE", armed=True)
        self.state_after = _FakeState("ESCAPE_JUMP", armed=False)
        self.driving_readout = "wing"


class _FakeState:
    def __init__(self, program: str, *, armed: bool) -> None:
        self._program = program
        self._armed = armed

    def to_dict(self) -> dict[str, Any]:
        return {
            "armed": self._armed,
            "hold_remaining": 0,
            "program": self._program,
            "refractory_remaining": 0,
        }


class _FakeEvidence:
    def __init__(self, command_seq: int) -> None:
        self.command_seq = command_seq
        self.decision = _FakeDecision()


class _FakeController:
    """Only what the writer reads: an identity, a policy hash, and last evidence."""

    def __init__(self, evidence: _FakeEvidence | None = None) -> None:
        self.identity = IDENTITY
        self.policy_hash = POLICY_HASH
        self.last_evidence = evidence


def _writer(tmp_path: Path, controller: Any) -> tuple[ProgramDecisionWriter, Path]:
    path = tmp_path / "program-decisions.ndjson"
    writer = ProgramDecisionWriter(path, controller, controller_profile="full_mcns_loom_program_v1")
    return writer, path


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def test_issued_line_carries_the_frame_and_the_bound_policy_state(tmp_path: Path) -> None:
    controller = _FakeController(_FakeEvidence(command_seq=0))
    writer, path = _writer(tmp_path, controller)

    writer.accept_issued_command(_frame())

    (sample,) = _lines(path)
    assert set(sample) == ISSUED_KEYS
    assert sample["schema"] == PROGRAM_DECISION_SCHEMA == "neurocraft.program_decision.v1"
    assert sample["diagnostic_only"] is True
    assert sample["kind"] == "issued"
    assert sample["run_id"] == "run-1"
    assert sample["controller_profile"] == "full_mcns_loom_program_v1"
    assert sample["program"] == "CRUISE"
    assert sample["policy_hash"] == POLICY_HASH
    assert sample["readout"] == {"dna01_turn": -0.25, "ttmn_jump": 0.0, "wing_common": 0.61}
    assert sample["parameters"] == {"turn_bias": -0.25}
    assert sample["policy_state_bound"] is True
    assert sample["policy_state_before"]["program"] == "CRUISE"
    assert sample["policy_state_after"]["program"] == "ESCAPE_JUMP"
    assert sample["driving_readout"] == "wing"


def test_unbound_evidence_writes_null_state_rather_than_guessing(tmp_path: Path) -> None:
    """Coalescing can replace the controller's evidence before the sink runs.

    The frame's own fields are still authoritative and are written; the policy state is not
    reconstructed from a different tick, it is written as null and flagged unbound.
    """

    for evidence in (None, _FakeEvidence(command_seq=9)):
        controller = _FakeController(evidence)
        writer, path = _writer(tmp_path, controller)
        writer.accept_issued_command(_frame(command_seq=0))
        (sample,) = _lines(path)
        assert sample["policy_state_bound"] is False
        assert sample["policy_state_before"] is None
        assert sample["policy_state_after"] is None
        assert sample["driving_readout"] is None
        assert sample["program"] == "CRUISE"  # the frame itself is still recorded


def test_receipt_line_records_what_the_authority_reported(tmp_path: Path) -> None:
    writer, path = _writer(tmp_path, _FakeController())

    writer.accept_receipt(_receipt())

    (sample,) = _lines(path)
    assert set(sample) == RECEIPT_KEYS
    assert sample["kind"] == "receipt"
    assert sample["validation"] == "PROGRAM_APPLIED"
    assert sample["executed_program"] == "CRUISE"
    assert sample["reason"] is None
    assert sample["fault_code"] is None


def test_an_operator_refusal_is_recorded_as_received_and_never_normalized(
    tmp_path: Path,
) -> None:
    """``operator`` legitimately reports a program other than HOVER.

    An operator hold ran its own program, so forcing HOVER here would make the record
    assert a program ran that did not. The writer records what arrived.
    """

    writer, path = _writer(tmp_path, _FakeController())

    writer.accept_receipt(
        _receipt(
            validation=ProgramValidation.PROGRAM_REFUSED,
            executed=ProgramId.WALK,
            reason=ProgramRefusalReason.OPERATOR,
        )
    )

    (sample,) = _lines(path)
    assert sample["validation"] == "PROGRAM_REFUSED"
    assert sample["executed_program"] == "WALK"
    assert sample["reason"] == "operator"


def test_raw_actions_and_actuation_receipts_are_ignored(tmp_path: Path) -> None:
    """A paused, replaying, or faulted service emits neutral ``action`` frames.

    They carry no selection, so there is nothing to record and nothing is written.
    """

    writer, path = _writer(tmp_path, _FakeController())

    writer.accept_issued_command(
        Action(
            identity=IDENTITY,
            command_seq=0,
            observation_seq=0,
            observation_tick=100,
            target_tick=101,
            valid_through_tick=102,
            neural_step_start=0,
            neural_step_end=0,
            state_digest="66" * 32,
            motors=MotorVector.neutral(),
            neutral=True,
        )
    )

    assert path.read_bytes() == b""


def test_lines_carry_no_neural_activity_or_action_hash(tmp_path: Path) -> None:
    controller = _FakeController(_FakeEvidence(command_seq=0))
    writer, path = _writer(tmp_path, controller)
    writer.accept_issued_command(_frame())
    writer.accept_receipt(_receipt())

    encoded = path.read_bytes()
    assert b"activity" not in encoded
    assert b"action_sha256" not in encoded
    assert b"state_after_activity" not in encoded
    # Each line declares itself diagnostic, so a reader cannot mistake the file for a trace.
    assert all(sample["diagnostic_only"] is True for sample in _lines(path))


def test_writer_truncates_stale_samples_from_a_previous_run(tmp_path: Path) -> None:
    path = tmp_path / "program-decisions.ndjson"
    path.write_text('{"stale": true}\n', encoding="utf-8")
    ProgramDecisionWriter(path, _FakeController(), controller_profile="full_mcns_loom_program_v1")
    assert path.read_bytes() == b""


def test_writer_failures_self_disable_and_never_propagate_to_the_command_path(
    tmp_path: Path,
) -> None:
    """The service treats sink exceptions as terminal trace faults, so a broken
    diagnostic writer must swallow its own failure rather than kill the session."""

    writer, path = _writer(tmp_path, _FakeController())

    path.unlink()
    path.mkdir()  # the append target is now unwritable
    writer.accept_issued_command(_frame())  # must not raise
    assert writer.disabled
    writer.accept_receipt(_receipt())  # must not raise either

    path.rmdir()
    path.write_bytes(b"")
    writer.accept_issued_command(_frame(command_seq=1))
    assert path.read_bytes() == b""  # disabled: silent even once writable again


def test_writer_reports_the_policy_hash_it_was_built_over(tmp_path: Path) -> None:
    writer, _ = _writer(tmp_path, _FakeController())
    assert writer.policy_hash == POLICY_HASH
    assert writer.disabled is False


INPUT_KEYS = {
    "schema",
    "diagnostic_only",
    "identity",
    "command_seq",
    "observation_seq",
    "observation_tick",
    "state_digest",
    "bound",
    "encoded_stimulation",
    "before_control",
    "stimulation_sha256",
    "scope",
}
INPUT_VALUES = ("encoded_stimulation", "before_control", "stimulation_sha256")


def _interaction_step(mode: str = "faithful") -> tuple[Any, ProgramAction]:
    """Exercise the real input/policy wrapper on an existing small NumPy test graph."""

    core, _, _ = paired_controllers(mode)
    controller = InteractionProgramSelectingController(core)
    for tick in range(21):
        observation = Observation(
            core.identity, tick, tick, tick * 50_000_000, sensor_frame("brush", tick)
        )
        command = controller.step(observation, command_seq=tick)
    return controller, command


def _input_writer(tmp_path: Path, controller: Any) -> tuple[ProgramDecisionWriter, Path, Path]:
    path = tmp_path / "program-decisions.ndjson"
    writer = ProgramDecisionWriter(
        path, controller, controller_profile="full_mcns_interactions_program_v2"
    )
    return writer, path, path.with_suffix(".inputs.ndjson")


def test_bound_real_input_evidence_uses_a_separate_versioned_sidecar(tmp_path: Path) -> None:
    controller, command = _interaction_step()
    inner = controller.last_evidence.inner
    writer, path, inputs_path = _input_writer(tmp_path, controller)
    writer.accept_issued_command(command)
    writer.accept_receipt(_receipt())

    issued, receipt = _lines(path)
    assert set(issued) == ISSUED_KEYS
    assert set(receipt) == RECEIPT_KEYS
    assert issued["schema"] == receipt["schema"] == "neurocraft.program_decision.v1"
    (sample,) = _lines(inputs_path)
    assert set(sample) == INPUT_KEYS
    assert sample["schema"] == "neurocraft.program_input_diagnostic.v1"
    assert sample["diagnostic_only"] is True
    assert sample["identity"] == command.identity.to_dict()
    assert sample["bound"] is True
    assert sample["command_seq"] == command.command_seq
    assert sample["observation_seq"] == inner.observation.observation_seq
    assert sample["observation_tick"] == inner.observation.game_tick
    assert sample["state_digest"] == inner.state_after_sha256
    assert sample["encoded_stimulation"] == dict(inner.encoded_stimulation)
    assert any(value != 0 for value in sample["encoded_stimulation"].values())
    assert sample["before_control"] is None
    assert sample["stimulation_sha256"] == inner.encoded_stimulation_sha256
    assert "not a dense sensory vector" in sample["scope"]


def test_no_input_retains_nonzero_requested_drive_and_delivered_zeros(tmp_path: Path) -> None:
    controller, command = _interaction_step("no_input")
    inner = controller.last_evidence.inner
    writer, _, inputs_path = _input_writer(tmp_path, controller)
    writer.accept_issued_command(command)

    (sample,) = _lines(inputs_path)
    assert sample["bound"] is True
    assert sample["before_control"] == dict(inner.encoder_stimulation_before_control)
    assert any(value != 0 for value in sample["before_control"].values())
    assert sample["encoded_stimulation"] == dict(inner.encoded_stimulation)
    assert sample["encoded_stimulation"]
    assert all(value == 0 for value in sample["encoded_stimulation"].values())
    assert sample["stimulation_sha256"] == inner.encoded_stimulation_sha256


def test_release_comparison_accepts_real_no_input_sidecar_with_summary_fields(
    tmp_path: Path,
) -> None:
    controller, command = _interaction_step("no_input")
    writer, main_path, inputs_path = _input_writer(tmp_path, controller)
    writer.accept_issued_command(command)
    (request,) = _lines(main_path)
    (sample,) = _lines(inputs_path)

    before, delivered = sample["before_control"], sample["encoded_stimulation"]
    extra_summary_keys = set(before) - set(delivered)
    assert {key.split(":", 1)[0] for key in extra_summary_keys} == {"eem_raw", "eem_mean"}
    assert any(value != 0 for value in before.values())
    assert delivered and all(value == 0 for value in delivered.values())
    assert input_matches(sample, request, command.identity.to_dict(), "no_input") is True


@pytest.mark.parametrize("replacement", ["missing", "later_command"])
def test_missing_or_stale_input_evidence_is_unavailable(tmp_path: Path, replacement: str) -> None:
    controller, command = _interaction_step()
    evidence = controller.last_evidence
    if replacement == "missing":
        evidence = None
    else:
        evidence = replace(evidence, inner=replace(evidence.inner, command_seq=999))
    mocked = SimpleNamespace(
        identity=controller.identity, policy_hash=controller.policy_hash, last_evidence=evidence
    )
    writer, main_path, inputs_path = _input_writer(tmp_path, mocked)
    writer.accept_issued_command(command)

    (sample,) = _lines(inputs_path)
    assert sample["bound"] is False
    assert all(sample[key] is None for key in INPUT_VALUES)
    assert _lines(main_path)[0]["policy_state_bound"] is False
    assert writer.disabled is False


@pytest.mark.parametrize(
    "mismatch",
    ["observation_seq", "game_tick", "state_digest", "neural_step_start", "neural_step_end"],
)
def test_input_sidecar_refuses_wrong_observation_state_or_interval(
    tmp_path: Path, mismatch: str
) -> None:
    controller, command = _interaction_step()
    evidence = controller.last_evidence
    inner = evidence.inner
    if mismatch == "state_digest":
        inner = replace(inner, state_after_sha256="ff" * 32)
    elif mismatch in ("neural_step_start", "neural_step_end"):
        inner = replace(inner, **{mismatch: getattr(inner, mismatch) + 1})
    else:
        inner = replace(inner, observation=replace(inner.observation, **{mismatch: 999}))
    mocked = SimpleNamespace(
        identity=controller.identity,
        policy_hash=controller.policy_hash,
        last_evidence=replace(evidence, inner=inner),
    )
    writer, main_path, inputs_path = _input_writer(tmp_path, mocked)
    writer.accept_issued_command(command)

    (sample,) = _lines(inputs_path)
    assert sample["bound"] is False
    assert all(sample[key] is None for key in INPUT_VALUES)
    # Sidecar binding is stronger than the preserved v1 command-sequence-only policy binding.
    assert _lines(main_path)[0]["policy_state_bound"] is True


@pytest.mark.parametrize("changed_side", ["command", "observation"])
@pytest.mark.parametrize(
    "identity_field",
    [
        "run_id",
        "session_epoch",
        "entity_id",
        "graph_hash",
        "adapter_hash",
        "dynamics_hash",
        "decoder_hash",
        "observation_schema_hash",
    ],
)
def test_input_binding_requires_the_full_command_and_observation_identity(
    tmp_path: Path, changed_side: str, identity_field: str
) -> None:
    controller, original_command = _interaction_step()
    evidence = controller.last_evidence
    original_identity = original_command.identity
    value = getattr(original_identity, identity_field)
    replacement = (
        value + 1
        if identity_field == "session_epoch"
        else value + "-other"
        if identity_field in ("run_id", "entity_id")
        else "fe" * 32
    )
    changed_identity = replace(original_identity, **{identity_field: replacement})
    assert changed_identity != original_identity
    command = original_command
    if changed_side == "command":
        command = replace(command, identity=changed_identity)
    else:
        evidence = replace(
            evidence,
            inner=replace(
                evidence.inner,
                observation=replace(evidence.inner.observation, identity=changed_identity),
            ),
        )
    mocked = SimpleNamespace(
        identity=controller.identity,
        policy_hash=controller.policy_hash,
        last_evidence=evidence,
    )
    writer, main_path, inputs_path = _input_writer(tmp_path, mocked)
    writer.accept_issued_command(command)

    (sample,) = _lines(inputs_path)
    assert sample["identity"] == command.identity.to_dict()
    assert sample["bound"] is False
    assert all(sample[key] is None for key in INPUT_VALUES)
    assert _lines(main_path)[0]["policy_state_bound"] is True
    assert writer.disabled is False

    # A mismatched frame is unavailable, not a permanent writer failure.
    mocked.last_evidence = controller.last_evidence
    writer.accept_issued_command(original_command)
    assert _lines(inputs_path)[1]["bound"] is True


def test_sidecar_initialization_failure_preserves_main_writer_and_stays_disabled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    controller, command = _interaction_step()
    blocked_path = tmp_path / "program-decisions.inputs.ndjson"
    blocked_path.mkdir()

    writer, main_path, inputs_path = _input_writer(tmp_path, controller)
    assert inputs_path == blocked_path
    assert "program input diagnostics disabled" in capsys.readouterr().err
    writer.accept_issued_command(command)
    writer.accept_receipt(_receipt())
    assert writer.disabled is False
    assert [row["kind"] for row in _lines(main_path)] == ["issued", "receipt"]

    blocked_path.rmdir()
    blocked_path.write_bytes(b"")
    writer.accept_issued_command(command)
    writer.accept_receipt(_receipt())
    assert inputs_path.read_bytes() == b""
    assert [row["kind"] for row in _lines(main_path)] == ["issued", "receipt", "issued", "receipt"]
    assert writer.disabled is False
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("failure", ["oversized", "nan", "infinity", "before_control_nan"])
def test_bad_input_rows_disable_only_sidecar_and_main_receipts_continue(
    tmp_path: Path, failure: str, capsys: pytest.CaptureFixture[str]
) -> None:
    controller, command = _interaction_step()
    evidence = controller.last_evidence
    bad_values = {
        "oversized": {"x" * 32768: 1.0},
        "nan": {"drive": float("nan")},
        "infinity": {"drive": float("inf")},
        "before_control_nan": {"drive": float("nan")},
    }[failure]
    field = (
        "encoder_stimulation_before_control"
        if failure == "before_control_nan"
        else "encoded_stimulation"
    )
    mocked = SimpleNamespace(
        identity=controller.identity,
        policy_hash=controller.policy_hash,
        last_evidence=replace(evidence, inner=replace(evidence.inner, **{field: bad_values})),
    )
    writer, main_path, inputs_path = _input_writer(tmp_path, mocked)
    writer.accept_issued_command(command)
    writer.accept_receipt(_receipt())
    assert writer.disabled is False
    assert inputs_path.read_bytes() == b""
    assert "program input diagnostics disabled" in capsys.readouterr().err

    mocked.last_evidence = evidence
    writer.accept_issued_command(command)
    writer.accept_receipt(_receipt())
    assert [row["kind"] for row in _lines(main_path)] == ["issued", "receipt", "issued", "receipt"]
    assert inputs_path.read_bytes() == b""  # sidecar stays disabled after the row is repaired
    assert writer.disabled is False


def test_non_interaction_profile_never_creates_an_input_sidecar(tmp_path: Path) -> None:
    controller, command = _interaction_step()
    writer, main_path = _writer(tmp_path, controller)
    writer.accept_issued_command(command)
    writer.accept_receipt(_receipt())
    assert len(_lines(main_path)) == 2
    assert not main_path.with_suffix(".inputs.ndjson").exists()
