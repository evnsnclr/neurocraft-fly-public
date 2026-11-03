from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from neurocraft_protocol import (
    CausalRecord,
    ClaimScope,
    FaultCode,
    MotorVector,
    ProtocolError,
    TraceIntegrityError,
    TraceManifest,
    TraceWriter,
    ValidationResult,
    canonical_json_bytes,
    verify_trace,
)

from .helpers import action, causal_record, identity, manifest, receipt


def write_trace(path: Path, *, records: int = 2) -> Path:
    with TraceWriter(path, manifest()) as writer:
        for sequence in range(records):
            writer.append(causal_record(sequence=sequence, tick=100 + sequence))
    return path


def write_hash_authentic_trace(
    path: Path,
    *,
    trace_manifest: TraceManifest,
    record: CausalRecord,
) -> Path:
    """Build a valid hash chain without using TraceWriter's policy checks."""

    entries: list[bytes] = []
    previous_hash = "0" * 64

    def append_entry(kind: str, payload: dict[str, Any]) -> str:
        nonlocal previous_hash
        material = {"kind": kind, "payload": payload, "previous_hash": previous_hash}
        entry_hash = hashlib.sha256(canonical_json_bytes(material)).hexdigest()
        entries.append(canonical_json_bytes({**material, "entry_hash": entry_hash}) + b"\n")
        previous_hash = entry_hash
        return entry_hash

    manifest_hash = append_entry("manifest", trace_manifest.to_dict())
    record_hash = append_entry("record", record.to_dict())
    append_entry(
        "footer",
        {
            "record_count": 1,
            "last_record_hash": record_hash,
            "manifest_hash": manifest_hash,
            "final_record_seq": 0,
        },
    )
    path.write_bytes(b"".join(entries))
    return path


def test_trace_round_trip_and_exact_metadata(tmp_path: Path) -> None:
    path = write_trace(tmp_path / "run.ndjson")
    result = verify_trace(path)
    assert result.record_count == 2
    assert result.first_tick == 100
    assert result.last_tick == 101
    assert result.manifest.run_id == "test-run"
    assert len(result.manifest_hash) == 64
    assert len(result.final_hash) == 64
    assert len(result.file_sha256) == 64


def test_same_records_produce_byte_identical_trace(tmp_path: Path) -> None:
    first = write_trace(tmp_path / "first.ndjson").read_bytes()
    second = write_trace(tmp_path / "second.ndjson").read_bytes()
    assert first == second


def test_writer_refuses_overwrite_and_cleans_aborted_temp(tmp_path: Path) -> None:
    destination = write_trace(tmp_path / "run.ndjson", records=0)
    with pytest.raises(FileExistsError):
        TraceWriter(destination, manifest())

    aborted = tmp_path / "aborted.ndjson"
    with pytest.raises(RuntimeError), TraceWriter(aborted, manifest()):
        raise RuntimeError("test abort")
    assert not aborted.exists()
    assert list(tmp_path.glob(".aborted.ndjson.*.tmp")) == []


def test_writer_enforces_record_order_tick_and_artifact_identity(tmp_path: Path) -> None:
    writer = TraceWriter(tmp_path / "bad.ndjson", manifest())
    with pytest.raises(ProtocolError, match="record sequence"):
        writer.append(causal_record(sequence=1, tick=101))
    wrong = causal_record()
    wrong_identity = identity(graph_hash="d" * 64)
    wrong_observation = replace(wrong.observation, identity=wrong_identity)
    wrong_action = replace(wrong.issued_action, identity=wrong_identity)
    wrong = replace(
        wrong,
        identity=wrong_identity,
        observation=wrong_observation,
        issued_action=wrong_action,
    )
    with pytest.raises(ProtocolError, match="artifact identity"):
        writer.append(wrong)
    writer.abort()


def test_record_mappings_are_immutable_snapshots() -> None:
    source = {"DNg07_L": 0.2}
    record = replace(causal_record(), motor_pool_values=source)
    source["DNg07_L"] = 0.9
    assert record.motor_pool_values["DNg07_L"] == 0.2
    with pytest.raises(TypeError):
        record.motor_pool_values["DNg07_L"] = 0.5  # type: ignore[index]


def test_trace_rejects_prohibited_secret_key_and_absolute_user_path(tmp_path: Path) -> None:
    writer = TraceWriter(tmp_path / "secret.ndjson", manifest())
    with pytest.raises(ProtocolError, match="prohibited"):
        writer.append(replace(causal_record(), encoded_stimulation={"token": 0.1}))
    writer.abort()

    bad_manifest = replace(manifest(), machine="/Users/example/private-host")
    with pytest.raises(ProtocolError, match="absolute user path"):
        TraceWriter(tmp_path / "path.ndjson", bad_manifest)


def test_fault_or_rejected_record_must_apply_neutral_action() -> None:
    with pytest.raises(ProtocolError, match="neutral action"):
        replace(
            causal_record(),
            validation=ValidationResult.NEUTRAL_FAULT,
            neutral_reason="disconnect",
            applied_action=MotorVector(forward_thrust=0.1),
        )


def test_record_validation_semantics_mirror_authoritative_receipt() -> None:
    record = causal_record()
    bound = replace(record, actuation_receipt=receipt(record.issued_action))
    assert CausalRecord.from_dict(bound.to_dict()) == bound
    swapped = receipt(action(command_sequence=9, observation_sequence=9))
    with pytest.raises(ProtocolError, match="does not bind action/world"):
        replace(record, actuation_receipt=swapped)
    with pytest.raises(ProtocolError, match="tick does not match observation age"):
        replace(
            bound,
            actuation_receipt=replace(
                bound.actuation_receipt,
                applied_tick=bound.actuation_receipt.applied_tick + 1,
            ),
        )
    with pytest.raises(ProtocolError, match="command sequence"):
        replace(record, command_seq=None)
    with pytest.raises(ProtocolError, match="neutral or fault"):
        replace(record, neutral_reason="scripted_neutral")
    with pytest.raises(ProtocolError, match="present together"):
        replace(record, direct_stimulation={"77185": 1.0})
    with pytest.raises(ProtocolError, match="equal decoder output"):
        replace(record, applied_action=MotorVector(forward_thrust=0.2, yaw_torque=0.02))
    with pytest.raises(ProtocolError, match="undeclared motor axis"):
        replace(
            record,
            clamp_fields=("forward_thrust",),
            applied_action=MotorVector(forward_thrust=0.2, yaw_torque=0.03),
        )
    with pytest.raises(ProtocolError, match="reason and fault"):
        replace(
            record,
            validation=ValidationResult.REJECTED,
            applied_action=MotorVector.neutral(),
            neutral_reason="minecraft_rejected",
        )
    with pytest.raises(ProtocolError, match="cannot carry clamp"):
        replace(
            record,
            validation=ValidationResult.NEUTRAL_FAULT,
            applied_action=MotorVector.neutral(),
            neutral_reason="minecraft_neutral_fault",
            fault_code=FaultCode.STALE,
            clamp_fields=("forward_thrust",),
        )


def test_world_after_lineage_is_required_and_causally_ordered() -> None:
    record = causal_record()
    raw = record.to_dict()
    assert raw["world_after_observation_seq"] == record.observation.observation_seq + 1
    assert raw["world_after_game_tick"] == (record.game_tick + record.observation_age_ticks + 1)
    assert CausalRecord.from_dict(raw) == record

    missing = dict(raw)
    del missing["world_after_observation_seq"]
    with pytest.raises(ProtocolError, match="missing fields"):
        CausalRecord.from_dict(missing)
    with pytest.raises(ProtocolError, match="sequence must move forward"):
        replace(record, world_after_observation_seq=record.observation.observation_seq)
    with pytest.raises(ProtocolError, match="next authoritative tick"):
        replace(record, world_after_game_tick=record.game_tick)
    with pytest.raises(ProtocolError, match="next authoritative tick"):
        replace(record, world_after_game_tick=record.world_after_game_tick + 1)


def test_small_fixture_trace_requires_full_neural_state(tmp_path: Path) -> None:
    writer = TraceWriter(tmp_path / "missing-state.ndjson", manifest())
    with pytest.raises(ProtocolError, match="require full neural state"):
        writer.append(
            replace(
                causal_record(),
                state_before_activity=None,
                state_after_activity=None,
            )
        )
    writer.abort()


@pytest.mark.parametrize(
    "claim_scope",
    (ClaimScope.G3, ClaimScope.G4, ClaimScope.G5, ClaimScope.G6),
)
def test_receipt_and_measured_telemetry_policy_is_cumulative(
    tmp_path: Path,
    claim_scope: ClaimScope,
) -> None:
    trace_manifest = replace(manifest(), claim_scope=claim_scope)
    writer = TraceWriter(tmp_path / f"missing-receipt-{claim_scope}.ndjson", trace_manifest)
    with pytest.raises(ProtocolError, match="canonical actuation receipt"):
        writer.append(causal_record())
    writer.abort()

    record = causal_record()
    bound = replace(
        record,
        actuation_receipt=receipt(record.issued_action),
        compute_ns=0,
    )
    writer = TraceWriter(tmp_path / f"zero-timing-{claim_scope}.ndjson", trace_manifest)
    with pytest.raises(ProtocolError, match="positive measured operational timings"):
        writer.append(bound)
    writer.abort()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_receipt", "canonical actuation receipt"),
        ("zero_timing", "positive measured operational timings"),
        ("missing_state", "require full neural state"),
    ),
)
def test_verifier_reapplies_manifest_record_policy_to_hash_authentic_trace(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    trace_manifest = replace(manifest(), claim_scope=ClaimScope.G3)
    record = causal_record()
    bound = replace(record, actuation_receipt=receipt(record.issued_action))
    if mutation == "missing_receipt":
        mutated = replace(bound, actuation_receipt=None)
    elif mutation == "zero_timing":
        mutated = replace(bound, serialization_ns=0)
    elif mutation == "missing_state":
        mutated = replace(
            bound,
            state_before_activity=None,
            state_after_activity=None,
        )
    else:
        raise AssertionError(mutation)
    trace = write_hash_authentic_trace(
        tmp_path / f"hash-authentic-{mutation}.ndjson",
        trace_manifest=trace_manifest,
        record=mutated,
    )

    with pytest.raises(TraceIntegrityError, match=message):
        verify_trace(trace)


def mutate_lines(source: Path, destination: Path, mutation: str) -> None:
    lines = source.read_bytes().splitlines(keepends=True)
    if mutation == "payload":
        lines[1] = lines[1].replace(b"0.15", b"0.16", 1)
    elif mutation == "reorder":
        lines[1], lines[2] = lines[2], lines[1]
    elif mutation == "delete":
        del lines[1]
    elif mutation == "duplicate":
        lines.insert(2, lines[1])
    elif mutation == "footer":
        lines.pop()
    elif mutation == "append":
        lines.append(lines[1])
    else:
        raise AssertionError(mutation)
    destination.write_bytes(b"".join(lines))


@pytest.mark.parametrize(
    "mutation", ["payload", "reorder", "delete", "duplicate", "footer", "append"]
)
def test_tamper_reorder_delete_duplicate_and_append_are_detected(
    tmp_path: Path, mutation: str
) -> None:
    source = write_trace(tmp_path / "source.ndjson")
    damaged = tmp_path / f"{mutation}.ndjson"
    mutate_lines(source, damaged, mutation)
    with pytest.raises(TraceIntegrityError):
        verify_trace(damaged)


def test_noncanonical_json_is_rejected_even_when_values_are_unchanged(tmp_path: Path) -> None:
    source = write_trace(tmp_path / "source.ndjson", records=0)
    lines = source.read_text().splitlines()
    lines[0] = lines[0].replace(":", ": ", 1)
    changed = tmp_path / "noncanonical.ndjson"
    changed.write_text("\n".join(lines) + "\n")
    with pytest.raises(TraceIntegrityError, match="canonical"):
        verify_trace(changed)
