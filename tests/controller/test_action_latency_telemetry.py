"""The latency sink writes durations only, and never faults the action path."""

from __future__ import annotations

import json
from pathlib import Path

from neurocraft_service import ActionTelemetry

from scripts.action_latency_telemetry import ACTION_LATENCY_SCHEMA, ActionLatencyWriter

#: Every key an emitted sample is allowed to carry. Widening this set is a
#: boundary change, not a formatting change: the full-artifact profiles accept
#: this writer precisely because it copies no controller evidence.
ALLOWED_KEYS = {
    "schema",
    "run_id",
    "controller_profile",
    "command_seq",
    "compute_ns",
    "serialization_ns",
    "write_ns",
}


def _telemetry(command_seq: int) -> ActionTelemetry:
    return ActionTelemetry(
        command_seq=command_seq,
        action_sha256="a" * 64,
        queue_coalesced=3,
        queue_overruns=1,
        compute_ns=24_400_000,
        serialization_ns=120_000,
        write_ns=45_000,
    )


def _writer(tmp_path: Path) -> tuple[ActionLatencyWriter, Path]:
    path = tmp_path / "action-latency.ndjson"
    writer = ActionLatencyWriter(path, run_id="run-1", controller_profile="full_mcns_wing_v1")
    return writer, path


def test_writer_appends_one_sample_per_action(tmp_path: Path) -> None:
    writer, path = _writer(tmp_path)

    writer.accept(_telemetry(1))
    writer.accept(_telemetry(2))

    lines = path.read_bytes().splitlines()
    assert len(lines) == 2
    sample = json.loads(lines[0])
    assert sample["schema"] == ACTION_LATENCY_SCHEMA
    assert sample["run_id"] == "run-1"
    assert sample["controller_profile"] == "full_mcns_wing_v1"
    assert sample["command_seq"] == 1
    assert sample["compute_ns"] == 24_400_000
    assert sample["serialization_ns"] == 120_000
    assert sample["write_ns"] == 45_000
    assert json.loads(lines[1])["command_seq"] == 2


def test_samples_carry_durations_only(tmp_path: Path) -> None:
    """The boundary this writer relies on: no controller evidence is copied,
    and no per-action identity is bound, so nothing it emits can be read as
    connectome activity."""

    writer, path = _writer(tmp_path)
    writer.accept(_telemetry(1))

    sample = json.loads(path.read_bytes().splitlines()[0])
    assert set(sample) == ALLOWED_KEYS
    encoded = path.read_bytes()
    assert b"a" * 64 not in encoded  # the action hash is never written
    assert b"queue" not in encoded
    assert b"activity" not in encoded


def test_writer_truncates_stale_samples_from_a_previous_run(tmp_path: Path) -> None:
    path = tmp_path / "action-latency.ndjson"
    path.write_text('{"stale": true}\n', encoding="utf-8")
    ActionLatencyWriter(path, run_id="run-1", controller_profile="pair_v1")
    assert path.read_bytes() == b""


def test_writer_failures_self_disable_and_never_propagate_to_the_action_path(
    tmp_path: Path,
) -> None:
    """The service treats sink exceptions as terminal trace faults, so a broken
    diagnostic writer must swallow its own failure, not abort the session."""

    writer, path = _writer(tmp_path)

    path.unlink()
    path.mkdir()  # the append target is now unwritable
    writer.accept(_telemetry(1))  # must not raise
    assert writer.disabled

    path.rmdir()
    path.write_bytes(b"")
    writer.accept(_telemetry(2))  # disabled: stays silent even once writable again
    assert path.read_bytes() == b""
