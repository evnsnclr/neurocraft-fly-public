"""The presentation telemetry writer copies evidence; it never fabricates it."""

from __future__ import annotations

import json
from pathlib import Path

from neurocraft_connectome import RuntimeGraph
from neurocraft_service import ActionTelemetry

from scripts.run_connectome_service import PresentationTelemetryWriter

from .conftest import make_controller, make_observation


def _telemetry(command_seq: int) -> ActionTelemetry:
    return ActionTelemetry(
        command_seq=command_seq,
        action_sha256="0" * 64,
        queue_coalesced=0,
        queue_overruns=0,
        compute_ns=1,
        serialization_ns=1,
        write_ns=1,
    )


def test_writer_appends_a_sample_bound_to_the_last_evidence(
    controller_graph: RuntimeGraph, tmp_path: Path
) -> None:
    controller = make_controller(controller_graph)
    path = tmp_path / "presentation-telemetry.ndjson"
    writer = PresentationTelemetryWriter(path, controller, "pair_v1")
    controller.step(make_observation(controller, 1), command_seq=1)
    writer.accept(_telemetry(1))

    lines = path.read_bytes().splitlines()
    assert len(lines) == 1
    sample = json.loads(lines[0])
    evidence = controller.last_evidence
    assert evidence is not None
    assert sample["presentation_only"] is True
    assert sample["controller_profile"] == "pair_v1"
    assert sample["run_id"] == controller.identity.run_id
    assert sample["command_seq"] == 1
    assert sample["game_tick"] == evidence.observation.game_tick
    assert len(sample["activity"]) == len(controller_graph.nodes)
    assert set(sample["motor_frame"]) == {name for name, _ in evidence.motor_frame.values}
    assert sample["encoded"]
    assert all(":" in name for name in sample["encoded"])


def test_writer_skips_missing_or_mismatched_evidence(
    controller_graph: RuntimeGraph, tmp_path: Path
) -> None:
    controller = make_controller(controller_graph)
    path = tmp_path / "presentation-telemetry.ndjson"
    writer = PresentationTelemetryWriter(path, controller, "pair_v1")

    writer.accept(_telemetry(1))  # no step has produced evidence yet
    assert path.read_bytes() == b""

    controller.step(make_observation(controller, 1), command_seq=1)
    writer.accept(_telemetry(2))  # stale sequence must not borrow newer evidence
    assert path.read_bytes() == b""


def test_writer_truncates_stale_samples_from_a_previous_run(
    controller_graph: RuntimeGraph, tmp_path: Path
) -> None:
    controller = make_controller(controller_graph)
    path = tmp_path / "presentation-telemetry.ndjson"
    path.write_bytes(b'{"stale": true}\n')
    PresentationTelemetryWriter(path, controller, "pair_v1")
    assert path.read_bytes() == b""


def test_writer_failures_self_disable_and_never_propagate_to_the_action_path(
    controller_graph: RuntimeGraph, tmp_path: Path
) -> None:
    """The service treats sink exceptions as terminal trace faults; a broken
    presentation writer must swallow its own failure, not abort the session."""

    controller = make_controller(controller_graph)
    path = tmp_path / "presentation-telemetry.ndjson"
    writer = PresentationTelemetryWriter(path, controller, "pair_v1")
    controller.step(make_observation(controller, 1), command_seq=1)

    path.unlink()
    path.mkdir()  # the append target is now unwritable
    writer.accept(_telemetry(1))  # must not raise

    path.rmdir()
    path.write_bytes(b"")
    writer.accept(_telemetry(1))  # disabled: stays silent even once writable again
    assert path.read_bytes() == b""
