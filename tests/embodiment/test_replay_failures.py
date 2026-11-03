"""One test per failure branch in ``replay_embodiment_trace``.

Every guard in ``_replay`` used to be unexercised except one. World-before
mismatch, lineage mismatch, state-before mismatch and the causal-replay mismatch --
the four that carry the entire replay guarantee -- had no test at all, so the
function that pronounces on evidence was itself almost entirely unevidenced.

Each corruption below is rebuilt into a trace with a valid hash chain and footer
(``conftest.rewrite_trace``), so it reaches the replay guard rather than being
stopped one layer earlier by a trace-integrity error. That distinction is the whole
point: these tests exercise the replay guarantee, not the trace format.
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from neurocraft_connectome import RuntimeGraph, load_malecns_fixture
from neurocraft_embodiment import (
    EmbodimentError,
    EmbodimentIdentity,
    ReferenceArena,
    ReferenceEmbodiment,
    replay_embodiment_trace,
    run_embodiment,
)
from neurocraft_protocol import DeterminismLevel

from .conftest import FIXTURE, make_controller, record_reference_trace, rewrite_trace


def corrupt_and_replay(
    graph: RuntimeGraph,
    tmp_path: Path,
    mutate: Any,
    *,
    ticks: int = 4,
    mutate_manifest: Any = None,
) -> dict[str, Any]:
    source = tmp_path / "source.ndjson"
    record_reference_trace(graph, source, ticks=ticks)
    corrupted = rewrite_trace(source, tmp_path / "corrupted.ndjson", mutate, mutate_manifest)
    return replay_embodiment_trace(
        trace=corrupted,
        graph_factory=lambda: load_malecns_fixture(FIXTURE),
        embodiment_factory=lambda: ReferenceEmbodiment(ReferenceArena()),
    )


def test_replay_rejects_a_trace_with_no_records(graph: RuntimeGraph, tmp_path: Path) -> None:
    with pytest.raises(EmbodimentError, match="record count is empty or inconsistent"):
        corrupt_and_replay(graph, tmp_path, lambda index, payload: None)


def test_replay_rejects_a_trace_recorded_against_another_graph(
    graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    """The controller is rebuilt from the trace's own identity, so this is the check
    that the *graph* behind that identity is the one the trace names."""

    swapped = "f" * 64

    def mutate(index: int, payload: dict[str, Any]) -> dict[str, Any]:
        observation = {
            **payload["observation"],
            "identity": {**payload["observation"]["identity"], "graph_hash": swapped},
        }
        issued = {
            **payload["issued_action"],
            "identity": {**payload["issued_action"]["identity"], "graph_hash": swapped},
        }
        return {
            **payload,
            "identity": {**payload["identity"], "graph_hash": swapped},
            "issued_action": issued,
            "observation": observation,
        }

    def mutate_manifest(payload: dict[str, Any]) -> dict[str, Any]:
        artifacts = {**payload["artifacts"], "runtime_graph_hash": swapped}
        return {**payload, "artifacts": artifacts}

    with pytest.raises(EmbodimentError, match="controller identity does not match the trace"):
        corrupt_and_replay(graph, tmp_path, mutate, mutate_manifest=mutate_manifest)


def test_replay_rejects_a_world_before_the_observation_does_not_produce(
    graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    def mutate(index: int, payload: dict[str, Any]) -> dict[str, Any]:
        if index != 0:
            return payload
        world = dict(payload["world_before"])
        world["yaw_rad"] = world["yaw_rad"] + 0.25
        return {**payload, "world_before": world}

    # Record 0 rather than a later one: every other record's ``world_before`` is
    # also the previous record's ``world_after``, so corrupting it trips the
    # lineage cross-link first. Record 0 is the only one this branch owns alone.
    with pytest.raises(EmbodimentError, match="world-before mismatch at record 0"):
        corrupt_and_replay(graph, tmp_path, mutate)


def test_replay_rejects_a_world_after_the_next_record_contradicts(
    graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    def mutate(index: int, payload: dict[str, Any]) -> dict[str, Any]:
        if index != 0:
            return payload
        world = dict(payload["world_after"])
        world["pitch_rad"] = world["pitch_rad"] + 0.125
        return {**payload, "world_after": world}

    with pytest.raises(EmbodimentError, match="world-after lineage mismatch at record 0"):
        corrupt_and_replay(graph, tmp_path, mutate)


def test_replay_rejects_a_world_after_lineage_that_points_outside_the_trace(
    graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    def mutate(index: int, payload: dict[str, Any]) -> dict[str, Any]:
        if index != 0:
            return payload
        return {**payload, "world_after_observation_seq": 900}

    with pytest.raises(EmbodimentError, match="lineage is missing within trace at record 0"):
        corrupt_and_replay(graph, tmp_path, mutate)


def test_duplicate_observation_lineage_is_unreachable_through_a_valid_trace() -> None:
    """LOW-4, reported rather than faked: this replay branch cannot be reached.

    ``_replay`` raises "trace contains duplicate observation lineage" when two
    records share ``(observation.observation_seq, observation.game_tick)``. The trace
    format already forbids that: ``CausalRecord.__post_init__`` requires
    ``observation.game_tick == record.game_tick`` (trace.py:527) and ``verify_trace``
    rejects a record whose ``game_tick`` did not increase (trace.py:1148), so the
    keys are distinct in every trace the verifier accepts. The check is a defensive
    restatement of a format invariant, not a live guard, and no honest test can make
    it fire -- so this test records the reasoning instead of manufacturing one.
    """

    from neurocraft_protocol import trace as trace_module

    source = inspect.getsource(trace_module)
    assert "self.observation.game_tick != self.game_tick" in source
    assert "record game tick did not increase" in source


def test_replay_rejects_a_state_before_the_engine_does_not_reach(
    graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    def mutate(index: int, payload: dict[str, Any]) -> dict[str, Any]:
        if index != 1:
            return payload
        activity = list(payload["state_before_activity"])
        activity[0] = activity[0] + 0.5
        return {**payload, "state_before_activity": activity}

    with pytest.raises(EmbodimentError, match="full state-before mismatch at record 1"):
        corrupt_and_replay(graph, tmp_path, mutate)


def test_replay_rejects_a_recorded_action_the_connectome_does_not_reproduce(
    graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    def mutate(index: int, payload: dict[str, Any]) -> dict[str, Any]:
        if index != 0:
            return payload
        stimulation = dict(payload["encoded_stimulation"])
        first = sorted(stimulation)[0]
        stimulation[first] = stimulation[first] + 0.25
        return {**payload, "encoded_stimulation": stimulation}

    with pytest.raises(EmbodimentError, match="causal replay mismatch at record 0"):
        corrupt_and_replay(graph, tmp_path, mutate)


class _ToleranceBody(ReferenceEmbodiment):
    """Declares the Tier-2 determinism level v1 has no replay tolerance for."""

    @property
    def identity(self) -> EmbodimentIdentity:
        return replace(
            super().identity,
            determinism_level=DeterminismLevel.NUMERICAL_TOLERANCE,
        )


def test_replay_refuses_a_body_that_does_not_declare_same_build_exact(
    graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    """No ``neural_exact_body_within_tolerance``: v1 defines no such tolerance.

    Inventing that status now would put a string nobody can produce evidence for
    into the one function that pronounces on evidence. ``FlyGymBodyConfig`` already
    refuses ``enable_vision`` for exactly this reason.
    """

    output = tmp_path / "tolerance.ndjson"
    run_embodiment(
        controller=make_controller(graph),
        graph=graph,
        embodiment=_ToleranceBody(ReferenceArena()),
        output=output,
        ticks=3,
    )
    with pytest.raises(EmbodimentError, match="requires a same_build_exact body"):
        replay_embodiment_trace(
            trace=output,
            graph_factory=lambda: load_malecns_fixture(FIXTURE),
            embodiment_factory=lambda: _ToleranceBody(ReferenceArena()),
        )


def test_replay_reconstructs_a_non_default_controller_configuration(
    graph: RuntimeGraph, tmp_path: Path
) -> None:
    """A trace recorded under a non-default configuration replays only under it.

    ``_replay`` used to hard-code the default ``ControllerConfiguration``, so any
    trace recorded under another configuration could never be replayed at all: the
    first real FlyGym execution (steering-fixture motor bodies) exposed this. The
    optional ``configuration`` parameter reconstructs the recorded controller, and
    the manifest hash comparison still refuses a configuration that does not match
    the trace, so the parameter widens what can be replayed and not what passes.
    """

    from neurocraft_controller import ConnectomeController, ControllerConfiguration

    configuration = ControllerConfiguration(motor_pool_gain=7.0)
    trace = tmp_path / "non-default-config.ndjson"
    run_embodiment(
        controller=ConnectomeController.create(
            graph,
            run_id="reference-embodiment",
            entity_id="reference-fly",
            configuration=configuration,
        ),
        graph=graph,
        embodiment=ReferenceEmbodiment(ReferenceArena()),
        output=trace,
        ticks=4,
    )
    result = replay_embodiment_trace(
        trace=trace,
        graph_factory=lambda: load_malecns_fixture(FIXTURE),
        embodiment_factory=lambda: ReferenceEmbodiment(ReferenceArena()),
        configuration=ControllerConfiguration(motor_pool_gain=7.0),
    )
    assert result["status"] == "neural_and_body_exact"
    with pytest.raises(EmbodimentError, match="does not match the trace"):
        replay_embodiment_trace(
            trace=trace,
            graph_factory=lambda: load_malecns_fixture(FIXTURE),
            embodiment_factory=lambda: ReferenceEmbodiment(ReferenceArena()),
        )
