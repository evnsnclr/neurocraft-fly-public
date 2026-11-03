"""Fixtures and golden-artifact builders for the AB-11 mutation-power suite.

The builders here are shared between the faithful baseline half and the mutant half of
every AB-11 test, so both halves are produced by exactly the same code path and the
only difference under test is the mutation itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from neurocraft_connectome import RuntimeGraph, load_malecns_fixture
from neurocraft_controller import (
    ConnectomeController,
    causal_record_from_applied_action,
    world_state_from_observation,
)
from neurocraft_neural import NeuralEngine
from neurocraft_protocol import (
    ACTION_TARGET_OFFSET_TICKS,
    CausalRecord,
    ClaimScope,
    TraceWriter,
)

from scripts.fixture_harness import HarnessWorld, build_manifest
from scripts.prepare_gametest_demo import _code_bundle, _dependency_bundle

# Reused rather than re-implemented: the AB-11 baseline must be byte-identical to the
# artifact the existing integration tests already accept.
from tests.integration.test_replay_minecraft_trace import _receipt

from .mutants import make_mutant_engine_controller

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "data" / "fixtures" / "mcns-v1.0-flight-power-v1.json"
TRACE_ENTITY_ID = "00000000-0000-0000-0000-0000000000ab"
TRACE_RECORD_COUNT = 3


@pytest.fixture(scope="session")
def controller_graph() -> RuntimeGraph:
    return load_malecns_fixture(FIXTURE)


def faithful_trace_controller(graph: RuntimeGraph, *, run_id: str) -> ConnectomeController:
    return ConnectomeController.create(graph, run_id=run_id, entity_id=TRACE_ENTITY_ID)


def mutant_trace_controller(
    graph: RuntimeGraph,
    *,
    engine_type: type[NeuralEngine],
    run_id: str,
) -> ConnectomeController:
    """A mutant author for the same trace identity the faithful author would publish."""

    return make_mutant_engine_controller(
        graph,
        engine_type=engine_type,
        run_id=run_id,
        entity_id=TRACE_ENTITY_ID,
    )


def minecraft_labelled_trace(path: Path, controller: ConnectomeController) -> None:
    """Write a G3-labelled Minecraft-shaped trace using the supplied controller.

    Structurally identical to ``tests/integration/test_replay_minecraft_trace.py::
    _minecraft_labelled_trace``, with the controller injected so a mutant controller
    can author the same artifact the faithful controller authors.
    """

    graph = controller.graph
    base = build_manifest(controller, graph)
    manifest = replace(
        base,
        claim_scope=ClaimScope.G3,
        artifacts=replace(
            base.artifacts,
            code_hash=_code_bundle()[0],
            dependency_lock_hash=_dependency_bundle()[0],
        ),
        java="21",
        minecraft="1.21.5",
        fabric="loader-0.16.10-api-0.128.2+1.21.5",
        mod="neurocraft-fly-0.1.0-ab11",
    )
    world = HarnessWorld()
    # An action applies ACTION_TARGET_OFFSET_TICKS after the observation it answers, so the
    # world it acts on is that observation plus the budget and the world it produced is one
    # tick further on.
    observations = tuple(
        world.observe(controller, tick)
        for tick in range(TRACE_RECORD_COUNT + ACTION_TARGET_OFFSET_TICKS + 1)
    )
    with TraceWriter(path, manifest) as writer:
        for tick in range(TRACE_RECORD_COUNT):
            observation = observations[tick]
            action = controller.step(observation, command_seq=tick)
            applied = tick + ACTION_TARGET_OFFSET_TICKS
            record = causal_record_from_applied_action(
                controller,
                record_seq=tick,
                observation=observation,
                world_after=world_state_from_observation(observations[applied + 1]),
                world_after_observation_seq=observations[applied + 1].observation_seq,
                world_after_game_tick=observations[applied + 1].game_tick,
                observation_age_ticks=ACTION_TARGET_OFFSET_TICKS,
            )
            writer.append(
                replace(
                    record,
                    actuation_receipt=_receipt(action, observations[applied]),
                    world_before=world_state_from_observation(observations[applied]),
                    compute_ns=100,
                    serialization_ns=20,
                    write_ns=30,
                )
            )


@dataclass(frozen=True, slots=True)
class DivergenceCounts:
    """The three counters the replay counterfactual publishes in its receipt."""

    state_divergent_records: int
    action_divergent_records: int
    neutral_action_records: int

    @property
    def replay_refuses(self) -> bool:
        """The refusal predicate of ``scripts/replay_minecraft_trace.py`` (:314-317)."""

        return self.state_divergent_records < 1 or self.action_divergent_records < 1


def zero_recurrence_divergences(
    records: Sequence[CausalRecord],
    counterfactual: ConnectomeController,
) -> DivergenceCounts:
    """Mirror of the zero-recurrence counterfactual in ``replay_minecraft_trace``.

    Reproduces the intervention-free path of ``scripts/replay_minecraft_trace.py``
    (:291-317): the recorded observations are rerun from the same initial state through
    a controller whose edge signs are mapped to zero, and state/action divergence is
    counted. The production script hardwires a faithful counterfactual controller, so
    this mirror is the only way to ask what that anti-bypass machinery would report if
    the recurrence itself were decorative.
    """

    state_divergences = 0
    action_divergences = 0
    neutral_actions = 0
    for record in records:
        command_seq = record.command_seq
        assert command_seq is not None
        action = counterfactual.step(
            replace(record.observation, identity=counterfactual.identity),
            command_seq=command_seq,
        )
        state_divergences += action.state_digest != record.state_after_digest
        action_divergences += action.motors != record.decoder_output
        neutral_actions += action.motors.is_neutral
    return DivergenceCounts(
        state_divergent_records=state_divergences,
        action_divergent_records=action_divergences,
        neutral_action_records=neutral_actions,
    )
