from __future__ import annotations

from dataclasses import replace
from itertools import pairwise

import pytest
from neurocraft_neural import (
    DelayOverlay,
    MotorDecoder,
    MotorPool,
    NeuralEngine,
    NeuralIntegrityError,
    PoolMember,
    SensorChannel,
    SensorTarget,
    SensoryAdapter,
    SignOverlay,
    Stimulation,
)


def test_snapshot_restore_replays_state_records_and_chain(two_node_graph) -> None:
    engine = NeuralEngine(
        two_node_graph,
        sign_overlay=SignOverlay.explicit(two_node_graph, [1], ["synthetic"]),
    )
    pulse = Stimulation.from_body_values(two_node_graph, "pulse", {1: 0.6})
    engine.step(stimulation=pulse)
    engine.step()
    snapshot = engine.snapshot()

    first = [engine.step(stimulation=pulse if index == 1 else None) for index in range(5)]
    engine.restore(snapshot)
    second = [engine.step(stimulation=pulse if index == 1 else None) for index in range(5)]

    assert first == second
    assert first[-1].record.trace_sha256 == engine.trace_sha256


def test_corrupt_snapshot_is_refused_without_mutating_engine(two_node_graph) -> None:
    engine = NeuralEngine(two_node_graph, sign_overlay=SignOverlay.unknown_zero(two_node_graph))
    engine.step()
    before = engine.current_state()
    snapshot = engine.snapshot()
    corrupt = replace(snapshot, activity=(0.5, 0.5))

    with pytest.raises(NeuralIntegrityError, match="snapshot content digest"):
        engine.restore(corrupt)
    assert engine.current_state() == before


def test_ablation_clamps_state_and_clears_delayed_outgoing_signal(two_node_graph) -> None:
    engine = NeuralEngine(
        two_node_graph,
        sign_overlay=SignOverlay.explicit(two_node_graph, [1], ["synthetic"]),
    )
    engine.reset({1: 1.0}, prehistory=[{1: 1.0}])
    engine.set_ablated([1])

    state = engine.step().state
    assert state.activity == (0.0, 0.0)
    snapshot = engine.snapshot()
    assert snapshot.delay_history[0][0] == 0.0
    engine.set_ablated([1], ablated=False)
    assert engine.step().state.activity[0] == 0.0


def test_neural_trace_records_are_hash_linked(two_node_graph) -> None:
    engine = NeuralEngine(two_node_graph, sign_overlay=SignOverlay.unknown_zero(two_node_graph))

    records = [engine.step().record for _ in range(6)]
    assert records[0].previous_trace_sha256 == "0" * 64
    assert all(
        current.previous_trace_sha256 == previous.trace_sha256
        for previous, current in pairwise(records)
    )


def test_graph_support_and_counts_are_unchanged_by_every_runtime_overlay(
    male_fixture_graph,
) -> None:
    support = male_fixture_graph.edge_support_sha256
    counts = male_fixture_graph.raw_counts_sha256
    signs = SignOverlay.coarse_presynaptic_consensus(male_fixture_graph)
    delays = DelayOverlay.uniform(male_fixture_graph, 1)
    engine = NeuralEngine(
        male_fixture_graph,
        sign_overlay=signs,
        delay_overlay=delays,
    )
    stimulus = Stimulation.from_body_values(male_fixture_graph, "named-probe", {27143: 0.25})
    engine.set_ablated([28949])
    engine.step(stimulation=stimulus)

    assert male_fixture_graph.edge_support_sha256 == support
    assert male_fixture_graph.raw_counts_sha256 == counts
    male_fixture_graph.assert_integrity()


def test_mismatched_or_mutated_overlay_is_refused(two_node_graph) -> None:
    overlay = SignOverlay.explicit(two_node_graph, [1], ["synthetic"])
    corrupted = replace(overlay, values=(-1,))

    with pytest.raises(NeuralIntegrityError, match="identity mismatch"):
        NeuralEngine(two_node_graph, sign_overlay=corrupted)


def test_malecns_fixture_complete_trace_is_same_build_deterministic(
    male_fixture_graph,
) -> None:
    adapter = SensoryAdapter(
        male_fixture_graph,
        [
            SensorChannel("right_airflow", "normalized", (SensorTarget(77185, 1.0),)),
            SensorChannel("left_strain", "normalized", (SensorTarget(805722, 1.0),)),
        ],
    )
    signs = SignOverlay.coarse_presynaptic_consensus(male_fixture_graph)
    decoder = MotorDecoder(
        male_fixture_graph,
        [
            MotorPool("left_wing", (PoolMember(800241, 1.0),)),
            MotorPool("right_wing", (PoolMember(800917, 1.0),)),
        ],
    )

    def execute():
        engine = NeuralEngine(
            male_fixture_graph,
            sign_overlay=signs,
            sensory_adapter_sha256=adapter.identity_sha256,
        )
        frames = []
        for step in range(16):
            encoded = adapter.encode(
                {
                    "right_airflow": ((step % 5) - 2) / 2,
                    "left_strain": ((step * 3 % 7) - 3) / 3,
                }
            )
            result = engine.step(encoded)
            frames.append((result, decoder.decode(result.state)))
        return engine.trace_sha256, tuple(frames)

    first = execute()
    second = execute()
    assert first == second
    # libm-backed activation can produce different bitwise digests across
    # hardware/OS combinations. Exact same-build replay above is the contract.
    assert len(first[0]) == 64
    bytes.fromhex(first[0])
