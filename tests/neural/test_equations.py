from __future__ import annotations

import math

import pytest
from neurocraft_connectome import SyntheticEdge, SyntheticNode, build_synthetic_graph
from neurocraft_neural import (
    DelayOverlay,
    DynamicsConfig,
    NeuralConfigurationError,
    NeuralEngine,
    NeuralIntegrityError,
    SignOverlay,
    Stimulation,
)


def test_one_node_zero_drive_matches_analytic_leak() -> None:
    graph = build_synthetic_graph("synthetic:one-node-leak", [SyntheticNode(1)], [])
    config = DynamicsConfig()
    engine = NeuralEngine(graph, sign_overlay=SignOverlay.unknown_zero(graph), config=config)
    engine.reset({1: 0.5})

    for step in range(1, 9):
        state = engine.step().state
        assert state.activity == pytest.approx((0.5 * config.leak**step,), abs=1e-15)


def test_direct_stimulation_starts_and_stops_on_exact_step() -> None:
    graph = build_synthetic_graph("synthetic:stimulus", [SyntheticNode(1)], [])
    config = DynamicsConfig()
    engine = NeuralEngine(graph, sign_overlay=SignOverlay.unknown_zero(graph), config=config)
    stimulus = Stimulation.from_body_values(graph, "pulse", {1: 0.75})

    stimulated = engine.step(stimulation=stimulus).state.activity[0]
    expected = (1.0 - config.leak) * math.tanh(0.75)
    assert stimulated == pytest.approx(expected, abs=1e-15)
    released = engine.step().state.activity[0]
    assert released == pytest.approx(config.leak * expected, abs=1e-15)


def test_one_edge_delay_and_sign_have_exact_causal_timing(two_node_graph) -> None:
    config = DynamicsConfig()
    positive = NeuralEngine(
        two_node_graph,
        sign_overlay=SignOverlay.explicit(two_node_graph, [1], ["synthetic_excitatory"]),
        config=config,
    )
    negative = NeuralEngine(
        two_node_graph,
        sign_overlay=SignOverlay.explicit(two_node_graph, [-1], ["synthetic_inhibitory"]),
        config=config,
    )
    for engine in (positive, negative):
        engine.reset({1: 1.0})

    first_positive = positive.step().state
    first_negative = negative.step().state
    assert first_positive.activity[1] == 0.0
    assert first_negative.activity[1] == 0.0

    second_positive = positive.step().state.activity[1]
    second_negative = negative.step().state.activity[1]
    expected = (1.0 - config.leak) * math.tanh(config.recurrent_gain)
    assert second_positive == pytest.approx(expected, abs=1e-15)
    assert second_negative == pytest.approx(-expected, abs=1e-15)


def test_two_step_edge_reads_literal_a_n_minus_d(two_node_graph) -> None:
    delay = DelayOverlay.explicit(two_node_graph, [2], provenance="synthetic_two_step_delay")
    engine = NeuralEngine(
        two_node_graph,
        sign_overlay=SignOverlay.explicit(two_node_graph, [1], ["synthetic"]),
        delay_overlay=delay,
    )
    engine.reset({1: 1.0})

    assert engine.step().state.activity[1] == 0.0
    assert engine.step().state.activity[1] == 0.0
    assert engine.step().state.activity[1] > 0.0


def test_synchronous_commit_does_not_use_new_source_values() -> None:
    graph = build_synthetic_graph(
        "synthetic:synchronous",
        [SyntheticNode(1), SyntheticNode(2), SyntheticNode(3), SyntheticNode(4)],
        [
            SyntheticEdge(1, 2, 1, 0),
            SyntheticEdge(2, 3, 1, 1),
            SyntheticEdge(3, 4, 1, 2),
        ],
    )
    signs = SignOverlay.explicit(graph, [1, 1, 1], ["synthetic"] * 3)
    engine = NeuralEngine(graph, sign_overlay=signs)
    engine.reset(
        {1: 0.8, 2: 0.4},
        prehistory=[{1: 0.7, 2: -0.25, 3: 0.1}],
    )
    config = engine.config

    result = engine.step().state.activity
    expected_three = config.leak * 0.0 + (1.0 - config.leak) * math.tanh(
        config.recurrent_gain * -0.25
    )
    assert result[2] == pytest.approx(expected_three, abs=1e-15)


def test_sparse_reference_matches_independent_dense_oracle() -> None:
    graph = build_synthetic_graph(
        "synthetic:dense-oracle",
        [SyntheticNode(10), SyntheticNode(20), SyntheticNode(30), SyntheticNode(40)],
        [
            SyntheticEdge(10, 20, 3, 0),
            SyntheticEdge(30, 20, 1, 1),
            SyntheticEdge(20, 30, 2, 2),
            SyntheticEdge(30, 30, 1, 3),
            SyntheticEdge(30, 40, 5, 4),
        ],
    )
    signs = SignOverlay.explicit(graph, [1, -1, 1, -1, 1], ["dense_oracle"] * 5)
    delays = DelayOverlay.explicit(graph, [1, 2, 1, 2, 1], provenance="dense_oracle")
    engine = NeuralEngine(graph, sign_overlay=signs, delay_overlay=delays)
    initial = {10: 0.4, 20: -0.1, 30: 0.25}
    engine.reset(initial, prehistory=[{10: 0.3}, {10: -0.2, 30: 0.15}])

    config = engine.config
    dense_activity = [initial.get(node.body_id, 0.0) for node in graph.nodes]
    dense_history = [
        [0.3, 0.0, 0.0, 0.0],
        [-0.2, 0.0, 0.15, 0.0],
    ]
    denominators = [0] * 4
    for edge in graph.edges:
        denominators[edge.post_index] += edge.raw_count

    for _ in range(12):
        dense_next: list[float] = []
        for post in range(4):
            contributions = []
            for edge in graph.edges:
                if edge.post_index == post:
                    source = dense_history[delays.steps[edge.dense_index] - 1][edge.pre_index]
                    weight = (
                        signs.values[edge.dense_index] * edge.raw_count / max(1, denominators[post])
                    )
                    contributions.append(weight * source)
            z = config.recurrent_gain * math.fsum(contributions)
            dense_next.append(
                config.leak * dense_activity[post] + (1.0 - config.leak) * math.tanh(z)
            )
        dense_history = [dense_activity.copy(), dense_history[0].copy()]
        dense_activity = dense_next
        sparse = engine.step().state.activity
        assert sparse == pytest.approx(dense_activity, abs=1e-15)


def test_long_recurrent_run_remains_finite_and_bounded(two_node_graph) -> None:
    engine = NeuralEngine(
        two_node_graph,
        sign_overlay=SignOverlay.explicit(two_node_graph, [1], ["synthetic"]),
        config=DynamicsConfig(recurrent_gain=10.0, baseline=100.0),
    )

    for _ in range(2_000):
        state = engine.step().state
        assert all(math.isfinite(value) and -1 <= value <= 1 for value in state.activity)


def test_fixed_drive_has_same_physical_time_solution_at_half_timestep() -> None:
    graph = build_synthetic_graph("synthetic:timestep", [SyntheticNode(1)], [])
    coarse_config = DynamicsConfig(neural_step_ms=10.0, tau_ms=50.0)
    fine_config = DynamicsConfig(neural_step_ms=5.0, tau_ms=50.0)
    coarse = NeuralEngine(
        graph,
        sign_overlay=SignOverlay.unknown_zero(graph),
        config=coarse_config,
    )
    fine = NeuralEngine(
        graph,
        sign_overlay=SignOverlay.unknown_zero(graph),
        config=fine_config,
    )
    pulse = Stimulation.from_body_values(graph, "constant-drive", {1: 0.4})

    for _ in range(10):
        coarse_state = coarse.step(stimulation=pulse).state
    for _ in range(20):
        fine_state = fine.step(stimulation=pulse).state
    analytic = math.tanh(0.4) * (1.0 - math.exp(-100.0 / 50.0))
    assert coarse_state.activity[0] == pytest.approx(analytic, abs=1e-15)
    assert fine_state.activity[0] == pytest.approx(analytic, abs=1e-15)


def test_nonfinite_or_forbidden_configuration_is_rejected_atomically(
    two_node_graph,
) -> None:
    with pytest.raises(NeuralConfigurationError, match="finite"):
        DynamicsConfig(baseline=float("nan"))
    with pytest.raises(NeuralConfigurationError, match="requires activation"):
        DynamicsConfig(activation="relu")
    engine = NeuralEngine(two_node_graph, sign_overlay=SignOverlay.unknown_zero(two_node_graph))
    before = engine.current_state()
    with pytest.raises(NeuralConfigurationError, match="finite"):
        engine.reset({1: float("inf")})
    assert engine.current_state() == before

    huge = Stimulation.from_body_values(two_node_graph, "overflow", {1: 1.7e308})
    overflow_engine = NeuralEngine(
        two_node_graph,
        sign_overlay=SignOverlay.unknown_zero(two_node_graph),
        config=DynamicsConfig(baseline=1.7e308),
    )
    overflow_before = overflow_engine.current_state()
    with pytest.raises(NeuralIntegrityError, match="non-finite drive"):
        overflow_engine.step(stimulation=huge)
    assert overflow_engine.current_state() == overflow_before
