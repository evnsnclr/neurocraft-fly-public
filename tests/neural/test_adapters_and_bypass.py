from __future__ import annotations

from dataclasses import replace

import pytest
from neurocraft_connectome import SyntheticEdge, SyntheticNode, build_synthetic_graph
from neurocraft_neural import (
    MotorDecoder,
    MotorPool,
    NeuralConfigurationError,
    NeuralEngine,
    NeuralIntegrityError,
    PoolMember,
    SensorChannel,
    SensorTarget,
    SensoryAdapter,
    SignOverlay,
)


def _adapter_graph():
    return build_synthetic_graph(
        "synthetic:adapter-chain",
        [
            SyntheticNode(10, ("sensory",), type_name="wind receptor"),
            SyntheticNode(20, ("intermediate",), type_name="relay"),
            SyntheticNode(30, ("motor",), type_name="wing motor"),
            SyntheticNode(
                40,
                ("named:DNg07",),
                type_name="DNg07",
                superclass="descending_neuron",
            ),
        ],
        [
            SyntheticEdge(10, 20, 4, 0),
            SyntheticEdge(20, 30, 4, 1),
            SyntheticEdge(20, 40, 2, 2),
        ],
    )


def test_sensory_adapter_is_sparse_named_and_schema_exact() -> None:
    graph = _adapter_graph()
    adapter = SensoryAdapter(
        graph,
        [
            SensorChannel(
                "airflow",
                "normalized",
                (SensorTarget(10, 0.5),),
                gain=2.0,
                clip_min=-1.0,
                clip_max=1.0,
            )
        ],
    )

    encoded = adapter.encode({"airflow": 0.75})
    assert encoded.vector == (0.5, 0.0, 0.0, 0.0)
    assert encoded.assignments == (("airflow", 0, 0.5),)
    encoded.assert_integrity()
    with pytest.raises(NeuralConfigurationError, match="schema mismatch"):
        adapter.encode({})
    with pytest.raises(NeuralConfigurationError, match="schema mismatch"):
        adapter.encode({"airflow": 0.0, "target_label": 1.0})
    with pytest.raises(NeuralConfigurationError, match="finite"):
        adapter.encode({"airflow": float("nan")})
    with pytest.raises(NeuralConfigurationError, match="field names must be strings"):
        adapter.encode({1: 0.0})  # type: ignore[dict-item]


def test_sensory_adapter_rejects_internal_or_motor_projection() -> None:
    graph = _adapter_graph()

    for target in (20, 30, 40):
        with pytest.raises(NeuralConfigurationError, match="not a named sensory"):
            SensoryAdapter(
                graph,
                [SensorChannel("raw", "normalized", (SensorTarget(target, 1.0),))],
            )


def test_motor_decoder_reads_only_named_output_state() -> None:
    graph = _adapter_graph()
    decoder = MotorDecoder(
        graph,
        [
            MotorPool("thrust", (PoolMember(30, 1.0),)),
            MotorPool("yaw", (PoolMember(40, -1.0),)),
        ],
    )
    engine = NeuralEngine(graph, sign_overlay=SignOverlay.unknown_zero(graph))
    engine.reset({30: 0.6, 40: -0.2})

    frame = decoder.decode(engine.current_state())
    assert dict(frame.values) == pytest.approx({"thrust": 0.6, "yaw": 0.2})
    with pytest.raises(NeuralConfigurationError, match="only NeuralState"):
        decoder.decode({"airflow": 1.0})  # type: ignore[arg-type]
    with pytest.raises(NeuralIntegrityError, match="activity digest"):
        decoder.decode(replace(engine.current_state(), activity=(0.0, 0.0, 1.0, 0.0)))


def test_motor_decoder_rejects_sensory_or_arbitrary_internal_readout() -> None:
    graph = _adapter_graph()

    for body_id in (10, 20):
        with pytest.raises(NeuralConfigurationError, match="not motor/descending"):
            MotorDecoder(graph, [MotorPool("bad", (PoolMember(body_id, 1.0),))])
    with pytest.raises(NeuralConfigurationError, match="empty"):
        MotorDecoder(graph, [MotorPool("empty", ())])


def test_observation_cannot_bypass_recurrence_into_motor_command() -> None:
    graph = _adapter_graph()
    adapter = SensoryAdapter(
        graph,
        [SensorChannel("airflow", "normalized", (SensorTarget(10, 1.0),))],
    )
    decoder = MotorDecoder(graph, [MotorPool("thrust", (PoolMember(30, 1.0),))])
    active = NeuralEngine(
        graph,
        sign_overlay=SignOverlay.explicit(graph, [1, 1, 1], ["synthetic"] * 3),
        sensory_adapter_sha256=adapter.identity_sha256,
    )
    disconnected = NeuralEngine(
        graph,
        sign_overlay=SignOverlay.unknown_zero(graph),
        sensory_adapter_sha256=adapter.identity_sha256,
    )
    encoded = adapter.encode({"airflow": 1.0})

    with pytest.raises(NeuralConfigurationError, match="never raw observations"):
        active.step({"airflow": 1.0})  # type: ignore[arg-type]
    for _ in range(5):
        active_state = active.step(encoded).state
        disconnected_state = disconnected.step(encoded).state
    assert dict(decoder.decode(active_state).values)["thrust"] > 0
    assert dict(decoder.decode(disconnected_state).values)["thrust"] == 0


def test_encoded_frame_from_unbound_adapter_is_refused() -> None:
    graph = _adapter_graph()
    first = SensoryAdapter(graph, [SensorChannel("a", "unit", (SensorTarget(10, 1.0),))])
    second = SensoryAdapter(graph, [SensorChannel("b", "unit", (SensorTarget(10, 1.0),))])
    engine = NeuralEngine(
        graph,
        sign_overlay=SignOverlay.unknown_zero(graph),
        sensory_adapter_sha256=first.identity_sha256,
    )

    with pytest.raises(NeuralIntegrityError, match="unbound adapter"):
        engine.step(second.encode({"b": 1.0}))


def test_coarse_sign_overlay_preserves_unknown_source_layer(male_fixture_graph) -> None:
    support = male_fixture_graph.edge_support_sha256
    counts = male_fixture_graph.raw_counts_sha256
    overlay = SignOverlay.coarse_presynaptic_consensus(male_fixture_graph)

    assert set(overlay.values) <= {-1, 0, 1}
    assert all(edge.canonical_sign_status == "unknown" for edge in male_fixture_graph.edges)
    assert male_fixture_graph.edge_support_sha256 == support
    assert male_fixture_graph.raw_counts_sha256 == counts
    assert all(
        item.startswith("inferred_from_presynaptic_consensus_nt:") for item in overlay.provenance
    )
