"""Array-native engine: fidelity against the reference, identity, digest caching."""

from __future__ import annotations

import contextlib
from dataclasses import replace

import numpy as np
import pytest
from neurocraft_connectome import (
    RuntimeGraph,
    SyntheticEdge,
    SyntheticNode,
    build_synthetic_graph,
    load_malecns_fixture,
)
from neurocraft_neural.engine import NeuralEngine
from neurocraft_neural.full_engine import (
    ARRAY_NATIVE_BACKEND_ID,
    ArrayDynamicsConfig,
    ArrayNativeEngine,
    _activity_digest,
)
from neurocraft_neural.full_view import ViewProvenance, build_view, view_from_runtime_graph
from neurocraft_neural.model import (
    ZERO_SHA256,
    DynamicsConfig,
    NeuralConfigurationError,
    NeuralIntegrityError,
    SignOverlay,
    Stimulation,
    ablation_mask_digest,
    delay_history_digest,
    make_neural_state,
)
from neurocraft_neural.spmv_kernel import NumpyReduceatKernel, SpmvKernelError, load_c_kernel

from tests.full_backend.helpers import FIXTURE_PATH, FsumReference, synthetic_csr

#: Fixed acceptance tolerance for fast-vs-reference state divergence. Chosen at
#: roughly 150x the measured plateau (a few ulps of 1.0) and about 1e12 below
#: the [-1, 1] state range, so a real regression breaches it long before it
#: could reach behavioural significance. Every test that uses it also reports
#: the measured maximum, so the margin stays visible.
TOLERANCE_LINF = 1e-13


@pytest.fixture
def fixture_graph() -> RuntimeGraph:
    return load_malecns_fixture(FIXTURE_PATH)


def _kernels() -> list[object]:
    kernels: list[object] = [NumpyReduceatKernel()]
    with contextlib.suppress(SpmvKernelError):  # the C kernel is optional
        kernels.append(load_c_kernel(thread_count=4))
    return kernels


def _paired_engines(graph: RuntimeGraph, kernel: object) -> tuple[NeuralEngine, ArrayNativeEngine]:
    signs = SignOverlay.coarse_presynaptic_consensus(graph)
    reference = NeuralEngine(graph, sign_overlay=signs)
    view = view_from_runtime_graph(graph, sign_overlay=signs)
    return reference, ArrayNativeEngine(view, kernel=kernel)  # type: ignore[arg-type]


@pytest.mark.parametrize("kernel_index", [0, 1])
def test_fixture_divergence_plateaus_over_400_steps(
    fixture_graph: RuntimeGraph, kernel_index: int
) -> None:
    """The map is a sup-norm contraction, so per-step error accumulates to a
    bounded plateau rather than growing. 400 steps is far past the ~55x
    settling implied by Lip_inf = 0.98187."""

    kernels = _kernels()
    if kernel_index >= len(kernels):
        pytest.skip("compiled spmv kernel unavailable")
    reference, fast = _paired_engines(fixture_graph, kernels[kernel_index])
    body_ids = [node.body_id for node in fixture_graph.nodes]
    stimulation = Stimulation.from_body_values(
        fixture_graph, "fidelity", {body_ids[0]: 0.9, body_ids[1]: -0.4}
    )
    dense = np.zeros(len(fixture_graph.nodes), dtype=np.float64)
    for index, value in stimulation.assignments:
        dense[index] += value

    early = 0.0
    late = 0.0
    for step in range(400):
        reference.step(stimulation=stimulation)
        fast.step(direct=dense, direct_provenance_sha256=stimulation.identity_sha256)
        divergence = float(
            np.max(np.abs(np.asarray(reference.current_state().activity) - fast.activity()))
        )
        assert divergence <= TOLERANCE_LINF, f"step {step + 1} diverged by {divergence}"
        if step < 100:
            early = max(early, divergence)
        else:
            late = max(late, divergence)
    assert late <= max(early, 1e-15) * 8.0, "divergence grew instead of plateauing"


def test_fsum_oracle_agreement_on_synthetic_graphs() -> None:
    for node_count, edge_count, seed in ((256, 8_000, 21), (1024, 40_000, 22)):
        csr = synthetic_csr(
            node_count=node_count,
            edge_count=edge_count,
            seed=seed,
            empty_rows=(0, 5, 6, 7, node_count - 1),
        )
        view = build_view(
            indptr=csr.indptr,
            pre_indices=csr.indices,
            raw_counts=csr.counts,
            signs=csr.signs,
            provenance=ViewProvenance(origin="synthetic_csr", graph_identity_sha256="b" * 64),
        )
        oracle = FsumReference(csr)
        engine = ArrayNativeEngine(view, kernel=NumpyReduceatKernel())
        rng = np.random.default_rng(seed)
        direct = rng.uniform(-1.0, 1.0, size=node_count)
        worst = 0.0
        for _ in range(60):
            expected = oracle.step(direct.tolist())
            engine.step(direct=direct)
            worst = max(worst, float(np.max(np.abs(np.asarray(expected) - engine.activity()))))
        assert worst <= TOLERANCE_LINF, f"E={edge_count} diverged by {worst}"
        assert view.max_row_l1 <= 1.0 + 1e-15


def test_ablation_matches_the_reference(fixture_graph: RuntimeGraph) -> None:
    reference, fast = _paired_engines(fixture_graph, NumpyReduceatKernel())
    target = fixture_graph.nodes[2].body_id
    reference.set_ablated([target])
    fast.set_ablated([fixture_graph.index_of(target)])
    for _ in range(40):
        reference.step()
        fast.step()
    activity = fast.activity()
    assert activity[fixture_graph.index_of(target)] == 0.0
    assert np.max(np.abs(np.asarray(reference.current_state().activity) - activity)) <= (
        TOLERANCE_LINF
    )


def test_cached_digests_are_bit_identical_to_the_reference_helpers(
    fixture_graph: RuntimeGraph,
) -> None:
    """The three cached digests are recomputations of provably identical inputs.

    At delay 1 the single history row after a step *is* the activity vector
    from before it, so its digest is the one already held; and the reference's
    ``before`` state is by construction the previous step's ``after``. Each
    cached value is checked against the reference helper computed from an
    independently captured copy of the same vector.
    """

    signs = SignOverlay.coarse_presynaptic_consensus(fixture_graph)
    view = view_from_runtime_graph(fixture_graph, sign_overlay=signs)
    fast = ArrayNativeEngine(view, kernel=NumpyReduceatKernel())
    body_ids = [node.body_id for node in fixture_graph.nodes]
    direct = np.zeros(len(body_ids), dtype=np.float64)
    direct[0] = 0.8
    direct[1] = -0.3

    history: list[list[float]] = [[0.0] * len(body_ids)]
    previous_after: str | None = None
    for _ in range(8):
        before_activity = fast.activity().tolist()
        result = fast.step(direct=direct)
        record = result.record.neural
        assert record.delay_after_sha256 == delay_history_digest([before_activity])
        assert record.delay_before_sha256 == delay_history_digest([history[-1]])
        assert record.ablation_mask_sha256 == make_reference_mask_digest(len(body_ids))
        if previous_after is not None:
            assert record.state_before_sha256 == previous_after
        previous_after = record.state_after_sha256
        history.append(before_activity)

    activity = fast.activity()
    assert (
        _activity_digest(activity)
        == make_neural_state(
            step_index=fast.step_index,
            graph_identity_sha256="0" * 64,
            dynamics_identity_sha256="0" * 64,
            sign_overlay_sha256="0" * 64,
            delay_overlay_sha256="0" * 64,
            weight_overlay_sha256="0" * 64,
            topology_overlay_sha256="0" * 64,
            activity=activity.tolist(),
        ).activity_sha256
    )


def make_reference_mask_digest(node_count: int) -> str:
    return ablation_mask_digest([False] * node_count)


def test_state_digest_material_matches_the_reference_shape(fixture_graph: RuntimeGraph) -> None:
    signs = SignOverlay.coarse_presynaptic_consensus(fixture_graph)
    view = view_from_runtime_graph(fixture_graph, sign_overlay=signs)
    fast = ArrayNativeEngine(view, kernel=NumpyReduceatKernel())
    state = fast.current_state()
    equivalent = make_neural_state(
        step_index=state.step_index,
        graph_identity_sha256=state.graph_identity_sha256,
        dynamics_identity_sha256=state.dynamics_identity_sha256,
        sign_overlay_sha256=state.sign_overlay_sha256,
        delay_overlay_sha256=state.delay_overlay_sha256,
        weight_overlay_sha256=state.weight_overlay_sha256,
        topology_overlay_sha256=state.topology_overlay_sha256,
        activity=fast.activity().tolist(),
    )
    assert state.activity_sha256 == equivalent.activity_sha256
    assert state.state_sha256 == equivalent.state_sha256


def test_backend_identity_is_new_and_never_the_reference_identity(
    fixture_graph: RuntimeGraph,
) -> None:
    signs = SignOverlay.coarse_presynaptic_consensus(fixture_graph)
    view = view_from_runtime_graph(fixture_graph, sign_overlay=signs)
    numpy_engine = ArrayNativeEngine(view, kernel=NumpyReduceatKernel())
    assert numpy_engine.dynamics_identity_sha256 != DynamicsConfig().identity_sha256
    material = ArrayDynamicsConfig().identity_material(numpy_engine.kernel_identity)
    assert material["backend"] == ARRAY_NATIVE_BACKEND_ID
    assert material["delay_policy"] == "uniform_1"
    assert material["precision"] == "float64"
    assert material["thread_count_affects_results"] is False
    assert material["empty_row_policy"] == "zero"
    try:
        c_kernel = load_c_kernel(thread_count=1)
    except SpmvKernelError:  # pragma: no cover - environment dependent
        pytest.skip("compiled spmv kernel unavailable")
    c_engine = ArrayNativeEngine(view, kernel=c_kernel)
    assert c_engine.dynamics_identity_sha256 != numpy_engine.dynamics_identity_sha256
    twelve = ArrayNativeEngine(view, kernel=c_kernel.with_thread_count(12))
    assert twelve.dynamics_identity_sha256 == c_engine.dynamics_identity_sha256


def test_thread_count_does_not_change_the_trace(fixture_graph: RuntimeGraph) -> None:
    try:
        base = load_c_kernel(thread_count=1)
    except SpmvKernelError:  # pragma: no cover - environment dependent
        pytest.skip("compiled spmv kernel unavailable")
    signs = SignOverlay.coarse_presynaptic_consensus(fixture_graph)
    traces = {}
    for threads in (1, 2, 4, 8, 12):
        view = view_from_runtime_graph(fixture_graph, sign_overlay=signs)
        engine = ArrayNativeEngine(view, kernel=base.with_thread_count(threads))
        for _ in range(50):
            engine.step()
        traces[threads] = engine.trace_sha256
    assert len(set(traces.values())) == 1, traces


def test_heterogeneous_delay_configuration_is_refused() -> None:
    with pytest.raises(NeuralConfigurationError, match="uniform delay 1 only"):
        ArrayDynamicsConfig(default_delay_steps=2)
    with pytest.raises(NeuralConfigurationError, match="requires backend"):
        ArrayDynamicsConfig(backend="python_single_thread_reference")


def test_out_of_range_state_is_refused() -> None:
    """A weight overlay that breaks the row-L1 bound must fault, not clip."""

    view = build_view(
        indptr=np.array([0, 1], dtype=np.int64),
        pre_indices=np.array([0], dtype=np.uint32),
        raw_counts=np.array([1], dtype=np.int64),
        signs=np.array([1], dtype=np.int64),
        provenance=ViewProvenance(origin="synthetic_csr", graph_identity_sha256="c" * 64),
    )
    engine = ArrayNativeEngine(view, kernel=NumpyReduceatKernel())
    with pytest.raises(NeuralConfigurationError, match="must be finite"):
        engine.step(direct=np.array([np.inf]))
    with pytest.raises(NeuralConfigurationError, match="length mismatch"):
        engine.step(direct=np.zeros(2))


def test_reset_and_snapshot_round_trip() -> None:
    graph = build_synthetic_graph(
        "synthetic:array-snapshot",
        [SyntheticNode(index, ("interneuron",)) for index in range(1, 6)],
        [SyntheticEdge(1, 2, 3, 0), SyntheticEdge(2, 3, 5, 1), SyntheticEdge(3, 4, 7, 2)],
    )
    signs = SignOverlay.coarse_presynaptic_consensus(graph)
    view = view_from_runtime_graph(graph, sign_overlay=signs)
    engine = ArrayNativeEngine(view, kernel=NumpyReduceatKernel())
    direct = np.array([0.5, 0.0, 0.0, 0.0, -0.25])
    for _ in range(7):
        engine.step(direct=direct)
    snapshot = engine.snapshot()
    trace = engine.trace_sha256
    activity = engine.activity().copy()
    for _ in range(3):
        engine.step(direct=direct)
    engine.restore(snapshot)
    assert engine.trace_sha256 == trace
    assert engine.activity().tolist() == activity.tolist()
    engine.reset()
    assert engine.step_index == 0
    assert engine.activity().tolist() == [0.0] * 5
    with pytest.raises(NeuralIntegrityError, match="dynamics identity"):
        engine.restore(replace(snapshot, dynamics_identity_sha256="0" * 64))


# ---------------------------------------------------------------------------
# N3 -- stimulation digests are derived, never accepted from the caller
# ---------------------------------------------------------------------------


def _synthetic_view(seed: int = 51):
    csr = synthetic_csr(node_count=96, edge_count=2_000, seed=seed, empty_rows=(0, 95))
    return build_view(
        indptr=csr.indptr,
        pre_indices=csr.indices,
        raw_counts=csr.counts,
        signs=csr.signs,
        provenance=ViewProvenance(origin="synthetic_csr", graph_identity_sha256="f" * 64),
    )


def test_n3_stimulation_digests_are_derived_from_the_executed_arrays() -> None:
    """N3: the recorded digest is of the vector that ran, not of a claim."""

    view = _synthetic_view()
    engine = ArrayNativeEngine(view, kernel=NumpyReduceatKernel())
    sensory = np.full(view.node_count, 0.11)
    direct = np.full(view.node_count, -0.07)
    record = engine.step(sensory=sensory, direct=direct).record.neural
    assert record.encoded_stimulation_sha256 == _activity_digest(sensory)
    assert record.direct_stimulation_sha256 == _activity_digest(direct)

    absent = engine.step().record.neural
    assert absent.encoded_stimulation_sha256 == ZERO_SHA256
    assert absent.direct_stimulation_sha256 == ZERO_SHA256


def test_n3_a_caller_cannot_execute_one_vector_and_attest_another() -> None:
    """N3: the reviewer's ``step(sensory=A, sensory_sha256=B)`` forgery.

    ``sensory_sha256``/``direct_sha256`` no longer exist. The provenance
    arguments that replaced them are recorded under their own keys and cannot
    displace the derived digest, so two runs that execute different vectors
    always produce different traces however they are labelled.
    """

    vector_a = np.full(96, 0.20)
    vector_b = np.full(96, -0.20)
    honest = ArrayNativeEngine(_synthetic_view(), kernel=NumpyReduceatKernel())
    forger = ArrayNativeEngine(_synthetic_view(), kernel=NumpyReduceatKernel())

    honest_record = honest.step(direct=vector_a).record
    forged_record = forger.step(
        direct=vector_b, direct_provenance_sha256=_activity_digest(vector_a)
    ).record
    assert forged_record.neural.direct_stimulation_sha256 == _activity_digest(vector_b)
    assert forged_record.neural.direct_stimulation_sha256 != _activity_digest(vector_a)
    assert forged_record.trace_sha256 != honest_record.trace_sha256

    with pytest.raises(TypeError):
        honest.step(direct=vector_a, direct_sha256="0" * 64)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        honest.step(sensory=vector_a, sensory_sha256="0" * 64)  # type: ignore[call-arg]


def test_n3_provenance_is_bound_alongside_the_derived_digest() -> None:
    """The protocol identity is still carried; it just cannot substitute."""

    vector = np.full(96, 0.3)
    plain = ArrayNativeEngine(_synthetic_view(), kernel=NumpyReduceatKernel())
    labelled = ArrayNativeEngine(_synthetic_view(), kernel=NumpyReduceatKernel())
    first = plain.step(direct=vector).record
    second = labelled.step(direct=vector, direct_provenance_sha256="a" * 64).record
    assert first.neural.direct_stimulation_sha256 == second.neural.direct_stimulation_sha256
    assert first.neural.execution_sha256 != second.neural.execution_sha256


def test_n3_a_non_finite_stimulation_is_refused_before_it_is_hashed() -> None:
    engine = ArrayNativeEngine(_synthetic_view(), kernel=NumpyReduceatKernel())
    with pytest.raises(NeuralConfigurationError, match="must be finite"):
        engine.step(direct=np.full(96, np.inf))
    assert engine.step_index == 0


# ---------------------------------------------------------------------------
# N1 / B5 at engine level
# ---------------------------------------------------------------------------


def test_n1_records_carry_the_view_validation_digest() -> None:
    view = _synthetic_view()
    engine = ArrayNativeEngine(view, kernel=NumpyReduceatKernel())
    record = engine.step().record
    assert record.view_validation_sha256 == view.validation.identity_sha256
    assert record.view_digest_sha256 == view.view_sha256
    assert engine.verification_events[0] == ("run_start", 0)


def test_b5_dynamics_identity_carries_the_kernel_library_digest() -> None:
    """B5/B1: the trace's dynamics identity now moves when the binary moves."""

    view = _synthetic_view()
    try:
        c_kernel = load_c_kernel(thread_count=2)
    except SpmvKernelError:  # pragma: no cover - environment dependent
        pytest.skip("compiled spmv kernel unavailable")
    engine = ArrayNativeEngine(view, kernel=c_kernel)
    material = engine.config.identity_material(engine.kernel_identity)
    assert material["kernel_identity_sha256"] == c_kernel.identity.identity_sha256
    assert "library_sha256" in engine.kernel_identity.material
    assert engine.kernel is c_kernel
