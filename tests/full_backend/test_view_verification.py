"""Verification policy: pinning, amortised checkpoints, fail-closed behaviour.

These are the array-backend analogues of the AB-11 mutants: the per-step O(E)
rebuild is gone, so each surface it used to cover needs its own named detector
and its own mutant.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from neurocraft_neural import view_verification
from neurocraft_neural.full_engine import ArrayNativeEngine
from neurocraft_neural.full_view import ViewProvenance, build_view
from neurocraft_neural.spmv_kernel import NumpyReduceatKernel
from neurocraft_neural.view_verification import (
    UNCONDITIONAL_VERIFICATION_TRIGGERS,
    WITHDRAWN_VERIFICATION_TRIGGERS,
    IncrementalViewDigest,
    VerificationPolicy,
    VerifiedView,
    ViewIntegrityError,
    assert_buffers_pinned,
    capture_pins,
    hash_buffers,
)

from tests.full_backend.helpers import synthetic_csr


def _view(node_count: int = 128, edge_count: int = 4_000, seed: int = 31):
    csr = synthetic_csr(
        node_count=node_count,
        edge_count=edge_count,
        seed=seed,
        empty_rows=(0, 9, 10, 11, node_count - 1),
    )
    return build_view(
        indptr=csr.indptr,
        pre_indices=csr.indices,
        raw_counts=csr.counts,
        signs=csr.signs,
        provenance=ViewProvenance(origin="synthetic_csr", graph_identity_sha256="d" * 64),
    )


def _engine(view, period: int = 4) -> ArrayNativeEngine:
    return ArrayNativeEngine(
        view,
        kernel=NumpyReduceatKernel(),
        verification_policy=VerificationPolicy(full_coverage_period_steps=period),
    )


def _stream_offset(view, buffer_name: str, element_index: int) -> int:
    """Byte offset of one element inside the declared logical byte stream."""

    offset = 0
    for name, array in view.named_buffers():
        if name == buffer_name:
            return offset + element_index * array.dtype.itemsize
        offset += int(array.nbytes)
    raise AssertionError(f"no buffer named {buffer_name!r}")


def _slice_index(view, period: int, offset: int) -> int:
    budget = max(
        1, VerificationPolicy(full_coverage_period_steps=period).bytes_per_step(view.buffer_bytes)
    )
    return offset // budget


def _perturb(array, index: int, value=None) -> None:
    """Write through the frozen flag and restore it, as an aliasing writer would."""

    array.flags.writeable = True
    array[index] = np.nextafter(array[index], 1.0) if value is None else value
    array.flags.writeable = False


def test_load_time_digest_is_recomputed_not_trusted() -> None:
    view = _view()
    forged = type(view)(
        indptr=view.indptr,
        indices=view.indices,
        weights=view.weights,
        empty_rows=view.empty_rows,
        node_count=view.node_count,
        edge_count=view.edge_count,
        max_in_degree=view.max_in_degree,
        view_sha256="0" * 64,
        provenance=view.provenance,
        validation=view.validation,
    )
    with pytest.raises(ViewIntegrityError, match="does not match its buffers at load"):
        _engine(forged)


def test_b4_a_view_that_misreports_its_index_bound_is_refused_at_load() -> None:
    """B4: the validation record is re-derived from the buffers, not trusted."""

    view = _view()
    lying = type(view)(
        indptr=view.indptr,
        indices=view.indices,
        weights=view.weights,
        empty_rows=view.empty_rows,
        node_count=view.node_count,
        edge_count=view.edge_count,
        max_in_degree=view.max_in_degree,
        view_sha256=view.view_sha256,
        provenance=view.provenance,
        validation=replace(view.validation, max_observed_index=0),
    )
    with pytest.raises(ViewIntegrityError, match="validation record does not describe"):
        _engine(lying)


def test_m_view_a_unfrozen_and_perturbed_weight_is_caught() -> None:
    """Flip ``writeable`` back and perturb one weight by 1 ulp between steps."""

    view = _view()
    engine = _engine(view)
    engine.step()
    view.weights.flags.writeable = True
    view.weights[7] = np.nextafter(view.weights[7], 1.0)
    with pytest.raises(ViewIntegrityError, match="made writeable"):
        engine.step()


def test_m_view_a_refrozen_mutation_is_caught_by_the_amortised_checkpoint() -> None:
    """A mutation that restores the frozen flag survives pinning but not hashing."""

    view = _view()
    engine = _engine(view, period=3)
    engine.step()
    view.weights.flags.writeable = True
    view.weights[11] = np.nextafter(view.weights[11], 1.0)
    view.weights.flags.writeable = False
    with pytest.raises(ViewIntegrityError, match="modified runtime view"):
        for _ in range(10):
            engine.step()


def test_m_view_b_rebound_indices_are_caught_on_the_next_step() -> None:
    """Same dtype, same shape, different buffer: the data pointer pin catches it."""

    view = _view()
    engine = _engine(view)
    engine.step()
    shuffled = view.indices.copy()
    np.random.default_rng(0).shuffle(shuffled)
    shuffled.flags.writeable = False
    object.__setattr__(view, "indices", shuffled)
    with pytest.raises(ViewIntegrityError, match="rebound or reshaped"):
        engine.step()


def test_a_faulted_engine_refuses_every_later_step() -> None:
    view = _view()
    engine = _engine(view, period=2)
    engine.step()
    view.weights.flags.writeable = True
    view.weights[3] = -view.weights[3] if view.weights[3] else 0.5
    view.weights.flags.writeable = False
    with pytest.raises(ViewIntegrityError):
        for _ in range(6):
            engine.step()
    with pytest.raises(ViewIntegrityError, match="engine is faulted"):
        engine.step()


def test_records_carry_the_coverage_claim() -> None:
    view = _view()
    engine = _engine(view, period=3)
    policy_digest = VerificationPolicy(full_coverage_period_steps=3).identity_sha256
    covered: list[int] = []
    for _ in range(9):
        record = engine.step().record
        assert record.view_digest_sha256 == view.view_sha256
        assert record.verification_policy_sha256 == policy_digest
        covered.append(record.verified_through_step)
    assert max(covered) > 0, "no full checkpoint completed inside the declared period"
    assert engine.uncovered_steps <= 3


def test_incremental_digest_equals_the_single_shot_digest() -> None:
    view = _view()
    buffers = view.named_buffers()
    expected = hash_buffers(buffers)
    incremental = IncrementalViewDigest(buffers, bytes_per_step=1024)
    assert incremental.expected_view_sha256 == expected
    assert incremental.slice_count == -(-incremental.total_bytes // 1024)
    for step in range(10_000):
        outcome = incremental.advance()
        assert outcome.matched
        assert outcome.slice_index == step % incremental.slice_count
        if outcome.completed_view_sha256 is not None:
            assert outcome.completed_view_sha256 == expected
            assert step + 1 == incremental.slice_count
            break
    else:  # pragma: no cover - would mean the budget never covered the view
        pytest.fail("incremental digest never completed a pass")


def test_b2_a_mismatch_is_reported_on_the_slice_that_reads_it() -> None:
    """B2: per-slice comparison, not pass-completion-only comparison.

    The mismatch must surface on the advance that reads the changed bytes, and
    on no earlier one; every slice before it is honestly clean.
    """

    budget = 4096
    view = _view()
    incremental = IncrementalViewDigest(view.named_buffers(), bytes_per_step=budget)
    recorded = incremental.expected_slice_digests
    offset = _stream_offset(view, "weights", 0)
    guilty = offset // budget
    assert 0 < guilty < incremental.slice_count

    _perturb(view.weights, 0)
    for expected_index in range(guilty):
        clean = incremental.advance()
        assert clean.slice_index == expected_index
        assert clean.matched, f"slice {expected_index} does not contain the mutation"
        assert clean.completed_view_sha256 is None
    caught = incremental.advance()
    assert caught.slice_index == guilty
    assert caught.matched is False
    assert caught.start_offset <= offset < caught.end_offset
    assert incremental.expected_slice_digests == recorded, "expectations must not drift"


def test_pin_capture_detects_dtype_and_shape_changes() -> None:
    view = _view()
    token = VerifiedView(
        view_sha256=view.view_sha256,
        verified_at_step=0,
        buffers=capture_pins(view.named_buffers()),
    )
    assert_buffers_pinned(view.named_buffers(), token)
    reshaped = view.weights.reshape(-1, 1)
    with pytest.raises(ViewIntegrityError, match="rebound or reshaped"):
        assert_buffers_pinned(
            (
                ("indptr", view.indptr),
                ("indices", view.indices),
                ("weights", reshaped),
                ("empty_rows", view.empty_rows),
            ),
            token,
        )
    with pytest.raises(ViewIntegrityError, match="buffer count changed"):
        assert_buffers_pinned((("indptr", view.indptr),), token)


def test_policy_identity_binds_the_cadence() -> None:
    fast = VerificationPolicy(full_coverage_period_steps=10)
    slow = VerificationPolicy(full_coverage_period_steps=1000)
    assert fast.identity_sha256 != slow.identity_sha256
    assert fast.bytes_per_step(1000) == 100
    assert VerificationPolicy(mode="load_only").bytes_per_step(1000) == 0
    with pytest.raises(ViewIntegrityError):
        VerificationPolicy(mode="never")
    with pytest.raises(ViewIntegrityError):
        VerificationPolicy(full_coverage_period_steps=0)


def test_unknown_trigger_is_refused() -> None:
    engine = _engine(_view())
    with pytest.raises(ViewIntegrityError, match="not a declared verification trigger"):
        engine.verify_now("whenever_convenient")
    assert engine.verify_now("run_start") == engine.view.view_sha256


# ---------------------------------------------------------------------------
# B2 -- the amortised checkpoint must not fail open
# ---------------------------------------------------------------------------


def test_b2_persistent_mutation_is_detected_within_the_declared_bound() -> None:
    """B2: the worst case equals the declared period, not twice it.

    The reviewer mutated a byte immediately *after* its offset had been
    consumed by the current pass. The old checkpoint compared only the
    completed-pass digest, so it reported that no earlier than the end of the
    following pass: measured 312 steps at the production default period of
    200, against a theoretical worst case of ``2 * period - 1 = 399``. With
    per-slice comparison the same mutation is reported exactly one full
    coverage cycle later, which is the bound the policy declares.
    """

    period = 8
    view = _view()
    engine = _engine(view, period=period)
    slices = engine.coverage_slice_count
    assert 1 < slices <= period

    target = 11
    index = _slice_index(view, period, _stream_offset(view, "weights", target))
    for _ in range(index + 1):  # steps 1..index+1 read slices 0..index
        engine.step()
    _perturb(view.weights, target)

    latency = 0
    with pytest.raises(ViewIntegrityError, match="modified runtime view"):
        for _ in range(4 * slices):
            latency += 1
            engine.step()
    assert latency == slices, "detection latency is not one coverage cycle"
    assert latency <= period, "the declared bound was breached"
    assert latency < 2 * slices - 1, "this is the old fail-open latency"
    assert engine.run_is_covered is False


def test_b2_mutation_of_slice_zero_right_after_it_was_read_is_still_bounded() -> None:
    """The worst phase for the old scheme: slice 0 changed just after step 1.

    Under pass-completion-only comparison this was the maximal-latency case --
    the change is invisible for the rest of this pass *and* only surfaces when
    the following pass finishes. Here it surfaces exactly one cycle later.
    """

    period = 6
    view = _view()
    engine = _engine(view, period=period)
    slices = engine.coverage_slice_count
    target = 10
    assert _slice_index(view, period, _stream_offset(view, "indices", target)) == 0

    engine.step()  # slice 0 has now been consumed by this pass
    replacement = (int(view.indices[target]) + 1) % view.node_count
    _perturb(view.indices, target, value=replacement)

    latency = 0
    with pytest.raises(ViewIntegrityError, match="modified runtime view"):
        for _ in range(4 * slices):
            latency += 1
            engine.step()
    assert latency == slices
    assert latency <= period


def test_b2_transient_mutation_inside_one_cycle_is_declared_undetectable() -> None:
    """B2: the honest statement of what an amortised checkpoint cannot do.

    The reviewer corrupted ``weights[0]`` at step 88 and restored it at step
    197: 110 steps of wrong dynamics, no fault in 600 steps, every record
    attesting the clean load-time digest. That is not a bug that per-slice
    comparison fixes -- no byte is read on the steps between its own re-hashes
    -- so the module must *say* it rather than claim coverage. This test pins
    both halves: the behaviour, and the disclosure.
    """

    period = 8
    view = _view()
    engine = _engine(view, period=period)
    slices = engine.coverage_slice_count
    index = _slice_index(view, period, _stream_offset(view, "weights", 3))
    for _ in range(index + 1):
        engine.step()

    clean = float(view.weights[3])
    _perturb(view.weights, 3, value=clean * 0.25)
    for _ in range(slices - 1):  # every step before that slice is read again
        engine.step()
    _perturb(view.weights, 3, value=clean)
    for _ in range(3 * slices):  # no fault, ever
        record = engine.step().record
        assert record.view_digest_sha256 == view.view_sha256
    assert engine.fault_reason is None

    disclosure = view_verification.__doc__ or ""
    assert "not detected at all" in disclosure
    assert "Transient mutation" in disclosure


def test_b2_in_place_mutation_of_a_whole_graph_is_detected() -> None:
    """The reviewer's shape-preserving graph swap: same shapes, other numbers."""

    period = 5
    view = _view()
    other = _view(seed=32)
    engine = _engine(view, period=period)
    engine.step()
    view.weights.flags.writeable = True
    view.indices.flags.writeable = True
    view.weights[:] = other.weights[: view.weights.shape[0]]
    view.indices[:] = other.indices[: view.indices.shape[0]]
    view.weights.flags.writeable = False
    view.indices.flags.writeable = False
    with pytest.raises(ViewIntegrityError, match="modified runtime view"):
        for _ in range(engine.coverage_slice_count + 1):
            engine.step()


# ---------------------------------------------------------------------------
# B3 -- empty_rows is dereferenced, therefore verified
# ---------------------------------------------------------------------------


def test_b3_empty_rows_corruption_faults_within_the_declared_bound() -> None:
    """B3: ``empty_rows[0] = 7`` zeroed a real row's drive and never faulted."""

    period = 6
    view = _view()
    engine = _engine(view, period=period)
    assert "empty_rows" in [name for name, _ in view.named_buffers()]
    non_empty = int(np.flatnonzero(np.diff(view.indptr) > 0)[0])
    engine.step()
    _perturb(view.empty_rows, 0, value=non_empty)
    with pytest.raises(ViewIntegrityError, match="modified runtime view"):
        for _ in range(2 * engine.coverage_slice_count + 2):
            engine.step()
    assert engine.run_is_covered is False


def test_b3_unfrozen_empty_rows_is_caught_by_the_pin_on_the_next_step() -> None:
    view = _view()
    engine = _engine(view)
    engine.step()
    view.empty_rows.flags.writeable = True
    with pytest.raises(ViewIntegrityError, match="made writeable"):
        engine.step()


# ---------------------------------------------------------------------------
# B5 -- the kernel and its identity are immutable per engine
# ---------------------------------------------------------------------------


def test_b5_kernel_attribute_cannot_be_rebound() -> None:
    """B5: ``engine.kernel = other`` is the AB-11 idiom; it must not be legal."""

    engine = _engine(_view())
    with pytest.raises(AttributeError):
        engine.kernel = NumpyReduceatKernel()  # type: ignore[misc]
    assert engine.kernel.name == "numpy_add_reduceat"


def test_b5_kernel_substitution_behind_the_property_faults_on_the_next_step() -> None:
    """The reviewer swapped the kernel at step 20 of a 60-step run.

    Every later record still published the C kernel's identity while numpy
    arithmetic produced the trace. Object-identity pinning ends that: the step
    after the swap faults and nothing further is recorded.
    """

    view = _view()
    engine = _engine(view, period=1000)
    for _ in range(20):
        engine.step()
    published = engine.dynamics_identity_sha256
    engine._kernel = NumpyReduceatKernel()
    with pytest.raises(ViewIntegrityError, match="kernel was substituted"):
        engine.step()
    assert engine.dynamics_identity_sha256 == published
    assert engine.run_is_covered is False
    with pytest.raises(ViewIntegrityError, match="engine is faulted"):
        engine.step()


def test_b5_an_in_place_kernel_edit_is_caught_at_the_next_unconditional_trigger() -> None:
    view = _view()
    engine = _engine(view, period=1000)
    engine.step()
    # The object is pinned, so the per-step check is silent -- an in-place edit
    # to what the kernel *reports* is caught the next time the identity is
    # re-derived, which is what every unconditional trigger does.
    object.__setattr__(engine.kernel, "name", "numpy_add_reduceat_but_different")
    with pytest.raises(ViewIntegrityError, match="kernel identity changed"):
        engine.end_run()


# ---------------------------------------------------------------------------
# N1 -- declared triggers exist, and an uncovered run says so
# ---------------------------------------------------------------------------


def test_n1_every_declared_trigger_is_invoked_by_the_engine() -> None:
    """N1: five of seven declared triggers previously had no call site at all."""

    view = _view()
    engine = _engine(view, period=4)
    engine.step()
    engine.set_ablated([3])
    snapshot = engine.snapshot()
    engine.step()
    engine.restore(snapshot)
    engine.end_run()
    invoked = {trigger for trigger, _ in engine.verification_events}
    assert invoked == set(UNCONDITIONAL_VERIFICATION_TRIGGERS) - {"first_fault"}

    faulting = _engine(_view(seed=33), period=3)
    faulting.step()
    _perturb(faulting.view.weights, 2)
    with pytest.raises(ViewIntegrityError):
        for _ in range(4 * faulting.coverage_slice_count):
            faulting.step()
    assert "first_fault" in {trigger for trigger, _ in faulting.verification_events}
    assert faulting.fault_reason is not None


def test_n1_reset_verifies_and_reopens_a_closed_run() -> None:
    view = _view()
    engine = _engine(view, period=4)
    engine.step()
    engine.end_run()
    with pytest.raises(ViewIntegrityError, match="closed by end_run"):
        engine.step()
    engine.reset()
    assert ("run_start", 0) in engine.verification_events[1:]
    engine.step()


def test_n1_a_run_with_uncovered_steps_at_exit_does_not_report_success() -> None:
    """N1: every tracked benchmark row exited with ``uncovered_steps == 152``.

    The same module said "a run with uncovered records at exit is not a passed
    run", so the measurement contradicted the policy it was measured under.
    ``run_is_covered`` is now the machine-readable form of that sentence, and
    ``end_run`` is the declared way to reach it.
    """

    view = _view()
    engine = _engine(view, period=32)
    for _ in range(5):
        engine.step()
    assert engine.uncovered_steps > 0
    assert engine.run_is_covered is False
    engine.end_run()
    assert engine.uncovered_steps == 0
    assert engine.run_is_covered is True


def test_n1_withdrawn_triggers_are_documented_not_silently_dropped() -> None:
    assert set(WITHDRAWN_VERIFICATION_TRIGGERS) == {"stimulation_change", "trace_flush"}
    for name, reason in WITHDRAWN_VERIFICATION_TRIGGERS.items():
        assert name not in UNCONDITIONAL_VERIFICATION_TRIGGERS
        assert len(reason) > 60, f"{name} was withdrawn without a stated reason"
    engine = _engine(_view())
    for name in WITHDRAWN_VERIFICATION_TRIGGERS:
        with pytest.raises(ViewIntegrityError, match="not a declared verification trigger"):
            engine.verify_now(name)


def test_n1_policy_identity_reflects_the_reduced_trigger_set() -> None:
    policy = VerificationPolicy()
    assert policy.unconditional_triggers == UNCONDITIONAL_VERIFICATION_TRIGGERS
    widened = VerificationPolicy(
        unconditional_triggers=(*UNCONDITIONAL_VERIFICATION_TRIGGERS, "trace_flush")
    )
    assert widened.identity_sha256 != policy.identity_sha256


def test_b5_engine_bindings_that_feed_the_identity_are_all_read_only() -> None:
    """B5, generalised: every binding the cached identity depends on is pinned.

    ``kernel`` was the one the reviewer exercised, but ``config`` is the same
    hole -- ``dynamics_identity_sha256`` is derived from it once at
    construction -- and ``view`` and ``policy`` feed the coverage claim. All
    four are properties without setters.
    """

    engine = _engine(_view())
    for attribute, replacement in (
        ("kernel", NumpyReduceatKernel()),
        ("config", engine.config),
        ("policy", VerificationPolicy()),
        ("view", engine.view),
    ):
        with pytest.raises(AttributeError):
            setattr(engine, attribute, replacement)


def test_b5_a_substituted_config_cannot_ride_the_published_identity() -> None:
    from neurocraft_neural.full_engine import ArrayDynamicsConfig

    engine = _engine(_view())
    published = engine.dynamics_identity_sha256
    weaker = ArrayDynamicsConfig(recurrent_gain=0.1)
    assert weaker.identity_sha256(engine.kernel_identity) != published
    with pytest.raises(AttributeError):
        engine.config = weaker  # type: ignore[misc]
    assert engine.config.recurrent_gain == 0.9


def test_first_fault_verification_reports_whether_the_bytes_changed() -> None:
    """The ``first_fault`` re-hash distinguishes rebinding from mutation."""

    mutated_view = _view()
    mutated = _engine(mutated_view, period=3)
    mutated.step()
    _perturb(mutated_view.weights, 4)
    with pytest.raises(ViewIntegrityError):
        for _ in range(4 * mutated.coverage_slice_count):
            mutated.step()
    assert mutated.fault_view_sha256 not in (None, mutated_view.view_sha256)

    rebound_view = _view()
    rebound = _engine(rebound_view, period=3)
    rebound.step()
    twin = rebound_view.indices.copy()
    twin.flags.writeable = False
    object.__setattr__(rebound_view, "indices", twin)
    with pytest.raises(ViewIntegrityError, match="rebound or reshaped"):
        rebound.step()
    assert rebound.fault_view_sha256 == rebound_view.view_sha256
