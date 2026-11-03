"""Array-native full-graph neural backend (Phase 1, uniform delay 1).

This is **not** an optimisation of :class:`~neurocraft_neural.engine.NeuralEngine`.
That engine is the correctness oracle at fixture scale and stays byte-frozen.
This module is a separate backend with a separate dynamics identity, and every
trace it produces says so.

Update law -- identical to the reference
----------------------------------------
::

    a[n+1] = leak * a[n]
           + (1 - leak) * tanh(recurrent_gain * (W @ a[n-1])
                               + sensory_gain * s[n] + p[n] + baseline)
    leak   = exp(-neural_step_ms / tau_ms)

Note the ``a[n-1]``: at delay 1 the reference reads history row 0, which holds
the state committed one index *before* the current one
(``engine.py`` lines 227-237 and 265). This backend keeps a single ``previous``
vector rather than a delay ring, which is exactly equivalent at ``d = 1``.

Delay scope
-----------
Phase 1 implements **uniform delay 1 only**, and says so in its identity
(``delay_policy = "uniform_1"``). Heterogeneous delays are not partially
implemented here and must not be emulated by a caller: they turn the gather
into a 2-D read over a ``(D, N)`` history buffer (85 MB at the allowed maximum
of 64), destroy the L2 residency the kernel's speed depends on, and require
bucketing edges into per-delay CSR passes. That re-permutes the matrix and
therefore changes the summation order, which is a *different dynamics
identity*, not a configuration flag. It is a separate future backend.

Verification policy -- what is checked, and when
------------------------------------------------
The reference re-derives graph and overlays every step. At full scale that is
about 98% of the step cost, so it cannot be here. Instead:

* **Once, at construction:** the loader has already bound each artifact array's
  SHA-256 to its mapping; this engine additionally recomputes ``view_sha256``
  over ``indptr || indices || weights || empty_rows`` and refuses to start on a
  mismatch, re-derives the view's structural validation record (including
  ``max(indices) < node_count``) from the buffers rather than trusting the
  view object, and pins the sparse kernel by object identity. That construction
  check *is* the declared ``run_start`` verification and is recorded as such.
* **Every step, O(1) (microseconds):** :func:`assert_buffers_pinned` re-checks
  base address, byte length, dtype, shape and writeability of all four
  buffers, and the bound kernel is re-checked against the pinned object.
  Rebinding, resizing, dtype swaps, re-enabled writeability, view substitution
  and kernel substitution are all caught on the very next step.
* **Amortised, bounded:** the view is re-hashed in fixed per-step slices, each
  compared against its recorded digest *on the step that reads it*. A
  persistent in-place mutation is therefore detected within
  ``full_coverage_period_steps`` steps. A mutation reverted before its slice is
  next read is **not detected at all**; that limit is stated in
  :mod:`neurocraft_neural.view_verification` and must not be described as
  coverage.
* **Unconditionally:** at run start (construction and :meth:`reset`), at run
  end (:meth:`end_run`), on ``restore``, around every lesion change, and once
  on the first fault of any kind. Those five are exactly the triggers hashed
  into ``verification_policy_sha256``; every one has a call site here.

Out-of-range corruption of ``indptr`` or ``indices`` after construction is not
a checkpoint problem at all -- the kernel dereferences the buffers at the top
of the step, before any checkpoint runs -- so both kernels range-check every
offset and index they dereference and refuse the call. It is contained, not
undefined behaviour.

Every step record carries ``view_digest_sha256``, ``view_validation_sha256``,
``verified_through_step`` and ``verification_policy_sha256``, all inside the
hashed execution material, so a replayer can refuse any record whose coverage
claim does not hold. On any mismatch the engine faults permanently and refuses
further steps: records after ``verified_through_step`` are uncovered, and a run
with uncovered records at exit is not a passed run --
:attr:`ArrayNativeEngine.run_is_covered` is the machine-readable form of that
sentence, and :meth:`end_run` is how a clean run reaches it. See
:mod:`neurocraft_neural.view_verification` for the full AB-05 / AB-06 / AB-11
argument.

Stimulation identity
--------------------
``step`` derives ``encoded_stimulation_sha256`` and
``direct_stimulation_sha256`` from the arrays it actually executes. A caller
may additionally pass a *provenance* digest naming where a vector came from
(a protocol ``Stimulation`` identity, say); that value is recorded under its
own key and can never stand in for the executed vector's digest. Executing
vector A while attesting vector B is therefore not expressible.

Digest caching -- recomputation, not verification
-------------------------------------------------
Three of the reference's five per-step digests are provably recomputations of
identical inputs, and are cached rather than recomputed:

* ``before`` is by construction the previous step's ``after``;
* at delay 1, ``delay_before`` is the previous step's ``delay_after``, and the
  single history row's digest is the previous step's activity digest;
* the ablation-mask digest changes only when a lesion changes.

These are bit-identical substitutions, and dedicated tests assert equality
against :func:`~neurocraft_neural.model.delay_history_digest` and
:func:`~neurocraft_neural.model.make_neural_state`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from hashlib import sha256
from typing import Any, NoReturn

import numpy as np
from neurocraft_connectome.model import sha256_json
from numpy.typing import NDArray

from neurocraft_neural.full_records import (
    ARRAY_NATIVE_BACKEND_ID,
    DELAY_POLICY,
    ArrayDynamicsConfig,
    ArrayNeuralState,
    ArraySnapshot,
    ArrayStepRecord,
    ArrayStepResult,
    _activity_digest,
    _state_material,
    array_snapshot_digest,
)
from neurocraft_neural.full_view import FullRuntimeView, ViewValidation
from neurocraft_neural.model import (
    ZERO_SHA256,
    NeuralConfigurationError,
    NeuralIntegrityError,
    NeuralStepRecord,
)
from neurocraft_neural.spmv_kernel import KernelIdentity, SpmvKernel, load_kernel
from neurocraft_neural.view_verification import (
    IncrementalViewDigest,
    VerificationPolicy,
    VerifiedView,
    ViewIntegrityError,
    assert_buffers_pinned,
    capture_pins,
    hash_buffers,
)

__all__ = [
    "ARRAY_NATIVE_BACKEND_ID",
    "DELAY_POLICY",
    "ArrayDynamicsConfig",
    "ArrayNativeEngine",
    "ArrayNeuralState",
    "ArraySnapshot",
    "ArrayStepRecord",
    "ArrayStepResult",
]


class ArrayNativeEngine:
    """Array-native engine over a frozen :class:`FullRuntimeView`."""

    def __init__(
        self,
        view: FullRuntimeView,
        *,
        config: ArrayDynamicsConfig | None = None,
        kernel: SpmvKernel | None = None,
        verification_policy: VerificationPolicy | None = None,
    ) -> None:
        # ``view``, ``config``, ``policy`` and ``kernel`` are all exposed as
        # read-only properties over these private slots. Every one of them
        # feeds the cached ``dynamics_identity_sha256`` or the pinned buffer
        # token, so a setter would be a way to change what executes while the
        # trace keeps publishing the original identity -- the AB-11
        # runtime-substitution idiom. They are bound once, here.
        self._view = view
        self._config = config or ArrayDynamicsConfig()
        # The kernel is bound once. ``self.kernel`` is a read-only property and
        # ``__pinned_kernel`` is a name-mangled second reference, so the AB-11
        # runtime-substitution idiom -- rebind the engine's kernel mid-run while
        # every record keeps publishing the original identity -- cannot happen
        # silently: it either raises on assignment or faults on the next step.
        self._kernel = kernel if kernel is not None else load_kernel()
        self.__pinned_kernel = self._kernel
        self._policy = verification_policy or VerificationPolicy()
        self._kernel_identity = self._kernel.identity
        self._dynamics_identity_sha256 = self._config.identity_sha256(self._kernel_identity)

        node_count = view.node_count
        self._node_count = node_count
        self._activity = np.zeros(node_count, dtype=np.float64)
        self._previous = np.zeros(node_count, dtype=np.float64)
        self._recurrent = np.empty(node_count, dtype=np.float64)
        self._drive = np.empty(node_count, dtype=np.float64)
        self._next = np.empty(node_count, dtype=np.float64)
        self._gather = np.empty(node_count, dtype=np.float64)
        self._scratch = np.empty(node_count, dtype=np.float64)
        self._ablated = np.zeros(node_count, dtype=bool)
        self._any_ablated = False
        self._step_index = 0
        self._trace_sha256 = ZERO_SHA256
        self._faulted: str | None = None
        self._fault_view_sha256: str | None = None
        self._run_ended = False
        self._verification_events: list[tuple[str, int]] = []

        # Unconditional load-time verification -- the declared ``run_start``
        # trigger. Both the digest and the structural validation record are
        # recomputed from the frozen buffers, never taken on trust from the
        # view object, so a hand-built view cannot assert an index bound it
        # does not satisfy.
        observed = hash_buffers(view.named_buffers())
        if observed != view.view_sha256:
            raise ViewIntegrityError("runtime view digest does not match its buffers at load")
        derived = ViewValidation.derive(
            indptr=view.indptr, indices=view.indices, empty_rows=view.empty_rows
        )
        if derived != view.validation:
            raise ViewIntegrityError(
                "runtime view validation record does not describe its buffers at load"
            )
        if derived.edge_count and derived.max_observed_index >= derived.index_upper_bound_exclusive:
            raise ViewIntegrityError(
                "runtime view presynaptic index bound is violated at load: "
                f"max index {derived.max_observed_index} >= node count "
                f"{derived.index_upper_bound_exclusive}"
            )
        self._view_validation_sha256 = derived.identity_sha256
        self._token = VerifiedView(
            view_sha256=view.view_sha256,
            verified_at_step=0,
            buffers=capture_pins(view.named_buffers()),
        )
        self._verified_through_step = 0
        self._verification_events.append(("run_start", 0))
        budget = self._policy.bytes_per_step(view.buffer_bytes)
        self._checkpoint = (
            IncrementalViewDigest(view.named_buffers(), bytes_per_step=max(1, budget))
            if self._policy.mode == "amortized_in_thread" and view.buffer_bytes > 0
            else None
        )
        self._refresh_digests()

    # ------------------------------------------------------------------ identity

    @property
    def dynamics_identity_sha256(self) -> str:
        return self._dynamics_identity_sha256

    @property
    def view(self) -> FullRuntimeView:
        """The bound runtime view. Read-only by construction."""

        return self._view

    @property
    def config(self) -> ArrayDynamicsConfig:
        """The bound dynamics configuration. Read-only by construction.

        Swapping this would change ``recurrent_gain``, ``leak`` or a gain while
        ``dynamics_identity_sha256`` -- which is cached from it at construction
        -- kept publishing the original value. Same hole as a mutable kernel,
        same closure.
        """

        return self._config

    @property
    def policy(self) -> VerificationPolicy:
        """The bound verification policy. Read-only by construction."""

        return self._policy

    @property
    def kernel(self) -> SpmvKernel:
        """The bound sparse kernel. Read-only by construction.

        There is deliberately no setter: ``engine.kernel = other`` raises
        ``AttributeError`` rather than silently decoupling the executed
        arithmetic from the ``kernel_identity_sha256`` every record publishes.
        """

        return self._kernel

    @property
    def kernel_identity(self) -> KernelIdentity:
        return self._kernel_identity

    @property
    def verification_events(self) -> tuple[tuple[str, int], ...]:
        """``(trigger, step_index)`` for every unconditional verification run."""

        return tuple(self._verification_events)

    @property
    def fault_reason(self) -> str | None:
        return self._faulted

    @property
    def fault_view_sha256(self) -> str | None:
        """View digest observed by the ``first_fault`` verification, if any.

        Compare it against ``view.view_sha256`` to separate "the bytes were
        changed" from "the buffers were rebound but the bytes are intact".
        """

        return self._fault_view_sha256

    @property
    def coverage_slice_count(self) -> int:
        """Steps in one full coverage cycle; 0 when no checkpoint is running.

        This is the true detection bound for a persistent in-place mutation:
        every byte is re-read once per this many steps. It is at most
        ``policy.full_coverage_period_steps``.
        """

        return 0 if self._checkpoint is None else self._checkpoint.slice_count

    @property
    def run_is_covered(self) -> bool:
        """True only when the run is unfaulted and every record is covered.

        This is the machine-readable form of "a run with uncovered records at
        exit is not a passed run". A run that never calls :meth:`end_run`
        finishes mid-pass and is *not* covered, by design.
        """

        return self._faulted is None and self._step_index == self._verified_through_step

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def trace_sha256(self) -> str:
        return self._trace_sha256

    @property
    def verified_through_step(self) -> int:
        return self._verified_through_step

    @property
    def uncovered_steps(self) -> int:
        return self._step_index - self._verified_through_step

    def activity(self) -> NDArray[np.float64]:
        """Read-only copy of the committed activity vector."""

        copy = self._activity.copy()
        copy.flags.writeable = False
        return copy

    def _state_digest_now(self) -> str:
        provenance = self._view.provenance
        return sha256_json(
            _state_material(
                step_index=self._step_index,
                graph_identity_sha256=provenance.graph_identity_sha256,
                dynamics_identity_sha256=self._dynamics_identity_sha256,
                sign_overlay_sha256=provenance.sign_overlay_sha256,
                delay_overlay_sha256=provenance.delay_overlay_sha256,
                weight_overlay_sha256=provenance.weight_overlay_sha256,
                topology_overlay_sha256=provenance.topology_overlay_sha256,
                activity_sha256=self._activity_sha256,
            )
        )

    def current_state(self) -> ArrayNeuralState:
        """The committed state identity, from the cached digest.

        ``state_sha256`` is recomputed only when the step index, the activity
        vector or an overlay identity changes -- never once per call and never
        twice per step. Within a step the ``before`` identity is literally the
        previous step's ``after``, which is the first of the three cached
        digests documented in the module header.
        """

        provenance = self._view.provenance
        return ArrayNeuralState(
            step_index=self._step_index,
            graph_identity_sha256=provenance.graph_identity_sha256,
            dynamics_identity_sha256=self._dynamics_identity_sha256,
            sign_overlay_sha256=provenance.sign_overlay_sha256,
            delay_overlay_sha256=provenance.delay_overlay_sha256,
            weight_overlay_sha256=provenance.weight_overlay_sha256,
            topology_overlay_sha256=provenance.topology_overlay_sha256,
            activity_sha256=self._activity_sha256,
            state_sha256=self._state_sha256,
        )

    # -------------------------------------------------------------- verification

    def _ablation_digest(self) -> str:
        return sha256(self._ablated.astype(np.uint8).tobytes()).hexdigest()

    def _refresh_digests(self) -> None:
        """Rebuild every cached digest from scratch after a non-step mutation."""

        self._activity_sha256 = _activity_digest(self._activity)
        self._history_sha256 = _activity_digest(self._previous)
        self._mask_sha256 = self._ablation_digest()
        self._state_sha256 = self._state_digest_now()
        self._delay_sha256 = sha256_json([self._history_sha256])

    def verify_now(self, trigger: str) -> str:
        """Unconditional full re-verification; returns the observed digest.

        Re-derives the live kernel's identity as well as the view digest: the
        per-step check only pins the kernel *object*, while this also catches
        an in-place edit of the kernel's own build binding.
        """

        if trigger not in self._policy.unconditional_triggers:
            raise ViewIntegrityError(f"{trigger!r} is not a declared verification trigger")
        self._assert_kernel_is_pinned()
        if self._kernel.identity.identity_sha256 != self._kernel_identity.identity_sha256:
            self._fault(f"the bound sparse kernel identity changed (trigger {trigger})")
        assert_buffers_pinned(self._view.named_buffers(), self._token)
        observed = hash_buffers(self._view.named_buffers())
        if observed != self._token.view_sha256:
            self._fault(f"runtime view digest changed (trigger {trigger})")
        if self._checkpoint is not None:
            self._checkpoint.restart()
        self._verified_through_step = self._step_index
        self._verification_events.append((trigger, self._step_index))
        return observed

    def end_run(self) -> str:
        """Declared ``run_end`` trigger: verify, then close the run.

        A run that ends without this call stops mid-coverage-pass and reports
        ``uncovered_steps > 0``; that is not a passed run and must not be
        reported as one. After it returns, :meth:`step` refuses until
        :meth:`reset`.
        """

        observed = self.verify_now("run_end")
        self._run_ended = True
        return observed

    def _assert_kernel_is_pinned(self) -> None:
        if self._kernel is not self.__pinned_kernel:
            self._fault("the bound sparse kernel was substituted after construction")

    def _fault(self, reason: str) -> NoReturn:
        """Fault permanently, running the declared ``first_fault`` verification.

        The full re-hash happens exactly once per engine: it characterises the
        first fault (were the bytes changed, or was it a rebinding?) and is
        recorded in :attr:`verification_events`. Later faults reuse it.
        """

        if self._faulted is None:
            observed = hash_buffers(self._view.named_buffers())
            self._fault_view_sha256 = observed
            detail = (
                "view bytes still match the load-time digest"
                if observed == self._token.view_sha256
                else f"view digest is now {observed}"
            )
            self._faulted = f"{reason} ({detail})"
            self._verification_events.append(("first_fault", self._step_index))
        raise ViewIntegrityError(
            f"{self._faulted}; records after step {self._verified_through_step} are uncovered"
        )

    def _verify_step(self) -> None:
        if self._faulted is not None:
            raise ViewIntegrityError(f"engine is faulted: {self._faulted}")
        if self._run_ended:
            raise ViewIntegrityError("this run was closed by end_run(); call reset() to reopen")
        self._assert_kernel_is_pinned()
        try:
            assert_buffers_pinned(self._view.named_buffers(), self._token)
        except ViewIntegrityError as exc:
            self._fault(str(exc))
        if self._checkpoint is None:
            return
        outcome = self._checkpoint.advance()
        if not outcome.matched:
            self._fault(
                "amortised checkpoint observed a modified runtime view in declared-stream "
                f"bytes [{outcome.start_offset}, {outcome.end_offset})"
            )
        if outcome.completed_view_sha256 is None:
            return
        if outcome.completed_view_sha256 != self._token.view_sha256:
            self._fault("amortised checkpoint observed a modified runtime view")
        self._verified_through_step = self._step_index

    # ---------------------------------------------------------------- mutation

    def reset(self, activity: Sequence[float] | NDArray[Any] | None = None) -> ArrayNeuralState:
        """Reset to index zero; the single delay row is zeroed with it."""

        proposed = (
            np.zeros(self._node_count, dtype=np.float64)
            if activity is None
            else self._validated_state_vector(
                activity, "initial activity", NeuralConfigurationError
            )
        )
        # Reject invalid requests and an invalid bound view before changing live state.
        self.verify_now("run_start")
        self._activity[:] = proposed
        self._previous[:] = 0.0
        self._ablated[:] = False
        self._any_ablated = False
        self._step_index = 0
        self._trace_sha256 = ZERO_SHA256
        self._run_ended = False
        self._refresh_digests()
        self.verify_now("run_start")
        return self.current_state()

    def set_ablated(self, indices: Sequence[int], *, ablated: bool = True) -> None:
        """Apply a persistent lesion, clearing state and history like the reference."""

        if not isinstance(ablated, bool):
            raise NeuralConfigurationError("ablated flag must be boolean")
        positions: list[int] = []
        for index in indices:
            if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
                raise NeuralConfigurationError("ablation indices must be exact integers")
            position = int(index)
            if not 0 <= position < self._node_count:
                raise NeuralConfigurationError(f"ablation index {position} is out of range")
            positions.append(position)
        self.verify_now("lesion_change")
        for position in positions:
            self._ablated[position] = ablated
            if ablated:
                self._activity[position] = 0.0
                self._previous[position] = 0.0
        self._any_ablated = bool(self._ablated.any())
        self._refresh_digests()
        self.verify_now("lesion_change")

    # -------------------------------------------------------------------- step

    def step(
        self,
        *,
        sensory: NDArray[np.float64] | None = None,
        sensory_provenance_sha256: str = ZERO_SHA256,
        direct: NDArray[np.float64] | None = None,
        direct_provenance_sha256: str = ZERO_SHA256,
    ) -> ArrayStepResult:
        """Advance one neural interval atomically.

        ``sensory`` and ``direct`` are dense float64 vectors of length
        ``node_count``. Their identity digests are **derived here from the
        arrays that are about to be executed**, never accepted from the caller:
        a caller-supplied digest could otherwise execute vector A while
        attesting vector B. The optional ``*_provenance_sha256`` arguments name
        where a vector came from -- typically a protocol
        ``EncodedStimulation.identity_sha256`` or
        ``Stimulation.identity_sha256`` -- and are bound into the execution
        material under their own keys, alongside but never instead of the
        derived digests.
        """

        self._verify_step()
        sensory_values = None if sensory is None else self._validated(sensory, "sensory")
        direct_values = None if direct is None else self._validated(direct, "direct stimulation")
        sensory_sha256 = ZERO_SHA256 if sensory_values is None else _activity_digest(sensory_values)
        direct_sha256 = ZERO_SHA256 if direct_values is None else _activity_digest(direct_values)
        step_before = self._step_index
        before_state_sha256 = self._state_sha256
        delay_before_sha256 = self._delay_sha256

        source = self._previous
        if self._any_ablated:
            # Assign rather than multiply by a 0/1 mask: `-x * 0.0` is -0.0,
            # while the reference simply omits the term. Omitting a term and
            # adding an exact +0.0 are additively identical in IEEE-754.
            self._gather[:] = self._previous
            self._gather[self._ablated] = 0.0
            source = self._gather
        self.kernel.matvec(
            indptr=self._view.indptr,
            indices=self._view.indices,
            values=self._view.weights,
            x=source,
            out=self._recurrent,
            empty_rows=self._view.empty_rows,
        )

        # Addition order mirrors engine.py lines 246-251 exactly:
        #   ((recurrent_gain * r + sensory_gain * s) + p) + baseline.
        # A `None` sensory or direct vector is skipped rather than added as an
        # explicit zero vector. That is exact, not an approximation: adding
        # +0.0 can only change a -0.0 accumulator to +0.0, and the baseline
        # term -- which the reference also applies last -- is always added
        # here, so any such -0.0 is normalised at the same point in both.
        np.multiply(self._recurrent, self._config.recurrent_gain, out=self._drive)
        if sensory_values is not None:
            np.multiply(sensory_values, self._config.sensory_gain, out=self._scratch)
            self._drive += self._scratch
        if direct_values is not None:
            self._drive += direct_values
        self._drive += self._config.baseline
        if not bool(np.isfinite(self._drive).all()):
            raise NeuralIntegrityError("non-finite drive proposed by the array-native backend")

        leak = self._config.leak
        np.tanh(self._drive, out=self._next)
        np.multiply(self._next, 1.0 - leak, out=self._next)
        np.multiply(self._activity, leak, out=self._scratch)
        self._next += self._scratch
        if self._any_ablated:
            self._next[self._ablated] = 0.0
        if not bool(np.isfinite(self._next).all()):
            raise NeuralIntegrityError("non-finite state proposed by the array-native backend")
        if bool((self._next < -1.0).any()) or bool((self._next > 1.0).any()):
            raise NeuralIntegrityError("bounded recurrence produced out-of-range state")

        # Cached digests: the new history row is the old activity, so its
        # digest is the activity digest already held.
        delay_after_sha256 = sha256_json([self._activity_sha256])
        self._previous, self._activity = self._activity, self._previous
        self._activity[:] = self._next
        self._history_sha256 = self._activity_sha256
        self._delay_sha256 = delay_after_sha256
        self._activity_sha256 = _activity_digest(self._activity)
        step_after = step_before + 1
        self._step_index = step_after
        self._state_sha256 = self._state_digest_now()
        after_state = self.current_state()

        execution_sha256 = sha256_json(
            {
                "ablation_mask_sha256": self._mask_sha256,
                "delay_after_sha256": delay_after_sha256,
                "delay_before_sha256": delay_before_sha256,
                "direct_stimulation_provenance_sha256": direct_provenance_sha256,
                "direct_stimulation_sha256": direct_sha256,
                "encoded_stimulation_provenance_sha256": sensory_provenance_sha256,
                "encoded_stimulation_sha256": sensory_sha256,
                "state_after_sha256": after_state.state_sha256,
                "state_before_sha256": before_state_sha256,
                "step_after": step_after,
                "step_before": step_before,
                "verification_policy_sha256": self._policy.identity_sha256,
                "verified_through_step": self._verified_through_step,
                "view_digest_sha256": self._token.view_sha256,
                "view_validation_sha256": self._view_validation_sha256,
            }
        )
        trace_sha256 = sha256(
            bytes.fromhex(self._trace_sha256) + bytes.fromhex(execution_sha256)
        ).hexdigest()
        record = ArrayStepRecord(
            neural=NeuralStepRecord(
                step_before=step_before,
                step_after=step_after,
                state_before_sha256=before_state_sha256,
                state_after_sha256=after_state.state_sha256,
                delay_before_sha256=delay_before_sha256,
                delay_after_sha256=delay_after_sha256,
                encoded_stimulation_sha256=sensory_sha256,
                direct_stimulation_sha256=direct_sha256,
                ablation_mask_sha256=self._mask_sha256,
                execution_sha256=execution_sha256,
                previous_trace_sha256=self._trace_sha256,
                trace_sha256=trace_sha256,
            ),
            view_digest_sha256=self._token.view_sha256,
            view_validation_sha256=self._view_validation_sha256,
            verified_through_step=self._verified_through_step,
            verification_policy_sha256=self._policy.identity_sha256,
        )
        self._trace_sha256 = trace_sha256
        return ArrayStepResult(state=after_state, record=record)

    def _validated(self, values: NDArray[np.float64], label: str) -> NDArray[np.float64]:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (self._node_count,):
            raise NeuralConfigurationError(f"{label} vector length mismatch")
        if not bool(np.isfinite(array).all()):
            raise NeuralConfigurationError(f"{label} vector must be finite")
        return array

    # ---------------------------------------------------------------- snapshots

    def snapshot(self) -> ArraySnapshot:
        snapshot = ArraySnapshot(
            dynamics_identity_sha256=self._dynamics_identity_sha256,
            view_sha256=self._token.view_sha256,
            step_index=self._step_index,
            activity=tuple(float(value) for value in self._activity),
            previous=tuple(float(value) for value in self._previous),
            ablation_mask=tuple(bool(value) for value in self._ablated),
            trace_sha256=self._trace_sha256,
        )
        return replace(snapshot, snapshot_sha256=array_snapshot_digest(snapshot))

    def restore(
        self, snapshot: ArraySnapshot, *, allow_legacy_snapshot: bool = False
    ) -> ArrayNeuralState:
        """Restore validated state atomically.

        Legacy snapshots lack a content digest and require explicit opt-in. The
        opt-in does not attest their historical contents; all state, identity and
        metadata validation still applies before any live buffer changes.
        """

        if not isinstance(snapshot, ArraySnapshot):
            raise NeuralIntegrityError("snapshot must be an ArraySnapshot")
        if not isinstance(allow_legacy_snapshot, bool):
            raise NeuralIntegrityError("allow_legacy_snapshot must be boolean")
        if snapshot.dynamics_identity_sha256 != self._dynamics_identity_sha256:
            raise NeuralIntegrityError("snapshot dynamics identity mismatch")
        if snapshot.view_sha256 != self._token.view_sha256:
            raise NeuralIntegrityError("snapshot runtime view identity mismatch")
        if (
            isinstance(snapshot.step_index, (bool, np.bool_))
            or not isinstance(snapshot.step_index, (int, np.integer))
            or snapshot.step_index < 0
        ):
            raise NeuralIntegrityError("snapshot step index must be a non-negative integer")
        if not _is_sha256(snapshot.trace_sha256):
            raise NeuralIntegrityError("snapshot trace identity must be lowercase SHA-256")
        activity = self._validated_state_vector(
            snapshot.activity, "snapshot current activity", NeuralIntegrityError
        )
        previous = self._validated_state_vector(
            snapshot.previous, "snapshot delay history", NeuralIntegrityError
        )
        try:
            mask = np.asarray(snapshot.ablation_mask)
        except (TypeError, ValueError) as exc:
            raise NeuralIntegrityError("snapshot ablation mask is invalid") from exc
        if mask.shape != (self._node_count,):
            raise NeuralIntegrityError("snapshot ablation mask length mismatch")
        if mask.size and mask.dtype != np.bool_:
            raise NeuralIntegrityError("snapshot ablation mask must contain booleans")
        mask = np.array(mask, dtype=bool, copy=True)
        if bool((activity[mask] != 0.0).any()) or bool((previous[mask] != 0.0).any()):
            raise NeuralIntegrityError("snapshot ablated neurons must have zero state and history")
        validated = replace(
            snapshot,
            step_index=int(snapshot.step_index),
            activity=tuple(float(value) for value in activity),
            previous=tuple(float(value) for value in previous),
            ablation_mask=tuple(bool(value) for value in mask),
        )
        if snapshot.snapshot_sha256 == "":
            if not allow_legacy_snapshot:
                raise NeuralIntegrityError("legacy snapshot requires allow_legacy_snapshot=True")
        elif not _is_sha256(snapshot.snapshot_sha256) or (
            array_snapshot_digest(validated) != snapshot.snapshot_sha256
        ):
            raise NeuralIntegrityError("snapshot content digest mismatch")
        self.verify_now("snapshot_restore")
        self._activity[:] = activity
        self._previous[:] = previous
        self._ablated[:] = mask
        self._any_ablated = bool(mask.any())
        self._step_index = validated.step_index
        self._trace_sha256 = snapshot.trace_sha256
        self._refresh_digests()
        self.verify_now("snapshot_restore")
        return self.current_state()

    def _validated_state_vector(
        self,
        values: Sequence[float] | NDArray[Any],
        label: str,
        error: type[ValueError] | type[NeuralIntegrityError],
    ) -> NDArray[np.float64]:
        """Copy a complete bounded numeric vector before a state transition."""

        try:
            array = np.asarray(values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise error(f"{label} must be a numeric vector of length {self._node_count}") from exc
        if array.shape != (self._node_count,):
            raise error(f"{label} vector length mismatch")
        if array.dtype.kind not in "fiu":
            raise error(f"{label} must contain real numbers")
        try:
            proposed = np.array(array, dtype=np.float64, copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise error(f"{label} must contain real numbers") from exc
        if not bool(np.isfinite(proposed).all()):
            raise error(f"{label} must be finite")
        if bool((np.abs(proposed) > 1.0).any()):
            raise error(f"{label} must lie in [-1, 1]")
        return proposed


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
