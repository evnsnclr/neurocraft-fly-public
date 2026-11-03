"""Value types and hashed material for the array-native full-graph backend.

Split out of :mod:`neurocraft_neural.full_engine` so each file stays small and
one concern per module: everything here is a frozen record or a pure digest
helper with no engine state. :mod:`neurocraft_neural.full_engine` re-exports
every public name, so the import path callers already use keeps working.

The dynamics identity lives here because it is a *description* of arithmetic --
config fields plus the kernel identity, including that kernel's library digest
-- rather than anything the engine mutates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Final

import numpy as np
from neurocraft_connectome.model import sha256_json
from numpy.typing import NDArray

from neurocraft_neural.full_view import view_construction_sha256
from neurocraft_neural.model import (
    ZERO_SHA256,
    NeuralConfigurationError,
    NeuralStepRecord,
    _finite_float,
    _float_hex,
)
from neurocraft_neural.spmv_kernel import KernelIdentity

ARRAY_NATIVE_BACKEND_ID: Final = "array_native_csr_float64_uniform_delay1"
DELAY_POLICY: Final = "uniform_1"


@dataclass(frozen=True, slots=True)
class ArrayDynamicsConfig:
    """Versioned assumptions for the array-native backend.

    Field values mirror :class:`~neurocraft_neural.model.DynamicsConfig` so the
    two backends are comparable, but ``backend`` and ``delay_policy`` are
    locked to this backend's own values and can never collide with the
    reference identity.
    """

    neural_step_ms: float = 10.0
    tau_ms: float = 50.0
    recurrent_gain: float = 0.9
    sensory_gain: float = 1.0
    baseline: float = 0.0
    default_delay_steps: int = 1
    activation: str = "tanh"
    normalization: str = "incoming_raw_count_l1"
    precision: str = "float64"
    backend: str = ARRAY_NATIVE_BACKEND_ID
    delay_policy: str = DELAY_POLICY
    empty_row_policy: str = "zero"
    noise: str = "none"

    def __post_init__(self) -> None:
        for name in ("neural_step_ms", "tau_ms", "recurrent_gain", "sensory_gain", "baseline"):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))
        if self.neural_step_ms <= 0 or self.tau_ms <= 0:
            raise NeuralConfigurationError("neural_step_ms and tau_ms must be positive")
        if self.recurrent_gain < 0:
            raise NeuralConfigurationError("recurrent_gain must be non-negative")
        if self.default_delay_steps != 1 or self.delay_policy != DELAY_POLICY:
            raise NeuralConfigurationError(
                "the array-native Phase 1 backend implements uniform delay 1 only; "
                "heterogeneous delays are a separate backend identity"
            )
        fixed = {
            "activation": "tanh",
            "normalization": "incoming_raw_count_l1",
            "precision": "float64",
            "backend": ARRAY_NATIVE_BACKEND_ID,
            "empty_row_policy": "zero",
            "noise": "none",
        }
        for field, expected in fixed.items():
            if getattr(self, field) != expected:
                raise NeuralConfigurationError(
                    f"array-native backend requires {field}={expected!r}"
                )

    @property
    def leak(self) -> float:
        return math.exp(-self.neural_step_ms / self.tau_ms)

    def identity_material(self, kernel_identity: KernelIdentity) -> dict[str, Any]:
        """Exact hashed material for this backend's dynamics identity."""

        return {
            "activation": self.activation,
            "backend": self.backend,
            "baseline": _float_hex(self.baseline),
            "default_delay_steps": self.default_delay_steps,
            "delay_policy": self.delay_policy,
            "empty_row_policy": self.empty_row_policy,
            "kernel_identity_sha256": kernel_identity.identity_sha256,
            "kernel_source_sha256": kernel_identity.material.get(
                "kernel_source_sha256", ZERO_SHA256
            ),
            "neural_step_ms": _float_hex(self.neural_step_ms),
            "noise": self.noise,
            "normalization": self.normalization,
            "numeric_reproducibility_class": kernel_identity.numeric_reproducibility_class,
            "precision": self.precision,
            "recurrent_gain": _float_hex(self.recurrent_gain),
            "sensory_gain": _float_hex(self.sensory_gain),
            "summation_kernel": kernel_identity.kernel,
            "tau_ms": _float_hex(self.tau_ms),
            "thread_count_affects_results": False,
            "view_construction_sha256": view_construction_sha256(),
        }

    def identity_sha256(self, kernel_identity: KernelIdentity) -> str:
        return sha256_json(self.identity_material(kernel_identity))


@dataclass(frozen=True, slots=True)
class ArrayNeuralState:
    """Committed state identity without materialising 166,700 Python floats."""

    step_index: int
    graph_identity_sha256: str
    dynamics_identity_sha256: str
    sign_overlay_sha256: str
    delay_overlay_sha256: str
    weight_overlay_sha256: str
    topology_overlay_sha256: str
    activity_sha256: str
    state_sha256: str


@dataclass(frozen=True, slots=True)
class ArrayStepRecord:
    """Reference step record plus the view-coverage fields a replayer needs."""

    neural: NeuralStepRecord
    view_digest_sha256: str
    view_validation_sha256: str
    verified_through_step: int
    verification_policy_sha256: str

    @property
    def trace_sha256(self) -> str:
        return self.neural.trace_sha256


@dataclass(frozen=True, slots=True)
class ArrayStepResult:
    state: ArrayNeuralState
    record: ArrayStepRecord


@dataclass(frozen=True, slots=True)
class ArraySnapshot:
    """Complete restorable engine state with a versioned content digest.

    The empty digest default preserves construction of legacy snapshots. Restoring
    one requires an explicit opt-in because those bytes have no content attestation.
    """

    dynamics_identity_sha256: str
    view_sha256: str
    step_index: int
    activity: tuple[float, ...]
    previous: tuple[float, ...]
    ablation_mask: tuple[bool, ...]
    trace_sha256: str
    snapshot_sha256: str = ""


def array_snapshot_digest(snapshot: ArraySnapshot) -> str:
    """Bind state, delayed history, lesion mask and trace to their runtime view."""

    return sha256_json(
        {
            "schema": "neurocraft.array_snapshot.v1",
            "dynamics_identity_sha256": snapshot.dynamics_identity_sha256,
            "view_sha256": snapshot.view_sha256,
            "step_index": snapshot.step_index,
            "activity": [_float_hex(value) for value in snapshot.activity],
            "previous": [_float_hex(value) for value in snapshot.previous],
            "ablation_mask": list(snapshot.ablation_mask),
            "trace_sha256": snapshot.trace_sha256,
        }
    )


def _activity_digest(values: NDArray[np.float64]) -> str:
    """Bit-identical vectorised form of ``_float_tuple_digest``.

    The reference packs each element with ``struct.pack(">d", value)`` in a
    Python loop; ``astype(">f8").tobytes()`` produces the same bytes 20x
    faster. Non-finite values are refused before this is called, matching the
    reference's ``NeuralIntegrityError``.
    """

    return sha256(values.astype(">f8").tobytes()).hexdigest()


def _state_material(
    *,
    step_index: int,
    graph_identity_sha256: str,
    dynamics_identity_sha256: str,
    sign_overlay_sha256: str,
    delay_overlay_sha256: str,
    weight_overlay_sha256: str,
    topology_overlay_sha256: str,
    activity_sha256: str,
) -> dict[str, Any]:
    """Same material and same conditional-key rule as ``model._state_digest``."""

    material: dict[str, Any] = {
        "activity_sha256": activity_sha256,
        "delay_overlay_sha256": delay_overlay_sha256,
        "dynamics_identity_sha256": dynamics_identity_sha256,
        "graph_identity_sha256": graph_identity_sha256,
        "sign_overlay_sha256": sign_overlay_sha256,
        "step_index": step_index,
    }
    if weight_overlay_sha256 != ZERO_SHA256:
        material["weight_overlay_sha256"] = weight_overlay_sha256
    if topology_overlay_sha256 != ZERO_SHA256:
        material["topology_overlay_sha256"] = topology_overlay_sha256
    return material
