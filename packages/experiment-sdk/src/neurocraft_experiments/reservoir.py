"""Trained-Readout Mode reservoir harness foundation (ADR 0008).

Machinery only, deliberately results-free: per ADR 0008 the Trained-Readout
Mode preregistration must be recorded before any result is generated under it,
so this module provides the deterministic building blocks — condition-built
engines, a declared drive loop, a closed-form ridge readout, and a
content-addressed run identity — and defines no task and reports no finding.

Like :mod:`neurocraft_experiments.nulls`, this is a new file on purpose: pilot
v3 pins the exact bytes of eleven implementation files, and every constructor
used here (``NeuralEngine``, the overlay classmethods, ``Stimulation``) is
consumed as-is from those pinned modules, never edited.

Declared modeling choice for the rewire condition: sign values travel with
**edge slots**, exactly as the engine composes a sign overlay with a topology
overlay. Under ``degree_preserving_rewire`` an edge therefore keeps its
original source edge's sign even though its presynaptic endpoint moved; whether
a preregistration instead wants signs re-derived from the rewired presynaptic
neuron is a protocol decision that must be recorded there, not silently made
here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from neurocraft_connectome import RuntimeGraph
from neurocraft_connectome.model import sha256_json
from neurocraft_neural import NeuralEngine
from neurocraft_neural.model import SignOverlay, Stimulation, TopologyOverlay, WeightOverlay

from .nulls import permuted_sign_overlay

SUPPORTED_CONDITIONS = (
    "connectome",
    "sign_permuted",
    "sign_ablated_zero",
    "weight_shuffled",
    "degree_preserving_rewire",
)

MAX_DRIVE_STEPS = 100_000


class ReservoirConditionError(ValueError):
    """Raised when a reservoir condition, drive, or readout request is invalid."""


def build_condition_engine(
    graph: RuntimeGraph,
    condition: str,
    *,
    seed: int,
    protocol_id: str,
    requested_swaps: int = 1,
) -> NeuralEngine:
    """One engine per named condition; unknown names are refused, never guessed."""

    if not isinstance(protocol_id, str) or not protocol_id:
        raise ReservoirConditionError("reservoir conditions require a non-empty protocol_id")
    consensus = SignOverlay.coarse_presynaptic_consensus(graph)
    if condition == "connectome":
        return NeuralEngine(graph, sign_overlay=consensus)
    if condition == "sign_permuted":
        return NeuralEngine(
            graph,
            sign_overlay=permuted_sign_overlay(
                graph, consensus, seed=seed, protocol_id=protocol_id
            ),
        )
    if condition == "sign_ablated_zero":
        return NeuralEngine(graph, sign_overlay=SignOverlay.unknown_zero(graph))
    if condition == "weight_shuffled":
        return NeuralEngine(
            graph,
            sign_overlay=consensus,
            weight_overlay=WeightOverlay.shuffled(
                graph, seed=seed, provenance=f"{protocol_id}-weight-shuffled"
            ),
        )
    if condition == "degree_preserving_rewire":
        return NeuralEngine(
            graph,
            sign_overlay=consensus,
            topology_overlay=TopologyOverlay.degree_preserving_random(
                graph,
                seed=seed,
                requested_swaps=requested_swaps,
                provenance=f"{protocol_id}-degree-preserving",
            ),
        )
    raise ReservoirConditionError(f"unknown reservoir condition {condition!r}")


def _validated_drive(
    drive: Sequence[Mapping[int, float] | None],
) -> tuple[Mapping[int, float] | None, ...]:
    entries = tuple(drive)
    if not entries:
        raise ReservoirConditionError("a reservoir drive requires at least one step")
    if len(entries) > MAX_DRIVE_STEPS:
        raise ReservoirConditionError(f"a reservoir drive is bounded at {MAX_DRIVE_STEPS} steps")
    return entries


def drive_and_collect(
    engine: NeuralEngine,
    drive: Sequence[Mapping[int, float] | None],
    *,
    protocol_id: str,
) -> tuple[tuple[float, ...], ...]:
    """Step the engine once per drive entry and collect the full hidden state.

    Each non-empty entry becomes a named :class:`Stimulation` over exact body
    IDs; unknown bodies fail closed inside ``Stimulation.from_body_values``.
    The returned matrix is ``drive_steps x node_count``, the state *after* each
    step.
    """

    if not isinstance(protocol_id, str) or not protocol_id:
        raise ReservoirConditionError("drive_and_collect requires a non-empty protocol_id")
    states: list[tuple[float, ...]] = []
    for index, entry in enumerate(_validated_drive(drive)):
        if entry:
            stimulation = Stimulation.from_body_values(
                engine.graph, f"{protocol_id}-drive-{index}", dict(entry)
            )
            engine.step(stimulation=stimulation)
        else:
            engine.step()
        states.append(engine.current_state().activity)
    return tuple(states)


def ridge_readout(
    states: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    *,
    l2: float,
) -> tuple[tuple[float, ...], ...]:
    """Closed-form ridge regression: ``(X^T X + l2 I) W = X^T Y``.

    Deterministic and dependency-light; returns ``features x outputs`` weights
    as plain floats so downstream identities can hash them canonically.
    """

    if not isinstance(l2, float) or not np.isfinite(l2) or l2 <= 0.0:
        raise ReservoirConditionError(
            "ridge l2 must be a finite strictly positive float: at l2=0 a "
            "rank-deficient gram matrix either raises a raw LinAlgError or "
            "returns BLAS-dependent, numerically meaningless weights"
        )
    x = np.asarray(states, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] == 0:
        raise ReservoirConditionError(
            "ridge readout requires matching non-empty 2-D shape: "
            f"states {x.shape}, targets {y.shape}"
        )
    gram = x.T @ x + l2 * np.eye(x.shape[1], dtype=np.float64)
    try:
        weights = np.linalg.solve(gram, x.T @ y)
    except np.linalg.LinAlgError as exc:
        raise ReservoirConditionError(f"ridge system is singular: {exc}") from exc
    return tuple(tuple(float(value) for value in row) for row in weights)


def _drive_material(
    drive: Sequence[Mapping[int, float] | None],
) -> list[list[object]]:
    material: list[list[object]] = []
    for index, entry in enumerate(_validated_drive(drive)):
        assignments = (
            [[body_id, float(value).hex()] for body_id, value in sorted(entry.items())]
            if entry
            else []
        )
        material.append([index, assignments])
    return material


def reservoir_run_identity(
    engine: NeuralEngine,
    *,
    condition: str,
    seed: int,
    protocol_id: str,
    drive: Sequence[Mapping[int, float] | None],
) -> str:
    """Content-address one prospective run: graph, overlays, condition, drive."""

    if condition not in SUPPORTED_CONDITIONS:
        raise ReservoirConditionError(f"unknown reservoir condition {condition!r}")
    return sha256_json(
        {
            "condition": condition,
            "delay_overlay_sha256": engine.delay_overlay.identity_sha256,
            "drive": _drive_material(drive),
            "dynamics_sha256": engine.config.identity_sha256,
            "graph_identity_sha256": engine.graph.manifest.runtime_identity_sha256,
            "protocol_id": protocol_id,
            "schema": "neurocraft.trm.reservoir_run.v1",
            "seed": seed,
            "sign_overlay_sha256": engine.sign_overlay.identity_sha256,
            "topology_overlay_sha256": engine.topology_overlay.identity_sha256,
            "weight_overlay_sha256": engine.weight_overlay.identity_sha256,
        }
    )
