"""Shared fixtures for the array-native backend tests.

Nothing here touches the full MaleCNS artifact or the compiled shared library;
both are optional and their tests skip when absent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "data/fixtures/mcns-v1.0-flight-power-v1.json"
FULL_ARTIFACT_DIRECTORY = REPO_ROOT / "data/processed/male-cns-v1.0-superclass-v1"

#: Empirical MaleCNS in-degree landmarks (min 0, median 98, mean 153.5,
#: p99 858, max 11,526) used to shape the "measured" synthetic profile.
MEASURED_DEGREE_LANDMARKS: dict[str, int] = {
    "median": 98,
    "mean": 154,
    "p99": 858,
    "max": 11526,
}


@dataclass(frozen=True, slots=True)
class SyntheticCsr:
    """A synthetic CSR graph in exactly the layout the view consumes."""

    indptr: NDArray[np.int64]
    indices: NDArray[np.uint32]
    counts: NDArray[np.int64]
    signs: NDArray[np.int64]

    @property
    def node_count(self) -> int:
        return int(self.indptr.shape[0]) - 1

    @property
    def edge_count(self) -> int:
        return int(self.indices.shape[0])


def _degree_profile(
    node_count: int,
    edge_count: int,
    profile: str,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    if profile == "uniform":
        raw = np.full(node_count, max(1.0, edge_count / max(1, node_count)))
    elif profile == "powerlaw":
        raw = rng.pareto(1.5, size=node_count) + 1.0
    elif profile == "measured":
        # Log-normal body plus a heavy upper tail, matched to the landmarks.
        raw = rng.lognormal(mean=math.log(98.0), sigma=1.0, size=node_count)
        tail = rng.random(node_count) < 0.005
        raw[tail] *= rng.uniform(3.0, 12.0, size=int(tail.sum()))
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown degree profile {profile!r}")
    raw = np.clip(raw, 0.0, float(MEASURED_DEGREE_LANDMARKS["max"]))
    total = raw.sum()
    if total <= 0:
        return np.zeros(node_count, dtype=np.int64)
    scaled = np.maximum(np.rint(raw * (edge_count / total)), 0.0)
    return scaled.astype(np.int64)


def synthetic_csr(
    *,
    node_count: int,
    edge_count: int,
    seed: int,
    profile: str = "measured",
    empty_rows: tuple[int, ...] = (),
    max_row_override: tuple[int, int] | None = None,
) -> SyntheticCsr:
    """Build a CSR graph with sorted column indices, like the real artifact.

    Column indices are sorted within each row but not deduplicated: repeated
    ``(post, pre)`` pairs are numerically harmless here and avoid an O(N) draw
    per row. ``empty_rows`` forces the named rows to in-degree zero so leading,
    trailing and consecutive empty rows can be exercised deliberately.
    """

    rng = np.random.default_rng(seed)
    degrees = _degree_profile(node_count, edge_count, profile, rng)
    if max_row_override is not None:
        row, degree = max_row_override
        degrees[row] = degree
    for row in empty_rows:
        degrees[row] = 0
    total = int(degrees.sum())
    indptr = np.zeros(node_count + 1, dtype=np.int64)
    np.cumsum(degrees, out=indptr[1:])
    columns = rng.integers(0, node_count, size=total, dtype=np.int64)
    rows = np.repeat(np.arange(node_count, dtype=np.int64), degrees)
    order = np.lexsort((columns, rows))
    columns = columns[order]
    counts = rng.integers(1, 2592, size=total, dtype=np.int64)
    signs = rng.choice(np.array([-1, 0, 1], dtype=np.int64), size=total)
    return SyntheticCsr(
        indptr=indptr,
        indices=columns.astype(np.uint32),
        counts=counts,
        signs=signs,
    )


def effective_weights(csr: SyntheticCsr) -> list[float]:
    """The reference engine's weight derivation, written out in pure Python."""

    denominators = [0] * csr.node_count
    starts = csr.indptr.tolist()
    counts = csr.counts.tolist()
    for row in range(csr.node_count):
        denominators[row] = sum(counts[starts[row] : starts[row + 1]])
    signs = csr.signs.tolist()
    weights: list[float] = []
    for row in range(csr.node_count):
        divisor = max(1, denominators[row])
        for edge in range(starts[row], starts[row + 1]):
            weights.append(signs[edge] * counts[edge] / divisor)
    return weights


class FsumReference:
    """Pure-Python ``math.fsum`` oracle for the same update law.

    This is the arithmetic the frozen reference engine performs, expressed
    directly over CSR arrays so it can be driven at sizes where building a
    ``RuntimeGraph`` would be prohibitive.
    """

    def __init__(
        self,
        csr: SyntheticCsr,
        *,
        neural_step_ms: float = 10.0,
        tau_ms: float = 50.0,
        recurrent_gain: float = 0.9,
        sensory_gain: float = 1.0,
        baseline: float = 0.0,
    ) -> None:
        self.indptr = csr.indptr.tolist()
        self.indices = csr.indices.tolist()
        self.weights = effective_weights(csr)
        self.node_count = csr.node_count
        self.leak = math.exp(-neural_step_ms / tau_ms)
        self.recurrent_gain = recurrent_gain
        self.sensory_gain = sensory_gain
        self.baseline = baseline
        self.activity = [0.0] * self.node_count
        self.previous = [0.0] * self.node_count

    def recurrent(self) -> list[float]:
        result = [0.0] * self.node_count
        for row in range(self.node_count):
            terms = [
                self.weights[edge] * self.previous[self.indices[edge]]
                for edge in range(self.indptr[row], self.indptr[row + 1])
            ]
            result[row] = math.fsum(terms)
        return result

    def step(self, direct: list[float] | None = None) -> list[float]:
        recurrent = self.recurrent()
        drive_scale = 1.0 - self.leak
        nxt: list[float] = []
        for index in range(self.node_count):
            z = (
                self.recurrent_gain * recurrent[index]
                + self.sensory_gain * 0.0
                + (0.0 if direct is None else direct[index])
                + self.baseline
            )
            nxt.append(self.leak * self.activity[index] + drive_scale * math.tanh(z))
        self.previous = self.activity
        self.activity = nxt
        return nxt
