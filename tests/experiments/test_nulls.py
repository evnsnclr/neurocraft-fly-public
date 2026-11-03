"""Sign-permutation null: a true permutation, deterministic, engine-compatible.

The permuted overlay must carry the base overlay's exact sign multiset with the
value/provenance pairs kept together, so nothing about the sign *distribution*
changes — only which edge carries which sign. That is the whole point of the
control: it isolates the contribution of sign *placement* from sign statistics.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from neurocraft_connectome import RuntimeGraph, load_malecns_fixture
from neurocraft_connectome.flight_pin import MALECNS_FLIGHT_FIXTURE_V1_PIN
from neurocraft_experiments.nulls import (
    PERMUTED_SIGN_PROVENANCE_PREFIX,
    permuted_sign_overlay,
    permuted_sign_pairs,
)
from neurocraft_neural import NeuralEngine
from neurocraft_neural.model import SignOverlay

REPO_ROOT = Path(__file__).resolve().parents[2]
STEERING_FIXTURE = REPO_ROOT / "data" / "fixtures" / "mcns-v1.0-flight-steering-v1.json"
STEERING_FIXTURE_BYTES = 284_949


@pytest.fixture(scope="module")
def graph() -> RuntimeGraph:
    return load_malecns_fixture(
        STEERING_FIXTURE,
        expected_pin=MALECNS_FLIGHT_FIXTURE_V1_PIN,
        expected_file_bytes=STEERING_FIXTURE_BYTES,
    )


@pytest.fixture(scope="module")
def base(graph: RuntimeGraph) -> SignOverlay:
    return SignOverlay.coarse_presynaptic_consensus(graph)


def test_permutation_preserves_value_provenance_pairs(
    graph: RuntimeGraph, base: SignOverlay
) -> None:
    values, provenance, permutation = permuted_sign_pairs(base, seed=7)
    assert sorted(permutation) == list(range(len(base.values)))
    original_pairs = Counter(zip(base.values, base.provenance, strict=True))
    permuted_sources = Counter(
        (base.values[source], base.provenance[source]) for source in permutation
    )
    assert permuted_sources == original_pairs
    assert values == tuple(base.values[source] for source in permutation)
    assert Counter(values) == Counter(base.values)
    assert len(provenance) == len(values)


def test_same_seed_reproduces_and_different_seed_differs(base: SignOverlay) -> None:
    first = permuted_sign_pairs(base, seed=7)
    second = permuted_sign_pairs(base, seed=7)
    other = permuted_sign_pairs(base, seed=8)
    assert first == second
    assert first[2] != other[2]


def test_permutation_actually_moves_signs(base: SignOverlay) -> None:
    """The steering fixture's sign vector is mixed (+698/-171/0x18), so a real
    permutation must change at least one slot's value; an identity mapping here
    would mean the generator is broken, not unlucky."""

    values, _, permutation = permuted_sign_pairs(base, seed=7)
    assert permutation != tuple(range(len(base.values)))
    assert values != base.values


def test_overlay_is_explicit_engine_compatible_and_marked(
    graph: RuntimeGraph, base: SignOverlay
) -> None:
    overlay = permuted_sign_overlay(graph, base, seed=7, protocol_id="test-prereg")
    overlay.assert_compatible(graph)
    assert overlay.identity_sha256 != base.identity_sha256
    for entry in overlay.provenance:
        assert entry.startswith(f"{PERMUTED_SIGN_PROVENANCE_PREFIX}:test-prereg:seed=7:")
    engine = NeuralEngine(graph, sign_overlay=overlay)
    assert engine.step_index == 0


def test_incompatible_base_is_refused(graph: RuntimeGraph, base: SignOverlay) -> None:
    with pytest.raises(Exception, match="sign overlay"):
        permuted_sign_overlay(
            graph,
            SignOverlay(
                graph_support_sha256="0" * 64,
                graph_raw_counts_sha256="0" * 64,
                values=base.values,
                provenance=base.provenance,
                identity_sha256="0" * 64,
            ),
            seed=1,
            protocol_id="test-prereg",
        )


def test_seed_validation_is_fail_closed(base: SignOverlay) -> None:
    """The seed guard mirrors the pinned _control_seed contract exactly."""

    for bad in (True, False, -1, 2**64, 7.0, "7", None):
        with pytest.raises(ValueError, match="seed"):
            permuted_sign_pairs(base, seed=bad)  # type: ignore[arg-type]
    _values, _provenance, permutation = permuted_sign_pairs(base, seed=2**64 - 1)
    assert sorted(permutation) == list(range(len(base.values)))
