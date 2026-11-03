"""SIGN-OVERLAY-01: the committed census must equal a fresh measurement.

The receipt is only evidence while it reproduces byte-identically from the
committed fixture and the executed ``coarse_presynaptic_consensus`` policy. A
loader, fixture, or policy change must trip this suite instead of silently
inheriting the old census.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest
from neurocraft_connectome import load_malecns_fixture
from neurocraft_connectome.flight_pin import MALECNS_FLIGHT_FIXTURE_V1_PIN
from neurocraft_connectome.model import RuntimeGraph, canonical_json_bytes

from scripts.measure_sign_overlay import (
    FLIGHT_FIXTURE,
    FLIGHT_FIXTURE_BYTES,
    SignOverlayMeasurementError,
    _provenance_transmitter,
    measure,
    write_or_verify,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_CENSUS = REPO_ROOT / "docs" / "receipts" / "SIGN-OVERLAY-01.json"


@pytest.fixture(scope="module")
def flight_graph() -> RuntimeGraph:
    return load_malecns_fixture(
        FLIGHT_FIXTURE,
        expected_pin=MALECNS_FLIGHT_FIXTURE_V1_PIN,
        expected_file_bytes=FLIGHT_FIXTURE_BYTES,
    )


def test_census_counts_are_internally_consistent(flight_graph: RuntimeGraph) -> None:
    census = measure(flight_graph)
    edges = census["edges"]
    sign_counts = edges["sign_counts"]
    assert sum(sign_counts.values()) == census["fixture"]["edge_count"]
    assert edges["signed_edge_count"] == sign_counts["positive"] + sign_counts["negative"]
    per_transmitter_total = sum(
        count
        for counts in edges["per_presynaptic_transmitter"].values()
        for count in counts.values()
    )
    assert per_transmitter_total == census["fixture"]["edge_count"]
    assert census["fixture"]["node_count"] == 128
    assert census["fixture"]["edge_count"] == 887


def test_census_is_deterministic(flight_graph: RuntimeGraph) -> None:
    first = canonical_json_bytes(measure(flight_graph))
    second = canonical_json_bytes(measure(flight_graph))
    assert first == second


def test_committed_receipt_reproduces_byte_identically(flight_graph: RuntimeGraph) -> None:
    assert COMMITTED_CENSUS.exists(), "SIGN-OVERLAY-01.json must be committed evidence"
    fresh = canonical_json_bytes(measure(flight_graph))
    committed = COMMITTED_CENSUS.read_bytes()
    assert sha256(committed).hexdigest() == sha256(fresh).hexdigest()


def test_write_then_verify_then_refuse_divergence(
    flight_graph: RuntimeGraph, tmp_path: Path
) -> None:
    census = measure(flight_graph)
    target = tmp_path / "census.json"
    digest, action = write_or_verify(target, census)
    assert action == "wrote"
    verified_digest, action = write_or_verify(target, census)
    assert action == "verified"
    assert verified_digest == digest
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(SignOverlayMeasurementError):
        write_or_verify(target, census)


def test_foreign_provenance_is_refused() -> None:
    with pytest.raises(SignOverlayMeasurementError):
        _provenance_transmitter("explicit_assignment:acetylcholine")
