from __future__ import annotations

import hashlib
import json
from pathlib import Path

from neurocraft_protocol import canonical_json_bytes

from scripts import benchmark_controller


def test_percentile_uses_deterministic_nearest_rank() -> None:
    values = list(range(1, 101))
    assert benchmark_controller._percentile(values, 0.50) == 50
    assert benchmark_controller._percentile(values, 0.95) == 95
    assert benchmark_controller._percentile(values, 0.99) == 99
    assert benchmark_controller._percentile(values, 0.999) == 100


def test_benchmark_receipt_is_explicitly_fixture_bounded() -> None:
    root = Path(__file__).resolve().parents[2]
    receipt = benchmark_controller._receipt(
        root / "data" / "fixtures" / "mcns-v1.0-flight-power-v1.json",
        warmup=1,
        iterations=2,
    )
    assert receipt["schema"] == "neurocraft.benchmark.fixture-controller.v1"
    assert receipt["graph"]["node_count"] == 14
    assert receipt["graph"]["edge_count"] == 22
    assert "not full-graph" in receipt["evidence_boundary"]
    assert receipt["benchmark"]["iterations"] == 2
    assert len(receipt["benchmark"]["samples_ns"]) == 2
    assert receipt["implementation"]["script_sha256"]
    assert receipt["implementation"]["uv_lock_sha256"]


def test_tracked_benchmark_retains_recomputable_raw_samples() -> None:
    root = Path(__file__).resolve().parents[2]
    receipt = json.loads((root / "docs/receipts/BENCH-FIXTURE-01.json").read_text())
    benchmark = receipt["benchmark"]
    samples = benchmark["samples_ns"]
    ordered = sorted(samples)

    assert len(samples) == benchmark["sample_count"] == benchmark["iterations"]
    assert hashlib.sha256(canonical_json_bytes(samples)).hexdigest() == benchmark["sample_sha256"]
    assert benchmark["p50_tick_ns"] == benchmark_controller._percentile(ordered, 0.50)
    assert benchmark["p95_tick_ns"] == benchmark_controller._percentile(ordered, 0.95)
    assert benchmark["p99_tick_ns"] == benchmark_controller._percentile(ordered, 0.99)
    assert benchmark["p99_9_tick_ns"] == benchmark_controller._percentile(ordered, 0.999)
