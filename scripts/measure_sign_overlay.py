#!/usr/bin/env python3
"""Measure sign-overlay coverage for the committed flight-steering fixture.

The 128-node ``mcns-v1.0-flight-steering-v1`` fixture stores every edge with
``canonical_sign_status`` unknown; signs exist only as the runtime
``SignOverlay.coarse_presynaptic_consensus`` policy. Before any experiment
depends on signed dynamics, this script measures — never assumes — how much of
the fixture that policy actually signs, and writes the census as
content-addressed evidence for ``docs/receipts/SIGN-OVERLAY-01.md``.

Every classification below is derived from the executed overlay itself: the
sign comes from ``overlay.values`` and the normalized transmitter from the
``inferred_from_presynaptic_consensus_nt:<nt>`` provenance the policy emitted,
so this census cannot drift from the policy implementation it measures.

Topology and annotation census only — no participation, behavior, or
biological claim.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

from neurocraft_connectome import load_malecns_fixture
from neurocraft_connectome.flight_pin import MALECNS_FLIGHT_FIXTURE_V1_PIN
from neurocraft_connectome.model import RuntimeGraph, canonical_json_bytes
from neurocraft_neural.model import SignOverlay

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FLIGHT_FIXTURE = REPOSITORY_ROOT / "data" / "fixtures" / "mcns-v1.0-flight-steering-v1.json"
FLIGHT_FIXTURE_BYTES = 284_949
DEFAULT_OUTPUT = REPOSITORY_ROOT / "docs" / "receipts" / "SIGN-OVERLAY-01.json"
PROVENANCE_PREFIX = "inferred_from_presynaptic_consensus_nt:"
SIGN_LABELS = {1: "positive", -1: "negative", 0: "zero"}


class SignOverlayMeasurementError(RuntimeError):
    """Raised when the census cannot be produced or verified exactly."""


def _provenance_transmitter(entry: str) -> str:
    if not entry.startswith(PROVENANCE_PREFIX):
        raise SignOverlayMeasurementError(
            f"unexpected sign provenance {entry!r}: the census only understands "
            "the coarse presynaptic consensus policy"
        )
    return entry.removeprefix(PROVENANCE_PREFIX)


def measure(graph: RuntimeGraph) -> dict[str, Any]:
    """Census the executed coarse-consensus overlay for one fixture graph."""

    overlay = SignOverlay.coarse_presynaptic_consensus(graph)
    sign_counts = Counter(SIGN_LABELS[value] for value in overlay.values)
    per_transmitter: dict[str, Counter[str]] = {}
    for value, entry in zip(overlay.values, overlay.provenance, strict=True):
        transmitter = _provenance_transmitter(entry)
        per_transmitter.setdefault(transmitter, Counter())[SIGN_LABELS[value]] += 1

    zeroing_transmitters = sorted(
        transmitter for transmitter, counts in per_transmitter.items() if counts["zero"] > 0
    )
    # Same fallback the executed policy uses, so an empty consensus_nt lands in the
    # same bucket ("unknown") in both the node census and the zeroed-node scan.
    node_nt_counts = Counter(
        (node.consensus_nt.casefold() if node.consensus_nt else "unknown") for node in graph.nodes
    )
    nodes_with_outgoing = {edge.pre_index for edge in graph.edges}
    unmapped_presynaptic_nodes = sorted(
        node.body_id
        for node in graph.nodes
        if node.dense_index in nodes_with_outgoing
        and (node.consensus_nt.casefold() if node.consensus_nt else "unknown")
        in zeroing_transmitters
    )
    edge_count = len(graph.edges)
    signed = sign_counts["positive"] + sign_counts["negative"]
    canonical_edge_statuses = Counter(edge.canonical_sign_status for edge in graph.edges)
    return {
        "receipt_id": "SIGN-OVERLAY-01",
        "claim_scope": (
            "topology_and_annotation_census_only_no_participation_behavior_or_biological_claim"
        ),
        "fixture": {
            "dataset_id": graph.manifest.dataset_id,
            "fixture_id": graph.manifest.fixture_id,
            "fixture_file_sha256": graph.manifest.fixture_file_sha256,
            "runtime_identity_sha256": graph.manifest.runtime_identity_sha256,
            "node_count": graph.manifest.node_count,
            "edge_count": graph.manifest.edge_count,
            "canonical_edge_sign_status_counts": dict(sorted(canonical_edge_statuses.items())),
        },
        "policy": {
            "name": "coarse_presynaptic_consensus",
            "implementation": "neurocraft_neural.model.SignOverlay.coarse_presynaptic_consensus",
            "sign_overlay_identity_sha256": overlay.identity_sha256,
        },
        "edges": {
            "sign_counts": {
                label: sign_counts[label] for label in ("positive", "negative", "zero")
            },
            "signed_edge_count": signed,
            "signed_edge_fraction": signed / edge_count,
            "per_presynaptic_transmitter": {
                transmitter: {
                    label: counts[label]
                    for label in ("positive", "negative", "zero")
                    if counts[label]
                }
                for transmitter, counts in sorted(per_transmitter.items())
            },
            "zeroing_transmitters": zeroing_transmitters,
        },
        "nodes": {
            "consensus_nt_counts": dict(sorted(node_nt_counts.items())),
            "presynaptic_nodes_with_zeroed_outgoing_edges": unmapped_presynaptic_nodes,
            "presynaptic_nodes_with_zeroed_outgoing_edge_count": len(unmapped_presynaptic_nodes),
        },
    }


def write_or_verify(output: Path, census: dict[str, Any]) -> tuple[str, str]:
    """Write the census once; afterwards only byte-identical reproduction passes."""

    payload = canonical_json_bytes(census)
    digest = sha256(payload).hexdigest()
    if output.exists():
        existing = output.read_bytes()
        if existing != payload:
            raise SignOverlayMeasurementError(
                f"existing census at {output} differs from this measurement; "
                "receipts are immutable — investigate instead of overwriting"
            )
        return digest, "verified"
    output.write_bytes(payload)
    return digest, "wrote"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Census the coarse presynaptic consensus sign overlay on the committed "
            "flight-steering fixture. Annotation census only; no biological claim."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    graph = load_malecns_fixture(
        FLIGHT_FIXTURE,
        expected_pin=MALECNS_FLIGHT_FIXTURE_V1_PIN,
        expected_file_bytes=FLIGHT_FIXTURE_BYTES,
    )
    census = measure(graph)
    digest, action = write_or_verify(args.output, census)
    edges = census["edges"]
    print(
        f"{action} SIGN-OVERLAY-01 census: "
        f"signed={edges['signed_edge_count']}/{census['fixture']['edge_count']} "
        f"(+{edges['sign_counts']['positive']}/-{edges['sign_counts']['negative']}"
        f"/0:{edges['sign_counts']['zero']}) file_sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
