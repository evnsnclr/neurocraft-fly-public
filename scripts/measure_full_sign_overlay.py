#!/usr/bin/env python3
"""Measure coarse-consensus sign coverage over the full MaleCNS artifact.

Stage 1 of the full-brain-live lane: the 25,582,938-edge
``male-cns-v1.0-superclass-v1`` artifact carries no signs, so before any
signed full-graph runtime view exists this script measures — never assumes —
what the transmitter-derived ``coarse_presynaptic_consensus`` policy actually
signs, and writes the census as content-addressed evidence for
``docs/receipts/SIGN-OVERLAY-02.md``.

The policy executed here is the array-native duplicate in
``neurocraft_neural.full_signs``, pinned edge-for-edge against the pinned
fixture-scope implementation by ``tests/neural/test_full_signs.py``. The
annotations (neurotransmitters) table is verified against the SHA-256 the
artifact manifest declares for it before a single row is read, closing the
verification gap a prior census disclosed.

The artifact manifest SHA-256 is a mandatory external pin, exactly as in
``scripts/benchmark_full_engine.py``: a colocated manifest never authorizes
itself. The canonical evidence JSON contains no timestamps or environment
data; wall time is printed and belongs to the environment-scoped prose of the
receipt, never to the hashed payload, so a rerun must reproduce the evidence
byte-identically.

Annotation census and derived arrays only — no participation, behavior, or
biological claim.
"""

from __future__ import annotations

import argparse
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _package in ("connectome-data", "neural-runtime"):
    sys.path.insert(0, str(REPOSITORY_ROOT / "packages" / _package / "src"))

from neurocraft_connectome.full_runtime import (  # noqa: E402
    FullRuntimeGraph,
    load_full_runtime_artifact,
)
from neurocraft_connectome.model import canonical_json_bytes  # noqa: E402
from neurocraft_neural.full_signs import (  # noqa: E402
    CONSENSUS_COLUMN,
    NEGATIVE_TRANSMITTERS,
    POSITIVE_TRANSMITTERS,
    TRANSMITTER_SOURCE_ID,
    compute_full_sign_overlay,
)

FULL_ARTIFACT_DIRECTORY = REPOSITORY_ROOT / "data" / "processed" / "male-cns-v1.0-superclass-v1"
DEFAULT_ANNOTATIONS = (
    REPOSITORY_ROOT / "data" / "cache" / "male-cns-v1.0"
) / "body-neurotransmitters-male-cns-v1.0.feather"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "artifacts" / "analysis" / "full-sign-overlay.json"


class FullSignOverlayMeasurementError(RuntimeError):
    """Raised when the census cannot be produced or verified exactly."""


def measure(graph: FullRuntimeGraph, annotations_path: Path) -> dict[str, Any]:
    """Census the executed full-graph coarse-consensus overlay."""

    result = compute_full_sign_overlay(graph, annotations_path)
    declared_source = next(
        source
        for source in graph.manifest.source_artifacts
        if source.source_id == TRANSMITTER_SOURCE_ID
    )
    edge_count = graph.manifest.edge_count
    return {
        "receipt_id": "SIGN-OVERLAY-02",
        "claim_scope": (
            "annotation_census_and_derived_sign_arrays_only"
            "_no_participation_behavior_or_biological_claim"
        ),
        "artifact": {
            "dataset_id": graph.manifest.dataset_id,
            "artifact_id": graph.manifest.artifact_id,
            "manifest_file_sha256": graph.manifest.manifest_file_sha256,
            "structural_graph_sha256": graph.manifest.structural_graph_sha256,
            "provenance_graph_sha256": graph.manifest.provenance_graph_sha256,
            "node_count": graph.manifest.node_count,
            "edge_count": edge_count,
        },
        "annotations_source": {
            "source_id": declared_source.source_id,
            "filename": declared_source.filename,
            "declared_bytes": declared_source.byte_length,
            "declared_sha256": declared_source.sha256,
            "observed_sha256": result.annotations_sha256,
            "verified": result.annotations_sha256 == declared_source.sha256,
            "declared_in": (
                "full artifact manifest sources[], which load_full_runtime_artifact "
                "verifies against the trusted acquisition pins before mapping any array"
            ),
            "source_row_count": graph.manifest.counts["transmitter_source_rows"],
            "consensus_column": CONSENSUS_COLUMN,
        },
        "policy": {
            "name": result.policy_id,
            "implementation": result.policy_implementation,
            "duplication": (
                "positive/negative transmitter sets duplicated from the pinned "
                "fixture-scope implementation because that file is identity-bound; "
                "pinned by the policy-equality regression tests in "
                "tests/neural/test_full_signs.py"
            ),
            "positive_transmitters": sorted(POSITIVE_TRANSMITTERS),
            "negative_transmitters": sorted(NEGATIVE_TRANSMITTERS),
            "identity_sha256": result.identity_sha256,
        },
        "edges": {
            "sign_counts": {
                label: result.sign_counts[label] for label in ("positive", "negative", "zero")
            },
            "signed_edge_count": result.signed_edge_count,
            "signed_edge_fraction": result.signed_edge_count / edge_count,
            "per_presynaptic_transmitter": {
                transmitter: dict(sorted(counts.items()))
                for transmitter, counts in sorted(result.per_transmitter_edge_signs.items())
            },
            "zeroing_transmitters": list(result.zeroing_transmitters),
            "sign_array_dtype": "int8",
            "sign_array_sha256": result.sign_array_sha256,
        },
        "nodes": {
            "consensus_nt_counts": dict(sorted(result.node_transmitter_counts.items())),
            "nodes_missing_transmitter_row": result.nodes_missing_transmitter_row,
            "presynaptic_nodes_with_zeroed_outgoing_edge_count": (
                result.zeroing_presynaptic_node_count
            ),
        },
    }


def write_or_verify(output: Path, census: dict[str, Any]) -> tuple[str, str]:
    """Write the census once; afterwards only byte-identical reproduction passes."""

    payload = canonical_json_bytes(census)
    digest = sha256(payload).hexdigest()
    if output.exists():
        existing = output.read_bytes()
        if existing != payload:
            raise FullSignOverlayMeasurementError(
                f"existing census at {output} differs from this measurement; "
                "evidence is immutable — investigate instead of overwriting"
            )
        return digest, "verified"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return digest, "wrote"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Census the coarse presynaptic consensus sign overlay over the full "
            "MaleCNS runtime artifact. Annotation census only; no biological claim."
        )
    )
    parser.add_argument("--artifact-directory", type=Path, default=FULL_ARTIFACT_DIRECTORY)
    parser.add_argument("--artifact-manifest-sha256", default=None)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.artifact_manifest_sha256 is None:
        raise SystemExit("--artifact-manifest-sha256 is mandatory: the pin must be external")

    started = time.perf_counter()
    graph = load_full_runtime_artifact(
        args.artifact_directory, expected_manifest_sha256=args.artifact_manifest_sha256
    )
    try:
        loaded = time.perf_counter()
        census = measure(graph, args.annotations)
        measured = time.perf_counter()
    finally:
        graph.close()
    digest, action = write_or_verify(args.output, census)
    edges = census["edges"]
    print(
        f"{action} SIGN-OVERLAY-02 census: "
        f"signed={edges['signed_edge_count']}/{census['artifact']['edge_count']} "
        f"({100.0 * edges['signed_edge_fraction']:.4f}%) "
        f"(+{edges['sign_counts']['positive']}/-{edges['sign_counts']['negative']}"
        f"/0:{edges['sign_counts']['zero']}) "
        f"sign_array_sha256={edges['sign_array_sha256']} file_sha256={digest}"
    )
    print(
        "wall time (environment-scoped, not part of the canonical evidence): "
        f"artifact load+verify {loaded - started:.3f}s, "
        f"overlay compute+census {measured - loaded:.3f}s, "
        f"total {measured - started:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
