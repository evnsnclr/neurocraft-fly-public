#!/usr/bin/env python3
"""Phase 2 feasibility assessment: what a retinotopic visual injection would cost.

The bounded question. The committed flight-steering fixture drives the entity
from two mechanosensory entries. A retinotopic visual injection would instead
address the optic lobe by hex column. This analysis measures, over the pinned G1
artifact and the pinned annotations table, four things a decision to build that
injection would rest on: whether the committed fixture contains any visual
substrate to inject into, how large a bounded retinotopic patch is, whether such
a patch connects to the committed fixture at all, and how large a circuit the
fixture's own selection rule would close around one. It also sizes the one
alternative that does fit — the lobula-plate tangential pool — and measures the
composition of its input.

Method, and why it was chosen:

* Identity. Every array comes from ``load_full_runtime_artifact`` and the
  fixture from ``load_malecns_fixture`` under its own pin, so the manifest,
  payload, structural, provenance and per-array digests are all checked before a
  number is computed. NumPy views are taken over the exact mappings the loader
  hashed rather than by reopening a path.

* Integrity first, through the shared implementation. The annotation join and
  its gates are :mod:`scripts.readout_annotations` unchanged, applied twice —
  once to the artifact's 166,700 retained rows and once to the fixture's 128 —
  so the same four gates that guard ``READOUT-CENSUS-01`` guard this report, and
  the fixture's own superclass composition is reconciled against the rows its
  annotation pointers reach.

* Reproduction, not re-implementation. The column census must reproduce
  ``docs/receipts/OPTIC-CENSUS-01.md`` exactly; a deviation is a blocking error.
  Traversal is :mod:`scripts.reachability_graph` unchanged.

* Direct measurement of the load-bearing zero. The patch-to-fixture edge count
  is taken off the edge list in both directions, never by differencing two
  aggregates, because a zero has to be a zero that was looked for.

* Layers. :mod:`scripts.vision_columns` owns column addressing and patch
  geometry, :mod:`scripts.vision_circuits` owns everything needing the edge
  support, and this file binds them to the verified manifests and writes the
  document.

Claim discipline. This is a topology and annotation census and nothing else. The
word "causal" is reserved by ``docs/SCIENTIFIC_CLAIMS.md`` for the G3 gate and is
not used here. Reachability **bounds participation and never demonstrates it**: a
path may carry vanishing, cancelling or sign-inverted influence under the
assigned leaky-rate dynamics, which this analysis does not execute. No behaviour,
receptive field or visual function is assigned to any neuron in the hashed body;
where a cell type has a function reported in the literature, that reading is
labelled literature-inferred and carried in an unhashed sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

# ``python scripts/analyze_vision_feasibility.py`` places ``scripts/`` rather
# than the repository root on sys.path. Add only this script's resolved
# repository root so the sibling analysis modules have the same import identity
# under direct CLI and test use; this is the convention analyze_readout.py
# already follows.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurocraft_connectome import load_full_runtime_artifact, load_malecns_fixture
from neurocraft_connectome.flight_pin import (
    MALECNS_FLIGHT_FIXTURE_V1_EDGE_COUNT,
    MALECNS_FLIGHT_FIXTURE_V1_FILE_BYTES,
    MALECNS_FLIGHT_FIXTURE_V1_NODE_COUNT,
    MALECNS_FLIGHT_FIXTURE_V1_PATH,
    MALECNS_FLIGHT_FIXTURE_V1_PIN,
)
from neurocraft_connectome.full_runtime import FullRuntimeManifest
from neurocraft_connectome.model import RuntimeGraph, canonical_json_bytes

from scripts.reachability_graph import mapped_numpy
from scripts.readout_annotations import (
    IntegrityGate,
    RetainedAnnotations,
    annotation_gates,
    gates_report,
    histogram,
    require_all_gates,
    take_retained_annotations,
)
from scripts.readout_connectivity import weight_sum_gate
from scripts.vision_circuits import (
    FLIGHT_FIXTURE_EDGE_CEILING,
    LPTC_TYPE_PREFIXES,
    PATH_EDGE_MINIMUM_WEIGHT,
    EdgeSupport,
    closure_report,
    dense_indices_for,
    descending_pool,
    descending_support,
    motion_pathway_addressing,
    pool_summary,
    tangential_census,
    tangential_circuits,
    tangential_input_composition,
    type_prefix_members,
    wing_motor_pool,
)
from scripts.vision_columns import (
    COLUMNAR_TYPES,
    PATCH_TYPES,
    build_column_index,
    column_census,
    hex_addresses,
    reproduction_check,
    select_disc,
)
from scripts.vision_report import (
    cross_connectivity_report,
    fixture_visual_substrate,
    metric_ambiguity,
    patch_sizing,
    peak_rss_bytes,
    print_summary,
    write_report,
)

SCHEMA = "neurocraft.analysis.vision.v1"
DEFAULT_ARTIFACT_DIR = Path("data/processed/male-cns-v1.0-superclass-v1")
DEFAULT_ANNOTATIONS = Path(
    "data/cache/male-cns-v1.0/body-annotations-male-cns-v1.0-minconf-0.5.feather"
)
DEFAULT_FIXTURE = Path(MALECNS_FLIGHT_FIXTURE_V1_PATH)
TRACKED_MANIFEST_SHA256 = "b33bdda8b1a9abfc6b23f7346197442546f7a2aec0a03fc7c07fbdfe4f67c98c"
DEFAULT_OUTPUT = Path("artifacts/analysis/vision-feasibility.json")

#: Patch sizes, in complete right-eye hex columns.
DEFAULT_PATCH_COLUMNS: tuple[int, ...] = (4, 8, 16, 32)

#: Patch sizes the full circuit closure is computed for. The closure needs two
#: whole-graph traversals per size, so it is reported at a subset.
DEFAULT_CLOSURE_COLUMNS: tuple[int, ...] = (4, 16, 32)

#: Weight floors the descending-support measurement is repeated at.
DEFAULT_SUPPORT_THRESHOLDS: tuple[int, ...] = (PATH_EDGE_MINIMUM_WEIGHT, 20)

#: Function readings taken from the cell-biology literature rather than measured
#: here. Kept OUT of the hashed analysis body on purpose: a hashed body must
#: contain only what this run measured from the pinned bytes.
LITERATURE_INFERRED_NOTES: dict[str, Any] = {
    "status": "literature-inferred, not measured by this analysis",
    "scope": (
        "these readings are context for the annotation strings the hashed body records."
        " They are not evidence, they were not produced by this run, and no figure in"
        " the hashed body depends on them. This analysis assigns no behaviour, receptive"
        " field or visual function to any neuron"
    ),
    "cell_types": {
        "R1-R6": "reported as the outer photoreceptors of the compound eye",
        "L1-L5": "reported as lamina monopolar cells of the first optic neuropil",
        "Mi1, Tm1, Tm2, Tm9": "reported as medulla columnar relay cells",
        "T4, T5": (
            "reported as direction-selective columnar cells of the medulla and lobula,"
            " commonly described as the elementary motion detector output stage"
        ),
        "VS, HS, H2 families": (
            "reported as lobula-plate tangential cells with large receptive fields,"
            " commonly described as wide-field optic-flow integrators"
        ),
        "LoVP50": "reported as a lobula and ventrolateral-protocerebrum projection type",
    },
    "why_this_matters_to_a_reader": (
        "the hashed body measures that the tangential pool's retained input weight is"
        " dominated by the T4/T5 columnar types. What that composition would mean for an"
        " encoder is a literature reading and is deliberately not part of the"
        " measurement"
    ),
}


class VisionReproductionError(RuntimeError):
    """Raised when the column census fails to reproduce OPTIC-CENSUS-01."""


# ------------------------------------------------------------------- artifact


def artifact_identity(manifest: FullRuntimeManifest) -> dict[str, Any]:
    """Manifest-derived identity, matching the sibling analyses field for field."""

    return {
        "artifact_id": manifest.artifact_id,
        "dataset_id": manifest.dataset_id,
        "manifest_file_sha256": manifest.manifest_file_sha256,
        "payload_sha256": manifest.payload_sha256,
        "structural_graph_sha256": manifest.structural_graph_sha256,
        "provenance_graph_sha256": manifest.provenance_graph_sha256,
        "source_manifest_sha256": manifest.source_manifest_sha256,
        "builder_implementation_sha256": manifest.builder_implementation_sha256,
        "builder_configuration_sha256": manifest.builder_configuration_sha256,
        "boundary_role_overlay_sha256": manifest.boundary_role_overlay.overlay_sha256,
        "node_count": manifest.node_count,
        "edge_count": manifest.edge_count,
        "raw_count_sum": manifest.raw_count_sum,
        "retained_weight_sum": manifest.counts["retained_weight_sum"],
        "annotation_source_rows": manifest.counts["annotation_source_rows"],
    }


def declared_annotations_sha256(manifest: FullRuntimeManifest) -> str:
    """The source digest the artifact manifest declares for the pinned table."""

    for source in manifest.source_artifacts:
        if source.source_id == "annotations":
            return source.sha256
    raise KeyError("the artifact manifest declares no 'annotations' source")


def fixture_annotations_sha256(graph: RuntimeGraph) -> str:
    """The source digest the fixture manifest declares for the pinned table."""

    for source in graph.manifest.source_artifacts:
        if source.source_id == "annotations":
            return source.sha256
    raise KeyError("the fixture manifest declares no 'annotations' source")


def fixture_gates(
    graph: RuntimeGraph, artifact_declared_sha256: str, declared_source_rows: int
) -> list[IntegrityGate]:
    """Gates binding the loaded fixture to the same pinned annotations table."""

    observed = fixture_annotations_sha256(graph)
    return [
        IntegrityGate(
            name="fixture_and_artifact_share_one_annotations_source",
            requirement=(
                "the flight-steering fixture and the full artifact must declare the same"
                " pinned annotations digest, or a count taken from one would not be"
                " comparable with a count taken from the other"
            ),
            passed=observed == artifact_declared_sha256,
            observed=observed,
            expected=artifact_declared_sha256,
        ),
        IntegrityGate(
            name="fixture_shape_matches_its_pin",
            requirement=(
                "the loaded fixture must carry exactly the node and edge counts its pin"
                " declares, and its emitted edge count must stay under the ceiling its"
                " own selection policy declares"
            ),
            passed=bool(
                graph.manifest.node_count == MALECNS_FLIGHT_FIXTURE_V1_NODE_COUNT
                and graph.manifest.edge_count == MALECNS_FLIGHT_FIXTURE_V1_EDGE_COUNT
                and graph.manifest.edge_count < FLIGHT_FIXTURE_EDGE_CEILING
            ),
            observed={
                "node_count": graph.manifest.node_count,
                "edge_count": graph.manifest.edge_count,
            },
            expected={
                "node_count": MALECNS_FLIGHT_FIXTURE_V1_NODE_COUNT,
                "edge_count": MALECNS_FLIGHT_FIXTURE_V1_EDGE_COUNT,
                "edge_count_below": FLIGHT_FIXTURE_EDGE_CEILING,
            },
        ),
        IntegrityGate(
            name="fixture_annotation_rows_are_inside_the_source_table",
            requirement=(
                "every fixture annotation pointer must lie inside the pinned source"
                " table the artifact manifest declares"
            ),
            passed=all(
                0 <= node.annotation_source_row < declared_source_rows for node in graph.nodes
            ),
            observed={
                "maximum_row": max(node.annotation_source_row for node in graph.nodes),
                "nodes": len(graph.nodes),
            },
            expected={"maximum_row_below": declared_source_rows, "nodes": len(graph.nodes)},
        ),
    ]


# ---------------------------------------------------------------- assembly


def analyse(
    annotations: RetainedAnnotations,
    taken: Any,
    support: EdgeSupport,
    fixture_graph: RuntimeGraph,
    fixture_annotations: RetainedAnnotations,
    fixture_taken: Any,
    *,
    identity: dict[str, Any],
    gates: Sequence[IntegrityGate],
    sizes: Sequence[int] = DEFAULT_PATCH_COLUMNS,
    closure_sizes: Sequence[int] = DEFAULT_CLOSURE_COLUMNS,
    support_thresholds: Sequence[int] = DEFAULT_SUPPORT_THRESHOLDS,
    patch_types: Sequence[str] = PATCH_TYPES,
    columnar_types: Sequence[str] = COLUMNAR_TYPES,
) -> dict[str, Any]:
    """Produce the whole deterministic analysis body."""

    require_all_gates(gates)
    types = annotations.column("type")
    sides = annotations.column("somaSide")
    addresses = hex_addresses(taken)
    census = column_census(
        addresses, types, sides, patch_types=patch_types, columnar_types=columnar_types
    )
    reproduction = reproduction_check(census, int(np.count_nonzero(addresses.addressed)))
    index = build_column_index(addresses, types, sides)
    complete = index.complete(patch_types)
    fixture_nodes = dense_indices_for(
        annotations.body_ids, [node.body_id for node in fixture_graph.nodes]
    )
    descending = descending_pool(annotations)
    motor = wing_motor_pool(annotations)

    support_rows: list[dict[str, Any]] = []
    closure_rows: list[dict[str, Any]] = []
    for threshold in support_thresholds:
        graph = support.subgraph(threshold)
        for size in sizes:
            patch = index.cells(select_disc(complete, size).columns, patch_types)
            row = descending_support(
                graph, patch=patch, descending=descending, minimum_weight=threshold
            )
            row["columns"] = size
            support_rows.append(row)
        if threshold == PATH_EDGE_MINIMUM_WEIGHT:
            for size in closure_sizes:
                patch = index.cells(select_disc(complete, size).columns, patch_types)
                row = closure_report(
                    support, graph, patch=patch, descending=descending, motor=motor
                )
                row["columns"] = size
                closure_rows.append(row)

    tangential = type_prefix_members(types, LPTC_TYPE_PREFIXES)
    return {
        "schema": SCHEMA,
        "artifact": identity,
        "integrity_gates": gates_report(gates),
        "question": (
            "whether a retinotopic visual injection — addressing the optic lobe by hex"
            " column — is buildable inside this repository's fixture discipline. Every"
            " figure below is a count over the pinned artifact, the pinned annotations"
            " table, or the pinned flight-steering fixture"
        ),
        "flight_fixture_visual_substrate": fixture_visual_substrate(
            fixture_graph, fixture_annotations, fixture_taken
        ),
        "column_address_census": census,
        "optic_census_01_reproduction": reproduction,
        "retinotopic_patch_sizing": patch_sizing(
            support, index, annotations, complete, sizes=sizes, patch_types=patch_types
        ),
        "hex_metric_ambiguity": metric_ambiguity(
            support, index, complete, sizes=sizes, patch_types=patch_types
        ),
        "patch_to_flight_fixture_cross_connectivity": cross_connectivity_report(
            support,
            index,
            annotations,
            addresses,
            complete,
            fixture_nodes,
            sizes=sizes,
            patch_types=patch_types,
            columnar_types=columnar_types,
        ),
        "patch_to_descending_support": {
            "definition": (
                "how much of a patch lies within two hops of the descending layer over"
                " the weight-thresholded subgraph the fixture's path rule uses"
            ),
            "descending_pool": pool_summary(annotations, descending, "descending_neuron"),
            "rows": support_rows,
        },
        "patch_circuit_closure": {
            "definition": (
                "the size of the circuit the flight-steering fixture's own one-to-three"
                " edge path rule would close around a retinotopic patch, at three nested"
                " constructions of the intermediate pool"
            ),
            "wing_motor_readout_pool": pool_summary(annotations, motor, "vnc_motor_wm"),
            "rows": closure_rows,
        },
        "lobula_plate_tangential_alternative": {
            "definition": (
                "the one wide-field visual population that closes a circuit inside the"
                " fixture's declared edge ceiling, its census, its circuit sizes, and the"
                " composition of its retained input weight"
            ),
            "census": tangential_census(annotations, tangential, prefixes=LPTC_TYPE_PREFIXES),
            "circuits": tangential_circuits(
                support, annotations, tangential=tangential, motor=motor
            ),
            "input_composition": tangential_input_composition(
                support, annotations, tangential=tangential
            ),
            "motion_pathway_column_addressing": motion_pathway_addressing(
                annotations, addresses.addressed
            ),
        },
        "interpretation_bounds": {
            "topology_and_annotation_only": (
                "every figure is either a count of source-observed annotation strings or"
                " a topological measurement over the retained edge support. Nothing here"
                " executes dynamics, and nothing here is evidence in the sense"
                " docs/SCIENTIFIC_CLAIMS.md reserves for G3"
            ),
            "reachability_is_an_upper_bound": (
                "reachability bounds participation and never demonstrates it. A node"
                " inside a path-support set is only topologically able to lie on a path,"
                " at an unmodelled and possibly vanishing, cancelling or sign-inverted"
                " gain. No route measured here is evidence that any signal travels it"
            ),
            "a_zero_edge_count_is_a_topological_fact": (
                "the absence of a retained edge between a patch and the committed fixture"
                " is a statement about this artifact's retained edge support at this"
                " confidence threshold. It bounds what a single-fixture design could"
                " connect; it is not a claim about the animal"
            ),
            "closure_size_is_not_a_capability_claim": (
                "a closure that exceeds the fixture ceiling is a statement about how many"
                " bodies and edges the selection rule would emit, not about whether the"
                " resulting circuit would work, be faithful, or produce any behaviour"
            ),
            "patch_geometry_is_this_analysis_own_choice": (
                "the patch types, the eye, the centroid rule and the lattice metric are"
                " choices made here and recorded here. They are not repository rules, and"
                " a different defensible choice would move the patch figures"
            ),
            "annotation_labels_are_dataset_labels": (
                "superclass, subclass, type, somaSide and the column address are pinned"
                " table annotations. They are not measured functions, and this analysis"
                " assigns no behaviour, receptive field or visual function to any neuron"
            ),
            "weights_are_raw_counts": (
                "edge weight is the retained raw synapse count as shipped: no"
                " normalisation, no sign policy and no efficacy claim is applied"
            ),
            "no_gate_movement": (
                "this analysis promotes no gate, invalidates no existing receipt, and"
                " changes no runtime behaviour"
            ),
        },
    }


# ----------------------------------------------------------------------- CLI


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--manifest-sha256", default=TRACKED_MANIFEST_SHA256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-reproduction-drift",
        action="store_true",
        help=(
            "record an OPTIC-CENSUS-01 mismatch in the report instead of refusing;"
            " for diagnosing a deviation, never for producing evidence"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.perf_counter()
    with load_full_runtime_artifact(
        args.artifact, expected_manifest_sha256=args.manifest_sha256
    ) as artifact:
        manifest = artifact.manifest
        identity = artifact_identity(manifest)
        declared_sha256 = declared_annotations_sha256(manifest)
        retained_by_superclass = dict(manifest.retained_by_superclass)
        declared_rows = manifest.counts["annotation_source_rows"]
        retained_weight_sum = manifest.counts["retained_weight_sum"]
        source_rows = np.array(mapped_numpy(artifact.annotation_source_rows, "<u8"))
        body_ids = np.array(mapped_numpy(artifact.body_ids, "<u8"))
        support = EdgeSupport(
            node_count=manifest.node_count,
            pre=np.array(mapped_numpy(artifact.edge_pre_indices, "<u4")),
            post=np.array(mapped_numpy(artifact.edge_post_indices, "<u4")),
            weights=np.array(mapped_numpy(artifact.edge_raw_counts, "<u8")).astype(np.int64),
        )

    annotations, taken = take_retained_annotations(
        args.annotations, source_rows=source_rows, body_ids=body_ids
    )
    fixture_graph = load_malecns_fixture(
        args.fixture,
        expected_pin=MALECNS_FLIGHT_FIXTURE_V1_PIN,
        expected_file_bytes=MALECNS_FLIGHT_FIXTURE_V1_FILE_BYTES,
    )
    fixture_graph.assert_integrity()
    fixture_annotations, fixture_taken = take_retained_annotations(
        args.annotations,
        source_rows=np.asarray(
            [node.annotation_source_row for node in fixture_graph.nodes], dtype=np.uint64
        ),
        body_ids=np.asarray([node.body_id for node in fixture_graph.nodes], dtype=np.uint64),
    )
    gates = [
        *annotation_gates(
            annotations,
            declared_annotations_sha256=declared_sha256,
            declared_source_rows=declared_rows,
            retained_by_superclass=retained_by_superclass,
        ),
        weight_sum_gate(support.weights, retained_weight_sum),
        *annotation_gates(
            fixture_annotations,
            declared_annotations_sha256=fixture_annotations_sha256(fixture_graph),
            declared_source_rows=declared_rows,
            retained_by_superclass=histogram(node.superclass for node in fixture_graph.nodes),
        ),
        *fixture_gates(fixture_graph, declared_sha256, declared_rows),
    ]
    analysis = analyse(
        annotations,
        taken,
        support,
        fixture_graph,
        fixture_annotations,
        fixture_taken,
        identity=identity,
        gates=gates,
    )
    if not analysis["optic_census_01_reproduction"]["reproduced"] and not (
        args.allow_reproduction_drift
    ):
        raise VisionReproductionError(
            "the column census did not reproduce"
            f" {analysis['optic_census_01_reproduction']['receipt']}; refusing to write"
            " a report"
        )
    digest = hashlib.sha256(canonical_json_bytes(analysis)).hexdigest()
    document = {
        "analysis": analysis,
        "analysis_sha256": digest,
        "literature_inferred_notes": LITERATURE_INFERRED_NOTES,
        "execution": {
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "peak_rss_bytes": peak_rss_bytes(),
            "numpy": np.__version__,
            "python": sys.version,
            "os": platform.platform(),
        },
    }
    write_report(args.output, document)
    print_summary(analysis, digest)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
