"""Report fragments and the human-readable summary for the Phase 2 assessment.

Split out of ``scripts/analyze_vision_feasibility.py`` so that the analysis
script stays inside the repository's file-size ceiling, exactly as
``scripts/reachability_graph.py`` was split out of the Phase 0 analysis. Nothing
here reads a file or decides anything: it turns the verified arrays the analysis
script binds into the serializable blocks of the report, writes the document
atomically, and prints the summary a reader sees at the terminal.

Claim discipline is inherited unchanged from the modules this one composes.
Every block below is a count of source-observed annotation strings or a
topological measurement over the retained edge support, no behaviour is assigned
to any neuron, and the word "causal" — reserved by
``docs/SCIENTIFIC_CLAIMS.md`` for the G3 gate — appears in nothing this module
produces.
"""

from __future__ import annotations

import json
import os
import resource
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from neurocraft_connectome.model import RuntimeGraph

from scripts.readout_annotations import (
    IntArray,
    RetainedAnnotations,
    full_row,
    histogram,
    key_of,
)
from scripts.vision_circuits import WEIGHT_THRESHOLDS, EdgeSupport, cross_connectivity
from scripts.vision_columns import (
    COLUMNAR_TYPES,
    DEFAULT_HEX_METRIC,
    HEX_METRICS,
    PATCH_SOMA_SIDE,
    PATCH_TYPES,
    ColumnIndex,
    Disc,
    HexAddresses,
    address_census,
    hex_addresses,
    is_contiguous,
    select_disc,
    type_family_report,
)

# ---------------------------------------------------------- fixture identity


def fixture_identity(graph: RuntimeGraph) -> dict[str, Any]:
    """Pinned identity of the flight-steering fixture as loaded."""

    manifest = graph.manifest
    return {
        "fixture_id": manifest.fixture_id,
        "dataset_id": manifest.dataset_id,
        "fixture_tier": str(manifest.fixture_tier),
        "fixture_file_sha256": manifest.fixture_file_sha256,
        "fixture_payload_sha256": manifest.fixture_payload_sha256,
        "fixture_semantic_sha256": manifest.fixture_semantic_sha256,
        "source_manifest_sha256": manifest.source_manifest_sha256,
        "canonical_graph_sha256": manifest.canonical_graph_sha256,
        "runtime_identity_sha256": manifest.runtime_identity_sha256,
        "node_count": manifest.node_count,
        "edge_count": manifest.edge_count,
        "raw_count_sum": manifest.raw_count_sum,
    }


# --------------------------------------------------- fixture visual substrate


def fixture_visual_substrate(
    graph: RuntimeGraph,
    annotations: RetainedAnnotations,
    taken: Any,
) -> dict[str, Any]:
    """Whether the committed flight fixture contains anything visual to inject into."""

    addresses = hex_addresses(taken)
    types = annotations.column("type")
    projection = [
        index
        for index, value in enumerate(annotations.column("superclass").tolist())
        if key_of(value).startswith("visual")
    ]
    rows = [
        {
            "body_id": int(annotations.body_ids[index]),
            "dense_index": index,
            "annotation_source_row": int(annotations.source_rows[index]),
            "superclass": key_of(annotations.column("superclass")[index]),
            "type": key_of(types[index]),
            "soma_side": key_of(annotations.column("somaSide")[index]),
            "roles": list(graph.nodes[index].roles),
            "column_address": (
                None
                if not bool(addresses.addressed[index])
                else {
                    "assignedOlHex1": int(addresses.hex1[index]),
                    "assignedOlHex2": int(addresses.hex2[index]),
                }
            ),
            "annotation_row": full_row(taken, index),
        }
        for index in projection
    ]
    return {
        "definition": (
            "the visual content of the committed flight-steering fixture, measured over"
            " the same pinned annotations table the full artifact uses. This is the fact"
            " that decides whether a retinotopic injection could reuse the committed"
            " fixture or would require a third one"
        ),
        "fixture_identity": fixture_identity(graph),
        "fixture_nodes": int(annotations.node_count),
        "superclass_composition": histogram(annotations.column("superclass").tolist()),
        "role_composition": histogram(role for node in graph.nodes for role in node.roles),
        "column_address_census": address_census(addresses, types, annotations.column("somaSide")),
        "visual_type_families": type_family_report(
            types, scope="the 128 nodes of the flight-steering fixture"
        ),
        "visual_superclass_nodes": rows,
        "finding": (
            "the fixture carries no column-addressed node and no member of any declared"
            " lamina, medulla or photoreceptor type family. Its only node with a visual"
            " superclass is a single projection cell holding an intermediate role. A"
            " retinotopic injection therefore has nothing in the committed fixture to"
            " address, and would require a third fixture with its own identity"
        ),
    }


# ------------------------------------------------------------- patch sizing


def _patch_row(
    support: EdgeSupport,
    index: ColumnIndex,
    disc: Disc,
    annotations: RetainedAnnotations,
    *,
    columns: int,
    patch_types: Sequence[str],
    thresholds: Sequence[int],
) -> dict[str, Any]:
    nodes = index.cells(disc.columns, patch_types)
    types = annotations.column("type")
    per_type = histogram(types[node] for node in nodes.tolist())
    return {
        "columns": columns,
        "hex_metric": disc.metric,
        "centre_column": list(disc.centre),
        "radius": disc.radius,
        "contiguous": is_contiguous(disc.columns, disc.metric),
        "selected_columns": [list(column) for column in disc.columns],
        "nodes": int(nodes.size),
        "nodes_per_type": per_type,
        "cells_per_column": _fraction(int(nodes.size), columns),
        "induced_edges": support.induced_edges(nodes, thresholds=thresholds),
        "induced_weight": support.induced_weight(nodes),
        "body_ids": sorted(int(annotations.body_ids[node]) for node in nodes.tolist()),
    }


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def patch_sizing(
    support: EdgeSupport,
    index: ColumnIndex,
    annotations: RetainedAnnotations,
    complete: Sequence[tuple[int, int]],
    *,
    sizes: Sequence[int],
    patch_types: Sequence[str] = PATCH_TYPES,
    metric: str = DEFAULT_HEX_METRIC,
    thresholds: Sequence[int] = WEIGHT_THRESHOLDS,
) -> dict[str, Any]:
    """Node and induced-edge counts of the retinotopic patch at each size."""

    rows = [
        _patch_row(
            support,
            index,
            select_disc(complete, size, metric=metric),
            annotations,
            columns=size,
            patch_types=patch_types,
            thresholds=thresholds,
        )
        for size in sizes
    ]
    return {
        "definition": (
            "a patch is every cell of the declared patch types inside a selected set of"
            " complete right-eye hex columns. Induced edges are the retained edges with"
            " both endpoints inside the patch"
        ),
        "selection_rule": (
            "the centroid is the arithmetic mean of the complete columns' addresses; the"
            " centre is the complete column closest to it; the patch is the K complete"
            " columns nearest the centre, ordered by lattice distance then by address."
            " This is this analysis's own choice of patch geometry and is NOT a frozen"
            " repository rule; the selected columns are listed so any other rule can be"
            " compared against it"
        ),
        "soma_side": PATCH_SOMA_SIDE,
        "patch_types": list(patch_types),
        "hex_metric": metric,
        "complete_columns_available": len(complete),
        "rows": rows,
    }


def metric_ambiguity(
    support: EdgeSupport,
    index: ColumnIndex,
    complete: Sequence[tuple[int, int]],
    *,
    sizes: Sequence[int],
    patch_types: Sequence[str] = PATCH_TYPES,
    metrics: Sequence[str] = HEX_METRICS,
    reference: str = DEFAULT_HEX_METRIC,
    thresholds: Sequence[int] = WEIGHT_THRESHOLDS,
) -> dict[str, Any]:
    """How much the patch depends on an unfrozen hex-axis convention."""

    measured: dict[str, dict[int, dict[str, Any]]] = {}
    for metric in metrics:
        measured[metric] = {}
        for size in sizes:
            disc = select_disc(complete, size, metric=metric)
            nodes = index.cells(disc.columns, patch_types)
            measured[metric][size] = {
                "columns": [list(column) for column in disc.columns],
                "column_set": {tuple(column) for column in disc.columns},
                "nodes": int(nodes.size),
                "induced_edges": support.induced_edges(nodes, thresholds=thresholds),
            }
    rows: list[dict[str, Any]] = []
    worst = 0.0
    for metric in metrics:
        for size in sizes:
            entry = measured[metric][size]
            base = measured[reference][size]
            shared = len(entry["column_set"] & base["column_set"])
            deviations = {
                key: abs(int(entry["induced_edges"][key]) - int(base["induced_edges"][key]))
                / int(base["induced_edges"][key])
                for key in base["induced_edges"]
                if int(base["induced_edges"][key]) > 0
            }
            largest = max(deviations.values(), default=0.0)
            if metric != reference:
                worst = max(worst, largest)
            rows.append(
                {
                    "hex_metric": metric,
                    "columns": size,
                    "nodes": int(entry["nodes"]),
                    "induced_edges": entry["induced_edges"],
                    "columns_shared_with_reference": shared,
                    "columns_shared_with_reference_fraction": _fraction(shared, size),
                    "largest_relative_edge_deviation_from_reference": largest,
                }
            )
    return {
        "definition": (
            "the same patch selection repeated under every candidate lattice metric over"
            " the (assignedOlHex1, assignedOlHex2) address pair, with the column overlap"
            " and the largest relative induced-edge deviation against the reference"
        ),
        "ambiguity": (
            "neither the pinned annotations table nor this repository declares which of"
            " the two hex axes are sixty degrees apart and which are one hundred and"
            " twenty, so the distance used to grow a disc is a convention this analysis"
            " had to choose. The spread below is the cost of that choice"
        ),
        "reference_metric": reference,
        "rows": rows,
        "largest_relative_edge_deviation_across_metrics": worst,
    }


# -------------------------------------------------------------- cross-edges


def cross_connectivity_report(
    support: EdgeSupport,
    index: ColumnIndex,
    annotations: RetainedAnnotations,
    addresses: HexAddresses,
    complete: Sequence[tuple[int, int]],
    fixture_nodes: IntArray,
    *,
    sizes: Sequence[int],
    patch_types: Sequence[str] = PATCH_TYPES,
    columnar_types: Sequence[str] = COLUMNAR_TYPES,
    metric: str = DEFAULT_HEX_METRIC,
) -> dict[str, Any]:
    """Edges between each candidate visual population and the committed fixture."""

    types = annotations.column("type")
    populations: list[tuple[str, IntArray]] = [
        (
            f"patch_of_{size}_complete_columns",
            index.cells(select_disc(complete, size, metric=metric).columns, patch_types),
        )
        for size in sizes
    ]
    populations.append(
        (
            "all_patch_type_cells_in_every_complete_column",
            index.cells(complete, patch_types),
        )
    )
    populations.append(
        (
            "all_cells_in_every_complete_column",
            index.cells(complete, columnar_types),
        )
    )
    populations.append(
        (
            "all_column_addressed_nodes",
            np.flatnonzero(addresses.addressed).astype(np.int64),
        )
    )
    rows = [
        cross_connectivity(
            support,
            left=nodes,
            right=fixture_nodes,
            left_name=name,
            right_name="flight_steering_fixture",
            types=types,
            body_ids=annotations.body_ids,
            threshold=1,
        )
        for name, nodes in populations
    ]
    return {
        "definition": (
            "retained edges of weight at least one between each candidate visual"
            " population and the 128 bodies of the committed flight-steering fixture,"
            " counted directly on the edge list in both directions"
        ),
        "why_direct": (
            "this measurement exists to establish a zero. A zero obtained by subtracting"
            " two large aggregates is not the same evidence as a zero obtained by"
            " scanning the edge list for an edge and finding none, so the direct scan is"
            " the only route used"
        ),
        "rows": rows,
        "populations_with_no_edge_to_the_fixture": sorted(
            str(row["left_set"]) for row in rows if int(row["total_edges"]) == 0
        ),
    }


# --------------------------------------------------------------------- output


def peak_rss_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return observed if sys.platform == "darwin" else observed * 1024


def write_report(path: Path, document: dict[str, Any]) -> None:
    encoded = json.dumps(document, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _print_gates(analysis: Mapping[str, Any]) -> None:
    gates = analysis["integrity_gates"]
    print(f"integrity gates: {len(gates['gates'])} evaluated, all_passed={gates['all_passed']}")
    for gate in gates["gates"]:
        print(f"  [{'PASS' if gate['passed'] else 'FAIL'}] {gate['name']}")


def _print_fixture(analysis: Mapping[str, Any]) -> None:
    block = analysis["flight_fixture_visual_substrate"]
    census = block["column_address_census"]
    print(
        f"flight fixture {block['fixture_identity']['fixture_id']}:"
        f" {block['fixture_nodes']} nodes,"
        f" {census['column_addressed_nodes']} column-addressed"
    )
    for row in block["visual_type_families"]["families"]:
        print(f"  family {row['family']:<8} cells {row['cells']}")
    for row in block["visual_superclass_nodes"]:
        print(
            f"  visual node body {row['body_id']} type {row['type']}"
            f" side {row['soma_side']} roles {row['roles']}"
        )


def _print_columns(analysis: Mapping[str, Any]) -> None:
    for row in analysis["column_address_census"]["rows"]:
        print(
            f"eye {row['soma_side']}: {row['columns']} columns,"
            f" {row['columns_complete_over_all_columnar_types']} complete over all types,"
            f" {row['columns_complete_over_the_patch_types']} complete over the patch types"
        )
    check = analysis["optic_census_01_reproduction"]
    print(f"OPTIC-CENSUS-01 reproduction: {check['reproduced']}")
    for row in check["checks"]:
        print(
            f"  {row['quantity']:<52} observed {row['observed']:>7} published {row['published']:>7}"
        )


def _print_patches(analysis: Mapping[str, Any]) -> None:
    print("columns  nodes   w>=1   w>=5  w>=10  w>=20  contiguous")
    for row in analysis["retinotopic_patch_sizing"]["rows"]:
        edges = row["induced_edges"]
        print(
            f"{row['columns']:>7}  {row['nodes']:>5}"
            f"  {edges['1']:>5}  {edges['5']:>5}  {edges['10']:>5}  {edges['20']:>5}"
            f"  {row['contiguous']}"
        )
    ambiguity = analysis["hex_metric_ambiguity"]
    print(
        "largest relative induced-edge deviation across hex metrics:"
        f" {ambiguity['largest_relative_edge_deviation_across_metrics']:.4%}"
    )


def _print_cross(analysis: Mapping[str, Any]) -> None:
    print("population                                       nodes  ->fix  fix->")
    for row in analysis["patch_to_flight_fixture_cross_connectivity"]["rows"]:
        print(
            f"{row['left_set']:<46}  {row['left_nodes']:>6}"
            f"  {row['left_to_right_edges']:>5}  {row['right_to_left_edges']:>5}"
        )


def _print_support(analysis: Mapping[str, Any]) -> None:
    print("w>=  columns  cells  within2  fraction")
    for row in analysis["patch_to_descending_support"]["rows"]:
        print(
            f"{row['minimum_edge_weight']:>3}  {row['columns']:>7}  {row['patch_cells']:>5}"
            f"  {row['cells_within_maximum_hops']:>7}"
            f"  {row['fraction_within_maximum_hops']:>8.4%}"
        )


def _print_closure(analysis: Mapping[str, Any]) -> None:
    print("columns  closure                       pool   nodes      w>=1      w>=5  xceiling")
    for block in analysis["patch_circuit_closure"]["rows"]:
        for row in block["tiers"]:
            edges = row["induced_edges"]
            print(
                f"{block['columns']:>7}  {row['closure']:<26}"
                f"  {row['intermediate_pool_bodies']:>6}  {row['closure_nodes']:>6}"
                f"  {edges['1']:>8}  {edges['5']:>8}"
                f"  {row['multiple_of_the_fixture_edge_ceiling']:>7.2f}"
            )


def _print_tangential(analysis: Mapping[str, Any]) -> None:
    block = analysis["lobula_plate_tangential_alternative"]
    census = block["census"]
    print(f"tangential cells: {census['cells']} sides {census['soma_side_counts']}")
    print(f"  types: {census['type_counts']}")
    for row in block["circuits"]["rows"]:
        edges = row["induced_edges"]
        print(
            f"  {row['circuit']:<38} nodes {row['nodes']:>4}"
            f"  {edges['1']:>5}  {edges['5']:>5}  {edges['10']:>5}  {edges['20']:>5}"
        )
    for row in block["input_composition"]["rows"]:
        print(
            f"  input w>={row['minimum_weight']:>2}: motion pathway"
            f" {row['motion_pathway_weight']:>8,} of {row['total_incoming_weight']:>8,}"
            f" = {row['motion_pathway_weight_fraction']:.4%}"
        )
    addressing = block["motion_pathway_column_addressing"]
    print(
        f"  motion pathway cells {addressing['cells']},"
        f" column-addressed {addressing['cells_with_a_column_address']}"
    )


def print_summary(analysis: Mapping[str, Any], digest: str) -> None:
    print(f"artifact {analysis['artifact']['artifact_id']}")
    print(f"manifest {analysis['artifact']['manifest_file_sha256'][:12]}")
    _print_gates(analysis)
    _print_fixture(analysis)
    _print_columns(analysis)
    _print_patches(analysis)
    _print_cross(analysis)
    _print_support(analysis)
    _print_closure(analysis)
    _print_tangential(analysis)
    print(f"analysis_sha256 {digest}")
