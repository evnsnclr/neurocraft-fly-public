#!/usr/bin/env python3
"""Deterministically extract the MaleCNS v1.0 flight-steering fixture.

This is a second, independent circuit selection over the same three pinned
Feather objects that ``scripts/extract_malecns_fixture.py`` consumes. It does not
modify, re-point, or supersede the flight-power fixture
``mcns-v1.0-flight-power-v1``; the two fixtures coexist, each carries its own
identity, and the flight-power fixture remains the authority for the full
artifact boundary overlay.

The selection is wider than the flight-power circuit. It resolves a named
bilateral descending population, the wing-motor bodies that population contacts
directly, the two pinned sensory entries, and the maximum-bottleneck path support
linking each sensory entry to each descending body. The emitted graph is the
weight-thresholded closure over exactly those bodies.

Discipline mirrors the flight-power extractor: only manifest-verified source
bytes are parsed, every directed pair is checked for duplicates across the whole
weight table, every emitted row keeps zero-based source-row lineage, no
transmitter-to-sign mapping is applied, and the output is byte-identical across
runs.

Layering: :mod:`scripts.flight_fixture_sources` owns the verified source bytes,
:mod:`scripts.flight_fixture_circuit` owns the frozen selection policy, and this
module owns the canonical document, its provenance and rights block, and the CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # direct `python scripts/extract_flight_fixture.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.flight_fixture_circuit import (
    DESCENDING_TYPES,
    INDUCED_MIN_WEIGHT,
    MAX_EMITTED_EDGES,
    MAX_PATH_HOPS,
    MOTOR_SUBCLASS,
    MOTOR_SUPERCLASS,
    PATH_MIN_WEIGHT,
    READOUT_MIN_WEIGHT,
    ROLE_ORDER,
    SENSORY_BODY_IDS,
    Edge,
    SelectedPath,
    derive_side,
    resolve_circuit,
)
from scripts.flight_fixture_sources import (
    DATASET_ID,
    EXPECTED_MANIFEST_SHA256,
    REQUIRED_SOURCE_IDS,
    AnnotationRecord,
    CircuitResolutionError,
    FixtureExtractionError,
    Manifest,
    RetainedWeights,
    canonical_bytes,
    load_annotations,
    load_manifest,
    load_transmitters,
    load_weights,
    normalize_json,
    require_mapping,
    select_transmitters,
    sha256_bytes,
    sha256_file,
    verify_sources,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "data" / "manifests" / "male-cns-v1.0.sources.json"
DEFAULT_CACHE = REPOSITORY_ROOT / "data" / "cache" / "male-cns-v1.0"
FIXTURE_ID = "mcns-v1.0-flight-steering-v1"
TRANSMITTER_EVIDENCE_FIELDS = (
    "ground_truth",
    "consensus_nt",
    "predicted_nt",
    "celltype_predicted_nt",
)
IMPLEMENTATION_FILES = (
    "extract_flight_fixture.py",
    "flight_fixture_circuit.py",
    "flight_fixture_sources.py",
)


def implementation_hashes() -> dict[str, str]:
    """Hash every project-owned file that defines this extraction."""

    directory = Path(__file__).resolve().parent
    per_file = {name: sha256_file(directory / name) for name in IMPLEMENTATION_FILES}
    combined = "".join(f"{name}:{per_file[name]};" for name in IMPLEMENTATION_FILES)
    return {**per_file, "combined": sha256_bytes(combined.encode("ascii"))}


def _edge_document(edge: Edge) -> dict[str, Any]:
    return {
        "body_post": edge.body_post,
        "body_pre": edge.body_pre,
        "sign_status": "unknown",
        "source_row": edge.source_row,
        "weight": edge.weight,
    }


def _path_document(path: SelectedPath) -> dict[str, Any]:
    return {
        "body_id_tuple": list(path.nodes),
        "bottleneck_weight": path.bottleneck,
        "descending_body_id": path.target_body_id,
        "edge_count": len(path.edges),
        "edges": [_edge_document(edge) for edge in path.edges],
        "sensory_body_id": path.source_body_id,
    }


def _transmitter_document(
    body_id: int, transmitters: Mapping[int, tuple[int, Mapping[str, Any]]]
) -> dict[str, Any]:
    item = transmitters.get(body_id)
    if item is None:
        raise CircuitResolutionError(
            f"selected body {body_id} has no aggregate transmitter source row"
        )
    source_row, record = item
    evidence = [record.get(field) for field in TRANSMITTER_EVIDENCE_FIELDS if field in record]

    def is_resolved(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip().lower() not in {"", "unclear", "unknown", "n/a", "na"}
        return True

    reported = any(is_resolved(value) for value in evidence)
    return {
        "record": record,
        "sign_status": "unknown",
        "source_row": source_row,
        "status": "reported_unmapped" if reported else "unknown_transmitter",
    }


def selection_policy() -> dict[str, Any]:
    """The frozen, hashed rules that define this fixture's semantics."""

    return {
        "candidate_resolution": {
            "descending_types_exact": list(DESCENDING_TYPES),
            "motor_readout": {
                "definition": (
                    "every exact vnc_motor/wm body that receives at least one direct source "
                    "edge from a declared descending body"
                ),
                "minimum_weight_inclusive": READOUT_MIN_WEIGHT,
                "subclass_exact": MOTOR_SUBCLASS,
                "superclass_exact": MOTOR_SUPERCLASS,
            },
            "sensory_entries_pinned": list(sorted(SENSORY_BODY_IDS)),
            "sensory_entry_verification": (
                "each pinned entry must independently satisfy the exact rule 'type starts with "
                "JO-E' or 'subclass campaniform sensilla entering ADMN/DMetaN'; a pinned ID is "
                "never accepted on its number alone"
            ),
            "string_matching": "case-sensitive exact except the declared JO-E prefix",
        },
        "fixture_closure": {
            "definition": "all source edges whose endpoints are both selected bodies",
            "emitted_edge_ceiling_exclusive": MAX_EMITTED_EDGES,
            "minimum_weight_inclusive": INDUCED_MIN_WEIGHT,
            "self_edges": "included when present",
        },
        "path_support": {
            "definition": (
                "For each ordered (pinned sensory entry, declared descending body) pair, one "
                "simple directed path of 1..3 source edges is selected over the subgraph of "
                "source edges with weight >= 10 whose endpoints both carry an annotation row. "
                "Every body on every selected path joins the fixture. Pairs with no qualifying "
                "path are recorded and skipped, never substituted."
            ),
            "direction": "body_pre -> body_post",
            "duplicate_directed_pair_policy": "fail across every source weight row",
            "maximum_edges_per_path": MAX_PATH_HOPS,
            "objective": "maximize the minimum raw source weight over the path",
            "path_edge_minimum_weight_inclusive": PATH_MIN_WEIGHT,
            "path_vertex_eligibility": "every path body must have a source annotation row",
            "tie_break": (
                "lexicographically smallest ordered source body-ID tuple among all "
                "maximum-bottleneck paths, compared across path lengths"
            ),
            "vertex_repetition": "forbidden within a path",
        },
        "roles": {
            "assignment": (
                "sensory for the pinned entries, descending for the declared types, motor for "
                "the resolved wing-motor readout, intermediate for every other selected body"
            ),
            "order": list(ROLE_ORDER),
        },
        "side_derivation": {
            "allowed_evidence": [
                "rootSide exact L/R",
                "somaSide exact L/R",
                "instance _L/_R suffix",
            ],
            "conflict": "fail",
            "unresolved_side": "retain as null",
        },
        "transmitter_and_sign": {
            "mapping": "none",
            "missing_transmitter_row": "fail",
            "policy": (
                "copy the exact aggregate transmitter row for every selected body; "
                "retain unknown edge signs"
            ),
        },
    }


def _strip_source_rows(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_source_rows(item) for key, item in value.items() if key != "source_row"}
    if isinstance(value, list):
        return [_strip_source_rows(item) for item in value]
    return value


def _semantic_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    projection = {
        key: value
        for key, value in payload.items()
        if key not in {"input_receipt", "extraction_receipt"}
    }
    stripped: dict[str, Any] = _strip_source_rows(projection)
    return stripped


def _rights_document(manifest: Manifest, *, node_count: int, edge_count: int) -> dict[str, Any]:
    citation = require_mapping(manifest.dataset.get("citation"), "manifest.dataset.citation")
    return {
        "attribution": (
            "MaleCNS v1.0 connectome data by FlyEM (HHMI Janelia), the "
            "University of Cambridge Department of Zoology, the MRC Laboratory "
            "of Molecular Biology, and Google Research."
        ),
        "changes": (
            f"Deterministically selected {node_count} source neurons: the two pinned sensory "
            "entries, every exact DNg02_a-g, DNp03 and DNp31 descending body, the exact "
            "vnc_motor/wm bodies those descending bodies contact directly at weight >= 5, and "
            "the maximum-bottleneck sensory-to-descending path support of at most three source "
            f"edges of weight >= 10; emitted all {edge_count} directed source edges of "
            "weight >= 5 among them; copied selected annotation and aggregate transmitter "
            "evidence; retained every edge sign as unknown; did not modify source values."
        ),
        "citation": normalize_json(citation),
        "license": "CC BY 4.0",
        "license_spdx": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/legalcode.en",
        "redistribution_decision": "permitted_with_attribution_and_change_notice",
        "redistribution_scope": (
            "This small derived JSON fixture may be committed and redistributed "
            "with this attribution, license link, citation, provenance, and change "
            "notice. The three bulk source objects remain excluded from Git."
        ),
        "source": "https://male-cns.janelia.org/download/",
        "source_dataset_remains_third_party": True,
    }


def _node_document(
    body_id: int,
    annotation: AnnotationRecord,
    roles: Sequence[str],
    transmitters: Mapping[int, tuple[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    side, evidence = derive_side(annotation.record, body_id)
    return {
        "annotation": {"record": annotation.record, "source_row": annotation.source_row},
        "body_id": body_id,
        "roles": list(roles),
        "side": side,
        "side_evidence": evidence,
        "transmitter": _transmitter_document(body_id, transmitters),
    }


def _composition(
    body_ids: Sequence[int], annotations: Mapping[int, AnnotationRecord]
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for body_id in body_ids:
        value = annotations[body_id].record.get("superclass")
        key = value if isinstance(value, str) else "null"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _side_balance(
    body_ids: Sequence[int], annotations: Mapping[int, AnnotationRecord]
) -> dict[str, int]:
    counts = {"L": 0, "R": 0, "null": 0}
    for body_id in body_ids:
        side, _ = derive_side(annotations[body_id].record, body_id)
        counts[side if side is not None else "null"] += 1
    return counts


def _source_receipts(manifest: Manifest) -> list[dict[str, Any]]:
    return [
        {
            "bytes": manifest.sources[source_id].expected_bytes,
            "filename": manifest.sources[source_id].filename,
            "generation": manifest.sources[source_id].generation,
            "id": source_id,
            "sha256": manifest.sources[source_id].sha256,
            "url": manifest.sources[source_id].url,
        }
        for source_id in REQUIRED_SOURCE_IDS
    ]


def build_fixture(
    manifest: Manifest,
    annotations: Mapping[int, AnnotationRecord],
    annotation_row_count: int,
    transmitter_table: Any,
    retained: RetainedWeights,
) -> dict[str, Any]:
    """Resolve the circuit and return the complete canonical fixture document."""

    circuit = resolve_circuit(annotations, retained)
    selected = circuit.selected
    edges = circuit.edges
    transmitters = select_transmitters(transmitter_table, set(selected))
    role_sets = {
        "sensory": set(circuit.sensory),
        "descending": set(circuit.descending),
        "motor": set(circuit.readout),
    }
    nodes: list[dict[str, Any]] = []
    unreported = 0
    for body_id in selected:
        roles = [role for role in ROLE_ORDER if body_id in role_sets.get(role, set())]
        document = _node_document(
            body_id, annotations[body_id], roles or ["intermediate"], transmitters
        )
        if document["transmitter"]["status"] != "reported_unmapped":
            unreported += 1
        nodes.append(document)

    policy = selection_policy()
    hashes = implementation_hashes()
    payload: dict[str, Any] = {
        "circuit_resolution": {
            "descending_by_type": {
                name: {"body_ids": list(bodies), "count": len(bodies)}
                for name, bodies in sorted(circuit.descending_by_type.items())
            },
            "descending_side_balance": _side_balance(circuit.descending, annotations),
            "intermediate_body_ids": list(circuit.intermediates),
            "motor_candidate_count": circuit.motor_candidate_count,
            "motor_readout_body_ids": list(circuit.readout),
            "motor_readout_side_balance": _side_balance(circuit.readout, annotations),
            "path_support_body_ids": list(circuit.path_support),
            "selected_side_balance": _side_balance(selected, annotations),
            "sensory_body_ids": list(circuit.sensory),
            "superclass_composition": _composition(selected, annotations),
            "unresolved_pairs": [
                {"descending_body_id": target, "sensory_body_id": source}
                for source, target in circuit.unresolved_pairs
            ],
        },
        "counts": {
            "annotation_source_rows": annotation_row_count,
            "descending_body_rows": len(circuit.descending),
            "descending_to_motor_contact_rows": circuit.descending_to_motor_contact_rows,
            "emitted_edge_rows": len(edges),
            "emitted_node_rows": len(nodes),
            "emitted_weight_sum": sum(edge.weight for edge in edges),
            "intermediate_body_rows": len(circuit.intermediates),
            "motor_candidate_rows": circuit.motor_candidate_count,
            "motor_readout_rows": len(circuit.readout),
            "neurotransmitter_source_rows": int(transmitter_table.num_rows),
            "path_subgraph_edge_rows": circuit.path_subgraph_edge_rows,
            "path_support_body_rows": len(circuit.path_support),
            "resolved_path_pairs": len(circuit.paths),
            "selected_nodes_without_reported_transmitter": unreported,
            "unresolved_path_pairs": len(circuit.unresolved_pairs),
            **dict(retained.counts),
        },
        "dataset_id": DATASET_ID,
        "edges": [_edge_document(edge) for edge in edges],
        "extraction_receipt": {
            "code_sha256": hashes["combined"],
            "code_scope": (
                "SHA-256 over concatenated 'name:file_sha256;' fields for "
                "extract_flight_fixture.py, flight_fixture_circuit.py, and "
                "flight_fixture_sources.py in that order"
            ),
            "configuration_sha256": sha256_bytes(canonical_bytes(policy)),
            "implementation": "extract_flight_fixture.py",
            "implementation_files": {name: hashes[name] for name in sorted(IMPLEMENTATION_FILES)},
        },
        "fixture_id": FIXTURE_ID,
        "input_receipt": {
            "dataset": normalize_json(manifest.dataset),
            "manifest": {
                "bytes": len(manifest.raw_bytes),
                "required_sha256": manifest.required_sha256,
                "sha256": manifest.raw_sha256,
            },
            "sources": _source_receipts(manifest),
        },
        "nodes": nodes,
        "rights": _rights_document(manifest, node_count=len(nodes), edge_count=len(edges)),
        "schema_version": 1,
        "selected_paths": [_path_document(path) for path in circuit.paths],
        "selection_policy": policy,
        "sibling_fixture_boundary": (
            "This fixture coexists with mcns-v1.0-flight-power-v1 and never replaces it; the "
            "flight-power fixture remains the authority for the full artifact boundary overlay."
        ),
        "transmitter_sign_boundary": (
            "No transmitter-to-sign mapping was applied; every emitted edge sign remains unknown."
        ),
    }

    payload["hashes"] = {
        "payload_sha256": sha256_bytes(canonical_bytes(payload)),
        "payload_scope": "canonical JSON document with the hashes object omitted",
        "semantic_sha256": sha256_bytes(canonical_bytes(_semantic_projection(payload))),
        "semantic_scope": (
            "canonical payload excluding input/extraction receipts and every source_row; "
            "this identifies selection semantics while payload_sha256 preserves exact lineage"
        ),
    }
    return payload


def write_fixture(document: Mapping[str, Any], output_path: Path) -> None:
    """Write the canonical bytes atomically, never leaving a partial file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_bytes(document) + b"\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def extract_fixture(
    manifest_path: Path,
    cache_dir: Path,
    output_path: Path,
    *,
    scratch_dir: Path | None = None,
    expected_manifest_sha256: str = EXPECTED_MANIFEST_SHA256,
) -> dict[str, Any]:
    """Verify, extract, atomically write, and return the fixture document."""

    manifest = load_manifest(manifest_path, expected_sha256=expected_manifest_sha256)
    source_paths = verify_sources(manifest, cache_dir)
    annotations, annotation_rows = load_annotations(source_paths["annotations"])
    transmitters = load_transmitters(source_paths["neurotransmitters"])
    retained = load_weights(
        source_paths["weights"],
        path_min_weight=PATH_MIN_WEIGHT,
        induced_min_weight=INDUCED_MIN_WEIGHT,
        scratch_dir=scratch_dir,
    )
    document = build_fixture(manifest, annotations, annotation_rows, transmitters, retained)
    write_fixture(document, output_path)
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract the deterministic MaleCNS v1.0 flight-steering fixture"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=None,
        help=(
            "directory for the exact duplicate-pair sort (about 8 bytes per source edge; "
            "system temporary storage by default)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        document = extract_fixture(
            args.manifest,
            args.cache_dir,
            args.output,
            scratch_dir=args.scratch_dir,
        )
    except FixtureExtractionError as exc:
        print(f"fixture extraction failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "edge_count": document["counts"]["emitted_edge_rows"],
                "fixture_id": document["fixture_id"],
                "node_count": document["counts"]["emitted_node_rows"],
                "output": str(args.output),
                "payload_sha256": document["hashes"]["payload_sha256"],
                "semantic_sha256": document["hashes"]["semantic_sha256"],
                "status": "written",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
