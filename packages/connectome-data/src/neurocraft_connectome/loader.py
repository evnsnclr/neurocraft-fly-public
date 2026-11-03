"""Strict loader for the pinned MaleCNS v1.0 CI fixture."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any

from neurocraft_connectome.model import (
    UINT64_MAX,
    FixturePin,
    FixtureTier,
    RuntimeEdge,
    RuntimeGraph,
    RuntimeNode,
    SourceArtifact,
    SyntheticEdge,
    SyntheticNode,
    canonical_json_bytes,
    graph_hashes,
    manifest_identity_material,
    sha256_json,
)

LOADER_NAME = "neurocraft-connectome"
LOADER_VERSION = "1"
RUNTIME_SCHEMA_VERSION = 1
LOADER_CONFIGURATION = {
    "duplicate_directed_pair_policy": "reject",
    "edge_order": ["body_post", "body_pre", "source_row"],
    "node_order": ["body_id"],
    "raw_count_semantics": "source synapse count, immutable",
    "self_edge_policy": "preserve",
    "unknown_sign_policy": "preserve_unknown_in_canonical_layer",
}


def _loader_implementation_sha256() -> str:
    """Hash all project-owned code that defines canonical graph semantics."""

    digest = sha256()
    package_root = Path(__file__).resolve().parent
    for filename in ("loader.py", "model.py"):
        raw = (package_root / filename).read_bytes()
        digest.update(filename.encode("ascii"))
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


class FixtureIdentityError(RuntimeError):
    """Fixture bytes or embedded identities do not match the trusted pin."""


class FixtureSchemaError(RuntimeError):
    """Fixture content cannot be represented without an unsafe assumption."""


class GraphIntegrityError(RuntimeError):
    """Runtime arrays or their content-addressed manifest were modified."""


MALECNS_FIXTURE_V1_PIN = FixturePin(
    dataset_id="male-cns:v1.0",
    fixture_id="mcns-v1.0-flight-power-v1",
    schema_version=1,
    file_sha256="3076c22fd0c87041c601b865c9fc8e913466200491f80775060e28412a087ff8",
    payload_sha256="69b9a929edd84fa02584de2c18d4bc1cacd1e3fe20ff64be6fe0ca8657bd9888",
    semantic_sha256="8360e414a3df4004fe7bfcf03bdfba8d2fb4d188c6cdd93ef30b7760055f9232",
    source_manifest_sha256=("9412105f54884a57521bbdf7fe46659caab559ef7877e6a62af95f50e1ee3768"),
    source_artifacts=(
        SourceArtifact(
            source_id="annotations",
            filename="body-annotations-male-cns-v1.0-minconf-0.5.feather",
            byte_length=14_483_314,
            sha256="2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
            generation="1780494878811468",
            url=(
                "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/"
                "flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather"
                "?generation=1780494878811468"
            ),
        ),
        SourceArtifact(
            source_id="neurotransmitters",
            filename="body-neurotransmitters-male-cns-v1.0.feather",
            byte_length=43_282_834,
            sha256="95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
            generation="1780894899156750",
            url=(
                "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/"
                "flat-connectome/body-neurotransmitters-male-cns-v1.0.feather"
                "?generation=1780894899156750"
            ),
        ),
        SourceArtifact(
            source_id="weights",
            filename="connectome-weights-male-cns-v1.0-minconf-0.5.feather",
            byte_length=1_051_241_946,
            sha256="e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
            generation="1780494887545976",
            url=(
                "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/"
                "flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
                "?generation=1780494887545976"
            ),
        ),
    ),
)
MALECNS_FIXTURE_V1_FILE_BYTES = 36_707
"""Exact byte length of the pinned fixture, gated before any byte is read."""


def _read_exact_regular_file(path: Path, *, expected_bytes: int) -> bytes:
    """Read a file only after one descriptor proves its type and exact length.

    The size gate runs against ``fstat`` on the descriptor that is then read, so
    a file that is not regular, or that does not have the pinned byte length, is
    refused before any of its content is allocated.
    """

    with path.open("rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise FixtureIdentityError("fixture must be a regular file")
        if info.st_size != expected_bytes:
            raise FixtureIdentityError(
                "fixture file size mismatch: "
                f"expected {expected_bytes} bytes, observed {info.st_size}"
            )
        raw = handle.read(expected_bytes + 1)
    if len(raw) != expected_bytes:
        raise FixtureIdentityError(
            f"fixture file length changed during the gated read: read {len(raw)} bytes"
        )
    return raw


def _reject_constant(value: str) -> None:
    raise FixtureSchemaError(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FixtureSchemaError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise FixtureSchemaError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise FixtureSchemaError(f"{label} must be an array")
    return value


def _string(value: Any, label: str, *, allow_none: bool = False) -> str | None:
    if allow_none and value is None:
        return None
    if not isinstance(value, str) or not value:
        raise FixtureSchemaError(f"{label} must be a non-empty string")
    return value


def _integer(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = UINT64_MAX,
    allow_none: bool = False,
) -> int | None:
    if allow_none and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise FixtureSchemaError(f"{label} must be an integer")
    if value < minimum or value > maximum:
        raise FixtureSchemaError(f"{label} must be in the inclusive range [{minimum}, {maximum}]")
    return value


def _strip_source_rows(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_source_rows(item) for key, item in value.items() if key != "source_row"}
    if isinstance(value, list):
        return [_strip_source_rows(item) for item in value]
    return value


def _verify_document_identity(document: Mapping[str, Any], pin: FixturePin) -> None:
    if document.get("schema_version") != pin.schema_version:
        raise FixtureIdentityError("fixture schema version does not match trusted pin")
    if document.get("dataset_id") != pin.dataset_id:
        raise FixtureIdentityError("fixture dataset identity does not match trusted pin")
    if document.get("fixture_id") != pin.fixture_id:
        raise FixtureIdentityError("fixture identity does not match trusted pin")
    hashes = _mapping(document.get("hashes"), "hashes")
    declared_payload = hashes.get("payload_sha256")
    declared_semantic = hashes.get("semantic_sha256")
    payload = {key: value for key, value in document.items() if key != "hashes"}
    observed_payload = sha256(canonical_json_bytes(payload)).hexdigest()
    semantic = {
        key: value
        for key, value in payload.items()
        if key not in {"input_receipt", "extraction_receipt"}
    }
    observed_semantic = sha256(canonical_json_bytes(_strip_source_rows(semantic))).hexdigest()
    expected_values = {
        "declared payload": (declared_payload, pin.payload_sha256),
        "computed payload": (observed_payload, pin.payload_sha256),
        "declared semantic": (declared_semantic, pin.semantic_sha256),
        "computed semantic": (observed_semantic, pin.semantic_sha256),
    }
    mismatches = [name for name, pair in expected_values.items() if pair[0] != pair[1]]
    if mismatches:
        raise FixtureIdentityError("fixture embedded identity mismatch: " + ", ".join(mismatches))

    receipt = _mapping(document.get("input_receipt"), "input_receipt")
    manifest = _mapping(receipt.get("manifest"), "input_receipt.manifest")
    if manifest.get("sha256") != pin.source_manifest_sha256:
        raise FixtureIdentityError("source manifest hash does not match trusted pin")
    if manifest.get("required_sha256") != pin.source_manifest_sha256:
        raise FixtureIdentityError("required source manifest hash does not match trusted pin")
    sources = _sequence(receipt.get("sources"), "input_receipt.sources")
    observed_sources: dict[str, Mapping[str, Any]] = {}
    for item in sources:
        source = _mapping(item, "input_receipt.sources[]")
        source_id = _string(source.get("id"), "source.id")
        assert source_id is not None
        if source_id in observed_sources:
            raise FixtureSchemaError(f"duplicate source identity: {source_id}")
        observed_sources[source_id] = source
    if set(observed_sources) != {source.source_id for source in pin.source_artifacts}:
        raise FixtureIdentityError("source identity set does not match trusted pin")
    for expected in pin.source_artifacts:
        observed = observed_sources[expected.source_id]
        fields = {
            "filename": expected.filename,
            "bytes": expected.byte_length,
            "sha256": expected.sha256,
            "generation": expected.generation,
            "url": expected.url,
        }
        for field, expected_value in fields.items():
            if observed.get(field) != expected_value:
                raise FixtureIdentityError(
                    f"source {expected.source_id!r} field {field!r} does not match pin"
                )


def load_malecns_fixture(
    path: str | Path,
    *,
    expected_pin: FixturePin = MALECNS_FIXTURE_V1_PIN,
    expected_file_bytes: int = MALECNS_FIXTURE_V1_FILE_BYTES,
) -> RuntimeGraph:
    """Load only bytes that match the independently pinned MaleCNS fixture.

    ``expected_file_bytes`` is the pinned exact length; it is enforced on the
    open descriptor before the file is read, so an oversized or truncated
    candidate never reaches memory. A caller that supplies its own
    ``expected_pin`` must supply the matching length as well.
    """

    fixture_path = Path(path)
    raw = _read_exact_regular_file(fixture_path, expected_bytes=expected_file_bytes)
    observed_file_hash = sha256(raw).hexdigest()
    if observed_file_hash != expected_pin.file_sha256:
        raise FixtureIdentityError(
            "fixture file SHA-256 mismatch: "
            f"expected {expected_pin.file_sha256}, observed {observed_file_hash}"
        )
    try:
        document = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FixtureSchemaError(f"fixture is not strict UTF-8 JSON: {exc}") from exc
    document = _mapping(document, "fixture")
    _verify_document_identity(document, expected_pin)
    return _build_malecns_runtime_graph(
        document,
        fixture_file_sha256=observed_file_hash,
        pin=expected_pin,
    )


def _build_malecns_runtime_graph(
    document: Mapping[str, Any], *, fixture_file_sha256: str, pin: FixturePin
) -> RuntimeGraph:
    nodes_raw = _sequence(document.get("nodes"), "nodes")
    edges_raw = _sequence(document.get("edges"), "edges")
    nodes: list[RuntimeNode] = []
    seen_body_ids: set[int] = set()
    for raw_node in nodes_raw:
        node = _mapping(raw_node, "nodes[]")
        body_id = _integer(node.get("body_id"), "node.body_id", minimum=1)
        assert body_id is not None
        if body_id in seen_body_ids:
            raise FixtureSchemaError(f"duplicate node body ID: {body_id}")
        seen_body_ids.add(body_id)
        roles_raw = _sequence(node.get("roles"), f"node[{body_id}].roles")
        role_values: set[str] = set()
        for item in roles_raw:
            role = _string(item, f"node[{body_id}].roles[]")
            assert role is not None
            role_values.add(role)
        roles = tuple(sorted(role_values))
        if not roles:
            raise FixtureSchemaError(f"node {body_id} has no declared role")
        annotation = _mapping(node.get("annotation"), f"node[{body_id}].annotation")
        annotation_record = _mapping(annotation.get("record"), f"node[{body_id}].annotation.record")
        if annotation_record.get("bodyId") != body_id:
            raise FixtureSchemaError(f"node {body_id} annotation bodyId mismatch")
        annotation_row = _integer(
            annotation.get("source_row"), f"node[{body_id}].annotation.source_row"
        )
        assert annotation_row is not None
        transmitter = _mapping(node.get("transmitter"), f"node[{body_id}].transmitter")
        transmitter_record = _mapping(
            transmitter.get("record"), f"node[{body_id}].transmitter.record"
        )
        if transmitter_record.get("body") != body_id:
            raise FixtureSchemaError(f"node {body_id} transmitter body mismatch")
        transmitter_row = _integer(
            transmitter.get("source_row"),
            f"node[{body_id}].transmitter.source_row",
            allow_none=True,
        )
        sign_status = _string(transmitter.get("sign_status"), f"node[{body_id}].sign_status")
        if sign_status != "unknown":
            raise FixtureSchemaError(f"node {body_id} canonical sign status must remain unknown")
        side = _string(node.get("side"), f"node[{body_id}].side", allow_none=True)
        if side not in {None, "L", "R"}:
            raise FixtureSchemaError(f"node {body_id} has unsupported side {side!r}")
        nodes.append(
            RuntimeNode(
                dense_index=-1,
                body_id=body_id,
                roles=roles,
                side=side,
                type_name=_string(
                    annotation_record.get("type"),
                    f"node[{body_id}].annotation.type",
                    allow_none=True,
                ),
                instance=_string(
                    annotation_record.get("instance"),
                    f"node[{body_id}].annotation.instance",
                    allow_none=True,
                ),
                superclass=_string(
                    annotation_record.get("superclass"),
                    f"node[{body_id}].annotation.superclass",
                    allow_none=True,
                ),
                annotation_source_row=annotation_row,
                transmitter_source_row=transmitter_row,
                predicted_nt=_string(
                    transmitter_record.get("predicted_nt"),
                    f"node[{body_id}].transmitter.predicted_nt",
                    allow_none=True,
                ),
                consensus_nt=_string(
                    transmitter_record.get("consensus_nt"),
                    f"node[{body_id}].transmitter.consensus_nt",
                    allow_none=True,
                ),
                ground_truth_nt=_string(
                    transmitter_record.get("ground_truth"),
                    f"node[{body_id}].transmitter.ground_truth",
                    allow_none=True,
                ),
                canonical_sign_status=sign_status,
            )
        )
    nodes.sort(key=lambda item: item.body_id)
    nodes = [replace(node, dense_index=index) for index, node in enumerate(nodes)]
    body_to_index = {node.body_id: node.dense_index for node in nodes}

    edges_intermediate: list[tuple[int, int, int, int, str]] = []
    pairs: set[tuple[int, int]] = set()
    source_rows: set[int] = set()
    for raw_edge in edges_raw:
        edge = _mapping(raw_edge, "edges[]")
        body_pre = _integer(edge.get("body_pre"), "edge.body_pre", minimum=1)
        body_post = _integer(edge.get("body_post"), "edge.body_post", minimum=1)
        raw_count = _integer(edge.get("weight"), "edge.weight", minimum=1)
        source_row = _integer(edge.get("source_row"), "edge.source_row")
        assert body_pre is not None
        assert body_post is not None
        assert raw_count is not None
        assert source_row is not None
        if body_pre not in body_to_index or body_post not in body_to_index:
            raise FixtureSchemaError(f"edge {body_pre}->{body_post} references an absent node")
        pair = (body_pre, body_post)
        if pair in pairs:
            raise FixtureSchemaError(f"duplicate directed pair {body_pre}->{body_post}")
        if source_row in source_rows:
            raise FixtureSchemaError(f"duplicate edge source row {source_row}")
        pairs.add(pair)
        source_rows.add(source_row)
        sign_status = _string(edge.get("sign_status"), "edge.sign_status")
        if sign_status != "unknown":
            raise FixtureSchemaError(
                f"edge {body_pre}->{body_post} canonical sign must remain unknown"
            )
        edges_intermediate.append((body_pre, body_post, raw_count, source_row, sign_status))
    edges_intermediate.sort(key=lambda item: (item[1], item[0], item[3]))
    edges = tuple(
        RuntimeEdge(
            dense_index=index,
            pre_index=body_to_index[body_pre],
            post_index=body_to_index[body_post],
            body_pre=body_pre,
            body_post=body_post,
            raw_count=raw_count,
            source_row=source_row,
            canonical_sign_status=sign_status,
        )
        for index, (body_pre, body_post, raw_count, source_row, sign_status) in enumerate(
            edges_intermediate
        )
    )
    counts = _mapping(document.get("counts"), "counts")
    expected_counts = {
        "emitted_node_rows": len(nodes),
        "emitted_edge_rows": len(edges),
        "emitted_weight_sum": sum(edge.raw_count for edge in edges),
    }
    for field, observed in expected_counts.items():
        if counts.get(field) != observed:
            raise FixtureSchemaError(
                f"counts.{field} does not reconcile: {counts.get(field)!r} != {observed}"
            )
    if counts.get("duplicate_directed_pairs") != 0:
        raise FixtureSchemaError("fixture declares duplicate directed pairs")
    if not any("sensory" in node.roles for node in nodes):
        raise FixtureSchemaError("fixture has no named sensory boundary")
    if not any("motor" in node.roles for node in nodes):
        raise FixtureSchemaError("fixture has no named motor boundary")

    extraction = _mapping(document.get("extraction_receipt"), "extraction_receipt")
    extraction_code_sha256 = _string(
        extraction.get("code_sha256"), "extraction_receipt.code_sha256"
    )
    extraction_configuration_sha256 = _string(
        extraction.get("configuration_sha256"),
        "extraction_receipt.configuration_sha256",
    )
    assert extraction_code_sha256 is not None
    assert extraction_configuration_sha256 is not None
    return _finish_graph(
        nodes=tuple(nodes),
        edges=edges,
        dataset_id=pin.dataset_id,
        fixture_id=pin.fixture_id,
        fixture_tier=FixtureTier.MALECNS_FIXTURE,
        fixture_file_sha256=fixture_file_sha256,
        fixture_payload_sha256=pin.payload_sha256,
        fixture_semantic_sha256=pin.semantic_sha256,
        source_manifest_sha256=pin.source_manifest_sha256,
        extraction_code_sha256=extraction_code_sha256,
        extraction_configuration_sha256=extraction_configuration_sha256,
        source_artifacts=pin.source_artifacts,
    )


def build_synthetic_graph(
    synthetic_id: str,
    nodes: Sequence[SyntheticNode],
    edges: Sequence[SyntheticEdge],
) -> RuntimeGraph:
    """Build an explicitly synthetic canonical graph for algebra tests.

    This API deliberately refuses MaleCNS-like identities, preventing synthetic
    inputs from inheriting a biological evidence label.
    """

    if not isinstance(synthetic_id, str) or not synthetic_id.startswith("synthetic:"):
        raise FixtureSchemaError("synthetic graph identity must start with 'synthetic:'")
    if not nodes:
        raise FixtureSchemaError("synthetic graph must contain at least one node")
    runtime_nodes: list[RuntimeNode] = []
    seen: set[int] = set()
    for node in nodes:
        body_id = _integer(node.body_id, "synthetic node body_id", minimum=1)
        assert body_id is not None
        if body_id in seen:
            raise FixtureSchemaError(f"duplicate synthetic body ID: {body_id}")
        seen.add(body_id)
        if not node.roles or any(not isinstance(role, str) or not role for role in node.roles):
            raise FixtureSchemaError(f"synthetic node {body_id} must have named roles")
        runtime_nodes.append(
            RuntimeNode(
                dense_index=-1,
                body_id=body_id,
                roles=tuple(sorted(set(node.roles))),
                side=node.side,
                type_name=node.type_name,
                instance=None,
                superclass=node.superclass,
                annotation_source_row=body_id,
                transmitter_source_row=None,
                predicted_nt=node.predicted_nt,
                consensus_nt=node.consensus_nt,
                ground_truth_nt=None,
                canonical_sign_status="unknown",
            )
        )
    runtime_nodes.sort(key=lambda item: item.body_id)
    runtime_nodes = [replace(node, dense_index=index) for index, node in enumerate(runtime_nodes)]
    body_to_index = {node.body_id: node.dense_index for node in runtime_nodes}
    pairs: set[tuple[int, int]] = set()
    rows: set[int] = set()
    edge_values: list[SyntheticEdge] = []
    for edge in edges:
        pre = _integer(edge.body_pre, "synthetic edge body_pre", minimum=1)
        post = _integer(edge.body_post, "synthetic edge body_post", minimum=1)
        count = _integer(edge.raw_count, "synthetic edge raw_count", minimum=1)
        row = _integer(edge.source_row, "synthetic edge source_row")
        assert pre is not None
        assert post is not None
        assert count is not None
        assert row is not None
        if pre not in body_to_index or post not in body_to_index:
            raise FixtureSchemaError(f"synthetic edge {pre}->{post} has absent endpoint")
        if (pre, post) in pairs:
            raise FixtureSchemaError(f"duplicate directed pair {pre}->{post}")
        if row in rows:
            raise FixtureSchemaError(f"duplicate synthetic source row {row}")
        pairs.add((pre, post))
        rows.add(row)
        edge_values.append(edge)
    edge_values.sort(key=lambda item: (item.body_post, item.body_pre, item.source_row))
    runtime_edges = tuple(
        RuntimeEdge(
            dense_index=index,
            pre_index=body_to_index[edge.body_pre],
            post_index=body_to_index[edge.body_post],
            body_pre=edge.body_pre,
            body_post=edge.body_post,
            raw_count=edge.raw_count,
            source_row=edge.source_row,
            canonical_sign_status="unknown",
        )
        for index, edge in enumerate(edge_values)
    )
    synthetic_payload = {
        "edges": [
            {
                "body_post": edge.body_post,
                "body_pre": edge.body_pre,
                "raw_count": edge.raw_count,
                "source_row": edge.source_row,
            }
            for edge in runtime_edges
        ],
        "fixture_tier": FixtureTier.SYNTHETIC_ALGEBRA,
        "nodes": [
            {
                "body_id": node.body_id,
                "roles": list(node.roles),
                "side": node.side,
                "type_name": node.type_name,
            }
            for node in runtime_nodes
        ],
        "synthetic_id": synthetic_id,
    }
    content_hash = sha256_json(synthetic_payload)
    return _finish_graph(
        nodes=tuple(runtime_nodes),
        edges=runtime_edges,
        dataset_id=synthetic_id,
        fixture_id=synthetic_id,
        fixture_tier=FixtureTier.SYNTHETIC_ALGEBRA,
        fixture_file_sha256=content_hash,
        fixture_payload_sha256=content_hash,
        fixture_semantic_sha256=content_hash,
        source_manifest_sha256=content_hash,
        extraction_code_sha256="synthetic-algebra-builder-v1",
        extraction_configuration_sha256=sha256_json(LOADER_CONFIGURATION),
        source_artifacts=(),
    )


def _finish_graph(
    *,
    nodes: tuple[RuntimeNode, ...],
    edges: tuple[RuntimeEdge, ...],
    dataset_id: str,
    fixture_id: str,
    fixture_tier: FixtureTier,
    fixture_file_sha256: str,
    fixture_payload_sha256: str,
    fixture_semantic_sha256: str,
    source_manifest_sha256: str,
    extraction_code_sha256: str,
    extraction_configuration_sha256: str,
    source_artifacts: tuple[SourceArtifact, ...],
) -> RuntimeGraph:
    hashes = graph_hashes(nodes, edges)
    values = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "fixture_id": fixture_id,
        "fixture_tier": fixture_tier,
        "fixture_file_sha256": fixture_file_sha256,
        "fixture_payload_sha256": fixture_payload_sha256,
        "fixture_semantic_sha256": fixture_semantic_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "extraction_code_sha256": extraction_code_sha256,
        "extraction_configuration_sha256": extraction_configuration_sha256,
        "loader_name": LOADER_NAME,
        "loader_version": LOADER_VERSION,
        "loader_implementation_sha256": _loader_implementation_sha256(),
        "loader_configuration_sha256": sha256_json(LOADER_CONFIGURATION),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "raw_count_sum": sum(edge.raw_count for edge in edges),
        **hashes,
        "source_artifacts": source_artifacts,
    }
    graph = RuntimeGraph(
        manifest=manifest_identity_material(values),
        nodes=nodes,
        edges=edges,
    )
    graph.assert_integrity()
    return graph
