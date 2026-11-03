from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from pathlib import Path

import pytest
from neurocraft_connectome import (
    BOUNDARY_ROLE_BODY_IDS,
    MALECNS_FIXTURE_V1_PIN,
    FixtureIdentityError,
    FixtureSchemaError,
    GraphIntegrityError,
    load_malecns_fixture,
)
from neurocraft_connectome.loader import MALECNS_FIXTURE_V1_FILE_BYTES
from neurocraft_connectome.model import canonical_json_bytes

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "data/fixtures/mcns-v1.0-flight-power-v1.json"


def test_pinned_fixture_loads_into_stable_target_major_arrays() -> None:
    graph = load_malecns_fixture(FIXTURE_PATH)

    assert graph.body_ids == (
        10026,
        10069,
        10180,
        10236,
        10691,
        27143,
        28949,
        29142,
        73753,
        77185,
        536501,
        800241,
        800917,
        805722,
    )
    assert graph.manifest.node_count == 14
    assert graph.manifest.edge_count == 22
    assert graph.manifest.raw_count_sum == 1516
    assert graph.manifest.fixture_file_sha256 == MALECNS_FIXTURE_V1_PIN.file_sha256
    assert graph.manifest.canonical_graph_sha256 == (
        "7b8dd723eb2a9e5245869dd4459115030aa5a7fb62e70668257029a28ab6d0d1"
    )
    assert graph.manifest.runtime_identity_sha256 == (
        "49001caafff67ab5dfc303b169155136482b7574fd42e93736bdbbfa6f66fc07"
    )
    assert graph.manifest.loader_implementation_sha256 == (
        "1909ff88521e11d9af1857c3486f748217e2d7a045dd7546e9b3326583827368"
    )
    assert [(edge.body_post, edge.body_pre) for edge in graph.edges] == sorted(
        (edge.body_post, edge.body_pre) for edge in graph.edges
    )
    assert tuple(edge.dense_index for edge in graph.edges) == tuple(range(22))
    graph.assert_integrity()


def test_every_runtime_element_has_exact_source_lineage() -> None:
    graph = load_malecns_fixture(FIXTURE_PATH)

    assert all(node.annotation_source_row >= 0 for node in graph.nodes)
    assert all(node.transmitter_source_row is not None for node in graph.nodes)
    assert len({node.annotation_source_row for node in graph.nodes}) == 14
    assert len({edge.source_row for edge in graph.edges}) == 22
    assert all(edge.raw_count > 0 for edge in graph.edges)
    assert all(edge.canonical_sign_status == "unknown" for edge in graph.edges)
    assert graph.role_body_ids("sensory") == (77185, 805722)
    assert graph.role_body_ids("motor") == (800241, 800917)
    assert graph.node_for(27143).type_name == "DNg07"
    assert graph.node_for(28949).type_name == "DNg07"
    assert {source.source_id for source in graph.manifest.source_artifacts} == {
        "annotations",
        "neurotransmitters",
        "weights",
    }
    assert graph.manifest.source_manifest_sha256 == (
        "9412105f54884a57521bbdf7fe46659caab559ef7877e6a62af95f50e1ee3768"
    )


def test_full_runtime_boundary_contract_is_exactly_the_frozen_fixture_roles() -> None:
    graph = load_malecns_fixture(FIXTURE_PATH)

    assert dict(BOUNDARY_ROLE_BODY_IDS) == {
        "motor": graph.role_body_ids("motor"),
        "sensory": graph.role_body_ids("sensory"),
    }


def test_fixture_exact_pinned_size_accepted() -> None:
    assert FIXTURE_PATH.stat().st_size == MALECNS_FIXTURE_V1_FILE_BYTES

    graph = load_malecns_fixture(FIXTURE_PATH)

    assert graph.manifest.fixture_file_sha256 == MALECNS_FIXTURE_V1_PIN.file_sha256


def test_fixture_oversize_refused_before_read(tmp_path: Path) -> None:
    """The size gate must fire on the descriptor, not after the bytes are held."""

    padded = tmp_path / "padded.json"
    padded.write_bytes(FIXTURE_PATH.read_bytes() + b"\x00" * 4096)

    with pytest.raises(FixtureIdentityError, match="file size mismatch") as oversize:
        load_malecns_fixture(padded)
    assert "SHA-256" not in str(oversize.value)

    truncated = tmp_path / "truncated.json"
    truncated.write_bytes(FIXTURE_PATH.read_bytes()[:-1])

    with pytest.raises(FixtureIdentityError, match="file size mismatch"):
        load_malecns_fixture(truncated)

    with pytest.raises(FixtureIdentityError, match="must be a regular file"):
        load_malecns_fixture(Path("/dev/null"))


def test_file_mutation_is_refused_before_use(tmp_path: Path) -> None:
    raw = FIXTURE_PATH.read_bytes()
    mutated = raw.replace(b'"weight":25', b'"weight":26', 1)
    path = tmp_path / "mutated.json"
    path.write_bytes(mutated)

    with pytest.raises(FixtureIdentityError, match="file SHA-256 mismatch"):
        load_malecns_fixture(path)


def test_embedded_payload_mutation_is_refused_even_if_file_pin_is_replaced(
    tmp_path: Path,
) -> None:
    raw = FIXTURE_PATH.read_bytes()
    mutated = raw.replace(b'"weight":25', b'"weight":26', 1)
    path = tmp_path / "mutated.json"
    path.write_bytes(mutated)
    permissive_file_pin = replace(
        MALECNS_FIXTURE_V1_PIN,
        file_sha256=sha256(mutated).hexdigest(),
    )

    with pytest.raises(FixtureIdentityError, match="embedded identity mismatch"):
        load_malecns_fixture(path, expected_pin=permissive_file_pin)


def test_independent_source_pin_is_not_self_authorized_by_fixture(tmp_path: Path) -> None:
    document = json.loads(FIXTURE_PATH.read_text())
    document["input_receipt"]["sources"][0]["sha256"] = "f" * 64
    payload = {key: value for key, value in document.items() if key != "hashes"}
    payload_sha = sha256(canonical_json_bytes(payload)).hexdigest()
    document["hashes"]["payload_sha256"] = payload_sha
    raw = canonical_json_bytes(document) + b"\n"
    path = tmp_path / "self-authorized.json"
    path.write_bytes(raw)
    pin = replace(
        MALECNS_FIXTURE_V1_PIN,
        file_sha256=sha256(raw).hexdigest(),
        payload_sha256=payload_sha,
    )

    with pytest.raises(
        FixtureIdentityError,
        match=r"source 'annotations'.*does not match pin",
    ):
        load_malecns_fixture(path, expected_pin=pin, expected_file_bytes=len(raw))


def test_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    raw = FIXTURE_PATH.read_bytes().replace(
        b'{"candidate_resolution":', b'{"schema_version":1,"candidate_resolution":', 1
    )
    path = tmp_path / "duplicate-key.json"
    path.write_bytes(raw)
    pin = replace(MALECNS_FIXTURE_V1_PIN, file_sha256=sha256(raw).hexdigest())

    with pytest.raises(FixtureSchemaError, match="duplicate JSON key"):
        load_malecns_fixture(path, expected_pin=pin, expected_file_bytes=len(raw))


def test_graph_types_are_immutable_and_integrity_is_recomputed() -> None:
    graph = load_malecns_fixture(FIXTURE_PATH)

    with pytest.raises(FrozenInstanceError):
        graph.nodes[0].body_id = 99  # type: ignore[misc]
    altered = replace(
        graph,
        manifest=replace(graph.manifest, edge_support_sha256="0" * 64),
    )
    with pytest.raises(GraphIntegrityError, match="runtime graph arrays"):
        altered.assert_integrity()

    altered_index = replace(
        graph,
        edges=(
            replace(graph.edges[0], pre_index=graph.edges[0].post_index),
            *graph.edges[1:],
        ),
    )
    with pytest.raises(GraphIntegrityError, match="pre-index/body-ID mismatch"):
        altered_index.assert_integrity()


def test_unknown_body_id_never_uses_fuzzy_substitution() -> None:
    graph = load_malecns_fixture(FIXTURE_PATH)

    with pytest.raises(KeyError, match="not present"):
        graph.index_of(800242)
