"""The flight-steering fixture must load through the unmodified loader.

``mcns-v1.0-flight-steering-v1`` is a second tracked MaleCNS-derived circuit. It
is verified by passing its own pin to ``load_malecns_fixture``'s existing
``expected_pin``/``expected_file_bytes`` parameters, never by editing
``loader.py`` or ``model.py`` — their bytes are hashed into every fixture graph
identity, so a one-byte change there would rotate every downstream hash in the
repository.

These tests also re-assert that the flight-power fixture keeps its own identity
and remains the sole authority for the full-artifact boundary overlay.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
from neurocraft_connectome import (
    BOUNDARY_ROLE_BODY_IDS,
    MALECNS_FIXTURE_V1_PIN,
    FixtureIdentityError,
    FixtureSchemaError,
    load_malecns_fixture,
)
from neurocraft_connectome.flight_pin import (
    MALECNS_FLIGHT_FIXTURE_V1_EDGE_COUNT,
    MALECNS_FLIGHT_FIXTURE_V1_FILE_BYTES,
    MALECNS_FLIGHT_FIXTURE_V1_NODE_COUNT,
    MALECNS_FLIGHT_FIXTURE_V1_PATH,
    MALECNS_FLIGHT_FIXTURE_V1_PIN,
)
from neurocraft_connectome.model import RuntimeGraph, canonical_json_bytes

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / MALECNS_FLIGHT_FIXTURE_V1_PATH
POWER_FIXTURE_PATH = REPO_ROOT / "data/fixtures/mcns-v1.0-flight-power-v1.json"

#: The two pinned sensory entries, shared with the flight-power fixture.
SENSORY_BODY_IDS = (77185, 805722)
#: Exact descending types the extractor resolves; counts are source-observed.
DESCENDING_TYPE_COUNTS = {
    "DNg02_a": 10,
    "DNg02_b": 5,
    "DNg02_c": 4,
    "DNg02_d": 2,
    "DNg02_e": 2,
    "DNg02_f": 2,
    "DNg02_g": 4,
    "DNp03": 2,
    "DNp31": 2,
}
#: The declared controller ceiling this circuit had to fit under.
EMITTED_EDGE_CEILING = 2000


def load_flight_fixture(path: Path | None = None) -> RuntimeGraph:
    return load_malecns_fixture(
        FIXTURE_PATH if path is None else path,
        expected_pin=MALECNS_FLIGHT_FIXTURE_V1_PIN,
        expected_file_bytes=MALECNS_FLIGHT_FIXTURE_V1_FILE_BYTES,
    )


def test_pinned_flight_fixture_loads_with_its_own_pin() -> None:
    assert FIXTURE_PATH.stat().st_size == MALECNS_FLIGHT_FIXTURE_V1_FILE_BYTES

    graph = load_flight_fixture()

    assert graph.manifest.fixture_id == "mcns-v1.0-flight-steering-v1"
    assert graph.manifest.dataset_id == "male-cns:v1.0"
    assert graph.manifest.node_count == MALECNS_FLIGHT_FIXTURE_V1_NODE_COUNT == 128
    assert graph.manifest.edge_count == MALECNS_FLIGHT_FIXTURE_V1_EDGE_COUNT == 887
    assert graph.manifest.raw_count_sum == 26_729
    assert graph.manifest.fixture_file_sha256 == MALECNS_FLIGHT_FIXTURE_V1_PIN.file_sha256
    assert graph.manifest.source_manifest_sha256 == (
        "9412105f54884a57521bbdf7fe46659caab559ef7877e6a62af95f50e1ee3768"
    )
    assert len(graph.body_ids) == len(set(graph.body_ids))
    assert list(graph.body_ids) == sorted(graph.body_ids)
    assert tuple(edge.dense_index for edge in graph.edges) == tuple(range(887))
    graph.assert_integrity()


def test_emitted_closure_stays_under_the_declared_controller_ceiling() -> None:
    graph = load_flight_fixture()

    assert graph.manifest.edge_count < EMITTED_EDGE_CEILING
    assert all(edge.raw_count >= 5 for edge in graph.edges)
    assert len({edge.source_row for edge in graph.edges}) == graph.manifest.edge_count
    assert len({(edge.body_pre, edge.body_post) for edge in graph.edges}) == 887


def test_every_runtime_element_keeps_exact_source_lineage() -> None:
    graph = load_flight_fixture()

    assert all(node.annotation_source_row >= 0 for node in graph.nodes)
    assert all(node.transmitter_source_row is not None for node in graph.nodes)
    assert len({node.annotation_source_row for node in graph.nodes}) == 128
    assert all(edge.canonical_sign_status == "unknown" for edge in graph.edges)
    assert all(node.canonical_sign_status == "unknown" for node in graph.nodes)
    assert {source.source_id for source in graph.manifest.source_artifacts} == {
        "annotations",
        "neurotransmitters",
        "weights",
    }


def test_sensory_boundary_bodies_carry_the_expected_superclasses() -> None:
    graph = load_flight_fixture()

    assert graph.role_body_ids("sensory") == SENSORY_BODY_IDS

    jo_ev3 = graph.node_for(77185)
    assert jo_ev3.type_name == "JO-EV3"
    assert jo_ev3.superclass == "cb_sensory"
    assert jo_ev3.side == "R"

    campaniform = graph.node_for(805722)
    assert campaniform.type_name == "SNpp10"
    assert campaniform.superclass == "vnc_sensory"
    assert campaniform.side == "L"


def test_wing_motor_readout_bodies_carry_the_expected_superclass() -> None:
    graph = load_flight_fixture()

    readout = graph.role_body_ids("motor")
    assert len(readout) == 52
    assert readout == tuple(sorted(readout))
    assert all(graph.node_for(body_id).superclass == "vnc_motor" for body_id in readout)
    assert 800917 in readout
    assert graph.node_for(800917).type_name == "hg1 MN"

    contacted = {
        edge.body_post
        for edge in graph.edges
        if edge.body_pre in set(graph.role_body_ids("descending"))
    }
    assert set(readout) <= contacted


def test_descending_population_is_the_exact_declared_type_set() -> None:
    graph = load_flight_fixture()

    descending = graph.role_body_ids("descending")
    assert len(descending) == 33
    observed = Counter(graph.node_for(body_id).type_name for body_id in descending)
    assert dict(observed) == DESCENDING_TYPE_COUNTS
    assert all(graph.node_for(body_id).superclass == "descending_neuron" for body_id in descending)
    sides = Counter(graph.node_for(body_id).side for body_id in descending)
    assert sides["L"] == 17
    assert sides["R"] == 16


def test_roles_admit_the_intended_adapter_targets() -> None:
    """Sensor channels require role ``sensory``; motor pools require motor/descending."""

    graph = load_flight_fixture()

    for body_id in graph.role_body_ids("sensory"):
        assert "sensory" in graph.node_for(body_id).roles

    motor_pool_eligible = [
        node
        for node in graph.nodes
        if "motor" in node.roles or node.superclass == "descending_neuron"
    ]
    assert len(motor_pool_eligible) == 88
    assert {node.body_id for node in motor_pool_eligible} >= set(graph.role_body_ids("motor"))
    assert {node.body_id for node in motor_pool_eligible} >= set(graph.role_body_ids("descending"))
    assert all(node.roles for node in graph.nodes)
    assert set(graph.role_body_ids("sensory")).isdisjoint(graph.role_body_ids("motor"))


def test_composition_and_bilateral_balance_are_source_observed() -> None:
    graph = load_flight_fixture()

    composition = Counter(node.superclass for node in graph.nodes)
    assert dict(sorted(composition.items())) == {
        "ascending_neuron": 7,
        "cb_intrinsic": 27,
        "cb_sensory": 1,
        "descending_neuron": 36,
        "visual_projection": 1,
        "vnc_intrinsic": 3,
        "vnc_motor": 52,
        "vnc_sensory": 1,
    }
    sides = Counter(node.side for node in graph.nodes)
    assert sides["L"] == 57
    assert sides["R"] == 71
    assert sides[None] == 0


def test_oversize_and_truncated_candidates_are_refused_before_read(tmp_path: Path) -> None:
    raw = FIXTURE_PATH.read_bytes()

    padded = tmp_path / "padded.json"
    padded.write_bytes(raw + b"\x00" * 4096)
    with pytest.raises(FixtureIdentityError, match="file size mismatch") as oversize:
        load_flight_fixture(padded)
    assert "SHA-256" not in str(oversize.value)

    truncated = tmp_path / "truncated.json"
    truncated.write_bytes(raw[:-1])
    with pytest.raises(FixtureIdentityError, match="file size mismatch"):
        load_flight_fixture(truncated)

    with pytest.raises(FixtureIdentityError, match="must be a regular file"):
        load_flight_fixture(Path("/dev/null"))


def test_file_mutation_is_refused_before_use(tmp_path: Path) -> None:
    raw = FIXTURE_PATH.read_bytes()
    mutated = raw.replace(b'"weight":23', b'"weight":24', 1)
    assert mutated != raw
    path = tmp_path / "mutated.json"
    path.write_bytes(mutated)

    with pytest.raises(FixtureIdentityError, match="file SHA-256 mismatch"):
        load_flight_fixture(path)


def test_embedded_payload_mutation_is_refused_even_if_the_file_pin_is_replaced(
    tmp_path: Path,
) -> None:
    raw = FIXTURE_PATH.read_bytes()
    mutated = raw.replace(b'"weight":23', b'"weight":24', 1)
    path = tmp_path / "mutated.json"
    path.write_bytes(mutated)
    permissive = replace(MALECNS_FLIGHT_FIXTURE_V1_PIN, file_sha256=sha256(mutated).hexdigest())

    with pytest.raises(FixtureIdentityError, match="embedded identity mismatch"):
        load_malecns_fixture(
            path,
            expected_pin=permissive,
            expected_file_bytes=len(mutated),
        )


def test_source_pin_is_not_self_authorized_by_the_fixture(tmp_path: Path) -> None:
    document = json.loads(FIXTURE_PATH.read_text())
    document["input_receipt"]["sources"][0]["sha256"] = "f" * 64
    payload = {key: value for key, value in document.items() if key != "hashes"}
    payload_sha = sha256(canonical_json_bytes(payload)).hexdigest()
    document["hashes"]["payload_sha256"] = payload_sha
    raw = canonical_json_bytes(document) + b"\n"
    path = tmp_path / "self-authorized.json"
    path.write_bytes(raw)
    pin = replace(
        MALECNS_FLIGHT_FIXTURE_V1_PIN,
        file_sha256=sha256(raw).hexdigest(),
        payload_sha256=payload_sha,
    )

    with pytest.raises(FixtureIdentityError, match=r"source 'annotations'.*does not match pin"):
        load_malecns_fixture(path, expected_pin=pin, expected_file_bytes=len(raw))


def test_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    raw = FIXTURE_PATH.read_bytes().replace(
        b'{"circuit_resolution":', b'{"schema_version":1,"circuit_resolution":', 1
    )
    path = tmp_path / "duplicate-key.json"
    path.write_bytes(raw)
    pin = replace(MALECNS_FLIGHT_FIXTURE_V1_PIN, file_sha256=sha256(raw).hexdigest())

    with pytest.raises(FixtureSchemaError, match="duplicate JSON key"):
        load_malecns_fixture(path, expected_pin=pin, expected_file_bytes=len(raw))


def test_the_two_fixtures_never_authorize_each_other() -> None:
    with pytest.raises(FixtureIdentityError):
        load_malecns_fixture(FIXTURE_PATH)

    with pytest.raises(FixtureIdentityError):
        load_malecns_fixture(
            POWER_FIXTURE_PATH,
            expected_pin=MALECNS_FLIGHT_FIXTURE_V1_PIN,
            expected_file_bytes=MALECNS_FLIGHT_FIXTURE_V1_FILE_BYTES,
        )

    assert MALECNS_FLIGHT_FIXTURE_V1_PIN.fixture_id != MALECNS_FIXTURE_V1_PIN.fixture_id
    assert MALECNS_FLIGHT_FIXTURE_V1_PIN.file_sha256 != MALECNS_FIXTURE_V1_PIN.file_sha256
    assert MALECNS_FLIGHT_FIXTURE_V1_PIN.source_artifacts == MALECNS_FIXTURE_V1_PIN.source_artifacts


def test_flight_power_fixture_remains_the_boundary_overlay_authority() -> None:
    """Adding this circuit must not move the G1 full-artifact boundary contract."""

    power = load_malecns_fixture(POWER_FIXTURE_PATH)

    overlay = dict(BOUNDARY_ROLE_BODY_IDS)
    assert overlay == {
        "motor": power.role_body_ids("motor"),
        "sensory": power.role_body_ids("sensory"),
    }
    assert overlay["motor"] == (800241, 800917)
    assert overlay["sensory"] == SENSORY_BODY_IDS
