"""AB-11 test power against graph-hash substitution (M6), in its three layers.

Layer 1 substitutes fixture bytes, layer 2 substitutes the runtime graph object, and
layer 3 substitutes graph provenance inside an otherwise hash-authentic golden trace.
Every layer must be refused by a named production detector, and the faithful artifact
must pass the same detector in the same test.

The substitution builders live in ``tests/ab11/mutants.py`` and are falsification
artifacts only, never Faithful Organism Mode components.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
from neurocraft_connectome import (
    MALECNS_FIXTURE_V1_PIN,
    FixtureIdentityError,
    RuntimeGraph,
    SyntheticEdge,
    SyntheticNode,
    build_synthetic_graph,
    load_malecns_fixture,
)
from neurocraft_controller import ConnectomeController, ControllerIntegrityError
from neurocraft_protocol import RunMode, verify_trace

from scripts.replay_minecraft_trace import replay_minecraft_trace
from tests.controller.conftest import make_observation

from .conftest import (
    FIXTURE,
    TRACE_RECORD_COUNT,
    faithful_trace_controller,
    minecraft_labelled_trace,
)
from .mutants import (
    MUTANT_ENTITY_ID,
    substituted_canonical_graph_trace,
    substituted_graph_trace,
)

SUBSTITUTED_RUNTIME_GRAPH_HASH = "a" * 64
SUBSTITUTED_CANONICAL_GRAPH_HASH = "b" * 64


def test_ab11_graph_byte_substitution_is_refused_by_the_fixture_loader(tmp_path: Path) -> None:
    """AB-11(M6 graph-hash substitution / byte layer) vs the pinned fixture loader.

    Named tests: ``tests/connectome/test_fixture_loader.py::
    test_file_mutation_is_refused_before_use`` and
    ``test_embedded_payload_mutation_is_refused_even_if_file_pin_is_replaced``.
    Detector: ``load_malecns_fixture`` raising ``FixtureIdentityError``.
    """

    assert load_malecns_fixture(FIXTURE).manifest.fixture_file_sha256 == (
        MALECNS_FIXTURE_V1_PIN.file_sha256
    )

    substituted = tmp_path / "substituted-bytes.json"
    substituted.write_bytes(FIXTURE.read_bytes().replace(b'"weight":25', b'"weight":26', 1))
    with pytest.raises(FixtureIdentityError, match="file SHA-256 mismatch"):
        load_malecns_fixture(substituted)

    permissive_pin = replace(
        MALECNS_FIXTURE_V1_PIN,
        file_sha256=sha256(substituted.read_bytes()).hexdigest(),
    )
    with pytest.raises(FixtureIdentityError, match="embedded identity mismatch"):
        load_malecns_fixture(substituted, expected_pin=permissive_pin)


def test_ab11_synthetic_graph_substitution_is_refused_by_the_controller(
    controller_graph: RuntimeGraph,
) -> None:
    """AB-11(M6 graph-hash substitution / runtime layer) vs the controller graph binding.

    Named test: ``tests/controller/test_causal_controller.py::
    test_synthetic_graph_cannot_silently_replace_faithful_graph``.
    Detectors: ``ControllerIntegrityError("synthetic graph cannot replace a connectome
    controller")`` at construction, and ``ControllerIntegrityError("bound graph identity
    changed")`` on every subsequent step for a post-construction swap.
    """

    faithful = ConnectomeController.create(
        controller_graph,
        run_id="ab11-m6-baseline",
        entity_id=MUTANT_ENTITY_ID,
    )
    assert faithful.run_mode is RunMode.FAITHFUL
    assert faithful.identity.graph_hash == controller_graph.manifest.runtime_identity_sha256

    synthetic = build_synthetic_graph(
        "synthetic:ab11-graph-substitution",
        [SyntheticNode(1, ("sensory",)), SyntheticNode(2, ("motor",))],
        [SyntheticEdge(1, 2, 1, 0)],
    )
    with pytest.raises(ControllerIntegrityError, match="cannot replace"):
        ConnectomeController.create(
            synthetic,
            run_id="ab11-m6-mutant",
            entity_id=MUTANT_ENTITY_ID,
        )

    swapped = ConnectomeController.create(
        controller_graph,
        run_id="ab11-m6-swap",
        entity_id=MUTANT_ENTITY_ID,
    )
    observation = make_observation(swapped, 0)
    swapped.graph = synthetic
    with pytest.raises(ControllerIntegrityError, match="bound graph identity changed"):
        swapped.step(observation, command_seq=0)


def test_ab11_graph_hash_substitution_trips_replay_environment_binding(
    controller_graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    """AB-11(M6 graph-hash substitution / trace layer) vs the replay environment binding.

    Named test: ``tests/integration/test_replay_minecraft_trace.py::
    test_replay_rejects_hash_authentic_tampered_cross_link`` (the same hash-authentic
    rewrite pattern applied to graph provenance instead of world lineage).
    Detectors: ``scripts/replay_minecraft_trace.py`` raising "current controller
    identities do not match the trace" for a fully consistent runtime-graph-hash
    substitution, and "trace canonical_graph_hash does not match the current replay
    environment" for the canonical-provenance substitution.
    """

    source = tmp_path / "m6-faithful.ndjson"
    minecraft_labelled_trace(
        source,
        faithful_trace_controller(controller_graph, run_id="ab11-m6-trace"),
    )
    assert (
        replay_minecraft_trace(fixture=FIXTURE, trace=source, stimulations={})["status"]
        == "exact_neural_match"
    )

    runtime_substituted = tmp_path / "m6-runtime-graph.ndjson"
    substituted_graph_trace(
        source,
        runtime_substituted,
        graph_hash=SUBSTITUTED_RUNTIME_GRAPH_HASH,
    )
    assert verify_trace(runtime_substituted).record_count == TRACE_RECORD_COUNT
    with pytest.raises(ValueError, match="current controller identities do not match the trace"):
        replay_minecraft_trace(fixture=FIXTURE, trace=runtime_substituted, stimulations={})

    canonical_substituted = tmp_path / "m6-canonical-graph.ndjson"
    substituted_canonical_graph_trace(
        source,
        canonical_substituted,
        canonical_graph_hash=SUBSTITUTED_CANONICAL_GRAPH_HASH,
    )
    assert verify_trace(canonical_substituted).record_count == TRACE_RECORD_COUNT
    with pytest.raises(
        ValueError,
        match="trace canonical_graph_hash does not match the current replay environment",
    ):
        replay_minecraft_trace(fixture=FIXTURE, trace=canonical_substituted, stimulations={})
