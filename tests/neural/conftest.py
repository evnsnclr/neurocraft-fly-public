from __future__ import annotations

from pathlib import Path

import pytest
from neurocraft_connectome import (
    RuntimeGraph,
    SyntheticEdge,
    SyntheticNode,
    build_synthetic_graph,
    load_malecns_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def male_fixture_graph() -> RuntimeGraph:
    return load_malecns_fixture(REPO_ROOT / "data/fixtures/mcns-v1.0-flight-power-v1.json")


@pytest.fixture
def two_node_graph() -> RuntimeGraph:
    return build_synthetic_graph(
        "synthetic:two-node",
        [
            SyntheticNode(1, ("sensory",), type_name="sensor"),
            SyntheticNode(2, ("motor",), type_name="motor"),
        ],
        [SyntheticEdge(1, 2, 5, 0)],
    )
