#!/usr/bin/env python3
"""Build the brain-viewer's committed layout asset for the steering fixture.

Presentation-only: this layout feeds `apps/brain-viewer`, which renders live
neural activity beside the game. Nothing here is evidence; the asset carries
the fixture's real topology (every node and edge, signs from the executed
coarse-consensus overlay) laid out as a causal left-to-right flow —
sensory entries → interneurons → descending neurons → wing motor pools —
with deterministic, seeded positions so the asset regenerates byte-identically.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
from typing import Any

from neurocraft_connectome import RuntimeGraph, load_malecns_fixture
from neurocraft_connectome.flight_pin import MALECNS_FLIGHT_FIXTURE_V1_PIN
from neurocraft_connectome.model import canonical_json_bytes
from neurocraft_controller.flight_pools import (
    LEFT_WING_MOTOR_BODY_IDS,
    RIGHT_WING_MOTOR_BODY_IDS,
)
from neurocraft_neural.model import SignOverlay

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FLIGHT_FIXTURE = REPOSITORY_ROOT / "data" / "fixtures" / "mcns-v1.0-flight-steering-v1.json"
FLIGHT_FIXTURE_BYTES = 284_949
DEFAULT_OUTPUT = REPOSITORY_ROOT / "apps" / "brain-viewer" / "layout-flight-steering-v1.json"

LAYOUT_SCHEMA = "neurocraft.presentation.brain_layout.v1"
SENSORY_BODY_IDS = {77_185, 805_722}

#: Column x-positions for the causal flow, in a 0..1 viewport.
GROUP_COLUMNS = {
    "sensory": 0.06,
    "interneuron": 0.40,
    "descending": 0.70,
    "motor_left": 0.93,
    "motor_right": 0.93,
}


def _jitter(seed_material: str, span: float) -> float:
    digest = sha256(seed_material.encode()).digest()
    unit = int.from_bytes(digest[:8], "big") / 2**64
    return (unit - 0.5) * span


def node_group(body_id: int, superclass: str | None) -> str:
    if body_id in SENSORY_BODY_IDS:
        return "sensory"
    if body_id in set(LEFT_WING_MOTOR_BODY_IDS):
        return "motor_left"
    if body_id in set(RIGHT_WING_MOTOR_BODY_IDS):
        return "motor_right"
    if superclass == "descending_neuron":
        return "descending"
    return "interneuron"


def build_layout(graph: RuntimeGraph) -> dict[str, Any]:
    overlay = SignOverlay.coarse_presynaptic_consensus(graph)
    groups: dict[str, list[int]] = {name: [] for name in GROUP_COLUMNS}
    node_groups: list[str] = []
    for node in graph.nodes:
        group = node_group(node.body_id, node.superclass)
        node_groups.append(group)
        groups[group].append(node.dense_index)

    nodes: list[dict[str, Any]] = []
    for node in graph.nodes:
        group = node_groups[node.dense_index]
        members = groups[group]
        rank = members.index(node.dense_index)
        count = len(members)
        if group == "motor_left":
            y = 0.06 + 0.40 * (rank / max(1, count - 1))
        elif group == "motor_right":
            y = 0.56 + 0.40 * (rank / max(1, count - 1))
        else:
            y = 0.08 + 0.84 * (rank / max(1, count - 1)) if count > 1 else 0.5
        x = GROUP_COLUMNS[group] + _jitter(f"x|{node.body_id}", 0.045)
        y = y + _jitter(f"y|{node.body_id}", 0.02)
        nodes.append(
            {
                "index": node.dense_index,
                "body_id": node.body_id,
                "group": group,
                "type": node.type_name or "",
                "x": round(min(0.98, max(0.02, x)), 4),
                "y": round(min(0.97, max(0.03, y)), 4),
            }
        )

    edges = [
        {
            "pre": edge.pre_index,
            "post": edge.post_index,
            "sign": overlay.values[edge_index],
        }
        for edge_index, edge in enumerate(graph.edges)
    ]
    return {
        "schema": LAYOUT_SCHEMA,
        "presentation_only": True,
        "fixture_id": graph.manifest.fixture_id,
        "graph_runtime_identity_sha256": graph.manifest.runtime_identity_sha256,
        "sign_overlay_identity_sha256": overlay.identity_sha256,
        "group_counts": {name: len(members) for name, members in groups.items()},
        "nodes": nodes,
        "edges": edges,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    graph = load_malecns_fixture(
        FLIGHT_FIXTURE,
        expected_pin=MALECNS_FLIGHT_FIXTURE_V1_PIN,
        expected_file_bytes=FLIGHT_FIXTURE_BYTES,
    )
    document = build_layout(graph)
    payload = canonical_json_bytes(document) + b"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = args.output.read_bytes() if args.output.exists() else None
    if existing is not None and existing != payload:
        raise SystemExit(
            f"{args.output} exists with different content; layouts regenerate "
            "byte-identically — delete deliberately before rebuilding"
        )
    args.output.write_bytes(payload)
    digest = sha256(payload).hexdigest()
    print(
        f"layout nodes={len(document['nodes'])} edges={len(document['edges'])} file_sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
