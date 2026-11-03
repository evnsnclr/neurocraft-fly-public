from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from neurocraft_connectome import RuntimeGraph, load_malecns_fixture
from neurocraft_controller import ConnectomeController
from neurocraft_embodiment import (
    AdhesionBinding,
    FlyGymMotorDecoder,
    JointBinding,
    ReferenceArena,
    ReferenceEmbodiment,
    replay_embodiment_trace,
    run_embodiment,
)
from neurocraft_neural import MotorPool, PoolMember, SensorChannel, SensorTarget
from neurocraft_protocol import canonical_json_bytes

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "data/fixtures/mcns-v1.0-flight-power-v1.json"
PACKAGE_SRC = REPO_ROOT / "packages/embodiment-runtime/src/neurocraft_embodiment"
#: A second real fixture, so a replay can be handed a graph the trace never used.
OTHER_FIXTURE = REPO_ROOT / "data/fixtures/mcns-v1.0-flight-steering-v1.json"

_ZERO_HASH = "0" * 64

#: Long enough that the world-state lineage has real cross-links to check.
TICKS = 24

# The 14-node CI fixture exposes two sensory nodes and six motor/descending nodes.
# That is the entire eligible boundary population, so wide-boundary tests are sized
# to it rather than to NeuroMechFly's 42 DOFs; a test asserts the real 42-DOF
# binding is rejected against this fixture with an explicit error.
FIXTURE_SENSORY_BODY_IDS = (805_722, 77_185)
FIXTURE_OUTPUT_BODY_IDS = (10_026, 10_236, 27_143, 28_949, 800_241, 800_917)


@pytest.fixture(scope="session")
def graph() -> RuntimeGraph:
    return load_malecns_fixture(FIXTURE)


@pytest.fixture
def controller(graph: RuntimeGraph) -> ConnectomeController:
    return ConnectomeController.create(
        graph,
        run_id="embodiment-test",
        entity_id="test-fly",
    )


@pytest.fixture
def embodiment() -> ReferenceEmbodiment:
    return ReferenceEmbodiment(ReferenceArena())


def wide_sensor_channels(count: int = 4) -> tuple[SensorChannel, ...]:
    """``count`` distinct channels over the fixture's two sensory nodes."""

    return tuple(
        SensorChannel(
            name=f"wide_channel_{index}",
            unit="fraction",
            targets=(SensorTarget(FIXTURE_SENSORY_BODY_IDS[index % 2], 0.5 + 0.1 * index),),
            clip_min=0.0,
            clip_max=1.0,
        )
        for index in range(count)
    )


def wide_sensor_free_parameters(count: int = 4) -> int:
    """One target weight plus gain, offset, clip_min and clip_max per channel."""

    return count * 5


def fixture_joint_bindings(count: int = 4) -> tuple[JointBinding, ...]:
    return tuple(
        JointBinding(
            dof=f"lf_dof_{index}",
            pool=MotorPool(
                name=f"joint_pool_{index}",
                members=(PoolMember(FIXTURE_OUTPUT_BODY_IDS[index], 1.0),),
                gain=1.0,
            ),
            neutral_rad=0.1 * index,
            half_range_rad=0.4,
        )
        for index in range(count)
    )


def fixture_adhesion_bindings(count: int = 2) -> tuple[AdhesionBinding, ...]:
    return tuple(
        AdhesionBinding(
            leg=f"leg_{index}",
            pool=MotorPool(
                name=f"adhesion_pool_{index}",
                members=(PoolMember(FIXTURE_OUTPUT_BODY_IDS[4 + index], 1.0),),
                gain=1.0,
            ),
        )
        for index in range(count)
    )


def fixture_decoder_free_parameters(joints: int = 4, adhesion: int = 2) -> int:
    """Per pool: one member weight + gain + dead zone. Per joint: neutral + half range."""

    return (joints + adhesion) * 3 + joints * 2


def make_fixture_decoder(graph: RuntimeGraph) -> FlyGymMotorDecoder:
    return FlyGymMotorDecoder(
        graph,
        fixture_joint_bindings(),
        fixture_adhesion_bindings(),
        declared_free_parameters=fixture_decoder_free_parameters(),
    )


def kinematic_flygym_body(graph: RuntimeGraph) -> Any:
    """The kinematic-stub FlyGym body, single-sourced from ``test_flygym_body``.

    One import site for that module's private stub, so the run/replay lane is
    coupled to it in exactly one place. Nothing this body produces is a FlyGym or
    MuJoCo result: flygym is not installed, has never been executed, and the stub
    integrates no equation of motion.
    """

    from .test_flygym_body import _kinematic_body

    return _kinematic_body(graph)


def rewrite_trace(
    source: Path,
    destination: Path,
    mutate: Callable[[int, dict[str, Any]], dict[str, Any] | None],
    mutate_manifest: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> Path:
    """Rebuild a trace with mutated record payloads and a *valid* hash chain.

    ``mutate`` receives ``(record_index, payload)`` and returns a replacement payload,
    or ``None`` to drop the record. The manifest is copied verbatim, the entry hash
    chain is recomputed with the protocol's own canonical encoder, and the footer is
    rebuilt from the surviving records -- so ``verify_trace`` accepts the result and
    the corruption reaches ``replay_embodiment_trace``'s own checks rather than being
    stopped one layer earlier by an integrity error. That distinction is the whole
    point: these tests exercise the replay guarantee, not the trace format.
    """

    entries = [json.loads(line) for line in source.read_bytes().splitlines() if line]
    kept: list[tuple[str, dict[str, Any]]] = []
    record_index = 0
    for entry in entries:
        kind = entry["kind"]
        if kind == "footer":
            continue
        if kind == "record":
            payload = mutate(record_index, entry["payload"])
            record_index += 1
            if payload is None:
                continue
        elif kind == "manifest" and mutate_manifest is not None:
            payload = mutate_manifest(entry["payload"])
        else:
            payload = entry["payload"]
        kept.append((kind, payload))

    previous = _ZERO_HASH
    lines: list[bytes] = []
    manifest_hash = ""
    last_record_hash = ""
    record_count = 0
    for kind, payload in kept:
        material = {"kind": kind, "payload": payload, "previous_hash": previous}
        digest = hashlib.sha256(canonical_json_bytes(material)).hexdigest()
        lines.append(
            canonical_json_bytes(
                {
                    "kind": kind,
                    "payload": payload,
                    "previous_hash": previous,
                    "entry_hash": digest,
                }
            )
        )
        previous = digest
        if kind == "manifest":
            manifest_hash = digest
        elif kind == "record":
            record_count += 1
            last_record_hash = digest
    footer = {
        "record_count": record_count,
        "last_record_hash": last_record_hash or manifest_hash,
        "manifest_hash": manifest_hash,
        "final_record_seq": None if record_count == 0 else record_count - 1,
    }
    material = {"kind": "footer", "payload": footer, "previous_hash": previous}
    digest = hashlib.sha256(canonical_json_bytes(material)).hexdigest()
    lines.append(
        canonical_json_bytes(
            {
                "kind": "footer",
                "payload": footer,
                "previous_hash": previous,
                "entry_hash": digest,
            }
        )
    )
    destination.write_bytes(b"\n".join(lines) + b"\n")
    return destination


def make_controller(
    graph: RuntimeGraph,
    run_id: str = "reference-embodiment",
) -> ConnectomeController:
    return ConnectomeController.create(graph, run_id=run_id, entity_id="reference-fly")


def record_reference_trace(
    graph: RuntimeGraph,
    output: Path,
    ticks: int = TICKS,
) -> dict[str, Any]:
    return run_embodiment(
        controller=make_controller(graph),
        graph=graph,
        embodiment=ReferenceEmbodiment(ReferenceArena()),
        output=output,
        ticks=ticks,
    )


def replay_reference_trace(trace: Path, seed: int = 0) -> dict[str, Any]:
    return replay_embodiment_trace(
        trace=trace,
        graph_factory=lambda: load_malecns_fixture(FIXTURE),
        embodiment_factory=lambda: ReferenceEmbodiment(ReferenceArena(seed=seed)),
    )
