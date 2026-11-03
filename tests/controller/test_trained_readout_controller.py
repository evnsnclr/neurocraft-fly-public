"""TRM demo controller: truthful identity rebinding, clipped decode, refusals."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
from neurocraft_connectome import RuntimeGraph, load_malecns_fixture
from neurocraft_connectome.flight_pin import MALECNS_FLIGHT_FIXTURE_V1_PIN
from neurocraft_connectome.model import canonical_json_bytes, sha256_json
from neurocraft_controller import ControllerConfiguration
from neurocraft_controller.controller import ControllerIntegrityError
from neurocraft_controller.flight_controller import (
    DEFAULT_V1_CONFIGURATION_SHA256,
    FlightConnectomeController,
)
from neurocraft_controller.trained_readout_controller import (
    TRM_DEMO_CHANNELS,
    TRM_DEMO_CONFIG_SCHEMA,
    TrainedReadoutController,
    TrmDemoConfiguration,
)

ROOT = Path(__file__).resolve().parents[2]
STEERING_FIXTURE = ROOT / "data" / "fixtures" / "mcns-v1.0-flight-steering-v1.json"


@pytest.fixture(scope="module")
def graph() -> RuntimeGraph:
    return load_malecns_fixture(
        STEERING_FIXTURE,
        expected_pin=MALECNS_FLIGHT_FIXTURE_V1_PIN,
        expected_file_bytes=284_949,
    )


def _artifact(
    graph: RuntimeGraph, *, intercept: tuple[float, float, float]
) -> tuple[bytes, dict[str, Any], str]:
    width = len(graph.nodes) + 1
    weights = [[0.0, 0.0, 0.0] for _ in range(width)]
    weights[-1] = list(intercept)
    hex_rows = [[float(v).hex() for v in row] for row in weights]
    document: dict[str, Any] = {
        "schema": "neurocraft.trm.cloned_readout.v1",
        "demo_id": "trm-live-demo-v1",
        "graph_runtime_identity_sha256": graph.manifest.runtime_identity_sha256,
        "channels": list(TRM_DEMO_CHANNELS),
        "weights_float_hex": hex_rows,
        "weights_identity_sha256": sha256_json(hex_rows),
    }
    payload = canonical_json_bytes(document) + b"\n"
    return payload, document, sha256(payload).hexdigest()


def _controller(graph: RuntimeGraph, intercept=(0.25, -0.1, 5.0)) -> TrainedReadoutController:
    payload, document, digest = _artifact(graph, intercept=intercept)
    return TrainedReadoutController(
        graph,
        run_id="trm-demo-test",
        entity_id="trm-demo-test-fly",
        artifact_bytes=payload,
        artifact_document=document,
        configuration=TrmDemoConfiguration(readout_artifact_sha256=digest),
    )


def test_configuration_has_its_own_schema_and_binds_the_artifact(graph: RuntimeGraph) -> None:
    _, _, digest = _artifact(graph, intercept=(0.0, 0.0, 0.0))
    config = TrmDemoConfiguration(readout_artifact_sha256=digest)
    assert config.schema == TRM_DEMO_CONFIG_SCHEMA
    assert config.identity_sha256 != DEFAULT_V1_CONFIGURATION_SHA256
    material = config.to_identity_material()
    assert material["readout_artifact_sha256"] == digest
    assert material["projection"]["vertical_thrust"] == "clip(forward)"
    with pytest.raises(ValueError, match="SHA-256"):
        TrmDemoConfiguration(readout_artifact_sha256="not-a-hash")
    with pytest.raises(ValueError, match="inert"):
        TrmDemoConfiguration(readout_artifact_sha256=digest, left_motor_body_id=800_184)


def test_controller_rebinds_every_decoder_identity(graph: RuntimeGraph) -> None:
    controller = _controller(graph)
    wing = FlightConnectomeController(graph, run_id="trm-demo-test", entity_id="trm-demo-test-fly")
    assert controller.identity.decoder_hash != wing.identity.decoder_hash
    assert controller.identity.adapter_hash == wing.identity.adapter_hash
    assert controller.identity.dynamics_hash == wing.identity.dynamics_hash
    # The composite decoder hash must change when the artifact changes.
    other = _controller(graph, intercept=(0.3, -0.1, 5.0))
    assert other.identity.decoder_hash != controller.identity.decoder_hash


def test_decode_clips_and_projects_with_vertical_equal_forward(graph: RuntimeGraph) -> None:
    controller = _controller(graph, intercept=(0.25, -0.1, 5.0))
    frame = controller.decoder.decode(controller.engine.current_state())
    values = dict(frame.values)
    assert values["forward"] == pytest.approx(0.25)
    assert values["lateral"] == pytest.approx(-0.1)
    assert values["yaw"] == 1.0  # intercept 5.0 clipped
    motors = controller._project_motor_frame(frame)
    assert motors.forward_thrust == pytest.approx(0.25)
    assert motors.vertical_thrust == motors.forward_thrust
    assert motors.lateral_thrust == pytest.approx(-0.1)
    assert motors.yaw_torque == 1.0
    assert motors.pitch_torque == 0.0


def test_refusals(graph: RuntimeGraph) -> None:
    payload, document, digest = _artifact(graph, intercept=(0.0, 0.0, 0.0))
    with pytest.raises(ControllerIntegrityError, match="TrmDemoConfiguration"):
        TrainedReadoutController(
            graph,
            run_id="t",
            entity_id="e",
            artifact_bytes=payload,
            artifact_document=document,
            configuration=ControllerConfiguration(),
        )
    with pytest.raises(ControllerIntegrityError, match="do not match"):
        TrainedReadoutController(
            graph,
            run_id="t",
            entity_id="e",
            artifact_bytes=payload + b" ",
            artifact_document=document,
            configuration=TrmDemoConfiguration(readout_artifact_sha256=digest),
        )
    bad = dict(document)
    bad["channels"] = ["forward", "yaw"]
    # Any document divergence from the bytes now trips the binding gate first.
    with pytest.raises(ControllerIntegrityError, match="does not match the artifact bytes"):
        TrainedReadoutController(
            graph,
            run_id="t",
            entity_id="e",
            artifact_bytes=payload,
            artifact_document=bad,
            configuration=TrmDemoConfiguration(readout_artifact_sha256=digest),
        )


#: The current profile artifact (trm-live-demo-v2), the one the bridge
#: profile row loads today.
COMMITTED_ARTIFACT = ROOT / "data" / "readouts" / "trm-cloned-readout-v2.json"
COMMITTED_ARTIFACT_SHA256 = "23938cd78cc7783de63af02343f7a0d89b40e707e2951c4203e2a90c6b6d8db0"
#: The superseded v1 artifact stays committed as recorded history (D-047,
#: EXTERNAL-BRIDGE-06); its bytes must never drift.
HISTORICAL_V1_ARTIFACT = ROOT / "data" / "readouts" / "trm-cloned-readout-v1.json"
HISTORICAL_V1_ARTIFACT_SHA256 = "527eb4f7042eee4eaab17f7b371c5a57cff5a92028e73c230b17b3fcf19dc3c6"


def test_forged_document_with_honest_bytes_is_refused(graph: RuntimeGraph) -> None:
    """The demonstrated spoof: honest artifact bytes plus a self-consistent
    forged document must never construct (adversarial-review HIGH finding)."""

    import json

    payload, document, digest = _artifact(graph, intercept=(0.0, 0.0, 0.0))
    forged = json.loads(json.dumps(document))
    width = len(graph.nodes) + 1
    forged_weights = [[0.0, 0.0, 0.0] for _ in range(width)]
    forged_weights[-1] = [1.0, 0.0, 0.0]
    forged["weights_float_hex"] = [[float(v).hex() for v in row] for row in forged_weights]
    forged["weights_identity_sha256"] = sha256_json(forged["weights_float_hex"])
    with pytest.raises(ControllerIntegrityError, match="does not match the artifact bytes"):
        TrainedReadoutController(
            graph,
            run_id="t",
            entity_id="e",
            artifact_bytes=payload,
            artifact_document=forged,
            configuration=TrmDemoConfiguration(readout_artifact_sha256=digest),
        )


def test_committed_artifact_is_pinned_and_constructs(graph: RuntimeGraph) -> None:
    """CI must notice a corrupted or drifted committed artifact (review gap)."""

    import json

    payload = COMMITTED_ARTIFACT.read_bytes()
    assert sha256(payload).hexdigest() == COMMITTED_ARTIFACT_SHA256
    document = json.loads(payload)
    controller = TrainedReadoutController(
        graph,
        run_id="trm-demo-pin-test",
        entity_id="trm-demo-pin-fly",
        artifact_bytes=payload,
        artifact_document=document,
        configuration=TrmDemoConfiguration(readout_artifact_sha256=COMMITTED_ARTIFACT_SHA256),
    )
    assert controller.identity.decoder_hash != DEFAULT_V1_CONFIGURATION_SHA256
    assert len(controller.identity.decoder_hash) == 64
    assert document["demo_id"] == "trm-live-demo-v2"


def test_superseded_v1_artifact_bytes_remain_the_recorded_history() -> None:
    """The v1 artifact is history (D-047, EXTERNAL-BRIDGE-06), not the live
    profile artifact; its committed bytes must stay exactly as receipted."""

    payload = HISTORICAL_V1_ARTIFACT.read_bytes()
    assert sha256(payload).hexdigest() == HISTORICAL_V1_ARTIFACT_SHA256

    import json

    assert json.loads(payload)["demo_id"] == "trm-live-demo-v1"


def test_trained_readout_refusal_paths(graph: RuntimeGraph) -> None:
    """Artifact-parsing refusals: schema, graph binding, shape, finiteness,
    weights-identity (adversarial-review LOW gap)."""

    import json

    from neurocraft_controller.trained_readout_controller import TrainedReadout

    _, document, _ = _artifact(graph, intercept=(0.0, 0.0, 0.0))

    bad = json.loads(json.dumps(document))
    bad["schema"] = "neurocraft.trm.other.v1"
    with pytest.raises(ControllerIntegrityError, match="schema"):
        TrainedReadout(bad, graph=graph)

    bad = json.loads(json.dumps(document))
    bad["channels"] = ["forward", "yaw"]
    with pytest.raises(ControllerIntegrityError, match="channels"):
        TrainedReadout(bad, graph=graph)

    bad = json.loads(json.dumps(document))
    bad["graph_runtime_identity_sha256"] = "0" * 64
    with pytest.raises(ControllerIntegrityError, match="different graph"):
        TrainedReadout(bad, graph=graph)

    bad = json.loads(json.dumps(document))
    bad["weights_float_hex"] = bad["weights_float_hex"][:-1]
    with pytest.raises(ControllerIntegrityError, match="hidden state"):
        TrainedReadout(bad, graph=graph)

    bad = json.loads(json.dumps(document))
    bad["weights_float_hex"][0] = [float("inf").hex(), "0x0.0p+0", "0x0.0p+0"]
    with pytest.raises(ControllerIntegrityError, match="finite"):
        TrainedReadout(bad, graph=graph)

    bad = json.loads(json.dumps(document))
    bad["weights_identity_sha256"] = "0" * 64
    with pytest.raises(ControllerIntegrityError, match="identity mismatch"):
        TrainedReadout(bad, graph=graph)
