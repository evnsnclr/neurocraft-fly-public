"""Opt-in flight controller: wing readout bound truthfully, defaults untouched.

The whole point of the module is identity hygiene: the wing-motor readout must
carry its own configuration schema and decoder identity while the shipped v1
defaults — pinned inside the EXTERNAL-BRIDGE-02/-03/-04 trace manifests — stay
byte-for-byte reproducible. These tests pin both directions.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from neurocraft_connectome import RuntimeGraph, load_malecns_fixture
from neurocraft_connectome.flight_pin import MALECNS_FLIGHT_FIXTURE_V1_PIN
from neurocraft_controller import ConnectomeController, ControllerConfiguration
from neurocraft_controller.controller import ControllerIntegrityError
from neurocraft_controller.flight_controller import (
    DEFAULT_V1_CONFIGURATION_SHA256,
    FLIGHT_CONFIG_SCHEMA,
    FlightConnectomeController,
    FlightControllerConfiguration,
)
from neurocraft_controller.flight_pools import (
    LEFT_WING_MOTOR_BODY_IDS,
    RIGHT_WING_MOTOR_BODY_IDS,
    FlightPoolError,
)

ROOT = Path(__file__).resolve().parents[2]
STEERING_FIXTURE = ROOT / "data" / "fixtures" / "mcns-v1.0-flight-steering-v1.json"
POWER_FIXTURE = ROOT / "data" / "fixtures" / "mcns-v1.0-flight-power-v1.json"


@pytest.fixture(scope="module")
def steering_graph() -> RuntimeGraph:
    return load_malecns_fixture(
        STEERING_FIXTURE,
        expected_pin=MALECNS_FLIGHT_FIXTURE_V1_PIN,
        expected_file_bytes=284_949,
    )


def test_default_v1_configuration_identity_is_pinned() -> None:
    """The EXTERNAL-BRIDGE-02/-03/-04 manifests pin this exact hash; any change
    to the default configuration identity silently breaks current-tree replay
    of the tracked EB-04 evidence, so it is a named regression here."""

    assert ControllerConfiguration().identity_sha256 == DEFAULT_V1_CONFIGURATION_SHA256


def test_flight_configuration_has_its_own_schema_and_identity() -> None:
    config = FlightControllerConfiguration()
    assert config.schema == FLIGHT_CONFIG_SCHEMA
    assert config.identity_sha256 != DEFAULT_V1_CONFIGURATION_SHA256
    material = config.to_identity_material()
    assert material["motor_pool_profile"] == "wing_motor_soma_side_v1"
    # The inert single motor bodies must not appear in the identity material.
    assert "left_motor_body_id" not in str(material)
    # The a-priori gain policy: the unchanged v1 boundary values, declared
    # before any steering-circuit bridge execution.
    assert config.motor_pool_gain == 8.0
    assert config.motor_dead_zone == 0.0


def test_flight_configuration_refuses_customized_inert_motor_ids() -> None:
    with pytest.raises(ValueError, match="inert"):
        FlightControllerConfiguration(left_motor_body_id=800_184)
    with pytest.raises(ValueError, match="schema"):
        FlightControllerConfiguration(schema="neurocraft.controller.config.v1")
    with pytest.raises(ValueError, match="profile"):
        FlightControllerConfiguration(motor_pool_profile="functional_split_v1")


def test_flight_controller_binds_the_derived_wing_pools(
    steering_graph: RuntimeGraph,
) -> None:
    from neurocraft_controller.flight_pools import derive_wing_motor_pools
    from neurocraft_neural.adapters import MotorDecoder

    controller = FlightConnectomeController(
        steering_graph,
        run_id="flight-controller-test",
        entity_id="flight-test-fly",
    )
    expected = MotorDecoder(
        steering_graph,
        derive_wing_motor_pools(steering_graph, gain=8.0, dead_zone=0.0),
    )
    assert controller.decoder.identity_sha256 == expected.identity_sha256
    assert controller.configuration.identity_sha256 != DEFAULT_V1_CONFIGURATION_SHA256
    # The membership constants are still pinned end-to-end: a decoder built from
    # any other membership or gain policy cannot reproduce this identity.
    assert len(LEFT_WING_MOTOR_BODY_IDS) == 25
    assert len(RIGHT_WING_MOTOR_BODY_IDS) == 27


def test_flight_controller_decoder_identity_differs_from_pair_readout(
    steering_graph: RuntimeGraph,
) -> None:
    """The same fixture under the v1 pair readout (EMBODIMENT-02's workaround)
    and under the wing profile must be distinguishable by decoder identity."""

    pair = ConnectomeController.create(
        steering_graph,
        run_id="flight-controller-test",
        entity_id="flight-test-fly",
        configuration=ControllerConfiguration(
            left_motor_body_id=LEFT_WING_MOTOR_BODY_IDS[0],
            right_motor_body_id=RIGHT_WING_MOTOR_BODY_IDS[0],
        ),
    )
    wing = FlightConnectomeController(
        steering_graph,
        run_id="flight-controller-test",
        entity_id="flight-test-fly",
    )
    assert wing.identity.decoder_hash != pair.identity.decoder_hash
    assert wing.identity.adapter_hash == pair.identity.adapter_hash
    assert wing.identity.dynamics_hash == pair.identity.dynamics_hash


def test_flight_controller_refuses_a_plain_v1_configuration(
    steering_graph: RuntimeGraph,
) -> None:
    with pytest.raises(ControllerIntegrityError, match="FlightControllerConfiguration"):
        FlightConnectomeController(
            steering_graph,
            run_id="flight-controller-test",
            entity_id="flight-test-fly",
            configuration=ControllerConfiguration(),
        )


def test_flight_controller_refuses_the_power_fixture() -> None:
    """The 14-node fixture does not carry the declared wing-motor membership;
    the derivation must hard-fail rather than fall back to anything."""

    graph = load_malecns_fixture(POWER_FIXTURE)
    with pytest.raises(FlightPoolError):
        FlightConnectomeController(
            graph,
            run_id="flight-controller-test",
            entity_id="flight-test-fly",
        )


def test_flight_configuration_refuses_a_wing_member_as_sensor() -> None:
    """The parent's sensor-vs-motor disjointness rule, restated against the
    real 52-member readout (adversarial-review finding F2)."""

    with pytest.raises(ValueError, match="wing-motor pool member"):
        FlightControllerConfiguration(left_sensor_body_id=RIGHT_WING_MOTOR_BODY_IDS[0])
    with pytest.raises(ValueError, match="wing-motor pool member"):
        FlightControllerConfiguration(right_sensor_body_id=LEFT_WING_MOTOR_BODY_IDS[0])


def test_flight_configuration_inert_motor_ids_require_exact_ints() -> None:
    """A float that compares equal to the default is still a type violation
    (adversarial-review finding F3)."""

    with pytest.raises(ValueError, match="inert"):
        FlightControllerConfiguration(left_motor_body_id=800_241.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="inert"):
        FlightControllerConfiguration(right_motor_body_id=True)  # type: ignore[arg-type]
