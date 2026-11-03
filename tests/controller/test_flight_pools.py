from __future__ import annotations

from pathlib import Path

import pytest
from neurocraft_connectome import load_malecns_fixture
from neurocraft_connectome.flight_pin import MALECNS_FLIGHT_FIXTURE_V1_PIN
from neurocraft_controller.flight_pools import (
    LEFT_WING_MOTOR_BODY_IDS,
    RIGHT_WING_MOTOR_BODY_IDS,
    FlightPoolError,
    derive_wing_motor_pools,
)
from neurocraft_neural.adapters import MAX_MOTOR_MEMBERS_TOTAL

ROOT = Path(__file__).resolve().parents[2]
FLIGHT_FIXTURE = ROOT / "data" / "fixtures" / "mcns-v1.0-flight-steering-v1.json"
FLIGHT_FIXTURE_BYTES = 284_949


def _flight_graph():
    return load_malecns_fixture(
        FLIGHT_FIXTURE,
        expected_pin=MALECNS_FLIGHT_FIXTURE_V1_PIN,
        expected_file_bytes=FLIGHT_FIXTURE_BYTES,
    )


def test_declared_membership_matches_the_committed_fixture() -> None:
    """The declared body IDs are re-derived, never trusted.

    A fixture that stops yielding exactly this set must not silently produce a
    different readout under the same pool name.
    """

    pools = derive_wing_motor_pools(_flight_graph(), gain=8.0, dead_zone=0.0)
    assert tuple(pool.name for pool in pools) == ("left_motor", "right_motor")
    left, right = pools
    assert tuple(member.body_id for member in left.members) == LEFT_WING_MOTOR_BODY_IDS
    assert tuple(member.body_id for member in right.members) == RIGHT_WING_MOTOR_BODY_IDS
    assert len(left.members) == 25
    assert len(right.members) == 27


def test_every_member_carries_unit_weight_so_the_pool_is_a_side_mean() -> None:
    """MotorDecoder divides by the summed absolute weight.

    Unit weights therefore make each pool the mean activity of its soma side, which
    keeps the readout on the same scale as the shipped single-neuron pools instead of
    summing to roughly 26x them.
    """

    for pool in derive_wing_motor_pools(_flight_graph(), gain=8.0, dead_zone=0.0):
        assert {member.weight for member in pool.members} == {1.0}


def test_the_membership_fits_the_decoder_sparse_budget() -> None:
    total = len(LEFT_WING_MOTOR_BODY_IDS) + len(RIGHT_WING_MOTOR_BODY_IDS)
    assert total == 52
    assert total <= MAX_MOTOR_MEMBERS_TOTAL


def test_the_shipped_fixture_cannot_host_this_readout() -> None:
    """The 14-node circuit has two motor neurons, not these fifty-two."""

    shipped = load_malecns_fixture(ROOT / "data" / "fixtures" / "mcns-v1.0-flight-power-v1.json")
    with pytest.raises(FlightPoolError, match="does not match the declared set"):
        derive_wing_motor_pools(shipped, gain=8.0, dead_zone=0.0)


def test_the_two_sides_are_disjoint_and_the_split_is_the_datasets_own() -> None:
    """No neuron may be counted on both sides of a differential readout."""

    assert not set(LEFT_WING_MOTOR_BODY_IDS) & set(RIGHT_WING_MOTOR_BODY_IDS)
    graph = _flight_graph()
    sides = {node.body_id: node.side for node in graph.nodes if "motor" in node.roles}
    assert {sides[body_id] for body_id in LEFT_WING_MOTOR_BODY_IDS} == {"L"}
    assert {sides[body_id] for body_id in RIGHT_WING_MOTOR_BODY_IDS} == {"R"}


def test_the_shipped_right_readout_neuron_is_retained_but_the_left_one_is_absent() -> None:
    """Recorded because it is surprising and bears on any comparison between circuits.

    Body 800917, the right `hg1 MN` the 14-node circuit reads out through, is also a
    member here. Its left partner 800241 is not in the flight fixture at all, which is
    why the shipped decoder cannot be built on this graph.
    """

    assert 800_917 in RIGHT_WING_MOTOR_BODY_IDS
    assert 800_241 not in LEFT_WING_MOTOR_BODY_IDS
    assert 800_241 not in RIGHT_WING_MOTOR_BODY_IDS
