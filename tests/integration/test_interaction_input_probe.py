"""Prescribed V4 observations and actual small engines; no Minecraft acceptance."""

from dataclasses import replace

import pytest
from neurocraft_protocol import Observation, SensorFrameV4

from scripts import probe_interaction_inputs as probe
from tests.controller.test_interaction_senses import paired_controllers


@pytest.mark.parametrize("case", probe.CASES)
def test_diagnostic_histories_have_valid_v4_observations_and_one_event_onset(case):
    for tick in range(probe.BASELINE_TICKS + probe.RESPONSE_TICKS):
        frame = probe.sensor_frame(case, tick)
        assert SensorFrameV4.from_dict(frame.to_dict()) == frame
        for event_name, expected_case in (
            ("food", "food"),
            ("brush", "brush"),
            ("attack_attempt", "attack"),
        ):
            counter = getattr(frame.events, event_name)
            active = tick >= probe.BASELINE_TICKS and case == expected_case
            assert counter.count == int(active)
            assert counter.last_game_tick == (probe.BASELINE_TICKS if active else None)


def test_actor_histories_differ_only_by_supplied_category_and_food_has_collision_control():
    for tick in range(probe.BASELINE_TICKS + probe.RESPONSE_TICKS):
        player = probe.sensor_frame("player_approach", tick)
        mob = probe.sensor_frame("mob_approach", tick)
        assert replace(player, ray_actor_kinds=mob.ray_actor_kinds) == mob
        if tick >= probe.BASELINE_TICKS:
            assert player.ray_actor_kinds[0] == 1 and mob.ray_actor_kinds[0] == 2
        food = probe.sensor_frame("food", tick)
        contact = probe.sensor_frame("collision", tick)
        assert replace(food, events=contact.events) == contact


def test_actual_probe_records_direct_state_and_identical_actor_dynamics():
    measured, _, _ = paired_controllers()
    direct, _, _ = paired_controllers()
    result = probe.measure_case(measured, "attack")
    for tick, row in enumerate(result["rows"]):
        direct.step(
            Observation(
                direct.identity, tick, tick, tick * 50_000_000, probe.sensor_frame("attack", tick)
            ),
            command_seq=tick,
        )
        assert row["state_after_sha256"] == direct.last_evidence.state_after_sha256
        assert row["neural_step_end"] - row["neural_step_start"] == 5
        assert row["completed_pulse"]["attack"] == (tick == probe.BASELINE_TICKS)
        assert not row["completed_pulse"]["groom"]
    player = probe.measure_case(measured, "player_approach")
    mob = probe.measure_case(measured, "mob_approach")
    assert [row["state_after_sha256"] for row in player["rows"]] == [
        row["state_after_sha256"] for row in mob["rows"]
    ]


@pytest.mark.parametrize("case", ["attack", "brush", "food", "light_off", "player_approach"])
def test_no_input_readouts_remain_zero_for_new_histories(case):
    core, _, _ = paired_controllers("no_input")
    result = probe.measure_case(core, case)
    assert all(value == 0 for row in result["rows"] for value in row["raw_signed_means"].values())
    assert any(
        any(value != 0 for value in row["requested_before_control"].values())
        for row in result["rows"]
    )
