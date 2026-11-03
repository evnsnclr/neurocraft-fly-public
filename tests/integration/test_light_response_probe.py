"""Probe validation on small real engines, with no full graph or live Minecraft."""

from __future__ import annotations

import pytest
from neurocraft_protocol import Observation

from scripts import probe_light_response as probe
from tests.controller.test_events_senses import controllers


@pytest.mark.parametrize("case", probe.CASES)
def test_transition_changes_only_light_against_its_reference(case):
    reference = probe.reference_case(case)
    before, after, ground = probe.CASES[case]
    for tick in range(36):
        frame = probe.sensor_frame(case, tick)
        control = probe.sensor_frame(reference, tick)
        actual, expected = frame.to_dict(), control.to_dict()
        assert actual.pop("local_light") == (before if tick < 20 else after)
        expected.pop("local_light")
        assert actual == expected
        assert frame.on_ground is ground
        assert (
            frame.events.food.count
            == frame.events.brush.count
            == frame.events.attack_attempt.count
            == 0
        )


def test_pre_post_score_retains_background_drift_and_paired_difference_is_separate():
    reference = [float(i) / 100 for i in range(36)]
    signal = reference[:20] + [value - 0.5 for value in reference[20:]]
    null = probe.summarize(reference, reference)
    effect = probe.summarize(signal, reference)
    assert null["max_absolute_pre_post_change"] > 0
    assert null["max_absolute_paired_change_offline_only"] == 0
    assert effect["pre_event_four_observation_mean"] == pytest.approx(0.175)
    assert effect["max_negative_pre_post_change_magnitude"] == pytest.approx(0.475)
    assert effect["max_positive_pre_post_change"] == 0
    assert effect["max_absolute_paired_change_offline_only"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="complete"):
        probe.summarize(signal[:-1], reference)


def test_probe_matches_independent_direct_steps_and_reset_replays_exact_state():
    measured, _ = controllers()
    direct, _ = controllers()
    result = probe.measure_case(measured, "dim")
    for tick, row in enumerate(result["rows"]):
        sensor = probe.sensor_frame("dim", tick)
        direct.step(
            Observation(direct.identity, tick, tick, tick * 50_000_000, sensor), command_seq=tick
        )
        assert direct.last_evidence.state_after_sha256 == row["state_after_sha256"]
        assert row["neural_step_end"] - row["neural_step_start"] == 5
        assert row["raw_signed_means"]["antenna"] == direct.last_event_readout.antenna_raw
        assert not direct.last_event_readout.pulse.food
        assert not direct.last_event_readout.pulse.groom
    repeated = probe.measure_case(measured, "dim")
    assert [row["state_after_sha256"] for row in result["rows"]] == [
        row["state_after_sha256"] for row in repeated["rows"]
    ]


def test_no_input_control_has_zero_readouts_despite_nonzero_light_request():
    core, _ = controllers("no_input")
    result = probe.measure_case(core, "ground_brighten")
    assert result["rows"][20]["observation"]["sensors"]["local_light"] == 15
    assert all(value == 0 for row in result["rows"] for value in row["raw_signed_means"].values())
