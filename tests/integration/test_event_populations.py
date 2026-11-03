"""Real small recurrent-engine probes; no full artifact or biological outcomes."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
from neurocraft_neural.full_controls import FullControlKind, build_control_view
from neurocraft_neural.full_engine import ArrayDynamicsConfig, ArrayNativeEngine
from neurocraft_neural.full_view import FullRuntimeView, ViewProvenance, build_view
from neurocraft_neural.spmv_kernel import NumpyReduceatKernel

from scripts import probe_event_populations as probe
from scripts.readout_annotations import RetainedAnnotations


def fixture_annotations() -> RetainedAnnotations:
    # MN9 is deliberately inside pm, matching the published census nesting.
    # Sensory somaSide is absent: reading it instead of rootSide must fail.
    rows = [
        ("cb_sensory", "gustatory", "labellar bristle", "LB1", None, "L"),
        ("cb_sensory", "gustatory", "labellar bristle", "LB1", None, "R"),
        ("cb_sensory", "mechanosensory", "grooming", "JO-FV", None, "L"),
        ("cb_sensory", "mechanosensory", "grooming", "JO-FV", None, "R"),
        ("cb_motor", None, "pm", "MN9", "L", None),
        ("cb_motor", None, "pm", "MN9", "R", None),
        ("vnc_motor", None, "fl", "leg", "L", None),
        ("vnc_motor", None, "fl", "leg", "R", None),
        ("cb_motor", None, "am", "antenna", "L", None),
        ("cb_motor", None, "am", "antenna", "R", None),
        ("vnc_motor", None, "other", "TTMn", "L", None),
        ("vnc_motor", None, "other", "TTMn", "R", None),
        ("descending_neuron", None, None, "DNa01", "L", None),
        ("descending_neuron", None, None, "DNa01", "R", None),
        ("vnc_motor", None, "wm", "wing", "L", None),
        ("vnc_motor", None, "wm", "wing", "R", None),
        # Confusers must not get included by substring/one-column selection.
        ("cb_sensory", "gustatory", "taste peg", "other", None, "L"),
        ("cb_sensory", "other", "grooming", "other", None, "R"),
    ]
    columns = {
        name: np.asarray([row[i] for row in rows], dtype=object)
        for i, name in enumerate(
            ("superclass", "class", "subclass", "type", "somaSide", "rootSide")
        )
    }
    bodies = np.arange(100, 118, dtype=np.uint64)
    columns["bodyId"] = bodies.astype(object)
    return RetainedAnnotations(bodies, np.arange(18), columns, 18, "a" * 64)


def fixture_populations() -> dict[str, probe.Population]:
    return probe.select_populations(
        fixture_annotations(),
        expected_counts={name: (1, 1) for name in probe.PINNED_COUNTS},
        wing_ids=((114,), (115,)),
    )


def fixture_view() -> tuple[FullRuntimeView, np.ndarray, np.ndarray]:
    # Target rows: source -> MN9 (one positive, one negative); MN9 -> antenna.
    # DNa01 is disconnected; two unequal inputs to TTMn expose weight shuffling.
    rows = {
        4: [(0, 1, 1)],
        5: [(1, 1, -1)],
        6: [(2, 1, 1)],
        7: [(3, 1, 1)],
        8: [(4, 1, 1)],
        9: [(5, 1, 1)],
        10: [(0, 1, 1), (2, 7, 1)],
        11: [(1, 1, 1), (3, 7, 1)],
        14: [(8, 1, 1)],
        15: [(9, 1, 1)],
    }
    offsets, pre, counts, signs = [0], [], [], []
    for i in range(18):
        for source, count, sign in rows.get(i, []):
            pre.append(source)
            counts.append(count)
            signs.append(sign)
        offsets.append(len(pre))
    view = build_view(
        indptr=offsets,
        pre_indices=pre,
        raw_counts=counts,
        signs=signs,
        provenance=ViewProvenance("synthetic_event_probe", "a" * 64),
    )
    return view, np.asarray(counts), np.asarray(signs)


def test_exact_nonempty_sided_selection_and_membership_identity() -> None:
    populations = fixture_populations()
    assert populations["food_contact_source"].indices == (0, 1)
    assert populations["groom_touch_source"].indices == (2, 3)
    assert populations["food_contact_source"].side_column == "rootSide"
    assert populations["mn9"].indices == populations["proboscis_pm"].indices == (4, 5)
    assert populations["wing"].indices == (14, 15)
    original = populations["food_contact_source"]
    changed = replace(original, members=((0, 101, "L"), (1, 100, "R")))
    assert original.document()["membership_sha256"] != changed.document()["membership_sha256"]
    assert original.document()["side_counts"] == {"L": 1, "R": 1}
    with pytest.raises(ValueError, match="unexpected count/side"):
        probe.select_populations(fixture_annotations())  # Cannot impersonate full census.


@pytest.mark.parametrize("mutation", ["empty", "unknown_side", "extra_member"])
def test_selector_fails_closed_on_source_drift(mutation: str) -> None:
    annotations = fixture_annotations()
    if mutation == "empty":
        annotations.columns["class"][:2] = "other"
    elif mutation == "unknown_side":
        annotations.columns["rootSide"][0] = None
    else:
        annotations.columns["subclass"][16] = "labellar bristle"
    with pytest.raises(ValueError, match="food_contact_source"):
        probe.select_populations(
            annotations,
            expected_counts={name: (1, 1) for name in probe.PINNED_COUNTS},
            wing_ids=((114,), (115,)),
        )


def test_real_engine_lag_signed_rectification_no_event_and_recovery() -> None:
    view, _, _ = fixture_view()
    pulse = probe.Pulse(baseline_steps=2, duration_steps=2, recovery_steps=250)
    result = probe.measure_lane(
        view, fixture_populations(), pulse, kernel=NumpyReduceatKernel(), include_grooming=True
    )
    food = result["trials"]["food_contact_source"]
    means = food["raw_signed_means"]
    baseline = result["trials"]["no_event"]["raw_signed_means"]
    assert all(value == 0 for series in baseline.values() for value in series)
    baseline_states = result["trials"]["no_event"]["state_sha256_by_step"][: pulse.baseline_steps]
    for trial in result["trials"].values():
        assert trial["state_sha256_by_step"][: pulse.baseline_steps] == baseline_states
    expected_source_onset = (1 - math.exp(-10 / 50)) * math.tanh(1)
    assert means["food_contact_source"][2] == pytest.approx(expected_source_onset)
    assert food["responses"]["food_contact_source"]["first_above_floor_lag_steps"] == 0
    assert food["responses"]["mn9_left"]["first_above_floor_lag_steps"] == 2
    assert food["responses"]["antennal_am_left"]["first_above_floor_lag_steps"] == 4
    assert means["mn9_left"][4] > 0 and means["mn9_right"][4] < 0
    assert all(value == 0 for value in means["mn9"])  # Signed cancellation preserved.
    assert all(value == 0 for value in food["uncalibrated_rectified_means"]["mn9_right"])
    assert food["event_minus_no_event_signed_means"]["mn9_right"] == means["mn9_right"]
    assert food["responses"]["dna01"]["first_above_floor_step"] is None
    assert food["responses"]["dna01"]["recovery_tail_below_floor_from_step"] is None
    assert food["responses"]["mn9_left"]["recovery_tail_below_floor_from_step"] > pulse.onset
    assert abs(means["mn9_left"][-1]) < probe.NUMERICAL_FLOOR
    assert food["source_membership_overlap"]["food_contact_source"]["count"] == 2
    assert food["source_membership_overlap"]["mn9"]["count"] == 0
    assert food["executed_direct_input_overlap_counts"]["food_contact_source"] == 2
    assert food["executed_direct_input_overlap_counts"]["mn9"] == 0
    assert (
        result["trials"]["groom_touch_source"]["responses"]["front_leg_fl"][
            "first_above_floor_lag_steps"
        ]
        == 2
    )


def test_controls_are_executed_and_original_repeats_exactly() -> None:
    view, counts, signs = fixture_view()
    pools = fixture_populations()
    pulse = probe.Pulse(baseline_steps=2, duration_steps=5, recovery_steps=10)
    kernel = NumpyReduceatKernel()
    original = probe.measure_lane(view, pools, pulse, kernel=kernel)
    for kind in (FullControlKind.SHUFFLED_WEIGHT, FullControlKind.NO_INPUT):
        control = build_control_view(
            view,
            kind=kind,
            seed=7,
            artifact_manifest_sha256="a" * 64,
            raw_counts=counts,
            edge_signs=signs,
        )
        measured = probe.measure_lane(control.view, pools, pulse, kernel=kernel, control=control)
        assert measured["lane"] == kind.value
        assert measured["control_identity_sha256"] == control.identity.identity_sha256
        event = measured["trials"]["food_contact_source"]
        if kind == FullControlKind.NO_INPUT:
            assert event["executed_pulse_nonzero_nodes"] == 0
            assert event["source_membership_overlap"]["food_contact_source"]["count"] == 2
            assert all(
                count == 0 for count in event["executed_direct_input_overlap_counts"].values()
            )
            assert event["requested_pulse_vector_sha256"] != event["executed_pulse_vector_sha256"]
            assert all(
                value == 0 for series in event["raw_signed_means"].values() for value in series
            )
        else:
            assert event["executed_pulse_nonzero_nodes"] == 2
            assert (
                event["raw_signed_means"]["ttmn"]
                != original["trials"]["food_contact_source"]["raw_signed_means"]["ttmn"]
            )
    assert probe.measure_lane(view, pools, pulse, kernel=kernel) == original


def test_nonzero_no_event_baseline_is_subtracted_at_the_same_step(monkeypatch) -> None:
    # This test-only baseline makes failure to subtract the paired control observable.
    def engine_with_baseline(view, *, kernel):
        return ArrayNativeEngine(view, kernel=kernel, config=ArrayDynamicsConfig(baseline=0.05))

    monkeypatch.setattr(probe, "ArrayNativeEngine", engine_with_baseline)
    view, _, _ = fixture_view()
    pulse = probe.Pulse(baseline_steps=3, duration_steps=2, recovery_steps=5)
    result = probe.measure_lane(view, fixture_populations(), pulse, kernel=NumpyReduceatKernel())
    baseline = result["trials"]["no_event"]["raw_signed_means"]["mn9_left"]
    event = result["trials"]["food_contact_source"]
    raw = event["raw_signed_means"]["mn9_left"]
    delta = event["event_minus_no_event_signed_means"]["mn9_left"]
    assert all(value > 0 for value in baseline)
    assert delta == [value - control for value, control in zip(raw, baseline, strict=True)]
    assert delta[: pulse.baseline_steps] == [0.0] * pulse.baseline_steps
    assert delta != raw
    assert result["dynamics_configuration_values"]["baseline"] == 0.05
    assert (
        probe.canonical_sha256(result["dynamics_configuration"])
        == result["backend_dynamics_sha256"]
    )


def test_response_alignment_subtraction_and_bounded_recovery() -> None:
    pulse = probe.Pulse(baseline_steps=2, duration_steps=2, recovery_steps=3)
    result = probe.response_summary([0, 0, 0, -2e-12, 0, 3e-12, 0], pulse)
    assert result["first_above_floor_lag_steps"] == 1
    assert result["signed_response_at_peak"] == 3e-12
    assert result["recovery_tail_below_floor_from_step"] == 7  # A later excursion counts.
    assert result["recovery_tail_maximum_absolute_response"] == 3e-12
    assert result["recovery_tail_final_absolute_response"] == 0
    assert result["final_absolute_fraction_of_peak"] == 0
    assert probe.response_summary([0] * 7, pulse)["final_absolute_fraction_of_peak"] is None
    assert (
        probe.response_summary([0] * 6 + [2e-12], pulse)["recovery_tail_below_floor_from_step"]
        is None
    )


@pytest.mark.parametrize(
    "flags",
    [
        ["--pulse-amplitude", "nan"],
        ["--pulse-amplitude", "0"],
        ["--pulse-amplitude", "1.1"],
        ["--pulse-steps", "0"],
        ["--recovery-steps", "2000"],
        ["--threads", "0"],
        ["--seed", "8"],
        ["--controls", "--seed", str(2**64)],
    ],
)
def test_invalid_arguments_refuse_before_artifact_load(flags, monkeypatch) -> None:
    monkeypatch.setattr(probe, "run_probe", lambda *args: pytest.fail("unexpected artifact load"))
    with pytest.raises(SystemExit):
        probe.main(flags)


def test_json_output_exclusive_and_stdout_clean(tmp_path, monkeypatch, capsys) -> None:
    document = {"schema": "synthetic-test", "diagnostic_only": True}
    monkeypatch.setattr(probe, "run_probe", lambda *args: document)
    output = tmp_path / "probe.json"
    assert probe.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text()) == document
    assert capsys.readouterr().out == ""
    monkeypatch.setattr(probe, "run_probe", lambda *args: pytest.fail("unexpected artifact load"))
    with pytest.raises(SystemExit):
        probe.main(["--json", str(output)])
    assert json.loads(output.read_text()) == document


def test_output_created_during_measurement_is_not_overwritten(tmp_path, monkeypatch) -> None:
    output = tmp_path / "probe.json"

    def race(*args):
        output.write_text("keep concurrent output")
        return {"measured": True}

    monkeypatch.setattr(probe, "run_probe", race)
    with pytest.raises(FileExistsError):
        probe.main(["--json", str(output)])
    assert output.read_text() == "keep concurrent output"


def test_direct_cli_resolves_sibling_census_from_another_directory(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, str(probe.ROOT / "scripts/probe_event_populations.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "--pulse-amplitude" in result.stdout
