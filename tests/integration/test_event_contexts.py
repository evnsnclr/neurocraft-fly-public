"""Small real-controller comparisons; no full artifact or live world."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from neurocraft_controller.full_brain_controller import FullBrainController
from neurocraft_neural.full_controls import FullControlKind, build_control_view
from neurocraft_neural.full_engine import ArrayNativeEngine
from neurocraft_neural.spmv_kernel import NumpyReduceatKernel

from scripts import probe_event_contexts as probe
from scripts.probe_event_populations import Population
from tests.controller.test_embodied_senses import _observation
from tests.controller.test_loom_senses_v2 import typed_controller


def controller(mode: str = "faithful") -> FullBrainController:
    original = typed_controller()
    if mode == "faithful":
        return original
    view = original.engine.view
    control = build_control_view(
        view,
        kind=FullControlKind(mode),
        seed=7,
        artifact_manifest_sha256=original.identity.graph_hash,
        raw_counts=np.tile([3, 5], view.node_count),
        edge_signs=np.tile([1, -1], view.node_count),
    )
    return FullBrainController(
        run_id=original.identity.run_id,
        entity_id=original.identity.entity_id,
        configuration=replace(original.configuration, mode=mode),
        engine=ArrayNativeEngine(control.view, kernel=NumpyReduceatKernel()),
        sensory_binding=original._sensory_binding,
        motor_binding=original.escape_binding.wing,
        embodied=original.embodied_senses,
        escape=original.escape_binding,
        control=control,
    )


def populations() -> dict[str, Population]:
    # These are synthetic pairs in the test controller's node space. They do
    # not assert that fixture indices carry the full artifact's annotations.
    names = (*probe.PULSE_SOURCES, "mn9", "antennal_am", "ttmn", "dna01", "wing")
    return {
        name: Population(
            name,
            "synthetic pair",
            "fixture side",
            ((2 * i, 100 + 2 * i, "L"), (2 * i + 1, 101 + 2 * i, "R")),
        )
        for i, name in enumerate(names)
    }


@pytest.mark.parametrize("background", probe.BACKGROUNDS)
def test_background_path_is_exactly_actual_production_step(background: str) -> None:
    experimental, production = controller(), controller()
    encoder = probe.LoomV2BackgroundEncoder(experimental)
    encoder.reset_trial()
    schedule = probe.Schedule(1, 1, 2)
    for tick in range(schedule.total_ticks):
        sensors = probe.background_frame(background, tick, schedule)
        vector, record = encoder.encode(sensors)
        assert not vector.flags.writeable
        digest = probe.vector_sha(vector)
        production.step(_observation(production, tick, sensors), command_seq=tick)
        for _ in range(5):
            experimental.engine.step(sensory=vector, sensory_provenance_sha256="0" * 64)
        assert experimental.engine.step_index == production.engine.step_index == 5 * (tick + 1)
        np.testing.assert_array_equal(experimental.engine.activity(), production.engine.activity())
        assert (
            experimental.engine.current_state().state_sha256
            == production.engine.current_state().state_sha256
        )
        assert probe.vector_sha(vector) == digest
        evidence = production.last_evidence
        assert evidence is not None
        for name, value in record["eem_channels"].items():
            assert evidence.encoded_stimulation[f"eem:{name}"] == value


def test_backgrounds_exercise_tonic_light_transient_touch_and_entity_gating() -> None:
    core = controller()
    encoder = probe.LoomV2BackgroundEncoder(core)
    schedule = probe.Schedule(1, 1, 2)
    records = {}
    for background in probe.BACKGROUNDS:
        encoder.reset_trial()
        records[background] = [
            encoder.encode(probe.background_frame(background, tick, schedule))[1]
            for tick in range(schedule.total_ticks)
        ]
    assert all(row["eem_channels"]["light"] == 1 for row in records["open_air_bright"])
    assert all(row["eem_channels"]["light"] == 0 for row in records["open_air_dark"])
    for background in ("grounded_contact", "wall_contact"):
        values = [row["eem_channels"]["touch"] for row in records[background]]
        assert values[0] == 0 and values[1] > values[2] > values[3] > 0
    assert all(row["eem_channels"]["threat_front"] == 0 for row in records["wall_contact"])
    assert records["entity_approach"][1]["eem_channels"]["threat_front"] > 0


def test_composite_adds_after_background_freezes_values_and_gates_every_input_last() -> None:
    base = np.array([0.7, 0.0, 0.3])
    source = populations()["food_contact_source"]
    added = probe.compose_input(base, source, 1.0, zero_input=False)
    np.testing.assert_array_equal(added, [1.7, 1.0, 0.3])
    np.testing.assert_array_equal(base, [0.7, 0.0, 0.3])
    assert not added.flags.writeable
    assert probe.vector_sha(base) != probe.vector_sha(added)
    np.testing.assert_array_equal(
        probe.compose_input(base, source, 1.0, zero_input=True), [0, 0, 0]
    )


def test_paired_backgrounds_history_reset_deltas_and_explicit_experimental_identity() -> None:
    core = controller()
    schedule = probe.Schedule(1, 1, 2)
    result = probe.measure_context(core, populations(), "wall_contact", schedule)
    first_epoch = core.identity.session_epoch
    repeated = probe.measure_context(core, populations(), "wall_contact", schedule)
    assert core.identity.session_epoch == first_epoch + 3
    reference = result["trials"]["no_added_pulse"]
    for name, trial in result["trials"].items():
        descriptor = trial["experimental_descriptor"]
        assert descriptor["production_profile_claim"] is False
        assert probe.canonical_sha256(descriptor) == trial["experimental_descriptor_sha256"]
        assert trial["raw_signed_means"] == repeated["trials"][name]["raw_signed_means"]
        assert (
            trial["state_sha256_by_neural_step"][:5] == reference["state_sha256_by_neural_step"][:5]
        )
        assert [row["encoder"] for row in trial["inputs_by_game_tick"]] == [
            row["encoder"] for row in reference["inputs_by_game_tick"]
        ]
        if name == "no_added_pulse":
            continue
        for pool, values in trial["raw_signed_means"].items():
            baseline = reference["raw_signed_means"][pool]
            expected = [a - b for a, b in zip(values, baseline, strict=True)]
            assert trial["paired_raw_signed_difference"][pool] == expected
            assert trial["paired_rectified_mean_difference"][pool] == [
                max(a, 0) - max(b, 0) for a, b in zip(values, baseline, strict=True)
            ]
    assert any(
        v != 0
        for vs in result["trials"]["food_contact_source"]["paired_raw_signed_difference"].values()
        for v in vs
    )
    assert (
        reference["inputs_by_game_tick"][1]["requested_pulse_overlap_counts"]["food_contact_source"]
        == 0
    )
    assert (
        result["trials"]["food_contact_source"]["inputs_by_game_tick"][1][
            "requested_pulse_overlap_counts"
        ]["food_contact_source"]
        == 2
    )


@pytest.mark.parametrize("mode", ["shuffled_weight", "no_input"])
def test_real_controls_keep_requested_context_but_zero_final_composite_for_no_input(
    mode: str,
) -> None:
    core = controller(mode)
    result = probe.measure_context(core, populations(), "grounded_contact", probe.Schedule(1, 1, 1))
    for trial in result["trials"].values():
        component = trial["experimental_descriptor"]["component"]
        assert component["control_kind"] == mode
        assert component["control_identity_sha256"] == core.control_identity.identity_sha256
        rows = trial["inputs_by_game_tick"]
        assert rows[1]["encoder"]["eem_channels"]["touch"] > 0
        assert rows[1]["maximum_absolute_requested_input"] > 0
        if mode == "no_input":
            assert all(row["delivered_nonzero_nodes"] == 0 for row in rows)
            assert all(v == 0 for values in trial["raw_signed_means"].values() for v in values)
            assert all(
                v == 0 for row in rows for v in row["delivered_pulse_overlap_counts"].values()
            )
        else:
            assert all(row["delivered_nonzero_nodes"] > 0 for row in rows)


def test_default_schedule_preserves_v1_neural_timing() -> None:
    schedule = probe.Schedule()
    pulse = schedule.neural_pulse(5)
    assert (pulse.baseline_steps, pulse.duration_steps, pulse.recovery_steps) == (20, 5, 100)
    assert pulse.onset == 21 and pulse.total_steps == 125


def test_provenance_hashes_actual_imported_sources() -> None:
    sources = probe.code_sources()
    assert "neurocraft_neural.full_engine" in sources
    assert "neurocraft_controller.full_brain_controller" in sources
    for entry in sources.values():
        assert Path(entry["path"]).is_absolute()
        assert probe.file_sha256(Path(entry["path"])) == entry["sha256"]


@pytest.mark.parametrize(
    "flags",
    [
        ["--amplitude", "nan"],
        ["--amplitude", "0"],
        ["--baseline-ticks", "0"],
        ["--recovery-ticks", "401"],
        ["--threads", "0"],
        ["--seed", "8"],
        ["--controls", "--seed", str(2**64)],
        ["--backgrounds", "wall_contact", "wall_contact"],
    ],
)
def test_invalid_arguments_refuse_before_loading(flags, monkeypatch) -> None:
    monkeypatch.setattr(probe, "run_probe", lambda *args: pytest.fail("unexpected full artifact"))
    with pytest.raises(SystemExit):
        probe.main(flags)


def test_output_cannot_replace_existing_or_concurrently_created_file(tmp_path, monkeypatch) -> None:
    path = tmp_path / "output.json"
    monkeypatch.setattr(probe, "run_probe", lambda *args: {"synthetic": True})
    assert probe.main(["--output", str(path)]) == 0
    assert json.loads(path.read_text()) == {"synthetic": True}
    with pytest.raises(SystemExit):
        probe.main(["--output", str(path)])
    other = tmp_path / "concurrent.json"

    def concurrent(*args):
        other.write_text("preserve")
        return {}

    monkeypatch.setattr(probe, "run_probe", concurrent)
    with pytest.raises(FileExistsError):
        probe.main(["--output", str(other)])
    assert other.read_text() == "preserve"
