"""Actual arithmetic, recording and restoration checks for offline playground."""

import json
from hashlib import sha256

import numpy as np
import pytest
from neurocraft_connectome import SyntheticEdge, SyntheticNode, build_synthetic_graph
from neurocraft_controller.interaction_program_controller import (
    InteractionProgramSelectingController,
)
from neurocraft_neural.full_engine import ArrayNativeEngine
from neurocraft_neural.full_view import view_from_runtime_graph
from neurocraft_neural.model import SignOverlay
from neurocraft_neural.spmv_kernel import NumpyReduceatKernel

from scripts import run_circuit_playground as pilot
from tests.controller.test_interaction_senses import paired_controllers


def chain():
    graph = build_synthetic_graph(
        "synthetic:playground-active-chain",
        [SyntheticNode(i, ("interneuron",), consensus_nt="acetylcholine") for i in range(1, 4)],
        [SyntheticEdge(1, 2, 1, 0), SyntheticEdge(2, 3, 1, 1)],
    )
    view = view_from_runtime_graph(
        graph, sign_overlay=SignOverlay.coarse_presynaptic_consensus(graph)
    )
    ordinary = ArrayNativeEngine(view, kernel=NumpyReduceatKernel())
    recorder = pilot.RecordingEngine(view, kernel=NumpyReduceatKernel())
    recorder.configure_recording({"ttmn": [2], "internal": [1]})
    return ordinary, recorder


def test_passive_recorder_preserves_numerics_and_actual_input_hash():
    ordinary, recorder = chain()
    sensory = np.array([0.8, 0, 0])
    recorder.expected_input = sha256(sensory.astype(">f8").tobytes()).hexdigest()
    for _ in range(20):
        expected = ordinary.step(sensory=sensory)
        observed = recorder.step(sensory=sensory)
        assert expected == observed
    assert ordinary.activity()[2] > 0
    assert recorder.records[-1]["neural"]["encoded_stimulation_sha256"] == recorder.expected_input
    recorder.end_run()
    assert recorder.run_is_covered


def test_internal_clamp_suppresses_active_chain_and_reset_restores():
    ordinary, recorder = chain()
    sensory = np.array([0.8, 0, 0])
    for _ in range(20):
        ordinary.step(sensory=sensory)
        recorder.step(sensory=sensory)
    recorder.set_ablated([1])
    for _ in range(100):
        recorder.step(sensory=sensory, direct=np.array([0.0, 1.0, 0.0]))
        assert recorder.activity()[1] == 0
    assert recorder.activity()[2] < ordinary.activity()[2] * 1e-6
    assert recorder.view.view_sha256 == ordinary.view.view_sha256
    recorder.reset()
    ordinary.reset()
    for _ in range(20):
        recorder.step(sensory=sensory)
        ordinary.step(sensory=sensory)
    np.testing.assert_array_equal(recorder.activity(), ordinary.activity())


def test_changed_input_is_refused_before_step():
    _, recorder = chain()
    recorder.expected_input = "f" * 64
    with pytest.raises(RuntimeError, match="input differs"):
        recorder.step(sensory=np.ones(3))
    assert recorder.step_index == 0


def wrapped(mode="faithful"):
    core, _, _ = paired_controllers(mode)
    wrapper = InteractionProgramSelectingController(core)
    indices = [member.dense_index for member in core.escape_binding.jump_pool.members]
    engine = pilot.instrument(wrapper, {"ttmn": indices})
    return wrapper, engine


def test_actual_wrapper_evidence_and_exact_restore(tmp_path):
    wrapper, engine = wrapped()
    base = {"name": "original", "mode": "faithful", "indices": [], "body_ids": []}
    rows, _ = pilot.measure(wrapper, engine, "player_approach", base, tmp_path)
    restored, _ = pilot.measure(
        wrapper, engine, "player_approach", {**base, "name": "restored"}, tmp_path, rows
    )
    assert pilot.semantic_rows(rows) == pilot.semantic_rows(restored)
    coverage = json.loads((tmp_path / "restored-player_approach-coverage.json").read_text())
    assert coverage["run_is_covered"] and coverage["verified_through_step"] == 500
    assert all(not row["applied_in_minecraft"] for row in rows)
    assert all(len(row["neural_steps"]) == 5 for row in rows)


def test_no_input_zero_activity_and_no_event_selection(tmp_path):
    wrapper, engine = wrapped("no_input")
    condition = {"name": "no_input", "mode": "no_input", "indices": [], "body_ids": []}
    rows, _ = pilot.measure(wrapper, engine, "attack", condition, tmp_path)
    assert all(row["decision"]["source"] is None for row in rows)
    assert all(row["issued_program"]["program"] == "CRUISE" for row in rows)
    assert all(row["rectified_jump"] == 0 for row in rows)


@pytest.mark.parametrize("case", pilot.PILOT_CASES)
def test_fixed_confirmation_schedule_is_valid_and_baseline_paired(case):
    for tick in range(100):
        frame = pilot.frame(case, tick)
        assert frame.to_dict()
        if tick < 20:
            assert frame == pilot.frame("bright_sham", tick)


def test_output_refuses_existing_directory(tmp_path):
    with pytest.raises(FileExistsError):
        pilot.run(tmp_path, 1, [7, 11, 23], 10)


def test_restoration_detects_changed_issued_parameters(tmp_path):
    import copy

    wrapper, engine = wrapped()
    base = {"name": "original", "mode": "faithful", "indices": [], "body_ids": []}
    rows, _ = pilot.measure(wrapper, engine, "player_approach", base, tmp_path)
    changed = copy.deepcopy(rows)
    changed[20]["issued_program"]["modulation"]["urgency"] = 0.123456
    assert pilot.semantic_rows(changed) != pilot.semantic_rows(rows)
    changed = copy.deepcopy(rows)
    changed[20]["issued_program"]["parameters"]["turn_bias"] = 0.987654
    assert pilot.semantic_rows(changed) != pilot.semantic_rows(rows)
    changed = copy.deepcopy(rows)
    changed[20]["issued_program"]["target_tick"] += 1
    assert pilot.semantic_rows(changed) != pilot.semantic_rows(rows)
