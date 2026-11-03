"""Offline benchmark evidence checked against actual small engines and direct wrappers."""

import json
from dataclasses import replace
from hashlib import sha256

import pytest
from neurocraft_controller.interaction_program_controller import (
    InteractionProgramSelectingController,
)
from neurocraft_protocol import Observation, ProgramId

from scripts import benchmark_interaction_policy as benchmark
from tests.controller.test_interaction_senses import paired_controllers


def wrapper(mode="faithful"):
    core, _, _ = paired_controllers(mode)
    return InteractionProgramSelectingController(core)


@pytest.mark.parametrize("case", ["attack", "mob_approach", "light_off"])
def test_measure_case_matches_direct_actual_commands_evidence_and_step_bounds(case):
    measured, direct = wrapper(), wrapper()
    result = benchmark.measure_case(measured, case)
    direct.reset_session(session_epoch=direct.identity.session_epoch + 1)
    assert len(result["rows"]) == 100
    for tick, row in enumerate(result["rows"]):
        observation = Observation(
            direct.identity,
            tick,
            tick,
            tick * 50_000_000,
            benchmark.sensor_frame(case, tick),
        )
        command = direct.step(observation, command_seq=tick)
        assert row["observation"] == observation.to_dict()
        assert row["program_action"] == command.to_dict()
        assert row["named_evidence"] == direct.last_event_evidence.to_dict()
        assert row["state_after_sha256"] == direct.controller.last_evidence.state_after_sha256
        assert (
            row["full_activity_sha256"] == direct.controller.engine.current_state().activity_sha256
        )
        assert row["executed_stimulation_sha256"] is None
        assert row["neural_step_start"] == 5 * tick
        assert row["neural_step_end"] == 5 * (tick + 1)
        assert row["raw_projection_not_applied"] == direct.last_evidence.raw_projection.to_dict()
        assert row["wrapper_compute_ns"] >= 0
        assert "application" not in row
    summary = result["summary"]
    assert summary["observations"] == 100 and summary["neural_updates"] == 500
    assert sum(summary["program_counts"].values()) == 100
    assert sum(segment["issued_observations"] for segment in summary["issued_segments"]) == 100
    assert summary["compute"]["samples"] == 100
    assert "not live" in summary["compute"]["scope"]
    # Every reported entry joins an actual issued command and evidence source.
    for entry in summary["entries"]:
        row = result["rows"][entry["game_tick"]]
        assert entry["program"] == row["program_action"]["program"]
        assert entry["source"] == row["selection_source"]
        assert row["selection_reason"] == "fresh_eligible_response"
    repeat = benchmark.measure_case(measured, case)
    assert repeat["identity"]["session_epoch"] == result["identity"]["session_epoch"] + 1
    assert [row["state_after_sha256"] for row in repeat["rows"]] == [
        row["state_after_sha256"] for row in result["rows"]
    ]


@pytest.mark.parametrize("case", benchmark.inputs.CASES)
def test_no_input_complete_zero_vectors_and_actual_programs_for_every_prescribed_history(case):
    controller = wrapper("no_input")
    result = benchmark.measure_case(controller, case)
    zero = sha256(bytes(controller.controller.engine.view.node_count * 8)).hexdigest()
    assert controller.controller.engine.step_index == 500
    for row in result["rows"]:
        assert row["executed_stimulation_sha256"] == row["full_activity_sha256"] == zero
        assert row["selection_source"] is None
        assert row["program_action"]["program"] == ProgramId.CRUISE.value
    assert result["summary"]["entries"] == []
    assert result["summary"]["program_counts"]["CRUISE"] == 100


@pytest.mark.parametrize("failure", ["delivered_input", "activity", "program"])
def test_no_input_guard_rejects_nonzero_complete_digest_or_invented_event_program(
    monkeypatch, failure
):
    controller = wrapper("no_input")
    original = controller.step

    def corrupted(observation, *, command_seq):
        command = original(observation, command_seq=command_seq)
        if failure == "delivered_input":
            evidence = controller.last_evidence
            controller._last_evidence = replace(
                evidence, inner=replace(evidence.inner, executed_stimulation_sha256="f" * 64)
            )
        elif failure == "activity":
            engine = controller.controller.engine
            state = engine.current_state()
            monkeypatch.setattr(
                engine, "current_state", lambda: replace(state, activity_sha256="f" * 64)
            )
        else:
            command = replace(command, program=ProgramId.FLEE)
        return command

    monkeypatch.setattr(controller, "step", corrupted)
    with pytest.raises(RuntimeError, match="complete sensory/activity zero or event-program gate"):
        benchmark.measure_case(controller, "attack")


@pytest.mark.parametrize("mode", benchmark.LANES)
def test_production_builder_receives_exact_profile_control_seed_and_threads(monkeypatch, mode):
    actual = wrapper(mode)
    received = []

    def build(args):
        received.append(args)
        return actual

    monkeypatch.setattr(benchmark.service, "_build_program_selecting_controller", build)
    assert benchmark.build_controller(mode, 8) is actual
    args = received[0]
    assert args.controller_profile == "full_mcns_interactions_program_v2"
    assert args.mode == mode and args.control_seed == 7 and args.kernel_threads == 8
    assert args.full_uncalibrated is False


def test_full_schedule_denominators_and_shared_policy_with_real_small_engines(monkeypatch):
    # Exercise the full orchestrator on small actual engines. No source-unchanged
    # acceptance is inferred from this test's fixed injected inventory.
    inventory = benchmark.sources()
    monkeypatch.setattr(benchmark, "sources", lambda: inventory)
    built = []

    def build(mode, threads):
        result = wrapper(mode)
        built.append(result)
        return result

    monkeypatch.setattr(benchmark, "build_controller", build)
    result = benchmark.run_benchmark(1)
    assert result["case_count"] == 30 and result["observations"] == 3000
    assert result["neural_updates"] == 15000
    assert result["same_policy_declaration_across_lanes"]
    assert len({lane["policy_hash"] for lane in result["lanes"]}) == 1
    assert result["applied_commands"] is None
    assert "unobserved" in result["application_status"]
    assert all(
        lane["observations"] == 1000 and lane["neural_updates"] == 5000 for lane in result["lanes"]
    )
    assert all(controller.controller.engine.run_is_covered for controller in built)
    assert json.loads(json.dumps(result, allow_nan=False))["case_count"] == 30


def test_source_inventory_contains_executed_policy_builder_and_benchmark():
    inventory = benchmark.sources()
    assert "scripts.run_connectome_service" in inventory
    assert "scripts.benchmark_interaction_policy" in inventory
    assert "scripts.prepare_gametest_demo" in inventory
    assert any(name.endswith("interaction_program_policy.py") for name in inventory)
    assert any(name.endswith("interaction_program_controller.py") for name in inventory)


@pytest.mark.parametrize("threads", [0, 65, True, 1.0])
def test_thread_bounds_rejected_before_builder(monkeypatch, threads):
    monkeypatch.setattr(
        benchmark.service,
        "_build_program_selecting_controller",
        lambda args: pytest.fail("invalid thread bound reached builder"),
    )
    with pytest.raises(ValueError):
        benchmark.build_controller("faithful", threads)


def test_cli_refuses_existing_output_before_work(tmp_path, monkeypatch):
    path = tmp_path / "existing.json"
    path.write_text("preserve existing receipt")
    monkeypatch.setattr(benchmark.sys, "argv", ["benchmark", "--output", str(path)])
    monkeypatch.setattr(
        benchmark, "run_benchmark", lambda threads: pytest.fail("receipt overwrite")
    )
    with pytest.raises(SystemExit):
        benchmark.main()
    assert path.read_text() == "preserve existing receipt"


@pytest.mark.parametrize("case", benchmark.inputs.CASES)
def test_extended_schedule_holds_final_frame_without_refreshing_event_counters(case):
    last = benchmark.inputs.sensor_frame(case, 35)
    assert benchmark.sensor_frame(case, 35) == last
    assert benchmark.sensor_frame(case, 36) == last
    assert benchmark.sensor_frame(case, 99) == last
    for counter in (last.events.food, last.events.brush, last.events.attack_attempt):
        assert counter.last_game_tick in (None, 20)
    for tick in (-1, 100, True):
        with pytest.raises(ValueError):
            benchmark.sensor_frame(case, tick)


def test_extra_binding_material_is_copied_from_actual_components_without_graph_arrays():
    controller = wrapper()
    core = controller.controller
    material = benchmark.binding_material(controller)
    assert material["sensory_binding"] == core._sensory_binding.to_identity_material()
    assert material["motor_binding"] == core._motor_binding.to_identity_material()
    assert material["escape_binding"] == core.escape_binding.to_identity_material()
    assert material["view_sha256"] == core.engine.view.view_sha256
    assert material["view_provenance_sha256"] == core.engine.view.provenance.identity_sha256
    assert material["sign_overlay_identity_sha256"] == core.sign_overlay_identity_sha256
    assert "weights" not in material and "indices" not in material
    json.dumps(material, allow_nan=False)


@pytest.mark.parametrize("kind", ["benchmark", "light", "inputs"])
@pytest.mark.parametrize("change", [None, "before", "during", "origin"])
def test_public_diagnostics_verify_archive_before_and_after_small_engine_work(
    tmp_path, monkeypatch, kind, change
):
    from scripts import probe_light_response as light
    from scripts.prepare_gametest_demo import PreparationError
    from tests.controller.test_events_senses import controllers
    from tests.integration.test_prepare_gametest_demo import _write_release_fixture

    root = tmp_path / "archive"
    _write_release_fixture(root, {"LICENSE": b"test license", "scripts/entry.py": b"pass\n"})
    entry = root / "scripts/entry.py"
    inventory = {"fixture": {"path": str(entry), "sha256": sha256(entry.read_bytes()).hexdigest()}}
    module = {"benchmark": benchmark, "light": light, "inputs": benchmark.inputs}[kind]
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "code_sources" if kind == "light" else "sources", lambda: inventory)
    built = []

    def build(mode):
        if change == "before":
            pytest.fail("invalid archive reached expensive controller construction")
        result = controllers(mode)[0] if kind == "light" else wrapper(mode)
        if kind != "benchmark" and kind != "light":
            result = result.controller
        built.append(result)
        return result

    if kind == "benchmark":
        monkeypatch.setattr(module, "build_controller", lambda mode, threads: build(mode))
    else:
        monkeypatch.setattr(
            module.FullBrainController,
            "from_artifact",
            lambda **kwargs: build(kwargs["configuration"].mode),
        )
    original = module.measure_case

    def measure(*args):
        result = original(*args)
        if change == "during":
            entry.write_bytes(b"xxxx\n")
        if change == "origin":
            path = root / "RELEASE_MANIFEST.json"
            manifest = json.loads(path.read_bytes())
            manifest["source"]["commit"] = "c" * 40
            path.write_text(json.dumps(manifest))
        return result

    monkeypatch.setattr(module, "measure_case", measure)
    run = module.run_benchmark if kind == "benchmark" else module.run_probe
    if change == "before":
        entry.write_bytes(b"xxxx\n")
    if change is not None:
        with pytest.raises((PreparationError, RuntimeError), match=r"hash mismatch|origin changed"):
            run(1)
    else:
        result = run(1)
        origin = result["source_origin_at_start"]
        assert origin == result["source_origin_at_end"]
        assert origin["kind"] == "verified_source_archive"
        assert origin["git_head"] is None and origin["source_commit"] == "b" * 40
        assert origin["publisher_authenticity_verified"] is False
        assert (
            origin["release_manifest_sha256"]
            == sha256((root / "RELEASE_MANIFEST.json").read_bytes()).hexdigest()
        )
        if kind == "benchmark":
            assert result["git_head_at_start"] is result["git_head_at_end"] is None
            assert result["case_count"] == 30
        else:
            assert result["git_head"] is None
            assert len(result["lanes"]) == 3
    assert all(
        (item.controller if kind == "benchmark" else item).engine.run_is_covered for item in built
    )
