#!/usr/bin/env python3
"""Bounded offline V2 issued-policy benchmark; no Minecraft application is exercised.

All prescribed cases and every issued command are retained, including unchanged
or unexpected behavior. This does not establish six live interaction responses.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from neurocraft_controller.interaction_program_controller import (
    InteractionProgramSelectingController,
)
from neurocraft_protocol import Observation, ProgramId, SensorFrameV4

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import probe_interaction_inputs as inputs
from scripts import run_connectome_service as service
from scripts.prepare_gametest_demo import execution_source_origin
from scripts.probe_light_response import BASELINE_TICKS, RESPONSE_TICKS

ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS_PER_CASE = 100
UPDATES_PER_OBSERVATION = 5
LANES = ("faithful", "shuffled_weight", "no_input")
TIMING_SCOPE = (
    "unpaced wall time immediately around wrapper.step only, including actual core compute, "
    "policy and issued evidence construction; excludes observation construction, subsequent "
    "digest copying, serialization, startup and reset; not live, end-to-end, TPS or body timing"
)


def _threads(value: int) -> None:
    if type(value) is not int or not 1 <= value <= 64:
        raise ValueError("benchmark threads must be an integer in [1,64]")


def sources() -> dict[str, dict[str, str]]:
    result = inputs.sources()
    for module in (
        __name__,
        "scripts.run_connectome_service",
        "scripts.action_latency_telemetry",
        "scripts.program_decision_telemetry",
    ):
        filename = importlib.import_module(module).__file__
        if filename is None:
            raise RuntimeError("missing benchmark source")
        path = Path(filename).resolve()
        result[module] = {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}
    return result


def sensor_frame(case: str, tick: int) -> SensorFrameV4:
    """Hold the final prescribed input after tick35; never renew event timestamps."""
    if type(tick) is not int or not 0 <= tick < OBSERVATIONS_PER_CASE:
        raise ValueError("benchmark tick outside the fixed 100-observation schedule")
    return inputs.sensor_frame(case, min(tick, BASELINE_TICKS + RESPONSE_TICKS - 1))


def build_controller(mode: str, threads: int) -> InteractionProgramSelectingController:
    """Use the production builder, with the same explicit mode injection as lane setup."""
    _threads(threads)
    if mode not in LANES:
        raise ValueError("unknown benchmark control lane")
    args = service.build_parser().parse_args(
        [
            "--controller-profile",
            service.FULL_MCNS_INTERACTIONS_PROGRAM_V2_PROFILE,
            "--run-id",
            "interaction-policy-offline-benchmark",
            "--entity-id",
            "synthetic-fly",
            "--control-seed",
            "7",
        ]
    )
    # The standalone CLI accepts faithful/lesion; its comparison lane dispatcher
    # injects these internal control modes before calling this same builder.
    args.mode, args.kernel_threads = mode, threads
    controller = service._build_program_selecting_controller(args)
    if not isinstance(controller, InteractionProgramSelectingController):
        controller.controller.engine.end_run()
        raise RuntimeError("production builder did not return the V2 interaction wrapper")
    return controller


def timing_summary(samples: list[int]) -> dict[str, Any]:
    if not samples or any(type(value) is not int or value < 0 for value in samples):
        raise ValueError("timing summary requires nonnegative measured nanoseconds")
    ordered = sorted(samples)

    def percentile(fraction: float) -> float:
        # Nearest rank, explicitly declared; no discarded startup/slow observations.
        return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)] / 1e6

    return {
        "scope": TIMING_SCOPE,
        "samples": len(samples),
        "percentile_method": "nearest rank; all observations included",
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "max_ms": ordered[-1] / 1e6,
        "total_ms": sum(samples) / 1e6,
    }


def summarize_case(rows: list[dict[str, Any]]) -> dict[str, Any]:
    program_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    first_entries: dict[str, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []
    holds: list[dict[str, Any]] = []
    for row in rows:
        tick, command, named = (
            row["observation"]["game_tick"],
            row["program_action"],
            row["named_evidence"],
        )
        source, reason = row["selection_source"], row["selection_reason"]
        program = command["program"]
        program_counts[program] += 1
        source_counts["none" if source is None else source] += 1
        reason_counts[reason] += 1
        if reason == "fresh_eligible_response":
            entry = {
                "game_tick": tick,
                "program": program,
                "source": source,
                "command_seq": command["command_seq"],
                "source_score": named["interaction"]["scores"][source],
            }
            entries.append(entry)
            first_entries.setdefault(source, entry)
        # Segments count issued commands under one source/program, never body ticks.
        trigger_seq = None if row["trigger_command_seq"] is None else row["trigger_command_seq"]
        key = (source, program, trigger_seq)
        if not holds or key != (
            holds[-1]["source"],
            holds[-1]["program"],
            holds[-1]["trigger_command_seq"],
        ):
            holds.append(
                {
                    "source": source,
                    "program": program,
                    "trigger_command_seq": trigger_seq,
                    "first_observation_tick": tick,
                    "last_observation_tick": tick,
                    "issued_observations": 0,
                }
            )
        holds[-1]["last_observation_tick"] = tick
        holds[-1]["issued_observations"] += 1
    return {
        "observations": len(rows),
        "neural_updates": sum(row["neural_step_end"] - row["neural_step_start"] for row in rows),
        "program_counts": {program.value: program_counts[program.value] for program in ProgramId},
        "selected_source_counts": dict(source_counts),
        "selection_reason_counts": dict(reason_counts),
        "first_entry_by_source": first_entries,
        "entries": entries,
        "issued_segments": holds,
        "segment_scope": "contiguous issued observations, not actual Minecraft applications",
        "compute": timing_summary([row["wrapper_compute_ns"] for row in rows]),
    }


def measure_case(controller: InteractionProgramSelectingController, case: str) -> dict[str, Any]:
    if case not in inputs.CASES:
        raise ValueError("unknown prescribed interaction case")
    controller.reset_session(session_epoch=controller.identity.session_epoch + 1)
    core = controller.controller
    if core.engine.step_index != 0:
        raise RuntimeError("case reset did not clear neural step state")
    zero_digest = sha256(bytes(core.engine.view.node_count * 8)).hexdigest()
    identity = controller.identity.to_dict()
    policy_hash = controller.policy_hash
    rows = []
    for tick in range(OBSERVATIONS_PER_CASE):
        observation = Observation(
            controller.identity, tick, tick, tick * 50_000_000, sensor_frame(case, tick)
        )
        start = perf_counter_ns()
        command = controller.step(observation, command_seq=tick)
        compute_ns = perf_counter_ns() - start
        evidence, named = controller.last_evidence, controller.last_event_evidence
        if evidence is None or named is None:
            raise RuntimeError("missing actual wrapper evidence")
        inner = evidence.inner
        if (
            inner.neural_step_start != tick * UPDATES_PER_OBSERVATION
            or inner.neural_step_end != (tick + 1) * UPDATES_PER_OBSERVATION
            or command.neural_step_start != inner.neural_step_start
            or command.neural_step_end != inner.neural_step_end
            or command.policy_hash != policy_hash
        ):
            raise RuntimeError("issued command steps or policy identity changed unexpectedly")
        activity_digest = core.engine.current_state().activity_sha256
        if core.control_kind == "no_input" and (
            inner.executed_stimulation_sha256 != zero_digest
            or activity_digest != zero_digest
            or command.program is not ProgramId.CRUISE
            or named.decision.source is not None
            or named.trigger is not None
            or any(score.score != 0 for score in named.decision.scores.values())
        ):
            raise RuntimeError(
                "no-input complete sensory/activity zero or event-program gate failed"
            )
        rows.append(
            {
                "observation": observation.to_dict(),
                "program_action": command.to_dict(),
                "named_evidence": named.to_dict(),
                "selection_source": named.decision.source,
                "selection_reason": named.decision.reason,
                "trigger_command_seq": None
                if named.trigger is None
                else named.trigger.command.command_seq,
                "policy_state_after": controller.policy_state.to_dict(),
                "raw_projection_not_applied": evidence.raw_projection.to_dict(),
                "encoded_stimulation": dict(inner.encoded_stimulation),
                "requested_before_control": None
                if inner.encoder_stimulation_before_control is None
                else dict(inner.encoder_stimulation_before_control),
                "encoded_stimulation_sha256": inner.encoded_stimulation_sha256,
                # Faithful currently exposes no separate delivered-vector digest.
                "executed_stimulation_sha256": inner.executed_stimulation_sha256,
                "full_activity_sha256": activity_digest,
                "state_before_sha256": inner.state_before_sha256,
                "state_after_sha256": inner.state_after_sha256,
                "neural_step_start": inner.neural_step_start,
                "neural_step_end": inner.neural_step_end,
                "wrapper_compute_ns": compute_ns,
            }
        )
    if core.engine.step_index != OBSERVATIONS_PER_CASE * UPDATES_PER_OBSERVATION:
        raise RuntimeError("case denominator does not match actual engine updates")
    return {
        "case": case,
        "identity": identity,
        "policy_hash": policy_hash,
        "summary": summarize_case(rows),
        "rows": rows,
    }


def binding_material(controller: InteractionProgramSelectingController) -> dict[str, Any]:
    """Copy executed bindings and graph-view provenance without bulk runtime arrays."""
    core = controller.controller
    escape = core.escape_binding
    if escape is None:
        raise RuntimeError("missing actual escape binding")
    return {
        "sensory_binding": core._sensory_binding.to_identity_material(),
        "motor_binding": core._motor_binding.to_identity_material(),
        "escape_binding": escape.to_identity_material(),
        "view_sha256": core.engine.view.view_sha256,
        "view_provenance": asdict(core.engine.view.provenance),
        "view_provenance_sha256": core.engine.view.provenance.identity_sha256,
        "sign_overlay_identity_sha256": core.sign_overlay_identity_sha256,
    }


def run_benchmark(threads: int) -> dict[str, Any]:
    _threads(threads)
    code = sources()
    origin = execution_source_origin(ROOT, code)
    lanes = []
    shared_policy = None
    for mode in LANES:
        print(f"Benchmarking offline issued V2 policy: {mode}", flush=True)
        controller = build_controller(mode, threads)
        core = controller.controller
        try:
            declaration = controller.policy.to_identity_material()
            if shared_policy is None:
                shared_policy = declaration
            elif declaration != shared_policy:
                raise RuntimeError("control lanes must share the exact policy declaration")
            cases = {}
            for case in inputs.CASES:
                print(f"  case={case}, observations={OBSERVATIONS_PER_CASE}", flush=True)
                cases[case] = measure_case(controller, case)
            encoder, steering, escape, embodied = (
                core.event_encoder,
                core.steering_binding,
                core.escape_binding,
                core.embodied_senses,
            )
            if encoder is None or steering is None or escape is None or embodied is None:
                raise RuntimeError("missing executed source/readout bindings")
            lane_rows = [row for case in cases.values() for row in case["rows"]]
            lanes.append(
                {
                    "lane": mode,
                    "label": {
                        "faithful": "original wiring / Faithful Organism Mode",
                        "shuffled_weight": "shuffled weights / seed 7",
                        "no_input": "no input",
                    }[mode],
                    "identity_after_last_case": controller.identity.to_dict(),
                    "policy_hash": controller.policy_hash,
                    "policy_declaration": declaration,
                    **binding_material(controller),
                    "configuration": core.configuration.to_identity_material(),
                    "engine_configuration": asdict(core.engine.config),
                    "node_count": core.engine.view.node_count,
                    "edge_count": core.engine.view.edge_count,
                    "kernel_identity": core.engine.kernel_identity.material,
                    "control_identity": None
                    if core.control_identity is None
                    else core.control_identity.material,
                    "event_encoder": encoder.to_identity_material(),
                    "event_population_binding": encoder.populations.to_identity_material(),
                    "embodied_population_binding": embodied.populations.to_identity_material(),
                    "background_encoder": embodied.encoder.to_identity_material(),
                    "steering_binding": steering.to_identity_material(),
                    "steering_binding_sha256": steering.identity_sha256,
                    "escape_binding_sha256": escape.identity_sha256,
                    "zero_float64_vector_sha256": sha256(
                        bytes(core.engine.view.node_count * 8)
                    ).hexdigest(),
                    "case_count": len(cases),
                    "observations": len(lane_rows),
                    "neural_updates": sum(
                        case["summary"]["neural_updates"] for case in cases.values()
                    ),
                    "compute": timing_summary([row["wrapper_compute_ns"] for row in lane_rows]),
                    "cases": cases,
                }
            )
        finally:
            core.engine.end_run()
        del controller, core
    if sources() != code:
        raise RuntimeError("executed source inventory changed during benchmark")
    final_origin = execution_source_origin(ROOT, code)
    if final_origin != origin:
        raise RuntimeError("execution source origin changed during benchmark")
    all_rows = [row for lane in lanes for case in lane["cases"].values() for row in case["rows"]]
    return {
        "schema": "neurocraft.interaction_policy_benchmark.v2",
        "scope": "offline actual full-network core and production wrapper on prescribed inputs; "
        "issued policy commands only; no Minecraft body, application, "
        "live acceptance or biological claim",
        "applied_commands": None,
        "application_status": "unobserved; this benchmark does not connect to Minecraft",
        "git_head_at_start": origin["git_head"],
        "git_head_at_end": final_origin["git_head"],
        "source_origin_at_start": origin,
        "source_origin_at_end": final_origin,
        "source_files": code,
        "source_inventory_scope": "inputs.sources inventory plus this benchmark and service CLI "
        "with its telemetry scripts; all package Python, pyproject and lockfile included; "
        "per-lane native kernel material recorded; third-party runtime bytes not inventoried",
        "source_inventory_unchanged": True,
        "schedule": {
            "cases": list(inputs.CASES),
            "lanes": list(LANES),
            "observations_per_case": OBSERVATIONS_PER_CASE,
            "baseline_observations": BASELINE_TICKS,
            "onset_tick": BASELINE_TICKS,
            "extension": "observations 0..35 match the input diagnostic; observations 36..99 "
            "hold its final sensor frame, retaining original event timestamps20 while "
            "observation clocks and neural updates advance",
            "neural_updates_per_observation": UPDATES_PER_OBSERVATION,
            "controls": "same prescribed histories, reset per case, fixed shuffle seed 7; "
            "no body feedback; no expected outcome enforced",
        },
        "case_count": sum(lane["case_count"] for lane in lanes),
        "observations": len(all_rows),
        "neural_updates": sum(lane["neural_updates"] for lane in lanes),
        "same_policy_declaration_across_lanes": True,
        "compute": timing_summary([row["wrapper_compute_ns"] for row in all_rows]),
        "lanes": lanes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.threads <= 64:
        parser.error("new output and threads in [1,64] required")
    result = run_benchmark(args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256(args.output.read_bytes()).hexdigest(),
                "observations": result["observations"],
                "neural_updates": result["neural_updates"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
