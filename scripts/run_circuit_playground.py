"""Offline Circuit Playground: fixed-input fingerprints and internal silencing.

Runs the production V4 interaction encoder and selection policy. Issued programs
are not Minecraft applications. No production model, binding or threshold changes.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from neurocraft_connectome.full_runtime import load_full_runtime_artifact
from neurocraft_controller.embodied_senses import _verified_annotation_table, _verified_graph_array
from neurocraft_controller.full_brain_controller import FULL_MCNS_ARTIFACT_MANIFEST_SHA256
from neurocraft_controller.interaction_program_controller import (
    InteractionProgramSelectingController,
)
from neurocraft_neural.full_engine import ArrayNativeEngine
from neurocraft_neural.full_records import ArrayStepResult
from neurocraft_neural.full_signs import compute_full_sign_overlay
from neurocraft_neural.model import ZERO_SHA256, ablation_mask_digest
from neurocraft_protocol import Observation

from scripts import benchmark_interaction_policy as benchmark
from scripts import run_connectome_service as service
from scripts.measure_escape_arc import (
    DEFAULT_ANNOTATIONS,
    DEFAULT_ARTIFACT_DIRECTORY,
    DEFAULT_BODY_ANNOTATIONS,
    select_dnp01_pair,
)
from scripts.prepare_gametest_demo import execution_source_origin

ROOT = Path(__file__).resolve().parents[1]
CASES = ("bright_sham", "player_approach", "mob_approach", "attack", "brush", "food", "light_off")
PILOT_CASES = ("bright_sham", "player_approach", "slow_approach", "two_approaches")
TICKS = 100


def digest(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def save(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def frame(case: str, tick: int) -> Any:
    if case not in {"slow_approach", "two_approaches"}:
        return benchmark.sensor_frame(case, tick)
    base = benchmark.sensor_frame("bright_sham", tick)
    front = 6.0
    if case == "slow_approach" and 20 <= tick < 60:
        front = max(1.2, 6.0 - 0.16 * (tick - 19))
    elif case == "two_approaches":
        if 20 <= tick < 40:
            front = max(1.2, 6.0 - 0.4 * (tick - 19))
        elif 50 <= tick < 70:
            front = max(2.4, 6.0 - 0.3 * (tick - 49))
    return replace(
        base,
        ray_distances_m=(front, 6.0, 6.0, 6.0, 6.0, 6.0),
        ray_hit_kinds=(2 if front < 6 else 0, 0, 0, 0, 0, 0),
        ray_actor_kinds=(1 if front < 6 else 0, 0, 0, 0, 0, 0),
    )


class RecordingEngine(ArrayNativeEngine):
    """Passive arithmetic recorder; verifies actual input against the frozen run."""

    def configure_recording(self, selected: dict[str, list[int]]) -> None:
        self.selected = selected
        self.records: list[dict[str, Any]] = []
        self.expected_input: str | None = None
        self.driven: set[int] = set()

    def step(self, **kwargs: Any) -> ArrayStepResult:
        sensory = kwargs.get("sensory")
        actual = (
            ZERO_SHA256
            if sensory is None
            else sha256(np.asarray(sensory, dtype=">f8").tobytes()).hexdigest()
        )
        if self.expected_input is not None and actual != self.expected_input:
            raise RuntimeError("executed input differs from frozen original trajectory")
        if sensory is not None:
            self.driven.update(int(i) for i in np.flatnonzero(sensory))
        result = super().step(**kwargs)
        activity = self.activity()
        self.records.append(
            {
                **asdict(result.record),
                "sensory_provenance_sha256": kwargs.get("sensory_provenance_sha256", ZERO_SHA256),
                "direct_provenance_sha256": kwargs.get("direct_provenance_sha256", ZERO_SHA256),
                "activity_sha256": result.state.activity_sha256,
                "selected_activity": {
                    name: [float(activity[i]) for i in ids] for name, ids in self.selected.items()
                },
            }
        )
        return result


def instrument(
    wrapper: InteractionProgramSelectingController, selected: dict[str, list[int]]
) -> RecordingEngine:
    old = wrapper.controller.engine
    old.end_run()
    engine = RecordingEngine(
        old.view, config=old.config, kernel=old.kernel, verification_policy=old.policy
    )
    engine.configure_recording(selected)
    if engine.dynamics_identity_sha256 != old.dynamics_identity_sha256:
        raise RuntimeError("instrumentation changed dynamics identity")
    wrapper.controller.engine = engine
    return engine


def build(mode: str, threads: int, seed: int = 7) -> InteractionProgramSelectingController:
    args = service.build_parser().parse_args(
        [
            "--controller-profile",
            service.FULL_MCNS_INTERACTIONS_PROGRAM_V2_PROFILE,
            "--run-id",
            "circuit-playground-offline",
            "--entity-id",
            "synthetic-fly",
            "--control-seed",
            str(seed),
        ]
    )
    args.mode, args.kernel_threads = mode, threads
    wrapper = service._build_program_selecting_controller(args)
    if not isinstance(wrapper, InteractionProgramSelectingController):
        raise RuntimeError("wrong production profile")
    return wrapper


def member_indices(material: Any) -> set[int]:
    """Conservative union of all explicitly bound inputs and readouts."""
    found: set[int] = set()
    if isinstance(material, dict):
        for key, value in material.items():
            if key in {"members", "members_body_dense_side", "targets"}:
                found.update(int(row[1]) for row in value)
            else:
                found.update(member_indices(value))
    elif isinstance(material, list):
        for item in material:
            found.update(member_indices(item))
    return found


def metadata(wrapper: InteractionProgramSelectingController) -> dict[str, Any]:
    core = wrapper.controller
    if core.event_encoder is None or core.embodied_senses is None or core.steering_binding is None:
        raise RuntimeError("missing required production bindings")
    return {
        **benchmark.binding_material(wrapper),
        "base_controller_identity": wrapper.identity.to_dict(),
        "policy_hash": wrapper.policy_hash,
        "policy": wrapper.policy.to_identity_material(),
        "engine_config": asdict(core.engine.config),
        "kernel": core.engine.kernel_identity.material,
        "control": None if core.control_identity is None else core.control_identity.material,
        "event_encoder": core.event_encoder.to_identity_material(),
        "event_populations": core.event_encoder.populations.to_identity_material(),
        "embodied_populations": core.embodied_senses.populations.to_identity_material(),
        "steering": core.steering_binding.to_identity_material(),
    }


def measure(
    wrapper: InteractionProgramSelectingController,
    engine: RecordingEngine,
    case: str,
    condition: dict[str, Any],
    directory: Path,
    reference: list[dict[str, Any]] | None = None,
    collect_activity: bool = False,
) -> tuple[list[dict[str, Any]], list[Any]]:
    wrapper.reset_session(session_epoch=wrapper.identity.session_epoch + 1)
    mask = condition["indices"]
    engine.selected["silenced"] = list(mask)
    engine.set_ablated(mask)  # Even empty sham operation; reset always clears masks.
    actual_mask = np.flatnonzero(engine.snapshot().ablation_mask).tolist()
    if actual_mask != sorted(mask):
        raise RuntimeError("mask application failed")
    descriptor = {
        **condition,
        "case": case,
        "base_identity": wrapper.identity.to_dict(),
        "base_identity_scope": "unmodified controller identity; condition hash binds intervention",
        "mask_sha256": ablation_mask_digest(engine.snapshot().ablation_mask),
    }
    condition_hash = digest(descriptor)
    save(
        directory / f"{condition['name']}-{case}-condition.json",
        {"condition": descriptor, "sha256": condition_hash},
    )
    rows, activities = [], []
    stream_path = directory / f"{condition['name']}-{case}.ndjson"
    try:
        with stream_path.open("x") as stream:
            for tick in range(TICKS):
                engine.records.clear()
                engine.expected_input = (
                    None
                    if reference is None
                    else reference[tick]["neural_steps"][0]["neural"]["encoded_stimulation_sha256"]
                )
                observation = Observation(
                    wrapper.identity, tick, tick, tick * 50_000_000, frame(case, tick)
                )
                command = wrapper.step(observation, command_seq=tick)
                named, evidence = wrapper.last_event_evidence, wrapper.last_evidence
                if named is None or evidence is None or len(engine.records) != 5:
                    raise RuntimeError("missing production evidence or unexpected update count")
                if any(
                    r["neural"]["ablation_mask_sha256"] != descriptor["mask_sha256"]
                    for r in engine.records
                ):
                    raise RuntimeError("executed mask differs from condition")
                if any(
                    r["neural"]["direct_stimulation_sha256"] != ZERO_SHA256 for r in engine.records
                ):
                    raise RuntimeError("unexpected direct stimulation")
                activity = engine.activity()
                if mask and (
                    np.any(activity[mask] != 0)
                    or any(
                        any(value != 0 for value in r["selected_activity"]["silenced"])
                        for r in engine.records
                    )
                ):
                    raise RuntimeError("silenced cells have nonzero activity")
                if condition["mode"] == "no_input":
                    zero = sha256(bytes(engine.view.node_count * 8)).hexdigest()
                    if any(
                        r["neural"]["encoded_stimulation_sha256"] != zero
                        or r["activity_sha256"] != zero
                        for r in engine.records
                    ):
                        raise RuntimeError("no-input state/input is nonzero")
                    if command.program.value != "CRUISE" or named.decision.source is not None:
                        raise RuntimeError("no-input issued an event-selected program")
                row = {
                    "tick": tick,
                    "condition_sha256": condition_hash,
                    "observation": observation.to_dict(),
                    "issued_program": command.to_dict(),
                    "applied_in_minecraft": False,
                    "named_evidence": named.to_dict(),
                    "decision": {
                        "source": named.decision.source,
                        "reason": named.decision.reason,
                        "scores": {k: v.to_dict() for k, v in named.decision.scores.items()},
                    },
                    "rectified_jump": evidence.readout.ttmn_jump,
                    "neural_steps": list(engine.records),
                }
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
                if collect_activity:
                    activities.append(activity)
        engine.end_run()
        if not engine.run_is_covered:
            raise RuntimeError("uncovered neural execution")
        save(
            directory / f"{condition['name']}-{case}-coverage.json",
            {
                "run_is_covered": True,
                "verified_through_step": engine.verified_through_step,
                "condition_sha256": condition_hash,
                "rows": len(rows),
                "trace_file_sha256": sha256(stream_path.read_bytes()).hexdigest(),
            },
        )
    finally:
        engine.expected_input = None
    print(f"  {condition['name']}: {case} ({len(rows)} observations)", flush=True)
    return rows, activities


def semantic_command(command: dict[str, Any]) -> dict[str, Any]:
    """Retain every issued action field except reset identity and absolute neural offset."""
    return {
        **{
            key: value
            for key, value in command.items()
            if key not in {"identity", "neural_step_start", "neural_step_end"}
        },
        "neural_step_span": command["neural_step_end"] - command["neural_step_start"],
    }


def semantic_rows(rows: list[dict[str, Any]]) -> list[Any]:
    return [
        {
            "states": [r["activity_sha256"] for r in row["neural_steps"]],
            "inputs": [r["neural"]["encoded_stimulation_sha256"] for r in row["neural_steps"]],
            "command": semantic_command(row["issued_program"]),
            "decision": row["decision"],
            "jump": row["rectified_jump"],
        }
        for row in rows
    ]


def response(rows: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray([r["neural_steps"][-1]["selected_activity"]["ttmn"] for r in rows])
    background = np.asarray([r["neural_steps"][-1]["selected_activity"]["ttmn"] for r in baseline])
    evoked = values - background
    mean = evoked.mean(axis=1)
    return {
        "ttmn_signed_evoked_lr": evoked.tolist(),
        "ttmn_signed_evoked_mean": mean.tolist(),
        "peak_abs_mean": float(np.abs(mean).max()),
        "integral_abs_mean_tick": float(np.abs(mean).sum()),
        "peak_abs_lr": np.abs(evoked).max(axis=0).tolist(),
        "issued_program_counts": {
            program: sum(row["issued_program"]["program"] == program for row in rows)
            for program in sorted({row["issued_program"]["program"] for row in rows})
        },
        "selected_sources": sorted(
            {row["decision"]["source"] for row in rows if row["decision"]["source"] is not None}
        ),
    }


def run(output: Path, threads: int, seeds: list[int], comparisons: int) -> None:
    from scripts.circuit_playground_masks import select_comparison_pairs

    output.mkdir(parents=True, exist_ok=False)
    sources = benchmark.sources()
    for filename in ("run_circuit_playground.py", "circuit_playground_masks.py"):
        path = ROOT / "scripts" / filename
        sources[filename] = {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}
    plan = {
        "schema": "circuit-playground-offline-v1",
        "sources": sources,
        "origin": execution_source_origin(ROOT, sources),
        "seeds": seeds,
        "comparison_count": comparisons,
        "cases": {
            case: [frame(case, tick).to_dict() for tick in range(TICKS)]
            for case in dict.fromkeys((*CASES, *PILOT_CASES))
        },
        "matching_seed": 20261024,
        "matching_scope": "size/side/sign matched, activity/strength ranked; no activity calipers",
        "threads": threads,
        "scope": (
            "prescribed offline inputs and issued scripted programs; "
            "no Minecraft applications or biological necessity"
        ),
        "metrics": (
            "same-mask no-event subtraction; raw signed TTMn L/R and mean; "
            "rectified production readout separately"
        ),
        "onset": (
            "first tick >=20 exceeding 10% of intact peak absolute evoked mean; "
            "shared threshold; null if absent"
        ),
        "ratio_floor": 1e-12,
        "timing_resolution": "5 neural steps / observation; no four-step latency claim",
    }
    save(output / "plan.json", plan)
    try:
        _execute(output, plan, threads, seeds, comparisons, select_comparison_pairs)
    except BaseException as exc:
        save(
            output / "failure.json",
            {"type": type(exc).__name__, "message": str(exc), "partial_evidence_retained": True},
        )
        raise


def _execute(
    output: Path,
    plan: dict[str, Any],
    threads: int,
    seeds: list[int],
    comparisons: int,
    selector: Any,
) -> None:
    with load_full_runtime_artifact(
        DEFAULT_ARTIFACT_DIRECTORY, expected_manifest_sha256=FULL_MCNS_ARTIFACT_MANIFEST_SHA256
    ) as graph:
        pair = select_dnp01_pair(graph, DEFAULT_BODY_ANNOTATIONS)
        table, _ = _verified_annotation_table(graph, DEFAULT_BODY_ANNOTATIONS)
        taken = table.take(
            _verified_graph_array(graph, "annotation_source_rows", "<u8").astype(np.int64)
        )
        bodies = _verified_graph_array(graph, "body_ids", "<u8").copy()
        types = np.asarray(taken.column("type").to_pylist(), dtype=object)
        sides = np.asarray(taken.column("somaSide").to_pylist(), dtype=object)
        signs = compute_full_sign_overlay(graph, DEFAULT_ANNOTATIONS).node_signs.copy()
        raw = _verified_graph_array(graph, "edge_raw_counts", "<u8")
        post = _verified_graph_array(graph, "edge_post_indices", "<u4")
        incoming = np.bincount(post, weights=raw, minlength=len(bodies))
        del raw, post
    target = [entry[1] for entry in sorted(pair, key=lambda item: item[2])]
    original = build("faithful", threads)
    core = original.controller
    meta = metadata(original)
    save(output / "original-bindings.json", meta)
    if core.escape_binding is None:
        raise RuntimeError("missing TTMn binding")
    ttmn = sorted(
        [m.dense_index for m in core.escape_binding.jump_pool.members], key=lambda i: str(sides[i])
    )
    selected = {"dnp01": target, "ttmn": ttmn}
    save(
        output / "populations.json",
        {
            name: [{"body_id": int(bodies[i]), "index": i, "side": str(sides[i])} for i in ids]
            for name, ids in selected.items()
        },
    )
    engine = instrument(original, selected)
    outgoing = np.bincount(
        engine.view.indices, weights=np.abs(engine.view.weights), minlength=len(bodies)
    )
    intact: dict[str, list[dict[str, Any]]] = {}
    discovery: dict[str, Any] = {}
    base = {"name": "original", "mode": "faithful", "indices": [], "body_ids": []}
    for case in dict.fromkeys((*CASES, *PILOT_CASES)):
        intact[case], activity = measure(
            original,
            engine,
            case,
            base,
            output,
            collect_activity=case in {"bright_sham", "player_approach"},
        )
        if activity:
            discovery[case] = np.asarray(activity)
    delta = np.abs(discovery["player_approach"] - discovery["bright_sham"])
    features = np.column_stack((delta.max(axis=0), delta.sum(axis=0), incoming, outgoing))
    excluded = member_indices(meta) | engine.driven
    if set(target) & excluded:
        raise RuntimeError("DNp01 overlaps a driven or decoded population")
    pairs = selector(
        bodies, types, sides, signs, excluded, features, target, count=comparisons, seed=20261024
    )
    save(
        output / "frozen-comparison-masks.json",
        {
            "pairs": pairs,
            "feature_names": [
                "evoked_peak_abs",
                "evoked_integral_abs_tick",
                "incoming_raw_count",
                "outgoing_normalized_abs_weight",
            ],
            "features_sha256": sha256(features.astype("<f8").tobytes()).hexdigest(),
            "excluded_indices": sorted(excluded),
            "plan_sha256": digest(plan),
        },
    )
    del discovery, delta
    pilot: dict[str, Any] = {
        "original": {
            case: response(intact[case], intact["bright_sham"]) for case in PILOT_CASES[1:]
        }
    }
    conditions: list[dict[str, Any]] = [
        {**base, "name": "sham"},
        {
            **base,
            "name": "dnp01_silenced",
            "indices": target,
            "body_ids": [int(bodies[i]) for i in target],
        },
    ]
    conditions += [
        {**base, "name": f"comparison_{n:02d}", "indices": p["indices"], "body_ids": p["body_ids"]}
        for n, p in enumerate(pairs)
    ]
    conditions += [{**base, "name": "restored"}]
    restoration = {}
    for condition in conditions:
        trials = {}
        for case in PILOT_CASES:
            trials[case], _ = measure(original, engine, case, condition, output, intact[case])
        pilot[condition["name"]] = {
            case: response(trials[case], trials["bright_sham"]) for case in PILOT_CASES[1:]
        }
        if condition["name"] in {"sham", "restored"}:
            restoration[condition["name"]] = all(
                semantic_rows(trials[c]) == semantic_rows(intact[c]) for c in PILOT_CASES
            )
            if not restoration[condition["name"]]:
                raise RuntimeError("sham/restoration failed exact numerical and semantic replay")
    matrix = {
        "original": {case: response(intact[case], intact["bright_sham"]) for case in CASES[1:]}
    }
    policy = meta["policy"]
    del original, core, engine
    gc.collect()
    for mode, seed in [("shuffled_weight", seed) for seed in seeds] + [("no_input", 7)]:
        wrapper = build(mode, threads, seed)
        engine = instrument(wrapper, selected)
        name = f"shuffle_{seed}" if mode == "shuffled_weight" else "no_input"
        meta = metadata(wrapper)
        if meta["policy"] != policy:
            raise RuntimeError("policy declaration changed across conditions")
        save(output / f"{name}-bindings.json", meta)
        condition = {"name": name, "mode": mode, "seed": seed, "indices": [], "body_ids": []}
        trials = {}
        for case in dict.fromkeys((*CASES, *PILOT_CASES)):
            trials[case], _ = measure(
                wrapper,
                engine,
                case,
                condition,
                output,
                None if mode == "no_input" else intact[case],
            )
        matrix[name] = {case: response(trials[case], trials["bright_sham"]) for case in CASES[1:]}
        pilot[name] = {
            case: response(trials[case], trials["bright_sham"]) for case in PILOT_CASES[1:]
        }
        del wrapper, engine
        gc.collect()
    for case in PILOT_CASES[1:]:
        peak = pilot["original"][case]["peak_abs_mean"]
        threshold = 0.1 * peak
        for conditions_results in pilot.values():
            result = conditions_results[case]
            result["peak_difference_from_original"] = result["peak_abs_mean"] - peak
            result["peak_ratio_to_original"] = (
                result["peak_abs_mean"] / peak if peak > 1e-12 else None
            )
            result["onset_tick"] = (
                next(
                    (
                        i
                        for i, value in enumerate(result["ttmn_signed_evoked_mean"])
                        if i >= 20 and abs(value) > threshold
                    ),
                    None,
                )
                if peak > 1e-12
                else None
            )
    for entry in plan["sources"].values():
        if sha256(Path(entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError("executed source inventory changed during measurement")
    save(
        output / "summary.json",
        {
            "plan_sha256": digest(plan),
            "pilot": pilot,
            "fingerprints": matrix,
            "restoration_exact": restoration,
            "status": "complete",
            "claim_scope": plan["scope"],
            "files": {
                p.name: sha256(p.read_bytes()).hexdigest()
                for p in sorted(output.iterdir())
                if p.is_file()
            },
        },
    )
    print(f"Complete: {output / 'summary.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23])
    parser.add_argument("--comparison-pairs", type=int, default=10)
    args = parser.parse_args()
    if (
        args.threads <= 0
        or len(set(args.seeds)) != len(args.seeds)
        or any(s < 0 for s in args.seeds)
    ):
        parser.error("positive threads and unique nonnegative seeds required")
    if not 1 <= args.comparison_pairs <= 10:
        parser.error("comparison-pairs must be between 1 and 10")
    run(args.output, args.threads, args.seeds, args.comparison_pairs)


if __name__ == "__main__":
    main()
