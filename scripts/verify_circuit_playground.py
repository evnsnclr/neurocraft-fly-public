"""Independently check retained offline playground receipts without rerunning a model.

Verifies artifact hashes, executed-record hash chains, condition/input/mask
bindings, replay semantics and numerical reducers. This is internal consistency
verification, not independent acquisition, neural simulation or Minecraft play.
Current source files are checked only with --check-current-sources; historical
receipts remain inspectable after source changes. Existing files are never edited.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

ZERO = "0" * 64
PILOT = ("player_approach", "slow_approach", "two_approaches")
FINGERPRINTS = ("player_approach", "mob_approach", "attack", "brush", "food", "light_off")


class EvidenceVerificationError(ValueError):
    """A retained evidence artifact fails a necessary consistency check."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceVerificationError(message)


def _digest(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _hash(value: Any) -> str:
    _require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        "invalid SHA-256 value",
    )
    return str(value)


def _invalid_constant(value: str) -> Any:
    raise EvidenceVerificationError(f"nonfinite JSON value: {value}")


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(), parse_constant=_invalid_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceVerificationError(f"cannot read {path.name}: {exc}") from exc


def _close(actual: Any, expected: Any, context: str) -> None:
    """Reducers use independent Python arithmetic; tolerate only rounding differences."""
    if isinstance(expected, dict):
        _require(
            isinstance(actual, dict) and set(actual) == set(expected), f"{context}: keys differ"
        )
        for key, value in expected.items():
            _close(actual[key], value, f"{context}.{key}")
    elif isinstance(expected, list):
        _require(
            isinstance(actual, list) and len(actual) == len(expected), f"{context}: length differs"
        )
        for index, value in enumerate(expected):
            _close(actual[index], value, f"{context}[{index}]")
    elif isinstance(expected, float):
        _require(
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15),
            f"{context}: numerical result differs",
        )
    else:
        _require(
            actual == expected and type(actual) is type(expected), f"{context}: result differs"
        )


def _semantic_command(command: dict[str, Any]) -> dict[str, Any]:
    # Deliberately independent of the runner's comparator. All other fields,
    # including modulation, parameters, readouts and action timing, are retained.
    result = {
        key: value
        for key, value in command.items()
        if key not in {"identity", "neural_step_start", "neural_step_end"}
    }
    result["neural_step_span"] = command["neural_step_end"] - command["neural_step_start"]
    return result


def _execution_material(record: dict[str, Any]) -> dict[str, Any]:
    neural = record["neural"]
    keys = (
        "ablation_mask_sha256",
        "delay_after_sha256",
        "delay_before_sha256",
        "direct_stimulation_sha256",
        "encoded_stimulation_sha256",
        "state_after_sha256",
        "state_before_sha256",
        "step_after",
        "step_before",
    )
    return {
        **{key: neural[key] for key in keys},
        "direct_stimulation_provenance_sha256": record["direct_provenance_sha256"],
        "encoded_stimulation_provenance_sha256": record["sensory_provenance_sha256"],
        **{
            key: record[key]
            for key in (
                "verification_policy_sha256",
                "verified_through_step",
                "view_digest_sha256",
                "view_validation_sha256",
            )
        },
    }


@dataclass
class _Trace:
    condition: dict[str, Any]
    ttmn: list[list[float]]
    inputs: list[str]
    semantics: list[dict[str, Any]]
    programs: dict[str, int]
    sources: dict[str, int]


def _verify_trace(
    directory: Path,
    condition: dict[str, Any],
    condition_hash: str,
    frames: list[Any],
    node_count: int,
    view_hash: str,
    policy_hash: str,
) -> _Trace:
    stem = f"{condition['name']}-{condition['case']}"
    path = directory / f"{stem}.ndjson"
    coverage = _read(directory / f"{stem}-coverage.json")
    _require(coverage["condition_sha256"] == condition_hash, f"{stem}: coverage condition mismatch")
    _require(
        coverage["trace_file_sha256"] == sha256(path.read_bytes()).hexdigest(),
        f"{stem}: coverage trace file hash mismatch",
    )
    mask = bytearray(node_count)
    indices = condition["indices"]
    _require(
        len(indices) == len(set(indices))
        and all(type(i) is int and 0 <= i < node_count for i in indices),
        f"{stem}: invalid mask indices",
    )
    for index in indices:
        mask[index] = 1
    mask_hash = sha256(mask).hexdigest()
    _require(condition["mask_sha256"] == mask_hash, f"{stem}: descriptor mask hash mismatch")
    zero_activity = sha256(bytes(node_count * 8)).hexdigest()
    inputs, semantics, ttmn = [], [], []
    programs: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    previous_trace, previous_state, previous_delay = ZERO, None, None
    last_verified = 0
    issued: dict[int, dict[str, Any]] = {}
    with path.open() as stream:
        for tick, line in enumerate(stream):
            row = json.loads(line, parse_constant=_invalid_constant)
            _require(tick < len(frames) and row["tick"] == tick, f"{stem}: row count/tick mismatch")
            _require(row["condition_sha256"] == condition_hash, f"{stem}: row condition mismatch")
            _require(
                row["applied_in_minecraft"] is False, f"{stem}: invented Minecraft application"
            )
            observation, command, named = (
                row["observation"],
                row["issued_program"],
                row["named_evidence"],
            )
            _require(
                observation["sensors"] == frames[tick], f"{stem}: observation differs from plan"
            )
            _require(
                observation["identity"] == command["identity"] == condition["base_identity"],
                f"{stem}: base identity mismatch",
            )
            _require(
                observation["game_tick"] == command["observation_tick"] == tick
                and observation["observation_seq"] == command["observation_seq"] == tick
                and command["command_seq"] == tick,
                f"{stem}: command observation mismatch",
            )
            _require(command["policy_hash"] == policy_hash, f"{stem}: policy binding mismatch")
            _require(
                named["command_sha256"] == _digest(command), f"{stem}: named command hash mismatch"
            )
            for key in command.keys() & named.keys():
                _require(
                    command[key] == named[key], f"{stem}: named command semantics mismatch: {key}"
                )
            _require(
                named["reason"] == row["decision"]["reason"]
                and named["interaction"]["source"] == row["decision"]["source"],
                f"{stem}: named decision mismatch",
            )
            for source, score in row["decision"]["scores"].items():
                _require(
                    all(
                        named["interaction"]["scores"][source][key] == value
                        for key, value in score.items()
                    ),
                    f"{stem}: named source score mismatch",
                )
            issued[tick] = command
            trigger = named.get("issued_trigger")
            if trigger is not None:
                onset = trigger["issued_command"]
                _require(
                    issued.get(onset["command_seq"]) == onset
                    and trigger["command_sha256"] == _digest(onset),
                    f"{stem}: onset does not bind an actual issued command",
                )
            records = row["neural_steps"]
            _require(len(records) == 5, f"{stem}: expected five neural records per observation")
            step_inputs, step_activity = [], []
            for offset, record in enumerate(records):
                neural = record["neural"]
                step = tick * 5 + offset
                _require(
                    neural["step_before"] == step and neural["step_after"] == step + 1,
                    f"{stem}: neural step sequence mismatch",
                )
                _require(
                    neural["ablation_mask_sha256"] == mask_hash, f"{stem}: executed mask mismatch"
                )
                _require(
                    neural["direct_stimulation_sha256"] == ZERO,
                    f"{stem}: direct stimulation present",
                )
                _require(record["view_digest_sha256"] == view_hash, f"{stem}: graph view mismatch")
                verified = record["verified_through_step"]
                _require(
                    type(verified) is int and last_verified <= verified <= step + 1,
                    f"{stem}: invalid verification coverage",
                )
                last_verified = verified
                _require(
                    previous_state is None or neural["state_before_sha256"] == previous_state,
                    f"{stem}: state chain mismatch",
                )
                _require(
                    previous_delay is None or neural["delay_before_sha256"] == previous_delay,
                    f"{stem}: delay chain mismatch",
                )
                _require(
                    neural["execution_sha256"] == _digest(_execution_material(record)),
                    f"{stem}: execution digest mismatch",
                )
                _require(
                    neural["previous_trace_sha256"] == previous_trace,
                    f"{stem}: previous trace mismatch",
                )
                trace = sha256(
                    bytes.fromhex(previous_trace) + bytes.fromhex(_hash(neural["execution_sha256"]))
                ).hexdigest()
                _require(neural["trace_sha256"] == trace, f"{stem}: trace chain mismatch")
                previous_trace, previous_state, previous_delay = (
                    trace,
                    neural["state_after_sha256"],
                    neural["delay_after_sha256"],
                )
                selected = record["selected_activity"]
                _require(
                    "silenced" in selected
                    and len(selected["silenced"]) == len(indices)
                    and all(value == 0 for value in selected["silenced"]),
                    f"{stem}: silenced-cell activities missing or nonzero",
                )
                for values in selected.values():
                    _require(
                        all(
                            isinstance(v, (int, float))
                            and not isinstance(v, bool)
                            and math.isfinite(v)
                            and -1 <= v <= 1
                            for v in values
                        ),
                        f"{stem}: invalid selected activity",
                    )
                if condition["name"] == "dnp01_silenced":
                    _require(
                        all(value == 0 for value in selected["dnp01"]), f"{stem}: DNp01 is nonzero"
                    )
                sensory = _hash(neural["encoded_stimulation_sha256"])
                if condition["mode"] == "no_input":
                    _require(
                        sensory == record["activity_sha256"] == zero_activity
                        and all(value == 0 for values in selected.values() for value in values),
                        f"{stem}: no-input state/input is nonzero",
                    )
                step_inputs.append(sensory)
                step_activity.append(_hash(record["activity_sha256"]))
            _require(
                command["neural_step_end"] - command["neural_step_start"] == 5
                and command["state_digest"] == records[-1]["neural"]["state_after_sha256"],
                f"{stem}: command neural envelope mismatch",
            )
            _require(
                row["rectified_jump"] == command["readout"]["ttmn_jump"],
                f"{stem}: rectified readout mismatch",
            )
            if condition["mode"] == "no_input":
                _require(
                    command["program"] == "CRUISE" and row["decision"]["source"] is None,
                    f"{stem}: no-input selected an event program",
                )
            values = records[-1]["selected_activity"]["ttmn"]
            _require(len(values) == 2, f"{stem}: bilateral TTMn values missing")
            ttmn.append(values)
            inputs.extend(step_inputs)
            semantics.append(
                {
                    "states": step_activity,
                    "inputs": step_inputs,
                    "command": _semantic_command(command),
                    "decision": row["decision"],
                    "jump": row["rectified_jump"],
                }
            )
            programs[command["program"]] += 1
            if row["decision"]["source"] is not None:
                sources[row["decision"]["source"]] += 1
    _require(len(ttmn) == len(frames) == coverage["rows"], f"{stem}: coverage row count mismatch")
    _require(
        coverage["run_is_covered"] is True and coverage["verified_through_step"] == len(frames) * 5,
        f"{stem}: final coverage incomplete",
    )
    return _Trace(condition, ttmn, inputs, semantics, dict(sorted(programs.items())), dict(sources))


def _response(trace: _Trace, baseline: _Trace) -> dict[str, Any]:
    evoked = [
        [left - base_left, right - base_right]
        for (left, right), (base_left, base_right) in zip(trace.ttmn, baseline.ttmn, strict=True)
    ]
    means = [(left + right) / 2 for left, right in evoked]
    return {
        "ttmn_signed_evoked_lr": evoked,
        "ttmn_signed_evoked_mean": means,
        "peak_abs_mean": max(abs(value) for value in means),
        "integral_abs_mean_tick": math.fsum(abs(value) for value in means),
        "peak_abs_lr": [max(abs(row[side]) for row in evoked) for side in range(2)],
        "issued_program_counts": trace.programs,
        "selected_sources": sorted(trace.sources),
    }


def _verify_control_binding(
    condition: dict[str, Any],
    metadata: dict[str, Any],
    original: dict[str, Any],
    seeds: list[int],
) -> None:
    """Bind a lane's descriptive seed/kind to its actual view and run identity."""
    name = condition["name"]
    base = metadata["base_controller_identity"]
    intact_base = original["base_controller_identity"]
    artifact = intact_base["graph_hash"]
    _require(
        base["graph_hash"] == metadata["sensory_binding"]["graph_manifest_sha256"] == artifact,
        f"{name}: source artifact binding differs",
    )
    _require(
        {key: value for key, value in condition["base_identity"].items() if key != "session_epoch"}
        == {key: value for key, value in base.items() if key != "session_epoch"},
        f"{name}: condition/base metadata identity differs",
    )
    provenance, intact_provenance = metadata["view_provenance"], original["view_provenance"]
    _require(
        metadata["view_provenance_sha256"] == _digest(provenance),
        f"{name}: view provenance hash differs",
    )
    for key in (
        "graph_identity_sha256",
        "source_manifest_sha256",
        "sign_overlay_sha256",
        "delay_overlay_sha256",
        "topology_overlay_sha256",
    ):
        _require(
            provenance[key] == intact_provenance[key], f"{name}: view source/sign/topology differs"
        )
    control = metadata["control"]
    if condition["mode"] == "faithful":
        _require(
            control is None and condition.get("seed") is None,
            f"{name}: original wiring must carry no control or seed",
        )
        _require(
            metadata["view_sha256"] == original["view_sha256"] and provenance == intact_provenance,
            f"{name}: original view differs",
        )
        return
    _require(isinstance(control, dict), f"{name}: control metadata missing")
    _require(control["kind"] == condition["mode"], f"{name}: control kind differs from lane")
    _require(
        type(control["seed"]) is int
        and 0 <= control["seed"] < 2**64
        and type(condition.get("seed")) is int
        and condition["seed"] == control["seed"],
        f"{name}: control seed differs from condition",
    )
    _require(
        control["schema"] == "neurocraft.full_control.v2"
        and control["artifact_manifest_sha256"] == artifact
        and control["faithful_view_sha256"] == original["view_sha256"],
        f"{name}: control artifact/faithful view binding differs",
    )
    _require(
        type(control["edge_count"]) is int and control["edge_count"] >= 0,
        f"{name}: invalid control edge count",
    )
    _require(
        control["rng"] == "numpy.random.Generator(PCG64(seed)).permutation(E)",
        f"{name}: control RNG declaration differs",
    )
    if name.startswith("shuffle_"):
        expected_seed = int(name.removeprefix("shuffle_"))
        _require(
            expected_seed in seeds and control["seed"] == expected_seed,
            f"{name}: control seed differs from planned/named seed",
        )
        _require(
            _hash(control["permutation_sha256"]) != ZERO,
            f"{name}: shuffled control permutation missing",
        )
        _require(
            provenance["weight_overlay_sha256"] == _digest(control)
            and provenance["origin"] == intact_provenance["origin"] + ":control:shuffled_weight",
            f"{name}: shuffled view does not bind control identity",
        )
    else:
        _require(
            name == "no_input"
            and control["kind"] == "no_input"
            and control["permutation_sha256"] == ZERO,
            f"{name}: no-input control kind/permutation differs",
        )
        _require(
            metadata["view_sha256"] == original["view_sha256"] and provenance == intact_provenance,
            f"{name}: no-input must retain original view",
        )
    expected_dynamics = _digest(
        {
            "schema": "neurocraft.dynamics.identity.full_control.v1",
            "backend_dynamics_sha256": intact_base["dynamics_hash"],
            "control_identity_sha256": _digest(control),
        }
    )
    _require(
        base["dynamics_hash"] == expected_dynamics,
        f"{name}: run dynamics does not bind control identity",
    )


def verify_directory(directory: Path, *, check_current_sources: bool = False) -> dict[str, Any]:
    """Validate a complete frozen pilot directory; return a small JSON-safe report."""
    try:
        return _verify(directory, check_current_sources)
    except EvidenceVerificationError:
        raise
    except (KeyError, TypeError, ValueError, OSError, IndexError) as exc:
        raise EvidenceVerificationError(f"malformed or incomplete evidence: {exc}") from exc


def _verify(directory: Path, current_sources: bool) -> dict[str, Any]:
    summary, plan = _read(directory / "summary.json"), _read(directory / "plan.json")
    _require(
        summary["status"] == "complete" and not (directory / "failure.json").exists(),
        "candidate is failed/incomplete",
    )
    manifest = summary["files"]
    actual_files = {path.name for path in directory.iterdir() if path.is_file()} - {"summary.json"}
    _require(set(manifest) == actual_files, "summary file inventory differs")
    for name, expected in manifest.items():
        _require(Path(name).name == name, "manifest path escapes evidence directory")
        _require(
            sha256((directory / name).read_bytes()).hexdigest() == _hash(expected),
            f"manifest hash mismatch: {name}",
        )
    plan_hash = _digest(plan)
    _require(summary["plan_sha256"] == plan_hash, "summary plan hash mismatch")
    _require(plan["schema"] == "circuit-playground-offline-v1", "unsupported plan schema")
    _require(summary["claim_scope"] == plan["scope"], "claim scope differs from plan")
    _require(
        set(plan["cases"]) == {"bright_sham", *PILOT, *FINGERPRINTS}, "unexpected case schedule"
    )
    _require(plan["ratio_floor"] == 1e-12, "unsupported ratio floor")
    for entry in plan["sources"].values():
        _hash(entry["sha256"])
        if current_sources:
            _require(
                sha256(Path(entry["path"]).read_bytes()).hexdigest() == entry["sha256"],
                f"current source changed: {entry['path']}",
            )
    source_hash = _digest(plan["sources"])
    if "source_inventory_sha256" in plan:
        _require(plan["source_inventory_sha256"] == source_hash, "source inventory hash mismatch")
    frozen, populations = (
        _read(directory / "frozen-comparison-masks.json"),
        _read(directory / "populations.json"),
    )
    _require(frozen["plan_sha256"] == plan_hash, "comparison masks plan hash mismatch")
    pairs = frozen["pairs"]
    _require(len(pairs) == plan["comparison_count"], "comparison mask denominator differs")
    target = [member["index"] for member in populations["dnp01"]]
    target_bodies = [member["body_id"] for member in populations["dnp01"]]
    masks = {
        "original": ([], []),
        "sham": ([], []),
        "restored": ([], []),
        "dnp01_silenced": (target, target_bodies),
    }
    used = set(target)
    for index, pair in enumerate(pairs):
        indices = pair["indices"]
        _require(
            len(indices) == 2
            and len(set(indices)) == 2
            and not used.intersection(indices)
            and not set(frozen["excluded_indices"]).intersection(indices),
            "comparison masks overlap targets, exclusions or each other",
        )
        used.update(indices)
        masks[f"comparison_{index:02d}"] = (indices, pair["body_ids"])
    controls = [f"shuffle_{seed}" for seed in plan["seeds"]] + ["no_input"]
    _require(len(set(controls)) == len(controls), "duplicate control conditions")
    for name in controls:
        masks[name] = ([], [])
    all_cases, pilot_cases = set(plan["cases"]), {"bright_sham", *PILOT}
    expected_trials = {
        (name, case)
        for name in masks
        for case in (all_cases if name in {"original", *controls} else pilot_cases)
    }
    original_bindings = _read(directory / "original-bindings.json")
    _require(original_bindings["control"] is None, "original wiring control must be null")
    node_count = original_bindings["sensory_binding"]["node_count"]
    _require(type(node_count) is int and node_count > 0, "invalid node count")
    traces = {}
    for path in sorted(directory.glob("*-condition.json")):
        envelope = _read(path)
        condition, condition_hash = envelope["condition"], envelope["sha256"]
        name, case = condition["name"], condition["case"]
        _require(
            (name, case) in expected_trials and (name, case) not in traces,
            "unexpected/duplicate condition trial",
        )
        _require(path.name == f"{name}-{case}-condition.json", "condition filename mismatch")
        _require(_digest(condition) == condition_hash, "condition descriptor hash mismatch")
        _require(
            (condition["indices"], condition["body_ids"]) == masks[name],
            "condition differs from frozen target/comparison mask",
        )
        expected_mode = (
            "no_input"
            if name == "no_input"
            else ("shuffled_weight" if name.startswith("shuffle_") else "faithful")
        )
        _require(condition["mode"] == expected_mode, "condition mode mismatch")
        metadata = (
            _read(directory / f"{name}-bindings.json") if name in controls else original_bindings
        )
        _require(
            metadata["policy"] == original_bindings["policy"]
            and metadata["policy_hash"] == original_bindings["policy_hash"],
            "policy changes across conditions",
        )
        _verify_control_binding(condition, metadata, original_bindings, plan["seeds"])
        traces[name, case] = _verify_trace(
            directory,
            condition,
            condition_hash,
            plan["cases"][case],
            node_count,
            metadata["view_sha256"],
            metadata["policy_hash"],
        )
    _require(set(traces) == expected_trials, "scheduled condition trials missing")
    for (name, case), trace in traces.items():
        if name != "no_input":
            _require(
                trace.inputs == traces["original", case].inputs,
                f"{name}-{case}: actual executed inputs differ from original",
            )
    restoration = {}
    for name in ("sham", "restored"):
        restoration[name] = all(
            traces[name, case].semantics == traces["original", case].semantics
            for case in pilot_cases
        )
        _require(restoration[name], f"{name}: full numerical/ProgramAction restoration differs")
    _close(summary["restoration_exact"], restoration, "restoration_exact")
    expected_pilot = {}
    for name in masks:
        expected_pilot[name] = {
            case: _response(traces[name, case], traces[name, "bright_sham"]) for case in PILOT
        }
    for case in PILOT:
        original_peak = expected_pilot["original"][case]["peak_abs_mean"]
        for results in expected_pilot.values():
            result = results[case]
            result["peak_difference_from_original"] = result["peak_abs_mean"] - original_peak
            result["peak_ratio_to_original"] = (
                result["peak_abs_mean"] / original_peak
                if original_peak > plan["ratio_floor"]
                else None
            )
            result["onset_tick"] = (
                next(
                    (
                        i
                        for i, value in enumerate(result["ttmn_signed_evoked_mean"])
                        if i >= 20 and abs(value) > 0.1 * original_peak
                    ),
                    None,
                )
                if original_peak > plan["ratio_floor"]
                else None
            )
    expected_fingerprints = {
        name: {
            case: _response(traces[name, case], traces[name, "bright_sham"])
            for case in FINGERPRINTS
        }
        for name in ("original", *controls)
    }
    _close(summary["pilot"], expected_pilot, "pilot")
    _close(summary["fingerprints"], expected_fingerprints, "fingerprints")
    return {
        "verified": True,
        "plan_sha256": plan_hash,
        "source_inventory_sha256": source_hash,
        "summary_sha256": sha256((directory / "summary.json").read_bytes()).hexdigest(),
        "manifest_files": len(manifest),
        "condition_trials": len(traces),
        "observations": sum(len(trace.ttmn) for trace in traces.values()),
        "neural_steps": sum(len(trace.inputs) for trace in traces.values()),
        "current_source_bytes_checked": current_sources,
        "restoration_exact": restoration,
        "scope": (
            "retained artifact consistency and independently recomputed reducers; "
            "no neural rerun or live play"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--check-current-sources", action="store_true")
    args = parser.parse_args()
    try:
        result = verify_directory(args.directory, check_current_sources=args.check_current_sources)
    except EvidenceVerificationError as exc:
        parser.exit(1, f"Evidence verification failed: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
