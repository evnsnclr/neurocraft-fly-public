"""Predeclared release comparisons: automated diagnostics, not human/biological evidence.

Use ``python -m scripts.run_release_comparisons plan --output NEW_DIR`` before
the first take. Execute three blocks in fresh diagnostic sessions, then ``report``.
A failed block is retained and never replaced in the same plan. This tool does
not launch Minecraft, access a network or publish anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any

from neurocraft_protocol import RunIdentity

from scripts import run_interaction_trials as trials

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "neurocraft.release_comparison.v1"
ANCHOR = "@e[type=minecraft:marker,tag=nf_release_anchor,limit=1]"
DISTANCES = (5.5, 4.5, 3.5, 2.5, 1.5)
MODEL_KEYS = (
    "graph_hash",
    "adapter_hash",
    "dynamics_hash",
    "decoder_hash",
    "observation_schema_hash",
)
FILES = (
    "scripts/run_release_comparisons.py",
    "scripts/run_interaction_trials.py",
    "scripts/program_decision_telemetry.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_identity(identity: dict[str, Any]) -> dict[str, str]:
    RunIdentity.from_dict(identity)
    return {key: identity[key] for key in MODEL_KEYS}


def write_new(path: Path, data: dict[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write(trials.canonical(data) + b"\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        raw = stream.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise trials.TrialError("metadata exceeds 4 MiB")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise trials.TrialError("expected a metadata object")
    return value


def block_cases(block: int) -> tuple[trials.Case, ...]:
    if type(block) is not int or block not in range(3):
        raise trials.TrialError("block must be 0, 1 or 2")
    order = trials.LANES[block:] + trials.LANES[:block]
    cases = [
        trials.Case(index, lane, block, source)
        for index, (source, lane) in enumerate(
            (source, lane) for source in trials.SOURCES for lane in order
        )
    ]
    cases.append(trials.Case(18, "original", block, "brush"))
    return tuple(cases)


def declaration(root: Path = ROOT) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "runner_hashes": {name: sha(root / name) for name in FILES},
        "blocks": [[asdict(case) for case in block_cases(i)] for i in range(3)],
        "core_cases": 54,
        "restoration_probe_index": 18,
        "shuffle_seed": 7,
        "profile": trials.PROFILE,
        "normal_tps_target": 20,
        "model_binding": "first block preflight locks all lane models before any stimulus",
        "reset": (
            "fresh epoch per case; ten eligible matched observations "
            "with stable light/event counters"
        ),
        "approach": {
            "anchor": "marker at fly pose once before stimulus; never follows later movement",
            "distances_blocks": list(DISTANCES),
            "minimum_observed_hold_ticks": 6,
            "readout_adaptation": False,
            "timing": "confirm each requested position; wait six observed ticks after confirmation",
        },
        "inputs": "prepared scene, scripted actor displacement, synthetic native brush/attack",
        "histories": "embodied histories may differ; not identical-input replay",
        "rules": [
            "All scheduled cases remain, including misses, refusals and unattempted cases.",
            "No block retry in this plan; changed fixtures need a new plan.",
            "No-input fallback movement and shuffled retained responses are valid outcomes.",
            "Qualification requires delivered fixtures and no unexplained missing receipts.",
            "Each block ends with a restored-original brush and original handover.",
            "One shuffle and repeated model trials do not establish biological validity.",
        ],
    }


def make_plan(output: Path) -> dict[str, Any]:
    value = declaration()
    output.mkdir()
    write_new(output / "plan.json", value)
    (output / "plan.sha256").write_text(sha(output / "plan.json") + "\n")
    return value


def verify_plan(directory: Path) -> dict[str, Any]:
    path = directory / "plan.json"
    if sha(path) != (directory / "plan.sha256").read_text().strip():
        raise trials.TrialError("plan hash mismatch")
    value = read_json(path)
    if value != declaration():
        raise trials.TrialError("plan differs from fixed protocol or runner bytes")
    return value


def approach_commands(source: str) -> tuple[str, ...]:
    if source not in ("player_approach", "mob_approach"):
        raise trials.TrialError("not an approach stimulus")
    target = trials.PLAYER if source == "player_approach" else trials.COW
    height = "-1.5" if source == "player_approach" else "0"
    return tuple(
        f"execute at {ANCHOR} run tp {target} ^0 ^{height} ^{distance} ~180 0"
        for distance in DISTANCES
    )


class ReleaseRunner(trials.Runner):
    reset_each_case = True
    comparison_scope = (
        "one of three predeclared 18-case blocks plus restored brush; fresh epoch per case; "
        "seed7 only; fixed anchor approach without readout adaptation; histories may differ"
    )

    def bind_policy(self) -> None:
        assert self.epoch is not None
        row = self.epoch.latest()
        policy = row.get("policy_hash") if row is not None else None
        if not isinstance(policy, str) or re.fullmatch(r"[0-9a-f]{64}", policy) is None:
            raise trials.TrialError("applied baseline lacks a policy identity")
        path = self.output.parent / "policy.json"
        value = {"schema": SCHEMA, "policy_hash": policy}
        if path.exists():
            if read_json(path) != value:
                raise trials.TrialError("applied policy differs from first baseline")
        else:
            write_new(path, value)
        self.policy_hash = policy

    def observe_current_epoch(self) -> None:
        super().observe_current_epoch()
        self.bind_policy()

    def handover(self, lane: str) -> None:
        super().handover(lane)
        self.bind_policy()

    def server_vector(self, target: str, field: str, deadline: float) -> list[float]:
        raw = self.server_query(
            f"data get entity {target} {field}",
            r"has the following entity data: \[([^\]\r\n]+)\]",
            deadline,
        )
        values = [float(part.strip().rstrip("df")) for part in raw.split(",")]
        if len(values) != (3 if field == "Pos" else 2) or not all(map(math.isfinite, values)):
            raise trials.TrialError("invalid server position or rotation")
        return values

    def perform_approach(self, case: trials.Case, start_tick: int, deadline: float) -> None:
        self.send("kill @e[type=minecraft:marker,tag=nf_release_anchor]")
        self.send(
            f"execute at {trials.FLY} run summon minecraft:marker ~ ~ ~ "
            '{Tags:["nf_release_anchor"]}'
        )
        self.send(f"execute at {trials.FLY} rotated ~ 0 run tp {ANCHOR} ~ ~ ~ ~ 0")
        anchor = self.server_vector(ANCHOR, "Pos", deadline)
        yaw = math.radians(self.server_vector(ANCHOR, "Rotation", deadline)[0])
        if case.source == "player_approach":
            self.send(f"gamemode creative {trials.PLAYER}")
        else:
            self.send(
                f"execute at {ANCHOR} run summon minecraft:cow ^0 ^0 ^5.5 "
                '{NoAI:1b,NoGravity:1b,Invulnerable:1b,Tags:["nf_v2_cow"]}'
            )
        self.record(
            "fixed_approach",
            case_index=case.index,
            start_tick=start_tick,
            commands=list(approach_commands(case.source)),
            readout_adaptation=False,
        )
        points = []
        for index, command in enumerate(approach_commands(case.source)):
            # Issue every point, irrespective of readouts or resulting body response.
            self.send(command)
            target = trials.PLAYER if case.source == "player_approach" else trials.COW
            actual = self.server_vector(target, "Pos", deadline)
            expected = [
                anchor[0] - math.sin(yaw) * DISTANCES[index],
                anchor[2] + math.cos(yaw) * DISTANCES[index],
            ]
            # The receipt's observation can predate the position query. Use a
            # server command processed after that query as the temporal barrier.
            confirmed_tick = int(
                self.server_query("time query gametime", r"The time is (\d+)", deadline)
            )
            held = self.observed_after(confirmed_tick + 6, deadline)
            point = {
                "command": command,
                "distance": DISTANCES[index],
                "position": actual,
                "expected_xz": expected,
                "confirmed_tick": confirmed_tick,
                "hold_end_tick": held["observation_tick"],
                "position_confirmed": math.dist([actual[0], actual[2]], expected) <= 0.2,
            }
            points.append(point)
            self.record("waypoint_observed", case_index=case.index, **point)
        self.results[case.index]["approach_waypoints"] = points

    def run_case(self, case: trials.Case) -> None:
        super().run_case(case)
        row = self.results[case.index]
        if row.get("status") != "observed":
            return
        assert self.epoch is not None
        start, end = row["start_tick"], row["end_tick"]
        window = {
            seq: issued
            for seq, issued in self.epoch.issued.items()
            if start <= issued["observation_tick"] <= end
        }
        row["policy_hash"] = self.policy_hash
        row["policy_verified"] = bool(window) and all(
            request.get("policy_hash") == self.policy_hash for request in window.values()
        )
        source = self.epoch.path / "program-decisions.inputs.ndjson"
        samples: dict[int, dict[str, Any]] = {}
        if source.exists():
            with source.open("rb") as stream:
                data = stream.read(4 * 1024 * 1024 + 1)
            if len(data) > 4 * 1024 * 1024:
                raise trials.TrialError("input sidecar exceeds bounded epoch window")
            for line in data.split(b"\n")[:-1]:
                sample = json.loads(line)
                seq = sample.get("command_seq")
                if seq in window:
                    if seq in samples:
                        raise trials.TrialError("duplicate input diagnostic sequence")
                    samples[seq] = sample
        matched = [
            seq
            for seq, request in window.items()
            if input_matches(samples.get(seq), request, self.epoch.identity, case.lane)
        ]
        row["input_samples_expected"] = end - start + 1
        row["input_samples_matched"] = len(matched)
        row["input_samples_missing_or_invalid"] = sorted(window.keys() - set(matched))
        row["input_gate_verified"] = len(matched) == end - start + 1
        # Separate ledger kind; do not increase the legacy reducer's per-case row budget.
        for seq in sorted(samples):
            self.record("input_diagnostic", case_index=case.index, sample=samples[seq])
        if case.source in ("player_approach", "mob_approach"):
            points = row.get("approach_waypoints", [])
            row["waypoints_verified"] = len(points) == 5 and all(
                p["position_confirmed"] is True
                and p["hold_end_tick"] - p["confirmed_tick"] >= 6
                and start <= p["confirmed_tick"] <= p["hold_end_tick"] <= end
                for p in points
            )
        self.record("case_qualification", case_index=case.index, result=row)


def input_matches(
    sample: Any, request: dict[str, Any], identity: dict[str, Any], lane: str
) -> bool:
    if (
        not isinstance(sample, dict)
        or sample.get("schema") != "neurocraft.program_input_diagnostic.v1"
    ):
        return False
    if sample.get("bound") is not True or sample.get("identity") != identity:
        return False
    if any(
        sample.get(k) != request.get(k)
        for k in ("command_seq", "observation_seq", "observation_tick", "state_digest")
    ):
        return False
    values = sample.get("encoded_stimulation")
    if (
        not isinstance(values, dict)
        or not values
        or not all(type(v) in (int, float) and math.isfinite(v) for v in values.values())
    ):
        return False
    if lane == "no_input":
        before = sample.get("before_control")
        return (
            isinstance(before, dict)
            and all(isinstance(key, str) for key in before)
            and {key for key in before if not key.startswith(("eem_raw:", "eem_mean:"))}
            == set(values)
            and all(type(v) in (int, float) and math.isfinite(v) for v in before.values())
            and all(v == 0 for v in values.values())
        )
    return sample.get("before_control") is None


def qualify_case(row: dict[str, Any]) -> bool:
    if not (
        row.get("status") == "observed"
        and row.get("fixture_verified") is True
        and row.get("source_verified") is True
    ):
        return False
    start, end = row.get("start_tick"), row.get("end_tick")
    if type(start) is not int or type(end) is not int or not 0 < end - start + 1 <= 100:
        return False
    if (
        row.get("response_window_issued") != end - start + 1
        or row.get("response_window_missing_receipts") != []
        or row.get("response_window_unapplied") != []
        or row.get("input_gate_verified") is not True
        or row.get("policy_verified") is not True
    ):
        return False
    if (
        row.get("source") in ("player_approach", "mob_approach")
        and row.get("waypoints_verified") is not True
    ):
        return False
    if row.get("issued_without_applied_receipt"):
        return False
    # An incomplete target response cannot become a clean null outcome.
    if row.get("issued_response") is True:
        return row.get("applied_response") is True and row.get("complete_applied_hold") is True
    return row.get("issued_response") is False and row.get("applied_response") is False


def aggregate(directory: Path) -> dict[str, Any]:
    plan = verify_plan(directory)
    rows: list[dict[str, Any]] = []
    restorations: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    sessions: set[str] = set()
    failures: dict[str, str | None] = {}
    models_path = directory / "models.json"
    bound_models = read_json(models_path).get("models") if models_path.exists() else None
    policy_path = directory / "policy.json"
    policy = read_json(policy_path).get("policy_hash") if policy_path.exists() else None
    for block, expected in enumerate(plan["blocks"]):
        path = directory / f"block-{block}" / "summary.json"
        result = read_json(path) if path.exists() else None
        if result is not None:
            provenance = read_json(path.parent / "block.json")
            if provenance.get("plan_sha256") != sha(directory / "plan.json"):
                raise trials.TrialError("block belongs to another plan")
            if provenance.get("block") != block:
                raise trials.TrialError("block index mismatch")
            run_id = provenance.get("run_id")
            if not isinstance(run_id, str) or not run_id or run_id in sessions:
                raise trials.TrialError("blocks require unique fresh run_id values")
            sessions.add(run_id)
            if not isinstance(bound_models, dict) or set(bound_models) != set(trials.LANES):
                raise trials.TrialError("missing bound lane model identities")
            if provenance.get("model_identities") != bound_models:
                raise trials.TrialError("block model identities differ from bound models")
            actual = result.get("cases")
            if not isinstance(actual, list) or len(actual) != len(expected):
                raise trials.TrialError("block omitted or added scheduled cases")
            for want, got in zip(expected, actual, strict=True):
                if not isinstance(got, dict) or any(got.get(k) != v for k, v in want.items()):
                    raise trials.TrialError("block case identity/order mismatch")
                identity = got.get("epoch_identity")
                if identity is not None and (
                    not isinstance(identity, dict) or identity.get("run_id") != run_id
                ):
                    raise trials.TrialError("case belongs to another block session")
                if got.get("status") == "observed" and identity is None:
                    raise trials.TrialError("observed case has no epoch identity")
                if identity is not None and model_identity(identity) != bound_models[got["lane"]]:
                    raise trials.TrialError("case model identity differs from bound lane")
                if got.get("status") == "observed" and (
                    not isinstance(policy, str)
                    or re.fullmatch(r"[0-9a-f]{64}", policy) is None
                    or got.get("policy_hash") != policy
                ):
                    raise trials.TrialError("case policy differs from bound policy")
            hashes[f"block-{block}/summary.json"] = sha(path)
            failures[str(block)] = result.get("failure")
        else:
            actual = [
                {**case, "status": "unattempted", "reason": "no completed block summary"}
                for case in expected
            ]
        rows.extend({"block": block, **row, "qualified": qualify_case(row)} for row in actual[:18])
        probe = actual[18]
        restorations.append(
            {
                "block": block,
                "handover_confirmed": bool(result and result.get("original_restored") is True),
                "brush_probe_qualified": qualify_case(probe)
                and probe.get("applied_response") is True,
                "probe": probe,
            }
        )
    groups = []
    for source in trials.SOURCES:
        for lane in trials.LANES:
            subset = [row for row in rows if row["source"] == source and row["lane"] == lane]
            groups.append(
                {
                    "source": source,
                    "lane": lane,
                    "scheduled": len(subset),
                    "qualified": sum(row["qualified"] for row in subset),
                    "target_applied": sum(
                        row["qualified"] and row.get("applied_response") is True for row in subset
                    ),
                    "outcomes": dict(Counter(row.get("outcome", row["status"]) for row in subset)),
                }
            )
    return {
        "schema": SCHEMA,
        "plan_sha256": sha(directory / "plan.json"),
        "block_summary_sha256": hashes,
        "model_identities": bound_models,
        "models_sha256": sha(models_path) if models_path.exists() else None,
        "policy_hash": policy,
        "policy_binding_sha256": sha(policy_path) if policy_path.exists() else None,
        "block_failures": failures,
        "scheduled": 54,
        "qualified": sum(row["qualified"] for row in rows),
        "comparison_qualified": all(f is None for f in failures.values())
        and all(row["qualified"] for row in rows)
        and all(r["handover_confirmed"] and r["brush_probe_qualified"] for r in restorations),
        "groups": groups,
        "cases": rows,
        "restorations": restorations,
        "scope": "automated embodied diagnostic; seed7; not human or biological validation",
    }


def run_block(directory: Path, run: Path, block: int) -> dict[str, Any]:
    verify_plan(directory)
    cases = block_cases(block)
    output = directory / f"block-{block}"
    if output.exists():
        raise trials.TrialError("block already attempted; preserve it and create a new plan")
    session = read_json(run / "session.json")
    if session.get("requested_capture_seconds") != 720:
        raise trials.TrialError("release block requires --capture-seconds 720")
    for prior in directory.glob("block-*/block.json"):
        if read_json(prior).get("run_id") == session.get("run_id"):
            raise trials.TrialError("each block requires a fresh diagnostic session")
    runner = ReleaseRunner(run, output, cases, native_targeting="aim_at_dispatch")
    try:
        models = {lane: model_identity(runner.catalog[lane]["identity"]) for lane in trials.LANES}
        models_path = directory / "models.json"
        if models_path.exists():
            if read_json(models_path) != {"schema": SCHEMA, "models": models}:
                raise trials.TrialError("session models differ from first block preflight")
        else:
            write_new(models_path, {"schema": SCHEMA, "models": models})
        write_new(
            output / "block.json",
            {
                "schema": SCHEMA,
                "block": block,
                "plan_sha256": sha(directory / "plan.json"),
                "session_sha256": sha(run / "session.json"),
                "lane_catalog_sha256": sha(run / "lane-catalog.json"),
                "run_id": runner.session["run_id"],
                "model_identities": models,
            },
        )
        return runner.execute()
    except BaseException as exc:
        with suppress(OSError):
            write_new(output / "block_failure.json", {"failure": str(exc), "block": block})
        raise
    finally:
        for resource in reversed(runner.resources):
            if not resource.closed:
                resource.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--output", type=Path, required=True)
    block = sub.add_parser("run-block")
    block.add_argument("--plan", type=Path, required=True)
    block.add_argument("--run", type=Path, required=True)
    block.add_argument("--block", type=int, choices=range(3), required=True)
    report = sub.add_parser("report")
    report.add_argument("--plan", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            make_plan(args.output)
            print(json.dumps({"plan": str(args.output), "scheduled": 54, "restoration_probes": 3}))
        elif args.command == "run-block":
            result = run_block(args.plan, args.run, args.block)
            print(
                json.dumps(
                    {
                        "all_cases_observed": result["all_cases_observed"],
                        "original_restored": result["original_restored"],
                    }
                )
            )
            return 0 if result["all_cases_observed"] and result["original_restored"] else 1
        else:
            result = aggregate(args.plan)
            write_new(args.output, result)
            print(
                json.dumps(
                    {k: result[k] for k in ("scheduled", "qualified", "comparison_qualified")}
                )
            )
            return 0 if result["comparison_qualified"] else 1
    except (OSError, ValueError, trials.TrialError) as exc:
        parser.exit(1, f"release comparison refused: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
