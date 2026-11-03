"""Tamper checks on small constructed receipt bundles, not neural validation."""

import json
import struct
from copy import deepcopy
from hashlib import sha256

import pytest

from scripts import run_circuit_playground as producer
from scripts import verify_circuit_playground as verifier


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


def manifest(directory):
    summary = json.loads((directory / "summary.json").read_text())
    summary["files"] = {
        p.name: sha256(p.read_bytes()).hexdigest()
        for p in directory.iterdir()
        if p.is_file() and p.name != "summary.json"
    }
    save(directory / "summary.json", summary)


def trace_rows(directory, name):
    return [json.loads(line) for line in (directory / name).read_text().splitlines()]


def rewrite_trace(directory, name, rows, *, rechain=False):
    if rechain:
        previous = verifier.ZERO
        for row in rows:
            for record in row["neural_steps"]:
                neural = record["neural"]
                neural["execution_sha256"] = verifier._digest(verifier._execution_material(record))
                neural["previous_trace_sha256"] = previous
                previous = sha256(
                    bytes.fromhex(previous) + bytes.fromhex(neural["execution_sha256"])
                ).hexdigest()
                neural["trace_sha256"] = previous
    path = directory / name
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    coverage_path = directory / name.replace(".ndjson", "-coverage.json")
    coverage = json.loads(coverage_path.read_text())
    coverage["trace_file_sha256"] = sha256(path.read_bytes()).hexdigest()
    save(coverage_path, coverage)
    manifest(directory)


def make_trial(directory, name, case, frames, mask, body_ids, metadata, epoch):
    descriptor = {
        "name": name,
        "case": case,
        "mode": "no_input"
        if name == "no_input"
        else "shuffled_weight"
        if name.startswith("shuffle_")
        else "faithful",
        "indices": mask,
        "body_ids": body_ids,
        "base_identity": {**metadata["base_controller_identity"], "session_epoch": epoch},
        "mask_sha256": sha256(bytes(int(i in mask) for i in range(6))).hexdigest(),
    }
    if metadata["control"] is not None:
        descriptor["seed"] = metadata["control"]["seed"]
    condition_hash = verifier._digest(descriptor)
    save(
        directory / f"{name}-{case}-condition.json",
        {"condition": descriptor, "sha256": condition_hash},
    )
    zero = sha256(bytes(48)).hexdigest()
    factor = {"dnp01_silenced": 0.3, "comparison_00": 0.9, "shuffle_7": 0.8, "no_input": 0}.get(
        name, 1
    )
    previous_trace, previous_state, previous_delay = (
        verifier.ZERO,
        verifier._digest("start"),
        verifier._digest("delay"),
    )
    rows = []
    for tick, frame in enumerate(frames):
        amplitude = 0 if case == "bright_sham" else max(0, tick - 19) * 0.02 * factor
        vector = [
            amplitude / 2,
            amplitude / 3,
            amplitude,
            amplitude * 0.8,
            amplitude / 4,
            amplitude / 5,
        ]
        for index in mask:
            vector[index] = 0
        activity_hash = sha256(struct.pack(">6d", *vector)).hexdigest()
        assert name != "no_input" or activity_hash == zero
        sensory = zero if name == "no_input" else verifier._digest({"case": case, "tick": tick})
        records = []
        for offset in range(5):
            step = 5 * tick + offset
            after_state = verifier._digest({"step": step + 1, "activity": activity_hash})
            after_delay = verifier._digest({"delay_step": step + 1})
            neural = {
                "step_before": step,
                "step_after": step + 1,
                "ablation_mask_sha256": descriptor["mask_sha256"],
                "direct_stimulation_sha256": verifier.ZERO,
                "encoded_stimulation_sha256": sensory,
                "state_before_sha256": previous_state,
                "state_after_sha256": after_state,
                "delay_before_sha256": previous_delay,
                "delay_after_sha256": after_delay,
            }
            record = {
                "neural": neural,
                "activity_sha256": activity_hash,
                "sensory_provenance_sha256": verifier._digest("sensory provenance"),
                "direct_provenance_sha256": verifier.ZERO,
                "verification_policy_sha256": verifier._digest("verification policy"),
                "view_digest_sha256": metadata["view_sha256"],
                "view_validation_sha256": verifier._digest("view validation"),
                "verified_through_step": 0,
                "selected_activity": {
                    "dnp01": vector[:2],
                    "ttmn": vector[2:4],
                    "silenced": [vector[i] for i in mask],
                },
            }
            neural["execution_sha256"] = verifier._digest(verifier._execution_material(record))
            neural["previous_trace_sha256"] = previous_trace
            previous_trace = sha256(
                bytes.fromhex(previous_trace) + bytes.fromhex(neural["execution_sha256"])
            ).hexdigest()
            neural["trace_sha256"] = previous_trace
            previous_state, previous_delay = after_state, after_delay
            records.append(record)
        source = "player_approach" if amplitude > 0 else None
        program = (
            "ESCAPE_JUMP"
            if amplitude > 0 and name in {"original", "sham", "restored"}
            else "CRUISE"
        )
        if name == "no_input":
            source = None
        command = {
            "identity": descriptor["base_identity"],
            "command_seq": tick,
            "observation_seq": tick,
            "observation_tick": tick,
            "target_tick": tick + 2,
            "valid_through_tick": tick + 2,
            "neural_step_start": tick * 5,
            "neural_step_end": tick * 5 + 5,
            "policy_hash": metadata["policy_hash"],
            "state_digest": previous_state,
            "program": program,
            "modulation": {"speed": 1.0, "turn_rate": 1.0, "amplitude": 1.0, "urgency": 0.0},
            "parameters": {"turn_bias": 0.0},
            "readout": {"ttmn_jump": amplitude, "dna01_turn": 0.0, "wing_common": 0.0},
            "type": "program_action",
            "protocol": "neurocraft.protocol.v2",
        }
        decision = {
            "source": source,
            "reason": "held" if source else "no_eligible_response",
            "scores": {},
        }
        named = {
            **command,
            "command_sha256": verifier._digest(command),
            "reason": decision["reason"],
            "interaction": {"source": source, "scores": {}},
            "issued_trigger": None,
        }
        rows.append(
            {
                "tick": tick,
                "condition_sha256": condition_hash,
                "applied_in_minecraft": False,
                "observation": {
                    "identity": descriptor["base_identity"],
                    "sensors": frame,
                    "game_tick": tick,
                    "observation_seq": tick,
                },
                "issued_program": command,
                "named_evidence": named,
                "decision": decision,
                "rectified_jump": amplitude,
                "neural_steps": records,
            }
        )
    path = directory / f"{name}-{case}.ndjson"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    save(
        directory / f"{name}-{case}-coverage.json",
        {
            "run_is_covered": True,
            "rows": len(rows),
            "verified_through_step": len(rows) * 5,
            "condition_sha256": condition_hash,
            "trace_file_sha256": sha256(path.read_bytes()).hexdigest(),
        },
    )
    return rows


@pytest.fixture
def bundle(tmp_path):
    directory = tmp_path / "bundle"
    directory.mkdir()
    source = tmp_path / "frozen_source.py"
    source.write_text("# frozen receipt fixture\n")
    cases = {
        case: [{"synthetic_case": case, "tick": tick} for tick in range(24)]
        for case in {"bright_sham", *verifier.PILOT, *verifier.FINGERPRINTS}
    }
    plan = {
        "schema": "circuit-playground-offline-v1",
        "cases": cases,
        "seeds": [7],
        "comparison_count": 1,
        "ratio_floor": 1e-12,
        "scope": "constructed test receipts",
        "sources": {
            "test": {"path": str(source), "sha256": sha256(source.read_bytes()).hexdigest()}
        },
    }
    save(directory / "plan.json", plan)
    pair = {"indices": [4, 5], "body_ids": [104, 105]}
    save(
        directory / "frozen-comparison-masks.json",
        {"pairs": [pair], "excluded_indices": [2, 3], "plan_sha256": verifier._digest(plan)},
    )
    save(
        directory / "populations.json",
        {"dnp01": [{"index": 0, "body_id": 100}, {"index": 1, "body_id": 101}]},
    )
    artifact = verifier._digest("artifact")
    provenance = {
        "origin": "test:original",
        "graph_identity_sha256": verifier._digest("graph"),
        **{
            key: verifier._digest(key)
            for key in (
                "sign_overlay_sha256",
                "delay_overlay_sha256",
                "topology_overlay_sha256",
                "source_manifest_sha256",
                "weight_overlay_sha256",
            )
        },
    }
    metadata = {
        "sensory_binding": {"node_count": 6, "graph_manifest_sha256": artifact},
        "base_controller_identity": {
            "session_epoch": 0,
            "graph_hash": artifact,
            "dynamics_hash": verifier._digest("backend"),
        },
        "control": None,
        "view_provenance": provenance,
        "view_provenance_sha256": verifier._digest(provenance),
        "policy": {"frozen": True},
        "view_sha256": verifier._digest("view"),
        "policy_hash": verifier._digest("policy"),
    }
    save(directory / "original-bindings.json", metadata)
    conditions = {
        "original": ([], []),
        "sham": ([], []),
        "restored": ([], []),
        "dnp01_silenced": ([0, 1], [100, 101]),
        "comparison_00": ([4, 5], [104, 105]),
        "shuffle_7": ([], []),
        "no_input": ([], []),
    }
    traces = {}
    for epoch, (name, (mask, bodies)) in enumerate(conditions.items()):
        lane_metadata = deepcopy(metadata)
        if name in {"shuffle_7", "no_input"}:
            control = {
                "schema": "neurocraft.full_control.v2",
                "artifact_manifest_sha256": artifact,
                "faithful_view_sha256": metadata["view_sha256"],
                "edge_count": 6,
                "kind": "no_input" if name == "no_input" else "shuffled_weight",
                "seed": 7,
                "permutation_sha256": verifier.ZERO
                if name == "no_input"
                else verifier._digest("perm"),
                "rng": "numpy.random.Generator(PCG64(seed)).permutation(E)",
            }
            lane_metadata["control"] = control
            lane_metadata["base_controller_identity"]["dynamics_hash"] = verifier._digest(
                {
                    "schema": "neurocraft.dynamics.identity.full_control.v1",
                    "backend_dynamics_sha256": metadata["base_controller_identity"][
                        "dynamics_hash"
                    ],
                    "control_identity_sha256": verifier._digest(control),
                }
            )
            if name == "shuffle_7":
                lane_metadata["view_provenance"]["origin"] += ":control:shuffled_weight"
                lane_metadata["view_provenance"]["weight_overlay_sha256"] = verifier._digest(
                    control
                )
                lane_metadata["view_provenance_sha256"] = verifier._digest(
                    lane_metadata["view_provenance"]
                )
                lane_metadata["view_sha256"] = verifier._digest("shuffle view")
            save(directory / f"{name}-bindings.json", lane_metadata)
        scheduled = (
            cases
            if name in {"original", "shuffle_7", "no_input"}
            else {"bright_sham", *verifier.PILOT}
        )
        for case in scheduled:
            traces[name, case] = make_trial(
                directory, name, case, cases[case], mask, bodies, lane_metadata, epoch
            )
    pilot = {
        name: {
            case: producer.response(traces[name, case], traces[name, "bright_sham"])
            for case in verifier.PILOT
        }
        for name in conditions
    }
    for case in verifier.PILOT:
        original = pilot["original"][case]["peak_abs_mean"]
        for results in pilot.values():
            result = results[case]
            result["peak_difference_from_original"] = result["peak_abs_mean"] - original
            result["peak_ratio_to_original"] = result["peak_abs_mean"] / original
            result["onset_tick"] = next(
                (
                    i
                    for i, value in enumerate(result["ttmn_signed_evoked_mean"])
                    if i >= 20 and abs(value) > 0.1 * original
                ),
                None,
            )
    fingerprints = {
        name: {
            case: producer.response(traces[name, case], traces[name, "bright_sham"])
            for case in verifier.FINGERPRINTS
        }
        for name in ("original", "shuffle_7", "no_input")
    }
    save(
        directory / "summary.json",
        {
            "status": "complete",
            "plan_sha256": verifier._digest(plan),
            "claim_scope": plan["scope"],
            "pilot": pilot,
            "fingerprints": fingerprints,
            "restoration_exact": {"sham": True, "restored": True},
            "files": {},
        },
    )
    manifest(directory)
    return directory


def test_complete_bundle_reduces_and_checks_all_traces_without_changing_files(bundle):
    before = {p.name: p.read_bytes() for p in bundle.iterdir()}
    report = verifier.verify_directory(bundle, check_current_sources=True)
    assert report["verified"] and report["condition_trials"] == 43
    assert report["observations"] == 43 * 24 and report["neural_steps"] == 43 * 24 * 5
    assert report["restoration_exact"] == {"sham": True, "restored": True}
    assert before == {p.name: p.read_bytes() for p in bundle.iterdir()}


def test_manifest_catches_changed_artifact_bytes(bundle):
    (bundle / "populations.json").write_text("{}")
    with pytest.raises(verifier.EvidenceVerificationError, match="manifest hash mismatch"):
        verifier.verify_directory(bundle)


@pytest.mark.parametrize(
    "field,value",
    [
        ("peak_abs_mean", 9.0),
        ("peak_ratio_to_original", 4.0),
        ("onset_tick", 21),
        ("issued_program_counts", {"CRUISE": 24}),
        ("selected_sources", ["brush"]),
    ],
)
def test_summary_is_recomputed_rather_than_trusting_complete_status(bundle, field, value):
    path = bundle / "summary.json"
    summary = json.loads(path.read_text())
    summary["pilot"]["original"]["player_approach"][field] = value
    save(path, summary)
    with pytest.raises(
        verifier.EvidenceVerificationError, match=r"pilot\.original\.player_approach"
    ):
        verifier.verify_directory(bundle)


def test_rehashed_executed_input_tampering_is_detected_against_original(bundle):
    name = "comparison_00-player_approach.ndjson"
    rows = trace_rows(bundle, name)
    rows[20]["neural_steps"][0]["neural"]["encoded_stimulation_sha256"] = "f" * 64
    rewrite_trace(bundle, name, rows, rechain=True)
    with pytest.raises(verifier.EvidenceVerificationError, match="executed inputs differ"):
        verifier.verify_directory(bundle)


def test_masked_cells_must_be_recorded_and_zero_at_every_step(bundle):
    name = "comparison_00-player_approach.ndjson"
    rows = trace_rows(bundle, name)
    rows[10]["neural_steps"][1]["selected_activity"]["silenced"][0] = 0.1
    rewrite_trace(bundle, name, rows)
    with pytest.raises(verifier.EvidenceVerificationError, match="silenced-cell activities"):
        verifier.verify_directory(bundle)


@pytest.mark.parametrize("field", ["modulation", "parameters", "target_tick"])
def test_full_program_semantics_detect_rehashed_restoration_changes(bundle, field):
    name = "restored-player_approach.ndjson"
    rows = trace_rows(bundle, name)
    command = rows[23]["issued_program"]
    if field == "modulation":
        command[field]["urgency"] = 0.4
    elif field == "parameters":
        command[field]["turn_bias"] = 0.4
    else:
        command[field] += 1
    rows[23]["named_evidence"][field] = command[field]
    rows[23]["named_evidence"]["command_sha256"] = verifier._digest(command)
    rewrite_trace(bundle, name, rows)
    with pytest.raises(verifier.EvidenceVerificationError, match="full numerical/ProgramAction"):
        verifier.verify_directory(bundle)


def test_changed_coverage_counts_are_not_accepted_after_rehashing(bundle):
    path = bundle / "original-player_approach-coverage.json"
    value = json.loads(path.read_text())
    value["verified_through_step"] -= 1
    save(path, value)
    manifest(bundle)
    with pytest.raises(verifier.EvidenceVerificationError, match="final coverage incomplete"):
        verifier.verify_directory(bundle)


def test_condition_descriptor_hash_is_checked_even_with_valid_manifest(bundle):
    path = bundle / "comparison_00-player_approach-condition.json"
    value = json.loads(path.read_text())
    value["condition"]["body_ids"][0] += 1
    save(path, value)
    manifest(bundle)
    with pytest.raises(verifier.EvidenceVerificationError, match="condition descriptor hash"):
        verifier.verify_directory(bundle)


def test_rehashed_descriptive_shuffle_seeds_must_match_the_actual_control(bundle):
    for path in bundle.glob("shuffle_7-*-condition.json"):
        envelope = json.loads(path.read_text())
        envelope["condition"]["seed"] = 999
        envelope["sha256"] = verifier._digest(envelope["condition"])
        save(path, envelope)
        trace_path = path.name.replace("-condition.json", ".ndjson")
        rows = trace_rows(bundle, trace_path)
        for row in rows:
            row["condition_sha256"] = envelope["sha256"]
        coverage_path = bundle / path.name.replace("-condition.json", "-coverage.json")
        coverage = json.loads(coverage_path.read_text())
        coverage["condition_sha256"] = envelope["sha256"]
        save(coverage_path, coverage)
        rewrite_trace(bundle, trace_path, rows)
    with pytest.raises(verifier.EvidenceVerificationError, match="control seed differs"):
        verifier.verify_directory(bundle)


@pytest.mark.parametrize(
    "lane,field,value,message",
    [
        ("shuffle_7", "kind", "no_input", "control kind"),
        ("no_input", "kind", "shuffled_weight", "control kind"),
        ("shuffle_7", "seed", 999, "control seed"),
        ("shuffle_7", "faithful_view_sha256", "f" * 64, "faithful view binding"),
        ("shuffle_7", "artifact_manifest_sha256", "f" * 64, "artifact/faithful view binding"),
        ("no_input", "permutation_sha256", "f" * 64, "kind/permutation"),
    ],
)
def test_control_metadata_cannot_be_relabeled_after_manifest_refresh(
    bundle, lane, field, value, message
):
    path = bundle / f"{lane}-bindings.json"
    metadata = json.loads(path.read_text())
    metadata["control"][field] = value
    save(path, metadata)
    manifest(bundle)
    with pytest.raises(verifier.EvidenceVerificationError, match=message):
        verifier.verify_directory(bundle)


def test_original_wiring_cannot_acquire_a_control_overlay(bundle):
    path = bundle / "original-bindings.json"
    metadata = json.loads(path.read_text())
    metadata["control"] = json.loads((bundle / "shuffle_7-bindings.json").read_text())["control"]
    save(path, metadata)
    manifest(bundle)
    with pytest.raises(
        verifier.EvidenceVerificationError, match="original wiring control must be null"
    ):
        verifier.verify_directory(bundle)


def test_current_source_check_is_optional_and_historical_inventory_remains_bound(bundle):
    plan = json.loads((bundle / "plan.json").read_text())
    source = next(iter(plan["sources"].values()))["path"]
    from pathlib import Path

    Path(source).write_text("# changed after the frozen run\n")
    assert verifier.verify_directory(bundle)["verified"]
    with pytest.raises(verifier.EvidenceVerificationError, match="current source changed"):
        verifier.verify_directory(bundle, check_current_sources=True)
