#!/usr/bin/env python3
"""Verify and exactly re-execute a receipt-gated Minecraft neural trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from neurocraft_connectome import RuntimeGraph, load_malecns_fixture
from neurocraft_connectome.flight_pin import (
    MALECNS_FLIGHT_FIXTURE_V1_FILE_BYTES,
    MALECNS_FLIGHT_FIXTURE_V1_PIN,
)
from neurocraft_controller import (
    WING_MOTOR_POOL_PROFILE,
    ConnectomeController,
    ControllerConfiguration,
    FlightConnectomeController,
    FlightControllerConfiguration,
    world_state_from_observation,
)
from neurocraft_controller.trained_readout_controller import (
    TrainedReadoutController,
    TrmDemoConfiguration,
)
from neurocraft_protocol import (
    CausalRecord,
    ClaimScope,
    DataTier,
    ValidationResult,
    canonical_json_bytes,
    parse_json_frame,
    verify_trace,
)

# Direct script execution puts ``scripts/`` rather than the repository root at
# the front of sys.path. The execution-bundle contract lives in the preparer and
# must be shared rather than duplicated by the verifier.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_gametest_demo import (
    PAIR_CONTROLLER_PROFILE,
    TRM_CLONED_READOUT_PROFILE,
    TRM_READOUT_ARTIFACT_PATH,
    PreparationError,
    _code_bundle,
    _dependency_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "data" / "fixtures" / "mcns-v1.0-flight-power-v1.json"
CONTROLLER_PROFILES = (
    PAIR_CONTROLLER_PROFILE,
    WING_MOTOR_POOL_PROFILE,
    TRM_CLONED_READOUT_PROFILE,
)


def _load_replay_graph(fixture: Path, controller_profile: str) -> RuntimeGraph:
    """Load the fixture under the identity pin the selected profile declares."""

    if controller_profile in {WING_MOTOR_POOL_PROFILE, TRM_CLONED_READOUT_PROFILE}:
        return load_malecns_fixture(
            fixture,
            expected_pin=MALECNS_FLIGHT_FIXTURE_V1_PIN,
            expected_file_bytes=MALECNS_FLIGHT_FIXTURE_V1_FILE_BYTES,
        )
    return load_malecns_fixture(fixture)


def _read_trm_readout() -> tuple[bytes, dict[str, Any], str]:
    """Read, parse, and hash the committed readout artifact; the digest is
    always derived from the file at call time, never pinned here."""

    try:
        raw = TRM_READOUT_ARTIFACT_PATH.read_bytes()
    except OSError as exc:
        raise ValueError(f"could not read the committed readout artifact: {exc.strerror}") from exc
    try:
        parsed: object = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("committed readout artifact is not valid JSON") from exc
    if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
        raise ValueError("committed readout artifact must be a JSON object")
    return raw, cast(dict[str, Any], parsed), hashlib.sha256(raw).hexdigest()


def _build_replay_controller(
    graph: RuntimeGraph,
    *,
    controller_profile: str,
    run_id: str,
    entity_id: str,
    session_epoch: int,
    sign_policy: str | None = None,
) -> ConnectomeController:
    """Construct the exact replay controller (or its zero-recurrence twin)."""

    if controller_profile == WING_MOTOR_POOL_PROFILE:
        return FlightConnectomeController(
            graph,
            run_id=run_id,
            entity_id=entity_id,
            session_epoch=session_epoch,
            configuration=(
                FlightControllerConfiguration()
                if sign_policy is None
                else FlightControllerConfiguration(sign_policy=sign_policy)
            ),
        )
    if controller_profile == TRM_CLONED_READOUT_PROFILE:
        artifact_bytes, artifact_document, artifact_sha256 = _read_trm_readout()
        return TrainedReadoutController(
            graph,
            run_id=run_id,
            entity_id=entity_id,
            session_epoch=session_epoch,
            artifact_bytes=artifact_bytes,
            artifact_document=artifact_document,
            configuration=(
                TrmDemoConfiguration(readout_artifact_sha256=artifact_sha256)
                if sign_policy is None
                else TrmDemoConfiguration(
                    sign_policy=sign_policy,
                    readout_artifact_sha256=artifact_sha256,
                )
            ),
        )
    return ConnectomeController.create(
        graph,
        run_id=run_id,
        entity_id=entity_id,
        session_epoch=session_epoch,
        configuration=(
            None if sign_policy is None else ControllerConfiguration(sign_policy=sign_policy)
        ),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_records(path: Path) -> Iterator[CausalRecord]:
    with path.open("rb") as stream:
        for raw in stream:
            envelope = parse_json_frame(raw.rstrip(b"\n"))
            if envelope.get("kind") == "record":
                yield CausalRecord.from_dict(envelope["payload"])


def parse_stimulation_declarations(values: Sequence[str]) -> dict[str, dict[int, float]]:
    declarations: dict[str, dict[int, float]] = {}
    for declaration in values:
        try:
            content_id, raw_assignments = declaration.split("=", 1)
        except ValueError as exc:
            raise ValueError("stimulation declarations must use ID=BODY_ID=VALUE") from exc
        if not content_id or content_id in declarations:
            raise ValueError("stimulation IDs must be non-empty and unique")
        assignments: dict[int, float] = {}
        for assignment in raw_assignments.split(","):
            try:
                body_text, value_text = assignment.split("=", 1)
                body_id = int(body_text)
                value = float(value_text)
            except ValueError as exc:
                raise ValueError(
                    "stimulation declarations must use ID=BODY_ID=VALUE[,BODY_ID=VALUE]"
                ) from exc
            if body_id in assignments:
                raise ValueError(f"duplicate body ID in stimulation {content_id}")
            assignments[body_id] = value
        if not assignments:
            raise ValueError(f"stimulation {content_id} has no assignments")
        declarations[content_id] = assignments
    return declarations


def verify_world_transition_lineage(
    records: Sequence[CausalRecord],
) -> tuple[int, int, int, int]:
    """Cross-link receipt pre-states and exact next-tick post-states."""

    indexed_observations: dict[tuple[int, int], tuple[int, CausalRecord]] = {}
    observations_by_tick: dict[int, tuple[int, CausalRecord]] = {}
    for index, record in enumerate(records):
        key = (record.observation.observation_seq, record.observation.game_tick)
        if key in indexed_observations:
            raise ValueError(f"duplicate trace observation lineage at record {record.record_seq}")
        if record.observation.game_tick in observations_by_tick:
            raise ValueError(f"duplicate trace observation tick at record {record.record_seq}")
        indexed_observations[key] = (index, record)
        observations_by_tick[record.observation.game_tick] = (index, record)

    final_observation = records[-1].observation
    world_before_cross_linked = 0
    terminal_world_before_receipt_only = 0
    world_after_cross_linked = 0
    terminal_world_after_not_cross_linked = 0
    for index, record in enumerate(records):
        applied_tick = record.game_tick + record.observation_age_ticks
        if record.world_after_game_tick != applied_tick + 1:
            raise ValueError(f"world-after tick is not the next tick at record {record.record_seq}")

        pre_actuation = observations_by_tick.get(applied_tick)
        if pre_actuation is not None:
            pre_index, pre_record = pre_actuation
            if pre_index <= index:
                raise ValueError(
                    f"world-before lineage does not move forward at record {record.record_seq}"
                )
            if record.world_before != world_state_from_observation(pre_record.observation):
                raise ValueError(f"world-before lineage mismatch at record {record.record_seq}")
            world_before_cross_linked += 1
        elif applied_tick > final_observation.game_tick:
            terminal_world_before_receipt_only += 1
        else:
            raise ValueError(
                f"world-before lineage is missing within trace at record {record.record_seq}"
            )

        lineage = (
            record.world_after_observation_seq,
            record.world_after_game_tick,
        )
        target = indexed_observations.get(lineage)
        if target is not None:
            target_index, target_record = target
            if target_index <= index:
                raise ValueError(
                    f"world-after lineage does not move forward at record {record.record_seq}"
                )
            if record.world_after != world_state_from_observation(target_record.observation):
                raise ValueError(f"world-after lineage mismatch at record {record.record_seq}")
            world_after_cross_linked += 1
            continue

        is_beyond_trace_tail = (
            record.world_after_observation_seq > final_observation.observation_seq
            and record.world_after_game_tick > final_observation.game_tick
        )
        if not is_beyond_trace_tail:
            raise ValueError(
                f"world-after lineage is missing within trace at record {record.record_seq}"
            )
        terminal_world_after_not_cross_linked += 1

    return (
        world_before_cross_linked,
        terminal_world_before_receipt_only,
        world_after_cross_linked,
        terminal_world_after_not_cross_linked,
    )


def replay_minecraft_trace(
    *,
    fixture: Path,
    trace: Path,
    stimulations: Mapping[str, Mapping[int, float]],
    controller_profile: str = PAIR_CONTROLLER_PROFILE,
) -> dict[str, Any]:
    if controller_profile not in CONTROLLER_PROFILES:
        raise ValueError(f"unknown controller profile: {controller_profile!r}")
    verification = verify_trace(trace)
    manifest = verification.manifest
    if manifest.claim_scope is not ClaimScope.G3:
        raise ValueError("Minecraft replay requires a G3-labelled receipt trace")
    if manifest.data_tier not in {DataTier.MALECNS_FIXTURE, DataTier.MALECNS_CIRCUIT}:
        raise ValueError("this replay command supports only MaleCNS fixture/circuit traces")
    labels = (manifest.java, manifest.minecraft, manifest.fabric, manifest.mod)
    if any(not value.strip() or "not_used" in value.lower() for value in labels):
        raise ValueError("trace is not explicitly labelled with the Minecraft runtime tuple")

    records = tuple(iter_records(trace))
    if not records or len(records) != verification.record_count:
        raise ValueError("trace has no records or its record count is inconsistent")
    graph = _load_replay_graph(fixture, controller_profile)
    first = records[0]
    if (
        first.command_seq != 0
        or first.observation.observation_seq != 0
        or first.neural_step_start != 0
        or any(first.state_before_activity or ())
    ):
        raise ValueError(
            "Minecraft trace must begin at the acknowledged pre-spawn recording boundary"
        )
    controller = _build_replay_controller(
        graph,
        controller_profile=controller_profile,
        run_id=first.identity.run_id,
        entity_id=first.identity.entity_id,
        session_epoch=first.identity.session_epoch,
    )
    if controller.identity != first.identity:
        raise ValueError("current controller identities do not match the trace")

    artifacts = manifest.artifacts
    # Both bundles are enumerated from the Git index (D-072) and refuse an
    # untracked file inside a hashed root; that belongs with this function's
    # other identity refusals, which the CLI reports as a message.
    try:
        current_dependency_hash, _ = _dependency_bundle()
        current_code_hash, _ = _code_bundle()
    except PreparationError as exc:
        raise ValueError(str(exc)) from exc
    expected = {
        "source_hash": graph.manifest.source_manifest_sha256,
        "canonical_graph_hash": graph.manifest.canonical_graph_sha256,
        "runtime_graph_hash": graph.manifest.runtime_identity_sha256,
        "fixture_hash": graph.manifest.fixture_file_sha256,
        "code_hash": current_code_hash,
        "dependency_lock_hash": current_dependency_hash,
        "adapter_hash": controller.identity.adapter_hash,
        "dynamics_hash": controller.identity.dynamics_hash,
        "sign_hash": controller.sign_overlay.identity_sha256,
        "decoder_hash": controller.identity.decoder_hash,
        "configuration_hash": controller.configuration.identity_sha256,
        "observation_schema_hash": controller.identity.observation_schema_hash,
    }
    for name, value in expected.items():
        if getattr(artifacts, name) != value:
            raise ValueError(f"trace {name} does not match the current replay environment")

    (
        world_before_cross_linked,
        terminal_world_before_receipt_only,
        world_transitions_cross_linked,
        terminal_world_transitions_not_cross_linked,
    ) = verify_world_transition_lineage(records)
    active_stimulation: str | None = None
    stimulated_records = 0
    for record in records:
        if record.identity != controller.identity:
            raise ValueError(f"identity changed at record {record.record_seq}")
        if record.validation is not ValidationResult.APPLIED or record.command_seq is None:
            raise ValueError("exact neural replay requires receipt-validated applied records")
        if record.fault_code is not None:
            raise ValueError("exact neural replay refuses fault-bearing records")
        declared = tuple(record.interventions)
        unknown = set(declared).difference(stimulations)
        if unknown:
            raise ValueError(f"undeclared intervention in trace: {sorted(unknown)}")
        if len(declared) > 1:
            raise ValueError("v2 replay supports at most one concurrent named stimulation")
        desired = declared[0] if declared else None
        if desired != active_stimulation:
            if desired is None:
                controller.clear_stimulation()
            else:
                controller.set_stimulation(desired, stimulations[desired])
            active_stimulation = desired
        if desired is not None:
            stimulated_records += 1

        if controller.engine.current_state().state_sha256 != record.state_before_digest:
            raise ValueError(f"state-before mismatch at record {record.record_seq}")
        if controller.engine.current_state().activity != record.state_before_activity:
            raise ValueError(f"full state-before mismatch at record {record.record_seq}")
        action = controller.step(record.observation, command_seq=record.command_seq)
        evidence = controller.last_evidence
        if evidence is None:
            raise ValueError(f"controller evidence missing at record {record.record_seq}")
        checks = (
            action == record.issued_action,
            action.state_digest == record.state_after_digest,
            action.motors == record.decoder_output,
            action.neural_step_start == record.neural_step_start,
            action.neural_step_end == record.neural_step_end,
            evidence.state_before_sha256 == record.state_before_digest,
            evidence.state_after_sha256 == record.state_after_digest,
            dict(evidence.encoded_stimulation) == dict(record.encoded_stimulation),
            dict(evidence.direct_stimulation) == dict(record.direct_stimulation),
            evidence.direct_stimulation_sha256 == record.direct_stimulation_sha256,
            evidence.state_after_activity == record.state_after_activity,
            dict(evidence.motor_frame.values) == dict(record.motor_pool_values),
            evidence.neural_step_records[-1].delay_after_sha256 == record.delay_state_digest,
            controller.decoder.identity_sha256 == record.pool_membership_hash,
            evidence.interventions == record.interventions,
        )
        if not all(checks):
            raise ValueError(f"causal neural replay mismatch at record {record.record_seq}")

    zero_controller = _build_replay_controller(
        graph,
        controller_profile=controller_profile,
        run_id=first.identity.run_id,
        entity_id=first.identity.entity_id,
        session_epoch=first.identity.session_epoch,
        sign_policy="unknown_zero",
    )
    zero_active_stimulation: str | None = None
    zero_state_divergences = 0
    zero_action_divergences = 0
    zero_neutral_actions = 0
    for record in records:
        declared = tuple(record.interventions)
        desired = declared[0] if declared else None
        if desired != zero_active_stimulation:
            if desired is None:
                zero_controller.clear_stimulation()
            else:
                zero_controller.set_stimulation(desired, stimulations[desired])
            zero_active_stimulation = desired
        command_seq = record.command_seq
        if command_seq is None:
            raise ValueError("APPLIED trace record is missing command_seq")
        counterfactual = zero_controller.step(
            replace(record.observation, identity=zero_controller.identity),
            command_seq=command_seq,
        )
        zero_state_divergences += counterfactual.state_digest != record.state_after_digest
        zero_action_divergences += counterfactual.motors != record.decoder_output
        zero_neutral_actions += counterfactual.motors.is_neutral
    if zero_state_divergences < 1 or zero_action_divergences < 1:
        raise ValueError(
            "zero-recurrence counterfactual did not alter both model state and motor output"
        )
    return {
        "schema": "neurocraft.minecraft_trace_replay.v1",
        "status": "exact_neural_match",
        "replay_implementation_sha256": sha256_file(Path(__file__)),
        "evidence_boundary": (
            "Same-build exact neural re-execution of receipt-gated Minecraft observations "
            "through the pinned MaleCNS fixture controller. Receipt-sampled pre-actuation "
            "states and exact next-tick world-after states are cross-linked to in-trace "
            "observations when present; terminal lineages remain hash-authenticated but are "
            "not independently cross-linked. Minecraft physics is not re-simulated by Python."
        ),
        "trace_file_sha256": verification.file_sha256,
        "trace_final_hash": verification.final_hash,
        "manifest_hash": verification.manifest_hash,
        "record_count": len(records),
        "first_tick": verification.first_tick,
        "last_tick": verification.last_tick,
        "stimulated_record_count": stimulated_records,
        "zero_recurrence_counterfactual": {
            "mode": "unknown_zero_sign_overlay",
            "record_count": len(records),
            "state_divergent_record_count": zero_state_divergences,
            "action_divergent_record_count": zero_action_divergences,
            "neutral_action_record_count": zero_neutral_actions,
            "evidence_boundary": (
                "The exact recorded Minecraft observations and named interventions were rerun "
                "from the same initial state with every fixture edge's unknown sign mapped to "
                "zero. This is a software anti-bypass counterfactual, not a biological sign model."
            ),
        },
        "world_before_cross_linked": world_before_cross_linked,
        "terminal_world_before_receipt_only": terminal_world_before_receipt_only,
        "world_transitions_cross_linked": world_transitions_cross_linked,
        "terminal_world_transitions_not_cross_linked": (
            terminal_world_transitions_not_cross_linked
        ),
        "final_state_sha256": records[-1].state_after_digest,
        "execution_code_bundle_sha256": current_code_hash,
        "dependency_bundle_sha256": current_dependency_hash,
    }


def write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(receipt) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--controller-profile",
        choices=CONTROLLER_PROFILES,
        default=PAIR_CONTROLLER_PROFILE,
        help=(
            "pair_v1 keeps the shipped v1 replay controller; wing_motor_soma_side_v1 "
            "replays through the flight controller over the pinned steering fixture; "
            "trm_cloned_readout_v1 replays through the trained-readout demo controller "
            "bound to the committed cloned-readout artifact"
        ),
    )
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--stimulation",
        action="append",
        default=[],
        metavar="ID=BODY_ID=VALUE[,BODY_ID=VALUE]",
    )
    parser.add_argument("--receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        declarations = parse_stimulation_declarations(args.stimulation)
        receipt = replay_minecraft_trace(
            fixture=args.fixture,
            trace=args.trace,
            stimulations=declarations,
            controller_profile=args.controller_profile,
        )
        if args.receipt is not None:
            write_receipt(args.receipt, receipt)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    json.dump(receipt, sys.stdout, allow_nan=False, separators=(",", ":"), sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
