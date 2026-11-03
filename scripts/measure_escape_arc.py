#!/usr/bin/env python3
"""Measure the escape-arc feasibility quantities under committed evidence.

Phase-2 item 1 of the frozen escape identity
(``experiments/preregistration/eem-escape-v1.md``, D-061): re-derive the
D-060 scratchpad feasibility numbers from the frozen tree as
content-addressed evidence for ``docs/receipts/ESCAPE-ARC-01.md``, executing
the pinned method exactly:

* 30-tick holds from fresh reset on the CALIBRATED escape controller for
  baseline all-open (rays 6.0, light 15, no collision), blocked-front
  (front ray 0.5), and all-rays-0.5, recording per-tick DNp01 L/R and TTMn
  L/R activity, the LC4 population mean, and the wing pool means;
* a 150-step direct-drive diagnostic (LC4 = 1.0 dense stimulation on a raw
  engine, labeled diagnostic-only);
* the four quantities exactly as the frozen document defines them (DNp01
  response ratio, TTMn attenuation, TTMn event contrast, arc latency).

The DNp01 read here is offline and diagnostic-only: the live decoder never
reads DNp01, and this measurement does not weaken that claim boundary. The
frozen document declares no pass/fail criterion; whatever this script
measures is recorded.

Because the pinned method names the calibrated escape controller, the
phase-2 calibration (item 2) is necessarily measured before this receipt;
the executed order is recorded in the evidence.

The canonical evidence JSON contains no timestamps or environment data; wall
time is printed and belongs to the environment-scoped prose of the receipt,
never to the hashed payload, so a rerun must reproduce the evidence
byte-identically (the ``write_or_verify`` discipline of
``scripts/measure_full_sign_overlay.py``).

Offline dynamics measurement over declared boundaries only — no behavior,
performance, or biological claim.
"""

from __future__ import annotations

import argparse
import math
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from neurocraft_controller.config import canonical_sha256
from neurocraft_controller.embodied_senses import (
    _MISSING_ANNOTATION_ROW,
    SIDE_COLUMN,
    TYPE_COLUMN,
    _string_column,
    _verified_annotation_table,
    _verified_graph_array,
    load_embodied_population_binding,
)
from neurocraft_controller.escape_reflex import load_escape_ttmn_binding
from neurocraft_controller.full_brain_controller import (
    FULL_MCNS_ARTIFACT_MANIFEST_SHA256,
    FullBrainController,
    load_escape_readout_calibration,
)
from neurocraft_protocol import Observation, SensorFrame, canonical_json_bytes

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIRECTORY = REPO_ROOT / "data" / "processed" / "male-cns-v1.0-superclass-v1"
DEFAULT_ANNOTATIONS = (
    REPO_ROOT / "data" / "cache" / "male-cns-v1.0"
) / "body-neurotransmitters-male-cns-v1.0.feather"
DEFAULT_BODY_ANNOTATIONS = (
    REPO_ROOT / "data" / "cache" / "male-cns-v1.0"
) / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
DEFAULT_CALIBRATION = REPO_ROOT / "data" / "calibration" / "full-readout-calibration-escape-v1.json"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "analysis" / "escape-arc-v1.json"

RECEIPT_ID = "ESCAPE-ARC-01"
HOLD_TICKS = 30
SUMMARY_WINDOW_TICKS = 10
DIRECT_DRIVE_STEPS = 150
RAY_MAX_DISTANCE_M = 6.0
BLOCKED_RAY_M = 0.5
GAME_TICK_NS = 50_000_000

DNP01_TYPE = "DNp01"
EXPECTED_DNP01_NODES = 2


class EscapeArcMeasurementError(RuntimeError):
    """The escape-arc evidence could not be produced or verified exactly."""


def _float_pair(value: float) -> dict[str, Any]:
    """A measured float as both its exact hex and its decimal form."""

    return {"float_hex": float(value).hex(), "value": float(value)}


def _hex_series(values: list[float]) -> dict[str, Any]:
    """A per-tick series: exact float hexes plus a digest over them."""

    hexes = [float(value).hex() for value in values]
    return {
        "float_hex": hexes,
        "series_sha256": sha256("\n".join(hexes).encode() + b"\n").hexdigest(),
    }


def select_dnp01_pair(
    graph: Any, body_annotations_path: Path
) -> tuple[tuple[int, int, str], tuple[int, int, str]]:
    """The DNp01 giant-fiber pair, via the same verified join the selectors use.

    Fail-closed: exactly two ``type == 'DNp01'`` rows, somaSide split exactly
    1 L / 1 R, dense order verified. Returned sorted by body ID as
    ``(body_id, dense_index, soma_side)`` tuples. Diagnostic-only: nothing in
    the live decoder reads these indices.
    """

    table, _annotations_sha256 = _verified_annotation_table(graph, body_annotations_path)
    annotation_rows = _verified_graph_array(graph, "annotation_source_rows", "<u8")
    body_ids = _verified_graph_array(graph, "body_ids", "<u8")
    if bool((annotation_rows == np.uint64(_MISSING_ANNOTATION_ROW)).any()):
        raise EscapeArcMeasurementError("a retained node has no annotation row")
    rows = annotation_rows.astype(np.int64)
    if int(rows.min()) < 0 or int(rows.max()) >= int(table.num_rows):
        raise EscapeArcMeasurementError("annotation source row is out of table range")
    if np.unique(rows).size != rows.size:
        raise EscapeArcMeasurementError("annotation source rows are not unique")
    taken = table.take(rows)
    taken_bodies = np.asarray(taken.column("bodyId").to_pylist(), dtype=np.uint64)
    if not np.array_equal(taken_bodies, body_ids):
        raise EscapeArcMeasurementError("bodyId join is not bitwise identical")
    types = _string_column(taken, TYPE_COLUMN)
    sides = _string_column(taken, SIDE_COLUMN)
    members: list[tuple[int, int, str]] = []
    for position, type_value in enumerate(types):
        if type_value != DNP01_TYPE:
            continue
        body_id = int(body_ids[position])
        dense_index = int(graph.index_of(body_id))
        if dense_index != position:
            raise EscapeArcMeasurementError("DNp01 annotation rows are not in dense body order")
        members.append((body_id, dense_index, str(sides[position])))
    if len(members) != EXPECTED_DNP01_NODES:
        raise EscapeArcMeasurementError(
            f"DNp01 count drifted: selected {len(members)} nodes, expected exactly "
            f"{EXPECTED_DNP01_NODES}"
        )
    members.sort(key=lambda member: member[0])
    if sorted(side for _, _, side in members) != ["L", "R"]:
        raise EscapeArcMeasurementError("DNp01 somaSide split drifted: expected exactly 1 L / 1 R")
    return members[0], members[1]


def _condition_sensors(name: str) -> SensorFrame:
    """The three pinned encoder-path conditions, held constant per tick."""

    rays = {
        "baseline_all_open": (RAY_MAX_DISTANCE_M,) * 6,
        "blocked_front": (
            BLOCKED_RAY_M,
            RAY_MAX_DISTANCE_M,
            RAY_MAX_DISTANCE_M,
            RAY_MAX_DISTANCE_M,
            RAY_MAX_DISTANCE_M,
            RAY_MAX_DISTANCE_M,
        ),
        "all_rays_blocked": (BLOCKED_RAY_M,) * 6,
    }[name]
    return SensorFrame(
        position_m=(0.0, 66.0, 0.0),
        velocity_m_s=(0.0, 0.0, 0.0),
        yaw_rad=0.0,
        pitch_rad=0.0,
        on_ground=False,
        horizontal_collision=False,
        vertical_collision=False,
        in_fluid=False,
        local_light=15,
        ray_max_distance_m=RAY_MAX_DISTANCE_M,
        ray_distances_m=rays,
    )


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def measure_conditions(
    controller: FullBrainController,
    *,
    dnp01: tuple[tuple[int, int, str], tuple[int, int, str]],
    ttmn_by_side: dict[str, int],
    lc4_indices: np.ndarray[Any, np.dtype[np.int64]],
) -> dict[str, dict[str, list[float]]]:
    """Hold each pinned condition 30 ticks from a fresh reset; record series."""

    dnp01_by_side = {side: dense_index for _, dense_index, side in dnp01}
    series: dict[str, dict[str, list[float]]] = {}
    tick = 0
    for condition in ("baseline_all_open", "blocked_front", "all_rays_blocked"):
        controller.reset()
        sensors = _condition_sensors(condition)
        recorded: dict[str, list[float]] = {
            "dnp01_left": [],
            "dnp01_right": [],
            "dnp01_pair_mean": [],
            "ttmn_left": [],
            "ttmn_right": [],
            "ttmn_pool_mean": [],
            "lc4_population_mean": [],
            "wing_left_mean": [],
            "wing_right_mean": [],
            "wing_pool_mean": [],
        }
        for _ in range(HOLD_TICKS):
            controller.step(
                Observation(
                    identity=controller.identity,
                    observation_seq=tick,
                    game_tick=tick,
                    sent_monotonic_ns=tick * GAME_TICK_NS,
                    sensors=sensors,
                ),
                command_seq=tick,
            )
            tick += 1
            activity = controller.engine.activity()
            left = float(activity[dnp01_by_side["L"]])
            right = float(activity[dnp01_by_side["R"]])
            recorded["dnp01_left"].append(left)
            recorded["dnp01_right"].append(right)
            recorded["dnp01_pair_mean"].append((left + right) / 2.0)
            recorded["ttmn_left"].append(float(activity[ttmn_by_side["L"]]))
            recorded["ttmn_right"].append(float(activity[ttmn_by_side["R"]]))
            recorded["ttmn_pool_mean"].append(controller.jump_readout_mean())
            recorded["lc4_population_mean"].append(
                float(np.mean(activity[lc4_indices], dtype=np.float64))
            )
            wing_means = dict(controller.pool_readout_means())
            recorded["wing_left_mean"].append(wing_means["left_motor"])
            recorded["wing_right_mean"].append(wing_means["right_motor"])
            recorded["wing_pool_mean"].append(
                (wing_means["left_motor"] + wing_means["right_motor"]) / 2.0
            )
        series[condition] = recorded
    return series


def measure_direct_drive(
    controller: FullBrainController,
    *,
    dnp01: tuple[tuple[int, int, str], tuple[int, int, str]],
    lc4_indices: np.ndarray[Any, np.dtype[np.int64]],
) -> dict[str, Any]:
    """The 150-step LC4 = 1.0 direct-drive diagnostic on the raw engine.

    Diagnostic-only: the engine is stepped directly with a dense stimulation
    vector — no encoder, no decode, no controller clocks — from a fresh
    reset, to time the anatomical arc in neural steps.
    """

    controller.reset()
    engine = controller.engine
    node_count = engine.view.node_count
    vector = np.zeros(node_count, dtype=np.float64)
    vector[lc4_indices] = 1.0
    provenance = canonical_sha256(
        {
            "amplitude_float_hex": (1.0).hex(),
            "lc4_member_count": int(lc4_indices.size),
            "note": (
                "diagnostic-only direct LC4 drive; not an encoder path and not "
                "part of any live profile"
            ),
            "schema": "neurocraft.escape_arc.direct_drive.v1",
        }
    )
    binding = controller.escape_binding
    assert binding is not None
    ttmn_indices = [member.dense_index for member in binding.jump_pool.members]
    dnp01_indices = [dense_index for _, dense_index, _ in dnp01]
    dnp01_series: list[float] = []
    ttmn_series: list[float] = []
    first_dnp01_step = 0
    first_ttmn_step = 0
    for step in range(1, DIRECT_DRIVE_STEPS + 1):
        engine.step(sensory=vector, sensory_provenance_sha256=provenance)
        activity = engine.activity()
        dnp01_mean = float(math.fsum(float(activity[i]) for i in dnp01_indices) / 2.0)
        ttmn_mean = float(math.fsum(float(activity[i]) for i in ttmn_indices) / 2.0)
        dnp01_series.append(dnp01_mean)
        ttmn_series.append(ttmn_mean)
        if first_dnp01_step == 0 and dnp01_mean != 0.0:
            first_dnp01_step = step
        if first_ttmn_step == 0 and ttmn_mean != 0.0:
            first_ttmn_step = step
    if first_ttmn_step == 0:
        raise EscapeArcMeasurementError(
            "the direct-drive diagnostic produced no nonzero TTMn response in "
            f"{DIRECT_DRIVE_STEPS} steps; the arc latency is undefined"
        )
    return {
        "claim_scope": "diagnostic_only_offline_direct_drive_no_live_profile_reads_dnp01",
        "steps": DIRECT_DRIVE_STEPS,
        "stimulation": {
            "amplitude": _float_pair(1.0),
            "lc4_member_count": int(lc4_indices.size),
            "provenance_sha256": provenance,
        },
        "lc4_commit_step": 1,
        "first_nonzero_dnp01_pair_mean_step": first_dnp01_step,
        "first_nonzero_ttmn_pool_mean_step": first_ttmn_step,
        "dnp01_pair_mean_per_step": _hex_series(dnp01_series),
        "ttmn_pool_mean_per_step": _hex_series(ttmn_series),
        "neural_steps_per_game_tick": 5,
    }


def derive_quantities(
    conditions: dict[str, dict[str, list[float]]], direct: dict[str, Any]
) -> dict[str, Any]:
    """The four pinned quantities, with every interpretive choice disclosed."""

    def window_mean(condition: str, series_name: str) -> float:
        return _mean(conditions[condition][series_name][-SUMMARY_WINDOW_TICKS:])

    baseline_dnp01 = window_mean("baseline_all_open", "dnp01_pair_mean")
    baseline_wing = window_mean("baseline_all_open", "wing_pool_mean")
    baseline_ttmn_abs = abs(window_mean("baseline_all_open", "ttmn_pool_mean"))

    response_ratios: dict[str, Any] = {}
    event_contrasts: dict[str, Any] = {}
    for condition in ("blocked_front", "all_rays_blocked"):
        dnp01_delta = abs(window_mean(condition, "dnp01_pair_mean") - baseline_dnp01)
        wing_delta = abs(window_mean(condition, "wing_pool_mean") - baseline_wing)
        response_ratios[condition] = {
            "dnp01_pair_delta_abs": _float_pair(dnp01_delta),
            "wing_pool_delta_abs": _float_pair(wing_delta),
            "ratio": _float_pair(dnp01_delta / wing_delta),
        }
        peak = max(abs(value) for value in conditions[condition]["ttmn_pool_mean"])
        event_contrasts[condition] = {
            "ttmn_pool_abs_peak": _float_pair(peak),
            "baseline_ttmn_pool_abs": _float_pair(baseline_ttmn_abs),
            "contrast": _float_pair(peak / baseline_ttmn_abs),
        }
    attenuation_ttmn = window_mean("all_rays_blocked", "ttmn_pool_mean")
    attenuation_dnp01 = window_mean("all_rays_blocked", "dnp01_pair_mean")
    latency_steps = int(direct["first_nonzero_ttmn_pool_mean_step"]) - int(
        direct["lc4_commit_step"]
    )
    return {
        "summary_window": {
            "definition": (
                "condition summaries are means over the last "
                f"{SUMMARY_WINDOW_TICKS} of the {HOLD_TICKS} held ticks, the "
                "window the frozen document names for the DNp01 response "
                "ratio; the same window is applied to the attenuation and to "
                "the baselines, and that uniform choice is disclosed here "
                "because the frozen text names the window explicitly only for "
                "the ratio"
            ),
            "ticks": SUMMARY_WINDOW_TICKS,
        },
        "dnp01_response_ratio": {
            "definition": (
                "|mean DNp01 pair activity over the last 10 ticks, condition "
                "minus baseline| / |same for the wing-pool mean| (wing-pool "
                "mean = mean of the two per-tick pool means)"
            ),
            "per_condition": response_ratios,
        },
        "ttmn_attenuation": {
            "definition": (
                "TTMn-pool mean / DNp01-pair mean at the all-rays condition "
                "(both over the summary window)"
            ),
            "ttmn_pool_mean": _float_pair(attenuation_ttmn),
            "dnp01_pair_mean": _float_pair(attenuation_dnp01),
            "attenuation": _float_pair(attenuation_ttmn / attenuation_dnp01),
        },
        "ttmn_event_contrast": {
            "definition": (
                "TTMn-pool |mean| at condition peak (max over the condition's "
                "held ticks of the per-tick |pool mean|) / baseline TTMn-pool "
                "|mean| (|summary-window mean| of the baseline hold)"
            ),
            "per_condition": event_contrasts,
        },
        "arc_latency": {
            "definition": (
                "neural steps from LC4 commit (the first direct-drive step "
                "with the stimulation applied) to the first step with a "
                "nonzero TTMn pool mean, in the direct phase"
            ),
            "neural_steps": latency_steps,
            "within_one_game_tick": latency_steps < 5,
        },
    }


def measure(
    *,
    artifact_directory: Path,
    artifact_manifest_sha256: str,
    annotations_path: Path,
    body_annotations_path: Path,
    calibration_path: Path,
) -> dict[str, Any]:
    """Produce the canonical, environment-free escape-arc evidence document."""

    from neurocraft_connectome.full_runtime import load_full_runtime_artifact

    calibration = load_escape_readout_calibration(calibration_path.read_bytes())
    graph = load_full_runtime_artifact(
        artifact_directory, expected_manifest_sha256=artifact_manifest_sha256
    )
    try:
        populations = load_embodied_population_binding(graph, body_annotations_path)
        ttmn = load_escape_ttmn_binding(graph, body_annotations_path, populations)
        dnp01 = select_dnp01_pair(graph, body_annotations_path)
    finally:
        graph.close()
    controller = FullBrainController.from_artifact(
        run_id="escape-arc-measurement",
        entity_id="escape-arc-fly",
        artifact_directory=artifact_directory,
        annotations_path=annotations_path,
        calibration_path=calibration_path,
        expected_manifest_sha256=artifact_manifest_sha256,
        embodied_senses=True,
        body_annotations_path=body_annotations_path,
        escape_reflex=True,
    )
    binding = controller.escape_binding
    senses = controller.embodied_senses
    assert binding is not None and senses is not None
    if binding.ttmn_pool_identity_sha256 != ttmn.identity_sha256:
        raise EscapeArcMeasurementError("controller TTMn binding does not match the selection")
    if senses.populations.identity_sha256 != populations.identity_sha256:
        raise EscapeArcMeasurementError("controller populations do not match the selection")
    if controller.configuration.readout_calibration_sha256 != calibration.document_sha256:
        raise EscapeArcMeasurementError("controller calibration does not match the document")
    lc4_indices = np.asarray(
        populations.population("lc4_left").dense_indices
        + populations.population("lc4_right").dense_indices,
        dtype=np.int64,
    )
    ttmn_by_side = {member.soma_side: member.dense_index for member in ttmn.members}
    conditions = measure_conditions(
        controller, dnp01=dnp01, ttmn_by_side=ttmn_by_side, lc4_indices=lc4_indices
    )
    direct = measure_direct_drive(controller, dnp01=dnp01, lc4_indices=lc4_indices)
    controller.engine.end_run()
    quantities = derive_quantities(conditions, direct)
    return {
        "receipt_id": RECEIPT_ID,
        "claim_scope": (
            "offline_dynamics_measurement_over_declared_boundaries_only"
            "_no_behavior_performance_or_biological_claim"
        ),
        "identity_document": "experiments/preregistration/eem-escape-v1.md",
        "executed_order": (
            "phase-2 item 2 (boundary calibration) was measured before this "
            "receipt because the frozen item-1 method names the CALIBRATED "
            "escape controller; the calibration is itself a measurement the "
            "frozen document orders"
        ),
        "artifact": {
            "manifest_file_sha256": controller.identity.graph_hash,
            "node_count": controller.engine.view.node_count,
            "edge_count": controller.engine.view.edge_count,
            "runtime_view_sha256": controller.engine.view.view_sha256,
        },
        "sign_overlay": {
            "policy": controller.configuration.sign_policy,
            "identity_sha256": controller.sign_overlay_identity_sha256,
        },
        "calibration": {
            "document": "data/calibration/full-readout-calibration-escape-v1.json",
            "document_sha256": calibration.document_sha256,
            "wing_scale": _float_pair(calibration.wing_scale),
            "jump_scale": _float_pair(calibration.jump_scale),
            "stream_id": calibration.stream_id,
        },
        "controller_identity": {
            "profile": "full_mcns_escape_v1",
            "adapter_hash": controller.identity.adapter_hash,
            "decoder_hash": controller.identity.decoder_hash,
            "dynamics_hash": controller.identity.dynamics_hash,
            "observation_schema_hash": controller.identity.observation_schema_hash,
            "configuration_identity_sha256": controller.configuration.identity_sha256,
            "encoder_identity_sha256": senses.encoder.identity_sha256,
            "population_binding_sha256": senses.populations.identity_sha256,
            "ttmn_pool_binding_sha256": binding.ttmn_pool_identity_sha256,
        },
        "selections": {
            "dnp01_pair": {
                "selector": f"{TYPE_COLUMN} == '{DNP01_TYPE}'",
                "members": [[body_id, dense_index, side] for body_id, dense_index, side in dnp01],
                "read_scope": ("offline diagnostic-only: the live decoder never reads DNp01"),
            },
            "ttmn_pool": {
                "members": [
                    [member.body_id, member.dense_index, member.soma_side]
                    for member in ttmn.members
                ],
            },
            "lc4_member_count": int(lc4_indices.size),
        },
        "conditions": {
            "hold_ticks": HOLD_TICKS,
            "fresh_reset_per_condition": True,
            "ray_max_distance_m": _float_pair(RAY_MAX_DISTANCE_M),
            "blocked_ray_m": _float_pair(BLOCKED_RAY_M),
            "light": 15,
            "collisions": False,
            "series": {
                condition: {name: _hex_series(values) for name, values in recorded.items()}
                for condition, recorded in conditions.items()
            },
        },
        "direct_drive_diagnostic": direct,
        "quantities": quantities,
        "d060_comparators": {
            "note": (
                "the D-060 scratchpad values this receipt re-derives, restated "
                "for comparison only; the scratchpad run was not "
                "content-addressed and this receipt supersedes it"
            ),
            "dnp01_response_ratio": "~300x above the wing-pool means",
            "ttmn_attenuation": "1.40% (census anatomical fraction 1.49%)",
            "ttmn_event_contrast": "45-76x peak-over-baseline (vs ~7x through the wings)",
            "arc_latency": "full arc within one game tick",
        },
        "coverage": {
            "final_trace_sha256": controller.engine.trace_sha256,
            "verified_through_step": controller.engine.verified_through_step,
            "uncovered_steps_at_exit": controller.engine.uncovered_steps,
            "run_is_covered": controller.engine.run_is_covered,
        },
    }


def write_or_verify(output: Path, evidence: dict[str, Any]) -> tuple[str, str]:
    """Write the evidence once; afterwards only byte-identical reproduction passes."""

    payload = canonical_json_bytes(evidence)
    digest = sha256(payload).hexdigest()
    if output.exists():
        if output.read_bytes() != payload:
            raise EscapeArcMeasurementError(
                f"existing evidence at {output} differs from this measurement; "
                "evidence is immutable -- investigate instead of overwriting"
            )
        return digest, "verified"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return digest, "wrote"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-directory", type=Path, default=DEFAULT_ARTIFACT_DIRECTORY)
    parser.add_argument(
        "--artifact-manifest-sha256",
        default=FULL_MCNS_ARTIFACT_MANIFEST_SHA256,
        help=(
            "external pin for the artifact manifest; defaults to the reviewed module "
            "constant recorded by RUNTIME-MCNS-01"
        ),
    )
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--body-annotations", type=Path, default=DEFAULT_BODY_ANNOTATIONS)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)

    started = time.perf_counter()
    evidence = measure(
        artifact_directory=arguments.artifact_directory,
        artifact_manifest_sha256=arguments.artifact_manifest_sha256,
        annotations_path=arguments.annotations,
        body_annotations_path=arguments.body_annotations,
        calibration_path=arguments.calibration,
    )
    measured = time.perf_counter()
    digest, action = write_or_verify(arguments.output, evidence)
    quantities = evidence["quantities"]
    ratios = quantities["dnp01_response_ratio"]["per_condition"]
    contrasts = quantities["ttmn_event_contrast"]["per_condition"]
    print(
        f"{action} {RECEIPT_ID}: "
        f"dnp01_ratio blocked_front={ratios['blocked_front']['ratio']['value']:.3f} "
        f"all_rays={ratios['all_rays_blocked']['ratio']['value']:.3f} | "
        f"ttmn_attenuation={quantities['ttmn_attenuation']['attenuation']['value']:.6f} | "
        f"ttmn_contrast blocked_front={contrasts['blocked_front']['contrast']['value']:.3f} "
        f"all_rays={contrasts['all_rays_blocked']['contrast']['value']:.3f} | "
        f"arc_latency={quantities['arc_latency']['neural_steps']} neural steps "
        f"(within_one_game_tick={quantities['arc_latency']['within_one_game_tick']}) | "
        f"covered {evidence['coverage']['run_is_covered']}"
    )
    print(f"{action} {arguments.output}")
    print(f"evidence sha256: {digest}")
    print(
        "wall time (environment-scoped, not part of the canonical evidence): "
        f"{measured - started:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
