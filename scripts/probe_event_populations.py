#!/usr/bin/env python3
"""Offline synthetic event-population screening; no encoder, policy or Minecraft.

Food contact is a declared pulse into labellar-bristle gustatory annotations;
optional touch is a pulse into grooming-labeled mechanosensory afferents (not DNs).
Neither annotation establishes sweet-food valence or biological grooming. Every
member receives the same [0,1] amplitude, as in the existing engineered sensory
mapping. All other sensory inputs are zero, including light and physical features.
No gain or FEED threshold is fitted. Reported rectification is max(pool mean, 0),
uncalibrated, and does not invoke a runtime decoder or select a body program.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

# Match the census CLI: direct script execution must resolve sibling scripts
# through this checkout, not through an unrelated working directory.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurocraft_connectome.full_runtime import load_full_runtime_artifact
from neurocraft_controller.config import canonical_sha256
from neurocraft_controller.flight_pools import (
    LEFT_WING_MOTOR_BODY_IDS,
    RIGHT_WING_MOTOR_BODY_IDS,
)
from neurocraft_controller.full_brain_controller import (
    FULL_MCNS_ARTIFACT_MANIFEST_SHA256,
    SIGN_OVERLAY_POLICY_PREFIX,
)
from neurocraft_neural.full_controls import FullControlKind, FullControlView, build_control_view
from neurocraft_neural.full_engine import ArrayNativeEngine
from neurocraft_neural.full_signs import _verified_graph_array, compute_full_sign_overlay
from neurocraft_neural.full_view import FullRuntimeView, view_from_full_runtime_graph
from neurocraft_neural.spmv_kernel import SpmvKernel, load_kernel
from numpy.typing import NDArray

from scripts.readout_annotations import (
    RetainedAnnotations,
    annotation_gates,
    equals,
    file_sha256,
    require_all_gates,
    take_retained_annotations,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "data/processed/male-cns-v1.0-superclass-v1"
ANNOTATIONS = ROOT / "data/cache/male-cns-v1.0"
NUMERICAL_FLOOR = 1e-12  # Declared before measurement; numerical detection, not behavior.
PINNED_COUNTS = {
    "food_contact_source": (82, 81),
    "groom_touch_source": (40, 25),
    "mn9": (1, 1),
    "proboscis_pm": (35, 32),
    "front_leg_fl": (68, 67),
    "antennal_am": (6, 7),
    "ttmn": (1, 1),
    "dna01": (1, 1),
    "wing": (25, 27),
}


@dataclass(frozen=True)
class Pulse:
    amplitude: float = 1.0
    baseline_steps: int = 20
    duration_steps: int = 5
    recovery_steps: int = 100

    def __post_init__(self) -> None:
        if not math.isfinite(self.amplitude) or not 0 < self.amplitude <= 1:
            raise ValueError("pulse amplitude must be finite in (0, 1]")
        for value in (self.baseline_steps, self.duration_steps, self.recovery_steps):
            if type(value) is not int or value < 1:
                raise ValueError("baseline, pulse and recovery steps must be positive integers")
        if self.total_steps > 2000:
            raise ValueError("probe is bounded to 2000 neural steps per trial")

    @property
    def total_steps(self) -> int:
        return self.baseline_steps + self.duration_steps + self.recovery_steps

    @property
    def onset(self) -> int:
        return self.baseline_steps + 1

    def active(self, step: int) -> bool:
        return self.baseline_steps < step <= self.baseline_steps + self.duration_steps


@dataclass(frozen=True)
class Population:
    name: str
    selector: str
    side_column: str
    # (dense index, source body ID, source side); ascending dense index.
    members: tuple[tuple[int, int, str], ...]

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(member[0] for member in self.members)

    def document(self) -> dict[str, Any]:
        membership = {
            "selector": self.selector,
            "side_column": self.side_column,
            "members_dense_body_side": self.members,
        }
        return {
            **membership,
            "count": len(self.members),
            "side_counts": {side: sum(row[2] == side for row in self.members) for side in "LR"},
            "membership_sha256": canonical_sha256(membership),
        }


def select_populations(
    annotations: RetainedAnnotations,
    *,
    expected_counts: Mapping[str, tuple[int, int]] = PINNED_COUNTS,
    wing_ids: tuple[tuple[int, ...], tuple[int, ...]] = (
        LEFT_WING_MOTOR_BODY_IDS,
        RIGHT_WING_MOTOR_BODY_IDS,
    ),
) -> dict[str, Population]:
    """Exact annotation selectors; fixture callers must declare their own counts/wing IDs."""
    specs: dict[str, tuple[dict[str, str], str]] = {
        "food_contact_source": ({"class": "gustatory", "subclass": "labellar bristle"}, "rootSide"),
        "groom_touch_source": ({"class": "mechanosensory", "subclass": "grooming"}, "rootSide"),
        "mn9": ({"type": "MN9"}, "somaSide"),
        "proboscis_pm": ({"superclass": "cb_motor", "subclass": "pm"}, "somaSide"),
        "front_leg_fl": ({"superclass": "vnc_motor", "subclass": "fl"}, "somaSide"),
        "antennal_am": ({"superclass": "cb_motor", "subclass": "am"}, "somaSide"),
        "ttmn": ({"type": "TTMn"}, "somaSide"),
        "dna01": ({"type": "DNa01"}, "somaSide"),
        "wing": ({}, "somaSide"),
    }
    if set(expected_counts) != set(specs):
        raise ValueError("expected counts must cover exactly the declared populations")
    populations: dict[str, Population] = {}
    for name, (terms, side_column) in specs.items():
        mask = np.ones(annotations.node_count, dtype=bool)
        for column, value in terms.items():
            mask &= equals(annotations.column(column), value)
        selector = " and ".join(f"{key} == {value!r}" for key, value in terms.items())
        if name == "wing":
            wanted = wing_ids[0] + wing_ids[1]
            if len(set(wanted)) != len(wanted):
                raise ValueError("wing body IDs must be unique")
            mask &= np.isin(annotations.body_ids, wanted)
            selector = "bodyId in existing LEFT/RIGHT_WING_MOTOR_BODY_IDS"
        members = tuple(
            (int(i), int(annotations.body_ids[i]), str(annotations.column(side_column)[i]))
            for i in np.flatnonzero(mask)
        )
        counts = tuple(sum(row[2] == side for row in members) for side in "LR")
        if (
            not members
            or counts != expected_counts[name]
            or len(members) != sum(counts)
            or min(counts) < 1
        ):
            raise ValueError(f"population {name} has missing members or unexpected count/side")
        if name == "wing" and any(
            body not in wing_ids[0 if side == "L" else 1] for _, body, side in members
        ):
            raise ValueError("wing sides disagree with the existing binding")
        populations[name] = Population(name, selector, side_column, members)
    return populations


def _series_samples(
    activity: NDArray[np.float64], populations: Mapping[str, Population]
) -> dict[str, float]:
    result = {}
    for name, population in populations.items():
        for suffix, indices in (
            ("", population.indices),
            ("_left", tuple(i for i, _, side in population.members if side == "L")),
            ("_right", tuple(i for i, _, side in population.members if side == "R")),
        ):
            result[name + suffix] = math.fsum(float(activity[i]) for i in indices) / len(indices)
    return result


def response_summary(values: Sequence[float], pulse: Pulse) -> dict[str, Any]:
    """Detection uses absolute *signed mean difference*, with no rectification first."""
    first = next((i for i, value in enumerate(values) if abs(value) > NUMERICAL_FLOOR), None)
    peak = max(range(len(values)), key=lambda i: abs(values[i]))
    recovery_start = pulse.baseline_steps + pulse.duration_steps
    # A bounded observation of the remaining tail, not a claim about future behavior.
    tail = values[recovery_start:]
    last_above = max(
        (i for i, value in enumerate(tail) if abs(value) > NUMERICAL_FLOOR), default=-1
    )
    recovered = first is not None and last_above < len(tail) - 1
    return {
        "first_above_floor_step": None if first is None else first + 1,
        "first_above_floor_lag_steps": None if first is None else first + 1 - pulse.onset,
        "peak_absolute_signed_response": abs(values[peak]),
        "signed_response_at_peak": values[peak],
        "peak_step": peak + 1,
        "final_signed_response": values[-1],
        "recovery_tail_final_absolute_response": abs(values[-1]),
        "recovery_tail_maximum_absolute_response": max(abs(value) for value in tail),
        "final_absolute_fraction_of_peak": (
            None if values[peak] == 0 else abs(values[-1]) / abs(values[peak])
        ),
        "recovery_tail_below_floor_from_step": (
            recovery_start + last_above + 2 if recovered else None
        ),
        "recovery_observation_end_step": len(values),
    }


def measure_lane(
    view: FullRuntimeView,
    populations: Mapping[str, Population],
    pulse: Pulse,
    *,
    kernel: SpmvKernel,
    control: FullControlView | None = None,
    include_grooming: bool = False,
) -> dict[str, Any]:
    if control is not None and control.view is not view:
        raise ValueError("control metadata does not describe the executed view")
    engine = ArrayNativeEngine(view, kernel=kernel)
    sources = ["food_contact_source"] + (["groom_touch_source"] if include_grooming else [])
    trials: dict[str, Any] = {}
    for condition in ["no_event", *sources]:
        engine.reset()
        vector = np.zeros(view.node_count, dtype=np.float64)
        if condition != "no_event":
            vector[list(populations[condition].indices)] = pulse.amplitude
        requested_sha = sha256(vector.astype("<f8").tobytes()).hexdigest()
        if control is not None and control.zeroes_sensory_input:
            vector.fill(0.0)  # After every declared input has been constructed.
        executed_sha = sha256(vector.astype("<f8").tobytes()).hexdigest()
        provenance = canonical_sha256(
            {
                "schema": "neurocraft.event_population_input.v1",
                "condition": condition,
                "pulse": asdict(pulse),
                "requested_pulse_vector_sha256": requested_sha,
                "executed_pulse_vector_sha256": executed_sha,
            }
        )
        series: dict[str, list[float]] = {}
        state_series = []
        for step in range(1, pulse.total_steps + 1):
            engine.step(
                sensory=vector if pulse.active(step) else None,
                sensory_provenance_sha256=provenance,
            )
            for name, mean in _series_samples(engine.activity(), populations).items():
                series.setdefault(name, []).append(mean)
            state_series.append(engine.current_state().state_sha256)
        engine.end_run()
        if not engine.run_is_covered:
            raise RuntimeError("engine run did not finish with verified view coverage")
        trials[condition] = {
            "requested_pulse_vector_sha256": requested_sha,
            "executed_pulse_vector_sha256": executed_sha,
            "executed_pulse_nonzero_nodes": int(np.count_nonzero(vector)),
            "input_provenance_sha256": provenance,
            "raw_signed_means": series,
            "uncalibrated_rectified_means": {
                name: [max(0.0, value) for value in values] for name, values in series.items()
            },
            "state_sha256_by_step": state_series,
            "final_state_sha256": engine.current_state().state_sha256,
        }
    for condition in sources:
        source = populations[condition]
        delta = {
            name: [
                event - baseline
                for event, baseline in zip(
                    values, trials["no_event"]["raw_signed_means"][name], strict=True
                )
            ]
            for name, values in trials[condition]["raw_signed_means"].items()
        }
        trials[condition]["event_minus_no_event_signed_means"] = delta
        trials[condition]["responses"] = {
            name: response_summary(values, pulse) for name, values in delta.items()
        }
        # Structural membership is unchanged by a no-input control. Keep it
        # distinct from the indices that actually received a nonzero pulse.
        source_indices = set(source.indices)
        executed_indices = (
            set() if control is not None and control.zeroes_sensory_input else source_indices
        )
        trials[condition]["source_membership_overlap"] = {
            name: {
                "count": len(set(source.indices) & set(population.indices)),
                "body_ids": sorted(
                    body for index, body, _ in population.members if index in source.indices
                ),
            }
            for name, population in populations.items()
        }
        trials[condition]["executed_direct_input_overlap_counts"] = {
            name: len(executed_indices & set(population.indices))
            for name, population in populations.items()
        }
    return {
        "lane": "original" if control is None else control.kind.value,
        "control_identity": None if control is None else control.identity.material,
        "control_identity_sha256": None if control is None else control.identity.identity_sha256,
        "view_sha256": view.view_sha256,
        "view_provenance_sha256": view.provenance.identity_sha256,
        "view_provenance": asdict(view.provenance),
        "backend_dynamics_sha256": engine.dynamics_identity_sha256,
        "dynamics_configuration": engine.config.identity_material(engine.kernel_identity),
        "dynamics_configuration_values": asdict(engine.config),
        "kernel_identity": engine.kernel_identity.material,
        "neural_step_ms": engine.config.neural_step_ms,
        "trials": trials,
    }


def run_probe(args: argparse.Namespace, pulse: Pulse) -> dict[str, Any]:
    with load_full_runtime_artifact(
        args.artifact, expected_manifest_sha256=FULL_MCNS_ARTIFACT_MANIFEST_SHA256
    ) as graph:
        rows = _verified_graph_array(graph, "annotation_source_rows", "<u8").copy()
        bodies = _verified_graph_array(graph, "body_ids", "<u8").copy()
        annotations, _ = take_retained_annotations(
            args.body_annotations, source_rows=rows, body_ids=bodies
        )
        annotation_sha = next(
            source.sha256
            for source in graph.manifest.source_artifacts
            if source.source_id == "annotations"
        )
        gates = annotation_gates(
            annotations,
            declared_annotations_sha256=annotation_sha,
            declared_source_rows=graph.manifest.counts["annotation_source_rows"],
            retained_by_superclass=dict(graph.manifest.retained_by_superclass),
        )
        require_all_gates(gates)
        populations = select_populations(annotations)
        overlay = compute_full_sign_overlay(graph, args.annotations)
        faithful = view_from_full_runtime_graph(
            graph,
            signs=overlay.edge_signs,
            sign_provenance=f"{SIGN_OVERLAY_POLICY_PREFIX}:{overlay.identity_sha256}",
        )
        counts = (
            _verified_graph_array(graph, "edge_raw_counts", "<u8").astype(np.int64)
            if args.controls
            else None
        )
    kernel = load_kernel(thread_count=args.threads)
    lanes = [
        measure_lane(
            faithful, populations, pulse, kernel=kernel, include_grooming=args.include_grooming
        )
    ]
    if args.controls:
        for kind in (FullControlKind.SHUFFLED_WEIGHT, FullControlKind.NO_INPUT):
            control = build_control_view(
                faithful,
                kind=kind,
                seed=args.seed,
                artifact_manifest_sha256=FULL_MCNS_ARTIFACT_MANIFEST_SHA256,
                raw_counts=counts,
                edge_signs=overlay.edge_signs.astype(np.int64),
            )
            lanes.append(
                measure_lane(
                    control.view,
                    populations,
                    pulse,
                    kernel=kernel,
                    control=control,
                    include_grooming=args.include_grooming,
                )
            )
    return {
        "schema": "neurocraft.event_population_probe.v1",
        "diagnostic_only": True,
        "probe_source_sha256": file_sha256(Path(__file__)),
        "scope": (
            "synthetic isolated per-capita sensory pulses, full retained network; "
            "no live encoder, calibrated decoder, program selection, Minecraft or biological claim"
        ),
        "source": {
            "artifact_path": str(args.artifact.resolve()),
            "graph_manifest_sha256": FULL_MCNS_ARTIFACT_MANIFEST_SHA256,
            "body_annotations_path": str(args.body_annotations.resolve()),
            "body_annotations_sha256": annotations.annotations_sha256,
            "neurotransmitter_annotations_sha256": file_sha256(args.annotations),
            "annotation_source_rows_sha256": sha256(rows.astype("<u8").tobytes()).hexdigest(),
            "body_ids_sha256": sha256(bodies.astype("<u8").tobytes()).hexdigest(),
            "modeled_sign_overlay_sha256": overlay.identity_sha256,
            "node_count": faithful.node_count,
            "edge_count": faithful.edge_count,
            "annotation_integrity_gates": [gate.as_json() for gate in gates],
        },
        "stimulus": {
            **asdict(pulse),
            "mapping": "same amplitude per member; no population-size division",
            "pulse_onset_neural_step": pulse.onset,
            "neural_step_ms": lanes[0]["neural_step_ms"],
            "other_inputs": "exactly zero; no physical features or tonic light",
            "side_convention": "sensory rootSide; motor and DNa01 somaSide",
            "interpretation": "food/touch association engineered; no sweet valence or grooming DN",
        },
        "measurement": {
            "units": "dimensionless modeled activity",
            "rectification": "max(raw population mean, 0); uncalibrated",
            "response": "event raw signed mean minus same-step no-event raw signed mean",
            "numerical_nonzero_absolute_floor": NUMERICAL_FLOOR,
            "lag_zero": "first stimulated update; direct input can respond at lag zero",
            "recovery": (
                "bounded tail amplitude and final fraction of peak; below-floor timestamp is "
                "limited to the remaining observed tail. Null does not establish persistence"
            ),
            "no_behavior_threshold_selected": True,
        },
        "threads_requested": args.threads,
        "seed": str(args.seed) if args.controls else None,
        "neural_steps": list(range(1, pulse.total_steps + 1)),
        "populations": {name: pool.document() for name, pool in populations.items()},
        "lanes": lanes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ANNOTATIONS / "body-neurotransmitters-male-cns-v1.0.feather",
    )
    parser.add_argument(
        "--body-annotations",
        type=Path,
        default=ANNOTATIONS / "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--controls", action="store_true", help="also run shuffled_weight/no_input")
    parser.add_argument(
        "--include-grooming", action="store_true", help="also pulse sensory afferents"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pulse-amplitude", type=float, default=1.0)
    parser.add_argument("--pulse-steps", type=int, default=5)
    parser.add_argument("--baseline-steps", type=int, default=20)
    parser.add_argument("--recovery-steps", type=int, default=100)
    parser.add_argument(
        "--json", "--output", type=Path, help="exclusive JSON output; default JSON stdout"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        pulse = Pulse(
            args.pulse_amplitude, args.baseline_steps, args.pulse_steps, args.recovery_steps
        )
        if not 1 <= args.threads <= 64 or not 0 <= args.seed < 2**64:
            raise ValueError("threads must be 1..64 and seed must be uint64")
        if not args.controls and args.seed != 7:
            raise ValueError("non-default seed requires --controls")
        if args.json is not None and args.json.exists():
            raise ValueError("refusing to overwrite existing JSON output")
    except ValueError as exc:
        parser.error(str(exc))
    document = run_probe(args, pulse)
    encoded = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.json is None:
        print(encoded, end="")
    else:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
