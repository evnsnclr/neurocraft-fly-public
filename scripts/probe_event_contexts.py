#!/usr/bin/env python3
"""Experimental pulses added to the existing loom-v2 encoder, offline only.

This reuses production encoder components and neural dynamics, not a production
controller profile. The composite sensory input has its own descriptor and hash.
No policy, calibrated motor decoding, Minecraft behavior or threshold fitting is
performed. Backgrounds are declared synthetic sensor histories, not world trials.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from neurocraft_connectome.full_runtime import load_full_runtime_artifact
from neurocraft_controller.config import canonical_sha256
from neurocraft_controller.full_brain_controller import (
    FULL_MCNS_ARTIFACT_MANIFEST_SHA256,
    FullBrainConfiguration,
    FullBrainController,
    FullBrainLoomV2Configuration,
)
from neurocraft_controller.loom_senses_v2 import EemEncoderDocumentLoomV2
from neurocraft_neural.full_signs import _verified_graph_array
from neurocraft_protocol import SensorFrameV2
from numpy.typing import NDArray

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.probe_event_populations import (
    ANNOTATIONS,
    ARTIFACT,
    NUMERICAL_FLOOR,
    ROOT,
    Population,
    Pulse,
    _series_samples,
    response_summary,
    select_populations,
)
from scripts.readout_annotations import (
    annotation_gates,
    file_sha256,
    require_all_gates,
    take_retained_annotations,
)

Json = dict[str, Any]
BACKGROUNDS = (
    "open_air_bright",
    "open_air_dark",
    "grounded_contact",
    "wall_contact",
    "entity_approach",
)
PULSE_SOURCES = ("food_contact_source", "groom_touch_source")
CALIBRATION = ROOT / "data/calibration/full-readout-calibration-loom-v2.json"


@dataclass(frozen=True)
class Schedule:
    baseline_ticks: int = 4
    pulse_ticks: int = 1
    recovery_ticks: int = 20
    amplitude: float = 1.0

    def __post_init__(self) -> None:
        for value in (self.baseline_ticks, self.pulse_ticks, self.recovery_ticks):
            if type(value) is not int or value < 1:
                raise ValueError("all tick durations must be positive integers")
        if self.total_ticks * 5 > 2000:
            raise ValueError("each trial is bounded to 2000 neural updates")
        if not math.isfinite(self.amplitude) or not 0 < self.amplitude <= 1:
            raise ValueError("pulse amplitude must be finite in (0, 1]")

    @property
    def total_ticks(self) -> int:
        return self.baseline_ticks + self.pulse_ticks + self.recovery_ticks

    def active(self, tick: int) -> bool:
        return self.baseline_ticks <= tick < self.baseline_ticks + self.pulse_ticks

    def neural_pulse(self, steps_per_tick: int) -> Pulse:
        return Pulse(
            self.amplitude,
            self.baseline_ticks * steps_per_tick,
            self.pulse_ticks * steps_per_tick,
            self.recovery_ticks * steps_per_tick,
        )


def background_frame(name: str, tick: int, schedule: Schedule) -> SensorFrameV2:
    """Zero-based game ticks; contact/approach starts with the added pulse.

    Bright/dark are tonic from tick zero. Contact remains present after onset;
    approach advances the front entity ray by 0.8 blocks/tick to 1.2 blocks and
    holds there. Other rays/features remain explicitly stated below. These
    prescribed histories do not simulate motion or collision geometry.
    """
    if name not in BACKGROUNDS or not 0 <= tick < schedule.total_ticks:
        raise ValueError("unknown background or out-of-range tick")
    rays = [6.0] * 6
    kinds = [0] * 6
    ground = horizontal = vertical = False
    if tick >= schedule.baseline_ticks:
        if name == "grounded_contact":
            ground = vertical = True
            rays[5], kinds[5] = 0.11, 1
        elif name == "wall_contact":
            horizontal = True
            rays[0], kinds[0] = 0.3, 1
        elif name == "entity_approach":
            rays[0] = max(1.2, 6.0 - 0.8 * (tick - schedule.baseline_ticks + 1))
            kinds[0] = 2
    return SensorFrameV2(
        position_m=(0.0, 3.0, 0.0),
        velocity_m_s=(0.0, 0.0, 0.0),
        yaw_rad=0.0,
        pitch_rad=0.0,
        on_ground=ground,
        horizontal_collision=horizontal,
        vertical_collision=vertical,
        in_fluid=False,
        local_light=0 if name == "open_air_dark" else 15,
        ray_max_distance_m=6.0,
        ray_distances_m=(rays[0], rays[1], rays[2], rays[3], rays[4], rays[5]),
        ray_hit_kinds=(kinds[0], kinds[1], kinds[2], kinds[3], kinds[4], kinds[5]),
    )


def vector_sha(vector: NDArray[np.float64]) -> str:
    return sha256(vector.astype("<f8", copy=False).tobytes()).hexdigest()


class LoomV2BackgroundEncoder:
    """Reuse the live input methods directly; never replace or wrap engine.step.

    These private methods are intentional source-level dependencies, pinned by
    source digests in each report and by a fixture comparison with controller.step.
    Only this offline owner accesses its controller, synchronously.
    """

    def __init__(self, core: FullBrainController) -> None:
        embodied = core.embodied_senses
        if (
            not isinstance(core.configuration, FullBrainLoomV2Configuration)
            or embodied is None
            or not isinstance(embodied.encoder, EemEncoderDocumentLoomV2)
            or core.configuration.neural_steps_per_game_tick != 5
        ):
            raise ValueError("requires the existing five-update loom-v2 components")
        self.core = core
        self.embodied = embodied

    def reset_trial(self) -> None:
        self.core.reset_session(session_epoch=self.core.identity.session_epoch + 1)

    def encode(self, sensors: SensorFrameV2) -> tuple[NDArray[np.float64], Json]:
        features = self.core.features.encode(sensors)
        vector, assignments = self.core._encode_dense(features.values)
        raw, means, channels, drives = self.core._apply_embodied_drives_v2(vector, sensors)
        # Detach and freeze before any experimental addition or engine call.
        snapshot = np.array(vector, dtype=np.float64, copy=True)
        snapshot.flags.writeable = False
        return snapshot, {
            "feature_values": dict(features.values),
            "faithful_assignments": assignments,
            "eem_raw": dict(raw),
            "eem_means_used": dict(means),
            "eem_channels": dict(channels),
            "eem_population_drives": dict(drives),
            "background_vector_sha256": vector_sha(snapshot),
        }

    def component_descriptor(self) -> Json:
        core = self.core
        return {
            "scope": "reused components only; identity does not bind the experimental input",
            "identity": core.identity.to_dict(),
            "encoder_document": self.embodied.encoder.to_identity_material(),
            "encoder_sha256": self.embodied.encoder.identity_sha256,
            "embodied_population_binding_sha256": self.embodied.populations.identity_sha256,
            "features_sha256": core.features.identity_sha256,
            "sign_overlay_sha256": core.sign_overlay_identity_sha256,
            "view_sha256": core.engine.view.view_sha256,
            "node_count": core.engine.view.node_count,
            "edge_count": core.engine.view.edge_count,
            "view_provenance": asdict(core.engine.view.provenance),
            "engine_configuration": asdict(core.engine.config),
            "neural_steps_per_game_tick": core.configuration.neural_steps_per_game_tick,
            "backend_dynamics_sha256": core.engine.dynamics_identity_sha256,
            "kernel_identity": core.engine.kernel_identity.material,
            "control_kind": core.control_kind,
            "control_identity": (
                None if core.control_identity is None else core.control_identity.material
            ),
            "control_identity_sha256": (
                None if core.control_identity is None else core.control_identity.identity_sha256
            ),
        }


def compose_input(
    background: NDArray[np.float64],
    source: Population | None,
    amplitude: float,
    *,
    zero_input: bool,
) -> NDArray[np.float64]:
    """Add per member without clipping, then apply the final no-input gate."""
    vector = np.array(background, dtype=np.float64, copy=True)
    if source is not None:
        vector[list(source.indices)] += amplitude
    if zero_input:
        vector.fill(0.0)
    if not np.isfinite(vector).all():
        raise ValueError("composite input is non-finite")
    vector.flags.writeable = False
    return vector


def _overlap(vector: NDArray[np.float64], populations: dict[str, Population]) -> dict[str, int]:
    return {
        name: int(np.count_nonzero(vector[list(pop.indices)])) for name, pop in populations.items()
    }


def measure_context(
    core: FullBrainController, populations: dict[str, Population], name: str, schedule: Schedule
) -> Json:
    encoder = LoomV2BackgroundEncoder(core)
    samples = [background_frame(name, tick, schedule) for tick in range(schedule.total_ticks)]
    background_descriptor = {"name": name, "sensors_by_game_tick": [s.to_dict() for s in samples]}
    background_hash = canonical_sha256(background_descriptor)
    trials: Json = {}
    for condition in ("no_added_pulse", *PULSE_SOURCES):
        encoder.reset_trial()
        source = None if condition == "no_added_pulse" else populations[condition]
        descriptor = {
            "schema": "neurocraft.experimental_context_input.v1",
            "diagnostic_only": True,
            "production_profile_claim": False,
            "component": encoder.component_descriptor(),
            "background_sha256": background_hash,
            "schedule": asdict(schedule),
            "pulse_source": None if source is None else source.document(),
            "composition": "background plus amplitude per member; no clipping; final input gate",
            "input_gate": "zero" if core.control_kind == "no_input" else "identity",
        }
        descriptor_hash = canonical_sha256(descriptor)
        series: dict[str, list[float]] = {}
        states, input_rows = [], []
        for tick, sensors in enumerate(samples):
            base, encoded = encoder.encode(sensors)
            active_source = source if schedule.active(tick) else None
            requested = compose_input(base, active_source, schedule.amplitude, zero_input=False)
            delivered = compose_input(
                base, active_source, schedule.amplitude, zero_input=core.control_kind == "no_input"
            )
            pulse_indices = set() if active_source is None else set(active_source.indices)
            row = {
                "game_tick": tick,
                "encoder": encoded,
                "requested_composite_sha256": vector_sha(requested),
                "delivered_composite_sha256": vector_sha(delivered),
                "delivered_nonzero_nodes": int(np.count_nonzero(delivered)),
                "maximum_absolute_requested_input": float(np.max(np.abs(requested))),
                "background_nonzero_overlap_counts": _overlap(base, populations),
                "delivered_composite_nonzero_overlap_counts": _overlap(delivered, populations),
                "requested_pulse_overlap_counts": {
                    key: len(pulse_indices & set(pop.indices)) for key, pop in populations.items()
                },
                "delivered_pulse_overlap_counts": {
                    key: 0
                    if core.control_kind == "no_input"
                    else len(pulse_indices & set(pop.indices))
                    for key, pop in populations.items()
                },
            }
            provenance = canonical_sha256(
                {
                    "descriptor_sha256": descriptor_hash,
                    "game_tick": tick,
                    "delivered_vector_sha256": row["delivered_composite_sha256"],
                }
            )
            for _ in range(core.configuration.neural_steps_per_game_tick):
                core.engine.step(sensory=delivered, sensory_provenance_sha256=provenance)
                for pool, value in _series_samples(core.engine.activity(), populations).items():
                    series.setdefault(pool, []).append(value)
                states.append(core.engine.current_state().state_sha256)
            # The immutable pre-step snapshots must still describe the executed data.
            if vector_sha(delivered) != row["delivered_composite_sha256"]:
                raise RuntimeError("engine mutated the frozen delivered input")
            input_rows.append(row)
        core.engine.end_run()
        if not core.engine.run_is_covered:
            raise RuntimeError("trial did not finish with verified view coverage")
        trials[condition] = {
            "experimental_descriptor": descriptor,
            "experimental_descriptor_sha256": descriptor_hash,
            "inputs_by_game_tick": input_rows,
            "raw_signed_means": series,
            "uncalibrated_rectified_means": {
                pool: [max(0.0, value) for value in values] for pool, values in series.items()
            },
            "state_sha256_by_neural_step": states,
        }
    pulse = schedule.neural_pulse(core.configuration.neural_steps_per_game_tick)
    reference = trials["no_added_pulse"]
    for condition in PULSE_SOURCES:
        trial = trials[condition]
        raw_deltas, rectified_deltas, metrics = {}, {}, {}
        for pool, values in trial["raw_signed_means"].items():
            base = reference["raw_signed_means"][pool]
            delta = [a - b for a, b in zip(values, base, strict=True)]
            rectified = trial["uncalibrated_rectified_means"][pool]
            base_rectified = reference["uncalibrated_rectified_means"][pool]
            rectified_delta = [a - b for a, b in zip(rectified, base_rectified, strict=True)]
            raw_deltas[pool], rectified_deltas[pool] = delta, rectified_delta
            background_peak = max(base_rectified[pulse.baseline_steps :])
            positive_increment = max(0.0, max(rectified_delta[pulse.baseline_steps :]))
            metrics[pool] = {
                "signed_difference_response": response_summary(delta, pulse),
                "background_rectified_peak_from_onset": background_peak,
                "with_pulse_rectified_peak_from_onset": max(rectified[pulse.baseline_steps :]),
                "peak_positive_paired_rectified_increment": positive_increment,
                "positive_increment_to_background_peak": (
                    None if background_peak == 0 else positive_increment / background_peak
                ),
            }
        trial["paired_raw_signed_difference"] = raw_deltas
        trial["paired_rectified_mean_difference"] = rectified_deltas
        trial["responses"] = metrics
    return {
        "background": background_descriptor,
        "background_sha256": background_hash,
        "trials": trials,
    }


def code_sources() -> Json:
    """Hash actual imported source locations, including editable worktree installations."""
    sources: Json = {
        "probe": {"path": str(Path(__file__).resolve()), "sha256": file_sha256(Path(__file__))}
    }
    for name in (
        "scripts.probe_event_populations",
        "scripts.readout_annotations",
        "neurocraft_controller.full_brain_controller",
        "neurocraft_controller.loom_senses_v2",
        "neurocraft_controller.loom_senses",
        "neurocraft_controller.embodied_senses",
        "neurocraft_controller.features",
        "neurocraft_neural.full_engine",
        "neurocraft_neural.full_records",
        "neurocraft_neural.full_view",
        "neurocraft_neural.full_signs",
        "neurocraft_neural.full_controls",
        "neurocraft_neural.spmv_kernel",
        "neurocraft_neural.view_verification",
        "neurocraft_connectome.full_runtime",
    ):
        filename = importlib.import_module(name).__file__
        if filename is None:
            raise RuntimeError(f"source unavailable for {name}")
        path = Path(filename).resolve()
        sources[name] = {"path": str(path), "sha256": file_sha256(path)}
    return sources


def run_probe(args: argparse.Namespace, schedule: Schedule) -> Json:
    sources = code_sources()
    with load_full_runtime_artifact(
        args.artifact, expected_manifest_sha256=FULL_MCNS_ARTIFACT_MANIFEST_SHA256
    ) as graph:
        rows = _verified_graph_array(graph, "annotation_source_rows", "<u8").copy()
        bodies = _verified_graph_array(graph, "body_ids", "<u8").copy()
        annotations, _ = take_retained_annotations(
            args.body_annotations, source_rows=rows, body_ids=bodies
        )
        annotation_sha = next(
            s.sha256 for s in graph.manifest.source_artifacts if s.source_id == "annotations"
        )
        gates = annotation_gates(
            annotations,
            declared_annotations_sha256=annotation_sha,
            declared_source_rows=graph.manifest.counts["annotation_source_rows"],
            retained_by_superclass=dict(graph.manifest.retained_by_superclass),
        )
        require_all_gates(gates)
        populations = select_populations(annotations)
    lanes = []
    for mode in ("faithful", "shuffled_weight", "no_input") if args.controls else ("faithful",):
        core = FullBrainController.from_artifact(
            run_id="experimental-event-contexts",
            entity_id="synthetic-fly",
            configuration=FullBrainConfiguration(mode=mode, kernel_threads=args.threads),
            artifact_directory=args.artifact,
            annotations_path=args.annotations,
            body_annotations_path=args.body_annotations,
            calibration_path=args.calibration,
            embodied_senses=True,
            escape_reflex=True,
            escape_version=2,
            loom_encoder=True,
            loom_version=2,
            steering_readout=True,
            control_seed=args.seed,
        )
        try:
            lanes.append(
                {
                    "lane": "original" if mode == "faithful" else mode,
                    "contexts": [
                        measure_context(core, populations, name, schedule)
                        for name in args.backgrounds
                    ],
                }
            )
        finally:
            # No active worker or socket exists; end_run checks any partially completed trial.
            core.engine.end_run()
        del core
    if code_sources() != sources:
        raise RuntimeError("source files changed during measurement; refusing report")
    return {
        "schema": "neurocraft.event_context_probe.v1",
        "diagnostic_only": True,
        "scope": (
            "experimental additive pulses over existing loom-v2 synthetic sensor histories; "
            "no production profile, live world, motor decoding, policy or fitted threshold"
        ),
        "source": {
            "artifact_manifest_sha256": FULL_MCNS_ARTIFACT_MANIFEST_SHA256,
            "artifact_path": str(args.artifact.resolve()),
            "body_annotations_sha256": annotations.annotations_sha256,
            "neurotransmitter_annotations_sha256": file_sha256(args.annotations),
            "calibration_sha256": file_sha256(args.calibration),
            "calibration_scope": "loaded unchanged for component binding; decoder never invoked",
            "annotation_source_rows_sha256": sha256(rows.astype("<u8").tobytes()).hexdigest(),
            "body_ids_sha256": sha256(bodies.astype("<u8").tobytes()).hexdigest(),
            "annotation_gates": [gate.as_json() for gate in gates],
            "files": sources,
        },
        "schedule": asdict(schedule),
        "seed": str(args.seed) if args.controls else None,
        "measurement": {
            "units": "dimensionless model activity; lag in neural updates, not wall time",
            "step_duration": "component.engine_configuration.neural_step_ms",
            "raw_difference": "with-added-pulse raw mean minus same-step same-background mean",
            "rectified_difference": (
                "max(event raw mean, 0) minus max(paired background raw mean, 0)"
            ),
            "numerical_floor": NUMERICAL_FLOOR,
            "recovery": "remaining observed tail only; null does not establish persistence",
            "overlaps": "population membership may overlap; MN9 is nested inside proboscis_pm",
            "no_threshold_selected": True,
        },
        "populations": {name: pool.document() for name, pool in populations.items()},
        "lanes": lanes,
    }


def main(argv: list[str] | None = None) -> int:
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
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument("--backgrounds", nargs="+", choices=BACKGROUNDS, default=list(BACKGROUNDS))
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--baseline-ticks", type=int, default=4)
    parser.add_argument("--pulse-ticks", type=int, default=1)
    parser.add_argument("--recovery-ticks", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        schedule = Schedule(
            args.baseline_ticks, args.pulse_ticks, args.recovery_ticks, args.amplitude
        )
        if not 1 <= args.threads <= 64 or not 0 <= args.seed < 2**64:
            raise ValueError("threads must be 1..64 and seed uint64")
        if not args.controls and args.seed != 7:
            raise ValueError("non-default seed requires controls")
        if len(args.backgrounds) != len(set(args.backgrounds)):
            raise ValueError("duplicate background")
        if args.output is not None and args.output.exists():
            raise ValueError("refusing to overwrite output")
    except ValueError as exc:
        parser.error(str(exc))
    result = run_probe(args, schedule)
    encoded = json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
