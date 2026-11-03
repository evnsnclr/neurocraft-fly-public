#!/usr/bin/env python3
"""Matched synthetic loom-v2 stimuli: original, shuffled weights, no input, restoration.

Uses the same committed original calibration and policy in every lane. Repeated
fresh-state trials are deterministic repeats, not biological or statistical
replicates. A control may preserve or strengthen a response; neither outcome
fails this diagnostic. It fails only broken bindings, input gates, or restoration.
"""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from neurocraft_controller.full_brain_controller import (  # noqa: E402
    FullBrainConfiguration,
    FullBrainController,
)
from neurocraft_controller.program_policy import ProgramSelectingController  # noqa: E402
from neurocraft_protocol import Observation, SensorFrameV2, canonical_json_bytes  # noqa: E402

from scripts.probe_program_policy import (  # noqa: E402
    DEFAULT_ANNOTATIONS,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_BODY_ANNOTATIONS,
    approach_ramp_frames,
    held_frames,
    summarize,
)


def build(mode: str, args: argparse.Namespace) -> ProgramSelectingController:
    return ProgramSelectingController(
        FullBrainController.from_artifact(
            run_id="program-control-probe",
            entity_id="probe-fly",
            configuration=FullBrainConfiguration(mode=mode, kernel_threads=args.threads),
            artifact_directory=args.artifact_dir,
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
            control_permutation_directory=args.permutation_directory,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--body-annotations", type=Path, default=DEFAULT_BODY_ANNOTATIONS)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=ROOT / "data/calibration/full-readout-calibration-loom-v2.json",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--permutation-directory", type=Path, default=ROOT / "artifacts/controls")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.trials <= 100:
        parser.error("--trials must be in [1,100]")
    conditions = {
        "front_entity_approach": (approach_ramp_frames(), 2),
        "front_wall_approach": (approach_ramp_frames(), 1),
        "no_event": (held_frames(None, 15, False), 0),
    }
    lanes: dict[str, Any] = {}
    for label, mode in (
        ("original", "faithful"),
        ("shuffled_weight", "shuffled_weight"),
        ("no_input", "no_input"),
        ("original_restored", "faithful"),
    ):
        controller = build(mode, args)
        inner = controller.controller
        epoch = 0
        lane: dict[str, Any] = {
            "run_identity": controller.identity.to_dict(),
            "mode": inner.control_kind,
            "control": None if inner.control_identity is None else inner.control_identity.material,
            "control_identity_sha256": None
            if inner.control_identity is None
            else inner.control_identity.identity_sha256,
            "policy_hash": controller.policy_hash,
            "calibration_sha256": inner.configuration.readout_calibration_sha256,
            "readout_scale_hex": inner.readout_scale.hex(),
            "jump_scale_hex": inner.escape_binding.jump_scale.hex()
            if inner.escape_binding is not None
            else None,
            "conditions": {},
        }
        try:
            for name, (frames, kind) in conditions.items():
                trials = []
                for trial in range(args.trials):
                    epoch += 1
                    controller.reset_session(session_epoch=epoch)
                    issued = []
                    for tick, frame in enumerate(frames):
                        issued.append(
                            controller.step(
                                Observation(
                                    identity=controller.identity,
                                    observation_seq=tick,
                                    game_tick=tick,
                                    sent_monotonic_ns=tick * 50_000_000,
                                    sensors=SensorFrameV2.from_sensor_frame(frame, (kind,) * 6),
                                ),
                                command_seq=tick,
                            )
                        )
                        if mode == "no_input":
                            evidence = inner.last_evidence
                            assert evidence is not None
                            if any(evidence.encoded_stimulation.values()):
                                raise RuntimeError("no_input reported a delivered nonzero channel")
                    trials.append(
                        {
                            "trial": trial,
                            "session_epoch": epoch,
                            **summarize(name, issued),
                            "program_action_stream_sha256": sha256(
                                canonical_json_bytes([action.to_dict() for action in issued])
                            ).hexdigest(),
                        }
                    )
                lane["conditions"][name] = {"ray_hit_kinds": [kind] * 6, "trials": trials}
                first = trials[0]
                print(
                    f"{label} {name}: peak_jump={first['peak_ttmn_jump']:.17g} "
                    f"selections={first['selection_counts']}",
                    flush=True,
                )
        finally:
            inner.engine.end_run()
        lanes[label] = lane
        del controller, inner
    binding_keys = ("policy_hash", "calibration_sha256", "readout_scale_hex", "jump_scale_hex")
    checks = {
        "fixed_calibration_readouts_policy": all(
            len({lane[key] for lane in lanes.values()}) == 1 for key in binding_keys
        ),
        # Conditions include complete canonical ProgramAction stream digests,
        # not just summaries that omit state, envelope, and modulation fields.
        "original_restored_exactly": lanes["original"]["conditions"]
        == lanes["original_restored"]["conditions"],
        "distinct_control_dynamics": len(
            {lane["run_identity"]["dynamics_hash"] for lane in lanes.values()}
        )
        == 3,
    }
    document = {
        "schema": "neurocraft.program_control_probe.v1",
        "diagnostic_only": True,
        "stimulus_scope": "synthetic typed sensor frames; deterministic repeats, not live replay",
        "seed": args.seed,
        "lanes": lanes,
        "checks": checks,
        "response_difference_is_measured_not_required": True,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps(checks, sort_keys=True), flush=True)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
