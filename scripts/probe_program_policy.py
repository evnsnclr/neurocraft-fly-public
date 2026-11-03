#!/usr/bin/env python3
"""Offline probe: what the declared selection policy asks for under held conditions.

Local-only, like the readout census: it needs the gitignored full artifact, the pinned
annotation tables, and the committed loom-v1 calibration, so it never runs on CI.

**What this measures.** For each of ten synthetic sensor conditions, it steps the
``full_mcns_loom_program_v1`` controller 30 ticks from a fresh reset and records what the
declared hand-chosen mapping asked for on each tick: the three labeled readout scalars, the
selected program, the modulation, and the turn bias.

**What this does not measure, establish, or imply.** That the connectome selected anything.
That the mapping is good. That a fly would do this. That any of these numbers generalize
past these ten synthetic conditions on one deterministic run with no dispersion estimate. The
policy is a declared function of three scalars; the probe reports the function's output on
inputs the brain produced under held stimuli, and nothing more.

The ten conditions are: the census's six held conditions, three approach ramps (front,
left-only, right-only), and one contact ONSET.

The six held conditions are the readout census's own
(``scripts/analyze_readout_candidates.py``: ray_max 6.0, blocked ray 0.5, ray slots
front=0 / left=2 / right=3, 30 ticks). The seventh is the front approach ramp, whose peak jump the
``ESCAPE_ON`` and ``ESCAPE_URGENCY_SPAN`` derivation rules are written against: front-ray
proximity rising linearly from 0 to blocked over 20 ticks (the
``scripts/probe_loom_encoder.py`` ramp geometry, verbatim), then held for the remaining 9
ticks of the 30-tick window.

The eighth and ninth are the same ramp on ONE lateral ray -- left only, then right only,
with every other ray held open. They exist for one reason: the declared ``dna01_turn`` sign
convention says positive means "turn toward the body's right", chosen so the escape-v2
census rows read as turning AWAY from a blocked side, and no held condition under loom-v1
produces a nonzero bias at all (the approach gate leaves a standing wall almost silent, so
the difference stays inside the deadband). A one-sided APPROACH is the only stimulus in
this battery that could exercise the convention. Whether it does is a measurement, reported
below either way; a bias that stays zero, or comes out same-signed on both sides, means the
convention is unexercised and the CRUISE turn bias is inert -- which is a result, not a
failure of the probe.

The tenth is a contact ONSET: the collision flag false for ten ticks, then true for the
rest. It exists because the census's ``collision_on`` condition holds the flag true from
reset, and under the encoder's declared first-tick rule (``m0 = raw0``) a held input is its
own baseline -- so that condition is bit-identical to ``all_open_bright`` and says nothing
about touch. Only a rising edge can. Without this row, any sentence about what contact does
under this policy would be describing a condition that never presented one.

Usage::

    UV_CACHE_DIR=.uv-cache uv run --frozen python scripts/probe_program_policy.py
    UV_CACHE_DIR=.uv-cache uv run --frozen python scripts/probe_program_policy.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO_ROOT))

from neurocraft_protocol import Observation, ProgramAction, SensorFrame, SensorFrameV2  # noqa: E402

if TYPE_CHECKING:  # heavy imports stay inside build_controller (numpy/pyarrow)
    from neurocraft_controller.program_policy import ProgramSelectingController

DEFAULT_ARTIFACT_DIR = REPO_ROOT / "data" / "processed" / "male-cns-v1.0-superclass-v1"
DEFAULT_ANNOTATIONS = (
    REPO_ROOT / "data" / "cache" / "male-cns-v1.0" / "body-neurotransmitters-male-cns-v1.0.feather"
)
DEFAULT_BODY_ANNOTATIONS = (
    REPO_ROOT
    / "data"
    / "cache"
    / "male-cns-v1.0"
    / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)
DEFAULT_CALIBRATION = REPO_ROOT / "data" / "calibration" / "full-readout-calibration-loom-v1.json"

#: The census battery's declared geometry, verbatim.
HOLD_TICKS = 30
RAY_MAX = 6.0
BLOCKED_RAY = 0.5
GAME_TICK_NS = 50_000_000

#: The approach ramp's declared geometry, verbatim from ``probe_loom_encoder.py``:
#: proximity rising linearly 0 -> blocked over 20 ticks (21 frames, including the
#: all-open frame at step 0), then held.
RAMP_TICKS = 20
BLOCKED_PROXIMITY = 1.0 - BLOCKED_RAY / RAY_MAX

#: The tick at which the contact-onset condition's collision flag rises.
CONTACT_ONSET_TICK = 10

#: Census ray slots, verbatim.
FRONT_RAY_SLOT = 0
LEFT_RAY_SLOT = 2
RIGHT_RAY_SLOT = 3

#: The three approach ramps: front (the one the constants are derived from), then the two
#: one-sided ones that exist to exercise the declared dna01_turn sign convention.
APPROACH_RAMPS: tuple[tuple[str, int], ...] = (
    ("approach_ramp", FRONT_RAY_SLOT),
    ("approach_ramp_left", LEFT_RAY_SLOT),
    ("approach_ramp_right", RIGHT_RAY_SLOT),
)

#: (name, blocked ray slot or None, local light, raw horizontal collision)
HELD_CONDITIONS: tuple[tuple[str, int | None, int, bool], ...] = (
    ("all_open_bright", None, 15, False),
    ("all_open_dark", None, 0, False),
    ("blocked_front", 0, 15, False),
    ("blocked_left", 2, 15, False),
    ("blocked_right", 3, 15, False),
    ("collision_on", None, 15, True),
)


class ProgramPolicyProbeError(RuntimeError):
    """The probe could not run exactly as declared."""


def _frame(rays: Sequence[float], light: int, collision: bool) -> SensorFrame:
    if len(rays) != 6:
        raise ProgramPolicyProbeError("the declared sensor frame carries exactly six rays")
    return SensorFrame(
        position_m=(0.0, 66.0, 0.0),
        velocity_m_s=(0.0, 0.0, 0.0),
        yaw_rad=0.0,
        pitch_rad=0.0,
        on_ground=False,
        horizontal_collision=collision,
        vertical_collision=False,
        in_fluid=False,
        local_light=light,
        ray_max_distance_m=RAY_MAX,
        ray_distances_m=(rays[0], rays[1], rays[2], rays[3], rays[4], rays[5]),
    )


def held_frames(slot: int | None, light: int, collision: bool) -> list[SensorFrame]:
    """One held condition: the same frame for all 30 ticks."""

    rays = [RAY_MAX] * 6
    if slot is not None:
        rays[slot] = BLOCKED_RAY
    return [_frame(rays, light, collision)] * HOLD_TICKS


def contact_onset_frames(onset_tick: int = CONTACT_ONSET_TICK) -> list[SensorFrame]:
    """Every ray open; the collision flag rises at ``onset_tick`` and stays true.

    The census's held ``collision_on`` condition is bit-identical to ``all_open_bright``
    because the encoder's first-tick rule makes a held input its own baseline. A rising
    edge is the only contact this encoder can see, so it is the only contact worth probing.
    """

    rays = [RAY_MAX] * 6
    return [_frame(rays, 15, tick >= onset_tick) for tick in range(HOLD_TICKS)]


def approach_ramp_frames(slot: int = 0) -> list[SensorFrame]:
    """One ray's proximity rising 0 -> blocked over 20 ticks, then held to 30 ticks.

    ``slot`` selects which ray approaches, in the census's own slot order (front=0,
    left=2, right=3); every other ray stays open for the whole sequence.
    """

    def rays(distance: float) -> list[float]:
        values = [RAY_MAX] * 6
        values[slot] = distance
        return values

    ramp = [
        _frame(rays(RAY_MAX * (1.0 - BLOCKED_PROXIMITY * step / RAMP_TICKS)), 15, False)
        for step in range(RAMP_TICKS + 1)
    ]
    held = [_frame(rays(BLOCKED_RAY), 15, False)] * (HOLD_TICKS - len(ramp))
    return [*ramp, *held]


def build_controller(
    *,
    artifact_directory: Path,
    annotations: Path,
    body_annotations: Path,
    calibration: Path,
    threads: int | None,
    loom_version: int = 1,
) -> ProgramSelectingController:
    """The selected loom generation wrapped in the unchanged policy rules."""

    from neurocraft_controller.full_brain_controller import (
        FullBrainConfiguration,
        FullBrainController,
    )
    from neurocraft_controller.program_policy import ProgramSelectingController

    inner = FullBrainController.from_artifact(
        run_id="program-policy-probe",
        entity_id="probe-fly",
        configuration=FullBrainConfiguration(kernel_threads=threads),
        artifact_directory=artifact_directory,
        annotations_path=annotations,
        calibration_path=calibration,
        embodied_senses=True,
        body_annotations_path=body_annotations,
        escape_reflex=True,
        escape_version=2,
        loom_encoder=True,
        loom_version=loom_version,
        steering_readout=True,
    )
    return ProgramSelectingController(inner)


def run_condition(
    controller: ProgramSelectingController,
    frames: Sequence[SensorFrame],
    start_tick: int,
) -> list[ProgramAction]:
    """Step one condition from a fresh reset; return the issued frames.

    ``reset()`` clears the neural state, the encoder's running means, and the policy state
    together, so each condition starts from the same declared first tick and no condition
    can inherit another's hold, refractory, or arming latch.
    """

    controller.reset()
    issued: list[ProgramAction] = []
    for offset, sensors in enumerate(frames):
        tick = start_tick + offset
        issued.append(
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
        )
    return issued


def summarize(condition: str, issued: Sequence[ProgramAction]) -> dict[str, Any]:
    programs = [frame.program.value for frame in issued]
    counts = {name: programs.count(name) for name in sorted(set(programs))}
    first_escape = programs.index("ESCAPE_JUMP") if "ESCAPE_JUMP" in programs else None
    jumps = [frame.readout.ttmn_jump for frame in issued]
    turns = [frame.readout.dna01_turn for frame in issued]
    wings = [frame.readout.wing_common for frame in issued]
    return {
        "condition": condition,
        "ticks": len(issued),
        "selection_counts": counts,
        "first_escape_tick": first_escape,
        "peak_ttmn_jump": max(jumps),
        "peak_ttmn_jump_tick": jumps.index(max(jumps)),
        "final_dna01_turn": turns[-1],
        "peak_abs_dna01_turn": max(turns, key=abs),
        "final_wing_common": wings[-1],
        "final_speed": issued[-1].modulation.speed,
        "peak_urgency": max(frame.modulation.urgency for frame in issued),
        "series": {
            "ttmn_jump": jumps,
            "dna01_turn": turns,
            "wing_common": wings,
            "program": programs,
            "speed": [frame.modulation.speed for frame in issued],
            "urgency": [frame.modulation.urgency for frame in issued],
            "turn_bias": [frame.parameters.turn_bias for frame in issued],
        },
    }


def _row(summary: dict[str, Any]) -> str:
    counts = summary["selection_counts"]
    first = summary["first_escape_tick"]
    return (
        f"{summary['condition']:<18} "
        f"CRUISE {counts.get('CRUISE', 0):>3}  ESCAPE_JUMP {counts.get('ESCAPE_JUMP', 0):>3}  "
        f"first_escape {'-' if first is None else first:>4}  "
        f"peak_jump {summary['peak_ttmn_jump']:.6f}  "
        f"dna01(final) {summary['final_dna01_turn']:+.6f}  "
        f"wing(final) {summary['final_wing_common']:+.6f}  "
        f"speed {summary['final_speed']:.4f}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loom-version", type=int, choices=(1, 2), default=1)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--body-annotations", type=Path, default=DEFAULT_BODY_ANNOTATIONS)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--json", type=Path, help="also write the full record here")
    args = parser.parse_args(argv)
    if args.loom_version == 2 and args.calibration == DEFAULT_CALIBRATION:
        args.calibration = DEFAULT_CALIBRATION.with_name("full-readout-calibration-loom-v2.json")

    for path, label in (
        (args.artifact_dir / "manifest.json", "full runtime artifact"),
        (args.annotations, "neurotransmitter annotations table"),
        (args.body_annotations, "body-annotations table"),
        (args.calibration, f"loom-v{args.loom_version} readout calibration"),
    ):
        if not Path(path).is_file():
            raise SystemExit(f"this probe is local-only; the {label} is not present at {path}")

    controller = build_controller(
        artifact_directory=args.artifact_dir,
        annotations=args.annotations,
        body_annotations=args.body_annotations,
        calibration=args.calibration,
        threads=args.threads,
        loom_version=args.loom_version,
    )
    steering = controller.steering_binding
    if steering is None:  # pragma: no cover - the constructor refuses this
        raise SystemExit("the program-selecting controller reported no steering binding")
    print("policy_hash               ", controller.policy_hash)
    print("dna01 pair binding sha256 ", steering.identity_sha256)
    print(
        "dna01 members             ",
        ", ".join(
            f"{member.body_id} ({member.soma_side}, dense {member.dense_index})"
            for member in steering.members
        ),
    )
    print("decoder_hash              ", controller.identity.decoder_hash)
    print("adapter_hash              ", controller.identity.adapter_hash)
    print()
    print(
        "Declared hand-chosen mapping (ADR 0019). Selection counts below are what the "
        "policy ASKED FOR under ten synthetic sensor conditions on ONE deterministic "
        "run. They are not evidence that the connectome selected anything, that a fly "
        "would behave this way, or that the mapping is good."
    )
    print()

    def tagged(frames: list[SensorFrame], kind: int) -> list[SensorFrame]:
        if args.loom_version == 1:
            return frames
        return [SensorFrameV2.from_sensor_frame(frame, (kind,) * 6) for frame in frames]

    summaries: list[dict[str, Any]] = []
    tick = 0
    try:
        for name, slot, light, collision in HELD_CONDITIONS:
            issued = run_condition(
                controller,
                tagged(held_frames(slot, light, collision), 1 if slot is not None else 0),
                tick,
            )
            tick += len(issued)
            summaries.append(summarize(name, issued))
        for name, slot in APPROACH_RAMPS:
            issued = run_condition(controller, tagged(approach_ramp_frames(slot), 2), tick)
            tick += len(issued)
            summaries.append(summarize(name, issued))
        issued = run_condition(controller, tagged(contact_onset_frames(), 0), tick)
        tick += len(issued)
        summaries.append(summarize("contact_onset", issued))
    finally:
        controller.engine.end_run()

    for summary in summaries:
        print(_row(summary))

    # The sign convention's only chance to be exercised: print the per-tick bias for the
    # two one-sided approaches side by side, and say plainly what it does or does not show.
    by_name = {summary["condition"]: summary for summary in summaries}
    left = by_name.get("approach_ramp_left")
    right = by_name.get("approach_ramp_right")
    if left is not None and right is not None:
        print()
        print("dna01_turn per tick under the two one-sided approaches")
        print("  left-ray approach : ", " ".join(f"{v:+.3f}" for v in left["series"]["dna01_turn"]))
        print(
            "  right-ray approach: ", " ".join(f"{v:+.3f}" for v in right["series"]["dna01_turn"])
        )
        left_peak = left["peak_abs_dna01_turn"]
        right_peak = right["peak_abs_dna01_turn"]
        print()
        if left_peak == 0.0 and right_peak == 0.0:
            print(
                "The DNa01 bias is exactly zero on both one-sided approaches, so the "
                "declared sign convention is UNEXERCISED by every stimulus in this "
                "battery and the CRUISE turn bias is inert. A HUD reading DNa01 +-0.00 "
                "is not steering; it is a readout the deadband zeroed."
            )
        elif (left_peak > 0.0) == (right_peak > 0.0):
            print(
                "The DNa01 bias is nonzero but SAME-SIGNED on both one-sided approaches, "
                "so it does not distinguish the blocked side and the declared sign "
                "convention is still unexercised as a steering signal."
            )
        else:
            direction = "away from" if left_peak > 0.0 else "toward"
            print(
                "The DNa01 bias is nonzero and OPPOSITE-SIGNED between the two one-sided "
                f"approaches: left-ray peak {left_peak:+.6f}, right-ray peak "
                f"{right_peak:+.6f}. Under the declared convention (positive = toward the "
                f"body's right) that points {direction} the approaching side. This is one "
                "deterministic run of two synthetic stimuli, not evidence about a fly."
            )

    wall_checks = {}
    if args.loom_version == 2:
        for summary in summaries:
            name = summary["condition"]
            summary["ray_hit_kinds"] = [
                2 if name.startswith("approach_ramp") else 1 if name.startswith("blocked_") else 0
            ] * 6
        walls = [s for s in summaries if s["condition"].startswith("blocked_")]
        wall_checks = {
            "cruise_only": all(s["selection_counts"] == {"CRUISE": s["ticks"]} for s in walls),
            "jump_exactly_zero": all(s["peak_ttmn_jump"] == 0.0 for s in walls),
        }
        print("held-wall checks:", wall_checks)
    if args.json is not None:
        document = {
            "schema": f"neurocraft.program_policy_probe.v{args.loom_version}",
            "diagnostic_only": True,
            "policy_hash": controller.policy_hash,
            "policy": controller.policy.to_identity_material(),
            "identity": controller.identity.to_dict(),
            "steering_binding_sha256": steering.identity_sha256,
            "conditions": summaries,
            **({"held_wall_checks": wall_checks} if args.loom_version == 2 else {}),
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.json}")
    return 0 if not wall_checks or wall_checks["cruise_only"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
