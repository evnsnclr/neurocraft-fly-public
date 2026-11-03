#!/usr/bin/env python3
"""Start the neural service with an explicit pinned connectome controller."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import sys
import time
from collections.abc import Callable, Coroutine, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from neurocraft_connectome import load_malecns_fixture
from neurocraft_connectome.flight_pin import (
    MALECNS_FLIGHT_FIXTURE_V1_FILE_BYTES,
    MALECNS_FLIGHT_FIXTURE_V1_PIN,
)
from neurocraft_controller import (
    WING_MOTOR_POOL_PROFILE,
    ConnectomeControlHandler,
    ConnectomeController,
    ConnectomeRuntimeControlHandler,
    ControllerConfiguration,
    ControllerIntegrityError,
    DeclaredTransitionForces,
    FlightConnectomeController,
    FlightControllerConfiguration,
    MinecraftReceiptTraceRecorder,
    NamedIntervention,
    NamedLesion,
    NamedStimulation,
)
from neurocraft_controller.trained_readout_controller import (
    TrainedReadoutController,
    TrmDemoConfiguration,
)
from neurocraft_protocol import (
    Action,
    ActuationReceipt,
    Command,
    ProgramReceipt,
    RunIdentity,
    TraceManifest,
    TraceWriter,
    canonical_json_bytes,
    parse_json_frame,
)
from neurocraft_service import ActionTelemetry, ServiceConfig, serve
from neurocraft_service.lane_sessions import (
    LaneName,
    LaneSessionDispatcher,
    LaneSpec,
    SessionHooks,
    create_lane_server,
)

# ``python scripts/run_connectome_service.py`` places ``scripts/`` rather than the
# repository root on sys.path. Add only this script's resolved repository root so
# the sibling latency sink has the same import identity under direct CLI and test
# use; this is the convention analyze_reachability.py already follows.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.action_latency_telemetry import ActionLatencyWriter
from scripts.program_decision_telemetry import ProgramDecisionWriter

if TYPE_CHECKING:
    from neurocraft_controller.brain_display import BrainDisplayBinding
    from neurocraft_controller.full_brain_controller import FullBrainController
    from neurocraft_controller.program_policy import ProgramSelectingController

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "data" / "fixtures" / "mcns-v1.0-flight-power-v1.json"

#: ``pair_v1`` is the shipped default: the exact v1 controller construction,
#: byte-for-byte. ``wing_motor_soma_side_v1`` is the opt-in flight controller
#: over the pinned 128-node steering fixture, ``trm_cloned_readout_v1`` is
#: the opt-in trained-readout demo controller over the same pinned fixture,
#: bound to the committed cloned-readout artifact's derived digest,
#: ``full_mcns_wing_v1`` is the opt-in local-only full-artifact profile
#: (whole-brain recurrent context behind the same wing boundary), and
#: ``full_mcns_eem_v1`` is the opt-in local-only Engineered-Encoder profile
#: (ADR 0008 Mode 3: the same full artifact and wing readout with Minecraft
#: observations additionally driving the labeled LC4/visual/bristle sensory
#: populations through a declared, hashed encoder), ``full_mcns_eem_v2``
#: is its temporal-contrast revision (each channel drives its population with
#: the deviation from a declared leaky running mean; touch stays absolute),
#: ``full_mcns_escape_v1`` is the escape-reflex profile (the eem v1
#: sensory boundary plus the declared TTMn jump readout on vertical thrust),
#: ``full_mcns_escape_v2`` is its transient-touch revision (touch alone
#: under the declared temporal contrast, jump mapping rectified upward), and
#: ``full_mcns_loom_v1`` is the approach-gated revision of that (the three
#: threat channels fire on the rise of proximity, not on its level, so a
#: standing wall is no longer a standing escape);
#: none activates without the explicit ``--controller-profile`` flag.
PAIR_CONTROLLER_PROFILE = "pair_v1"
TRM_CLONED_READOUT_PROFILE = "trm_cloned_readout_v1"
FULL_MCNS_WING_PROFILE = "full_mcns_wing_v1"
FULL_MCNS_EEM_PROFILE = "full_mcns_eem_v1"
#: ``full_mcns_eem_v2`` is the temporal-contrast Engineered-Encoder profile
#: (identity recorded in experiments/preregistration/eem-embodied-senses-v2.md):
#: the same populations and per-capita placement as eem v1, but each channel
#: drives its population with the deviation from a declared leaky running
#: mean (alpha = 1/8 per game tick) instead of the absolute level; touch
#: stays absolute.
FULL_MCNS_EEM_V2_PROFILE = "full_mcns_eem_v2"
#: ``full_mcns_escape_v1`` is the escape-reflex Engineered-Encoder profile
#: (identity recorded in experiments/preregistration/eem-escape-v1.md): the
#: eem v1 sensory boundary verbatim (absolute encoder over the labeled
#: populations) plus the declared TTMn jump-motoneuron readout, whose pooled
#: activity adds one declared term to the vertical thrust axis.
FULL_MCNS_ESCAPE_PROFILE = "full_mcns_escape_v1"
#: ``full_mcns_escape_v2`` is the escape-reflex v2 profile (identity recorded
#: in experiments/preregistration/eem-escape-v2.md): the same populations,
#: TTMn pool, and per-capita placement as escape v1, with the ``touch``
#: channel alone moving under the frozen temporal-contrast formula and the
#: jump mapping rectified to the non-negative half of its declared codomain.
FULL_MCNS_ESCAPE_V2_PROFILE = "full_mcns_escape_v2"
#: ``full_mcns_loom_v1`` is the approach-gated looming profile (declared in
#: docs/encoders/eem-loom-v1.md): the same populations, TTMn pool, per-capita
#: placement, and rectified jump decoder as escape v2, with the three threat
#: channels driving their LC4 populations with the rectified RISE of proximity
#: above a declared leaky running mean instead of with the absolute level, so
#: an approaching object fires and a standing wall does not.
FULL_MCNS_LOOM_V1_PROFILE = "full_mcns_loom_v1"
#: ``full_mcns_loom_program_v1`` is the body-program lane over that same
#: controller: identical artifact, encoder, dynamics, and decoder, plus the
#: declared DNa01 steering readout, wrapped in a ProgramSelectingController that
#: issues ``program_action`` frames instead of raw motor actions. The mapping
#: from the three labeled readouts to a scripted body program is hand-chosen
#: engineering (ADR 0019), its identity rides on every frame as ``policy_hash``,
#: and the connectome is not deciding anything.
FULL_MCNS_LOOM_PROGRAM_V1_PROFILE = "full_mcns_loom_program_v1"
FULL_MCNS_LOOM_V2_PROFILE = "full_mcns_loom_v2"
FULL_MCNS_LOOM_PROGRAM_V2_PROFILE = "full_mcns_loom_program_v2"
FULL_MCNS_EVENTS_PROGRAM_V1_PROFILE = "full_mcns_events_program_v1"
FULL_MCNS_INTERACTIONS_PROGRAM_V2_PROFILE = "full_mcns_interactions_program_v2"
FULL_MCNS_LOOM_V2_PROFILES = (FULL_MCNS_LOOM_V2_PROFILE, FULL_MCNS_LOOM_PROGRAM_V2_PROFILE)
FULL_MCNS_EEM_PROFILES = (FULL_MCNS_EEM_PROFILE, FULL_MCNS_EEM_V2_PROFILE)
FULL_MCNS_ESCAPE_PROFILES = (FULL_MCNS_ESCAPE_PROFILE, FULL_MCNS_ESCAPE_V2_PROFILE)
#: Every profile built over the loom-v1 (approach-gated) encoder and its own
#: calibration document: the raw-action lane and the body-program lane.
FULL_MCNS_LOOM_V1_PROFILES = (
    FULL_MCNS_LOOM_V1_PROFILE,
    FULL_MCNS_LOOM_PROGRAM_V1_PROFILE,
)
#: Every profile that issues ``program_action`` frames rather than raw actions.
FULL_MCNS_LOOM_PROFILES = (*FULL_MCNS_LOOM_V1_PROFILES, *FULL_MCNS_LOOM_V2_PROFILES)
FULL_MCNS_PROGRAM_PROFILES = (
    FULL_MCNS_LOOM_PROGRAM_V1_PROFILE,
    FULL_MCNS_LOOM_PROGRAM_V2_PROFILE,
    FULL_MCNS_EVENTS_PROGRAM_V1_PROFILE,
    FULL_MCNS_INTERACTIONS_PROGRAM_V2_PROFILE,
)
FULL_MCNS_COMPARISON_PROFILES = (
    FULL_MCNS_LOOM_PROGRAM_V2_PROFILE,
    FULL_MCNS_EVENTS_PROGRAM_V1_PROFILE,
    FULL_MCNS_INTERACTIONS_PROGRAM_V2_PROFILE,
)
#: Every profile whose calibration carries the two named scales
#: {wing_scale, jump_scale} and the ``jump_readout`` section: the two escape
#: generations plus the loom-v1 profiles, which reuse the escape-v2 decoder
#: unchanged.
FULL_MCNS_JUMP_READOUT_PROFILES = (
    *FULL_MCNS_ESCAPE_PROFILES,
    *FULL_MCNS_LOOM_PROFILES,
    FULL_MCNS_EVENTS_PROGRAM_V1_PROFILE,
    FULL_MCNS_INTERACTIONS_PROGRAM_V2_PROFILE,
)
#: Every profile that selects populations from the body-annotations table
#: (the eem profiles plus the jump-readout profiles, whose TTMn selection uses
#: the same digest-verified join).
FULL_MCNS_BODY_ANNOTATION_PROFILES = (
    *FULL_MCNS_EEM_PROFILES,
    *FULL_MCNS_JUMP_READOUT_PROFILES,
)
FULL_MCNS_PROFILES = (FULL_MCNS_WING_PROFILE, *FULL_MCNS_BODY_ANNOTATION_PROFILES)
CONTROLLER_PROFILES = (
    PAIR_CONTROLLER_PROFILE,
    WING_MOTOR_POOL_PROFILE,
    TRM_CLONED_READOUT_PROFILE,
    FULL_MCNS_WING_PROFILE,
    FULL_MCNS_EEM_PROFILE,
    FULL_MCNS_EEM_V2_PROFILE,
    FULL_MCNS_ESCAPE_PROFILE,
    FULL_MCNS_ESCAPE_V2_PROFILE,
    FULL_MCNS_LOOM_V1_PROFILE,
    FULL_MCNS_LOOM_PROGRAM_V1_PROFILE,
    FULL_MCNS_LOOM_V2_PROFILE,
    FULL_MCNS_LOOM_PROGRAM_V2_PROFILE,
    FULL_MCNS_EVENTS_PROGRAM_V1_PROFILE,
    FULL_MCNS_INTERACTIONS_PROGRAM_V2_PROFILE,
)
TRM_READOUT_ARTIFACT_PATH = ROOT / "data" / "readouts" / "trm-cloned-readout-v2.json"
#: Local-only inputs for the full-artifact profiles. All are gitignored bulk
#: data; the artifact manifest is pinned externally inside
#: ``neurocraft_controller.full_brain_controller`` (RUNTIME-MCNS-01).
DEFAULT_FULL_ARTIFACT_DIR = ROOT / "data" / "processed" / "male-cns-v1.0-superclass-v1"
FULL_ANNOTATIONS_PATH = (
    ROOT / "data" / "cache" / "male-cns-v1.0" / "body-neurotransmitters-male-cns-v1.0.feather"
)
#: The EEM profile's additional local-only input: the pinned body-annotations
#: table its population selectors are verified against.
FULL_BODY_ANNOTATIONS_PATH = (
    ROOT / "data" / "cache" / "male-cns-v1.0" / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)
#: The committed declared readout calibrations for the full profiles. Unlike
#: the artifact and the annotations tables these files ARE tracked: each is
#: one measured scalar plus its provenance, and the controller binds its
#: SHA-256 into both the configuration and the decoder identity. Each full
#: profile loads its own document by default (the loader refuses the other
#: profile's document in both directions); ``--full-uncalibrated`` opts out
#: for an explicit A/B comparison.
DEFAULT_FULL_CALIBRATION = ROOT / "data" / "calibration" / "full-readout-calibration-v1.json"
DEFAULT_FULL_EEM_CALIBRATION = (
    ROOT / "data" / "calibration" / "full-readout-calibration-eem-v1.json"
)
#: The eem-v2 calibration is a phase-2 deliverable: until it is measured and
#: committed the profile refuses to start (naming the measurement command)
#: unless ``--full-uncalibrated`` opts out explicitly.
DEFAULT_FULL_EEM_V2_CALIBRATION = (
    ROOT / "data" / "calibration" / "full-readout-calibration-eem-v2.json"
)
#: The escape calibration ({wing_scale, jump_scale}, its own schema) is a
#: phase-2 deliverable of the escape lane: until it is measured and committed
#: the profile refuses to start (naming the measurement command) unless
#: ``--full-uncalibrated`` opts out explicitly.
DEFAULT_FULL_ESCAPE_CALIBRATION = (
    ROOT / "data" / "calibration" / "full-readout-calibration-escape-v1.json"
)
#: The escape-v2 calibration (its own schema and stream id) is a phase-2
#: deliverable of the escape-v2 lane: until it is measured and committed the
#: profile refuses to start (naming the measurement command) unless
#: ``--full-uncalibrated`` opts out explicitly.
DEFAULT_FULL_ESCAPE_V2_CALIBRATION = (
    ROOT / "data" / "calibration" / "full-readout-calibration-escape-v2.json"
)
#: The loom-v1 calibration (its own schema and stream id), measured under the
#: approach-gated encoder; the profile refuses to start until it is present
#: unless ``--full-uncalibrated`` opts out explicitly.
DEFAULT_FULL_LOOM_V1_CALIBRATION = (
    ROOT / "data" / "calibration" / "full-readout-calibration-loom-v1.json"
)

DEFAULT_FULL_LOOM_V2_CALIBRATION = (
    ROOT / "data" / "calibration" / "full-readout-calibration-loom-v2.json"
)


def _full_profile_calibration(profile: str) -> tuple[Path, str]:
    """The profile's committed calibration path and the command that measures it."""

    return {
        # This is explicitly a reused decoder component. It does not calibrate
        # or validate the candidate's added event drives or selection thresholds.
        FULL_MCNS_INTERACTIONS_PROGRAM_V2_PROFILE: (
            DEFAULT_FULL_LOOM_V2_CALIBRATION,
            "scripts/calibrate_full_readout.py --profile loom-v2 (decoder component only)",
        ),
        FULL_MCNS_EVENTS_PROGRAM_V1_PROFILE: (
            DEFAULT_FULL_LOOM_V2_CALIBRATION,
            "scripts/calibrate_full_readout.py --profile loom-v2 (decoder component only)",
        ),
        FULL_MCNS_WING_PROFILE: (
            DEFAULT_FULL_CALIBRATION,
            "scripts/calibrate_full_readout.py",
        ),
        FULL_MCNS_EEM_PROFILE: (
            DEFAULT_FULL_EEM_CALIBRATION,
            "scripts/calibrate_full_readout.py --profile eem",
        ),
        FULL_MCNS_EEM_V2_PROFILE: (
            DEFAULT_FULL_EEM_V2_CALIBRATION,
            "scripts/calibrate_full_readout.py --profile eem-v2",
        ),
        FULL_MCNS_ESCAPE_PROFILE: (
            DEFAULT_FULL_ESCAPE_CALIBRATION,
            "scripts/calibrate_full_readout.py --profile escape",
        ),
        FULL_MCNS_ESCAPE_V2_PROFILE: (
            DEFAULT_FULL_ESCAPE_V2_CALIBRATION,
            "scripts/calibrate_full_readout.py --profile escape-v2",
        ),
        FULL_MCNS_LOOM_V1_PROFILE: (
            DEFAULT_FULL_LOOM_V1_CALIBRATION,
            "scripts/calibrate_full_readout.py --profile loom-v1",
        ),
        # The body-program lane wraps the loom-v1 controller unchanged, so it
        # loads the same committed calibration document.
        FULL_MCNS_LOOM_PROGRAM_V1_PROFILE: (
            DEFAULT_FULL_LOOM_V1_CALIBRATION,
            "scripts/calibrate_full_readout.py --profile loom-v1",
        ),
        **{
            profile: (
                DEFAULT_FULL_LOOM_V2_CALIBRATION,
                "scripts/calibrate_full_readout.py --profile loom-v2",
            )
            for profile in FULL_MCNS_LOOM_V2_PROFILES
        },
    }[profile]


def _full_brain_controller_type() -> type[FullBrainController]:
    """The full-brain controller class, imported lazily (numpy/pyarrow at import)."""

    from neurocraft_controller.full_brain_controller import FullBrainController

    return FullBrainController


def record_issued_command(recorder: MinecraftReceiptTraceRecorder, command: Command) -> None:
    """Record one issued fixture-lane ``action``; refuse a ``program_action``, which cannot be.

    The receipt trace pipeline is fixture-typed — ``--trace-out`` is refused for every full
    profile — and the ``program_record`` trace entry for body-program frames is disclosed open
    work, so no recorder can record one today. A recorder is only ever built for
    ``ConnectomeController``, which issues ``action`` frames, so this refusal is unreachable in
    every profile that exists; it fails closed rather than dropping a frame the trace would then
    be silently missing.
    """

    if not isinstance(command, Action):
        raise TypeError("the fixture receipt trace recorder cannot record a program_action frame")
    recorder.accept_issued_action(command)


def record_receipt(
    recorder: MinecraftReceiptTraceRecorder, receipt: ActuationReceipt | ProgramReceipt
) -> None:
    """Record one authoritative fixture-lane actuation receipt; refuse a program receipt.

    Same boundary as :func:`record_issued_command`, from the authority's side.
    """

    if isinstance(receipt, ProgramReceipt):
        raise TypeError("the fixture receipt trace recorder cannot record a program_receipt frame")
    recorder.accept_receipt(receipt)


def record_action_telemetry(
    recorder: MinecraftReceiptTraceRecorder, telemetry: ActionTelemetry
) -> None:
    """Adapt measured service telemetry into the trace recorder's package boundary."""

    recorder.accept_action_telemetry(
        command_seq=telemetry.command_seq,
        action_sha256=telemetry.action_sha256,
        queue_coalesced=telemetry.queue_coalesced,
        queue_overruns=telemetry.queue_overruns,
        compute_ns=telemetry.compute_ns,
        serialization_ns=telemetry.serialization_ns,
        write_ns=telemetry.write_ns,
    )


class PresentationTelemetryWriter:
    """Append presentation-only NDJSON samples for the local brain viewer.

    Never an evidence surface: each sample is a convenience copy of the
    controller's per-step evidence, keyed by command_seq so a mismatched or
    missing evidence frame is skipped rather than fabricated. Causal evidence
    stays in the hash-linked trace recorder. Best-effort by contract: the
    service treats sink exceptions as terminal trace faults, so a failure
    here must disable this writer, never abort the live action path.
    """

    def __init__(
        self,
        path: Path,
        controller: ConnectomeController | FullBrainController,
        profile: str,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")  # truncate: the viewer tails live samples only
        self._path = path
        self._controller = controller
        self._profile = profile
        self._disabled = False

    def accept(self, telemetry: ActionTelemetry) -> None:
        if self._disabled:
            return
        evidence = self._controller.last_evidence
        if evidence is None or evidence.command_seq != telemetry.command_seq:
            return
        try:
            sample = {
                "presentation_only": True,
                "run_id": self._controller.identity.run_id,
                "controller_profile": self._profile,
                "game_tick": evidence.observation.game_tick,
                "command_seq": evidence.command_seq,
                "activity": [round(value, 5) for value in evidence.state_after_activity],
                "encoded": {
                    name: round(value, 5) for name, value in evidence.encoded_stimulation.items()
                },
                "motor_frame": {
                    name: round(value, 5) for name, value in evidence.motor_frame.values
                },
            }
            with self._path.open("ab") as stream:
                stream.write(json.dumps(sample, sort_keys=True).encode() + b"\n")
        except Exception as exc:  # deliberate: presentation must never fault the loop
            self._disabled = True
            print(
                f"presentation telemetry disabled after write failure: {exc}",
                file=sys.stderr,
            )


def _read_readout_artifact(path: Path) -> tuple[bytes, dict[str, Any], str]:
    """Read, parse, and hash the committed readout artifact; the digest is
    always derived from the file at call time, never pinned here."""

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SystemExit(f"could not read the committed readout artifact: {exc.strerror}") from exc
    try:
        parsed: object = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit("committed readout artifact is not valid JSON") from exc
    if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
        raise SystemExit("committed readout artifact must be a JSON object")
    return raw, cast(dict[str, Any], parsed), hashlib.sha256(raw).hexdigest()


async def _serve_with_idle_rewarm(
    serve_coroutine: Coroutine[Any, Any, None],
    controller: FullBrainController | ProgramSelectingController,
    *,
    interval_seconds: float = 5.0,
    continue_while: Callable[[], bool] | None = None,
) -> None:
    """Serve while re-warming the full-brain compute path until the first step.

    Minecraft's server and client boot for minutes between service start and
    fly spawn, evicting the warmth the startup pass created; the mod then
    grants the first live action a hard 2-tick deadline. Controller steps run
    in a worker thread (the service de-blinds compute off the event loop), so
    a warm pass could overlap the first in-flight step; the controller's
    compute lock makes them mutually exclusive instead — a warm pass that
    would contend simply skips (the numpy fallback kernel caches a mutable
    scratch buffer, so overlap would be a real data race there). The warm
    pass runs on this event loop and CAN delay loop progress by its own
    length (~3 ms, ~3% of the first-action budget, ~0.05% duty cycle), a
    disclosed residual accepted in exchange for removing the near-certain
    cold-start spike. Stops forever once stepping begins; neural state is
    untouched (warm_compute_path asserts that). A completed command remains a
    permanent stop within the session even if RESET occurs between polling times.
    The optional ownership predicate permanently stops warming after handover.

    Each pass's own cost is printed rather than discarded, because it is the
    only direct measurement of the page-eviction hypothesis: the pass firing
    immediately before spawn costs about as much as a steady-state warm pass
    if nothing was evicted, and far more if the compute path went cold. The
    print itself adds microseconds to a pass already disclosed at ~3 ms.
    """

    async def rewarm() -> None:
        pass_index = 0

        def eligible() -> bool:
            return not controller.session_has_stepped and (
                continue_while is None or continue_while()
            )

        while eligible():
            await asyncio.sleep(interval_seconds)
            if not eligible():
                break
            seconds = controller.warm_compute_path(passes=1)
            pass_index += 1
            # warm_compute_path returns exactly 0.0 only when it declined the
            # contended compute lock; a pass that ran always spends time.
            measurement = (
                "skipped (a step held the compute lock)"
                if seconds == 0.0
                else f"took {seconds * 1000.0:.2f} ms"
            )
            print(f"idle re-warm pass {pass_index} {measurement}", flush=True)

    rewarm_task = asyncio.create_task(rewarm())
    try:
        await serve_coroutine
    finally:
        rewarm_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await rewarm_task


def _refuse_full_profile_conflicts(args: argparse.Namespace) -> None:
    """Fail closed on option combinations the full-artifact profile excludes.

    Every refusal here triggers BEFORE any artifact byte is read, so a
    misconfigured invocation is rejected in milliseconds rather than after a
    multi-second verified load.
    """

    if (
        args.program_decisions_out is not None
        and args.controller_profile not in FULL_MCNS_PROGRAM_PROFILES
    ):
        raise SystemExit(
            "--program-decisions-out applies to the body-program profiles only "
            f"({', '.join(FULL_MCNS_PROGRAM_PROFILES)}): every other profile issues raw "
            "action frames, which carry no program selection to record"
        )
    if args.full_uncalibrated and args.controller_profile not in FULL_MCNS_PROFILES:
        raise SystemExit(
            "--full-uncalibrated applies to the full MaleCNS profiles only "
            "(full_mcns_wing_v1, full_mcns_eem_v1, full_mcns_eem_v2, "
            "full_mcns_escape_v1, full_mcns_escape_v2, full_mcns_loom_v1): no other "
            "profile carries a declared readout calibration"
        )
    profile = args.controller_profile
    if profile not in FULL_MCNS_PROFILES:
        if getattr(args, "kernel_threads", None) is not None:
            raise SystemExit("--kernel-threads applies only to full MaleCNS profiles")
        return
    calibration_path, calibration_hint = _full_profile_calibration(profile)
    if not args.full_uncalibrated and not calibration_path.is_file():
        raise SystemExit(
            f"{profile} loads the declared readout calibration at "
            f"{calibration_path}, which is not present; measure it with "
            f"{calibration_hint}, or pass --full-uncalibrated to run "
            "the uncalibrated (scale 1.0) readout deliberately"
        )
    if profile in FULL_MCNS_BODY_ANNOTATION_PROFILES:
        if not FULL_BODY_ANNOTATIONS_PATH.is_file():
            raise SystemExit(
                f"{profile} is a local-only profile; the body-annotations "
                f"table is not present at {FULL_BODY_ANNOTATIONS_PATH} (fetch it "
                "with scripts/acquire_malecns.py --include-metadata)"
            )
        if not args.full_uncalibrated and profile in FULL_MCNS_JUMP_READOUT_PROFILES:
            # Cheap shape gate for the jump-readout profiles: each calibration
            # carries its own schema ({wing_scale, jump_scale} plus the TTMn
            # pool identity), so a wing/eem document -- or ANOTHER
            # generation's document -- at this path is refused here in
            # milliseconds; the exact encoder and TTMn identities are still
            # cross-checked against the loaded binding inside from_artifact.
            from neurocraft_controller.full_brain_controller import (
                FullBrainProfileError,
                load_escape_readout_calibration,
                load_escape_v2_readout_calibration,
                load_loom_v1_readout_calibration,
                load_loom_v2_readout_calibration,
            )

            # Keyed off the loom profile TUPLE, not the single raw-lane name: every
            # profile in FULL_MCNS_JUMP_READOUT_PROFILES reaches this gate, and a
            # literal-per-profile mapping silently becomes a KeyError the moment a
            # profile joins that tuple without joining this one. Pinned by
            # `test_every_jump_readout_profile_has_a_cheap_calibration_gate`.
            loaders: dict[str, Callable[[bytes], Any]] = {
                FULL_MCNS_EVENTS_PROGRAM_V1_PROFILE: load_loom_v2_readout_calibration,
                FULL_MCNS_INTERACTIONS_PROGRAM_V2_PROFILE: load_loom_v2_readout_calibration,
                FULL_MCNS_ESCAPE_PROFILE: load_escape_readout_calibration,
                FULL_MCNS_ESCAPE_V2_PROFILE: load_escape_v2_readout_calibration,
            }
            for loom_profile in FULL_MCNS_LOOM_V1_PROFILES:
                loaders[loom_profile] = load_loom_v1_readout_calibration
            for loom_profile in FULL_MCNS_LOOM_V2_PROFILES:
                loaders[loom_profile] = load_loom_v2_readout_calibration
            missing = sorted(set(FULL_MCNS_JUMP_READOUT_PROFILES) - set(loaders))
            if missing:  # pragma: no cover - the pin above makes this unreachable
                raise SystemExit(
                    f"no cheap calibration gate is declared for {missing}; refusing to "
                    "start a jump-readout profile whose calibration shape is unchecked"
                )
            load_escape = loaders[profile]
            try:
                load_escape(calibration_path.read_bytes())
            except (OSError, FullBrainProfileError) as exc:
                raise SystemExit(
                    f"could not validate the escape calibration at {calibration_path}: {exc}"
                ) from exc
        elif not args.full_uncalibrated:
            # Cheap shape gate before the multi-second verified load: a wing
            # calibration at an EEM path is refused here in milliseconds; the
            # exact encoder identity (which also separates eem v1 from eem v2)
            # is still cross-checked against the loaded binding inside
            # from_artifact.
            from neurocraft_controller.full_brain_controller import (
                FullBrainProfileError,
                load_readout_calibration,
            )

            try:
                calibration = load_readout_calibration(calibration_path.read_bytes())
            except (OSError, FullBrainProfileError) as exc:
                raise SystemExit(
                    f"could not validate the EEM calibration at {calibration_path}: {exc}"
                ) from exc
            if calibration.encoder_identity_sha256 == "":
                raise SystemExit(
                    f"the calibration at {calibration_path} was measured without an "
                    "engineered-encoder binding (it is a wing calibration); "
                    f"{profile} refuses it -- measure with {calibration_hint}"
                )
    if any(
        value is not None for value in (args.trace_out, args.trace_manifest, args.transition_forces)
    ):
        raise SystemExit(
            f"{profile} does not support --trace-out/--trace-manifest/"
            "--transition-forces: the receipt trace pipeline is fixture-typed, and "
            "full-graph tracing is disclosed open work"
        )
    if args.mode == "lesion":
        raise SystemExit(
            f"{profile} supports --mode faithful only: lesion interventions "
            "exceed the per-tick compute budget at full scale (disclosed v1 limitation)"
        )
    if args.lesion_body_id or args.stimulate or args.declare_lesion or args.declare_stimulation:
        raise SystemExit(
            f"{profile} does not support lesion/stimulation options: "
            "interventions exceed the per-tick compute budget at full scale "
            "(disclosed v1 limitation)"
        )
    if args.fixture != DEFAULT_FIXTURE:
        raise SystemExit(
            f"{profile} reads the full runtime artifact, not a fixture; "
            "use --full-artifact-dir instead of --fixture"
        )
    if args.presentation_telemetry_out is not None:
        raise SystemExit(
            f"{profile} does not support --presentation-telemetry-out: the "
            "brain viewer renders the steering fixture, and the full profile's "
            "evidence exposes a declared boundary subset the writer would emit "
            "as unlabeled activity"
        )
    # ``--action-latency-out`` is deliberately absent from this list. The refusal
    # above is about neural content, not about telemetry: ActionLatencyWriter reads
    # no controller evidence at all, only the sequence number and the three measured
    # durations the service hands its sink, so nothing it writes could be mistaken
    # for full-graph activity.


def _build_full_brain_controller(args: argparse.Namespace) -> FullBrainController:
    """Construct the full-artifact controller; imported lazily because the
    array-native backend needs numpy/pyarrow, which the isolated no-dev demo
    environment deliberately omits for every other profile."""

    from neurocraft_controller.full_brain_controller import (
        FullBrainConfiguration,
        FullBrainController,
        FullBrainProfileError,
    )

    embodied = args.controller_profile in FULL_MCNS_BODY_ANNOTATION_PROFILES
    escape = args.controller_profile in FULL_MCNS_JUMP_READOUT_PROFILES
    interactions = args.controller_profile == FULL_MCNS_INTERACTIONS_PROGRAM_V2_PROFILE
    events = interactions or args.controller_profile == FULL_MCNS_EVENTS_PROGRAM_V1_PROFILE
    loom = args.controller_profile in FULL_MCNS_LOOM_PROFILES or events
    escape_version = 2 if loom or args.controller_profile == FULL_MCNS_ESCAPE_V2_PROFILE else 1
    # The DNa01 steering readout is selected only for the body-program lane: it
    # adds no term to any motor axis and enters no RunIdentity field, so the raw
    # loom-v1 lane's identity is unchanged by its existence.
    steering = args.controller_profile in FULL_MCNS_PROGRAM_PROFILES
    calibration_path = _full_profile_calibration(args.controller_profile)[0]
    try:
        controller = FullBrainController.from_artifact(
            run_id=args.run_id,
            entity_id=args.entity_id,
            session_epoch=args.session_epoch,
            configuration=FullBrainConfiguration(
                mode=args.mode, kernel_threads=getattr(args, "kernel_threads", None)
            ),
            artifact_directory=args.full_artifact_dir,
            annotations_path=FULL_ANNOTATIONS_PATH,
            calibration_path=None if args.full_uncalibrated else calibration_path,
            embodied_senses=embodied,
            embodied_encoder_version=(
                2 if args.controller_profile == FULL_MCNS_EEM_V2_PROFILE else 1
            ),
            body_annotations_path=FULL_BODY_ANNOTATIONS_PATH if embodied else None,
            escape_reflex=escape,
            escape_version=escape_version,
            loom_encoder=loom,
            loom_version=2
            if events or args.controller_profile in FULL_MCNS_LOOM_V2_PROFILES
            else 1,
            event_inputs=events,
            event_inputs_version=2 if interactions else 1,
            steering_readout=steering,
            control_seed=getattr(args, "control_seed", 7),
        )
        print(
            f"full-brain execution kernel_threads={controller.engine.kernel_identity.thread_count}",
            flush=True,
        )
        return controller
    except FullBrainProfileError as exc:
        raise SystemExit(str(exc)) from exc


def _build_program_selecting_controller(
    args: argparse.Namespace,
) -> ProgramSelectingController:
    """Wrap the profile's full-brain controller in the declared selection policy.

    The wrapped controller is byte-for-byte the one the raw lane builds (plus the
    read-only DNa01 binding): same artifact, encoder, dynamics, decoder, and
    ``RunIdentity``. What changes is only which frame kind leaves the service.
    """

    from neurocraft_controller.program_policy import (
        ProgramPolicyError,
        ProgramSelectingController,
    )

    inner = _build_full_brain_controller(args)
    try:
        controller: ProgramSelectingController
        if args.controller_profile == FULL_MCNS_INTERACTIONS_PROGRAM_V2_PROFILE:
            from neurocraft_controller.interaction_program_controller import (
                InteractionProgramSelectingController,
            )

            controller = InteractionProgramSelectingController(inner)
            print(
                "candidate interactions v2: supplied Minecraft actor classes and declared "
                "neural change gates select scripted programs; attack is a tactile proxy; "
                "loom-v2 calibration is a reused decoder component",
                flush=True,
            )
        elif args.controller_profile == FULL_MCNS_EVENTS_PROGRAM_V1_PROFILE:
            from neurocraft_controller.event_program_policy import EventProgramSelectingController

            controller = EventProgramSelectingController(inner)
            print(
                "candidate event profile: loom-v2 calibration is a reused decoder component; "
                "added events and hand-chosen thresholds have no biological validation",
                flush=True,
            )
        else:
            controller = ProgramSelectingController(inner)
    except ProgramPolicyError as exc:
        raise SystemExit(str(exc)) from exc
    # Printed unconditionally, before the socket opens, so the policy identity is in
    # every session log whether or not --program-decisions-out was supplied, and so the
    # launcher can quote it in its banner.
    print(f"body-program lane: policy_hash={controller.policy_hash}", flush=True)
    print(
        "The connectome's labeled readouts select and modulate scripted body programs "
        "through a declared hand-chosen mapping. The mapping is engineering. The "
        "programs are scripted. The connectome is not deciding anything.",
        flush=True,
    )
    return controller


def build_controller(
    args: argparse.Namespace,
) -> ConnectomeController | FullBrainController | ProgramSelectingController:
    _refuse_full_profile_conflicts(args)
    if args.controller_profile in FULL_MCNS_PROGRAM_PROFILES:
        return _build_program_selecting_controller(args)
    if args.controller_profile in FULL_MCNS_PROFILES:
        return _build_full_brain_controller(args)
    controller: ConnectomeController
    if args.controller_profile == WING_MOTOR_POOL_PROFILE:
        graph = load_malecns_fixture(
            args.fixture,
            expected_pin=MALECNS_FLIGHT_FIXTURE_V1_PIN,
            expected_file_bytes=MALECNS_FLIGHT_FIXTURE_V1_FILE_BYTES,
        )
        controller = FlightConnectomeController(
            graph,
            run_id=args.run_id,
            entity_id=args.entity_id,
            session_epoch=args.session_epoch,
            configuration=FlightControllerConfiguration(mode=args.mode),
        )
    elif args.controller_profile == TRM_CLONED_READOUT_PROFILE:
        graph = load_malecns_fixture(
            args.fixture,
            expected_pin=MALECNS_FLIGHT_FIXTURE_V1_PIN,
            expected_file_bytes=MALECNS_FLIGHT_FIXTURE_V1_FILE_BYTES,
        )
        artifact_bytes, artifact_document, artifact_sha256 = _read_readout_artifact(
            TRM_READOUT_ARTIFACT_PATH
        )
        controller = TrainedReadoutController(
            graph,
            run_id=args.run_id,
            entity_id=args.entity_id,
            session_epoch=args.session_epoch,
            artifact_bytes=artifact_bytes,
            artifact_document=artifact_document,
            configuration=TrmDemoConfiguration(
                mode=args.mode,
                readout_artifact_sha256=artifact_sha256,
            ),
        )
    else:
        graph = load_malecns_fixture(args.fixture)
        controller = ConnectomeController.create(
            graph,
            run_id=args.run_id,
            entity_id=args.entity_id,
            session_epoch=args.session_epoch,
            configuration=ControllerConfiguration(mode=args.mode),
        )
    if args.lesion_body_id:
        controller.set_named_lesion("cli-lesion", tuple(args.lesion_body_id))
    stimulation = _parse_stimulation_assignments(args.stimulate, option="--stimulate")
    if stimulation:
        controller.set_stimulation("cli-stimulation", stimulation)
    return controller


def _parse_stimulation_assignments(
    values: Sequence[str],
    *,
    option: str,
) -> dict[int, float]:
    stimulation: dict[int, float] = {}
    for item in values:
        try:
            body_text, value_text = item.split("=", 1)
            body_id = int(body_text)
            value = float(value_text)
        except ValueError as exc:
            raise SystemExit(f"{option} must use BODY_ID=VALUE assignments") from exc
        if body_id in stimulation:
            raise SystemExit(f"duplicate {option} body ID: {body_id}")
        stimulation[body_id] = value
    return stimulation


def _split_declaration(value: str, option: str) -> tuple[str, str]:
    try:
        content_id, payload = value.split("=", 1)
    except ValueError as exc:
        raise SystemExit(f"{option} must be ID=PAYLOAD") from exc
    if not content_id or not payload:
        raise SystemExit(f"{option} must be ID=PAYLOAD")
    return content_id, payload


def build_control_handler(
    args: argparse.Namespace,
    controller: ConnectomeController,
) -> ConnectomeControlHandler | None:
    interventions: list[NamedIntervention] = []
    if args.lesion_body_id:
        interventions.append(
            NamedLesion.create(controller, "cli-lesion", tuple(args.lesion_body_id))
        )
    stimulation = _parse_stimulation_assignments(args.stimulate, option="--stimulate")
    if stimulation:
        interventions.append(NamedStimulation.create(controller, "cli-stimulation", stimulation))
    for declaration in args.declare_lesion:
        content_id, payload = _split_declaration(declaration, "--declare-lesion")
        try:
            body_ids = tuple(int(item) for item in payload.split(","))
        except ValueError as exc:
            raise SystemExit("--declare-lesion body IDs must be comma-separated integers") from exc
        interventions.append(NamedLesion.create(controller, content_id, body_ids))
    for declaration in args.declare_stimulation:
        content_id, payload = _split_declaration(declaration, "--declare-stimulation")
        assignments = _parse_stimulation_assignments(
            payload.split(","), option="--declare-stimulation"
        )
        interventions.append(NamedStimulation.create(controller, content_id, assignments))
    return None if not interventions else ConnectomeControlHandler(controller, interventions)


def control_catalog(handler: ConnectomeControlHandler | None) -> dict[str, Any]:
    hashes: Mapping[str, str] = {} if handler is None else handler.content_hashes
    return {
        "schema": "neurocraft.control_catalog.v1",
        "interventions": [
            {"content_id": content_id, "content_hash": content_hash}
            for content_id, content_hash in sorted(hashes.items())
        ],
    }


def build_trace_recorder(
    args: argparse.Namespace,
    controller: ConnectomeController,
) -> MinecraftReceiptTraceRecorder | None:
    configured = (args.trace_out, args.trace_manifest, args.transition_forces)
    if any(value is not None for value in configured) and any(
        value is None for value in configured
    ):
        raise SystemExit(
            "--trace-out, --trace-manifest, and --transition-forces must be supplied together"
        )
    if args.trace_out is None:
        return None
    raw = parse_json_frame(args.trace_manifest.read_bytes())
    manifest = TraceManifest.from_dict(raw)
    if manifest.mode is not controller.run_mode:
        raise SystemExit("trace manifest mode does not match controller run mode")
    force_document = parse_json_frame(args.transition_forces.read_bytes())
    expected_keys = {
        "schema",
        "passive_forces",
        "external_forces",
        "provenance",
        "arena_material_hash",
    }
    if set(force_document) != expected_keys:
        raise SystemExit("transition force document has missing or unknown fields")
    if force_document["schema"] != "neurocraft.declared_transition_forces.v1":
        raise SystemExit("unsupported transition force document schema")
    passive = force_document["passive_forces"]
    external = force_document["external_forces"]
    if not isinstance(passive, list) or not isinstance(external, list):
        raise SystemExit("transition forces must be JSON arrays")
    provenance = force_document["provenance"]
    arena_material_hash = force_document["arena_material_hash"]
    if not isinstance(provenance, str) or not isinstance(arena_material_hash, str):
        raise SystemExit("transition force provenance/hash must be strings")
    forces = DeclaredTransitionForces.create(
        passive_forces=tuple(passive),
        external_forces=tuple(external),
        provenance=provenance,
        arena_material_hash=arena_material_hash,
    )
    writer = TraceWriter(args.trace_out, manifest)
    try:
        return MinecraftReceiptTraceRecorder(controller, writer, forces=forces)
    except Exception:
        writer.abort()
        raise


def write_identity(
    path: Path,
    controller: ConnectomeController | FullBrainController | ProgramSelectingController,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(controller.identity.to_dict()) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def write_canonical_document(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel-threads",
        type=int,
        choices=(1, 2, 4, 8),
        help="full-network execution setting; default uses the existing CPU-derived cap",
    )
    parser.add_argument(
        "--comparison-lanes",
        action="store_true",
        help=(
            "preload original/shuffled_weight/no_input for authenticated live epoch changes; "
            "requires a supported loom-v2 or event program profile and --mode faithful. Existing "
            "diagnostic output flags enable canonical files beneath --epoch-diagnostics-dir "
            "instead of writing their supplied single-file paths"
        ),
    )
    parser.add_argument("--lane-catalog-out", type=Path)
    parser.add_argument("--epoch-diagnostics-dir", type=Path)
    parser.add_argument("--control-seed", type=int, default=7)
    parser.add_argument(
        "--brain-display-out",
        type=Path,
        help="optional atomic model-activity snapshot for the local HUD; presentation only",
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--run-id", default="local-fixture-demo")
    parser.add_argument("--entity-id", default="neurocraft-fly")
    parser.add_argument("--session-epoch", type=int, default=0)
    parser.add_argument("--mode", choices=("faithful", "lesion"), default="faithful")
    parser.add_argument(
        "--controller-profile",
        choices=CONTROLLER_PROFILES,
        default=PAIR_CONTROLLER_PROFILE,
        help=(
            "pair_v1 keeps the shipped v1 readout; wing_motor_soma_side_v1 opts in to "
            "the flight controller over the pinned 128-node steering fixture; "
            "trm_cloned_readout_v1 opts in to the trained-readout demo controller "
            "bound to the committed cloned-readout artifact; full_mcns_wing_v1 opts in "
            "to the local-only full MaleCNS artifact (166,700 neurons / 25.6M signed "
            "edges) behind the same wing-motor boundary; full_mcns_eem_v1 opts in to "
            "the Engineered-Encoder Mode profile over the same artifact (declared "
            "encoder driving the labeled LC4/visual/bristle populations); "
            "full_mcns_eem_v2 opts in to the temporal-contrast revision of that "
            "encoder (drives are deviations from a declared leaky running mean; "
            "touch stays absolute); full_mcns_escape_v1 opts in to the "
            "escape-reflex profile (the eem v1 sensory boundary plus the declared "
            "TTMn jump readout adding one term to vertical thrust); and "
            "full_mcns_escape_v2 opts in to its transient-touch revision (touch "
            "alone moves under the declared temporal contrast, and the jump "
            "mapping is rectified to the non-negative half of its codomain); "
            "and full_mcns_loom_v1 opts in to the approach-gated revision of "
            "that (the three threat channels drive their LC4 populations with "
            "the rectified RISE of proximity above a declared leaky running "
            "mean, so an approaching object fires and a standing wall does not); "
            "and full_mcns_loom_program_v1 runs that same controller behind a "
            "declared hand-chosen mapping from its labeled readouts (TTMn jump, "
            "DNa01 L-R, wing common) to scripted body programs, emitting "
            "program_action frames instead of raw motor actions -- the mapping is "
            "engineering, the programs are scripted, and the connectome is not "
            "deciding anything. That frame kind is only admitted by a mod build "
            "carrying the program_action decode arm; against an older build the "
            "authority fails closed on every frame"
        ),
    )
    parser.add_argument(
        "--full-artifact-dir",
        type=Path,
        default=DEFAULT_FULL_ARTIFACT_DIR,
        help=(
            "directory of the local full MaleCNS runtime artifact "
            "(full_mcns_wing_v1/full_mcns_eem_v1/full_mcns_eem_v2/"
            "full_mcns_escape_v1/full_mcns_escape_v2/full_mcns_loom_v1 only; "
            "gitignored, manifest pinned in "
            "neurocraft_controller.full_brain_controller)"
        ),
    )
    parser.add_argument(
        "--full-uncalibrated",
        action="store_true",
        help=(
            "run a full MaleCNS profile with the uncalibrated readout (scale 1.0) "
            "instead of its committed declared calibration; for explicit A/B "
            "comparison, and it rotates the controller identity accordingly"
        ),
    )
    parser.add_argument("--lesion-body-id", type=int, action="append", default=[])
    parser.add_argument("--stimulate", action="append", default=[])
    parser.add_argument(
        "--declare-lesion",
        action="append",
        default=[],
        metavar="ID=BODY_ID[,BODY_ID]",
        help="predeclare a hash-bound lesion without applying it",
    )
    parser.add_argument(
        "--declare-stimulation",
        action="append",
        default=[],
        metavar="ID=BODY_ID=VALUE[,BODY_ID=VALUE]",
        help="predeclare a hash-bound stimulation without applying it",
    )
    parser.add_argument("--host", default=os.environ.get("NEUROCRAFT_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=int,
        default=os.environ.get("NEUROCRAFT_PORT", "8765"),
    )
    parser.add_argument("--identity-out", type=Path)
    parser.add_argument(
        "--presentation-telemetry-out",
        type=Path,
        help="append presentation-only NDJSON samples for the local brain viewer",
    )
    parser.add_argument(
        "--action-latency-out",
        type=Path,
        help=(
            "append latency-only NDJSON samples (command_seq plus the measured "
            "compute/serialization/write nanoseconds); no neural value is written, "
            "so unlike --presentation-telemetry-out every profile accepts it"
        ),
    )
    parser.add_argument(
        "--program-decisions-out",
        type=Path,
        help=(
            "append DIAGNOSTIC NDJSON lines for the body-program lane: one per issued "
            "program_action (program, modulation, parameters, the three labeled readout "
            "scalars, policy_hash) and one per program_receipt. Diagnostic only -- not "
            "content-addressed, not evidence, and not a trace: it records what the "
            "declared hand-chosen mapping asked for and what the authority answered, and "
            "nothing about the connectome deciding anything"
        ),
    )
    parser.add_argument("--control-catalog-out", type=Path)
    parser.add_argument("--trace-out", type=Path)
    parser.add_argument("--trace-manifest", type=Path)
    parser.add_argument(
        "--transition-forces",
        type=Path,
        help="strict declared force document bound by the trace manifest arena hash",
    )
    parser.add_argument(
        "--identity-only",
        action="store_true",
        help="print the canonical RunIdentity without opening a socket",
    )
    return parser


def build_brain_display(
    args: argparse.Namespace,
    controller: ConnectomeController | FullBrainController | ProgramSelectingController,
) -> tuple[Callable[[Command], None] | None, Callable[[], None]]:
    """Optional display startup is isolated from the authoritative command service."""
    if args.brain_display_out is None:
        return None, lambda: None
    publisher = None
    try:
        from neurocraft_connectome.full_runtime import load_full_runtime_artifact
        from neurocraft_controller.brain_display import (
            AtomicBrainDisplayPublisher,
            BrainDisplayProducer,
            load_brain_display_binding,
        )
        from neurocraft_controller.program_policy import ProgramSelectingController

        if isinstance(controller, ConnectomeController):
            raise ValueError("the named-population HUD requires the full escape/program profile")
        core = (
            controller.controller
            if isinstance(controller, ProgramSelectingController)
            else controller
        )
        if core.embodied_senses is None or core.escape_binding is None:
            raise ValueError("the controller has no LC4/TTMn binding")
        with load_full_runtime_artifact(
            args.full_artifact_dir, expected_manifest_sha256=core.identity.graph_hash
        ) as graph:
            binding = load_brain_display_binding(
                graph,
                FULL_BODY_ANNOTATIONS_PATH,
                core.embodied_senses.populations,
                core.escape_binding.ttmn_pool_identity_sha256,
            )
        publisher = AtomicBrainDisplayPublisher(args.brain_display_out)
        producer = BrainDisplayProducer(binding, publisher)

        def observe(command: Command) -> None:
            producer.capture_completed_step(controller, command)

        def close() -> None:
            publisher.close(timeout=1.0)
            if producer.last_error or publisher.last_error:
                print(
                    "brain display ended unavailable; optional display errors recorded", flush=True
                )

        print(
            f"brain display: binding={binding.identity_sha256}; "
            "raw model activity, presentation only",
            flush=True,
        )
        return observe, close
    except Exception as exc:
        if publisher is not None:
            publisher.close(timeout=1.0)
        print(
            f"brain display unavailable ({type(exc).__name__}); command service continues",
            flush=True,
        )
        return None, lambda: None


def _with_event_display(
    controller: ConnectomeController | FullBrainController | ProgramSelectingController,
    path: Path | None,
    observer: Callable[[Command], None] | None,
    close: Callable[[], None] | None,
) -> tuple[Callable[[Command], None] | None, Callable[[], None]]:
    """Compose issued event diagnostics under the existing serialized step fence."""
    close_primary = close or (lambda: None)
    if path is None:
        return observer, close_primary
    try:
        from neurocraft_controller.brain_display import AtomicBrainDisplayPublisher
        from neurocraft_controller.event_program_policy import EventProgramSelectingController
        from neurocraft_controller.interaction_program_controller import (
            InteractionProgramSelectingController,
        )

        if not isinstance(
            controller, (EventProgramSelectingController, InteractionProgramSelectingController)
        ):
            return observer, close_primary
        publisher = AtomicBrainDisplayPublisher(path)
    except Exception as exc:
        print(f"event display unavailable ({type(exc).__name__}); commands continue", flush=True)
        return observer, close_primary

    def capture(command: Command) -> None:
        # Each channel is optional and independently isolated. A failed legacy
        # population capture must not suppress a valid event snapshot, or vice versa.
        if observer is not None:
            with contextlib.suppress(Exception):
                observer(command)
        with contextlib.suppress(Exception):
            evidence = controller.last_event_evidence
            if evidence is not None and evidence.command == command:
                publisher.offer(evidence)

    def finish() -> None:
        try:
            close_primary()
        finally:
            if not publisher.close(timeout=1.0):
                print("event display writer did not close cleanly", flush=True)

    return capture, finish


def _refuse_comparison_conflicts(args: argparse.Namespace) -> None:
    """Refuse incompatible comparison options before loading any full artifact."""
    if not args.comparison_lanes:
        if (
            args.lane_catalog_out is not None
            or args.epoch_diagnostics_dir is not None
            or args.control_seed != 7
        ):
            raise SystemExit("lane catalog, epoch diagnostics and control seed require comparison")
        return
    if args.controller_profile not in FULL_MCNS_COMPARISON_PROFILES or args.mode != "faithful":
        raise SystemExit(
            "--comparison-lanes requires full_mcns_loom_program_v2 or "
            "full_mcns_events_program_v1 or full_mcns_interactions_program_v2, and --mode faithful"
        )
    if args.lane_catalog_out is None or args.epoch_diagnostics_dir is None:
        raise SystemExit(
            "--comparison-lanes requires --lane-catalog-out and --epoch-diagnostics-dir"
        )
    if args.identity_only:
        raise SystemExit("--comparison-lanes cannot be combined with --identity-only")
    if args.full_uncalibrated:
        raise SystemExit("--comparison-lanes requires the declared calibrated readout")
    if args.fixture != DEFAULT_FIXTURE:
        raise SystemExit("--comparison-lanes reads --full-artifact-dir; --fixture is incompatible")
    incompatible = {
        "trace-out": args.trace_out,
        "trace-manifest": args.trace_manifest,
        "transition-forces": args.transition_forces,
        "presentation-telemetry-out": args.presentation_telemetry_out,
        "control-catalog-out": args.control_catalog_out,
        "lesion-body-id": args.lesion_body_id,
        "stimulate": args.stimulate,
        "declare-lesion": args.declare_lesion,
        "declare-stimulation": args.declare_stimulation,
    }
    for name, value in incompatible.items():
        if value:
            raise SystemExit(f"--comparison-lanes does not support --{name}")
    if not 0 <= args.control_seed < (1 << 64):
        raise SystemExit("--control-seed must be a uint64 integer")
    if not 0 <= args.session_epoch < (1 << 63):
        raise SystemExit("--session-epoch must be a nonnegative signed-64-bit integer")
    directory = args.epoch_diagnostics_dir.resolve()
    outputs = [args.lane_catalog_out, args.identity_out]
    resolved = [path.resolve() for path in outputs if path is not None]
    if len(resolved) != len(set(resolved)) or any(
        path.is_relative_to(directory) for path in resolved
    ):
        raise SystemExit("catalog/identity outputs must be distinct and outside epoch diagnostics")
    for path in outputs:
        if path is not None and path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {path}")
    if (directory / str(args.session_epoch)).exists():
        raise SystemExit("refusing to reuse an existing initial epoch diagnostics directory")


def _lane_metadata(lane: LaneSpec) -> dict[str, Any]:
    controller = cast("ProgramSelectingController", lane.controller)
    control = controller.controller.control_identity
    return {
        "name": lane.name,
        "identity": controller.identity.to_dict(),
        "control_kind": controller.controller.control_kind,
        "control_sha256": None if control is None else control.identity_sha256,
        "seed": None if control is None else str(control.seed),
    }


def _comparison_display_binding(
    args: argparse.Namespace, original: ProgramSelectingController
) -> BrainDisplayBinding | None:
    """Load the shared verified binding once, outside any live Hello transaction."""
    if args.brain_display_out is None:
        return None
    try:
        from neurocraft_connectome.full_runtime import load_full_runtime_artifact
        from neurocraft_controller.brain_display import load_brain_display_binding

        core = original.controller
        if core.embodied_senses is None or core.escape_binding is None:
            raise ValueError("comparison display requires the LC4/TTMn bindings")
        with load_full_runtime_artifact(
            args.full_artifact_dir, expected_manifest_sha256=core.identity.graph_hash
        ) as graph:
            return load_brain_display_binding(
                graph,
                FULL_BODY_ANNOTATIONS_PATH,
                core.embodied_senses.populations,
                core.escape_binding.ttmn_pool_identity_sha256,
            )
    except Exception as exc:
        print(
            f"comparison display unavailable ({type(exc).__name__}); commands continue",
            flush=True,
        )
        return None


def _comparison_hooks_factory(
    args: argparse.Namespace, binding: BrainDisplayBinding | None
) -> Callable[[LaneSpec, RunIdentity], SessionHooks]:
    def create(lane: LaneSpec, identity: RunIdentity) -> SessionHooks:
        started_ns = time.monotonic_ns()

        def diagnostic(stage: str) -> None:
            with contextlib.suppress(Exception):
                logging.getLogger("neurocraft_service.lanes").warning(
                    "lane_hooks epoch=%d stage=%s hooks_elapsed_ms=%.3f",
                    identity.session_epoch,
                    stage,
                    (time.monotonic_ns() - started_ns) / 1_000_000,
                )

        controller = cast("ProgramSelectingController", lane.controller)
        if controller.identity != identity:
            raise ValueError("epoch diagnostics require the controller's current full identity")
        directory = args.epoch_diagnostics_dir / str(identity.session_epoch)
        diagnostic("sidecar_create_begin")
        directory.mkdir()  # A consumed epoch's diagnostics must never be truncated or reused.
        write_canonical_document(
            directory / "epoch.json",
            {
                "schema": "neurocraft.lane_epoch.v1",
                "diagnostic_only": True,
                "controller_profile": args.controller_profile,
                **_lane_metadata(lane),
            },
        )
        diagnostic("sidecar_create_complete")
        decisions = (
            None
            if args.program_decisions_out is None
            else ProgramDecisionWriter(
                directory / "program-decisions.ndjson",
                controller,
                controller_profile=args.controller_profile,
            )
        )
        latency = (
            None
            if args.action_latency_out is None
            else ActionLatencyWriter(
                directory / "action-latency.ndjson",
                run_id=identity.run_id,
                controller_profile=args.controller_profile,
            )
        )
        observer: Callable[[Command], None] | None = None
        close_display: Callable[[], None] | None = None
        if binding is not None:
            from neurocraft_controller.brain_display import (
                AtomicBrainDisplayPublisher,
                BrainDisplayProducer,
            )

            publisher = None
            try:
                publisher = AtomicBrainDisplayPublisher(directory / "brain-display.json")
                producer = BrainDisplayProducer(binding, publisher)

                def observe(command: Command) -> None:
                    producer.capture_completed_step(controller, command)

                def close() -> None:
                    if not publisher.close(timeout=1.0):
                        print("epoch display writer did not close cleanly", flush=True)

                observer, close_display = observe, close
            except Exception as exc:
                if publisher is not None:
                    with contextlib.suppress(Exception):
                        publisher.close(timeout=1.0)
                print(
                    f"epoch display unavailable ({type(exc).__name__}); commands continue",
                    flush=True,
                )
        observer, close_display = _with_event_display(
            controller,
            directory / "event-selection-display.json" if args.brain_display_out else None,
            observer,
            close_display,
        )
        return SessionHooks(
            receipt_sink=None if decisions is None else decisions.accept_receipt,
            issued_action_sink=None if decisions is None else decisions.accept_issued_command,
            action_telemetry_sink=None if latency is None else latency.accept,
            completed_step_observer=observer,
            close=close_display,
        )

    return create


def _initial_comparison_warm_allowed(
    dispatcher: LaneSessionDispatcher, initial_identity: RunIdentity
) -> bool:
    claimed = dispatcher.last_claimed_epoch
    return claimed < initial_identity.session_epoch or (
        claimed == initial_identity.session_epoch
        and dispatcher.active_lane == "original"
        and dispatcher.active_identity == initial_identity
    )


async def _serve_comparison(dispatcher: LaneSessionDispatcher) -> None:
    try:
        async with create_lane_server(dispatcher) as server:
            print(
                f"neural comparison service listening on loopback port "
                f"{server.sockets[0].getsockname()[1]}; fresh authenticated epochs required",
                flush=True,
            )
            await server.serve_forever()
    finally:
        await dispatcher.close()


def _run_comparison(args: argparse.Namespace) -> int:
    from neurocraft_controller.program_policy import ProgramSelectingController

    token = os.environ.get("NEUROCRAFT_RUN_TOKEN")
    if token is None:
        raise SystemExit("NEUROCRAFT_RUN_TOKEN must be supplied out of band")
    lanes: list[LaneSpec] = []
    names: tuple[LaneName, ...] = ("original", "shuffled_weight", "no_input")
    for name in names:
        lane_args = argparse.Namespace(**vars(args))
        lane_args.mode = "faithful" if name == "original" else name
        controller = build_controller(lane_args)
        if not isinstance(controller, ProgramSelectingController):
            raise SystemExit("comparison requires actual full-brain program controllers")
        lanes.append(LaneSpec(name, controller))
    original = cast("ProgramSelectingController", lanes[0].controller)
    binding = _comparison_display_binding(args, original)
    args.epoch_diagnostics_dir.mkdir(parents=True, exist_ok=True)
    dispatcher = LaneSessionDispatcher(
        ServiceConfig(identity=original.identity, token=token, host=args.host, port=args.port),
        lanes,
        hooks_factory=_comparison_hooks_factory(args, binding),
    )
    write_canonical_document(
        args.lane_catalog_out,
        {
            "schema": "neurocraft.lane_catalog.v1",
            "initial_identity": original.identity.to_dict(),
            "lanes": [_lane_metadata(lane) for lane in lanes],
        },
    )
    if args.identity_out is not None:
        write_identity(args.identity_out, original)
    print(
        "comparison diagnostics are not causal traces; original wiring uses engineered senses "
        f"and a hand-chosen mapping to scripted body programs; policy_hash={original.policy_hash}",
        flush=True,
    )
    try:
        initial_identity = original.identity
        asyncio.run(
            _serve_with_idle_rewarm(
                _serve_comparison(dispatcher),
                original,
                continue_while=lambda: _initial_comparison_warm_allowed(
                    dispatcher, initial_identity
                ),
            )
        )
    except KeyboardInterrupt:
        return 130
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _refuse_comparison_conflicts(args)
    if args.comparison_lanes:
        return _run_comparison(args)
    controller = build_controller(args)
    # The intervention and receipt-trace lanes are fixture-typed; the full
    # profile's argument conflicts were already refused before construction,
    # so both lanes are simply absent for it.
    control_handler = (
        build_control_handler(args, controller)
        if isinstance(controller, ConnectomeController)
        else None
    )
    if args.identity_out is not None:
        write_identity(args.identity_out, controller)
    if args.control_catalog_out is not None:
        write_canonical_document(args.control_catalog_out, control_catalog(control_handler))
    if args.identity_only:
        sys.stdout.buffer.write(canonical_json_bytes(controller.identity.to_dict()) + b"\n")
        return 0
    token = os.environ.get("NEUROCRAFT_RUN_TOKEN")
    if token is None:
        raise SystemExit("NEUROCRAFT_RUN_TOKEN must be supplied out of band")
    if not isinstance(controller, ConnectomeController):
        # Full profile: the mod grants the first live action a hard 2-tick
        # deadline, and a cold first step can spike past it and latch
        # DEADLINE_MISSED. Warm before the socket opens, then keep warm via
        # _serve_with_idle_rewarm until stepping begins; state untouched.
        warm_seconds = controller.warm_compute_path()
        print(f"full-brain compute path warmed in {warm_seconds * 1000.0:.1f} ms", flush=True)
    recorder = (
        build_trace_recorder(args, controller)
        if isinstance(controller, ConnectomeController)
        else None
    )
    action_telemetry_sinks: list[Callable[[ActionTelemetry], None]] = []
    if recorder is not None:
        action_telemetry_sinks.append(partial(record_action_telemetry, recorder))
    if args.presentation_telemetry_out is not None:
        # Unreachable: --presentation-telemetry-out is refused for every full
        # profile, and the body-program lane is one. Kept as a narrowing gate so
        # the writer can never be handed a controller whose evidence it cannot read.
        if not isinstance(controller, ConnectomeController | _full_brain_controller_type()):
            raise SystemExit(
                "--presentation-telemetry-out is not available for the body-program lane: "
                "its per-step evidence is a program selection, not the boundary activity "
                "subset the brain viewer renders"
            )
        presentation_writer = PresentationTelemetryWriter(
            args.presentation_telemetry_out, controller, args.controller_profile
        )
        action_telemetry_sinks.append(presentation_writer.accept)
    if args.action_latency_out is not None:
        latency_writer = ActionLatencyWriter(
            args.action_latency_out,
            run_id=controller.identity.run_id,
            controller_profile=args.controller_profile,
        )
        action_telemetry_sinks.append(latency_writer.accept)
    program_decisions: ProgramDecisionWriter | None = None
    if args.program_decisions_out is not None:
        # _refuse_full_profile_conflicts already refused this flag for every
        # profile that does not issue program frames.
        from neurocraft_controller.program_policy import ProgramSelectingController

        assert isinstance(controller, ProgramSelectingController)
        program_decisions = ProgramDecisionWriter(
            args.program_decisions_out,
            controller,
            controller_profile=args.controller_profile,
        )
        print(
            "program decisions (DIAGNOSTIC ONLY, not evidence) -> "
            f"{args.program_decisions_out}; policy_hash={program_decisions.policy_hash}",
            flush=True,
        )

    def fan_out_action_telemetry(telemetry: ActionTelemetry) -> None:
        for sink in action_telemetry_sinks:
            sink(telemetry)

    runtime_control_handler = (
        None
        if control_handler is None and recorder is None
        else ConnectomeRuntimeControlHandler(
            interventions=control_handler,
            recorder=recorder,
        )
    )
    # The fixture trace recorder and the diagnostic program-decision writer are
    # mutually exclusive by profile: a recorder is built only for the fixture
    # lane, and --program-decisions-out is refused for every non-program profile.
    receipt_sink: Callable[[ActuationReceipt | ProgramReceipt], None] | None = None
    issued_action_sink: Callable[[Command], None] | None = None
    if recorder is not None:
        receipt_sink = partial(record_receipt, recorder)
        issued_action_sink = partial(record_issued_command, recorder)
    elif program_decisions is not None:
        receipt_sink = program_decisions.accept_receipt
        issued_action_sink = program_decisions.accept_issued_command
    display_observer, close_display = build_brain_display(args, controller)
    display_observer, close_display = _with_event_display(
        controller,
        None
        if args.brain_display_out is None
        else (args.brain_display_out.with_name("event-selection-display.json")),
        display_observer,
        close_display,
    )
    serve_coroutine = serve(
        ServiceConfig(
            identity=controller.identity,
            token=token,
            host=args.host,
            port=args.port,
        ),
        controller,
        receipt_sink=receipt_sink,
        issued_action_sink=issued_action_sink,
        action_telemetry_sink=(fan_out_action_telemetry if action_telemetry_sinks else None),
        control_handler=runtime_control_handler,
        completed_step_observer=display_observer,
    )
    try:
        if isinstance(controller, ConnectomeController):
            asyncio.run(serve_coroutine)
        else:
            asyncio.run(_serve_with_idle_rewarm(serve_coroutine, controller))
    except KeyboardInterrupt:
        if recorder is not None:
            recorder.abort()
        return 130
    except BaseException:
        if recorder is not None:
            recorder.abort()
        raise
    finally:
        close_display()
    if recorder is not None:
        try:
            recorder.close()
        except ControllerIntegrityError:
            recorder.abort()
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
