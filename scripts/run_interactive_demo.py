#!/usr/bin/env python3
"""Launch the fixture-controlled fly in an isolated interactive Minecraft client."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import queue
import re
import secrets
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from types import FrameType, ModuleType
from typing import BinaryIO, TextIO, cast

ROOT = Path(__file__).resolve().parents[1]
MOD_ROOT = ROOT / "mod"
RUN_ROOT = ROOT / "artifacts" / "live-interactive"
ACCEPTED_EULA = MOD_ROOT / "run" / "eula.txt"
FIXTURE = ROOT / "data" / "fixtures" / "mcns-v1.0-flight-power-v1.json"
STEERING_FIXTURE = ROOT / "data" / "fixtures" / "mcns-v1.0-flight-steering-v1.json"
FULL_ARTIFACT_DIR = ROOT / "data" / "processed" / "male-cns-v1.0-superclass-v1"
FULL_ANNOTATIONS_PATH = (
    ROOT / "data" / "cache" / "male-cns-v1.0" / "body-neurotransmitters-male-cns-v1.0.feather"
)
FULL_BODY_ANNOTATIONS_PATH = (
    ROOT / "data" / "cache" / "male-cns-v1.0" / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)
FULL_MCNS_WING_PROFILE = "full_mcns_wing_v1"
FULL_MCNS_EEM_PROFILE = "full_mcns_eem_v1"
FULL_MCNS_EEM_V2_PROFILE = "full_mcns_eem_v2"
FULL_MCNS_ESCAPE_PROFILE = "full_mcns_escape_v1"
FULL_MCNS_ESCAPE_V2_PROFILE = "full_mcns_escape_v2"
FULL_MCNS_LOOM_V1_PROFILE = "full_mcns_loom_v1"
FULL_MCNS_LOOM_PROGRAM_V1_PROFILE = "full_mcns_loom_program_v1"
FULL_MCNS_LOOM_V2_PROFILE = "full_mcns_loom_v2"
FULL_MCNS_LOOM_PROGRAM_V2_PROFILE = "full_mcns_loom_program_v2"
FULL_MCNS_EVENTS_PROGRAM_V1_PROFILE = "full_mcns_events_program_v1"
FULL_MCNS_INTERACTIONS_PROGRAM_V2_PROFILE = "full_mcns_interactions_program_v2"
FULL_MCNS_EEM_PROFILES = (FULL_MCNS_EEM_PROFILE, FULL_MCNS_EEM_V2_PROFILE)
FULL_MCNS_ESCAPE_PROFILES = (FULL_MCNS_ESCAPE_PROFILE, FULL_MCNS_ESCAPE_V2_PROFILE)
#: The loom-v1 encoder's two lanes: raw motor actions, and body-program frames.
FULL_MCNS_LOOM_V1_PROFILES = (
    FULL_MCNS_LOOM_V1_PROFILE,
    FULL_MCNS_LOOM_PROGRAM_V1_PROFILE,
)
#: Every profile that issues ``program_action`` frames rather than raw actions.
FULL_MCNS_LOOM_V2_PROFILES = (FULL_MCNS_LOOM_V2_PROFILE, FULL_MCNS_LOOM_PROGRAM_V2_PROFILE)
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
#: Every profile that carries the TTMn jump readout: the two escape
#: generations plus the loom-v1 lanes, which reuse the escape-v2 decoder unchanged.
FULL_MCNS_JUMP_READOUT_PROFILES = (
    *FULL_MCNS_ESCAPE_PROFILES,
    *FULL_MCNS_LOOM_PROFILES,
    FULL_MCNS_EVENTS_PROGRAM_V1_PROFILE,
    FULL_MCNS_INTERACTIONS_PROGRAM_V2_PROFILE,
)
#: Every profile that needs the body-annotations table (population selection
#: for the eem profiles; population plus TTMn selection for the jump-readout
#: ones).
FULL_MCNS_BODY_ANNOTATION_PROFILES = (
    *FULL_MCNS_EEM_PROFILES,
    *FULL_MCNS_JUMP_READOUT_PROFILES,
)
FULL_MCNS_PROFILES = (FULL_MCNS_WING_PROFILE, *FULL_MCNS_BODY_ANNOTATION_PROFILES)

#: Profile names must match run_connectome_service.py exactly; the service is
#: the authority and refuses unknown profiles. Each row pins the fixture the
#: profile is defined over and the claim-boundary description for session.json.
#: A ``None`` fixture means the profile reads no fixture at all -- the service
#: resolves its own local-only artifact -- so ``--fixture`` is omitted from the
#: service command entirely.
CONTROLLER_PROFILES: dict[str, tuple[Path | None, str]] = {
    FULL_MCNS_INTERACTIONS_PROGRAM_V2_PROFILE: (
        None,
        "166,700-neuron full MaleCNS with separate brush/attack inputs, explicit Minecraft "
        "actor metadata and light-change readout gates; hand-chosen readouts select "
        "scripted body programs (candidate interactions v2, reused decoder component, local-only)",
    ),
    FULL_MCNS_EVENTS_PROGRAM_V1_PROFILE: (
        None,
        "166,700-neuron full MaleCNS with engineered entity approach and food/brush/attack "
        "event inputs; hand-chosen readouts select scripted body programs "
        "(candidate events v1, reused loom-v2 decoder component, local-only)",
    ),
    FULL_MCNS_LOOM_V2_PROFILE: (
        None,
        "166,700-neuron full MaleCNS with declared entity-gated approach senses "
        "and rectified escape readout (EEM loom v2, local-only)",
    ),
    FULL_MCNS_LOOM_PROGRAM_V2_PROFILE: (
        None,
        "166,700-neuron full MaleCNS with entity-gated approach senses; "
        "hand-chosen labeled readouts select and modulate scripted body programs "
        "(EEM loom v2, local-only)",
    ),
    "pair_v1": (FIXTURE, "14-node/22-edge MaleCNS-derived fixture"),
    "wing_motor_soma_side_v1": (
        STEERING_FIXTURE,
        "128-node/887-edge MaleCNS-derived steering fixture",
    ),
    "trm_cloned_readout_v1": (
        STEERING_FIXTURE,
        "128-node/887-edge MaleCNS-derived steering fixture with trained cloned readout",
    ),
    FULL_MCNS_WING_PROFILE: (
        None,
        "166,700-neuron / 25.6M-edge full MaleCNS artifact (local-only)",
    ),
    FULL_MCNS_EEM_PROFILE: (
        None,
        "166,700-neuron full MaleCNS with declared embodied senses (EEM, local-only)",
    ),
    FULL_MCNS_EEM_V2_PROFILE: (
        None,
        "166,700-neuron full MaleCNS with declared temporal-contrast embodied "
        "senses (EEM v2, local-only)",
    ),
    FULL_MCNS_ESCAPE_PROFILE: (
        None,
        "166,700-neuron full MaleCNS with declared embodied senses and "
        "escape-reflex jump readout (EEM escape, local-only)",
    ),
    FULL_MCNS_ESCAPE_V2_PROFILE: (
        None,
        "166,700-neuron full MaleCNS with declared transient-touch embodied "
        "senses and rectified escape-reflex jump readout (EEM escape v2, "
        "local-only)",
    ),
    FULL_MCNS_LOOM_V1_PROFILE: (
        None,
        "166,700-neuron full MaleCNS with declared approach-gated embodied "
        "senses and rectified escape-reflex jump readout (EEM loom v1, "
        "local-only)",
    ),
    FULL_MCNS_LOOM_PROGRAM_V1_PROFILE: (
        None,
        "166,700-neuron full MaleCNS on the EEM loom v1 boundary, with its "
        "labeled readouts selecting scripted body programs through a declared "
        "hand-chosen mapping (body-program lane, local-only; emits program_action "
        "frames, which the mod admits under READOUT authority — the launcher "
        "still refuses to start if the mod build it launches has no decode arm)",
    ),
}
DEFAULT_CONTROLLER_PROFILE = "pair_v1"
USERNAME = "NeuroFlyObserver"
TOKEN_NAME = "NEUROCRAFT_RUN_TOKEN"
IDENTITY_KEYS = {
    "run_id",
    "session_epoch",
    "entity_id",
    "graph_hash",
    "adapter_hash",
    "dynamics_hash",
    "decoder_hash",
    "observation_schema_hash",
}
HASH_KEYS = (
    "graph_hash",
    "adapter_hash",
    "dynamics_hash",
    "decoder_hash",
    "observation_schema_hash",
)

#: How often the interactive monitor loop asks the server for `neurocraftfly status`.
#: Fault codes only reach the server capture through status lines, which used to be
#: printed solely during startup polling -- a session that latched after startup left
#: the terminal-fault watcher blind while the operator interacted with a dead
#: (neutral) fly. At 0.5 Hz the watcher fires within ~2 s of a latch and the extra
#: status noise in the capture stays tolerable.
MONITOR_STATUS_INTERVAL_SECONDS = 2.0
# These affect only the fresh development client, never the user's normal install.
# The full neural service and two Minecraft processes share this host. Avoid the
# client's default 120 FPS / 12-chunk rendering workload in a six-chunk arena.
CLIENT_SETTINGS = {
    "onboardAccessibility": "false",
    "skipMultiplayerWarning": "true",
    "startedCleanly": "true",
    "tutorialStep": "none",
    "maxFps": "60",
    # Make the declared render ceiling independent of an inherited display-sync limit.
    # This applies only to the fresh diagnostic client; achieved cadence is measured.
    "enableVsync": "false",
    # Console/client diagnostics do not refresh Minecraft's native-input timer.
    # Skip AFK 30/10-FPS caps while this recording window is not minimized.
    "inactivityFpsLimit": '"minimized"',
    "renderDistance": "6",
    "simulationDistance": "6",
    "renderClouds": '"false"',
    "bobView": "false",
    "guiScale": "2",
}
CLIENT_SETTLE_SECONDS = 3.0

#: The service prints this line before it opens its socket for the body-program lane.
#: The launcher quotes the hash in its ready banner rather than recomputing it, so the
#: banner can never name a policy other than the one that is actually running.
POLICY_HASH_PATTERN = re.compile(r"\bpolicy_hash=([0-9a-f]{64})\b")

#: The honesty sentence the ready banner prints for the body-program lane
#: (stage-4b design section 7). Nothing here may be softened or split: the lane is a
#: declared hand-chosen mapping over labeled readouts selecting scripted programs, and
#: every surface that shows it says exactly that.
#: The mod decodes inbound frames by an exhaustive switch and fails closed on any type it
#: does not name (``ControlProtocol.decodeInbound``). Until the mod-side lane adds a
#: ``program_action`` arm, a body-program session would be refused frame by frame, so the
#: launcher checks for the arm rather than starting three processes that cannot talk.
#: Parsed from the Java source for the same reason ``TERMINAL_FAULT_PATTERN`` is: a
#: hand-maintained "is it landed yet" flag would drift silently.
#: Anchored at the start of a line (after indentation only) so a commented-out
#: ``// case "program_action" ->`` cannot be read as a landed decoder; block comments are
#: stripped before the search for the same reason.
PROGRAM_ACTION_DECODE_ARM = re.compile(r'^[ \t]*case\s+"program_action"\s*->', flags=re.MULTILINE)
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", flags=re.DOTALL)
CONTROL_PROTOCOL = "src/main/java/dev/neurocraft/fly/control/ControlProtocol.java"
CONTROL_PROTOCOL_SOURCE = MOD_ROOT / CONTROL_PROTOCOL

#: Stated on the profile, in the banner, and in the refusal: this lane emits a frame kind
#: the authority must first learn to admit.
PROGRAM_LANE_NOT_RUNNABLE = (
    "full_mcns_loom_program_v1 emits program_action frames, and this mod build has no "
    "program_action decode arm, so the authority would fail closed on every frame and the "
    "fly would never move. The mod-side lane (the frame, the validator lane latch, the "
    "bridge, and READOUT authority) has landed on main, so this normally means the tree "
    "being launched is older than that lane or its ControlProtocol.java was edited; as "
    "launched, this profile is NOT runnable live. The service half is testable on its own "
    "with scripts/probe_program_policy.py, which needs no Minecraft."
)

PROGRAM_LANE_BANNER = (
    "The connectome is running and its labeled readouts (TTMn jump, DNa01 L-R, wing "
    "common) select and modulate scripted body programs through a declared hand-chosen "
    "mapping (`policy_hash {policy_hash}`). The mapping is engineering. The programs are "
    "scripted. The connectome is not deciding anything."
)

#: Every latching ControllerFault name from
#: mod/src/main/java/dev/neurocraft/fly/control/ControllerFault.java, plus the
#: service-side terminal error codes RECEIPT_MISMATCH and TRACE_FAILURE as
#: defense-in-depth only: NeuralBridge.mapRemoteFault currently normalizes both
#: to INTERNAL_ERROR before any server console line is printed, so INTERNAL_ERROR
#: is what actually matches today; the raw codes are kept in case a future bridge
#: or log path surfaces them verbatim. The non-latching states STARTING and
#: ACTIVE are deliberately absent so healthy status lines never match.
TERMINAL_FAULT_PATTERN = re.compile(
    r"\b("
    r"MISSING_CONFIGURATION|DISCONNECTED|PROTOCOL_ERROR|IDENTITY_MISMATCH"
    r"|STALE_COMMAND|FUTURE_COMMAND|DEADLINE_MISSED|SEQUENCE_ERROR"
    r"|OVERSIZE_MESSAGE|QUEUE_OVERFLOW|INTERNAL_ERROR"
    r"|RECEIPT_MISMATCH|TRACE_FAILURE"
    r")\b"
)


class LauncherError(RuntimeError):
    """The interactive demo could not be launched safely."""


@dataclass(frozen=True)
class RunPlan:
    run_id: str
    entity_id: str
    session_epoch: int
    bridge_port: int
    game_port: int
    run_dir: Path
    server_dir: Path
    client_dir: Path
    identity_path: Path
    session_path: Path
    controller_profile: str = DEFAULT_CONTROLLER_PROFILE
    viewer_port: int | None = None
    action_latency: bool = False
    game_tick_rate: int = 20
    diagnostic_tools: bool = False
    comparison_lanes: bool = False
    control_seed: int = 7
    kernel_threads: int | None = None
    capture_fps: int = 1
    capture_seconds: int = 120
    capture_png_compression: int | None = None
    client_width: int | None = None
    client_height: int | None = None
    client_max_fps: int = 60
    showcase_world: bool = False
    circuit_playground: bool = False
    showcase_commands: tuple[str, ...] = ()
    installed_runtime: Path | None = None
    manual_client_name: str | None = None

    @property
    def requested_client_settings(self) -> dict[str, str]:
        """Fresh options for this run; window dimensions are logical, not framebuffer pixels."""

        settings = dict(CLIENT_SETTINGS)
        settings["maxFps"] = str(self.client_max_fps)
        if self.diagnostic_tools:
            # Losing focus must not open a menu over a recorded native-input probe.
            # Explicit menus still refuse native actions; only this fresh client is configured.
            settings["pauseOnLostFocus"] = "false"
        if self.client_width is not None and self.client_height is not None:
            settings["overrideWidth"] = str(self.client_width)
            settings["overrideHeight"] = str(self.client_height)
        return settings

    @property
    def lane_catalog_path(self) -> Path:
        return self.run_dir / "lane-catalog.json"

    @property
    def epoch_diagnostics_dir(self) -> Path:
        return self.run_dir / "epochs"

    @property
    def fixture(self) -> Path | None:
        """The profile's pinned fixture, or ``None`` for the fixture-free
        full-artifact profile (the service resolves its own artifact)."""

        return CONTROLLER_PROFILES[self.controller_profile][0]

    @property
    def fixture_description(self) -> str:
        return CONTROLLER_PROFILES[self.controller_profile][1]

    @property
    def telemetry_path(self) -> Path | None:
        if self.viewer_port is None:
            return None
        return self.run_dir / "presentation-telemetry.ndjson"

    @property
    def action_latency_path(self) -> Path | None:
        """Where the service appends its latency-only samples, when asked.

        Deliberately independent of ``viewer_port``: the brain viewer is refused
        for every non-fixture profile, and the full-artifact profiles are exactly
        the ones whose first-action latency cannot otherwise be measured.
        """

        if not self.action_latency:
            return None
        return self.run_dir / "action-latency.ndjson"

    @property
    def program_decisions_path(self) -> Path | None:
        """Where the body-program lane appends its DIAGNOSTIC decision samples.

        Always on for that lane and absent for every other, because it is the only
        per-tick account a program session has: no trace of the full-brain lane can be
        committed today, so without this file a session's record of what the declared
        mapping asked for is the server log alone. It is diagnostic, not content-addressed,
        and not evidence; the banner says so.
        """

        if self.controller_profile not in FULL_MCNS_PROGRAM_PROFILES:
            return None
        return self.run_dir / "program-decisions.ndjson"

    @property
    def brain_display_path(self) -> Path | None:
        if self.controller_profile not in FULL_MCNS_PROGRAM_PROFILES:
            return None
        return self.run_dir / "brain-display.json"

    @property
    def console_path(self) -> Path | None:
        return self.run_dir / "diagnostic-commands.ndjson" if self.diagnostic_tools else None

    @property
    def client_actions_path(self) -> Path | None:
        return self.run_dir / "diagnostic-client-actions.ndjson" if self.diagnostic_tools else None


class CompanionStop(Exception):
    """An explicit companion stop request, including loss of its control pipe."""


class StdinControl:
    """Bounded, opt-in supervisor input; never accepts arbitrary console text.

    EOF means the companion no longer supervises this session. Stop works during
    startup and gameplay; conditions queue until the live monitor can apply them.
    A requested condition is not an application receipt from Minecraft.
    """

    CONDITIONS = frozenset(("original", "shuffled_weight", "no_input"))

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.stop_requested = threading.Event()
        self.conditions: queue.Queue[str] = queue.Queue(maxsize=8)
        self.thread = threading.Thread(target=self._read, name="companion-control", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _read(self) -> None:
        try:
            try:
                descriptor = self.stream.fileno()
            except (OSError, ValueError):
                self._records(self.stream)  # In-memory test streams have no descriptor.
            else:
                # A daemon blocked on sys.stdin's BufferedReader can abort Python
                # at interpreter shutdown. An unbuffered duplicate has no such
                # lock; it does not close or alter the caller's stdin descriptor.
                with os.fdopen(os.dup(descriptor), "rb", buffering=0) as reader:
                    self._records(reader)
        except (OSError, ValueError):
            self.stop_requested.set()

    def _records(self, reader: TextIO | BinaryIO) -> None:
        try:
            while not self.stop_requested.is_set():
                raw = reader.readline(4097)
                if self.stop_requested.is_set():
                    return
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                if not line:
                    self.stop_requested.set()
                    return
                if len(line) > 4096 or not line.endswith("\n"):
                    print("[companion control refused] incomplete or excessive record", flush=True)
                    # Disable this inbox after an unbounded/incomplete record;
                    # never parse another chunk as a separate command.
                    self.stop_requested.set()
                    return
                try:
                    request = json.loads(line)
                    if request == {"action": "stop"}:
                        self.stop_requested.set()
                        return
                    if (
                        not isinstance(request, dict)
                        or set(request) != {"action", "condition"}
                        or request["action"] != "condition"
                        or not isinstance(request["condition"], str)
                        or request["condition"] not in self.CONDITIONS
                    ):
                        raise ValueError("unknown request")
                    self.conditions.put_nowait(request["condition"])
                except (ValueError, RecursionError, queue.Full):
                    # Input may contain secrets or terminal escapes; do not echo it.
                    print("[companion control refused] invalid request or full queue", flush=True)
        except (OSError, ValueError):
            self.stop_requested.set()


def _check_companion_stop(stop_requested: threading.Event | None) -> None:
    if stop_requested is not None and stop_requested.is_set():
        raise CompanionStop("companion requested stop or closed its control pipe")


def _windows() -> bool:
    return sys.platform == "win32"


def _process_lifecycle_module() -> ModuleType:
    path = ROOT / "scripts" / "process_lifecycle.py"
    spec = importlib.util.spec_from_file_location("neurocraft_process_lifecycle", path)
    if spec is None or spec.loader is None:
        raise LauncherError("process ownership helper is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shutdown_signals() -> tuple[int, ...]:
    return tuple(
        int(getattr(signal, name))
        for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGBREAK")
        if hasattr(signal, name)
    )


class LineCapture:
    """Tee a child stream to a log and retain bounded lines for readiness checks."""

    def __init__(self, stream: TextIO, path: Path, prefix: str) -> None:
        self._stream = stream
        self._path = path
        self._prefix = prefix
        self._lines: list[str] = []
        self._condition = threading.Condition()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.stop_requested: threading.Event | None = None

    def start(self) -> None:
        self._thread.start()

    def join(self, *, timeout: float = 2.0) -> None:
        self._thread.join(timeout=timeout)

    def search(self, pattern: re.Pattern[str]) -> re.Match[str] | None:
        with self._condition:
            for line in reversed(self._lines):
                match = pattern.search(line)
                if match is not None:
                    return match
        return None

    def _run(self) -> None:
        with self._path.open("x", encoding="utf-8") as log:
            for line in self._stream:
                log.write(line)
                log.flush()
                print(f"[{self._prefix}] {line}", end="", flush=True)
                with self._condition:
                    self._lines.append(line.rstrip("\n"))
                    if len(self._lines) > 4000:
                        del self._lines[:1000]
                    self._condition.notify_all()

    def wait_for(
        self,
        pattern: re.Pattern[str],
        *,
        timeout: float,
        process: subprocess.Popen[str] | None = None,
    ) -> re.Match[str]:
        deadline = time.monotonic() + timeout
        cursor = 0
        with self._condition:
            while True:
                _check_companion_stop(self.stop_requested)
                while cursor < len(self._lines):
                    match = pattern.search(self._lines[cursor])
                    cursor += 1
                    if match is not None:
                        return match
                remaining = deadline - time.monotonic()
                if process is not None and process.poll() is not None:
                    raise LauncherError(
                        f"child exited with status {process.returncode} before {pattern.pattern}"
                    )
                if remaining <= 0:
                    raise LauncherError(f"timed out waiting for {pattern.pattern}")
                self._condition.wait(timeout=min(remaining, 0.25))


def _write_exclusive(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())


def _mod_admits_program_frames(source: Path | None = None) -> bool:
    """Whether this mod build's inbound decoder names ``program_action``.

    A missing source file counts as "cannot admit": an absent decoder is not evidence of
    a present one.
    """

    path = CONTROL_PROTOCOL_SOURCE if source is None else source
    try:
        source_text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return PROGRAM_ACTION_DECODE_ARM.search(_BLOCK_COMMENT.sub("", source_text)) is not None


def _accepted_eula(path: Path = ACCEPTED_EULA) -> bool:
    if not path.is_file():
        return False
    values = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    return "eula=true" in values


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        stream.bind(("127.0.0.1", 0))
        return int(stream.getsockname()[1])


def _prepare_jvm_tmp(run_dir: Path) -> Path:
    """Keep native library extraction away from a shared system temp directory.

    CPython 3.13's Windows mkdir(0700) creates a protected user/admin ACL;
    POSIX enforces owner-only permission bits. Existing paths are never reused.
    """
    if _windows() and (sys.implementation.name != "cpython" or sys.version_info < (3, 13)):
        raise LauncherError("Private JVM temporary directories require CPython 3.13+ on Windows")
    directory = run_dir / "jvm-tmp"
    directory.mkdir(mode=0o700)
    if os.name != "nt":
        metadata = directory.stat()
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
            raise LauncherError("JVM temporary directory must be owned by this user with mode 0700")
    return directory


def prepare_run(
    *,
    run_root: Path = RUN_ROOT,
    accepted_eula: Path = ACCEPTED_EULA,
    now: datetime | None = None,
    controller_profile: str = DEFAULT_CONTROLLER_PROFILE,
    brain_viewer: bool = False,
    action_latency: bool = False,
    game_tick_rate: int = 20,
    diagnostic_tools: bool = False,
    comparison_lanes: bool = False,
    control_seed: int = 7,
    kernel_threads: int | None = None,
    capture_fps: int = 1,
    capture_seconds: int = 120,
    capture_png_compression: int | None = None,
    client_width: int | None = None,
    client_height: int | None = None,
    client_max_fps: int = 60,
    showcase_world: bool = False,
    circuit_playground: bool = False,
    installed_runtime: Path | None = None,
    manual_client_name: str | None = None,
) -> RunPlan:
    """Create an isolated, no-clobber run without starting processes."""

    if (installed_runtime is None) != (manual_client_name is None):
        raise LauncherError("installed runtime and manual client name must be provided together")
    if manual_client_name is not None:
        if re.fullmatch(r"[A-Za-z0-9_]{3,16}", manual_client_name) is None:
            raise LauncherError("manual client name must be a valid Java player name")
        if diagnostic_tools or capture_png_compression is not None:
            raise LauncherError("manual clients do not support launcher diagnostic capture tools")
        assert installed_runtime is not None
        installed_runtime = installed_runtime.absolute()
        try:
            _installed_mod_module().validate_runtime(installed_runtime)
        except (OSError, ValueError) as exc:
            raise LauncherError(f"installed runtime refused: {exc}") from exc

    if type(game_tick_rate) is not int or game_tick_rate not in (10, 20):
        raise LauncherError("game tick rate must be 10 (half-speed diagnostic) or 20 (normal)")
    if (client_width is None) != (client_height is None):
        raise LauncherError("client width and height must be provided together")
    if client_width is not None and (
        type(client_width) is not int
        or not 320 <= client_width <= 1920
        or type(client_height) is not int
        or not 240 <= client_height <= 1080
    ):
        raise LauncherError(
            "client logical dimensions require width 320..1920 and height 240..1080"
        )
    if type(client_max_fps) is not int or not 20 <= client_max_fps <= 120:
        raise LauncherError("client max fps must be an integer in 20..120")
    if controller_profile not in CONTROLLER_PROFILES:
        raise LauncherError(f"unknown controller profile: {controller_profile}")
    if kernel_threads is not None and (
        type(kernel_threads) is not int
        or kernel_threads not in (1, 2, 4, 8)
        or controller_profile not in FULL_MCNS_PROFILES
    ):
        raise LauncherError("kernel threads require a full MaleCNS profile and 1,2,4,8 threads")
    if comparison_lanes and controller_profile not in FULL_MCNS_COMPARISON_PROFILES:
        raise LauncherError("comparison lanes require a registered comparison program profile")
    if type(control_seed) is not int or not 0 <= control_seed < 2**64:
        raise LauncherError("control seed must be an unsigned 64-bit integer")
    if not comparison_lanes and control_seed != 7:
        raise LauncherError("a non-default control seed requires comparison lanes")
    if (
        type(capture_fps) is not int
        or not 1 <= capture_fps <= 20
        or type(capture_seconds) is not int
        or not 1 <= capture_seconds <= 720
    ):
        raise LauncherError("capture requires 1..20 fps and 1..720 seconds")
    if not diagnostic_tools and (capture_fps, capture_seconds) != (1, 120):
        raise LauncherError("capture settings require --diagnostic-tools")
    if capture_png_compression is not None and (
        type(capture_png_compression) is not int or not 0 <= capture_png_compression <= 9
    ):
        raise LauncherError("capture PNG compression must be an integer in 0..9")
    if capture_png_compression is not None and not diagnostic_tools:
        raise LauncherError("capture PNG compression requires --diagnostic-tools")
    if brain_viewer and CONTROLLER_PROFILES[controller_profile][0] != STEERING_FIXTURE:
        raise LauncherError(
            "the brain viewer renders the 128-node steering fixture; use a "
            "steering-fixture controller profile with --brain-viewer"
        )
    if controller_profile in FULL_MCNS_PROGRAM_PROFILES and not _mod_admits_program_frames():
        raise LauncherError(PROGRAM_LANE_NOT_RUNNABLE)
    if controller_profile in FULL_MCNS_PROFILES:
        if not (FULL_ARTIFACT_DIR / "manifest.json").is_file():
            raise LauncherError(
                f"{controller_profile} is a local-only profile; the full artifact "
                f"is not present at {FULL_ARTIFACT_DIR} (build it with "
                "scripts/build_malecns_runtime.py)"
            )
        if not FULL_ANNOTATIONS_PATH.is_file():
            raise LauncherError(
                f"{controller_profile} is a local-only profile; the neurotransmitter "
                f"annotations table is not present at {FULL_ANNOTATIONS_PATH} (fetch it "
                "with scripts/acquire_malecns.py --include-metadata)"
            )
    if (
        controller_profile in FULL_MCNS_BODY_ANNOTATION_PROFILES
        and not FULL_BODY_ANNOTATIONS_PATH.is_file()
    ):
        raise LauncherError(
            f"{controller_profile} is a local-only profile; the body-annotations "
            f"table is not present at {FULL_BODY_ANNOTATIONS_PATH} (fetch it with "
            "scripts/acquire_malecns.py --include-metadata)"
        )
    if not _accepted_eula(accepted_eula):
        raise LauncherError("mod/run/eula.txt must contain eula=true before launching")
    timestamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"interactive-{timestamp}-{secrets.token_hex(4)}"
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    _prepare_jvm_tmp(run_dir)
    server_dir = run_dir / "server"
    client_dir = run_dir / "client"
    server_dir.mkdir(mode=0o700)
    client_dir.mkdir(mode=0o700)
    bridge_port = _available_port()
    game_port = _available_port()
    while game_port == bridge_port:
        game_port = _available_port()
    entity_id = str(uuid.uuid4())
    session_epoch = time.time_ns()
    _write_exclusive(
        server_dir / "eula.txt", "# Mirrored from the user's accepted EULA\neula=true\n"
    )
    properties = {
        "allow-flight": "true",
        "difficulty": "peaceful",
        "enable-command-block": "false",
        "enable-query": "false",
        "enable-rcon": "false",
        "enforce-secure-profile": "false",
        "force-gamemode": "true",
        "gamemode": "creative",
        "generate-structures": "false",
        "level-name": "world",
        "level-type": "minecraft:flat",
        "max-players": "2",
        "motd": "NeuroCraft Fly local interactive demo",
        "online-mode": "false",
        "pause-when-empty-seconds": "-1",
        "region-file-compression": "deflate",
        "server-ip": "127.0.0.1",
        "server-port": str(game_port),
        "spawn-protection": "0",
        "view-distance": "6",
        "white-list": "false",
    }
    _write_exclusive(
        server_dir / "server.properties",
        "".join(f"{key}={properties[key]}\n" for key in sorted(properties)),
    )
    viewer_port: int | None = None
    if brain_viewer:
        viewer_port = _available_port()
        while viewer_port in (bridge_port, game_port):
            viewer_port = _available_port()
    plan = RunPlan(
        run_id=run_id,
        entity_id=entity_id,
        session_epoch=session_epoch,
        bridge_port=bridge_port,
        game_port=game_port,
        run_dir=run_dir,
        server_dir=server_dir,
        client_dir=client_dir,
        identity_path=run_dir / "identity.json",
        session_path=run_dir / "session.json",
        controller_profile=controller_profile,
        viewer_port=viewer_port,
        action_latency=action_latency,
        game_tick_rate=game_tick_rate,
        diagnostic_tools=diagnostic_tools,
        comparison_lanes=comparison_lanes,
        control_seed=control_seed,
        kernel_threads=kernel_threads,
        capture_fps=capture_fps,
        capture_seconds=capture_seconds,
        capture_png_compression=capture_png_compression,
        client_width=client_width,
        client_height=client_height,
        client_max_fps=client_max_fps,
        showcase_world=showcase_world or circuit_playground,
        circuit_playground=circuit_playground,
        showcase_commands=(
            _showcase_commands(circuit_playground=circuit_playground)
            if showcase_world or circuit_playground
            else ()
        ),
        installed_runtime=installed_runtime,
        manual_client_name=manual_client_name,
    )
    if installed_runtime is not None:
        try:
            _installed_mod_module().copy_runtime(installed_runtime, server_dir)
        except (OSError, ValueError) as exc:
            raise LauncherError(f"installed runtime copy refused: {exc}") from exc
    _write_exclusive(
        client_dir / "options.txt",
        "".join(f"{key}:{value}\n" for key, value in plan.requested_client_settings.items()),
    )
    if plan.console_path is not None:
        _write_exclusive(plan.console_path, "")
    if plan.client_actions_path is not None:
        _write_exclusive(plan.client_actions_path, "")
    _write_session(plan, state="prepared", identity=None)
    return plan


def _write_session(
    plan: RunPlan,
    *,
    state: str,
    identity: Mapping[str, object] | None,
    exit_reason: str | None = None,
    catalog_sha256: str | None = None,
) -> None:
    document: dict[str, object] = {
        "schema": "neurocraft.interactive_session.v1",
        "state": state,
        "claim_boundary": (
            "installed mod server with manually launched client; client installation and "
            "visual behavior require separate observation; not causal or biological evidence"
            if plan.installed_runtime is not None
            else "interactive development client; not production-JAR, causal-trace, "
            "G3, G3-R, full-MaleCNS, or G6 evidence"
        ),
        "launch_mode": "installed_mod" if plan.installed_runtime is not None else "development",
        "runtime_manifest_sha256": (
            sha256((plan.installed_runtime / "runtime-manifest.json").read_bytes()).hexdigest()
            if plan.installed_runtime is not None
            else None
        ),
        "client_settings_managed": plan.installed_runtime is None,
        "fixture": plan.fixture_description,
        "controller_profile": plan.controller_profile,
        "run_id": plan.run_id,
        "entity_id": plan.entity_id,
        "session_epoch": plan.session_epoch,
        "bridge": f"127.0.0.1:{plan.bridge_port}",
        "minecraft": f"127.0.0.1:{plan.game_port}",
        "server_directory": str(plan.server_dir.relative_to(ROOT)),
        "client_directory": str(plan.client_dir.relative_to(ROOT)),
        "authentication": "loopback-only development offline mode",
        "requested_client_settings": plan.requested_client_settings,
        "showcase_world": plan.showcase_world,
        "circuit_playground": plan.circuit_playground,
        "starting_view": _starting_view(plan.circuit_playground),
        "showcase_commands_sha256": (
            sha256(json.dumps(plan.showcase_commands, separators=(",", ":")).encode()).hexdigest()
            if plan.showcase_world
            else None
        ),
        "client_settle_seconds": CLIENT_SETTLE_SECONDS,
        "requested_game_ticks_per_wall_second": plan.game_tick_rate,
        "playback_speed": "normal" if plan.game_tick_rate == 20 else "half-speed diagnostic",
        "diagnostic_tools": plan.diagnostic_tools,
        "diagnostic_client_actions_path": str(plan.client_actions_path.relative_to(ROOT))
        if plan.client_actions_path is not None
        else None,
        "requested_capture_fps": plan.capture_fps if plan.diagnostic_tools else None,
        "requested_capture_seconds": plan.capture_seconds if plan.diagnostic_tools else None,
        "requested_capture_png_compression": plan.capture_png_compression,
        "comparison_lanes": plan.comparison_lanes,
        "lane_catalog_sha256": catalog_sha256,
        "control_seed": str(plan.control_seed) if plan.comparison_lanes else None,
        "requested_kernel_threads": plan.kernel_threads,
        "lane_catalog": str(plan.lane_catalog_path.relative_to(ROOT))
        if plan.comparison_lanes
        else None,
        "epoch_diagnostics_directory": str(plan.epoch_diagnostics_dir.relative_to(ROOT))
        if plan.comparison_lanes
        else None,
        "secret": f"{TOKEN_NAME} supplied to child environment; value omitted",
    }
    if identity is not None:
        document["identity"] = dict(identity)
    if exit_reason is not None:
        document["exit_reason"] = exit_reason
    encoded = json.dumps(document, allow_nan=False, separators=(",", ":"), sort_keys=True)
    temporary = plan.session_path.with_suffix(".json.tmp")
    temporary.write_text(encoded + "\n", encoding="utf-8")
    os.replace(temporary, plan.session_path)


def _identity_command(plan: RunPlan) -> list[str]:
    fixture_arguments = [] if plan.fixture is None else ["--fixture", str(plan.fixture)]
    return [
        sys.executable,
        "-I",
        "-X",
        "utf8",
        str(ROOT / "scripts" / "run_connectome_service.py"),
        *fixture_arguments,
        "--controller-profile",
        plan.controller_profile,
        "--run-id",
        plan.run_id,
        "--entity-id",
        plan.entity_id,
        "--session-epoch",
        str(plan.session_epoch),
        "--mode",
        "faithful",
        "--identity-out",
        str(plan.identity_path),
        "--identity-only",
        *(
            ["--kernel-threads", str(plan.kernel_threads)]
            if plan.kernel_threads is not None
            else []
        ),
    ]


def _load_identity(path: Path, plan: RunPlan) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != IDENTITY_KEYS:
        raise LauncherError("controller identity has missing or unknown fields")
    if raw["run_id"] != plan.run_id or raw["entity_id"] != plan.entity_id:
        raise LauncherError("controller identity does not match the fresh run")
    if type(raw["session_epoch"]) is not int or raw["session_epoch"] != plan.session_epoch:
        raise LauncherError("controller session epoch does not match the fresh run")
    for key in HASH_KEYS:
        value = raw[key]
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise LauncherError(f"controller identity {key} is not a SHA-256")
    return raw


def _service_command(plan: RunPlan) -> list[str]:
    command = _identity_command(plan)
    identity_index = command.index("--identity-out")
    del command[identity_index : identity_index + 3]
    command.extend(["--host", "127.0.0.1", "--port", str(plan.bridge_port)])
    if plan.telemetry_path is not None:
        command.extend(["--presentation-telemetry-out", str(plan.telemetry_path)])
    if plan.action_latency_path is not None:
        command.extend(["--action-latency-out", str(plan.action_latency_path)])
    if plan.program_decisions_path is not None:
        command.extend(["--program-decisions-out", str(plan.program_decisions_path)])
    if plan.brain_display_path is not None:
        command.extend(["--brain-display-out", str(plan.brain_display_path)])
    if plan.comparison_lanes:
        command.extend(
            [
                "--comparison-lanes",
                "--lane-catalog-out",
                str(plan.lane_catalog_path),
                "--epoch-diagnostics-dir",
                str(plan.epoch_diagnostics_dir),
                "--control-seed",
                str(plan.control_seed),
            ]
        )
    return command


def _catalog_digest(path: Path, initial_identity: Mapping[str, object]) -> str:
    """Bind the launch to bounded service-produced bytes, including the initial identity.

    The server independently checks the raw digest and full catalog contract before
    connecting. It never trusts mutable catalog contents on a later lane switch.
    """

    def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate catalog key")
            result[key] = value
        return result

    try:
        with path.open("rb") as stream:
            data = stream.read(16_385)
        if len(data) > 16_384:
            raise ValueError("catalog exceeds 16 KiB")
        catalog = json.loads(data.decode("utf-8"), object_pairs_hook=unique_pairs)
        if not isinstance(catalog, dict) or catalog.get("schema") != "neurocraft.lane_catalog.v1":
            raise ValueError("catalog schema does not match")
        identity = catalog.get("initial_identity")
        if (
            not isinstance(identity, dict)
            or identity != dict(initial_identity)
            or type(identity.get("session_epoch")) is not int
        ):
            raise ValueError("catalog initial identity does not match")
    except (OSError, ValueError, RecursionError) as exc:
        raise LauncherError(f"comparison catalog refused: {exc}") from exc
    return sha256(data).hexdigest()


def _viewer_command(plan: RunPlan) -> list[str]:
    if plan.viewer_port is None or plan.telemetry_path is None:
        raise LauncherError("viewer command requested without a viewer port")
    return [
        sys.executable,
        "-I",
        "-X",
        "utf8",
        str(ROOT / "scripts" / "run_brain_viewer.py"),
        "--telemetry",
        str(plan.telemetry_path),
        "--port",
        str(plan.viewer_port),
    ]


def _server_command(
    plan: RunPlan, identity: Mapping[str, object], *, catalog_sha256: str | None = None
) -> list[str]:
    if plan.comparison_lanes and (
        catalog_sha256 is None or re.fullmatch(r"[0-9a-f]{64}", catalog_sha256) is None
    ):
        raise LauncherError("comparison server requires the verified catalog digest")
    command = [
        "gradlew.bat" if _windows() else "./gradlew",
        "--no-daemon",
        "--dependency-verification",
        "strict",
        f"-Djava.io.tmpdir={plan.run_dir / 'jvm-tmp'}",
        f"-Dneurocraft.fly.runDirectory={plan.server_dir}",
        f"-Dneurocraft.fly.bridgeUri=ws://127.0.0.1:{plan.bridge_port}/control",
        f"-Dneurocraft.fly.runId={plan.run_id}",
        f"-Dneurocraft.fly.sessionEpoch={plan.session_epoch}",
        f"-Dneurocraft.fly.entityId={plan.entity_id}",
        f"-Dneurocraft.fly.graphHash={identity['graph_hash']}",
        f"-Dneurocraft.fly.adapterHash={identity['adapter_hash']}",
        f"-Dneurocraft.fly.dynamicsHash={identity['dynamics_hash']}",
        f"-Dneurocraft.fly.decoderHash={identity['decoder_hash']}",
        f"-Dneurocraft.fly.observationSchemaHash={identity['observation_schema_hash']}",
        *(
            []
            if plan.brain_display_path is None or plan.comparison_lanes
            else [f"-Dneurocraft.fly.brainDisplayPath={plan.brain_display_path}"]
        ),
        *(
            [f"-Dneurocraft.fly.eventDisplayPath={plan.run_dir / 'event-selection-display.json'}"]
            if plan.controller_profile
            in (FULL_MCNS_EVENTS_PROGRAM_V1_PROFILE, FULL_MCNS_INTERACTIONS_PROGRAM_V2_PROFILE)
            and not plan.comparison_lanes
            else []
        ),
        *(
            [
                f"-Dneurocraft.fly.laneCatalogPath={plan.lane_catalog_path}",
                f"-Dneurocraft.fly.laneCatalogSha256={catalog_sha256}",
                f"-Dneurocraft.fly.epochDiagnosticsDirectory={plan.epoch_diagnostics_dir}",
            ]
            if plan.comparison_lanes
            else []
        ),
        "runServer",
    ]
    if plan.installed_runtime is not None:
        return [
            str(_installed_mod_module().java_executable()),
            "-Xmx2G",
            *command[4:-1],
            *_installed_mod_module().java_runtime.server_arguments(plan.server_dir),
        ]
    return command


def _installed_mod_module() -> ModuleType:
    path = ROOT / "scripts" / "installed_mod.py"
    spec = importlib.util.spec_from_file_location("neurocraft_installed_mod", path)
    if spec is None or spec.loader is None:
        raise LauncherError("installed runtime helper is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _client_command(plan: RunPlan) -> list[str]:
    return [
        "gradlew.bat" if _windows() else "./gradlew",
        "--no-daemon",
        "--dependency-verification",
        "strict",
        f"-Djava.io.tmpdir={plan.run_dir / 'jvm-tmp'}",
        f"-Dneurocraft.fly.runDirectory={plan.client_dir}",
        f"-Dneurocraft.fly.quickPlayServer=127.0.0.1:{plan.game_port}",
        f"-Dneurocraft.fly.username={USERNAME}",
        *(
            [
                f"-Dneurocraft.fly.captureDirectory={plan.run_dir / 'capture'}",
                f"-Dneurocraft.fly.captureFps={plan.capture_fps}",
                f"-Dneurocraft.fly.captureSeconds={plan.capture_seconds}",
                f"-Dneurocraft.fly.clientActionsPath={plan.client_actions_path}",
            ]
            if plan.diagnostic_tools
            else []
        ),
        *(
            [f"-Dneurocraft.fly.capturePngCompression={plan.capture_png_compression}"]
            if plan.capture_png_compression is not None
            else []
        ),
        "runClient",
    ]


def read_diagnostic_commands(path: Path, cursor: int) -> tuple[int, list[str]]:
    """Read bounded complete command records from this fresh world's opt-in inbox.

    Dispatch is a console request, not an application receipt. Partial final records
    wait for a newline; malformed input disables only the diagnostic inbox.
    """
    if path.stat().st_size > 65_536:
        raise ValueError("diagnostic command file exceeds 64 KiB")
    with path.open("rb") as stream:
        stream.seek(cursor)
        pending = stream.read(65_537)
    if path.stat().st_size < cursor:
        raise ValueError("diagnostic command file was truncated")
    complete = pending[: pending.rfind(b"\n") + 1].splitlines(keepends=True)[:16]
    commands = []
    for line in complete:
        try:
            item = json.loads(line)
        except RecursionError as exc:
            raise ValueError("diagnostic record nesting exceeds parser limits") from exc
        if not isinstance(item, dict) or set(item) != {"command"}:
            raise ValueError("diagnostic record must contain only command")
        command = item["command"]
        if (
            not isinstance(command, str)
            or not 1 <= len(command) <= 512
            or command != command.strip()
            or command.startswith("/")
            or any(ord(character) < 32 for character in command)
        ):
            raise ValueError("diagnostic command must be one bounded console line")
        commands.append(command)
    return cursor + sum(map(len, complete)), commands


def _tcp_ready(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        stream.settimeout(0.1)
        return stream.connect_ex(("127.0.0.1", port)) == 0


def _settle_client(
    *processes: subprocess.Popen[str], stop_requested: threading.Event | None = None
) -> None:
    """Let initial arena chunks render before spawning a controlled fly.

    This is startup presentation grace, not a change to the command deadline or
    a claim that the host can sustain the neural workload. There is no fly yet.
    """
    deadline = time.monotonic() + CLIENT_SETTLE_SECONDS
    while (remaining := deadline - time.monotonic()) > 0:
        _check_companion_stop(stop_requested)
        if any(process.poll() is not None for process in processes):
            raise LauncherError("a demo process exited while the client was settling")
        time.sleep(min(0.1, remaining))


def _wait_tcp(
    process: subprocess.Popen[str],
    port: int,
    timeout: float,
    stop_requested: threading.Event | None = None,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _check_companion_stop(stop_requested)
        if process.poll() is not None:
            raise LauncherError(f"child exited before opening loopback port {port}")
        if _tcp_ready(port):
            return
        time.sleep(0.05)
    raise LauncherError(f"child did not open loopback port {port}")


def _send_server(process: subprocess.Popen[str], command: str) -> None:
    if process.stdin is None or process.poll() is not None:
        raise LauncherError("Minecraft server is not accepting console commands")
    process.stdin.write(command + "\n")
    process.stdin.flush()


def _terminate(process: subprocess.Popen[str] | None, *, timeout: float = 8.0) -> None:
    """Stop our entire session group, even if its Gradle wrapper already exited.

    Only pass children created by ``_start`` (``start_new_session=True``). A
    wrapper's exit says nothing about its surviving JVMs. SIGKILL cannot run
    Python cleanup, and a child that deliberately starts another session escapes
    this process-group boundary.
    """
    if process is None:
        return
    if _windows():
        try:
            _process_lifecycle_module().terminate_windows(process, timeout=timeout)
        except OSError as exc:
            raise LauncherError(f"Windows child cleanup failed: {exc}") from exc
        return
    try:
        _process_lifecycle_module().terminate_posix(process, timeout=timeout)
    except OSError as exc:
        raise LauncherError(str(exc)) from exc.__cause__


@contextmanager
def _cleanup_signals() -> Iterator[None]:
    """A repeated interrupt must not abandon the remaining owned children."""
    handlers = {}
    if threading.current_thread() is threading.main_thread():
        for signum in _shutdown_signals():
            handlers[signum] = signal.signal(signum, signal.SIG_IGN)
    try:
        yield
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def _cleanup_children(
    children: Mapping[str, subprocess.Popen[str] | None], captures: Sequence[LineCapture]
) -> list[str]:
    """Attempt every cleanup even if a pipe is broken or one group refuses to exit."""
    errors = []
    with _cleanup_signals():
        for name, process in children.items():
            if name == "server" and process is not None:
                try:
                    if process.poll() is None:
                        _send_server(process, "neurocraftfly status")
                        _send_server(process, "stop")
                        process.wait(timeout=15.0)
                except Exception as exc:
                    errors.append(f"server console shutdown: {type(exc).__name__}: {exc}")
            try:
                _terminate(process)
            except Exception as exc:
                errors.append(f"{name} termination: {type(exc).__name__}: {exc}")
        for capture in captures:
            try:
                capture.join()
            except Exception as exc:
                errors.append(f"log capture shutdown: {type(exc).__name__}: {exc}")
    return errors


def _child_environment(token: str) -> dict[str, str]:
    forbidden = (
        "PYTHONPATH",
        "PYTHONHOME",
        "JAVA_TOOL_OPTIONS",
        "JDK_JAVA_OPTIONS",
        "GRADLE_OPTS",
    )
    present = [key for key in forbidden if os.environ.get(key)]
    if present:
        raise LauncherError(f"remove environment injection variables before launch: {present}")
    allowed = (
        "HOME",
        "JAVA_HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "TMPDIR",
        "GRADLE_USER_HOME",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "COMMONPROGRAMFILES",
        "HOMEDRIVE",
        "HOMEPATH",
        "PATHEXT",
    )
    # Windows environment names are case insensitive (for example SystemRoot).
    allowed_names = {key.upper() for key in allowed}
    environment = {key: value for key, value in os.environ.items() if key.upper() in allowed_names}
    environment[TOKEN_NAME] = token
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    return environment


def _start(
    argv: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    stdin: int | None = None,
) -> subprocess.Popen[str]:
    if _windows():
        return cast(
            subprocess.Popen[str],
            _process_lifecycle_module().start_windows(
                argv, cwd=cwd, environment=environment, stdin=stdin
            ),
        )
    return subprocess.Popen(
        list(argv),
        cwd=cwd,
        env=dict(environment),
        stdin=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        start_new_session=True,
        shell=False,
    )


# The arena is a sealed box centred on the origin: a smooth-stone floor plus glass walls
# and a glass ceiling, each ARENA_SHELL_BLOCKS thick, enclosing
# (2*ARENA_HALF_WIDTH + 1) x (2*ARENA_HALF_WIDTH + 1) blocks of air between
# ARENA_INTERIOR_FLOOR_Y and ARENA_INTERIOR_CEILING_Y inclusive.
#
# Width (D-067). Under the ADR 0014 hover envelope a held forward command f cruises at
# 0.04*f/(1 - 0.91) = 0.4444*f blocks/tick -- PrimitiveActuator adds the thrust,
# FlyEntity.integratePassivePhysics moves by the sum and then applies the 0.91
# horizontal drag -- and yaw turns 6 deg/tick per unit torque, so a held (f, yaw) pair
# traces a circle of radius R = 4.2441*f/|yaw| blocks. Against the escape-v2 amplitudes
# recorded in D-066 that is R = 10.2 blocks (blocked-right: f = 0.0835 with yaw
# -0.0348) and R = 16.9 blocks (the blocked-front f = 0.1383 at that same strongest
# measured yaw). The previous 23-block interior contained neither circle -- the fly
# could not complete one turn inside its own box -- while 49 blocks contains both with
# at least 7 blocks of clearance per side. Circles at the weaker measured yaws do not
# fit and no plausible box holds them: R = 31 blocks at the 0.0188 yaw the D-059 live
# sessions actually held, R = 112 blocks for the blocked-left pair. That is a
# steering-authority limit -- the measured yaw torque is 2-3.5% of the +/-1.0 rail --
# not an arena limit, and this constant does not address it. The box is not widened
# further because FlySensors.RAY_LENGTH_BLOCKS is 6.0: every extra block of width moves
# more of the volume out of ray range of any wall, toward the all-open condition where
# the measured commands are ~0.001..0.005 (D-064/D-066) and the fly would drift at
# ~0.04 blocks/s.
#
# Height (D-067). At the D-066 loom vertical command (+0.2014, i.e. 0.008056
# blocks/tick^2 against the 0.98 vertical drag) the fly climbs 5 blocks in ~39 ticks
# (~2.0 s), which was the entire old headroom above the spawn. 24 blocks of air leaves
# ~20 blocks above the spawn -- ~91 ticks, ~4.6 s to the ceiling -- and comfortably
# contains the measured +0.4787 touch transient.
#
# The shell is double-thick because a live fly escaped the single-layer box while the
# operator was placing and breaking blocks nearby (D-056): one misplaced block-break
# must not open the arena. The floor is double-thick here too -- the world below it is
# open air down to the superflat surface, so it is an escape route like any other face.
#
# These are launcher constants with no hash: the arena rotates no identity. The
# identity-bound worlds are HarnessWorld (scripts/fixture_harness.py) and ReferenceArena
# (packages/embodiment-runtime); neither reads anything from this module.
ARENA_HALF_WIDTH = 24
ARENA_INTERIOR_FLOOR_Y = 64
ARENA_INTERIOR_CEILING_Y = 87
ARENA_SHELL_BLOCKS = 2
#: Vanilla `/fill` refuses more than 32768 blocks in one command, and the refusal never
#: reaches the launcher -- a shell face would simply be missing from a box the tests
#: still believe is sealed. The interior air (57,624 blocks) is therefore emitted as
#: slabs under this cap, and a test pins every emitted fill against it.
ARENA_MAX_FILL_BLOCKS = 32768

# Both coordinates below must stay strictly inside the enclosed air, and the fly is
# spawned one block above the observer, so SPAWN_Y + 1 must also clear the ceiling.
SPAWN_X = 0.5
SPAWN_Y = 66
SPAWN_Z = 0.5
# Where the observer is moved after the fly exists, so the fly is in front of them
# rather than inside their camera. The standoff grew with the arena: from 21 blocks back
# the far wall (49 blocks wide, 44.5 blocks away) sits inside even the vertical 70-degree
# default FOV, while the fly -- 0.38 x 0.22 blocks -- still subtends about a degree and
# stays findable. VIEW_Y is a starting vantage rather than a held one: a creative
# observer who has not engaged flight settles back to the floor within about a second.
VIEW_X = 0.5
VIEW_Y = 70
VIEW_Z = -20.5


def _fill(low: tuple[int, int, int], high: tuple[int, int, int], block: str) -> str:
    """One `/fill` over the inclusive box between two corners."""

    return f"fill {low[0]} {low[1]} {low[2]} {high[0]} {high[1]} {high[2]} minecraft:{block}"


def _arena_fills() -> tuple[str, ...]:
    """The sealed-box fills, derived from the arena constants above.

    Derived rather than written out so the shell cannot drift out of agreement with
    the interior it encloses, and so the air volume is split into as many slabs as
    ARENA_MAX_FILL_BLOCKS requires instead of silently exceeding it.
    """

    low = -ARENA_HALF_WIDTH
    high = ARENA_HALF_WIDTH
    outer_low = low - ARENA_SHELL_BLOCKS
    outer_high = high + ARENA_SHELL_BLOCKS
    floor_top = ARENA_INTERIOR_FLOOR_Y - 1
    floor_bottom = floor_top - ARENA_SHELL_BLOCKS + 1
    ceiling_bottom = ARENA_INTERIOR_CEILING_Y + 1
    ceiling_top = ceiling_bottom + ARENA_SHELL_BLOCKS - 1
    width = high - low + 1
    layers = ARENA_INTERIOR_CEILING_Y - ARENA_INTERIOR_FLOOR_Y + 1
    slabs = -(-layers // (ARENA_MAX_FILL_BLOCKS // (width * width)))
    slab_layers = -(-layers // slabs)
    air = tuple(
        _fill(
            (low, base, low),
            (high, min(base + slab_layers - 1, ARENA_INTERIOR_CEILING_Y), high),
            "air",
        )
        for base in range(ARENA_INTERIOR_FLOOR_Y, ARENA_INTERIOR_CEILING_Y + 1, slab_layers)
    )
    return (
        _fill(
            (outer_low, floor_bottom, outer_low),
            (outer_high, floor_top, outer_high),
            "smooth_stone",
        ),
        *air,
        # Walls span the interior height; the floor and ceiling slabs already cover the
        # full outer footprint above and below them, so the box closes at every corner.
        _fill(
            (outer_low, ARENA_INTERIOR_FLOOR_Y, outer_low),
            (low - 1, ARENA_INTERIOR_CEILING_Y, outer_high),
            "glass",
        ),
        _fill(
            (high + 1, ARENA_INTERIOR_FLOOR_Y, outer_low),
            (outer_high, ARENA_INTERIOR_CEILING_Y, outer_high),
            "glass",
        ),
        _fill(
            (outer_low, ARENA_INTERIOR_FLOOR_Y, outer_low),
            (outer_high, ARENA_INTERIOR_CEILING_Y, low - 1),
            "glass",
        ),
        _fill(
            (outer_low, ARENA_INTERIOR_FLOOR_Y, high + 1),
            (outer_high, ARENA_INTERIOR_CEILING_Y, outer_high),
            "glass",
        ),
        _fill(
            (outer_low, ceiling_bottom, outer_low),
            (outer_high, ceiling_top, outer_high),
            "glass",
        ),
    )


def _starting_view(circuit_playground: bool) -> tuple[float, float, float, float, float]:
    """Scene preparation only; leave a grounded player within native brush reach."""

    return (
        (0.5, 65.0, -2.0, 0.0, -11.0) if circuit_playground else (VIEW_X, VIEW_Y, VIEW_Z, 0.0, 0.0)
    )


def _showcase_commands(*, circuit_playground: bool = False) -> tuple[str, ...]:
    # Load our adjacent source explicitly: this launcher also runs with Python -I,
    # which intentionally excludes the script directory from import search paths.
    path = Path(__file__).with_name("showcase_world.py")
    spec = importlib.util.spec_from_file_location("neurocraft_showcase_world", path)
    if spec is None or spec.loader is None:
        raise LauncherError("showcase world source is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    commands = module.playground_commands() if circuit_playground else module.showcase_commands()
    return cast("tuple[str, ...]", commands)


def _prepare_arena(
    server: subprocess.Popen[str], *, showcase_commands: tuple[str, ...] = ()
) -> None:
    commands = (
        "gamerule doDaylightCycle false",
        "gamerule doWeatherCycle false",
        "time set day",
        "weather clear",
        *_arena_fills(),
        *showcase_commands,
        f"setworldspawn 0 {ARENA_INTERIOR_FLOOR_Y} 0",
    )
    for command in commands:
        _send_server(server, command)


def run_interactive(
    plan: RunPlan,
    *,
    startup_timeout: float = 90.0,
    client_join_timeout: float = 600.0,
    control_stdin: bool = False,
) -> int:
    """Run the service and fresh server until the managed client exits or manual user leaves."""

    identity: dict[str, object] | None = None
    catalog_sha256: str | None = None
    token = ""
    service: subprocess.Popen[str] | None = None
    viewer: subprocess.Popen[str] | None = None
    server: subprocess.Popen[str] | None = None
    client: subprocess.Popen[str] | None = None
    captures: list[LineCapture] = []
    exit_reason = "launcher failure"
    failed = False
    control = StdinControl(sys.stdin) if control_stdin else None
    stop_requested = None if control is None else control.stop_requested
    if control is not None:
        control.start()
    try:
        _check_companion_stop(stop_requested)
        if _tcp_ready(plan.bridge_port) or _tcp_ready(plan.game_port):
            raise LauncherError("a selected loopback port acquired a listener before launch")
        identity_result = subprocess.run(
            _identity_command(plan),
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=30,
        )
        if identity_result.returncode != 0:
            raise LauncherError(
                f"could not derive controller identity: {identity_result.stderr.strip()}"
            )
        _check_companion_stop(stop_requested)
        identity = _load_identity(plan.identity_path, plan)
        token = secrets.token_urlsafe(48)
        environment = _child_environment(token)
        service = _start(
            _service_command(plan), cwd=ROOT, environment=environment, stdin=subprocess.DEVNULL
        )
        assert service.stdout is not None
        service_capture = LineCapture(
            cast(TextIO, service.stdout), plan.run_dir / "service.log", "service"
        )
        captures.append(service_capture)
        service_capture.stop_requested = stop_requested
        service_capture.start()
        _wait_tcp(
            service, plan.bridge_port, startup_timeout, *([stop_requested] if control else [])
        )
        catalog_sha256 = (
            _catalog_digest(plan.lane_catalog_path, identity) if plan.comparison_lanes else None
        )

        if plan.viewer_port is not None:
            viewer = _start(
                _viewer_command(plan), cwd=ROOT, environment=environment, stdin=subprocess.DEVNULL
            )
            assert viewer.stdout is not None
            viewer_capture = LineCapture(
                cast(TextIO, viewer.stdout), plan.run_dir / "viewer.log", "viewer"
            )
            captures.append(viewer_capture)
            viewer_capture.stop_requested = stop_requested
            viewer_capture.start()
            _wait_tcp(viewer, plan.viewer_port, 15.0, *([stop_requested] if control else []))

        server = _start(
            _server_command(plan, identity, catalog_sha256=catalog_sha256),
            cwd=plan.server_dir if plan.installed_runtime is not None else MOD_ROOT,
            environment=environment,
            stdin=subprocess.PIPE,
        )
        assert server.stdout is not None
        server_capture = LineCapture(
            cast(TextIO, server.stdout), plan.run_dir / "server-gradle.log", "server"
        )
        captures.append(server_capture)
        server_capture.stop_requested = stop_requested
        server_capture.start()
        server_capture.wait_for(re.compile(r"Done \("), timeout=startup_timeout, process=server)
        _wait_tcp(server, plan.game_port, 10.0, *([stop_requested] if control else []))
        deadline = time.monotonic() + 15.0
        while True:
            _send_server(server, "neurocraftfly status")
            try:
                server_capture.wait_for(
                    re.compile(r"bridge=ACTIVE loaded_flies=0"),
                    timeout=1.0,
                    process=server,
                )
                break
            except LauncherError as exc:
                if time.monotonic() >= deadline:
                    raise LauncherError(
                        "bridge did not become ACTIVE before client launch"
                    ) from exc
        _prepare_arena(server, showcase_commands=plan.showcase_commands)
        _send_server(server, f"tick rate {plan.game_tick_rate}")
        server_capture.wait_for(
            re.compile(rf"Set the target tick rate to {plan.game_tick_rate}(?:\.0+)? per second"),
            timeout=10.0,
            process=server,
        )

        username = plan.manual_client_name or USERNAME
        if plan.installed_runtime is None:
            client = _start(
                _client_command(plan),
                cwd=MOD_ROOT,
                environment=environment,
                stdin=subprocess.DEVNULL,
            )
            assert client.stdout is not None
            client_capture = LineCapture(
                cast(TextIO, client.stdout), plan.run_dir / "client-gradle.log", "client"
            )
            captures.append(client_capture)
            client_capture.start()
        else:
            print(
                f"Join 127.0.0.1:{plan.game_port} as {username} "
                "using your installed Fabric client. "
                "The client needs the same NeuroCraft Fly and Fabric API JARs. "
                "Leave the server or press Ctrl-C here to stop.",
                flush=True,
            )
        server_capture.wait_for(
            re.compile(rf"{re.escape(username)} joined the game"),
            timeout=client_join_timeout if client is None else startup_timeout,
            process=server if client is None else client,
        )
        _send_server(server, f"op {username}")
        # `neurocraftfly spawn` places the fly at the command source (FlyCommands.java:115),
        # and the launcher runs it as the observer. Minecraft will not spawn a player inside
        # the sealed arena, so it puts them on the glass roof at y=73 and the fly then
        # appeared at y=74 -- outside the cage, where it slid off and flew away. Put the
        # observer inside the arena first so the fly is spawned into the enclosure.
        _send_server(server, f"tp {username} {SPAWN_X} {SPAWN_Y} {SPAWN_Z} 0 0")
        _settle_client(
            *([client] if client is not None else []),
            server,
            service,
            **({"stop_requested": stop_requested} if control else {}),
        )
        _send_server(server, f"execute as {username} at @s run neurocraftfly spawn")
        server_capture.wait_for(
            re.compile(rf"Spawned one NeuroCraft fly with entity_id {re.escape(plan.entity_id)}"),
            timeout=15.0,
            process=server,
        )
        # The fly is spawned one block above the observer, which puts it inside their
        # camera. Step back along -Z and face +Z so it is visible from the start; the
        # fly is 0.38 x 0.22 blocks and is easy to lose otherwise.
        view = " ".join(map(str, _starting_view(plan.circuit_playground)))
        _send_server(server, f"tp {username} {view}")
        live_pattern = re.compile(
            rf"bridge=ACTIVE loaded_flies=1.*entity_id={re.escape(plan.entity_id)}"
            r".*command_seq=(?!none)"
        )
        deadline = time.monotonic() + 20.0
        while True:
            _send_server(server, "neurocraftfly status")
            try:
                server_capture.wait_for(live_pattern, timeout=1.0, process=server)
                break
            except LauncherError as exc:
                if time.monotonic() >= deadline:
                    raise LauncherError("fly did not produce a live circuit action") from exc
        _write_session(plan, state="interactive", identity=identity, catalog_sha256=catalog_sha256)
        print("\nNeuroCraft Fly is ready in Minecraft.", flush=True)
        print(f"This is the live {plan.fixture_description} circuit.")
        print(f"Requested server rate: {plan.game_tick_rate} game ticks per wall-clock second.")
        if plan.game_tick_rate != 20:
            print("HALF-SPEED DIAGNOSTIC: not a normal-speed performance benchmark.")
        if plan.viewer_port is not None:
            print(f"Brain viewer: http://127.0.0.1:{plan.viewer_port}/  (presentation only)")
        if plan.action_latency_path is not None and not plan.comparison_lanes:
            print(
                "Action latency samples: "
                f"{plan.action_latency_path.relative_to(ROOT)}  (diagnostic only)"
            )
        if plan.controller_profile in FULL_MCNS_PROGRAM_PROFILES:
            policy_match = service_capture.search(POLICY_HASH_PATTERN)
            if policy_match is None:
                raise LauncherError(
                    "the body-program lane did not report its policy_hash; refusing to "
                    "describe a selection policy this launcher cannot name"
                )
            print(PROGRAM_LANE_BANNER.format(policy_hash=policy_match.group(1)))
            if not _mod_admits_program_frames():
                print(PROGRAM_LANE_NOT_RUNNABLE)
            if plan.program_decisions_path is not None and not plan.comparison_lanes:
                print(
                    "Program decisions: "
                    f"{plan.program_decisions_path.relative_to(ROOT)}  (diagnostic only, "
                    "not content-addressed, not evidence)"
                )
        if plan.comparison_lanes:
            print("Comparison controls: /neurocraftfly control original|shuffled_weight|no_input")
            print(f"Fresh diagnostics for each accepted epoch: {plan.epoch_diagnostics_dir}")
            print(
                "Switches reset model state; world pose continues. Match trial geometry explicitly."
            )
        print("Place blocks near the fly and watch its movement and the F/L/V/Yaw/Pitch HUD.")
        print("Use /neurocraftfly status for diagnostics. Close Minecraft to stop cleanly.")
        print("Do not use RESET yet; relaunch if the bridge reports a terminal fault.\n")
        last_status_query = time.monotonic()
        console_cursor = 0
        console_enabled = plan.console_path is not None
        while client is None or client.poll() is None:
            _check_companion_stop(stop_requested)
            if control is not None:
                try:
                    condition = control.conditions.get_nowait()
                except queue.Empty:
                    pass
                else:
                    if plan.comparison_lanes:
                        _send_server(server, "neurocraftfly control " + condition)
                        print(f"[companion condition requested] {condition}", flush=True)
                    else:
                        print("[companion control refused] comparisons are disabled", flush=True)
            if client is None and server_capture.search(
                re.compile(rf"{re.escape(username)} left the game")
            ):
                exit_reason = "manual client left the server"
                return 0
            if service.poll() is not None:
                raise LauncherError("neural service exited during the interactive session")
            if server.poll() is not None:
                raise LauncherError("Minecraft server exited during the interactive session")
            fault = server_capture.search(TERMINAL_FAULT_PATTERN)
            if fault is not None:
                raise LauncherError(
                    f"bridge entered terminal fault {fault.group(1)}; start a fresh session"
                )
            now = time.monotonic()
            if console_enabled:
                assert plan.console_path is not None
                try:
                    console_cursor, requests = read_diagnostic_commands(
                        plan.console_path, console_cursor
                    )
                    for request in requests:
                        _send_server(server, request)
                        print(f"[diagnostic console requested] {request}", flush=True)
                except (OSError, ValueError) as exc:
                    console_enabled = False
                    print(f"diagnostic console disabled: {exc}", flush=True)
            if now - last_status_query >= MONITOR_STATUS_INTERVAL_SECONDS:
                last_status_query = now
                # Keep fault codes flowing into the capture so the watcher above sees
                # a latch within ~2 s. A failed send is not itself fatal: the server
                # may be mid-exit, and the poll() check above raises the clearer
                # error on the next iteration.
                if server.poll() is None:
                    with suppress(LauncherError, OSError, ValueError):
                        _send_server(server, "neurocraftfly status")
            time.sleep(0.25)
        assert client is not None
        exit_reason = f"client exited with status {client.returncode}"
        return int(client.returncode or 0)
    except CompanionStop as exc:
        exit_reason = str(exc)
        return 0
    except KeyboardInterrupt as exc:
        failed = True
        exit_reason = f"launcher interrupted: {exc}" if exc.args else "user interrupted launcher"
        return 130
    except Exception as exc:
        failed = True
        exit_reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if control is not None:
            control.stop_requested.set()
        cleanup_errors = _cleanup_children(
            {"client": client, "server": server, "viewer": viewer, "service": service}, captures
        )
        if cleanup_errors:
            exit_reason += "; cleanup errors: " + "; ".join(cleanup_errors)
        if token:
            exit_reason = exit_reason.replace(token, "[REDACTED]")
        _write_session(
            plan,
            state="cleanup_failed" if cleanup_errors else "stopped",
            identity=identity,
            exit_reason=exit_reason,
            catalog_sha256=catalog_sha256,
        )
        secret = token.encode()
        if secret:
            for path in plan.run_dir.rglob("*"):
                if path.is_file() and secret in path.read_bytes():
                    raise LauncherError(f"secret appeared in generated file: {path.name}")
        if cleanup_errors and not failed:
            raise LauncherError(exit_reason)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="create and print a redacted isolated run plan without starting Minecraft",
    )
    parser.add_argument("--startup-timeout", type=float, default=90.0)
    parser.add_argument("--client-join-timeout", type=float, default=600.0)
    parser.add_argument("--installed-runtime", type=Path, help="verified private Fabric runtime")
    parser.add_argument(
        "--manual-client-name", help="Java player name joining from a normal client"
    )
    parser.add_argument(
        "--showcase-world",
        action="store_true",
        help="build the authored four-station lab inside the fresh arena before spawning",
    )
    parser.add_argument(
        "--circuit-playground",
        action="store_true",
        help="opt in to close-up pads and a grounded starting camera; includes showcase world",
    )
    parser.add_argument(
        "--client-width", type=int, help="logical window width, 320..1920; requires --client-height"
    )
    parser.add_argument(
        "--client-height",
        type=int,
        help="logical window height, 240..1080; requires --client-width",
    )
    parser.add_argument(
        "--client-max-fps", type=int, default=60, help="requested client render ceiling, 20..120"
    )
    parser.add_argument(
        "--comparison-lanes",
        action="store_true",
        help="preload original, shuffled-weight, and zero-input controllers for lane switches",
    )
    parser.add_argument("--control-seed", type=int, default=7, help="unsigned 64-bit shuffle seed")
    parser.add_argument("--kernel-threads", type=int, choices=(1, 2, 4, 8))
    parser.add_argument(
        "--capture-fps", type=int, default=1, help="requested capture ceiling, 1..20"
    )
    parser.add_argument(
        "--capture-seconds", type=int, default=120, help="capture window, 1..720 seconds"
    )
    parser.add_argument(
        "--capture-png-compression",
        type=int,
        help="opt-in per-image PNG Deflater level 0..9; default uses Minecraft's writer",
    )
    parser.add_argument(
        "--diagnostic-tools",
        action="store_true",
        help="opt in to bounded client snapshots and a local Minecraft console-command inbox",
    )
    parser.add_argument(
        "--game-tick-rate",
        type=int,
        choices=(10, 20),
        default=20,
        help="20 for normal playback; 10 for explicitly labeled half-speed diagnostics",
    )
    parser.add_argument(
        "--controller-profile",
        choices=sorted(CONTROLLER_PROFILES),
        default=DEFAULT_CONTROLLER_PROFILE,
        help="which pinned service controller profile drives the fly",
    )
    parser.add_argument(
        "--brain-viewer",
        action="store_true",
        help="also serve the presentation-only live brain viewer on loopback",
    )
    parser.add_argument(
        "--action-latency",
        action="store_true",
        help=(
            "have the service append latency-only NDJSON samples to "
            "action-latency.ndjson in the run directory; accepted for every profile"
        ),
    )
    parser.add_argument(
        "--control-stdin",
        action="store_true",
        help="accept bounded companion JSON stop/condition requests on stdin (EOF stops)",
    )
    return parser


def _sigterm_to_keyboard_interrupt(signum: int, frame: FrameType | None) -> None:
    """Route supervisor SIGTERM and terminal-close SIGHUP through bounded cleanup."""

    raise KeyboardInterrupt(f"received {signal.Signals(signum).name}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for signum in _shutdown_signals():
        if signum != signal.SIGINT:
            signal.signal(signum, _sigterm_to_keyboard_interrupt)
    plan = prepare_run(
        controller_profile=args.controller_profile,
        brain_viewer=args.brain_viewer,
        action_latency=args.action_latency,
        game_tick_rate=args.game_tick_rate,
        diagnostic_tools=args.diagnostic_tools,
        comparison_lanes=args.comparison_lanes,
        control_seed=args.control_seed,
        kernel_threads=args.kernel_threads,
        capture_fps=args.capture_fps,
        capture_seconds=args.capture_seconds,
        capture_png_compression=args.capture_png_compression,
        client_width=args.client_width,
        client_height=args.client_height,
        client_max_fps=args.client_max_fps,
        showcase_world=args.showcase_world,
        circuit_playground=args.circuit_playground,
        installed_runtime=args.installed_runtime,
        manual_client_name=args.manual_client_name,
    )
    print(f"Run directory: {plan.run_dir.relative_to(ROOT)}")
    print(f"Entity UUID: {plan.entity_id}")
    print(f"Controller profile: {plan.controller_profile}")
    print(
        "Mode: loopback-only; "
        + (
            "installed mod server, manually launched client (local offline authentication)"
            if plan.installed_runtime is not None
            else "development client (offline authentication)"
        )
    )
    if args.prepare_only:
        return 0
    return run_interactive(
        plan,
        startup_timeout=args.startup_timeout,
        client_join_timeout=args.client_join_timeout,
        control_stdin=args.control_stdin,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LauncherError as exc:
        print(f"interactive launcher failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
