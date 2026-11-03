#!/usr/bin/env python3
"""Launch the three-brains gauntlet: isolated interactive sessions side by side.

Each session is one unchanged ``run_interactive_demo.py`` child — its own
world, service, bridge port, run identity, and token. Nothing is shared and
nothing in the mod or protocol changes shape: the comparison is three fully
isolated causal chains running concurrently, which keeps every session's
identity claims exactly as strong as a solo run. Startup is staggered so the
Gradle-heavy launch phases do not contend.
"""

from __future__ import annotations

import argparse
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO, cast

# ``python scripts/run_gauntlet_demo.py`` places ``scripts/`` rather than the
# repository root on sys.path. Add only this script's resolved repository root
# so the launcher module has the same import identity under CLI and test use.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_interactive_demo import CONTROLLER_PROFILES, STEERING_FIXTURE

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_interactive_demo.py"
GAUNTLET_ROOT = ROOT / "artifacts" / "live-interactive"

DEFAULT_PROFILES = ("pair_v1", "wing_motor_soma_side_v1", "trm_cloned_readout_v1")
#: The exact readiness line run_interactive_demo.py prints; a lockstep test
#: pins this literal against that script's source.
READY_LINE = "NeuroCraft Fly is ready in Minecraft."
READY_PATTERN = re.compile(re.escape(READY_LINE))
DEFAULT_SESSION_TIMEOUT = 240.0
SHUTDOWN_GRACE_SECONDS = 75.0


class GauntletError(RuntimeError):
    """The gauntlet could not be launched safely."""


def build_session_command(profile: str, *, brain_viewer: bool) -> list[str]:
    if profile not in CONTROLLER_PROFILES:
        raise GauntletError(f"unknown controller profile: {profile}")
    command = [
        sys.executable,
        "-I",
        str(LAUNCHER),
        "--controller-profile",
        profile,
    ]
    if brain_viewer and CONTROLLER_PROFILES[profile][0] == STEERING_FIXTURE:
        command.append("--brain-viewer")
    return command


def validate_profiles(profiles: Sequence[str]) -> tuple[str, ...]:
    if not profiles:
        raise GauntletError("the gauntlet needs at least one controller profile")
    if len(set(profiles)) != len(profiles):
        raise GauntletError(
            "duplicate controller profiles are refused; each gauntlet lane "
            "must carry a distinct identity"
        )
    for profile in profiles:
        if profile not in CONTROLLER_PROFILES:
            raise GauntletError(f"unknown controller profile: {profile}")
    return tuple(profiles)


class SessionCapture:
    """Tee one child launcher's console to a log with a profile prefix.

    The watch pattern is matched on each line as it arrives, so readiness
    detection cannot depend on any bounded replay buffer — a session that
    logs arbitrarily much before its readiness line is still seen as ready.
    """

    def __init__(self, stream: TextIO, path: Path, prefix: str, watch: re.Pattern[str]) -> None:
        self._stream = stream
        self._path = path
        self._prefix = prefix
        self._watch = watch
        self._matched = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self, *, timeout: float = 2.0) -> None:
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        with self._path.open("x", encoding="utf-8") as log:
            for line in self._stream:
                log.write(line)
                log.flush()
                print(f"[{self._prefix}] {line}", end="", flush=True)
                if self._watch.search(line) is not None:
                    self._matched.set()

    def wait_until_matched(self, *, timeout: float, process: subprocess.Popen[str]) -> None:
        deadline = time.monotonic() + timeout
        while True:
            if self._matched.wait(timeout=0.25):
                return
            if process.poll() is not None:
                raise GauntletError(
                    f"session {self._prefix} exited with status "
                    f"{process.returncode} before becoming ready"
                )
            if time.monotonic() >= deadline:
                raise GauntletError(f"session {self._prefix} did not become ready in time")


def _interrupt_session(process: subprocess.Popen[str]) -> None:
    """Ask a child launcher to run its own bounded cleanup path.

    SIGINT and SIGTERM both reach the child's KeyboardInterrupt cleanup (the
    session launcher converts SIGTERM); SIGKILL is a last resort that skips
    that cleanup and can orphan the session's own server/client/service
    children, so it is loudly disclosed when reached.
    """

    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=SHUTDOWN_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=SHUTDOWN_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            print(
                "[gauntlet] a session ignored SIGINT and SIGTERM; sending SIGKILL — "
                "its Minecraft/service processes may be orphaned, check the run "
                f"directories under {GAUNTLET_ROOT}",
                file=sys.stderr,
            )
            process.kill()
            process.wait(timeout=10.0)


def run_gauntlet(
    profiles: Sequence[str],
    *,
    brain_viewer: bool,
    session_timeout: float,
) -> int:
    validated = validate_profiles(profiles)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_dir = GAUNTLET_ROOT / f"gauntlet-{stamp}"
    log_dir.mkdir(parents=True, exist_ok=False)
    sessions: list[tuple[str, subprocess.Popen[str], SessionCapture]] = []
    try:
        for profile in validated:
            if brain_viewer and CONTROLLER_PROFILES[profile][0] != STEERING_FIXTURE:
                print(f"[{profile}] brain viewer skipped: not a steering-fixture profile")
            process = subprocess.Popen(
                build_session_command(profile, brain_viewer=brain_viewer),
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            assert process.stdout is not None
            capture = SessionCapture(
                cast(TextIO, process.stdout), log_dir / f"{profile}.log", profile, READY_PATTERN
            )
            capture.start()
            sessions.append((profile, process, capture))
            capture.wait_until_matched(timeout=session_timeout, process=process)
            print(f"\n=== gauntlet lane ready: {profile} ===\n", flush=True)

        print(f"\nAll {len(sessions)} gauntlet lanes are live. Close any client to end.\n")
        while all(process.poll() is None for _, process, _ in sessions):
            time.sleep(0.5)
        # Codes are read at the moment the first lane closes: that lane's code
        # wins, and lanes still running (shut down in the finally) count as 0.
        return max(int(process.poll() or 0) for _, process, _ in sessions)
    except KeyboardInterrupt:
        return 130
    finally:
        for _, process, _ in sessions:
            _interrupt_session(process)
        for _, _, capture in sessions:
            capture.join()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(DEFAULT_PROFILES),
        help="distinct controller profiles, one isolated session each",
    )
    parser.add_argument(
        "--brain-viewer",
        action="store_true",
        help="attach the presentation-only brain viewer to steering-fixture lanes",
    )
    parser.add_argument("--session-timeout", type=float, default=DEFAULT_SESSION_TIMEOUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_gauntlet(
        args.profiles,
        brain_viewer=args.brain_viewer,
        session_timeout=args.session_timeout,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GauntletError as exc:
        print(f"gauntlet launcher failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
