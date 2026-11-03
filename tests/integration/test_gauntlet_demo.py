"""The gauntlet launcher composes isolated sessions; it never blends identities."""

from __future__ import annotations

import io
import signal
import sys
from pathlib import Path
from typing import TextIO, cast

import pytest

from scripts import run_gauntlet_demo as gauntlet
from scripts import run_interactive_demo as launcher

ROOT = Path(__file__).resolve().parents[2]


class _StubProcess:
    """Just enough of Popen for SessionCapture readiness checks."""

    def __init__(self, returncode: int | None = None) -> None:
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode


def test_default_profiles_are_the_three_distinct_brains() -> None:
    assert gauntlet.DEFAULT_PROFILES == (
        "pair_v1",
        "wing_motor_soma_side_v1",
        "trm_cloned_readout_v1",
    )
    assert set(gauntlet.DEFAULT_PROFILES) <= set(launcher.CONTROLLER_PROFILES)


def test_duplicate_and_unknown_profiles_are_refused() -> None:
    with pytest.raises(gauntlet.GauntletError, match="distinct identity"):
        gauntlet.validate_profiles(("pair_v1", "pair_v1"))
    with pytest.raises(gauntlet.GauntletError, match="unknown controller profile"):
        gauntlet.validate_profiles(("pair_v1", "scripted_baseline"))
    with pytest.raises(gauntlet.GauntletError, match="at least one"):
        gauntlet.validate_profiles(())


def test_session_commands_run_the_unchanged_launcher_per_profile() -> None:
    command = gauntlet.build_session_command("trm_cloned_readout_v1", brain_viewer=False)
    assert command[:2] == [sys.executable, "-I"]
    assert command[2] == str(ROOT / "scripts" / "run_interactive_demo.py")
    profile_index = command.index("--controller-profile")
    assert command[profile_index + 1] == "trm_cloned_readout_v1"
    assert "--brain-viewer" not in command


def test_brain_viewer_attaches_only_to_steering_fixture_lanes() -> None:
    with_viewer = gauntlet.build_session_command("trm_cloned_readout_v1", brain_viewer=True)
    assert "--brain-viewer" in with_viewer
    wing = gauntlet.build_session_command("wing_motor_soma_side_v1", brain_viewer=True)
    assert "--brain-viewer" in wing
    pair = gauntlet.build_session_command("pair_v1", brain_viewer=True)
    assert "--brain-viewer" not in pair


def test_ready_line_stays_in_lockstep_with_the_session_launcher() -> None:
    """The gauntlet staggers sessions on the launcher's printed readiness line;
    if that literal drifts, the gauntlet would hang, so pin the coupling."""

    source = (ROOT / "scripts" / "run_interactive_demo.py").read_text(encoding="utf-8")
    assert gauntlet.READY_LINE in source
    assert gauntlet.READY_PATTERN.search(f"prefix {gauntlet.READY_LINE}") is not None


def test_session_argv_flags_stay_in_lockstep_with_the_session_launcher() -> None:
    parsed = launcher.build_parser().parse_args(
        ["--controller-profile", "trm_cloned_readout_v1", "--brain-viewer"]
    )
    assert parsed.controller_profile == "trm_cloned_readout_v1"
    assert parsed.brain_viewer is True


def test_readiness_survives_arbitrarily_long_startup_logs(tmp_path: Path) -> None:
    """The readiness line must be seen even after thousands of log lines —
    a regression guard for buffer-pruning ever re-entering the watch path."""

    noise = "".join(f"noise line {index}\n" for index in range(4500))
    stream = io.StringIO(noise + gauntlet.READY_LINE + "\n")
    capture = gauntlet.SessionCapture(
        cast(TextIO, stream), tmp_path / "lane.log", "test", gauntlet.READY_PATTERN
    )
    capture.start()
    capture.wait_until_matched(
        timeout=10.0, process=cast("gauntlet.subprocess.Popen[str]", _StubProcess())
    )
    capture.join()
    assert (tmp_path / "lane.log").read_text(encoding="utf-8").endswith(gauntlet.READY_LINE + "\n")


def test_readiness_failure_reports_a_dead_session(tmp_path: Path) -> None:
    stream = io.StringIO("no readiness here\n")
    capture = gauntlet.SessionCapture(
        cast(TextIO, stream), tmp_path / "dead.log", "test", gauntlet.READY_PATTERN
    )
    capture.start()
    with pytest.raises(gauntlet.GauntletError, match="exited with status 7"):
        capture.wait_until_matched(
            timeout=5.0, process=cast("gauntlet.subprocess.Popen[str]", _StubProcess(7))
        )


def test_sigterm_reaches_the_session_launchers_cleanup_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gauntlet escalates SIGINT to SIGTERM; the session launcher must
    convert SIGTERM into the same bounded KeyboardInterrupt cleanup."""

    with pytest.raises(KeyboardInterrupt):
        launcher._sigterm_to_keyboard_interrupt(15, None)
    registered: dict[int, object] = {}
    monkeypatch.setattr(
        launcher.signal, "signal", lambda signum, handler: registered.setdefault(signum, handler)
    )

    class PreparationReached(Exception):
        pass

    def stop_before_preparation(**kwargs: object) -> None:
        raise PreparationReached

    monkeypatch.setattr(launcher, "prepare_run", stop_before_preparation)
    with pytest.raises(PreparationReached):
        launcher.main([])
    assert registered[signal.SIGTERM] is launcher._sigterm_to_keyboard_interrupt
