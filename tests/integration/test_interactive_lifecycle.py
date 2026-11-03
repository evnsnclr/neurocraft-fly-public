from __future__ import annotations

import io
import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_interactive_demo as launcher


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group/select ownership proof")
@pytest.mark.parametrize("ignore_term", [False, True])
def test_cleanup_reaches_a_descendant_after_its_wrapper_exits(ignore_term: bool) -> None:
    """An exited wrapper must not hide a live child, including one ignoring TERM."""
    child_code = (
        "import os, signal, time\n"
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "")
        + "print(os.getpid(), flush=True)\n"
        + "while True: time.sleep(1)\n"
    )
    wrapper_code = (
        f"import subprocess, sys\nsubprocess.Popen([sys.executable, '-u', '-c', {child_code!r}])\n"
    )
    wrapper = launcher._start(
        [sys.executable, "-u", "-c", wrapper_code],
        cwd=Path.cwd(),
        environment=os.environ,
        stdin=subprocess.DEVNULL,
    )
    assert wrapper.stdout is not None
    try:
        assert wrapper.wait(timeout=5.0) == 0
        assert select.select([wrapper.stdout], [], [], 5.0)[0], "descendant never became ready"
        child_pid = int(wrapper.stdout.readline())
        os.kill(child_pid, 0)  # Prove the fixture really survived its wrapper.
        started = time.monotonic()
        launcher._terminate(wrapper, timeout=0.5)
        assert time.monotonic() - started < 2.0, "cleanup exceeded its bounded escalation"
        assert select.select([wrapper.stdout], [], [], 2.0)[0], "child still holds the output pipe"
        assert wrapper.stdout.read(1) == "", "child output pipe never closed"
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(wrapper.pid, signal.SIGKILL)
        wrapper.wait(timeout=5.0)
        wrapper.stdout.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group permission semantics")
def test_group_probe_retries_transient_permission_error(monkeypatch: pytest.MonkeyPatch) -> None:
    probes = []

    def killpg(pid: int, signum: int) -> None:
        if signum == 0:
            probes.append(pid)
            if len(probes) == 1:
                raise PermissionError("exiting orphan group")
            raise ProcessLookupError

    monkeypatch.setattr(launcher.os, "killpg", killpg)
    launcher._terminate(SimpleNamespace(pid=123, poll=lambda: 0), timeout=0.2)
    assert probes == [123, 123]


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group permission semantics")
def test_persistent_group_permission_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    signals = []

    def killpg(pid: int, signum: int) -> None:
        signals.append(signum)
        raise PermissionError("group cannot be signalled")

    monkeypatch.setattr(launcher.os, "killpg", killpg)
    with pytest.raises(launcher.LauncherError, match="still exists after SIGKILL") as caught:
        launcher._terminate(SimpleNamespace(pid=123, poll=lambda: 0), timeout=0.0)
    assert signals == [signal.SIGTERM, 0, signal.SIGKILL, 0]
    assert isinstance(caught.value.__cause__, PermissionError)


@pytest.fixture
def stopped_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[launcher.RunPlan, list[object]]:
    """Exercise run_interactive's actual failure/finally path without a service or game."""
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    eula = tmp_path / "accepted-eula.txt"
    eula.write_text("eula=true\n", encoding="utf-8")
    plan = launcher.prepare_run(run_root=tmp_path / "runs", accepted_eula=eula)
    monkeypatch.setattr(launcher, "_tcp_ready", lambda port: False)
    monkeypatch.setattr(
        launcher.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0)
    )
    monkeypatch.setattr(launcher, "_load_identity", lambda *args: {"run_id": plan.run_id})
    monkeypatch.setattr(launcher, "_child_environment", lambda token: {})
    monkeypatch.setattr(launcher.secrets, "token_urlsafe", lambda length: "test-only-secret")
    process = SimpleNamespace(stdout=io.StringIO())
    monkeypatch.setattr(launcher, "_start", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        launcher,
        "LineCapture",
        lambda *args: SimpleNamespace(start=lambda: None, join=lambda: None),
    )
    terminated: list[object] = []
    monkeypatch.setattr(launcher, "_terminate", lambda child: terminated.append(child))
    return plan, terminated


@pytest.mark.parametrize(
    "failure",
    [
        launcher.LauncherError("bridge entered terminal fault DEADLINE_MISSED"),
        BrokenPipeError("Minecraft console closed"),
        RuntimeError("unexpected startup failure with test-only-secret"),
    ],
)
def test_failure_records_real_reason_and_still_cleans_up(
    stopped_startup: tuple[launcher.RunPlan, list[object]],
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    plan, terminated = stopped_startup

    def fail(*args: object) -> None:
        raise failure

    monkeypatch.setattr(launcher, "_wait_tcp", fail)
    with pytest.raises(type(failure)) as caught:
        launcher.run_interactive(plan)
    assert caught.value is failure
    session = json.loads(plan.session_path.read_text(encoding="utf-8"))
    assert session["state"] == "stopped"
    assert session["exit_reason"] == (
        f"{type(failure).__name__}: {failure}".replace("test-only-secret", "[REDACTED]")
    )
    assert "test-only-secret" not in plan.session_path.read_text(encoding="utf-8")
    assert len([child for child in terminated if child is not None]) == 1


@pytest.mark.parametrize("signum", [s for s in launcher._shutdown_signals() if s != signal.SIGINT])
def test_shutdown_signal_records_its_name_and_cleans_up(
    stopped_startup: tuple[launcher.RunPlan, list[object]],
    monkeypatch: pytest.MonkeyPatch,
    signum: int,
) -> None:
    plan, terminated = stopped_startup

    def interrupt(*args: object) -> None:
        launcher._sigterm_to_keyboard_interrupt(signum, None)

    monkeypatch.setattr(launcher, "_wait_tcp", interrupt)
    assert launcher.run_interactive(plan) == 130
    session = json.loads(plan.session_path.read_text(encoding="utf-8"))
    assert session["exit_reason"] == f"launcher interrupted: received {signal.Signals(signum).name}"
    assert len([child for child in terminated if child is not None]) == 1


def test_cleanup_continues_after_broken_console_and_failed_termination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    children = {
        name: SimpleNamespace(poll=lambda: None) for name in ("client", "server", "service")
    }
    attempted = []
    joined = []

    def terminate(process: object) -> None:
        attempted.append(process)
        if process is children["client"]:
            raise launcher.LauncherError("stubborn process group")

    def broken_console(*args: object) -> None:
        raise BrokenPipeError("closed console")

    monkeypatch.setattr(launcher, "_terminate", terminate)
    monkeypatch.setattr(launcher, "_send_server", broken_console)
    handlers = {sig: signal.getsignal(sig) for sig in launcher._shutdown_signals()}
    errors = launcher._cleanup_children(
        children, [SimpleNamespace(join=lambda: joined.append(True))]
    )
    assert attempted == list(children.values())
    assert joined == [True]
    assert len(errors) == 2
    assert "stubborn process group" in errors[0]
    assert "BrokenPipeError: closed console" in errors[1]
    assert {sig: signal.getsignal(sig) for sig in handlers} == handlers


def test_preflight_failure_is_recorded(
    stopped_startup: tuple[launcher.RunPlan, list[object]], monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _ = stopped_startup
    monkeypatch.setattr(launcher, "_tcp_ready", lambda port: True)
    with pytest.raises(launcher.LauncherError, match="acquired a listener"):
        launcher.run_interactive(plan)
    session = json.loads(plan.session_path.read_text(encoding="utf-8"))
    assert "acquired a listener" in session["exit_reason"]
    assert "identity" not in session


def test_missing_comparison_catalog_stops_service_before_starting_minecraft(
    stopped_startup: tuple[launcher.RunPlan, list[object]], monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, terminated = stopped_startup
    plan = replace(
        plan,
        comparison_lanes=True,
        controller_profile=launcher.FULL_MCNS_LOOM_PROGRAM_V2_PROFILE,
    )
    started: list[object] = []

    def start(*args: object, **kwargs: object) -> object:
        process = SimpleNamespace(stdout=io.StringIO())
        started.append(process)
        return process

    monkeypatch.setattr(launcher, "_start", start)
    monkeypatch.setattr(launcher, "_wait_tcp", lambda *args: None)
    with pytest.raises(launcher.LauncherError, match="comparison catalog refused"):
        launcher.run_interactive(plan)
    assert len(started) == 1, "server/client must not start without the catalog"
    assert [child for child in terminated if child is not None] == started
    metadata = json.loads(plan.session_path.read_text())
    assert "comparison catalog refused" in metadata["exit_reason"]
    assert metadata["lane_catalog_sha256"] is None


def test_cleanup_failure_does_not_replace_the_original_fault(
    stopped_startup: tuple[launcher.RunPlan, list[object]], monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _ = stopped_startup
    failure = launcher.LauncherError("bridge entered terminal fault DEADLINE_MISSED")
    attempted = []

    def fail(*args: object) -> None:
        raise failure

    def bad_termination(child: object) -> None:
        attempted.append(child)
        raise OSError("could not stop group")

    monkeypatch.setattr(launcher, "_wait_tcp", fail)
    monkeypatch.setattr(launcher, "_terminate", bad_termination)
    with pytest.raises(launcher.LauncherError) as caught:
        launcher.run_interactive(plan)
    assert caught.value is failure
    session = json.loads(plan.session_path.read_text(encoding="utf-8"))
    assert session["state"] == "cleanup_failed"
    assert session["exit_reason"].startswith(f"LauncherError: {failure}; cleanup errors:")
    assert "OSError: could not stop group" in session["exit_reason"]
    assert len(attempted) == 4


def test_main_registers_terminal_close_handler(
    stopped_startup: tuple[launcher.RunPlan, list[object]], monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _ = stopped_startup
    registered = {}
    monkeypatch.setattr(launcher, "prepare_run", lambda **kwargs: plan)
    monkeypatch.setattr(signal, "signal", lambda sig, handler: registered.update({sig: handler}))
    assert launcher.main(["--prepare-only"]) == 0
    assert registered[signal.SIGTERM] is launcher._sigterm_to_keyboard_interrupt
    for signum in launcher._shutdown_signals():
        if signum != signal.SIGINT:
            assert registered[signum] is launcher._sigterm_to_keyboard_interrupt


def test_companion_stop_before_start_records_clean_exit_without_children(
    stopped_startup: tuple[launcher.RunPlan, list[object]], monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, terminated = stopped_startup
    stop = threading.Event()
    stop.set()
    control = SimpleNamespace(stop_requested=stop, start=lambda: None)
    monkeypatch.setattr(launcher, "StdinControl", lambda stream: control)
    assert launcher.run_interactive(plan, control_stdin=True) == 0
    assert all(child is None for child in terminated)
    session = json.loads(plan.session_path.read_text())
    assert session["state"] == "stopped"
    assert "companion requested stop" in session["exit_reason"]
