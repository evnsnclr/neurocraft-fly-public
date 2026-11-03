from __future__ import annotations

import io
import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import process_lifecycle as ownership
from scripts import run_interactive_demo as launcher


def test_windows_child_is_suspended_and_owned_before_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    options: dict[str, Any] = {}
    process = SimpleNamespace()
    job = SimpleNamespace(
        assign=lambda child: events.append("assign"),
        close=lambda: events.append("close"),
    )
    monkeypatch.setattr(ownership, "is_windows", lambda: True)
    monkeypatch.setattr(ownership, "_WindowsJob", lambda: job)

    def popen(argv: list[str], **kwargs: Any) -> Any:
        events.append("create")
        options.update(kwargs)
        assert argv == ["python.exe", "a path with spaces.py"]
        return process

    def resume(child: Any, owned_job: Any) -> None:
        events.append("resume")
        assert child._neurocraft_job is owned_job

    monkeypatch.setattr(ownership.subprocess, "Popen", popen)
    monkeypatch.setattr(ownership, "_resume_suspended_process", resume)
    assert (
        ownership.start_windows(
            ["python.exe", "a path with spaces.py"],
            cwd=Path.cwd(),
            environment={"SystemRoot": "C:/W"},
        )
        is process
    )
    assert events == ["create", "assign", "resume"]
    assert options["creationflags"] & ownership.CREATE_SUSPENDED
    assert options["creationflags"] & ownership.CREATE_NEW_PROCESS_GROUP
    assert options["creationflags"] & ownership.CREATE_NO_WINDOW
    assert options["shell"] is False
    assert options["env"] == {"SystemRoot": "C:/W"}
    assert options["encoding"] == "utf-8"


def test_failed_job_assignment_never_resumes_or_searches_other_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def assign(child: Any) -> None:
        events.append("assign")
        raise OSError("job assignment refused")

    process = SimpleNamespace(
        kill=lambda: events.append("kill suspended child"),
        wait=lambda **kwargs: events.append("wait"),
    )
    job = SimpleNamespace(assign=assign, close=lambda: events.append("close"))
    monkeypatch.setattr(ownership, "is_windows", lambda: True)
    monkeypatch.setattr(ownership, "_WindowsJob", lambda: job)
    monkeypatch.setattr(ownership.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        ownership, "_resume_suspended_process", lambda *args: events.append("resume")
    )
    with pytest.raises(OSError, match="assignment refused"):
        ownership.start_windows(["python.exe"], cwd=Path.cwd(), environment={})
    assert events == ["assign", "kill suspended child", "wait", "close"]


def test_windows_cleanup_reaches_job_after_wrapper_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    counts = iter((1, 0))
    job = SimpleNamespace(
        active_processes=lambda: next(counts),
        terminate=lambda: events.append("terminate owned job"),
        close=lambda: events.append("close"),
    )
    process = SimpleNamespace(
        poll=lambda: 0,
        _neurocraft_job=job,
        _neurocraft_job_state=ownership._OwnedWindowsJob(job),
        wait=lambda **kwargs: events.append("reap"),
    )
    ownership.terminate_windows(process)
    assert events == ["terminate owned job", "reap", "close"]
    assert process._neurocraft_job_state.closed
    ownership.terminate_windows(process)
    assert events == ["terminate owned job", "reap", "close"]


def test_windows_cleanup_refuses_an_unowned_child() -> None:
    with pytest.raises(OSError, match="without its owned job handle"):
        ownership.terminate_windows(SimpleNamespace(pid=os.getpid()))  # type: ignore[arg-type]


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows Job Object APIs")
def test_native_windows_job_kills_orphan_descendant_and_leaves_unrelated_process() -> None:
    """Run this without skipping in Windows CI; POSIX mocks cannot prove this API."""
    unrelated = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(30)"])
    child_code = "import os,time; print(os.getpid(),flush=True); time.sleep(30)"
    wrapper_code = (
        f"import subprocess,sys; subprocess.Popen([sys.executable,'-I','-u','-c',{child_code!r}])"
    )
    wrapper = ownership.start_windows(
        [sys.executable, "-I", "-u", "-c", wrapper_code],
        cwd=Path.cwd(),
        environment=launcher._child_environment("local-test-token"),
        stdin=subprocess.DEVNULL,
    )
    assert wrapper.stdout is not None
    lines: queue.Queue[str] = queue.Queue()
    reader = threading.Thread(target=lambda: lines.put(wrapper.stdout.readline()), daemon=True)
    reader.start()
    try:
        assert wrapper.wait(timeout=10) == 0
        child_pid = int(lines.get(timeout=10))
        assert child_pid != wrapper.pid and child_pid != unrelated.pid
        assert wrapper._neurocraft_job.active_processes() >= 1  # type: ignore[attr-defined]
        ownership.terminate_windows(wrapper, timeout=5)
        reader.join(timeout=1)
        assert wrapper.stdout.read() == "", "descendant still holds the inherited pipe"
        assert unrelated.poll() is None, "cleanup reached an unrelated process"
    finally:
        if hasattr(wrapper, "_neurocraft_job"):
            ownership.terminate_windows(wrapper, timeout=5)
        unrelated.kill()
        unrelated.wait(timeout=5)
        wrapper.stdout.close()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"action":"condition","condition":"original"}\n', ["original"]),
        ('{"action":"condition","condition":"shuffled_weight"}\n', ["shuffled_weight"]),
        ('{"action":"condition","condition":"no_input"}\n', ["no_input"]),
        ('{"action":"stop"}\n', []),
        ('{"action":"console","command":"secret-value"}\n', []),
        ('{"action":"condition","condition":"no_input\\nstop"}\n', []),
        ('{"action":"condition","condition":"shuffled_weights"}\n', []),
        ('{"action":"condition","condition":"original","extra":true}\n', []),
        ('{"action":"stop"}', []),
        ("secret-value" * 1000 + "\n", []),
        ("[]\n", []),
    ],
)
def test_companion_control_is_bounded_and_never_echoes_input(
    raw: str, expected: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    control = launcher.StdinControl(io.StringIO(raw))
    control.start()
    control.thread.join(timeout=2)
    assert not control.thread.is_alive()
    # Stop and EOF both cleanly end an unattended session.
    assert control.stop_requested.is_set()
    actual = []
    while not control.conditions.empty():
        actual.append(control.conditions.get_nowait())
    assert actual == expected
    assert "secret-value" not in capsys.readouterr().out


def test_companion_stop_interrupts_startup_waits_without_os_signals(tmp_path: Path) -> None:
    stop = threading.Event()
    stop.set()
    process = SimpleNamespace(poll=lambda: None)
    with pytest.raises(launcher.CompanionStop):
        launcher._wait_tcp(process, 1, 10, stop)  # type: ignore[arg-type]
    with pytest.raises(launcher.CompanionStop):
        launcher._settle_client(process, stop_requested=stop)  # type: ignore[arg-type]
    capture = launcher.LineCapture(io.StringIO(), tmp_path / "not-created.log", "test")
    capture.stop_requested = stop
    with pytest.raises(launcher.CompanionStop):
        capture.wait_for(re.compile("never"), timeout=10)
    assert not (tmp_path / "not-created.log").exists()


def test_companion_control_open_pipe_does_not_block_or_abort_interpreter_exit() -> None:
    code = (
        "import sys,time; from scripts.run_interactive_demo import StdinControl; "
        "control=StdinControl(sys.stdin); control.start(); time.sleep(0.1); "
        "control.stop_requested.set()"
    )
    process = subprocess.Popen(
        [sys.executable, "-B", "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert process.wait(timeout=5) == 0
        assert process.stderr is not None
        assert b"_enter_buffered_busy" not in process.stderr.read()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()


def test_concurrent_windows_cleanup_closes_one_owned_job_once() -> None:
    events: list[str] = []
    active = [True]

    def terminate() -> None:
        events.append("terminate")
        active[0] = False

    job = SimpleNamespace(
        active_processes=lambda: int(active[0]),
        terminate=terminate,
        close=lambda: events.append("close"),
    )
    process = SimpleNamespace(
        _neurocraft_job=job,
        _neurocraft_job_state=ownership._OwnedWindowsJob(job),
        wait=lambda **kwargs: events.append("reap"),
    )
    errors: list[BaseException] = []

    def cleanup() -> None:
        try:
            ownership.terminate_windows(process)
        except BaseException as exc:
            errors.append(exc)

    callers = [threading.Thread(target=cleanup) for _ in range(8)]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(timeout=2)
        assert not caller.is_alive()
    assert errors == []
    assert events == ["terminate", "reap", "close"]


def test_repeated_windows_cleanup_retains_first_failure() -> None:
    failure = OSError("synthetic job query failure")
    closes: list[bool] = []

    def refuse() -> int:
        raise failure

    job = SimpleNamespace(active_processes=refuse, close=lambda: closes.append(True))
    process = SimpleNamespace(
        _neurocraft_job=job,
        _neurocraft_job_state=ownership._OwnedWindowsJob(job),
    )
    with pytest.raises(OSError) as caught:
        ownership.terminate_windows(process)
    assert caught.value is failure
    with pytest.raises(OSError, match="previous cleanup") as repeated:
        ownership.terminate_windows(process)
    assert repeated.value.__cause__ is failure
    assert closes == [True]
