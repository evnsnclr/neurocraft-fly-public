"""Own child trees with POSIX groups or Windows jobs; never kill by image name.

A child starts suspended, joins a private kill-on-close job, and only then runs.
An abrupt owner exit in the small creation-to-assignment window may leave a
suspended, nonrunning child; ordinary assignment errors kill it before resuming.
Normal protocol/console shutdown belongs to the caller; termination is the bounded
fallback. Windows 10/11 support nested jobs, including a companion-owned outer job.
https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
"""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from ctypes import wintypes
from pathlib import Path
from typing import Any, cast

CREATE_SUSPENDED = 0x00000004
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000


def is_windows() -> bool:
    return os.name == "nt"


def terminate_posix(process: subprocess.Popen[Any], *, timeout: float = 8.0) -> None:
    """Stop a child created with start_new_session, including surviving descendants.

    A wrapper's exit does not end its group. Only pass a process whose session
    this caller created; a descendant deliberately creating a new session escapes
    this POSIX boundary. Windows callers must retain their Job Object instead.
    """
    if is_windows():
        raise OSError("POSIX process ownership requested on Windows")
    signal_error: PermissionError | None = None
    for stop_signal in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, stop_signal)
        except ProcessLookupError:
            process.poll()
            return
        except PermissionError as exc:
            signal_error = exc
        deadline = time.monotonic() + timeout
        while True:
            process.poll()  # Reap the wrapper so its zombie cannot keep the group alive.
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return
            except PermissionError:
                # macOS can briefly report EPERM while an orphaned group exits.
                # Keep waiting; a persistent permission failure is never success.
                pass
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.05, remaining))
    raise OSError(f"child process group {process.pid} still exists after SIGKILL") from signal_error


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint64)
        for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )
    ]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _BasicAccountingInformation(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_int64),
        ("TotalKernelTime", ctypes.c_int64),
        ("ThisPeriodTotalUserTime", ctypes.c_int64),
        ("ThisPeriodTotalKernelTime", ctypes.c_int64),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    ]


class _ThreadEntry(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ThreadID", wintypes.DWORD),
        ("th32OwnerProcessID", wintypes.DWORD),
        ("tpBasePri", wintypes.LONG),
        ("tpDeltaPri", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
    ]


def _kernel32() -> Any:
    # Resolve only on Windows. Explicit prototypes preserve 64-bit HANDLE values.
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    signatures = {
        "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
        "SetInformationJobObject": (
            [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD],
            wintypes.BOOL,
        ),
        "QueryInformationJobObject": (
            [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p],
            wintypes.BOOL,
        ),
        "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
        "TerminateJobObject": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
        "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
        "CreateToolhelp32Snapshot": ([wintypes.DWORD, wintypes.DWORD], wintypes.HANDLE),
        "Thread32First": ([wintypes.HANDLE, ctypes.POINTER(_ThreadEntry)], wintypes.BOOL),
        "Thread32Next": ([wintypes.HANDLE, ctypes.POINTER(_ThreadEntry)], wintypes.BOOL),
        "OpenThread": ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
        "ResumeThread": ([wintypes.HANDLE], wintypes.DWORD),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(kernel, name)
        function.argtypes = arguments
        function.restype = result
    return kernel


def _windows_error() -> OSError:
    return cast(OSError, ctypes.WinError(ctypes.get_last_error()))  # type: ignore[attr-defined]


class _WindowsJob:
    def __init__(self) -> None:
        self.kernel = _kernel32()
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise _windows_error()
        limits = _ExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(
            self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            error = _windows_error()
            self.close()
            raise error

    def assign(self, process: subprocess.Popen[str]) -> None:
        # Popen holds this process handle until its object is destroyed, avoiding
        # PID reuse. No member can request breakaway from this job.
        if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):  # type: ignore[attr-defined]
            raise _windows_error()

    def active_processes(self) -> int:
        information = _BasicAccountingInformation()
        if not self.kernel.QueryInformationJobObject(
            self.handle, 1, ctypes.byref(information), ctypes.sizeof(information), None
        ):
            raise _windows_error()
        return int(information.ActiveProcesses)

    def terminate(self) -> None:
        if not self.kernel.TerminateJobObject(self.handle, 1):
            raise _windows_error()

    def close(self) -> None:
        if self.handle:
            handle, self.handle = self.handle, None
            if not self.kernel.CloseHandle(handle):
                raise _windows_error()

    def __del__(self) -> None:
        # The kernel also closes our non-inheritable handle when the owner exits.
        if getattr(self, "handle", None):
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def _resume_suspended_process(process: subprocess.Popen[str], job: _WindowsJob) -> None:
    """Resume the single primary thread of our not-yet-running child."""
    kernel = job.kernel
    snapshot = kernel.CreateToolhelp32Snapshot(0x00000004, 0)  # TH32CS_SNAPTHREAD
    if snapshot == ctypes.c_void_p(-1).value:
        raise _windows_error()
    try:
        entry = _ThreadEntry()
        entry.dwSize = ctypes.sizeof(entry)
        found = kernel.Thread32First(snapshot, ctypes.byref(entry))
        while found:
            if entry.th32OwnerProcessID == process.pid:
                thread = kernel.OpenThread(0x0002, False, entry.th32ThreadID)  # SUSPEND_RESUME
                if not thread:
                    raise _windows_error()
                try:
                    if kernel.ResumeThread(thread) == 0xFFFFFFFF:
                        raise _windows_error()
                    return
                finally:
                    kernel.CloseHandle(thread)
            found = kernel.Thread32Next(snapshot, ctypes.byref(entry))
        raise OSError("suspended child primary thread was not found")
    finally:
        kernel.CloseHandle(snapshot)


class _OwnedWindowsJob:
    """Retain ownership and one cleanup result across concurrent callers."""

    def __init__(self, job: _WindowsJob) -> None:
        self.job = job
        self.lock = threading.Lock()
        self.closed = False
        self.error: BaseException | None = None


def start_windows(
    argv: Sequence[str], *, cwd: Path, environment: Mapping[str, str], stdin: int | None = None
) -> subprocess.Popen[str]:
    """Create a child tree that belongs to one private job before it executes."""
    if not is_windows():
        raise OSError("Windows process ownership requested on another platform")
    job = _WindowsJob()
    process = None
    try:
        process = subprocess.Popen(
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
            shell=False,
            creationflags=CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
        )
        job.assign(process)
        process._neurocraft_job = job  # type: ignore[attr-defined]
        process._neurocraft_job_state = _OwnedWindowsJob(job)  # type: ignore[attr-defined]
        _resume_suspended_process(process, job)
        return process
    except BaseException:
        # Assignment failure is fail-closed: the suspended child has run no code
        # and has no descendants. Never fall back to PID-tree discovery/taskkill.
        try:
            if process is not None:
                process.kill()
                process.wait(timeout=5)
        finally:
            job.close()
        raise


def terminate_windows(process: subprocess.Popen[str], *, timeout: float = 8.0) -> None:
    """Stop one held job; repeated or concurrent cleanup retains its first result.

    A closed ownership record is retained on this Popen object. It is never
    inferred from a PID or a wrapper's exit, so unrelated processes stay refused.
    """
    job = getattr(process, "_neurocraft_job", None)
    state = getattr(process, "_neurocraft_job_state", None)
    if job is None or state is None or state.job is not job:
        raise OSError("refusing to terminate a Windows child without its owned job handle")
    with state.lock:
        if state.error is not None:
            raise OSError("previous cleanup of the owned Windows job failed") from state.error
        if state.closed:
            return
        try:
            if job.active_processes():
                job.terminate()
            deadline = time.monotonic() + timeout
            while job.active_processes():
                if time.monotonic() >= deadline:
                    raise OSError("owned Windows job still has active children after termination")
                time.sleep(0.05)
            process.wait(timeout=max(0.1, timeout))
        except BaseException as exc:
            state.error = exc
            raise
        finally:
            try:
                job.close()
            except BaseException as exc:
                state.error = exc
                raise
            finally:
                state.closed = True
