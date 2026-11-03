from __future__ import annotations

import http.client
import io
import json
import socket
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import companion_app as app
from scripts import process_lifecycle as ownership


@dataclass
class RunningCompanion:
    companion: app.Companion
    server: app.CompanionServer
    thread: threading.Thread

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: Any = None,
        body: bytes | None = None,
        headers: dict[str, str | None] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        selected = {
            "Authorization": "Bearer " + self.server.token,
            "Origin": self.server.origin,
            "Content-Type": "application/json",
            "Connection": "close",
        }
        for key, value in (headers or {}).items():
            if value is None:
                selected.pop(key, None)
            else:
                selected[key] = value
        if method == "POST" and body is None:
            body = json.dumps({} if payload is None else payload).encode()
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        try:
            connection.request(method, path, body=body, headers=selected)
            response = connection.getresponse()
            return response.status, dict(response.headers), response.read()
        finally:
            connection.close()


@pytest.fixture
def running(tmp_path: Path) -> Iterator[RunningCompanion]:
    root = tmp_path / "isolated app"
    static = root / "apps/companion"
    static.mkdir(parents=True)
    for name, content in (
        ("index.html", "<!doctype html><title>Private test</title>"),
        ("app.js", "// local static fixture"),
        ("style.css", "body { color: black; }"),
    ):
        (static / name).write_text(content)
    companion = app.Companion(root)
    with app.CompanionServer(companion) as server:
        thread = threading.Thread(
            target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
        )
        thread.start()
        try:
            yield RunningCompanion(companion, server, thread)
        finally:
            # Test workers are always synthetic. Never call a real launcher here.
            companion.cancel.set()
            if companion.worker:
                companion.worker.join(timeout=2)
            server.shutdown()
            thread.join(timeout=2)


def _join_operation(companion: app.Companion) -> None:
    assert companion.worker is not None
    companion.worker.join(timeout=3)
    assert not companion.worker.is_alive(), "synthetic companion worker did not finish"


def _snapshot(root: Path) -> dict[str, bytes | None]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


@pytest.mark.parametrize(
    "override",
    [
        {"Authorization": None},
        {"Authorization": "Bearer wrong"},
        {"Origin": None},
        {"Origin": "https://attacker.invalid"},
        {"Origin": "null"},
        {"Host": "attacker.invalid"},
        {"Host": "localhost:1234"},
    ],
)
def test_mutations_require_local_host_origin_and_bearer_together(
    running: RunningCompanion, monkeypatch: pytest.MonkeyPatch, override: dict[str, str | None]
) -> None:
    calls = []
    monkeypatch.setattr(running.companion, "launch", lambda *args: calls.append(args))
    status, _, body = running.request(
        "POST", "/api/setup", payload={"accept_eula": True}, headers=override
    )
    assert status == 403
    assert calls == []
    assert running.server.token.encode() not in body
    assert not running.companion.directory.exists()


def test_non_ascii_bearer_is_rejected_as_unauthorized_instead_of_crashing_handler(
    running: RunningCompanion,
) -> None:
    status, _, _ = running.request(
        "POST", "/api/stop", headers={"Authorization": "Bearer caf\u00e9"}
    )
    assert status == 403


def test_authorized_mutation_reaches_only_the_named_operation(
    running: RunningCompanion, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(running.companion, "launch", lambda *args: calls.append(args))
    payload = {"accept_eula": False, "allow_downloads": False}
    status, _, _ = running.request("POST", "/api/setup", payload=payload)
    assert status == 202
    assert calls == [("setup", payload)]


@pytest.mark.parametrize("path", ["/", "/app.js", "/style.css"])
def test_static_ui_is_local_only_with_restrictive_headers(
    running: RunningCompanion, path: str
) -> None:
    status, headers, body = running.request("GET", path, headers={"Authorization": None})
    assert status == 200 and body
    assert headers["Cache-Control"] == "no-store"
    assert headers["X-Content-Type-Options"] == "nosniff"
    policy = headers["Content-Security-Policy"]
    assert "default-src 'none'" in policy
    assert "frame-ancestors 'none'" in policy
    assert "base-uri 'none'" in policy
    assert "'unsafe-inline'" not in policy and "'unsafe-eval'" not in policy
    assert not any(key.lower().startswith("access-control-allow") for key in headers)
    status, _, _ = running.request("GET", path, headers={"Host": "untrusted.invalid"})
    assert status == 403


@pytest.mark.parametrize("path", ["/api/state", "/api/report"])
def test_private_state_and_support_report_need_authentication(
    running: RunningCompanion, path: str
) -> None:
    assert running.request("GET", path, headers={"Authorization": None})[0] == 403
    assert running.request("GET", path, headers={"Authorization": "Bearer wrong"})[0] == 403
    assert running.request("GET", path)[0] == 200


def test_support_report_omits_token_logs_account_material_and_installation_paths(
    running: RunningCompanion,
) -> None:
    private = "fixture-player-account-secret"
    running.companion.note(f"{private} {running.server.token} {running.companion.root}")
    running.companion.settings.update(player_name=private, java_home=str(running.companion.root))
    running.companion.error = "setup_failed"
    status, _, body = running.request("GET", "/api/report")
    assert status == 200
    report = json.loads(body)
    assert report["error_category"] == "setup_failed"
    for secret in (private, running.server.token, str(running.companion.root)):
        assert secret.encode() not in body
    assert not {"log", "settings", "installation", "token"} & set(report)


@pytest.mark.parametrize(
    "path", ["/missing", "/../../STATUS.md", "/%2e%2e/STATUS.md", "/api/state?token=anything"]
)
def test_unknown_get_paths_do_not_expose_files_or_token_query_auth(
    running: RunningCompanion, path: str
) -> None:
    (running.companion.root / "STATUS.md").write_text("private file must not be served")
    status, _, body = running.request("GET", path)
    assert status == 404
    assert b"private file must not be served" not in body


@pytest.mark.parametrize(
    "body,headers",
    [
        (b"{}", {"Content-Length": "50000"}),
        (b"{}", {"Content-Length": "-1"}),
        (b"{}", {"Content-Length": "invalid"}),
        (b"{}", {"Content-Type": "text/plain"}),
        (b"[]", {}),
        (b"null", {}),
        (b"invalid-json", {}),
        (b"{}", {"Transfer-Encoding": "chunked"}),
    ],
)
def test_malformed_or_unbounded_request_is_refused_before_action(
    running: RunningCompanion, monkeypatch: pytest.MonkeyPatch, body: bytes, headers: dict[str, str]
) -> None:
    calls = []
    monkeypatch.setattr(running.companion, "launch", lambda *args: calls.append(args))
    status, _, _ = running.request("POST", "/api/setup", body=body, headers=headers)
    assert status in (400, 413)
    assert calls == []


def test_truncated_declared_body_cannot_launch_an_operation(
    running: RunningCompanion, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(running.companion, "launch", lambda *args: calls.append(args))
    request = (
        f"POST /api/setup HTTP/1.1\r\nHost: 127.0.0.1:{running.server.server_port}\r\n"
        f"Authorization: Bearer {running.server.token}\r\nOrigin: {running.server.origin}\r\n"
        "Content-Type: application/json\r\nContent-Length: 9\r\nConnection: close\r\n\r\n{}"
    ).encode()
    with socket.create_connection(("127.0.0.1", running.server.server_port), timeout=3) as stream:
        stream.sendall(request)
        stream.shutdown(socket.SHUT_WR)
        response = http.client.HTTPResponse(stream)
        response.begin()
        assert response.status == 400
        response.read()
    assert calls == []


def test_busy_operation_serializes_setup_and_start(
    running: RunningCompanion, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, release = threading.Event(), threading.Event()
    calls = []

    def setup(payload: Any) -> None:
        calls.append("setup")
        entered.set()
        assert release.wait(timeout=3)

    monkeypatch.setattr(running.companion, "setup", setup)
    monkeypatch.setattr(running.companion, "play", lambda payload: calls.append("play"))
    try:
        assert running.request("POST", "/api/setup")[0] == 202
        assert entered.wait(timeout=2)
        assert running.request("POST", "/api/setup")[0] == 400
        assert running.request("POST", "/api/start")[0] == 400
        assert (
            running.request(
                "POST",
                "/api/install-client",
                payload={"client_mods_dir": str(running.companion.root / "client mods")},
            )[0]
            == 400
        )
        assert not (running.companion.root / "client mods").exists()
        assert json.loads(running.request("GET", "/api/state")[2])["busy"] is True
        assert calls == ["setup"]
    finally:
        release.set()
        _join_operation(running.companion)
    assert json.loads(running.request("GET", "/api/state")[2])["busy"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"accept_eula": True},
        {"allow_downloads": True},
        {"accept_eula": 1, "allow_downloads": True},
        {"accept_eula": True, "allow_downloads": "yes"},
    ],
)
def test_setup_requires_both_explicit_consents_before_files_or_commands(
    running: RunningCompanion, monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]
) -> None:
    before = _snapshot(running.companion.root)
    monkeypatch.setattr(
        running.companion, "run_command", lambda *args, **kwargs: pytest.fail("unconsented command")
    )
    monkeypatch.setattr(
        running.companion,
        "java_home",
        lambda *args, **kwargs: pytest.fail("unconsented Java setup"),
    )
    assert running.request("POST", "/api/setup", payload=payload)[0] == 202
    _join_operation(running.companion)
    assert running.companion.phase == "error"
    assert _snapshot(running.companion.root) == before
    assert not running.companion.directory.exists()


@pytest.mark.parametrize("condition", ["original", "shuffled_weight", "no_input"])
def test_condition_and_stop_forward_json_to_owned_launcher_without_shell(
    running: RunningCompanion, monkeypatch: pytest.MonkeyPatch, condition: str
) -> None:
    child = SimpleNamespace(stdin=io.StringIO(), poll=lambda: None)
    running.companion.process, running.companion.process_kind = child, "game"
    running.companion.phase = "playing"
    monkeypatch.setattr(
        app.launcher,
        "_terminate",
        lambda *args, **kwargs: pytest.fail("graceful game stop force-killed child"),
    )
    assert running.request("POST", "/api/condition", payload={"condition": condition})[0] == 202
    assert running.request("POST", "/api/stop")[0] == 202
    assert [json.loads(line) for line in child.stdin.getvalue().splitlines()] == [
        {"action": "condition", "condition": condition},
        {"action": "stop"},
    ]
    assert running.companion.cancel.is_set()


@pytest.mark.parametrize(
    "condition",
    ["original\nstop", "$(touch /tmp/not-created)", "shuffled_weights", "original; stop", None],
)
def test_invalid_condition_cannot_add_shell_or_console_commands(
    running: RunningCompanion, condition: Any
) -> None:
    child = SimpleNamespace(stdin=io.StringIO(), poll=lambda: None)
    running.companion.process, running.companion.process_kind = child, "game"
    running.companion.phase = "playing"
    status, _, _ = running.request("POST", "/api/condition", payload={"condition": condition})
    assert status == 400
    assert child.stdin.getvalue() == ""


def test_condition_requires_live_gameplay_instead_of_setup_child(running: RunningCompanion) -> None:
    child = SimpleNamespace(stdin=io.StringIO(), poll=lambda: None)
    running.companion.process, running.companion.process_kind = child, "setup"
    running.companion.phase = "setting_up"
    assert running.request("POST", "/api/condition", payload={"condition": "original"})[0] == 400
    assert child.stdin.getvalue() == ""


def test_play_checks_current_source_server_and_client_before_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    companion = app.Companion(tmp_path)
    jar = tmp_path / "candidate.jar"
    mods = tmp_path / "client mods"
    order: list[Any] = []
    monkeypatch.setattr(app.public_alpha, "source_mod", lambda root: (jar, "candidate-version"))

    def validate(*args: Any, **kwargs: Any) -> dict[str, str]:
        order.append(("validate", args, kwargs))
        return {}

    monkeypatch.setattr(app.installed_mod, "validate_candidate", validate)
    monkeypatch.setattr(companion, "java_home", lambda *args, **kwargs: "private-java")
    monkeypatch.setattr(
        companion, "run_command", lambda label, argv, **kwargs: order.append(("run", argv, kwargs))
    )
    companion.play({"player_name": "Test_Player", "client_mods_dir": str(mods)})
    assert order[0] == (
        "validate",
        (companion.runtime, jar, mods),
        {"expected_version": "candidate-version"},
    )
    assert order[1][0] == "run" and order[1][2] == {"kind": "game"}
    command = order[1][1]
    assert isinstance(command, list)
    assert command[command.index("--installed-runtime") + 1] == str(companion.runtime)
    assert command[command.index("--manual-client-name") + 1] == "Test_Player"
    assert "--control-stdin" in command and "--comparison-lanes" in command
    assert not companion.directory.exists(), "the mocked launch wrote installation state"


def test_invalid_candidate_prevents_java_or_game_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    companion = app.Companion(tmp_path)
    monkeypatch.setattr(app.public_alpha, "source_mod", lambda root: (tmp_path / "mod.jar", "rc"))

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise ValueError("candidate client bytes differ")

    monkeypatch.setattr(app.installed_mod, "validate_candidate", refuse)
    monkeypatch.setattr(
        companion, "java_home", lambda *args, **kwargs: pytest.fail("Java ran before validation")
    )
    monkeypatch.setattr(
        companion, "run_command", lambda *args, **kwargs: pytest.fail("invalid candidate launched")
    )
    with pytest.raises(ValueError, match="client bytes differ"):
        companion.play({"player_name": "Test_Player", "client_mods_dir": str(tmp_path / "mods")})
    assert not companion.directory.exists()


def test_setup_reuses_existing_candidate_validator_with_temporary_acceptance_intercepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    companion = app.Companion(tmp_path)
    companion.runtime.mkdir(parents=True)
    (tmp_path / "data/processed/male-cns-v1.0-superclass-v1").mkdir(parents=True)
    jar = tmp_path / "candidate.jar"
    monkeypatch.setattr(app.public_alpha, "source_mod", lambda root: (jar, "candidate-version"))
    monkeypatch.setattr(companion, "java_home", lambda *args, **kwargs: "private-java")
    commands, checks, acceptance_intents = [], [], []
    monkeypatch.setattr(companion, "run_command", lambda *args, **kwargs: commands.append(args))
    monkeypatch.setattr(
        app.installed_mod,
        "validate_candidate",
        lambda *args, **kwargs: checks.append((args, kwargs)),
    )
    eula = tmp_path / "mod/run/eula.txt"
    original_write = Path.write_text

    def intercept(path: Path, text: str, *args: Any, **kwargs: Any) -> int:
        if path == eula:
            acceptance_intents.append(text)
            return len(text)
        return original_write(path, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", intercept)
    companion.setup({"accept_eula": True, "allow_downloads": True})
    assert checks == [((companion.runtime, jar), {"expected_version": "candidate-version"})]
    assert commands, "explicit setup did not request its mocked acquisition check"
    assert acceptance_intents and not eula.exists(), (
        "the fixture must never write actual acceptance"
    )
    assert companion.phase == "ready"


def test_setup_stop_and_worker_cleanup_share_windows_job_ownership_safely(
    running: RunningCompanion, monkeypatch: pytest.MonkeyPatch
) -> None:
    reading, terminated = threading.Event(), threading.Event()
    events: list[str] = []
    active = [True]

    def terminate() -> None:
        events.append("terminate")
        active[0] = False
        terminated.set()

    job = SimpleNamespace(
        active_processes=lambda: int(active[0]),
        terminate=terminate,
        close=lambda: events.append("close"),
    )

    class BlockingOutput:
        def __iter__(self) -> Iterator[str]:
            reading.set()
            assert terminated.wait(timeout=3)
            return iter(())

    child = SimpleNamespace(
        _neurocraft_job=job,
        _neurocraft_job_state=ownership._OwnedWindowsJob(job),
        stdout=BlockingOutput(),
        stdin=None,
        poll=lambda: None if active[0] else 0,
        wait=lambda **kwargs: 0,
    )
    monkeypatch.setattr(app.launcher, "_windows", lambda: True)
    monkeypatch.setattr(app.launcher, "_process_lifecycle_module", lambda: ownership)
    monkeypatch.setattr(app.launcher, "_start", lambda *args, **kwargs: child)
    monkeypatch.setattr(running.companion, "environment", lambda: {})
    monkeypatch.setattr(
        running.companion,
        "setup",
        lambda payload: running.companion.run_command("synthetic setup", ["never-executed"]),
    )
    try:
        assert running.request("POST", "/api/setup")[0] == 202
        assert reading.wait(timeout=2)
        assert running.request("POST", "/api/stop")[0] == 202
        _join_operation(running.companion)
        assert running.companion.phase == "stopped"
        assert running.companion.error == ""
        assert running.companion.process is None
        assert events == ["terminate", "close"]
    finally:
        terminated.set()


def test_install_client_copies_only_verified_pair_without_running_commands(
    running: RunningCompanion, monkeypatch: pytest.MonkeyPatch
) -> None:
    companion = running.companion
    source = companion.runtime / "server/mods"
    source.mkdir(parents=True)
    expected = {"neurocraft-fly.jar": b"mod fixture", "fabric-api.jar": b"API fixture"}
    for name, data in expected.items():
        (source / name).write_bytes(data)
    target = companion.root / "isolated client mods"
    jar = companion.root / "candidate.jar"
    calls = []
    monkeypatch.setattr(app.public_alpha, "source_mod", lambda root: (jar, "fixture-version"))

    def validate(
        runtime: Path, candidate: Path, client: Path | None = None, **kwargs: Any
    ) -> dict[str, str]:
        assert runtime == companion.runtime and candidate == jar
        assert kwargs == {"expected_version": "fixture-version"}
        calls.append(client)
        if client is not None:
            assert {path.name: path.read_bytes() for path in client.iterdir()} == expected
        return {}

    monkeypatch.setattr(app.installed_mod, "validate_candidate", validate)
    monkeypatch.setattr(
        app.installed_mod.java_runtime, "install_client", lambda *args: "patched-fixture"
    )
    monkeypatch.setattr(
        companion,
        "run_command",
        lambda *args, **kwargs: pytest.fail("client copy launched a command"),
    )
    assert (
        running.request(
            "POST",
            "/api/install-client",
            payload={"client_mods_dir": str(target), "client_launcher_dir": str(companion.root)},
        )[0]
        == 202
    )
    assert calls == [None, target]
    assert {path.name: path.read_bytes() for path in target.iterdir()} == expected
    assert companion.settings["client_mods_dir"] == str(target)


def test_install_client_preserves_matching_versioned_filenames_and_installs_profile(
    running: RunningCompanion, monkeypatch: pytest.MonkeyPatch
) -> None:
    companion = running.companion
    target = companion.root / "existing client mods"
    target.mkdir()
    expected = {
        "neurocraft-fly-0.2.0-rc.3.jar": b"matching mod",
        "fabric-api-0.128.2+1.21.5.jar": b"matching API",
    }
    for name, data in expected.items():
        (target / name).write_bytes(data)
    files = {
        "mods/neurocraft-fly.jar": app.installed_mod.digest(target / next(iter(expected))),
        "mods/fabric-api.jar": app.installed_mod.digest(target / "fabric-api-0.128.2+1.21.5.jar"),
    }
    monkeypatch.setattr(app.public_alpha, "source_mod", lambda root: (root / "candidate.jar", "rc"))

    def validate(
        runtime: Path, jar: Path, client: Path | None = None, **kwargs: Any
    ) -> dict[str, str]:
        if client is not None:
            app.installed_mod.validate_client_mods(files, client)
        return files

    monkeypatch.setattr(app.installed_mod, "validate_candidate", validate)
    installed = []

    def install_profile(server: Path, launcher: Path, mods: Path) -> str:
        installed.append((server, launcher, mods))
        return "patched-fixture"

    monkeypatch.setattr(app.installed_mod.java_runtime, "install_client", install_profile)
    companion.install_client(
        {"client_mods_dir": str(target), "client_launcher_dir": str(companion.root)}
    )
    assert installed == [(companion.runtime / "server", companion.root, target)]
    assert {path.name: path.read_bytes() for path in target.iterdir()} == expected
    assert companion.settings["client_version"] == "patched-fixture"


def test_install_client_rejects_existing_mismatched_jars_without_overwriting(
    running: RunningCompanion, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = running.companion.root / "existing client mods"
    target.mkdir()
    (target / "other-mod.jar").write_bytes(b"belongs to player")
    before = _snapshot(target)
    monkeypatch.setattr(app.public_alpha, "source_mod", lambda root: (root / "candidate.jar", "rc"))

    def validate(
        runtime: Path, jar: Path, client: Path | None = None, **kwargs: Any
    ) -> dict[str, str]:
        if client is not None:
            raise ValueError("isolated client has mismatched JARs")
        return {}

    monkeypatch.setattr(app.installed_mod, "validate_candidate", validate)
    assert (
        running.request(
            "POST",
            "/api/install-client",
            payload={
                "client_mods_dir": str(target),
                "client_launcher_dir": str(running.companion.root),
            },
        )[0]
        == 400
    )
    assert _snapshot(target) == before


def test_install_client_rolls_back_only_files_created_by_failed_copy(
    running: RunningCompanion, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = running.companion.runtime / "server/mods"
    source.mkdir(parents=True)
    (source / "neurocraft-fly.jar").write_bytes(b"only first fixture file is present")
    target = running.companion.root / "client mods"
    target.mkdir()
    (target / "README.txt").write_bytes(b"player-owned non-mod file")
    before = _snapshot(target)
    monkeypatch.setattr(app.public_alpha, "source_mod", lambda root: (root / "candidate.jar", "rc"))
    monkeypatch.setattr(app.installed_mod, "validate_candidate", lambda *args, **kwargs: {})
    assert (
        running.request(
            "POST",
            "/api/install-client",
            payload={
                "client_mods_dir": str(target),
                "client_launcher_dir": str(running.companion.root),
            },
        )[0]
        == 400
    )
    assert _snapshot(target) == before


def test_install_client_invalid_server_does_not_create_destination(
    running: RunningCompanion, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = running.companion.root / "absent client mods"
    monkeypatch.setattr(app.public_alpha, "source_mod", lambda root: (root / "candidate.jar", "rc"))

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise ValueError("server candidate changed")

    monkeypatch.setattr(app.installed_mod, "validate_candidate", refuse)
    assert (
        running.request(
            "POST",
            "/api/install-client",
            payload={
                "client_mods_dir": str(target),
                "client_launcher_dir": str(running.companion.root),
            },
        )[0]
        == 400
    )
    assert not target.exists()


def test_installation_lock_rejects_second_supervisor_and_releases_on_close(tmp_path: Path) -> None:
    path = tmp_path / "companion/app.lock"
    with app.acquire_lock(path), pytest.raises(ValueError, match="already running"):
        app.acquire_lock(path)
    with app.acquire_lock(path) as stream:
        assert not stream.closed


def test_bootstrap_loads_isolated_data_builder_help_without_data_or_minecraft() -> None:
    import subprocess
    import sys

    root = Path(app.__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(root / "scripts/companion_bootstrap.py"),
            "build_malecns_runtime.py",
            "--help",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--receipt" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr


def test_bootstrap_refuses_arbitrary_script_before_executing_it(tmp_path: Path) -> None:
    import subprocess
    import sys

    root = Path(app.__file__).resolve().parents[1]
    marker = tmp_path / "must-not-exist"
    script = tmp_path / "unapproved.py"
    script.write_text(f"from pathlib import Path; Path({str(marker)!r}).touch()")
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(root / "scripts/companion_bootstrap.py"), str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode != 0
    assert "Unknown companion entrypoint" in result.stderr
    assert not marker.exists()


def test_setup_cancelled_after_java_does_not_write_eula_or_start_data_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    companion = app.Companion(tmp_path)

    def java_home(*args: Any, **kwargs: Any) -> str:
        companion.cancel.set()
        return "synthetic-java"

    monkeypatch.setattr(companion, "java_home", java_home)
    monkeypatch.setattr(
        companion,
        "run_command",
        lambda *args, **kwargs: pytest.fail("cancelled setup ran data step"),
    )
    with pytest.raises(InterruptedError):
        companion.setup({"accept_eula": True, "allow_downloads": True})
    assert not (tmp_path / "mod/run/eula.txt").exists()


def test_unknown_mutation_route_cannot_reach_an_operation(
    running: RunningCompanion, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(running.companion, "launch", lambda *args: calls.append(args))
    assert running.request("POST", "/api/setup?unexpected=path")[0] in (400, 404)
    assert calls == []
    assert not running.companion.directory.exists()


def test_java_shim_uses_reported_runtime_home_instead_of_shim_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import subprocess

    companion = app.Companion(tmp_path)
    executable = "java.exe" if os.name == "nt" else "java"
    shim = tmp_path / "system/bin" / executable
    shim.parent.mkdir(parents=True)
    shim.write_bytes(b"synthetic system shim; never executed")
    actual_home = tmp_path / "actual Java runtime/Contents/Home"
    actual_java = actual_home / "bin" / executable
    actual_java.parent.mkdir(parents=True)
    actual_java.write_bytes(b"synthetic Java executable; never executed")
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.setattr(app.shutil, "which", lambda name: str(shim))
    monkeypatch.setattr(companion, "environment", lambda: {})
    probes = []

    def probe(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        probes.append(argv)
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="",
            stderr=f'Property settings:\n    java.home = {actual_home}\nopenjdk version "21.0.8"\n',
        )

    monkeypatch.setattr(app.subprocess, "run", probe)
    assert companion.java_home("", allow_downloads=False) == str(actual_home)
    assert probes and probes[0][0] == str(shim)
    assert "-XshowSettings:properties" in probes[0]
    assert actual_home != shim.parent.parent
    assert not companion.directory.exists()


def test_version_only_java_shim_cannot_be_accepted_as_a_runtime_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import subprocess

    companion = app.Companion(tmp_path)
    executable = "java.exe" if os.name == "nt" else "java"
    shim = tmp_path / "system/bin" / executable
    shim.parent.mkdir(parents=True)
    shim.write_bytes(b"synthetic system shim; never executed")
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.setattr(app.shutil, "which", lambda name: str(shim))
    monkeypatch.setattr(companion, "environment", lambda: {})
    monkeypatch.setattr(
        app.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(
            argv, 0, stdout="", stderr='openjdk version "21.0.8"\n'
        ),
    )
    with pytest.raises(ValueError, match="Java 21"):
        companion.java_home("", allow_downloads=False)
    assert not companion.directory.exists()


@pytest.mark.parametrize("failure", ["installer_failed", "cancelled", "validation_failed"])
def test_failed_server_attempt_is_retained_and_retry_promotes_only_new_valid_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    companion = app.Companion(tmp_path)
    (tmp_path / "data/processed/male-cns-v1.0-superclass-v1").mkdir(parents=True)
    jar = tmp_path / "candidate.jar"
    monkeypatch.setattr(app.public_alpha, "source_mod", lambda root: (jar, "fixture-version"))
    monkeypatch.setattr(companion, "java_home", lambda *args, **kwargs: "synthetic-java")
    eula = tmp_path / "mod/run/eula.txt"
    original_write = Path.write_text
    acceptance_intents = []

    def intercept_eula(path: Path, text: str, *args: Any, **kwargs: Any) -> int:
        if path == eula:
            acceptance_intents.append(text)
            return len(text)
        return original_write(path, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", intercept_eula)
    attempts: list[Path] = []
    validations: list[Path] = []

    def command(label: str, argv: list[str], **kwargs: Any) -> None:
        if "--output" not in argv:
            return  # Acquisition/build commands are entirely synthetic here.
        attempt = Path(argv[argv.index("--output") + 1])
        assert not attempt.exists(), "retry reused an incomplete installation directory"
        attempt.mkdir()
        attempts.append(attempt)
        (attempt / "synthetic-attempt").write_bytes(str(len(attempts)).encode())
        if len(attempts) == 1:
            if failure == "installer_failed":
                raise RuntimeError("synthetic interrupted installer download")
            if failure == "cancelled":
                companion.cancel.set()

    def validate(runtime: Path, candidate: Path, **kwargs: Any) -> dict[str, str]:
        assert candidate == jar and kwargs == {"expected_version": "fixture-version"}
        validations.append(runtime)
        if failure == "validation_failed" and len(attempts) == 1:
            raise ValueError("synthetic candidate did not validate")
        assert runtime == attempts[-1] and runtime.is_dir()
        return {}

    monkeypatch.setattr(companion, "run_command", command)
    monkeypatch.setattr(app.installed_mod, "validate_candidate", validate)
    payload = {"accept_eula": True, "allow_downloads": True}
    companion.launch("setup", payload)
    _join_operation(companion)
    assert companion.phase == ("stopped" if failure == "cancelled" else "error")
    assert len(attempts) == 1 and attempts[0].is_dir()
    assert not companion.runtime.exists()
    retained = _snapshot(attempts[0])
    companion.launch("setup", payload)
    _join_operation(companion)
    assert companion.phase == "ready" and companion.error == ""
    assert len(attempts) == 2 and attempts[0] != attempts[1]
    assert _snapshot(attempts[0]) == retained, "retry changed the failed diagnostic attempt"
    assert not attempts[1].exists(), "successful attempt was copied rather than promoted"
    assert (companion.runtime / "synthetic-attempt").read_bytes() == b"2"
    assert validations[-1] == attempts[1], "candidate was promoted without source validation"
    assert len(acceptance_intents) == 2 and not eula.exists()
