"""Private-test companion: guided local setup and a loopback-only player interface.

Importing this module performs no I/O. All gameplay uses the existing installed
Fabric launcher, authenticated neural bridge and declared control conditions.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
import threading
import webbrowser
from collections import deque
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import companion_downloads as downloads  # noqa: E402
from scripts import installed_mod, public_alpha  # noqa: E402
from scripts import run_interactive_demo as launcher  # noqa: E402

CONDITIONS = ("original", "shuffled_weight", "no_input")
MAX_REQUEST_BYTES = 16_384
STATIC_FILES = {
    "/": ("index.html", "text/html"),
    "/app.js": ("app.js", "text/javascript"),
    "/style.css": ("style.css", "text/css"),
}


class Companion:
    def __init__(self, root: Path = ROOT) -> None:
        self.root = root
        self.directory = root / "artifacts/companion"
        self.runtime = self.directory / "fabric-runtime"
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.closed = threading.Event()
        self.worker: threading.Thread | None = None
        self.process: subprocess.Popen[str] | None = None
        self.process_kind = ""
        self.phase = "idle"
        self.error = ""
        self.join_address = ""
        self.lines: deque[str] = deque(maxlen=120)
        self.settings: dict[str, str] = {}

    def note(self, line: str) -> None:
        # The UI is local. Support reports deliberately omit these local paths/logs.
        with self.lock:
            self.lines.append(line.rstrip()[:1200])
            match = re.search(r"Join (127\.0\.0\.1:[0-9]{1,5}) as", line)
            if match:
                self.join_address = match[1]
            if "NeuroCraft Fly is ready in Minecraft." in line:
                self.phase = "playing"

    def state(self) -> dict[str, Any]:
        with self.lock:
            return {
                "phase": self.phase,
                "busy": bool(self.worker and self.worker.is_alive()),
                "error": self.error,
                "join_address": self.join_address,
                "log": list(self.lines),
                "platform": downloads.platform_key(),
                "testing_candidate": True,
                "settings": self.settings.copy(),
                "installation": str(self.root.parent),
            }

    def report(self) -> dict[str, Any]:
        # No usernames, paths, session tokens, local logs or game/account material.
        return {
            "schema": "neurocraft.companion-support.v1",
            "platform": downloads.platform_key(),
            "python": platform.python_version(),
            "phase": self.phase,
            "error_category": self.error,
            "testing_candidate": True,
            "notice": "This report establishes no installation or gameplay qualification.",
        }

    def environment(self) -> dict[str, str]:
        environment = launcher._child_environment("")
        environment.pop(launcher.TOKEN_NAME, None)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        if self.settings.get("java_home"):
            environment["JAVA_HOME"] = self.settings["java_home"]
        return environment

    def run_command(self, label: str, argv: Sequence[str], *, kind: str = "setup") -> None:
        if self.cancel.is_set():
            raise InterruptedError("Setup or launch cancelled.")
        # A fixed bootstrap keeps -I isolation while allowing the data builder's
        # historical sibling imports. No arbitrary entrypoint can be requested.
        command = list(argv)
        for index, part in enumerate(command):
            if part.startswith("scripts/") and part.endswith(".py"):
                command[index : index + 1] = [
                    str(self.root / "scripts/companion_bootstrap.py"),
                    Path(part).name,
                ]
                break
        self.note(label)
        process = launcher._start(
            command,
            cwd=self.root,
            environment=self.environment(),
            stdin=subprocess.PIPE if kind == "game" else subprocess.DEVNULL,
        )
        with self.lock:
            self.process, self.process_kind = process, kind
        try:
            if self.cancel.is_set():
                launcher._terminate(process)
            assert process.stdout is not None
            for line in process.stdout:
                self.note(line)
            code = process.wait()
            if self.cancel.is_set():
                raise InterruptedError("Setup or launch cancelled.")
            if code:
                raise RuntimeError(f"{label} failed (exit {code}). See the local progress log.")
        finally:
            launcher._terminate(process)
            with self.lock:
                self.process, self.process_kind = None, ""

    def java_home(self, requested: str, *, allow_downloads: bool) -> str:
        executable = "java.exe" if os.name == "nt" else "java"
        candidates = [Path(requested) / "bin" / executable] if requested else []
        if not requested:
            if os.environ.get("JAVA_HOME"):
                candidates.append(Path(os.environ["JAVA_HOME"]) / "bin" / executable)
            if found := shutil.which(executable):
                candidates.append(Path(found))
            private = self.directory / "java"
            if private.is_dir():
                candidates.extend(private.rglob("bin/" + executable))
        for candidate in candidates:
            try:
                result = subprocess.run(
                    [str(candidate), "-XshowSettings:properties", "-version"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    env=self.environment(),
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            if result.returncode == 0 and re.search(
                r'version "21(?:\.|\")', result.stderr + result.stdout
            ):
                found_home = re.search(
                    r"(?m)^\s*java\.home\s*=\s*(.+?)\s*$", result.stderr + result.stdout
                )
                if found_home:
                    actual_home = Path(found_home[1])
                    if (actual_home / "bin" / executable).is_file():
                        return str(actual_home)
        if requested:
            raise ValueError("The selected Java home does not contain a working Java 21 runtime.")
        if not allow_downloads:
            raise ValueError("Choose Java 21 or allow the pinned private runtime download.")
        pins = json.loads((self.root / "release/companion-runtime-pins.json").read_bytes())
        pin = pins["platforms"][downloads.platform_key()]["java"]
        archive = self.directory / "downloads" / ("java-" + pin["sha256"] + ".archive")
        downloads.download(pin, archive, progress=self.note, cancelled=self.cancel.is_set)
        target = self.directory / "java"
        if not target.exists():
            downloads.extract(archive, target, cancelled=self.cancel.is_set)
        return self.java_home("", allow_downloads=False)

    def setup(self, payload: Mapping[str, Any]) -> None:
        if payload.get("accept_eula") is not True:
            raise ValueError("Read and accept the Minecraft EULA before preparing a server.")
        if payload.get("allow_downloads") is not True:
            raise ValueError(
                "Setup needs permission for the displayed local runtime/model downloads."
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        self.settings["java_home"] = self.java_home(
            str(payload.get("java_home", "")).strip(), allow_downloads=True
        )
        if self.cancel.is_set():
            raise InterruptedError("Setup cancelled.")
        eula = self.root / "mod/run/eula.txt"
        eula.parent.mkdir(parents=True, exist_ok=True)
        if eula.is_symlink():
            raise ValueError("The EULA acceptance path must not be a symlink.")
        eula.write_text("# Accepted explicitly in NeuroCraft Fly Companion\neula=true\n")
        python = sys.executable
        self.run_command(
            "Verify or acquire the pinned model inputs (~1.1 GB).",
            [python, "-I", "-B", "scripts/acquire_malecns.py", "--include-metadata"],
        )
        graph = self.root / "data/processed/male-cns-v1.0-superclass-v1"
        if not graph.exists():
            receipt = self.directory / ("data-build-" + secrets.token_hex(6) + ".json")
            self.run_command(
                "Build and verify the local network. This can take several minutes.",
                [python, "-I", "-B", "scripts/build_malecns_runtime.py", "--receipt", str(receipt)],
            )
        else:
            self.note("Keeping the existing graph. Full identities are checked before play.")
        jar, version = public_alpha.source_mod(self.root)
        if not self.runtime.exists():
            attempt = self.directory / ("fabric-attempt-" + secrets.token_hex(6))
            try:
                self.run_command(
                    "Prepare the isolated Fabric server from official downloads.",
                    [
                        python,
                        "-I",
                        "-B",
                        "scripts/installed_mod.py",
                        "--output",
                        str(attempt),
                        "--mod-jar",
                        str(jar),
                    ],
                )
                installed_mod.validate_candidate(attempt, jar, expected_version=version)
                if self.cancel.is_set():
                    raise InterruptedError("Server setup cancelled.")
                if self.runtime.exists():
                    raise ValueError("A runtime appeared during setup; refusing to replace it.")
                attempt.rename(self.runtime)
            except BaseException:
                self.note(
                    "Incomplete server attempt retained at "
                    + str(attempt)
                    + ". Retry uses a new folder."
                )
                raise
        else:
            installed_mod.validate_candidate(self.runtime, jar, expected_version=version)
        self.note(
            "Setup complete. Use Install client files to prepare matching mods "
            "and the patched launcher version for your isolated game folder."
        )
        self.phase = "ready"

    def play(self, payload: Mapping[str, Any]) -> None:
        name = payload.get("player_name", "")
        mods = payload.get("client_mods_dir", "")
        if not isinstance(name, str) or not re.fullmatch("[A-Za-z0-9_]{3,16}", name):
            raise ValueError(
                "Enter your Minecraft Java player name (3-16 letters, digits or underscores)."
            )
        if not isinstance(mods, str) or not mods.strip():
            raise ValueError(
                "Choose the matching Fabric client's mods folder so both JARs can be verified."
            )
        jar, version = public_alpha.source_mod(self.root)
        installed_mod.validate_candidate(
            self.runtime, jar, Path(mods).expanduser(), expected_version=version
        )
        self.settings.update(player_name=name, client_mods_dir=mods)
        self.settings["java_home"] = self.java_home(
            str(payload.get("java_home", self.settings.get("java_home", ""))), allow_downloads=False
        )
        self.phase = "starting"
        self.join_address = ""
        self.run_command(
            "Starting the playground. Open your Fabric client and wait for the join address.",
            [
                sys.executable,
                "-I",
                "-B",
                "-u",
                "-X",
                "utf8",
                "scripts/run_interactive_demo.py",
                "--controller-profile",
                public_alpha.PROFILE,
                "--showcase-world",
                "--circuit-playground",
                "--comparison-lanes",
                "--control-seed",
                "7",
                "--kernel-threads",
                "8",
                "--action-latency",
                "--control-stdin",
                "--installed-runtime",
                str(self.runtime),
                "--manual-client-name",
                name,
            ],
            kind="game",
        )
        self.phase = "ready"

    def install_client(self, payload: Mapping[str, Any]) -> None:
        """Install matching mods and a new patched launcher version, without account edits."""
        raw = payload.get("client_mods_dir", "")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Choose an isolated Fabric profile's mods folder first.")
        directory = Path(raw).expanduser().absolute()
        launcher_raw = payload.get("client_launcher_dir", "")
        if not isinstance(launcher_raw, str) or not launcher_raw.strip():
            raise ValueError("Choose your Minecraft launcher folder first.")
        launcher_root = Path(launcher_raw).expanduser().absolute()
        with self.lock:
            if self.worker and self.worker.is_alive():
                raise ValueError("Wait for the current operation before installing client files.")
            if installed_mod._redirected(directory):
                raise ValueError("Client mods folder must not redirect through a link or junction.")
            jar, version = public_alpha.source_mod(self.root)
            files = installed_mod.validate_candidate(self.runtime, jar, expected_version=version)
            matching_pair = False
            if directory.exists():
                if not directory.is_dir():
                    raise ValueError("Client mods path is not a folder.")
                if any(p.suffix.lower() == ".jar" for p in directory.iterdir()):
                    installed_mod.validate_client_mods(files, directory)
                    matching_pair = True
            else:
                directory.mkdir()
            created: list[Path] = []
            try:
                for name in () if matching_pair else ("neurocraft-fly.jar", "fabric-api.jar"):
                    destination = directory / name
                    if destination.exists():
                        continue
                    with destination.open("xb") as target:
                        created.append(destination)
                        with (self.runtime / "server/mods" / name).open("rb") as source:
                            shutil.copyfileobj(source, target)
                profile = installed_mod.java_runtime.install_client(
                    self.runtime / "server", launcher_root, directory
                )
                installed_mod.validate_candidate(
                    self.runtime, jar, directory, expected_version=version
                )
            except BaseException:
                for destination in created:
                    destination.unlink(missing_ok=True)
                raise
            self.settings["client_mods_dir"] = str(directory)
            self.settings["client_launcher_dir"] = str(launcher_root)
            self.settings["client_version"] = profile
            self.note(
                "Client files installed. In your launcher, select version "
                + profile
                + " for the isolated game folder, then start Minecraft."
            )

    def launch(self, action: str, payload: Mapping[str, Any]) -> None:
        if action not in ("setup", "start"):
            raise ValueError("Unknown companion action.")
        with self.lock:
            if self.worker and self.worker.is_alive():
                raise ValueError(
                    "An operation is already running. Stop it before starting another."
                )
            self.cancel.clear()
            self.error = ""
            self.phase = "setting_up" if action == "setup" else "starting"

            def work() -> None:
                try:
                    (self.setup if action == "setup" else self.play)(payload)
                except InterruptedError:
                    self.phase = "stopped"
                    self.note("Operation stopped. Verified data is retained.")
                except Exception as exc:
                    self.error = "setup_failed" if action == "setup" else "launch_failed"
                    self.phase = "error"
                    self.note(str(exc))
                finally:
                    self.join_address = ""

            self.worker = threading.Thread(target=work, name="companion-operation", daemon=True)
            self.worker.start()

    def condition(self, condition: str) -> None:
        if condition not in CONDITIONS:
            raise ValueError("Unknown control condition.")
        with self.lock:
            if (
                self.phase != "playing"
                or self.process_kind != "game"
                or self.process is None
                or self.process.stdin is None
            ):
                raise ValueError("Join the live playground before changing conditions.")
            self.process.stdin.write(
                json.dumps({"action": "condition", "condition": condition}) + "\n"
            )
            self.process.stdin.flush()
        self.note(
            "Requested condition: " + condition + ". Confirm the acknowledged condition on the HUD."
        )

    def stop(self) -> None:
        self.cancel.set()
        with self.lock:
            process, kind = self.process, self.process_kind
            if process is None:
                return
            if kind == "game" and process.poll() is None and process.stdin is not None:
                try:
                    process.stdin.write('{"action":"stop"}\n')
                    process.stdin.flush()
                    return
                except OSError:
                    pass
        launcher._terminate(process)

    def shutdown(self) -> None:
        self.stop()
        if self.worker:
            self.worker.join(timeout=35)
        with self.lock:
            process = self.process
        if process is not None:
            launcher._terminate(process)
        self.closed.set()


class CompanionServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, companion: Companion, port: int = 0) -> None:
        self.companion = companion
        self.token = secrets.token_urlsafe(32)
        super().__init__(("127.0.0.1", port), CompanionHandler)
        self.origin = f"http://127.0.0.1:{self.server_address[1]}"


class CompanionHandler(BaseHTTPRequestHandler):
    server: CompanionServer

    def log_message(self, format: str, *args: Any) -> None:
        pass  # Never log the browser token or local player/settings data.

    def reply(self, status: int, content: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'",
        )
        self.end_headers()
        self.wfile.write(content)

    def authorized(self) -> bool:
        return self.headers.get("Host") == self.server.origin.removeprefix(
            "http://"
        ) and secrets.compare_digest(
            self.headers.get("Authorization", "").encode("utf-8"),
            ("Bearer " + self.server.token).encode("ascii"),
        )

    def do_GET(self) -> None:
        if self.headers.get("Host") != self.server.origin.removeprefix("http://"):
            self.reply(403, b'{"error":"Unexpected host"}')
            return
        if self.path in STATIC_FILES:
            name, kind = STATIC_FILES[self.path]
            self.reply(
                200, (self.server.companion.root / "apps/companion" / name).read_bytes(), kind
            )
            return
        if not self.authorized():
            self.reply(403, b'{"error":"Open Companion using its launch shortcut"}')
            return
        if self.path in ("/api/state", "/api/report"):
            content = (
                self.server.companion.state()
                if self.path == "/api/state"
                else self.server.companion.report()
            )
            self.reply(200, json.dumps(content, allow_nan=False).encode())
        else:
            self.reply(404, b'{"error":"Unknown route"}')

    def do_POST(self) -> None:
        if not self.authorized() or self.headers.get("Origin") != self.server.origin:
            self.reply(403, b'{"error":"Request must come from this Companion window"}')
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if (
                self.headers.get("Content-Type") != "application/json"
                or not 0 < length <= MAX_REQUEST_BYTES
            ):
                raise ValueError("Expected a bounded JSON request.")
            if self.headers.get("Transfer-Encoding") is not None:
                raise ValueError("Transfer encodings are not accepted.")
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Truncated JSON request.")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object.")
            if self.path in ("/api/setup", "/api/start"):
                self.server.companion.launch(self.path.removeprefix("/api/"), payload)
            elif self.path == "/api/install-client":
                self.server.companion.install_client(payload)
            elif self.path == "/api/condition":
                self.server.companion.condition(payload.get("condition", ""))
            elif self.path == "/api/stop":
                self.server.companion.stop()
            elif self.path == "/api/close":
                threading.Thread(target=self.server.companion.shutdown, daemon=True).start()
            else:
                raise ValueError("Unknown action.")
            self.reply(202, b'{"accepted":true}')
        except (ValueError, OSError, TypeError) as exc:
            self.reply(400, json.dumps({"error": str(exc)[:500]}).encode())


def acquire_lock(path: Path) -> BinaryIO:
    """OS-held locks release on crashes and prevent duplicate supervisors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            if not stream.read(1):
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            windows_lock: Any = msvcrt  # Windows-only symbols are absent from POSIX stubs.
            windows_lock.locking(stream.fileno(), windows_lock.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise ValueError("Companion is already running for this installation.") from None
    return stream


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    companion = Companion()
    try:
        with acquire_lock(companion.directory / "app.lock"), CompanionServer(companion) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = server.origin + "/#" + server.token
            print(
                "Companion is running locally. Use Stop and close in its window.",
                flush=True,
            )
            if args.no_browser:
                print(url, flush=True)
            else:
                webbrowser.open(url)
            try:
                while not companion.closed.wait(0.25):
                    pass
            except KeyboardInterrupt:
                pass
            finally:
                companion.shutdown()
                server.shutdown()
                thread.join(timeout=3)
        return 0
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
