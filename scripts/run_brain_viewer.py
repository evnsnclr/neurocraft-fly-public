#!/usr/bin/env python3
"""Serve the presentation-only brain viewer on loopback.

Serves ``apps/brain-viewer`` statics plus a ``/telemetry`` Server-Sent-Events
endpoint that tails the presentation telemetry file the neural service appends
to. Loopback only, stdlib only, read-only over the repository: this is a
viewing aid for the live demo and never an evidence surface.
"""

from __future__ import annotations

import argparse
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPOSITORY_ROOT / "apps" / "brain-viewer"
DEFAULT_PORT = 8791

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}

#: Static names served, allow-listed; nothing else on disk is reachable.
STATIC_ALLOWLIST = {
    "/": "index.html",
    "/index.html": "index.html",
    "/app.js": "app.js",
    "/styles.css": "styles.css",
    "/layout-flight-steering-v1.json": "layout-flight-steering-v1.json",
}


class ViewerError(RuntimeError):
    """Raised when the viewer cannot start safely."""


def build_handler(telemetry_path: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:
            return  # quiet; this is a local viewing aid

        def do_GET(self) -> None:  # http.server API name
            if self.path == "/telemetry":
                self._stream_telemetry()
                return
            name = STATIC_ALLOWLIST.get(self.path)
            if name is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            payload = (APP_ROOT / name).read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", CONTENT_TYPES[Path(name).suffix])
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _stream_telemetry(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            # Tail from the current end so the view is live, not historical.
            position = telemetry_path.stat().st_size if telemetry_path.exists() else 0
            try:
                while True:
                    if telemetry_path.exists():
                        with telemetry_path.open("rb") as stream:
                            stream.seek(position)
                            chunk = stream.read()
                        if chunk:
                            position += len(chunk)
                            for line in chunk.splitlines():
                                if line.strip():
                                    self.wfile.write(b"data: " + line + b"\n\n")
                            self.wfile.flush()
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                return

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    if not APP_ROOT.is_dir():
        raise ViewerError(f"viewer assets missing at {APP_ROOT}")
    if not 1024 <= args.port <= 65535:
        raise ViewerError("viewer port must be in [1024, 65535]")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), build_handler(args.telemetry))
    print(f"brain viewer: http://127.0.0.1:{args.port}/  (presentation only)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
