"""The brain viewer is loopback, allow-listed, and presentation-only."""

from __future__ import annotations

import http.client
import json
import threading
import time
from collections.abc import Iterator
from hashlib import sha256
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from neurocraft_connectome import load_malecns_fixture
from neurocraft_connectome.flight_pin import MALECNS_FLIGHT_FIXTURE_V1_PIN
from neurocraft_connectome.model import canonical_json_bytes

from scripts.build_brain_viewer_layout import (
    FLIGHT_FIXTURE,
    FLIGHT_FIXTURE_BYTES,
    build_layout,
)
from scripts.build_brain_viewer_layout import main as build_layout_main
from scripts.run_brain_viewer import APP_ROOT, ViewerError, build_handler
from scripts.run_brain_viewer import main as viewer_main

ROOT = Path(__file__).resolve().parents[2]
LAYOUT_ASSET = ROOT / "apps" / "brain-viewer" / "layout-flight-steering-v1.json"


@pytest.fixture(scope="module")
def layout_document() -> dict[str, object]:
    graph = load_malecns_fixture(
        FLIGHT_FIXTURE,
        expected_pin=MALECNS_FLIGHT_FIXTURE_V1_PIN,
        expected_file_bytes=FLIGHT_FIXTURE_BYTES,
    )
    return build_layout(graph)


def test_committed_layout_asset_is_byte_identical_to_the_generator(
    layout_document: dict[str, object],
) -> None:
    assert LAYOUT_ASSET.read_bytes() == canonical_json_bytes(layout_document) + b"\n"


def test_layout_is_presentation_only_and_covers_the_whole_fixture(
    layout_document: dict[str, object],
) -> None:
    assert layout_document["schema"] == "neurocraft.presentation.brain_layout.v1"
    assert layout_document["presentation_only"] is True
    assert layout_document["fixture_id"] == "mcns-v1.0-flight-steering-v1"
    nodes = layout_document["nodes"]
    edges = layout_document["edges"]
    assert isinstance(nodes, list) and len(nodes) == 128
    assert isinstance(edges, list) and len(edges) == 887
    assert layout_document["group_counts"] == {
        "descending": 36,
        "interneuron": 38,
        "motor_left": 25,
        "motor_right": 27,
        "sensory": 2,
    }
    assert all(edge["sign"] in (-1, 0, 1) for edge in edges)


def test_layout_builder_refuses_to_clobber_a_divergent_asset(tmp_path: Path) -> None:
    target = tmp_path / "layout.json"
    target.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="delete deliberately"):
        build_layout_main(["--output", str(target)])


@pytest.fixture()
def viewer_server(tmp_path: Path) -> Iterator[tuple[int, Path]]:
    telemetry = tmp_path / "telemetry.ndjson"
    server = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(telemetry))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], telemetry
    finally:
        server.shutdown()
        server.server_close()


def _get(port: int, path: str) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def test_viewer_serves_only_the_allow_listed_assets(viewer_server: tuple[int, Path]) -> None:
    port, _ = viewer_server
    status, body = _get(port, "/")
    assert status == 200
    assert b"PRESENTATION VIEW" in body
    assert _get(port, "/layout-flight-steering-v1.json")[0] == 200
    assert _get(port, "/app.js")[0] == 200
    assert _get(port, "/styles.css")[0] == 200
    for refused in ("/../pyproject.toml", "/pyproject.toml", "/telemetry/../app.js", "/x.json"):
        assert _get(port, refused)[0] == 404


def test_telemetry_stream_tails_only_lines_written_after_connect(
    viewer_server: tuple[int, Path],
) -> None:
    port, telemetry = viewer_server
    with telemetry.open("ab") as stream:
        stream.write(b'{"stale": true}\n')

    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request("GET", "/telemetry")
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type") == "text/event-stream"
        time.sleep(0.2)  # let the tail position settle past the stale line
        fresh = {"presentation_only": True, "command_seq": 7}
        with telemetry.open("ab") as stream:
            stream.write(json.dumps(fresh).encode() + b"\n")
        line = response.fp.readline(65536)
        assert line.startswith(b"data: ")
        assert json.loads(line[len(b"data: ") :]) == fresh
    finally:
        connection.close()


def test_viewer_main_refuses_privileged_ports_and_requires_assets() -> None:
    assert APP_ROOT.is_dir()
    with pytest.raises(ViewerError, match="port"):
        viewer_main(["--telemetry", "unused.ndjson", "--port", "80"])


def test_committed_layout_asset_hash_is_recorded() -> None:
    digest = sha256(LAYOUT_ASSET.read_bytes()).hexdigest()
    assert digest == "cf1950d462b873c65f92fa322b165aefcab6e277ce4dd9985322583c680e558e"
