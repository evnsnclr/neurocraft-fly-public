from __future__ import annotations

import functools
import json
import math
import re
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

import pytest

from scripts import run_interactive_demo as launcher


def _prepared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, action_latency: bool = False
) -> launcher.RunPlan:
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    eula = tmp_path / "accepted-eula.txt"
    eula.write_text("# accepted locally\neula=true\n", encoding="utf-8")
    return launcher.prepare_run(
        run_root=tmp_path / "artifacts" / "live-interactive",
        accepted_eula=eula,
        now=datetime(2026, 8, 17, 20, 0, tzinfo=UTC),
        action_latency=action_latency,
    )


def test_prepare_requires_existing_eula_acceptance(tmp_path: Path) -> None:
    missing = tmp_path / "missing-eula.txt"
    with pytest.raises(launcher.LauncherError, match="eula=true"):
        launcher.prepare_run(run_root=tmp_path / "runs", accepted_eula=missing)

    rejected = tmp_path / "eula.txt"
    rejected.write_text("eula=false\n", encoding="utf-8")
    with pytest.raises(launcher.LauncherError, match="eula=true"):
        launcher.prepare_run(run_root=tmp_path / "runs", accepted_eula=rejected)


def test_showcase_world_is_opt_in_and_records_the_actual_ordered_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ordinary = _prepared(tmp_path, monkeypatch)
    enabled = launcher.prepare_run(
        run_root=tmp_path / "showcase-runs",
        accepted_eula=tmp_path / "accepted-eula.txt",
        showcase_world=True,
    )
    assert not ordinary.showcase_world
    ordinary_server, showcase_server = _RecordingServer(), _RecordingServer()
    launcher._prepare_arena(ordinary_server)  # type: ignore[arg-type]
    added = list(enabled.showcase_commands)
    # A source reload later in the session must not relabel the already-built world.
    monkeypatch.setattr(launcher, "_showcase_commands", lambda: ("changed source",))
    launcher._prepare_arena(
        showcase_server,
        showcase_commands=enabled.showcase_commands,  # type: ignore[arg-type]
    )
    launcher._write_session(enabled, state="stopped", identity=None)
    assert showcase_server.commands == (
        ordinary_server.commands[:-1] + added + ordinary_server.commands[-1:]
    )
    metadata = json.loads(enabled.session_path.read_text())
    assert (
        metadata["showcase_commands_sha256"]
        == sha256(json.dumps(added, separators=(",", ":")).encode()).hexdigest()
    )


def test_half_speed_is_explicit_and_recorded_without_replacing_normal_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ordinary = _prepared(tmp_path, monkeypatch)
    slow = launcher.prepare_run(
        run_root=tmp_path / "slow-runs",
        accepted_eula=tmp_path / "accepted-eula.txt",
        game_tick_rate=10,
    )
    assert ordinary.game_tick_rate == launcher.build_parser().parse_args([]).game_tick_rate == 20
    metadata = json.loads(slow.session_path.read_text())
    assert metadata["playback_speed"] == "half-speed diagnostic"
    assert metadata["requested_game_ticks_per_wall_second"] == 10
    with pytest.raises(launcher.LauncherError, match="game tick rate"):
        launcher.prepare_run(game_tick_rate=0)


def test_diagnostic_tools_are_opt_in_and_use_the_fresh_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ordinary = _prepared(tmp_path, monkeypatch)
    enabled = launcher.prepare_run(
        run_root=tmp_path / "diagnostic-runs",
        accepted_eula=tmp_path / "accepted-eula.txt",
        diagnostic_tools=True,
    )
    assert ordinary.console_path is None
    assert ordinary.client_actions_path is None
    assert not any("captureDirectory" in arg for arg in launcher._client_command(ordinary))
    assert not any("clientActionsPath" in arg for arg in launcher._client_command(ordinary))
    assert enabled.console_path is not None
    assert enabled.console_path.read_bytes() == b""
    assert f"-Dneurocraft.fly.captureDirectory={enabled.run_dir / 'capture'}" in (
        launcher._client_command(enabled)
    )
    assert json.loads(enabled.session_path.read_text())["diagnostic_tools"] is True
    assert "pauseOnLostFocus" not in ordinary.requested_client_settings
    assert "pauseOnLostFocus:" not in (ordinary.client_dir / "options.txt").read_text()
    assert "pauseOnLostFocus:false\n" in (enabled.client_dir / "options.txt").read_text()
    assert (
        json.loads(enabled.session_path.read_text())["requested_client_settings"][
            "pauseOnLostFocus"
        ]
        == "false"
    )
    assert enabled.client_actions_path == enabled.run_dir / "diagnostic-client-actions.ndjson"
    assert enabled.client_actions_path.read_bytes() == b""
    assert f"-Dneurocraft.fly.clientActionsPath={enabled.client_actions_path}" in (
        launcher._client_command(enabled)
    )
    assert json.loads(enabled.session_path.read_text())["diagnostic_client_actions_path"] == str(
        enabled.client_actions_path.relative_to(tmp_path)
    )


def test_high_rate_capture_remains_opt_in_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepared(tmp_path, monkeypatch)
    for fps, seconds in ((0, 120), (21, 120), (20, 0), (20, 721), (True, 120)):
        with pytest.raises(launcher.LauncherError, match="capture requires"):
            launcher.prepare_run(capture_fps=fps, capture_seconds=seconds)
    with pytest.raises(launcher.LauncherError, match="require --diagnostic-tools"):
        launcher.prepare_run(capture_fps=20)
    enabled = launcher.prepare_run(
        run_root=tmp_path / "recording-runs",
        accepted_eula=tmp_path / "accepted-eula.txt",
        diagnostic_tools=True,
        capture_fps=20,
        capture_seconds=660,
    )
    command = launcher._client_command(enabled)
    assert "-Dneurocraft.fly.captureFps=20" in command
    assert "-Dneurocraft.fly.captureSeconds=660" in command
    metadata = json.loads(enabled.session_path.read_text())
    assert metadata["requested_capture_fps"] == 20
    assert metadata["requested_capture_seconds"] == 660


def test_png_writer_choice_is_opt_in_bounded_and_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ordinary = _prepared(tmp_path, monkeypatch)
    assert ordinary.capture_png_compression is None
    assert not any("capturePngCompression" in arg for arg in launcher._client_command(ordinary))
    for bad in (-1, 10, True, 1.5):
        with pytest.raises(launcher.LauncherError, match="PNG compression"):
            launcher.prepare_run(diagnostic_tools=True, capture_png_compression=bad)
    with pytest.raises(launcher.LauncherError, match="requires --diagnostic-tools"):
        launcher.prepare_run(capture_png_compression=1)
    enabled = launcher.prepare_run(
        run_root=tmp_path / "compression-runs",
        accepted_eula=tmp_path / "accepted-eula.txt",
        diagnostic_tools=True,
        capture_png_compression=1,
    )
    assert "-Dneurocraft.fly.capturePngCompression=1" in launcher._client_command(enabled)
    assert json.loads(enabled.session_path.read_text())["requested_capture_png_compression"] == 1


@pytest.mark.parametrize(
    "width,height,max_fps",
    [(None, None, 40), (320, 240, 20), (640, 360, 30), (1920, 1080, 120)],
)
def test_client_display_options_are_per_run_and_exactly_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    width: int | None,
    height: int | None,
    max_fps: int,
) -> None:
    ordinary = _prepared(tmp_path, monkeypatch)
    configured = launcher.prepare_run(
        run_root=tmp_path / "display-runs",
        accepted_eula=tmp_path / "accepted-eula.txt",
        client_width=width,
        client_height=height,
        client_max_fps=max_fps,
    )
    for plan in (ordinary, configured):
        options = dict(
            line.split(":", 1)
            for line in (plan.client_dir / "options.txt").read_text().splitlines()
        )
        assert options["tutorialStep"] == "none"
        assert options["enableVsync"] == "false"
        assert options["inactivityFpsLimit"] == '"minimized"'
        assert options == json.loads(plan.session_path.read_text())["requested_client_settings"]
        # Later session updates retain the per-run request, not another run's settings.
        launcher._write_session(plan, state="stopped", identity=None)
        assert options == json.loads(plan.session_path.read_text())["requested_client_settings"]
    if width is None:
        assert "overrideWidth" not in configured.requested_client_settings
        assert "overrideHeight" not in configured.requested_client_settings
    else:
        assert configured.requested_client_settings["overrideWidth"] == str(width)
        assert configured.requested_client_settings["overrideHeight"] == str(height)
    assert configured.requested_client_settings["maxFps"] == str(max_fps)
    assert "overrideWidth" not in ordinary.requested_client_settings
    assert "overrideHeight" not in ordinary.requested_client_settings
    assert ordinary.requested_client_settings["maxFps"] == "60"
    assert launcher.CLIENT_SETTINGS["maxFps"] == "60"


@pytest.mark.parametrize(
    "width,height,max_fps",
    [
        (640, None, 60),
        (None, 360, 60),
        (319, 360, 60),
        (1921, 360, 60),
        (640, 239, 60),
        (640, 1081, 60),
        (True, 360, 60),
        (640, True, 60),
        (640.0, 360, 60),
        (640, 360.0, 60),
        (None, None, 19),
        (None, None, 121),
        (None, None, True),
        (None, None, 60.0),
    ],
)
def test_client_display_refuses_invalid_requests_before_creating_a_run(
    tmp_path: Path, width: int | None, height: int | None, max_fps: int
) -> None:
    runs = tmp_path / "runs"
    with pytest.raises(launcher.LauncherError, match="client"):
        launcher.prepare_run(
            run_root=runs,
            client_width=width,
            client_height=height,
            client_max_fps=max_fps,
        )
    assert not runs.exists()


def test_client_display_cli_reaches_prepare_without_launching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepared(tmp_path, monkeypatch)
    defaults = launcher.build_parser().parse_args([])
    assert (defaults.client_width, defaults.client_height, defaults.client_max_fps) == (
        None,
        None,
        60,
    )
    runs = tmp_path / "cli-display-runs"
    monkeypatch.setattr(
        launcher,
        "prepare_run",
        functools.partial(
            launcher.prepare_run, run_root=runs, accepted_eula=tmp_path / "accepted-eula.txt"
        ),
    )
    monkeypatch.setattr(launcher.signal, "signal", lambda *_: None)
    assert (
        launcher.main(
            [
                "--prepare-only",
                "--client-width",
                "640",
                "--client-height",
                "360",
                "--client-max-fps",
                "30",
            ]
        )
        == 0
    )
    session = json.loads(next(runs.glob("*/session.json")).read_text())
    assert session["state"] == "prepared"
    assert session["requested_client_settings"]["overrideWidth"] == "640"
    assert session["requested_client_settings"]["overrideHeight"] == "360"
    assert session["requested_client_settings"]["maxFps"] == "30"


def test_prepare_creates_fresh_loopback_only_offline_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _prepared(tmp_path, monkeypatch)
    eula = tmp_path / "accepted-eula.txt"
    second = launcher.prepare_run(
        run_root=tmp_path / "artifacts" / "live-interactive",
        accepted_eula=eula,
        now=datetime(2026, 8, 17, 20, 0, tzinfo=UTC),
    )

    assert first.run_dir != second.run_dir
    assert first.entity_id != second.entity_id
    assert first.session_epoch != second.session_epoch
    assert first.server_dir != first.client_dir
    assert not (first.server_dir / "world").exists()
    properties = dict(
        line.split("=", 1)
        for line in (first.server_dir / "server.properties").read_text().splitlines()
    )
    assert properties["server-ip"] == "127.0.0.1"
    assert properties["server-port"] == str(first.game_port)
    assert properties["online-mode"] == "false"
    assert properties["enforce-secure-profile"] == "false"
    assert properties["enable-rcon"] == "false"
    assert properties["enable-query"] == "false"
    assert (first.server_dir / "eula.txt").read_text().endswith("eula=true\n")
    client_options = (first.client_dir / "options.txt").read_text(encoding="utf-8")
    assert "onboardAccessibility:false" in client_options
    assert "skipMultiplayerWarning:true" in client_options

    session = json.loads(first.session_path.read_text(encoding="utf-8"))
    assert session["state"] == "prepared"
    assert session["authentication"] == "loopback-only development offline mode"
    assert "value omitted" in session["secret"]
    assert "G3" in session["claim_boundary"]


def test_commands_bind_exact_identity_without_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _prepared(tmp_path, monkeypatch)
    identity: dict[str, object] = {
        "run_id": plan.run_id,
        "session_epoch": plan.session_epoch,
        "entity_id": plan.entity_id,
        "graph_hash": "1" * 64,
        "adapter_hash": "2" * 64,
        "dynamics_hash": "3" * 64,
        "decoder_hash": "4" * 64,
        "observation_schema_hash": "5" * 64,
    }
    plan.identity_path.write_text(json.dumps(identity), encoding="utf-8")
    assert launcher._load_identity(plan.identity_path, plan) == identity

    server = launcher._server_command(plan, identity)
    client = launcher._client_command(plan)
    service = launcher._service_command(plan)
    all_arguments = "\n".join((*server, *client, *service))
    assert launcher.TOKEN_NAME not in all_arguments
    assert "/summon" not in all_arguments
    assert server[-1] == "runServer"
    assert client[-1] == "runClient"
    assert f"ws://127.0.0.1:{plan.bridge_port}/control" in all_arguments
    assert f"127.0.0.1:{plan.game_port}" in all_arguments
    assert str(plan.server_dir) in all_arguments
    assert str(plan.client_dir) in all_arguments
    assert "--identity-only" not in service
    assert "--identity-out" not in service


def test_identity_rejects_mismatch_and_bad_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _prepared(tmp_path, monkeypatch)
    identity: dict[str, object] = {
        "run_id": plan.run_id,
        "session_epoch": plan.session_epoch,
        "entity_id": plan.entity_id,
        "graph_hash": "1" * 64,
        "adapter_hash": "2" * 64,
        "dynamics_hash": "3" * 64,
        "decoder_hash": "4" * 64,
        "observation_schema_hash": "5" * 64,
    }
    identity["entity_id"] = "wrong"
    plan.identity_path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(launcher.LauncherError, match="does not match"):
        launcher._load_identity(plan.identity_path, plan)

    identity["entity_id"] = plan.entity_id
    identity["graph_hash"] = "not-a-hash"
    plan.identity_path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(launcher.LauncherError, match="not a SHA-256"):
        launcher._load_identity(plan.identity_path, plan)


def test_comparison_launch_binds_catalog_before_server_and_separates_epoch_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ordinary = _prepared(tmp_path, monkeypatch)
    plan = replace(
        ordinary,
        comparison_lanes=True,
        control_seed=2**64 - 1,
        controller_profile=launcher.FULL_MCNS_LOOM_PROGRAM_V2_PROFILE,
    )
    identity: dict[str, object] = {
        "run_id": plan.run_id,
        "entity_id": plan.entity_id,
        "session_epoch": plan.session_epoch,
        **{key: "a" * 64 for key in launcher.HASH_KEYS},
    }
    # Complete lane-entry semantics are separately enforced by LaneCatalog in the server.
    content = json.dumps({"schema": "neurocraft.lane_catalog.v1", "initial_identity": identity})
    plan.lane_catalog_path.write_text(content)
    digest = launcher._catalog_digest(plan.lane_catalog_path, identity)
    assert digest == sha256(content.encode()).hexdigest()
    with pytest.raises(launcher.LauncherError, match="verified catalog digest"):
        launcher._server_command(plan, identity)
    server = launcher._server_command(plan, identity, catalog_sha256=digest)
    assert f"-Dneurocraft.fly.laneCatalogSha256={digest}" in server
    assert f"-Dneurocraft.fly.epochDiagnosticsDirectory={plan.epoch_diagnostics_dir}" in server
    assert not any("brainDisplayPath" in arg for arg in server)
    command = launcher._service_command(plan)
    assert command[command.index("--control-seed") + 1] == "18446744073709551615"
    assert command[command.index("--epoch-diagnostics-dir") + 1] == str(plan.epoch_diagnostics_dir)
    assert "--comparison-lanes" in command
    assert "--comparison-lanes" not in launcher._identity_command(plan)
    assert "--comparison-lanes" not in launcher._service_command(ordinary)
    launcher._write_session(plan, state="interactive", identity=identity, catalog_sha256=digest)
    metadata = json.loads(plan.session_path.read_text())
    assert metadata["control_seed"] == "18446744073709551615"
    assert metadata["comparison_lanes"] is True
    assert metadata["lane_catalog_sha256"] == digest


@pytest.mark.parametrize("mutation", ["mismatch", "duplicate", "float_epoch", "oversize", "utf8"])
def test_comparison_catalog_refuses_ambiguous_launch_identity(
    tmp_path: Path, mutation: str
) -> None:
    identity: dict[str, object] = {"run_id": "a", "session_epoch": 7}
    catalog = {"schema": "neurocraft.lane_catalog.v1", "initial_identity": dict(identity)}
    if mutation == "mismatch":
        catalog["initial_identity"] = {"run_id": "b", "session_epoch": 7}
    elif mutation == "float_epoch":
        catalog["initial_identity"] = {"run_id": "a", "session_epoch": 7.0}
    data = json.dumps(catalog).encode()
    if mutation == "duplicate":
        data = data.replace(b'"run_id": "a"', b'"run_id": "b", "run_id": "a"')
    elif mutation == "oversize":
        data += b" " * 16385
    elif mutation == "utf8":
        data += b"\xff"
    path = tmp_path / "catalog.json"
    path.write_bytes(data)
    with pytest.raises(launcher.LauncherError, match="comparison catalog refused"):
        launcher._catalog_digest(path, identity)


def test_comparison_and_seed_rejected_before_creating_files(tmp_path: Path) -> None:
    with pytest.raises(launcher.LauncherError, match="registered comparison program profile"):
        launcher.prepare_run(
            controller_profile="pair_v1", comparison_lanes=True, run_root=tmp_path / "runs"
        )
    for seed in (-1, 2**64, True):
        with pytest.raises(launcher.LauncherError, match="unsigned 64-bit"):
            launcher.prepare_run(control_seed=seed, run_root=tmp_path / "runs")
    with pytest.raises(launcher.LauncherError, match="requires comparison lanes"):
        launcher.prepare_run(control_seed=123, run_root=tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_environment_keeps_token_out_of_files_and_rejects_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _prepared(tmp_path, monkeypatch)
    token = "DistinctSecretValue" * 3
    monkeypatch.delenv("PYTHONPATH", raising=False)
    environment = launcher._child_environment(token)
    assert environment[launcher.TOKEN_NAME] == token
    assert token.encode() not in plan.session_path.read_bytes()
    assert all(
        token.encode() not in path.read_bytes()
        for path in plan.run_dir.rglob("*")
        if path.is_file()
    )

    monkeypatch.setenv("PYTHONPATH", "/tmp/injected")
    with pytest.raises(launcher.LauncherError, match="injection"):
        launcher._child_environment(token)


def test_live_status_pattern_requires_expected_uuid_and_command() -> None:
    entity_id = "12345678-1234-4234-8234-123456789abc"
    pattern = re.compile(
        rf"bridge=ACTIVE loaded_flies=1.*entity_id={re.escape(entity_id)}"
        r".*command_seq=(?!none)"
    )
    assert pattern.search(
        f"bridge=ACTIVE loaded_flies=1 entity_id={entity_id} position=(0,65,0) command_seq=12"
    )
    assert not pattern.search(
        f"bridge=ACTIVE loaded_flies=1 entity_id={entity_id} position=(0,65,0) command_seq=none"
    )


def test_terminal_fault_pattern_covers_every_latching_controller_fault() -> None:
    """The watcher must fire on every fault the bridge can latch, and never on a
    healthy status line.

    Parsed from ControllerFault.java rather than restated: a live session latched
    DEADLINE_MISSED while the watcher's hand-written list only knew four names, so
    a fault name added to the mod without a matching pattern update must fail here.
    """

    controller_fault = (
        Path(launcher.__file__).resolve().parents[1]
        / "mod"
        / "src"
        / "main"
        / "java"
        / "dev"
        / "neurocraft"
        / "fly"
        / "control"
        / "ControllerFault.java"
    )
    source = controller_fault.read_text(encoding="utf-8")
    faults = dict(re.findall(r"^\s*([A-Z][A-Z_]*)\((true|false)\)", source, flags=re.MULTILINE))
    latching = sorted(name for name, paused in faults.items() if paused == "true")
    benign = sorted(name for name, paused in faults.items() if paused == "false")
    # Parse sanity: the enum must yield both classes, or the coverage claim is empty.
    assert "DEADLINE_MISSED" in latching
    assert "ACTIVE" in benign and "STARTING" in benign

    for name in latching:
        line = f"NeuroCraft bridge={name} loaded_flies=1 command_seq=242"
        match = launcher.TERMINAL_FAULT_PATTERN.search(line)
        assert match is not None, f"terminal-fault watcher misses latching fault {name}"
        assert match.group(1) == name
    for name in benign:
        line = f"NeuroCraft bridge={name} loaded_flies=1 command_seq=242"
        assert launcher.TERMINAL_FAULT_PATTERN.search(line) is None, (
            f"terminal-fault watcher would false-positive on non-latching state {name}"
        )
    # Defense-in-depth literals only: the bridge normalizes both codes to
    # INTERNAL_ERROR before any console line is printed, so this pins pattern
    # membership, not reachability under the current wiring.
    for code in ("RECEIPT_MISMATCH", "TRACE_FAILURE"):
        assert launcher.TERMINAL_FAULT_PATTERN.search(code)


def test_monitor_loop_polls_status_so_the_fault_watcher_stays_fed() -> None:
    """After the session is interactive, fault codes reach the server capture only
    through `neurocraftfly status` lines. The monitor loop must keep issuing them
    (the startup pollers stop at readiness), or the terminal-fault watcher goes
    blind and the operator keeps interacting with a dead, neutral fly."""

    assert launcher.MONITOR_STATUS_INTERVAL_SECONDS == 2.0

    source = Path(launcher.__file__).read_text(encoding="utf-8")
    monitor = source.index("while client is None or client.poll() is None:")
    cleanup = source.index("except KeyboardInterrupt as exc:", monitor)
    status_send = source.index('_send_server(server, "neurocraftfly status")', monitor)
    interval_use = source.index("MONITOR_STATUS_INTERVAL_SECONDS", monitor)
    assert monitor < status_send < cleanup
    assert monitor < interval_use < cleanup


class _RecordingServer:
    """Stand-in for the Minecraft server process that records console commands."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    def poll(self) -> None:
        return None

    @property
    def stdin(self) -> _RecordingServer:
        return self

    def write(self, payload: str) -> None:
        self.commands.append(payload.rstrip("\n"))

    def flush(self) -> None:
        return None


#: How far outside the outer shell the sealedness flood fill is allowed to travel. A
#: hole anywhere in the box lets the fill leak into this margin, which every geometry
#: assertion below then sees as an interior that is the wrong shape.
ARENA_LEAK_MARGIN = 4

_Box = tuple[tuple[int, int, int], tuple[int, int, int], str]


@functools.cache
def _arena_fill_boxes() -> tuple[_Box, ...]:
    """The launcher's `/fill` commands, parsed into ordered (low, high, block) boxes.

    Deliberately parsed rather than restated: the fly spawned on the glass roof for
    every run before this was pinned, because the observer's spawn is outside the
    sealed box and `neurocraftfly spawn` places the fly at the command source. A
    hard-coded copy of the bounds here would not have caught that, and would not catch
    a future arena resize either.
    """

    server = _RecordingServer()
    launcher._prepare_arena(server)  # type: ignore[arg-type]
    boxes: list[_Box] = []
    for command in server.commands:
        if not command.startswith("fill "):
            continue
        parts = command.split()
        x1, y1, z1, x2, y2, z2 = (int(part) for part in parts[1:7])
        boxes.append(
            (
                (min(x1, x2), min(y1, y2), min(z1, z2)),
                (max(x1, x2), max(y1, y2), max(z1, z2)),
                parts[7],
            )
        )
    return tuple(boxes)


@functools.cache
def _arena_blocks() -> Mapping[tuple[int, int, int], str]:
    """Apply the launcher's fill commands, in order, to a sparse world model."""

    blocks: dict[tuple[int, int, int], str] = {}
    for low, high, block in _arena_fill_boxes():
        for x in range(low[0], high[0] + 1):
            for y in range(low[1], high[1] + 1):
                for z in range(low[2], high[2] + 1):
                    if block == "minecraft:air":
                        blocks.pop((x, y, z), None)
                    else:
                        blocks[(x, y, z)] = block
    return MappingProxyType(blocks)


@functools.cache
def _reachable_air() -> frozenset[tuple[int, int, int]]:
    """Every cell the fly can reach from its spawn without passing through a block.

    Flood-filled from the spawn rather than computed from the fill arithmetic, and
    allowed to travel ARENA_LEAK_MARGIN blocks past the outer shell, so that an
    unsealed face, edge, or corner shows up as extra reachable cells instead of
    cancelling out of a bounds calculation.
    """

    blocks = _arena_blocks()
    boxes = _arena_fill_boxes()
    lows = tuple(min(low[axis] for low, _, _ in boxes) - ARENA_LEAK_MARGIN for axis in range(3))
    highs = tuple(max(high[axis] for _, high, _ in boxes) + ARENA_LEAK_MARGIN for axis in range(3))
    start = (math.floor(launcher.SPAWN_X), launcher.SPAWN_Y + 1, math.floor(launcher.SPAWN_Z))
    seen = {start}
    frontier = [start]
    while frontier:
        x, y, z = frontier.pop()
        for cell in (
            (x + 1, y, z),
            (x - 1, y, z),
            (x, y + 1, z),
            (x, y - 1, z),
            (x, y, z + 1),
            (x, y, z - 1),
        ):
            if cell in seen or cell in blocks:
                continue
            if any(not lows[axis] <= cell[axis] <= highs[axis] for axis in range(3)):
                continue
            seen.add(cell)
            frontier.append(cell)
    return frozenset(seen)


@functools.cache
def _arena_interior() -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """The bounding box of the air the fly can actually reach from its spawn."""

    air = _reachable_air()
    xs = [cell[0] for cell in air]
    ys = [cell[1] for cell in air]
    zs = [cell[2] for cell in air]
    return ((min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs)))


def _declared_interior_cells() -> set[tuple[int, int, int]]:
    """Every cell the launcher's arena constants claim is enclosed air."""

    half = launcher.ARENA_HALF_WIDTH
    return {
        (x, y, z)
        for x in range(-half, half + 1)
        for y in range(launcher.ARENA_INTERIOR_FLOOR_Y, launcher.ARENA_INTERIOR_CEILING_Y + 1)
        for z in range(-half, half + 1)
    }


def test_the_fly_is_spawned_inside_the_sealed_arena() -> None:
    """The fly must land in the enclosure, not on its roof.

    `neurocraftfly spawn` places the fly one block above the command source, and the
    launcher runs it as the observer. Minecraft refuses to spawn a player inside a
    sealed box, so before this was pinned the observer arrived on the glass ceiling at
    y=73 and the fly appeared at y~74 -- outside the cage, where it slid off the roof
    and flew away with nothing containing it.
    """

    (x_low, x_high), (y_low, y_high), (z_low, z_high) = _arena_interior()

    fly_y = launcher.SPAWN_Y + 1.0  # FlyCommands.spawn offsets one block up.
    assert x_low <= launcher.SPAWN_X <= x_high
    assert y_low <= launcher.SPAWN_Y <= y_high
    assert z_low <= launcher.SPAWN_Z <= z_high
    assert y_low <= fly_y <= y_high, (
        f"the fly spawns at y={fly_y}, outside the enclosed air {y_low}..{y_high}"
    )


def test_the_observer_viewpoint_is_inside_the_sealed_arena() -> None:
    (x_low, x_high), (y_low, y_high), (z_low, z_high) = _arena_interior()
    assert x_low <= launcher.VIEW_X <= x_high
    assert y_low <= launcher.VIEW_Y <= y_high
    assert z_low <= launcher.VIEW_Z <= z_high


def test_the_enclosed_air_volume_matches_the_declared_arena_dimensions() -> None:
    """The observable arena is pinned to literals (D-067).

    49 x 49 x 24: wide enough to hold the tightest measured escape-v2 turn circles
    (radius 10.2 blocks blocked-right, 16.9 blocks at the blocked-front forward command
    with the strongest measured yaw) with at least 7 blocks of clearance per side, where
    the old 23 x 23 x 8 box held neither, and 24 blocks tall so the ~2.0 s climb to the
    ceiling at the measured loom vertical command becomes ~4.6 s. Restated here as
    literals on purpose: the launcher derives its fills from constants, so a changed
    constant has to be re-declared here before the suite goes green.
    """

    assert _arena_interior() == ((-24, 24), (64, 87), (-24, 24))
    assert len(_reachable_air()) == 49 * 24 * 49


def test_the_arena_shell_is_sealed_against_a_flood_fill_from_the_fly_spawn() -> None:
    """A live fly escaped the single-layer glass box while the operator was placing and
    breaking blocks nearby (D-056), so the enclosure is pinned by reachability, not by
    arithmetic: air flood-filled from the fly's spawn cell reaches every declared
    interior cell and nothing else, even though it is free to travel ARENA_LEAK_MARGIN
    blocks past the outer shell. One hole in any face, edge, or corner fails this."""

    reachable = set(_reachable_air())
    declared = _declared_interior_cells()
    assert reachable - declared == set(), "the fly can reach air outside the sealed arena"
    assert declared - reachable == set(), "part of the declared interior is not reachable air"


def test_the_arena_shell_is_two_blocks_thick_and_transparent_on_every_face() -> None:
    """Sealed is not enough: one misplaced operator block-break must not open the arena,
    so every interior boundary cell carries ARENA_SHELL_BLOCKS solid blocks outward on
    all six faces -- the floor included, since the world below it is open air down to
    the superflat surface. The walls and ceiling must stay glass, or the operator cannot
    watch the fly through them."""

    blocks = _arena_blocks()
    (x_low, x_high), (y_low, y_high), (z_low, z_high) = _arena_interior()
    depths = range(1, launcher.ARENA_SHELL_BLOCKS + 1)
    for x in range(x_low, x_high + 1):
        for z in range(z_low, z_high + 1):
            for depth in depths:
                assert blocks.get((x, y_low - depth, z)) == "minecraft:smooth_stone", (
                    f"floor layer {depth} is open under ({x},{z})"
                )
                assert blocks.get((x, y_high + depth, z)) == "minecraft:glass", (
                    f"ceiling layer {depth} is open above ({x},{z})"
                )
    for y in range(y_low, y_high + 1):
        for z in range(z_low, z_high + 1):
            for depth in depths:
                assert blocks.get((x_low - depth, y, z)) == "minecraft:glass", (
                    f"-x wall layer {depth} is open at ({y},{z})"
                )
                assert blocks.get((x_high + depth, y, z)) == "minecraft:glass", (
                    f"+x wall layer {depth} is open at ({y},{z})"
                )
        for x in range(x_low, x_high + 1):
            for depth in depths:
                assert blocks.get((x, y, z_low - depth)) == "minecraft:glass", (
                    f"-z wall layer {depth} is open at ({x},{y})"
                )
                assert blocks.get((x, y, z_high + depth)) == "minecraft:glass", (
                    f"+z wall layer {depth} is open at ({x},{y})"
                )


def test_every_arena_fill_stays_under_the_vanilla_fill_cap() -> None:
    """A `/fill` over more than 32768 blocks is refused by the server and the refusal
    never reaches the launcher, so an oversized fill would leave a shell face missing
    from a box everything else still treats as sealed. The interior air exceeds the cap
    on its own and is emitted as slabs; this pins that every fill, not just those, fits."""

    for low, high, block in _arena_fill_boxes():
        volume = (high[0] - low[0] + 1) * (high[1] - low[1] + 1) * (high[2] - low[2] + 1)
        assert volume <= launcher.ARENA_MAX_FILL_BLOCKS, (
            f"{block} fill {low}..{high} covers {volume} blocks"
        )


def test_the_observer_is_moved_back_and_up_far_enough_to_frame_the_arena() -> None:
    """The enlarged box is only watchable if the observer can see across it: the far
    wall must fit inside even the vertical 70-degree default FOV (the horizontal field
    is wider on any normal aspect ratio), and the observer must end up behind and above
    the fly rather than on top of it."""

    (x_low, x_high), _, (_, z_high) = _arena_interior()
    assert launcher.VIEW_Z < launcher.SPAWN_Z
    assert launcher.VIEW_Y > launcher.SPAWN_Y
    framed = 2.0 * (z_high - launcher.VIEW_Z) * math.tan(math.radians(35.0))
    assert framed >= x_high - x_low + 1, f"only {framed:.1f} blocks of the far wall are in frame"


def test_the_launcher_moves_the_observer_inside_before_spawning_the_fly() -> None:
    """Ordering matters: a teleport after the spawn would leave the fly on the roof."""

    source = Path(launcher.__file__).read_text(encoding="utf-8")
    teleport = source.index("tp {username} {SPAWN_X} {SPAWN_Y} {SPAWN_Z}")
    spawn = source.index("run neurocraftfly spawn")
    assert teleport < spawn


def test_trm_profile_plan_binds_steering_fixture_and_brain_viewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    eula = tmp_path / "accepted-eula.txt"
    eula.write_text("eula=true\n", encoding="utf-8")
    plan = launcher.prepare_run(
        run_root=tmp_path / "artifacts" / "live-interactive",
        accepted_eula=eula,
        now=datetime(2026, 8, 17, 20, 0, tzinfo=UTC),
        controller_profile="trm_cloned_readout_v1",
        brain_viewer=True,
    )
    assert plan.viewer_port is not None
    assert plan.viewer_port not in (plan.bridge_port, plan.game_port)
    assert plan.telemetry_path is not None
    assert plan.telemetry_path.is_relative_to(plan.run_dir)
    assert plan.fixture == launcher.STEERING_FIXTURE

    identity_command = launcher._identity_command(plan)
    assert str(launcher.STEERING_FIXTURE) in identity_command
    profile_index = identity_command.index("--controller-profile")
    assert identity_command[profile_index + 1] == "trm_cloned_readout_v1"

    service_command = launcher._service_command(plan)
    telemetry_index = service_command.index("--presentation-telemetry-out")
    assert service_command[telemetry_index + 1] == str(plan.telemetry_path)

    viewer_command = launcher._viewer_command(plan)
    assert str(plan.viewer_port) in viewer_command
    assert str(plan.telemetry_path) in viewer_command

    session = json.loads(plan.session_path.read_text(encoding="utf-8"))
    assert session["controller_profile"] == "trm_cloned_readout_v1"
    assert "trained cloned readout" in session["fixture"]


def test_default_plan_keeps_pair_profile_and_no_viewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _prepared(tmp_path, monkeypatch)
    assert plan.controller_profile == "pair_v1"
    assert plan.viewer_port is None
    assert plan.telemetry_path is None
    assert plan.fixture == launcher.FIXTURE
    assert "--presentation-telemetry-out" not in launcher._service_command(plan)
    with pytest.raises(launcher.LauncherError, match="viewer port"):
        launcher._viewer_command(plan)
    session = json.loads(plan.session_path.read_text(encoding="utf-8"))
    assert session["controller_profile"] == "pair_v1"
    assert session["fixture"].startswith("14-node/22-edge")


def test_action_latency_flag_is_independent_of_the_brain_viewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The viewer is refused for every non-fixture profile, and those are exactly
    the profiles whose first-action latency needs measuring, so the latency file
    must not be gated on a viewer port."""

    plan = _prepared(tmp_path, monkeypatch, action_latency=True)
    assert plan.viewer_port is None
    assert plan.telemetry_path is None
    assert plan.action_latency_path is not None
    assert plan.action_latency_path.is_relative_to(plan.run_dir)

    command = launcher._service_command(plan)
    assert "--presentation-telemetry-out" not in command
    latency_index = command.index("--action-latency-out")
    assert command[latency_index + 1] == str(plan.action_latency_path)


def test_action_latency_defaults_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _prepared(tmp_path, monkeypatch)
    assert plan.action_latency_path is None
    assert "--action-latency-out" not in launcher._service_command(plan)
    assert not launcher.build_parser().parse_args([]).action_latency
    assert launcher.build_parser().parse_args(["--action-latency"]).action_latency


@pytest.mark.parametrize("kernel_threads", [None, 4])
def test_full_profile_plan_omits_the_fixture_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kernel_threads: int | None
) -> None:
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    artifact_dir = tmp_path / "full-artifact"
    artifact_dir.mkdir()
    (artifact_dir / "manifest.json").write_text("{}", encoding="utf-8")
    annotations = tmp_path / "neurotransmitters.feather"
    annotations.write_bytes(b"stub")
    monkeypatch.setattr(launcher, "FULL_ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr(launcher, "FULL_ANNOTATIONS_PATH", annotations)
    eula = tmp_path / "accepted-eula.txt"
    eula.write_text("eula=true\n", encoding="utf-8")
    plan = launcher.prepare_run(
        run_root=tmp_path / "artifacts" / "live-interactive",
        accepted_eula=eula,
        now=datetime(2026, 8, 17, 20, 0, tzinfo=UTC),
        controller_profile="full_mcns_wing_v1",
        kernel_threads=kernel_threads,
    )
    assert plan.fixture is None
    identity_command = launcher._identity_command(plan)
    assert "--fixture" not in identity_command
    profile_index = identity_command.index("--controller-profile")
    assert identity_command[profile_index + 1] == "full_mcns_wing_v1"
    assert "--fixture" not in launcher._service_command(plan)
    session = json.loads(plan.session_path.read_text(encoding="utf-8"))
    assert session["controller_profile"] == "full_mcns_wing_v1"
    assert session["fixture"].startswith("166,700-neuron")
    assert session["requested_kernel_threads"] == kernel_threads
    for command in (identity_command, launcher._service_command(plan)):
        assert ("--kernel-threads" in command) == (kernel_threads is not None)
        if kernel_threads is not None:
            assert command[command.index("--kernel-threads") + 1] == str(kernel_threads)


def test_full_profile_is_refused_when_the_artifact_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launcher, "FULL_ARTIFACT_DIR", tmp_path / "missing-artifact")
    eula = tmp_path / "accepted-eula.txt"
    eula.write_text("eula=true\n", encoding="utf-8")
    with pytest.raises(launcher.LauncherError, match="not present"):
        launcher.prepare_run(
            run_root=tmp_path / "runs",
            accepted_eula=eula,
            controller_profile="full_mcns_wing_v1",
        )


def test_brain_viewer_is_refused_for_the_full_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eula = tmp_path / "accepted-eula.txt"
    eula.write_text("eula=true\n", encoding="utf-8")
    with pytest.raises(launcher.LauncherError, match="steering fixture"):
        launcher.prepare_run(
            run_root=tmp_path / "runs",
            accepted_eula=eula,
            controller_profile="full_mcns_wing_v1",
            brain_viewer=True,
        )


def test_unknown_controller_profile_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eula = tmp_path / "accepted-eula.txt"
    eula.write_text("eula=true\n", encoding="utf-8")
    with pytest.raises(launcher.LauncherError, match="unknown controller profile"):
        launcher.prepare_run(
            run_root=tmp_path / "runs",
            accepted_eula=eula,
            controller_profile="scripted_baseline",
        )


def test_brain_viewer_is_refused_for_the_power_fixture_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eula = tmp_path / "accepted-eula.txt"
    eula.write_text("eula=true\n", encoding="utf-8")
    with pytest.raises(launcher.LauncherError, match="steering fixture"):
        launcher.prepare_run(
            run_root=tmp_path / "runs",
            accepted_eula=eula,
            controller_profile="pair_v1",
            brain_viewer=True,
        )


def test_eem_v2_profile_row_is_registered_fixture_free() -> None:
    fixture, description = launcher.CONTROLLER_PROFILES["full_mcns_eem_v2"]
    assert fixture is None
    assert "temporal-contrast" in description
    assert "full_mcns_eem_v2" in launcher.FULL_MCNS_PROFILES
    assert "full_mcns_eem_v2" in launcher.FULL_MCNS_EEM_PROFILES


def test_escape_profile_row_is_registered_fixture_free() -> None:
    fixture, description = launcher.CONTROLLER_PROFILES["full_mcns_escape_v1"]
    assert fixture is None
    assert "escape-reflex" in description
    assert "full_mcns_escape_v1" in launcher.FULL_MCNS_PROFILES
    assert "full_mcns_escape_v1" in launcher.FULL_MCNS_BODY_ANNOTATION_PROFILES
    assert "full_mcns_escape_v1" not in launcher.FULL_MCNS_EEM_PROFILES


def test_escape_v2_profile_row_is_registered_fixture_free() -> None:
    fixture, description = launcher.CONTROLLER_PROFILES["full_mcns_escape_v2"]
    assert fixture is None
    assert "transient-touch" in description
    assert "EEM escape v2" in description
    assert "full_mcns_escape_v2" in launcher.FULL_MCNS_PROFILES
    assert "full_mcns_escape_v2" in launcher.FULL_MCNS_BODY_ANNOTATION_PROFILES
    assert "full_mcns_escape_v2" in launcher.FULL_MCNS_ESCAPE_PROFILES
    assert "full_mcns_escape_v2" not in launcher.FULL_MCNS_EEM_PROFILES
    # The launcher's profile names are the service's, exactly.
    import scripts.run_connectome_service as service

    assert set(launcher.CONTROLLER_PROFILES) == set(service.CONTROLLER_PROFILES)


def test_loom_profile_row_is_registered_fixture_free() -> None:
    fixture, description = launcher.CONTROLLER_PROFILES["full_mcns_loom_v1"]
    assert fixture is None
    assert "approach-gated" in description
    assert "EEM loom v1" in description
    assert "full_mcns_loom_v1" in launcher.FULL_MCNS_PROFILES
    assert "full_mcns_loom_v1" in launcher.FULL_MCNS_BODY_ANNOTATION_PROFILES
    assert "full_mcns_loom_v1" in launcher.FULL_MCNS_JUMP_READOUT_PROFILES
    # The loom lane is its own row: it is not an escape or an eem generation.
    assert "full_mcns_loom_v1" not in launcher.FULL_MCNS_ESCAPE_PROFILES
    assert "full_mcns_loom_v1" not in launcher.FULL_MCNS_EEM_PROFILES
    assert launcher.FULL_MCNS_ESCAPE_PROFILES == (
        "full_mcns_escape_v1",
        "full_mcns_escape_v2",
    )


def test_program_profile_row_is_registered_fixture_free() -> None:
    fixture, description = launcher.CONTROLLER_PROFILES["full_mcns_loom_program_v1"]
    assert fixture is None
    assert "scripted body programs" in description
    assert "hand-chosen" in description
    # The description is printed on the live ready banner, so it states the lane as it
    # is: the mod admits program_action frames (lane M), and the launcher still refuses
    # a mod build without the decode arm. It said the opposite on the 2026-09-03
    # sessions, months after the mod-side lane landed.
    assert "the mod admits" in description
    assert "still refuses" in description
    assert "cannot admit" not in description
    assert "full_mcns_loom_program_v1" in launcher.FULL_MCNS_PROFILES
    assert "full_mcns_loom_program_v1" in launcher.FULL_MCNS_BODY_ANNOTATION_PROFILES
    assert "full_mcns_loom_program_v1" in launcher.FULL_MCNS_JUMP_READOUT_PROFILES
    assert launcher.FULL_MCNS_PROGRAM_PROFILES == (
        "full_mcns_loom_program_v1",
        "full_mcns_loom_program_v2",
        "full_mcns_events_program_v1",
        "full_mcns_interactions_program_v2",
    )
    # It is its own row, not an escape or an eem generation, and not the raw loom lane.
    assert "full_mcns_loom_program_v1" not in launcher.FULL_MCNS_ESCAPE_PROFILES
    assert "full_mcns_loom_program_v1" not in launcher.FULL_MCNS_EEM_PROFILES
    assert "full_mcns_loom_v1" not in launcher.FULL_MCNS_PROGRAM_PROFILES
    # The launcher's profile names are the service's, exactly.
    import scripts.run_connectome_service as service

    assert set(launcher.CONTROLLER_PROFILES) == set(service.CONTROLLER_PROFILES)


def test_program_lane_banner_carries_the_three_honesty_sentences() -> None:
    """The banner text is pinned, not paraphrased: it is the surface a frame grab shows."""

    banner = launcher.PROGRAM_LANE_BANNER.format(policy_hash="ab" * 32)
    assert "labeled readouts" in banner
    assert "scripted body programs" in banner
    assert "declared hand-chosen mapping" in banner
    assert f"policy_hash {'ab' * 32}" in banner
    assert "The mapping is engineering." in banner
    assert "The programs are scripted." in banner
    assert "The connectome is not deciding anything." in banner
    # Never blended with the OPERATOR lane's wording, and never a decision claim.
    # ("hand-chosen" is the mapping's own honest description and is required above; the
    # forbidden strings are the phrasings that would assert the organism decided.)
    assert "CONNECTOME NOT DRIVING" not in banner
    for forbidden in (
        "the fly decided",
        "the fly chose",
        "the connectome decided",
        "the connectome chose",
        "the connectome steers",
        "reacts to you",
        "the fly wanted",
    ):
        assert forbidden not in banner.lower()


def test_program_lane_refuses_to_launch_against_a_mod_that_cannot_admit_the_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mod with no ``program_action`` decode arm would fail closed on every frame.

    Detected by parsing the Java source, for the same reason ``TERMINAL_FAULT_PATTERN`` is:
    a hand-maintained flag would drift. The refusal clears itself when the mod-side lane
    lands, with no edit here.
    """

    without_arm = tmp_path / "ControlProtocolWithout.java"
    without_arm.write_text(
        'case "hello_ack" -> decodeAck(values);\ncase "action" -> decodeAction(values);\n',
        encoding="utf-8",
    )
    assert launcher._mod_admits_program_frames(without_arm) is False
    # A missing file is not evidence of a decoder.
    assert launcher._mod_admits_program_frames(tmp_path / "absent.java") is False

    with_arm = tmp_path / "ControlProtocolWith.java"
    with_arm.write_text(
        'case "action" -> decodeAction(values);\n'
        'case "program_action" -> decodeProgramAction(values);\n',
        encoding="utf-8",
    )
    assert launcher._mod_admits_program_frames(with_arm) is True

    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "CONTROL_PROTOCOL_SOURCE", without_arm)
    eula = tmp_path / "accepted-eula.txt"
    eula.write_text("eula=true\n", encoding="utf-8")
    with pytest.raises(launcher.LauncherError, match="NOT runnable live"):
        launcher.prepare_run(
            run_root=tmp_path / "runs",
            accepted_eula=eula,
            controller_profile="full_mcns_loom_program_v1",
        )
    # The sentence names the reason, the missing lane, and the offline alternative.
    assert "program_action" in launcher.PROGRAM_LANE_NOT_RUNNABLE
    assert "fail closed" in launcher.PROGRAM_LANE_NOT_RUNNABLE
    assert "probe_program_policy.py" in launcher.PROGRAM_LANE_NOT_RUNNABLE
    # The refusal is about the tree being launched, not about the lane being unwritten.
    assert "has landed on main" in launcher.PROGRAM_LANE_NOT_RUNNABLE
    # Every other profile is unaffected by the mod's frame vocabulary.
    launcher.prepare_run(
        run_root=tmp_path / "runs",
        accepted_eula=eula,
        controller_profile="pair_v1",
    )


def test_program_plan_writes_a_diagnostic_decisions_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The body-program lane always records its decisions; every other lane never does."""

    artifact_dir = tmp_path / "full-artifact"
    artifact_dir.mkdir()
    (artifact_dir / "manifest.json").write_text("{}", encoding="utf-8")
    annotations = tmp_path / "neurotransmitters.feather"
    annotations.write_bytes(b"stub")
    body_annotations = tmp_path / "body-annotations.feather"
    body_annotations.write_bytes(b"stub")
    protocol = tmp_path / "ControlProtocol.java"
    protocol.write_text('case "program_action" -> decodeProgramAction(values);\n', "utf-8")
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "CONTROL_PROTOCOL_SOURCE", protocol)
    monkeypatch.setattr(launcher, "FULL_ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr(launcher, "FULL_ANNOTATIONS_PATH", annotations)
    monkeypatch.setattr(launcher, "FULL_BODY_ANNOTATIONS_PATH", body_annotations)
    eula = tmp_path / "accepted-eula.txt"
    eula.write_text("eula=true\n", encoding="utf-8")
    plan = launcher.prepare_run(
        run_root=tmp_path / "runs",
        accepted_eula=eula,
        controller_profile="full_mcns_loom_program_v1",
    )
    assert plan.program_decisions_path == plan.run_dir / "program-decisions.ndjson"
    command = launcher._service_command(plan)
    assert "--program-decisions-out" in command
    assert str(plan.program_decisions_path) in command

    raw_plan = launcher.prepare_run(
        run_root=tmp_path / "runs",
        accepted_eula=eula,
        controller_profile="full_mcns_loom_v1",
    )
    assert raw_plan.program_decisions_path is None
    assert "--program-decisions-out" not in launcher._service_command(raw_plan)


def test_program_plan_accepts_the_action_latency_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``make live-demo-program`` passes ``--action-latency``; the plan must carry it.

    The first live program-lane sessions died on latency bursts the mod's own bounded
    latency log could not characterise, so this lane records every frame's
    compute_ns / serialization_ns / write_ns split. It is diagnostic, not evidence, and
    the path stays inside the run directory.
    """

    artifact_dir = tmp_path / "full-artifact"
    artifact_dir.mkdir()
    (artifact_dir / "manifest.json").write_text("{}", encoding="utf-8")
    annotations = tmp_path / "neurotransmitters.feather"
    annotations.write_bytes(b"stub")
    body_annotations = tmp_path / "body-annotations.feather"
    body_annotations.write_bytes(b"stub")
    protocol = tmp_path / "ControlProtocol.java"
    protocol.write_text('case "program_action" -> decodeProgramAction(values);\n', "utf-8")
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "CONTROL_PROTOCOL_SOURCE", protocol)
    monkeypatch.setattr(launcher, "FULL_ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr(launcher, "FULL_ANNOTATIONS_PATH", annotations)
    monkeypatch.setattr(launcher, "FULL_BODY_ANNOTATIONS_PATH", body_annotations)
    eula = tmp_path / "accepted-eula.txt"
    eula.write_text("eula=true\n", encoding="utf-8")
    plan = launcher.prepare_run(
        run_root=tmp_path / "runs",
        accepted_eula=eula,
        controller_profile="full_mcns_loom_program_v1",
        action_latency=True,
    )
    assert plan.action_latency_path == plan.run_dir / "action-latency.ndjson"
    command = launcher._service_command(plan)
    latency_index = command.index("--action-latency-out")
    assert command[latency_index + 1] == str(plan.action_latency_path)


def test_loom_plan_requires_the_body_annotations_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the artifact and neurotransmitters stubbed present, the missing
    body-annotations table is the refusal, and it names the loom profile."""

    artifact_dir = tmp_path / "full-artifact"
    artifact_dir.mkdir()
    (artifact_dir / "manifest.json").write_text("{}", encoding="utf-8")
    annotations = tmp_path / "neurotransmitters.feather"
    annotations.write_bytes(b"stub")
    monkeypatch.setattr(launcher, "FULL_ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr(launcher, "FULL_ANNOTATIONS_PATH", annotations)
    monkeypatch.setattr(
        launcher, "FULL_BODY_ANNOTATIONS_PATH", tmp_path / "absent-annotations.feather"
    )
    eula = tmp_path / "accepted-eula.txt"
    eula.write_text("eula=true\n", encoding="utf-8")
    with pytest.raises(launcher.LauncherError, match="full_mcns_loom_v1 is a local-only"):
        launcher.prepare_run(
            run_root=tmp_path / "runs",
            accepted_eula=eula,
            controller_profile="full_mcns_loom_v1",
        )


def test_escape_v2_plan_requires_the_body_annotations_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the artifact and neurotransmitters stubbed present, the missing
    body-annotations table is the refusal, and it names the escape-v2 profile."""

    artifact_dir = tmp_path / "full-artifact"
    artifact_dir.mkdir()
    (artifact_dir / "manifest.json").write_text("{}", encoding="utf-8")
    annotations = tmp_path / "neurotransmitters.feather"
    annotations.write_bytes(b"stub")
    monkeypatch.setattr(launcher, "FULL_ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr(launcher, "FULL_ANNOTATIONS_PATH", annotations)
    monkeypatch.setattr(
        launcher, "FULL_BODY_ANNOTATIONS_PATH", tmp_path / "absent-annotations.feather"
    )
    eula = tmp_path / "accepted-eula.txt"
    eula.write_text("eula=true\n", encoding="utf-8")
    with pytest.raises(launcher.LauncherError, match="full_mcns_escape_v2 is a local-only"):
        launcher.prepare_run(
            run_root=tmp_path / "runs",
            accepted_eula=eula,
            controller_profile="full_mcns_escape_v2",
        )


def test_escape_plan_requires_the_body_annotations_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the artifact and neurotransmitters stubbed present, the missing
    body-annotations table is the refusal, and it names the escape profile."""

    artifact_dir = tmp_path / "full-artifact"
    artifact_dir.mkdir()
    (artifact_dir / "manifest.json").write_text("{}", encoding="utf-8")
    annotations = tmp_path / "neurotransmitters.feather"
    annotations.write_bytes(b"stub")
    monkeypatch.setattr(launcher, "FULL_ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr(launcher, "FULL_ANNOTATIONS_PATH", annotations)
    monkeypatch.setattr(
        launcher, "FULL_BODY_ANNOTATIONS_PATH", tmp_path / "absent-annotations.feather"
    )
    eula = tmp_path / "accepted-eula.txt"
    eula.write_text("eula=true\n", encoding="utf-8")
    with pytest.raises(launcher.LauncherError, match="full_mcns_escape_v1 is a local-only"):
        launcher.prepare_run(
            run_root=tmp_path / "runs",
            accepted_eula=eula,
            controller_profile="full_mcns_escape_v1",
        )


def test_eem_v2_plan_requires_the_body_annotations_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the artifact and neurotransmitters stubbed present, the missing
    body-annotations table is the refusal, and it names the v2 profile."""

    artifact_dir = tmp_path / "full-artifact"
    artifact_dir.mkdir()
    (artifact_dir / "manifest.json").write_text("{}", encoding="utf-8")
    annotations = tmp_path / "neurotransmitters.feather"
    annotations.write_bytes(b"stub")
    monkeypatch.setattr(launcher, "FULL_ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr(launcher, "FULL_ANNOTATIONS_PATH", annotations)
    monkeypatch.setattr(
        launcher, "FULL_BODY_ANNOTATIONS_PATH", tmp_path / "absent-annotations.feather"
    )
    eula = tmp_path / "accepted-eula.txt"
    eula.write_text("eula=true\n", encoding="utf-8")
    with pytest.raises(launcher.LauncherError, match="full_mcns_eem_v2 is a local-only"):
        launcher.prepare_run(
            run_root=tmp_path / "runs",
            accepted_eula=eula,
            controller_profile="full_mcns_eem_v2",
        )


def test_circuit_playground_is_opt_in_and_records_closeup_scene_and_camera(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ordinary = _prepared(tmp_path, monkeypatch)
    enabled = launcher.prepare_run(
        run_root=tmp_path / "playground-runs",
        accepted_eula=tmp_path / "accepted-eula.txt",
        circuit_playground=True,
    )
    assert not ordinary.circuit_playground and not ordinary.showcase_world
    assert enabled.circuit_playground and enabled.showcase_world
    assert enabled.showcase_commands == launcher._showcase_commands(circuit_playground=True)
    launcher._write_session(enabled, state="stopped", identity=None)
    metadata = json.loads(enabled.session_path.read_text())
    assert metadata["circuit_playground"] is True
    x, y, z, yaw, pitch = metadata["starting_view"]
    assert y == launcher.ARENA_INTERIOR_FLOOR_Y + 1
    assert z < launcher.SPAWN_Z and yaw == 0 and -20 < pitch < 0
    # Grounded eye to initial fly centre is within native brush reach. This is
    # scene geometry, not an assertion that a moving fly was manually brushed.
    distance = math.dist(
        (x, y + 1.62, z), (launcher.SPAWN_X, launcher.SPAWN_Y + 1.11, launcher.SPAWN_Z)
    )
    assert distance < 3
    parsed = launcher.build_parser().parse_args(["--circuit-playground"])
    assert parsed.circuit_playground


def test_windows_child_env_keeps_required_paths_and_discards_unrelated_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in ("PYTHONPATH", "PYTHONHOME", "JAVA_TOOL_OPTIONS", "JDK_JAVA_OPTIONS", "GRADLE_OPTS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SystemRoot", "C:/Windows")
    monkeypatch.setenv("USERPROFILE", "C:/NeuroCraftTestProfile")
    monkeypatch.setenv("LOCALAPPDATA", "C:/NeuroCraftTestProfile/AppData/Local")
    monkeypatch.setenv("TEMP", "C:/NeuroCraftTestProfile/AppData/Local/Temp")
    monkeypatch.setenv("UNRELATED_PRIVATE_VALUE", "hidden")
    environment = launcher._child_environment("test-token")
    normalized = {key.upper(): value for key, value in environment.items()}
    assert normalized["SYSTEMROOT"] == "C:/Windows"
    assert environment["USERPROFILE"] == "C:/NeuroCraftTestProfile"
    assert environment["LOCALAPPDATA"] == "C:/NeuroCraftTestProfile/AppData/Local"
    assert environment["TEMP"] == "C:/NeuroCraftTestProfile/AppData/Local/Temp"
    assert environment["PYTHONIOENCODING"] == "utf-8"
    assert "UNRELATED_PRIVATE_VALUE" not in environment


def test_windows_uses_batch_wrapper_only_for_dev_launches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _prepared(tmp_path, monkeypatch)
    identity = {
        name: "a" * 64
        for name in (
            "graph_hash",
            "adapter_hash",
            "dynamics_hash",
            "decoder_hash",
            "observation_schema_hash",
        )
    }
    monkeypatch.setattr(launcher, "_windows", lambda: True)
    assert launcher._server_command(plan, identity)[0] == "gradlew.bat"
    assert launcher._client_command(plan)[0] == "gradlew.bat"


def test_managed_jvms_use_new_private_temp_directory_in_both_launch_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import stat
    from types import SimpleNamespace

    plan = _prepared(tmp_path, monkeypatch)
    private = plan.run_dir / "jvm-tmp"
    assert private.is_dir() and not private.is_symlink()
    if os.name != "nt":
        assert stat.S_IMODE(private.stat().st_mode) == 0o700
        assert stat.S_IMODE(plan.run_dir.stat().st_mode) == 0o700
    identity = {
        name: "a" * 64
        for name in (
            "graph_hash",
            "adapter_hash",
            "dynamics_hash",
            "decoder_hash",
            "observation_schema_hash",
        )
    }
    option = "-Djava.io.tmpdir=" + str(private)
    assert option in launcher._server_command(plan, identity)
    assert option in launcher._client_command(plan)
    monkeypatch.setattr(
        launcher,
        "_installed_mod_module",
        lambda: SimpleNamespace(
            java_executable=lambda: "java",
            java_runtime=SimpleNamespace(server_arguments=lambda server: ["fixture-KnotServer"]),
        ),
    )
    assert option in launcher._server_command(
        replace(plan, installed_runtime=tmp_path / "installed-runtime"), identity
    )
    with pytest.raises(FileExistsError):
        launcher._prepare_jvm_tmp(plan.run_dir)


def test_windows_private_temp_requires_documented_acl_capable_cpython(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launcher, "_windows", lambda: True)
    monkeypatch.setattr(launcher.sys, "version_info", (3, 12, 0))
    with pytest.raises(launcher.LauncherError, match=r"CPython 3\.13"):
        launcher._prepare_jvm_tmp(tmp_path)
    assert not (tmp_path / "jvm-tmp").exists()
