"""Comparison CLI with tiny real controllers; no full artifact or Minecraft process."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from neurocraft_controller.brain_display import BrainDisplayBinding
from neurocraft_controller.full_brain_controller import (
    FullBrainController,
    FullBrainLoomV2Configuration,
)
from neurocraft_controller.loom_senses_v2 import bind_embodied_senses_loom_v2
from neurocraft_controller.program_policy import ProgramSelectingController
from neurocraft_neural.full_controls import FullControlKind, build_control_view
from neurocraft_neural.full_engine import ArrayNativeEngine
from neurocraft_protocol import (
    WEBSOCKET_SUBPROTOCOL,
    Hello,
    HelloAck,
    PeerRole,
    ProgramAction,
    SensorFrameV2,
    canonical_json_bytes,
    encode_message,
    parse_message,
)
from neurocraft_service.lane_sessions import LaneSessionDispatcher, create_lane_server
from websockets.asyncio.client import connect
from websockets.typing import Subprotocol

import scripts.run_connectome_service as cli
from scripts.run_interactive_demo import POLICY_HASH_PATTERN
from tests.controller.test_brain_display import TinySpace
from tests.controller.test_embodied_senses import _observation, _sensors
from tests.protocol.helpers import program_receipt

INITIAL_EPOCH = 1_783_000_000_000_000_007
TOKEN = "T" * 43


@pytest.mark.parametrize(
    ("epoch_delta", "active_lane", "active_identity_delta", "expected"),
    [
        (-1, None, None, True),
        (0, "original", 0, True),
        (0, None, None, False),  # Pending, failed, or closed initial ownership.
        (0, "shuffled_weight", 0, False),
        (0, "original", 1, False),
        (1, None, None, False),
        (1, "original", 1, False),
    ],
)
def test_initial_rewarm_requires_unclaimed_or_exact_active_initial_original(
    epoch_delta: int, active_lane: str | None, active_identity_delta: int | None, expected: bool
) -> None:
    from tests.protocol.helpers import identity

    initial = identity(session_epoch=INITIAL_EPOCH)
    dispatcher = SimpleNamespace(
        last_claimed_epoch=INITIAL_EPOCH + epoch_delta,
        active_lane=active_lane,
        active_identity=None
        if active_identity_delta is None
        else replace(initial, session_epoch=INITIAL_EPOCH + active_identity_delta),
    )
    assert cli._initial_comparison_warm_allowed(dispatcher, initial) is expected


def _arguments(directory: Path, *extra: str) -> list[str]:
    return [
        "--comparison-lanes",
        "--controller-profile",
        cli.FULL_MCNS_LOOM_PROGRAM_V2_PROFILE,
        "--session-epoch",
        str(INITIAL_EPOCH),
        "--port",
        "0",
        "--lane-catalog-out",
        str(directory / "lanes.json"),
        "--epoch-diagnostics-dir",
        str(directory / "epochs"),
        *extra,
    ]


def _small_core(space: TinySpace, args: argparse.Namespace) -> FullBrainController:
    base = space.controller()
    assert base.embodied_senses is not None and base.escape_binding is not None
    senses = bind_embodied_senses_loom_v2(base.embodied_senses.populations)
    control = (
        None
        if args.mode == "faithful"
        else build_control_view(
            base.engine.view,
            kind=FullControlKind(args.mode),
            seed=args.control_seed,
            artifact_manifest_sha256=base.identity.graph_hash,
            raw_counts=np.tile([3, 5], base.engine.view.node_count),
            edge_signs=np.tile([1, -1], base.engine.view.node_count),
        )
    )
    return FullBrainController(
        run_id=args.run_id,
        entity_id=args.entity_id,
        session_epoch=args.session_epoch,
        configuration=FullBrainLoomV2Configuration(
            mode=args.mode, encoder_identity_sha256=senses.encoder.identity_sha256
        ),
        engine=ArrayNativeEngine(base.engine.view if control is None else control.view),
        sensory_binding=base._sensory_binding,
        motor_binding=base._motor_binding,
        embodied=senses,
        escape=replace(base.escape_binding, encoder_identity_sha256=senses.encoder.identity_sha256),
        steering=base.steering_binding,
        control=control,
    )


@pytest.mark.parametrize(
    "extra",
    [
        ["--controller-profile", "pair_v1"],
        ["--controller-profile", "full_mcns_loom_program_v1"],
        ["--mode", "lesion"],
        ["--full-uncalibrated"],
        ["--fixture", "other.json"],
        ["--trace-out", "trace"],
        ["--trace-manifest", "manifest"],
        ["--transition-forces", "forces"],
        ["--presentation-telemetry-out", "viewer"],
        ["--control-catalog-out", "interventions"],
        ["--lesion-body-id", "1"],
        ["--stimulate", "1=0.2"],
        ["--declare-lesion", "x=1"],
        ["--declare-stimulation", "x=1=0.2"],
        ["--identity-only"],
        ["--control-seed", "-1"],
        ["--control-seed", str(1 << 64)],
        ["--session-epoch", "-1"],
        ["--session-epoch", str(1 << 63)],
    ],
)
def test_incompatible_comparison_flags_refuse_before_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: list[str]
) -> None:
    def forbidden(args: argparse.Namespace) -> None:
        pytest.fail("incompatible invocation reached controller construction")

    monkeypatch.setattr(cli, "build_controller", forbidden)
    with pytest.raises(SystemExit):
        cli.main(_arguments(tmp_path, *extra))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "arguments",
    [
        ["--lane-catalog-out", "catalog"],
        ["--epoch-diagnostics-dir", "epochs"],
        ["--control-seed", "8"],
        ["--comparison-lanes", "--controller-profile", cli.FULL_MCNS_LOOM_PROGRAM_V2_PROFILE],
    ],
)
def test_standalone_or_incomplete_comparison_options_refuse_before_build(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    monkeypatch.setattr(cli, "build_controller", lambda args: pytest.fail("unexpected build"))
    with pytest.raises(SystemExit):
        cli.main(arguments)


@pytest.mark.parametrize("existing", ["catalog", "epoch", "identity"])
def test_existing_outputs_are_preserved_before_expensive_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: str
) -> None:
    path = {
        "catalog": tmp_path / "lanes.json",
        "epoch": tmp_path / "epochs" / str(INITIAL_EPOCH),
        "identity": tmp_path / "identity.json",
    }[existing]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("preserved")
    monkeypatch.setattr(cli, "build_controller", lambda args: pytest.fail("unexpected build"))
    with pytest.raises(SystemExit, match="refusing"):
        cli.main(_arguments(tmp_path, "--identity-out", str(tmp_path / "identity.json")))
    assert path.read_text() == "preserved"


def test_main_catalog_and_epoch_diagnostics_follow_actual_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    space = TinySpace()
    builds: list[argparse.Namespace] = []
    binding_calls: list[str] = []
    controllers: list[ProgramSelectingController] = []
    seed = (1 << 64) - 1

    def build(args: argparse.Namespace) -> ProgramSelectingController:
        builds.append(args)
        controller = ProgramSelectingController(_small_core(space, args))
        controllers.append(controller)
        return controller

    def binding(
        args: argparse.Namespace, original: ProgramSelectingController
    ) -> BrainDisplayBinding:
        binding_calls.append(original.controller.control_kind)
        return space.select()

    async def run(dispatcher: LaneSessionDispatcher) -> None:
        issued: dict[tuple[int, int], ProgramAction] = {}
        catalog = json.loads((tmp_path / "lanes.json").read_bytes())
        assert catalog["initial_identity"] == controllers[0].identity.to_dict()
        assert json.loads((tmp_path / "identity.json").read_bytes()) == catalog["initial_identity"]
        assert len({lane["identity"]["dynamics_hash"] for lane in catalog["lanes"]}) == 3
        assert set(catalog) == {"schema", "initial_identity", "lanes"}
        assert catalog["schema"] == "neurocraft.lane_catalog.v1"
        assert (tmp_path / "lanes.json").read_bytes() == canonical_json_bytes(catalog) + b"\n"
        sockets = []
        try:
            async with create_lane_server(dispatcher) as server:
                for offset, lane_index in enumerate((0, 1, 2, 0)):
                    lane = dispatcher.lanes[lane_index]
                    current = replace(lane.identity_template, session_epoch=INITIAL_EPOCH + offset)
                    socket = await connect(
                        f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
                        subprotocols=[Subprotocol(WEBSOCKET_SUBPROTOCOL)],
                    )
                    sockets.append(socket)
                    hello = Hello.create(
                        identity=current, role=PeerRole.MINECRAFT_SERVER, token=TOKEN
                    )
                    await socket.send(encode_message(hello))
                    ack = parse_message(await asyncio.wait_for(socket.recv(), 3))
                    assert isinstance(ack, HelloAck)
                    assert ack.identity == current and ack.verify(
                        TOKEN, client_nonce=hello.client_nonce
                    )
                    observed = _observation(
                        controllers[lane_index].controller,
                        0,
                        SensorFrameV2.from_sensor_frame(_sensors(forward=1.0), (2,) * 6),
                    )
                    observed = replace(observed, game_tick=100 + offset * 10)
                    await socket.send(encode_message(observed))
                    command = parse_message(await asyncio.wait_for(socket.recv(), 3))
                    assert isinstance(command, ProgramAction)
                    assert command.identity == current and command.command_seq == 0
                    issued[(offset, 0)] = command
                    await socket.send(encode_message(program_receipt(command)))
                    # A second command proves the receipt passed its serialized sink.
                    await socket.send(
                        encode_message(
                            replace(observed, observation_seq=1, game_tick=111 + offset * 10)
                        )
                    )
                    second = parse_message(await asyncio.wait_for(socket.recv(), 3))
                    assert isinstance(second, ProgramAction)
                    issued[(offset, 1)] = second
        finally:
            await dispatcher.close()
            for socket in sockets:
                await socket.close()
        for offset, lane_index in enumerate((0, 1, 2, 0)):
            directory = tmp_path / "epochs" / str(INITIAL_EPOCH + offset)
            metadata = json.loads((directory / "epoch.json").read_bytes())
            assert metadata["identity"]["session_epoch"] == INITIAL_EPOCH + offset
            assert metadata["name"] == dispatcher.lanes[lane_index].name
            assert metadata["controller_profile"] == cli.FULL_MCNS_LOOM_PROGRAM_V2_PROFILE
            assert metadata["diagnostic_only"] is True
            assert metadata["seed"] == (None if lane_index == 0 else str(seed))
            assert metadata["control_sha256"] == catalog["lanes"][lane_index]["control_sha256"]
            assert set(catalog["lanes"][lane_index]) == {
                "name",
                "identity",
                "control_kind",
                "control_sha256",
                "seed",
            }
            decisions = [
                json.loads(line)
                for line in (directory / "program-decisions.ndjson").read_bytes().splitlines()
            ]
            assert [line["kind"] for line in decisions].count("issued") == 2
            assert any(line["kind"] == "receipt" for line in decisions)
            latency = [
                json.loads(line)
                for line in (directory / "action-latency.ndjson").read_bytes().splitlines()
            ]
            assert len(latency) == 2
            assert all(
                line["controller_profile"] == cli.FULL_MCNS_LOOM_PROGRAM_V2_PROFILE
                for line in decisions + latency
            )
            display = json.loads((directory / "brain-display.json").read_bytes())
            assert display["identity"] == metadata["identity"]
            assert display["presentation_only"] is True
            # The nonblocking presentation publisher may drop a contended offer.
            assert display["command_seq"] in {0, 1}
            assert display["state_digest"] == issued[(offset, display["command_seq"])].state_digest

    monkeypatch.setattr(cli, "build_controller", build)
    monkeypatch.setattr(cli, "_comparison_display_binding", binding)
    monkeypatch.setattr(cli, "_serve_comparison", run)
    monkeypatch.setenv("NEUROCRAFT_RUN_TOKEN", TOKEN)
    markers = [tmp_path / name for name in ("single-program", "single-latency", "single-display")]
    result = cli.main(
        _arguments(
            tmp_path,
            "--identity-out",
            str(tmp_path / "identity.json"),
            "--program-decisions-out",
            str(markers[0]),
            "--action-latency-out",
            str(markers[1]),
            "--brain-display-out",
            str(markers[2]),
            "--control-seed",
            str(seed),
        )
    )
    assert result == 0
    policy_match = POLICY_HASH_PATTERN.search(capsys.readouterr().out)
    assert policy_match is not None
    assert policy_match.group(1) == controllers[0].policy_hash
    assert [args.mode for args in builds] == ["faithful", "shuffled_weight", "no_input"]
    assert all(args.session_epoch == INITIAL_EPOCH and args.control_seed == seed for args in builds)
    assert all(not args.full_uncalibrated for args in builds)
    assert binding_calls == ["faithful"]
    assert not any(path.exists() for path in markers)
    assert TOKEN not in (tmp_path / "lanes.json").read_text()


@pytest.mark.parametrize("failure", [False, True])
def test_comparison_listener_always_closes_dispatcher(
    monkeypatch: pytest.MonkeyPatch, failure: bool
) -> None:
    events: list[str] = []

    class Dispatcher:
        async def close(self) -> None:
            events.append("dispatcher_closed")

    class Socket:
        def getsockname(self) -> tuple[str, int]:
            return "127.0.0.1", 1234

    class Server:
        def __init__(self) -> None:
            self.sockets = [Socket()]

        async def serve_forever(self) -> None:
            events.append("serving")
            if failure:
                raise RuntimeError("synthetic listener failure")

    @asynccontextmanager
    async def listener(dispatcher: object) -> AsyncIterator[Server]:
        try:
            yield Server()
        finally:
            events.append("listener_closed")

    monkeypatch.setattr(cli, "create_lane_server", listener)
    if failure:
        with pytest.raises(RuntimeError, match="synthetic"):
            asyncio.run(cli._serve_comparison(Dispatcher()))  # type: ignore[arg-type]
    else:
        asyncio.run(cli._serve_comparison(Dispatcher()))  # type: ignore[arg-type]
    assert events == ["serving", "listener_closed", "dispatcher_closed"]


def test_full_builder_passes_requested_control_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    args = cli.build_parser().parse_args(_arguments(Path("unused"), "--control-seed", "37"))
    args.mode = "no_input"
    calls: list[dict[str, Any]] = []
    expected = _small_core(TinySpace(), args)

    def from_artifact(**kwargs: Any) -> FullBrainController:
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(FullBrainController, "from_artifact", from_artifact)
    assert cli._build_full_brain_controller(args) is expected
    assert calls[0]["control_seed"] == 37
    assert calls[0]["configuration"].mode == "no_input"
    assert calls[0]["calibration_path"] == cli.DEFAULT_FULL_LOOM_V2_CALIBRATION


def test_existing_later_epoch_is_refused_without_truncating_diagnostics(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    args = cli.build_parser().parse_args(_arguments(tmp_path))
    args.epoch_diagnostics_dir.mkdir()
    controller = ProgramSelectingController(_small_core(TinySpace(), args))
    from neurocraft_service.lane_sessions import LaneSpec

    lane = LaneSpec("original", controller)
    hooks = cli._comparison_hooks_factory(args, None)
    hooks(lane, controller.identity)
    path = args.epoch_diagnostics_dir / str(INITIAL_EPOCH) / "epoch.json"
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        hooks(lane, controller.identity)
    assert path.read_bytes() == before
    stages = [
        record.getMessage() for record in caplog.records if "lane_hooks " in record.getMessage()
    ]
    assert len(stages) == 3
    assert "stage=sidecar_create_begin" in stages[0]
    assert "stage=sidecar_create_complete" in stages[1]
    assert "stage=sidecar_create_begin" in stages[2]
    assert all(f"epoch={INITIAL_EPOCH} " in stage for stage in stages)


@pytest.mark.parametrize("failure", ["binding", "publisher", "producer"])
def test_optional_display_startup_failure_keeps_main_command_and_diagnostics_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from neurocraft_connectome import full_runtime
    from neurocraft_controller import brain_display

    space = TinySpace()
    closed: list[str] = []

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("synthetic optional display setup failure")

    monkeypatch.setattr(
        cli, "build_controller", lambda args: ProgramSelectingController(_small_core(space, args))
    )
    monkeypatch.setenv("NEUROCRAFT_RUN_TOKEN", TOKEN)
    if failure == "binding":
        monkeypatch.setattr(full_runtime, "load_full_runtime_artifact", fail)
    else:
        monkeypatch.setattr(
            cli, "_comparison_display_binding", lambda args, original: space.select()
        )
        if failure == "publisher":
            monkeypatch.setattr(brain_display, "AtomicBrainDisplayPublisher", fail)
        else:

            class Publisher(brain_display.AtomicBrainDisplayPublisher):
                def close(self, *, timeout: float = 1.0) -> bool:
                    closed.append("publisher_closed")
                    return super().close(timeout=timeout)

            monkeypatch.setattr(brain_display, "AtomicBrainDisplayPublisher", Publisher)
            monkeypatch.setattr(brain_display, "BrainDisplayProducer", fail)

    async def run(dispatcher: LaneSessionDispatcher) -> None:
        lane = dispatcher.lanes[0]
        hooks = dispatcher._hooks_factory(lane, lane.controller.identity)
        assert hooks.completed_step_observer is None
        command = lane.controller.step(
            _observation(
                lane.controller.controller,  # type: ignore[attr-defined]
                0,
                SensorFrameV2.from_sensor_frame(_sensors(), (2,) * 6),
            ),
            command_seq=0,
        )
        assert isinstance(command, ProgramAction)
        assert hooks.issued_action_sink is not None
        hooks.issued_action_sink(command)
        assert hooks.receipt_sink is not None
        hooks.receipt_sink(program_receipt(command))
        if hooks.close is not None:
            hooks.close()
        await dispatcher.close()

    monkeypatch.setattr(cli, "_serve_comparison", run)
    assert (
        cli.main(
            _arguments(
                tmp_path,
                "--brain-display-out",
                str(tmp_path / "display"),
                "--program-decisions-out",
                str(tmp_path / "decisions"),
            )
        )
        == 0
    )
    directory = tmp_path / "epochs" / str(INITIAL_EPOCH)
    assert [
        json.loads(line)["kind"]
        for line in (directory / "program-decisions.ndjson").read_bytes().splitlines()
    ] == ["issued", "receipt"]
    assert not (directory / "brain-display.json").exists()
    assert closed == (["publisher_closed"] if failure == "producer" else [])
