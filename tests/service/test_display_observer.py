"""Optional presentation must preserve the authenticated command stream."""

from __future__ import annotations

import asyncio
import hashlib
import threading
from collections.abc import Coroutine, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from neurocraft_protocol import (
    WEBSOCKET_SUBPROTOCOL,
    Action,
    Command,
    Hello,
    HelloAck,
    MotorVector,
    Observation,
    PeerRole,
    ProgramAction,
    ProgramId,
    encode_message,
    parse_message,
)
from neurocraft_service import ServiceConfig, create_server
from websockets.asyncio.client import connect
from websockets.typing import Subprotocol

from tests.protocol.helpers import action, identity, observation, program_action, program_receipt
from tests.protocol.helpers import receipt as action_receipt

TOKEN = "T" * 43
STEP_COUNT = 5
ObserverMode = Literal["off", "on", "raising"]


class StepProbe:
    """Synthetic changing commands, with thread/order evidence independent of the hook."""

    def __init__(self, *, program_lane: bool) -> None:
        self.program_lane = program_lane
        self.returned: list[Command] = []
        self.events: list[tuple[str, int, int]] = []

    def step(self, observed: Observation, *, command_seq: int) -> Command:
        self.events.append(("step_start", command_seq, threading.get_ident()))
        digest = hashlib.sha256(encode_message(observed).encode()).hexdigest()
        if self.program_lane:
            issued: Command = program_action(
                command_sequence=command_seq,
                observation_sequence=observed.observation_seq,
                observation_tick=observed.game_tick,
                neural_start=len(self.returned) * 5,
                state_digest=digest,
                run_identity=observed.identity,
                program=ProgramId.ESCAPE_JUMP if command_seq % 2 else ProgramId.CRUISE,
            )
        else:
            issued = action(
                command_sequence=command_seq,
                observation_sequence=observed.observation_seq,
                observation_tick=observed.game_tick,
                neural_start=len(self.returned) * 5,
                state_digest=digest,
                run_identity=observed.identity,
                motors=MotorVector(forward_thrust=0.1 + command_seq * 0.05),
            )
        self.returned.append(issued)
        self.events.append(("step_end", command_seq, threading.get_ident()))
        return issued


@pytest.mark.parametrize("program_lane", [False, True], ids=["motors", "programs"])
def test_display_off_on_and_failure_preserve_complete_wire_commands(
    program_lane: bool, caplog: pytest.LogCaptureFixture
) -> None:
    async def run(mode: ObserverMode) -> list[str]:
        loop_thread = threading.get_ident()
        controller = StepProbe(program_lane=program_lane)
        captures: list[tuple[Command, int]] = []

        def observe(command: Command) -> None:
            controller.events.append(("observe", command.command_seq, threading.get_ident()))
            captures.append((command, len(controller.returned)))
            if mode == "raising":
                raise RuntimeError("deliberate optional display failure")

        wire: list[str] = []
        config = ServiceConfig(identity=identity(), token=TOKEN, port=0)
        async with create_server(
            config,
            controller,
            completed_step_observer=None if mode == "off" else observe,
        ) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(
                f"ws://127.0.0.1:{port}",
                subprotocols=[Subprotocol(WEBSOCKET_SUBPROTOCOL)],
                compression=None,
            ) as socket:
                hello = Hello.create(
                    identity=identity(), role=PeerRole.MINECRAFT_SERVER, token=TOKEN
                )
                await socket.send(encode_message(hello))
                ack = parse_message(await asyncio.wait_for(socket.recv(), timeout=2.0))
                assert isinstance(ack, HelloAck)
                assert ack.verify(TOKEN, client_nonce=hello.client_nonce)
                for sequence in range(STEP_COUNT):
                    await socket.send(
                        encode_message(observation(sequence=sequence, tick=100 + sequence))
                    )
                    raw = await asyncio.wait_for(socket.recv(), timeout=2.0)
                    assert isinstance(raw, str)
                    issued = parse_message(raw)
                    assert isinstance(issued, ProgramAction if program_lane else Action)
                    assert isinstance(issued, Action | ProgramAction)
                    assert issued.command_seq == sequence
                    assert raw == encode_message(controller.returned[sequence])
                    wire.append(raw)
                    receipt = (
                        program_receipt(issued)
                        if isinstance(issued, ProgramAction)
                        else action_receipt(issued)
                    )
                    await socket.send(encode_message(receipt))

        expected_captures = {"off": 0, "on": STEP_COUNT, "raising": 1}[mode]
        assert len(controller.returned) == len(wire) == STEP_COUNT
        assert len(captures) == expected_captures
        for index, (captured, completed_count) in enumerate(captures):
            assert captured is controller.returned[index]
            assert completed_count == index + 1
        for sequence in range(STEP_COUNT):
            events = [event for event in controller.events if event[1] == sequence]
            assert [event[0] for event in events] == ["step_start", "step_end"] + (
                ["observe"] if sequence < expected_captures else []
            )
            assert len({event[2] for event in events}) == 1
            assert events[0][2] != loop_thread
        return wire

    async def scenario() -> None:
        baseline = await run("off")
        assert await run("on") == baseline
        assert await run("raising") == baseline
        # Prevent a constant/no-command double from making the equality vacuous.
        commands = [parse_message(raw) for raw in baseline]
        if program_lane:
            assert {frame.program for frame in commands if isinstance(frame, ProgramAction)} == {
                ProgramId.CRUISE,
                ProgramId.ESCAPE_JUMP,
            }
        else:
            assert len({frame.motors for frame in commands if isinstance(frame, Action)}) == 5

    asyncio.run(scenario())
    warnings = [
        record for record in caplog.records if "display observer failed" in record.getMessage()
    ]
    assert len(warnings) == 1


@pytest.mark.parametrize("failure", ["off", "graph", "binding", "publisher", "producer", None])
def test_cli_optional_display_startup_does_not_block_service_and_closes_publisher(
    failure: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import neurocraft_connectome.full_runtime as runtime
    import neurocraft_controller.brain_display as display

    import scripts.run_connectome_service as cli

    calls: list[str] = []
    core = SimpleNamespace(
        identity=identity(),
        embodied_senses=SimpleNamespace(populations=object()),
        escape_binding=SimpleNamespace(ttmn_pool_identity_sha256="b" * 64),
        warm_compute_path=lambda: 0.0,
    )
    graph = object()
    binding = SimpleNamespace(identity_sha256="a" * 64)

    def enter_stage(stage: str) -> None:
        calls.append(stage)
        if failure == "off":
            pytest.fail("disabled display attempted optional setup")
        if failure == stage:
            raise RuntimeError(f"deliberate display {stage} startup failure")

    @contextmanager
    def load_graph(path: Path, *, expected_manifest_sha256: str) -> Iterator[object]:
        enter_stage("graph")
        assert expected_manifest_sha256 == core.identity.graph_hash
        try:
            yield graph
        finally:
            calls.append("graph_closed")

    def load_binding(
        supplied_graph: object, annotations: Path, populations: object, ttmn_hash: str
    ) -> SimpleNamespace:
        enter_stage("binding")
        assert supplied_graph is graph
        assert populations is core.embodied_senses.populations
        assert ttmn_hash == core.escape_binding.ttmn_pool_identity_sha256
        return binding

    class Publisher:
        last_error = None

        def __init__(self, path: Path) -> None:
            enter_stage("publisher")
            assert path == tmp_path / "display.json"

        def close(self, *, timeout: float) -> bool:
            assert timeout == 1.0
            calls.append("publisher_closed")
            return True

    class Producer:
        last_error = None

        def __init__(self, supplied_binding: object, publisher: Publisher) -> None:
            enter_stage("producer")
            assert supplied_binding is binding

        def capture_completed_step(self, controller: object, command: Command) -> None:
            assert controller is core
            assert command.command_seq == 0
            calls.append("captured")

    async def run_service(config: ServiceConfig, controller: object, **kwargs: Any) -> None:
        calls.append("service_started")
        assert controller is core
        assert config.identity == core.identity
        observer = kwargs["completed_step_observer"]
        if failure is None:
            assert callable(observer)
            observer(action())
        else:
            assert observer is None

    async def passthrough_rewarm(coroutine: Coroutine[Any, Any, None], controller: object) -> None:
        assert controller is core
        await coroutine

    monkeypatch.setenv("NEUROCRAFT_RUN_TOKEN", TOKEN)
    monkeypatch.setattr(cli, "build_controller", lambda args: core)
    monkeypatch.setattr(cli, "serve", run_service)
    monkeypatch.setattr(cli, "_serve_with_idle_rewarm", passthrough_rewarm)
    monkeypatch.setattr(runtime, "load_full_runtime_artifact", load_graph)
    monkeypatch.setattr(display, "load_brain_display_binding", load_binding)
    monkeypatch.setattr(display, "AtomicBrainDisplayPublisher", Publisher)
    monkeypatch.setattr(display, "BrainDisplayProducer", Producer)
    argv = ["--controller-profile", "full_mcns_loom_program_v2"]
    if failure != "off":
        argv.extend(["--brain-display-out", str(tmp_path / "display.json")])
    assert cli.main(argv) == 0
    assert calls.count("service_started") == 1
    assert calls.count("publisher_closed") == int(failure in (None, "producer"))
    assert calls.count("graph_closed") == int(failure not in ("off", "graph"))
    if failure == "producer":
        assert calls.index("publisher_closed") < calls.index("service_started")
    if failure is None:
        assert calls[-2:] == ["captured", "publisher_closed"]
    elif failure != "off":
        assert "brain display unavailable (RuntimeError); command service continues" in (
            capsys.readouterr().out
        )
    assert not (tmp_path / "display.json").exists()
