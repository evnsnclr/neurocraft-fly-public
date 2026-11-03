from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from neurocraft_protocol import (
    Action,
    Fault,
    Hello,
    HelloAck,
    PeerRole,
    SensorFrameV2,
    encode_message,
    parse_message,
)
from neurocraft_protocol.sensor_contract import OBSERVATION_SCHEMA_V2_SHA256
from neurocraft_service import ServiceConfig

from tests.protocol.helpers import identity, observation
from tests.service.test_service import TOKEN, CountingController, authenticated_connection


@pytest.mark.parametrize(
    "session_v2,frame_v2", [(False, False), (True, True), (True, False), (False, True)]
)
def test_live_session_requires_its_own_sensor_contract(session_v2: bool, frame_v2: bool) -> None:
    async def scenario() -> None:
        run_identity = identity()
        if session_v2:
            run_identity = replace(
                run_identity, observation_schema_hash=OBSERVATION_SCHEMA_V2_SHA256
            )
        controller = CountingController()
        server, socket = await authenticated_connection(
            ServiceConfig(identity=run_identity, token=TOKEN, port=0),
            controller=controller,
        )
        try:
            await socket.send(
                encode_message(
                    Hello.create(
                        identity=run_identity,
                        role=PeerRole.MINECRAFT_SERVER,
                        token=TOKEN,
                    )
                )
            )
            assert isinstance(parse_message(await socket.recv()), HelloAck)
            frame = observation(run_identity=run_identity)
            if frame_v2:
                frame = replace(
                    frame, sensors=SensorFrameV2.from_sensor_frame(frame.sensors, (2,) * 6)
                )
            await socket.send(encode_message(frame))
            answer = parse_message(await asyncio.wait_for(socket.recv(), timeout=2.0))
            if session_v2 == frame_v2:
                assert isinstance(answer, Action)
                assert controller.steps == 1
            else:
                assert isinstance(answer, Fault)
                assert controller.steps == 0
        finally:
            await socket.close()
            await server.__aexit__(None, None, None)

    asyncio.run(scenario())
