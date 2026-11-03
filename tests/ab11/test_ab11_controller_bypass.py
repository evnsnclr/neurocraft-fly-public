"""AB-11 test power against the two Controller-protocol bypass mutants (M4, M5).

Both mutants let the connectome run for real and then replace the motors that reach
Minecraft. The service's own action validation checks identity, sequence, and tick but
deliberately not motors-versus-state, so the load-bearing detectors are the
issued-action hash binding in the receipt trace recorder and an independent re-decode
of the committed neural state.

The mutants live in ``tests/ab11/mutants.py`` and are falsification artifacts only,
never Faithful Organism Mode components.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from neurocraft_connectome import RuntimeGraph
from neurocraft_controller import (
    ConnectomeController,
    ConnectomeRuntimeControlHandler,
    ControllerIntegrityError,
    MinecraftReceiptTraceRecorder,
)
from neurocraft_protocol import (
    WEBSOCKET_SUBPROTOCOL,
    Action,
    ControlAck,
    ControlCommand,
    ControlOperation,
    Fault,
    FaultCode,
    Hello,
    HelloAck,
    PeerRole,
    TraceWriter,
    encode_message,
    parse_message,
)
from neurocraft_service import ServiceConfig, create_server
from websockets.asyncio.client import connect
from websockets.typing import Subprotocol

from tests.controller.conftest import make_controller, make_observation, make_sensors
from tests.integration.test_fixture_service_loop import (
    TOKEN,
    _forces,
    _manifest,
    _observation,
    _record_telemetry,
)
from tests.integration.test_fixture_service_loop import _receipt as _service_receipt

from . import mutants
from .acceptance import advance, redecoded_motors
from .mutants import (
    MUTANT_ENTITY_ID,
    MutantDirectSteeringController,
    MutantTelemetryOnlyController,
)

RECORDING_CONTENT_ID = "ab11-mutation-power"
ISSUED_ACTION_HASH_REFUSAL = "issued action hash does not match evidence"


@contextmanager
def _recording_recorder(
    controller: ConnectomeController,
    output: Path,
) -> Iterator[MinecraftReceiptTraceRecorder]:
    """Open the production receipt recorder on one controller, already RECORDING.

    Same class, same constructor, and same RECORD_START control path the live service
    uses in ``_recorded_service_exchange``; only the transport is removed, so the
    issued-action hash binding can be exercised directly on an issued Action.
    """

    forces = _forces()
    recorder = MinecraftReceiptTraceRecorder(
        controller,
        TraceWriter(output, _manifest(controller, forces)),
        forces=forces,
    )
    recorder.handle_control(
        ControlCommand(
            identity=controller.identity,
            control_seq=0,
            issued_tick=0,
            operation=ControlOperation.RECORD_START,
            content_id=RECORDING_CONTENT_ID,
        )
    )
    try:
        yield recorder
    finally:
        recorder.abort()


async def _recorded_service_exchange(
    service_controller: Any,
    recorder_controller: ConnectomeController,
    output: Path,
    *,
    observation_count: int,
) -> tuple[MinecraftReceiptTraceRecorder, list[Any]]:
    """Drive the real in-process service with a recorder wired to the issued actions.

    Follows ``tests/integration/test_fixture_service_loop.py`` exactly, except that the
    object handed to ``create_server`` is decoupled from the object handed to the
    recorder so a Controller-protocol mutant can sit between them.
    """

    forces = _forces()
    recorder = MinecraftReceiptTraceRecorder(
        recorder_controller,
        TraceWriter(output, _manifest(recorder_controller, forces)),
        forces=forces,
    )
    context = create_server(
        ServiceConfig(identity=recorder_controller.identity, token=TOKEN, port=0),
        service_controller,
        issued_action_sink=recorder.accept_issued_action,
        action_telemetry_sink=partial(_record_telemetry, recorder),
        receipt_sink=recorder.accept_receipt,
        control_handler=ConnectomeRuntimeControlHandler(recorder=recorder),
    )
    server = await context.__aenter__()
    port = server.sockets[0].getsockname()[1]
    websocket = await connect(
        f"ws://127.0.0.1:{port}",
        subprotocols=[Subprotocol(WEBSOCKET_SUBPROTOCOL)],
        compression=None,
    )
    responses: list[Any] = []
    try:
        await websocket.send(
            encode_message(
                Hello.create(
                    identity=recorder_controller.identity,
                    role=PeerRole.MINECRAFT_SERVER,
                    token=TOKEN,
                )
            )
        )
        assert isinstance(parse_message(await websocket.recv()), HelloAck)
        await websocket.send(
            encode_message(
                ControlCommand(
                    identity=recorder_controller.identity,
                    control_seq=0,
                    issued_tick=0,
                    operation=ControlOperation.RECORD_START,
                    content_id=RECORDING_CONTENT_ID,
                )
            )
        )
        acknowledgement = parse_message(await websocket.recv())
        assert isinstance(acknowledgement, ControlAck)
        assert acknowledgement.accepted
        for sequence in range(observation_count):
            await websocket.send(
                encode_message(_observation(recorder_controller, sequence, open_world=True))
            )
            message = parse_message(await websocket.recv())
            responses.append(message)
            if not isinstance(message, Action):
                break
            await websocket.send(encode_message(_service_receipt(message)))
    finally:
        await websocket.close()
        await context.__aexit__(None, None, None)
        recorder.abort()
    return recorder, responses


def test_ab11_fake_telemetry_only_trips_issued_action_hash_binding(
    controller_graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    """AB-11(M4 fake telemetry-only) vs the issued-action hash binding.

    Named tests: ``tests/integration/test_fixture_service_loop.py::
    test_authenticated_service_executes_connectome_controller_not_neutral_fake`` and
    ``tests/service/test_service.py::
    test_issued_action_sink_failure_fails_closed_as_trace_failure``.
    Detector: ``MinecraftReceiptTraceRecorder.accept_issued_action`` raising
    ``ControllerIntegrityError("issued action hash does not match evidence")``, which
    the live service reports to the peer as ``FaultCode.TRACE_FAILURE``.
    """

    faithful = ConnectomeController.create(
        controller_graph,
        run_id="ab11-m4-baseline",
        entity_id=MUTANT_ENTITY_ID,
    )
    recorder, responses = asyncio.run(
        _recorded_service_exchange(
            faithful,
            faithful,
            tmp_path / "m4-baseline.ndjson",
            # Four, not three: an action applies one action budget after the observation it
            # answers, and a record is written only once a later observation proves the
            # transition, so the first record needs the fourth observation.
            observation_count=4,
        )
    )
    assert all(isinstance(message, Action) for message in responses)
    assert not responses[-1].motors.is_neutral
    assert recorder.record_count >= 1

    inner = ConnectomeController.create(
        controller_graph,
        run_id="ab11-m4-mutant",
        entity_id=MUTANT_ENTITY_ID,
    )
    mutant_recorder, mutant_responses = asyncio.run(
        _recorded_service_exchange(
            MutantTelemetryOnlyController(inner),
            inner,
            tmp_path / "m4-mutant.ndjson",
            observation_count=3,
        )
    )
    fault = mutant_responses[0]
    assert isinstance(fault, Fault)
    assert fault.code is FaultCode.TRACE_FAILURE
    assert fault.neutral_applied
    assert mutant_recorder.record_count == 0

    # The neural evidence really was computed; only the issued motors were scripted.
    evidence = inner.last_evidence
    assert evidence is not None
    assert not evidence.motors.is_neutral
    assert evidence.motors != replace(
        evidence.motors,
        forward_thrust=mutants.SCRIPTED_FORWARD_THRUST,
        vertical_thrust=mutants.SCRIPTED_VERTICAL_THRUST,
    )

    # Same refusal at recorder level, message and all: the binding itself is the
    # detector, not the transport that happens to surface it as a TRACE_FAILURE.
    accepted = ConnectomeController.create(
        controller_graph,
        run_id="ab11-m4-recorder-faithful",
        entity_id=MUTANT_ENTITY_ID,
    )
    with _recording_recorder(accepted, tmp_path / "m4-recorder-faithful.ndjson") as recorder:
        recorder.accept_issued_action(accepted.step(make_observation(accepted, 0), command_seq=0))
        assert recorder.pending_state.issued_without_receipt == (0,)
        assert recorder.pending_state.actions_without_controller_evidence == ()

    refused_inner = ConnectomeController.create(
        controller_graph,
        run_id="ab11-m4-recorder-mutant",
        entity_id=MUTANT_ENTITY_ID,
    )
    with _recording_recorder(refused_inner, tmp_path / "m4-recorder-mutant.ndjson") as recorder:
        scripted = MutantTelemetryOnlyController(refused_inner).step(
            make_observation(refused_inner, 0),
            command_seq=0,
        )
        with pytest.raises(ControllerIntegrityError, match=ISSUED_ACTION_HASH_REFUSAL):
            recorder.accept_issued_action(scripted)


def test_ab11_direct_steering_defeats_sensory_lesion_neutrality(
    controller_graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    """AB-11(M5 direct target steering) vs the sensory-lesion acceptance predicate.

    Named test: ``tests/controller/test_causal_controller.py::
    test_lesion_changes_state_and_output`` (a controller with both sensory bodies
    ablated must fall to neutral motors).
    Detectors: the lesion-neutrality predicate, an independent re-decode of the
    committed state through the production decoder and primitive projection, and the
    issued-action hash binding in ``MinecraftReceiptTraceRecorder`` exercised at
    recorder level on the very actions this mutant issues.
    """

    faithful = make_controller(controller_graph, mode="lesion", run_id="ab11-m5-baseline")
    faithful.set_lesion(
        (
            faithful.configuration.left_sensor_body_id,
            faithful.configuration.right_sensor_body_id,
        )
    )
    faithful_actions = advance(faithful, count=4)
    assert faithful_actions[-1].motors.is_neutral
    assert redecoded_motors(faithful) == faithful_actions[-1].motors

    inner = make_controller(controller_graph, mode="lesion", run_id="ab11-m5-mutant")
    inner.set_lesion(
        (
            inner.configuration.left_sensor_body_id,
            inner.configuration.right_sensor_body_id,
        )
    )
    mutant = MutantDirectSteeringController(inner)
    mutant_actions = [
        mutant.step(
            make_observation(inner, index, sensors=make_sensors(forward=6.0, left=1.0, right=5.0)),
            command_seq=index,
        )
        for index in range(4)
    ]

    # The lesioned connectome is silent, but the mutant still drives the entity.
    assert inner.engine.current_state().activity == faithful.engine.current_state().activity
    assert not mutant_actions[-1].motors.is_neutral
    assert redecoded_motors(inner) != mutant_actions[-1].motors
    assert mutant_actions[-1].motors.yaw_torque != 0.0

    # The production issued-action binding refuses these steered motors, and accepts
    # the faithful lesioned controller's own action on the identical recorder path.
    accepted = make_controller(controller_graph, mode="lesion", run_id="ab11-m5-recorder-faithful")
    accepted.set_lesion(
        (
            accepted.configuration.left_sensor_body_id,
            accepted.configuration.right_sensor_body_id,
        )
    )
    with _recording_recorder(accepted, tmp_path / "m5-recorder-faithful.ndjson") as recorder:
        recorder.accept_issued_action(accepted.step(make_observation(accepted, 0), command_seq=0))
        assert recorder.pending_state.issued_without_receipt == (0,)
        assert recorder.pending_state.actions_without_controller_evidence == ()

    refused_inner = make_controller(
        controller_graph,
        mode="lesion",
        run_id="ab11-m5-recorder-mutant",
    )
    refused_inner.set_lesion(
        (
            refused_inner.configuration.left_sensor_body_id,
            refused_inner.configuration.right_sensor_body_id,
        )
    )
    with _recording_recorder(refused_inner, tmp_path / "m5-recorder-mutant.ndjson") as recorder:
        steered = MutantDirectSteeringController(refused_inner).step(
            make_observation(
                refused_inner,
                0,
                sensors=make_sensors(forward=6.0, left=1.0, right=5.0),
            ),
            command_seq=0,
        )
        with pytest.raises(ControllerIntegrityError, match=ISSUED_ACTION_HASH_REFUSAL):
            recorder.accept_issued_action(steered)
