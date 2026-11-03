"""R-29 at the wire: a paused window is not lateness, and lateness is still lateness.

Observed live on 2026-09-04 (``STATUS.md`` lane L, session
``interactive-20260904T155926Z-25199c93``): while the operator's HOVER owned the body for 1,786
ticks the service logged ``retired 1 validated command(s) the authority can no longer apply at
game tick N; K retired this session`` once per tick and ended at 1,790 against the four frames
the authority had actually retired as late. These tests drive the same shapes through the real
service session: a paused window produces no WARNING and one INFO summary; a driving frame the
world outran -- during a pause or not -- still WARNs and still counts; the superseded path is
unchanged; and the session summary keeps the two figures apart.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace

import pytest
from neurocraft_protocol import (
    MAX_FRAME_BYTES,
    WEBSOCKET_SUBPROTOCOL,
    Action,
    ControlAck,
    ControlCommand,
    ControlOperation,
    Fault,
    encode_message,
    parse_message,
)
from neurocraft_service import NeuralService, SessionSummary
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve as websocket_serve
from websockets.typing import Subprotocol

from tests.protocol.helpers import identity, observation, receipt
from tests.service.test_service import (
    CountingController,
    authenticate,
    config,
    materialize_effectful_control,
)

RETIREMENT = "retired 1 validated command(s) the authority can no longer apply at game tick"
SUPERSEDED = "superseded 1 unreceipted command(s) the authority consumed unapplied"
PAUSED_WINDOW = "paused window ticks"
SESSION_SUMMARY = "session summary:"


async def connect_service(service: NeuralService):
    """Serve one ``NeuralService`` we keep a handle on, so its session summary can be read."""

    server_context = websocket_serve(
        service.handle,
        "127.0.0.1",
        0,
        subprotocols=[Subprotocol(WEBSOCKET_SUBPROTOCOL)],
        compression=None,
        max_size=MAX_FRAME_BYTES,
    )
    server = await server_context.__aenter__()
    port = server.sockets[0].getsockname()[1]
    websocket = await connect(
        f"ws://127.0.0.1:{port}",
        subprotocols=[WEBSOCKET_SUBPROTOCOL],
        compression=None,
        max_size=65_536,
    )
    return server_context, websocket


async def expect_action(websocket) -> Action:
    frame = parse_message(await websocket.recv())
    assert isinstance(frame, Action), f"expected an action, got {frame!r}"
    return frame


def records(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.levelno == level]


def containing(messages: list[str], needle: str) -> list[str]:
    return [message for message in messages if needle in message]


def control(sequence: int, tick: int, operation: ControlOperation) -> ControlCommand:
    return ControlCommand(
        identity=identity(), control_seq=sequence, issued_tick=tick, operation=operation
    )


def test_a_paused_window_is_counted_apart_from_lateness_and_summarised_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Eight paused ticks with no authority receipts: 0 WARNINGs, 1 INFO.

    Under the fixed-count log this was eight ``retired 1 validated command(s)`` WARNINGs and a
    session count of eight "late" frames against zero late frames.
    """

    caplog.set_level(logging.INFO)
    service = NeuralService(config(), CountingController())

    async def scenario() -> None:
        server_context, websocket = await connect_service(service)
        try:
            await authenticate(websocket)
            await websocket.send(encode_message(observation(sequence=0, tick=100)))
            driving = await expect_action(websocket)
            await websocket.send(encode_message(receipt(driving)))

            pause_ack, first_neutral = await materialize_effectful_control(
                websocket,
                control(0, 101, ControlOperation.PAUSE),
                observation(sequence=1, tick=101),
            )
            assert pause_ack.accepted and first_neutral.neutral
            for sequence in range(2, 9):
                await websocket.send(
                    encode_message(observation(sequence=sequence, tick=100 + sequence))
                )
                assert (await expect_action(websocket)).neutral
            # Ticks 104..108 retired the neutral actions from ticks 101..105 -- five of them.
            assert records(caplog, logging.WARNING) == []
            assert containing(records(caplog, logging.INFO), PAUSED_WINDOW) == []

            await websocket.send(encode_message(control(1, 109, ControlOperation.RESUME)))
            await websocket.send(encode_message(observation(sequence=9, tick=109)))
            resume_ack = parse_message(await websocket.recv())
            assert isinstance(resume_ack, ControlAck) and resume_ack.accepted
            assert not (await expect_action(websocket)).neutral
            # The window's last three neutral actions (ticks 106..108) are still pending, so the
            # summary waits for them rather than reporting five.
            assert containing(records(caplog, logging.INFO), PAUSED_WINDOW) == []
            for sequence, tick in ((10, 110), (11, 111)):
                await websocket.send(encode_message(observation(sequence=sequence, tick=tick)))
                assert not (await expect_action(websocket)).neutral
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())

    assert records(caplog, logging.WARNING) == []
    assert containing(records(caplog, logging.INFO), PAUSED_WINDOW) == [
        "paused window ticks 101..108: 8 neutral action(s) issued while PAUSED were unreceipted "
        "(body ownership and cause unconfirmed; counted separately from driving lateness)"
    ]
    (summary_line,) = containing(records(caplog, logging.INFO), SESSION_SUMMARY)
    assert summary_line == (
        "session summary: 12 command(s) issued, 0 sent late, 0 retired as lateness, "
        "8 paused-window neutral action(s) unapplied across 1 paused window(s) (not lateness), "
        "8 window retirement(s) in total"
    )
    assert service.last_session_summary == SessionSummary(
        commands_issued=12,
        late_sends=0,
        lateness_retired=0,
        paused_window_unapplied=8,
        paused_windows=1,
        window_retirements_total=8,
        unattributed_retirements=0,
        ledger_desyncs=0,
    )


def test_a_late_driving_frame_retired_during_a_paused_window_still_warns_and_counts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The reviewer's question: a driving frame the world outran is lateness even mid-pause.

    Action 0 (target 102) is never receipted; the PAUSE lands at 101 and the world reaches 103
    while the service is paused. Its retirement is the one WARNING, the paused neutrals around it
    produce none, and the two counts stay apart in the summary.
    """

    caplog.set_level(logging.INFO)
    service = NeuralService(config(), CountingController())

    async def scenario() -> None:
        server_context, websocket = await connect_service(service)
        try:
            await authenticate(websocket)
            await websocket.send(encode_message(observation(sequence=0, tick=100)))
            assert not (await expect_action(websocket)).neutral

            pause_ack, _ = await materialize_effectful_control(
                websocket,
                control(0, 101, ControlOperation.PAUSE),
                observation(sequence=1, tick=101),
            )
            assert pause_ack.accepted
            for sequence in range(2, 5):
                await websocket.send(
                    encode_message(observation(sequence=sequence, tick=100 + sequence))
                )
                assert (await expect_action(websocket)).neutral
            late = containing(records(caplog, logging.WARNING), RETIREMENT)
            assert late == [f"{RETIREMENT} 103; 1 retired this session"]

            await websocket.send(encode_message(control(1, 105, ControlOperation.RESUME)))
            for sequence, tick in ((5, 105), (6, 106), (7, 107)):
                await websocket.send(encode_message(observation(sequence=sequence, tick=tick)))
                if sequence == 5:
                    resume_ack = parse_message(await websocket.recv())
                    assert isinstance(resume_ack, ControlAck) and resume_ack.accepted
                assert not (await expect_action(websocket)).neutral
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())

    assert records(caplog, logging.WARNING) == [f"{RETIREMENT} 103; 1 retired this session"]
    assert containing(records(caplog, logging.INFO), PAUSED_WINDOW) == [
        "paused window ticks 101..104: 4 neutral action(s) issued while PAUSED were unreceipted "
        "(body ownership and cause unconfirmed; counted separately from driving lateness)"
    ]
    assert service.last_session_summary == SessionSummary(
        commands_issued=8,
        late_sends=0,
        lateness_retired=1,
        paused_window_unapplied=4,
        paused_windows=1,
        window_retirements_total=5,
        unattributed_retirements=0,
        ledger_desyncs=0,
    )


def test_genuinely_late_driving_frames_keep_the_warning_and_the_count_exactly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """RUNNING, two un-receipted frames the world outruns: two WARNINGs, counting 1 then 2."""

    caplog.set_level(logging.INFO)
    service = NeuralService(config(), CountingController())

    async def scenario() -> None:
        server_context, websocket = await connect_service(service)
        try:
            await authenticate(websocket)
            for sequence, tick in ((0, 100), (1, 101), (2, 103), (3, 104)):
                await websocket.send(encode_message(observation(sequence=sequence, tick=tick)))
                assert not (await expect_action(websocket)).neutral
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())

    assert records(caplog, logging.WARNING) == [
        f"{RETIREMENT} 103; 1 retired this session",
        f"{RETIREMENT} 104; 2 retired this session",
    ]
    assert containing(records(caplog, logging.INFO), PAUSED_WINDOW) == []
    assert service.last_session_summary == SessionSummary(
        commands_issued=4,
        late_sends=0,
        lateness_retired=2,
        paused_window_unapplied=0,
        paused_windows=0,
        window_retirements_total=2,
        unattributed_retirements=0,
        ledger_desyncs=0,
    )


def test_a_superseding_receipt_keeps_the_warning_and_the_count_exactly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The ADR 0011 supersession path is untouched: it WARNs and counts as lateness."""

    caplog.set_level(logging.INFO)
    service = NeuralService(config(), CountingController())

    async def scenario() -> None:
        server_context, websocket = await connect_service(service)
        try:
            await authenticate(websocket)
            await websocket.send(encode_message(observation(sequence=0, tick=100)))
            await expect_action(websocket)
            await websocket.send(encode_message(observation(sequence=1, tick=101)))
            second = await expect_action(websocket)
            await websocket.send(encode_message(receipt(second)))
            await websocket.send(encode_message(observation(sequence=2, tick=102)))
            await expect_action(websocket)
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())

    assert records(caplog, logging.WARNING) == [
        f"receipt for command 1 {SUPERSEDED}; 1 retired this session"
    ]
    assert service.last_session_summary == SessionSummary(
        commands_issued=3,
        late_sends=0,
        lateness_retired=1,
        paused_window_unapplied=0,
        paused_windows=0,
        window_retirements_total=1,
        unattributed_retirements=0,
        ledger_desyncs=0,
    )


def test_the_session_summary_is_still_written_when_the_session_fails_closed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two late frames, then a duplicate observation sequence ends the session on a protocol
    error. The summary is written all the same, with the lateness it had counted, so a session
    that dies is not also a session that vanishes from the log."""

    caplog.set_level(logging.INFO)
    service = NeuralService(config(), CountingController())

    async def scenario() -> None:
        server_context, websocket = await connect_service(service)
        try:
            await authenticate(websocket)
            for sequence, tick in ((0, 100), (1, 101), (2, 103), (3, 104)):
                await websocket.send(encode_message(observation(sequence=sequence, tick=tick)))
                assert not (await expect_action(websocket)).neutral
            await websocket.send(encode_message(observation(sequence=3, tick=105)))
            fault = parse_message(await websocket.recv())
            assert isinstance(fault, Fault)
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())

    assert containing(records(caplog, logging.WARNING), RETIREMENT) == [
        f"{RETIREMENT} 103; 1 retired this session",
        f"{RETIREMENT} 104; 2 retired this session",
    ]
    (summary_line,) = containing(records(caplog, logging.INFO), SESSION_SUMMARY)
    assert summary_line == (
        "session summary: 4 command(s) issued, 0 sent late, 2 retired as lateness, "
        "0 paused-window neutral action(s) unapplied across 0 paused window(s) (not lateness), "
        "2 window retirement(s) in total"
    )
    assert service.last_session_summary == SessionSummary(
        commands_issued=4,
        late_sends=0,
        lateness_retired=2,
        paused_window_unapplied=0,
        paused_windows=0,
        window_retirements_total=2,
        unattributed_retirements=0,
        ledger_desyncs=0,
    )


def test_a_receipt_that_supersedes_a_head_and_then_fails_to_bind_leaves_the_gap_named(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The validator pops command 0 as superseded before it refuses the receipt for command 1
    (its observation sequence is wrong), and the session fails closed. The ledger never saw that
    release, so the summary names the gap as unattributed instead of letting lateness and
    paused-window figures that sum to zero stand beside a count of one."""

    caplog.set_level(logging.INFO)
    service = NeuralService(config(), CountingController())

    async def scenario() -> None:
        server_context, websocket = await connect_service(service)
        try:
            await authenticate(websocket)
            await websocket.send(encode_message(observation(sequence=0, tick=100)))
            await expect_action(websocket)
            await websocket.send(encode_message(observation(sequence=1, tick=101)))
            second = await expect_action(websocket)
            forged = replace(receipt(second), observation_seq=second.observation_seq + 7)
            await websocket.send(encode_message(forged))
            fault = parse_message(await websocket.recv())
            assert isinstance(fault, Fault)
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())

    assert containing(records(caplog, logging.WARNING), SUPERSEDED) == []
    (summary_line,) = containing(records(caplog, logging.WARNING), SESSION_SUMMARY)
    assert summary_line == (
        "session summary: 2 command(s) issued, 0 sent late, 0 retired as lateness, "
        "0 paused-window neutral action(s) unapplied across 0 paused window(s) (not lateness), "
        "1 window retirement(s) in total; 1 retirement(s) unattributed "
        "(neither lateness nor paused-window)"
    )
    assert service.last_session_summary == SessionSummary(
        commands_issued=2,
        late_sends=0,
        lateness_retired=0,
        paused_window_unapplied=0,
        paused_windows=0,
        window_retirements_total=1,
        unattributed_retirements=1,
        ledger_desyncs=0,
    )


def test_a_plain_pause_whose_neutral_actions_are_receipted_is_not_unapplied(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No body program: the authority applies and receipts the neutral actions.

    Nothing is retired, nothing is lateness, and the window's one line says what happened.
    """

    caplog.set_level(logging.INFO)
    service = NeuralService(config(), CountingController())

    async def scenario() -> None:
        server_context, websocket = await connect_service(service)
        try:
            await authenticate(websocket)
            await websocket.send(encode_message(observation(sequence=0, tick=100)))
            await websocket.send(encode_message(receipt(await expect_action(websocket))))
            pause_ack, paused = await materialize_effectful_control(
                websocket,
                control(0, 101, ControlOperation.PAUSE),
                observation(sequence=1, tick=101),
            )
            assert pause_ack.accepted
            await websocket.send(encode_message(receipt(paused)))
            await websocket.send(encode_message(observation(sequence=2, tick=102)))
            await websocket.send(encode_message(receipt(await expect_action(websocket))))

            await websocket.send(encode_message(control(1, 103, ControlOperation.RESUME)))
            await websocket.send(encode_message(observation(sequence=3, tick=103)))
            resume_ack = parse_message(await websocket.recv())
            assert isinstance(resume_ack, ControlAck) and resume_ack.accepted
            assert not (await expect_action(websocket)).neutral
        finally:
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())

    assert records(caplog, logging.WARNING) == []
    assert containing(records(caplog, logging.INFO), PAUSED_WINDOW) == [
        "paused window ticks 101..102: all 2 neutral action(s) were applied and receipted by the "
        "authority (not lateness)"
    ]
    assert service.last_session_summary == SessionSummary(
        commands_issued=4,
        late_sends=0,
        lateness_retired=0,
        paused_window_unapplied=0,
        paused_windows=1,
        window_retirements_total=0,
        unattributed_retirements=0,
        ledger_desyncs=0,
    )
