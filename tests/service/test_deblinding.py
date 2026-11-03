"""Service de-blinding: compute off the event loop, and late frames sent rather than dropped.

Live-diagnosed failure chain (2026-08-30 full-brain runs): whole-brain compute
ran inline in ``process_latest``, monopolizing the event loop, so
``receive_frames`` starved (observation-driven window retirement never fired;
three consecutive receipt-less accepts fill the two-slot window by
construction) and ``send_outbound`` starved (frames left late, the mod latched
DEADLINE_MISSED). These tests pin the de-blinding fix: the controller step runs
in a worker thread while observations keep flowing.

They also pin the correction to its companion. ADR 0018 added a pre-send drop for
an action the world outran during compute, which reused the dropped frame's
sequence number; that reuse killed two live program-lane sessions on 2026-09-03
because the controller had already recorded the sequence inside ``step``. A late
frame is now sent on exactly the path an on-time frame takes, logged and counted
as late, and retired fail-neutral by the authority.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import threading
from dataclasses import replace

import pytest
from neurocraft_controller import ControllerIntegrityError
from neurocraft_protocol import (
    ACTION_TARGET_OFFSET_TICKS,
    Action,
    ControlAck,
    ControlCommand,
    ControlOperation,
    Fault,
    FaultCode,
    MotorVector,
    ProgramAction,
    SessionValidator,
    encode_message,
    parse_message,
)
from neurocraft_service import ActionTelemetry

from tests.protocol.helpers import action as make_action
from tests.protocol.helpers import identity, observation, program_receipt, receipt
from tests.protocol.helpers import program_action as make_program_action
from tests.service.test_service import (
    RecordingControlHandler,
    authenticate,
    authenticated_connection,
    config,
)

RETIREMENT_MESSAGE = "retired 1 validated command"
DROP_MESSAGE = "dropping expired action"
LATE_SEND_MESSAGE = "sending late action"


class GatedStepController:
    """Controller whose selected step calls block until the test releases them.

    It carries the real controllers' command-sequence guard verbatim:
    ``ConnectomeController._validate_observation`` (controller.py:360) and
    ``FullBrainController._validate_observation`` (full_brain_controller.py:2063) both
    raise ``ControllerIntegrityError("command sequence is stale or duplicated")`` when
    ``command_seq <= self._last_command_seq``, and both record the sequence inside
    ``step`` before the service can decide anything about the action it returns. A
    double without that guard cannot see a caller that reuses a sequence number, which
    is how the pre-send expiry drop reached two live sessions unpinned.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.stepped: list[tuple[int, int, int]] = []
        self._last_command_seq = -1
        self._gates: dict[int, tuple[threading.Event, threading.Event]] = {}

    def gate_call(self, index: int) -> tuple[threading.Event, threading.Event]:
        entered, release = threading.Event(), threading.Event()
        self._gates[index] = (entered, release)
        return entered, release

    def step(self, observed, *, command_seq: int) -> Action:
        index = self.calls
        self.calls += 1
        if command_seq <= self._last_command_seq:
            raise ControllerIntegrityError("command sequence is stale or duplicated")
        gate = self._gates.get(index)
        if gate is not None:
            entered, release = gate
            entered.set()
            if not release.wait(timeout=5.0):
                raise RuntimeError("test did not release the gated controller step")
        self._last_command_seq = command_seq
        self.stepped.append((command_seq, observed.observation_seq, observed.game_tick))
        return make_action(
            command_sequence=command_seq,
            observation_sequence=observed.observation_seq,
            observation_tick=observed.game_tick,
            neural_start=index * 5,
            run_identity=observed.identity,
            motors=MotorVector(forward_thrust=0.4),
        )


def _log_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records]


def test_receiptless_accepts_retire_instead_of_filling_the_window(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Observations are accepted DURING a blocked step; no window-full fault.

    The mod never returns a receipt here. With inline compute, three per-tick
    accepts fill the two-slot window before observation-driven retirement can
    fire (retirement needs current_tick past the head's applicability) and the
    session dies with SequenceError("validated command window is full").
    De-blinded, the world keeps advancing during compute: retirement fires
    mid-step, so the window never fills even though every computed frame --
    including the one the world outran -- is sent.

    The window cannot fill: before the Nth action is accepted its observation was
    received, and that receive ran ``retire_expired_actions`` at its tick, which
    retires everything older than the (N-1)th. At most one earlier action can still
    hold a slot, and the window holds two. That is the invariant this drives four
    receipt-less accepts through.
    """

    caplog.set_level(logging.INFO)
    controller = GatedStepController()
    entered_0, release_0 = controller.gate_call(0)
    entered_1, release_1 = controller.gate_call(1)

    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(config(), controller=controller)
        try:
            await authenticate(websocket)

            # Cycle 1: the step for observation 0 blocks in its worker thread
            # while a further observation is pumped through the session.
            await websocket.send(encode_message(observation(sequence=0, tick=100)))
            assert await asyncio.to_thread(entered_0.wait, 2.0)
            await websocket.send(encode_message(observation(sequence=1, tick=101)))
            release_0.set()
            first = parse_message(await websocket.recv())
            assert isinstance(first, Action)
            assert first.command_seq == 0 and first.observation_seq == 0
            # Tick 101 is action 0's target, not past it: this frame is on time.
            assert not any(LATE_SEND_MESSAGE in message for message in _log_messages(caplog))
            # No receipt is sent: the accepted action occupies a window slot.

            # Cycle 2: observation 1 is stepped next and blocks; an observation
            # past action 0's target (101) arrives DURING the step and retires
            # it. This is the de-blinding proof: the inline service could not
            # run receive_frames until the step returned.
            assert await asyncio.to_thread(entered_1.wait, 2.0)
            await websocket.send(encode_message(observation(sequence=2, tick=105)))
            retired_at_105 = f"{RETIREMENT_MESSAGE}(s) the authority can no longer apply at game "
            for _ in range(1000):
                if any(retired_at_105 + "tick 105" in message for message in _log_messages(caplog)):
                    break
                await asyncio.sleep(0.005)
            assert controller.calls == 2 and len(controller.stepped) == 1  # step still blocked
            assert any(retired_at_105 + "tick 105" in message for message in _log_messages(caplog))

            # Released, the action computed from tick 101 (target 102) is behind
            # the accepted tick 105. It is sent late under its own sequence.
            release_1.set()
            second = parse_message(await websocket.recv())
            assert isinstance(second, Action)
            assert second.command_seq == 1
            assert second.observation_seq == 1 and second.observation_tick == 101

            # Third and fourth receipt-less accepts: retirement keeps pace, so the
            # window still never fills and no fault frame interrupts the stream.
            third = parse_message(await websocket.recv())
            assert isinstance(third, Action)
            assert third.command_seq == 2 and third.observation_seq == 2
            await websocket.send(encode_message(observation(sequence=3, tick=108)))
            fourth = parse_message(await websocket.recv())
            assert isinstance(fourth, Action)
            assert fourth.command_seq == 3 and fourth.observation_seq == 3

            assert controller.stepped == [(0, 0, 100), (1, 1, 101), (2, 2, 105), (3, 3, 108)]
            late_sends = [
                message for message in _log_messages(caplog) if LATE_SEND_MESSAGE in message
            ]
            assert len(late_sends) == 1
            assert "1 sent late this session" in late_sends[0]
            assert not any(DROP_MESSAGE in message for message in _log_messages(caplog))
        finally:
            release_0.set()
            release_1.set()
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_a_late_action_reaches_the_issued_action_and_telemetry_sinks(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A late frame is evidence like any other: it is recorded, not withheld.

    Under the removed drop, the withheld frame reached neither sink, so the trace
    and the latency record silently lost every cycle the world outran -- exactly the
    cycles a latency investigation needs. Both sinks now see it, in wire order.
    """

    caplog.set_level(logging.INFO)
    controller = GatedStepController()
    entered_0, release_0 = controller.gate_call(0)
    issued_actions: list[Action] = []
    telemetry: list[ActionTelemetry] = []

    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(
            config(),
            controller=controller,
            issued_action_sink=issued_actions.append,
            action_telemetry_sink=telemetry.append,
        )
        try:
            await authenticate(websocket)
            await websocket.send(encode_message(observation(sequence=0, tick=100)))
            assert await asyncio.to_thread(entered_0.wait, 2.0)
            # Accepted while the step is in flight; tick 103 is past the
            # in-flight action's valid_through of 102.
            await websocket.send(encode_message(observation(sequence=1, tick=103)))
            await asyncio.sleep(0.2)
            release_0.set()

            late = parse_message(await websocket.recv())
            assert isinstance(late, Action)
            assert late.command_seq == 0
            assert late.observation_seq == 0 and late.observation_tick == 100
            follow_on = parse_message(await websocket.recv())
            assert isinstance(follow_on, Action)
            assert follow_on.command_seq == 1
            assert follow_on.observation_seq == 1 and follow_on.observation_tick == 103
            assert controller.stepped == [(0, 0, 100), (1, 1, 103)]

            late_sends = [
                message for message in _log_messages(caplog) if LATE_SEND_MESSAGE in message
            ]
            assert len(late_sends) == 1
            assert "seq 0 computed from game tick 100" in late_sends[0]
            assert "world reached tick 103" in late_sends[0]
            assert "1 sent late this session" in late_sends[0]

            # Both sinks saw both frames, late one first; the session stays healthy.
            assert issued_actions == [late, follow_on]
            await websocket.send(encode_message(receipt(follow_on)))
            await websocket.send(encode_message(observation(sequence=2, tick=104)))
            third = parse_message(await websocket.recv())
            assert isinstance(third, Action)
            assert third.command_seq == 2 and third.observation_seq == 2
            for _ in range(50):
                if len(telemetry) == 3:
                    break
                await asyncio.sleep(0.001)
            assert [item.command_seq for item in telemetry] == [0, 1, 2]
            assert issued_actions == [late, follow_on, third]
        finally:
            release_0.set()
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("mid_step_tick", "expected_late_sends"),
    [(102, 0), (103, 1)],
    ids=["world_at_target_is_on_time", "world_past_target_is_late"],
)
def test_the_lateness_boundary_is_target_tick_and_neither_side_is_dropped(
    caplog: pytest.LogCaptureFixture, mid_step_tick: int, expected_late_sends: int
) -> None:
    """The late/on-time boundary is target_tick, and both sides of it are SENT.

    The in-flight action answers observation 100 and, under the two-tick action budget,
    targets 102: tick 102 is exactly its target, and tick 103 is past it. (The budget
    also collapses the validity window onto the target, because valid_through may not
    exceed observation + ACTION_TARGET_OFFSET_TICKS, so there is no longer a tick that
    is past the target but inside validity.) The authority applies a command on exactly
    its target tick, so the tick-103 frame can now only be retired -- ADR 0018 drew the
    same boundary and used it to withhold the frame. Withholding was the defect: it
    reused a sequence number the controller had already recorded. The boundary now
    decides only what is logged and counted, never what is sent, so the sequence the
    wire carries is the sequence the controller kept on both sides.
    """

    caplog.set_level(logging.INFO)
    controller = GatedStepController()
    entered_0, release_0 = controller.gate_call(0)

    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(config(), controller=controller)
        try:
            await authenticate(websocket)
            await websocket.send(encode_message(observation(sequence=0, tick=100)))
            assert await asyncio.to_thread(entered_0.wait, 2.0)
            await websocket.send(encode_message(observation(sequence=1, tick=mid_step_tick)))
            await asyncio.sleep(0.2)
            release_0.set()

            first = parse_message(await websocket.recv())
            assert isinstance(first, Action)
            assert first.command_seq == 0
            assert first.observation_seq == 0 and first.observation_tick == 100
            second = parse_message(await websocket.recv())
            assert isinstance(second, Action)
            assert second.command_seq == 1
            assert second.observation_seq == 1 and second.observation_tick == mid_step_tick

            late_sends = [
                message for message in _log_messages(caplog) if LATE_SEND_MESSAGE in message
            ]
            assert len(late_sends) == expected_late_sends
            if expected_late_sends:
                assert "seq 0 computed from game tick 100" in late_sends[0]
                assert f"world reached tick {mid_step_tick}" in late_sends[0]
        finally:
            release_0.set()
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_pending_control_effect_still_materializes_on_a_late_action(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A late action carrying a pending control effect is sent and acknowledged on schedule.

    Accepted action-effect controls acknowledge against their first governed
    action, and that deadline semantics must not change. This was the one case the
    removed drop already exempted -- and the exemption is what proves the always-send
    path was always sound: it ran on every control-bearing late frame. It is now the
    only path, so this case is no longer special, and the late frame is logged as late
    here too.
    """

    caplog.set_level(logging.INFO)
    controller = GatedStepController()
    entered_0, release_0 = controller.gate_call(0)
    handler = RecordingControlHandler()

    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(
            config(), controller=controller, control_handler=handler
        )
        try:
            await authenticate(websocket)
            command = ControlCommand(
                identity=identity(),
                control_seq=0,
                issued_tick=100,
                operation=ControlOperation.STIMULATION_APPLY,
                content_id="declared-stimulation",
                content_hash="b" * 64,
            )
            await websocket.send(encode_message(command))
            # No acknowledgement yet: an action-effect control materializes
            # against the next issued action, so the effect is pending when the
            # step below acquires the execution fence.
            await websocket.send(encode_message(observation(sequence=0, tick=100)))
            assert await asyncio.to_thread(entered_0.wait, 2.0)
            await websocket.send(encode_message(observation(sequence=1, tick=103)))
            await asyncio.sleep(0.2)
            release_0.set()

            acknowledgement = parse_message(await websocket.recv())
            late = parse_message(await websocket.recv())
            assert isinstance(acknowledgement, ControlAck)
            assert isinstance(late, Action)
            assert acknowledgement.accepted
            assert late.command_seq == 0 and late.observation_seq == 0
            assert acknowledgement.effective_tick == late.observation_tick == 100
            assert handler.commands == [command]
            late_sends = [
                message for message in _log_messages(caplog) if LATE_SEND_MESSAGE in message
            ]
            assert len(late_sends) == 1
            assert "seq 0 computed from game tick 100" in late_sends[0]
            assert not any(DROP_MESSAGE in message for message in _log_messages(caplog))

            follow_up = parse_message(await websocket.recv())
            assert isinstance(follow_up, Action)
            assert follow_up.command_seq == 1 and follow_up.observation_seq == 1
        finally:
            release_0.set()
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_a_late_action_is_sent_instead_of_reusing_its_command_sequence(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The live defect of 2026-09-03: a dropped frame reuses a sequence the controller kept.

    Two program-lane sessions died here (``interactive-20260903T233114Z-eb5b518b``
    and ``interactive-20260903T233246Z-252075e6``, operator-observed logs, gitignored):
    action latencies of 87-99 ms against a 50 ms tick made the world outrun almost every
    action, the service dropped each one and deliberately did not advance
    ``command_counter``, and the very next ``controller.step`` raised
    ``ControllerIntegrityError("command sequence is stale or duplicated")`` because the
    controller had already recorded that sequence inside the dropped step. The service
    fails closed on it (``INTERNAL``), the mod reports ``bridge=INTERNAL_ERROR``, and the
    launcher exits.

    The service never had authority to reuse a sequence number: recording it is the
    controller's, and it happens before the service sees the action at all.
    """

    caplog.set_level(logging.INFO)
    controller = GatedStepController()
    entered_0, release_0 = controller.gate_call(0)

    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(config(), controller=controller)
        try:
            await authenticate(websocket)
            await websocket.send(encode_message(observation(sequence=0, tick=100)))
            assert await asyncio.to_thread(entered_0.wait, 2.0)
            # The world outruns the in-flight step: tick 103 is past its target of 101.
            await websocket.send(encode_message(observation(sequence=1, tick=103)))
            await asyncio.sleep(0.2)
            release_0.set()

            # The late frame is sent under its own sequence, exactly as an on-time
            # frame is. The mod retires it fail-neutral (ADR 0011) and this service's
            # observation-driven retirement frees its window slot on the next
            # observation; neither needs the sequence number back.
            late = parse_message(await websocket.recv())
            assert isinstance(late, Action), f"expected the late action, got {late!r}"
            assert late.command_seq == 0
            assert late.observation_seq == 0 and late.observation_tick == 100

            replacement = parse_message(await websocket.recv())
            assert isinstance(replacement, Action)
            assert replacement.command_seq == 1
            assert replacement.observation_seq == 1 and replacement.observation_tick == 103

            # A third step succeeds: the session is still alive and the sequence
            # the controller recorded is the sequence the wire carries.
            await websocket.send(encode_message(observation(sequence=2, tick=104)))
            third = parse_message(await websocket.recv())
            assert isinstance(third, Action)
            assert third.command_seq == 2 and third.observation_seq == 2
            assert controller.stepped == [(0, 0, 100), (1, 1, 103), (2, 2, 104)]

            late_sends = [
                message for message in _log_messages(caplog) if LATE_SEND_MESSAGE in message
            ]
            assert len(late_sends) == 1
            assert "seq 0 computed from game tick 100" in late_sends[0]
            assert "world reached tick 103" in late_sends[0]
            assert "1 sent late this session" in late_sends[0]
        finally:
            release_0.set()
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_a_late_send_keeps_command_and_neural_step_continuity(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Both continuity invariants the removed drop comment cited still hold across a late frame.

    The old drop comment claimed reusing the sequence number preserved the service
    validator's ``last_command_seq + 1`` continuity and the mod's
    ``commandSequence == last + 1`` continuity. Sending the frame preserves them for the
    honest reason -- nothing is skipped -- and it additionally repairs the neural
    interval, which the drop broke: the dropped frame's neural steps had already been
    executed, so the next frame's ``neural_step_start`` opened past the last
    ``neural_step_end`` the mod had seen. Every sent frame here is contiguous:
    ``neural_step_start == previous neural_step_end``, exactly.
    """

    caplog.set_level(logging.INFO)
    controller = GatedStepController()
    entered_0, release_0 = controller.gate_call(0)
    sent: list[Action] = []

    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(config(), controller=controller)
        try:
            await authenticate(websocket)
            await websocket.send(encode_message(observation(sequence=0, tick=100)))
            assert await asyncio.to_thread(entered_0.wait, 2.0)
            await websocket.send(encode_message(observation(sequence=1, tick=102)))
            await asyncio.sleep(0.2)
            release_0.set()
            for _ in range(2):
                frame = parse_message(await websocket.recv())
                assert isinstance(frame, Action)
                sent.append(frame)
            await websocket.send(encode_message(observation(sequence=2, tick=103)))
            frame = parse_message(await websocket.recv())
            assert isinstance(frame, Action)
            sent.append(frame)
        finally:
            release_0.set()
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())

    assert [frame.command_seq for frame in sent] == [0, 1, 2]
    intervals = [(frame.neural_step_start, frame.neural_step_end) for frame in sent]
    assert intervals == [(0, 5), (5, 10), (10, 15)]
    for previous, following in itertools.pairwise(sent):
        assert following.command_seq == previous.command_seq + 1
        assert following.neural_step_start == previous.neural_step_end

    # A peer validator admits the whole sequence, late frame included: the mod's
    # CommandValidator applies the same sequence-continuity and strictly-forward
    # neural-interval rules over the frames the wire actually carried.
    peer = SessionValidator(identity())
    for frame in sent:
        peer.accept_observation(
            observation(sequence=frame.observation_seq, tick=frame.observation_tick),
            current_tick=frame.observation_tick,
        )
        peer.accept_action(frame, current_tick=frame.observation_tick)
    assert peer.last_command_seq == 2
    assert peer.last_neural_step_end == 15


def test_a_step_outrun_by_three_observations_is_sent_and_left_to_the_authority() -> None:
    """One tick late is D-080's case, and the ledger has to be wide enough to reach it.

    The step answers observation 100 and targets 102 under the two-tick budget; the world
    reaches 103 while it computes, so it is one tick late. It must still be SENT: the authority
    applies only on ``target_tick``, so it retires the frame fail-neutral (ADR 0011) and this
    service's observation-driven retirement frees its window slot. If the observation ledger sat
    exactly on the on-time boundary, this frame would instead lose its binding and fail the
    session closed on STALE -- which is why the ledger is one entry wider than that boundary.
    """

    controller = GatedStepController()
    entered_0, release_0 = controller.gate_call(0)

    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(config(), controller=controller)
        try:
            await authenticate(websocket)
            await websocket.send(encode_message(observation(sequence=0, tick=100)))
            assert await asyncio.to_thread(entered_0.wait, 2.0)
            for sequence, tick in ((1, 101), (2, 102), (3, 103)):
                await websocket.send(encode_message(observation(sequence=sequence, tick=tick)))
            await asyncio.sleep(0.2)
            release_0.set()

            late = parse_message(await websocket.recv())
            assert isinstance(late, Action), f"expected the late action, got {late!r}"
            assert late.command_seq == 0
            assert late.observation_tick == 100
            assert late.target_tick == 100 + ACTION_TARGET_OFFSET_TICKS
        finally:
            release_0.set()
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


def _accepted_observation_gate(monkeypatch: pytest.MonkeyPatch, sequence: int) -> threading.Event:
    """Release tests only after the real validator has accepted the last outrunning frame."""

    accepted = threading.Event()
    original = SessionValidator.accept_observation

    def record(self, observed, *, current_tick=None):
        original(self, observed, current_tick=current_tick)
        if observed.observation_seq == sequence:
            accepted.set()

    monkeypatch.setattr(SessionValidator, "accept_observation", record)
    return accepted


@pytest.mark.parametrize("outrunning", range(4, 8))
@pytest.mark.parametrize("program_lane", [False, True])
def test_a_step_within_authority_history_is_sent_retired_and_followed_by_fresh_command(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    outrunning: int,
    program_lane: bool,
) -> None:
    """The live four-entry cliff must not preempt the authority's existing neutral hold.

    Send the completed late frame unchanged, never reuse its sequence, and let the next
    observation retire its service reservation. The Java companion test pins actual neutral
    retirement and the unchanged starvation deadline; no receipt is invented for the late frame.
    """

    caplog.set_level(logging.INFO)
    stepped = GatedStepController()
    entered, release = stepped.gate_call(0)
    accepted = _accepted_observation_gate(monkeypatch, outrunning)

    class Controller:
        def step(self, observed, *, command_seq):
            frame = stepped.step(observed, command_seq=command_seq)
            if not program_lane:
                return frame
            return make_program_action(
                command_sequence=frame.command_seq,
                observation_sequence=frame.observation_seq,
                observation_tick=frame.observation_tick,
                neural_start=frame.neural_step_start,
                run_identity=frame.identity,
                state_digest=frame.state_digest,
            )

    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(
            config(), controller=Controller()
        )
        try:
            await authenticate(websocket)
            await websocket.send(encode_message(observation(sequence=0, tick=100)))
            assert await asyncio.to_thread(entered.wait, 2.0)
            for sequence in range(1, outrunning + 1):
                await websocket.send(
                    encode_message(observation(sequence=sequence, tick=100 + sequence))
                )
            assert await asyncio.to_thread(accepted.wait, 2.0)
            release.set()

            late = parse_message(await asyncio.wait_for(websocket.recv(), 2.0))
            expected_type = ProgramAction if program_lane else Action
            assert isinstance(late, expected_type), late
            fresh = parse_message(await asyncio.wait_for(websocket.recv(), 2.0))
            assert isinstance(fresh, expected_type), fresh
            assert (late.command_seq, late.observation_seq, late.observation_tick) == (0, 0, 100)
            assert late.identity == identity() and late.state_digest == "6" * 64
            assert late.target_tick == late.valid_through_tick == 102
            assert (fresh.command_seq, fresh.observation_seq) == (1, outrunning)
            assert fresh.observation_tick == 100 + outrunning
            assert fresh.neural_step_start == late.neural_step_end == 5

            # Observation retirement consumes only the late reservation. This later applied
            # receipt still binds, and a further command proves the session remains usable.
            await websocket.send(
                encode_message(observation(sequence=outrunning + 1, tick=101 + outrunning))
            )
            third = parse_message(await asyncio.wait_for(websocket.recv(), 2.0))
            assert isinstance(third, expected_type)
            assert third.command_seq == 2 and third.neural_step_start == 10
            await websocket.send(
                encode_message(observation(sequence=outrunning + 2, tick=102 + outrunning))
            )
            fourth = parse_message(await asyncio.wait_for(websocket.recv(), 2.0))
            assert isinstance(fourth, expected_type)
            assert fourth.command_seq == 3 and fourth.neural_step_start == 15
            # Match the authority's wire order: target-tick observation, then its receipt,
            # then the following observation. The fifth response proves receipt acceptance.
            acknowledged = program_receipt(fresh) if program_lane else receipt(fresh)
            await websocket.send(encode_message(acknowledged))
            await websocket.send(
                encode_message(observation(sequence=outrunning + 3, tick=103 + outrunning))
            )
            fifth = parse_message(await asyncio.wait_for(websocket.recv(), 2.0))
            assert isinstance(fifth, expected_type)
            assert fifth.command_seq == 4 and fifth.neural_step_start == 20
            assert stepped.stepped == [
                (0, 0, 100),
                (1, outrunning, 100 + outrunning),
                (2, outrunning + 1, 101 + outrunning),
                (3, outrunning + 2, 102 + outrunning),
                (4, outrunning + 3, 103 + outrunning),
            ]
            retirements = [
                message for message in _log_messages(caplog) if RETIREMENT_MESSAGE in message
            ]
            assert len(retirements) == 1
            assert "1 retired this session" in retirements[0]
        finally:
            release.set()
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


@pytest.mark.parametrize("outrunning", [8, 9])
def test_a_step_outrun_beyond_authority_history_still_fails_closed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, outrunning: int
) -> None:
    """Eight newer observations evict the binding on both sides; STALE remains terminal."""

    controller = GatedStepController()
    entered_0, release_0 = controller.gate_call(0)
    accepted = _accepted_observation_gate(monkeypatch, outrunning)

    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(config(), controller=controller)
        try:
            await authenticate(websocket)
            await websocket.send(encode_message(observation(sequence=0, tick=100)))
            assert await asyncio.to_thread(entered_0.wait, 2.0)
            for sequence in range(1, outrunning + 1):
                await websocket.send(
                    encode_message(observation(sequence=sequence, tick=100 + sequence))
                )
            assert await asyncio.to_thread(accepted.wait, 2.0)
            release_0.set()

            fault = parse_message(await asyncio.wait_for(websocket.recv(), 2.0))
            assert isinstance(fault, Fault)
            assert fault.code is FaultCode.STALE
            assert fault.neutral_applied
            context = [
                record
                for record in caplog.records
                if record.msg.startswith("completed command refused")
            ]
            assert len(context) == 1
            assert context[0].args[:7] == (0, 0, 100, 102, outrunning, 100 + outrunning, outrunning)
            assert isinstance(context[0].args[7], int) and context[0].args[7] > 0
            assert context[0].args[8] == identity().session_epoch
        finally:
            release_0.set()
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())


@pytest.mark.parametrize("counterfeit", ["identity", "tick", "unseen_observation"])
def test_retained_history_does_not_admit_a_counterfeit_completed_binding(
    monkeypatch: pytest.MonkeyPatch, counterfeit: str
) -> None:
    controller = GatedStepController()
    entered, release = controller.gate_call(0)
    accepted = _accepted_observation_gate(monkeypatch, 7)
    step = controller.step

    def forged(observed, *, command_seq):
        frame = step(observed, command_seq=command_seq)
        if counterfeit == "identity":
            return replace(frame, identity=identity(session_epoch=2))
        if counterfeit == "tick":
            return replace(frame, observation_tick=99, target_tick=101, valid_through_tick=101)
        return replace(frame, observation_seq=8)

    monkeypatch.setattr(controller, "step", forged)

    async def scenario() -> None:
        server_context, websocket = await authenticated_connection(config(), controller=controller)
        try:
            await authenticate(websocket)
            await websocket.send(encode_message(observation(sequence=0, tick=100)))
            assert await asyncio.to_thread(entered.wait, 2.0)
            for sequence in range(1, 8):
                await websocket.send(
                    encode_message(observation(sequence=sequence, tick=100 + sequence))
                )
            assert await asyncio.to_thread(accepted.wait, 2.0)
            release.set()
            fault = parse_message(await asyncio.wait_for(websocket.recv(), 2.0))
            assert isinstance(fault, Fault)
            assert fault.code is (
                FaultCode.IDENTITY_MISMATCH if counterfeit == "identity" else FaultCode.SEQUENCE
            )
            assert fault.neutral_applied
        finally:
            release.set()
            await websocket.close()
            await server_context.__aexit__(None, None, None)

    asyncio.run(scenario())
