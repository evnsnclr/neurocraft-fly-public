"""Compose intervention and receipt-trace controls without fake capabilities."""

from __future__ import annotations

from neurocraft_protocol import ControlCommand, ControlOperation, FaultCode

from .controls import ConnectomeControlHandler, InterventionControlError
from .trace import MinecraftReceiptTraceRecorder


class ConnectomeRuntimeControlHandler:
    """Delegate only configured recording and named-intervention operations."""

    def __init__(
        self,
        *,
        interventions: ConnectomeControlHandler | None = None,
        recorder: MinecraftReceiptTraceRecorder | None = None,
    ) -> None:
        self.interventions = interventions
        self.recorder = recorder

    def handle_control(self, command: ControlCommand) -> None:
        if command.operation in {ControlOperation.RECORD_START, ControlOperation.RECORD_STOP}:
            if self.recorder is None:
                raise InterventionControlError(
                    FaultCode.CONTROL_UNAVAILABLE, "receipt trace recording is not configured"
                )
            self.recorder.handle_control(command)
            return
        if command.operation in {ControlOperation.REPLAY_START, ControlOperation.REPLAY_STOP}:
            raise InterventionControlError(
                FaultCode.CONTROL_UNAVAILABLE, "no explicit replay engine is configured"
            )
        if command.operation in {
            ControlOperation.LESION_APPLY,
            ControlOperation.LESION_CLEAR,
            ControlOperation.STIMULATION_APPLY,
            ControlOperation.STIMULATION_CLEAR,
        }:
            if self.interventions is None:
                raise InterventionControlError(
                    FaultCode.CONTROL_UNAVAILABLE,
                    "named intervention registry is not configured",
                )
            self.interventions.handle_control(command)
            return
        raise InterventionControlError(
            FaultCode.CONTROL_UNAVAILABLE,
            "control operation is owned by the service state machine",
        )
