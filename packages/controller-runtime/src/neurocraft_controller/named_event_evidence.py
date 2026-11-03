"""Immutable issued-decision diagnostic; no writer, queue, or motor authority.

Consumers may display APPLIED only after matching the complete command envelope
against authoritative receipts. Missing or malformed display is unavailable, not
an invitation to infer values. Means are signed model activity, not spike rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from neurocraft_protocol import EventSample, ProgramAction, canonical_json_bytes

from .config import canonical_sha256
from .events_senses import CompletedEventReadout

if TYPE_CHECKING:
    from .event_program_policy import EventScore


@dataclass(frozen=True, slots=True)
class EventSelectionTrigger:
    """Original issued threshold crossing, retained through bounded LAND and FEED."""

    command: ProgramAction
    readout: str
    score: EventScore
    events: EventSample

    def to_dict(self) -> dict[str, Any]:
        frame = self.command
        return {
            "identity": frame.identity.to_dict(),
            "command_seq": frame.command_seq,
            "observation_seq": frame.observation_seq,
            "observation_tick": frame.observation_tick,
            "target_tick": frame.target_tick,
            "valid_through_tick": frame.valid_through_tick,
            "neural_step_start": frame.neural_step_start,
            "neural_step_end": frame.neural_step_end,
            "state_digest": frame.state_digest,
            "policy_hash": frame.policy_hash,
            "issued_program": frame.program.value,
            "issued_command": frame.to_dict(),
            "command_sha256": canonical_sha256(frame.to_dict()),
            "readout": self.readout,
            "score": self.score.to_dict(),
            "events": self.events.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class NamedEventEvidence:
    command: ProgramAction
    encoder_hash: str
    binding_hash: str
    completed: CompletedEventReadout
    mn9: EventScore
    antenna: EventScore
    reason: str
    events: EventSample
    trigger: EventSelectionTrigger | None

    def to_dict(self) -> dict[str, Any]:
        frame = self.command
        return {
            "schema": "neurocraft.event_selection_display.v1",
            "presentation_only": True,
            "selection_status": "issued; requires authoritative applied-command match",
            "identity": frame.identity.to_dict(),
            "command_seq": frame.command_seq,
            "observation_seq": frame.observation_seq,
            "observation_tick": frame.observation_tick,
            "target_tick": frame.target_tick,
            "valid_through_tick": frame.valid_through_tick,
            "neural_step_start": frame.neural_step_start,
            "neural_step_end": frame.neural_step_end,
            "state_digest": frame.state_digest,
            "policy_hash": frame.policy_hash,
            "encoder_hash": self.encoder_hash,
            "binding_hash": self.binding_hash,
            "program": frame.program.value,
            "modulation": frame.modulation.to_dict(),
            "parameters": frame.parameters.to_dict(),
            "readout": frame.readout.to_dict(),
            "command_sha256": canonical_sha256(frame.to_dict()),
            "reason": self.reason,
            "issued_trigger": None if self.trigger is None else self.trigger.to_dict(),
            "events": self.events.to_dict(),
            "requested_pulse": self.completed.pulse.to_dict(),
            "pulse_scope": "requested before final control input gate; no_input delivers zero",
            "mn9": {
                **self.mn9.to_dict(),
                "left": self.completed.mn9_left,
                "right": self.completed.mn9_right,
                "left_count": 1,
                "right_count": 1,
            },
            "antenna": {
                **self.antenna.to_dict(),
                "left": self.completed.antenna_left,
                "right": self.completed.antenna_right,
                "left_count": 6,
                "right_count": 7,
            },
            "quantity": "signed dimensionless model activity; not spike rates",
            "mapping": "engineered eligibility and hand-chosen readouts select scripted programs",
        }

    def json_bytes(self) -> bytes:
        """For a future optional publisher; serialization does not modify the controller."""
        return canonical_json_bytes(self.to_dict())
