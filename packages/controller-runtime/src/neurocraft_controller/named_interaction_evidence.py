"""Versioned issued interaction evidence; presentation never grants motor authority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from neurocraft_protocol import ProgramAction, SensorFrameV4, canonical_json_bytes

from .events_senses import CompletedEventReadout
from .named_event_evidence import EventSelectionTrigger, NamedEventEvidence

if TYPE_CHECKING:
    from .interaction_program_policy import InteractionDecision


SOURCE_CHANNELS: Final = {
    "attack": ("absolute_change", "ttmn_raw"),
    "player_approach": ("absolute_readout", "ttmn_jump"),
    "mob_approach": ("absolute_readout", "ttmn_jump"),
    "food": ("positive_change", "mn9"),
    "brush": ("positive_change", "antenna"),
    "light": ("absolute_change", "dna01_bilateral"),
}


@dataclass(frozen=True, slots=True)
class InteractionSelectionTrigger(EventSelectionTrigger):
    source: str
    formula: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **EventSelectionTrigger.to_dict(self),
            "source": self.source,
            "formula": self.formula,
        }


@dataclass(frozen=True, slots=True)
class NamedInteractionEvidence:
    command: ProgramAction
    encoder_hash: str
    binding_hash: str
    completed: CompletedEventReadout
    decision: InteractionDecision
    sensors: SensorFrameV4
    ttmn_raw: float
    dna01_means: tuple[float, float]
    trigger: InteractionSelectionTrigger | None

    def to_dict(self) -> dict[str, Any]:
        decision = self.decision
        base = NamedEventEvidence(
            self.command,
            self.encoder_hash,
            self.binding_hash,
            self.completed,
            decision.scores["food"],
            decision.scores["brush"],
            decision.reason,
            self.sensors.events,
            self.trigger,
        ).to_dict()
        return {
            **base,
            "schema": "neurocraft.interaction_selection_display.v2",
            "interaction": {
                "source": decision.source,
                "scores": {
                    source: {
                        **decision.scores[source].to_dict(),
                        "formula": formula,
                        "readout": readout,
                    }
                    for source, (formula, readout) in SOURCE_CHANNELS.items()
                },
                "raw_ttmn": self.ttmn_raw,
                "dna01_left": self.dna01_means[0],
                "dna01_right": self.dna01_means[1],
                "ray_hit_kinds": list(self.sensors.ray_hit_kinds),
                "ray_actor_kinds": list(self.sensors.ray_actor_kinds),
                "local_light": self.sensors.local_light,
            },
        }

    def json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())
