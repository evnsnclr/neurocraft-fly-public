"""Strict world-event counters for the opt-in engineered sensor revision.

These report Minecraft interactions, not biological sensation or health damage.
Counters survive transport coalescing; interpretation and reset baselines belong
to the separately identified encoder.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .errors import ProtocolError, UnknownFieldError

MAX_EVENT_INTEGER = 2**63 - 1


def _integer(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= MAX_EVENT_INTEGER:
        raise ProtocolError(f"{name} must be an integer in 0..{MAX_EVENT_INTEGER}")
    return value


def _object(value: Any, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ProtocolError(f"{name} must be an object with string keys")
    if set(value) - keys:
        raise UnknownFieldError(f"{name} contains unknown fields")
    if set(value) != keys:
        raise ProtocolError(f"{name} is missing fields")
    return value


@dataclass(frozen=True, slots=True)
class EventCounter:
    count: int
    last_game_tick: int | None

    def __post_init__(self) -> None:
        _integer(self.count, "event.count")
        if (self.count == 0) != (self.last_game_tick is None):
            raise ProtocolError("event count is zero if and only if last_game_tick is null")
        if self.last_game_tick is not None:
            _integer(self.last_game_tick, "event.last_game_tick")

    @classmethod
    def from_dict(cls, raw: Any) -> EventCounter:
        value = _object(raw, {"count", "last_game_tick"}, "event counter")
        return cls(value["count"], value["last_game_tick"])

    def to_dict(self) -> dict[str, int | None]:
        return {"count": self.count, "last_game_tick": self.last_game_tick}


@dataclass(frozen=True, slots=True)
class EventSample:
    food_contact: bool
    food_near: bool
    food: EventCounter
    brush: EventCounter
    attack_attempt: EventCounter

    def __post_init__(self) -> None:
        if type(self.food_contact) is not bool:
            raise ProtocolError("events.food_contact must be boolean")
        if type(self.food_near) is not bool:
            raise ProtocolError("events.food_near must be boolean")
        if self.food_contact and not self.food_near:
            raise ProtocolError("events.food_contact requires food_near")
        if any(not isinstance(getattr(self, name), EventCounter) for name in self.counter_names()):
            raise ProtocolError("events requires three EventCounter values")

    @staticmethod
    def counter_names() -> tuple[str, ...]:
        return ("food", "brush", "attack_attempt")

    @classmethod
    def from_dict(cls, raw: Any) -> EventSample:
        value = _object(raw, {"food_contact", "food_near", *cls.counter_names()}, "events")
        return cls(
            food_contact=value["food_contact"],
            food_near=value["food_near"],
            food=EventCounter.from_dict(value["food"]),
            brush=EventCounter.from_dict(value["brush"]),
            attack_attempt=EventCounter.from_dict(value["attack_attempt"]),
        )

    def require_observation_tick(self, game_tick: int) -> None:
        _integer(game_tick, "observation.game_tick")
        for name in self.counter_names():
            tick = getattr(self, name).last_game_tick
            if tick is not None and tick > game_tick:
                raise ProtocolError(f"events.{name} occurs after the observation tick")

    def to_dict(self) -> dict[str, Any]:
        return {
            "food_contact": self.food_contact,
            "food_near": self.food_near,
            **{name: getattr(self, name).to_dict() for name in self.counter_names()},
        }
