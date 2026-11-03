"""Controller boundary and the Milestone-1 synthetic neutral fake."""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

from neurocraft_protocol import (
    ACTION_TARGET_OFFSET_TICKS,
    Action,
    Command,
    ControlCommand,
    MotorVector,
    Observation,
    canonical_json_bytes,
)


class Controller(Protocol):
    """A replaceable neural-process boundary; implementations return one driving command.

    A controller returns either an ``action`` (bounded primitive motors) or a ``program_action``
    (a scripted body-program selection with the labeled readout scalars a declared mapping
    consumed). One profile returns one kind for the whole session: the frame kind is a property
    of the controller, not a per-tick choice, and the authority latches the lane independently.
    """

    def step(self, observation: Observation, *, command_seq: int) -> Command: ...


@runtime_checkable
class ResettableController(Controller, Protocol):
    def reset(self) -> None: ...


class ControlHandler(Protocol):
    """Explicit delegate for recording/replay lifecycle operations."""

    def handle_control(self, command: ControlCommand) -> None: ...


class DeterministicNeutralController:
    """Transport test double, never a Faithful Organism controller.

    It returns an explicitly neutral command and hashes only the received observation.
    This exercises handshake, sequencing, identity, and failure behavior without hiding
    a behavioral policy in the Milestone-1 fake service.
    """

    def step(self, observation: Observation, *, command_seq: int) -> Action:
        step_start = observation.observation_seq * 5
        digest = hashlib.sha256(
            canonical_json_bytes(
                {
                    "controller": "synthetic_neutral_v1",
                    "observation": observation.to_dict(),
                    "step_start": step_start,
                }
            )
        ).hexdigest()
        return Action(
            identity=observation.identity,
            command_seq=command_seq,
            observation_seq=observation.observation_seq,
            observation_tick=observation.game_tick,
            target_tick=observation.game_tick + ACTION_TARGET_OFFSET_TICKS,
            valid_through_tick=observation.game_tick + ACTION_TARGET_OFFSET_TICKS,
            neural_step_start=step_start,
            neural_step_end=step_start + 5,
            state_digest=digest,
            motors=MotorVector.neutral(),
            neutral=True,
        )

    def reset(self) -> None:
        """Explicit reset capability; this stateless fake has no mutable model state."""
