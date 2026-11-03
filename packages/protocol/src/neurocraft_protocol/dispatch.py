"""The single parser and encoder for every `neurocraft.protocol.v2` frame.

This module exists so the frame modules stay acyclic. ``program_messages`` imports
``messages`` for the shared validators and value objects, so ``messages`` cannot import
``program_messages`` back; the dispatch that has to know both therefore sits above both.
Everything in the repository reaches these three names through the package's ``__init__``,
so the move costs no call site.

There is exactly one ``parse_message`` on this wire. A second parser that knew only some of
the frame kinds would silently refuse the rest, which is precisely the failure the closed
key sets and the unknown-type refusal exist to prevent.
"""

from __future__ import annotations

from typing import TypeAlias

from .canonical import encode_json_frame, parse_json_frame
from .errors import ProtocolError
from .messages import (
    Action,
    ActuationReceipt,
    ControlAck,
    ControlCommand,
    Fault,
    Hello,
    HelloAck,
    Observation,
)
from .program_messages import ProgramAction, ProgramReceipt

ProtocolMessage: TypeAlias = (
    Hello
    | HelloAck
    | Observation
    | Action
    | ProgramAction
    | ActuationReceipt
    | ProgramReceipt
    | ControlCommand
    | ControlAck
    | Fault
)
_MESSAGE_TYPES = {
    Hello.TYPE: Hello,
    HelloAck.TYPE: HelloAck,
    Observation.TYPE: Observation,
    Action.TYPE: Action,
    ProgramAction.TYPE: ProgramAction,
    ActuationReceipt.TYPE: ActuationReceipt,
    ProgramReceipt.TYPE: ProgramReceipt,
    ControlCommand.TYPE: ControlCommand,
    ControlAck.TYPE: ControlAck,
    Fault.TYPE: Fault,
}


def parse_message(frame: str | bytes) -> ProtocolMessage:
    value = parse_json_frame(frame)
    message_type = value.get("type")
    if not isinstance(message_type, str) or message_type not in _MESSAGE_TYPES:
        raise ProtocolError("message type is missing or unknown")
    if message_type == Hello.TYPE:
        return Hello.from_dict(value)
    if message_type == HelloAck.TYPE:
        return HelloAck.from_dict(value)
    if message_type == Observation.TYPE:
        return Observation.from_dict(value)
    if message_type == Action.TYPE:
        return Action.from_dict(value)
    if message_type == ProgramAction.TYPE:
        return ProgramAction.from_dict(value)
    if message_type == ActuationReceipt.TYPE:
        return ActuationReceipt.from_dict(value)
    if message_type == ProgramReceipt.TYPE:
        return ProgramReceipt.from_dict(value)
    if message_type == ControlCommand.TYPE:
        return ControlCommand.from_dict(value)
    if message_type == ControlAck.TYPE:
        return ControlAck.from_dict(value)
    return Fault.from_dict(value)


def encode_message(message: ProtocolMessage) -> str:
    if not isinstance(message, tuple(_MESSAGE_TYPES.values())):
        raise ProtocolError("value is not a protocol message")
    return encode_json_frame(message)
