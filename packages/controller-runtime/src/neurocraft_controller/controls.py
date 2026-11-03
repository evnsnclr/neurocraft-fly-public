"""Hash-bound named neural interventions for the authenticated control channel."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

from neurocraft_neural import Stimulation
from neurocraft_protocol import ControlCommand, ControlOperation, FaultCode

from .config import canonical_sha256
from .controller import ConnectomeController, ControllerIntegrityError

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class InterventionControlError(ControllerIntegrityError):
    """Expected rejection that the service can translate without mutating state."""

    def __init__(self, code: FaultCode, message: str) -> None:
        if code not in {FaultCode.CONTROL_UNAVAILABLE, FaultCode.INVALID_STATE}:
            raise ValueError("intervention rejection must use a control fault code")
        super().__init__(message)
        self.code = code.value


def _content_id(value: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError("intervention content_id must be a bounded protocol identifier")
    return value


@dataclass(frozen=True, slots=True)
class NamedLesion:
    content_id: str
    graph_hash: str
    body_ids: tuple[int, ...]
    content_hash: str

    @classmethod
    def create(
        cls,
        controller: ConnectomeController,
        content_id: str,
        body_ids: Sequence[int],
    ) -> NamedLesion:
        resolved_id = _content_id(content_id)
        resolved_body_ids = tuple(sorted(body_ids))
        if not resolved_body_ids or len(set(resolved_body_ids)) != len(resolved_body_ids):
            raise ValueError("named lesion requires unique body IDs")
        for body_id in resolved_body_ids:
            if isinstance(body_id, bool) or not isinstance(body_id, int) or body_id <= 0:
                raise ValueError("named lesion body IDs must be positive integers")
            try:
                controller.graph.index_of(body_id)
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
        graph_hash = controller.identity.graph_hash
        material = {
            "body_ids": list(resolved_body_ids),
            "content_id": resolved_id,
            "graph_hash": graph_hash,
            "schema": "neurocraft.named_lesion.v1",
        }
        return cls(
            content_id=resolved_id,
            graph_hash=graph_hash,
            body_ids=resolved_body_ids,
            content_hash=canonical_sha256(material),
        )


@dataclass(frozen=True, slots=True)
class NamedStimulation:
    content_id: str
    graph_hash: str
    body_values: Mapping[int, float]
    content_hash: str

    @classmethod
    def create(
        cls,
        controller: ConnectomeController,
        content_id: str,
        body_values: Mapping[int, float],
    ) -> NamedStimulation:
        resolved_id = _content_id(content_id)
        if not isinstance(body_values, Mapping) or not body_values:
            raise ValueError("named stimulation requires a non-empty body-value mapping")
        resolved: dict[int, float] = {}
        for body_id, raw_value in body_values.items():
            if isinstance(body_id, bool) or not isinstance(body_id, int) or body_id <= 0:
                raise ValueError("stimulation body IDs must be positive integers")
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError("stimulation values must be finite numbers")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError("stimulation values must be finite numbers")
            resolved[body_id] = value
        stimulation = Stimulation.from_body_values(controller.graph, resolved_id, resolved)
        return cls(
            content_id=resolved_id,
            graph_hash=controller.identity.graph_hash,
            body_values=MappingProxyType(dict(sorted(resolved.items()))),
            content_hash=stimulation.identity_sha256,
        )


NamedIntervention: TypeAlias = NamedLesion | NamedStimulation


class ConnectomeControlHandler:
    """Apply only predeclared, graph-bound lesion and stimulation controls."""

    def __init__(
        self,
        controller: ConnectomeController,
        interventions: Sequence[NamedIntervention],
    ) -> None:
        registry: dict[str, NamedIntervention] = {}
        lesion_bodies: set[int] = set()
        for intervention in interventions:
            if intervention.graph_hash != controller.identity.graph_hash:
                raise ValueError("intervention graph identity does not match controller")
            if intervention.content_id in registry:
                raise ValueError("intervention content IDs must be unique")
            if isinstance(intervention, NamedLesion):
                overlap = lesion_bodies.intersection(intervention.body_ids)
                if overlap:
                    raise ValueError("named lesion registry entries cannot overlap")
                lesion_bodies.update(intervention.body_ids)
            registry[intervention.content_id] = intervention
        self.controller = controller
        self._registry = MappingProxyType(registry)

    @property
    def content_hashes(self) -> Mapping[str, str]:
        return MappingProxyType(
            {content_id: item.content_hash for content_id, item in self._registry.items()}
        )

    def handle_control(self, command: ControlCommand) -> None:
        operation = command.operation
        if operation not in {
            ControlOperation.LESION_APPLY,
            ControlOperation.LESION_CLEAR,
            ControlOperation.STIMULATION_APPLY,
            ControlOperation.STIMULATION_CLEAR,
        }:
            raise InterventionControlError(
                FaultCode.CONTROL_UNAVAILABLE,
                "control handler supports only declared neural interventions",
            )
        assert command.content_id is not None
        intervention = self._registry.get(command.content_id)
        expected_type = (
            NamedLesion
            if operation in {ControlOperation.LESION_APPLY, ControlOperation.LESION_CLEAR}
            else NamedStimulation
        )
        if intervention is None or not isinstance(intervention, expected_type):
            raise InterventionControlError(
                FaultCode.CONTROL_UNAVAILABLE, "unknown intervention content ID or kind"
            )
        if command.content_hash is not None and command.content_hash != intervention.content_hash:
            raise InterventionControlError(
                FaultCode.CONTROL_UNAVAILABLE, "intervention content hash mismatch"
            )
        if (
            operation
            in {
                ControlOperation.LESION_APPLY,
                ControlOperation.STIMULATION_APPLY,
            }
            and command.content_hash != intervention.content_hash
        ):
            raise InterventionControlError(
                FaultCode.CONTROL_UNAVAILABLE, "intervention apply requires the registry hash"
            )

        if isinstance(intervention, NamedLesion):
            self._handle_lesion(intervention, apply=operation is ControlOperation.LESION_APPLY)
        else:
            self._handle_stimulation(
                intervention,
                apply=operation is ControlOperation.STIMULATION_APPLY,
            )

    def _handle_lesion(self, intervention: NamedLesion, *, apply: bool) -> None:
        if self.controller.configuration.mode not in {"lesion", "synthetic"}:
            raise InterventionControlError(
                FaultCode.CONTROL_UNAVAILABLE,
                "lesions are unavailable outside an explicit lesion/synthetic run",
            )
        active = intervention.content_id in self.controller.active_named_lesions
        if active == apply:
            raise InterventionControlError(
                FaultCode.INVALID_STATE, "lesion is already in the requested lifecycle state"
            )
        self.controller.set_named_lesion(
            intervention.content_id,
            intervention.body_ids,
            ablated=apply,
        )

    def _handle_stimulation(self, intervention: NamedStimulation, *, apply: bool) -> None:
        active_id = self.controller.active_stimulation_id
        if apply:
            if active_id is not None:
                raise InterventionControlError(
                    FaultCode.INVALID_STATE, "a named stimulation is already active"
                )
            self.controller.set_stimulation(
                intervention.content_id,
                intervention.body_values,
            )
            return
        if active_id != intervention.content_id:
            raise InterventionControlError(
                FaultCode.INVALID_STATE, "named stimulation is not active"
            )
        self.controller.clear_stimulation()
