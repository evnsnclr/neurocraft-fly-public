#!/usr/bin/env python3
"""Latency-only NDJSON sink for the service's measured action telemetry.

Why this exists separately from ``PresentationTelemetryWriter``: the full
MaleCNS profiles refuse ``--presentation-telemetry-out`` because that writer
copies the controller's per-step evidence, and at full scale the evidence
exposes a declared boundary subset that the writer would emit as unlabeled
per-node activity. This writer reads none of that. It touches only the fields
``ActionTelemetry`` already carries -- one sequence number and three measured
durations -- so it crosses no evidence boundary and is permitted for every
controller profile.

It is diagnostic, not evidence. Causal evidence stays in the hash-linked trace
recorder; nothing here binds an action hash, a neural state, or a world
transition, and the samples are deliberately not content-addressed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from neurocraft_service import ActionTelemetry

#: Schema tag on every emitted line. ``v1`` is exactly the six keys written by
#: ``ActionLatencyWriter.accept``; widening the set rotates this tag.
ACTION_LATENCY_SCHEMA = "neurocraft.action_latency.v1"


class ActionLatencyWriter:
    """Append one latency-only NDJSON sample per serialized Action.

    Each sample carries the run identity, the controller profile, the command
    sequence, and the three service-side durations the sink is handed:
    ``compute_ns`` (the controller step inside the execution fence),
    ``serialization_ns``, and ``write_ns``. Nothing else from
    ``ActionTelemetry`` is written -- not the action hash, not the queue
    counters -- so the emitted set stays exactly the measured durations plus
    the sequence they belong to.

    Best-effort by contract: the service treats sink exceptions as terminal
    trace faults, so a write failure here must disable this writer rather than
    abort the live action path.
    """

    def __init__(self, path: Path, *, run_id: str, controller_profile: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")  # truncate: samples belong to this run only
        self._path = path
        self._run_id = run_id
        self._controller_profile = controller_profile
        self._disabled = False

    @property
    def disabled(self) -> bool:
        """True once a write failure has permanently silenced this writer."""

        return self._disabled

    def accept(self, telemetry: ActionTelemetry) -> None:
        if self._disabled:
            return
        try:
            sample = {
                "schema": ACTION_LATENCY_SCHEMA,
                "run_id": self._run_id,
                "controller_profile": self._controller_profile,
                "command_seq": telemetry.command_seq,
                "compute_ns": telemetry.compute_ns,
                "serialization_ns": telemetry.serialization_ns,
                "write_ns": telemetry.write_ns,
            }
            with self._path.open("ab") as stream:
                stream.write(json.dumps(sample, sort_keys=True).encode() + b"\n")
        except Exception as exc:  # deliberate: diagnostics must never fault the loop
            self._disabled = True
            print(
                f"action latency telemetry disabled after write failure: {exc}",
                file=sys.stderr,
            )
