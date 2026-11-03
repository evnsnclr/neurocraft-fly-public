#!/usr/bin/env python3
"""Diagnostic NDJSON sink for issued ``program_action`` frames and their receipts.

**Diagnostic, not evidence. Not content-addressed.** Nothing written here binds a neural
state to a world transition, and no line is hash-linked to any other. Causal evidence lives
in the hash-linked trace recorder, and today no trace of the full-brain lane can be
committed at all: the receipt trace pipeline is fixture-typed, and the ``program_record``
trace entry for body-program frames is disclosed open work. Until that exists, a session's
account of what was selected is this file plus the mod's log, status, and HUD -- and a
sentence written from it must say so.

What a line here does support: "at command_seq N the declared policy asked for program P on
readout scalars (t, d, w), and the authority answered V". What it does not support: that
the connectome selected P, that the fly reacted to anything, or that the mapping is
anything but hand-chosen.

Modelled on ``scripts/action_latency_telemetry.py``: best-effort by contract, because the
service treats sink exceptions as terminal trace faults, so a write failure here must
disable this writer rather than abort the live command path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from neurocraft_protocol import (
    ActuationReceipt,
    Command,
    ProgramAction,
    ProgramReceipt,
)

if TYPE_CHECKING:
    from neurocraft_controller.program_policy import ProgramSelectingController

#: Schema tag on every emitted line. ``v1`` is exactly the key sets written by
#: :meth:`ProgramDecisionWriter.accept_issued_command` and
#: :meth:`ProgramDecisionWriter.accept_receipt`; widening either rotates this tag.
PROGRAM_DECISION_SCHEMA = "neurocraft.program_decision.v1"


class ProgramDecisionWriter:
    """Append one diagnostic NDJSON line per issued program frame and per receipt.

    Two record kinds, both keyed by ``command_seq`` so a reader joins them:

    * ``issued`` -- the frame's own fields (program, modulation, parameters, the three
      readout scalars, ``policy_hash``) plus, when the controller's evidence still belongs
      to this command, the policy state before and after. The service hands the sink the
      frame after the step; if a later step has already replaced the evidence, the state
      fields are written as ``null`` rather than guessed.
    * ``receipt`` -- what the authority reported: the validation, the executed program, and
      the refusal reason if any, recorded exactly as received. The ``operator`` refusal
      reason legitimately reports a program other than HOVER (an operator hold ran its own
      program), so nothing here normalizes it.

    The policy is open loop with respect to receipts: this writer observes them, and no
    value it records ever reaches :meth:`ProgramSelectionPolicy.select`.
    """

    def __init__(
        self,
        path: Path,
        controller: ProgramSelectingController,
        *,
        controller_profile: str,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")  # truncate: samples belong to this run only
        self._path = path
        self._controller = controller
        self._controller_profile = controller_profile
        self._run_id = controller.identity.run_id
        self._policy_hash = controller.policy_hash
        self._disabled = False
        self._inputs_path = (
            path.with_suffix(".inputs.ndjson")
            if controller_profile == "full_mcns_interactions_program_v2"
            else None
        )
        self._inputs_disabled = False
        if self._inputs_path is not None:
            try:
                self._inputs_path.write_bytes(b"")
            except OSError as exc:
                self._inputs_disabled = True
                print(f"program input diagnostics disabled: {exc}", file=sys.stderr)

    @property
    def disabled(self) -> bool:
        """True once a write failure has permanently silenced this writer."""

        return self._disabled

    @property
    def policy_hash(self) -> str:
        return self._policy_hash

    # ------------------------------------------------------------------- sinks

    def accept_issued_command(self, command: Command) -> None:
        """Record one issued ``program_action``; ignore a raw ``action``.

        A profile issues one command kind for the whole session, so a raw action reaching
        this sink means a paused, replaying, or faulted service emitted its neutral frame.
        Those carry no selection, so there is nothing to record.
        """

        if not isinstance(command, ProgramAction):
            return
        evidence = self._controller.last_evidence
        bound = evidence is not None and evidence.command_seq == command.command_seq
        decision = evidence.decision if (bound and evidence is not None) else None
        before = None if decision is None else decision.state_before.to_dict()
        after = None if decision is None else decision.state_after.to_dict()
        self._write(
            {
                "kind": "issued",
                "command_seq": command.command_seq,
                "observation_seq": command.observation_seq,
                "observation_tick": command.observation_tick,
                "target_tick": command.target_tick,
                "state_digest": command.state_digest,
                "neural_step_start": command.neural_step_start,
                "neural_step_end": command.neural_step_end,
                "program": command.program.value,
                "modulation": command.modulation.to_dict(),
                "parameters": command.parameters.to_dict(),
                "readout": command.readout.to_dict(),
                "policy_hash": command.policy_hash,
                # Null rather than guessed: a later step may already have replaced the
                # controller's evidence by the time this sink runs.
                "policy_state_bound": bound,
                "policy_state_before": before,
                "policy_state_after": after,
                "driving_readout": None if decision is None else decision.driving_readout,
            }
        )
        if self._inputs_path is not None and not self._inputs_disabled:
            try:
                inner = evidence.inner if bound and evidence is not None else None
                if inner is not None and (
                    inner.observation.observation_seq != command.observation_seq
                    or inner.observation.identity != command.identity
                    or inner.observation.game_tick != command.observation_tick
                    or inner.state_after_sha256 != command.state_digest
                    or inner.neural_step_start != command.neural_step_start
                    or inner.neural_step_end != command.neural_step_end
                ):
                    inner = None
                sample = {
                    "schema": "neurocraft.program_input_diagnostic.v1",
                    "diagnostic_only": True,
                    "identity": command.identity.to_dict(),
                    "command_seq": command.command_seq,
                    "observation_seq": command.observation_seq,
                    "observation_tick": command.observation_tick,
                    "state_digest": command.state_digest,
                    "bound": inner is not None,
                    "encoded_stimulation": None
                    if inner is None
                    else dict(inner.encoded_stimulation),
                    "before_control": None
                    if inner is None or inner.encoder_stimulation_before_control is None
                    else dict(inner.encoder_stimulation_before_control),
                    "stimulation_sha256": None
                    if inner is None
                    else inner.encoded_stimulation_sha256,
                    "scope": (
                        "declared assignments/population drive summaries, "
                        "not a dense sensory vector or hash-linked trace"
                    ),
                }
                raw = json.dumps(sample, sort_keys=True, allow_nan=False).encode() + b"\n"
                if len(raw) > 32768:
                    raise ValueError("input diagnostic exceeds row budget")
                with self._inputs_path.open("ab") as stream:
                    stream.write(raw)
            except Exception as exc:
                self._inputs_disabled = True
                print(f"program input diagnostics disabled: {exc}", file=sys.stderr)

    def accept_receipt(self, receipt: ActuationReceipt | ProgramReceipt) -> None:
        """Record one authoritative ``program_receipt``; ignore an actuation receipt."""

        if not isinstance(receipt, ProgramReceipt):
            return
        executed = receipt.executed_program
        self._write(
            {
                "kind": "receipt",
                "command_seq": receipt.command_seq,
                "observation_seq": receipt.observation_seq,
                "target_tick": receipt.target_tick,
                "applied_tick": receipt.applied_tick,
                "requested_program": receipt.requested_program.value,
                "executed_program": None if executed is None else executed.value,
                "validation": receipt.validation.value,
                "reason": None if receipt.reason is None else receipt.reason.value,
                "fault_code": None if receipt.fault_code is None else receipt.fault_code.value,
            }
        )

    # ------------------------------------------------------------------- write

    def _write(self, sample: dict[str, Any]) -> None:
        if self._disabled:
            return
        try:
            line = {
                "schema": PROGRAM_DECISION_SCHEMA,
                "diagnostic_only": True,
                "run_id": self._run_id,
                "controller_profile": self._controller_profile,
                **sample,
            }
            with self._path.open("ab") as stream:
                stream.write(json.dumps(line, sort_keys=True).encode() + b"\n")
        except Exception as exc:  # deliberate: diagnostics must never fault the loop
            self._disabled = True
            print(
                f"program decision telemetry disabled after write failure: {exc}",
                file=sys.stderr,
            )
