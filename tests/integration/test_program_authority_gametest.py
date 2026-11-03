"""Pure-function pins for the opt-in ADR 0020 operator-lane probe launcher.

The launcher's live half (a real service, Gradle, the GameTest server) is exercised by
running it; these tests hold the parts that decide what counts as a passing run, above all
the receipt verifier's relation between the acknowledgement the service stamps and the tick
the mod engaged on.
"""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from neurocraft_protocol import ACTION_TARGET_OFFSET_TICKS

from scripts import run_program_authority_gametest as probe
from scripts.run_interactive_demo import LauncherError


def _plan(tmp_path: Path) -> probe.ProbePlan:
    return probe.prepare_plan(
        run_root=tmp_path / "runs", now=datetime(2026, 9, 3, 19, 0, tzinfo=UTC)
    )


def _identity(plan: probe.ProbePlan) -> dict[str, object]:
    return {
        "run_id": plan.run_id,
        "session_epoch": plan.session_epoch,
        "entity_id": plan.entity_id,
        "graph_hash": "1" * 64,
        "adapter_hash": "2" * 64,
        "dynamics_hash": "3" * 64,
        "decoder_hash": "4" * 64,
        "observation_schema_hash": "5" * 64,
    }


def _receipt() -> dict[str, Any]:
    """A receipt shaped like the probe's.

    The relation is ``engaged == effective + ACTION_TARGET_OFFSET_TICKS``: the acknowledgement is
    stamped with an observation tick, and the handover engages on the tick the paused service's
    first neutral command governs, which is one action budget later.
    """

    return {
        "schema": probe.SCHEMA,
        "protocol": "neurocraft.protocol.v2",
        "entity_id": "95402e8e-5696-443f-abf8-75983bbb0a61",
        "bridge_state": "ACTIVE",
        "pacing_mode": probe.EXPECTED_PACING_MODE,
        "pacing_period_ns": probe.EXPECTED_PACING_PERIOD_NS,
        "driving_command_seq": 4,
        "neural_cursor_before_pause": 25,
        "pause": {
            "issued_tick": 12,
            "engaged_tick": 15,
            "program_owned_ticks_at_engage": 1,
            "status": "applied effective_tick=13 engaged_tick=15",
            "message": "Applied accepted neural PAUSE (effective from observation tick 13) at"
            " tick 15; motor authority is now SCRIPTED",
        },
        "scripted_window": {
            "program": "HOVER",
            "source": "OPERATOR",
            "program_owned_ticks": 21,
            "paused_commands_observed": 21,
            "last_paused_command_seq": 25,
            "frozen_neural_cursor": 30,
            "max_abs_vertical_thrust": 0.012,
        },
        "release": {
            "issued_tick": 34,
            "released_tick": 37,
            "status": "applied effective_tick=35 engaged_tick=37",
            "message": "Applied accepted neural RESUME (effective from observation tick 35) at"
            " tick 37; motor authority is now NEURAL",
            "ledger_line": "BODY PROGRAM HOVER -> HOVER tick=37 source=OPERATOR reason=released",
        },
        "resumed": {
            "command_seq": 26,
            "neural_step_start": 30,
            "neural_step_end": 35,
            "commands_skipped_since_pause": 0,
        },
        "reset": {
            "issued_tick": 38,
            "applied_tick": 40,
            "status": "applied effective_tick=40",
            "message": "Applied accepted neural RESET at effective tick 40",
            "target_blocks": [2.5, 4.0, 2.5],
            "position_blocks": [2.5, 4.0, 2.5],
        },
        "claim_boundary": "scripted presentation over a paused service; transport and"
        " transaction evidence only, not connectome behaviour",
        "completed_game_tick": 41,
    }


def test_commands_bind_the_fresh_identity_and_keep_the_token_out_of_argv(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    identity = _identity(plan)
    service = probe.service_command(plan)
    gradle = probe.gradle_command(plan, identity)
    identity_only = probe.identity_command(plan)
    everything = "\n".join((*service, *gradle, *identity_only))

    assert probe.TOKEN_NAME not in everything
    assert plan.run_dir.is_dir() and plan.run_dir.parent == tmp_path / "runs"
    # The probe drives a paused service: no trace recorder, no interventions, fixture profile.
    assert "--trace-out" not in service and "--declare-stimulation" not in service
    assert str(probe.FIXTURE) in service and "--mode" in service
    assert service[service.index("--port") + 1] == str(plan.bridge_port)
    assert "--identity-only" in identity_only and "--identity-only" not in service
    # The external lane is opted in and the program-authority scenario selected explicitly.
    assert "-Dneurocraft.fly.externalBridgeTest=true" in gradle
    assert f"-Dneurocraft.fly.externalBridgeScenario={probe.SCENARIO}" in gradle
    assert f"-Dneurocraft.fly.bridgeUri=ws://127.0.0.1:{plan.bridge_port}/control" in gradle
    assert f"-Dneurocraft.fly.entityId={plan.entity_id}" in gradle
    assert f"-Dneurocraft.fly.graphHash={'1' * 64}" in gradle
    assert gradle[-1] == "runExternalBridgeGameTest"


def test_load_identity_rejects_mismatch_and_bad_hash(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    identity = _identity(plan)
    plan.identity_path.write_text(json.dumps(identity), encoding="utf-8")
    assert probe.load_identity(plan.identity_path, plan) == identity

    identity["entity_id"] = "wrong"
    plan.identity_path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(LauncherError, match="does not match"):
        probe.load_identity(plan.identity_path, plan)

    identity["entity_id"] = plan.entity_id
    identity["graph_hash"] = "not-a-hash"
    plan.identity_path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(LauncherError, match="not a SHA-256"):
        probe.load_identity(plan.identity_path, plan)


def test_verify_receipt_accepts_the_engaged_relation_and_summarizes_it() -> None:
    summary = probe.verify_receipt(_receipt())

    assert summary.pause_issued_tick == 12
    assert summary.pause_effective_tick == 13
    assert summary.pause_engaged_tick == 15
    assert summary.pause_engaged_tick == summary.pause_effective_tick + ACTION_TARGET_OFFSET_TICKS
    assert (
        summary.release_engaged_tick == summary.release_effective_tick + ACTION_TARGET_OFFSET_TICKS
    )
    assert summary.reset_applied_tick == summary.reset_effective_tick == 40
    assert summary.program_owned_ticks == 21
    assert summary.neural_cursor_before_pause == 25
    assert summary.resumed_neural_step_start == summary.frozen_neural_cursor == 30
    assert any("engaged_tick=15 (effective_tick + 2)" in line for line in summary.lines())


def test_verify_receipt_refuses_the_pre_fix_relation() -> None:
    """The defect this lane fixed: an engage tick equal to the effective tick is impossible live."""

    document = _receipt()
    document["pause"]["status"] = "applied effective_tick=13 engaged_tick=13"
    document["pause"]["engaged_tick"] = 13
    with pytest.raises(LauncherError, match="engaged_tick 13 is not effective_tick plus"):
        probe.verify_receipt(document)

    # One tick past the effective tick is the pre-two-tick-budget relation: it is now too early,
    # because the paused service's first neutral command has not applied yet.
    early_engage = _receipt()
    early_engage["pause"]["status"] = "applied effective_tick=13 engaged_tick=14"
    early_engage["pause"]["engaged_tick"] = 14
    with pytest.raises(LauncherError, match="engaged_tick 14 is not effective_tick plus"):
        probe.verify_receipt(early_engage)

    disagreeing = _receipt()
    disagreeing["pause"]["engaged_tick"] = 16
    with pytest.raises(LauncherError, match="saw the authority change at tick 16"):
        probe.verify_receipt(disagreeing)

    stale = _receipt()
    stale["pause"]["status"] = "applied effective_tick=15 engaged_tick=17"
    stale["pause"]["engaged_tick"] = 17
    with pytest.raises(LauncherError, match="not issued_tick or the next tick"):
        probe.verify_receipt(stale)

    unresolved = _receipt()
    unresolved["release"]["status"] = "missed fault=LATE_ACK"
    with pytest.raises(LauncherError, match="not an applied program-authority status"):
        probe.verify_receipt(unresolved)

    idle_engage_tick = _receipt()
    idle_engage_tick["pause"]["program_owned_ticks_at_engage"] = 0
    with pytest.raises(LauncherError, match="did not own the tick scripted authority engaged on"):
        probe.verify_receipt(idle_engage_tick)


def test_verify_receipt_refuses_a_release_by_preemption_or_a_short_window() -> None:
    preempted = _receipt()
    preempted["release"]["ledger_line"] = (
        "BODY PROGRAM HOVER -> HOVER tick=37 source=OPERATOR reason=preempted"
    )
    with pytest.raises(LauncherError, match="not released by the accepted RESUME"):
        probe.verify_receipt(preempted)

    short = _receipt()
    short["scripted_window"]["program_owned_ticks"] = probe.MINIMUM_PROGRAM_OWNED_TICKS - 1
    with pytest.raises(LauncherError, match="owned only"):
        probe.verify_receipt(short)

    # Engaged at 15 with a 20-tick minimum, the earliest honest release is issued at tick 34.
    earliest = _receipt()
    earliest["release"]["issued_tick"] = 34
    earliest["release"]["status"] = "applied effective_tick=35 engaged_tick=37"
    earliest["release"]["released_tick"] = 37
    assert probe.verify_receipt(earliest).release_issued_tick == 34

    early = _receipt()
    early["release"]["issued_tick"] = 33
    early["release"]["status"] = "applied effective_tick=34 engaged_tick=36"
    early["release"]["released_tick"] = 36
    with pytest.raises(LauncherError, match="before the scripted window was long enough"):
        probe.verify_receipt(early)

    still = _receipt()
    still["scripted_window"]["max_abs_vertical_thrust"] = 0.0
    with pytest.raises(LauncherError, match="no vertical thrust"):
        probe.verify_receipt(still)

    unpaused = _receipt()
    unpaused["scripted_window"]["paused_commands_observed"] = 0
    with pytest.raises(LauncherError, match="no paused-neutral command"):
        probe.verify_receipt(unpaused)


def test_verify_receipt_requires_the_connectome_to_resume_from_the_frozen_cursor() -> None:
    document = _receipt()
    document["resumed"]["neural_step_start"] = 25
    document["resumed"]["neural_step_end"] = 30
    with pytest.raises(LauncherError, match="frozen cursor"):
        probe.verify_receipt(document)

    rewound = _receipt()
    rewound["scripted_window"]["frozen_neural_cursor"] = 20
    rewound["resumed"]["neural_step_start"] = 20
    with pytest.raises(LauncherError, match="precedes the last stepped command"):
        probe.verify_receipt(rewound)

    # A retired command between the last paused one and the resumed one is accounted for
    # explicitly, and then the cursor may only have moved forward.
    gapped = _receipt()
    gapped["resumed"]["command_seq"] = 27
    gapped["resumed"]["commands_skipped_since_pause"] = 1
    gapped["resumed"]["neural_step_start"] = 35
    gapped["resumed"]["neural_step_end"] = 40
    assert probe.verify_receipt(gapped).resumed_neural_step_start == 35

    miscounted = _receipt()
    miscounted["resumed"]["commands_skipped_since_pause"] = 1
    with pytest.raises(LauncherError, match="disagrees with the sequences"):
        probe.verify_receipt(miscounted)

    unstepped = _receipt()
    unstepped["resumed"]["neural_step_end"] = 30
    with pytest.raises(LauncherError, match="did not step the connectome"):
        probe.verify_receipt(unstepped)


def test_verify_receipt_holds_the_reset_to_its_effective_tick_and_target() -> None:
    late = _receipt()
    late["reset"]["applied_tick"] = 41
    with pytest.raises(LauncherError, match="not applied on its effective tick"):
        probe.verify_receipt(late)

    # Issued at 38, the accepted window is 40 or 41 under the two-tick action budget.
    latest = _receipt()
    latest["reset"]["status"] = "applied effective_tick=41"
    latest["reset"]["applied_tick"] = 41
    assert probe.verify_receipt(latest).reset_effective_tick == 41

    outside = _receipt()
    outside["reset"]["status"] = "applied effective_tick=42"
    outside["reset"]["applied_tick"] = 42
    with pytest.raises(LauncherError, match="outside its action-budget window"):
        probe.verify_receipt(outside)

    elsewhere = _receipt()
    elsewhere["reset"]["position_blocks"] = [2.5, 4.5, 2.5]
    with pytest.raises(LauncherError, match="not at the RESET target"):
        probe.verify_receipt(elsewhere)


def test_verify_receipt_refuses_unknown_keys_wrong_schema_and_a_faulted_bridge() -> None:
    extra = _receipt()
    extra["note"] = "unexpected"
    with pytest.raises(LauncherError, match="receipt keys differ"):
        probe.verify_receipt(extra)

    wrong_schema = _receipt()
    wrong_schema["schema"] = "neurocraft.external_bridge_gametest.v1"
    with pytest.raises(LauncherError, match="receipt schema"):
        probe.verify_receipt(wrong_schema)

    faulted = _receipt()
    faulted["bridge_state"] = "DEADLINE_MISSED"
    with pytest.raises(LauncherError, match="did not stay ACTIVE"):
        probe.verify_receipt(faulted)

    unpaced = copy.deepcopy(_receipt())
    unpaced["pacing_mode"] = "unpaced"
    with pytest.raises(LauncherError, match="20 Hz pacer"):
        probe.verify_receipt(unpaced)
