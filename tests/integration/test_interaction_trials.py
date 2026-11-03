from __future__ import annotations

import copy
import hashlib
import io
import json
import threading
import time
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from scripts import run_interaction_trials as trials

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _isolated_trial_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    # Other integration tests can leave a daemon server winding down. Patching
    # process-global sleep lets that unrelated thread advance a fake trial clock.
    # Keep real time for other modules while this test controls only the runner.
    monkeypatch.setattr(
        trials,
        "time",
        SimpleNamespace(
            monotonic=time.monotonic,
            sleep=time.sleep,
            time=time.time,
            time_ns=time.time_ns,
        ),
    )


def _clock() -> SimpleNamespace:
    return cast(SimpleNamespace, vars(trials)["time"])


def test_fake_trial_clock_is_not_advanced_by_an_unrelated_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 0.0
    actual_sleep = time.sleep

    def advance(seconds: float) -> None:
        nonlocal now
        now += seconds

    monkeypatch.setattr(_clock(), "monotonic", lambda: now)
    monkeypatch.setattr(_clock(), "sleep", advance)
    background = threading.Thread(target=lambda: time.sleep(0.01), daemon=True)
    background.start()
    background.join(timeout=1.0)
    assert not background.is_alive()
    assert time.sleep is actual_sleep
    assert _clock().monotonic() == 0.0
    _clock().sleep(0.05)
    assert _clock().monotonic() == 0.05


def identity() -> dict[str, Any]:
    return cast(
        "dict[str, Any]",
        json.loads((ROOT / "packages/protocol/golden/interaction_display_v2.json").read_text())[
            "identity"
        ],
    )


def sensors() -> dict[str, Any]:
    result = json.loads((ROOT / "packages/protocol/golden/observation_v4.json").read_text())[
        "sensors"
    ]
    result.update(
        ray_actor_kinds=[0] * 6, ray_hit_kinds=[0] * 6, ray_distances_m=[6.0] * 6, local_light=15
    )
    result["events"] = {
        "food_contact": False,
        "food_near": False,
        **{
            key: {"count": 0, "last_game_tick": None} for key in ("food", "brush", "attack_attempt")
        },
    }
    return cast("dict[str, Any]", result)


def issued(seq: int, source: str | None = None, *, program: str | None = None) -> dict[str, Any]:
    tick = 100 + seq
    before = {
        "last_game_tick": tick - 1,
        "source": None,
        "held_event": None,
        "active_until": 0,
        "previous_sensors": sensors(),
    }
    after = {
        **copy.deepcopy(before),
        "last_game_tick": tick,
        "program": program or "CRUISE",
        "hold_remaining": 0,
        "cooldown_until": 0,
        "history": [[0.0] * 4] * 4,
        "pending": dict.fromkeys(trials.SOURCES),
        "approaching": [False, False],
    }
    if source:
        after.update(
            source=source,
            held_event={"event_tick": 100, "baseline": 0.0},
            active_until=102,
            hold_remaining=102 - tick,
        )
        after["program"] = program or trials.PROGRAMS[source][0]
    return {
        "schema": "neurocraft.program_decision.v1",
        "diagnostic_only": True,
        "kind": "issued",
        "run_id": identity()["run_id"],
        "controller_profile": trials.PROFILE,
        "command_seq": seq,
        "observation_seq": seq,
        "observation_tick": tick,
        "target_tick": tick + 2,
        "neural_step_start": seq * 5,
        "neural_step_end": (seq + 1) * 5,
        "program": after["program"],
        "policy_state_bound": True,
        "policy_state_before": before,
        "policy_state_after": after,
    }


def receipt(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "neurocraft.program_decision.v1",
        "diagnostic_only": True,
        "kind": "receipt",
        "run_id": row["run_id"],
        "controller_profile": trials.PROFILE,
        "command_seq": row["command_seq"],
        "observation_seq": row["observation_seq"],
        "target_tick": row["target_tick"],
        "applied_tick": row["target_tick"],
        "requested_program": row["program"],
        "executed_program": row["program"],
        "validation": "PROGRAM_APPLIED",
        "fault_code": None,
        "reason": None,
    }


def bound(*rows: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"epoch_identity": identity(), "row": row} for row in rows]


def changed(row: dict[str, Any], source: str) -> None:
    before = row["policy_state_before"]["previous_sensors"]
    after = row["policy_state_after"]["previous_sensors"]
    if source in ("attack", "brush", "food"):
        key = "attack_attempt" if source == "attack" else source
        after["events"][key] = {"count": 1, "last_game_tick": row["observation_tick"]}
        if source == "food":
            after["events"]["food_near"] = True
    elif source in ("player_approach", "mob_approach"):
        actor = 1 if source == "player_approach" else 2
        before["ray_actor_kinds"][0] = after["ray_actor_kinds"][0] = actor
        before["ray_hit_kinds"][0] = after["ray_hit_kinds"][0] = 2
        before["ray_distances_m"][0] = 4.0
        after["ray_distances_m"][0] = 3.0
    else:
        after["local_light"] = 0


def reduce(source: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    return trials.reduce_case(
        trials.Case(0, "original", 0, source), identity(), records, start_tick=100, end_tick=110
    )


def test_fixed_schedule_and_single_case_modes_preserve_scope_and_request_budget() -> None:
    cases = trials.schedule()
    assert len(cases) == 30
    assert [case.index for case in cases] == list(range(30))
    for lane, count in (("original", 3), ("shuffled_weight", 1), ("no_input", 1)):
        for source in trials.SOURCES:
            assert sum(case.lane == lane and case.source == source for case in cases) == count
    assert sum(case.source in ("brush", "attack") for case in cases) == 10 < trials.NATIVE_REQUESTS
    assert trials.schedule("attack", "original") == (trials.Case(0, "original", 0, "attack"),)
    assert len(trials.schedule("light")) == 5
    with pytest.raises(trials.TrialError, match="unknown"):
        trials.schedule("injury")


@pytest.mark.parametrize("source", trials.SOURCES)
def test_source_proof_and_applied_response_are_independent(source: str) -> None:
    row = issued(0, source)
    changed(row, source)
    result = reduce(source, bound(row))
    assert result["source_verified"] and result["issued_response"]
    assert not result["applied_response"]
    assert result["issued_without_applied_receipt"] == [0]
    result = reduce(source, bound(row, receipt(row)))
    assert result["applied_response"]
    assert not result["complete_applied_hold"]  # A single successful onset is insufficient.


@pytest.mark.parametrize("lane", ("shuffled_weight", "no_input"))
def test_controls_with_delivered_source_and_no_response_are_valid_observations(lane: str) -> None:
    row = issued(0)
    changed(row, "light")
    result = trials.reduce_case(
        trials.Case(0, lane, 0, "light"),
        identity(),
        bound(row, receipt(row)),
        start_tick=100,
        end_tick=110,
    )
    assert result["source_verified"]
    assert result["outcome"] == "no_applied_response"
    assert not result["issued_response"] and not result["applied_response"]


def test_unobserved_wrong_actor_or_wrong_source_cannot_become_a_response() -> None:
    row = issued(0, "player_approach")
    changed(row, "mob_approach")
    result = reduce("player_approach", bound(row, receipt(row)))
    assert result["outcome"] == "source_miss" and not result["applied_response"]
    changed(row, "player_approach")
    row["policy_state_after"]["source"] = "mob_approach"
    result = reduce("player_approach", bound(row, receipt(row)))
    assert result["source_verified"] and not result["issued_response"]
    row = issued(0, "light")
    assert reduce("light", bound(row, receipt(row)))["outcome"] == "source_miss"


@pytest.mark.parametrize(
    "field",
    ("observation_seq", "target_tick", "applied_tick", "requested_program", "executed_program"),
)
def test_mismatched_receipt_envelopes_are_refused(field: str) -> None:
    row = issued(0, "light")
    changed(row, "light")
    actual = receipt(row)
    actual[field] = "CRUISE" if "program" in field else 999
    with pytest.raises(trials.TrialError, match="receipt"):
        reduce("light", bound(row, actual))


def test_identity_duplicate_and_terminal_fault_refusal() -> None:
    row = issued(0, "light")
    changed(row, "light")
    records = bound(row, receipt(row))
    records[1]["epoch_identity"] = {**identity(), "session_epoch": identity()["session_epoch"] + 1}
    with pytest.raises(trials.TrialError, match="identity"):
        reduce("light", records)
    with pytest.raises(trials.TrialError, match="duplicate"):
        reduce("light", bound(row, row))
    actual = receipt(row)
    actual["fault_code"] = "DISCONNECTED"
    with pytest.raises(trials.TrialError, match="terminal"):
        reduce("light", bound(row, actual))


def test_complete_hold_requires_every_applied_tick_and_natural_episode_end() -> None:
    first = issued(0, "light")
    changed(first, "light")
    second = issued(1, "light")
    second["policy_state_before"] = copy.deepcopy(first["policy_state_after"])
    second["policy_state_after"]["previous_sensors"] = copy.deepcopy(
        first["policy_state_after"]["previous_sensors"]
    )
    last = issued(2)
    last["policy_state_before"] = copy.deepcopy(second["policy_state_after"])
    last["policy_state_after"]["active_until"] = 102
    records = bound(first, receipt(first), second, receipt(second), last, receipt(last))
    assert reduce("light", records)["complete_applied_hold"]
    assert not reduce("light", bound(first, receipt(first), second, last, receipt(last)))[
        "complete_applied_hold"
    ]
    missing_both = reduce("light", bound(first, receipt(first), last, receipt(last)))
    assert missing_both["episode_expired"] and missing_both["applied_response"]
    assert not missing_both["complete_applied_hold"]
    last["policy_state_before"]["active_until"] = 110
    assert not reduce("light", bound(first, receipt(first), second, receipt(second), last))[
        "complete_applied_hold"
    ]


def episode_rows(source: str) -> list[dict[str, Any]]:
    """A separately observed pulse followed by entry, held ticks, and natural expiry."""
    onset = issued(0)
    changed(onset, source)
    rows = [onset]
    expiry = 63 if source == "food" else 41
    for seq in range(1, expiry + 1):
        row = issued(seq)
        before = copy.deepcopy(rows[-1]["policy_state_after"])
        after = copy.deepcopy(before)
        after["last_game_tick"] = row["observation_tick"]
        if seq == expiry:
            after.update(
                source=None,
                held_event=None,
                program="CRUISE",
                hold_remaining=0,
                cooldown_until=100 + expiry + 120,
            )
        else:
            program = (
                "LAND" if source == "food" and seq < 3 else "FEED" if source == "food" else "HOVER"
            )
            until = 301 if program == "LAND" else 100 + expiry
            after.update(
                source=source,
                held_event={"event_tick": 100, "baseline": 0.0},
                program=program,
                active_until=until,
                hold_remaining=until - (100 + seq),
            )
            if program == "FEED":
                after["previous_sensors"]["events"]["food_contact"] = True
        row.update(policy_state_before=before, policy_state_after=after, program=after["program"])
        rows.append(row)
    return rows


def with_receipts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return bound(*(item for row in rows for item in (row, receipt(row))))


@pytest.mark.parametrize("missing", (1, 2, 20, 40, 41))
def test_missing_entry_interior_final_hold_or_expiry_is_incomplete(missing: int) -> None:
    rows = episode_rows("light")
    assert reduce("light", with_receipts(rows))["complete_applied_hold"]
    result = reduce("light", with_receipts([row for row in rows if row["command_seq"] != missing]))
    # Source proof deliberately survives even when the actual entry is absent.
    assert result["source_verified"] and result["applied_response"]
    assert result["episode_expired"] is (missing != 41)
    assert not result["complete_applied_hold"]


@pytest.mark.parametrize("missing", (1, 2, 3, 30, 62, 63))
def test_land_to_feed_requires_both_complete_intervals_and_the_new_expiry(missing: int) -> None:
    rows = episode_rows("food")
    full = reduce("food", with_receipts(rows))
    assert full["complete_applied_hold"]
    assert full["applied_program_counts"] == {"LAND": 2, "FEED": 60}
    result = reduce("food", with_receipts([row for row in rows if row["command_seq"] != missing]))
    assert result["source_verified"] and result["applied_program_counts"]["FEED"] > 0
    assert not result["complete_applied_hold"]


@pytest.mark.parametrize("damage", ("state_link", "deadline", "coalesced_expiry", "unbound_expiry"))
def test_declared_expiry_cannot_replace_missing_or_contradictory_interval_evidence(
    damage: str,
) -> None:
    rows = episode_rows("light")
    if damage == "state_link":
        rows[2]["policy_state_before"]["held_event"]["baseline"] = -0.0
    elif damage == "deadline":
        # Even a consistently copied hidden deadline change cannot shorten a hold.
        rows[1]["policy_state_after"]["active_until"] += 1
        rows[1]["policy_state_after"]["hold_remaining"] += 1
        rows[2]["policy_state_before"] = copy.deepcopy(rows[1]["policy_state_after"])
    elif damage == "coalesced_expiry":
        rows[-1]["observation_tick"] += 1
        rows[-1]["target_tick"] += 1
        rows[-1]["policy_state_after"]["last_game_tick"] += 1
    else:
        rows[-1]["policy_state_bound"] = False
    assert not reduce("light", with_receipts(rows))["complete_applied_hold"]


def test_fixture_commands_native_target_and_inbox_bounds(tmp_path: Path) -> None:
    for case in trials.schedule():
        commands = trials.preparation_commands(case)
        assert all(0 < len(command) <= 512 and not command.startswith("/") for command in commands)
        assert sum(command.startswith(f"tp {trials.FLY}") for command in commands) == 1
        if case.source == "attack":
            assert not any(command.startswith(f"tp {trials.PLAYER}") for command in commands)
            assert commands[-1].endswith("minecraft:air")
    assert "light[level=15]" in trials.lighting(True)
    assert "replace minecraft:light" in trials.lighting(False)
    path = tmp_path / "inbox"
    path.touch()
    inbox = trials.AppendInbox(path, 80)
    try:
        inbox.append({"command": "one"})
        with pytest.raises(BlockingIOError):
            trials.AppendInbox(path, 80)
        with path.open("ab") as stream:
            stream.write(b"{}\n")
        with pytest.raises(trials.TrialError, match="another"):
            inbox.append({"command": "two"})
    finally:
        inbox.close()
    bounded = trials.AppendInbox(path, path.stat().st_size)
    try:
        with pytest.raises(trials.TrialError, match="budget"):
            bounded.append({"command": "two"})
    finally:
        bounded.close()


def test_readiness_uses_only_v2_state_and_waits_through_cooldown() -> None:
    row = issued(0)
    assert trials.eligible(row)
    row["policy_state_after"]["cooldown_until"] = 101
    assert not trials.eligible(row)
    assert not trials.eligible(issued(0, "attack"))
    session = {
        "state": "interactive",
        "controller_profile": trials.PROFILE,
        "requested_game_ticks_per_wall_second": 20,
        "playback_speed": "normal",
        "diagnostic_tools": True,
        "comparison_lanes": True,
        "control_seed": "7",
    }
    trials.require_session(session)
    for key, value in (
        ("state", "stopped"),
        ("requested_game_ticks_per_wall_second", 10),
        ("control_seed", "8"),
        ("controller_profile", "full_mcns_events_program_v1"),
    ):
        with pytest.raises(trials.TrialError):
            trials.require_session({**session, key: value})


@pytest.mark.parametrize(
    "field,value",
    (
        ("pending", {}),
        ("pending", {**dict.fromkeys(trials.SOURCES), "light": {"event_tick": 99}}),
        ("approaching", [True, False]),
        ("approaching", [False, True]),
        ("approaching", [0, 0]),
        ("approaching", None),
    ),
)
def test_pending_events_and_approach_edges_are_not_ready(field: str, value: Any) -> None:
    row = issued(0)
    row["policy_state_after"][field] = value
    assert not trials.eligible(row)


def baseline_epoch(count: int = 10) -> trials.Epoch:
    rows = {seq: issued(seq) for seq in range(count)}
    return cast(
        trials.Epoch,
        SimpleNamespace(
            issued=rows,
            receipts={seq: receipt(row) for seq, row in rows.items()},
            poll=lambda: None,
        ),
    )


def test_preparation_requires_ten_distinct_new_rows_not_repeated_polls() -> None:
    epoch = baseline_epoch(9)
    for _ in range(20):
        assert trials.stable_preparation(epoch, -1, 15) is None
    epoch.issued[9] = issued(9)
    epoch.receipts[9] = receipt(epoch.issued[9])
    assert trials.stable_preparation(epoch, -1, 15) is epoch.issued[9]
    assert trials.stable_preparation(epoch, 0, 15) is None
    assert trials.stable_preparation(epoch, -1, 0) is None


@pytest.mark.parametrize(
    "change",
    (
        "pending",
        "approach",
        "light",
        "counter",
        "missing_receipt",
        "wrong_receipt",
        "tick_gap",
    ),
)
def test_preparation_streak_resets_on_changes_or_missing_evidence(change: str) -> None:
    epoch = baseline_epoch()
    row = epoch.issued[5]
    if change == "pending":
        row["policy_state_after"]["pending"]["light"] = {"event_tick": 105}
    elif change == "approach":
        row["policy_state_after"]["approaching"][0] = True
    elif change == "light":
        row["policy_state_after"]["previous_sensors"]["local_light"] = 14
    elif change == "counter":
        for seq in range(5, 10):
            epoch.issued[seq]["policy_state_after"]["previous_sensors"]["events"]["brush"] = {
                "count": 1,
                "last_game_tick": 105,
            }
    elif change == "missing_receipt":
        del epoch.receipts[5]
    elif change == "wrong_receipt":
        epoch.receipts[5]["executed_program"] = "HOVER"
    else:
        row["observation_tick"] += 1
        row["target_tick"] += 1
        row["policy_state_after"]["last_game_tick"] += 1
        epoch.receipts[5] = receipt(row)
    assert trials.stable_preparation(epoch, -1, 15) is None
    for seq in range(10, 16):
        row = issued(seq)
        if change == "counter":
            row["policy_state_after"]["previous_sensors"]["events"]["brush"] = {
                "count": 1,
                "last_game_tick": 105,
            }
        epoch.issued[seq], epoch.receipts[seq] = row, receipt(row)
        if seq < (14 if change == "counter" else 15):
            assert trials.stable_preparation(epoch, -1, 15) is None
    assert trials.stable_preparation(epoch, -1, 15) is epoch.issued[15]


def test_newer_unmatched_pending_row_invalidates_an_older_clean_baseline() -> None:
    epoch = baseline_epoch()
    epoch.issued[10] = issued(10)
    assert trials.stable_preparation(epoch, -1, 15) is epoch.issued[9]
    epoch.issued[10]["policy_state_after"]["pending"]["light"] = {"event_tick": 110}
    assert trials.stable_preparation(epoch, -1, 15) is None


def test_settle_waits_for_ten_rows_after_its_post_command_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = object.__new__(trials.Runner)
    epoch = baseline_epoch()
    runner.epoch = epoch
    records: list[dict[str, Any]] = []

    def wait(condition: Any, deadline: float) -> dict[str, Any]:
        assert condition() is None  # The ten old eligible rows cannot satisfy preparation.
        for seq in range(10, 20):
            epoch.issued[seq] = issued(seq)
            epoch.receipts[seq] = receipt(epoch.issued[seq])
            result = condition()
            if seq < 19:
                assert result is None
        return cast("dict[str, Any]", result)

    monkeypatch.setattr(runner, "wait", wait)
    monkeypatch.setattr(runner, "record", lambda kind, **fields: records.append(fields))
    assert runner.settle_preparation(1.0, 15) is epoch.issued[19]
    assert records[0]["after_command_seq"] == 9
    assert records[0]["first_observation_tick"] == 110
    assert records[0]["last_observation_tick"] == 119


def test_rotation_recheck_reuses_the_stable_window_after_two_fresh_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = object.__new__(trials.Runner)
    # 0..9 established the physical baseline; 10 and 11 arrived after the aim request.
    epoch = baseline_epoch(12)
    runner.epoch = epoch
    records: list[dict[str, Any]] = []
    monkeypatch.setattr(runner, "wait", lambda condition, deadline: condition())
    monkeypatch.setattr(runner, "record", lambda kind, **fields: records.append(fields))
    assert runner.settle_preparation(1.0, 15, after_seq=-1) is epoch.issued[11]
    assert len(epoch.issued) == 12  # No ten-observation aim delay was added.
    assert records[0]["phase"] == "rolling_after_rotation"
    assert records[0]["first_observation_tick"] == 102
    assert records[0]["last_observation_tick"] == 111


@pytest.mark.parametrize(
    "actor,ray,distance,expected",
    (
        (1, 1, 0.1, True),
        (1, 1, 2.5, True),
        (1, 1, 2.50001, False),
        (1, 1, 0.0, False),
        (1, 0, 1.0, False),
        (1, 2, 1.0, False),
        (1, 3, 1.0, False),
        (2, 1, 1.0, False),
        (3, 1, 1.0, False),
    ),
)
def test_native_geometry_requires_a_close_rear_player(
    actor: int,
    ray: int,
    distance: float,
    expected: bool,
) -> None:
    row = issued(0)
    frame = row["policy_state_after"]["previous_sensors"]
    frame["ray_actor_kinds"][ray], frame["ray_hit_kinds"][ray] = actor, 2
    frame["ray_distances_m"][ray] = distance
    assert trials.native_geometry(row) is expected


def test_full_arena_lighting_and_forward_food_preserve_world_fixture_bounds() -> None:
    for bright in (False, True):
        tokens = trials.lighting(bright).split()
        assert tokens[1:7] == ["-24", "65", "-24", "24", "70", "24"]
        assert 49 * 49 * 6 == 14406 < 32768
        assert tokens[-2] == "replace"
    assert trials.native_observer_position().endswith("^0 ^-1.0 ^-1.0 ~ 0")
    command = trials.food_stimulus()
    assert command.startswith(
        f"execute at {trials.FLY} rotated ~ 0 run summon minecraft:item ^0 ^-0.8 ^0.7 "
    )
    assert "Motion:[0d,0d,0d]" in command and "NoGravity" not in command
    assert len(command) <= 512


@pytest.mark.parametrize(
    "lane,repetition",
    (
        ("original", 0),
        ("original", 1),
        ("original", 2),
        ("shuffled_weight", 0),
        ("no_input", 0),
    ),
)
def test_light_repetitions_prepare_bright_then_request_dark_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lane: str,
    repetition: int,
) -> None:
    """The fixture direction repeats; this mock asserts no applied neural response."""
    runner = object.__new__(trials.Runner)
    case = trials.Case(0, lane, repetition, "light")
    runner.results = [{**asdict(case), "status": "unattempted"}]
    runner.run, runner.deadline, runner.safe = tmp_path, 200.0, True
    current = issued(0)
    events: list[tuple[str, Any]] = []
    runner.epoch = cast(
        trials.Epoch,
        SimpleNamespace(
            latest=lambda: current,
            issued={},
            receipts={},
            observer=None,
            identity=identity(),
            raw=b"epoch",
            poll=lambda: None,
        ),
    )

    def observed_after(tick: int, deadline: float) -> dict[str, Any]:
        nonlocal current
        current = issued(tick - 100)
        return current

    def settle(
        deadline: float,
        expected_light: int,
        *,
        after_seq: int | None = None,
    ) -> dict[str, Any]:
        assert expected_light == 15
        events.append(("baseline", expected_light))
        return observed_after(current["observation_tick"] + 10, deadline)

    monkeypatch.setattr(_clock(), "monotonic", lambda: 0.0)
    monkeypatch.setattr(runner, "send", lambda command: events.append(("send", command)))
    monkeypatch.setattr(runner, "record", lambda kind, **fields: events.append((kind, fields)))
    monkeypatch.setattr(runner, "ready", lambda deadline: current)
    monkeypatch.setattr(runner, "settle_preparation", settle)
    monkeypatch.setattr(runner, "observed_after", observed_after)
    runner.run_case(case)
    bright = events.index(("send", trials.lighting(True)))
    dark = events.index(("send", trials.lighting(False)))
    begin = next(index for index, (kind, _) in enumerate(events) if kind == "case_begin")
    assert bright < events.index(("baseline", 15)) < begin < dark
    assert sum(value == trials.lighting(False) for kind, value in events if kind == "send") == 1
    assert all(kind != "send" for kind, _ in events[dark + 1 :])
    assert runner.safe and runner.expected_light == 0
    assert runner.results[0]["outcome"] == "source_miss"


def test_ready_returns_the_observed_eligible_row_without_a_second_racing_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, newly_held = issued(0), issued(1, "light")
    rows = iter((baseline, newly_held))
    runner = object.__new__(trials.Runner)
    runner.epoch = cast(trials.Epoch, SimpleNamespace(latest=lambda: next(rows), issued={}))
    monkeypatch.setattr(runner, "wait", lambda condition, deadline: condition())
    assert runner.ready(1.0) is baseline
    assert next(rows) is newly_held


@pytest.mark.parametrize("source_pending", (False, True))
def test_ready_waits_for_newer_unpaired_source_and_its_cooldown_to_resolve(
    monkeypatch: pytest.MonkeyPatch,
    source_pending: bool,
) -> None:
    epoch = object.__new__(trials.Epoch)
    baseline, tail = issued(0), issued(1, None if source_pending else "light")
    if source_pending:
        tail["policy_state_after"]["pending"]["light"] = {"event_tick": 101}
    epoch.issued, epoch.receipts = {0: baseline, 1: tail}, {0: receipt(baseline)}
    monkeypatch.setattr(epoch, "poll", lambda: None)
    runner = object.__new__(trials.Runner)
    runner.epoch = epoch

    def wait(condition: Any, deadline: float) -> Any:
        assert epoch.latest() is baseline and trials.eligible(baseline)
        assert condition() is None
        epoch.receipts[1] = receipt(tail)
        assert condition() is None
        cooling = issued(2)
        cooling["policy_state_after"]["cooldown_until"] = 103
        epoch.issued[2], epoch.receipts[2] = cooling, receipt(cooling)
        assert condition() is None
        eligible_now = issued(3)
        epoch.issued[3], epoch.receipts[3] = eligible_now, receipt(eligible_now)
        return condition()

    monkeypatch.setattr(runner, "wait", wait)
    assert runner.ready(1.0) is epoch.issued[3]


@pytest.mark.parametrize("resolves", (False, True))
def test_preparation_hold_timeout_is_failed_without_cleanup_or_epoch_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resolves: bool,
) -> None:
    """Run execute() with fake diagnostics and clock, never an actual command inbox."""
    runner = object.__new__(trials.Runner)
    runner.cases = (trials.Case(0, "original", 0, "brush"),)
    runner.results = [{**asdict(case), "status": "unattempted"} for case in runner.cases]
    runner.output, runner.deadline, runner.safe = tmp_path, 200.0, False
    console, native, ledger = io.BytesIO(), io.BytesIO(), io.BytesIO()
    runner.console = cast(Any, console)
    runner.native = cast(Any, native)
    runner.ledger = cast(Any, ledger)
    now, pending, polls = 0.0, False, 0
    current = issued(0)
    events: list[tuple[str, Any]] = []
    observed_states: list[tuple[str, bool]] = []

    def latest() -> dict[str, Any]:
        observed_states.append((current["program"], trials.eligible(current)))
        return current

    epoch = SimpleNamespace(
        latest=latest,
        observer=None,
        poll=lambda: None,
        issued={0: current},
        receipts={0: receipt(current)},
    )
    runner.epoch = cast(trials.Epoch, epoch)

    def observe_current() -> None:
        assert not runner.safe
        events.append(("current_baseline", None))
        runner.safe = True

    def handover(lane: str) -> None:
        assert runner.safe and trials.eligible(current)
        events.append(("handover", lane))

    def send(command: str) -> None:
        nonlocal pending, current
        # Setup, preparation, and cleanup each invalidate readiness before mutation.
        assert not runner.safe
        events.append(("send", command))
        if runner.results[0]["status"] == "preparing" and command == trials.lighting(True):
            pending = True
            current = issued(current["command_seq"], "light")

    def live(deadline: float | None = None) -> None:
        nonlocal now, polls, current
        now += 0.1
        if now >= min(runner.deadline, deadline or runner.deadline):
            events.append(("timeout", current["program"]))
            raise trials.TrialError("bounded diagnostic deadline exhausted")
        seq = current["command_seq"] + 1
        row = issued(seq)
        if pending:
            polls += 1
            if not resolves or polls == 1:
                row = issued(seq, "light")
                row["policy_state_after"].update(
                    active_until=row["observation_tick"] + 40,
                    hold_remaining=40,
                )
            elif polls == 2:
                row["policy_state_after"]["cooldown_until"] = row["observation_tick"] + 1
        current = row
        epoch.issued[seq], epoch.receipts[seq] = row, receipt(row)
        observed_states.append((row["program"], trials.eligible(row)))

    def observed_after(tick: int, deadline: float) -> dict[str, Any]:
        nonlocal current
        assert not runner.safe
        light = current["policy_state_after"]["previous_sensors"]["local_light"]
        current = issued(tick - 100)
        epoch.issued[current["command_seq"]] = current
        epoch.receipts[current["command_seq"]] = receipt(current)
        current["policy_state_after"]["previous_sensors"]["local_light"] = light
        return current

    monkeypatch.setattr(_clock(), "monotonic", lambda: now)
    monkeypatch.setattr(_clock(), "sleep", lambda seconds: None)
    monkeypatch.setattr(runner, "observe_current_epoch", observe_current)
    monkeypatch.setattr(runner, "handover", handover)
    monkeypatch.setattr(runner, "send", send)
    monkeypatch.setattr(runner, "live", live)
    monkeypatch.setattr(runner, "observed_after", observed_after)
    monkeypatch.setattr(
        runner, "record", lambda kind, **fields: events.append((kind, copy.deepcopy(fields)))
    )
    result = runner.execute()
    assert now < 100  # The timeout is bounded and no real sleeping/launching occurred.
    assert json.loads((tmp_path / "summary.json").read_text()) == result
    assert epoch.observer is None
    assert console.closed and native.closed and ledger.closed
    assert trials.lighting(False) not in [value for kind, value in events if kind == "send"]
    if resolves:
        assert ("HOVER", False) in observed_states and ("CRUISE", False) in observed_states
        assert result["cases"][0]["status"] == "fixture_miss"
        assert result["original_restored"] and runner.safe
        assert [value for kind, value in events if kind == "handover"] == ["original", "original"]
    else:
        assert result["cases"][0]["status"] == "failed"
        assert result["cases"][0]["failure_phase"] == "preparing"
        assert not result["original_restored"] and not runner.safe
        assert current["program"] == "HOVER"  # Unresolved state was not erased.
        stopped = next(i for i, (kind, _) in enumerate(events) if kind == "timeout")
        assert not any(kind in ("send", "handover") for kind, _ in events[stopped:])
        assert [value for kind, value in events if kind == "handover"] == ["original"]
        assert any(
            kind == "case_end" and value["result"]["status"] == "failed" for kind, value in events
        )


@pytest.mark.parametrize(
    "failure",
    (None, "unready", "unpaired_hold", "unpaired_pending", "cleanup_unsettled"),
)
def test_cleanup_observes_readiness_then_restores_bright_before_original_handover(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str | None,
) -> None:
    runner = object.__new__(trials.Runner)
    case = trials.Case(0, "original", 0, "light")
    runner.cases, runner.results = (case,), [{**asdict(case), "status": "unattempted"}]
    runner.output, runner.deadline, runner.safe = tmp_path, 200.0, False
    runner.console, runner.native, runner.ledger = (
        cast(Any, io.BytesIO()),
        cast(Any, io.BytesIO()),
        cast(Any, io.BytesIO()),
    )
    current, case_finished = issued(0), False
    events: list[tuple[str, Any]] = []
    epoch = object.__new__(trials.Epoch)
    epoch.issued, epoch.receipts = {0: current}, {0: receipt(current)}
    monkeypatch.setattr(epoch, "poll", lambda: None)
    runner.epoch = epoch

    def observe_current() -> None:
        runner.safe = True

    def run_case(selected: trials.Case) -> None:
        nonlocal current, case_finished
        current = issued(10, "light" if failure == "unready" else None)
        current["policy_state_after"]["previous_sensors"]["local_light"] = 0
        epoch.issued[10], epoch.receipts[10] = current, receipt(current)
        if failure in ("unpaired_hold", "unpaired_pending"):
            tail = issued(11, "light" if failure == "unpaired_hold" else None)
            if failure == "unpaired_pending":
                tail["policy_state_after"]["pending"]["light"] = {"event_tick": 111}
            epoch.issued[11] = tail
            assert epoch.latest() is current and trials.eligible(current)
            assert 11 not in epoch.receipts and not trials.eligible(tail)
        runner.results[0]["status"] = "observed"
        runner.expected_light, runner.safe, case_finished = 0, True, True
        events.append(("case_finished", None))

    def wait(condition: Any, deadline: float) -> Any:
        row = condition()
        events.append(("ready", row is not None))
        if row is None:
            raise trials.TrialError("cleanup readiness deadline exhausted")
        return row

    def send(command: str) -> None:
        nonlocal current
        assert not runner.safe
        events.append(("send", command))
        if case_finished and command == trials.lighting(True):
            current = issued(11, "light")  # The cleanup light itself can induce a hold.

    def settle(deadline: float, expected_light: int) -> dict[str, Any]:
        nonlocal current
        assert not runner.safe and expected_light == 15
        events.append(("settle", expected_light))
        if case_finished:
            assert current["program"] == "HOVER" and not trials.eligible(current)
            if failure == "cleanup_unsettled":
                raise trials.TrialError("cleanup baseline deadline exhausted")
            current = issued(200)
            epoch.issued[200], epoch.receipts[200] = current, receipt(current)
        return current

    def handover(lane: str) -> None:
        assert runner.safe and trials.eligible(current)
        events.append(("handover", lane))

    monkeypatch.setattr(_clock(), "monotonic", lambda: 0.0)
    monkeypatch.setattr(runner, "observe_current_epoch", observe_current)
    monkeypatch.setattr(runner, "run_case", run_case)
    monkeypatch.setattr(runner, "wait", wait)
    monkeypatch.setattr(runner, "send", send)
    monkeypatch.setattr(runner, "settle_preparation", settle)
    monkeypatch.setattr(runner, "handover", handover)
    monkeypatch.setattr(runner, "record", lambda kind, **fields: events.append((kind, fields)))
    summary = runner.execute()
    cleanup = events[events.index(("case_finished", None)) + 1 :]
    unready = failure in ("unready", "unpaired_hold", "unpaired_pending")
    assert cleanup[0] == ("ready", not unready)
    if unready:
        assert all(kind not in ("send", "settle", "handover") for kind, _ in cleanup)
    else:
        bright = cleanup.index(("send", trials.lighting(True)))
        settled = cleanup.index(("settle", 15))
        assert bright < settled
        if failure is None:
            assert settled < cleanup.index(("handover", "original"))
        else:
            assert all(kind != "handover" for kind, _ in cleanup)
    assert summary["original_restored"] is (failure is None)
    assert runner.safe is (failure is None)
    assert any(kind == "restore_failed" for kind, _ in cleanup) is (failure is not None)


def test_native_dispatch_must_match_action_target_and_tick_window() -> None:
    request = {
        "seq": 2,
        "action": "attack",
        "entity_uuid": "fly",
        "game_tick": 100,
        "valid_through_tick": 180,
    }
    result = {
        "schema": "neurocraft.client_action_dispatch.v1",
        "status": "DISPATCHED",
        "seq": 2,
        "action": "attack",
        "requested_entity_uuid": "fly",
        "crosshair_entity_uuid": "fly",
        "request_game_tick": 100,
        "valid_through_tick": 180,
        "client_world_game_tick": 105,
    }
    assert trials.native_dispatch_matches(request, result)
    for key, value in (
        ("action", "brush"),
        ("seq", 1),
        ("requested_entity_uuid", "other"),
        ("crosshair_entity_uuid", "other"),
        ("client_world_game_tick", 181),
        ("request_game_tick", 99),
        ("status", "REFUSED"),
    ):
        assert not trials.native_dispatch_matches(request, {**result, key: value})


def aimed_result(request: dict[str, Any], *, brush_fallback: bool = True) -> dict[str, Any]:
    methods = [
        "player.lookAt(EYES,target.getBoundingBox().getCenter())",
        "gameRenderer.pick(1.0F)",
    ]
    if request["action"] == "aim_and_brush":
        methods.append("gameMode.interactAt(MAIN_HAND)")
        if brush_fallback:
            methods.append("gameMode.interact(MAIN_HAND)")
    else:
        methods.extend(("gameMode.attack", "player.swing(MAIN_HAND)"))
    return {
        "schema": "neurocraft.client_action_dispatch.v1",
        "status": "DISPATCHED",
        "reason": "server_result_unconfirmed",
        "seq": request["seq"],
        "action": request["action"],
        "requested_entity_uuid": request["entity_uuid"],
        "crosshair_entity_uuid": request["entity_uuid"],
        "request_game_tick": request["game_tick"],
        "valid_through_tick": request["valid_through_tick"],
        "client_world_game_tick": request["game_tick"] + 1,
        "attempted_methods": methods,
        "aim": {
            "before_client_view": {"yaw_degrees": 180.0, "pitch_degrees": 0.0},
            "after_client_view": {"yaw_degrees": 0.0, "pitch_degrees": -12.0},
            "eye_client_position_blocks": [0.0, 66.12, 0.0],
            "target_client_position_blocks": [0.0, 66.0, 1.0],
            "target_client_center_blocks": [0.0, 66.11, 1.0],
            "picker_result": {
                "type": "ENTITY",
                "entity_uuid": request["entity_uuid"],
                "hit_position_blocks": [0.0, 66.11, 0.81],
            },
        },
    }


@pytest.mark.parametrize("action", ("aim_and_brush", "aim_and_attack"))
@pytest.mark.parametrize(
    "defect",
    (
        None,
        "missing_aim",
        "null_aim",
        "missing_field",
        "extra_field",
        "null_picker",
        "wrong_picker_type",
        "wrong_picker_uuid",
        "short_hit",
        "boolean_eye",
        "string_target",
        "infinite_center",
        "huge_center",
        "missing_before_view",
        "boolean_after_view",
        "nan_after_view",
        "missing_methods",
        "wrong_method_order",
        "missing_native_method",
        "extra_method",
        "boolean_seq",
        "boolean_request_tick",
        "boolean_expiry",
    ),
)
def test_aimed_dispatch_requires_real_typed_target_picker_and_bound_methods(
    action: str, defect: str | None
) -> None:
    request = {
        "seq": 1,
        "action": action,
        "entity_uuid": "fly",
        "game_tick": 1,
        "valid_through_tick": 1,
    }
    result = aimed_result(request)
    result["client_world_game_tick"] = 1
    aim = result["aim"]
    if defect == "missing_aim":
        del result["aim"]
    elif defect == "null_aim":
        result["aim"] = None
    elif defect == "missing_field":
        del aim["target_client_position_blocks"]
    elif defect == "extra_field":
        aim["invented_target"] = "fly"
    elif defect == "null_picker":
        aim["picker_result"] = None
    elif defect == "wrong_picker_type":
        aim["picker_result"]["type"] = "BLOCK"
    elif defect == "wrong_picker_uuid":
        aim["picker_result"]["entity_uuid"] = "cow"
    elif defect == "short_hit":
        aim["picker_result"]["hit_position_blocks"] = [0.0, 1.0]
    elif defect == "boolean_eye":
        aim["eye_client_position_blocks"][0] = False
    elif defect == "string_target":
        aim["target_client_position_blocks"][0] = "0"
    elif defect == "infinite_center":
        aim["target_client_center_blocks"][0] = float("inf")
    elif defect == "huge_center":
        aim["target_client_center_blocks"][0] = 10**500
    elif defect == "missing_before_view":
        del aim["before_client_view"]["pitch_degrees"]
    elif defect == "boolean_after_view":
        aim["after_client_view"]["yaw_degrees"] = True
    elif defect == "nan_after_view":
        aim["after_client_view"]["pitch_degrees"] = float("nan")
    elif defect == "missing_methods":
        result.pop("attempted_methods")
    elif defect == "wrong_method_order":
        result["attempted_methods"][:2] = result["attempted_methods"][1::-1]
    elif defect == "missing_native_method":
        result["attempted_methods"] = result["attempted_methods"][:2]
    elif defect == "extra_method":
        result["attempted_methods"].append("authored_hit_result")
    elif defect == "boolean_seq":
        result["seq"] = True
    elif defect == "boolean_request_tick":
        result["request_game_tick"] = True
    elif defect == "boolean_expiry":
        result["valid_through_tick"] = True
    assert trials.native_dispatch_matches(request, result) is (defect is None)
    if defect is None and action == "aim_and_brush":
        direct = aimed_result(request, brush_fallback=False)
        direct["client_world_game_tick"] = 1
        assert trials.native_dispatch_matches(request, direct)


def walking_rows(game_tick: int = 100) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    request = {
        "seq": 2,
        "action": "walk_forward",
        "entity_uuid": "fly",
        "game_tick": game_tick,
        "valid_through_tick": game_tick + 80,
    }
    pose = {"position_blocks": [1.2, 65.0, 4.0], "on_ground": True, "flying": False}
    common = {
        "schema": "neurocraft.client_action_dispatch.v1",
        "seq": 2,
        "action": "walk_forward",
        "requested_entity_uuid": "fly",
        "request_game_tick": game_tick,
        "valid_through_tick": game_tick + 80,
        "player_uuid": "player",
        "dimension": "minecraft:overworld",
    }
    hold = {
        "start_game_tick": game_tick + 4,
        "start_client_pose": pose,
        "fixed_client_ticks": 20,
        "wall_timeout_ns": 3_000_000_000,
    }
    return request, [
        {
            **common,
            "status": "WALK_STARTED",
            "client_world_game_tick": game_tick + 4,
            "attempted_methods": ["options.keyUp.setDown(true)"],
            "walk": {**hold, "observed_client_ticks": 0, "forward_key_released": False},
        },
        {
            **common,
            "status": "WALK_COMPLETED",
            "client_world_game_tick": game_tick + 24,
            "attempted_methods": ["options.keyUp.setDown(false)"],
            "walk": {
                **hold,
                "observed_client_ticks": 20,
                "end_game_tick": game_tick + 24,
                "elapsed_ns": 1_000_000_000,
                "forward_key_released": True,
                "end_client_pose": {**pose, "position_blocks": [0.1, 65.0, 1.2]},
            },
        },
    ]


def tracking_rows(game_tick: int = 100) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    request, rows = walking_rows(game_tick)
    request["action"] = "walk_track_target"
    initial = {
        "game_tick": game_tick + 4,
        "before_client_view": {"yaw_degrees": 180.0, "pitch_degrees": 0.0},
        "after_client_view": {"yaw_degrees": 15.0, "pitch_degrees": 10.0},
        "eye_client_position_blocks": [1.2, 66.62, 4.0],
        "target_client_position_blocks": [0.0, 66.2, 5.5],
        "target_client_center_blocks": [0.0, 66.31, 5.5],
    }
    last = {
        "game_tick": game_tick + 23,
        "before_client_view": {"yaw_degrees": 17.0, "pitch_degrees": 9.0},
        "after_client_view": {"yaw_degrees": 20.0, "pitch_degrees": 8.0},
        "eye_client_position_blocks": [0.15, 66.62, 1.3],
        "target_client_position_blocks": [-0.5, 66.25, 3.2],
        "target_client_center_blocks": [-0.5, 66.36, 3.2],
    }
    for index, row in enumerate(rows):
        row["action"] = request["action"]
        walk = row["walk"]
        walk["start_client_pose"].update(yaw_degrees=180.0, pitch_degrees=0.0)
        walk["tracking"] = {
            "mode": "direct_fresh_target_center",
            "rule": "lookAt(EYES,target bounding-box center) before forward key-down and each "
            "continuing held callback; no aim on finish or abort",
            "aim_count": 20 if index else 1,
            "initial_observation": copy.deepcopy(initial),
            "last_observation": copy.deepcopy(last if index else initial),
        }
    rows[0]["walk"].update(elapsed_ns=0, end_game_tick=None, end_client_pose=None)
    rows[0]["attempted_methods"].insert(
        0, "player.lookAt(EYES,target.getBoundingBox().getCenter())"
    )
    rows[1]["walk"]["end_client_pose"].update(yaw_degrees=20.0, pitch_degrees=8.0)
    return request, rows


def approach_rows(
    game_tick: int = 100, *, callbacks: int = 12, distance: float = 0.5
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    request, rows = tracking_rows(game_tick)
    request["action"] = "walk_approach_target"
    for row in rows:
        row["action"] = request["action"]
        walk = row["walk"]
        del walk["fixed_client_ticks"]
        walk.update(
            max_client_ticks=20,
            stop_horizontal_distance_blocks=0.65,
            completion_reason=None,
            release_scope="forward key release; native momentum and subsequent coasting remain",
            terminal_geometry=None,
        )
    last = rows[1]["walk"]
    tick = game_tick + 4 + callbacks
    rows[1]["client_world_game_tick"] = tick
    last.update(
        observed_client_ticks=callbacks,
        elapsed_ns=callbacks * 50_000_000,
        end_game_tick=tick,
        completion_reason="stop_distance_reached" if distance <= 0.65 else "callback_limit_reached",
        terminal_geometry={
            "game_tick": tick,
            "eye_client_position_blocks": [0.1, 66.62, 1.2],
            "target_client_position_blocks": [0.1 + distance, 66.0, 1.2],
            "target_client_center_blocks": [0.1 + distance, 66.11, 1.2],
            "horizontal_distance_blocks": distance,
        },
    )
    last["tracking"]["aim_count"] = callbacks
    last["tracking"]["last_observation"]["game_tick"] = tick - 1
    rows[1]["reason"] = (
        "walking_stop_distance_reached" if distance <= 0.65 else "walking_callback_limit_reached"
    )
    return request, rows


@pytest.mark.parametrize(
    ("callbacks", "distance", "expected"),
    (
        (1, 0.65, True),
        (12, 0.5, True),
        (20, 0.65, True),
        (20, 0.65001, True),
        (19, 0.65001, False),
        (0, 0.5, False),
        (21, 0.5, False),
        (12, 0.05, False),
    ),
)
def test_approach_completion_distinguishes_arrival_callback_cap_and_undefined_heading(
    callbacks: int, distance: float, expected: bool
) -> None:
    request, rows = approach_rows(callbacks=callbacks, distance=distance)
    assert trials.walking_dispatch_evidence(request, rows)["native_hold_completed"] is expected


@pytest.mark.parametrize(
    "defect",
    (
        "aborted",
        "fixed_ticks",
        "float_max",
        "wrong_max",
        "bool_count",
        "float_count",
        "aim_count",
        "float_aim_count",
        "wrong_radius",
        "string_radius",
        "release_scope",
        "start_completion",
        "missing_start_completion",
        "start_terminal",
        "missing_start_terminal",
        "missing_terminal",
        "extra_terminal_field",
        "terminal_tick",
        "bool_terminal_tick",
        "nan_terminal",
        "wrong_distance",
        "string_distance",
        "nonfinite_distance",
        "terminal_endpoint",
        "arrival_reason",
        "top_reason",
        "last_aim_inside_stop_radius",
        "start_inside_stop_radius",
        "missing_tracking",
        "unchanged_initial",
        "late_aim_method",
        "start_endpoint",
        "start_elapsed",
        "start_end_tick",
        "last_aim_out_of_range",
        "huge_finite_integer_delta",
        "terminal_vertical_out_of_range",
        "terminal_target_feet",
    ),
)
def test_approach_receipts_cannot_relabel_aborts_or_malformed_arrival_as_completed(
    defect: str,
) -> None:
    request, rows = approach_rows()
    first, last = rows[0]["walk"], rows[1]["walk"]
    geometry = last["terminal_geometry"]
    if defect == "aborted":
        rows[1]["status"] = "WALK_ABORTED"
    elif defect == "fixed_ticks":
        last["fixed_client_ticks"] = 20
    elif defect in ("float_max", "wrong_max"):
        last["max_client_ticks"] = 20.0 if defect == "float_max" else 19
    elif defect in ("bool_count", "float_count"):
        last["observed_client_ticks"] = True if defect == "bool_count" else 12.0
    elif defect in ("aim_count", "float_aim_count"):
        last["tracking"]["aim_count"] = 13 if defect == "aim_count" else 12.0
    elif defect in ("wrong_radius", "string_radius"):
        last["stop_horizontal_distance_blocks"] = 0.7 if defect == "wrong_radius" else "0.65"
    elif defect == "release_scope":
        last["release_scope"] = "player stopped immediately"
    elif defect == "start_completion":
        first["completion_reason"] = "stop_distance_reached"
    elif defect == "missing_start_completion":
        del first["completion_reason"]
    elif defect == "start_terminal":
        first["terminal_geometry"] = geometry
    elif defect == "missing_start_terminal":
        del first["terminal_geometry"]
    elif defect == "missing_terminal":
        del last["terminal_geometry"]
    elif defect == "extra_terminal_field":
        geometry["predicted"] = True
    elif defect in ("terminal_tick", "bool_terminal_tick"):
        geometry["game_tick"] = 117 if defect == "terminal_tick" else True
    elif defect == "nan_terminal":
        geometry["target_client_center_blocks"][0] = float("nan")
    elif defect in ("wrong_distance", "string_distance", "nonfinite_distance"):
        geometry["horizontal_distance_blocks"] = {
            "wrong_distance": 0.55,
            "string_distance": "0.5",
            "nonfinite_distance": float("inf"),
        }[defect]
    elif defect == "terminal_endpoint":
        geometry["eye_client_position_blocks"][0] += 0.1
        geometry["target_client_center_blocks"][0] += 0.1
    elif defect == "arrival_reason":
        last["completion_reason"] = "callback_limit_reached"
    elif defect == "top_reason":
        rows[1]["reason"] = "key_hold_complete_server_movement_unconfirmed"
    elif defect == "last_aim_inside_stop_radius":
        observation = last["tracking"]["last_observation"]
        observation["target_client_center_blocks"] = observation["eye_client_position_blocks"][:]
    elif defect == "start_inside_stop_radius":
        for walk in (first, last):
            observation = walk["tracking"]["initial_observation"]
            observation["target_client_center_blocks"] = observation["eye_client_position_blocks"][
                :
            ]
        first["tracking"]["last_observation"] = copy.deepcopy(
            first["tracking"]["initial_observation"]
        )
    elif defect == "missing_tracking":
        del last["tracking"]
    elif defect == "unchanged_initial":
        last["tracking"]["initial_observation"]["target_client_center_blocks"][0] += 0.1
    elif defect == "late_aim_method":
        rows[1]["attempted_methods"].append(
            "player.lookAt(EYES,target.getBoundingBox().getCenter())"
        )
    elif defect == "start_endpoint":
        for walk in (first, last):
            walk["tracking"]["initial_observation"]["eye_client_position_blocks"][0] += 0.1
        first["tracking"]["last_observation"] = copy.deepcopy(
            first["tracking"]["initial_observation"]
        )
    elif defect == "start_elapsed":
        first["elapsed_ns"] = False
    elif defect == "start_end_tick":
        first["end_game_tick"] = rows[0]["client_world_game_tick"]
    elif defect == "last_aim_out_of_range":
        last["tracking"]["last_observation"]["target_client_center_blocks"][0] += 20
    elif defect == "huge_finite_integer_delta":
        observation = last["tracking"]["last_observation"]
        observation["eye_client_position_blocks"][0] = 10**308
        observation["target_client_center_blocks"][0] = -(10**308)
    elif defect == "terminal_vertical_out_of_range":
        geometry["target_client_center_blocks"][1] += 100
    elif defect == "terminal_target_feet":
        geometry["target_client_position_blocks"][0] += 100
    assert not trials.walking_dispatch_evidence(request, rows)["native_hold_completed"]


@pytest.mark.parametrize(
    "defect",
    (
        None,
        "missing_tracking",
        "wrong_mode",
        "wrong_rule",
        "extra_tracking_field",
        "boolean_start_aims",
        "wrong_start_aims",
        "short_end_aims",
        "extra_end_aim",
        "float_end_aims",
        "different_initial",
        "different_first_last",
        "missing_observation",
        "extra_observation_field",
        "boolean_observation_tick",
        "wrong_initial_tick",
        "early_last_tick",
        "late_last_tick",
        "boolean_eye",
        "short_target",
        "infinite_center",
        "huge_pose",
        "boolean_pose_yaw",
        "string_before_yaw",
        "nan_after_pitch",
        "extra_view_field",
        "missing_pose_view",
        "different_start_view",
        "boolean_player_uuid",
        "different_player",
        "wrong_action",
        "boolean_seq",
        "float_request_tick",
        "float_expiry",
        "missing_aim_method",
        "aim_on_finish",
        "float_fixed_ticks",
        "float_timeout",
        "float_retained_start_tick",
        "numeric_retained_ground",
    ),
)
def test_tracking_walk_requires_declared_rule_typed_aim_endpoints_and_matching_identity(
    defect: str | None,
) -> None:
    request, rows = tracking_rows()
    first, last = rows[0]["walk"], rows[1]["walk"]
    tracking = last["tracking"]
    observation = tracking["last_observation"]
    if defect == "missing_tracking":
        del last["tracking"]
    elif defect == "wrong_mode":
        tracking["mode"] = "predicted_target"
    elif defect == "wrong_rule":
        tracking["rule"] += "; retry"
    elif defect == "extra_tracking_field":
        tracking["predicted"] = True
    elif defect == "boolean_start_aims":
        first["tracking"]["aim_count"] = True
    elif defect == "wrong_start_aims":
        first["tracking"]["aim_count"] = 0
    elif defect == "short_end_aims":
        tracking["aim_count"] = 19
    elif defect == "extra_end_aim":
        tracking["aim_count"] = 21
    elif defect == "float_end_aims":
        tracking["aim_count"] = 20.0
    elif defect == "different_initial":
        tracking["initial_observation"]["target_client_center_blocks"][0] += 0.1
    elif defect == "different_first_last":
        first["tracking"]["last_observation"]["target_client_center_blocks"][0] += 0.1
    elif defect == "missing_observation":
        tracking["last_observation"] = None
    elif defect == "extra_observation_field":
        observation["invented_pose"] = [0.0] * 3
    elif defect == "boolean_observation_tick":
        observation["game_tick"] = True
    elif defect == "wrong_initial_tick":
        for walk in (first, last):
            walk["tracking"]["initial_observation"]["game_tick"] -= 1
        first["tracking"]["last_observation"]["game_tick"] -= 1
    elif defect == "early_last_tick":
        observation["game_tick"] = request["game_tick"] - 1
    elif defect == "late_last_tick":
        observation["game_tick"] = request["valid_through_tick"] + 1
    elif defect == "boolean_eye":
        observation["eye_client_position_blocks"][0] = False
    elif defect == "short_target":
        observation["target_client_position_blocks"].pop()
    elif defect == "infinite_center":
        observation["target_client_center_blocks"][0] = float("inf")
    elif defect == "huge_pose":
        last["end_client_pose"]["position_blocks"][0] = 10**500
    elif defect == "boolean_pose_yaw":
        last["end_client_pose"]["yaw_degrees"] = True
    elif defect == "string_before_yaw":
        observation["before_client_view"]["yaw_degrees"] = "17.0"
    elif defect == "nan_after_pitch":
        observation["after_client_view"]["pitch_degrees"] = float("nan")
    elif defect == "extra_view_field":
        observation["after_client_view"]["roll_degrees"] = 0.0
    elif defect == "missing_pose_view":
        del last["end_client_pose"]["pitch_degrees"]
    elif defect == "different_start_view":
        for walk in (first, last):
            walk["tracking"]["initial_observation"]["before_client_view"]["yaw_degrees"] = 1.0
        first["tracking"]["last_observation"]["before_client_view"]["yaw_degrees"] = 1.0
    elif defect == "boolean_player_uuid":
        for row in rows:
            row["player_uuid"] = True
    elif defect == "different_player":
        rows[1]["player_uuid"] = "someone_else"
    elif defect == "wrong_action":
        rows[1]["action"] = "walk_forward"
    elif defect == "boolean_seq":
        request["seq"] = rows[0]["seq"] = 1
        rows[1]["seq"] = True
    elif defect == "float_request_tick":
        rows[1]["request_game_tick"] = float(request["game_tick"])
    elif defect == "float_expiry":
        rows[1]["valid_through_tick"] = float(request["valid_through_tick"])
    elif defect == "missing_aim_method":
        rows[0]["attempted_methods"].pop(0)
    elif defect == "aim_on_finish":
        rows[1]["attempted_methods"].insert(0, rows[0]["attempted_methods"][0])
    elif defect == "float_fixed_ticks":
        last["fixed_client_ticks"] = 20.0
    elif defect == "float_timeout":
        last["wall_timeout_ns"] = 3_000_000_000.0
    elif defect == "float_retained_start_tick":
        last["start_game_tick"] = float(first["start_game_tick"])
    elif defect == "numeric_retained_ground":
        last["start_client_pose"] = {**last["start_client_pose"], "on_ground": 1}
    assert trials.walking_dispatch_evidence(request, rows)["native_hold_completed"] is (
        defect is None
    )


def test_tracking_walk_allows_corrected_endpoint_clocks_without_inventing_terminal_aim() -> None:
    request, rows = tracking_rows()
    rows[1]["client_world_game_tick"] = rows[1]["walk"]["end_game_tick"] = 102
    rows[1]["walk"]["tracking"]["last_observation"]["game_tick"] = 101
    assert rows[0]["client_world_game_tick"] == 104
    assert trials.walking_dispatch_evidence(request, rows)["native_hold_completed"]
    # A correction may also happen on the terminal callback after the last recorded aim.
    rows[1]["walk"]["tracking"]["last_observation"]["game_tick"] = 123
    assert trials.walking_dispatch_evidence(request, rows)["native_hold_completed"]


@pytest.mark.parametrize("tracking", (False, True))
@pytest.mark.parametrize(
    "defect",
    (
        None,
        "missing_end",
        "duplicate",
        "aborted",
        "wrong_target",
        "not_released",
        "short_hold",
        "timeout",
        "different_player",
        "airborne",
        "nonfinite",
        "missing_start_clock",
        "wrong_start_clock",
        "missing_end_clock",
        "wrong_end_clock",
        "boolean_start_count",
    ),
)
def test_native_walking_completion_requires_bounded_matching_hold_and_release(
    defect: str | None,
    tracking: bool,
) -> None:
    request, rows = tracking_rows() if tracking else walking_rows()
    if defect == "missing_end":
        rows.pop()
    elif defect == "duplicate":
        rows.append(rows[-1])
    elif defect == "aborted":
        rows[-1]["status"] = "WALK_ABORTED"
    elif defect == "wrong_target":
        rows[-1]["requested_entity_uuid"] = "other"
    elif defect == "not_released":
        rows[-1]["walk"]["forward_key_released"] = False
    elif defect == "short_hold":
        rows[-1]["walk"]["observed_client_ticks"] = 19
    elif defect == "timeout":
        rows[-1]["walk"]["elapsed_ns"] = 3_000_000_000
    elif defect == "different_player":
        rows[-1]["player_uuid"] = "other"
    elif defect == "airborne":
        rows[-1]["walk"]["end_client_pose"]["on_ground"] = False
    elif defect == "nonfinite":
        rows[-1]["walk"]["end_client_pose"]["position_blocks"][0] = float("nan")
    elif defect == "missing_start_clock":
        for row in rows:
            row["walk"].pop("start_game_tick")
    elif defect == "wrong_start_clock":
        for row in rows:
            row["walk"]["start_game_tick"] = 103
    elif defect == "missing_end_clock":
        rows[-1]["walk"].pop("end_game_tick")
    elif defect == "wrong_end_clock":
        rows[-1]["walk"]["end_game_tick"] = 123
    elif defect == "boolean_start_count":
        rows[0]["walk"]["observed_client_ticks"] = False
    assert trials.walking_dispatch_evidence(request, rows)["native_hold_completed"] is (
        defect is None
    )


def test_completed_walking_allows_bound_same_world_client_clock_correction() -> None:
    request, rows = walking_rows()
    rows[-1]["client_world_game_tick"] = 102
    rows[-1]["walk"]["end_game_tick"] = 102
    assert trials.walking_dispatch_evidence(request, rows)["native_hold_completed"]


@pytest.mark.parametrize(
    ("ticks", "distance", "completed", "expected"),
    [
        ([111], 1.0, True, True),
        ([109, 111], 1.0, True, False),
        ([110], 1.0, True, False),
        ([], 1.0, True, False),
        ([111], 0.5, True, False),
        ([111], float("nan"), True, False),
        ([111], 1.0, False, False),
        ([121], 1.0, True, False),
    ],
)
def test_walking_requires_actual_server_motion_and_source_after_start_barrier(
    ticks: list[int],
    distance: float,
    completed: bool,
    expected: bool,
) -> None:
    evidence = {
        "native_hold_completed": completed,
        "server_start_barrier_tick": 110,
        "server_end_barrier_tick": 120,
        "server_horizontal_displacement": distance,
    }
    assert trials.walking_trial_verified(evidence, ticks) is expected


@pytest.mark.parametrize("mode", ("native_walk", "native_tracking_walk", "native_approach_walk"))
@pytest.mark.parametrize("completed", (True, False))
def test_native_walking_framing_follows_released_gesture_and_final_server_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    completed: bool,
) -> None:
    runner = object.__new__(trials.Runner)
    factory = {
        "native_walk": walking_rows,
        "native_tracking_walk": tracking_rows,
        "native_approach_walk": approach_rows,
    }[mode]
    request, rows = factory(109)
    if not completed:
        rows[-1]["status"] = "WALK_ABORTED"
    runner.native_seq = 2
    runner.epoch, runner.expected_light = baseline_epoch(), 15
    now = 0.0
    events: list[tuple[str, Any]] = []
    runner.native = cast(
        Any, SimpleNamespace(append=lambda value: events.append(("native", value)))
    )
    positions = iter(([1.2, 65.0, 4.0], [0.1, 65.0, 1.2]))

    def position(deadline: float) -> list[float]:
        events.append(("position_query", None))
        return next(positions)

    def query(command: str, pattern: str, deadline: float) -> str:
        events.append(("query", command))
        return "110" if now == 0 else "125"

    def sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    monkeypatch.setattr(_clock(), "monotonic", lambda: now)
    monkeypatch.setattr(_clock(), "sleep", sleep)
    monkeypatch.setattr(runner, "live", lambda deadline=None: None)
    monkeypatch.setattr(runner, "record", lambda kind, **fields: events.append((kind, fields)))
    monkeypatch.setattr(runner, "native_results", lambda seq: rows[:1] if now == 0 else rows)
    monkeypatch.setattr(runner, "server_player_position", position)
    monkeypatch.setattr(runner, "server_query", query)
    monkeypatch.setattr(runner, "send", lambda command: events.append(("send", command)))
    result = runner.perform_walk(trials.Case(0, "original", 0, "player_approach"), request, 10.0)
    operations = [
        (kind, value)
        for kind, value in events
        if kind in ("position_query", "native", "query", "send")
    ]
    expected = [
        ("position_query", None),
        ("native", request),
        ("query", "time query gametime"),
        ("query", "time query gametime"),
        ("position_query", None),
    ]
    framed = mode == "native_approach_walk" and completed
    if framed:
        expected.append(("send", trials.camera_rotation(offset=True)))
        observed = next(i for i, (kind, _) in enumerate(events) if kind == "walking_observed")
        assert observed < next(i for i, (kind, _) in enumerate(events) if kind == "send")
        framing = next(value for kind, value in events if kind == "camera_framing_request")
        assert framing["after_server_end_barrier_tick"] == 125
        assert "source envelope unchanged" in framing["scope"]
    else:
        assert not any(kind == "camera_framing_request" for kind, _ in events)
    assert operations == expected
    assert runner.native_seq == 3 and result["native_hold_completed"] is completed
    assert result["server_horizontal_displacement"] > 0.5
    assert result["server_start_barrier_tick"] == 110
    assert result["server_end_barrier_tick"] == 125


@pytest.mark.parametrize(
    "change",
    (
        "neutral",
        "neutral_unmatched",
        "pending",
        "source",
        "hold",
        "cooldown",
        "approach",
        "light",
        "counter",
        "unmatched_pending",
        "unmatched_source",
        "deferred_pending",
        "missing",
        "stale",
        "wrong_receipt",
        "missing_epoch",
        "transient_pending",
        "transient_light",
        "settled_counter",
    ),
)
@pytest.mark.parametrize(
    "locomotion", ("native_walk", "native_tracking_walk", "native_approach_walk")
)
def test_walk_admission_rechecks_query_updates_without_dispatch_retry_or_fabricated_motion(
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    locomotion: str,
) -> None:
    runner = object.__new__(trials.Runner)
    epoch = baseline_epoch()
    runner.epoch, runner.expected_light, runner.native_seq = epoch, 15, 2
    factory = {
        "native_walk": walking_rows,
        "native_tracking_walk": tracking_rows,
        "native_approach_walk": approach_rows,
    }[locomotion]
    request, native_rows = factory(110 if change == "stale" else 109)
    runner.player_locomotion = locomotion
    commands: list[dict[str, Any]] = []
    records: list[tuple[str, dict[str, Any]]] = []
    positions: list[list[float]] = []
    pending: list[dict[str, Any]] = []
    accepted = change in ("neutral", "neutral_unmatched")
    if change == "missing_epoch":
        runner.epoch = None

    def put(row: dict[str, Any], *, matched: bool = True) -> None:
        epoch.issued[row["command_seq"]] = row
        if matched:
            epoch.receipts[row["command_seq"]] = receipt(row)

    def poll() -> None:
        for row in pending:
            put(row, matched=False)
        pending.clear()

    def position(deadline: float) -> list[float]:
        if positions:
            assert accepted
            result = [0.1, 65.0, 1.2]
        else:
            result = [1.2, 65.0, 4.0]
            row = issued(10)
            state = row["policy_state_after"]
            if change in ("pending", "unmatched_pending", "deferred_pending", "transient_pending"):
                state["pending"]["player_approach"] = {"event_tick": 110, "baseline": 0.0}
            elif change in ("source", "unmatched_source"):
                changed(row, "player_approach")
                state.update(
                    source="player_approach",
                    program="ESCAPE_JUMP",
                    held_event={"event_tick": 110, "baseline": 0.0},
                    active_until=134,
                    hold_remaining=24,
                )
                row["program"] = "ESCAPE_JUMP"
            elif change == "hold":
                state["hold_remaining"] = 1
            elif change == "cooldown":
                state["cooldown_until"] = 111
            elif change == "approach":
                state["approaching"][0] = True
            elif change in ("light", "transient_light"):
                state["previous_sensors"]["local_light"] = 14
            elif change in ("counter", "settled_counter"):
                state["previous_sensors"]["events"]["brush"] = {"count": 1, "last_game_tick": 110}
            if change == "deferred_pending":
                pending.append(row)  # Published only by the post-query admission poll.
            elif change not in ("missing_epoch", "stale"):
                put(
                    row,
                    matched=change
                    not in ("neutral_unmatched", "unmatched_pending", "unmatched_source"),
                )
            if change == "missing":
                epoch.receipts.clear()
            elif change == "wrong_receipt":
                epoch.receipts[10]["executed_program"] = "HOVER"
            elif change in ("transient_pending", "transient_light", "settled_counter"):
                for seq in range(11, 22):
                    later = issued(seq)
                    if change == "settled_counter":
                        later["policy_state_after"]["previous_sensors"]["events"]["brush"] = {
                            "count": 1,
                            "last_game_tick": 110,
                        }
                    put(later)
                # The current ten rows are neutral; the earlier contradiction still belongs here.
                assert trials.stable_preparation(epoch, -1, 15) is not None
        positions.append(result)
        return result

    def native_results(seq: int) -> list[dict[str, Any]]:
        assert accepted, "refused admission must not wait for or retry native dispatch"
        return native_rows

    barriers = iter((120, 140))
    runner.native = cast(Any, SimpleNamespace(append=commands.append))
    monkeypatch.setattr(epoch, "poll", poll)
    monkeypatch.setattr(runner, "server_player_position", position)
    monkeypatch.setattr(runner, "record", lambda kind, **fields: records.append((kind, fields)))
    monkeypatch.setattr(runner, "native_results", native_results)
    monkeypatch.setattr(runner, "server_query", lambda *args: str(next(barriers)))
    framing_commands: list[str] = []
    monkeypatch.setattr(runner, "send", framing_commands.append)
    result = runner.perform_walk(trials.Case(0, "original", 0, "player_approach"), request, 10.0)
    if accepted:
        assert commands == [request] and runner.native_seq == 3
        assert result["native_hold_completed"] and len(positions) == 2
        assert result["server_start_barrier_tick"] == 120
        assert result["server_end_barrier_tick"] == 140
        assert framing_commands == (
            [trials.camera_rotation(offset=True)] if locomotion == "native_approach_walk" else []
        )
        return
    assert not commands and runner.native_seq == 2 and len(positions) == 1
    assert not framing_commands
    assert result["admission_refused"] and result["reason"]
    assert result["native_hold_completed"] is False and result["native_results"] == []
    assert result["server_player_start"] == positions[0]
    for field in (
        "server_player_end",
        "server_horizontal_displacement",
        "server_start_barrier_tick",
        "server_end_barrier_tick",
    ):
        assert result[field] is None
    assert not trials.walking_trial_verified(result, [111])
    assert [kind for kind, _ in records] == ["walking_admission_refused"]


@pytest.mark.parametrize("targeting", trials.NATIVE_TARGETING)
@pytest.mark.parametrize(
    ("locomotion", "query_source", "tracking_outcome"),
    (
        ("native_walk", False, None),
        ("scripted", False, None),
        ("native_walk", True, None),
        ("native_tracking_walk", False, None),
        ("native_tracking_walk", True, None),
        ("native_tracking_walk", False, "verified"),
        ("native_tracking_walk", False, "bad_metadata"),
        ("native_tracking_walk", False, "early_source"),
        ("native_tracking_walk", False, "late_source"),
        ("native_approach_walk", False, None),
        ("native_approach_walk", True, None),
        ("native_approach_walk", False, "verified"),
        ("native_approach_walk", False, "bad_metadata"),
        ("native_approach_walk", False, "early_source"),
        ("native_approach_walk", False, "late_source"),
    ),
)
def test_native_heading_is_once_after_physical_preparation_before_hold_and_default_is_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    locomotion: str,
    query_source: bool,
    targeting: str,
    tracking_outcome: str | None,
) -> None:
    runner = object.__new__(trials.Runner)
    case = trials.Case(0, "original", 0, "player_approach")
    runner.player_locomotion = locomotion
    runner.native_targeting = targeting
    tracking = locomotion in ("native_tracking_walk", "native_approach_walk")
    approach = locomotion == "native_approach_walk"
    runner.results = [{**asdict(case), "status": "unattempted"}]
    runner.run, runner.deadline, runner.safe = tmp_path, 200.0, True
    runner.session = {"entity_id": "fly"}
    runner.native_seq = 2
    now, native_reads = 0.0, 0
    source_tick: int | None = None
    current = issued(0)
    events: list[tuple[str, Any]] = []
    requests: list[dict[str, Any]] = []
    epoch = cast(
        trials.Epoch,
        SimpleNamespace(
            latest=lambda: current,
            issued={0: current},
            receipts={0: receipt(current)},
            observer=None,
            identity=identity(),
            raw=b"epoch",
            poll=lambda: None,
        ),
    )
    runner.epoch = epoch

    def publish(row: dict[str, Any]) -> None:
        nonlocal current
        current = row
        seq = row["command_seq"]
        epoch.issued[seq], epoch.receipts[seq] = row, receipt(row)
        if epoch.observer is not None:
            for item in (row, epoch.receipts[seq]):
                epoch.observer({"epoch_identity": epoch.identity, "row": item})

    def observed_after(tick: int, deadline: float) -> dict[str, Any]:
        nonlocal source_tick
        events.append(("observation", tick))
        for seq in range(current["command_seq"] + 1, tick - 99):
            row = issued(seq)
            if requests and tracking_outcome is not None:
                delta = (
                    6
                    if tracking_outcome == "early_source"
                    else 27
                    if tracking_outcome == "late_source"
                    else 8
                )
                onset = requests[0]["game_tick"] + delta
                if row["observation_tick"] == onset - 1:
                    sensed = row["policy_state_after"]["previous_sensors"]
                    sensed["ray_hit_kinds"][0], sensed["ray_actor_kinds"][0] = 2, 1
                    sensed["ray_distances_m"][0] = 4.0
                elif row["observation_tick"] == onset:
                    source_tick = onset
            if source_tick is not None:
                before = copy.deepcopy(current["policy_state_after"])
                after = copy.deepcopy(before)
                after["last_game_tick"] = row["observation_tick"]
                if row["observation_tick"] == source_tick:
                    after.update(
                        source="player_approach",
                        held_event={"event_tick": source_tick, "baseline": 0.0},
                        program="ESCAPE_JUMP",
                        active_until=source_tick + 24,
                    )
                    after["previous_sensors"]["ray_distances_m"][0] = 3.0
                if row["observation_tick"] < source_tick + 24:
                    after["hold_remaining"] = source_tick + 24 - row["observation_tick"]
                else:
                    after.update(
                        source=None,
                        held_event=None,
                        program="CRUISE",
                        hold_remaining=0,
                        cooldown_until=source_tick + 144,
                    )
                row.update(
                    policy_state_before=before, policy_state_after=after, program=after["program"]
                )
            publish(row)
        return current

    def settle(
        deadline: float, expected_light: int, *, after_seq: int | None = None
    ) -> dict[str, Any]:
        if after_seq is None:
            events.append(("settle", expected_light))
            return observed_after(current["observation_tick"] + 10, deadline)
        events.append(("rolling_recheck", after_seq))
        return current

    def append(request: dict[str, Any]) -> None:
        requests.append(request)
        events.append(("native", request))

    def native_results(seq: int) -> list[dict[str, Any]]:
        nonlocal native_reads
        native_reads += 1
        request = requests[0]
        factory = approach_rows if approach else tracking_rows if tracking else walking_rows
        _, rows = factory(request["game_tick"])
        for row in rows:
            row.update(
                seq=seq,
                request_game_tick=request["game_tick"],
                valid_through_tick=request["valid_through_tick"],
            )
        if tracking_outcome == "bad_metadata":
            rows[1]["walk"]["tracking"]["aim_count"] = 19
        events.append(("walk_started" if native_reads == 1 else "walk_completed", None))
        return rows[:1] if native_reads == 1 else rows

    def sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    def query(command: str, pattern: str, deadline: float) -> str:
        events.append(("query", command))
        return str(requests[0]["game_tick"] + (6 if native_reads == 1 else 26))

    positions = iter(([1.2, 65.0, 4.0], [-1.0, 65.0, 4.0]))

    def position(deadline: float) -> list[float]:
        nonlocal source_tick
        if query_source:
            assert source_tick is None, "refusal must not fabricate a second endpoint query"
            # The read-only query polls a newly visible Player and then an actual decreasing ray.
            prior = issued(current["command_seq"] + 1)
            sensed = prior["policy_state_after"]["previous_sensors"]
            sensed["ray_hit_kinds"][0], sensed["ray_actor_kinds"][0] = 2, 1
            sensed["ray_distances_m"][0] = 4.0
            publish(prior)
            row = issued(current["command_seq"] + 1)
            before = copy.deepcopy(current["policy_state_after"])
            after = copy.deepcopy(before)
            source_tick = row["observation_tick"]
            after.update(
                last_game_tick=source_tick,
                source="player_approach",
                held_event={"event_tick": source_tick, "baseline": 0.0},
                program="ESCAPE_JUMP",
                active_until=source_tick + 24,
                hold_remaining=24,
            )
            after["previous_sensors"]["ray_distances_m"][0] = 3.0
            row.update(policy_state_before=before, policy_state_after=after, program="ESCAPE_JUMP")
            publish(row)
        return next(positions)

    def ready(deadline: float) -> dict[str, Any]:
        events.append(("ready", current["observation_tick"]))
        if source_tick is not None and current["observation_tick"] < source_tick + 144:
            assert not trials.eligible(current)
            observed_after(source_tick + 144, deadline)
        return current

    runner.native = cast(Any, SimpleNamespace(append=append))
    monkeypatch.setattr(_clock(), "monotonic", lambda: now)
    monkeypatch.setattr(_clock(), "sleep", sleep)
    monkeypatch.setattr(runner, "live", lambda deadline=None: None)
    monkeypatch.setattr(runner, "send", lambda command: events.append(("send", command)))
    monkeypatch.setattr(runner, "record", lambda kind, **fields: events.append((kind, fields)))
    monkeypatch.setattr(runner, "ready", ready)
    monkeypatch.setattr(runner, "settle_preparation", settle)
    monkeypatch.setattr(runner, "observed_after", observed_after)
    monkeypatch.setattr(runner, "native_results", native_results)
    monkeypatch.setattr(runner, "server_query", query)
    monkeypatch.setattr(runner, "server_player_position", position)
    runner.run_case(case)

    commands = [value for kind, value in events if kind == "send"]
    heading = f"execute at {trials.FLY} rotated ~90 0 run rotate {trials.PLAYER} ~ ~"
    assert trials.walking_heading() == heading
    if locomotion == "scripted":
        assert heading not in commands and not requests
        assert trials.walking_observer_position() not in commands
        assert trials.walking_observer_position(tracking=True) not in commands
        assert commands.count(trials.camera_rotation(offset=True)) == 1
        assert [command for command in commands if "run tp" in command] == [
            f"execute at {trials.FLY} rotated ~ 0 run tp {trials.PLAYER} "
            f"^0 ^-1.5 ^{distance} ~180 0"
            for distance in (5.5, 4.5, 3.5, 2.5, 1.5)
        ]
        return

    assert commands.count(heading) == (0 if tracking else 1)
    farther = tracking and not approach
    walking_position = trials.walking_observer_position(tracking=farther, approach=approach)
    assert walking_position == (
        f"execute at {trials.FLY} rotated ~ 0 positioned ^"
        + "1.2"
        + " ^0 ^"
        + ("5.5" if farther else "4.0")
        + f" run tp {trials.PLAYER} ~ 65 ~ ~180 0"
    )
    assert commands.count(walking_position) == 1
    assert trials.walking_observer_position(tracking=not farther) not in commands
    post_walk_framed = approach and not query_source and tracking_outcome != "bad_metadata"
    assert commands.count(trials.camera_rotation(offset=True)) == int(post_walk_framed)
    placement = events.index(("send", walking_position))
    aim = (
        next(index for index, (kind, _) in enumerate(events) if kind == "framing_observations")
        if tracking
        else events.index(("send", heading))
    )
    if query_source:
        result = runner.results[0]
        assert source_tick == 136 and not requests and runner.native_seq == 2
        assert result["start_tick"] == 135 and result["end_tick"] == 234
        assert result["observed_source_ticks"] == [source_tick]
        assert result["complete_applied_hold"] and result["applied_program_counts"] == {
            "ESCAPE_JUMP": 24
        }
        assert result["status"] == "stimulus_failed" and not result["fixture_verified"]
        assert result["walking"]["admission_refused"]
        assert result["walking"]["native_results"] == []
        assert ("observation", 236) in events and ("ready", 236) in events
        assert ("observation", source_tick + 144) in events and runner.safe
        refusal = next(
            index for index, (kind, _) in enumerate(events) if kind == "walking_admission_refused"
        )
        assert not [value for kind, value in events[refusal:] if kind in ("send", "native")]
        return
    dispatch = next(index for index, (kind, _) in enumerate(events) if kind == "native")
    assert placement < aim < dispatch
    # Ten physical baseline observations precede the heading; only the existing two fresh
    # observations and rolling baseline recheck separate that heading from the native hold.
    assert ("settle", 15) in events[placement:aim]
    assert ("observation", 132) in events[placement:aim]
    assert ("observation", 134) in events[placement if tracking else aim : dispatch]
    assert ("rolling_recheck", 22) in events[placement if tracking else aim : dispatch]
    assert [value for kind, value in events[dispatch:] if kind == "send"] == (
        [trials.camera_rotation(offset=True)] if post_walk_framed else []
    )
    assert [kind for kind, _ in events if kind in ("walk_started", "walk_completed")] == [
        "walk_started",
        "walk_completed",
    ]
    result = runner.results[0]
    assert result["walking"]["native_hold_completed"] is (tracking_outcome != "bad_metadata")
    assert len(requests) == 1 and runner.native_seq == 3
    assert requests[0]["action"] == (
        "walk_approach_target" if approach else "walk_track_target" if tracking else "walk_forward"
    )
    assert requests[0]["game_tick"] == 134
    assert requests[0]["valid_through_tick"] - requests[0]["game_tick"] == 80
    if tracking:
        assert not any(kind == "walking_heading_request" for kind, _ in events)
        assert "synthetic native target-tracking" in result["walking"]["scope"]
        assert (
            "synthetic target-tracking aim starts with native key hold" in events[aim][1]["scope"]
        )
        assert result["fixture_verified"] is (tracking_outcome == "verified")
        if tracking_outcome is not None:
            assert result["observed_source_ticks"] == [source_tick]
            assert result["complete_applied_hold"] and result["applied_program_counts"] == {
                "ESCAPE_JUMP": 24
            }
            assert ("observation", 236) in events and ("ready", 236) in events
            assert source_tick is not None and ("observation", source_tick + 144) in events
            assert result["walking"]["server_start_barrier_tick"] == 140
            assert result["walking"]["server_end_barrier_tick"] == 160
            assert result["walking"]["server_horizontal_displacement"] > 0.5
    else:
        heading_record = next(
            fields for kind, fields in events if kind == "walking_heading_request"
        )
        assert heading_record["command"] == heading and "unmeasured" in heading_record["scope"]


def test_server_time_barrier_ignores_old_and_partial_log_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = object.__new__(trials.Runner)
    runner.run = tmp_path
    path = tmp_path / "server-gradle.log"
    path.write_text("[old] The time is 20\n")
    observations: list[Any] = []

    def send(command: str) -> None:
        assert command == "time query gametime"
        with path.open("a") as stream:
            stream.write("[fresh] The time is 1")

    def wait(condition: Any, deadline: float) -> Any:
        observations.append(condition())
        with path.open("a") as stream:
            stream.write("10\n")
        return condition()

    monkeypatch.setattr(runner, "send", send)
    monkeypatch.setattr(runner, "wait", wait)
    monkeypatch.setattr(runner, "record", lambda kind, **fields: None)
    assert (
        runner.server_query("time query gametime", r"The time is (\d+)", time.monotonic() + 2)
        == "110"
    )
    assert observations == [None]


@pytest.mark.parametrize("receipt_delay", (1, 2))
@pytest.mark.parametrize(
    "collection_fault", (None, "missing_receipt", "snapshot_budget", "observer_budget")
)
def test_case_attachment_keeps_buffered_issues_and_later_receipts_for_the_full_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt_delay: int,
    collection_fault: str | None,
) -> None:
    """Exercise real Epoch.poll offsets with a matched baseline behind issued rows."""
    ident = identity()
    path = tmp_path / str(ident["session_epoch"])
    path.mkdir()
    metadata = {"controller_profile": trials.PROFILE, "name": "original", "identity": ident}
    (path / "epoch.json").write_bytes(trials.canonical(metadata))
    ledger = path / "program-decisions.ndjson"
    emitted: dict[int, dict[str, Any]] = {}
    events: list[tuple[str, Any]] = []
    dark, withheld_seq = False, None

    def append_until(last_seq: int) -> None:
        with ledger.open("ab") as stream:
            for seq in range(len(emitted), last_seq + 1):
                row = issued(seq)
                if seq:
                    row["policy_state_before"] = copy.deepcopy(
                        emitted[seq - 1]["policy_state_after"]
                    )
                row["policy_state_after"]["previous_sensors"]["local_light"] = 0 if dark else 15
                emitted[seq] = row
                stream.write(trials.canonical(row) + b"\n")
                applied_seq = seq - receipt_delay
                if applied_seq >= 0 and applied_seq != withheld_seq:
                    stream.write(trials.canonical(receipt(emitted[applied_seq])) + b"\n")

    append_until(9 + receipt_delay)
    epoch = trials.Epoch(path, metadata, ident)
    runner = object.__new__(trials.Runner)
    case = trials.Case(0, "original", 0, "light")
    runner.results = [{**asdict(case), "status": "unattempted"}]
    runner.run, runner.deadline, runner.safe, runner.epoch = tmp_path, 200.0, True, epoch

    def latest() -> dict[str, Any]:
        row = epoch.latest()
        assert row is not None
        return row

    def observed_after(tick: int, deadline: float) -> dict[str, Any]:
        append_until(tick - 100 + receipt_delay)
        row = latest()
        assert row["observation_tick"] == tick
        return row

    def settle(
        deadline: float, expected_light: int, *, after_seq: int | None = None
    ) -> dict[str, Any]:
        observed_after(latest()["observation_tick"] + trials.BASELINE_OBSERVATIONS, deadline)
        row = trials.stable_preparation(epoch, after_seq if after_seq is not None else -1, 15)
        assert row is not None
        assert max(epoch.issued) == row["command_seq"] + receipt_delay
        if after_seq is not None and collection_fault == "snapshot_budget":
            monkeypatch.setattr(trials, "MAX_CASE_ROWS", 1 + receipt_delay)
        return row

    def send(command: str) -> None:
        nonlocal dark, withheld_seq
        if command == trials.lighting(False):
            dark = True
            if collection_fault == "missing_receipt":
                withheld_seq = latest()["command_seq"] + 1
            if collection_fault == "observer_budget":
                monkeypatch.setattr(trials, "MAX_CASE_ROWS", 2 + receipt_delay)

    monkeypatch.setattr(_clock(), "monotonic", lambda: 0.0)
    monkeypatch.setattr(runner, "send", send)
    monkeypatch.setattr(runner, "record", lambda kind, **fields: events.append((kind, fields)))
    monkeypatch.setattr(runner, "ready", lambda deadline: latest())
    monkeypatch.setattr(runner, "settle_preparation", settle)
    monkeypatch.setattr(runner, "observed_after", observed_after)
    if collection_fault in ("snapshot_budget", "observer_budget"):
        with pytest.raises(trials.TrialError, match="case row budget exhausted"):
            runner.run_case(case)
        assert runner.results[0]["status"] == "failed"
        assert epoch.observer is None
        assert not any(kind == "decision_row" for kind, _ in events)
        return
    runner.run_case(case)

    result = runner.results[0]
    assert result["status"] == "observed" and result["source_verified"]
    assert result["outcome"] == "no_applied_response"  # Collection never invents neural responses.
    assert result["response_window_issued"] == 100
    expected_missing = [withheld_seq] if collection_fault == "missing_receipt" else []
    assert result["response_window_missing_receipts"] == expected_missing
    assert result["response_window_unapplied"] == expected_missing
    beginning = next(fields for kind, fields in events if kind == "case_begin")
    initial = beginning["baseline_records"]
    assert len(initial) == 2 + receipt_delay
    assert [entry["row"]["kind"] for entry in initial] == [
        "issued",
        "receipt",
        *["issued"] * receipt_delay,
    ]
    collected = initial + [fields for kind, fields in events if kind == "decision_row"]
    keys = [(entry["row"]["kind"], entry["row"]["command_seq"]) for entry in collected]
    assert len(keys) == len(set(keys))
    assert all(entry["epoch_identity"] == ident for entry in collected)
    observed = [
        entry["row"]
        for entry in collected
        if entry["row"]["kind"] == "issued"
        and result["start_tick"] <= entry["row"]["observation_tick"] <= result["end_tick"]
    ]
    assert [row["observation_tick"] for row in observed] == list(
        range(result["start_tick"], result["end_tick"] + 1)
    )
    assert all(
        second["observation_seq"] == first["observation_seq"] + 1
        and second["neural_step_start"] == first["neural_step_end"]
        for first, second in pairwise(observed)
    )
    assert epoch.observer is None


def test_current_epoch_attachment_drains_old_eligible_prefix_and_waits_for_new_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ident = identity()
    path = tmp_path / "epochs" / str(ident["session_epoch"])
    path.mkdir(parents=True)
    metadata = {"controller_profile": trials.PROFILE, "name": "original", "identity": ident}
    (path / "epoch.json").write_bytes(trials.canonical(metadata))
    raw = bytearray()
    seq = 0
    while len(raw) <= 2 * 524_288:
        row = issued(seq)
        raw += trials.canonical(row) + b"\n" + trials.canonical(receipt(row)) + b"\n"
        seq += 1
    held = issued(seq, "light")
    held["policy_state_after"].update(active_until=held["observation_tick"] + 40, hold_remaining=40)
    raw += trials.canonical(held) + b"\n" + trials.canonical(receipt(held)) + b"\n"
    ledger = path / "program-decisions.ndjson"
    ledger.write_bytes(raw)
    runner = object.__new__(trials.Runner)
    runner.run, runner.deadline, runner.safe, runner.epoch = tmp_path, 100.0, False, None
    runner.catalog = {"original": metadata}
    runner.session = {"run_id": ident["run_id"], "entity_id": ident["entity_id"]}
    now, drained_polls = 0.0, 0
    states_seen: list[bool] = []

    def live(deadline: float | None = None) -> None:
        nonlocal now, drained_polls
        now += 0.1
        assert now < (deadline or runner.deadline)
        if runner.epoch is None:
            return
        epoch = runner.epoch
        previous_new = epoch.last_new
        epoch.poll()
        assert not runner.safe
        if epoch.offset <= epoch.attachment_end:
            assert epoch.last_new == previous_new  # Reading backlog is not new liveness.
        if epoch.offset >= epoch.attachment_end:
            drained_polls += 1
            states_seen.append(trials.eligible(epoch.latest()))
            # Keep the attached tail held, then observe cooldown before fresh eligibility.
            if drained_polls in (2, 3):
                row = issued(seq + drained_polls)
                row["policy_state_after"]["cooldown_until"] = (
                    row["observation_tick"] + 1 if drained_polls == 2 else 0
                )
                with ledger.open("ab") as stream:
                    stream.write(trials.canonical(row) + b"\n")
                    stream.write(trials.canonical(receipt(row)) + b"\n")

    monkeypatch.setattr(_clock(), "monotonic", lambda: now)
    monkeypatch.setattr(_clock(), "sleep", lambda seconds: None)
    monkeypatch.setattr(runner, "live", live)
    commands: list[str] = []
    monkeypatch.setattr(runner, "send", commands.append)
    runner.observe_current_epoch()
    assert runner.safe and not commands
    assert drained_polls >= 3 and states_seen and not any(states_seen)
    attached = cast(trials.Epoch, runner.epoch)
    latest = attached.latest()
    assert latest is not None and latest["command_seq"] == seq + 3


@pytest.mark.parametrize(
    ("targeting", "source", "dispatch"),
    [
        (targeting, source, dispatch)
        for targeting in trials.NATIVE_TARGETING
        for source in ("brush", "attack")
        for dispatch in (
            "DISPATCHED",
            "REFUSED",
            "late",
            "geometry_miss",
            "no_source",
            "wrong_counter",
        )
    ]
    + [("aim_at_dispatch", source, "bad_aim") for source in ("brush", "attack")],
)
def test_native_aim_and_post_dispatch_framing_are_rotation_only_and_never_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    targeting: str,
    source: str,
    dispatch: str,
) -> None:
    runner = object.__new__(trials.Runner)
    runner.native_targeting = targeting
    case = trials.Case(0, "original", 0, source)
    aimed = targeting == "aim_at_dispatch"
    hold_ticks = 60 if source == "brush" else 40
    runner.results = [{**asdict(case), "status": "unattempted"}]
    runner.run, runner.deadline, runner.safe = tmp_path, 200.0, True
    runner.session = {"entity_id": "fly"}
    runner.native_seq = 0
    now, response_window_observed = 0.0, False
    current = issued(0)
    events: list[tuple[str, Any]] = []
    requests: list[dict[str, Any]] = []
    runner.epoch = cast(
        trials.Epoch,
        SimpleNamespace(
            latest=lambda: current,
            issued={},
            receipts={},
            observer=None,
            identity=identity(),
            raw=b"epoch",
            poll=lambda: None,
        ),
    )

    def append(request: dict[str, Any]) -> None:
        requests.append(request)
        events.append(("native", request))

    runner.native = cast(Any, SimpleNamespace(append=append))

    def observed_after(tick: int, deadline: float) -> dict[str, Any]:
        nonlocal current, response_window_observed
        events.append(("observation", tick))
        for observed_tick in range(current["observation_tick"] + 1, tick + 1):
            row = issued(observed_tick - 100)
            before = copy.deepcopy(current["policy_state_after"])
            after = copy.deepcopy(before)
            after["last_game_tick"] = observed_tick
            sensed = after["previous_sensors"]
            if dispatch != "geometry_miss":
                sensed["ray_hit_kinds"][1], sensed["ray_actor_kinds"][1] = 2, 1
                sensed["ray_distances_m"][1] = 1.8
            row.update(policy_state_before=before, policy_state_after=after)
            if requests:
                response_window_observed = True
                onset = requests[0]["game_tick"] + 1
                if observed_tick == onset and dispatch != "no_source":
                    changed(
                        row,
                        ("brush" if source == "attack" else "attack")
                        if dispatch == "wrong_counter"
                        else source,
                    )
                    if dispatch != "wrong_counter":
                        after.update(
                            source=source,
                            held_event={"event_tick": onset, "baseline": 0.0},
                            active_until=onset + hold_ticks,
                            program=trials.PROGRAMS[source][0],
                        )
                if after["source"] is not None:
                    after["hold_remaining"] = max(0, after["active_until"] - observed_tick)
                    if after["hold_remaining"] == 0:
                        after.update(
                            source=None,
                            held_event=None,
                            program="CRUISE",
                            cooldown_until=onset + hold_ticks + 120,
                        )
                row["program"] = after["program"]
            current = row
            assert runner.epoch is not None
            runner.epoch.receipts[row["command_seq"]] = receipt(row)
            if runner.epoch.observer is not None:
                for entry in bound(row, receipt(row)):
                    runner.epoch.observer(entry)
        return current

    def ready(deadline: float) -> dict[str, Any]:
        events.append(("ready", current["observation_tick"]))
        cooldown = current["policy_state_after"]["cooldown_until"]
        if current["observation_tick"] < cooldown:
            assert not trials.eligible(current)
            observed_after(cooldown, deadline)
        assert trials.eligible(current)
        return current

    def settle(
        deadline: float,
        expected_light: int,
        *,
        after_seq: int | None = None,
    ) -> dict[str, Any]:
        if after_seq is None:
            events.append(("settle", expected_light))
            return observed_after(current["observation_tick"] + 10, deadline)
        events.append(("rolling_recheck", after_seq))
        return current

    def results(seq: int) -> list[dict[str, Any]]:
        if dispatch == "late" and not response_window_observed:
            return []
        request = requests[0]
        if aimed:
            result = aimed_result(request)
            if dispatch == "bad_aim":
                result["aim"]["picker_result"]["entity_uuid"] = "cow"
            elif dispatch == "REFUSED":
                result.update(
                    status="REFUSED", reason="line_of_sight_blocked", aim=None, attempted_methods=[]
                )
            return [result]
        return [
            {
                "schema": "neurocraft.client_action_dispatch.v1",
                "status": "REFUSED" if dispatch == "REFUSED" else "DISPATCHED",
                "seq": seq,
                "action": request["action"],
                "requested_entity_uuid": "fly",
                "crosshair_entity_uuid": "fly",
                "request_game_tick": request["game_tick"],
                "valid_through_tick": request["valid_through_tick"],
                "client_world_game_tick": request["game_tick"] + 1,
            }
        ]

    def sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    monkeypatch.setattr(_clock(), "monotonic", lambda: now)
    monkeypatch.setattr(_clock(), "sleep", sleep)
    monkeypatch.setattr(runner, "live", lambda deadline=None: None)
    monkeypatch.setattr(runner, "send", lambda command: events.append(("send", command)))
    monkeypatch.setattr(runner, "record", lambda kind, **fields: events.append((kind, fields)))
    monkeypatch.setattr(runner, "ready", ready)
    monkeypatch.setattr(runner, "settle_preparation", settle)
    monkeypatch.setattr(runner, "observed_after", observed_after)
    monkeypatch.setattr(runner, "native_results", results)
    runner.run_case(case)
    result = runner.results[0]
    assert (
        sum(value == trials.native_observer_position() for kind, value in events if kind == "send")
        == 1
    )
    light = events.index(("send", trials.lighting(True)))
    body = next(
        index
        for index, (kind, value) in enumerate(events)
        if kind == "send" and value.startswith(f"tp {trials.FLY}")
    )
    placement = events.index(("send", trials.native_observer_position()))
    assert light < events.index(("settle", 15)) < body < placement
    if dispatch == "geometry_miss" and not aimed:
        assert result["status"] == "fixture_miss" and runner.safe and not requests
        assert runner.native_seq == 0
        return
    assert len(requests) == 1 and response_window_observed and runner.safe
    assert runner.native_seq == 1
    assert set(requests[0]) == {"seq", "action", "entity_uuid", "game_tick", "valid_through_tick"}
    assert requests[0]["action"] == (trials.AIM_ACTIONS[source] if aimed else source)
    assert requests[0]["valid_through_tick"] - requests[0]["game_tick"] == 80
    direct = (
        f"execute as {trials.PLAYER} at @s anchored eyes facing entity {trials.FLY} eyes "
        "run rotate @s ~ ~"
    )
    offset = (
        f"execute as {trials.PLAYER} at @s anchored eyes facing entity {trials.FLY} eyes "
        "rotated ~-20 ~ run rotate @s ~ ~"
    )
    assert trials.camera_rotation(offset=False) == direct
    assert trials.camera_rotation(offset=True) == offset
    native = next(index for index, (kind, _) in enumerate(events) if kind == "native")
    if aimed:
        assert ("send", direct) not in events
        assert ("observation", 134) in events[:native]
        assert ("rolling_recheck", 22) in events[:native]
    else:
        aim = events.index(("send", direct))
        assert events[aim + 2 : native][:2] == [
            ("observation", 134),
            ("rolling_recheck", 22),
        ]
    assert sum(kind == "settle" for kind, _ in events[:native]) == 3
    framing = next(fields for kind, fields in events if kind == "framing_observations")
    assert framing["request_after_observation_tick"] == 132
    assert framing["fresh_observation_tick"] == 134
    assert requests[0]["game_tick"] == 134
    assert result["start_tick"] == 135 and result["end_tick"] == 234
    assert ("observation", 236) in events and ("ready", 236) in events
    post_commands = [value for kind, value in events[native + 1 :] if kind == "send"]
    verified = dispatch not in ("REFUSED", "late", "bad_aim")
    source_verified = dispatch not in ("no_source", "wrong_counter")
    assert post_commands == ([offset] if verified else [])
    assert result["native_dispatch_verified"] is verified
    assert result["source_verified"] is source_verified
    assert result["observed_source_ticks"] == ([135] if source_verified else [])
    assert result["complete_applied_hold"] is source_verified
    assert result["applied_program_counts"] == {
        trials.PROGRAMS[source][0]: hold_ticks if source_verified else 0
    }
    assert result["fixture_verified"] is (verified and source_verified)
    if source_verified:
        assert ("observation", 135 + hold_ticks + 120) in events
    if not verified:
        assert result["status"] == "stimulus_failed"
        assert result["outcome"] == "native_dispatch_unconfirmed"
    assert all(
        fields["scope"] == "rotation-only framing request; not visual evidence"
        for kind, fields in events
        if kind == "camera_framing_request"
    )


def test_native_results_are_bounded_and_ignore_partial_final_rows(tmp_path: Path) -> None:
    runner = object.__new__(trials.Runner)
    runner.run = tmp_path
    path = tmp_path / "diagnostic-client-actions-results.ndjson"
    assert runner.native_results(1) == []
    path.write_bytes(b'{"seq":0}\n{"seq":1}\n{"seq":1')
    assert runner.native_results(1) == [{"seq": 1}]
    path.write_bytes(b" " * (trials.NATIVE_RESULTS_LIMIT + 1))
    with pytest.raises(trials.TrialError, match="byte budget"):
        runner.native_results(1)


def trial_session(tmp_path: Path, *, capture_seconds: int = 60) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    (run / "capture").mkdir()
    lanes = [
        {"name": name, "identity": identity(), "seed": None if name == "original" else "7"}
        for name in trials.LANES
    ]
    catalog = trials.canonical({"lanes": lanes})
    (run / "lane-catalog.json").write_bytes(catalog)
    session = {
        "state": "interactive",
        "controller_profile": trials.PROFILE,
        "requested_game_ticks_per_wall_second": 20,
        "playback_speed": "normal",
        "diagnostic_tools": True,
        "comparison_lanes": True,
        "control_seed": "7",
        "lane_catalog_sha256": hashlib.sha256(catalog).hexdigest(),
    }
    (run / "session.json").write_text(json.dumps(session))
    (run / "capture/frames.ndjson").write_text(
        json.dumps(
            {
                "request_utc_epoch_ms": int(_clock().time() * 1000),
                "request_wall_elapsed_ns": 0,
                "requested_duration_seconds": capture_seconds,
            }
        )
        + "\n"
    )
    for name in ("diagnostic-commands.ndjson", "diagnostic-client-actions.ndjson"):
        (run / name).touch()
    return run


@pytest.mark.parametrize(
    ("elapsed_ns", "duration", "allowance"),
    (
        (0, 720, 570.0),
        (30_000_000_000, 720, 550.0),
        (30_125_000_000, 720, 549.875),
        (80_000_000_000, 720, 500.0),
        (0, 600, 460.0),
    ),
)
def test_actual_initialization_preserves_operator_reserve_after_delayed_first_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    elapsed_ns: int,
    duration: int,
    allowance: float,
) -> None:
    now, started = 1_788_600_000.0, 750.0
    monkeypatch.setattr(_clock(), "time", lambda: now)
    monkeypatch.setattr(_clock(), "monotonic", lambda: started)
    run = trial_session(tmp_path, capture_seconds=duration)
    # Same timing keys and request status as LiveFrameCapture's first producer receipt.
    capture = {
        "schema": "neurocraft.live_frame_capture.v1",
        "diagnostic_sampling": True,
        "status": "requested",
        "frame": 1,
        "file": "screenshots/frame-0001.png",
        "requested_fps": 15,
        "schedule_kind": "fixed_slots_half_period_v1",
        "schedule_slot": 0,
        "missed_slots_before_request": 0,
        "request_game_tick": 247,
        "request_utc_epoch_ms": int((now - 30.0) * 1000),
        "request_wall_elapsed_ns": elapsed_ns,
        "requested_duration_seconds": duration,
    }
    (run / "capture/frames.ndjson").write_bytes(trials.canonical(capture) + b"\n")
    runner = trials.Runner(run, tmp_path / "output", trials.schedule())
    try:
        assert runner.started == started and runner.deadline == pytest.approx(started + allowance)
        assert runner.deadline - runner.started <= trials.TOTAL_SECONDS == 570.0
        operator_end = (now - 30) - elapsed_ns / 1e9 + duration
        assert operator_end - (now + runner.deadline - started) >= trials.OPERATOR_RESERVE_SECONDS
        assert trials.OPERATOR_RESERVE_SECONDS == 110.0 and trials.RESTORE_RESERVE_SECONDS == 25.0
        if elapsed_ns == 30_000_000_000:
            # The former 570s deadline left only90s under V6's origin and guaranteed refusal.
            assert operator_end - (now + 570) == 90.0
            assert operator_end - (now + allowance) == 110.0
        assert (run / "diagnostic-commands.ndjson").read_bytes() == b""
        assert (run / "diagnostic-client-actions.ndjson").read_bytes() == b""
    finally:
        for resource in reversed(runner.resources):
            resource.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (field, value)
        for field in (
            "request_wall_elapsed_ns",
            "request_utc_epoch_ms",
            "requested_duration_seconds",
        )
        for value in (None, True, -1, "30", 1.0, float("nan"), float("inf"), 2**63)
    ]
    + [
        ("request_wall_elapsed_ns", 2_000_000_000_000_000_000),
        ("requested_duration_seconds", 0),
        ("requested_duration_seconds", 721),
    ],
)
def test_invalid_capture_timing_retains_failed_schedule_without_inbox_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
) -> None:
    monkeypatch.setattr(_clock(), "time", lambda: 1_788_600_000.0)
    run = trial_session(tmp_path, capture_seconds=720)
    path = run / "capture/frames.ndjson"
    capture = json.loads(path.read_bytes())
    if value is None:
        del capture[field]
    else:
        capture[field] = value
    # Permit intentionally malformed nonfinite JSON numbers in this rejection fixture.
    path.write_text(json.dumps(capture) + "\n")
    console = run / "diagnostic-commands.ndjson"
    native = run / "diagnostic-client-actions.ndjson"
    console.write_bytes(b'{"command":"retained prior record"}\n')
    native.write_bytes(b'{"seq":0}\n')
    before = {path: path.read_bytes() for path in (console, native)}
    output = tmp_path / "output"
    with pytest.raises(trials.TrialError, match="invalid capture"):
        trials.Runner(run, output, trials.schedule())
    failure = json.loads((output / "preflight_failure.json").read_bytes())
    assert len(failure["cases"]) == 30
    assert {case["status"] for case in failure["cases"]} == {"unattempted"}
    assert (output / "ledger.ndjson").read_bytes() == b""
    for inbox_path, content in before.items():
        assert inbox_path.read_bytes() == content
        trials.AppendInbox(inbox_path, 1000).close()


def test_delayed_first_frame_can_refuse_initialization_without_consuming_reserved_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_clock(), "time", lambda: 1_788_600_000.0)
    run = trial_session(tmp_path, capture_seconds=720)
    path = run / "capture/frames.ndjson"
    capture = json.loads(path.read_bytes())
    capture["request_wall_elapsed_ns"] = 570_000_000_000
    path.write_bytes(trials.canonical(capture) + b"\n")
    with pytest.raises(trials.TrialError, match="insufficient capture time"):
        trials.Runner(run, tmp_path / "output", trials.schedule())
    assert (run / "diagnostic-commands.ndjson").read_bytes() == b""
    assert (run / "diagnostic-client-actions.ndjson").read_bytes() == b""


@pytest.mark.parametrize("targeting", trials.NATIVE_TARGETING)
def test_capture_preflight_preserves_unattempted_schedule_and_releases_locks(
    tmp_path: Path,
    targeting: str,
) -> None:
    run = trial_session(tmp_path)
    output = tmp_path / "output"
    with pytest.raises(trials.TrialError, match="capture time"):
        trials.Runner(run, output, trials.schedule(), native_targeting=targeting)
    result = json.loads((output / "preflight_failure.json").read_text())
    assert result["native_targeting"] == targeting
    assert len(result["cases"]) == 30
    assert {case["status"] for case in result["cases"]} == {"unattempted"}
    for name in ("diagnostic-commands.ndjson", "diagnostic-client-actions.ndjson"):
        assert (run / name).read_bytes() == b""
        inbox = trials.AppendInbox(run / name, 100)
        inbox.close()
    with pytest.raises(FileExistsError):
        trials.Runner(run, output, trials.schedule(), native_targeting=targeting)
    assert json.loads((output / "preflight_failure.json").read_text()) == result


@pytest.mark.parametrize("targeting", trials.NATIVE_TARGETING)
@pytest.mark.parametrize("locomotion", trials.PLAYER_LOCOMOTION)
@pytest.mark.parametrize("budget_available", (False, True))
def test_targeting_plan_summary_and_request_budget_preserve_declared_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    targeting: str,
    locomotion: str,
    budget_available: bool,
) -> None:
    run = trial_session(tmp_path, capture_seconds=720)
    output = tmp_path / "output"
    count = (
        trials.NATIVE_REQUESTS
        - (10 if locomotion == "scripted" else 15)
        + int(not budget_available)
    )
    native_path = run / "diagnostic-client-actions.ndjson"
    previous = b"".join(trials.canonical({"seq": seq}) + b"\n" for seq in range(count))
    native_path.write_bytes(previous)
    kwargs = {
        "native_targeting": targeting,
        "player_locomotion": locomotion,
    }
    if not budget_available:
        with pytest.raises(trials.TrialError, match="native request budget"):
            trials.Runner(run, output, trials.schedule(), **kwargs)
        assert (
            json.loads((output / "preflight_failure.json").read_bytes())["player_locomotion"]
            == locomotion
        )
        assert native_path.read_bytes() == previous
        return
    runner = trials.Runner(run, output, trials.schedule(), **kwargs)
    plan = json.loads((output / "ledger.ndjson").read_bytes().splitlines()[0])
    assert plan["kind"] == "plan" and len(plan["schedule"]) == 30
    assert (
        plan["native_targeting"] == targeting
        and plan["player_locomotion"] == kwargs["player_locomotion"]
    )
    assert runner.targeting_scope() in plan["scope"]
    assert runner.locomotion_scope() in plan["scope"]
    assert ("synthetic native target-tracking" in plan["scope"]) is (
        locomotion in ("native_tracking_walk", "native_approach_walk")
    )
    if locomotion == "native_approach_walk":
        assert "maximum 20 callbacks, 0.65-block horizontal stop-request radius" in plan["scope"]
        assert "native coasting remains" in plan["scope"]
        assert "preparation forward 4.0 and signed lateral +1.2 blocks" in plan["scope"]
    assert ("synthetic client aiming" in plan["scope"]) is (targeting == "aim_at_dispatch")

    def stop_before_live() -> None:
        raise trials.TrialError("test stop before any live work")

    monkeypatch.setattr(runner, "observe_current_epoch", stop_before_live)
    summary = runner.execute()
    assert json.loads((output / "summary.json").read_bytes()) == summary
    assert summary["native_targeting"] == targeting
    assert summary["player_locomotion"] == kwargs["player_locomotion"]
    assert runner.targeting_scope() in summary["scope"]
    assert runner.locomotion_scope() in summary["scope"]
    assert {case["status"] for case in summary["cases"]} == {"unattempted"}
    assert not summary["original_restored"]
    assert native_path.read_bytes() == previous
    assert (run / "diagnostic-commands.ndjson").read_bytes() == b""


@pytest.mark.parametrize("targeting", (None, "strict", "aim_at_dispatch"))
@pytest.mark.parametrize("locomotion", (None, *trials.PLAYER_LOCOMOTION))
def test_native_targeting_cli_is_explicit_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    targeting: str | None,
    locomotion: str | None,
) -> None:
    calls: list[dict[str, Any]] = []

    def runner(run: Path, output: Path, cases: tuple[trials.Case, ...], **kwargs: Any) -> Any:
        calls.append(kwargs)
        assert len(cases) == 30
        return SimpleNamespace(
            execute=lambda: {"all_cases_observed": False, "original_restored": False}
        )

    monkeypatch.setattr(trials, "Runner", runner)
    args = [str(tmp_path), "--output", str(tmp_path / "output")]
    if targeting is not None:
        args.extend(("--native-targeting", targeting))
    if locomotion is not None:
        args.extend(("--player-locomotion", locomotion))
    assert trials.main(args) == 1
    assert calls == [
        {"native_targeting": targeting or "strict", "player_locomotion": locomotion or "scripted"}
    ]


def test_unknown_targeting_refused_before_creating_output(tmp_path: Path) -> None:
    with pytest.raises(trials.TrialError, match="unknown native targeting"):
        trials.Runner(
            tmp_path, tmp_path / "output", trials.schedule(), native_targeting="guess_hit"
        )
    assert not (tmp_path / "output").exists()
