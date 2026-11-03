#!/usr/bin/env python3
"""Bounded V2 world fixtures and diagnostic receipt reduction, never a neural controller.

Run only while exclusively owning the fresh diagnostic console. Player approaches default to
scripted displacement; native_walk opts into synthetic forward-key gameplay.
native_tracking_walk additionally aims at the fresh target during the native key hold.
native_approach_walk stops tracking walking at its declared distance or callback cap.
Brush/attack targeting is strict by default; aim_at_dispatch opts into synthetic client aiming.
Native dispatch is not proof of a server event. Source observations and applied responses
are reported separately.
Rows are bound to their immutable epoch directory; this is not a hash-linked causal trace.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from neurocraft_protocol import RunIdentity, SensorFrameV4

PROFILE = "full_mcns_interactions_program_v2"
SCHEMA = "neurocraft.interaction_trials.v1"
SOURCES = ("player_approach", "mob_approach", "brush", "attack", "food", "light")
LANES = ("original", "shuffled_weight", "no_input")
NATIVE_TARGETING = ("strict", "aim_at_dispatch")
AIM_ACTIONS = {"brush": "aim_and_brush", "attack": "aim_and_attack"}
PLAYER_LOCOMOTION = ("scripted", "native_walk", "native_tracking_walk", "native_approach_walk")
WALK_ACTIONS = {
    "native_walk": "walk_forward",
    "native_tracking_walk": "walk_track_target",
    "native_approach_walk": "walk_approach_target",
}
TRACKING_ACTIONS = ("walk_track_target", "walk_approach_target")
APPROACH_RELEASE_SCOPE = "forward key release; native momentum and subsequent coasting remain"
TRACKING_RULE = (
    "lookAt(EYES,target bounding-box center) before forward key-down and each "
    "continuing held callback; no aim on finish or abort"
)
PROGRAMS = {
    "player_approach": ("ESCAPE_JUMP",),
    "mob_approach": ("BANK",),
    "brush": ("GROOM",),
    "attack": ("FLEE",),
    "food": ("LAND", "FEED"),
    "light": ("HOVER",),
}
FLY = "@e[type=neurocraft_fly:fly,limit=1]"
PLAYER = "NeuroFlyObserver"
COW = "@e[type=minecraft:cow,tag=nf_v2_cow,limit=1]"
CONSOLE_LIMIT = 65_536
NATIVE_LIMIT = 16_384
NATIVE_REQUESTS = 32
NATIVE_DISPATCH_SECONDS = 1.0
NATIVE_RESULTS_LIMIT = 65_536
BASELINE_OBSERVATIONS = 10
NATIVE_REAR_MAX_DISTANCE = 2.5
CASE_SECONDS = 35.0
TOTAL_SECONDS = 570.0
OPERATOR_RESERVE_SECONDS = 110.0
RESTORE_RESERVE_SECONDS = 25.0
MIN_CASE_ALLOWANCE_SECONDS = 22.0
MAX_CASE_ROWS = 2_000
MAX_LEDGER_BYTES = 128 * 1024 * 1024


class TrialError(RuntimeError):
    """An explicit diagnostic boundary; never silently restart or retry."""


@dataclass(frozen=True)
class Case:
    index: int
    lane: str
    repetition: int
    source: str


def schedule(source: str | None = None, lane: str | None = None) -> tuple[Case, ...]:
    if source is not None and source not in SOURCES:
        raise TrialError("unknown source")
    if lane is not None and lane not in LANES:
        raise TrialError("unknown lane")
    # A one-source/one-lane invocation is a single exploratory diagnostic, not the fixed30.
    entries = [
        (name, repetition, stimulus)
        for name in LANES
        if lane is None or name == lane
        for repetition in range(
            1 if source is not None and lane is not None else 3 if name == "original" else 1
        )
        for stimulus in SOURCES
        if source is None or source == stimulus
    ]
    return tuple(Case(index, *entry) for index, entry in enumerate(entries))


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def integer(value: Any, name: str) -> int:
    if type(value) is not int or not 0 <= value < 2**63:
        raise TrialError(f"invalid {name}")
    return value


def require_session(value: Mapping[str, Any]) -> None:
    if (
        value.get("state") != "interactive"
        or value.get("controller_profile") != PROFILE
        or value.get("requested_game_ticks_per_wall_second") != 20
        or value.get("playback_speed") != "normal"
        or value.get("diagnostic_tools") is not True
        or value.get("comparison_lanes") is not True
        or value.get("control_seed") != "7"
    ):
        raise TrialError("requires live V2 comparison session, seed7, diagnostics and normal20TPS")


def source_ticks(source: str, row: Mapping[str, Any], start: int, end: int) -> list[int]:
    """Observed stimulus separately from the policy source or its issued program."""
    tick = integer(row["observation_tick"], "observation tick")
    if not start <= tick <= end or row.get("policy_state_bound") is not True:
        return []
    before, after = row["policy_state_before"], row["policy_state_after"]
    if after["last_game_tick"] != tick or before["last_game_tick"] >= tick:
        raise TrialError("policy observation clocks disagree")
    previous = before.get("previous_sensors")
    if previous is None:
        return []
    old = SensorFrameV4.from_dict(previous)
    current = SensorFrameV4.from_dict(after["previous_sensors"])
    if source in ("attack", "brush", "food"):
        key = {"attack": "attack_attempt", "brush": "brush", "food": "food"}[source]
        prior_counter, counter = previous["events"][key], after["previous_sensors"]["events"][key]
        event_tick = counter["last_game_tick"]
        if counter["count"] < prior_counter["count"]:
            raise TrialError("event counter regressed in one epoch")
        return (
            [event_tick]
            if (
                counter["count"] > prior_counter["count"]
                and type(event_tick) is int
                and start <= event_tick <= min(end, tick)
                and (source != "food" or current.events.food_near)
            )
            else []
        )
    if source in ("player_approach", "mob_approach"):
        actor = 1 if source == "player_approach" else 2
        return (
            [tick]
            if any(
                old.ray_actor_kinds[i] == current.ray_actor_kinds[i] == actor
                and current.ray_distances_m[i] < old.ray_distances_m[i]
                for i in (0, 2, 3)
            )
            else []
        )
    return [tick] if current.local_light != old.local_light else []


def hold_interval(
    source: str,
    issued: Mapping[int, Mapping[str, Any]],
    selected: Sequence[int],
    applied: Sequence[int],
) -> tuple[bool, bool]:
    """Return observed expiry and complete application, never infer omitted hold rows.

    A complete interval needs the actual entry, every held observation, and the
    exact exclusive expiry boundary. State and command clocks must connect across
    that whole interval. LAND→FEED may replace the deadline but keeps its event.
    An observed later expiry alone cannot establish any missing application.
    """
    if not selected:
        return False, False
    first = issued[selected[0]]
    event = first["policy_state_after"]["held_event"]

    def same_episode(state: Mapping[str, Any]) -> bool:
        return state.get("source") == source and canonical(state.get("held_event")) == canonical(
            event
        )

    expiries = [
        row
        for row in issued.values()
        if row.get("policy_state_bound") is True
        and same_episode(row.get("policy_state_before") or {})
        and (row.get("policy_state_after") or {}).get("source") is None
        and row["observation_tick"]
        >= (row.get("policy_state_before") or {}).get("active_until", 2**63)
    ]
    finished = bool(expiries)
    if (
        len(expiries) != 1
        or list(selected) != list(applied)
        or (first.get("policy_state_before") or {}).get("source") == source
    ):
        return finished, False
    expiry = expiries[0]
    rows = [issued[seq] for seq in selected]
    for row in rows:
        before, after = row["policy_state_before"], row["policy_state_after"]
        tick = row["observation_tick"]
        if (
            not same_episode(after)
            or after.get("program") != row["program"]
            or after.get("last_game_tick") != tick
            or before.get("last_game_tick") != tick - 1
            or after.get("active_until", 0) <= tick
            or after.get("hold_remaining") != after["active_until"] - tick
        ):
            return finished, False
    before, after = expiry["policy_state_before"], expiry["policy_state_after"]
    if (
        expiry["observation_tick"] != before["active_until"]
        or after.get("last_game_tick") != expiry["observation_tick"]
        or after.get("active_until") != before["active_until"]
        or after.get("held_event") is not None
        or after.get("hold_remaining") != 0
        or after.get("program") != "CRUISE"
        or expiry["program"] != "CRUISE"
    ):
        return finished, False
    for previous, current in pairwise([*rows, expiry]):
        prior_state, current_state = previous["policy_state_after"], current["policy_state_after"]
        feeding_transition = (
            source == "food"
            and prior_state["program"] == "LAND"
            and current_state["program"] == "FEED"
        )
        if (
            current["observation_tick"] != previous["observation_tick"] + 1
            or current["command_seq"] != previous["command_seq"] + 1
            or current["observation_seq"] != previous["observation_seq"] + 1
            or current["neural_step_start"] != previous["neural_step_end"]
            or canonical(current["policy_state_before"]) != canonical(prior_state)
            or (
                current is not expiry
                and current_state["program"] != prior_state["program"]
                and not feeding_transition
            )
            or (
                current_state["active_until"] != prior_state["active_until"]
                and not feeding_transition
            )
        ):
            return finished, False
    return finished, source != "food" or any(row["program"] == "FEED" for row in rows)


def reduce_case(
    case: Case,
    identity: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    start_tick: int,
    end_tick: int,
) -> dict[str, Any]:
    """Exact joins of available diagnostic fields; every row retains epoch-directory identity."""
    RunIdentity.from_dict(identity)
    issued: dict[int, Mapping[str, Any]] = {}
    receipts: dict[int, Mapping[str, Any]] = {}
    observed: set[int] = set()
    for bound in records:
        if bound.get("epoch_identity") != identity:
            raise TrialError("row belongs to a different epoch identity")
        row = bound["row"]
        if (
            row.get("schema") != "neurocraft.program_decision.v1"
            or row.get("controller_profile") != PROFILE
            or row.get("run_id") != identity["run_id"]
            or row.get("diagnostic_only") is not True
        ):
            raise TrialError("foreign diagnostic row")
        seq = integer(row["command_seq"], "command sequence")
        dest = (
            issued
            if row.get("kind") == "issued"
            else receipts
            if row.get("kind") == "receipt"
            else None
        )
        if dest is None or seq in dest:
            raise TrialError("duplicate or unsupported diagnostic row")
        dest[seq] = row
        if row["kind"] == "issued":
            if row["target_tick"] != row["observation_tick"] + 2:
                raise TrialError("issued command violates normal two-tick window")
            if row["neural_step_end"] - row["neural_step_start"] != 5:
                raise TrialError("issued neural interval differs from five updates")
            observed.update(source_ticks(case.source, row, start_tick, end_tick))
        elif row.get("fault_code") is not None:
            raise TrialError("terminal controller fault receipt")
    matching: dict[int, Mapping[str, Any]] = {}
    for seq in issued.keys() & receipts.keys():
        request, receipt = issued[seq], receipts[seq]
        if (
            any(request[key] != receipt[key] for key in ("observation_seq", "target_tick"))
            or request["program"] != receipt["requested_program"]
        ):
            raise TrialError("receipt does not bind its issued command")
        if receipt["validation"] == "PROGRAM_APPLIED":
            if (
                receipt["applied_tick"] != request["target_tick"]
                or receipt["executed_program"] != request["program"]
                or receipt.get("reason") is not None
            ):
                raise TrialError("applied receipt clock/program contradicts its request")
            matching[seq] = receipt
    selected: list[int] = []
    for seq, row in issued.items():
        after = row.get("policy_state_after") or {}
        held = after.get("held_event") or {}
        if (
            row.get("policy_state_bound") is True
            and after.get("source") == case.source
            and held.get("event_tick") in observed
            and row["program"] in PROGRAMS[case.source]
        ):
            selected.append(seq)
    selected.sort()
    applied = [seq for seq in selected if seq in matching]
    finished, complete = hold_interval(case.source, issued, selected, applied)
    programs = {
        program: sum(issued[seq]["program"] == program for seq in applied)
        for program in PROGRAMS[case.source]
    }
    window = {
        seq: row for seq, row in issued.items() if start_tick <= row["observation_tick"] <= end_tick
    }
    return {
        "response_window_issued": len(window),
        "response_window_missing_receipts": sorted(window.keys() - receipts.keys()),
        "response_window_unapplied": sorted(window.keys() - matching.keys()),
        "source_verified": bool(observed),
        "observed_source_ticks": sorted(observed),
        "issued_response": bool(selected),
        "applied_response": bool(applied),
        "issued_sequences": selected,
        "applied_sequences": applied,
        "issued_without_applied_receipt": [seq for seq in selected if seq not in matching],
        "applied_program_counts": programs,
        "complete_applied_hold": complete,
        "episode_expired": finished,
        "outcome": "source_miss"
        if not observed
        else "response_applied"
        if applied
        else "no_applied_response",
        "comparison_rule": "no-input silence and shuffled retained responses are valid outcomes",
        "identity_scope": "immutable epoch directory; exact available diagnostic receipt keys",
    }


def preparation_commands(case: Case, *, native_walk: bool = False) -> tuple[str, ...]:
    height = 66.0 if case.source == "food" or native_walk else 66.5
    commands = [
        f"gamemode spectator {PLAYER}",
        "kill @e[tag=nf_v2_food]",
        "kill @e[tag=nf_v2_cow]",
        f"tp {FLY} 0 {height} 0 0 0",
        f"data merge entity {FLY} {{Motion:[0d,0d,0d]}}",
    ]
    if case.source in ("brush", "attack"):
        item = "minecraft:brush" if case.source == "brush" else "minecraft:air"
        commands.append(f"item replace entity {PLAYER} weapon.mainhand with {item}")
    elif case.source == "player_approach":
        commands.append(f"tp {PLAYER} 0 65 5.5 180 0")
    else:
        commands.append(f"tp {PLAYER} -1 65 -2 -25 15")
    return tuple(commands)


def lighting(bright: bool) -> str:
    return (
        "fill -24 65 -24 24 70 24 minecraft:light[level=15] replace minecraft:air"
        if bright
        else "fill -24 65 -24 24 70 24 minecraft:air replace minecraft:light"
    )


def camera_rotation(*, offset: bool) -> str:
    """Rotation-only framing request; neither player displacement nor visual evidence."""
    return (
        f"execute as {PLAYER} at @s anchored eyes facing entity {FLY} eyes "
        + ("rotated ~-20 ~ " if offset else "")
        + "run rotate @s ~ ~"
    )


def native_observer_position() -> str:
    return f"execute at {FLY} rotated ~ 0 run tp {PLAYER} ^0 ^-1.0 ^-1.0 ~ 0"


def walking_observer_position(*, tracking: bool = False, approach: bool = False) -> str:
    # Preparation only: stand on the authored y=65 floor, outside the forward ray.
    # Off-axis placement avoids having the cruising fly approach a stationary
    # on-ray player before the native key hold. Its success still needs observation.
    # The first tracking run caught the fly during its escape crouch. The farther
    # preparation is a new candidate; only actual dispatch geometry can qualify it.
    if tracking and approach:
        raise TrialError("walking preparation modes cannot be combined")
    forward = "5.5" if tracking else "4.0"
    # The .6 client sweep omitted the live baseline dwell and triggered an escape
    # during preparation. Restore the earlier live-tested side/spacing for walk-up.
    lateral = "1.2"
    return (
        f"execute at {FLY} rotated ~ 0 positioned ^{lateral} ^0 ^{forward} "
        f"run tp {PLAYER} ~ 65 ~ ~180 0"
    )


def walking_heading() -> str:
    """Unmeasured preparation-only crossing candidate: fly yaw +90, level pitch."""
    return f"execute at {FLY} rotated ~90 0 run rotate {PLAYER} ~ ~"


def food_stimulus() -> str:
    return (
        f"execute at {FLY} rotated ~ 0 run summon minecraft:item ^0 ^-0.8 ^0.7 "
        '{Item:{id:"minecraft:sugar",count:1},Motion:[0d,0d,0d],'
        'PickupDelay:32767,Tags:["nf_v2_food"]}'
    )


def native_geometry(row: Mapping[str, Any]) -> bool:
    sensors = SensorFrameV4.from_dict(row["policy_state_after"]["previous_sensors"])
    # FlySensors.sample orders forward/back/left/right/up/down; its Player code is 1.
    # This rear geometry check is a fixture prerequisite, not a neural input change.
    return (
        sensors.ray_actor_kinds[1] == 1
        and sensors.ray_hit_kinds[1] == 2
        and 0.0 < sensors.ray_distances_m[1] <= NATIVE_REAR_MAX_DISTANCE
    )


class AppendInbox:
    """Advisory ownership plus exact external-append detection; never rewrite consumed bytes."""

    def __init__(self, path: Path, limit: int):
        self.fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW)
        self.closed = False
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.size = os.fstat(self.fd).st_size
            self.path, self.limit = path, limit
            if self.size > limit:
                raise TrialError("inbox already exceeds bound")
            self.initial = path.read_bytes()
            self.contents = self.initial
            if self.initial and not self.initial.endswith(b"\n"):
                raise TrialError("inbox has incomplete external request")
        except BaseException:
            os.close(self.fd)
            raise

    def append(self, value: Mapping[str, Any]) -> None:
        if self.closed:
            raise TrialError("diagnostic inbox is closed")
        data = canonical(value) + b"\n"
        if (
            os.fstat(self.fd).st_size != self.size
            or self.path.stat().st_ino != os.fstat(self.fd).st_ino
            or self.path.read_bytes() != self.contents
        ):
            raise TrialError("another console/native writer changed the inbox")
        if self.size + len(data) > self.limit:
            raise TrialError("inbox budget exhausted")
        if os.write(self.fd, data) != len(data):
            raise TrialError("partial diagnostic request write; never retry")
        self.size += len(data)
        self.contents += data

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            os.close(self.fd)


class Epoch:
    def __init__(self, path: Path, expected: Mapping[str, Any], session: Mapping[str, Any]):
        self.path = path
        self.raw = (path / "epoch.json").read_bytes()
        self.metadata = json.loads(self.raw)
        self.identity = self.metadata["identity"]
        RunIdentity.from_dict(self.identity)
        if (
            self.metadata.get("controller_profile") != PROFILE
            or self.metadata["name"] != expected["name"]
        ):
            raise TrialError("unexpected epoch profile/lane")
        for key, value in expected["identity"].items():
            if key != "session_epoch" and self.identity.get(key) != value:
                raise TrialError("epoch differs from trusted lane catalog")
        if (
            str(self.identity["session_epoch"]) != path.name
            or self.identity["run_id"] != session["run_id"]
            or self.identity["entity_id"] != session["entity_id"]
            or any(
                self.metadata.get(key) != expected.get(key)
                for key in ("seed", "control_kind", "control_sha256")
            )
        ):
            raise TrialError("epoch identity/control metadata mismatch")
        self.offset, self.pending = 0, b""
        decisions = self.path / "program-decisions.ndjson"
        self.attachment_end = decisions.stat().st_size if decisions.exists() else 0
        if self.attachment_end > MAX_LEDGER_BYTES:
            raise TrialError("existing epoch diagnostics exceed attachment byte budget")
        self.issued: dict[int, dict[str, Any]] = {}
        self.receipts: dict[int, dict[str, Any]] = {}
        self.last_new = time.monotonic()
        self.observer: Callable[[dict[str, Any]], None] | None = None

    def poll(self) -> None:
        if (self.path / "epoch.json").read_bytes() != self.raw:
            raise TrialError("epoch metadata changed")
        path = self.path / "program-decisions.ndjson"
        if not path.exists():
            return
        if path.stat().st_size < self.offset:
            raise TrialError("epoch decisions truncated")
        with path.open("rb") as stream:
            stream.seek(self.offset)
            data = stream.read(524_288)
        self.offset += len(data)
        line_end = self.offset - len(data) - len(self.pending)
        lines = (self.pending + data).split(b"\n")
        self.pending = lines.pop()
        if len(self.pending) > 32_768:
            raise TrialError("oversize/incomplete diagnostic row")
        for line in lines:
            line_end += len(line) + 1
            row = json.loads(line)
            if row.get("fault_code") is not None:
                raise TrialError("terminal controller fault")
            if row.get("kind") not in ("issued", "receipt"):
                raise TrialError("unexpected epoch decision kind")
            target = self.issued if row["kind"] == "issued" else self.receipts
            seq = integer(row["command_seq"], "command sequence")
            if seq in target:
                raise TrialError("duplicate diagnostic sequence")
            target[seq] = row
            if len(target) > MAX_CASE_ROWS:
                del target[min(target)]
            if line_end > self.attachment_end:
                self.last_new = time.monotonic()
            if self.observer is not None:
                self.observer({"epoch_identity": self.identity, "row": row})

    def latest(self) -> dict[str, Any] | None:
        self.poll()
        common = self.issued.keys() & self.receipts.keys()
        if not common:
            return None
        seq = max(common)
        row, receipt = self.issued[seq], self.receipts[seq]
        return row if exact_applied(row, receipt) else None


def exact_applied(row: Mapping[str, Any], receipt: Mapping[str, Any]) -> bool:
    return (
        receipt.get("validation") == "PROGRAM_APPLIED"
        and receipt.get("applied_tick") == row.get("target_tick")
        and all(
            row.get(key) == receipt.get(key)
            for key in ("command_seq", "observation_seq", "target_tick")
        )
        and row.get("program")
        == receipt.get("requested_program")
        == receipt.get("executed_program")
        and receipt.get("reason") is None
        and receipt.get("fault_code") is None
    )


def eligible(row: Mapping[str, Any] | None) -> bool:
    if row is None or row.get("policy_state_bound") is not True:
        return False
    state = row["policy_state_after"]
    pending, approaching = state.get("pending"), state.get("approaching")
    return (
        state.get("source") is None
        and state.get("program") == "CRUISE"
        and state.get("hold_remaining") == 0
        and len(state.get("history", ())) == 4
        and row["observation_tick"] >= state.get("cooldown_until", 2**63)
        and isinstance(pending, dict)
        and set(pending) == set(SOURCES)
        and all(value is None for value in pending.values())
        and isinstance(approaching, list)
        and len(approaching) == 2
        and all(value is False for value in approaching)
    )


def stable_preparation(epoch: Epoch, after_seq: int, expected_light: int) -> dict[str, Any] | None:
    """Ten distinct consecutive matched observations, not ten polls of the latest row."""
    common = epoch.issued.keys() & epoch.receipts.keys()
    if not common:
        return None
    last = max(common)
    first = last - BASELINE_OBSERVATIONS + 1
    if first <= after_seq or not all(seq in common for seq in range(first, last + 1)):
        return None
    rows = [epoch.issued[seq] for seq in range(first, last + 1)]
    events: bytes | None = None
    # Unmatched newer issued rows cannot contradict the matched baseline we accept.
    tail = [epoch.issued[seq] for seq in sorted(epoch.issued) if seq > last]
    for row in [*rows, *tail]:
        if not eligible(row):
            return None
        state = row["policy_state_after"]
        sensors = SensorFrameV4.from_dict(state["previous_sensors"])
        current = canonical(sensors.events.to_dict())
        if (
            sensors.local_light != expected_light
            or (events is not None and events != current)
            or state["last_game_tick"] != row["observation_tick"]
            or row["target_tick"] != row["observation_tick"] + 2
            or row["neural_step_end"] - row["neural_step_start"] != 5
        ):
            return None
        events = current
    if not all(exact_applied(row, epoch.receipts[row["command_seq"]]) for row in rows):
        return None
    if any(
        current["observation_tick"] != previous["observation_tick"] + 1
        or current["observation_seq"] != previous["observation_seq"] + 1
        or current["neural_step_start"] != previous["neural_step_end"]
        for previous, current in pairwise(rows)
    ):
        return None
    return rows[-1]


def walking_admission_refusal(epoch: Epoch, request_tick: int, expected_light: int) -> str | None:
    """Recheck the original preparation after the read-only position query; never wait/retry."""
    latest = stable_preparation(epoch, -1, expected_light)
    if latest is None:
        return "no_stable_preparation_after_position_query"
    if latest["observation_tick"] < request_tick:
        return "preparation_older_than_native_request"
    baseline = [row for row in epoch.issued.values() if row["observation_tick"] == request_tick]
    if (
        len(baseline) != 1
        or not eligible(baseline[0])
        or baseline[0]["command_seq"] not in epoch.receipts
        or not exact_applied(baseline[0], epoch.receipts[baseline[0]["command_seq"]])
    ):
        return "native_request_baseline_unavailable"
    first = baseline[0]
    events = canonical(first["policy_state_after"]["previous_sensors"]["events"])
    # A source may have appeared and cleared while the console query was in flight.
    # A later neutral streak must not erase that contradiction to this request's baseline.
    for seq, row in epoch.issued.items():
        if seq < first["command_seq"]:
            continue
        if not eligible(row):
            return "preparation_changed_during_position_query"
        sensors = SensorFrameV4.from_dict(row["policy_state_after"]["previous_sensors"])
        if sensors.local_light != expected_light or canonical(sensors.events.to_dict()) != events:
            return "preparation_changed_during_position_query"
    return None


def finite_number(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def finite_coordinates(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(finite_number(number) for number in value)
    )


def finite_view(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"yaw_degrees", "pitch_degrees"}
        and all(finite_number(number) for number in value.values())
    )


def aimed_dispatch_matches(request: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    """Validate the client aim/picker receipt; this is not a server event receipt."""
    aim = result.get("aim")
    if not isinstance(aim, dict) or set(aim) != {
        "before_client_view",
        "after_client_view",
        "eye_client_position_blocks",
        "target_client_position_blocks",
        "target_client_center_blocks",
        "picker_result",
    }:
        return False

    for key in ("before_client_view", "after_client_view"):
        if not finite_view(aim[key]):
            return False
    if not all(
        finite_coordinates(aim[key])
        for key in (
            "eye_client_position_blocks",
            "target_client_position_blocks",
            "target_client_center_blocks",
        )
    ):
        return False
    picker = aim["picker_result"]
    if (
        not isinstance(picker, dict)
        or set(picker) != {"type", "entity_uuid", "hit_position_blocks"}
        or picker["type"] != "ENTITY"
        or picker["entity_uuid"] != request["entity_uuid"]
        or not finite_coordinates(picker["hit_position_blocks"])
        or any(
            type(result.get(key)) is not int
            for key in ("seq", "request_game_tick", "valid_through_tick")
        )
    ):
        return False
    prefix = [
        "player.lookAt(EYES,target.getBoundingBox().getCenter())",
        "gameRenderer.pick(1.0F)",
    ]
    methods = result.get("attempted_methods")
    if request["action"] == "aim_and_brush":
        brush = [*prefix, "gameMode.interactAt(MAIN_HAND)"]
        return methods in (brush, [*brush, "gameMode.interact(MAIN_HAND)"])
    return request["action"] == "aim_and_attack" and methods == [
        *prefix,
        "gameMode.attack",
        "player.swing(MAIN_HAND)",
    ]


def native_dispatch_matches(request: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    tick = result.get("client_world_game_tick")
    return (
        result.get("schema") == "neurocraft.client_action_dispatch.v1"
        and result.get("status") == "DISPATCHED"
        and result.get("seq") == request["seq"]
        and result.get("action") == request["action"]
        and result.get("requested_entity_uuid") == request["entity_uuid"]
        and result.get("crosshair_entity_uuid") == request["entity_uuid"]
        and result.get("request_game_tick") == request["game_tick"]
        and result.get("valid_through_tick") == request["valid_through_tick"]
        and type(tick) is int
        and request["game_tick"] <= tick <= request["valid_through_tick"]
        and (
            request["action"] not in AIM_ACTIONS.values() or aimed_dispatch_matches(request, result)
        )
    )


def tracking_walk_matches(request: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> bool:
    """Bind the declared client tracking rule and its recorded endpoints, not an aim stream."""
    if request.get("action") not in TRACKING_ACTIONS or len(rows) != 2:
        return False
    approach = request["action"] == "walk_approach_target"
    initial: dict[str, Any] | None = None
    for index, row in enumerate(rows):
        if any(
            type(row.get(key)) is not int
            for key in ("seq", "request_game_tick", "valid_through_tick")
        ) or any(
            type(row.get(key)) is not str or not row[key] for key in ("player_uuid", "dimension")
        ):
            return False
        walk = row.get("walk")
        if not isinstance(walk, dict):
            return False
        tracking = walk.get("tracking")
        if (
            not isinstance(tracking, dict)
            or set(tracking)
            != {"mode", "rule", "aim_count", "initial_observation", "last_observation"}
            or tracking["mode"] != "direct_fresh_target_center"
            or tracking["rule"] != TRACKING_RULE
            or type(tracking["aim_count"]) is not int
            or tracking["aim_count"]
            != (1 if index == 0 else walk.get("observed_client_ticks") if approach else 20)
            or any(
                type(walk.get(key)) is not int
                for key in (
                    "max_client_ticks" if approach else "fixed_client_ticks",
                    "wall_timeout_ns",
                    "start_game_tick",
                )
            )
        ):
            return False
        for key in ("initial_observation", "last_observation"):
            observation = tracking[key]
            if (
                not isinstance(observation, dict)
                or set(observation)
                != {
                    "game_tick",
                    "before_client_view",
                    "after_client_view",
                    "eye_client_position_blocks",
                    "target_client_position_blocks",
                    "target_client_center_blocks",
                }
                or type(observation["game_tick"]) is not int
                or not request["game_tick"]
                <= observation["game_tick"]
                <= request["valid_through_tick"]
                or not all(
                    finite_view(observation[view])
                    for view in ("before_client_view", "after_client_view")
                )
                or not all(
                    finite_coordinates(observation[pose])
                    for pose in (
                        "eye_client_position_blocks",
                        "target_client_position_blocks",
                        "target_client_center_blocks",
                    )
                )
            ):
                return False
        first = tracking["initial_observation"]
        if first["game_tick"] != rows[0]["client_world_game_tick"]:
            return False
        if index == 0:
            initial = first
            if first != tracking["last_observation"]:
                return False
        elif initial != first:
            return False
        # Start pose is captured immediately before the first aim, and retained unchanged.
        for key in ("start_client_pose", "end_client_pose") if index else ("start_client_pose",):
            pose = walk.get(key)
            if (
                not isinstance(pose, dict)
                or set(pose)
                != {"position_blocks", "yaw_degrees", "pitch_degrees", "on_ground", "flying"}
                or not finite_coordinates(pose["position_blocks"])
                or pose["on_ground"] is not True
                or pose["flying"] is not False
                or not all(finite_number(pose[angle]) for angle in ("yaw_degrees", "pitch_degrees"))
            ):
                return False
        if first["before_client_view"] != {
            angle: walk["start_client_pose"][angle] for angle in ("yaw_degrees", "pitch_degrees")
        }:
            return False
    # The last aim is a continuing callback, not the terminal callback. Client game time
    # may correct backwards; no chronological ordering between these endpoints is assumed.
    return True


def approach_walk_matches(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Validate an explicitly bounded walk-up gesture, including why it released W."""
    first, last = rows[0]["walk"], rows[1]["walk"]
    for row in rows:
        walk = row["walk"]
        if (
            "fixed_client_ticks" in walk
            or type(walk.get("max_client_ticks")) is not int
            or walk["max_client_ticks"] != 20
            or not finite_number(walk.get("stop_horizontal_distance_blocks"))
            or walk["stop_horizontal_distance_blocks"] != 0.65
            or walk.get("release_scope") != APPROACH_RELEASE_SCOPE
        ):
            return False
        # Every aim precedes arrival. The terminal callback never aims.
        for key in ("initial_observation", "last_observation"):
            observation = walk["tracking"][key]
            eye, target = (
                observation["eye_client_position_blocks"],
                observation["target_client_center_blocks"],
            )
            if (
                math.dist((eye[0], eye[2]), (target[0], target[2])) <= 0.65
                or math.dist(eye, target) > 6.0
            ):
                return False
    if (
        any(
            key not in first or first[key] is not None
            for key in (
                "completion_reason",
                "terminal_geometry",
                "end_game_tick",
                "end_client_pose",
            )
        )
        or type(first.get("elapsed_ns")) is not int
        or first["elapsed_ns"] != 0
        or any(
            first["tracking"]["initial_observation"]["eye_client_position_blocks"][index]
            != first["start_client_pose"]["position_blocks"][index]
            for index in (0, 2)
        )
    ):
        return False
    geometry = last.get("terminal_geometry")
    if (
        not isinstance(geometry, dict)
        or set(geometry)
        != {
            "game_tick",
            "eye_client_position_blocks",
            "target_client_position_blocks",
            "target_client_center_blocks",
            "horizontal_distance_blocks",
        }
        or type(geometry["game_tick"]) is not int
        or geometry["game_tick"] != rows[1]["client_world_game_tick"]
        or not all(
            finite_coordinates(geometry[key])
            for key in (
                "eye_client_position_blocks",
                "target_client_position_blocks",
                "target_client_center_blocks",
            )
        )
        or not finite_number(geometry["horizontal_distance_blocks"])
    ):
        return False
    eye, target = geometry["eye_client_position_blocks"], geometry["target_client_center_blocks"]
    distance = geometry["horizontal_distance_blocks"]
    endpoint = last["end_client_pose"]["position_blocks"]
    if (
        not math.isclose(
            distance, math.dist((eye[0], eye[2]), (target[0], target[2])), abs_tol=1e-9
        )
        or distance <= 0.05
        or math.dist(eye, target) > 6.0
        or any(eye[index] != endpoint[index] for index in (0, 2))
        or any(
            not math.isclose(
                target[index], geometry["target_client_position_blocks"][index], abs_tol=1e-9
            )
            for index in (0, 2)
        )
    ):
        return False
    if distance <= 0.65:
        return (
            last.get("completion_reason") == "stop_distance_reached"
            and rows[1].get("reason") == "walking_stop_distance_reached"
        )
    return (
        last["observed_client_ticks"] == 20
        and last.get("completion_reason") == "callback_limit_reached"
        and rows[1].get("reason") == "walking_callback_limit_reached"
    )


def walking_dispatch_evidence(
    request: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """A completed native key hold is not server movement or a neural response."""
    tracking = request.get("action") in TRACKING_ACTIONS
    approach = request.get("action") == "walk_approach_target"
    valid = request.get("action") in WALK_ACTIONS.values() and len(rows) == 2
    for index, row in enumerate(rows):
        tick = row.get("client_world_game_tick")
        valid = valid and (
            row.get("schema") == "neurocraft.client_action_dispatch.v1"
            and row.get("status") == ("WALK_STARTED" if index == 0 else "WALK_COMPLETED")
            and all(row.get(key) == request[key] for key in ("seq", "action"))
            and row.get("requested_entity_uuid") == request["entity_uuid"]
            and row.get("request_game_tick") == request["game_tick"]
            and row.get("valid_through_tick") == request["valid_through_tick"]
            and type(tick) is int
            and request["game_tick"] <= tick <= request["valid_through_tick"]
        )
    if valid:
        start, end = rows
        first, last = start.get("walk"), end.get("walk")
        valid = isinstance(first, dict) and isinstance(last, dict)
        if valid:
            assert isinstance(first, dict) and isinstance(last, dict)
            valid = (
                all(first.get(k) == last.get(k) for k in ("start_game_tick", "start_client_pose"))
                and type(first.get("start_game_tick")) is int
                and first["start_game_tick"] == start["client_world_game_tick"]
                and type(last.get("end_game_tick")) is int
                and last["end_game_tick"] == end["client_world_game_tick"]
                and type(first.get("observed_client_ticks")) is int
                and first.get("observed_client_ticks") == 0
                and first.get("forward_key_released") is False
                and last.get("forward_key_released") is True
                and type(last.get("observed_client_ticks")) is int
                and (
                    1 <= last["observed_client_ticks"] <= 20
                    if approach
                    else last["observed_client_ticks"] == 20
                    and last.get("fixed_client_ticks") == first.get("fixed_client_ticks") == 20
                )
                and last.get("wall_timeout_ns") == first.get("wall_timeout_ns") == 3_000_000_000
                and type(last.get("elapsed_ns")) is int
                and 0 < last["elapsed_ns"] < 3_000_000_000
                and start.get("attempted_methods")
                == (
                    [
                        "player.lookAt(EYES,target.getBoundingBox().getCenter())",
                        "options.keyUp.setDown(true)",
                    ]
                    if tracking
                    else ["options.keyUp.setDown(true)"]
                )
                and end.get("attempted_methods") == ["options.keyUp.setDown(false)"]
                and all(
                    start.get(k) == end.get(k) and start.get(k)
                    for k in ("player_uuid", "dimension")
                )
            )
            valid = valid and (not tracking or tracking_walk_matches(request, rows))
            valid = valid and (not approach or approach_walk_matches(rows))
            for pose in (first.get("start_client_pose"), last.get("end_client_pose")):
                valid = valid and isinstance(pose, dict)
                if isinstance(pose, dict):
                    xyz = pose.get("position_blocks")
                    valid = valid and (
                        pose.get("on_ground") is True
                        and pose.get("flying") is False
                        and isinstance(xyz, list)
                        and len(xyz) == 3
                        and all(type(v) in (int, float) and math.isfinite(v) for v in xyz)
                    )
    return {"native_hold_completed": bool(valid)}


def walking_trial_verified(walk: Mapping[str, Any], source_ticks: Sequence[int]) -> bool:
    barrier = walk.get("server_start_barrier_tick")
    end_barrier = walk.get("server_end_barrier_tick")
    distance = walk.get("server_horizontal_displacement")
    return (
        walk.get("native_hold_completed") is True
        and type(barrier) is int
        and type(end_barrier) is int
        and bool(source_ticks)
        and barrier < min(source_ticks) <= end_barrier
        and isinstance(distance, (int, float))
        and not isinstance(distance, bool)
        and math.isfinite(distance)
        and distance > 0.5
    )


class Runner:
    reset_each_case = False
    comparison_scope = (
        "three original repetitions; shuffle7 and no-input once each; "
        "realized histories may differ; this does not replace historical fixed60"
    )
    player_locomotion = "scripted"
    native_targeting = "strict"

    def __init__(
        self,
        run: Path,
        output: Path,
        cases: tuple[Case, ...],
        *,
        player_locomotion: str = "scripted",
        native_targeting: str = "strict",
    ):
        if player_locomotion not in PLAYER_LOCOMOTION:
            raise TrialError("unknown Player locomotion mode")
        if native_targeting not in NATIVE_TARGETING:
            raise TrialError("unknown native targeting mode")
        self.player_locomotion = player_locomotion
        self.native_targeting = native_targeting
        self.run, self.output, self.cases = run.resolve(), output.resolve(), cases
        self.resources: list[Any] = []
        self.created_output = False
        try:
            self.prepare()
        except BaseException as exc:
            if self.created_output:
                with (
                    contextlib.suppress(OSError),
                    (self.output / "preflight_failure.json").open("xb") as stream,
                ):
                    stream.write(
                        canonical(
                            {
                                "schema": SCHEMA,
                                "failure": str(exc),
                                "native_targeting": self.native_targeting,
                                "player_locomotion": self.player_locomotion,
                                "cases": [
                                    {**asdict(case), "status": "unattempted"} for case in cases
                                ],
                            }
                        )
                        + b"\n"
                    )
            for resource in reversed(self.resources):
                resource.close()
            raise

    def prepare(self) -> None:
        self.session = json.loads((self.run / "session.json").read_text())
        require_session(self.session)
        catalog_bytes = (self.run / "lane-catalog.json").read_bytes()
        if hashlib.sha256(catalog_bytes).hexdigest() != self.session["lane_catalog_sha256"]:
            raise TrialError("lane catalog differs from launcher identity")
        self.catalog = {lane["name"]: lane for lane in json.loads(catalog_bytes)["lanes"]}
        if set(self.catalog) != set(LANES) or self.catalog["shuffled_weight"].get("seed") != "7":
            raise TrialError("unexpected comparison catalog")
        self.output.mkdir()  # No reuse/truncation, including failed diagnostics.
        self.created_output = True
        self.ledger = (self.output / "ledger.ndjson").open("xb")
        self.resources.append(self.ledger)
        self.console = AppendInbox(self.run / "diagnostic-commands.ndjson", CONSOLE_LIMIT)
        self.resources.append(self.console)
        self.native = AppendInbox(self.run / "diagnostic-client-actions.ndjson", NATIVE_LIMIT)
        self.resources.append(self.native)
        self.native_seq = len(self.native.initial.splitlines())
        for index, line in enumerate(self.native.initial.splitlines()):
            if json.loads(line).get("seq") != index:
                raise TrialError("existing native inbox sequence invalid")
        required = sum(
            case.source in ("brush", "attack") or self.walking_case(case) for case in self.cases
        )
        if self.native_seq + required > NATIVE_REQUESTS:
            raise TrialError("insufficient native request budget for declared schedule")
        with (self.run / "capture" / "frames.ndjson").open("rb") as stream:
            capture = json.loads(stream.readline())
        if not isinstance(capture, dict):
            raise TrialError("invalid capture timing metadata")
        first_request_ms = integer(capture.get("request_utc_epoch_ms"), "capture request UTC")
        first_elapsed_ns = integer(
            capture.get("request_wall_elapsed_ns"), "capture request elapsed"
        )
        duration = integer(capture.get("requested_duration_seconds"), "capture duration")
        if not 1 <= duration <= 720 or first_elapsed_ns > first_request_ms * 1_000_000:
            raise TrialError("invalid capture duration or origin")
        # Use the same conservative construction-time origin as the operator segment.
        # The producer schedules frames from its first request, which may be delayed.
        capture_end = first_request_ms / 1000 - first_elapsed_ns / 1e9 + duration
        self.started = time.monotonic()
        self.deadline = min(
            self.started + TOTAL_SECONDS,
            self.started + capture_end - time.time() - OPERATOR_RESERVE_SECONDS,
        )
        if self.deadline - self.started < RESTORE_RESERVE_SECONDS + MIN_CASE_ALLOWANCE_SECONDS:
            raise TrialError("insufficient capture time; operator reserve retained")
        self.epoch: Epoch | None = None
        self.safe = False  # No applied eligible baseline has been observed yet.
        self.expected_light = 15
        self.results = [{**asdict(case), "status": "unattempted"} for case in self.cases]
        self.record(
            "plan",
            schedule=[asdict(case) for case in self.cases],
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            lane_catalog_sha256=self.session["lane_catalog_sha256"],
            player_locomotion=self.player_locomotion,
            native_targeting=self.native_targeting,
            scope=self.locomotion_scope() + "; " + self.targeting_scope() + "; "
            "native gesture dispatch, source observations and applied responses are separate",
            comparison_scope=self.comparison_scope,
            operator_reserve_seconds=OPERATOR_RESERVE_SECONDS,
        )

    def walking_case(self, case: Case) -> bool:
        return case.source == "player_approach" and self.player_locomotion in WALK_ACTIONS

    def tracking_case(self, case: Case) -> bool:
        return case.source == "player_approach" and self.player_locomotion in (
            "native_tracking_walk",
            "native_approach_walk",
        )

    def aimed_case(self, case: Case) -> bool:
        return case.source in AIM_ACTIONS and self.native_targeting == "aim_at_dispatch"

    def targeting_scope(self) -> str:
        if self.native_targeting == "aim_at_dispatch":
            return "synthetic client aiming immediately before native brush/attack dispatch"
        return "strict native brush/attack crosshair targeting"

    def locomotion_scope(self) -> str:
        if self.player_locomotion == "native_approach_walk":
            return (
                "scripted preparation followed by synthetic native target-tracking walk-up; "
                "not human input; maximum 20 callbacks, 0.65-block horizontal stop-request "
                "radius, native coasting remains; preparation forward 4.0 "
                "and signed lateral +1.2 blocks"
            )
        if self.player_locomotion == "native_tracking_walk":
            return (
                "scripted preparation followed by synthetic native target-tracking view "
                "and forward-key walking; not human input; preparation forward 5.5 "
                "and lateral 1.2 blocks, actual key-down separation must be measured"
            )
        if self.player_locomotion == "native_walk":
            return (
                "scripted preparation followed by synthetic native forward-key walking; "
                "not human input"
            )
        return "scripted world fixtures; Player displacement is automated, not keyboard walking"

    def record(self, kind: str, **fields: Any) -> None:
        line = (
            canonical({"schema": SCHEMA, "kind": kind, "wall_time_ns": time.time_ns(), **fields})
            + b"\n"
        )
        if self.ledger.tell() + len(line) > MAX_LEDGER_BYTES:
            raise TrialError("diagnostic ledger size exhausted")
        self.ledger.write(line)
        self.ledger.flush()

    def live(self, deadline: float | None = None) -> None:
        current = json.loads((self.run / "session.json").read_text())
        require_session(current)
        if current["run_id"] != self.session["run_id"]:
            raise TrialError("session replaced")
        if time.monotonic() >= min(self.deadline, deadline or self.deadline):
            raise TrialError("bounded diagnostic deadline exhausted")
        if self.epoch is not None:
            self.epoch.poll()
            if time.monotonic() - self.epoch.last_new > 3:
                raise TrialError("live program diagnostics stopped; no automatic restart")

    def wait(self, condition: Callable[[], Any], deadline: float) -> Any:
        while True:
            self.live(deadline)
            result = condition()
            if result:
                return result
            time.sleep(0.05)

    def send(self, command: str) -> None:
        self.live()
        if (
            not 1 <= len(command) <= 512
            or command.startswith("/")
            or any(ord(c) < 32 for c in command)
        ):
            raise TrialError("invalid diagnostic command")
        self.record("console_request", command=command, scope="request; not observed effect")
        self.console.append({"command": command})

    def current_epoch_number(self) -> int:
        return max(
            (
                int(path.name)
                for path in (self.run / "epochs").iterdir()
                if path.is_dir() and path.name.isdigit()
            ),
            default=-1,
        )

    def observe_current_epoch(self) -> None:
        """Attach read-only before setup; an existing hold cannot be reset by startup."""
        self.safe = False
        deadline = min(self.deadline, time.monotonic() + CASE_SECONDS)

        def opened() -> Epoch | None:
            path = self.run / "epochs" / str(self.current_epoch_number())
            if not (path / "epoch.json").exists():
                return None
            name = json.loads((path / "epoch.json").read_bytes()).get("name")
            if name not in self.catalog:
                raise TrialError("current epoch has an unknown lane")
            return Epoch(path, self.catalog[name], self.session)

        self.epoch = cast("Epoch", self.wait(opened, deadline))
        epoch = self.epoch
        # Prefix eligibility is not current readiness. Drain the bounded attachment
        # snapshot, then wait for a newly matched row beyond everything just read.
        while epoch.offset < epoch.attachment_end:
            self.live(deadline)
        baseline = max(epoch.issued.keys() | epoch.receipts.keys(), default=-1)
        self.wait(
            lambda: (
                row
                if eligible(row := epoch.latest())
                and row is not None
                and row["command_seq"] > baseline
                else None
            ),
            deadline,
        )
        self.safe = True

    def handover(self, lane: str) -> None:
        if not self.safe:
            raise TrialError("unresolved stimulus/hold prevents handover")
        previous = self.current_epoch_number()
        self.safe = False
        self.epoch = None
        self.send(f"neurocraftfly control {lane}")
        deadline = min(self.deadline, time.monotonic() + 20)

        def opened() -> Epoch | None:
            number = self.current_epoch_number()
            if number <= previous:
                return None
            path = self.run / "epochs" / str(number)
            if not (path / "epoch.json").exists():
                return None
            return Epoch(path, self.catalog[lane], self.session)

        epoch = cast("Epoch", self.wait(opened, deadline))
        self.epoch = epoch
        self.ready(deadline)
        self.safe = True
        self.record(
            "lane_applied",
            lane=lane,
            epoch=self.epoch.metadata,
            epoch_sha256=hashlib.sha256(self.epoch.raw).hexdigest(),
            scope="fresh epoch metadata matched to an applied diagnostic receipt",
        )

    def ready(self, deadline: float) -> dict[str, Any]:
        assert self.epoch is not None
        epoch = self.epoch

        def completed() -> dict[str, Any] | None:
            row = epoch.latest()
            if row is None or not eligible(row):
                return None
            # A newer issued hold/pending source is already known even while its
            # receipt is in flight. It must not hide behind an older matched pair.
            if any(
                not eligible(newer)
                for seq, newer in epoch.issued.items()
                if seq > row["command_seq"]
            ):
                return None
            return row

        return cast("dict[str, Any]", self.wait(completed, deadline))

    def settle_preparation(
        self, deadline: float, expected_light: int, *, after_seq: int | None = None
    ) -> dict[str, Any]:
        """Fresh physical baseline, or a rolling recheck after rotation-only framing."""
        assert self.epoch is not None
        epoch = self.epoch
        epoch.poll()
        physical = after_seq is None
        marker = (
            max(epoch.issued.keys() | epoch.receipts.keys(), default=-1)
            if after_seq is None
            else after_seq
        )

        def completed() -> dict[str, Any] | None:
            epoch.poll()
            return stable_preparation(epoch, marker, expected_light)

        row = cast("dict[str, Any]", self.wait(completed, deadline))
        self.record(
            "preparation_baseline",
            after_command_seq=marker,
            phase="physical_preparation" if physical else "rolling_after_rotation",
            first_observation_tick=row["observation_tick"] - BASELINE_OBSERVATIONS + 1,
            last_observation_tick=row["observation_tick"],
            observations=BASELINE_OBSERVATIONS,
            expected_local_light=expected_light,
            scope="consecutive matched eligible observations; stable light and event counters",
        )
        return row

    def observed_after(self, tick: int, deadline: float) -> dict[str, Any]:
        assert self.epoch is not None
        epoch = self.epoch
        return cast(
            "dict[str, Any]",
            self.wait(
                lambda: (
                    row
                    if (row := epoch.latest()) is not None and row["observation_tick"] >= tick
                    else None
                ),
                deadline,
            ),
        )

    def native_results(self, seq: int) -> list[dict[str, Any]]:
        path = self.run / "diagnostic-client-actions-results.ndjson"
        if not path.exists():
            return []
        with path.open("rb") as stream:
            raw = stream.read(NATIVE_RESULTS_LIMIT + 1)
        if len(raw) > NATIVE_RESULTS_LIMIT:
            raise TrialError("native dispatch results exceed byte budget")
        # A writer may be between bytes of the final line; only complete rows count.
        rows = [json.loads(line) for line in raw.split(b"\n")[:-1]]
        if any(not isinstance(row, dict) for row in rows):
            raise TrialError("invalid native dispatch result row")
        return [row for row in rows if row.get("seq") == seq]

    def server_query(self, command: str, pattern: str, deadline: float) -> str:
        """Fresh read-only console result from this sole writer, bounded to 64 KiB."""
        path = self.run / "server-gradle.log"
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            cursor = stream.tell()
            self.send(command)
            pending = bytearray()

            def result() -> str | None:
                chunk = stream.read(65_537 - len(pending))
                pending.extend(chunk)
                if len(pending) > 65_536:
                    raise TrialError("server query exceeded bounded log window")
                complete = bytes(pending).rsplit(b"\n", 1)[0] if b"\n" in pending else b""
                matches = re.findall(
                    pattern, complete.decode("utf-8", errors="strict"), re.MULTILINE
                )
                if len(matches) > 1:
                    raise TrialError("ambiguous server query result")
                return matches[0] if matches else None

            value = str(self.wait(result, min(deadline, time.monotonic() + 2.0)))
        self.record(
            "server_query_result",
            command=command,
            log_offset=cursor,
            result=value,
            scope="fresh local server console output; read-only query, not an authenticated trace",
        )
        return value

    def server_player_position(self, deadline: float) -> list[float]:
        raw = self.server_query(
            f"data get entity {PLAYER} Pos",
            rf"{PLAYER} has the following entity data: \[([^\]\r\n]+)\]",
            deadline,
        )
        values = [float(part.strip().removesuffix("d")) for part in raw.split(",")]
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            raise TrialError("invalid server Player position")
        return values

    def perform_walk(self, case: Case, request: dict[str, Any], deadline: float) -> dict[str, Any]:
        epoch = self.epoch
        before = self.server_player_position(deadline)
        refusal: str | None
        if epoch is None or self.epoch is not epoch:
            refusal = "native_request_epoch_unavailable"
        else:
            epoch.poll()
            refusal = walking_admission_refusal(epoch, request["game_tick"], self.expected_light)
        if refusal is not None:
            evidence = {
                "native_hold_completed": False,
                "admission_refused": True,
                "reason": refusal,
                "server_player_start": before,
                "server_player_end": None,
                "server_horizontal_displacement": None,
                "server_start_barrier_tick": None,
                "server_end_barrier_tick": None,
                "native_results": [],
                "scope": (
                    "synthetic native target-tracking candidate; "
                    if request["action"] in TRACKING_ACTIONS
                    else ""
                )
                + "post-position-query admission refused; no native request appended; "
                "only the initial server position was measured",
            }
            self.record(
                "walking_admission_refused", case_index=case.index, request=request, **evidence
            )
            return evidence
        self.native.append(request)
        self.native_seq += 1
        self.record("native_request", case_index=case.index, request=request)
        # Refusal and missing evidence are retained; neither triggers a retry.
        start_deadline = min(deadline, time.monotonic() + NATIVE_DISPATCH_SECONDS)
        found = self.native_results(request["seq"])
        while not found and time.monotonic() < start_deadline:
            self.live(deadline)
            time.sleep(0.05)
            found = self.native_results(request["seq"])
        barrier, end_barrier = None, None
        if found and found[0].get("status") == "WALK_STARTED":
            # This server query executes after STARTED was read. A source at or
            # before this barrier is ambiguous and cannot qualify walking onset.
            barrier = int(self.server_query("time query gametime", r"The time is (\d+)", deadline))
            terminal_deadline = min(deadline, time.monotonic() + 3.2)
            while len(found) == 1 and time.monotonic() < terminal_deadline:
                self.live(deadline)
                time.sleep(0.05)
                found = self.native_results(request["seq"])
            if len(found) >= 2:
                end_barrier = int(
                    self.server_query("time query gametime", r"The time is (\d+)", deadline)
                )
        after = self.server_player_position(deadline)
        evidence = walking_dispatch_evidence(request, found)
        evidence.update(
            server_player_start=before,
            server_player_end=after,
            server_horizontal_displacement=math.hypot(after[0] - before[0], after[2] - before[2]),
            server_start_barrier_tick=barrier,
            server_end_barrier_tick=end_barrier,
            native_results=found,
            scope=(
                "synthetic native target-tracking walk-up; bounded gesture completion, "
                "not a fixed 20-callback hold; native coasting remains; "
                if request["action"] == "walk_approach_target"
                else "synthetic native target-tracking walking; "
                if request["action"] == "walk_track_target"
                else "synthetic native walking; "
            )
            + "actual server endpoint positions; "
            "source inside the server-query envelope is temporal evidence, "
            "not exclusive causal attribution or an exact key-hold/reaction-time window",
        )
        self.record("walking_observed", case_index=case.index, **evidence)
        if request["action"] == "walk_approach_target" and evidence["native_hold_completed"]:
            # Endpoints and the source envelope are already sealed. This one view
            # request cannot extend them or establish visible body movement.
            command = camera_rotation(offset=True)
            self.send(command)
            self.record(
                "camera_framing_request",
                case_index=case.index,
                command=command,
                after_server_end_barrier_tick=end_barrier,
                scope="rotation-only framing after native release and endpoint measurement; "
                "not visual evidence; source envelope unchanged",
            )
        return evidence

    def frame_after_native_dispatch(
        self, case: Case, request: Mapping[str, Any], deadline: float
    ) -> tuple[list[dict[str, Any]], bool]:
        """One bounded dispatch observation, no retries or actor displacement."""
        stop = min(deadline, time.monotonic() + NATIVE_DISPATCH_SECONDS)
        found = self.native_results(request["seq"])
        while not found and time.monotonic() < stop:
            self.live(deadline)
            time.sleep(0.05)
            found = self.native_results(request["seq"])
        verified = len(found) == 1 and native_dispatch_matches(request, found[0])
        self.record(
            "immediate_native_dispatch", case_index=case.index, results=found, verified=verified
        )
        if verified:
            command = camera_rotation(offset=True)
            self.send(command)
            self.record(
                "camera_framing_request",
                case_index=case.index,
                command=command,
                scope="rotation-only framing request; not visual evidence",
            )
        return found, verified

    def run_case(self, case: Case) -> None:
        assert self.epoch is not None
        deadline = min(self.deadline - RESTORE_RESERVE_SECONDS, time.monotonic() + CASE_SECONDS)
        result = self.results[case.index]
        result["status"] = "preparing"
        self.safe = False
        epoch = self.epoch
        try:
            self.ready(deadline)
            # Every repetition tests the same bright-to-dark stimulus. Raw skylight
            # bleeds under the roof near its edge, preventing a stable dark preparation.
            self.expected_light = 15
            self.send(lighting(self.expected_light == 15))
            self.settle_preparation(deadline, self.expected_light)
            walking = self.walking_case(case)
            tracking = self.tracking_case(case)
            for command in preparation_commands(case, native_walk=walking):
                self.send(command)
            row = self.settle_preparation(deadline, self.expected_light)
            if case.source in ("brush", "attack"):
                self.send(native_observer_position())
                self.send(f"gamemode creative {PLAYER}")
                self.observed_after(row["observation_tick"] + 2, deadline)
                row = self.settle_preparation(deadline, self.expected_light)
            if walking:
                self.send(
                    walking_observer_position(
                        tracking=self.player_locomotion == "native_tracking_walk",
                        approach=self.player_locomotion == "native_approach_walk",
                    )
                )
                self.send(f"gamemode survival {PLAYER}")
                self.observed_after(row["observation_tick"] + 2, deadline)
                row = self.settle_preparation(deadline, self.expected_light)
            aim_after_tick = row["observation_tick"]
            physical_marker = row["command_seq"] - BASELINE_OBSERVATIONS
            # The native-only candidate crosses the future front ray from the existing
            # off-axis preparation. It is unmeasured and is never changed during the hold.
            aimed = self.aimed_case(case)
            if not aimed and not tracking:
                command = (
                    walking_heading()
                    if walking
                    else camera_rotation(offset=case.source not in ("brush", "attack"))
                )
                self.send(command)
                self.record(
                    "walking_heading_request" if walking else "camera_framing_request",
                    case_index=case.index,
                    command=command,
                    after_observation_tick=aim_after_tick,
                    scope=(
                        "preparation-only native walking heading candidate; fly yaw +90 degrees, "
                        "pitch 0; unmeasured"
                        if walking
                        else "rotation-only framing request; not visual evidence"
                    ),
                )
            self.observed_after(row["observation_tick"] + 2, deadline)
            row = self.settle_preparation(
                deadline,
                self.expected_light,
                after_seq=physical_marker,
            )
            self.record(
                "framing_observations",
                case_index=case.index,
                request_after_observation_tick=aim_after_tick,
                fresh_observation_tick=row["observation_tick"],
                scope=(
                    "stable preparation rechecked; synthetic client aim deferred to native dispatch"
                    if aimed
                    else "stable preparation rechecked; synthetic target-tracking aim "
                    "starts with native key hold"
                    if tracking
                    else "observations advanced after rotation-only request; not visual evidence"
                ),
            )
            self.safe = True
            baseline = row["policy_state_after"]["previous_sensors"]
            if case.source == "light" and baseline["local_light"] != 15:
                result.update(
                    status="fixture_miss",
                    reason="covered fixture did not yield declared light baseline",
                )
                self.record("case_end", result=result)
                return
            if case.source in ("brush", "attack") and not aimed and not native_geometry(row):
                result.update(
                    status="fixture_miss",
                    reason="fresh rear ray did not retain a close Player",
                    fixture_verified=False,
                    geometry_observation=row,
                )
                self.record("case_end", result=result)
                return
            records = [{"epoch_identity": self.epoch.identity, "row": row}]
            # latest() selects a matched baseline, but its poll can already have read
            # newer issued rows whose receipts have not arrived. Snapshot that tail
            # before attaching the observer; poll() will deliver only later log rows.
            receipt = self.epoch.receipts.get(row["command_seq"])
            if receipt is not None:
                records.append({"epoch_identity": self.epoch.identity, "row": receipt})
            for seq in sorted(self.epoch.issued.keys() | self.epoch.receipts.keys()):
                if seq <= row["command_seq"]:
                    continue
                for buffered in (self.epoch.issued.get(seq), self.epoch.receipts.get(seq)):
                    if buffered is not None:
                        if len(records) >= MAX_CASE_ROWS:
                            raise TrialError("case row budget exhausted")
                        records.append({"epoch_identity": self.epoch.identity, "row": buffered})

            def retain(bound: dict[str, Any]) -> None:
                if len(records) >= MAX_CASE_ROWS:
                    raise TrialError("case row budget exhausted")
                records.append(bound)
                self.record("decision_row", case_index=case.index, **bound)

            self.epoch.observer = retain
            start_tick, end_tick = row["observation_tick"] + 1, row["observation_tick"] + 100
            result.update(
                status="stimulus_requested",
                start_tick=start_tick,
                end_tick=end_tick,
                epoch_identity=self.epoch.identity,
            )
            self.record(
                "case_begin",
                case=asdict(case),
                identity=self.epoch.identity,
                epoch_sha256=hashlib.sha256(self.epoch.raw).hexdigest(),
                start_tick=start_tick,
                end_tick=end_tick,
                baseline_records=records.copy(),
            )
            self.safe = False
            if walking:
                request = {
                    "seq": self.native_seq,
                    "action": WALK_ACTIONS[self.player_locomotion],
                    "entity_uuid": self.session["entity_id"],
                    "game_tick": start_tick - 1,
                    "valid_through_tick": start_tick + 79,
                }
                result["native_request"] = request
                result["walking"] = self.perform_walk(case, request, deadline)
            elif case.source in ("brush", "attack"):
                request = {
                    "seq": self.native_seq,
                    "action": AIM_ACTIONS[case.source] if aimed else case.source,
                    "entity_uuid": self.session["entity_id"],
                    "game_tick": start_tick - 1,
                    "valid_through_tick": start_tick + 79,
                }
                self.native.append(request)
                self.native_seq += 1
                self.record("native_request", case_index=case.index, request=request)
                result["native_request"] = request
                found, verified = self.frame_after_native_dispatch(case, request, deadline)
                result["immediate_native_dispatch"] = found
                result["immediate_native_dispatch_verified"] = verified
            elif case.source == "food":
                self.send(food_stimulus())
            elif case.source == "light":
                self.send(lighting(False))
                self.expected_light = 0
            else:
                self.perform_approach(case, start_tick, deadline)
            # Do not move/reset any actor or body through the response window and observed hold.
            self.observed_after(end_tick + 2, deadline)
            self.ready(deadline)  # All holds and the full declared cooldown must finish first.
            self.safe = True
            self.epoch.poll()
            summary = reduce_case(
                case, self.epoch.identity, records, start_tick=start_tick, end_tick=end_tick
            )
            result.update(status="observed", **summary)
            result["fixture_verified"] = result["source_verified"]
            if walking:
                walk = result["walking"]
                barrier = walk["server_start_barrier_tick"]
                source_ticks = result["observed_source_ticks"]
                walk["source_after_start_barrier"] = (
                    type(barrier) is int and bool(source_ticks) and min(source_ticks) > barrier
                )
                result["fixture_verified"] = walking_trial_verified(walk, source_ticks)
                if not result["fixture_verified"]:
                    result.update(status="stimulus_failed", outcome="native_walking_unconfirmed")
            if case.source in ("brush", "attack"):
                found = self.native_results(result["native_request"]["seq"])
                result["native_dispatch"] = found
                result["native_dispatch_verified"] = (
                    result["immediate_native_dispatch_verified"]
                    and len(found) == 1
                    and native_dispatch_matches(result["native_request"], found[0])
                )
                if not result["native_dispatch_verified"]:
                    result["status"] = "stimulus_failed"
                    result["fixture_verified"] = False
                    result["outcome"] = "native_dispatch_unconfirmed"
            self.record("case_end", result=result)
        except BaseException as exc:
            result.update(failure_phase=result["status"], status="failed", reason=str(exc))
            self.record("case_end", result=result)
            raise
        finally:
            epoch.observer = None

    def perform_approach(self, case: Case, start_tick: int, deadline: float) -> None:
        """Historical display fixture; a release protocol overrides its adaptive schedule."""
        assert self.epoch is not None
        if case.source == "player_approach":
            self.send(f"gamemode creative {PLAYER}")
        else:
            self.send(
                f"execute at {FLY} rotated ~ 0 run summon minecraft:cow ^0 ^0 ^5.5 "
                '{NoAI:1b,NoGravity:1b,Invulnerable:1b,Tags:["nf_v2_cow"]}'
            )
        for distance in (5.5, 4.5, 3.5, 2.5, 1.5):
            latest = self.epoch.latest()
            assert latest is not None
            if latest["policy_state_after"]["source"] is not None:
                break
            target = PLAYER if case.source == "player_approach" else COW
            height = "-1.5" if case.source == "player_approach" else "0"
            self.send(
                f"execute at {FLY} rotated ~ 0 run tp {target} ^0 ^{height} ^{distance} ~180 0"
            )
            self.observed_after(latest["observation_tick"] + 6, deadline)

    def execute(self) -> dict[str, Any]:
        restored, failure = False, None
        try:
            self.observe_current_epoch()
            self.safe = False
            self.send("gamerule sendCommandFeedback false")
            self.send("gamerule doMobLoot false")
            self.send("fill -24 74 -24 24 74 24 minecraft:black_concrete")
            self.send(lighting(True))
            self.expected_light = 15
            self.settle_preparation(
                min(self.deadline - RESTORE_RESERVE_SECONDS, time.monotonic() + CASE_SECONDS),
                self.expected_light,
            )
            self.safe = True
            previous = None
            for case in self.cases:
                if (
                    self.deadline - time.monotonic()
                    < RESTORE_RESERVE_SECONDS + MIN_CASE_ALLOWANCE_SECONDS
                ):
                    raise TrialError(
                        "insufficient capture time for another case and operator reserve"
                    )
                if case.lane != previous or self.reset_each_case:
                    self.handover(case.lane)
                    previous = case.lane
                self.run_case(case)
        except (TrialError, OSError, ValueError, KeyError, KeyboardInterrupt) as exc:
            if isinstance(exc, KeyboardInterrupt):
                self.safe = False
            failure = str(exc)
            self.record("schedule_stopped", reason=failure, safe_to_reconfigure=self.safe)
        finally:
            if self.safe:
                try:
                    self.safe = False
                    self.ready(self.deadline)
                    self.send(f"gamemode spectator {PLAYER}")
                    self.send("kill @e[tag=nf_v2_food]")
                    self.send("kill @e[tag=nf_v2_cow]")
                    self.send(lighting(True))
                    self.expected_light = 15
                    self.settle_preparation(self.deadline, self.expected_light)
                    self.safe = True
                    self.handover("original")
                    restored = True
                except (TrialError, OSError, ValueError, KeyError) as exc:
                    self.record("restore_failed", reason=str(exc))
            for result in self.results:
                if result["status"] in ("unattempted", "preparing"):
                    result.update(reason=failure or "not reached")
            summary = {
                "schema": SCHEMA,
                "cases": self.results,
                "original_restored": restored,
                "failure": failure,
                "all_cases_observed": all(row["status"] == "observed" for row in self.results),
                "player_locomotion": self.player_locomotion,
                "native_targeting": self.native_targeting,
                "scope": "diagnostic source/receipt matching; video inspection still required; "
                + self.locomotion_scope()
                + "; "
                + self.targeting_scope(),
            }
            self.record("run_end", summary=summary)
            with (self.output / "summary.json").open("xb") as stream:
                stream.write(canonical(summary) + b"\n")
            self.console.close()
            self.native.close()
            self.ledger.close()
        return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new directory; never reused")
    parser.add_argument("--source", choices=SOURCES)
    parser.add_argument("--player-locomotion", choices=PLAYER_LOCOMOTION, default="scripted")
    parser.add_argument("--lane", choices=LANES)
    parser.add_argument("--native-targeting", choices=NATIVE_TARGETING, default="strict")
    args = parser.parse_args(argv)
    runner = Runner(
        args.run,
        args.output,
        schedule(args.source, args.lane),
        player_locomotion=args.player_locomotion,
        native_targeting=args.native_targeting,
    )
    result = runner.execute()
    print(
        json.dumps(
            {
                "output": str(args.output),
                "original_restored": result["original_restored"],
                "all_cases_observed": result["all_cases_observed"],
            }
        )
    )
    return 0 if result["all_cases_observed"] and result["original_restored"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
