#!/usr/bin/env python3
"""Run the opt-in ADR 0020 operator-lane probe against a real neural service.

The default GameTest lane has no bridge, so its body-program pins hand-build PAUSE
acknowledgements and cannot see *when* a real acknowledgement reaches the entity tick
relative to the tick it names. That is exactly the property that decides whether
``/neurocraftfly program <id>`` is reachable in a live session, so this launcher drives
it for real: it starts the fixture neural service (without a trace recorder -- a paused
service issues neutral actions the fixture recorder cannot bind, ADR 0020), runs the
mod's ``ExternalBridgeProgramAuthorityGameTest`` through Gradle with the bridge pointed
at that service, and verifies the receipt the probe writes: the PAUSE acknowledgement's
effective tick, the tick scripted authority engaged on, the program-owned window over a
frozen neural cursor, the transaction-driven release, the connectome resuming from the
same cursor, and an operator RESET applied at its own accepted effective tick.

Nothing here is evidence about the connectome. The body program is scripted
presentation and the probe asserts the connectome was *not* driving while it ran.
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO, cast

# ``python scripts/run_program_authority_gametest.py`` places ``scripts/`` rather than the
# repository root on sys.path; add the root so the sibling launcher module imports.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurocraft_protocol import ACTION_TARGET_OFFSET_TICKS

from scripts.run_interactive_demo import (
    HASH_KEYS,
    IDENTITY_KEYS,
    TOKEN_NAME,
    LauncherError,
    LineCapture,
    _available_port,
    _child_environment,
    _start,
    _terminate,
    _wait_tcp,
)

ROOT = Path(__file__).resolve().parents[1]
MOD_ROOT = ROOT / "mod"
RUN_ROOT = ROOT / "artifacts" / "program-authority-gametest"
FIXTURE = ROOT / "data" / "fixtures" / "mcns-v1.0-flight-power-v1.json"
#: Where mod/build.gradle points the external lane's receipt; the probe writes it CREATE_NEW
#: and the Gradle task deletes the previous one before the run.
RECEIPT = MOD_ROOT / "build" / "receipts" / "external-bridge-gametest.json"
SCHEMA = "neurocraft.external_bridge_program_authority.v1"
SCENARIO = "program_authority"
EXPECTED_PACING_MODE = "external_gametest_probe_monotonic_20hz"
EXPECTED_PACING_PERIOD_NS = 50_000_000
#: The probe's own bar: one second of scripted authority at 20 Hz.
MINIMUM_PROGRAM_OWNED_TICKS = 20
POSITION_EPSILON = 1.0e-6

RECEIPT_KEYS = {
    "schema",
    "protocol",
    "entity_id",
    "bridge_state",
    "pacing_mode",
    "pacing_period_ns",
    "driving_command_seq",
    "neural_cursor_before_pause",
    "pause",
    "scripted_window",
    "release",
    "resumed",
    "reset",
    "claim_boundary",
    "completed_game_tick",
}
_PROGRAM_STATUS = re.compile(r"^applied effective_tick=(\d+) engaged_tick=(\d+)$")
_RESET_STATUS = re.compile(r"^applied effective_tick=(\d+)$")
_HASH = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ProbePlan:
    run_id: str
    entity_id: str
    session_epoch: int
    bridge_port: int
    run_dir: Path

    @property
    def identity_path(self) -> Path:
        return self.run_dir / "identity.json"

    @property
    def service_log(self) -> Path:
        return self.run_dir / "service.log"

    @property
    def receipt_copy(self) -> Path:
        return self.run_dir / "program-authority-receipt.json"


@dataclass(frozen=True)
class ProbeSummary:
    """The verified relation between what the service stamped and what the mod did."""

    pause_issued_tick: int
    pause_effective_tick: int
    pause_engaged_tick: int
    program_owned_ticks: int
    paused_commands_observed: int
    neural_cursor_before_pause: int
    frozen_neural_cursor: int
    release_issued_tick: int
    release_effective_tick: int
    release_engaged_tick: int
    resumed_neural_step_start: int
    reset_issued_tick: int
    reset_effective_tick: int
    reset_applied_tick: int

    def lines(self) -> list[str]:
        return [
            f"PAUSE issued_tick={self.pause_issued_tick} effective_tick={self.pause_effective_tick}"
            f" engaged_tick={self.pause_engaged_tick}"
            f" (effective_tick + {ACTION_TARGET_OFFSET_TICKS})",
            f"scripted window: program_owned_ticks={self.program_owned_ticks}"
            f" paused_commands_observed={self.paused_commands_observed}"
            f" neural_cursor_frozen_at={self.frozen_neural_cursor}"
            f" (last stepped command before the request ended at"
            f" {self.neural_cursor_before_pause})",
            f"RESUME issued_tick={self.release_issued_tick}"
            f" effective_tick={self.release_effective_tick}"
            f" released_tick={self.release_engaged_tick}"
            f" (effective_tick + {ACTION_TARGET_OFFSET_TICKS});"
            f" connectome resumed at neural step {self.resumed_neural_step_start}",
            f"RESET issued_tick={self.reset_issued_tick} effective_tick={self.reset_effective_tick}"
            f" applied_tick={self.reset_applied_tick} (== effective_tick)",
        ]


def prepare_plan(*, run_root: Path = RUN_ROOT, now: datetime | None = None) -> ProbePlan:
    """Create a fresh, no-clobber run directory and a fresh identity for one probe run."""

    timestamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"program-authority-{timestamp}-{secrets.token_hex(4)}"
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return ProbePlan(
        run_id=run_id,
        entity_id=str(uuid.uuid4()),
        session_epoch=time.time_ns(),
        bridge_port=_available_port(),
        run_dir=run_dir,
    )


def identity_command(plan: ProbePlan) -> list[str]:
    return [
        sys.executable,
        "-I",
        str(ROOT / "scripts" / "run_connectome_service.py"),
        "--fixture",
        str(FIXTURE),
        "--run-id",
        plan.run_id,
        "--entity-id",
        plan.entity_id,
        "--session-epoch",
        str(plan.session_epoch),
        "--mode",
        "faithful",
        "--identity-out",
        str(plan.identity_path),
        "--identity-only",
    ]


def service_command(plan: ProbePlan) -> list[str]:
    """The fixture service with no trace recorder and no interventions: PAUSE needs neither."""

    return [
        sys.executable,
        "-I",
        str(ROOT / "scripts" / "run_connectome_service.py"),
        "--fixture",
        str(FIXTURE),
        "--run-id",
        plan.run_id,
        "--entity-id",
        plan.entity_id,
        "--session-epoch",
        str(plan.session_epoch),
        "--mode",
        "faithful",
        "--host",
        "127.0.0.1",
        "--port",
        str(plan.bridge_port),
    ]


def gradle_command(plan: ProbePlan, identity: Mapping[str, object]) -> list[str]:
    return [
        "./gradlew",
        "--no-daemon",
        "--dependency-verification",
        "strict",
        "-Dneurocraft.fly.externalBridgeTest=true",
        f"-Dneurocraft.fly.externalBridgeScenario={SCENARIO}",
        f"-Dneurocraft.fly.bridgeUri=ws://127.0.0.1:{plan.bridge_port}/control",
        f"-Dneurocraft.fly.runId={plan.run_id}",
        f"-Dneurocraft.fly.sessionEpoch={plan.session_epoch}",
        f"-Dneurocraft.fly.entityId={plan.entity_id}",
        f"-Dneurocraft.fly.graphHash={identity['graph_hash']}",
        f"-Dneurocraft.fly.adapterHash={identity['adapter_hash']}",
        f"-Dneurocraft.fly.dynamicsHash={identity['dynamics_hash']}",
        f"-Dneurocraft.fly.decoderHash={identity['decoder_hash']}",
        f"-Dneurocraft.fly.observationSchemaHash={identity['observation_schema_hash']}",
        "runExternalBridgeGameTest",
    ]


def load_identity(path: Path, plan: ProbePlan) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != IDENTITY_KEYS:
        raise LauncherError("controller identity has missing or unknown fields")
    identity: dict[str, object] = dict(raw)
    if identity["run_id"] != plan.run_id or identity["entity_id"] != plan.entity_id:
        raise LauncherError("controller identity does not match the fresh run")
    if identity["session_epoch"] != plan.session_epoch:
        raise LauncherError("controller session epoch does not match the fresh run")
    for key in HASH_KEYS:
        value = identity[key]
        if not isinstance(value, str) or _HASH.fullmatch(value) is None:
            raise LauncherError(f"controller identity {key} is not a SHA-256")
    return identity


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise LauncherError(f"{name} must be an object")
    return dict(value)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LauncherError(f"{name} must be an integer")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise LauncherError(f"{name} must be a string")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise LauncherError(f"{name} must be a number")
    return float(value)


def _vector(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise LauncherError(f"{name} must be a 3-vector")
    return (_number(value[0], name), _number(value[1], name), _number(value[2], name))


def _program_transaction(
    section: Mapping[str, object], name: str, *, observed_key: str
) -> tuple[int, int, int]:
    """Return (issued_tick, effective_tick, engaged_tick) for one PAUSE/RESUME transaction.

    The status string is the mod's own record (``FlyControllerRuntime.programAuthorityStatus``),
    and the observed tick is the game tick at which the probe -- running after the entity tick
    -- saw the authority change. The three must agree with the wire relation: the service
    stamps ``effective_tick`` with the tick of the observation it first answered in the new
    state, which is ``issued_tick`` or ``issued_tick + 1``; the neutral action issued beside that
    acknowledgement targets one action budget later, and a scripted program may own only a tick
    whose admitted command is neutral, so the mod engages exactly
    ``ACTION_TARGET_OFFSET_TICKS`` ticks later. That is also at or after the first tick the
    acknowledgement can be polled, because the observation is sent after the entity tick's
    acknowledgement poll.
    """

    issued = _integer(section["issued_tick"], f"{name}.issued_tick")
    observed = _integer(section[observed_key], f"{name}.{observed_key}")
    status = _PROGRAM_STATUS.fullmatch(_text(section["status"], f"{name}.status"))
    if status is None:
        raise LauncherError(f"{name}.status is not an applied program-authority status")
    effective = int(status.group(1))
    engaged = int(status.group(2))
    if not issued <= effective <= issued + 1:
        raise LauncherError(
            f"{name}: effective_tick {effective} is not issued_tick or the next tick"
        )
    if engaged != effective + ACTION_TARGET_OFFSET_TICKS:
        raise LauncherError(
            f"{name}: engaged_tick {engaged} is not effective_tick plus the action budget"
        )
    if observed != engaged:
        raise LauncherError(
            f"{name}: the probe saw the authority change at tick {observed}, "
            f"but the transaction applied at tick {engaged}"
        )
    return issued, effective, engaged


def verify_receipt(document: Mapping[str, object]) -> ProbeSummary:
    """Fail closed unless the receipt shows the whole lane worked under the real timeline."""

    if set(document) != RECEIPT_KEYS:
        raise LauncherError(
            f"receipt keys differ from the schema: {sorted(set(document) ^ RECEIPT_KEYS)}"
        )
    if document["schema"] != SCHEMA:
        raise LauncherError(f"receipt schema is {document['schema']!r}, expected {SCHEMA!r}")
    if document["bridge_state"] != "ACTIVE":
        raise LauncherError(f"bridge did not stay ACTIVE: {document['bridge_state']!r}")
    if document["pacing_mode"] != EXPECTED_PACING_MODE:
        raise LauncherError("probe did not run under the declared 20 Hz pacer")
    if _integer(document["pacing_period_ns"], "pacing_period_ns") != EXPECTED_PACING_PERIOD_NS:
        raise LauncherError("probe pacing period is not 50 ms")
    cursor = _integer(document["neural_cursor_before_pause"], "neural_cursor_before_pause")

    pause = _object(document["pause"], "pause")
    pause_issued, pause_effective, pause_engaged = _program_transaction(
        pause, "pause", observed_key="engaged_tick"
    )
    # The engage tick is the paused action's target_tick, so the neutral action issued beside
    # the acknowledgement lands on that same tick and the program owns it. An engaged-but-idle
    # first tick would mean the acknowledgement outran its own action.
    owned_at_engage = _integer(
        pause["program_owned_ticks_at_engage"], "pause.program_owned_ticks_at_engage"
    )
    if owned_at_engage < 1:
        raise LauncherError("the program did not own the tick scripted authority engaged on")

    window = _object(document["scripted_window"], "scripted_window")
    if window["program"] != "HOVER" or window["source"] != "OPERATOR":
        raise LauncherError("the scripted window did not run HOVER on the OPERATOR lane")
    owned = _integer(window["program_owned_ticks"], "scripted_window.program_owned_ticks")
    if owned < MINIMUM_PROGRAM_OWNED_TICKS:
        raise LauncherError(f"the program owned only {owned} ticks")
    paused_commands = _integer(
        window["paused_commands_observed"], "scripted_window.paused_commands_observed"
    )
    if paused_commands < 1:
        raise LauncherError("no paused-neutral command was admitted during the scripted window")
    frozen = _integer(window["frozen_neural_cursor"], "scripted_window.frozen_neural_cursor")
    if frozen < cursor:
        # A stepped command still in flight when the PAUSE was issued may advance the cursor
        # past the pre-request snapshot; it can never move it backwards.
        raise LauncherError("the frozen neural cursor precedes the last stepped command")
    if (
        not _number(window["max_abs_vertical_thrust"], "scripted_window.max_abs_vertical_thrust")
        > 0
    ):
        raise LauncherError("HOVER presented no vertical thrust while it owned the body")

    release = _object(document["release"], "release")
    release_issued, release_effective, release_engaged = _program_transaction(
        release, "release", observed_key="released_tick"
    )
    # The engage tick itself is the first tick the program can own, so a window of N owned
    # ticks spans engaged_tick .. engaged_tick + N - 1 and the release is issued no earlier than
    # its last tick.
    if release_issued < pause_engaged + MINIMUM_PROGRAM_OWNED_TICKS - 1:
        raise LauncherError("the release was issued before the scripted window was long enough")
    if "reason=released" not in _text(release["ledger_line"], "release.ledger_line"):
        raise LauncherError("authority was not released by the accepted RESUME transaction")

    resumed = _object(document["resumed"], "resumed")
    resumed_start = _integer(resumed["neural_step_start"], "resumed.neural_step_start")
    resumed_end = _integer(resumed["neural_step_end"], "resumed.neural_step_end")
    skipped = _integer(
        resumed["commands_skipped_since_pause"], "resumed.commands_skipped_since_pause"
    )
    last_paused = _integer(
        window["last_paused_command_seq"], "scripted_window.last_paused_command_seq"
    )
    if _integer(resumed["command_seq"], "resumed.command_seq") != last_paused + 1 + skipped:
        raise LauncherError("resumed.commands_skipped_since_pause disagrees with the sequences")
    if resumed_end <= resumed_start:
        raise LauncherError("the resumed command did not step the connectome")
    # With no command retired between the last paused one and the first resumed one, the
    # cursor must pick up exactly where it froze; a retired command in between can only have
    # carried it forward.
    if (skipped == 0 and resumed_start != frozen) or (skipped > 0 and resumed_start < frozen):
        raise LauncherError("the connectome did not resume stepping from the frozen cursor")

    reset = _object(document["reset"], "reset")
    reset_issued = _integer(reset["issued_tick"], "reset.issued_tick")
    reset_applied = _integer(reset["applied_tick"], "reset.applied_tick")
    reset_status = _RESET_STATUS.fullmatch(_text(reset["status"], "reset.status"))
    if reset_status is None:
        raise LauncherError("reset.status is not an applied reset status")
    reset_effective = int(reset_status.group(1))
    if (
        not reset_issued + ACTION_TARGET_OFFSET_TICKS
        <= reset_effective
        <= reset_issued + ACTION_TARGET_OFFSET_TICKS + 1
    ):
        raise LauncherError("RESET effective_tick is outside its action-budget window")
    if reset_applied != reset_effective:
        raise LauncherError("RESET was not applied on its effective tick")
    target = _vector(reset["target_blocks"], "reset.target_blocks")
    position = _vector(reset["position_blocks"], "reset.position_blocks")
    if any(abs(a - b) > POSITION_EPSILON for a, b in zip(target, position, strict=True)):
        raise LauncherError("the fly was not at the RESET target after the accepted RESET")

    return ProbeSummary(
        pause_issued_tick=pause_issued,
        pause_effective_tick=pause_effective,
        pause_engaged_tick=pause_engaged,
        program_owned_ticks=owned,
        paused_commands_observed=paused_commands,
        neural_cursor_before_pause=cursor,
        frozen_neural_cursor=frozen,
        release_issued_tick=release_issued,
        release_effective_tick=release_effective,
        release_engaged_tick=release_engaged,
        resumed_neural_step_start=resumed_start,
        reset_issued_tick=reset_issued,
        reset_effective_tick=reset_effective,
        reset_applied_tick=reset_applied,
    )


def _compute_identity(plan: ProbePlan, environment: Mapping[str, str]) -> dict[str, object]:
    completed = subprocess.run(
        identity_command(plan),
        cwd=ROOT,
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120.0,
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        raise LauncherError(
            f"could not derive the fixture controller identity: {completed.stdout.strip()}"
        )
    return load_identity(plan.identity_path, plan)


def run_probe(
    plan: ProbePlan,
    *,
    startup_timeout: float,
    gametest_timeout: float,
) -> ProbeSummary:
    """Start the service, run the opt-in GameTest against it, verify and keep the receipt."""

    token = secrets.token_urlsafe(48)
    environment = _child_environment(token)
    identity = _compute_identity(plan, environment)
    if RECEIPT.exists():
        # The Gradle task deletes it too; refuse to read a stale receipt if the run dies early.
        RECEIPT.unlink()
    service = _start(service_command(plan), cwd=ROOT, environment=environment)
    capture = LineCapture(cast(TextIO, service.stdout), plan.service_log, "service")
    capture.start()
    try:
        _wait_tcp(service, plan.bridge_port, startup_timeout)
        print(f"[probe] service listening on 127.0.0.1:{plan.bridge_port}; running the GameTest")
        completed = subprocess.run(
            gradle_command(plan, identity),
            cwd=MOD_ROOT,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            timeout=gametest_timeout,
            check=False,
            shell=False,
        )
        if completed.returncode != 0:
            raise LauncherError(
                f"the program-authority GameTest failed (Gradle exit {completed.returncode}); "
                f"see the Gradle output above and {plan.service_log}"
            )
        if not RECEIPT.is_file():
            raise LauncherError(f"the GameTest passed but wrote no receipt at {RECEIPT}")
        document = json.loads(RECEIPT.read_text(encoding="utf-8"))
        summary = verify_receipt(_object(document, "receipt"))
        shutil.copyfile(RECEIPT, plan.receipt_copy)
        return summary
    finally:
        _terminate(service)
        capture.join()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--startup-timeout", type=float, default=90.0)
    parser.add_argument(
        "--gametest-timeout",
        type=float,
        default=1800.0,
        help="seconds allowed for Gradle to build the mod and run the probe",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=RUN_ROOT,
        help="where the service log, identity, and receipt copy are kept (gitignored)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = prepare_plan(run_root=args.run_root)
    print(f"[probe] run_id={plan.run_id} entity_id={plan.entity_id} run_dir={plan.run_dir}")
    print(f"[probe] the run token is supplied to children as {TOKEN_NAME} and never written")
    try:
        summary = run_probe(
            plan,
            startup_timeout=args.startup_timeout,
            gametest_timeout=args.gametest_timeout,
        )
    except LauncherError as exc:
        print(f"[probe] FAILED: {exc}", file=sys.stderr)
        return 1
    print("[probe] the operator program lane engaged under the real acknowledgement timeline:")
    for line in summary.lines():
        print(f"[probe]   {line}")
    print(f"[probe] receipt kept at {plan.receipt_copy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
