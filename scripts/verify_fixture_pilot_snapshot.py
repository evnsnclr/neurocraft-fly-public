#!/usr/bin/env python3
"""Verify the frozen fixture-pilot snapshot's recorded bytes, on every platform.

**This no longer re-executes the pilot.** It used to, in the recorded execution environment
alone, and compare the rebuilt result core bit for bit. That comparison is now history, recorded
against tree :data:`neurocraft_experiments.pilot.RESULT_CORE_TREE_COMMIT`, for the reason written
there: the two-tick action budget moved every trial's action- and trace-stream digest (they
carry the issued ``Action`` envelope) and moved no metric at all, while a result is write-once
and a preregistration is append-only, so the recorded core can never be refreshed.

**The loss is real and is not dressed up: nothing bit-pins the fixture stack's numbers on any
platform now.** Linux never re-executed, so what is gone is the macOS check -- which was the only
one there was. What remains, on every platform, is every recorded byte: the snapshot file's
SHA-256, the envelope's internal digest over its own core, the recorded execution environment,
and the binding to the frozen preregistration.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

from neurocraft_experiments import (
    load_pilot_protocol,
    verify_result_envelope,
)
from neurocraft_experiments.arena import read_bounded_file_bytes
from neurocraft_experiments.pilot import MAX_RESULT_BYTES, RESULT_CORE_TREE_COMMIT

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL: Final = (
    REPOSITORY_ROOT / "experiments/preregistration/fixture-physics-pilot-v3.json"
)
DEFAULT_SNAPSHOT: Final = REPOSITORY_ROOT / "experiments/results/fixture-physics-pilot-v3.json"
EXPECTED_SNAPSHOT_SHA256: Final = "62458225913c11901a7d9ff609d53a3cc648761ae08679973748b377a273354b"
EXPECTED_RESULT_SHA256: Final = "57b42a36e72e72dff7fa3a60119730d2fa1d588f7de8d0c6549cdce2a0a563dd"
RECORDED_EXECUTION_ENVIRONMENT: Final[Mapping[str, str]] = {
    "platform": "macOS-15.6-arm64-arm-64bit-Mach-O",
    "python_implementation": "CPython",
    "python_version": "3.13.5",
}


@dataclass(frozen=True)
class SnapshotVerification:
    """Outcome of immutable-snapshot validation.

    ``recorded_environment`` says whether this machine is the one the snapshot was executed on.
    It is reported because it is true and interesting, **not** because anything different is
    verified there: the same recorded bytes are checked on every platform, and no reexecution
    happens on any of them.
    """

    current_environment: Mapping[str, str]
    recorded_environment: bool
    result_sha256: str
    snapshot_sha256: str


def current_execution_environment() -> Mapping[str, str]:
    """Return the same runtime fields recorded in a fixture-pilot envelope."""

    return {
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
    }


def verify_snapshot_integrity(path: Path) -> tuple[str, Mapping[str, Any]]:
    """Validate the exact frozen bytes and the envelope's internal digest."""

    payload = read_bounded_file_bytes(
        path, maximum_bytes=MAX_RESULT_BYTES, label="fixture-pilot-v3 snapshot"
    )
    digest = sha256(payload).hexdigest()
    if digest != EXPECTED_SNAPSHOT_SHA256:
        raise ValueError(
            "fixture-pilot-v3 snapshot SHA-256 mismatch: "
            f"expected {EXPECTED_SNAPSHOT_SHA256}, got {digest}"
        )
    envelope = verify_result_envelope(json.loads(payload))
    if envelope["execution_environment"] != RECORDED_EXECUTION_ENVIRONMENT:
        raise ValueError("fixture-pilot-v3 recorded execution environment mismatch")
    if envelope["result_sha256"] != EXPECTED_RESULT_SHA256:
        raise ValueError("fixture-pilot-v3 result-core SHA-256 mismatch")
    return digest, envelope


def verify_frozen_snapshot(
    snapshot_path: Path,
    protocol_path: Path,
    *,
    repository_root: Path,
    current_environment: Mapping[str, str] | None = None,
) -> SnapshotVerification:
    """Validate the frozen evidence's recorded bytes; never reexecute."""

    snapshot_digest, envelope = verify_snapshot_integrity(snapshot_path)
    protocol = load_pilot_protocol(protocol_path, repository_root=repository_root)
    result = envelope["result"]
    if not isinstance(result, Mapping):
        raise ValueError("fixture-pilot-v3 result core is not an object")
    if result.get("protocol_sha256") != protocol.sha256:
        raise ValueError("fixture-pilot-v3 result does not bind the active frozen protocol")

    actual_environment = (
        current_execution_environment() if current_environment is None else current_environment
    )
    return SnapshotVerification(
        current_environment=dict(actual_environment),
        recorded_environment=actual_environment == RECORDED_EXECUTION_ENVIRONMENT,
        result_sha256=EXPECTED_RESULT_SHA256,
        snapshot_sha256=snapshot_digest,
    )


def _environment_text(environment: Mapping[str, str]) -> str:
    return json.dumps(dict(environment), sort_keys=True, separators=(",", ":"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the immutable fixture-pilot-v3 snapshot on every platform and "
            "reexecute it only in its exact recorded runtime environment."
        )
    )
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outcome = verify_frozen_snapshot(
        args.snapshot,
        args.protocol,
        repository_root=REPOSITORY_ROOT,
    )
    print(
        "verified immutable fixture-pilot-v3 snapshot "
        f"file_sha256={outcome.snapshot_sha256} "
        f"result_sha256={outcome.result_sha256}"
    )
    print(
        "fixture-pilot-v3 reexecution is not attempted on any platform: the result core last "
        f"reproduced bit for bit at tree {RESULT_CORE_TREE_COMMIT} in "
        f"{_environment_text(RECORDED_EXECUTION_ENVIRONMENT)}; "
        f"current_environment={_environment_text(outcome.current_environment)}"
        f"{' (the recorded one)' if outcome.recorded_environment else ''}. "
        "Recorded bytes are verified above; the fixture stack's numbers are pinned by nothing "
        "on any platform"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
