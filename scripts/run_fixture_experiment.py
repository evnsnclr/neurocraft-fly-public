#!/usr/bin/env python3
"""Run or byte-verify the frozen non-Minecraft fixture-physics pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

from neurocraft_experiments import (
    build_pilot_result,
    load_pilot_protocol,
    verify_reexecution,
    write_result_no_clobber,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = REPOSITORY_ROOT / "experiments/preregistration/fixture-physics-pilot-v3.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "experiments/results/fixture-physics-pilot-v3.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the deterministic MaleCNS-fixture local physics pilot. "
            "This is not Minecraft execution or biological validation."
        )
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="reexecute all cells and require an exact platform-neutral result core",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = load_pilot_protocol(args.protocol, repository_root=REPOSITORY_ROOT)
    if args.verify_existing:
        digest = verify_reexecution(
            args.output,
            protocol,
            repository_root=REPOSITORY_ROOT,
        )
        print(
            f"verified exact platform-neutral NON_MINECRAFT_LOCAL_FIXTURE_PHYSICS_ONLY "
            f"trials={protocol.planned_trial_count} file_sha256={digest}"
        )
        return 0
    result = build_pilot_result(protocol, repository_root=REPOSITORY_ROOT)
    digest = write_result_no_clobber(args.output, result)
    summary = result["result"]["summary"]
    print(
        f"wrote NON_MINECRAFT_LOCAL_FIXTURE_PHYSICS_ONLY "
        f"completed={summary['completed_trial_count']} failed={summary['failed_trial_count']} "
        f"file_sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
