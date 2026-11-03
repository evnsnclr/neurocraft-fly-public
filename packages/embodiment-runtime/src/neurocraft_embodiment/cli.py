"""Run or replay a reference-embodiment trace without touching the Makefile.

Lives inside the package rather than under ``scripts/`` so that ``mypy packages``
and ``ruff .`` cover it with no build-system wiring:

    uv run --frozen python -m neurocraft_embodiment.cli run --ticks 200
    uv run --frozen python -m neurocraft_embodiment.cli replay
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from neurocraft_connectome import load_malecns_fixture
from neurocraft_controller import ConnectomeController

from .loop import replay_embodiment_trace, repository_root, run_embodiment
from .reference import ReferenceArena, ReferenceEmbodiment

DEFAULT_RUN_ID = "reference-embodiment"
DEFAULT_ENTITY_ID = "reference-fly"


def default_fixture() -> Path:
    return repository_root() / "data" / "fixtures" / "mcns-v1.0-flight-power-v1.json"


def default_trace() -> Path:
    return repository_root() / "artifacts" / "traces" / "reference-embodiment.ndjson"


def _run(fixture: Path, output: Path, ticks: int, seed: int) -> dict[str, Any]:
    graph = load_malecns_fixture(fixture)
    controller = ConnectomeController.create(
        graph,
        run_id=DEFAULT_RUN_ID,
        entity_id=DEFAULT_ENTITY_ID,
    )
    embodiment = ReferenceEmbodiment(ReferenceArena(seed=seed))
    return run_embodiment(
        controller=controller,
        graph=graph,
        embodiment=embodiment,
        output=output,
        ticks=ticks,
    )


def _replay(fixture: Path, trace: Path, seed: int) -> dict[str, Any]:
    return replay_embodiment_trace(
        trace=trace,
        graph_factory=lambda: load_malecns_fixture(fixture),
        embodiment_factory=lambda: ReferenceEmbodiment(ReferenceArena(seed=seed)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="write a new no-clobber embodiment trace")
    run.add_argument("--fixture", type=Path, default=None)
    run.add_argument("--output", type=Path, default=None)
    run.add_argument("--ticks", type=int, default=200)
    run.add_argument("--seed", type=int, default=0)
    replay = subparsers.add_parser("replay", help="verify and re-execute an embodiment trace")
    replay.add_argument("--fixture", type=Path, default=None)
    replay.add_argument("--trace", type=Path, default=None)
    replay.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fixture = args.fixture or default_fixture()
    if args.command == "run":
        receipt = _run(fixture, args.output or default_trace(), args.ticks, args.seed)
    else:
        receipt = _replay(fixture, args.trace or default_trace(), args.seed)
    json.dump(receipt, sys.stdout, allow_nan=False, separators=(",", ":"), sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
