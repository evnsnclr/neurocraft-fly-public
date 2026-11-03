"""Run one trusted packaged entrypoint with explicit local source import roots."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ENTRYPOINTS = frozenset(
    {
        "acquire_malecns.py",
        "build_malecns_runtime.py",
        "installed_mod.py",
        "run_interactive_demo.py",
    }
)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ENTRYPOINTS:
        raise SystemExit("Unknown companion entrypoint.")
    scripts = Path(__file__).resolve().parent
    sys.path[:0] = [str(scripts.parent), str(scripts)]
    sys.argv = [str(scripts / sys.argv[1]), *sys.argv[2:]]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
