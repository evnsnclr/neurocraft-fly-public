"""Build-time retrieval of a pinned CPython distribution; never downloads Minecraft."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_sqlite_runtime import build  # noqa: E402
from scripts.companion_downloads import platform_key  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, help="Hash-verified upstream build-input cache")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output must be a new local path")
    pin = json.loads((ROOT / "release/companion-runtime-pins.json").read_bytes())["platforms"][
        platform_key()
    ]["python"]
    args.output.mkdir(parents=True)
    build(args.output / "python", cache=args.cache)
    (args.output / "python-runtime-provenance.json").write_text(
        json.dumps(
            {
                "python_base": pin,
                "sqlite_build": "python/SQLITE_BUILD.json",
                "scope": "verified upstream inputs plus locally rebuilt SQLite",
            },
            indent=2,
        )
        + "\n"
    )
    print("Patched Python runtime verified; original notices and derivative receipt retained.")


if __name__ == "__main__":
    main()
