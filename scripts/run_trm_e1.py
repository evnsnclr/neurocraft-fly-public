#!/usr/bin/env python3
"""Execute the frozen TRM E1 protocol (Structure vs. Learning v1).

Runs only under the frozen preregistration: every experimental choice lives as
a constant in ``neurocraft_experiments.trm_e1``, whose bytes the freeze pins.
This entry point adds nothing but a path, a worker count, and a smoke limit.
"""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

from neurocraft_experiments.trm_e1 import PROTOCOL_ID, execute_protocol

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "experiments" / "results" / "trm-e1-v1.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="smoke-run cell cap; a limited run writes a partial document only",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.output.resolve() == DEFAULT_OUTPUT.resolve():
        parser.error(
            "--limit writes a partial document and would permanently occupy the "
            "immutable canonical result path; pass an explicit --output"
        )
    document = execute_protocol(
        REPOSITORY_ROOT, args.output, workers=args.workers, limit=args.limit
    )
    digest = sha256(args.output.read_bytes()).hexdigest()
    summary = {
        "cell_count": document["cell_count"],
        "output": str(args.output),
        "output_sha256": digest,
        "protocol_id": PROTOCOL_ID,
    }
    json.dump(summary, sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
