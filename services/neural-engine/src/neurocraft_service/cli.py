"""Command-line entrypoint for the bounded loopback service."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from collections.abc import Sequence
from pathlib import Path

from neurocraft_protocol import RunIdentity, parse_json_frame

from .server import ServiceConfig, serve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the NeuroCraft Fly loopback neural service")
    parser.add_argument(
        "--identity-json",
        required=True,
        type=Path,
        help="JSON file containing the strict RunIdentity object (never the run token)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="loopback IP literal")
    parser.add_argument("--port", default=8765, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    token = os.environ.get("NEUROCRAFT_RUN_TOKEN")
    if token is None:
        raise SystemExit("NEUROCRAFT_RUN_TOKEN must be supplied out of band")
    identity = RunIdentity.from_dict(parse_json_frame(args.identity_json.read_bytes()))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        asyncio.run(
            serve(
                ServiceConfig(
                    identity=identity,
                    token=token,
                    host=args.host,
                    port=args.port,
                )
            )
        )
    except KeyboardInterrupt:
        return 130
    return 0
