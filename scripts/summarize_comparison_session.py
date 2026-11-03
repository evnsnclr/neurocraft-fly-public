#!/usr/bin/env python3
"""Summarize comparison diagnostics without joining command numbers across epochs.

Directory/sidecar attribution is diagnostic, not an authenticated replay trace.
Prepared epoch metadata alone never establishes an active controller or an applied body program.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_interactive_session import (
    DiagnosticError,
    _check_identity,
    _integer,
    _latency,
    _programs,
    _records,
)

Json = dict[str, Any]
LANES = {"original": "faithful", "shuffled_weight": "shuffled_weight", "no_input": "no_input"}
HASH_FIELDS = {
    "graph_hash",
    "adapter_hash",
    "dynamics_hash",
    "decoder_hash",
    "observation_schema_hash",
}
IDENTITY_FIELDS = HASH_FIELDS | {"run_id", "entity_id", "session_epoch"}
ENTRY_FIELDS = {"name", "identity", "control_kind", "control_sha256", "seed"}


def _bounded(path: Path) -> tuple[Json, str]:
    def unique(pairs: list[tuple[str, Any]]) -> Json:
        result: Json = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        with path.open("rb") as stream:
            data = stream.read(16_385)
        if len(data) > 16_384:
            raise ValueError("exceeds 16 KiB")
        value = json.loads(data.decode("utf-8"), object_pairs_hook=unique)
        if not isinstance(value, dict):
            raise ValueError("expected object")
    except (OSError, ValueError, RecursionError) as exc:
        raise DiagnosticError(f"{path.name}: {exc}") from exc
    return value, sha256(data).hexdigest()


def _identity(value: object) -> Json:
    if not isinstance(value, dict) or set(value) != IDENTITY_FIELDS:
        raise DiagnosticError("invalid identity fields")
    epoch = _integer(value, "session_epoch", "identity")
    if epoch > 2**63 - 1:
        raise DiagnosticError("identity epoch exceeds Java long")
    for key in HASH_FIELDS:
        if not isinstance(value[key], str) or re.fullmatch(r"[0-9a-f]{64}", value[key]) is None:
            raise DiagnosticError(f"invalid identity {key}")
    if any(not isinstance(value[key], str) or not value[key] for key in ("run_id", "entity_id")):
        raise DiagnosticError("invalid run or entity identity")
    return value


def _catalog(directory: Path, session: Json) -> tuple[dict[str, Json], str]:
    catalog, digest = _bounded(directory / "lane-catalog.json")
    if digest != session.get("lane_catalog_sha256"):
        raise DiagnosticError("catalog digest differs from the launcher session record")
    if (
        set(catalog) != {"schema", "initial_identity", "lanes"}
        or catalog["schema"] != "neurocraft.lane_catalog.v1"
    ):
        raise DiagnosticError("unsupported catalog")
    initial = _identity(catalog["initial_identity"])
    if initial != _identity(session.get("identity")):
        raise DiagnosticError("initial catalog identity differs from session")
    if any(session.get(key) != initial[key] for key in ("run_id", "entity_id", "session_epoch")):
        raise DiagnosticError("session envelope disagrees with its full identity")
    _integer(session, "session_epoch", "session")
    rows = catalog["lanes"]
    if not isinstance(rows, list) or len(rows) != 3:
        raise DiagnosticError("catalog requires three lanes")
    lanes: dict[str, Json] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != ENTRY_FIELDS:
            raise DiagnosticError("invalid catalog entry")
        name = row["name"]
        if not isinstance(name, str) or name not in LANES or name in lanes:
            raise DiagnosticError("invalid or duplicate catalog lane")
        identity = _identity(row["identity"])
        if any(identity[key] != initial[key] for key in IDENTITY_FIELDS - {"dynamics_hash"}):
            raise DiagnosticError("lane changes a fixed identity field")
        if row["control_kind"] != LANES[name]:
            raise DiagnosticError("control kind differs from lane")
        if name == "original":
            if identity != initial or row["control_sha256"] is not None or row["seed"] is not None:
                raise DiagnosticError("invalid original lane")
        else:
            control, seed = row["control_sha256"], row["seed"]
            if not isinstance(control, str) or re.fullmatch(r"[0-9a-f]{64}", control) is None:
                raise DiagnosticError("invalid control hash")
            if (
                not isinstance(seed, str)
                or re.fullmatch(r"0|[1-9][0-9]{0,19}", seed) is None
                or int(seed) >= 2**64
            ):
                raise DiagnosticError("invalid control seed")
        lanes[name] = row
    if len({row["identity"]["dynamics_hash"] for row in lanes.values()}) != 3:
        raise DiagnosticError("control dynamics must differ")
    return lanes, digest


def summarize_comparison(directory: Path) -> Json:
    session, _ = _bounded(directory / "session.json")
    if session.get("comparison_lanes") is not True:
        raise DiagnosticError("session is not labeled as a comparison")
    lanes, digest = _catalog(directory, session)
    base_epoch = session["identity"]["session_epoch"]
    epochs_dir = directory / "epochs"
    if not epochs_dir.is_dir():
        raise DiagnosticError("epoch diagnostics directory is unavailable")
    epochs: list[Json] = []
    for path in sorted(epochs_dir.iterdir(), key=lambda item: item.name):
        if (
            path.is_symlink()
            or not path.is_dir()
            or re.fullmatch(r"0|[1-9][0-9]*", path.name) is None
        ):
            raise DiagnosticError("unexpected epoch directory entry")
        epoch = int(path.name)
        metadata, metadata_digest = _bounded(path / "epoch.json")
        if (
            metadata.get("schema") != "neurocraft.lane_epoch.v1"
            or metadata.get("diagnostic_only") is not True
        ):
            raise DiagnosticError("unsupported epoch metadata")
        name = metadata.get("name")
        if not isinstance(name, str) or name not in lanes:
            raise DiagnosticError("epoch names an unknown lane")
        template = lanes[name]
        identity = _identity(metadata.get("identity"))
        expected = {**template["identity"], "session_epoch": epoch}
        if (
            identity != expected
            or epoch < base_epoch
            or (epoch == base_epoch and name != "original")
        ):
            raise DiagnosticError("epoch identity disagrees with catalog or directory")
        for key in ENTRY_FIELDS - {"identity"}:
            if metadata.get(key) != template[key]:
                raise DiagnosticError(f"epoch {key} disagrees with catalog")
        if metadata.get("controller_profile") != session.get("controller_profile"):
            raise DiagnosticError("epoch controller profile disagrees with session")
        warnings = [
            f"{filename}: unavailable (missing file)"
            for filename in ("program-decisions.ndjson", "action-latency.ndjson")
            if not (path / filename).exists()
        ]
        decisions = _records(path / "program-decisions.ndjson", warnings)
        latency = _records(path / "action-latency.ndjson", warnings)
        _check_identity(
            (decisions or []) + (latency or []),
            {"run_id": identity["run_id"], "controller_profile": session["controller_profile"]},
            f"epoch {epoch}",
        )
        programs = _programs(decisions, warnings)
        epochs.append(
            {
                "epoch_metadata": metadata,
                "epoch_metadata_sha256": metadata_digest,
                "recorded_applied_program": programs is not None
                and programs["applied_receipts"] > 0,
                "programs": programs,
                "compute_latency": _latency(latency),
                "warnings": warnings,
            }
        )
    epochs.sort(key=lambda row: row["epoch_metadata"]["identity"]["session_epoch"])
    return {
        "schema": "neurocraft.comparison_diagnostic_summary.v1",
        "diagnostic_only": True,
        "claim_boundary": (
            "Directory and sidecar attribution; not authenticated replay or matched trials. "
            "Prepared metadata alone does not establish an active lane. "
            "Command sequences stay per epoch."
        ),
        "run_id": session.get("run_id"),
        "catalog_sha256": digest,
        "recorded_exit_reason": session.get("exit_reason"),
        "epochs": epochs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize_comparison(args.session_directory)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
