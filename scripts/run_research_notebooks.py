#!/usr/bin/env python3
"""Execute the three project notebooks using this interpreter in a fresh local kernel."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "01_meet_the_network.ipynb",
    "02_stimulate_and_observe.ipynb",
    "03_change_and_compare.ipynb",
)


class NotebookRenderingError(RuntimeError):
    """A completed notebook did not emit the promised visible figure."""


def _write_receipt(output: Path, report: dict[str, Any]) -> None:
    (output / "receipt.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _save_notebook(nbformat: Any, notebook: Any, output: Path, record: dict[str, Any]) -> None:
    executed = output / record["path"]
    nbformat.write(notebook, executed)
    record.update(
        {
            "executed_path": executed.name,
            "executed_sha256": hashlib.sha256(executed.read_bytes()).hexdigest(),
            "error_cell_indices_zero_based": [
                index
                for index, cell in enumerate(notebook.cells)
                if cell.cell_type == "code"
                and any(item.get("output_type") == "error" for item in cell.outputs)
            ],
            "image_output_count": sum(
                any(mime in item.get("data", {}) for mime in ("image/png", "image/svg+xml"))
                for cell in notebook.cells
                if cell.cell_type == "code"
                for item in cell.outputs
            ),
            "executed_code_cells": sum(
                cell.cell_type == "code" and cell.execution_count is not None
                for cell in notebook.cells
            ),
        }
    )


def execute(output: Path, *, timeout: int = 120) -> dict[str, Any]:
    """Retain outputs and a receipt; refuse to replace an earlier execution."""
    if timeout < 1:
        raise ValueError("timeout must be positive")
    nbformat = importlib.import_module("nbformat")
    nbclient = importlib.import_module("nbclient")
    kernelspec = importlib.import_module("jupyter_client.kernelspec")
    manager = importlib.import_module("jupyter_client.manager")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema": "neurocraft.research-notebooks.v1",
        "scope": "14-neuron, 22-edge fixture; synthetic stimulation; no Minecraft",
        "python": platform.python_version(),
        "status": "running",
        "notebooks": records,
    }
    try:
        _write_receipt(output, report)
        with tempfile.TemporaryDirectory(prefix="neurocraft-notebook-kernel-") as temporary:
            kernel_dir = Path(temporary) / "neurocraft-research"
            kernel_dir.mkdir()
            (kernel_dir / "kernel.json").write_text(
                json.dumps(
                    {
                        "argv": [
                            sys.executable,
                            "-m",
                            "ipykernel_launcher",
                            "-f",
                            "{connection_file}",
                        ],
                        "display_name": "NeuroCraft research",
                        "language": "python",
                    }
                ),
                encoding="utf-8",
            )
            for name in NOTEBOOKS:
                source = ROOT / "notebooks" / name
                record: dict[str, Any] = {"path": name, "status": "running"}
                records.append(record)
                report["current_notebook"] = name
                raw = source.read_bytes()
                record["source_sha256"] = hashlib.sha256(raw).hexdigest()
                _write_receipt(output, report)
                notebook = nbformat.reads(raw.decode("utf-8"), as_version=4)
                nbformat.validate(notebook)
                # Every run proves the cells anew, including notebooks with saved outputs.
                for cell in notebook.cells:
                    if cell.cell_type == "code":
                        cell.outputs = []
                        cell.execution_count = None
                    cell.metadata.pop("execution", None)
                km = manager.KernelManager(
                    kernel_name="neurocraft-research",
                    kernel_spec_manager=kernelspec.KernelSpecManager(kernel_dirs=[temporary]),
                )
                client = nbclient.NotebookClient(
                    notebook,
                    km=km,
                    timeout=timeout,
                    allow_errors=False,
                    record_timing=False,
                    resources={"metadata": {"path": str(ROOT)}},
                )
                env = dict(os.environ)
                env["MPLBACKEND"] = "Agg"
                env["NEUROCRAFT_RESEARCH_OUTPUT"] = str(output / "figures")
                env["MPLCONFIGDIR"] = str(output / ".matplotlib")
                try:
                    try:
                        # The provided manager makes lifecycle ownership explicit.
                        client.execute(env=env, cleanup_kc=True)
                    finally:
                        if km.has_kernel:
                            km.shutdown_kernel(now=True)
                except BaseException as exc:
                    record["status"] = (
                        "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                    )
                    record["error_type"] = type(exc).__name__
                    record["partial_execution"] = True
                    try:
                        _save_notebook(nbformat, notebook, output, record)
                    except Exception as save_error:
                        # Keep the execution exception authoritative if writing also fails.
                        record["partial_save_error_type"] = type(save_error).__name__
                    raise
                _save_notebook(nbformat, notebook, output, record)
                record["partial_execution"] = False
                if record["image_output_count"] < 1:
                    raise NotebookRenderingError(
                        f"{name} completed without an inline PNG/SVG figure"
                    )
                record["status"] = "passed"
                _write_receipt(output, report)
        report["status"] = "passed"
        report["figures"] = [
            {
                "path": str(path.relative_to(output)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted((output / "figures").glob("*"))
            if path.suffix in {".png", ".svg"}
        ]
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        # Detailed traceback remains in the caller's terminal and partial notebook.
        report["error_type"] = type(exc).__name__
        if records and records[-1]["status"] == "running":
            records[-1]["status"] = report["status"]
            records[-1]["error_type"] = type(exc).__name__
        raise
    finally:
        _write_receipt(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--timeout", type=int, default=120, help="seconds allowed per code cell")
    args = parser.parse_args()
    try:
        result = execute(args.output, timeout=args.timeout)
    except ModuleNotFoundError as exc:
        raise SystemExit("Notebook tools missing. Run uv sync --frozen --group research.") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
