"""Exercise notebook-run receipts without importing any optional Jupyter dependency."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import run_research_notebooks as runner


class CellFailure(Exception):
    """Stand-in for an exception raised after executing a notebook cell."""


class FakeNotebook:
    def __init__(self) -> None:
        self.cells = [
            SimpleNamespace(
                cell_type="code",
                outputs=["stale"],
                execution_count=99,
                metadata={"execution": {"stale": True}},
            ),
            SimpleNamespace(cell_type="code", outputs=["stale"], execution_count=100, metadata={}),
        ]


@pytest.fixture
def facades(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    root = tmp_path / "source"
    (root / "notebooks").mkdir(parents=True)
    for name in runner.NOTEBOOKS:
        (root / "notebooks" / name).write_text(json.dumps({"name": name}))
    monkeypatch.setattr(runner, "ROOT", root)
    observed = SimpleNamespace(
        root=root,
        output=tmp_path / "executed",
        managers=[],
        executions=[],
        failure=None,
        failure_on=0,
        save_failure=False,
        render_images=True,
    )

    class KernelManager:
        def __init__(self, **kwargs: Any) -> None:
            self.has_kernel = False
            self.shutdown_calls: list[bool] = []
            observed.managers.append(self)

        def shutdown_kernel(self, *, now: bool) -> None:
            self.shutdown_calls.append(now)
            self.has_kernel = False

    class Client:
        def __init__(self, notebook: FakeNotebook, *, km: KernelManager, **kwargs: Any) -> None:
            self.notebook = notebook
            self.km = km

        def execute(self, *, env: dict[str, str], cleanup_kc: bool) -> None:
            assert cleanup_kc
            assert env["MPLBACKEND"] == "Agg"
            assert all(
                not cell.outputs and cell.execution_count is None for cell in self.notebook.cells
            )
            assert all("execution" not in cell.metadata for cell in self.notebook.cells)
            receipt = json.loads((observed.output / "receipt.json").read_text())
            current = receipt["notebooks"][-1]
            source = root / "notebooks" / current["path"]
            assert current["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
            assert current["status"] == "running"
            observed.executions.append(current["path"])
            self.km.has_kernel = True
            self.notebook.cells[0].execution_count = 1
            self.notebook.cells[0].outputs = [
                {"output_type": "stream", "name": "stdout", "text": "fresh result before failure"}
            ]
            if observed.failure is not None and len(observed.executions) - 1 == observed.failure_on:
                self.notebook.cells[1].outputs = [{"output_type": "error", "ename": "deliberate"}]
                raise observed.failure
            self.notebook.cells[1].execution_count = 2
            self.notebook.cells[1].outputs = [
                {"output_type": "display_data", "data": {"image/png": "test-image"}, "metadata": {}}
                if observed.render_images
                else {"output_type": "stream", "name": "stdout", "text": "figure repr only"}
            ]

    def write(notebook: FakeNotebook, destination: Path) -> None:
        if observed.save_failure:
            raise OSError("deliberate disk failure")
        destination.write_text(json.dumps({"cells": [vars(cell) for cell in notebook.cells]}))

    modules = {
        "nbformat": SimpleNamespace(
            reads=lambda raw, as_version: FakeNotebook(),
            validate=lambda notebook: None,
            write=write,
        ),
        "nbclient": SimpleNamespace(NotebookClient=Client),
        "jupyter_client.kernelspec": SimpleNamespace(KernelSpecManager=lambda **kwargs: kwargs),
        "jupyter_client.manager": SimpleNamespace(KernelManager=KernelManager),
    }
    monkeypatch.setattr(
        runner, "importlib", SimpleNamespace(import_module=lambda name: modules[name])
    )
    return observed


@pytest.mark.parametrize(
    "exception,status",
    [(KeyboardInterrupt(), "interrupted"), (CellFailure("cell failed"), "failed")],
)
def test_failure_retains_current_source_partial_results_and_stops_kernel(
    facades: SimpleNamespace, exception: BaseException, status: str
) -> None:
    facades.failure = exception
    facades.failure_on = 1
    with pytest.raises(type(exception)) as raised:
        runner.execute(facades.output)
    assert raised.value is exception
    receipt = json.loads((facades.output / "receipt.json").read_text())
    assert receipt["status"] == status
    assert receipt["error_type"] == type(exception).__name__
    assert receipt["current_notebook"] == runner.NOTEBOOKS[1]
    assert len(receipt["notebooks"]) == 2
    assert receipt["notebooks"][0]["status"] == "passed"
    record = receipt["notebooks"][1]
    assert record["status"] == status
    assert record["partial_execution"]
    assert record["executed_code_cells"] == 1
    assert record["error_cell_indices_zero_based"] == [1]
    source = facades.root / "notebooks" / runner.NOTEBOOKS[1]
    assert record["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    partial = facades.output / record["executed_path"]
    assert record["executed_sha256"] == hashlib.sha256(partial.read_bytes()).hexdigest()
    cells = json.loads(partial.read_text())["cells"]
    assert cells[0]["outputs"][0]["text"] == "fresh result before failure"
    assert cells[1]["outputs"][0]["ename"] == "deliberate"
    assert all(manager.shutdown_calls == [True] for manager in facades.managers)
    assert all(not manager.has_kernel for manager in facades.managers)
    assert not (facades.output / runner.NOTEBOOKS[2]).exists()


def test_completed_notebooks_report_passed_and_are_not_overwritten(
    facades: SimpleNamespace,
) -> None:
    receipt = runner.execute(facades.output)
    assert receipt["status"] == "passed"
    assert len(receipt["notebooks"]) == 3
    assert all(
        record["status"] == "passed"
        and record["executed_code_cells"] == 2
        and record["image_output_count"] == 1
        and not record["partial_execution"]
        for record in receipt["notebooks"]
    )
    before = {path.name: path.read_bytes() for path in facades.output.iterdir()}
    with pytest.raises(FileExistsError):
        runner.execute(facades.output)
    assert before == {path.name: path.read_bytes() for path in facades.output.iterdir()}
    assert len(facades.executions) == 3
    assert all(manager.shutdown_calls == [True] for manager in facades.managers)


def test_partial_save_failure_preserves_execution_exception_and_cleanup(
    facades: SimpleNamespace,
) -> None:
    exception = KeyboardInterrupt()
    facades.failure = exception
    facades.save_failure = True
    with pytest.raises(KeyboardInterrupt) as raised:
        runner.execute(facades.output)
    assert raised.value is exception
    receipt = json.loads((facades.output / "receipt.json").read_text())
    assert receipt["status"] == "interrupted"
    record = receipt["notebooks"][0]
    assert record["partial_save_error_type"] == "OSError"
    assert "executed_path" not in record
    assert facades.managers[0].shutdown_calls == [True]


def test_completed_cells_without_inline_figure_fail_with_saved_output(
    facades: SimpleNamespace,
) -> None:
    facades.render_images = False
    with pytest.raises(runner.NotebookRenderingError, match="without an inline"):
        runner.execute(facades.output)
    receipt = json.loads((facades.output / "receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert len(receipt["notebooks"]) == 1
    record = receipt["notebooks"][0]
    assert record["error_type"] == "NotebookRenderingError"
    assert record["image_output_count"] == 0
    assert record["executed_code_cells"] == 2
    assert not record["partial_execution"]
    assert (facades.output / record["executed_path"]).is_file()
    assert facades.managers[0].shutdown_calls == [True]
