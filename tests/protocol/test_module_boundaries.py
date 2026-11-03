"""The protocol package's own modules import acyclically.

Before the dispatch was extracted, `messages` imported `program_messages` at the bottom of the
file and `program_messages` imported `messages` back. That cycle resolved only because the
package's `__init__` happened to import `messages` first; reordering those two lines, or any
entry point that reached a frame module without the package `__init__` running, would have hit a
partially-initialized module. The dispatch that has to know both frame modules now sits above
both, so nothing imports its own importer.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[2] / "packages/protocol/src/neurocraft_protocol"
FRAME_MODULES = ("neurocraft_protocol.messages", "neurocraft_protocol.program_messages")


def _intra_package_imports(module: Path) -> set[str]:
    """Return the sibling module names ``module`` imports from this package."""

    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    edges: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            edges.add(node.module.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is None:
            # ``from . import sibling`` names the sibling modules as aliases.
            for alias in node.names:
                edges.add(alias.name)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module is not None
            and node.module.startswith("neurocraft_protocol.")
        ):
            # ``from neurocraft_protocol.sibling import X`` is the absolute spelling.
            edges.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("neurocraft_protocol."):
                    edges.add(alias.name.split(".")[1])
    return edges


def _import_graph() -> dict[str, set[str]]:
    return {
        path.stem: _intra_package_imports(path)
        for path in sorted(PACKAGE_DIR.glob("*.py"))
        if path.stem != "__init__"
    }


def test_the_protocol_package_has_no_intra_package_import_cycle() -> None:
    graph = _import_graph()
    # Parse sanity: an empty graph would make the acyclicity claim vacuous.
    assert "messages" in graph and "program_messages" in graph and "dispatch" in graph
    assert "messages" in graph["program_messages"]
    assert {"messages", "program_messages"} <= graph["dispatch"]

    visiting: set[str] = set()
    done: set[str] = set()

    def walk(name: str, trail: tuple[str, ...]) -> None:
        if name in done or name not in graph:
            return
        if name in visiting:
            raise AssertionError(f"import cycle: {' -> '.join((*trail, name))}")
        visiting.add(name)
        for target in sorted(graph[name]):
            walk(target, (*trail, name))
        visiting.discard(name)
        done.add(name)

    for module in sorted(graph):
        walk(module, ())


def test_neither_frame_module_imports_the_other_back() -> None:
    """The specific edge that was the cycle, named so a reintroduction is unambiguous."""

    graph = _import_graph()
    assert "program_messages" not in graph["messages"]


@pytest.mark.parametrize("module", FRAME_MODULES)
def test_each_frame_module_imports_first_in_a_fresh_interpreter(module: str) -> None:
    """Import each frame module as the process's first protocol import.

    This passes under the package ``__init__``'s own ordering either way, so on its own it is
    weaker than the acyclicity check above; it is here because it is what an unlucky import
    order actually looks like from the outside, and it costs one subprocess.
    """

    completed = subprocess.run(
        [sys.executable, "-c", f"import {module} as m; print(m.__name__)"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == module
