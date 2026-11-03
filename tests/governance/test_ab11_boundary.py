from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_ROOTS = (
    ROOT / "packages",
    ROOT / "services",
    ROOT / "scripts",
    ROOT / "experiments",
    ROOT / "apps",
)
AB11_PACKAGE = ROOT / "tests" / "ab11"
# Matches an import of the AB-11 test package under either name pytest can give it
# ("tests.ab11" from the repository root, "ab11" from the tests directory). A string
# such as "tests/ab11" in a harness command line is deliberately not an import.
AB11_IMPORT = re.compile(r"^\s*(?:from|import)\s+[\w.]*\bab11\b", re.MULTILINE)
# A statement-level import is not the only way in: importlib.import_module,
# __import__, and pkgutil-style loaders all take the dotted module name as a plain
# string, which the statement regex above cannot see. No production source mentions
# the dotted name for any other reason (the scan below is what checks that), so the
# dotted name itself is the guard.
AB11_DOTTED_NAME = re.compile(r"\btests\.ab11\b")


def _production_sources() -> list[Path]:
    """Every production Python file: the named roots, plus repository-root modules."""

    packaged = [
        path
        for root in PRODUCTION_ROOTS
        if root.is_dir()
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    ]
    # A module added at the repository root would belong to none of the roots above.
    # There are none today; including the glob closes the gap before one appears.
    return sorted([*packaged, *ROOT.glob("*.py")])


def test_ab11_production_source_never_imports_the_mutant_package() -> None:
    """AB-11(governance boundary) — mutants must stay unreachable from production.

    Detector: this test — no file under packages/, services/, scripts/, experiments/,
    or apps/, and no repository-root module, may import the ``tests.ab11`` package
    under either module name, by import statement or by dynamic dotted-name lookup.
    """

    sources = _production_sources()
    assert len(sources) >= 20, "production source scan found too few files to be meaningful"

    offenders: list[str] = []
    for path in sources:
        text = path.read_text(encoding="utf-8")
        if AB11_IMPORT.search(text) or AB11_DOTTED_NAME.search(text):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_ab11_mutants_live_outside_every_code_identity_bundle() -> None:
    """AB-11(governance boundary) — the mutant package must stay outside hashed roots.

    Detector: this test — ``tests/ab11`` lives under ``tests/`` only, so it is outside
    ``replay_implementation_sha256`` and ``_code_bundle``; both scan patterns are also
    checked against known positive and negative cases so neither can pass vacuously.
    """

    assert (AB11_PACKAGE / "mutants.py").is_file()
    assert AB11_PACKAGE.is_relative_to(ROOT / "tests")
    assert not any(AB11_PACKAGE.is_relative_to(root) for root in PRODUCTION_ROOTS)

    # The AB-11 regexes must be able to fail, not merely never fire.
    assert AB11_IMPORT.search("from tests.ab11.mutants import MutantZeroPropagationEngine")
    assert AB11_IMPORT.search("import ab11.mutants")
    assert not AB11_IMPORT.search('COMMAND = ["pytest", "-q", "tests/ab11"]')
    assert AB11_DOTTED_NAME.search('importlib.import_module("tests.ab11.mutants")')
    assert AB11_DOTTED_NAME.search('__import__("tests.ab11")')
    assert not AB11_DOTTED_NAME.search('COMMAND = ["pytest", "-q", "tests/ab11"]')
