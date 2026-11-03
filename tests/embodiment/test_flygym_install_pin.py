"""What this lane may import, and at exactly which version.

FlyGym is an optional extra the core lane deliberately does not install, so every
FlyGym call in ``flygym_body`` is written from FlyGym 2.1.0's documentation and none
of it has ever been executed here. Not one MuJoCo step has run. That is the honest
state of this lane, and it is the reason the pin is enforced rather than recorded:
there is no accumulated evidence that any other version behaves the same way, because
there is no accumulated evidence at all.

MEDIUM-1 from the sweep: ``REQUIRED_FLYGYM_VERSION`` was compared against exactly one
thing, the string in ``pyproject.toml``, by a test -- string to string, never to an
imported package. ``require_flygym`` read ``package_version("flygym")`` and wrote
whatever came back straight into the identity's provenance. A 2.0 or a 3.0 install
would have been driven with a 2.1.0 API surface, and the mismatch would have been
faithfully hashed into a provenance field nobody was checking.
"""

from __future__ import annotations

import importlib.util
import sys
import tomllib
import types

import pytest
from neurocraft_connectome import RuntimeGraph
from neurocraft_embodiment import (
    INSTALL_COMMAND,
    REQUIRED_FLYGYM_VERSION,
    FlyGymEmbodiment,
    FlyGymUnavailableError,
)
from neurocraft_embodiment import flygym_body as body_module

from .conftest import REPO_ROOT, make_fixture_decoder

FLYGYM_INSTALLED = importlib.util.find_spec("flygym") is not None


def test_module_imports_without_flygym_installed() -> None:
    """The seam must be usable, lintable and type-checkable with no MuJoCo present."""

    assert "neurocraft_embodiment.flygym_body" in sys.modules
    assert "flygym" not in sys.modules or FLYGYM_INSTALLED


@pytest.mark.skipif(FLYGYM_INSTALLED, reason="flygym is installed in this environment")
def test_missing_flygym_gives_an_actionable_error(graph: RuntimeGraph) -> None:
    with pytest.raises(FlyGymUnavailableError) as excinfo:
        FlyGymEmbodiment(make_fixture_decoder(graph))
    message = str(excinfo.value)
    assert INSTALL_COMMAND in message
    assert "imageio-ffmpeg" in message
    assert "do not fold it into the core lane" in message


def test_install_pin_matches_the_declarative_pyproject_pin() -> None:
    """The runtime error message and the recorded pin cannot drift apart.

    Necessary and, on its own, nowhere near sufficient: both sides of this comparison
    are strings this repository wrote. It says the pin is stated consistently. It says
    nothing whatever about what is installed, which is what the next two tests are for.
    """

    raw = (REPO_ROOT / "packages/embodiment-runtime/pyproject.toml").read_bytes()
    pin = tomllib.loads(raw.decode("utf-8"))["tool"]["neurocraft"]["optional-embodiment"]["flygym"]
    assert pin["version"] == REQUIRED_FLYGYM_VERSION
    assert pin["license"] == "Apache-2.0"
    assert pin["resolved-by-uv"] is False
    assert REQUIRED_FLYGYM_VERSION in INSTALL_COMMAND


def test_flygym_is_absent_from_the_frozen_lock() -> None:
    """The compliance guarantee, asserted rather than asserted-in-prose."""

    lock = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
    for forbidden in ("flygym", "mujoco", "imageio-ffmpeg", "numba", "llvmlite"):
        assert f'name = "{forbidden}"' not in lock


def install_named_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put empty modules named ``flygym*`` and ``mujoco`` in ``sys.modules``.

    Stated plainly, because the point of this whole file is that names are not
    evidence: these are empty ``ModuleType`` objects with no FlyGym code in them at
    all. They exist so ``require_flygym``'s ``import`` statements resolve and the
    version comparison after them can be reached. Nothing here runs FlyGym, and no
    result obtained through them is a FlyGym or MuJoCo result.
    """

    package = types.ModuleType("flygym")
    monkeypatch.setitem(sys.modules, "flygym", package)
    for leaf in ("anatomy", "compose", "simulation"):
        submodule = types.ModuleType(f"flygym.{leaf}")
        monkeypatch.setitem(sys.modules, f"flygym.{leaf}", submodule)
        monkeypatch.setattr(package, leaf, submodule, raising=False)
    # ``require_flygym`` also imports ``flygym.utils.math`` for ``Rotation3D``
    # (EMBODIMENT-02 defect 1); both parent and leaf must resolve, and the class
    # attribute must exist, for the version comparison to be reachable.
    utils = types.ModuleType("flygym.utils")
    monkeypatch.setitem(sys.modules, "flygym.utils", utils)
    monkeypatch.setattr(package, "utils", utils, raising=False)
    utils_math = types.ModuleType("flygym.utils.math")
    utils_math.Rotation3D = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "flygym.utils.math", utils_math)
    monkeypatch.setattr(utils, "math", utils_math, raising=False)
    mujoco = types.ModuleType("mujoco")
    mujoco.__version__ = "0.0.0-placeholder"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mujoco", mujoco)


@pytest.mark.skipif(FLYGYM_INSTALLED, reason="flygym is installed in this environment")
def test_require_flygym_refuses_a_version_that_is_not_the_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MEDIUM-1: the constant existed and nothing ever compared it to an install."""

    install_named_placeholders(monkeypatch)
    monkeypatch.setattr(body_module, "package_version", lambda _name: "3.0.0")
    with pytest.raises(FlyGymUnavailableError) as excinfo:
        body_module.require_flygym()
    message = str(excinfo.value)
    assert REQUIRED_FLYGYM_VERSION in message
    assert "3.0.0" in message
    assert INSTALL_COMMAND in message


@pytest.mark.skipif(FLYGYM_INSTALLED, reason="flygym is installed in this environment")
def test_require_flygym_accepts_exactly_the_pin_and_records_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal must be a comparison, not a blanket rejection.

    A guard that refused everything would pass the test above and make the extra
    uninstallable. This is the control: at the pin, the call returns, and the version
    it records in provenance is the one it resolved rather than the constant it was
    comparing against -- so a future drift is still visible in the hash.
    """

    install_named_placeholders(monkeypatch)
    monkeypatch.setattr(body_module, "package_version", lambda _name: REQUIRED_FLYGYM_VERSION)
    modules = body_module.require_flygym()
    assert modules.version == REQUIRED_FLYGYM_VERSION
    assert modules.mujoco_version == "0.0.0-placeholder"


def test_no_pin_is_claimed_for_mujoco() -> None:
    """The absent guard, recorded rather than left as a silence.

    ``mujoco`` arrives as a FlyGym dependency and this repository declares no version
    for it, so there is nothing here to compare an install against. Inventing a
    constant would be a claim rather than a check. Its version is recorded in
    provenance and left as a record; this asserts that no such pin has quietly
    appeared without the comparison that would make it mean something.
    """

    names = {name for name in vars(body_module) if "MUJOCO" in name.upper()}
    assert names == set()
