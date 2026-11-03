"""Cross-language pin: the wire program vocabulary is the Java enum, verbatim.

Parsed from ``BodyProgramId.java`` rather than restated, for the reason the launcher parses
``ControllerFault.java`` rather than restating it: a hand-written copy of a Java enum drifts
silently, and here the drift would be a program the service can name and the authority cannot
decode (or the reverse), which is exactly the failure ``requireExactKeys`` and ``requireEnum``
exist to make impossible.
"""

from __future__ import annotations

import re
from pathlib import Path

from neurocraft_protocol import ProgramId

ROOT = Path(__file__).resolve().parents[2]
BODY_PROGRAM_ID = ROOT / "mod/src/main/java/dev/neurocraft/fly/body/BodyProgramId.java"


def _java_program_names() -> tuple[str, ...]:
    source = BODY_PROGRAM_ID.read_text(encoding="utf-8")
    body = source.split("public enum BodyProgramId {", 1)[1].split("}", 1)[0]
    return tuple(re.findall(r"^\s*([A-Z][A-Z_]*)\s*,?\s*$", body, flags=re.MULTILINE))


def test_program_vocabulary_matches_the_java_enum() -> None:
    java_names = _java_program_names()
    # Parse sanity: an empty or truncated parse would make the equality claim vacuous.
    assert len(java_names) == 10, java_names
    assert java_names[0] == "HOVER"
    assert "ESCAPE_JUMP" in java_names

    assert {member.value for member in ProgramId} == set(java_names)
    assert {member.name for member in ProgramId} == set(java_names)


def test_program_wire_values_are_the_java_constant_names_not_identifiers() -> None:
    """``_identifier`` forbids underscores, so these can only travel as an enum."""

    for member in ProgramId:
        assert member.value == member.name
        assert member.value.isupper()
