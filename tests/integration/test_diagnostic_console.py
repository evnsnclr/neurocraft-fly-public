"""The opt-in local console never dispatches partial or multiline requests."""

import json
from pathlib import Path

import pytest

from scripts.run_interactive_demo import read_diagnostic_commands


def test_only_complete_records_advance_and_repeated_polls_do_not_repeat(tmp_path: Path) -> None:
    path = tmp_path / "commands.ndjson"
    first = json.dumps({"command": "neurocraftfly status"}) + "\n"
    second = json.dumps({"command": "tp Observer 0 65 -4"})
    path.write_text(first + second)
    cursor, commands = read_diagnostic_commands(path, 0)
    assert cursor == len(first.encode())
    assert commands == ["neurocraftfly status"]
    assert read_diagnostic_commands(path, cursor) == (cursor, [])
    with path.open("a") as stream:
        stream.write("\n")
    final, commands = read_diagnostic_commands(path, cursor)
    assert commands == ["tp Observer 0 65 -4"]
    assert read_diagnostic_commands(path, final) == (final, [])


@pytest.mark.parametrize(
    "record",
    [
        {"command": "say one\nsay two"},
        {"command": "say one\rsay two"},
        {"command": " say padded"},
        {"command": "/stop"},
        {"command": ""},
        {"command": 2},
        {"command": "x" * 513},
        {"command": "list", "other": True},
        [],
    ],
)
def test_malformed_records_are_refused(tmp_path: Path, record: object) -> None:
    path = tmp_path / "commands.ndjson"
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError):
        read_diagnostic_commands(path, 0)


def test_poll_and_file_size_are_bounded_and_truncation_refused(tmp_path: Path) -> None:
    path = tmp_path / "commands.ndjson"
    path.write_text(' {"command":"list"}\n' * 20)
    cursor, commands = read_diagnostic_commands(path, 0)
    assert len(commands) == 16
    assert len(read_diagnostic_commands(path, cursor)[1]) == 4
    path.write_text("")
    with pytest.raises(ValueError, match="truncated"):
        read_diagnostic_commands(path, cursor)
    path.write_bytes(b"x" * 65_537)
    with pytest.raises(ValueError, match="64 KiB"):
        read_diagnostic_commands(path, 0)
