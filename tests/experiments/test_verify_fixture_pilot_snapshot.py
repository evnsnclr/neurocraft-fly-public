from __future__ import annotations

from pathlib import Path

import pytest
from neurocraft_experiments import pilot

from scripts import verify_fixture_pilot_snapshot as snapshot_verifier

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "experiments/preregistration/fixture-physics-pilot-v3.json"
SNAPSHOT = ROOT / "experiments/results/fixture-physics-pilot-v3.json"


def test_frozen_snapshot_bytes_envelope_and_environment_are_pinned() -> None:
    digest, envelope = snapshot_verifier.verify_snapshot_integrity(SNAPSHOT)

    assert digest == snapshot_verifier.EXPECTED_SNAPSHOT_SHA256
    assert envelope["result_sha256"] == snapshot_verifier.EXPECTED_RESULT_SHA256
    assert envelope["execution_environment"] == snapshot_verifier.RECORDED_EXECUTION_ENVIRONMENT


def test_snapshot_integrity_rejects_any_byte_change(tmp_path: Path) -> None:
    corrupted = tmp_path / "fixture-physics-pilot-v3.json"
    corrupted.write_bytes(SNAPSHOT.read_bytes() + b" ")

    with pytest.raises(ValueError, match="snapshot SHA-256 mismatch"):
        snapshot_verifier.verify_snapshot_integrity(corrupted)


@pytest.mark.parametrize(
    "environment",
    [
        {
            "platform": "Linux-6.11-x86_64-with-glibc2.39",
            "python_implementation": "CPython",
            "python_version": "3.13.5",
        },
        dict(snapshot_verifier.RECORDED_EXECUTION_ENVIRONMENT),
    ],
    ids=["cross_platform", "recorded_environment"],
)
def test_verification_never_reexecutes_the_pilot_on_any_platform(
    monkeypatch: pytest.MonkeyPatch, environment: dict[str, str]
) -> None:
    """The historicized boundary, pinned from both sides.

    The recorded environment used to rebuild the whole result core and compare it bit for bit.
    It no longer does, and the point of driving the recorded environment here is that it is no
    longer treated differently: the same recorded bytes are verified, and a rebuild would be a
    defect. Every byte pin above still holds, on both platforms.
    """

    def unexpected_rebuild(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the frozen-snapshot verifier rebuilt the pilot result")

    monkeypatch.setattr(pilot, "build_pilot_result", unexpected_rebuild)

    outcome = snapshot_verifier.verify_frozen_snapshot(
        SNAPSHOT,
        PROTOCOL,
        repository_root=ROOT,
        current_environment=environment,
    )

    assert outcome.snapshot_sha256 == snapshot_verifier.EXPECTED_SNAPSHOT_SHA256
    assert outcome.result_sha256 == snapshot_verifier.EXPECTED_RESULT_SHA256
    assert outcome.current_environment == environment
    assert outcome.recorded_environment is (
        environment == snapshot_verifier.RECORDED_EXECUTION_ENVIRONMENT
    )


def test_the_historicized_result_core_names_the_tree_it_last_reproduced_on() -> None:
    """A demotion to history has to say which history: a bare 64-hex string is not evidence."""

    assert len(pilot.RESULT_CORE_TREE_COMMIT) == 40
    assert set(pilot.RESULT_CORE_TREE_COMMIT) <= set("0123456789abcdef")
    assert pilot.RESULT_CORE_TREE_COMMIT != pilot.IMPLEMENTATION_HASH_TREE_COMMIT
