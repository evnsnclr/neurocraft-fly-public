from __future__ import annotations

import pytest
from neurocraft_experiments.identity import DataTier, RunMode, TrialIdentity, canonical_sha256

HASH = "a" * 64


def make_trial(**overrides: object) -> TrialIdentity:
    values: dict[str, object] = {
        "run_id": "trial-001",
        "mode": RunMode.FAITHFUL,
        "data_tier": DataTier.MALECNS_FIXTURE,
        "seed": 7,
        "graph_sha256": HASH,
        "adapter_sha256": HASH,
        "dynamics_sha256": HASH,
        "decoder_sha256": HASH,
        "arena_sha256": HASH,
    }
    values.update(overrides)
    return TrialIdentity(**values)  # type: ignore[arg-type]


def test_identity_hash_is_stable_and_mode_bound() -> None:
    first = make_trial()
    assert first.identity_sha256 == make_trial().identity_sha256
    assert first.identity_sha256 != make_trial(mode=RunMode.LESION).identity_sha256


def test_synthetic_and_faithful_labels_cannot_cross() -> None:
    with pytest.raises(ValueError, match="faithful"):
        make_trial(data_tier=DataTier.SYNTHETIC)
    with pytest.raises(ValueError, match="synthetic mode"):
        make_trial(mode=RunMode.SYNTHETIC)


def test_hashes_and_finite_json_fail_closed() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        make_trial(graph_sha256="unpinned")
    with pytest.raises(ValueError, match="non-finite"):
        canonical_sha256({"activity": float("nan")})
