from __future__ import annotations

import pytest
from neurocraft_controller import PhysicalFeatureEncoder

from .conftest import make_sensors


def test_feature_encoder_is_exact_physical_normalization() -> None:
    encoder = PhysicalFeatureEncoder()
    encoded = encoder.encode(make_sensors(forward=3.0, left=1.5, right=4.5, light=6))

    assert dict(encoded.values) == {
        "forward_clearance_fraction": 0.5,
        "left_clearance_fraction": 0.25,
        "right_clearance_fraction": 0.75,
        "local_light_fraction": 0.4,
    }
    assert len(encoder.identity_sha256) == 64
    assert len(encoded.identity_sha256) == 64


def test_feature_encoder_rejects_raw_mapping_bypass() -> None:
    encoder = PhysicalFeatureEncoder()
    with pytest.raises(TypeError, match="SensorFrame"):
        encoder.encode({"left": 1.0})  # type: ignore[arg-type]
