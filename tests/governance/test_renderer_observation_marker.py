"""The renderer-observation receipt must carry its ADR 0016 presentation-gain marker.

`runClientGameTest` writes `mod/build/receipts/fly-renderer-observation.json` beside a
PNG of the drawn fly. That PNG is the artifact most likely to be lifted out of this
repository into a figure or a message, and the body it shows is banked through a
declared display scaling of the yaw channel: the bank is `yaw_torque` divided by
`FlyAnimation.YAW_TORQUE_VISUAL_REFERENCE` and clamped, so a fully banked fly is
executing at least about 3.5% of its yaw authority (ADR 0016, D-069). The receipt used
to record only `automated` and `manual_gameplay_observation`, so the caveat did not
travel with the image (the gap ADR 0016 disclosed). These checks pin the marker that
closes it (D-074).

The receipt itself is a build output and is not tracked, so the guards here are on the
writer, on the Gradle task that refuses a receipt without the marker, and on the records.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WRITER = ROOT / "mod/src/gametest/java/dev/neurocraft/fly/gametest/FlyClientGameTest.java"
ANIMATION = ROOT / "mod/src/client/java/dev/neurocraft/fly/client/FlyAnimation.java"
BUILD_GRADLE = ROOT / "mod/build.gradle"
ADR = ROOT / "docs/adr/0016-motor-driven-client-animation.md"
HISTORICAL_RECEIPT = ROOT / "docs/receipts/RENDERER-CLIENT-01.json"

SCHEMA = "neurocraft.renderer_observation.v2"
MARKER_METHOD = re.compile(
    r"private static String presentationVisualGainJson\(\) \{(.*?)\n        \}", re.DOTALL
)


def _writer_source() -> str:
    return WRITER.read_text(encoding="utf-8")


def test_writer_emits_the_marker_at_a_bumped_schema_version() -> None:
    source = _writer_source()
    assert f'\\"schema\\": \\"{SCHEMA}\\"' in source
    assert "neurocraft.renderer_observation.v1" not in source, (
        "the marker changes the receipt's shape, so the schema version moves with it and "
        "the v1 receipt stays history"
    )
    assert '\\"bank_is_display_scaled\\": true' in source
    for key in (
        '\\"presentation_visual_gain\\": {',
        '\\"reference_amplitude\\":',
        '\\"max_bank_degrees\\":',
        '\\"gain\\":',
    ):
        assert key in source, key


def test_marker_numbers_are_read_from_the_declared_constants_not_restated() -> None:
    match = MARKER_METHOD.search(_writer_source())
    assert match is not None, "the presentation-gain marker is no longer built in one place"
    body = match.group(1)
    assert "FlyAnimation.YAW_TORQUE_VISUAL_REFERENCE" in body
    assert "FlyAnimation.MAX_BANK_DEGREES" in body
    for restated in ("0.035", "25.0", "28.5", "3.5"):
        assert restated not in body, (
            f"the marker restates {restated} instead of deriving it from FlyAnimation; a "
            "changed presentation constant would leave the receipt describing the old mapping"
        )


def test_marker_carries_the_prohibition_the_image_would_otherwise_travel_without() -> None:
    body = MARKER_METHOD.search(_writer_source())
    assert body is not None
    caveat = " ".join(body.group(1).split())
    assert "must never be reported as the fly" in caveat
    assert "steering hard" in caveat


def test_declared_presentation_constants_still_match_the_adr_prose() -> None:
    """The marker follows the constants; the ADR's ~28.6x and ~3.5% do not."""

    animation = ANIMATION.read_text(encoding="utf-8")
    assert "public static final float YAW_TORQUE_VISUAL_REFERENCE = 0.035F;" in animation
    assert "public static final float MAX_BANK_DEGREES = 25.0F;" in animation


def test_gradle_refuses_a_receipt_without_the_marker() -> None:
    gradle = BUILD_GRADLE.read_text(encoding="utf-8")
    for marker in (
        f'\'"schema": "{SCHEMA}"\'',
        "'\"bank_is_display_scaled\": true'",
        "'\"presentation_visual_gain\": {'",
    ):
        assert marker in gradle, marker
    assert "missingMarkers" in gradle
    assert "renderer-observation receipt is missing its ADR 0016 presentation-gain " in gradle


def test_the_committed_v1_receipt_is_left_as_history() -> None:
    receipt = json.loads(HISTORICAL_RECEIPT.read_text(encoding="utf-8"))
    assert receipt["schema"] == "neurocraft.renderer_observation.v1"
    assert "presentation_visual_gain" not in receipt
    assert "bank_is_display_scaled" not in receipt


def test_adr_0016_records_that_the_marker_now_exists() -> None:
    adr = ADR.read_text(encoding="utf-8")
    assert "bank_is_display_scaled" in adr
    assert SCHEMA in adr
