"""TRM E1 task generators: deterministic streams, exact targets, fail-closed.

The frozen protocol binds these generators by byte hash; these tests pin the
properties the protocol relies on — cross-run determinism, stream independence
per role, exact target arithmetic, and refusal of divergent recurrences.
"""

from __future__ import annotations

import pytest
from neurocraft_experiments.trm_tasks import (
    TaskGenerationError,
    build_drive,
    delayed_reconstruction_targets,
    narma10_targets,
    unit_stream,
)

PROTOCOL = "trm-structure-vs-learning-v1"
BODIES = (77_185, 805_722)


def test_unit_stream_is_deterministic_and_bounded() -> None:
    first = unit_stream(PROTOCOL, "drive", seed_index=3, channel=0, length=64)
    second = unit_stream(PROTOCOL, "drive", seed_index=3, channel=0, length=64)
    assert first == second
    assert len(first) == 64
    assert all(0.0 <= value < 1.0 for value in first)


def test_unit_stream_varies_by_every_derivation_part() -> None:
    base = unit_stream(PROTOCOL, "drive", seed_index=3, channel=0, length=32)
    assert unit_stream(PROTOCOL, "drive", seed_index=4, channel=0, length=32) != base
    assert unit_stream(PROTOCOL, "drive", seed_index=3, channel=1, length=32) != base
    assert unit_stream(PROTOCOL, "target", seed_index=3, channel=0, length=32) != base
    assert unit_stream("other-protocol", "drive", seed_index=3, channel=0, length=32) != base


def test_build_drive_drives_both_bodies_every_step() -> None:
    drive, streams = build_drive(
        PROTOCOL, seed_index=5, steps=48, sensory_body_ids=BODIES, amplitude=0.5
    )
    assert len(drive) == 48
    assert set(streams) == set(BODIES)
    for step, entry in enumerate(drive):
        assert entry is not None and set(entry) == set(BODIES)
        for body in BODIES:
            assert entry[body] == pytest.approx(streams[body][step] * 0.5)
            assert 0.0 <= entry[body] < 0.5


def test_delayed_reconstruction_targets_are_exact_shifts() -> None:
    stream = tuple(float(v) / 10.0 for v in range(10))
    targets = delayed_reconstruction_targets({7: stream}, delays=(1, 3))
    assert len(targets) == 10
    # Row t carries [u(t-1), u(t-3)]; pre-history is exactly zero.
    assert targets[0] == (0.0, 0.0)
    assert targets[4] == (stream[3], stream[1])
    assert targets[9] == (stream[8], stream[6])


def test_delayed_reconstruction_orders_channels_and_delays() -> None:
    a = tuple(float(v) for v in range(6))
    b = tuple(float(v) * 2 for v in range(6))
    targets = delayed_reconstruction_targets({2: b, 1: a}, delays=(2, 1))
    # Channels ordered by body id, delays as declared: [a(t-2), a(t-1), b(t-2), b(t-1)].
    assert targets[3] == (a[1], a[2], b[1], b[2])


def test_narma10_matches_the_declared_recurrence() -> None:
    u = unit_stream(PROTOCOL, "drive", seed_index=1, channel=0, length=200)
    scaled = tuple(value * 0.5 for value in u)
    y = narma10_targets(scaled)
    assert len(y) == 200
    # Re-derive one step independently: the declared NARMA-10 recurrence.
    t = 57
    window = sum(y[t - i] for i in range(1, 11))
    expected = (
        0.3 * y[t - 1] + 0.05 * y[t - 1] * window + 1.5 * scaled[t - 10] * scaled[t - 1] + 0.1
    )
    assert y[t] == pytest.approx(expected, rel=1e-12)


def test_narma10_refuses_divergent_input() -> None:
    with pytest.raises(TaskGenerationError, match="finite"):
        narma10_targets((1e6,) * 64)


def test_generators_refuse_bad_shapes() -> None:
    with pytest.raises(TaskGenerationError, match="length"):
        unit_stream(PROTOCOL, "drive", seed_index=0, channel=0, length=0)
    with pytest.raises(TaskGenerationError, match="delay"):
        delayed_reconstruction_targets({1: (0.0, 1.0)}, delays=(0,))
    with pytest.raises(TaskGenerationError, match="amplitude"):
        build_drive(PROTOCOL, seed_index=0, steps=4, sensory_body_ids=BODIES, amplitude=0.0)
