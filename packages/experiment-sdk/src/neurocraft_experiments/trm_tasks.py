"""Deterministic task generators for the frozen TRM E1 protocol (ADR 0008).

A new module on purpose, like :mod:`nulls` and :mod:`reservoir`: the pilot pins
stay byte-identical, and the TRM freeze binds this file's exact bytes instead.

Every stream is derived from SHA-256 over the protocol id, a role label, a seed
index, a channel, and a counter — the same derivation discipline the null
generators use — so no two roles, channels, or seed indices can ever share a
stream, on any platform, and no ``random`` state is involved anywhere.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping, Sequence
from hashlib import sha256

MAX_STREAM_LENGTH = 1_000_000

#: The declared NARMA-10 recurrence constants; standard reservoir benchmark.
NARMA_ALPHA = 0.3
NARMA_BETA = 0.05
NARMA_GAMMA = 1.5
NARMA_DELTA = 0.1
NARMA_ORDER = 10


class TaskGenerationError(ValueError):
    """Raised when a task stream or target request is invalid or diverges."""


def unit_stream(
    protocol_id: str, role: str, *, seed_index: int, channel: int, length: int
) -> tuple[float, ...]:
    """A deterministic stream of floats in ``[0, 1)`` from SHA-256 counters."""

    if not isinstance(protocol_id, str) or not protocol_id:
        raise TaskGenerationError("streams require a non-empty protocol id")
    if not isinstance(role, str) or not role:
        raise TaskGenerationError("streams require a non-empty role label")
    for name, value in (("seed_index", seed_index), ("channel", channel)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TaskGenerationError(f"{name} must be a non-negative integer")
    if (
        isinstance(length, bool)
        or not isinstance(length, int)
        or not 1 <= length <= MAX_STREAM_LENGTH
    ):
        raise TaskGenerationError(f"stream length must be in [1, {MAX_STREAM_LENGTH}]")
    prefix = f"{protocol_id}|{role}|{seed_index}|{channel}".encode()
    values: list[float] = []
    for counter in range(length):
        digest = sha256(prefix + b"|" + struct.pack(">Q", counter)).digest()
        values.append(int.from_bytes(digest[:8], "big") / 2**64)
    return tuple(values)


def build_drive(
    protocol_id: str,
    *,
    seed_index: int,
    steps: int,
    sensory_body_ids: Sequence[int],
    amplitude: float,
) -> tuple[tuple[Mapping[int, float], ...], dict[int, tuple[float, ...]]]:
    """Continuous two-channel drive plus the raw unit streams that made it.

    Returns ``(drive, streams)``: one stimulation mapping per step covering
    every sensory body, and the per-body unit streams so target generators can
    consume exactly the inputs the engine saw.
    """

    bodies = tuple(sensory_body_ids)
    if len(bodies) != len(set(bodies)) or not bodies:
        raise TaskGenerationError("sensory bodies must be non-empty and unique")
    if (
        isinstance(amplitude, bool)
        or not isinstance(amplitude, float)
        or not math.isfinite(amplitude)
        or amplitude <= 0.0
    ):
        raise TaskGenerationError("drive amplitude must be a finite positive float")
    ordered = tuple(sorted(bodies))
    streams = {
        body: unit_stream(
            protocol_id, "drive", seed_index=seed_index, channel=channel, length=steps
        )
        for channel, body in enumerate(ordered)
    }
    drive = tuple(
        {body: streams[body][step] * amplitude for body in ordered} for step in range(steps)
    )
    return drive, streams


def delayed_reconstruction_targets(
    streams_by_body: Mapping[int, Sequence[float]],
    *,
    delays: Sequence[int],
) -> tuple[tuple[float, ...], ...]:
    """Row ``t`` holds ``u_c(t - k)`` for every channel ``c`` and delay ``k``.

    Channels are ordered by body id and delays exactly as declared; history
    before ``t = 0`` is exactly zero, and the protocol's washout must cover the
    largest delay so those zero rows never reach a fit.
    """

    if not streams_by_body:
        raise TaskGenerationError("delayed reconstruction requires at least one stream")
    delay_values = tuple(delays)
    if not delay_values or any(
        isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in delay_values
    ):
        raise TaskGenerationError("every delay must be an integer >= 1")
    ordered = sorted(streams_by_body)
    lengths = {len(streams_by_body[body]) for body in ordered}
    if len(lengths) != 1:
        raise TaskGenerationError("all streams must share one length")
    (length,) = lengths
    rows: list[tuple[float, ...]] = []
    for step in range(length):
        row: list[float] = []
        for body in ordered:
            stream = streams_by_body[body]
            for delay in delay_values:
                row.append(float(stream[step - delay]) if step >= delay else 0.0)
        rows.append(tuple(row))
    return tuple(rows)


def narma10_targets(scaled_input: Sequence[float]) -> tuple[float, ...]:
    """The declared NARMA-10 recurrence over one scaled input channel.

    ``y(t) = 0.3 y(t-1) + 0.05 y(t-1) sum(y(t-1..t-10)) + 1.5 u(t-10) u(t-1) + 0.1``
    with zero history. Divergence fails closed: the recurrence is only stable
    for inputs roughly in ``[0, 0.5]``, and a non-finite or runaway value is a
    protocol violation, never something to clip quietly.
    """

    inputs = tuple(float(value) for value in scaled_input)
    if not inputs:
        raise TaskGenerationError("narma10 requires a non-empty input")
    outputs: list[float] = []
    for step in range(len(inputs)):
        previous = outputs[step - 1] if step >= 1 else 0.0
        window = sum(
            outputs[step - lag] if step >= lag else 0.0 for lag in range(1, NARMA_ORDER + 1)
        )
        u_old = inputs[step - NARMA_ORDER] if step >= NARMA_ORDER else 0.0
        u_prev = inputs[step - 1] if step >= 1 else 0.0
        value = (
            NARMA_ALPHA * previous
            + NARMA_BETA * previous * window
            + NARMA_GAMMA * u_old * u_prev
            + NARMA_DELTA
        )
        if not math.isfinite(value) or abs(value) > 1e6:
            raise TaskGenerationError(
                f"narma10 output is not finite/bounded at step {step}: the input "
                "range violates the recurrence's declared stability envelope"
            )
        outputs.append(value)
    return tuple(outputs)
