#!/usr/bin/env python3
"""Check, from a committed trace alone, that the connectome was not driving.

A body program (ADR 0020) may own a tick only while the neural service is paused. A
paused service does not stop talking: it returns a *neutral* action for every
observation instead of stepping the controller
(``services/neural-engine/src/neurocraft_service/server.py:543-553``), and it stamps
that action with a digest over a fixed recipe
(``server.py:201-231``)::

    sha256(canonical_json({
        "kind": "service_neutral_v1",
        "reason": <control state, lowercased>,
        "observation_seq": ...,
        "game_tick": ...,
        "neural_cursor": ...,
    }))

So "the connectome was not driving during this footage" is **checkable** rather than
asserted: recompute the digest with ``reason = "paused"`` and compare it against the
trace. It is not a plaintext label — a forger would have to know the recipe — and it
is bound to the same action envelope the world applied. The neural cursor not moving
across the window (``neural_step_start == neural_step_end``) is the second, independent
signature of the same fact.

**What this does not establish.** A matching digest proves the *service* reported
itself paused for that action. It is not evidence that a body program ran, that the
footage shows anything in particular, or that the session was healthy. It is one
negative fact, checked exactly.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from neurocraft_protocol import (
    Action,
    CausalRecord,
    canonical_json_bytes,
    parse_json_frame,
    verify_trace,
)

#: The digest recipe's frozen discriminator, mirrored from ``server.py``.
NEUTRAL_DIGEST_KIND: Final = "service_neutral_v1"

#: Control states for which the service emits a neutral action instead of stepping.
NEUTRAL_REASONS: Final[tuple[str, ...]] = ("paused", "replaying", "faulted")

#: The reason a scripted body-program window must carry.
PAUSED_REASON: Final = "paused"

#: Classification for an action the service produced by actually stepping the controller.
DRIVING: Final = "driving"

#: Classification for a neutral action whose digest matches no declared neutral reason.
UNEXPLAINED: Final = "unexplained-neutral"


@dataclass(frozen=True)
class TickClassification:
    """One action's verdict, keyed by the tick the authority was asked to apply it on."""

    game_tick: int
    command_seq: int
    classification: str
    neural_cursor_advanced: bool


@dataclass(frozen=True)
class PausedWindowVerification:
    """Outcome of checking every action in a trace against the neutral-digest recipe."""

    ticks: tuple[TickClassification, ...]
    paused_ticks: int
    driving_ticks: int
    unexplained_ticks: int
    longest_paused_window: tuple[int, int] | None

    @property
    def fully_paused(self) -> bool:
        """True when every action checked was proven paused by its own digest."""

        return bool(self.ticks) and self.paused_ticks == len(self.ticks)


def neutral_digest(reason: str, *, observation_seq: int, game_tick: int, neural_cursor: int) -> str:
    """Recompute the service's neutral-action state digest for one action."""

    return hashlib.sha256(
        canonical_json_bytes(
            {
                "kind": NEUTRAL_DIGEST_KIND,
                "reason": reason,
                "observation_seq": observation_seq,
                "game_tick": game_tick,
                "neural_cursor": neural_cursor,
            }
        )
    ).hexdigest()


def classify_action(action: Action) -> TickClassification:
    """Decide, from the action alone, whether the connectome was driving on its tick."""

    advanced = action.neural_step_end != action.neural_step_start
    if not action.neutral:
        return TickClassification(action.target_tick, action.command_seq, DRIVING, advanced)
    classification = UNEXPLAINED
    for reason in NEUTRAL_REASONS:
        candidate = neutral_digest(
            reason,
            observation_seq=action.observation_seq,
            game_tick=action.observation_tick,
            neural_cursor=action.neural_step_start,
        )
        if candidate == action.state_digest:
            classification = reason
            break
    return TickClassification(action.target_tick, action.command_seq, classification, advanced)


def verify_actions(actions: Sequence[Action]) -> PausedWindowVerification:
    """Classify a sequence of issued actions and summarize the paused windows."""

    ticks = tuple(classify_action(action) for action in actions)
    paused = sum(1 for tick in ticks if tick.classification == PAUSED_REASON)
    driving = sum(1 for tick in ticks if tick.classification == DRIVING)
    unexplained = sum(1 for tick in ticks if tick.classification == UNEXPLAINED)
    return PausedWindowVerification(
        ticks=ticks,
        paused_ticks=paused,
        driving_ticks=driving,
        unexplained_ticks=unexplained,
        longest_paused_window=_longest_paused_window(ticks),
    )


def read_actions(path: Path) -> tuple[Action, ...]:
    """Verify the trace's own integrity, then return every issued action it carries."""

    verify_trace(path)
    return tuple(record.issued_action for record in _iter_records(path))


def verify_trace_file(path: Path) -> PausedWindowVerification:
    """Full check of a committed trace file."""

    return verify_actions(read_actions(path))


def _iter_records(path: Path) -> Iterator[CausalRecord]:
    with path.open("rb") as stream:
        for raw in stream:
            envelope = parse_json_frame(raw.rstrip(b"\n"))
            if envelope.get("kind") == "record":
                yield CausalRecord.from_dict(envelope["payload"])


def _longest_paused_window(
    ticks: Sequence[TickClassification],
) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    best_length = 0
    start: int | None = None
    length = 0
    for tick in ticks:
        if tick.classification == PAUSED_REASON and not tick.neural_cursor_advanced:
            start = tick.game_tick if start is None else start
            length += 1
            if length > best_length:
                best_length = length
                best = (start, tick.game_tick)
        else:
            start = None
            length = 0
    return best


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute the service's neutral-action digest over a committed trace and "
            "report, per tick, whether the connectome was driving. A scripted body-program "
            "window must be paused on every tick it claims."
        )
    )
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--require-paused",
        action="store_true",
        help="exit non-zero unless every action in the trace is proven paused",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outcome = verify_trace_file(args.trace)
    window = outcome.longest_paused_window
    print(
        f"checked {len(outcome.ticks)} actions: paused={outcome.paused_ticks} "
        f"driving={outcome.driving_ticks} unexplained={outcome.unexplained_ticks}"
    )
    if window is None:
        print("no paused window: the connectome was stepping for every action checked")
    else:
        print(
            f"longest paused window: target ticks {window[0]}..{window[1]}, "
            "neural cursor did not advance"
        )
    if args.require_paused and not outcome.fully_paused:
        print(
            "REFUSED: not every action is proven paused, so a scripted body-program claim "
            "over this trace is not supported by the trace"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
