"""The declared action budget: how far ahead of its observation a command targets."""

from typing import Final

ACTION_TARGET_OFFSET_TICKS: Final = 2
"""Ticks between the observation a command answers and the tick it is applied on.

Every command frame — an ``action`` or a ``program_action`` — declares
``target_tick = observation_tick + ACTION_TARGET_OFFSET_TICKS`` and
``valid_through_tick = target_tick``, which is also the protocol's ceiling
(``valid_through_tick <= observation_tick + ACTION_TARGET_OFFSET_TICKS``). The
authority applies a frame **only** on its ``target_tick``, so this constant is the
whole round-trip budget: encode, send, compute, return, decode, and be sitting in
the authority's window when that tick arrives.

**Why two and not one.** Under a one-tick budget (ADR 0018) the whole round trip
had to finish inside one game tick, 50 ms. Offline on the operator's machine the raw
``FullBrainController.step`` alone measures p50 38.9 / p90 41.5 / p99 56.7 / max
62.9 ms over 300 steps, and the program wrapper adds 0.13 ms — so roughly half of
all frames missed by construction, and the first live sessions of
``make live-demo-program`` died within about two seconds of engaging, on
``DEADLINE_MISSED`` after 7- and 13-tick bursts of late frames (mod-measured action
latency 43-75 ms over the first eight frames). Two ticks buy 100 ms for a step that
worst-cases near 63 ms, at the cost of exactly one tick (50 ms) of control lag.

**It is an envelope rule, not a binding.** No identity moves with it: it is not
hashed into ``RunIdentity``, no calibration or readout binding depends on it, and a
frame's causal chain — which observation it answers, which neural interval produced
it — is unchanged. It only says which tick the answer is due.

The frame model stays permissive on purpose: ``Action`` and ``ProgramAction`` accept
any ``observation_tick <= target_tick <= valid_through_tick <=
observation_tick + ACTION_TARGET_OFFSET_TICKS``, so historical frames recorded under
the one-tick budget still parse. The offset itself is enforced where frames are
produced (``_validate_generated_action`` in the service) and where they are admitted
(``SessionValidator.accept_action`` here, ``NeuralBridge`` on the mod side).
"""

__all__ = ["ACTION_TARGET_OFFSET_TICKS"]
