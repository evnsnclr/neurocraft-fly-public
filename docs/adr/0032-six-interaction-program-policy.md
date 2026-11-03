# ADR 0032: six source-gated interaction responses and matched HUD evidence

Status: implemented, independently reviewed and offline measured; throughput and live acceptance pending.
Date: 2026-09-05

## Decision

Introduce the opt-in `full_mcns_interactions_program_v2` service/launcher profile
over the V4 actor and separate event inputs in ADRs 0030–0031. It has the same
full graph, dynamics, component calibration, five updates per observation and
two-tick action deadline. The default existing profiles remain unchanged.

The new pure policy requires both an observed source and a declared neural gate:

| Source | Named gate | Scripted program |
| --- | --- | --- |
| Fresh player attack-attempt | Absolute change in signed raw TTMn mean ≥0.002 | FLEE |
| Observed player approach | Existing rectified TTMn jump scalar ≥existing escape threshold | ESCAPE_JUMP |
| Observed Mob approach | Same rectified TTMn gate | BANK |
| Fresh food event, current food near | Positive MN9 change ≥0.0001 | LAND, then FEED on contact |
| Fresh brushing, outside fluid | Positive antennal motor change ≥0.0003 | GROOM |
| Observed local-light change | Absolute change in bilateral raw DNa01 mean ≥0.000001 | HOVER |

These are hand-chosen development thresholds informed by the recorded input probe.
The original attack response decreased signed TTMn activity; an absolute-change
gate does not imply positive escape-neuron activation. Actor class is supplied by
Minecraft, not recognized by the network. Attack remains a tactile proxy with no
health, injury or nociception model. All six release interactions remain required.

For change scores, freeze the mean of four completed observations before the
source event. A source is eligible for eight game ticks including the boundary.
Approach uses the rising edge of decreasing distances with the same actor category
on consecutive observations in forward/left/right rays, explicitly indices 0/2/3.
Newly seeing an actor is insufficient, and other pickable entity classes do not
qualify. Approach scores use the existing rectified scalar with a declared zero
baseline. Light averages actual left/right DNa01 means before taking its change;
the normalized steering difference would discard a shared increase or decrease.

Priority follows the table. A held episode retains its original source, baseline
and entry modulation. A fresh higher-priority source can interrupt only if its
neural gate passes on that observation; other blocked sources are discarded.
LAND cancels on near loss, fluid or its 200-tick deadline. Contact transitions
LAND to FEED for 60 ticks; FEED requires continued contact, and GROOM excludes
fluid. BANK, FLEE and light HOVER request at most 40 game ticks, urgency zero.
FLEE/BANK have explicitly nonzero speed and turn modulation; BANK's bias is +1.
Escape retains the existing declared hold and latched output formulas. A global
120-tick cooldown requires a new source afterward; delayed counter delivery cannot
revive a source whose timestamp falls inside cooldown.

These are issued intervals. Java still checks every program's body preconditions,
and missing/late commands can shorten actual execution. FLEE targets the body's
eligible nearest visible player, not necessarily the attacker, and can exit early.
The policy receives no applied-receipt feedback and does not infer completion.

Both reset methods clear neural-input and policy histories, pending sources, holds
and display triggers. In-world reset preserves advancing transport/game clocks;
new-session reset starts fresh sequences under the existing service fence.
Original, shuffled and no-input use the same policy declaration. No-input gates
all sensory assignments before the neural steps and cannot pass a zero-signal
event gate; baseline scripted CRUISE remains part of the declared policy.

## Display and delivery consequences

`neurocraft.interaction_selection_display.v2` adds six named scores, their exact
formulas, raw signed TTMn and left/right DNa01 values, source class context and the
original issued trigger. Every value is a model quantity. Current display values
are shown only with an exact applied-command match and both existing freshness
checks. An original trigger remains labeled ISSUED; if it uses the current command
sequence it must equal that command in full. Different-sequence holds permit only
the same program or the original LAND→current FEED progression.

The optional publisher and bounded 16 KiB Java parser remain outside control
authority. Invalid or unavailable display cannot change commands. Compact HUD
lines adapt to the active source and preserve all three GOAL facts. A READOUT
HOVER selected after light change remains distinct from a refused-command HOVER.

Focused pure-policy, actual small-engine wrapper, service, Java parser and client
checks plus independent source review are available. Review corrected wrong ray
indices and a same-sequence trigger-envelope mismatch; both have regressions.
The bounded offline wrapper benchmark completed 15,000 updates; the separate
dated report records all original/shuffled responses and zero no-input responses.
Its median step time was 60.056 ms. Throughput and a fresh sustained, filmed
Minecraft session must verify the integrated candidate before it replaces the
existing film. Offline issued commands do not establish actual body application.
