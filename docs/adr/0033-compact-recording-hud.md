# ADR 0033: Compact recording HUD and accessible model details

Date: 2026-09-05
Status: implemented; verification recorded separately in STATUS.md.
Authority: owner request to improve video quality, record experiments personally,
and remove the large all-caps hand-chosen/scripted/not-deciding banner.

## Decision

Use a small default overlay with the actual active network condition, controller
state and body authority. Separate the current synchronized body program from
the matched historical event/program sample and from the separately sequenced
brain sample. Show real signed model means and each sample's own read age;
unavailable or stale values are not zero. Switching, operator pause and faults
suppress the corresponding live claims.

Replace repeated architecture warnings with a quiet, neutral footer:
`Neural readouts · scripted motion` (or the actual operator/default authority).
A remappable H key opens a scrollable Model & experiment details screen. It
explains the engineered sensory mapping, hand-chosen readouts, scripted body,
quantity/freshness semantics, comparison conditions and the historical
95.66x/403.70x navigation and ~300x/35.1x/four-step escape measurements. Live
diagnostic values remain accessible there. Opening/closing this client screen
does not pause the world, send commands, reset the model or change authority.

This supersedes the permanent on-screen placement and literal all-caps wording
requirements in ADRs 0019/0021/0024/0032 and GOAL.md, only for presentation.
Their model/behavior boundaries and historical evidence remain unchanged.
During a walkthrough, explain the model once in the details view or narration;
the architecture does not need to occupy every gameplay frame.

Provide `make live-demo-recording`: a fresh showcase world, three comparison
lanes, normal 20 TPS target, 1280x720 logical window and 60 rendered-FPS ceiling.
Record with OBS rather than the diagnostic PNG sampler. Recording resolution,
achieved frame rate and observed controller performance must be checked in the
operator's rehearsal; launch settings are not measurements. The manual guide
retains native interaction prerequisites and distinguishes prepared scenes,
operator movement tours and neural comparison experiments.

No graph, encoder, readout, model, body program, protocol or production dependency
changes. Historical recordings and receipts remain untouched.
