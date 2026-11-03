# ADR 0031: separate brush and attack-attempt inputs

Status: implemented, independently reviewed and diagnostically measured; live responses pending.
Date: 2026-09-05

## Context

The recorded v3 showcase mapped brushing and player attack attempts to the same
65 grooming mechanosensory inputs and the same GROOM response. Six distinct
repeatable interactions remain a release requirement. Sensor revision v4 now
supplies honest actor context (ADR 0030); an input revision also needs to preserve
the difference between these two native event counters.

The verified retained annotation census contains 2,206 mechanosensory bristle
members, also used by the existing collision/touch component. Of these, 2,130 have
the mechanosensory_tactile class and 76 the unknown_sensory class. This pool is a
declared tactile proxy. No retained class/subclass/type name matching nociception,
pain or injury was found in that census; this is not a biological absence claim.

## Decision

An explicit `event_inputs_version=2` opt-in creates the separate
`neurocraft.full_brain.interactions.configuration.v2` configuration and
`neurocraft.interactions.encoder.v2` binding. It requires SensorFrameV4. The default
remains version 1; no existing launcher or service profile enables the new input.

Keep food's 163 gustatory inputs and brush's 65 grooming mechanosensory inputs.
A fresh player attack-attempt counter instead supplies one amplitude-1 pulse per
selected bristle member, during the observation's existing five neural updates.
It no longer supplies the grooming pulse. Simultaneous fresh events add once to
each of their disjoint source pools; jumps in cumulative counters do not multiply
the pulse. First observations absorb existing history, the eight-game-tick age
limit remains, and reset clears independent freshness state. Generic contact and
the new attack pulse may add at the same bristle cells; the overlap is declared.

Bind the bristle body IDs, dense indices, selector and expected count, the full
embodied population identity, source annotation/graph identities, observation
revision and pulse formula. Refuse duplicate, out-of-range or overlapping event
population memberships. Fixture tests may explicitly declare smaller counts under
a different hash. The full builder requires the actual 2,206-member binding.

The controller verifies that this binding names its actual embodied populations.
New configuration and new encoder must pair exactly. V4 identity and field-quality
checks precede counter or neural mutation. The final no-input gate executes after
every existing and new contribution. Requested attack stimulation remains visible
separately from delivered control input.

The graph, modeled signs/dynamics, loom-v2 absolute-light arithmetic and motor
decoder are unchanged. Existing calibration is explicitly reused as a component;
it is not validation of the new composite. Only the new input declaration and
observation identity rotate. Actor categories alone do not alter neural input;
any later category-conditioned program selection must be labeled as engineering.

## Verification and remaining delivery

Nine focused component/controller tests passed. They compare actual delivered
vectors over 35 neural steps, including simultaneous collision and attack, and
check independent counter ages, reset, configuration/population mismatch refusal,
unchanged component identities and full no-input activity. Seventeen diagnostic
tests passed on small real engines, including direct-step state reproduction and
player/mob histories that differ only in actor metadata. Lint and type checks
passed, and independent source review covered both the component and diagnostic.

`scripts/probe_interaction_inputs.py` is a bounded full-network measurement of
prescribed brush, attack, contact, food, actor-approach and light histories in
original, one fixed shuffled-weight and no-input lanes. It retains full observations,
raw named readouts, source inventories, kernel identity and actual state/input
digests. Full no-input vector and activity digests must be zero. Its paired reference
differences are offline diagnostics, not live controller observations.

The completed measurement executed 1,080 observations and 5,400 neural updates.
Original and shuffled wiring responded differently; all no-input delivered vectors
and full activity states were exactly zero. Player/mob cases had identical neural
states when only the supplied actor category differed. The attack proxy decreased
original TTMn activity and increased it with shuffled weights, so the existing
positive escape readout cannot simply be reused as an attack detector.
`docs/interaction-input-probe-2026-09-05.md` records the numerical result and limits.

This change selects or applies no new body program. Policy thresholds, modulation,
service/HUD integration, sustained live verification and a new filmed session remain
required. Attack remains a player attack attempt with no health/damage model. The
original six interaction requirements and all three visible GOAL facts remain.
